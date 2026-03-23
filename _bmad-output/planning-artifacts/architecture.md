---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
sectionsAdded: [14, 15, 16, 17, 18, 19, 20]
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-03-14'
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/project-context.md'
  - 'docs/architecture.md'
  - 'docs/project-overview.md'
  - 'docs/agentic-setup.md'
  - 'docs/agentic-needs-analysis.md'
workflowType: 'architecture'
project_name: 'QUANTMINDX'
user_name: 'Mubarak'
date: '2026-03-13'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

---

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**

79 FRs across 9 capability domains, supporting 52 user journeys:

| Domain | FRs | Key Architectural Demand |
|--------|-----|--------------------------|
| Live Trading & Execution | FR1–FR9 | Real-time WebSocket push, MT5 ZMQ bridge, sub-3s kill switch path |
| Autonomous Agent System | FR10–FR22 | Anthropic Agent SDK, 3-tier hierarchy, Department Mail bus, Skill Registry |
| Alpha Forge — Strategy Factory | FR23–FR31 | Stateful distributed pipeline across Contabo + Cloudzy, human gate at live promotion |
| Risk Management & Compliance | FR32–FR41 | Physics sensors (Ising + HMM + Lyapunov) → Commander pipeline, Islamic force-close, prop firm rule sets |
| Knowledge & Research | FR42–FR50 | Semantic search (ChromaDB/Qdrant), video ingest, provenance metadata, news feed |
| Portfolio & Multi-Broker | FR51–FR58 | Routing matrix, multi-account Kelly multipliers, account-level risk isolation |
| Monitoring, Audit & Notifications | FR59–FR65 | Immutable audit log (3yr), natural language query via Copilot, OS tray notifications |
| System Management & Infrastructure | FR66–FR73 | Runtime provider swap, 3-node sequential update with rollback, machine-portable ITT |
| Traceability Gap Closures | FR74–FR79 | A/B live testing, strategy rollback, cross-strategy patch propagation, EA deployment pipeline |

**Non-Functional Requirements:**

| NFR Category | Critical Constraints |
|---|---|
| Performance | ≤3s live P&L/regime lag; Copilot first token ≤5s; canvas transitions ≤200ms; backtest matrix ≤4h |
| Security | SSH key-only on both servers; IP firewall on Cloudzy; all secrets in `.env` only; proprietary IP never in public repos or raw AI prompts |
| Reliability | Node-independent fault isolation; MT5 ZMQ reconnect ≤10s; WebSocket auto-reconnect; Cloudzy trading continues if Contabo unreachable |
| Data Integrity | Trade records to SQLite before acknowledgment; audit logs immutable; tiered tick storage (Hot/Warm/Cold); cold storage integrity verification |
| Integration Reliability | Timeout + retry on all AI provider APIs; ZMQ disconnect detected ≤10s; rate limit errors queue rather than crash |
| Maintainability | No new LangGraph/LangChain; department system is sole agent architecture; 500-line file limit; risk-critical components test-covered before modification |

**Scale & Complexity:**

- Primary domain: Distributed systems + Real-time data + Multi-agent AI + Desktop frontend
- Complexity level: **Enterprise-level**
- Estimated architectural components: ~15 major bounded contexts
- Brownfield context: ~60–70% of core backend exists; needs reorganization + full frontend redesign

### Rebuild Scope — What Stays, What Gets Rebuilt From Scratch

This system is being significantly restructured. The source of truth for what to build is: PRD user journeys + UX design specification + this architecture.md. Do NOT build on top of old agentic or UI code — start fresh for those layers.

**Stays untouched (do not modify):**
- `PhysicsAwareKellyEngine` + full risk pipeline (Ising, HMM, Lyapunov, Eigenvalue, Monte Carlo)
- `RoutingMatrix` + broker registry (extend only as per §7.2–7.3)
- MT5 ZMQ bridge
- Tick data pipeline (collection, tiering, cold storage)
- **Backtest engine** — `src/backtesting/` (`core_engine.py`, `mt5_engine.py`, `mode_runner.py`, `walk_forward.py`, `monte_carlo.py`, `pbo_calculator.py`, `full_backtest_pipeline.py`, `multi_asset_engine.py`). All 6 modes confirmed working: VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C. Alpha Forge workflows call these engines — they do not rebuild them. See §20 for how the workflows use them.

**Stays but extends (targeted additions only — see referenced sections):**
- `src/memory/graph/` — graph memory 80-90% done; add columns + ReflectionExecutor + embeddings only (§6.1)
- `src/agents/memory/session_checkpoint_service.py` — wire to graph memory commit flow
- `flows/` — some existing assembled flows adapted to new structure
- MCP server configs — retain, update endpoints as APIs reorganize
- Existing Prefect flows that are pure data pipelines (HMM retrain, tick sync, etc.)

**Rebuilt from scratch (do not extend old code):**
- `src/agents/` — entire department system rebuilt per this architecture
- `quantmind-ide/` — entire frontend rebuilt from zero per UX design specification (Svelte 5 runes, new canvas structure)
- `shared_assets/skills/` — hierarchical structure, new files (§9, §15)
- `flows/` components + directives library — new structure per §5.1–5.2
- Department system prompts — new templates (§4.3, §15)

**Legacy code (exists but will not be touched in current phase):**
- Old agent files in `src/agents/` not yet refactored — treat as read-only reference
- Old Svelte 4 components — do not extend; rebuild target in separate sprint

### Technical Constraints & Dependencies

- **Anthropic Agent SDK**: sole canonical agent runtime — no LangGraph, no LangChain in new code
- **Svelte 5 runes** + static adapter (no SSR) — 65+ components to migrate from Svelte 4
- **ZMQ for MT5 tick feed**: MetaTrader 5 on Windows/Wine only; bridge must run on Cloudzy (Windows or Wine-compatible Linux)
- **Dukascopy for historical backtests**: proprietary data accumulates over time to reduce dependency
- **Islamic compliance hard constraint**: force-close 21:45 GMT enforced at Commander level AND in every EA template
- **Prop firm rules are configurable, not hardcoded**: Prop Firm Registry pattern — any firm can be added at runtime
- **No authentication in Phase 1**: firewall-based trust; JWT (admin/viewer/client) in Phase 2
- **Private IP only for strategy IP**: EA source code, Alpha Forge output, knowledge base never exposed in public repositories

### Cross-Cutting Concerns Identified

| Concern | Affected Components |
|---------|---------------------|
| **Audit trail** (immutable, 3-year, SQL-first) | DB schema, every API endpoint, agent task dispatcher, Commander pipeline |
| **Node independence** (Cloudzy must trade without Contabo) | Service startup order, inter-node communication pattern, fallback state caching |
| **Provider abstraction** (runtime AI model swap) | Agent SDK integration, ProvidersPanel settings, tier-to-model mapping, fallback on outage |
| **Knowledge provenance chain** | All knowledge writes, Alpha Forge lineage tracking, strategy version history |
| **Islamic compliance** | Commander force-close scheduler, EA tag input requirements, broker registry validation |
| **Concurrency** (parallel subagents, parallel workflows) | Async patterns, SQLite WAL mode, Department Mail queue, Agent SDK concurrent calls |
| **Security** (secrets, IP restriction, no IP in prompts) | Both FastAPI backends, Tauri ↔ backend communication, agent context construction |

---

## Starter Template Evaluation

### Primary Technology Domain

Brownfield upgrade — distributed desktop app (Tauri 2 + SvelteKit 5) + two-node FastAPI backend. No greenfield scaffolding; three targeted migration/split decisions.

### Verified Current Versions (March 2026)

| Technology | Current Stable | Source |
|-----------|---------------|--------|
| Svelte | 5.53.x | [svelte.dev/blog/svelte-5-is-alive](https://svelte.dev/blog/svelte-5-is-alive) |
| SvelteKit | 2.51.x | [github.com/sveltejs/kit/releases](https://github.com/sveltejs/kit/releases) |
| Tauri | 2.10.3 | [v2.tauri.app/release](https://v2.tauri.app/release/) |
| Anthropic Agent SDK (Python) | Active / latest | [github.com/anthropics/claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python) |
| Python | 3.12 (pinned) | Project constraint |
| FastAPI | latest | Project constraint |

### Three Scaffolding Decisions

#### Decision 1: Svelte 4 → Svelte 5 Migration — In-place via `sv migrate`

**Chosen approach:** In-place migration using the official Svelte CLI migration codemod.

**Rationale:** Existing SvelteKit routing, Tauri config, Svelte stores, and API layer are working foundations — no reason to discard the scaffold. The `sv migrate svelte-5` codemod handles `$:` → `$derived`/`$effect`, `export let` → `$props()`, and other runes transformations automatically. Canvas/department redesign (Frosted Terminal, canvas navigation model) proceeds component-by-component alongside migration.

**Initialization command:**
```bash
cd quantmind-ide
npx sv migrate svelte-5
```

**Architectural decisions this provides:**
- Svelte 5 runes syntax across all components
- `$app/state` replaces `$app/stores` (SvelteKit 2.12+)
- Static adapter retained (`@sveltejs/adapter-static`, `strict: true`) — no SSR
- Existing TypeScript strict mode, Vite 5, Tailwind retained

#### Decision 2: FastAPI Monolith → Two-Node Split — Config-based `NODE_ROLE` (Phase 1)

**Chosen approach:** Single codebase, environment variable `NODE_ROLE=contabo|cloudzy|local` controls which router groups register at startup.

**Rationale:** Least disruptive for a solo developer on a brownfield codebase with 50+ router modules. Avoids a full import-path refactor on day one. Cleanly separable to two distinct packages in Phase 2 once node responsibilities are stable.

**Pattern:**
```python
# src/api/server.py
NODE_ROLE = os.getenv("NODE_ROLE", "contabo")  # contabo | cloudzy | local

if NODE_ROLE in ("contabo", "local"):
    app.include_router(agents_router, prefix="/api")
    app.include_router(floor_manager_router, prefix="/api")
    app.include_router(knowledge_router, prefix="/api")
    app.include_router(research_router, prefix="/api")
    app.include_router(settings_router, prefix="/api")
    # ... all non-trading routers

if NODE_ROLE in ("cloudzy", "local"):
    app.include_router(strategy_router, prefix="/api")
    app.include_router(kill_switch_router, prefix="/api")
    app.include_router(mt5_bridge_router, prefix="/api")
    app.include_router(trading_router, prefix="/api")
    # ... all trading-execution routers
```

**Systemd service files** set the `NODE_ROLE` env var per node. Same `git pull` deploys to both.

#### Decision 3: Agent SDK Migration — No New Scaffold, Extend Existing Department System

**Chosen approach:** Remove LangGraph/LangChain from `requirements.txt`, migrate `src/agents/core/base_agent.py` and `src/agents/registry.py` to the Anthropic Agent SDK. Wire the existing `src/agents/departments/` structure to `claude-agent-sdk-python`.

**Rationale:** The Department System (`FloorManager` → Department Heads → Sub-agents) is already the canonical pattern. No new project scaffold needed — the migration is about replacing the runtime (LangGraph → Claude Agent SDK) while preserving the department structure, mail bus, tool registry, and memory facade.

**No initialization command** — migration is incremental within existing `src/agents/`.

### Note on Project Initialization

No `create-*` command is needed. All three decisions are migrations within the existing monorepo. The first implementation stories will be:
1. Run `npx sv migrate svelte-5` and resolve flagged components
2. Add `NODE_ROLE` env var handling to `src/api/server.py`
3. Remove `langchain`, `langgraph`, `langchain_openai` from `requirements.txt` and fix resulting import errors

---

## Core Architectural Decisions

_All decisions below were reached through collaborative discovery. Brownfield note: ~60–70% of the backend exists — decisions specify what to build, extend, or replace rather than build from scratch._

---

### 1. Data Architecture

#### 1.1 — Pipeline State Storage

**Decision:** Dedicated `workflows.db` (Prefect's SQLite backend) for ALL workflow types — Alpha Forge, knowledge sync, HMM retraining, drawdown reviews, and any future Prefect-managed flow.

**Rationale:** A single durable state store across all workflow types simplifies observability, restart resilience, and human approval gate tracking. Prefect's built-in SQLite backend provides this without additional infrastructure.

**Scope:** Contabo only. `workflows.db` does not live on Cloudzy.

---

#### 1.2 — Async Queue & Department Mail

**Decision:** [Redis Streams](https://redis.io/docs/latest/develop/data-types/streams/) replace the existing SQLite-backed `DepartmentMailService`.

**Rationale:** The current SQLite mail bus (`department_mail.db`) was sufficient for prototype scale but is not designed for proper async queue semantics, workflow event publication, or concurrent consumer groups. Redis Streams provide consumer groups, acknowledgment, replay, and dead-letter queue support.

**Pattern:**
```
dept:{dept}:{workflow_id}:queue    → tasks assigned to dept in this workflow
mail:dept:{dept}:{workflow_id}     → messages to dept in this workflow
mail:broadcast:{workflow_id}       → all-dept broadcast for this workflow
workflow:{wf_id}:events            → Prefect-level workflow event stream
```

**Namespace isolation:** Every workflow run has its own key namespace. Departments working on multiple concurrent workflows maintain separate queues — no cross-workflow contamination.

**Atomic task checkout ([Paperclip pattern](https://github.com/paperclipai/paperclip)):** Department Heads claim tasks via Redis `SETNX task:{task_id}:owner {dept_id}`. If already claimed, skip to next unclaimed task. Prevents double-work.

**Docker deployment:** Redis runs in Docker on Contabo alongside Prefect.

---

#### 1.3 — Knowledge & Vector Search Stack

**Decision:** Four-layer knowledge stack — no single system handles all cases.

| Layer | Technology | Purpose |
|---|---|---|
| Full-text search | PageIndex (existing Docker service) | Scraped articles, news, web content |
| Semantic search | [ChromaDB](https://docs.trychroma.com) + [sentence-transformers](https://sbert.net) | Strategy docs, knowledge base, embeddings (free, local, no API cost) |
| Cross-session memory | Graph Memory (`docs/plans/2026-03-09-graph-memory-system.md`) | Agent memory across sessions — no embeddings, uses importance scores + tags |
| Graph upgrade | Neo4j (Phase 2) | Replace DuckDB-backed graph once scale requires it |

**PREREQUISITE:** The Graph Memory system (`docs/plans/2026-03-09-graph-memory-system.md`) must be fully implemented before the Canvas Context System is built. The canvas context layer is a consumer of graph memory, not its own storage.

**Context engineering principle ([Anthropic, March 2026](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)):** Minimum high-signal tokens. Agents maintain lightweight identifiers (file paths, IDs) and load content just-in-time via tools. No pre-loading of large documents (e.g., 32MB MQL5 PDFs) into context. Context rot increases with window size — load only what is needed for the current task.

**MQL5 PDFs** (`mql5.pdf` 32.5MB, `mql5book.pdf` 14.3MB): Indexed in PageIndex. Agents query specific sections JIT via tool call. Never pre-loaded into context.

---

#### 1.4 — Tick Data Pipeline

**Decision:** Three-tier tiered storage (existing `TieredStorageRouter` in codebase, confirmed implemented).

| Tier | Storage | Retention | Notes |
|---|---|---|---|
| HOT | PostgreSQL `tick_cache` | < 1 hour | Real-time, Cloudzy-local |
| WARM | DuckDB `market_data` | 7 days | Adjusted from 30-day default in code |
| COLD | Parquet files (symbol/year/month/day) | Indefinite | Contabo cold storage |

**Cloudzy → Contabo sync:** Nightly rsync cron. Not continuous streaming. 3-day backup cadence for essentials (tick data, DB, strategy files, config).

**Gap:** rsync cron is not yet implemented. Everything else exists.

---

#### 1.5 — HMM Training Pipeline

**Decision:** Two-phase training with permanent shadow mode during initial deployment.

**Phase 1 (now, until sufficient MT5 data collected):**
- Training data: [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) historical data
- Process: Distill → variant formats per strategy type → train base HMM model
- Retrain: Weekly cron (Saturday 02:00 UTC — already live on Contabo via `scripts/schedule_hmm_training.py`)
- Shadow mode: Strategy Router ignores HMM output for 3+ months. HMM runs in ISING_ONLY → HMM_SHADOW stage.

**Phase 2 (once MT5 collected data threshold met):**
- Training data: MT5 live collected tick data (multi-pair, multi-timeframe)
- Automatic transition once volume threshold crossed (configurable in `hmm_config.json`)
- Same distillation pipeline, different source

**Manual trigger:** `/api/hmm/train` (already exists). `trigger_mode: scheduled|manual` parameter.

**A/B testing via deployment stages (replaces separate A/B system):** ISING_ONLY → HMM_SHADOW → HMM_HYBRID → HMM_ONLY. Log-based comparison engine reads shadow output vs actual. No live capital splitting for A/B.

---

### 2. API & Communication

#### 2.1 — Inter-Node Protocol (Contabo ↔ Cloudzy)

**Decision:** HTTP REST for command dispatch between nodes.

**Rationale:** Contabo-to-Cloudzy latency is ~3–8ms (European VPS pair). HTTP REST is adequate for workflow command dispatch (trigger trade, kill switch, deploy EA). The latency constraint is on the trading data path (Cloudzy-local ZMQ), not on the orchestration path.

**Cloudzy node independence:** Cloudzy must be able to trade without Contabo reachable. All live trading decisions execute locally on Cloudzy. Contabo sends commands, but Cloudzy's execution does not block on Contabo responses.

---

#### 2.2 — Agent Streaming to UI

**Decision:** Dual streaming — two separate channels for two different data types.

| Stream | Protocol | Source | Purpose |
|---|---|---|---|
| Agent events | SSE (`GET /api/agents/stream`) | Contabo | Department activity, workflow status, task updates, Copilot responses |
| Trading data | WebSocket | Cloudzy | Live P&L, tick feed, position updates, kill switch events |

**Frontend multi-backend routing:** `apiFetch` uses endpoint-prefix routing. Both Contabo and Cloudzy URLs configurable in Settings → Connections panel. No hardcoded URLs in components.

---

#### 2.3 — MCP Server Stack

**Decision:** Minimal MCP server set. Only what is actively needed.

| MCP Server | Purpose |
|---|---|
| [`context7`](https://github.com/upstash/context7) | Library documentation (MQL5, Python, Svelte, etc.) |
| [`sequential-thinking`](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking) | Complex multi-step reasoning for agents |
| `web-fetch` | Web content retrieval |
| RAG/CAG MCP (build) | Internal knowledge base query tool — wraps ChromaDB + PageIndex + Graph Memory |
| Internal MCP servers (existing) | MT5 compiler, backtest execution, knowledge base — in `mcp-servers/` |

**Not included:** OpenRouter (never for Agent SDK). No external agent coordination MCPs.

---

### 3. Infrastructure & Deployment

#### 3.1 — Node Responsibilities (Hard Rule)

**Cloudzy — ONLY these components:**
- ZMQ tick data stream reception
- MT5 Bridge (live order execution)
- Strategy Router (Sentinel → Governor → Commander → RoutingMatrix)
- Kill Switch (ProgressiveKillSwitch, SmartKillSwitch, BotCircuitBreaker)
- Live trade records → SQLite before acknowledgment

**Contabo — everything else:**
- FloorManager + Department System
- Prefect workflow orchestration
- Redis (department mail + workflow event bus)
- Docker image: MT5 + MQL5 compiler (compilation + testing ONLY — not live trading)
- HMM training + retraining pipeline
- Custom backtest engine (primary)
- Alpha Forge pipeline
- Knowledge base + graph memory
- All agent tools and sub-agents
- Human approval gate
- Monitoring (Prometheus + Loki + Grafana)

**Rule:** Never add load to Cloudzy beyond live trading execution. All compilation, testing, backtesting, and AI workloads stay on Contabo.

---

#### 3.2 — EA Deployment Pipeline (FR77–79)

```
Compile (Docker/MT5 on Contabo)
    → Syntax validation (mql5_tools.validate_mql5_syntax())
    → Backtest: Monte Carlo + Walk-Forward (custom engine, Contabo)
      [MT5 backtest engine = secondary fallback if primary overloaded]
    → Human approval gate (7-day PENDING_REVIEW, daily re-surface)
    → Paper trading on Cloudzy (shadow mode, real data, no capital)
    → EA Circuit Breaker check: 3 consecutive losses → stays in paper
    → Mubarak explicit approval → live promotion
    → Deploy .ex5 to Cloudzy MT5 terminal
      [Deploy gate: Friday 22:00–Sunday 22:00 UTC only, no open positions]
```

**EA version system:**
- JSON config + version tag + .ex5 binary snapshot per deployment
- `pin_template_version: true` flag for frozen live strategies (prop firm evaluations)
- Git-hosted strategy source files (private repo)

**Strategy rollback:** Revert to prior JSON config + .ex5 binary snapshot. Git `checkout` + MT5 restart.

---

#### 3.3 — Cross-Strategy Patch Propagation (FR76)

**Decision:** Pre-compilation template system. Shared MQL5 template files (`.mqh` includes) for common components (Islamic compliance, risk sizing, session filters).

When a template is patched:
1. Prefect scheduled flow runs during market closure (markets-closed check: Friday 22:00–Sunday 22:00 UTC)
2. Recompile all strategies using that template (Docker/MT5 compile container on Contabo)
3. Strategies with `pin_template_version: true` are skipped
4. Newly compiled .ex5 binaries staged for Cloudzy deployment in next maintenance window

**Dependency tracking:** Each strategy config records which shared templates it uses. Patch = update template → recompile dependents.

**Borrow pattern:** React/npm-style dependency update model. Templates are the "library." Strategy configs declare template dependencies. Patch propagation is deterministic, not runtime.

---

#### 3.4 — Cloudzy Deployment Gate

**Rule:** Cloudzy is NEVER updated while markets are open or positions are open.

```python
def can_deploy_to_cloudzy(utc_now: datetime) -> bool:
    # Markets closed: Friday 22:00 UTC to Sunday 22:00 UTC
    is_weekend_window = ...  # check UTC weekday + hour
    no_open_positions = check_open_positions() == 0
    return is_weekend_window and no_open_positions
```

**CI/CD scope:** Auto-deploys to Contabo only. Cloudzy deployment = manual trigger via `/update-cloudzy` slash command, which runs the gate check first.

**Error handling requirement:** All codebase refactoring must include comprehensive `try/except` blocks and error logging before any Cloudzy deployment. Tested in dev ≠ tested in prod — catch errors defensively.

---

#### 3.5 — Runtime Provider Swap (FR66–70)

**Existing:** Provider CRUD + per-agent model swap via `PATCH /api/agent-config/{agent_id}/model` (confirmed in `src/api/model_config_endpoints.py`).

**Gaps to fill:**
- Hot-swap without agent restart (streaming connections need graceful reconnect)
- Provider fallback chain (Anthropic primary → fallback if rate-limited)

**Provider runtime rule:** Anthropic Agent SDK for ALL agent runtime. Other providers (GLM, MiniMax, DeepSeek, Alibaba) accessible via `base_url` swap for LLM API calls only — never for agent runtime orchestration. OpenRouter = never for Agent SDK.

**Vertical scaling:** NODE_ROLE env var means any server upgrade = no code change. Operational runbook (not architecture).

---

### 4. Agent Architecture

#### 4.1 — Department System (Canonical — No Changes)

```
FloorManager (Opus tier)
    ↓ classify_task() → dispatch via Redis Streams  [session_id assigned here]
    ↓
Department Heads (Sonnet tier): Research, Development, Trading, Risk, Portfolio
    ↓ Planning Sub-Agent → atomic task checkout → spawn sub-agents
    ↓
SubAgents (Haiku tier): mql5_dev, backtester,
    data_researcher, code_reviewer, strategy_optimizer, report_writer,
    market_analyst, portfolio_rebalancer, trade_monitor, fill_tracker,
    risk_calculator, executor, copilot
```

**Removed sub-agent types:** `analyst` and `quantcode` — deprecated and deleted from codebase. Do not re-add.

**Department Head role:** Monitor, manage, communicate, and orchestrate. Department Heads do NOT execute tasks — sub-agents do. Department Heads plan, route, and track.

**Critical rule — Agents do NOT trade:** All trade execution flows through the MT5 bridge on Cloudzy (Commander → RoutingMatrix → MT5Bridge → MT5 terminal). EAs execute trades. Agents MONITOR what EAs are doing. There is no such thing as an agent "paper trading" — paper trading means an EA runs on a MT5 demo account with real market data and no capital. The agent's role is to observe and report on that EA's performance.

**Trading Department sub-agent roles (corrected):**

| Sub-agent | Role |
|---|---|
| `TRADE_MONITOR` | Reads MT5 positions/stats via bridge, tracks EA performance, flags circuit breaker conditions |
| `FILL_TRACKER` | Reads fill data from MT5 bridge, logs to audit trail, detects slippage anomalies |
| `PERFORMANCE_REPORTER` | Aggregates EA stats across paper + live accounts → generates reports for morning digest |

**Brownfield note:** `src/agents/departments/subagents/trading_subagent.py` currently has `"PAPER TRADING ONLY"` hardcoded and simulates order execution via agent. This must be refactored to monitoring-only. The `ORDER_EXECUTOR` sub-agent type must be removed — order execution is not an agent responsibility.

**Planning Sub-Agent (new):** Each department gets a Planning Sub-Agent. When a batch of tasks arrives, the Planning Sub-Agent:
1. Reads the batch
2. Writes a brief spec per task (goal + success criteria + non-negotiables)
3. Routes each planned task to the appropriate sub-agent type
4. Monitors completion against the plan

**Department boundary rule:** No agent crosses department lines. If a department is overloaded, it queues internally and spawns more sub-agents within that department. No cross-department sub-agent spawning.

---

#### 4.2 — Task Priority System

**Decision:** Three-tier priority on all Redis task entries.

| Priority | Use Cases |
|---|---|
| HIGH | Urgent workflow gates, human-blocked tasks awaiting response, active Alpha Forge compile step |
| MEDIUM | Standard Alpha Forge stages, knowledge sync, workflow tasks |
| LOW | Background research, HMM retraining, knowledge indexing, maintenance jobs |

FloorManager sets priority at dispatch time. Department Heads claim HIGH tasks first.

---

#### 4.3 — Agent System Awareness & Identity Schema

**Decision:** System prompt indexing + formal agent identity. Each agent carries a typed identity that controls which sessions, skills, and tools it can access.

**Agent Identity Schema:**
```yaml
agent_identity:
  tier: dept_head | sub_agent
  department: research | development | trading | risk | portfolio
  sub_agent_type: mql5_dev | backtester | data_researcher | ...  # null for dept_head
  session_type: interactive | autonomous   # set at session spawn time
  session_id: "{uuid}"                     # assigned by FloorManager at dispatch
  memory_namespace:
    episodic:  ["dept", "{department}", "{session_id}", "episodic"]
    semantic:  ["dept", "{department}", "{session_id}", "semantic"]
    profile:   ["dept", "{department}", "profile"]    # cross-session, committed only
    global:    ["global", "strategies"]               # read-only shared knowledge
  allowed_skills:                          # loaded JIT via read_skill() tool
    global: [context-enrich, morning-digest]
    department: [skill_ids from own dept skill directory]
  activated_tools:                         # resolved at session start (see §16)
    global: [read_skill, write_memory, search_memory, read_canvas_context, request_tool]
    department: []                         # dept-specific tools — see §16 Tools Registry
    workflow_context: []                   # injected by Prefect flow directive if autonomous
```

**System prompt structure — indexed, not embedded:**
```markdown
# {Department} Head — System Prompt

You are the {Department} Department Head. [brief role description — 3–5 sentences]

## Responsibilities
[bullet list — brief]

## When [task type A], use:
→ Skill: `shared_assets/skills/departments/{dept}/{skill-name}/skill.md`  [load JIT]

## When [task type B], use:
→ Directive: `flows/directives/{workflow-name}/directive.md`  [load JIT]

## System Components Available
- Risk Engine: `src/risk/sizing/kelly_engine.py` — PhysicsAwareKellyEngine
- Strategy Router: `src/router/routing_matrix.py` — RoutingMatrix
- Session Detection: `src/router/sessions.py` — SessionDetector
- Islamic Compliance: `src/router/sessions.py` — is_past_islamic_cutoff()
- Kill Switch: `src/router/kill_switch.py` — ProgressiveKillSwitch
- Position Sizing: `src/risk/sizing/kelly_engine.py` — SizingRecommendation

## Tools Available
[see activated_tools from identity — injected at session start]
```

**JIT chain:** System prompt lists skill index → agent reads prompt → encounters task → calls `read_skill(skill_id)` → skill.md loaded into context → skill references tool → agent checks `activated_tools` → executes or calls `request_tool()` if missing.

System prompt templates are version-controlled (same lifecycle as skills: draft → active → deprecated). See §15 for template files.

---

#### 4.4 — Copilot Agent Design

**Copilot role:** Primary user-facing agent. Messenger, planner, and workflow trigger. Copilot does NOT execute trading actions, does NOT write Python/bash code autonomously.

**Copilot tools (to be defined in implementation stories):**
- Context Enrichment Skill (load before routing any request)
- Department mail dispatch tool
- Workflow trigger tools (one per registered Prefect flow)
- Shared assets query tool (RAG/CAG MCP)
- Task board read tool (view current state)
- Human approval gate tools (surface, approve, reject)
- Morning digest generator

**Slash commands = skills:** Every Copilot capability is invocable as a slash command. See Skill Catalogue below.

---

### 5. Workflow Orchestration

#### 5.1 — Prefect + Agent SDK + Flow Forge

**Decision:** Three-layer workflow system.

| Layer | Technology | Role |
|---|---|---|
| Scheduling & durability | [Prefect](https://docs.prefect.io) (self-hosted, Contabo) | What runs, when, state persistence, restart resilience |
| Agentic steps | Anthropic Agent SDK | Steps within flows that require department intelligence |
| Flow authoring | Flow Forge (Monaco + sandbox + Development agent) | Building new flows |

**Flow Forge pattern:**
1. Copilot writes a Directive spec (SDD-style: goal, success criteria, constraints)
2. Development Department's specialized flow-writing agent opens Monaco editor
3. Agent assembles flow from component library (`flows/components/`)
4. Sandboxed test run with component mock data fixtures
5. If tests pass → PR to `flows/assembled/` → committed to flow library
6. Prefect registers new flow → available for scheduling

**Copilot cannot:** Write Python/bash code directly, execute untested code against live systems, register flows without commit + sandbox validation.

---

#### 5.2 — Flow Component Library (DOE Execution Layer)

```
flows/
├── components/        ← atomic building blocks (typed inputs/outputs + JSON schema)
│   ├── fetch_tick_data.py
│   ├── run_backtest.py
│   ├── classify_regime.py
│   ├── call_department.py
│   ├── await_human_approval.py
│   └── send_notification.py
├── mock_data/         ← fixtures per component for sandbox testing
├── directives/        ← SOP .md files per flow (Directive layer)
├── assembled/         ← flows built from components (Execution layer)
└── sandbox/           ← isolated test runner
```

Each component has a JSON schema (inputs, outputs, error types). Type-safe wiring at assembly time. New components built by Development Department when genuinely new primitives are needed.

---

#### 5.3 — DOE Methodology (Architectural Principle)

The system follows the D.O.E. architecture across all agent work (source: `/home/mubarkahimself/Desktop/ROI/D.O.E_method.md` — Mubarak's personal methodology document defining Directive/Orchestration/Execution layers and the Self-Annealing pattern for autonomous failure recovery):

| Layer | Implementation |
|---|---|
| **Directive** (the What) | Department system prompts + skill `.md` files + Directives in `flows/directives/` |
| **Orchestration** (the Who) | FloorManager + Department Heads + Planning Sub-Agent |
| **Execution** (the How) | Flow component library + skill `executor.py` files + Agent tools + MQL5 compiler |

**DOE vs Prefect — distinct layers:**
- **Prefect** = workflow runtime (scheduling, state persistence, retry, durability)
- **DOE** = methodology for structuring what Prefect runs (Directive → Orchestration → Execution layers)
They are complementary, not competing.

**Self-annealing — three tiers (extended):**

| Tier | Trigger | Response |
|---|---|---|
| **Flow component** | Prefect step throws exception | Development Dept patches `executor.py` → sandbox validates → PR submitted. Flow Directive updated with failure pattern. |
| **Skill SOP** | Skill used + task completed but quality low (Reflector detects) | ReflectionExecutor queues Kanban LOW task → Development Dept patches `skill.md` → new `version` promoted. |
| **System prompt** | Prompt optimization cycle detects consistent failure pattern | Gradient optimizer proposes prompt diff → `risk_class` gate applies → new prompt version promoted. |

**Prompt optimization algorithms (native — no LangMem import):**
- `Gradient`: Reflection → critique → single-step update (2–3 LLM calls). For high-quality critical prompts.
- `Meta-Prompt`: Direct inference from patterns (1 LLM call). For fast iteration on low-risk prompts.
- `Prompt Memory`: Extract patterns from session traces (1 LLM call). For real-time feedback incorporation.

---

#### 5.4 — SDD (Spec-Driven Development) — Selective Use

**Decision:** SDD ([Spec-Driven Development — Martin Fowler, 2024](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)) is a discoverable, hookable skill — not a mandatory workflow layer.

**Where SDD is hook-enforced:**
- Alpha Forge entry point (video/request → strategy Directive): SDD fires ONCE to structure context before Development begins. Output = Directive. DOE takes over from Directive onward.
- PRD user journey entry points: SDD structures the task before FloorManager dispatches.

**Where SDD is blocked:**
- Alpha Forge autonomous improvement loop: SDD cannot run in continuous loops — it would freeze autonomous operation.

**What SDD does (correctly):** Structured context gathering — requirements, design considerations, success criteria. It does NOT enforce system constraints. Constraint application happens at compile time via templates, not at SDD stage.

**Alpha Forge — first version rule:** The first EA generated from a video source must mirror the video strategy as closely as possible. SDD structures the development brief. Vanilla variant = mirror image. Spiced variants = system-native adaptations added in subsequent Alpha Forge loop iterations.

---

#### 5.5 — Workflow Kill Switch (new)

**Two kill switches — completely separate:**

| Kill Switch | Scope | Mechanism |
|---|---|---|
| Trading Kill Switch (existing) | Cloudzy execution | ProgressiveKillSwitch → stops bot execution |
| Workflow Kill Switch (new) | Contabo Prefect | Cancels Prefect flow runs → marks CANCELLED in workflows.db |

Workflow Kill Switch does NOT affect live trading. Trading Kill Switch does NOT affect Prefect flows. Separate UI control surfaces.

**Workflow resume:** If a workflow is killed by mistake, Prefect's state persistence allows re-triggering from last completed step. A resume command is a separate Copilot slash command.

---

#### 5.6 — Human Approval Gate

**Decision:** Non-blocking with daily re-surface.

- Approval requested → workflow enters `PENDING_REVIEW` state
- Copilot morning digest re-surfaces all pending approvals daily
- 7-day hard timeout — strategy moves to `EXPIRED_REVIEW` (not auto-rejected, retrievable)
- Mubarak can resume at any time: "Resume EUR/USD supply demand strategy approval"
- Batch question surfacing: all pending questions from a workflow run held and presented together, not interrupt-per-question

---

### 6. Knowledge, Memory & Context

#### 6.1 — Graph Memory System

**Status: 80–90% implemented — EXTEND ONLY. Do not rebuild.**

Implementation lives at `src/memory/graph/` (types, store, operations, tier_manager, compaction, facade, migration). REST API at `src/api/graph_memory_endpoints.py`. Full plan at `docs/plans/2026-03-09-graph-memory-system.md`.

**Ontology positioning:** The graph memory system implements the **QUANTMINDX Domain Ontology**. `types.py` is the ontology layer — it defines the vocabulary of the domain (14 entity types, 18 relationship types). `store.py` is the instantiation layer — it holds the actual live data. This distinction matters: agents interact with a structured knowledge graph that enables explicit typed reasoning, not a flat memory store. The ontology is the architectural foundation that allows workflows and downstream agents to reason over what prior agents knew, decided, and did. See §19 for full ontology architectural positioning.

**Existing architecture (keep):**
- 14 node types: WORLD, BANK, OBSERVATION, OPINION, WORKING, PERSONA, PROCEDURAL, EPISODIC, CONVERSATION, MESSAGE, AGENT, DEPARTMENT, TASK, SESSION, DECISION
- 3 tiers: HOT (< 1hr) / WARM (< 30 days) / COLD (archived)
- 18 relationship types on edges
- Compaction with anchor preservation
- SQLite WAL mode with 8 indexes on `nodes` and `edges` tables
- `session_id`, `agent_id`, `department` already on `MemoryNode` ✓

**LangMem type mapping — already covered (no changes to node types):**

| LangMem Concept | Existing Node Type | Status |
|---|---|---|
| Episodic memory | `EPISODIC` | ✓ exact match |
| Semantic / factual triples | `WORLD` (FACTUAL category) | ✓ covered |
| Agent profile | `PERSONA` | ✓ exact match |
| Short-term / working | `WORKING` + `CONVERSATION` | ✓ covered |

LangMem pattern reference: [langchain-ai.github.io/langmem](https://langchain-ai.github.io/langmem/guides/use_tools_in_custom_agent/) and all guides listed below — patterns adopted natively. **LangMem / LangChain import is BANNED.**

LangMem guide index (patterns only, no library import):
- [use_tools_in_custom_agent](https://langchain-ai.github.io/langmem/guides/use_tools_in_custom_agent/) — tool integration pattern
- [dynamically_configure_namespaces](https://langchain-ai.github.io/langmem/guides/dynamically_configure_namespaces/) — namespace isolation
- [memory_tools](https://langchain-ai.github.io/langmem/guides/memory_tools/) — tool API design
- [extract_episodic_memories](https://langchain-ai.github.io/langmem/guides/extract_episodic_memories/) — episode capture
- [extract_semantic_memories](https://langchain-ai.github.io/langmem/guides/extract_semantic_memories/) — triple extraction
- [manage_user_profile](https://langchain-ai.github.io/langmem/guides/manage_user_profile/) — profile update pattern
- [summarization](https://langchain-ai.github.io/langmem/guides/summarization/) — running summary / token budget
- [delayed_processing](https://langchain-ai.github.io/langmem/guides/delayed_processing/) — ReflectionExecutor debounce pattern
- [optimize_memory_prompt](https://langchain-ai.github.io/langmem/guides/optimize_memory_prompt/) — prompt optimization
- [optimize_compound_system](https://langchain-ai.github.io/langmem/guides/optimize_compound_system/) — multi-prompt optimization
- [reference/memory](https://langchain-ai.github.io/langmem/reference/memory/) — store interface
- [reference/tools](https://langchain-ai.github.io/langmem/reference/tools/) — tool reference
- [reference/prompt_optimization](https://langchain-ai.github.io/langmem/reference/prompt_optimization/) — optimization algorithms
- [reference/utils NamespaceTemplate](https://langchain-ai.github.io/langmem/reference/utils/#langmem.utils.NamespaceTemplate) — namespace template
- [reference/short_term](https://langchain-ai.github.io/langmem/reference/short_term/) — short-term memory

**Schema additions (migration only — no rebuild):**

```sql
-- Add to existing `nodes` table:
ALTER TABLE nodes ADD COLUMN session_status TEXT DEFAULT 'committed';
-- Values: 'draft' (in-progress session) | 'committed' (session approved)

ALTER TABLE nodes ADD COLUMN embedding BLOB;
-- Vector embedding for semantic search. NULL until embedding pipeline active.
```

**New `sessions` table:**
```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    session_type     TEXT NOT NULL,         -- interactive | autonomous
    owner_dept       TEXT NOT NULL,
    owner_tier       TEXT NOT NULL,         -- dept_head | sub_agent
    status           TEXT NOT NULL,         -- active | waiting_human | idle | complete | failed
    created_at_utc   TEXT NOT NULL,
    completed_at_utc TEXT,
    workflow_id      TEXT                   -- Prefect flow run ID if autonomous session
);
```

**Workspace isolation via `session_status`:**
- All memory writes from active sessions → `session_status = 'draft'`
- Interactive sessions read only `session_status = 'committed'` (no dirty reads)
- On session complete + DeptHead approval → `UPDATE nodes SET session_status = 'committed' WHERE session_id = ?`
- Multiple concurrent autonomous sessions are isolated — each writes its own `session_id` draft; merge order determined by DeptHead evaluation sequence

**Embeddings (new — sentence-transformers, local on Contabo):**
- Model: `all-MiniLM-L6-v2` (384-dim, free, no API cost per query)
- Same embedding backend as ChromaDB — unified infrastructure
- Enables semantic search to replace keyword-only `RECALL` operation
- `embedding` column nullable — keyword search fallback if embedding not yet generated

**ReflectionExecutor (new file: `src/memory/graph/reflection_executor.py`):**
- Async, debounced: waits 5 minutes after session settles before processing
- Cancels if new session activity arrives (avoids processing mid-session)
- Extracts episodic + semantic memories from session traces → writes to `draft` namespace
- FloorManager reviews on session commit → approves/rejects memory batch

**What still needs implementation (gaps from plan review):**
1. LLM integration in `REFLECT` operation (currently plain concatenation) → call Anthropic API
2. Connect `session_checkpoint_service.py` → graph memory session commit flow
3. Add `session_status` + `embedding` columns via migration
4. Implement `ReflectionExecutor`
5. Vector search in `store.py` `query_nodes()` using embedding cosine similarity
6. Tests (`tests/memory/` — TDD plan exists, not yet implemented)

---

#### 6.2 — Canvas Context System (Generative UI pattern)

**Decision:** Each canvas has a `CanvasContextTemplate` — a map of identifiers loaded when a new interactive session starts. Autonomous workflow sessions use flow directives instead (see §14).

**Extended CanvasContextTemplate schema:**
```python
CanvasContextTemplate(canvas="RISK"):
    base_descriptor: "You are the Risk Department Copilot..."
    memory_scope: ["risk.*", "portfolio.*", "trading.*"]   # Graph memory namespaces (committed only)
    workflow_namespaces: ["risk_workflows", "portfolio_workflows"]
    department_mailbox: "risk_dept_mail stream"
    shared_assets: ["risk_templates", "prop_firm_rules"]
    skill_index:                                           # NEW — indexed skills for this canvas
      - id: "drawdown-review"
        path: "shared_assets/skills/departments/risk/drawdown-review/skill.md"
        trigger: "when reviewing drawdown or risk thresholds"
      - id: "context-enrich"
        path: "shared_assets/skills/global/context-enrich/skill.md"
        trigger: "before routing any request"
    required_tools:                                        # NEW — tools activated at session start
      - risk_calculator.compute_kelly_size
      - write_memory
      - search_memory
      - read_skill
      - request_tool
    session_type: interactive                              # NEW — always interactive for canvas
    # All values are IDENTIFIERS, not content
    # Agents load content JIT when needed
```

**CAG + RAG combined pattern:**
- CAG layer: stable identifiers (canvas descriptor, dept SOP path, skill index, tool list) — pre-assembled on canvas load
- RAG layer: live state fetched JIT (HOT graph memory nodes `committed` only, current task board, recent department mail)

**Memory isolation for interactive sessions:** Template loads only `session_status = committed` memory nodes. In-progress autonomous sessions' drafts are invisible until committed. No context poisoning.

**Direct department chat:** Each Department Head has an entity-specific template. "Chat with Research Head" → loads `CanvasContextTemplate(entity=RESEARCH_HEAD)`. Default = FloorManager template.

**New chat on canvas:** Template loaded fresh. Session spawned with new `session_id`. Live state (memory, workflows, task board) fetched at session start. Templates define what to fetch, not the data itself — always current.

---

#### 6.3 — Shared Assets Library

**Structure:**
```
/shared_assets/
├── docs/          ← scraped articles (PageIndex-indexed)
├── books/         ← mql5.pdf, mql5book.pdf (PageIndex-indexed, JIT query)
├── strategies/    ← templates, AlgoForge pulled assets
├── indicators/    ← MQL5 indicator libraries
├── skills/        ← agentic skills (.md files, slash commands)
├── mcps/          ← MCP server configs
└── components/    ← Flow Forge component library
```

**Agent registration:** Department config declares `shared_resources: [...]`. Agent SDK hook pre-loads identifiers (not content) on session start.

**Skill access control:**
```yaml
skill: /sdd-spec
used_by: [development_dept, alpha_forge_workflow, copilot]
access: restricted   # hidden from agents not in used_by list
```
Prevents context poisoning — agents cannot accidentally load skills outside their access scope.

**AlgoForge integration:** Pull strategies/libraries from [AlgoForge marketplace](https://www.algomql5.com) (MQL5 community marketplace — `mql5.com` ecosystem) → upload to QuantMindX → sync to `shared_assets/strategies/` → accessible to all departments. UI: upload/download panel in AssetsView.

---

#### 6.4 — OPINION Node Pattern (Mandatory Reasoning Artifact)

Every OPINION node is a **persisted reasoning record** — not a memory of what happened (that is OBSERVATION), but a record of *why* an agent made a particular decision or assessment. OPINION nodes are the mechanism by which downstream agents understand the intent behind prior work without needing to re-derive it.

**The core value:** Agent A acts → writes OPINION node with reasoning + SUPPORTED_BY edges to evidence → Agent B (downstream, different session) reads OPINION → understands why Agent A made that choice → acts with full context, not just artifacts.

---

**When an OPINION node is mandatory — "consequential action" definition:**

An action is consequential if it does any of the following:

| Category | Examples |
|---|---|
| Changes persisted state | DB write, file write, configuration change, EA deployment |
| Produces an artifact | EA file, strategy config, backtest result, knowledge entry, compiled `.ex5` |
| Makes a routing or prioritization decision | "Prioritise EURUSD over GBPUSD this session", "Defer this strategy to next week" |
| Takes a position on market conditions or strategy | Market assessment, regime classification, risk score interpretation |
| Approves, rejects, or escalates a task | Promote strategy to live, reject EA variant, escalate to FloorManager |
| Triggers or blocks another agent's work | "Hold off on this workflow step until ECB clears" |

NOT consequential (no OPINION node required):
- Reading files, querying memory, context enrichment
- Routine status updates to FloorManager
- Formatting or transforming data without decision-making
- Loading a skill or tool

**Scope:** Mandatory in **both session types** — autonomous workflows AND interactive sessions. In autonomous sessions, enforced mechanically by the flow directive. In interactive sessions, enforced by the system prompt and context-enrich skill.

---

**OPINION node content schema:**

The `content_summary` field on OPINION nodes must follow this structure:

```json
{
  "action": "Brief description of what was decided or done",
  "reasoning": "Why this decision was made — the actual reasoning, not a restatement of the action",
  "confidence": "low | medium | high",
  "alternatives_considered": ["Option A rejected because...", "Option B deferred because..."],
  "constraints_applied": ["islamic_cutoff", "prop_firm_max_dd", "calendar_blackout"],
  "agent_role": "research_dept_head | risk_sub_agent | ..."
}
```

`importance_score` for OPINION nodes: default `0.7`. Increase to `0.9` for decisions that gate downstream workflows. Decrease to `0.5` for assessments that are informational only.

---

**Mandatory edge rule — no orphaned OPINIONs:**

Every OPINION node must have at least one `SUPPORTED_BY` edge to an OBSERVATION, WORLD, or DECISION node that provided the factual basis for the opinion. An OPINION with no supporting evidence edges is architecturally invalid and must not be committed.

```
OPINION (agent's decision/assessment)
  └── SUPPORTED_BY ──► OBSERVATION (what the agent observed)
  └── SUPPORTED_BY ──► WORLD (market fact / price data / calendar event)
  └── INFORMED_BY  ──► DECISION (a prior decision this builds on)
```

**Forbidden:** Orphaned OPINION nodes (no `SUPPORTED_BY` edges). These cannot be meaningfully queried by downstream agents and defeat the purpose of the pattern.

---

**Where OPINIONs originate:**

| Source | How OPINION is created | Example |
|---|---|---|
| **Agent runtime** | Agent writes OPINION node immediately after consequential action | Research head assesses a market regime → writes OPINION node |
| **Skills** | Skill procedure includes OPINION-write step as part of execution | `/drawdown-review` skill: final step writes OPINION with assessment and recommended action |
| **System prompts** | System prompt instructs agent to write OPINION after any action meeting the consequential threshold | All dept head prompts include the OPINION obligation rule |

No dedicated workflow component is needed. The obligation is embedded in skills and system prompts — it is part of how agents are instructed to behave, not a separate runtime mechanism.

---

**How downstream agents use OPINION nodes:**

At context enrichment time (session start, `read_skill("global/context-enrich")`), agents query OPINION nodes relevant to their current task context:

```python
# In context-enrich skill procedure:
opinions = graph.recall(
    node_type=NodeType.OPINION,
    department=["research", "risk"],      # relevant depts
    session_status="committed",
    tags=["eurusd", "strategy_123"],      # task-relevant tags
    tier=[Tier.HOT, Tier.WARM],
    limit=5
)
# Load into WORKING memory nodes for this session
```

Downstream agents do not re-derive prior reasoning — they load committed OPINIONs and treat them as trusted context from prior agents.

---

### 7. Risk Engine & Trading

#### 7.1 — Kelly Engine (Confirmed — Brownfield)

**Status:** `PhysicsAwareKellyEngine` is already extensively enhanced. Do NOT replace — extend.

**Current enhancements confirmed:** Lyapunov multiplier, Ising multiplier, Eigenvalue multiplier, weakest-link aggregation (M_physics), Monte Carlo validation, Governor integration, PropFirm overlay, House Money State (DB model).

**Gap to fill:** Fee module (`EnhancedKellyCalculator`) is a separate file — needs integration directly into the main Kelly engine so fee awareness is applied at sizing time, not as a post-processing step.

---

#### 7.2 — Routing Matrix (Confirmed — Brownfield)

**Status:** `RoutingMatrix` exists with Two-Front War strategy (Machine Gun + Sniper accounts). Bot-to-broker assignment via compatibility scoring. Do NOT replace — extend.

**Purpose (confirmed):** Distribute strategies across brokers so no single broker is overloaded with too many bots. Not account reassignment — bots stay on their assigned broker.

**Gap to fill — Fee-aware selection:** Add to `AccountConfig`:
```python
commission_per_lot: float    # USD round-trip
avg_spread_pips: dict        # {"EURUSD": 0.1, ...}
```
Integrate into compatibility scoring so high-frequency scalpers are assigned to lowest-commission broker.

---

#### 7.3 — Broker Registry Enhancement

**Extend existing `BrokerRegistry` with:**
```python
BrokerConfig(
    ...existing fields...,
    commission_per_lot: float,
    avg_spread_pips: dict,
    account_types: list,        # ["ECN", "PRIME", "RAW"]
    supports_scalping: bool,
    server_timezone: str,       # IANA timezone string for MT5 timestamp normalisation
)
```

**Broker registration flow:** `/register-broker` slash command → Copilot asks questions → fills BrokerConfig schema → routes to Settings. Aligns with existing broker registration in codebase.

---

#### 7.4 — Session Registry & Multi-Session Trading

**Decision:** Replace hardcoded session enum with dynamic `SessionConfig` registry.

```python
SessionConfig(
    name: str,                     # "SYDNEY", "LONDON", "NEW_YORK", "TOKYO", "CUSTOM"
    utc_open: str,                 # "22:00"
    utc_close: str,                # "07:00"
    crosses_midnight: bool,
    iana_timezone: str,            # "Australia/Sydney"
    active_instruments: list,      # ["AUDUSD", "AUDJPY"]
    dst_aware: bool,
)
```

**Strategy session preference:**
```python
StrategyConfig(
    ...
    preferred_sessions: list,    # ["LONDON", "NEW_YORK"]
    blocked_sessions: list,      # ["ASIAN"] — skip signal routing during these
)
```

**Session-aware routing gate:** `SessionDetector.is_strategy_session_active(strategy, utc_now)` called before routing any tick signal. If outside preferred sessions → signal skipped.

**Multi-component update scope:** Adding a new trading session requires updates to: (1) Session Registry data, (2) Strategy Router routing gate, (3) Kelly/risk config for session-specific risk multipliers, (4) UI journey — "Add Trading Session" panel.

**DST handling:** Session windows updated twice yearly via Prefect scheduled flow (`flows/assembled/update_session_windows.py`).

---

#### 7.5 — Time Zones (Architectural Rule)

**Rule: UTC everywhere internally. No exceptions.**

| Boundary | Rule |
|---|---|
| All DB timestamps | UTC |
| All Prefect schedules | UTC cron expressions |
| All log entries | ISO 8601 UTC |
| MT5 tick timestamps | Normalised to UTC at ZMQ ingest (broker GMT+2/+3 → UTC) |
| UI display | Convert from UTC to user's local timezone at display layer only |
| Islamic force-close | 21:45 UTC (not GMT+2, not broker time — UTC hard-coded) |

**SessionDetector addition:** `is_past_islamic_cutoff(utc_time: datetime) -> bool` — first-class method, not a string comparison. Called by Commander and injected into every EA template compilation.

**MT5 server timezone:** Stored in `BrokerConfig.server_timezone`. ZMQ bridge reads this to perform UTC normalisation on tick ingestion.

---

#### 7.6 — Account Isolation (Confirmed — Brownfield)

**Status:** `AccountMonitor`, `PropFirmAccount`, `DailySnapshot`, `AccountLossState` confirmed in codebase.

**EA Circuit Breaker:** 3 consecutive losses → automatic move to paper trading. `BotCircuitBreaker` already exists.

**Cross-account margin safeguard:** NOT applicable. Account A (RoboForex) and Account B (Exness) are separate brokers with separate equity pools. No combined equity exposure calculation needed.

**Runtime account reassignment:** NOT implemented and NOT desired. Bots stay on their assigned broker. Manual override only via Copilot command: "Move bot XYZ to demo account."

---

#### 7.7 — Strategy Router Clarification

**What the Strategy Router does:** Finds optimal market CONDITIONS (volatility, liquidity, session overlap, regime classification) and routes EAs towards those conditions.

**What it does NOT do:** Find setups — setup logic lives inside the EA itself. Reassign bots between brokers. The routing matrix handles bot-to-broker assignment separately.

**A/B live testing:** Implemented via HMM deployment stages — ISING_ONLY → HMM_SHADOW → HMM_HYBRID → HMM_ONLY. Log-based performance comparison between stages. No live capital splitting required.

---

### 8. Alpha Forge Pipeline (FR23–FR31, FR74–79)

#### 8.1 — Full Pipeline

```
SOURCE (video URL / Mubarak request / existing strategy)
    ↓
RESEARCH DEPARTMENT — VideoIngestProcessor (5 stages):
    Download → Frame extraction → Audio extraction → AI analysis → Timeline
    VideoAnalysisTools → extracted_elements (indicators, rules, risk params)
    Output: VideoAnalysisResult — UNBIASED extraction, no system constraints applied
    ↓
SDD SKILL (fires ONCE — context structuring, not constraint enforcement):
    "What are we building? What does success look like? What's missing?"
    Output: Strategy Directive (.md) — goal, success criteria, open questions
    Questions routed: check graph memory → check shared assets → Copilot → Mubarak if unresolved
    ↓
DEVELOPMENT DEPARTMENT:
    Vanilla variant: mirror image of source video strategy (as close as possible)
    MQL5 compliance templates injected at compilation (Islamic, prop firm, session rules)
    MQL5Tools.generate_ea_from_strategy() → raw MQL5 code
    Docker compile (MT5 compiler, Contabo) → .ex5 binary
    ↓
BACKTESTING (DOE autonomous loop begins here):
    Data: [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) → distillation variants (tool in component library)
    Spiced variants: system-native adaptations (session filters, ATR-based stops, prop firm rules)
    Engines: Custom backtest engine (primary), MT5 engine (secondary fallback)
    Tests: Monte Carlo + Walk-Forward + SIT
    ↓
HUMAN APPROVAL GATE:
    Results surfaced in morning digest (daily re-surface)
    7-day PENDING_REVIEW timeout
    ↓
PAPER TRADING (Cloudzy):
    Shadow mode — real data, no capital
    HMM shadow output monitored
    EA Circuit Breaker active (3 consecutive losses → stays in paper)
    ↓
LIVE PROMOTION (explicit Mubarak approval only):
    RoutingMatrix assigns to Machine Gun or Sniper broker account
    Deploy .ex5 to Cloudzy MT5 terminal (weekend deploy window)
    Strategy Router begins routing conditions to this EA
```

#### 8.2 — Video Ingest Provider Migration

**Current state:** Video ingest uses OpenRouter as primary provider (confirmed in codebase).
**Decision:** Migrate to direct Anthropic Claude (multimodal/vision) via Anthropic API. OpenRouter is deprecated in the video analysis pipeline.

---

### 9. Skill Catalogue

**Skill system design — see §15 (Living Skills System) for full architecture.**

Skills are two-part units: `skill.md` (Directive — what to do and how) + optional `executor.py` (Execution — deterministic code). Written by Claude Code during development. Templates at `shared_assets/skills/templates/`. Skill ≠ Tool (see §16 Tools Registry).

**Canonical skill structure (hierarchical):**
```
shared_assets/skills/
├── global/                          # any dept, any session type
│   ├── context-enrich/
│   │   └── skill.md
│   └── morning-digest/
│       └── skill.md
└── departments/
    ├── development/
    │   ├── forge-strategy/
    │   │   ├── skill.md
    │   │   └── executor.py
    │   ├── review-ea/
    │   │   └── skill.md
    │   ├── sdd-spec/
    │   │   └── skill.md
    │   ├── validate-sdd/
    │   │   └── skill.md
    │   └── update-cloudzy/
    │       └── skill.md
    ├── risk/
    │   └── drawdown-review/
    │       └── skill.md
    └── portfolio/
        └── system-targets/
            └── skill.md
```

**Skill metadata (frontmatter in `skill.md`):**
```yaml
id: forge-strategy
version: "1.0.0"
status: active            # draft | active | deprecated
tier: department          # global | department
owned_by: development
slash_command: /forge-strategy
session_types:
  - autonomous            # interactive | autonomous | both
risk_class: medium        # low | medium | high  (controls autonomous self-improvement)
has_executor: false       # true if executor.py exists
```

**Existing 12 skills mapped to new hierarchy:**

| Skill | Slash Command | Tier | Dept | `session_types` | `risk_class` |
|---|---|---|---|---|---|
| Context Enrichment | `/context-enrich` | global | — | both | low |
| Morning Digest | `/morning-digest` | global | — | interactive | low |
| SDD Spec Builder | `/sdd-spec` | department | development | both | medium |
| SDD Validator | `/validate-sdd` | department | development | both | medium |
| Forge Strategy | `/forge-strategy` | department | development | autonomous | medium |
| Retrain HMM | `/retrain-hmm` | department | development | autonomous | medium |
| Review EA | `/review-ea` | department | development | both | high |
| Update Cloudzy | `/update-cloudzy` | department | development | interactive | high |
| Migration Runbook | `/migration-runbook` | department | development | interactive | high |
| Drawdown Review | `/drawdown-review` | department | risk | both | medium |
| System Targets | `/system-targets` | department | portfolio | both | medium |
| Register Broker | `/register-broker` | department | portfolio | interactive | medium |

**Workflow-specific directives** (not skills — live in `flows/directives/`):
These are flow-scoped SOPs used only within a specific Prefect workflow. They are not in `shared_assets/skills/`.

---

### 10. Performance Targets Journey (New FR)

**Decision:** Add user journey — "Set a performance target and let the system work autonomously towards it."

**Flow:**
- Mubarak sets target via `/system-targets` (e.g., "15% monthly growth")
- Copilot structures a plan using Context Enrichment Skill + existing strategy inventory
- FloorManager holds the target as persistent context — every dispatched task is evaluated against: "does this move us towards the target?"
- Alpha Forge loop runs autonomously towards target
- Morning digest reports progress against target
- Human in the loop at all approval gates

---

### 11. Audit Log (FR59–65)

**Decision:**
- General logs: 6 months–1 year retention
- Trading records (trade history, P&L, position logs): 3 years
- Database is primary store — logs are supplementary/search layer
- Natural language query: Copilot calls `get_audit_log_schema()` tool → constructs DuckDB/SQLite query → returns structured result. Schema queried dynamically (not hardcoded) — schema evolves without breaking the query tool.

---

## Implementation Patterns & Consistency Rules

_These patterns prevent AI agents writing incompatible code. Every agent implementing any part of QUANTMINDX MUST follow these rules — they are not preferences._

---

### Naming Patterns

#### Database Naming

| Convention | Rule | Example |
|---|---|---|
| Tables | snake_case, plural | `calendar_events`, `trade_records`, `workflow_runs` |
| Columns | snake_case | `impact_tier`, `account_id`, `commission_per_lot` |
| Timestamp columns | `*_utc` suffix — mandatory | `created_at_utc`, `published_utc`, `expires_at_utc` |
| Foreign keys | `{table_singular}_id` | `broker_id`, `strategy_id`, `workflow_id` |
| Indexes | `idx_{table}_{column}` | `idx_trade_records_strategy_id` |

The `_utc` suffix on every timestamp column is non-optional. It enforces UTC awareness at the schema level and prevents silent local-time bugs.

#### Database Table Inventory — Canonical Names

All new tables introduced in this architecture cycle. These are the locked names — agents must not invent variations.

**Trading & Execution (Cloudzy SQLite)**

| Table | Purpose |
|---|---|
| `trade_records` | Immutable live trade log — written before acknowledgment |
| `trading_journal` | Per-bot annotated trade history — entry, exit, P&L, session, duration, notes, win/loss |
| `ea_deployments` | Deployment history per EA — version_tag, ex5_path, config_path, status, broker_id, deployed_at_utc |

**Strategy & Config (Contabo SQLite)**

| Table | Purpose |
|---|---|
| `strategies` | Strategy registry — strategy_id, name, source_type, source_url, current_phase, created_at_utc |
| `ea_versions` | EA version history — version_tag, source_hash, template_deps (JSON), pin_template_version, strategy_id (FK), variant_type (vanilla\|spiced\|mode_b\|mode_c), improvement_cycle (int) |
| `ea_backtest_reports` | One row per report per variant — mode (VANILLA/SPICED/VANILLA_FULL/SPICED_FULL/MODE_B/MODE_C), report_type (basic/monte_carlo/walk_forward/pbo), data_variant (NULL=standard or Dukascopy slice label), report_path, net_profit, sharpe_ratio, max_drawdown_pct, profit_factor, passed_mc, passed_wf, passed_pbo |
| `ea_improvement_cycles` | Workflow 2 iteration tracker — strategy_id, cycle_number, status (running/complete/failed), variants_produced, best_version_tag |
| `broker_configs` | Broker registry (existing, extended) — commission_per_lot, avg_spread_pips, server_timezone, account_types |
| `session_configs` | Dynamic session registry replacing hardcoded enum — name, utc_open, utc_close, iana_timezone, active_instruments |
| `prop_firm_configs` | Prop firm rule sets — drawdown limits, daily loss, news blackout windows, prohibited behaviors |
| `flow_registry` | Registered Prefect flows — flow_name, flow_path, description, last_updated_utc, is_active |

**Knowledge & Memory (Contabo SQLite / DuckDB)**

| Table | Purpose |
|---|---|
| `graph_memory_nodes` | Agent cross-session memory — node_id, node_type (WORLD/BANK/OBSERVATION/OPINION/WORKING/PERSONA/PROCEDURAL/EPISODIC), content_summary, importance_score, tags, tier (HOT/WARM/COLD), created_at_utc, last_accessed_utc |
| `graph_memory_edges` | Relationships between memory nodes — source_node_id, target_node_id, relationship_type, weight |

**News & Calendar (Contabo SQLite — synced subset to Cloudzy)**

| Table | Purpose |
|---|---|
| `calendar_events` | Economic calendar cache — event_id, datetime_utc, currency, event_name, impact_tier (1/2/3), actual, forecast, previous |
| `news_items` | Live news feed — item_id, headline, summary, source, published_utc, url, related_instruments (JSON), severity, action_type |

**Audit & Monitoring (Contabo SQLite — 3-year retention for trades)**

| Table | Purpose |
|---|---|
| `audit_log` | Immutable event log — log_id, entity_type, entity_id, action, actor, timestamp_utc, payload_json |
| `performance_targets` | Goal tracking — target_id, metric, target_value, period, created_at_utc, status |

**What is NOT a database table:**
- `canvas_context_templates` → YAML config files per canvas, not DB rows
- Flow components → Python files in `flows/components/`, not DB rows
- Graph memory COLD tier → `data/departments/cold_storage.db` (separate DB file, not a table in the main DB)
- Prefect workflow state → `workflows.db` managed entirely by Prefect — never write to it directly

#### API Endpoint Naming

- Lowercase, hyphenated, plural nouns: `/api/calendar-events`, `/api/broker-configs`
- Verb sub-resources for actions: `/api/hmm/train`, `/api/workflows/{id}/cancel`
- No trailing slashes. Never camelCase in URL paths.

#### Python Code Naming

| Element | Convention | Example |
|---|---|---|
| Files / modules | snake_case | `calendar_governor.py`, `news_provider.py` |
| Functions / methods | snake_case | `check_calendar_restriction()`, `is_past_islamic_cutoff()` |
| Classes / Pydantic models | PascalCase | `CalendarGovernor`, `BrokerConfig`, `NewsItem` |
| Constants | UPPER_SNAKE_CASE | `NODE_ROLE`, `MAX_RETRY_COUNT`, `ISLAMIC_CUTOFF_UTC` |
| Variables | snake_case | `impact_tier`, `account_id` |

#### Frontend (Svelte / TypeScript) Naming

| Element | Convention | Example |
|---|---|---|
| Component files | PascalCase | `StatusBand.svelte`, `TileCard.svelte` |
| TypeScript interfaces | PascalCase, no `I` prefix | `BrokerConfig`, `CalendarEvent` |
| Props | camelCase | `brokerConfig`, `impactTier` |
| Event handlers | `on` + PascalCase | `onApprovalSubmit`, `onCanvasSwitch` |
| CSS custom properties | Exact token names from design system | `--color-accent-amber`, `--font-data` |

#### Redis Key Naming

Colon-separated namespacing, all lowercase, no dots:

```
dept:research:wf_001:queue
mail:dept:development:wf_001
workflow:wf_001:events
calendar:events:upcoming_7d
news:alerts:high_severity
task:task_abc123:owner
```

---

### Structure Patterns

#### Backend File Organization

```
src/
├── agents/
│   ├── departments/heads/       ← department head implementations
│   ├── departments/subagents/   ← sub-agent implementations (monitor only — no order execution)
│   └── departments/types.py     ← SubAgentType enum (source of truth)
├── api/
│   ├── contabo_routers/         ← Contabo-only (agents, knowledge, workflow, settings)
│   ├── cloudzy_routers/         ← Cloudzy-only (trading, kill switch, MT5 bridge)
│   └── server.py                ← NODE_ROLE controls which router groups register
├── risk/                        ← Kelly engine, physics sensors, prop firm registry
├── router/                      ← RoutingMatrix, SessionDetector, CalendarGovernor, kill switches
├── knowledge/                   ← ChromaDB, PageIndex, graph memory
├── flows/                       ← Prefect flows (components/, directives/, assembled/, sandbox/)
└── monitoring/                  ← Prometheus, Loki, metrics
tests/                           ← mirrors src/ structure exactly
```

**Import prefix rule — `from src.{module}` always:**
```python
# CORRECT
from src.risk.sizing.kelly_engine import PhysicsAwareKellyEngine
from src.agents.departments.types import SubAgentType

# WRONG
from risk.sizing.kelly_engine import PhysicsAwareKellyEngine
from .kelly_engine import PhysicsAwareKellyEngine   # only for same-package siblings
```

**Test location:** `tests/` mirroring `src/` — not co-located. `tests/risk/test_kelly_engine.py` tests `src/risk/sizing/kelly_engine.py`.

#### Frontend File Organization

```
quantmind-ide/src/lib/
├── components/
│   ├── shell/           ← TopBar, StatusBand, ActivityBar, AgentPanel (global shell)
│   ├── live-trading/    ← Live Trading canvas components
│   ├── research/        ← Research canvas components
│   ├── development/     ← Development canvas components
│   ├── risk/            ← Risk canvas components
│   ├── trading/         ← Trading canvas (paper monitoring, backtest review)
│   ├── portfolio/       ← Portfolio canvas + Trading Journal
│   ├── shared-assets/   ← Shared Assets canvas components
│   ├── workshop/        ← Workshop canvas + FloorManager Copilot
│   ├── flowforge/       ← FlowForge visual canvas + Prefect Kanban
│   └── shared/          ← Reusable cross-canvas components (TileCard, GlassSurface, etc.)
├── api/                 ← apiFetch.ts ONLY — no raw fetch() anywhere else
├── stores/              ← Global reactive state (node health, StatusBand data)
└── types/               ← Shared TypeScript interfaces
```

---

### Format Patterns

#### API Response Formats

| Case | Format |
|---|---|
| Single resource | Direct Pydantic model JSON |
| List | `{ "items": [...], "total": N }` — never a bare array |
| Error | FastAPI default `{ "detail": "..." }` — do not invent custom envelopes |
| Validation error | FastAPI 422 automatic |

Never return `200 OK` for error states.

#### Date / Time

```python
# CORRECT
from datetime import datetime, timezone
now = datetime.now(timezone.utc)         # always timezone-aware

# WRONG
now = datetime.now()                     # naive datetime — forbidden
now = datetime.utcnow()                  # deprecated — use timezone.utc
```

All DB columns storing dates: `DATETIME` type, UTC value, `_utc` column name suffix.
API responses: ISO 8601 UTC string — `"2026-03-14T10:30:00Z"`.
UI display: convert UTC → user local timezone at the **display layer only** — never store local time.

#### JSON Field Naming

All API payloads (request + response): **snake_case** only. Never camelCase.

#### Pydantic v2 — Mandatory Syntax

```python
# CORRECT — Pydantic v2
class BrokerConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    broker_id: str
    commission_per_lot: float = Field(gt=0)

obj.model_dump()              # not .dict()
BrokerConfig.model_validate(d) # not .parse_obj()

# WRONG — Pydantic v1, do not write
class Config:
    orm_mode = True
obj.dict()
BrokerConfig.parse_obj(d)
```

---

### Communication Patterns

#### Frontend API Calls — `apiFetch` Only

```typescript
// CORRECT
import { apiFetch } from '$lib/api/apiFetch';
const result = await apiFetch<BrokerListResponse>('/api/broker-configs');

// WRONG — never in a component
const result = await fetch('http://localhost:8000/api/broker-configs').then(r => r.json());
```

`apiFetch` routes by endpoint prefix: `/api/trading/*`, `/api/kill-switch/*`, `/api/mt5/*` → Cloudzy. All other `/api/*` → Contabo. Components never know which server they're talking to.

#### Agent → Agent Communication — Redis Streams Only

```python
# CORRECT
await redis.xadd(f"mail:dept:{target_dept}:{workflow_id}", {"task": task_json})

# WRONG — do not write to SQLite mail bus in new code
department_mail_service.send(...)
```

#### Department Mail — Structured Payload (Anti Context Rot)

Redis Streams is the transport. The payload must follow a structured template — never free-form JSON blobs. This prevents context rot when recipients load mail content.

**Mail payload schema:**
```python
DepartmentMail(
    mail_id: str,                          # uuid
    from_dept: str,                        # "research", "development", "risk", "floor_manager"
    to_dept: str,                          # "development" | "broadcast"
    workflow_id: str,
    subject: str,                          # one-line summary, ≤100 chars
    body: str,                             # structured SUMMARY only — never the content of attachments
    attachments: list[MailAttachment],     # references only — content loaded JIT by recipient
    priority: Literal["HIGH", "MEDIUM", "LOW"],
    reply_to_mail_id: str | None,
    sent_at_utc: str,                      # ISO 8601 UTC
)

MailAttachment(
    tag: str,                              # semantic tag: "strategy_ea", "research_doc",
                                           #   "backtest_result", "url", "knowledge_entry",
                                           #   "workflow_artifact", "mql5_code"
    label: str,                            # human-readable: "EURUSD Supply Demand EA v2"
    reference: str,                        # file path OR URL — NEVER the content itself
    attachment_type: Literal["file", "url", "knowledge_entry", "workflow_artifact"],
    indexed_id: str | None,               # PageIndex/ChromaDB ID if already indexed — recipient
                                           #   can fetch directly without re-indexing
)
```

**Rules:**
- `body` = structured summary of what is being sent and why. It contains NO file content, NO embedded MQL5 code, NO raw article text.
- Attachments are **references only** (file paths, URLs, knowledge entry IDs). The recipient loads content JIT via tool call when they actually need it.
- URLs must include `indexed_id` if already scraped into PageIndex — recipient navigates the index, not the raw URL dump.
- EA files: referenced by file path + version tag. Binary content is never embedded in mail.

**Who can send mail (hard rule):**

| Sender | Can mail | Cannot mail |
|---|---|---|
| FloorManager | Any department | — |
| Department Head | Other Department Heads, FloorManager | Cannot bypass FloorManager to spawn work in another dept |
| Sub-agent | Their own Department Head ONLY (internal report up) | Any other department, any other sub-agent |

Sub-agents report UP within their department. They do NOT send cross-department mail. Cross-department communication is exclusively Department Head → Department Head via the `mail:dept:{dept}:{workflow_id}` stream.

**Internal (sub-agent → dept head) uses a different stream:**
```
dept:{dept}:{workflow_id}:internal    ← sub-agent results/reports to their dept head
mail:dept:{dept}:{workflow_id}        ← cross-dept mail (dept heads only)
```

#### Frontend Streaming — Two Separate Channels

| Data | Protocol | Source |
|---|---|---|
| Agent events, Copilot responses, workflow status | SSE `GET /api/agents/stream` | Contabo |
| Live P&L, positions, kill switch events | WebSocket | Cloudzy |

Never merge these. Agent Panel consumes SSE. StatusBand trading data consumes WebSocket.

#### Svelte 5 State — Runes Only for New Code

```svelte
<!-- CORRECT — Svelte 5 runes -->
<script lang="ts">
  let count = $state(0);
  let doubled = $derived(count * 2);
  $effect(() => { syncToServer(count); });
  let { broker }: { broker: BrokerConfig } = $props();
</script>

<!-- WRONG — Svelte 4 syntax. Do not write in new components. -->
<script lang="ts">
  export let broker: BrokerConfig;    // use $props() instead
  $: doubled = count * 2;             // use $derived() instead
</script>
```

Canvas-local state lives in the canvas component. Global state (node health, StatusBand) in `src/lib/stores/`.

---

### Visual Design Patterns

#### CSS Custom Properties — Mandatory

```svelte
<!-- CORRECT — always tokens -->
<div style="background: var(--color-bg-surface); font-family: var(--font-data)">

<!-- WRONG — hardcoded values break all theme switching -->
<div style="background: rgba(8,13,20,0.6); font-family: 'JetBrains Mono'">
```

#### Numbers Always Monospace

Every financial figure (P&L, balance, lot size, risk score, timestamp) uses `var(--font-data)` (JetBrains Mono). Never rendered in a variable-width face.

#### Icons — Lucide Only, Never Emoji

```svelte
<!-- CORRECT -->
import { AlertTriangle } from 'lucide-svelte';
<AlertTriangle size={16} color="var(--color-accent-red)" />

<!-- WRONG -->
<span>⚠️ Risk alert</span>
```

#### Canvas Transition Budget

- Canvas switches (ActivityBar click): ≤200ms perceived
- Tile → sub-page: ≤200ms
- Loading states: skeleton pulse using `--color-bg-elevated` — never a white flash

#### Kill Switch Visual Rules

- **Trading Kill Switch** — TopBar ONLY. Amber glow when live → confirm modal → red when stopped. Two-step activation always.
- **Workflow Kill Switch** — FlowForge Kanban row ONLY. Red destructive per-workflow action. Two-step activation.
- These two controls never share a surface. Neither causes a global app halt.

---

### Process Patterns

#### Error Handling — Backend

```python
# CORRECT — all new backend code, mandatory on Cloudzy-bound paths
try:
    result = await some_operation()
except SomeSpecificError as e:
    logger.error("Operation failed: %s", str(e), exc_info=True)
    raise HTTPException(status_code=500, detail="Operation failed")
except Exception as e:
    logger.error("Unexpected error in %s: %s", __name__, str(e), exc_info=True)
    raise

# WRONG
except:
    pass    # never silently swallow errors
```

#### Retry Policies

| Operation | Max Retries | Backoff |
|---|---|---|
| AI API calls | 3 | Exponential: 2s → 4s → 8s |
| Redis operations | 5 | Linear: 100ms |
| MT5 ZMQ reconnect | Continuous | Until reconnected, ≤10s target |
| Inter-node HTTP (Contabo ↔ Cloudzy) | 3 | Exponential: 1s → 2s → 4s |

Never retry trading execution without checking current position state first.

#### Node Placement Decision Rule

Before implementing any backend feature:

| Question | Answer → Node |
|---|---|
| Needs MT5 ZMQ tick data or order execution? | Cloudzy |
| Runs AI agents or Prefect flows? | Contabo |
| Must survive Contabo being unreachable? | Cloudzy |
| Stores/processes backtesting, HMM, knowledge? | Contabo |
| When in doubt? | Contabo — keep Cloudzy minimal |

---

#### OPINION Node Obligation (Mandatory — All Agents)

After any **consequential action** (see §6.4 for definition), an agent MUST write an OPINION node to the graph before returning or handing off. This is not optional.

**The checklist every agent checks after acting:**

```
□ Did this action change persisted state?          → write OPINION
□ Did this action produce an artifact?             → write OPINION
□ Did this action make a routing/priority decision? → write OPINION
□ Did this action assess market conditions?         → write OPINION
□ Did this action approve/reject/escalate?          → write OPINION
□ None of the above?                               → no OPINION needed
```

**Minimum valid OPINION write:**
```python
graph.retain(MemoryNode(
    node_type=NodeType.OPINION,
    content_summary=json.dumps({
        "action": "...",
        "reasoning": "...",
        "confidence": "medium",
        "alternatives_considered": [],
        "constraints_applied": [],
        "agent_role": agent_identity.role,
    }),
    department=agent_identity.department,
    session_id=session_id,
    session_status="draft",
    importance_score=0.7,
    tags=[...],           # task-relevant tags for downstream retrieval
))
# Must follow immediately with SUPPORTED_BY edge to at least one OBSERVATION or WORLD node
```

**Enforcement points:** Skills that include consequential actions must include the OPINION-write as an explicit step. System prompts include the obligation as a standing instruction. Code review rejects agent implementations that take consequential actions without OPINION writes.

---

### Forbidden Patterns — Hard Rules

| Forbidden | Reason |
|---|---|
| `import langchain` / `import langgraph` | Deprecated — Anthropic Agent SDK only |
| `paper_trader` or `live_trader` SubAgentType | Agents monitor. EAs trade. No agent "paper trades." |
| Agent calling MT5 bridge to place orders | Agents are read-only observers of trading |
| Raw `fetch(...)` in Svelte components | Use `apiFetch<T>()` only |
| `datetime.now()` without `timezone.utc` | Naive datetime — always explicit UTC |
| Pydantic `.dict()`, `.parse_obj()`, `class Config:` | Pydantic v1 — use v2 equivalents |
| Writing to `department_mail.db` in new code | Use Redis Streams |
| Hardcoded server URLs in Svelte components | Route via `apiFetch` |
| Emoji in the UI | Use `lucide-svelte` icons |
| Hardcoded color / spacing values in CSS | Use CSS custom property tokens |
| Svelte 4 `export let`, `$:` reactive, `writable()` in new code | Use Svelte 5 runes |
| Financial numbers in variable-width fonts | Always `var(--font-data)` — JetBrains Mono |
| Any Trading Kill Switch outside TopBar | TopBar only |
| Workflow Kill Switch as a global button | Contextual per-workflow in FlowForge Kanban only |

---

### 13. News & Economic Calendar

---

#### 13.1 — News Provider Abstraction

**Decision:** Same abstraction pattern as AI providers. `NewsProvider` interface with swappable implementations. One free primary, one paid slot (FMP).

| Provider | Cost | Economic Calendar | Live News Wire | Status |
|---|---|---|---|---|
| [Finnhub](https://finnhub.io/docs/api) | Free (60 req/min) | Yes — impact tier 1/2/3 | Yes — forex + general headlines | **Primary (default)** |
| [Financial Modeling Prep (FMP)](https://financialmodelingprep.com/developer/docs) | ~$15/mo | Yes — richer, more currencies | Yes — fuller wire coverage | **Paid slot (optional)** |

`FINNHUB_API_KEY` and `FMP_API_KEY` in `.env`. Settings → Providers panel adds a **News Provider** row alongside AI provider rows — same toggle pattern. Switching provider takes effect immediately (no restart).

**Python SDK:** `finnhub-python` for Finnhub. FMP uses direct REST calls.

---

#### 13.2 — Economic Calendar Kill Switch (Cloudzy)

**Decision:** Extend `EnhancedGovernor` with a `CalendarGovernor` mixin. Lives on **Cloudzy** — calendar enforcement must be local, never dependent on Contabo being reachable.

**Cloudzy calendar cache:** Contabo polls the news provider daily → caches the next 7 days of events → syncs to Cloudzy via the existing rsync cron. Cloudzy reads its local cache. Never polls the external API directly.

**Calendar event schema:**
```python
CalendarEvent(
    event_id: str,
    datetime_utc: datetime,
    currency: str,            # "USD", "EUR", "GBP"
    event_name: str,          # "NFP", "CPI", "FOMC Rate Decision"
    impact_tier: int,         # 1=low, 2=medium, 3=high
    actual: float | None,
    forecast: float | None,
    previous: float | None,
)
```

**Kill switch windows — configurable per PropFirmConfig:**
```python
PropFirmConfig(
    ...existing fields...,
    news_blackout_pre_minutes: dict,    # {1: 0, 2: 15, 3: 30}   — per tier
    news_blackout_post_minutes: dict,   # {1: 0, 2: 15, 3: 60}
    news_no_new_positions_minutes: int, # FTMO: 120 (2h) for Tier 2+3 events
)
```

**Default config (non-prop-firm accounts):**
- Tier 3 (High): pause all scalpers 30 min before, 15 min after
- Tier 2 (Medium): scale to 50% lot size 15 min before, normal after
- Tier 1 (Low): no restriction

**Governor check:**
```python
def check_calendar_restriction(strategy, utc_now) -> CalendarRestriction | None:
    upcoming = get_events_in_window(strategy.currencies, utc_now, lookahead_minutes=120)
    for event in sorted(upcoming, key=lambda e: e.impact_tier, reverse=True):
        rule = get_blackout_rule(event.impact_tier, strategy.account.prop_firm_config)
        if within_pre_window(event, utc_now, rule.pre_minutes):
            return CalendarRestriction(type="PAUSE", reason=f"{event.event_name} ★{event.impact_tier}")
        if within_post_window(event, utc_now, rule.post_minutes):
            return CalendarRestriction(type="SCALE", factor=0.0, reason=f"Post-{event.event_name} blackout")
        if rule.no_new_positions_minutes and within_pre_window(event, utc_now, rule.no_new_positions_minutes):
            return CalendarRestriction(type="NO_NEW_ENTRIES", reason=f"Prop firm rule: {event.event_name}")
    return None
```

**Journey 45 example in practice:** NFP every Friday — Thursday 18:00 UTC → Tier 3 30-min pre-window begins. Friday 13:15 UTC → pause. Friday 14:00 UTC → regime check runs, Governor evaluates post-window expiry. Friday 15:00 UTC → post-window cleared, normal operation resumes. FTMO account: 2h no-new-entries window applied on top.

---

#### 13.3 — Live News Feed (FR50 — Contabo → UI)

**Decision:** Research sub-agent polls news provider every 60s on Contabo. Articles stored in `news_items` table (SQLite). Frontend polls `GET /api/news/feed` every 60s. Displayed in Live Trading canvas news section.

**News article schema:**
```python
NewsItem(
    item_id: str,
    headline: str,
    summary: str,
    source: str,
    published_utc: datetime,
    url: str,
    related_instruments: list[str],   # ["USDJPY", "AUDJPY"]
    severity: Literal["LOW", "MEDIUM", "HIGH"] | None,  # set by sub-agent after classification
    action_type: Literal["MONITOR", "ALERT", "FAST_TRACK"] | None,
)
```

**Severity classification (geopolitical sub-agent):**
- Runs as a Research Department sub-agent on Contabo
- Reads each new article → classifies severity + affected instruments + action type
- HIGH + ALERT → department mail to Copilot → Copilot surfaces priority notification on UI
- HIGH + FAST_TRACK → triggers Fast-Track Event Workflow (see §13.4)
- **Target: sub-90s from article publish to Copilot alert** (poll 60s + classify <30s)

---

#### 13.4 — Fast-Track Event Workflow

**Trigger:** Geopolitical sub-agent classifies article as `FAST_TRACK`.

**Flow:**
```
News article arrives (Finnhub / FMP)
    → Research sub-agent classifies FAST_TRACK
    → Identifies affected instruments (currency pairs)
    → Confirms via Sentinel: volatility + volume elevated on affected pairs
    → If confirmed → department mail to Development dept
    → Development: match template from shared_assets/strategies/news_templates/
        (match by: trigger_currencies + event_type + direction_bias)
    → Template deployed directly to Cloudzy (skips full Alpha Forge pipeline)
    → Strategy tagged: expiry=7d, source=fast_track, event={event_name}
    → Copilot notifies Mubarak: "Fast-track deployed: {template_name} on {pairs}"
```

**Constraints:**
- Excluded: scalping strategies (lower-timeframe logic). Fast-track = trend-following and swing templates only.
- Requires at least one matching template in library — no template = alert only, no deployment
- Cadence: weekly/monthly (not continuous auto-deployment)
- Mubarak can reject/cancel via Copilot: "Cancel fast-track EUR/USD deployment"

---

#### 13.5 — News Event Templates (Journey 22)

**Decision:** Pre-built templates stored in `shared_assets/strategies/news_templates/`. Separate from the main EA factory. Built once via Prefect historical analysis flow.

**Template build flow (Prefect, Contabo):**
1. Research queries Dukascopy data for last 5 years around scheduled event dates (NFP, CPI, FOMC, BOE, ECB)
2. Statistical analysis per instrument: pre-event drift, release spike magnitude, post-event continuation rate
3. Templates with sufficient edge (confidence score ≥ threshold) → saved to `news_templates/`
4. Templates below threshold → discarded (not archived as active)
5. Multi-condition rules embedded at build time: "no entry within 48h of FOMC", "GBPUSD only during BOE"

**Template schema:**
```yaml
# news_templates/nfp_usd_momentum.yaml
name: "NFP USD Momentum"
trigger_currencies: ["USD"]
event_types: ["NFP", "Non-Farm Payrolls"]
direction_bias: "momentum"       # follow the surprise direction
entry_delay_minutes: 5           # wait for initial spike to settle
expiry_days: 7
excluded_within_hours: 48        # of FOMC
confidence_score: 0.72
built_from: "2020-01-01/2025-12-31 Dukascopy"
```

---

#### 13.6 — Smart Money / Hedge Fund Tracking

**Status: Phase 2 / Vision — NOT in current architecture.**

Explicitly deferred. Requires a separate ML layer to infer institutional positioning from public filings (COT reports, 13F), order flow imbalance, or dark pool prints. The Research sub-agent and knowledge base infrastructure are the correct hooks when this is eventually built. No architectural decisions needed now.

---

## Project Structure & Boundaries

_Legend: `[exists]` = confirmed in codebase · `[extend]` = exists, needs changes · `[new]` = does not exist yet · `[refactor]` = exists but must be rewritten_

---

### Monorepo Root

```
QUANTMINDX/
├── src/                          [exists] Python FastAPI backend (both nodes, NODE_ROLE splits)
├── quantmind-ide/                [exists] Tauri 2 + SvelteKit 5 frontend
├── tests/                        [exists] Python tests — mirrors src/ structure
├── flows/                        [new]    Prefect flow library (components, directives, assembled)
├── shared_assets/                [new]    Cross-agent resources (skills, templates, MCP configs)
├── scripts/                      [exists] Utility + cron scripts
├── config/                       [exists] YAML config files
├── docker/                       [exists] Docker configs
├── systemd/                      [exists] Systemd service files (NODE_ROLE set per server)
├── mcp-servers/                  [exists] Internal MCP servers
├── monitoring/                   [exists] Prometheus + Loki + Grafana configs
├── data/                         [exists] Cold storage, cold_storage.db
├── docs/                         [exists] Architecture docs, plans, API contracts
├── docker-compose.yml            [exists] Local dev
├── docker-compose.production.yml [exists] Production (both nodes)
├── requirements.txt              [extend] Remove langchain, langgraph, langchain_openai
├── pytest.ini                    [exists]
├── .env                          [exists] All secrets — NEVER committed
└── CLAUDE.md                     [exists] AI agent instructions
```

---

### Backend — `src/`

#### API Layer — Router Reorganization

```
src/api/
├── server.py                         [extend] Add NODE_ROLE router group selection
│
├── contabo_routers/                   [new]    Contabo-only router group
│   ├── __init__.py
│   ├── agents.py                      [new]    → migrated from agent_management_endpoints.py
│   ├── floor_manager.py               [new]    → migrated from floor_manager_endpoints.py
│   ├── knowledge.py                   [new]    → migrated from ide_knowledge.py + pdf_endpoints.py
│   ├── workflows.py                   [new]    → migrated from workflow_endpoints.py
│   ├── hmm.py                         [new]    → migrated from hmm_endpoints.py
│   ├── news.py                        [new]    News feed + calendar endpoints (FR50)
│   ├── research.py                    [new]    Research dept endpoints
│   ├── backtest.py                    [new]    → migrated from ide_backtest.py
│   ├── settings.py                    [new]    → migrated from settings_endpoints.py
│   ├── providers.py                   [new]    → migrated from provider_config_endpoints.py
│   ├── alpha_forge.py                 [new]    Alpha Forge pipeline trigger endpoints
│   └── graph_memory.py               [new]    → migrated from graph_memory_endpoints.py
│
├── cloudzy_routers/                   [new]    Cloudzy-only router group
│   ├── __init__.py
│   ├── trading.py                     [new]    → migrated from trading_endpoints.py
│   ├── mt5_bridge.py                  [new]    → migrated from ide_mt5.py
│   ├── kill_switch.py                 [new]    → migrated from kill_switch_endpoints.py
│   ├── strategy_router.py             [new]    → migrated from router_endpoints.py
│   └── calendar_gate.py              [new]    Calendar kill switch enforcement (FR — Journey 45)
│
└── shared_routers/                    [new]    Available on both nodes
    ├── health.py                      [new]    → migrated from health_endpoints.py
    ├── session.py                     [new]    → migrated from session_endpoints.py
    └── audit.py                       [new]    Audit log query endpoint (FR59–65)
```

#### Agent System

```
src/agents/
├── departments/
│   ├── types.py                       [extend] Remove paper_trader, live_trader SubAgentType
│   │                                           Add TRADE_MONITOR, FILL_TRACKER, PERFORMANCE_REPORTER
│   ├── heads/
│   │   ├── floor_manager_head.py      [exists]
│   │   ├── research_head.py           [exists]
│   │   ├── development_head.py        [exists]
│   │   ├── trading_head.py            [extend] Monitor-only — remove order execution
│   │   ├── risk_head.py               [exists]
│   │   └── portfolio_head.py          [exists]
│   └── subagents/
│       ├── trading_subagent.py        [refactor] Remove "PAPER TRADING ONLY" order execution
│       │                                         Rewrite as TRADE_MONITOR / FILL_TRACKER / PERFORMANCE_REPORTER
│       ├── mql5_dev_subagent.py       [exists]
│       ├── research_subagent.py       [exists]
│       ├── geopolitical_subagent.py   [new]    News severity classifier + fast-track trigger (Journey 46)
│       └── planning_subagent.py       [new]    Planning Sub-Agent per department
│
├── mail/
│   ├── __init__.py
│   ├── department_mail.py             [new]    DepartmentMail + MailAttachment schema
│   └── mail_router.py                 [new]    Redis Stream dispatch — dept heads only
│
├── skills/                            [exists] Skill execution engine
└── registry.py                        [extend] Migrate from LangGraph to Anthropic Agent SDK
```

#### Risk Engine

```
src/risk/
├── sizing/
│   ├── kelly_engine.py                [extend] Integrate EnhancedKellyCalculator fee module
│   ├── kelly_fee_integration.py       [new]    Fee-aware sizing bridge
│   └── portfolio_kelly.py             [exists]
├── physics/                           [exists] Ising, Lyapunov, HMM sensors
├── governor.py                        [exists]
├── enhanced_governor.py               [extend] Add CalendarGovernor mixin
└── prop/
    └── prop_firm_registry.py          [extend] Add news_blackout_pre/post_minutes fields
```

#### Strategy Router (Cloudzy)

```
src/router/
├── routing_matrix.py                  [extend] Add fee-aware broker selection
├── sessions.py                        [extend] Add is_past_islamic_cutoff(), SessionConfig registry
├── session_configs.py                 [new]    Dynamic SessionConfig DB model + CRUD
├── calendar_governor.py               [new]    CalendarGovernor mixin for EnhancedGovernor
├── kill_switch.py                     [exists] ProgressiveKillSwitch, SmartKillSwitch
├── progressive_kill_switch.py         [exists]
├── bot_circuit_breaker.py             [exists] BotCircuitBreaker
└── broker_registry.py                 [extend] Add commission_per_lot, server_timezone fields
```

#### Knowledge & Memory

```
src/knowledge/
├── chromadb_store.py                  [new]    ChromaDB semantic search wrapper
├── page_index_client.py               [new]    PageIndex query client (JIT section retrieval)
├── graph_memory/
│   ├── __init__.py
│   ├── graph_memory_store.py          [extend] Finish partial implementation
│   ├── memory_nodes.py                [extend] All 8 node types + HOT/WARM/COLD tiers
│   └── memory_edges.py               [new]    graph_memory_edges table model
├── news/
│   ├── news_provider.py               [new]    NewsProvider abstract interface
│   ├── finnhub_provider.py            [new]    Finnhub implementation (primary, free)
│   ├── fmp_provider.py                [new]    FMP implementation (paid slot)
│   └── news_classifier.py             [new]    Severity + action_type classification
└── canvas_context/
    ├── templates/                     [new]    YAML CanvasContextTemplate files per canvas
    │   ├── live_trading.yaml
    │   ├── research.yaml
    │   ├── development.yaml
    │   ├── risk.yaml
    │   ├── trading.yaml
    │   ├── portfolio.yaml
    │   ├── shared_assets.yaml
    │   ├── workshop.yaml
    │   └── flowforge.yaml
    └── context_loader.py              [new]    CAG+RAG assembly on canvas load
```

#### Data & Storage

```
src/database/
├── tiered_storage.py                  [exists] TieredStorageRouter HOT/WARM/COLD
├── models/
│   ├── trade_records.py               [exists]
│   ├── trading_journal.py             [new]    trading_journal table model
│   ├── ea_deployments.py              [new]    ea_deployments table model
│   ├── ea_versions.py                 [new]    ea_versions table model
│   ├── strategies.py                  [new]    strategies registry table model
│   ├── calendar_events.py             [new]    calendar_events table model
│   ├── news_items.py                  [new]    news_items table model
│   ├── audit_log.py                   [new]    audit_log table model (immutable)
│   ├── performance_targets.py         [new]    performance_targets table model
│   ├── flow_registry.py               [new]    flow_registry table model
│   ├── broker_configs.py              [extend] Add commission, spread, timezone fields
│   ├── session_configs.py             [new]    session_configs table model
│   └── prop_firm_configs.py           [extend] Add news blackout fields
└── migrations/                        [new]    Alembic migrations
```

---

### Prefect Flow Library — `flows/`

```
flows/
├── components/                        [new]    Atomic typed building blocks
│   ├── fetch_tick_data.py
│   ├── run_backtest.py
│   ├── classify_regime.py
│   ├── call_department.py
│   ├── await_human_approval.py
│   ├── send_notification.py
│   ├── compile_ea.py
│   ├── deploy_ea.py
│   ├── sync_calendar_events.py
│   ├── scrape_news.py
│   └── distill_historical_data.py
├── mock_data/                         [new]    Fixtures per component for sandbox tests
├── directives/                        [new]    SOP .md files — the Directive layer per flow
│   ├── alpha_forge.md
│   ├── hmm_retrain.md
│   ├── cross_strategy_patch.md
│   ├── drawdown_review.md
│   ├── calendar_sync.md
│   └── news_template_build.md
├── assembled/                         [new]    Complete flows built from components
│   ├── alpha_forge_flow.py
│   ├── hmm_retrain_flow.py
│   ├── cross_strategy_patch_flow.py
│   ├── ea_deployment_flow.py
│   ├── calendar_sync_flow.py
│   ├── update_session_windows.py
│   └── news_template_builder_flow.py
└── sandbox/                           [new]    Isolated flow test runner
    └── runner.py
```

---

### Shared Assets — `shared_assets/`

```
shared_assets/
├── skills/                            [new]    Slash command skill .md files
│   ├── context-enrich.md
│   ├── sdd-spec.md
│   ├── validate-sdd.md
│   ├── morning-digest.md
│   ├── forge-strategy.md
│   ├── retrain-hmm.md
│   ├── review-ea.md
│   ├── drawdown-review.md
│   ├── system-targets.md
│   ├── register-broker.md
│   ├── update-cloudzy.md
│   └── migration-runbook.md
├── strategies/
│   ├── templates/                     [new]    Base EA templates (Islamic, prop firm, session)
│   └── news_templates/                [new]    News event strategy templates (NFP, CPI, FOMC)
├── indicators/                        [new]    Shared MQL5 indicator libraries
├── docs/                              [new]    Scraped articles (PageIndex-indexed)
├── books/                             [new]    mql5.pdf + mql5book.pdf (symlinks or copies)
├── mcps/                              [new]    MCP server config files
└── components/                        [new]    Flow component docs + schemas (mirrors flows/components/)
```

---

### Frontend — `quantmind-ide/`

#### Global Shell (all canvases)

```
quantmind-ide/src/lib/components/shell/
├── TopBar.svelte                      [refactor] Add kill switch (Trading scope), Copilot shortcut,
│                                                 Notifications, Settings — 40px fixed
├── StatusBand.svelte                  [extend]  Make all segments clickable nav shortcuts
├── ActivityBar.svelte                 [extend]  8 canvas icons + Settings — icon-only
└── AgentPanel.svelte                  [refactor] Dept-head-specific, streaming thoughts + tool calls,
                                                  CanvasContextTemplate loads on new chat
```

#### Canvas Components

```
quantmind-ide/src/lib/components/
├── live-trading/                      [new]    Canvas 1 — CEO morning dashboard
│   ├── LiveTradingCanvas.svelte
│   ├── tiles/
│   │   ├── ActiveBotsTile.svelte
│   │   ├── EquityCurveTile.svelte
│   │   ├── OpenPositionsTile.svelte
│   │   ├── DrawdownTile.svelte
│   │   ├── RegimeIndicatorTile.svelte
│   │   ├── NodeHealthTile.svelte
│   │   ├── NewsFeedTile.svelte       ← FR50 live news feed
│   │   └── WorkflowStatusTile.svelte
│   └── sub-pages/
│       └── PositionDetail.svelte
│
├── research/                          [new]    Canvas 2 — Research dept
│   ├── ResearchCanvas.svelte
│   ├── tiles/
│   │   ├── AlphaForgeEntryTile.svelte ← YouTube URL input → pipeline trigger
│   │   ├── KnowledgeBaseTile.svelte
│   │   ├── VideoIngestTile.svelte
│   │   └── HypothesisPipelineTile.svelte
│   └── sub-pages/
│       ├── KnowledgeArticle.svelte
│       └── VideoAnalysisResult.svelte
│
├── development/                       [new]    Canvas 3 — Development dept
│   ├── DevelopmentCanvas.svelte
│   ├── tiles/
│   │   ├── EALibraryTile.svelte
│   │   ├── AlphaForgePipelineTile.svelte
│   │   └── BacktestQueueTile.svelte
│   └── sub-pages/
│       ├── EAEditor.svelte            ← Monaco editor
│       └── BacktestResults.svelte
│
├── risk/                              [new]    Canvas 4 — Risk dept
│   ├── RiskCanvas.svelte
│   ├── tiles/
│   │   ├── KellyEngineTile.svelte
│   │   ├── PhysicsSensorsTile.svelte  ← shadow mode indicator
│   │   ├── PropFirmComplianceTile.svelte
│   │   └── ValidationQueueTile.svelte
│   └── sub-pages/
│       └── PropFirmDetail.svelte
│
├── trading/                           [new]    Canvas 5 — Trading dept (monitoring)
│   ├── TradingCanvas.svelte
│   ├── tiles/
│   │   ├── PaperTradingMonitorTile.svelte ← EA performance monitoring, not agent trading
│   │   ├── BacktestResultsTile.svelte
│   │   └── EAPerformanceTile.svelte
│   └── sub-pages/
│       └── EAPerformanceDetail.svelte
│
├── portfolio/                         [new]    Canvas 6 — Portfolio + Trading Journal
│   ├── PortfolioCanvas.svelte
│   ├── tiles/
│   │   ├── LivePnLTile.svelte
│   │   ├── AllocationTile.svelte
│   │   ├── CorrelationMatrixTile.svelte
│   │   └── TradingJournalTile.svelte  ← FR — confirmed from old IDE
│   └── sub-pages/
│       ├── AccountDetail.svelte
│       └── TradingJournalEntry.svelte
│
├── shared-assets/                     [new]    Canvas 7 — Shared resources hub
│   ├── SharedAssetsCanvas.svelte
│   └── tiles/
│       ├── DocsLibraryTile.svelte
│       ├── StrategyTemplatesTile.svelte
│       ├── IndicatorsTile.svelte
│       ├── SkillsTile.svelte
│       └── FlowComponentsTile.svelte
│
├── workshop/                          [new]    Canvas 8 — FloorManager Copilot home
│   ├── WorkshopCanvas.svelte          ← Claude.ai-inspired, morning digest on first load
│   ├── MorningDigest.svelte
│   ├── ChatHistory.svelte
│   └── SuggestionChips.svelte         ← CAG+RAG-powered slash command suggestions
│
├── flowforge/                         [new]    Canvas 9 — Prefect Kanban + visual flow builder
│   ├── FlowForgeCanvas.svelte
│   ├── PrefectKanban.svelte           ← Workflow Kill Switch lives HERE (per-workflow row)
│   ├── FlowNodeGraph.svelte           ← N8N-style visual composer
│   └── FlowComponentPalette.svelte
│
└── shared/                            [new]    Reusable across all canvases
    ├── TileCard.svelte                ← base glass tile (frosted terminal aesthetic)
    ├── GlassSurface.svelte            ← backdrop-filter + scan-line mixin
    ├── SkeletonLoader.svelte          ← pulse animation with --color-bg-elevated
    ├── FilePreviewOverlay.svelte      ← non-destructive read/inspect overlay
    ├── NotificationTray.svelte        ← frosted glass overlay (approval gates, alerts)
    ├── ConfirmModal.svelte            ← two-step confirmation (kill switches)
    ├── Breadcrumb.svelte              ← tile → sub-page back navigation
    └── RichRenderer.svelte            ← inline diagrams, tables, charts in agent output
```

#### Frontend Core

```
quantmind-ide/src/lib/
├── api/
│   └── apiFetch.ts                    [extend] Add Cloudzy prefix routing, typed generics
├── stores/
│   ├── nodeHealth.ts                  [new]    Contabo + Cloudzy + Local health state
│   ├── statusBand.ts                  [new]    StatusBand reactive data (bots, P&L, session)
│   ├── activeCanvas.ts                [new]    Current canvas state machine
│   └── notifications.ts              [new]    Approval gate + alert notification queue
├── types/
│   ├── trading.ts                     [new]    BrokerConfig, SessionConfig, EADeployment, etc.
│   ├── agents.ts                      [new]    DepartmentMail, MailAttachment, SubAgentType, etc.
│   ├── knowledge.ts                   [new]    NewsItem, CalendarEvent, GraphMemoryNode, etc.
│   └── ui.ts                          [new]    Canvas, Tile, Theme, NotificationItem, etc.
└── ws-client.ts                       [exists] WebSocket factory (Cloudzy trading stream)
```

---

### FR → File Mapping

| FR Domain | Key Files |
|---|---|
| FR1–9 Live Trading | `src/router/kill_switch.py`, `src/router/routing_matrix.py`, `cloudzy_routers/trading.py`, `cloudzy_routers/kill_switch.py` |
| FR10–22 Agent System | `src/agents/departments/`, `src/agents/mail/`, `contabo_routers/agents.py`, `contabo_routers/floor_manager.py` |
| FR23–31 Alpha Forge | `flows/assembled/alpha_forge_flow.py`, `flows/directives/alpha_forge.md`, `contabo_routers/alpha_forge.py`, `shared_assets/strategies/templates/` |
| FR32–41 Risk & Compliance | `src/risk/sizing/kelly_engine.py`, `src/risk/enhanced_governor.py`, `src/router/calendar_governor.py`, `src/router/sessions.py` |
| FR42–50 Knowledge & Research | `src/knowledge/`, `src/video_ingest/`, `contabo_routers/knowledge.py`, `components/research/` |
| FR50 News Feed | `src/knowledge/news/`, `contabo_routers/news.py`, `components/live-trading/tiles/NewsFeedTile.svelte` |
| FR51–58 Portfolio & Broker | `src/router/broker_registry.py`, `src/database/models/broker_configs.py`, `components/portfolio/` |
| FR59–65 Audit & Notifications | `src/database/models/audit_log.py`, `shared_routers/audit.py`, `components/shared/NotificationTray.svelte` |
| FR66–73 Infrastructure | `src/api/server.py` (NODE_ROLE), `systemd/`, `src/api/contabo_routers/providers.py` |
| FR74–79 EA Pipeline & A/B | `flows/assembled/ea_deployment_flow.py`, `flows/assembled/cross_strategy_patch_flow.py`, `src/router/hmm_deployment.py` |
| Journey 45 Calendar Gate | `src/router/calendar_governor.py`, `src/database/models/calendar_events.py`, `cloudzy_routers/calendar_gate.py` |
| Journey 46 Live News | `src/knowledge/news/geopolitical_subagent.py`, `src/agents/subagents/geopolitical_subagent.py` |
| Trading Journal | `src/database/models/trading_journal.py`, `components/portfolio/tiles/TradingJournalTile.svelte` |
| Graph Memory | `src/knowledge/graph_memory/`, `src/database/models/` (graph_memory_nodes + edges) |
| Canvas Context System | `src/knowledge/canvas_context/`, `components/shell/AgentPanel.svelte` |
| FlowForge | `flows/`, `components/flowforge/`, `contabo_routers/workflows.py` |
| Skill Catalogue | `shared_assets/skills/` (12 × .md files) |

---

### 14. Session Architecture

#### 14.1 — Two Session Types (Architecturally Separate)

All agent activity runs inside a **session**. Every session has a UUID `session_id` assigned by FloorManager at dispatch time. Two types — their skill sets, tool sets, and memory access patterns are kept strictly separate.

| Property | Interactive Session | Autonomous Workflow Session |
|---|---|---|
| **Initiator** | Human (Copilot chat, DeptHead direct chat) | Prefect flow, Alpha Forge loop, skill improvement cycle |
| **UI surface** | Full chat panel in Agent Panel | Status card only (no chat input) |
| **Context source** | `CanvasContextTemplate` (CAG, loaded at session start) | Flow directive `flows/directives/{name}/directive.md` (DOE) |
| **Skills available** | `session_types: interactive` or `both` | `session_types: autonomous` or `both` |
| **Memory reads** | `session_status = committed` only | `session_status = committed` only |
| **Memory writes** | `session_status = draft` → committed on session end | `session_status = draft` → committed after DeptHead evaluation |
| **Tools** | From agent identity + canvas context `required_tools` | From agent identity + flow directive `required_tools` |

#### 14.2 — Workspace Isolation (Custom Worktrees)

Multiple concurrent sessions can work on related artifacts without interfering. This is the equivalent of git worktrees — each session works in its own namespace until committed.

**How it works:**
1. Session spawned → `session_id` assigned → Redis namespace `session:{session_id}:*` created
2. All memory writes tagged `(session_id, session_status='draft')`
3. Interactive sessions read only `session_status = committed` — never see drafts from parallel sessions
4. Session completes → DeptHead evaluation + SDD validation
5. If approved: `UPDATE nodes SET session_status = 'committed' WHERE session_id = ?` (batch commit)
6. If rejected: draft nodes purged, session marked `failed`

**Conflict resolution:** If two autonomous sessions modify related artifacts, DeptHead reviews the conflict at merge time (like a git merge review). The second commit waits for the first to be evaluated.

**Context poisoning prevention:** A developer session writing code and an interactive DeptHead chat session can run simultaneously. The DeptHead sees only committed state — the in-progress code is invisible until it's evaluated and committed.

#### 14.3 — Session Concurrency Model

```
FloorManager Session Registry
├── session:abc123 (autonomous, Development, status=active)
│   └── AlphaForge loop — developing indicator
├── session:def456 (interactive, Development, status=active)
│   └── DeptHead direct chat — discussing strategy
└── session:ghi789 (autonomous, Risk, status=waiting_human)
    └── Drawdown analysis — awaiting human approval gate
```

**Cross-session communication:** Only via committed memory state. Session A cannot call Session B directly. If Session A needs Session B's output, it must wait for B to commit, then read from committed namespace.

**Sub-agent visibility:** Sub-agents are the "non-interactive" view in the Agent Panel. Each active sub-agent within a session is visible as a status indicator (running / idle / blocked).

**UI indicator:** Active session badge (count bubble) on session history/plus icon in ActivityBar. Each session shows: type (interactive/autonomous), owner dept, status.

#### 14.4 — Session State Machine

```
created → active → (waiting_human) → complete → [commit or reject]
                                   → failed
```

**`sessions` table:** See §6.1 schema additions.

**Redis real-time state:** `session:{session_id}:status` — updated in real time for UI badge.
**SQLite persistent record:** `sessions` table — permanent audit trail of all sessions.

---

### 15. Living Skills System

#### 15.1 — Skills vs Tools vs Directives (Definitions)

| Concept | What it is | Format | Where it lives |
|---|---|---|---|
| **Skill** | SOP — tells an agent HOW to think about and approach a task | `skill.md` + optional `executor.py` | `shared_assets/skills/` |
| **Tool** | Callable function — what an agent can DO | Python function registered with Agent SDK | `src/agents/tools/` (see §16) |
| **Directive** | Workflow-scoped SOP — used only within one Prefect flow | `directive.md` | `flows/directives/{workflow}/` |
| **System Prompt** | Agent identity + indexed skills + tools list | `system_prompt.md` | `src/agents/departments/{dept}/` |

Skills reference tools. Tools execute actions. Directives orchestrate workflows. System prompts wire them together.

#### 15.2 — Skill Hierarchy (Dept Toolbox Model)

Each department is a toolbox. Any sub-agent spawned within that department can access the department's skill toolbox. Sub-agents are not fixed in type — they can be spawned dynamically. The toolbox is available to all of them.

```
shared_assets/skills/
├── templates/                       ← canonical templates (see §15.4)
│   ├── skill-with-autonomy.md
│   ├── skill-no-autonomy.md
│   └── system-prompt-template.md
├── global/                          ← any dept, any session type
│   ├── context-enrich/
│   │   └── skill.md
│   └── morning-digest/
│       └── skill.md
└── departments/
    ├── development/
    │   ├── forge-strategy/
    │   │   ├── skill.md
    │   │   └── executor.py          ← optional: deterministic execution layer
    │   ├── review-ea/
    │   │   └── skill.md
    │   ├── sdd-spec/
    │   │   └── skill.md
    │   ├── validate-sdd/
    │   │   └── skill.md
    │   └── update-cloudzy/
    │       └── skill.md
    ├── research/
    ├── trading/
    ├── risk/
    │   └── drawdown-review/
    │       └── skill.md
    └── portfolio/
        ├── system-targets/
        │   └── skill.md
        └── register-broker/
            └── skill.md
```

**Workflow-specific skills** (used only within one flow) live in `flows/directives/{workflow}/` as directives — NOT in `shared_assets/skills/`. They are scoped to that workflow and are not reusable across departments.

#### 15.3 — Skill Metadata Schema

```yaml
# Frontmatter in skill.md
id: forge-strategy
version: "1.0.0"
status: active              # draft | active | deprecated
tier: department            # global | department
owned_by: development
slash_command: /forge-strategy
session_types:
  - autonomous              # interactive | autonomous | both
risk_class: medium          # low | medium | high
has_executor: false         # true if executor.py exists
```

**`risk_class` controls autonomous self-improvement eligibility:**
- `low` — Reflector may self-improve automatically
- `medium` — Reflector proposes; FloorManager approves version bump
- `high` — human-only changes (risk management SOPs, prop firm compliance, live trading controls)

#### 15.4 — Skill Templates (Canonical Templates — Written in Codebase)

Two canonical templates written as actual files in the codebase, indexed here:

**`shared_assets/skills/templates/skill-with-autonomy.md`** — template for `risk_class: low | medium` skills. Includes:
- Full frontmatter schema with all fields
- Directive body structure (goal, when-to-use, steps, success-criteria, guardrails)
- `executor.py` interface contract (inputs, outputs, error handling)
- Trace logging hooks (what to log to `traces.jsonl` after each execution)
- Autoresearch integration note: how batch pre-deployment testing works
- Reflector integration note: what traces are mined and how improvement proposals are structured

**`shared_assets/skills/templates/skill-no-autonomy.md`** — template for `risk_class: high` skills. Includes:
- Same structure as above, minus autoresearch/reflector sections
- Human approval gate documentation requirement
- Mandatory human review before any version bump

**`shared_assets/skills/templates/system-prompt-template.md`** — canonical system prompt structure for Department Heads. Includes:
- Identity block (role, dept, tier, responsibilities)
- Skill index section (indexed by task type, not embedded)
- System components available section (JIT-loaded file paths)
- Tools section (injected from agent identity at session start)
- Communication rules (mail, Redis, no cross-dept spawning)

All three templates are the authoritative standard. When Development Dept or Claude Code writes a new skill or system prompt, they load the relevant template first.

#### 15.5 — Skill Lifecycle

```
[Claude Code writes draft] → status: draft
    ↓
[Autoresearch batch test] → N variants, keep best (risk_class: low/medium only)
    ↓
[status: active] → used in sessions, traces.jsonl accumulated
    ↓
[ReflectionExecutor fires on session idle/complete]
    → mines traces, extracts failure patterns + improvement opportunities
    → proposes patch → Kanban LOW task created
    ↓
[Development Dept executes refinement] → new version x.x.x+1
    → risk_class: low → auto-promoted
    → risk_class: medium → FloorManager approves version bump
    → risk_class: high → human approves before any change
    ↓
[status: deprecated] → when new version supersedes or skill removed
```

**Autoresearch pattern** (ref: [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)) — applied to pre-deployment skill validation:
- Generate N skill variants (prompt mutations)
- Evaluate each against defined criteria (task completion rate, token efficiency, error rate)
- Keep best variant → mark `status: active`

**ACE pattern** (ref: [github.com/kayba-ai/agentic-context-engine](https://github.com/kayba-ai/agentic-context-engine)) — applied to continuous in-production improvement:
- Agent executes task using skill → execution trace logged
- ReflectionExecutor mines traces (even on successes, not just failures)
- SkillManager (FloorManager) curates: refine active, deprecate failing

#### 15.6 — Skill Reflector (ReflectionExecutor Pattern)

**Trigger:** Session state transition to `complete` or `waiting_human` (idle time, not interrupting active work).

**Implementation:** `src/memory/graph/reflection_executor.py` — async, debounced 5 minutes. Cancels if new session activity arrives.

**What it mines:** `shared_assets/skills/{tier}/{skill-id}/traces.jsonl`
- Which steps were skipped or re-attempted
- Which tool calls failed
- Where the agent deviated from the SOP
- Whether task was completed vs abandoned

**Output:** Structured improvement proposal → Kanban LOW-priority task in FloorManager's queue.

**Self-Annealing vs Reflector (distinct):**

| | Self-Annealing | Reflector |
|---|---|---|
| Trigger | FAILURE (exception thrown) | OUTCOME (session complete, quality signal) |
| Target | `executor.py` (broken code) | `skill.md` SOP (suboptimal instructions) |
| Response | Immediate patch + PR | Kanban task, scheduled improvement |
| Philosophy | Reactive recovery | Proactive quality improvement |

Both are needed. They are complementary.

#### 15.7 — Skill Creation During Development

Skills are written by **Claude Code** during development sprints. The Anthropic skill-creator tool ([github.com/elinasto/anthropic-skill-creator](https://github.com/elinasto/anthropic-skill-creator)) was evaluated but not adopted — our architecture is custom and the templates above are the canonical standard.

Agents that are authorized to PROPOSE skill updates (via Kanban task) during runtime: FloorManager (SkillManager role), Development Dept Head. No agent can modify `status: active` skills directly — only via the Kanban + review cycle.

---

### 16. Tools Registry

#### 16.1 — Skills ≠ Tools

Skills tell agents WHAT to do and HOW to think. Tools are what agents can actually DO — callable functions registered with the Anthropic Agent SDK. A skill may reference one or more tools. If a referenced tool is not activated in the agent's session, the agent calls `request_tool()`.

#### 16.2 — Tool Registry Schema

```sql
CREATE TABLE IF NOT EXISTS tool_registry (
    tool_id          TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    description      TEXT NOT NULL,
    category         TEXT NOT NULL,  -- global | trading | research | development | risk | portfolio | sandbox
    dept_scope       TEXT,           -- JSON array of depts, NULL = global
    requires_approval BOOLEAN DEFAULT FALSE,
    safe_for_autonomous BOOLEAN DEFAULT TRUE,
    implementation_path TEXT,        -- e.g. src/agents/tools/mt5_bridge_tools.py
    version          TEXT DEFAULT '1.0.0',
    status           TEXT DEFAULT 'active'  -- active | deprecated
);
```

#### 16.3 — Tool Tiers

| Tier | Available to | Examples |
|---|---|---|
| **Global** | All agents, all sessions | `read_skill`, `write_memory`, `search_memory`, `read_canvas_context`, `request_tool`, `send_department_mail` |
| **Department** | Agents in that dept only | `mt5_bridge.read_positions` (Trading), `backtest.run` (Development), `risk_calculator.kelly_size` (Risk) |
| **Workflow-context** | Only when inside a specific Prefect flow | `alpha_forge.compile_ea`, `alpha_forge.submit_to_review` |
| **Sandbox-only** | Autonomous sessions in Docker container | `container.exec_python`, `container.run_mql5_compiler` |

#### 16.4 — Tool Activation at Session Start

1. Agent identity loaded → dept-scoped tools resolved
2. Session context loaded (Canvas template or flow directive) → `required_tools: [tool_ids]` parsed
3. Required tools checked against `tool_registry`:
   - Tool exists + in agent's dept scope → **activated** for this session
   - Tool exists + NOT in agent's scope + `requires_approval: false` → **auto-request** to FloorManager (fast path)
   - Tool exists + NOT in agent's scope + `requires_approval: true` → **Kanban task** created, session proceeds without tool (graceful degradation)
   - Tool does NOT exist in registry → **Kanban task** created for Development Dept: "Build tool X"
4. Final activated tool list injected into agent context at session start

#### 16.5 — Tool Creation Policy

**Two-tier creation:**

| Tier | Who creates | Where it runs | Approval |
|---|---|---|---|
| **Sandbox tool** | Agent (proposes code, executed in Docker container) | Windows Docker container — isolated from main system | Auto-approved for container scope |
| **Production tool** | Development Dept only | Main system (`src/agents/tools/`) | Sandbox test + PR + human approval before `tool_registry` insertion |

Agents cannot autonomously create production tools. They can REQUEST tool creation via `request_tool(tool_id, reason)`. Development Dept builds + validates + registers. This boundary protects live trading systems from agent-authored code that could accidentally interact with MT5.

#### 16.6 — `request_tool()` — Universal Tool Request

Any agent in any session can call:
```python
request_tool(
    tool_id: str,       # ID of existing tool or proposed new tool name
    reason: str,        # Why this session needs it
    urgency: str,       # HIGH | MEDIUM | LOW
    session_id: str     # current session
)
```

**If `tool_id` exists in registry:**
- `requires_approval: false` → FloorManager auto-activates for this session
- `requires_approval: true` → Kanban task + session proceeds with graceful degradation

**If `tool_id` does NOT exist:**
- Kanban task: Development Dept → build, sandbox test, register
- Current session proceeds without it; next session can activate once built

---

### 17. Docker Agent Toolbox (Windows Container)

#### 17.1 — Purpose

A sandboxed Windows Docker container that agents can access for CLI tool execution, isolated from the main system. Agents interact via a local API or exec interface — they never have direct shell access to the main backend.

**Why Windows:** MT5, MQL5 compiler, and Wine-dependent tools require Windows or Wine-compatible runtime. Consolidating these in a container isolates dependencies and simplifies deployment on Cloudzy.

#### 17.2 — Container Contents

| Tool | Purpose | Status |
|---|---|---|
| MQL5 compiler | Compile `.mq5` → `.ex5` EA files | Core — required |
| Wine runtime | Run Windows-native tools on Linux host if needed | Optional |
| CLI utilities | File manipulation, git operations for agent workflows | Phase 2 |
| NotebookLM CLI / Google auth | Video ingest authentication (see §17.3) | Flagged risk |
| Python sandbox | Agent-authored sandbox tools (safe tier, see §16.5) | Phase 2 |

#### 17.3 — Video Ingest Authentication Risk

**Problem:** NotebookLM CLI uses Google OAuth — requires browser interaction for initial auth. Headless containers cannot complete browser-based OAuth flows.

**Mitigation options (resolve in implementation story):**
1. **Token pre-refresh:** Authenticate once manually on the container's browser session; store long-lived refresh token in `.env` / secrets manager. Container auto-refreshes access tokens without re-auth. Works for most Google APIs.
2. **yt-dlp + Anthropic transcription:** Download video via `yt-dlp` (API-key-free, no OAuth) → transcribe via Anthropic API → bypass NotebookLM entirely. Simpler and cheaper.
3. **YouTube Data API v3:** If metadata + transcript only needed, YouTube API uses API key (no OAuth).

**Architectural risk flag:** Video ingest auth via container is UNRESOLVED until implementation story. Do not begin implementation of video ingest container tooling until the auth path is confirmed by Mubarak.

#### 17.4 — Deferred to Phase 2

The Docker Agent Toolbox is a Phase 2 component. Phase 1 uses the existing MQL5 compiler on Cloudzy directly. Container wrapping is added when agent-authored sandbox tools are needed.

---

### 18. Autoresearch, ACE & Self-Optimization References

The following external projects informed the Living Skills System (§15) design. Patterns are adopted conceptually — no library imports from any of these.

| Project | Reference | Adopted Pattern |
|---|---|---|
| **Autoresearch** (Karpathy) | [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Batch generate → evaluate → score → mutate → keep best. Applied to pre-deployment skill validation. |
| **ACE** (kayba-ai) | [github.com/kayba-ai/agentic-context-engine](https://github.com/kayba-ai/agentic-context-engine) | Agent + Reflector + SkillManager = living Skillbook. 20-35% task improvement, 49% token reduction demonstrated. Applied as ReflectionExecutor + FloorManager-as-SkillManager pattern. |
| **LangMem** | [langchain-ai.github.io/langmem](https://langchain-ai.github.io/langmem/) | Memory types (episodic/semantic/profile/short-term), namespace isolation, delayed processing, prompt optimization. **LangMem/LangChain import BANNED** — patterns implemented natively. |
| **Anthropic Context Engineering** | [anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | JIT loading, minimum high-signal tokens, indexed system prompts. Applied throughout agent identity + CanvasContextTemplate design. |
| **Ontologies with LLMs** | [enterprise-knowledge.com/the-role-of-ontologies-with-llms](https://enterprise-knowledge.com/the-role-of-ontologies-with-llms/) | Ontology as type system (schema/data separation), graph as primary RAG source, explicit edge reasoning vs. statistical inference, decision traceability via SUPPORTED_BY edges. Applied in §6.1 ontology positioning, §6.4 OPINION node pattern, §19. |

---

### 19. Knowledge Graph Ontology — Architectural Positioning

#### 19.1 — The QUANTMINDX Domain Ontology

The graph memory system is not merely a persistence layer. It implements a **domain ontology** — a typed, relational model of the QUANTMINDX knowledge domain. Two distinct layers:

| Layer | File | Role |
|---|---|---|
| **Ontology** (schema) | `src/memory/graph/types.py` | Defines entity types, relationship types, and valid connections. This is the vocabulary of the domain. |
| **Knowledge graph** (data) | `src/memory/graph/store.py` | Instantiates the ontology with live data — actual nodes, edges, and properties written by agents and data pipelines. |

The 14 `MemoryNodeType` values and 18 `RelationshipType` values in `types.py` constitute the **QUANTMINDX trading domain vocabulary**. Every agent, workflow, and skill interacts with this typed system — not with free-text storage.

**Why this matters architecturally:**
- Agents reason over *explicit typed relationships* — not statistical associations
- Downstream agents know what kind of thing they are reading (an OPINION vs. an OBSERVATION vs. a WORLD fact)
- The ontology is stable — node types require a PR to `types.py` before they can be used anywhere
- Cross-department queries use the shared type vocabulary — no department-local aliases

---

#### 19.2 — Graph as Primary Knowledge Retrieval Layer

Three retrieval systems coexist. Each has a distinct role:

| System | Source | Query method | Purpose |
|---|---|---|---|
| **Graph (this system)** | SQLite knowledge graph | Typed traversal + semantic search (`recall()`) | Agent episodic memory, decisions, market observations, inter-agent reasoning |
| **ChromaDB** | External research documents, strategy papers | Vector similarity | Deep document retrieval — research, academic papers, strategy references |
| **CAG** | CanvasContextTemplate | Pre-assembled at session start | Stable session identifiers, skill index, tool list — not data |

The graph is the **primary retrieval layer for all agent reasoning context**. ChromaDB handles external research. CAG handles session scaffolding. These do not overlap.

At context enrichment time (session start), agents query the graph first — not ChromaDB, not the canvas. The graph contains what prior agents knew and decided. That is the most relevant context for continuing any piece of work.

---

#### 19.3 — Edge Traceability as Architectural Obligation

Explicit edges between nodes produce queryable reasoning chains. This is the mechanism by which the system avoids losing context across agent sessions and workflow steps.

**The canonical decision chain:**

```
DECISION ──── SUPPORTED_BY ──►  OBSERVATION  (what was observed)
         ──── SUPPORTED_BY ──►  WORLD        (market fact / price / calendar event)
         ──── INFORMED_BY  ──►  OPINION       (prior agent's assessment)
         ──── INFORMED_BY  ──►  PROCEDURAL    (the skill/rule that was followed)

OPINION  ──── SUPPORTED_BY ──►  OBSERVATION  (mandatory — see §6.4)
         ──── SUPPORTED_BY ──►  WORLD        (at least one required)
```

**Query pattern for downstream agents:**

```python
# "Why was strategy X deferred?"
decisions = graph.traverse(
    start_node="decision_node_id",
    edge_types=[RelationshipType.SUPPORTED_BY, RelationshipType.INFORMED_BY],
    depth=2
)
# Returns: the OPINION nodes that informed the decision,
#          the OBSERVATION nodes that grounded the opinions,
#          the WORLD nodes (calendar, price) at time of decision
```

**Implementation rule:** When agents write DECISION or OPINION nodes, they must immediately write the edge(s) in the same operation. Never write a node and defer the edges — a node without edges is orphaned and cannot be traversed.

---

### 20. Alpha Forge — Two-Workflow Architecture

_Alpha Forge is the autonomous strategy factory described in PRD §3 (FR23–FR31) and user journeys 2, 4, 9, 13, 19, 44. It is now architecturally split into two independent workflows. The PRD journeys remain valid — the user experience is unchanged. The split is a backend workflow boundary, not a UX change._

#### 20.1 — Alpha Forge = Two Independent Workflows

| | Workflow 1 — Creation | Workflow 2 — Enhancement Loop |
|---|---|---|
| **Trigger** | Manual: Mubarak provides video URL or document source | Automatic: runs continuously, independent of Workflow 1 |
| **Input** | Video / document source | EA Library — any strategy in `draft`, `testing`, or `archived` status |
| **Goal** | Turn a source into a tested EA variant | Make as many strategies as possible, test as many as possible, improve as many as possible |
| **Output** | New strategy + vanilla/spiced variants deposited to EA Library | Improved EA versions promoted through paper trading to live |
| **Relation** | Creates the raw material | Consumes and continuously improves the raw material |

They share the EA Library and the backtest engine. They do not share a trigger, a loop, or a completion state.

---

#### 20.2 — Workflow 1: Creation Pipeline

```
[Source: YouTube video URL / PDF / article]
        ↓
[Research Dept] → transcribe + extract strategy hypothesis → TRD
        ↓
[Development Dept] → SDD spec from TRD → MQL5 EA code
        ↓
[Development Dept] → compile (.mq5 → .ex5)
        ↓
[Backtest engine] → VANILLA basic + SPICED basic only
        ↓  (sanity check — "can this EA perform at all?")
[EA Library] → strategy_id created, vanilla + spiced variants deposited
               status: draft
               improvement_cycle: 0
WORKFLOW 1 ENDS HERE
```

**What Workflow 1 does NOT do:** Full Monte Carlo, Walk Forward, PBO, MODE_B, MODE_C, data variant testing, paper trading, improvement iterations. All of that is Workflow 2.

---

#### 20.3 — Workflow 2: Enhancement Loop (Continuous)

Workflow 2 runs against the EA Library. It is not triggered by Workflow 1 — it is always running, picking up strategies in `draft` or `testing` status. The loop does not stop. Its goal is consistent profitability across diverse conditions, not hitting an arbitrary metric.

```
[EA Library] → pick strategy in draft / testing / archived status
        ↓
──── ROUND 1: Full backtest validation ────────────────────────────────
[Backtest engine] → run full matrix:
    VANILLA           × basic, monte_carlo, pbo
    SPICED            × basic, monte_carlo
    VANILLA_FULL      × basic (Walk Forward)
    SPICED_FULL       × basic (Walk Forward)
    MODE_B            × basic, monte_carlo
    MODE_C (SIT)      × basic, monte_carlo, walk_forward
[ea_backtest_reports] → all results persisted
[OPINION node] → agent writes assessment: which variants hold up, which fail
        ↓
──── ROUND 2: Research-driven improvement ─────────────────────────────
[Research Dept] → web search + shared_assets docs + PDFs + articles
               + reads OPINION nodes from prior cycles (graph memory)
               → identifies specific improvement opportunities
[Development Dept] → generates improvement variants targeting identified gaps
[Backtest engine] → full matrix on new variants
[OPINION node] → agent writes assessment of improvement delta
        ↓ (loop if improvement found, move forward if passing)
──── ROUND 3+: Dukascopy data variant stress testing ──────────────────
[Dukascopy] → distill into data variants:
    - different time windows (2018–2020, 2020–2022, 2022–2024)
    - market regime slices (trending, ranging, crisis/event periods)
    - instrument splits (per currency pair tested separately)
[Backtest engine] → run improved variants against each data variant
Anti-overfitting guard: variant must pass MC + WF across multiple data slices, not just one
[ea_backtest_reports] → data_variant column captures which slice was used
[OPINION node] → agent writes cross-data robustness assessment
        ↓
──── SIT GATE ─────────────────────────────────────────────────────────
MODE_C (SIT) must pass: EA + Strategy Router + Kelly + Regime Filter together
Only variants passing SIT proceed to paper trading (per PRD FR27, FR28)
Failed SIT → Development receives specific integration failure report → loop back to Round 2
        ↓
──── PAPER TRADING ────────────────────────────────────────────────────
[Trading Dept] → deploy variant to MT5 demo account
                 status: paper_trading
                 monitor: win rate, avg R, max drawdown, Sharpe, regime breakdown, spread sensitivity
                 minimum run: configurable (default 7 days)
                 parallel races supported (PRD Journey 44): multiple variants compete simultaneously
        ↓
[Trading Dept] → generate paper trade dossier (PRD Journey 19):
    win rate, avg R, max drawdown, Sharpe, regime-specific breakdown,
    spread sensitivity log, correlation to live portfolio, Risk Dept sign-off,
    Development Dept notes
    status: paper_validated
        ↓
──── HUMAN APPROVAL GATE ──────────────────────────────────────────────
[Copilot] → surfaces dossier to Mubarak
Mubarak reviews equity curve + dossier → approves or rejects
approval: status → active → deploy to live via Strategy Router
rejection: status → archived (with reason) → loop may retry with different variant
```

**Loop driver:** The loop runs continuously. There is no "done" state — there is always a next strategy to test or improve. The system maintains 25–50 strategies evolving simultaneously (PRD §1 product vision).

---

#### 20.4 — EA Variant Promotion Lifecycle

```
draft           ← Workflow 1 deposits here
testing         ← Workflow 2 Round 1 begins
sit_passed      ← passed MODE_C SIT validation
paper_trading   ← deployed to MT5 demo
paper_validated ← passed paper trading criteria (dossier generated)
pending_approval← awaiting Mubarak review
active          ← deployed to live via Strategy Router
archived        ← superseded by newer version OR rejected at approval gate (retrievable)
retired         ← manually retired by Mubarak (permanent)
```

`archived` ≠ `retired`. Archived strategies can be re-entered into Workflow 2 (e.g., "Bring EA_AUDUSD_Range_V1 out of retirement" — PRD Journey 28). Retired strategies are permanently deactivated.

---

#### 20.5 — EA Storage Structure

**Filesystem root:** `data/ea-library/{strategy_id}/`

```
{strategy_id}/
├── meta.md                        ← source URL, strategy summary, creation date
├── trd.md                         ← Technical Requirements Document (Research Dept output)
├── sdd.md                         ← SDD spec (Development Dept output)
├── versions/
│   ├── v1.0.0/                    ← Workflow 1 produces this
│   │   ├── vanilla/
│   │   │   ├── ea.mq5
│   │   │   ├── ea.ex5
│   │   │   └── reports/
│   │   │       └── vanilla_basic.json     ← Workflow 1 sanity check only
│   │   ├── spiced/
│   │   │   ├── ea.mq5
│   │   │   ├── ea.ex5
│   │   │   └── reports/
│   │   │       └── spiced_basic.json
│   │   └── reports/               ← Workflow 2 fills these in
│   │       ├── vanilla_mc.json
│   │       ├── vanilla_full.json
│   │       ├── spiced_full.json
│   │       ├── mode_b_basic.json
│   │       ├── mode_b_mc.json
│   │       ├── mode_c_basic.json
│   │       ├── mode_c_mc.json
│   │       ├── mode_c_wf.json
│   │       └── pbo.json
│   └── v1.1.0/                    ← Workflow 2 improvement cycle output
│       └── ...                    ← same structure
└── active -> versions/v1.1.0/     ← symlink to currently deployed version
```

**EA tagging system (extend existing — do not rebuild):**

The existing `EARegistry` (`src/router/ea_registry.py`) and `BotManifest` (`src/database/models/bots.py`) constitute the EA tagging system. It is retained and extended:

| Tag tier | Source | Examples |
|---|---|---|
| Mode tags (auto) | `EARegistry` | `@demo`, `@live` |
| Prop firm tags | `BotManifest` | `@fundednext`, `@challenge_active`, `@funded` |
| Lifecycle tags (new) | Alpha Forge workflow | `@paper_trading`, `@sit_passed`, `@pending_approval` |
| Custom tags | Mubarak or agents | `@session_london`, `@low_drawdown`, `@audusd_only` |

`BotLifecycleLog` already tracks all tag transitions as an audit trail — this is the EA provenance record (FR31). No rebuild needed.

**DB tables (new — see DB inventory in §5):**
- `ea_backtest_reports` — one row per report file, queryable by mode/type/data_variant
- `ea_improvement_cycles` — lightweight cycle tracker linking rounds to variant outputs
- `ea_versions` extended — add `strategy_id`, `variant_type`, `improvement_cycle` columns

---

### 12. Deferred Decisions

| Decision | Reason Deferred |
|---|---|
| Canvas navigation routing (3.1) | UX design phase — run `/bmad-bmm-create-ux-design` first |
| Department Operations Board exact UI layout | UX design phase |
| Provider fallback chain order | Minor — decide in implementation story |
| AlgoForge pull UI flow details | Minor — decide in implementation story |
| Pine Script | Permanently removed. MT5 only across all phases. |

---

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**
All technology choices are mutually compatible. Anthropic Agent SDK is the sole agent runtime with LangGraph/LangChain explicitly forbidden. Svelte 5 runes + static adapter is consistent with Tauri desktop deployment. SQLite WAL mode is used consistently across all persistence layers. Redis Streams cleanly replaces the deprecated SQLite DepartmentMailService with no parallel path remaining. Prefect (runtime) and DOE (methodology) are correctly separated. sentence-transformers all-MiniLM-L6-v2 unifies the embedding infrastructure with ChromaDB. NODE_ROLE provides clean node placement with no ambiguity. Pydantic v2 is consistently enforced throughout.

**Pattern Consistency:**
Naming conventions are consistent across all five surfaces (DB, API, Python, Svelte/TypeScript, Redis). Memory namespace tuple pattern is applied consistently across §6.1, §6.2, §14, and §15. The `session_status: draft → committed` lifecycle is coherently applied to memory nodes, skills, and system prompts. DOE layers are consistently mapped to their implementation files. OPINION node obligation is defined in §6.4 and cross-referenced in Process Patterns and §19.3.

**Structure Alignment:**
Project structure directly mirrors the department hierarchy and DOE layers. EA Library folder structure uses the real backtest mode names from the existing engine. Graph memory ontology layer (`types.py`) and instantiation layer (`store.py`) are cleanly separated. All integration points (ZMQ, WebSocket, SSE, Redis) have explicit node ownership.

---

### Requirements Coverage Validation ✅

**Functional Requirements Coverage:**
All 79 FRs across 9 domains are architecturally supported.

| Domain | FRs | Coverage |
|---|---|---|
| Live Trading & Execution | FR1–FR9 | MT5 ZMQ preserved, WebSocket direct channel, two independent kill switches, Islamic force-close at Commander level + every EA template |
| Autonomous Agent System | FR10–FR22 | 3-tier hierarchy, Redis department mail, skill registry, session workspace isolation, tool registry + activation |
| Alpha Forge | FR23–FR31 | Two-workflow architecture, 6-mode backtest suite, SIT gate, paper trading dossier, 8-state promotion lifecycle, EA tagging + provenance chain |
| Risk Management | FR32–FR41 | Physics engine untouched, CalendarGovernor local cache on Cloudzy, prop firm configurable registry |
| Knowledge & Research | FR42–FR50 | Graph ontology + ChromaDB + CAG three-layer retrieval, news provider abstraction, OPINION node provenance |
| Portfolio & Multi-Broker | FR51–FR58 | Routing matrix untouched, broker registry extended with commission/spread/timezone |
| Monitoring & Audit | FR59–FR65 | Immutable audit_log, 3-year trade record retention, natural language query via DuckDB |
| System Management | FR66–FR73 | NODE_ROLE runtime config, provider swap, Tauri portability |
| Traceability Gap Closures | FR74–FR79 | A/B via HMM stages, rollback via ea_versions + BotLifecycleLog, OPINION node provenance chain |

**Non-Functional Requirements Coverage:**

| NFR | Architectural response |
|---|---|
| ≤3s P&L lag | WebSocket direct from Cloudzy — not routed through Contabo |
| ≤5s Copilot first token | SSE streaming from Contabo with Anthropic streaming API |
| Canvas transitions ≤200ms | Svelte 5 static adapter — no SSR hydration penalty |
| Backtest matrix ≤4h | Contabo as heavy compute, Prefect parallelism |
| SSH key-only + IP firewall | NODE_ROLE security model, all secrets in `.env`, forbidden patterns enforce no hardcoded credentials |
| ZMQ reconnect ≤10s | Continuous retry policy until reconnected |
| Cloudzy independent | All trading-critical components local; CalendarGovernor local 7-day cache |
| Trade records before ACK | `trade_records` written before acknowledgment (§DB inventory) |
| 500-line file limit | Consistency rule — enforced at code review |

---

### Implementation Readiness Validation ✅

**Decision Completeness:**
All critical decisions are documented with technology versions and rationale. The forbidden patterns table prevents the highest-impact implementation conflicts upfront. Code examples are provided for all major patterns across Python, Svelte, and SQL. DB table names are locked — agents cannot invent variations.

**Structure Completeness:**
Full directory tree with `[exists]`/`[extend]`/`[new]`/`[refactor]` labels. FR→file mapping provided. Rebuild scope is explicit: stays untouched, stays but extends, rebuilt from scratch, legacy read-only. EA Library folder structure uses confirmed real backtest mode names.

**Pattern Completeness:**
All five naming surfaces covered. Error handling with correct/wrong code examples. Retry policies per operation type. Communication patterns with who-can-send-to-whom permission matrix. OPINION node obligation with minimum valid code. Memory isolation lifecycle documented with enforcement points at skills, system prompts, and code review.

---

### Gap Analysis Results

**Critical Gaps:** None identified. No missing decisions that would block implementation.

**Important Gaps (story-level decisions — not architectural blockers):**

1. **Workflow 2 Prefect scheduling mechanism (§20)** — The Alpha Forge Enhancement Loop is specified as continuously running but the Prefect scheduling pattern is unspecified (long-running flow with internal loop vs. scheduled flow vs. event-driven on EA Library update). Flag for the `flows/assembled/alpha_forge_enhancement_loop.py` implementation story — confirm with Mubarak before implementing.

2. **Paper trading pass criteria (§20.3)** — The paper trading gate is structurally defined but evaluation criteria are "configurable" with no default benchmark. Suggested baseline for the implementing story to confirm: minimum 7 days, win rate ≥ 50%, max drawdown ≤ 2%, profit factor ≥ 1.2. Trading Dept implementation story must confirm thresholds before coding the paper trade evaluation logic.

**Nice-to-Have Gaps:**
- Global skill stubs (`context-enrich`, `morning-digest`) not yet authored — expected during implementation sprints, templates are ready.
- Redis Stream retention policy not specified — implementation story confirms default.
- `sessions` table cleanup policy not defined — defer until session volume is understood in production.

---

### Validation Issues Addressed

No critical issues requiring resolution before implementation. The two important gaps are flagged as story-level decisions with clear owners and are not architectural blockers. All architectural decisions are coherent, complete, and consistent across all 20 sections.

---

### Architecture Completeness Checklist

**✅ Requirements Analysis**
- [x] Project context thoroughly analysed — 79 FRs, 52 user journeys, 9 capability domains
- [x] Scale and complexity assessed — enterprise-level, ~15 major bounded contexts
- [x] Technical constraints identified — Anthropic SDK, Svelte 5, ZMQ, Islamic compliance, prop firm rules
- [x] Cross-cutting concerns mapped — UTC timestamps, Pydantic v2, 500-line limit, forbidden patterns

**✅ Architectural Decisions**
- [x] Critical decisions documented with technology versions and rationale
- [x] Technology stack fully specified with forbidden alternatives named
- [x] Integration patterns defined — ZMQ, WebSocket, SSE, Redis, Department Mail
- [x] Performance considerations addressed via node placement decisions

**✅ Implementation Patterns**
- [x] Naming conventions established — DB, API, Python, Svelte/TypeScript, Redis
- [x] Structure patterns defined — backend, frontend, flows, skills, EA Library
- [x] Communication patterns specified — mail format, send permissions matrix, internal vs cross-dept streams
- [x] Process patterns documented — error handling, retry policies, node placement rule, OPINION obligation

**✅ Project Structure**
- [x] Complete directory structure defined with [exists]/[extend]/[new]/[refactor] labels
- [x] Component boundaries established — 5 departments, global vs department skills, flow components
- [x] Integration points mapped — ZMQ (Cloudzy), WebSocket (Cloudzy), SSE (Contabo), Redis (Contabo)
- [x] Requirements to structure mapping complete — FR→file mapping provided

**✅ Extended Architecture (this session)**
- [x] Session Architecture (§14) — two session types, workspace isolation, concurrency model
- [x] Living Skills System (§15) — hierarchy, lifecycle, Reflector, autoresearch pattern, canonical templates
- [x] Tools Registry (§16) — skills≠tools distinction, tool_registry table, activation sequence, request_tool()
- [x] Docker Agent Toolbox (§17) — Phase 2, video auth risk flagged with three mitigation paths
- [x] OPINION Node Pattern (§6.4) — mandatory reasoning artifact, consequential action definition, edge obligation, downstream query pattern
- [x] Knowledge Graph Ontology (§19) — types.py as ontology layer, graph as primary retrieval, edge traceability obligation
- [x] Alpha Forge Two-Workflow Architecture (§20) — Workflow 1 + 2 separated, full 6-mode backtest suite, SIT gate, paper trading gate, 8-state promotion lifecycle, EA storage structure

---

### Architecture Readiness Assessment

**Overall Status: READY FOR IMPLEMENTATION**

**Confidence Level: High**

The architecture is unusually thorough for a brownfield project. Every major decision is locked with explicit rationale, every forbidden pattern is named, and every implementation surface has concrete examples. The document is self-consistent across all 20 sections and 2900+ lines.

**Key Strengths:**
- Domain ontology approach (§19) gives the agent memory system an explicit typed knowledge structure — agents reason explicitly over typed relationships rather than statistically
- OPINION node obligation (§6.4) makes agent reasoning persistent and cross-session — the system accumulates institutional knowledge over time
- Alpha Forge two-workflow separation (§20) gives implementation teams a clean boundary — Workflow 1 is a simple creation pipeline, Workflow 2 is the continuous improvement engine
- Forbidden patterns table is decisive — prevents the most expensive implementation mistakes upfront
- NODE_ROLE split keeps Cloudzy minimal and autonomous; Contabo handles all AI/compute

**Areas for Future Enhancement (Phase 2):**
- Docker Agent Toolbox — Windows container for MQL5 compiler (video auth path to confirm)
- JWT authentication — admin/viewer/client roles
- Smart Money tracking — institutional flow analysis
- Paper trading pass criteria — confirm defaults in first Trading Dept implementation story
- Workflow 2 Prefect scheduling — decide at flow implementation time

---

### Implementation Handoff

**AI Agent Guidelines:**
- Follow all architectural decisions exactly as documented — this is the constraint set, not a reference
- Forbidden patterns are hard stops — any code violating them fails code review
- OPINION node writes are mandatory after consequential actions — not optional
- Use `types.py` as the canonical domain vocabulary — no new node types without a PR to `types.py`
- All new Svelte components use Svelte 5 runes — no Svelte 4 syntax in new code
- All timestamps use `_utc` suffix and `datetime.now(timezone.utc)` — no exceptions
- `apiFetch<T>()` is the only permitted API call method in Svelte — never raw `fetch()`

**First Implementation Priority:**
Run `/bmad-bmm-create-epics-and-stories` to break the architecture into implementable epics and user stories. Recommended starting point: graph memory extensions (§6.1) + department agent system (§4) as the foundational layer everything else builds on.
