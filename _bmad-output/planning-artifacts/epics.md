---
stepsCompleted: ['step-01-validate-prerequisites', 'step-02-design-epics', 'step-03-create-stories', 'step-04-final-validation']
status: complete
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/planning-artifacts/architecture.md'
  - '_bmad-output/planning-artifacts/ux-design-specification.md'
  - '_bmad-output/planning-artifacts/prd-validation-report.md'
---

# QUANTMINDX - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for QUANTMINDX, decomposing the requirements from the PRD, UX Design, Architecture, and Validation Report into implementable stories.

## Requirements Inventory

### Functional Requirements

**Category 1: Live Trading & Execution**

- FR1: The trader can view all active bot statuses, live P&L, and open positions in real time
- FR2: The system can execute trade entries and exits via the MT5 bridge on behalf of configured bots
- FR3: The trader can manually close any open position directly from the ITT
- FR4: The system can route all strategy signals through the Sentinel → Governor → Commander pipeline before any order execution
- FR5: The trader can view the active market regime classification and strategy router state at all times
- FR6: The system can activate strategies based on session context (London, New York, Asian, Swiss, custom) per the session mask configured in each EA's input parameters
- FR7: The trader can monitor live tick stream data and spread conditions per instrument in real time
- FR8: The system can enforce daily loss caps and maximum trade limits at the aggregate account level; each EA enforces its own force-close timing and overnight hold rules autonomously via MQL5 input parameters
- FR9: The trader can activate any tier of the kill switch — soft stop through full emergency position close — at any time from any canvas

**Category 2: Autonomous Agent System**

- FR10: The Copilot can receive natural language instructions and orchestrate the appropriate department, agent, or workflow response
- FR11: The Floor Manager can classify incoming requests and route them to the appropriate Department Head without trader intervention; the Floor Manager can also handle administrative tasks directly when appropriate
- FR12: Each Department Head can spawn parallel sub-agents to handle complex multi-part tasks concurrently
- FR13: The Copilot can create, schedule, and manage automated workflows — cron jobs, hooks, triggers — on behalf of the trader
- FR14: The system can maintain persistent agent memory across sessions — conversation context, strategy knowledge, and trader preferences
- FR15: Department agents can communicate with each other via the Department Mail bus asynchronously
- FR16: The Copilot can surface Department Mail notifications and agent task completions to the trader
- FR17: The trader can review agent activity, task queues, and sub-agent spawning history
- FR18: The system can register and execute agentic skills as composable, reusable capabilities across department workflows
- FR19: Department Heads and the Copilot can author new agentic skills and register them to the skill library (Skill Forge)
- FR20: The Copilot can operate context-aware on any canvas, with tools and commands appropriate to the active department
- FR21: The Copilot can maintain a task list and route at least 5 simultaneous tasks to different agents and tools concurrently, with delegation to the Floor Manager for tasks that do not require Copilot-level orchestration
- FR22: The trader can directly converse with any individual Department Head via the Department Chat panel

**Category 3: Alpha Forge — Strategy Factory**

> Architecture note (2026-03-14): Alpha Forge is two independent workflows. Workflow 1 (Creation): video/source → TRD → SDD → build → compile → basic backtest → EA Library. Workflow 2 (Enhancement Loop): full backtest suite (all 6 modes) → research-driven improvement → Dukascopy data variant stress testing → SIT gate (MODE_C) → paper trading → human approval → live.

- FR23: The system can execute a complete Alpha Forge loop from research hypothesis to live deployment candidate without mandatory manual intervention at intermediate stages
- FR24: The Research Department can autonomously generate strategy hypotheses from knowledge base content, news events, and market data
- FR25: The Development Department can generate MQL5 EA code from a strategy specification, including all required EA input tags
- FR26: The system can produce Vanilla and Spiced EA variants from the same base strategy specification
- FR27: The system can run a full backtest matrix — Standard, Monte Carlo, Walk-Forward, and System Integration Test — per EA variant
- FR28: The Risk Department can evaluate backtest results and present a promotion recommendation; a human trader must explicitly approve before any EA advances beyond paper trading to live deployment
- FR29: The Trading Department can monitor paper trading performance of a new EA before live promotion is available for human approval
- FR30: The trader can review, modify, or reject any Alpha Forge output at each gate in the pipeline
- FR31: The system can maintain a versioned strategy library with full knowledge provenance chain per strategy — from source article through to live deployment

**Category 4: Risk Management & Compliance**

- FR32: The physics risk engine can classify the market regime (TREND / RANGE / BREAKOUT / CHAOS) in real time using the Ising Model and Lyapunov Exponent sensors as the primary signal sources
- FR33: The HMM sensor can run in shadow mode — observing and logging regime classifications — without its output controlling the strategy router until the model is validated and deemed fit
- FR34: The Ising Model sensor can detect systemic correlation risk across assets
- FR35: The Lyapunov Exponent sensor can detect chaotic instability conditions and trigger risk escalation
- FR36: The Physics-Aware Kelly Engine can calculate position sizes with regime-based physics multipliers and fee awareness (commission, spread cost, swap); when account equity exceeds the initial deposit by a configured threshold, it applies house-of-money scaling to reflect the expanded trading buffer
- FR36b: The Portfolio Department can monitor account equity state against configured initial deposit thresholds and propagate house-of-money status system-wide — adjusting the Governor risk scalar, Commander strategy activation profile, Alpha Forge promotion criteria, and Copilot contextual awareness accordingly
- FR37: The BotCircuitBreaker can automatically quarantine any bot reaching its configured consecutive loss threshold — configurable per account tag (prop firm vs. personal book)
- FR38: The Governor can enforce aggregate account-level compliance rules across all active bots simultaneously — total drawdown limits, daily loss caps, and prop-firm-specific halt conditions
- FR39: The system can enforce Islamic trading compliance rules — no overnight positions, swap-free accounts, effective 1:1 leverage — per account configuration
- FR40: Each EA autonomously manages its own per-trade parameters — spread filter, force-close timing, session mask, overnight hold — via MQL5 input tags; the system displays these parameters for transparency and does not impose independent overrides
- FR41: The trader can configure risk parameters per account tag independently

**Category 5: Knowledge & Research**

- FR42: The system can scrape and ingest financial articles and market research from configured external sources
- FR43: The Research Department can query the knowledge base using semantic search to retrieve relevant strategy context
- FR44: The system can index all knowledge base content for full-text and semantic retrieval via the knowledge indexing system
- FR45: The trader can add personal knowledge entries, research notes, and strategy observations to the knowledge base
- FR46: The system can maintain provenance metadata for every knowledge entry — source, date, relevance, and linked strategy
- FR47: The system can ingest video content — individual videos or playlists — via the video ingest pipeline and extract strategy signals from transcripts
- FR48: The system can partition the knowledge base by type — scraped, ingested, personal, system-generated — with appropriate access per partition
- FR49: The trader can query the knowledge base via Copilot using natural language
- FR50: The ITT can display a live news feed with enriched macro context for the trader to monitor during active sessions

**Category 6: Portfolio & Multi-Broker Management**

- FR51: The trader can register and configure at least 4 broker accounts simultaneously in the broker registry
- FR52: The system can auto-detect MT5 account properties — broker, account type, leverage, currency — on connection
- FR53: The routing matrix can assign strategies to specific broker accounts based on account tag, regime, and strategy type
- FR54: The system can manage simultaneous operation of multiple strategy types — scalpers, trend-followers, mean-reversion, event-driven — across different broker accounts and sessions
- FR55: The trader can review portfolio-level performance metrics — total equity, drawdown, P&L attribution per strategy and per broker
- FR56 *(Phase 2)*: The system can generate client performance reports with P&L attribution, drawdown analysis, and trade history
- FR57 *(Phase 2)*: The trader can configure copy trading replication from a master account to other funded accounts
- FR58 *(Phase 2)*: The system can prepare a verified track record export for DarwinEx capital allocation application

**Category 7: Monitoring, Audit & Notifications**

- FR59: The system logs all system events at the appropriate level — debug, info, warning, error, critical — with no events silently excluded
- FR60: The trader can query the complete event log and audit trail via Copilot using natural language
- FR61: The system can retain all logs for a minimum of 3 years with automated cold storage sync to the agent/data server
- FR62: The trader can configure which events trigger active notifications, at what severity, and via which delivery channel
- FR63: The system can deliver notifications via OS system tray; mobile push delivery available in Phase 2
- FR64: The trader can access all agent communications and task outputs via the Department Mail inbox
- FR65: The system can monitor and report server health, connectivity, and latency for both the agent server and trading server nodes in real time

**Category 8: System Management & Infrastructure**

- FR66: The trader can configure AI provider settings — model tier, API key, base URL — and swap providers at runtime without system restart
- FR67: The Copilot can trigger a sequential system update across all three nodes with health check and automatic rollback on failure
- FR68: The trader can add, modify, or remove prop firm registry entries and their associated compliance rule sets
- FR69: The trader can rebuild the ITT on a new machine from source without loss of agent state, strategy library, or configuration
- FR70: The trader can migrate server-side infrastructure to a new provider without loss of data or functionality
- FR71: The system can manage scheduled background tasks on the agent/data server, configurable via Copilot
- FR72: The trader can build the ITT on Linux, Windows, and macOS from source
- FR73 *(Phase 2)*: The trader can access a mobile companion interface for system monitoring, full kill switch, and Copilot chat when away from the primary machine

**Category 9: Traceability Gap Closures**

- FR74: The system can run parallel A/B comparisons of strategy variants on paper or live accounts with statistical confidence scoring and a live race board
- FR75: The trader can roll back a live strategy to any previous version in the versioned strategy library with one instruction
- FR76: The system can extract lessons from retired or underperforming strategies and propagate pattern-matched fixes across active strategies sharing the same base logic
- FR77: The system can integrate economic calendar events and apply calendar-aware trading rules — lot scaling, entry pauses, post-event reactivation — per account or strategy configuration
- FR78: The Copilot can explain its reasoning chain for any past recommendation or action when asked, surfacing the factors, data sources, and decision logic that contributed to the output
- FR79: The system can deploy compiled EA files to the MT5 terminal via the bridge — including chart attachment, parameter injection, health check, and ZMQ stream registration

**Total: 80 FRs (FR1–FR79 + FR36b)**

---

### NonFunctional Requirements

**Performance:**
- NFR-P1: The kill switch protocol must execute its configured tier (soft stop / progressive / full close) in full, in order, without skipping steps — correctness over raw speed
- NFR-P2: The Live Trading Canvas and StatusBand must reflect live P&L, regime, and bot state with no more than 3 seconds of lag under normal network conditions
- NFR-P3: Copilot responses must begin streaming within 5 seconds of request submission
- NFR-P4: Canvas transitions must complete within 200ms; UI interactions must respond within 100ms
- NFR-P5: HMM shadow mode classification must update within one tick cycle of new data arriving
- NFR-P6: Backtest matrix (Standard + Monte Carlo + Walk-Forward + SIT) for a standard 1-year dataset must complete within 4 hours

**Security:**
- NFR-S1: SSH key-only access on Contabo — no password authentication, no root login
- NFR-S2: All proprietary IP (EA source code, Alpha Forge output, knowledge base, strategy library, MQL5 PDFs) stored exclusively on private infrastructure; never exposed in public repositories or transmitted raw to third-party AI APIs
- NFR-S3: All API keys and credentials in `.env` files only — never in source code or version control
- NFR-S4: Agent memory, knowledge base, and strategy files restricted to authenticated server processes
- NFR-S5: Cloudzy firewall-restricted to trusted IPs (Contabo agent server + developer machine)
- NFR-S6: No sensitive intellectual property stored on Cloudzy node

**Reliability & Availability:**
- NFR-R1: Individual component failure (MT5 bridge restart, agent crash, API timeout) must not cascade to full system failure — components recover independently
- NFR-R2: MT5 bridge must reconnect automatically after disconnection without manual intervention
- NFR-R3: WebSocket subscriptions (tick feed, agent stream) must auto-reconnect with exponential backoff
- NFR-R4: The strategy router and kill switch must remain operational on Cloudzy even if the Contabo agent server is temporarily unreachable
- NFR-R5: MT5 ZMQ connection loss must be detected within 10 seconds and trigger automatic reconnection
- NFR-R6: Periodic data backups of all Contabo state (agent memory, knowledge base, strategy library, SQLite databases) to a secondary backup location

**Data Integrity:**
- NFR-D1: Trade records and position data must be persisted to SQLite before any system acknowledgment of execution
- NFR-D2: Audit log entries are immutable once written — no deletion, no modification
- NFR-D3: Strategy versions and backtest results must be associated with the exact EA code version that generated them — provenance chain preserved on all updates
- NFR-D4: Knowledge base ingestion must preserve full provenance metadata (source, date, linked strategy) on every entry
- NFR-D5: Cold storage sync to Contabo must include integrity verification — corrupted or incomplete transfers flagged and retried
- NFR-D6: Tiered tick data storage: Hot (PostgreSQL, <1hr, Cloudzy), Warm (DuckDB, 7 days), Cold (long-term archive on Contabo)

**Integration Reliability:**
- NFR-I1: All configured AI provider APIs must implement timeout handling and retry logic — failures queue rather than crash
- NFR-I2: Provider API rate limit or quota errors must be handled gracefully — agent tasks degrade to queued state, not failure state
- NFR-I3: The Anthropic Agent SDK and Anthropic API must both degrade gracefully on connectivity issues
- NFR-I4: Knowledge pipeline tool failures (scraper, video ingest) must be logged and retried without blocking other system operations

**Maintainability:**
- NFR-M1: Existing LangChain and LangGraph code is technical debt — migrate to Claude Agent SDK; no new LangChain/LangGraph code introduced
- NFR-M2: The Claude Agent SDK + department paradigm (FloorManager → Department Heads → Sub-agents) is the sole canonical agent architecture
- NFR-M3: All Python backend files kept under 500 lines — refactor at boundary
- NFR-M4: All Svelte components kept under 500 lines
- NFR-M5: New agent capabilities added via skill registration, not modification of core department code
- NFR-M6: Risk-critical components (Kelly Engine, BotCircuitBreaker, Governor, Sentinel) must have test coverage before any modification
- NFR-M7: All FastAPI endpoints follow the established router-per-module pattern

**Total: 27 NFRs**

---

### Additional Requirements

**From Architecture:**

- **No greenfield scaffold** — brownfield upgrade. First implementation actions: (1) `npx sv migrate svelte-5` in `quantmind-ide/`, (2) add `NODE_ROLE` env var handling to `src/api/server.py`, (3) remove `langchain`/`langgraph`/`langchain_openai` from `requirements.txt`
- **Single codebase, `NODE_ROLE` deployment split**: `NODE_ROLE=contabo|cloudzy|local` controls which FastAPI router groups register at startup — Contabo gets agent/knowledge/research routers; Cloudzy gets trading/kill-switch/MT5 routers
- **Svelte 4 → Svelte 5 in-place migration** via `sv migrate svelte-5` codemod — retain static adapter, SvelteKit routing, TypeScript strict mode, Vite 5, Tailwind. 65+ components to migrate
- **Agent SDK migration** — remove LangGraph/LangChain runtime, migrate `src/agents/core/base_agent.py` and `src/agents/registry.py` to Claude Agent SDK. Department structure preserved, runtime replaced
- **Stay untouched (do not modify)**: PhysicsAwareKellyEngine + full risk pipeline, RoutingMatrix + broker registry, MT5 ZMQ bridge, tick data pipeline, entire backtest engine (`src/backtesting/` — 6 confirmed working modes)
- **Stay but extend (targeted additions only)**: `src/memory/graph/` (add columns + ReflectionExecutor + embeddings), `src/agents/memory/session_checkpoint_service.py`, existing Prefect data pipeline flows, MCP server configs
- **Rebuild from scratch**: `src/agents/` department system, `quantmind-ide/` entire frontend, `shared_assets/skills/` hierarchy, department system prompts
- **Redis Streams** replaces SQLite-backed DepartmentMailService on Contabo — consumer groups, acknowledgment, namespace isolation per workflow
- **Four-layer knowledge stack**: PageIndex (full-text scraping), ChromaDB + sentence-transformers (semantic search), Graph Memory (cross-session agent memory), Neo4j (Phase 2)
- **Graph Memory prerequisite**: Graph Memory system must be fully implemented before the Canvas Context System (CanvasContextTemplate per department) is built
- **Prefect** manages all workflow orchestration (Alpha Forge, knowledge sync, HMM retraining, drawdown reviews); `workflows.db` (SQLite) on Contabo only
- **Docker on Contabo**: Redis + Prefect run in Docker containers
- **Audit trail cross-cuts everything**: every API endpoint, agent task dispatcher, Commander pipeline must write immutable audit entries
- **Node independence**: Cloudzy must trade without Contabo reachable — strategy router + kill switch run self-contained on Cloudzy

**From UX Design:**

- **Desktop-only Tauri 2 app** — fullscreen usage expected; single screen primary; no SSR (static adapter)
- **8-canvas + FlowForge structure** (9 canvases total): Live Trading, Research, Development, Risk, Trading, Portfolio, Shared Assets, Workshop, FlowForge — each is a full-screen department workspace
- **Global UI shell is locked**: TopBar (Kill Switch · Copilot shortcut · Notifications · Settings), StatusBand (28–32px ambient ticker), ActivityBar (icon-only, 8 entries + Settings), Canvas Workspace, Agent Panel (collapsible right rail)
- **StatusBand is clickable ambient navigation**: session clocks → Live Trading, active bots → Portfolio, risk mode → Risk canvas, node health dot → node status overlay
- **Trading Kill Switch in TopBar always** — two-step deliberate activation (armed state → confirm modal). Workflow Kill Switch is per-workflow in FlowForge Prefect Kanban (independent controls)
- **Canvas navigation pattern**: tile grid (default) → sub-page on click → breadcrumb [← Back]. Cross-canvas via 3-dot contextual menus on entities (EAs, workflows, strategies)
- **Canvas transitions ≤200ms** — no perceived delay; skeleton loaders or graceful fades for data (no blank-then-populated)
- **CanvasContextTemplate per canvas**: when Agent Panel chat opens, department head pre-loaded with correct memory scope, workflow namespaces, skill access
- **Two distinct Copilot modes**: (a) Workshop canvas — FloorManager full-screen (Claude.ai-inspired); (b) Agent Panel per canvas — department-head-specific, streaming thoughts + tool use visible, collapsible
- **Degraded mode handling**: if Contabo drops, UI remains functional; Live Trading canvas stays live via WebSocket; Contabo-dependent canvases show degraded indicators (not error screens)
- **Session-scoped time model**: temporal reference is the trading session, not wall clock. Three live session clocks in StatusBand (Tokyo/Asian, London/European, New York/NY) — real current time + open/closed state
- **Frosted Terminal aesthetic**: deep space blue-black (#080d14), frosted glass panels (backdrop-filter blur), scan-line overlay, amber (#f0a500) active, cyan (#00d4ff) AI, red (#ff3b3b) kill/danger. JetBrains Mono (data) + Syne 700/800 (headings). Lucide icons throughout (no emoji)
- **Rich media rendering in agent output**: diagrams, tables, charts rendered inline — not raw markdown. File references clickable and navigable
- **Keyboard shortcuts reserved for power actions**: canvas switching (keys 1–8), trading kill switch trigger, Copilot focus

---

### FR Coverage Map

| FR | Epic | Notes |
|---|---|---|
| FR1 | Epic 3 | Live bot status, P&L, open positions |
| FR2 | Epic 3 | MT5 bridge trade execution |
| FR3 | Epic 3 | Manual position close |
| FR4 | Epic 3 | Sentinel → Governor → Commander pipeline |
| FR5 | Epic 3 | Regime + router state display |
| FR6 | Epic 3 | Session mask activation |
| FR7 | Epic 3 | Live tick stream + spread monitoring |
| FR8 | Epic 3 | Daily loss caps + trade limits |
| FR9 | Epic 3 | Kill switch tiers (TopBar) |
| FR10 | Epics 5+7 | Copilot orchestration (core in E5, full in E7) |
| FR11 | Epics 5+7 | FloorManager routing (core in E5, semantic in E7) |
| FR12 | Epic 7 | Department Head parallel sub-agents |
| FR13 | Epic 5 | Copilot workflow/cron/hook creation |
| FR14 | Epic 5 | Persistent agent memory (memory unification) |
| FR15 | Epics 5+7 | Department Mail bus (SQLite E5, Redis E7) |
| FR16 | Epic 5 | Notification surfacing via Copilot |
| FR17 | Epic 7 | Agent activity + task queue review |
| FR18 | Epic 7 | Skill registration + execution (Skill Forge) |
| FR19 | Epic 7 | Skill authoring by Department Heads |
| FR20 | Epic 5 | Canvas-aware Copilot context (CanvasContextTemplate) |
| FR21 | Epic 7 | Concurrent task routing (≥5 tasks) |
| FR22 | Epic 7 | Direct Department Head chat |
| FR23 | Epic 8 | Full Alpha Forge loop |
| FR24 | Epic 8 | Research hypothesis generation |
| FR25 | Epic 8 | MQL5 EA code generation |
| FR26 | Epic 8 | Vanilla + Spiced variants |
| FR27 | Epic 8 | Full 6-mode backtest matrix |
| FR28 | Epic 8 | Risk evaluation + human approval gate |
| FR29 | Epic 8 | Paper trading monitoring |
| FR30 | Epic 8 | Alpha Forge gate review/modify/reject |
| FR31 | Epic 8 | Versioned strategy library + provenance chain |
| FR32 | Epic 4 | Regime classification (Ising + Lyapunov) |
| FR33 | Epic 4 | HMM shadow mode display |
| FR34 | Epic 4 | Ising Model correlation risk visualization |
| FR35 | Epic 4 | Lyapunov chaotic instability detection |
| FR36 | Epic 4 | Physics-Aware Kelly Engine + house-of-money |
| FR36b | Epic 4 | Portfolio equity → house-of-money propagation |
| FR37 | Epic 4 | BotCircuitBreaker quarantine |
| FR38 | Epic 4 | Governor account-level compliance |
| FR39 | Epic 4 | Islamic compliance enforcement |
| FR40 | Epic 4 | EA per-trade parameter transparency |
| FR41 | Epic 4 | Per account tag risk config |
| FR42 | Epic 6 | Web scraping + article ingestion (Firecrawl) |
| FR43 | Epic 6 | Semantic knowledge base query |
| FR44 | Epic 6 | Full-text + semantic indexing (PageIndex) |
| FR45 | Epic 6 | Personal knowledge entry |
| FR46 | Epic 6 | Provenance metadata |
| FR47 | Epic 6 | Video ingest pipeline (YouTube → TRD) |
| FR48 | Epic 6 | Knowledge base partitioning |
| FR49 | Epic 6 | Natural language knowledge query via Copilot |
| FR50 | Epic 6 | Live news feed with macro enrichment |
| FR51 | Epic 9 | Broker account registry (≥4) |
| FR52 | Epic 9 | MT5 account auto-detection |
| FR53 | Epic 9 | Routing matrix assignment |
| FR54 | Epic 9 | Multi-strategy-type concurrent operation |
| FR55 | Epic 9 | Portfolio-level performance metrics |
| FR56 | *Phase 2* | Client reports — deferred |
| FR57 | *Phase 2* | Copy trading — deferred |
| FR58 | *Phase 2* | DarwinEx track record — deferred |
| FR59 | Epic 10 | System event logging (all levels) |
| FR60 | Epic 10 | Natural language audit trail query |
| FR61 | Epic 10 | 3-year log retention + cold storage sync |
| FR62 | Epic 10 | Configurable notification triggers |
| FR63 | Epic 10 | OS tray notification delivery |
| FR64 | Epic 10 | Department Mail inbox |
| FR65 | Epic 10 | Server health + latency monitoring |
| FR66 | Epic 2 | Runtime AI provider swap |
| FR67 | Epic 11 | 3-node sequential update with rollback |
| FR68 | Epic 4 | Prop firm registry CRUD |
| FR69 | Epic 11 | Machine portability (rebuild without loss) |
| FR70 | Epic 11 | Server migration without data loss |
| FR71 | Epic 11 | Scheduled background tasks (Copilot-configurable) |
| FR72 | Epic 1 | Build from source (Linux, Windows, macOS) |
| FR73 | *Phase 2* | Mobile companion — deferred |
| FR74 | Epic 8 | A/B strategy variant comparison + race board |
| FR75 | Epic 8 | Strategy rollback (one instruction) |
| FR76 | Epic 8 | Cross-strategy lesson extraction + propagation |
| FR77 | Epic 4 | Calendar-aware trading rules |
| FR78 | Epic 10 | Copilot reasoning chain explanation |
| FR79 | Epic 8 | EA deployment pipeline (file → chart → ZMQ) |

## Epic List

### Epic 1: Platform Foundation & Global Shell

The rebuilt ITT launches. Svelte 5 migration complete. `NODE_ROLE` env split in place. The global shell — TopBar (Kill Switch, Copilot shortcut, Notifications, Settings), StatusBand (ambient ticker), ActivityBar (9 canvases), Agent Panel skeleton — is implemented in the Frosted Terminal aesthetic. `.env` removed from git tracking.

**Work type:** Migration + Rebuild
**FRs covered:** FR72
**Architecture work:** Svelte 4→5 migration (65+ components), LangGraph/LangChain import cleanup (5-8 files, deps already removed), `NODE_ROLE` env var in `src/api/server.py`, global shell component rebuild, 9-canvas routing skeleton, `.env` git tracking fix

---

### Epic 2: AI Providers & Server Connections

Mubarak can configure AI providers (Anthropic, GLM, MiniMax, OpenRouter, DeepSeek), assign models to the three tiers (FloorManager / Department Heads / Sub-agents), test connections with live responses, and switch providers at runtime without restart. Server connectivity tests show Cloudzy/Contabo/Local latency live.

**Work type:** Wire + Verify + New UI
**FRs covered:** FR66
**Journeys:** 7, 15

---

### Epic 3: Live Trading Command Center

Mubarak opens the ITT to the Live Trading canvas as home screen. StatusBand shows live session clocks (Tokyo/London/New York with open/closed state), active bots, daily P&L, node health dots, risk mode, router mode. Kill switch always in TopBar (two-step: armed → confirm modal). Live positions, equity, and regime state stream via WebSocket with ≤3s lag. Degraded mode (Contabo drop) handled gracefully.

**Work type:** Wire + Rebuild UI
**FRs covered:** FR1–FR9
**Journeys:** 1, 4, 8, 18, 41

---

### Epic 4: Risk Management & Compliance

The Risk canvas is live. Physics sensors (Ising Model, Lyapunov Exponent, HMM in shadow mode) are visualized. Kelly Engine settings configurable with house-of-money tracking. BotCircuitBreaker and Governor enforcement visible and configurable. Islamic compliance (force-close 21:45 GMT) is structural. Prop firm registry fully configurable (Challenge Mode with StatusBand progress). Session configuration editable (custom sessions per Journey 26). Calendar-aware trading rules built once, run automatically.

**Work type:** New UI + Wire (backend is production-ready)
**FRs covered:** FR32–FR41, FR36b, FR68, FR77
**Journeys:** 10, 17, 23, 26, 45

---

### Epic 5: Unified Memory & Copilot Core

Workshop canvas becomes the Copilot's home (Claude.ai-inspired: centered input, history, projects, memory, skills). Mubarak can talk to the Copilot and have instructions routed through the FloorManager. Memory unified across 4 fragmented systems (agent/global/graph/department markdown) into a single coherent layer. Agent Panel appears on each canvas pre-loaded with the correct `CanvasContextTemplate`. Department Mail (SQLite) verified working.

**Work type:** Build (memory unification) + Wire (Agent Panel components exist) + Rebuild (Workshop canvas)
**FRs covered:** FR10, FR11, FR13, FR14, FR15, FR16, FR20
**Journeys:** 5, 38, 49

---

### Epic 6: Knowledge & Research Engine

Mubarak can feed the system any source — web articles (Firecrawl), YouTube URLs, personal notes, PDFs, Obsidian vault exports — and have it ingested, indexed with full provenance, and semantically searchable via Copilot or Research Department. Live news feed monitors macro events. Sub-90-second news-to-alert pipeline (Research geopolitical sub-agent).

**Work type:** Wire + Verify + New (personal knowledge, news alert pipeline)
**FRs covered:** FR42–FR50
**Journeys:** 13, 25, 30, 46

---

### Epic 7: Department Agent Platform — Departments Do Real Work

Department heads stop being stubs. Research generates real hypotheses. Development writes real MQL5 EA code. Risk evaluates real backtest results. Trading monitors real paper trades. Portfolio reports real performance. Sub-agent spawning works (parallel Haiku-tier workers). Skill Forge live (skill registration + self-authoring). Department Mail switches to Redis Streams. Concurrent task routing ≥5 simultaneous tasks.

**Work type:** Build (stubs → real implementations)
**FRs covered:** FR10–FR22 (completing), FR17, FR18, FR19, FR21, FR22
**Journeys:** 11, 12, 16, 29, 32, 34, 35, 47, 48, 49, 50

---

### Epic 8: Alpha Forge — Strategy Factory

Mubarak pastes a YouTube URL. 72 hours later, EA variants are paper trading. The complete Alpha Forge pipeline (both workflows) is operational: source → Research → TRD → Development → compile → 6-mode backtest → SIT gate → paper trading → human approval → live deployment. Strategy Template Library for fast-track deployment. Full version control (rollback one instruction), A/B variant comparison (live race board), cross-strategy loss propagation, EA deployment pipeline to MT5 (file transfer → terminal registration → parameter injection → health check → ZMQ registration).

**Work type:** Wire + Build (orchestrator stages exist, departments now real)
**FRs covered:** FR23–FR31, FR74, FR75, FR76, FR79
**Journeys:** 2, 3, 9, 12, 19, 21, 22, 33, 37, 39, 43, 44, 52

---

### Epic 9: Portfolio & Multi-Broker Management

Mubarak can register at least 4 broker accounts, configure the routing matrix to assign strategies by account tag/regime/strategy type, view portfolio-level equity/drawdown/P&L attribution per strategy and per broker, and review the correlation matrix. Multi-strategy type concurrent operation is fully visible and managed.

**Work type:** Wire + Rebuild UI (canvas model)
**FRs covered:** FR51–FR55
**Journeys:** 6, 20, 27

---

### Epic 10: Audit, Monitoring & Notifications

Mubarak can ask "Why was EA_X paused yesterday?" and get a full timestamped causal chain. All 5 audit layers (trade, strategy lifecycle, risk param, agent action, system health) are queryable in natural language via Copilot. Notifications configurable per event type. Server health live for both nodes. Copilot explains its reasoning for any past decision on request. Notification analytics + AI-suggested suppressions reduce volume.

**Work type:** Build (NL query, reasoning transparency) + Wire (audit infra exists)
**FRs covered:** FR59–FR65, FR78
**Journeys:** 29, 34, 36, 51

---

### Epic 11: System Management & Resilience

Mubarak can trigger a coordinated 3-node sequential update with health check and automatic rollback. He can migrate to a new server provider without losing data. The ITT is rebuild-portable (new machine, full restore). Scheduled background tasks run on Contabo configurable via Copilot. Weekend compute protocol fires automatically (full Monte Carlo, HMM retraining, KB cross-document pass).

**Work type:** Build + Wire
**FRs covered:** FR67, FR69, FR70, FR71
**Journeys:** 7, 28, 42

---


## Stories

> **Story ordering per epic:** Exploration/Audit first → Backend (data models, services, APIs) → Middleware (routing, SDK wiring, Redis, integration) → Frontend (canvas layouts, components, UI)

---

## Epic 1: Platform Foundation & Global Shell

### Story 1.0: Platform Codebase Exploration & Audit

**As a** developer starting work on the QUANTMINDX brownfield project,
**I want** a complete audit of the current frontend, backend, and configuration state,
**So that** all subsequent stories in this epic are grounded in verified existing reality rather than assumptions.

**Acceptance Criteria:**

**Given** the brownfield codebase in `quantmind-ide/` and `src/`,
**When** the exploration runs,
**Then** a findings document is produced covering: (a) complete list of all Svelte components and their reactive syntax (Svelte 4 vs 5), (b) all LangChain/LangGraph import occurrences with file paths, (c) current `.env` git tracking state, (d) existing `server.py` router registration structure, (e) existing `StatusBand.svelte` and `TradingFloorPanel.svelte` state.

**Given** the findings document exists,
**When** implementation begins on Stories 1.1–1.6,
**Then** each story references the audit findings to scope work accurately,
**And** no story overwrites working functionality discovered during the audit.

**Notes:**
- Output: findings appended to this story's notes — not a separate file
- Scan targets: `quantmind-ide/src/`, `src/agents/`, `src/server.py`, `.gitignore`, `.env`
- This story is read-only exploration — no code changes

---

### Story 1.1: Security Hardening & Legacy Import Cleanup

**As a** developer setting up QUANTMINDX for ongoing development,
**I want** all secrets removed from git tracking and all dead LangGraph/LangChain imports eliminated,
**So that** the codebase has no security vulnerabilities at the foundation layer and no import errors from removed packages.

**Acceptance Criteria:**

**Given** `.env` is currently tracked in git (confirmed by Story 1.0 audit),
**When** the story tasks run,
**Then** `.env` is added to `.gitignore`, removed from git tracking via `git rm --cached .env`, and a `.env.example` template with all required keys (no values) is committed in its place.

**Given** LangGraph/LangChain imports exist in 5–8 backend files (identified in Story 1.0 audit),
**When** each import is removed or stubbed,
**Then** `from langchain` and `from langgraph` produce zero results in `grep -r` across `src/`,
**And** `pip install -r requirements.txt` completes without errors on a clean environment.

**Given** the backend runs after cleanup,
**When** `uvicorn src.server:app` starts,
**Then** no `ImportError` or `ModuleNotFoundError` appears in startup logs.

**Notes:**
- Do NOT delete `.env` from disk — only untrack from git
- `.env.example` must document every key used across all backend modules
- NFR-S3: API keys in `.env` only — never in source code

---

### Story 1.2: Svelte 5 Migration

**As a** frontend developer,
**I want** the entire frontend migrated from Svelte 4 to Svelte 5 runes syntax,
**So that** reactive patterns use `$state`, `$derived`, and `$effect` throughout and the build passes cleanly.

**Acceptance Criteria:**

**Given** the component count established in Story 1.0 audit,
**When** `npx sv migrate svelte-5` runs in `quantmind-ide/`,
**Then** the migration tool completes and produces a report of all flagged files.

**Given** the migration tool flags components with ambiguous patterns,
**When** each flagged file is resolved manually,
**Then** no Svelte 4 reactive declarations (`$:`) remain in any `.svelte` file,
**And** `let` reactive bindings are converted to `$state()`,
**And** derived computations use `$derived()`,
**And** side effects use `$effect()`.

**Given** all components are migrated,
**When** `npm run build` runs from `quantmind-ide/`,
**Then** build completes with zero errors and zero Svelte 4 deprecation warnings.

**Notes:**
- Run `npx sv migrate svelte-5` as the starting point — do not hand-migrate
- Architecture constraint: no `export let`, no `$:`, no `writable()` in new code
- NFR-M4: Svelte components kept under 500 lines

---

### Story 1.3: NODE_ROLE Backend Deployment Split

**As a** system operator deploying QUANTMINDX across two servers,
**I want** the FastAPI backend to register only the routers appropriate to each node based on `NODE_ROLE`,
**So that** Cloudzy runs only live trading routers and Contabo runs only agent/compute routers.

**Acceptance Criteria:**

**Given** `NODE_ROLE=cloudzy` is set,
**When** the FastAPI server starts,
**Then** only live trading routers (MT5 bridge, execution, sentinel, risk) register,
**And** agent/compute routers are not mounted.

**Given** `NODE_ROLE=contabo` is set,
**When** the FastAPI server starts,
**Then** only agent/compute routers register,
**And** live trading routers are not mounted.

**Given** `NODE_ROLE=local` is set or the variable is absent,
**When** the FastAPI server starts,
**Then** all routers register (full local development mode).

**Given** an invalid `NODE_ROLE` value is set,
**When** the server starts,
**Then** it logs a clear warning and defaults to `local` mode without crashing.

**Notes:**
- Implement in `src/server.py` — conditional router registration block
- Add `NODE_ROLE` to `.env.example` with accepted values: `contabo | cloudzy | local`
- Architecture constraint: Cloudzy must trade without Contabo reachable (NFR-R4)

---

### Story 1.4: TopBar & ActivityBar — Frosted Terminal Aesthetic

**As a** trader using QUANTMINDX,
**I want** the TopBar and ActivityBar implemented in the Frosted Terminal aesthetic with Lucide icons,
**So that** the global shell frames every canvas with a consistent visual identity.

**Acceptance Criteria:**

**Given** the application loads,
**When** the TopBar (48px fixed) renders,
**Then** it displays: QUANTMINDX wordmark + active canvas name (left), system status indicators — TradingKillSwitch, Workshop button, Notifications, Settings (right),
**And** background is `rgba(8, 13, 20, 0.08)` with `backdrop-filter: blur(24px) saturate(160%)`,
**And** all icons are `lucide-svelte` — no emoji.

**Given** the ActivityBar renders,
**When** it mounts (56px collapsed, 200px expanded),
**Then** it shows navigation icons for all 9 canvases in a left-side vertical strip,
**And** the active canvas icon uses cyan left-border indicator (`--color-accent-cyan`),
**And** inactive icons render at 60% opacity, hover transitions to 100%.

**Given** a canvas icon is clicked,
**When** the navigation fires,
**Then** the canvas switches within ≤200ms,
**And** keyboard shortcuts 1–9 also trigger canvas switches.

**Given** the TradingKillSwitch in TopBar is in ARMED state,
**When** it renders,
**Then** it pulses red `#ff3b3b` at 2s interval (Lucide `shield-alert`),
**And** clicking it opens a two-step confirm modal (arm → confirm) — Enter does NOT confirm destructive action.

**Notes:**
- Two-tier glass: shell layer `0.08` opacity, content layer `0.35` opacity
- Fonts: Syne 800 (wordmark), Space Grotesk 500 (nav labels), JetBrains Mono (data)
- Lucide only — `lucide-svelte` package; verify installation in `package.json`

---

### Story 1.5: StatusBand Redesign — Frosted Terminal Ticker

**As a** trader using QUANTMINDX,
**I want** the StatusBand redesigned as a 28–32px Frosted Terminal ambient ticker,
**So that** live portfolio metrics, system health, and regime data are always visible and navigable.

**Acceptance Criteria:**

**Given** the application is running,
**When** the StatusBand (32px fixed bottom) renders,
**Then** it displays scrolling segments: session clocks (Tokyo/London/NY with open/closed state), active bot count, daily P&L, node health dots (Cloudzy · Contabo · Local), workflow count, challenge progress,
**And** background is Tier 1 glass (`rgba(8,13,20,0.08)`, `blur(24px)`),
**And** ticker text uses Fragment Mono 400 11–12px.

**Given** a StatusBand segment is clicked,
**When** the click registers,
**Then** session clocks → Live Trading canvas, active bots → Portfolio canvas, risk mode → Risk canvas, node health dot → node status overlay.

**Given** a metric value changes,
**When** the update renders,
**Then** positive P&L flashes green `#00c896` (100ms), negative flashes red `#ff3b3b` (100ms),
**And** `aria-live="polite"` announces changes to screen readers without disruption.

**Given** Contabo node is unreachable,
**When** the StatusBand renders degraded state,
**Then** Contabo dot shows red with Lucide `wifi-off` icon,
**And** all live data continues displaying last-known values with `[stale]` label.

**Notes:**
- Read existing `StatusBand.svelte` before editing (Story 1.0 audit should have catalogued its state)
- All icons: Lucide only — no emoji
- `aria-live="polite"` on ambient updates

---

### Story 1.6: 9-Canvas Routing Skeleton

**As a** developer building QUANTMINDX features,
**I want** the main content area structured to route between all 9 named canvases,
**So that** every canvas has a placeholder that later epics build into without routing conflicts.

**Acceptance Criteria:**

**Given** the application loads,
**When** the routing system initialises,
**Then** all 9 routes register: `live-trading`, `research`, `development`, `risk`, `trading`, `portfolio`, `shared-assets`, `workshop`, `flowforge`.

**Given** a canvas route activates,
**When** the main content area renders,
**Then** the correct canvas component mounts within ≤200ms,
**And** the previous canvas unmounts cleanly (no leaked `$effect` subscriptions).

**Given** a canvas is a placeholder (not yet built in later epics),
**When** it renders,
**Then** it displays canvas name, responsible epic number, and "Coming in Epic N" label without console errors.

**Notes:**
- Restructure `MainContent.svelte` as the canvas host
- 9 canvases correspond 1:1 to ActivityBar icons from Story 1.4
- BreadcrumbNav component: `{Canvas Name}` → `{Sub-page}` — hidden at canvas home level
- Svelte 5 runes syntax throughout

---

## Epic 2: AI Providers & Server Connections

### Story 2.0: Provider Infrastructure Audit

**As a** developer starting Epic 2,
**I want** a complete audit of the current provider and server connection state,
**So that** stories 2.1–2.6 build on verified existing code rather than assumptions.

**Acceptance Criteria:**

**Given** the `src/` backend,
**When** the audit runs,
**Then** a findings document covers: (a) existing provider config files or classes, (b) any hardcoded API keys or model names in agent code, (c) current `ProvidersPanel.svelte` implementation state (partial implementation known), (d) existing server connection config (if any), (e) how Claude Agent SDK is currently initialised (if at all).

**Notes:**
- Scan: `src/agents/`, `src/services/`, `quantmind-ide/src/lib/components/settings/ProvidersPanel.svelte`
- Read-only exploration — no code changes

---

### Story 2.1: Provider Configuration Storage & Schema

**As a** trader setting up QUANTMINDX for the first time,
**I want** a secure, typed storage schema for AI provider credentials,
**So that** all downstream agent components retrieve provider settings from a single source of truth without touching `.env` directly.

**Acceptance Criteria:**

**Given** no provider has been configured,
**When** the backend initialises,
**Then** a providers config store is created at `config/providers.db` (SQLite) with schema: `{ id, provider_type, display_name, api_key_encrypted, base_url, model_list, tier_assignment, is_active, created_at_utc }`.

**Given** an API key is saved,
**When** written to the store,
**Then** it is encrypted at rest using Fernet keyed to a machine-local secret (not hardcoded),
**And** the raw key never appears in logs or API responses.

**Notes:**
- Supported provider types: `anthropic`, `openrouter`, `openai`, `custom_openai_compatible`
- Tier assignment: `{ floor_manager: model_name, dept_heads: model_name, sub_agents: model_name }`
- Encryption: `cryptography` package, Fernet, keyed to machine UUID
- NFR-S3: API keys never in source code

---

### Story 2.2: Providers & Servers API Endpoints

**As a** frontend developer building the Providers UI,
**I want** a complete CRUD API for AI providers and server connections,
**So that** the UI can perform all operations without direct file system access.

**Acceptance Criteria:**

**Given** the providers API is running,
**When** `GET /api/providers` is called,
**Then** it returns all configured providers with `{ id, display_name, provider_type, is_active, tier_assignment, model_count }` — no API keys.

**Given** `POST /api/providers` is called with a valid payload,
**When** processed,
**Then** provider is created, UUID assigned, encrypted key stored, `201 Created` returned with `{ id, display_name }`.

**Given** `PUT /api/providers/{id}` is called,
**When** processed,
**Then** only provided fields are updated; if `api_key` is absent, existing key is preserved.

**Given** `DELETE /api/providers/{id}` is called for an active provider,
**When** processed,
**Then** `409 Conflict` is returned explaining the provider is in use.

**Given** `POST /api/providers/{id}/test` is called,
**When** the test executes,
**Then** a minimal API call fires to the provider,
**And** returns `{ success: true, latency_ms, model_count }` or `{ success: false, error }`.

**Notes:**
- Server connection config (Cloudzy/Contabo hostnames, ports) follows same CRUD pattern under `/api/servers`
- All endpoints under `contabo_routers/providers.py` (NODE_ROLE=contabo or local)
- NFR-M7: router-per-module pattern

---

### Story 2.3: Claude Agent SDK Provider Routing

**As a** developer building the agent platform,
**I want** the Claude Agent SDK initialised with the active provider configuration at runtime,
**So that** all agents route through the configured provider without hardcoded keys anywhere.

**Acceptance Criteria:**

**Given** at least one active provider exists with tier assignments,
**When** the agent runtime initialises,
**Then** Claude Agent SDK is instantiated using the active provider's decrypted key and base URL,
**And** model assignment follows tier config: FloorManager → Opus tier, Department Heads → Sonnet tier, Sub-agents → Haiku tier.

**Given** no provider is configured or all are inactive,
**When** any agent tries to initialise,
**Then** a `ProviderNotConfiguredError` is raised with a human-readable message,
**And** the UI shows: "No AI provider configured — go to Settings → Providers."

**Notes:**
- Claude Agent SDK only — not LangChain, not LangGraph, not raw `anthropic` client
- NFR-M1, NFR-M2: Claude Agent SDK is the sole canonical agent runtime
- This story is a hard dependency for Epics 5, 7, 8

---

### Story 2.4: Provider Hot-Swap Without Restart

**As a** trader changing AI providers at runtime,
**I want** the agent runtime to reload its SDK client when the active provider changes,
**So that** I can swap providers without restarting the server (FR66).

**Acceptance Criteria:**

**Given** an active provider is in use,
**When** a different provider is activated via `PUT /api/providers/{id}` with `{ is_active: true }`,
**Then** the agent runtime detects the change within ≤5 seconds,
**And** the SDK client is re-initialised with the new provider's credentials,
**And** any in-flight agent tasks complete on the old provider before the swap.

**Given** the new provider fails to initialise (bad key, unreachable),
**When** the hot-swap fails,
**Then** the runtime falls back to the previous active provider,
**And** an error notification fires: "Provider swap failed — continuing on [previous provider]."

**Notes:**
- FR66: runtime provider swap without system restart
- Implement via a `ProviderRegistry` singleton with a `reload()` method triggered by config change event
- NFR-I1: failures queue rather than crash

---

### Story 2.5: ProvidersPanel UI — Add, Edit, Test, Delete

**As a** trader configuring QUANTMINDX,
**I want** a ProvidersPanel where I can add, edit, test, and remove AI providers,
**So that** I can manage all LLM connections without touching config files.

**Acceptance Criteria:**

**Given** I navigate to Settings → Providers,
**When** the ProvidersPanel loads,
**Then** all configured providers are listed with: display name, provider type badge, tier assignment, active toggle, test button, edit button, delete button,
**And** the panel uses Tier 2 glass aesthetic.

**Given** I click "Add Provider" and enter valid credentials,
**When** I click "Test Connection",
**Then** a spinner shows during the test,
**And** on success: green checkmark + latency + model count,
**And** on failure: red error message with provider's error text.

**Given** I attempt to delete a provider that is in use,
**When** the API returns `409 Conflict`,
**Then** the UI shows a modal: "Provider in use — remove references first."

**Notes:**
- Read existing `ProvidersPanel.svelte` before editing (partial implementation from Story 2.0 audit)
- All icons: Lucide only
- Optimistic toggle update with rollback on error

---

### Story 2.6: Server Connection Configuration Panel

**As a** trader managing QUANTMINDX deployment,
**I want** a Server Connections panel to configure Cloudzy and Contabo node hostnames, ports, and health-check intervals,
**So that** StatusBand health indicators and system management tools know where each node lives.

**Acceptance Criteria:**

**Given** I open Settings → Servers,
**When** the panel loads,
**Then** it lists both nodes (Cloudzy — live trading, Contabo — agents/compute) with current connection status.

**Given** I configure a server entry and click save,
**When** the config is stored,
**Then** the StatusBand health indicator for that node updates within 10 seconds.

**Given** I click "Test Connection",
**When** the test runs,
**Then** the backend pings the node's `/health` endpoint and returns round-trip latency in ms.

**Notes:**
- Health-check polling interval: configurable 5s–60s, default 15s
- Do not store SSH private keys — only API credentials and HTTP endpoints
- Feeds into StatusBand Story 1.5 health indicators

---

## Epic 3: Live Trading Command Center

### Story 3.0: Live Trading Backend & MT5 Bridge Audit

**As a** developer starting Epic 3,
**I want** a complete audit of the live trading backend, MT5 bridge, WebSocket infrastructure, and kill switch state,
**So that** stories 3.1–3.7 target real existing code rather than building blindly.

**Acceptance Criteria:**

**Given** the backend in `src/`,
**When** the audit runs,
**Then** a findings document covers: (a) MT5 ZMQ bridge implementation state and connection management, (b) existing kill switch classes (ProgressiveKillSwitch, SmartKillSwitch, BotCircuitBreaker) and their API exposure, (c) WebSocket endpoint(s) for position/P&L streaming, (d) Sentinel → Governor → Commander pipeline wiring state, (e) session mask and Islamic compliance implementation, (f) existing `TradingFloorPanel.svelte` and live trading UI component state.

**Notes:**
- Scan: `src/router/`, `src/trading/`, `src/bridge/`, `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
- Confirm: Cloudzy-only routers include kill switch, MT5 bridge, execution, sentinel
- Read-only exploration — no code changes

---

### Story 3.1: WebSocket Position & P&L Streaming Backend

**As a** developer wiring the live trading canvas,
**I want** the WebSocket endpoint verified and streaming live position, P&L, and regime data,
**So that** the frontend canvas receives real-time data with ≤3s lag (NFR-P2).

**Acceptance Criteria:**

**Given** the MT5 bridge is connected and Cloudzy is running,
**When** a client connects to the `/ws/trading` WebSocket endpoint,
**Then** it receives a stream of events: `position_update`, `pnl_update`, `regime_change`, `bot_status_change`,
**And** each event includes `timestamp_utc` (never naive datetime).

**Given** the MT5 ZMQ connection drops,
**When** the disconnection is detected within 10 seconds (NFR-R5),
**Then** automatic reconnection with exponential backoff fires,
**And** the WebSocket clients receive a `bridge_status` event: `{ status: "reconnecting", last_known_utc }`.

**Given** WebSocket clients disconnect and reconnect,
**When** reconnection occurs,
**Then** the last known state is replayed before live updates resume (NFR-R3).

**Notes:**
- UTC everywhere — MT5 tick timestamps normalized to UTC at ingest
- Architecture: Cloudzy WebSocket only — this must work without Contabo reachable (NFR-R4)
- Tiered tick storage: Hot → PostgreSQL, Warm → DuckDB, Cold → Parquet

---

### Story 3.2: Kill Switch Backend — All Tiers

**As a** risk manager,
**I want** all three kill switch tiers implemented as atomic, audited backend operations,
**So that** each tier executes completely and correctly (NFR-P1 — correctness over speed).

**Acceptance Criteria:**

**Given** Tier 1 (Soft Stop — no new entries) is activated,
**When** the command fires,
**Then** all EAs stop opening new positions immediately,
**And** existing positions remain open,
**And** an immutable audit log entry is written: `{ tier: 1, activated_at_utc, activator }` (NFR-D2).

**Given** Tier 2 (Strategy Pause — pause specific strategies) is activated with strategy IDs,
**When** the command fires,
**Then** only specified strategies pause,
**And** other strategies continue normally.

**Given** Tier 3 (Emergency Close — close all positions) is activated,
**When** the command fires,
**Then** close orders are sent for all open positions via MT5 bridge,
**And** results (filled/partial/rejected) are captured and returned,
**And** the audit log records each close attempt and outcome.

**Notes:**
- NFR-P1: kill switch protocol executes in full, in order — never skip steps
- Architecture: kill switch runs self-contained on Cloudzy even if Contabo is unreachable
- ProgressiveKillSwitch class exists — verify, extend, do NOT rebuild

---

### Story 3.3: Session Mask, Islamic Compliance & Loss Cap APIs

**As a** developer wiring the live trading canvas,
**I want** API endpoints exposing EA session mask state, Islamic compliance status, and daily loss cap usage,
**So that** the canvas can display these parameters transparently without modifying EA behaviour.

**Acceptance Criteria:**

**Given** `GET /api/trading/bots/{id}/params` is called,
**When** processed,
**Then** it returns: `{ session_mask, force_close_hour, overnight_hold, daily_loss_cap, current_loss_pct, islamic_compliance, swap_free }`.

**Given** the current time is within 60 minutes of the force-close time (21:45 UTC for Islamic mode),
**When** the endpoint is polled,
**Then** the response includes `{ force_close_countdown_seconds }`.

**Given** a daily loss cap is breached,
**When** the event fires,
**Then** an audit log entry records the breach,
**And** a notification event fires to the frontend.

**Notes:**
- FR8: daily loss cap enforcement — display only, EAs enforce autonomously via MQL5 inputs
- FR39: Islamic compliance display (force-close 21:45 UTC)
- EA parameters are read-only in the ITT — no independent overrides (FR40)
- All timestamps: `_utc` suffix mandatory (naming convention from architecture)

---

### Story 3.4: Live Trading Canvas — Layout, Bot Status Grid & Streaming UI

**As a** trader monitoring active bots,
**I want** the Live Trading canvas to display all active bot statuses, live P&L, and open positions streaming in real time,
**So that** I have complete operational awareness of every running EA at a glance.

**Acceptance Criteria:**

**Given** I navigate to the Live Trading canvas (default home canvas),
**When** the canvas loads,
**Then** a GlassTile grid renders with one tile per active EA,
**And** each tile shows: EA name, symbol, current P&L (colour-coded), open position count, current regime, session active/inactive,
**And** all tiles use Tier 2 glass (`rgba(8,13,20,0.35)`, `blur(16px)`).

**Given** the WebSocket stream is active,
**When** position or P&L data arrives,
**Then** the affected tiles update within ≤3s,
**And** P&L changes flash green (`#00c896`, 100ms) or red (`#ff3b3b`, 100ms).

**Given** I expand a bot tile (click),
**When** the sub-page opens,
**Then** BreadcrumbNav appears: `Live Trading > [EA Name]`,
**And** the sub-page shows: session mask (active sessions highlighted), force_close_hour, overnight_hold, daily loss cap bar, current equity exposure.

**Notes:**
- NFR-P2: ≤3s lag on live data
- GlassTile component: Tier 2 glass, header + metric body + timestamp footer
- Data transition: skeleton pulse until first data arrives — no jarring blank-then-populated

---

### Story 3.5: Kill Switch UI — All Tiers

**As a** trader managing risk,
**I want** all kill switch tiers accessible from the TopBar and Live Trading canvas,
**So that** I can stop any level of activity at any moment from any canvas.

**Acceptance Criteria:**

**Given** I am on any canvas,
**When** the TradingKillSwitch renders in TopBar,
**Then** it shows Lucide `shield-alert` icon in ready state (grey),
**And** clicking it arms the switch (red pulse, 2s countdown visible),
**And** a second click (or Enter does NOT work — must click Confirm button) opens the confirmation modal.

**Given** the confirm modal is open,
**When** I select a tier and confirm,
**Then** the appropriate kill switch tier API fires,
**And** the TopBar switch shows "FIRED" state (grey, disabled),
**And** recovery requires app restart (TradingKillSwitch scope: Cloudzy MT5 only).

**Given** Tier 3 (Emergency Close) is selected,
**When** the double-confirmation modal renders,
**Then** it shows current open positions, estimated exposure, and "This will close all positions" warning in red border.

**Notes:**
- TradingKillSwitch is TopBar ONLY — never elsewhere (architecture hard rule)
- Distinct from Workflow Kill Switch (per-workflow in FlowForge, Epic 11)
- `aria-label` updates reactively: "Emergency stop — click to arm" → "Armed — click Confirm" → "Trading stopped"

---

### Story 3.6: Manual Trade Controls UI

**As a** trader who needs to intervene manually,
**I want** to manually close any open position directly from the Live Trading canvas,
**So that** I am never locked out of controlling my own trades.

**Acceptance Criteria:**

**Given** a position is open in a bot tile,
**When** I click "Close Position",
**Then** a confirmation modal shows: symbol, direction, current P&L, lot size,
**And** on confirm, `POST /api/trading/close` fires with the position ticket.

**Given** the close order executes,
**When** MT5 responds,
**Then** the result (filled price, slippage, final P&L) shows in the bot tile,
**And** the position disappears from the grid if fully closed.

**Given** I click "Close All" and confirm the double-confirmation modal,
**When** close orders fire,
**Then** a summary modal shows results per position (filled/partial/rejected).

**Notes:**
- FR3: manual position close from Live Trading canvas
- Audit log entry required for every manual close
- Cross-canvas 3-dot contextual menu on any EA tile → "Close Position" also triggers this flow

---

### Story 3.7: MorningDigestCard & Degraded Mode Rendering

**As a** trader opening the ITT at the start of the trading day,
**I want** a Morning Digest card on first load and graceful degraded mode when Contabo is offline,
**So that** I get the overnight summary immediately and the UI never shows error screens during node drops.

**Acceptance Criteria:**

**Given** it is the first time I open Live Trading canvas in a session,
**When** the canvas loads,
**Then** a MorningDigestCard renders at the top of the tile grid,
**And** it shows: overnight agent activity summary, pending approvals count (with chip), node health, critical alerts, active market session indicator.

**Given** Contabo is unreachable,
**When** the Live Trading canvas renders,
**Then** all Contabo-dependent data (agent activity, workflow state) shows degraded indicators — amber label "Contabo offline — retrying",
**And** Cloudzy WebSocket data continues displaying normally,
**And** TradingKillSwitch renders normally (Cloudzy-independent),
**And** NO blank screens or error states appear anywhere.

**Given** Contabo reconnects,
**When** the connection resumes,
**Then** degraded indicators clear automatically within 10 seconds,
**And** live data resumes from the reconnection point.

**Notes:**
- NFR-R4: strategy router + kill switch operational on Cloudzy even without Contabo
- Architecture: "Nothing disappears — all last-known data remains visible. QUANTMINDX never shows blank screen."
- MorningDigestCard pre-loads via `/morning-digest` Prefect-triggered aggregation on Contabo at session start

---

## Epic 4: Risk Management & Compliance

### Story 4.0: Risk Pipeline State Audit

**As a** developer starting Epic 4,
**I want** a complete audit of the existing risk pipeline implementation state,
**So that** stories 4.1–4.7 wire the UI to verified working backend components without rebuilding production-ready code.

**Acceptance Criteria:**

**Given** the backend in `src/`,
**When** the audit runs,
**Then** a findings document covers: (a) PhysicsAwareKellyEngine — current API, inputs, outputs, (b) Ising Model, Lyapunov, HMM sensor state and API exposure, (c) Governor, Sentinel, Commander wiring, (d) BotCircuitBreaker configuration interface, (e) existing CalendarGovernor or EnhancedGovernor classes, (f) existing prop firm registry entries, (g) any existing risk API endpoints.

**Notes:**
- Architecture constraint: do NOT modify PhysicsAwareKellyEngine, Ising Model, Lyapunov, HMM, BotCircuitBreaker, Governor, Sentinel — these are production-ready
- Scan: `src/risk/`, `src/router/`, `src/trading/`
- Read-only exploration — no code changes

---

### Story 4.1: CalendarGovernor — News Blackout & Calendar-Aware Trading Rules

**As a** risk system enforcing event-driven rules,
**I want** a CalendarGovernor mixin that applies economic calendar rules — lot scaling, entry pauses, post-event reactivation — per account or strategy configuration,
**So that** NFP, ECB, CPI, and other scheduled events automatically adjust position sizing and entry eligibility (FR77).

**Acceptance Criteria:**

**Given** a high-impact news event is within the configured blackout window (e.g., 30 minutes before),
**When** the CalendarGovernor evaluates,
**Then** affected strategies are paused or lot-scaled per the account's calendar rule configuration,
**And** the rule activation is logged to the audit trail.

**Given** the event window passes,
**When** the post-event reactivation timer fires,
**Then** strategies resume normal operation automatically,
**And** resumption is logged to the audit trail.

**Given** Journey 45 (The Calendar Gate) scenario: NFP every Friday,
**When** the cron triggers at Thu 18:00 (EUR/USD scalpers 0.5x), Fri 13:15 (all pause), Fri 14:00 (regime-check reactivate), Fri 15:00 (normal),
**Then** each phase activates correctly via the CalendarGovernor.

**Notes:**
- FR77: calendar-aware trading rules
- Extend existing `EnhancedGovernor` — add CalendarGovernor mixin, do NOT rebuild Governor
- Calendar events sourced from economic calendar API or manual entry
- FTMO layer: no positions within 2 hours of Tier 1 news

---

### Story 4.2: Risk Parameters & Prop Firm Registry APIs

**As a** developer wiring the Risk canvas,
**I want** API endpoints for reading and writing risk parameters and prop firm registry entries,
**So that** the Risk canvas UI can display live state and allow configuration changes.

**Acceptance Criteria:**

**Given** `GET /api/risk/params/{account_tag}` is called,
**When** processed,
**Then** it returns current risk params: `{ daily_loss_cap_pct, max_trades_per_day, kelly_fraction, position_multiplier, lyapunov_threshold, hmm_retrain_trigger }`.

**Given** `PUT /api/risk/params/{account_tag}` is called with updated values,
**When** processed,
**Then** only provided fields are updated,
**And** the change takes effect on the next risk evaluation cycle (≤30 seconds),
**And** the change is written to the risk audit layer.

**Given** `GET /api/risk/prop-firms` is called,
**When** processed,
**Then** all configured prop firm entries are returned with their rule sets.

**Given** `POST /api/risk/prop-firms` is called with a new prop firm config,
**When** processed,
**Then** the entry is created and available in the routing matrix account tag assignment.

**Notes:**
- FR41: per account tag risk config
- FR68: prop firm registry CRUD
- Out-of-range values (Kelly fraction > 1.0) rejected with 422 Unprocessable Entity

---

### Story 4.3: Strategy Router & Regime State APIs

**As a** developer wiring the Risk canvas,
**I want** API endpoints for the current regime classification and strategy router state,
**So that** the Risk canvas displays why strategies are active or paused.

**Acceptance Criteria:**

**Given** `GET /api/risk/regime` is called,
**When** processed,
**Then** it returns: `{ regime, confidence_pct, transition_at_utc, previous_regime, active_strategy_count, paused_strategy_count }`.

**Given** `GET /api/risk/router/state` is called,
**When** processed,
**Then** it returns per-strategy router state: `{ strategy_id, status, pause_reason, eligible_regimes }`.

**Given** `GET /api/risk/physics` is called,
**When** processed,
**Then** it returns current Ising Model, Lyapunov, and HMM outputs with their alert states.

**Notes:**
- FR5: regime classification always visible
- FR32–FR35: physics sensor outputs
- HMM in shadow mode — outputs logged but not controlling router until validated (FR33)

---

### Story 4.4: Backtest Results API

**As a** developer wiring the Risk canvas backtest results viewer,
**I want** API endpoints to list and retrieve backtest results,
**So that** results from all 6 confirmed working modes are accessible to the frontend.

**Acceptance Criteria:**

**Given** `GET /api/backtests` is called,
**When** processed,
**Then** it returns all completed backtests: `{ id, ea_name, mode, run_at_utc, net_pnl, sharpe, max_drawdown, win_rate }`.

**Given** `GET /api/backtests/{id}` is called,
**When** processed,
**Then** it returns full backtest detail including equity curve data points, trade distribution, and mode-specific parameters.

**Given** a backtest is in progress,
**When** `GET /api/backtests/running` is called,
**Then** it returns running backtests with progress pct and partial metrics.

**Notes:**
- Architecture: backtest engine (`src/backtesting/`) is production-ready — do NOT modify
- 6 modes: VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C

---

### Story 4.5: Risk Canvas — Physics Sensor Tiles & Live Dashboard

**As a** trader monitoring risk,
**I want** the Risk canvas to display physics pipeline stages as live tiles with visual representations,
**So that** Ising Model, Lyapunov, HMM shadow mode, and Kelly Engine states are visible at a glance.

**Acceptance Criteria:**

**Given** I navigate to the Risk canvas,
**When** the canvas loads,
**Then** it shows PhysicsSensorTile components: Ising Model (magnetization chart + correlation matrix), Lyapunov Exponent (divergence rate + chaos metric), HMM (regime state + transition probabilities with "shadow mode" badge), Kelly Engine (current fraction + physics multiplier + house-of-money state).

**Given** a physics sensor enters an alert state,
**When** the tile renders,
**Then** it shows red `#ff3b3b` border + Lucide `alert-triangle` + inline explanation,
**And** the alert persists until the condition resolves.

**Given** PhysicsSensorTile variants render,
**When** sensor type is `scalar` or `time-series` or `distribution`,
**Then** the appropriate visualization renders: sparkline for time-series, bar for scalar, histogram for distribution.

**Notes:**
- FR32–FR35: physics sensors (do NOT rebuild — wire UI to existing API)
- NFR-R1: component failure must not cascade — each tile fetches independently
- Updates: poll `/api/risk/physics` at 5-second interval

---

### Story 4.6: Risk Canvas — Compliance, Prop Firm, Calendar & Backtest Viewer

**As a** trader configuring risk and reviewing backtest performance,
**I want** compliance rules, prop firm configuration, calendar gate status, and backtest results visible on the Risk canvas,
**So that** all risk configuration and historical performance is managed from one place.

**Acceptance Criteria:**

**Given** I view the compliance tile on the Risk canvas,
**When** it renders,
**Then** it shows: BotCircuitBreaker state per account tag, prop firm rules (drawdown limit, daily halt conditions), Islamic compliance status (force-close countdown if within 60 minutes of 21:45 UTC).

**Given** I open the prop firm configuration sub-page,
**When** it renders,
**Then** it shows all registered prop firm entries with editable fields,
**And** saving calls `PUT /api/risk/prop-firms/{id}`.

**Given** I view the calendar gate tile,
**When** it renders,
**Then** upcoming high-impact news events are listed with configured blackout windows,
**And** currently active blackout windows show which strategies are affected.

**Given** I navigate to Backtest Results sub-page,
**When** I select a backtest,
**Then** equity curve and drawdown charts render with the 6-mode result matrix.

**Notes:**
- FR37: BotCircuitBreaker display
- FR38: Governor compliance rules display
- FR68: prop firm registry CRUD
- Journey 10: Challenge Mode — StatusBand shows challenge progress indicator

---

## Epic 5: Unified Memory & Copilot Core

### Story 5.0: Memory Architecture & Copilot Infrastructure Audit

**As a** developer starting Epic 5,
**I want** a complete audit of the current memory systems, Copilot wiring, and agent infrastructure,
**So that** stories 5.1–5.9 consolidate verified existing code rather than creating parallel systems.

**Acceptance Criteria:**

**Given** the backend in `src/`,
**When** the audit runs,
**Then** a findings document covers: (a) all 4 memory systems (agent memory, global memory, graph memory `src/memory/graph/`, department markdown files) — completeness state, (b) current FloorManager class and its initialisation state, (c) existing `session_checkpoint_service.py` wiring, (d) any existing CanvasContextTemplate files, (e) current Workshop canvas implementation state, (f) current AgentPanel component state.

**Notes:**
- Architecture: Graph Memory is 80–90% done (needs columns + ReflectionExecutor + embeddings)
- Scan: `src/memory/`, `src/agents/`, `quantmind-ide/src/lib/components/`
- Read-only exploration — no code changes

---

### Story 5.1: Graph Memory Completion — ReflectionExecutor, OPINION Nodes & Embeddings

**As a** developer building the persistent agent memory layer,
**I want** the Graph Memory system completed with ReflectionExecutor, OPINION Node pattern, and vector embeddings,
**So that** all agents have a consistent, queryable memory backbone for cross-session knowledge (FR14).

**Acceptance Criteria:**

**Given** the existing `src/memory/graph/` implementation (80–90% done),
**When** the completion work runs,
**Then** the required columns are added (`session_status`, `embedding`, agent-specific fields identified in audit),
**And** all timestamps have `_utc` suffix (naming convention).

**Given** an agent takes a consequential action (state change, artifact, routing decision, assessment, approval/rejection),
**When** the action completes,
**Then** an OPINION node is written with: `action`, `reasoning`, `confidence`, `alternatives_considered`, `constraints_applied`, `agent_role`,
**And** the OPINION node has at least one `SUPPORTED_BY` edge to evidence (OBSERVATION, WORLD, or DECISION node).

**Given** the ReflectionExecutor runs after a session settles,
**When** it processes session memories,
**Then** episodic and semantic memories are extracted from the session's draft nodes,
**And** validated memories are promoted from `session_status='draft'` to `session_status='committed'`,
**And** downstream agents can load committed OPINIONs without re-deriving reasoning.

**Given** vector embeddings are added,
**When** a semantic memory query runs,
**Then** relevant committed nodes are retrieved via embedding similarity,
**And** retrieval uses `all-MiniLM-L6-v2` sentence-transformers (ChromaDB backend).

**Notes:**
- Architecture prerequisite: Graph Memory MUST be complete before Canvas Context System (Story 5.3)
- 14 node types: WORLD, BANK, OBSERVATION, OPINION, WORKING, PERSONA, PROCEDURAL, EPISODIC, CONVERSATION, MESSAGE, AGENT, DEPARTMENT, TASK, SESSION, DECISION
- Session isolation: writes tagged `session_id + session_status='draft'` — never dirty reads from interactive sessions

---

### Story 5.2: Session Checkpoint → Graph Memory Commit Flow

**As a** developer building persistent agent memory,
**I want** the session checkpoint service wired to the graph memory commit flow,
**So that** agent work persists across sessions and is not lost on disconnect.

**Acceptance Criteria:**

**Given** an agent session is active,
**When** a checkpoint fires (configurable interval or on agent milestone),
**Then** all `session_status='draft'` nodes for the session are evaluated by ReflectionExecutor,
**And** nodes passing quality threshold are promoted to `session_status='committed'`,
**And** stale draft nodes are archived or discarded.

**Given** a session ends unexpectedly (crash, disconnect),
**When** the agent restarts,
**Then** it loads all committed nodes from the last checkpoint,
**And** resumes work from the last known committed state.

**Notes:**
- FR14: persistent agent memory across sessions
- `src/agents/memory/session_checkpoint_service.py` exists — wire to graph commit, do not rebuild
- Session tiers: HOT (<1hr working memory), WARM (<30 days), COLD (archived)

---

### Story 5.3: Canvas Context System — CanvasContextTemplate per Department

**As a** developer building the Agent Panel,
**I want** a CanvasContextTemplate system that pre-loads the correct department head with relevant memory scope when a chat opens on any canvas,
**So that** agents are context-aware from the first message rather than starting blind.

**Acceptance Criteria:**

**Given** I open the Agent Panel on the Risk canvas,
**When** a new chat initialises,
**Then** the Risk Department Head is pre-loaded with: active risk params, current regime, recent audit events, and relevant committed OPINION nodes from graph memory.

**Given** I open the Agent Panel on the Live Trading canvas,
**When** a new chat initialises,
**Then** the FloorManager or Trading Department Head is pre-loaded with: active bot states, current session mask, recent P&L, and trading-relevant memory nodes.

**Given** memory pre-loading runs,
**When** context is assembled,
**Then** only minimum high-signal tokens load (identifiers first, content fetched JIT via tools),
**And** total pre-loaded context does not exceed the configured token budget.

**Notes:**
- Architecture prerequisite: this story depends on Story 5.1 (Graph Memory) being complete
- FR20: canvas-aware Copilot context (CanvasContextTemplate per department)
- Architecture hard rule: "No memory pre-loading — load identifiers, fetch content JIT via tools"
- CanvasContextTemplate YAML configs stored under `flows/directives/`

---

### Story 5.4: FloorManager Agent Wiring

**As a** developer wiring the Copilot,
**I want** the Copilot to route messages through the actual FloorManager agent (Claude Opus via Claude Agent SDK),
**So that** user messages receive intelligent responses and department delegation begins working.

**Acceptance Criteria:**

**Given** a configured and active Opus-tier provider exists (from Epic 2),
**When** I send a message via Copilot,
**Then** the message passes to the FloorManager agent instance via Claude Agent SDK,
**And** the FloorManager responds using the Opus-tier model.

**Given** FloorManager receives a task belonging to a department (e.g., "run research on GBPUSD"),
**When** it routes the task,
**Then** the appropriate Department Head agent is invoked via Redis Streams task dispatch,
**And** a routing notification appears in the conversation: "Delegating to Research Department."

**Given** the agent SDK call fails (timeout, rate limit, provider error),
**When** the error occurs,
**Then** the conversation shows: "Agent error: [reason]. Retry?" with a retry button,
**And** the error is logged to the audit trail,
**And** the system degrades gracefully — never crashes (NFR-I1).

**Notes:**
- FR10, FR11: FloorManager orchestration — Opus tier
- Claude Agent SDK only — no raw `anthropic` client calls outside the provider routing module
- Three-tier priority routing: HIGH/MEDIUM/LOW via session_id namespace

---

### Story 5.5: Copilot Panel — Conversation Thread & Streaming

**As a** trader using the QUANTMINDX AI assistant,
**I want** the Agent Panel to render a persistent conversation thread with streaming responses,
**So that** I can chat with the AI system in real time.

**Acceptance Criteria:**

**Given** the Agent Panel is open (320px collapsible right rail),
**When** it renders,
**Then** conversation history shows: user messages (right-aligned, amber tint), AI responses (left-aligned, cyan `#00d4ff` accent), timestamps (IBM Plex Mono 12px).

**Given** the FloorManager streams a response,
**When** SSE tokens arrive,
**Then** the response renders token-by-token with a typing cursor (`|`, 600ms blink) at stream end,
**And** the panel auto-scrolls to keep latest content visible,
**And** scrolling up pauses auto-scroll; new message resumes it.

**Given** the FloorManager uses a tool,
**When** the tool call renders inline,
**Then** it shows "Using: [tool_name]…" with pulsing dot → collapses to `✓` on completion.

**Given** I send a follow-up message,
**When** processed,
**Then** conversation history is included in context (prior turns sent to FloorManager).

**Notes:**
- NFR-P3: Copilot first token within ≤5 seconds
- Streaming pattern: Svelte 5 `$state` array appended per SSE chunk — targeted DOM update, no full re-render
- Agent Panel exists on ALL canvases — conversation persists across canvas switches

---

### Story 5.6: Copilot Kill Switch

**As a** trader who needs to immediately stop AI activity,
**I want** a dedicated Copilot kill switch that halts all agent activity without affecting live trades,
**So that** I can stop runaway AI loops without triggering the trading kill switch.

**Acceptance Criteria:**

**Given** the Copilot kill switch is activated (separate from TradingKillSwitch),
**When** confirmed,
**Then** all running FloorManager and department agent tasks are terminated,
**And** the Agent Panel shows "Agent activity suspended" in amber,
**And** live trading on Cloudzy continues unaffected.

**Given** the kill switch is activated while the AI is mid-task,
**When** termination fires,
**Then** partial results are preserved in the conversation thread with "Task interrupted" marker,
**And** no partial state changes are committed to graph memory.

**Notes:**
- FR43: Copilot kill switch — distinct from TradingKillSwitch
- Two kill switches are architecturally independent and must never be conflated in the UI
- Recovery: "Resume" button in Agent Panel reactivates the agent system

---

### Story 5.7: NL System Commands & Context-Aware Canvas Binding

**As a** trader controlling the system through Copilot,
**I want** to issue system commands in natural language with canvas context automatically applied,
**So that** I can operate the platform conversationally regardless of which canvas I am on.

**Acceptance Criteria:**

**Given** I type "pause GBPUSD strategy" in Copilot,
**When** FloorManager interprets the command,
**Then** it identifies intent as `STRATEGY_PAUSE`, confirms: "I'll pause the GBPUSD strategy. Confirm?",
**And** on confirmation, executes the pause via the risk API.

**Given** I am on the Live Trading canvas,
**When** I ask "what are my open positions?",
**Then** FloorManager resolves the question against live trading data (canvas context binding).

**Given** the command is ambiguous,
**When** FloorManager cannot determine intent with high confidence,
**Then** it asks a clarifying question rather than executing blindly.

**Notes:**
- FR20: canvas-aware context via CanvasContextTemplate (Story 5.3)
- FR46: all agent decisions log reasoning for transparency
- Canvas context passed as metadata: `{ message, canvas_context: "live-trading", session_id }`
- All destructive commands (pause, close, stop) require in-conversation confirmation

---

### Story 5.8: Workshop Canvas — Full Copilot Home UI

**As a** trader using the Workshop canvas,
**I want** the Workshop canvas implemented as the FloorManager's full-screen home (Claude.ai-inspired),
**So that** I have a dedicated space for extended AI conversations, morning digest, skill browsing, and memory exploration.

**Acceptance Criteria:**

**Given** I navigate to the Workshop canvas,
**When** it loads,
**Then** it shows: left sidebar (New Chat, History, Projects/Workflows, Memory, Skills) + centered Copilot input + conversation above.

**Given** it is the first Workshop open of the day,
**When** the canvas loads,
**Then** FloorManager auto-triggers `/morning-digest`,
**And** the morning digest (overnight agent activity, pending approvals, market outlook, critical alerts) renders as the first message.

**Given** I browse Skills in the left sidebar,
**When** the skill browser opens,
**Then** all registered skills show with name, description, slash command, and usage count.

**Given** I browse Memory in the left sidebar,
**When** the memory explorer opens,
**Then** committed graph memory nodes are browsable (list or tree view) filtered by node type.

**Notes:**
- FR14: memory exploration (graph memory from Story 5.1)
- FR13: Copilot workflow/cron creation via natural language in Workshop
- Rich media rendering: diagrams, tables, charts inline; file references clickable
- Suggestion chips: 3–5 CAG+RAG-powered action chips at bottom of input area

---

### Story 5.9: Suggestion Chips & Cross-Canvas Entity Navigation

**As a** trader using the Agent Panel and Workshop,
**I want** context-aware suggestion chips and cross-canvas entity navigation via 3-dot menus,
**So that** common actions are one click away and I can navigate to any entity from anywhere.

**Acceptance Criteria:**

**Given** I open the Agent Panel on the Risk canvas,
**When** the SuggestionChipBar renders,
**Then** it shows 3–5 chips relevant to risk context: `/kelly-settings`, `/reduce-exposure`, `/drawdown-review`,
**And** chips update dynamically as live system state changes.

**Given** I open the Agent Panel on the Live Trading canvas,
**When** the canvas is loaded,
**Then** suggestion chips surface: `/morning-digest`, `/show-positions`, `/pause-strategy [name]`.

**Given** I hover over any EA tile, strategy card, or workflow card,
**When** the 3-dot (Lucide `more-horizontal`) contextual menu reveals,
**Then** it offers cross-canvas shortcuts: "View Code" → Development canvas, "View Performance" → Trading canvas, "View History" → Portfolio canvas,
**And** clicking a shortcut navigates both canvas and sub-page to the entity detail.

**Notes:**
- Suggestion chips: CAG+RAG-powered — CanvasContextTemplate provides the context (Story 5.3)
- Cross-canvas navigation: entity is the link — navigation follows the thing, not memory of last canvas
- SuggestionChipBar: horizontally scrollable, Tier 1 glass pill, Lucide icon + label

---

## Epic 6: Knowledge & Research Engine

### Story 6.0: Knowledge Infrastructure Audit

**As a** developer starting Epic 6,
**I want** a complete audit of the current knowledge base infrastructure,
**So that** stories 6.1–6.7 wire the UI to verified existing components without rebuilding production services.

**Acceptance Criteria:**

**Given** the backend in `src/`,
**When** the audit runs,
**Then** a findings document covers: (a) PageIndex Docker instances (articles, books, logs) — running state, endpoints, document counts, (b) ChromaDB setup and sentence-transformer model configuration, (c) current video ingest pipeline (`src/video_ingest/`) state, (d) Firecrawl integration state, (e) existing knowledge API endpoints, (f) any existing news feed infrastructure.

**Notes:**
- PageIndex is primary knowledge system — 3 Docker instances (articles, books, logs)
- ChromaDB + `all-MiniLM-L6-v2` for semantic search
- Scan: `src/knowledge/`, `src/video_ingest/`, Docker compose configs
- Read-only exploration — no code changes

---

### Story 6.1: PageIndex Integration & Knowledge API

**As a** research developer,
**I want** the 3 PageIndex Docker instances integrated with unified knowledge API endpoints,
**So that** agents and the Research canvas can query all knowledge sources in parallel.

**Acceptance Criteria:**

**Given** the 3 PageIndex instances are running (articles, books, logs),
**When** `GET /api/knowledge/sources` is called,
**Then** all 3 sources are listed with `{ id, type, status, document_count }`.

**Given** a search query is submitted to `POST /api/knowledge/search`,
**When** it executes,
**Then** the query fans out to all 3 PageIndex instances in parallel,
**And** results are merged and ranked by relevance score,
**And** each result includes: `{ source_type, title, excerpt, relevance_score, provenance }`.

**Given** a PageIndex instance is offline,
**When** the fanout query runs,
**Then** the offline instance is skipped with a warning in the response,
**And** results from remaining instances are returned.

**Notes:**
- FR44: full-text + semantic indexing
- FR46: provenance metadata preserved on every entry
- NFR-I4: knowledge pipeline failures logged and retried without blocking other operations

---

### Story 6.2: Web Scraping, Article Ingestion & Personal Knowledge API

**As a** knowledge base operator,
**I want** Firecrawl web scraping and personal knowledge ingestion wired with provenance,
**So that** any content source can be indexed into the knowledge base with full traceability.

**Acceptance Criteria:**

**Given** `POST /api/knowledge/ingest` is called with a URL,
**When** Firecrawl scrapes the content,
**Then** the article is chunked, embedded, and indexed into PageIndex `articles` instance,
**And** provenance metadata `{ source_url, scraped_at_utc, relevance_tags }` is stored.

**Given** `POST /api/knowledge/personal` is called with a note, PDF, or Obsidian export,
**When** it is processed,
**Then** the content is indexed into a `personal` partition,
**And** it is tagged `{ type: "personal", created_at_utc, source_description }`.

**Given** an ingestion fails (rate limit, timeout),
**When** the error occurs,
**Then** it retries with exponential backoff (max 3 attempts),
**And** the failure is logged (NFR-I4).

**Notes:**
- FR42: web scraping + article ingestion (Firecrawl)
- FR45: personal knowledge entries
- FR48: knowledge base partitioning by type (scraped, personal, system-generated)

---

### Story 6.3: Live News Feed & Geopolitical Sub-agent Backend

**As a** trader monitoring macro events,
**I want** a live news feed pipeline with sub-90-second news-to-alert latency and a geopolitical sub-agent,
**So that** unscheduled events (ECB signals, geopolitical shocks) are detected and surfaced within 90 seconds (FR50, Journey 46).

**Acceptance Criteria:**

**Given** the news feed provider (Finnhub/FMP) is configured,
**When** a new high-impact news event arrives,
**Then** the geopolitical sub-agent classifies it: `{ impact_tier: HIGH|MEDIUM|LOW, affected_symbols, event_type }`,
**And** the classification arrives within ≤90 seconds of publication.

**Given** a HIGH-severity event is classified,
**When** the alert fires,
**Then** `POST /api/news/alert` stores the event,
**And** a notification event is broadcast to connected WebSocket clients,
**And** the affected strategy exposure is calculated and included in the alert payload.

**Given** the news feed provider is offline,
**When** the pipeline runs,
**Then** it retries with exponential backoff and logs the outage (NFR-I4).

**Notes:**
- FR50: live news feed with enriched macro context
- Journey 46: unscheduled ECB signal caught in <90 seconds
- Geopolitical sub-agent: Haiku tier via Claude Agent SDK
- `src/knowledge/news/` — verify existing scaffolding before building

---

### Story 6.4: Research Canvas — Knowledge Query Interface

**As a** trader querying the knowledge base,
**I want** the Research canvas to provide a search interface over the full knowledge base,
**So that** I can directly query all indexed content without going through Copilot.

**Acceptance Criteria:**

**Given** I navigate to the Research canvas,
**When** the canvas loads,
**Then** it shows: search bar, source filter (articles / books / logs / personal / all), hypothesis pipeline tile, video ingest entry tile, knowledge base tile.

**Given** I enter a query and press Enter,
**When** the search executes,
**Then** results appear within ≤2 seconds,
**And** each result shows: title, source type badge, relevance score, excerpt, "View Full" action.

**Given** I click "View Full",
**When** the detail sub-page opens,
**Then** the full document renders with BreadcrumbNav,
**And** a "Send to Copilot" button passes the document as context to the active conversation.

**Notes:**
- FR43: semantic knowledge base query
- FR49: NL knowledge query via Copilot (Story 5.7 handles this)
- Research canvas also houses the hypothesis pipeline entry (visible tile)

---

### Story 6.5: YouTube Video Ingest UI & Pipeline Tracking

**As a** researcher building the knowledge base,
**I want** to paste a YouTube URL and track the ingest pipeline step by step,
**So that** video content is indexed and searchable without leaving the ITT.

**Acceptance Criteria:**

**Given** I paste a YouTube URL into the Research canvas ingest field,
**When** I click "Ingest" (or auto-trigger on paste),
**Then** `POST /api/knowledge/video-ingest` fires with the URL.

**Given** ingestion is in progress,
**When** the progress tracker renders,
**Then** it shows step-by-step stages: Downloading → Transcribing → Chunking → Embedding → Indexing,
**And** the current stage pulses cyan (Fragment Mono progress label).

**Given** ingestion completes,
**When** the indexed document is ready,
**Then** it appears in the knowledge base with source type "YouTube" and the video title,
**And** it is immediately searchable via Story 6.4.

**Notes:**
- FR47: video ingest pipeline
- Alpha Forge Workflow 1 entry point: YouTube URL paste on Research canvas triggers the full pipeline
- VIDEO_INGEST is stage 1 of the Alpha Forge workflow orchestrator — wire to existing stage

---

### Story 6.6: Live News Feed Tile & News Canvas Integration

**As a** trader monitoring macro context during active sessions,
**I want** a live news feed tile on the Live Trading canvas and a full news view on the Research canvas,
**So that** macro events are visible during trading and researchable in context.

**Acceptance Criteria:**

**Given** the Live Trading canvas is open,
**When** the news feed tile renders,
**Then** it shows the latest 5 news items: headline, source, timestamp (UTC), impact_tier badge.

**Given** a HIGH-impact news item arrives,
**When** the tile updates,
**Then** the new item flashes amber `#f0a500` border (400ms),
**And** the affected strategy exposure count shows inline: "8 EUR strategies exposed."

**Given** I navigate to Research canvas,
**When** the full news view opens,
**Then** historical news with impact assessments is browsable,
**And** I can filter by impact tier, symbol, date range.

**Notes:**
- FR50: live news feed
- Journey 46: HIGH severity event caught < 90 seconds by geopolitical sub-agent (Story 6.3)
- News tile on Live Trading canvas is a GlassTile pulling from `/api/news/feed`

---

### Story 6.7: Shared Assets Canvas

**As a** developer and trader managing cross-departmental resources,
**I want** the Shared Assets canvas (Canvas 7) to provide a browsable library of docs, templates, indicators, skills, MCP configs, and flow components,
**So that** all reusable assets are discoverable and accessible from one place.

**Acceptance Criteria:**

**Given** I navigate to the Shared Assets canvas,
**When** the canvas loads,
**Then** a GlassTile grid renders with tiles grouped by asset type: Docs, Strategy Templates, Indicators, Skills, Flow Components, MCP Configs.

**Given** I click an asset tile,
**When** the sub-page opens,
**Then** it shows: asset name, type, version, usage count (how many workflows reference it), last updated, BreadcrumbNav.

**Given** I click a code asset (indicator, flow component),
**When** the detail view opens,
**Then** a MonacoEditorStub renders in read mode (Python/MQL5 syntax highlighting),
**And** an "Edit" button switches to edit mode with Save/Diff actions.

**Notes:**
- Assets are generated by agents and registered here — UI is browse/read primarily
- No create-from-UI workflows for assets — backend (agents) generate assets; UI is the read surface
- Usage tracking: how many workflows reference each skill/component — surfaced from workflow metadata

---

## Epic 7: Department Agent Platform — Departments Do Real Work

### Story 7.0: Department System Audit

**As a** developer starting Epic 7,
**I want** a complete audit of all department agent implementations and supporting infrastructure,
**So that** stories 7.1–7.10 convert stubs to real implementations with precise knowledge of what exists.

**Acceptance Criteria:**

**Given** `src/agents/departments/`,
**When** the audit runs,
**Then** a findings document covers: (a) each Department Head class — stub vs. real implementation percentage, (b) all SubAgent types defined and their implementation state, (c) existing skill registry and skill files, (d) current Redis Streams integration state (vs. SQLite DepartmentMailService), (e) session workspace isolation current state, (f) concurrent task routing current implementation.

**Notes:**
- Architecture: department heads are ~80% stubs — this audit identifies the exact 20% that's real
- No `paper_trader` or `live_trader` SubAgentType (deprecated — agents MONITOR; EAs execute via MT5)
- Scan: `src/agents/departments/`, `src/agents/mail/`, `src/agents/skills/`
- Read-only exploration — no code changes

---

### Story 7.1: Research Department — Real Hypothesis Generation

**As a** researcher using QUANTMINDX,
**I want** the Research Department Head to generate real market hypotheses using knowledge base + web research,
**So that** research pipeline produces actionable intelligence rather than stub responses.

**Acceptance Criteria:**

**Given** FloorManager delegates a research task,
**When** the Research Department Head processes it,
**Then** it queries PageIndex (full-text), ChromaDB (semantic), and web research tools,
**And** returns a structured hypothesis: `{ symbol, timeframe, hypothesis, supporting_evidence[], confidence_score, recommended_next_steps }`.

**Given** confidence ≥ 0.75,
**When** the Department Head evaluates,
**Then** it recommends escalating to TRD Generation,
**And** the recommendation appears in the conversation thread with a "Proceed to TRD?" prompt.

**Given** Research spawns sub-agents for parallel research,
**When** sub-agents execute (Haiku tier via Claude Agent SDK),
**Then** results merge into a coherent output,
**And** the merge is visible: "Research complete — [N] sub-agents contributed."

**Notes:**
- FR24: Research hypothesis generation
- OPINION node written after consequential research action (Story 5.1 prerequisite)
- No `paper_trader` sub-agents — research sub-agents only (data_researcher, market_analyst, etc.)

---

### Story 7.2: Development Department — Real MQL5 EA Code Generation

**As a** strategy developer,
**I want** the Development Department Head to generate real MQL5 EA code from TRD documents,
**So that** the strategy factory pipeline produces actual tradeable EA files.

**Acceptance Criteria:**

**Given** a validated TRD document is passed to Development,
**When** the Development Department Head processes it,
**Then** it generates a complete MQL5 `.mq5` file with: EA header, `OnInit()`, `OnTick()`, `OnDeinit()`, and all parameters from the TRD (session_mask, force_close_hour, overnight_hold, daily_loss_cap, spread_filter, etc.).

**Given** the TRD is ambiguous on a parameter,
**When** the Development agent encounters ambiguity,
**Then** it flags it and asks FloorManager for clarification rather than guessing.

**Given** the EA file is generated,
**When** it is saved,
**Then** it is stored in the EA output directory with a version number,
**And** a compile trigger fires (Story 7.3).

**Notes:**
- FR25: MQL5 EA code generation
- First Version Rule: first EA from a video source must mirror the source strategy (Vanilla = mirror image)
- EA Tag Architecture parameters must all be present: session_mask, force_close_hour, overnight_hold, etc.
- NFR-M3: Python files under 500 lines

---

### Story 7.3: MQL5 Compilation Integration

**As a** developer building the strategy factory,
**I want** the Development Department to trigger MQL5 compilation after EA code generation,
**So that** compile errors are caught immediately and the `.ex5` file is ready for backtesting.

**Acceptance Criteria:**

**Given** a new `.mq5` file is generated,
**When** the compile step triggers,
**Then** the MetaEditor compiler is invoked via the MT5 bridge or Docker-based MT5 compiler on Contabo,
**And** compilation output (errors, warnings) is captured and returned.

**Given** compilation succeeds,
**When** `.ex5` is produced,
**Then** it is stored alongside `.mq5` in the EA output directory,
**And** the strategy record updates: `compile_status: success`.

**Given** compilation fails,
**When** error output is received,
**Then** the Development Department analyses errors and attempts auto-correction (≤2 iterations),
**And** if still failing, escalates to FloorManager with error detail.

**Notes:**
- FR25: MQL5 compilation
- Architecture: Docker MT5 compiler on Contabo (compilation + testing only — no live trading)
- Compilation is a hard dependency for backtesting

---

### Story 7.4: Skill Catalogue — Registry, Authoring & Skill Forge

**As a** developer building the agent skill system,
**I want** a complete skill catalogue with 12 core skills registered, and the Skill Forge enabling agents to author and register new skills,
**So that** agents compose reusable capabilities rather than reinventing logic (FR18, FR19).

**Acceptance Criteria:**

**Given** the skill registry initialises,
**When** it loads,
**Then** 12 core skills are registered: `financial_data_fetch`, `pattern_scanner`, `statistical_edge`, `hypothesis_document_writer`, `mql5_generator`, `backtest_launcher`, `news_classifier`, `risk_evaluator`, `report_writer`, `strategy_optimizer`, `institutional_data_fetch`, `calendar_gate_check`.

**Given** a Department Head identifies a repeated workflow pattern,
**When** Skill Forge authoring triggers,
**Then** the Department Head produces a `skill.md` file defining: name, description, inputs, outputs, SOP steps,
**And** the skill is registered in the skill catalogue with a version number,
**And** it is immediately available to all department agents.

**Given** a skill is registered,
**When** `GET /api/skills` is called,
**Then** all skills return with `{ name, description, slash_command, version, usage_count }`.

**Notes:**
- FR18: skill registration + execution
- FR19: skill authoring by Department Heads
- Skills stored as `skill.md` files under `shared_assets/skills/`
- NFR-M5: new agent capabilities added via skill registration, not modification of core dept code
- ReflectionExecutor (Story 5.1) reviews skill quality before commit

---

### Story 7.5: Session Workspace Isolation for Concurrent Sessions

**As a** developer building multi-session agent coordination,
**I want** session workspace isolation so multiple concurrent Alpha Forge or research sessions don't contaminate each other,
**So that** parallel work produces clean, independent outputs.

**Acceptance Criteria:**

**Given** two concurrent research sessions are running on different strategies,
**When** both sessions write to graph memory,
**Then** each write is tagged with its own `session_id + session_status='draft'`,
**And** neither session reads the other's draft nodes.

**Given** a Department Head completes a session,
**When** it commits its work,
**Then** committed nodes become available to all subsequent sessions via `session_status='committed'`,
**And** the commit log records: `{ session_id, committed_at_utc, node_count, department }`.

**Given** concurrent Alpha Forge sessions work on the same strategy,
**When** both try to commit,
**Then** the second commit waits for DeptHead review (like a merge review),
**And** conflicts are flagged for FloorManager resolution.

**Notes:**
- Architecture: session workspace isolation prevents context poisoning
- Multiple concurrent sessions with `session_id` namespace
- DeptHead evaluates + commits — session isolation is a FloorManager orchestration concern

---

### Story 7.6: Department Mail — Redis Streams Migration

**As a** developer building reliable agent communication,
**I want** Department Mail to use Redis Streams for inter-department messaging,
**So that** message delivery is reliable, ordered, and auditable (FR15).

**Acceptance Criteria:**

**Given** a Department Head sends a message to another department,
**When** it is dispatched,
**Then** it publishes to a Redis Stream with namespace: `mail:dept:{recipient}:{workflow_id}`,
**And** payload includes: `sender`, `recipient`, `message_type`, `payload`, `timestamp_utc`.

**Given** a department subscribes to its stream,
**When** a message arrives,
**Then** it is consumed within ≤500ms,
**And** the consumer group acknowledges receipt.

**Given** a department agent is offline when a message arrives,
**When** it comes back online,
**Then** it reads all unacknowledged messages from the stream in order (no message loss).

**Notes:**
- Architecture: Department Mail currently SQLite-backed (obsolete) — migrate ALL uses to Redis Streams
- Redis key namespacing: `dept:{name}:wf_{id}:queue`, `mail:dept:{name}:wf_{id}` (colon-separated, lowercase)
- Forbidden pattern: `writing to department_mail.db in new code` — use Redis Streams only

---

### Story 7.7: Concurrent Task Routing — ≥5 Simultaneous Tasks

**As a** trader running complex multi-domain requests,
**I want** FloorManager to route and manage at least 5 simultaneous department tasks,
**So that** research, development, risk, trading, and portfolio tasks proceed in parallel (FR21).

**Acceptance Criteria:**

**Given** 5 tasks are submitted simultaneously,
**When** FloorManager routes them,
**Then** each is dispatched to the appropriate department concurrently via Redis Streams,
**And** task status is visible in the Agent Panel: "Research: running | Development: queued | Risk: running…"

**Given** all 5 tasks complete,
**When** results consolidate,
**Then** total wall-clock time ≤ max(individual_task_times) × 1.2 (parallelism overhead ≤20%).

**Notes:**
- FR21: concurrent task routing ≥5 tasks
- FR12: Department Head parallel sub-agent spawning (15 defined SubAgent types)
- Architecture: FloorManager uses three-tier priority (HIGH/MEDIUM/LOW) with session_id namespacing

---

### Story 7.8: Risk, Trading & Portfolio Department — Real Implementations

**As a** developer completing the department agent platform,
**I want** the Risk, Trading, and Portfolio Department Heads converted from stubs to real implementations,
**So that** evaluation, paper trade monitoring, and portfolio reporting work end-to-end.

**Acceptance Criteria:**

**Given** a compiled EA passes the SIT gate,
**When** the Risk Department Head initiates backtesting,
**Then** all 6 modes queue and execute (Architecture: backtest engine production-ready — reuse),
**And** a pass/fail verdict generates: ≥4/6 modes must pass (Sharpe ≥1.0, max_drawdown ≤15%, win_rate ≥50%).

**Given** a strategy is in paper trading,
**When** the Trading Department Head monitors it,
**Then** it tracks P&L per trade, win/loss ratio, drawdown, avg hold time, regime correlation,
**And** Copilot receives periodic updates: "Paper trading: 12 trades, +2.3% P&L, 65% win rate."

**Given** the Portfolio Department Head generates a report,
**When** `GET /api/portfolio/report` is called,
**Then** it returns: total equity, P&L attribution per strategy, P&L attribution per broker, drawdown per account.

**Notes:**
- FR27: full 6-mode backtest matrix (existing engine — wire only)
- FR28: Risk evaluation + human approval gate
- FR29: paper trading monitoring
- FR36b: Portfolio Department monitors account equity vs. initial deposit threshold and propagates house-of-money status system-wide — adjusts Governor risk scalar, Commander strategy activation profile, Alpha Forge promotion criteria, and Copilot contextual awareness accordingly

---

### Story 7.9: Department Kanban Sub-page UI

**As a** trader reviewing department activity,
**I want** a Department Kanban sub-page on each canvas showing the current task queue for the relevant department,
**So that** I can see what agents are working on in real time.

**Acceptance Criteria:**

**Given** I navigate to the Research canvas and click "Research Department Tasks",
**When** the Kanban sub-page opens,
**Then** it shows a 4-column DepartmentKanbanCard: TODO / IN_PROGRESS / BLOCKED / DONE,
**And** each card shows: task name, dept badge, priority badge (HIGH=red, MEDIUM=amber, LOW=grey), duration.

**Given** a task state changes via SSE,
**When** the update arrives,
**Then** the card moves to the new column with a 400ms cyan border flash,
**And** no full re-render — targeted DOM update via Svelte 5 `$state`.

**Notes:**
- FR17: agent activity + task queue review
- DepartmentKanbanCard component: 4-column state machine, real-time SSE updates
- Each department's kanban lives on its primary canvas: Research dept → Research canvas, Development dept → Development canvas, Risk dept → Risk canvas, Trading dept → Trading canvas, Portfolio dept → Portfolio canvas

---

## Epic 8: Alpha Forge — Strategy Factory

### Story 8.0: Alpha Forge Pipeline Audit

**As a** developer starting Epic 8,
**I want** a complete audit of the Alpha Forge pipeline orchestration state,
**So that** stories 8.1–8.10 wire real implementations rather than rebuilding existing pipeline stages.

**Acceptance Criteria:**

**Given** `flows/assembled/alpha_forge_flow.py` and `src/`,
**When** the audit runs,
**Then** a findings document covers: (a) all 9 pipeline stages and their implementation state (VIDEO_INGEST → RESEARCH → TRD → DEVELOPMENT → COMPILE → BACKTEST → VALIDATION → EA_LIFECYCLE → APPROVAL), (b) Prefect workflow registration state, (c) existing strategy version control schema, (d) existing fast-track template library, (e) current EA deployment pipeline state (`flows/assembled/ea_deployment_flow.py`).

**Notes:**
- Architecture: `workflow_orchestrator.py` stages exist — this epic wires departments (now real from Epic 7) through the orchestrator
- Scan: `flows/`, `src/agents/`, `src/backtesting/`
- Read-only exploration — no code changes

---

### Story 8.1: Alpha Forge Orchestrator — Wiring Departments Through Pipeline Stages

**As a** developer building the strategy factory,
**I want** the Alpha Forge Prefect workflow wired to call real department implementations,
**So that** the full pipeline operates end-to-end: source → paper trading.

**Acceptance Criteria:**

**Given** a YouTube URL is submitted to the Alpha Forge pipeline,
**When** Workflow 1 (Creation) runs,
**Then** VIDEO_INGEST → RESEARCH (Research Dept, now real) → TRD_GENERATION → DEVELOPMENT (Dev Dept, now real) → COMPILE → basic BACKTEST stages execute in sequence,
**And** each stage writes its result to Prefect `workflows.db` for durability.

**Given** a compiled EA passes basic backtest,
**When** Workflow 2 (Enhancement Loop) triggers,
**Then** full 6-mode backtest → SIT gate → paper trading → human approval → live deployment stages execute,
**And** the human approval gate (PENDING_REVIEW state) halts the pipeline until confirmed.

**Given** any stage fails,
**When** the failure occurs,
**Then** Prefect marks the workflow FAILED,
**And** Mubarak is notified via Copilot: "Alpha Forge stage [name] failed — [reason]. Review?"

**Notes:**
- FR23: complete Alpha Forge loop without mandatory manual intervention at intermediate stages
- Architecture: DOE methodology — Directive (what) + Orchestration (who) + Execution (how)
- First Version Rule: first EA must mirror source strategy; spiced variants in subsequent loops only

---

### Story 8.2: TRD Generation Stage

**As a** strategy developer,
**I want** the TRD Generation stage to produce structured, complete implementation specifications,
**So that** the Development department has an unambiguous spec to code from (FR24).

**Acceptance Criteria:**

**Given** a research hypothesis passes confidence threshold,
**When** TRD_GENERATION stage fires,
**Then** the Research Department produces a TRD with: strategy name, symbol, timeframe, entry/exit conditions, position sizing rules, risk parameters, full EA input parameter spec (session_mask, force_close_hour, overnight_hold, daily_loss_cap, spread_filter, etc.).

**Given** the TRD is validated,
**When** the TRD validator runs,
**Then** all required MQL5 parameter fields are present,
**And** if any are missing, the TRD is rejected with a list of missing fields for Research to fill.

**Notes:**
- TRD is the contract between Research and Development — quality here prevents compile failures
- TRD stored in strategy version control with unique strategy ID
- Islamic compliance parameters (force_close_hour, overnight_hold) must always be present

---

### Story 8.3: Fast-Track Event Workflow — Template Library & Matching

**As a** trader responding to market events,
**I want** a strategy template library and fast-track matching so that I can deploy a relevant strategy within 11 minutes of a hot news event,
**So that** I can act on event-driven opportunities without waiting for the full 72-hour Alpha Forge loop (Journey 21).

**Acceptance Criteria:**

**Given** the strategy template library contains at least 3 templates,
**When** `GET /api/alpha-forge/templates` is called,
**Then** all templates return with: name, strategy_type, applicable_events, risk_profile, avg_deployment_time.

**Given** a HIGH-impact news event fires (from Story 6.3),
**When** the template matching runs,
**Then** matching templates are ranked by `{ template_name, confidence_score, estimated_deployment_time }`,
**And** the Copilot surfaces: "Fast-track available — GBPUSD event strategy template matches. Deploy? [11 min]"

**Given** Mubarak approves fast-track deployment,
**When** the fast-track Prefect workflow runs,
**Then** the strategy is live-deployed within 15 minutes with: conservative lot sizing, auto-expiry tag, Islamic compliance parameters,
**And** the deployment is a full pipeline run (compile → SIT gate only — no full 6-mode backtest).

**Notes:**
- FR3 (Story 3.3), Journey 21 (The Template Pull)
- Fast-track requires SIT gate (MODE_C) pass minimum — not the full 4/6 mode threshold
- Template library initially seeded manually; Research dept adds templates over time

---

### Story 8.4: Strategy Version Control & Rollback API

**As a** strategy developer managing EA evolution,
**I want** all strategy artifacts version-controlled with one-instruction rollback,
**So that** I can safely iterate without losing previous working versions (FR75).

**Acceptance Criteria:**

**Given** a new strategy version is created,
**When** it is saved,
**Then** it is assigned a semantic version (or sequential ID),
**And** all artifacts are linked: `.mq5`, `.ex5`, TRD, backtest results.

**Given** "Rollback [strategy] to version [N]" is issued via Copilot,
**When** FloorManager processes it,
**Then** all artifacts from version N are restored,
**And** the strategy is re-compiled and SIT-validated,
**And** rollback is recorded in the audit log.

**Notes:**
- FR75: one-instruction rollback
- NFR-D3: provenance chain preserved on all updates
- Internal versioning system — not git (strategies are living documents, not code commits)

---

### Story 8.5: Human Approval Gates Backend

**As a** trader responsible for deployment decisions,
**I want** mandatory human approval gates at post-backtest and pre-live checkpoints,
**So that** no strategy deploys without my explicit sign-off (FR28, FR30).

**Acceptance Criteria:**

**Given** a strategy passes the backtest gate (≥4/6 modes),
**When** the PENDING_REVIEW state is reached,
**Then** an `ApprovalGateBadge` appears in the TopBar with the strategy name,
**And** Mubarak receives a Copilot notification with: strategy name, backtest summary, key metrics, "Approve / Reject / Request Revision" options.

**Given** Mubarak approves,
**When** approval is recorded,
**Then** the approval event is stored as an immutable audit record: `{ strategy_id, approver, approved_at_utc, gate_type, metrics_snapshot }` (NFR-D2).

**Given** Mubarak requests revision,
**When** the revision request is submitted,
**Then** the agent re-executes with Mubarak's feedback as context,
**And** a new approval gate is created — no manual editing of agent artifacts.

**Notes:**
- Approval gate max hold: 15 minutes before `PENDING_REVIEW → EXPIRED_REVIEW` (7-day timeout before considered non-blocking)
- FR28: human approval required before paper trading AND before live deployment
- ApprovalGateBadge pattern: batch-surfaced, accumulates within workflow run

---

### Story 8.6: EA Deployment Pipeline — MT5 Registration

**As a** developer completing the strategy factory,
**I want** the EA deployment pipeline to move a compiled EA from storage to live operation on the MT5 terminal,
**So that** approved strategies go live automatically after human sign-off (FR79).

**Acceptance Criteria:**

**Given** Mubarak approves live deployment,
**When** the EA_LIFECYCLE stage fires,
**Then** the deployment pipeline executes in sequence: (1) file transfer to Cloudzy, (2) MT5 terminal registration, (3) credential/parameter injection (session_mask, force_close_hour, etc.), (4) health check (ACTIVE state + first tick received), (5) ZMQ stream registration.

**Given** any deployment step fails,
**When** the failure occurs,
**Then** the pipeline halts at the failed step,
**And** Mubarak is notified: "EA deployment failed at [step] — [reason]."

**Given** deployment succeeds,
**When** health check confirms ACTIVE,
**Then** the strategy appears on the Live Trading canvas with a "New" badge,
**And** a deployment audit record is written.

**Notes:**
- FR79: EA deployment pipeline (file → chart attachment → parameter injection → health check → ZMQ)
- Architecture: comprehensive try/except blocks mandatory before any Cloudzy deployment path
- Deploy window: Friday 22:00 – Sunday 22:00 UTC (never during open market hours with active positions)

---

### Story 8.7: Alpha Forge Canvas — Pipeline Status Board

**As a** strategy developer,
**I want** the Alpha Forge pipeline status visible on the Development canvas,
**So that** I can track every strategy run from URL to deployment.

**Acceptance Criteria:**

**Given** I navigate to the Development canvas,
**When** the canvas loads,
**Then** the Alpha Forge pipeline board shows one row per active strategy run,
**And** each row shows: strategy name, current stage (9-stage pipeline), stage status (running/passed/failed/waiting).

**Given** a pipeline stage completes,
**When** the status updates,
**Then** the stage animates from "running" (cyan pulse) to "passed" (cyan checkmark),
**And** the next stage activates automatically if no human gate is required.

**Given** the pipeline hits a human approval gate (PENDING_REVIEW),
**When** the row renders,
**Then** it shows amber "Awaiting Approval" badge + ApprovalGateBadge in TopBar.

**Notes:**
- Pipeline board pulls state from Prefect API (workflows.db on Contabo)
- 9 stages: VIDEO_INGEST → RESEARCH → TRD → DEVELOPMENT → COMPILE → BACKTEST → VALIDATION → EA_LIFECYCLE → APPROVAL

---

### Story 8.8: Development Canvas — EA Variant Browser & Monaco Editor

**As a** strategy developer reviewing EA code,
**I want** an EA variant browser and Monaco editor on the Development canvas,
**So that** I can review variant code, compare backtest results, and track improvement cycle history.

**Acceptance Criteria:**

**Given** I navigate to the Development canvas,
**When** the EA library tile opens,
**Then** a variant browser grid shows: vanilla/spiced/mode_b/mode_c per strategy with backtest summary per variant.

**Given** I click a variant,
**When** the sub-page opens,
**Then** MonacoEditorStub renders with MQL5 syntax highlighting in read mode,
**And** improvement cycle history shows a version timeline (v1 → v2 → v3),
**And** promotion status tracker shows the pipeline stage (paper trading → SIT → live approval).

**Given** I click "Edit" on a code file,
**When** Monaco switches to edit mode,
**Then** Save, Run (triggers compile), and Diff (vs previous version) actions appear in the action bar.

**Notes:**
- MonacoEditorStub: MQL5 + Python language support, language selector top-right, file breadcrumb top-left
- Improvement cycle history = version timeline from Strategy Version Control (Story 8.4)
- Read mode by default — edit mode requires explicit click (intentional friction for live EA code)

---

### Story 8.9: A/B Race Board, Cross-Strategy Loss Propagation & Provenance Chain

**As a** trader managing strategy variants,
**I want** the A/B race board, cross-strategy loss propagation, and EA provenance chain visible,
**So that** I can empirically select winning variants, manage correlated risk, and trace any EA's origin.

**Acceptance Criteria:**

**Given** two strategy variants are in paper trading,
**When** I open the A/B comparison view on the Development canvas,
**Then** side-by-side metrics show: P&L, trade count, drawdown, Sharpe — updating in real time.

**Given** statistical significance emerges (≥50 trades, p < 0.05),
**When** the analysis updates,
**Then** the winning variant shows an amber crown indicator,
**And** Copilot notifies: "Variant B has statistically significant edge. Recommend promoting Variant B."

**Given** Strategy A hits its daily loss cap,
**When** the loss event fires,
**Then** all strategies with correlation ≥ 0.5 to Strategy A receive tightened risk params (Kelly fraction × 0.75),
**And** a loss propagation event records in the risk audit log (FR76).

**Given** I ask "what's the origin of this EA?",
**When** Copilot queries the provenance chain,
**Then** it traces: source (MQL5 scrape URL / YouTube URL) → Research score → Dev build → code review → approval (FR31).

**Notes:**
- FR74: A/B testing with statistical significance
- FR76: cross-strategy loss propagation
- FR31: versioned strategy library + full provenance chain

---

## Epic 9: Portfolio & Multi-Broker Management

### Story 9.0: Portfolio & Broker Infrastructure Audit

**As a** developer starting Epic 9,
**I want** a complete audit of the current broker registry, routing matrix, and portfolio data state,
**So that** stories 9.1–9.5 build on verified existing infrastructure.

**Acceptance Criteria:**

**Given** the backend in `src/`,
**When** the audit runs,
**Then** a findings document covers: (a) existing broker registry implementation and schema, (b) RoutingMatrix class state and API exposure, (c) existing portfolio metrics API endpoints, (d) MT5 account auto-detection implementation, (e) correlation matrix computation state.

**Notes:**
- Architecture: RoutingMatrix + broker registry stay untouched — extend only for fee-awareness
- Scan: `src/broker/`, `src/portfolio/`, `src/router/routing_matrix.py`
- Read-only exploration — no code changes

---

### Story 9.1: Broker Account Registry & Routing Matrix API

**As a** trader managing multiple broker accounts,
**I want** API endpoints for broker account registration and routing matrix configuration,
**So that** the Portfolio canvas can manage all accounts and route strategies correctly.

**Acceptance Criteria:**

**Given** `POST /api/brokers` is called with account details,
**When** processed,
**Then** the account is registered with: `{ id, broker_name, account_number, account_type, account_tag, mt5_server, login_encrypted, swap_free, leverage }`,
**And** MT5 auto-detection runs: `{ broker, account_type, leverage, currency }`.

**Given** `GET /api/routing-matrix` is called,
**When** processed,
**Then** it returns the full matrix of strategies × broker accounts with current assignment state.

**Given** a routing rule is configured,
**When** `PUT /api/routing-matrix/rules` is called,
**Then** the rule is stored: assign strategy by `{ account_tag, regime_filter, strategy_type }`.

**Notes:**
- FR51: broker account registry (≥4 accounts)
- FR52: MT5 account auto-detection
- FR53: routing matrix assignment
- Swap-free accounts required for Islamic compliance

---

### Story 9.2: Portfolio Metrics & Attribution API

**As a** developer wiring the Portfolio canvas,
**I want** API endpoints for portfolio-level metrics and P&L attribution,
**So that** the canvas can display aggregate and per-strategy performance.

**Acceptance Criteria:**

**Given** `GET /api/portfolio/summary` is called,
**When** processed,
**Then** it returns: `{ total_equity, daily_pnl, daily_pnl_pct, total_drawdown, active_strategies, accounts[] }`.

**Given** `GET /api/portfolio/attribution` is called,
**When** processed,
**Then** it returns P&L attribution per strategy and per broker account.

**Given** `GET /api/portfolio/correlation` is called,
**When** processed,
**Then** it returns an N×N correlation matrix of strategy returns with `{ strategy_a, strategy_b, correlation, period_days }`.

**Notes:**
- FR55: portfolio-level performance metrics
- FR54: multi-strategy-type concurrent operation visibility
- Portfolio drawdown > 10% of total equity triggers alert notification

---

### Story 9.3: Portfolio Canvas — Multi-Account Dashboard & Routing UI

**As a** trader managing multiple broker accounts,
**I want** the Portfolio canvas to show all account performance and allow routing matrix configuration,
**So that** I can see portfolio-level health and assign strategies to the right accounts.

**Acceptance Criteria:**

**Given** I navigate to the Portfolio canvas,
**When** the canvas loads,
**Then** it shows a GlassTile grid: one tile per registered broker account (equity, drawdown, exposure),
**And** a portfolio summary tile (total equity, daily P&L, total drawdown).

**Given** I open the routing matrix sub-page,
**When** it renders,
**Then** a matrix shows [strategies] × [broker accounts] with assignment toggles and regime/strategy-type filter dropdowns.

**Given** portfolio drawdown exceeds 10%,
**When** the threshold is breached,
**Then** a red alert banner appears at the top of the canvas,
**And** a Copilot notification fires: "Portfolio drawdown alert: [X]%."

**Notes:**
- FR54: multi-strategy type concurrent operation — visible per account
- Journey 20: dual account (FTMO Challenge $100k + Personal $8k)

---

### Story 9.4: Portfolio Canvas — Attribution, Correlation Matrix & Performance

**As a** trader reviewing portfolio performance,
**I want** P&L attribution, correlation matrix, and per-strategy performance on the Portfolio canvas,
**So that** I understand where returns and risks are concentrated.

**Acceptance Criteria:**

**Given** I open the attribution sub-page,
**When** it renders,
**Then** each strategy shows: equity contribution, P&L contribution, drawdown contribution, % of portfolio, broker account.

**Given** I view the correlation matrix tile,
**When** it renders,
**Then** an N×N heatmap of strategy-to-strategy return correlations displays,
**And** cells with |correlation| ≥ 0.7 highlight in red `#ff3b3b`.

**Given** I hover a correlation cell,
**When** the tooltip renders,
**Then** it shows: strategy A, strategy B, correlation coefficient, data period used.

**Notes:**
- FR55: portfolio-level performance metrics
- Correlation matrix feeds cross-strategy loss propagation (Epic 8 Story 8.9)

---

### Story 9.5: Trading Journal Component

**As a** trader maintaining a trading log,
**I want** a Trading Journal on the Portfolio canvas showing per-bot trade logs with annotation capability,
**So that** I can review individual trades, annotate decisions, and export my trading history.

**Acceptance Criteria:**

**Given** I navigate to Portfolio → Trading Journal,
**When** the sub-page opens,
**Then** it shows a filterable trade log: entry time (UTC), exit time, symbol, direction, P&L, session, hold duration, EA name.

**Given** I click a trade row,
**When** the detail view opens,
**Then** I see: entry/exit prices, spread at entry, slippage, strategy version active at time of trade, notes/annotation field.

**Given** I add a note to a trade,
**When** I save,
**Then** the annotation is stored with `{ trade_id, note, annotated_at_utc }`,
**And** the annotation persists across sessions.

**Given** I click "Export",
**When** the export runs,
**Then** a CSV is downloaded with all filtered trades including annotations.

**Notes:**
- UX spec: Trading Journal tile on Portfolio canvas; active bots (StatusBand) → Portfolio + Trading Journal visible
- Trade data sourced from SQLite trade records (persisted before system acknowledgement, NFR-D1)
- Journey 27: client accounts need per-account trade log filtering

---

## Epic 10: Audit, Monitoring & Notifications

### Story 10.0: Audit Infrastructure Audit

**As a** developer starting Epic 10,
**I want** a complete audit of the existing audit logging, monitoring, and notification infrastructure,
**So that** stories 10.1–10.6 build the query interface over verified existing audit layers.

**Acceptance Criteria:**

**Given** the backend in `src/`,
**When** the audit runs,
**Then** a findings document covers: (a) existing `audit_log.py` and `shared_routers/audit.py` state, (b) 5 audit layer coverage (trade, strategy lifecycle, risk param, agent action, system health), (c) existing notification delivery mechanism, (d) log retention and cold storage sync state, (e) server health monitoring current implementation.

**Notes:**
- Architecture: audit trail cross-cuts everything — every API endpoint, agent task dispatcher, Commander pipeline writes immutable audit entries
- Scan: `src/audit/`, `src/monitoring/`, `shared_routers/`
- Read-only exploration — no code changes

---

### Story 10.1: 5-Layer Audit System — NL Query API

**As a** trader investigating past decisions,
**I want** an API that searches all 5 audit layers and returns causal chains in response to natural language queries,
**So that** "Why was EA_X paused yesterday?" returns a full timestamped explanation.

**Acceptance Criteria:**

**Given** `POST /api/audit/query` is called with `{ query: "Why was EA_X paused at 14:30 yesterday?" }`,
**When** processed,
**Then** the backend searches all 5 layers: trade events, strategy lifecycle, risk param changes, agent actions, system health,
**And** returns a chronological causal chain: `[{ timestamp_utc, layer, event_type, actor, reason }]`.

**Given** the query spans a large time range,
**When** results are ranked,
**Then** events are ordered chronologically and the most causally relevant ones are ranked first.

**Notes:**
- FR60: NL audit trail query
- FR59: all system events logged at appropriate level — no silent exclusions
- NFR-D2: audit log entries are immutable once written — no deletion, no modification

---

### Story 10.2: Agent Reasoning Transparency Log & API

**As a** trader who wants to understand AI decisions,
**I want** an API that retrieves full reasoning chains for any past agent decision,
**So that** no agent action is a black box (FR78).

**Acceptance Criteria:**

**Given** `GET /api/audit/reasoning/{decision_id}` is called,
**When** processed,
**Then** it returns: `{ context_at_decision, model_used, prompt_summary, response_summary, action_taken, opinion_nodes[] }`,
**And** opinion_nodes includes the OPINION node written by the agent (from Story 5.1).

**Given** I ask Copilot "Why did the Research department recommend GBPUSD short?",
**When** FloorManager queries the reasoning log,
**Then** it returns: hypothesis chain, evidence sources, confidence scores at each step, contributing sub-agents.

**Notes:**
- FR78: Copilot explains reasoning for any past recommendation
- OPINION nodes (Story 5.1) are the primary source of agent reasoning
- Journey 51: "The Transparent Reasoning" — full 8-of-31 strategy exposure chain

---

### Story 10.3: Notification Configuration API & Cold Storage Sync

**As a** trader managing alert fatigue,
**I want** configurable notifications and 3-year log retention with cold storage sync,
**So that** I receive actionable alerts without noise, and all logs are permanently accessible.

**Acceptance Criteria:**

**Given** `GET /api/notifications/config` is called,
**When** processed,
**Then** all configurable event types return: `{ event_type, category, severity, enabled, delivery_channel }`.

**Given** `PUT /api/notifications/config/{event_type}` is called,
**When** an event type is toggled off,
**Then** that event type no longer delivers notifications,
**And** events are still written to the audit log (toggle suppresses delivery, not recording).

**Given** high-priority events fire (kill switch activated, daily loss cap hit),
**When** they occur,
**Then** they are always-on (cannot be suppressed by notification config),
**And** they appear in the Copilot thread regardless of active canvas.

**Given** a nightly log sync runs,
**When** it completes,
**Then** logs older than the hot retention window are synced to cold storage on Contabo with integrity verification (NFR-R6, NFR-D5).

**Notes:**
- FR61: 3-year log retention + cold storage sync
- FR62: configurable notifications
- FR63: OS system tray delivery

---

### Story 10.4: NL Audit Query UI & Reasoning Explorer

**As a** trader investigating past system behaviour,
**I want** a natural language audit query interface in the Workshop canvas (via Copilot),
**So that** I can ask "Why was EA_X paused?" and get a readable causal chain.

**Acceptance Criteria:**

**Given** I type "Why was EA_GBPUSD paused at 14:30 yesterday?" in Workshop Copilot,
**When** FloorManager processes the query,
**Then** the response renders as a formatted timeline: `[14:28 UTC] HMM regime → HIGH_VOL → [14:29] Governor tightened limits → [14:30] EA_GBPUSD paused by Commander`.

**Given** I ask "Show me the reasoning for the risk reduction recommendation",
**When** FloorManager fetches the reasoning log,
**Then** it renders the OPINION node chain inline with: confidence scores, evidence sources, action taken.

**Notes:**
- Journey 34: "The Decision Audit"
- Journey 51: "The Transparent Reasoning"
- Rendered inline in Workshop Copilot — not a separate audit screen

---

### Story 10.5: Notification Configuration Panel & Server Health

**As a** trader managing alerts and monitoring infrastructure,
**I want** a Notification Settings panel and a Server Health panel,
**So that** I can manage which events notify me and monitor both Cloudzy and Contabo node health.

**Acceptance Criteria:**

**Given** I open Settings → Notifications,
**When** the panel loads,
**Then** all notifiable event types are listed by category with on/off toggles,
**And** always-on events (kill switch, loss cap, system critical) are shown greyed out with a lock icon.

**Given** I navigate to Settings → Server Health,
**When** the panel loads,
**Then** Cloudzy and Contabo nodes each show: CPU %, memory %, disk %, network latency, uptime, last heartbeat.

**Given** a node metric crosses a threshold (CPU > 85%, disk > 90%),
**When** the threshold is breached,
**Then** the affected metric turns red `#ff3b3b`,
**And** Copilot notifies: "Contabo: disk usage at 91%. Action recommended."

**Notes:**
- FR62: configurable notifications
- FR65: server health monitoring
- Server health feeds into StatusBand health indicators (Epic 1 Story 1.5)

---

## Epic 11: System Management & Resilience

### Story 11.0: Infrastructure & System State Audit

**As a** developer starting Epic 11,
**I want** a complete audit of the current system management infrastructure,
**So that** stories 11.1–11.8 build on verified existing components.

**Acceptance Criteria:**

**Given** the full project,
**When** the audit runs,
**Then** a findings document covers: (a) existing Prefect flows on Contabo and their registration state, (b) systemd service configurations, (c) existing backup/restore scripts, (d) existing rsync cron state (architecture notes this is not yet implemented), (e) Prometheus + Loki + Grafana setup state on Contabo, (f) current theme/wallpaper configuration state.

**Notes:**
- Architecture: Prefect self-hosted on Contabo with SQLite backend (`workflows.db`)
- Architecture gap confirmed: "rsync cron not yet implemented"
- Scan: `flows/`, `systemd/`, `scripts/`, Docker compose files
- Read-only exploration — no code changes

---

### Story 11.1: Nightly Rsync Cron — Cloudzy→Contabo Data Sync

**As a** system operator maintaining data durability,
**I want** a nightly rsync cron from Cloudzy to Contabo,
**So that** trade records, tick data warm storage, and local configs are backed up on a 3-day cadence (NFR-R6).

**Acceptance Criteria:**

**Given** the nightly rsync cron is configured (runs 02:00 UTC),
**When** it executes,
**Then** it syncs: Cloudzy SQLite trade records, warm DuckDB tick data, and local config files to Contabo backup directory.

**Given** the rsync transfer completes,
**When** integrity verification runs,
**Then** file checksums are validated,
**And** corrupted or incomplete transfers are flagged and retried (NFR-D5).

**Given** the rsync fails (Cloudzy unreachable, disk full),
**When** the failure occurs,
**Then** it is logged to the audit trail,
**And** Mubarak receives a notification the next morning: "Nightly rsync failed — [reason]. Manual sync recommended."

**Notes:**
- Architecture: rsync cron is a confirmed gap — not yet implemented
- 3-day backup cadence (not continuous)
- Contabo-only: runs with NODE_ROLE=contabo cron context

---

### Story 11.2: Weekend Compute Protocol — Scheduled Background Tasks

**As a** system operator running compute-intensive weekend analysis,
**I want** the weekend compute protocol to run automatically on Contabo,
**So that** the system self-improves over the weekend without manual intervention (FR71).

**Acceptance Criteria:**

**Given** it is Saturday 00:00 UTC,
**When** the weekend compute trigger fires (Prefect scheduled workflow),
**Then** the following tasks queue on Contabo: full Monte Carlo simulation for all active strategies, HMM model retraining, PageIndex cross-document semantic pass, 90-day correlation refresh.

**Given** a weekend task is running,
**When** I query "What's running this weekend?",
**Then** FloorManager lists all tasks with progress and estimated completion.

**Given** a weekend task fails,
**When** the failure is detected,
**Then** it retries once (exponential backoff),
**And** if retry fails, Mubarak is notified Monday morning.

**Notes:**
- FR71: scheduled background tasks configurable via Copilot
- Contabo-only: NODE_ROLE=contabo tasks
- HMM retraining + Monte Carlo exist in Prefect flows — schedule and wire

---

### Story 11.3: 3-Node Sequential Update & Automatic Rollback

**As a** system administrator updating QUANTMINDX,
**I want** a coordinated sequential update across all 3 nodes with health check and automatic rollback,
**So that** updates are applied safely (FR67).

**Acceptance Criteria:**

**Given** "update all nodes" is issued via Copilot,
**When** the update sequence starts,
**Then** nodes update in order: Contabo → Cloudzy → Desktop (Tauri),
**And** each node health-checks before the next begins.

**Given** a node fails post-update health check,
**When** the health check fails,
**Then** the update sequence halts,
**And** the failed node automatically rolls back to the previous version,
**And** Mubarak is notified: "Cloudzy update failed — rolled back. Contabo on new version."

**Notes:**
- FR67: coordinated 3-node update with automatic rollback
- Deploy window: Friday 22:00 – Sunday 22:00 UTC only (never during open market hours)
- Sequential order is mandatory — not parallel (prevents simultaneous downtime)

---

### Story 11.4: ITT Rebuild Portability — Full Backup & Restore

**As a** system owner,
**I want** the ITT to be fully restorable on a new machine from backup,
**So that** hardware failure does not cause permanent data or configuration loss (FR69).

**Acceptance Criteria:**

**Given** a full system backup has run,
**When** the restore procedure runs on a clean machine,
**Then** all configs (provider credentials, server connections, broker accounts, risk parameters) are restored,
**And** the knowledge base (PageIndex data) is restored,
**And** all strategy artifacts (TRDs, EAs, backtest results, graph memory) are restored.

**Given** a restore completes,
**When** I start the application,
**Then** the system is fully operational without manual re-configuration,
**And** a restore completion report shows in Copilot.

**Notes:**
- FR69: machine portability
- Backup includes: configs, knowledge base, strategy artifacts, graph memory — NOT live trade state (MT5 owns that)
- FR70: server migration without data loss uses the same backup/restore mechanism

---

### Story 11.5: FlowForge Canvas — Prefect Kanban & Node Graph

**As a** trader and developer managing workflows,
**I want** the FlowForge canvas to show all Prefect workflows in a Kanban view with per-workflow kill switches and a node graph viewer,
**So that** I can monitor, control, and understand all running workflows.

**Acceptance Criteria:**

**Given** I navigate to the FlowForge canvas (Canvas 9),
**When** the canvas loads,
**Then** it shows a PrefectKanbanCard layout with 6 columns: PENDING, RUNNING, PENDING_REVIEW, DONE, CANCELLED, EXPIRED_REVIEW.

**Given** a workflow is in RUNNING state,
**When** the card renders,
**Then** it shows: workflow name, dept, state badge (cyan pulse border), duration, step progress (X/Y), next step label,
**And** a Workflow Kill Switch (Lucide `square` "Stop") appears on the card,
**And** clicking it shows a two-step confirmation modal scoped to THAT workflow only.

**Given** I click a workflow card to view its node graph,
**When** the FlowForgeNodeGraph opens,
**Then** an SVG dependency graph renders with: task boxes (coloured by state), directed edges, zoom+pan, minimap,
**And** selecting a node shows task detail tooltip.

**Given** the Workflow Kill Switch fires for one workflow,
**When** it confirms,
**Then** only that Prefect workflow is cancelled,
**And** all other running workflows and live trading on Cloudzy continue unaffected.

**Notes:**
- Workflow Kill Switch is per-card in FlowForge ONLY — not a global button (architecture hard rule)
- Workflow Kill Switch ≠ TradingKillSwitch — these are architecturally independent
- Recovery: `/resume-workflow` command re-triggers from last completed Prefect step
- Phase 3: visual flow builder (drag-to-add nodes) — NOT in this story (view only)

---

### Story 11.6: Server Migration & Multi-Platform Build

**As a** system owner migrating infrastructure,
**I want** server migration without data loss and confirmed builds on Linux, Windows, and macOS,
**So that** infrastructure changes are non-destructive and the ITT is platform-portable (FR70, FR72).

**Acceptance Criteria:**

**Given** a server migration is triggered (e.g., Cloudzy → Hetzner per Journey 28),
**When** the migration procedure runs,
**Then** the new server is configured with the same NODE_ROLE, all credentials, and health checks pass,
**And** strategies resume on the new server with no interruption to configured EAs.

**Given** the ITT source is checked out on Linux, Windows, and macOS,
**When** `npm run build` and the Tauri build run,
**Then** the app compiles and launches on all three platforms (FR72).

**Notes:**
- FR70: server migration without data loss
- FR72: cross-platform build (Linux, Windows, macOS)
- Journey 28: latency improvement from migration (12ms → 4ms) is a happy path, not a requirement

---

### Story 11.7: Theme Presets & Wallpaper System

**As a** trader personalising the ITT,
**I want** theme presets (Frosted Terminal, Ghost Panel, Open Air, Breathing Space) and a wallpaper configuration system,
**So that** the display adapts to different working contexts and lighting conditions.

**Acceptance Criteria:**

**Given** I open Settings → Appearance,
**When** the panel loads,
**Then** the 4 theme presets are listed: Frosted Terminal (default), Ghost Panel (Kanagawa), Open Air (Tokyo Night), Breathing Space (Catppuccin Mocha).

**Given** I select a theme preset,
**When** it applies,
**Then** CSS custom properties swap atomically: `--glass-opacity`, `--glass-blur`, `--sb-density`, `--tile-min-width`, `--tile-gap`,
**And** the change takes effect immediately without page reload.

**Given** a wallpaper is configured,
**When** the ITT renders,
**Then** the Tauri window background is transparent (`"transparent": true` in `tauri.conf.json`),
**And** OS wallpaper shows through the glass shell surfaces.

**Notes:**
- Memory: Mubarak's preference is Balanced Terminal (Frosted Terminal) as default
- Scan-line overlay: 0.03 opacity — imperceptible at distance
- `@media (prefers-reduced-motion: reduce)`: all animations optional

---
