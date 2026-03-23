---
stepsCompleted: ['step-01-init', 'step-02-discovery', 'step-02b-vision', 'step-02c-executive-summary', 'step-03-success', 'step-04-journeys', 'step-05-domain', 'step-06-innovation', 'step-07-project-type', 'step-08-scoping', 'step-09-functional', 'step-10-nonfunctional', 'step-11-polish', 'step-12-complete']
workflowStatus: 'complete'
completedAt: '2026-03-12'
inputDocuments:
  - '_bmad-output/project-context.md'
  - 'docs/project-overview.md'
  - 'docs/component-inventory.md'
  - 'docs/plans/2026-03-06-ide-ui-ux-redesign.md'
  - 'docs/plans/2026-03-08-agent-architecture-migration-map.md'
  - 'docs/plans/2026-03-09-new-chat-feature-v2.md'
  - 'docs/plans/2026-03-09-account-risk-router-mode.md'
  - 'party-mode-session-2026-03-11'
documentCounts:
  briefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 7
classification:
  projectType: 'desktop_app + api_backend hybrid'
  domain: 'fintech'
  complexity: 'high'
  projectContext: 'brownfield'
  productName: 'QUANTMINDX — Integrated Trading Terminal (ITT)'
workflowType: 'prd'
partyModeInsights:
  aesthetic: >
    Frosted Terminal — deep space blue-black (#080d14), frosted glass surfaces (backdrop-filter blur),
    scan-line overlay (preserved from workshop-v2 HTML), amber (#f0a500) for live/active states,
    cyan (#00d4ff) for AI/copilot states, red (#ff3b3b) for kill switch/danger.
    Fonts: JetBrains Mono (data/numbers/code) + Syne 700/800 (headings/brand).
    Department accent variants: Research=amber, Risk=red undertone, Dev=cyan, Trading=green pulse.
    Wallpapers: anime/Arch Linux Hyprland aesthetic — blurred behind frosted panels.
    Full creative freedom given to visual design (Caravaggio directive).
    Sound: lo-fi ambient layer, volume in StatusBand, regime-reactive potential.

  navigationModel: >
    Canvas/Department model — sidebar becomes a LAUNCHER (router), not a container.
    Each department = full-screen canvas with its own workspace.
    Three layers per canvas: (1) Department Core tools, (2) Shared Tools Rail (right side, collapsible,
    available on every canvas — shared assets, database browser, workflow status, department mail),
    (3) Global Copilot Bubble (floats above all canvases).
    Department canvases: Live Trading (home), Research, Development, Risk, Portfolio, Workshop.

  globalCopilot: >
    Persistent floating bubble — bottom-right, context-aware per canvas.
    Has commands/tools specific to the active canvas.
    Orchestrates entire platform via Anthropic Agent SDK.
    Can: trigger workflows, approve strategies, detect macro events, create N8N-style workflows,
    use cron jobs, hooks, agentic plugins, send notifications via department mail.
    Derived from Workshop Copilot (renamed from deprecated QuantMind Copilot).
    NOT the Research Department — it is the orchestration layer. Research dept sends it intel.
    Department mail = notification channel between departments and to Mubarak via Copilot.

  alphaForgeLoop: >
    CORRECTED — Not a simple linear pipeline. Departments work TOGETHER autonomously:
    Goal: Turn strategy SOURCE into production-ready, evolving trading logic.
    Source → Research Dept (extract hypothesis/market conditions using available tools)
    → Development Dept (write code: Python/MQL5/PineScript, compile check)
    → Variable Optimization / Parameter Sweep (agentic — no fixed parameters, agents know what to do)
    → Backtest (Monte Carlo + Walk-Forward)
    → Data Distillation (Portfolio/Data dept generates synthetic datasets, stress tests variables)
    → Loop back to improve (not overfitting — EVOLVING into beta logic for multiple scenarios)
    → Paper Trade (Trading Dept monitors — NOT the final gate, it is a checkpoint)
    → Mubarak manually changes EA tag OR N days pass → auto-deploy to live via Strategy Router.
    KEY: Manages 25-50 strategies simultaneously. Multiple versions per strategy.
    Different pairs, different timeframes, multiple configurations.
    Agents evolve strategies WITHOUT Mubarak's intervention (system prompts guide them).
    This is the core differentiator — LLMs now capable of long coding tasks, strategies are code.
    Agents given the environment + framework = they autonomously develop, improve, evolve strategies
    from ANY source OR from their own initiative.

  fastTrackWorkflow: >
    Event-driven strategy pipeline — for NON-SCALPING strategies (buy-low-sell-high, trend-following).
    Scalping is EXCLUDED from this track (scalping = lower timeframes, different logic).
    Trigger: Geopolitical/macro event detected by Research sub-agent (geopolitical intelligence).
    Research identifies affected instruments (volatility + volume confirmed via Sentinel).
    Research Dept emails/notifies Development Dept via department mail.
    Dev Dept pulls from STRATEGY TEMPLATES (skips full Alpha Forge pipeline — time-sensitive).
    Strategy deployed directly. Copilot notifies Mubarak.
    Cadence: weekly/monthly.
    UI: News section/tab in Live Trading canvas or dedicated news feed.
    Future: Smart money / hedge fund tracking to enhance signal quality (Phase 2, needs ML).

  liveTrading_canvas: >
    Main Control Dashboard — home screen on open.
    StatusBand = persistent nerve bar across ALL canvases (sessions, regime, PnL, active bots).
    Canvas content: Strategy Router visualization (which bots running, which regime),
    account equity curve, open positions, drawdown live, active workflows status,
    department mail queue, Alpha Forge pipeline state, Kelly criterion live calc,
    risk mode indicator, data latency display (MT5 ZMQ latency).
    News feed section within this canvas.
    Navigation links to all other department canvases.
    Server connection status: Cloudzy (live trading), Contabo (HMM/storage), Local.

  framework: 'Upgrade Svelte 4 → Svelte 5 (runes). Same ecosystem. Keep SvelteKit routing. Activity bar becomes router.'

  dataAndDeployment: >
    3-server architecture: Local IDE (Tauri desktop) ↔ Cloudzy (FastAPI live trading, low latency)
    ↔ Contabo (HMM training, cold storage, backtesting compute).
    Data: MT5 ZMQ (primary, live trading) + Dukascopy (historical backtesting).
    Polygon.io: Phase 2 when live stock/equity data needed.
    Scraped articles: Add data/ to .gitignore, sync to Contabo cold storage via existing script.
    Auto-sync cron job for scraped articles. Copilot can trigger manual sync.

  deferredPhase2:
    - 'smart-money-ml-layer (hedge fund / institutional flow tracking — needs ML)'
    - 'voice-interface (too early, no audio model budget)'
    - 'order-book-data (institutional grade, Tardis/Rithmic — Phase 2)'
    - 'full-porting-news-trading (too risky for now)'
    - 'quant-mind-dex (separate crypto project — build after QUANTMINDX stable)'
    - 'bloomberg-full-news-terminal (Phase 2 extension)'

  productAudience: >
    PERSONAL USE ONLY — built for Mubarak. Not commercial currently.
    Commercial consideration only after: Alpha Forge Loop produces strategies that pass
    prop firm challenges (FTMO, The5ers, FundingPips).
    Then potentially productize for other traders.

  dexter_reference: >
    GitHub: https://github.com/virattt/dexter — Autonomous financial research agent.
    TypeScript/Bun + OpenAI + Financial Datasets API (income statements, balance sheets, cash flow).
    Pattern to BORROW (not import): task decomposition → tool selection → iterative refinement → validation.
    Implementation: Clone, strip git, borrow logic as Research Dept tools in Python with Claude.
    Financial Datasets API: institutional financial data for Research Department stock analysis.

vision:
  statement: >
    QUANTMINDX ITT is an autonomous trading terminal where AI departments (Research, Development,
    Risk, Trading, Portfolio) continuously develop, evolve, and deploy trading strategies —
    while the trader maintains oversight as final decision-maker.
  differentiators:
    - 'Alpha Forge Loop — autonomous strategy factory, 25-50 strategies evolving simultaneously'
    - 'Global Copilot Bubble — AI orchestration layer across entire platform'
    - 'Frosted Terminal UX — modern, beautiful, precision instrument aesthetic'
  coreInsight: >
    Solo prop-firm traders need institutional capabilities. LLMs in 2026 can code, research,
    and iterate autonomously on trading strategies. Give them the environment and they build the edge.
  valueProposition: >
    The AI-powered trading terminal that autonomously builds, tests, and deploys strategies —
    so you focus on oversight, not execution.
---

# Product Requirements Document - QUANTMINDX

**Author:** Mubarak
**Date:** 2026-03-11

---

# QUANTMINDX — Integrated Trading Terminal (ITT)
# Product Requirements Document

---

## Executive Summary

QUANTMINDX is a personal **Integrated Trading Terminal (ITT)** — a Tauri desktop application backed by a FastAPI Python service — built to give a single prop-firm trader the operational capability of an institutional trading floor. The system is not commercially positioned. It is engineered to autonomously develop, evolve, and deploy algorithmic trading strategies across MetaTrader 5 markets (forex, indices, commodities) with a target of running 25–50 concurrent strategies, all actively managed and improved by AI departments without the operator's constant intervention.

The platform is optimized for scalping and high-frequency logic on lower timeframes, with a secondary track for longer-timeframe, trend-following strategies triggered by macro/geopolitical events. Profitability and passing prop firm evaluation challenges (FTMO, The5ers, FundingPips) are the primary success benchmarks before any productization is considered.

### What Makes This Special

**1. The Alpha Forge Loop — Autonomous EA Factory**
The defining capability of QUANTMINDX. Five AI departments (Research, Development, Risk, Trading, Portfolio) operate as a coordinated autonomous loop: sourcing strategy ideas, writing trading logic in Python/MQL5/PineScript, running variable optimization and parameter sweeps, stress-testing through Monte Carlo + Walk-Forward backtesting, monitoring paper trading performance, and promoting to live via the Strategy Router — all without human intervention between stages. Strategies do not merely pass backtests; they are iteratively evolved by agents into robust, multi-scenario logic. The operator's role is oversight: reviewing promoted strategies, manually adjusting EA tags, and monitoring system state.

**2. Global Copilot Bubble — AI Orchestration Layer**
A persistent, context-aware AI assistant that floats across every canvas in the UI, powered by the Anthropic Agent SDK. It is not a chat panel — it is an operating layer. From any screen, the operator can create workflows, trigger cron jobs, approve strategy promotions, receive department intelligence briefings, and interact with any component of the platform. The Research Department's geopolitical intelligence sub-agent sends findings through the department mail system; the Copilot surfaces them and coordinates response.

**3. Frosted Terminal UX — Redesigned from Scratch**
A complete visual and navigation overhaul from the current VS Code-inspired compaction model to a **department canvas model**: the activity bar becomes a launcher/router, each department opens as a full-screen workspace with shared tooling on a collapsible rail. The aesthetic is "Frosted Terminal" — deep space blue-black, frosted glass panels, scan-line texture overlay, amber/cyan accents, JetBrains Mono for data, Syne for headings. Svelte 4 is upgraded to Svelte 5 (runes). Themes, wallpapers (anime/Hyprland-inspired), and ambient lo-fi audio are first-class features.

## Project Classification

| Attribute | Value |
|---|---|
| **Product Type** | Desktop app (Tauri) + API backend (FastAPI) hybrid |
| **Domain** | Fintech — algorithmic trading, prop firm compliance |
| **Complexity** | High — regulated domain, real-time operations, multi-agent AI, physics-based risk engine |
| **Context** | Brownfield — active production codebase, significant recent development |
| **Primary User** | Mubarak (sole operator, personal use) |
| **Framework** | SvelteKit + Svelte 5 (upgrade from Svelte 4) + Python 3.12 FastAPI |
| **AI Stack** | Anthropic Agent SDK with configurable multi-provider support — default Anthropic (Opus-4/Sonnet-4/Haiku-4) but providers (GLM, MiniMax, Alibaba, OpenRouter) can be swapped via base URL in Settings → Providers. Models map to the three-tier hierarchy (FloorManager / Department Heads / SubAgents). Provider + model selection changeable at runtime. |
| **Trading Integration** | MetaTrader 5 via ZMQ bridge + Dukascopy for historical data |
| **Deployment** | Local IDE (monitoring/control only) ↔ Cloudzy (all trading execution, 5–20ms to broker) ↔ Contabo (HMM, cold storage) |

## Success Criteria

### User Success

- **Alpha Forge Loop running autonomously**: At least one complete cycle (source → code → backtest → paper trade → live) completes without Mubarak's direct involvement in intermediate stages. He intervenes only to approve EA tag promotion.
- **Daily monitoring under 30 minutes**: The system handles research, strategy evolution, and risk monitoring while Mubarak observes via the Live Trading canvas and StatusBand nerve bar.
- **No navigation traps**: Every section of the UI reachable in ≤2 clicks from any canvas. No compaction, no duplicate navigation bars.
- **Global Copilot responds accurately in context**: Copilot on the Risk canvas offers risk-specific commands; on Research canvas offers research tools. Zero irrelevant suggestions.
- **Strategy evolution without intervention**: Departments improve existing strategies (parameter sweeps, multi-scenario validation) on a running cycle without prompting.

### Business Success

- **First prop firm challenge passed**: At least one strategy, produced or significantly evolved by the Alpha Forge Loop, passes a prop firm evaluation (FTMO, The5ers, or FundingPips). This is the go/no-go gate for any future productization consideration.
- **25 strategies running concurrently**: System stable with 25 active live/paper strategies across different pairs and timeframes — validates the architecture at target scale.
- **Zero manual coding for strategy iteration**: Development Department handles parameter sweeps and variant generation without Mubarak writing code for it.

### Technical Success

- **Svelte 5 migration complete**: All 65+ components migrated to Svelte 5 runes. No Svelte 4 reactive patterns remaining.
- **Canvas navigation functional**: All 6 department canvases (Live Trading, Research, Development, Risk, Portfolio, Workshop) fully operational with shared tools rail and breadcrumb navigation. Zero bottom-nav duplication bugs.
- **Execution latency: 5–20ms**: Cloudzy backend ↔ MT5 broker round-trip latency target 5–20ms. Trading execution runs entirely on Cloudzy — local IDE is monitoring/control only, so remote/slow internet does not affect execution. Cloudzy region selected based on broker MT5 server proximity (e.g. LD4 for London brokers).
- **Multi-provider AI runtime working**: Providers panel allows runtime swap of FloorManager/Department Head/SubAgent model tiers. GLM, MiniMax, OpenRouter all confirmed working via base URL configuration.
- **Cold storage sync operational**: Scraped articles removed from git tracking. Auto-sync cron job running between Cloudzy and Contabo. Manual sync triggerable via Copilot Bubble.
- **Department mail → Copilot notification pipeline**: Geopolitical intelligence sub-agent → department mail → Copilot surface notification working end-to-end.

### Measurable Outcomes

| Outcome | Target |
|---|---|
| Cloudzy ↔ MT5 broker round-trip latency | 5–20ms |
| Alpha Forge full cycle time | ≤72 hours from source ingestion to paper trade start |
| Prop firm challenge pass | 1 within 6 months of Alpha Forge Loop deployment |
| Concurrent strategies at stable state | 25 minimum |
| UI navigation depth | ≤2 clicks to any feature from any canvas |
| Daily active monitoring time | ≤30 min/day |
| Cold storage sync lag | ≤1 hour from scrape to Contabo |

---

## Product Vision & Phases

### MVP — Minimum Viable Product

1. **UI Complete Redesign** — Svelte 5 upgrade, canvas/department navigation model, Frosted Terminal aesthetic (scan lines, JetBrains Mono + Syne, frosted glass panels, amber/cyan accents), department workspaces with shared tools rail, no duplicate nav
2. **Global Copilot Bubble** — Persistent floating overlay, context-aware per canvas, Agent SDK integration, workflow creation, cron/hook access, department mail notifications surfaced
3. **Alpha Forge Loop Core Pipeline** — Research → Development → Variable Optimization → Backtest (Monte Carlo + Walk-Forward) → Paper Trade (Trading Dept monitoring) → EA tag promotion gate → Live via Strategy Router
4. **Cold Storage Fix** — Scraped articles to .gitignore, Contabo sync script + auto-cron
5. **Multi-provider AI settings** — ProvidersPanel fully functional, runtime model swap per tier (Anthropic, GLM, MiniMax, OpenRouter via base URL)
6. **Live Trading Canvas** — Main control dashboard, strategy router visualization, account equity, active bots, news feed section, StatusBand as global nerve bar, server latency indicators

### Growth Features (Post-MVP)

- Geopolitical Intelligence sub-agent (Research Dept) + Fast-Track Event Workflow (non-scalping, trend-following strategies triggered by macro events)
- Financial Datasets API integration (Dexter-pattern: income statements, balance sheets, institutional financial data for Research Dept)
- N8N-style visual workflow editor in Workshop canvas
- News feed tab with enriched macro context
- Polygon.io integration for stock/equity data when scaling beyond forex
- Wallpaper (anime/Hyprland-inspired) + ambient lo-fi audio system enhancement
- Department-specific accent color variants per canvas

### Vision (Future)

- Smart money / institutional flow tracking ML layer (hedge fund position inference from public filings)
- Prop firm challenge automation (Alpha Forge Loop tuned specifically for challenge evaluation rules)
- Commercial productization for other prop firm traders (post first challenge pass)
- QuantMind DEX (separate crypto project — built after QUANTMINDX stable and profitable)
- Voice interface (deferred — no audio model budget currently)

---

## User Journeys

_52 journeys covering the complete operational picture of QUANTMINDX ITT — from first boot to managing client funds, from black swan survival to infrastructure migration, from autonomous overnight research to the system writing its own capabilities._

---

### Journey 1 — The Morning Operator (Primary — Happy Path)

It's 7:45am London time. The London session opens in 15 minutes.

Mubarak launches QUANTMINDX on his desktop. The Live Trading canvas loads immediately — his home screen. The StatusBand pulses: London session OPENING, regime TREND_STABLE, 12 active bots, daily P&L +$340. He doesn't have to click anything to see this.

He glances at the strategy router visualization — 3 scalpers are queued for the London open, risk mode set to Dynamic, drawdown at 1.2% (well within FTMO's 3% limit). The Copilot Bubble in the bottom right shows a dim notification badge. He taps it: "Research Dept completed overnight cycle — 2 new strategy variants generated from last week's EUR/USD analysis. Ready for your review."

He clicks through to the Development canvas, sees the two variants listed with their backtest summary scores. Approves one for paper trading with a single tag change. Returns to Live Trading. Total time: 8 minutes. The session opens. He watches.

**Capabilities revealed:** Live Trading canvas as home screen, StatusBand nerve bar, strategy router visualization, Copilot notification surface, EA tag promotion workflow, cross-canvas navigation in ≤2 clicks.

---

### Journey 2 — The Alpha Forge Trigger (Primary — Strategy Development)

It's Sunday afternoon. Mubarak has been watching a YouTube breakdown of a supply/demand scalping approach.

He opens the Workshop canvas, types to the Copilot Bubble: "I want to build a supply/demand scalping EA for M5 EUR/USD from this video." He pastes the YouTube link. The Copilot confirms: "Starting Alpha Forge Loop. Research Dept will extract the strategy logic. I'll notify you when Development has a draft."

He closes the laptop. Over the next 48 hours, the departments work: Research extracts the hypothesis and conditions, Development writes an MQL5 EA, Risk validates position sizing and stop logic, the system runs Monte Carlo + Walk-Forward on Dukascopy historical data, the Portfolio sub-agent runs a parameter sweep across 6 variable combinations. Two variants survive. They enter the paper trading queue tagged to the London–NY overlap session only.

72 hours later: "EA_EURUSD_SD_V2 has been paper trading for 3 days — 71% win rate, 1.6R avg, max drawdown 0.8%. Approve for live?" Mubarak reviews the equity curve, approves. It goes live.

**Capabilities revealed:** Global Copilot Bubble as workflow trigger, Alpha Forge Loop pipeline, Video Ingest tool, department autonomous coordination, session mask tagging, paper trade monitoring, human approval gate, push notification.

---

### Journey 3 — The Geopolitical Setup (Growth Feature — Event-Driven)

Monday morning. Over the weekend, the US imposed new trade tariffs affecting Japanese exports.

Mubarak opens the platform. The Copilot has a priority notification: "Research — Geopolitical alert: US tariff impact on JPY instruments detected. Affected: USD/JPY, AUD/JPY, Nikkei225. Volatility elevated. No scalping strategies deployed on these pairs. Recommend: trend-following setup review."

He opens the Research canvas. The geopolitical intelligence sub-agent has already prepared a structured brief. He instructs the Copilot: "Have Development pull a trend-following template for USD/JPY and deploy it for this week." The fast-track workflow fires — no backtest, template-based, direct to live with a 7-day expiry tag.

**Capabilities revealed:** Geopolitical intelligence sub-agent, Research canvas, department mail notification to Copilot, fast-track event workflow (template-based, no full pipeline), strategy expiry tagging.

---

### Journey 4 — The Crisis Recovery (Edge Case — Emergency)

Mid-session. An unexpected central bank statement drops. EUR/USD gaps 80 pips in seconds.

Mubarak is away from his desk — slow hotel WiFi. He gets a Copilot push notification: "ALERT: Regime shifted to HIGH_CHAOS. 4 scalpers still active. Drawdown accelerating — 2.1% and rising."

He opens QUANTMINDX. The kill switch button in the top bar is pulsing amber. He hits it. The progressive kill switch fires: soft halt first, then closes all open positions via MT5 bridge on Cloudzy at 8ms — not his hotel WiFi speed. Total closure: under 2 seconds.

Damage: -$210, drawdown back to 1.8%, FTMO limits intact. System auto-tags affected strategies as "PAUSED — HIGH_CHAOS."

**Capabilities revealed:** Cloudzy-side execution (local network irrelevant), kill switch in topbar, regime-triggered strategy auto-pause, push notifications, remote monitoring from any connection quality.

---

### Journey 5 — The Weekly War Room (Planning)

Sunday evening, before the Asian session opens.

Mubarak opens the Workshop canvas: "Let's plan this week." Copilot pulls Portfolio's performance summary, surfaces Research's recommendations. Together they set the week's configuration: session priorities, strategy activations/pauses, risk budget, router mode (Auction), weekly drawdown hard stop at 4%. All departments receive the week's "operating brief" via department mail.

Total time: 25 minutes.

**Capabilities revealed:** Weekly planning workflow via Copilot, Portfolio performance summary, Research weekly brief, router mode configuration, risk budget setting, weekly drawdown stop, department mail broadcast.

---

### Journey 6 — The Strategy Portfolio Audit (Portfolio Management)

End of month. Portfolio canvas shows the monthly performance report: 18 profitable, 8 flat, 5 losing. Copilot recommends retiring two underperformers and replacing with two Alpha Forge candidates. Mubarak agrees. Tags them RETIRED. New candidates go live.

He checks the correlation matrix — 4 EUR/USD strategies >0.85 correlated. He instructs: "Flag any future strategy pairs exceeding 0.80 correlation before paper trade approval." Rule saved as a system-level risk gate.

**Capabilities revealed:** Portfolio performance audit, strategy correlation matrix, RETIRED tag state, dynamic risk gate rules, Copilot as rule-setter, strategy lifecycle management.

---

### Journey 7 — The New User Setup (First Run / Configuration)

Day one. New machine. Copilot greets him and walks through connection setup. Tests: "Cloudzy ✅ 12ms. MT5 bridge ✅. ZMQ stream ✅ ticking." He sets up Anthropic + GLM providers with live tests. Sets risk defaults. Picks Frosted Terminal theme, anime wallpaper, lo-fi volume 30%.

**Capabilities revealed:** First-run onboarding via Copilot, server connectivity test with latency readout, multi-provider API key setup with live test, risk defaults configuration, theme/wallpaper/audio settings.

---

### Journey 8 — The Silent Failure (Backend Resilience)

3:14am. Contabo drops connection. Cloudzy trading continues normally. Sentinel falls back to cached regime and flags a health warning. At 8am the Copilot brief: "Contabo connection lost at 03:14. Sentinel fell back to cached regime. Trading continued normally. Reconnection at 03:41. No trades missed. No action required."

**Capabilities revealed:** Sentinel fallback mode (cached regime on data loss), degraded-mode trading, Copilot deferred notification queue, server status panel.

---

### Journey 9 — The Strategy Source Ingestion (Book / PDF / Article)

Mubarak uploads a PDF chapter on volume-profile-based entry logic. Research processes it via vision model — extracts hypothesis, identifies required data, flags missing dependency (volume profile not native in MT5), generates a structured hypothesis document with confidence qualifier. Mubarak instructs Development to proceed with tick volume approximation. Alpha Forge begins.

**Capabilities revealed:** PDF/document ingest via Copilot, Research document analysis, data dependency detection and flagging, structured hypothesis document output, conditional Alpha Forge trigger.

---

### Journey 10 — The Challenge Mode (Prop Firm Compliance)

Mubarak activates Challenge Mode for an FTMO $100k challenge — inputs rules (10% max loss, 5% daily, 10% target, 30 days). Governor layer locks these as hard constraints. Kelly Engine recalculates. StatusBand gains a challenge progress line. Alpha Forge validates new strategies against challenge constraints before paper trade. On Day 14, Copilot alerts about combined EUR exposure approaching daily loss limit. Mubarak pauses one strategy. He passes on Day 28.

**Capabilities revealed:** Challenge Mode configuration, prop firm rule parameterization, challenge-aware position sizing, StatusBand challenge progress display, challenge-constraint Alpha Forge validation, correlation-based exposure warning.

---

### Journey 11 — The Overnight Research Cycle (Research Dept Autonomy)

Mubarak closes the laptop at 11pm. Research runs its nightly autonomous cycle without instructions — scans knowledge base, generates 3 new hypothesis documents ranked by confidence score, routes them via department mail to Development, flags one live strategy whose confidence score has degraded. At 8am: "Research completed overnight cycle. 3 new hypotheses queued. 1 strategy flagged for Risk review. No action required."

**Capabilities revealed:** Research autonomous scheduling (cron-driven), knowledge base pattern detection, hypothesis scoring and prioritization, department mail routing with priority tags, strategy confidence degradation detection.

---

### Journey 12 — The Parameter Sweep (Dev Dept Deep Work)

Development launches a parameter sweep for a new EA: 8 variables, 512 combinations. Portfolio sub-agent generates a synthetic stress-test dataset, eliminating 480 combinations in 4 hours. 32 survivors go to full Dukascopy Walk-Forward backtest. 6 pass. Correlation check removes 2. 4 proceed to paper trading. Copilot notification: "512 combinations tested. 4 survived correlation filter. Estimated paper trade review: 5 days."

**Capabilities revealed:** Automated parameter sweep, synthetic dataset pre-filtering, Walk-Forward backtest on survivors only, correlation gate before paper trade, session + instrument tagging at optimization output.

---

### Journey 13 — The Knowledge Base Feed (Data Pipeline)

Every Monday 6am: Firecrawl scrapes MQL5.com for new articles matching keyword filters → writes to `data/scraped/` (gitignored) → rsync syncs to Contabo cold storage → PageIndex indexes into vector knowledge base. By Research's overnight cycle, fresh articles are searchable. Manual trigger available via Copilot: "Sync knowledge base now."

**Capabilities revealed:** Firecrawl scraper with keyword filters, data/ excluded from git, Contabo rsync sync, PageIndex vector indexing, cron-scheduled full pipeline, manual trigger via Copilot.

---

### Journey 14 — The Custom Workflow (Copilot as Automation Builder)

Mubarak: "Create a weekly workflow — every Friday 4pm, pull all strategies with weekly drawdown > 2%, show me a summary, and ask if I want to pause any for the weekend. Also auto-pause anything over 3%."

Copilot creates `FRIDAY_DRAWDOWN_REVIEW`. Next Friday 4:02pm: "2 strategies flagged. 1 auto-paused (3.4%). Awaiting your decision on 2 flagged." Done in 90 seconds.

**Capabilities revealed:** Copilot conversational workflow creation, cron-triggered execution, data query integration, human decision response gate, conditional auto-action rule, workflow management.

---

### Journey 15 — The Model Swap (Runtime Provider Switch)

Mubarak routes SubAgents to GLM-4 Flash via base URL in Settings → Providers. Cost drops 60% on the next parameter sweep. Two weeks later GLM has a service outage — Copilot detects SubAgent failures and automatically falls back to claude-haiku-4. Mubarak sees the notification. Already handled.

**Capabilities revealed:** Runtime model tier swap (no restart), base URL provider entry with live test, three-tier model assignment UI, provider fallback on outage, cost optimization pattern.

---

### Journey 16 — The Code Review (Development Canvas)

EA_EURUSD_Momentum_V3 is displayed in Monaco Editor with Department Head annotations. Mubarak spots lot size is fixed rather than Kelly-based. He types to Copilot: "Lot sizing should use Kelly Engine output, not fixed lots." Development receives the feedback via department mail and generates V4 in 8 minutes. Mubarak approves. Enters backtest queue.

**Capabilities revealed:** Development canvas with Monaco Editor integration, AI-annotated code review summary, inline feedback via Copilot → department mail loop, iterative revision cycle, compilation check before backtest queue admission.

---

### Journey 17 — The Physics Lens (Risk Canvas Deep Dive)

Risk canvas: Ising Model shows high EUR correlation clustering. HMM shows 28% BREAKOUT probability rising. Lyapunov Exponent up 40% in 6 hours. Mubarak asks: "What does the physics engine say about risk right now?" Copilot synthesizes all three sensors: reduce EUR exposure 30% preemptively. He adjusts to Conservative for EUR pairs. Sets Lyapunov threshold alert at 0.8. Four hours later, EUR/USD breaks 80 pips. Drawdown: 0.4% instead of 1.8%.

**Capabilities revealed:** Risk canvas with Ising/HMM/Lyapunov visualizations, Copilot physics synthesis, manual risk scalar override per instrument group, threshold-based alert creation, Commander real-time recalculation.

---

### Journey 18 — The Multi-Type Orchestration Day (Commander in Full)

31 active strategies across four types simultaneously: Scalpers (12, session-locked London–NY overlap, TREND/RANGE only, spread filter), Trend-followers (6, continuous, TREND only, 2x sizing, 3 currently in open trades), Mean-reversion (8, all regime-paused — HMM 62% TREND, reactivate automatically when RANGE returns), Event-driven (5, expiry-tagged, one expires tonight and auto-archives).

StatusBand: 26 active / 31 total. Strategy router shows a color-coded grid by type + state. Mubarak glances 30 seconds — understands the full system state.

**Capabilities revealed:** Commander multi-type strategy orchestration, regime filter per strategy type, session lock vs. continuous activation, spread filter with auto-pause, event strategy expiry + auto-archive, strategy router visualization.

---

### Journey 19 — The Paper Trade Graduation (The Formal Dossier)

EA_XAUUSD_Scalp_V2 completes 11 days of paper trading. A structured dossier arrives — win rate, avg R, max drawdown, Sharpe, regime-specific performance breakdown, spread sensitivity log, correlation to live portfolio, Risk Dept sign-off, Development Dept notes.

Mubarak notices BREAKOUT regime is slightly negative. Types: "Add a regime filter — pause when HMM BREAKOUT probability exceeds 40%." Condition embedded. He approves. Strategy goes live with the regime guard built in.

**Capabilities revealed:** Structured paper trade dossier (auto-generated), benchmark comparison table, regime-specific performance breakdown, Copilot condition addition at approval time, formal approval gate as a document.

---

### Journey 20 — The Dual Account (Prop Firm + Personal Capital)

FTMO Challenge ($100k) and Personal ($8k) run simultaneously. FTMO receives only strategies with >10 days paper trade history and max drawdown <1.5%. Personal receives strategies in "live observation" tier plus strategies ineligible for FTMO. A strategy passing paper trade with 1.8% max drawdown gets routed to Personal only — below FTMO threshold. Six weeks later with 1.1% live drawdown, Mubarak manually re-evaluates for FTMO.

**Capabilities revealed:** Multi-account routing with separate eligibility rules, account-level Kelly multiplier, Governor hard stop per account, live observation tier, automatic account assignment at graduation, manual re-evaluation workflow.

---

### Journey 21 — The Template Pull (Fast-Track Source)

US CPI prints hot. Mubarak: "Fast-track — USD inflation play, trend-following, H1, USD/JPY and EUR/USD short bias." Copilot queries the Strategy Template Library, surfaces two matching templates with historical profiles. Development instantiates the chosen template with current conditions, NY session only, 3-day expiry, conservative sizing, deploys to Personal Account.

Total time: 11 minutes from event to live trade.

**Capabilities revealed:** Strategy Template Library, template search by type + instrument + condition, Development Dept template instantiation, fast-track deployment path, auto-expiry tagging, account selection at deploy time.

---

### Journey 22 — The Template Forge Session (Historical News → Strategy Templates)

Saturday session with Copilot: "Build a library of news-event strategy templates — starting with NFP, CPI, and FOMC." Research analyzes every release from the last 5 years in Dukascopy data — what happened to major pairs pre/during/post each release. Four consistent patterns surfaced with confidence scores. Three become templates (one discarded — no edge). Each gets multi-condition rules embedded ("no entry within 48h of FOMC"). Templates saved to library.

**Capabilities revealed:** Copilot-guided template-building session, Research multi-year historical news analysis, statistical edge quantification, conflicting-signal detection and discard, multi-condition rule embedding, Dukascopy used for event analysis.

---

### Journey 23 — The Black Swan (Catastrophe)

Sunday night. SNB removes EUR/CHF floor without warning. EUR/CHF drops 1,500 pips in 4 minutes. Lyapunov spikes past 0.95.

Automated cascade fires without human intervention: Commander pauses all EUR/GBP scalpers (CHAOS), Governor halts EUR/GBP, BotCircuitBreaker quarantines 4 scalpers with open positions, ProgressiveKillSwitch soft halt (spread 8 pip — untradeable), Copilot priority alert fires: "BLACK SWAN ALERT. Exposure at halt: -$340."

Mubarak sees it at 6am. Damage contained to -$340 (estimated without cascade: -$2,100+). FTMO challenge intact. Post-event Risk Dept analysis already generated.

**Capabilities revealed:** Automated multi-layer cascade on black swan, Lyapunov threshold as cascade trigger, BotCircuitBreaker auto-quarantine, post-event Risk Dept auto-analysis, conditional auto-resume rule.

---

### Journey 24 — The Ceiling (Account Growth at Scale)

Eighteen months after deployment. 87 strategies in library (34 active live). Three strategies originated entirely by Research's autonomous cycles — no human instruction at initiation. FTMO $100k passed month 3. The5ers $200k month 7. FundingPips $400k month 11. $700k funded capital plus $22k personal. Daily monitoring: 15 minutes. Weekly War Room: 25 minutes. Mubarak types to Copilot: "Start scoping what it would take to offer this to three other traders."

**Capabilities revealed:** System at 34 active live strategies, Research-originated strategies (fully autonomous sourcing), compound equity curve in Portfolio canvas, system sustaining <30 min/day at full scale, commercialization trigger pathway.

---

### Journey 25 — The Knowledge Injection (Adding Your Own Intelligence)

Mubarak uploads his Obsidian vault export (markdown files — personal trading journal, session notes, pattern observations). System indexes via PageIndex, tags as `source: personal_notes`. He uploads photographed book pages — Research processes via vision model, extracts text, indexes. He types a direct observation: "EUR/USD tends to fake-break the Asian session high during the first 30 minutes of London — then reverse. Consistent since 2023." It becomes a structured hypothesis document. The next EUR/USD London open EA includes the fake-breakout filter.

**Capabilities revealed:** Personal knowledge layer (separate from auto-scraped), Obsidian vault import, image/photo ingestion with vision-model extraction, direct note-to-hypothesis pipeline, knowledge source tagging (scraped / ingested / personal / system).

---

### Journey 26 — The Extended Sessions (Zurich, Sydney, and Beyond)

Mubarak opens Risk canvas → Session Configuration. He adds Zurich (07:00–08:00 GMT): EUR/CHF, USD/CHF, EUR/USD only; trend-following only; Kelly 0.7x; SNB event filter (auto-pause if SNB speaks within 4 hours). He adds Sydney (22:00–06:00 GMT): AUD/NZD pairs only, mean-reversion scalpers allowed.

Alpha Forge Loop now knows these sessions exist. Six weeks later, Zurich slot runs 4 strategies — the hour Mubarak was leaving on the table.

**Capabilities revealed:** Session configuration as editable system map, per-session instrument allowlist, per-session strategy type filter, per-session Kelly multiplier, event-based auto-pause per session, arbitrarily extensible session model.

---

### Journey 27 — The Client Layer (Managing Capital for Others)

Two years in. Mubarak adds two client accounts with separate risk profiles. Only strategies with >60 days live history and max drawdown <1.5% eligible for client routing — enforced automatically at assignment. At month end, Copilot generates a structured client report for each (return, drawdown, strategy breakdown, risk events, outlook). Mubarak reviews, exports to PDF, sends. Total time: 8 minutes for both.

**Capabilities revealed:** Multi-client account management, client-specific routing eligibility filter, per-client Kelly multiplier and drawdown cap, automated client report generation, PDF export, client accounts as distinct risk-isolated namespaces.

---

### Journey 28 — The Infrastructure Migration (Cloudzy → New Provider)

Three Cloudzy outages in two months. Mubarak migrates to Hetzner Frankfurt. Copilot generates a migration runbook. Mubarak provisions Hetzner, runs connectivity test from QUANTMINDX: "Hetzner ✅ 4ms | MT5 bridge ✅ | ZMQ stream ✅." Switches active server. Cloudzy demoted to standby. StatusBand latency drops from 12ms to 4ms. Migration invisible to strategies.

**Capabilities revealed:** Multi-server configuration in Settings → Connections, server connectivity test with live latency to broker MT5, Copilot-generated migration runbook, server role assignment (active / standby / cold storage), zero-downtime server switch.

---

### Journey 29 — The Department Mail Morning Read

08:15am. StatusBand: mail badge 7 unread. Department Mail inbox tells the overnight story — Research routing a new hypothesis to Development, Development completing a prototype, Risk flagging a correlation spike, Trading reporting paper trade day 5 completion, Research sending a weekly KB update, Portfolio suggesting a rebalance, Risk broadcasting the morning regime. Mubarak reads through in 4 minutes. He replies to Portfolio: "Show me the rebalance suggestion."

**Capabilities revealed:** Department Mail inbox as a readable structured layer, inter-department message threading, horizontal department-to-department mail, reply-to-action from mail panel, regime morning broadcast.

---

### Journey 30 — The Knowledge Search (Research as a Tool)

Before building a new AUD/USD mean-reversion strategy, Mubarak asks: "What do we know about AUD/USD mean-reversion?" Copilot queries all four knowledge partitions via PageIndex in 8 seconds — 3 scraped articles, 1 personal note, 2 Research hypothesis documents, EA_AUDUSD_Range_V1 (retired — underperformed on London session, performed better Sydney), 1 backtest result (Sharpe 1.4, Sydney session only).

"Bring EA_AUDUSD_Range_V1 out of retirement. Apply a Sydney session mask and re-enter paper trading." Done in seconds.

**Capabilities revealed:** Unified knowledge base search across all partitions, all asset types in results (articles, notes, hypotheses, EAs, backtest results), strategy resurrection from RETIRED state via Copilot, knowledge search as natural workflow step.

---

### Journey 31 — The Batch Trigger (Multiple Workflows in Parallel)

Sunday afternoon. Single Copilot message: "Start these in parallel: (1) weekly KB scrape and sync to Contabo, (2) Alpha Forge for AUD/JPY hypothesis, (3) parameter sweep on EA_GBPUSD_Momentum_V3, (4) generate monthly portfolio report for all accounts."

Copilot fires all four simultaneously. Workshop canvas shows a real-time workflow status panel with progress per workflow. Report 4 finishes first — Mubarak reads it while others run. He closes the laptop. Two hours later: Alpha Forge notification.

**Capabilities revealed:** Multi-workflow batch trigger via single Copilot message, parallel workflow execution, real-time workflow status dashboard, workflow persistence across session (continues on Cloudzy after laptop closes).

---

### Journey 32 — The Department Council (Talking to Multiple Agents)

Mubarak: "I want to hear from Research, Risk, Portfolio, and Development on whether we should scale to 35 active strategies."

Copilot broadcasts to all four simultaneously. Each responds within 3 minutes with domain-specific assessments. Departments cross-reference each other's outputs. Copilot synthesizes: "Consensus: feasible. Two conditions must be met first." A phased scaling plan is created and routed to all departments as the new operating directive.

**Capabilities revealed:** Multi-department broadcast query, simultaneous department response collection, departments cross-referencing each other's outputs, Copilot cross-department synthesis, structured decision document from council output, directive broadcast as outcome.

---

### Journey 33 — The A/B Live Test (Competing Strategy Versions)

EA_EURUSD_Scalp_V4 is live. V5 exists with refined spread filter. Both run simultaneously at half lot size each. Portfolio canvas shows a live race board updating daily. Day 14: Copilot surfaces "V5 statistically outperforms V4. Confidence: 89%. Recommend retiring V4." Mubarak approves. V4 retired. V5 takes full allocation. The A/B record is saved permanently to strategy history.

**Capabilities revealed:** A/B live test framework, half-lot allocation per variant, live side-by-side comparison dashboard, statistical confidence scoring, strategy replacement as A/B outcome, permanent A/B record.

---

### Journey 34 — The Decision Audit (Why Did the System Do That?)

Mubarak notices EA_GBPUSD_Trend_V2 was paused for 6 hours without his intervention. He asks: "Why was EA_GBPUSD_Trend_V2 paused yesterday between 14:00 and 20:00?"

Copilot pulls the full timestamped decision audit trail — HMM regime shift at 14:02, Commander evaluation, regime filter trigger, Governor confirmation, department mail routing, resume at 19:47. Everything correct. Zero missed trades. He adds a notification rule: "Notify me in real-time when any strategy is paused for more than 2 hours."

**Capabilities revealed:** Full decision audit trail per strategy event (timestamped, causal chain), natural language audit query, "0 missed trades" counter, custom notification rule creation via Copilot.

---

### Journey 35 — The Seasonal Calibration (Research Reads the Calendar)

Late November. Research detects a pattern: the last 4 Decembers show EUR/USD and GBP/USD volatility dropping 35–40% in the last two weeks of December. Scalping strategies underperform. Only JPY range strategies show consistent December returns. Research generates a seasonal brief. Mubarak: "Create a recurring seasonal workflow — Dec 15: implement these three changes. Jan 4: reverse them." `SEASONAL_DEC_THINNING` fires annually without Mubarak remembering.

**Capabilities revealed:** Research autonomous seasonal pattern detection, seasonal intelligence brief, Copilot seasonal workflow with annual recurrence, multi-action paired workflow, Alpha Forge pause during low-data-quality periods.

---

### Journey 36 — The Fatigue Filter (Managing Notification Overload)

Three months in. 40–60 notifications per day. Copilot analyzes 30 days of notification history, categorizes by type and resolution outcome, suggests suppressions based on patterns. All approved. Volume drops to 8–12/day — all requiring a decision. Department mail batched into daily 08:00 digest.

**Capabilities revealed:** Notification history analytics, AI-suggested suppression rules based on resolution patterns, per-notification-type configuration, notification batching into daily digest, Copilot-assisted preference calibration.

---

### Journey 37 — The Rollback (Strategy Version Control)

EA_EURUSD_Scalp_V5 has been degrading 4 days — Sharpe 1.9 to 0.9. Development canvas shows full version history (V1–V5 with states and performance). V4 ran 31 days with Sharpe 1.6. "Roll back to V4." V5 demoted to `paused-degrading` (preserved for investigation). V4 reactivated. Development receives an automatic investigation task. Three days later: V5 was calibrated on February volatility. March is different. V5.1 queued.

**Capabilities revealed:** Full strategy version history, live performance degradation detection, one-instruction rollback, rollback as demotion not deletion, automatic investigation task on rollback, version state labels.

---

### Journey 38 — The Memory Thread (Copilot Remembers Across Sessions)

Thursday. "Where did we leave off on the AUD/JPY strategy discussion?" Copilot pulls the thread from three days ago: what was requested, what Research returned, what Mubarak said, what's still pending (an AUD/USD comparison not yet done). "Want me to run it?" Yes. Both outstanding items execute simultaneously. Zero ramp-up time.

**Capabilities revealed:** Cross-session memory for active work threads, pending decision tracking, proactive surfacing of outstanding items, natural language thread retrieval, zero-ramp-up session resumption.

---

### Journey 39 — The Loss Propagation (System Learns from Failure)

EA_GBPUSD_Breakout_V2 is retired. Development extracts the lesson: ATR filter set too loosely. Same filter logic exists in 6 other active strategies. Development generates a cross-strategy patch recommendation — routes to Risk for validation, then to Mubarak. Approved. Batch patch applied to all 6 simultaneously. 7-day paper trade confirmation before going back live. One retirement improved six live strategies.

**Capabilities revealed:** Failure analysis as structured output, cross-strategy pattern search, batch patch proposal workflow, backtest comparison as approval evidence, batch version increment, learning propagation as system behavior.

---

### Journey 40 — The Provenance Chain (Where Did This Strategy Come From?)

Mubarak asks: "Where did EA_XAUUSD_VolCluster_V3 come from?" Copilot traces the full chain — MQL5 article scraped Jan 14 (auto-indexed), Research overnight scoring Jan 17 (0.82 confidence), Development autonomous build Jan 19 (no human instruction), Mubarak's single code review comment Jan 21, paper trade approval Feb 2.

"Mubarak's total direct contribution: 1 code review comment. 1 approval. Everything else: autonomous."

**Capabilities revealed:** Full knowledge provenance trace per strategy, natural language provenance query, human contribution log vs. system contribution, autonomous strategy origin tracking.

---

### Journey 41 — The Liquidity Sentinel (Real-Time Liquidity Detection)

14:58 GMT. EUR/USD spread widens from 0.8 pip to 2.1 pip. Sentinel detects liquidity thinning (spread + tick frequency + bid-ask bounce composite) and flags LOW LIQUIDITY. Commander responds: scalpers no new entries, pending limit orders cancelled, open scalper losers closed, trend-follower winners held. By 15:05 spread normalizes. Scalpers resume 15:08 for NY open. StatusBand showed a 10-minute orange indicator, then green. Mubarak saw it from across the room. Touched nothing.

**Capabilities revealed:** Real-time liquidity monitoring (composite signal), strategy-type-differentiated liquidity response, selective position close (losers closed, winners held), automatic resumption, StatusBand liquidity indicator.

---

### Journey 42 — The Weekend Engine (Downtime as Compute Time)

Saturday 09:00 GMT. Markets closed. Automated weekend protocol: full 10,000-simulation Monte Carlo on all 34 active strategies (too heavy for weekdays), HMM model retraining on Contabo, PageIndex cross-document relationship pass, full 90-day correlation matrix refresh, Alpha Forge backlog processing. By Sunday 06:00: "3 strategies flagged after full Monte Carlo. HMM retrained — sensitivity improved 12%. 847 cross-document links added."

**Capabilities revealed:** Automated weekend compute protocol, Contabo as heavy compute node, HMM weekly retraining, full vs. rolling correlation matrix, PageIndex cross-document relationship pass, Alpha Forge backlog queue.

---

### Journey 43 — The Contradicted Hypothesis (Research Challenges a Live Strategy)

EA_EURUSD_Momentum_V4 has been live 47 days. Its founding hypothesis was 68% continuation rate. Research's autonomous cycle finds the rate has dropped to 51% over 6 weeks (p=0.03, statistically significant). Risk cross-checks: Sharpe declined from 1.6 to 0.9 over same period. Copilot: "Warning: V4's founding hypothesis has degraded. Recommend 14-day performance watch with auto-retire trigger if Sharpe < 0.8." Mubarak sets the watch. Day 9: Sharpe hits 0.78. V4 auto-retires.

**Capabilities revealed:** Ongoing hypothesis validity monitoring, statistical significance testing, contradiction report as Research output type, conditional auto-retire trigger, market structure change detection.

---

### Journey 44 — The Strategy Race (Parallel Paper Trade Competition)

Three AUD/JPY prototypes — backtests close, no clear winner. "Run all three in parallel paper trading. Best performer after 14 days wins." All three enter simultaneously at half-size. Portfolio canvas shows a live race board. Day 14: V3 wins on Sharpe and max drawdown. V1 shows edge in low-ATR windows — assigned as regime-conditional backup. V2 archived.

**Capabilities revealed:** Multi-candidate parallel paper trade competition, race board as Portfolio canvas view, equal conditions enforcement, nuanced outcome (conditional activation vs. discard), strategy archive distinct from retire.

---

### Journey 45 — The Calendar Gate (Economic Calendar-Driven Rules)

NFP every Friday. System fires automatically: Thursday 18:00 → EUR/USD scalpers at 0.5x lot size. Friday 13:15 → all EUR/USD scalpers pause. Friday 14:00 → regime check determines what reactivates. Friday 15:00 → full normal operation. FTMO account has an additional layer: no new positions within 2 hours of any Tier 1 news event. Built once via Copilot. Runs automatically every occurrence.

**Capabilities revealed:** Economic calendar integration, calendar-aware lot size scaling, tiered window protocol (pre/during/post), conditional post-event reactivation, account-level calendar rule override, once-built repeating protocol.

---

### Journey 46 — The Live News Room (Research Responds in Real Time)

10:42 GMT. Unscheduled news wire: "ECB sources signal emergency rate discussion possible." Research's geopolitical sub-agent catches it within 90 seconds. Classifies severity HIGH. Identifies affected instruments. Checks current EUR exposure ($2,400). Copilot surfaces at 10:44: "RESEARCH ALERT — ECB emergency signal. 8 EUR strategies active. Recommend reduce EUR exposure 50%."

Mubarak reduces. By 10:55, EUR/USD moves 60 pips. Cost: $180 instead of $420. At 11:30: Research sends all-clear. Normal exposure resumes.

**Capabilities revealed:** Live news feed monitoring, sub-90-second detection-to-alert pipeline, automated severity classification, cross-check with live exposure, all-clear signal as distinct Research output type.

---

### Journey 47 — The Spawn Chain (Sub-Agent Parallelism Inside a Department)

Development Head receives a 4-component EA build task. Complexity threshold exceeded. The Head autonomously spawns 4 parallel Haiku-tier sub-agents — one per component — without asking Copilot or FloorManager. Sub-agents work simultaneously. Head synthesizes: reviews interfaces, resolves timing conflicts, patches compilation errors found by a fifth dedicated compile-check sub-agent. Clean compile. Enters backtest queue.

Total: 18 minutes. Sequential single-agent approach: 60–90 minutes.

**Capabilities revealed:** Department Head as orchestrator (not just generator), parallel sub-agent spawning per EA component, sub-agent specialization, Department Head synthesis layer, dedicated compilation-check sub-agent, spawning as autonomous Department Head decision, Haiku-tier cost optimization.

---

### Journey 48 — The Skill Layer (Agentic Skills in Department Workflows)

Research chains four registered agentic skills for hypothesis validation: `financial_data_fetch` → `pattern_scanner` → `statistical_edge` → `hypothesis_document_writer`. Each skill is encapsulated — the Department Head just calls them. A structured, validated hypothesis document lands in Development's queue.

Mubarak uploads a Python function calling the Financial Datasets API (Dexter pattern). Copilot wraps it as registered skill `institutional_financial_data`. Now any department can invoke it.

**Capabilities revealed:** Agentic skills as registered reusable capabilities, skill chaining within department workflows, skill encapsulation, skills available to Copilot-created workflows, Skills Registry panel in Workshop canvas, user-created skills, skill invocation count as observability metric.

---

### Journey 49 — The Orchestration Topology (Who Tells Who to Do What)

A complex multi-department task arrives. The topology:

- **Mubarak → Copilot** (intent)
- **Copilot → FloorManager** (task graph + dependency map + parallel dispatch)
- **Department Heads** (execution strategy — autonomous; spawning is the Head's decision)
- **Sub-Agents** (task execution — spawned by Department Head)
- **Copilot** (intervenes only at: stall detection, human decision gates, scope changes)

Departments communicate horizontally via department mail for peer-to-peer coordination. Copilot handles delta re-dispatch on scope changes without restarting the full pipeline.

**Capabilities revealed:** FloorManager as task graph generator (not sequential queue), parallel department dispatch with dependency rules, Department Head autonomous spawning threshold, Copilot's three intervention points only, horizontal department-to-department mail, delta re-dispatch on scope change.

---

### Journey 50 — The Skill Forge (System Writing Its Own Skills)

Research Head notices it has run the same 4-step hypothesis validation sub-agent chain four times. It routes a skill request to Development: "Encapsulate as registered skill `hypothesis_validator`." Development builds it (Python function + schema + acceptance tests), routes to Skills Registry. Copilot surfaces it to Mubarak: "Development created a new skill. Auto-approve?" Approved. Next Research hypothesis validation: 8 minutes instead of 40.

Three days later, Copilot identifies Mubarak has requested portfolio report formatting three times this week. "Should I write a `portfolio_report_generator` skill?" Mubarak says yes.

**Capabilities revealed:** Department Head self-monitoring for repeated patterns, autonomous skill request generation, Development as skill builder (code + schema + acceptance test), Copilot as skill suggester from Mubarak's repeated requests, skill versioning with automatic propagation, skill registry as a growing self-improving asset.

---

### Journey 51 — The Transparent Reasoning (Copilot Explains Itself)

Mubarak asks: "Explain to me why you recommended reducing EUR exposure yesterday at 10:44."

The Copilot walks through the full reasoning chain — exposure concentration (8 of 31 strategies EUR-exposed, $2,400 at risk), historical precedent (7 of last 9 unscheduled ECB signals produced EUR moves >40 pips in 30 minutes, from knowledge base), current regime (HMM 62% TREND, trend-followers in active positions with open profit at risk). The 50% reduction was calibrated to leave exposure for a continuation move while cutting the loss scenario in half. "Signal said 'discussion possible' — not confirmed action. Confirmed action would have triggered a full exit recommendation."

Reading these explanations over time teaches Mubarak to see markets the way the system does.

**Capabilities revealed:** Copilot reasoning transparency on request, multi-factor decision explanation (exposure + historical precedent + regime simultaneously), calibration explanation (why 50% not 100%), knowledge base historical precedent citation in reasoning, signal-strength-to-action mapping, Copilot as a learning tool over time.

---

### Journey 52 — The EA Deployment Pipeline (MQL5 to MT5)

Development produces `EA_XAUUSD_Scalp_V2.mq5` — compiled, approved for paper trading. The MT5 bridge on Cloudzy handles deployment:

1. **File transfer** — compiled `.ex5` pushed to Cloudzy MT5 `Experts/` directory via mt5-bridge API
2. **Terminal registration** — mt5-bridge instructs the MT5 terminal to register the new EA
3. **Credential + parameter attachment** — routing matrix assigns EA to correct broker account and chart (XAUUSD, M5) with parameters injected as EA inputs (lot size, session mask, risk settings)
4. **Health check** — bridge verifies EA status: ACTIVE, first tick received, no initialization errors
5. **ZMQ stream registration** — EA registered in ZMQ stream so tick data and trade events flow to Sentinel in real time

Copilot confirms: "EA_XAUUSD_Scalp_V2 deployed to MT5. XAUUSD M5, Personal Account. Status: ACTIVE. First tick received. ZMQ registered."

When a strategy is retired: EA detached from chart, removed from routing matrix, `.ex5` file archived to Contabo cold storage (not deleted). Version updates are clean swaps — old EA detached before new one attached, no overlap period.

**Capabilities revealed:** Full EA deployment pipeline (QUANTMINDX → Cloudzy MT5 bridge → MT5 terminal → chart attachment → ZMQ registration), credential + parameter injection at deployment time, health check with first-tick verification, ZMQ stream registration as part of deployment, retirement reverse pipeline (detach → archive to Contabo), version update as clean swap, mt5-bridge as deployment orchestration layer.

---

### Journey Requirements Summary

52 journeys across 10 capability domains, surfacing 55+ distinct capability areas:

| Domain | Journeys |
|---|---|
| Daily operator workflows | 1, 5, 8, 29, 36 |
| Alpha Forge Loop pipeline | 2, 12, 13, 44 |
| Event-driven / fast-track | 3, 21, 22, 45, 46 |
| Crisis / catastrophe / resilience | 4, 23, 28, 37, 43 |
| Portfolio + multi-account management | 6, 20, 27, 33 |
| Physics risk engine | 17, 18, 41 |
| Research + knowledge | 11, 25, 30, 40 |
| Agentic architecture | 32, 47, 48, 49, 50 |
| System self-improvement | 35, 39, 42, 50 |
| Planning + configuration | 7, 10, 14, 15, 24, 26, 34, 51, 52 |


---

## Domain-Specific Requirements

_Domain: Fintech — Algorithmic Trading | Complexity: High_

---

### 1. Prop Firm Compliance Engine (Extensible)

The system does NOT hardcode any specific prop firm. It implements a configurable **Prop Firm Registry** — any firm can be added with custom parameters:

- Name, max total drawdown %, max daily loss %, min trading days, profit target %, challenge duration
- Prohibited behaviors: martingale (yes/no), hedging (yes/no), news trading (yes/no), weekend holding (yes/no), copy trade relay (yes/no)
- Governor layer enforces whichever rule set is assigned to each account
- Multiple prop firms active simultaneously with different rule sets per account

**Capital pathways:**
- **Prop firm challenges** (FTMO, The5ers, FundingPips, and any new firm added to registry): primary route — faster capital access on challenge pass
- **DarwinEx**: secondary route — capital allocation based on verified long-term track record
- **Copy trading model**: QUANTMINDX trades ONE master prop firm account. External copy trading tools (outside QUANTMINDX) replicate trades to additional funded accounts. QUANTMINDX needs only to ensure master account signals are clean, consistent, and executable.
- Alpha Forge Loop develops **parallel variants** — one optimized for personal account growth (full Kelly), one optimized for prop firm compliance (conservative, challenge-rule-validated). Agents develop both simultaneously.

---

### 2. Broker Management (Extensible, Multi-Broker)

**Broker Registry** — add any broker without code changes:

- Broker name, MT5 account number, MT5 server address
- Account type: ECN (preferred for scalping) or Standard (market maker)
- Swap-free / Islamic account: yes/no (flagged during registration if not swap-free)
- Instrument allowlist per broker
- Minimum lot size per instrument
- Commission per lot (ECN accounts)
- Active/inactive status

**Multi-broker operation:** MT5 terminal connects to multiple broker accounts simultaneously (e.g., Exness + FBS + others). The MT5 bridge on Cloudzy reads from all connected accounts in parallel. The routing matrix determines which strategy executes on which broker account. Different strategies can run on different brokers based on account type, instrument support, and spread characteristics.

**Broker auto-detection:** MT5 bridge scans the MT5 terminal for all connected live accounts and surfaces them for Mubarak to add to the registry — no manual server address entry needed if already connected in MT5.

**Jurisdiction:** Non-US brokers only. FIFO rules do not apply. US leverage limits do not apply. System enforces no US-specific broker regulations.

---

### 3. Account & Money Architecture

QUANTMINDX is an **instruction layer, not a custodian.** Money never passes through the system:

- Each account = MT5 account number + broker server address in the routing matrix
- MT5 bridge reads: balance, equity, open positions, trade history — in real time via MT5 API
- Trade execution: Commander sends instruction → ZMQ → MT5 bridge → MT5 terminal → broker. Money moves at broker. QUANTMINDX sees the result.
- **What's visible:** Live equity curve, open P&L, balance per account — all read from actual broker accounts. Live Trading canvas shows real numbers.
- **Deposits/withdrawals:** Always done directly in the broker's app or MT5 terminal. QUANTMINDX does not initiate money movement.
- **Client accounts:** A client account = another MT5 account number in the routing matrix. Their money stays with their broker. QUANTMINDX knows the account exists, reads its state, routes trades to it, generates reports. No money custody.

---

### 4. Islamic Trading Compliance

- **No overnight positions (hard constraint):** Commander force-closes all open positions by a configurable daily cutoff time (default: 21:45 GMT, before MT5 22:00 swap processing). Applies to all accounts unless explicitly disabled per account. Intraday-only — no swing trades, no multi-day holds.
- **Swap-free accounts:** All broker accounts configured as Islamic/swap-free at broker level. System flags non-swap-free accounts during broker registration.
- **Effective 1:1 leverage:** Kelly-based position sizing calculates lot size purely on account equity percentage — does not utilize broker's available leverage multiplier. Conservative by design.
- **No interest-bearing instruments:** Bonds, interest rate products excluded. Forex, commodities, indices, and select halal stocks (when available via broker CFDs) permitted.
- **Stocks (future):** Islamic-compliant stocks supported via the Research Department's stock analysis capability when broker and MT5 account support the instrument. Not MVP scope but architecture supports it.

---

### 5. Alpha Forge Variant Architecture

**Vanilla vs. Spiced** (existing system, to be enhanced):
- **Vanilla:** Clean implementation of the Research hypothesis — source logic only
- **Spiced:** Vanilla + knowledge base enhancements + shared assets (accumulated system intelligence layered on top)

**Backtest matrix (runs for both variants):**

| Test Type | Vanilla | Spiced |
|---|---|---|
| Standard Backtest | V-STD | S-STD |
| Monte Carlo | V-MC | S-MC |
| Walk-Forward | V-WF | S-WF |
| System Integration Test | V-SIT | S-SIT |

**System Integration Test (SIT) — practical validation:**
Not a backtest — a compatibility test. Validates that the EA can:
- Register with the Strategy Router
- Respond correctly to Governor halt signals and risk scalars
- Receive regime data from Sentinel and react correctly
- Fire BotCircuitBreaker correctly on consecutive losses
- Communicate cleanly via ZMQ (MT5 bridge)
- Respect account-level risk rules assigned to it

Only EAs passing SIT proceed to paper trading. Failed SIT → Development receives specific integration failure report.

**Enhancement via journeys:** The 52 user journeys define richer variant management patterns (A/B testing, strategy racing, parameter sweep survivors, walk-forward validation) that supersede hard-coded backtest matrix logic. The matrix above is the foundation — journeys guide enhancements. Do not hard-code beyond this foundation.

**Variant tracking:** Alpha Forge tracks all 8 variants per strategy (minimum), keeps lineage, selects best-performing SIT-passing variant for paper trading. All others archived with test results — not deleted.

---

### 6. EA Tag Architecture (MQL5-Implemented)

Tags are not just metadata labels — they must be compiled EA inputs in every MQL5 template:

```mql5
// Required inputs on ALL EA templates
input string   session_mask        = "London,NY";    // Which sessions this EA trades
input int      force_close_hour    = 21;             // Hour to force-close all positions (GMT)
input int      force_close_minute  = 45;             // Minute to force-close
input bool     overnight_hold      = false;          // Must always be false (Islamic compliance)
input int      consecutive_loss_halt_pf = 3;         // Prop firm: halt after N consecutive losses
input int      consecutive_loss_halt_pb = 5;         // Personal book: halt after N consecutive losses
input double   spread_filter_pips  = 1.2;            // Max spread to allow new entry
input int      daily_max_trades    = 20;             // Max trades per day (0 = unlimited)
input double   daily_loss_cap_pct  = 2.0;            // EA-level daily loss cap (% of account)
input string   account_tag         = "personal";     // Which account this EA is assigned to
input string   strategy_version    = "V1";           // Version identifier for routing matrix
```

The routing matrix, Commander, Governor, and BotCircuitBreaker all reference these tags at runtime. Journeys that mention tags (session mask, overnight hold, BotCircuitBreaker thresholds) rely on these being real compiled inputs — not virtual labels.

---

### 7. Data Pipeline — Proprietary Historical Dataset

The existing live data collection pipeline serves a dual purpose:
1. **Immediate:** Real-time tick/OHLCV data feeds the Sentinel and strategy signals
2. **Long-term:** Accumulates into a proprietary historical dataset stored on Contabo cold storage

As months and years pass, QUANTMINDX builds its own historical data — eventually reducing dependency on Dukascopy for backtesting. The system uses Dukascopy for historical backtests until sufficient proprietary data is accumulated. New strategy backtests can then run against the proprietary dataset (real execution conditions from actual broker feeds rather than third-party historical data).

Data collection pipeline stores to Contabo via the same rsync mechanism used for scraped articles.

---

### 8. Audit Trail Requirements

Comprehensive audit trail across three layers — backend log, SQLite database record, and UI-accessible via Copilot query:

- **Trade audit:** Every entry/exit — timestamp, instrument, lot size, price, P&L, regime at time, strategy version, account, session, signal source
- **Strategy lifecycle audit:** Every state change — activated, paused, retired, rolled back, version increment — with timestamp and reason (human or system-triggered)
- **Risk parameter audit:** Every change to Governor settings, Kelly multiplier, risk mode, prop firm rules — who changed it, what changed, when
- **Agent action audit:** Every department task dispatch, sub-agent spawn, skill invocation — timestamped, with input/output summary
- **System health audit:** Every regime shift, Sentinel fallback, server connection change, provider failover, workflow execution event
- **Retention:** All audit logs archived to Contabo cold storage, minimum 3 years

Audit trail is directly useful for: Copilot workflow optimization, daily operational review, prop firm dispute resolution, client reporting, and the decision audit pattern (Journey 34).

---

### 9. EA Safety Architecture

**Primary safety mechanism:** BotCircuitBreaker — consecutive loss quarantine. Thresholds configurable per account (prop firm vs. personal book). On quarantine: Development Department receives the EA with a failure report, analyzes conditions, routes back with recommendation. Agentic setup handles the review autonomously.

**Trade frequency limits:** NOT enforced as a hard system-level constraint. If risk management and position sizing have validated a trade, the system executes it. The BotCircuitBreaker (consecutive loss threshold) is the actual runaway protection. High-frequency valid signals (e.g., scalping in fast market) are not blocked by a trades-per-second limit.

**EA-level safety (MQL5 base template):** Consecutive loss counter, spread filter, force-close time, daily max trades cap, daily loss cap — all compiled into every EA as configurable inputs. These are the EA's own safety mechanisms independent of the system-level Governor.

**Session mask:** A recommendation tag, not a hard block. Same EA logic can run as different versioned instances across different sessions and instruments. The Commander reads the session_mask input and activates the correct EA version at the correct time — it does not prevent deployment of an EA to a different session if Mubarak explicitly configures it.

---

### 10. Infrastructure Portability & System Updates

**All infrastructure configuration editable at runtime** via Settings → Connections — not hardcoded to any provider:
- Trading execution server (currently Cloudzy): configurable, migratable (Journey 28 pattern)
- Heavy compute / HMM training server (currently Contabo): configurable
- Cold storage target: configurable rsync destination
- Local IDE: always local, Tauri desktop

**System update mechanism — all three servers:**
- **Local IDE:** Tauri app update
- **Cloudzy:** git pull + FastAPI service restart + MT5 bridge restart
- **Contabo:** git pull + HMM training scripts + any services running there
- **Process:** Copilot triggers "Update system" → sequential pull + database migration + health check across all three servers → rollback if any step fails
- Database migrations run automatically on version update
- Zero-downtime updates targeted for Cloudzy (trading must continue during update)

**System backups:**
- SQLite operational DB: daily backup to Contabo
- Strategy database (all EA files, version history, backtest results): daily backup to Contabo
- HMM model files: backup after each weekly retraining
- Configuration (settings, provider config, broker registry, prop firm registry — encrypted): daily backup to Contabo
- Backup restore: Copilot can trigger restore from a single instruction

---

### 11. Client Fund Management (Africa Jurisdiction)

Mubarak is based in Africa. Regulatory burden for managing outside capital is significantly lower than US/UK/EU jurisdictions. No FCA, SEC, or ESMA registration required.

Adding a client account to QUANTMINDX = portfolio tracking only. No regulatory flags, no compliance gates. System tracks their capital, generates their reports, routes strategies to their account. Legal review is Mubarak's responsibility external to the system.

Client accounts are structurally identical to personal accounts in the routing matrix — a separate MT5 account number with its own risk profile, Kelly multiplier, and drawdown cap. The money stays with their broker. QUANTMINDX reads it and trades it.

Future hedge fund model: architecture already supports it without changes — separate account namespaces, per-account risk profiles, automated client reporting, strategy eligibility filtering per account.

---

### 12. Domain Risk Register

| Risk | Mitigation |
|---|---|
| Prop firm rule violation | Governor hard stops per account (configurable per firm rule set); challenge mode validation in Alpha Forge SIT |
| Overnight position left open | Force-close at 21:45 GMT hard-coded in every EA template; Commander monitors for any positions remaining at cutoff |
| Broker credential compromise | ZMQ CURVE encryption, SSH key-only on all servers, secrets manager, never in git |
| Runaway EA | BotCircuitBreaker (configurable consecutive loss quarantine), Governor position limits, kill switch |
| Strategy curve-fitting (passes backtest, fails live) | Walk-Forward + Monte Carlo + SIT validation, paper trade gate, hypothesis contradiction monitoring (Journey 43) |
| Infrastructure provider outage | Sentinel fallback mode, configurable server migration, daily database backups to Contabo |
| Copy trade detection by prop firm | QUANTMINDX trades master account only; copy trading is external and operator-managed outside the system |
| Data quality (broker data manipulation) | Dukascopy for historical backtests (independent source); proprietary data pipeline builds over time; demo account for live system testing |
| System update failure | Sequential update with health check + rollback; Contabo and Cloudzy both covered |


---

## Innovation & Novel Patterns

### Detected Innovation Areas

**1. The Alpha Forge Loop — Autonomous Strategy Factory**

No retail trading platform autonomously develops, tests, and deploys trading strategies without human coding involvement. The existing approach is: human writes strategy → human backtests → human deploys. QUANTMINDX inverts this entirely: human provides a source (video, PDF, idea, or nothing) → AI departments produce a production-ready, backtested, paper-traded EA. The operator's role shifts from builder to overseer.

The combination that makes this novel: LLMs capable of long-form MQL5/Python coding (2025–2026 capability threshold) + agentic sub-agent parallelism + Walk-Forward/Monte Carlo validation + autonomous parameter sweeps + paper trade monitoring = a strategy factory that runs itself.

**2. Physics-Aware Risk Engine in Retail Trading**

Ising Model (spin magnetization → market correlation), Hidden Markov Model (regime classification), and Lyapunov Exponent (chaos/instability detection) are tools of academic econophysics and institutional quantitative research. Integrating all three into a real-time regime detection layer in a personal trading terminal — and wiring them directly to position sizing (Kelly Engine) and strategy activation (Commander) — is genuinely novel at the retail level.

No retail platform (MT5, cTrader, NinjaTrader, TradingView) offers physics-based regime detection. Even most hedge funds use only one of these methods, not all three as a composite signal.

**3. Self-Extending Agentic System (Skills That Write Themselves)**

The Skill Forge (Journey 50) describes something that hasn't been widely demonstrated in production: a system that identifies its own repeated patterns, designs new tools to address them, validates those tools, and registers them for autonomous use — without the operator writing code. The system becomes more capable over time through its own initiative. This is an early, practical implementation of self-extending AI systems.

**4. The Solo Trader / Institutional Parity Thesis**

The core insight driving the entire product: in 2026, LLMs can do what a research team, quant dev team, and risk team used to require. One person with the right environment can operate at institutional scale. QUANTMINDX is the environment.

This is not a gradual improvement on existing retail tools — it is a category shift. The competition is not TradingView or MetaTrader. The competition is a small quant fund.

**5. Department Canvas UX — AI Orchestration as Interface**

Most AI tools present a chat interface. QUANTMINDX presents a workspace model where AI departments are peers — they have inboxes, they send reports, they dispute each other's findings, they get assigned tasks. The Global Copilot Bubble is not a chatbot; it is an operating layer. This is a novel UX paradigm for human-AI collaboration in high-stakes domains.

**6. Session-Aware Multi-Type Portfolio Orchestration**

The Commander simultaneously manages scalpers (session-locked, spread-filtered, regime-gated), trend-followers (continuous, TREND-only, wider sizing), mean-reversion (RANGE-only, auto-paused when regime shifts), and event-driven strategies (expiry-tagged, template-deployed) — all with different activation rules per regime, per session, per account, in real time.

No retail tool operates at this level of simultaneous portfolio orchestration. Most platforms run strategies independently with no awareness of each other. The Commander treats the portfolio as a living system — activating, pausing, and rebalancing strategy types based on real-time market physics.

**7. Knowledge Provenance Chain — Research Attribution**

Every strategy in QUANTMINDX has a full traceable intellectual lineage: the scraped article or PDF or personal observation that inspired it, the Research Department's confidence scoring on the hypothesis, the Development Department's autonomous build, every human interaction point (code review comments, approval decisions), every version increment and its reason.

In institutional research, this is called "research attribution" — knowing exactly where an idea came from and how it evolved. It is a hard, expensive problem in professional quant research. In QUANTMINDX, it emerges naturally from the agentic architecture as a built-in property of how departments communicate and how the Alpha Forge Loop tracks lineage. The system knows where every idea came from.

---

### Market Context & Competitive Landscape

| Category | Existing Tools | QUANTMINDX Position |
|---|---|---|
| Retail trading platforms | MT5, TradingView, cTrader | Not a platform — a terminal that operates the platform |
| Algo trading tools | QuantConnect, Zipline, Backtrader | Backtesting only, no autonomous development loop |
| AI trading bots | 3Commas, Cryptohopper | Pre-built bots, no autonomous strategy creation |
| Quant research tools | Bloomberg, Refinitiv, Kensho | Institutional only, $50k+/year, no autonomous loop |
| AI coding assistants | GitHub Copilot, Cursor | Code only — no trading domain, no autonomous deployment |
| Multi-agent frameworks | LangGraph, CrewAI | General purpose — no trading domain, no physics risk layer |

**The gap:** Nothing automates the full loop from strategy source → production deployment with AI departments doing the intermediate work, while simultaneously running a physics-aware risk engine and orchestrating a heterogeneous live portfolio. QUANTMINDX targets this gap for a single operator.

---

### Validation Approach

| Innovation | Validation Method |
|---|---|
| Alpha Forge Loop | First complete autonomous cycle (source → paper trade) without Mubarak coding anything — measured in Success Criteria |
| Physics risk engine | Regime-triggered auto-pause prevents losses during High Chaos events (Journey 23 is the live test case) |
| Self-extending skills | Measure: skill count at month 1 vs. month 6, percentage generated autonomously vs. by Mubarak |
| Solo/institutional parity | Prop firm challenge pass rate — one pass validates the thesis commercially |
| Department canvas UX | ≤2 clicks to any feature, ≤30 min/day monitoring, Copilot context accuracy — all in Success Criteria |
| Multi-type orchestration | System stable with 25+ strategies across 4 types simultaneously — measured in Business Success Criteria |
| Knowledge provenance | Every live strategy has a traceable lineage chain — verifiable via Copilot provenance query (Journey 40) |

---

### Risk Mitigation

| Innovation Risk | Mitigation |
|---|---|
| Alpha Forge Loop produces overfit strategies | Walk-Forward + Monte Carlo + SIT + paper trade gate — four sequential filters before live |
| Physics engine gives false signals | Fallback to cached regime on sensor failure (Journey 8); human override always available |
| Self-extending skills introduce bugs | Skill acceptance test suite required before registration (Journey 50); skill versioning with rollback |
| LLM capability regression (model changes) | Multi-provider support — swap tiers if one provider degrades; three-tier hierarchy provides flexibility |
| Multi-type orchestration conflicts | Correlation gate prevents strategy overlap; regime filters prevent wrong-type activation |
| Knowledge provenance gaps | All agent communications logged to department mail with timestamps; audit trail is mandatory architecture |


## Platform Architecture & Infrastructure Requirements

### Project-Type Overview

QUANTMINDX ITT operates on a **three-node cloud-native architecture** with strictly separated concerns. The user's local machine is a **thin client access terminal** — not a host for any backend service.

| Node | Role | Provider | Always On |
|------|------|----------|-----------|
| **Contabo — Agent, Dev & Data Server** | All AI agents, development, research, knowledge base, Alpha Forge, HMM training, cold storage, logging, monitoring, Department Mail | Contabo (existing, upgradeable) | ✅ 99%+ SLA |
| **Cloudzy — Trading Node** | MT5 bridge, Strategy Router, Sentinel/Governor/Commander, live execution | Cloudzy | ✅ 99%+ SLA |
| **User's Machine** | ITT frontend (Tauri), thin client access terminal | Local — any platform | ⚠️ Not required for continuity |

**Design principle**: Every autonomous process in every user journey runs on a cloud node. The user's machine going offline, losing power, or being unavailable has zero impact on system operation.

---

### Why Cloud-Hosted Agents (Architecture Decision)

The 52+ user journeys include continuous autonomous activity: research loops, Alpha Forge development sprints, knowledge indexing, regime monitoring, skill forging, portfolio rebalancing analysis, and logging. These cannot tolerate interruption. Local hosting in an environment with inconsistent power, internet, and physical presence cannot provide the reliability these workloads require.

**Decision**: All agentic processes are hosted on Contabo. The existing Contabo instance (already running HMM training and cold storage) will be upgraded to accommodate the full agent workload. This is an incremental addition to proven infrastructure.

**Security posture for cloud-hosted agents** (required):
- Full disk encryption on Contabo server
- SSH key-only access — no password authentication
- All proprietary strategy code in private repository
- Agent memory and knowledge base encrypted at rest
- Regular encrypted backups to secondary location

---

### Contabo — Agent, Dev & Data Server

Hosts the complete agentic intelligence and data layer of the system:

- FastAPI main backend (all non-trading endpoints)
- Floor Manager + all Department Heads + Sub-agent spawning
- Department Mail bus (SQLite)
- Alpha Forge full loop (Research → Dev → Optimize → Backtest → Paper → Live)
- Knowledge base (all partitions: scraped, ingested, personal, system-generated)
- Shared Assets library
- **HMM training** (existing — regime model updates, Lyapunov, Ising sensor retraining)
- **Cold storage** (existing — historical data archive, 3-year audit trail, log retention)
- Logging, monitoring, full audit trail
- Cron jobs and scheduled agent tasks
- Skill registry and Skill Forge

**Upgrade path**: Existing Contabo instance extended with additional compute/RAM as agent workloads grow. Migration to alternative provider (Hetzner, OVH, or equivalent) is supported by the infrastructure portability journey.

---

### Cloudzy — Trading Node

Strictly scoped to live trading execution:

- MT5 Bridge (ZMQ tick stream)
- Strategy Router: Sentinel → Governor → Commander
- Kill Switch hierarchy (ProgressiveKillSwitch + SmartKillSwitch + BotCircuitBreaker)
- Live position and order state
- Trading-only FastAPI endpoints

Cloudzy does **not** host agents, development work, knowledge base, or any private intellectual property.

---

### User's Machine — Thin Client Terminal

The ITT desktop (Tauri 2 + SvelteKit) is a pure frontend connecting to Contabo and Cloudzy APIs. It hosts no backend processes.

| Platform | Status | Build Method |
|----------|--------|--------------|
| Linux | ✅ Primary | `tauri build` from source |
| Windows | ⬜ Preview | `tauri build` from source |
| macOS | ⬜ Optional | `tauri build` from source |

**Machine migration**: When switching machines, only the ITT frontend needs rebuilding from source. All agent state, knowledge base, strategy library, and configuration lives on Contabo — nothing is lost on machine change.

---

### Mobile Companion (Phase 2)

Away-from-desk monitoring and command interface. Not a development tool.

| Feature | Included | Notes |
|---------|----------|-------|
| System health dashboard | ✅ | Both nodes |
| HMM regime indicator | ✅ | Live |
| Bot status + P&L summary | ✅ | Live |
| Full kill switch | ✅ | All tiers — soft through hard stop |
| Customisable notifications | ✅ | User-defined events, thresholds, channels |
| Department Mail inbox | ✅ | Agent communications |
| Copilot chat | ✅ | Full orchestration — trigger Alpha Forge, give instructions |
| Shared assets browser | ✅ | Read access |
| Code editor / Alpha Forge UI | ❌ | Desktop only |
| Full knowledge base UI | ❌ | Desktop only |

**Always reachable**: Mobile connects directly to Contabo backend — no dependency on user's machine.

---

### API Architecture

**Contabo FastAPI** (port 8000): All agent, development, research, settings, knowledge, and system endpoints.

**Cloudzy FastAPI** (port 8000): Trading execution, live position, Strategy Router, kill switch endpoints.

**API Versioning**: Flat `/api/` prefix maintained. No URL versioning — internal consumers only.

---

### Authentication

**Phase 1**: No auth. Both nodes restrict via firewall (trusted IPs). SSH key-only for server access.

**Phase 2**: JWT scoped tokens — `admin`, `viewer`, `client-report-only`.

---

### Notifications

Fully customisable per event type:
- User configures which events trigger notifications (regime change, kill switch, Alpha Forge completion, agent task complete, P&L threshold, drawdown alert, etc.)
- Severity and delivery channel (OS tray, mobile push in Phase 2, Department Mail)
- Per-event thresholds configurable

---

### Connectivity & Resilience

| Scenario | Impact |
|----------|--------|
| User's laptop off | Zero — all agents run on Contabo |
| User on road trip | Mobile companion: full monitoring, kill switch, Copilot |
| Cloudzy down | Trading halted; Contabo agents continue all other work |
| Contabo down | Agents pause; trading continues on Cloudzy independently |
| Both nodes down | Full recovery on reconnect — no backlog accumulation by design |


## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**Approach**: Journey-Coverage MVP — Brownfield Acceleration

This is a brownfield project. The complex foundations — physics risk engine, strategy router, MT5 bridge, agent department system, backtesting engine, knowledge pipeline — already exist in significant form. The next major release is not about building from scratch. It is about connecting, rebuilding, and proving that the system can execute its defined user journeys end-to-end.

**The measure of done**: If the system executes its user journeys reliably, it is robust, unique, and production-ready. Feature lists are secondary. Journey coverage is primary.

**Total goal**: 100% of all 52 defined journeys across Phase 1 and Phase 2. No journey is optional — they are all being built.

**Phase 1 target**: 90% of the 52 journeys (≈47 journeys) operational end-to-end. This is not a compromise — it is the natural Phase 1 ceiling. The remaining ≈5 journeys have external prerequisites that cannot be built around (mobile app, 6-month verified track record, client infrastructure). They are Phase 2 by definition, not by difficulty.

**Resource profile**: Solo developer with existing brownfield codebase. This changes how scope is managed — working over perfect, journeys over polish, depth per sprint.

**Post-PRD decomposition path**: The PRD defines "what" and "why." Implementation decomposition follows via:
1. `/bmad-bmm-create-ux-design` — every canvas, panel, and interaction designed from this PRD
2. `/bmad-bmm-create-architecture` — system architecture decisions, backend capability mapping
3. `/bmad-bmm-create-epics-and-stories` — each journey broken into implementation-ready epics and user stories (backend + frontend tasks)

---

### Phase 1 — Next Major Release (~90% Journey Coverage)

**Objective**: Prove the system executes its daily, autonomous, and agentic workflows reliably. ITT redesign complete. Copilot orchestrating the floor. Alpha Forge cycle running. Trading executing reliably on the cloud-native 3-node architecture.

**UI & Platform Foundation (prerequisite for all journeys):**
- Complete UI redesign — Svelte 5 runes migration, canvas/department navigation, Frosted Terminal aesthetic
- Global Copilot Bubble — persistent, context-aware, Anthropic Agent SDK backed
- Live Trading Canvas as home screen
- StatusBand as global nerve bar
- Cloud-native migration — all agents on Contabo, Cloudzy strictly for trading execution

**Journey Categories in Phase 1:**

| Category | Coverage |
|----------|---------|
| Daily trading operations (morning operator, live session, prop firm compliance) | ✅ Phase 1 |
| Autonomous overnight (night watch, research loop, Alpha Forge overnight) | ✅ Phase 1 |
| Alpha Forge — Vanilla + Spiced pipeline, full backtest matrix, live promotion gate | ✅ Phase 1 |
| Risk management (HMM regime, physics engine, BotCircuitBreaker, drawdown monitoring) | ✅ Phase 1 |
| Strategy development (EA code gen, PineScript, MQL5 compilation via MT5 sync) | ✅ Phase 1 |
| Research (market research, knowledge base query, news/event ingestion) | ✅ Phase 1 |
| Knowledge management (PageIndex indexing, Firecrawl scraping, provenance chain) | ✅ Phase 1 |
| Multi-broker (registry, routing matrix, account tagging, Islamic compliance) | ✅ Phase 1 |
| Prop firm compliance (FTMO/The5ers/FundingPips rules, force-close, swap-free) | ✅ Phase 1 |
| Agentic skills (skill registry, department chaining, Skill Forge) | ✅ Phase 1 |
| Agent monitoring (Department Mail, activity feed, task queue, audit trail) | ✅ Phase 1 |
| System operations (3-server update, cold storage sync, provider settings, notifications) | ✅ Phase 1 |

**Production-Critical (non-negotiable for Phase 1 release):**

| Capability | Reason |
|-----------|--------|
| Live Trading Canvas + Strategy Router | System is inert without it |
| Kill Switch — all tiers | Prop firm requirement — hard stop capability |
| BotCircuitBreaker per-bot | First line of defence, especially on funded accounts |
| HMM regime detection live | Risk engine backbone |
| Alpha Forge Vanilla path end-to-end | Core value proposition |
| Copilot orchestration (Agent SDK) | The differentiator — without it, a dashboard, not a platform |
| Multi-broker routing matrix | Operational necessity for simultaneous Exness + FBS |
| Department Mail bus | Agent coordination backbone |

---

### Phase 2 — Growth (~10% remaining journeys + expansion)

**Objective**: External capital, mobile access, client management, and commercial readiness.

**Journeys completing the 100%:**
- Mobile Companion app (monitoring dashboard, full kill switch, Copilot chat, Department Mail)
- Client fund management — reporting, performance attribution, Africa-jurisdiction compliance
- DarwinEx external capital pathway — 6-month verified track record export, DARWIN instrument setup
- Copy trading orchestration — master prop firm account → funded account replication
- Infrastructure portability / migration to new server or provider

**Additional Phase 2 features:**
- JWT authentication (admin / viewer / client-report-only)
- N8N-style visual workflow editor in Workshop canvas
- Financial Datasets API integration (institutional flow, income statements)
- Geopolitical Intelligence sub-agent fully autonomous
- Fully customisable notification system (OS + mobile push)
- Wallpaper + ambient lo-fi audio system

---

### Phase 3 — Vision

**Objective**: Commercial extension beyond personal use, if system is profitable and stable.

- Smart money / institutional flow tracking ML layer
- Prop firm challenge automation (Alpha Forge tuned for challenge evaluation rules)
- Commercial productization for other prop firm traders
- QuantMind DEX — separate crypto project (deferred until QUANTMINDX stable)
- Voice interface (deferred — no audio model budget currently)
- External API versioning for third-party integrations

---

### Risk Analysis

**Technical Risks:**

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Svelte 5 runes migration breaking existing components | Medium | Component-by-component migration; working state maintained at each step |
| Anthropic Agent SDK sub-agent spawning reliability | Medium | Existing department system already wired; SDK replaces LangGraph cleanly |
| Contabo cloud-native refactor (moving agents to server) | Medium | Brownfield foundation minimises unknowns; architecture already designed |
| HMM regime model accuracy without retraining | Low | Contabo training cron already exists; Lyapunov exponent cross-checks |
| MT5 bridge ZMQ stability on Cloudzy | Low | Existing and tested; reconnection logic in place |

**Resource Risks:**

| Risk | Mitigation |
|------|-----------|
| Solo developer — velocity constraints | Journey-coverage approach prioritises working flows over polish |
| External timeline pressure | Phase 1 = ≥90% journey coverage — a concrete, measurable milestone |
| Scope creep | PRD is the scope boundary; new ideas route to Phase 2 backlog |

**Market Risks**: Minimal. QUANTMINDX is a personal platform. The user is the market. No external validation risk.


## Functional Requirements

_This is the binding capability contract for QUANTMINDX ITT. UX designers will only design what is listed here. Architects will only support what is listed here. Epics and stories will only implement what is listed here. Any capability not listed will not exist in the final product._

### 1. Live Trading & Execution

- **FR1**: The trader can view all active bot statuses, live P&L, and open positions in real time
- **FR2**: The system can execute trade entries and exits via the MT5 bridge on behalf of configured bots
- **FR3**: The trader can manually close any open position directly from the ITT
- **FR4**: The system can route all strategy signals through the Sentinel → Governor → Commander pipeline before any order execution
- **FR5**: The trader can view the active market regime classification and strategy router state at all times
- **FR6**: The system can activate strategies based on session context (London, New York, Asian, Swiss, custom) per the session mask configured in each EA's input parameters
- **FR7**: The trader can monitor live tick stream data and spread conditions per instrument in real time
- **FR8**: The system can enforce daily loss caps and maximum trade limits at the aggregate account level; each EA enforces its own force-close timing and overnight hold rules autonomously via MQL5 input parameters
- **FR9**: The trader can activate any tier of the kill switch — soft stop through full emergency position close — at any time from any canvas

### 2. Autonomous Agent System

- **FR10**: The Copilot can receive natural language instructions and orchestrate the appropriate department, agent, or workflow response
- **FR11**: The Floor Manager can classify incoming requests and route them to the appropriate Department Head without trader intervention; the Floor Manager can also handle administrative tasks directly when appropriate
- **FR12**: Each Department Head can spawn parallel sub-agents to handle complex multi-part tasks concurrently
- **FR13**: The Copilot can create, schedule, and manage automated workflows — cron jobs, hooks, triggers — on behalf of the trader
- **FR14**: The system can maintain persistent agent memory across sessions — conversation context, strategy knowledge, and trader preferences
- **FR15**: Department agents can communicate with each other via the Department Mail bus asynchronously
- **FR16**: The Copilot can surface Department Mail notifications and agent task completions to the trader
- **FR17**: The trader can review agent activity, task queues, and sub-agent spawning history
- **FR18**: The system can register and execute agentic skills as composable, reusable capabilities across department workflows
- **FR19**: Department Heads and the Copilot can author new agentic skills and register them to the skill library (Skill Forge)
- **FR20**: The Copilot can operate context-aware on any canvas, with tools and commands appropriate to the active department
- **FR21**: The Copilot can maintain a task list and route at least 5 simultaneous tasks to different agents and tools concurrently, with delegation to the Floor Manager for tasks that do not require Copilot-level orchestration
- **FR22**: The trader can directly converse with any individual Department Head via the Department Chat panel

### 3. Alpha Forge — Strategy Factory

> **Architecture note (added 2026-03-14):** Alpha Forge is now architecturally defined as two independent workflows — see architecture.md §20.
> - **Workflow 1 (Creation):** video/source → TRD → SDD → build → compile → basic backtest → EA Library. Runs once per source. Ends when variants are deposited.
> - **Workflow 2 (Enhancement Loop):** full backtest suite (all 6 modes) → research-driven improvement → Dukascopy data variant stress testing → SIT gate (MODE_C) → paper trading → human approval → live. Runs continuously and independently.
> The journeys below remain valid as written — the user experience is unchanged. The split is a backend workflow boundary only.

- **FR23**: The system can execute a complete Alpha Forge loop from research hypothesis to live deployment candidate without mandatory manual intervention at intermediate stages
- **FR24**: The Research Department can autonomously generate strategy hypotheses from knowledge base content, news events, and market data
- **FR25**: The Development Department can generate MQL5 EA code from a strategy specification, including all required EA input tags
- **FR26**: The system can produce Vanilla and Spiced EA variants from the same base strategy specification
- **FR27**: The system can run a full backtest matrix — Standard, Monte Carlo, Walk-Forward, and System Integration Test — per EA variant
- **FR28**: The Risk Department can evaluate backtest results and present a promotion recommendation; a human trader must explicitly approve before any EA advances beyond paper trading to live deployment
- **FR29**: The Trading Department can monitor paper trading performance of a new EA before live promotion is available for human approval
- **FR30**: The trader can review, modify, or reject any Alpha Forge output at each gate in the pipeline
- **FR31**: The system can maintain a versioned strategy library with full knowledge provenance chain per strategy — from source article through to live deployment

### 4. Risk Management & Compliance

- **FR32**: The physics risk engine can classify the market regime (TREND / RANGE / BREAKOUT / CHAOS) in real time using the Ising Model and Lyapunov Exponent sensors as the primary signal sources
- **FR33**: The HMM sensor can run in shadow mode — observing and logging regime classifications — without its output controlling the strategy router until the model is validated and deemed fit
- **FR34**: The Ising Model sensor can detect systemic correlation risk across assets
- **FR35**: The Lyapunov Exponent sensor can detect chaotic instability conditions and trigger risk escalation
- **FR36**: The Physics-Aware Kelly Engine can calculate position sizes with regime-based physics multipliers and fee awareness (commission, spread cost, swap); when account equity exceeds the initial deposit by a configured threshold, it applies house-of-money scaling to reflect the expanded trading buffer
- **FR36b**: The Portfolio Department can monitor account equity state against configured initial deposit thresholds and propagate house-of-money status system-wide — adjusting the Governor risk scalar, Commander strategy activation profile, Alpha Forge promotion criteria, and Copilot contextual awareness accordingly
- **FR37**: The BotCircuitBreaker can automatically quarantine any bot reaching its configured consecutive loss threshold — configurable per account tag (prop firm vs. personal book)
- **FR38**: The Governor can enforce aggregate account-level compliance rules across all active bots simultaneously — total drawdown limits, daily loss caps, and prop-firm-specific halt conditions — operating on the full account picture that no individual EA can access
- **FR39**: The system can enforce Islamic trading compliance rules — no overnight positions, swap-free accounts, effective 1:1 leverage — per account configuration
- **FR40**: Each EA autonomously manages its own per-trade parameters — spread filter, force-close timing, session mask, overnight hold — via MQL5 input tags; the system displays these parameters for transparency and does not impose independent overrides
- **FR41**: The trader can configure risk parameters per account tag independently

### 5. Knowledge & Research

- **FR42**: The system can scrape and ingest financial articles and market research from configured external sources
- **FR43**: The Research Department can query the knowledge base using semantic search to retrieve relevant strategy context
- **FR44**: The system can index all knowledge base content for full-text and semantic retrieval via the knowledge indexing system
- **FR45**: The trader can add personal knowledge entries, research notes, and strategy observations to the knowledge base
- **FR46**: The system can maintain provenance metadata for every knowledge entry — source, date, relevance, and linked strategy
- **FR47**: The system can ingest video content — individual videos or playlists — via the video ingest pipeline and extract strategy signals from transcripts
- **FR48**: The system can partition the knowledge base by type — scraped, ingested, personal, system-generated — with appropriate access per partition
- **FR49**: The trader can query the knowledge base via Copilot using natural language
- **FR50**: The ITT can display a live news feed with enriched macro context for the trader to monitor during active sessions

### 6. Portfolio & Multi-Broker Management

- **FR51**: The trader can register and configure at least 4 broker accounts simultaneously in the broker registry
- **FR52**: The system can auto-detect MT5 account properties — broker, account type, leverage, currency — on connection
- **FR53**: The routing matrix can assign strategies to specific broker accounts based on account tag, regime, and strategy type
- **FR54**: The system can manage simultaneous operation of multiple strategy types — scalpers, trend-followers, mean-reversion, event-driven — across different broker accounts and sessions
- **FR55**: The trader can review portfolio-level performance metrics — total equity, drawdown, P&L attribution per strategy and per broker
- **FR56** *(Phase 2)*: The system can generate client performance reports with P&L attribution, drawdown analysis, and trade history
- **FR57** *(Phase 2)*: The trader can configure copy trading replication from a master account to other funded accounts
- **FR58** *(Phase 2)*: The system can prepare a verified track record export for DarwinEx capital allocation application

### 7. Monitoring, Audit & Notifications

- **FR59**: The system logs all system events at the appropriate level — debug, info, warning, error, critical — with no events silently excluded
- **FR60**: The trader can query the complete event log and audit trail via Copilot using natural language
- **FR61**: The system can retain all logs for a minimum of 3 years with automated cold storage sync to the agent/data server
- **FR62**: The trader can configure which events trigger active notifications, at what severity, and via which delivery channel
- **FR63**: The system can deliver notifications via OS system tray; mobile push delivery available in Phase 2
- **FR64**: The trader can access all agent communications and task outputs via the Department Mail inbox
- **FR65**: The system can monitor and report server health, connectivity, and latency for both the agent server and trading server nodes in real time

### 8. System Management & Infrastructure

- **FR66**: The trader can configure AI provider settings — model tier, API key, base URL — and swap providers at runtime without system restart
- **FR67**: The Copilot can trigger a sequential system update across all three nodes with health check and automatic rollback on failure
- **FR68**: The trader can add, modify, or remove prop firm registry entries and their associated compliance rule sets
- **FR69**: The trader can rebuild the ITT on a new machine from source without loss of agent state, strategy library, or configuration
- **FR70**: The trader can migrate server-side infrastructure to a new provider without loss of data or functionality
- **FR71**: The system can manage scheduled background tasks on the agent/data server, configurable via Copilot
- **FR72**: The trader can build the ITT on Linux, Windows, and macOS from source
- **FR73** *(Phase 2)*: The trader can access a mobile companion interface for system monitoring, full kill switch, and Copilot chat when away from the primary machine

### 9. Traceability Gap Closures

- **FR74**: The system can run parallel A/B comparisons of strategy variants on paper or live accounts with statistical confidence scoring and a live race board
- **FR75**: The trader can roll back a live strategy to any previous version in the versioned strategy library with one instruction
- **FR76**: The system can extract lessons from retired or underperforming strategies and propagate pattern-matched fixes across active strategies sharing the same base logic
- **FR77**: The system can integrate economic calendar events and apply calendar-aware trading rules — lot scaling, entry pauses, post-event reactivation — per account or strategy configuration
- **FR78**: The Copilot can explain its reasoning chain for any past recommendation or action when asked, surfacing the factors, data sources, and decision logic that contributed to the output
- **FR79**: The system can deploy compiled EA files to the MT5 terminal via the bridge — including chart attachment, parameter injection, health check, and ZMQ stream registration


## Non-Functional Requirements

### Performance

The performance contract for QUANTMINDX is about **protocol correctness and responsiveness**, not raw millisecond targets. The kill switch value is in executing the right tier of the protocol — controlling the bleeding progressively — rather than instant brute-force closure.

- The kill switch protocol must execute its configured tier (soft stop / progressive / full close) in full, in order, without skipping steps — correctness over raw speed
- The Live Trading Canvas and StatusBand must reflect live P&L, regime, and bot state with no more than 3 seconds of lag under normal network conditions
- Copilot responses must begin streaming within 5 seconds of request submission
- Canvas transitions must complete within 200ms; UI interactions must respond within 100ms
- HMM shadow mode classification must update within one tick cycle of new data arriving
- Backtest matrix (Standard + Monte Carlo + Walk-Forward + SIT) for a standard 1-year dataset must complete within 4 hours

### Security

Two-tier security model matching the two-tier infrastructure:

**Contabo (Agent + Dev Server — high security):**
- SSH key-only access — no password authentication, no root login
- Provider-level DDoS protection relied upon (Contabo handles at infrastructure layer)
- All proprietary intellectual property — EA source code, Alpha Forge output, knowledge base, shared assets, strategy library, MQL5 reference PDFs — stored exclusively on private infrastructure; never exposed in public repositories or transmitted raw to third-party AI APIs in agent prompts
- All API keys and credentials in `.env` files only — never in source code or version control
- Agent memory, knowledge base, and strategy files treated as confidential — access restricted to authenticated server processes

**Cloudzy (Trading Node — standard security):**
- Firewall-restricted to trusted IPs (Contabo agent server + developer machine)
- No sensitive intellectual property stored on this node
- Trading credentials managed by MT5 terminal — not stored in application layer

### Reliability & Availability

- Both Contabo and Cloudzy nodes rely on provider SLA as the availability baseline — this is the rationale for cloud-hosting over local infrastructure
- Planned maintenance handled via provider email notification; the system update and migration workflow supports graceful handoff before scheduled downtime
- Individual component failure (MT5 bridge restart, agent crash, API timeout) must not cascade to full system failure — components recover independently
- MT5 bridge must reconnect automatically after disconnection without manual intervention
- WebSocket subscriptions (tick feed, agent stream) must auto-reconnect with exponential backoff
- Periodic data backups of all Contabo state (agent memory, knowledge base, strategy library, SQLite databases) to a secondary backup location — automated in later stages; manual initially
- The strategy router and kill switch must remain operational on Cloudzy even if the Contabo agent server is temporarily unreachable

### Data Integrity & Tiered Data Architecture

- Trade records and position data must be persisted to SQLite before any system acknowledgment of execution
- Audit log entries are immutable once written — no deletion, no modification
- Strategy versions and backtest results must be associated with the exact EA code version that generated them — provenance chain preserved on all updates
- Knowledge base ingestion must preserve full provenance metadata (source, date, linked strategy) on every entry
- Cold storage sync to Contabo must include integrity verification — corrupted or incomplete transfers flagged and retried

**Tick data pipeline — tiered storage:**
- **Hot**: Live tick data in-memory / fast storage — active session, real-time strategy router input
- **Warm**: Recent historical tick data (rolling window, weeks to months) — fast-access DuckDB for backtesting and regime calibration
- **Cold**: Long-term tick archive on Contabo — multi-year dataset for HMM training, walk-forward analysis, and proprietary historical data accumulation

### Integration Reliability

- All configured AI provider APIs (Anthropic, OpenRouter, GLM, MiniMax, and others registered at runtime) must implement timeout handling and retry logic — failures queue rather than crash
- The Anthropic Agent SDK (agent orchestration runtime) is distinct from the Anthropic API (model inference); both must degrade gracefully on connectivity issues
- MT5 ZMQ connection loss must be detected within 10 seconds and trigger automatic reconnection
- Provider API rate limit or quota errors must be handled gracefully — agent tasks degrade to queued state, not failure state
- Firecrawl, PageIndex, Gemini CLI, and QWEN pipeline failures must be logged and retried without blocking other system operations

### Maintainability

- Existing LangChain and LangGraph code is acknowledged technical debt — it is to be migrated to the Anthropic Agent SDK department system during the ITT rebuild; no new LangChain or LangGraph code is introduced
- The Anthropic Agent SDK + department paradigm (FloorManager → Department Heads → Sub-agents) is the sole canonical agent architecture going forward
- All Python backend files kept under 500 lines — refactor at boundary
- All Svelte components kept under 500 lines
- New agent capabilities added via skill registration, not modification of core department code
- Risk-critical components (Kelly Engine, BotCircuitBreaker, Governor, Sentinel) must have test coverage before any modification
- All FastAPI endpoints follow the established router-per-module pattern

_Accessibility and user-base scalability are not applicable — QUANTMINDX is a single-user personal platform._

