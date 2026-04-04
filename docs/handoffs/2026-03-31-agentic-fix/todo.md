# QuantMindX Agentic/UI TODO Tracker (Production, No Mock Data)

Last updated: 2026-04-04

## Active Correction Track (contract-first)

- [x] P0: Canonicalize active agent identities (`floor_manager`, department heads) and keep legacy aliases compatibility-only.
- [x] P0: Manifest-first chat-context normalization in chat endpoints (strip heavy payloads).
- [x] P1: Workspace Resource Contract backend (manifest/search/read) for natural resource discovery across canvases.
- [x] P1: Session contract normalization (`interactive_session`, `workflow_session`) at session creation/update boundaries.
- [x] P1: Standardized mutating tool return envelope with `ui_projection_event` (task/mail/workflow/resource updates).
- [x] P1: Replace monolithic department/floor-manager prompt defaults with slim base prompts plus dynamic skill/MCP/memory/compaction/session contracts.
- [x] P2: Frontend attachment/natural-search wiring to consume backend resource-contract endpoints.
- [x] P2: AgentPanel right-rail resize behavior with per-canvas width persistence and keyboard accessibility.
- [x] P2: AgentPanel visual step/status colors decoupled from hardcoded blue; now uses theme-correlated accent variables.
- [x] P2: AgentPanel internal scroll/overflow hardening for long attachment/session/message lists.
- [x] P2: Structured streaming status events parity across department and floor-manager chat paths.
- [x] P2: AgentPanel mail list/detail workspace now marks messages as read on open via live backend read-state endpoint.
- [x] P2: AgentPanel tool/thinking stream rows now render as compact collapsible events instead of permanently expanded broad lines.
- [ ] P2: Full Chrome navigation verification across all canvases for session isolation, attachments, live update projections, and mail send/receive.
  - `Research -> send_mail -> Portfolio MAIL inbox` is now verified live.
  - `Portfolio -> MAIL` detail workspace and session-history rendering were re-verified live on 2026-04-04.
  - Remaining Chrome verification work is broader cross-canvas session/isolation/writeback coverage plus live tool-event rendering during active agent responses.
- [x] P2: Align stream termination + provider fallback behavior with Claude Agent SDK event/env contract (`message_stop`, `ANTHROPIC_BASE_URL`).

## Deployment Readiness Track

- [ ] Remove/contain remaining runtime reliance on deprecated legacy agent config files (keep explicit compat routes only).
- [x] Ensure active node/session/model-config runtime stays provider-neutral and resolves defaults from configured runtime edges instead of hardcoded vendor fallbacks.
- [x] Remove synthetic provider/model fallback rows from active settings UI surfaces that should reflect only live runtime state.
- [ ] Validate zero mock-data fallbacks in active runtime UI paths (tasks/mail/assets/chat/session history).
- [ ] Rebuild/fix the active department tool inventory so the canonical tool set is real and loadable in production (current ToolRegistry still logs missing tool modules/classes).
  - `strategy_extraction` is now rebuilt and live in the canonical registry.
  - Remaining degraded inventory is mostly the deliberately gated/unconfigured surfaces such as `broker_tools`, plus any other legacy permissions that still point at non-production adapters.
- [ ] Audit remaining non-active MT5 simulated fallback helpers (`src/risk/integrations/mt5/*`) and either gate them behind explicit test-only config or remove them from production paths.
- [ ] Re-run backend/frontend focused regression tests and capture commands + outcomes in handoff progress.
- [x] Fix Risk physics UI regression so Linux/non-MT5 hosts render an honest host-unavailable state instead of a broken failure banner.
- [x] Remove or replace Trading canvas placeholders (`TRADING JOURNAL — coming soon —`, `RISK PHYSICS — routed from Risk —`) with production-grade live surfaces.
- [x] Harden `/api/paper-trading/active` and related paper-trading API imports so non-MT5 hosts degrade safely instead of returning a 500.

## Immediate Next Slice (Now)

1. Wire WF1 state projection into FlowForge and department surfaces from the coordinator path now that the canonical shared-assets tree exists:
   - active department
   - active stage
   - blocking error
   - halt/waiting state
   - latest artifact
2. Finish the remaining cross-canvas Chrome verification beyond the now-passing mail send/receive path: session isolation, attachment behavior, live tool-event rendering, and writeback projection checks.
3. Finish the active video-ingest runtime modernization before any paid OpenRouter test:
   - evolve the now-live model-budget-aware chunk sizing from static budgets to measured/runtime-aware budgets if needed
   - evaluate whether the next step after captions-first should be provider-native `video_url` or staying on `frames + captions` for v1
   - verify both single-video and playlist runs against the canonical WF1 tree
   - wire persisted `source/audio/` and `source/captions/` artifacts into Shared Assets browsing / agent retrieval surfaces
4. Explore and track workflow runtime readiness explicitly:
   - AlgoForge / video-ingest path
   - OpenRouter-backed video processing path
   - Kanban/workflow live projection
   - shared-assets artifact browsing / resource projection
5. Continue rebuilding the remaining active department tool inventory against the new manifest-first contract and Claude-style tool usage model.
6. Extend the new prompt contract into workflow/session compaction playbooks and long-running harness behavior.
7. Continue no-mock-data sweeps and deployment hardening.
8. Remove/contain remaining runtime reliance on deprecated legacy agent config files (compat-only surface).
9. Continue MT5 production hardening beyond this slice: verify UI surfaces that expose MT5 status/bridge setup and eliminate any remaining fake terminal/account states.
10. Continue provider-neutral cleanup in remaining legacy/non-active surfaces without disturbing the active department-based runtime.
11. Continue replacing old `copilot` / `analyst` / `quantcode` labels and config dependencies in active runtime surfaces with canonical department-based identities, leaving compat routes only where required.
12. Continue cross-canvas UI cleanup after the Agents/settings stabilization:
    the next live UX targets remain the right-rail chat/session surfaces, mail viewport use, and canvas tiling density.
