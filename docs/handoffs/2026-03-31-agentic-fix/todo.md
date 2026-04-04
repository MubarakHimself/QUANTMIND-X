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
- [ ] P2: Full Chrome navigation verification across all canvases for session isolation, attachments, live update projections, and mail send/receive.
  - `Research -> send_mail -> Portfolio MAIL inbox` is now verified live.
  - Remaining Chrome verification work is broader cross-canvas session/isolation/writeback coverage.
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

1. Finish the remaining cross-canvas Chrome verification beyond the now-passing mail send/receive path: session isolation, attachment behavior, and live writeback projection checks.
2. Continue rebuilding the remaining active department tool inventory against the new manifest-first contract and Claude-style tool usage model.
3. Extend the new prompt contract into workflow/session compaction playbooks and long-running harness behavior.
4. Continue no-mock-data sweeps and deployment hardening.
5. Remove/contain remaining runtime reliance on deprecated legacy agent config files (compat-only surface).
6. Continue MT5 production hardening beyond this slice: verify UI surfaces that expose MT5 status/bridge setup and eliminate any remaining fake terminal/account states.
7. Continue provider-neutral cleanup in remaining legacy/non-active surfaces without disturbing the active department-based runtime.
8. Continue replacing old `copilot` / `analyst` / `quantcode` labels and config dependencies in active runtime surfaces with canonical department-based identities, leaving compat routes only where required.
9. Continue cross-canvas UI cleanup after the Agents/settings stabilization:
   the next live UX targets remain the right-rail chat/session surfaces, mail viewport use, and canvas tiling density.
