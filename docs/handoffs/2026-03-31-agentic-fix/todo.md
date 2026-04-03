# QuantMindX Agentic/UI TODO Tracker (Production, No Mock Data)

Last updated: 2026-04-03

## Active Correction Track (contract-first)

- [x] P0: Canonicalize active agent identities (`floor_manager`, department heads) and keep legacy aliases compatibility-only.
- [x] P0: Manifest-first chat-context normalization in chat endpoints (strip heavy payloads).
- [x] P1: Workspace Resource Contract backend (manifest/search/read) for natural resource discovery across canvases.
- [x] P1: Session contract normalization (`interactive_session`, `workflow_session`) at session creation/update boundaries.
- [x] P1: Standardized mutating tool return envelope with `ui_projection_event` (task/mail/workflow/resource updates).
- [x] P2: Frontend attachment/natural-search wiring to consume backend resource-contract endpoints.
- [ ] P2: Structured streaming status events parity across department and floor-manager chat paths.
- [ ] P2: Full Chrome navigation verification across all canvases for session isolation, attachments, and live update projections.

## Deployment Readiness Track

- [ ] Remove/contain remaining runtime reliance on deprecated legacy agent config files (keep explicit compat routes only).
- [ ] Ensure provider routing defaults to configured Minimax across all active heads and sub-agents.
- [ ] Validate zero mock-data fallbacks in active runtime UI paths (tasks/mail/assets/chat/session history).
- [ ] Re-run backend/frontend focused regression tests and capture commands + outcomes in handoff progress.

## Immediate Next Slice (Now)

1. Complete structured streaming/status event parity in floor-manager + department chat UI.
2. Finish cross-canvas end-to-end Chrome verification (mail send/receive, resource attach, kanban projection updates).
3. Fix remaining Settings modal close trap regression during navigation.
4. Continue no-mock-data sweeps and deployment hardening.
