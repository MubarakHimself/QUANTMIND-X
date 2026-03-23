# Story 2.0: Provider Infrastructure Audit

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer starting Epic 2,
I want a complete audit of the current provider and server connection state,
so that stories 2.1–2.6 build on verified existing code rather than assumptions.

## Acceptance Criteria

1. [AC-1] Given the `src/` backend, when the audit runs, then a findings document covers: (a) existing provider config files or classes, (b) any hardcoded API keys or model names in agent code, (c) current `ProvidersPanel.svelte` implementation state (partial implementation known), (d) existing server connection config (if any), (e) how Claude Agent SDK is currently initialised (if at all).

2. [AC-2] Audit is read-only — no code changes permitted

## Tasks / Subtasks

- [ ] Task 1: Scan `src/agents/` for provider configuration (AC: 1a)
  - [ ] Subtask 1.1: List all files that reference `provider`, `api_key`, `model`, `anthropic`, `openai`
  - [ ] Subtask 1.2: Document any hardcoded values found
- [ ] Task 2: Scan `src/services/` for provider infrastructure (AC: 1a)
  - [ ] Subtask 2.1: List provider-related service files
  - [ ] Subtask 2.2: Document initialization patterns
- [ ] Task 3: Analyze `ProvidersPanel.svelte` state (AC: 1c)
  - [ ] Subtask 3.1: Read full component file
  - [ ] Subtask 3.2: Document current features and gaps
- [ ] Task 4: Check server connection configuration (AC: 1d)
  - [ ] Subtask 4.1: Search for Cloudzy/Contabo config files
  - [ ] Subtask 4.2: Document connection patterns
- [ ] Task 5: Analyze Claude Agent SDK initialization (AC: 1e)
  - [ ] Subtask 5.1: Find all SDK initialization points
  - [ ] Subtask 5.2: Document initialization patterns and configuration
- [ ] Task 6: Compile findings document (AC: all)
  - [ ] Subtask 6.1: Create comprehensive audit report
  - [ ] Subtask 6.2: Include recommendations for 2.1-2.6

## Dev Notes

### Project Structure Notes

- **Source tree locations to audit:**
  - `src/agents/` — Department system and sub-agents
  - `src/services/` — Service layer
  - `quantmind-ide/src/lib/components/settings/ProvidersPanel.svelte` — UI component
  - `src/api/` — API endpoints (check for provider endpoints)
  - Check `.env` for provider-related env vars

### Key Architectural Context

- **Frontend**: SvelteKit with static adapter (no SSR), Tauri desktop shell
- **Backend**: FastAPI on Python 3.12, two-node deployment (Cloudzy/Contabo)
- **AI Providers needed**: Anthropic, GLM, MiniMax, OpenRouter, DeepSeek
- **Tier assignment**: FloorManager (Opus), Department Heads (Sonnet), Sub-agents (Haiku)
- **Current SDK**: Anthropic Agent SDK for agent runtime

### Critical Rules from Project Context

1. **Agent paradigm**: Use Department System (`src/agents/departments/`) — not deprecated LangGraph
2. **Provider rule**: Anthropic Agent SDK for ALL agent runtime; other providers (GLM, MiniMax, DeepSeek) via `base_url` swap for LLM API calls only
3. **Never hardcode API keys** — use environment variables
4. **Python imports**: Use `src.` prefix from project root

### Testing Standards Summary

- Python: pytest with `asyncio_mode = auto`
- Frontend: Vitest with @testing-library/svelte
- This is an audit story — no tests required (read-only)

### References

- Epic 2 overview: `docs/epics.md#Epic-2`
- Architecture §3.5 (Runtime Provider Swap): `docs/architecture.md#3.5`
- Project Context §7 (AI Model Usage): `docs/project-context.md#section-7`

---

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (via Claude Code)

### Debug Log References

N/A — Audit story

### Completion Notes List

- Findings document created with comprehensive provider infrastructure analysis

### File List

**Files to create:**
- `{implementation_artifacts}/2-0-provider-infrastructure-audit.md` — Main audit findings

**Files to scan (read-only):**
- `src/agents/departments/` — Department system files
- `src/services/` — Service layer
- `quantmind-ide/src/lib/components/settings/ProvidersPanel.svelte` — UI
- `src/api/` — API endpoints
- `.env.example` or similar — Env var templates
