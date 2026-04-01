# QuantMind IDE Backend URL Cleanup (2026-04-01)

## Context
- The quantmind-ide frontend still contains `http://localhost:8000` literals in some of the most sensitive surfaces (SharedAssetsView, KillSwitchView, EAManagerView, the department mail/chat stores, and the canvas/shell chat plumbing), which bypass the shared API configuration that handles Cloudzy/Contabo routing.
- Leaving those literals in place creates deployment risk whenever the UI is mounted in a different environment and makes it harder to use `API_CONFIG` overrides or the centralized `apiFetch`.

## Goals
1. Remove remaining high-risk hardcoded backend URLs from the flagged views/stores without changing their behavior.
2. Leverage the existing `API_CONFIG`, `getBaseUrl`, and `apiFetch` helpers so the UI automatically respects environment overrides.
3. Keep the streaming/delegation flow in the department chat store intact while still building its URLs through the shared helpers.

## Assumptions & Constraints
- Work stays within the `quantmind-ide` frontend. No backend code will be touched.
- Scope is the surfaces you listed; if you want additional fetch sites cleaned up later we can extend this.
- Tests will not be introduced from scratch, but existing unit/vitest suites should still pass once the refactor is applied.

## Approaches
1. **Manual template swap:** individually replace each `http://localhost:8000/...` in the target files with `API_CONFIG.API_BASE` or a computed base string. Pros: minimal new code. Cons: still leaves duplicated base-building logic and makes the streaming path in `departmentChatStore` harder to keep aligned.
2. **Helper-backed cleanup (recommended):** add a shared `buildApiUrl(endpoint)` helper in `src/lib/api.ts` (used by `apiFetch` already) and convert the targeted code paths to call `apiFetch` or `buildApiUrl` so every URL automatically honors the current environment. This reduces duplication, keeps SSE streaming code in sync, and preserves credentials/headers. Trade-off: one more helper to maintain but it keeps the rest of the work DRY.
3. **Service rearchitecture:** build a new API wrapper per surface that centralizes all fetches (e.g., separate SDK modules for assets, bots, chat). Pros: very explicit. Cons: out of scope for this cleanup and would take much longer.

## Proposed Design (approach 2)

### Shared API helpers
- Extend `src/lib/api.ts` with a `buildApiUrl(endpoint: string): string` helper that encapsulates the `/api`-prefix logic already inside `apiFetch`, then have `apiFetch` call that helper. Export the helper so streaming/legacy fetch paths can reuse the same URL construction.

### SharedAssetsView.svelte
- Import `apiFetch` from `$lib/api`.
- Replace `fetch('http://localhost:8000/api/assets/...')` calls with `apiFetch('/assets/...')` and handle errors via `try/catch` the same as today.
- This aligns SharedAssetsView with the existing API helpers without changing how the UI updates state.

### KillSwitchView.svelte
- Import `apiFetch`.
- `loadBotData` and `loadKillSwitchHistory` should call `apiFetch('/kill-switch/bots')` and `apiFetch('/kill-switch/history')` respectively so they inherit credentials/cors handling and avoid hardcoded hosts.

### EAManagerView.svelte
- Import `apiFetch`.
- Replace the GET/POST calls for `/ea/tags`, `/ea/bots`, `/ea/reviews`, and `/video-ingest/start` with `apiFetch`, preserving the existing `try/catch` logic and request bodies.
- This keeps the UI functioning identically while routing through `API_CONFIG`.

### departmentMailStore.ts
- Import `apiFetch` instead of defining `API_BASE`.
- Update `fetchDepartmentMail`, `markMessageRead`, `deleteMessage`, and any other write operations to call `apiFetch('/departments/mail/...')` (matching their current paths) so the message store honors the shared API configuration.

### departmentChatStore.ts
- Remove the hardcoded `API_BASE` constant and import the new `buildApiUrl`.
- Use `buildApiUrl('/chat/departments/...')` when hitting the streaming endpoint so the `fetch` call stays in sync with `apiFetch`'s URL logic.
- Likewise build the `/trading-floor/delegate` URL via `buildApiUrl`, keeping the existing SSE parsing and delegation logic intact.

## Risks & Mitigations
- The new helper must faithfully mirror the `/api`-prefix handling inside `apiFetch` to avoid breaking streaming code; we mitigate this by reusing the same logic (helper + single source of truth).
- No unit tests cover these particular modules today, so we'll rely on the UI smoke tests to verify nothing regresses.

## Testing & Verification
- No new tests are proposed; rely on existing vitest/browser smoke checks for the affected views and stores.
- Manual verification on `npm run dev` should confirm the endpoints still load under the normal API proxy once the refactor is applied.

## Next Step
- Once you approve this design, I will run the `writing-plans` skill to break the implementation into executable steps and then start coding.
