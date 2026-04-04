# Production Deployment & Data Path Readiness (2026-04-02)

## Context
- `docker-compose.production.yml` wires the API, Redis, knowledge services, and monitoring, but the runtime still defaults `INTERNAL_API_BASE_URL` to `http://127.0.0.1:8000`, which makes most containers call themselves instead of the API host once deployed in the QuantMindX network.
- The `.env.example` sheet still advertises `localhost` for the API, Redis, and knowledge service URLs even though the production stack runs all of those services behind container hostnames like `quantmind-api` and `redis`.
- Graph memory migrations, the tiered storage router, and a handful of API endpoints keep resolving `data/…` paths directly inside the working directory, so overriding storage locations via environment variables (e.g., `/app/state/graph_memory.db` or `/data/cold_storage`) can leave parent directories missing and block startup.
- The audit must stay focused on the production/Contabo stack while keeping these configuration knobs usable in local or custom deployments. MT5/statusband surfaces are excluded per request unless a clear bug crosses them.

## Goals
1. Ensure every container in `docker-compose.production.yml` speaks to `quantmind-api` (or another environment override) rather than `localhost` so production wiring actually reaches the API hosting service.
2. Update the `.env.example` to describe the production hostnames/paths that the Compose files expect while still allowing a local developer to override with `localhost` when running out of Docker.
3. Harden the backend startup path handling (graph memory migrations and tiered storage) so environment-driven storage locations are created before use, avoiding silent startup failures.

## Assumptions & Constraints
- Target is the production/Contabo deployment configuration; however, everyday developers still run `./data` + `localhost` in local mode so we keep all environment overrides intact.
- “No mock data” means we only edit configuration defaults and path management, not seeding fake rows or logs.
- The MT5/statusband surface remains untouched unless a bug directly touches one of the planned fixes.
- Tests will cover the touched modules (`server`, `tiered_storage`, Compose/environment settings) but no new integration harness is being built.

## Approach Options
1. **Minimal patch (not recommended):** leave existing defaults and rely on documentation to remind operators to override `INTERNAL_API_BASE_URL` and create directories manually. Risk: production stacks will continue to misroute internal calls and may fail when env-driven storage paths do not exist.
2. **Balanced fix (recommended):** update the Compose file to default `INTERNAL_API_BASE_URL` to `http://quantmind-api:8000`, refresh `.env.example` with production hostnames while retaining notes for local overrides, and strengthen the backend startup logic to resolve and create the directories for any env-provided storage paths before migration/queries.
3. **Deep restructure:** introduce a shared `StoragePaths` configuration service that centralizes every data path and automatically creates parent directories. This would take longer and touches a wider surface (all endpoints in `src/api` that reference `data/…`); it also risks introducing regressions outside the requested scope.

## Proposed Design (approach 2)

### 1. docker-compose.production.yml: internal URL wiring
- Change the `quantmind-api` environment block to default `INTERNAL_API_BASE_URL=${INTERNAL_API_BASE_URL:-http://quantmind-api:8000}` so that every service (pagindex, routers, scheduler tasks) gets a sensible production host without requiring every operator to override the variable.
- Keep the ability to override the variable from `.env` or the shell for non-Docker runs.

### 2. `.env.example`: production-ready defaults
- Replace the `localhost` defaults for the API, Redis, and knowledge services with their Compose hostnames (`quantmind-api`, `redis`, `pageindex-*`) and explain in comments when `localhost` should be used (local dev without Compose).
- Highlight the storage-related variables (`GRAPH_MEMORY_DB`, `WARM_DB_PATH`, `COLD_STORAGE_PATH`) and note that they must point at mounted volumes (e.g., `/app/data/...` in Docker) instead of relative paths when overriding them for production.
- Document that `INTERNAL_API_BASE_URL` should match the host that exposes `/api` in production, with `/app/state` as a typical machine key location.

### 3. Backend path hardening
- In `src/api/server.py`’s startup hook, resolve `GRAPH_MEMORY_DB` via `Path(graph_db).expanduser()` and run `mkdir(parents=True)` on the parent directory before calling `migrate_graph_memory_db`. This way, setting `GRAPH_MEMORY_DB=/app/state/graph_memory.db` no longer depends on a pre-created `data/` folder.
- Add similar protection to `TieredStorageRouter.__init__`: after reading `WARM_DB_PATH` and `COLD_STORAGE_PATH`, expand them, log the resolved absolute paths, and ensure their parent directories exist (`Path(self._warm_db_path).parent.mkdir(...)` and `Path(self._cold_storage_path).mkdir(...)`). This prevents the warm/cold tiers from failing when the volume mount only contains e.g. `/app/data` but the env path points elsewhere.

## Risks & Mitigations
- Changing the Compose default for `INTERNAL_API_BASE_URL` may unset the implicit `127.0.0.1` used by local developers. Mitigation: document in `.env.example` and README that the variable can be overridden with `http://127.0.0.1:8000` when running outside Compose.
- Creating directories for env-provided paths might hide cases where volumes are missing entirely. Mitigation: emit INFO/DEBUG logs with the resolved paths so operators see which directories are being created.
- Graph memory migrations run earlier in `startup_event`; adding path resolution must not break existing migrations. Mitigation: keep the `migrate_graph_memory_db` call and simply sanitize the path beforehand.

## Testing & Verification
- `pytest tests/database/test_engine_config.py tests/database/test_db_manager.py` (covers config/path resolution and engine initialization).
- `pytest tests/api/test_node_update_endpoints.py` (validates `get_node_url` still resolves override environment values even after the Compose/default change).
- Manual smoke check: run `docker-compose -f docker-compose.production.yml config` to ensure there are no syntax errors and the env defaults look correct (informally described in the implementation message).

## Next Step
- After you review and approve this spec file, I will invoke the `writing-plans` skill to turn the approved design into concrete implementation steps and then begin coding.
