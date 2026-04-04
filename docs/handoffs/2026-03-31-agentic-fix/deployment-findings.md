# Deployment Findings

Source: deployment-readiness subagent review performed during this session.

## High Severity

- `src/api/server.py`
  - Intended `NODE_ROLE` isolation is not consistently enforced.
  - After conditional router logic, overlapping routers still appear to be included unconditionally.
  - Risk: `node_backend` and `node_trading` expose overlapping production surfaces.

- `.github/workflows/deploy-contabo.yml`
  - Job outputs likely are not passed correctly because values are echoed instead of written to `$GITHUB_OUTPUT`.
  - The workflow also appears to deploy through `systemctl` and `git reset --hard`, which does not match the Docker-oriented production manifests.

- `config/settings/settings.json`
  - Appears to contain real API keys / secrets committed in repo.
  - Immediate secret-exposure risk. Do not quote or redistribute the values.

- `src/database/encryption.py`
  - Encryption key stored at `~/.quantmind/machine.key`.
- `docker-compose.production.yml`
  - Does not persist that path.
  - Risk: encrypted provider/server credentials become undecryptable after container replacement or migration.

## Medium Severity

- `src/database/engine.py`
  - Not production-ready for Postgres despite the repo advertising `HOT_DB_URL`.
  - SQLite-specific engine settings appear to be applied unconditionally.

- Storage/state paths are inconsistent across:
  - `data/*`
  - `.quantmind/*`
  - DuckDB files
  - Prefect SQLite
  - graph memory paths
  - volume mounts only partially cover them

- Production environment posture looks incomplete:
  - `SECRET_KEY`
  - `API_BASE_URL`
  - Auth0-related settings expected by auth code

- `src/api/chat_endpoints.py`
  - Legacy workshop chat route was fixed this session to call `WorkshopCopilotService`.
  - Remaining work is broader endpoint consolidation, not the placeholder response itself.

- `config/agents/copilot.yaml` and `config/brokers.yaml`
  - Default broker path still points at a mock/demo broker while the production MT5 socket broker is disabled.

- `src/agents/departments/heads/portfolio_head.py`
  - Portfolio/account methods still return hardcoded demo-mode data.

- `src/router/workflow_orchestrator.py`
  - Workflow fallback still generates canned code/artifacts when the real code-generation path is unavailable.

- `src/router/workflow_orchestrator.py`
  - Internal approval callbacks still use hardcoded `http://localhost:8000` instead of a configurable internal service base URL.

## Recommended Next Deployment Track

1. Remove committed secrets and rotate any exposed credentials.
2. Make `NODE_ROLE` router registration truly exclusive and test all three roles.
3. Align deployment automation with the actual Docker production topology.
4. Persist encryption key material explicitly.
5. Normalize stateful storage locations and volume mounts.
6. Make database engine configuration backend-specific for SQLite vs Postgres.
7. Remove runtime fake/demo UI data and hardcoded localhost service routes from the user-facing Svelte app.
8. Disable or replace remaining placeholder backend responses before calling the system production-ready.
