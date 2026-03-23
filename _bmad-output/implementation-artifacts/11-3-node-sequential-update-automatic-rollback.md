# Story 11.3: 3-Node Sequential Update & Automatic Rollback

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system administrator updating QUANTMINDX,
I want a coordinated sequential update across all 3 nodes with health check and automatic rollback,
so that updates are applied safely (FR67).

## Acceptance Criteria

1. **Given** "update all nodes" is issued via Copilot,
   **When** the update sequence starts,
   **Then** nodes update in order: Contabo → Cloudzy → Desktop (Tauri),
   **And** each node health-checks before the next begins.

2. **Given** a node fails post-update health check,
   **When** the health check fails,
   **Then** the update sequence halts,
   **And** the failed node automatically rolls back to the previous version,
   **And** Mubarak is notified: "Cloudzy update failed — rolled back. Contabo on new version."

3. **Given** the update completes successfully on all nodes,
   **When** the final node completes,
   **Then** a completion summary is displayed in Copilot.

## Tasks / Subtasks

- [x] Task 1: Sequential update orchestrator (AC: #1)
  - [x] Subtask 1.1: Node order logic (Contabo → Cloudzy → Desktop)
  - [x] Subtask 1.2: Health check integration before each node transition
  - [x] Subtask 1.3: Update trigger via Copilot command
- [x] Task 2: Health check system (AC: #1, #2)
  - [x] Subtask 2.1: Per-node health endpoint implementation
  - [x] Subtask 2.2: Health response schema definition
  - [x] Subtask 2.3: Timeout handling (configurable per node type)
- [x] Task 3: Automatic rollback mechanism (AC: #2)
  - [x] Subtask 3.1: Version snapshot/rollback point creation pre-update
  - [x] Subtask 3.2: Rollback execution on health check failure
  - [x] Subtask 3.3: Notification service integration
- [x] Task 4: Deployment window enforcement (Notes)
  - [x] Subtask 4.1: Time window validation (Friday 22:00 – Sunday 22:00 UTC)
  - [x] Subtask 4.2: Block update attempts outside window with user feedback

## Dev Notes

### Architecture Patterns & Constraints

- **Sequential update order**: Contabo → Cloudzy → Desktop (Tauri) - MANDATORY, not parallel
- **Deploy window**: Friday 22:00 – Sunday 22:00 UTC only (never during open market hours)
- **Health check pattern**: Each node must pass health check before next node begins
- **Rollback scope**: Only failed node rolls back - previously updated nodes maintain new version unless explicitly rolled back
- **Notification**: Must integrate with existing notification system for failure alerts
- **Three deployment nodes**: Contabo (primary/compute), Cloudzy (live trading), Desktop (local dev/Tauri)
- **NODE_ROLE context**: Each node has distinct NODE_ROLE value for context-aware operations

### Technical Stack & Versions

**Python Packages Required:**
- `prefect>=2.14.0` - Workflow orchestration (same as Stories 11.1, 11.2)
- `httpx` - For node health API calls
- `gitpython>=3.1.0` - For version management and rollback

**Node Health Endpoint Pattern (reuse from Story 10-2):**
- Endpoint: `/api/health` on each node
- Response schema: `{ "status": "healthy|degraded|unhealthy", "version": "x.y.z", "uptime_seconds": int, "checks": {...} }`
- Timeout: 30 seconds per node (configurable)

**Existing Infrastructure to Reuse:**
- Story 1.3 NODE_ROLE split: `src/config.py` has NODE_ROLE detection
- Story 10-2 health endpoints: `src/api/server_health_endpoints.py`
- Story 11.1/11.2 Prefect flows: `flows/` directory for scheduling patterns

### Files to Create/Modify

**NEW FILES:**
- `flows/node_update_flow.py` — Prefect flow for sequential node updates
- `scripts/node-health-check.sh` — Shell script for health verification
- `src/api/node_update_endpoints.py` — API for update orchestration

**MODIFY:**
- `src/agents/departments/floor_manager.py` — Add "update all nodes" intent handler
- `src/api/server.py` — Register node update endpoints
- `src/config.py` — Add version tracking for rollback

### Technical Specifications

**Sequential Update Flow:**
```python
from prefect import flow, task
from datetime import datetime, timezone

NODE_UPDATE_ORDER = ["contabo", "cloudzy", "desktop"]
DEPLOY_WINDOW_START = 22  # Friday 22:00 UTC
DEPLOY_WINDOW_END = 22     # Sunday 22:00 UTC

@flow(log_prints=True)
def update_all_nodes_flow():
    """Sequential node update with health checks and rollback."""
    # Verify deploy window
    if not is_valid_deploy_window():
        raise ValueError("Update only allowed Friday 22:00 - Sunday 22:00 UTC")

    # Create rollback points
    for node in NODE_UPDATE_ORDER:
        create_rollback_snapshot(node)

    # Sequential update with health checks
    for node in NODE_UPDATE_ORDER:
        update_node(node)
        if not health_check(node):
            rollback_node(node)
            notify_failure(node)
            return {"status": "partial", "failed_node": node}

    notify_success()

@task(retries=1, retry_delay_seconds=60)
def health_check(node: str) -> bool:
    """Verify node health post-update."""
    import httpx
    response = httpx.get(f"https://{node}/api/health", timeout=30)
    return response.json().get("status") == "healthy"
```

**Deployment Window Validation:**
```python
def is_valid_deploy_window() -> bool:
    now = datetime.now(timezone.utc)
    # Friday = 4, Saturday = 5, Sunday = 6
    if now.weekday() < 4:  # Mon-Thu
        return False
    if now.weekday() == 4 and now.hour < DEPLOY_WINDOW_START:  # Before Friday 22:00
        return False
    if now.weekday() == 6 and now.hour >= DEPLOY_WINDOW_END:  # After Sunday 22:00
        return False
    return True
```

**Copilot Command Integration:**
- Intent: "update all nodes" or "deploy new version"
- Parser extracts version target from command
- FloorManager orchestrates the flow

### Testing Standards

- Unit test: Individual functions (health_check, is_valid_deploy_window, rollback)
- Integration test: Full flow execution (Prefect test runner)
- Failure simulation: Health check failure, rollback execution, notification delivery
- Deploy window test: Verify blocking outside allowed times
- Use patterns from `tests/flows/` and `tests/api/`

### Project Structure Notes

**Epic 11: System Management & Resilience**
- Previous stories: 11-1 (rsync), 11-2 (weekend compute) - both use similar Contabo scheduling patterns
- Integration with existing: NODE_ROLE system, Prefect flows, health endpoints
- Three deployment nodes: Contabo (primary), Cloudzy (trading), Desktop (Tauri)

**Deployment Context:**
- Contabo is primary deployment target for infrastructure changes
- Cloudzy handles live trading (must not disrupt during market hours)
- Desktop (Tauri) is local development - different update mechanism

### Previous Story Intelligence

**From Story 11.2 (Weekend Compute Protocol):**
- Pattern: Scheduled task on Contabo using Prefect
- Similar notification integration required (morning summary)
- Error handling: retry with exponential backoff, audit logging, notification
- Key learning: Use Prefect flow for orchestration, separate task for each operation

**From Story 11.1 (Nightly Rsync Cron):**
- Uses NODE_ROLE=contabo for Contabo-specific tasks
- Notification format: morning summary with failure reason
- Script location: `scripts/` for shell, `flows/` for Prefect

**Key Differences from Previous Stories:**
- This story spans multiple nodes (not Contabo-only)
- Requires inter-node communication for health checks
- Rollback mechanism is more complex (version snapshots)
- Deploy window enforcement is critical (market hours protection)

### Git Intelligence

Recent commits show:
- Story 11.2 completed: weekend-compute-protocol-scheduled-background-tasks
- Story 11.1 completed: nightly-rsync-cron
- Story 1.3 completed: node-role-backend-deployment-split (core NODE_ROLE infrastructure)
- Pattern: Shell scripts in `scripts/`, Prefect flows in `flows/`, API endpoints in `src/api/`

### References

- FR67: coordinated 3-node update with automatic rollback [Source: docs/architecture.md]
- Story 1.3: NODE_ROLE Backend Deployment Split [Source: _bmad-output/implementation-artifacts/1-3-node-role-backend-deployment-split.md]
- Story 10.2: Agent Reasoning Transparency Log API (health endpoint patterns)
- Story 11.1: Nightly Rsync Cron (Contabo cron context, notification patterns)
- Story 11.2: Weekend Compute Protocol (Prefect flow patterns)
- Story 11.4: ITT Rebuild Portability (backup/restore mechanisms for rollback)
- Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Epics: `_bmad-output/planning-artifacts/epics.md` (Story 11.3)

---

## Developer Implementation Guide

### What NOT to Do

1. **DO NOT** update nodes in parallel — sequential order is MANDATORY
2. **DO NOT** skip health checks between node updates
3. **DO NOT** run during market hours — deploy window enforcement is critical
4. **DO NOT** forget rollback point creation before updating
5. **DO NOT** update Cloudzy during trading hours — live trading protection
6. **DO NOT** use blocking update commands — use async patterns

### What TO Do

1. **DO** follow Contabo → Cloudzy → Desktop order strictly
2. **DO** verify health before proceeding to next node
3. **DO** create rollback snapshots before each node update
4. **DO** integrate with existing notification system
5. **DO** use Prefect flow for orchestration (like Stories 11.1, 11.2)
6. **DO** implement deploy window validation
7. **DO** log all operations to audit trail

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4 (claude-sonnet-4-20250514)

### Debug Log References

### Completion Notes List

- Implemented sequential node update flow following Contabo → Cloudzy → Desktop order
- Added deploy window validation (Friday 22:00 - Sunday 22:00 UTC only)
- Integrated health checks before each node transition
- Implemented automatic rollback mechanism on health check failure
- Added notification service integration for failure/success alerts
- Added Copilot command trigger for "update all nodes" and "deploy new version"
- Created unit tests for deploy window validation and node update patterns

### File List

**NEW FILES:**
- `flows/node_update_flow.py` — Prefect flow for sequential 3-node updates with health checks and automatic rollback
- `scripts/node-health-check.sh` — Shell script for health verification of nodes
- `src/api/node_update_endpoints.py` — API endpoints for node update orchestration
- `tests/flows/test_node_update_flow.py` — Unit tests for node update flow and deploy window

**MODIFIED FILES:**
- `src/api/server.py` — Added import and registration for node_update_router
- `src/intent/patterns.py` — Added NODE_UPDATE command intent and patterns
- `src/intent/classifier.py` — Added handler for NODE_UPDATE intent (updated to call API and return completion summary)
- `src/api/notification_config_endpoints.py` — Added send_notification function for notification integration

### Code Review Fixes (2026-03-21)

**Issues Fixed:**
1. Added `send_notification` function to `notification_config_endpoints.py` - resolves runtime import error
2. Updated classifier's `_execute_node_update` to call the API endpoint and return completion summary in Copilot (AC3)
3. Fixed test failures in `test_node_update_flow.py` - corrected mock paths and Prefect context handling