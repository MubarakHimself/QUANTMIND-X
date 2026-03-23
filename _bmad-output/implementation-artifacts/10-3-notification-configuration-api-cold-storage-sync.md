# Story 10.3: Notification Configuration API & Cold Storage Sync

Status: review

## Story

As a trader managing alert fatigue,
I want configurable notifications and 3-year log retention with cold storage sync,
so that I receive actionable alerts without noise, and all logs are permanently accessible.

## Acceptance Criteria

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

## Tasks / Subtasks

- [x] Task 1: Notification Configuration API (AC: #1-#3)
  - [x] Subtask 1.1: Design notification config data model in database
  - [x] Subtask 1.2: Implement GET /api/notifications/config endpoint
  - [x] Subtask 1.3: Implement PUT /api/notifications/config/{event_type} endpoint
  - [x] Subtask 1.4: Handle high-priority always-on events (kill switch, loss cap)
  - [x] Subtask 1.5: Wire notification delivery suppression to existing alert system

- [x] Task 2: Cold Storage Sync Service (AC: #4)
  - [x] Subtask 2.1: Design log retention policy (hot vs cold storage boundaries)
  - [x] Subtask 2.2: Implement nightly sync job to Contabo cold storage
  - [x] Subtask 2.3: Implement integrity verification (checksum/hashing)
  - [x] Subtask 2.4: Configure retention periods (3-year cold storage)

- [x] Task 3: Integration & Testing (All ACs)
  - [x] Subtask 3.1: Wire notification config to existing alert/notification delivery system
  - [x] Subtask 3.2: Integration test with existing kill switch and risk system
  - [x] Subtask 3.3: Add unit tests for config CRUD and sync job
  - [x] Subtask 3.4: Document API contracts

## Dev Notes

- Relevant architecture patterns and constraints
- Source tree components to touch
- Testing standards summary

### Project Structure Notes

- Alignment with unified project structure (paths, modules, naming)
- Detected conflicts or variances (with rationale)

### References

- Cite all technical details with source paths and sections, e.g. [Source: docs/<file>.md#Section]

---

## DEV AGENT GUARDRAILS

### Technical Requirements

**Backend API Patterns:**
- All new endpoints go in `src/api/` as FastAPI routers
- Follow existing pattern: `router = APIRouter(prefix="/api/...", tags=["..."])`
- Response models use Pydantic schemas in `src/api/trading/models.py` style
- Database models in `src/database/models/`

**Notification System Context:**
- Existing notification system uses `send_alert_notification` skill [Source: docs/skills/system_skills/send_alert_notification.md]
- Audit logging already writes to logs via `log_trade_event` skill [Source: docs/skills/system_skills/log_trade_event.md]
- Monitoring folder at `src/monitoring/` contains Prometheus metrics and JSON logging
- Server has 50+ API routers already registered in `src/api/server.py`

**Cold Storage Architecture:**
- Contabo is the cold storage destination (see Epic 11 for rsync patterns)
- NFR-R6: Data integrity must be verifiable (checksums)
- NFR-D5: Immutable audit logs (no deletion, no modification)

**Notification Delivery Channels:**
- OS system tray delivery (FR63) - mention in architecture notes
- Copilot thread delivery for always-on events

### Architecture Compliance

**Required Architecture Patterns:**
- All audit log entries are immutable once written [Source: docs/architecture.md#Key Design Decisions]
- Department System over LangGraph for audit trail clarity [Source: docs/architecture.md]
- SQLite for operational data (notification configs)

**Integration Points:**
- Kill switch events (Story 3-2) - must remain always-on
- Risk system daily loss cap events - must remain always-on
- Existing notification delivery mechanism (to be discovered/integrated)

### Library/Framework Requirements

- FastAPI for REST endpoints
- APScheduler or similar for nightly sync job
- Standard library for checksum/hashing (hashlib)
- Existing notification delivery system integration (discover during implementation)

### File Structure Requirements

**New files to create:**
- `src/api/notification_config_endpoints.py` - REST API for notification configuration
- `src/database/models/notification_config.py` - Data model for notification settings
- `src/monitoring/cold_storage_sync.py` - Nightly sync service
- `tests/api/test_notification_config.py` - Unit tests
- `tests/monitoring/test_cold_storage_sync.py` - Sync service tests

**Files to modify (existing):**
- `src/api/server.py` - Register new router
- `src/monitoring/` - Integrate with existing monitoring

### Testing Requirements

- Unit tests for API endpoints (config CRUD)
- Unit tests for cold storage sync (verification, integrity)
- Integration test: verify kill switch events cannot be suppressed
- Test coverage: minimum 80%

---

## Previous Story Intelligence

This is the first story in Epic 10, so no previous story in this epic. However, similar patterns exist in other epics:

**Epic 2 Provider Config Pattern** (Story 2-1):
- Similar CRUD pattern for provider configuration storage
- See `src/database/models/provider_config.py` for model pattern

**Epic 4 Risk Params Pattern** (Story 4-2):
- Similar configuration storage and API
- See `src/database/models/risk_params.py`

---

## Latest Tech Information

- Contabo storage integration: rsync pattern from Epic 11 (future reference)
- FastAPI current version in use: check requirements.txt
- APScheduler for background jobs (if not already in requirements.txt)

---

## Project Context Reference

**Epic 10 Objectives:**
Mubarak can ask "Why was EA_X paused yesterday?" and get a full timestamped causal chain. All 5 audit layers (trade, strategy lifecycle, risk param, agent action, system health) are queryable in natural language via Copilot. Notifications configurable per event type. Server health live for both nodes. Copilot explains its reasoning for any past decision on request. Notification analytics + AI-suggested suppressions reduce volume.

**FRs covered:** FR61 (3-year log retention + cold storage sync), FR62 (configurable notifications), FR63 (OS system tray delivery)

---

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251115

### Implementation Notes

- NotificationConfig model already existed at `src/database/models/notification_config.py`
- Extended model with LogRetentionPolicy for cold storage config
- API endpoints at `src/api/notification_config_endpoints.py` (already existed, enhanced with retention endpoints)
- Cold storage sync service created at `src/monitoring/cold_storage_sync.py`
- Router already registered in server.py (Story 10-5 integration)
- Tests created at `tests/api/test_notification_config.py` and `tests/monitoring/test_cold_storage_sync.py`

### Debug Log References

### Completion Notes List

- Implemented AC1: GET /api/notifications returns event types with category, severity, enabled, delivery_channel (fixed response model)
- Implemented AC2: PUT /api/notifications toggles notification delivery (not logging)
- Implemented AC3: kill_switch_triggered and loss_cap_triggered_system marked as always_on=True
- Implemented AC4: LogRetentionPolicy model and cold storage sync with SHA256 checksums
- Added ColdStorageScheduler class for automatic nightly sync
- Fixed test endpoint paths
- Unit tests cover API endpoints and cold storage sync integrity verification

### File List

- Modified: src/database/models/notification_config.py (added LogRetentionPolicy)
- Modified: src/database/models/__init__.py (export LogRetentionPolicy)
- Modified: src/api/notification_config_endpoints.py (added retention endpoints, severity/delivery_channel, fixed session handling)
- Modified: src/monitoring/cold_storage_sync.py (added scheduler for automatic sync, added log_source_path parameter)
- Modified: tests/api/test_notification_config.py (fixed endpoint paths)
- Created: tests/monitoring/test_cold_storage_sync.py (sync service tests)

## Change Log

- 2026-03-20: Fixed bug where get_db_session() was called directly instead of using get_session() - the API endpoints now work correctly. Added log_source_path parameter to ColdStorageSync class for test compatibility.