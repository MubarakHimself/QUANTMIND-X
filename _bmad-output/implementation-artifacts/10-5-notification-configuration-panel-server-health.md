# Story 10.5: Notification Configuration Panel & Server Health

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader managing alerts and monitoring infrastructure,
I want a Notification Settings panel and a Server Health panel,
so that I can manage which events notify me and monitor both Cloudzy and Contabo node health.

## Acceptance Criteria


1. **Notification Settings Panel (Settings → Notifications)**
   - Given I open Settings → Notifications,
   - When the panel loads,
   - Then all notifiable event types are listed by category with on/off toggles,
   - And always-on events (kill switch, loss cap, system critical) are shown greyed out with a lock icon.

2. **Server Health Panel (Settings → Server Health)**
   - Given I navigate to Settings → Server Health,
   - When the panel loads,
   - Then Cloudzy and Contabo nodes each show: CPU %, memory %, disk %, network latency, uptime, last heartbeat.

3. **Threshold Alerts**
   - Given a node metric crosses a threshold (CPU > 85%, disk > 90%),
   - When the threshold is breached,
   - Then the affected metric turns red `#ff3b3b`,
   - And Copilot notifies: "Contabo: disk usage at 91%. Action recommended."

## Tasks / Subtasks

- [x] Task 1: Backend - Notification Config API (AC: #1)
  - [x] Subtask 1.1: Create notification_config table in database
  - [x] Subtask 1.2: Add GET/PUT /api/settings/notifications endpoints
  - [x] Subtask 1.3: Add always-on event types (kill_switch, loss_cap, system_critical)
- [x] Task 2: Backend - Server Health Metrics API (AC: #2)
  - [x] Subtask 2.1: Add GET /api/server/health/metrics endpoint
  - [x] Subtask 2.2: Implement system metrics collection (psutil for CPU, memory, disk)
  - [x] Subtask 2.3: Add network latency measurement
  - [x] Subtask 2.4: Add uptime and heartbeat tracking
- [x] Task 3: Frontend - Notification Settings Panel (AC: #1)
  - [x] Subtask 3.1: Create NotificationSettingsPanel.svelte component
  - [x] Subtask 3.2: Add category-based event type listing
  - [x] Subtask 3.3: Implement toggle switches with disabled state for always-on events
  - [x] Subtask 3.4: Add to settings/index.ts and SettingsView.svelte
- [x] Task 4: Frontend - Server Health Panel (AC: #2)
  - [x] Subtask 4.1: Create ServerHealthPanel.svelte component
  - [x] Subtask 4.2: Display node metrics with visual indicators
  - [x] Subtask 4.3: Implement threshold alerting with red color (#ff3b3b)
  - [x] Subtask 4.4: Integrate with existing node-health.ts store
  - [x] Subtask 4.5: Add to settings/index.ts and SettingsView.svelte
- [x] Task 5: Integration - Copilot Notification (AC: #3)
  - [x] Subtask 5.1: Add threshold breach detection logic
  - [x] Subtask 5.2: Integrate with CopilotPanel for notification delivery

## Code Review Findings — 2026-03-21

### Issues Found and Fixed: 2 High, 1 Medium | Known Gap: 1 High (AC3)

**[HIGH - FIXED] `class:disabled` used wrong property name**
- `NotificationSettingsPanel.svelte:141` used `event.is_enabled` but the interface property is `event.enabled`
- Visual disabled state never applied. Fixed: changed to `class:disabled={!event.enabled}`

**[HIGH - FIXED] Toggle state never reflected in UI**
- `NotificationSettingsPanel.svelte:71` read `updated.enabled` but backend PUT returns `is_enabled`
- After toggling, the UI always showed stale state. Fixed: changed to `updated.is_enabled`

**[MEDIUM - FIXED] test_notification_config.py had hollow mocks**
- Tests mocked `get_db_session` but endpoint uses `get_session` — mocks did nothing
- Tests accepted `[200, 401, 403, 500]` as valid — untestable. Rewritten with proper `get_session` mocks and precise assertions.

**[HIGH - FIXED] AC3 (Copilot notification on threshold breach) — now implemented**
- Created `quantmind-ide/src/lib/stores/serverHealthAlerts.ts` — `serverHealthAlertEvent` writable store
- `ServerHealthPanel.svelte` — `checkThresholdBreaches()` fires on each poll; uses a `currentlyBreached` Set to avoid alert spam (only fires on transition into critical state; clears on recovery)
- `CopilotPanel.svelte` — `onMount` subscribes to `serverHealthAlertEvent`; injects `role:'system'` message: `[SERVER ALERT] Contabo: disk at 91.0%. Action recommended.`
- All 4 metrics covered: CPU (>85%), Memory (>90%), Disk (>90%), Latency (>500ms)

**[LOW] API endpoint path mismatch**
- Story spec: `/api/settings/notifications`, actual: `/api/notifications`

## Dev Notes

- Relevant architecture patterns and constraints
- Source tree components to touch
- Testing standards summary

### Project Structure Notes

- Alignment with unified project structure (paths, modules, naming)
- Detected conflicts or variances (with rationale)

### References

- Cite all technical details with source paths and sections, e.g. [Source: docs/<file>.md#Section]

## Dev Agent Record

### Agent Model Used
Claude Sonnet 4.6 (verification session 2026-03-21)

### Debug Log References

### Completion Notes List

- Task 1: Created NotificationConfig model with default event types across 5 categories (trade, strategy, risk, system, agent)
- Task 2: Created server health metrics endpoint with CPU, memory, disk, latency, uptime tracking
- Task 3: Created NotificationSettingsPanel with category grouping, toggles, and always-on lock icons
- Task 4: Created ServerHealthPanel with dual-column node display and threshold indicators
- Task 5: Implemented — `serverHealthAlerts.ts` store + `checkThresholdBreaches()` in ServerHealthPanel + `serverHealthAlertEvent` subscription in CopilotPanel

### File List

## Files Created/Modified

### Backend (Python)
- `src/database/models/notification_config.py` (NEW) - NotificationConfig model (was pre-existing in 10.3, this story added more fields)
- `src/database/models/__init__.py` (MODIFIED) - Added NotificationConfig export
- `src/api/notification_config_endpoints.py` (PRE-EXISTING) - Modified for this story (already existed from 10.3)
- `src/api/server_health_endpoints.py` (NEW) - Server health metrics endpoints
- `src/api/server.py` (MODIFIED) - Added router imports and registration

### Frontend (Svelte)
- `quantmind-ide/src/lib/components/settings/NotificationSettingsPanel.svelte` (NEW)
- `quantmind-ide/src/lib/components/settings/ServerHealthPanel.svelte` (NEW)
- `quantmind-ide/src/lib/components/settings/index.ts` (MODIFIED) - Added exports
- `quantmind-ide/src/lib/components/SettingsView.svelte` (MODIFIED) - Added tabs and panel rendering

### Frontend (Svelte) — Task 5 additions
- `quantmind-ide/src/lib/stores/serverHealthAlerts.ts` (NEW) - `ServerHealthAlert` type + `serverHealthAlertEvent` store
- `quantmind-ide/src/lib/components/settings/ServerHealthPanel.svelte` (MODIFIED) - `checkThresholdBreaches()` detection + store write
- `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` (MODIFIED) - `serverHealthAlertEvent` subscription → system message injection

### Testing
- `tests/api/test_notification_config.py` (REWRITTEN) - Proper `get_session` mocks, strict assertions, 10 tests
- `tests/api/test_server_health.py` (NEW) - Tests for server health metrics API (6 tests)

---

## Developer Context - COMPREHENSIVE IMPLEMENTATION GUIDE

### EPIC ANALYSIS: Epic 10 - Audit, Monitoring & Notifications

**Epic Objectives:**
- Mubarak can ask "Why was EA_X paused yesterday?" and get a full timestamped causal chain
- All 5 audit layers queryable in natural language via Copilot
- Notifications configurable per event type
- Server health live for both nodes
- Copilot explains its reasoning for any past decision

**FRs Covered:**
- FR62: Configurable notifications
- FR65: Server health monitoring (OS tray notification delivery)

**Business Context:**
- This story builds on Epic 10 Story 10.3 (Notification Configuration API - Cold Storage Sync)
- Server health feeds into StatusBand health indicators (Epic 1 Story 1.5)
- Journey 29: "The Decision Audit" - user asks about past system behavior

---

### STORY FOUNDATION

**User Story:**
As a trader managing alerts and monitoring infrastructure,
I want a Notification Settings panel and a Server Health panel,
so that I can manage which events notify me and monitor both Cloudzy and Contabo node health.

**Acceptance Criteria:**
1. Settings → Notifications: Event types listed by category with toggles; always-on events (kill switch, loss cap, system critical) greyed out with lock icon
2. Settings → Server Health: Cloudzy and Contabo nodes show CPU %, memory %, disk %, network latency, uptime, last heartbeat
3. Threshold breach (CPU > 85%, disk > 90%): metric turns red #ff3b3b, Copilot notifies

---

### TECHNICAL REQUIREMENTS

**Backend Components:**

1. **Notification Config API** (AC: #1)
   - File: Create `src/api/notification_config_endpoints.py`
   - Endpoints needed:
     - `GET /api/settings/notifications` - fetch all notification preferences
     - `PUT /api/settings/notifications` - update notification preferences
   - Database model: `NotificationConfig` in `src/database/models/`
   - Event categories: `trade`, `strategy`, `risk`, `system`, `agent`
   - Always-on events: `kill_switch`, `loss_cap`, `system_critical`

2. **Server Health Metrics API** (AC: #2)
   - File: Extend existing or create new in `src/api/server_health_endpoints.py`
   - Endpoint: `GET /api/server/health/metrics`
   - Response: `{ contabo: { cpu, memory, disk, latency_ms, uptime_seconds, last_heartbeat }, cloudzy: { ... } }`
   - Use `psutil` for system metrics collection
   - Implement in both Contabo (primary) and Cloudzy (trading node)

**Frontend Components:**

1. **NotificationSettingsPanel.svelte** (AC: #1)
   - Location: `quantmind-ide/src/lib/components/settings/NotificationSettingsPanel.svelte`
   - Uses Lucide icons (Bell, BellOff, Lock)
   - Toggle switches with disabled state for always-on events
   - Category headers: Trade Events, Strategy Events, Risk Events, System Events, Agent Events
   - Follow existing ServersPanel.svelte pattern

2. **ServerHealthPanel.svelte** (AC: #2)
   - Location: `quantmind-ide/src/lib/components/settings/ServerHealthPanel.svelte`
   - Display metrics per node: CPU, Memory, Disk, Latency, Uptime, Heartbeat
   - Visual threshold indicators: green/yellow/red based on thresholds
   - Red (#ff3b3b) for threshold breach
   - Integrate with existing `node-health.ts` store

3. **Settings Integration**
   - Add exports to `quantmind-ide/src/lib/components/settings/index.ts`
   - Add navigation entries to SettingsView.svelte

---

### ARCHITECTURE COMPLIANCE

**Frontend Architecture (from architecture.md):**
- Component files: PascalCase (`NotificationSettingsPanel.svelte`, `ServerHealthPanel.svelte`)
- Props: camelCase, use Svelte 5 `$props()` rune
- Derived state: use `$derived()`
- Stores: Global reactive state in `src/lib/stores/`
- Import prefix: use `apiFetch` from `src/lib/api/`

**Backend Architecture:**
- API endpoints in `src/api/`
- Models in `src/database/models/`
- Follow existing patterns in `src/api/provider_config_endpoints.py`

**Svelte 5 State - Runes Only:**
```svelte
<script lang="ts">
  let notifications = $state([]);
  let servers = $state([]);
  let isLoading = $state(false);
</script>
```

---

### LIBRARY FRAMEWORK REQUIREMENTS

**Required Packages:**
- `psutil` - system metrics collection (add to requirements.txt)
- `lucide-svelte` - icons (already in use)
- No new frontend packages expected

**Existing Store:**
- `quantmind-ide/src/lib/stores/node-health.ts` - extend for metrics

---

### FILE STRUCTURE REQUIREMENTS

**Backend Files to Create/Modify:**
```
src/
├── api/
│   ├── notification_config_endpoints.py  (NEW)
│   └── server_health_endpoints.py        (NEW or EXTEND)
└── database/
    └── models/
        └── notification_config.py         (NEW)
```

**Frontend Files to Create/Modify:**
```
quantmind-ide/src/lib/
├── components/
│   └── settings/
│       ├── NotificationSettingsPanel.svelte  (NEW)
│       ├── ServerHealthPanel.svelte          (NEW)
│       └── index.ts                           (MODIFY - add exports)
└── stores/
    └── node-health.ts                     (EXTEND - add metrics)
```

**Settings Navigation:**
- Add "Notifications" and "Server Health" to SettingsView.svelte sidebar

---

### TESTING REQUIREMENTS

**Backend Tests:**
- `tests/api/test_notification_config.py` - test GET/PUT endpoints
- `tests/api/test_server_health.py` - test metrics endpoint

**Frontend Tests:**
- Component tests for toggle behavior
- Test disabled state for always-on events

**Acceptance Criteria Testing:**
1. Open Settings → Notifications, verify event types by category with toggles
2. Verify always-on events are greyed with lock icon
3. Open Settings → Server Health, verify both nodes show all 6 metrics
4. Simulate threshold breach, verify red color (#ff3b3b)
5. Verify Copilot notification triggered on threshold breach

---

### PREVIOUS STORY INTELLIGENCE

**No previous Epic 10 stories exist.** This is the first story in Epic 10. The audit infrastructure (Story 10.0) is still in backlog.

---

### GIT INTELLIGENCE

Recent commits don't show Epic 10 work. Last Epic 10 related changes were Story 10.3:
- FR61: 3-year log retention + cold storage sync
- FR62: configurable notifications (BUILD THIS STORY)
- FR63: OS system tray delivery

---

### LATEST TECH INFORMATION

**No web research required** - this story uses existing technologies:
- psutil for system metrics (mature, stable)
- Existing notification patterns in codebase
- Svelte 5 (already in use)

---

### PROJECT CONTEXT REFERENCE

From project overview:
- Dual-node architecture: Contabo (agent/compute), Cloudzy (live trading)
- StatusBand already shows node health dots (Epic 1 Story 1.5)
- Kill Switch always in TopBar (two-step: armed → confirm modal)
- Frosted Terminal aesthetic: near-transparent fills + heavy backdrop-filter blur

---

### IMPLEMENTATION NOTES

1. **Notification Panel UX:**
   - Group events by category (Trade, Strategy, Risk, System, Agent)
   - Each event has: name, description, toggle switch
   - Always-on events: grey background, lock icon, toggle disabled

2. **Server Health Panel UX:**
   - Two columns: Contabo | Cloudzy
   - Each node card shows: CPU%, Memory%, Disk%, Latency, Uptime, Last Heartbeat
   - Metrics with threshold breach: background highlight red (#ff3b3b)
   - Auto-refresh every 10 seconds (use existing polling pattern)

3. **Copilot Integration:**
   - When threshold breached, call Copilot notification system
   - Message format: "{node}: {metric} at {value}%. Action recommended."

4. **Threshold Values (hardcoded for now):**
   - CPU: 85%
   - Memory: 90%
   - Disk: 90%
   - Latency: 500ms

---

### COMPLETION CHECKLIST

- [ ] Backend: notification_config table and API
- [ ] Backend: server_health metrics endpoint
- [ ] Frontend: NotificationSettingsPanel.svelte
- [ ] Frontend: ServerHealthPanel.svelte
- [ ] Settings navigation entries
- [ ] Integration with existing node-health store
- [ ] Threshold alerting with Copilot notification
- [ ] Tests passing
- [ ] Build succeeds

---

**Status: review**
**Created: 2026-03-20**
**Story Key: 10-5-notification-configuration-panel-server-health**

### Completion Notes

**Workflow Update:**
- Story status updated from "backlog" to "ready-for-dev"
- Ultimate context engine analysis completed - comprehensive developer guide created

**Implementation Status:**
This story was already fully implemented in the codebase. The story file documents the complete implementation including:
- NotificationSettingsPanel with category grouping and always-on badge
- ServerHealthPanel with threshold alerting and auto-refresh
- Backend APIs at /api/notifications and /api/server/health/metrics
- SettingsView integration with both panels wired in

**Known Gap:**
- Task 5 (Copilot notification on threshold breach) remains incomplete
- This requires additional integration work with CopilotPanel
