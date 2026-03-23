# Story 7.5: Session Workspace Isolation for Concurrent Sessions

Status: done

## Story

As a developer building multi-session agent coordination,
I want session workspace isolation so multiple concurrent Alpha Forge or research sessions don't contaminate each other,
so that parallel work produces clean, independent outputs.

## Acceptance Criteria

1. **Given** two concurrent research sessions are running on different strategies,
   **When** both sessions write to graph memory,
   **Then** each write is tagged with its own `session_id + session_status='draft'`,
   **And** neither session reads the other's draft nodes.

2. **Given** a Department Head completes a session,
   **When** it commits its work,
   **Then** committed nodes become available to all subsequent sessions via `session_status='committed'`,
   **And** the commit log records: `{ session_id, committed_at_utc, node_count, department }`.

3. **Given** concurrent Alpha Forge sessions work on the same strategy,
   **When** both try to commit,
   **Then** the second commit waits for DeptHead review (like a merge review),
   **And** conflicts are flagged for FloorManager resolution.

## Tasks / Subtasks

- [x] Task 1: Session Workspace Isolation Infrastructure (AC: #1)
  - [x] Subtask 1.1: Extend graph store to support session_id tagging on nodes
  - [x] Subtask 1.2: Implement draft node filtering by session_id in queries
  - [x] Subtask 1.3: Add session_status field to graph node schema
- [x] Task 2: Session Commit Workflow (AC: #2)
  - [x] Subtask 2.1: Create session commit API/function in FloorManager
  - [x] Subtask 2.2: Implement commit log recording with session metadata
  - [x] Subtask 2.3: Add node status transition logic (draft → committed)
- [x] Task 3: Concurrent Conflict Resolution (AC: #3)
  - [x] Subtask 3.1: Detect concurrent writes to same strategy namespace
  - [x] Subtask 3.2: Implement DeptHead review queue for conflicts
  - [x] Subtask 3.3: Add FloorManager conflict resolution API
- [x] Task 4: Integration Tests & Validation
  - [x] Subtask 4.1: Test isolation between two concurrent sessions
  - [x] Subtask 4.2: Test commit workflow and visibility
  - [x] Subtask 4.3: Test conflict detection and resolution

## Dev Notes

### Architecture Patterns & Constraints

- **Session workspace isolation** prevents context poisoning between parallel sessions
- **Multiple concurrent sessions** with `session_id` namespace for isolation
- **DeptHead evaluates + commits** — session isolation is a FloorManager orchestration concern
- **Graph memory**: session_id tagging on nodes with session_status (draft/committed)
- **Redis Streams** for concurrent task dispatch (per Story 7.7 pattern)

### Source Tree Components to Touch

1. **Backend Python:**
   - `src/memory/graph/store.py` - EXTEND with session_id tagging and filtering
   - `src/memory/graph/types.py` - ADD session_status to node schema
   - `src/agents/departments/floor_manager.py` - ADD session commit orchestration
   - `src/api/session_endpoints.py` - NEW - Session management endpoints (if needed)
   - `src/memory/graph/facade.py` - EXTEND with session-aware query methods

2. **Testing:**
   - `tests/memory/graph/test_session_isolation.py` - NEW - Session isolation tests
   - `tests/agents/test_floor_manager.py` - EXTEND with commit workflow tests

### Testing Standards

- Python files under 500 lines (NFR-M3)
- Unit tests for session isolation (draft vs committed visibility)
- Integration tests for concurrent session scenarios
- Verify FloorManager orchestration patterns

### Project Structure Notes

- **ALWAYS extend existing graph store** - Do NOT replace existing functionality
- **session_id format**: UUID string (e.g., "sess-abc123")
- **session_status values**: 'draft' (private to session), 'committed' (visible to all)
- **Commit log**: Stored in graph memory with metadata

### Key Technical Decisions

1. **Isolation strategy**: Tag each node with session_id + session_status; filter queries by session
2. **Commit mechanism**: Update session_status from 'draft' to 'committed'; record in commit log
3. **Conflict detection**: Compare session_id + strategy_id namespace; flag if second commit detected
4. **Resolution**: Queue to DeptHead for manual review; FloorManager coordinates

### Previous Story Intelligence

**From Story 7-4 (Skill Catalogue):**
- Story 7-4 established the pattern for extending existing managers (SkillManager → floor_manager session handling)
- Key learnings: FloorManager as orchestrator for department workflows
- Pattern: Each story extends existing infrastructure, never replaces it
- This story follows same pattern: extend graph store with session isolation, not rewrite

**From Story 7-7 (Concurrent Task Routing):**
- Redis Streams pattern for concurrent task dispatch (reference for parallel session handling)
- Three-tier priority (HIGH/MEDIUM/LOW) with session_id namespacing - reuse this pattern
- Task status visible in Agent Panel - consider session status visibility

### References

- [Source: src/memory/graph/store.py] - Graph storage foundation
- [Source: src/memory/graph/types.py] - Node type definitions
- [Source: src/agents/departments/floor_manager.py] - Department orchestration
- [Source: epics.md#Story 7.5] - Original story requirements
- [Source: _bmad-output/implementation-artifacts/7-4-skill-catalogue-registry-authoring-skill-forge.md] - Previous story context
- [Source: docs/architecture.md] - System architecture for context

## Dev Agent Record

### Agent Model Used
- Claude Sonnet 4 (via FloorManager)

### Debug Log References
- N/A - All tests passing

### Completion Notes List
- Implemented session workspace isolation infrastructure in graph store and facade
- Added session_id and session_status fields to node schema (already existed but indexes added)
- Implemented draft node filtering by session_id in queries
- Created session commit workflow with commit log recording
- Added concurrent conflict detection for same strategy namespace
- Added FloorManager session commit orchestration methods
- All 10 unit tests passing

### File List
- src/memory/graph/store.py - Extended with session isolation methods, indexes
- src/memory/graph/facade.py - Extended with session-aware query and commit methods (entity_id support added)
- src/memory/graph/operations.py - Extended with entity_id parameter for retain()
- src/agents/departments/floor_manager.py - Added session commit orchestration methods (entity_id + conflict check)
- src/api/floor_manager_endpoints.py - Added POST /session/commit endpoint
- tests/memory/graph/test_session_isolation.py - NEW - Session isolation tests (10 tests)

## Code Review Findings (2026-03-20)

### Issues Fixed
1. **HIGH - Missing entity_id in retain()**: Added entity_id parameter to operations.retain() and facade.retain() to enable conflict detection
2. **HIGH - AC #3 conflict detection broken**: Updated facade.commit_session() and floor_manager.commit_session() to check for conflicts when entity_id provided, returning 'pending_review' status
3. **MEDIUM - Missing REST API endpoint**: Added POST /api/floor-manager/session/commit endpoint with SessionCommitRequest/Response models

### Verification
- All 10 unit tests passing
- Conflict detection now functional when entity_id passed to retain()
- Commit now checks conflicts and returns pending_review when detected
