# Story 5.2: Session Checkpoint → Graph Memory Commit Flow

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer building persistent agent memory,
I want the session checkpoint service wired to the graph memory commit flow,
so that agent work persists across sessions and is not lost on disconnect.

## Acceptance Criteria

**Given** an agent session is active,
**When** a checkpoint fires (configurable interval or on agent milestone),
**Then** all `session_status='draft'` nodes for the session are evaluated by ReflectionExecutor,
**And** nodes passing quality threshold are promoted to `session_status='committed'`,
**And** stale draft nodes are archived or discarded.

**Given** a session ends unexpectedly (crash, disconnect),
**When** the agent restarts,
**Then** it loads all committed nodes from the last checkpoint,
**And** resumes work from the last known committed state.

**Notes:**
- FR14: persistent agent memory across sessions

## Tasks / Subtasks

- [x] Task 1 (AC #1: Checkpoint triggers ReflectionExecutor)
  - [x] Subtask 1.1: Wire SessionCheckpointService to call ReflectionExecutor.trigger_reflection() on checkpoint
  - [x] Subtask 1.2: Add configurable checkpoint interval (default: 5 minutes)
  - [x] Subtask 1.3: Add milestone-based checkpoint trigger (after significant agent actions)
- [x] Task 2 (AC #2: Resume from committed state)
  - [x] Subtask 2.1: Implement session recovery logic on agent restart
  - [x] Subtask 2.2: Add load_committed_state(session_id) method to GraphMemoryFacade
  - [x] Subtask 2.3: Test crash recovery scenarios

## Dev Notes

### Previous Story Context (Story 5.1)

Story 5.1 implemented the foundation:
- **ReflectionExecutor** created at `src/memory/graph/reflection_executor.py`
- **Embedding service** created at `src/memory/graph/embedding_service.py`
- **session_status** column added ('draft' | 'committed')
- **OPINION node** pattern implemented with SUPPORTED_BY edges
- **trigger_reflection()** method already wired in SessionCheckpointService (line 457)

**Key insight from Story 5.1:** The ReflectionExecutor is already wired to SessionCheckpointService.trigger_reflection() — this story completes the wiring and adds session recovery.

### Architecture Prerequisites (CRITICAL)

- **MUST complete after Story 5.1** — graph memory foundation must exist
- **SessionCheckpointService** exists at `src/agents/memory/session_checkpoint_service.py`
- **ReflectionExecutor** exists at `src/memory/graph/reflection_executor.py`
- **GraphMemoryFacade** exists at `src/memory/graph/facade.py`

### File Structure

```
src/memory/graph/
├── __init__.py           [exists]
├── types.py              [exists] - SessionStatus, 14 node types
├── store.py             [exists] - session_status, embedding columns
├── operations.py        [exists] - commit methods
├── facade.py            [exists] - needs load_committed_state()
├── reflection_executor.py [exists] - needs session recovery methods
├── embedding_service.py [exists]
└── tier_manager.py      [exists]

src/agents/memory/
└── session_checkpoint_service.py  [exists] - trigger_reflection() wired
```

### Session Checkpoint Service Interface

```python
# Existing method (from Story 5.1)
async def trigger_reflection(
    session_id: str,
    graph_db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Trigger ReflectionExecutor to commit session memories."""
```

### What to Implement

1. **Checkpoint → ReflectionExecutor wiring:**
   - Ensure checkpoint events call `trigger_reflection()`
   - Add configurable checkpoint interval (read from config)
   - Add milestone-based triggers (after significant actions)

2. **Session Recovery:**
   - `load_committed_state(session_id)` - load all committed nodes for session
   - Resume from last committed state on restart
   - Handle edge cases: no committed state, partial commits

3. **Stale Draft Handling:**
   - Archive or discard draft nodes older than threshold
   - Configurable stale threshold (default: 24 hours)

### Configuration

```python
# Add to config (e.g., config.yaml or environment)
SESSION_CHECKPOINT_INTERVAL_MINUTES: int = 5  # default
SESSION_STALE_DRAFT_THRESHOLD_HOURS: int = 24
SESSION_CHECKPOINT_ON_MILESTONE: bool = True
```

### Testing Standards

- Tests go in `tests/memory/`
- Integration tests for checkpoint → reflection flow
- Session recovery tests (simulate crash/restart)
- Stale draft cleanup tests

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md` (§6.1 Graph Memory)
- Epic context: `_bmad-output/planning-artifacts/epics.md` (Epic 5, Story 5.2)
- Previous story: `_bmad-output/implementation-artifacts/5-1-graph-memory-completion-reflectionexecutor-opinion-nodes-embeddings.md`
- Session checkpoint: `src/agents/memory/session_checkpoint_service.py`

## Dev Agent Record

### Agent Model Used

claude-opus-4-6 / claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- **Task 1.1**: Verified trigger_reflection() was already wired in SessionCheckpointService (line 457) from Story 5.1
- **Task 1.2**: Added configurable checkpoint interval with environment variable support (SESSION_CHECKPOINT_INTERVAL_MINUTES, default: 5 min) and SESSION_STALE_DRAFT_THRESHOLD_HOURS (default: 24 hours)
- **Task 1.3**: Added milestone-based checkpoint triggers via checkpoint_on_agent_milestone() method with configurable enable/disable (SESSION_CHECKPOINT_ON_MILESTONE)
- **Task 2.1**: Added session recovery via ReflectionExecutor.recover_session() method that loads all committed nodes
- **Task 2.2**: Added GraphMemoryFacade.load_committed_state() method for loading committed nodes on restart
- **Task 2.3**: Created comprehensive test suite covering checkpoint flow and session recovery scenarios

**Implementation Summary:**
- SessionCheckpointService now supports: configurable interval, milestone triggers, auto-checkpoint, stale draft cleanup
- ReflectionExecutor now supports: recover_session(), cleanup_stale_drafts()
- GraphMemoryFacade now supports: load_committed_state(session_id)
- All 15 new tests pass

### File List

**Backend Files to Modify:**
- `src/memory/graph/facade.py` — Add load_committed_state() method
- `src/memory/graph/reflection_executor.py` — Add session recovery methods
- `src/agents/memory/session_checkpoint_service.py` — Add checkpoint config, milestone triggers

**Backend Files to Create:**
- `tests/memory/test_session_checkpoint_flow.py` — [NEW]
- `tests/memory/test_session_recovery.py` — [NEW]
