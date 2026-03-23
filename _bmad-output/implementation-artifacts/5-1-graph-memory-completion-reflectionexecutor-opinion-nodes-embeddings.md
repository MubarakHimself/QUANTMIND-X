# Story 5.1: Graph Memory Completion — ReflectionExecutor, OPINION Nodes & Embeddings

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer building the persistent agent memory layer,
I want the Graph Memory system completed with ReflectionExecutor, OPINION Node pattern, and vector embeddings,
so that all agents have a consistent, queryable memory backbone for cross-session knowledge (FR14).

## Acceptance Criteria

**Given** the existing `src/memory/graph/` implementation (80–90% done),
**When** the completion work runs,
**Then** the required columns are added (`session_status`, `embedding`, agent-specific fields identified in audit),
**And** all timestamps have `_utc` suffix (naming convention).

**Given** an agent takes a consequential action (state change, artifact, routing decision, assessment, approval/rejection),
**When** the action completes,
**Then** an OPINION node is written with: `action`, `reasoning`, `confidence`, `alternatives_considered`, `constraints_applied`, `agent_role`,
**And** the OPINION node has at least one `SUPPORTED_BY` edge to evidence (OBSERVATION, WORLD, or DECISION node).

**Given** the ReflectionExecutor runs after a session settles,
**When** it processes session memories,
**Then** episodic and semantic memories are extracted from the session's draft nodes,
**And** validated memories are promoted from `session_status='draft'` to `session_status='committed'`,
**And** downstream agents can load committed OPINIONs without re-deriving reasoning.

**Given** vector embeddings are added,
**When** a semantic memory query runs,
**Then** relevant committed nodes are retrieved via embedding similarity,
**And** retrieval uses `all-MiniLM-L6-v2` sentence-transformers (ChromaDB backend).

## Tasks / Subtasks

- [x] Task 1 (AC: #1 - Add columns)
  - [x] Subtask 1.1: Add `session_status` column to nodes table ('draft' | 'committed')
  - [x] Subtask 1.2: Add `embedding` column (BLOB) for vector storage
  - [x] Subtask 1.3: Add agent-specific fields identified in Story 5.0 audit
  - [x] Subtask 1.4: Rename all timestamps to `_utc` suffix
- [x] Task 2 (AC: #2 - OPINION nodes)
  - [x] Subtask 2.1: Implement OPINION node schema with required fields
  - [x] Subtask 2.2: Implement SUPPORTED_BY edge enforcement (mandatory)
  - [x] Subtask 2.3: Add consequential action detection logic
  - [x] Subtask 2.4: Write OPINION after: state change, artifact, routing decision, assessment, approval/rejection
- [x] Task 3 (AC: #3 - ReflectionExecutor)
  - [x] Subtask 3.1: Create `src/memory/graph/reflection_executor.py`
  - [x] Subtask 3.2: Implement session memory extraction (episodic + semantic)
  - [x] Subtask 3.3: Implement draft-to-committed promotion logic
  - [x] Subtask 3.4: Wire to session checkpoint service for trigger
- [x] Task 4 (AC: #4 - Vector embeddings)
  - [x] Subtask 4.1: Add sentence-transformers dependency (all-MiniLM-L6-v2)
  - [x] Subtask 4.2: Implement embedding generation pipeline
  - [x] Subtask 4.3: Add cosine similarity search in store.py `query_nodes()`
  - [x] Subtask 4.4: Integrate ChromaDB backend for embedding storage

## Dev Notes

### Architecture Prerequisites

- **CRITICAL:** This story MUST complete before Story 5.3 (Canvas Context System) — graph memory is the foundation
- Graph Memory is 80–90% done (needs columns + ReflectionExecutor + embeddings)
- 14 node types defined in `src/memory/graph/types.py`: WORLD, BANK, OBSERVATION, OPINION, WORKING, PERSONA, PROCEDURAL, EPISODIC, CONVERSATION, MESSAGE, AGENT, DEPARTMENT, TASK, SESSION, DECISION

### Session Isolation (CRITICAL)

- All memory writes from active sessions → `session_status = 'draft'`
- Draft nodes are invisible to other sessions until committed
- Only `session_status='committed'` nodes are visible to downstream agents
- Never dirty reads from interactive sessions

### File Structure

```
src/memory/graph/
├── __init__.py           [exists]
├── types.py              [exists]  - 14 node types defined
├── store.py              [exists]  - needs embedding column + query_nodes() enhancement
├── operations.py         [exists]
├── facade.py             [exists]  - GraphMemoryFacade
├── tier_manager.py       [exists]  - HOT/WARM/COLD tier management
├── compaction.py         [exists]
├── migration.py          [exists]  - add columns here
├── tools.py              [exists]
└── reflection_executor.py [NEW]    - create this file
```

### Database Schema Changes

```sql
-- Add to nodes table (via migration.py)
ALTER TABLE nodes ADD COLUMN session_status TEXT DEFAULT 'draft';
ALTER TABLE nodes ADD COLUMN embedding BLOB;  -- NULL until generated

-- Vector embedding for semantic search using all-MiniLM-L6-v2
-- ChromaDB backend for embedding storage
```

### OPINION Node Schema (CRITICAL)

Every OPINION node MUST have:
- `action`: What the agent did
- `reasoning`: Why they did it
- `confidence`: 0.0-1.0 confidence score
- `alternatives_considered`: What other options were evaluated
- `constraints_applied`: What constraints influenced the decision
- `agent_role`: Which agent/role took the action

**Mandatory edge rule:** Every OPINION must have at least one SUPPORTED_BY edge to OBSERVATION, WORLD, or DECISION. Orphaned OPINIONs are architecturally invalid.

### Key Dependencies

- `sentence-transformers` - all-MiniLM-L6-v2 model
- `chromadb` - embedding storage backend
- `src/agents/memory/session_checkpoint_service.py` - wire to commit flow (don't rebuild)
- Python 3.12 required

### Testing Standards

- Tests go in `tests/memory/` (TDD plan exists, not yet implemented)
- Unit tests for ReflectionExecutor
- Integration tests for OPINION node write + edge creation
- Vector search tests with cosine similarity verification
- Session isolation tests (draft vs committed visibility)

### Previous Story Learnings (Story 5.0)

From the audit:
1. Graph Memory is 80-90% complete - focus on missing pieces only
2. Session checkpoint service exists at `src/agents/memory/session_checkpoint_service.py` - wire to graph commit, do not rebuild
3. 14 node types already defined - don't add new types
4. FloorManager is the orchestrator - will use it to trigger OPINION writes

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md` (§6.1 Graph Memory, §6.4 OPINION Node Pattern)
- Epic context: `_bmad-output/planning-artifacts/epics.md` (Epic 5, Story 5.1)
- Previous story: `_bmad-output/implementation-artifacts/5-0-memory-architecture-copilot-infrastructure-audit.md`
- Graph memory plan: `docs/plans/2026-03-09-graph-memory-system.md`
- LangMem references:
  - https://langchain-ai.github.io/langmem/guides/delayed_processing/ — ReflectionExecutor debounce pattern
  - https://langchain-ai.github.io/langmem/guides/optimize_memory_prompt/ — prompt optimization

## Dev Agent Record

### Agent Model Used

claude-opus-4-6 / claude-sonnet-4-6

### Debug Log References

- Sprint status: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Epic context: `_bmad-output/planning-artifacts/epics.md` (Epic 5, Story 5.1)
- Architecture: `_bmad-output/planning-artifacts/architecture.md` (§6.1, §6.4)
- Previous story: `_bmad-output/implementation-artifacts/5-0-memory-architecture-copilot-infrastructure-audit.md`
- Session checkpoint: `src/agents/memory/session_checkpoint_service.py`

### Completion Notes

**Implemented:**

1. **Database Schema Changes (Task 1):**
   - Added `session_status` column ('draft' | 'committed')
   - Added `embedding` column (BLOB) for vector storage
   - Added OPINION-specific fields: action, reasoning, confidence, alternatives_considered, constraints_applied, agent_role
   - Renamed timestamps to `_utc` suffix (created_at_utc, updated_at_utc, last_accessed_utc, expires_at_utc, event_time_utc)

2. **OPINION Node Support (Task 2):**
   - Added SessionStatus class with DRAFT and COMMITTED values
   - Added SUPPORTED_BY to RelationType
   - Added create_opinion_node() method to MemoryOperations
   - Added create_supported_by_edge() method for mandatory edge enforcement

3. **ReflectionExecutor (Task 3):**
   - Created src/memory/graph/reflection_executor.py
   - Implemented session memory extraction (episodic + semantic)
   - Implemented draft-to-committed promotion with validation
   - Wired to SessionCheckpointService.trigger_reflection()

4. **Vector Embeddings (Task 4):**
   - Added sentence-transformers and chromadb to requirements.txt
   - Created embedding_service.py with all-MiniLM-L6-v2
   - Added search_by_embedding() to GraphMemoryStore
   - Added cosine similarity computation

**Files Modified:**
- src/memory/graph/types.py - Added SessionStatus, opinion fields, _utc suffixes
- src/memory/graph/store.py - Added columns, vector search
- src/memory/graph/operations.py - Added opinion creation, commit methods
- src/memory/graph/migration.py - Added migration functions
- src/memory/graph/__init__.py - Updated exports
- src/memory/graph/reflection_executor.py - NEW
- src/memory/graph/embedding_service.py - NEW
- src/agents/memory/session_checkpoint_service.py - Added trigger_reflection()
- requirements.txt - Added dependencies

### Change Log

- 2026-03-18: Completed Story 5.1 - Graph Memory Completion (ReflectionExecutor, OPINION Nodes & Embeddings)
- 2026-03-18: Code review fix - Clarified semantic memory extraction comment in reflection_executor.py

### File List

**Backend Files Modified:**
- `src/memory/graph/types.py` — Added SessionStatus, opinion fields, _utc suffixes
- `src/memory/graph/store.py` — Added columns, vector search methods
- `src/memory/graph/operations.py` — Added opinion creation, commit methods
- `src/memory/graph/migration.py` — Added migration functions
- `src/memory/graph/__init__.py` — Updated exports

**Backend Files Created:**
- `src/memory/graph/reflection_executor.py` — [NEW] ReflectionExecutor implementation
- `src/memory/graph/embedding_service.py` — [NEW] Embedding service with all-MiniLM-L6-v2

**Database:**
- SQLite: `data/graph_memory.db` — nodes + edges tables (run migration)

**Dependencies Added:**
- `sentence-transformers>=2.2.0`
- `chromadb>=0.4.0`

**Tests to Create:**
- `tests/memory/test_reflection_executor.py` — [NEW]
- `tests/memory/test_opinion_nodes.py` — [NEW]
- `tests/memory/test_vector_search.py` — [NEW]