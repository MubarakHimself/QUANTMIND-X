# Story 10.2: Agent Reasoning Transparency Log & API

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader who wants to understand AI decisions,
I want an API that retrieves full reasoning chains for any past agent decision,
so that no agent action is a black box (FR78).

## Acceptance Criteria

1. **Given** `GET /api/audit/reasoning/{decision_id}` is called,
   **When** processed,
   **Then** it returns: `{ context_at_decision, model_used, prompt_summary, response_summary, action_taken, opinion_nodes[] }`,
   **And** opinion_nodes includes the OPINION node written by the agent (from Story 5.1).

2. **Given** I ask Copilot "Why did the Research department recommend GBPUSD short?",
   **When** FloorManager queries the reasoning log,
   **Then** it returns: hypothesis chain, evidence sources, confidence scores at each step, contributing sub-agents.

## Tasks / Subtasks

- [x] Task 1: Backend - Reasoning Log API (AC: #1)
  - [x] Subtask 1.1: Create `src/api/reasoning_log_endpoints.py`
  - [x] Subtask 1.2: Implement `GET /api/audit/reasoning/{decision_id}` endpoint
  - [x] Subtask 1.3: Query Memory Graph for OPINION nodes by decision_id/session
  - [x] Subtask 1.4: Build response with context_at_decision, model_used, prompt_summary, response_summary, action_taken, opinion_nodes[]
- [x] Task 2: Backend - Department Reasoning Query (AC: #2)
  - [x] Subtask 2.1: Add `GET /api/audit/reasoning/department/{department}` endpoint
  - [x] Subtask 2.2: Query OPINION nodes filtered by agent_role (department)
  - [x] Subtask 2.3: Return hypothesis chain, evidence sources, confidence scores
  - [x] Subtask 2.4: Aggregate contributing sub-agents from opinion_nodes
- [x] Task 3: Database Integration (AC: #1, #2)
  - [x] Subtask 3.1: Leverage existing Memory Graph store (Story 5.1)
  - [x] Subtask 3.2: Extend MemoryNode query to include decision_id lookup
  - [x] Subtask 3.3: Handle nodes with no OPINION (return empty array)
- [ ] Task 4: Frontend - Reasoning Explorer UI (Optional/Future)
  - [ ] Subtask 4.1: Create ReasoningExplorer.svelte component (future story)
  - [ ] Subtask 4.2: Integrate with CopilotPanel for NL queries (future story)

## Dev Notes

### Project Structure Notes

- Alignment with unified project structure (paths, modules, naming)
- Opinion nodes stored in memory graph (from Story 5.1: Graph Memory with ReflectionExecutor)
- Department agents (Research, Development, Risk, Trading, Portfolio) store OPINION nodes

### References

- [Source: src/memory/graph/types.py - MemoryNodeType.OPINION fields]
- [Source: src/memory/graph/store.py - query_nodes method]
- [Source: src/memory/graph/operations.py - Graph memory operations]
- [Source: Story 5.1 - Graph Memory with ReflectionExecutor]

## Dev Agent Record

### Agent Model Used

Claude Code (MiniMax-M2.5)

### Debug Log References

- src/api/reasoning_log_endpoints.py - Created new API endpoints
- src/api/server.py - Added reasoning_router import and registration
- tests/api/test_reasoning_log.py - Created test file with 8 tests

### Completion Notes List

- Created `src/api/reasoning_log_endpoints.py` with two endpoints:
  - `GET /api/audit/reasoning/{decision_id}` - Returns reasoning chain by decision ID
  - `GET /api/audit/reasoning/department/{department}` - Returns department reasoning with query params start_date, end_date, limit
- Integrated with existing Memory Graph via GraphMemoryFacade
- Added router registration in server.py
- Created comprehensive tests (8 tests, all passing)
- AC1: Returns context_at_decision, model_used, prompt_summary, response_summary, action_taken, opinion_nodes[]
- AC2: Returns hypothesis chain, evidence sources, confidence scores, contributing sub-agents

### File List

- src/api/reasoning_log_endpoints.py (NEW)
- src/api/server.py (MODIFIED - added import and router registration)
- tests/api/test_reasoning_log.py (NEW)

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
- FR78: Copilot reasoning chain explanation

**Business Context:**
- This story builds on Story 5.1 (Graph Memory with ReflectionExecutor/OPINION nodes)
- Opinion nodes contain: action, reasoning, confidence, alternatives_considered, constraints_applied, agent_role
- Journey 51: "The Transparent Reasoning" — full 8-of-31 strategy exposure chain

---

### STORY FOUNDATION

**User Story:**
As a trader who wants to understand AI decisions,
I want an API that retrieves full reasoning chains for any past agent decision,
so that no agent action is a black box (FR78).

**Acceptance Criteria:**
1. `GET /api/audit/reasoning/{decision_id}` returns full reasoning chain with OPINION nodes
2. Copilot asks "Why did Research department recommend GBPUSD short?" → returns hypothesis chain, evidence sources, confidence scores, contributing sub-agents

---

### TECHNICAL REQUIREMENTS

**Backend Components:**

1. **Reasoning Log API** (AC: #1)
   - File: Create `src/api/reasoning_log_endpoints.py`
   - Endpoint: `GET /api/audit/reasoning/{decision_id}`
   - Query Memory Graph for OPINION nodes matching decision_id
   - Response schema:
     ```python
     {
         "decision_id": str,
         "context_at_decision": dict,  # Agent's context at decision time
         "model_used": str,             # Model that generated the decision
         "prompt_summary": str,         # Summarized prompt/input
         "response_summary": str,       # Summarized response/output
         "action_taken": str,           # What action was taken
         "opinion_nodes": [             # Array of OPINION nodes
             {
                 "node_id": str,
                 "action": str,
                 "reasoning": str,
                 "confidence": float,
                 "alternatives_considered": str,
                 "constraints_applied": str,
                 "agent_role": str,
                 "created_at_utc": str
             }
         ]
     }
     ```

2. **Department Reasoning Query** (AC: #2)
   - File: Extend `src/api/reasoning_log_endpoints.py`
   - Endpoint: `GET /api/audit/reasoning/department/{department}`
   - Query Parameters: `start_date`, `end_date`, `limit`
   - Response schema:
     ```python
     {
         "department": str,
         "reasoning_chain": [
             {
                 "hypothesis": str,
                 "evidence_sources": [str],
                 "confidence_score": float,
                 "sub_agents": [str],
                 "decision_timestamp_utc": str
             }
         ],
         "total_decisions": int
     }
     ```

3. **Memory Graph Integration**
   - Use existing `MemoryGraphStore.query_nodes()` method
   - Filter by: `node_type=MemoryNodeType.OPINION`, `session_id`, `agent_role`
   - Handle empty results gracefully (return empty opinion_nodes array)

**Existing Code to Leverage:**
- `src/memory/graph/types.py` - MemoryNodeType.OPINION enum and fields
- `src/memory/graph/store.py` - GraphStore.query_nodes() method
- `src/memory/graph/operations.py` - Graph operations
- Department heads already write OPINION nodes (research_head.py, etc.)

---

### ARCHITECTURE COMPLIANCE

**Backend Architecture:**
- API endpoints in `src/api/`
- Follow existing patterns in `src/api/kill_switch_endpoints.py`, `src/api/provider_config_endpoints.py`
- Use FastAPI with async endpoints
- Register routes in `src/api/server.py`

**Data Flow:**
```
CopilotPanel → FloorManager → reasoning_log_endpoints → MemoryGraphStore → SQLite
```

**OPINION Node Structure (from Story 5.1):**
```python
# From src/memory/graph/types.py
class MemoryNodeType:
    OPINION = "opinion"

# OPINION-specific fields in MemoryNode:
- action: What the agent did
- reasoning: Why they did it
- confidence: Confidence score 0.0-1.0
- alternatives_considered: Options evaluated
- constraints_applied: Constraints that influenced the decision
- agent_role: Which agent/role took the action
```

---

### LIBRARY FRAMEWORK REQUIREMENTS

**Required Packages:**
- None new - uses existing packages
- `fastapi` - already in use
- `sqlalchemy` - for graph store
- `pydantic` - for response models

**Existing Components:**
- Memory Graph Store (Story 5.1)
- Department heads writing OPINION nodes
- FloorManager for routing queries

---

### FILE STRUCTURE REQUIREMENTS

**Backend Files to Create/Modify:**
```
src/
├── api/
│   ├── reasoning_log_endpoints.py    (NEW)
│   └── server.py                       (MODIFY - add router)
```

**API Route Registration:**
```python
# In src/api/server.py
from src.api.reasoning_log_endpoints import router as reasoning_router
app.include_router(reasoning_router, prefix="/api/audit/reasoning", tags=["audit"])
```

---

### TESTING REQUIREMENTS

**Backend Tests:**
- `tests/api/test_reasoning_log.py` - test GET endpoints
- Test with mock OPINION nodes in memory graph
- Test department filtering and date range queries

**Acceptance Criteria Testing:**
1. Call `GET /api/audit/reasoning/{decision_id}` with known decision_id
2. Verify response includes all required fields
3. Verify opinion_nodes array contains OPINION nodes with all fields
4. Call `GET /api/audit/reasoning/department/research`
5. Verify hypothesis chain, evidence sources, confidence scores returned

---

### PREVIOUS STORY INTELLIGENCE

**Related Stories in Epic 10:**
- Story 10.1: 5-Layer Audit System — NL Query API (backlog)
- Story 10.3: Notification Configuration API & Cold Storage Sync (backlog)
- Story 10.5: Notification Configuration Panel & Server Health (ready-for-dev)

**Dependencies:**
- Story 5.1: Graph Memory with ReflectionExecutor - creates OPINION nodes
- Story 5.4: FloorManager Agent Wiring - routes department queries

**Key Learnings:**
- OPINION nodes already contain reasoning, confidence, alternatives_considered
- Graph store query_nodes() supports filtering by department/role
- Need to link decision_id to OPINION nodes (could use session_id + timestamp)

---

### GIT INTELLIGENCE

Recent commits don't show Epic 10 work on reasoning transparency.

Relevant past commits:
- Story 5.1: Added OPINION node type to graph memory
- Story 5.2: Session checkpoint with graph memory commit flow
- Story 5.4: FloorManager agent wiring

---

### LATEST TECH INFORMATION

**No web research required** - uses existing technologies:
- FastAPI for REST endpoints (already in use)
- SQLAlchemy for graph store (already in use)
- Opinion nodes from Story 5.1 (already implemented)

---

### PROJECT CONTEXT REFERENCE

From project overview:
- Dual-node architecture: Contabo (agent/compute), Cloudzy (live trading)
- 5 Department agents: Research, Development, Risk, Trading, Portfolio
- OPINION nodes store reasoning for each department decision
- Journey 51: "The Transparent Reasoning" - full 8-of-31 strategy exposure chain

**Key Constraints:**
- Opinion nodes created by department heads (research_head.py, etc.)
- decision_id should map to session_id + timestamp for lookup
- Confidence scores are 0.0-1.0 floats

---

### IMPLEMENTATION NOTES

1. **Decision ID Lookup:**
   - Use session_id + decision_timestamp as composite key
   - Store decision_id in MemoryNode.metadata for lookup

2. **Response Construction:**
   - context_at_decision: Extract from memory node context
   - model_used: From MemoryNode metadata or agent config
   - prompt_summary: First 200 chars of node content
   - response_summary: Extract action + reasoning from OPINION fields

3. **Department Query:**
   - Filter by agent_role (department name)
   - Aggregate hypothesis chain from multiple OPINION nodes
   - Return confidence scores as-is from nodes

4. **Edge Cases:**
   - No OPINION nodes found: Return empty opinion_nodes array
   - Missing fields in node: Return None/null for optional fields
   - Invalid decision_id: Return 404 with clear error

---

### COMPLETION CHECKLIST

- [x] Backend: reasoning_log_endpoints.py created
- [x] Backend: GET /api/audit/reasoning/{decision_id} endpoint
- [x] Backend: GET /api/audit/reasoning/department/{department} endpoint
- [x] Backend: Register routes in server.py
- [x] Integration: Query Memory Graph for OPINION nodes
- [x] Tests passing
- [x] Build succeeds

---

**Status: done**
**Created: 2026-03-20**
**Story Key: 10-2-agent-reasoning-transparency-log-api**