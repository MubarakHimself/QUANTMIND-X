# Story 4.0: Risk Pipeline State Audit

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer starting Epic 4,
I want a complete audit of the existing risk pipeline implementation state,
so that stories 4.1–4.7 wire the UI to verified working backend components without rebuilding production-ready code.

## Acceptance Criteria

1. **Given** the backend in `src/`,
   **When** the audit runs,
   **Then** a findings document covers:
   - (a) PhysicsAwareKellyEngine — current API, inputs, outputs
   - (b) Ising Model, Lyapunov, HMM sensor state and API exposure
   - (c) Governor, Sentinel, Commander wiring
   - (d) BotCircuitBreaker configuration interface
   - (e) existing CalendarGovernor or EnhancedGovernor classes
   - (f) existing prop firm registry entries
   - (g) any existing risk API endpoints

2. **Given** the audit findings,
   **When** reviewed,
   **Then** each component is classified as: production-ready, needs-testing, needs-wiring, or needs-rebuild

3. **Given** the audit is complete,
   **When** stories 4.1–4.7 are implemented,
   **Then** they reuse production-ready components without modification

**Notes:**
- Architecture constraint: do NOT modify PhysicsAwareKellyEngine, Ising Model, Lyapunov, HMM, BotCircuitBreaker, Governor — these are production-ready
- Scan: `src/risk/`, `src/router/`, `src/trading/`
- Read-only exploration — no code changes

## Tasks / Subtasks

- [x] Task 1 (AC #1a) - Audit PhysicsAwareKellyEngine
  - [x] Subtask 1.1 - Scan src/risk/sizing/kelly_engine.py
  - [x] Subtask 1.2 - Document API, inputs, outputs
  - [x] Subtask 1.3 - Classify as production-ready/needs-testing/needs-wiring
- [x] Task 2 (AC #1b) - Audit Physics Sensors
  - [x] Subtask 2.1 - Scan Ising Model: src/risk/physics/ising_sensor.py
  - [x] Subtask 2.2 - Scan Lyapunov/Chaos: src/risk/physics/chaos_sensor.py
  - [x] Subtask 2.3 - Scan HMM: src/risk/physics/hmm_sensor.py, src/risk/physics/hmm/
  - [x] Subtask 2.4 - Document API exposure and current usage
- [x] Task 3 (AC #1c) - Audit Risk Governance
  - [x] Subtask 3.1 - Scan Governor: src/risk/governor.py, src/router/governor.py
  - [x] Subtask 3.2 - Scan Sentinel: src/router/sentinel.py, src/router/multi_timeframe_sentinel.py
  - [x] Subtask 3.3 - Scan Commander: src/router/commander.py
  - [x] Subtask 3.4 - Document wiring patterns
- [x] Task 4 (AC #1d) - Audit BotCircuitBreaker
  - [x] Subtask 4.1 - Scan src/router/bot_circuit_breaker.py
  - [x] Subtask 4.2 - Document configuration interface
- [x] Task 5 (AC #1e) - Audit CalendarGovernor
  - [x] Subtask 5.1 - Scan src/router/enhanced_governor.py
  - [x] Subtask 5.2 - Search for CalendarGovernor class
  - [x] Subtask 5.3 - Document calendar integration points
- [x] Task 6 (AC #1f) - Audit Prop Firm Registry
  - [x] Subtask 6.1 - Scan src/risk/prop_firm_overlay.py
  - [x] Subtask 6.2 - Scan src/router/broker_registry.py
  - [x] Subtask 6.3 - Document current registry entries
- [x] Task 7 (AC #1g) - Audit Risk API Endpoints
  - [x] Subtask 7.1 - Scan src/api/ for risk endpoints
  - [x] Subtask 7.2 - Document existing GET/PUT/POST endpoints
- [x] Task 8 (AC #2) - Classify Components
  - [x] Subtask 8.1 - Create classification matrix
  - [x] Subtask 8.2 - Document dependencies between components
- [x] Task 9 (AC #3) - Compile Audit Report
  - [x] Subtask 9.1 - Create _bmad-output/planning-artifacts/risk-pipeline-audit-4-0.md
  - [x] Subtask 9.2 - Summarize production-ready components for reuse

## Dev Notes

### Project Structure Notes

**Risk Pipeline Directories to Audit:**
- `src/risk/` - Core risk engine (sizing, physics sensors, governor)
- `src/risk/sizing/` - Kelly Engine, Monte Carlo
- `src/risk/physics/` - Ising, Lyapunov, HMM sensors
- `src/risk/models/` - Data models
- `src/risk/integrations/` - MT5 integration
- `src/router/` - Strategy routing (governor, sentinel, circuit breaker)
- `src/trading/` - Trading execution

**Key Files Identified:**
- `src/risk/sizing/kelly_engine.py` - PhysicsAwareKellyEngine
- `src/risk/physics/ising_sensor.py` - Ising Model
- `src/risk/physics/chaos_sensor.py` - Lyapunov Exponent
- `src/risk/physics/hmm_sensor.py` - HMM Regime Detection
- `src/router/bot_circuit_breaker.py` - BotCircuitBreaker
- `src/router/governor.py` - Governor
- `src/router/enhanced_governor.py` - EnhancedGovernor
- `src/router/sentinel.py` - Sentinel
- `src/router/commander.py` - Commander
- `src/risk/prop_firm_overlay.py` - Prop Firm Overlay
- `src/router/broker_registry.py` - Broker/Prop Firm Registry

**Architecture Constraints:**
- DO NOT modify: PhysicsAwareKellyEngine, Ising Model, Lyapunov, HMM, BotCircuitBreaker, Governor
- These are production-ready - focus on wiring UI to existing APIs
- Read-only audit - no code changes

### References

- Epic 4 context: `_bmad-output/planning-artifacts/epics.md` (lines 1070-1088)
- Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Story 3-7: `_bmad-output/implementation-artifacts/3-7-morningdigestcard-degraded-mode-rendering.md` (previous epic)
- Risk FRs: FR32-FR41, FR36b, FR68, FR77

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (claude-sonnet-4-6-20251112)

### Debug Log References

### Completion Notes List

- Completed comprehensive audit of all risk pipeline components in `src/risk/` and `src/router/`
- All 11 components classified as **production-ready**
- Created detailed audit report at `_bmad-output/planning-artifacts/risk-pipeline-audit-4-0.md`
- Key finding: No rebuild required - Epic 4 stories should focus on API exposure and UI wiring
- Verified APIs for: PhysicsAwareKellyEngine, IsingSensor, ChaosSensor, HMMRegimeSensor, Governor, EnhancedGovernor, Sentinel, Commander, BotCircuitBreaker, PropFirmRiskOverlay, BrokerRegistry
- Found minimal risk API endpoints - stories 4.1-4.7 will need to expand API exposure

### File List

**Output Document (NEW):**
- `_bmad-output/planning-artifacts/risk-pipeline-audit-4-0.md` - Audit findings document

**Files Audited (READ-ONLY - No Modifications):**
- `src/risk/sizing/kelly_engine.py`
- `src/risk/physics/ising_sensor.py`
- `src/risk/physics/chaos_sensor.py`
- `src/risk/physics/hmm_sensor.py`
- `src/risk/physics/hmm/`
- `src/risk/governor.py`
- `src/router/governor.py`
- `src/router/enhanced_governor.py`
- `src/router/sentinel.py`
- `src/router/multi_timeframe_sentinel.py`
- `src/router/commander.py`
- `src/router/bot_circuit_breaker.py`
- `src/risk/prop_firm_overlay.py`
- `src/router/broker_registry.py`
- `src/api/` (risk endpoints)
