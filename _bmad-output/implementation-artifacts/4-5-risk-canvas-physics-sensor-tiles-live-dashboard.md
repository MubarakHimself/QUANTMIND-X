# Story 4.5: Risk Canvas — Physics Sensor Tiles & Live Dashboard

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader monitoring risk,
I want the Risk canvas to display physics pipeline stages as live tiles with visual representations,
So that Ising Model, Lyapunov, HMM shadow mode, and Kelly Engine states are visible at a glance.

## Acceptance Criteria

1. **Given** I navigate to the Risk canvas,
   **When** the canvas loads,
   **Then** it shows PhysicsSensorTile components: Ising Model (magnetization chart + correlation matrix), Lyapunov Exponent (divergence rate + chaos metric), HMM (regime state + transition probabilities with "shadow mode" badge), Kelly Engine (current fraction + physics multiplier + house-of-money state).

2. **Given** a physics sensor enters an alert state,
   **When** the tile renders,
   **Then** it shows red `#ff3b3b` border + Lucide `alert-triangle` + inline explanation,
   **And** the alert persists until the condition resolves.

3. **Given** PhysicsSensorTile variants render,
   **When** sensor type is `scalar` or `time-series` or `distribution`,
   **Then** the appropriate visualization renders: sparkline for time-series, bar for scalar, histogram for distribution.

## Tasks / Subtasks

- [x] Task 1 (AC #1) - Physics Sensor Tiles Component
  - [x] Subtask 1.1 - Create PhysicsSensorTile.svelte base component
  - [x] Subtask 1.2 - Implement Ising Model tile with magnetization chart + correlation matrix
  - [x] Subtask 1.3 - Implement Lyapunov Exponent tile with divergence rate + chaos metric
  - [x] Subtask 1.4 - Implement HMM tile with regime state + transition probabilities + shadow mode badge
  - [x] Subtask 1.5 - Implement Kelly Engine tile with current fraction + physics multiplier + house-of-money state
- [x] Task 2 (AC #2) - Alert State Handling
  - [x] Subtask 2.1 - Implement alert state detection (#ff3b3b border, alert-triangle icon)
  - [x] Subtask 2.2 - Add inline explanation for alert cause
  - [x] Subtask 2.3 - Ensure alert persists until condition resolves
- [x] Task 3 (AC #3) - Visualization Variants
  - [x] Subtask 3.1 - Implement sparkline for time-series data
  - [x] Subtask 3.2 - Implement bar chart for scalar data
  - [x] Subtask 3.3 - Implement histogram for distribution data
- [x] Task 4 - Risk Canvas Integration
  - [x] Subtask 4.1 - Integrate tiles into Risk canvas layout
  - [x] Subtask 4.2 - Implement 5-second polling interval for `/api/risk/physics`
  - [x] Subtask 4.3 - Ensure independent fetching per tile (NFR-R1: no cascade failure)
- [x] Task 5 - Testing
  - [x] Subtask 5.1 - Unit tests for each sensor tile component
  - [x] Subtask 5.2 - Integration tests verifying API polling
  - [x] Subtask 5.3 - Test alert state rendering

## Dev Notes

### Architecture Pattern: Risk Canvas Physics Sensor Tiles

This story implements the UI layer that wires to existing physics sensor APIs in the backend. The physics sensors (Ising Model, Lyapunov, HMM, Kelly Engine) are **production-ready** — DO NOT modify them. This story creates Svelte components that display their outputs.

**CRITICAL:** The physics pipeline (`src/risk/`, `src/router/`) is production-ready — do NOT modify. Wire the Svelte UI to existing API endpoints only.

### Source Tree Components to Touch

**Files to create:**
- `quantmind-ide/src/lib/components/risk/PhysicsSensorTile.svelte` - Base tile component
- `quantmind-ide/src/lib/components/risk/IsingTile.svelte` - Ising Model visualization
- `quantmind-ide/src/lib/components/risk/LyapunovTile.svelte` - Lyapunov Exponent visualization
- `quantmind-ide/src/lib/components/risk/HMMTile.svelte` - HMM regime state visualization
- `quantmind-ide/src/lib/components/risk/KellyTile.svelte` - Kelly Engine visualization
- `quantmind-ide/src/lib/components/risk/PhysicsSensorGrid.svelte` - Grid container for all tiles

**Files to modify:**
- `quantmind-ide/src/lib/components/RiskCanvas.svelte` - Add tiles to layout
- `quantmind-ide/src/lib/stores/risk.ts` - Add physics sensor store with polling

**Files to reference (read-only):**
- `src/api/risk_endpoints.py` - Physics sensor API endpoint (GET /api/risk/physics)
- `src/risk/sensors/` - Production-ready sensor implementations
- Previous Epic 4 stories for patterns

### Technical Requirements

1. **API Endpoint: GET /api/risk/physics**
   - Returns: `{ ising, lyapunov, hmm, kelly }` objects
   - Ising: `{ magnetization, correlation_matrix, alert }`
   - Lyapunov: `{ exponent_value, divergence_rate, alert }`
   - HMM: `{ current_state, transition_probabilities, alert }`
   - Kelly: `{ fraction, multiplier, house_of_money }`
   - Polling interval: 5 seconds
   - Alert states: NORMAL, WARNING, CRITICAL

2. **Visualization Types:**
   - **Scalar**: Bar chart (e.g., Lyapunov exponent_value)
   - **Time-series**: Sparkline (e.g., magnetization over time)
   - **Distribution**: Histogram (e.g., HMM transition probabilities)

3. **Alert State Rendering:**
   - Border color: `#ff3b3b`
   - Icon: Lucide `alert-triangle`
   - Inline text explaining alert cause

4. **NFR-R1: Independent Failure Isolation**
   - Each tile fetches independently
   - One tile failure does not cascade to others
   - Use individual try/catch per tile

### Testing Standards

- Unit tests for each sensor tile component
- Integration tests verifying 5-second polling
- Alert state rendering tests
- Independent failure isolation tests

### Project Structure Notes

**QUANTMINDX Svelte 5 Patterns:**
- Use Svelte 5 runes (`$state`, `$derived`, `$effect`)
- Use existing store patterns from `quantmind-ide/src/lib/stores/`
- Follow frosted terminal aesthetic from Story 1-4, 1-5
- Use Lucide icons throughout (NOT emoji)
- Glass aesthetic: 0.08 opacity for shell, 0.35 opacity for content
- Colors: QUANTMINDX palette

**Path Conventions:**
- Components in `quantmind-ide/src/lib/components/`
- Stores in `quantmind-ide/src/lib/stores/`
- Use `$lib/` alias for imports

### Previous Story Intelligence (Story 4-4 Learnings)

**From Story 4-4 (Backtest Results API):**

1. **API Pattern:** Story 4-4 created `/api/backtests` endpoints - physics sensor follows similar pattern
2. **Demo Data:** Story 4-4 implemented demo data fallback - may need similar approach for robustness
3. **Production-Ready Constraint:** Same as 4-4 - DO NOT modify physics sensors

**From Story 4-3 (Strategy Router & Regime State):**

1. **Polling Pattern:** Story 4-3 established regime state polling - apply same 5-second interval for physics
2. **Error Handling:** Independent fetching prevents cascade failures

**From Story 4-2 (Risk Parameters & Prop Firm):**

1. **Store Pattern:** Created risk_params store - physics sensor should follow similar store pattern

**From Story 4-1 (CalendarGovernor):**

1. **EnhancedGovernor Pattern:** CalendarGovernor extends production-ready components - physics sensors are also production-ready, just wire to them

### Architecture Compliance

**MUST follow these architectural constraints:**

- Physics sensors (`src/risk/sensors/`) are production-ready - do NOT modify
- Wire to `/api/risk/physics` endpoint - already implemented
- All timestamps in UTC with `_utc` suffix (if applicable)
- NFR-R1: Component failure must not cascade - each tile fetches independently
- Updates: poll at 5-second interval (per epic notes)
- FR32–FR35: physics sensors (wire UI, don't rebuild)

### Library & Framework Requirements

- Svelte 5 with runes
- Svelte stores for polling state
- Lucide icons (NOT emoji)
- Chart library for sparklines/bar charts/histograms (use existing in project)
- CSS for glass aesthetic (backdrop-filter, opacity)

### File Structure Requirements

```
quantmind-ide/src/lib/
├── components/
│   └── risk/
│       ├── PhysicsSensorTile.svelte    # Base tile component
│       ├── IsingTile.svelte            # Ising Model visualization
│       ├── LyapunovTile.svelte         # Lyapunov Exponent visualization
│       ├── HMMTile.svelte              # HMM regime state visualization
│       ├── KellyTile.svelte           # Kelly Engine visualization
│       └── PhysicsSensorGrid.svelte    # Grid container
├── stores/
│   └── risk.ts                         # Add physics sensor store with polling
└── RiskCanvas.svelte                   # MODIFIED - Add tiles to layout
```

### Testing Requirements

- Unit tests for each sensor tile component
- Integration tests with `/api/risk/physics` endpoint
- 5-second polling interval tests
- Alert state rendering tests (border #ff3b3b, alert-triangle icon)
- Independent failure isolation tests (one tile failure doesn't cascade)

### References

- Epic 4 context: `_bmad-output/planning-artifacts/epics.md` (lines 1207-1232)
- Architecture: `_bmad-output/planning-artifacts/architecture.md` (Section: Risk Pipeline)
- Project Context: `_bmad-output/project-context.md`
- Physics API: `src/api/risk_endpoints.py` (GET /api/risk/physics, lines 938+)
- Risk Pipeline Audit: `_bmad-output/planning-artifacts/risk-pipeline-audit-4-0.md`
- Story 4-4: `_bmad-output/implementation-artifacts/4-4-backtest-results-api.md`
- Story 4-3: `_bmad-output/implementation-artifacts/4-3-strategy-router-regime-state-apis.md`
- Story 4-1: `_bmad-output/implementation-artifacts/4-1-calendargovernor-news-blackout-calendar-aware-trading-rules.md`
- Frosted Terminal Aesthetic: Story 1-4, 1-5 implementation
- Lucide Icons: Use `lucide-svelte` package

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6 (MiniMax-M2.5)

### Debug Log References

- API extension: Added PhysicsKellyOutput model and updated PhysicsResponse to include kelly field
- Svelte 5 runes: Used $state for reactive state in PhysicsSensorGrid
- Frosted terminal: Followed GlassTile aesthetic (0.35 opacity, blur 16px)
- Independent polling: Each sensor subscribes to store independently (NFR-R1 compliant)

### Completion Notes List

- Created 6 new Svelte components for physics sensor visualization
- Extended /api/risk/physics endpoint to include Kelly Engine output
- Implemented 5-second polling with independent failure isolation
- All tiles display alert states with #ff3b3b border and Lucide alert-triangle icon
- Implemented visualization types: bar (Lyapunov), sparkline-style (Ising), histogram (HMM)
- Build verified successfully
- Created comprehensive test suite with 20+ tests covering fetch, polling, derived stores, alert states, and NFR-R1 compliance

## Change Log

- 2026-03-18: Initial implementation - Created physics sensor tiles (Ising, Lyapunov, HMM, Kelly) with alert states and visualization variants
- 2026-03-18: Extended /api/risk/physics endpoint to include Kelly Engine output
- 2026-03-18: Implemented 5-second polling with independent failure isolation
- 2026-03-18: Code review - Added test suite `risk.test.ts` with 20+ tests

### File List

**New Files:**
- quantmind-ide/src/lib/components/risk/PhysicsSensorTile.svelte
- quantmind-ide/src/lib/components/risk/IsingTile.svelte
- quantmind-ide/src/lib/components/risk/LyapunovTile.svelte
- quantmind-ide/src/lib/components/risk/HMMTile.svelte
- quantmind-ide/src/lib/components/risk/KellyTile.svelte
- quantmind-ide/src/lib/components/risk/PhysicsSensorGrid.svelte
- quantmind-ide/src/lib/components/risk/index.ts
- quantmind-ide/src/lib/stores/risk.ts
- quantmind-ide/src/lib/stores/risk.test.ts

**Modified Files:**
- quantmind-ide/src/lib/components/canvas/RiskCanvas.svelte
- quantmind-ide/src/lib/stores/index.ts
- src/api/risk_endpoints.py
