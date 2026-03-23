# Story 1.5: StatusBand Redesign — Frosted Terminal Ticker

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader using QUANTMINDX,
I want the StatusBand redesigned as a 28–32px Frosted Terminal ambient ticker,
so that live portfolio metrics, system health, and regime data are always visible and navigable.

## Acceptance Criteria

1. **Given** the application is running,
   **When** the StatusBand (32px fixed bottom) renders,
   **Then** it displays scrolling segments: session clocks (Tokyo/London/NY with open/closed state), active bot count, daily P&L, node health dots (Cloudzy · Contabo · Local), workflow count, challenge progress,
   **And** background is Tier 1 glass (`rgba(8,13,20,0.08)`, `blur(24px)`),
   **And** ticker text uses Fragment Mono 400 11–12px.

2. **Given** a StatusBand segment is clicked,
   **When** the click registers,
   **Then** session clocks → Live Trading canvas, active bots → Portfolio canvas, risk mode → Risk canvas, node health dot → node status overlay.

3. **Given** a metric value changes,
   **When** the update renders,
   **Then** positive P&L flashes green `#00c896` (100ms), negative flashes red `#ff3b3b` (100ms),
   **And** `aria-live="polite"` announces changes to screen readers without disruption.

4. **Given** Contabo node is unreachable,
   **When** the StatusBand renders degraded state,
   **Then** Contabo dot shows red with Lucide `wifi-off` icon,
   **And** all live data continues displaying last-known values with `[stale]` label.

## Tasks / Subtasks

- [x] Task 1: Read existing StatusBand.svelte (AC: #1-4)
  - [x] Read `quantmind-ide/src/lib/components/StatusBand.svelte` completely
  - [x] Understand current state variables and API calls
  - [x] Note what needs to be preserved vs redesigned

- [x] Task 2: Redesign StatusBand layout (AC: #1)
  - [x] Fixed 32px height at bottom of screen
  - [x] Apply Tier 1 glass: `rgba(8,13,20,0.08)`, `blur(24px)`
  - [x] Scrolling ticker segments layout
  - [x] Use Fragment Mono 400 11–12px font

- [x] Task 3: Implement ticker segments (AC: #1)
  - [x] Session clocks (Tokyo/London/NY with open/closed state icons)
  - [x] Active bot count
  - [x] Daily P&L with color coding
  - [x] Node health dots (Cloudzy · Contabo · Local)
  - [x] Workflow count
  - [x] Challenge progress (prop firm)

- [x] Task 4: Implement click navigation (AC: #2)
  - [x] Session clocks → Live Trading canvas
  - [x] Active bots → Portfolio canvas
  - [x] Risk mode → Risk canvas
  - [x] Node health dot → node status overlay

- [x] Task 5: Implement value change animations (AC: #3)
  - [x] Positive P&L: flash green `#00c896` (100ms)
  - [x] Negative P&L: flash red `#ff3b3b` (100ms)
  - [x] Add `aria-live="polite"` for accessibility

- [x] Task 6: Implement degraded mode (AC: #4)
  - [x] Detect Contabo node unreachable
  - [x] Show red dot with `wifi-off` icon
  - [x] Display last-known values with `[stale]` label

- [x] Task 7: Apply Frosted Terminal aesthetic
  - [x] Use CSS variables from design spec
  - [x] All Lucide icons (no emoji)
  - [x] Ensure accessibility compliance

## Dev Notes

### Critical Context from Story 1.0 Audit

**STOP — READ BEFORE CODING.** Story 1.0 pre-populated the following verified findings.

#### Pre-populated Findings

| Item | Status | Evidence |
|------|--------|----------|
| StatusBand.svelte exists | Yes | `quantmind-ide/src/lib/components/StatusBand.svelte` |
| Syntax | Svelte 4 | Uses `let` state vars, `onMount`, `onDestroy` |
| State vars | 16 vars | loading, sessions, regime, activeBots, dailyPnl, etc. |
| Session constants | SESSION_ORDER | ['ASIAN', 'LONDON', 'NEW_YORK', 'OVERLAP'] |
| lucide-svelte | Installed | ^0.3000.0 |
| Frosted Terminal design | Not applied | Needs redesign |

#### Current StatusBand State (from Story 1.0)

**Location:** `quantmind-ide/src/lib/components/StatusBand.svelte`

**State variables:**
- `loading`, `sessions`, `sessionsError`, `regime`
- `activeBots`, `dailyPnl`, `winRate`, `openPositions`, `tradesToday`
- `currentSession`, `currentTime`, `riskMode`, `routerMode`
- `refreshInterval`, `timeInterval`

**Imports:**
- `onMount`, `onDestroy` from `svelte`
- Lucide icons: Bot, DollarSign, Percent, Shield, Route, TrendingUp, Activity, Target, Clock
- API functions from `$lib/api`
- `navigationStore` from `../stores/navigationStore`

**Current issues (to fix):**
- Uses Svelte 4 syntax (needs migration to Svelte 5 per Story 1.2)
- NOT matching Frosted Terminal ticker design from UX spec
- Needs click navigation to canvases
- Needs degraded mode handling

### Frosted Terminal Aesthetic

**From UX Design (epics.md line 219):**
> Frosted Terminal aesthetic: deep space blue-black (#080d14), frosted glass panels, amber (#f0a500) active, cyan (#00d4ff) AI, red (#ff3b3b) kill/danger. JetBrains Mono (data) + Syne 700/800 (headings). Lucide icons throughout (no emoji)

**Two-tier glass system:**
- **Tier 1 (Shell):** `rgba(8, 13, 20, 0.08)` — TopBar, ActivityBar, StatusBand
- `backdrop-filter: blur(24px) saturate(160%)`

### Session Clocks

**From UX Design (epics.md line 218):**
> Session-scoped time model: temporal reference is the trading session, not wall clock. Three live session clocks in StatusBand (Tokyo/Asian, London/European, New York/NY) — real current time + open/closed state

### Node Health Monitoring

**From epics.md:**
- Cloudzy: Trading node (MT5, execution)
- Contabo: Agent/compute node
- Local: Development machine

### Accessibility Requirements

- `aria-live="polite"` on ambient updates
- Screen reader announcements without disruption
- Keyboard navigable

### What NOT to Touch

| Area | Reason |
|------|--------|
| Backend API endpoints | Already exist, wire to them |
| Kill Switch backend | Already implemented |
| Session time logic | Keep existing, apply new styling |
| Canvas routing | Story 1.6 handles |

### References

- Epic 1 Story 1.5 definition: [Source: _bmad-output/planning-artifacts/epics.md#line-583]
- Frosted Terminal aesthetic: [Source: _bmad-output/planning-artifacts/epics.md#line-219]
- Session time model: [Source: _bmad-output/planning-artifacts/epics.md#line-218]
- Story 1.0 audit findings (StatusBand state): [Source: _bmad-output/implementation-artifacts/1-0-platform-codebase-exploration-audit.md#Section-A]
- NFR-P2 (live data ≤3s lag): [Source: _bmad-output/planning-artifacts/epics.md#NonFunctional-Requirements]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- Migrated from Svelte 4 to Svelte 5 runes ($state, $derived, $effect)
- Applied Tier 1 Frosted Terminal glass aesthetic (rgba(8,13,20,0.08), blur(24px))
- Fixed 32px height at bottom of screen with position: fixed
- Implemented scrolling ticker with session clocks (Tokyo/London/NY)
- Added node health dots (Cloudzy, Contabo, Local) with degraded mode
- Implemented P&L flash animations (100ms green/red)
- Added click navigation to Live Trading, Portfolio, and Risk views
- Added aria-live="polite" for accessibility
- All Lucide icons used (no emoji)
- Build passes successfully

### File List

- `quantmind-ide/src/lib/components/StatusBand.svelte`
