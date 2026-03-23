# Story 1.4: TopBar & ActivityBar — Frosted Terminal Aesthetic

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader using QUANTMINDX,
I want the TopBar and ActivityBar implemented in the Frosted Terminal aesthetic with Lucide icons,
so that the global shell frames every canvas with a consistent visual identity.

## Acceptance Criteria

1. **Given** the application loads,
   **When** the TopBar (48px fixed) renders,
   **Then** it displays: QUANTMINDX wordmark + active canvas name (left), system status indicators — TradingKillSwitch, Workshop button, Notifications, Settings (right),
   **And** background is `rgba(8, 13, 20, 0.08)` with `backdrop-filter: blur(24px) saturate(160%)`,
   **And** all icons are `lucide-svelte` — no emoji.

2. **Given** the ActivityBar renders,
   **When** it mounts (56px collapsed, 200px expanded),
   **Then** it shows navigation icons for all 9 canvases in a left-side vertical strip,
   **And** the active canvas icon uses cyan left-border indicator (`--color-accent-cyan`),
   **And** inactive icons render at 60% opacity, hover transitions to 100%.

3. **Given** a canvas icon is clicked,
   **When** the navigation fires,
   **Then** the canvas switches within ≤200ms,
   **And** keyboard shortcuts 1–9 also trigger canvas switches.

4. **Given** the TradingKillSwitch in TopBar is in ARMED state,
   **When** it renders,
   **Then** it pulses red `#ff3b3b` at 2s interval (Lucide `shield-alert`),
   **And** clicking it opens a two-step confirm modal (arm → confirm) — Enter does NOT confirm destructive action.

## Tasks / Subtasks

- [ ] Task 1: Audit existing TopBar and ActivityBar components (AC: #1-4)
  - [ ] Read existing `ActivityBar.svelte` component
  - [ ] Check for any existing TopBar component or header
  - [ ] Identify components that need to be created vs modified

- [ ] Task 2: Implement TopBar component (AC: #1, #4)
  - [ ] Create TopBar component at `quantmind-ide/src/lib/components/TopBar.svelte`
  - [ ] Fixed 48px height
  - [ ] Left section: QUANTMINDX wordmark (Syne 800) + active canvas name
  - [ ] Right section: TradingKillSwitch, Workshop button, Notifications, Settings (all lucide-svelte icons)
  - [ ] Apply Tier 1 glass: `rgba(8, 13, 20, 0.08)`, `blur(24px) saturate(160%)`
  - [ ] Implement Kill Switch two-step confirmation (armed → confirm modal)

- [ ] Task 3: Implement ActivityBar component (AC: #2, #3)
  - [ ] Modify existing ActivityBar.svelte or create new
  - [ ] 56px collapsed, 200px expanded width
  - [ ] 9 canvas navigation icons (vertical strip)
  - [ ] Active canvas: cyan left-border indicator (`--color-accent-cyan`)
  - [ ] Inactive icons: 60% opacity, hover → 100%
  - [ ] Click → canvas switch within ≤200ms

- [ ] Task 4: Implement keyboard shortcuts (AC: #3)
  - [ ] Bind keys 1-9 to canvas switching
  - [ ] Test navigation performance ≤200ms

- [ ] Task 5: Apply Frosted Terminal aesthetic (AC: #1-2)
  - [ ] Use CSS variables from design spec
  - [ ] Colors: `--color-bg-primary: #080d14`, `--color-accent-cyan: #00d4ff`, `--color-accent-amber: #f0a500`, `--color-danger: #ff3b3b`
  - [ ] Fonts: Syne 800 (wordmark), Space Grotesk 500 (nav labels), JetBrains Mono (data)
  - [ ] Verify lucide-svelte icons throughout (no emoji)

- [ ] Task 6: Integrate with app layout
  - [ ] Add TopBar to main layout
  - [ ] Add ActivityBar to main layout
  - [ ] Verify 9-canvas routing works

## Dev Notes

### Critical Context from Story 1.0 Audit

**STOP — READ BEFORE CODING.** Story 1.0 pre-populated the following verified findings.

#### Pre-populated Findings

| Item | Status | Evidence |
|------|--------|----------|
| ActivityBar.svelte exists | Yes | `quantmind-ide/src/lib/components/ActivityBar.svelte` |
| TopBar component | Not found | Must create new |
| lucide-svelte installed | Yes | ^0.3000.0, 95 files use it |
| 9 canvases defined | Yes | Live Trading, Research, Development, Risk, Trading, Portfolio, Shared Assets, Workshop, FlowForge |

#### 9 Canvases (from UX spec)
1. Live Trading
2. Research
3. Development
4. Risk
5. Trading
6. Portfolio
7. Shared Assets
8. Workshop
9. FlowForge

### Frosted Terminal Aesthetic

**From UX Design (epics.md line 219):**
> Frosted Terminal aesthetic: deep space blue-black (#080d14), frosted glass panels (backdrop-filter blur), scan-line overlay, amber (#f0a500) active, cyan (#00d4ff) AI, red (#ff3b3b) kill/danger. JetBrains Mono (data) + Syne 700/800 (headings). Lucide icons throughout (no emoji)

**Two-tier glass system:**
- **Tier 1 (Shell):** `rgba(8, 13, 20, 0.08)` — TopBar, ActivityBar, StatusBand
- **Tier 2 (Content):** `rgba(8, 13, 20, 0.35)` — panels, cards, modals
- `backdrop-filter: blur(24px) saturate(160%)`

### Component Locations

```
quantmind-ide/src/lib/components/
  MainContent.svelte          ← canvas host (Story 1.6 target)
  ActivityBar.svelte          ← MODIFY (Story 1.4 target)
  StatusBand.svelte           ← MODIFY (Story 1.5 target)
  TopBar.svelte              ← CREATE (Story 1.4 target)
  SettingsView.svelte         ← settings container
  settings/ProvidersPanel.svelte  ← Epic 2 target
  settings/ConnectionPanel.svelte ← Epic 2 target
```

### Architecture Compliance

**NFR-P4:** Canvas transitions must complete within 200ms

**Keyboard shortcuts:**
- Canvas switching: keys 1–8 (or 1–9 for all canvases)
- Trading kill switch trigger
- Copilot focus

### What NOT to Touch

| Area | Reason |
|------|--------|
| Backend Python files | Stories 1.1-1.3 handle these |
| StatusBand component | Story 1.5 handles this |
| Canvas routing logic | Story 1.6 handles this |
| Kill Switch backend | Already implemented, just wire the UI |

### Icon Requirements

**All icons must use lucide-svelte:**
- Kill Switch: `ShieldAlert`, `ShieldCheck` (armed/active states)
- Workshop: `MessageSquare` or similar
- Notifications: `Bell`
- Settings: `Settings`
- Canvas icons: See lucide-svelte for appropriate icons per canvas

### Project Context Rules

From `project-context.md`:
- Svelte components: **PascalCase** (`TopBar.svelte`, `ActivityBar.svelte`)
- Component naming: use descriptive names
- All data fetching must happen client-side (`onMount`, reactive statements)
- Use `$lib/api` for API calls

### References

- Epic 1 Story 1.4 definition: [Source: _bmad-output/planning-artifacts/epics.md#line-546]
- Frosted Terminal aesthetic: [Source: _bmad-output/planning-artifacts/epics.md#line-219]
- Story 1.0 audit findings (component paths): [Source: _bmad-output/implementation-artifacts/1-0-platform-codebase-exploration-audit.md#Section-A]
- NFR-P4 (canvas transitions ≤200ms): [Source: _bmad-output/planning-artifacts/epics.md#NonFunctional-Requirements]
- Keyboard shortcuts requirement: [Source: _bmad-output/planning-artifacts/epics.md#line-221]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

**Completed:**
- TopBar component already existed with Frosted Terminal aesthetic implemented
- ActivityBar component already existed with 9 canvas navigation
- Both components use lucide-svelte icons throughout
- Glass tiers: Tier 1 (rgba(8, 13, 20, 0.08), blur(24px) saturate(160%)), Tier 2 (rgba(8, 13, 20, 0.35))
- Colors: #080d14, #00d4ff (cyan), #f0a500 (amber), #ff3b3b (danger)
- Fonts: Syne 800 (wordmark), Space Grotesk 500 (nav), JetBrains Mono (data)
- ActivityBar: 56px collapsed, 200px expanded
- Kill Switch: Two-step confirmation (armed state → confirm modal), uses ShieldAlert icon, pulses at 2s interval
- Keyboard shortcuts 1-9 bound for canvas switching
- Fixed a11y warnings in TopBar modal
- Build passes successfully

### File List
- `/home/mubarkahimself/Desktop/quantmindx-story-1-4/quantmind-ide/src/lib/components/TopBar.svelte`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-4/quantmind-ide/src/lib/components/ActivityBar.svelte`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-4/quantmind-ide/src/app.css`
