# Story 12.2: Design Token Consistency Pass

Status: done

## Story

As a **trader (Mubarak)**,
I want **all visual styles across every canvas and component to draw from a single, consistent Frosted Terminal token system** — eliminating the competing OKLCH variable layer and the mismatched colour values that cause visual noise,
So that every canvas renders with the correct colours, glass opacity, typography, and spacing that match the UX spec exactly, and so that any developer building future components has one authoritative token reference to follow.

## Acceptance Criteria

**AC 12-2-1: No OKLCH vars in `:root`**
- **Given** `app.css` is opened
- **When** a developer searches for `oklch(`
- **Then** zero results are found in the `:root` block

**AC 12-2-2: Canonical colour values at default theme**
- **Given** the app loads with no theme override (Balanced Terminal default)
- **When** any canvas is rendered
- **Then** background resolves to `#080d14`, cyan = `#00d4ff`, amber = `#f0a500`, green = `#00c896`, red = `#ff3b3b`, text primary = `#e8edf5`, text muted = `#5a6a80`

**AC 12-2-3: Dept accent overrides resolve correctly**
- **Given** Research canvas is active (root element carries `data-dept="research"`)
- **When** dept-accented elements render
- **Then** `--dept-accent` resolves to `--color-accent-amber` (#f0a500)
- **And** `data-dept="risk"` resolves to `--color-accent-red`, `data-dept="development"` to `--color-accent-cyan`

**AC 12-2-4: Zero components using removed OKLCH vars**
- **Given** all `.svelte` files are scanned
- **When** a developer searches for `var(--bg-primary)`, `var(--accent-primary)`, `var(--text-primary)`, `var(--color-danger)`, `var(--accent-danger)`
- **Then** no results are found — all replaced with Frosted Terminal equivalents

**AC 12-2-5: Theme presets functional after token rename**
- **Given** the user switches to "Ghost Panel" in Settings → Appearance
- **When** the theme applies
- **Then** `--tile-min-width` = 220px, `--tile-gap` = 10px (compact tiles — per `ux-design-directions.html` DIR-3)
- **And** glass opacity drops to near-transparent (0.10) — wallpaper shows through
- **And** the AppearancePanel theme switch continues to work without error

**AC 12-2-6: Balanced Terminal default density (launched default)**
- **Given** no theme override is active (Balanced Terminal is the out-of-box default)
- **When** a tile grid renders
- **Then** `--tile-min-width` = 280px, `--tile-gap` = 18px — per `ux-design-directions.html` DIR-5 exact values
- **And** Breathing Space override (`data-theme="breathing-space"`) sets 4-item StatusBand density, same tile sizing (280px / 18px)
- **And** Ghost Panel override (`data-theme="ghost-panel"`) sets `--tile-min-width` = 220px, `--tile-gap` = 10px, near-transparent glass (`glassOp:0.10`)

**AC 12-2-7: No `--color-danger` token**
- **Given** `app.css` is inspected
- **When** searching for `--color-danger`
- **Then** no such property is defined — canonical token is `--color-accent-red`

**AC 12-2-8: Full spacing + typography scale defined**
- **Given** a developer needs spacing or type tokens
- **When** they check `app.css` `:root`
- **Then** `--space-1` through `--space-12`, `--text-xs` through `--text-2xl`, `--font-data`, `--font-heading`, `--font-body`, `--font-ambient` are all present and correct

## Tasks / Subtasks

- [x] Task 1: Audit and document all OKLCH token usages in `app.css` (AC: #1, #7)
  - [x] 1.1: Count all `oklch(` occurrences in `app.css` `:root` block
  - [x] 1.2: Map each OKLCH var to its Frosted Terminal replacement (use token replacement map in Dev Notes)
  - [x] 1.3: Remove all OKLCH vars from `:root` — `--bg-primary`, `--bg-secondary`, `--bg-tertiary`, `--bg-input`, `--bg-glass`, `--accent-primary`, `--accent-secondary`, `--accent-finance`, `--accent-success`, `--accent-warning`, `--accent-danger`, `--text-primary`, `--text-secondary`, `--text-muted`, `--text-accent`, `--border-subtle`, `--border-medium`, `--border-strong`, `--border-accent`
  - [x] 1.4: Remove `--color-danger` from `:root` (duplicate of `--color-accent-red`)
  - [x] 1.5: Remove syntax color vars from `:root` (OKLCH-based `--syntax-*` — deferred to Epic 8 Monaco)

- [x] Task 2: Expand Frosted Terminal token block in `app.css` `:root` (AC: #2, #3, #8)
  - [x] 2.1: Add canonical background tokens: `--color-bg-base`, `--color-bg-surface`, `--color-bg-elevated`, `--color-bg-input` (see canonical values in Dev Notes §Token Canonical Values)
  - [x] 2.2: Confirm accent tokens already present: `--color-accent-cyan`, `--color-accent-amber`, `--color-accent-green`, `--color-accent-red` — verify hex values match spec
  - [x] 2.3: Add text tokens: `--color-text-primary`, `--color-text-muted`, `--color-text-secondary`
  - [x] 2.4: Add border tokens: `--color-border-subtle`, `--color-border-medium`
  - [x] 2.5: Add glass tier tokens: `--glass-shell-bg` (Tier 1: `rgba(8,13,20,0.08)`), `--glass-content-bg` (Tier 2: `rgba(8,13,20,0.35)`), `--glass-blur` (`blur(12px) saturate(160%)`)
  - [x] 2.6: Add typography tokens: `--font-data`, `--font-heading`, `--font-body`, `--font-ambient` (see values in Dev Notes)
  - [x] 2.7: Add full spacing scale: `--space-1` (4px) through `--space-12` (48px) — 9 tokens total (see Dev Notes §Spacing Scale)
  - [x] 2.8: Add full type scale: `--text-xs` (11px) through `--text-2xl` (36px) — 7 tokens total (see Dev Notes §Type Scale)
  - [x] 2.9: Add dept accent system: `--dept-accent` default + `[data-dept="research"]`, `[data-dept="risk"]`, `[data-dept="development"]`, `[data-dept="trading"]`, `[data-dept="portfolio"]` overrides
  - [x] 2.10: Set Balanced Terminal default density: `--tile-min-width: 280px`, `--tile-gap: 18px`, `--sb-density: dense` (8 items)

- [x] Task 3: Update 4 theme preset blocks in `app.css` (AC: #5, #6)
  - [x] 3.1: `[data-theme="ghost-panel"]` — replace all OKLCH vars with Frosted Terminal equivalents; set `--glass-opacity: 0.10`, `--tile-min-width: 220px`, `--tile-gap: 10px` (DIR-3 compact)
  - [x] 3.2: `[data-theme="balanced-terminal"]` — ensure explicit `:root` default values; `--glass-opacity: 0.78`, `--tile-min-width: 280px`, `--tile-gap: 18px`, `--sb-density: dense` (DIR-5)
  - [x] 3.3: `[data-theme="breathing-space"]` — replace OKLCH vars; set `--glass-opacity: 0.78`, `--tile-min-width: 280px`, `--tile-gap: 18px`, `--sb-density: comfortable` (4-item minimal per DIR-2)
  - [x] 3.4: `[data-theme="open-air"]` — replace OKLCH vars with Frosted Terminal equivalents; preserve distinct styling intent (minimal glass)
  - [x] 3.5: Verify `AppearancePanel.svelte` theme switch selector values match the `data-theme` attribute values used in `app.css` (do NOT edit AppearancePanel internals — Settings is preserved as-is)

- [x] Task 4: Scan and replace OKLCH token usage in `.svelte` files (AC: #4)
  - [x] 4.1: Run grep across all `.svelte` files for `--bg-primary`, `--bg-secondary`, `--bg-tertiary`, `--bg-input`, `--bg-glass` → replace with canonical equivalents per token replacement map
  - [x] 4.2: Run grep for `--accent-primary`, `--accent-secondary`, `--accent-danger`, `--accent-success`, `--accent-warning` → replace with canonical equivalents
  - [x] 4.3: Run grep for `--text-primary`, `--text-muted`, `--text-secondary`, `--text-accent` → replace with canonical equivalents
  - [x] 4.4: Run grep for `--border-subtle`, `--border-medium`, `--border-strong` → replace with canonical equivalents
  - [x] 4.5: Run grep for `--color-danger` (the exact duplicate) → replace with `--color-accent-red`
  - [x] 4.6: **DO NOT edit** files in `settings/` sub-directory (`AppearancePanel.svelte`, `NotificationSettingsPanel.svelte`, `ServerHealthPanel.svelte`, `ServersPanel.svelte`, `ProvidersPanel.svelte`) — Settings is preserved as-is per epic mandate
  - [x] 4.7: For each replaced file — verify the colour semantics are correct (danger → red, success → green, warning → amber, primary accent → cyan) before committing

- [x] Task 5: Update scrollbar and utility classes to use new tokens (AC: #2)
  - [x] 5.1: Replace `var(--bg-tertiary)` in scrollbar thumb with `var(--color-bg-elevated)`
  - [x] 5.2: Replace `var(--bg-primary)` in scrollbar track/border with `var(--color-bg-base)`
  - [x] 5.3: Replace `var(--border-strong)` in scrollbar hover with `var(--color-border-medium)`
  - [x] 5.4: Update `html, body { color: var(--text-primary) }` → `var(--color-text-primary)`

- [x] Task 6: Write Vitest snapshot/unit tests (AC: all)
  - [x] 6.1: Verify `app.css` contains no `oklch(` using a file content assertion
  - [x] 6.2: Verify all 4 theme preset blocks set `--tile-min-width` and `--tile-gap`
  - [x] 6.3: Verify `--color-danger` is not defined (AC 12-2-7)
  - [x] 6.4: Verify `--font-data`, `--font-heading`, `--font-body`, `--font-ambient` are defined in `:root`
  - [x] 6.5: Verify spacing scale tokens `--space-1` through `--space-12` are defined

## Dev Notes

### CRITICAL ANTI-PATTERNS — DO NOT DO THESE

1. **DO NOT edit Settings sub-panels** — `AppearancePanel.svelte`, `NotificationSettingsPanel.svelte`, `ServerHealthPanel.svelte`, `ServersPanel.svelte`, `ProvidersPanel.svelte` are preserved as-is per Mubarak's epic mandate. Even if they reference OKLCH tokens — leave them. The Settings section is excluded from this story.

2. **DO NOT change Monaco syntax highlighting vars** — `--syntax-keyword`, `--syntax-string`, `--syntax-number`, `--syntax-comment`, `--syntax-function`, `--syntax-variable`, `--syntax-operator`, `--syntax-background` are deferred to Epic 8 (Monaco integration). Remove them from `:root` but do not replace with named equivalents yet — they are unused outside Monaco.

3. **DO NOT modify font file loading** — fonts are loaded via the `@import url('https://fonts.googleapis.com/css2?...')` at line 1 of `app.css`. The `--font-*` token additions only add CSS variable names pointing to the already-loaded font families. Do not change the import statement.

4. **DO NOT use hardcoded colours anywhere** — after this story, every colour in every component must come from a CSS custom property token. If a `.svelte` file has a hardcoded hex value (e.g., `color: #080d14`), replace it with the appropriate token but only if it is within the OKLCH migration scope.

5. **DO NOT change the OKLCH-based theme overrides to match the wrong values** — the resolution is: Frosted Terminal hex values in `app.css` are canonical (match UX spec exactly). The `ux-design-directions.html` prototype uses slightly different values (`--c-amber: #d4920e`, `--c-cyan: #00aacc`) — those are prototype approximations, NOT the authoritative spec. Trust `app.css` Frosted Terminal hex values.

6. **DO NOT use `export let` or `writable()` if creating any test helpers** — all new code uses Svelte 5 runes: `$state`, `$derived`, `$props`, `$effect`.

7. **DO NOT re-invent the glass tier classes** — `--glass-shell-bg` (Tier 1, `rgba(8,13,20,0.08)`) and `--glass-content-bg` (Tier 2, `rgba(8,13,20,0.35)`) already exist partially in `app.css` as `--glass-tier-1` / `--glass-tier-2`. Rename them to the canonical names. Do not create new utility classes — just consolidate the token names.

---

### Token Replacement Map (OKLCH → Frosted Terminal)

| Old OKLCH token | New canonical Frosted Terminal token | Hex / value |
|---|---|---|
| `--bg-primary` | `--color-bg-base` | `#080d14` |
| `--bg-secondary` | `--color-bg-surface` | `rgba(8,13,20,0.6)` |
| `--bg-tertiary` | `--color-bg-elevated` | `rgba(16,24,36,0.8)` |
| `--bg-input` | `--color-bg-elevated` | same as elevated |
| `--bg-glass` | `--glass-content-bg` | `rgba(8,13,20,0.35)` |
| `--text-primary` | `--color-text-primary` | `#e8edf5` |
| `--text-secondary` | `--color-text-secondary` | `rgba(232,237,245,0.6)` |
| `--text-muted` | `--color-text-muted` | `#5a6a80` |
| `--text-accent` | `--color-accent-cyan` | `#00d4ff` |
| `--border-subtle` | `--color-border-subtle` | `rgba(255,255,255,0.06)` |
| `--border-medium` | `--color-border-medium` | `rgba(255,255,255,0.12)` |
| `--border-strong` | `--color-border-medium` | same as medium (strong is unused) |
| `--border-accent` | `--color-accent-cyan` | `#00d4ff` |
| `--accent-primary` | `--color-accent-cyan` | `#00d4ff` (default accent) |
| `--accent-danger` / `--color-danger` | `--color-accent-red` | `#ff3b3b` |
| `--accent-success` | `--color-accent-green` | `#00c896` |
| `--accent-warning` | `--color-accent-amber` | `#f0a500` |
| `--accent-secondary` | `--color-accent-amber` | `#f0a500` (nearest semantic match) |
| `--accent-finance` | `--color-accent-amber` | `#f0a500` (gold/amber) |

---

### Token Canonical Values (`:root` Target State)

**Background tokens:**
```css
--color-bg-base:        #080d14;
--color-bg-surface:     rgba(8, 13, 20, 0.6);
--color-bg-elevated:    rgba(16, 24, 36, 0.8);
```

**Glass tier tokens (rename from existing `--glass-tier-1` / `--glass-tier-2`):**
```css
--glass-shell-bg:       rgba(8, 13, 20, 0.08);   /* Tier 1 — shell surfaces */
--glass-content-bg:     rgba(8, 13, 20, 0.35);   /* Tier 2 — content tiles */
--glass-blur:           blur(12px) saturate(160%);
```

**Accent tokens (already in `app.css` — verify hex values):**
```css
--color-accent-cyan:    #00d4ff;
--color-accent-amber:   #f0a500;
--color-accent-green:   #00c896;
--color-accent-red:     #ff3b3b;
```

**Text tokens:**
```css
--color-text-primary:   #e8edf5;
--color-text-secondary: rgba(232, 237, 245, 0.6);
--color-text-muted:     #5a6a80;
```

**Border tokens:**
```css
--color-border-subtle:  rgba(255, 255, 255, 0.06);
--color-border-medium:  rgba(255, 255, 255, 0.12);
```

**Typography tokens (canonical — UX Spec §Font Stack):**
```css
--font-data:      'JetBrains Mono', monospace;
--font-heading:   'Syne', sans-serif;
--font-body:      'Space Grotesk', 'IBM Plex Sans', sans-serif;
--font-ambient:   'Fragment Mono', 'Geist Mono', monospace;
```

Note: `app.css` currently has partial font vars (`--font-family`, `--font-display`, `--font-nav`, `--font-mono`). Replace / augment with canonical names. Existing `--font-mono: 'JetBrains Mono'` maps to `--font-data`. The `--font-family` (Inter) is the page default — retain as-is since it predates the Frosted Terminal system.

---

### Spacing Scale (UX Spec §Spacing)

```css
--space-1:   4px;
--space-2:   8px;
--space-3:   12px;
--space-4:   16px;
--space-5:   20px;
--space-6:   24px;
--space-8:   32px;
--space-10:  40px;
--space-12:  48px;
```

---

### Type Scale (UX Spec §Typography)

```css
--text-xs:    11px;   /* ambient, ticker (Fragment Mono) */
--text-sm:    12px;   /* data, secondary labels */
--text-base:  14px;   /* body, labels */
--text-md:    16px;   /* prominent body */
--text-lg:    20px;   /* section headers */
--text-xl:    28px;   /* canvas titles (Syne) */
--text-2xl:   36px;   /* hero numbers — P&L, balance */
```

---

### Dept Accent System

All canvas components in Story 12-3 will carry `data-dept` on their root element. The CSS token resolver applies `--dept-accent` per dept:

```css
/* Default */
:root {
  --dept-accent: var(--color-accent-cyan);
}

[data-dept="research"]    { --dept-accent: var(--color-accent-amber); }
[data-dept="risk"]        { --dept-accent: var(--color-accent-red); }
[data-dept="development"] { --dept-accent: var(--color-accent-cyan); }
[data-dept="trading"]     { --dept-accent: var(--color-accent-green); }
[data-dept="portfolio"]   { --dept-accent: var(--color-accent-cyan); }
[data-dept="workshop"]    { --dept-accent: var(--color-accent-cyan); }
[data-dept="flowforge"]   { --dept-accent: var(--color-accent-cyan); }
[data-dept="shared"]      { --dept-accent: var(--color-text-muted); }
```

These overrides live in `app.css` — they do NOT require JS. Story 12-3 needs them to exist before `data-dept` attributes are added to canvas roots.

---

### Theme Preset Blocks Target State

**Balanced Terminal (`:root` default — DIR-5):**
```css
/* These are the `:root` defaults — no data-theme needed */
--glass-opacity:       0.78;
--glass-blur-radius:   18px;
--tile-min-width:      280px;
--tile-gap:            18px;
--sb-density:          dense;  /* 8-item StatusBand */
```

**Ghost Panel (`[data-theme="ghost-panel"]` — DIR-3):**
```css
--glass-opacity:       0.10;
--glass-blur-radius:   6px;
--tile-min-width:      220px;
--tile-gap:            10px;
--sb-density:          dense;  /* 8-item StatusBand */
```

**Breathing Space (`[data-theme="breathing-space"]` — DIR-2):**
```css
--glass-opacity:       0.78;
--glass-blur-radius:   18px;
--tile-min-width:      280px;
--tile-gap:            18px;
--sb-density:          comfortable;  /* 4-item minimal StatusBand */
```

**Open Air (`[data-theme="open-air"]`):**
```css
--glass-opacity:       0.05;
--glass-blur-radius:   24px;
--tile-min-width:      260px;
--tile-gap:            14px;
--sb-density:          dense;
```

Important: Current `app.css` theme blocks only partially override OKLCH vars (e.g., `--bg-primary`, `--accent-primary`) — these must be replaced with Frosted Terminal equivalents. Ghost Panel currently has wrong background hue (220 hue Kanagawa) — remove the bg override and let `:root --color-bg-base: #080d14` apply. Only glass + density overrides are needed per theme.

---

### Files to Edit (Exact Paths)

**Primary target (1 file — all token changes):**
- `quantmind-ide/src/app.css`

**Secondary targets (OKLCH var replacement in Svelte components):**
- 148 files currently use OKLCH-derived tokens (confirmed via grep)
- Focus on files that use: `--bg-primary`, `--accent-primary`, `--text-primary`, `--color-danger`, `--accent-danger`, `--text-muted` (bare, no suffix), `--border-subtle`, `--border-medium`
- Settings sub-panels (`settings/`) are explicitly excluded — DO NOT EDIT THEM

**Confirmed files with OKLCH usage (non-Settings, highest priority):**
- `quantmind-ide/src/routes/+page.svelte` (main shell)
- `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` (created in Story 12-1 — may need token fix)
- `quantmind-ide/src/lib/components/shared/RichRenderer.svelte` (created in Story 12-1)
- `quantmind-ide/src/lib/components/canvas/CanvasPlaceholder.svelte`
- `quantmind-ide/src/lib/components/TopBar.svelte`
- `quantmind-ide/src/lib/components/ActivityBar.svelte`
- `quantmind-ide/src/lib/components/StatusBar.svelte`
- `quantmind-ide/src/lib/components/MainContent.svelte`
- `quantmind-ide/src/lib/components/Sidebar.svelte`
- All `quantmind-ide/src/lib/components/live-trading/*.svelte`
- All `quantmind-ide/src/lib/components/kill-switch/*.svelte`

---

### Project Structure Notes

- Token system lives exclusively in `quantmind-ide/src/app.css` `:root` — no other CSS files own root tokens
- Theme overrides are `[data-theme="..."]` attribute selectors on `<html>` or `<body>` — the `AppearancePanel.svelte` sets this attribute at runtime; that logic is not touched
- The `--agent-panel-width: 320px` token was added by Story 12-1 — retain it
- Svelte 5 runes throughout — no `export let` or `writable()`
- All components under 500 lines (NFR-MAINT-1)
- Lucide icons only — no emoji (NFR-MAINT-3)

### Dependency Context

- **Story 12-1 is DONE** — `AgentPanel.svelte` in `components/shell/` was created. It likely uses Frosted Terminal tokens directly but was written after the OKLCH era. Verify it uses canonical token names (e.g., `--color-accent-cyan` not `--accent-primary`).
- **Story 12-3 is blocked on 12-2** — it needs `--dept-accent` overrides and `--tile-min-width`/`--tile-gap` to exist before building the tile grid. This story must deliver those tokens.
- **Story 12-3 also needs** `--glass-shell-bg`, `--glass-content-bg`, `--glass-blur` for `GlassSurface.svelte` and `TileCard.svelte`.

### References

- [Source: _bmad-output/planning-artifacts/epic-12-stories.md#Story 12-2]
- [Source: _bmad-output/planning-artifacts/ux-design-specification.md#Canonical Token Values §CSS Custom Properties]
- [Source: _bmad-output/planning-artifacts/ux-design-specification.md#Typography §Font Stack]
- [Source: _bmad-output/planning-artifacts/ux-design-specification.md#Spacing §Spacing Scale]
- [Source: _bmad-output/implementation-artifacts/tech-spec-epic-12-ui-refactor.md#Current app.css token conflict]
- [Source: quantmind-ide/src/app.css — current token state (lines 1–120)]
- [Source: _bmad-output/planning-artifacts/ux-design-directions.html — DIR-2 (Breathing Space), DIR-3 (Ghost Panel), DIR-5 (Balanced Terminal) tile density values]
- [Source: Memory `project_design_direction_prefs.md` — Balanced Terminal is launched default; Breathing Space is Mubarak's personal top pick]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

No blocking issues encountered.

### Completion Notes List

- Fully rewrote `quantmind-ide/src/app.css` — removed all 19 OKLCH-derived tokens (`--bg-*`, `--accent-*`, `--text-*`, `--border-*`, `--syntax-*`) and the legacy `--color-danger` duplicate. Replaced with canonical Frosted Terminal hex/rgba token system.
- Added full canonical token set: background (4), accent (4), text (3), border (2), glass tier (3), spacing scale (9), type scale (7), font stack (4 canonical + 3 legacy aliases), dept accent system (8 overrides), Balanced Terminal density defaults.
- All 4 theme preset blocks (`balanced-terminal`, `ghost-panel`, `open-air`, `breathing-space`) updated to use only Frosted Terminal tokens — no OKLCH, correct `--glass-opacity`, `--tile-min-width`, `--tile-gap`, `--sb-density` per UX spec DIR-2/DIR-3/DIR-5.
- Scrollbar and `html, body` color references updated to canonical tokens.
- Ran bulk `sed` replacement across 228 non-settings Svelte files (all 19 old token patterns replaced). Zero old tokens remain outside settings/.
- Settings sub-panels (all files under `settings/`) were explicitly excluded from token replacement per story mandate.
- Wrote 49-test Vitest suite covering all ACs (AC 12-2-1 through AC 12-2-8). All 49 pass.
- Full regression suite: 271 tests pass, 4 skipped (pre-existing skips), 0 failures.

### Code Review Fixes (2026-03-22)

- **H1 Fixed** — `AgentPanel.test.ts` COLOR_MAP and `subAgentStatusColor` test both referenced old `var(--text-muted)` token instead of canonical `var(--color-text-muted)`. All 3 occurrences corrected. AgentPanel.test.ts: 42 tests pass.
- **M1 Fixed** — `src/lib/styles/components.css` (orphaned, 1137 lines) contained 282 old token usages (`--bg-*`, `--accent-*`, `--text-*`, `--border-*`). All replaced with canonical Frosted Terminal equivalents via bulk `sed`. Zero old tokens remain.
- **M2 Fixed** — `src/lib/styles/global.css` (orphaned, 1389 lines) defined a `:root` block with wrong hex values for all old tokens. Colour definitions replaced to forward to canonical Frosted Terminal tokens (e.g., `--bg-primary: var(--color-bg-base)`). Utility tokens (spacing, radius, shadow, transition, icon sizes, font sizes) preserved unchanged.
- **L1 Fixed** — Added 5 AC 12-2-4 regression tests to `design-tokens.test.ts` that walk all non-settings `.svelte` files and assert zero occurrences of each banned token. CI will now fail if old tokens re-enter components. Total test count: 54.

### File List

- `quantmind-ide/src/app.css` — **modified** (complete token system rewrite)
- `quantmind-ide/src/lib/design-tokens/design-tokens.test.ts` — **modified** (54 Vitest tests; +5 AC 12-2-4 Svelte regression tests added in code review)
- `quantmind-ide/src/lib/components/shell/AgentPanel.test.ts` — **modified** (code review: corrected `var(--text-muted)` → `var(--color-text-muted)` in COLOR_MAP and subAgentStatusColor test)
- `quantmind-ide/src/lib/styles/components.css` — **modified** (code review: replaced all 282 old token usages with canonical Frosted Terminal tokens)
- `quantmind-ide/src/lib/styles/global.css` — **modified** (code review: replaced old colour definitions in :root with Frosted Terminal canonical forwards)
- `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` — modified (token replacement)
- `quantmind-ide/src/lib/components/shared/RichRenderer.svelte` — modified (token replacement)
- `quantmind-ide/src/routes/+page.svelte` — modified (token replacement)
- `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` — modified
- `quantmind-ide/src/lib/components/SettingsView.svelte` — modified
- `quantmind-ide/src/lib/components/StrategyRaw.svelte` — modified
- `quantmind-ide/src/lib/components/SharedResources.svelte` — modified
- `quantmind-ide/src/lib/components/CopilotPanel.svelte` — modified
- `quantmind-ide/src/lib/components/Sidebar.svelte` — modified
- `quantmind-ide/src/lib/components/StatusBar.svelte` — modified
- `quantmind-ide/src/lib/components/ActivityBar.svelte` — modified
- `quantmind-ide/src/lib/components/TopBar.svelte` — modified
- `quantmind-ide/src/lib/components/SessionCard.svelte` — modified
- `quantmind-ide/src/lib/components/BreadcrumbNav.svelte` — modified
- `quantmind-ide/src/lib/components/MemoryPanel.svelte` — modified
- `quantmind-ide/src/lib/components/WorkflowStatusPanel.svelte` — modified
- `quantmind-ide/src/lib/components/ViewHeader.svelte` — modified
- `quantmind-ide/src/lib/components/MarketScannerPanel.svelte` — modified
- `quantmind-ide/src/lib/components/MainContent.svelte` — modified
- `quantmind-ide/src/lib/components/agent-panel/MessagesArea.svelte` — modified
- `quantmind-ide/src/lib/components/skills/SkillCatalogue.svelte` — modified
- `quantmind-ide/src/lib/components/live-trading/CloseAllModal.svelte` — modified
- `quantmind-ide/src/lib/components/live-trading/PositionCloseModal.svelte` — modified
- `quantmind-ide/src/lib/components/kill-switch/EmergencyCloseModal.svelte` — modified
- `quantmind-ide/src/lib/components/kill-switch/KillSwitchModal.svelte` — modified
- `quantmind-ide/src/lib/components/canvas/CanvasPlaceholder.svelte` — modified
- `quantmind-ide/src/lib/components/trading-floor/DepartmentChatPanel.svelte` — modified
- `quantmind-ide/src/lib/components/trading-floor/DepartmentMailPanel.svelte` — modified
- All remaining non-settings Svelte files (228 total) processed with token replacements where applicable
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — modified (status updated)
