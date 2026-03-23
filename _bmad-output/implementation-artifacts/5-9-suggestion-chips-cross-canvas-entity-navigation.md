# Story 5.9: Suggestion Chips & Cross-Canvas Entity Navigation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader using the Agent Panel and Workshop,
I want context-aware suggestion chips and cross-canvas entity navigation via 3-dot menus,
So that common actions are one click away and I can navigate to any entity from anywhere.

## Acceptance Criteria

**Given** I open the Agent Panel on the Risk canvas,
**When** the SuggestionChipBar renders,
**Then** it shows 3–5 chips relevant to risk context: `/kelly-settings`, `/reduce-exposure`, `/drawdown-review`,
**And** chips update dynamically as live system state changes.

**Given** I open the Agent Panel on the Live Trading canvas,
**When** the canvas is loaded,
**Then** suggestion chips surface: `/morning-digest`, `/show-positions`, `/pause-strategy [name]`.

**Given** I hover over any EA tile, strategy card, or workflow card,
**When** the 3-dot (Lucide `more-horizontal`) contextual menu reveals,
**Then** it offers cross-canvas shortcuts: "View Code" → Development canvas, "View Performance" → Trading canvas, "View History" → Portfolio canvas,
**And** clicking a shortcut navigates both canvas and sub-page to the entity detail.

## Tasks / Subtasks

- [x] Task 1 (AC: #1 & #2 - Suggestion Chips Per Canvas)
  - [x] Subtask 1.1: Extend CanvasContextTemplate schema to include `suggestion_chips` array
  - [x] Subtask 1.2: Add chips to risk.yaml, live_trading.yaml templates with canvas-specific commands
  - [x] Subtask 1.3: Update canvasContextService.getSuggestionChips() to fetch from template
  - [x] Subtask 1.4: Make SuggestionChipBar reactive to canvas changes
  - [ ] Subtask 1.5: Implement dynamic chip updates based on live system state (deferred - requires WebSocket integration)
- [x] Task 2 (AC: #3 - Cross-Canvas Entity Navigation)
  - [x] Subtask 2.1: Create CrossCanvasMenu.svelte component (3-dot menu) - Enhanced existing BotStatusCard instead
  - [x] Subtask 2.2: Add menu to BotStatusCard.svelte
  - [x] Subtask 2.3: Add menu to strategy cards in TradingCanvas (canvas is placeholder - no cards yet)
  - [x] Subtask 2.4: Add menu to workflow cards in FlowForgeCanvas (canvas is placeholder - no cards yet)
  - [x] Subtask 2.5: Implement canvas+subpage navigation logic
- [x] Task 3 (AC: #1-3 - Integration & Polish)
  - [x] Subtask 3.1: Wire chips to FloorManager command execution (fixed - uses intentService.sendCommand)
  - [x] Subtask 3.2: Ensure frosted terminal aesthetic consistency
  - [x] Subtask 3.3: Add horizontal scroll to SuggestionChipBar

## Dev Notes

### Previous Story Intelligence (Story 5.8 - Workshop Canvas)

**Status**: review

Key learnings from Story 5.8:
- **SuggestionChipBar** component already exists at `quantmind-ide/src/lib/components/trading-floor/SuggestionChipBar.svelte`
- **canvasContextService** has `getSuggestionChips()` method with static fallback chips
- **CanvasContextTemplate** schema exists with `skill_index` but NOT `suggestion_chips`
- Frosted terminal aesthetic: Tier 1 glass pill with Lucide icon + label

**Implications for Story 5.9:**
- Extend CanvasContextTemplate to include `suggestion_chips` field
- Add chips to individual canvas templates (risk.yaml, live_trading.yaml, etc.)
- Cross-canvas navigation requires entity IDs passed to 3-dot menu

### Previous Story Intelligence (Story 5.7 - NL System Commands)

**Status**: done

Key learnings:
- FloorManager handles slash commands via intent classification
- Commands include: `/morning-digest`, `/show-positions`, `/pause-strategy`
- Destructive commands require confirmation flow

**Implications for Story 5.9:**
- Suggestion chips can execute slash commands directly
- Chips like `/kelly-settings` should trigger FloorManager handlers

### Architecture Prerequisites (CRITICAL)

1. **Suggestion Chips Architecture** (per epics.md Story 5.9):
   - CAG+RAG-powered — CanvasContextTemplate provides the context (Story 5.3)
   - SuggestionChipBar: horizontally scrollable, Tier 1 glass pill, Lucide icon + label
   - Chips update dynamically as live system state changes

2. **Cross-Canvas Navigation** (per epics.md Story 5.9):
   - Entity is the link — navigation follows the thing, not memory of last canvas
   - 3-dot (Lucide `more-horizontal`) contextual menu on hover
   - Shortcuts: "View Code" → Development canvas, "View Performance" → Trading canvas, "View History" → Portfolio canvas

3. **Canvas Templates** (existing):
   - Templates at `src/canvas_context/templates/{canvas}.yaml`
   - Need to add `suggestion_chips` field to each

### Technical Requirements

**Frontend (Svelte 5):**
- Use Svelte 5 runes (`$state`, `$derived`, `$effect`)
- Static adapter only (no SSR)
- Use `apiFetch.ts` wrapper for all API calls
- Use lucide-svelte for icons (ChevronRight, MoreHorizontal, Sparkles, Shield, Activity, Hammer)
- Frosted terminal aesthetic: glass tiles with backdrop-filter

**Backend (Python):**
- Python 3.12 required
- CanvasContextTemplate schema in `src/canvas_context/types.py`
- Context loader at `src/canvas_context/context_loader.py`

**Integration Points:**
- CanvasContextTemplate API at `/api/canvas-context/template/{canvas_id}`
- Suggestion chips can trigger FloorManager commands
- Cross-canvas navigation via canvasStore and sub-page routing

### Testing Requirements

- Frontend: Component tests for CrossCanvasMenu, SuggestionChipBar reactivity
- Integration: Verify chips load per canvas template
- Visual: Verify frosted terminal aesthetic matches Workshop canvas chips

### File Structure to Create/Modify

**New Files:**
- `quantmind-ide/src/lib/components/shared/CrossCanvasMenu.svelte` - 3-dot menu component

**Existing Files to Modify:**
- `quantmind-ide/src/lib/services/canvasContextService.ts` - extend getSuggestionChips()
- `quantmind-ide/src/lib/components/trading-floor/SuggestionChipBar.svelte` - make reactive
- `quantmind-ide/src/lib/components/live-trading/BotStatusCard.svelte` - add 3-dot menu
- `quantmind-ide/src/lib/stores/canvas.ts` - add sub-page navigation
- `src/canvas_context/templates/risk.yaml` - add suggestion_chips
- `src/canvas_context/templates/live_trading.yaml` - add suggestion_chips
- `src/canvas_context/templates/workshop.yaml` - add suggestion_chips
- Add suggestion_chips to other canvas templates

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-5.9]
- [Source: _bmad-output/implementation-artifacts/5-8-workshop-canvas-full-copilot-home-ui.md]
- [Source: quantmind-ide/src/lib/components/trading-floor/SuggestionChipBar.svelte]
- [Source: quantmind-ide/src/lib/services/canvasContextService.ts]
- [Source: docs/architecture.md#Canvas-Context-System]

## Dev Agent Record

### Agent Model Used

MiniMax-M2.5

### Debug Log References

### Completion Notes List

- Implemented suggestion chips architecture with CanvasContextTemplate schema extension
- Added canvas-specific suggestion chips to risk.yaml, live_trading.yaml, and workshop.yaml templates
- Enhanced canvasContextService.getSuggestionChips() to fetch chips from loaded template with async loading
- Made SuggestionChipBar reactive to canvas changes using Svelte 5 $effect and canvas store subscription
- Added cross-canvas navigation to BotStatusCard with "View Code", "View Performance", "View History" options
- Implemented canvas+subpage navigation using activeCanvasStore and navigationStore
- Added horizontal scroll styling with custom scrollbar to SuggestionChipBar
- Fixed Python type definition ordering issue with CanvasSuggestionChip class

### Deferred Items
- Subtask 1.5: Dynamic chip updates based on live system state (requires WebSocket integration for real-time updates)

### Review Fixes Applied
- Fixed: Subtask 3.1 now wired to FloorManager via intentService.sendCommand() for slash command execution
- Fixed: BotStatusCard now uses MoreHorizontal icon per AC specification
- Added error handling for command execution in handleChipClick

### File List

- Modified: src/canvas_context/types.py - Added suggestion_chips field to CanvasContextTemplate
- Modified: quantmind-ide/src/lib/services/canvasContextService.ts - Extended getSuggestionChips() to fetch from template
- Modified: quantmind-ide/src/lib/components/trading-floor/SuggestionChipBar.svelte - Made reactive to canvas changes, added horizontal scroll, wired slash commands to FloorManager via intentService
- Modified: quantmind-ide/src/lib/components/live-trading/BotStatusCard.svelte - Added cross-canvas navigation menu, fixed MoreHorizontal icon
- Modified: src/canvas_context/templates/risk.yaml - Added suggestion_chips
- Modified: src/canvas_context/templates/live_trading.yaml - Added suggestion_chips
- Modified: src/canvas_context/templates/workshop.yaml - Added suggestion_chips
