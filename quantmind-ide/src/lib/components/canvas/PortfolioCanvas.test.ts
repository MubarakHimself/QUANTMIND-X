/**
 * Story 12-5 — Portfolio Canvas + Cross-Canvas Navigation Fixes
 *
 * Uses file content assertions (Svelte 5 + @testing-library/svelte incompatibility workaround).
 * Consistent with TradingCanvas.test.ts established pattern in this project.
 *
 * AC 12-5-1/2/3: PortfolioCanvas — local $state sub-page routing, no navigationStore
 * AC 12-5-4:     +page.svelte — no activeView = $state
 * AC 12-5-5:     ActivityBar — no dispatch, uses activeCanvasStore only
 * AC 12-5-7:     StatusBand — navigation functions use activeCanvasStore
 * AC 12-5-6:     canvasStore CANVAS_SHORTCUTS mapping (1-9)
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const CANVAS_DIR = resolve(__dirname);
const COMPONENTS_DIR = resolve(__dirname, '..');
const ROUTES_DIR = resolve(__dirname, '../../../routes');
const STORES_DIR = resolve(__dirname, '../../stores');

function read(dir: string, name: string): string {
  return readFileSync(resolve(dir, name), 'utf-8');
}

// ─── AC 12-5-2: PortfolioCanvas — no navigationStore ──────────────────────────

describe('PortfolioCanvas.svelte — AC 12-5-2: No navigationStore dependency', () => {
  const src = read(CANVAS_DIR, 'PortfolioCanvas.svelte');
  // Strip comments before checking — comments may reference the removed pattern by name
  const noComments = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');

  it('does not import navigationStore', () => {
    expect(noComments).not.toContain('navigationStore');
  });

  it('does not call navigateToView', () => {
    expect(noComments).not.toContain('navigateToView');
  });

  it('contains PortfolioSubPage type declaration', () => {
    expect(src).toContain('PortfolioSubPage');
  });

  it('uses $state<PortfolioSubPage> for sub-page state', () => {
    expect(src).toContain('$state<PortfolioSubPage>');
  });

  it('initialises currentSubPage to grid', () => {
    expect(src).toContain("$state<PortfolioSubPage>('grid')");
  });

  it('has department-kanban as a sub-page value', () => {
    expect(src).toContain("'department-kanban'");
  });

  it('has trading-journal as a sub-page value', () => {
    expect(src).toContain("'trading-journal'");
  });

  it('has routing-matrix as a sub-page value', () => {
    expect(src).toContain("'routing-matrix'");
  });
});

// ─── AC 12-5-1/3: Sub-page routing and back navigation ────────────────────────

describe('PortfolioCanvas.svelte — AC 12-5-1/3: Sub-page routing and back', () => {
  const src = read(CANVAS_DIR, 'PortfolioCanvas.svelte');

  it('renders DepartmentKanban when currentSubPage === department-kanban', () => {
    expect(src).toContain("currentSubPage === 'department-kanban'");
  });

  it('renders TradingJournal when currentSubPage === trading-journal', () => {
    expect(src).toContain("currentSubPage === 'trading-journal'");
  });

  it('renders RoutingMatrix when currentSubPage === routing-matrix', () => {
    expect(src).toContain("currentSubPage === 'routing-matrix'");
  });

  it('DepartmentKanban onClose resets currentSubPage to grid', () => {
    expect(src).toContain("onClose={() => currentSubPage = 'grid'}");
  });

  it('back button present and resets to grid', () => {
    expect(src).toContain("currentSubPage = 'grid'");
  });

  it('back button conditional on currentSubPage !== grid', () => {
    expect(src).toContain("currentSubPage !== 'grid'");
  });

  it('canvas header is at root level — back button guard is NOT inside the {:else} grid block (H1 fix)', () => {
    // The header must appear BEFORE the first {#if currentSubPage === } sub-page block
    // so the back button can actually render when a sub-page is active
    const headerPos = src.indexOf('<header class="canvas-header">');
    const firstSubPagePos = src.indexOf("{#if currentSubPage === 'department-kanban'}");
    expect(headerPos).toBeGreaterThan(-1);
    expect(firstSubPagePos).toBeGreaterThan(-1);
    // Header must come BEFORE the sub-page conditional block
    expect(headerPos).toBeLessThan(firstSubPagePos);
  });
});

// ─── AC 12-5-2: PortfolioCanvas keeps required elements ───────────────────────

describe('PortfolioCanvas.svelte — AC 12-5-9: Required elements preserved', () => {
  const src = read(CANVAS_DIR, 'PortfolioCanvas.svelte');

  it('data-dept="portfolio" on root element (Arch-UI-5)', () => {
    expect(src).toContain('data-dept="portfolio"');
  });

  it('canvasContextService.loadCanvasContext(portfolio) in onMount (Arch-UI-8)', () => {
    expect(src).toContain("loadCanvasContext('portfolio')");
  });

  it('no isLoading={true} skeleton TileCards remain (AC 12-5-1.8)', () => {
    expect(src).not.toContain('isLoading={true}');
  });

  it('no TileCard import from shared (skeleton tiles removed)', () => {
    const noComments = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(noComments).not.toContain("import TileCard");
  });

  it('no kill-switch import (Arch-UI-3)', () => {
    expect(src).not.toMatch(/from.*kill-switch\//);
    expect(src).not.toContain('KillSwitch');
  });

  it('does not use boolean state flags (old pattern removed)', () => {
    expect(src).not.toContain('showDepartmentKanban');
    expect(src).not.toContain('showTradingJournal');
    expect(src).not.toContain('showRoutingMatrix');
  });

  it('does not use open/close function pairs (old pattern removed)', () => {
    expect(src).not.toContain('openDepartmentKanban');
    expect(src).not.toContain('closeDepartmentKanban');
    expect(src).not.toContain('openTradingJournal');
    expect(src).not.toContain('openRoutingMatrix');
  });

  it('under 500 lines — NFR-MAINT-1', () => {
    const lineCount = src.split('\n').length;
    expect(lineCount).toBeLessThanOrEqual(500);
  });
});

// ─── AC 12-5-5: ActivityBar — no dispatch, uses activeCanvasStore ──────────────

describe('ActivityBar.svelte — AC 12-5-5: Single activeCanvasStore dispatch', () => {
  const src = read(COMPONENTS_DIR, 'ActivityBar.svelte');

  it('does not import navigationStore', () => {
    expect(src).not.toContain('navigationStore');
  });

  it('does not import createEventDispatcher', () => {
    expect(src).not.toContain('createEventDispatcher');
  });

  it('does not call dispatch(viewChange)', () => {
    expect(src).not.toContain('dispatch("viewChange"');
    expect(src).not.toContain("dispatch('viewChange'");
  });

  it('calls activeCanvasStore.setActiveCanvas in selectCanvas', () => {
    expect(src).toContain('activeCanvasStore.setActiveCanvas(canvasId)');
  });

  it('does not import run from svelte/legacy', () => {
    expect(src).not.toContain("from 'svelte/legacy'");
  });

  it('does not use $bindable (no more activeView prop)', () => {
    expect(src).not.toContain('$bindable');
  });

  it('derives activeView from $activeCanvasStore (AC 12-5-4)', () => {
    expect(src).toContain('$derived($activeCanvasStore)');
  });

  it('no navigateToView call in ActivityBar', () => {
    expect(src).not.toContain('navigateToView');
  });
});

// ─── AC 12-5-4: +page.svelte — no activeView = $state ─────────────────────────

describe('+page.svelte — AC 12-5-4: No local activeView state', () => {
  const src = read(ROUTES_DIR, '+page.svelte');

  it('no activeView = $state declaration', () => {
    // Must not have the local activeView state variable
    expect(src).not.toMatch(/let\s+activeView\s*=\s*\$state/);
  });

  it('no handleViewChange function', () => {
    expect(src).not.toContain('handleViewChange');
  });

  it('no on:viewChange event binding', () => {
    expect(src).not.toContain('on:viewChange');
  });

  it('no bind:activeView on ActivityBar', () => {
    expect(src).not.toContain('bind:activeView');
  });

  it('currentCanvas derived from $activeCanvasStore (kept for AgentPanel)', () => {
    expect(src).toContain('$derived($activeCanvasStore)');
  });
});

// ─── AC 12-5-7: StatusBand — navigation uses activeCanvasStore ─────────────────

describe('StatusBand.svelte — AC 12-5-7: activeCanvasStore navigation', () => {
  const src = read(COMPONENTS_DIR, 'StatusBand.svelte');

  it('does not import navigationStore', () => {
    expect(src).not.toContain("from '../stores/navigationStore'");
  });

  it('imports activeCanvasStore from canvasStore', () => {
    expect(src).toContain("from '../stores/canvasStore'");
    expect(src).toContain('activeCanvasStore');
  });

  it('navigateToLiveTrading calls setActiveCanvas(live-trading)', () => {
    expect(src).toContain("activeCanvasStore.setActiveCanvas('live-trading')");
  });

  it('navigateToPortfolio calls setActiveCanvas(portfolio)', () => {
    expect(src).toContain("activeCanvasStore.setActiveCanvas('portfolio')");
  });

  it('navigateToRisk calls setActiveCanvas(risk)', () => {
    expect(src).toContain("activeCanvasStore.setActiveCanvas('risk')");
  });

  it('showNodeStatus does NOT call setActiveCanvas (stays as overlay)', () => {
    const showNodeFn = src.match(/function showNodeStatus\(\)[^}]*\}/s)?.[0] ?? '';
    expect(showNodeFn).not.toContain('setActiveCanvas');
  });

  it('session-clocks segment onclick calls navigateToLiveTrading', () => {
    expect(src).toContain('onclick={navigateToLiveTrading}');
  });

  it('bots segment onclick calls navigateToPortfolio', () => {
    expect(src).toContain('onclick={navigateToPortfolio}');
  });

  it('risk segment onclick calls navigateToRisk', () => {
    expect(src).toContain('onclick={navigateToRisk}');
  });

  it('nodes segment onclick calls showNodeStatus (not a canvas nav)', () => {
    expect(src).toContain('onclick={showNodeStatus}');
  });

  it('navigateToRouter function calls setActiveCanvas(risk) — AC 12-5-7 router mode', () => {
    expect(src).toContain('function navigateToRouter');
    // navigateToRouter must call setActiveCanvas('risk')
    const routerFn = src.match(/function navigateToRouter\(\)[^}]*\}/s)?.[0] ?? '';
    expect(routerFn).toContain("setActiveCanvas('risk')");
  });

  it('router segment exists and calls navigateToRouter — AC 12-5-7', () => {
    expect(src).toContain('onclick={navigateToRouter}');
  });

  it('routerMode is rendered in the template — AC 12-5-7', () => {
    expect(src).toContain('{routerMode}');
  });
});

// ─── AC 12-5-6: canvasStore CANVAS_SHORTCUTS mapping ──────────────────────────

describe('canvasStore.ts — AC 12-5-6: All 9 keyboard shortcuts correct', () => {
  const src = read(STORES_DIR, 'canvasStore.ts');

  it('key 1 maps to live-trading', () => {
    expect(src).toContain("'1': 'live-trading'");
  });

  it('key 2 maps to research', () => {
    expect(src).toContain("'2': 'research'");
  });

  it('key 3 maps to development', () => {
    expect(src).toContain("'3': 'development'");
  });

  it('key 4 maps to risk', () => {
    expect(src).toContain("'4': 'risk'");
  });

  it('key 5 maps to trading', () => {
    expect(src).toContain("'5': 'trading'");
  });

  it('key 6 maps to portfolio', () => {
    expect(src).toContain("'6': 'portfolio'");
  });

  it('key 7 maps to shared-assets', () => {
    expect(src).toContain("'7': 'shared-assets'");
  });

  it('key 8 maps to workshop', () => {
    expect(src).toContain("'8': 'workshop'");
  });

  it('key 9 maps to flowforge', () => {
    expect(src).toContain("'9': 'flowforge'");
  });

  it('setActiveCanvas method exists on activeCanvasStore', () => {
    expect(src).toContain('setActiveCanvas');
  });
});

// ─── AC 12-5-8: MainContent — no duplicate keyboard handler ───────────────────

describe('MainContent.svelte — AC 12-5-8: No duplicate keyboard shortcut handler', () => {
  const src = read(COMPONENTS_DIR, 'MainContent.svelte');

  it('no handleKeydown function in MainContent (removed duplicate)', () => {
    // The comment explaining removal may exist, but the function definition must not
    expect(src).not.toMatch(/function handleKeydown\s*\(/);
  });

  it('no window.addEventListener(keydown) in MainContent', () => {
    expect(src).not.toMatch(/window\.addEventListener\s*\(\s*['"]keydown['"]/);
  });
});

// ─── Cross-canvas kill switch compliance — Arch-UI-3 ──────────────────────────

describe('Kill switch compliance — Arch-UI-3 (Story 12-5 files)', () => {
  const files = [
    ['PortfolioCanvas.svelte', CANVAS_DIR],
    ['ActivityBar.svelte', COMPONENTS_DIR],
    ['StatusBand.svelte', COMPONENTS_DIR],
  ] as const;

  for (const [file, dir] of files) {
    it(`${file} has no kill-switch import`, () => {
      const src = read(dir, file);
      expect(src).not.toMatch(/from.*kill-switch\//);
      expect(src).not.toContain('KillSwitch');
    });
  }
});

// ─── Story 12-6: DeptKanbanTile integration in PortfolioCanvas ────────────────

describe('PortfolioCanvas.svelte — Story 12-6: DeptKanbanTile', () => {
  const src = read(CANVAS_DIR, 'PortfolioCanvas.svelte');

  it('DeptKanbanTile is imported', () => {
    expect(src).toContain('DeptKanbanTile');
  });

  it('DeptKanbanTile renders with dept="portfolio" in grid view', () => {
    expect(src).toContain('dept="portfolio"');
  });

  it('DeptKanbanTile onNavigate sets currentSubPage to department-kanban (hyphen, not underscore)', () => {
    expect(src).toContain("currentSubPage = 'department-kanban'");
  });

  it('does NOT change department-kanban to dept-kanban (preserve existing PortfolioSubPage union)', () => {
    expect(src).toContain("'department-kanban'");
    // The existing PortfolioSubPage type must be preserved
    expect(src).toContain('PortfolioSubPage');
  });

  it('DepartmentKanban department prop is portfolio', () => {
    expect(src).toContain('department="portfolio"');
  });

  it('DeptKanbanTile is scoped inside the dashboard tab (not visible on attribution/correlation/performance tabs)', () => {
    // DeptKanbanTile must appear AFTER the {:else} dashboard block, not in attribution/correlation/performance blocks
    const attributionPos = src.indexOf("activeTab === 'attribution'");
    // Use the DeptKanbanTile component tag (not data-dept attribute on root div)
    const tilePos = src.indexOf('<DeptKanbanTile dept="portfolio"');
    // The tile should come after the attribution/correlation/performance tab checks (inside dashboard {:else} block)
    expect(attributionPos).toBeGreaterThan(-1);
    expect(tilePos).toBeGreaterThan(-1);
    expect(tilePos).toBeGreaterThan(attributionPos);
  });

  it('DeptKanbanTile is scoped inside the grid sub-page (not visible in department-kanban/trading-journal/routing-matrix)', () => {
    // The {:else} block for grid sub-page starts after all sub-page checks
    const deptKanbanRoutePos = src.indexOf("currentSubPage === 'department-kanban'");
    // Use the DeptKanbanTile component tag (not data-dept attribute on root div)
    const tilePos = src.indexOf('<DeptKanbanTile dept="portfolio"');
    // Tile must come after the sub-page routing checks (inside {:else} grid view)
    expect(deptKanbanRoutePos).toBeGreaterThan(-1);
    expect(tilePos).toBeGreaterThan(-1);
    expect(tilePos).toBeGreaterThan(deptKanbanRoutePos);
  });
});
