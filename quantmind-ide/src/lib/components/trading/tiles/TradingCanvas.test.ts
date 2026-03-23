/**
 * Story 12-4 — Trading Canvas: Paper Trading & Backtesting Content
 *
 * Uses file content assertions (Svelte 5 + @testing-library/svelte incompatibility workaround).
 * Consistent with tile-grid.test.ts established pattern in this project.
 *
 * AC 12-4-1: No skeleton isLoading TileCards, no epicNumber, no CanvasPlaceholder
 * AC 12-4-2/3: PaperTradingMonitorTile — structure, empty-state text, status-dot colors
 * AC 12-4-4: BacktestResultsTile — structure, financial-value, date formatting
 * AC 12-4-5/6: EAPerformanceTile — section-label, stage counts, color tokens
 * AC 12-4-7: TradingCanvas — sub-page routing state and back button wiring
 * AC 12-4-8: All tiles handle errors as empty state (no throw in catch blocks)
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const TILES_DIR = resolve(__dirname);
const CANVAS_DIR = resolve(__dirname, '../../canvas');

function read(dir: string, name: string): string {
  return readFileSync(resolve(dir, name), 'utf-8');
}

// ─── AC 12-4-1: TradingCanvas.svelte — real tile grid ────────────────────────

describe('TradingCanvas.svelte — AC 12-4-1: Real tile grid, no skeleton', () => {
  const src = read(CANVAS_DIR, 'TradingCanvas.svelte');

  it('no isLoading={true} skeleton TileCards remain', () => {
    expect(src).not.toContain('isLoading={true}');
  });

  it('no epicNumber prop reference exists in template or script', () => {
    // Strip comments before checking — comment docs can mention the removed pattern
    const noComments = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(noComments).not.toContain('epicNumber');
  });

  it('no CanvasPlaceholder import or usage in code', () => {
    // Strip comments — comment docs can mention the removed pattern
    const noComments = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(noComments).not.toContain('CanvasPlaceholder');
  });

  it('uses CanvasTileGrid with dept="trading"', () => {
    expect(src).toContain('dept="trading"');
  });

  it('imports PaperTradingMonitorTile', () => {
    expect(src).toContain('PaperTradingMonitorTile');
  });

  it('imports BacktestResultsTile', () => {
    expect(src).toContain('BacktestResultsTile');
  });

  it('imports EAPerformanceTile', () => {
    expect(src).toContain('EAPerformanceTile');
  });

  it('uses TradingSubPage type with grid | backtest-detail | ea-performance-detail', () => {
    expect(src).toContain('backtest-detail');
    expect(src).toContain('ea-performance-detail');
    expect(src).toContain("'grid'");
  });

  it('wires showBackButton to currentSubPage !== grid', () => {
    expect(src).toContain("currentSubPage !== 'grid'");
  });

  it('onBack resets currentSubPage to grid', () => {
    expect(src).toContain("currentSubPage = 'grid'");
  });

  it('uses $state for currentSubPage (Svelte 5 rune)', () => {
    expect(src).toContain('$state');
  });

  it('does not import kill-switch — Arch-UI-3', () => {
    expect(src).not.toMatch(/from.*kill-switch\//);
  });

  it('does not import GlassTile — wrong tile for this canvas', () => {
    expect(src).not.toContain('GlassTile');
  });
});

// ─── PaperTradingMonitorTile.svelte ───────────────────────────────────────────

describe('PaperTradingMonitorTile.svelte', () => {
  const src = read(TILES_DIR, 'PaperTradingMonitorTile.svelte');

  it('AC 12-4-2: contains empty-state text for no EAs', () => {
    expect(src).toContain('No EAs in paper monitoring phase');
  });

  it('AC 12-4-2: empty state references Alpha Forge in the message', () => {
    expect(src).toContain('Alpha Forge');
  });

  it('imports TileCard from shared/TileCard.svelte', () => {
    expect(src).toContain("from '$lib/components/shared/TileCard.svelte'");
  });

  it('uses size="lg" (spans 2 cols)', () => {
    expect(src).toContain('size="lg"');
  });

  it('imports apiFetch from $lib/api — no raw fetch()', () => {
    expect(src).toContain("from '$lib/api'");
    expect(src).toContain('apiFetch');
  });

  it('AC 12-4-3: renders ea_name in template', () => {
    expect(src).toContain('ea_name');
  });

  it('AC 12-4-3: renders win_rate as a displayed value (required by AC)', () => {
    expect(src).toContain('win_rate');
    // win_rate must appear in template output (not just in interface definition)
    expect(src).toContain('win_rate.toFixed');
  });

  it('AC 12-4-3: financial-value class on numeric values', () => {
    expect(src).toContain('financial-value');
  });

  it('AC 12-4-3: status-dot element present', () => {
    expect(src).toContain('status-dot');
  });

  it('AC 12-4-3: --color-accent-cyan for running status', () => {
    expect(src).toContain('--color-accent-cyan');
  });

  it('AC 12-4-3: --color-accent-amber for paused status', () => {
    expect(src).toContain('--color-accent-amber');
  });

  it('AC 12-4-3: --color-accent-red for failed/other status', () => {
    expect(src).toContain('--color-accent-red');
  });

  it('uses onMount (not $effect) for Svelte lifecycle', () => {
    expect(src).toContain('onMount');
  });

  it('AC 12-4-8: try/catch wraps API call — errors → empty state', () => {
    expect(src).toContain('try {');
    expect(src).toContain('} catch');
  });

  it('uses $state rune (Svelte 5) — not $: reactive', () => {
    expect(src).toContain('$state');
    expect(src).not.toContain('$:');
  });

  it('uses $props rune (Svelte 5) — not export let', () => {
    expect(src).toContain('$props()');
    expect(src).not.toContain('export let');
  });

  it('does not hardcode colors (uses CSS tokens)', () => {
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{6}/);
  });

  it('navigable and onNavigate props wired to TileCard', () => {
    expect(src).toContain('navigable');
    expect(src).toContain('onNavigate');
  });

  it('does not import kill-switch', () => {
    expect(src).not.toContain('kill-switch');
  });
});

// ─── BacktestResultsTile.svelte ───────────────────────────────────────────────

describe('BacktestResultsTile.svelte', () => {
  const src = read(TILES_DIR, 'BacktestResultsTile.svelte');

  it('imports TileCard from shared/TileCard.svelte', () => {
    expect(src).toContain("from '$lib/components/shared/TileCard.svelte'");
  });

  it('uses size="md"', () => {
    expect(src).toContain('size="md"');
  });

  it('calls /backtests?limit=5 (not /backtest/recent)', () => {
    expect(src).toContain('/backtests?limit=5');
    expect(src).not.toContain('/backtest/recent');
  });

  it('AC 12-4-4: empty state text present', () => {
    expect(src).toContain('No backtest results yet');
  });

  it('AC 12-4-4: renders ea_name', () => {
    expect(src).toContain('ea_name');
  });

  it('AC 12-4-4: sharpe value wrapped in financial-value', () => {
    expect(src).toContain('financial-value');
    expect(src).toContain('sharpe');
  });

  it('AC 12-4-4: run_at_utc formatted to local timezone', () => {
    expect(src).toContain('run_at_utc');
    expect(src).toContain('toLocaleDateString');
  });

  it('AC 12-4-4: pass/fail indicator uses --color-accent-cyan for pass', () => {
    expect(src).toContain('--color-accent-cyan');
  });

  it('AC 12-4-4: fail indicator uses --color-accent-red', () => {
    expect(src).toContain('--color-accent-red');
  });

  it('pass/fail uses Lucide icons (CheckCircle/XCircle) — not raw Unicode symbols', () => {
    expect(src).toContain('CheckCircle');
    expect(src).toContain('XCircle');
    expect(src).not.toContain("'✓'");
    expect(src).not.toContain("'✗'");
  });

  it('AC 12-4-8: try/catch wraps API call', () => {
    expect(src).toContain('try {');
    expect(src).toContain('} catch');
  });

  it('uses $state rune (Svelte 5)', () => {
    expect(src).toContain('$state');
    expect(src).not.toContain('$:');
  });

  it('uses $props rune — no export let', () => {
    expect(src).toContain('$props()');
    expect(src).not.toContain('export let');
  });

  it('navigable and onNavigate wired through', () => {
    expect(src).toContain('navigable');
    expect(src).toContain('onNavigate');
  });
});

// ─── EAPerformanceTile.svelte ─────────────────────────────────────────────────

describe('EAPerformanceTile.svelte', () => {
  const src = read(TILES_DIR, 'EAPerformanceTile.svelte');

  it('imports TileCard from shared/TileCard.svelte', () => {
    expect(src).toContain("from '$lib/components/shared/TileCard.svelte'");
  });

  it('uses size="xl" (full width)', () => {
    expect(src).toContain('size="xl"');
  });

  it('calls /pipeline/status (not /pipeline-status/stages)', () => {
    expect(src).toContain('/pipeline/status');
    expect(src).not.toContain('/pipeline-status/');
  });

  it('AC 12-4-5: all-zero state is valid (initializes to 0)', () => {
    expect(src).toContain('backtesting: 0');
    expect(src).toContain('sitGate: 0');
    expect(src).toContain('paperMonitoring: 0');
    expect(src).toContain('awaitingApproval: 0');
  });

  it('AC 12-4-5: section-label class used on stage headings (Fragment Mono 10px caps)', () => {
    expect(src).toContain('section-label');
  });

  it('AC 12-4-6: stage labels contain correct text', () => {
    expect(src).toContain('Backtesting');
    expect(src).toContain('SIT Gate');
    expect(src).toContain('Paper Monitoring');
    expect(src).toContain('Awaiting Approval');
  });

  it('AC 12-4-6: running stage count uses --color-accent-cyan', () => {
    expect(src).toContain('--color-accent-cyan');
  });

  it('AC 12-4-5: zero counts use --color-text-muted', () => {
    expect(src).toContain('--color-text-muted');
  });

  it('maps BACKTEST stage to backtesting bucket', () => {
    expect(src).toContain("'BACKTEST'");
  });

  it('maps VALIDATION stage to SIT Gate bucket', () => {
    expect(src).toContain("'VALIDATION'");
  });

  it('maps EA_LIFECYCLE stage to Paper Monitoring bucket', () => {
    expect(src).toContain("'EA_LIFECYCLE'");
  });

  it('maps APPROVAL stage to Awaiting Approval bucket', () => {
    expect(src).toContain("'APPROVAL'");
  });

  it('AC 12-4-8: try/catch for error → neutral zero state', () => {
    expect(src).toContain('try {');
    expect(src).toContain('} catch');
  });

  it('uses $state rune (Svelte 5)', () => {
    expect(src).toContain('$state');
    expect(src).not.toContain('$:');
  });

  it('financial-value class on count numbers', () => {
    expect(src).toContain('financial-value');
  });
});

// ─── EAPerformanceDetailPage.svelte placeholder ──────────────────────────────

describe('EAPerformanceDetailPage.svelte — AC 12-4-7 placeholder', () => {
  const src = read(TILES_DIR, 'EAPerformanceDetailPage.svelte');

  it('contains EA Performance Detail heading', () => {
    expect(src).toContain('EA Performance Detail');
  });

  it('contains Epic 7/8 reference note', () => {
    expect(src).toContain('Epic 7/8');
  });

  it('has no API calls (placeholder only)', () => {
    expect(src).not.toContain('apiFetch');
    expect(src).not.toContain('fetch(');
    expect(src).not.toContain('onMount');
  });
});

// ─── BacktestDetailPage.svelte placeholder ───────────────────────────────────

describe('BacktestDetailPage.svelte — AC 12-4-7 placeholder', () => {
  const src = read(TILES_DIR, 'BacktestDetailPage.svelte');

  it('contains Backtest Detail heading', () => {
    expect(src).toContain('Backtest Detail');
  });

  it('contains Epic 8 reference note', () => {
    expect(src).toContain('Epic 8');
  });

  it('has no API calls (placeholder only)', () => {
    expect(src).not.toContain('apiFetch');
    expect(src).not.toContain('fetch(');
    expect(src).not.toContain('onMount');
  });
});

// ─── Sub-page routing — AC 12-4-7 ────────────────────────────────────────────

describe('TradingCanvas sub-page routing — AC 12-4-7', () => {
  const src = read(CANVAS_DIR, 'TradingCanvas.svelte');

  it('imports EAPerformanceDetailPage sub-page', () => {
    expect(src).toContain('EAPerformanceDetailPage');
  });

  it('imports BacktestDetailPage sub-page', () => {
    expect(src).toContain('BacktestDetailPage');
  });

  it('navigates PaperTradingMonitorTile to ea-performance-detail', () => {
    expect(src).toContain("ea-performance-detail");
  });

  it('navigates BacktestResultsTile to backtest-detail', () => {
    expect(src).toContain("backtest-detail");
  });

  it('onBack callback resets to grid sub-page', () => {
    expect(src).toContain("currentSubPage = 'grid'");
  });

  it('showBackButton is false when on grid sub-page', () => {
    expect(src).toContain("currentSubPage !== 'grid'");
  });
});

// ─── Kill switch compliance — Arch-UI-3 ──────────────────────────────────────

describe('Kill switch compliance — Arch-UI-3', () => {
  const files = [
    ['TradingCanvas.svelte', CANVAS_DIR],
    ['PaperTradingMonitorTile.svelte', TILES_DIR],
    ['BacktestResultsTile.svelte', TILES_DIR],
    ['EAPerformanceTile.svelte', TILES_DIR],
    ['EAPerformanceDetailPage.svelte', TILES_DIR],
    ['BacktestDetailPage.svelte', TILES_DIR],
  ] as const;

  for (const [file, dir] of files) {
    it(`${file} has no kill-switch import`, () => {
      const src = read(dir, file);
      expect(src).not.toMatch(/from.*kill-switch\//);
      expect(src).not.toContain('KillSwitch');
    });
  }
});

// ─── No GlassTile anti-pattern ────────────────────────────────────────────────

describe('Anti-pattern: No GlassTile usage', () => {
  const files = [
    ['TradingCanvas.svelte', CANVAS_DIR],
    ['PaperTradingMonitorTile.svelte', TILES_DIR],
    ['BacktestResultsTile.svelte', TILES_DIR],
    ['EAPerformanceTile.svelte', TILES_DIR],
  ] as const;

  for (const [file, dir] of files) {
    it(`${file} does not import GlassTile`, () => {
      const src = read(dir, file);
      expect(src).not.toContain('GlassTile');
    });
  }
});

// ─── File size compliance — NFR-MAINT-1 ──────────────────────────────────────

describe('File size compliance — NFR-MAINT-1 (max 500 lines per component)', () => {
  const files = [
    ['TradingCanvas.svelte', CANVAS_DIR],
    ['PaperTradingMonitorTile.svelte', TILES_DIR],
    ['BacktestResultsTile.svelte', TILES_DIR],
    ['EAPerformanceTile.svelte', TILES_DIR],
    ['EAPerformanceDetailPage.svelte', TILES_DIR],
    ['BacktestDetailPage.svelte', TILES_DIR],
  ] as const;

  for (const [file, dir] of files) {
    it(`${file} is under 500 lines`, () => {
      const src = read(dir, file);
      const lineCount = src.split('\n').length;
      expect(lineCount).toBeLessThanOrEqual(500);
    });
  }
});
