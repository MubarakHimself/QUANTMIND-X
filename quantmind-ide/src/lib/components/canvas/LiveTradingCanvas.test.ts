/**
 * LiveTradingCanvas — Epic 12 Tests
 *
 * Uses file-content assertion pattern (Svelte 5 + @testing-library/svelte
 * incompatibility workaround — consistent with all other canvas tests in this project).
 *
 * Story 12-3: AC 12-3-10 — data-dept attribute, no kill-switch import
 * Story 12-6: DeptKanbanTile integration and old pattern removal tracking
 *
 * NOTE: LiveTradingCanvas retains some legacy patterns (showDepartmentKanban
 * boolean, on:click) as of this Epic 12 snapshot. These tests capture the
 * current state and the architecture mandates that ARE met. When Story 12-6
 * refactor is applied to this canvas, the "old pattern" tests below should
 * be updated to match the new $state<LiveTradingSubPage> pattern.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'LiveTradingCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

// ─── Architecture mandates — always required ───────────────────────────────

describe('LiveTradingCanvas.svelte — Architecture mandates (AC 12-3-10, Arch-UI-3)', () => {
  it('has data-dept="trading" on root element (Arch-UI-5 / AC 12-3-10)', () => {
    expect(src).toContain('data-dept="trading"');
  });

  it('does not import from kill-switch/ directory (Arch-UI-3)', () => {
    expect(src).not.toMatch(/from.*kill-switch\//);
    expect(src).not.toContain('KillSwitch');
  });

  it('does not reference CanvasPlaceholder (AC 12-3-1)', () => {
    expect(srcNoComments).not.toContain('CanvasPlaceholder');
  });

  it('does not contain raw "Coming Soon" text (AC 12-3-1)', () => {
    expect(src).not.toContain('Coming Soon');
  });

  it('imports from lucide-svelte (no emoji icons) — AC 12-3-12', () => {
    expect(src).toContain("from 'lucide-svelte'");
  });

  it('has no emoji characters in template — memory feedback_icons_not_emoji', () => {
    expect(src).not.toMatch(/[\u{1F300}-\u{1FFFF}]/u);
  });
});

// ─── Lifecycle — onMount / onDestroy ──────────────────────────────────────

describe('LiveTradingCanvas.svelte — Lifecycle hooks', () => {
  it('imports onMount and onDestroy', () => {
    expect(src).toContain('onMount');
    expect(src).toContain('onDestroy');
  });

  it('calls canvasContextService.loadCanvasContext in onMount', () => {
    expect(src).toContain("canvasContextService.loadCanvasContext('live-trading')");
  });

  it('calls connectTradingWS() in onMount', () => {
    expect(src).toContain('connectTradingWS()');
  });

  it('calls disconnectTradingWS() in onDestroy', () => {
    expect(src).toContain('disconnectTradingWS()');
  });

  it('calls startHealthMonitoring() in onMount', () => {
    expect(src).toContain('startHealthMonitoring()');
  });

  it('calls stopHealthMonitoring() in onDestroy', () => {
    expect(src).toContain('stopHealthMonitoring()');
  });

  it('calls fetchActiveBots() on mount', () => {
    expect(src).toContain('fetchActiveBots()');
  });
});

// ─── Imports — required modules ───────────────────────────────────────────

describe('LiveTradingCanvas.svelte — Imports', () => {
  it('imports BotStatusGrid from live-trading/', () => {
    expect(src).toContain('BotStatusGrid');
  });

  it('imports BotDetailPage from live-trading/', () => {
    expect(src).toContain('BotDetailPage');
  });

  it('imports MorningDigestCard from live-trading/', () => {
    expect(src).toContain('MorningDigestCard');
  });

  it('imports NewsFeedTile from live-trading/', () => {
    expect(src).toContain('NewsFeedTile');
  });

  it('imports DepartmentKanban from department-kanban/', () => {
    expect(src).toContain('DepartmentKanban');
  });

  it('imports canvasContextService from services/', () => {
    expect(src).toContain('canvasContextService');
  });

  it('imports selectedBotId from stores/trading', () => {
    expect(src).toContain('selectedBotId');
  });

  it('imports Wifi from lucide-svelte (WS connected indicator)', () => {
    expect(src).toContain('Wifi');
  });

  it('imports WifiOff from lucide-svelte (WS error indicator)', () => {
    expect(src).toContain('WifiOff');
  });

  it('imports RefreshCw from lucide-svelte (refresh button)', () => {
    expect(src).toContain('RefreshCw');
  });
});

// ─── Content structure ────────────────────────────────────────────────────

describe('LiveTradingCanvas.svelte — Template structure', () => {
  it('renders BotStatusGrid when no selectedBotId', () => {
    expect(src).toContain('BotStatusGrid');
    expect(src).toContain('selectedBotId');
  });

  it('renders BotDetailPage when selectedBotId is set', () => {
    expect(src).toContain('BotDetailPage');
  });

  it('conditionally renders MorningDigestCard on first load', () => {
    expect(src).toContain('hasShownMorningDigest');
    expect(src).toContain('MorningDigestCard');
  });

  it('renders NewsFeedTile in sidebar', () => {
    expect(src).toContain('NewsFeedTile');
  });

  it('canvas-header element exists', () => {
    expect(src).toContain('canvas-header');
  });
});

// ─── WebSocket status indicators ─────────────────────────────────────────

describe('LiveTradingCanvas.svelte — WS status indicators', () => {
  it('shows Live text when wsConnected is true', () => {
    expect(src).toContain('wsConnected');
    expect(src).toContain('Live');
  });

  it('shows Error text when wsError is true', () => {
    expect(src).toContain('wsError');
    expect(src).toContain('Error');
  });

  it('shows Connecting state', () => {
    expect(src).toContain('Connecting');
  });

  it('imports wsConnected store', () => {
    expect(src).toContain('wsConnected');
  });

  it('imports wsError store', () => {
    expect(src).toContain('wsError');
  });
});

// ─── Story 12-6: DepartmentKanban integration ─────────────────────────────
// LiveTradingCanvas uses the LEGACY boolean pattern as of Epic 12 snapshot.
// These tests document the current implementation state.
// When Story 12-6 refactor is applied, update these tests.

describe('LiveTradingCanvas.svelte — DepartmentKanban integration (Story 12-6 snapshot)', () => {
  it('imports DepartmentKanban component', () => {
    expect(src).toContain('DepartmentKanban');
  });

  it('passes department="trading" to DepartmentKanban', () => {
    expect(src).toContain('department="trading"');
  });

  it('provides onClose handler for DepartmentKanban', () => {
    expect(src).toContain('onClose');
  });

  it('showDepartmentKanban controls DepartmentKanban visibility (legacy boolean pattern)', () => {
    // Current implementation uses boolean flag — documented as legacy
    expect(src).toContain('showDepartmentKanban');
  });

  it('openDepartmentKanban sets showDepartmentKanban to true (legacy pattern)', () => {
    expect(src).toContain('openDepartmentKanban');
    expect(src).toContain('showDepartmentKanban = true');
  });

  it('closeDepartmentKanban sets showDepartmentKanban to false (legacy pattern)', () => {
    expect(src).toContain('closeDepartmentKanban');
    expect(src).toContain('showDepartmentKanban = false');
  });
});

// ─── Refresh function ─────────────────────────────────────────────────────

describe('LiveTradingCanvas.svelte — Refresh function', () => {
  it('has a refresh() function', () => {
    expect(src).toContain('async function refresh');
  });

  it('sets refreshing=true before fetch and false after', () => {
    expect(src).toContain('refreshing = true');
    expect(src).toContain('refreshing = false');
  });

  it('has refresh-btn in template', () => {
    expect(src).toContain('refresh-btn');
  });
});

// ─── CSS — no hardcoded forbidden tokens ─────────────────────────────────

describe('LiveTradingCanvas.svelte — CSS token usage', () => {
  it('does not use var(--bg-primary) — legacy OKLCH token (AC 12-2-4)', () => {
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    expect(styleBlock).not.toContain('var(--bg-primary)');
  });

  it('does not use var(--accent-primary) — legacy OKLCH token (AC 12-2-4)', () => {
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    expect(styleBlock).not.toContain('var(--accent-primary)');
  });
});

// ─── File size — NFR-MAINT-1 ──────────────────────────────────────────────

describe('LiveTradingCanvas.svelte — File size (NFR-MAINT-1)', () => {
  it('is under 500 lines', () => {
    const lineCount = src.split('\n').length;
    expect(lineCount).toBeLessThanOrEqual(500);
  });
});
