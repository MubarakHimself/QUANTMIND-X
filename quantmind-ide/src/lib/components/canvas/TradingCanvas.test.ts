/**
 * TradingCanvas — Story 12-6 Tests
 *
 * Uses file-content assertion pattern.
 * Updates Story 12-4 tests to include dept-kanban sub-page.
 * AC 12-6-1: DeptKanbanTile present
 * AC 12-6-3/4: Sub-page routing via existing TradingSubPage union
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'TradingCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

describe('TradingCanvas.svelte — Story 12-6 (extends Story 12-4)', () => {
  // Story 12-4 baseline checks
  it('uses TradingSubPage union type', () => {
    expect(src).toContain('TradingSubPage');
  });

  it('uses $state<TradingSubPage> for sub-page state starting at grid', () => {
    expect(src).toContain("$state<TradingSubPage>('grid')");
  });

  it('uses CanvasTileGrid wrapper', () => {
    expect(src).toContain('CanvasTileGrid');
  });

  // Story 12-6: dept-kanban added to union
  it("'dept-kanban' is in TradingSubPage union type", () => {
    expect(src).toContain("'dept-kanban'");
    // The union should include all 4 options
    expect(src).toContain("'grid' | 'backtest-detail' | 'ea-performance-detail' | 'dept-kanban'");
  });

  it('DeptKanbanTile is imported', () => {
    expect(src).toContain('DeptKanbanTile');
  });

  it('DepartmentKanban is imported', () => {
    expect(src).toContain('DepartmentKanban');
  });

  it('DeptKanbanTile renders in grid view with dept="trading"', () => {
    expect(src).toContain('dept="trading"');
  });

  it('DeptKanbanTile onNavigate sets currentSubPage to dept-kanban', () => {
    expect(src).toContain("currentSubPage = 'dept-kanban'");
  });

  it('routes to dept-kanban sub-page showing DepartmentKanban', () => {
    expect(src).toContain("currentSubPage === 'dept-kanban'");
    expect(src).toContain('department="trading"');
  });

  it('back from dept-kanban sub-page returns to grid', () => {
    expect(src).toContain("currentSubPage = 'grid'");
  });

  it('showBackButton is true when not on grid', () => {
    expect(src).toContain("showBackButton={currentSubPage !== 'grid'}");
  });
});
