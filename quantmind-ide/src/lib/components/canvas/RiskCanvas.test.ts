/**
 * RiskCanvas — Story 12-6 Tests
 *
 * Uses file-content assertion pattern.
 * AC 12-6-1: DeptKanbanTile present
 * AC 12-6-3/4: Sub-page routing with currentSubPage
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'RiskCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

describe('RiskCanvas.svelte — Story 12-6', () => {
  it('does NOT use showDepartmentKanban state', () => {
    expect(srcNoComments).not.toContain('showDepartmentKanban');
  });

  it('does NOT have openDepartmentKanban / closeDepartmentKanban functions', () => {
    expect(srcNoComments).not.toContain('openDepartmentKanban');
    expect(srcNoComments).not.toContain('closeDepartmentKanban');
  });

  it("does NOT have 'kanban' in activeTab union type (removed per Story 12-6)", () => {
    // The activeTab type should NOT contain kanban
    expect(src).not.toMatch(/'kanban'[^;]*activeTab/);
    // Also check the state declaration pattern — should be 4 tabs only
    expect(src).toContain("$state<'physics' | 'compliance' | 'calendar' | 'backtest'>");
  });

  it('uses RiskSubPage union type', () => {
    expect(src).toContain('RiskSubPage');
  });

  it('uses $state<RiskSubPage> for sub-page state', () => {
    expect(src).toContain("$state<RiskSubPage>('grid')");
  });

  it('routes to dept-kanban sub-page', () => {
    expect(src).toContain("currentSubPage === 'dept-kanban'");
  });

  it('DepartmentKanban onClose returns to grid', () => {
    expect(src).toContain("currentSubPage = 'grid'");
  });

  it('has DeptKanbanTile with dept="risk"', () => {
    expect(src).toContain('DeptKanbanTile');
    expect(src).toContain('dept="risk"');
  });

  it('DeptKanbanTile onNavigate sets currentSubPage to dept-kanban', () => {
    expect(src).toContain("currentSubPage = 'dept-kanban'");
  });

  it('does NOT have dept-tasks-btn header button', () => {
    expect(srcNoComments).not.toContain('dept-tasks-btn');
  });

  it('has data-dept="risk" on root element (AC: architecture mandate)', () => {
    expect(src).toContain('data-dept="risk"');
  });

  it('has back button using ArrowLeft icon', () => {
    expect(src).toContain('back-btn');
    expect(src).toContain('ArrowLeft');
  });

  it('does NOT import Kanban from lucide (removed — no longer needed as header button)', () => {
    expect(srcNoComments).not.toMatch(/import.*Kanban.*from.*lucide/);
  });
});
