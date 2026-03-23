/**
 * DevelopmentCanvas — Story 12-6 Tests
 *
 * Uses file-content assertion pattern.
 * AC 12-6-1: DeptKanbanTile present
 * AC 12-6-3/4: Sub-page routing with currentSubPage
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'DevelopmentCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

describe('DevelopmentCanvas.svelte — Story 12-6', () => {
  it('does NOT use showDepartmentKanban boolean flag', () => {
    expect(srcNoComments).not.toContain('showDepartmentKanban');
  });

  it('does NOT have openDepartmentKanban / closeDepartmentKanban functions', () => {
    expect(srcNoComments).not.toContain('openDepartmentKanban');
    expect(srcNoComments).not.toContain('closeDepartmentKanban');
  });

  it('uses DevSubPage union type', () => {
    expect(src).toContain('DevSubPage');
  });

  it('uses $state<DevSubPage> for sub-page state', () => {
    expect(src).toContain("$state<DevSubPage>('grid')");
  });

  it('currentSubPage state starts at grid', () => {
    expect(src).toContain("currentSubPage = $state<DevSubPage>('grid')");
  });

  it('routes to dept-kanban sub-page', () => {
    expect(src).toContain("currentSubPage === 'dept-kanban'");
  });

  it('DepartmentKanban onClose returns to grid', () => {
    expect(src).toContain("currentSubPage = 'grid'");
  });

  it('has DeptKanbanTile with dept="development"', () => {
    expect(src).toContain('DeptKanbanTile');
    expect(src).toContain('dept="development"');
  });

  it('DeptKanbanTile onNavigate sets currentSubPage to dept-kanban', () => {
    expect(src).toContain("currentSubPage = 'dept-kanban'");
  });

  it('does NOT have stale skeleton tile grid', () => {
    expect(srcNoComments).not.toContain('skeleton-tile-grid');
  });

  it('does NOT have dept-tasks-btn header button', () => {
    expect(srcNoComments).not.toContain('dept-tasks-btn');
  });

  it('has data-dept="development" on root element (AC: architecture mandate)', () => {
    expect(src).toContain('data-dept="development"');
  });

  it('has back button for non-grid sub-pages', () => {
    expect(src).toContain('back-btn');
    expect(src).toContain('ArrowLeft');
  });

  it('keeps activeView state for tab navigation within grid', () => {
    expect(src).toContain('activeView');
  });
});
