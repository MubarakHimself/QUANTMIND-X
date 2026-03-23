/**
 * SharedAssetsCanvas — Story 12-6 Tests
 *
 * Uses file-content assertion pattern.
 * AC 12-6-1: DeptKanbanTile present in grid view
 * AC 12-6-3/4: Sub-page routing with currentSubPage (content | dept-kanban)
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'SharedAssetsCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

describe('SharedAssetsCanvas.svelte — Story 12-6', () => {
  // Sub-page layer
  it('uses SharedAssetsSubPage union type', () => {
    expect(src).toContain('SharedAssetsSubPage');
  });

  it("SharedAssetsSubPage includes 'content' and 'dept-kanban'", () => {
    expect(src).toContain("'content' | 'dept-kanban'");
  });

  it('uses $state<SharedAssetsSubPage> starting at content', () => {
    expect(src).toContain("$state<SharedAssetsSubPage>('content')");
  });

  it('routes to dept-kanban sub-page', () => {
    expect(src).toContain("currentSubPage === 'dept-kanban'");
  });

  it('DepartmentKanban is imported', () => {
    expect(src).toContain('DepartmentKanban');
  });

  it('DepartmentKanban uses department="shared-assets"', () => {
    expect(src).toContain('department="shared-assets"');
  });

  it('DepartmentKanban onClose returns to content', () => {
    expect(src).toContain("currentSubPage = 'content'");
  });

  // DeptKanbanTile
  it('DeptKanbanTile is imported', () => {
    expect(src).toContain('DeptKanbanTile');
  });

  it('DeptKanbanTile has dept="shared-assets"', () => {
    expect(src).toContain('dept="shared-assets"');
  });

  it('DeptKanbanTile onNavigate sets currentSubPage to dept-kanban', () => {
    expect(src).toContain("currentSubPage = 'dept-kanban'");
  });

  it('DeptKanbanTile only visible when currentView === grid', () => {
    // DeptKanbanTile should be inside the {#if currentView === 'grid'} block
    const gridBlockStart = src.indexOf("currentView === 'grid'");
    const tilePos = src.indexOf('dept="shared-assets"');
    expect(gridBlockStart).toBeGreaterThan(-1);
    expect(tilePos).toBeGreaterThan(gridBlockStart);
  });

  // Back button
  it('has back button shown when currentSubPage !== content', () => {
    expect(src).toContain("currentSubPage !== 'content'");
    expect(src).toContain('back-btn');
    expect(src).toContain('ArrowLeft');
  });

  // data-dept preserved
  it('has data-dept="shared" on root element (architecture mandate)', () => {
    expect(src).toContain('data-dept="shared"');
  });

  it('existing ViewState (grid | list | detail) asset browsing is preserved', () => {
    expect(src).toContain('ViewState');
    expect(src).toContain("'grid' | 'list' | 'detail'");
  });
});
