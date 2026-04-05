/**
 * SharedAssetsCanvas contract tests
 *
 * Shared Assets is a library surface only. It should not expose
 * department-kanban/task tiles.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'SharedAssetsCanvas.svelte'), 'utf-8');

describe('SharedAssetsCanvas.svelte', () => {
  it('preserves shared-assets root identity', () => {
    expect(src).toContain('data-dept="shared"');
  });

  it('keeps asset browsing view states only', () => {
    expect(src).toContain('ViewState');
    expect(src).toContain("'grid' | 'list' | 'detail'");
  });

  it('bootstraps tile counts without preloading every asset payload on mount', () => {
    expect(src).toContain('fetchAssetCounts()');
    expect(src).not.toContain('fetchAssets();');
  });

  it('rehydrates into detail view when a selected shared asset already exists', () => {
    expect(src).toContain("if (asset && currentView !== 'detail')");
    expect(src).toContain("currentView = 'detail'");
  });

  it('rehydrates into list view when a shared-assets type is preselected', () => {
    expect(src).toContain("if (!asset && selectedType && currentView === 'grid')");
    expect(src).toContain("currentView = 'list'");
  });

  it('tracks nested list path segments from AssetList for breadcrumb rendering', () => {
    expect(src).toContain('let nestedPathSegments = $state<string[]>([])');
    expect(src).toContain('onPathChange={(segments) => { nestedPathSegments = segments; }}');
    expect(src).toContain('{#each nestedPathSegments as segment}');
  });

  it('does not import or render DepartmentKanban or DeptKanbanTile', () => {
    expect(src).not.toContain('DepartmentKanban');
    expect(src).not.toContain('DeptKanbanTile');
    expect(src).not.toContain('dept-kanban');
    expect(src).not.toContain('DEPT TASKS');
  });
});
