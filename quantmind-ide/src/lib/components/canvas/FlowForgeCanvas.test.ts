/**
 * FlowForgeCanvas — Story 12-6 Tests
 *
 * Uses file-content assertion pattern.
 * AC 12-6-1: DeptKanbanTile present
 * AC 12-6-3/4: Sub-page routing
 * AC 12-6-8: Prefect Kanban (6-col) and Dept Kanban (4-col) are visually and structurally distinct
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'FlowForgeCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

describe('FlowForgeCanvas.svelte — Story 12-6', () => {
  // Sub-page routing
  it('uses FlowForgeSubPage union type', () => {
    expect(src).toContain('FlowForgeSubPage');
  });

  it("FlowForgeSubPage includes 'prefect' and 'dept-kanban'", () => {
    expect(src).toContain("'prefect' | 'dept-kanban'");
  });

  it("currentSubPage starts at 'prefect'", () => {
    expect(src).toContain("$state<FlowForgeSubPage>('prefect')");
  });

  it('routes to dept-kanban sub-page', () => {
    expect(src).toContain("currentSubPage === 'dept-kanban'");
  });

  // DeptKanbanTile
  it('DeptKanbanTile is imported', () => {
    expect(src).toContain('DeptKanbanTile');
  });

  it('DeptKanbanTile has dept="flowforge"', () => {
    expect(src).toContain('dept="flowforge"');
  });

  it('DeptKanbanTile onNavigate sets currentSubPage to dept-kanban', () => {
    expect(src).toContain("currentSubPage = 'dept-kanban'");
  });

  // DepartmentKanban
  it('DepartmentKanban is imported', () => {
    expect(src).toContain('DepartmentKanban');
  });

  it('DepartmentKanban uses department="flowforge"', () => {
    expect(src).toContain('department="flowforge"');
  });

  it('back button from dept-kanban returns to prefect', () => {
    expect(src).toContain("currentSubPage = 'prefect'");
  });

  it('has canvas-level back button in dept-kanban view (AC 12-6-4)', () => {
    // A canvas-level back button must be shown when in dept-kanban sub-page (not just onClose from DepartmentKanban)
    expect(src).toContain('dept-kanban-header');
    expect(src).toContain('back-btn');
    expect(src).toContain('ArrowLeft');
  });

  // AC 12-6-8: Kill switch modal ONLY in prefect view (Arch-UI-3)
  it('WorkflowKillSwitchModal exists in the canvas', () => {
    expect(src).toContain('WorkflowKillSwitchModal');
  });

  it('WorkflowKillSwitchModal is inside the prefect sub-page block (Arch-UI-3)', () => {
    // Template starts after </script> tag
    const templateStart = src.indexOf('</script>');
    const templateSrc = src.slice(templateStart);
    // Find position of dept-kanban check and kill switch modal use IN TEMPLATE
    const prefectBlockPos = templateSrc.indexOf("currentSubPage === 'dept-kanban'");
    // Find the last occurrence of WorkflowKillSwitchModal (the usage, not the import)
    const killSwitchLastPos = templateSrc.lastIndexOf('WorkflowKillSwitchModal');
    expect(prefectBlockPos).toBeGreaterThan(-1);
    expect(killSwitchLastPos).toBeGreaterThan(-1);
    // Kill switch usage in template should come AFTER the dept-kanban routing block
    // (it's in the {:else} prefect view, not in the dept-kanban view)
    expect(killSwitchLastPos).toBeGreaterThan(prefectBlockPos);
  });

  // AC 12-6-8: AC structural distinction — 6-col Prefect vs 4-col Dept
  it('Prefect Kanban is 6-column board using KANBAN_COLUMNS', () => {
    expect(src).toContain('KANBAN_COLUMNS');
  });

  // data-dept preserved
  it('has data-dept="flowforge" on root element (architecture mandate)', () => {
    expect(src).toContain('data-dept="flowforge"');
  });

  it('DeptKanbanTile positioned in canvas header area (not inside kanban board)', () => {
    // Template starts after </script> tag
    const templateStart = src.indexOf('</script>');
    const templateSrc = src.slice(templateStart);
    // DeptKanbanTile usage (dept="flowforge") should come before kanban-board div in template
    // The DeptKanbanTile is in the header-right, which is before the .kanban-board
    const deptTilePos = templateSrc.indexOf('<DeptKanbanTile dept="flowforge"');
    const kanbanBoardPos = templateSrc.indexOf('class="kanban-board"');
    expect(deptTilePos).toBeGreaterThan(-1);
    expect(kanbanBoardPos).toBeGreaterThan(-1);
    // DeptKanbanTile must come before the main kanban-board
    expect(deptTilePos).toBeLessThan(kanbanBoardPos);
  });
});
