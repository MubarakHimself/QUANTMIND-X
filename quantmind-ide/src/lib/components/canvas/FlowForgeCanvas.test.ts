/**
 * FlowForgeCanvas layout contract tests
 *
 * Uses file-content assertions because most canvas tests in this repo avoid
 * mounting Svelte 5 components directly.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'FlowForgeCanvas.svelte'), 'utf-8');
const template = src.slice(src.indexOf('</script>'));

describe('FlowForgeCanvas.svelte', () => {
  it('uses activeTab state for the two sub-pages', () => {
    expect(src).toContain("type FlowForgeSubPage = 'prefect' | 'dept-kanban'");
    expect(src).toContain("let activeTab = $state<FlowForgeSubPage>('prefect')");
    expect(template).toContain("activeTab === 'dept-kanban'");
  });

  it('renders the department kanban for flowforge', () => {
    expect(src).toContain('DepartmentKanban');
    expect(src).toContain('department="flowforge"');
    expect(src).toContain("onClose={() => activeTab = 'prefect'}");
  });

  it('keeps the canvas-level back button for the dept-kanban view', () => {
    expect(src).toContain('dept-kanban-header');
    expect(src).toContain('back-btn');
    expect(src).toContain('ArrowLeft');
    expect(src).toContain("onclick={() => activeTab = 'prefect'}");
  });

  it('keeps the kill-switch modal inside the prefect/workflows view', () => {
    const deptViewPos = template.indexOf("activeTab === 'dept-kanban'");
    const killSwitchPos = template.lastIndexOf('<WorkflowKillSwitchModal');
    expect(deptViewPos).toBeGreaterThan(-1);
    expect(killSwitchPos).toBeGreaterThan(deptViewPos);
  });

  it('uses the tab strip for workflows and department tasks navigation', () => {
    expect(template).toContain('class="tab-nav"');
    expect(template).toContain('<span>Workflows</span>');
    expect(template).toContain('<span>Dept Tasks</span>');
    expect(template).toContain("onclick={() => activeTab = 'dept-kanban'}");
  });

  it('does not render the old DeptKanbanTile header summary', () => {
    expect(src).not.toContain('DeptKanbanTile');
    expect(template).not.toContain('<DeptKanbanTile');
  });

  it('does not render the old standalone refresh button in the header', () => {
    expect(src).not.toContain('class="refresh-btn"');
    expect(src).not.toContain('handleRefresh()');
    expect(template).not.toContain('<span>Refresh</span>');
  });

  it('keeps the FlowForge identity header compact', () => {
    expect(template).toContain('class="canvas-header"');
    expect(template).toContain('<h2>FlowForge</h2>');
    expect(template).toContain('Global Workflow Orchestration &amp; Floor Manager');
  });

  it('keeps the root element tagged for flowforge canvas context', () => {
    expect(template).toContain('data-dept="flowforge"');
  });
});
