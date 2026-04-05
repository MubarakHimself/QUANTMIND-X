import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'DevelopmentCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

describe('DevelopmentCanvas.svelte', () => {
  it('marks the root canvas as development', () => {
    expect(src).toContain('data-dept="development"');
  });

  it('keeps development tabs focused on department work, not FlowForge pipeline duplication', () => {
    expect(src).toContain("label: 'Active EAs'");
    expect(src).toContain("label: 'Variants'");
    expect(src).toContain("label: 'Backtest'");
    expect(src).toContain("label: 'Workflows'");
    expect(src).toContain("label: 'Dept Tasks'");
    expect(src).not.toContain("label: 'Pipeline'");
    expect(src).not.toContain("id: 'development:pipeline-board'");
    expect(src).not.toContain('<PipelineBoard />');
  });

  it('exposes the department kanban only through the dedicated dept-tasks tab', () => {
    const deptTaskResourcesSection = src.slice(
      src.indexOf("if (activeTab === 'dept-tasks')"),
      src.indexOf('return baseResources;'),
    );
    expect(deptTaskResourcesSection).toContain("id: 'development:department-kanban'");
    expect(deptTaskResourcesSection).toContain("label: 'Department Task Board'");

    const deptTaskRenderSection = src.slice(
      src.indexOf("{#if activeTab === 'dept-tasks'}"),
      src.indexOf("{:else if activeTab === 'variants'}"),
    );
    expect(deptTaskRenderSection).toContain('DepartmentKanban department="development"');
  });

  it('exposes canonical knowledge books as development reference-book resources', () => {
    expect(src).toContain("resource_type: 'reference-book'");
    expect(src).toContain("source: 'knowledge/books'");
    expect(src).toContain('reference_book_count: referenceBooks.length');
  });

  it('does not keep stale pipeline-specific attachable resources or render branches', () => {
    expect(srcNoComments).not.toContain("if (activeTab === 'pipeline')");
    expect(srcNoComments).not.toContain("{:else if activeTab === 'pipeline'}");
    expect(srcNoComments).not.toContain('pipeline-wrapper');
    expect(srcNoComments).not.toContain('pipeline-board-section');
  });
});
