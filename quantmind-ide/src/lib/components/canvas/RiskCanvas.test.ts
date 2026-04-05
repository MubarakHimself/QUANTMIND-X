import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'RiskCanvas.svelte'), 'utf-8');

describe('RiskCanvas.svelte', () => {
  it('keeps the canonical risk tabs without a backtest tab', () => {
    expect(src).toContain("type RiskTab = 'physics' | 'compliance' | 'calendar' | 'dept-tasks' | 'dpr';");
    expect(src).not.toContain("id: 'backtest'");
    expect(src).not.toContain("activeTab === 'backtest'");
    expect(src).not.toContain('BacktestResultsPanel');
  });

  it('retains the department kanban surface in dept-tasks', () => {
    expect(src).toContain("id: 'dept-tasks'");
    expect(src).toContain('DepartmentKanban department="risk"');
  });

  it('preserves risk canvas identity for runtime context', () => {
    expect(src).toContain('data-dept="risk"');
    expect(src).toContain("canvasContextService.setRuntimeState('risk'");
  });
});
