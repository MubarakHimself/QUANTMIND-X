import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'DepartmentKanban.svelte'), 'utf-8');

describe('DepartmentKanban.svelte — task detail pane', () => {
  it('renders a persistent detail panel with provenance fields for the selected task', () => {
    expect(src).toContain('selectedTaskId');
    expect(src).toContain('task-detail-panel');
    expect(src).toContain('selectedTask.description');
    expect(src).toContain('selectedTask.source_dept');
    expect(src).toContain('selectedTask.workflow_id');
    expect(src).toContain('selectedTask.mail_message_id');
    expect(src).toContain('selectedTask.kanban_card_id');
  });
});
