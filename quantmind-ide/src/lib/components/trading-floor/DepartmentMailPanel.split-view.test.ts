import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'DepartmentMailPanel.svelte'), 'utf-8');

describe('DepartmentMailPanel.svelte — split mail workspace', () => {
  it('keeps the inbox list visible while rendering a detail pane for the selected message', () => {
    expect(src).toContain('message-workspace');
    expect(src).toContain('message-list-panel');
    expect(src).toContain('message-detail-panel');
    expect(src).toContain('item-preview');
    expect(src).toContain('No message selected');
  });
});
