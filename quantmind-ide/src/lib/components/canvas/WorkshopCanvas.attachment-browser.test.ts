import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'WorkshopCanvas.svelte'), 'utf-8');

describe('WorkshopCanvas attachment browser', () => {
  it('uses the shared attachment browser helper for hierarchical attach browsing', () => {
    expect(src).toContain("from '$lib/services/attachmentBrowser'");
    expect(src).toContain('buildAttachmentResourceGroups(');
    expect(src).toContain('attachMenuGroupPath');
    expect(src).toContain('attachMenuCanvasId');
    expect(src).not.toContain('attachMenuGroupId = $state');
    expect(src).toContain('Attach full');
    expect(src).not.toContain('Attach a visible file or tile');
  });
});
