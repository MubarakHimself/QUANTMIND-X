import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const componentPath = resolve(__dirname, 'AgentPanel.svelte');

describe('AgentPanel context attachment', () => {
  it('loads compact chat context payloads and avoids sending full visible canvas text', () => {
    const src = readFileSync(componentPath, 'utf8');

    expect(src).toContain('buildCanvasAttachmentContract(');
    expect(src).toContain('buildResourceAttachmentContext(attachment.resource)');
    expect(src).toContain('workspace_resource_hints');
    expect(src).toContain("strategy: 'manifest-first'");
    expect(src).not.toContain('template.base_descriptor');
    expect(src).not.toContain('visible_canvas_text');
    expect(src).not.toContain('Attached canvas context:');
  });

  it('scopes automatic resource hints to the active canvas unless the canvas is an orchestrator', () => {
    const src = readFileSync(componentPath, 'utf8');

    expect(src).toContain("const GLOBAL_CONTEXT_CANVASES = new Set(['workshop', 'flowforge']);");
    expect(src).toContain('const searchCanvases = getNaturalSearchCanvases(activeCanvas);');
    expect(src).toContain("|| resource.canvas === 'shared-assets'");
  });

  it('uses the shared attachment browser helper to render hierarchical attach menus', () => {
    const src = readFileSync(componentPath, 'utf8');

    expect(src).toContain("from '$lib/services/attachmentBrowser'");
    expect(src).toContain('buildAttachmentResourceGroups(');
    expect(src).toContain('attachMenuGroupPath');
    expect(src).toContain('attachMenuCanvasId');
    expect(src).not.toContain('attachMenuGroupId = $state');
    expect(src).toContain('Attach full');
    expect(src).not.toContain('Attach a visible file or tile');
  });
});
