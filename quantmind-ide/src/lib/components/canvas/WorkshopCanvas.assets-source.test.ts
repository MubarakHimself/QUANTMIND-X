import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'WorkshopCanvas.svelte'), 'utf-8');

describe('WorkshopCanvas assets source', () => {
  it('uses shared-assets API instead of legacy /api/projects endpoint', () => {
    expect(src).toContain("import { listAllAssets } from '$lib/api/sharedAssetsApi';");
    expect(src).toContain('const grouped = await listAllAssets();');
    expect(src).not.toContain('`${API_BASE}/projects`');
  });

  it('labels the sidebar and section as Shared Assets', () => {
    expect(src).toContain('Shared Assets');
  });

  it('does not render a workshop-local settings button in the sidebar footer', () => {
    expect(src).not.toContain('<span class="nav-label">Settings</span>');
    expect(src).not.toContain('settingsTrigger.update');
  });
});
