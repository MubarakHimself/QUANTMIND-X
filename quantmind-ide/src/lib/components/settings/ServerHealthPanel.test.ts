import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const panelPath = resolve(process.cwd(), 'src/lib/components/settings/ServerHealthPanel.svelte');

describe('ServerHealthPanel disconnected-node rendering', () => {
  it('renders disconnected status detail separately from last heartbeat', () => {
    const src = readFileSync(panelPath, 'utf8');

    expect(src).toContain('status_detail?: string | null;');
    expect(src).toContain("healthData.cloudzy.status === 'disconnected'");
    expect(src).toContain('Status Detail');
    expect(src).toContain('healthData.cloudzy.status_detail');
  });
});
