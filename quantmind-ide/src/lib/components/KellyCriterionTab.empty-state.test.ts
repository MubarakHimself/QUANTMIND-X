import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'KellyCriterionTab.svelte'), 'utf-8');

describe('KellyCriterionTab.svelte — empty live data handling', () => {
  it('guards against empty Kelly datasets and renders honest unavailable states', () => {
    expect(src).toContain('kellyEntries.length > 0 ?');
    expect(src).toContain('statusMessage');
    expect(src).toContain('No per-bot Kelly rankings are available from the live backend.');
    expect(src).toContain('No Kelly adjustment history is available.');
  });
});
