import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'ResearchCanvas.svelte'), 'utf-8');

describe('ResearchCanvas.svelte — news tab wiring', () => {
  it('declares news as a valid research tab', () => {
    expect(src).toContain("| 'news' |");
  });

  it('adds News to the research tab config', () => {
    expect(src).toContain("{ id: 'news',      label: 'News'");
  });

  it('renders NewsView when the news tab is active', () => {
    expect(src).toContain("{:else if activeTab === 'news'}");
    expect(src).toContain('<NewsView />');
  });
});
