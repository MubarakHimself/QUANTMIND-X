import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'AssetList.svelte'), 'utf-8');

describe('AssetList.svelte — article sync controls', () => {
  it('hydrates Firecrawl settings and knowledge sync status for docs/articles', () => {
    expect(src).toContain('fetchFirecrawlSettings');
    expect(src).toContain('fetchKnowledgeSyncStatus');
    expect(src).toContain("selectedType === 'docs' && activeDocGroup === 'articles'");
  });

  it('renders Shared Assets article sync controls', () => {
    expect(src).toContain('Article Sync');
    expect(src).toContain('Sync Now');
    expect(src).toContain('Set API Key');
    expect(src).toContain('Update API Key');
  });

  it('uses real backend knowledge endpoints instead of mock data', () => {
    expect(src).toContain("const KNOWLEDGE_API_BASE = `${API_CONFIG.API_BASE}/knowledge`");
    expect(src).toContain('/firecrawl/settings');
    expect(src).toContain('/sync/status');
    expect(src).toContain('/sync?${params.toString()}');
  });
});
