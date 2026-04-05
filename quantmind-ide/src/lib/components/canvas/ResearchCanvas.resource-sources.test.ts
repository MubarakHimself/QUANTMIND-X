import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'ResearchCanvas.svelte'), 'utf-8');

describe('ResearchCanvas.svelte — canonical resource sources', () => {
  it('does not load duplicate research articles locally', () => {
    expect(src).not.toContain("apiFetch<ArticleItem[]>('/knowledge/articles')");
    expect(src).not.toContain("activeTab === 'articles'");
    expect(src).not.toContain('let articleGroups = $derived(');
  });

  it('does not keep local book upload or shared-assets book merge in research', () => {
    expect(src).not.toContain("buildApiUrl('/assets/upload')");
    expect(src).not.toContain("formData.append('category', 'Books')");
    expect(src).not.toContain('class="books-upload-btn"');
    expect(src).not.toContain("apiFetch<SharedAssetItem[]>('/assets/shared')");
    expect(src).not.toContain("asset.category === 'Books'");
  });
});
