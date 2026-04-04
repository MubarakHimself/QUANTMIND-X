import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'ResearchCanvas.svelte'), 'utf-8');

describe('ResearchCanvas.svelte — article grouping', () => {
  it('derives grouped article sections by category', () => {
    expect(src).toContain('let articleGroups = $derived(');
    expect(src).toContain('getArticleCategory(article)');
  });

  it('renders article group headers and counts', () => {
    expect(src).toContain('class="article-groups"');
    expect(src).toContain('class="group-header"');
    expect(src).toContain('{group.label}');
    expect(src).toContain('{group.articles.length}');
  });

  it('shows a category chip on article cards', () => {
    expect(src).toContain('class="category-chip"');
    expect(src).toContain('{group.label}</span>');
  });

  it('supports backend category keys when grouping scraped articles', () => {
    expect(src).toContain('article.category');
    expect(src).toContain('article.source_category');
  });

  it('supports live book uploads via shared assets endpoint', () => {
    expect(src).toContain("buildApiUrl('/assets/upload')");
    expect(src).toContain("formData.append('category', 'Books')");
    expect(src).toContain('class="books-upload-btn"');
  });

  it('merges books from shared assets into research books view', () => {
    expect(src).toContain("apiFetch<SharedAssetItem[]>('/assets/shared')");
    expect(src).toContain("asset.category === 'Books'");
  });
});
