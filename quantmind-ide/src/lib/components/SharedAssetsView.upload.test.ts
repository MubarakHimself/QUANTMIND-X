import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'SharedAssetsView.svelte'), 'utf-8');

describe('SharedAssetsView upload bridge', () => {
  it('offers docs, books, and articles upload categories', () => {
    expect(src).toContain('<option value="Docs">Docs</option>');
    expect(src).toContain('<option value="Books">Books</option>');
    expect(src).toContain('<option value="Articles">Articles</option>');
  });

  it('uses the live assets upload endpoint for file-backed categories', () => {
    expect(src).toContain("buildApiUrl('/assets/upload')");
    expect(src).toContain("isFileUploadCategory(newAsset.category)");
  });
});
