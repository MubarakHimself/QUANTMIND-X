import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'sharedAssetsApi.ts'), 'utf-8');

describe('sharedAssetsApi data contract', () => {
  it('uses backend assets endpoints without mock-data fallback', () => {
    expect(src).toContain("await apiFetch<Record<string, any>[]>('/assets/shared')");
    expect(src).toContain("const legacy = await apiFetch<Record<string, any>[]>('/assets')");
    expect(src).toContain("return apiFetch<Record<AssetType, number>>('/assets/counts')");
    expect(src).not.toContain('MOCK_ASSETS');
  });

  it('maps both category and type tokens when grouping assets', () => {
    expect(src).toContain('function getAssetCategoryToken');
    expect(src).toContain('asset.category ?? asset.type ?? asset.asset_type');
    expect(src).toContain('categoryMatchesType(type, getAssetCategoryToken(asset))');
  });

  it('treats strategies as canonical roots instead of merging generic shared-file scans', () => {
    expect(src).toContain("if (type === 'strategies')");
    expect(src).toContain('return Array.from(byId.values())');
  });
});
