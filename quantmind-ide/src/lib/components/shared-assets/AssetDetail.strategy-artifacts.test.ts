import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'AssetDetail.svelte'), 'utf-8');

describe('AssetDetail strategy stages', () => {
  it('renders explicit WF1 stage groups beyond source ingest artifacts', () => {
    expect(src).toContain('Research');
    expect(src).toContain('Development');
    expect(src).toContain('Variants');
    expect(src).toContain('Compilation');
    expect(src).toContain('Reports');
  });

  it('reads the stage file arrays from strategy detail payloads', () => {
    expect(src).toContain("strategyDetail.research_files?.length");
    expect(src).toContain("strategyDetail.development_files?.length");
    expect(src).toContain("strategyDetail.variant_files?.length");
    expect(src).toContain("strategyDetail.compilation_files?.length");
    expect(src).toContain("strategyDetail.report_files?.length");
  });
});
