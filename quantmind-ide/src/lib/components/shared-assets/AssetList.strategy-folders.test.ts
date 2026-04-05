import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'AssetList.svelte'), 'utf-8');

describe('AssetList strategy folders', () => {
  it('groups strategies by family and source bucket before listing roots', () => {
    expect(src).toContain('activeStrategyFamily');
    expect(src).toContain('activeStrategyBucket');
    expect(src).toContain('getStrategyFamily');
    expect(src).toContain('getStrategyBucket');
    expect(src).toContain("selectedType === 'strategies' && !activeStrategyFamily");
    expect(src).toContain("selectedType === 'strategies' && activeStrategyFamily && !activeStrategyBucket");
    expect(src).toContain('Open {formatCategoryLabel(family).toLowerCase()} strategies');
    expect(src).toContain('Open {formatStrategyBucketLabel(bucket).toLowerCase()} roots');
  });

  it('emits nested path segments and uses specific list titles while drilling into strategy folders', () => {
    expect(src).toContain('onPathChange?: (segments: string[]) => void;');
    expect(src).toContain("if (selectedType === 'strategies')");
    expect(src).toContain('if (activeStrategyFamily) segments.push(formatCategoryLabel(activeStrategyFamily));');
    expect(src).toContain('if (activeStrategyBucket) segments.push(formatStrategyBucketLabel(activeStrategyBucket));');
    expect(src).toContain('onPathChange?.(segments);');
    expect(src).toContain('if (activeStrategyBucket) return formatStrategyBucketLabel(activeStrategyBucket);');
    expect(src).toContain('if (activeStrategyFamily) return formatCategoryLabel(activeStrategyFamily);');
    expect(src).toContain('{getListTitle()}');
  });
});
