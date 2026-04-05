import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'VideoIngestWorkflow.svelte'), 'utf-8');

describe('VideoIngestWorkflow strategy handoff', () => {
  it('imports the shared-assets and canvas stores for strategy navigation', () => {
    expect(src).toContain("import { activeCanvasStore } from '$lib/stores/canvasStore';");
    expect(src).toContain("import { sharedAssetsStore } from '$lib/stores/sharedAssets';");
  });

  it('provides a handler that routes a job strategy into Shared Assets', () => {
    expect(src).toContain('async function openStrategyAsset(job: VideoIngestJob)');
    expect(src).toContain("await sharedAssetsStore.fetchAssetsByType('strategies');");
    expect(src).toContain("activeCanvasStore.setActiveCanvas('shared-assets');");
  });

  it('renders an Open Strategy action when a job has a canonical strategy asset id', () => {
    expect(src).toContain("{#if job.strategy_asset_id}");
    expect(src).toContain('Open Strategy');
    expect(src).toContain('onclick={() => openStrategyAsset(job)}');
  });
});
