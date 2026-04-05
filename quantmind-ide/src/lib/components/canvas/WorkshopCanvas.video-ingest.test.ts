import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'WorkshopCanvas.svelte'), 'utf-8');

describe('WorkshopCanvas.svelte — video ingest integration', () => {
  it('imports the durable VideoIngestWorkflow surface into the active workshop canvas', () => {
    expect(src).toContain("import VideoIngestWorkflow from '$lib/components/VideoIngestWorkflow.svelte';");
  });

  it('exposes a sidebar entry for video ingest navigation', () => {
    expect(src).toContain("class:active={activeSection === 'video-ingest'}");
    expect(src).toContain("onclick={() => navigateTo('video-ingest')}");
    expect(src).toContain('<span class=\"nav-label\">Video Ingest</span>');
  });

  it('renders VideoIngestWorkflow when the active section is video-ingest', () => {
    expect(src).toContain("{:else if activeSection === 'video-ingest'}");
    expect(src).toContain('<VideoIngestWorkflow />');
  });
});
