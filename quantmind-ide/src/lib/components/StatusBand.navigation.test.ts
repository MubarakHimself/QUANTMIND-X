import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const statusBandPath = resolve(process.cwd(), 'src/lib/components/StatusBand.svelte');
const canvasStorePath = resolve(process.cwd(), 'src/lib/stores/canvasStore.ts');

describe('StatusBand navigation', () => {
  it('routes bots to Trading and workflows to FlowForge', () => {
    const src = readFileSync(statusBandPath, 'utf8');

    expect(src).toContain("function navigateToTrading()");
    expect(src).toContain("activeCanvasStore.setActiveCanvas('trading')");
    expect(src).toContain("function navigateToFlowForge()");
    expect(src).toContain("activeCanvasStore.setActiveCanvas('flowforge')");
    expect(src).toContain('onclick={navigateToTrading}');
    expect(src).toContain('onclick={navigateToFlowForge}');
  });

  it('uses duplicated ticker tracks instead of one long duplicated wrapper block', () => {
    const src = readFileSync(statusBandPath, 'utf8');

    expect(src).toContain('<div class="ticker-viewport">');
    expect(src).toContain('{#each [0, 1] as tickerLoop (tickerLoop)}');
    expect(src).toContain('<div class="ticker-track" aria-hidden={tickerLoop === 1}>');
    expect(src).toContain('.ticker-track {');
  });
});

describe('canvasStore default canvas', () => {
  it('defaults the active canvas to live-trading', () => {
    const src = readFileSync(canvasStorePath, 'utf8');

    expect(src).toContain("writable<string>('live-trading')");
  });
});
