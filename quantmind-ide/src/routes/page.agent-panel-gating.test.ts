import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const src = readFileSync(resolve(__dirname, '+page.svelte'), 'utf-8');

describe('Root page agent-panel gating', () => {
  it('keeps live-trading out of the top-level agent-panel canvases', () => {
    expect(src).toContain("const AGENT_PANEL_CANVASES = new Set(['research', 'development', 'risk', 'trading', 'portfolio', 'flowforge']);");
    expect(src).not.toContain("const AGENT_PANEL_CANVASES = new Set(['live-trading'");
  });

  it('only mounts AgentPanel when the active canvas is agent-enabled', () => {
    expect(src).toContain('{#if showAgentPanel}');
    expect(src).toContain('<AgentPanel');
  });
});
