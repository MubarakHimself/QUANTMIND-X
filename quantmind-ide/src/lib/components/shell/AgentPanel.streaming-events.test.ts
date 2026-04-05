import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'AgentPanel.svelte'), 'utf-8');

describe('AgentPanel streaming event contract', () => {
  it('upserts one canonical stream-event row for status and tool SSE updates', () => {
    expect(src).toContain('streamEventMessageId');
    expect(src).toContain('upsertStreamingToolMessage(');
    expect(src).not.toContain("id: crypto.randomUUID(),\n                  type: 'tool'");
  });
});
