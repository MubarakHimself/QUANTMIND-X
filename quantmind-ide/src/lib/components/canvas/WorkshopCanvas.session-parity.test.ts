import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'WorkshopCanvas.svelte'), 'utf-8');

describe('WorkshopCanvas.svelte — session parity', () => {
  it('loads floor-manager sessions and legacy workshop sessions into the recent sidebar', () => {
    expect(src).toContain("chatApi.listSessions(undefined, 'floor-manager')");
    expect(src).toContain("chatApi.listSessions(undefined, 'workshop')");
  });

  it('does not eagerly create an empty session on new chat', () => {
    expect(src).toContain('function startNewChat');
    expect(src).toContain('currentSessionId = null');
    expect(src).not.toContain('const newSession = await chatApi.createSession');
  });

  it('sends workshop messages through the persisted floor-manager chat endpoint', () => {
    expect(src).toContain("chatApi.sendMessage(\n        'floor-manager'");
    expect(src).not.toContain("chatApi.sendMessage(\n        'workshop'");
  });
});
