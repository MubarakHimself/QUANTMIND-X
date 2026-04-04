import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'WorkshopCanvas.svelte'), 'utf-8');

describe('WorkshopCanvas.svelte — session parity', () => {
  it('loads floor-manager sessions and legacy workshop sessions into the recent sidebar', () => {
    expect(src).toContain("chatApi.listSessions(undefined, 'floor-manager')");
    expect(src).toContain("chatApi.listSessions(undefined, 'workshop')");
  });

  it('creates a persisted session immediately on new chat so recent history updates before first send', () => {
    expect(src).toContain('function startNewChat');
    expect(src).toContain('await chatApi.createSession');
    expect(src).toContain("agentType: 'floor-manager'");
  });

  it('sends workshop messages through the persisted floor-manager chat endpoint', () => {
    expect(src).toContain("chatApi.sendMessage(\n        'floor-manager'");
    expect(src).not.toContain("chatApi.sendMessage(\n        'workshop'");
  });

  it('renders a clear-history control for deleting all recent workshop sessions', () => {
    expect(src).toContain('Clear history');
    expect(src).toContain('deleteAllSessions');
    expect(src).toContain('const allSessions = await listWorkshopSessions()');
  });

  it('supports per-session rename in the recent sidebar', () => {
    expect(src).toContain('function startRenameSession');
    expect(src).toContain('function commitRenameSession');
    expect(src).toContain('chatApi.updateSessionTitle');
    expect(src).toContain('title="Rename"');
  });

  it('does not render a synthetic placeholder session after history is cleared', () => {
    expect(src).not.toContain("Today's session");
    expect(src).toContain('No recent chats');
  });
});
