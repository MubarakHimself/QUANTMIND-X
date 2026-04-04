import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const src = readFileSync(resolve(__dirname, 'AgentPanel.svelte'), 'utf-8');

describe('AgentPanel session management', () => {
  it('provides per-session delete controls in the department panel history', () => {
    expect(src).toContain('function deleteSession');
    expect(src).toContain('chatApi.deleteSession');
    expect(src).toContain('title="Delete"');
  });

  it('provides per-session rename controls in the department panel history', () => {
    expect(src).toContain('function beginRenameSession');
    expect(src).toContain('function saveSessionRename');
    expect(src).toContain('chatApi.updateSessionTitle');
    expect(src).toContain('title="Rename"');
  });

  it('supports deleting all sessions from department panel history', () => {
    expect(src).toContain('function deleteAllSessions');
    expect(src).toContain('Promise.all(backendSessionIds.map((id) => chatApi.deleteSession(id)))');
    expect(src).toContain('Clear history');
  });
});
