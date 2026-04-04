import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const componentPath = resolve(
  process.cwd(),
  'src/lib/components/trading-floor/CopilotPanel.svelte'
);

describe('CopilotPanel session parity', () => {
  it('uses the session-backed floor-manager chat endpoint', () => {
    const src = readFileSync(componentPath, 'utf8');

    expect(src).toContain('const apiEndpoint = `${API_BASE}/chat/floor-manager/message`;');
    expect(src).not.toContain('/copilot/chat');
    expect(src).toContain('session_id: currentSessionId ?? undefined');
  });

  it('hydrates saved sessions from the shared chat session store', () => {
    const src = readFileSync(componentPath, 'utf8');

    expect(src).toContain('const stored: StoredChatMessage[] = await chatApi.getSessionMessages(sessionId);');
    expect(src).toContain("role: msg.role === 'assistant' ? 'floor_manager'");
  });

  it('supports bulk history deletion from the copilot history sidebar', () => {
    const src = readFileSync(componentPath, 'utf8');

    expect(src).toContain('function clearAllSessions');
    expect(src).toContain('Clear history');
    expect(src).toContain('Promise.all(sessionIds.map((sessionId) => chatApi.deleteSession(sessionId)))');
  });

  it('creates timestamped sessions for immediate recent-list visibility', () => {
    const src = readFileSync(componentPath, 'utf8');

    expect(src).toContain('title: `Chat ${now.toLocaleString()}`');
    expect(src).toContain('applySessions([newSession, ...sessions.filter((session) => session.id !== newSession.id)])');
  });

  it('supports per-session rename in history groups', () => {
    const src = readFileSync(componentPath, 'utf8');

    expect(src).toContain('function startSessionRename');
    expect(src).toContain('function saveSessionRename');
    expect(src).toContain('chatApi.updateSessionTitle');
    expect(src).toContain('title="Rename"');
  });
});
