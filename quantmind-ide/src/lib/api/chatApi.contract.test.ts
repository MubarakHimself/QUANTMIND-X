import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'chatApi.ts'), 'utf-8');

describe('chatApi.ts — backend contract', () => {
  it('serializes createSession payload in the snake_case shape expected by the FastAPI backend', () => {
    expect(src).toContain('agent_type: data.agentType');
    expect(src).toContain('agent_id: data.agentId');
    expect(src).toContain('user_id: data.userId');
  });

  it('uses PATCH /sessions/{id} with a title payload for chat rename', () => {
    expect(src).toContain("method: 'PATCH'");
    expect(src).toContain("body: JSON.stringify({ title: data.title })");
    expect(src).toContain('updateSessionTitle');
  });
});
