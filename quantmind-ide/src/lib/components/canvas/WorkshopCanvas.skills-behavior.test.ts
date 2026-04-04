import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'WorkshopCanvas.svelte'), 'utf-8');

describe('WorkshopCanvas.svelte — skills behavior', () => {
  it('keeps the user on the skills section when queueing a skill command', () => {
    const match = src.match(/function invokeSkill\(skill: Skill\) \{([\s\S]*?)\n  \}/);
    expect(match?.[1]).toContain('inputMessage =');
    expect(match?.[1]).not.toContain("activeSection = 'chat'");
  });

  it('renders a queued-skill hint with a direct open-chat action', () => {
    expect(src).toContain('Queued in chat draft');
    expect(src).toContain('Open Chat');
  });
});
