/**
 * WorkshopCanvas — Epic 12 Tests
 *
 * Uses file-content assertion pattern (Svelte 5 + @testing-library/svelte
 * incompatibility workaround — consistent with all other canvas tests in this project).
 *
 * Story 12-3: AC 12-3-12 — corrected Lucide icons (MessageSquare, GitBranch, Brain, Zap)
 * Story 12-3: AC 12-3-10 — data-dept="workshop"
 * Story 12-6: No DeptKanbanTile (Workshop maps to FloorManager — no dept head kanban)
 * Story 12-6: No showDepartmentKanban boolean flag
 *
 * WorkshopCanvas is the full Copilot Home UI. It does NOT follow the
 * DeptKanbanTile/DepartmentKanban pattern because Workshop = FloorManager
 * (not a department head) and has its own sidebar + chat architecture.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'WorkshopCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

// ─── Architecture mandates ────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Architecture mandates (AC 12-3-10, Arch-UI-3)', () => {
  it('has data-dept="workshop" on root element (Arch-UI-5 / AC 12-3-10)', () => {
    expect(src).toContain('data-dept="workshop"');
  });

  it('does not import from kill-switch/ directory (Arch-UI-3)', () => {
    // Must not import from kill-switch/ directory (Trading Kill Switch module)
    // Note: copilotKillSwitchService is from services/ — that is allowed and expected
    expect(src).not.toMatch(/from.*kill-switch\//);
    // Must not import the trading KillSwitch component from components/kill-switch/
    expect(src).not.toMatch(/from.*['"].*kill-switch\/KillSwitch/);
  });

  it('does not reference CanvasPlaceholder (AC 12-3-1)', () => {
    expect(srcNoComments).not.toContain('CanvasPlaceholder');
  });

  it('does not contain raw "Coming Soon" text (AC 12-3-1)', () => {
    expect(src).not.toContain('Coming Soon');
  });

  it('has no emoji characters in template — memory feedback_icons_not_emoji', () => {
    expect(src).not.toMatch(/[\u{1F300}-\u{1FFFF}]/u);
  });
});

// ─── Story 12-3: Corrected Lucide icon imports (AC 12-3-12) ──────────────

describe('WorkshopCanvas.svelte — Corrected icon imports (AC 12-3-12)', () => {
  it('imports MessageSquare (was Clock — History sidebar item)', () => {
    expect(src).toContain('MessageSquare');
  });

  it('imports GitBranch (was FolderOpen — Projects sidebar item)', () => {
    expect(src).toContain('GitBranch');
  });

  it('imports Brain (was Database — Memory sidebar item)', () => {
    expect(src).toContain('Brain');
  });

  it('imports Zap (was Sparkles — Skills sidebar item)', () => {
    expect(src).toContain('Zap');
  });

  it('does NOT import Clock as a named identifier (was wrong History icon)', () => {
    // Strip comments first — inline comment might mention Clock for documentation
    const stripComments = (code: string) =>
      code.replace(/\/\/.*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    const importBlock =
      stripComments(src).match(
        /import\s*\{[^}]+\}\s*from\s*['"]lucide-svelte['"]/s
      )?.[0] ?? '';
    expect(importBlock).not.toContain('Clock');
  });

  it('does NOT import FolderOpen (was wrong Projects icon)', () => {
    const stripComments = (code: string) =>
      code.replace(/\/\/.*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    const importBlock =
      stripComments(src).match(
        /import\s*\{[^}]+\}\s*from\s*['"]lucide-svelte['"]/s
      )?.[0] ?? '';
    expect(importBlock).not.toContain('FolderOpen');
  });

  it('does NOT import Database (was wrong Memory icon)', () => {
    const stripComments = (code: string) =>
      code.replace(/\/\/.*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    const importBlock =
      stripComments(src).match(
        /import\s*\{[^}]+\}\s*from\s*['"]lucide-svelte['"]/s
      )?.[0] ?? '';
    expect(importBlock).not.toContain('Database');
  });

  it('does NOT use <Sparkles in template (was wrong Skills icon)', () => {
    expect(src).not.toContain('<Sparkles');
  });

  it('imports Plus (New Chat button)', () => {
    expect(src).toContain('Plus');
  });

  it('imports Send (send message button)', () => {
    expect(src).toContain('Send');
  });

  it('imports Bot (assistant avatar)', () => {
    expect(src).toContain('Bot');
  });

  it('imports User (user message avatar)', () => {
    expect(src).toContain('User');
  });
});

// ─── Svelte 5 runes pattern ───────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Svelte 5 runes (NFR-MAINT-2)', () => {
  it('uses $state rune (not writable())', () => {
    expect(src).toContain('$state');
  });

  it('uses $state for messages array', () => {
    expect(src).toContain('messages = $state<Message[]>([])');
  });

  it('uses $state for inputMessage', () => {
    expect(src).toContain("inputMessage = $state('')");
  });

  it('uses $state for isLoading', () => {
    expect(src).toContain('isLoading = $state(false)');
  });

  it('uses $state for currentSessionId', () => {
    expect(src).toContain('currentSessionId = $state<string | null>(null)');
  });

  it('uses $state for showSidebar', () => {
    expect(src).toContain('showSidebar = $state(true)');
  });

  it('uses $state for activeSection', () => {
    expect(src).toContain("activeSection = $state<'chat' | 'projects' | 'memory' | 'skills'>('chat')");
  });

  it('does NOT use $: reactive declarations (Svelte 4 legacy)', () => {
    expect(src).not.toContain('$:');
  });

  it('does NOT use export let (Svelte 4 props pattern)', () => {
    expect(srcNoComments).not.toContain('export let');
  });
});

// ─── Story 12-6: No DeptKanban pattern (expected — Workshop has no dept head) ──

describe('WorkshopCanvas.svelte — No DeptKanbanTile (AC 12-6 scope: Workshop excluded)', () => {
  it('does NOT import DeptKanbanTile (Workshop = FloorManager, not a dept head)', () => {
    expect(srcNoComments).not.toContain('DeptKanbanTile');
  });

  it('does NOT have showDepartmentKanban state (Workshop has its own nav pattern)', () => {
    expect(srcNoComments).not.toContain('showDepartmentKanban');
  });

  it('does NOT have openDepartmentKanban / closeDepartmentKanban functions', () => {
    expect(srcNoComments).not.toContain('openDepartmentKanban');
    expect(srcNoComments).not.toContain('closeDepartmentKanban');
  });
});

// ─── Lifecycle ────────────────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Lifecycle', () => {
  it('imports onMount and onDestroy', () => {
    expect(src).toContain('onMount');
    expect(src).toContain('onDestroy');
  });

  it('calls loadSessions() in onMount', () => {
    expect(src).toContain('loadSessions()');
  });

  it('calls loadSkills() in onMount', () => {
    expect(src).toContain('loadSkills()');
  });

  it('calls canvasContextService.loadCanvasContext(workshop)', () => {
    expect(src).toContain("canvasContextService.loadCanvasContext('workshop')");
  });

  it('calls copilotKillSwitchService.getStatus() in sendMessage before sending', () => {
    expect(src).toContain('copilotKillSwitchService.getStatus()');
  });
});

// ─── Send message logic ───────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — sendMessage logic', () => {
  it('sendMessage function exists', () => {
    expect(src).toContain('async function sendMessage');
  });

  it('checks copilot kill switch before sending (AC 5-6)', () => {
    expect(src).toContain('ks.active');
  });

  it('does not send empty message (checks trim)', () => {
    expect(src).toContain('text.trim()');
  });

  it('adds user message to messages array', () => {
    expect(src).toContain("role: 'user'");
  });

  it('adds assistant message for streaming', () => {
    expect(src).toContain("role: 'assistant'");
  });

  it('clears inputMessage to empty string after send', () => {
    expect(src).toContain("inputMessage = ''");
  });

  it('sends workshop messages through chatApi session persistence', () => {
    expect(src).toContain("chatApi.sendMessage('workshop'");
    expect(src).not.toContain('/floor-manager/chat');
  });

  it('sets isLoading to false in finally block', () => {
    expect(src).toContain('finally {');
    expect(src).toContain('isLoading = false');
  });
});

// ─── Keyboard handler ─────────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Keyboard handler', () => {
  it('handleKeyDown function exists', () => {
    expect(src).toContain('function handleKeyDown');
  });

  it('Enter key (without Shift) triggers sendMessage', () => {
    expect(src).toContain("e.key === 'Enter'");
    expect(src).toContain('!e.shiftKey');
    expect(src).toContain('e.preventDefault()');
  });
});

// ─── Sidebar sections ─────────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Sidebar sections', () => {
  it('sidebarItems includes history section', () => {
    expect(src).toContain("'history'");
  });

  it('sidebarItems includes projects section', () => {
    expect(src).toContain("'projects'");
  });

  it('sidebarItems includes memory section', () => {
    expect(src).toContain("'memory'");
  });

  it('sidebarItems includes skills section', () => {
    expect(src).toContain("'skills'");
  });

  it('showSidebar toggle button exists', () => {
    expect(src).toContain('showSidebar');
  });

  it('loads memory nodes when memory section activated', () => {
    expect(src).toContain("item.id === 'memory'");
    expect(src).toContain('loadMemoryNodes()');
  });
});

// ─── Memory filter logic ──────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Memory filter logic', () => {
  it('memoryFilter state supports hot/warm/all values', () => {
    expect(src).toContain("'hot'");
    expect(src).toContain("'warm'");
    expect(src).toContain("'all'");
  });

  it('loadMemoryNodes calls getHotNodes when filter is hot', () => {
    expect(src).toContain("memoryFilter === 'hot'");
    expect(src).toContain('getHotNodes');
  });

  it('loadMemoryNodes calls getWarmNodes when filter is warm', () => {
    expect(src).toContain("memoryFilter === 'warm'");
    expect(src).toContain('getWarmNodes');
  });

  it('loads both hot and warm when filter is all', () => {
    // When filter is 'all', loads both hot(25) and warm(25)
    expect(src).toContain('getHotNodes(25)');
    expect(src).toContain('getWarmNodes(25)');
  });

  it('expandedNodeId tracks which memory node is expanded', () => {
    expect(src).toContain('expandedNodeId');
    expect(src).toContain('toggleNodeExpansion');
  });
});

// ─── Skills ───────────────────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Skills section', () => {
  it('listSkills is imported and called', () => {
    expect(src).toContain('listSkills');
  });

  it('invokeSkill sets inputMessage to slash_command', () => {
    expect(src).toContain('skill.slash_command');
    expect(src).toContain('inputMessage = skill.slash_command');
  });

  it('skillsLoading state controls loading indicator', () => {
    expect(src).toContain('skillsLoading');
  });
});

// ─── Session management ───────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Session management', () => {
  it('startNewChat creates a persisted floor-manager session before first send', () => {
    expect(src).toContain('function startNewChat');
    expect(src).toContain('messages = []');
    expect(src).toContain('await chatApi.createSession');
    expect(src).toContain("agentType: 'floor-manager'");
  });

  it('selectSession sets currentSessionId', () => {
    expect(src).toContain('function selectSession');
    expect(src).toContain('currentSessionId = session.id');
  });

  it('deleteSession filters sessions array', () => {
    expect(src).toContain('function deleteSession');
    expect(src).toContain('sessions.filter');
  });

  it('deleteSession calls stopPropagation to prevent session selection', () => {
    expect(src).toContain('e.stopPropagation()');
  });

  it('loads both floor-manager and legacy workshop sessions into the recent sidebar', () => {
    expect(src).toContain("chatApi.listSessions(undefined, 'floor-manager')");
    expect(src).toContain("chatApi.listSessions(undefined, 'workshop')");
  });

  it('renders a clear-history control that deletes all recent sessions', () => {
    expect(src).toContain('deleteAllSessions');
    expect(src).toContain('Clear history');
  });
});

// ─── Imports from API modules ─────────────────────────────────────────────

describe('WorkshopCanvas.svelte — API module imports', () => {
  it('imports chatApi from $lib/api/chatApi', () => {
    expect(src).toContain('chatApi');
  });

  it('imports listSkills from $lib/api/skillsApi', () => {
    expect(src).toContain('skillsApi');
  });

  it('imports getHotNodes / getWarmNodes from $lib/api/graphMemory', () => {
    expect(src).toContain('graphMemory');
  });

  it('imports API_CONFIG for API base URL', () => {
    expect(src).toContain('API_CONFIG');
  });

  it('imports copilotKillSwitchService', () => {
    expect(src).toContain('copilotKillSwitchService');
  });
});

// ─── Audit rendering (Story 10.4 support) ────────────────────────────────

describe('WorkshopCanvas.svelte — Audit message rendering (Story 10.4)', () => {
  it('renders audit_timeline message type with dedicated class', () => {
    expect(src).toContain("messageType === 'audit_timeline'");
    expect(src).toContain('audit-timeline');
  });

  it('renders audit_reasoning message type with dedicated class', () => {
    expect(src).toContain("messageType === 'audit_reasoning'");
    expect(src).toContain('audit-reasoning');
  });
});

// ─── Suggestion chips ─────────────────────────────────────────────────────

describe('WorkshopCanvas.svelte — Suggestion chips in welcome state', () => {
  it('has suggestion-chips container in welcome state', () => {
    expect(src).toContain('suggestion-chips');
  });

  it('includes /morning-digest suggestion chip', () => {
    expect(src).toContain('/morning-digest');
  });

  it('includes audit query suggestion chips (Story 10.4)', () => {
    expect(src).toContain('Why was EA_GBPUSD paused yesterday?');
  });
});

// ─── File size — NFR-MAINT-1 ──────────────────────────────────────────────

describe('WorkshopCanvas.svelte — File size (NFR-MAINT-1)', () => {
  it('does not exceed 1200 lines (Workshop is intentionally large — complex Copilot UI)', () => {
    // Workshop is the most complex canvas — 1200 lines is the relaxed limit
    const lineCount = src.split('\n').length;
    expect(lineCount).toBeLessThanOrEqual(1200);
  });
});
