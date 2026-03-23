/**
 * DeptKanbanTile — Story 12-6 Tests
 *
 * Uses file-content assertion pattern (Svelte 5 + @testing-library/svelte incompatibility workaround).
 * AC 12-6-1: Tile shows active/blocked/done counts
 * AC 12-6-6: Empty state — neutral, not error
 * AC 12-6-7: BLOCKED count uses --color-accent-amber
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(
  resolve(__dirname, 'DeptKanbanTile.svelte'),
  'utf-8'
);
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

describe('DeptKanbanTile.svelte — Story 12-6', () => {
  // Task structure
  it('uses Svelte 5 $props()', () => {
    expect(src).toContain('$props()');
  });

  it('uses Svelte 5 $state for activeCount, blockedCount, doneCount', () => {
    expect(src).toContain('activeCount = $state(0)');
    expect(src).toContain('blockedCount = $state(0)');
    expect(src).toContain('doneCount = $state(0)');
  });

  it('uses onMount for data fetching', () => {
    expect(src).toContain('onMount');
  });

  it('uses apiFetch — NOT raw fetch (architecture mandate)', () => {
    expect(src).toContain('apiFetch');
    // Should not use raw fetch() directly in this component
    expect(srcNoComments).not.toMatch(/await fetch\(/);
  });

  it('imports apiFetch from $lib/api', () => {
    expect(src).toContain("from '$lib/api'");
  });

  // API endpoint path — apiFetch prepends /api internally, so pass /tasks/{dept}
  it('calls apiFetch with /tasks/${dept} path (apiFetch prepends /api automatically)', () => {
    // Correct: apiFetch('/tasks/${dept}') → resolves to /api/tasks/{dept} at runtime
    expect(src).toContain('`/tasks/${dept}`');
    // Incorrect would be: apiFetch('/api/tasks/${dept}') which would double the /api prefix
    expect(src).not.toContain('`/api/tasks/${dept}`');
  });

  it('wraps with TileCard', () => {
    expect(src).toContain('TileCard');
  });

  it('imports Kanban from lucide-svelte (no emoji)', () => {
    expect(src).toContain('Kanban');
    expect(src).toContain('lucide-svelte');
  });

  // AC 12-6-1: Task count data points
  it('renders active count', () => {
    expect(src).toContain('activeCount');
    expect(src).toContain('active');
  });

  it('renders blocked count', () => {
    expect(src).toContain('blockedCount');
    expect(src).toContain('blocked');
  });

  it('renders done count', () => {
    expect(src).toContain('doneCount');
    expect(src).toContain('done');
  });

  // AC 12-6-6: Empty state — neutral, not error
  it('shows empty state message when all counts are zero', () => {
    expect(src).toContain('No active tasks — dept head is idle');
  });

  it('empty state checks all three counts are zero', () => {
    expect(src).toContain('activeCount === 0 && blockedCount === 0 && doneCount === 0');
  });

  it('empty state class is neutral (empty-state) — not error', () => {
    expect(src).toContain('class="empty-state"');
    expect(srcNoComments).not.toContain('class="error');
  });

  // AC 12-6-7: BLOCKED count uses --color-accent-amber
  it('blocked count uses --color-accent-amber CSS custom property', () => {
    expect(src).toContain('--color-accent-amber');
  });

  it('blocked count CSS class is distinct', () => {
    expect(src).toContain('count.blocked');
  });

  // Navigation
  it('passes onNavigate to TileCard', () => {
    expect(src).toContain('onNavigate');
  });

  it('computes navigable prop from onNavigate existence', () => {
    expect(src).toContain('!!onNavigate');
  });

  // Counts computed correctly from task status
  it('counts TODO + IN_PROGRESS as active', () => {
    expect(src).toContain("t.status === 'TODO' || t.status === 'IN_PROGRESS'");
  });

  it('counts BLOCKED as blocked', () => {
    expect(src).toContain("t.status === 'BLOCKED'");
  });

  it('counts DONE as done', () => {
    expect(src).toContain("t.status === 'DONE'");
  });

  // Types
  it('imports DepartmentTasksResponse type', () => {
    expect(src).toContain('DepartmentTasksResponse');
  });

  // AC 12-6-5: SSE for real-time tile count updates
  it('opens an SSE EventSource for real-time updates (AC 12-6-5)', () => {
    expect(src).toContain('EventSource');
    expect(src).toContain('/api/sse/tasks/${dept}');
  });

  it('SSE uses full /api/sse/tasks/ path (not via apiFetch — EventSource requires direct URL)', () => {
    // EventSource uses direct URL with /api prefix — it does NOT use apiFetch
    expect(src).toContain('new EventSource(`/api/sse/tasks/${dept}`)');
  });

  it('SSE error handler closes connection without crashing tile (AC 12-6-6)', () => {
    expect(src).toContain('eventSource.onerror');
    expect(src).toContain('eventSource?.close()');
  });
});
