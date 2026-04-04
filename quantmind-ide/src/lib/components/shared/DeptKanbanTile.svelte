<script lang="ts">
  /**
   * DeptKanbanTile — Summary tile for department task counts.
   * Story 12-6: Department Kanban Sub-Page (All Canvases)
   *
   * AC 12-6-1: Shows active/blocked/done task counts per department
   * AC 12-6-6: Empty state — neutral, not error
   * AC 12-6-7: BLOCKED count uses --color-accent-amber
   */
  import { onMount } from 'svelte';
  import { apiFetch } from '$lib/api';
  import TileCard from './TileCard.svelte';
  import { Kanban } from 'lucide-svelte';
  import type { DepartmentTasksResponse } from '$lib/components/department-kanban/types';
  import { API_CONFIG } from '$lib/config/api';

  interface Props {
    dept: string;
    onNavigate?: () => void;
  }

  let { dept, onNavigate }: Props = $props();

  let activeCount = $state(0);
  let blockedCount = $state(0);
  let doneCount = $state(0);
  let isLoading = $state(true);

  // SSE for real-time tile count updates (AC 12-6-5)
  let eventSource: EventSource | null = null;

  function getTasksApiUrl(path: string): string {
    return `${API_CONFIG.API_URL}${path}`;
  }

  function parseCounts(tasks: DepartmentTasksResponse['tasks']) {
    activeCount = tasks.filter(t => t.status === 'TODO' || t.status === 'IN_PROGRESS').length;
    blockedCount = tasks.filter(t => t.status === 'BLOCKED').length;
    doneCount = tasks.filter(t => t.status === 'DONE').length;
  }

  onMount(async () => {
    // NOTE: apiFetch prepends /api internally — pass /tasks/{dept}, NOT /api/tasks/{dept}
    try {
      const data = await apiFetch<DepartmentTasksResponse>(`/tasks/${dept}`);
      parseCounts(data.tasks || []);
    } catch {
      // Keep counts at 0 on error — empty state will render (AC 12-6-6)
    } finally {
      isLoading = false;
    }

    // SSE for real-time task count updates (AC 12-6-5)
    try {
      eventSource = new EventSource(getTasksApiUrl(`/api/sse/tasks/${dept}`));
      eventSource.onopen = () => {
        void apiFetch<DepartmentTasksResponse>(`/tasks/${dept}`)
          .then((data) => parseCounts(data.tasks || []))
          .catch(() => {
            // Keep last known counts if refresh fails
          });
      };
      eventSource.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          // Re-fetch full task list on any SSE event to keep counts accurate
          if (update && update.tasks) {
            parseCounts(update.tasks);
          }
        } catch {
          // Ignore malformed SSE data
        }
      };
      eventSource.onerror = () => {
        // SSE error is non-fatal — tile shows last known counts
        eventSource?.close();
      };
    } catch {
      // SSE not available — tile shows static counts fetched on mount
    }

    return () => {
      eventSource?.close();
    };
  });
</script>

<TileCard title="Dept Tasks" navigable={!!onNavigate} {onNavigate} {isLoading}>
  <div class="tile-icon">
    <Kanban size={14} />
  </div>

  {#if !isLoading && activeCount === 0 && blockedCount === 0 && doneCount === 0}
    <!-- AC 12-6-6: Empty state — neutral, no error styling -->
    <p class="empty-state">No active tasks — dept head is idle</p>
  {:else if !isLoading}
    <!-- AC 12-6-1: 3 data points on tile face -->
    <div class="task-counts">
      <span class="count active">{activeCount} active</span>
      <span class="count blocked">{blockedCount} blocked</span>
      <span class="count done">{doneCount} done</span>
    </div>
  {/if}
</TileCard>

<style>
  .tile-icon {
    color: var(--color-text-muted);
    margin-bottom: var(--space-2);
    display: flex;
    align-items: center;
  }

  .empty-state {
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    color: var(--color-text-muted);
    margin: 0;
    line-height: 1.4;
  }

  .task-counts {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .count {
    font-family: var(--font-data);
    font-size: var(--text-xs);
  }

  .count.active {
    color: var(--color-text-primary);
  }

  /* AC 12-6-7: Blocked count uses --color-accent-amber */
  .count.blocked {
    color: var(--color-accent-amber);
  }

  .count.done {
    color: var(--color-accent-green);
  }
</style>
