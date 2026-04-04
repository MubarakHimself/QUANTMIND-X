<script lang="ts">
  /**
   * Department Kanban Card Component
   *
   * Individual task card for the Department Kanban board.
   * Displays task name, department badge, priority badge, and duration.
   */
  import { onMount, onDestroy } from 'svelte';
  import type { DepartmentTask, TaskPriority } from './types';
  import { Clock, ArrowRightLeft, MessageSquareText } from 'lucide-svelte';

  // Props via Svelte 5 $state
  interface Props {
    task: DepartmentTask;
    onStatusChange?: (taskId: string, newStatus: DepartmentTask['status']) => void;
    onSelect?: (task: DepartmentTask) => void;
    selected?: boolean;
  }

  let { task, onStatusChange, onSelect, selected = false }: Props = $props();

  // Priority colors per AC-1 — use CSS custom properties at runtime via getComputedStyle if available
  const priorityColors: Record<TaskPriority, string> = {
    HIGH: 'var(--color-accent-red, #ff3b3b)',
    MEDIUM: 'var(--color-accent-amber, #f0a500)',
    LOW: '#6b7280'
  };

  // Department badge colors
  const deptColors: Record<string, string> = {
    research: 'var(--color-accent-cyan, #00d4ff)',
    development: 'var(--color-accent-purple, #a855f7)',
    risk: 'var(--color-accent-red, #ff3b3b)',
    trading: 'var(--color-accent-green, #22c55e)',
    portfolio: 'var(--color-accent-amber, #f59e0b)'
  };

  // Calculate duration from start time (AC-4)
  let currentTime = $state(new Date());

  // Update current time every minute
  let interval: ReturnType<typeof setInterval> | undefined;

  onMount(() => {
    interval = setInterval(() => {
      currentTime = new Date();
    }, 60000);
  });

  onDestroy(() => {
    if (interval) clearInterval(interval);
  });

  // Derived duration calculation (fixed: $derived as expression, not function)
  let duration = $derived.by(() => {
    if (!task.started_at) return '--';

    const start = new Date(task.started_at);
    const diffMs = currentTime.getTime() - start.getTime();
    const diffMinutes = Math.floor(diffMs / 60000);

    if (diffMinutes < 60) {
      return `${diffMinutes}m`;
    } else if (diffMinutes < 1440) {
      const hours = Math.floor(diffMinutes / 60);
      const mins = diffMinutes % 60;
      return `${hours}h ${mins}m`;
    } else {
      const days = Math.floor(diffMinutes / 1440);
      return `${days}d`;
    }
  });

  // Get priority color
  let priorityColor = $derived(priorityColors[task.priority]);

  // Get department color
  let deptColor = $derived(deptColors[task.department] || '#6b7280');

  // Animation state for card movement
  let isFlashing = $state(false);

  // Trigger flash animation when status changes
  export function triggerFlash() {
    isFlashing = true;
    setTimeout(() => {
      isFlashing = false;
    }, 400);
  }
</script>

<div
  class="task-card"
  class:flashing={isFlashing}
  class:selected
  style="--priority-color: {priorityColor}; --dept-color: {deptColor};"
  role="button"
  tabindex="0"
  onclick={() => onSelect?.(task)}
  onkeydown={(event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onSelect?.(task);
    }
  }}
>
  <div class="card-header">
    <span class="task-name">{task.task_name}</span>
  </div>

  {#if task.description}
    <div class="task-preview">{task.description}</div>
  {/if}

  <div class="card-badges">
    <!-- Department Badge -->
    <span class="badge dept-badge" style="background: {deptColor}20; color: {deptColor}; border-color: {deptColor}40;">
      {task.department}
    </span>

    <!-- Priority Badge -->
    <span class="badge priority-badge" style="background: {priorityColor}20; color: {priorityColor}; border-color: {priorityColor}40;">
      {task.priority}
    </span>
  </div>

  <div class="card-meta">
    {#if task.source_dept}
      <span class="meta-chip">
        <ArrowRightLeft size={11} />
        {task.source_dept}
      </span>
    {/if}
    {#if task.message_type}
      <span class="meta-chip">
        <MessageSquareText size={11} />
        {task.message_type}
      </span>
    {/if}
  </div>

  <div class="card-footer">
    <Clock size={12} />
    <span class="duration">{duration}</span>
  </div>
</div>

<style>
  .task-card {
    background: rgba(30, 35, 50, 0.6);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 12px;
    transition: all 0.2s ease;
    cursor: pointer;
  }

  .task-card:hover {
    background: rgba(40, 45, 60, 0.7);
    border-color: rgba(255, 255, 255, 0.12);
  }

  .task-card.selected {
    border-color: rgba(var(--color-accent-cyan-rgb, 0, 212, 255), 0.45);
    box-shadow: 0 0 0 1px rgba(var(--color-accent-cyan-rgb, 0, 212, 255), 0.2);
    background: rgba(30, 44, 60, 0.72);
  }

  /* AC-2: 400ms cyan border flash on card move */
  .task-card.flashing {
    animation: flashBorder 400ms ease-out;
  }

  @keyframes flashBorder {
    0% {
      border-color: rgba(var(--color-accent-cyan-rgb, 0, 212, 255), 0);
      box-shadow: none;
    }
    50% {
      border-color: var(--color-accent-cyan, #00d4ff);
      box-shadow: 0 0 12px rgba(var(--color-accent-cyan-rgb, 0, 212, 255), 0.4);
    }
    100% {
      border-color: rgba(255, 255, 255, 0.08);
      box-shadow: none;
    }
  }

  .card-header {
    margin-bottom: 8px;
  }

  .task-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.9);
    line-height: 1.4;
  }

  .task-preview {
    margin-bottom: 10px;
    color: rgba(255, 255, 255, 0.58);
    font-size: 12px;
    line-height: 1.45;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .card-badges {
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }

  .badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .dept-badge {
    /* Department-specific color set inline */
  }

  .priority-badge {
    /* Priority-specific color set inline */
  }

  .card-footer {
    display: flex;
    align-items: center;
    gap: 4px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .card-meta {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }

  .meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 7px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.58);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .duration {
    color: rgba(255, 255, 255, 0.5);
  }
</style>
