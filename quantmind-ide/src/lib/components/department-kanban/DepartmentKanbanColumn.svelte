<script lang="ts">
  /**
   * Department Kanban Column Component
   *
   * Container for a single kanban column (TODO, IN_PROGRESS, BLOCKED, DONE).
   */
  import type { TaskStatus, DepartmentTask } from './types';
  import DepartmentKanbanCard from './DepartmentKanbanCard.svelte';

  // Props via Svelte 5 $state
  interface Props {
    status: TaskStatus;
    tasks: DepartmentTask[];
    onTaskDrop?: (taskId: string, newStatus: TaskStatus) => void;
    onTaskSelect?: (task: DepartmentTask) => void;
    selectedTaskId?: string | null;
  }

  let { status, tasks, onTaskDrop, onTaskSelect, selectedTaskId = null }: Props = $props();

  // Column header config — colors reference CSS custom properties where possible
  const columnConfig: Record<TaskStatus, { label: string; cssVar: string; fallback: string }> = {
    TODO: { label: 'TODO', cssVar: '--color-text-muted', fallback: 'rgba(255, 255, 255, 0.6)' },
    IN_PROGRESS: { label: 'IN PROGRESS', cssVar: '--color-accent-cyan', fallback: '#00d4ff' },
    BLOCKED: { label: 'BLOCKED', cssVar: '--color-accent-red', fallback: '#ff3b3b' },
    DONE: { label: 'DONE', cssVar: '--color-accent-green', fallback: '#22c55e' }
  };

  let config = $derived(columnConfig[status]);
  let columnColor = $derived(`var(${config.cssVar}, ${config.fallback})`);
  let taskCount = $derived(tasks.length);

  // Drag state
  let isDragOver = $state(false);

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    isDragOver = true;
  }

  function handleDragLeave() {
    isDragOver = false;
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    isDragOver = false;

    const taskId = e.dataTransfer?.getData('text/plain');
    if (taskId && onTaskDrop) {
      onTaskDrop(taskId, status);
    }
  }
</script>

<div
  class="kanban-column"
  class:drag-over={isDragOver}
  role="region"
  aria-label="{config.label} column"
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
>
  <div class="column-header">
    <span class="column-title" style="color: {columnColor};">
      {config.label}
    </span>
    <span class="task-count">{taskCount}</span>
  </div>

  <div class="column-content">
    {#each tasks as task (task.task_id)}
      <div class="task-wrapper" draggable="true">
        <DepartmentKanbanCard
          {task}
          onSelect={onTaskSelect}
          selected={selectedTaskId === task.task_id}
        />
      </div>
    {/each}

    {#if tasks.length === 0}
      <div class="empty-state">
        <span>No tasks</span>
      </div>
    {/if}
  </div>
</div>

<style>
  .kanban-column {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 200px;
    background: rgba(20, 25, 35, 0.4);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    overflow: hidden;
  }

  .kanban-column.drag-over {
    border-color: rgba(var(--color-accent-cyan-rgb, 0, 212, 255), 0.3);
    background: rgba(var(--color-accent-cyan-rgb, 0, 212, 255), 0.05);
  }

  .column-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(10, 15, 25, 0.3);
  }

  .column-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .task-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    color: rgba(255, 255, 255, 0.5);
  }

  .column-content {
    flex: 1;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow-y: auto;
    min-height: 200px;
  }

  .task-wrapper {
    cursor: grab;
  }

  .task-wrapper:active {
    cursor: grabbing;
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 80px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }
</style>
