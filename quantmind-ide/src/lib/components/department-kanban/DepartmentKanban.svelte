<script lang="ts">
  /**
   * Department Kanban Component
   *
   * Main kanban board for department task tracking.
   * Connects to SSE endpoint for real-time updates.
   */
  import { onMount, onDestroy } from 'svelte';
  import type { DepartmentTask, TaskStatus, DepartmentName, TaskUpdate } from './types';
  import DepartmentKanbanColumn from './DepartmentKanbanColumn.svelte';
  import { RefreshCw, X } from 'lucide-svelte';

  // Props via Svelte 5 $state
  interface Props {
    department: DepartmentName;
    onClose?: () => void;
  }

  let { department, onClose }: Props = $props();

  // Task state organized by status (AC-1: 4 columns)
  let tasksByStatus = $state<Record<TaskStatus, DepartmentTask[]>>({
    TODO: [],
    IN_PROGRESS: [],
    BLOCKED: [],
    DONE: []
  });

  // SSE connection state
  let eventSource: EventSource | null = null;
  let isConnected = $state(false);
  let isConnecting = $state(false);
  let connectionError = $state<string | null>(null);

  // Refetch interval for fallback
  let refetchInterval: ReturnType<typeof setInterval> | undefined;

  // Task reference for flash animation
  let taskCardRefs = $state<Record<string, { triggerFlash: () => void }>>({});

  /**
   * Fetch initial task state from REST endpoint (AC-3 fallback)
   */
  async function fetchTasks() {
    try {
      const response = await fetch(`/api/tasks/${department}`);
      if (!response.ok) throw new Error('Failed to fetch tasks');

      const data = await response.json();
      updateTasksFromResponse(data.tasks || []);
    } catch (e) {
      console.error('Failed to fetch tasks:', e);
    }
  }

  /**
   * Connect to SSE endpoint for real-time updates (AC-3)
   */
  function connectSSE() {
    if (eventSource) {
      eventSource.close();
    }

    isConnecting = true;
    connectionError = null;

    const sseUrl = `/api/sse/tasks/${department}`;
    eventSource = new EventSource(sseUrl);

    eventSource.onopen = () => {
      isConnected = true;
      isConnecting = false;
      connectionError = null;
    };

    eventSource.onmessage = (event) => {
      try {
        const update: TaskUpdate = JSON.parse(event.data);
        handleTaskUpdate(update);
      } catch (e) {
        console.error('Failed to parse SSE message:', e);
      }
    };

    eventSource.onerror = (e) => {
      console.error('SSE error:', e);
      isConnected = false;
      isConnecting = false;
      connectionError = 'Connection lost. Reconnecting...';

      // Attempt reconnect after delay
      setTimeout(connectSSE, 5000);
    };
  }

  /**
   * Handle task update from SSE (AC-2: targeted DOM update)
   */
  function handleTaskUpdate(update: TaskUpdate) {
    // Find task in current state and move to new column
    const oldStatus = findTaskStatus(update.task_id);

    if (oldStatus && oldStatus !== update.status) {
      // Trigger flash animation before moving
      const taskRef = taskCardRefs[update.task_id];
      if (taskRef) {
        taskRef.triggerFlash();
      }

      // Move task to new status column
      moveTask(update.task_id, oldStatus, update.status);
    }
  }

  /**
   * Find which column a task is currently in
   */
  function findTaskStatus(taskId: string): TaskStatus | null {
    for (const [status, tasks] of Object.entries(tasksByStatus)) {
      if (tasks.some(t => t.task_id === taskId)) {
        return status as TaskStatus;
      }
    }
    return null;
  }

  /**
   * Move task between columns (AC-2: targeted DOM update, no full re-render)
   */
  function moveTask(taskId: string, fromStatus: TaskStatus, toStatus: TaskStatus) {
    const taskIndex = tasksByStatus[fromStatus].findIndex(t => t.task_id === taskId);
    if (taskIndex === -1) return;

    const [task] = tasksByStatus[fromStatus].splice(taskIndex, 1);
    tasksByStatus[toStatus] = [...tasksByStatus[toStatus], { ...task, status: toStatus }];
  }

  /**
   * Update tasks from REST response
   */
  function updateTasksFromResponse(tasks: DepartmentTask[]) {
    tasksByStatus = {
      TODO: tasks.filter(t => t.status === 'TODO'),
      IN_PROGRESS: tasks.filter(t => t.status === 'IN_PROGRESS'),
      BLOCKED: tasks.filter(t => t.status === 'BLOCKED'),
      DONE: tasks.filter(t => t.status === 'DONE')
    };
  }

  /**
   * Handle task drop from drag
   */
  function handleTaskDrop(taskId: string, newStatus: TaskStatus) {
    const oldStatus = findTaskStatus(taskId);
    if (oldStatus && oldStatus !== newStatus) {
      moveTask(taskId, oldStatus, newStatus);

      // Optionally notify backend
      fetch(`/api/tasks/${taskId}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      }).catch(console.error);
    }
  }

  /**
   * Register task card for animation ref
   */
  export function registerTaskCard(taskId: string, ref: { triggerFlash: () => void }) {
    taskCardRefs[taskId] = ref;
  }

  onMount(() => {
    // Initial fetch
    fetchTasks();

    // Connect to SSE
    connectSSE();

    // Fallback refetch every 30 seconds if SSE fails
    refetchInterval = setInterval(() => {
      if (!isConnected) {
        fetchTasks();
      }
    }, 30000);
  });

  onDestroy(() => {
    if (eventSource) {
      eventSource.close();
    }
    if (refetchInterval) {
      clearInterval(refetchInterval);
    }
  });

  // Department title
  let departmentTitle = $derived(
    department.charAt(0).toUpperCase() + department.slice(1)
  );
</script>

<div class="department-kanban">
  <!-- Header -->
  <header class="kanban-header">
    <div class="header-left">
      <h2>{departmentTitle} Tasks</h2>
      <div class="connection-status" class:connected={isConnected} class:connecting={isConnecting}>
        {#if isConnected}
          <span class="status-dot"></span>
          <span>Live</span>
        {:else if isConnecting}
          <RefreshCw size={12} class="spin" />
          <span>Connecting...</span>
        {:else}
          <span class="status-dot error"></span>
          <span>Offline</span>
        {/if}
      </div>
    </div>
    <div class="header-right">
      <button class="close-btn" onclick={onClose} aria-label="Close">
        <X size={18} />
      </button>
    </div>
  </header>

  <!-- Kanban Board -->
  <div class="kanban-board">
    <DepartmentKanbanColumn
      status="TODO"
      tasks={tasksByStatus.TODO}
      onTaskDrop={handleTaskDrop}
    />
    <DepartmentKanbanColumn
      status="IN_PROGRESS"
      tasks={tasksByStatus.IN_PROGRESS}
      onTaskDrop={handleTaskDrop}
    />
    <DepartmentKanbanColumn
      status="BLOCKED"
      tasks={tasksByStatus.BLOCKED}
      onTaskDrop={handleTaskDrop}
    />
    <DepartmentKanbanColumn
      status="DONE"
      tasks={tasksByStatus.DONE}
      onTaskDrop={handleTaskDrop}
    />
  </div>
</div>

<style>
  .department-kanban {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    min-width: 0;
    background: rgba(10, 15, 26, 0.95);
    backdrop-filter: blur(12px);
  }

  .kanban-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid rgba(var(--color-accent-cyan-rgb, 0, 212, 255), 0.1);
    background: rgba(8, 13, 20, 0.6);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .header-left h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: #e0e0e0;
    margin: 0;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.3);
    color: rgba(255, 255, 255, 0.5);
  }

  .connection-status.connected {
    color: var(--color-accent-green, #00c896);
    background: rgba(0, 200, 150, 0.1);
  }

  .connection-status.connecting {
    color: var(--color-accent-amber, #f0a500);
    background: rgba(240, 165, 0, 0.1);
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--color-accent-green, #00c896);
  }

  .status-dot.error {
    background: var(--color-accent-red, #ff3b3b);
  }

  .header-right {
    display: flex;
    align-items: center;
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.6);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .close-btn:hover {
    background: rgba(255, 59, 59, 0.1);
    border-color: rgba(255, 59, 59, 0.3);
    color: var(--color-accent-red, #ff3b3b);
  }

  .kanban-board {
    flex: 1;
    display: flex;
    gap: 16px;
    padding: 20px;
    overflow-x: auto;
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>