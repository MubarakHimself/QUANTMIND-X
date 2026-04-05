<script lang="ts">
  /**
   * Department Kanban Component
   *
   * Main kanban board for department task tracking.
   * Connects to SSE endpoint for real-time updates.
   */
  import { onMount, onDestroy } from 'svelte';
  import type {
    DepartmentTask,
    TaskStatus,
    DepartmentName,
    TaskUpdate,
    DepartmentTaskEvent
  } from './types';
  import DepartmentKanbanColumn from './DepartmentKanbanColumn.svelte';
  import { RefreshCw, X, Clock, ArrowRightLeft, Workflow, Mail } from 'lucide-svelte';
  import { API_CONFIG } from '$lib/config/api';

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
  let selectedTaskId = $state<string | null>(null);

  let allTasks = $derived([
    ...tasksByStatus.TODO,
    ...tasksByStatus.IN_PROGRESS,
    ...tasksByStatus.BLOCKED,
    ...tasksByStatus.DONE,
  ]);

  let selectedTask = $derived(
    selectedTaskId ? allTasks.find((task) => task.task_id === selectedTaskId) ?? null : null
  );

  function getTasksApiUrl(path: string): string {
    return `${API_CONFIG.API_URL}${path}`;
  }

  /**
   * Fetch initial task state from REST endpoint (AC-3 fallback)
   */
  async function fetchTasks() {
    try {
      const response = await fetch(getTasksApiUrl(`/api/tasks/${department}`));
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

    const sseUrl = getTasksApiUrl(`/api/sse/tasks/${department}`);
    eventSource = new EventSource(sseUrl);

    eventSource.onopen = () => {
      isConnected = true;
      isConnecting = false;
      connectionError = null;
      void fetchTasks();
    };

    eventSource.onmessage = (event) => {
      try {
        const update: DepartmentTaskEvent = JSON.parse(event.data);
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
  function handleTaskUpdate(update: DepartmentTaskEvent) {
    if ('type' in update) {
      if (update.type === 'initial') {
        updateTasksFromResponse(update.tasks || []);
      }
      return;
    }

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

    if (selectedTaskId && !tasks.some((task) => task.task_id === selectedTaskId)) {
      selectedTaskId = tasks[0]?.task_id ?? null;
    } else if (!selectedTaskId && tasks.length > 0) {
      selectedTaskId = tasks[0].task_id;
    }
  }

  /**
   * Handle task drop from drag
   */
  function handleTaskDrop(taskId: string, newStatus: TaskStatus) {
    const task = allTasks.find((candidate) => candidate.task_id === taskId);
    if (!task || task.read_only) {
      return;
    }

    const oldStatus = findTaskStatus(taskId);
    if (oldStatus && oldStatus !== newStatus) {
      moveTask(taskId, oldStatus, newStatus);

      // Optionally notify backend
      fetch(getTasksApiUrl(`/api/tasks/${taskId}/status`), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      }).catch(console.error);
    }
  }

  function handleTaskSelect(task: DepartmentTask) {
    selectedTaskId = task.task_id;
  }

  function formatDateTime(value?: string): string {
    if (!value) return 'Unavailable';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
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

  <div class="kanban-layout">
    <div class="kanban-board">
      <DepartmentKanbanColumn
        status="TODO"
        tasks={tasksByStatus.TODO}
        onTaskDrop={handleTaskDrop}
        onTaskSelect={handleTaskSelect}
        {selectedTaskId}
      />
      <DepartmentKanbanColumn
        status="IN_PROGRESS"
        tasks={tasksByStatus.IN_PROGRESS}
        onTaskDrop={handleTaskDrop}
        onTaskSelect={handleTaskSelect}
        {selectedTaskId}
      />
      <DepartmentKanbanColumn
        status="BLOCKED"
        tasks={tasksByStatus.BLOCKED}
        onTaskDrop={handleTaskDrop}
        onTaskSelect={handleTaskSelect}
        {selectedTaskId}
      />
      <DepartmentKanbanColumn
        status="DONE"
        tasks={tasksByStatus.DONE}
        onTaskDrop={handleTaskDrop}
        onTaskSelect={handleTaskSelect}
        {selectedTaskId}
      />
    </div>

    <aside class="task-detail-panel">
      {#if selectedTask}
        <div class="detail-header">
          <div>
            <div class="detail-eyebrow">{selectedTask.department} task</div>
            <h3>{selectedTask.task_name}</h3>
          </div>
          <div class="detail-badges">
            <span class="detail-badge priority">{selectedTask.priority}</span>
            <span class="detail-badge status">{selectedTask.status.replace('_', ' ')}</span>
          </div>
        </div>

        <div class="detail-body">
          <section class="detail-section">
            <h4>Summary</h4>
            <p>{selectedTask.description || 'No task description was provided with this item.'}</p>
          </section>

          <section class="detail-grid">
            <div class="detail-card">
              <div class="detail-card-label">
                <ArrowRightLeft size={13} />
                Source Department
              </div>
              <div class="detail-card-value">{selectedTask.source_dept || 'Not linked to department mail'}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-label">
                <Mail size={13} />
                Message Type
              </div>
              <div class="detail-card-value">{selectedTask.message_type || 'Unknown'}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-label">
                <Workflow size={13} />
                Workflow
              </div>
              <div class="detail-card-value">{selectedTask.workflow_id || 'Not attached'}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-label">
                <Workflow size={13} />
                Stage
              </div>
              <div class="detail-card-value">{selectedTask.current_stage || 'Unknown'}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-label">
                <Clock size={13} />
                Updated
              </div>
              <div class="detail-card-value">{formatDateTime(selectedTask.updated_at || selectedTask.created_at)}</div>
            </div>
          </section>

          <section class="detail-grid">
            <div class="detail-card">
              <div class="detail-card-label">Strategy</div>
              <div class="detail-card-value">{selectedTask.strategy_id || 'Not linked'}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-label">Source</div>
              <div class="detail-card-value">{selectedTask.source_kind || 'mail'}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-label">Next Step</div>
              <div class="detail-card-value">{selectedTask.next_step || 'None'}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-label">Editability</div>
              <div class="detail-card-value">{selectedTask.read_only ? 'Workflow state only' : 'Interactive task'}</div>
            </div>
          </section>

          {#if selectedTask.blocking_error || selectedTask.waiting_reason || selectedTask.latest_artifact}
            <section class="detail-section">
              <h4>Workflow State</h4>
              <dl class="trace-list">
                <div>
                  <dt>Waiting Reason</dt>
                  <dd>{selectedTask.waiting_reason || 'None'}</dd>
                </div>
                <div>
                  <dt>Blocking Error</dt>
                  <dd>{selectedTask.blocking_error || 'None'}</dd>
                </div>
                <div>
                  <dt>Latest Artifact</dt>
                  <dd>{selectedTask.latest_artifact?.path || selectedTask.latest_artifact?.name || 'Unavailable'}</dd>
                </div>
              </dl>
            </section>
          {/if}

          <section class="detail-section">
            <h4>Trace</h4>
            <dl class="trace-list">
              <div>
                <dt>Task ID</dt>
                <dd>{selectedTask.task_id}</dd>
              </div>
              <div>
                <dt>Mail ID</dt>
                <dd>{selectedTask.mail_message_id || 'Unavailable'}</dd>
              </div>
              <div>
                <dt>Kanban ID</dt>
                <dd>{selectedTask.kanban_card_id || 'Unavailable'}</dd>
              </div>
              <div>
                <dt>Created</dt>
                <dd>{formatDateTime(selectedTask.created_at)}</dd>
              </div>
              <div>
                <dt>Started</dt>
                <dd>{formatDateTime(selectedTask.started_at)}</dd>
              </div>
              <div>
                <dt>Completed</dt>
                <dd>{formatDateTime(selectedTask.completed_at)}</dd>
              </div>
            </dl>
          </section>
        </div>
      {:else}
        <div class="empty-detail">
          <h3>No task selected</h3>
          <p>Select a card to inspect its source, description, workflow link, and lifecycle timestamps.</p>
        </div>
      {/if}
    </aside>
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

  .kanban-layout {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 2.2fr) minmax(320px, 0.95fr);
  }

  .kanban-board {
    display: flex;
    gap: 16px;
    padding: 20px;
    overflow-x: auto;
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
  }

  .task-detail-panel {
    border-left: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(6, 10, 18, 0.88);
    padding: 20px;
    overflow-y: auto;
    min-width: 0;
  }

  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 20px;
  }

  .detail-eyebrow {
    margin-bottom: 8px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255, 255, 255, 0.42);
  }

  .detail-header h3,
  .empty-detail h3 {
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    color: #f8fafc;
    line-height: 1.35;
  }

  .detail-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .detail-badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.07);
    color: rgba(255, 255, 255, 0.78);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .detail-body {
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .detail-section,
  .detail-grid,
  .empty-detail {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 16px;
  }

  .detail-section h4 {
    margin: 0 0 10px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255, 255, 255, 0.5);
  }

  .detail-section p,
  .empty-detail p {
    margin: 0;
    color: rgba(255, 255, 255, 0.72);
    line-height: 1.6;
    font-size: 13px;
  }

  .detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }

  .detail-card {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 12px;
  }

  .detail-card-label {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    color: rgba(255, 255, 255, 0.48);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .detail-card-value {
    color: rgba(255, 255, 255, 0.82);
    font-size: 13px;
    line-height: 1.5;
    word-break: break-word;
  }

  .trace-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px 16px;
    margin: 0;
  }

  .trace-list div {
    min-width: 0;
  }

  .trace-list dt {
    margin-bottom: 4px;
    color: rgba(255, 255, 255, 0.48);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .trace-list dd {
    margin: 0;
    color: rgba(255, 255, 255, 0.82);
    font-size: 13px;
    line-height: 1.45;
    word-break: break-word;
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  @media (max-width: 1280px) {
    .kanban-layout {
      grid-template-columns: 1fr;
    }

    .task-detail-panel {
      border-left: none;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
  }

  @media (max-width: 900px) {
    .detail-grid,
    .trace-list {
      grid-template-columns: 1fr;
    }
  }
</style>
