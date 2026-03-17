<script lang="ts">
  /**
   * Agent Queue Panel Component
   *
   * Displays task queues for each agent with management controls.
   */
  import { onMount } from 'svelte';

  // Types
  interface QueueStatus {
    agent_type: string;
    exists: boolean;
    pending_count: number;
    running_count: number;
    max_concurrent: number;
    is_full: boolean;
  }

  interface Task {
    task_id: string;
    name: string;
    description: string;
    agent_type: string;
    priority: number;
    status: string;
    created_at: string;
    started_at?: string;
    completed_at?: string;
    duration_seconds?: number;
    error?: string;
    retries: number;
  }

  // State
  let queueStatuses: QueueStatus[] = $state([]);
  let selectedAgent: string | null = $state(null);
  let agentTasks: Task[] = $state([]);
  let loading = $state(false);
  let tasksLoading = false;

  // Agent display names
  const agentNames: Record<string, string> = {
    copilot: 'Copilot',
    analyst: 'Analyst',
    quantcode: 'QuantCode',
  };

  // Agent types to fetch
  const agentTypes = ['copilot', 'analyst', 'quantcode'];

  // Methods
  async function fetchQueueStatuses() {
    loading = true;
    try {
      // Fetch all queue statuses at once
      const response = await fetch('/api/agents/queue/all');
      if (response.ok) {
        const data = await response.json();
        queueStatuses = Object.values(data.queues || {});
      } else {
        // Fallback: fetch each queue individually
        const statusPromises = agentTypes.map(async (agentType) => {
          try {
            const res = await fetch(`/api/agents/${agentType}/queue`);
            if (res.ok) {
              return await res.json();
            }
          } catch {
            // Return default on error
          }
          return {
            agent_type: agentType,
            exists: true,
            pending_count: 0,
            running_count: 0,
            max_concurrent: 1,
            is_full: false,
          };
        });
        queueStatuses = await Promise.all(statusPromises);
      }
    } catch (err) {
      console.error('Failed to fetch queue statuses:', err);
      // Set default values on error
      queueStatuses = agentTypes.map((agentType) => ({
        agent_type: agentType,
        exists: true,
        pending_count: 0,
        running_count: 0,
        max_concurrent: 1,
        is_full: false,
      }));
    } finally {
      loading = false;
    }
  }

  async function selectAgentQueue(agentType: string) {
    selectedAgent = agentType;
    tasksLoading = true;
    agentTasks = [];

    try {
      const response = await fetch(`/api/agents/${agentType}/queue/tasks`);
      if (response.ok) {
        const data = await response.json();
        agentTasks = data.tasks || [];
      }
    } catch (err) {
      console.error('Failed to fetch tasks:', err);
      agentTasks = [];
    } finally {
      tasksLoading = false;
    }
  }

  async function cancelTask(taskId: string) {
    if (!selectedAgent) return;

    try {
      const response = await fetch(`/api/agents/${selectedAgent}/queue/${taskId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        // Refresh both tasks and queue status
        await Promise.all([selectAgentQueue(selectedAgent), fetchQueueStatuses()]);
      }
    } catch (err) {
      console.error('Failed to cancel task:', err);
    }
  }

  async function retryTask(taskId: string) {
    if (!selectedAgent) return;

    try {
      const response = await fetch(`/api/agents/${selectedAgent}/queue/${taskId}/retry`, {
        method: 'POST',
      });
      if (response.ok) {
        // Refresh both tasks and queue status
        await Promise.all([selectAgentQueue(selectedAgent), fetchQueueStatuses()]);
      }
    } catch (err) {
      console.error('Failed to retry task:', err);
    }
  }

  function getStatusColor(status: string): string {
    const colors: Record<string, string> = {
      pending: 'bg-gray-500',
      queued: 'bg-blue-500',
      running: 'bg-green-500',
      completed: 'bg-green-600',
      failed: 'bg-red-500',
      cancelled: 'bg-gray-400',
    };
    return colors[status] || 'bg-gray-500';
  }

  function formatTime(isoString: string | undefined): string {
    if (!isoString) return '-';
    return new Date(isoString).toLocaleTimeString();
  }

  // Lifecycle
  onMount(() => {
    fetchQueueStatuses();
  });
</script>

<div class="queue-panel h-full flex flex-col">
  <!-- Header -->
  <div class="flex items-center justify-between p-4 border-b border-gray-700">
    <h2 class="text-lg font-semibold">Agent Queues</h2>
    <button
      class="px-3 py-1 text-sm bg-blue-600 rounded hover:bg-blue-700"
      onclick={fetchQueueStatuses}
      disabled={loading}
    >
      Refresh
    </button>
  </div>

  <!-- Agent List -->
  <div class="flex-1 overflow-auto">
    <div class="p-4 space-y-3">
      {#each queueStatuses as queue (queue.agent_type)}
        <button
          class="w-full p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 text-left transition-colors"
          class:border-blue-500={selectedAgent === queue.agent_type}
          onclick={() => selectAgentQueue(queue.agent_type)}
        >
          <div class="flex items-center justify-between">
            <span class="font-medium">{agentNames[queue.agent_type] || queue.agent_type}</span>
            <div class="flex items-center gap-2">
              {#if queue.running_count > 0}
                <span class="px-2 py-0.5 text-xs bg-green-600 rounded">
                  {queue.running_count} running
                </span>
              {/if}
              {#if queue.pending_count > 0}
                <span class="px-2 py-0.5 text-xs bg-blue-600 rounded">
                  {queue.pending_count} pending
                </span>
              {/if}
              {#if queue.pending_count === 0 && queue.running_count === 0}
                <span class="px-2 py-0.5 text-xs bg-gray-600 rounded">
                  Empty
                </span>
              {/if}
            </div>
          </div>

          <!-- Capacity indicator -->
          <div class="mt-2">
            <div class="h-1 bg-gray-700 rounded-full overflow-hidden">
              <div
                class="h-full bg-blue-500 transition-all"
                style="width: {(queue.running_count / queue.max_concurrent) * 100}%"
              ></div>
            </div>
            <div class="flex justify-between text-xs text-gray-500 mt-1">
              <span>Capacity</span>
              <span>{queue.running_count}/{queue.max_concurrent}</span>
            </div>
          </div>
        </button>
      {/each}
    </div>

    <!-- Task List for Selected Agent -->
    {#if selectedAgent && agentTasks.length > 0}
      <div class="border-t border-gray-700 p-4">
        <h3 class="text-sm font-medium text-gray-400 mb-3">
          {agentNames[selectedAgent] || selectedAgent} Tasks
        </h3>

        <div class="space-y-2">
          {#each agentTasks as task (task.task_id)}
            <div class="p-3 bg-gray-800 rounded border border-gray-700">
              <div class="flex items-center justify-between mb-1">
                <span class="font-medium text-sm">{task.name}</span>
                <span class="px-2 py-0.5 text-xs rounded {getStatusColor(task.status)}">
                  {task.status}
                </span>
              </div>

              {#if task.description}
                <p class="text-xs text-gray-400 mb-2">{task.description}</p>
              {/if}

              <div class="flex items-center justify-between text-xs text-gray-500">
                <span>Priority: {task.priority}</span>
                <span>Created: {formatTime(task.created_at)}</span>
              </div>

              {#if task.error}
                <p class="text-xs text-red-400 mt-2">{task.error}</p>
              {/if}

              <!-- Task Actions -->
              <div class="mt-2 flex gap-2">
                {#if task.status === 'pending' || task.status === 'queued'}
                  <button
                    class="px-2 py-0.5 text-xs bg-red-600 rounded hover:bg-red-700"
                    onclick={() => cancelTask(task.task_id)}
                  >
                    Cancel
                  </button>
                {/if}
                {#if task.status === 'failed' && task.retries < 3}
                  <button
                    class="px-2 py-0.5 text-xs bg-blue-600 rounded hover:bg-blue-700"
                    onclick={() => retryTask(task.task_id)}
                  >
                    Retry
                  </button>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {:else if selectedAgent}
      <div class="border-t border-gray-700 p-4 text-center text-gray-500">
        <p>No tasks in queue</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .queue-panel {
    min-width: 280px;
  }
</style>
