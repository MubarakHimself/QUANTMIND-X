/**
 * Agent Stream Store
 *
 * Svelte writable store for managing live streaming state per task.
 * Components subscribe to this store to display real-time agent output.
 *
 * **Phase 7.2 - Agent Stream Store**
 */

import { writable, derived, get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';
import type { WebSocketEvent, ProgressData, ToolCall } from './claudeCodeAgent';

// Stream state for a single task
export interface TaskStreamState {
  task_id: string;
  agent_id: string;
  status: 'idle' | 'streaming' | 'completed' | 'error';
  output: string;
  tool_calls: ToolCall[];
  error?: string;
  started_at?: Date;
  completed_at?: Date;
}

// Store state: map of task_id to stream state
export type StreamStateMap = Map<string, TaskStreamState>;

// Create the main store
function createAgentStreamStore() {
  const { subscribe, set, update }: Writable<StreamStateMap> = writable(new Map());

  return {
    subscribe,

    /**
     * Initialize a new task stream
     */
    initTask(taskId: string, agentId: string): void {
      update((state) => {
        const newState = new Map(state);
        newState.set(taskId, {
          task_id: taskId,
          agent_id: agentId,
          status: 'streaming',
          output: '',
          tool_calls: [],
          started_at: new Date(),
        });
        return newState;
      });
    },

    /**
     * Handle a WebSocket event for a task
     */
    handleEvent(event: WebSocketEvent): void {
      update((state) => {
        const taskState = state.get(event.task_id);
        if (!taskState) {
          // Initialize if not exists
          const newState = new Map(state);
          newState.set(event.task_id, {
            task_id: event.task_id,
            agent_id: event.agent_id,
            status: 'streaming',
            output: '',
            tool_calls: [],
            started_at: new Date(),
          });
          return newState;
        }

        const newState = new Map(state);
        const updatedTask = { ...taskState };

        switch (event.type) {
          case 'started':
            updatedTask.status = 'streaming';
            break;

          case 'progress':
            if (event.data?.output_delta) {
              updatedTask.output += event.data.output_delta;
            }
            break;

          case 'tool_call':
            if (event.data) {
              updatedTask.tool_calls = [...updatedTask.tool_calls, event.data];
            }
            break;

          case 'completed':
            updatedTask.status = 'completed';
            updatedTask.completed_at = new Date();
            if (event.data?.output) {
              updatedTask.output = event.data.output;
            }
            break;

          case 'error':
            updatedTask.status = 'error';
            updatedTask.error = event.data?.error || 'Unknown error';
            updatedTask.completed_at = new Date();
            break;
        }

        newState.set(event.task_id, updatedTask);
        return newState;
      });
    },

    /**
     * Update output for a task (for progress updates)
     */
    updateOutput(taskId: string, delta: string): void {
      update((state) => {
        const taskState = state.get(taskId);
        if (!taskState) return state;

        const newState = new Map(state);
        newState.set(taskId, {
          ...taskState,
          output: taskState.output + delta,
        });
        return newState;
      });
    },

    /**
     * Mark a task as completed
     */
    completeTask(taskId: string, output?: string, error?: string): void {
      update((state) => {
        const taskState = state.get(taskId);
        if (!taskState) return state;

        const newState = new Map(state);
        newState.set(taskId, {
          ...taskState,
          status: error ? 'error' : 'completed',
          output: output || taskState.output,
          error,
          completed_at: new Date(),
        });
        return newState;
      });
    },

    /**
     * Clear a specific task from the store
     */
    clearTask(taskId: string): void {
      update((state) => {
        const newState = new Map(state);
        newState.delete(taskId);
        return newState;
      });
    },

    /**
     * Clear all tasks
     */
    clearAll(): void {
      set(new Map());
    },

    /**
     * Get the current state for a task
     */
    getTask(taskId: string): TaskStreamState | undefined {
      return get({ subscribe }).get(taskId);
    },

    /**
     * Get the number of tasks in the store
     * Use this instead of direct store.size access
     */
    getSize(): number {
      return get({ subscribe }).size;
    },

    /**
     * Clean up old completed tasks (older than specified milliseconds)
     */
    cleanupOldTasks(maxAgeMs: number = 3600000): void {
      update((state) => {
        const newState = new Map(state);
        const now = Date.now();

        for (const [taskId, taskState] of newState.entries()) {
          if (
            (taskState.status === 'completed' || taskState.status === 'error') &&
            taskState.completed_at
          ) {
            const completedTime = new Date(taskState.completed_at).getTime();
            if (now - completedTime > maxAgeMs) {
              newState.delete(taskId);
            }
          }
        }

        return newState;
      });
    },
  };
}

// Export the store instance
export const agentStreamStore = createAgentStreamStore();

// Derived store for active streams (streaming status)
export const activeStreams: Readable<TaskStreamState[]> = derived(
  agentStreamStore,
  ($state) => {
    return Array.from($state.values()).filter((task) => task.status === 'streaming');
  }
);

// Derived store for completed streams
export const completedStreams: Readable<TaskStreamState[]> = derived(
  agentStreamStore,
  ($state) => {
    return Array.from($state.values()).filter(
      (task) => task.status === 'completed' || task.status === 'error'
    );
  }
);

// Derived store for the latest task
export const latestTask: Readable<TaskStreamState | null> = derived(
  agentStreamStore,
  ($state) => {
    const tasks = Array.from($state.values());
    if (tasks.length === 0) return null;

    // Sort by started_at descending
    tasks.sort((a, b) => {
      const aTime = a.started_at?.getTime() || 0;
      const bTime = b.started_at?.getTime() || 0;
      return bTime - aTime;
    });

    return tasks[0];
  }
);

/**
 * Helper function to connect streaming to the store
 * Use this to automatically pipe events from streamAgent to the store
 *
 * Cleans up completed tasks after a delay to prevent memory bloat.
 */
import { streamAgent } from './claudeCodeAgent';
import type { AgentMessage, AgentContext } from './claudeCodeAgent';

const CLEANUP_DELAY_MS = 60000; // Clean up completed tasks after 1 minute

export async function streamToStore(
  agentId: string,
  messages: AgentMessage[],
  context?: AgentContext
): Promise<TaskStreamState | null> {
  let taskId: string | null = null;

  try {
    for await (const event of streamAgent(agentId, messages, context)) {
      if (!taskId) {
        taskId = event.task_id;
        agentStreamStore.initTask(taskId, agentId);
      }

      agentStreamStore.handleEvent(event);

      if (event.type === 'completed' || event.type === 'error') {
        const result = agentStreamStore.getTask(taskId) || null;

        // Schedule cleanup after delay
        if (taskId) {
          setTimeout(() => {
            agentStreamStore.clearTask(taskId);
          }, CLEANUP_DELAY_MS);
        }

        return result;
      }
    }
  } catch (error) {
    if (taskId) {
      agentStreamStore.completeTask(
        taskId,
        undefined,
        error instanceof Error ? error.message : 'Stream error'
      );

      // Still schedule cleanup on error
      setTimeout(() => {
        agentStreamStore.clearTask(taskId);
      }, CLEANUP_DELAY_MS);
    }
  }

  const result = taskId ? agentStreamStore.getTask(taskId) || null : null;

  // Schedule cleanup for incomplete streams that exit abnormally
  if (taskId) {
    setTimeout(() => {
      agentStreamStore.clearTask(taskId);
    }, CLEANUP_DELAY_MS);
  }

  return result;
}