/**
 * Agent Stream Service
 *
 * Handles real-time SSE streaming for agent events including:
 * - Agent started/completed/failed events
 * - Response chunks
 * - Tool calls and progress
 * - Progress updates
 */

import { writable, get } from 'svelte/store';

export type AgentStreamEventType =
  | 'connected'
  | 'disconnected'
  | 'error'
  | 'agent_started'
  | 'agent_completed'
  | 'agent_failed'
  | 'response_start'
  | 'response_chunk'
  | 'response_complete'
  | 'tool_start'
  | 'tool_progress'
  | 'tool_complete'
  | 'tool_error'
  | 'progress_update'
  | 'progress_complete'
  | 'heartbeat';

export interface AgentStreamEvent {
  type: AgentStreamEventType;
  timestamp: string;
  agent_id: string | null;
  task_id: string | null;
  tool_name: string | null;
  request_id: string | null;
  [key: string]: unknown;
}

export interface AgentTaskStreamEvent {
  type: string;
  agent_id: string;
  task_id: string;
  status?: string;
  output?: string;
  error?: string;
  tool_calls?: Array<{
    name: string;
    arguments: Record<string, unknown>;
  }>;
  progress?: number;
  message?: string;
}

// Connection state
export const streamConnected = writable(false);
export const streamError = writable<string | null>(null);
export const streamEvents = writable<AgentStreamEvent[]>([]);

// SSE connection state
let eventSource: EventSource | null = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;
const RECONNECT_DELAY = 3000;

// Active task streams (WebSocket)
let taskWs: WebSocket | null = null;
const activeTaskStreams = new Map<string, {
  ws: WebSocket;
  events: ReturnType<typeof writable<AgentTaskStreamEvent[]>>;
}>();

/**
 * Connect to agent event stream (SSE)
 */
export function connectAgentStream(
  agentId?: string,
  taskId?: string,
  eventType?: string,
  onEvent?: (event: AgentStreamEvent) => void
): void {
  const baseUrl = `${window.location.protocol === 'https:' ? 'https:' : 'http:'}//${window.location.host}`;
  let url = `${baseUrl}/api/agents/stream`;

  const params = new URLSearchParams();
  if (agentId) params.set('agent_id', agentId);
  if (taskId) params.set('task_id', taskId);
  if (eventType) params.set('event_type', eventType);

  const queryString = params.toString();
  if (queryString) url += `?${queryString}`;

  // Close existing connection
  if (eventSource) {
    eventSource.close();
  }

  try {
    eventSource = new EventSource(url);

    eventSource.onopen = () => {
      console.log('[AgentStream] Connected to SSE stream');
      streamConnected.set(true);
      streamError.set(null);
      reconnectAttempts = 0;
    };

    eventSource.onmessage = (event) => {
      try {
        const data: AgentStreamEvent = JSON.parse(event.data);

        // Update events store (keep last 100)
        streamEvents.update((events) => {
          const newEvents = [data, ...events].slice(0, 100);
          return newEvents;
        });

        // Call event callback if provided
        if (onEvent) {
          onEvent(data);
        }
      } catch (e) {
        console.error('[AgentStream] Failed to parse event:', e);
      }
    };

    eventSource.onerror = (error) => {
      console.error('[AgentStream] SSE error:', error);
      streamConnected.set(false);
      streamError.set('Connection error');

      // Attempt reconnect
      if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts++;
        console.log(`[AgentStream] Reconnecting in ${RECONNECT_DELAY}ms (attempt ${reconnectAttempts})`);
        setTimeout(
          () => connectAgentStream(agentId, taskId, eventType, onEvent),
          RECONNECT_DELAY
        );
      } else {
        streamError.set('Max reconnection attempts reached');
      }
    };
  } catch (e) {
    console.error('[AgentStream] Failed to connect:', e);
    streamError.set(e instanceof Error ? e.message : 'Failed to connect');
  }
}

/**
 * Disconnect from agent event stream
 */
export function disconnectAgentStream(): void {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  streamConnected.set(false);
}

/**
 * Connect to specific agent task stream (WebSocket)
 */
export function connectTaskStream(
  agentId: string,
  taskId: string
): { events: ReturnType<typeof writable<AgentTaskStreamEvent[]>> } {
  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v2/agents/${agentId}/stream/${taskId}`;

  // Close existing connection for this task
  const existing = activeTaskStreams.get(`${agentId}:${taskId}`);
  if (existing) {
    existing.ws.close();
  }

  const events = writable<AgentTaskStreamEvent[]>([]);

  try {
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`[AgentStream] Task stream connected: ${agentId}/${taskId}`);
    };

    ws.onmessage = (event) => {
      try {
        const data: AgentTaskStreamEvent = JSON.parse(event.data);

        // Update events store
        events.update((evts) => {
          const newEvents = [data, ...evts].slice(0, 200);
          return newEvents;
        });

        // Auto-reconnect on unexpected close (but not on completion)
        if (data.type === 'completed' || data.type === 'error') {
          // Connection will close naturally
        }
      } catch (e) {
        console.error('[AgentStream] Failed to parse task event:', e);
      }
    };

    ws.onerror = (error) => {
      console.error(`[AgentStream] Task stream error: ${agentId}/${taskId}`, error);
    };

    ws.onclose = () => {
      console.log(`[AgentStream] Task stream closed: ${agentId}/${taskId}`);
      activeTaskStreams.delete(`${agentId}:${taskId}`);
    };

    activeTaskStreams.set(`${agentId}:${taskId}`, { ws, events });
    return { events };
  } catch (e) {
    console.error('[AgentStream] Failed to connect to task stream:', e);
    return { events };
  }
}

/**
 * Disconnect from a specific task stream
 */
export function disconnectTaskStream(agentId: string, taskId: string): void {
  const key = `${agentId}:${taskId}`;
  const stream = activeTaskStreams.get(key);
  if (stream) {
    stream.ws.close();
    activeTaskStreams.delete(key);
  }
}

/**
 * Disconnect all task streams
 */
export function disconnectAllTaskStreams(): void {
  for (const [key, stream] of activeTaskStreams) {
    stream.ws.close();
  }
  activeTaskStreams.clear();
}

/**
 * Get event store for a specific task stream
 */
export function getTaskEvents(agentId: string, taskId: string) {
  const key = `${agentId}:${taskId}`;
  const stream = activeTaskStreams.get(key);
  return stream?.events ?? writable<AgentTaskStreamEvent[]>([]);
}

// Event type utilities
export const eventTypeLabels: Record<AgentStreamEventType, string> = {
  connected: 'Connected',
  disconnected: 'Disconnected',
  error: 'Error',
  agent_started: 'Agent Started',
  agent_completed: 'Agent Completed',
  agent_failed: 'Agent Failed',
  response_start: 'Response Start',
  response_chunk: 'Response Chunk',
  response_complete: 'Response Complete',
  tool_start: 'Tool Started',
  tool_progress: 'Tool Progress',
  tool_complete: 'Tool Complete',
  tool_error: 'Tool Error',
  progress_update: 'Progress Update',
  progress_complete: 'Progress Complete',
  heartbeat: 'Heartbeat',
};

export const eventTypeColors: Record<AgentStreamEventType, string> = {
  connected: '#22c55e',
  disconnected: '#6b7280',
  error: '#ef4444',
  agent_started: '#3b82f6',
  agent_completed: '#22c55e',
  agent_failed: '#ef4444',
  response_start: '#8b5cf6',
  response_chunk: '#8b5cf6',
  response_complete: '#22c55e',
  tool_start: '#f59e0b',
  tool_progress: '#f59e0b',
  tool_complete: '#22c55e',
  tool_error: '#ef4444',
  progress_update: '#06b6d4',
  progress_complete: '#22c55e',
  heartbeat: '#6b7280',
};

// Export for convenience
export const agentStreamService = {
  connect: connectAgentStream,
  disconnect: disconnectAgentStream,
  connectTaskStream,
  disconnectTaskStream,
  disconnectAllTaskStreams,
  getTaskEvents,
  connected: streamConnected,
  error: streamError,
  events: streamEvents,
};
