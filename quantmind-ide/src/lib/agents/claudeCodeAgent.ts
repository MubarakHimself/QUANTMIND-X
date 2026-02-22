/**
 * Claude Code Agent Client
 *
 * Client for the v2 Claude-powered agent API.
 * Replaces LangChain-based agent invocation with Claude CLI orchestration.
 *
 * **Phase 7.1 - Claude Code Agent Client**
 */

import { AGENT_CONFIG } from '../config/api';

// TypeScript interfaces for agent communication

export interface AgentMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface AgentContext {
  mission_id?: string;
  trd_path?: string;
  nprd_content?: string;
  mode?: 'PLAN' | 'ASK' | 'BUILD';
  [key: string]: any;
}

export interface RunAgentRequest {
  messages: AgentMessage[];
  context?: AgentContext;
  session_id?: string;
}

export interface RunAgentResponse {
  task_id: string;
  agent_id: string;
  status: string;
  poll_url: string;
  stream_url: string;
}

export interface AgentTask {
  task_id: string;
  agent_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at?: string;
  completed_at?: string;
  output?: string;
  error?: string;
}

export interface ToolCall {
  name: string;
  args: Record<string, any>;
  result?: any;
}

export interface AgentResult {
  task_id: string;
  agent_id: string;
  status: 'completed' | 'failed';
  completed_at: string;
  output: string;
  tool_calls: ToolCall[];
  error?: string;
}

// WebSocket event types
export type WebSocketEventType = 'started' | 'tool_call' | 'progress' | 'completed' | 'error';

export interface WebSocketEvent {
  type: WebSocketEventType;
  task_id: string;
  agent_id: string;
  timestamp: string;
  data?: any;
}

// Progress data for streaming
export interface ProgressData {
  output_delta: string;
  total_length: number;
}

/**
 * AgentClient - Client for Claude-powered agents
 */
export class AgentClient {
  private baseUrl: string;
  private wsBaseUrl: string;

  constructor(baseUrl: string = '/api/v2/agents') {
    this.baseUrl = baseUrl;
    // Determine WebSocket URL based on current location
    const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.wsBaseUrl = `${protocol}//${typeof window !== 'undefined' ? window.location.host : 'localhost:8000'}${baseUrl}`;
  }

  /**
   * Get list of available agents
   */
  async listAgents(): Promise<string[]> {
    const response = await fetch(this.baseUrl);
    if (!response.ok) {
      throw new Error(`Failed to list agents: ${response.statusText}`);
    }
    const data = await response.json();
    return data.agents;
  }

  /**
   * Run an agent with messages
   * Returns immediately with task ID and URLs for polling/streaming
   */
  async invoke(
    agentId: string,
    messages: AgentMessage[],
    context?: AgentContext
  ): Promise<RunAgentResponse> {
    const request: RunAgentRequest = {
      messages,
      context: context || {},
      session_id: this.getSessionId(),
    };

    const response = await fetch(`${this.baseUrl}/${agentId}/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `Failed to run agent: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Poll for task status
   * Returns current status and result if complete
   */
  async getStatus(agentId: string, taskId: string): Promise<AgentTask> {
    const response = await fetch(`${this.baseUrl}/${agentId}/status/${taskId}`);

    if (!response.ok) {
      throw new Error(`Failed to get status: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Poll for result with exponential backoff
   * Blocks until the task is complete or timeout
   */
  async pollForResult(
    agentId: string,
    taskId: string,
    options: {
      maxAttempts?: number;
      initialDelay?: number;
      maxDelay?: number;
      onProgress?: (status: AgentTask) => void;
    } = {}
  ): Promise<AgentResult> {
    const {
      maxAttempts = 60,
      initialDelay = 1000,
      maxDelay = 10000,
      onProgress,
    } = options;

    let delay = initialDelay;
    let attempts = 0;

    while (attempts < maxAttempts) {
      const status = await this.getStatus(agentId, taskId);

      if (onProgress) {
        onProgress(status);
      }

      if (status.status === 'completed' || status.status === 'failed') {
        // Fetch full result
        const result = await this.getStatus(agentId, taskId);
        return result as AgentResult;
      }

      // Wait with exponential backoff
      await new Promise((resolve) => setTimeout(resolve, delay));
      delay = Math.min(delay * 1.5, maxDelay);
      attempts++;
    }

    throw new Error(`Polling timeout after ${maxAttempts} attempts`);
  }

  /**
   * Stream result via WebSocket
   * Yields events as they occur
   */
  async *stream(
    agentId: string,
    taskId: string,
    options: {
      onOpen?: () => void;
      onClose?: () => void;
    } = {}
  ): AsyncGenerator<WebSocketEvent> {
    const wsUrl = `${this.wsBaseUrl}/${agentId}/stream/${taskId}`;

    const ws = new WebSocket(wsUrl);
    const eventQueue: WebSocketEvent[] = [];
    let resolveNext: ((value: WebSocketEvent) => void) | null = null;
    let done = false;
    let error: Error | null = null;

    ws.onopen = () => {
      if (options.onOpen) {
        options.onOpen();
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketEvent;
        eventQueue.push(data);

        if (resolveNext) {
          resolveNext(eventQueue.shift()!);
          resolveNext = null;
        }
      } catch (e) {
        error = new Error(`Failed to parse WebSocket message: ${event.data}`);
      }
    };

    ws.onerror = (event) => {
      error = new Error('WebSocket error');
      done = true;
    };

    ws.onclose = () => {
      done = true;
      if (options.onClose) {
        options.onClose();
      }
    };

    try {
      while (!done || eventQueue.length > 0) {
        if (error) {
          throw error;
        }

        if (eventQueue.length > 0) {
          const event = eventQueue.shift()!;
          yield event;

          if (event.type === 'completed' || event.type === 'error') {
            break;
          }
        } else {
          // Wait for next event
          yield await new Promise<WebSocketEvent>((resolve) => {
            resolveNext = resolve;
          });
        }
      }
    } finally {
      ws.close();
    }
  }

  /**
   * Cancel a running task
   */
  async cancel(agentId: string, taskId: string): Promise<boolean> {
    const response = await fetch(`${this.baseUrl}/${agentId}/tasks/${taskId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Failed to cancel task: ${response.statusText}`);
    }

    const data = await response.json();
    return data.success;
  }

  /**
   * Get or create session ID for continuity
   */
  private getSessionId(): string {
    if (typeof window === 'undefined') {
      return `session-${Date.now()}`;
    }

    const storageKey = 'quantmind_session_id';
    let sessionId = localStorage.getItem(storageKey);

    if (!sessionId) {
      sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem(storageKey, sessionId);
    }

    return sessionId;
  }

  /**
   * Clear session ID (for new session)
   */
  clearSession(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('quantmind_session_id');
    }
  }
}

// Singleton instance
let agentClient: AgentClient | null = null;

/**
 * Get the global agent client instance
 */
export function getAgentClient(): AgentClient {
  if (!agentClient) {
    agentClient = new AgentClient();
  }
  return agentClient;
}

/**
 * Convenience function to run an agent and wait for result
 */
export async function runAgent(
  agentId: string,
  messages: AgentMessage[],
  context?: AgentContext,
  options?: {
    onProgress?: (status: AgentTask) => void;
  }
): Promise<AgentResult> {
  const client = getAgentClient();

  // Start the agent
  const response = await client.invoke(agentId, messages, context);

  // Poll for result
  return client.pollForResult(agentId, response.task_id, options);
}

/**
 * Convenience function to stream agent output
 */
export async function* streamAgent(
  agentId: string,
  messages: AgentMessage[],
  context?: AgentContext
): AsyncGenerator<WebSocketEvent> {
  const client = getAgentClient();

  // Start the agent
  const response = await client.invoke(agentId, messages, context);

  // Stream results
  yield* client.stream(agentId, response.task_id);
}