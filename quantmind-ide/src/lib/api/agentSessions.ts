/**
 * Agent Session API
 *
 * Provides functions for interacting with the agent session endpoints.
 */

import { writable, get } from 'svelte/store';
import type { Writable } from 'svelte/store';

export interface SessionMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface AgentSession {
  session_id: string;
  name: string;
  agent_type: string;
  status: 'active' | 'completed' | 'failed' | 'archived';
  conversation_history: SessionMessage[];
  variables: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string;
  modified_at: string;
  completed_at?: string;
  message_count: number;
}

export interface SessionSummary {
  session_id: string;
  name: string;
  agent_type: string;
  status: 'active' | 'completed' | 'failed' | 'archived';
  created_at: string;
  modified_at: string;
  completed_at?: string;
  message_count: number;
}

export interface SessionListResponse {
  sessions: SessionSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface SessionStats {
  total_sessions: number;
  active_sessions: number;
  completed_sessions: number;
  archived_sessions: number;
  by_agent_type: Record<string, number>;
}

export interface CreateSessionRequest {
  name: string;
  agent_type: string;
  session_id?: string;
  metadata?: Record<string, unknown>;
  variables?: Record<string, unknown>;
}

const API_BASE = '/api/agent-sessions';

// ============================================================================
// API Functions
// ============================================================================

/**
 * Create a new agent session.
 */
export async function createSession(request: CreateSessionRequest): Promise<AgentSession> {
  const response = await fetch(API_BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to create session' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * List all sessions with optional filtering.
 */
export async function listSessions(
  agentType?: string,
  status?: string,
  limit: number = 50,
  offset: number = 0
): Promise<SessionListResponse> {
  const params = new URLSearchParams();
  if (agentType) params.set('agent_type', agentType);
  if (status) params.set('status', status);
  params.set('limit', limit.toString());
  params.set('offset', offset.toString());

  const response = await fetch(`${API_BASE}?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to list sessions' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Get session statistics.
 */
export async function getSessionStats(): Promise<SessionStats> {
  const response = await fetch(`${API_BASE}/stats`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to get stats' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Get a specific session by ID.
 */
export async function getSession(sessionId: string, includeHistory: boolean = true): Promise<AgentSession> {
  const params = new URLSearchParams();
  params.set('include_history', includeHistory.toString());

  const response = await fetch(`${API_BASE}/${sessionId}?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to get session' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Update session metadata.
 */
export async function updateSession(
  sessionId: string,
  updates: {
    name?: string;
    status?: string;
    metadata?: Record<string, unknown>;
    variables?: Record<string, unknown>;
  }
): Promise<{ success: boolean; session_id: string }> {
  const response = await fetch(`${API_BASE}/${sessionId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to update session' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Delete a session.
 */
export async function deleteSession(sessionId: string): Promise<{ success: boolean; session_id: string }> {
  const response = await fetch(`${API_BASE}/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to delete session' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Add a message to session conversation.
 */
export async function addMessage(
  sessionId: string,
  role: string,
  content: string,
  metadata?: Record<string, unknown>
): Promise<{ success: boolean; session_id: string; message_count: number }> {
  const response = await fetch(`${API_BASE}/${sessionId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role, content, metadata }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to add message' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Replace conversation history.
 */
export async function setConversationHistory(
  sessionId: string,
  history: SessionMessage[]
): Promise<{ success: boolean; session_id: string; message_count: number }> {
  const response = await fetch(`${API_BASE}/${sessionId}/history`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ history }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to set history' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Export session for backup.
 */
export async function exportSession(sessionId: string): Promise<AgentSession> {
  const response = await fetch(`${API_BASE}/${sessionId}/export`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to export session' }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Import session from backup.
 */
export async function importSession(
  sessionData: AgentSession,
  newSessionId?: string
): Promise<AgentSession> {
  const response = await fetch(`${API_BASE}/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_data: sessionData, new_session_id: newSessionId }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to import session' }));
    throw new Error(error.detail);
  }

  return response.json();
}

// ============================================================================
// Svelte Store
// ============================================================================

interface SessionStore {
  sessions: SessionSummary[];
  currentSession: AgentSession | null;
  stats: SessionStats | null;
  loading: boolean;
  error: string | null;
  total: number;
  limit: number;
  offset: number;
}

function createSessionStore() {
  const { subscribe, set, update }: Writable<SessionStore> = writable({
    sessions: [],
    currentSession: null,
    stats: null,
    loading: false,
    error: null,
    total: 0,
    limit: 50,
    offset: 0,
  });

  return {
    subscribe,

    // Load sessions with optional filtering
    loadSessions: async (agentType?: string, status?: string) => {
      update(s => ({ ...s, loading: true, error: null }));
      try {
        const result = await listSessions(agentType, status, 50, 0);
        update(s => ({
          ...s,
          sessions: result.sessions,
          total: result.total,
          limit: result.limit,
          offset: result.offset,
          loading: false,
        }));
      } catch (e) {
        update(s => ({
          ...s,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to load sessions',
        }));
      }
    },

    // Load more sessions (pagination)
    loadMore: async (agentType?: string, status?: string) => {
      const currentState = get({ subscribe });
      update(s => ({ ...s, loading: true, error: null }));
      try {
        const result = await listSessions(agentType, status, 50, currentState.offset + 50);
        update(s => ({
          ...s,
          sessions: [...s.sessions, ...result.sessions],
          offset: result.offset,
          loading: false,
        }));
      } catch (e) {
        update(s => ({
          ...s,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to load more sessions',
        }));
      }
    },

    // Load stats
    loadStats: async () => {
      try {
        const stats = await getSessionStats();
        update(s => ({ ...s, stats }));
      } catch (e) {
        console.error('Failed to load session stats:', e);
      }
    },

    // Load a specific session
    loadSession: async (sessionId: string, includeHistory: boolean = true) => {
      update(s => ({ ...s, loading: true, error: null }));
      try {
        const session = await getSession(sessionId, includeHistory);
        update(s => ({ ...s, currentSession: session, loading: false }));
      } catch (e) {
        update(s => ({
          ...s,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to load session',
        }));
      }
    },

    // Create a new session
    createSession: async (request: CreateSessionRequest) => {
      update(s => ({ ...s, loading: true, error: null }));
      try {
        const session = await createSession(request);
        update(s => ({
          ...s,
          currentSession: session,
          sessions: [session, ...s.sessions],
          total: s.total + 1,
          loading: false,
        }));
        return session;
      } catch (e) {
        update(s => ({
          ...s,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to create session',
        }));
        throw e;
      }
    },

    // Update current session
    updateSession: async (sessionId: string, updates: Parameters<typeof updateSession>[1]) => {
      update(s => ({ ...s, loading: true, error: null }));
      try {
        const result = await updateSession(sessionId, updates);
        if (result.success) {
          // Reload the session to get updated data
          await getSessionStore().loadSession(sessionId);
        }
        update(s => ({ ...s, loading: false }));
      } catch (e) {
        update(s => ({
          ...s,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to update session',
        }));
        throw e;
      }
    },

    // Delete a session
    deleteSession: async (sessionId: string) => {
      update(s => ({ ...s, loading: true, error: null }));
      try {
        await deleteSession(sessionId);
        update(s => ({
          ...s,
          sessions: s.sessions.filter(sess => sess.session_id !== sessionId),
          currentSession: s.currentSession?.session_id === sessionId ? null : s.currentSession,
          total: s.total - 1,
          loading: false,
        }));
      } catch (e) {
        update(s => ({
          ...s,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to delete session',
        }));
        throw e;
      }
    },

    // Add message to current session
    addMessage: async (sessionId: string, role: string, content: string) => {
      try {
        const result = await addMessage(sessionId, role, content);
        // Reload session to get updated history
        await getSessionStore().loadSession(sessionId);
        return result;
      } catch (e) {
        throw e;
      }
    },

    // Export session
    exportSession: async (sessionId: string) => {
      try {
        return await exportSession(sessionId);
      } catch (e) {
        throw e;
      }
    },

    // Import session
    importSession: async (sessionData: AgentSession, newSessionId?: string) => {
      update(s => ({ ...s, loading: true, error: null }));
      try {
        const session = await importSession(sessionData, newSessionId);
        update(s => ({
          ...s,
          sessions: [session, ...s.sessions],
          total: s.total + 1,
          loading: false,
        }));
        return session;
      } catch (e) {
        update(s => ({
          ...s,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to import session',
        }));
        throw e;
      }
    },

    // Clear current session
    clearCurrentSession: () => {
      update(s => ({ ...s, currentSession: null }));
    },

    // Clear error
    clearError: () => {
      update(s => ({ ...s, error: null }));
    },

    // Reset store
    reset: () => {
      set({
        sessions: [],
        currentSession: null,
        stats: null,
        loading: false,
        error: null,
        total: 0,
        limit: 50,
        offset: 0,
      });
    },
  };
}

export const sessionStore = createSessionStore();

// Export derived stores
export const sessions = writable<SessionSummary[]>([]);
export const currentSession = writable<AgentSession | null>(null);
export const sessionStats = writable<SessionStats | null>(null);
export const sessionLoading = writable(false);
export const sessionError = writable<string | null>(null);

// Subscribe to main store
sessionStore.subscribe(state => {
  sessions.set(state.sessions);
  currentSession.set(state.currentSession);
  sessionStats.set(state.stats);
  sessionLoading.set(state.loading);
  sessionError.set(state.error);
});

// Re-export store methods
export const {
  loadSessions,
  loadMore,
  loadStats,
  loadSession,
  createSession: createNewSession,
  updateSession: updateExistingSession,
  deleteSession: deleteExistingSession,
  addMessage: addSessionMessage,
  exportSession: exportExistingSession,
  importSession: importExistingSession,
  clearCurrentSession,
  clearError,
  reset: resetSessionStore,
} = sessionStore;
