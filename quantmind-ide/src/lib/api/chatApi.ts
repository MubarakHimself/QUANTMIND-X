const API_BASE = '/api/chat';

export interface CreateSessionRequest {
  agentType: 'workshop' | 'floor-manager' | 'department';
  agentId: string;
  userId: string;
  title?: string;
  context?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  agent_type: string;
  agent_id: string;
  title: string;
  user_id: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageRequest {
  message: string;
  session_id?: string;
  context?: Record<string, unknown>;
  history?: Array<{ role: string; content: string }>;
  stream?: boolean;
}

export interface ChatMessageResponse {
  session_id: string;
  message_id: string;
  reply: string;
  artifacts: unknown[];
  action_taken?: string;
  delegation?: string | Record<string, unknown>;
  type?: string;
  tool_calls?: Array<{ name: string; input: Record<string, unknown>; result?: string }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
}

export interface StoredChatMessage {
  id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  metadata?: Record<string, unknown>;
}

export interface UpdateSessionRequest {
  title: string;
}

export const chatApi = {
  async createSession(data: CreateSessionRequest): Promise<ChatSession> {
    const response = await fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent_type: data.agentType,
        agent_id: data.agentId,
        user_id: data.userId,
        title: data.title,
        context: data.context,
        metadata: data.metadata,
      })
    });
    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }
    return response.json();
  },

  async getSession(id: string): Promise<ChatSession> {
    const response = await fetch(`${API_BASE}/sessions/${id}`, {
      credentials: 'include'
    });
    if (!response.ok) {
      throw new Error(`Failed to get session: ${response.statusText}`);
    }
    return response.json();
  },

  async listSessions(userId?: string, agentType?: string): Promise<ChatSession[]> {
    const params = new URLSearchParams();
    if (userId) params.set('user_id', userId);
    if (agentType) params.set('agent_type', agentType);

    const response = await fetch(`${API_BASE}/sessions?${params}`, {
      credentials: 'include'
    });
    if (!response.ok) {
      throw new Error(`Failed to list sessions: ${response.statusText}`);
    }
    return response.json();
  },

  async getSessionMessages(sessionId: string): Promise<StoredChatMessage[]> {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/messages`, {
      credentials: 'include'
    });
    if (!response.ok) {
      throw new Error(`Failed to get session messages: ${response.statusText}`);
    }
    return response.json();
  },

  async deleteSession(id: string): Promise<void> {
    const response = await fetch(`${API_BASE}/sessions/${id}`, {
      method: 'DELETE',
      credentials: 'include'
    });
    if (!response.ok) {
      throw new Error(`Failed to delete session: ${response.statusText}`);
    }
  },

  async updateSessionTitle(id: string, data: UpdateSessionRequest): Promise<ChatSession> {
    const response = await fetch(`${API_BASE}/sessions/${id}`, {
      method: 'PATCH',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: data.title })
    });
    if (!response.ok) {
      throw new Error(`Failed to update session: ${response.statusText}`);
    }
    return response.json();
  },

  async sendMessage(
    endpoint: 'workshop' | 'floor-manager' | 'department',
    message: string,
    sessionId?: string,
    stream = false,
    dept?: string,
    context?: Record<string, unknown>,
    history?: Array<{ role: string; content: string }>
  ): Promise<ChatMessageResponse> {
    // Department messages route to /api/chat/departments/{dept}/message
    // All other endpoints route to /api/chat/{endpoint}/message
    const path = endpoint === 'department'
      ? `/departments/${dept}/message`
      : `/${endpoint}/message`;
    const response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId, context, history, stream })
    });
    if (!response.ok) {
      throw new Error(`Failed to send message: ${response.statusText}`);
    }
    return response.json();
  }
};
