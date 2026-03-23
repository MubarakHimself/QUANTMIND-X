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
  stream?: boolean;
}

export interface ChatMessageResponse {
  session_id: string;
  message_id: string;
  reply: string;
  artifacts: unknown[];
  action_taken?: string;
  delegation?: string;
}

export const chatApi = {
  async createSession(data: CreateSessionRequest): Promise<ChatSession> {
    const response = await fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
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

  async deleteSession(id: string): Promise<void> {
    const response = await fetch(`${API_BASE}/sessions/${id}`, {
      method: 'DELETE',
      credentials: 'include'
    });
    if (!response.ok) {
      throw new Error(`Failed to delete session: ${response.statusText}`);
    }
  },

  async sendMessage(
    endpoint: 'workshop' | 'floor-manager' | 'department',
    message: string,
    sessionId?: string,
    stream = false,
    dept?: string
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
      body: JSON.stringify({ message, session_id: sessionId, stream })
    });
    if (!response.ok) {
      throw new Error(`Failed to send message: ${response.statusText}`);
    }
    return response.json();
  }
};
