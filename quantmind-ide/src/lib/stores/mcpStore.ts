/**
 * MCP Store
 *
 * Manages MCP server connections and tool availability.
 */

import { writable, derived, get } from 'svelte/store';
import type { Readable, Writable } from 'svelte/store';

// Types
export interface MCPServer {
  server_id: string;
  name: string;
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
  connected_at?: string;
  tools_count: number;
  last_error?: string;
}

export interface MCPTool {
  name: string;
  server_id: string;
  description: string;
  input_schema: Record<string, any>;
}

export interface MCPServerStatus {
  server_id: string;
  status: string;
  healthy: boolean;
}

// State
interface MCPState {
  servers: MCPServer[];
  tools: MCPTool[];
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: MCPState = {
  servers: [],
  tools: [],
  loading: false,
  error: null,
  lastUpdated: null,
};

// Create stores
const mcpState = writable<MCPState>(initialState);

// Derived stores
export const servers: Readable<MCPServer[]> = derived(mcpState, ($state) => $state.servers);
export const tools: Readable<MCPTool[]> = derived(mcpState, ($state) => $state.tools);
export const connectedServers: Readable<MCPServer[]> = derived(mcpState, ($state) =>
  $state.servers.filter((s) => s.status === 'connected')
);
export const loading: Readable<boolean> = derived(mcpState, ($state) => $state.loading);
export const error: Readable<string | null> = derived(mcpState, ($state) => $state.error);

// Actions
export const mcpStore = {
  subscribe: mcpState.subscribe,

  /**
   * Fetch all MCP servers
   */
  async fetchServers(includeTools = false): Promise<void> {
    mcpState.update((s) => ({ ...s, loading: true, error: null }));

    try {
      const response = await fetch(`/api/mcp/servers?include_tools=${includeTools}`);
      if (!response.ok) throw new Error('Failed to fetch servers');

      const data = await response.json();

      mcpState.update((s) => ({
        ...s,
        servers: data.servers,
        tools: data.servers.flatMap((s: any) => s.tools || []),
        loading: false,
        lastUpdated: new Date().toISOString(),
      }));
    } catch (err) {
      mcpState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Unknown error',
      }));
    }
  },

  /**
   * Connect to an MCP server
   */
  async connect(serverId: string, config?: any): Promise<boolean> {
    try {
      const response = await fetch(`/api/mcp/servers/${serverId}/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
      });

      if (!response.ok) throw new Error('Failed to connect');

      // Refresh servers
      await this.fetchServers();
      return true;
    } catch (err) {
      mcpState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Connection failed',
      }));
      return false;
    }
  },

  /**
   * Disconnect from an MCP server
   */
  async disconnect(serverId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/mcp/servers/${serverId}/disconnect`, {
        method: 'POST',
      });

      if (!response.ok) throw new Error('Failed to disconnect');

      // Update local state
      mcpState.update((s) => ({
        ...s,
        servers: s.servers.map((server) =>
          server.server_id === serverId
            ? { ...server, status: 'disconnected' as const }
            : server
        ),
      }));

      return true;
    } catch (err) {
      mcpState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Disconnection failed',
      }));
      return false;
    }
  },

  /**
   * Get tools for a specific server
   */
  async fetchServerTools(serverId: string): Promise<MCPTool[]> {
    try {
      const response = await fetch(`/api/mcp/servers/${serverId}/tools`);
      if (!response.ok) throw new Error('Failed to fetch tools');

      const data = await response.json();
      return data.tools;
    } catch (err) {
      console.error('Failed to fetch tools:', err);
      return [];
    }
  },

  /**
   * Call an MCP tool
   */
  async callTool(serverId: string, toolName: string, args: Record<string, any>): Promise<any> {
    try {
      const response = await fetch(`/api/mcp/call-tool?server_id=${serverId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_name: toolName,
          arguments: args,
        }),
      });

      if (!response.ok) throw new Error('Tool call failed');

      const data = await response.json();
      return data.result;
    } catch (err) {
      mcpState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Tool call failed',
      }));
      throw err;
    }
  },

  /**
   * Check health of all servers
   */
  async checkHealth(): Promise<{ healthy: boolean; servers: MCPServerStatus[] }> {
    try {
      const response = await fetch('/api/mcp/health');
      if (!response.ok) throw new Error('Health check failed');

      return await response.json();
    } catch (err) {
      return { healthy: false, servers: [] };
    }
  },

  /**
   * Get server by ID
   */
  getServer(serverId: string): MCPServer | undefined {
    const state = get(mcpState);
    return state.servers.find((s) => s.server_id === serverId);
  },

  /**
   * Get tools by server
   */
  getToolsByServer(serverId: string): MCPTool[] {
    const state = get(mcpState);
    return state.tools.filter((t) => t.server_id === serverId);
  },

  /**
   * Clear error
   */
  clearError(): void {
    mcpState.update((s) => ({ ...s, error: null }));
  },

  /**
   * Reset store
   */
  reset(): void {
    mcpState.set(initialState);
  },
};

export default mcpStore;
