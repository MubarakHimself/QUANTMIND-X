// Settings Types - Shared type definitions for settings
export type AgentType = 'copilot' | 'quantcode' | 'analyst' | 'research' | 'development' | 'trading' | 'risk' | 'portfolio' | 'floor_manager';

export type ModelProvider = 'anthropic' | 'zhipu' | 'minimax' | 'openai' | 'openrouter' | 'deepseek';

export interface ModelConfig {
  provider: ModelProvider;
  baseUrl: string;
  apiKey: string;
  model: string;
  availableModels: Array<{ id: string; name: string; tier: string }>;
}

export interface GeneralSettings {
  theme: 'dark' | 'light' | 'system';
  fontSize: 'small' | 'medium' | 'large';
  language: string;
  autoSave: boolean;
  autoSaveInterval: number;
  sendOnEnter: boolean;
  showTimestamps: boolean;
  compactMode: boolean;
  notifications: boolean;
  soundEffects: boolean;
  emailAlerts: boolean;
  webhookUrl: string;
  notificationFrequency: 'immediate' | 'daily' | 'weekly';
}

export interface APIKeys {
  google: string;
  anthropic: string;
  openai: string;
  qwen: string;
  zhipu: string;
  minimax: string;
}

export interface MCPServer {
  id: string;
  name: string;
  description?: string;
  // Server transport type: 'stdio' (local command), 'http' (remote HTTP), 'sse' (Server-Sent Events)
  type: 'stdio' | 'http' | 'sse';
  // For HTTP and SSE servers
  url?: string;
  headers?: Record<string, string>;
  // For Stdio servers
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  // Status
  status: 'connected' | 'disconnected' | 'error';
  capabilities: string[];
  lastConnected?: Date;
  autoConnect: boolean;
}

export interface Skill {
  id: string;
  name: string;
  description: string;
  category: 'core' | 'advanced' | 'custom';
  enabled: boolean;
  icon?: string;
  departments?: string[];
}

export interface AgentSkills {
  skills: Skill[];
  lastUpdated: Date;
}

export interface Rule {
  id: string;
  name: string;
  content: string;
  priority: number;
  enabled: boolean;
  agent?: AgentType;
  createdAt: Date;
  updatedAt: Date;
}

export interface MemoryConfig {
  semanticEnabled: boolean;
  episodicEnabled: boolean;
  proceduralEnabled: boolean;
  maxEntries: number;
  retentionDays: number;
}

export interface MemoryStats {
  semantic: number;
  episodic: number;
  procedural: number;
  total: number;
}

export interface WorkflowStep {
  id: string;
  name: string;
  action: string;
  params: Record<string, unknown>;
  order: number;
}

export interface Workflow {
  id: string;
  name: string;
  description: string;
  steps: WorkflowStep[];
  category: string;
  isTemplate: boolean;
  createdAt: Date;
  lastRun?: Date;
}

export interface AgentPermissions {
  fileSystem: 'none' | 'read' | 'write' | 'full';
  broker: 'none' | 'read' | 'trade' | 'full';
  database: 'none' | 'read' | 'write' | 'full';
  external: boolean;
  memory: boolean;
  customRules: string[];
}

export interface ConnectionSettings {
  t1Url: string;  // Cloudzy — trading backend (Kamatera T1)
  t2Url: string;  // Contabo — HMM/research backend (Kamatera T2)
}

// Settings state interface
export interface SettingsStoreState {
  general: GeneralSettings;
  apiKeys: APIKeys;
  mcpServers: MCPServer[];
  skills: Record<AgentType, AgentSkills>;
  rules: Rule[];
  memories: MemoryConfig;
  workflows: Workflow[];
  permissions: Record<AgentType, AgentPermissions>;
  modelConfig: ModelConfig;
  connections: ConnectionSettings;
  isLoading: boolean;
  isDirty: boolean;
  error: string | null;
}
