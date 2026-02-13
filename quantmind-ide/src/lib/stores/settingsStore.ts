// Settings Store - Manages application settings with Svelte stores
import { writable, derived, get } from 'svelte/store';
import type { Readable, Writable } from 'svelte/store';
import { settingsManager } from '$lib/services/settingsManager';

// Re-export types from settingsTypes
export type {
  AgentType,
  GeneralSettings,
  APIKeys,
  MCPServer,
  Skill,
  AgentSkills,
  Rule,
  MemoryConfig,
  MemoryStats,
  WorkflowStep,
  Workflow,
  AgentPermissions,
  SettingsStoreState
} from './settingsTypes';

// Import types for internal use
import type {
  AgentType,
  GeneralSettings,
  APIKeys,
  MCPServer,
  Skill,
  AgentSkills,
  Rule,
  MemoryConfig,
  Workflow,
  AgentPermissions,
  SettingsStoreState
} from './settingsTypes';

// Default settings
const defaultGeneralSettings: GeneralSettings = {
  theme: 'dark',
  fontSize: 'medium',
  language: 'en',
  autoSave: true,
  autoSaveInterval: 30,
  sendOnEnter: true,
  showTimestamps: true,
  compactMode: false,
  notifications: true,
  soundEffects: false
};

const defaultAPIKeys: APIKeys = {
  google: '',
  anthropic: '',
  openai: '',
  qwen: ''
};

const defaultMemoryConfig: MemoryConfig = {
  semanticEnabled: true,
  episodicEnabled: true,
  proceduralEnabled: true,
  maxEntries: 1000,
  retentionDays: 90
};

const defaultPermissions: AgentPermissions = {
  fileSystem: 'read',
  broker: 'read',
  database: 'read',
  external: false,
  memory: true,
  customRules: []
};

const defaultSkills: AgentSkills = {
  skills: [
    { id: 'code-gen', name: 'Code Generation', description: 'Generate code snippets', category: 'core', enabled: true },
    { id: 'analysis', name: 'Strategy Analysis', description: 'Analyze trading strategies', category: 'core', enabled: true },
    { id: 'backtest', name: 'Backtesting', description: 'Run strategy backtests', category: 'core', enabled: true },
    { id: 'debug', name: 'Debugging', description: 'Debug code issues', category: 'advanced', enabled: false },
    { id: 'optimize', name: 'Optimization', description: 'Optimize strategy parameters', category: 'advanced', enabled: false }
  ],
  lastUpdated: new Date()
};

// Initial state
const initialState: SettingsStoreState = {
  general: defaultGeneralSettings,
  apiKeys: defaultAPIKeys,
  mcpServers: [],
  skills: {
    copilot: { ...defaultSkills },
    quantcode: { ...defaultSkills },
    analyst: { ...defaultSkills }
  },
  rules: [],
  memories: defaultMemoryConfig,
  workflows: [],
  permissions: {
    copilot: { ...defaultPermissions },
    quantcode: { ...defaultPermissions },
    analyst: { ...defaultPermissions }
  },
  isLoading: false,
  isDirty: false,
  error: null
};

// Create the main store
function createSettingsStore() {
  const { subscribe, set, update }: Writable<SettingsStoreState> = writable(initialState);

  return {
    subscribe,
    
    // Initialize store from storage
    async initialize() {
      update(state => ({ ...state, isLoading: true }));
      try {
        const settings = await settingsManager.loadSettings();
        set({
          ...initialState,
          ...settings,
          isLoading: false,
          isDirty: false
        });
      } catch (error) {
        update(state => ({
          ...state,
          isLoading: false,
          error: 'Failed to load settings'
        }));
      }
    },

    // Update general settings
    updateGeneral(updates: Partial<GeneralSettings>) {
      update(state => ({
        ...state,
        general: { ...state.general, ...updates },
        isDirty: true
      }));
    },

    // Update API keys
    updateAPIKeys(updates: Partial<APIKeys>) {
      update(state => ({
        ...state,
        apiKeys: { ...state.apiKeys, ...updates },
        isDirty: true
      }));
    },

    // Add MCP server
    addMCPServer(server: Omit<MCPServer, 'id'>) {
      update(state => {
        const newServer: MCPServer = {
          ...server,
          id: `mcp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        };
        return {
          ...state,
          mcpServers: [...state.mcpServers, newServer],
          isDirty: true
        };
      });
    },

    // Update MCP server
    updateMCPServer(serverId: string, updates: Partial<MCPServer>) {
      update(state => ({
        ...state,
        mcpServers: state.mcpServers.map(s =>
          s.id === serverId ? { ...s, ...updates } : s
        ),
        isDirty: true
      }));
    },

    // Remove MCP server
    removeMCPServer(serverId: string) {
      update(state => ({
        ...state,
        mcpServers: state.mcpServers.filter(s => s.id !== serverId),
        isDirty: true
      }));
    },

    // Update agent skills
    updateAgentSkills(agent: AgentType, skills: Skill[]) {
      update(state => ({
        ...state,
        skills: {
          ...state.skills,
          [agent]: {
            skills,
            lastUpdated: new Date()
          }
        },
        isDirty: true
      }));
    },

    // Toggle skill
    toggleSkill(agent: AgentType, skillId: string) {
      update(state => {
        const agentSkills = state.skills[agent];
        const updatedSkills = agentSkills.skills.map(s =>
          s.id === skillId ? { ...s, enabled: !s.enabled } : s
        );
        return {
          ...state,
          skills: {
            ...state.skills,
            [agent]: {
              skills: updatedSkills,
              lastUpdated: new Date()
            }
          },
          isDirty: true
        };
      });
    },

    // Add rule
    addRule(rule: Omit<Rule, 'id' | 'createdAt' | 'updatedAt'>) {
      update(state => {
        const newRule: Rule = {
          ...rule,
          id: `rule_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          createdAt: new Date(),
          updatedAt: new Date()
        };
        return {
          ...state,
          rules: [...state.rules, newRule],
          isDirty: true
        };
      });
    },

    // Update rule
    updateRule(ruleId: string, updates: Partial<Rule>) {
      update(state => ({
        ...state,
        rules: state.rules.map(r =>
          r.id === ruleId ? { ...r, ...updates, updatedAt: new Date() } : r
        ),
        isDirty: true
      }));
    },

    // Remove rule
    removeRule(ruleId: string) {
      update(state => ({
        ...state,
        rules: state.rules.filter(r => r.id !== ruleId),
        isDirty: true
      }));
    },

    // Update memory config
    updateMemoryConfig(updates: Partial<MemoryConfig>) {
      update(state => ({
        ...state,
        memories: { ...state.memories, ...updates },
        isDirty: true
      }));
    },

    // Add workflow
    addWorkflow(workflow: Omit<Workflow, 'id' | 'createdAt'>) {
      update(state => {
        const newWorkflow: Workflow = {
          ...workflow,
          id: `wf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          createdAt: new Date()
        };
        return {
          ...state,
          workflows: [...state.workflows, newWorkflow],
          isDirty: true
        };
      });
    },

    // Update workflow
    updateWorkflow(workflowId: string, updates: Partial<Workflow>) {
      update(state => ({
        ...state,
        workflows: state.workflows.map(w =>
          w.id === workflowId ? { ...w, ...updates } : w
        ),
        isDirty: true
      }));
    },

    // Remove workflow
    removeWorkflow(workflowId: string) {
      update(state => ({
        ...state,
        workflows: state.workflows.filter(w => w.id !== workflowId),
        isDirty: true
      }));
    },

    // Update agent permissions
    updateAgentPermissions(agent: AgentType, updates: Partial<AgentPermissions>) {
      update(state => ({
        ...state,
        permissions: {
          ...state.permissions,
          [agent]: { ...state.permissions[agent], ...updates }
        },
        isDirty: true
      }));
    },

    // Save all settings
    async save() {
      const state = get({ subscribe });
      try {
        await settingsManager.saveSettings(state);
        update(s => ({ ...s, isDirty: false }));
      } catch (error) {
        update(s => ({ ...s, error: 'Failed to save settings' }));
        throw error;
      }
    },

    // Reset to defaults
    resetToDefaults() {
      set({
        ...initialState,
        isLoading: false,
        isDirty: true
      });
    },

    // Clear error
    clearError() {
      update(state => ({ ...state, error: null }));
    },

    // Export settings
    exportSettings(): string {
      const state = get({ subscribe });
      return JSON.stringify({
        general: state.general,
        mcpServers: state.mcpServers,
        skills: state.skills,
        rules: state.rules,
        memories: state.memories,
        workflows: state.workflows,
        permissions: state.permissions
        // Note: API keys excluded for security
      }, null, 2);
    },

    // Import settings
    importSettings(jsonData: string) {
      try {
        const imported = JSON.parse(jsonData);
        update(state => ({
          ...state,
          ...imported,
          isDirty: true
        }));
        return true;
      } catch (error) {
        update(state => ({
          ...state,
          error: 'Failed to import settings'
        }));
        return false;
      }
    }
  };
}

// Export the store instance
export const settingsStore = createSettingsStore();

// Derived stores for computed values
export const connectedMCPServers: Readable<MCPServer[]> = derived(
  settingsStore,
  $store => $store.mcpServers.filter(s => s.status === 'connected')
);

export const enabledRules: Readable<Rule[]> = derived(
  settingsStore,
  $store => $store.rules.filter(r => r.enabled).sort((a, b) => b.priority - a.priority)
);

export const workflowTemplates: Readable<Workflow[]> = derived(
  settingsStore,
  $store => $store.workflows.filter(w => w.isTemplate)
);

// Permission presets
export const permissionPresets = {
  restricted: {
    fileSystem: 'none' as const,
    broker: 'none' as const,
    database: 'none' as const,
    external: false,
    memory: false,
    customRules: []
  },
  standard: {
    fileSystem: 'read' as const,
    broker: 'read' as const,
    database: 'read' as const,
    external: false,
    memory: true,
    customRules: []
  },
  fullAccess: {
    fileSystem: 'full' as const,
    broker: 'full' as const,
    database: 'full' as const,
    external: true,
    memory: true,
    customRules: []
  }
};
