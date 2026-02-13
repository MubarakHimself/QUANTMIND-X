// Stores Export Index
export { 
  chatStore, 
  activeChat, 
  activeMessages, 
  activeContext, 
  pinnedChats, 
  unpinnedChats, 
  agentChats,
  agentGreetings 
} from './chatStore';

export { 
  settingsStore, 
  connectedMCPServers, 
  enabledRules, 
  workflowTemplates, 
  permissionPresets 
} from './settingsStore';

// Re-export types from chatStore
export type {
  Chat,
  Message,
  ChatContext,
  FileReference,
  StrategyReference,
  BrokerReference,
  BacktestReference,
  AgentType
} from './chatStore';

// Re-export types from settingsTypes (via settingsStore)
export type {
  SettingsStoreState,
  GeneralSettings,
  APIKeys,
  MCPServer,
  Skill,
  AgentSkills,
  Rule,
  MemoryConfig,
  MemoryStats,
  Workflow,
  WorkflowStep,
  AgentPermissions
} from './settingsTypes';
