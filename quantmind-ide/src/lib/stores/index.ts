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

// Memory Management Stores
export { memoryStore, memories, filteredMemories, selectedMemory, memoryStats, memoryLoading, memoryError, memoryFilters } from './memoryStore';
export { cronStore, cronJobs, enabledCronJobs, cronLoading, cronError } from './cronStore';
export { hooksStore, hooks, enabledHooks, hookLogs, hooksLoading } from './hooksStore';

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

// Re-export types from memory management stores
export type { MemoryEntry, MemoryFilters } from './memoryStore';
export type { CronJob } from './cronStore';
export type { Hook, HookLogEntry } from './hooksStore';

// Re-export Trading Floor types and store
export { tradingFloorStore } from './tradingFloorStore';
export type {
  AgentState,
  DepartmentPosition,
  MailMessage,
  TradingFloorState
} from './tradingFloorStore';
