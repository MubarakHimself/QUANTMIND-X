// Services Export Index
export { chatManager } from './chatManager';
export { settingsManager } from './settingsManager';
export { contextManager } from './contextManager';
export { fileHistoryManager } from './fileHistoryManager';
export { commandHandler } from './commandHandler';
export { agentStreamService, connectAgentStream, disconnectAgentStream, connectTaskStream, disconnectTaskStream } from './agentStreamService';
export { batchService } from './batchService';

// Re-export types
export type { FileVersion, FileHistory, HistoryStats } from './fileHistoryManager';
export type { Command, CommandContext, CommandResult, ParsedCommand } from './commandHandler';
export type { AgentStreamEvent, AgentStreamEventType, AgentTaskStreamEvent } from './agentStreamService';
export type {
  BatchSubmitRequest,
  BatchSubmitResponse,
  BatchStatusResponse,
  BatchItemStatusResponse,
  BatchResultResponse,
  BatchStatsResponse,
  BatchListItem,
} from './batchService';
