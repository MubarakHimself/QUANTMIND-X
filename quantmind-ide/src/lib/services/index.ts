// Services Export Index
export { chatManager } from './chatManager';
export { settingsManager } from './settingsManager';
export { contextManager } from './contextManager';
export { fileHistoryManager } from './fileHistoryManager';
export { commandHandler } from './commandHandler';

// Re-export types
export type { FileVersion, FileHistory, HistoryStats } from './fileHistoryManager';
export type { Command, CommandContext, CommandResult, ParsedCommand } from './commandHandler';
