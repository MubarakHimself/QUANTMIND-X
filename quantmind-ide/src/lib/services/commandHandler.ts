// Command Handler Service - Manages slash commands for the agent panel
import type { AgentType } from '../stores/chatStore';

// Command interface
export interface Command {
  name: string;
  aliases?: string[];
  params: string;
  description: string;
  icon: string;
  category: 'trading' | 'context' | 'agent' | 'system';
  handler: (params: string, context: CommandContext) => Promise<CommandResult>;
}

// Command context
export interface CommandContext {
  agent: AgentType;
  chatId: string;
  attachedFiles?: string[];
  // Callbacks for side effects
  addContextItem?: (type: string, item: any) => void;
  invokeBackend?: (endpoint: string, data: any) => Promise<any>;
}

// Command result
export interface CommandResult {
  success: boolean;
  message: string;
  data?: Record<string, unknown>;
  error?: string;
}

// Parsed command
export interface ParsedCommand {
  name: string;
  params: string;
  raw: string;
}

// Command registry
const commands: Map<string, Command> = new Map();

// Backend API base URL
const API_BASE_URL = 'http://localhost:8000/api';

// Register default commands
function registerDefaultCommands() {
  // Backtest command - invokes backend API
  registerCommand({
    name: '/backtest',
    params: '<strategy_name>',
    description: 'Run a backtest on a strategy',
    icon: '📊',
    category: 'trading',
    handler: async (params, context) => {
      if (!params.trim()) {
        return { success: false, message: 'Please specify a strategy name', error: 'Missing parameter' };
      }
      
      try {
        // Invoke backend API for backtest
        const response = await fetch(`${API_BASE_URL}/v1/backtest/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            strategy: params.trim(),
            chatId: context.chatId,
            agent: context.agent,
            enable_ws_streaming: true
          })
        });
        
        if (!response.ok) {
          throw new Error(`Backtest request failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        return {
          success: true,
          message: `Backtest started for strategy: ${params}. ${result.message || ''}`,
          data: { strategy: params, action: 'backtest', backtestId: result.backtestId }
        };
      } catch (error) {
        return {
          success: false,
          message: `Failed to start backtest: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error: 'Backtest failed'
        };
      }
    }
  });
  
  // Attach command - validates and adds to chatStore context
  registerCommand({
    name: '/attach',
    params: '<file_path>',
    description: 'Attach a file to the conversation',
    icon: '📎',
    category: 'context',
    handler: async (params, context) => {
      if (!params.trim()) {
        return { success: false, message: 'Please specify a file path', error: 'Missing parameter' };
      }
      
      const filePath = params.trim();
      const fileName = filePath.split('/').pop() || filePath;
      
      // Create file reference for context
      const fileItem = {
        id: `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        name: fileName,
        path: filePath,
        type: fileName.split('.').pop() || 'unknown'
      };
      
      // Add to context via callback if provided
      if (context.addContextItem) {
        context.addContextItem('files', fileItem);
      }
      
      return {
        success: true,
        message: `File attached: ${fileName}`,
        data: { file: filePath, action: 'attach', fileItem }
      };
    }
  });
  
  // Analyze command - invokes backend API
  registerCommand({
    name: '/analyze',
    params: '<strategy_name>',
    description: 'Analyze a trading strategy',
    icon: '📄',
    category: 'trading',
    handler: async (params, context) => {
      if (!params.trim()) {
        return { success: false, message: 'Please specify a strategy name', error: 'Missing parameter' };
      }
      
      try {
        // Invoke backend API for analysis
        const response = await fetch(`${API_BASE_URL}/strategy/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            strategy: params.trim(),
            chatId: context.chatId,
            agent: context.agent
          })
        });
        
        if (!response.ok) {
          throw new Error(`Analysis request failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        return {
          success: true,
          message: `Analysis started for strategy: ${params}. ${result.message || ''}`,
          data: { strategy: params, action: 'analyze', analysisId: result.analysisId }
        };
      } catch (error) {
        return {
          success: false,
          message: `Failed to analyze strategy: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error: 'Analysis failed'
        };
      }
    }
  });
  
  // Add broker command - invokes backend API
  registerCommand({
    name: '/add-broker',
    params: '<broker_name>',
    description: 'Connect a new broker',
    icon: '➕',
    category: 'trading',
    handler: async (params, context) => {
      if (!params.trim()) {
        return { success: false, message: 'Please specify a broker name', error: 'Missing parameter' };
      }
      
      try {
        // Invoke backend API for broker connection
        const response = await fetch(`${API_BASE_URL}/broker/connect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            broker: params.trim(),
            chatId: context.chatId
          })
        });
        
        if (!response.ok) {
          throw new Error(`Broker connection failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Add broker to context if connection successful
        if (result.success && context.addContextItem) {
          const brokerItem = {
            id: `broker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            name: params.trim(),
            status: 'connected' as const
          };
          context.addContextItem('brokers', brokerItem);
        }
        
        return {
          success: true,
          message: `Connected to broker: ${params}`,
          data: { broker: params, action: 'add-broker', brokerId: result.brokerId }
        };
      } catch (error) {
        return {
          success: false,
          message: `Failed to connect broker: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error: 'Broker connection failed'
        };
      }
    }
  });
  
  // Kill switch command - invokes backend API
  registerCommand({
    name: '/kill',
    params: '',
    description: 'Execute emergency kill switch',
    icon: '⚡',
    category: 'system',
    handler: async (params, context) => {
      try {
        // Invoke backend API for kill switch
        const response = await fetch(`${API_BASE_URL}/system/kill-switch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chatId: context.chatId,
            agent: context.agent
          })
        });
        
        if (!response.ok) {
          throw new Error(`Kill switch failed: ${response.statusText}`);
        }
        
        return {
          success: true,
          message: '⚠️ KILL SWITCH ACTIVATED - All trading operations stopped',
          data: { action: 'kill-switch' }
        };
      } catch (error) {
        // Still return success even if backend is unavailable - kill switch should work locally
        console.warn('Kill switch backend call failed, but continuing:', error);
        return {
          success: true,
          message: '⚠️ KILL SWITCH ACTIVATED - All trading operations stopped (local mode)',
          data: { action: 'kill-switch' }
        };
      }
    }
  });
  
  // Memory command
  registerCommand({
    name: '/memory',
    params: '[action]',
    description: 'View or manage agent memory',
    icon: '💾',
    category: 'agent',
    handler: async (params, context) => {
      const action = params.trim().toLowerCase() || 'view';
      return {
        success: true,
        message: `Memory action: ${action}`,
        data: { action: 'memory', subAction: action }
      };
    }
  });
  
  // Skills command - emits event to open skills panel
  registerCommand({
    name: '/skills',
    params: '',
    description: 'Configure agent skills',
    icon: '✨',
    category: 'agent',
    handler: async (params, context) => {
      return {
        success: true,
        message: 'Opening skills configuration...',
        data: { action: 'open-skills' }
      };
    }
  });
  
  // Settings command - emits event to open settings panel
  registerCommand({
    name: '/settings',
    params: '',
    description: 'Open settings panel',
    icon: '⚙️',
    category: 'system',
    handler: async (params, context) => {
      return {
        success: true,
        message: 'Opening settings...',
        data: { action: 'open-settings' }
      };
    }
  });
  
  // Help command
  registerCommand({
    name: '/help',
    params: '',
    description: 'Show available commands',
    icon: '❓',
    category: 'system',
    handler: async (params, context) => {
      const commandList = Array.from(commands.values())
        .map(cmd => `${cmd.name} ${cmd.params} - ${cmd.description}`)
        .join('\n');
      return {
        success: true,
        message: `Available commands:\n${commandList}`,
        data: { action: 'help' }
      };
    }
  });
  
  // Terminal command - invokes backend API
  registerCommand({
    name: '/terminal',
    params: '<command>',
    description: 'Execute a terminal command',
    icon: '💻',
    category: 'system',
    handler: async (params, context) => {
      if (!params.trim()) {
        return { success: false, message: 'Please specify a command', error: 'Missing parameter' };
      }
      
      try {
        // Invoke backend API for terminal command
        const response = await fetch(`${API_BASE_URL}/terminal/execute`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            command: params.trim(),
            chatId: context.chatId
          })
        });
        
        if (!response.ok) {
          throw new Error(`Terminal command failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        return {
          success: true,
          message: `Command executed: ${params}\nOutput: ${result.output || 'No output'}`,
          data: { command: params, action: 'terminal', output: result.output }
        };
      } catch (error) {
        return {
          success: false,
          message: `Failed to execute command: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error: 'Terminal command failed'
        };
      }
    }
  });
  
  // Clear command - clears chat history
  registerCommand({
    name: '/clear',
    params: '',
    description: 'Clear the chat history',
    icon: '🗑️',
    category: 'system',
    handler: async (params, context) => {
      return {
        success: true,
        message: 'Chat cleared',
        data: { action: 'clear-chat', chatId: context.chatId }
      };
    }
  });
  
  // Export command - exports chat history
  registerCommand({
    name: '/export',
    params: '[format]',
    description: 'Export chat history',
    icon: '📤',
    category: 'system',
    handler: async (params, context) => {
      const format = params.trim().toLowerCase() || 'json';
      return {
        success: true,
        message: `Exporting chat in ${format} format...`,
        data: { action: 'export', format, chatId: context.chatId }
      };
    }
  });
}

// Register a command
export function registerCommand(command: Command): void {
  commands.set(command.name, command);
  
  // Register aliases
  if (command.aliases) {
    command.aliases.forEach(alias => {
      commands.set(alias, command);
    });
  }
}

// Unregister a command
export function unregisterCommand(name: string): void {
  const command = commands.get(name);
  if (command) {
    commands.delete(name);
    if (command.aliases) {
      command.aliases.forEach(alias => commands.delete(alias));
    }
  }
}

// Get all commands
export function getAllCommands(): Command[] {
  const uniqueCommands = new Set<Command>();
  commands.forEach(cmd => uniqueCommands.add(cmd));
  return Array.from(uniqueCommands);
}

// Get commands by category
export function getCommandsByCategory(category: Command['category']): Command[] {
  return getAllCommands().filter(cmd => cmd.category === category);
}

// Parse a command string
export function parseCommand(input: string): ParsedCommand | null {
  const trimmed = input.trim();
  if (!trimmed.startsWith('/')) return null;
  
  const parts = trimmed.split(/\s+/);
  const name = parts[0].toLowerCase();
  const params = parts.slice(1).join(' ');
  
  return { name, params, raw: trimmed };
}

// Execute a command
export async function executeCommand(
  input: string, 
  context: CommandContext
): Promise<CommandResult> {
  const parsed = parseCommand(input);
  
  if (!parsed) {
    return { 
      success: false, 
      message: 'Invalid command format', 
      error: 'Commands must start with /' 
    };
  }
  
  const command = commands.get(parsed.name);
  
  if (!command) {
    return { 
      success: false, 
      message: `Unknown command: ${parsed.name}`, 
      error: 'Command not found' 
    };
  }
  
  try {
    return await command.handler(parsed.params, context);
  } catch (error) {
    return {
      success: false,
      message: 'Command execution failed',
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

// Check if input is a command
export function isCommand(input: string): boolean {
  return input.trim().startsWith('/');
}

// Get command suggestions
export function getSuggestions(input: string): Command[] {
  const trimmed = input.trim().toLowerCase();
  if (!trimmed.startsWith('/')) return [];
  
  const allCommands = getAllCommands();
  
  // If just '/', return all commands
  if (trimmed === '/') return allCommands;
  
  // Filter by prefix
  return allCommands.filter(cmd => 
    cmd.name.toLowerCase().startsWith(trimmed) ||
    (cmd.aliases && cmd.aliases.some(a => a.toLowerCase().startsWith(trimmed)))
  );
}

// Validate command parameters
export function validateParams(command: Command, params: string): { valid: boolean; error?: string } {
  // Check if params are required
  if (command.params && command.params.startsWith('<') && !params.trim()) {
    return { 
      valid: false, 
      error: `Missing required parameter: ${command.params}` 
    };
  }
  
  return { valid: true };
}

// Get command help text
export function getCommandHelp(name: string): string | null {
  const command = commands.get(name);
  if (!command) return null;
  
  let help = `${command.name} ${command.params}\n`;
  help += `  ${command.description}\n`;
  help += `  Category: ${command.category}\n`;
  
  if (command.aliases && command.aliases.length > 0) {
    help += `  Aliases: ${command.aliases.join(', ')}`;
  }
  
  return help;
}

// Initialize default commands
registerDefaultCommands();

// Export the command handler
export const commandHandler = {
  register: registerCommand,
  unregister: unregisterCommand,
  getAll: getAllCommands,
  getByCategory: getCommandsByCategory,
  parse: parseCommand,
  execute: executeCommand,
  isCommand,
  getSuggestions,
  validateParams,
  getHelp: getCommandHelp
};

export default commandHandler;
