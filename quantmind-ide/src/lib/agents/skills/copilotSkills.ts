/**
 * Copilot Skills - General trading assistant capabilities
 *
 * These skills provide file operations, broker connections, EA deployment,
 * and MT5 synchronization capabilities for the Copilot agent.
 */

import { z } from 'zod';
import type { Skill, SkillContext } from './index';
import { fileHistoryManager } from '../../services/fileHistoryManager';

// ============================================================================
// FILE OPERATIONS SKILLS
// ============================================================================

/**
 * Read a file from the workspace
 */
const readFile: Skill = {
  id: 'copilot_read_file',
  name: 'Read File',
  description: 'Read the contents of a file from the QuantMindX workspace. Supports .mq5, .py, .json, .yaml, .md files.',
  agents: ['copilot'],
  category: 'file_operations',
  schema: z.object({
    path: z.string().describe('Relative path to the file from workspace root'),
    encoding: z.string().default('utf-8').describe('File encoding (default: utf-8)')
  }),
  examples: [
    {
      input: { path: 'strategies/my_strategy.mq5' },
      output: '// MQL5 strategy code...',
      description: 'Read an MQL5 strategy file'
    },
    {
      input: { path: 'config/trading_settings.json' },
      output: '{\n  "symbol": "EURUSD",\n  "risk": 0.02\n}',
      description: 'Read configuration file'
    }
  ],
  defaultEnabled: true,
  execute: async ({ path, encoding }, context) => {
    try {
      // Fetch from backend API
      const response = await fetch('http://localhost:8000/api/files/read', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, encoding })
      });

      if (!response.ok) {
        throw new Error(`Failed to read file: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          content: data.content,
          path,
          size: data.size,
          lastModified: data.lastModified
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }
};

/**
 * Write a file to the workspace
 */
const writeFile: Skill = {
  id: 'copilot_write_file',
  name: 'Write File',
  description: 'Write content to a file in the QuantMindX workspace. Creates directories if needed. Supports .mq5, .py, .json, .yaml, .md files.',
  agents: ['copilot'],
  category: 'file_operations',
  schema: z.object({
    path: z.string().describe('Relative path to the file from workspace root'),
    content: z.string().describe('File content to write'),
    createBackup: z.boolean().default(true).describe('Create backup before overwriting')
  }),
  examples: [
    {
      input: {
        path: 'strategies/new_strategy.mq5',
        content: '//+------------------------------------------------------------------+\n//| Expert initialization function                                   |\n//+------------------------------------------------------------------+'
      },
      output: 'File written successfully to strategies/new_strategy.mq5',
      description: 'Create a new MQL5 strategy file'
    }
  ],
  defaultEnabled: true,
  execute: async ({ path, content, createBackup }, context) => {
    try {
      // Get previous content for history tracking
      let previousContent: string | undefined;
      try {
        const readResponse = await fetch('http://localhost:8000/api/files/read', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path })
        });
        if (readResponse.ok) {
          const readData = await readResponse.json();
          previousContent = readData.content;
        }
      } catch {
        // File doesn't exist, that's okay
      }

      const response = await fetch('http://localhost:8000/api/files/write', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, content, createBackup })
      });

      if (!response.ok) {
        throw new Error(`Failed to write file: ${response.statusText}`);
      }

      const data = await response.json();

      // Record the operation in file history
      const fileName = path.split('/').pop() || path;
      const fileId = `file_${path.replace(/[^a-zA-Z0-9]/g, '_')}`;
      const action = previousContent === undefined ? 'created' : 'modified';
      
      fileHistoryManager.recordOperation(
        fileId,
        fileName,
        path,
        'copilot', // Agent type
        action,
        content,
        previousContent
      );

      return {
        success: true,
        data: {
          path,
          size: data.size,
          backup: data.backupPath,
          action
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }
};

/**
 * List files in a directory
 */
const listFiles: Skill = {
  id: 'copilot_list_files',
  name: 'List Files',
  description: 'List files and directories in a specified path. Supports filtering by extension and recursive listing.',
  agents: ['copilot'],
  category: 'file_operations',
  schema: z.object({
    path: z.string().default('.').describe('Directory path to list (default: current directory)'),
    extension: z.string().optional().describe('Filter by file extension (e.g., ".mq5", ".py")'),
    recursive: z.boolean().default(false).describe('List files recursively')
  }),
  examples: [
    {
      input: { path: 'strategies', extension: '.mq5' },
      output: ['strategies/ma_cross.mq5', 'strategies/rsi_strategy.mq5'],
      description: 'List all MQL5 files in strategies directory'
    }
  ],
  defaultEnabled: true,
  execute: async ({ path, extension, recursive }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/files/list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, extension, recursive })
      });

      if (!response.ok) {
        throw new Error(`Failed to list files: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          files: data.files,
          count: data.files.length
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }
};

// ============================================================================
// BROKER CONNECTION SKILLS
// ============================================================================

/**
 * Connect to a trading broker
 */
const connectBroker: Skill = {
  id: 'copilot_connect_broker',
  name: 'Connect Broker',
  description: 'Establish connection to a MetaTrader 5 broker terminal. Supports connection via login credentials or path to portable MT5 terminal.',
  agents: ['copilot'],
  category: 'broker',
  requirements: ['MT5_TERMINAL_PATH or broker login credentials'],
  schema: z.object({
    broker: z.string().describe('Broker name (e.g., "MetaQuotes-Demo", "ICMarketsEU-Demo")'),
    login: z.number().optional().describe('Account login number'),
    password: z.string().optional().describe('Account password'),
    server: z.string().optional().describe('Server address'),
    terminalPath: z.string().optional().describe('Path to portable MT5 terminal (alternative to login)')
  }),
  examples: [
    {
      input: {
        broker: 'MetaQuotes-Demo',
        login: 12345678,
        password: 'demo123',
        server: 'MetaQuotes-Demo'
      },
      output: 'Successfully connected to MetaQuotes-Demo (Account: 12345678)',
      description: 'Connect using account credentials'
    },
    {
      input: {
        broker: 'ICMarketsEU-Demo',
        terminalPath: '/path/to/terminal64.exe'
      },
      output: 'Successfully connected to ICMarketsEU-Demo via portable terminal',
      description: 'Connect using portable terminal path'
    }
  ],
  defaultEnabled: true,
  execute: async (params, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/broker/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });

      if (!response.ok) {
        throw new Error(`Failed to connect: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          broker: params.broker,
          account: data.account,
          balance: data.balance,
          equity: data.equity,
          currency: data.currency,
          connected: true
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }
};

/**
 * Get broker account information
 */
const getAccountInfo: Skill = {
  id: 'copilot_account_info',
  name: 'Get Account Info',
  description: 'Retrieve current broker account information including balance, equity, margin, and open positions.',
  agents: ['copilot'],
  category: 'broker',
  requirements: ['Active broker connection'],
  schema: z.object({
    includePositions: z.boolean().default(true).describe('Include open positions in response')
  }),
  examples: [
    {
      input: { includePositions: true },
      output: {
        account: 12345678,
        balance: 10000.00,
        equity: 10250.00,
        margin: 250.00,
        freeMargin: 10000.00,
        profit: 250.00,
        positions: []
      },
      description: 'Get account information with positions'
    }
  ],
  defaultEnabled: true,
  execute: async ({ includePositions }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/broker/account', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ includePositions })
      });

      if (!response.ok) {
        throw new Error(`Failed to get account info: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }
};

// ============================================================================
// EA DEPLOYMENT SKILLS
// ============================================================================

/**
 * Deploy an Expert Advisor to MT5
 */
const deployEA: Skill = {
  id: 'copilot_deploy_ea',
  name: 'Deploy Expert Advisor',
  description: 'Compile and deploy an MQL5 Expert Advisor to the MetaTrader 5 terminal. Optionally start the EA on a specific chart.',
  agents: ['copilot'],
  category: 'deployment',
  requirements: ['Active broker connection', 'MQL5 compiler'],
  schema: z.object({
    eaPath: z.string().describe('Path to the .mq5 or .ex5 file'),
    symbol: z.string().describe('Trading symbol (e.g., EURUSD, XAUUSD)'),
    timeframe: z.string().describe('Timeframe (e.g., M1, M5, H1, D1)'),
    parameters: z.record(z.any()).optional().describe('EA input parameters'),
    autoStart: z.boolean().default(false).describe('Automatically start the EA after deployment')
  }),
  examples: [
    {
      input: {
        eaPath: 'strategies/ma_cross_strategy.mq5',
        symbol: 'EURUSD',
        timeframe: 'H1',
        parameters: {
          'LotSize': 0.1,
          'StopLoss': 50,
          'TakeProfit': 100
        },
        autoStart: true
      },
      output: 'Successfully deployed and started MA Cross EA on EURUSD H1',
      description: 'Deploy and start EA on chart'
    }
  ],
  defaultEnabled: true,
  execute: async ({ eaPath, symbol, timeframe, parameters, autoStart }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/deployment/deploy-ea', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ eaPath, symbol, timeframe, parameters, autoStart })
      });

      if (!response.ok) {
        throw new Error(`Failed to deploy EA: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          eaPath,
          symbol,
          timeframe,
          chartId: data.chartId,
          eaHandle: data.eaHandle,
          status: autoStart ? 'running' : 'deployed'
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }
};

/**
 * Stop a running Expert Advisor
 */
const stopEA: Skill = {
  id: 'copilot_stop_ea',
  name: 'Stop Expert Advisor',
  description: 'Stop a running Expert Advisor on a specific chart.',
  agents: ['copilot'],
  category: 'deployment',
  requirements: ['Active broker connection'],
  schema: z.object({
    chartId: z.number().describe('Chart ID where EA is running'),
    eaHandle: z.number().optional().describe('EA handle (optional, can stop all EAs on chart)')
  }),
  examples: [
    {
      input: { chartId: 1234567890 },
      output: 'Successfully stopped EA on chart 1234567890',
      description: 'Stop EA on specific chart'
    }
  ],
  defaultEnabled: true,
  execute: async ({ chartId, eaHandle }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/deployment/stop-ea', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chartId, eaHandle })
      });

      if (!response.ok) {
        throw new Error(`Failed to stop EA: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          chartId,
          stopped: true
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }
};

// ============================================================================
// MT5 SYNC SKILLS
// ============================================================================

/**
 * Sync data with MT5 terminal
 */
const syncMT5: Skill = {
  id: 'copilot_sync_mt5',
  name: 'Sync MT5 Data',
  description: 'Synchronize historical data, account information, and open positions with MetaTrader 5 terminal.',
  agents: ['copilot'],
  category: 'sync',
  requirements: ['Active broker connection'],
  schema: z.object({
    symbols: z.array(z.string()).optional().describe('Specific symbols to sync (default: all)'),
    timeframe: z.string().optional().describe('Timeframe for historical data (default: all)'),
    daysBack: z.number().default(30).describe('Number of days of historical data to sync'),
    includePositions: z.boolean().default(true).describe('Include open positions'),
    includeOrders: z.boolean().default(true).describe('Include pending orders')
  }),
  examples: [
    {
      input: {
        symbols: ['EURUSD', 'GBPUSD'],
        timeframe: 'H1',
        daysBack: 90
      },
      output: 'Synced 90 days of H1 data for EURUSD and GBPUSD',
      description: 'Sync specific symbols and timeframe'
    }
  ],
  defaultEnabled: true,
  execute: async ({ symbols, timeframe, daysBack, includePositions, includeOrders }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/sync/mt5', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols, timeframe, daysBack, includePositions, includeOrders })
      });

      if (!response.ok) {
        throw new Error(`Failed to sync MT5: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          synced: data.synced,
          symbols: data.symbols,
          bars: data.bars,
          positions: data.positions,
          orders: data.orders
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }
};

// ============================================================================
// EXPORT ALL SKILLS
// ============================================================================

export const copilotSkills: Skill[] = [
  // File Operations
  readFile,
  writeFile,
  listFiles,

  // Broker Connection
  connectBroker,
  getAccountInfo,

  // EA Deployment
  deployEA,
  stopEA,

  // MT5 Sync
  syncMT5
];
