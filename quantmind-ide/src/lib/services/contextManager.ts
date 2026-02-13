// Context Manager Service - Manages context attachments for agent conversations
import type { 
  ChatContext, 
  FileReference, 
  StrategyReference, 
  BrokerReference, 
  BacktestReference 
} from '../stores/chatStore';

// Context templates
export const contextTemplates = {
  fullStrategy: {
    name: 'Full Strategy Context',
    description: 'Include strategy file, backtest results, and broker connection',
    includes: ['files', 'strategies', 'backtests', 'brokers']
  },
  backtestOnly: {
    name: 'Backtest Context',
    description: 'Include only backtest results and related strategy',
    includes: ['strategies', 'backtests']
  },
  codeReview: {
    name: 'Code Review Context',
    description: 'Include source files for review',
    includes: ['files']
  },
  deployment: {
    name: 'Deployment Context',
    description: 'Include strategy and broker for deployment',
    includes: ['strategies', 'brokers']
  }
};

// Context validation result
interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

// Context statistics
interface ContextStats {
  totalItems: number;
  byType: Record<string, number>;
  lastUpdated: Date | null;
}

// Context Manager Service
export const contextManager = {
  // Validate a context item
  validateFile(file: FileReference): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    if (!file.id) errors.push('File ID is required');
    if (!file.name) errors.push('File name is required');
    if (!file.path) errors.push('File path is required');
    
    // Check file size (max 10MB)
    if (file.size && file.size > 10 * 1024 * 1024) {
      warnings.push('File is larger than 10MB, may affect performance');
    }
    
    return { valid: errors.length === 0, errors, warnings };
  },
  
  validateStrategy(strategy: StrategyReference): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    if (!strategy.id) errors.push('Strategy ID is required');
    if (!strategy.name) errors.push('Strategy name is required');
    
    return { valid: errors.length === 0, errors, warnings };
  },
  
  validateBroker(broker: BrokerReference): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    if (!broker.id) errors.push('Broker ID is required');
    if (!broker.name) errors.push('Broker name is required');
    
    if (broker.status === 'disconnected') {
      warnings.push('Broker is not connected');
    }
    
    return { valid: errors.length === 0, errors, warnings };
  },
  
  validateBacktest(backtest: BacktestReference): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    if (!backtest.id) errors.push('Backtest ID is required');
    if (!backtest.name) errors.push('Backtest name is required');
    
    if (backtest.status === 'failed') {
      warnings.push('Backtest has failed');
    } else if (backtest.status === 'running') {
      warnings.push('Backtest is still running');
    }
    
    return { valid: errors.length === 0, errors, warnings };
  },
  
  // Validate entire context
  validateContext(context: ChatContext): ValidationResult {
    const allErrors: string[] = [];
    const allWarnings: string[] = [];
    
    context.files.forEach(file => {
      const result = this.validateFile(file);
      allErrors.push(...result.errors.map(e => `File "${file.name}": ${e}`));
      allWarnings.push(...result.warnings.map(w => `File "${file.name}": ${w}`));
    });
    
    context.strategies.forEach(strategy => {
      const result = this.validateStrategy(strategy);
      allErrors.push(...result.errors.map(e => `Strategy "${strategy.name}": ${e}`));
      allWarnings.push(...result.warnings.map(w => `Strategy "${strategy.name}": ${w}`));
    });
    
    context.brokers.forEach(broker => {
      const result = this.validateBroker(broker);
      allErrors.push(...result.errors.map(e => `Broker "${broker.name}": ${e}`));
      allWarnings.push(...result.warnings.map(w => `Broker "${broker.name}": ${w}`));
    });
    
    context.backtests.forEach(backtest => {
      const result = this.validateBacktest(backtest);
      allErrors.push(...result.errors.map(e => `Backtest "${backtest.name}": ${e}`));
      allWarnings.push(...result.warnings.map(w => `Backtest "${backtest.name}": ${w}`));
    });
    
    return { valid: allErrors.length === 0, errors: allErrors, warnings: allWarnings };
  },
  
  // Serialize context for agent invocation
  serializeContext(context: ChatContext): string {
    const serialized = {
      files: context.files.map(f => ({
        name: f.name,
        path: f.path,
        type: f.type
      })),
      strategies: context.strategies.map(s => ({
        name: s.name,
        type: s.type
      })),
      brokers: context.brokers.map(b => ({
        name: b.name,
        status: b.status
      })),
      backtests: context.backtests.map(b => ({
        name: b.name,
        status: b.status
      }))
    };
    
    return JSON.stringify(serialized);
  },
  
  // Create context from template
  createContextFromTemplate(
    templateName: keyof typeof contextTemplates,
    items: {
      files?: FileReference[];
      strategies?: StrategyReference[];
      brokers?: BrokerReference[];
      backtests?: BacktestReference[];
    }
  ): ChatContext {
    const template = contextTemplates[templateName];
    const context: ChatContext = {
      files: [],
      strategies: [],
      brokers: [],
      backtests: []
    };
    
    if (template.includes.includes('files')) {
      context.files = items.files || [];
    }
    if (template.includes.includes('strategies')) {
      context.strategies = items.strategies || [];
    }
    if (template.includes.includes('brokers')) {
      context.brokers = items.brokers || [];
    }
    if (template.includes.includes('backtests')) {
      context.backtests = items.backtests || [];
    }
    
    return context;
  },
  
  // Get context statistics
  getContextStats(context: ChatContext): ContextStats {
    return {
      totalItems: context.files.length + context.strategies.length + 
                  context.brokers.length + context.backtests.length,
      byType: {
        files: context.files.length,
        strategies: context.strategies.length,
        brokers: context.brokers.length,
        backtests: context.backtests.length
      },
      lastUpdated: new Date()
    };
  },
  
  // Merge contexts
  mergeContexts(...contexts: ChatContext[]): ChatContext {
    const merged: ChatContext = {
      files: [],
      strategies: [],
      brokers: [],
      backtests: []
    };
    
    contexts.forEach(context => {
      // Add unique files
      context.files.forEach(file => {
        if (!merged.files.some(f => f.id === file.id)) {
          merged.files.push(file);
        }
      });
      
      // Add unique strategies
      context.strategies.forEach(strategy => {
        if (!merged.strategies.some(s => s.id === strategy.id)) {
          merged.strategies.push(strategy);
        }
      });
      
      // Add unique brokers
      context.brokers.forEach(broker => {
        if (!merged.brokers.some(b => b.id === broker.id)) {
          merged.brokers.push(broker);
        }
      });
      
      // Add unique backtests
      context.backtests.forEach(backtest => {
        if (!merged.backtests.some(b => b.id === backtest.id)) {
          merged.backtests.push(backtest);
        }
      });
    });
    
    return merged;
  },
  
  // Filter context by type
  filterContext(context: ChatContext, types: string[]): ChatContext {
    const filtered: ChatContext = {
      files: [],
      strategies: [],
      brokers: [],
      backtests: []
    };
    
    if (types.includes('files')) filtered.files = context.files;
    if (types.includes('strategies')) filtered.strategies = context.strategies;
    if (types.includes('brokers')) filtered.brokers = context.brokers;
    if (types.includes('backtests')) filtered.backtests = context.backtests;
    
    return filtered;
  },
  
  // Create file reference from path
  createFileReference(path: string, name?: string): FileReference {
    const fileName = name || path.split('/').pop() || 'unknown';
    const extension = fileName.split('.').pop() || '';
    
    return {
      id: `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: fileName,
      path,
      type: extension
    };
  },
  
  // Create strategy reference
  createStrategyReference(name: string, type: string): StrategyReference {
    return {
      id: `strategy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      type
    };
  },
  
  // Create broker reference
  createBrokerReference(name: string, status: 'connected' | 'disconnected' = 'disconnected'): BrokerReference {
    return {
      id: `broker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      status
    };
  },
  
  // Create backtest reference
  createBacktestReference(name: string, status: BacktestReference['status'] = 'pending'): BacktestReference {
    return {
      id: `backtest_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      status
    };
  },
  
  // Format context for display
  formatContextSummary(context: ChatContext): string {
    const parts: string[] = [];
    
    if (context.files.length > 0) {
      parts.push(`${context.files.length} file${context.files.length > 1 ? 's' : ''}`);
    }
    if (context.strategies.length > 0) {
      parts.push(`${context.strategies.length} strateg${context.strategies.length > 1 ? 'ies' : 'y'}`);
    }
    if (context.brokers.length > 0) {
      parts.push(`${context.brokers.length} broker${context.brokers.length > 1 ? 's' : ''}`);
    }
    if (context.backtests.length > 0) {
      parts.push(`${context.backtests.length} backtest${context.backtests.length > 1 ? 's' : ''}`);
    }
    
    return parts.length > 0 ? parts.join(', ') : 'No context';
  },
  
  // Check if context is empty
  isContextEmpty(context: ChatContext): boolean {
    return context.files.length === 0 &&
           context.strategies.length === 0 &&
           context.brokers.length === 0 &&
           context.backtests.length === 0;
  },
  
  // Clear context
  clearContext(): ChatContext {
    return {
      files: [],
      strategies: [],
      brokers: [],
      backtests: []
    };
  }
};

export default contextManager;
