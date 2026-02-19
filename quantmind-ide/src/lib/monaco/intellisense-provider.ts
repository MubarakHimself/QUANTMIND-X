/**
 * IntelliSense Provider for Monaco Editor
 *
 * Provides auto-completion, hover information, and signature help
 * for MQL5, Python, and JavaScript languages.
 */

import type * as Monaco from 'monaco-editor';
import { mql5Keywords, mql5Builtins, mql5Constants } from './mql5-language';

// MQL5 function signatures and documentation
const mql5FunctionDocs: Record<string, { signature: string; description: string; params: string[] }> = {
  'OrderSend': {
    signature: 'OrderSend(request: MqlTradeRequest, result: MqlTradeResult): bool',
    description: 'Sends a trade request to the server.',
    params: ['request - Trade request structure', 'result - Trade result structure']
  },
  'PositionOpen': {
    signature: 'PositionOpen(symbol: string, order_type: ENUM_ORDER_TYPE, volume: double, price: double, sl: double, tp: double, comment: string): bool',
    description: 'Opens a position with specified parameters.',
    params: ['symbol - Trading symbol', 'order_type - ORDER_TYPE_BUY or ORDER_TYPE_SELL', 'volume - Lot size', 'price - Entry price', 'sl - Stop loss', 'tp - Take profit', 'comment - Order comment']
  },
  'PositionClose': {
    signature: 'PositionClose(ticket: int, volume: double): bool',
    description: 'Closes a position by ticket number.',
    params: ['ticket - Position ticket', 'volume - Volume to close (0 = all)']
  },
  'SymbolInfoDouble': {
    signature: 'SymbolInfoDouble(symbol: string, prop_id: ENUM_SYMBOL_INFO_DOUBLE): double',
    description: 'Returns a double type property of a symbol.',
    params: ['symbol - Symbol name', 'prop_id - Property identifier']
  },
  'SymbolInfoTick': {
    signature: 'SymbolInfoTick(symbol: string, tick: MqlTick): bool',
    description: 'Returns current prices of a symbol.',
    params: ['symbol - Symbol name', 'tick - Structure to receive tick data']
  },
  'CopyRates': {
    signature: 'CopyRates(symbol: string, timeframe: ENUM_TIMEFRAMES, start_pos: int, count: int, rates_array: MqlRates[]): int',
    description: 'Gets historical price data as an array of MqlRates structures.',
    params: ['symbol - Symbol name', 'timeframe - Timeframe', 'start_pos - Start position', 'count - Number of bars', 'rates_array - Array to receive data']
  },
  'iMA': {
    signature: 'iMA(symbol: string, timeframe: ENUM_TIMEFRAMES, ma_period: int, ma_shift: int, ma_method: ENUM_MA_METHOD, applied_price: ENUM_APPLIED_PRICE): int',
    description: 'Creates Moving Average indicator handle.',
    params: ['symbol - Symbol name', 'timeframe - Timeframe', 'ma_period - MA period', 'ma_shift - MA shift', 'ma_method - MA method', 'applied_price - Applied price']
  },
  'CopyBuffer': {
    signature: 'CopyBuffer(indicator_handle: int, buffer_num: int, start_pos: int, count: int, buffer: double[]): int',
    description: 'Gets data from an indicator buffer.',
    params: ['indicator_handle - Indicator handle', 'buffer_num - Buffer number', 'start_pos - Start position', 'count - Number of values', 'buffer - Array to receive data']
  },
  'AccountInfoDouble': {
    signature: 'AccountInfoDouble(property_id: ENUM_ACCOUNT_INFO_DOUBLE): double',
    description: 'Returns the value of a double type account property.',
    params: ['property_id - Property identifier (ACCOUNT_BALANCE, ACCOUNT_EQUITY, etc.)']
  },
  'Print': {
    signature: 'Print(...args: any): void',
    description: 'Prints a message to the Experts journal.',
    params: ['args - Values to print']
  },
  'Alert': {
    signature: 'Alert(...args: any): void',
    description: 'Displays an alert dialog and prints to the Experts journal.',
    params: ['args - Values to display']
  },
  'StringFormat': {
    signature: 'StringFormat(format: string, ...args: any): string',
    description: 'Formats a string according to the specified format.',
    params: ['format - Format string with placeholders', 'args - Values to format']
  },
  'MathMax': {
    signature: 'MathMax(value1: double, value2: double): double',
    description: 'Returns the maximum of two values.',
    params: ['value1 - First value', 'value2 - Second value']
  },
  'MathMin': {
    signature: 'MathMin(value1: double, value2: double): double',
    description: 'Returns the minimum of two values.',
    params: ['value1 - First value', 'value2 - Second value']
  },
  'NormalizeDouble': {
    signature: 'NormalizeDouble(value: double, digits: int): double',
    description: 'Rounds a floating point number to specified precision.',
    params: ['value - Value to normalize', 'digits - Number of decimal places']
  }
};

// Python built-in functions for completion
const pythonBuiltins = [
  'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
  'open', 'input', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr', 'delattr',
  'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter', 'any', 'all', 'sum', 'min', 'max',
  'abs', 'round', 'pow', 'divmod', 'hex', 'oct', 'bin', 'ord', 'chr', 'repr', 'hash',
  'super', 'property', 'staticmethod', 'classmethod', 'iter', 'next', 'slice', 'format',
  'exec', 'eval', 'compile', 'globals', 'locals', 'vars', 'dir', 'help', 'id', 'callable'
];

// JavaScript built-in functions
const jsBuiltins = [
  'console', 'log', 'warn', 'error', 'info', 'debug',
  'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
  'fetch', 'Promise', 'async', 'await', 'then', 'catch', 'finally',
  'JSON', 'parse', 'stringify', 'Object', 'keys', 'values', 'entries',
  'Array', 'from', 'isArray', 'push', 'pop', 'shift', 'unshift', 'slice', 'splice',
  'String', 'Number', 'Boolean', 'Date', 'Math', 'RegExp',
  'Map', 'Set', 'WeakMap', 'WeakSet', 'Symbol', 'Proxy', 'Reflect'
];

/**
 * Register completion providers for all supported languages
 */
export function registerIntelliSense(monaco: typeof Monaco): void {
  // MQL5 completion provider
  monaco.languages.registerCompletionItemProvider('mql5', {
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn
      };

      const suggestions: Monaco.languages.CompletionItem[] = [];

      // Keywords
      mql5Keywords.forEach(keyword => {
        suggestions.push({
          label: keyword,
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText: keyword,
          range,
          detail: 'MQL5 Keyword'
        });
      });

      // Built-in functions
      mql5Builtins.forEach(func => {
        const docInfo = mql5FunctionDocs[func];
        suggestions.push({
          label: func,
          kind: monaco.languages.CompletionItemKind.Function,
          insertText: docInfo ? `${func}($0)` : func,
          insertTextRules: docInfo ? monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet : undefined,
          range,
          detail: docInfo?.signature || 'MQL5 Built-in Function',
          documentation: docInfo?.description
        });
      });

      // Constants
      mql5Constants.forEach(constant => {
        suggestions.push({
          label: constant,
          kind: monaco.languages.CompletionItemKind.Constant,
          insertText: constant,
          range,
          detail: 'MQL5 Constant'
        });
      });

      return { suggestions };
    }
  });

  // MQL5 hover provider
  monaco.languages.registerHoverProvider('mql5', {
    provideHover: (model, position) => {
      const word = model.getWordAtPosition(position);
      if (!word) return null;

      const docInfo = mql5FunctionDocs[word.word];
      if (docInfo) {
        return {
          contents: [
            { value: `**${word.word}**` },
            { value: '```mql5\n' + docInfo.signature + '\n```' },
            { value: docInfo.description },
            { value: '**Parameters:**\n' + docInfo.params.map(p => `- ${p}`).join('\n') }
          ]
        };
      }

      // Check if it's a constant
      if (mql5Constants.includes(word.word)) {
        return {
          contents: [
            { value: `**${word.word}**` },
            { value: '*MQL5 Constant*' }
          ]
        };
      }

      return null;
    }
  });

  // MQL5 signature help provider
  monaco.languages.registerSignatureHelpProvider('mql5', {
    signatureHelpTriggerCharacters: ['(', ','],
    provideSignatureHelp: (model, position) => {
      const textUntilPosition = model.getValueInRange({
        startLineNumber: 1,
        startColumn: 1,
        endLineNumber: position.lineNumber,
        endColumn: position.column
      });

      // Find the function name being called
      const match = textUntilPosition.match(/(\w+)\s*\([^)]*$/);
      if (!match) return null;

      const funcName = match[1];
      const docInfo = mql5FunctionDocs[funcName];
      if (!docInfo) return null;

      return {
        value: {
          signatures: [{
            label: docInfo.signature,
            documentation: docInfo.description,
            parameters: docInfo.params.map(p => ({
              label: p.split(' - ')[0].trim(),
              documentation: p
            }))
          }],
          activeSignature: 0,
          activeParameter: 0
        },
        dispose: () => {}
      };
    }
  });

  // Python completion provider
  monaco.languages.registerCompletionItemProvider('python', {
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn
      };

      const suggestions: Monaco.languages.CompletionItem[] = [];

      pythonBuiltins.forEach(func => {
        suggestions.push({
          label: func,
          kind: monaco.languages.CompletionItemKind.Function,
          insertText: func,
          range,
          detail: 'Python Built-in'
        });
      });

      // Common Python keywords
      const pythonKeywords = ['def', 'class', 'import', 'from', 'return', 'if', 'elif', 'else',
        'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'lambda', 'yield',
        'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'pass', 'break', 'continue'];

      pythonKeywords.forEach(keyword => {
        suggestions.push({
          label: keyword,
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText: keyword,
          range
        });
      });

      return { suggestions };
    }
  });

  // JavaScript completion provider
  monaco.languages.registerCompletionItemProvider('javascript', {
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn
      };

      const suggestions: Monaco.languages.CompletionItem[] = [];

      jsBuiltins.forEach(func => {
        suggestions.push({
          label: func,
          kind: monaco.languages.CompletionItemKind.Function,
          insertText: func,
          range,
          detail: 'JavaScript Built-in'
        });
      });

      // Common JS keywords
      const jsKeywords = ['const', 'let', 'var', 'function', 'return', 'if', 'else',
        'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch',
        'finally', 'throw', 'new', 'this', 'class', 'extends', 'import', 'export', 'default',
        'true', 'false', 'null', 'undefined', 'typeof', 'instanceof'];

      jsKeywords.forEach(keyword => {
        suggestions.push({
          label: keyword,
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText: keyword,
          range
        });
      });

      return { suggestions };
    }
  });
}

/**
 * Register project-specific completions from backend API
 */
export async function fetchProjectCompletions(
  monaco: typeof Monaco,
  projectPath: string
): Promise<void> {
  try {
    const response = await fetch(`/api/ide/completions?project=${encodeURIComponent(projectPath)}`);
    if (!response.ok) return;

    const data = await response.json();

    // Register custom completions for the project
    if (data.symbols) {
      monaco.languages.registerCompletionItemProvider('mql5', {
        provideCompletionItems: (model, position) => {
          const word = model.getWordUntilPosition(position);
          const range = {
            startLineNumber: position.lineNumber,
            endLineNumber: position.lineNumber,
            startColumn: word.startColumn,
            endColumn: word.endColumn
          };

          const suggestions: Monaco.languages.CompletionItem[] = data.symbols.map((symbol: any) => ({
            label: symbol.name,
            kind: symbol.kind === 'function'
              ? monaco.languages.CompletionItemKind.Function
              : monaco.languages.CompletionItemKind.Variable,
            insertText: symbol.name,
            range,
            detail: symbol.detail || symbol.type,
            documentation: symbol.documentation
          }));

          return { suggestions };
        }
      });
    }
  } catch (error) {
    console.warn('Failed to fetch project completions:', error);
  }
}

export { mql5FunctionDocs };
