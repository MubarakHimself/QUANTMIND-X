/**
 * QuantCode Skills - MQL5 code generation and development capabilities
 *
 * These skills provide MQL5 code generation, syntax validation, compilation,
 * debugging, and documentation generation for the QuantCode agent.
 */

import { z } from 'zod';
import type { Skill, SkillContext } from './index';

// ============================================================================
// MQL5 CODE GENERATION SKILLS
// ============================================================================

/**
 * Generate MQL5 code from NPRD or TRD
 */
const generateMQL5: Skill = {
  id: 'quantcode_generate_mql5',
  name: 'Generate MQL5 Code',
  description: 'Generate complete MQL5 Expert Advisor code from NPRD or TRD specifications. Includes signal logic, risk management, trade execution, and error handling.',
  agents: ['quantcode'],
  category: 'code_generation',
  requirements: ['OpenAI API or Anthropic API for code generation'],
  schema: z.object({
    specification: z.any().describe('NPRD or TRD specification object'),
    template: z.string().optional().describe('Use a specific code template (default: standard EA)'),
    includeComments: z.boolean().default(true).describe('Include detailed code comments'),
    optimize: z.boolean().default(true).describe('Optimize generated code for performance'),
    version: z.string().default('1.0.0').describe('EA version number')
  }),
  examples: [
    {
      input: {
        specification: {
          name: 'MA Cross EA',
          strategy: 'Moving average crossover',
          entry: { long: 'fast MA crosses above slow MA' },
          exit: { long: 'fast MA crosses below slow MA' },
          risk: { stopLoss: 50, takeProfit: 100, riskPercent: 2 }
        },
        template: 'standard_ea',
        includeComments: true
      },
      output: {
        code: '//+------------------------------------------------------------------+\n//|                                             MA_Cross_EA.mq5      |\n//+------------------------------------------------------------------+\n#property copyright "QuantMindX"\n#property version   "1.00"\n...',
        filename: 'MA_Cross_EA.mq5',
        size: 5420,
        warnings: []
      },
      description: 'Generate MQL5 EA from specification'
    }
  ],
  defaultEnabled: true,
  execute: async ({ specification, template, includeComments, optimize, version }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ specification, template, includeComments, optimize, version })
      });

      if (!response.ok) {
        throw new Error(`Failed to generate MQL5: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          code: data.code,
          filename: data.filename,
          size: data.size,
          includes: data.includes,
          warnings: data.warnings
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
 * Generate specific code component
 */
const generateComponent: Skill = {
  id: 'quantcode_generate_component',
  name: 'Generate Code Component',
  description: 'Generate a specific MQL5 code component (signal class, money management, trailing stop, etc.) to be integrated into an existing EA.',
  agents: ['quantcode'],
  category: 'code_generation',
  requirements: ['OpenAI API or Anthropic API for code generation'],
  schema: z.object({
    componentType: z.enum(['signal', 'money_management', 'trailing_stop', 'indicator', 'filter', 'utility']).describe('Type of component to generate'),
    description: z.string().describe('Natural language description of component behavior'),
    parameters: z.record(z.any()).optional().describe('Component input parameters'),
    dependencies: z.array(z.string()).optional().describe('Required includes or dependencies')
  }),
  examples: [
    {
      input: {
        componentType: 'signal',
        description: 'RSI divergence signal: Detect bullish divergence when price makes lower low but RSI makes higher low',
        parameters: {
          'RSI_Period': 14,
          'Lookback': 5
        }
      },
      output: {
        code: '//+------------------------------------------------------------------+\n//|                                            RSIDivergenceSignal |\n//+------------------------------------------------------------------+\nclass CRDIDivergenceSignal : public CExpertSignal\n{\n  int m_rsi_period;\n  int m_lookback;\n  ...\n}',
        className: 'CRDIDivergenceSignal',
        includes: ['<Expert/ExpertSignal.mqh>'],
        integrationNotes: 'Add to signal module and override LongCondition/ShortCondition'
      },
      description: 'Generate RSI divergence signal class'
    }
  ],
  defaultEnabled: true,
  execute: async ({ componentType, description, parameters, dependencies }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/generate-component', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ componentType, description, parameters, dependencies })
      });

      if (!response.ok) {
        throw new Error(`Failed to generate component: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          code: data.code,
          className: data.className,
          includes: data.includes,
          integrationNotes: data.integrationNotes
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
// SYNTAX VALIDATION SKILLS
// ============================================================================

/**
 * Validate MQL5 syntax
 */
const validateSyntax: Skill = {
  id: 'quantcode_validate_syntax',
  name: 'Validate MQL5 Syntax',
  description: 'Validate MQL5 code syntax and identify errors, warnings, and style issues before compilation.',
  agents: ['quantcode'],
  category: 'validation',
  schema: z.object({
    code: z.string().describe('MQL5 source code to validate'),
    checkStyle: z.boolean().default(true).describe('Check coding style and best practices'),
    strictMode: z.boolean().default(false).describe('Enable strict mode for more warnings'),
    version: z.string().default('MT5').describe('MQL version (MT5 or MT4)')
  }),
  examples: [
    {
      input: {
        code: 'int OnInit() {\n  return INIT_SUCCEEDED;\n}\n\nvoid OnTick() {\n  double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);\n  Print("Bid: ", bid);\n}',
        checkStyle: true
      },
      output: {
        valid: true,
        errors: [],
        warnings: [],
        style: [
          { line: 5, message: 'Consider caching SYMBOL_BID value', severity: 'info' }
        ]
      },
      description: 'Validate MQL5 code'
    }
  ],
  defaultEnabled: true,
  execute: async ({ code, checkStyle, strictMode, version }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, checkStyle, strictMode, version })
      });

      if (!response.ok) {
        throw new Error(`Failed to validate syntax: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          valid: data.valid,
          errors: data.errors,
          warnings: data.warnings,
          style: data.style,
          metrics: data.metrics
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
 * Fix syntax errors automatically
 */
const fixSyntaxErrors: Skill = {
  id: 'quantcode_fix_syntax',
  name: 'Fix Syntax Errors',
  description: 'Automatically fix common MQL5 syntax errors and provide corrected code.',
  agents: ['quantcode'],
  category: 'validation',
  requirements: ['OpenAI API or Anthropic API for code fixing'],
  schema: z.object({
    code: z.string().describe('MQL5 source code with errors'),
    errors: z.array(z.object({
      line: z.number(),
      column: z.number().optional(),
      message: z.string()
    })).optional().describe('Specific errors to fix (optional, will auto-detect)'),
    preserveStyle: z.boolean().default(true).describe('Preserve original code style')
  }),
  examples: [
    {
      input: {
        code: 'int OnInit() {\n  return INIT_SUCCEEDED\n}\n\nvoid OnTick() {',
        preserveStyle: true
      },
      output: {
        fixed: 'int OnInit() {\n  return INIT_SUCCEEDED;\n}\n\nvoid OnTick() {\n  // TODO: Implement OnTick logic\n}',
        fixes: [
          { line: 2, message: 'Added semicolon', type: 'syntax' },
          { line: 4, message: 'Added closing brace and placeholder', type: 'structure' }
        ],
        remaining: []
      },
      description: 'Fix syntax errors'
    }
  ],
  defaultEnabled: true,
  execute: async ({ code, errors, preserveStyle }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/fix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, errors, preserveStyle })
      });

      if (!response.ok) {
        throw new Error(`Failed to fix syntax: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          fixed: data.fixed,
          fixes: data.fixes,
          remaining: data.remaining
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
// COMPILATION SKILLS
// ============================================================================

/**
 * Compile MQL5 code
 */
const compileMQL5: Skill = {
  id: 'quantcode_compile',
  name: 'Compile MQL5',
  description: 'Compile MQL5 source code using MetaEditor compiler. Returns compilation result, errors, warnings, and output file.',
  agents: ['quantcode'],
  category: 'compilation',
  requirements: ['MetaEditor compiler (metaeditor.exe or metacompiler.exe)'],
  schema: z.object({
    code: z.string().describe('MQL5 source code to compile'),
    filename: z.string().describe('Output filename (e.g., "MyEA.mq5")'),
    includePath: z.string().optional().describe('Additional include directories'),
    optimization: z.enum(['none', 'speed', 'size']).default('speed').describe('Compilation optimization level'),
    strictMode: z.boolean().default(true).describe('Enable strict compilation mode')
  }),
  examples: [
    {
      input: {
        code: '//+------------------------------------------------------------------+\n//|                                                    TestEA.mq5 |\n//+------------------------------------------------------------------+',
        filename: 'TestEA.mq5',
        optimization: 'speed',
        strictMode: true
      },
      output: {
        success: true,
        ex5File: 'MQL5/Experts/TestEA.ex5',
        errors: [],
        warnings: [],
        compileTime: 1.2,
        fileSize: 15240
      },
      description: 'Compile MQL5 code'
    }
  ],
  defaultEnabled: true,
  execute: async ({ code, filename, includePath, optimization, strictMode }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, filename, includePath, optimization, strictMode })
      });

      if (!response.ok) {
        throw new Error(`Failed to compile: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: data.success,
        data: {
          ex5File: data.ex5File,
          errors: data.errors,
          warnings: data.warnings,
          compileTime: data.compileTime,
          fileSize: data.fileSize
        },
        error: data.success ? undefined : 'Compilation failed'
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
// DEBUGGING SKILLS
// ============================================================================

/**
 * Debug MQL5 code
 */
const debugCode: Skill = {
  id: 'quantcode_debug',
  name: 'Debug MQL5 Code',
  description: 'Analyze MQL5 code for bugs, logic errors, and potential issues. Provides detailed analysis and fix suggestions.',
  agents: ['quantcode'],
  category: 'debugging',
  schema: z.object({
    code: z.string().describe('MQL5 source code to debug'),
    issueDescription: z.string().optional().describe('Description of the issue or bug being investigated'),
    checkMemory: z.boolean().default(true).describe('Check for memory leaks and issues'),
    checkConcurrency: z.boolean().default(true).describe('Check for multi-threading issues'),
    deepAnalysis: z.boolean().default(false).describe('Enable deep static analysis')
  }),
  examples: [
    {
      input: {
        code: 'void OnTick() {\n  static int counter = 0;\n  counter++;\n  if (counter > 1000) {\n    // Reset\n  }\n}',
        issueDescription: 'Counter not resetting properly',
        checkMemory: true
      },
      output: {
        issues: [
          {
            type: 'logic',
            severity: 'warning',
            line: 5,
            message: 'Counter reset condition exists but does not actually reset counter',
            suggestion: 'Add counter = 0; inside the if block'
          }
        ],
        memoryIssues: [],
        suggestions: [
          'Consider using event-driven reset instead of counter',
          'Add logging for debugging counter state'
        ]
      },
      description: 'Debug MQL5 code'
    }
  ],
  defaultEnabled: true,
  execute: async ({ code, issueDescription, checkMemory, checkConcurrency, deepAnalysis }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/debug', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, issueDescription, checkMemory, checkConcurrency, deepAnalysis })
      });

      if (!response.ok) {
        throw new Error(`Failed to debug code: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          issues: data.issues,
          memoryIssues: data.memoryIssues,
          concurrencyIssues: data.concurrencyIssues,
          suggestions: data.suggestions,
          fixedCode: data.fixedCode
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
 * Optimize MQL5 code
 */
const optimizeCode: Skill = {
  id: 'quantcode_optimize',
  name: 'Optimize MQL5 Code',
  description: 'Optimize MQL5 code for better performance, reduced memory usage, and faster execution.',
  agents: ['quantcode'],
  category: 'optimization',
  schema: z.object({
    code: z.string().describe('MQL5 source code to optimize'),
    optimizationLevel: z.enum(['basic', 'aggressive']).default('basic').describe('Optimization aggressiveness'),
    target: z.enum(['speed', 'memory', 'balanced']).default('balanced').describe('Optimization target'),
    preserveReadability: z.boolean().default(true).describe('Preserve code readability')
  }),
  examples: [
    {
      input: {
        code: 'double GetPrice(string symbol) {\n  return SymbolInfoDouble(symbol, SYMBOL_BID);\n}\n\nvoid OnTick() {\n  double price = GetPrice(Symbol());\n}',
        target: 'speed',
        preserveReadability: true
      },
      output: {
        optimized: 'static double s_lastPrice = 0;\nstatic string s_lastSymbol = "";\n\nvoid OnTick() {\n  string symbol = Symbol();\n  double price = SymbolInfoDouble(symbol, SYMBOL_BID);\n  ...\n}',
        improvements: [
          { metric: 'API call reduction', before: 'Every tick', after: 'Cached when needed' },
          { metric: 'Function call overhead', improvement: 'Eliminated direct call' }
        ],
        estimatedSpeedup: '15-20%'
      },
      description: 'Optimize code for speed'
    }
  ],
  defaultEnabled: true,
  execute: async ({ code, optimizationLevel, target, preserveReadability }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, optimizationLevel, target, preserveReadability })
      });

      if (!response.ok) {
        throw new Error(`Failed to optimize code: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          optimized: data.optimized,
          improvements: data.improvements,
          estimatedSpeedup: data.estimatedSpeedup,
          memoryReduction: data.memoryReduction
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
// DOCUMENTATION SKILLS
// ============================================================================

/**
 * Generate code documentation
 */
const generateDocumentation: Skill = {
  id: 'quantcode_documentation',
  name: 'Generate Documentation',
  description: 'Generate comprehensive documentation for MQL5 code including function descriptions, parameter docs, usage examples, and best practices.',
  agents: ['quantcode'],
  category: 'documentation',
  schema: z.object({
    code: z.string().describe('MQL5 source code to document'),
    format: z.enum(['inline', 'markdown', 'html']).default('markdown').describe('Documentation format'),
    includeExamples: z.boolean().default(true).describe('Include usage examples'),
    includeDiagrams: z.boolean().default(false).describe('Include flow diagrams (if available)'),
    targetAudience: z.enum(['beginner', 'intermediate', 'advanced']).default('intermediate').describe('Target audience level')
  }),
  examples: [
    {
      input: {
        code: 'class CSimpleStrategy : public CExpertStrategy {\npublic:\n  CSimpleStrategy();\n  virtual bool Init();\n  virtual bool Process();\n};',
        format: 'markdown',
        includeExamples: true,
        targetAudience: 'intermediate'
      },
      output: {
        documentation: '# CSimpleStrategy Class Documentation\n\n## Description\n...\n\n## Methods\n### Init()\nInitializes the strategy...\n\n## Usage Example\n```mql5\nCSimpleStrategy strategy;\nstrategy.Init();\n```',
        sections: ['Description', 'Parameters', 'Return Values', 'Examples', 'Notes']
      },
      description: 'Generate documentation'
    }
  ],
  defaultEnabled: true,
  execute: async ({ code, format, includeExamples, includeDiagrams, targetAudience }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, format, includeExamples, includeDiagrams, targetAudience })
      });

      if (!response.ok) {
        throw new Error(`Failed to generate documentation: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          documentation: data.documentation,
          sections: data.sections,
          format: data.format
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
 * Add Context7 documentation lookup integration
 */
const lookupDocumentation: Skill = {
  id: 'quantcode_lookup_docs',
  name: 'Lookup Documentation',
  description: 'Look up MQL5 documentation from Context7 or official MetaQuotes documentation for specific functions, classes, or concepts.',
  agents: ['quantcode'],
  category: 'documentation',
  requirements: ['Context7 MCP server or internet connection'],
  schema: z.object({
    query: z.string().describe('Function name, class name, or concept to look up'),
    source: z.enum(['context7', 'metaquotes', 'both']).default('both').describe('Documentation source preference'),
    includeExamples: z.boolean().default(true).describe('Include code examples from documentation')
  }),
  examples: [
    {
      input: {
        query: 'OrderSend',
        source: 'context7',
        includeExamples: true
      },
      output: {
        function: 'OrderSend',
        signature: 'bool OrderSend(MqlTradeRequest& request, MqlTradeResult& result)',
        description: 'Sends a trade order to the server...',
        parameters: [
          { name: 'request', type: 'MqlTradeRequest&', description: 'Trade request structure' },
          { name: 'result', type: 'MqlTradeResult&', description: 'Trade result structure' }
        ],
        returnValue: 'Returns true on success, false on failure',
        examples: [
          {
            code: 'MqlTradeRequest request = {};\nrequest.action = TRADE_ACTION_DEAL;\n// ... fill request\nMqlTradeResult result = {};\nif (OrderSend(request, result)) {\n  // Order sent successfully\n}'
          }
        ]
      },
      description: 'Look up OrderSend documentation'
    }
  ],
  defaultEnabled: true,
  execute: async ({ query, source, includeExamples }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/quantcode/lookup-docs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, source, includeExamples })
      });

      if (!response.ok) {
        throw new Error(`Failed to lookup documentation: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          function: data.function,
          signature: data.signature,
          description: data.description,
          parameters: data.parameters,
          returnValue: data.returnValue,
          examples: data.examples,
          seeAlso: data.seeAlso
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

export const quantcodeSkills: Skill[] = [
  // Code Generation
  generateMQL5,
  generateComponent,

  // Syntax Validation
  validateSyntax,
  fixSyntaxErrors,

  // Compilation
  compileMQL5,

  // Debugging
  debugCode,
  optimizeCode,

  // Documentation
  generateDocumentation,
  lookupDocumentation
];
