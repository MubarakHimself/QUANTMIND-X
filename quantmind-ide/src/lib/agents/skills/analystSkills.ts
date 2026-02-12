/**
 * Analyst Skills - Trading strategy analysis capabilities
 *
 * These skills provide strategy analysis, NPRD parsing, TRD generation,
 * and performance evaluation for the Analyst agent.
 */

import { z } from 'zod';
import type { Skill, SkillContext } from './index';

// ============================================================================
// STRATEGY ANALYSIS SKILLS
// ============================================================================

/**
 * Analyze a trading strategy's backtest results
 */
const analyzeBacktest: Skill = {
  id: 'analyst_analyze_backtest',
  name: 'Analyze Backtest',
  description: 'Analyze backtest results and provide detailed performance metrics, risk assessment, and optimization recommendations.',
  agents: ['analyst'],
  category: 'analysis',
  schema: z.object({
    backtestId: z.string().describe('Backtest ID or file path'),
    metrics: z.array(z.string()).optional().describe('Specific metrics to analyze (default: all)'),
    includeCharts: z.boolean().default(false).describe('Include performance charts in analysis'),
    compareWith: z.string().optional().describe('Compare with another backtest ID')
  }),
  examples: [
    {
      input: {
        backtestId: 'bt_20240210_eurusd_h1',
        metrics: ['profit', 'sharpe', 'max_drawdown', 'win_rate']
      },
      output: {
        summary: 'Strategy shows positive returns with acceptable risk',
        metrics: {
          profit: 2500.50,
          sharpe: 1.85,
          maxDrawdown: 8.5,
          winRate: 62.3
        },
        recommendations: [
          'Consider reducing drawdown with tighter stops',
          'Win rate is good, focus on position sizing'
        ]
      },
      description: 'Analyze specific backtest metrics'
    }
  ],
  defaultEnabled: true,
  execute: async ({ backtestId, metrics, includeCharts, compareWith }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/analysis/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backtestId, metrics, includeCharts, compareWith })
      });

      if (!response.ok) {
        throw new Error(`Failed to analyze backtest: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          backtestId,
          analysis: data.analysis,
          metrics: data.metrics,
          charts: data.charts,
          comparison: data.comparison
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
 * Compare multiple strategies
 */
const compareStrategies: Skill = {
  id: 'analyst_compare_strategies',
  name: 'Compare Strategies',
  description: 'Compare multiple trading strategies side-by-side with detailed performance metrics and statistical significance tests.',
  agents: ['analyst'],
  category: 'analysis',
  schema: z.object({
    strategies: z.array(z.string()).describe('Array of strategy IDs or names to compare'),
    metrics: z.array(z.string()).default(['profit', 'sharpe', 'max_drawdown', 'win_rate', 'profit_factor']).describe('Metrics to compare'),
    symbol: z.string().optional().describe('Filter by symbol'),
    timeframe: z.string().optional().describe('Filter by timeframe'),
    period: z.string().optional().describe('Filter by time period')
  }),
  examples: [
    {
      input: {
        strategies: ['ma_cross', 'rsi_strategy', 'macd_trend'],
        metrics: ['profit', 'sharpe', 'max_drawdown'],
        symbol: 'EURUSD',
        timeframe: 'H1'
      },
      output: {
        winner: 'ma_cross',
        ranking: [
          { name: 'ma_cross', score: 8.5 },
          { name: 'rsi_strategy', score: 7.2 },
          { name: 'macd_trend', score: 6.8 }
        ],
        details: {
          'Best for returns': 'ma_cross',
          'Lowest risk': 'rsi_strategy',
          'Most consistent': 'ma_cross'
        }
      },
      description: 'Compare three strategies on EURUSD H1'
    }
  ],
  defaultEnabled: true,
  execute: async ({ strategies, metrics, symbol, timeframe, period }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/analysis/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategies, metrics, symbol, timeframe, period })
      });

      if (!response.ok) {
        throw new Error(`Failed to compare strategies: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          strategies,
          comparison: data.comparison,
          winner: data.winner,
          ranking: data.ranking,
          statisticalTests: data.statisticalTests
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
// NPRD PARSING SKILLS
// ============================================================================

/**
 * Parse an NPRD (Natural Product Requirements Document)
 */
const parseNPRD: Skill = {
  id: 'analyst_parse_nprd',
  name: 'Parse NPRD',
  description: 'Parse a Natural Product Requirements Document to extract strategy logic, entry/exit conditions, risk parameters, and technical indicators.',
  agents: ['analyst'],
  category: 'nprd',
  schema: z.object({
    nprdContent: z.string().describe('NPRD document content as text'),
    extractSections: z.array(z.string()).default(['strategy', 'entry', 'exit', 'risk', 'indicators']).describe('Sections to extract'),
    validateSyntax: z.boolean().default(true).describe('Validate NPRD syntax and structure')
  }),
  examples: [
    {
      input: {
        nprdContent: '# Strategy: Moving Average Crossover\n\n## Entry\nBuy when fast MA crosses above slow MA\n\n## Exit\nSell when fast MA crosses below slow MA\n\n## Risk\nStop loss: 50 pips\nTake profit: 100 pips',
        extractSections: ['strategy', 'entry', 'exit', 'risk']
      },
      output: {
        strategy: {
          name: 'Moving Average Crossover',
          type: 'trend_following'
        },
        entry: {
          long: 'fast MA > slow MA',
          short: 'fast MA < slow MA'
        },
        exit: {
          long: 'fast MA < slow MA',
          short: 'fast MA > slow MA'
        },
        risk: {
          stopLoss: 50,
          takeProfit: 100,
          unit: 'pips'
        }
      },
      description: 'Parse moving average crossover NPRD'
    }
  ],
  defaultEnabled: true,
  execute: async ({ nprdContent, extractSections, validateSyntax }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/nprd/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: nprdContent, sections: extractSections, validate: validateSyntax })
      });

      if (!response.ok) {
        throw new Error(`Failed to parse NPRD: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          parsed: data.parsed,
          sections: data.sections,
          validation: data.validation,
          errors: data.errors
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
 * Validate NPRD syntax
 */
const validateNPRD: Skill = {
  id: 'analyst_validate_nprd',
  name: 'Validate NPRD',
  description: 'Validate NPRD document syntax and structure against the QuantMindX NPRD specification.',
  agents: ['analyst'],
  category: 'nprd',
  schema: z.object({
    nprdContent: z.string().describe('NPRD document content to validate'),
    specVersion: z.string().default('1.0').describe('NPRD specification version'),
    strictMode: z.boolean().default(false).describe('Enable strict validation mode')
  }),
  examples: [
    {
      input: {
        nprdContent: '# Strategy: My Strategy\n\n## Entry\nBuy when RSI < 30',
        specVersion: '1.0'
      },
      output: {
        valid: true,
        warnings: ['Missing exit conditions', 'Missing risk parameters'],
        errors: []
      },
      description: 'Validate NPRD document'
    }
  ],
  defaultEnabled: true,
  execute: async ({ nprdContent, specVersion, strictMode }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/nprd/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: nprdContent, version: specVersion, strict: strictMode })
      });

      if (!response.ok) {
        throw new Error(`Failed to validate NPRD: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          valid: data.valid,
          errors: data.errors,
          warnings: data.warnings,
          suggestions: data.suggestions
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
// TRD GENERATION SKILLS
// ============================================================================

/**
 * Generate TRD (Technical Requirements Document) from NPRD
 */
const generateTRD: Skill = {
  id: 'analyst_generate_trd',
  name: 'Generate TRD',
  description: 'Generate a Technical Requirements Document from a parsed NPRD. Includes MQL5 code structure, parameter definitions, and implementation specifications.',
  agents: ['analyst'],
  category: 'trd',
  schema: z.object({
    nprdData: z.any().describe('Parsed NPRD data object (from parseNPRD)'),
    includeCode: z.boolean().default(true).describe('Include MQL5 code skeleton'),
    includeTests: z.boolean().default(true).describe('Include test scenarios'),
    targetPlatform: z.string().default('MT5').describe('Target platform (MT5 or MT4)')
  }),
  examples: [
    {
      input: {
        nprdData: {
          strategy: { name: 'MA Cross', type: 'trend_following' },
          entry: { long: 'fast MA > slow MA' },
          exit: { long: 'fast MA < slow MA' },
          risk: { stopLoss: 50, takeProfit: 100 }
        },
        includeCode: true,
        includeTests: true
      },
      output: {
        trd: {
          title: 'MA Cross Technical Requirements',
          parameters: [
            { name: 'FastMAPeriod', type: 'int', default: 10 },
            { name: 'SlowMAPeriod', type: 'int', default: 20 },
            { name: 'StopLoss', type: 'int', default: 50 }
          ],
          codeSkeleton: '//+------------------------------------------------------------------+...',
          testScenarios: [
            { name: 'Bullish crossover', expected: 'Open long position' }
          ]
        }
      },
      description: 'Generate TRD from NPRD'
    }
  ],
  defaultEnabled: true,
  execute: async ({ nprdData, includeCode, includeTests, targetPlatform }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/trd/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nprdData, includeCode, includeTests, targetPlatform })
      });

      if (!response.ok) {
        throw new Error(`Failed to generate TRD: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: {
          trd: data.trd,
          parameters: data.parameters,
          codeSkeleton: data.codeSkeleton,
          testScenarios: data.testScenarios
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
// PERFORMANCE EVALUATION SKILLS
// ============================================================================

/**
 * Evaluate strategy performance with advanced metrics
 */
const evaluatePerformance: Skill = {
  id: 'analyst_evaluate_performance',
  name: 'Evaluate Performance',
  description: 'Calculate advanced performance metrics including Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, and value at risk (VaR).',
  agents: ['analyst'],
  category: 'evaluation',
  schema: z.object({
    equityCurve: z.array(z.number()).describe('Array of equity values over time'),
    returns: z.array(z.number()).optional().describe('Array of periodic returns (optional, calculated from equity)'),
    benchmarkReturns: z.array(z.number()).optional().describe('Benchmark returns for comparison'),
    riskFreeRate: z.number().default(0.02).describe('Annual risk-free rate (default: 2%)'),
    confidenceLevel: z.number().default(0.95).describe('Confidence level for VaR (default: 95%)')
  }),
  examples: [
    {
      input: {
        equityCurve: [10000, 10150, 10200, 10300, 10180, 10250],
        riskFreeRate: 0.02,
        confidenceLevel: 0.95
      },
      output: {
        totalReturn: 0.025,
        annualizedReturn: 0.087,
        sharpeRatio: 1.35,
        sortinoRatio: 1.82,
        calmarRatio: 0.95,
        maxDrawdown: 0.092,
        var95: -150.50,
        expectedShortfall: -210.30
      },
      description: 'Calculate performance metrics'
    }
  ],
  defaultEnabled: true,
  execute: async ({ equityCurve, returns, benchmarkReturns, riskFreeRate, confidenceLevel }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/analysis/performance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ equityCurve, returns, benchmarkReturns, riskFreeRate, confidenceLevel })
      });

      if (!response.ok) {
        throw new Error(`Failed to evaluate performance: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: data.metrics
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
 * Generate optimization report
 */
const generateOptimizationReport: Skill = {
  id: 'analyst_optimization_report',
  name: 'Generate Optimization Report',
  description: 'Generate a comprehensive optimization report with parameter sensitivity analysis, correlation heatmaps, and recommendation rankings.',
  agents: ['analyst'],
  category: 'evaluation',
  schema: z.object({
    optimizationId: z.string().describe('Optimization run ID'),
    topResults: z.number().default(10).describe('Number of top results to highlight'),
    includeCharts: z.boolean().default(true).describe('Include visual charts'),
    format: z.string().default('json').describe('Output format (json, html, pdf)')
  }),
  examples: [
    {
      input: {
        optimizationId: 'opt_20240210_ma_cross',
        topResults: 10,
        includeCharts: true,
        format: 'json'
      },
      output: {
        summary: {
          totalRuns: 500,
          bestProfit: 3500.50,
          bestSharpe: 2.15,
          bestParameters: { 'FastMAPeriod': 12, 'SlowMAPeriod': 26 }
        },
        topResults: [...],
        parameterSensitivity: {...},
        recommendations: [
          'Fast MA between 10-14 performs best',
          'Avoid Slow MA > 30 (diminishing returns)'
        ]
      },
      description: 'Generate optimization report'
    }
  ],
  defaultEnabled: true,
  execute: async ({ optimizationId, topResults, includeCharts, format }, context) => {
    try {
      const response = await fetch('http://localhost:8000/api/analysis/optimization-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ optimizationId, topResults, includeCharts, format })
      });

      if (!response.ok) {
        throw new Error(`Failed to generate report: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        success: true,
        data: data.report
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

export const analystSkills: Skill[] = [
  // Strategy Analysis
  analyzeBacktest,
  compareStrategies,

  // NPRD Parsing
  parseNPRD,
  validateNPRD,

  // TRD Generation
  generateTRD,

  // Performance Evaluation
  evaluatePerformance,
  generateOptimizationReport
];
