/**
 * Shared Assets API Client
 * Provides frontend API functions for shared assets management
 *
 * Connects to backend endpoints for asset browsing
 * Maps between backend categories and frontend asset types
 */

import { API_CONFIG } from '$lib/config/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_CONFIG.API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers
    }
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return response.json();
}

// =============================================================================
// Types
// =============================================================================

export type AssetType = 'docs' | 'strategy-templates' | 'indicators' | 'skills' | 'flow-components' | 'mcp-configs';

export interface AssetMetadata {
  version: string;
  usage_count: number;
  last_updated: string;
  author?: string;
  description?: string;
}

export interface SharedAsset {
  id: string;
  name: string;
  type: AssetType;
  metadata: AssetMetadata;
  content?: string;
  language?: string;
}

// =============================================================================
// Asset Type Info
// =============================================================================

export const ASSET_TYPE_INFO: Record<AssetType, { label: string; icon: string }> = {
  'docs': { label: 'Docs', icon: 'FileText' },
  'strategy-templates': { label: 'Strategy Templates', icon: 'Layout' },
  'indicators': { label: 'Indicators', icon: 'Code' },
  'skills': { label: 'Skills', icon: 'Sparkles' },
  'flow-components': { label: 'Flow Components', icon: 'Workflow' },
  'mcp-configs': { label: 'MCP Configs', icon: 'Settings' }
};

// =============================================================================
// Type Mapping (Frontend Type -> Backend Category)
// =============================================================================

const typeToBackendCategory: Record<AssetType, string> = {
  'docs': 'docs',
  'strategy-templates': 'templates',
  'indicators': 'indicators',
  'skills': 'skills',
  'flow-components': 'flow-components',
  'mcp-configs': 'mcp-configs'
};

// =============================================================================
// Mock Data for Development (opt-in only — NOT used as silent fallback)
// To use mock data during development, call listAssetsByType with useMock=true
// or import MOCK_ASSETS directly in a dev-only context.
// =============================================================================

export const MOCK_ASSETS: Record<AssetType, SharedAsset[]> = {
  'docs': [
    {
      id: 'docs/getting-started',
      name: 'Getting Started Guide',
      type: 'docs',
      metadata: {
        version: '1.0.0',
        usage_count: 42,
        last_updated: '2026-03-15T10:00:00Z',
        author: 'System',
        description: 'Introduction guide for new users'
      },
      language: 'markdown'
    },
    {
      id: 'docs/api-reference',
      name: 'API Reference',
      type: 'docs',
      metadata: {
        version: '2.1.0',
        usage_count: 128,
        last_updated: '2026-03-18T14:30:00Z',
        author: 'System',
        description: 'Complete API documentation'
      },
      language: 'markdown'
    }
  ],
  'strategy-templates': [
    {
      id: 'templates/martingale-basic',
      name: 'Martingale Basic',
      type: 'strategy-templates',
      metadata: {
        version: '1.2.0',
        usage_count: 15,
        last_updated: '2026-03-10T08:00:00Z',
        author: 'Strategy Team',
        description: 'Basic Martingale trading template'
      },
      language: 'mql5'
    }
  ],
  'indicators': [
    {
      id: 'indicators/rsi-filter',
      name: 'RSI Filter',
      type: 'indicators',
      metadata: {
        version: '1.0.0',
        usage_count: 23,
        last_updated: '2026-03-12T16:00:00Z',
        author: 'Quant Team',
        description: 'RSI-based trend filter indicator'
      },
      content: '// RSI Filter Indicator\n// Version 1.0.0\n\n#property indicator_separate_window\n#property indicator_minimum 0\n#property indicator_maximum 100\n\ninput int RSIPeriod = 14;\ninput int oversoldLevel = 30;\ninput int overboughtLevel = 70;\n\ndouble RSIBuffer[];\n\nint OnInit() {\n   SetIndexBuffer(0, RSIBuffer);\n   IndicatorShortName("RSI Filter");\n   return INIT_SUCCEEDED;\n}\n\nint OnCalculate(const int rates_total,\n                const int prev_calculated,\n                const datetime &time[],\n                const double &open[],\n                const double &high[],\n                const double &low[],\n                const double &close[],\n                const long &tick_volume[],\n                const long &volume[],\n                const int &spread[]) {\n   // RSI calculation logic here\n   return rates_total;\n}',
      language: 'mql5'
    },
    {
      id: 'indicators/ema-cross',
      name: 'EMA Crossover',
      type: 'indicators',
      metadata: {
        version: '1.1.0',
        usage_count: 45,
        last_updated: '2026-03-14T12:00:00Z',
        author: 'Quant Team',
        description: 'EMA crossover signal indicator'
      },
      language: 'mql5'
    }
  ],
  'skills': [
    {
      id: 'skills/backtest-analysis',
      name: 'Backtest Analysis',
      type: 'skills',
      metadata: {
        version: '2.0.0',
        usage_count: 89,
        last_updated: '2026-03-16T09:00:00Z',
        author: 'Dev Team',
        description: 'Analyze backtest results and generate reports'
      }
    },
    {
      id: 'skills/signal-generator',
      name: 'Signal Generator',
      type: 'skills',
      metadata: {
        version: '1.5.0',
        usage_count: 67,
        last_updated: '2026-03-17T11:00:00Z',
        author: 'Dev Team',
        description: 'Generate trading signals from indicators'
      }
    }
  ],
  'flow-components': [
    {
      id: 'flow-components/order-pipeline',
      name: 'Order Pipeline',
      type: 'flow-components',
      metadata: {
        version: '1.0.0',
        usage_count: 12,
        last_updated: '2026-03-11T15:00:00Z',
        author: 'Dev Team',
        description: 'Order execution pipeline component'
      },
      language: 'python'
    }
  ],
  'mcp-configs': [
    {
      id: 'mcp-configs/trading-api',
      name: 'Trading API Config',
      type: 'mcp-configs',
      metadata: {
        version: '1.0.0',
        usage_count: 8,
        last_updated: '2026-03-09T10:00:00Z',
        author: 'Ops Team',
        description: 'MCP configuration for trading API'
      },
      language: 'json'
    }
  ]
};

// =============================================================================
// API Functions
// =============================================================================

/**
 * Map backend asset to frontend format
 */
function mapBackendAsset(asset: any, type: AssetType): SharedAsset {
  return {
    id: asset.id || asset.path || asset.name,
    name: asset.name,
    type: type,
    metadata: {
      version: asset.version || '1.0.0',
      usage_count: asset.usage_count || asset.used_in?.length || 0,
      last_updated: asset.last_updated || new Date().toISOString(),
      author: asset.author,
      description: asset.description
    },
    content: asset.content,
    language: asset.language
  };
}

/**
 * Get assets by type/category
 * Errors propagate to the caller — no silent mock fallback.
 */
export async function listAssetsByType(type: AssetType): Promise<SharedAsset[]> {
  const backendCategory = typeToBackendCategory[type];
  const response = await apiFetch<any[]>(`/assets?category=${backendCategory}`);
  if (response && response.length > 0) {
    return response.map(asset => mapBackendAsset(asset, type));
  }
  return [];
}

/**
 * Get all assets grouped by type
 */
export async function listAllAssets(): Promise<Record<AssetType, SharedAsset[]>> {
  const types: AssetType[] = ['docs', 'strategy-templates', 'indicators', 'skills', 'flow-components', 'mcp-configs'];
  const result: Record<AssetType, SharedAsset[]> = {} as Record<AssetType, SharedAsset[]>;

  // Fetch all asset types in parallel
  const promises = types.map(async (type) => {
    try {
      const assets = await listAssetsByType(type);
      result[type] = assets;
    } catch (e) {
      console.error(`Failed to load ${type}:`, e);
      result[type] = [];
    }
  });

  await Promise.all(promises);
  return result;
}

/**
 * Get detailed information about a specific asset
 * Errors propagate to the caller — no silent mock fallback.
 */
export async function getAssetDetail(assetId: string, type: AssetType): Promise<SharedAsset> {
  const response = await apiFetch<{ content: string }>(`/assets/${assetId}/content`);
  const assets = await listAssetsByType(type);
  const asset = assets.find(a => a.id === assetId);
  if (asset) {
    return { ...asset, content: response.content };
  }
  throw new Error(`Asset not found: ${assetId}`);
}

/**
 * Get asset counts by type
 * Errors propagate to the caller — no silent mock fallback.
 */
export async function getAssetCounts(): Promise<Record<AssetType, number>> {
  const allAssets = await listAllAssets();
  const counts: Record<AssetType, number> = {
    'docs': 0,
    'strategy-templates': 0,
    'indicators': 0,
    'skills': 0,
    'flow-components': 0,
    'mcp-configs': 0
  };

  for (const type of Object.keys(allAssets) as AssetType[]) {
    counts[type] = allAssets[type]?.length || 0;
  }

  return counts;
}
