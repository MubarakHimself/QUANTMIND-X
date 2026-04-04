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
  source_path?: string;
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

const typeCategoryAliases: Record<AssetType, string[]> = {
  'docs': ['docs', 'doc', 'articles', 'article', 'books', 'book', 'libraries', 'library', 'risk', 'utils'],
  'strategy-templates': ['templates', 'template', 'strategy-templates', 'strategy_templates'],
  'indicators': ['indicators', 'indicator'],
  'skills': ['skills', 'skill'],
  'flow-components': ['flow-components', 'flow_components', 'flow components'],
  'mcp-configs': ['mcp-configs', 'mcp_configs', 'mcp configs', 'mcp'],
};

function normalizeCategory(value: unknown): string {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, '-');
}

function categoryMatchesType(type: AssetType, category: unknown): boolean {
  const normalized = normalizeCategory(category);
  const aliases = typeCategoryAliases[type];
  return aliases.includes(normalized);
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Map backend asset to frontend format
 */
function mapBackendAsset(asset: Record<string, any>, type: AssetType): SharedAsset {
  return {
    id: asset.id || asset.path || asset.filesystem_path || asset.name,
    name: asset.name || asset.title || asset.id || 'Untitled Asset',
    type: type,
    metadata: {
      version: asset.version || asset.checksum?.slice(0, 8) || '1.0.0',
      usage_count: asset.usage_count || asset.used_by_count || asset.used_in?.length || 0,
      last_updated: asset.last_updated || asset.updated_at || asset.created_at || new Date().toISOString(),
      author: asset.author,
      description: asset.description
    },
    content: asset.content || asset.excerpt,
    language: asset.language,
    source_path: asset.path || asset.filesystem_path
  };
}

function getAssetCategoryToken(asset: Record<string, any>): unknown {
  return asset.category ?? asset.type ?? asset.asset_type;
}

async function listSharedAssetsRaw(): Promise<Record<string, any>[]> {
  try {
    const shared = await apiFetch<Record<string, any>[]>('/assets/shared');
    if (Array.isArray(shared)) return shared;
  } catch {
    // fall through to legacy endpoint
  }
  const legacy = await apiFetch<Record<string, any>[]>('/assets');
  return Array.isArray(legacy) ? legacy : [];
}

/**
 * Get assets by type/category
 * Errors propagate to the caller — no silent mock fallback.
 */
export async function listAssetsByType(type: AssetType): Promise<SharedAsset[]> {
  const backendCategory = typeToBackendCategory[type];
  const preferred = await apiFetch<Record<string, any>[]>(`/assets?category=${backendCategory}`).catch(() => []);
  const byId = new Map<string, SharedAsset>();

  if (Array.isArray(preferred) && preferred.length > 0) {
    for (const asset of preferred) {
      const mapped = mapBackendAsset(asset, type);
      byId.set(mapped.id, mapped);
    }
  }

  const sharedAssets = await listSharedAssetsRaw();
  const filtered = sharedAssets.filter((asset) => categoryMatchesType(type, getAssetCategoryToken(asset)));
  for (const asset of filtered) {
    const mapped = mapBackendAsset(asset, type);
    if (!byId.has(mapped.id)) {
      byId.set(mapped.id, mapped);
    }
  }
  return Array.from(byId.values());
}

/**
 * Get all assets grouped by type
 */
export async function listAllAssets(): Promise<Record<AssetType, SharedAsset[]>> {
  const types: AssetType[] = ['docs', 'strategy-templates', 'indicators', 'skills', 'flow-components', 'mcp-configs'];
  const result: Record<AssetType, SharedAsset[]> = {} as Record<AssetType, SharedAsset[]>;
  const sharedAssets = await listSharedAssetsRaw();

  for (const type of types) {
    const filtered = sharedAssets.filter((asset) => categoryMatchesType(type, getAssetCategoryToken(asset)));
    result[type] = filtered.map((asset) => mapBackendAsset(asset, type));
  }

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
