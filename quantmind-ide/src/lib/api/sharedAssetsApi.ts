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

export type AssetType = 'docs' | 'strategy-templates' | 'indicators' | 'skills' | 'flow-components' | 'mcp-configs' | 'strategies';

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
  details?: Record<string, any>;
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
  'mcp-configs': { label: 'MCP Configs', icon: 'Settings' },
  'strategies': { label: 'Strategies', icon: 'FolderTree' }
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
  'mcp-configs': 'mcp-configs',
  'strategies': 'strategies'
};

const typeCategoryAliases: Record<AssetType, string[]> = {
  'docs': ['docs', 'doc', 'articles', 'article', 'books', 'book', 'libraries', 'library', 'risk', 'utils'],
  'strategy-templates': ['templates', 'template', 'strategy-templates', 'strategy_templates'],
  'indicators': ['indicators', 'indicator'],
  'skills': ['skills', 'skill'],
  'flow-components': ['flow-components', 'flow_components', 'flow components'],
  'mcp-configs': ['mcp-configs', 'mcp_configs', 'mcp configs', 'mcp'],
  'strategies': ['strategies', 'strategy', 'wf1', 'workflow-artifacts', 'workflow_artifacts'],
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

function describeStrategy(detail: Record<string, any> | undefined, fallback: string | undefined): string | undefined {
  if (!detail) return fallback;
  const signals = [
    detail.has_video_ingest ? 'video ingest' : null,
    detail.has_trd ? 'trd' : null,
    detail.has_ea ? 'ea' : null,
    detail.has_backtest ? 'backtest' : null,
    detail.has_source_captions ? 'captions' : null,
    detail.has_source_audio ? 'audio' : null,
    detail.has_chunk_manifest ? 'chunks' : null,
  ].filter(Boolean);
  if (signals.length === 0) {
    return fallback || `status: ${detail.status || 'pending'}`;
  }
  return `${detail.status || 'pending'} · ${signals.join(' · ')}`;
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

  if (type === 'strategies') {
    const strategyDetails = await apiFetch<Record<string, any>[]>('/strategies').catch(() => []);
    const detailByAssetId = new Map(
      strategyDetails.map((detail) => [String(detail.asset_id || detail.relative_root || detail.id || detail.name), detail])
    );
    if (strategyDetails.length > 0) {
      for (const asset of preferred) {
        const detail = detailByAssetId.get(String(asset.id));
        const mapped = mapBackendAsset(
          {
            ...asset,
            description: describeStrategy(detail, asset.description),
          },
          type,
        );
        mapped.details = detail;
        byId.set(mapped.id, mapped);
      }
      for (const detail of strategyDetails) {
        const detailAssetId = String(detail.asset_id || detail.relative_root || detail.id || detail.name);
        const existing = byId.get(detailAssetId)
          || Array.from(byId.values()).find((asset) => asset.name === detail.name || asset.id === detailAssetId);
        if (existing) {
          existing.details = detail;
          existing.metadata.description = describeStrategy(detail, existing.metadata.description);
          continue;
        }
        const mapped = mapBackendAsset(
          {
            id: detailAssetId.startsWith('strategies/') ? detailAssetId : `strategies/${detailAssetId}`,
            name: detail.name || detail.id,
            type: 'strategies',
            description: describeStrategy(detail, undefined),
            updated_at: detail.created_at,
          },
          type,
        );
        mapped.details = detail;
        byId.set(mapped.id, mapped);
      }
    }
    return Array.from(byId.values());
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
  const types: AssetType[] = ['docs', 'strategy-templates', 'indicators', 'skills', 'flow-components', 'mcp-configs', 'strategies'];
  const entries = await Promise.all(
    types.map(async (type) => [type, await listAssetsByType(type)] as const)
  );
  const result = Object.fromEntries(entries) as Record<AssetType, SharedAsset[]>;
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
  return apiFetch<Record<AssetType, number>>('/assets/counts');
}
