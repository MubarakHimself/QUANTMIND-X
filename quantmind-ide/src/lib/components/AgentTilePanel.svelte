<script lang="ts">
  /**
   * AgentTilePanel
   *
   * Compact panel that polls /api/agent-tiles for a given canvas and renders
   * agent-generated insight cards (insight, alert, metric, hypothesis, report).
   *
   * Frosted Terminal aesthetic — bg rgba(8,13,20,0.92), glass rgba(8,13,20,0.35).
   * Svelte 5 runes only — no export let, no $:.
   */
  import { onMount, onDestroy } from 'svelte';
  import {
    Lightbulb,
    AlertTriangle,
    TrendingUp,
    FlaskConical,
    FileText,
    X,
    Inbox
  } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  // =============================================================================
  // Types
  // =============================================================================

  type TileType = 'insight' | 'alert' | 'metric' | 'hypothesis' | 'report';

  interface AgentTile {
    id: string;
    tile_type: TileType;
    title: string;
    content: string;
    canvas: string;
    department: string;
    created_at: string;
    is_read: boolean;
    metadata?: Record<string, unknown>;
  }

  // =============================================================================
  // Props
  // =============================================================================

  interface Props {
    canvas: string;
    maxHeight?: string;
    showHeader?: boolean;
    /** Optional callback to report unread count to parent (e.g. for a badge) */
    onUnreadCount?: (count: number) => void;
  }

  let { canvas, maxHeight = '300px', showHeader = true, onUnreadCount }: Props = $props();

  // =============================================================================
  // State
  // =============================================================================

  let tiles = $state<AgentTile[]>([]);
  let loading = $state(false);
  let pollInterval: ReturnType<typeof setInterval> | null = null;

  // =============================================================================
  // Derived
  // =============================================================================

  let unreadCount = $derived(tiles.filter(t => !t.is_read).length);

  // Notify parent of unread count changes
  $effect(() => {
    onUnreadCount?.(unreadCount);
  });

  // =============================================================================
  // Tile type config
  // =============================================================================

  type TileConfig = {
    icon: typeof Lightbulb;
    borderColor: string;
    iconColor: string;
  };

  const TYPE_CONFIG: Record<TileType, TileConfig> = {
    insight:    { icon: Lightbulb,     borderColor: '#00d4ff', iconColor: '#00d4ff' },
    alert:      { icon: AlertTriangle, borderColor: '#ff3b3b', iconColor: '#ff3b3b' },
    metric:     { icon: TrendingUp,    borderColor: '#00c896', iconColor: '#00c896' },
    hypothesis: { icon: FlaskConical,  borderColor: '#f0a500', iconColor: '#f0a500' },
    report:     { icon: FileText,      borderColor: '#00d4ff', iconColor: '#00d4ff' },
  };

  function getConfig(type: TileType): TileConfig {
    return TYPE_CONFIG[type] ?? TYPE_CONFIG.insight;
  }

  // =============================================================================
  // Relative time
  // =============================================================================

  function relativeTime(iso: string): string {
    const diff = Date.now() - new Date(iso).getTime();
    const s = Math.floor(diff / 1000);
    if (s < 60)  return `${s}s ago`;
    const m = Math.floor(s / 60);
    if (m < 60)  return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24)  return `${h}h ago`;
    const d = Math.floor(h / 24);
    return `${d}d ago`;
  }

  // =============================================================================
  // API helpers
  // =============================================================================

  async function fetchTiles() {
    if (loading) return;
    loading = true;
    try {
      const data = await apiFetch<AgentTile[]>(`/agent-tiles?canvas=${encodeURIComponent(canvas)}&limit=20`);
      tiles = Array.isArray(data) ? data : [];
    } catch {
      // Backend unavailable — silently keep existing tiles
    } finally {
      loading = false;
    }
  }

  async function dismissTile(id: string, e: MouseEvent) {
    e.stopPropagation();
    try {
      await apiFetch(`/agent-tiles/${encodeURIComponent(id)}`, { method: 'DELETE' });
      tiles = tiles.filter(t => t.id !== id);
    } catch {
      // Optimistic removal anyway
      tiles = tiles.filter(t => t.id !== id);
    }
  }

  async function markRead(tile: AgentTile) {
    if (tile.is_read) return;
    try {
      await apiFetch(`/agent-tiles/${encodeURIComponent(tile.id)}/read`, { method: 'POST' });
      tiles = tiles.map(t => t.id === tile.id ? { ...t, is_read: true } : t);
    } catch {
      tiles = tiles.map(t => t.id === tile.id ? { ...t, is_read: true } : t);
    }
  }

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(() => {
    fetchTiles();
    pollInterval = setInterval(fetchTiles, 30_000);
  });

  onDestroy(() => {
    if (pollInterval !== null) clearInterval(pollInterval);
  });
</script>

<div class="agent-tile-panel">
  {#if showHeader}
    <div class="panel-header">
      <Lightbulb size={12} class="header-icon" />
      <span class="header-label">Agent Insights</span>
      {#if unreadCount > 0}
        <span class="header-unread">{unreadCount}</span>
      {/if}
    </div>
  {/if}

  <div class="tile-list" style="max-height: {maxHeight};">
    {#if tiles.length === 0}
      <div class="empty-state">
        <Inbox size={20} class="empty-icon" />
        <span>No agent insights yet</span>
      </div>
    {:else}
      {#each tiles as tile (tile.id)}
        {@const cfg = getConfig(tile.tile_type)}
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <div
          class="tile-card"
          class:unread={!tile.is_read}
          style="--border-color: {cfg.borderColor};"
          onclick={() => markRead(tile)}
        >
          <!-- Unread dot -->
          {#if !tile.is_read}
            <span class="unread-dot" style="background:{cfg.borderColor};box-shadow:0 0 5px {cfg.borderColor};"></span>
          {/if}

          <!-- Icon + title row -->
          <div class="tile-top">
            <svelte:component this={cfg.icon} size={13} style="color:{cfg.iconColor};flex-shrink:0;" />
            <span class="tile-title">{tile.title}</span>
            <button
              class="dismiss-btn"
              onclick={(e) => dismissTile(tile.id, e)}
              title="Dismiss"
              aria-label="Dismiss tile"
            >
              <X size={11} />
            </button>
          </div>

          <!-- Content -->
          <p class="tile-content">{tile.content}</p>

          <!-- Footer: dept badge + time -->
          <div class="tile-footer">
            <span class="dept-badge" style="border-color: {cfg.borderColor}33; color:{cfg.borderColor};">
              {tile.department.toUpperCase()}
            </span>
            <span class="tile-time">{relativeTime(tile.created_at)}</span>
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  /* =============================================================================
     Panel shell
     ============================================================================= */

  .agent-tile-panel {
    display: flex;
    flex-direction: column;
    background: rgba(8, 13, 20, 0.92);
    font-family: 'JetBrains Mono', monospace;
    color: #e0e0e0;
    width: 100%;
  }

  /* =============================================================================
     Panel header (shown when showHeader=true)
     ============================================================================= */

  .panel-header {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 8px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    flex-shrink: 0;
  }

  :global(.header-icon) {
    color: #00d4ff;
    flex-shrink: 0;
  }

  .header-label {
    font-size: 11px;
    color: rgba(224, 224, 224, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .header-unread {
    margin-left: 4px;
    padding: 1px 6px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 10px;
    font-size: 10px;
    color: #00d4ff;
    font-weight: 700;
    line-height: 1.4;
  }

  /* =============================================================================
     Scrollable list
     ============================================================================= */

  .tile-list {
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 10px;
  }

  .tile-list::-webkit-scrollbar {
    width: 3px;
  }

  .tile-list::-webkit-scrollbar-track {
    background: transparent;
  }

  .tile-list::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.15);
    border-radius: 2px;
  }

  /* =============================================================================
     Tile card
     ============================================================================= */

  .tile-card {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 9px 10px 8px 12px;
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 3px solid var(--border-color, #00d4ff);
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.14s, border-color 0.14s;
  }

  .tile-card:hover {
    background: rgba(255, 255, 255, 0.04);
  }

  .tile-card.unread {
    background: rgba(255, 255, 255, 0.03);
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.04);
  }

  /* =============================================================================
     Unread dot indicator
     ============================================================================= */

  .unread-dot {
    position: absolute;
    top: 8px;
    right: 30px;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* =============================================================================
     Tile top row
     ============================================================================= */

  .tile-top {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    padding-right: 20px; /* space for dismiss button */
  }

  .tile-title {
    flex: 1;
    font-size: 12px;
    font-weight: 600;
    color: rgba(224, 224, 224, 0.9);
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  /* =============================================================================
     Dismiss button
     ============================================================================= */

  .dismiss-btn {
    position: absolute;
    top: 7px;
    right: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    background: transparent;
    border: none;
    border-radius: 3px;
    color: rgba(224, 224, 224, 0.3);
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
    padding: 0;
  }

  .dismiss-btn:hover {
    background: rgba(255, 59, 59, 0.12);
    color: #ff3b3b;
  }

  /* =============================================================================
     Content text
     ============================================================================= */

  .tile-content {
    font-size: 11px;
    color: rgba(224, 224, 224, 0.55);
    line-height: 1.5;
    margin: 0;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  /* =============================================================================
     Footer
     ============================================================================= */

  .tile-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 2px;
  }

  .dept-badge {
    display: inline-flex;
    align-items: center;
    padding: 1px 6px;
    background: transparent;
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 3px;
    font-size: 9px;
    color: #00d4ff;
    letter-spacing: 0.05em;
    line-height: 1.5;
  }

  .tile-time {
    font-size: 10px;
    color: rgba(224, 224, 224, 0.28);
    margin-left: auto;
  }

  /* =============================================================================
     Empty state
     ============================================================================= */

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 24px 16px;
    color: rgba(224, 224, 224, 0.25);
    font-size: 11px;
    text-align: center;
  }

  :global(.empty-icon) {
    color: rgba(224, 224, 224, 0.15);
  }
</style>
