<script lang="ts">
  /**
   * Risk Canvas
   *
   * Risk Department Head workspace.
   * Tab-navigated tile grid (Physics, Compliance, Calendar, Backtest, Dept Tasks)
   *
   * Frosted Terminal aesthetic — red (#ff3b3b) dept accent, cyan (#00d4ff) highlights.
   * Svelte 5 runes only.
   */
  import { onMount, onDestroy } from 'svelte';
  import {
    Layers,
    Shield,
    Calendar,
    BarChart3,
    Kanban,
    RefreshCw,
    Clock,
    Lightbulb,
    ChevronDown,
    Trophy
  } from 'lucide-svelte';
  import PhysicsSensorGrid from '$lib/components/risk/PhysicsSensorGrid.svelte';
  import ComplianceTile from '$lib/components/risk/ComplianceTile.svelte';
  import PropFirmConfigPanel from '$lib/components/risk/PropFirmConfigPanel.svelte';
  import CalendarGateTile from '$lib/components/risk/CalendarGateTile.svelte';
  import BacktestResultsPanel from '$lib/components/risk/BacktestResultsPanel.svelte';
  import DprLeaderboard from '$lib/components/risk/DprLeaderboard.svelte';
  import DepartmentKanban from '$lib/components/department-kanban/DepartmentKanban.svelte';
  import AgentTilePanel from '$lib/components/AgentTilePanel.svelte';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import { physicsSensorStore, physicsLastUpdated, type PhysicsSensorState } from '$lib/stores/risk';

  // =============================================================================
  // Types
  // =============================================================================

  type RiskTab = 'physics' | 'compliance' | 'calendar' | 'backtest' | 'dept-tasks' | 'dpr';

  // =============================================================================
  // State
  // =============================================================================

  let activeTab = $state<RiskTab>('physics');

  // Physics last-updated
  let lastUpdated = $state<Date | null>(null);
  let physicsState = $state<PhysicsSensorState>({
    data: null,
    loading: false,
    error: null,
    lastUpdated: null
  });

  // Agent insights strip
  let insightsExpanded = $state(false);
  let insightsUnread = $state(0);

  // Tab config
  const tabs: { id: RiskTab; label: string; icon: typeof Layers }[] = [
    { id: 'physics',    label: 'Physics',    icon: Layers    },
    { id: 'compliance', label: 'Compliance', icon: Shield    },
    { id: 'calendar',  label: 'Calendar',   icon: Calendar  },
    { id: 'backtest',  label: 'Backtest',   icon: BarChart3 },
    { id: 'dept-tasks', label: 'Dept Tasks', icon: Kanban   },
    { id: 'dpr',       label: 'DPR',        icon: Trophy    },
  ];

  // =============================================================================
  // Lifecycle
  // =============================================================================

  const unsub = physicsLastUpdated.subscribe(v => { lastUpdated = v; });
  const unsubPhysics = physicsSensorStore.subscribe(v => { physicsState = v; });

  onMount(async () => {
    try {
      await canvasContextService.loadCanvasContext('risk');
    } catch {
      // canvas context is optional
    }
  });

  onDestroy(() => {
    unsub();
    unsubPhysics();
  });

  // =============================================================================
  // Helpers
  // =============================================================================

  function formatTime(date: Date | null): string {
    if (!date) return 'Never';
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  }

  $effect(() => {
    const physicsResources =
      physicsState.data && typeof physicsState.data === 'object'
        ? Object.entries(physicsState.data as Record<string, unknown>).slice(0, 25).map(([key, value]) => ({
            id: `risk:${key}`,
            label: key.replace(/_/g, ' '),
            canvas: 'risk',
            resource_type: 'risk-metric',
            metadata: { value },
          }))
        : [];

    canvasContextService.setRuntimeState('risk', {
      active_tab: activeTab,
      visible_tabs: tabs.map((tab) => tab.id),
      last_updated: lastUpdated?.toISOString() ?? null,
      physics: physicsState.data,
      physics_loading: physicsState.loading,
      physics_error: physicsState.error,
      attachable_resources: [
        ...physicsResources,
        ...tabs.map((tab) => ({
          id: `risk-tab:${tab.id}`,
          label: tab.label,
          canvas: 'risk',
          resource_type: 'risk-tab',
          metadata: { active: activeTab === tab.id },
        })),
      ],
    });
  });

</script>

<div class="risk-canvas" data-dept="risk">

  <!-- =========================================================
       Canvas header
       ========================================================= -->
  <header class="canvas-header">
    <div class="header-left">
      <Shield size={16} class="header-icon" />
      <div class="header-titles">
        <h1 class="canvas-title">Risk</h1>
        <span class="canvas-subtitle">Risk Department</span>
      </div>
    </div>
    <div class="header-right">
      <span class="last-updated">
        <Clock size={11} />
        {formatTime(lastUpdated)}
      </span>
      {#if activeTab === 'physics'}
        <button class="refresh-btn" onclick={() => physicsSensorStore.fetch()}>
          <RefreshCw size={13} />
          Refresh
        </button>
      {/if}
    </div>
  </header>

  <!-- =========================================================
       Tab navigation
       ========================================================= -->
  <nav class="tab-nav" role="tablist">
    {#each tabs as tab}
      <button
        class="tab-btn"
        class:active={activeTab === tab.id}
        role="tab"
        aria-selected={activeTab === tab.id}
        onclick={() => { activeTab = tab.id; }}
      >
        <tab.icon size={13} />
        {tab.label}
      </button>
    {/each}
  </nav>

  <!-- =========================================================
       Tile grid / content area
       ========================================================= -->
  <div class="canvas-body">

    {#if activeTab === 'physics'}
      <PhysicsSensorGrid />

    {:else if activeTab === 'compliance'}
      <div class="compliance-grid">
        <ComplianceTile />
        <PropFirmConfigPanel />
      </div>

    {:else if activeTab === 'calendar'}
      <div class="calendar-grid">
        <CalendarGateTile />
      </div>

    {:else if activeTab === 'backtest'}
      <BacktestResultsPanel />

    {:else if activeTab === 'dept-tasks'}
      <div class="kanban-wrapper">
        <DepartmentKanban department="risk" />
      </div>

    {:else if activeTab === 'dpr'}
      <div class="dpr-grid">
        <DprLeaderboard />
      </div>
    {/if}

  </div>

  <!-- =========================================================
       Agent Insights strip (always visible, collapsible)
       ========================================================= -->
  <div class="agent-insights-strip" class:expanded={insightsExpanded}>
    <button class="insights-toggle" onclick={() => { insightsExpanded = !insightsExpanded; }}>
      <Lightbulb size={12} />
      <span>Agent Insights</span>
      {#if insightsUnread > 0}
        <span class="unread-badge">{insightsUnread}</span>
      {/if}
      <span class:rotated={insightsExpanded} style="display:inline-flex;align-items:center;transition:transform 0.2s;"><ChevronDown size={12} /></span>
    </button>
    {#if insightsExpanded}
      <AgentTilePanel
        canvas="risk"
        maxHeight="200px"
        showHeader={false}
        onUnreadCount={(n: number) => { insightsUnread = n; }}
      />
    {/if}
  </div>

</div>

<style>
  /* =============================================================================
     Shell layout
     ============================================================================= */

  .risk-canvas {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: rgba(10, 14, 22, 0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    color: #e0e0e0;
    overflow: hidden;
  }

  /* =============================================================================
     Canvas header
     ============================================================================= */

  .canvas-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 20px;
    background: rgba(8, 12, 20, 0.5);
    border-bottom: 1px solid rgba(255, 59, 59, 0.15);
    flex-shrink: 0;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #ff3b3b;
  }

  :global(.header-icon) {
    color: #ff3b3b;
    flex-shrink: 0;
  }

  .header-titles {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .canvas-title {
    font-size: 17px;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.95);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.4px;
    line-height: 1.2;
  }

  .canvas-subtitle {
    font-size: 10px;
    color: #ff3b3b;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    opacity: 0.85;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .last-updated {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.35);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 11px;
    background: rgba(255, 59, 59, 0.08);
    border: 1px solid rgba(255, 59, 59, 0.2);
    border-radius: 5px;
    color: #ff3b3b;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.18s ease;
  }

  .refresh-btn:hover {
    background: rgba(255, 59, 59, 0.15);
    border-color: rgba(255, 59, 59, 0.4);
  }

  /* =============================================================================
     Tab navigation
     ============================================================================= */

  .tab-nav {
    display: flex;
    gap: 2px;
    padding: 8px 20px;
    background: rgba(8, 12, 20, 0.3);
    border-bottom: 1px solid rgba(255, 59, 59, 0.1);
    flex-shrink: 0;
    overflow-x: auto;
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 7px 14px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    color: rgba(255, 255, 255, 0.45);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.18s ease;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .tab-btn:hover {
    color: rgba(255, 255, 255, 0.75);
    background: rgba(255, 59, 59, 0.05);
    border-color: rgba(255, 59, 59, 0.12);
  }

  .tab-btn.active {
    color: #ff3b3b;
    background: rgba(255, 59, 59, 0.1);
    border-color: rgba(255, 59, 59, 0.25);
  }

  /* =============================================================================
     Content body
     ============================================================================= */

  .canvas-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    min-height: 0;
  }

  /* When kanban-wrapper is present, remove padding so kanban fills edge-to-edge */
  .canvas-body:has(.kanban-wrapper) {
    padding: 0;
    overflow: hidden;
  }

  .compliance-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  @media (max-width: 1100px) {
    .compliance-grid {
      grid-template-columns: 1fr;
    }
  }

  .calendar-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    align-items: start;
  }

  .dpr-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    height: 100%;
  }

  /* Kanban fills available space */
  .kanban-wrapper {
    display: flex;
    flex-direction: column;
    width: 100%;
    min-width: 0;
    height: 100%;
    min-height: 400px;
  }

  /* Spin animation for loader */
  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }

  /* =============================================================================
     Agent Insights strip
     ============================================================================= */

  .agent-insights-strip {
    flex-shrink: 0;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(8, 13, 20, 0.85);
  }

  .insights-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 6px 16px;
    background: transparent;
    border: none;
    color: rgba(224, 224, 224, 0.45);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: color 0.15s, background 0.15s;
  }

  .insights-toggle:hover {
    color: rgba(224, 224, 224, 0.75);
    background: rgba(255, 255, 255, 0.03);
  }

  .agent-insights-strip.expanded .insights-toggle {
    color: #00d4ff;
  }

  .unread-badge {
    padding: 1px 6px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 10px;
    font-size: 10px;
    color: #00d4ff;
    font-weight: 700;
    line-height: 1.4;
  }

  .insights-toggle .rotated {
    transform: rotate(180deg);
  }
</style>
