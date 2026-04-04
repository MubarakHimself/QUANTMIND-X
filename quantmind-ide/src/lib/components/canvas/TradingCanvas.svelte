<script lang="ts">
  /**
   * Trading Canvas (slot 5) — Paper Trading & Backtesting.
   * AC 12-4-1: Real tile grid — no skeleton, no CanvasPlaceholder, no epicNumber
   * AC 12-4-7: Sub-page routing via currentSubPage state
   * Arch-UI-3: No kill switch here — shell-level only
   * Story 12-4, Story 12-6
   */
  import CanvasTileGrid from '$lib/components/shared/CanvasTileGrid.svelte';
  import { onMount } from 'svelte';
  import PaperTradingMonitorTile from '$lib/components/trading/tiles/PaperTradingMonitorTile.svelte';
  import BacktestResultsTile from '$lib/components/trading/tiles/BacktestResultsTile.svelte';
  import EAPerformanceTile from '$lib/components/trading/tiles/EAPerformanceTile.svelte';
  import TradingJournalSummaryTile from '$lib/components/trading/tiles/TradingJournalSummaryTile.svelte';
  import RiskPhysicsSummaryTile from '$lib/components/trading/tiles/RiskPhysicsSummaryTile.svelte';
  import EAPerformanceDetailPage from '$lib/components/trading/tiles/EAPerformanceDetailPage.svelte';
  import BacktestDetailPage from '$lib/components/trading/tiles/BacktestDetailPage.svelte';
  import DeptKanbanTile from '$lib/components/shared/DeptKanbanTile.svelte';
  import DepartmentKanban from '$lib/components/department-kanban/DepartmentKanban.svelte';
  import { canvasContextService } from '$lib/services/canvasContextService';

  type TradingSubPage = 'grid' | 'backtest-detail' | 'ea-performance-detail' | 'dept-kanban';

  let currentSubPage = $state<TradingSubPage>('grid');

  onMount(async () => {
    try {
      await canvasContextService.loadCanvasContext('trading');
    } catch {
      // canvas context is optional
    }
  });

  $effect(() => {
    canvasContextService.setRuntimeState('trading', {
      current_subpage: currentSubPage,
      attachable_resources: [
        {
          id: 'trading:paper-trading-monitor',
          label: 'Paper Trading Monitor',
          canvas: 'trading',
          resource_type: 'tile',
          metadata: {
            destination: 'ea-performance-detail',
          },
        },
        {
          id: 'trading:backtest-results',
          label: 'Backtest Results',
          canvas: 'trading',
          resource_type: 'tile',
          metadata: {
            destination: 'backtest-detail',
          },
        },
        {
          id: 'trading:ea-performance',
          label: 'EA Performance',
          canvas: 'trading',
          resource_type: 'tile',
          metadata: {
            destination: 'ea-performance-detail',
          },
        },
        {
          id: 'trading:department-kanban',
          label: 'Department Task Board',
          canvas: 'trading',
          resource_type: 'kanban',
          metadata: {
            destination: 'dept-kanban',
            department: 'trading',
          },
        },
        {
          id: 'trading:trading-journal',
          label: 'Trading Journal',
          canvas: 'trading',
          resource_type: 'tile',
          metadata: {
            source: '/api/journal/trades',
          },
        },
        {
          id: 'trading:risk-physics',
          label: 'Risk Physics',
          canvas: 'trading',
          resource_type: 'tile',
          metadata: {
            source: '/api/risk/physics',
          },
        },
      ],
    });
  });

</script>

<div class="trading-canvas-wrapper" data-dept="trading">
  <CanvasTileGrid
    title="Trading"
    dept="trading"
    showBackButton={currentSubPage !== 'grid'}
    onBack={() => { currentSubPage = 'grid'; }}
  >
    {#if currentSubPage === 'grid'}
      <PaperTradingMonitorTile
        navigable={true}
        onNavigate={() => { currentSubPage = 'ea-performance-detail'; }}
      />
      <BacktestResultsTile
        navigable={true}
        onNavigate={() => { currentSubPage = 'backtest-detail'; }}
      />
      <EAPerformanceTile />
      <DeptKanbanTile dept="trading" onNavigate={() => { currentSubPage = 'dept-kanban'; }} />
      <TradingJournalSummaryTile />
      <RiskPhysicsSummaryTile />
    {:else if currentSubPage === 'backtest-detail'}
      <BacktestDetailPage />
    {:else if currentSubPage === 'ea-performance-detail'}
      <EAPerformanceDetailPage />
    {:else if currentSubPage === 'dept-kanban'}
      <DepartmentKanban department="trading" onClose={() => { currentSubPage = 'grid'; }} />
    {/if}
  </CanvasTileGrid>

</div>

<style>
  /* Outer wrapper */
  .trading-canvas-wrapper {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }

  /* Give CanvasTileGrid all remaining vertical space */
  .trading-canvas-wrapper :global(.canvas-tile-grid) {
    flex: 1;
    min-height: 0;
  }

  /* Override tile grid columns to dense 240px min, 12px gap for Trading canvas */
  .trading-canvas-wrapper :global(.ctg-grid) {
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
  }

  /* When dept-kanban sub-page is active, the ctg-grid becomes a single-child
     container — make the kanban board fill the full grid area */
  .trading-canvas-wrapper :global(.ctg-grid:has(.department-kanban)) {
    display: flex;
    flex-direction: column;
    padding: 0;
    overflow: hidden;
  }

  .trading-canvas-wrapper :global(.department-kanban) {
    width: 100%;
    min-width: 0;
  }

  /* Fix kanban board overflow */
  .trading-canvas-wrapper :global(.kanban-board) {
    overflow-x: auto;
    width: 100%;
    min-width: 0;
  }
  /* Spin animation for loading icon */
  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
