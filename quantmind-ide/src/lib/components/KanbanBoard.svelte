<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { RefreshCw, Plus, Settings } from 'lucide-svelte';
  import KanbanColumn from './KanbanColumn.svelte';
  import { getStrategies, type StrategyFolder } from '../api';

  // Column definitions
  interface Column {
    id: 'inbox' | 'processing' | 'extracting' | 'done';
    title: string;
    statusMap: string[];
  }

  const columns: Column[] = [
    { id: 'inbox', title: 'Inbox', statusMap: ['pending'] },
    { id: 'processing', title: 'Processing', statusMap: ['processing'] },
    { id: 'extracting', title: 'Extracting', statusMap: ['ready'] },
    { id: 'done', title: 'Done', statusMap: ['primal'] }
  ];

  // State
  let strategies: StrategyFolder[] = [];
  let isLoading = false;
  let error: string | null = null;
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  // Grouped strategies by column
  $: groupedStrategies = () => {
    const grouped: Record<string, StrategyFolder[]> = {
      inbox: [],
      processing: [],
      extracting: [],
      done: []
    };

    for (const strategy of strategies) {
      // Handle quarantined strategies - show in their current column but marked
      for (const column of columns) {
        if (column.statusMap.includes(strategy.status)) {
          grouped[column.id].push(strategy);
          break;
        }
      }
    }

    return grouped;
  };

  onMount(() => {
    fetchStrategies();
    // Auto-refresh every 30 seconds
    refreshInterval = setInterval(fetchStrategies, 30000);
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  async function fetchStrategies() {
    isLoading = true;
    error = null;

    try {
      strategies = await getStrategies();
    } catch (e) {
      console.error('Failed to fetch strategies:', e);
      error = e instanceof Error ? e.message : 'Failed to fetch strategies';
      strategies = [];
    } finally {
      isLoading = false;
    }
  }

  function handleRefresh() {
    fetchStrategies();
  }

  function handleCreateStrategy() {
    // TODO: Implement create strategy dialog
    console.log('Create new strategy');
  }

  function handleColumnClick(columnId: string) {
    console.log('Column clicked:', columnId);
  }
</script>

<div class="kanban-board">
  <!-- Board Header -->
  <div class="board-header">
    <div class="header-left">
      <h2>Strategy Pipeline</h2>
      <span class="strategy-count">
        {strategies.length} {strategies.length === 1 ? 'strategy' : 'strategies'}
      </span>
    </div>
    <div class="header-actions">
      {#if error}
        <div class="error-banner">
          {error}
        </div>
      {/if}
      <button class="action-btn primary" on:click={handleCreateStrategy} title="Create new strategy">
        <Plus size={16} />
        <span>New</span>
      </button>
      <button class="action-btn" on:click={handleRefresh} title="Refresh" class:loading={isLoading}>
        <RefreshCw size={16} class:spin={isLoading} />
      </button>
      <button class="action-btn" on:click={() => console.log('Settings')} title="Board settings">
        <Settings size={16} />
      </button>
    </div>
  </div>

  <!-- Kanban Columns -->
  <div class="board-content">
    {#each columns as column (column.id)}
      {@const grouped = groupedStrategies()}
      <KanbanColumn
        title={column.title}
        status={column.id}
        strategies={grouped[column.id]}
        isLoading={isLoading && strategies.length === 0}
      />
    {/each}
  </div>
</div>

<style>
  .kanban-board {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #0f172a);
    overflow: hidden;
  }

  .board-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color, #1e293b);
    background: var(--bg-secondary, #1e293b);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary, #f1f5f9);
  }

  .strategy-count {
    padding: 4px 10px;
    background: var(--bg-tertiary, #334155);
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary, #94a3b8);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-tertiary, #334155);
    border: 1px solid var(--border-color, #475569);
    border-radius: 6px;
    color: var(--text-primary, #f1f5f9);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    background: var(--bg-hover, #475569);
    border-color: var(--accent-primary, #3b82f6);
  }

  .action-btn.primary {
    background: var(--accent-primary, #3b82f6);
    border-color: var(--accent-primary, #3b82f6);
    color: white;
  }

  .action-btn.primary:hover {
    background: var(--accent-primary-hover, #2563eb);
    border-color: var(--accent-primary-hover, #2563eb);
  }

  .action-btn :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .error-banner {
    padding: 6px 12px;
    background: var(--accent-danger, #ef4444);
    color: white;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
  }

  .board-content {
    flex: 1;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 16px;
    overflow-x: auto;
    min-height: 0;
  }

  @media (max-width: 1200px) {
    .board-content {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 640px) {
    .board-header {
      flex-direction: column;
      gap: 12px;
      align-items: stretch;
    }

    .header-left {
      justify-content: space-between;
    }

    .header-actions {
      justify-content: space-between;
    }

    .board-content {
      grid-template-columns: 1fr;
    }
  }

  /* Scrollbar styling for board content */
  .board-content::-webkit-scrollbar {
    height: 8px;
  }

  .board-content::-webkit-scrollbar-track {
    background: var(--bg-secondary, #1e293b);
    border-radius: 4px;
  }

  .board-content::-webkit-scrollbar-thumb {
    background: var(--border-color, #475569);
    border-radius: 4px;
  }

  .board-content::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted, #64748b);
  }
</style>
