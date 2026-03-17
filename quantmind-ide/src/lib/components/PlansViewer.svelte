<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { FileText, Calendar, Clock, ChevronRight, X, Search, RefreshCw } from 'lucide-svelte';

  const dispatch = createEventDispatcher();


  interface Props {
    plans?: Array<{
    id: string;
    name: string;
    filename: string;
    path: string;
    modified: string;
    size: number;
  }>;
    selectedPlan?: string | null;
    planContent?: string;
    loading?: boolean;
  }

  let {
    plans = [],
    selectedPlan = null,
    planContent = '',
    loading = false
  }: Props = $props();

  let searchQuery = $state('');

  let filteredPlans = $derived(searchQuery
    ? plans.filter(p => p.name.toLowerCase().includes(searchQuery.toLowerCase()))
    : plans);

  function selectPlan(plan: typeof plans[0]) {
    dispatch('select', plan);
  }

  function closeViewer() {
    dispatch('close');
  }

  function formatDate(dateStr: string): string {
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch {
      return dateStr;
    }
  }

  function formatSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }
</script>

<div class="plans-viewer">
  <div class="plans-sidebar">
    <div class="sidebar-header">
      <h3>Implementation Plans</h3>
      <button class="close-btn" onclick={closeViewer}>
        <X size={18} />
      </button>
    </div>

    <div class="search-box">
      <Search size={16} />
      <input
        type="text"
        placeholder="Search plans..."
        bind:value={searchQuery}
      />
    </div>

    <div class="plans-list">
      {#if loading}
        <div class="loading">Loading plans...</div>
      {:else if filteredPlans.length === 0}
        <div class="empty">No plans found</div>
      {:else}
        {#each filteredPlans as plan}
          <button
            class="plan-item"
            class:selected={selectedPlan === plan.id}
            onclick={() => selectPlan(plan)}
          >
            <FileText size={16} />
            <div class="plan-info">
              <span class="plan-name">{plan.name}</span>
              <span class="plan-meta">
                <Calendar size={12} />
                {formatDate(plan.modified)}
              </span>
            </div>
            <ChevronRight size={16} />
          </button>
        {/each}
      {/if}
    </div>
  </div>

  <div class="plans-content">
    {#if selectedPlan && planContent}
      <div class="content-header">
        <h2>{plans.find(p => p.id === selectedPlan)?.name || selectedPlan}</h2>
      </div>
      <div class="content-body">
        <pre class="markdown-content">{planContent}</pre>
      </div>
    {:else if selectedPlan && loading}
      <div class="loading">Loading content...</div>
    {:else}
      <div class="no-selection">
        <FileText size={48} />
        <p>Select a plan to view its contents</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .plans-viewer {
    display: flex;
    height: 100%;
    background: var(--bg-primary, #1a1a2e);
    color: var(--text-primary, #e4e4e7);
  }

  .plans-sidebar {
    width: 320px;
    border-right: 1px solid var(--border-color, #3f3f46);
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary, #16162a);
  }

  .sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid var(--border-color, #3f3f46);
  }

  .sidebar-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }

  .close-btn {
    background: none;
    border: none;
    color: var(--text-secondary, #a1a1aa);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
  }

  .close-btn:hover {
    background: var(--bg-hover, #27273a);
    color: var(--text-primary, #e4e4e7);
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color, #3f3f46);
    color: var(--text-secondary, #a1a1aa);
  }

  .search-box input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary, #e4e4e7);
    outline: none;
    font-size: 14px;
  }

  .search-box input::placeholder {
    color: var(--text-secondary, #71717a);
  }

  .plans-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  .plan-item {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 12px;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-primary, #e4e4e7);
    cursor: pointer;
    text-align: left;
    transition: background 0.2s;
  }

  .plan-item:hover {
    background: var(--bg-hover, #27273a);
  }

  .plan-item.selected {
    background: var(--accent-color, #6366f1);
    color: white;
  }

  .plan-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }

  .plan-name {
    font-size: 14px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .plan-meta {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary, #a1a1aa);
  }

  .plan-item.selected .plan-meta {
    color: rgba(255, 255, 255, 0.8);
  }

  .plans-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .content-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border-color, #3f3f46);
  }

  .content-header h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
  }

  .content-body {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
  }

  .markdown-content {
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .no-selection {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    color: var(--text-secondary, #71717a);
  }

  .loading, .empty {
    padding: 24px;
    text-align: center;
    color: var(--text-secondary, #71717a);
  }
</style>
