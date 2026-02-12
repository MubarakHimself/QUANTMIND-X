<script lang="ts">
  import { onMount } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import { ChevronRight, ChevronDown, File, Folder, FolderOpen, RefreshCw } from 'lucide-svelte';
  import { navigationStore } from '../stores/navigationStore';
  import Breadcrumbs from './Breadcrumbs.svelte';

  export let activeView = 'knowledge';

  const dispatch = createEventDispatcher();
  
  // API base URL
  const API_BASE = 'http://localhost:8000/api';
  
  // Tree data state
  let treeData: Record<string, any[]> = {
    knowledge: [],
    assets: [],
    ea: [],
    backtest: [],
    live: [],
    settings: []
  };
  
  let expandedFolders: Set<string> = new Set();
  let loading = false;
  let error = '';
  
  const viewConfig: Record<string, {title: string, endpoint: string}> = {
    knowledge: { title: 'Knowledge Hub', endpoint: '/knowledge' },
    assets: { title: 'Shared Assets', endpoint: '/assets' },
    ea: { title: 'EA Management', endpoint: '/strategies' },
    backtest: { title: 'Backtests', endpoint: '/strategies' },
    live: { title: 'Live Trading', endpoint: '/trading/bots' },
    settings: { title: 'Settings', endpoint: '' }
  };
  
  onMount(() => {
    loadData(activeView);
    // Initialize navigation with the current view
    navigationStore.navigateToView(activeView, viewConfig[activeView]?.title || 'Explorer');
  });

  $: if (activeView) {
    loadData(activeView);
    // Update navigation when view changes
    navigationStore.navigateToView(activeView, viewConfig[activeView]?.title || 'Explorer');
  }
  
  async function loadData(view: string) {
    const config = viewConfig[view];
    if (!config) {
      // Use fallback for unknown views
      treeData[view] = getFallbackData(view);
      treeData = treeData; // Trigger reactivity
      return;
    }
    
    // If no endpoint configured (like settings), use fallback directly
    if (!config.endpoint) {
      treeData[view] = getFallbackData(view);
      treeData = treeData;
      return;
    }
    
    loading = true;
    error = '';
    
    try {
      const response = await fetch(`${API_BASE}${config.endpoint}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      
      // Transform data based on view type
      if (view === 'knowledge') {
        treeData.knowledge = groupByCategory(data, 'category');
      } else if (view === 'assets') {
        treeData.assets = groupByCategory(data, 'type');
      } else if (view === 'ea' || view === 'backtest') {
        treeData[view] = transformStrategies(data);
      } else if (view === 'live') {
        // Transform bots data
        treeData.live = [{
          id: 'active-bots',
          name: `Active Bots (${data.length})`,
          type: 'folder',
          children: data.map((bot: any) => ({
            id: `bot/${bot.id}`,
            name: `${bot.name} @${bot.symbol}`,
            type: 'bot',
            status: bot.state
          }))
        }];
      } else {
        treeData[view] = data;
      }
      treeData = treeData; // Trigger reactivity
    } catch (e: any) {
      error = e.message || 'Failed to load data';
      // Use fallback data for demo
      treeData[view] = getFallbackData(view);
      treeData = treeData; // Trigger reactivity
    } finally {
      loading = false;
    }
  }
  
  function groupByCategory(items: any[], key: string) {
    const groups: Record<string, any[]> = {};
    
    for (const item of items) {
      const category = item[key] || 'Other';
      if (!groups[category]) groups[category] = [];
      groups[category].push(item);
    }
    
    return Object.entries(groups).map(([name, children]) => ({
      id: name,
      name: name.charAt(0).toUpperCase() + name.slice(1),
      type: 'folder',
      children: children.map(c => ({
        id: c.id,
        name: c.name,
        type: 'file',
        path: c.path
      }))
    }));
  }
  
  function transformStrategies(strategies: any[]) {
    return strategies.map(s => ({
      id: s.id,
      name: s.name,
      type: 'folder',
      status: s.status,
      children: [
        ...(s.has_nprd ? [{ id: `${s.id}/nprd`, name: 'NPRD Output', type: 'folder', children: [] }] : []),
        ...(s.has_trd ? [{ id: `${s.id}/trd`, name: 'TRD', type: 'folder', children: [] }] : []),
        ...(s.has_ea ? [{ id: `${s.id}/ea`, name: 'EA Code', type: 'folder', children: [] }] : []),
        ...(s.has_backtest ? [{ id: `${s.id}/backtest`, name: 'Backtest Reports', type: 'folder', children: [] }] : [])
      ]
    }));
  }
  
  function getFallbackData(view: string) {
    const fallbacks: Record<string, any[]> = {
      knowledge: [
        { id: 'articles', name: 'Articles', type: 'folder', children: [
          { id: 'articles/ict-concepts', name: 'ICT Concepts.pdf', type: 'file' },
          { id: 'articles/smc-basics', name: 'SMC Basics.md', type: 'file' }
        ]},
        { id: 'books', name: 'Books', type: 'folder', children: [
          { id: 'books/trading-zone', name: 'Trading in the Zone.pdf', type: 'file' }
        ]},
        { id: 'logs', name: 'Strategy Notes', type: 'folder', children: [] }
      ],
      assets: [
        { id: 'indicators', name: 'Indicators', type: 'folder', children: [
          { id: 'indicators/atr-filter', name: 'ATR_Filter.mqh', type: 'file' },
          { id: 'indicators/session-filter', name: 'SessionFilter.mqh', type: 'file' }
        ]},
        { id: 'libraries', name: 'Libraries', type: 'folder', children: [
          { id: 'libraries/risk-manager', name: 'RiskManager.mqh', type: 'file' },
          { id: 'libraries/order-manager', name: 'OrderManager.mqh', type: 'file' }
        ]}
      ],
      ea: [
        { id: 'ict-scalper', name: 'ICT Scalper v2', type: 'folder', status: 'primal', children: [
          { id: 'ict-scalper/nprd', name: 'NPRD Output', type: 'folder', children: [
            { id: 'ict-scalper/nprd/transcript', name: 'transcript.md', type: 'file' }
          ]},
          { id: 'ict-scalper/trd', name: 'TRD', type: 'folder', children: [
            { id: 'ict-scalper/trd/spec', name: 'strategy_spec.md', type: 'file' }
          ]},
          { id: 'ict-scalper/ea', name: 'EA Code', type: 'folder', children: [
            { id: 'ict-scalper/ea/main', name: 'ICT_Scalper_v2.mq5', type: 'file' }
          ]},
          { id: 'ict-scalper/backtest', name: 'Backtest Reports', type: 'folder', children: [
            { id: 'ict-scalper/backtest/a', name: 'mode_a_report.html', type: 'file' },
            { id: 'ict-scalper/backtest/b', name: 'mode_b_report.html', type: 'file' },
            { id: 'ict-scalper/backtest/c', name: 'mode_c_report.html', type: 'file' }
          ]}
        ]},
        { id: 'smc-reversal', name: 'SMC Reversal', type: 'folder', status: 'pending', children: [
          { id: 'smc-reversal/nprd', name: 'NPRD Output', type: 'folder', children: [] }
        ]},
        { id: 'new-strategy', name: '+ New Strategy (NPRD)', type: 'action', action: 'new-nprd' }
      ],
      backtest: [
        { id: 'recent', name: 'Recent', type: 'folder', children: [
          { id: 'recent/ict-2025', name: 'ICT_Scalper_2025-02.html', type: 'file' },
          { id: 'recent/smc-eurusd', name: 'SMC_EURUSD.html', type: 'file' }
        ]},
        { id: 'scheduled', name: 'Scheduled', type: 'folder', children: [] }
      ],
      live: [
        { id: 'active-bots', name: 'Active Bots (3)', type: 'folder', children: [
          { id: 'bot/ict-eurusd', name: 'ICT_Scalper @EURUSD', type: 'bot', status: 'primal' },
          { id: 'bot/ict-gbpusd', name: 'ICT_Scalper @GBPUSD', type: 'bot', status: 'primal' },
          { id: 'bot/smc-usdjpy', name: 'SMC_Rev @USDJPY', type: 'bot', status: 'ready' }
        ]},
        { id: 'paused-bots', name: 'Paused', type: 'folder', children: [] },
        { id: 'quarantined', name: 'Quarantined', type: 'folder', children: [] }
      ],
      settings: [
        { id: 'ai-settings', name: 'AI & Agents', type: 'folder', children: [
          { id: 'settings/models', name: 'Model Selection', type: 'setting' },
          { id: 'settings/yolo', name: 'YOLO Mode', type: 'setting' },
          { id: 'settings/agents', name: 'Agent Prompts', type: 'setting' }
        ]},
        { id: 'risk-settings', name: 'Risk & Governor', type: 'folder', children: [
          { id: 'settings/tiers', name: 'Balance Tiers', type: 'setting' },
          { id: 'settings/kelly', name: 'Kelly Parameters', type: 'setting' },
          { id: 'settings/squad', name: 'Squad Limit', type: 'setting' }
        ]},
        { id: 'broker-settings', name: 'Brokers', type: 'folder', children: [
          { id: 'settings/roboforex', name: 'RoboForex Prime', type: 'setting' },
          { id: 'settings/exness', name: 'Exness Raw', type: 'setting' }
        ]},
        { id: 'data-settings', name: 'Data & Storage', type: 'folder', children: [
          { id: 'settings/paths', name: 'Directory Paths', type: 'setting' },
          { id: 'settings/pageindex', name: 'PageIndex Config', type: 'setting' }
        ]}
      ]
    };
    return fallbacks[view] || [];
  }
  
  function toggleFolder(folderId: string) {
    if (expandedFolders.has(folderId)) {
      expandedFolders.delete(folderId);
    } else {
      expandedFolders.add(folderId);
    }
    expandedFolders = expandedFolders;
  }
  
  // Helper to extract parent path from a folder ID
  function getParentPath(folderId: string): string {
    const parts = folderId.split('/');
    parts.pop(); // Remove the last element (current folder)
    return parts.join('/');
  }

  function handleItemClick(item: any) {
    if (item.type === 'folder') {
      // Check if this is a top-level strategy folder (in EA view)
      // Top-level strategy IDs don't contain '/' (e.g., 'ict-scalper', 'smc-reversal')
      // Nested folders have '/' (e.g., 'ict-scalper/nprd', 'ict-scalper/trd')
      const isTopLevelStrategy = activeView === 'ea' && !item.id.includes('/');

      if (isTopLevelStrategy) {
        // Navigate to strategy detail view
        navigationStore.navigateToStrategy(item.id, item.name);
        // Auto-expand to show children
        if (!expandedFolders.has(item.id)) {
          expandedFolders.add(item.id);
          expandedFolders = expandedFolders;
        }
      } else {
        // Nested folder - toggle expansion
        toggleFolder(item.id);

        // Build breadcrumb path for nested folders
        // For nested folders like 'ict-scalper/nprd', we need the parent path
        const parentPath = getParentPath(item.id);

        // Navigate to update breadcrumbs with full path context
        navigationStore.navigateToFolder(item.id, item.name, parentPath || undefined);
      }
    } else if (item.type === 'action') {
      // Handle action items like "New Strategy"
      dispatch('action', { action: item.action });
    } else {
      // Emit event to open file in editor
      dispatch('openFile', {
        id: item.id,
        name: item.name,
        path: item.path,
        view: activeView
      });
    }
  }
  
  function getStatusBadge(status: string) {
    const badges: Record<string, string> = {
      primal: '🟢',
      ready: '🔵',
      pending: '🟡',
      processing: '⏳',
      quarantined: '🔴'
    };
    return badges[status] || '';
  }
</script>

<aside class="sidebar">
  <div class="sidebar-header">
    <div class="header-top">
      <span class="title">{viewConfig[activeView]?.title || 'Explorer'}</span>
      <button class="refresh-btn" on:click={() => loadData(activeView)} title="Refresh">
        <span class:spinning={loading}>
          <RefreshCw size={14} />
        </span>
      </button>
    </div>
    <!-- Breadcrumbs navigation -->
    {#if $navigationStore.breadcrumbs && $navigationStore.breadcrumbs.length > 0}
      <Breadcrumbs
        items={$navigationStore.breadcrumbs}
        onNavigate={(path) => navigationStore.navigateToPath(path)}
        showHome={false}
      />
    {/if}
  </div>
  
  {#if error}
    <div class="error-msg">{error}</div>
  {/if}
  
  <div class="tree-view">
    {#each treeData[activeView] || [] as folder}
      <div 
        class="tree-item folder"
        class:expanded={expandedFolders.has(folder.id)}
        on:click={() => handleItemClick(folder)}
        on:keypress={(e) => e.key === 'Enter' && handleItemClick(folder)}
        role="treeitem"
        tabindex="0"
      >
        {#if expandedFolders.has(folder.id)}
          <ChevronDown size={14} />
        {:else}
          <ChevronRight size={14} />
        {/if}
        
        {#if expandedFolders.has(folder.id)}
          <FolderOpen size={16} />
        {:else}
          <Folder size={16} />
        {/if}
        
        <span class="item-name">{folder.name}</span>
        
        {#if folder.status}
          <span class="status-badge">{getStatusBadge(folder.status)}</span>
        {/if}
      </div>
      
      {#if expandedFolders.has(folder.id) && folder.children}
        {#each folder.children as child}
          {#if child.type === 'folder'}
            <div 
              class="tree-item folder nested"
              class:expanded={expandedFolders.has(child.id)}
              on:click|stopPropagation={() => handleItemClick(child)}
              on:keypress={(e) => e.key === 'Enter' && handleItemClick(child)}
              role="treeitem"
              tabindex="0"
            >
              {#if expandedFolders.has(child.id)}
                <ChevronDown size={14} />
                <FolderOpen size={16} />
              {:else}
                <ChevronRight size={14} />
                <Folder size={16} />
              {/if}
              <span class="item-name">{child.name}</span>
            </div>
          {:else}
            <div 
              class="tree-item file"
              on:click|stopPropagation={() => handleItemClick(child)}
              on:keypress={(e) => e.key === 'Enter' && handleItemClick(child)}
              role="treeitem"
              tabindex="0"
            >
              <File size={16} />
              <span class="item-name">{child.name}</span>
            </div>
          {/if}
        {/each}
      {/if}
    {/each}
  </div>
</aside>

<style>
  .sidebar {
    grid-column: 2;
    grid-row: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
    overflow: hidden;
  }
  
  .sidebar-header {
    display: flex;
    flex-direction: column;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-subtle);
  }
  
  .header-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
  }
  
  .title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
  }
  
  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }
  
  .refresh-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  :global(.spinning) {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  
  .error-msg {
    padding: 8px 16px;
    font-size: 11px;
    color: var(--accent-warning);
    background: rgba(255, 200, 0, 0.1);
  }
  
  .tree-view {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }
  
  .tree-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 16px;
    cursor: pointer;
    color: var(--text-secondary);
    transition: background 0.1s ease;
    user-select: none;
  }
  
  .tree-item:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .tree-item:focus {
    outline: none;
    background: var(--bg-tertiary);
  }
  
  .tree-item.file {
    padding-left: 36px;
  }
  
  .tree-item.nested {
    padding-left: 36px;
  }
  
  .tree-item.nested + .tree-item.file {
    padding-left: 56px;
  }
  
  .item-name {
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
  }
  
  .status-badge {
    font-size: 10px;
    margin-left: auto;
  }
</style>
