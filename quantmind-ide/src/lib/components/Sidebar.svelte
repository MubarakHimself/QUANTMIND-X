<script lang="ts">
  import { onMount } from "svelte";
  import { createEventDispatcher } from "svelte";
  import {
    ChevronRight,
    ChevronDown,
    File,
    Folder,
    FolderOpen,
    RefreshCw,
  } from "lucide-svelte";
  import { navigationStore } from "../stores/navigationStore";
  import Breadcrumbs from "./Breadcrumbs.svelte";

  export let activeView = "knowledge";

  const dispatch = createEventDispatcher();

  // API base URL
  const API_BASE = "http://localhost:8000/api";

  // Tree data state
  let treeData: Record<string, any[]> = {
    knowledge: [],
    assets: [],
    ea: [],
    backtest: [],
    live: [],
    settings: [],
  };

  let expandedFolders: Set<string> = new Set();
  let loading = false;
  let error = "";

  const viewConfig: Record<string, { title: string; endpoint: string }> = {
    knowledge: { title: "Knowledge Hub", endpoint: "/knowledge" },
    assets: { title: "Shared Assets", endpoint: "/assets" },
    ea: { title: "EA Management", endpoint: "/strategies" },
    backtest: { title: "Backtests", endpoint: "/strategies" },
    live: { title: "Live Trading", endpoint: "/trading/bots" },
    settings: { title: "Settings", endpoint: "" },
  };

  onMount(() => {
    loadData(activeView);
    // Initialize navigation with the current view
    navigationStore.navigateToView(
      activeView,
      viewConfig[activeView]?.title || "Explorer",
    );
  });

  $: if (activeView) {
    loadData(activeView);
    // Update navigation when view changes
    navigationStore.navigateToView(
      activeView,
      viewConfig[activeView]?.title || "Explorer",
    );
  }

  async function loadData(view: string) {
    const config = viewConfig[view];
    if (!config) {
      // Unknown views get empty state
      treeData[view] = [];
      treeData = treeData; // Trigger reactivity
      return;
    }

    // If no endpoint configured (like settings), use empty state
    if (!config.endpoint) {
      treeData[view] = [];
      treeData = treeData;
      return;
    }

    loading = true;
    error = "";

    try {
      const response = await fetch(`${API_BASE}${config.endpoint}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();

      // Transform data based on view type
      if (view === "knowledge") {
        treeData.knowledge = groupByCategory(data, "category");
      } else if (view === "assets") {
        treeData.assets = groupByCategory(data, "type");
      } else if (view === "ea" || view === "backtest") {
        treeData[view] = transformStrategies(data);
      } else if (view === "live") {
        // Transform bots data
        treeData.live = [
          {
            id: "active-bots",
            name: `Active Bots (${data.length})`,
            type: "folder",
            children: data.map((bot: any) => ({
              id: `bot/${bot.id}`,
              name: `${bot.name} @${bot.symbol}`,
              type: "bot",
              status: bot.state,
            })),
          },
        ];
      } else {
        treeData[view] = data;
      }
      treeData = treeData; // Trigger reactivity
    } catch (e: any) {
      error = e.message || "Failed to load data";
      // On error, show empty state
      treeData[view] = [];
      treeData = treeData; // Trigger reactivity
    } finally {
      loading = false;
    }
  }

  function groupByCategory(items: any[], key: string) {
    const groups: Record<string, any[]> = {};

    for (const item of items) {
      const category = item[key] || "Other";
      if (!groups[category]) groups[category] = [];
      groups[category].push(item);
    }

    return Object.entries(groups).map(([name, children]) => ({
      id: name,
      name: name.charAt(0).toUpperCase() + name.slice(1),
      type: "folder",
      children: children.map((c) => ({
        id: c.id,
        name: c.name,
        type: "file",
        path: c.path,
      })),
    }));
  }

  function transformStrategies(strategies: any[]) {
    return strategies.map((s) => ({
      id: s.id,
      name: s.name,
      type: "folder",
      status: s.status,
      children: [
        ...(s.has_video_ingest
          ? [
              {
                id: `${s.id}/video_ingest`,
                name: "Video Ingest Output",
                type: "folder",
                children: [],
              },
            ]
          : []),
        ...(s.has_trd
          ? [{ id: `${s.id}/trd`, name: "TRD", type: "folder", children: [] }]
          : []),
        ...(s.has_ea
          ? [
              {
                id: `${s.id}/ea`,
                name: "EA Code",
                type: "folder",
                children: [],
              },
            ]
          : []),
        ...(s.has_backtest
          ? [
              {
                id: `${s.id}/backtest`,
                name: "Backtest Reports",
                type: "folder",
                children: [],
              },
            ]
          : []),
      ],
    }));
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
    const parts = folderId.split("/");
    parts.pop(); // Remove the last element (current folder)
    return parts.join("/");
  }

  function handleItemClick(item: any) {
    if (item.type === "folder") {
      // Check if this is a top-level strategy folder (in EA view)
      // Top-level strategy IDs don't contain '/' (e.g., 'ict-scalper', 'smc-reversal')
      // Nested folders have '/' (e.g., 'ict-scalper/video_ingest', 'ict-scalper/trd')
      const isTopLevelStrategy = activeView === "ea" && !item.id.includes("/");

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
        // For nested folders like 'ict-scalper/video_ingest', we need the parent path
        const parentPath = getParentPath(item.id);

        // Navigate to update breadcrumbs with full path context
        navigationStore.navigateToFolder(
          item.id,
          item.name,
          parentPath || undefined,
        );
      }
    } else if (item.type === "action") {
      // Handle action items like "New Strategy"
      dispatch("action", { action: item.action });
    } else {
      // Emit event to open file in editor
      dispatch("openFile", {
        id: item.id,
        name: item.name,
        path: item.path,
        view: activeView,
      });
    }
  }

  function getStatusBadge(status: string) {
    const badges: Record<string, string> = {
      primal: "🟢",
      ready: "🔵",
      pending: "🟡",
      processing: "⏳",
      quarantined: "🔴",
    };
    return badges[status] || "";
  }
</script>

<aside class="sidebar">
  <div class="sidebar-header">
    <div class="header-top">
      <span class="title">{viewConfig[activeView]?.title || "Explorer"}</span>
      <button
        class="refresh-btn"
        on:click={() => loadData(activeView)}
        title="Refresh"
      >
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
    {#if !loading && (!treeData[activeView] || treeData[activeView].length === 0)}
      <div class="empty-state">
        <span class="empty-text">No items to display</span>
        {#if error}
          <button class="retry-btn" on:click={() => loadData(activeView)}>
            Retry
          </button>
        {/if}
      </div>
    {:else}
    {#each treeData[activeView] || [] as folder}
      <div
        class="tree-item folder"
        class:expanded={expandedFolders.has(folder.id)}
        on:click={() => handleItemClick(folder)}
        on:keypress={(e) => e.key === "Enter" && handleItemClick(folder)}
        role="treeitem"
        aria-selected={expandedFolders.has(folder.id)}
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
          {#if child.type === "folder"}
            <div
              class="tree-item folder nested"
              class:expanded={expandedFolders.has(child.id)}
              on:click|stopPropagation={() => handleItemClick(child)}
              on:keypress={(e) => e.key === "Enter" && handleItemClick(child)}
              role="treeitem"
              aria-selected={expandedFolders.has(child.id)}
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
              on:keypress={(e) => e.key === "Enter" && handleItemClick(child)}
              role="treeitem"
              aria-selected="false"
              tabindex="0"
            >
              <File size={16} />
              <span class="item-name">{child.name}</span>
            </div>
          {/if}
        {/each}
      {/if}
    {/each}
    {/if}
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
    width: var(--panel-width);
  }

  .sidebar-header {
    display: flex;
    flex-direction: column;
    height: var(--header-height);
    padding: 0 12px;
    border-bottom: 1px solid var(--border-subtle);
    justify-content: center;
    background: var(--bg-secondary);
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
    color: var(--text-muted);
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
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
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .error-msg {
    padding: 8px 12px;
    font-size: 11px;
    color: var(--accent-danger);
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px 16px;
    gap: 12px;
  }

  .empty-text {
    font-size: 12px;
    color: var(--text-muted);
    text-align: center;
  }

  .retry-btn {
    padding: 6px 12px;
    font-size: 11px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .retry-btn:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
  }

  .tree-view {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
  }

  .tree-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    cursor: pointer;
    color: var(--text-secondary);
    transition: background 0.05s ease;
    user-select: none;
    font-size: 13px;
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
    padding-left: 32px;
  }

  .tree-item.nested {
    padding-left: 32px;
  }

  .item-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
  }

  .status-badge {
    font-size: 10px;
    margin-left: auto;
    opacity: 0.8;
  }

  /* Icon adjustments */
  :global(.tree-item .lucide) {
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .tree-item:hover :global(.lucide) {
    color: var(--text-primary);
  }

  .tree-item.folder :global(.lucide) {
    color: var(--accent-primary);
  }
</style>
