<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import {
    Search,
    Grid,
    List,
    RefreshCw,
    Upload,
    Plus,
    Play,
    ArrowLeft,
    ChevronRight,
    Folder,
    Home,
  } from "lucide-svelte";

  interface Props {
    breadcrumbs?: Array<{ name: string; id?: string; type?: string }>;
    subPage?: string;
    searchQuery?: string;
    viewMode?: string;
    activeView?: string;
    viewConfig?: Record<string, any>;
  }

  let {
    breadcrumbs = [],
    subPage = "",
    searchQuery = $bindable(""),
    viewMode = "grid",
    activeView = "",
    viewConfig = {}
  }: Props = $props();

  const dispatch = createEventDispatcher();

  function navigateBack() {
    dispatch("navigateBack");
  }

  function navigateToBreadcrumb(index: number) {
    dispatch("navigateToBreadcrumb", { index });
  }

  function setViewMode(mode: string) {
    dispatch("setViewMode", { mode });
  }

  function handleRefresh() {
    dispatch("refresh");
  }

  function handleUpload() {
    dispatch("openUpload");
  }

  function handleVideoIngest() {
    dispatch("openVideoIngest");
  }

  function handleRunBacktest() {
    dispatch("openBacktest");
  }
</script>

<div class="view-header">
  <div class="breadcrumb-nav">
    {#if breadcrumbs.length > 1 || subPage}
      <button
        class="back-btn"
        onclick={navigateBack}
        title="Navigate back"
      >
        <ArrowLeft size={16} />
      </button>
    {/if}
    {#each breadcrumbs as crumb, i}
      {#if i > 0}<ChevronRight size={14} />{/if}
      {#if i === breadcrumbs.length - 1 && !subPage}
        <span class="breadcrumb-current">{crumb.name}</span>
      {:else}
        <button
          class="breadcrumb-item"
          onclick={() => navigateToBreadcrumb(i)}
          title="Navigate to {crumb.name}"
        >
          {#if crumb.type === "view"}
            {@const SvelteComponent = viewConfig[crumb.id]?.icon ? viewConfig[crumb.id].icon : Folder}
            <SvelteComponent
              size={16}
            />
          {:else}
            <Folder size={14} />
          {/if}
          <span>{crumb.name}</span>
        </button>
      {/if}
    {/each}
    {#if subPage}
      <ChevronRight size={14} />
      <span class="breadcrumb-current">{subPage}</span>
    {/if}
  </div>

  <div class="view-toolbar">
    {#if activeView === "database-view"}
    <div class="search-box">
      <Search size={14} />
      <input
        type="text"
        placeholder="Search..."
        bind:value={searchQuery}
      />
    </div>
    <div class="view-toggle">
      <button
        class:active={viewMode === "grid"}
        onclick={() => setViewMode("grid")}
      ><Grid size={14} /></button
      >
      <button
        class:active={viewMode === "list"}
        onclick={() => setViewMode("list")}
      ><List size={14} /></button
      >
    </div>
    {/if}
    <button class="toolbar-btn" onclick={handleRefresh}
      ><RefreshCw size={14} /></button
    >
    {#if activeView === "knowledge"}
      <button
        class="toolbar-btn primary"
        onclick={handleUpload}
      ><Upload size={14} /> Upload</button
      >
    {:else if activeView === "ea"}
      <button
        class="toolbar-btn primary"
        onclick={handleVideoIngest}
      >
        <Plus size={14} /> Video Ingest
      </button>
    {:else if activeView === "backtest"}
      <button
        class="toolbar-btn primary"
        onclick={handleRunBacktest}
      >
        <Play size={14} /> Run Backtest
      </button>
    {/if}
  </div>
</div>

<style>
  .view-header {
    margin-bottom: 20px;
  }

  .breadcrumb-nav {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 16px;
    padding: 8px 12px;
    background: var(--glass-content-bg);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    min-height: 40px;
    backdrop-filter: blur(var(--glass-blur));
  }

  .back-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-muted);
    cursor: pointer;
    margin-right: 8px;
    transition: all 0.2s ease;
  }

  .back-btn:hover {
    background: var(--glass-content-bg);
    color: var(--color-text-primary);
    border-color: var(--color-accent-cyan);
    box-shadow: 0 0 0 1px var(--color-accent-cyan);
  }

  .back-btn:active {
    transform: scale(0.95);
  }

  .breadcrumb-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: var(--color-bg-elevated);
    border: 1px solid transparent;
    border-radius: 4px;
    color: var(--color-text-secondary);
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s ease;
  }

  .breadcrumb-item:hover {
    background: var(--glass-content-bg);
    color: var(--color-accent-cyan);
    border-color: var(--color-border-subtle);
  }

  .breadcrumb-item:active {
    transform: scale(0.98);
  }

  .breadcrumb-current {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    color: var(--color-text-primary);
    font-weight: 500;
    font-size: 13px;
  }

  .breadcrumb-nav :global(svg) {
    flex-shrink: 0;
  }

  .view-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-muted);
  }

  .search-box input {
    background: transparent;
    border: none;
    color: var(--color-text-primary);
    font-size: 13px;
    outline: none;
    width: 180px;
  }

  .view-toggle {
    display: flex;
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    overflow: hidden;
  }

  .view-toggle button {
    padding: 6px 10px;
    background: var(--color-bg-surface);
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
  }

  .view-toggle button.active {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .toolbar-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-secondary);
    font-size: 12px;
    cursor: pointer;
  }

  .toolbar-btn:hover {
    background: var(--color-bg-elevated);
  }

  .toolbar-btn.primary {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }
</style>
