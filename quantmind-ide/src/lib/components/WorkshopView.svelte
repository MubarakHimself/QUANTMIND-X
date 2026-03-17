<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import {
    Bot, Send, Loader, RefreshCw, Database, Mail, Users, MessageCircle, Server, Sparkles, Workflow, FlaskConical, ChevronDown, ChevronUp
  } from "lucide-svelte";
  import { memoryStore } from "$lib/stores/memoryStore";
  import * as memoryApi from "$lib/api/memory";
  import { API_CONFIG } from "$lib/config/api";
  import TradingFloorCanvas from "$lib/components/trading-floor/TradingFloorCanvas.svelte";
  import TradingFloorPanel from "$lib/components/TradingFloorPanel.svelte";
  import CopilotPanel from "$lib/components/trading-floor/CopilotPanel.svelte";
  import DepartmentChatPanel from "$lib/components/trading-floor/DepartmentChatPanel.svelte";
  import DepartmentMailPanel from "$lib/components/trading-floor/DepartmentMailPanel.svelte";
  import MCPSettings from "$lib/components/agent-panel/settings/MCPSettings.svelte";
  import SkillsSettings from "$lib/components/agent-panel/settings/SkillsSettings.svelte";
  import VideoIngestWorkflow from "$lib/components/VideoIngestWorkflow.svelte";
  import WorkflowPanel from "$lib/components/WorkflowPanel.svelte";
  import WorkflowBuilder from "$lib/components/WorkflowBuilder.svelte";
  import EvaluationPanel from "$lib/components/EvaluationPanel.svelte";
  import {
    tradingFloorStore,
    updateAgentState,
    addSubAgent,
    sendMail,
    reset,
    type AgentState
  } from "$lib/stores/tradingFloorStore";
  import { connectTradingFloorWS, disconnectTradingFloorWS, wsConnected } from "$lib/services/tradingFloorWebSocket";

  // API base URL
  const API_BASE = API_CONFIG.API_BASE;

  // Simplified tabs
  let activeTab = $state("trading-floor");
  let tradingFloorCollapsed = $state(false);

  // Resizable panel widths
  let leftPanelWidth = $state(280);
  let rightPanelWidth = $state(320);
  let isDraggingLeft = $state(false);
  let isDraggingRight = $state(false);

  function startResizeLeft(e: MouseEvent) {
    isDraggingLeft = true;
    document.addEventListener('mousemove', handleResizeLeft);
    document.addEventListener('mouseup', stopResize);
  }

  function handleResizeLeft(e: MouseEvent) {
    if (!isDraggingLeft) return;
    const newWidth = Math.max(200, Math.min(450, e.clientX - 60));
    leftPanelWidth = newWidth;
  }

  function startResizeRight(e: MouseEvent) {
    isDraggingRight = true;
    document.addEventListener('mousemove', handleResizeRight);
    document.addEventListener('mouseup', stopResize);
  }

  function handleResizeRight(e: MouseEvent) {
    if (!isDraggingRight) return;
    const newWidth = Math.max(240, Math.min(500, window.innerWidth - e.clientX - 60));
    rightPanelWidth = newWidth;
  }

  function stopResize() {
    isDraggingLeft = false;
    isDraggingRight = false;
    document.removeEventListener('mousemove', handleResizeLeft);
    document.removeEventListener('mousemove', handleResizeRight);
    document.removeEventListener('mouseup', stopResize);
  }
  const tabs = [
    { id: "trading-floor", label: "Trading Floor", icon: Users },
    { id: "workflows", label: "Workflows", icon: Workflow },
    { id: "mail", label: "Mail", icon: Mail },
    { id: "memory", label: "Memory", icon: Database },
    { id: "skills", label: "Skills", icon: Sparkles },
    { id: "mcp", label: "MCP Servers", icon: Server },
    { id: "evaluation", label: "Evaluation", icon: FlaskConical },
    { id: "video-ingest", label: "Video Ingest", icon: Bot },
  ];

  // Memory state
  let memories = $derived($memoryStore.filteredMemories);
  let memoryStats = $derived($memoryStore.stats);
  let memoryLoading = $derived($memoryStore.loading);
  let memoryError = $derived($memoryStore.error);

  // Trading Floor state
  let floorStats = $derived($tradingFloorStore.floorStats || { totalTasks: 0, activeTasks: 0 });
  let isConnected = $derived($wsConnected || false);

  onMount(() => {
    loadMemoryData();
    initializeTradingFloor();
    connectTradingFloorWS();
  });

  onDestroy(() => {
    disconnectTradingFloorWS();
  });

  function initializeTradingFloor() {
    reset();
    const positions: Record<string, { x: number; y: number }> = {
      development: { x: 170, y: 130 },
      research: { x: 370, y: 130 },
      risk: { x: 570, y: 130 },
      trading: { x: 270, y: 300 },
      portfolio: { x: 470, y: 300 },
    };

    const departments = ['development', 'research', 'risk', 'trading', 'portfolio'];
    departments.forEach((dept) => {
      const agent: AgentState = {
        id: `${dept}-head`,
        name: `${dept.charAt(0).toUpperCase() + dept.slice(1)} Head`,
        department: dept,
        status: 'idle',
        position: positions[dept],
        target: null,
        subAgents: [],
        isExpanded: false,
      };
      addSubAgent('floor-manager', agent);
    });
  }

  function getPositionForDept(dept: string): { x: number; y: number } {
    const positions: Record<string, { x: number; y: number }> = {
      development: { x: 170, y: 130 },
      research: { x: 370, y: 130 },
      risk: { x: 570, y: 130 },
      trading: { x: 270, y: 300 },
      portfolio: { x: 470, y: 300 },
    };
    return positions[dept] || { x: 300, y: 200 };
  }

  async function loadMemoryData() {
    memoryStore.setLoading(true);
    try {
      const result = await memoryApi.listMemoriesForStore("default", 100);
      memoryStore.setMemories(result.memories);
      const stats = await memoryApi.getMemoryStatsForStore();
      memoryStore.setStats(stats);
    } catch (e) {
      memoryStore.setError(e instanceof Error ? e.message : "Failed to load memories");
    } finally {
      memoryStore.setLoading(false);
    }
  }
</script>

<div class="workshop-view">
  <!-- Minimal Icon Navigation (left sidebar) -->
  <div class="nav-sidebar">
    {#each tabs as tab}
      <button
        class="nav-icon-btn"
        class:active={activeTab === tab.id}
        onclick={() => activeTab = tab.id}
        title={tab.label}
      >
        <tab.icon size={18} />
      </button>
    {/each}
  </div>

  <!-- Tab Content -->
  <div class="tab-content" class:resizing={isDraggingLeft || isDraggingRight}>
    <!-- Trading Floor Tab - Unified View -->
    {#if activeTab === "trading-floor"}
    <div class="trading-floor-unified">
      <!-- Left Panel: Tabbed Copilot / Floor Manager -->
      <div class="tf-left-panel" style="width: {leftPanelWidth}px; flex: none;">
        <TradingFloorPanel />
      </div>

      <!-- Resize Handle Left -->
      <div class="resize-handle" onmousedown={startResizeLeft}></div>

      <!-- Center: Trading Floor Canvas -->
      <div class="tf-center-panel" class:collapsed={tradingFloorCollapsed}>
        <div class="panel-header">
          <Users size={16} />
          <span>Department Flow</span>
          <div class="tf-controls">
            <button class="tf-btn-sm tf-btn-outline" onclick={initializeTradingFloor}>
              Reset
            </button>
            <button class="tf-btn-sm tf-btn-outline" onclick={() => tradingFloorCollapsed = !tradingFloorCollapsed}>
              {#if tradingFloorCollapsed}
                <ChevronDown size={14} />
              {:else}
                <ChevronUp size={14} />
              {/if}
            </button>
          </div>
        </div>
        <div class="tf-stats">
          <div class="tf-stat">
            <span class="tf-stat-value">{floorStats.activeTasks}</span>
            <span class="tf-stat-label">Active</span>
          </div>
          <div class="tf-stat">
            <span class="tf-stat-value">{floorStats.totalTasks}</span>
            <span class="tf-stat-label">Total</span>
          </div>
          <div class="tf-stat">
            <span class="tf-stat-value ws-indicator" class:connected={isConnected}>
              {isConnected ? '●' : '○'}
            </span>
            <span class="tf-stat-label">WS</span>
          </div>
        </div>
        <div class="tf-canvas-container">
          <TradingFloorCanvas />
        </div>
      </div>

      <!-- Resize Handle Right -->
      <div class="resize-handle" onmousedown={startResizeRight}></div>

      <!-- Right Panel: Department Chat -->
      <div class="tf-right-panel" style="width: {rightPanelWidth}px; flex: none;">
        <div class="panel-header">
          <MessageCircle size={16} />
          <span>Department Chat</span>
        </div>
        <div class="panel-content">
          <DepartmentChatPanel />
        </div>
      </div>
    </div>
    {/if}

    <!-- Workflows Tab - Unified View -->
    {#if activeTab === "workflows"}
    <div class="workflows-unified">
      <div class="workflows-list-panel">
        <WorkflowPanel />
      </div>
      <div class="workflow-builder-panel">
        <WorkflowBuilder />
      </div>
    </div>
    {/if}

    <!-- Mail Tab -->
    {#if activeTab === "mail"}
    <div class="mail-panel-container">
      <DepartmentMailPanel />
    </div>
    {/if}

    <!-- Memory Tab -->
    {#if activeTab === "memory"}
    <div class="data-panel">
      <div class="panel-header-row">
        <h3>Memory Management</h3>
      </div>

      {#if memoryError}
        <div class="error-banner">{memoryError}</div>
      {/if}

      {#if memoryStats}
        <div class="stats-grid">
          <div class="stat-card">
            <span class="stat-value">{memoryStats.total_count || 0}</span>
            <span class="stat-label">Total Memories</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">{memoryStats.embedding_model || "N/A"}</span>
            <span class="stat-label">Embedding Model</span>
          </div>
        </div>
      {/if}

      <div class="data-list">
        {#if memoryLoading}
          <div class="loading-state">
            <RefreshCw size={24} class="spinning" />
            <span>Loading memories...</span>
          </div>
        {:else if memories.length === 0}
          <div class="empty-state">
            <Database size={32} />
            <p>No memories stored</p>
          </div>
        {:else}
          {#each memories as memory}
            <div class="data-item">
              <div class="item-header">
                <span class="item-key">{memory.key || memory.id}</span>
                <span class="item-namespace">{memory.namespace}</span>
              </div>
              <p class="item-content">{memory.content}</p>
              <span class="item-time">{memory.timestamp}</span>
            </div>
          {/each}
        {/if}
      </div>
    </div>
    {/if}

    <!-- Skills Tab -->
    {#if activeTab === "skills"}
    <div class="skills-view">
      <SkillsSettings />
    </div>
    {/if}

    <!-- MCP Servers Tab -->
    {#if activeTab === "mcp"}
    <div class="mcp-view">
      <MCPSettings />
    </div>
    {/if}

    <!-- Evaluation Tab -->
    {#if activeTab === "evaluation"}
    <div class="evaluation-view">
      <EvaluationPanel />
    </div>
    {/if}

    <!-- Video Ingest Tab -->
    {#if activeTab === "video-ingest"}
    <div class="video-ingest-view">
      <VideoIngestWorkflow />
    </div>
    {/if}
  </div>
</div>

<style>
  .workshop-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #0a0f1a);
    color: var(--text-primary, #e2e8f0);
  }

  /* Workshop takes full height */
  .workshop-view {
    height: 100%;
    display: flex;
  }

  /* Minimal Icon Navigation Sidebar */
  .nav-sidebar {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.5rem 0.375rem;
    background: var(--bg-secondary, #111827);
    border-right: 1px solid var(--border-color, #1e293b);
  }

  .nav-icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    cursor: pointer;
    transition: all 0.15s;
  }

  .nav-icon-btn:hover {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  .nav-icon-btn.active {
    background: var(--accent-primary, #3b82f6);
    color: white;
  }

  .tab-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .tab-content.resizing {
    cursor: col-resize;
    user-select: none;
  }

  /* Trading Floor Unified Layout */
  .trading-floor-unified {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .tf-left-panel {
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary, #111827);
    border-right: 1px solid var(--border-color, #1e293b);
    transition: background 0.2s;
  }

  /* Resize Handle */
  .resize-handle {
    width: 4px;
    background: transparent;
    cursor: col-resize;
    transition: background 0.2s;
    flex-shrink: 0;
  }

  .resize-handle:hover,
  .tab-content.resizing .resize-handle {
    background: var(--accent-primary, #3b82f6);
  }

  .tf-center-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg-primary, #0a0f1a);
    transition: flex 0.3s ease;
    min-width: 200px;
  }

  .tf-center-panel.collapsed {
    flex: 0 0 40px;
  }

  .tf-center-panel:not(.collapsed) {
    flex: 1;
    min-width: 300px;
  }

  .tf-right-panel {
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary, #111827);
    border-left: 1px solid var(--border-color, #1e293b);
    transition: background 0.2s;
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-tertiary, #1e293b);
    border-bottom: 1px solid var(--border-color, #334155);
    font-size: 0.6875rem;
    font-weight: 600;
    color: var(--text-secondary, #94a3b8);
  }

  .panel-content {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .tf-controls {
    margin-left: auto;
    display: flex;
    gap: 0.25rem;
  }

  .tf-btn-sm {
    padding: 0.25rem 0.5rem;
    font-size: 0.6875rem;
    border-radius: 0.25rem;
    border: none;
    background: var(--accent-primary, #3b82f6);
    color: white;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tf-btn-sm:hover:not(:disabled) {
    background: var(--accent-hover, #2563eb);
  }

  .tf-btn-sm:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .tf-btn-sm.tf-btn-secondary {
    background: var(--bg-input, #0f172a);
  }

  .tf-btn-sm.tf-btn-outline {
    background: transparent;
    border: 1px solid var(--border-color, #475569);
    color: var(--text-secondary, #94a3b8);
  }

  .tf-stats {
    display: flex;
    justify-content: center;
    gap: 1rem;
    padding: 0.25rem 0.5rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
  }

  .tf-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .tf-stat-label {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
  }

  .tf-stat-value {
    font-size: 0.8125rem;
    font-weight: 600;
  }

  .ws-indicator {
    color: #ef4444;
  }

  .ws-indicator.connected {
    color: #22c55e;
  }

  .tf-canvas-container {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    min-height: 200px;
  }

  /* Mail Panel */
  .mail-panel-container {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Memory Panel */
  .data-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 1rem;
  }

  .panel-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .panel-header-row h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-hover, #334155);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error-banner {
    padding: 0.5rem 0.75rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.375rem;
    color: #fca5a5;
    font-size: 0.75rem;
    margin-bottom: 1rem;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .stat-card {
    display: flex;
    flex-direction: column;
    padding: 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color, #334155);
  }

  .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--accent-primary, #3b82f6);
  }

  .stat-label {
    font-size: 0.6875rem;
    color: var(--text-secondary, #94a3b8);
    text-transform: uppercase;
  }

  .data-list {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .data-item {
    padding: 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
  }

  .item-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .item-key {
    font-weight: 500;
    font-size: 0.875rem;
  }

  .item-namespace {
    font-size: 0.6875rem;
    padding: 0.125rem 0.375rem;
    background: var(--bg-input, #0f172a);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
  }

  .item-content {
    margin: 0;
    font-size: 0.75rem;
    color: var(--text-secondary, #94a3b8);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .item-time {
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
  }

  .loading-state, .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    color: var(--text-muted, #64748b);
    gap: 0.75rem;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .mcp-view {
    height: 100%;
    overflow-y: auto;
    padding: 16px;
  }

  .evaluation-view {
    flex: 1;
    overflow-y: auto;
    background: var(--bg-primary, #0a0f1a);
  }

  .skills-view {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    background: var(--bg-primary, #0a0f1a);
  }

  .workflows-view {
    flex: 1;
    overflow: hidden;
    display: flex;
    background: var(--bg-primary, #0a0f1a);
  }

  /* Workflows Unified Split View */
  .workflows-unified {
    flex: 1;
    display: grid;
    grid-template-columns: minmax(280px, 35%) 1fr;
    gap: 0;
    overflow: hidden;
    background: var(--bg-primary, #0a0f1a);
  }

  .workflows-list-panel {
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary, #111827);
    border-right: 1px solid var(--border-color, #1e293b);
    overflow: hidden;
  }

  .workflow-builder-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg-primary, #0a0f1a);
  }

  .video-ingest-view {
    height: 100%;
    overflow-y: auto;
    padding: 16px;
  }

  .mcp-empty-state {
    text-align: center;
    color: var(--text-secondary, #888);
  }

  .mcp-empty-state h3 {
    margin: 16px 0 8px;
    color: var(--text-primary, #e0e0e0);
  }

  .btn-primary {
    background: var(--accent-color, #4a9eff);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 16px;
  }
</style>
