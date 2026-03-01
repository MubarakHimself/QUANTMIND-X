<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import {
    Bot, Send, Loader, RefreshCw, Wrench, Database, Mail, Users, MessageCircle, Server
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
  import NPRDWorkflow from "$lib/components/NPRDWorkflow.svelte";
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
  let activeTab = "trading-floor";
  const tabs = [
    { id: "trading-floor", label: "Trading Floor", icon: Users },
    { id: "mail", label: "Mail", icon: Mail },
    { id: "memory", label: "Memory", icon: Database },
    { id: "mcp", label: "MCP Servers", icon: Server },
    { id: "nprd", label: "NPRD", icon: Bot },
  ];

  // Memory state
  $: memories = $memoryStore.filteredMemories;
  $: memoryStats = $memoryStore.stats;
  $: memoryLoading = $memoryStore.loading;
  $: memoryError = $memoryStore.error;

  // Trading Floor state
  let isRunning = false;
  let demoInterval: number | null = null;
  const departments = ['analysis', 'research', 'risk', 'execution', 'portfolio'];
  const sampleTasks = [
    { from: 'analysis', to: 'execution', type: 'dispatch', subject: 'EURUSD Signal: BUY' },
    { from: 'research', to: 'analysis', type: 'result', subject: 'Backtest Complete' },
    { from: 'risk', to: 'execution', type: 'question', subject: 'Position Size Query' },
    { from: 'analysis', to: 'portfolio', type: 'dispatch', subject: 'Market Update' },
    { from: 'execution', to: 'risk', type: 'status', subject: 'Order Filled' },
  ];
  $: floorStats = $tradingFloorStore.floorStats || { totalTasks: 0, activeTasks: 0 };
  $: isConnected = $wsConnected || false;

  onMount(() => {
    loadMemoryData();
    initializeTradingFloor();
    connectTradingFloorWS();
  });

  onDestroy(() => {
    stopDemo();
    disconnectTradingFloorWS();
  });

  function initializeTradingFloor() {
    reset();
    const positions: Record<string, { x: number; y: number }> = {
      analysis: { x: 170, y: 130 },
      research: { x: 370, y: 130 },
      risk: { x: 570, y: 130 },
      execution: { x: 270, y: 300 },
      portfolio: { x: 470, y: 300 },
    };

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

  function startDemo() {
    if (isRunning) return;
    isRunning = true;
    demoInterval = window.setInterval(() => runDemoStep(), 2000);
  }

  function stopDemo() {
    isRunning = false;
    if (demoInterval) {
      clearInterval(demoInterval);
      demoInterval = null;
    }
  }

  let stepCount = 0;
  function runDemoStep() {
    stepCount++;
    const agentStates: AgentState['status'][] = ['thinking', 'typing', 'reading', 'idle'];
    const randomDept = departments[stepCount % departments.length];
    const randomState = agentStates[stepCount % agentStates.length];

    updateAgentState(`${randomDept}-head`, {
      status: randomState,
      speechBubble: randomState === 'thinking'
        ? { text: 'Analyzing...', type: 'thinking', duration: 2000 }
        : undefined,
    });

    if (stepCount % 3 === 0) {
      const task = sampleTasks[stepCount % sampleTasks.length];
      sendMail({
        id: `mail-${Date.now()}`,
        fromDept: task.from,
        toDept: task.to,
        type: task.type as 'dispatch' | 'result' | 'question' | 'status',
        subject: task.subject,
        startX: 0,
        startY: 0,
        progress: 0,
        duration: 1500,
      });
    }

    if (stepCount % 5 === 0) {
      const dept = departments[stepCount % departments.length];
      const workerTypes: Record<string, string> = {
        analysis: 'market_analyst',
        research: 'backtester',
        risk: 'position_sizer',
        execution: 'order_router',
        portfolio: 'rebalancer',
      };
      const parentPos = getPositionForDept(dept);
      const subAgent: AgentState = {
        id: `${dept}-worker-${stepCount}`,
        name: workerTypes[dept] || 'worker',
        department: dept,
        status: 'thinking',
        position: {
          x: parentPos.x + (Math.random() * 60 - 30),
          y: parentPos.y + (Math.random() * 60 - 30),
        },
        target: null,
        subAgents: [],
        isExpanded: false,
      };
      addSubAgent(`${dept}-head`, subAgent);
    }
  }

  function getPositionForDept(dept: string): { x: number; y: number } {
    const positions: Record<string, { x: number; y: number }> = {
      analysis: { x: 170, y: 130 },
      research: { x: 370, y: 130 },
      risk: { x: 570, y: 130 },
      execution: { x: 270, y: 300 },
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
  <!-- Header with tabs -->
  <div class="workshop-header">
    <div class="header-title">
      <Wrench size={20} />
      <h2>QuantMind Workshop</h2>
    </div>
    <div class="tabs">
      {#each tabs as tab}
        <button
          class="tab-btn"
          class:active={activeTab === tab.id}
          on:click={() => activeTab = tab.id}
        >
          <svelte:component this={tab.icon} size={14} />
          {tab.label}
        </button>
      {/each}
    </div>
  </div>

  <!-- Tab Content -->
  <div class="tab-content">
    <!-- Trading Floor Tab - Unified View -->
    {#if activeTab === "trading-floor"}
    <div class="trading-floor-unified">
      <!-- Left Panel: Tabbed Copilot / Floor Manager -->
      <div class="tf-left-panel">
        <TradingFloorPanel />
      </div>

      <!-- Center: Trading Floor Canvas -->
      <div class="tf-center-panel">
        <div class="panel-header">
          <Users size={16} />
          <span>Department Flow</span>
          <div class="tf-controls">
            <button class="tf-btn-sm" on:click={startDemo} disabled={isRunning}>
              {isRunning ? 'Running...' : 'Demo'}
            </button>
            <button class="tf-btn-sm tf-btn-secondary" on:click={stopDemo} disabled={!isRunning}>
              Stop
            </button>
            <button class="tf-btn-sm tf-btn-outline" on:click={initializeTradingFloor}>
              Reset
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

      <!-- Right Panel: Department Chat -->
      <div class="tf-right-panel">
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
        <button class="refresh-btn" on:click={loadMemoryData} disabled={memoryLoading}>
          <RefreshCw size={14} class={memoryLoading ? "spinning" : ""} />
          Refresh
        </button>
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

    <!-- MCP Servers Tab -->
    {#if activeTab === "mcp"}
    <div class="mcp-view">
      <MCPSettings />
    </div>
    {/if}

    <!-- NPRD Tab -->
    {#if activeTab === "nprd"}
    <div class="nprd-view">
      <NPRDWorkflow />
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

  .workshop-header {
    display: flex;
    align-items: center;
    padding: 0.5rem 1rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
    gap: 1rem;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--accent-primary, #3b82f6);
  }

  .header-title h2 {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 600;
  }

  .tabs {
    display: flex;
    gap: 0.25rem;
    margin-left: 1rem;
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: transparent;
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tab-btn:hover {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  .tab-btn.active {
    background: var(--accent-primary, #3b82f6);
    border-color: var(--accent-primary, #3b82f6);
    color: white;
  }

  .tab-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Trading Floor Unified Layout */
  .trading-floor-unified {
    flex: 1;
    display: grid;
    grid-template-columns: 280px 1fr 320px;
    gap: 0;
    overflow: hidden;
  }

  .tf-left-panel,
  .tf-right-panel {
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary, #111827);
    border-right: 1px solid var(--border-color, #1e293b);
  }

  .tf-right-panel {
    border-right: none;
    border-left: 1px solid var(--border-color, #1e293b);
  }

  .tf-center-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg-primary, #0a0f1a);
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border-bottom: 1px solid var(--border-color, #334155);
    font-size: 0.75rem;
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
    gap: 1.5rem;
    padding: 0.5rem;
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
    font-size: 1rem;
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
    min-height: 300px;
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

  .nprd-view {
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
