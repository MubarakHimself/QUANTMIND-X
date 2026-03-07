<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { WebSocketClient, createBacktestClient } from "../ws-client";
  import type { WebSocketMessage } from "../ws-client";
  import {
    Play,
    Square,
    FileText,
    X,
    Plus,
    ArrowLeft,
  } from "lucide-svelte";
  import { navigationStore } from "../stores/navigationStore";

  // Import extracted sub-components
  import {
    PaperTradingAgentCard,
    PaperTradingDeployModal,
    PaperTradingLogsModal,
    PaperTradingPromoteModal,
  } from "./paper-trading";

  // Props
  export let baseUrl: string = "http://localhost:8000";

  // Performance metrics interface
  interface PerformanceMetrics {
    agent_id: string;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_pnl: number;
    average_pnl: number;
    max_drawdown: number;
    profit_factor: number;
    sharpe_ratio: number | null;
    validation_status: string;
    days_validated: number;
    meets_criteria: boolean;
    validation_thresholds?: {
      min_sharpe_ratio: number;
      min_win_rate: number;
      min_validation_days: number;
    };
  }

  // State
  let agents: Array<{
    agent_id: string;
    container_id: string;
    container_name: string;
    status: string;
    strategy_name: string;
    symbol?: string;
    timeframe?: string;
    mt5_account?: number;
    mt5_server?: string;
    magic_number?: number;
    uptime_seconds?: number;
    created_at: string;
  }> = [];

  // Performance metrics per agent
  let performanceMetrics: Record<string, PerformanceMetrics> = {};

  let selectedAgent: (typeof agents)[0] | null = null;
  let agentLogs: string[] = [];
  let isLoading = false;
  let error: string | null = null;
  let wsClient: WebSocketClient | null = null;
  let tickData: Record<
    string,
    { bid: number; ask: number; spread: number; timestamp: string }
  > = {};
  let showDeployForm = false;
  let showPromoteModal = false;
  let promoteAgentId: string | null = null;
  let promoteLoading = false;
  let promoteResult: {
    success: boolean;
    bot_id?: string;
    error?: string;
  } | null = null;

  // Deploy form state
  let deployForm = {
    strategy_name: "",
    strategy_code: "",
    symbol: "EURUSD",
    timeframe: "H1",
    mt5_account: "",
    mt5_password: "",
    mt5_server: "MetaQuotes-Demo",
    magic_number: Math.floor(Math.random() * 100000000),
  };

  // Promote form state
  let promoteForm = {
    target_account: "account_b_sniper",
    strategy_name: "",
    strategy_type: "STRUCTURAL",
  };

  // Options
  const symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"];
  const timeframes = ["M1", "M5", "H1", "H4", "D1"];
  const accountOptions = [
    { value: "account_a_machine_gun", label: "Machine Gun (HFT/Scalpers)" },
    { value: "account_b_sniper", label: "Sniper (Structural/ICT)" },
    { value: "account_c_prop", label: "Prop Firm Safe" },
  ];
  const strategyTypes = ["SCALPER", "STRUCTURAL", "SWING", "HFT"];

  // Lifecycle
  onMount(async () => {
    await loadAgents();
    await connectWebSocket();
  });

  onDestroy(() => {
    disconnectWebSocket();
  });

  // WebSocket connection
  async function connectWebSocket() {
    try {
      wsClient = await createBacktestClient(baseUrl);

      // Subscribe to tick data
      wsClient.subscribe("tick_data");
      wsClient.on("tick_data", handleTickData);

      // Subscribe to paper trading updates (backend uses 'paper-trading' topic)
      wsClient.subscribe("paper-trading");
      wsClient.on("paper_trading_update", handlePaperTradingUpdate);

      // Performance updates come on same topic, different message type
      wsClient.on("paper_trading_performance_update", handlePerformanceUpdate);
    } catch (e) {
      console.error("Failed to connect WebSocket:", e);
      error = `Failed to connect to WebSocket: ${e}`;
    }
  }

  function disconnectWebSocket() {
    if (wsClient) {
      wsClient.off("tick_data", handleTickData);
      wsClient.off("paper_trading_update", handlePaperTradingUpdate);
      wsClient.off("paper_trading_performance_update", handlePerformanceUpdate);
      wsClient.disconnect();
      wsClient = null;
    }
  }

  function handleTickData(message: WebSocketMessage) {
    const data = message.data as {
      symbol: string;
      bid: number;
      ask: number;
      spread: number;
      timestamp: string;
    };
    if (data && data.symbol) {
      tickData[data.symbol] = {
        bid: data.bid,
        ask: data.ask,
        spread: data.spread,
        timestamp: data.timestamp,
      };
    }
  }

  function handlePaperTradingUpdate(message: WebSocketMessage) {
    const data = message.data as {
      agent_id: string;
      status: string;
      uptime_seconds?: number;
    };
    if (data && data.agent_id) {
      const index = agents.findIndex((a) => a.agent_id === data.agent_id);
      if (index !== -1) {
        agents[index] = { ...agents[index], ...data };
      }
    }
  }

  function handlePerformanceUpdate(message: WebSocketMessage) {
    const data = message.data as PerformanceMetrics;
    if (data && data.agent_id) {
      performanceMetrics[data.agent_id] = data;
    }
  }

  // API calls
  async function loadAgents() {
    isLoading = true;
    error = null;
    try {
      const response = await fetch(`${baseUrl}/api/paper-trading/agents`);
      if (!response.ok) {
        throw new Error(`Failed to load agents: ${response.statusText}`);
      }
      agents = await response.json();

      // Load performance metrics for each agent
      for (const agent of agents) {
        await loadPerformanceMetrics(agent.agent_id);
      }
    } catch (e) {
      console.error("Failed to load agents:", e);
      error = `Failed to load agents: ${e}`;
    } finally {
      isLoading = false;
    }
  }

  async function loadPerformanceMetrics(agentId: string) {
    try {
      const response = await fetch(
        `${baseUrl}/api/paper-trading/agents/${agentId}/performance`,
      );
      if (response.ok) {
        const metrics = await response.json();
        performanceMetrics[agentId] = metrics;
      }
    } catch (e) {
      console.error(`Failed to load performance for ${agentId}:`, e);
    }
  }

  async function deployAgent() {
    // Validate form
    if (
      !deployForm.strategy_name ||
      !deployForm.strategy_code ||
      !deployForm.mt5_account ||
      !deployForm.mt5_password
    ) {
      error = "Please fill in all required fields";
      return;
    }

    isLoading = true;
    error = null;
    try {
      const response = await fetch(`${baseUrl}/api/paper-trading/deploy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          strategy_name: deployForm.strategy_name,
          strategy_code: deployForm.strategy_code,
          config: {
            symbol: deployForm.symbol,
            timeframe: deployForm.timeframe,
          },
          mt5_credentials: {
            account: parseInt(deployForm.mt5_account),
            password: deployForm.mt5_password,
            server: deployForm.mt5_server,
          },
          magic_number: deployForm.magic_number,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Deployment failed");
      }

      // Reload agents and subscribe to tick data
      await loadAgents();
      showDeployForm = false;

      // Subscribe to tick data for the symbol
      if (wsClient) {
        await fetch(
          `${baseUrl}/api/paper-trading/tick-data/subscribe?symbol=${deployForm.symbol}`,
          {
            method: "POST",
          },
        );
      }

      // Reset form
      deployForm = {
        strategy_name: "",
        strategy_code: "",
        symbol: "EURUSD",
        timeframe: "H1",
        mt5_account: "",
        mt5_password: "",
        mt5_server: "MetaQuotes-Demo",
        magic_number: Math.floor(Math.random() * 100000000),
      };
    } catch (e) {
      console.error("Failed to deploy agent:", e);
      error = `Failed to deploy agent: ${e}`;
    } finally {
      isLoading = false;
    }
  }

  async function stopAgent(agentId: string) {
    if (!confirm("Are you sure you want to stop this agent?")) {
      return;
    }

    isLoading = true;
    error = null;
    try {
      const response = await fetch(
        `${baseUrl}/api/paper-trading/agents/${agentId}/stop`,
        {
          method: "POST",
        },
      );

      if (!response.ok) {
        throw new Error(`Failed to stop agent: ${response.statusText}`);
      }

      await loadAgents();
    } catch (e) {
      console.error("Failed to stop agent:", e);
      error = `Failed to stop agent: ${e}`;
    } finally {
      isLoading = false;
    }
  }

  async function viewLogs(agentId: string) {
    const agent = agents.find((a) => a.agent_id === agentId);
    if (!agent) return;

    selectedAgent = agent;
    isLoading = true;
    error = null;
    try {
      const response = await fetch(
        `${baseUrl}/api/paper-trading/agents/${agentId}/logs?tail_lines=100`,
      );
      if (!response.ok) {
        throw new Error(`Failed to load logs: ${response.statusText}`);
      }
      const result = await response.json();
      agentLogs = result.logs || [];
    } catch (e) {
      console.error("Failed to load logs:", e);
      error = `Failed to load logs: ${e}`;
    } finally {
      isLoading = false;
    }
  }

  function openPromoteModal(agentId: string) {
    promoteAgentId = agentId;
    const agent = agents.find((a) => a.agent_id === agentId);
    promoteForm.strategy_name = agent?.strategy_name || "";
    promoteResult = null;
    showPromoteModal = true;
  }

  async function promoteAgent() {
    if (!promoteAgentId) return;

    promoteLoading = true;
    promoteResult = null;
    try {
      const response = await fetch(
        `${baseUrl}/api/paper-trading/agents/${promoteAgentId}/promote`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(promoteForm),
        },
      );

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Promotion failed");
      }

      promoteResult = { success: true, bot_id: result.bot_id };

      // Refresh agents after successful promotion
      await loadAgents();
    } catch (e) {
      console.error("Failed to promote agent:", e);
      promoteResult = { success: false, error: String(e) };
    } finally {
      promoteLoading = false;
    }
  }

  function closeLogs() {
    selectedAgent = null;
    agentLogs = [];
  }

  function formatUptime(seconds?: number): string {
    if (!seconds) return "-";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case "running":
        return "text-green-500";
      case "stopped":
        return "text-gray-500";
      case "error":
        return "text-red-500";
      case "starting":
        return "text-yellow-500";
      default:
        return "text-gray-400";
    }
  }

  function getStatusBadgeColor(status: string): string {
    switch (status) {
      case "running":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "stopped":
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
      case "error":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      case "starting":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  }

  function getValidationBadgeColor(status: string): string {
    switch (status) {
      case "validated":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "validating":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "pending":
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
      case "failed":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  }

  function getSharpeColor(sharpe: number | null): string {
    if (sharpe === null) return "text-gray-400";
    if (sharpe >= 1.5) return "text-green-400";
    if (sharpe >= 1.0) return "text-yellow-400";
    return "text-red-400";
  }

  function getWinRateColor(winRate: number): string {
    if (winRate >= 0.55) return "text-green-400";
    if (winRate >= 0.45) return "text-yellow-400";
    return "text-red-400";
  }

  function canPromote(metrics: PerformanceMetrics | undefined): boolean {
    if (!metrics) return false;
    return (
      metrics.validation_status === "validated" &&
      metrics.meets_criteria &&
      metrics.days_validated >= 30
    );
  }
</script>

<div class="paper-trading-panel">
  <!-- Header with back navigation -->
  <div class="panel-header">
    <div class="header-left">
      <button
        class="back-button"
        on:click={() => navigationStore.navigateToView("live", "Live Trading")}
        title="Back to Live Trading"
      >
        <ArrowLeft size={20} />
      </button>
      <h2 class="panel-title">Paper Trading</h2>
    </div>
    <button
      class="btn btn-primary"
      on:click={() => (showDeployForm = true)}
      disabled={isLoading}
    >
      <Plus size={18} />
      Deploy New Agent
    </button>
  </div>

  <!-- Error Banner -->
  {#if error}
    <div class="error-banner">
      <span class="error-icon">⚠️</span>
      <span class="error-text">{error}</span>
      <button class="error-close" on:click={() => (error = null)}>
        <X size={16} />
      </button>
    </div>
  {/if}

  <!-- Agents Grid -->
  <div class="agents-container">
    {#if isLoading && agents.length === 0}
      <div class="loading-state">Loading agents...</div>
    {:else if agents.length === 0}
      <div class="empty-state">
        <div class="empty-icon">📊</div>
        <h3>No Paper Trading Agents</h3>
        <p>Deploy your first paper trading agent to get started.</p>
        <button
          class="btn btn-primary"
          on:click={() => (showDeployForm = true)}
        >
          <Plus size={18} />
          Deploy Agent
        </button>
      </div>
    {:else}
      <div class="agents-grid">
        {#each agents as agent (agent.agent_id)}
          <PaperTradingAgentCard
            {agent}
            performanceMetrics={performanceMetrics[agent.agent_id]}
            tickData={agent.symbol ? tickData[agent.symbol] : undefined}
            on:stop={(e) => stopAgent(e.detail)}
            on:logs={(e) => viewLogs(e.detail)}
            on:promote={(e) => openPromoteModal(e.detail)}
          />
        {/each}
      </div>
    {/if}
  </div>

  <!-- Deploy Form Modal -->
  <PaperTradingDeployModal
    isOpen={showDeployForm}
    {form}
    {isLoading}
    on:close={() => (showDeployForm = false)}
    on:deploy={(e) => {
      deployForm = e.detail;
      deployAgent();
    }}
  />

  <!-- Logs Modal -->
  <PaperTradingLogsModal
    isOpen={!!selectedAgent}
    logs={agentLogs}
    isLoading={isLoading}
    on:close={closeLogs}
  />

  <!-- Promotion Modal -->
  <!-- Promotion Modal -->
  <PaperTradingPromoteModal
    isOpen={showPromoteModal && !!promoteAgentId}
    form={promoteForm}
    isLoading={promoteLoading}
    {result}
    on:close={() => (showPromoteModal = false)}
    on:promote={(e) => {
      promoteForm = e.detail;
      promoteAgent();
    }}
  />
</div>

<style>
  .paper-trading-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #1e1e1e;
    color: #e0e0e0;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #2d2d2d;
  }

  .panel-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e0e0e0;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-primary {
    background: #10b981;
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    background: #059669;
  }

  .btn-secondary {
    background: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
  }

  .btn-secondary:hover:not(:disabled) {
    background: #3d3d3d;
  }

  .btn-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 0.375rem;
    background: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    cursor: pointer;
    transition: all 0.2s;
    padding: 0;
  }

  .btn-icon:hover {
    background: #3d3d3d;
  }

  .btn-danger {
    color: #ef4444;
  }

  .btn-danger:hover {
    background: #3d2d2d;
    border-color: #ef4444;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.375rem;
    margin: 0 1.5rem 1rem;
  }

  .error-text {
    flex: 1;
    color: #ef4444;
    font-size: 0.875rem;
  }

  .error-close {
    background: transparent;
    border: none;
    color: #ef4444;
    cursor: pointer;
    padding: 0;
  }

  .agents-container {
    flex: 1;
    overflow-y: auto;
    padding: 0 1.5rem 1.5rem;
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
    color: #9ca3af;
  }

  .empty-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .empty-state h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
  }

  .empty-state p {
    color: #9ca3af;
    margin-bottom: 1.5rem;
  }

  .agents-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
  }

  .agent-card {
    background: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 0.5rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .agent-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
  }

  .agent-name {
    font-weight: 600;
    color: #e0e0e0;
    font-size: 1rem;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .agent-status-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    border: 1px solid;
  }

  .agent-details {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
  }

  .detail-label {
    color: #9ca3af;
  }

  .detail-value {
    color: #e0e0e0;
    font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tick-data {
    background: #1e1e1e;
    border-radius: 0.375rem;
    padding: 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .tick-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
  }

  .tick-label {
    color: #9ca3af;
  }

  .tick-value {
    color: #e0e0e0;
    font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
  }

  .agent-actions {
    display: flex;
    gap: 0.5rem;
    margin-top: auto;
  }

  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 0.5rem;
    width: 100%;
    max-width: 500px;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
  }

  .modal-large {
    max-width: 800px;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #3d3d3d;
  }

  .modal-header h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #e0e0e0;
  }

  .modal-body {
    padding: 1.5rem;
    overflow-y: auto;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
    padding: 1rem 1.5rem;
    border-top: 1px solid #3d3d3d;
  }

  .form-group {
    margin-bottom: 1rem;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .form-label {
    display: block;
    font-size: 0.875rem;
    font-weight: 500;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
  }

  .form-input {
    width: 100%;
    padding: 0.5rem 0.75rem;
    background: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 0.375rem;
    color: #e0e0e0;
    font-size: 0.875rem;
  }

  .form-input:focus {
    outline: none;
    border-color: #10b981;
  }

  .form-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .form-textarea {
    width: 100%;
    padding: 0.5rem 0.75rem;
    background: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 0.375rem;
    color: #e0e0e0;
    font-size: 0.875rem;
    font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
    resize: vertical;
  }

  .form-textarea:focus {
    outline: none;
    border-color: #10b981;
  }

  .logs-container {
    background: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 0.375rem;
    padding: 1rem;
    max-height: 400px;
    overflow-y: auto;
  }

  .log-line {
    font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
    font-size: 0.875rem;
    color: #e0e0e0;
    white-space: pre-wrap;
    word-break: break-all;
    padding: 0.25rem 0;
  }

  .text-green-500 {
    color: #10b981;
  }

  .text-gray-500 {
    color: #6b7280;
  }

  .text-red-500 {
    color: #ef4444;
  }

  .text-yellow-500 {
    color: #f59e0b;
  }

  .text-gray-400 {
    color: #9ca3af;
  }

  .bg-green-500\/20 {
    background-color: rgba(16, 185, 129, 0.2);
  }

  .bg-gray-500\/20 {
    background-color: rgba(107, 114, 128, 0.2);
  }

  .bg-red-500\/20 {
    background-color: rgba(239, 68, 68, 0.2);
  }

  .bg-yellow-500\/20 {
    background-color: rgba(245, 158, 11, 0.2);
  }

  .border-green-500\/30 {
    border-color: rgba(16, 185, 129, 0.3);
  }

  .border-gray-500\/30 {
    border-color: rgba(107, 114, 128, 0.3);
  }

  .border-red-500\/30 {
    border-color: rgba(239, 68, 68, 0.3);
  }

  .border-yellow-500\/30 {
    border-color: rgba(245, 158, 11, 0.3);
  }

  /* Performance Metrics Styles */
  .performance-section {
    background: #1e1e1e;
    border-radius: 0.375rem;
    padding: 0.75rem;
    margin-top: 0.5rem;
  }

  .performance-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    font-weight: 500;
    color: #9ca3af;
  }

  .validation-badge {
    margin-left: auto;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-size: 0.625rem;
    font-weight: 600;
    text-transform: uppercase;
    border: 1px solid;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
  }

  .metric-item {
    display: flex;
    flex-direction: column;
    padding: 0.375rem;
    background: #2d2d2d;
    border-radius: 0.25rem;
  }

  .metric-label {
    font-size: 0.625rem;
    color: #9ca3af;
    text-transform: uppercase;
  }

  .metric-value {
    font-size: 0.875rem;
    font-weight: 600;
    font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
  }

  .metric-check {
    color: #10b981;
    font-size: 0.75rem;
    margin-left: 0.25rem;
  }

  .validation-progress {
    margin-top: 0.5rem;
  }

  .progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 0.25rem;
  }

  .progress-bar {
    height: 0.375rem;
    background: #3d3d3d;
    border-radius: 0.25rem;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: #f59e0b;
    border-radius: 0.25rem;
    transition: width 0.3s ease;
  }

  .progress-fill.complete {
    background: #10b981;
  }

  /* Promotion Modal Styles */
  .validation-summary {
    background: #1e1e1e;
    border-radius: 0.375rem;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  .validation-summary h4 {
    font-size: 0.875rem;
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 0.75rem;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
  }

  .summary-item {
    display: flex;
    flex-direction: column;
    padding: 0.5rem;
    background: #2d2d2d;
    border-radius: 0.25rem;
  }

  .summary-label {
    font-size: 0.75rem;
    color: #9ca3af;
  }

  .summary-value {
    font-size: 1rem;
    font-weight: 600;
    font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .summary-threshold {
    font-size: 0.625rem;
    color: #6b7280;
  }

  .summary-value .check {
    color: #10b981;
  }

  .success-message {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 2rem;
    color: #10b981;
  }

  .success-message h4 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 1rem;
    color: #e0e0e0;
  }

  .success-message code {
    background: #1e1e1e;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
    color: #10b981;
  }

  .error-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.375rem;
    color: #ef4444;
    margin-bottom: 1rem;
    font-size: 0.875rem;
  }

  .btn-success {
    background: #10b981;
    color: white;
  }

  .btn-success:hover:not(:disabled) {
    background: #059669;
  }

  .btn-icon.btn-success {
    color: #10b981;
  }

  .btn-icon.btn-success:hover {
    background: rgba(16, 185, 129, 0.1);
    border-color: #10b981;
  }

  .text-green-400 {
    color: #4ade80;
  }

  .text-red-400 {
    color: #f87171;
  }

  .text-yellow-400 {
    color: #facc15;
  }

  .text-sm {
    font-size: 0.875rem;
  }

  .mt-2 {
    margin-top: 0.5rem;
  }
</style>
