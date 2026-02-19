<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { WebSocketClient } from "../ws-client";
  import type { WebSocketMessage } from "../ws-client";
  import {
    Play,
    Square,
    FileText,
    X,
    Plus,
    TrendingUp,
    Award,
    AlertCircle,
    ArrowUp,
    Bot,
    LayoutGrid,
    Code,
    GitBranch,
    Upload,
  } from "lucide-svelte";

  // Props
  export let baseUrl: string = "http://localhost:8000";

  // ==========================================================================
  // Type Definitions
  // ==========================================================================

  // Bot manifest interface (matching backend BotManifest)
  interface BotManifest {
    bot_id: string;
    name: string;
    format: string;
    symbol?: string;
    timeframe?: string;
    trading_mode: string;
    source_type: string;
    tags: string[];
    // Performance stats
    total_trades?: number;
    winning_trades?: number;
    losing_trades?: number;
    win_rate?: number;
    total_pnl?: number;
    sharpe_ratio?: number;
    // Lifecycle
    promotion_eligible?: boolean;
  }

  // Demo account interface
  interface DemoAccount {
    login: number;
    server: string;
    broker: string;
    nickname: string;
    account_type: string;
    is_active: boolean;
  }

  // Deployment request interface
  interface DeploymentRequest {
    format: "ea" | "pine_script" | "python";
    source: "github" | "local" | "tradingview" | "inline";
    ea_id?: number;
    pine_script_code?: string;
    python_code?: string;
    file_path?: string;
    strategy_name: string;
    symbol: string;
    timeframe: string;
    config: Record<string, any>;
    mt5_demo_login?: number;
    mt5_demo_password?: string;
    mt5_demo_server?: string;
    initial_tag: string;
  }

  // Deployment result interface
  interface DeploymentResult {
    bot_id: string;
    agent_id?: string;
    status: string;
    format: string;
  }

  // ==========================================================================
  // State
  // ==========================================================================

  // Bots organized by tag
  let botsByTag: Record<string, BotManifest[]> = {
    "@primal": [],
    "@pending": [],
    "@perfect": [],
    "@live": [],
  };

  // All bots flat list
  let allBots: BotManifest[] = [];

  // Demo accounts
  let demoAccounts: DemoAccount[] = [];

  // Selected tag filter
  let selectedTag: string = "@primal";

  // UI State
  let isLoading = false;
  let isDeploying = false;
  let error: string | null = null;
  let wsClient: WebSocketClient | null = null;
  let showDeployModal = false;

  // Deploy form state
  let deployForm: DeploymentRequest = {
    format: "pine_script",
    source: "inline",
    strategy_name: "",
    symbol: "EURUSD",
    timeframe: "H1",
    config: {},
    initial_tag: "@primal",
    pine_script_code: "",
    python_code: "",
    ea_id: undefined,
    file_path: undefined,
    mt5_demo_login: undefined,
    mt5_demo_password: undefined,
    mt5_demo_server: undefined,
  };

  // Tag thresholds for progress bars
  const tagThresholds = {
    "@primal": { trades: 20, winRate: 60, sharpe: 1.5 },
    "@pending": { trades: 50, winRate: 65, sharpe: 1.8 },
    "@perfect": { trades: 100, winRate: 70, sharpe: 2.0 },
  };

  // Options
  const symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "ETHUSD"];
  const timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"];
  const botFormats = [
    { value: "ea", label: "EA (MQL5)" },
    { value: "pine_script", label: "Pine Script" },
    { value: "python", label: "Python" },
  ];
  const botSources = [
    { value: "github", label: "GitHub" },
    { value: "local", label: "Local File" },
    { value: "tradingview", label: "TradingView" },
    { value: "inline", label: "Inline Code" },
  ];

  // ==========================================================================
  // Lifecycle
  // ==========================================================================

  onMount(async () => {
    await Promise.all([fetchBotsByTag(), fetchDemoAccounts()]);
    connectWebSocket();
  });

  onDestroy(() => {
    if (wsClient) {
      wsClient.disconnect();
    }
  });

  // ==========================================================================
  // API Calls
  // ==========================================================================

  async function fetchBotsByTag() {
    isLoading = true;
    error = null;
    try {
      const response = await fetch(`${baseUrl}/api/paper-trading/bots/by-tag`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      botsByTag = await response.json();
      
      // Flatten for display
      allBots = Object.values(botsByTag).flat();
    } catch (e) {
      error = `Failed to fetch bots: ${e instanceof Error ? e.message : String(e)}`;
      console.error(error);
    } finally {
      isLoading = false;
    }
  }

  async function fetchDemoAccounts() {
    try {
      const response = await fetch(`${baseUrl}/api/paper-trading/demo-accounts`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      demoAccounts = await response.json();
    } catch (e) {
      console.error("Failed to fetch demo accounts:", e);
    }
  }

  async function deployBot() {
    isDeploying = true;
    error = null;
    try {
      const response = await fetch(`${baseUrl}/api/paper-trading/deploy/enhanced`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(deployForm),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result: DeploymentResult = await response.json();
      
      // Refresh bots list
      await fetchBotsByTag();
      
      // Close modal and reset form
      showDeployModal = false;
      resetDeployForm();
      
      console.log("Deployment successful:", result);
    } catch (e) {
      error = `Deployment failed: ${e instanceof Error ? e.message : String(e)}`;
      console.error(error);
    } finally {
      isDeploying = false;
    }
  }

  async function promoteBot(botId: string) {
    try {
      const response = await fetch(
        `${baseUrl}/api/paper-trading/bots/${botId}/promote-tag`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // Refresh bots list
        await fetchBotsByTag();
      } else {
        throw new Error(result.error || "Promotion failed");
      }
    } catch (e) {
      error = `Promotion failed: ${e instanceof Error ? e.message : String(e)}`;
      console.error(error);
    }
  }

  async function verifyDemoAccount(login: number) {
    try {
      const response = await fetch(
        `${baseUrl}/api/paper-trading/demo-accounts/${login}/verify`
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    } catch (e) {
      console.error("Failed to verify account:", e);
      return null;
    }
  }

  // ==========================================================================
  // WebSocket
  // ==========================================================================

  function connectWebSocket() {
    wsClient = new WebSocketClient(baseUrl);
    
    wsClient.onMessage = (message: WebSocketMessage) => {
      if (message.type === "paper_trading_update") {
        // Refresh bots on update
        fetchBotsByTag();
      }
    };
    
    wsClient.connect();
  }

  // ==========================================================================
  // Helpers
  // ==========================================================================

  function resetDeployForm() {
    deployForm = {
      format: "pine_script",
      source: "inline",
      strategy_name: "",
      symbol: "EURUSD",
      timeframe: "H1",
      config: {},
      initial_tag: "@primal",
      pine_script_code: "",
      python_code: "",
      ea_id: undefined,
      file_path: undefined,
      mt5_demo_login: undefined,
      mt5_demo_password: undefined,
      mt5_demo_server: undefined,
    };
  }

  function getTagColor(tag: string): string {
    switch (tag) {
      case "@primal":
        return "#9333ea"; // Purple
      case "@pending":
        return "#f97316"; // Orange
      case "@perfect":
        return "#3b82f6"; // Blue
      case "@live":
        return "#22c55e"; // Green
      default:
        return "#6b7280"; // Gray
    }
  }

  function getFormatIcon(format: string) {
    switch (format) {
      case "EA":
        return Bot;
      case "Pine Script":
        return LayoutGrid;
      case "Python":
        return Code;
      default:
        return FileText;
    }
  }

  function calculateProgress(bot: BotManifest): number {
    const threshold = tagThresholds[bot.tags?.[0] as keyof typeof tagThresholds];
    if (!threshold || !bot.total_trades) return 0;
    
    const tradeProgress = Math.min(bot.total_trades / threshold.trades, 1) * 50;
    const winRateProgress = (bot.win_rate || 0) >= threshold.winRate ? 25 : (bot.win_rate || 0) / threshold.winRate * 25;
    const sharpeProgress = (bot.sharpe_ratio || 0) >= threshold.sharpe ? 25 : (bot.sharpe_ratio || 0) / threshold.sharpe * 25;
    
    return Math.round(tradeProgress + winRateProgress + sharpeProgress);
  }

  function formatPnL(pnl: number | undefined): string {
    if (pnl === undefined) return "-";
    const sign = pnl >= 0 ? "+" : "";
    return `${sign}$${pnl.toFixed(2)}`;
  }

  function formatWinRate(winRate: number | undefined): string {
    if (winRate === undefined) return "-";
    return `${(winRate * 100).toFixed(1)}%`;
  }

  // Filter bots by selected tag
  $: filteredBots = selectedTag 
    ? botsByTag[selectedTag] || [] 
    : allBots;
</script>

<div class="enhanced-paper-trading-panel">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-title">
      <TrendingUp size={20} />
      <h2>Enhanced Paper Trading</h2>
    </div>
    <button 
      class="deploy-button"
      on:click={() => showDeployModal = true}
    >
      <Plus size={16} />
      Deploy New Bot
    </button>
  </div>

  <!-- Error Display -->
  {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
      <button on:click={() => error = null}>
        <X size={14} />
      </button>
    </div>
  {/if}

  <!-- Tag Filter Tabs -->
  <div class="tag-tabs">
    {#each Object.keys(botsByTag) as tag}
      <button
        class="tag-tab"
        class:active={selectedTag === tag}
        style="--tag-color: {getTagColor(tag)}"
        on:click={() => selectedTag = tag}
      >
        <span class="tag-dot" style="background-color: {getTagColor(tag)}"></span>
        {tag}
        <span class="tag-count">({botsByTag[tag]?.length || 0})</span>
      </button>
    {/each}
  </div>

  <!-- Bots Grid -->
  <div class="bots-container">
    {#if isLoading}
      <div class="loading-state">
        <div class="spinner"></div>
        <span>Loading bots...</span>
      </div>
    {:else if filteredBots.length === 0}
      <div class="empty-state">
        <Bot size={48} />
        <h3>No bots in {selectedTag}</h3>
        <p>Deploy a new bot to get started with paper trading</p>
        <button class="deploy-button" on:click={() => showDeployModal = true}>
          <Plus size={16} />
          Deploy Bot
        </button>
      </div>
    {:else}
      <div class="bots-grid">
        {#each filteredBots as bot}
          <div class="bot-card" style="border-left-color: {getTagColor(bot.tags?.[0] || '@primal')}">
            <div class="bot-header">
              <span class="bot-tag" style="background-color: {getTagColor(bot.tags?.[0] || '@primal')}">
                {bot.tags?.[0] || '@primal'}
              </span>
              <span class="bot-format">{bot.format}</span>
            </div>
            
            <h3 class="bot-name">{bot.name}</h3>
            
            <div class="bot-details">
              <div class="detail-row">
                <span class="detail-label">Symbol:</span>
                <span class="detail-value">{bot.symbol || '-'}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Timeframe:</span>
                <span class="detail-value">{bot.timeframe || '-'}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Trades:</span>
                <span class="detail-value">{bot.total_trades || 0}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Win Rate:</span>
                <span class="detail-value">{formatWinRate(bot.win_rate)}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Sharpe:</span>
                <span class="detail-value">{bot.sharpe_ratio?.toFixed(2) || '-'}</span>
              </div>
              <div class="detail-row pnl">
                <span class="detail-label">PnL:</span>
                <span class="detail-value" class:positive={(bot.total_pnl || 0) >= 0} class:negative={(bot.total_pnl || 0) < 0}>
                  {formatPnL(bot.total_pnl)}
                </span>
              </div>
            </div>

            <!-- Progress Bar -->
            <div class="progress-section">
              <div class="progress-bar">
                <div 
                  class="progress-fill" 
                  style="width: {calculateProgress(bot)}%"
                ></div>
              </div>
              <span class="progress-label">{calculateProgress(bot)}% to next tag</span>
            </div>

            <!-- Actions -->
            <div class="bot-actions">
              <button class="action-btn secondary">
                <Square size={14} />
                Stop
              </button>
              {#if bot.promotion_eligible}
                <button 
                  class="action-btn primary"
                  on:click={() => promoteBot(bot.bot_id)}
                >
                  <ArrowUp size={14} />
                  Promote
                </button>
              {/if}
              <button class="action-btn secondary">
                <FileText size={14} />
                Details
              </button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Deploy Modal -->
  {#if showDeployModal}
    <div class="modal-overlay" on:click={() => showDeployModal = false}>
      <div class="modal-content" on:click|stopPropagation>
        <div class="modal-header">
          <h3>Deploy New Bot</h3>
          <button class="close-btn" on:click={() => showDeployModal = false}>
            <X size={18} />
          </button>
        </div>

        <div class="modal-body">
          <!-- Format Selection -->
          <div class="form-group">
            <label>Bot Format</label>
            <div class="button-group">
              {#each botFormats as format}
                <button
                  class="toggle-btn"
                  class:active={deployForm.format === format.value}
                  on:click={() => deployForm.format = format.value as any}
                >
                  {format.label}
                </button>
              {/each}
            </div>
          </div>

          <!-- Source Selection -->
          <div class="form-group">
            <label>Source</label>
            <div class="button-group">
              {#each botSources as source}
                <button
                  class="toggle-btn"
                  class:active={deployForm.source === source.value}
                  on:click={() => deployForm.source = source.value as any}
                >
                  {source.label}
                </button>
              {/each}
            </div>
          </div>

          <!-- Format-specific inputs -->
          {#if deployForm.format === 'ea' && deployForm.source === 'github'}
            <div class="form-group">
              <label>EA ID</label>
              <input
                type="number"
                bind:value={deployForm.ea_id}
                placeholder="Enter EA ID from GitHub"
              />
            </div>
          {:else if deployForm.format === 'ea' && deployForm.source === 'local'}
            <div class="form-group">
              <label>File Path</label>
              <input
                type="text"
                bind:value={deployForm.file_path}
                placeholder="/path/to/ea.mq5"
              />
            </div>
          {:else if deployForm.format === 'pine_script'}
            <div class="form-group">
              <label>Pine Script Code</label>
              <textarea
                bind:value={deployForm.pine_script_code}
                placeholder="//@version=5&#10;strategy('My Strategy')..."
                rows="8"
              ></textarea>
            </div>
          {:else if deployForm.format === 'python'}
            <div class="form-group">
              <label>Python Code</label>
              <textarea
                bind:value={deployForm.python_code}
                placeholder="# Python trading strategy..."
                rows="8"
              ></textarea>
            </div>
          {/if}

          <!-- Strategy Details -->
          <div class="form-row">
            <div class="form-group">
              <label>Strategy Name</label>
              <input
                type="text"
                bind:value={deployForm.strategy_name}
                placeholder="My Trading Strategy"
                required
              />
            </div>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label>Symbol</label>
              <select bind:value={deployForm.symbol}>
                {#each symbols as symbol}
                  <option value={symbol}>{symbol}</option>
                {/each}
              </select>
            </div>
            <div class="form-group">
              <label>Timeframe</label>
              <select bind:value={deployForm.timeframe}>
                {#each timeframes as tf}
                  <option value={tf}>{tf}</option>
                {/each}
              </select>
            </div>
          </div>

          <!-- Demo Account -->
          <div class="form-section">
            <h4>MT5 Demo Account</h4>
            <div class="form-row">
              <div class="form-group">
                <label>Login</label>
                <input
                  type="number"
                  bind:value={deployForm.mt5_demo_login}
                  placeholder="12345678"
                />
              </div>
              <div class="form-group">
                <label>Server</label>
                <input
                  type="text"
                  bind:value={deployForm.mt5_demo_server}
                  placeholder="MetaQuotes-Demo"
                />
              </div>
            </div>
            <div class="form-group">
              <label>Password</label>
              <input
                type="password"
                bind:value={deployForm.mt5_demo_password}
                placeholder="********"
              />
            </div>
          </div>
        </div>

        <div class="modal-footer">
          <button 
            class="cancel-btn" 
            on:click={() => showDeployModal = false}
          >
            Cancel
          </button>
          <button 
            class="submit-btn"
            disabled={isDeploying || !deployForm.strategy_name}
            on:click={deployBot}
          >
            {#if isDeploying}
              <div class="spinner-small"></div>
              Deploying...
            {:else}
              <Play size={16} />
              Deploy to Paper Trading
            {/if}
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .enhanced-paper-trading-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #1a1a2e;
    color: #e0e0e0;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid #2a2a4a;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .header-title h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
  }

  .deploy-button {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.2s;
  }

  .deploy-button:hover {
    background: #2563eb;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    background: rgba(239, 68, 68, 0.2);
    border-left: 3px solid #ef4444;
    color: #fca5a5;
    font-size: 14px;
  }

  .error-banner button {
    margin-left: auto;
    background: none;
    border: none;
    color: #fca5a5;
    cursor: pointer;
  }

  /* Tag Tabs */
  .tag-tabs {
    display: flex;
    gap: 8px;
    padding: 12px 20px;
    border-bottom: 1px solid #2a2a4a;
    background: #16162a;
  }

  .tag-tab {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: transparent;
    border: 1px solid #3a3a5a;
    border-radius: 20px;
    color: #a0a0b0;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
  }

  .tag-tab:hover {
    border-color: var(--tag-color);
    color: var(--tag-color);
  }

  .tag-tab.active {
    background: var(--tag-color);
    border-color: var(--tag-color);
    color: white;
  }

  .tag-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .tag-count {
    font-size: 11px;
    opacity: 0.8;
  }

  /* Bots Container */
  .bots-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: #6b7280;
  }

  .empty-state h3 {
    margin: 16px 0 8px;
    color: #9ca3af;
  }

  .empty-state p {
    margin-bottom: 20px;
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid #3a3a5a;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Bots Grid */
  .bots-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }

  .bot-card {
    background: #1e1e3a;
    border-radius: 8px;
    border-left: 4px solid #9333ea;
    padding: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
  }

  .bot-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
  }

  .bot-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .bot-tag {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    color: white;
  }

  .bot-format {
    font-size: 12px;
    color: #9ca3af;
  }

  .bot-name {
    margin: 0 0 12px;
    font-size: 16px;
    font-weight: 600;
  }

  .bot-details {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 12px;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
  }

  .detail-label {
    color: #9ca3af;
  }

  .detail-value {
    color: #e0e0e0;
    font-weight: 500;
  }

  .detail-value.positive {
    color: #22c55e;
  }

  .detail-value.negative {
    color: #ef4444;
  }

  /* Progress */
  .progress-section {
    margin-bottom: 12px;
  }

  .progress-bar {
    height: 6px;
    background: #2a2a4a;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 4px;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #22c55e);
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .progress-label {
    font-size: 11px;
    color: #9ca3af;
  }

  /* Actions */
  .bot-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.2s;
  }

  .action-btn.primary {
    background: #22c55e;
    color: white;
  }

  .action-btn.primary:hover {
    background: #16a34a;
  }

  .action-btn.secondary {
    background: #3a3a5a;
    color: #e0e0e0;
  }

  .action-btn.secondary:hover {
    background: #4a4a6a;
  }

  /* Modal */
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
    background: #1a1a2e;
    border-radius: 12px;
    width: 90%;
    max-width: 600px;
    max-height: 90vh;
    overflow-y: auto;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid #2a2a4a;
  }

  .modal-header h3 {
    margin: 0;
    font-size: 18px;
  }

  .close-btn {
    background: none;
    border: none;
    color: #9ca3af;
    cursor: pointer;
  }

  .modal-body {
    padding: 20px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 13px;
    color: #9ca3af;
  }

  .form-group input,
  .form-group select,
  .form-group textarea {
    width: 100%;
    padding: 10px 12px;
    background: #16162a;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    color: #e0e0e0;
    font-size: 14px;
    font-family: inherit;
  }

  .form-group textarea {
    resize: vertical;
    font-family: 'Monaco', 'Menlo', monospace;
  }

  .form-group input:focus,
  .form-group select:focus,
  .form-group textarea:focus {
    outline: none;
    border-color: #3b82f6;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .form-section {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #2a2a4a;
  }

  .form-section h4 {
    margin: 0 0 12px;
    font-size: 14px;
    color: #9ca3af;
  }

  .button-group {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .toggle-btn {
    padding: 8px 16px;
    background: #16162a;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    color: #9ca3af;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
  }

  .toggle-btn:hover {
    border-color: #3b82f6;
  }

  .toggle-btn.active {
    background: #3b82f6;
    border-color: #3b82f6;
    color: white;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 20px;
    border-top: 1px solid #2a2a4a;
  }

  .cancel-btn {
    padding: 10px 20px;
    background: transparent;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    color: #9ca3af;
    cursor: pointer;
    font-size: 14px;
  }

  .cancel-btn:hover {
    background: #2a2a4a;
  }

  .submit-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    background: #3b82f6;
    border: none;
    border-radius: 6px;
    color: white;
    cursor: pointer;
    font-size: 14px;
  }

  .submit-btn:hover:not(:disabled) {
    background: #2563eb;
  }

  .submit-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinner-small {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
</style>
