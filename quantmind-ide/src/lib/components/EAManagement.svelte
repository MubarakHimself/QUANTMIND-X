<script lang="ts">
  /**
   * EA Management Component
   *
   * Displays list of all registered EAs with mode indicators,
   * virtual account balances for demo EAs, and EA promotion workflow.
   * Also displays processed video strategies from YouTube.
   */
  import { onMount } from 'svelte';
  import ModeIndicator from './ModeIndicator.svelte';
  import Breadcrumbs from './Breadcrumbs.svelte';

  // Navigation handler for breadcrumbs
  function handleBreadcrumbNavigate(path: string) {
    console.log('Navigate to:', path);
    // Handle navigation - could route to different views or dispatch events
  }

  // Breadcrumb items for Video Ingest workflow navigation
  const breadcrumbItems = [
    { label: 'EA Management', path: '/ea-management' },
    { label: 'Video Ingest', path: '/video-ingest' }
  ];

  interface EAConfig {
    ea_id: string;
    name: string;
    symbol: string;
    timeframe: string;
    magic_number: number;
    mode: 'demo' | 'live';
    variant: 'vanilla' | 'spiced';
    virtual_balance: number;
    preferred_regime?: string;
    preferred_volatility?: string;
    max_lot_size: number;
    max_daily_loss_pct: number;
    tags: string[];
  }

  interface VirtualAccount {
    ea_id: string;
    initial_balance: number;
    current_balance: number;
    equity: number;
    margin_used: number;
    free_margin: number;
    pnl: number;
    pnl_pct: number;
    last_updated: string;
  }

  interface VideoStrategy {
    strategy_id: string;
    name: string;
    video_id: string;
    trd_file: string;
    config_file: string;
    created_at: string;
    tags: string[];
    symbols: string[];
    timeframes: string[];
    strategy_type: string;
    has_ea_code: boolean;
    has_backtest_reports: boolean;
  }

  let eas: EAConfig[] = [];
  let virtualAccounts: Record<string, VirtualAccount> = {};
  let videoStrategies: VideoStrategy[] = [];
  let loading = true;
  let loadingVideoStrategies = true;
  let error: string | null = null;
  let modeFilter: 'all' | 'demo' | 'live' = 'all';
  let showRegisterModal = false;
  let selectedEA: EAConfig | null = null;
  let activeTab: 'eas' | 'video-strategies' = 'eas';
  let expandedStrategy: string | null = null;
  let selectedTRD: { strategy_id: string; content: string } | null = null;

  // New EA form
  let newEA: Partial<EAConfig> = {
    mode: 'demo',
    variant: 'vanilla',
    virtual_balance: 1000.0,
    max_lot_size: 1.0,
    max_daily_loss_pct: 5.0
  };

  onMount(() => {
    loadEAs();
    loadVirtualAccounts();
    loadVideoStrategies();
  });

  async function loadEAs() {
    try {
      loading = true;
      const response = await fetch('/api/eas');
      if (response.ok) {
        eas = await response.json();
      } else {
        error = 'Failed to load EAs';
      }
    } catch (e) {
      error = 'Error loading EAs';
      console.error(e);
    } finally {
      loading = false;
    }
  }

  async function loadVirtualAccounts() {
    try {
      const response = await fetch('/api/virtual-accounts');
      if (response.ok) {
        const accounts = await response.json();
        virtualAccounts = accounts.reduce((acc: Record<string, VirtualAccount>, account: VirtualAccount) => {
          acc[account.ea_id] = account;
          return acc;
        }, {});
      }
    } catch (e) {
      console.error('Error loading virtual accounts:', e);
    }
  }

  async function loadVideoStrategies() {
    try {
      loadingVideoStrategies = true;
      const response = await fetch('/api/video-strategies');
      if (response.ok) {
        videoStrategies = await response.json();
      }
    } catch (e) {
      console.error('Error loading video strategies:', e);
    } finally {
      loadingVideoStrategies = false;
    }
  }

  async function loadTRDContent(strategyId: string) {
    try {
      const response = await fetch(`/api/video-strategies/${strategyId}/trd`);
      if (response.ok) {
        selectedTRD = await response.json();
      }
    } catch (e) {
      console.error('Error loading TRD:', e);
    }
  }

  async function promoteEA(eaId: string) {
    try {
      const response = await fetch(`/api/eas/${eaId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true })
      });

      if (response.ok) {
        await loadEAs();
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        error = `Failed to promote EA: ${errorData.detail || response.statusText}`;
        console.error('Promotion failed:', errorData);
      }
    } catch (e) {
      error = 'Error promoting EA';
      console.error(e);
    }
  }

  async function registerEA() {
    try {
      const response = await fetch('/api/eas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newEA)
      });

      if (response.ok) {
        showRegisterModal = false;
        newEA = {
          mode: 'demo',
          variant: 'vanilla',
          virtual_balance: 1000.0,
          max_lot_size: 1.0,
          max_daily_loss_pct: 5.0
        };
        await loadEAs();
        await loadVirtualAccounts();
      } else {
        error = 'Failed to register EA';
      }
    } catch (e) {
      error = 'Error registering EA';
      console.error(e);
    }
  }

  function toggleStrategy(strategyId: string) {
    if (expandedStrategy === strategyId) {
      expandedStrategy = null;
    } else {
      expandedStrategy = strategyId;
      selectedTRD = null;
    }
  }

  function viewTRD(strategy: VideoStrategy) {
    loadTRDContent(strategy.strategy_id);
  }

  $: filteredEAs = eas.filter(ea => {
    if (modeFilter === 'all') return true;
    return ea.mode === modeFilter;
  });

  $: demoCount = eas.filter(ea => ea.mode === 'demo').length;
  $: liveCount = eas.filter(ea => ea.mode === 'live').length;
</script>

<div class="ea-management">
  <!-- Breadcrumbs for Video Ingest workflow -->
  <Breadcrumbs items={breadcrumbItems} onNavigate={handleBreadcrumbNavigate} showHome={true} />

  <!-- Header -->
  <div class="header">
    <h2>EA Management</h2>
    <button class="btn-primary" on:click={() => showRegisterModal = true}>
      + Register EA
    </button>
  </div>
  
  <!-- Stats -->
  <div class="stats-bar">
    <div class="stat">
      <span class="stat-label">Total EAs</span>
      <span class="stat-value">{eas.length}</span>
    </div>
    <div class="stat demo">
      <span class="stat-label">🧪 Demo</span>
      <span class="stat-value">{demoCount}</span>
    </div>
    <div class="stat live">
      <span class="stat-label">🔴 Live</span>
      <span class="stat-value">{liveCount}</span>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button
      class="tab"
      class:active={activeTab === 'eas'}
      on:click={() => activeTab = 'eas'}
    >
      Registered EAs
    </button>
    <button
      class="tab"
      class:active={activeTab === 'video-strategies'}
      on:click={() => activeTab = 'video-strategies'}
    >
      Video Strategies
      {#if videoStrategies.length > 0}
        <span class="badge">{videoStrategies.length}</span>
      {/if}
    </button>
  </div>

  {#if activeTab === 'eas'}
    <!-- Filter -->
    <div class="filters">
      <label>Mode Filter:</label>
      <select bind:value={modeFilter}>
        <option value="all">All Modes</option>
        <option value="demo">Demo Only</option>
        <option value="live">Live Only</option>
      </select>
    </div>
  {/if}

  <!-- Loading State -->
  {#if loading && activeTab === 'eas'}
    <div class="loading">Loading EAs...</div>
  {/if}

  {#if loadingVideoStrategies && activeTab === 'video-strategies'}
    <div class="loading">Loading Video Strategies...</div>
  {/if}
  
  <!-- Error State -->
  {#if error}
    <div class="error">{error}</div>
  {/if}
  
  <!-- EA List -->
  <div class="ea-list">
    {#each filteredEAs as ea (ea.ea_id)}
      <div class="ea-card" class:demo={ea.mode === 'demo'} class:live={ea.mode === 'live'}>
        <div class="ea-header">
          <h3>{ea.name}</h3>
          <ModeIndicator mode={ea.mode} size="sm" />
        </div>
        
        <div class="ea-details">
          <div class="detail">
            <span class="label">Symbol:</span>
            <span class="value">{ea.symbol}</span>
          </div>
          <div class="detail">
            <span class="label">Timeframe:</span>
            <span class="value">{ea.timeframe}</span>
          </div>
          <div class="detail">
            <span class="label">Magic #:</span>
            <span class="value">{ea.magic_number}</span>
          </div>
          <div class="detail">
            <span class="label">Max Lot:</span>
            <span class="value">{ea.max_lot_size}</span>
          </div>
          <div class="detail">
            <span class="label">Variant:</span>
            <span class="value" class:spiced={ea.variant === 'spiced'}>{ea.variant || 'vanilla'}</span>
          </div>
        </div>
        
        <!-- Virtual Balance for Demo EAs -->
        {#if ea.mode === 'demo' && virtualAccounts[ea.ea_id]}
          <div class="virtual-balance">
            <h4>Virtual Account</h4>
            <div class="balance-info">
              <div class="balance">
                <span class="label">Balance:</span>
                <span class="value">${virtualAccounts[ea.ea_id].current_balance.toFixed(2)}</span>
              </div>
              <div class="pnl" class:positive={virtualAccounts[ea.ea_id].pnl >= 0} class:negative={virtualAccounts[ea.ea_id].pnl < 0}>
                <span class="label">P&L:</span>
                <span class="value">${virtualAccounts[ea.ea_id].pnl.toFixed(2)} ({virtualAccounts[ea.ea_id].pnl_pct.toFixed(2)}%)</span>
              </div>
              <div class="margin">
                <span class="label">Free Margin:</span>
                <span class="value">${virtualAccounts[ea.ea_id].free_margin.toFixed(2)}</span>
              </div>
            </div>
          </div>
        {/if}
        
        <!-- Tags -->
        {#if ea.tags && ea.tags.length > 0}
          <div class="tags">
            {#each ea.tags as tag}
              <span class="tag">{tag}</span>
            {/each}
          </div>
        {/if}
        
        <!-- Actions -->
        <div class="actions">
          {#if ea.mode === 'demo'}
            <button class="btn-promote" on:click={() => promoteEA(ea.ea_id)}>
              🚀 Promote to Live
            </button>
          {:else}
            <button class="btn-demote" on:click={() => selectedEA = ea}>
              ⬇️ Demote to Demo
            </button>
          {/if}
        </div>
      </div>
    {/each}

    {#if filteredEAs.length === 0 && !loading && activeTab === 'eas'}
      <div class="no-results">No EAs found</div>
    {/if}
  </div>
{/if}

<!-- Video Strategies Section -->
{#if activeTab === 'video-strategies'}
  <div class="video-strategies">
    {#if videoStrategies.length === 0 && !loadingVideoStrategies}
      <div class="no-results">
        <p>No video strategies found.</p>
        <p class="sub-text">Process YouTube videos to create strategies.</p>
      </div>
    {:else}
      <div class="strategy-list">
        {#each videoStrategies as strategy (strategy.strategy_id)}
          <div class="strategy-card" class:expanded={expandedStrategy === strategy.strategy_id}>
            <div class="strategy-header" on:click={() => toggleStrategy(strategy.strategy_id)}>
              <div class="strategy-info">
                <div class="strategy-icon">📹</div>
                <div class="strategy-details">
                  <h3>{strategy.name}</h3>
                  <div class="strategy-meta">
                    <span class="tag">{strategy.strategy_type}</span>
                    <span class="symbols">{strategy.symbols.join(', ') || 'No symbols'}</span>
                    <span class="timeframes">{strategy.timeframes.join(', ') || 'No timeframes'}</span>
                  </div>
                </div>
              </div>
              <div class="strategy-status">
                {#if strategy.has_ea_code}
                  <span class="status-badge ea" title="EA Code Generated">🤖 EA</span>
                {/if}
                {#if strategy.has_backtest_reports}
                  <span class="status-badge backtest" title="Backtest Reports Available">📊 Backtest</span>
                {/if}
                <span class="expand-icon">{expandedStrategy === strategy.strategy_id ? '▼' : '▶'}</span>
              </div>
            </div>

            {#if expandedStrategy === strategy.strategy_id}
              <div class="strategy-content">
                <!-- Folder Structure: Video → TRD → EA Code → Backtest Reports -->
                <div class="folder-structure">
                  <div class="folder-level">
                    <div class="folder-item video">
                      <span class="folder-icon">📹</span>
                      <span class="folder-label">Video Source</span>
                      <span class="folder-value">{strategy.video_id}</span>
                    </div>
                    <div class="connector">→</div>
                    <div class="folder-item trd" class:active={selectedTRD?.strategy_id === strategy.strategy_id}>
                      <span class="folder-icon">📄</span>
                      <span class="folder-label">TRD</span>
                      <button class="btn-view" on:click|stopPropagation={() => viewTRD(strategy)}>
                        View
                      </button>
                    </div>
                    <div class="connector">→</div>
                    <div class="folder-item ea-code">
                      <span class="folder-icon">🤖</span>
                      <span class="folder-label">EA Code</span>
                      {#if strategy.has_ea_code}
                        <span class="folder-value available">Generated</span>
                      {:else}
                        <span class="folder-value pending">Not Generated</span>
                      {/if}
                    </div>
                    <div class="connector">→</div>
                    <div class="folder-item backtest">
                      <span class="folder-icon">📊</span>
                      <span class="folder-label">Backtest</span>
                      {#if strategy.has_backtest_reports}
                        <span class="folder-value available">Available</span>
                      {:else}
                        <span class="folder-value pending">No Reports</span>
                      {/if}
                    </div>
                  </div>
                </div>

                <!-- TRD Content Preview -->
                {#if selectedTRD?.strategy_id === strategy.strategy_id && selectedTRD?.content}
                  <div class="trd-preview">
                    <div class="trd-header">
                      <h4>Technical Requirements Document</h4>
                      <button class="btn-close-trd" on:click={() => selectedTRD = null}>×</button>
                    </div>
                    <pre class="trd-content">{selectedTRD.content}</pre>
                  </div>
                {/if}

                <!-- Strategy Actions -->
                <div class="strategy-actions">
                  {#if !strategy.has_ea_code}
                    <button class="btn-generate-ea" on:click={() => console.log('Generate EA:', strategy.strategy_id)}>
                      Generate EA Code
                    </button>
                  {/if}
                  {#if strategy.has_ea_code && !strategy.has_backtest_reports}
                    <button class="btn-run-backtest" on:click={() => console.log('Run backtest:', strategy.strategy_id)}>
                      Run Backtest
                    </button>
                  {/if}
                  {#if strategy.has_ea_code}
                    <button class="btn-register-ea" on:click={() => {
                      newEA = {
                        ea_id: strategy.strategy_id,
                        name: strategy.name,
                        symbol: strategy.symbols[0] || 'EURUSD',
                        timeframe: strategy.timeframes[0] || 'H1',
                        magic_number: Math.floor(Math.random() * 900000) + 100000,
                        mode: 'demo',
                        variant: 'vanilla',
                        virtual_balance: 1000.0,
                        max_lot_size: 1.0,
                        max_daily_loss_pct: 5.0,
                        tags: strategy.tags
                      };
                      showRegisterModal = true;
                    }}>
                      Register as EA
                    </button>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
{/if}

<!-- Register EA Modal -->
{#if showRegisterModal}
  <div class="modal-overlay" on:click={() => showRegisterModal = false}>
    <div class="modal" on:click|stopPropagation>
      <h3>Register New EA</h3>
      
      <form on:submit|preventDefault={registerEA}>
        <div class="form-group">
          <label>EA ID *</label>
          <input type="text" bind:value={newEA.ea_id} required />
        </div>
        
        <div class="form-group">
          <label>Name *</label>
          <input type="text" bind:value={newEA.name} required />
        </div>
        
        <div class="form-group">
          <label>Symbol *</label>
          <input type="text" bind:value={newEA.symbol} placeholder="EURUSD" required />
        </div>
        
        <div class="form-group">
          <label>Timeframe *</label>
          <input type="text" bind:value={newEA.timeframe} placeholder="H1" required />
        </div>
        
        <div class="form-group">
          <label>Magic Number *</label>
          <input type="number" bind:value={newEA.magic_number} required />
        </div>
        
        <div class="form-group">
          <label>Mode</label>
          <select bind:value={newEA.mode}>
            <option value="demo">Demo</option>
            <option value="live">Live</option>
          </select>
        </div>

        <div class="form-group">
          <label>Variant</label>
          <select bind:value={newEA.variant}>
            <option value="vanilla">Vanilla (Basic)</option>
            <option value="spiced">Spiced (Enhanced)</option>
          </select>
        </div>

        {#if newEA.mode === 'demo'}
          <div class="form-group">
            <label>Virtual Balance</label>
            <input type="number" bind:value={newEA.virtual_balance} step="100" />
          </div>
        {/if}
        
        <div class="form-group">
          <label>Max Lot Size</label>
          <input type="number" bind:value={newEA.max_lot_size} step="0.01" />
        </div>
        
        <div class="form-actions">
          <button type="button" class="btn-cancel" on:click={() => showRegisterModal = false}>Cancel</button>
          <button type="submit" class="btn-primary">Register</button>
        </div>
      </form>
    </div>
  </div>
{/if}

<style>
  .ea-management {
    padding: 1rem;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }
  
  .header h2 {
    margin: 0;
    font-size: 1.5rem;
  }
  
  .stats-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  
  .stat {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.75rem 1rem;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  
  .stat.demo {
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.3);
  }
  
  .stat.live {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
  }
  
  .stat-label {
    font-size: 0.75rem;
    opacity: 0.8;
  }
  
  .stat-value {
    font-size: 1.25rem;
    font-weight: bold;
  }
  
  .filters {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }
  
  .filters select {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 6px;
    padding: 0.5rem 1rem;
    color: inherit;
  }
  
  .ea-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1rem;
  }
  
  .ea-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.2s ease;
  }
  
  .ea-card:hover {
    background: rgba(255, 255, 255, 0.08);
  }
  
  .ea-card.demo {
    border-color: rgba(251, 191, 36, 0.3);
  }
  
  .ea-card.live {
    border-color: rgba(239, 68, 68, 0.3);
  }
  
  .ea-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }
  
  .ea-header h3 {
    margin: 0;
    font-size: 1.1rem;
  }
  
  .ea-details {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  
  .detail {
    display: flex;
    gap: 0.5rem;
  }
  
  .detail .label {
    opacity: 0.7;
    font-size: 0.85rem;
  }
  
  .detail .value {
    font-weight: 500;
    font-size: 0.85rem;
  }

  .detail .value.spiced {
    color: #c084fc;
  }
  
  .virtual-balance {
    background: rgba(251, 191, 36, 0.1);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 1rem;
  }
  
  .virtual-balance h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.85rem;
    color: #fbbf24;
  }
  
  .balance-info {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }
  
  .balance, .pnl, .margin {
    display: flex;
    gap: 0.25rem;
    font-size: 0.8rem;
  }
  
  .pnl.positive .value {
    color: #22c55e;
  }
  
  .pnl.negative .value {
    color: #ef4444;
  }
  
  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  
  .tag {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
  }
  
  .actions {
    display: flex;
    gap: 0.5rem;
  }
  
  .btn-primary, .btn-promote, .btn-demote, .btn-cancel {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.2s ease;
  }
  
  .btn-primary {
    background: #3b82f6;
    color: white;
  }
  
  .btn-promote {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
  }
  
  .btn-demote {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.3);
  }
  
  .btn-cancel {
    background: rgba(255, 255, 255, 0.1);
    color: inherit;
  }
  
  .loading, .error, .no-results {
    text-align: center;
    padding: 2rem;
    opacity: 0.7;
  }
  
  .error {
    color: #ef4444;
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
    z-index: 100;
  }
  
  .modal {
    background: #1a1a2e;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    width: 100%;
    max-width: 400px;
  }
  
  .modal h3 {
    margin: 0 0 1rem 0;
  }
  
  .form-group {
    margin-bottom: 1rem;
  }
  
  .form-group label {
    display: block;
    margin-bottom: 0.25rem;
    font-size: 0.85rem;
    opacity: 0.8;
  }
  
  .form-group input, .form-group select {
    width: 100%;
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 6px;
    color: inherit;
  }
  
  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 1.5rem;
  }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 0.5rem;
  }

  .tab {
    background: transparent;
    border: none;
    color: rgba(255, 255, 255, 0.6);
    padding: 0.75rem 1.25rem;
    font-size: 0.95rem;
    cursor: pointer;
    border-radius: 8px 8px 0 0;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .tab:hover {
    background: rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.9);
  }

  .tab.active {
    color: #3b82f6;
    background: rgba(59, 130, 246, 0.1);
    border-bottom: 2px solid #3b82f6;
  }

  .badge {
    background: #3b82f6;
    color: white;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: bold;
  }

  /* Video Strategies Section */
  .video-strategies {
    padding: 0;
  }

  .strategy-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .strategy-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.2s ease;
  }

  .strategy-card:hover {
    background: rgba(255, 255, 255, 0.08);
  }

  .strategy-card.expanded {
    border-color: rgba(59, 130, 246, 0.3);
  }

  .strategy-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.25rem;
    cursor: pointer;
  }

  .strategy-info {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .strategy-icon {
    font-size: 1.5rem;
  }

  .strategy-details h3 {
    margin: 0 0 0.25rem 0;
    font-size: 1rem;
  }

  .strategy-meta {
    display: flex;
    gap: 0.75rem;
    font-size: 0.8rem;
    opacity: 0.7;
  }

  .strategy-meta .tag {
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
  }

  .strategy-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .status-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
  }

  .status-badge.ea {
    background: rgba(168, 85, 247, 0.2);
    color: #c084fc;
  }

  .status-badge.backtest {
    background: rgba(34, 197, 94, 0.2);
    color: #4ade80;
  }

  .expand-icon {
    opacity: 0.5;
    font-size: 0.75rem;
    margin-left: 0.5rem;
  }

  .strategy-content {
    padding: 1rem 1.25rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(0, 0, 0, 0.2);
  }

  /* Folder Structure */
  .folder-structure {
    margin-bottom: 1rem;
  }

  .folder-level {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .folder-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.75rem 1rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    min-width: 80px;
    transition: all 0.2s ease;
  }

  .folder-item.active {
    border-color: #3b82f6;
    background: rgba(59, 130, 246, 0.1);
  }

  .folder-icon {
    font-size: 1.25rem;
    margin-bottom: 0.25rem;
  }

  .folder-label {
    font-size: 0.7rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .folder-value {
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.25rem;
  }

  .folder-value.available {
    color: #4ade80;
  }

  .folder-value.pending {
    color: #fbbf24;
  }

  .connector {
    color: rgba(255, 255, 255, 0.3);
    font-size: 1rem;
  }

  .btn-view {
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border: none;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    cursor: pointer;
    margin-top: 0.25rem;
    transition: all 0.2s ease;
  }

  .btn-view:hover {
    background: rgba(59, 130, 246, 0.4);
  }

  /* TRD Preview */
  .trd-preview {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    margin-bottom: 1rem;
    overflow: hidden;
  }

  .trd-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .trd-header h4 {
    margin: 0;
    font-size: 0.9rem;
  }

  .btn-close-trd {
    background: transparent;
    border: none;
    color: rgba(255, 255, 255, 0.6);
    font-size: 1.25rem;
    cursor: pointer;
    line-height: 1;
  }

  .btn-close-trd:hover {
    color: white;
  }

  .trd-content {
    padding: 1rem;
    margin: 0;
    font-size: 0.8rem;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
    font-family: 'Consolas', 'Monaco', monospace;
    line-height: 1.5;
  }

  /* Strategy Actions */
  .strategy-actions {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  .btn-generate-ea, .btn-run-backtest, .btn-register-ea {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    border: none;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-generate-ea {
    background: rgba(168, 85, 247, 0.2);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.3);
  }

  .btn-generate-ea:hover {
    background: rgba(168, 85, 247, 0.4);
  }

  .btn-run-backtest {
    background: rgba(34, 197, 94, 0.2);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
  }

  .btn-run-backtest:hover {
    background: rgba(34, 197, 94, 0.4);
  }

  .btn-register-ea {
    background: #3b82f6;
    color: white;
  }

  .btn-register-ea:hover {
    background: #2563eb;
  }

  .sub-text {
    font-size: 0.85rem;
    opacity: 0.6;
    margin-top: 0.5rem;
  }
</style>