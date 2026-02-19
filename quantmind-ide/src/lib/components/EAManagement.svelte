<script lang="ts">
  /**
   * EA Management Component
   * 
   * Displays list of all registered EAs with mode indicators,
   * virtual account balances for demo EAs, and EA promotion workflow.
   */
  import { onMount } from 'svelte';
  import ModeIndicator from './ModeIndicator.svelte';
  
  interface EAConfig {
    ea_id: string;
    name: string;
    symbol: string;
    timeframe: string;
    magic_number: number;
    mode: 'demo' | 'live';
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
  
  let eas: EAConfig[] = [];
  let virtualAccounts: Record<string, VirtualAccount> = {};
  let loading = true;
  let error: string | null = null;
  let modeFilter: 'all' | 'demo' | 'live' = 'all';
  let showRegisterModal = false;
  let selectedEA: EAConfig | null = null;
  
  // New EA form
  let newEA: Partial<EAConfig> = {
    mode: 'demo',
    virtual_balance: 1000.0,
    max_lot_size: 1.0,
    max_daily_loss_pct: 5.0
  };
  
  onMount(() => {
    loadEAs();
    loadVirtualAccounts();
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
  
  async function promoteEA(eaId: string) {
    try {
      // Comment 3: Include confirm: true in request body as required by backend
      const response = await fetch(`/api/eas/${eaId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true })
      });
      
      if (response.ok) {
        await loadEAs();
      } else {
        // Comment 3: Handle non-200 responses to surface backend errors
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
  
  $: filteredEAs = eas.filter(ea => {
    if (modeFilter === 'all') return true;
    return ea.mode === modeFilter;
  });
  
  $: demoCount = eas.filter(ea => ea.mode === 'demo').length;
  $: liveCount = eas.filter(ea => ea.mode === 'live').length;
</script>

<div class="ea-management">
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
  
  <!-- Filter -->
  <div class="filters">
    <label>Mode Filter:</label>
    <select bind:value={modeFilter}>
      <option value="all">All Modes</option>
      <option value="demo">Demo Only</option>
      <option value="live">Live Only</option>
    </select>
  </div>
  
  <!-- Loading State -->
  {#if loading}
    <div class="loading">Loading EAs...</div>
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
    
    {#if filteredEAs.length === 0 && !loading}
      <div class="no-results">No EAs found</div>
    {/if}
  </div>
</div>

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
</style>