<script lang="ts">
  import { Globe, AlertCircle } from 'lucide-svelte';

  export let routerState: {
    mode: 'auction' | 'priority' | 'round-robin';
    auctionInterval: number;
  };

  export let mt5Connected: boolean;
  export let mt5Testing: boolean;
  export let mt5Error: string;
  export let mt5Config: {
    server: string;
    port: number;
    login: string;
    password: string;
    symbolMapping: string;
  };

  export let testMt5Connection: () => Promise<void>;
  export let saveMt5Config: () => Promise<void>;

  let mt5SymbolMappingPlaceholder = '{"EURUSDm": "EURUSD", "GBPUSDm": "GBPUSD", "USDJPYm": "USDJPY"}';
</script>

<div class="settings-section">
  <div class="setting-group">
    <h3>Router Mode</h3>
    <div class="setting-options">
      <label class="radio-option">
        <input type="radio" bind:group={routerState.mode} value="auction" />
        <span>Auction (Recommended)</span>
        <small>Best signal wins based on scoring</small>
      </label>
      <label class="radio-option">
        <input type="radio" bind:group={routerState.mode} value="priority" />
        <span>Priority</span>
        <small>Higher priority bots go first</small>
      </label>
      <label class="radio-option">
        <input type="radio" bind:group={routerState.mode} value="round-robin" />
        <span>Round Robin</span>
        <small>Equal opportunity for all bots</small>
      </label>
    </div>
  </div>

  <div class="setting-group">
    <h3>Auction Settings</h3>
    <div class="setting-row">
      <span>Interval (ms)</span>
      <input type="number" bind:value={routerState.auctionInterval} min="1000" max="60000" step="1000" />
    </div>
  </div>

  <div class="setting-group">
    <h3>Risk Limits</h3>
    <div class="setting-row">
      <span>Max Correlated Positions</span>
      <input type="number" value="2" min="1" max="5" />
    </div>
    <div class="setting-row">
      <span>Correlation Threshold</span>
      <input type="number" value="0.7" min="0" max="1" step="0.1" />
    </div>
  </div>

  <!-- MT5 Connection -->
  <div class="setting-group mt5-connection">
    <div class="setting-header">
      <Globe size={16} />
      <h3>MT5 Connection</h3>
      <span class="connection-status" class:connected={mt5Connected}>
        <span class="status-dot"></span>
        {mt5Connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>

    <div class="mt5-form">
      <div class="form-row">
        <label>Server</label>
        <input type="text" bind:value={mt5Config.server} placeholder="e.g., ICMarkets-Live" />
      </div>
      <div class="form-row">
        <label>Port</label>
        <input type="number" bind:value={mt5Config.port} placeholder="443" />
      </div>
      <div class="form-row">
        <label>Login</label>
        <input type="text" bind:value={mt5Config.login} placeholder="Account ID" />
      </div>
      <div class="form-row">
        <label>Password</label>
        <input type="password" bind:value={mt5Config.password} placeholder="***" />
      </div>
      <div class="form-row">
        <label>Symbol Mapping</label>
        <textarea
          bind:value={mt5Config.symbolMapping}
          placeholder={mt5SymbolMappingPlaceholder}
          rows="3"
        ></textarea>
      </div>
    </div>

    <div class="mt5-actions">
      <button class="btn" on:click={testMt5Connection} disabled={mt5Testing}>
        {mt5Testing ? 'Testing...' : 'Test Connection'}
      </button>
      <button class="btn primary" on:click={saveMt5Config}>
        Save Configuration
      </button>
    </div>

    {#if mt5Error}
      <div class="mt5-error">
        <AlertCircle size={14} />
        <span>{mt5Error}</span>
      </div>
    {/if}
  </div>
</div>

<style>
  .settings-section {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .setting-group {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .setting-group h3 {
    margin: 0 0 12px;
    font-size: 14px;
    color: var(--text-primary);
  }

  .setting-options {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .radio-option {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
  }

  .radio-option:hover {
    background: var(--bg-surface);
  }

  .radio-option input {
    display: none;
  }

  .radio-option span:first-of-type {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .radio-option small {
    font-size: 11px;
    color: var(--text-muted);
  }

  .radio-option:has(input:checked) {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid var(--accent-primary);
  }

  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-subtle);
  }

  .setting-row:last-child {
    border-bottom: none;
  }

  .setting-row span {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .setting-row input {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    width: 120px;
  }

  .setting-row input:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .setting-group.mt5-connection {
    padding: 0;
  }

  .setting-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .setting-header h3 {
    margin: 0;
    flex: 1;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border-radius: 12px;
    font-size: 11px;
  }

  .connection-status.connected {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
  }

  .mt5-form {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .form-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .form-row label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .form-row input,
  .form-row textarea {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .form-row input:focus,
  .form-row textarea:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .form-row textarea {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .mt5-actions {
    display: flex;
    gap: 8px;
    padding: 0 16px 16px;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:hover {
    background: var(--bg-surface);
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .mt5-error {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: rgba(239, 68, 68, 0.1);
    border-top: 1px solid rgba(239, 68, 68, 0.2);
    color: #ef4444;
    font-size: 12px;
  }
</style>
