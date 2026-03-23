<script lang="ts">
  import { Globe, AlertCircle, Settings2, Shield, Radio, Timer } from 'lucide-svelte';



  interface Props {
    routerState: {
    mode: 'auction' | 'priority' | 'round-robin';
    auctionInterval: number;
  };
    mt5Connected: boolean;
    mt5Testing: boolean;
    mt5Error: string;
    mt5Config: {
    server: string;
    port: number;
    login: string;
    password: string;
    symbolMapping: string;
  };
    testMt5Connection: () => Promise<void>;
    saveMt5Config: () => Promise<void>;
  }

  let {
    routerState = $bindable(),
    mt5Connected,
    mt5Testing,
    mt5Error,
    mt5Config = $bindable(),
    testMt5Connection,
    saveMt5Config
  }: Props = $props();

  let mt5SymbolMappingPlaceholder = '{"EURUSDm": "EURUSD", "GBPUSDm": "GBPUSD", "USDJPYm": "USDJPY"}';
</script>

<div class="settings-section">
  <!-- Router Mode Section -->
  <div class="setting-group">
    <div class="section-header">
      <div class="section-icon">
        <Radio size={16} />
      </div>
      <h3>Router Mode</h3>
    </div>
    <div class="setting-options">
      <label class="radio-card" class:selected={routerState.mode === 'auction'}>
        <input type="radio" bind:group={routerState.mode} value="auction" />
        <div class="radio-content">
          <span class="radio-title">Auction <span class="badge recommended">Recommended</span></span>
          <span class="radio-description">Best signal wins based on scoring</span>
        </div>
        <div class="radio-indicator"></div>
      </label>
      <label class="radio-card" class:selected={routerState.mode === 'priority'}>
        <input type="radio" bind:group={routerState.mode} value="priority" />
        <div class="radio-content">
          <span class="radio-title">Priority</span>
          <span class="radio-description">Higher priority bots go first</span>
        </div>
        <div class="radio-indicator"></div>
      </label>
      <label class="radio-card" class:selected={routerState.mode === 'round-robin'}>
        <input type="radio" bind:group={routerState.mode} value="round-robin" />
        <div class="radio-content">
          <span class="radio-title">Round Robin</span>
          <span class="radio-description">Equal opportunity for all bots</span>
        </div>
        <div class="radio-indicator"></div>
      </label>
    </div>
  </div>

  <!-- Auction Settings Section -->
  <div class="setting-group">
    <div class="section-header">
      <div class="section-icon">
        <Timer size={16} />
      </div>
      <h3>Auction Settings</h3>
    </div>
    <div class="setting-row">
      <div class="setting-label">
        <span>Interval (ms)</span>
        <span class="setting-hint">Time between auction cycles</span>
      </div>
      <div class="setting-control">
        <input
          type="number"
          bind:value={routerState.auctionInterval}
          min="1000"
          max="60000"
          step="1000"
        />
      </div>
    </div>
  </div>

  <!-- Risk Limits Section -->
  <div class="setting-group">
    <div class="section-header">
      <div class="section-icon">
        <Shield size={16} />
      </div>
      <h3>Risk Limits</h3>
    </div>
    <div class="setting-row">
      <div class="setting-label">
        <span>Max Correlated Positions</span>
        <span class="setting-hint">Maximum correlated trades per symbol</span>
      </div>
      <div class="setting-control">
        <input type="number" value="2" min="1" max="5" />
      </div>
    </div>
    <div class="setting-row">
      <div class="setting-label">
        <span>Correlation Threshold</span>
        <span class="setting-hint">Minimum correlation to flag as related</span>
      </div>
      <div class="setting-control">
        <input type="number" value="0.7" min="0" max="1" step="0.1" />
      </div>
    </div>
  </div>

  <!-- MT5 Connection Section -->
  <div class="setting-group mt5-connection">
    <div class="section-header mt5-header">
      <div class="section-icon">
        <Globe size={16} />
      </div>
      <h3>MT5 Connection</h3>
      <span class="connection-status" class:connected={mt5Connected}>
        <span class="status-dot"></span>
        {mt5Connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>

    <div class="mt5-form">
      <div class="form-grid">
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
          <input type="password" bind:value={mt5Config.password} placeholder="Enter password" />
        </div>
      </div>
      <div class="form-row full-width">
        <label>Symbol Mapping</label>
        <textarea
          bind:value={mt5Config.symbolMapping}
          placeholder={mt5SymbolMappingPlaceholder}
          rows="3"
        ></textarea>
      </div>
    </div>

    <div class="mt5-actions">
      <button class="btn" onclick={testMt5Connection} disabled={mt5Testing}>
        <Settings2 size={14} />
        {mt5Testing ? 'Testing...' : 'Test Connection'}
      </button>
      <button class="btn primary" onclick={saveMt5Config}>
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
    gap: 20px;
  }

  .setting-group {
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 12px;
    padding: 0;
    overflow: hidden;
  }

  /* Section Header Styles */
  .section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--color-border-subtle);
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, transparent 100%);
  }

  .section-header.mt5-header {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, transparent 100%);
  }

  .section-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: rgba(99, 102, 241, 0.15);
    border-radius: 6px;
    color: var(--color-accent-cyan);
  }

  .section-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
    flex: 1;
  }

  /* Radio Card Options */
  .setting-options {
    display: flex;
    flex-direction: column;
    padding: 12px;
    gap: 8px;
  }

  .radio-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    background: var(--color-bg-elevated);
    border: 1px solid transparent;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .radio-card:hover {
    background: var(--bg-surface);
    border-color: var(--color-border-subtle);
    transform: translateX(2px);
  }

  .radio-card.selected {
    background: rgba(99, 102, 241, 0.08);
    border-color: var(--color-accent-cyan);
  }

  .radio-card input {
    display: none;
  }

  .radio-content {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
  }

  .radio-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--color-text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .badge.recommended {
    font-size: 9px;
    font-weight: 600;
    padding: 2px 6px;
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .radio-description {
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .radio-indicator {
    width: 16px;
    height: 16px;
    border: 2px solid var(--color-border-subtle);
    border-radius: 50%;
    position: relative;
    transition: all 0.2s ease;
  }

  .radio-card.selected .radio-indicator {
    border-color: var(--color-accent-cyan);
    background: var(--color-accent-cyan);
  }

  .radio-card.selected .radio-indicator::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 6px;
    height: 6px;
    background: white;
    border-radius: 50%;
  }

  /* Setting Row */
  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .setting-row:last-of-type {
    border-bottom: none;
  }

  .setting-label {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .setting-label > span:first-child {
    font-size: 13px;
    font-weight: 500;
    color: var(--color-text-primary);
  }

  .setting-hint {
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .setting-control {
    display: flex;
    align-items: center;
  }

  .setting-row input {
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 13px;
    width: 100px;
    text-align: right;
    transition: all 0.15s ease;
  }

  .setting-row input:hover {
    border-color: var(--color-text-muted);
  }

  .setting-row input:focus {
    outline: none;
    border-color: var(--color-accent-cyan);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }

  /* MT5 Connection Styles */
  .setting-group.mt5-connection {
    padding: 0;
  }

  .mt5-header {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, transparent 100%);
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
  }

  .connection-status.connected {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .mt5-form {
    padding: 16px;
  }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 12px;
  }

  .form-row {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .form-row.full-width {
    grid-column: 1 / -1;
  }

  .form-row label {
    font-size: 11px;
    font-weight: 500;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .form-row input,
  .form-row textarea {
    padding: 10px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 13px;
    transition: all 0.15s ease;
  }

  .form-row input:hover,
  .form-row textarea:hover {
    border-color: var(--color-text-muted);
  }

  .form-row input:focus,
  .form-row textarea:focus {
    outline: none;
    border-color: var(--color-accent-cyan);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }

  .form-row input::placeholder,
  .form-row textarea::placeholder {
    color: var(--color-text-muted);
    opacity: 0.6;
  }

  .form-row textarea {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    resize: vertical;
    min-height: 70px;
  }

  .mt5-actions {
    display: flex;
    gap: 10px;
    padding: 0 16px 16px;
  }

  .btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 16px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-secondary);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn:hover:not(:disabled) {
    background: var(--bg-surface);
    border-color: var(--color-text-muted);
    transform: translateY(-1px);
  }

  .btn:active:not(:disabled) {
    transform: translateY(0);
  }

  .btn.primary {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: white;
    flex: 1;
  }

  .btn.primary:hover:not(:disabled) {
    background: #5558e6;
    border-color: #5558e6;
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
    animation: slideIn 0.2s ease;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-4px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
