<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Server, Key, Wallet, Check, X, Loader, Eye, EyeOff } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  export let show = false;

  // Declare brokers first before using it
  const brokers = [
    {
      id: 'metaquotes',
      name: 'MetaQuotes (MT5)',
      fields: ['account', 'password', 'server'],
      description: 'Connect to MetaTrader 5 terminal'
    },
    {
      id: 'ctrader',
      name: 'cTrader',
      fields: ['account', 'token'],
      description: 'Connect to cTrader platform'
    },
    {
      id: 'binance',
      name: 'Binance',
      fields: ['apiKey', 'apiSecret'],
      description: 'Connect to Binance exchange'
    },
    {
      id: 'bybit',
      name: 'Bybit',
      fields: ['apiKey', 'apiSecret'],
      description: 'Connect to Bybit exchange'
    }
  ];

  let selectedBroker = brokers[0];
  let accountType = 'demo';
  let isConnecting = false;
  let connectionStatus: 'idle' | 'connecting' | 'success' | 'error' = 'idle';
  let showPassword = false;

  // Form fields
  let credentials = {
    account: '',
    password: '',
    server: '',
    apiKey: '',
    apiSecret: '',
    token: ''
  };

  function hasField(field: string): boolean {
    return selectedBroker.fields.includes(field);
  }

  async function testConnection() {
    isConnecting = true;
    connectionStatus = 'connecting';

    try {
      const res = await fetch('http://localhost:8000/api/trading/broker/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          broker: selectedBroker.id,
          credentials: { ...credentials, type: accountType }
        })
      });

      if (res.ok) {
        connectionStatus = 'success';
        setTimeout(() => {
          dispatch('testSuccess', { broker: selectedBroker.id });
        }, 1000);
      } else {
        connectionStatus = 'error';
      }
    } catch (e) {
      connectionStatus = 'error';
      console.error('Connection test failed:', e);
    } finally {
      isConnecting = false;
    }
  }

  async function connect() {
    isConnecting = true;

    try {
      const res = await fetch('http://localhost:8000/api/trading/broker/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          broker: selectedBroker.id,
          credentials: { ...credentials, type: accountType }
        })
      });

      if (res.ok) {
        const data = await res.json();
        dispatch('connected', { broker: selectedBroker.id, data });
        close();
      }
    } catch (e) {
      console.error('Connection failed:', e);
    } finally {
      isConnecting = false;
    }
  }

  function close() {
    show = false;
    connectionStatus = 'idle';
    credentials = {
      account: '',
      password: '',
      server: '',
      apiKey: '',
      apiSecret: '',
      token: ''
    };
  }
</script>

{#if show}
<div class="modal-overlay" on:click={close}>
  <div class="modal broker-connect-modal" on:click|stopPropagation>
    <div class="modal-header">
      <div>
        <h2>Connect Broker</h2>
        <p>Connect your trading account to enable automated trading</p>
      </div>
      <button class="icon-btn" on:click={close}><X size={20} /></button>
    </div>

    <div class="modal-body">
      <!-- Broker Selection -->
      <div class="form-group">
        <label>Select Broker</label>
        <select bind:value={selectedBroker}>
          {#each brokers as broker}
            <option value={broker}>{broker.name}</option>
          {/each}
        </select>
        <p class="hint">{selectedBroker.description}</p>
      </div>

      <!-- Account Type -->
      <div class="form-group">
        <label>Account Type</label>
        <div class="radio-group">
          <label>
            <input type="radio" bind:group={accountType} value="demo" />
            <span>Demo Account</span>
          </label>
          <label>
            <input type="radio" bind:group={accountType} value="live" />
            <span>Live Account</span>
          </label>
        </div>
      </div>

      <!-- Credentials Form -->
      <div class="credentials-form">
        {#if hasField('account')}
          <div class="form-group">
            <label>Account Number</label>
            <input type="text" bind:value={credentials.account} placeholder="12345678" />
          </div>
        {/if}

        {#if hasField('password')}
          <div class="form-group">
            <label>Password</label>
            <div class="password-input">
              {#if showPassword}
                <input
                  type="text"
                  bind:value={credentials.password}
                  placeholder="Your password"
                />
              {:else}
                <input
                  type="password"
                  bind:value={credentials.password}
                  placeholder="Your password"
                />
              {/if}
              <button class="icon-btn" on:click={() => showPassword = !showPassword}>
                {#if showPassword}<EyeOff size={16} />{:else}<Eye size={16} />{/if}
              </button>
            </div>
          </div>
        {/if}

        {#if hasField('server')}
          <div class="form-group">
            <label>Server</label>
            <input type="text" bind:value={credentials.server} placeholder="Broker-Server-Demo" />
          </div>
        {/if}

        {#if hasField('apiKey')}
          <div class="form-group">
            <label>API Key</label>
            <input type="text" bind:value={credentials.apiKey} placeholder="Your API key" />
          </div>
        {/if}

        {#if hasField('apiSecret')}
          <div class="form-group">
            <label>API Secret</label>
            <input type="password" bind:value={credentials.apiSecret} placeholder="Your API secret" />
          </div>
        {/if}

        {#if hasField('token')}
          <div class="form-group">
            <label>Access Token</label>
            <input type="password" bind:value={credentials.token} placeholder="Your access token" />
          </div>
        {/if}
      </div>

      <!-- Connection Status -->
      {#if connectionStatus !== 'idle'}
        <div class="connection-status" class:{connectionStatus}>
          {#if connectionStatus === 'connecting'}
            <Loader size={16} class="spinning" />
            <span>Testing connection...</span>
          {:else if connectionStatus === 'success'}
            <Check size={16} />
            <span>Connection successful!</span>
          {:else if connectionStatus === 'error'}
            <X size={16} />
            <span>Connection failed. Please check your credentials.</span>
          {/if}
        </div>
      {/if}
    </div>

    <div class="modal-footer">
      <button class="btn secondary" on:click={close}>Cancel</button>
      <button class="btn secondary" on:click={testConnection} disabled={isConnecting}>
        {#if connectionStatus === 'connecting'}
          <Loader size={14} class="spinning" />
        {:else}
          Test Connection
        {/if}
      </button>
      <button class="btn primary" on:click={connect} disabled={isConnecting || connectionStatus !== 'idle'}>
        {#if isConnecting}
          <Loader size={14} class="spinning" />
          Connecting...
        {:else}
          Connect
        {/if}
      </button>
    </div>
  </div>
</div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .broker-connect-modal {
    width: 500px;
    max-width: 90vw;
    max-height: 90vh;
    overflow-y: auto;
    background: var(--bg-primary);
    border-radius: 12px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h2 {
    margin: 0 0 4px 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .modal-header p {
    margin: 0;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .modal-body {
    padding: 24px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .form-group input,
  .form-group select {
    width: 100%;
    padding: 10px 12px;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 14px;
    transition: border-color 0.15s ease;
  }

  .form-group input:focus,
  .form-group select:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }

  .hint {
    margin-top: 4px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .radio-group {
    display: flex;
    gap: 16px;
  }

  .radio-group label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    cursor: pointer;
  }

  .radio-group input[type="radio"] {
    width: auto;
    cursor: pointer;
  }

  .password-input {
    display: flex;
    gap: 8px;
  }

  .password-input input {
    flex: 1;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    border-radius: 6px;
    font-size: 14px;
  }

  .connection-status.connecting {
    background: rgba(59, 130, 246, 0.1);
    color: #3b82f6;
  }

  .connection-status.success {
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
  }

  .connection-status.error {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .modal-footer {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
    padding: 16px 24px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
    border-radius: 0 0 12px 12px;
  }

  .btn {
    padding: 10px 16px;
    border-radius: 6px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.secondary {
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border-subtle);
  }

  .btn.secondary:hover:not(:disabled) {
    background: var(--bg-tertiary);
  }

  .btn.primary {
    background: var(--accent-primary);
    color: white;
  }

  .btn.primary:hover:not(:disabled) {
    opacity: 0.9;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 4px;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
</style>
