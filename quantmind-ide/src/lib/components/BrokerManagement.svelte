<script lang="ts">
  import { createBubbler, stopPropagation } from 'svelte/legacy';

  const bubble = createBubbler();
  import { onMount, onDestroy } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import {
    Server, Wifi, WifiOff, RefreshCw, Settings, Plus, X,
    Check, AlertTriangle, ExternalLink, Link2, Unlink, DollarSign
  } from 'lucide-svelte';

  import {
    brokerWebSocket,
    brokers,
    pendingBrokers,
    connectionState,
    connectedBrokers
  } from '../services/brokerWebSocket';
  import type { BrokerInfo } from '../services/brokerWebSocket';

  const dispatch = createEventDispatcher();

  // Add broker modal state
  let showAddModal = $state(false);
  let newBrokerType: 'mt5' | 'binance' = $state('mt5');
  let newBrokerForm = $state({
    account_id: '',
    server: '',
    api_key: '',
    api_secret: '',
    is_testnet: false
  });

  let connectionStatus = $derived($connectionState);

  function formatCurrency(value: number, currency: string = 'USD'): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(value);
  }

  function formatTime(date: Date): string {
    const now = new Date();
    const diff = now.getTime() - new Date(date).getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);

    if (seconds < 60) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    return new Date(date).toLocaleTimeString();
  }

  function getStatusColor(status: BrokerInfo['status']): string {
    switch (status) {
      case 'connected': return 'var(--color-accent-green)';
      case 'pending': return 'var(--color-accent-amber)';
      case 'disconnected': return 'var(--color-accent-red)';
      case 'error': return 'var(--color-accent-red)';
      default: return 'var(--color-text-muted)';
    }
  }

  function handleConfirm(broker: BrokerInfo) {
    brokerWebSocket.confirmBroker(broker.id);
    dispatch('brokerConfirmed', { broker });
  }

  function handleIgnore(broker: BrokerInfo) {
    brokerWebSocket.ignoreBroker(broker.id);
    dispatch('brokerIgnored', { broker });
  }

  function handleSync(broker: BrokerInfo) {
    brokerWebSocket.syncBroker(broker.id);
    dispatch('brokerSynced', { broker });
  }

  function handleDisconnect(broker: BrokerInfo) {
    brokerWebSocket.disconnectBroker(broker.id);
    dispatch('brokerDisconnected', { broker });
  }

  function handleAddBroker() {
    brokerWebSocket.addManualBroker({
      type: newBrokerType,
      account_id: newBrokerForm.account_id,
      server: newBrokerForm.server,
      is_testnet: newBrokerForm.is_testnet
    });

    // Reset form
    newBrokerForm = {
      account_id: '',
      server: '',
      api_key: '',
      api_secret: '',
      is_testnet: false
    };
    showAddModal = false;
    dispatch('brokerAdded', { type: newBrokerType });
  }

  function openSettings(broker: BrokerInfo) {
    dispatch('openSettings', { broker });
  }

  onMount(() => {
    brokerWebSocket.connect();
    brokerWebSocket.requestNotificationPermission();
    
    // Subscribe to broker events for additional UI control
    const unsubscribe = brokerWebSocket.subscribe((message) => {
      handleWebSocketMessage(message);
    });
    
    // Store unsubscribe function for cleanup
    return () => {
      unsubscribe();
    };
  });

  function handleWebSocketMessage(message: any) {
    if (!message || !message.type) return;
    
    switch (message.type) {
      case 'broker_confirmed':
        handleBrokerConfirmed(message.data);
        break;
      case 'broker_synced':
        handleBrokerSynced(message.data);
        break;
      case 'broker_disconnected':
        handleBrokerDisconnected(message.data);
        break;
      case 'broker_error':
        handleBrokerError(message.data);
        break;
    }
  }

  function handleBrokerConfirmed(data: any) {
    const brokerId = data.broker_id || data.id;
    console.log(`[BrokerManagement] Broker ${brokerId} confirmed`);
    // Refresh broker list to show updated status
    brokerWebSocket.refreshBrokers();
  }

  function handleBrokerSynced(data: any) {
    const brokerId = data.broker_id || data.id;
    console.log(`[BrokerManagement] Broker ${brokerId} synced`);
    // Refresh broker list to show updated balance/equity
    brokerWebSocket.refreshBrokers();
  }

  function handleBrokerDisconnected(data: any) {
    const brokerId = data.broker_id || data.id;
    console.warn(`[BrokerManagement] Broker ${brokerId} disconnected`);
    // Refresh broker list to show disconnected status
    brokerWebSocket.refreshBrokers();
  }

  function handleBrokerError(data: any) {
    const brokerId = data.broker_id || data.id;
    console.error(`[BrokerManagement] Broker ${brokerId} error:`, data.error);
  }

  onDestroy(() => {
    // Keep connection alive
  });
</script>

<div class="broker-management">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-left">
      <Server size={20} />
      <h3>Brokers</h3>
      <span class="connection-dot" style="background: {getStatusColor($connectionState === 'connected' ? 'connected' : 'disconnected')}"></span>
    </div>
    <div class="header-right">
      <span class="broker-count">{$connectedBrokers.length} connected</span>
      <button class="icon-btn" onclick={() => showAddModal = true} title="Add Broker">
        <Plus size={16} />
      </button>
    </div>
  </div>

  <!-- Pending Brokers (Detected but not confirmed) -->
  {#if $pendingBrokers.length > 0}
    <div class="pending-section">
      <div class="section-header">
        <AlertTriangle size={16} />
        <span>New Brokers Detected ({$pendingBrokers.length})</span>
      </div>
      <div class="pending-list">
        {#each $pendingBrokers as broker (broker.id)}
          <div class="pending-item">
            <div class="broker-info">
              <span class="broker-name">{broker.broker_name}</span>
              <span class="broker-account">{broker.account_id}</span>
            </div>
            <div class="pending-actions">
              <button class="btn-confirm" onclick={() => handleConfirm(broker)}>
                <Check size={14} />
                Confirm
              </button>
              <button class="btn-ignore" onclick={() => handleIgnore(broker)}>
                <X size={14} />
                Ignore
              </button>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Broker List -->
  <div class="broker-list">
    {#if $brokers.length === 0}
      <div class="empty-state">
        <Server size={48} />
        <p>No brokers configured</p>
        <button class="btn-primary" onclick={() => showAddModal = true}>
          <Plus size={16} />
          Add Broker
        </button>
      </div>
    {:else}
      {#each $brokers as broker (broker.id)}
        <div class="broker-card" class:disconnected={broker.status === 'disconnected'}>
          <div class="card-header">
            <div class="broker-type">
              {#if broker.type === 'mt5'}
                <span class="type-badge mt5">MT5</span>
              {:else}
                <span class="type-badge binance">Binance</span>
              {/if}
              <span class="status-dot" style="background: {getStatusColor(broker.status)}"></span>
            </div>
            <div class="card-actions">
              <button class="icon-btn small" onclick={() => handleSync(broker)} title="Sync Now">
                <RefreshCw size={14} />
              </button>
              <button class="icon-btn small" onclick={() => openSettings(broker)} title="Settings">
                <Settings size={14} />
              </button>
              {#if broker.status === 'connected'}
                <button class="icon-btn small danger" onclick={() => handleDisconnect(broker)} title="Disconnect">
                  <Unlink size={14} />
                </button>
              {:else}
                <button class="icon-btn small success" onclick={() => handleSync(broker)} title="Connect">
                  <Link2 size={14} />
                </button>
              {/if}
            </div>
          </div>

          <div class="card-body">
            <div class="broker-main-info">
              <span class="broker-name">{broker.broker_name}</span>
              <span class="broker-account">{broker.account_id}</span>
              {#if broker.server}
                <span class="broker-server">{broker.server}</span>
              {/if}
            </div>

            <div class="broker-metrics">
              <div class="metric">
                <span class="metric-label">Balance</span>
                <span class="metric-value">{formatCurrency(broker.balance, broker.currency)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Equity</span>
                <span class="metric-value" class:positive={broker.equity > broker.balance} class:negative={broker.equity < broker.balance}>
                  {formatCurrency(broker.equity, broker.currency)}
                </span>
              </div>
              <div class="metric">
                <span class="metric-label">Margin</span>
                <span class="metric-value">{formatCurrency(broker.margin, broker.currency)}</span>
              </div>
              {#if broker.leverage}
                <div class="metric">
                  <span class="metric-label">Leverage</span>
                  <span class="metric-value">1:{broker.leverage}</span>
                </div>
              {/if}
            </div>
          </div>

          <div class="card-footer">
            <span class="last-seen">Last ping: {formatTime(broker.last_seen)}</span>
            {#if broker.is_testnet}
              <span class="testnet-badge">Testnet</span>
            {/if}
          </div>
        </div>
      {/each}
    {/if}
  </div>

  <!-- Add Broker Modal -->
  {#if showAddModal}
    <div class="modal-overlay" onclick={() => showAddModal = false}>
      <div class="modal" onclick={stopPropagation(bubble('click'))}>
        <div class="modal-header">
          <h4>Add Broker</h4>
          <button class="icon-btn" onclick={() => showAddModal = false}>
            <X size={16} />
          </button>
        </div>

        <div class="modal-body">
          <div class="type-selector">
            <button class:active={newBrokerType === 'mt5'} onclick={() => newBrokerType = 'mt5'}>
              MetaTrader 5
            </button>
            <button class:active={newBrokerType === 'binance'} onclick={() => newBrokerType = 'binance'}>
              Binance
            </button>
          </div>

          {#if newBrokerType === 'mt5'}
            <div class="form-group">
              <label>Account ID</label>
              <input type="text" bind:value={newBrokerForm.account_id} placeholder="Enter account number">
            </div>
            <div class="form-group">
              <label>Server</label>
              <input type="text" bind:value={newBrokerForm.server} placeholder="e.g., ICMarkets-Demo">
            </div>
          {:else}
            <div class="form-group">
              <label>API Key</label>
              <input type="text" bind:value={newBrokerForm.api_key} placeholder="Enter API key">
            </div>
            <div class="form-group">
              <label>API Secret</label>
              <input type="password" bind:value={newBrokerForm.api_secret} placeholder="Enter API secret">
            </div>
            <div class="form-group">
              <label class="checkbox-label">
                <input type="checkbox" bind:checked={newBrokerForm.is_testnet}>
                <span>Use Testnet</span>
              </label>
            </div>
          {/if}
        </div>

        <div class="modal-footer">
          <button class="btn-secondary" onclick={() => showAddModal = false}>Cancel</button>
          <button class="btn-primary" onclick={handleAddBroker}>
            <Plus size={14} />
            Add Broker
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .broker-management {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--color-bg-base);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--color-bg-surface);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-left h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .connection-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .broker-count {
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .icon-btn:hover {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }

  .icon-btn.small {
    width: 24px;
    height: 24px;
  }

  .icon-btn.danger:hover {
    background: rgba(244, 67, 54, 0.1);
    border-color: var(--color-accent-red);
    color: var(--color-accent-red);
  }

  .icon-btn.success:hover {
    background: rgba(76, 175, 80, 0.1);
    border-color: var(--color-accent-green);
    color: var(--color-accent-green);
  }

  .pending-section {
    padding: 12px;
    background: rgba(255, 152, 0, 0.1);
    border-bottom: 1px solid var(--color-accent-amber);
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--color-accent-amber);
  }

  .pending-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .pending-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: var(--color-bg-base);
    border-radius: 6px;
  }

  .pending-actions {
    display: flex;
    gap: 8px;
  }

  .btn-confirm, .btn-ignore {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-confirm {
    background: var(--color-accent-green);
    border: none;
    color: #000;
  }

  .btn-ignore {
    background: transparent;
    border: 1px solid var(--color-border-subtle);
    color: var(--color-text-secondary);
  }

  .broker-list {
    flex: 1;
    padding: 12px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 40px;
    color: var(--color-text-muted);
  }

  .btn-primary {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 16px;
    background: var(--color-accent-cyan);
    border: none;
    border-radius: 6px;
    color: #000;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-primary:hover {
    background: var(--color-accent-amber);
  }

  .btn-secondary {
    padding: 10px 16px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 13px;
    cursor: pointer;
  }

  .broker-card {
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.2s ease;
  }

  .broker-card:hover {
    border-color: var(--border-default);
  }

  .broker-card.disconnected {
    opacity: 0.6;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: var(--color-bg-elevated);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .broker-type {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .type-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .type-badge.mt5 {
    background: #2196f3;
    color: #fff;
  }

  .type-badge.binance {
    background: #f0b90b;
    color: #000;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .card-actions {
    display: flex;
    gap: 4px;
  }

  .card-body {
    padding: 12px;
  }

  .broker-main-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-bottom: 12px;
  }

  .broker-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .broker-account {
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .broker-server {
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .broker-metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .metric-label {
    font-size: 10px;
    color: var(--color-text-muted);
    text-transform: uppercase;
  }

  .metric-value {
    font-size: 13px;
    font-weight: 500;
    color: var(--color-text-primary);
  }

  .metric-value.positive {
    color: var(--color-accent-green);
  }

  .metric-value.negative {
    color: var(--color-accent-red);
  }

  .card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .testnet-badge {
    padding: 2px 6px;
    background: var(--color-accent-amber);
    color: #000;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
  }

  /* Modal Styles */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    width: 400px;
    background: var(--color-bg-base);
    border: 1px solid var(--color-border-subtle);
    border-radius: 12px;
    overflow: hidden;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .modal-header h4 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .modal-body {
    padding: 16px;
  }

  .type-selector {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
  }

  .type-selector button {
    flex: 1;
    padding: 10px;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .type-selector button.active {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: #000;
  }

  .form-group {
    margin-bottom: 12px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--color-text-secondary);
  }

  .form-group input[type="text"],
  .form-group input[type="password"] {
    width: 100%;
    padding: 10px 12px;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 13px;
  }

  .form-group input:focus {
    outline: none;
    border-color: var(--color-accent-cyan);
  }

  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
  }

  .checkbox-label input {
    accent-color: var(--color-accent-cyan);
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px;
    border-top: 1px solid var(--color-border-subtle);
  }
</style>
