<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let connectionSettings = {
    redisUrl: 'redis://localhost:6379',
    zmqEndpoint: 'tcp://localhost:5555',
    mt5Login: '',
    mt5Password: '',
    mt5Server: ''
  };

  const dispatch = createEventDispatcher();

  function updateConnectionSettings(field: string, value: any) {
    dispatch('updateConnectionSettings', { field, value });
  }
</script>

<div class="panel">
  <h3>Connection Settings</h3>

  <div class="setting-group">
    <label>Redis URL</label>
    <input
      type="text"
      bind:value={connectionSettings.redisUrl}
      class="text-input"
      on:input={(e) => updateConnectionSettings('redisUrl', e.currentTarget.value)}
    />
  </div>

  <div class="setting-group">
    <label>ZMQ Endpoint</label>
    <input
      type="text"
      bind:value={connectionSettings.zmqEndpoint}
      class="text-input"
      on:input={(e) => updateConnectionSettings('zmqEndpoint', e.currentTarget.value)}
    />
  </div>

  <h4>MetaTrader 5</h4>
  <div class="setting-group">
    <label>Login</label>
    <input
      type="number"
      bind:value={connectionSettings.mt5Login}
      class="text-input"
      on:input={(e) => updateConnectionSettings('mt5Login', e.currentTarget.value)}
    />
  </div>
  <div class="setting-group">
    <label>Password</label>
    <input
      type="password"
      bind:value={connectionSettings.mt5Password}
      class="text-input"
      on:input={(e) => updateConnectionSettings('mt5Password', e.currentTarget.value)}
    />
  </div>
  <div class="setting-group">
    <label>Server</label>
    <input
      type="text"
      bind:value={connectionSettings.mt5Server}
      class="text-input"
      on:input={(e) => updateConnectionSettings('mt5Server', e.currentTarget.value)}
    />
  </div>
</div>

<style>
  /* Panel Header */
  .panel h3 {
    margin: 0 0 20px;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Section Headers */
  h4 {
    margin: 24px 0 12px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }

  h4:first-of-type {
    margin-top: 0;
  }

  /* Setting Group */
  .setting-group {
    margin-bottom: 16px;
  }

  .setting-group:last-child {
    margin-bottom: 0;
  }

  .setting-group label {
    display: block;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  /* Text Input */
  .text-input {
    width: 100%;
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 14px;
    transition: all 0.15s;
    box-sizing: border-box;
  }

  .text-input:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
  }

  .text-input::placeholder {
    color: var(--text-muted);
    opacity: 0.6;
  }

  /* Input types */
  input[type="number"] {
    -moz-appearance: textfield;
  }

  input[type="number"]::-webkit-outer-spin-button,
  input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
</style>
