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
