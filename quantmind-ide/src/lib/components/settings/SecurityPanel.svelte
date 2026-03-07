<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { RefreshCw, Lock } from 'lucide-svelte';

  export let securitySettings = {
    secretKeyConfigured: false,
    secretKeyPrefix: ''
  };

  const dispatch = createEventDispatcher();

  function generateNewKey() {
    dispatch('generateNewKey');
  }
</script>

<div class="panel">
  <h3>Security</h3>

  <div class="setting-group">
    <label>Secret Key Status</label>
    {#if securitySettings.secretKeyConfigured}
      <span class="status-badge success">Configured</span>
      <p class="hint">Key starts with: {securitySettings.secretKeyPrefix}***</p>
    {:else}
      <span class="status-badge warning">Not Configured</span>
      <p class="hint">Set SECRET_KEY environment variable</p>
    {/if}
  </div>

  <div class="setting-group">
    <button class="btn btn-secondary" on:click={generateNewKey}>
      <RefreshCw size={16} />
      Generate New Key
    </button>
  </div>
</div>
