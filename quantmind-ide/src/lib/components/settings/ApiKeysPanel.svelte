<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import {
    Key, RefreshCw, Trash2, AlertCircle, Plus,
    Brain, Sparkles, Zap, Globe, Server, Cpu,
    Copy, Eye, EyeOff
  } from 'lucide-svelte';

  export let apiKeys: Array<{
    id: string;
    name: string;
    key: string;
    service: string;
    created: string;
    lastUsed?: string;
  }> = [];

  export let apiKeyModal = false;
  export let newApiKey = {
    name: '',
    key: '',
    service: 'openai'
  };

  const dispatch = createEventDispatcher();

  // Generate secure random API key
  function generateSecureKey(): string {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    const hexArray = Array.from(array).map(b => b.toString(16).padStart(2, '0'));
    return 'qm_' + hexArray.join('');
  }

  // Show/hide generated key
  let showGeneratedKey = false;

  function handleGenerateKey() {
    newApiKey.key = generateSecureKey();
    showGeneratedKey = true;
  }

  // Copy to clipboard
  async function copyToClipboard(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      dispatch('copySuccess', { text });
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }

  // Copy key from list
  async function copyKeyValue(key: string) {
    await copyToClipboard(key);
  }

  function getServiceIcon(service: string) {
    const icons: Record<string, typeof Brain> = {
      openai: Brain,
      anthropic: Sparkles,
      gemini: Zap,
      openrouter: Globe,
      together: Server,
      groq: Cpu
    };
    return icons[service] || Key;
  }

  function testConnection(service: string) {
    console.log(`Testing connection to ${service}...`);
    dispatch('testConnection', { service });
  }

  function addApiKey() {
    dispatch('addApiKey');
  }

  function removeApiKey(id: string) {
    dispatch('removeApiKey', { id });
  }

  function openModal() {
    apiKeyModal = true;
    dispatch('openModal');
  }

  function closeModal() {
    apiKeyModal = false;
    dispatch('closeModal');
  }
</script>

<div class="panel">
  <div class="panel-header">
    <h3>API Keys</h3>
    <button class="btn primary" on:click={openModal}>
      <Plus size={14} /> Add Key
    </button>
  </div>

  <div class="info-box">
    <AlertCircle size={16} />
    <span>Your API keys are stored locally and encrypted. Never share them with anyone.</span>
  </div>

  <div class="keys-list">
    {#each apiKeys as key}
      <div class="key-item">
        <div class="key-icon">
          <svelte:component this={getServiceIcon(key.service)} />
        </div>
        <div class="key-info">
          <div class="key-name">{key.name}</div>
          <div class="key-service">{key.service}</div>
        </div>
        <div class="key-value">
          <code>{key.key.slice(0, 8)}...</code>
          <button class="icon-btn copy-btn" on:click={() => copyKeyValue(key.key)} title="Copy Key">
            <Copy size={14} />
          </button>
        </div>
        <div class="key-actions">
          <button class="icon-btn" on:click={() => testConnection(key.service)} title="Test Connection">
            <RefreshCw size={14} />
          </button>
          <button class="icon-btn danger" on:click={() => removeApiKey(key.id)} title="Remove">
            <Trash2 size={14} />
          </button>
        </div>
      </div>
    {:else}
      <div class="empty-state">
        <Key size={32} />
        <p>No API keys configured</p>
        <button class="btn primary" on:click={openModal}>
          Add Your First API Key
        </button>
      </div>
    {/each}
  </div>
</div>

<!-- API Key Modal -->
{#if apiKeyModal}
  <div class="modal-overlay" on:click|self={closeModal}>
    <div class="modal">
      <div class="modal-header">
        <h3>Add API Key</h3>
        <button on:click={closeModal}><RefreshCw size={20} /></button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label>Name</label>
          <input type="text" placeholder="My OpenAI Key" bind:value={newApiKey.name} />
        </div>
        <div class="form-group">
          <label>Service</label>
          <select bind:value={newApiKey.service}>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="gemini">Google Gemini</option>
            <option value="openrouter">OpenRouter</option>
            <option value="together">Together AI</option>
            <option value="groq">Groq</option>
          </select>
        </div>
        <div class="form-group">
          <label>API Key</label>
          <div class="key-input-wrapper">
            {#if showGeneratedKey}
              <div class="generated-key-display">
                <code class="key-value">{newApiKey.key}</code>
                <button class="icon-btn" on:click={() => copyToClipboard(newApiKey.key)} title="Copy to clipboard">
                  <Copy size={14} />
                </button>
                <button class="icon-btn" on:click={() => showGeneratedKey = false} title="Hide">
                  <EyeOff size={14} />
                </button>
              </div>
            {:else}
              <input type="password" placeholder="sk-..." bind:value={newApiKey.key} />
            {/if}
          </div>
          <div class="generate-row">
            <button type="button" class="btn secondary generate-btn" on:click={handleGenerateKey}>
              <Key size={14} /> Generate Secure Key
            </button>
            {#if newApiKey.key && !showGeneratedKey}
              <button type="button" class="btn-text" on:click={() => showGeneratedKey = true}>
                <Eye size={12} /> Show
              </button>
            {/if}
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" on:click={closeModal}>Cancel</button>
        <button class="btn primary" on:click={addApiKey}>Add Key</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .key-input-wrapper {
    display: flex;
    align-items: center;
  }

  .generated-key-display {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    width: 100%;
  }

  .key-value {
    flex: 1;
    font-family: monospace;
    font-size: 12px;
    color: var(--text-primary);
    word-break: break-all;
  }

  .generate-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 8px;
  }

  .generate-btn {
    font-size: 12px;
    padding: 6px 12px;
  }

  .btn-text {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: none;
    border: none;
    color: var(--accent-primary);
    font-size: 12px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    transition: background 0.15s;
  }

  .btn-text:hover {
    background: var(--bg-tertiary);
  }

  .key-item .copy-btn {
    opacity: 0;
    transition: opacity 0.15s;
  }

  .key-item:hover .copy-btn {
    opacity: 1;
  }
</style>
