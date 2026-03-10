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
  /* Panel Header */
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Info Box */
  .info-box {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px 16px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 8px;
    margin-bottom: 20px;
    color: #60a5fa;
    font-size: 13px;
    line-height: 1.5;
  }

  .info-box :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
  }

  /* Keys List */
  .keys-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  /* Key Item */
  .key-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    transition: all 0.15s ease;
  }

  .key-item:hover {
    background: var(--bg-surface);
    border-color: var(--accent-primary);
    transform: translateX(2px);
  }

  .key-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: rgba(99, 102, 241, 0.15);
    border-radius: 8px;
    color: var(--accent-primary);
  }

  .key-info {
    flex: 1;
    min-width: 0;
  }

  .key-name {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 2px;
  }

  .key-service {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: capitalize;
  }

  .key-value {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: var(--bg-primary);
    border-radius: 6px;
    font-family: monospace;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .key-value code {
    color: var(--text-muted);
  }

  .key-actions {
    display: flex;
    gap: 4px;
  }

  /* Icon Button */
  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
  }

  .icon-btn.danger:hover {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
  }

  /* Copy button */
  .key-item .copy-btn {
    opacity: 0;
    transition: opacity 0.15s;
  }

  .key-item:hover .copy-btn {
    opacity: 1;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
    text-align: center;
    color: var(--text-muted);
  }

  .empty-state :global(svg) {
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .empty-state p {
    margin: 0 0 16px;
    font-size: 14px;
  }

  /* Modal */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 450px;
    max-width: 95vw;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
    overflow: hidden;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .modal-header button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .modal-header button:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .modal-body {
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .form-group label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .form-group input,
  .form-group select {
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 14px;
    transition: all 0.15s;
  }

  .form-group input:focus,
  .form-group select:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
  }

  .form-group input::placeholder {
    color: var(--text-muted);
    opacity: 0.6;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-subtle);
  }

  /* Buttons */
  .btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    border: none;
  }

  .btn.primary {
    background: var(--accent-primary);
    color: white;
  }

  .btn.primary:hover {
    opacity: 0.9;
    transform: translateY(-1px);
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .btn.secondary:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }

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
</style>
