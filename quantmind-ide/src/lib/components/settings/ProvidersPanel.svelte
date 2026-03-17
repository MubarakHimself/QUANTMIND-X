<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import {
    Key, RefreshCw, Trash2, AlertCircle, Plus,
    Eye, EyeOff, Save, Server, Check, X
  } from 'lucide-svelte';

  export let providersData: Record<string, {
    name: string;
    base_url: string;
    api_key?: string;
  }> = {};

  const dispatch = createEventDispatcher();

  const DEFAULT_PROVIDERS = [
    { id: 'anthropic', name: 'Anthropic (Claude)', base_url: 'https://api.anthropic.com/v1' },
    { id: 'zhipu', name: 'Zhipu (GLM)', base_url: 'https://open.bigmodel.cn/api/paas/v4' },
    { id: 'minimax', name: 'MiniMax', base_url: 'https://api.minimax.chat/v1' },
    { id: 'deepseek', name: 'DeepSeek', base_url: 'https://api.deepseek.com/v1' },
    { id: 'openai', name: 'OpenAI', base_url: 'https://api.openai.com/v1' },
    { id: 'openrouter', name: 'OpenRouter', base_url: 'https://openrouter.ai/api/v1' },
  ];

  // Initialize providers with data from props
  let providers = DEFAULT_PROVIDERS.map(p => ({
    ...p,
    api_key: providersData[p.id]?.api_key || '',
    base_url: providersData[p.id]?.base_url || p.base_url
  }));

  // Track show/hide state for API keys
  let showApiKey: Record<string, boolean> = {};
  let editingProvider: string | null = null;
  let isSaving = false;
  let isLoading = false;

  function isConfigured(providerId: string): boolean {
    const provider = providers.find(p => p.id === providerId);
    return !!(provider?.api_key && provider.api_key.length > 0);
  }

  function toggleShowApiKey(providerId: string) {
    showApiKey[providerId] = !showApiKey[providerId];
  }

  function startEditing(providerId: string) {
    editingProvider = providerId;
  }

  function cancelEditing(providerId: string) {
    const defaultProvider = DEFAULT_PROVIDERS.find(p => p.id === providerId);
    const current = providers.find(p => p.id === providerId);
    if (defaultProvider && current) {
      current.base_url = providersData[providerId]?.base_url || defaultProvider.base_url;
      current.api_key = providersData[providerId]?.api_key || '';
    }
    editingProvider = null;
  }

  async function saveProvider(providerId: string) {
    isSaving = true;
    const provider = providers.find(p => p.id === providerId);
    if (!provider) return;

    try {
      const response = await fetch('http://localhost:8000/api/providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: providerId,
          base_url: provider.base_url,
          api_key: provider.api_key,
          enabled: true
        })
      });

      if (response.ok) {
        dispatch('saveProvider', { providerId });
        editingProvider = null;
      }
    } catch (e) {
      console.error('Failed to save provider:', e);
    } finally {
      isSaving = false;
    }
  }

  async function deleteProvider(providerId: string) {
    try {
      const response = await fetch(`http://localhost:8000/api/providers/${providerId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        const provider = providers.find(p => p.id === providerId);
        if (provider) {
          provider.api_key = '';
        }
        dispatch('deleteProvider', { providerId });
      }
    } catch (e) {
      console.error('Failed to delete provider:', e);
    }
  }

  async function loadProviders() {
    isLoading = true;
    try {
      const response = await fetch('http://localhost:8000/api/providers');
      if (response.ok) {
        const data = await response.json();
        providers = DEFAULT_PROVIDERS.map(p => ({
          ...p,
          api_key: data[p.id]?.api_key || '',
          base_url: data[p.id]?.base_url || p.base_url
        }));
      }
    } catch (e) {
      console.error('Failed to load providers:', e);
    } finally {
      isLoading = false;
    }
  }

  // Load providers on mount
  import { onMount } from 'svelte';
  onMount(() => {
    loadProviders();
  });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Providers</h3>
    <div class="header-actions">
      <button class="icon-btn" on:click={loadProviders} title="Refresh">
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
    </div>
  </div>

  <div class="info-box">
    <AlertCircle size={16} />
    <span>Configure API providers for AI agents. Each provider requires an API key for authentication.</span>
  </div>

  <div class="providers-list">
    {#each providers as provider}
      <div class="provider-card" class:editing={editingProvider === provider.id}>
        <div class="provider-header">
          <div class="provider-info">
            <Server size={18} />
            <span class="provider-name">{provider.name}</span>
          </div>
          <div class="provider-status">
            {#if isConfigured(provider.id)}
              <span class="status-badge success">
                <Check size={12} /> Configured
              </span>
            {:else}
              <span class="status-badge stopped">
                Not configured
              </span>
            {/if}
          </div>
        </div>

        <div class="provider-fields">
          <div class="form-group">
            <label>Base URL</label>
            {#if editingProvider === provider.id}
              <input
                type="text"
                class="text-input"
                bind:value={provider.base_url}
                placeholder="Enter base URL"
              />
            {:else}
              <div class="field-value">{provider.base_url}</div>
            {/if}
          </div>

          <div class="form-group">
            <label>API Key</label>
            {#if editingProvider === provider.id}
              <div class="password-input-wrapper">
                <input
                  type={showApiKey[provider.id] ? 'text' : 'password'}
                  class="text-input"
                  bind:value={provider.api_key}
                  placeholder="Enter API key"
                />
                <button
                  class="icon-btn"
                  type="button"
                  on:click={() => toggleShowApiKey(provider.id)}
                >
                  {#if showApiKey[provider.id]}
                    <EyeOff size={14} />
                  {:else}
                    <Eye size={14} />
                  {/if}
                </button>
              </div>
            {:else}
              <div class="field-value">
                {#if provider.api_key}
                  {provider.api_key.slice(0, 8)}...
                {:else}
                  <span class="placeholder">Not set</span>
                {/if}
              </div>
            {/if}
          </div>
        </div>

        <div class="provider-actions">
          {#if editingProvider === provider.id}
            <button class="btn secondary" on:click={() => cancelEditing(provider.id)}>
              <X size={14} /> Cancel
            </button>
            <button class="btn primary" on:click={() => saveProvider(provider.id)} disabled={isSaving}>
              <Save size={14} /> Save
            </button>
          {:else}
            <button class="btn secondary" on:click={() => startEditing(provider.id)}>
              <Key size={14} /> Configure
            </button>
            {#if isConfigured(provider.id)}
              <button class="btn danger" on:click={() => deleteProvider(provider.id)}>
                <Trash2 size={14} /> Delete
              </button>
            {/if}
          {/if}
        </div>
      </div>
    {/each}
  </div>
</div>

<style>
  .providers-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .provider-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    transition: border-color 0.15s;
  }

  .provider-card:hover {
    border-color: var(--accent-primary);
  }

  .provider-card.editing {
    border-color: var(--accent-primary);
    background: var(--bg-tertiary);
  }

  .provider-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .provider-info {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-primary);
  }

  .provider-name {
    font-weight: 500;
    font-size: 14px;
  }

  .provider-status {
    display: flex;
    align-items: center;
  }

  .provider-fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 6px;
  }

  .field-value {
    font-size: 13px;
    color: var(--text-secondary);
    padding: 8px 0;
  }

  .field-value .placeholder {
    color: var(--text-muted);
    font-style: italic;
  }

  .password-input-wrapper {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .password-input-wrapper input {
    flex: 1;
  }

  .provider-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
  }

  .status-badge.success {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .status-badge.stopped {
    background: rgba(156, 163, 175, 0.2);
    color: #9ca3af;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
