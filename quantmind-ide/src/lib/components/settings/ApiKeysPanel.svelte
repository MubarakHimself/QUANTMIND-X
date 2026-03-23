<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Key, RefreshCw, Trash2, AlertCircle, Plus,
    Brain, Sparkles, Zap, Globe, Server, Cpu,
    Copy, Eye, EyeOff, Check, X
  } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  interface ApiKey {
    id: string;
    name: string;
    key_prefix: string;
    service: string;
    created_at: string;
    last_used_at?: string;
  }

  let apiKeys: ApiKey[] = $state([]);
  let isLoading = $state(false);
  let isSaving = $state(false);
  let error = $state<string | null>(null);
  let copySuccess = $state<string | null>(null);
  let showAddModal = $state(false);
  let showGeneratedKey = $state(false);

  let newKeyForm = $state({ name: '', key: '', service: 'openai' });

  const SERVICE_ICONS: Record<string, typeof Brain> = {
    openai: Brain, anthropic: Sparkles, gemini: Zap,
    openrouter: Globe, together: Server, groq: Cpu
  };

  function getServiceIcon(service: string) {
    return SERVICE_ICONS[service] || Key;
  }

  async function loadApiKeys() {
    isLoading = true;
    error = null;
    try {
      const data = await apiFetch<{ keys: ApiKey[] }>('/api-keys');
      apiKeys = data.keys || [];
    } catch (e) {
      error = 'Failed to load API keys';
      console.error(e);
    } finally {
      isLoading = false;
    }
  }

  async function addApiKey() {
    if (!newKeyForm.name || !newKeyForm.key) return;
    isSaving = true;
    error = null;
    try {
      await apiFetch('/api-keys', {
        method: 'POST',
        body: JSON.stringify({
          name: newKeyForm.name,
          key: newKeyForm.key,
          service: newKeyForm.service
        })
      });
      showAddModal = false;
      newKeyForm = { name: '', key: '', service: 'openai' };
      showGeneratedKey = false;
      await loadApiKeys();
    } catch (e) {
      error = 'Failed to add API key';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  async function deleteApiKey(id: string) {
    try {
      await apiFetch(`/api-keys/${id}`, { method: 'DELETE' });
      await loadApiKeys();
    } catch (e) {
      error = 'Failed to delete API key';
      console.error(e);
    }
  }

  function generateSecureKey(): string {
    const arr = new Uint8Array(32);
    crypto.getRandomValues(arr);
    return 'qm_' + Array.from(arr).map(b => b.toString(16).padStart(2, '0')).join('');
  }

  function handleGenerateKey() {
    newKeyForm.key = generateSecureKey();
    showGeneratedKey = true;
  }

  async function copyToClipboard(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      copySuccess = text.slice(0, 8) + '...';
      setTimeout(() => copySuccess = null, 2000);
    } catch (e) {
      console.error('Copy failed:', e);
    }
  }

  function formatDate(dateStr: string): string {
    try { return new Date(dateStr).toLocaleDateString(); } catch { return dateStr; }
  }

  onMount(() => { loadApiKeys(); });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>API Keys</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={loadApiKeys} title="Refresh">
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
      <button class="icon-btn accent" onclick={() => showAddModal = true} title="Add Key">
        <Plus size={16} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert-error"><AlertCircle size={14} /> <span>{error}</span></div>
  {/if}

  {#if copySuccess}
    <div class="alert-success"><Check size={14} /> <span>Copied {copySuccess}</span></div>
  {/if}

  <div class="info-box">
    <AlertCircle size={14} />
    <span>API keys are stored encrypted server-side. The full key is only shown once at creation.</span>
  </div>

  <div class="keys-list">
    {#if apiKeys.length === 0 && !isLoading}
      <div class="empty-state">
        <Key size={36} />
        <p>No API keys configured</p>
        <button class="btn primary" onclick={() => showAddModal = true}>
          <Plus size={13} /> Add Key
        </button>
      </div>
    {:else}
      {#each apiKeys as apiKey}
        {@const SvelteComponent = getServiceIcon(apiKey.service)}
        <div class="key-item">
          <div class="key-icon">
            <SvelteComponent size={16} />
          </div>
          <div class="key-info">
            <div class="key-name">{apiKey.name}</div>
            <div class="key-service">{apiKey.service}</div>
          </div>
          <div class="key-prefix">
            <code>{apiKey.key_prefix}••••••••</code>
          </div>
          <div class="key-meta">
            <span class="meta-date">{formatDate(apiKey.created_at)}</span>
          </div>
          <div class="key-actions">
            <button class="icon-btn" onclick={() => deleteApiKey(apiKey.id)} title="Delete key">
              <Trash2 size={14} />
            </button>
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>

{#if showAddModal}
  <div class="modal-backdrop" onclick={() => showAddModal = false} role="button" tabindex="0"
    onkeydown={(e) => e.key === 'Escape' && (showAddModal = false)}>
    <div class="modal" onclick={(e) => e.stopPropagation()} role="dialog">
      <div class="modal-header">
        <h4>Add API Key</h4>
        <button class="icon-btn" onclick={() => showAddModal = false}><X size={16} /></button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label>Name</label>
          <input type="text" class="text-input" placeholder="My OpenAI Key" bind:value={newKeyForm.name} />
        </div>
        <div class="form-group">
          <label>Service</label>
          <select class="text-input" bind:value={newKeyForm.service}>
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
          {#if showGeneratedKey}
            <div class="generated-key">
              <code class="generated-key-value">{newKeyForm.key}</code>
              <button class="icon-btn" onclick={() => copyToClipboard(newKeyForm.key)} title="Copy">
                <Copy size={13} />
              </button>
              <button class="icon-btn" onclick={() => showGeneratedKey = false} title="Hide">
                <EyeOff size={13} />
              </button>
            </div>
          {:else}
            <input type="password" class="text-input" placeholder="sk-..." bind:value={newKeyForm.key} />
          {/if}
          <div class="generate-row">
            <button type="button" class="btn secondary small" onclick={handleGenerateKey}>
              <Key size={12} /> Generate Secure Key
            </button>
            {#if newKeyForm.key && !showGeneratedKey}
              <button type="button" class="btn-link" onclick={() => showGeneratedKey = true}>
                <Eye size={11} /> Show
              </button>
            {/if}
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" onclick={() => showAddModal = false}>Cancel</button>
        <button class="btn primary" onclick={addApiKey} disabled={isSaving || !newKeyForm.name || !newKeyForm.key}>
          {isSaving ? 'Adding...' : 'Add Key'}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .header-actions { display: flex; gap: 8px; }

  .alert-error {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.25);
    border-radius: 6px;
    color: #ff3b3b;
    font-size: 12px;
    margin-bottom: 12px;
  }

  .alert-success {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    border-radius: 6px;
    color: #00c896;
    font-size: 12px;
    margin-bottom: 12px;
  }

  .info-box {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 6px;
    font-size: 12px;
    color: rgba(0, 212, 255, 0.8);
    margin-bottom: 16px;
  }

  .keys-list { display: flex; flex-direction: column; gap: 8px; }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: rgba(255, 255, 255, 0.3);
    gap: 14px;
    text-align: center;
  }

  .empty-state p { margin: 0; font-size: 13px; }

  .key-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    background: rgba(8, 13, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    transition: border-color 0.15s;
  }

  .key-item:hover { border-color: rgba(255, 255, 255, 0.14); }

  .key-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 7px;
    color: #00d4ff;
    flex-shrink: 0;
  }

  .key-info { flex: 1; min-width: 0; }

  .key-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .key-service {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.35);
    text-transform: capitalize;
    margin-top: 2px;
  }

  .key-prefix {
    display: flex;
    align-items: center;
  }

  .key-prefix code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
    background: rgba(255, 255, 255, 0.05);
    padding: 3px 7px;
    border-radius: 4px;
  }

  .key-meta { margin-left: 8px; }

  .meta-date {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .key-actions { display: flex; gap: 4px; }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover { background: rgba(255, 59, 59, 0.12); color: #ff3b3b; }

  .icon-btn.accent {
    background: rgba(0, 212, 255, 0.12);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.2);
  }

  .icon-btn.accent:hover { background: rgba(0, 212, 255, 0.2); }

  .spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 12px;
    border: none;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn.primary { background: #00d4ff; color: #080d14; }
  .btn.primary:hover { background: #00bce6; }
  .btn.primary:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn.secondary {
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .btn.secondary:hover { background: rgba(255, 255, 255, 0.1); color: #fff; }
  .btn.secondary.small { padding: 4px 9px; font-size: 11px; }

  .btn-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: none;
    border: none;
    color: #00d4ff;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    padding: 3px 6px;
    border-radius: 4px;
    transition: background 0.15s;
  }

  .btn-link:hover { background: rgba(0, 212, 255, 0.08); }

  .form-group { display: flex; flex-direction: column; gap: 5px; }

  .form-group label {
    font-size: 11px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .text-input {
    width: 100%;
    padding: 7px 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    box-sizing: border-box;
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  .text-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15);
  }

  select.text-input { cursor: pointer; }

  .generated-key {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 6px;
  }

  .generated-key-value {
    flex: 1;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: #00d4ff;
    word-break: break-all;
  }

  .generate-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 6px;
  }

  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.65);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(4px);
  }

  .modal {
    background: rgba(8, 13, 20, 0.97);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    width: 100%;
    max-width: 440px;
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.5);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
  }

  .modal-header h4 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .modal-body {
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    padding: 14px 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
  }
</style>
