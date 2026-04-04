<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { get } from 'svelte/store';
  import {
    Key, RefreshCw, Trash2, AlertCircle, Plus,
    Save, Server, Check, X, Zap, ShieldCheck, Cloud, Video,
    Brain, Sparkles, Globe, Cpu, MessageSquare
  } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';
  import { settingsStore } from '$lib/stores/settingsStore';

  // Icon per provider — must match Lucide exports above
  const PROVIDER_ICONS: Record<string, typeof Brain> = {
    anthropic: Sparkles,
    openai: Brain,
    openrouter: Globe,
    deepseek: Cpu,
    glm: MessageSquare,
    minimax: Zap,
    google: Cloud,
    qwen: Video,
    cohere: MessageSquare,
    mistral: Cloud,
  };

  const DEFAULT_PROVIDERS = [
    { id: 'anthropic', name: 'Anthropic (Claude)', base_url: 'https://api.anthropic.com' },
    { id: 'openai', name: 'OpenAI', base_url: 'https://api.openai.com/v1' },
    { id: 'openrouter', name: 'OpenRouter', base_url: 'https://openrouter.ai/api/v1' },
    { id: 'deepseek', name: 'DeepSeek', base_url: 'https://api.deepseek.com/v1' },
    { id: 'glm', name: 'Zhipu AI (GLM)', base_url: 'https://api.z.ai/api/anthropic' },
    { id: 'minimax', name: 'MiniMax', base_url: 'https://api.minimax.io/anthropic' },
    { id: 'google', name: 'Google Gemini', base_url: 'https://generativelanguage.googleapis.com/v1' },
    { id: 'qwen', name: 'Qwen / Alibaba', base_url: 'https://dashscope.aliyuncs.com/api/v1' },
    { id: 'cohere', name: 'Cohere', base_url: 'https://api.cohere.ai/v1' },
    { id: 'mistral', name: 'Mistral AI', base_url: 'https://api.mistral.ai/v1' },
  ];

  interface ProviderModel { id: string; name: string; }

  interface Provider {
    id: string;
    provider_type: string;
    display_name: string;
    base_url: string;
    api_key?: string;
    is_active: boolean;
    model_count: number;
    model_list?: ProviderModel[];
    primary_model?: string;
  }

  // Known models per provider type (mirrors backend PROVIDER_MODELS)
  const PROVIDER_MODELS_MAP: Record<string, ProviderModel[]> = {
    anthropic: [
      { id: 'claude-opus-4-6-20250514', name: 'Claude Opus 4.6' },
      { id: 'claude-sonnet-4-6-20250514', name: 'Claude Sonnet 4.6' },
      { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5' },
      { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet' },
      { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus' },
    ],
    openai: [
      { id: 'gpt-4o', name: 'GPT-4o' },
      { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
      { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
      { id: 'gpt-4', name: 'GPT-4' },
      { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
    ],
    openrouter: [
      { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet (OR)' },
      { id: 'anthropic/claude-3-opus', name: 'Claude 3 Opus (OR)' },
      { id: 'google/gemini-pro-1.5', name: 'Gemini Pro 1.5 (OR)' },
      { id: 'meta-llama/llama-3.1-70b-instruct', name: 'Llama 3.1 70B (OR)' },
      { id: 'mistralai/mistral-7b-instruct', name: 'Mistral 7B (OR)' },
    ],
    deepseek: [
      { id: 'deepseek-chat', name: 'DeepSeek Chat' },
      { id: 'deepseek-reasoner', name: 'DeepSeek R1 (Reasoning)' },
      { id: 'deepseek-coder', name: 'DeepSeek Coder' },
    ],
    glm: [
      { id: 'GLM-5.1', name: 'GLM-5.1' },
      { id: 'glm-4-plus', name: 'GLM-4 Plus' },
      { id: 'glm-4', name: 'GLM-4' },
      { id: 'glm-4-flash', name: 'GLM-4 Flash' },
    ],
    minimax: [
      { id: 'MiniMax-M2.7', name: 'MiniMax M2.7' },
      { id: 'MiniMax-M2.5', name: 'MiniMax M2.5' },
      { id: 'MiniMax-M2.1', name: 'MiniMax M2.1' },
    ],
    google: [
      { id: 'gemini-2.0-flash-exp', name: 'Gemini 2.0 Flash Exp' },
      { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
      { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash' },
    ],
    cohere: [
      { id: 'command-r-plus', name: 'Command R+' },
      { id: 'command-r', name: 'Command R' },
    ],
    mistral: [
      { id: 'mistral-large-latest', name: 'Mistral Large' },
      { id: 'mistral-small-latest', name: 'Mistral Small' },
    ],
  };

  let providers: Provider[] = $state([]);
  let editingProvider: string | null = $state(null);
  let isSaving = $state(false);
  let isLoading = $state(false);
  let isTesting: Record<string, boolean> = $state({});
  let testResults: Record<string, { success: boolean; latency_ms?: number; error?: string }> = $state({});
  let error = $state<string | null>(null);

  let showAddModal = $state(false);
  let newProviderForm = $state({
    provider_type: 'anthropic',
    base_url: 'https://api.anthropic.com/v1',
    api_key: '',
    display_name: '',
    primary_model: 'claude-sonnet-4-6-20250514'
  });

  let showDeleteConfirm = $state(false);
  let deletingProviderId = $state('');

  // ── Zero-Auth state ──────────────────────────────────────────────────────
  interface ZeroAuthStatus {
    qwen_configured: boolean;
    qwen_method: string;
    gemini_configured: boolean;
    gemini_method: string;
    gemini_project: string | null;
  }

  let zeroAuthStatus = $state<ZeroAuthStatus | null>(null);
  // Qwen
  let qwenAuthMode = $state<'browser' | 'apikey'>('browser');
  let qwenApiKey = $state('');
  let isAuthenticatingQwen = $state(false);
  let isSavingQwenKey = $state(false);
  let qwenResult = $state<{ status: string; method?: string; error?: string } | null>(null);
  // Gemini
  let geminiProjectId = $state('');
  let geminiCredentialsPath = $state('');
  let isSavingGemini = $state(false);
  let geminiTestResult = $state<{ status: string; project?: string; credentials?: string } | null>(null);

  async function loadZeroAuthStatus() {
    try {
      const data = await apiFetch<ZeroAuthStatus>('/zero-auth/status');
      zeroAuthStatus = data;
      if (data.gemini_project) geminiProjectId = data.gemini_project;
    } catch (e) {
      console.warn('Zero-auth status unavailable:', e);
    }
  }

  async function authenticateQwenBrowser() {
    isAuthenticatingQwen = true;
    qwenResult = null;
    try {
      const result = await apiFetch<{ status: string; method?: string; error?: string }>(
        '/zero-auth/qwen/authenticate',
        { method: 'POST' }
      );
      qwenResult = result;
      await loadZeroAuthStatus();
    } catch (e) {
      qwenResult = { status: 'error', error: String(e) };
    } finally {
      isAuthenticatingQwen = false;
    }
  }

  async function saveQwenApiKey() {
    if (!qwenApiKey) return;
    isSavingQwenKey = true;
    qwenResult = null;
    try {
      const result = await apiFetch<{ status: string; method?: string; error?: string }>(
        '/zero-auth/qwen/apikey',
        { method: 'POST', body: JSON.stringify({ api_key: qwenApiKey }) }
      );
      qwenResult = result;
      await loadZeroAuthStatus();
    } catch (e) {
      qwenResult = { status: 'error', error: String(e) };
    } finally {
      isSavingQwenKey = false;
    }
  }

  async function saveGemini() {
    if (!geminiProjectId) return;
    isSavingGemini = true;
    geminiTestResult = null;
    try {
      await apiFetch('/zero-auth/gemini/adc', {
        method: 'POST',
        body: JSON.stringify({
          project_id: geminiProjectId,
          credentials_path: geminiCredentialsPath || undefined
        })
      });
      geminiTestResult = await apiFetch('/zero-auth/gemini/test');
      await loadZeroAuthStatus();
    } catch (e) {
      geminiTestResult = { status: 'error' };
    } finally {
      isSavingGemini = false;
    }
  }

  function methodLabel(method: string): string {
    if (method === 'oauth' || method === 'browser_oauth') return 'Browser OAuth';
    if (method === 'adc') return 'Google ADC';
    if (method === 'api_key') return 'API Key';
    return 'Not configured';
  }

  async function loadProviders() {
    isLoading = true;
    error = null;
    try {
      const data = await apiFetch<{ providers: Provider[] }>('/providers');
      // Pull stored api_keys from settings store to pre-fill (backend doesn't return them)
      const storedKeys = get(settingsStore).apiKeys;
      providers = (data.providers || []).map(p => ({
        ...p,
        api_key: (storedKeys as unknown as Record<string, string>)[p.provider_type] || p.api_key || '',
        primary_model: p.primary_model || (PROVIDER_MODELS_MAP[p.provider_type]?.[0]?.id ?? '')
      }));
    } catch (e) {
      error = 'Failed to load providers';
      console.error(e);
    } finally {
      isLoading = false;
    }
  }

  function isConfigured(providerType: string): boolean {
    const p = providers.find(p => p.provider_type === providerType);
    return !!(p && p.is_active);
  }

  function getDefaultBaseUrl(providerType: string): string {
    const def = DEFAULT_PROVIDERS.find(p => p.id === providerType);
    const existing = providers.find(p => p.provider_type === providerType);
    return existing?.base_url || def?.base_url || '';
  }

  function startEditing(providerType: string) {
    editingProvider = providerType;
    // If provider not yet in DB, inject a temporary object so the edit form has something to bind to
    if (!providers.find(p => p.provider_type === providerType)) {
      const def = DEFAULT_PROVIDERS.find(p => p.id === providerType);
      providers = [...providers, {
        id: '',
        provider_type: providerType,
        display_name: def?.name || providerType,
        base_url: def?.base_url || '',
        api_key: '',
        is_active: false,
        model_count: 0,
        model_list: PROVIDER_MODELS_MAP[providerType] || [],
        primary_model: PROVIDER_MODELS_MAP[providerType]?.[0]?.id ?? ''
      }];
    }
  }

  function cancelEditing() {
    editingProvider = null;
    loadProviders();
  }

  async function saveProvider(providerType: string) {
    isSaving = true;
    const provider = providers.find(p => p.provider_type === providerType);
    if (!provider) { isSaving = false; return; }
    try {
      // Build model_list with selected primary model first
      const knownModels = PROVIDER_MODELS_MAP[providerType] || [];
      const primaryModelId = provider.primary_model || knownModels[0]?.id || '';
      const orderedModels = primaryModelId
        ? [
            ...knownModels.filter(m => m.id === primaryModelId),
            ...knownModels.filter(m => m.id !== primaryModelId)
          ]
        : knownModels;
      if (provider.id) {
        await apiFetch(`/providers/${provider.id}`, {
          method: 'PUT',
          body: JSON.stringify({
            provider_type: providerType,
            base_url: provider.base_url,
            api_key: provider.api_key || undefined,
            is_active: provider.is_active,
            model_list: orderedModels.length ? orderedModels : undefined
          })
        });
      } else {
        await apiFetch('/providers', {
          method: 'POST',
          body: JSON.stringify({
            provider_type: providerType,
            display_name: provider.display_name,
            base_url: provider.base_url,
            api_key: provider.api_key,
            is_active: true,
            model_list: orderedModels.length ? orderedModels : undefined
          })
        });
      }
      // Sync API key to settings store
      if (provider.api_key) {
        settingsStore.updateAPIKeys({ [providerType]: provider.api_key });
      }
      // Hot-refresh: invalidate router cache so new key takes effect immediately
      try {
        await apiFetch('/providers/refresh', { method: 'POST' });
      } catch (e) {
        console.warn('Provider router hot-refresh failed:', e);
      }
      editingProvider = null;
      await loadProviders();
    } catch (e) {
      error = 'Failed to save provider';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  async function testProvider(providerId: string) {
    const provider = providers.find(p => p.id === providerId);
    if (!provider) return;
    isTesting[providerId] = true;
    testResults[providerId] = { success: false };
    try {
      const result = await apiFetch<{ success: boolean; latency_ms?: number; error?: string }>(
        `/providers/test`,
        {
          method: 'POST',
          body: JSON.stringify({ provider_type: providerId })
        }
      );
      testResults[providerId] = result;
    } catch (e) {
      testResults[providerId] = { success: false, error: String(e) };
    } finally {
      isTesting[providerId] = false;
    }
  }

  function confirmDelete(providerType: string) {
    deletingProviderId = providerType;
    showDeleteConfirm = true;
  }

  async function deleteProvider() {
    if (!deletingProviderId) return;
    try {
      const provider = providers.find(p => p.provider_type === deletingProviderId);
      if (provider) {
        await apiFetch(`/providers/${provider.id}`, { method: 'DELETE' });
        try {
          await apiFetch('/providers/refresh', { method: 'POST' });
        } catch (e) {
          console.warn('Provider router hot-refresh failed:', e);
        }
        await loadProviders();
      }
    } catch (e) {
      error = 'Failed to delete provider';
      console.error(e);
    } finally {
      showDeleteConfirm = false;
      deletingProviderId = '';
    }
  }

  async function addProvider() {
    isSaving = true;
    error = null;
    try {
      const defP = DEFAULT_PROVIDERS.find(p => p.id === newProviderForm.provider_type);
      const knownModels = PROVIDER_MODELS_MAP[newProviderForm.provider_type] || [];
      const primaryId = newProviderForm.primary_model || knownModels[0]?.id || '';
      const orderedModels = primaryId
        ? [...knownModels.filter(m => m.id === primaryId), ...knownModels.filter(m => m.id !== primaryId)]
        : knownModels;
      await apiFetch('/providers', {
        method: 'POST',
        body: JSON.stringify({
          provider_type: newProviderForm.provider_type,
          display_name: newProviderForm.display_name || defP?.name || newProviderForm.provider_type,
          base_url: newProviderForm.base_url,
          api_key: newProviderForm.api_key,
          is_active: true,
          model_list: orderedModels.length ? orderedModels : undefined
        })
      });
      // Hot-refresh: invalidate router cache so new provider takes effect immediately
      try {
        await apiFetch('/providers/refresh', { method: 'POST' });
      } catch (e) {
        console.warn('Provider router hot-refresh failed:', e);
      }
      showAddModal = false;
      newProviderForm = { provider_type: 'anthropic', base_url: 'https://api.anthropic.com/v1', api_key: '', display_name: '', primary_model: '' };
      await loadProviders();
    } catch (e) {
      error = 'Failed to add provider';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  function openAddModal() {
    const first = DEFAULT_PROVIDERS[0];
    newProviderForm = {
      provider_type: first.id,
      base_url: first.base_url,
      api_key: '',
      display_name: '',
      primary_model: PROVIDER_MODELS_MAP[first.id]?.[0]?.id ?? ''
    };
    showAddModal = true;
  }

  function onProviderTypeChange() {
    const def = DEFAULT_PROVIDERS.find(p => p.id === newProviderForm.provider_type);
    if (def) {
      newProviderForm.base_url = def.base_url;
      newProviderForm.primary_model = PROVIDER_MODELS_MAP[def.id]?.[0]?.id ?? '';
    }
  }

  onMount(() => { loadProviders(); loadZeroAuthStatus(); });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>AI Providers</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={loadProviders} title="Refresh">
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
      <button class="icon-btn accent" onclick={openAddModal} title="Add Provider">
        <Plus size={16} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert-error">
      <AlertCircle size={14} />
      <span>{error}</span>
    </div>
  {/if}

  <div class="info-box">
    <AlertCircle size={14} />
    <span>Configure AI providers. Each provider requires an API key. Test connectivity after setup.</span>
  </div>

  <!-- ── Zero-Auth Configuration Section ───────────────────────────────── -->
  <div class="section-divider">
    <ShieldCheck size={14} />
    <span>Zero-Auth Configuration</span>
  </div>

  <div class="info-box">
    <AlertCircle size={14} />
    <span>Use Qwen CLI for video ingest (browser OAuth or API key) or Gemini via Google ADC (no API key).</span>
  </div>

  <!-- Qwen CLI OAuth -->
  <div class="zero-auth-card qwen-card" class:za-configured={zeroAuthStatus?.qwen_configured}>
    <div class="za-header">
      <div class="provider-info">
        <Video size={16} />
        <span class="provider-name">Qwen Code CLI — Video Analysis</span>
        <span class="badge-tag">Video Ingest</span>
      </div>
      {#if zeroAuthStatus}
        <span class="badge" class:active={zeroAuthStatus.qwen_configured} class:inactive={!zeroAuthStatus.qwen_configured}>
          {methodLabel(zeroAuthStatus.qwen_method)}
        </span>
      {/if}
    </div>
    <p class="za-description">
      Used by the video ingest pipeline to analyze trading videos. Authenticate via browser OAuth
      (free tier: 2,000 req/day) or provide an API key.
    </p>
    <div class="za-mode-selector">
      <button
        class="mode-tab"
        class:active={qwenAuthMode === 'browser'}
        onclick={() => { qwenAuthMode = 'browser'; qwenResult = null; }}
      >Browser OAuth</button>
      <button
        class="mode-tab"
        class:active={qwenAuthMode === 'apikey'}
        onclick={() => { qwenAuthMode = 'apikey'; qwenResult = null; }}
      >API Key</button>
    </div>
    {#if qwenAuthMode === 'browser'}
      <div class="za-fields">
        <p class="za-hint">Opens a browser window to authenticate with your Qwen account. Free tier includes 2,000 requests/day.</p>
      </div>
    {:else}
      <div class="za-fields">
        <div class="form-group">
          <label>QWEN_API_KEY</label>
          <input type="password" class="text-input" bind:value={qwenApiKey} placeholder="sk-..." autocomplete="off" />
        </div>
      </div>
    {/if}
    {#if qwenResult}
      <div class="test-result" class:ok={qwenResult.status === 'configured'} class:fail={qwenResult.status !== 'configured'}>
        {#if qwenResult.status === 'configured'}
          <Check size={13} /> Connected via {qwenResult.method}
        {:else}
          <AlertCircle size={13} /> {qwenResult.error || qwenResult.status}
        {/if}
      </div>
    {/if}
    <div class="provider-actions">
      {#if qwenAuthMode === 'browser'}
        <button class="btn qwen-btn" onclick={authenticateQwenBrowser} disabled={isAuthenticatingQwen}>
          <ShieldCheck size={13} /> {isAuthenticatingQwen ? 'Authenticating...' : 'Authenticate via Browser'}
        </button>
      {:else}
        <button class="btn qwen-btn" onclick={saveQwenApiKey} disabled={isSavingQwenKey || !qwenApiKey}>
          <Save size={13} /> {isSavingQwenKey ? 'Saving...' : 'Save API Key'}
        </button>
      {/if}
    </div>
  </div>

  <!-- Gemini ADC -->
  <div class="zero-auth-card" class:za-configured={zeroAuthStatus?.gemini_configured}>
    <div class="za-header">
      <div class="provider-info">
        <Cloud size={16} />
        <span class="provider-name">Gemini — Google ADC (No API Key)</span>
      </div>
      {#if zeroAuthStatus}
        <span class="badge" class:active={zeroAuthStatus.gemini_configured} class:inactive={!zeroAuthStatus.gemini_configured}>
          {methodLabel(zeroAuthStatus.gemini_method)}
        </span>
      {/if}
    </div>
    <div class="za-fields">
      <div class="form-group">
        <label>Google Cloud Project ID</label>
        <input type="text" class="text-input" bind:value={geminiProjectId} placeholder="my-gcp-project-123" />
      </div>
      <div class="form-group">
        <label>Credentials Path <span class="optional">(optional — leave blank for default ADC)</span></label>
        <input type="text" class="text-input" bind:value={geminiCredentialsPath} placeholder="/path/to/service-account.json" />
      </div>
    </div>
    {#if geminiTestResult}
      <div class="test-result" class:ok={geminiTestResult.status === 'configured'} class:fail={geminiTestResult.status !== 'configured'}>
        {#if geminiTestResult.status === 'configured'}
          <Check size={13} /> Project: {geminiTestResult.project} — {geminiTestResult.credentials}
        {:else}
          <AlertCircle size={13} /> Not configured
        {/if}
      </div>
    {/if}
    <div class="provider-actions">
      <button class="btn primary" onclick={saveGemini} disabled={isSavingGemini || !geminiProjectId}>
        <Save size={13} /> {isSavingGemini ? 'Saving...' : 'Save & Test'}
      </button>
    </div>
  </div>

  <!-- ── End Zero-Auth ──────────────────────────────────────────────────── -->

  <div class="providers-list">
    {#each DEFAULT_PROVIDERS as defP}
      {@const provider = providers.find(p => p.provider_type === defP.id)}
      {@const configured = provider?.is_active}
      {@const IconComponent = PROVIDER_ICONS[defP.id] || Server}
      <div class="provider-card" class:configured class:editing={editingProvider === defP.id}>
        <div class="provider-header">
          <div class="provider-info">
            <IconComponent size={16} />
            <span class="provider-name">{defP.name}</span>
          </div>
          <div class="provider-status">
            {#if configured}
              <span class="badge active"><Check size={11} /> Active</span>
              {#if provider?.model_count}
                <span class="model-count">{provider.model_count} models</span>
              {/if}
            {:else}
              <span class="badge inactive">Not configured</span>
            {/if}
          </div>
        </div>

        {#if editingProvider === defP.id && provider}
          {@const availableModels = PROVIDER_MODELS_MAP[defP.id] || []}
          <div class="provider-fields">
            <div class="form-group">
              <label>Base URL</label>
              <input type="text" class="text-input" bind:value={provider.base_url} placeholder="https://api.example.com/v1" />
            </div>
            <div class="form-group">
              <label>API Key</label>
              <input type="password" class="text-input" bind:value={provider.api_key} placeholder="sk-..." autocomplete="off" />
            </div>
            {#if availableModels.length > 0}
              <div class="form-group">
                <label>Primary Model</label>
                <select class="text-input" bind:value={provider.primary_model}>
                  {#each availableModels as m}
                    <option value={m.id}>{m.name}</option>
                  {/each}
                </select>
                <small class="field-hint">The model used by default for this provider.</small>
              </div>
            {:else}
              <div class="form-group">
                <label>Primary Model</label>
                <input type="text" class="text-input" bind:value={provider.primary_model} placeholder="e.g. gpt-4o or claude-sonnet-4-6" />
                <small class="field-hint">Custom model ID — check your provider's docs.</small>
              </div>
            {/if}
            <div class="form-group toggle-group">
              <label class="toggle-label">
                <input type="checkbox" bind:checked={provider.is_active} />
                <span>Active</span>
              </label>
            </div>
          </div>
        {:else if provider}
          <div class="provider-url">
            {provider.base_url}
            {#if provider.primary_model}<span class="model-badge">{provider.primary_model}</span>{/if}
          </div>
        {:else}
          <div class="provider-url muted">{defP.base_url}</div>
        {/if}

        {#if testResults[defP.id]}
          <div class="test-result" class:ok={testResults[defP.id].success} class:fail={!testResults[defP.id].success}>
            {#if testResults[defP.id].success}
              <Check size={13} /> Connected ({testResults[defP.id].latency_ms}ms)
            {:else}
              <AlertCircle size={13} /> {testResults[defP.id].error || 'Failed'}
            {/if}
          </div>
        {/if}

        <div class="provider-actions">
          {#if editingProvider === defP.id}
            <button class="btn secondary" onclick={cancelEditing}><X size={13} /> Cancel</button>
            <button class="btn primary" onclick={() => saveProvider(defP.id)} disabled={isSaving}>
              <Save size={13} /> {isSaving ? 'Saving...' : 'Save'}
            </button>
          {:else}
            {#if configured}
              <button class="btn secondary" onclick={() => testProvider(defP.id)} disabled={isTesting[defP.id]}>
                <Zap size={13} /> {isTesting[defP.id] ? 'Testing...' : 'Test'}
              </button>
            {/if}
            <button class="btn secondary" onclick={() => startEditing(defP.id)}>
              <Key size={13} /> Configure
            </button>
            {#if configured}
              <button class="btn danger" onclick={() => confirmDelete(defP.id)}>
                <Trash2 size={13} />
              </button>
            {/if}
          {/if}
        </div>
      </div>
    {/each}
  </div>
</div>

{#if showAddModal}
  <div class="modal-backdrop" onclick={() => showAddModal = false} role="button" tabindex="0"
    onkeydown={(e) => e.key === 'Escape' && (showAddModal = false)}>
    <div class="modal" onclick={(e) => e.stopPropagation()} role="dialog">
      <div class="modal-header">
        <h4>Add Provider</h4>
        <button class="icon-btn" onclick={() => showAddModal = false}><X size={16} /></button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label>Provider Type</label>
          <select class="text-input" bind:value={newProviderForm.provider_type} onchange={onProviderTypeChange}>
            {#each DEFAULT_PROVIDERS as p}
              <option value={p.id}>{p.name}</option>
            {/each}
          </select>
        </div>
        <div class="form-group">
          <label>Display Name <span class="optional">(optional)</span></label>
          <input type="text" class="text-input" bind:value={newProviderForm.display_name} placeholder="Custom name" />
        </div>
        <div class="form-group">
          <label>Base URL</label>
          <input type="text" class="text-input" bind:value={newProviderForm.base_url} placeholder="API base URL" />
        </div>
        <div class="form-group">
          <label>API Key</label>
          <input type="password" class="text-input" bind:value={newProviderForm.api_key} placeholder="sk-..." />
        </div>
        <div class="form-group">
          <label>Primary Model</label>
          {#if (PROVIDER_MODELS_MAP[newProviderForm.provider_type] || []).length > 0}
            <select class="text-input" bind:value={newProviderForm.primary_model}>
              {#each PROVIDER_MODELS_MAP[newProviderForm.provider_type] || [] as m}
                <option value={m.id}>{m.name}</option>
              {/each}
            </select>
          {:else}
            <input type="text" class="text-input" bind:value={newProviderForm.primary_model} placeholder="e.g. gpt-4o" />
          {/if}
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" onclick={() => showAddModal = false}>Cancel</button>
        <button class="btn primary" onclick={addProvider} disabled={isSaving || !newProviderForm.api_key}>
          {isSaving ? 'Adding...' : 'Add Provider'}
        </button>
      </div>
    </div>
  </div>
{/if}

{#if showDeleteConfirm}
  <div class="modal-backdrop" onclick={() => showDeleteConfirm = false} role="button" tabindex="0"
    onkeydown={(e) => e.key === 'Escape' && (showDeleteConfirm = false)}>
    <div class="modal" onclick={(e) => e.stopPropagation()} role="dialog">
      <div class="modal-header">
        <h4>Delete Provider</h4>
      </div>
      <div class="modal-body">
        <p class="confirm-text">Remove this provider configuration? This cannot be undone.</p>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" onclick={() => showDeleteConfirm = false}>Cancel</button>
        <button class="btn danger" onclick={deleteProvider}>Delete</button>
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
    letter-spacing: 0.04em;
    text-transform: uppercase;
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

  .providers-list { display: flex; flex-direction: column; gap: 10px; }

  .provider-card {
    background: rgba(8, 13, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 14px;
    transition: border-color 0.15s;
  }

  .provider-card:hover { border-color: rgba(255, 255, 255, 0.15); }
  .provider-card.configured { border-color: rgba(0, 200, 150, 0.3); }
  .provider-card.editing { border-color: rgba(0, 212, 255, 0.4); }

  .provider-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .provider-info {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-primary);
  }

  .provider-name {
    font-size: 13px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .provider-status { display: flex; align-items: center; gap: 8px; }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .badge.active {
    background: rgba(0, 200, 150, 0.15);
    color: #00c896;
    border: 1px solid rgba(0, 200, 150, 0.25);
  }

  .badge.inactive {
    background: rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .model-count {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.35);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .provider-url {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.35);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    margin-bottom: 10px;
  }

  .provider-url.muted { opacity: 0.5; }

  .model-badge {
    display: inline-block;
    margin-left: 8px;
    padding: 1px 6px;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 4px;
    color: rgba(0, 212, 255, 0.8);
    font-size: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .field-hint {
    display: block;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.35);
    margin-top: 3px;
  }

  .provider-fields {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr auto;
    gap: 12px;
    margin-bottom: 10px;
    align-items: end;
  }

  .provider-fields .toggle-group {
    display: flex;
    align-items: center;
  }

  .form-group { display: flex; flex-direction: column; gap: 5px; }

  .form-group label {
    font-size: 11px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.45);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .optional { font-weight: 400; opacity: 0.6; text-transform: none; }

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

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    font-size: 12px !important;
    color: rgba(255, 255, 255, 0.6) !important;
    padding: 7px 0;
  }

  .toggle-label input { accent-color: #00d4ff; width: 14px; height: 14px; }

  .test-result {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    margin-bottom: 10px;
  }

  .test-result.ok {
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    color: #00c896;
  }

  .test-result.fail {
    background: rgba(255, 59, 59, 0.08);
    border: 1px solid rgba(255, 59, 59, 0.2);
    color: #ff3b3b;
  }

  .provider-actions {
    display: flex;
    justify-content: flex-end;
    gap: 6px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    border: none;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn.primary {
    background: #00d4ff;
    color: #080d14;
  }

  .btn.primary:hover { background: #00bce6; }
  .btn.primary:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn.secondary {
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .btn.secondary:hover { background: rgba(255, 255, 255, 0.1); color: #fff; }
  .btn.secondary:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn.danger {
    background: rgba(255, 59, 59, 0.12);
    color: #ff3b3b;
    border: 1px solid rgba(255, 59, 59, 0.2);
  }

  .btn.danger:hover { background: rgba(255, 59, 59, 0.22); }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover { background: rgba(255, 255, 255, 0.1); color: var(--text-primary); }

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
    max-width: 460px;
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

  .confirm-text {
    margin: 0;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.6);
  }

  /* ── Zero-Auth ────────────────────────────────────────────────────────── */
  .section-divider {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 20px 0 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(0, 212, 255, 0.7);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .section-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0, 212, 255, 0.15);
  }

  .zero-auth-card {
    background: rgba(8, 13, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 10px;
    transition: border-color 0.15s;
  }

  .zero-auth-card:hover { border-color: rgba(255, 255, 255, 0.15); }
  .zero-auth-card.za-configured { border-color: rgba(0, 212, 255, 0.3); }

  .za-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .za-fields {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 10px;
  }

  /* ── Qwen card ────────────────────────────────────────────────────────── */
  .zero-auth-card.qwen-card { border-color: rgba(240, 165, 0, 0.2); }
  .zero-auth-card.qwen-card.za-configured { border-color: rgba(240, 165, 0, 0.4); }
  .zero-auth-card.qwen-card:hover { border-color: rgba(240, 165, 0, 0.3); }

  .badge-tag {
    display: inline-flex;
    align-items: center;
    padding: 2px 7px;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    background: rgba(240, 165, 0, 0.12);
    color: rgba(240, 165, 0, 0.9);
    border: 1px solid rgba(240, 165, 0, 0.25);
    letter-spacing: 0.04em;
  }

  .za-description {
    margin: 0 0 12px;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.45);
    line-height: 1.5;
  }

  .za-hint {
    margin: 0;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.4);
    line-height: 1.5;
    padding: 8px 10px;
    background: rgba(240, 165, 0, 0.05);
    border: 1px solid rgba(240, 165, 0, 0.12);
    border-radius: 6px;
  }

  .za-mode-selector {
    display: flex;
    gap: 4px;
    margin-bottom: 12px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 7px;
    padding: 3px;
  }

  .mode-tab {
    flex: 1;
    padding: 5px 10px;
    border: none;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    background: transparent;
    color: rgba(255, 255, 255, 0.45);
    transition: all 0.15s;
  }

  .mode-tab:hover { color: rgba(255, 255, 255, 0.7); }

  .mode-tab.active {
    background: rgba(240, 165, 0, 0.15);
    color: rgba(240, 165, 0, 0.95);
    border: 1px solid rgba(240, 165, 0, 0.25);
  }

  .btn.qwen-btn {
    background: rgba(240, 165, 0, 0.15);
    color: rgba(240, 165, 0, 0.95);
    border: 1px solid rgba(240, 165, 0, 0.3);
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn.qwen-btn:hover { background: rgba(240, 165, 0, 0.25); }
  .btn.qwen-btn:disabled { opacity: 0.45; cursor: not-allowed; }
</style>
