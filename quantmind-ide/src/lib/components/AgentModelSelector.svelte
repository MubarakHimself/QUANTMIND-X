<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { API_CONFIG } from '$lib/config/api';

  interface Props {
    agentId: string;
    currentModel?: string;
  }

  let { agentId, currentModel = '' }: Props = $props();

  interface Model {
    id: string;
    name: string;
    tier?: string;
  }

  interface ProviderInfo {
    id: string;
    name?: string;
    provider_type?: string;
    display_name: string;
    has_api_key: boolean;
    enabled: boolean;
    available: boolean;
    models: Model[];
  }

  interface ProviderGroup {
    name: string;
    displayName: string;
    models: Model[];
  }

  let providerGroups: ProviderGroup[] = $state([]);
  let selectedProvider = $state('');
  let selectedModel = $state(currentModel);
  let loading = $state(true);
  const dispatch = createEventDispatcher<{ modelchange: { model: string; provider?: string } }>();

  const API_BASE = API_CONFIG.API_BASE;

  onMount(async () => {
    try {
      const url = `${API_BASE}/providers/available`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error(`Expected JSON but got ${contentType}`);
      }

      const data = await response.json();

      if (!data || !data.providers) {
        throw new Error('Invalid response format: missing providers');
      }

      // Group models by provider - only include providers that are available
      const groups: ProviderGroup[] = [];
      const providers = data.providers as ProviderInfo[];

      for (const provider of providers) {
        // Only include providers that have API key configured and are enabled
        if (provider.available && provider.models && provider.models.length > 0) {
          const providerKey = provider.provider_type || provider.name || provider.id;
          if (!providerKey) {
            continue;
          }
          groups.push({
            name: providerKey,
            displayName: provider.display_name,
            models: provider.models
          });
        }
      }

      providerGroups = groups;

      if (groups.length > 0) {
        const matchingProvider = currentModel
          ? groups.find((group) => group.models.some((model) => model.id === currentModel))
          : undefined;
        selectedProvider = matchingProvider?.name || groups[0].name;
      }

      // Set selected model to the current configured value when present, otherwise first available.
      const allModels = groups.flatMap(g => g.models);
      if (allModels.length > 0 && !allModels.find(m => m.id === selectedModel)) {
        const providerModels = groups.find((group) => group.name === selectedProvider)?.models ?? [];
        selectedModel = providerModels[0]?.id || allModels[0].id;
      }
    } catch (e) {
      console.error('Failed to fetch models:', e);
      // Production-safe behavior: no synthetic model fallbacks.
      providerGroups = [];
      selectedProvider = '';
    }
    loading = false;
  });

  // Get models for the currently selected provider
  let currentProviderModels = $derived(
    providerGroups.find(p => p.name === selectedProvider)?.models ?? []
  );

  async function updateModel() {
    try {
      await fetch(`${API_BASE}/agent-config/${agentId}/model`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, provider: selectedProvider || undefined })
      });
      dispatch('modelchange', { model: selectedModel, provider: selectedProvider || undefined });
    } catch (e) {
      console.error('Failed to update model:', e);
    }
  }

  function handleProviderChange() {
    // Reset model to first available from new provider
    const providerModels = providerGroups.find(p => p.name === selectedProvider)?.models ?? [];
    if (providerModels.length > 0) {
      selectedModel = providerModels[0].id;
    }
  }
</script>

{#if loading}
  <div class="model-selector-row">
    <select class="model-selector" disabled>
      <option>Loading...</option>
    </select>
  </div>
{:else if providerGroups.length === 0}
  <div class="model-selector-row">
    <span class="no-providers">No providers configured</span>
  </div>
{:else}
  <div class="model-selector-row">
    <!-- Provider dropdown -->
    <select
      bind:value={selectedProvider}
      onchange={handleProviderChange}
      class="model-selector provider-select"
    >
      {#each providerGroups as group}
        <option value={group.name}>{group.displayName}</option>
      {/each}
    </select>

    <!-- Model dropdown (filtered by selected provider) -->
    <select
      bind:value={selectedModel}
      onchange={updateModel}
      class="model-selector model-select"
    >
      {#each currentProviderModels as model}
        <option value={model.id}>{model.name}</option>
      {/each}
    </select>
  </div>
{/if}

<style>
  .model-selector-row {
    display: flex;
    gap: 6px;
    margin-bottom: 0.5rem;
  }

  .model-selector {
    padding: 4px 8px;
    border-radius: 4px;
    background: var(--bg-secondary, #1e1e1e);
    color: var(--text-primary, #e0e0e0);
    border: 1px solid var(--border-color, #333);
    font-size: 12px;
    cursor: pointer;
  }

  .provider-select {
    min-width: 100px;
  }

  .model-select {
    flex: 1;
    min-width: 140px;
  }

  .model-selector:hover {
    border-color: var(--accent-color, #4a9eff);
  }

  .model-selector:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  :global(.model-selector option) {
    padding: 4px 8px;
    background: var(--bg-secondary, #1e1e1e);
    color: var(--text-primary, #e0e0e0);
  }

  .no-providers {
    font-size: 12px;
    color: var(--text-muted, #888);
    padding: 4px 8px;
  }
</style>
