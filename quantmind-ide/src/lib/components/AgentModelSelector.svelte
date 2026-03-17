<script lang="ts">
  import { onMount } from 'svelte';
  import { API_CONFIG } from '$lib/config/api';

  interface Props {
    agentId: string;
    currentModel?: string;
  }

  let { agentId, currentModel = 'sonnet' }: Props = $props();

  interface Model {
    id: string;
    name: string;
    tier?: string;
  }

  interface ProviderInfo {
    id: string;
    name: string;
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
  let selectedModel = $state(currentModel);
  let loading = $state(true);

  const API_BASE = API_CONFIG.API_BASE;

  onMount(async () => {
    try {
      const url = `${API_BASE}/api/providers/available`;

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
          groups.push({
            name: provider.name,
            displayName: provider.display_name,
            models: provider.models
          });
        }
      }

      providerGroups = groups;

      // Set selected model to first available or default
      const allModels = providerGroups.flatMap(g => g.models);
      if (allModels.length > 0 && !allModels.find(m => m.id === selectedModel)) {
        selectedModel = allModels[0].id;
      }
    } catch (e) {
      console.error('Failed to fetch models:', e);
      // Fallback to basic models
      providerGroups = [
        {
          name: 'anthropic',
          displayName: 'Anthropic',
          models: [
            { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4', tier: 'sonnet' },
            { id: 'claude-haiku-3-20240307', name: 'Claude Haiku 3.5', tier: 'haiku' },
          ]
        }
      ];
      selectedModel = providerGroups[0].models[0].id;
    }
    loading = false;
  });

  async function updateModel() {
    try {
      await fetch(`${API_BASE}/agent-config/${agentId}/model`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel })
      });
    } catch (e) {
      console.error('Failed to update model:', e);
    }
  }
</script>

{#if loading}
  <select class="model-selector" disabled>
    <option>Loading...</option>
  </select>
{:else}
  <select
    bind:value={selectedModel}
    onchange={updateModel}
    class="model-selector"
  >
    {#each providerGroups as group}
      <optgroup label={group.displayName}>
        {#each group.models as model}
          <option value={model.id}>{model.name}</option>
        {/each}
      </optgroup>
    {/each}
  </select>
{/if}

<style>
  .model-selector {
    padding: 4px 8px;
    border-radius: 4px;
    background: var(--bg-secondary, #1e1e1e);
    color: var(--text-primary, #e0e0e0);
    border: 1px solid var(--border-color, #333);
    font-size: 12px;
    cursor: pointer;
    min-width: 120px;
  }

  .model-selector:hover {
    border-color: var(--accent-color, #4a9eff);
  }

  .model-selector:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  :global(.model-selector optgroup) {
    font-weight: 600;
    color: var(--text-secondary, #a0a0a0);
  }

  :global(.model-selector option) {
    padding: 4px 8px;
    background: var(--bg-secondary, #1e1e1e);
    color: var(--text-primary, #e0e0e0);
  }
</style>
