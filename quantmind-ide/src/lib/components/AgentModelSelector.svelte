<script lang="ts">
  import { onMount } from 'svelte';
  import { API_CONFIG } from '$lib/config/api';

  export let agentId: string;
  export let currentModel: string = 'sonnet';

  interface Model {
    id: string;
    name: string;
    tier: string;
  }

  interface ProviderModels {
    available: boolean;
    models: Model[];
  }

  let models: Model[] = [];
  let selectedModel = currentModel;
  let loading = true;

  const API_BASE = API_CONFIG.API_BASE;

  onMount(async () => {
    try {
      const response = await fetch(`${API_BASE}/agent-config/available-models`);
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      const data = await response.json();

      // Collect all available models from providers
      const allModels: Model[] = [];
      for (const [provider, info] of Object.entries(data.providers)) {
        const providerInfo = info as ProviderModels;
        if (providerInfo.available && providerInfo.models) {
          allModels.push(...providerInfo.models);
        }
      }

      models = allModels;

      // Set selected model to first available or default
      if (models.length > 0 && !models.find(m => m.id === selectedModel)) {
        selectedModel = models[0].id;
      }
    } catch (e) {
      console.error('Failed to fetch models:', e);
      // Fallback to basic models
      models = [
        { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4', tier: 'sonnet' },
        { id: 'claude-haiku-3-20240307', name: 'Claude Haiku 3.5', tier: 'haiku' },
      ];
      selectedModel = models[0].id;
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
    on:change={updateModel}
    class="model-selector"
  >
    {#each models as model}
      <option value={model.id}>{model.name}</option>
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
</style>
