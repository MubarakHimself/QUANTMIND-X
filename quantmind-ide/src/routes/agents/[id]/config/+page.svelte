<script lang="ts">
  import { onMount } from "svelte";
  import { page } from "$app/stores";

  // Get agent_id from URL
  $: agentId = $page.params.id;

  // Types
  interface AgentConfig {
    agent_id: string;
    agent_type: string;
    name: string;
    llm_provider: string;
    llm_model: string;
    temperature: number;
    max_tokens: number;
    tools: string[];
    custom: Record<string, unknown>;
  }

  // State
  let config: AgentConfig | null = null;
  let loading = true;
  let saving = false;
  let error: string | null = null;
  let successMessage: string | null = null;

  // Editable fields
  let llm_model = "";
  let temperature = 0;
  let max_tokens = 4096;
  let customJson = "";

  // Load config
  async function loadConfig() {
    loading = true;
    error = null;

    try {
      const response = await fetch(`/api/agents/${agentId}`);
      const data = await response.json();

      if (data.success) {
        const agentConfig = data.data.config;
        config = agentConfig;
        // Initialize editable fields
        llm_model = agentConfig.llm_model;
        temperature = agentConfig.temperature;
        max_tokens = agentConfig.max_tokens;
        customJson = JSON.stringify(agentConfig.custom, null, 2);
      } else {
        error = "Failed to load agent config";
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load config";
    } finally {
      loading = false;
    }
  }

  // Save config
  async function saveConfig() {
    saving = true;
    error = null;
    successMessage = null;

    try {
      // Parse custom JSON
      let custom = {};
      try {
        custom = JSON.parse(customJson);
      } catch {
        error = "Invalid JSON in custom config";
        saving = false;
        return;
      }

      const response = await fetch(`/api/agents/${agentId}/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          llm_model,
          temperature,
          max_tokens,
          custom,
        }),
      });

      const data = await response.json();

      if (data.success) {
        successMessage = "Configuration saved successfully";
        setTimeout(() => (successMessage = null), 3000);
      } else {
        error = data.detail || "Failed to save config";
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to save config";
    } finally {
      saving = false;
    }
  }

  onMount(() => {
    loadConfig();
  });
</script>

<div class="config-page">
  <header class="page-header">
    <div class="header-content">
      <a href="/agents" class="back-link">← Back to Agents</a>
      <h1>⚙️ Agent Configuration</h1>
      <p class="agent-id">Agent: <code>{agentId}</code></p>
    </div>
    <button class="refresh-btn" on:click={loadConfig}>Reload</button>
  </header>

  {#if loading && !config}
    <div class="loading">Loading configuration...</div>
  {:else if error && !config}
    <div class="error">{error}</div>
  {:else if config}
    <div class="config-form">
      <!-- Agent Info (read-only) -->
      <div class="section">
        <h2>Agent Information</h2>
        <div class="info-grid">
          <div class="info-item">
            <span class="info-label">Agent ID</span>
            <span class="info-value">{config.agent_id}</span>
          </div>
          <div class="info-item">
            <span class="info-label">Type</span>
            <span class="info-value">{config.agent_type}</span>
          </div>
          <div class="info-item">
            <span class="info-label">Name</span>
            <span class="info-value">{config.name}</span>
          </div>
          <div class="info-item">
            <span class="info-label">LLM Provider</span>
            <span class="info-value">{config.llm_provider}</span>
          </div>
        </div>
      </div>

      <!-- LLM Settings -->
      <div class="section">
        <h2>LLM Settings</h2>

        <div class="form-group">
          <label for="llm_model">Model</label>
          <input
            type="text"
            id="llm_model"
            bind:value={llm_model}
            placeholder="e.g., gpt-4o-mini"
          />
        </div>

        <div class="form-group">
          <label for="temperature">Temperature: {temperature}</label>
          <input
            type="range"
            id="temperature"
            bind:value={temperature}
            min="0"
            max="2"
            step="0.1"
          />
          <span class="range-hint">0 = focused, 2 = creative</span>
        </div>

        <div class="form-group">
          <label for="max_tokens">Max Tokens</label>
          <input
            type="number"
            id="max_tokens"
            bind:value={max_tokens}
            min="1"
            max="128000"
          />
        </div>
      </div>

      <!-- Tools -->
      <div class="section">
        <h2>Tools</h2>
        <div class="tools-list">
          {#if config.tools && config.tools.length > 0}
            {#each config.tools as tool}
              <span class="tool-badge">{tool}</span>
            {/each}
          {:else}
            <p class="no-tools">No tools configured</p>
          {/if}
        </div>
      </div>

      <!-- Custom Config -->
      <div class="section">
        <h2>Custom Configuration (JSON)</h2>
        <div class="form-group">
          <textarea
            id="custom"
            bind:value={customJson}
            rows="10"
            placeholder={'{ "key": "value" }'}
          ></textarea>
        </div>
      </div>

      <!-- Messages -->
      {#if error}
        <div class="message error">{error}</div>
      {/if}

      {#if successMessage}
        <div class="message success">{successMessage}</div>
      {/if}

      <!-- Actions -->
      <div class="form-actions">
        <button class="save-btn" on:click={saveConfig} disabled={saving}>
          {saving ? "Saving..." : "Save Configuration"}
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  .config-page {
    padding: 1.5rem;
    max-width: 800px;
    margin: 0 auto;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 2rem;
  }

  .header-content {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .back-link {
    color: #3b82f6;
    text-decoration: none;
    font-size: 0.875rem;
  }

  .back-link:hover {
    text-decoration: underline;
  }

  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
  }

  .agent-id {
    color: #9ca3af;
    font-size: 0.875rem;
    margin: 0;
  }

  .agent-id code {
    background: #374151;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
  }

  .refresh-btn {
    padding: 0.5rem 1rem;
    background: #374151;
    color: white;
    border: none;
    border-radius: 0.375rem;
    cursor: pointer;
  }

  .loading,
  .error {
    text-align: center;
    padding: 3rem;
    color: #6b7280;
  }

  .error {
    color: #ef4444;
  }

  .config-form {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .section {
    background: #1f2937;
    border-radius: 0.5rem;
    padding: 1.5rem;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
    color: #f9fafb;
  }

  .info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
  }

  .info-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .info-label {
    font-size: 0.75rem;
    color: #9ca3af;
    text-transform: uppercase;
  }

  .info-value {
    color: #f9fafb;
    font-family: monospace;
  }

  .form-group {
    margin-bottom: 1rem;
  }

  .form-group:last-child {
    margin-bottom: 0;
  }

  label {
    display: block;
    font-size: 0.875rem;
    color: #f9fafb;
    margin-bottom: 0.5rem;
  }

  input[type="text"],
  input[type="number"] {
    width: 100%;
    padding: 0.625rem;
    background: #374151;
    border: 1px solid #4b5563;
    border-radius: 0.375rem;
    color: #f9fafb;
    font-size: 0.875rem;
  }

  input[type="range"] {
    width: 100%;
  }

  .range-hint {
    display: block;
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.25rem;
  }

  textarea {
    width: 100%;
    padding: 0.625rem;
    background: #374151;
    border: 1px solid #4b5563;
    border-radius: 0.375rem;
    color: #f9fafb;
    font-size: 0.875rem;
    font-family: monospace;
    resize: vertical;
  }

  .tools-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .tool-badge {
    background: #374151;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    color: #f9fafb;
  }

  .no-tools {
    color: #6b7280;
    font-size: 0.875rem;
  }

  .message {
    padding: 0.75rem 1rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }

  .message.error {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    border: 1px solid #ef4444;
  }

  .message.success {
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
    border: 1px solid #22c55e;
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
  }

  .save-btn {
    padding: 0.75rem 1.5rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }

  .save-btn:hover:not(:disabled) {
    background: #2563eb;
  }

  .save-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
</style>
