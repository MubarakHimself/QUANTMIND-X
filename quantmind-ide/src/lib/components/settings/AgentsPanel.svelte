<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import {
    Bot, Code, TrendingUp, Save, RefreshCw, Check, Download,
    Upload as UploadIcon, Plus, Trash2, Terminal, Sliders, FileText
  } from 'lucide-svelte';

  export let agentConfigs: Record<string, {
    name: string;
    role: string;
    provider: string;
    model: string;
    temperature: number;
    maxTokens: number;
    systemPrompt: string;
    skills: Array<{ id: string; name: string; description: string; enabled: boolean }>;
    tools: string[];
  }> = {};

  export let selectedAgent: string = 'copilot';
  export let showRawEditor: boolean = true;
  export let agentsMdContent: string = '';
  export let agentsMdLoading: boolean = false;
  export let agentsMdSaved: boolean = false;

  const dispatch = createEventDispatcher();

  function setSelectedAgent(agent: string) {
    selectedAgent = agent;
    dispatch('selectAgent', { agent });
  }

  function setShowRawEditor(value: boolean) {
    showRawEditor = value;
    dispatch('toggleEditor', { value });
  }

  function exportAgentsMd() {
    dispatch('exportAgentsMd');
  }

  function importAgentsMd() {
    dispatch('importAgentsMd');
  }

  function saveAgentsMd() {
    dispatch('saveAgentsMd');
  }

  function updateAgentConfig(field: string, value: any) {
    dispatch('updateAgentConfig', { agent: selectedAgent, field, value });
  }

  function addSkill() {
    const newId = `custom-${Date.now()}`;
    agentConfigs[selectedAgent].skills = [...agentConfigs[selectedAgent].skills, {
      id: newId,
      name: 'New Skill',
      description: 'Add skill description',
      enabled: true
    }];
    dispatch('updateAgentConfig', { agent: selectedAgent, field: 'skills', value: agentConfigs[selectedAgent].skills });
  }

  function removeSkill(index: number) {
    agentConfigs[selectedAgent].skills = agentConfigs[selectedAgent].skills.filter((_, i) => i !== index);
    dispatch('updateAgentConfig', { agent: selectedAgent, field: 'skills', value: agentConfigs[selectedAgent].skills });
  }

  function toggleTool(tool: string, checked: boolean) {
    if (checked) {
      agentConfigs[selectedAgent].tools = [...agentConfigs[selectedAgent].tools, tool];
    } else {
      agentConfigs[selectedAgent].tools = agentConfigs[selectedAgent].tools.filter(t => t !== tool);
    }
    dispatch('updateAgentConfig', { agent: selectedAgent, field: 'tools', value: agentConfigs[selectedAgent].tools });
  }

  const AVAILABLE_TOOLS = [
    'get_market_data',
    'run_backtest',
    'get_position_size',
    'store_semantic_memory',
    'search_semantic_memories'
  ];
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Agent Configuration</h3>
    <div class="header-actions">
      <button class="btn secondary" on:click={exportAgentsMd} title="Export AGENTS.md">
        <Download size={14} /> Export
      </button>
      <button class="btn secondary" on:click={importAgentsMd} title="Import AGENTS.md">
        <UploadIcon size={14} /> Import
      </button>
      <button
        class="btn primary"
        on:click={saveAgentsMd}
        disabled={agentsMdLoading}
        class:loading={agentsMdLoading}
        class:saved={agentsMdSaved}
      >
        {#if agentsMdLoading}
          <RefreshCw size={14} class="spinning" />
        {:else if agentsMdSaved}
          <Check size={14} />
        {:else}
          <Save size={14} />
        {/if}
        {agentsMdLoading ? 'Saving...' : agentsMdSaved ? 'Saved!' : 'Save'}
      </button>
    </div>
  </div>

  <div class="info-box">
    <FileText size={16} />
    <span>Configure agent behavior, prompts, model settings, and skills. Edit via the visual editor or raw markdown.</span>
  </div>

  <!-- Agent Selector Tabs -->
  <div class="agent-selector-tabs">
    <button
      class="agent-tab"
      class:active={selectedAgent === 'copilot'}
      on:click={() => setSelectedAgent('copilot')}
    >
      <Bot size={14} />
      <span>Copilot</span>
    </button>
    <button
      class="agent-tab"
      class:active={selectedAgent === 'quantcode'}
      on:click={() => setSelectedAgent('quantcode')}
    >
      <Code size={14} />
      <span>QuantCode</span>
    </button>
    <button
      class="agent-tab"
      class:active={selectedAgent === 'analyst'}
      on:click={() => setSelectedAgent('analyst')}
    >
      <TrendingUp size={14} />
      <span>Analyst</span>
    </button>
  </div>

  <!-- Editor Mode Toggle -->
  <div class="editor-mode-toggle">
    <button
      class="mode-btn"
      class:active={showRawEditor}
      on:click={() => setShowRawEditor(true)}
    >
      <Terminal size={14} /> Raw Markdown
    </button>
    <button
      class="mode-btn"
      class:active={!showRawEditor}
      on:click={() => setShowRawEditor(false)}
    >
      <Sliders size={14} /> Visual Editor
    </button>
  </div>

  {#if showRawEditor}
    <!-- Raw Markdown Editor -->
    <div class="agents-md-editor">
      <textarea
        bind:value={agentsMdContent}
        rows="30"
        class="code-editor"
        placeholder="# Agent Configuration

Configure your agent behavior here..."
      ></textarea>
    </div>
  {:else}
    <!-- Visual Per-Agent Configuration Editor -->
    {#if selectedAgent && agentConfigs[selectedAgent]}
      <div class="agent-config-editor">
        <!-- Agent Role & Description -->
        <div class="setting-group">
          <label>Agent Identity</label>
          <div class="setting-row">
            <span>Agent Name</span>
            <input type="text" value={agentConfigs[selectedAgent].name} class="text-input" readonly />
          </div>
          <div class="setting-row">
            <span>Role</span>
            <input type="text" bind:value={agentConfigs[selectedAgent].role} class="text-input" on:input={(e) => updateAgentConfig('role', e.currentTarget.value)} />
          </div>
        </div>

        <!-- Model Configuration -->
        <div class="setting-group">
          <label>Model Configuration</label>
          <div class="setting-row">
            <span>Provider</span>
            <select bind:value={agentConfigs[selectedAgent].provider} on:change={(e) => updateAgentConfig('provider', e.currentTarget.value)}>
              <option value="openrouter">OpenRouter</option>
              <option value="anthropic">Anthropic</option>
              <option value="zhipu">Zhipu AI</option>
            </select>
          </div>
          <div class="setting-row">
            <span>Model</span>
            <input type="text" bind:value={agentConfigs[selectedAgent].model} class="text-input" placeholder="anthropic/claude-sonnet-4" on:input={(e) => updateAgentConfig('model', e.currentTarget.value)} />
          </div>
          <div class="setting-row">
            <span>Temperature: {agentConfigs[selectedAgent].temperature.toFixed(2)}</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              bind:value={agentConfigs[selectedAgent].temperature}
              class="slider-input"
              on:input={(e) => updateAgentConfig('temperature', parseFloat(e.currentTarget.value))}
            />
          </div>
          <div class="setting-row">
            <span>Max Tokens</span>
            <input
              type="number"
              min="1024"
              max="128000"
              step="1024"
              bind:value={agentConfigs[selectedAgent].maxTokens}
              class="number-input"
              on:input={(e) => updateAgentConfig('maxTokens', parseInt(e.currentTarget.value))}
            />
          </div>
        </div>

        <!-- System Prompt -->
        <div class="setting-group">
          <label>System Prompt</label>
          <div class="prompt-editor">
            <textarea
              bind:value={agentConfigs[selectedAgent].systemPrompt}
              rows="12"
              class="prompt-textarea"
              placeholder="Enter the system prompt for this agent..."
              on:input={(e) => updateAgentConfig('systemPrompt', e.currentTarget.value)}
            ></textarea>
          </div>
          <small class="help-text">This prompt defines the agent's behavior and responsibilities.</small>
        </div>

        <!-- Skills -->
        <div class="setting-group">
          <label>
            Skills
            <button class="add-skill-btn" on:click={addSkill}>
              <Plus size={12} /> Add Skill
            </button>
          </label>
          <div class="skills-grid">
            {#each agentConfigs[selectedAgent].skills as skill, index}
              <div class="skill-config-item">
                <div class="skill-header">
                  <label class="switch small">
                    <input type="checkbox" bind:checked={skill.enabled} />
                    <span class="slider"></span>
                  </label>
                  <input
                    type="text"
                    bind:value={skill.name}
                    class="skill-name-input"
                    placeholder="Skill name"
                  />
                  <button
                    class="icon-btn danger"
                    on:click={() => removeSkill(index)}
                    title="Remove skill"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
                <input
                  type="text"
                  bind:value={skill.description}
                  class="skill-desc-input"
                  placeholder="Skill description"
                />
              </div>
            {/each}
          </div>
        </div>

        <!-- Tools -->
        <div class="setting-group">
          <label>Available Tools</label>
          <div class="tools-list">
            {#each AVAILABLE_TOOLS as tool}
              <label class="tool-checkbox">
                <input
                  type="checkbox"
                  checked={agentConfigs[selectedAgent].tools.includes(tool)}
                  on:change={(e) => toggleTool(tool, e.currentTarget.checked)}
                />
                <code>{tool}</code>
              </label>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  {/if}
</div>
