<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import {
    Bot, Code, TrendingUp, Save, RefreshCw, Check, Download,
    Upload as UploadIcon, Plus, Trash2, Terminal, Sliders, FileText,
    Briefcase, Users, Layers
  } from 'lucide-svelte';


  interface Props {
    agentConfigs?: Record<string, {
    name: string;
    role: string;
    provider: string;
    model: string;
    temperature: number;
    maxTokens: number;
    systemPrompt: string;
    skills: Array<{ id: string; name: string; description: string; enabled: boolean }>;
    tools: string[];
  }>;
    selectedAgent?: string;
    showRawEditor?: boolean;
    agentsMdContent?: string;
    agentsMdLoading?: boolean;
    agentsMdSaved?: boolean;
  }

  let {
    agentConfigs = $bindable({}),
    selectedAgent = $bindable('floor_manager'),
    showRawEditor = $bindable(true),
    agentsMdContent = $bindable(''),
    agentsMdLoading = false,
    agentsMdSaved = false
  }: Props = $props();

  const dispatch = createEventDispatcher();

  // All agents organized by category
  const ALL_AGENTS = [
    // Floor Manager
    { id: 'floor_manager', name: 'Floor Manager', role: 'Trading Floor Orchestrator', department: null, category: 'floor' },
    // Department Heads
    { id: 'research_head', name: 'Research Head', role: 'Strategy Research Lead', department: 'Research', category: 'head' },
    { id: 'development_head', name: 'Development Head', role: 'EA Development Lead', department: 'Development', category: 'head' },
    { id: 'trading_head', name: 'Trading Head', role: 'Order Execution Lead', department: 'Trading', category: 'head' },
    { id: 'risk_head', name: 'Risk Head', role: 'Risk Management Lead', department: 'Risk', category: 'head' },
    { id: 'portfolio_head', name: 'Portfolio Head', role: 'Portfolio Management Lead', department: 'Portfolio', category: 'head' },
    // Sub-agents - Research
    { id: 'strategy_researcher', name: 'Strategy Researcher', role: 'Research sub-agent', department: 'Research', category: 'subagent' },
    { id: 'market_analyst', name: 'Market Analyst', role: 'Research sub-agent', department: 'Research', category: 'subagent' },
    { id: 'backtester', name: 'Backtester', role: 'Research sub-agent', department: 'Research', category: 'subagent' },
    // Sub-agents - Development
    { id: 'python_dev', name: 'Python Developer', role: 'Development sub-agent', department: 'Development', category: 'subagent' },
    { id: 'pinescript_dev', name: 'PineScript Developer', role: 'Development sub-agent', department: 'Development', category: 'subagent' },
    { id: 'mql5_dev', name: 'MQL5 Developer', role: 'Development sub-agent', department: 'Development', category: 'subagent' },
    // Sub-agents - Trading
    { id: 'order_executor', name: 'Order Executor', role: 'Trading sub-agent', department: 'Trading', category: 'subagent' },
    { id: 'fill_tracker', name: 'Fill Tracker', role: 'Trading sub-agent', department: 'Trading', category: 'subagent' },
    { id: 'trade_monitor', name: 'Trade Monitor', role: 'Trading sub-agent', department: 'Trading', category: 'subagent' },
    // Sub-agents - Risk
    { id: 'position_sizer', name: 'Position Sizer', role: 'Risk sub-agent', department: 'Risk', category: 'subagent' },
    { id: 'drawdown_monitor', name: 'Drawdown Monitor', role: 'Risk sub-agent', department: 'Risk', category: 'subagent' },
    { id: 'var_calculator', name: 'VaR Calculator', role: 'Risk sub-agent', department: 'Risk', category: 'subagent' },
    // Sub-agents - Portfolio
    { id: 'allocation_manager', name: 'Allocation Manager', role: 'Portfolio sub-agent', department: 'Portfolio', category: 'subagent' },
    { id: 'rebalancer', name: 'Rebalancer', role: 'Portfolio sub-agent', department: 'Portfolio', category: 'subagent' },
    { id: 'performance_tracker', name: 'Performance Tracker', role: 'Portfolio sub-agent', department: 'Portfolio', category: 'subagent' },
    // UI Assistant
    { id: 'copilot', name: 'Copilot', role: 'UI Assistant', department: null, category: 'ui' },
  ];

  // Group agents by category for display
  let groupedAgents = $derived({
    floor: ALL_AGENTS.filter(a => a.category === 'floor'),
    heads: ALL_AGENTS.filter(a => a.category === 'head'),
    subagents: ALL_AGENTS.filter(a => a.category === 'subagent'),
    ui: ALL_AGENTS.filter(a => a.category === 'ui'),
  });

  // Available providers from API
  let availableProviders: Array<{ id: string; name: string; display_name: string; has_api_key: boolean; enabled: boolean }> = $state([]);

  // Available models from API (keyed by provider)
  let availableModels: Record<string, Array<{ id: string; name: string; tier: string }>> = {};

  // Fetch available providers and models on mount
  onMount(async () => {
    try {
      // Fetch providers with API key status
      const providersRes = await fetch('/api/providers/available');
      const providersData = await providersRes.json();
      availableProviders = providersData.providers || [];
    } catch (e) {
      console.error('Failed to load available providers:', e);
      // Fallback to default providers if API fails
      availableProviders = [
        { id: 'openrouter', name: 'openrouter', display_name: 'OpenRouter', has_api_key: false, enabled: false },
        { id: 'anthropic', name: 'anthropic', display_name: 'Anthropic', has_api_key: false, enabled: false },
        { id: 'zhipu', name: 'zhipu', display_name: 'Zhipu AI', has_api_key: false, enabled: false },
      ];
    }

    try {
      // Fetch available models from the models endpoint
      const modelsRes = await fetch('/agent-config/available-models');
      const modelsData = await modelsRes.json();
      if (modelsData.providers) {
        availableModels = {};
        for (const [provider, info] of Object.entries(modelsData.providers)) {
          const providerInfo = info as { available: boolean; models: Array<{ id: string; name: string; tier: string }> };
          if (providerInfo.available && providerInfo.models) {
            availableModels[provider] = providerInfo.models;
          }
        }
      }
    } catch (e) {
      console.error('Failed to load available models:', e);
      // Models will fallback to getModelsForProvider function
    }
  });

  // Get providers that have API keys configured
  let providersWithKeys = $derived(availableProviders.filter(p => p.has_api_key));

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

  // Hardcoded fallback models (used if API fails)
  const fallbackModelsByProvider: Record<string, Array<{id: string, name: string}>> = {
    anthropic: [
      { id: 'claude-opus-4-20250514', name: 'Claude Opus 4' },
      { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4' },
      { id: 'claude-haiku-3-20240307', name: 'Claude Haiku 3.5' }
    ],
    zhipu: [
      { id: 'glm-4', name: 'GLM-4' },
      { id: 'glm-4-flash', name: 'GLM-4-Flash' },
      { id: 'glm-4-long', name: 'GLM-4-Long' }
    ],
    minimax: [
      { id: 'MiniMax-M2.5', name: 'MiniMax M2.5' },
      { id: 'MiniMax-M2.1', name: 'MiniMax M2.1' },
      { id: 'MiniMax-M2', name: 'MiniMax M2' }
    ],
    deepseek: [
      { id: 'deepseek-chat', name: 'DeepSeek Chat' },
      { id: 'deepseek-coder', name: 'DeepSeek Coder' }
    ],
    openai: [
      { id: 'gpt-4', name: 'GPT-4' },
      { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
      { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' }
    ],
    openrouter: [
      { id: 'anthropic/claude-sonnet-4', name: 'Claude Sonnet 4 (Router)' },
      { id: 'anthropic/claude-haiku-3', name: 'Claude Haiku 3 (Router)' }
    ]
  };

  function getModelsForProvider(provider: string) {
    // First, try to use models fetched from API
    if (availableModels[provider] && availableModels[provider].length > 0) {
      return availableModels[provider];
    }
    // Fallback to hardcoded models if API didn't return models for this provider
    return fallbackModelsByProvider[provider] || [];
  }
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Agent Configuration</h3>
    <div class="header-actions">
      <button class="btn secondary" onclick={exportAgentsMd} title="Export AGENTS.md">
        <Download size={14} /> Export
      </button>
      <button class="btn secondary" onclick={importAgentsMd} title="Import AGENTS.md">
        <UploadIcon size={14} /> Import
      </button>
      <button
        class="btn primary"
        onclick={saveAgentsMd}
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
    <!-- Floor Manager -->
    {#each groupedAgents.floor as agent}
      <button
        class="agent-tab"
        class:active={selectedAgent === agent.id}
        onclick={() => setSelectedAgent(agent.id)}
      >
        <Bot size={14} />
        <span>{agent.name}</span>
      </button>
    {/each}

    <!-- Department Heads Group -->
    <div class="agent-group">
      <div class="agent-group-header">
        <Briefcase size={12} />
        <span>Department Heads</span>
      </div>
      {#each groupedAgents.heads as agent}
        <button
          class="agent-tab"
          class:active={selectedAgent === agent.id}
          onclick={() => setSelectedAgent(agent.id)}
        >
          <span class="agent-name">{agent.name}</span>
          <span class="agent-dept">{agent.department}</span>
        </button>
      {/each}
    </div>

    <!-- Sub-agents Group -->
    <div class="agent-group">
      <div class="agent-group-header">
        <Users size={12} />
        <span>Sub-agents</span>
      </div>
      {#each groupedAgents.subagents as agent}
        <button
          class="agent-tab subagent"
          class:active={selectedAgent === agent.id}
          onclick={() => setSelectedAgent(agent.id)}
        >
          <span class="agent-name">{agent.name}</span>
          <span class="agent-dept">{agent.department}</span>
        </button>
      {/each}
    </div>

    <!-- UI Assistant -->
    {#each groupedAgents.ui as agent}
      <button
        class="agent-tab"
        class:active={selectedAgent === agent.id}
        onclick={() => setSelectedAgent(agent.id)}
      >
        <Bot size={14} />
        <span>{agent.name}</span>
      </button>
    {/each}
  </div>

  <!-- Editor Mode Toggle -->
  <div class="editor-mode-toggle">
    <button
      class="mode-btn"
      class:active={showRawEditor}
      onclick={() => setShowRawEditor(true)}
    >
      <Terminal size={14} /> Raw Markdown
    </button>
    <button
      class="mode-btn"
      class:active={!showRawEditor}
      onclick={() => setShowRawEditor(false)}
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
            <input type="text" bind:value={agentConfigs[selectedAgent].role} class="text-input" oninput={(e) => updateAgentConfig('role', e.currentTarget.value)} />
          </div>
        </div>

        <!-- Model Configuration -->
        <div class="setting-group">
          <label>Model Configuration</label>
          <div class="setting-row">
            <span>Provider</span>
            <select bind:value={agentConfigs[selectedAgent].provider} onchange={(e) => updateAgentConfig('provider', e.currentTarget.value)}>
              {#if providersWithKeys.length > 0}
                {#each providersWithKeys as provider}
                  <option value={provider.name}>{provider.display_name}</option>
                {/each}
              {:else}
                <option value="">No providers configured</option>
              {/if}
            </select>
          </div>
          <div class="setting-row">
            <span>Model</span>
            <select bind:value={agentConfigs[selectedAgent].model}>
              {#each getModelsForProvider(agentConfigs[selectedAgent].provider) as model}
                <option value={model.id}>{model.name}</option>
              {/each}
            </select>
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
              oninput={(e) => updateAgentConfig('temperature', parseFloat(e.currentTarget.value))}
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
              oninput={(e) => updateAgentConfig('maxTokens', parseInt(e.currentTarget.value))}
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
              oninput={(e) => updateAgentConfig('systemPrompt', e.currentTarget.value)}
            ></textarea>
          </div>
          <small class="help-text">This prompt defines the agent's behavior and responsibilities.</small>
        </div>

        <!-- Skills -->
        <div class="setting-group">
          <label>
            Skills
            <button class="add-skill-btn" onclick={addSkill}>
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
                    onclick={() => removeSkill(index)}
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
                  onchange={(e) => toggleTool(tool, e.currentTarget.checked)}
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

<style>
  /* Agent Selector Tabs */
  .agent-selector-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  .agent-tab {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .agent-tab:hover {
    background: var(--bg-primary);
    border-color: var(--accent-secondary);
    color: var(--text-primary);
  }

  .agent-tab.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: white;
  }

  .agent-tab .agent-name {
    font-weight: 500;
  }

  .agent-tab .agent-dept {
    font-size: 11px;
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
  }

  .agent-tab.active .agent-dept {
    background: rgba(255, 255, 255, 0.25);
  }

  /* Agent Groups */
  .agent-group {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    width: 100%;
    margin-top: 4px;
  }

  .agent-group-header {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 8px 0 4px;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .agent-tab.subagent {
    padding: 6px 10px;
    font-size: 12px;
  }

  /* Editor Mode Toggle */
  .editor-mode-toggle {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
  }

  .mode-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .mode-btn:hover {
    background: var(--bg-primary);
    border-color: var(--accent-secondary);
    color: var(--text-primary);
  }

  .mode-btn.active {
    background: var(--accent-secondary);
    border-color: var(--accent-secondary);
    color: white;
  }

  /* Markdown Editor */
  .agents-md-editor {
    margin-bottom: 16px;
  }

  .code-editor {
    width: 100%;
    min-height: 400px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-primary);
    font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.6;
    resize: vertical;
  }

  .code-editor:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  /* Visual Config Editor */
  .agent-config-editor {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .setting-group {
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
  }

  .setting-group > label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-subtle);
  }

  .setting-row:last-child {
    border-bottom: none;
  }

  .setting-row span {
    color: var(--text-secondary);
    font-size: 13px;
  }

  .setting-row .text-input,
  .setting-row select {
    width: 200px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .setting-row select {
    cursor: pointer;
  }

  .slider-input {
    width: 150px;
    accent-color: var(--accent-primary);
  }

  .number-input {
    width: 100px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .prompt-editor {
    margin-bottom: 8px;
  }

  .prompt-textarea {
    width: 100%;
    min-height: 150px;
    padding: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-family: inherit;
    font-size: 13px;
    line-height: 1.5;
    resize: vertical;
  }

  .prompt-textarea:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .help-text {
    display: block;
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 6px;
  }

  .add-skill-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    background: var(--accent-primary);
    border: none;
    border-radius: 4px;
    color: white;
    font-size: 12px;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .add-skill-btn:hover {
    opacity: 0.9;
  }

  .skills-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .skill-config-item {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  .skill-header {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .skill-name-input {
    flex: 1;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .skill-desc-input {
    width: 100%;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 12px;
  }

  .tools-list {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }

  .tool-checkbox {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: var(--bg-secondary);
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
  }

  .tool-checkbox:hover {
    background: var(--bg-tertiary);
  }

  .tool-checkbox input {
    accent-color: var(--accent-primary);
  }

  .tool-checkbox code {
    font-size: 12px;
    color: var(--text-secondary);
  }

  /* Switch styling */
  .switch.small {
    position: relative;
    display: inline-block;
    width: 32px;
    height: 18px;
  }

  .switch.small input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .switch.small .slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    transition: 0.2s;
    border-radius: 18px;
  }

  .switch.small .slider:before {
    position: absolute;
    content: "";
    height: 12px;
    width: 12px;
    left: 2px;
    bottom: 2px;
    background: white;
    transition: 0.2s;
    border-radius: 50%;
  }

  .switch.small input:checked + .slider {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
  }

  .switch.small input:checked + .slider:before {
    transform: translateX(14px);
  }
</style>
