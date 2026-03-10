<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import {
    Bot, Code, TrendingUp, Save, RefreshCw, Check, Download,
    Upload as UploadIcon, Plus, Trash2, Terminal, Sliders, FileText,
    Briefcase, Users, Layers
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

  export let selectedAgent: string = 'floor_manager';
  export let showRawEditor: boolean = true;
  export let agentsMdContent: string = '';
  export let agentsMdLoading: boolean = false;
  export let agentsMdSaved: boolean = false;

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
  $: groupedAgents = {
    floor: ALL_AGENTS.filter(a => a.category === 'floor'),
    heads: ALL_AGENTS.filter(a => a.category === 'head'),
    subagents: ALL_AGENTS.filter(a => a.category === 'subagent'),
    ui: ALL_AGENTS.filter(a => a.category === 'ui'),
  };

  // Available providers from API
  let availableProviders: Array<{ id: string; name: string; display_name: string; has_api_key: boolean; enabled: boolean }> = [];

  // Fetch available providers on mount
  onMount(async () => {
    try {
      const res = await fetch('/api/providers/available');
      const data = await res.json();
      availableProviders = data.providers || [];
    } catch (e) {
      console.error('Failed to load available providers:', e);
      // Fallback to default providers if API fails
      availableProviders = [
        { id: 'openrouter', name: 'openrouter', display_name: 'OpenRouter', has_api_key: false, enabled: false },
        { id: 'anthropic', name: 'anthropic', display_name: 'Anthropic', has_api_key: false, enabled: false },
        { id: 'zhipu', name: 'zhipu', display_name: 'Zhipu AI', has_api_key: false, enabled: false },
      ];
    }
  });

  // Get providers that have API keys configured
  $: providersWithKeys = availableProviders.filter(p => p.has_api_key);

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
    <!-- Floor Manager -->
    {#each groupedAgents.floor as agent}
      <button
        class="agent-tab"
        class:active={selectedAgent === agent.id}
        on:click={() => setSelectedAgent(agent.id)}
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
          on:click={() => setSelectedAgent(agent.id)}
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
          on:click={() => setSelectedAgent(agent.id)}
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
        on:click={() => setSelectedAgent(agent.id)}
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
