<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Bot, Save, RefreshCw, Check, Plus, Trash2, FileText,
    Briefcase, Users, Shield, Lock, Globe, Database
  } from 'lucide-svelte';
  import { apiFetch, buildApiUrl } from '$lib/api';

  // Permission levels for access control
  const ACCESS_LEVELS = [
    { value: 'none', label: 'None', color: 'var(--text-muted)' },
    { value: 'read', label: 'Read', color: 'var(--accent-primary)' },
    { value: 'write', label: 'Write', color: 'var(--accent-warning)' },
    { value: 'full', label: 'Full', color: 'var(--accent-danger)' }
  ];

  // Permission types with icons
  const PERMISSION_TYPES = [
    { key: 'fileSystem', label: 'File System', description: 'Access to read/write files', icon: FileText },
    { key: 'broker', label: 'Broker', description: 'Access to trading accounts', icon: Briefcase },
    { key: 'database', label: 'Database', description: 'Access to data storage', icon: Database }
  ];

  interface AgentPermissions {
    fileSystem: 'none' | 'read' | 'write' | 'full';
    broker: 'none' | 'read' | 'write' | 'full';
    database: 'none' | 'read' | 'write' | 'full';
    external: boolean;
    memory: boolean;
  }

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
    permissions?: AgentPermissions;
  }>;
    selectedAgent?: string;
    onselectAgent?: (agent: string) => void;
    onupdateAgentConfig?: (agent: string, field: string, value: unknown) => void;
    onsaveAgentConfig?: (agent: string, config: Record<string, unknown>) => void;
  }

  let {
    agentConfigs = $bindable({}),
    selectedAgent = $bindable('floor_manager'),
    onselectAgent,
    onupdateAgentConfig,
    onsaveAgentConfig
  }: Props = $props();

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
  ];

  // Group agents by category for display
  let groupedAgents = $derived({
    floor: ALL_AGENTS.filter(a => a.category === 'floor'),
    heads: ALL_AGENTS.filter(a => a.category === 'head'),
    subagents: ALL_AGENTS.filter(a => a.category === 'subagent'),
  });

  let providerStateLoading = $state(false);
  let providerStateLoaded = $state(false);

  // Available providers from API
  let availableProviders: Array<{ id: string; name: string; display_name: string; has_api_key: boolean; enabled: boolean; provider_type?: string }> = $state([]);

  // Available models from API (keyed by provider)
  let availableModels: Record<string, Array<{ id: string; name: string; tier: string }>> = $state({});
  let visibleModels: Array<{ id: string; name: string; tier: string }> = $state([]);

  async function loadProviderState(force = false) {
    if (providerStateLoading || (providerStateLoaded && !force)) {
      return;
    }

    providerStateLoading = true;
    let providerLoadSucceeded = false;
    let modelLoadSucceeded = false;
    try {
      // Fetch providers with API key status
      const providersData = await apiFetch<{ providers?: Array<{ id: string; name: string; display_name: string; has_api_key: boolean; enabled: boolean; provider_type?: string }> }>('/providers/available');
      availableProviders = providersData.providers || [];
      providerLoadSucceeded = true;
    } catch (e) {
      console.error('Failed to load available providers:', e);
      availableProviders = [];
    }

    try {
      // Fetch available models from the models endpoint
      const modelsData = await apiFetch<{ providers?: Record<string, { available: boolean; models: Array<{ id: string; name: string; tier: string }> }> }>('/agent-config/available-models');
      if (modelsData.providers) {
        const nextModels: Record<string, Array<{ id: string; name: string; tier: string }>> = {};
        for (const [provider, info] of Object.entries(modelsData.providers)) {
          const providerInfo = info as { available: boolean; models: Array<{ id: string; name: string; tier: string }> };
          if (providerInfo.available && providerInfo.models) {
            nextModels[provider] = providerInfo.models;
          }
        }
        availableModels = nextModels;
      } else {
        availableModels = {};
      }
      modelLoadSucceeded = true;
    } catch (e) {
      console.error('Failed to load available models:', e);
      availableModels = {};
    } finally {
      providerStateLoading = false;
      providerStateLoaded = providerLoadSucceeded || modelLoadSucceeded;
      if (providerStateLoaded && selectedAgent) {
        syncAgentProviderAndModel(selectedAgent);
        refreshVisibleModels(selectedAgent);
      }
    }
  }

  // Fetch available providers and models on mount
  onMount(async () => {
    await loadProviderState(true);

    // Load all prompts and department config in parallel
    await Promise.all([
      loadAllPrompts(),
      loadDepartmentConfig(),
    ]);

    // Load system prompt for initially selected agent
    if (selectedAgent) {
      await loadSystemPrompt(selectedAgent);
    }
  });

  // Get providers that have API keys configured
  let providersWithKeys = $derived(availableProviders.filter(p => p.has_api_key));

  function setSelectedAgent(agent: string) {
    selectedAgent = agent;
    onselectAgent?.(agent);
    void loadSystemPrompt(agent);
    queueMicrotask(() => {
      syncAgentProviderAndModel(agent);
      refreshVisibleModels(agent);
    });
  }

  function updateAgentConfig(field: string, value: unknown) {
    onupdateAgentConfig?.(selectedAgent, field, value);
  }

  // Cache of all prompts (default + overrides) keyed by agent_id
  let allPromptsCache: Record<string, { system_prompt: string; is_override: boolean; default_prompt: string }> = $state({});

  async function loadAllPrompts() {
    try {
      allPromptsCache = await apiFetch<Record<string, { system_prompt: string; is_override: boolean; default_prompt: string }>>('/settings/agents/all-prompts');
    } catch (e) {
      console.error('Failed to load all prompts:', e);
    }
  }

  async function loadSystemPrompt(agentId: string) {
    // If cache is empty, load all prompts first
    if (Object.keys(allPromptsCache).length === 0) {
      await loadAllPrompts();
    }

    // Use cached data if available
    const cached = allPromptsCache[agentId];
    if (cached && agentConfigs[agentId]) {
      agentConfigs[agentId] = { ...agentConfigs[agentId], systemPrompt: cached.system_prompt };
      return;
    }

    // Fallback to per-agent endpoint
    try {
      const res = await fetch(buildApiUrl(`/api/settings/agents/${agentId}/system-prompt`), {
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        const prompt: string = data.system_prompt ?? data.systemPrompt ?? '';
        if (prompt && agentConfigs[agentId]) {
          agentConfigs[agentId] = { ...agentConfigs[agentId], systemPrompt: prompt };
        }
      }
    } catch (e) {
      console.error('Failed to load system prompt for', agentId, e);
    }
  }

  let promptSaved = $state(false);
  let promptSaving = $state(false);
  let promptIsOverride = $derived(allPromptsCache[selectedAgent]?.is_override ?? false);

  async function saveSystemPrompt(agentId: string, prompt: string) {
    promptSaving = true;
    promptSaved = false;
    try {
      await fetch(buildApiUrl(`/api/settings/agents/${agentId}/system-prompt`), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ system_prompt: prompt })
      });
      // Update cache
      if (allPromptsCache[agentId]) {
        allPromptsCache[agentId] = { ...allPromptsCache[agentId], system_prompt: prompt, is_override: true };
      }
      promptSaved = true;
      setTimeout(() => { promptSaved = false; }, 2000);
    } catch (e) {
      console.error('Failed to save system prompt for', agentId, e);
    } finally {
      promptSaving = false;
    }
  }

  async function savePromptNow() {
    if (agentConfigs[selectedAgent]) {
      await saveSystemPrompt(selectedAgent, agentConfigs[selectedAgent].systemPrompt);
    }
  }

  async function resetPromptToDefault() {
    try {
      await fetch(buildApiUrl(`/api/settings/agents/${selectedAgent}/system-prompt/reset`), {
        method: 'POST',
        credentials: 'include',
      });
      // Reload from defaults
      await loadAllPrompts();
      const cached = allPromptsCache[selectedAgent];
      if (cached && agentConfigs[selectedAgent]) {
        agentConfigs[selectedAgent] = { ...agentConfigs[selectedAgent], systemPrompt: cached.default_prompt };
      }
      promptSaved = true;
      setTimeout(() => { promptSaved = false; }, 2000);
    } catch (e) {
      console.error('Failed to reset system prompt for', selectedAgent, e);
    }
  }

  // Department config (skills + MCP servers)
  let departmentConfig: Record<string, { skills: any[]; mcp_servers: string[]; mcp_config_file: string; mcp_config_source_file?: string }> = $state({});

  async function loadDepartmentConfig() {
    try {
      departmentConfig = await apiFetch<Record<string, { skills: any[]; mcp_servers: string[]; mcp_config_file: string; mcp_config_source_file?: string }>>('/settings/departments/config');
    } catch (e) {
      console.error('Failed to load department config:', e);
    }
  }

  function addSkill() {
    const newId = `custom-${Date.now()}`;
    const newSkill = {
      id: newId,
      name: 'New Skill',
      description: 'Add skill description',
      enabled: true
    };
    const updatedSkills = [...agentConfigs[selectedAgent].skills, newSkill];
    agentConfigs[selectedAgent] = { ...agentConfigs[selectedAgent], skills: updatedSkills };
    onupdateAgentConfig?.(selectedAgent, 'skills', updatedSkills);
  }

  function removeSkill(index: number) {
    const updatedSkills = agentConfigs[selectedAgent].skills.filter((_, i) => i !== index);
    agentConfigs[selectedAgent] = { ...agentConfigs[selectedAgent], skills: updatedSkills };
    onupdateAgentConfig?.(selectedAgent, 'skills', updatedSkills);
  }

  function toggleTool(tool: string, checked: boolean) {
    let updatedTools: string[];
    if (checked) {
      updatedTools = [...agentConfigs[selectedAgent].tools, tool];
    } else {
      updatedTools = agentConfigs[selectedAgent].tools.filter(t => t !== tool);
    }
    agentConfigs[selectedAgent] = { ...agentConfigs[selectedAgent], tools: updatedTools };
    onupdateAgentConfig?.(selectedAgent, 'tools', updatedTools);
  }

  // Permission handlers
  function updatePermission(key: keyof AgentPermissions, value: string | boolean) {
    if (!agentConfigs[selectedAgent].permissions) {
      agentConfigs[selectedAgent].permissions = {
        fileSystem: 'read',
        broker: 'read',
        database: 'read',
        external: false,
        memory: true
      };
    }
    const updatedPermissions = { ...agentConfigs[selectedAgent].permissions!, [key]: value };
    agentConfigs[selectedAgent] = { ...agentConfigs[selectedAgent], permissions: updatedPermissions };
    onupdateAgentConfig?.(selectedAgent, 'permissions', updatedPermissions);
  }

  function getPermissionValue(key: keyof AgentPermissions): string {
    return agentConfigs[selectedAgent]?.permissions?.[key] as string || 'read';
  }

  function isPermissionActive(key: keyof AgentPermissions, value: string): boolean {
    return getPermissionValue(key) === value;
  }

  // Save handler
  function saveAgentConfig() {
    const config = { ...agentConfigs[selectedAgent] };
    onsaveAgentConfig?.(selectedAgent, config);
  }

  const AVAILABLE_TOOLS = [
    'get_market_data',
    'run_backtest',
    'get_position_size',
    'store_semantic_memory',
    'search_semantic_memories'
  ];

  function getModelsForProvider(provider: string) {
    const normalized = normalizeProviderKey(provider);
    if (availableModels[normalized] && availableModels[normalized].length > 0) {
      return availableModels[normalized];
    }
    return [];
  }

  const AGENT_TONES: Record<string, { accent: string; surface: string; border: string }> = {
    standard: {
      accent: 'var(--accent-primary)',
      surface: 'color-mix(in srgb, var(--accent-primary) 12%, var(--bg-primary))',
      border: 'color-mix(in srgb, var(--accent-primary) 28%, var(--border-subtle))'
    },
    research: {
      accent: '#4f8cff',
      surface: 'rgba(79, 140, 255, 0.14)',
      border: 'rgba(79, 140, 255, 0.34)'
    },
    development: {
      accent: '#28c17a',
      surface: 'rgba(40, 193, 122, 0.14)',
      border: 'rgba(40, 193, 122, 0.34)'
    },
    trading: {
      accent: '#d6a22a',
      surface: 'rgba(214, 162, 42, 0.14)',
      border: 'rgba(214, 162, 42, 0.34)'
    },
    risk: {
      accent: '#ff6b6b',
      surface: 'rgba(255, 107, 107, 0.14)',
      border: 'rgba(255, 107, 107, 0.34)'
    },
    portfolio: {
      accent: '#9b7bff',
      surface: 'rgba(155, 123, 255, 0.14)',
      border: 'rgba(155, 123, 255, 0.34)'
    }
  };

  function getAgentMeta(agentId: string) {
    return ALL_AGENTS.find((agent) => agent.id === agentId);
  }

  function getDepartmentKey(agentId: string) {
    const department = getAgentMeta(agentId)?.department?.toLowerCase();
    if (!department) {
      return 'standard';
    }
    if (department in AGENT_TONES) {
      return department;
    }
    return 'standard';
  }

  function getToneStyle(agentId: string) {
    const tone = AGENT_TONES[getDepartmentKey(agentId)] ?? AGENT_TONES.standard;
    return `--agent-accent: ${tone.accent}; --agent-surface: ${tone.surface}; --agent-border: ${tone.border};`;
  }

  function normalizeProviderKey(provider: string) {
    const raw = `${provider ?? ''}`.trim();
    if (!raw) {
      return '';
    }

    if (availableModels[raw]) {
      return raw;
    }

    const lower = raw.toLowerCase();
    if (availableModels[lower]) {
      return lower;
    }

    const match = availableProviders.find((item) =>
      item.id === raw ||
      item.name === raw ||
      item.display_name === raw ||
      item.id === lower ||
      item.name === lower ||
      item.display_name.toLowerCase() === lower
    );

    if (match?.name && availableModels[match.name]) {
      return match.name;
    }

    if (match?.provider_type && availableModels[match.provider_type]) {
      return match.provider_type;
    }

    return match?.name ?? match?.provider_type ?? match?.id ?? lower;
  }

  function refreshVisibleModels(agentId: string, providerOverride?: string) {
    const current = agentConfigs[agentId];
    if (!current) {
      visibleModels = [];
      return;
    }
    visibleModels = getModelsForProvider(providerOverride ?? current.provider);
  }

  function syncAgentProviderAndModel(agentId: string) {
    const current = agentConfigs[agentId];
    if (!current) {
      visibleModels = [];
      return;
    }

    const providerKey = normalizeProviderKey(current.provider);
    const models = getModelsForProvider(providerKey);
    let nextConfig = current;
    let changed = false;

    if (providerKey && current.provider !== providerKey) {
      nextConfig = { ...nextConfig, provider: providerKey };
      changed = true;
    }

    if (models.length > 0 && !models.some((model) => model.id === nextConfig.model)) {
      nextConfig = { ...nextConfig, model: models[0].id };
      changed = true;
    }

    if (changed) {
      agentConfigs[agentId] = nextConfig;
    }
    refreshVisibleModels(agentId, nextConfig.provider);
  }

  $effect(() => {
    const current = agentConfigs[selectedAgent];
    const provider = current?.provider ?? '';
    const modelCatalogVersion = Object.keys(availableModels).join('|');
    providerStateLoaded;
    modelCatalogVersion;

    if (!selectedAgent || !current) {
      visibleModels = [];
      return;
    }

    refreshVisibleModels(selectedAgent, provider);
  });
</script>

<div class="panel">
  <div class="agents-workspace">
    <aside class="agent-browser">
      <div class="agent-selector-tabs">
        {#each groupedAgents.floor as agent}
          <button
            class="agent-tab floor"
            class:active={selectedAgent === agent.id}
            style={getToneStyle(agent.id)}
            onclick={() => setSelectedAgent(agent.id)}
          >
            <div class="agent-tab-copy">
              <span class="agent-name">{agent.name}</span>
              <span class="agent-role">{agent.role}</span>
            </div>
            <Bot size={14} />
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
              style={getToneStyle(agent.id)}
              onclick={() => setSelectedAgent(agent.id)}
            >
              <div class="agent-tab-copy">
                <span class="agent-name">{agent.name}</span>
                <span class="agent-role">{agent.role}</span>
              </div>
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
              style={getToneStyle(agent.id)}
              onclick={() => setSelectedAgent(agent.id)}
            >
              <div class="agent-tab-copy">
                <span class="agent-name">{agent.name}</span>
                <span class="agent-role">{agent.role}</span>
              </div>
              <span class="agent-dept">{agent.department}</span>
            </button>
          {/each}
        </div>
      </div>
    </aside>

    <section class="agents-content">
      {#if selectedAgent && agentConfigs[selectedAgent]}
          <div class="agent-config-editor">
            <div class="agent-hero" style={getToneStyle(selectedAgent)}>
              <div>
                <h4>{agentConfigs[selectedAgent].name}</h4>
                <p class="agent-hero-role">{agentConfigs[selectedAgent].role}</p>
              </div>
              <div class="agent-hero-meta">
                <span class="meta-pill">{selectedAgent}</span>
                {#if getAgentMeta(selectedAgent)?.department}
                  <span class="meta-pill subtle">{getAgentMeta(selectedAgent)?.department}</span>
                {/if}
              </div>
            </div>

            <div class="editor-top-grid">
              <div class="setting-group prompt-group">
                <label class="prompt-label">
                  <span class="prompt-title">
                    <span>System Prompt</span>
                    {#if promptIsOverride}
                      <span class="override-badge">Custom Override</span>
                    {:else}
                      <span class="default-badge">Default</span>
                    {/if}
                  </span>
                  <span class="prompt-actions">
                    {#if promptSaved}
                      <span class="saved-badge"><Check size={11} /> Saved</span>
                    {/if}
                    {#if promptIsOverride}
                      <button class="btn secondary small" onclick={resetPromptToDefault} title="Reset to built-in default prompt">
                        <RefreshCw size={12} /> Reset
                      </button>
                    {/if}
                    <button class="btn secondary small" onclick={savePromptNow} disabled={promptSaving}>
                      <Save size={12} /> {promptSaving ? 'Saving…' : 'Save Prompt'}
                    </button>
                  </span>
                </label>
                <div class="prompt-editor">
                  <textarea
                    bind:value={agentConfigs[selectedAgent].systemPrompt}
                    rows="24"
                    class="prompt-textarea"
                    placeholder="Enter the system prompt for this agent..."
                  ></textarea>
                </div>
                <small class="help-text">Saved prompts override defaults at runtime. Changes take effect on next agent invocation.</small>
              </div>

              <div class="meta-column">
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

                <div class="setting-group">
                  <label>Model Configuration</label>
                  <div class="setting-row">
                    <span>Provider</span>
                    <select bind:value={agentConfigs[selectedAgent].provider} onchange={(e) => {
                      const newProvider = normalizeProviderKey(e.currentTarget.value);
                      updateAgentConfig('provider', newProvider);
                      const models = getModelsForProvider(newProvider);
                      refreshVisibleModels(selectedAgent, newProvider);
                      if (models.length > 0) {
                        updateAgentConfig('model', models[0].id);
                      }
                    }}>
                      {#if providersWithKeys.length > 0}
                        {#each providersWithKeys as provider}
                          <option value={provider.name || provider.id}>{provider.display_name}</option>
                        {/each}
                      {:else}
                        <option value="">No providers configured</option>
                      {/if}
                    </select>
                  </div>
                  <div class="setting-row">
                    <span>Model</span>
                    <select bind:value={agentConfigs[selectedAgent].model} onchange={(e) => updateAgentConfig('model', e.currentTarget.value)}>
                      {#each visibleModels as model}
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
              </div>
            </div>

        <!-- Department Skills & MCP -->
        {#if departmentConfig[selectedAgent] || departmentConfig[selectedAgent?.replace('_head', '')]}
          {@const deptKey = departmentConfig[selectedAgent] ? selectedAgent : selectedAgent?.replace('_head', '')}
          {@const deptCfg = departmentConfig[deptKey]}
          {#if deptCfg}
            <div class="setting-group">
              <label>Agentic Skills <span class="count-badge">{deptCfg.skills?.length ?? 0}</span></label>
              <div class="skills-grid">
                {#each deptCfg.skills ?? [] as skill}
                  <div class="skill-config-item">
                    <div class="skill-header">
                      <code class="slash-cmd">{skill.slash_command}</code>
                      <strong>{skill.name}</strong>
                    </div>
                    <small class="skill-desc">{skill.description}</small>
                  </div>
                {/each}
              </div>
            </div>
            <div class="setting-group">
              <label>MCP Servers <span class="count-badge">{deptCfg.mcp_servers?.length ?? 0}</span></label>
              {#if deptCfg.mcp_servers?.length > 0}
                <div class="mcp-chips">
                  {#each deptCfg.mcp_servers as server}
                    <span class="mcp-chip">{server}</span>
                  {/each}
                </div>
              {:else}
                <small class="help-text">No MCP servers assigned to this department.</small>
              {/if}
              <small class="help-text">Config file: <code>{deptCfg.mcp_config_file}</code></small>
              {#if deptCfg.mcp_config_source_file && deptCfg.mcp_config_source_file !== deptCfg.mcp_config_file}
                <small class="help-text">Compat source: <code>{deptCfg.mcp_config_source_file}</code></small>
              {/if}
            </div>
          {/if}
        {/if}

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

        <!-- Permissions -->
        <div class="setting-group">
          <label>
            <Shield size={14} />
            Access Permissions
          </label>
          <div class="permissions-grid">
            {#each PERMISSION_TYPES as perm}
              <div class="permission-row">
                <div class="permission-info">
                  <span class="permission-label">
                    <perm.icon size={12} />
                    {perm.label}
                  </span>
                  <span class="permission-desc">{perm.description}</span>
                </div>
                <div class="permission-levels">
                  {#each ACCESS_LEVELS as level}
                    <button
                      class="level-btn"
                      class:active={isPermissionActive(perm.key as keyof AgentPermissions, level.value)}
                      style="--level-color: {level.color}"
                      onclick={() => updatePermission(perm.key as keyof AgentPermissions, level.value)}
                    >
                      {level.label}
                    </button>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
          <div class="boolean-permissions">
            <div class="boolean-perm-row">
              <div class="permission-info">
                <span class="permission-label">
                  <Globe size={12} />
                  External APIs
                </span>
                <span class="permission-desc">Allow calls to external services</span>
              </div>
              <label class="toggle">
                <input
                  type="checkbox"
                  checked={agentConfigs[selectedAgent]?.permissions?.external ?? false}
                  onchange={(e) => updatePermission('external', e.currentTarget.checked)}
                />
                <span class="toggle-slider"></span>
              </label>
            </div>
            <div class="boolean-perm-row">
              <div class="permission-info">
                <span class="permission-label">
                  <Lock size={12} />
                  Memory Access
                </span>
                <span class="permission-desc">Access to persistent memory</span>
              </div>
              <label class="toggle">
                <input
                  type="checkbox"
                  checked={agentConfigs[selectedAgent]?.permissions?.memory ?? true}
                  onchange={(e) => updatePermission('memory', e.currentTarget.checked)}
                />
                <span class="toggle-slider"></span>
              </label>
            </div>
          </div>
        </div>

        <!-- Save Button -->
        <div class="save-section">
          <button class="btn primary save-btn" onclick={saveAgentConfig}>
            <Save size={14} />
            Save Agent Configuration
          </button>
        </div>
          </div>
      {/if}
    </section>
  </div>
</div>

<style>
  .agents-workspace {
    display: grid;
    grid-template-columns: minmax(240px, 300px) minmax(0, 1fr);
    gap: 18px;
    align-items: start;
  }

  .agent-browser {
    display: flex;
    flex-direction: column;
    position: sticky;
    top: 0;
    max-height: min(78vh, 1040px);
  }

  .agents-content {
    min-width: 0;
  }

  /* Agent Selector Tabs */
  .agent-selector-tabs {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 16px;
    background: color-mix(in srgb, var(--bg-secondary) 90%, transparent);
    border: 1px solid var(--border-subtle);
    border-radius: 18px;
    overflow: auto;
  }

  .agent-tab {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    width: 100%;
    padding: 12px 14px;
    background: color-mix(in srgb, var(--agent-surface, var(--bg-tertiary)) 76%, var(--bg-tertiary));
    border: 1px solid var(--agent-border, var(--border-subtle));
    border-radius: 12px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
  }

  .agent-tab:hover {
    background: var(--bg-primary);
    border-color: var(--agent-accent, var(--accent-secondary));
    color: var(--text-primary);
  }

  .agent-tab.active {
    background: color-mix(in srgb, var(--agent-accent, var(--accent-primary)) 18%, var(--bg-primary));
    border-color: var(--agent-accent, var(--accent-primary));
    color: var(--text-primary);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--agent-accent, var(--accent-primary)) 32%, transparent);
  }

  .agent-tab-copy {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
    flex: 1;
  }

  .agent-tab .agent-name {
    font-weight: 500;
  }

  .agent-role {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.35;
  }

  .agent-tab .agent-dept {
    font-size: 11px;
    padding: 2px 6px;
    background: color-mix(in srgb, var(--agent-accent, var(--accent-secondary)) 18%, transparent);
    border-radius: 999px;
    white-space: nowrap;
  }

  .agent-tab.active .agent-dept {
    background: color-mix(in srgb, var(--agent-accent, var(--accent-primary)) 20%, transparent);
    color: var(--text-primary);
  }

  .agent-tab.floor {
    background: linear-gradient(180deg, color-mix(in srgb, var(--agent-accent, var(--accent-primary)) 12%, var(--bg-tertiary)), var(--bg-tertiary));
  }

  /* Agent Groups */
  .agent-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
    padding: 12px;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-primary) 86%, transparent);
  }

  .agent-group-header {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 0 0 4px;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .agent-tab.subagent {
    padding: 10px 12px;
    font-size: 12px;
  }

  /* Visual Config Editor */
  .agent-config-editor {
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .agent-hero {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    padding: 16px 18px;
    border: 1px solid var(--agent-border, var(--border-subtle));
    border-radius: 14px;
    background: linear-gradient(180deg, color-mix(in srgb, var(--agent-accent, var(--accent-primary)) 10%, var(--bg-primary)), var(--bg-primary));
  }

  .agent-hero h4 {
    margin: 2px 0 6px;
    font-size: 18px;
    color: var(--text-primary);
  }

  .agent-hero-role {
    margin: 0;
    color: var(--text-secondary);
    font-size: 13px;
  }

  .agent-hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
  }

  .meta-pill {
    padding: 6px 10px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--agent-accent, var(--accent-primary)) 18%, transparent);
    color: var(--text-primary);
    font-size: 11px;
    font-weight: 600;
  }

  .meta-pill.subtle {
    background: color-mix(in srgb, var(--bg-tertiary) 90%, transparent);
    color: var(--text-secondary);
  }

  .editor-top-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.7fr) minmax(300px, 0.8fr);
    gap: 18px;
    align-items: start;
  }

  .meta-column {
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .setting-group {
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px;
  }

  .prompt-group {
    background: color-mix(in srgb, var(--agent-surface, var(--bg-primary)) 52%, var(--bg-primary));
    border-color: var(--agent-border, var(--border-subtle));
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
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-subtle);
  }

  .setting-row > span {
    min-width: 120px;
    color: var(--text-secondary);
  }

  .setting-row > :global(select),
  .setting-row > :global(input) {
    flex: 1;
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

  .prompt-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .prompt-title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .prompt-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .saved-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--accent-success, #4caf50);
    font-weight: 500;
  }

  .override-badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--accent-warning, #ff9800);
    color: #000;
    font-weight: 600;
  }

  .default-badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--bg-tertiary);
    color: var(--text-muted);
    font-weight: 500;
  }

  .count-badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    background: var(--accent-primary, #3b82f6);
    color: #fff;
    font-weight: 600;
    margin-left: 4px;
  }

  .slash-cmd {
    font-size: 10px;
    padding: 1px 4px;
    border-radius: 3px;
    background: var(--bg-tertiary);
    color: var(--accent-primary, #3b82f6);
    font-family: monospace;
  }

  .skill-desc {
    color: var(--text-muted);
    font-size: 11px;
    line-height: 1.3;
  }

  .mcp-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }

  .mcp-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    font-size: 11px;
    color: var(--text-secondary);
    font-family: monospace;
  }

  .btn.small {
    padding: 3px 8px;
    font-size: 11px;
  }

  .prompt-editor {
    margin-bottom: 8px;
  }

  .prompt-textarea {
    width: 100%;
    min-height: 560px;
    padding: 16px;
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

  /* Permissions */
  .permissions-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
  }

  .permission-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 12px;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  .permission-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .permission-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .permission-desc {
    font-size: 10px;
    color: var(--text-muted);
  }

  .permission-levels {
    display: flex;
    gap: 4px;
  }

  .level-btn {
    padding: 4px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .level-btn:hover {
    border-color: var(--level-color);
    color: var(--text-primary);
  }

  .level-btn.active {
    background: var(--level-color);
    border-color: var(--level-color);
    color: white;
  }

  .boolean-permissions {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }

  .boolean-perm-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  /* Toggle Switch */
  .toggle {
    position: relative;
    display: inline-block;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
  }

  .toggle input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    transition: 0.2s;
  }

  .toggle-slider:before {
    position: absolute;
    content: "";
    height: 14px;
    width: 14px;
    left: 2px;
    bottom: 2px;
    background: var(--text-muted);
    border-radius: 50%;
    transition: 0.2s;
  }

  .toggle input:checked + .toggle-slider {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
  }

  .toggle input:checked + .toggle-slider:before {
    transform: translateX(16px);
    background: white;
  }

  /* Save Section */
  .save-section {
    display: flex;
    justify-content: flex-end;
    padding: 16px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .save-btn {
    padding: 10px 20px;
    background: var(--accent-primary);
    color: white;
    font-weight: 500;
  }

  .save-btn:hover {
    opacity: 0.9;
  }

  @media (max-width: 1100px) {
    .agents-workspace {
      grid-template-columns: 1fr;
    }

    .agent-browser {
      position: static;
      max-height: none;
    }

    .editor-top-grid {
      grid-template-columns: 1fr;
    }
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
