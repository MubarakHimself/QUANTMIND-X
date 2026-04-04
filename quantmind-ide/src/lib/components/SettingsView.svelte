<script lang="ts">
  import { run } from 'svelte/legacy';
  import { createEventDispatcher } from 'svelte';
  import { tick } from 'svelte';
  import { onMount, onDestroy } from 'svelte';
  import {
    Settings as SettingsIcon, Key, Server, Bot, Database, Sliders,
    Save, RefreshCw, Plus, X, Check, AlertCircle, ChevronRight,
    Shield, Wallet, TrendingUp, Zap, Globe, Lock, User, Bell,
    Eye, EyeOff, Trash2, Edit3, Download, Upload as UploadIcon, Upload,
    FolderOpen, Code, Terminal, Cpu, HardDrive,
    Brain, Sparkles, FileText, Cpu as ModelIcon, Activity, Heart,
    Palette
  } from 'lucide-svelte';
  import {
    ApiKeysPanel,
    McpServersPanel,
    AgentsPanel,
    RiskPanel,
    ProvidersPanel,
    NotificationSettingsPanel,
    ServerHealthPanel,
    AppearancePanel,
    DeployPanel
  } from './settings';
  import { apiFetch, buildApiUrl, getRouterSettings, saveRouterSettings } from '$lib/api';

  const dispatch = createEventDispatcher();

  // Settings tabs
  type SettingsTab = 'appearance' | 'providers' | 'notifications' | 'server-health' | 'mcp-servers' | 'agents' | 'risk' | 'deploy';
  let activeTab: SettingsTab = $state('appearance');
  let settingsVisible = $state(false);
  let settingsPanelEl = $state<HTMLElement | null>(null);
  let lastFocusedElement = $state<HTMLElement | null>(null);

  // General settings
  let generalSettings = $state({
    theme: 'dark' as 'light' | 'dark' | 'auto',
    language: 'en',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    autoSave: true,
    autoSaveInterval: 30,
    debugMode: false,
    logLevel: 'info' as 'debug' | 'info' | 'warn' | 'error'
  });

  // API Keys
  let apiKeys: Array<{
    id: string;
    name: string;
    key: string;
    service: string;
    created: string;
    lastUsed?: string;
  }> = $state([]);

  let apiKeyModal = $state(false);
  let newApiKey = $state({
    name: '',
    key: '',
    service: 'openai'
  });

  // MCP Servers
  let mcpServers: Array<{
    id: string;
    name: string;
    command: string;
    args: string[];
    status: 'running' | 'stopped' | 'error';
    type: 'builtin' | 'custom';
    description?: string;
  }> = $state([
    {
      id: 'context7',
      name: 'Context7 MCP',
      command: 'npx',
      args: ['-y', '@context7/mcp-server'],
      status: 'stopped',
      type: 'builtin',
      description: 'MQL5 documentation retrieval and code context'
    },
    {
      id: 'filesystem',
      name: 'Filesystem MCP',
      command: 'npx',
      args: ['-y', '@anthropic-ai/mcp-server-filesystem', '--root', './workspace'],
      status: 'stopped',
      type: 'builtin',
      description: 'Local filesystem access and file operations'
    },
    {
      id: 'metatrader5',
      name: 'MetaTrader 5 MCP',
      command: 'npx',
      args: ['-y', '@anthropic-ai/mcp-server-mt5'],
      status: 'stopped',
      type: 'builtin',
      description: 'MetaTrader 5 trading platform integration'
    },
    {
      id: 'sequential_thinking',
      name: 'Sequential Thinking MCP',
      command: 'npx',
      args: ['-y', '@anthropic-ai/mcp-server-sequential-thinking'],
      status: 'stopped',
      type: 'builtin',
      description: 'Task decomposition and reasoning'
    },
    {
      id: 'svelte',
      name: 'Svelte MCP',
      command: 'npx',
      args: ['-y', '@sveltejs/mcp'],
      status: 'stopped',
      type: 'builtin',
      description: 'Svelte component development tools'
    },
    {
      id: 'chrome_devtools',
      name: 'Chrome DevTools MCP',
      command: 'npx',
      args: ['-y', 'chrome-devtools-mcp@latest'],
      status: 'stopped',
      type: 'builtin',
      description: 'Browser automation and testing'
    }
  ]);

  let mcpModalOpen = $state(false);
  let newMcpServer = $state({
    name: '',
    command: '',
    args: '',
    description: ''
  });

  // Agent settings - all agents from AgentsPanel
  let agentConfigs: Record<string, {
    name: string;
    role: string;
    provider: string;
    model: string;
    temperature: number;
    maxTokens: number;
    systemPrompt: string;
    skills: Array<{ id: string; name: string; description: string; enabled: boolean }>;
    tools: string[];
    permissions?: {
      fileSystem: 'none' | 'read' | 'write' | 'full';
      broker: 'none' | 'read' | 'write' | 'full';
      database: 'none' | 'read' | 'write' | 'full';
      external: boolean;
      memory: boolean;
    };
  }> = $state({
    // Floor Manager
    floor_manager: {
      name: 'Floor Manager',
      role: 'Trading Floor Orchestrator',
      provider: '',
      model: '',
      temperature: 0.7,
      maxTokens: 8192,
      systemPrompt: 'You are the Floor Manager, orchestrating the trading floor operations...',
      skills: [],
      tools: ['get_market_data', 'store_semantic_memory', 'search_semantic_memories'],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    // Department Heads
    research_head: {
      name: 'Research Head',
      role: 'Strategy Research Lead',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 8192,
      systemPrompt: 'You are the Research Head, leading the research department...',
      skills: [],
      tools: ['get_market_data', 'run_backtest', 'store_semantic_memory'],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    development_head: {
      name: 'Development Head',
      role: 'EA Development Lead',
      provider: '',
      model: '',
      temperature: 0.3,
      maxTokens: 8192,
      systemPrompt: 'You are the Development Head, leading the development department...',
      skills: [],
      tools: ['get_market_data', 'run_backtest', 'store_semantic_memory'],
      permissions: { fileSystem: 'write', broker: 'read', database: 'read', external: false, memory: true }
    },
    trading_head: {
      name: 'Trading Head',
      role: 'Order Execution Lead',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 8192,
      systemPrompt: 'You are the Trading Head, leading the trading department...',
      skills: [],
      tools: ['get_market_data', 'get_position_size', 'store_semantic_memory'],
      permissions: { fileSystem: 'read', broker: 'write', database: 'read', external: false, memory: true }
    },
    risk_head: {
      name: 'Risk Head',
      role: 'Risk Management Lead',
      provider: '',
      model: '',
      temperature: 0.4,
      maxTokens: 8192,
      systemPrompt: 'You are the Risk Head, leading the risk management department...',
      skills: [],
      tools: ['get_market_data', 'get_position_size', 'store_semantic_memory'],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    portfolio_head: {
      name: 'Portfolio Head',
      role: 'Portfolio Management Lead',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 8192,
      systemPrompt: 'You are the Portfolio Head, leading the portfolio management department...',
      skills: [],
      tools: ['get_market_data', 'store_semantic_memory', 'search_semantic_memories'],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    // Sub-agents
    strategy_researcher: {
      name: 'Strategy Researcher',
      role: 'Research sub-agent',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 4096,
      systemPrompt: 'You are a Strategy Researcher...',
      skills: [],
      tools: ['get_market_data', 'run_backtest'],
      permissions: { fileSystem: 'read', broker: 'none', database: 'read', external: false, memory: true }
    },
    market_analyst: {
      name: 'Market Analyst',
      role: 'Research sub-agent',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 4096,
      systemPrompt: 'You are a Market Analyst...',
      skills: [],
      tools: ['get_market_data'],
      permissions: { fileSystem: 'read', broker: 'none', database: 'read', external: false, memory: true }
    },
    backtester: {
      name: 'Backtester',
      role: 'Research sub-agent',
      provider: '',
      model: '',
      temperature: 0.3,
      maxTokens: 4096,
      systemPrompt: 'You are a Backtester...',
      skills: [],
      tools: ['run_backtest'],
      permissions: { fileSystem: 'read', broker: 'none', database: 'read', external: false, memory: true }
    },
    python_dev: {
      name: 'Python Developer',
      role: 'Development sub-agent',
      provider: '',
      model: '',
      temperature: 0.3,
      maxTokens: 4096,
      systemPrompt: 'You are a Python Developer...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'write', broker: 'none', database: 'read', external: false, memory: true }
    },
    pinescript_dev: {
      name: 'PineScript Developer',
      role: 'Development sub-agent',
      provider: '',
      model: '',
      temperature: 0.3,
      maxTokens: 4096,
      systemPrompt: 'You are a PineScript Developer...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'write', broker: 'none', database: 'read', external: false, memory: true }
    },
    mql5_dev: {
      name: 'MQL5 Developer',
      role: 'Development sub-agent',
      provider: '',
      model: '',
      temperature: 0.3,
      maxTokens: 4096,
      systemPrompt: 'You are an MQL5 Developer...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'write', broker: 'none', database: 'read', external: false, memory: true }
    },
    order_executor: {
      name: 'Order Executor',
      role: 'Trading sub-agent',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 4096,
      systemPrompt: 'You are an Order Executor...',
      skills: [],
      tools: ['get_position_size'],
      permissions: { fileSystem: 'read', broker: 'write', database: 'read', external: false, memory: true }
    },
    fill_tracker: {
      name: 'Fill Tracker',
      role: 'Trading sub-agent',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 4096,
      systemPrompt: 'You are a Fill Tracker...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    trade_monitor: {
      name: 'Trade Monitor',
      role: 'Trading sub-agent',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 4096,
      systemPrompt: 'You are a Trade Monitor...',
      skills: [],
      tools: ['get_market_data', 'get_position_size'],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    position_sizer: {
      name: 'Position Sizer',
      role: 'Risk sub-agent',
      provider: '',
      model: '',
      temperature: 0.4,
      maxTokens: 4096,
      systemPrompt: 'You are a Position Sizer...',
      skills: [],
      tools: ['get_position_size'],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    drawdown_monitor: {
      name: 'Drawdown Monitor',
      role: 'Risk sub-agent',
      provider: '',
      model: '',
      temperature: 0.4,
      maxTokens: 4096,
      systemPrompt: 'You are a Drawdown Monitor...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    var_calculator: {
      name: 'VaR Calculator',
      role: 'Risk sub-agent',
      provider: '',
      model: '',
      temperature: 0.4,
      maxTokens: 4096,
      systemPrompt: 'You are a VaR Calculator...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    allocation_manager: {
      name: 'Allocation Manager',
      role: 'Portfolio sub-agent',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 4096,
      systemPrompt: 'You are an Allocation Manager...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    rebalancer: {
      name: 'Rebalancer',
      role: 'Portfolio sub-agent',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 4096,
      systemPrompt: 'You are a Rebalancer...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    },
    performance_tracker: {
      name: 'Performance Tracker',
      role: 'Portfolio sub-agent',
      provider: '',
      model: '',
      temperature: 0.5,
      maxTokens: 4096,
      systemPrompt: 'You are a Performance Tracker...',
      skills: [],
      tools: [],
      permissions: { fileSystem: 'read', broker: 'read', database: 'read', external: false, memory: true }
    }
  });

  let selectedAgent = $state('floor_manager');
  let showRawEditor = $state(false);

  // AGENTS.md content
  let agentsMdContent = $state('');
  let agentsMdLoading = $state(false);
  let agentsMdSaved = $state(false);

  // Providers
  let providers: Array<{
    id: string;
    name: string;
    base_url: string;
    api_key?: string;
  }> = $state([]);

  // Risk Management settings
  let riskSettings = $state({
    houseMoneyEnabled: true,
    houseMoneyThreshold: 0.5,
    dailyLossLimit: 5,
    maxDrawdown: 10,
    riskMode: 'dynamic' as 'fixed' | 'dynamic' | 'conservative',
    propFirmPreset: 'custom' as 'ftmo' | 'the5ers' | 'fundingpips' | 'custom',
    balanceZones: {
      danger: 200,
      growth: 1000,
      scaling: 5000,
      guardian: Infinity
    },
    maxRiskPerTrade: 0.05
  });

  // Router settings
  let routerSettings = {
    active: true,
    mode: 'auction' as 'auction' | 'priority' | 'round-robin',
    auctionInterval: 5000
  };

  // Database settings
  let dbSettings = $state({
    connectionType: 'sqlite',
    databaseUrl: '',
    sqlitePath: './data/quantmind.db',
    duckdbPath: './data/analytics.duckdb',
    autoBackup: true,
    backupInterval: 3600,
    maxBackups: 10
  });

  // Connection settings
  let connectionSettings = $state({
    redisUrl: 'redis://localhost:6379',
    zmqEndpoint: 'tcp://localhost:5555',
    mt5Login: '',
    mt5Password: '',
    mt5Server: ''
  });

  // Security settings
  let securitySettings = $state({
    secretKeyConfigured: false,
    secretKeyPrefix: ''
  });

  // Model config
  type ModelProvider = 'anthropic' | 'zhipu' | 'minimax' | 'openai' | 'deepseek' | 'openrouter';
  const providerBaseUrls: Record<string, string> = {
    anthropic: '',
    zhipu: 'https://api.z.ai/api/coding/paas/v4',
    minimax: 'https://api.minimax.io/anthropic',
    openai: 'https://api.openai.com/v1',
    deepseek: 'https://api.deepseek.com/v1',
    openrouter: 'https://openrouter.ai/api/v1'
  };
  let modelConfig = $state({
    provider: '' as ModelProvider,
    baseUrl: '',
    apiKey: '',
    model: '',
    availableModels: [] as Array<{ id: string; name: string; tier: string }>,
    showApiKey: false,
    isSaving: false,
    isLoading: false
  });

  const AGENT_MODEL_CANONICAL_MAP: Record<string, string> = {
    floor_manager: 'floor_manager',
    research_head: 'research',
    development_head: 'development',
    trading_head: 'trading',
    risk_head: 'risk',
    portfolio_head: 'portfolio',
  };

  function getCanonicalAgentModelId(agentId: string): string | null {
    return AGENT_MODEL_CANONICAL_MAP[agentId] || null;
  }

  function applyModelConfigToAgentEntries(models: Record<string, { provider?: ModelProvider; model?: string }>) {
    for (const [agentId, canonicalId] of Object.entries(AGENT_MODEL_CANONICAL_MAP)) {
      const runtimeModel = models[canonicalId];
      if (!runtimeModel || !agentConfigs[agentId]) {
        continue;
      }
      agentConfigs[agentId] = {
        ...agentConfigs[agentId],
        provider: runtimeModel.provider || '',
        model: runtimeModel.model || '',
      };
    }
  }

  // Handlers for API Keys
  async function addApiKey() {
    if (!newApiKey.name || !newApiKey.key) return;

    const apiKey: typeof apiKeys[0] = {
      id: Date.now().toString(),
      name: newApiKey.name,
      key: newApiKey.key,
      service: newApiKey.service,
      created: new Date().toISOString()
    };

    try {
      const data = await apiFetch<typeof apiKey>('/settings/keys', {
        method: 'POST',
        body: JSON.stringify(apiKey)
      });
      apiKeys = [...apiKeys, data];
      newApiKey = { name: '', key: '', service: 'openai' };
      apiKeyModal = false;
    } catch (e) {
      console.error('Failed to add API key:', e);
    }
  }

  async function removeApiKey(id: string) {
    apiKeys = apiKeys.filter(k => k.id !== id);

    try {
      await fetch(buildApiUrl(`/api/settings/keys/${id}`), {
        method: 'DELETE',
        credentials: 'include'
      });
    } catch (e) {
      console.error('Failed to remove API key:', e);
    }
  }

  // Handlers for Providers
  async function loadProviders() {
    try {
      const data = await apiFetch<{ providers?: typeof providers }>('/providers');
      providers = data.providers || [];
    } catch (e) {
      console.error('Failed to load providers:', e);
    }
  }

  async function handleSaveProvider(event: CustomEvent) {
    const provider = event.detail;
    try {
      await apiFetch('/providers', {
        method: 'POST',
        body: JSON.stringify(provider)
      });
      await loadProviders();
    } catch (e) {
      console.error('Failed to save provider:', e);
    }
  }

  async function handleDeleteProvider(event: CustomEvent) {
    const { providerId } = event.detail;
    try {
      const res = await fetch(buildApiUrl(`/api/providers/${providerId}`), {
        method: 'DELETE',
        credentials: 'include'
      });
      if (res.ok) {
        await loadProviders();
      }
    } catch (e) {
      console.error('Failed to delete provider:', e);
    }
  }

  // Handlers for MCP Servers
  async function addMcpServer() {
    if (!newMcpServer.name || !newMcpServer.command) return;

    const server: typeof mcpServers[0] = {
      id: Date.now().toString(),
      name: newMcpServer.name,
      command: newMcpServer.command,
      args: newMcpServer.args.split(' ').filter(a => a),
      status: 'stopped',
      type: 'custom',
      description: newMcpServer.description
    };

    try {
      const data = await apiFetch<typeof server>('/settings/mcp', {
        method: 'POST',
        body: JSON.stringify(server)
      });
      mcpServers = [...mcpServers, data];
      newMcpServer = { name: '', command: '', args: '', description: '' };
      mcpModalOpen = false;
    } catch (e) {
      console.error('Failed to add MCP server:', e);
    }
  }

  async function toggleMcpServer(id: string) {
    const server = mcpServers.find(s => s.id === id);
    if (!server) return;

    const newStatus = server.status === 'running' ? 'stopped' : 'running';

    try {
      await fetch(buildApiUrl(`/api/settings/mcp/${id}`), {
        method: 'PATCH',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      });

      server.status = newStatus;
    } catch (e) {
      console.error('Failed to toggle MCP server:', e);
    }
  }

  async function removeMcpServer(id: string) {
    mcpServers = mcpServers.filter(s => s.id !== id);

    try {
      await fetch(buildApiUrl(`/api/settings/mcp/${id}`), {
        method: 'DELETE',
        credentials: 'include'
      });
    } catch (e) {
      console.error('Failed to remove MCP server:', e);
    }
  }

  // Handlers for Agents
  async function loadAgentsMd() {
    try {
      const data = await apiFetch<{ content?: string }>('/settings/agents-md');
      agentsMdContent = data.content || '';
    } catch (e) {
      console.error('Failed to load AGENTS.md:', e);
      agentsMdContent = '# Agent Configuration\n\nConfigure your agent behavior here.';
    }
  }

  async function saveAgentsMd() {
    agentsMdLoading = true;
    try {
      await apiFetch('/settings/agents-md', {
        method: 'POST',
        body: JSON.stringify({ content: agentsMdContent })
      });
      agentsMdSaved = true;
      setTimeout(() => agentsMdSaved = false, 2000);
    } catch (e) {
      console.error('Failed to save AGENTS.md:', e);
    } finally {
      agentsMdLoading = false;
    }
  }

  function exportAgentsMd() {
    const blob = new Blob([agentsMdContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'AGENTS.md';
    a.click();
    URL.revokeObjectURL(url);
  }

  function importAgentsMd() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.md';
    input.onchange = async (e) => {
      const target = e.target;
      const file = target.files?.[0];
      if (file) {
        const content = await file.text();
        agentsMdContent = content;
        await saveAgentsMd();
      }
    };
    input.click();
  }

  function handleAgentSelect(event: CustomEvent) {
    selectedAgent = event.detail.agent;
  }

  function handleAgentConfigUpdate(event: CustomEvent) {
    const { agent, field, value } = event.detail;
    if (agentConfigs[agent]) {
      (agentConfigs[agent] as any)[field] = value;
    }
  }

  // Handle saving agent configuration to backend
  async function handleSaveAgentConfig(agent: string, config: Record<string, unknown>) {
    try {
      const requests: Promise<Response>[] = [];
      const canonicalModelAgent = getCanonicalAgentModelId(agent);
      const provider = typeof config.provider === 'string' ? config.provider : '';
      const model = typeof config.model === 'string' ? config.model : '';
      const systemPrompt = typeof config.systemPrompt === 'string' ? config.systemPrompt : '';

      if (canonicalModelAgent && (provider || model)) {
        requests.push(
          fetch(buildApiUrl(`/api/agent-config/${canonicalModelAgent}/model`), {
            method: 'PATCH',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, model })
          })
        );
      }

      if (systemPrompt) {
        requests.push(
          fetch(buildApiUrl(`/api/settings/agents/${agent}/system-prompt`), {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ system_prompt: systemPrompt })
          })
        );
      }

      if (requests.length === 0) {
        return;
      }

      const results = await Promise.all(requests);
      if (results.some((res) => !res.ok)) {
        console.error('Failed to save agent config for', agent);
      }
    } catch (e) {
      console.error('Failed to save agent config:', e);
    }
  }

  // Handlers for Risk
  function handleRiskUpdate(event: CustomEvent) {
    const { field, value } = event.detail;
    (riskSettings as any)[field] = value;
  }

  // Handlers for Database
  function handleDbUpdate(event: CustomEvent) {
    const { field, value } = event.detail;
    (dbSettings as any)[field] = value;
  }

  // Handlers for Connection
  function handleConnectionUpdate(event: CustomEvent) {
    const { field, value } = event.detail;
    (connectionSettings as any)[field] = value;
  }

  // Handlers for Model Config
  function handleProviderChange() {
    // Set default base URL for the selected provider
    modelConfig.baseUrl = providerBaseUrls[modelConfig.provider] || '';
    // Clear model selection when provider changes
    modelConfig.model = '';
  }

  async function loadModelConfig() {
    modelConfig.isLoading = true;
    try {
      if (!modelConfig.provider) {
        modelConfig.availableModels = [];
        return;
      }
      const data = await apiFetch<{ providers?: Record<string, { models?: Array<{ id: string; name: string; tier: string }> }> }>('/agent-config/available-models');
      const providerData = data.providers?.[modelConfig.provider];
      if (providerData?.models) {
        modelConfig.availableModels = providerData.models;
      } else {
        modelConfig.availableModels = [];
      }
    } catch (e) {
      console.error('Failed to load model config:', e);
    } finally {
      modelConfig.isLoading = false;
    }
  }

  async function saveModelConfig() {
    modelConfig.isSaving = true;
    try {
      // Save only to canonical department-era agents.
      const agents = ['floor_manager', 'research', 'development', 'trading', 'risk', 'portfolio'];
      const promises = agents.map(agent =>
        fetch(buildApiUrl(`/api/agent-config/${agent}/model`), {
          method: 'PATCH',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: modelConfig.model,
            provider: modelConfig.provider
          })
        })
      );

      const results = await Promise.all(promises);
      const allSucceeded = results.every(r => r.ok);

      if (allSucceeded) {
        dispatch('settingsSaved');
      }
    } catch (e) {
      console.error('Failed to save model config:', e);
    } finally {
      modelConfig.isSaving = false;
    }
  }

  // Handlers for General Settings
  function applyTheme() {
    const theme = localStorage.getItem('theme') || generalSettings.theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }

  function applyLanguage() {
    const language = localStorage.getItem('language') || generalSettings.language;
    document.documentElement.lang = language;
    localStorage.setItem('language', language);
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && settingsVisible) {
      hide();
    }
  }

  async function loadSettings() {
    try {
      generalSettings = { ...generalSettings, ...await apiFetch<typeof generalSettings>('/settings/general') };
      apiKeys = await apiFetch<typeof apiKeys>('/settings/keys');
      mcpServers = await apiFetch<typeof mcpServers>('/settings/mcp');
      riskSettings = { ...riskSettings, ...await apiFetch<typeof riskSettings>('/settings/risk') };

      // Load router settings
      try {
        const routerData = await getRouterSettings();
        routerSettings = { ...routerSettings, ...routerData };
      } catch (e) {
        console.error('Failed to load router settings:', e);
      }

      dbSettings = { ...dbSettings, ...await apiFetch<typeof dbSettings>('/settings/database') };

      // Load model config
      const data = await apiFetch<Record<string, { provider?: ModelProvider; model?: string }>>('/agent-config/models');
      applyModelConfigToAgentEntries(data);
      const managerModel = data.floor_manager;
      if (managerModel) {
        modelConfig.provider = managerModel.provider || '';
        modelConfig.model = managerModel.model || '';
        modelConfig.baseUrl = modelConfig.provider ? (providerBaseUrls[modelConfig.provider] || '') : '';
      }

      // Load available models
      await loadModelConfig();
    } catch (e) {
      console.error('Failed to load settings:', e);
    }
  }

  async function saveSettings() {
    try {
      await Promise.all([
        fetch(buildApiUrl('/api/settings/general'), {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(generalSettings)
        }),
        fetch(buildApiUrl('/api/settings/risk'), {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(riskSettings)
        }),
        saveRouterSettings(routerSettings)
      ]);

      dispatch('settingsSaved');
    } catch (e) {
      console.error('Failed to save settings:', e);
    }
  }

  function exportSettings() {
    const settings = {
      general: generalSettings,
      risk: riskSettings,
      database: dbSettings
    };

    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quantmind-settings-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  export function show() {
    const active = document.activeElement;
    lastFocusedElement = active instanceof HTMLElement ? active : null;
    settingsVisible = true;
    void loadSettings();
    void tick().then(() => {
      if (settingsPanelEl instanceof HTMLElement) {
        settingsPanelEl.focus();
      }
    });
  }

  export function hide() {
    const active = document.activeElement;
    if (active instanceof HTMLElement && active.closest('.settings-panel')) {
      active.blur();
    }
    settingsVisible = false;
    dispatch('close');
    void tick().then(() => {
      if (lastFocusedElement?.isConnected) {
        lastFocusedElement.focus();
      }
    });
  }

  function selectTab(tab: SettingsTab) {
    activeTab = tab;
  }

  // Reactive statements
  run(() => {
    if (generalSettings.theme && settingsVisible) {
      applyTheme();
    }
  });

  run(() => {
    if (generalSettings.language && settingsVisible) {
      applyLanguage();
    }
  });

  run(() => {
    if (generalSettings.timezone && settingsVisible) {
      localStorage.setItem('timezone', generalSettings.timezone);
    }
  });

  // Global Escape key handler
  let handleGlobalEscape: ((e: KeyboardEvent) => void) | null = null;

  onMount(() => {
    // Force-close any auto-opened dialog on mount — dialogs can open via browser API independently of our state
    const dialogEl = document.querySelector('.settings-overlay dialog');
    if (dialogEl && dialogEl instanceof HTMLDialogElement) {
      dialogEl.close();
    }
    settingsVisible = false;
    handleGlobalEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && settingsVisible) {
        e.preventDefault();
        hide();
      }
    };
    document.addEventListener('keydown', handleGlobalEscape);
  });

  onDestroy(() => {
    if (handleGlobalEscape) {
      document.removeEventListener('keydown', handleGlobalEscape);
    }
  });
</script>

<div
  class="settings-overlay"
  class:visible={settingsVisible}
  hidden={!settingsVisible}
  inert={!settingsVisible}
  onclick={hide}
>
  <div
    class="settings-panel"
    bind:this={settingsPanelEl}
    onclick={(event) => event.stopPropagation()}
    onkeydown={handleKeydown}
    tabindex={settingsVisible ? 0 : -1}
    role="dialog"
    aria-modal="false"
    aria-labelledby="settings-title"
  >
    <!-- Header -->
    <div class="settings-header">
      <div class="header-left">
        <SettingsIcon size={24} />
        <div>
          <h2 id="settings-title">Settings</h2>
          <p>Configure your QuantMind IDE</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="icon-btn" onclick={exportSettings} title="Export Settings">
          <Download size={18} />
        </button>
        <button class="icon-btn" onclick={loadSettings} title="Refresh">
          <RefreshCw size={18} />
        </button>
        <button class="icon-btn primary" onclick={saveSettings} title="Save Settings">
          <Save size={18} />
        </button>
        <button class="icon-btn" onclick={hide} title="Close">
          <X size={18} />
        </button>
      </div>
    </div>

    <div class="settings-content">
      <!-- Sidebar Tabs -->
      <div class="settings-tabs">
        <button
          type="button"
          class="tab"
          class:active={activeTab === 'appearance'}
          onclick={() => selectTab('appearance')}
        >
          <Palette size={16} />
          <span>Appearance</span>
        </button>
        <button
          type="button"
          class="tab"
          class:active={activeTab === 'providers'}
          onclick={() => selectTab('providers')}
        >
          <Key size={16} />
          <span>Providers</span>
        </button>
        <button
          type="button"
          class="tab"
          class:active={activeTab === 'notifications'}
          onclick={() => selectTab('notifications')}
        >
          <Bell size={16} />
          <span>Notifications</span>
        </button>
        <button
          type="button"
          class="tab"
          class:active={activeTab === 'server-health'}
          onclick={() => selectTab('server-health')}
        >
          <Activity size={16} />
          <span>Server Health</span>
        </button>
        <button
          type="button"
          class="tab"
          class:active={activeTab === 'mcp-servers'}
          onclick={() => selectTab('mcp-servers')}
        >
          <Server size={16} />
          <span>MCP Servers</span>
          {#if mcpServers.length > 0}
            <span class="badge">{mcpServers.length}</span>
          {/if}
        </button>
        <button
          type="button"
          class="tab"
          class:active={activeTab === 'agents'}
          onclick={() => selectTab('agents')}
        >
          <Bot size={16} />
          <span>Agents</span>
        </button>
        <button
          type="button"
          class="tab"
          class:active={activeTab === 'risk'}
          onclick={() => selectTab('risk')}
        >
          <Shield size={16} />
          <span>Risk</span>
        </button>
        <button
          type="button"
          class="tab"
          class:active={activeTab === 'deploy'}
          onclick={() => selectTab('deploy')}
        >
          <Upload size={16} />
          <span>Deploy</span>
        </button>
      </div>

      <!-- Settings Panels -->
      <div class="settings-panels">
        <!-- Appearance Settings -->
        {#if activeTab === 'appearance'}
          <div class="panel">
            <AppearancePanel />
          </div>
        {/if}

        <!-- Providers -->
        {#if activeTab === 'providers'}
          <ProvidersPanel />
        {/if}

        <!-- Notifications -->
        {#if activeTab === 'notifications'}
          <NotificationSettingsPanel />
        {/if}

        <!-- Server Health -->
        {#if activeTab === 'server-health'}
          <ServerHealthPanel />
        {/if}

        <!-- MCP Servers -->
        {#if activeTab === 'mcp-servers'}
          <McpServersPanel />
        {/if}

        <!-- Agents -->
        {#if activeTab === 'agents'}
          <AgentsPanel
            bind:agentConfigs
            bind:selectedAgent
            bind:showRawEditor
            bind:agentsMdContent
            bind:agentsMdLoading
            bind:agentsMdSaved
            onexportAgentsMd={exportAgentsMd}
            onimportAgentsMd={importAgentsMd}
            onsaveAgentsMd={saveAgentsMd}
            onselectAgent={(agent) => { selectedAgent = agent; }}
            onupdateAgentConfig={(agent, field, value) => {
              if (agentConfigs[agent]) {
                (agentConfigs[agent] as Record<string, unknown>)[field] = value;
              }
            }}
            onsaveAgentConfig={handleSaveAgentConfig}
          />
        {/if}

        <!-- Risk -->
        {#if activeTab === 'risk'}
          <RiskPanel
            bind:riskSettings
            on:updateRiskSettings={handleRiskUpdate}
          />
        {/if}

        <!-- Deploy -->
        {#if activeTab === 'deploy'}
          <DeployPanel />
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .settings-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
  }

  .settings-overlay.visible {
    opacity: 1;
    pointer-events: auto;
  }

  .settings-panel {
    background: var(--color-bg-surface);
    border-radius: 12px;
    width: 900px;
    max-width: 95vw;
    height: 85vh;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
    pointer-events: auto;
  }

  .settings-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    color: var(--color-text-primary);
  }

  .header-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  /* Header action buttons - better styling */
  :global(.settings-header) .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    color: var(--color-text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  :global(.settings-header) .icon-btn:hover {
    background: var(--color-bg-base);
    border-color: var(--color-border-medium);
    color: var(--color-text-primary);
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  }

  :global(.settings-header) .icon-btn.primary {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: white;
  }

  :global(.settings-header) .icon-btn.primary:hover {
    background: var(--color-accent-amber);
    border-color: var(--color-accent-amber);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
  }

  .settings-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .settings-tabs {
    width: 200px;
    padding: 16px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    border-right: 1px solid var(--color-border-subtle);
    background: var(--color-bg-base);
    overflow-y: auto;
  }

  .settings-tabs .tab {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--color-text-secondary);
    font-size: 13px;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
    position: relative;
  }

  .settings-tabs .tab:hover {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }

  .settings-tabs .tab.active {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .settings-tabs .badge {
    position: absolute;
    right: 8px;
    background: var(--color-bg-elevated);
    color: var(--color-text-muted);
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
  }

  .settings-tabs .tab.active .badge {
    background: rgba(255, 255, 255, 0.2);
    color: var(--color-bg-base);
  }

  .settings-panels {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  .panel {
    background: var(--color-bg-surface);
    border-radius: 8px;
  }

  .panel h3 {
    margin: 0 0 20px;
    font-size: 16px;
    color: var(--color-text-primary);
  }

  .panel h4 {
    margin: 20px 0 12px;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  /* Common styles for panels */
  .panel :global(.panel-header) {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .panel :global(.panel-header h3) {
    margin: 0;
  }

  .panel :global(.header-actions) {
    display: flex;
    gap: 8px;
  }

  .panel :global(.setting-group) {
    margin-bottom: 24px;
    padding: 16px;
    background: var(--color-bg-base);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
  }

  .panel :global(.setting-group label) {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: var(--color-text-muted);
    margin-bottom: 12px;
  }

  .panel :global(.setting-row) {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    font-size: 13px;
  }

  .panel :global(.setting-row span:first-child) {
    color: var(--color-text-primary);
  }

  .panel :global(.info-box) {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    margin-bottom: 16px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 8px;
    font-size: 12px;
    color: var(--color-text-secondary);
  }

  .panel :global(.btn) {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .panel :global(.btn.primary) {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .panel :global(.btn.secondary) {
    background: var(--color-bg-elevated);
    color: var(--color-text-secondary);
  }

  .panel :global(.icon-btn) {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--color-text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .panel :global(.icon-btn:hover) {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }

  .panel :global(.icon-btn.primary) {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .panel :global(.icon-btn.danger:hover) {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .panel :global(input[type="text"]),
  .panel :global(input[type="number"]),
  .panel :global(input[type="password"]),
  .panel :global(select),
  .panel :global(textarea) {
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 13px;
    font-family: inherit;
  }

  .panel :global(input[type="text"]),
  .panel :global(input[type="number"]),
  .panel :global(input[type="password"]) {
    width: 200px;
  }

  .panel :global(.text-input) {
    width: 100%;
  }

  .panel :global(.number-input) {
    width: 100px;
  }

  .panel :global(textarea) {
    width: 100%;
    min-height: 80px;
    resize: vertical;
  }

  .panel :global(.switch) {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
  }

  .panel :global(.switch input) {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .panel :global(.slider) {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--color-border-subtle);
    transition: 0.3s;
    border-radius: 24px;
  }

  .panel :global(.slider:before) {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    transition: 0.3s;
    border-radius: 50%;
  }

  .panel :global(input:checked + .slider) {
    background-color: var(--color-accent-cyan);
  }

  .panel :global(input:checked + .slider:before) {
    transform: translateX(20px);
  }

  .panel :global(.slider-input) {
    -webkit-appearance: none;
    width: 120px;
    height: 6px;
    background: var(--color-bg-elevated);
    border-radius: 3px;
    outline: none;
  }

  .panel :global(.slider-input::-webkit-slider-thumb) {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: var(--color-accent-cyan);
    border-radius: 50%;
    cursor: pointer;
  }

  .panel :global(.status-badge) {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
  }

  .panel :global(.status-badge.success) {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .panel :global(.status-badge.warning) {
    background: rgba(234, 179, 8, 0.2);
    color: #eab308;
  }

  .panel :global(.status-badge.running) {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .panel :global(.status-badge.stopped) {
    background: rgba(156, 163, 175, 0.2);
    color: #9ca3af;
  }

  .panel :global(.status-badge.error) {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .panel :global(.hint) {
    margin-top: 8px;
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .panel :global(.empty-state) {
    text-align: center;
    padding: 40px 20px;
    color: var(--color-text-muted);
  }

  .panel :global(.empty-state p) {
    margin: 12px 0;
  }

  /* Modal styles */
  .panel :global(.modal-overlay) {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1001;
  }

  .panel :global(.modal) {
    background: var(--color-bg-surface);
    border-radius: 12px;
    width: 480px;
    max-width: 90vw;
    max-height: 90vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .panel :global(.modal-header) {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .panel :global(.modal-header h3) {
    margin: 0;
    font-size: 16px;
  }

  .panel :global(.modal-header button) {
    background: none;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 4px;
  }

  .panel :global(.modal-body) {
    padding: 20px;
    overflow-y: auto;
  }

  .panel :global(.modal-footer) {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--color-border-subtle);
  }

  .panel :global(.form-group) {
    margin-bottom: 16px;
  }

  .panel :global(.form-group label) {
    display: block;
    font-size: 13px;
    font-weight: 500;
    color: var(--color-text-primary);
    margin-bottom: 6px;
  }

  .panel :global(.form-group input),
  .panel :global(.form-group select),
  .panel :global(.form-group textarea) {
    width: 100%;
  }

  .panel :global(.form-group small) {
    display: block;
    margin-top: 4px;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .panel :global(.divider) {
    text-align: center;
    margin: 20px 0;
    color: var(--color-text-muted);
    font-size: 12px;
  }

  .panel :global(.template-grid) {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }

  .panel :global(.template-card) {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .panel :global(.template-card:hover) {
    border-color: var(--color-accent-cyan);
    background: var(--color-bg-base);
  }

  .panel :global(.password-input-wrapper) {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .panel :global(.password-input-wrapper input) {
    flex: 1;
  }

  .panel :global(.form-actions) {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--color-border-subtle);
  }

  .panel :global(.spinning) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .model-assignment-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    margin-bottom: 8px;
  }

  .model-assignment-row .agent-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--color-text-primary);
  }

  .model-assignment-row .provider-model {
    font-size: 12px;
    color: var(--color-text-muted);
    font-family: monospace;
  }
</style>
