<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { onMount, onDestroy } from 'svelte';
  import {
    Settings as SettingsIcon, Key, Server, Bot, Database, Sliders,
    Save, RefreshCw, Plus, X, Check, AlertCircle, ChevronRight,
    Shield, Wallet, TrendingUp, Zap, Globe, Lock, User, Bell,
    Eye, EyeOff, Trash2, Edit3, Download, Upload as UploadIcon,
    FolderOpen, Code, Terminal, Cpu, HardDrive,
    Brain, Sparkles, FileText
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // AGENTS.md content
  let agentsMdContent = '';
  let agentsMdLoading = false;
  let agentsMdSaved = false;

  // Per-agent configuration editing
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
  }> = {
    copilot: {
      name: 'copilot',
      role: 'Trading Assistant & Workflow Guide',
      provider: 'openrouter',
      model: 'anthropic/claude-sonnet-4',
      temperature: 0.7,
      maxTokens: 4096,
      systemPrompt: '',
      skills: [],
      tools: []
    },
    quantcode: {
      name: 'quantcode',
      role: 'MQL5 Code Expert',
      provider: 'openrouter',
      model: 'anthropic/claude-sonnet-4',
      temperature: 0.3,
      maxTokens: 8192,
      systemPrompt: '',
      skills: [],
      tools: []
    },
    analyst: {
      name: 'analyst',
      role: 'Trading Strategy Analyst',
      provider: 'openrouter',
      model: 'anthropic/claude-sonnet-4',
      temperature: 0.5,
      maxTokens: 6144,
      systemPrompt: '',
      skills: [],
      tools: []
    }
  };

  let selectedAgent: string = 'copilot';
  let agentEditModal = false;
  let showRawEditor = true;

  // Settings tabs
  type SettingsTab = 'general' | 'api-keys' | 'mcp-servers' | 'agents' | 'risk' | 'database';
  let activeTab: SettingsTab = 'general';
  let settingsVisible = false;

  // General settings
  let generalSettings = {
    theme: 'dark' as 'light' | 'dark' | 'auto',
    language: 'en',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    autoSave: true,
    autoSaveInterval: 30, // seconds
    debugMode: false,
    logLevel: 'info' as 'debug' | 'info' | 'warn' | 'error'
  };

  // API Keys
  let apiKeys: Array<{
    id: string;
    name: string;
    key: string;
    service: string;
    created: string;
    lastUsed?: string;
  }> = [];

  let apiKeyModal = false;
  let newApiKey = {
    name: '',
    key: '',
    service: 'openai'
  };

  // MCP Servers
  let mcpServers: Array<{
    id: string;
    name: string;
    command: string;
    args: string[];
    status: 'running' | 'stopped' | 'error';
    type: 'builtin' | 'custom';
    description?: string;
  }> = [
    {
      id: 'context7',
      name: 'Context7',
      command: 'npx',
      args: ['-y', '@context7/mcp-server'],
      status: 'stopped',
      type: 'builtin',
      description: 'Documentation lookup for LangChain, LangGraph, and libraries'
    }
  ];

  let mcpModalOpen = false;
  let newMcpServer = {
    name: '',
    command: '',
    args: '',
    description: ''
  };

  // Agent settings
  let agentSettings = {
    defaultModel: 'claude-sonnet-4',
    temperature: 0.7,
    maxTokens: 4096,
    enableMemory: true,
    memoryType: 'hybrid' as 'short' | 'long' | 'hybrid',
    skillsEnabled: true,
    autoDelegate: false
  };

  let agentSkills: Array<{
    id: string;
    name: string;
    agent: 'copilot' | 'analyst' | 'quantcode';
    enabled: boolean;
    description: string;
  }> = [];

  // Risk Management settings
  let riskSettings = {
    houseMoneyEnabled: true,
    houseMoneyThreshold: 0.5, // 50% of daily profit
    dailyLossLimit: 5, // percentage
    maxDrawdown: 10, // percentage
    riskMode: 'dynamic' as 'fixed' | 'dynamic' | 'conservative',
    balanceZones: {
      danger: 200,
      growth: 1000,
      scaling: 5000,
      guardian: Infinity
    }
  };

  // Database settings
  let dbSettings = {
    sqlitePath: './data/quantmind.db',
    duckdbPath: './data/analytics.duckdb',
    autoBackup: true,
    backupInterval: 3600, // seconds
    maxBackups: 10
  };

  async function loadAgentsMd() {
    try {
      const res = await fetch('http://localhost:8000/api/settings/agents-md');
      if (res.ok) {
        const data = await res.json();
        agentsMdContent = data.content || '';
      }
    } catch (e) {
      console.error('Failed to load AGENTS.md:', e);
      // Set default content
      agentsMdContent = '# Agent Configuration\n\nConfigure your agent behavior here.';
    }
  }

  async function saveAgentsMd() {
    agentsMdLoading = true;
    try {
      const res = await fetch('http://localhost:8000/api/settings/agents-md', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: agentsMdContent })
      });

      if (res.ok) {
        agentsMdSaved = true;
        setTimeout(() => agentsMdSaved = false, 2000);
        // Reload agent configs after saving
        await loadAgentConfigs();
      } else {
        console.error('Failed to save AGENTS.md');
      }
    } catch (e) {
      console.error('Failed to save AGENTS.md:', e);
    } finally {
      agentsMdLoading = false;
    }
  }

  async function loadAgentConfigs() {
    try {
      const res = await fetch('http://localhost:8000/api/settings/agents-config');
      if (res.ok) {
        const data = await res.json();
        if (data.agents) {
          agentConfigs = data.agents;
        }
      }
    } catch (e) {
      console.error('Failed to load agent configs:', e);
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
      // Load general settings
      const generalRes = await fetch('http://localhost:8000/api/settings/general');
      if (generalRes.ok) {
        const data = await generalRes.json();
        generalSettings = { ...generalSettings, ...data };
      }

      // Load API keys
      const keysRes = await fetch('http://localhost:8000/api/settings/keys');
      if (keysRes.ok) {
        apiKeys = await keysRes.json();
      }

      // Load MCP servers
      const mcpRes = await fetch('http://localhost:8000/api/settings/mcp');
      if (mcpRes.ok) {
        mcpServers = await mcpRes.json();
      }

      // Load agent settings
      const agentRes = await fetch('http://localhost:8000/api/settings/agents');
      if (agentRes.ok) {
        const data = await agentRes.json();
        agentSettings = { ...agentSettings, ...data };
      }

      // Load agent skills
      const skillsRes = await fetch('http://localhost:8000/api/settings/skills');
      if (skillsRes.ok) {
        agentSkills = await skillsRes.json();
      }

      // Load risk settings
      const riskRes = await fetch('http://localhost:8000/api/settings/risk');
      if (riskRes.ok) {
        const data = await riskRes.json();
        riskSettings = { ...riskSettings, ...data };
      }

      // Load database settings
      const dbRes = await fetch('http://localhost:8000/api/settings/database');
      if (dbRes.ok) {
        const data = await dbRes.json();
        dbSettings = { ...dbSettings, ...data };
      }
    } catch (e) {
      console.error('Failed to load settings:', e);
    }
  }

  async function saveSettings() {
    try {
      // Save all settings
      await Promise.all([
        fetch('http://localhost:8000/api/settings/general', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(generalSettings)
        }),
        fetch('http://localhost:8000/api/settings/agents', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(agentSettings)
        }),
        fetch('http://localhost:8000/api/settings/risk', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(riskSettings)
        })
      ]);

      dispatch('settingsSaved');
    } catch (e) {
      console.error('Failed to save settings:', e);
    }
  }

  async function addApiKey() {
    if (!newApiKey.name || !newApiKey.key) return;

    const apiKey: typeof apiKeys[0] = {
      id: Date.now().toString(),
      name: newApiKey.name,
      key: newApiKey.key, // In production, this should be encrypted
      service: newApiKey.service,
      created: new Date().toISOString()
    };

    try {
      const res = await fetch('http://localhost:8000/api/settings/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiKey)
      });

      if (res.ok) {
        const data = await res.json();
        // Use the returned API key with server-generated ID
        apiKeys = [...apiKeys, data];
        newApiKey = { name: '', key: '', service: 'openai' };
        apiKeyModal = false;
      } else {
        throw new Error('Failed to add API key');
      }
    } catch (e) {
      console.error('Failed to add API key:', e);
    }
  }

  async function editApiKey(id: string, updatedKey: Partial<typeof apiKeys[0]>) {
    const index = apiKeys.findIndex(k => k.id === id);
    if (index === -1) return;

    const originalKey = apiKeys[index];
    const updated = { ...originalKey, ...updatedKey };

    try {
      const res = await fetch(`http://localhost:8000/api/settings/keys/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updated)
      });

      if (res.ok) {
        apiKeys[index] = updated;
        apiKeys = [...apiKeys];
      } else {
        throw new Error('Failed to update API key');
      }
    } catch (e) {
      console.error('Failed to edit API key:', e);
    }
  }

  async function removeApiKey(id: string) {
    apiKeys = apiKeys.filter(k => k.id !== id);

    try {
      await fetch(`http://localhost:8000/api/settings/keys/${id}`, {
        method: 'DELETE'
      });
    } catch (e) {
      console.error('Failed to remove API key:', e);
    }
  }

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
      const res = await fetch('http://localhost:8000/api/settings/mcp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(server)
      });

      if (res.ok) {
        const data = await res.json();
        // Use the returned server with server-generated ID
        mcpServers = [...mcpServers, data];
        newMcpServer = { name: '', command: '', args: '', description: '' };
        mcpModalOpen = false;
      } else {
        throw new Error('Failed to add MCP server');
      }
    } catch (e) {
      console.error('Failed to add MCP server:', e);
    }
  }

  async function editMcpServer(id: string, updatedServer: Partial<typeof mcpServers[0]>) {
    const index = mcpServers.findIndex(s => s.id === id);
    if (index === -1) return;

    const originalServer = mcpServers[index];
    const updated = { ...originalServer, ...updatedServer };

    try {
      const res = await fetch(`http://localhost:8000/api/settings/mcp/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updated)
      });

      if (res.ok) {
        mcpServers[index] = updated;
        mcpServers = [...mcpServers];
      } else {
        throw new Error('Failed to update MCP server');
      }
    } catch (e) {
      console.error('Failed to edit MCP server:', e);
    }
  }

  async function toggleMcpServer(id: string) {
    const server = mcpServers.find(s => s.id === id);
    if (!server) return;

    const newStatus = server.status === 'running' ? 'stopped' : 'running';

    try {
      await fetch(`http://localhost:8000/api/settings/mcp/${id}`, {
        method: 'PATCH',
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
      await fetch(`http://localhost:8000/api/settings/mcp/${id}`, {
        method: 'DELETE'
      });
    } catch (e) {
      console.error('Failed to remove MCP server:', e);
    }
  }

  async function toggleAgentSkill(skillId: string) {
    const skill = agentSkills.find(s => s.id === skillId);
    if (skill) {
      skill.enabled = !skill.enabled;

      // Sync with backend
      try {
        const res = await fetch(`http://localhost:8000/api/settings/skills/${skillId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: skill.enabled })
        });

        if (!res.ok) {
          throw new Error('Failed to toggle skill');
        }
      } catch (e) {
        console.error('Failed to toggle skill:', e);
        // Revert on error
        skill.enabled = !skill.enabled;
      }
    }
  }

  // Helper functions for skills management
  function getSkillsByAgent(agent: string) {
    return agentSkills.filter(s => s.agent === agent);
  }

  function getAgentDisplayName(agent: string) {
    const names: Record<string, string> = {
      copilot: 'Copilot',
      analyst: 'Analyst',
      quantcode: 'QuantCode'
    };
    return names[agent] || agent;
  }

  function allAgentSkillsEnabled(agent: string) {
    const skills = getSkillsByAgent(agent);
    return skills.length > 0 && skills.every(s => s.enabled);
  }

  function toggleAllAgentSkills(agent: string, enabled: boolean) {
    for (const skill of agentSkills) {
      if (skill.agent === agent) {
        skill.enabled = enabled;
        toggleAgentSkill(skill.id);
      }
    }
  }

  function testConnection(service: string) {
    // Test API connection
    console.log(`Testing connection to ${service}...`);
  }

  function exportSettings() {
    const settings = {
      general: generalSettings,
      agent: agentSettings,
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

  // Expose methods
  export function show() {
    settingsVisible = true;
  }

  export function hide() {
    settingsVisible = false;
    dispatch('close'); // Emit close event for parent component
  }

  // Apply theme changes reactively
  $: if (generalSettings.theme && settingsVisible) {
    applyTheme();
  }

  // Apply language changes reactively
  $: if (generalSettings.language && settingsVisible) {
    applyLanguage();
  }

  // Apply timezone changes reactively
  $: if (generalSettings.timezone && settingsVisible) {
    localStorage.setItem('timezone', generalSettings.timezone);
  }

  function getServiceIcon(service: string) {
    const icons: Record<string, typeof Cpu> = {
      openai: Brain,
      anthropic: Sparkles,
      gemini: Zap,
      openrouter: Globe,
      together: Server,
      groq: Cpu
    };
    return icons[service] || Key;
  }

  // Global Escape key handler for accessibility
  let handleGlobalEscape: ((e: KeyboardEvent) => void) | null = null;

  onMount(() => {
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

<div class="settings-overlay" class:visible={settingsVisible} on:click={() => hide()}>
  <div class="settings-panel" on:click|stopPropagation on:keydown={handleKeydown} tabindex="0" role="dialog" aria-modal="true" aria-labelledby="settings-title">
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
        <button class="icon-btn" on:click={exportSettings} title="Export Settings">
          <Download size={18} />
        </button>
        <button class="icon-btn" on:click={loadSettings} title="Refresh">
          <RefreshCw size={18} />
        </button>
        <button class="icon-btn primary" on:click={saveSettings} title="Save Settings">
          <Save size={18} />
        </button>
        <button class="icon-btn" on:click={hide} title="Close">
          <X size={18} />
        </button>
      </div>
    </div>

    <div class="settings-content">
      <!-- Sidebar Tabs -->
      <div class="settings-tabs">
        <button
          class="tab"
          class:active={activeTab === 'general'}
          on:click={() => activeTab = 'general'}
        >
          <Sliders size={16} />
          <span>General</span>
        </button>
        <button
          class="tab"
          class:active={activeTab === 'api-keys'}
          on:click={() => activeTab = 'api-keys'}
        >
          <Key size={16} />
          <span>API Keys</span>
          {#if apiKeys.length > 0}
            <span class="badge">{apiKeys.length}</span>
          {/if}
        </button>
        <button
          class="tab"
          class:active={activeTab === 'mcp-servers'}
          on:click={() => activeTab = 'mcp-servers'}
        >
          <Server size={16} />
          <span>MCP Servers</span>
          {#if mcpServers.length > 0}
            <span class="badge">{mcpServers.length}</span>
          {/if}
        </button>
        <button
          class="tab"
          class:active={activeTab === 'agents'}
          on:click={() => activeTab = 'agents'}
        >
          <Bot size={16} />
          <span>Agents</span>
        </button>
        <button
          class="tab"
          class:active={activeTab === 'risk'}
          on:click={() => activeTab = 'risk'}
        >
          <Shield size={16} />
          <span>Risk</span>
        </button>
        <button
          class="tab"
          class:active={activeTab === 'database'}
          on:click={() => activeTab = 'database'}
        >
          <Database size={16} />
          <span>Database</span>
        </button>
      </div>

      <!-- Settings Panels -->
      <div class="settings-panels">
        <!-- General Settings -->
        {#if activeTab === 'general'}
          <div class="panel">
            <h3>General Settings</h3>

            <div class="setting-group">
              <label>Appearance</label>
              <div class="setting-row">
                <span>Theme</span>
                <select bind:value={generalSettings.theme}>
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="auto">Auto (System)</option>
                </select>
              </div>
              <div class="setting-row">
                <span>Language</span>
                <select bind:value={generalSettings.language}>
                  <option value="en">English</option>
                  <option value="es">Español</option>
                  <option value="fr">Français</option>
                  <option value="de">Deutsch</option>
                </select>
              </div>
              <div class="setting-row">
                <span>Timezone</span>
                <select bind:value={generalSettings.timezone}>
                  <option value="UTC">UTC</option>
                  <option value="America/New_York">Eastern Time</option>
                  <option value="America/Chicago">Central Time</option>
                  <option value="America/Los_Angeles">Pacific Time</option>
                  <option value="Europe/London">London</option>
                  <option value="Asia/Tokyo">Tokyo</option>
                </select>
              </div>
            </div>

            <div class="setting-group">
              <label>Editor</label>
              <div class="setting-row">
                <span>Auto-save</span>
                <label class="switch">
                  <input type="checkbox" bind:checked={generalSettings.autoSave} />
                  <span class="slider"></span>
                </label>
              </div>
              {#if generalSettings.autoSave}
                <div class="setting-row">
                  <span>Interval (seconds)</span>
                  <input
                    type="number"
                    min="10"
                    max="300"
                    bind:value={generalSettings.autoSaveInterval}
                    class="number-input"
                  />
                </div>
              {/if}
            </div>

            <div class="setting-group">
              <label>System</label>
              <div class="setting-row">
                <span>Debug Mode</span>
                <label class="switch">
                  <input type="checkbox" bind:checked={generalSettings.debugMode} />
                  <span class="slider"></span>
                </label>
              </div>
              <div class="setting-row">
                <span>Log Level</span>
                <select bind:value={generalSettings.logLevel}>
                  <option value="debug">Debug</option>
                  <option value="info">Info</option>
                  <option value="warn">Warning</option>
                  <option value="error">Error</option>
                </select>
              </div>
            </div>
          </div>
        {/if}

        <!-- API Keys -->
        {#if activeTab === 'api-keys'}
          <div class="panel">
            <div class="panel-header">
              <h3>API Keys</h3>
              <button class="btn primary" on:click={() => apiKeyModal = true}>
                <Plus size={14} /> Add Key
              </button>
            </div>

            <div class="info-box">
              <AlertCircle size={16} />
              <span>Your API keys are stored locally and encrypted. Never share them with anyone.</span>
            </div>

            <div class="keys-list">
              {#each apiKeys as key}
                <div class="key-item">
                  <div class="key-icon">
                    <svelte:component this={getServiceIcon(key.service)} />
                  </div>
                  <div class="key-info">
                    <div class="key-name">{key.name}</div>
                    <div class="key-service">{key.service}</div>
                  </div>
                  <div class="key-value">
                    <code>{key.key.slice(0, 8)}...</code>
                  </div>
                  <div class="key-actions">
                    <button class="icon-btn" on:click={() => testConnection(key.service)} title="Test Connection">
                      <RefreshCw size={14} />
                    </button>
                    <button class="icon-btn danger" on:click={() => removeApiKey(key.id)} title="Remove">
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              {:else}
                <div class="empty-state">
                  <Key size={32} />
                  <p>No API keys configured</p>
                  <button class="btn primary" on:click={() => apiKeyModal = true}>
                    Add Your First API Key
                  </button>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- MCP Servers -->
        {#if activeTab === 'mcp-servers'}
          <div class="panel">
            <div class="panel-header">
              <h3>MCP Servers</h3>
              <button class="btn primary" on:click={() => mcpModalOpen = true}>
                <Plus size={14} /> Add Server
              </button>
            </div>

            <div class="info-box">
              <Server size={16} />
              <span>MCP servers extend agent capabilities with external tools and data sources.</span>
            </div>

            <div class="servers-list">
              {#each mcpServers as server}
                <div class="server-item">
                  <div class="server-icon">
                    <Terminal size={20} />
                  </div>
                  <div class="server-info">
                    <div class="server-name">{server.name}</div>
                    <div class="server-desc">{server.description || 'Custom MCP server'}</div>
                    <div class="server-command">
                      <code>{server.command} {server.args.join(' ')}</code>
                    </div>
                  </div>
                  <div class="server-status">
                    <span class="status-badge" class:running={server.status === 'running'} class:stopped={server.status === 'stopped'} class:error={server.status === 'error'}>
                      {server.status}
                    </span>
                  </div>
                  <div class="server-actions">
                    <button
                      class="icon-btn"
                      on:click={() => toggleMcpServer(server.id)}
                      title={server.status === 'running' ? 'Stop' : 'Start'}
                    >
                      {#if server.status === 'running'}
                        <EyeOff size={14} />
                      {:else}
                        <Eye size={14} />
                      {/if}
                    </button>
                    {#if server.type === 'custom'}
                      <button class="icon-btn danger" on:click={() => removeMcpServer(server.id)} title="Remove">
                        <Trash2 size={14} />
                      </button>
                    {/if}
                  </div>
                </div>
              {:else}
                <div class="empty-state">
                  <Server size={32} />
                  <p>No MCP servers configured</p>
                  <button class="btn primary" on:click={() => mcpModalOpen = true}>
                    Add MCP Server
                  </button>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Agents Settings -->
        {#if activeTab === 'agents'}
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
                on:click={() => selectedAgent = 'copilot'}
              >
                <Bot size={14} />
                <span>Copilot</span>
              </button>
              <button
                class="agent-tab"
                class:active={selectedAgent === 'quantcode'}
                on:click={() => selectedAgent = 'quantcode'}
              >
                <Code size={14} />
                <span>QuantCode</span>
              </button>
              <button
                class="agent-tab"
                class:active={selectedAgent === 'analyst'}
                on:click={() => selectedAgent = 'analyst'}
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
                on:click={() => showRawEditor = true}
              >
                <Terminal size={14} /> Raw Markdown
              </button>
              <button
                class="mode-btn"
                class:active={!showRawEditor}
                on:click={() => showRawEditor = false}
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
                      <input type="text" bind:value={agentConfigs[selectedAgent].role} class="text-input" />
                    </div>
                  </div>

                  <!-- Model Configuration -->
                  <div class="setting-group">
                    <label>Model Configuration</label>
                    <div class="setting-row">
                      <span>Provider</span>
                      <select bind:value={agentConfigs[selectedAgent].provider}>
                        <option value="openrouter">OpenRouter</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="zhipu">Zhipu AI</option>
                      </select>
                    </div>
                    <div class="setting-row">
                      <span>Model</span>
                      <input type="text" bind:value={agentConfigs[selectedAgent].model} class="text-input" placeholder="anthropic/claude-sonnet-4" />
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
                        placeholder="Enter the system prompt for this agentConfigs[selectedAgent]..."
                      ></textarea>
                    </div>
                    <small class="help-text">This prompt defines the agent's behavior and responsibilities.</small>
                  </div>

                  <!-- Skills -->
                  <div class="setting-group">
                    <label>
                      Skills
                      <button class="add-skill-btn" on:click={() => {
                        const newId = `custom-${Date.now()}`;
                        agentConfigs[selectedAgent].skills = [...agentConfigs[selectedAgent].skills, {
                          id: newId,
                          name: 'New Skill',
                          description: 'Add skill description',
                          enabled: true
                        }];
                      }}>
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
                              on:click={() => agentConfigs[selectedAgent].skills = agentConfigs[selectedAgent].skills.filter((_, i) => i !== index)}
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
                      {#each ['get_market_data', 'run_backtest', 'get_position_size', 'store_semantic_memory', 'search_semantic_memories'] as tool}
                        <label class="tool-checkbox">
                          <input type="checkbox" checked={agentConfigs[selectedAgent].tools.includes(tool)} on:change={(e) => {
                            if (e.target.checked) {
                              agentConfigs[selectedAgent].tools = [...agentConfigs[selectedAgent].tools, tool];
                            } else {
                              agentConfigs[selectedAgent].tools = agentConfigs[selectedAgent].tools.filter(t => t !== tool);
                            }
                          }} />
                          <code>{tool}</code>
                        </label>
                      {/each}
                    </div>
                  </div>
                </div>
              {/if}
            {/if}
          </div>
        {/if}

        <!-- Risk Management Settings -->
        {#if activeTab === 'risk'}
          <div class="panel">
            <h3>Risk Management</h3>

            <div class="setting-group">
              <label>House Money Effect</label>
              <div class="setting-row">
                <span>Enable House Money</span>
                <label class="switch">
                  <input type="checkbox" bind:checked={riskSettings.houseMoneyEnabled} />
                  <span class="slider"></span>
                </label>
              </div>
              <div class="setting-row">
                <span>Threshold (% of daily profit)</span>
                <input
                  type="number"
                  min="0"
                  max="100"
                  bind:value={riskSettings.houseMoneyThreshold}
                  class="number-input"
                />
              </div>
            </div>

            <div class="setting-group">
              <label>Risk Limits</label>
              <div class="setting-row">
                <span>Daily Loss Limit (%)</span>
                <input
                  type="number"
                  min="1"
                  max="20"
                  bind:value={riskSettings.dailyLossLimit}
                  class="number-input"
                />
              </div>
              <div class="setting-row">
                <span>Max Drawdown (%)</span>
                <input
                  type="number"
                  min="1"
                  max="50"
                  bind:value={riskSettings.maxDrawdown}
                  class="number-input"
                />
              </div>
            </div>

            <div class="setting-group">
              <label>Risk Mode</label>
              <div class="setting-row">
                <span>Mode</span>
                <select bind:value={riskSettings.riskMode}>
                  <option value="fixed">Fixed (constant risk)</option>
                  <option value="dynamic">Dynamic (adjusts to conditions)</option>
                  <option value="conservative">Conservative (protects capital)</option>
                </select>
              </div>
            </div>

            <div class="setting-group">
              <label>Balance Zones</label>
              <div class="zones-grid">
                <div class="zone-item danger">
                  <span class="zone-label">DANGER</span>
                  <span class="zone-amount">${riskSettings.balanceZones.danger}</span>
                </div>
                <div class="zone-item growth">
                  <span class="zone-label">GROWTH</span>
                  <span class="zone-amount">${riskSettings.balanceZones.growth}</span>
                </div>
                <div class="zone-item scaling">
                  <span class="zone-label">SCALING</span>
                  <span class="zone-amount">${riskSettings.balanceZones.scaling}</span>
                </div>
                <div class="zone-item guardian">
                  <span class="zone-label">GUARDIAN</span>
                  <span class="zone-amount">{riskSettings.balanceZones.guardian === Infinity ? '∞' : '$' + riskSettings.balanceZones.guardian}</span>
                </div>
              </div>
            </div>
          </div>
        {/if}

        <!-- Database Settings -->
        {#if activeTab === 'database'}
          <div class="panel">
            <h3>Database Configuration</h3>

            <div class="setting-group">
              <label>SQLite (Transactional)</label>
              <div class="setting-row">
                <span>Path</span>
                <input type="text" bind:value={dbSettings.sqlitePath} class="text-input" />
              </div>
            </div>

            <div class="setting-group">
              <label>DuckDB (Analytics)</label>
              <div class="setting-row">
                <span>Path</span>
                <input type="text" bind:value={dbSettings.duckdbPath} class="text-input" />
              </div>
            </div>

            <div class="setting-group">
              <label>Backup</label>
              <div class="setting-row">
                <span>Auto Backup</span>
                <label class="switch">
                  <input type="checkbox" bind:checked={dbSettings.autoBackup} />
                  <span class="slider"></span>
                </label>
              </div>
              {#if dbSettings.autoBackup}
                <div class="setting-row">
                  <span>Interval (seconds)</span>
                  <input
                    type="number"
                    min="300"
                    max="86400"
                    bind:value={dbSettings.backupInterval}
                    class="number-input"
                  />
                </div>
                <div class="setting-row">
                  <span>Max Backups</span>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    bind:value={dbSettings.maxBackups}
                    class="number-input"
                  />
                </div>
              {/if}
            </div>

            <div class="setting-group">
              <label>Storage Info</label>
              <div class="storage-info">
                <div class="storage-item">
                  <HardDrive size={16} />
                  <div class="storage-details">
                    <span class="label">SQLite Database</span>
                    <span class="size">~2.4 MB</span>
                  </div>
                </div>
                <div class="storage-item">
                  <Cpu size={16} />
                  <div class="storage-details">
                    <span class="label">DuckDB Analytics</span>
                    <span class="size">~15.8 MB</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>

  <!-- API Key Modal -->
  {#if apiKeyModal}
    <div class="modal-overlay" on:click|self={() => apiKeyModal = false}>
      <div class="modal">
        <div class="modal-header">
          <h3>Add API Key</h3>
          <button on:click={() => apiKeyModal = false}><X size={20} /></button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>Name</label>
            <input type="text" placeholder="My OpenAI Key" bind:value={newApiKey.name} />
          </div>
          <div class="form-group">
            <label>Service</label>
            <select bind:value={newApiKey.service}>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="gemini">Google Gemini</option>
              <option value="openrouter">OpenRouter</option>
              <option value="together">Together AI</option>
              <option value="groq">Groq</option>
            </select>
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" placeholder="sk-..." bind:value={newApiKey.key} />
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn secondary" on:click={() => apiKeyModal = false}>Cancel</button>
          <button class="btn primary" on:click={addApiKey}>Add Key</button>
        </div>
      </div>
    </div>
  {/if}

  <!-- MCP Server Modal -->
  {#if mcpModalOpen}
    <div class="modal-overlay" on:click|self={() => mcpModalOpen = false}>
      <div class="modal">
        <div class="modal-header">
          <h3>Add MCP Server</h3>
          <button on:click={() => mcpModalOpen = false}><X size={20} /></button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>Server Name</label>
            <input type="text" placeholder="My Custom Server" bind:value={newMcpServer.name} />
          </div>
          <div class="form-group">
            <label>Command</label>
            <input type="text" placeholder="npx" bind:value={newMcpServer.command} />
            <small>The executable or command to run</small>
          </div>
          <div class="form-group">
            <label>Arguments (space-separated)</label>
            <input type="text" placeholder="-y @package/server --port 3000" bind:value={newMcpServer.args} />
            <small>Command line arguments, separated by spaces</small>
          </div>
          <div class="form-group">
            <label>Description</label>
            <textarea placeholder="What this server does..." bind:value={newMcpServer.description}></textarea>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn secondary" on:click={() => mcpModalOpen = false}>Cancel</button>
          <button class="btn primary" on:click={addMcpServer}>Add Server</button>
        </div>
      </div>
    </div>
  {/if}
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
    pointer-events: all;
  }

  .settings-panel {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 900px;
    max-width: 95vw;
    height: 85vh;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
  }

  /* Header */
  .settings-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    color: var(--text-primary);
  }

  .header-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  /* Content */
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
    border-right: 1px solid var(--border-subtle);
    background: var(--bg-primary);
  }

  .settings-tabs .tab {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
    position: relative;
  }

  .settings-tabs .tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .settings-tabs .tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .settings-tabs .badge {
    position: absolute;
    right: 8px;
    background: var(--bg-tertiary);
    color: var(--text-muted);
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
  }

  .settings-panels {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  .panel h3 {
    margin: 0 0 20px;
    font-size: 16px;
    color: var(--text-primary);
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .panel-header h3 {
    margin: 0;
  }

  /* Setting Groups */
  .setting-group {
    margin-bottom: 24px;
    padding: 16px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .setting-group label {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 12px;
  }

  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    font-size: 13px;
  }

  .setting-row span:first-child {
    color: var(--text-primary);
  }

  /* Inputs */
  input[type="text"],
  input[type="number"],
  input[type="password"],
  select,
  textarea {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: inherit;
  }

  input[type="text"],
  input[type="number"],
  input[type="password"] {
    width: 200px;
  }

  .text-input {
    width: 100%;
  }

  .number-input {
    width: 100px;
  }

  textarea {
    width: 100%;
    min-height: 80px;
    resize: vertical;
  }

  select {
    cursor: pointer;
  }

  /* Switch */
  .switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
  }

  .switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--border-subtle);
    transition: 0.3s;
    border-radius: 24px;
  }

  .slider:before {
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

  input:checked + .slider {
    background-color: var(--accent-primary);
  }

  input:checked + .slider:before {
    transform: translateX(20px);
  }

  /* Range Slider */
  .slider-input {
    -webkit-appearance: none;
    width: 120px;
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    outline: none;
  }

  .slider-input::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: var(--accent-primary);
    border-radius: 50%;
    cursor: pointer;
  }

  /* Buttons */
  .btn {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }

  .btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .icon-btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .icon-btn.danger:hover {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  /* Info Box */
  .info-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    margin-bottom: 16px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 8px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  /* Keys List */
  .keys-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .key-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .key-icon {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary);
    border-radius: 6px;
    color: var(--accent-primary);
  }

  .key-info {
    flex: 1;
  }

  .key-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .key-service {
    font-size: 11px;
    color: var(--text-muted);
  }

  .key-value code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-muted);
    background: var(--bg-tertiary);
    padding: 4px 8px;
    border-radius: 4px;
  }

  .key-actions {
    display: flex;
    gap: 4px;
  }

  /* MCP Servers */
  .servers-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .server-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .server-icon {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary);
    border-radius: 6px;
    color: var(--accent-primary);
  }

  .server-info {
    flex: 1;
  }

  .server-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .server-desc {
    font-size: 11px;
    color: var(--text-muted);
    margin: 2px 0;
  }

  .server-command code {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-tertiary);
    padding: 4px 8px;
    border-radius: 4px;
    margin-top: 4px;
  }

  .status-badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .status-badge.running {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .status-badge.stopped {
    background: var(--bg-tertiary);
    color: var(--text-muted);
  }

  .status-badge.error {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  /* Skills List - removed unused selectors */

  .stat-value.disabled {
    color: #ef4444;
  }

  /* Zones Grid */
  .zones-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .zone-item {
    padding: 16px;
    border-radius: 8px;
    text-align: center;
  }

  .zone-item.danger {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .zone-item.growth {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
  }

  .zone-item.scaling {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
  }

  .zone-item.guardian {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
  }

  .zone-label {
    display: block;
    font-size: 10px;
    font-weight: 600;
    margin-bottom: 4px;
  }

  .zone-amount {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Storage Info */
  .storage-info {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .storage-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .storage-details {
    display: flex;
    flex-direction: column;
  }

  .storage-details .label {
    font-size: 12px;
    color: var(--text-primary);
  }

  .storage-details .size {
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    text-align: center;
    color: var(--text-muted);
  }

  .empty-state span {
    margin: 12px 0;
  }

  /* Modal */
  .modal-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
  }

  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 480px;
    max-width: 90%;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .modal-body {
    padding: 20px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .form-group small {
    display: block;
    margin-top: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-subtle);
  }

  /* AGENTS.md Editor */
  .agents-md-editor {
    margin-top: 16px;
  }

  .code-editor {
    font-family: 'JetBrains Mono', 'Monaco', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.5;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    width: 100%;
    min-height: 400px;
    resize: vertical;
    color: var(--text-primary);
  }

  .code-editor:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  /* Loading and Saved States */
  .btn.loading {
    opacity: 0.7;
    cursor: not-allowed;
  }

  .btn.saved {
    background: #10b981;
    color: white;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  /* Settings Panel Focus */
  .settings-panel {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 900px;
    max-width: 95vw;
    height: 85vh;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
    outline: none;
  }

  .settings-panel:focus {
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), 0 0 0 2px var(--accent-primary);
  }

  .icon-btn[title="Close"] {
    margin-left: 8px;
  }

  .icon-btn[title="Close"]:hover {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  /* Agent Selector Tabs */
  .agent-selector-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    padding: 4px;
    background: var(--bg-primary);
    border-radius: 8px;
  }

  .agent-tab {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 16px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .agent-tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .agent-tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  /* Editor Mode Toggle */
  .editor-mode-toggle {
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
  }

  .mode-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .mode-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .mode-btn.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  /* Agent Configuration Editor */
  .agent-config-editor {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .prompt-editor {
    margin-top: 8px;
  }

  .prompt-textarea {
    font-family: 'JetBrains Mono', 'Monaco', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.6;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px;
    width: 100%;
    min-height: 200px;
    resize: vertical;
    color: var(--text-primary);
  }

  .prompt-textarea:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .help-text {
    display: block;
    margin-top: 6px;
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Skills Grid */
  .skills-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
    margin-top: 12px;
  }

  .skill-config-item {
    padding: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
  }

  .skill-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .skill-name-input {
    flex: 1;
    padding: 6px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    font-weight: 500;
  }

  .skill-desc-input {
    width: 100%;
    padding: 6px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 11px;
  }

  .switch.small {
    width: 32px;
    height: 18px;
  }

  .switch.small .slider:before {
    height: 14px;
    width: 14px;
    left: 2px;
    bottom: 2px;
  }

  .switch.small input:checked + .slider:before {
    transform: translateX(14px);
  }

  .add-skill-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
    padding: 4px 10px;
    background: var(--accent-primary);
    border: none;
    border-radius: 4px;
    color: var(--bg-primary);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .add-skill-btn:hover {
    opacity: 0.9;
  }

  /* Tools List */
  .tools-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
  }

  .tool-checkbox {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tool-checkbox:hover {
    background: var(--bg-primary);
  }

  .tool-checkbox input {
    width: 14px;
    height: 14px;
    accent-color: var(--accent-primary);
  }

  .tool-checkbox code {
    font-size: 11px;
    color: var(--text-primary);
  }

  /* Header Actions */
  .panel-header .header-actions {
    display: flex;
    gap: 8px;
  }

  /* Text Input Full Width */
  input.text-input {
    width: 200px;
  }
</style>
