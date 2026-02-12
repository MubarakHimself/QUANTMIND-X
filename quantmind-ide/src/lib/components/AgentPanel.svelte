<script lang="ts">
  import { createEventDispatcher, onMount, tick } from 'svelte';
  import { Bot, Code, Wand2, Settings, History, Server, X, ChevronLeft, ChevronRight, Send, Paperclip, Loader, Key, FileText, Slash, ChevronDown, Plus, Trash2, Eye, EyeOff, Edit3, Clock, List, FolderOpen } from 'lucide-svelte';
  
  const dispatch = createEventDispatcher();
  export let isOpen = true;
  
  let activeAgent = 'copilot';
  let activeSection: 'chat' | 'history' | 'settings' | 'mcp' = 'chat';
  let message = '';
  let loading = false;
  let textareaElement: HTMLTextAreaElement;
  let messagesContainer: HTMLDivElement;
  
  // Auto-resize textarea
  function autoResize() {
    if (textareaElement) {
      textareaElement.style.height = 'auto';
      const newHeight = Math.min(Math.max(textareaElement.scrollHeight, 80), 200);
      textareaElement.style.height = newHeight + 'px';
    }
  }
  
  // Character counter
  $: charCount = message.length;
  let selectedProvider = 'google';
  let selectedModel = 'gemini-2.5-pro';
  let showModelDropdown = false;
  let attachedFiles: string[] = [];
  
  // API Keys management
  type Provider = 'google' | 'anthropic' | 'openai' | 'qwen';
  let apiKeys: Record<Provider, string> = {
    google: '',
    anthropic: '',
    openai: '',
    qwen: ''
  };
  let showApiKeys: Record<Provider, boolean> = {
    google: false,
    anthropic: false,
    openai: false,
    qwen: false
  };
  
  function showToast(message: string, type: 'success' | 'error' = 'success') {
    // Simple toast notification - can be replaced with a proper toast store/component
    console.log(`Toast [${type}]: ${message}`);
  }
  let showAddMcpModal = false;
  let newMcpServer = { name: '', url: '' };
  
  onMount(() => {
    // Load API keys from localStorage
    const storedKeys = localStorage.getItem('quantmind_api_keys');
    if (storedKeys) {
      apiKeys = { ...apiKeys, ...JSON.parse(storedKeys) };
    }
  });
  
  function saveApiKeys() {
    localStorage.setItem('quantmind_api_keys', JSON.stringify(apiKeys));
    alert('API keys saved successfully!');
  }
  
  function toggleApiKeyVisibility(provider: Provider) {
    showApiKeys[provider] = !showApiKeys[provider];
    showApiKeys = showApiKeys;
  }

  // Helper functions for template access (avoids type errors in Svelte template)
  function getApiKey(provider: string): string {
    return apiKeys[provider as Provider] ?? '';
  }

  function setApiKey(provider: string, value: string) {
    apiKeys[provider as Provider] = value;
    apiKeys = apiKeys;
  }

  function getShowApiKey(provider: string): boolean {
    return showApiKeys[provider as Provider] ?? false;
  }
  
  function addMcpServer() {
    if (newMcpServer.name && newMcpServer.url) {
      mcpServers[activeAgent] = [...mcpServers[activeAgent], {
        name: newMcpServer.name,
        status: 'disconnected',
        url: newMcpServer.url
      }];
      mcpServers = mcpServers;
      newMcpServer = { name: '', url: '' };
      showAddMcpModal = false;
    }
  }
  
  function removeMcpServer(serverName: string) {
    mcpServers[activeAgent] = mcpServers[activeAgent].filter(s => s.name !== serverName);
    mcpServers = mcpServers;
  }
  
  // Auto-scroll to bottom
  async function scrollToBottom() {
    await tick();
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }
  
  // Per-agent message history
  let agentMessages: Record<string, Array<{role: string, content: string}>> = {
    copilot: [{ role: 'assistant', content: "Hello! I'm the QuantMind Copilot. I can help analyze strategies, run backtests, and manage bots." }],
    quantcode: [{ role: 'assistant', content: "I'm QuantCode. I can help write MQ5 code, debug EAs, and optimize parameters." }],
    analyst: [{ role: 'assistant', content: "I'm the Analyst. I analyze NPRD outputs and help interpret trading patterns." }]
  };
  
  // Per-agent chat history (saved conversations)
  let agentHistory: Record<string, Array<{id: string, title: string, date: string}>> = {
    copilot: [
      { id: 'c1', title: 'Backtest ICT Scalper', date: 'Today' },
      { id: 'c2', title: 'Setup Risk Management', date: 'Yesterday' }
    ],
    quantcode: [
      { id: 'q1', title: 'Debug entry logic', date: 'Today' }
    ],
    analyst: [
      { id: 'a1', title: 'Analyze SMC patterns', date: '2 days ago' }
    ]
  };
  
  // Agent Queues (pending tasks)
  let agentQueues: Record<string, Array<{id: string, task: string, status: 'pending' | 'processing' | 'completed', timestamp: Date}>> = {
    copilot: [
      { id: 'cq1', task: 'Analyze EURUSD H1 chart', status: 'processing', timestamp: new Date() },
      { id: 'cq2', task: 'Generate trade summary', status: 'pending', timestamp: new Date() }
    ],
    quantcode: [
      { id: 'qq1', task: 'Optimize EA parameters', status: 'pending', timestamp: new Date() }
    ],
    analyst: []
  };
  
  // Agent.md editing state
  let editingAgentMd = false;
  let agentMdContent = '';
  
  function openAgentMdEditor() {
    // Load agent.md content
    agentMdContent = `# ${agents.find(a => a.id === activeAgent)?.name} Agent\n\n## System Prompt\n${currentSettings.systemPrompt}\n\n## Capabilities\n- Analyze trading strategies\n- Run backtests\n- Manage trading bots\n\n## Limitations\n- Requires API access to trading platforms`;
    editingAgentMd = true;
  }
  
  async function saveAgentMd() {
    // Save to agent.md file via backend
    try {
      await fetch('http://localhost:8000/api/agents/' + activeAgent + '/system-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: agentMdContent })
      });
      editingAgentMd = false;
      showToast('Agent system prompt saved!');
    } catch (e) {
      console.error('Failed to save agent.md:', e);
      showToast('Failed to save changes', 'error');
    }
  }
  
  function cancelAgentMdEdit() {
    editingAgentMd = false;
  }
  
  // New chat creation
  function createNewChat() {
    const newId = 'new_' + Date.now();
    agentHistory[activeAgent] = [...agentHistory[activeAgent], {
      id: newId,
      title: 'New Chat',
      date: 'Just now'
    }];
    agentHistory = agentHistory;
    agentMessages[activeAgent] = [{ role: 'assistant', content: getAgentGreeting(activeAgent) }];
    agentMessages = agentMessages;
  }
  
  function loadChat(chatId: string) {
    // Load chat history for selected conversation
    showToast('Loading chat...');
  }
  
  function deleteChat(chatId: string) {
    agentHistory[activeAgent] = agentHistory[activeAgent].filter(c => c.id !== chatId);
    agentHistory = agentHistory;
    showToast('Chat deleted');
  }
  
  function getAgentGreeting(agentId: string): string {
    const greetings: Record<string, string> = {
      copilot: "Hello! I'm the QuantMind Copilot. I can help analyze strategies, run backtests, and manage bots.",
      quantcode: "I'm QuantCode. I can help write MQ5 code, debug EAs, and optimize parameters.",
      analyst: "I'm the Analyst. I analyze NPRD outputs and help interpret trading patterns."
    };
    return greetings[agentId] || 'Hello! How can I help?';
  }
  
  // Per-agent settings
  let agentSettings: Record<string, {systemPrompt: string, temperature: number}> = {
    copilot: { systemPrompt: 'You are a trading assistant...', temperature: 0.7 },
    quantcode: { systemPrompt: 'You are an MQ5 coding expert...', temperature: 0.3 },
    analyst: { systemPrompt: 'You analyze trading strategies...', temperature: 0.5 }
  };
  
  // MCP servers per agent
  let mcpServers: Record<string, Array<{name: string, status: string, url?: string}>> = {
    copilot: [
      { name: 'PageIndex', status: 'connected' },
      { name: 'File System', status: 'connected' }
    ],
    quantcode: [
      { name: 'MQL Reference', status: 'connected' }
    ],
    analyst: []
  };
  
  const providers = [
    { id: 'google', name: 'Google', models: ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash'] },
    { id: 'anthropic', name: 'Anthropic', models: ['claude-sonnet-4', 'claude-3.5-sonnet'] },
    { id: 'qwen', name: 'Qwen', models: ['qwen-32b-coder'] },
    { id: 'ollama', name: 'Ollama (Local)', models: ['codellama', 'mistral', 'qwen2.5-coder'] }
  ];
  
  const agents = [
    { id: 'copilot', name: 'Copilot', icon: Bot, description: 'General assistant' },
    { id: 'quantcode', name: 'QuantCode', icon: Code, description: 'Code & debug' },
    { id: 'analyst', name: 'Analyst', icon: Wand2, description: 'Strategy analysis' },
  ];
  
  $: currentMessages = agentMessages[activeAgent] || [];
  $: currentHistory = agentHistory[activeAgent] || [];
  $: currentSettings = agentSettings[activeAgent] || { systemPrompt: '', temperature: 0.7 };
  $: currentMcp = mcpServers[activeAgent] || [];
  
  function switchAgent(id: string) { 
    activeAgent = id; 
    activeSection = 'chat';
  }
  
  async function sendMessage() {
    if (!message.trim() || loading) return;
    const userMsg = message.trim();
    agentMessages[activeAgent] = [...agentMessages[activeAgent], { role: 'user', content: userMsg }];
    agentMessages = agentMessages;
    message = '';
    loading = true;
    
    await scrollToBottom();
    
    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg,
          agent: activeAgent,
          model: selectedModel,
          context: attachedFiles,
          api_keys: apiKeys
        })
      });
      const data = await res.json();
      agentMessages[activeAgent] = [...agentMessages[activeAgent], { role: 'assistant', content: data.response || data.reply || 'No response from server.' }];
      agentMessages = agentMessages;
    } catch (e) {
      console.error('Chat error:', e);
      agentMessages[activeAgent] = [...agentMessages[activeAgent], { role: 'assistant', content: 'Failed to connect to backend. Please ensure the server is running.' }];
      agentMessages = agentMessages;
    } finally {
      loading = false;
      attachedFiles = [];
      await scrollToBottom();
    }
  }
  
  function handleKeydown(e: KeyboardEvent) { 
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    // Slash commands
    if (e.key === '/' && message === '') {
      // Show slash command menu
    }
  }
  
  function attachFile() {
    // Would open file picker
    attachedFiles = [...attachedFiles, 'strategy.mq5'];
  }
  
  function removeAttachment(file: string) {
    attachedFiles = attachedFiles.filter(f => f !== file);
  }
  
  function newChat() {
    agentMessages[activeAgent] = [{ role: 'assistant', content: `New conversation started with ${agents.find(a => a.id === activeAgent)?.name}.` }];
    agentMessages = agentMessages;
  }
</script>

{#if isOpen}
<aside class="agent-panel">
  <!-- Agent selector - full width tabs -->
  <div class="agent-selector">
    {#each agents as agent}
      <button 
        class="agent-btn" 
        class:active={activeAgent === agent.id} 
        on:click={() => switchAgent(agent.id)}
      >
        <svelte:component this={agent.icon} size={18} />
        <span class="agent-name">{agent.name}</span>
      </button>
    {/each}
  </div>
  
  <!-- Section tabs below agents -->
  <div class="section-tabs">
    <button class:active={activeSection === 'chat'} on:click={() => activeSection = 'chat'}>
      <Bot size={14} /> Chat
    </button>
    <button class:active={activeSection === 'history'} on:click={() => activeSection = 'history'}>
      <History size={14} /> History
    </button>
    <button class:active={activeSection === 'settings'} on:click={() => activeSection = 'settings'}>
      <Settings size={14} /> Settings
    </button>
    <button class:active={activeSection === 'mcp'} on:click={() => activeSection = 'mcp'}>
      <Server size={14} /> MCP
    </button>
    <button class="close-btn" on:click={() => isOpen = false}>
      <ChevronRight size={14} />
    </button>
  </div>
  
  <!-- Content area based on section -->
  <div class="panel-content">
    {#if activeSection === 'chat'}
      <!-- Chat messages -->
      <div class="messages" bind:this={messagesContainer}>
        {#each currentMessages as msg}
          <div class="message {msg.role}">
            {#if msg.role === 'assistant'}
              <div class="msg-header">
                <svelte:component this={agents.find(a => a.id === activeAgent)?.icon || Bot} size={14} />
                <span>{agents.find(a => a.id === activeAgent)?.name}</span>
              </div>
            {/if}
            <div class="msg-content">{@html msg.content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>')}</div>
          </div>
        {/each}
        {#if loading}
          <div class="message assistant">
            <Loader size={14} class="spin" /> Thinking...
          </div>
        {/if}
      </div>
    
    {:else if activeSection === 'history'}
      <!-- Chat history for this agent -->
      <div class="history-section">
        <div class="history-header">
          <span>{agents.find(a => a.id === activeAgent)?.name} History</span>
          <button on:click={newChat}><Plus size={14} /> New Chat</button>
        </div>
        {#if currentHistory.length > 0}
          {#each currentHistory as chat}
            <div class="history-item" on:click={() => loadChat(chat.id)}>
              <span class="chat-title">{chat.title}</span>
              <span class="chat-date">{chat.date}</span>
            </div>
          {/each}
        {:else}
          <p class="empty-msg">No chat history for this agent</p>
        {/if}
      </div>
    
    {:else if activeSection === 'settings'}
      <!-- Agent settings -->
      <div class="settings-section">
        <h4>{agents.find(a => a.id === activeAgent)?.name} Settings</h4>
        
        <div class="setting-group">
          <label for="model-select">Model Configuration</label>
          <select id="model-select" bind:value={selectedModel} aria-label="Select AI model">
            {#each providers as provider}
              {#each provider.models as model}
                <option value={model}>{provider.name} - {model}</option>
              {/each}
            {/each}
          </select>
        </div>
        
        <div class="setting-group">
          <label for="temperature-range">Temperature: {currentSettings.temperature}</label>
          <input
            id="temperature-range"
            type="range"
            min="0"
            max="1"
            step="0.1"
            bind:value={currentSettings.temperature}
            aria-label="Adjust model temperature"
          />
        </div>
        
        <div class="setting-group">
          <label>API Keys</label>
          {#each ['google', 'anthropic', 'openai', 'qwen'] as provider (provider)}
            <div class="api-key-input">
              <label for="api-key-{provider}">{provider.charAt(0).toUpperCase() + provider.slice(1)}</label>
              <div class="key-input-wrapper">
                {#if getShowApiKey(provider)}
                  <input
                    id="api-key-{provider}"
                    type="text"
                    placeholder="Enter API key..."
                    value={getApiKey(provider)}
                    on:input={(e) => setApiKey(provider, e.currentTarget.value)}
                    aria-label={`${provider.charAt(0).toUpperCase() + provider.slice(1)} API key`}
                  />
                {:else}
                  <input
                    id="api-key-{provider}"
                    type="password"
                    placeholder="Enter API key..."
                    value={getApiKey(provider)}
                    on:input={(e) => setApiKey(provider, e.currentTarget.value)}
                    aria-label={`${provider.charAt(0).toUpperCase() + provider.slice(1)} API key`}
                  />
                {/if}
                <button
                  class="toggle-visibility"
                  on:click={() => toggleApiKeyVisibility(provider)}
                  aria-label={getShowApiKey(provider) ? 'Hide API key' : 'Show API key'}
                >
                  {#if getShowApiKey(provider)}
                    <EyeOff size={14} />
                  {:else}
                    <Eye size={14} />
                  {/if}
                </button>
              </div>
            </div>
          {/each}
          <button class="save-keys-btn" on:click={saveApiKeys} aria-label="Save all API keys">Save API Keys</button>
        </div>
        
        <div class="setting-group">
          <label>Agent Files</label>
          <div class="file-item"><FileText size={14} /> agent.md <button>Edit</button></div>
          <div class="file-item"><FileText size={14} /> skills/ <button>View</button></div>
        </div>
      </div>
    
    {:else if activeSection === 'mcp'}
      <!-- MCP servers for this agent -->
      <div class="mcp-section">
        <h4>{agents.find(a => a.id === activeAgent)?.name} MCP Servers</h4>
        
        {#if currentMcp.length > 0}
          {#each currentMcp as server}
            <div class="mcp-item">
              <Server size={14} />
              <div class="server-info">
                <span class="server-name">{server.name}</span>
                {#if server.url}
                  <span class="server-url">{server.url}</span>
                {/if}
              </div>
              <div class="server-actions">
                <span class="server-status" class:connected={server.status === 'connected'}>{server.status}</span>
                <button class="remove-mcp-btn" on:click={() => removeMcpServer(server.name)} title="Remove server">
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
          {/each}
        {:else}
          <p class="empty-msg">No MCP servers configured for this agent</p>
        {/if}
        
        <button class="add-mcp-btn" on:click={() => showAddMcpModal = true}><Plus size={14} /> Add MCP Server</button>
      </div>
      
      {#if showAddMcpModal}
        <div class="modal-overlay" on:click={() => showAddMcpModal = false}>
          <div class="modal-content" on:click|stopPropagation>
            <h4>Add MCP Server</h4>
            <div class="modal-form">
              <div class="form-group">
                <label>Server Name</label>
                <input type="text" placeholder="e.g., File System" bind:value={newMcpServer.name} />
              </div>
              <div class="form-group">
                <label>Server URL</label>
                <input type="text" placeholder="e.g., http://localhost:3000" bind:value={newMcpServer.url} />
              </div>
            </div>
            <div class="modal-actions">
              <button class="btn-cancel" on:click={() => showAddMcpModal = false}>Cancel</button>
              <button class="btn-confirm" on:click={addMcpServer}>Add Server</button>
            </div>
          </div>
        </div>
      {/if}
    {/if}
  </div>
  
  <!-- Chat input - redesigned with model selector inside -->
  {#if activeSection === 'chat'}
    <div class="input-area">
      <!-- Attached files -->
      {#if attachedFiles.length > 0}
        <div class="attachments">
          {#each attachedFiles as file}
            <div class="attachment-chip">
              <FileText size={12} />
              <span>{file}</span>
              <button on:click={() => removeAttachment(file)}><X size={10} /></button>
            </div>
          {/each}
        </div>
      {/if}
      
      <div class="input-container">
        <button class="input-btn" on:click={attachFile} title="Attach file">
          <Paperclip size={16} />
        </button>
        
        <textarea
          placeholder="Message {agents.find(a => a.id === activeAgent)?.name}... (/ for commands)"
          bind:value={message}
          on:keydown={handleKeydown}
          on:input={autoResize}
          bind:this={textareaElement}
          rows="1"
        ></textarea>
        
        <!-- Model selector inside input -->
        <div class="model-selector" on:click={() => showModelDropdown = !showModelDropdown}>
          <span class="selected-model">{selectedModel.split('-').slice(0, 2).join('-')}</span>
          <ChevronDown size={12} />
          
          {#if showModelDropdown}
            <div class="model-dropdown">
              {#each providers as provider}
                <div class="provider-group">
                  <span class="provider-name">{provider.name}</span>
                  {#each provider.models as model}
                    <button 
                      class:selected={selectedModel === model}
                      on:click|stopPropagation={() => { selectedModel = model; selectedProvider = provider.id; showModelDropdown = false; }}
                    >
                      {model}
                    </button>
                  {/each}
                </div>
              {/each}
            </div>
          {/if}
        </div>
        
        <button class="send-btn" on:click={sendMessage} disabled={!message.trim()}>
          <Send size={16} />
        </button>
      </div>
      
      <!-- Character counter -->
      <div class="char-counter">
        <span>{charCount} / 4000</span>
      </div>
    </div>
  {/if}
</aside>
{:else}
<button class="toggle-btn" on:click={() => isOpen = true}><ChevronLeft size={16} /></button>
{/if}

<style>
  .agent-panel { grid-area: agents; display: flex; flex-direction: column; width: 360px; background: var(--bg-secondary); border-left: 1px solid var(--border-subtle); }
  
  /* Agent selector - full width tabs */
  .agent-selector { display: flex; border-bottom: 1px solid var(--border-subtle); }
  .agent-btn { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 12px 8px; background: transparent; border: none; color: var(--text-muted); cursor: pointer; transition: all 0.2s; }
  .agent-btn:hover { background: var(--bg-tertiary); color: var(--text-primary); }
  .agent-btn.active { background: var(--bg-tertiary); color: var(--accent-primary); border-bottom: 2px solid var(--accent-primary); }
  .agent-name { font-size: 10px; }
  
  /* Section tabs */
  .section-tabs { display: flex; padding: 4px; gap: 2px; border-bottom: 1px solid var(--border-subtle); }
  .section-tabs button { display: flex; align-items: center; gap: 4px; padding: 6px 10px; background: transparent; border: none; border-radius: 4px; color: var(--text-muted); font-size: 11px; cursor: pointer; }
  .section-tabs button:hover { background: var(--bg-tertiary); color: var(--text-primary); }
  .section-tabs button.active { background: var(--bg-tertiary); color: var(--accent-primary); }
  .section-tabs .close-btn { margin-left: auto; }
  
  /* Content area */
  .panel-content { flex: 1; overflow-y: auto; }
  
  /* Messages */
  .messages { display: flex; flex-direction: column; gap: 12px; padding: 12px; }
  .message { padding: 10px 14px; border-radius: 12px; font-size: 13px; }
  .message.user { align-self: flex-end; max-width: 85%; background: var(--accent-primary); color: var(--bg-primary); }
  .message.assistant { align-self: flex-start; max-width: 90%; background: var(--bg-tertiary); color: var(--text-primary); }
  .msg-header { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--accent-primary); margin-bottom: 6px; }
  :global(.spin) { animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  
  /* History section */
  .history-section { padding: 12px; }
  .history-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 13px; color: var(--text-primary); }
  .history-header button { display: flex; align-items: center; gap: 4px; padding: 4px 8px; background: var(--accent-primary); border: none; border-radius: 4px; color: var(--bg-primary); font-size: 11px; cursor: pointer; }
  .history-item { display: flex; justify-content: space-between; padding: 10px 12px; background: var(--bg-tertiary); border-radius: 6px; margin-bottom: 6px; cursor: pointer; }
  .history-item:hover { background: var(--bg-primary); }
  .chat-title { font-size: 12px; color: var(--text-primary); }
  .chat-date { font-size: 10px; color: var(--text-muted); }
  .empty-msg { text-align: center; color: var(--text-muted); font-size: 12px; padding: 20px; }
  
  /* Settings section */
  .settings-section { padding: 12px; }
  .settings-section h4 { margin: 0 0 16px; font-size: 13px; color: var(--text-primary); }
  .setting-group { margin-bottom: 16px; }
  .setting-group label { display: block; font-size: 11px; color: var(--text-muted); margin-bottom: 6px; }
  .setting-group textarea { width: 100%; min-height: 80px; padding: 8px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-primary); font-size: 12px; resize: vertical; }
  .setting-group input[type="range"] { width: 100%; }
  .file-item { display: flex; align-items: center; gap: 8px; padding: 8px 10px; background: var(--bg-tertiary); border-radius: 6px; margin-bottom: 4px; font-size: 12px; color: var(--text-secondary); }
  .file-item button { margin-left: auto; padding: 2px 8px; background: var(--bg-primary); border: 1px solid var(--border-subtle); border-radius: 4px; color: var(--text-muted); font-size: 10px; cursor: pointer; }
  
  /* API Keys styling */
  .api-key-input { margin-bottom: 8px; }
  .provider-label { display: block; font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
  .key-input-wrapper { display: flex; gap: 4px; }
  .key-input-wrapper input { flex: 1; padding: 6px 8px; background: var(--bg-primary); border: 1px solid var(--border-subtle); border-radius: 4px; color: var(--text-primary); font-size: 12px; }
  .toggle-visibility { display: flex; align-items: center; justify-content: center; width: 28px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 4px; color: var(--text-muted); cursor: pointer; }
  .toggle-visibility:hover { color: var(--text-primary); }
  .save-keys-btn { width: 100%; padding: 8px; background: var(--accent-primary); border: none; border-radius: 4px; color: var(--bg-primary); font-size: 12px; font-weight: 600; cursor: pointer; margin-top: 8px; }
  
  /* MCP section */
  .mcp-section { padding: 12px; }
  .mcp-section h4 { margin: 0 0 16px; font-size: 13px; color: var(--text-primary); }
  .mcp-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: var(--bg-tertiary); border-radius: 6px; margin-bottom: 6px; }
  .server-info { flex: 1; display: flex; flex-direction: column; gap: 2px; }
  .server-name { font-size: 12px; color: var(--text-primary); }
  .server-url { font-size: 10px; color: var(--text-muted); }
  .server-actions { display: flex; align-items: center; gap: 8px; }
  .server-status { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: var(--bg-primary); color: var(--text-muted); }
  .server-status.connected { color: #10b981; }
  .remove-mcp-btn { display: flex; align-items: center; justify-content: center; width: 24px; height: 24px; background: transparent; border: none; color: var(--text-muted); cursor: pointer; border-radius: 4px; }
  .remove-mcp-btn:hover { color: #ef4444; background: rgba(239, 68, 68, 0.1); }
  .add-mcp-btn { display: flex; align-items: center; justify-content: center; gap: 6px; width: 100%; padding: 10px; background: transparent; border: 1px dashed var(--border-subtle); border-radius: 6px; color: var(--text-muted); font-size: 12px; cursor: pointer; margin-top: 8px; }
  .add-mcp-btn:hover { color: var(--accent-primary); border-color: var(--accent-primary); }
  
  /* Modal styling */
  .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0, 0, 0, 0.5); display: flex; align-items: center; justify-content: center; z-index: 1000; }
  .modal-content { background: var(--bg-secondary); border-radius: 12px; padding: 20px; width: 320px; max-width: 90%; border: 1px solid var(--border-subtle); }
  .modal-content h4 { margin: 0 0 16px; font-size: 14px; color: var(--text-primary); }
  .modal-form { display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }
  .form-group { display: flex; flex-direction: column; gap: 4px; }
  .form-group label { font-size: 11px; color: var(--text-muted); }
  .form-group input { padding: 8px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-primary); font-size: 12px; }
  .modal-actions { display: flex; gap: 8px; justify-content: flex-end; }
  .btn-cancel, .btn-confirm { padding: 8px 16px; border-radius: 6px; font-size: 12px; cursor: pointer; }
  .btn-cancel { background: var(--bg-tertiary); border: 1px solid var(--border-subtle); color: var(--text-primary); }
  .btn-confirm { background: var(--accent-primary); border: none; color: var(--bg-primary); }
  
  /* Input area - redesigned */
  .input-area { padding: 12px; border-top: 1px solid var(--border-subtle); }
  
  .attachments { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
  .attachment-chip { display: flex; align-items: center; gap: 4px; padding: 4px 8px; background: var(--bg-tertiary); border-radius: 4px; font-size: 11px; color: var(--text-secondary); }
  .attachment-chip button { background: none; border: none; color: var(--text-muted); cursor: pointer; padding: 0; }
  
  .input-container { display: flex; align-items: flex-end; gap: 8px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 8px 12px; }
  
  .input-btn { display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: transparent; border: none; color: var(--text-muted); cursor: pointer; border-radius: 6px; }
  .input-btn:hover { background: var(--bg-secondary); color: var(--text-primary); }
  
  .input-container textarea { flex: 1; background: transparent; border: none; color: var(--text-primary); font-size: 13px; padding: 6px 0; resize: none; font-family: inherit; min-height: 80px; max-height: 200px; outline: none; overflow-y: auto; }
  
  /* Model selector inside input */
  .model-selector { display: flex; align-items: center; gap: 4px; padding: 4px 8px; background: var(--bg-secondary); border-radius: 6px; cursor: pointer; position: relative; }
  .selected-model { font-size: 10px; color: var(--text-muted); max-width: 80px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  
  .model-dropdown { position: absolute; bottom: 100%; right: 0; margin-bottom: 8px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 8px; min-width: 180px; max-height: 300px; overflow-y: auto; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
  .provider-group { padding: 8px 0; border-bottom: 1px solid var(--border-subtle); }
  .provider-group:last-child { border-bottom: none; }
  .provider-name { display: block; padding: 4px 12px; font-size: 10px; color: var(--text-muted); text-transform: uppercase; }
  .provider-group button { display: block; width: 100%; padding: 8px 12px; background: transparent; border: none; color: var(--text-secondary); font-size: 12px; text-align: left; cursor: pointer; }
  .provider-group button:hover { background: var(--bg-tertiary); }
  .provider-group button.selected { color: var(--accent-primary); }
  
  .send-btn { display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; background: var(--accent-primary); border: none; border-radius: 8px; color: var(--bg-primary); cursor: pointer; }
  .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  
  .char-counter { display: flex; justify-content: flex-end; padding: 4px 12px 0; font-size: 10px; color: var(--text-muted); }
  
  .toggle-btn { position: fixed; right: 0; top: 50%; transform: translateY(-50%); width: 24px; height: 48px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-right: none; border-radius: 8px 0 0 8px; color: var(--text-muted); cursor: pointer; }
</style>
