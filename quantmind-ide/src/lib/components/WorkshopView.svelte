<script lang="ts">
  import { onMount } from "svelte";
  import {
    Bot, Send, Loader, RefreshCw, Wrench, Database, Clock, GitBranch,
    MessageSquare
  } from "lucide-svelte";
  import { memoryStore } from "$lib/stores/memoryStore";
  import { cronStore } from "$lib/stores/cronStore";
  import { hooksStore } from "$lib/stores/hooksStore";
  import * as memoryApi from "$lib/api/memory";

  // API base URL
  const API_BASE = "http://localhost:8000/api";

  // Tab state
  let activeTab = "chat";
  const tabs = [
    { id: "chat", label: "Chat", icon: MessageSquare },
    { id: "memory", label: "Memory", icon: Database },
    { id: "cron", label: "Cron Jobs", icon: Clock },
    { id: "hooks", label: "Hooks", icon: GitBranch },
  ];

  // Chat state
  let messages: Array<{ role: string; content: string; timestamp?: Date }> = [
    {
      role: "assistant",
      content: "Hello! I'm the QuantMind Copilot. I can help you design strategies, run backtests, analyze results, and manage your trading system. What would you like to do?",
    },
  ];
  let inputMessage = "";
  let loading = false;
  let error = "";
  let messagesContainer: HTMLDivElement;

  // Memory state
  $: memories = $memoryStore.filteredMemories;
  $: memoryStats = $memoryStore.stats;
  $: memoryLoading = $memoryStore.loading;
  $: memoryError = $memoryStore.error;

  // Cron state
  $: cronJobs = $cronStore.jobs;
  $: cronLoading = $cronStore.loading;
  $: cronError = $cronStore.error;

  // Hooks state
  $: hooks = $hooksStore.hooks;
  $: hookLogs = $hooksStore.logs;
  $: hooksLoading = $hooksStore.loading;
  $: hooksError = $hooksStore.error;

  onMount(() => {
    loadMemoryData();
    loadCronData();
    loadHooksData();
  });

  async function loadMemoryData() {
    memoryStore.setLoading(true);
    try {
      const result = await memoryApi.listMemoriesForStore("default", 100);
      memoryStore.setMemories(result.memories);
      const stats = await memoryApi.getMemoryStatsForStore();
      memoryStore.setStats(stats);
    } catch (e) {
      memoryStore.setError(e instanceof Error ? e.message : "Failed to load memories");
    } finally {
      memoryStore.setLoading(false);
    }
  }

  async function loadCronData() {
    cronStore.setLoading(true);
    try {
      const result = await memoryApi.listCronJobsForStore();
      cronStore.setJobs(result.jobs);
    } catch (e) {
      cronStore.setError(e instanceof Error ? e.message : "Failed to load cron jobs");
    } finally {
      cronStore.setLoading(false);
    }
  }

  async function loadHooksData() {
    hooksStore.setLoading(true);
    try {
      const result = await memoryApi.listHooksForStore();
      hooksStore.setHooks(result.hooks);
    } catch (e) {
      hooksStore.setError(e instanceof Error ? e.message : "Failed to load hooks");
    } finally {
      hooksStore.setLoading(false);
    }
  }

  // Send message to backend
  async function sendMessage() {
    if (!inputMessage.trim() || loading) return;

    const userMessage = inputMessage.trim();
    inputMessage = "";

    messages = [...messages, { role: "user", content: userMessage, timestamp: new Date() }];
    loading = true;
    error = "";

    setTimeout(() => {
      if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
    }, 0);

    try {
      const response = await fetch(`${API_BASE}/chat/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          agent: "quantmind",
          model: "default",
          provider: "anthropic",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      messages = [
        ...messages,
        {
          role: "assistant",
          content: data.reply || "I received your message but couldn't generate a response.",
          timestamp: new Date(),
        },
      ];
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to send message";
      console.error("Chat error:", e);
    } finally {
      loading = false;
      setTimeout(() => {
        if (messagesContainer) {
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
      }, 0);
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function clearChat() {
    messages = [
      {
        role: "assistant",
        content: "Hello! I'm the QuantMind Copilot. I can help you design strategies, run backtests, analyze results, and manage your trading system. What would you like to do?",
      },
    ];
    error = "";
  }

  function quickAction(text: string) {
    inputMessage = text;
    sendMessage();
  }

  async function toggleCronJob(jobId: string, enabled: boolean) {
    try {
      await memoryApi.toggleCronJob(jobId, enabled);
      cronStore.toggleJob(jobId);
    } catch (e) {
      cronStore.setError(e instanceof Error ? e.message : "Failed to toggle job");
    }
  }

  async function toggleHook(hookName: string, enabled: boolean) {
    try {
      await memoryApi.toggleHook(hookName, enabled);
      hooksStore.updateHook(hookName, { enabled });
    } catch (e) {
      hooksStore.setError(e instanceof Error ? e.message : "Failed to toggle hook");
    }
  }
</script>

<div class="workshop-view">
  <!-- Header with tabs -->
  <div class="workshop-header">
    <div class="header-title">
      <Wrench size={20} />
      <h2>QuantMind Workshop</h2>
    </div>
    <div class="tabs">
      {#each tabs as tab}
        <button
          class="tab-btn"
          class:active={activeTab === tab.id}
          on:click={() => activeTab = tab.id}
        >
          <svelte:component this={tab.icon} size={14} />
          {tab.label}
        </button>
      {/each}
    </div>
    {#if activeTab === "chat"}
    <button class="clear-btn" on:click={clearChat} title="Clear chat">
      <RefreshCw size={16} />
      Clear
    </button>
    {/if}
  </div>

  <!-- Tab Content -->
  <div class="tab-content">
    <!-- Chat Tab -->
    {#if activeTab === "chat"}
    <div class="chat-section">
      <div class="messages" bind:this={messagesContainer}>
        {#each messages as msg}
          <div class="message {msg.role}">
            <div class="message-avatar">
              {#if msg.role === "user"}
                <span>Y</span>
              {:else}
                <Bot size={16} />
              {/if}
            </div>
            <div class="message-body">
              <div class="message-text">{msg.content}</div>
            </div>
          </div>
        {/each}

        {#if loading}
          <div class="message assistant">
            <div class="message-avatar">
              <Bot size={16} />
            </div>
            <div class="message-body">
              <div class="typing">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        {/if}
      </div>

      {#if error}
        <div class="error">
          {error}
          <button on:click={() => (error = "")}>x</button>
        </div>
      {/if}

      <div class="input-row">
        <textarea
          class="chat-input"
          bind:value={inputMessage}
          on:keydown={handleKeydown}
          placeholder="Ask QuantMind Copilot..."
          rows="1"
          disabled={loading}
        ></textarea>
        <button
          class="send-btn"
          on:click={sendMessage}
          disabled={!inputMessage.trim() || loading}
        >
          {#if loading}
            <Loader size={18} class="spin" />
          {:else}
            <Send size={18} />
          {/if}
        </button>
      </div>
    </div>
    <div class="side-panel">
      <div class="panel-section">
        <h4>Quick Actions</h4>
        <div class="quick-actions">
          <button on:click={() => quickAction("Read the latest trading logs")}>View Logs</button>
          <button on:click={() => quickAction("Run a backtest on my current strategy")}>Run Backtest</button>
          <button on:click={() => quickAction("Analyze the current market regime")}>Market Analysis</button>
        </div>
      </div>
    </div>
    {/if}

    <!-- Memory Tab -->
    {#if activeTab === "memory"}
    <div class="data-panel">
      <div class="panel-header-row">
        <h3>Memory Management</h3>
        <button class="refresh-btn" on:click={loadMemoryData} disabled={memoryLoading}>
          <RefreshCw size={14} class={memoryLoading ? "spinning" : ""} />
          Refresh
        </button>
      </div>

      {#if memoryError}
        <div class="error-banner">{memoryError}</div>
      {/if}

      {#if memoryStats}
        <div class="stats-grid">
          <div class="stat-card">
            <span class="stat-value">{memoryStats.total_count || 0}</span>
            <span class="stat-label">Total Memories</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">{memoryStats.embedding_model || "N/A"}</span>
            <span class="stat-label">Embedding Model</span>
          </div>
        </div>
      {/if}

      <div class="data-list">
        {#if memoryLoading}
          <div class="loading-state">
            <RefreshCw size={24} class="spinning" />
            <span>Loading memories...</span>
          </div>
        {:else if memories.length === 0}
          <div class="empty-state">
            <Database size={32} />
            <p>No memories stored</p>
          </div>
        {:else}
          {#each memories as memory}
            <div class="data-item">
              <div class="item-header">
                <span class="item-key">{memory.key || memory.id}</span>
                <span class="item-namespace">{memory.namespace}</span>
              </div>
              <p class="item-content">{memory.content}</p>
              <span class="item-time">{memory.timestamp}</span>
            </div>
          {/each}
        {/if}
      </div>
    </div>
    {/if}

    <!-- Cron Jobs Tab -->
    {#if activeTab === "cron"}
    <div class="data-panel">
      <div class="panel-header-row">
        <h3>Cron Jobs</h3>
        <button class="refresh-btn" on:click={loadCronData} disabled={cronLoading}>
          <RefreshCw size={14} class={cronLoading ? "spinning" : ""} />
          Refresh
        </button>
      </div>

      {#if cronError}
        <div class="error-banner">{cronError}</div>
      {/if}

      <div class="data-list">
        {#if cronLoading}
          <div class="loading-state">
            <RefreshCw size={24} class="spinning" />
            <span>Loading cron jobs...</span>
          </div>
        {:else if cronJobs.length === 0}
          <div class="empty-state">
            <Clock size={32} />
            <p>No cron jobs configured</p>
          </div>
        {:else}
          {#each cronJobs as job}
            <div class="data-item">
              <div class="item-header">
                <span class="item-name">{job.name}</span>
                <button
                  class="toggle-btn"
                  on:click={() => toggleCronJob(job.id, !job.enabled)}
                >
                  {job.enabled ? "Disable" : "Enable"}
                </button>
              </div>
              <div class="item-details">
                <code>{job.schedule}</code>
                <span class="status-badge" class:enabled={job.enabled} class:disabled={!job.enabled}>
                  {job.enabled ? "Enabled" : "Disabled"}
                </span>
              </div>
              {#if job.lastRun}
                <span class="item-time">Last run: {job.lastRun}</span>
              {/if}
            </div>
          {/each}
        {/if}
      </div>
    </div>
    {/if}

    <!-- Hooks Tab -->
    {#if activeTab === "hooks"}
    <div class="data-panel">
      <div class="panel-header-row">
        <h3>Hook Management</h3>
        <button class="refresh-btn" on:click={loadHooksData} disabled={hooksLoading}>
          <RefreshCw size={14} class={hooksLoading ? "spinning" : ""} />
          Refresh
        </button>
      </div>

      {#if hooksError}
        <div class="error-banner">{hooksError}</div>
      {/if}

      <div class="data-list">
        {#if hooksLoading}
          <div class="loading-state">
            <RefreshCw size={24} class="spinning" />
            <span>Loading hooks...</span>
          </div>
        {:else if hooks.length === 0}
          <div class="empty-state">
            <GitBranch size={32} />
            <p>No hooks registered</p>
          </div>
        {:else}
          {#each hooks as hook}
            <div class="data-item">
              <div class="item-header">
                <span class="item-name">{hook.name}</span>
                <button
                  class="toggle-btn"
                  on:click={() => toggleHook(hook.name, !hook.enabled)}
                >
                  {hook.enabled ? "Disable" : "Enable"}
                </button>
              </div>
              <div class="item-details">
                <span class="status-badge" class:enabled={hook.enabled} class:disabled={!hook.enabled}>
                  {hook.enabled ? "Enabled" : "Disabled"}
                </span>
                {#if hook.priority}
                  <span class="priority">Priority: {hook.priority}</span>
                {/if}
              </div>
            </div>
          {/each}
        {/if}
      </div>
    </div>
    {/if}
  </div>
</div>

<style>
  .workshop-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #0a0f1a);
    color: var(--text-primary, #e2e8f0);
  }

  .workshop-header {
    display: flex;
    align-items: center;
    padding: 0.5rem 1rem;
    background: var(--bg-secondary, #111827);
    border-bottom: 1px solid var(--border-color, #1e293b);
    gap: 1rem;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--accent-primary, #3b82f6);
  }

  .header-title h2 {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 600;
  }

  .tabs {
    display: flex;
    gap: 0.25rem;
    margin-left: 1rem;
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: transparent;
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tab-btn:hover {
    background: var(--bg-tertiary, #1e293b);
    color: var(--text-primary, #e2e8f0);
  }

  .tab-btn.active {
    background: var(--accent-primary, #3b82f6);
    border-color: var(--accent-primary, #3b82f6);
    color: white;
  }

  .clear-btn {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    background: transparent;
    border: 1px solid var(--border-color, #334155);
    color: var(--text-secondary, #94a3b8);
    padding: 0.375rem 0.75rem;
    border-radius: 0.375rem;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .clear-btn:hover {
    background: var(--bg-tertiary, #1e293b);
  }

  .tab-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Chat styles */
  .chat-section {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .message {
    display: flex;
    gap: 0.5rem;
    max-width: 85%;
  }

  .message.user {
    align-self: flex-end;
    flex-direction: row-reverse;
  }

  .message.assistant {
    align-self: flex-start;
  }

  .message-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary, #1e293b);
    flex-shrink: 0;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .message.user .message-avatar {
    background: var(--accent-primary, #3b82f6);
    color: white;
  }

  .message.assistant .message-avatar {
    color: var(--accent-primary, #3b82f6);
  }

  .message-body {
    flex: 1;
    min-width: 0;
  }

  .message-text {
    padding: 0.5rem 0.75rem;
    border-radius: 0.75rem;
    font-size: 0.8125rem;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .message.user .message-text {
    background: var(--accent-primary, #3b82f6);
    color: white;
    border-bottom-right-radius: 0.25rem;
  }

  .message.assistant .message-text {
    background: var(--bg-tertiary, #1e293b);
    border-bottom-left-radius: 0.25rem;
  }

  .typing {
    display: flex;
    gap: 0.25rem;
    padding: 0.5rem;
  }

  .typing span {
    width: 6px;
    height: 6px;
    background: var(--text-secondary, #94a3b8);
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
  }

  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }

  .error {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.75rem;
    margin: 0 1rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.375rem;
    color: #fca5a5;
    font-size: 0.75rem;
  }

  .error button {
    background: transparent;
    border: none;
    color: inherit;
    cursor: pointer;
  }

  .input-row {
    display: flex;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border-top: 1px solid var(--border-color, #1e293b);
    background: var(--bg-secondary, #111827);
  }

  .chat-input {
    flex: 1;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.8125rem;
    resize: none;
    min-height: 36px;
    max-height: 100px;
  }

  .chat-input:focus {
    outline: none;
    border-color: var(--accent-primary, #3b82f6);
  }

  .send-btn {
    background: var(--accent-primary, #3b82f6);
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem;
    color: white;
    cursor: pointer;
    transition: background 0.2s;
  }

  .send-btn:hover:not(:disabled) {
    background: var(--accent-hover, #2563eb);
  }

  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spin {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  /* Side panel */
  .side-panel {
    width: 200px;
    padding: 0.75rem;
    background: var(--bg-secondary, #111827);
    border-left: 1px solid var(--border-color, #1e293b);
    overflow-y: auto;
  }

  .panel-section {
    margin-bottom: 1rem;
  }

  .panel-section h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.6875rem;
    font-weight: 600;
    color: var(--text-secondary, #94a3b8);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .quick-actions {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .quick-actions button {
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    padding: 0.375rem 0.5rem;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.6875rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s;
  }

  .quick-actions button:hover {
    background: var(--bg-hover, #334155);
    border-color: var(--accent-primary, #3b82f6);
  }

  /* Data panel styles */
  .data-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 1rem;
  }

  .panel-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .panel-header-row h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-hover, #334155);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error-banner {
    padding: 0.5rem 0.75rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.375rem;
    color: #fca5a5;
    font-size: 0.75rem;
    margin-bottom: 1rem;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .stat-card {
    display: flex;
    flex-direction: column;
    padding: 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color, #334155);
  }

  .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--accent-primary, #3b82f6);
  }

  .stat-label {
    font-size: 0.6875rem;
    color: var(--text-secondary, #94a3b8);
    text-transform: uppercase;
  }

  .data-list {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .data-item {
    padding: 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.5rem;
  }

  .item-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .item-name, .item-key {
    font-weight: 500;
    font-size: 0.875rem;
  }

  .item-namespace {
    font-size: 0.6875rem;
    padding: 0.125rem 0.375rem;
    background: var(--bg-input, #0f172a);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
  }

  .item-content {
    margin: 0;
    font-size: 0.75rem;
    color: var(--text-secondary, #94a3b8);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .item-details {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
  }

  .item-details code {
    background: var(--bg-input, #0f172a);
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-size: 0.6875rem;
  }

  .item-time {
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
  }

  .status-badge {
    font-size: 0.6875rem;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: 500;
  }

  .status-badge.enabled {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .status-badge.disabled {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .priority {
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
  }

  .toggle-btn {
    padding: 0.25rem 0.5rem;
    background: var(--bg-input, #0f172a);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.25rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.6875rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .toggle-btn:hover {
    background: var(--bg-hover, #334155);
    color: var(--text-primary, #e2e8f0);
  }

  .loading-state, .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    color: var(--text-muted, #64748b);
    gap: 0.75rem;
  }
</style>
