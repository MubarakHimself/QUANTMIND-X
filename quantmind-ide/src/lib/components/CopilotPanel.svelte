<script lang="ts">
  import { createEventDispatcher, onMount, tick } from "svelte";
  import {
    X,
    Send,
    Paperclip,
    Settings,
    Bot,
    Code,
    Wand2,
    Loader,
    Terminal,
    FileText,
    BarChart3,
    Plus,
    Power,
    Database,
  } from "lucide-svelte";

  const dispatch = createEventDispatcher();

  const API_BASE = "http://localhost:8000/api";

  let activeAgent = "copilot";
  let message = "";
  let messages: Array<{ role: string; content: string; agent?: string }> = [
    {
      role: "assistant",
      content:
        "Hello! I'm the QuantMind Copilot. I can help you analyze strategies, run backtests, and manage your trading bots. What would you like to do?",
      agent: "copilot",
    },
  ];
  let loading = false;
  let settingsOpen = false;
  let textareaElement: HTMLTextAreaElement;
  let messagesContainer: HTMLDivElement;

  // Auto-resize textarea
  function autoResize() {
    if (textareaElement) {
      textareaElement.style.height = "auto";
      const newHeight = Math.min(
        Math.max(textareaElement.scrollHeight, 80),
        200,
      );
      textareaElement.style.height = newHeight + "px";
    }
  }

  // Character counter
  $: charCount = message.length;

  // Slash Commands
  const slashCommands = [
    {
      command: "/backtest",
      description: "Run backtest on a strategy",
      params: "<strategy_name>",
      icon: BarChart3,
    },
    {
      command: "/attach",
      description: "Attach file to chat context",
      params: "<file_path>",
      icon: Paperclip,
    },
    {
      command: "/analyze",
      description: "Analyze strategy performance",
      params: "<strategy_name>",
      icon: FileText,
    },
    {
      command: "/add-broker",
      description: "Add new broker configuration",
      params: "<broker_name>",
      icon: Plus,
    },
    {
      command: "/kill",
      description: "Execute kill switch",
      params: "",
      icon: Power,
    },
    {
      command: "/memory",
      description: "Show agent memory",
      params: "",
      icon: Database,
    },
    {
      command: "/clear",
      description: "Clear chat history",
      params: "",
      icon: Terminal,
    },
  ];

  let showCommandPalette = false;
  let filteredCommands = slashCommands;
  let selectedCommandIndex = 0;

  // Reactive slash command filtering
  $: {
    if (message.startsWith("/")) {
      showCommandPalette = true;
      const query = message.slice(1).toLowerCase();
      filteredCommands = slashCommands.filter(
        (c) =>
          c.command.slice(1).startsWith(query) ||
          c.description.toLowerCase().includes(query),
      );
      selectedCommandIndex = Math.min(
        selectedCommandIndex,
        Math.max(0, filteredCommands.length - 1),
      );
    } else {
      showCommandPalette = false;
      selectedCommandIndex = 0;
    }
  }

  const agents = [
    {
      id: "copilot",
      name: "Copilot",
      icon: Bot,
      description: "Full system access",
    },
    {
      id: "quantcode",
      name: "QuantCode",
      icon: Code,
      description: "EA development",
    },
    {
      id: "analyst",
      name: "Analyst",
      icon: Wand2,
      description: "Strategy analysis",
    },
  ];

  // AI Settings
  let aiSettings = {
    model: "gemini-2.5-pro",
    temperature: 0.7,
    yoloMode: true,
  };

  // Enhanced Settings State (Sprint 5)
  let activeSettingsTab = "general"; // general, api, mcp
  let apiKeys = {
    openai: "",
    anthropic: "",
    gemini: "",
  };
  let mcpServers = [
    {
      id: "filesystem",
      name: "File System",
      enabled: true,
      status: "connected",
      description: "Access local files",
    },
    {
      id: "brave",
      name: "Brave Search",
      enabled: true,
      status: "connected",
      description: "Web search capability",
    },
    {
      id: "github",
      name: "GitHub",
      enabled: false,
      status: "disconnected",
      description: "Repository integration",
    },
    {
      id: "duckdb",
      name: "DuckDB Analytics",
      enabled: true,
      status: "connected",
      description: "Query parquet files",
    },
  ];

  onMount(() => {
    // Load settings from localStorage
    const storedKeys = localStorage.getItem("quantmind_api_keys");
    if (storedKeys) apiKeys = JSON.parse(storedKeys);

    const storedSettings = localStorage.getItem("quantmind_ai_settings");
    if (storedSettings)
      aiSettings = { ...aiSettings, ...JSON.parse(storedSettings) };

    // Default MCP servers if not stored
    const storedMcp = localStorage.getItem("quantmind_mcp");
    if (storedMcp) {
      const saved = JSON.parse(storedMcp);
      // Merge saved state with definitions to keep descriptions up to date
      mcpServers = mcpServers.map((def: any) => {
        const match = saved.find((s: any) => s.id === def.id);
        return match ? { ...def, enabled: match.enabled } : def;
      });
    }
  });

  function saveSettings() {
    localStorage.setItem("quantmind_api_keys", JSON.stringify(apiKeys));
    localStorage.setItem("quantmind_ai_settings", JSON.stringify(aiSettings));
    localStorage.setItem(
      "quantmind_mcp",
      JSON.stringify(mcpServers.map((s) => ({ id: s.id, enabled: s.enabled }))),
    );
    settingsOpen = false;
  }

  async function sendMessage() {
    if (!message.trim() || loading) return;

    const userMessage = message.trim();

    // Check for client-side slash commands first (e.g. /clear)
    const clientResponse = executeSlashCommand(userMessage);
    if (clientResponse === null && userMessage.startsWith("/")) {
      // Handled specialized command (like /clear) that returns no text
      message = "";
      showCommandPalette = false;
      return;
    }

    messages = [...messages, { role: "user", content: userMessage }];
    message = "";
    showCommandPalette = false;
    loading = true;

    await tick();
    scrollToBottom();

    try {
      if (clientResponse) {
        // If a slash command returned a client-side response immediately
        messages = [
          ...messages,
          { role: "assistant", content: clientResponse, agent: activeAgent },
        ];
      } else {
        // Send to backend
        const res = await fetch(`${API_BASE}/chat/send`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: userMessage,
            agent_id: activeAgent,
            history: messages
              .slice(0, -1)
              .map((m) => ({ role: m.role, content: m.content })),
            model: aiSettings.model,
            api_keys: apiKeys,
          }),
        });

        if (!res.ok) {
          throw new Error(`API Error: ${res.statusText}`);
        }

        const data = await res.json();

        // Add agent response
        messages = [
          ...messages,
          {
            role: "assistant",
            content: data.reply,
            agent: data.agent_id,
          },
        ];
      }
    } catch (e) {
      console.error("Chat error:", e);
      messages = [
        ...messages,
        {
          role: "assistant",
          content:
            "Unable to connect to agent backend. Please ensure the server is running.",
          agent: "system",
        },
      ];
    } finally {
      loading = false;
      await tick();
      scrollToBottom();
    }
  }

  // Auto-scroll to bottom
  async function scrollToBottom() {
    await tick();
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }

  function generateCopilotResponse(input: string): string {
    const lower = input.toLowerCase();

    if (lower.includes("backtest")) {
      return "I can run backtests in three modes:\n\n**Mode A** - EA only (fixed lot)\n**Mode B** - EA + Kelly sizing\n**Mode C** - EA + Full System (Kelly + Governor)\n\nWhich strategy folder would you like to backtest?";
    }
    if (lower.includes("nprd") || lower.includes("video")) {
      return "To process a new NPRD:\n\n1. Go to **EA Management** in the sidebar\n2. Click **New Strategy**\n3. Paste the YouTube URL\n\nThe Gemini CLI will transcribe and analyze the video, creating a new strategy folder.";
    }
    if (lower.includes("bot") || lower.includes("trading")) {
      return "You currently have **3 bots** active. Click the **Active** count in the status bar or the **Kill Switch** to manage live trading.\n\nFrom there you can:\n- Pause/resume individual bots\n- Quarantine problematic EAs\n- Trigger emergency kill all";
    }
    if (lower.includes("kelly") || lower.includes("risk")) {
      return "Current risk settings:\n\n- **Kelly Factor**: 0.85\n- **Balance Zone**: Growth Tier\n- **House Money**: +$12 today\n- **Squad Limit**: 8 bots max\n\nThe system automatically adjusts position sizing based on regime and daily performance.";
    }

    return "I understand. How can I help you with your trading system today? I can assist with:\n\n- Processing NPRDs from YouTube videos\n- Running backtests (Mode A/B/C)\n- Managing live bots\n- Analyzing strategy performance";
  }

  function switchAgent(agentId: string) {
    activeAgent = agentId;
    messages = [
      {
        role: "assistant",
        content: getWelcomeMessage(agentId),
        agent: agentId,
      },
    ];
  }

  function getWelcomeMessage(agentId: string): string {
    switch (agentId) {
      case "copilot":
        return "Hello! I'm the QuantMind Copilot with full system access. I can help you analyze strategies, run backtests, and manage your trading bots.";
      case "quantcode":
        return "I'm QuantCode, specialized in EA development. I can convert TRDs into MQL5 code, optimize strategies, and run backtests.";
      case "analyst":
        return "I'm the Analyst agent. I convert NPRDs into Technical Requirements Documents, correlating with your knowledge base and shared assets.";
      default:
        return "How can I assist you?";
    }
  }

  function toggleSettings() {
    settingsOpen = !settingsOpen;
  }

  function handleKeydown(e: KeyboardEvent) {
    // Handle command palette navigation
    if (showCommandPalette && filteredCommands.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        selectedCommandIndex =
          (selectedCommandIndex + 1) % filteredCommands.length;
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        selectedCommandIndex =
          (selectedCommandIndex - 1 + filteredCommands.length) %
          filteredCommands.length;
        return;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        selectCommand(filteredCommands[selectedCommandIndex]);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        showCommandPalette = false;
        return;
      }
    }

    // Normal enter to send
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function selectCommand(cmd: (typeof slashCommands)[0]) {
    message = cmd.command + " ";
    showCommandPalette = false;
    selectedCommandIndex = 0;
  }

  function executeSlashCommand(userMessage: string): string | null {
    // Check if message is a slash command and handle it
    if (userMessage.startsWith("/kill")) {
      // Trigger kill switch
      fetch(`${API_BASE}/trading/kill`, { method: "POST" });
      return "Kill switch activated. All trading bots have been stopped.";
    }
    if (userMessage.startsWith("/clear")) {
      messages = [
        {
          role: "assistant",
          content: getWelcomeMessage(activeAgent),
          agent: activeAgent,
        },
      ];
      return null; // Don't add response, we cleared
    }
    if (userMessage.startsWith("/memory")) {
      return "Memory contents:\n\n**Semantic**: 12 facts stored\n**Episodic**: 34 conversation episodes\n**Procedural**: 8 learned procedures\n\nUse `/memory clear` to reset.";
    }
    return null; // Not a special command, process normally
  }
</script>

<aside class="copilot-panel">
  <div class="panel-header">
    <div class="agent-tabs">
      {#each agents as agent}
        <button
          class="agent-tab"
          class:active={activeAgent === agent.id}
          on:click={() => switchAgent(agent.id)}
          title={agent.description}
        >
          <svelte:component this={agent.icon} size={16} />
          <span>{agent.name}</span>
        </button>
      {/each}
    </div>

    <div class="header-actions">
      <button class="icon-btn" title="AI Settings" on:click={toggleSettings}>
        <Settings size={16} />
      </button>
      <button class="icon-btn" title="Close" on:click={() => dispatch("close")}>
        <X size={16} />
      </button>
    </div>
  </div>

  {#if settingsOpen}
    <div class="settings-panel">
      <div class="settings-header">
        <h3>Assistant Settings</h3>
        <button class="icon-btn" on:click={() => (settingsOpen = false)}
          ><X size={16} /></button
        >
      </div>

      <div class="settings-tabs">
        <button
          class:active={activeSettingsTab === "general"}
          on:click={() => (activeSettingsTab = "general")}>General</button
        >
        <button
          class:active={activeSettingsTab === "api"}
          on:click={() => (activeSettingsTab = "api")}>API Keys</button
        >
        <button
          class:active={activeSettingsTab === "mcp"}
          on:click={() => (activeSettingsTab = "mcp")}>Skills (MCP)</button
        >
      </div>

      <div class="settings-content">
        {#if activeSettingsTab === "general"}
          <div class="setting-group">
            <label for="ai-model">Model</label>
            <select id="ai-model" bind:value={aiSettings.model}>
              <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
              <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
              <option value="claude-3-5-sonnet">Claude 3.5 Sonnet</option>
              <option value="gpt-4o">GPT-4o</option>
            </select>
          </div>
          <div class="setting-group">
            <label for="ai-temp">Temperature ({aiSettings.temperature})</label>
            <input
              id="ai-temp"
              type="range"
              min="0"
              max="1"
              step="0.1"
              bind:value={aiSettings.temperature}
            />
          </div>
          <div class="setting-group checkbox">
            <input
              type="checkbox"
              id="yolo"
              bind:checked={aiSettings.yoloMode}
            />
            <label for="yolo">YOLO Mode (Skip confirmations)</label>
          </div>
        {:else if activeSettingsTab === "api"}
          <div class="setting-group">
            <label for="openai-key">OpenAI API Key</label>
            <input
              id="openai-key"
              type="password"
              placeholder="sk-..."
              bind:value={apiKeys.openai}
            />
          </div>
          <div class="setting-group">
            <label for="anthropic-key">Anthropic API Key</label>
            <input
              id="anthropic-key"
              type="password"
              placeholder="sk-ant-..."
              bind:value={apiKeys.anthropic}
            />
          </div>
          <div class="setting-group">
            <label for="gemini-key">Gemini API Key</label>
            <input
              id="gemini-key"
              type="password"
              placeholder="AIza..."
              bind:value={apiKeys.gemini}
            />
          </div>
        {:else if activeSettingsTab === "mcp"}
          <div class="mcp-list">
            {#each mcpServers as server}
              <div class="mcp-item">
                <div class="mcp-header">
                  <span class="mcp-name">{server.name}</span>
                  <input type="checkbox" bind:checked={server.enabled} />
                </div>
                <div class="mcp-desc">{server.description}</div>
                <div
                  class="mcp-status"
                  class:connected={server.status === "connected"}
                >
                  <span class="status-dot"></span>
                  {server.status}
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <div class="settings-footer">
        <button class="btn-save" on:click={saveSettings}>Save Changes</button>
      </div>
    </div>
  {/if}

  <div class="messages" bind:this={messagesContainer}>
    {#each messages as msg}
      <div class="message {msg.role}">
        <div class="message-content">
          {#if msg.role === "assistant"}
            <div class="agent-badge">
              {agents.find((a) => a.id === msg.agent)?.name || "Assistant"}
            </div>
          {/if}
          <div class="message-text">
            {@html msg.content
              .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
              .replace(/\n/g, "<br>")}
          </div>
        </div>
      </div>
    {/each}

    {#if loading}
      <div class="message assistant">
        <div class="message-content">
          <div class="typing-indicator">
            <Loader size={16} class="spinning" />
            <span>Thinking...</span>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <div class="input-area">
    {#if showCommandPalette && filteredCommands.length > 0}
      <div class="command-palette">
        {#each filteredCommands as cmd, i}
          <button
            class="command-item"
            class:selected={i === selectedCommandIndex}
            on:click={() => selectCommand(cmd)}
            on:mouseenter={() => (selectedCommandIndex = i)}
          >
            <svelte:component this={cmd.icon} size={14} class="cmd-icon" />
            <span class="cmd-name">{cmd.command}</span>
            {#if cmd.params}
              <span class="cmd-params">{cmd.params}</span>
            {/if}
            <span class="cmd-desc">{cmd.description}</span>
          </button>
        {/each}
      </div>
    {/if}

    <div class="input-wrapper">
      <textarea
        placeholder="Ask {agents.find((a) => a.id === activeAgent)?.name}..."
        bind:value={message}
        on:keydown={handleKeydown}
        on:input={autoResize}
        bind:this={textareaElement}
        rows="1"
      ></textarea>

      <div class="model-badge-inline">
        <span class="model-name-tiny"
          >{aiSettings.model.replace("gemini-", "")}</span
        >
      </div>
    </div>

    <div class="input-footer">
      <div class="input-actions">
        <button class="action-btn" title="Attach Context">
          <Paperclip size={14} />
          <span>Context</span>
        </button>
        <button
          class="action-btn"
          title="Model Config"
          on:click={toggleSettings}
        >
          <Settings size={14} />
          <span>Config</span>
        </button>
      </div>

      <div class="input-actions">
        <div class="char-counter">
          {charCount} / 4000
        </div>
        <button
          class="action-btn primary"
          on:click={sendMessage}
          disabled={!message.trim() || loading}
        >
          {#if loading}
            <Loader size={14} class="spinning" />
          {:else}
            <Send size={14} />
          {/if}
        </button>
      </div>
    </div>
  </div>
</aside>
```

<style>
  .copilot-panel {
    grid-column: 4;
    grid-row: 1;
    display: flex;
    flex-direction: column;
    width: 380px;
    background: var(--bg-secondary);
    border-left: 1px solid var(--border-subtle);
    font-family: "Inter", sans-serif;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: var(--header-height);
    padding: 0 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .agent-tabs {
    display: flex;
    gap: 2px;
    height: 100%;
  }

  .agent-tab {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 12px;
    height: 100%;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.1s ease;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .agent-tab:hover {
    color: var(--text-primary);
  }

  .agent-tab.active {
    color: var(--accent-primary);
    border-bottom-color: var(--accent-primary);
  }

  .header-actions {
    display: flex;
    gap: 4px;
    align-items: center;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  /* Settings Panel - Flat & clean */
  .settings-panel {
    padding: 0;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .settings-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .settings-header h3 {
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-secondary);
    font-weight: 600;
  }

  .settings-tabs {
    display: flex;
    padding: 0 12px;
    gap: 16px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 12px;
  }

  .settings-tabs button {
    background: none;
    border: none;
    padding: 8px 0;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }

  .settings-tabs button.active {
    color: var(--accent-primary);
    border-bottom-color: var(--accent-primary);
  }

  .settings-content {
    padding: 0 12px 12px;
  }

  .setting-group {
    margin-bottom: 12px;
  }

  .setting-group label {
    display: block;
    font-size: 11px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .setting-group select,
  .setting-group input[type="password"] {
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: 6px 8px;
    border-radius: 2px;
    font-size: 12px;
    font-family: inherit;
  }

  .setting-group select:focus,
  .setting-group input:focus {
    border-color: var(--accent-primary);
    outline: none;
  }

  .setting-group.checkbox {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .setting-group.checkbox input {
    margin: 0;
  }

  .setting-group.checkbox label {
    margin: 0;
  }

  /* MCP List - Technical list style */
  .mcp-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .mcp-item {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    padding: 8px;
    border-radius: 4px;
  }
  .mcp-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
  }
  .mcp-name {
    font-weight: 600;
    font-size: 11px;
    color: var(--accent-secondary);
  }
  .mcp-desc {
    font-size: 10px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }
  .mcp-status {
    font-size: 9px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
  }
  .mcp-status.connected .status-dot {
    background: var(--accent-success);
  }
  .mcp-status.connected {
    color: var(--accent-success);
  }

  .settings-footer {
    padding: 12px;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    justify-content: flex-end;
  }

  .btn-save {
    background: var(--accent-primary);
    color: var(--bg-primary);
    border: none;
    padding: 6px 12px;
    border-radius: 2px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }

  /* Consolidated Layout Styles */
  .copilot-panel {
    grid-column: 3; /* Matches +page.svelte layout */
    grid-row: 2;
    display: flex;
    flex-direction: column;
    width: 380px;
    background: var(--bg-secondary);
    border-left: 1px solid var(--border-subtle);
    font-family: inherit;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: var(--header-height);
    padding: 0 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  /* Messaging Area */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 0;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
  }

  .message {
    width: 100%;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
    transition: background 0.1s ease;
  }

  .message.user {
    background: var(--bg-primary);
  }

  .message.assistant {
    background: var(--bg-input);
  }

  .message-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .message-content {
    color: var(--text-primary);
    line-height: 1.6;
    font-size: 13px;
  }

  /* Input Area - Cursor Aesthetic */
  .input-area {
    padding: 16px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-subtle);
  }

  .input-wrapper {
    position: relative;
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    transition: border-color 0.1s ease;
  }

  .input-wrapper:focus-within {
    border-color: var(--accent-primary);
  }

  .input-wrapper textarea {
    width: 100%;
    background: transparent;
    border: none;
    color: var(--text-primary);
    padding: 10px;
    outline: none;
    resize: none;
    font-family: inherit;
    min-height: 80px;
    max-height: 200px;
    line-height: 1.5;
  }

  .model-badge-inline {
    position: absolute;
    bottom: 8px;
    right: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-tertiary);
    padding: 2px 6px;
    border-radius: 2px;
    border: 1px solid var(--border-subtle);
    pointer-events: none;
  }

  .model-name-tiny {
    font-size: 9px;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .input-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
    padding: 0 4px;
  }

  .input-actions {
    display: flex;
    gap: 6px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.1s ease;
  }

  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .action-btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
    font-weight: 600;
  }

  .char-counter {
    font-size: 10px;
    color: var(--text-muted);
  }

  /* Command Palette */
  .command-palette {
    position: absolute;
    bottom: 100%;
    left: 0;
    right: 0;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 4px;
    margin-bottom: 8px;
    box-shadow: 0 -8px 24px rgba(0, 0, 0, 0.5);
    z-index: 100;
  }

  .command-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: none;
    border-radius: 2px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    text-align: left;
  }

  .command-item:hover,
  .command-item.selected {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .cmd-name {
    font-weight: 600;
  }
  .cmd-params {
    color: var(--text-muted);
    font-size: 10px;
    font-family: monospace;
  }
  .cmd-desc {
    margin-left: auto;
    font-size: 10px;
    opacity: 0.7;
  }

  /* Settings Panel */
  .settings-panel {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 100%;
    background: var(--bg-secondary);
    z-index: 200;
    display: flex;
    flex-direction: column;
  }

  .settings-header {
    padding: 12px 16px;
    height: var(--header-height);
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .settings-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .settings-tabs {
    display: flex;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-input);
  }

  .settings-tabs button {
    flex: 1;
    padding: 8px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }

  .settings-tabs button.active {
    color: var(--accent-primary);
    border-bottom-color: var(--accent-primary);
    background: var(--bg-secondary);
  }

  .settings-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  .setting-group {
    margin-bottom: 16px;
  }
  .setting-group label {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }
  .setting-group input,
  .setting-group select {
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: 8px;
    border-radius: 4px;
    font-size: 12px;
  }

  .setting-group.checkbox {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .setting-group.checkbox label {
    margin: 0;
  }

  .mcp-item {
    background: var(--bg-tertiary);
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 12px;
    border: 1px solid var(--border-subtle);
  }

  .mcp-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
  }
  .mcp-name {
    font-weight: 600;
    font-size: 12px;
    color: var(--text-primary);
  }
  .mcp-desc {
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .mcp-status {
    font-size: 10px;
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--text-muted);
  }
  .mcp-status.connected {
    color: #10b981;
  }
  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
  }

  .settings-footer {
    padding: 16px;
    border-top: 1px solid var(--border-subtle);
  }
  .btn-save {
    width: 100%;
    background: var(--accent-primary);
    color: white;
    border: none;
    padding: 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
  }

  .input-wrapper {
    position: relative;
    flex: 1;
  }

  .model-badge-inline {
    position: absolute;
    top: 6px;
    right: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 9px;
    pointer-events: none;
  }

  .model-name-tiny {
    color: var(--text-muted);
    font-family: monospace;
  }
  .yolo-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-warning);
  }

  .attachment-btn {
    display: flex;
    align-items: center;
    padding: 6px;
    color: var(--text-muted);
  }
</style>
