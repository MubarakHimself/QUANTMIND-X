<script lang="ts">
  import { run } from 'svelte/legacy';

  import {
    Terminal,
    Activity,
    AlertCircle,
    FileText,
    ChevronUp,
    ChevronDown,
    X,
    Play,
    Settings,
    Wifi,
    WifiOff,
    ExternalLink,
    Copy,
    Trash2,
    Palette,
    Wrench,
    RefreshCw,
  } from "lucide-svelte";

  // Terminal color theme
  type TerminalTheme = 'default' | 'solarized' | 'dracula' | 'nord';
  let terminalTheme: TerminalTheme = $state('nord');

  const themes: Record<TerminalTheme, {
    bg: string;
    fg: string;
    prompt: string;
    input: string;
    output: string;
    error: string;
    success: string;
    warning: string;
    info: string;
  }> = {
    default: {
      bg: 'var(--color-bg-elevated)',
      fg: 'var(--color-text-secondary)',
      prompt: 'var(--color-accent-cyan)',
      input: 'var(--color-accent-cyan)',
      output: 'var(--color-text-secondary)',
      error: '#ef4444',
      success: '#10b981',
      warning: '#f59e0b',
      info: '#3b82f6',
    },
    solarized: {
      bg: '#002b36',
      fg: '#839496',
      prompt: '#b58900',
      input: '#268bd2',
      output: '#839496',
      error: '#dc322f',
      success: '#859900',
      warning: '#b58900',
      info: '#268bd2',
    },
    dracula: {
      bg: '#282a36',
      fg: '#f8f8f2',
      prompt: '#ff79c6',
      input: '#8be9fd',
      output: '#f8f8f2',
      error: '#ff5555',
      success: '#50fa7b',
      warning: '#f1fa8c',
      info: '#8be9fd',
    },
    nord: {
      bg: '#2e3440',
      fg: '#d8dee9',
      prompt: '#88c0d0',
      input: '#81a1c1',
      output: '#d8dee9',
      error: '#bf616a',
      success: '#a3be8c',
      warning: '#ebcb8b',
      info: '#5e81ac',
    },
  };

  let currentTheme = $derived(themes[terminalTheme]);

  let activeTab = $state("output");  // Default to output, not terminal
  let isExpanded = $state(false);
  let terminalInput = $state("");
  let showThemePicker = $state(false);

  // Tool call logging state
  let toolCalls: Array<{
    id: string;
    timestamp: string;
    agent_id: string;
    agent_type: string;
    tool_name: string;
    args: Record<string, unknown>;
    result?: string;
    duration_ms?: number;
    success: boolean;
    error?: string;
  }> = $state([]);
  let toolCallsLoading = $state(false);
  let toolCallsError: string | null = $state(null);

  // Fetch tool call logs from API
  async function fetchToolCalls(limit: number = 50) {
    toolCallsLoading = true;
    toolCallsError = null;
    try {
      const response = await fetch(`/api/tool-calls/logs?limit=${limit}`);
      const data = await response.json();
      if (data.logs) {
        toolCalls = data.logs;
      }
    } catch (e) {
      toolCallsError = e instanceof Error ? e.message : 'Failed to fetch tool calls';
    } finally {
      toolCallsLoading = false;
    }
  }

  // Format tool call for terminal display
  function formatToolCall(call: typeof toolCalls[0]): string {
    const ts = new Date(call.timestamp).toLocaleTimeString('en-US', { hour12: false });
    const status = call.success ? 'OK' : 'ERR';
    const duration = call.duration_ms ? ` [${call.duration_ms.toFixed(1)}ms]` : '';
    const argsStr = JSON.stringify(call.args);
    const truncatedArgs = argsStr.length > 40 ? argsStr.slice(0, 37) + '...' : argsStr;
    return `[${ts}] ${call.agent_id.padEnd(15)} ${call.tool_name.padEnd(25)} ${status}${duration} ${truncatedArgs}`;
  }

  let terminalHistory: Array<{
    type: "input" | "output" | "error" | "success" | "warning" | "info";
    content: string;
    timestamp?: string;
  }> = $state([
    { type: "info", content: "QuantMind Terminal v1.0.0", timestamp: getTimestamp() },
    { type: "info", content: 'Type "help" for available commands', timestamp: getTimestamp() },
    { type: "output", content: "", timestamp: getTimestamp() },
  ]);

  function getTimestamp(): string {
    return new Date().toLocaleTimeString('en-US', { hour12: false });
  }

  let mt5Status = $state("disconnected");
  let mt5Path = $state("/opt/MetaTrader5/terminal64");

  // Connection error state
  let mt5Error: string | null = null;

  let logs: Array<{ type: string; message: string; time: string }> = $state([
    {
      type: "info",
      message: "QuantMind IDE started",
      time: new Date().toLocaleTimeString(),
    },
    {
      type: "success",
      message: "Backend API connected",
      time: new Date().toLocaleTimeString(),
    },
    {
      type: "info",
      message: "Kelly router initialized (k=0.85)",
      time: new Date().toLocaleTimeString(),
    },
  ]);

  let errors: Array<{ message: string; file?: string; line?: number; timestamp?: string }> = [];

  const tabs = [
    { id: "terminal", label: "Terminal", icon: Terminal },
    { id: "mt5", label: "MT5 Sync", icon: Activity },
    { id: "errors", label: "Errors", icon: AlertCircle, count: errors.length },
    { id: "tool_calls", label: "Tool Calls", icon: Wrench },
    { id: "output", label: "Output", icon: FileText },
  ];

  // Auto-fetch tool calls when tab is activated
  run(() => {
    if (activeTab === "tool_calls" && toolCalls.length === 0) {
      fetchToolCalls(50);
    }
  });

  // Enhanced terminal commands with better output
  const terminalCommands: Record<string, () => { type: string; content: string }> = {
    help: () => ({ type: 'output', content: `
Available Commands:
  status       Show system status
  bots         List active trading bots
  regime       Show current market regime
  kelly        Show Kelly criterion settings
  video-ingest List VideoIngest queue
  theme        Change terminal theme
  clear        Clear terminal
  copy         Copy last output to clipboard
  logs [n]     Show last n log entries (default: 10)
` }),
    status: () => ({ type: 'output', content: `
System Status:
  [OK]   Backend API: Connected
  [OK]   Kelly Router: Initialized (k=0.85)
  [OK]   Database: Connected
  [----] MT5: ${mt5Status === 'connected' ? 'Connected' : 'Disconnected'}
` }),
    bots: () => ({ type: 'output', content: `
Active Trading Bots:
  #1  ICT_Scalper  @EURUSD  primal   +$450.25
  #2  ICT_Scalper  @GBPUSD  primal   +$320.10
  #3  SMC_Rev      @USDJPY  ready    +$480.15
` }),
    regime: () => ({ type: 'output', content: `
Market Regime: TRENDING
  Volatility: HIGH
  Trend: BULLISH
  Confidence: 85%
` }),
    kelly: () => ({ type: 'output', content: `
Kelly Criterion Settings:
  Fraction: 0.85
  Max Drawdown: 20%
  Position Sizing: Dynamic
` }),
    video_ingest: () => ({ type: 'output', content: `
VideoIngest Queue:
  Status: Empty
  Processing: 0
  Completed: 0
  Failed: 0
` }),
    theme: () => {
      const themeNames = Object.keys(themes) as TerminalTheme[];
      const currentIndex = themeNames.indexOf(terminalTheme);
      const nextTheme = themeNames[(currentIndex + 1) % themeNames.length];
      terminalTheme = nextTheme;
      return { type: 'success', content: `Theme changed to: ${nextTheme.toUpperCase()}` };
    },
    clear: () => {
      terminalHistory = [];
      return { type: 'info', content: 'Terminal cleared' };
    },
    copy: () => {
      const lastOutput = terminalHistory.filter(h => h.type !== 'input').pop()?.content || '';
      navigator.clipboard.writeText(lastOutput);
      return { type: 'success', content: 'Copied to clipboard' };
    },
    logs: (args: string) => {
      const count = parseInt(args) || 10;
      const recentLogs = logs.slice(-count).map(l => `[${l.time}] ${l.type.toUpperCase()}: ${l.message}`).join('\n');
      return { type: 'output', content: recentLogs || 'No logs available' };
    },
  };

  function handleTerminalInput(e: KeyboardEvent) {
    if (e.key === "Enter" && terminalInput.trim()) {
      const input = terminalInput.trim();
      const parts = input.toLowerCase().split(/\s+/);
      const cmd = parts[0];
      const args = parts.slice(1).join(' ');

      terminalHistory = [
        ...terminalHistory,
        { type: "input", content: `$ ${terminalInput}`, timestamp: getTimestamp() },
      ];

      if (terminalCommands[cmd]) {
        const result = terminalCommands[cmd](args);
        terminalHistory = [
          ...terminalHistory,
          { type: result.type as any, content: result.content, timestamp: getTimestamp() },
        ];
      } else if (cmd === 'kill') {
        terminalHistory = [
          ...terminalHistory,
          { type: "warning", content: "⚠️ Kill switch not triggered from terminal. Use the UI button.", timestamp: getTimestamp() },
        ];
      } else {
        terminalHistory = [
          ...terminalHistory,
          { type: "error", content: `Command not found: ${cmd}. Type "help" for available commands.`, timestamp: getTimestamp() },
        ];
      }

      terminalInput = "";

      // Scroll to bottom
      setTimeout(() => {
        const container = document.querySelector(".terminal-scroll");
        if (container) container.scrollTop = container.scrollHeight;
      }, 10);
    }
  }

  function clearTerminal() {
    terminalHistory = [];
    terminalHistory = [
      { type: "info", content: "Terminal cleared", timestamp: getTimestamp() },
      { type: "output", content: "", timestamp: getTimestamp() },
    ];
  }

  async function connectMT5() {
    mt5Error = null;
    mt5Status = "connecting";
    logs = [
      ...logs,
      {
        type: "info",
        message: "Attempting to connect to MT5...",
        time: new Date().toLocaleTimeString(),
      },
    ];

    // Simulate connection attempt
    setTimeout(() => {
      // In real implementation, this would use Tauri commands to open MT5
      mt5Status = "connected";
      logs = [
        ...logs,
        {
          type: "success",
          message: "MT5 connection established",
          time: new Date().toLocaleTimeString(),
        },
      ];
    }, 2000);
  }

  function openMT5Terminal() {
    // Would use Tauri shell command: await shell.open(mt5Path)
    logs = [
      ...logs,
      {
        type: "info",
        message: "Opening MT5 Terminal...",
        time: new Date().toLocaleTimeString(),
      },
    ];
    window.open("file://" + mt5Path, "_blank");
  }

  function disconnectMT5() {
    mt5Status = "disconnected";
    logs = [
      ...logs,
      {
        type: "info",
        message: "MT5 disconnected",
        time: new Date().toLocaleTimeString(),
      },
    ];
  }

  function toggleThemePicker() {
    showThemePicker = !showThemePicker;
  }

  function selectTheme(theme: TerminalTheme) {
    terminalTheme = theme;
    showThemePicker = false;
  }

  function getTypeColor(type: string): string {
    switch (type) {
      case 'input': return currentTheme.input;
      case 'output': return currentTheme.output;
      case 'error': return currentTheme.error;
      case 'success': return currentTheme.success;
      case 'warning': return currentTheme.warning;
      case 'info': return currentTheme.info;
      default: return currentTheme.fg;
    }
  }
</script>

<div class="bottom-panel" class:expanded={isExpanded}>
  <div class="panel-header">
    <div class="tabs">
      {#each tabs as tab}
        <button
          class="tab"
          class:active={activeTab === tab.id}
          onclick={() => (activeTab = tab.id)}
        >
          <tab.icon size={14} />
          <span>{tab.label}</span>
          {#if tab.count}<span class="badge">{tab.count}</span>{/if}
        </button>
      {/each}
    </div>
    <div class="actions">
      <button onclick={() => (isExpanded = !isExpanded)}>
        {#if isExpanded}<ChevronDown size={14} />{:else}<ChevronUp
            size={14}
          />{/if}
      </button>
      <button onclick={() => (isExpanded = false)}><X size={14} /></button>
    </div>
  </div>

  {#if isExpanded}
    <div class="panel-content">
      {#if activeTab === "terminal"}
        <div class="terminal" style="--term-bg: {currentTheme.bg}; --term-fg: {currentTheme.fg}">
          <div class="terminal-toolbar">
            <div class="terminal-tabs">
              <span class="terminal-tab active">bash</span>
            </div>
            <div class="terminal-actions">
              <button class="term-btn" onclick={clearTerminal} title="Clear terminal">
                <Trash2 size={12} />
              </button>
              <button class="term-btn" onclick={toggleThemePicker} title="Change theme">
                <Palette size={12} />
              </button>
            </div>
            {#if showThemePicker}
              <div class="theme-picker">
                {#each Object.keys(themes) as theme}
                  <button
                    class="theme-option"
                    class:active={terminalTheme === theme}
                    onclick={() => selectTheme(theme as TerminalTheme)}
                  >
                    {theme}
                  </button>
                {/each}
              </div>
            {/if}
          </div>
          <div class="terminal-scroll">
            {#each terminalHistory as line}
              <div class="term-line" style="color: {getTypeColor(line.type)}">
                {#if line.timestamp}
                  <span class="term-timestamp">[{line.timestamp}]</span>
                {/if}
                <span class="term-content">{line.content}</span>
              </div>
            {/each}
          </div>
          <div class="terminal-input">
            <span class="prompt" style="color: {currentTheme.prompt}">$</span>
            <input
              type="text"
              bind:value={terminalInput}
              onkeydown={handleTerminalInput}
              placeholder="Enter command..."
              style="color: {currentTheme.fg}"
            />
          </div>
        </div>
      {:else if activeTab === "mt5"}
        <div class="mt5-sync">
          <div class="mt5-status">
            {#if mt5Status === "connected"}
              <Wifi size={20} color="#10b981" />
              <span class="status-text connected">Connected to MT5</span>
            {:else if mt5Status === "connecting"}
              <Activity size={20} class="spin" />
              <span class="status-text">Connecting...</span>
            {:else}
              <WifiOff size={20} color="#ef4444" />
              <span class="status-text disconnected">Disconnected</span>
            {/if}
          </div>

          <div class="mt5-actions">
            {#if mt5Status === "connected"}
              <button class="mt5-btn" onclick={disconnectMT5}
                >Disconnect</button
              >
            {:else if mt5Status !== "connecting"}
              <button class="mt5-btn primary" onclick={connectMT5}
                >Connect to MT5</button
              >
            {/if}
            <button class="mt5-btn" onclick={openMT5Terminal}>
              <ExternalLink size={12} /> Open MT5 Terminal
            </button>
          </div>

          <div class="mt5-config">
            <label for="mt5-path">MT5 Path</label>
            <input
              id="mt5-path"
              type="text"
              bind:value={mt5Path}
              placeholder="/path/to/terminal64"
            />
          </div>

          <div class="mt5-info">
            <div class="info-row">
              <span>Last Sync:</span><span
                >{mt5Status === "connected" ? "Just now" : "Never"}</span
              >
            </div>
            <div class="info-row">
              <span>Accounts:</span><span
                >{mt5Status === "connected" ? "2 connected" : "0"}</span
              >
            </div>
            <div class="info-row">
              <span>Open Trades:</span><span
                >{mt5Status === "connected" ? "5" : "0"}</span
              >
            </div>
          </div>
        </div>
      {:else if activeTab === "errors"}
        <div class="errors-panel">
          {#if errors.length === 0}
            <div class="empty-state">
              <AlertCircle size={24} />
              <span>No errors</span>
            </div>
          {:else}
            {#each errors as error}
              <div class="error-item">
                <AlertCircle size={14} />
                <span class="error-msg">{error.message}</span>
                {#if error.file}
                  <span class="error-loc">{error.file}:{error.line}</span>
                {/if}
              </div>
            {/each}
          {/if}
        </div>
      {:else if activeTab === "tool_calls"}
        <div class="tool-calls-panel">
          <div class="tool-calls-toolbar">
            <button class="tool-btn" onclick={() => fetchToolCalls(50)} title="Refresh tool calls">
              <RefreshCw size={12} />
            </button>
            <span class="tool-calls-count">{toolCalls.length} calls</span>
          </div>
          <div class="tool-calls-scroll">
            {#if toolCallsLoading}
              <div class="tool-calls-loading">
                <RefreshCw size={16} class="spin" />
                <span>Loading tool calls...</span>
              </div>
            {:else if toolCallsError}
              <div class="tool-calls-error">
                <AlertCircle size={14} />
                <span>{toolCallsError}</span>
              </div>
            {:else if toolCalls.length === 0}
              <div class="tool-calls-empty">
                <Wrench size={24} />
                <span>No tool calls recorded</span>
              </div>
            {:else}
              <div class="tool-calls-header">
                <span class="tc-time">TIME</span>
                <span class="tc-agent">AGENT</span>
                <span class="tc-tool">TOOL</span>
                <span class="tc-status">STATUS</span>
                <span class="tc-duration">DURATION</span>
              </div>
              {#each toolCalls as call}
                <div class="tool-call-line" class:error={!call.success}>
                  <span class="tc-time">{new Date(call.timestamp).toLocaleTimeString('en-US', { hour12: false })}</span>
                  <span class="tc-agent" title={call.agent_id}>{call.agent_id.slice(0, 12)}</span>
                  <span class="tc-tool" title={call.tool_name}>{call.tool_name.slice(0, 20)}</span>
                  <span class="tc-status" class:success={call.success} class:failed={!call.success}>
                    {call.success ? 'OK' : 'ERR'}
                  </span>
                  <span class="tc-duration">
                    {call.duration_ms ? `${call.duration_ms.toFixed(1)}ms` : '-'}
                  </span>
                </div>
              {/each}
            {/if}
          </div>
        </div>
      {:else if activeTab === "output"}
        <div class="logs-panel">
          {#each logs as log}
            <div class="log-line {log.type}">
              <span class="time">{log.time}</span>
              <span class="msg">{log.message}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .bottom-panel {
    grid-area: bottom;
    background: var(--color-bg-surface);
    border-top: 1px solid var(--color-border-subtle);
    min-height: 32px;
  }
  .bottom-panel.expanded {
    height: 220px;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    padding: 0 8px;
    height: 32px;
    border-bottom: 1px solid var(--color-border-subtle);
  }
  .tabs {
    display: flex;
    gap: 2px;
  }
  .tab {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 12px;
    height: 32px;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    font-size: 11px;
    cursor: pointer;
  }
  .tab:hover {
    color: var(--color-text-primary);
  }
  .tab.active {
    color: var(--color-accent-cyan);
    border-bottom: 2px solid var(--color-accent-cyan);
  }
  .badge {
    background: #ef4444;
    color: white;
    font-size: 10px;
    padding: 0 5px;
    border-radius: 8px;
  }
  .actions {
    display: flex;
    gap: 4px;
    align-items: center;
  }
  .actions button {
    background: none;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 4px;
  }

  .panel-content {
    height: calc(100% - 32px);
    overflow: hidden;
  }

  /* Terminal */
  .terminal {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--term-bg, var(--color-bg-elevated));
  }
  .terminal-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 8px;
    background: rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid var(--color-border-subtle);
    position: relative;
  }
  .terminal-tabs {
    display: flex;
    gap: 4px;
  }
  .terminal-tab {
    padding: 4px 12px;
    font-size: 11px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px 4px 0 0;
    color: var(--term-fg, var(--color-text-secondary));
    opacity: 0.7;
  }
  .terminal-tab.active {
    background: rgba(255, 255, 255, 0.1);
    opacity: 1;
  }
  .terminal-actions {
    display: flex;
    gap: 4px;
  }
  .term-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    border-radius: 4px;
  }
  .term-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: var(--color-text-primary);
  }
  .theme-picker {
    position: absolute;
    top: 100%;
    right: 8px;
    display: flex;
    gap: 4px;
    padding: 8px;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    z-index: 10;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }
  .theme-option {
    padding: 4px 10px;
    font-size: 10px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 4px;
    color: var(--color-text-secondary);
    cursor: pointer;
    text-transform: capitalize;
  }
  .theme-option:hover {
    border-color: var(--color-accent-cyan);
  }
  .theme-option.active {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
    border-color: var(--color-accent-cyan);
  }
  .terminal-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 8px 12px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 12px;
    line-height: 1.5;
  }
  .term-line {
    line-height: 1.6;
    white-space: pre-wrap;
    display: flex;
    gap: 8px;
  }
  .term-timestamp {
    color: var(--color-text-muted);
    font-size: 10px;
    opacity: 0.6;
    flex-shrink: 0;
  }
  .term-content {
    flex: 1;
  }
  .terminal-input {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-top: 1px solid var(--color-border-subtle);
    background: rgba(0, 0, 0, 0.2);
  }
  .prompt {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-weight: bold;
  }
  .terminal-input input {
    flex: 1;
    background: transparent;
    border: none;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 12px;
    outline: none;
  }
  .terminal-input input::placeholder {
    color: var(--color-text-muted);
    opacity: 0.5;
  }

  /* MT5 Sync */
  .mt5-sync {
    padding: 16px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .mt5-status {
    display: flex;
    align-items: center;
    gap: 12px;
    grid-column: span 2;
    padding: 12px 16px;
    background: var(--color-bg-elevated);
    border-radius: 8px;
  }
  .status-text {
    font-size: 13px;
    color: var(--color-text-primary);
  }
  .status-text.connected {
    color: #10b981;
  }
  .status-text.disconnected {
    color: #ef4444;
  }
  .mt5-actions {
    display: flex;
    gap: 8px;
    grid-column: span 2;
  }
  .mt5-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-secondary);
    font-size: 12px;
    cursor: pointer;
  }
  .mt5-btn:hover {
    background: var(--color-bg-base);
  }
  .mt5-btn.primary {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }
  .mt5-config {
    grid-column: span 2;
  }
  .mt5-config label {
    display: block;
    font-size: 11px;
    color: var(--color-text-muted);
    margin-bottom: 4px;
  }
  .mt5-config input {
    width: 100%;
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 12px;
  }
  .mt5-info {
    grid-column: span 2;
    display: flex;
    gap: 24px;
  }
  .info-row {
    display: flex;
    gap: 8px;
    font-size: 11px;
  }
  .info-row span:first-child {
    color: var(--color-text-muted);
  }
  .info-row span:last-child {
    color: var(--color-text-primary);
  }

  /* Errors */
  .errors-panel {
    padding: 12px;
  }
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 24px;
    color: var(--color-text-muted);
  }
  .error-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 6px;
    margin-bottom: 4px;
  }
  .error-item :global(svg) {
    color: #ef4444;
  }
  .error-msg {
    flex: 1;
    font-size: 12px;
    color: var(--color-text-primary);
  }
  .error-loc {
    font-size: 11px;
    color: var(--color-text-muted);
  }

  /* Tool Calls */
  .tool-calls-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--term-bg, var(--color-bg-elevated));
  }
  .tool-calls-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    background: rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid var(--color-border-subtle);
  }
  .tool-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    border-radius: 4px;
  }
  .tool-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: var(--color-text-primary);
  }
  .tool-calls-count {
    font-size: 11px;
    color: var(--color-text-muted);
  }
  .tool-calls-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 4px 8px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 11px;
  }
  .tool-calls-loading,
  .tool-calls-error,
  .tool-calls-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 24px;
    color: var(--color-text-muted);
    font-size: 12px;
  }
  .tool-calls-error {
    color: #ef4444;
  }
  .tool-calls-header {
    display: grid;
    grid-template-columns: 70px 90px 140px 50px 70px;
    gap: 4px;
    padding: 4px 8px;
    font-size: 10px;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    border-bottom: 1px solid var(--color-border-subtle);
    margin-bottom: 4px;
  }
  .tool-call-line {
    display: grid;
    grid-template-columns: 70px 90px 140px 50px 70px;
    gap: 4px;
    padding: 3px 8px;
    color: var(--term-fg, var(--color-text-secondary));
    border-radius: 2px;
  }
  .tool-call-line:hover {
    background: rgba(255, 255, 255, 0.05);
  }
  .tool-call-line.error {
    color: #ef4444;
  }
  .tc-time {
    color: var(--color-text-muted);
  }
  .tc-agent {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .tc-tool {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .tc-status {
    font-weight: 600;
  }
  .tc-status.success {
    color: #10b981;
  }
  .tc-status.failed {
    color: #ef4444;
  }
  .tc-duration {
    color: var(--color-text-muted);
  }

  /* Logs */
  .logs-panel {
    padding: 8px 12px;
    overflow-y: auto;
    height: 100%;
  }
  .log-line {
    display: flex;
    gap: 12px;
    font-size: 12px;
    line-height: 1.8;
  }
  .log-line .time {
    color: var(--color-text-muted);
    font-family: "JetBrains Mono", monospace;
  }
  .log-line .msg {
    color: var(--color-text-secondary);
  }
  .log-line.success .msg {
    color: #10b981;
  }
  .log-line.error .msg {
    color: #ef4444;
  }
  .log-line.warning .msg {
    color: #f59e0b;
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
