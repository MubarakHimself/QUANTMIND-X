<script lang="ts">
  import { Terminal, Activity, AlertCircle, FileText, ChevronUp, ChevronDown, X, Play, Settings, Wifi, WifiOff, ExternalLink } from 'lucide-svelte';
  
  let activeTab = 'terminal';
  let isExpanded = true;
  let terminalInput = '';
  let terminalHistory: Array<{type: 'input' | 'output' | 'error', content: string}> = [
    { type: 'output', content: 'QuantMind Terminal v1.0.0' },
    { type: 'output', content: 'Type "help" for available commands' },
    { type: 'output', content: '' }
  ];
  
  let mt5Status = 'disconnected';
  let mt5Path = '/opt/MetaTrader5/terminal64';
  
  let logs: Array<{type: string, message: string, time: string}> = [
    { type: 'info', message: 'QuantMind IDE started', time: new Date().toLocaleTimeString() },
    { type: 'success', message: 'Backend API connected', time: new Date().toLocaleTimeString() },
    { type: 'info', message: 'Kelly router initialized (k=0.85)', time: new Date().toLocaleTimeString() },
  ];
  
  let errors: Array<{message: string, file?: string, line?: number}> = [];
  
  const tabs = [
    { id: 'terminal', label: 'Terminal', icon: Terminal },
    { id: 'mt5', label: 'MT5 Sync', icon: Activity },
    { id: 'errors', label: 'Errors', icon: AlertCircle, count: errors.length },
    { id: 'output', label: 'Output', icon: FileText }
  ];
  
  const terminalCommands: Record<string, () => string> = {
    help: () => `Available commands:
  status    - Show system status
  bots      - List active bots
  kill      - Trigger kill switch (requires confirmation)
  regime    - Show current market regime
  kelly     - Show Kelly factor
  nprd      - List NPRD processing queue
  clear     - Clear terminal`,
    status: () => `System Status: OK
  Active Bots: 3
  Kelly Factor: 0.85
  Market Regime: Trending
  MT5 Status: ${mt5Status}`,
    bots: () => `Active Bots:
  1. ICT_Scalper @EURUSD - primal (+$450.25)
  2. ICT_Scalper @GBPUSD - primal (+$320.10)
  3. SMC_Rev @USDJPY - ready (+$480.15)`,
    regime: () => 'Current Regime: TRENDING (High Volatility)',
    kelly: () => 'Kelly Factor: 0.85 (Normal allocation)',
    nprd: () => 'NPRD Queue: Empty',
    kill: () => '⚠️ Kill switch not triggered from terminal. Use the UI button.',
    clear: () => { terminalHistory = []; return ''; }
  };
  
  function handleTerminalInput(e: KeyboardEvent) {
    if (e.key === 'Enter' && terminalInput.trim()) {
      const cmd = terminalInput.trim().toLowerCase();
      terminalHistory = [...terminalHistory, { type: 'input', content: `$ ${terminalInput}` }];
      
      if (terminalCommands[cmd]) {
        const output = terminalCommands[cmd]();
        if (output) {
          terminalHistory = [...terminalHistory, { type: 'output', content: output }];
        }
      } else {
        terminalHistory = [...terminalHistory, { type: 'error', content: `Command not found: ${cmd}` }];
      }
      
      terminalInput = '';
      
      // Scroll to bottom
      setTimeout(() => {
        const container = document.querySelector('.terminal-scroll');
        if (container) container.scrollTop = container.scrollHeight;
      }, 10);
    }
  }
  
  async function connectMT5() {
    mt5Status = 'connecting';
    logs = [...logs, { type: 'info', message: 'Attempting to connect to MT5...', time: new Date().toLocaleTimeString() }];
    
    // Simulate connection attempt
    setTimeout(() => {
      // In real implementation, this would use Tauri commands to open MT5
      mt5Status = 'connected';
      logs = [...logs, { type: 'success', message: 'MT5 connection established', time: new Date().toLocaleTimeString() }];
    }, 2000);
  }
  
  function openMT5Terminal() {
    // Would use Tauri shell command: await shell.open(mt5Path)
    logs = [...logs, { type: 'info', message: 'Opening MT5 Terminal...', time: new Date().toLocaleTimeString() }];
    window.open('file://' + mt5Path, '_blank');
  }
  
  function disconnectMT5() {
    mt5Status = 'disconnected';
    logs = [...logs, { type: 'info', message: 'MT5 disconnected', time: new Date().toLocaleTimeString() }];
  }
</script>

<div class="bottom-panel" class:expanded={isExpanded}>
  <div class="panel-header">
    <div class="tabs">
      {#each tabs as tab}
        <button class="tab" class:active={activeTab === tab.id} on:click={() => activeTab = tab.id}>
          <svelte:component this={tab.icon} size={14} />
          <span>{tab.label}</span>
          {#if tab.count}<span class="badge">{tab.count}</span>{/if}
        </button>
      {/each}
    </div>
    <div class="actions">
      <button on:click={() => isExpanded = !isExpanded}>
        {#if isExpanded}<ChevronDown size={14} />{:else}<ChevronUp size={14} />{/if}
      </button>
      <button on:click={() => isExpanded = false}><X size={14} /></button>
    </div>
  </div>
  
  {#if isExpanded}
    <div class="panel-content">
      {#if activeTab === 'terminal'}
        <div class="terminal">
          <div class="terminal-scroll">
            {#each terminalHistory as line}
              <div class="term-line {line.type}">{line.content}</div>
            {/each}
          </div>
          <div class="terminal-input">
            <span class="prompt">$</span>
            <input 
              type="text" 
              bind:value={terminalInput} 
              on:keydown={handleTerminalInput}
              placeholder="Enter command..."
            />
          </div>
        </div>
      
      {:else if activeTab === 'mt5'}
        <div class="mt5-sync">
          <div class="mt5-status">
            {#if mt5Status === 'connected'}
              <Wifi size={20} color="#10b981" />
              <span class="status-text connected">Connected to MT5</span>
            {:else if mt5Status === 'connecting'}
              <Activity size={20} class="spin" />
              <span class="status-text">Connecting...</span>
            {:else}
              <WifiOff size={20} color="#ef4444" />
              <span class="status-text disconnected">Disconnected</span>
            {/if}
          </div>
          
          <div class="mt5-actions">
            {#if mt5Status === 'connected'}
              <button class="mt5-btn" on:click={disconnectMT5}>Disconnect</button>
            {:else if mt5Status !== 'connecting'}
              <button class="mt5-btn primary" on:click={connectMT5}>Connect to MT5</button>
            {/if}
            <button class="mt5-btn" on:click={openMT5Terminal}>
              <ExternalLink size={12} /> Open MT5 Terminal
            </button>
          </div>
          
          <div class="mt5-config">
            <label>MT5 Path</label>
            <input type="text" bind:value={mt5Path} placeholder="/path/to/terminal64" />
          </div>
          
          <div class="mt5-info">
            <div class="info-row"><span>Last Sync:</span><span>{mt5Status === 'connected' ? 'Just now' : 'Never'}</span></div>
            <div class="info-row"><span>Accounts:</span><span>{mt5Status === 'connected' ? '2 connected' : '0'}</span></div>
            <div class="info-row"><span>Open Trades:</span><span>{mt5Status === 'connected' ? '5' : '0'}</span></div>
          </div>
        </div>
      
      {:else if activeTab === 'errors'}
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
      
      {:else if activeTab === 'output'}
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
  .bottom-panel { grid-area: bottom; background: var(--bg-secondary); border-top: 1px solid var(--border-subtle); min-height: 32px; }
  .bottom-panel.expanded { height: 220px; }
  
  .panel-header { display: flex; justify-content: space-between; padding: 0 8px; height: 32px; border-bottom: 1px solid var(--border-subtle); }
  .tabs { display: flex; gap: 2px; }
  .tab { display: flex; align-items: center; gap: 6px; padding: 0 12px; height: 32px; background: transparent; border: none; color: var(--text-muted); font-size: 11px; cursor: pointer; }
  .tab:hover { color: var(--text-primary); }
  .tab.active { color: var(--accent-primary); border-bottom: 2px solid var(--accent-primary); }
  .badge { background: #ef4444; color: white; font-size: 10px; padding: 0 5px; border-radius: 8px; }
  .actions { display: flex; gap: 4px; align-items: center; }
  .actions button { background: none; border: none; color: var(--text-muted); cursor: pointer; padding: 4px; }
  
  .panel-content { height: calc(100% - 32px); overflow: hidden; }
  
  /* Terminal */
  .terminal { display: flex; flex-direction: column; height: 100%; }
  .terminal-scroll { flex: 1; overflow-y: auto; padding: 8px 12px; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
  .term-line { line-height: 1.6; white-space: pre-wrap; }
  .term-line.input { color: var(--accent-primary); }
  .term-line.output { color: var(--text-secondary); }
  .term-line.error { color: #ef4444; }
  .terminal-input { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-top: 1px solid var(--border-subtle); background: var(--bg-tertiary); }
  .prompt { color: var(--accent-primary); font-family: 'JetBrains Mono', monospace; }
  .terminal-input input { flex: 1; background: transparent; border: none; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-size: 12px; outline: none; }
  
  /* MT5 Sync */
  .mt5-sync { padding: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .mt5-status { display: flex; align-items: center; gap: 12px; grid-column: span 2; padding: 12px 16px; background: var(--bg-tertiary); border-radius: 8px; }
  .status-text { font-size: 13px; color: var(--text-primary); }
  .status-text.connected { color: #10b981; }
  .status-text.disconnected { color: #ef4444; }
  .mt5-actions { display: flex; gap: 8px; grid-column: span 2; }
  .mt5-btn { display: flex; align-items: center; gap: 6px; padding: 8px 16px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-secondary); font-size: 12px; cursor: pointer; }
  .mt5-btn:hover { background: var(--bg-primary); }
  .mt5-btn.primary { background: var(--accent-primary); border-color: var(--accent-primary); color: var(--bg-primary); }
  .mt5-config { grid-column: span 2; }
  .mt5-config label { display: block; font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
  .mt5-config input { width: 100%; padding: 8px 12px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-primary); font-size: 12px; }
  .mt5-info { grid-column: span 2; display: flex; gap: 24px; }
  .info-row { display: flex; gap: 8px; font-size: 11px; }
  .info-row span:first-child { color: var(--text-muted); }
  .info-row span:last-child { color: var(--text-primary); }
  
  /* Errors */
  .errors-panel { padding: 12px; }
  .empty-state { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 24px; color: var(--text-muted); }
  .error-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: rgba(239,68,68,0.1); border-radius: 6px; margin-bottom: 4px; }
  .error-item :global(svg) { color: #ef4444; }
  .error-msg { flex: 1; font-size: 12px; color: var(--text-primary); }
  .error-loc { font-size: 11px; color: var(--text-muted); }
  
  /* Logs */
  .logs-panel { padding: 8px 12px; overflow-y: auto; height: 100%; }
  .log-line { display: flex; gap: 12px; font-size: 12px; line-height: 1.8; }
  .log-line .time { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
  .log-line .msg { color: var(--text-secondary); }
  .log-line.success .msg { color: #10b981; }
  .log-line.error .msg { color: #ef4444; }
  .log-line.warning .msg { color: #f59e0b; }
  
  :global(.spin) { animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
