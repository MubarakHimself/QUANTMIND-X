<!-- @migration-task Error while migrating Svelte code: Directive value must be a JavaScript expression enclosed in curly braces
https://svelte.dev/e/directive_invalid_value -->
<!-- @migration-task Error while migrating Svelte code: Directive value must be a JavaScript expression enclosed in curly braces
https://svelte.dev/e/directive_invalid_value -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import {
    Play, Pause, StepInto, StepOver, StepOut, Square, RefreshCw,
    Bug, ChevronRight, ChevronDown, X, Trash2, Plus
  } from 'lucide-svelte';

  export let isActive = false;
  export let currentFile = '';
  export let debugLine: number | null = null;

  const dispatch = createEventDispatcher();

  // Debug state
  let isRunning = false;
  let isPaused = false;

  // Call stack
  let callStack: Array<{ file: string; line: number; function: string }> = [
    { file: 'ICT_Scalper.mq5', line: 145, function: 'OnTick()' },
    { file: 'ICT_Scalper.mq5', line: 89, function: 'CheckEntrySignal()' },
    { file: 'ICT_Scalper.mq5', line: 234, function: 'ValidateOrderBlock()' }
  ];

  // Variables
  let variables: Record<string, any> = {
    'RiskPercent': 2.0,
    'MaxTrades': 3,
    'CurrentPrice': 1.0845,
    'StopLoss': 1.0825,
    'TakeProfit': 1.0885,
    'ActiveTrades': 1,
    'DailyPnL': 45.67
  };

  // Watch expressions
  let watchExpressions: Array<{ expression: string; value: string }> = [
    { expression: 'PositionGetDouble(POSITION_PROFIT)', value: '12.45' },
    { expression: 'SymbolInfoDouble(_Symbol, SYMBOL_BID)', value: '1.0845' }
  ];
  let newWatchExpression = '';

  // Breakpoints
  let breakpoints: Array<{ file: string; line: number; enabled: boolean; condition?: string }> = [
    { file: 'ICT_Scalper.mq5', line: 45, enabled: true },
    { file: 'ICT_Scalper.mq5', line: 89, enabled: true },
    { file: 'ICT_Scalper.mq5', line: 234, enabled: false }
  ];

  // Expanded sections
  let expandedSections = {
    callStack: true,
    variables: true,
    watch: true,
    breakpoints: true
  };

  function toggleSection(section: keyof typeof expandedSections) {
    expandedSections[section] = !expandedSections[section];
  }

  function startDebug() {
    isRunning = true;
    isPaused = false;
    dispatch('start');
  }

  function pauseDebug() {
    isPaused = true;
    dispatch('pause');
  }

  function continueDebug() {
    isPaused = false;
    dispatch('continue');
  }

  function stopDebug() {
    isRunning = false;
    isPaused = false;
    dispatch('stop');
  }

  function stepOver() {
    dispatch('stepOver');
  }

  function stepInto() {
    dispatch('stepInto');
  }

  function stepOut() {
    dispatch('stepOut');
  }

  function restartDebug() {
    dispatch('restart');
  }

  function addWatchExpression() {
    if (newWatchExpression.trim()) {
      watchExpressions = [...watchExpressions, {
        expression: newWatchExpression.trim(),
        value: 'evaluating...'
      }];
      newWatchExpression = '';
      dispatch('addWatch', { expression: watchExpressions[watchExpressions.length - 1].expression });
    }
  }

  function removeWatchExpression(index: number) {
    watchExpressions = watchExpressions.filter((_, i) => i !== index);
  }

  function toggleBreakpoint(index: number) {
    breakpoints = breakpoints.map((bp, i) =>
      i === index ? { ...bp, enabled: !bp.enabled } : bp
    );
    dispatch('toggleBreakpoint', { breakpoint: breakpoints[index] });
  }

  function removeBreakpoint(index: number) {
    const removed = breakpoints[index];
    breakpoints = breakpoints.filter((_, i) => i !== index);
    dispatch('removeBreakpoint', { breakpoint: removed });
  }

  function goToStackFrame(frame: typeof callStack[0]) {
    dispatch('goToFrame', { file: frame.file, line: frame.line });
  }
</script>

<div class="debug-panel" class:active={isActive}>
  <!-- Debug Controls -->
  <div class="debug-controls">
    <div class="control-group">
      {#if !isRunning}
        <button class="control-btn primary" on:click={startDebug} title="Start Debugging">
          <Play size={16} />
        </button>
      {:else if isPaused}
        <button class="control-btn primary" on:click={continueDebug} title="Continue">
          <Play size={16} />
        </button>
      {:else}
        <button class="control-btn" on:click={pauseDebug} title="Pause">
          <Pause size={16} />
        </button>
      {/if}

      <button class="control-btn" on:click={stopDebug} disabled={!isRunning} title="Stop">
        <Square size={16} />
      </button>
      <button class="control-btn" on:click={restartDebug} disabled={!isRunning} title="Restart">
        <RefreshCw size={16} />
      </button>
    </div>

    <div class="control-group">
      <button class="control-btn" on:click={stepOver} disabled={!isPaused} title="Step Over (F10)">
        <StepOver size={16} />
      </button>
      <button class="control-btn" on:click={stepInto} disabled={!isPaused} title="Step Into (F11)">
        <StepInto size={16} />
      </button>
      <button class="control-btn" on:click={stepOut} disabled={!isPaused} title="Step Out (Shift+F11)">
        <StepOut size={16} />
      </button>
    </div>

    <div class="debug-status">
      <Bug size={14} />
      <span>
        {#if isRunning && isPaused}
          Paused at line {debugLine}
        {:else if isRunning}
          Running...
        {:else}
          Ready
        {/if}
      </span>
    </div>
  </div>

  <!-- Panel Sections -->
  <div class="debug-sections">
    <!-- Call Stack -->
    <div class="section">
      <div class="section-header" on:click={() => toggleSection('callStack')}>
        {#if expandedSections.callStack}
          <ChevronDown size={14} />
        {:else}
          <ChevronRight size={14} />
        {/if}
        <span>Call Stack</span>
        <span class="count">{callStack.length}</span>
      </div>
      {#if expandedSections.callStack}
        <div class="section-content">
          {#each callStack as frame, index}
            <div class="stack-frame" on:click={() => goToStackFrame(frame)}>
              <span class="frame-index">{index}</span>
              <span class="frame-function">{frame.function}</span>
              <span class="frame-location">{frame.file}:{frame.line}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Variables -->
    <div class="section">
      <div class="section-header" on:click={() => toggleSection('variables')}>
        {#if expandedSections.variables}
          <ChevronDown size={14} />
        {:else}
          <ChevronRight size={14} />
        {/if}
        <span>Variables</span>
        <span class="count">{Object.keys(variables).length}</span>
      </div>
      {#if expandedSections.variables}
        <div class="section-content">
          {#each Object.entries(variables) as [name, value]}
            <div class="variable-row">
              <span class="var-name">{name}</span>
              <span class="var-value" class:string={typeof value === 'string'},
                      class:number={typeof value === 'number'},
                      class:boolean={typeof value === 'boolean'}>
                {typeof value === 'number' ? value.toFixed(value % 1 === 0 ? 0 : 5) : value}
              </span>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Watch Expressions -->
    <div class="section">
      <div class="section-header" on:click={() => toggleSection('watch')}>
        {#if expandedSections.watch}
          <ChevronDown size={14} />
        {:else}
          <ChevronRight size={14} />
        {/if}
        <span>Watch</span>
        <span class="count">{watchExpressions.length}</span>
      </div>
      {#if expandedSections.watch}
        <div class="section-content">
          {#each watchExpressions as watch, index}
            <div class="watch-row">
              <span class="watch-expr">{watch.expression}</span>
              <span class="watch-value">{watch.value}</span>
              <button class="remove-btn" on:click={() => removeWatchExpression(index)}>
                <X size={12} />
              </button>
            </div>
          {/each}
          <div class="add-watch">
            <input
              type="text"
              placeholder="Add watch expression..."
              bind:value={newWatchExpression}
              on:keydown={(e) => e.key === 'Enter' && addWatchExpression()}
            />
            <button class="add-btn" on:click={addWatchExpression}>
              <Plus size={14} />
            </button>
          </div>
        </div>
      {/if}
    </div>

    <!-- Breakpoints -->
    <div class="section">
      <div class="section-header" on:click={() => toggleSection('breakpoints')}>
        {#if expandedSections.breakpoints}
          <ChevronDown size={14} />
        {:else}
          <ChevronRight size={14} />
        {/if}
        <span>Breakpoints</span>
        <span class="count">{breakpoints.filter(b => b.enabled).length}</span>
      </div>
      {#if expandedSections.breakpoints}
        <div class="section-content">
          {#each breakpoints as bp, index}
            <div class="breakpoint-row" class:disabled={!bp.enabled}>
              <input
                type="checkbox"
                checked={bp.enabled}
                on:change={() => toggleBreakpoint(index)}
              />
              <span class="bp-location">{bp.file}:{bp.line}</span>
              {#if bp.condition}
                <span class="bp-condition">if {bp.condition}</span>
              {/if}
              <button class="remove-btn" on:click={() => removeBreakpoint(index)}>
                <Trash2 size={12} />
              </button>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .debug-panel {
    display: flex;
    flex-direction: column;
    width: 300px;
    background: var(--color-bg-surface);
    border-left: 1px solid var(--color-border-subtle);
    font-size: 12px;
    overflow: hidden;
  }

  .debug-panel.active {
    background: var(--color-bg-base);
  }

  .debug-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .control-group {
    display: flex;
    gap: 4px;
  }

  .control-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: 1px solid var(--color-border-subtle);
    border-radius: 4px;
    color: var(--color-text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .control-btn:hover:not(:disabled) {
    background: var(--color-bg-surface);
    color: var(--color-text-primary);
  }

  .control-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .control-btn.primary {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: #000;
  }

  .control-btn.primary:hover:not(:disabled) {
    background: var(--color-accent-amber);
  }

  .debug-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: var(--color-bg-base);
    border-radius: 4px;
    color: var(--color-text-muted);
    font-size: 11px;
    flex: 1;
    min-width: 120px;
  }

  .debug-sections {
    flex: 1;
    overflow-y: auto;
  }

  .section {
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    cursor: pointer;
    user-select: none;
  }

  .section-header:hover {
    background: var(--color-bg-surface);
  }

  .section-header span:first-of-type {
    font-weight: 500;
    color: var(--color-text-primary);
  }

  .count {
    margin-left: auto;
    background: var(--color-bg-base);
    padding: 2px 6px;
    border-radius: 10px;
    font-size: 10px;
    color: var(--color-text-muted);
  }

  .section-content {
    padding: 4px 0;
    max-height: 200px;
    overflow-y: auto;
  }

  .stack-frame {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px;
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .stack-frame:hover {
    background: var(--color-bg-elevated);
  }

  .frame-index {
    width: 20px;
    color: var(--color-text-muted);
    font-size: 10px;
  }

  .frame-function {
    color: var(--color-accent-cyan);
    font-weight: 500;
  }

  .frame-location {
    margin-left: auto;
    color: var(--color-text-muted);
    font-size: 10px;
  }

  .variable-row, .watch-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 12px;
  }

  .variable-row:hover, .watch-row:hover {
    background: var(--color-bg-elevated);
  }

  .var-name {
    color: var(--color-text-primary);
  }

  .var-value, .watch-value {
    color: var(--color-text-secondary);
  }

  .var-value.string {
    color: #ce9178;
  }

  .var-value.number {
    color: #b5cea8;
  }

  .var-value.boolean {
    color: #569cd6;
  }

  .watch-row {
    position: relative;
    padding-right: 28px;
  }

  .watch-expr {
    color: var(--color-text-secondary);
    font-style: italic;
  }

  .remove-btn {
    position: absolute;
    right: 8px;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 2px;
    opacity: 0;
    transition: opacity 0.15s ease;
  }

  .watch-row:hover .remove-btn,
  .breakpoint-row:hover .remove-btn {
    opacity: 1;
  }

  .add-watch {
    display: flex;
    gap: 4px;
    padding: 4px 12px;
  }

  .add-watch input {
    flex: 1;
    padding: 4px 8px;
    background: var(--color-bg-base);
    border: 1px solid var(--color-border-subtle);
    border-radius: 4px;
    color: var(--color-text-primary);
    font-size: 11px;
  }

  .add-watch input:focus {
    outline: none;
    border-color: var(--color-accent-cyan);
  }

  .add-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--color-accent-cyan);
    border: none;
    border-radius: 4px;
    color: #000;
    cursor: pointer;
  }

  .breakpoint-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px;
    padding-right: 28px;
    position: relative;
  }

  .breakpoint-row:hover {
    background: var(--color-bg-elevated);
  }

  .breakpoint-row.disabled {
    opacity: 0.5;
  }

  .breakpoint-row input[type="checkbox"] {
    accent-color: var(--color-accent-cyan);
  }

  .bp-location {
    color: var(--color-text-primary);
  }

  .bp-condition {
    color: var(--color-text-muted);
    font-style: italic;
    font-size: 10px;
  }
</style>
