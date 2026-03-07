<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { X, Play, Loader } from "lucide-svelte";

  export let isOpen = false;
  export let isRunning = false;
  export let config = {
    strategy: "ict-v2",
    period: "1M",
    monteCarlo: true,
    variant: "spiced",
    symbol: "EURUSD",
    timeframe: "H1"
  };

  const dispatch = createEventDispatcher();

  function handleClose() {
    dispatch("close");
  }

  function handleRun() {
    dispatch("run", { config });
  }
</script>

{#if isOpen}
  <div
    class="modal-overlay"
    on:click|self={handleClose}
    role="button"
    tabindex="0"
    on:keydown={(e) => e.key === "Enter" && handleClose()}
  >
    <div class="modal">
      <div class="modal-header">
        <h2><Play size={20} /> Run Backtest</h2>
        <button on:click={handleClose}><X size={20} /></button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label for="bt-strategy">Strategy</label>
          <select id="bt-strategy" bind:value={config.strategy}>
            <option value="ict-v2">ICT V2</option>
            <option value="smc-basic">SMC Basic</option>
            <option value="ma-crossover">MA Crossover</option>
          </select>
        </div>
        <div class="form-group">
          <label for="bt-symbol">Symbol</label>
          <select id="bt-symbol" bind:value={config.symbol}>
            <option value="EURUSD">EURUSD</option>
            <option value="GBPUSD">GBPUSD</option>
            <option value="USDJPY">USDJPY</option>
            <option value="XAUUSD">XAUUSD</option>
            <option value="BTCUSD">BTCUSD</option>
          </select>
        </div>
        <div class="form-group">
          <label for="bt-timeframe">Timeframe</label>
          <select id="bt-timeframe" bind:value={config.timeframe}>
            <option value="M1">M1</option>
            <option value="M5">M5</option>
            <option value="M15">M15</option>
            <option value="M30">M30</option>
            <option value="H1">H1</option>
            <option value="H4">H4</option>
            <option value="D1">D1</option>
          </select>
        </div>
        <div class="form-group">
          <label for="bt-mode">Backtest Mode</label>
          <select id="bt-mode" bind:value={config.variant}>
            <option value="vanilla">Vanilla (Basic)</option>
            <option value="spiced">Spiced (Regime Filter)</option>
            <option value="vanilla_full">Vanilla + Walk-Forward</option>
            <option value="spiced_full">Spiced + Walk-Forward</option>
          </select>
        </div>
        <div class="form-group">
          <label for="bt-period">Time Period</label>
          <select id="bt-period" bind:value={config.period}>
            <option value="1M">Last Month</option>
            <option value="3M">Last 3 Months</option>
            <option value="6M">Last 6 Months</option>
            <option value="1Y">Last Year</option>
            <option value="all">Max Available</option>
          </select>
        </div>
        <div class="form-group checkbox">
          <input
            type="checkbox"
            id="mc-sim"
            bind:checked={config.monteCarlo}
          />
          <label for="mc-sim"
            >Run Monte Carlo Simulation (1000 runs)</label
          >
        </div>

        {#if isRunning}
          <div class="progress-section">
            <div class="spinner-row">
              <Loader size={16} class="spinning" />
              <span>Running simulation...</span>
            </div>
          </div>
        {/if}
      </div>
      <div class="modal-footer">
        <button
          class="btn secondary"
          on:click={handleClose}
          disabled={isRunning}>Cancel</button
        >
        <button
          class="btn primary"
          on:click={handleRun}
          disabled={isRunning}
        >
          {#if isRunning}Running...{:else}Start Backtest{/if}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    width: 480px;
    max-width: 90%;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 16px;
    height: 48px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h2 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .modal-header button {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
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

  .form-group select {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .form-group.checkbox {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .form-group.checkbox input {
    width: 18px;
    height: 18px;
    cursor: pointer;
  }

  .form-group.checkbox label {
    margin: 0;
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-subtle);
  }

  .btn {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .btn.primary {
    background: var(--accent-primary);
    border: none;
    color: var(--bg-primary);
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .progress-section {
    margin-top: 16px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .spinner-row {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 13px;
  }

  :global(.spinning) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
