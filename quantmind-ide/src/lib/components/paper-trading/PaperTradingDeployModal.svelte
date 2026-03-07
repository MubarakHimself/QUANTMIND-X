<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { X } from 'lucide-svelte';

  export let isOpen = false;
  export let isLoading = false;

  export let form = {
    strategy_name: '',
    strategy_code: '',
    symbol: 'EURUSD',
    timeframe: 'H1',
    mt5_account: '',
    mt5_password: '',
    mt5_server: 'MetaQuotes-Demo',
    magic_number: Math.floor(Math.random() * 100000000),
  };

  const dispatch = createEventDispatcher();

  const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'];
  const timeframes = ['M1', 'M5', 'H1', 'H4', 'D1'];
  const accountOptions = [
    { value: 'account_a_machine_gun', label: 'Machine Gun (HFT/Scalpers)' },
    { value: 'account_b_sniper', label: 'Sniper (Structural/ICT)' },
    { value: 'account_c_prop', label: 'Prop Firm Safe' },
  ];
</script>

{#if isOpen}
  <div class="modal-overlay" on:click={() => dispatch('close')} on:keydown={(e) => e.key === 'Escape' && dispatch('close')} role="dialog" aria-modal="true">
    <div class="modal-content" on:click|stopPropagation role="presentation">
      <div class="modal-header">
        <h3>Deploy New Agent</h3>
        <button class="close-btn" on:click={() => dispatch('close')}>
          <X size={18} />
        </button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label for="strategy_name">Strategy Name *</label>
          <input
            id="strategy_name"
            type="text"
            bind:value={form.strategy_name}
            placeholder="e.g., EURUSD Scalper"
          />
        </div>
        <div class="form-row">
          <div class="form-group">
            <label for="symbol">Symbol</label>
            <select id="symbol" bind:value={form.symbol}>
              {#each symbols as sym}
                <option value={sym}>{sym}</option>
              {/each}
            </select>
          </div>
          <div class="form-group">
            <label for="timeframe">Timeframe</label>
            <select id="timeframe" bind:value={form.timeframe}>
              {#each timeframes as tf}
                <option value={tf}>{tf}</option>
              {/each}
            </select>
          </div>
        </div>
        <div class="form-group">
          <label for="strategy_code">Strategy Code *</label>
          <textarea
            id="strategy_code"
            bind:value={form.strategy_code}
            placeholder="Paste your MQL5 strategy code here..."
            rows="8"
          ></textarea>
        </div>
        <div class="form-group">
          <label for="mt5_account">MT5 Account *</label>
          <select id="mt5_account" bind:value={form.mt5_account}>
            <option value="">Select account...</option>
            {#each accountOptions as opt}
              <option value={opt.value}>{opt.label}</option>
            {/each}
          </select>
        </div>
        <div class="form-group">
          <label for="mt5_password">MT5 Password *</label>
          <input
            id="mt5_password"
            type="password"
            bind:value={form.mt5_password}
            placeholder="Enter MT5 password"
          />
        </div>
        <div class="form-group">
          <label for="mt5_server">MT5 Server</label>
          <input
            id="mt5_server"
            type="text"
            bind:value={form.mt5_server}
            placeholder="MetaQuotes-Demo"
          />
        </div>
        <div class="form-group">
          <label for="magic_number">Magic Number</label>
          <input
            id="magic_number"
            type="number"
            bind:value={form.magic_number}
          />
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" on:click={() => dispatch('close')}>Cancel</button>
        <button class="btn primary" on:click={() => dispatch('deploy', form)} disabled={isLoading}>
          {isLoading ? 'Deploying...' : 'Deploy Agent'}
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
    z-index: 1000;
  }

  .modal-content {
    background: #1e293b;
    border-radius: 12px;
    width: 90%;
    max-width: 600px;
    max-height: 90vh;
    overflow-y: auto;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid #334155;
  }

  .modal-header h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .close-btn {
    background: none;
    border: none;
    color: #94a3b8;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
  }

  .close-btn:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .modal-body {
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .form-group label {
    font-size: 13px;
    font-weight: 500;
    color: #e2e8f0;
  }

  .form-group input,
  .form-group select,
  .form-group textarea {
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid #334155;
    background: #0f172a;
    color: #e2e8f0;
    font-size: 14px;
    font-family: inherit;
  }

  .form-group input:focus,
  .form-group select:focus,
  .form-group textarea:focus {
    outline: none;
    border-color: #3b82f6;
  }

  .form-group textarea {
    font-family: monospace;
    resize: vertical;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 20px;
    border-top: 1px solid #334155;
  }

  .btn {
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn.secondary {
    background: #334155;
    color: #e2e8f0;
  }

  .btn.secondary:hover {
    background: #475569;
  }

  .btn.primary {
    background: #3b82f6;
    color: white;
  }

  .btn.primary:hover {
    background: #2563eb;
  }

  .btn.primary:disabled {
    background: #64748b;
    cursor: not-allowed;
  }
</style>
