<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { X, CheckCircle, AlertCircle } from 'lucide-svelte';

  export let isOpen = false;
  export let isLoading = false;
  export let result: { success: boolean; bot_id?: string; error?: string } | null = null;

  export let form = {
    target_account: 'account_b_sniper',
    strategy_name: '',
    strategy_type: 'STRUCTURAL',
  };

  const dispatch = createEventDispatcher();

  const accountOptions = [
    { value: 'account_b_sniper', label: 'Sniper (Structural/ICT)' },
    { value: 'account_c_prop', label: 'Prop Firm Safe' },
  ];

  const strategyTypes = ['SCALPER', 'STRUCTURAL', 'SWING', 'HFT'];
</script>

{#if isOpen}
  <div class="modal-overlay" on:click={() => dispatch('close')} on:keydown={(e) => e.key === 'Escape' && dispatch('close')} role="dialog" aria-modal="true">
    <div class="modal-content" on:click|stopPropagation role="presentation">
      <div class="modal-header">
        <h3>Promote to Production</h3>
        <button class="close-btn" on:click={() => dispatch('close')}>
          <X size={18} />
        </button>
      </div>
      <div class="modal-body">
        {#if result?.success}
          <div class="success-message">
            <CheckCircle size={48} />
            <h4>Promotion Successful!</h4>
            <p>Your agent has been promoted to production trading.</p>
            {#if result.bot_id}
              <p class="bot-id">Bot ID: {result.bot_id}</p>
            {/if}
          </div>
        {:else if result?.error}
          <div class="error-message">
            <AlertCircle size={48} />
            <h4>Promotion Failed</h4>
            <p>{result.error}</p>
          </div>
        {:else}
          <!-- Validation Summary -->
          <div class="validation-summary">
            <h4>Validation Requirements</h4>
            <div class="summary-grid">
              <div class="summary-item">
                <span class="check">✓</span>
                <span>Minimum 30 days validated</span>
              </div>
              <div class="summary-item">
                <span class="check">✓</span>
                <span>Win rate ≥ 50%</span>
              </div>
              <div class="summary-item">
                <span class="check">✓</span>
                <span>Sharpe ratio ≥ 1.0</span>
              </div>
              <div class="summary-item">
                <span class="check">✓</span>
                <span>All criteria met</span>
              </div>
            </div>
          </div>

          <!-- Promotion Form -->
          <div class="form-group">
            <label for="target_account">Target Account *</label>
            <select id="target_account" bind:value={form.target_account}>
              {#each accountOptions as opt}
                <option value={opt.value}>{opt.label}</option>
              {/each}
            </select>
          </div>
          <div class="form-group">
            <label for="strategy_name">Strategy Name</label>
            <input
              id="strategy_name"
              type="text"
              bind:value={form.strategy_name}
              placeholder="Enter strategy name for production"
            />
          </div>
          <div class="form-group">
            <label for="strategy_type">Strategy Type</label>
            <select id="strategy_type" bind:value={form.strategy_type}>
              {#each strategyTypes as st}
                <option value={st}>{st}</option>
              {/each}
            </select>
          </div>
        {/if}
      </div>
      {#if !result?.success}
        <div class="modal-footer">
          <button class="btn secondary" on:click={() => dispatch('close')}>Cancel</button>
          <button class="btn primary" on:click={() => dispatch('promote', form)} disabled={isLoading}>
            {isLoading ? 'Promoting...' : 'Promote to Production'}
          </button>
        </div>
      {/if}
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
    max-width: 500px;
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

  .success-message,
  .error-message {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 20px;
  }

  .success-message {
    color: #10b981;
  }

  .error-message {
    color: #ef4444;
  }

  .success-message h4,
  .error-message h4 {
    margin: 12px 0 8px;
    font-size: 18px;
    color: #e2e8f0;
  }

  .success-message p,
  .error-message p {
    color: #94a3b8;
    margin: 0;
  }

  .bot-id {
    font-family: monospace;
    background: #0f172a;
    padding: 8px 16px;
    border-radius: 6px;
    margin-top: 12px !important;
    color: #e2e8f0 !important;
  }

  .validation-summary {
    background: #0f172a;
    border-radius: 8px;
    padding: 16px;
  }

  .validation-summary h4 {
    margin: 0 0 12px;
    font-size: 14px;
    color: #e2e8f0;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .summary-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #94a3b8;
  }

  .check {
    color: #10b981;
    font-weight: bold;
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
  .form-group select {
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid #334155;
    background: #0f172a;
    color: #e2e8f0;
    font-size: 14px;
  }

  .form-group input:focus,
  .form-group select:focus {
    outline: none;
    border-color: #3b82f6;
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
    background: #8b5cf6;
    color: white;
  }

  .btn.primary:hover {
    background: #7c3aed;
  }

  .btn.primary:disabled {
    background: #64748b;
    cursor: not-allowed;
  }
</style>
