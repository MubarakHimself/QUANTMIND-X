<script lang="ts">
  /**
   * PropFirmConfigPanel - Prop Firm Configuration Panel
   *
   * Displays registered prop firm entries with editable fields.
   * AC #2: Shows all registered prop firm entries with editable fields,
   * saving calls PUT /api/risk/prop-firms/{id}
   */

  import { onMount, onDestroy } from 'svelte';
  import { Plus, Save, Trash2, Edit2, X, Building2 } from 'lucide-svelte';
  import {
    propFirmStore,
    propFirms,
    propFirmLoading,
    propFirmError,
    type PropFirm
  } from '$lib/stores';

  let firms = $state<PropFirm[]>([]);
  let loading = $state(false);
  let error: string | null = $state(null);
  let editingId: number | null = $state(null);
  let showCreateForm = $state(false);

  // Form state
  let formData = $state({
    firm_name: '',
    account_id: '',
    daily_loss_limit_pct: 5,
    target_profit_pct: 10,
    risk_mode: 'growth'
  });

  // Subscribe to store
  const unsubFirms = propFirms.subscribe(v => firms = v);
  const unsubLoading = propFirmLoading.subscribe(v => loading = v);
  const unsubError = propFirmError.subscribe(v => error = v);

  onMount(() => {
    propFirmStore.fetch();
  });

  onDestroy(() => {
    unsubFirms();
    unsubLoading();
    unsubError();
  });

  function startEdit(firm: PropFirm) {
    editingId = firm.id;
  }

  function cancelEdit() {
    editingId = null;
  }

  async function saveEdit(firm: PropFirm) {
    try {
      await propFirmStore.update(firm.id, {
        firm_name: firm.firm_name,
        daily_loss_limit_pct: firm.daily_loss_limit_pct,
        target_profit_pct: firm.target_profit_pct,
        risk_mode: firm.risk_mode
      });
      editingId = null;
    } catch (e) {
      console.error('Failed to save:', e);
    }
  }

  async function deleteFirm(id: number) {
    if (confirm('Are you sure you want to delete this prop firm?')) {
      try {
        await propFirmStore.delete(id);
      } catch (e) {
        console.error('Failed to delete:', e);
      }
    }
  }

  function showForm() {
    showCreateForm = true;
    formData = {
      firm_name: '',
      account_id: '',
      daily_loss_limit_pct: 5,
      target_profit_pct: 10,
      risk_mode: 'growth'
    };
  }

  function hideForm() {
    showCreateForm = false;
  }

  async function createFirm() {
    try {
      await propFirmStore.create(formData);
      showCreateForm = false;
    } catch (e) {
      console.error('Failed to create:', e);
    }
  }

  function getRiskModeColor(mode: string): string {
    switch (mode) {
      case 'growth': return '#00d4ff';
      case 'scaling': return '#7c3aed';
      case 'guardian': return '#10b981';
      default: return '#888';
    }
  }
</script>

<div class="prop-firm-panel">
  <div class="panel-header">
    <h3 class="panel-title">
      <Building2 size={16} />
      Prop Firm Config
    </h3>
    <button class="add-btn" onclick={showForm} disabled={loading}>
      <Plus size={14} />
      Add Firm
    </button>
  </div>

  {#if showCreateForm}
    <div class="create-form">
      <div class="form-header">
        <h4>New Prop Firm</h4>
        <button class="icon-btn" onclick={hideForm}>
          <X size={14} />
        </button>
      </div>
      <div class="form-fields">
        <div class="field">
          <label>Firm Name</label>
          <input type="text" bind:value={formData.firm_name} placeholder="e.g., FundedNext" />
        </div>
        <div class="field">
          <label>Account ID</label>
          <input type="text" bind:value={formData.account_id} placeholder="MT5 Account Number" />
        </div>
        <div class="field-row">
          <div class="field">
            <label>Daily Loss %</label>
            <input type="number" bind:value={formData.daily_loss_limit_pct} step="0.1" min="0" max="100" />
          </div>
          <div class="field">
            <label>Target Profit %</label>
            <input type="number" bind:value={formData.target_profit_pct} step="0.1" min="0" />
          </div>
        </div>
        <div class="field">
          <label>Risk Mode</label>
          <select bind:value={formData.risk_mode}>
            <option value="growth">Growth</option>
            <option value="scaling">Scaling</option>
            <option value="guardian">Guardian</option>
          </select>
        </div>
        <button class="save-btn" onclick={createFirm} disabled={loading}>
          <Save size={14} />
          {loading ? 'Creating...' : 'Create Firm'}
        </button>
      </div>
    </div>
  {/if}

  <div class="panel-content">
    {#if loading && firms.length === 0}
      <div class="loading-state">
        <span>Loading...</span>
      </div>
    {:else if error}
      <div class="error-state">
        <span>{error}</span>
      </div>
    {:else if firms.length === 0}
      <div class="empty-state">
        <span>No prop firms configured</span>
      </div>
    {:else}
      <div class="firm-list">
        {#each firms as firm (firm.id)}
          <div class="firm-item">
            {#if editingId === firm.id}
              <div class="firm-edit">
                <input type="text" bind:value={firm.firm_name} class="edit-name" />
                <div class="edit-fields">
                  <label>
                    Daily Loss %
                    <input type="number" bind:value={firm.daily_loss_limit_pct} step="0.1" min="0" max="100" />
                  </label>
                  <label>
                    Target %
                    <input type="number" bind:value={firm.target_profit_pct} step="0.1" min="0" />
                  </label>
                  <label>
                    Mode
                    <select bind:value={firm.risk_mode}>
                      <option value="growth">Growth</option>
                      <option value="scaling">Scaling</option>
                      <option value="guardian">Guardian</option>
                    </select>
                  </label>
                </div>
                <div class="edit-actions">
                  <button class="save-btn" onclick={() => saveEdit(firm)} disabled={loading}>
                    <Save size={12} />
                  </button>
                  <button class="cancel-btn" onclick={cancelEdit}>
                    <X size={12} />
                  </button>
                </div>
              </div>
            {:else}
              <div class="firm-header">
                <span class="firm-name">{firm.firm_name}</span>
                <span class="risk-mode" style="color: {getRiskModeColor(firm.risk_mode)}">
                  {firm.risk_mode}
                </span>
              </div>
              <div class="firm-details">
                <span class="account-id">{firm.account_id}</span>
                <span class="limits">
                  {firm.daily_loss_limit_pct}% / {firm.target_profit_pct}%
                </span>
              </div>
              <div class="firm-actions">
                <button class="icon-btn" onclick={() => startEdit(firm)}>
                  <Edit2 size={12} />
                </button>
                <button class="icon-btn delete" onclick={() => deleteFirm(firm.id)}>
                  <Trash2 size={12} />
                </button>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .prop-firm-panel {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 200px;
    max-height: 400px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .panel-title {
    font-size: 14px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .add-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    color: #00d4ff;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
  }

  .add-btn:hover {
    background: rgba(0, 212, 255, 0.2);
  }

  .create-form {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 6px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .form-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .form-header h4 {
    margin: 0;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .form-fields {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .field-row {
    display: flex;
    gap: 8px;
  }

  .field-row .field {
    flex: 1;
  }

  .field label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
  }

  .field input,
  .field select {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    padding: 6px 8px;
    color: rgba(255, 255, 255, 0.9);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .save-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 6px 12px;
    background: rgba(0, 212, 255, 0.2);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 4px;
    color: #00d4ff;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
  }

  .save-btn:hover {
    background: rgba(0, 212, 255, 0.3);
  }

  .panel-content {
    flex: 1;
    overflow-y: auto;
  }

  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
    min-height: 100px;
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .error-state {
    color: #ff3b3b;
  }

  .firm-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .firm-item {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .firm-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .firm-name {
    font-size: 13px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .risk-mode {
    font-size: 10px;
    text-transform: uppercase;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .firm-details {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .firm-actions {
    display: flex;
    gap: 4px;
    margin-top: 4px;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 3px;
    color: #00d4ff;
    cursor: pointer;
  }

  .icon-btn:hover {
    background: rgba(0, 212, 255, 0.2);
  }

  .icon-btn.delete:hover {
    background: rgba(255, 59, 59, 0.2);
    color: #ff3b3b;
  }

  .firm-edit {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .edit-name {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    padding: 6px 8px;
    color: rgba(255, 255, 255, 0.9);
    font-size: 13px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .edit-fields {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .edit-fields label {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 9px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
  }

  .edit-fields input,
  .edit-fields select {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 3px;
    padding: 4px 6px;
    color: rgba(255, 255, 255, 0.9);
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    width: 80px;
  }

  .edit-actions {
    display: flex;
    gap: 4px;
  }

  .cancel-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px 8px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    color: rgba(255, 255, 255, 0.6);
    font-size: 11px;
    cursor: pointer;
  }

  .cancel-btn:hover {
    background: rgba(255, 255, 255, 0.15);
  }
</style>
