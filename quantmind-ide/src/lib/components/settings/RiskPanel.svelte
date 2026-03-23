<script lang="ts">
  import { onMount } from 'svelte';
  import { Shield, Save, RefreshCw, Check, AlertCircle, TrendingDown } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  interface RiskSettings {
    houseMoneyEnabled: boolean;
    houseMoneyThreshold: number;
    dailyLossLimit: number;
    maxDrawdown: number;
    riskMode: 'fixed' | 'dynamic' | 'conservative';
    propFirmPreset: 'ftmo' | 'the5ers' | 'fundingpips' | 'custom';
    balanceZones: {
      danger: number;
      growth: number;
      scaling: number;
      guardian: number;
    };
  }

  const PROP_FIRM_PRESETS = {
    ftmo:        { name: 'FTMO',        maxRisk: 2,   dailyLoss: 5,  totalLoss: 10 },
    the5ers:     { name: 'The5ers',     maxRisk: 2.5, dailyLoss: 6,  totalLoss: 12 },
    fundingpips: { name: 'FundingPips', maxRisk: 3,   dailyLoss: 8,  totalLoss: 15 },
    custom:      { name: 'Custom',      maxRisk: 0,   dailyLoss: 0,  totalLoss: 0  }
  };

  let riskSettings: RiskSettings = $state({
    houseMoneyEnabled: true,
    houseMoneyThreshold: 0.5,
    dailyLossLimit: 5,
    maxDrawdown: 10,
    riskMode: 'dynamic',
    propFirmPreset: 'custom',
    balanceZones: { danger: 200, growth: 1000, scaling: 5000, guardian: 999999 }
  });

  let isLoading = $state(false);
  let isSaving = $state(false);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);

  async function loadSettings() {
    isLoading = true;
    error = null;
    try {
      const data = await apiFetch<RiskSettings>('/api/risk/params');
      riskSettings = { ...riskSettings, ...data };
    } catch (e) {
      console.warn('Risk params endpoint unavailable, using defaults');
    } finally {
      isLoading = false;
    }
  }

  async function saveSettings() {
    isSaving = true;
    error = null;
    success = null;
    try {
      await apiFetch('/api/risk/params', {
        method: 'POST',
        body: JSON.stringify(riskSettings)
      });
      success = 'Risk parameters saved';
      setTimeout(() => success = null, 3000);
    } catch (e) {
      error = 'Failed to save risk parameters';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  function handlePresetChange(preset: keyof typeof PROP_FIRM_PRESETS) {
    riskSettings.propFirmPreset = preset;
    if (preset !== 'custom') {
      const p = PROP_FIRM_PRESETS[preset];
      riskSettings.dailyLossLimit = p.dailyLoss;
      riskSettings.maxDrawdown = p.totalLoss;
    }
  }

  onMount(() => { loadSettings(); });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Risk Management</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={loadSettings} title="Refresh" disabled={isLoading}>
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert-error"><AlertCircle size={14} /> <span>{error}</span></div>
  {/if}
  {#if success}
    <div class="alert-success"><Check size={14} /> <span>{success}</span></div>
  {/if}

  <!-- Prop Firm Preset -->
  <div class="settings-section">
    <div class="section-title">Prop Firm Preset</div>
    <div class="preset-grid">
      {#each Object.entries(PROP_FIRM_PRESETS) as [key, preset]}
        <button
          class="preset-btn"
          class:active={riskSettings.propFirmPreset === key}
          onclick={() => handlePresetChange(key as keyof typeof PROP_FIRM_PRESETS)}
        >
          <span class="preset-name">{preset.name}</span>
          {#if key !== 'custom'}
            <span class="preset-detail">DL: {preset.dailyLoss}% / TD: {preset.totalLoss}%</span>
          {/if}
        </button>
      {/each}
    </div>
  </div>

  <!-- Risk Limits -->
  <div class="settings-section">
    <div class="section-title">Risk Limits</div>

    <div class="setting-row">
      <span>Daily Loss Limit (%)</span>
      <input
        type="number"
        min="1"
        max="20"
        step="0.5"
        class="number-input"
        bind:value={riskSettings.dailyLossLimit}
      />
    </div>
    <div class="setting-row">
      <span>Max Drawdown (%)</span>
      <input
        type="number"
        min="1"
        max="50"
        step="0.5"
        class="number-input"
        bind:value={riskSettings.maxDrawdown}
      />
    </div>
    <div class="setting-row">
      <span>Risk Mode</span>
      <select class="select-input" bind:value={riskSettings.riskMode}>
        <option value="fixed">Fixed</option>
        <option value="dynamic">Dynamic</option>
        <option value="conservative">Conservative</option>
      </select>
    </div>
  </div>

  <!-- House Money -->
  <div class="settings-section">
    <div class="section-title">House Money Effect</div>
    <div class="setting-row">
      <span>Enable House Money</span>
      <label class="switch">
        <input type="checkbox" bind:checked={riskSettings.houseMoneyEnabled} />
        <span class="slider"></span>
      </label>
    </div>
    {#if riskSettings.houseMoneyEnabled}
      <div class="setting-row">
        <span>Threshold (% of daily profit)</span>
        <input
          type="number"
          min="0"
          max="100"
          step="5"
          class="number-input"
          bind:value={riskSettings.houseMoneyThreshold}
        />
      </div>
    {/if}
  </div>

  <!-- Balance Zones -->
  <div class="settings-section">
    <div class="section-title">Balance Zones</div>
    <div class="zones-grid">
      <div class="zone-item danger">
        <TrendingDown size={12} />
        <span class="zone-label">Danger</span>
        <span class="zone-amount">${riskSettings.balanceZones.danger.toLocaleString()}</span>
      </div>
      <div class="zone-item growth">
        <Shield size={12} />
        <span class="zone-label">Growth</span>
        <span class="zone-amount">${riskSettings.balanceZones.growth.toLocaleString()}</span>
      </div>
      <div class="zone-item scaling">
        <Shield size={12} />
        <span class="zone-label">Scaling</span>
        <span class="zone-amount">${riskSettings.balanceZones.scaling.toLocaleString()}</span>
      </div>
      <div class="zone-item guardian">
        <Shield size={12} />
        <span class="zone-label">Guardian</span>
        <span class="zone-amount">∞</span>
      </div>
    </div>
  </div>

  <div class="action-row">
    <button class="btn primary" onclick={saveSettings} disabled={isSaving}>
      <Save size={14} />
      {isSaving ? 'Saving...' : 'Save Risk Parameters'}
    </button>
  </div>
</div>

<style>
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary, #e8eaf0);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .header-actions { display: flex; gap: 8px; }

  .alert-error {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.25);
    border-radius: 6px;
    color: #ff3b3b;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .alert-success {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    border-radius: 6px;
    color: #00c896;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .settings-section {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.5);
    margin-bottom: 14px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  /* Preset Grid */
  .preset-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }

  .preset-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 10px 8px;
    background: rgba(8, 13, 20, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .preset-btn:hover {
    border-color: rgba(255, 255, 255, 0.15);
  }

  .preset-btn.active {
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.35);
  }

  .preset-name {
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: rgba(255, 255, 255, 0.6);
  }

  .preset-btn.active .preset-name { color: #00d4ff; }

  .preset-detail {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-align: center;
  }

  /* Setting Row */
  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .setting-row:last-child { border-bottom: none; }

  .number-input {
    width: 80px;
    padding: 6px 10px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    color: #e8eaf0;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-align: right;
    transition: border-color 0.15s;
    -moz-appearance: textfield;
  }

  .number-input::-webkit-outer-spin-button,
  .number-input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }

  .number-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
  }

  .select-input {
    padding: 6px 10px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    color: #e8eaf0;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .select-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
  }

  /* Toggle Switch */
  .switch {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
  }

  .switch input { opacity: 0; width: 0; height: 0; }

  .slider {
    position: absolute;
    cursor: pointer;
    inset: 0;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    transition: 0.2s;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 14px;
    width: 14px;
    left: 3px;
    bottom: 3px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 50%;
    transition: 0.2s;
  }

  input:checked + .slider {
    background: rgba(0, 212, 255, 0.25);
    border-color: rgba(0, 212, 255, 0.4);
  }

  input:checked + .slider:before {
    transform: translateX(18px);
    background: #00d4ff;
  }

  /* Balance Zones */
  .zones-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }

  .zone-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 12px 6px;
    border-radius: 7px;
    text-align: center;
  }

  .zone-item.danger {
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.2);
    color: #ff3b3b;
  }

  .zone-item.growth {
    background: rgba(240, 165, 0, 0.1);
    border: 1px solid rgba(240, 165, 0, 0.2);
    color: #f0a500;
  }

  .zone-item.scaling {
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    color: #00d4ff;
  }

  .zone-item.guardian {
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid rgba(0, 200, 150, 0.2);
    color: #00c896;
  }

  .zone-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    opacity: 0.8;
  }

  .zone-amount {
    font-size: 13px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: #e8eaf0;
  }

  .action-row {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 4px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }

  .btn.primary {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.4);
    color: #00d4ff;
  }

  .btn.primary:hover { background: rgba(0, 212, 255, 0.25); }
  .btn.primary:disabled { opacity: 0.45; cursor: not-allowed; }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover { background: rgba(255, 255, 255, 0.1); color: #e8eaf0; }
  .icon-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  .spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
