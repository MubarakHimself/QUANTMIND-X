<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Shield } from 'lucide-svelte';

  export let riskSettings = {
    houseMoneyEnabled: true,
    houseMoneyThreshold: 0.5,
    dailyLossLimit: 5,
    maxDrawdown: 10,
    riskMode: 'dynamic' as 'fixed' | 'dynamic' | 'conservative',
    propFirmPreset: 'custom' as 'ftmo' | 'the5ers' | 'fundingpips' | 'custom',
    balanceZones: {
      danger: 200,
      growth: 1000,
      scaling: 5000,
      guardian: Infinity
    }
  };

  const PROP_FIRM_PRESETS = {
    ftmo: { name: 'FTMO', maxRisk: 2, dailyLoss: 5, totalLoss: 10 },
    the5ers: { name: 'The5ers', maxRisk: 2.5, dailyLoss: 6, totalLoss: 12 },
    fundingpips: { name: 'FundingPips', maxRisk: 3, dailyLoss: 8, totalLoss: 15 },
    custom: { name: 'Custom', maxRisk: 0, dailyLoss: 0, totalLoss: 0 }
  };

  const dispatch = createEventDispatcher();

  function updateRiskSettings(field: string, value: any) {
    dispatch('updateRiskSettings', { field, value });
  }

  // Auto-apply preset values when prop firm preset changes
  function handlePresetChange(event: Event) {
    const select = event.target as HTMLSelectElement;
    const preset = select.value as keyof typeof PROP_FIRM_PRESETS;

    // Dispatch preset change
    dispatch('updateRiskSettings', { field: 'propFirmPreset', value: preset });

    // Auto-apply preset values to risk settings
    if (preset !== 'custom' && PROP_FIRM_PRESETS[preset]) {
      const presetValues = PROP_FIRM_PRESETS[preset];
      // Apply max risk
      dispatch('updateRiskSettings', { field: 'maxRiskPerTrade', value: presetValues.maxRisk / 100 });
      // Apply daily loss limit
      dispatch('updateRiskSettings', { field: 'dailyLossLimit', value: presetValues.dailyLoss });
      // Apply max drawdown (total loss)
      dispatch('updateRiskSettings', { field: 'maxDrawdown', value: presetValues.totalLoss });
    }
  }
</script>

<div class="panel">
  <h3>Risk Management</h3>

  <div class="setting-group">
    <label>House Money Effect</label>
    <div class="setting-row">
      <span>Enable House Money</span>
      <label class="switch">
        <input type="checkbox" bind:checked={riskSettings.houseMoneyEnabled} on:change={(e) => updateRiskSettings('houseMoneyEnabled', e.currentTarget.checked)} />
        <span class="slider"></span>
      </label>
    </div>
    <div class="setting-row">
      <span>Threshold (% of daily profit)</span>
      <input
        type="number"
        min="0"
        max="100"
        bind:value={riskSettings.houseMoneyThreshold}
        class="number-input"
        on:input={(e) => updateRiskSettings('houseMoneyThreshold', parseFloat(e.currentTarget.value))}
      />
    </div>
  </div>

  <div class="setting-group">
    <label>Risk Limits</label>
    <div class="setting-row">
      <span>Daily Loss Limit (%)</span>
      <input
        type="number"
        min="1"
        max="20"
        bind:value={riskSettings.dailyLossLimit}
        class="number-input"
        on:input={(e) => updateRiskSettings('dailyLossLimit', parseFloat(e.currentTarget.value))}
      />
    </div>
    <div class="setting-row">
      <span>Max Drawdown (%)</span>
      <input
        type="number"
        min="1"
        max="50"
        bind:value={riskSettings.maxDrawdown}
        class="number-input"
        on:input={(e) => updateRiskSettings('maxDrawdown', parseFloat(e.currentTarget.value))}
      />
    </div>
  </div>

  <div class="setting-group">
    <label>Prop Firm Preset</label>
    <div class="setting-row">
      <span>Select Preset</span>
      <select bind:value={riskSettings.propFirmPreset} class="select-input" on:change={handlePresetChange}>
        <option value="ftmo">FTMO</option>
        <option value="the5ers">The5ers</option>
        <option value="fundingpips">FundingPips</option>
        <option value="custom">Custom</option>
      </select>
    </div>
    {#if riskSettings.propFirmPreset !== 'custom'}
      <p class="hint">
        Preset: {PROP_FIRM_PRESETS[riskSettings.propFirmPreset].name} -
        Max Risk: {PROP_FIRM_PRESETS[riskSettings.propFirmPreset].maxRisk}%,
        Daily Loss: {PROP_FIRM_PRESETS[riskSettings.propFirmPreset].dailyLoss}%,
        Total Loss: {PROP_FIRM_PRESETS[riskSettings.propFirmPreset].totalLoss}%
      </p>
    {/if}
  </div>

  <div class="setting-group">
    <label>Risk Mode</label>
    <div class="setting-row">
      <span>Mode</span>
      <select bind:value={riskSettings.riskMode} on:change={(e) => updateRiskSettings('riskMode', e.currentTarget.value)}>
        <option value="fixed">Fixed (constant risk)</option>
        <option value="dynamic">Dynamic (adjusts to conditions)</option>
        <option value="conservative">Conservative (protects capital)</option>
      </select>
    </div>
  </div>

  <div class="setting-group">
    <label>Balance Zones</label>
    <div class="zones-grid">
      <div class="zone-item danger">
        <span class="zone-label">DANGER</span>
        <span class="zone-amount">${riskSettings.balanceZones.danger}</span>
      </div>
      <div class="zone-item growth">
        <span class="zone-label">GROWTH</span>
        <span class="zone-amount">${riskSettings.balanceZones.growth}</span>
      </div>
      <div class="zone-item scaling">
        <span class="zone-label">SCALING</span>
        <span class="zone-amount">${riskSettings.balanceZones.scaling}</span>
      </div>
      <div class="zone-item guardian">
        <span class="zone-label">GUARDIAN</span>
        <span class="zone-amount">{riskSettings.balanceZones.guardian === Infinity ? '∞' : '$' + riskSettings.balanceZones.guardian}</span>
      </div>
    </div>
  </div>
</div>

<style>
  /* Panel Header */
  .panel h3 {
    margin: 0 0 20px;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Setting Group */
  .setting-group {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .setting-group:last-child {
    margin-bottom: 0;
  }

  .setting-group > label {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
  }

  /* Setting Row */
  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
  }

  .setting-row:not(:last-child) {
    border-bottom: 1px solid var(--border-subtle);
  }

  .setting-row span {
    font-size: 13px;
    color: var(--text-secondary);
  }

  /* Inputs */
  .number-input,
  .select-input {
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    min-width: 120px;
    transition: all 0.15s;
  }

  .number-input:focus,
  .select-input:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
  }

  /* Toggle Switch */
  .switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
  }

  .switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    transition: 0.2s;
    border-radius: 24px;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 2px;
    bottom: 2px;
    background-color: var(--text-muted);
    transition: 0.2s;
    border-radius: 50%;
  }

  input:checked + .slider {
    background-color: var(--accent-primary);
    border-color: var(--accent-primary);
  }

  input:checked + .slider:before {
    transform: translateX(20px);
    background-color: white;
  }

  /* Hint Text */
  .hint {
    font-size: 12px;
    color: var(--text-muted);
    margin: 8px 0 0;
    padding: 8px 12px;
    background: var(--bg-primary);
    border-radius: 6px;
  }

  /* Balance Zones Grid */
  .zones-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 12px;
  }

  .zone-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 8px;
    border-radius: 8px;
    text-align: center;
  }

  .zone-item.danger {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .zone-item.growth {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
  }

  .zone-item.scaling {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.3);
  }

  .zone-item.guardian {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
  }

  .zone-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }

  .zone-item.danger .zone-label { color: #ef4444; }
  .zone-item.growth .zone-label { color: #f59e0b; }
  .zone-item.scaling .zone-label { color: #3b82f6; }
  .zone-item.guardian .zone-label { color: #10b981; }

  .zone-amount {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Section Headers */
  h4 {
    margin: 24px 0 12px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  h4:first-child {
    margin-top: 0;
  }
</style>
