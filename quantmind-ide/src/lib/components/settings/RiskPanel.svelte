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
