<script lang="ts">
  import { Shield, ShieldAlert, ShieldX, X, AlertTriangle } from "lucide-svelte";
  import {
    showKillSwitchModal,
    selectedTier,
    killSwitchLoading,
    killSwitchError,
    cancelKillSwitch,
    triggerKillSwitch,
    showEmergencyCloseModal,
    TIER_DESCRIPTIONS,
    type KillSwitchTier
  } from "$lib/stores/kill-switch";

  const tiers: KillSwitchTier[] = [1, 2, 3];

  let selected: KillSwitchTier | null = $state(null);

  function handleSelect(tier: KillSwitchTier) {
    selected = tier;
  }

  async function handleConfirm() {
    if (!selected) return;
    if (selected === 3) {
      // Tier 3: show emergency double-confirmation modal
      selectedTier.set(3);
      showKillSwitchModal.set(false);
      showEmergencyCloseModal.set(true);
    } else {
      // Tiers 1 & 2: fire immediately
      selectedTier.set(selected);
      await triggerKillSwitch(selected);
      if (!$killSwitchError) {
        showKillSwitchModal.set(false);
        selected = null;
      }
    }
  }

  function handleCancel() {
    cancelKillSwitch();
    selected = null;
  }

  function getTierIcon(tier: KillSwitchTier) {
    switch (tier) {
      case 1:
        return Shield;
      case 2:
        return ShieldAlert;
      case 3:
        return ShieldX;
      default:
        return Shield;
    }
  }
</script>

<!-- Kill Switch Tier Selection Modal -->
{#if $showKillSwitchModal}
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
  <div class="modal-overlay" onclick={handleCancel} role="dialog" aria-modal="true" aria-labelledby="kill-modal-title">
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
    <div class="modal-content" onclick={(e) => e.stopPropagation()}>
      <button class="close-btn" onclick={handleCancel} aria-label="Close">
        <X size={18} />
      </button>

      <div class="modal-header">
        <div class="modal-icon">
          <AlertTriangle size={28} />
        </div>
        <h3 id="kill-modal-title">Select Kill Switch Tier</h3>
        <p class="modal-subtitle">Choose the level of protection you want to activate</p>
      </div>

      {#if $killSwitchError}
        <div class="error-banner">
          {$killSwitchError}
        </div>
      {/if}

      <div class="tier-options">
        {#each tiers as tier}
          {@const info = TIER_DESCRIPTIONS[tier]}
          {@const IconComponent = getTierIcon(tier)}
          <button
            class="tier-option"
            class:selected={selected === tier}
            class:tier-1={tier === 1}
            class:tier-2={tier === 2}
            class:tier-3={tier === 3}
            onclick={() => handleSelect(tier)}
          >
            <div class="tier-icon">
              <svelte:component this={IconComponent} size={24} />
            </div>
            <div class="tier-info">
              <span class="tier-label">Tier {tier}: {info.name}</span>
              <span class="tier-description">{info.description}</span>
            </div>
            <div class="tier-check">
              {#if selected === tier}
                <div class="check-circle"></div>
              {/if}
            </div>
          </button>
        {/each}
      </div>

      <div class="modal-actions">
        <button class="btn-cancel" onclick={handleCancel} disabled={$killSwitchLoading}>
          Cancel
        </button>
        <button
          class="btn-confirm"
          onclick={handleConfirm}
          disabled={!selected || $killSwitchLoading}
        >
          {#if $killSwitchLoading}
            <span class="spinner"></span>
            Activating...
          {:else}
            Confirm Tier {selected || '—'}
          {/if}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.75);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: var(--color-bg-surface);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    width: 90%;
    max-width: 480px;
    position: relative;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  }

  .close-btn {
    position: absolute;
    top: 16px;
    right: 16px;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 8px;
    border-radius: 8px;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .close-btn:hover {
    background: var(--glass-tier-2);
    color: var(--color-text-primary);
  }

  .modal-header {
    text-align: center;
    margin-bottom: 24px;
  }

  .modal-icon {
    display: flex;
    justify-content: center;
    margin-bottom: 12px;
    color: var(--color-warning);
  }

  .modal-header h3 {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 20px;
    color: var(--color-text-primary);
    margin-bottom: 4px;
  }

  .modal-subtitle {
    font-size: 13px;
    color: var(--color-text-secondary);
  }

  .error-banner {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
    color: #ef4444;
    font-size: 13px;
    text-align: center;
  }

  .tier-options {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 24px;
  }

  .tier-option {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: var(--color-bg-elevated);
    border: 2px solid transparent;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
    width: 100%;
  }

  .tier-option:hover {
    background: var(--glass-tier-2);
  }

  .tier-option.selected {
    border-color: var(--color-accent-cyan);
    background: rgba(6, 182, 212, 0.1);
  }

  .tier-option.tier-1 .tier-icon {
    color: #f59e0b;
  }

  .tier-option.tier-2 .tier-icon {
    color: #f97316;
  }

  .tier-option.tier-3 .tier-icon {
    color: #ef4444;
  }

  .tier-icon {
    flex-shrink: 0;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-surface);
    border-radius: 10px;
  }

  .tier-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .tier-label {
    font-family: var(--font-nav);
    font-weight: 600;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  .tier-description {
    font-size: 12px;
    color: var(--color-text-secondary);
    line-height: 1.4;
  }

  .tier-check {
    flex-shrink: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .check-circle {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--color-accent-cyan);
    border: 2px solid var(--color-accent-cyan);
    box-shadow: 0 0 8px var(--color-accent-cyan);
  }

  .modal-actions {
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .btn-cancel,
  .btn-confirm {
    padding: 12px 24px;
    border-radius: 8px;
    font-family: var(--font-nav);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .btn-cancel {
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    color: var(--color-text-secondary);
  }

  .btn-cancel:hover:not(:disabled) {
    background: var(--bg-surface);
    color: var(--color-text-primary);
  }

  .btn-confirm {
    background: var(--color-accent-red);
    border: none;
    color: white;
  }

  .btn-confirm:hover:not(:disabled) {
    background: #ff5555;
    box-shadow: 0 0 16px rgba(255, 85, 85, 0.4);
  }

  .btn-cancel:disabled,
  .btn-confirm:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
