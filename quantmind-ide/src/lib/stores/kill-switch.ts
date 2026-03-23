/**
 * Kill Switch State Store
 *
 * Manages the state for the TradingKillSwitch in TopBar:
 * - Arm/disarm state
 * - Countdown timer
 * - Selected tier
 * - API status
 */

import { writable, derived, get } from 'svelte/store';

// Kill switch states
export type KillSwitchState = 'ready' | 'armed' | 'fired';

// Kill switch tier types
export type KillSwitchTier = 1 | 2 | 3;

// Tier descriptions
export const TIER_DESCRIPTIONS: Record<KillSwitchTier, { name: string; description: string; icon: string }> = {
  1: {
    name: 'Soft Stop',
    description: 'Stop new positions from being opened. Existing positions remain open.',
    icon: 'shield'
  },
  2: {
    name: 'Strategy Pause',
    description: 'Pause specific trading strategies. Other strategies continue.',
    icon: 'pause'
  },
  3: {
    name: 'Emergency Close',
    description: 'Close all open positions immediately. Most aggressive option.',
    icon: 'x-circle'
  }
};

// Store for kill switch state
export const killSwitchState = writable<KillSwitchState>('ready');

// Store for countdown (in seconds)
export const killSwitchCountdown = writable<number>(0);

// Store for selected tier
export const selectedTier = writable<KillSwitchTier | null>(null);

// Store for modal visibility
export const showKillSwitchModal = writable<boolean>(false);

// Store for emergency close modal visibility
export const showEmergencyCloseModal = writable<boolean>(false);

// Store for fired status (persists until app restart)
export const killSwitchFired = writable<boolean>(false);

// Store for loading state during API calls
export const killSwitchLoading = writable<boolean>(false);

// Store for error messages
export const killSwitchError = writable<string | null>(null);

// Derived store for aria-label
export const killSwitchAriaLabel = derived(
  [killSwitchState, killSwitchFired],
  ([$state, $fired]) => {
    if ($fired) return 'Trading stopped';
    if ($state === 'armed') return 'Armed — click Confirm';
    return 'Emergency stop — click to arm';
  }
);

// Countdown timer interval reference
let countdownInterval: ReturnType<typeof setInterval> | null = null;

/**
 * Arm the kill switch - starts countdown
 */
export function armKillSwitch() {
  killSwitchError.set(null);
  killSwitchState.set('armed');
  killSwitchCountdown.set(2); // 2 second countdown

  // Clear any existing interval
  if (countdownInterval) {
    clearInterval(countdownInterval);
  }

  // Start countdown
  countdownInterval = setInterval(() => {
    killSwitchCountdown.update((count) => {
      if (count <= 1) {
        // Countdown complete - open modal
        if (countdownInterval) {
          clearInterval(countdownInterval);
          countdownInterval = null;
        }
        showKillSwitchModal.set(true);
        return 0;
      }
      return count - 1;
    });
  }, 1000);
}

/**
 * Disarm the kill switch - cancels countdown
 */
export function disarmKillSwitch() {
  killSwitchState.set('ready');
  killSwitchCountdown.set(0);

  if (countdownInterval) {
    clearInterval(countdownInterval);
    countdownInterval = null;
  }
}

/**
 * Cancel the modal without firing
 */
export function cancelKillSwitch() {
  selectedTier.set(null);
  showKillSwitchModal.set(false);
  showEmergencyCloseModal.set(false);
  killSwitchError.set(null);
  disarmKillSwitch();
}

/**
 * Select a tier and proceed
 */
export function selectTier(tier: KillSwitchTier) {
  selectedTier.set(tier);

  // If Tier 3, show emergency close modal first
  if (tier === 3) {
    showKillSwitchModal.set(false);
    showEmergencyCloseModal.set(true);
  }
}

/**
 * Trigger the kill switch - calls the API
 */
export async function triggerKillSwitch(tier: KillSwitchTier, strategyIds?: string[]): Promise<boolean> {
  killSwitchLoading.set(true);
  killSwitchError.set(null);

  try {
    const response = await fetch('/api/kill-switch/trigger', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        tier,
        strategy_ids: strategyIds || null,
        activator: 'UI_User'
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to trigger kill switch');
    }

    const data = await response.json();

    if (data.success) {
      killSwitchState.set('ready');
      killSwitchFired.set(true);
      showKillSwitchModal.set(false);
      showEmergencyCloseModal.set(false);
      selectedTier.set(null);
      disarmKillSwitch();
      return true;
    } else {
      throw new Error(data.message || 'Kill switch trigger failed');
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    killSwitchError.set(message);
    console.error('[KillSwitch] Trigger failed:', message);
    return false;
  } finally {
    killSwitchLoading.set(false);
  }
}

/**
 * Confirm and fire the selected tier
 */
export async function confirmKillSwitch(): Promise<boolean> {
  const tier = get(selectedTier);
  if (!tier) {
    killSwitchError.set('No tier selected');
    return false;
  }

  return triggerKillSwitch(tier);
}

/**
 * Get kill switch status from API
 */
export async function fetchKillSwitchStatus(): Promise<void> {
  try {
    const response = await fetch('/api/kill-switch/status');
    if (!response.ok) {
      throw new Error('Failed to fetch kill switch status');
    }

    const data = await response.json();

    // Hydrate kill switch state from backend response.
    // The backend KillSwitchStatusResponse shape:
    //   { enabled, current_alert_level, active_alerts_count, tiers: { tier1_bot, tier2_strategy, ... } }
    //
    // Additionally, check the tier-trigger audit log to determine if a UI-tier (1-3) was fired.
    if (data.tiers) {
      const tier1 = data.tiers.tier1_bot;
      const tier2 = data.tiers.tier2_strategy;
      const tier3 = data.tiers.tier3_account;

      // Tier 1 fired: quarantined_count > 0 indicates bots were halted
      const tier1Active = tier1 && (tier1.quarantined_count > 0 || tier1.status === 'active' || tier1.enabled === true);
      // Tier 2 fired: quarantined strategy families present
      const tier2Active = tier2 && (
        (Array.isArray(tier2.quarantined_families) && tier2.quarantined_families.length > 0) ||
        tier2.status === 'active' || tier2.enabled === true
      );
      // Tier 3 fired: accounts at risk
      const tier3Active = tier3 && (tier3.accounts_at_risk > 0 || tier3.status === 'active' || tier3.enabled === true);

      if (tier3Active) {
        killSwitchFired.set(true);
        selectedTier.set(3);
      } else if (tier2Active) {
        killSwitchFired.set(true);
        selectedTier.set(2);
      } else if (tier1Active) {
        killSwitchFired.set(true);
        selectedTier.set(1);
      }
    }
  } catch (error) {
    console.error('[KillSwitch] Failed to fetch status:', error);
  }
}
