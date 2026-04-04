/**
 * Inter-Session Cooldown Store
 *
 * Manages real-time state for the Inter-Session Cooldown Panel:
 * - Cooldown active state and countdown
 * - Current step within the 4-step cooldown sequence
 * - Session transition information (LONDON -> NEW_YORK)
 * - Intelligence window status
 * - Actions blocked indicator
 *
 * Data from /api/trading/cooldown/status
 * Primary updates come via polling (10s interval) as fallback for Redis pub/sub.
 */

import { writable, derived, get } from 'svelte/store';

// =============================================================================
// Types
// =============================================================================

export type CooldownPhase = 'STEP_1' | 'STEP_2' | 'STEP_3' | 'STEP_4' | 'COMPLETED';

export interface CooldownStatus {
  is_active: boolean;
  session_transition: string | null;  // "LONDON -> NEW_YORK" during cooldown
  cooldown_end_time: string | null;   // ISO timestamp
  hours_remaining: number;
  minutes_remaining: number;
  current_session: string | null;     // "LONDON"
  next_session: string | null;        // "NEW_YORK"
  intelligence_window_active: boolean;
  actions_blocked: boolean;
  progress: number;  // 0.0 to 1.0
  state: string;    // Internal cooldown state
  step_name: string | null;
  current_step: number;  // 0-4
  window_start: string | null;
  window_end: string | null;
  ny_roster_locked: boolean;
}

export interface CooldownStateEvent {
  state: string;
  current_step: number;
  step_name: string;
  window_start: string | null;
  window_end: string | null;
  ny_roster_locked: boolean;
  timestamp_utc: string;
  metadata: Record<string, unknown>;
}

// =============================================================================
// Store State
// =============================================================================

export const cooldownStatus = writable<CooldownStatus | null>(null);
export const cooldownStateEvent = writable<CooldownStateEvent | null>(null);
export const cooldownLoading = writable(false);
export const cooldownError = writable<string | null>(null);

// =============================================================================
// Polling State
// =============================================================================

let cooldownPollInterval: number | null = null;
const COOLDOWN_POLL_INTERVAL_MS = 10000; // 10 seconds

// =============================================================================
// API Fetch
// =============================================================================

function getBaseUrl(): string {
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  return '';
}

/**
 * Fetch cooldown status from API endpoint
 */
export async function fetchCooldownStatus(): Promise<CooldownStatus | null> {
  cooldownLoading.set(true);
  cooldownError.set(null);

  try {
    const baseUrl = getBaseUrl();
    const response = await fetch(`${baseUrl}/api/trading/cooldown/status`, {
      credentials: 'include'
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch cooldown status: ${response.status}`);
    }

    const data = await response.json() as CooldownStatus;
    cooldownStatus.set(data);
    cooldownLoading.set(false);
    return data;
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Unknown error';
    cooldownError.set(message);
    cooldownLoading.set(false);
    console.error('[CooldownStore] Failed to fetch cooldown status:', e);
    return null;
  }
}

/**
 * Fetch full cooldown state event from API endpoint
 */
export async function fetchCooldownStateEvent(): Promise<CooldownStateEvent | null> {
  try {
    const baseUrl = getBaseUrl();
    const response = await fetch(`${baseUrl}/api/trading/cooldown/state-event`, {
      credentials: 'include'
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch cooldown state event: ${response.status}`);
    }

    const data = await response.json() as CooldownStateEvent;
    cooldownStateEvent.set(data);
    return data;
  } catch (e) {
    console.error('[CooldownStore] Failed to fetch cooldown state event:', e);
    return null;
  }
}

// =============================================================================
// Polling Control
// =============================================================================

/**
 * Start polling cooldown status every 10 seconds
 */
export function startCooldownPolling(): void {
  if (cooldownPollInterval !== null) {
    return; // Already polling
  }

  // Fetch immediately
  fetchCooldownStatus();

  // Then poll every 10 seconds
  cooldownPollInterval = window.setInterval(() => {
    fetchCooldownStatus();
  }, COOLDOWN_POLL_INTERVAL_MS);
}

/**
 * Stop polling cooldown status
 */
export function stopCooldownPolling(): void {
  if (cooldownPollInterval !== null) {
    clearInterval(cooldownPollInterval);
    cooldownPollInterval = null;
  }
}

// =============================================================================
// Derived Stores
// =============================================================================

/**
 * Whether cooldown is currently active
 */
export const isCooldownActive = derived(
  cooldownStatus,
  ($cooldownStatus) => $cooldownStatus?.is_active ?? false
);

/**
 * Countdown display text (e.g., "2h 34m")
 */
export const cooldownCountdown = derived(
  cooldownStatus,
  ($cooldownStatus) => {
    if (!$cooldownStatus || !$cooldownStatus.is_active) {
      return null;
    }
    const { hours_remaining, minutes_remaining } = $cooldownStatus;
    if (hours_remaining > 0) {
      return `${hours_remaining}h ${minutes_remaining}m`;
    }
    return `${minutes_remaining}m`;
  }
);

/**
 * Session transition display (e.g., "LONDON -> NEW YORK")
 */
export const sessionTransition = derived(
  cooldownStatus,
  ($cooldownStatus) => $cooldownStatus?.session_transition ?? null
);

/**
 * Progress as percentage (0-100)
 */
export const cooldownProgressPercent = derived(
  cooldownStatus,
  ($cooldownStatus) => Math.round(($cooldownStatus?.progress ?? 0) * 100)
);

/**
 * Current step name
 */
export const cooldownStepName = derived(
  cooldownStatus,
  ($cooldownStatus) => $cooldownStatus?.step_name ?? null
);

/**
 * Whether actions are blocked during cooldown
 */
export const areActionsBlocked = derived(
  cooldownStatus,
  ($cooldownStatus) => $cooldownStatus?.actions_blocked ?? false
);

/**
 * Whether the intelligence window is active
 */
export const isIntelligenceWindowActive = derived(
  cooldownStatus,
  ($cooldownStatus) => $cooldownStatus?.intelligence_window_active ?? false
);

/**
 * Cooldown status color for UI display
 * - Amber (#F59E0B) when active cooldown
 * - Green (#10B981) when open
 */
export const cooldownStatusColor = derived(
  cooldownStatus,
  ($cooldownStatus) => {
    if (!$cooldownStatus) return '#6B7280'; // Default grey
    if ($cooldownStatus.is_active) return '#F59E0B'; // Amber
    return '#10B981'; // Green
  }
);

/**
 * Current step number (1-4)
 */
export const cooldownCurrentStep = derived(
  cooldownStatus,
  ($cooldownStatus) => $cooldownStatus?.current_step ?? 0
);

/**
 * Step descriptions for each cooldown step
 */
export const COOLDOWN_STEP_INFO: Record<number, { name: string; duration: string; description: string }> = {
  0: { name: 'Pending', duration: '10:00-10:30', description: 'London Mid transition' },
  1: { name: 'London Session Scoring', duration: '10:00-10:30', description: 'DPR finalises London performer scores' },
  2: { name: 'Paper Recovery Review', duration: '10:30-11:30', description: 'TIER_1 paper bots recovery review' },
  3: { name: 'NY Queue Order', duration: '11:30-12:40', description: 'Hybrid NY queue via DPR + Tier Remix' },
  4: { name: 'System Health Check', duration: '12:40-13:00', description: 'SVSS/SQS/Sentinel pre-check' },
};
