/**
 * Server Health Alert Store (Story 10-5 AC3)
 *
 * Fires when a node metric crosses a critical threshold.
 * ServerHealthPanel writes here; CopilotPanel reads and displays a system message.
 * Alerts are also persisted to sessionStorage so missed alerts (panel unmounted)
 * are restored when CopilotPanel remounts.
 */
import { writable } from 'svelte/store';

const STORAGE_KEY = 'lastServerHealthAlert';

export interface ServerHealthAlert {
  /** Unique ID: "{node}-{metric}-{timestamp}" */
  id: string;
  /** Node name: 'Contabo' | 'Cloudzy' */
  node: string;
  /** Metric name: 'CPU' | 'Memory' | 'Disk' | 'Latency' */
  metric: string;
  /** Current value */
  value: number;
  /** Unit: '%' | 'ms' */
  unit: string;
  /** Human-readable message for Copilot display — AC3 format:
   * "Contabo: disk usage at 91%. Action recommended." */
  message: string;
  timestamp: Date;
}

/** Persist last alert so missed alerts (panel unmounted) are restored on remount. */
function persistAlert(alert: ServerHealthAlert | null) {
  try {
    if (alert) {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(alert));
    } else {
      sessionStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // sessionStorage may be unavailable (e.g. SSR, private mode restrictions)
  }
}

/**
 * Fires when a server metric crosses its critical threshold.
 * Set to null initially; updated with each new breach event.
 * CopilotPanel subscribes and injects a system message on change.
 */
export const serverHealthAlertEvent = writable<ServerHealthAlert | null>(null);

/** Read the last persisted alert (for restoring missed alerts on CopilotPanel mount). */
export function getLastPersistedAlert(): ServerHealthAlert | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    // Rehydrate Date
    return { ...parsed, timestamp: new Date(parsed.timestamp) };
  } catch {
    return null;
  }
}

/** Write alert to store AND persist to sessionStorage. */
export function fireServerHealthAlert(alert: ServerHealthAlert) {
  serverHealthAlertEvent.set(alert);
  persistAlert(alert);
}

/** Clear the persisted alert (e.g. when metric recovers). */
export function clearServerHealthAlert() {
  serverHealthAlertEvent.set(null);
  persistAlert(null);
}

