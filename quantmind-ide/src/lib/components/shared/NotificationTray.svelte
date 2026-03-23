<script lang="ts">
  /**
   * NotificationTray — notification overlay for SSE-sourced alerts.
   * Positioned fixed bottom-right; uses --color-accent-cyan border.
   * Story 12-3
   */
  import { X, Bell } from 'lucide-svelte';

  interface Notification {
    id: string;
    message: string;
    type?: 'info' | 'warning' | 'success' | 'error';
    timestamp?: Date;
  }

  interface Props {
    notifications?: Notification[];
    onDismiss?: (id: string) => void;
  }

  let { notifications = [], onDismiss }: Props = $props();
</script>

{#if notifications.length > 0}
  <div class="notification-tray" role="log" aria-label="Notifications" aria-live="polite">
    <div class="tray-header">
      <Bell size={14} />
      <span>Alerts</span>
    </div>
    <div class="tray-list">
      {#each notifications as notif (notif.id)}
        <div class="tray-item tray-item--{notif.type ?? 'info'}">
          <span class="tray-message">{notif.message}</span>
          {#if onDismiss}
            <button
              class="tray-dismiss"
              onclick={() => onDismiss?.(notif.id)}
              aria-label="Dismiss notification"
            >
              <X size={12} />
            </button>
          {/if}
        </div>
      {/each}
    </div>
  </div>
{/if}

<style>
  .notification-tray {
    position: fixed;
    bottom: var(--space-4);
    right: var(--space-4);
    z-index: 1000;
    width: 300px;
    background: var(--glass-content-bg);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--color-accent-cyan);
    border-radius: 8px;
    overflow: hidden;
  }

  .tray-header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--color-accent-cyan);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .tray-list {
    display: flex;
    flex-direction: column;
    max-height: 240px;
    overflow-y: auto;
  }

  .tray-item {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .tray-item:last-child {
    border-bottom: none;
  }

  .tray-message {
    font-family: var(--font-body);
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    flex: 1;
  }

  .tray-dismiss {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--color-text-muted);
    padding: 0;
    display: flex;
    align-items: center;
    flex-shrink: 0;
    transition: color 0.15s ease;
  }

  .tray-dismiss:hover {
    color: var(--color-text-primary);
  }
</style>
