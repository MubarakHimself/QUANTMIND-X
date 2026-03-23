<script lang="ts">
  import { Bell, BellOff, Lock, RefreshCw, Check, AlertCircle } from 'lucide-svelte';

  interface NotificationEvent {
    event_type: string;
    category: string;
    severity: string;
    enabled: boolean;
    delivery_channel: string;
    is_always_on: boolean;
    description: string | null;
  }

  let events: NotificationEvent[] = $state([]);
  let categories: string[] = $state([]);
  let isLoading = $state(true);
  let isSaving = $state(false);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);

  // Group events by category
  let eventsByCategory = $derived(
    categories.reduce((acc, cat) => {
      acc[cat] = events.filter(e => e.category === cat);
      return acc;
    }, {} as Record<string, NotificationEvent[]>)
  );

  async function loadNotifications() {
    isLoading = true;
    error = null;
    try {
      const response = await fetch('/api/notifications');
      if (response.ok) {
        const data = await response.json();
        events = data.events;
        categories = data.categories;
      } else {
        error = 'Failed to load notification settings';
      }
    } catch (e) {
      error = 'Failed to load notification settings';
      console.error(e);
    } finally {
      isLoading = false;
    }
  }

  async function toggleNotification(event: NotificationEvent) {
    if (event.is_always_on) return;

    isSaving = true;
    error = null;
    success = null;

    try {
      const response = await fetch('/api/notifications', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event_type: event.event_type,
          is_enabled: !event.enabled
        })
      });

      if (response.ok) {
        const updated = await response.json();
        // Update local state
        const idx = events.findIndex(e => e.event_type === event.event_type);
        if (idx >= 0) {
          events[idx] = { ...events[idx], enabled: updated.is_enabled };
        }
        success = `${event.event_type} ${updated.enabled ? 'enabled' : 'disabled'}`;
        setTimeout(() => success = null, 3000);
      } else {
        const err = await response.json();
        error = err.detail || 'Failed to update notification';
      }
    } catch (e) {
      error = 'Failed to update notification';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  function getCategoryLabel(category: string): string {
    const labels: Record<string, string> = {
      trade: 'Trade Events',
      strategy: 'Strategy Events',
      risk: 'Risk Events',
      system: 'System Events',
      agent: 'Agent Events'
    };
    return labels[category] || category;
  }

  // Load on mount
  import { onMount } from 'svelte';
  onMount(() => {
    loadNotifications();
  });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Notification Settings</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={loadNotifications} title="Refresh">
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert error">
      <AlertCircle size={16} />
      <span>{error}</span>
    </div>
  {/if}

  {#if success}
    <div class="alert success">
      <Check size={16} />
      <span>{success}</span>
    </div>
  {/if}

  {#if isLoading}
    <div class="loading">
      <RefreshCw size={24} class="spinning" />
      <span>Loading notification settings...</span>
    </div>
  {:else}
    <div class="categories">
      {#each categories as category}
        <div class="category-section">
          <h4 class="category-header">{getCategoryLabel(category)}</h4>
          <div class="events-list">
            {#each eventsByCategory[category] || [] as event}
              <div class="event-item" class:disabled={!event.enabled} class:always-on={event.is_always_on}>
                <div class="event-info">
                  <span class="event-name">{event.event_type.replace(/_/g, ' ')}</span>
                  {#if event.description}
                    <span class="event-description">{event.description}</span>
                  {/if}
                </div>
                <div class="event-toggle">
                  {#if event.is_always_on}
                    <div class="always-on-badge">
                      <Lock size={14} />
                      <span>Always On</span>
                    </div>
                  {:else}
                    <label class="switch">
                      <input
                        type="checkbox"
                        checked={event.enabled}
                        onchange={() => toggleNotification(event)}
                        disabled={isSaving}
                      />
                      <span class="slider"></span>
                    </label>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .panel {
    padding: 0;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
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

  .header-actions {
    display: flex;
    gap: 8px;
  }

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

  .icon-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #e8eaf0;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .alert {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .alert.error {
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.25);
    color: #ff3b3b;
  }

  .alert.success {
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    color: #00c896;
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: rgba(255, 255, 255, 0.3);
    gap: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
  }

  .categories {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .category-section {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 16px;
  }

  .category-header {
    margin: 0 0 12px 0;
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .events-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .event-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: rgba(8, 13, 20, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 6px;
    transition: border-color 0.15s;
  }

  .event-item:hover {
    border-color: rgba(255, 255, 255, 0.09);
  }

  .event-item.disabled {
    opacity: 0.5;
  }

  .event-item.always-on {
    background: rgba(0, 212, 255, 0.04);
    border-color: rgba(0, 212, 255, 0.12);
  }

  .event-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .event-name {
    font-size: 12px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.75);
    text-transform: capitalize;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .event-description {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .event-toggle {
    display: flex;
    align-items: center;
  }

  .always-on-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  /* Toggle switch */
  .switch {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
  }

  .switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .slider {
    position: absolute;
    cursor: pointer;
    inset: 0;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.12);
    transition: 0.2s;
    border-radius: 22px;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 14px;
    width: 14px;
    left: 3px;
    bottom: 3px;
    background: rgba(255, 255, 255, 0.5);
    transition: 0.2s;
    border-radius: 50%;
  }

  input:checked + .slider {
    background: rgba(0, 212, 255, 0.25);
    border-color: rgba(0, 212, 255, 0.4);
  }

  input:checked + .slider:before {
    transform: translateX(18px);
    background: #00d4ff;
  }

  input:disabled + .slider {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
