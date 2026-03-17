<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { onMount } from 'svelte';
  import {
    ShieldAlert,
    Settings,
    Bell,
    Wrench,
    X,
    AlertTriangle
  } from "lucide-svelte";
  import { navigationStore } from "../stores/navigationStore";

  const dispatch = createEventDispatcher();

  let killSwitchArmed = $state(false);
  let showConfirmModal = $state(false);
  let activeCanvasName = $state("Live Trading");

  // Subscribe to navigation store to get current canvas name
  onMount(() => {
    const unsubscribe = navigationStore.subscribe((state) => {
      // Map view IDs to readable canvas names
      const canvasNames: Record<string, string> = {
        'workshop': 'Workshop',
        'knowledge': 'Knowledge Hub',
        'assets': 'Shared Assets',
        'ea': 'EA Management',
        'backtest': 'Backtests',
        'paper-trading': 'Paper Trading',
        'live': 'Live Trading',
        'router': 'Strategy Router',
        'hmm': 'HMM Dashboard'
      };
      activeCanvasName = canvasNames[state.currentView] || 'Live Trading';
    });
    return unsubscribe;
  });

  function handleKillSwitch() {
    if (!killSwitchArmed) {
      // First click arms it
      killSwitchArmed = true;
      // Auto-disarm after 5 seconds
      setTimeout(() => {
        if (killSwitchArmed) {
          killSwitchArmed = false;
        }
      }, 5000);
    } else {
      // Second click shows confirmation modal
      showConfirmModal = true;
    }
  }

  async function confirmKill() {
    try {
      await fetch('http://localhost:8000/api/trading/kill', { method: 'POST' });
      dispatch('killTriggered');
    } catch (e) {
      console.error('Kill switch failed:', e);
    }
    killSwitchArmed = false;
    showConfirmModal = false;
  }

  function cancelKill() {
    killSwitchArmed = false;
    showConfirmModal = false;
  }

  function openSettings() {
    dispatch('openSettings');
  }

  function openWorkshop() {
    navigationStore.navigateToView('workshop', 'Workshop');
  }
</script>

<header class="top-bar">
  <!-- Left Section: Wordmark + Canvas Name -->
  <div class="left-section">
    <div class="wordmark">
      <span class="wordmark-text">QUANTMINDX</span>
    </div>
    <div class="divider"></div>
    <div class="canvas-name">
      <span class="canvas-label">{activeCanvasName}</span>
    </div>
  </div>

  <!-- Right Section: Action Buttons -->
  <div class="right-section">
    <!-- Kill Switch -->
    <button
      class="action-btn kill-switch"
      class:armed={killSwitchArmed}
      onclick={handleKillSwitch}
      title={killSwitchArmed ? 'Click again to confirm' : 'Emergency Kill Switch'}
    >
      <ShieldAlert size={16} strokeWidth={2.5} />
      {#if killSwitchArmed}
        <span class="btn-label armed">ARMED</span>
      {:else}
        <span class="btn-label">Kill</span>
      {/if}
    </button>

    <!-- Workshop Button -->
    <button
      class="action-btn"
      onclick={openWorkshop}
      title="Workshop"
    >
      <Wrench size={18} strokeWidth={1.5} />
    </button>

    <!-- Notifications -->
    <button
      class="action-btn"
      title="Notifications"
    >
      <Bell size={18} strokeWidth={1.5} />
    </button>

    <!-- Settings -->
    <button
      class="action-btn"
      onclick={openSettings}
      title="Settings"
    >
      <Settings size={18} strokeWidth={1.5} />
    </button>
  </div>
</header>

<!-- Kill Switch Confirmation Modal -->
{#if showConfirmModal}
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
  <div class="modal-overlay" onclick={cancelKill} role="dialog" aria-modal="true" aria-labelledby="kill-modal-title">
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
    <div class="modal-content" onclick={(e) => e.stopPropagation()}>
      <div class="modal-icon">
        <AlertTriangle size={32} />
      </div>
      <h3 id="kill-modal-title">Confirm Emergency Kill</h3>
      <p>This will immediately close all active positions and stop all trading bots.</p>
      <div class="modal-actions">
        <button class="btn-cancel" onclick={cancelKill}>Cancel</button>
        <button class="btn-confirm" onclick={confirmKill}>KILL ALL</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .top-bar {
    grid-area: topbar;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: var(--header-height);
    padding: 0 16px;
    background: var(--glass-tier-1);
    backdrop-filter: var(--glass-blur);
    -webkit-app-region: drag;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    z-index: 100;
  }

  .left-section {
    display: flex;
    align-items: center;
    gap: 12px;
    -webkit-app-region: no-drag;
  }

  .wordmark {
    display: flex;
    align-items: center;
  }

  .wordmark-text {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 15px;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, var(--color-accent-cyan) 0%, #00ffcc 50%, var(--color-accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .divider {
    width: 1px;
    height: 20px;
    background: rgba(255, 255, 255, 0.12);
  }

  .canvas-name {
    display: flex;
    align-items: center;
  }

  .canvas-label {
    font-family: var(--font-nav);
    font-weight: 500;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .right-section {
    display: flex;
    align-items: center;
    gap: 8px;
    -webkit-app-region: no-drag;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    height: 32px;
    padding: 0 10px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .action-btn:hover {
    background: var(--glass-tier-2);
    color: var(--text-primary);
  }

  .btn-label {
    font-family: var(--font-nav);
    font-size: 11px;
    font-weight: 500;
  }

  /* Kill Switch Styles */
  .kill-switch {
    background: rgba(255, 59, 59, 0.08);
    border-color: rgba(255, 59, 59, 0.25);
    color: var(--color-danger);
  }

  .kill-switch:hover {
    background: rgba(255, 59, 59, 0.15);
    border-color: rgba(255, 59, 59, 0.4);
  }

  .kill-switch.armed {
    background: var(--color-danger);
    border-color: var(--color-danger);
    color: white;
    animation: pulse 2s ease-in-out infinite alternate;
  }

  .kill-switch.armed .btn-label.armed {
    font-weight: 700;
    letter-spacing: 0.05em;
  }

  @keyframes pulse {
    from { opacity: 1; box-shadow: 0 0 8px var(--color-danger); }
    to { opacity: 0.7; box-shadow: 0 0 20px var(--color-danger); }
  }

  /* Modal Styles */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: var(--bg-secondary);
    border: 1px solid var(--color-danger);
    border-radius: 12px;
    padding: 24px;
    max-width: 400px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255, 59, 59, 0.2);
  }

  .modal-icon {
    display: flex;
    justify-content: center;
    margin-bottom: 16px;
    color: var(--color-danger);
  }

  .modal-content h3 {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 18px;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .modal-content p {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
  }

  .modal-actions {
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .btn-cancel {
    padding: 10px 20px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-family: var(--font-nav);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-cancel:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }

  .btn-confirm {
    padding: 10px 20px;
    background: var(--color-danger);
    border: none;
    border-radius: 6px;
    color: white;
    font-family: var(--font-nav);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-confirm:hover {
    background: #ff5555;
    box-shadow: 0 0 16px rgba(255, 59, 59, 0.4);
  }
</style>