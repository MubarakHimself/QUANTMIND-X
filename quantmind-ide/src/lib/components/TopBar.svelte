<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { onMount } from 'svelte';
  import {
    ShieldAlert,
    Settings,
    Bell,
    MessageSquare,
  } from "lucide-svelte";
  import { activeCanvasStore, CANVASES } from "../stores/canvasStore";
  import KillSwitchModal from "./kill-switch/KillSwitchModal.svelte";
  import EmergencyCloseModal from "./kill-switch/EmergencyCloseModal.svelte";
  import {
    killSwitchState,
    killSwitchCountdown,
    killSwitchFired,
    killSwitchAriaLabel,
    armKillSwitch,
    fetchKillSwitchStatus
  } from "../stores/kill-switch";
  import { unreadCount } from "../stores/notifications";
  import NotificationTray from "./shell/NotificationTray.svelte";

  const dispatch = createEventDispatcher();

  let activeCanvasName = $derived(
    CANVASES.find(c => c.id === $activeCanvasStore)?.name ?? 'Live Trading'
  );

  let showNotifTray = $state(false);

  onMount(() => {
    fetchKillSwitchStatus();
  });

  function handleKillSwitch() {
    if ($killSwitchFired) return;
    if ($killSwitchState === 'ready') {
      armKillSwitch();
    }
  }

  function openSettings() {
    dispatch('openSettings');
  }
</script>

<header class="top-bar">
  <!-- Left: Brand + Canvas Name -->
  <div class="left-section">
    <span class="brand">QMX·ITT</span>
    <div class="divider"></div>
    <span class="canvas-label">{activeCanvasName}</span>
  </div>

  <!-- Action buttons -->
  <div class="action-group">
    <!-- Kill Switch -->
    <button
      class="tb-btn kill-switch"
      class:armed={$killSwitchState === 'armed'}
      class:fired={$killSwitchFired}
      onclick={handleKillSwitch}
      disabled={$killSwitchFired}
      title={$killSwitchAriaLabel}
      aria-label={$killSwitchAriaLabel}
    >
      <ShieldAlert size={11} strokeWidth={2.5} />
      {#if $killSwitchFired}
        <span>FIRED</span>
      {:else if $killSwitchState === 'armed'}
        {#if $killSwitchCountdown > 0}
          <span>{$killSwitchCountdown}s</span>
        {:else}
          <span>ARMED</span>
        {/if}
      {:else}
        <span>KILL</span>
      {/if}
    </button>

    <!-- Copilot → Workshop canvas -->
    <button class="tb-btn copilot" title="Open Copilot (Workshop)" onclick={() => activeCanvasStore.setActiveCanvas('workshop')}>
      <MessageSquare size={11} strokeWidth={2} />
      <span>CPLT</span>
    </button>

    <!-- Notifications -->
    <button
      class="tb-btn notif"
      class:has-unread={$unreadCount > 0}
      title="Notifications"
      onclick={() => showNotifTray = !showNotifTray}
    >
      <div class="bell-wrap">
        <Bell size={11} strokeWidth={2} />
        {#if $unreadCount > 0}
          <span class="notif-dot" aria-hidden="true"></span>
        {/if}
      </div>
      <span>N</span>
    </button>
  </div>

  <div class="spacer"></div>

  <!-- Right: Node health + Settings -->
  <div class="right-section">
    <div class="node-health">
      <span class="nh-node"><span class="nh-dot nh-green"></span>CZ</span>
      <span class="nh-node"><span class="nh-dot nh-green"></span>CN</span>
      <span class="nh-node"><span class="nh-dot nh-amber"></span>LO</span>
    </div>
    <button
      class="tb-btn settings-btn"
      onclick={openSettings}
      title="Settings"
      aria-label="Open settings"
    >
      <Settings size={13} strokeWidth={1.5} />
    </button>
  </div>
</header>

<!-- Kill Switch Modals -->
<KillSwitchModal />
<EmergencyCloseModal />

<!-- Notification Tray (fixed position, outside layout flow) -->
<NotificationTray open={showNotifTray} onClose={() => showNotifTray = false} />

<style>
  .top-bar {
    grid-area: topbar;
    display: flex;
    align-items: center;
    gap: 7px;
    height: var(--header-height);
    padding: 0 14px;
    background: var(--glass-tier-1);
    backdrop-filter: var(--glass-blur);
    -webkit-app-region: drag;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    z-index: 100;
  }

  /* ── Left ── */
  .left-section {
    display: flex;
    align-items: center;
    gap: 8px;
    -webkit-app-region: no-drag;
    flex-shrink: 0;
  }

  .brand {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: var(--color-accent-cyan);
    opacity: 0.9;
  }

  .divider {
    width: 1px;
    height: 16px;
    background: rgba(255, 255, 255, 0.1);
  }

  .canvas-label {
    font-family: var(--font-nav);
    font-weight: 500;
    font-size: 11px;
    color: var(--color-text-secondary);
    letter-spacing: 0.01em;
  }

  /* ── Action group ── */
  .action-group {
    display: flex;
    align-items: center;
    gap: 4px;
    -webkit-app-region: no-drag;
  }

  .spacer {
    flex: 1;
  }

  /* ── Right ── */
  .right-section {
    display: flex;
    align-items: center;
    gap: 8px;
    -webkit-app-region: no-drag;
  }

  /* ── Shared button base ── */
  .tb-btn {
    height: 22px;
    padding: 0 8px;
    border-radius: 3px;
    border: 1px solid;
    font-size: 10px;
    font-family: var(--font-mono);
    display: flex;
    align-items: center;
    gap: 4px;
    background: transparent;
    cursor: pointer;
    white-space: nowrap;
    letter-spacing: 0.02em;
    transition: all 0.12s ease;
  }

  /* Kill switch — amber (ready) */
  .kill-switch {
    border-color: rgba(212, 146, 14, 0.35);
    color: var(--color-accent-amber, #f0a500);
    background: rgba(212, 146, 14, 0.05);
  }

  .kill-switch:hover:not(:disabled) {
    background: rgba(212, 146, 14, 0.1);
    border-color: rgba(212, 146, 14, 0.55);
  }

  /* Kill switch — armed (red pulse) */
  .kill-switch.armed {
    background: var(--color-accent-red, #ff3b3b);
    border-color: var(--color-accent-red, #ff3b3b);
    color: white;
    animation: pulse 2s ease-in-out infinite alternate;
  }

  /* Kill switch — fired */
  .kill-switch.fired {
    background: rgba(107, 114, 128, 0.1);
    border-color: rgba(107, 114, 128, 0.3);
    color: var(--color-text-muted);
    cursor: not-allowed;
    animation: none;
    opacity: 0.6;
  }

  /* Copilot */
  .copilot {
    border-color: rgba(0, 170, 204, 0.25);
    color: var(--color-accent-cyan);
    background: rgba(0, 170, 204, 0.04);
  }

  .copilot:hover {
    background: rgba(0, 170, 204, 0.1);
    border-color: rgba(0, 170, 204, 0.45);
  }

  /* Notifications */
  .notif {
    border-color: rgba(255, 255, 255, 0.1);
    color: var(--color-text-secondary);
    background: rgba(255, 255, 255, 0.02);
  }

  .notif:hover {
    background: rgba(255, 255, 255, 0.06);
  }

  .notif.has-unread {
    border-color: rgba(0, 212, 255, 0.25);
    color: #00d4ff;
  }

  /* Bell icon wrapper for positioning the unread dot */
  .bell-wrap {
    position: relative;
    display: flex;
    align-items: center;
  }

  .notif-dot {
    position: absolute;
    top: -2px;
    right: -2px;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #ff3b3b;
    border: 1px solid rgba(8, 13, 20, 0.9);
    flex-shrink: 0;
  }

  /* Settings */
  .settings-btn {
    width: 24px;
    height: 24px;
    padding: 0;
    border-color: rgba(255, 255, 255, 0.08);
    color: var(--color-text-muted);
    background: transparent;
    justify-content: center;
    border-radius: 3px;
  }

  .settings-btn:hover {
    background: rgba(255, 255, 255, 0.06);
    color: var(--color-text-primary);
  }

  /* ── Node health ── */
  .node-health {
    display: flex;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--color-text-muted);
    align-items: center;
  }

  .nh-node {
    display: flex;
    align-items: center;
    gap: 3px;
  }

  .nh-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .nh-green {
    background: var(--color-accent-green, #00a878);
  }

  .nh-amber {
    background: var(--color-accent-amber, #f0a500);
  }

  @keyframes pulse {
    from { opacity: 1; box-shadow: 0 0 6px var(--color-accent-red); }
    to   { opacity: 0.7; box-shadow: 0 0 16px var(--color-accent-red); }
  }
</style>
