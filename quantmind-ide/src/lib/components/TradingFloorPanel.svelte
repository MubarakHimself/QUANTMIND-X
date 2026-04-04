<script lang="ts">
  import { ShieldAlert, AlertTriangle } from 'lucide-svelte';
  import CopilotPanel from './trading-floor/CopilotPanel.svelte';
  import {
    newsKillZoneState,
    newsSessionStatuses,
    newsUpcomingEvents,
    killSwitchLockState
  } from '../stores/kill-switch';

  let activeTab: 'copilot' | 'floor-manager' = $state('floor-manager');
  const tabs = [
    { id: 'floor-manager', label: 'Floor Manager' },
    { id: 'copilot', label: 'QuantMind Copilot' }
  ];

  function selectTab(tabId: string) {
    activeTab = tabId as 'copilot' | 'floor-manager';
  }

  // Track previous state to detect transitions
  let prevKillZoneState = $state<'SAFE' | 'PRE_NEWS' | 'KILL_ZONE' | 'POST_NEWS'>('SAFE');
  let justFired = $state(false);

  $effect(() => {
    const current = $newsKillZoneState;
    if (current === 'KILL_ZONE' && prevKillZoneState !== 'KILL_ZONE') {
      justFired = true;
      setTimeout(() => { justFired = false; }, 3000);
    }
    prevKillZoneState = current;
  });
</script>

<div class="trading-floor-panel">
  {#if $killSwitchLockState?.manual_market_lock_active}
    <div class="lock-banner manual-lock">
      <div class="banner-left">
        <ShieldAlert size={14} />
        <span class="banner-label">MARKET LOCK</span>
        <span class="banner-sep">—</span>
        <span class="banner-text">Manual market lock active.</span>
      </div>
      <div class="banner-right">
        {#if $killSwitchLockState?.reason}
          <span class="banner-event">{$killSwitchLockState.reason}</span>
        {/if}
      </div>
    </div>
  {:else if $killSwitchLockState?.hard_lock_active}
    <div class="lock-banner hard-lock">
      <div class="banner-left">
        <AlertTriangle size={14} />
        <span class="banner-label">ACCOUNT LOCK</span>
        <span class="banner-sep">—</span>
        <span class="banner-text">Global hard lock active.</span>
      </div>
      <div class="banner-right">
        {#if $killSwitchLockState?.pressure_state}
          <span class="banner-event">{$killSwitchLockState.pressure_state}</span>
        {/if}
      </div>
    </div>
  {/if}

  <!-- News Kill Zone Banner -->
  {#if $newsKillZoneState === 'KILL_ZONE'}
    <div class="kill-zone-banner" class:just-fired={justFired}>
      <div class="banner-left">
        <AlertTriangle size={14} />
        <span class="banner-label">KILL ZONE</span>
        <span class="banner-sep">—</span>
        <span class="banner-text">High-impact news active. Trading halted.</span>
      </div>
      <div class="banner-right">
        {#if $newsUpcomingEvents.length > 0}
          <span class="banner-event">{$newsUpcomingEvents[0].title}</span>
        {/if}
      </div>
    </div>
  {:else if $newsKillZoneState === 'PRE_NEWS'}
    <div class="kill-zone-banner pre-news">
      <div class="banner-left">
        <ShieldAlert size={14} />
        <span class="banner-label">PRE-NEWS</span>
        <span class="banner-sep">—</span>
        <span class="banner-text">News event within 30 min. Caution advised.</span>
      </div>
      <div class="banner-right">
        {#if $newsUpcomingEvents.length > 0}
          <span class="banner-event">{$newsUpcomingEvents[0].title}</span>
        {/if}
      </div>
    </div>
  {/if}

  <div class="panel-tabs">
    {#each tabs as tab}
      <button
        class="tab-btn"
        class:active={activeTab === tab.id}
        onclick={() => selectTab(tab.id)}
      >
        {tab.label}
      </button>
    {/each}
  </div>

  <div class="panel-content">
    {#if activeTab === 'copilot'}
      <div class="copilot-chat">
        <CopilotPanel />
      </div>
    {:else}
      <div class="floor-manager-panel">
        <CopilotPanel />
      </div>
    {/if}
  </div>
</div>

<style>
  .trading-floor-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary, #121212);
    overflow: hidden;
  }

  .panel-tabs {
    display: flex;
    flex-shrink: 0;
    border-bottom: 1px solid var(--border-color, #333);
  }

  .tab-btn {
    flex: 1;
    padding: 10px 8px;
    background: transparent;
    border: none;
    color: var(--text-secondary, #888);
    cursor: pointer;
    transition: all 0.2s;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .tab-btn:hover {
    color: var(--text-primary, #e0e0e0);
    background: var(--bg-hover, #1a1a1a);
  }

  .tab-btn.active {
    color: var(--accent-color, #4a9eff);
    border-bottom: 2px solid var(--accent-color, #4a9eff);
  }

  .panel-content {
    flex: 1;
    overflow: hidden;
    min-height: 0;
  }

  .copilot-chat,
  .floor-manager-panel {
    height: 100%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .copilot-chat :global(*),
  .floor-manager-panel :global(*) {
    overflow: hidden !important;
  }

  /* Kill Zone Banner */
  .kill-zone-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    font-size: 11px;
    font-family: var(--font-mono);
    flex-shrink: 0;
  }

  .lock-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    font-size: 11px;
    font-family: var(--font-mono);
    flex-shrink: 0;
  }

  .kill-zone-banner {
    background: rgba(239, 68, 68, 0.15);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    animation: killZoneFlash 1s ease-in-out infinite alternate;
  }

  .lock-banner.manual-lock {
    background: rgba(245, 158, 11, 0.14);
    border-bottom: 1px solid rgba(245, 158, 11, 0.28);
  }

  .lock-banner.hard-lock {
    background: rgba(220, 38, 38, 0.18);
    border-bottom: 1px solid rgba(220, 38, 38, 0.35);
  }

  .kill-zone-banner.just-fired {
    animation: killZoneFire 0.5s ease-out 3;
  }

  .kill-zone-banner.pre-news {
    background: rgba(240, 165, 0, 0.1);
    border-bottom-color: rgba(240, 165, 0, 0.25);
    color: #f0a500;
    animation: none;
  }

  .banner-left {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #ff3b3b;
  }

  .pre-news .banner-left {
    color: #f0a500;
  }

  .manual-lock .banner-left {
    color: #f59e0b;
  }

  .hard-lock .banner-left {
    color: #ff6464;
  }

  .banner-label {
    font-weight: 700;
    letter-spacing: 0.05em;
  }

  .banner-sep {
    opacity: 0.5;
  }

  .banner-text {
    color: rgba(255, 255, 255, 0.8);
  }

  .banner-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .banner-event {
    color: rgba(255, 255, 255, 0.6);
    font-size: 10px;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  @keyframes killZoneFlash {
    from { background: rgba(239, 68, 68, 0.15); }
    to   { background: rgba(239, 68, 68, 0.25); }
  }

  @keyframes killZoneFire {
    0%   { background: rgba(239, 68, 68, 0.5); transform: scaleY(1.05); }
    100% { background: rgba(239, 68, 68, 0.15); transform: scaleY(1); }
  }
</style>
