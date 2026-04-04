<script lang="ts">
  /**
   * InterSessionCooldownPanel - Inter-Session Cooldown Status Panel
   *
   * Displays the 3-hour intelligence window between London and NY sessions:
   * - Current session -> next session transition (LONDON -> NEW YORK)
   * - Countdown timer (hours:minutes) to cooldown end (13:00 GMT)
   * - Intelligence window status (ACTIVE / SLEEPING)
   * - Actions blocked indicator (new trades blocked during cooldown)
   * - Progress bar showing cooldown elapsed/remaining
   * - Current step within the 4-step cooldown sequence
   *
   * Color coding:
   * - Amber (#F59E0B): Active cooldown, actions blocked
   * - Green (#10B981): Open trading, no cooldown
   *
   * Polls /api/trading/cooldown/status every 10 seconds.
   */

  import { onMount, onDestroy } from 'svelte';
  import { Clock, BrainCircuit, ShieldAlert, CheckCircle2, Lock, Unlock } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  // =============================================================================
  // Types
  // =============================================================================

  interface CooldownStatus {
    is_active: boolean;
    session_transition: string | null;
    cooldown_end_time: string | null;
    hours_remaining: number;
    minutes_remaining: number;
    current_session: string | null;
    next_session: string | null;
    intelligence_window_active: boolean;
    actions_blocked: boolean;
    progress: number;
    state: string;
    step_name: string | null;
    current_step: number;
    window_start: string | null;
    window_end: string | null;
    ny_roster_locked: boolean;
  }

  // =============================================================================
  // Constants
  // =============================================================================

  const COOLDOWN_STEPS = [
    { step: 1, name: 'London Scoring', time: '10:00-10:30' },
    { step: 2, name: 'Paper Recovery', time: '10:30-11:30' },
    { step: 3, name: 'NY Queue Build', time: '11:30-12:40' },
    { step: 4, name: 'Health Check', time: '12:40-13:00' },
  ];

  const STATUS_COLOR_ACTIVE = '#F59E0B';   // Amber
  const STATUS_COLOR_OPEN = '#10B981';     // Green
  const STATUS_COLOR_INACTIVE = '#6B7280'; // Grey

  // =============================================================================
  // State
  // =============================================================================

  let cooldownStatus = $state<CooldownStatus | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let pollInterval: number;

  // Local countdown tick (updates every second for smooth display)
  let countdownText = $state<string>('--');
  let progressPercent = $state(0);

  // =============================================================================
  // Derived State
  // =============================================================================

  let isActive = $derived(cooldownStatus?.is_active ?? false);
  let isIntelligenceActive = $derived(cooldownStatus?.intelligence_window_active ?? false);
  let areActionsBlocked = $derived(cooldownStatus?.actions_blocked ?? false);
  let currentStep = $derived(cooldownStatus?.current_step ?? 0);
  let stepName = $derived(cooldownStatus?.step_name ?? null);
  let sessionTransition = $derived(cooldownStatus?.session_transition ?? null);
  let nyRosterLocked = $derived(cooldownStatus?.ny_roster_locked ?? false);

  let statusColor = $derived(
    isActive ? STATUS_COLOR_ACTIVE : STATUS_COLOR_OPEN
  );

  let statusLabel = $derived(
    isActive ? 'ACTIVE' : 'OPEN'
  );

  let intelligenceLabel = $derived(
    isIntelligenceActive ? 'ACTIVE' : 'SLEEPING'
  );

  let actionsBlockedLabel = $derived(
    areActionsBlocked ? 'BLOCKED' : 'ALLOWED'
  );

  let cooldownEndDisplay = $derived(() => {
    if (!cooldownStatus?.cooldown_end_time) return '--:-- GMT';
    try {
      const dt = new Date(cooldownStatus.cooldown_end_time);
      return dt.toLocaleTimeString('en-GB', {
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'UTC'
      }) + ' GMT';
    } catch {
      return '--:-- GMT';
    }
  });

  // =============================================================================
  // Data Fetching
  // =============================================================================

  async function fetchCooldownStatus() {
    try {
      const data = await apiFetch<CooldownStatus>('/api/trading/cooldown/status');
      cooldownStatus = data;
      loading = false;
      error = null;
      updateCountdown();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to fetch cooldown status';
      loading = false;
      console.error('[InterSessionCooldownPanel] Failed to fetch cooldown status:', e);
    }
  }

  function updateCountdown() {
    if (!cooldownStatus) {
      countdownText = '--:--';
      progressPercent = 0;
      return;
    }

    if (!cooldownStatus.is_active) {
      countdownText = '--:--';
      progressPercent = 0;
      return;
    }

    const { hours_remaining, minutes_remaining } = cooldownStatus;
    if (hours_remaining > 0) {
      countdownText = `${hours_remaining}h ${minutes_remaining}m`;
    } else if (minutes_remaining > 0) {
      countdownText = `${minutes_remaining}m`;
    } else {
      countdownText = 'COMPLETE';
    }

    progressPercent = Math.round((cooldownStatus.progress ?? 0) * 100);
  }

  // Tick the countdown every second for smooth display
  function startCountdownTicker() {
    setInterval(() => {
      if (cooldownStatus?.is_active && cooldownStatus?.cooldown_end_time) {
        const endTime = new Date(cooldownStatus.cooldown_end_time).getTime();
        const now = Date.now();
        const remaining = endTime - now;

        if (remaining <= 0) {
          countdownText = 'COMPLETE';
          progressPercent = 100;
          return;
        }

        const totalMinutes = Math.floor(remaining / 60000);
        const hours = Math.floor(totalMinutes / 60);
        const minutes = totalMinutes % 60;

        if (hours > 0) {
          countdownText = `${hours}h ${minutes}m`;
        } else {
          countdownText = `${minutes}m`;
        }

        // Recalculate progress from window_start and window_end
        if (cooldownStatus.window_start && cooldownStatus.window_end) {
          const start = new Date(cooldownStatus.window_start).getTime();
          const end = new Date(cooldownStatus.window_end).getTime();
          const elapsed = now - start;
          const total = end - start;
          progressPercent = Math.round(Math.min(100, Math.max(0, (elapsed / total) * 100)));
        }
      }
    }, 1000);
  }

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(() => {
    fetchCooldownStatus();
    pollInterval = window.setInterval(fetchCooldownStatus, 10000); // Poll every 10s
    startCountdownTicker();
  });

  onDestroy(() => {
    if (pollInterval) {
      clearInterval(pollInterval);
    }
  });
</script>

<div class="cooldown-panel" role="status" aria-label="Inter-Session Cooldown Status">
  <!-- Panel Header -->
  <div class="panel-header">
    <div class="header-left">
      <Clock size={14} />
      <span class="header-title">Inter-Session Cooldown</span>
    </div>
    <div class="header-right">
      <!-- Status Badge -->
      <div
        class="status-badge"
        style="background-color: {statusColor}20; border-color: {statusColor};"
      >
        <span class="status-dot" style="background-color: {statusColor};"></span>
        <span class="status-label" style="color: {statusColor};">{statusLabel}</span>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <div class="panel-content">
    {#if loading}
      <div class="loading-state">
        <span class="loading-text">Loading...</span>
      </div>
    {:else if error}
      <div class="error-state">
        <span class="error-text">Using local fallback</span>
      </div>
    {:else}
      <!-- Session Transition -->
      {#if sessionTransition}
        <div class="transition-row">
          <div class="transition-arrow">
            <span class="session-badge london">LONDON</span>
            <span class="arrow-symbol">-&gt;</span>
            <span class="session-badge ny">NEW YORK</span>
          </div>
        </div>
      {:else}
        <div class="transition-row">
          <span class="no-cooldown-text">No active cooldown</span>
        </div>
      {/if}

      <!-- Countdown Timer -->
      <div class="countdown-row">
        <div class="countdown-label">Ends at</div>
        <div class="countdown-value" style="color: {statusColor};">
          {countdownText}
        </div>
        <div class="countdown-end">{cooldownEndDisplay}</div>
      </div>

      <!-- Progress Bar -->
      <div class="progress-section">
        <div class="progress-bar-track">
          <div
            class="progress-bar-fill"
            style="width: {progressPercent}%; background-color: {statusColor};"
          ></div>
        </div>
        <div class="progress-labels">
          <span class="progress-label-left">10:00</span>
          <span class="progress-label-center">Cooldown Progress: {progressPercent}%</span>
          <span class="progress-label-right">13:00</span>
        </div>
      </div>

      <!-- Status Indicators Row -->
      <div class="indicators-row">
        <!-- Intelligence Window -->
        <div class="indicator-card">
          <div class="indicator-icon">
            <BrainCircuit size={14} />
          </div>
          <div class="indicator-content">
            <div class="indicator-label">Intelligence Window</div>
            <div
              class="indicator-value"
              style="color: {isIntelligenceActive ? STATUS_COLOR_ACTIVE : STATUS_COLOR_OPEN};"
            >
              {intelligenceLabel}
            </div>
          </div>
        </div>

        <!-- Actions Blocked -->
        <div class="indicator-card">
          <div class="indicator-icon">
            {#if areActionsBlocked}
              <Lock size={14} />
            {:else}
              <Unlock size={14} />
            {/if}
          </div>
          <div class="indicator-content">
            <div class="indicator-label">New Trades</div>
            <div
              class="indicator-value"
              style="color: {areActionsBlocked ? STATUS_COLOR_ACTIVE : STATUS_COLOR_OPEN};"
            >
              {actionsBlockedLabel}
            </div>
          </div>
        </div>

        <!-- NY Roster Locked -->
        <div class="indicator-card">
          <div class="indicator-icon">
            {#if nyRosterLocked}
              <ShieldAlert size={14} />
            {:else}
              <CheckCircle2 size={14} />
            {/if}
          </div>
          <div class="indicator-content">
            <div class="indicator-label">NY Roster</div>
            <div class="indicator-value" style="color: {nyRosterLocked ? '#F59E0B' : '#10B981'};">
              {nyRosterLocked ? 'LOCKED' : 'OPEN'}
            </div>
          </div>
        </div>
      </div>

      <!-- Current Step -->
      {#if isActive && stepName}
        <div class="step-section">
          <div class="step-header">
            <span class="step-label">Current Step</span>
            <span class="step-number">Step {currentStep} of 4</span>
          </div>
          <div class="step-name" style="color: {statusColor};">{stepName}</div>
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .cooldown-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px;
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(245, 158, 11, 0.15);
    border-radius: 8px;
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    min-width: 280px;
  }

  /* Header */
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #e5e7eb;
  }

  .header-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
  }

  /* Status Badge */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border: 1px solid;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .status-label {
    font-weight: 700;
    text-transform: uppercase;
  }

  /* Content */
  .panel-content {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  /* Loading / Error States */
  .loading-state,
  .error-state {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }

  .loading-text {
    font-size: 11px;
    color: #6b7280;
  }

  .error-text {
    font-size: 11px;
    color: #ef4444;
  }

  /* Session Transition */
  .transition-row {
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .transition-arrow {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .session-badge {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
  }

  .session-badge.london {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #60a5fa;
  }

  .session-badge.ny {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
  }

  .arrow-symbol {
    color: #6b7280;
    font-size: 12px;
  }

  .no-cooldown-text {
    font-size: 11px;
    color: #6b7280;
    text-align: center;
  }

  /* Countdown */
  .countdown-row {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .countdown-label {
    font-size: 9px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .countdown-value {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1;
  }

  .countdown-end {
    font-size: 10px;
    color: #6b7280;
  }

  /* Progress Bar */
  .progress-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .progress-bar-track {
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }

  .progress-labels {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 9px;
    color: #6b7280;
  }

  .progress-label-center {
    font-size: 9px;
    color: #9ca3af;
  }

  /* Indicators Row */
  .indicators-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
  }

  .indicator-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 8px 4px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .indicator-icon {
    color: #6b7280;
  }

  .indicator-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .indicator-label {
    font-size: 8px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    text-align: center;
  }

  .indicator-value {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Step Section */
  .step-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
  }

  .step-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .step-label {
    font-size: 9px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .step-number {
    font-size: 9px;
    color: #9ca3af;
  }

  .step-name {
    font-size: 11px;
    font-weight: 600;
  }
</style>
