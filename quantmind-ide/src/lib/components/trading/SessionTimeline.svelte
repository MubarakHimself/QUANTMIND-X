<script lang="ts">
  /**
   * SessionTimeline - Horizontal Timeline Bar for Canonical Session Windows
   *
   * Shows all 10 canonical session windows with:
   * - Current window highlighted (pulsing indicator)
   * - Time until next session boundary (countdown)
   * - Premium sessions with gold accent (Tokyo-London, London Open, London-NY)
   * - Dead Zone in muted grey
   * - Current Tilt state displayed with colour coding
   * - Regime Persistence Timer countdown badge
   *
   * Polls /api/trading/current-session for DST-aware session detection.
   * Polls /api/trading/tilt/regime-timer for regime timer display.
   */

  import { onMount, onDestroy } from 'svelte';
  import { Clock, AlertTriangle, Timer } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';
  import { fetchRegimeTimer, type RegimeTimer } from '$lib/stores/trading';

  // =============================================================================
  // Types
  // =============================================================================

  interface CanonicalSession {
    id: string;
    label: string;
    utcStart: string; // "HH:MM" in UTC
    utcEnd: string;   // "HH:MM" in UTC
    isPremium: boolean;
    isDeadZone: boolean;
  }

  type TiltPhase = 'IDLE' | 'LOCK' | 'WAIT' | 'RE_RANK' | 'ACTIVATE';

  interface TiltPhaseEvent {
    phase: TiltPhase;
    state: string;
    closing_session: string;
    incoming_session: string;
    regime_persistence_timer: number;
    timestamp_utc: string;
  }

  interface CurrentSessionResponse {
    current_window: string | null;
    current_window_start: string | null;
    current_window_end: string | null;
    next_window: string | null;
    next_window_start: string | null;
    minutes_until_next: number | null;
    is_premium: boolean;
    is_dead_zone: boolean;
    tilt_state: unknown;
  }

  // =============================================================================
  // Constants
  // =============================================================================

  const CANONICAL_SESSIONS: CanonicalSession[] = [
    { id: 'SYDNEY_OPEN',           label: 'Sydney',     utcStart: '21:00', utcEnd: '23:00', isPremium: false, isDeadZone: false },
    { id: 'SYDNEY_TOKYO_OVERLAP',  label: 'Sydney-Tokyo', utcStart: '23:00', utcEnd: '00:00', isPremium: false, isDeadZone: false },
    { id: 'TOKYO_OPEN',            label: 'Tokyo',      utcStart: '00:00', utcEnd: '03:00', isPremium: false, isDeadZone: false },
    { id: 'TOKYO_LONDON_OVERLAP',  label: 'Tokyo-London', utcStart: '07:00', utcEnd: '09:00', isPremium: true,  isDeadZone: false },
    { id: 'LONDON_OPEN',           label: 'London',     utcStart: '08:00', utcEnd: '10:30', isPremium: true,  isDeadZone: false },
    { id: 'LONDON_MID',            label: 'London Mid', utcStart: '10:30', utcEnd: '12:00', isPremium: false, isDeadZone: false },
    { id: 'INTER_SESSION_COOLDOWN',label: 'Cooldown',   utcStart: '12:00', utcEnd: '13:00', isPremium: false, isDeadZone: false },
    { id: 'LONDON_NY_OVERLAP',     label: 'London-NY',  utcStart: '13:00', utcEnd: '16:00', isPremium: true,  isDeadZone: false },
    { id: 'NY_WIND_DOWN',          label: 'NY Wind',    utcStart: '16:00', utcEnd: '20:00', isPremium: false, isDeadZone: false },
    { id: 'DEAD_ZONE',             label: 'Dead Zone',  utcStart: '20:00', utcEnd: '21:00', isPremium: false, isDeadZone: true  },
  ];

  const TILT_PHASE_COLORS: Record<string, string> = {
    IDLE:     '#6B7280', // muted grey
    LOCK:     '#F59E0B', // amber
    WAIT:     '#3B82F6', // blue
    RE_RANK:  '#8B5CF6', // purple
    ACTIVATE: '#10B981', // green
    SUSPENDED: '#EF4444', // red for suspended
  };

  // =============================================================================
  // State
  // =============================================================================

  // Backend-driven session state (populated by polling /api/trading/current-session)
  let currentWindowName = $state<string | null>(null);
  let nextWindowName = $state<string | null>(null);
  let minutesUntilNext = $state<number>(0);
  let tiltState = $state<TiltPhaseEvent | null>(null);
  let wsConnected = $state(false);
  let wsError = $state<string | null>(null);
  let ws: WebSocket | null = null;
  let pollInterval: number;
  let pollError = $state<string | null>(null);

  // Regime timer state (populated by polling /api/trading/tilt/regime-timer)
  let regimeTimer = $state<RegimeTimer | null>(null);
  let regimePollInterval: number;

  // Fallback: local time when backend is unavailable
  let currentTime = $state(new Date());
  let localFallback = $state(false);

  // =============================================================================
  // Derived State
  // =============================================================================

  /**
   * Find the next session using index-based lookup (F16 fix)
   */
  function getNextSession(currentWindowId: string): CanonicalSession {
    const idx = CANONICAL_SESSIONS.findIndex(s => s.id === currentWindowId);
    if (idx === -1) return CANONICAL_SESSIONS[0];
    const nextIdx = (idx + 1) % CANONICAL_SESSIONS.length;
    return CANONICAL_SESSIONS[nextIdx];
  }

  /**
   * Format countdown as "2h 34m"
   */
  let countdownText = $derived(() => {
    const totalMinutes = minutesUntilNext;
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  });

  // =============================================================================
  // Backend Polling (F12: DST-aware session detection)
  // =============================================================================

  async function pollSessionState() {
    try {
      const data = await apiFetch<CurrentSessionResponse>('/api/trading/current-session');
      currentWindowName = data.current_window;
      nextWindowName = data.next_window;
      minutesUntilNext = data.minutes_until_next ?? 0;
      localFallback = false;
      pollError = null;
    } catch (e) {
      // Fall back to local computation if backend is unavailable
      localFallback = true;
      pollError = 'Using local fallback';
      currentTime = new Date();
    }
  }

  // Tilt state polling (10-second interval for Story 16.1)
  // Complements WebSocket with HTTP fallback
  let tiltPollInterval: number;

  async function pollTiltState() {
    try {
      const data = await apiFetch<TiltStatusResponse>('/api/trading/tilt/status');
      tiltState = {
        phase: data.phase as TiltPhase,
        state: data.state,
        closing_session: data.session_name || '',
        incoming_session: data.regime_context?.split(' -> ')[1] || '',
        regime_persistence_timer: data.next_transition || 0,
        timestamp_utc: new Date().toISOString(),
      };
    } catch (e) {
      // Tilt polling failure is non-critical - WebSocket is primary source
      console.debug('[SessionTimeline] Tilt poll failed:', e);
    }
  }

  async function pollRegimeTimer() {
    try {
      const data = await apiFetch<RegimeTimer>('/api/trading/tilt/regime-timer');
      regimeTimer = data;
    } catch (e) {
      // Regime timer polling failure is non-critical
      console.debug('[SessionTimeline] Regime timer poll failed:', e);
    }
  }

  interface TiltStatusResponse {
    phase: string;
    time_in_phase_seconds: number;
    next_transition: number | null;
    regime_context: string | null;
    session_name: string | null;
    state: string;
  }

  // =============================================================================
  // Local Fallback Computation (used when backend is unavailable)
  // =============================================================================

  function parseTimeToMinutes(timeStr: string): number {
    const [hours, minutes] = timeStr.split(':').map(Number);
    return hours * 60 + minutes;
  }

  function getCurrentUTCMinutes(): number {
    const now = currentTime;
    return now.getUTCHours() * 60 + now.getUTCMinutes();
  }

  function isSessionActive(session: CanonicalSession): boolean {
    const currentMinutes = getCurrentUTCMinutes();
    const startMinutes = parseTimeToMinutes(session.utcStart);
    const endMinutes = parseTimeToMinutes(session.utcEnd);

    if (startMinutes > endMinutes) {
      return currentMinutes >= startMinutes || currentMinutes < endMinutes;
    }

    return currentMinutes >= startMinutes && currentMinutes < endMinutes;
  }

  let localCurrentSession = $derived(
    CANONICAL_SESSIONS.find(session => isSessionActive(session)) || CANONICAL_SESSIONS[CANONICAL_SESSIONS.length - 1]
  );

  /**
   * Find the currently active session by ID (backend-driven with local fallback)
   */
  let currentSession = $derived(
    CANONICAL_SESSIONS.find((session) => session.id === currentWindowName) ||
    (localFallback ? localCurrentSession : CANONICAL_SESSIONS[CANONICAL_SESSIONS.length - 1])
  );

  let nextSession = $derived(
    getNextSession(currentSession?.id || CANONICAL_SESSIONS[0].id)
  );

  /**
   * Get the tilt phase color
   */
  let tiltPhaseColor = $derived(() => {
    if (!tiltState) return TILT_PHASE_COLORS.IDLE;
    return TILT_PHASE_COLORS[tiltState.phase] || TILT_PHASE_COLORS.IDLE;
  });

  /**
   * Get tilt phase display text
   */
  let tiltPhaseText = $derived(() => {
    if (!tiltState) return 'IDLE';
    return tiltState.phase;
  });

  // =============================================================================
  // Regime Timer Derived State
  // =============================================================================

  /**
   * Format regime timer as "2m 30s" or just seconds
   */
  let regimeTimerText = $derived(() => {
    if (!regimeTimer || regimeTimer.regime_timer_seconds <= 0) return null;
    const seconds = regimeTimer.regime_timer_seconds;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    }
    return `${remainingSeconds}s`;
  });

  /**
   * Regime timer badge color: amber when running, green when expired/at action
   */
  let regimeTimerColor = $derived(() => {
    if (!regimeTimer) return '#6B7280'; // muted grey
    if (regimeTimer.regime_timer_seconds > 0) {
      return '#F59E0B'; // amber when counting down
    }
    return '#10B981'; // green when expired/at action
  });

  /**
   * Check if regime timer should be displayed
   */
  let showRegimeTimer = $derived(() => {
    return regimeTimer && (regimeTimer.regime_timer_seconds > 0 || regimeTimer.action_pending);
  });

  // =============================================================================
  // WebSocket Connection for Tilt Phase
  // =============================================================================

  function connectTiltWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/trading`;

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsConnected = true;
        wsError = null;

        // Subscribe to tilt:phase channel via the trading WS
        // The backend should relay Redis pub/sub tilt:phase messages
        ws?.send(JSON.stringify({
          type: 'subscribe',
          topic: 'tilt:phase'
        }));
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWSMessage(message);
        } catch {
          // Ignore parse errors
        }
      };

      ws.onerror = () => {
        wsError = 'WebSocket error';
      };

      ws.onclose = () => {
        wsConnected = false;
        // Reconnect after delay
        setTimeout(connectTiltWS, 5000);
      };
    } catch {
      wsError = 'Failed to connect';
      setTimeout(connectTiltWS, 5000);
    }
  }

  function handleWSMessage(message: Record<string, unknown>) {
    // Handle tilt:phase messages relayed from Redis pub/sub
    if (message.type === 'tilt_phase' || message.topic === 'tilt:phase') {
      tiltState = {
        phase: (message.phase as TiltPhase) || 'IDLE',
        state: (message.state as string) || 'idle',
        closing_session: (message.closing_session as string) || '',
        incoming_session: (message.incoming_session as string) || '',
        regime_persistence_timer: (message.regime_persistence_timer as number) || 0,
        timestamp_utc: (message.timestamp_utc as string) || new Date().toISOString(),
      };
    }

    // Also handle direct phase updates from the trading WS
    if (message.type === 'phase_update' && message.data) {
      const data = message.data as Record<string, unknown>;
      tiltState = {
        phase: (data.phase as TiltPhase) || 'IDLE',
        state: (data.state as string) || 'idle',
        closing_session: (data.closing_session as string) || '',
        incoming_session: (data.incoming_session as string) || '',
        regime_persistence_timer: (data.regime_persistence_timer as number) || 0,
        timestamp_utc: (data.timestamp_utc as string) || new Date().toISOString(),
      };
    }
  }

  function disconnectWS() {
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(() => {
    // Poll backend for session state every minute (F12)
    pollSessionState();
    pollInterval = window.setInterval(pollSessionState, 60000);

    // Also update local fallback time every minute
    window.setInterval(() => {
      if (localFallback) {
        currentTime = new Date();
      }
    }, 60000);

    // Poll tilt state every 10 seconds (Story 16.1 - HTTP fallback for WebSocket)
    pollTiltState();
    tiltPollInterval = window.setInterval(pollTiltState, 10000);

    // Poll regime timer every 10 seconds
    pollRegimeTimer();
    regimePollInterval = window.setInterval(pollRegimeTimer, 10000);

    // Connect to WebSocket for tilt phase updates (primary source)
    connectTiltWS();
  });

  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
    if (tiltPollInterval) clearInterval(tiltPollInterval);
    if (regimePollInterval) clearInterval(regimePollInterval);
    disconnectWS();
  });
</script>

<div class="session-timeline">
  <!-- Timeline Header -->
  <div class="timeline-header">
    <div class="header-left">
      <Clock size={14} />
      <span class="header-title">Trading Sessions</span>
    </div>
    <div class="header-right">
      <span class="countdown-label">Next: {nextSession.label} in {countdownText}</span>
    </div>
  </div>

  <!-- Horizontal Timeline Bar -->
  <div class="timeline-bar">
    {#each CANONICAL_SESSIONS as session, index}
      {@const isActive = session.id === currentSession.id}
      {@const isPrevActive = index > 0 && CANONICAL_SESSIONS[index - 1].id === currentSession.id}

      <div
        class="session-segment"
        class:active={isActive}
        class:premium={session.isPremium}
        class:dead-zone={session.isDeadZone}
        class:transitioning={isActive && tiltState && tiltState.phase !== 'IDLE'}
        title="{session.label}: {session.utcStart}-{session.utcEnd} UTC"
      >
        <!-- Pulsing indicator for active session -->
        {#if isActive}
          <div class="pulse-indicator"></div>
        {/if}

        <!-- Session label -->
        <span class="session-label">{session.label}</span>

        <!-- Premium gold accent -->
        {#if session.isPremium}
          <div class="premium-accent"></div>
        {/if}

        <!-- Dead zone indicator -->
        {#if session.isDeadZone}
          <div class="dead-zone-badge">
            <AlertTriangle size={8} />
            <span>No Trading</span>
          </div>
        {/if}
      </div>
    {/each}
  </div>

  <!-- Tilt State Badge -->
  <div class="tilt-state-bar">
    <div class="tilt-label">Tilt</div>
    <div
      class="tilt-badge"
      style="background-color: {tiltPhaseColor}20; border-color: {tiltPhaseColor};"
    >
      <span class="tilt-dot" style="background-color: {tiltPhaseColor};"></span>
      <span class="tilt-phase" style="color: {tiltPhaseColor};">{tiltPhaseText}</span>
    </div>

    {#if tiltState && tiltState.phase !== 'IDLE'}
      <div class="tilt-transition-info">
        <span class="transition-label">
          {tiltState.closing_session} <span class="arrow">-></span> {tiltState.incoming_session}
        </span>
      </div>
    {/if}

    <!-- Regime Persistence Timer Badge -->
    {#if showRegimeTimer() && regimeTimer}
      <div
        class="regime-timer-badge"
        style="background-color: {regimeTimerColor()}20; border-color: {regimeTimerColor()};"
      >
        <Timer size={12} style="color: {regimeTimerColor()};" />
        <span class="regime-timer-countdown" style="color: {regimeTimerColor()};">
          {regimeTimerText() || '0s'}
        </span>
        {#if regimeTimer.regime_name}
          <span class="regime-name" style="color: {regimeTimerColor()};">
            {regimeTimer.regime_name}
          </span>
        {/if}
        {#if regimeTimer.next_action}
          <span class="regime-action">
            {regimeTimer.next_action}
          </span>
        {/if}
      </div>
    {/if}

    <!-- Connection status -->
    <div class="connection-status" class:connected={wsConnected} class:error={wsError}>
      <span class="status-dot"></span>
      <span>{wsConnected ? 'Live' : wsError ? 'Error' : 'Connecting...'}</span>
    </div>
  </div>
</div>

<style>
  .session-timeline {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
  }

  /* Header */
  .timeline-header {
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
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 12px;
    font-weight: 600;
  }

  .header-right {
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: #6b7280;
  }

  .countdown-label {
    color: #00d4ff;
  }

  /* Timeline Bar */
  .timeline-bar {
    display: flex;
    gap: 2px;
    overflow-x: auto;
    padding: 4px 0;
  }

  .session-segment {
    flex: 1;
    min-width: 60px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 8px 4px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    position: relative;
    transition: all 0.2s ease;
  }

  .session-segment.active {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.3);
  }

  .session-segment.premium {
    border-top: 2px solid #fbbf24;
  }

  .session-segment.premium.active {
    background: rgba(251, 191, 36, 0.15);
    border-color: rgba(251, 191, 36, 0.4);
  }

  .session-segment.dead-zone {
    background: rgba(107, 114, 128, 0.2);
    border: 1px solid rgba(107, 114, 128, 0.3);
  }

  .session-segment.transitioning {
    animation: tilt-pulse 1.5s ease-in-out infinite;
  }

  @keyframes tilt-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  /* Pulse indicator */
  .pulse-indicator {
    position: absolute;
    top: 4px;
    right: 4px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00d4ff;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.3);
      opacity: 0.7;
    }
  }

  /* Session label */
  .session-label {
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 9px;
    color: #9ca3af;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
  }

  .session-segment.active .session-label {
    color: #e5e7eb;
    font-weight: 600;
  }

  /* Premium accent */
  .premium-accent {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #fbbf24, #f59e0b);
  }

  /* Dead zone badge */
  .dead-zone-badge {
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 2px 4px;
    background: rgba(107, 114, 128, 0.3);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 7px;
    color: #9ca3af;
    text-transform: uppercase;
  }

  /* Tilt State Bar */
  .tilt-state-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
  }

  .tilt-label {
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 10px;
    color: #6b7280;
    text-transform: uppercase;
  }

  .tilt-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    border: 1px solid;
    border-radius: 12px;
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 11px;
    font-weight: 600;
  }

  .tilt-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .tilt-phase {
    font-weight: 700;
  }

  .tilt-transition-info {
    flex: 1;
    display: flex;
    justify-content: center;
  }

  .transition-label {
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 10px;
    color: #9ca3af;
  }

  .arrow {
    color: #6b7280;
    margin: 0 4px;
  }

  /* Regime Timer Badge */
  .regime-timer-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border: 1px solid;
    border-radius: 12px;
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 11px;
  }

  .regime-timer-countdown {
    font-weight: 700;
    font-size: 12px;
  }

  .regime-name {
    font-weight: 600;
    font-size: 10px;
    opacity: 0.9;
  }

  .regime-action {
    color: #9ca3af;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* Connection status */
  .connection-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 9px;
    color: #6b7280;
    margin-left: auto;
  }

  .connection-status .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6b7280;
  }

  .connection-status.connected .status-dot {
    background: #10b981;
  }

  .connection-status.error .status-dot {
    background: #ef4444;
  }

  .connection-status.connected {
    color: #10b981;
  }

  .connection-status.error {
    color: #ef4444;
  }
</style>
