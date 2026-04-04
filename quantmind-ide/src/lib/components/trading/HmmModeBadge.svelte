<script lang="ts">
  /**
   * HmmModeBadge - HMM Mode Indicator Pill for Live Trading Header
   *
   * Shows the current HMM mode state:
   * - NORMAL MODE (green): default state
   * - HOUSE MONEY (yellow): when is_house_money_active is true
   * - PRESERVATION (red): when is_preservation_mode is true
   *
   * Data from /api/trading/session-risk-state and /api/hmm/status
   */

  import { onMount } from 'svelte';
  import { Circle } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  // =============================================================================
  // Types
  // =============================================================================

  type HmmMode = 'normal' | 'house_money' | 'preservation';

  interface SessionRiskState {
    is_preservation_mode: boolean;
    is_house_money_active: boolean;
    hmm_multiplier: number;
  }

  interface HmmStatusResponse {
    model_loaded: boolean;
    model_version: string | null;
    deployment_mode: string;
    hmm_weight: number;
    shadow_mode_active: boolean;
  }

  // =============================================================================
  // State
  // =============================================================================

  let hmmMode = $state<HmmMode>('normal');
  let hmmWeight = $state(0);
  let modelLoaded = $state(false);
  let deploymentMode = $state('ising_only');
  let fetchError = $state<string | null>(null);

  // =============================================================================
  // Derived State
  // =============================================================================

  let dotColor = $derived(
    hmmMode === 'preservation' ? '#EF4444' :
    hmmMode === 'house_money' ? '#F59E0B' :
    '#10B981'
  );

  let modeLabel = $derived(
    hmmMode === 'preservation' ? 'PRESERVATION' :
    hmmMode === 'house_money' ? 'HOUSE MONEY' :
    'NORMAL MODE'
  );

  let weightDisplay = $derived(
    modelLoaded ? `${(hmmWeight * 100).toFixed(0)}%` : null
  );

  // =============================================================================
  // Data Fetching
  // =============================================================================

  async function fetchHmmData() {
    try {
      // Fetch both endpoints in parallel
      const [riskState, hmmStatus] = await Promise.allSettled([
        apiFetch<SessionRiskState>('/api/trading/session-risk-state'),
        apiFetch<HmmStatusResponse>('/api/hmm/status')
      ]);

      // Process session risk state
      if (riskState.status === 'fulfilled') {
        const state = riskState.value;
        if (state.is_preservation_mode) {
          hmmMode = 'preservation';
        } else if (state.is_house_money_active) {
          hmmMode = 'house_money';
        } else {
          hmmMode = 'normal';
        }
      }

      // Process HMM status for weight indicator
      if (hmmStatus.status === 'fulfilled') {
        const status = hmmStatus.value;
        modelLoaded = status.model_loaded;
        hmmWeight = status.hmm_weight;
        deploymentMode = status.deployment_mode;
      }

      fetchError = null;
    } catch (err) {
      // Silently handle errors - badge will show default state
      fetchError = null; // Don't show errors in badge UI
    }
  }

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(() => {
    fetchHmmData();
    const interval = setInterval(fetchHmmData, 10000); // Poll every 10s
    return () => clearInterval(interval);
  });
</script>

<div
  class="hmm-badge"
  role="status"
  aria-label="HMM mode: {modeLabel}"
>
  <!-- Colored dot indicator -->
  <span class="dot" style="background-color: {dotColor};"></span>

  <!-- Mode label -->
  <span class="label">{modeLabel}</span>

  <!-- Optional weight indicator -->
  {#if weightDisplay && hmmMode !== 'normal'}
    <span class="weight">{weightDisplay}</span>
  {/if}
</div>

<style>
  .hmm-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 9999px;
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #E5E7EB;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .label {
    text-transform: uppercase;
  }

  .weight {
    padding-left: 4px;
    margin-left: 4px;
    border-left: 1px solid rgba(255, 255, 255, 0.2);
    color: #9CA3AF;
    font-size: 10px;
    font-weight: 500;
  }
</style>
