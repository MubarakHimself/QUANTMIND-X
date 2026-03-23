<script lang="ts">
  /**
   * EAPerformanceTile — Enhancement Loop pipeline stage counts.
   * AC 12-4-5: all-zero state shows "0" in neutral grey — not an error
   * AC 12-4-6: populated state shows counts with --color-accent-cyan
   * Story 12-4
   */
  import { onMount } from 'svelte';
  import TileCard from '$lib/components/shared/TileCard.svelte';
  import { apiFetch } from '$lib/api';

  interface PipelineRun {
    id: string;
    current_stage: string;
    [key: string]: unknown;
  }

  interface PipelineStatusResponse {
    runs: PipelineRun[];
    total: number;
    active_count: number;
  }

  interface StageCounts {
    backtesting: number;
    sitGate: number;
    paperMonitoring: number;
    awaitingApproval: number;
  }

  let counts = $state<StageCounts>({
    backtesting: 0,
    sitGate: 0,
    paperMonitoring: 0,
    awaitingApproval: 0
  });

  onMount(async () => {
    try {
      const data = await apiFetch<PipelineStatusResponse>('/pipeline/status');
      const runs = data.runs ?? [];
      counts = {
        backtesting: runs.filter(r => r.current_stage === 'BACKTEST').length,
        sitGate: runs.filter(r => r.current_stage === 'VALIDATION').length,
        paperMonitoring: runs.filter(r => r.current_stage === 'EA_LIFECYCLE').length,
        awaitingApproval: runs.filter(r => r.current_stage === 'APPROVAL').length
      };
    } catch {
      // On error, keep all-zero state — neutral, not broken (AC 12-4-5)
      counts = { backtesting: 0, sitGate: 0, paperMonitoring: 0, awaitingApproval: 0 };
    }
  });

  function countColor(n: number): string {
    return n > 0 ? 'var(--color-accent-cyan)' : 'var(--color-text-muted)';
  }
</script>

<TileCard title="Enhancement Loop" size="xl">
  <div class="stage-grid">
    <div class="stage-item">
      <span class="section-label">Backtesting</span>
      <span class="financial-value stage-count" style="color: {countColor(counts.backtesting)};">
        {counts.backtesting}
      </span>
    </div>
    <div class="stage-separator" aria-hidden="true">·</div>
    <div class="stage-item">
      <span class="section-label">at SIT Gate</span>
      <span class="financial-value stage-count" style="color: {countColor(counts.sitGate)};">
        {counts.sitGate}
      </span>
    </div>
    <div class="stage-separator" aria-hidden="true">·</div>
    <div class="stage-item">
      <span class="section-label">Paper Monitoring</span>
      <span class="financial-value stage-count" style="color: {countColor(counts.paperMonitoring)};">
        {counts.paperMonitoring}
      </span>
    </div>
    <div class="stage-separator" aria-hidden="true">·</div>
    <div class="stage-item">
      <span class="section-label">Awaiting Approval</span>
      <span class="financial-value stage-count" style="color: {countColor(counts.awaitingApproval)};">
        {counts.awaitingApproval}
      </span>
    </div>
  </div>
</TileCard>

<style>
  .stage-grid {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
  }

  .stage-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-1);
    min-width: 80px;
  }

  .stage-count {
    font-size: var(--text-lg, 1.125rem);
    line-height: 1;
  }

  .stage-separator {
    color: var(--color-text-muted);
    font-size: var(--text-sm);
    flex-shrink: 0;
  }
</style>
