<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Clock,
    CheckCircle,
    AlertCircle,
    RefreshCw,
    Play,
    FileText,
    Bot,
    TrendingUp,
    ShieldAlert,
    Award,
    Calendar,
    Sun,
    Moon,
    ChevronRight,
    ChevronDown,
    TrendingDown,
    Activity,
    Target,
    Layers,
    Gauge,
    Database,
    GitBranch
  } from 'lucide-svelte';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  // Workflow state
  let workflowStatus = $state<{
    run_id: string | null;
    status: string;
    started_at: string | null;
    completed_at: string | null;
    steps: Record<string, {
      step_name: string;
      scheduled_time: string;
      day: string;
      status: string;
      started_at: string | null;
      completed_at: string | null;
      error: string | null;
    }>;
    friday_analysis: any;
    saturday_refinement: any;
    sunday_calibration: any;
    monday_roster: any;
    error: string | null;
  }>({
    run_id: null,
    status: 'not_started',
    started_at: null,
    completed_at: null,
    steps: {},
    friday_analysis: null,
    saturday_refinement: null,
    sunday_calibration: null,
    monday_roster: null,
    error: null,
  });

  // HMM Training Pool status
  let hmmPoolStatus = $state<{
    total_in_buffer: number;
    total_eligible: number;
    total_trades: number;
    avg_lag_remaining: number;
    trades: Array<{
      trade_id: string;
      bot_id: string;
      close_date: string;
      eligible_date: string;
      outcome: string;
      pnl: number;
      regime_at_entry: string;
      in_lag_buffer: boolean;
      lag_days_remaining: number;
      time_remaining: string;
      status: string;
    }>;
  } | null>(null);

  // Collapsed sections state
  let expandedSections = $state<Record<string, boolean>>({
    friday_analysis: true,
    saturday_wfa: true,
    hmm_retrain: true,
    kelly_calibration: true,
    sunday_roster: true,
  });

  let loading = $state(true);
  let hmmLoading = $state(true);
  let error = $state<string | null>(null);
  let lastUpdate = $state<string | null>(null);
  let schedulerStatus = $state({
    running: false,
    friday_start: '21:00',
    saturday_start: '06:00',
    sunday_start: '06:00',
    monday_deploy: '05:00',
  });

  // Phase groupings
  const phases = [
    {
      name: 'Friday',
      day: 'Friday',
      icon: Moon,
      color: '#8b5cf6',
      steps: [
        { key: 'friday_analysis', label: 'Friday Analysis', time: '21:00 GMT' },
      ],
    },
    {
      name: 'Saturday',
      day: 'Saturday',
      icon: Sun,
      color: '#f59e0b',
      steps: [
        { key: 'saturday_refinement', label: 'Workflow 2 Refinement', time: '06:00 GMT' },
        { key: 'saturday_wfa', label: 'Walk-Forward Analysis', time: '09:00 GMT' },
        { key: 'saturday_hmm_retrain', label: 'HMM Retraining', time: '12:00 GMT' },
      ],
    },
    {
      name: 'Sunday',
      day: 'Sunday',
      icon: Calendar,
      color: '#10b981',
      steps: [
        { key: 'sunday_calibration', label: 'Pre-Market Calibration', time: '06:00 GMT' },
        { key: 'sunday_spread_profiles', label: 'Spread Profiles Update', time: '09:00 GMT' },
        { key: 'sunday_sqs_refresh', label: 'SQS Baselines Refresh', time: '12:00 GMT' },
        { key: 'sunday_kelly_calibration', label: 'Kelly Calibration', time: '15:00 GMT' },
      ],
    },
    {
      name: 'Monday',
      day: 'Monday',
      icon: ChevronRight,
      color: '#3b82f6',
      steps: [
        { key: 'monday_roster_deploy', label: 'Roster Deployment', time: '05:00 GMT' },
      ],
    },
  ];

  async function fetchWorkflowStatus() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/weekend-cycle/workflow/status`);

      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      const data = await response.json();
      workflowStatus = data;
      lastUpdate = new Date().toLocaleTimeString();
      error = null;
    } catch (e) {
      console.error('Failed to fetch workflow status:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  async function fetchHmmPoolStatus() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/hmm/training-pool-status`);

      if (!response.ok) {
        throw new Error(`Failed to fetch HMM pool status: ${response.status}`);
      }

      const data = await response.json();
      hmmPoolStatus = data;
    } catch (e) {
      console.error('Failed to fetch HMM pool status:', e);
    } finally {
      hmmLoading = false;
    }
  }

  async function fetchSchedulerStatus() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/weekend-cycle/scheduler/status`);

      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      schedulerStatus = await response.json();
    } catch (e) {
      console.error('Failed to fetch scheduler status:', e);
    }
  }

  async function triggerWorkflow() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/weekend-cycle/workflow/trigger`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Failed to trigger: ${response.status}`);
      }

      const result = await response.json();
      console.log('Workflow triggered:', result);
      await fetchWorkflowStatus();
    } catch (e) {
      console.error('Failed to trigger workflow:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  async function triggerStep(stepName: string) {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/weekend-cycle/workflow/step/${stepName}/trigger`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Failed to trigger step: ${response.status}`);
      }

      const result = await response.json();
      console.log('Step triggered:', result);
      await fetchWorkflowStatus();
    } catch (e) {
      console.error('Failed to trigger step:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  function toggleSection(section: string) {
    expandedSections[section] = !expandedSections[section];
  }

  function getStepStatus(key: string): string {
    return workflowStatus.steps[key]?.status || 'pending';
  }

  function getPhaseStatus(phase: typeof phases[0]): 'completed' | 'running' | 'pending' | 'failed' {
    const phaseSteps = phase.steps.map(s => getStepStatus(s.key));
    if (phaseSteps.every(s => s === 'completed')) return 'completed';
    if (phaseSteps.some(s => s === 'running')) return 'running';
    if (phaseSteps.some(s => s === 'failed')) return 'failed';
    return 'pending';
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return '#10b981';
      case 'running': return '#f59e0b';
      case 'failed': return '#ef4444';
      default: return '#6b7280';
    }
  }

  function getStatusBgColor(status: string): string {
    switch (status) {
      case 'completed': return 'rgba(16, 185, 129, 0.15)';
      case 'running': return 'rgba(245, 158, 11, 0.15)';
      case 'failed': return 'rgba(239, 68, 68, 0.15)';
      default: return 'rgba(107, 114, 128, 0.1)';
    }
  }

  function getKellyMultiplierColor(multiplier: number): string {
    if (multiplier >= 1.5) return '#10b981';  // House money - green
    if (multiplier >= 1.0) return '#3b82f6';   // Normal - blue
    if (multiplier >= 0.7) return '#f59e0b';   // Stress - amber
    return '#ef4444';                           // Critical - red
  }

  function getKellyMultiplierLabel(multiplier: number): string {
    if (multiplier >= 1.5) return 'HOUSE MONEY';
    if (multiplier >= 1.0) return 'NORMAL';
    if (multiplier >= 0.7) return 'STRESS';
    return 'CRITICAL';
  }

  function getWfaRecommendationColor(rec: string): string {
    switch (rec) {
      case 'APPROVE': return '#10b981';
      case 'REJECT': return '#ef4444';
      default: return '#6b7280';
    }
  }

  onMount(() => {
    fetchWorkflowStatus();
    fetchSchedulerStatus();
    fetchHmmPoolStatus();
    const interval = setInterval(() => {
      fetchWorkflowStatus();
      fetchHmmPoolStatus();
    }, 30000);
    return () => clearInterval(interval);
  });

  // Derived data from workflow status
  $effect(() => {
    // This runs reactively when workflowStatus changes
  });
</script>

<div class="weekend-cycle-panel">
  <div class="panel-header">
    <Clock size={18} />
    <h3>Weekend Update Cycle</h3>
    <span class="workflow-status" class:running={workflowStatus.status === 'running'}>
      {workflowStatus.status.toUpperCase()}
    </span>
    {#if lastUpdate}
      <span class="last-update">Updated: {lastUpdate}</span>
    {/if}
    <div class="header-actions">
      <button class="refresh-btn" onclick={fetchWorkflowStatus} title="Refresh">
        <RefreshCw size={14} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
    </div>
  {/if}

  {#if loading}
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Loading workflow status...</span>
    </div>
  {:else}
    <!-- Scheduler Status -->
    <div class="scheduler-status">
      <span class="scheduler-label">Scheduler:</span>
      <span class="scheduler-value" class:active={schedulerStatus.running}>
        {schedulerStatus.running ? 'ACTIVE' : 'INACTIVE'}
      </span>
      <span class="schedule-times">
        Fri {schedulerStatus.friday_start} | Sat-Sun {schedulerStatus.saturday_start} | Mon {schedulerStatus.monday_deploy}
      </span>
    </div>

    <!-- Phase Progress -->
    <div class="phases-container">
      {#each phases as phase, phaseIndex}
        {@const phaseStatus = getPhaseStatus(phase)}
        {@const PhaseIcon = phase.icon}
        <div class="phase-card" style="--phase-color: {phase.color}">
          <div class="phase-header">
            <div class="phase-icon-wrapper" style="background: {getStatusBgColor(phaseStatus)}">
              <PhaseIcon size={16} color={getStatusColor(phaseStatus)} />
            </div>
            <span class="phase-name">{phase.name}</span>
            <span class="phase-status" style="background: {getStatusBgColor(phaseStatus)}; color: {getStatusColor(phaseStatus)}">
              {phaseStatus.toUpperCase()}
            </span>
          </div>

          <div class="phase-steps">
            {#each phase.steps as step, stepIndex}
              {@const status = getStepStatus(step.key)}
              <div class="step-row" style="--status-color: {getStatusColor(status)}">
                <div class="step-connector" class:first={stepIndex === 0}>
                  {#if stepIndex > 0}
                    <div class="connector-line" class:completed={getStepStatus(phase.steps[stepIndex - 1].key) === 'completed'}></div>
                  {/if}
                </div>
                <div class="step-content">
                  <div class="step-label">{step.label}</div>
                  <div class="step-time">{step.time}</div>
                </div>
                <div class="step-badge" style="background: {getStatusBgColor(status)}">
                  {#if status === 'completed'}
                    <CheckCircle size={12} color={getStatusColor(status)} />
                  {:else if status === 'running'}
                    <RefreshCw size={12} color={getStatusColor(status)} class="spinning" />
                  {:else if status === 'failed'}
                    <AlertCircle size={12} color={getStatusColor(status)} />
                  {:else}
                    <Clock size={12} color={getStatusColor(status)} />
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/each}
    </div>

    <!-- Current Run Info -->
    {#if workflowStatus.run_id}
      <div class="run-info">
        <div class="run-id">Run ID: {workflowStatus.run_id.slice(0, 8)}...</div>
        {#if workflowStatus.started_at}
          <div class="run-time">Started: {new Date(workflowStatus.started_at).toLocaleString()}</div>
        {/if}
        {#if workflowStatus.error}
          <div class="run-error">Error: {workflowStatus.error}</div>
        {/if}
      </div>
    {/if}

    <!-- ============================================== -->
    <!-- EXPANDABLE DETAIL SECTIONS -->
    <!-- ============================================== -->

    <!-- FRIDAY ANALYSIS PANEL -->
    <div class="detail-section">
      <button class="section-header" onclick={() => toggleSection('friday_analysis')}>
        <div class="section-title">
          <Moon size={16} color="#8b5cf6" />
          <span>Friday Analysis</span>
          <span class="section-time">21:00 GMT</span>
        </div>
        <div class="section-right">
          {#if workflowStatus.friday_analysis}
            <span class="section-badge completed">COMPLETED</span>
          {/if}
          {#if expandedSections.friday_analysis}
            <ChevronDown size={16} />
          {:else}
            <ChevronRight size={16} />
          {/if}
        </div>
      </button>

      {#if expandedSections.friday_analysis && workflowStatus.friday_analysis}
        <div class="section-content">
          {#if workflowStatus.friday_analysis.week_summary}
            {@const summary = workflowStatus.friday_analysis.week_summary}
            <div class="data-grid">
              <div class="data-card">
                <div class="data-card-header">
                  <TrendingUp size={14} />
                  <span>Week Performance</span>
                </div>
                <div class="data-card-body">
                  <div class="metric-row">
                    <span class="metric-label">Week</span>
                    <span class="metric-value">{summary.week_number}/{summary.year}</span>
                  </div>
                  <div class="metric-row">
                    <span class="metric-label">Total Trades</span>
                    <span class="metric-value">{summary.total_trades}</span>
                  </div>
                  <div class="metric-row">
                    <span class="metric-label">Win Rate</span>
                    <span class="metric-value">{(summary.overall_wr * 100).toFixed(1)}%</span>
                  </div>
                  <div class="metric-row">
                    <span class="metric-label">Net PnL</span>
                    <span class="metric-value" class:positive={summary.net_pnl > 0} class:negative={summary.net_pnl < 0}>
                      {summary.net_pnl >= 0 ? '+' : ''}{summary.net_pnl.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>

              <div class="data-card">
                <div class="data-card-header">
                  <Award size={14} />
                  <span>Bot Rankings</span>
                </div>
                <div class="data-card-body">
                  <div class="metric-row">
                    <span class="metric-label">Best Bot</span>
                    <span class="metric-value positive">{summary.best_bot || 'N/A'}</span>
                  </div>
                  <div class="metric-row">
                    <span class="metric-label">Worst Bot</span>
                    <span class="metric-value negative">{summary.worst_bot || 'N/A'}</span>
                  </div>
                  <div class="metric-row">
                    <span class="metric-label">Most Improved</span>
                    <span class="metric-value">{summary.most_improved_bot || 'N/A'}</span>
                  </div>
                </div>
              </div>
            </div>
          {/if}

          {#if workflowStatus.friday_analysis.regime_behaviour_report}
            {@const regime = workflowStatus.friday_analysis.regime_behaviour_report}
            <div class="data-card full-width">
              <div class="data-card-header">
                <Activity size={14} />
                <span>Regime Analysis</span>
              </div>
              <div class="data-card-body">
                <div class="regime-info">
                  <div class="regime-badge" style="background: rgba(139, 92, 246, 0.2); color: #8b5cf6;">
                    Dominant: {regime.dominant_regime || 'UNKNOWN'}
                  </div>
                  <div class="regime-badge" style="background: rgba(59, 130, 246, 0.2); color: #3b82f6;">
                    Transitions: {regime.regime_transitions || 0}
                  </div>
                </div>
                {#if regime.unusual_patterns && regime.unusual_patterns.length > 0}
                  <div class="alert-list">
                    {#each regime.unusual_patterns as pattern}
                      <div class="alert-item">
                        <AlertCircle size={12} color="#f59e0b" />
                        <span>{pattern}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
                {#if regime.concern_bots && regime.concern_bots.length > 0}
                  <div class="concern-bots">
                    <span class="concern-label">Concern Bots:</span>
                    {#each regime.concern_bots as bot}
                      <span class="bot-tag">{bot}</span>
                    {/each}
                  </div>
                {/if}
              </div>
            </div>
          {/if}

          {#if workflowStatus.friday_analysis.correlation_shifts && workflowStatus.friday_analysis.correlation_shifts.length > 0}
            <div class="data-card full-width">
              <div class="data-card-header">
                <GitBranch size={14} />
                <span>Correlation Shifts</span>
              </div>
              <div class="data-card-body">
                <div class="shifts-table">
                  <div class="shifts-header">
                    <span>Bot Pair</span>
                    <span>Previous</span>
                    <span>Current</span>
                    <span>Shift</span>
                  </div>
                  {#each workflowStatus.friday_analysis.correlation_shifts.slice(0, 5) as shift}
                    <div class="shifts-row">
                      <span class="pair-id">{shift.bot_pair}</span>
                      <span class="corr-value">{shift.previous_correlation.toFixed(3)}</span>
                      <span class="corr-value">{shift.current_correlation.toFixed(3)}</span>
                      <span class="shift-value" class:large-shift={Math.abs(shift.shift_magnitude) > 0.2}>
                        {shift.shift_magnitude > 0 ? '+' : ''}{shift.shift_magnitude.toFixed(3)}
                      </span>
                    </div>
                  {/each}
                </div>
              </div>
            </div>
          {/if}

          {#if workflowStatus.friday_analysis.candidate_bots_for_refinement}
            <div class="data-card full-width">
              <div class="data-card-header">
                <Target size={14} />
                <span>WFA Candidates ({workflowStatus.friday_analysis.candidate_bots_for_refinement.length})</span>
              </div>
              <div class="data-card-body">
                <div class="candidate-list">
                  {#each workflowStatus.friday_analysis.candidate_bots_for_refinement as bot}
                    <span class="bot-tag candidate">{bot}</span>
                  {/each}
                </div>
              </div>
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <!-- SATURDAY WFA PANEL -->
    <div class="detail-section">
      <button class="section-header" onclick={() => toggleSection('saturday_wfa')}>
        <div class="section-title">
          <Sun size={16} color="#f59e0b" />
          <span>Saturday WFA</span>
          <span class="section-time">09:00 GMT</span>
        </div>
        <div class="section-right">
          {#if workflowStatus.saturday_refinement?.wfa}
            <span class="section-badge completed">COMPLETED</span>
          {/if}
          {#if expandedSections.saturday_wfa}
            <ChevronDown size={16} />
          {:else}
            <ChevronRight size={16} />
          {/if}
        </div>
      </button>

      {#if expandedSections.saturday_wfa && workflowStatus.saturday_refinement?.wfa}
        {@const wfa = workflowStatus.saturday_refinement.wfa}
        <div class="section-content">
          {#if wfa.wfa_results && wfa.wfa_results.length > 0}
            {#each wfa.wfa_results as result}
              <div class="data-card full-width">
                <div class="data-card-header">
                  <Bot size={14} />
                  <span>Bot: {result.bot_id}</span>
                  <span class="recommendation-badge" style="background: {getWfaRecommendationColor(result.recommendation)}20; color: {getWfaRecommendationColor(result.recommendation)}">
                    {result.recommendation}
                  </span>
                </div>
                <div class="data-card-body">
                  {#if result.validation_metrics}
                    {@const metrics = result.validation_metrics}
                    <div class="metrics-grid">
                      <div class="metric-item">
                        <span class="metric-label">Sharpe Ratio</span>
                        <span class="metric-value" class:good={metrics.sharpe_ratio >= 1.0} class:bad={metrics.sharpe_ratio < 1.0}>
                          {metrics.sharpe_ratio.toFixed(4)}
                        </span>
                      </div>
                      <div class="metric-item">
                        <span class="metric-label">Max Drawdown</span>
                        <span class="metric-value">{(metrics.max_drawdown * 100).toFixed(2)}%</span>
                      </div>
                      <div class="metric-item">
                        <span class="metric-label">Win Rate</span>
                        <span class="metric-value">{(metrics.win_rate * 100).toFixed(1)}%</span>
                      </div>
                      <div class="metric-item">
                        <span class="metric-label">Profit Factor</span>
                        <span class="metric-value">{metrics.profit_factor.toFixed(4)}</span>
                      </div>
                      <div class="metric-item">
                        <span class="metric-label">Total Trades</span>
                        <span class="metric-value">{metrics.total_trades}</span>
                      </div>
                    </div>
                  {/if}
                  {#if result.optimal_parameters && Object.keys(result.optimal_parameters).length > 0}
                    <div class="param-changes">
                      <span class="param-label">Optimal Parameters:</span>
                      <div class="param-grid">
                        {#each Object.entries(result.optimal_parameters) as [key, value]}
                          <div class="param-item">
                            <span class="param-key">{key}</span>
                            <span class="param-val">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                          </div>
                        {/each}
                      </div>
                    </div>
                  {/if}
                </div>
              </div>
            {/each}
          {:else}
            <div class="empty-section">No WFA results available</div>
          {/if}
        </div>
      {/if}
    </div>

    <!-- HMM RETRAIN PANEL -->
    <div class="detail-section">
      <button class="section-header" onclick={() => toggleSection('hmm_retrain')}>
        <div class="section-title">
          <Layers size={16} color="#8b5cf6" />
          <span>HMM Retrain</span>
          <span class="section-time">12:00 GMT</span>
        </div>
        <div class="section-right">
          {#if workflowStatus.saturday_refinement?.hmm_retrain}
            <span class="section-badge completed">COMPLETED</span>
          {/if}
          {#if expandedSections.hmm_retrain}
            <ChevronDown size={16} />
          {:else}
            <ChevronRight size={16} />
          {/if}
        </div>
      </button>

      {#if expandedSections.hmm_retrain}
        <div class="section-content">
          {#if workflowStatus.saturday_refinement?.hmm_retrain}
            <div class="data-card full-width">
              <div class="data-card-header">
                <Database size={14} />
                <span>HMM Retrain Result</span>
              </div>
              <div class="data-card-body">
                <pre class="json-preview">{JSON.stringify(workflowStatus.saturday_refinement.hmm_retrain, null, 2)}</pre>
              </div>
            </div>
          {/if}

          <!-- HMM Training Pool Status -->
          <div class="data-card full-width">
            <div class="data-card-header">
              <Gauge size={14} />
              <span>HMM Training Pool</span>
              {#if !hmmLoading && hmmPoolStatus}
                <span class="pool-stats">
                  <span class="pool-stat buffer">{hmmPoolStatus.total_in_buffer} in buffer</span>
                  <span class="pool-stat eligible">{hmmPoolStatus.total_eligible} eligible</span>
                </span>
              {/if}
            </div>
            <div class="data-card-body">
              {#if hmmLoading}
                <div class="loading-small">
                  <div class="spinner-small"></div>
                  <span>Loading pool status...</span>
                </div>
              {:else if hmmPoolStatus}
                <div class="hmm-stats-grid">
                  <div class="hmm-stat-card buffer">
                    <div class="hmm-stat-value">{hmmPoolStatus.total_in_buffer}</div>
                    <div class="hmm-stat-label">In Buffer</div>
                  </div>
                  <div class="hmm-stat-card eligible">
                    <div class="hmm-stat-value">{hmmPoolStatus.total_eligible}</div>
                    <div class="hmm-stat-label">Eligible</div>
                  </div>
                  <div class="hmm-stat-card trades">
                    <div class="hmm-stat-value">{hmmPoolStatus.total_trades}</div>
                    <div class="hmm-stat-label">Total Trades</div>
                  </div>
                  <div class="hmm-stat-card lag">
                    <div class="hmm-stat-value">{hmmPoolStatus.avg_lag_remaining.toFixed(1)}</div>
                    <div class="hmm-stat-label">Avg Days Left</div>
                  </div>
                </div>

                {#if hmmPoolStatus.trades && hmmPoolStatus.trades.length > 0}
                  <div class="hmm-trades-list">
                    <div class="hmm-trades-header">
                      <span>Trade</span>
                      <span>Bot</span>
                      <span>Close Date</span>
                      <span>Eligible</span>
                      <span>Days Left</span>
                      <span>Status</span>
                    </div>
                    {#each hmmPoolStatus.trades.slice(0, 10) as trade}
                      <div class="hmm-trade-row" class:in-buffer={trade.in_lag_buffer}>
                        <span class="trade-id">{trade.trade_id.slice(0, 8)}...</span>
                        <span class="trade-bot">{trade.bot_id}</span>
                        <span class="trade-date">{new Date(trade.close_date).toLocaleDateString()}</span>
                        <span class="trade-eligible">{new Date(trade.eligible_date).toLocaleDateString()}</span>
                        <span class="trade-lag" class:warning={trade.lag_days_remaining <= 3}>{trade.lag_days_remaining}d</span>
                        <span class="trade-status">
                          {#if trade.in_lag_buffer}
                            <span class="status-badge buffer-badge">
                              <AlertCircle size={10} /> BUFFER
                            </span>
                          {:else}
                            <span class="status-badge eligible-badge">
                              <CheckCircle size={10} /> ELIGIBLE
                            </span>
                          {/if}
                        </span>
                      </div>
                    {/each}
                  </div>
                {/if}
              {:else}
                <div class="empty-section">HMM pool data not available</div>
              {/if}
            </div>
          </div>
        </div>
      {/if}
    </div>

    <!-- KELLY CALIBRATION PANEL -->
    <div class="detail-section">
      <button class="section-header" onclick={() => toggleSection('kelly_calibration')}>
        <div class="section-title">
          <Gauge size={16} color="#10b981" />
          <span>Kelly Calibration</span>
          <span class="section-time">15:00 GMT</span>
        </div>
        <div class="section-right">
          {#if workflowStatus.sunday_calibration?.kelly_calibration}
            <span class="section-badge completed">COMPLETED</span>
          {/if}
          {#if expandedSections.kelly_calibration}
            <ChevronDown size={16} />
          {:else}
            <ChevronRight size={16} />
          {/if}
        </div>
      </button>

      {#if expandedSections.kelly_calibration && workflowStatus.sunday_calibration?.kelly_calibration}
        {@const kelly = workflowStatus.sunday_calibration.kelly_calibration}
        <div class="section-content">
          {#if kelly.modifiers}
            <div class="kelly-sessions">
              {#each Object.entries(kelly.modifiers) as [session, data]}
                {@const sessionData = data as { morning: number; afternoon: number; win_rate: number; trade_count: number }}
                <div class="session-card">
                  <div class="session-header">
                    <span class="session-name">{session}</span>
                    <span class="session-stats">
                      WR: {(sessionData.win_rate * 100).toFixed(1)}% | Trades: {sessionData.trade_count}
                    </span>
                  </div>
                  <div class="kelly-bars">
                    <div class="kelly-bar-row">
                      <span class="kelly-label">Morning</span>
                      <div class="kelly-bar-container">
                        <div class="kelly-bar" style="width: {sessionData.morning * 100}%; background: {getKellyMultiplierColor(sessionData.morning)}"></div>
                      </div>
                      <span class="kelly-value" style="color: {getKellyMultiplierColor(sessionData.morning)}">
                        {(sessionData.morning * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div class="kelly-bar-row">
                      <span class="kelly-label">Afternoon</span>
                      <div class="kelly-bar-container">
                        <div class="kelly-bar" style="width: {sessionData.afternoon * 100}%; background: {getKellyMultiplierColor(sessionData.afternoon)}"></div>
                      </div>
                      <span class="kelly-value" style="color: {getKellyMultiplierColor(sessionData.afternoon)}">
                        {(sessionData.afternoon * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <div class="session-threshold">
                    <span class="threshold-label">Status:</span>
                    <span class="threshold-value" style="color: {getKellyMultiplierColor(sessionData.morning)}">
                      {getKellyMultiplierLabel(sessionData.morning)}
                    </span>
                  </div>
                </div>
              {/each}
            </div>
          {/if}

          <!-- SQS Baselines -->
          {#if workflowStatus.sunday_calibration?.sqs_refresh?.baselines}
            {@const baselines = workflowStatus.sunday_calibration.sqs_refresh.baselines}
            <div class="data-card full-width">
              <div class="data-card-header">
                <TrendingUp size={14} />
                <span>SQS Baselines</span>
              </div>
              <div class="data-card-body">
                <div class="baselines-grid">
                  <div class="baseline-item">
                    <span class="baseline-label">Default</span>
                    <span class="baseline-value">{(baselines.default * 100).toFixed(1)}%</span>
                  </div>
                  <div class="baseline-item">
                    <span class="baseline-label">High Volatility</span>
                    <span class="baseline-value">{(baselines.high_volatility * 100).toFixed(1)}%</span>
                  </div>
                  <div class="baseline-item">
                    <span class="baseline-label">Low Volatility</span>
                    <span class="baseline-value">{(baselines.low_volatility * 100).toFixed(1)}%</span>
                  </div>
                  <div class="baseline-item current">
                    <span class="baseline-label">Current Adjusted</span>
                    <span class="baseline-value">{(baselines.current_adjusted * 100).toFixed(1)}%</span>
                  </div>
                </div>
                {#if baselines.volatility_factor !== undefined}
                  <div class="factor-row">
                    <span>Volatility Factor: {baselines.volatility_factor.toFixed(4)}</span>
                    <span>Correlation Factor: {baselines.correlation_factor?.toFixed(4) || 'N/A'}</span>
                    <span>Regime: {baselines.regime || 'N/A'}</span>
                  </div>
                {/if}
              </div>
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <!-- SUNDAY ROSTER DEPLOY PANEL -->
    <div class="detail-section">
      <button class="section-header" onclick={() => toggleSection('sunday_roster')}>
        <div class="section-title">
          <Calendar size={16} color="#3b82f6" />
          <span>Sunday Roster</span>
          <span class="section-time">05:00 Mon</span>
        </div>
        <div class="section-right">
          {#if workflowStatus.monday_roster}
            <span class="section-badge completed">COMPLETED</span>
          {/if}
          {#if expandedSections.sunday_roster}
            <ChevronDown size={16} />
          {:else}
            <ChevronRight size={16} />
          {/if}
        </div>
      </button>

      {#if expandedSections.sunday_roster}
        <div class="section-content">
          {#if workflowStatus.sunday_calibration?.calibration?.roster_prepared}
            <div class="roster-status ready">
              <CheckCircle size={16} color="#10b981" />
              <span>Roster Prepared for Monday</span>
            </div>
          {/if}

          {#if workflowStatus.monday_roster?.deployment}
            {@const deployment = workflowStatus.monday_roster.deployment}
            <div class="data-card full-width">
              <div class="data-card-header">
                <Bot size={14} />
                <span>Monday Roster Deployment</span>
                <span class="deployment-status" class:deployed={deployment.status === 'deployed'} class:failed={deployment.status === 'failed'}>
                  {deployment.status?.toUpperCase() || 'UNKNOWN'}
                </span>
              </div>
              <div class="data-card-body">
                <div class="deployment-info">
                  <div class="metric-row">
                    <span class="metric-label">Bots Deployed</span>
                    <span class="metric-value">{deployment.bots_deployed || 0}</span>
                  </div>
                  {#if deployment.deployed_at}
                    <div class="metric-row">
                      <span class="metric-label">Deployed At</span>
                      <span class="metric-value">{new Date(deployment.deployed_at).toLocaleString()}</span>
                    </div>
                  {/if}
                  {#if deployment.error}
                    <div class="metric-row error">
                      <span class="metric-label">Error</span>
                      <span class="metric-value negative">{deployment.error}</span>
                    </div>
                  {/if}
                </div>
              </div>
            </div>
          {/if}

          {#if workflowStatus.sunday_calibration?.spread_profiles?.profiles}
            {@const profiles = workflowStatus.sunday_calibration.spread_profiles.profiles}
            <div class="data-card full-width">
              <div class="data-card-header">
                <FileText size={14} />
                <span>Spread Profiles</span>
              </div>
              <div class="data-card-body">
                <div class="spread-grid">
                  {#each Object.entries(profiles) as [pair, data]}
                    {@const spreadData = data as { avg_spread: number; max_spread: number }}
                    <div class="spread-item">
                      <span class="spread-pair">{pair}</span>
                      <div class="spread-values">
                        <span class="spread-avg">{(spreadData.avg_spread).toFixed(1)}</span>
                        <span class="spread-max">({(spreadData.max_spread).toFixed(1)})</span>
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            </div>
          {/if}
        </div>
      {/if}
    </div>

  {/if}
</div>

<style>
  .weekend-cycle-panel {
    background: rgba(8, 8, 12, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    color: #e4e4e7;
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }

  .panel-header h3 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
    flex: 1;
  }

  .workflow-status {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
  }

  .workflow-status.running {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .last-update {
    font-size: 11px;
    color: #6b7280;
  }

  .header-actions {
    display: flex;
    gap: 6px;
  }

  .refresh-btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 6px;
    cursor: pointer;
    color: #9ca3af;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }

  .refresh-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #e4e4e7;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    margin-bottom: 16px;
    color: #ef4444;
    font-size: 13px;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 32px;
    color: #6b7280;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-top-color: #10b981;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .spinner-small {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .loading-small {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    color: #6b7280;
    font-size: 12px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .scheduler-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    margin-bottom: 20px;
    font-size: 12px;
    flex-wrap: wrap;
  }

  .scheduler-label {
    color: #6b7280;
  }

  .scheduler-value {
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
  }

  .scheduler-value.active {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .schedule-times {
    margin-left: auto;
    color: #6b7280;
    font-size: 11px;
  }

  .phases-container {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }

  .phase-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 12px;
    border-left: 3px solid var(--phase-color);
  }

  .phase-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .phase-icon-wrapper {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .phase-name {
    font-size: 13px;
    font-weight: 600;
    flex: 1;
  }

  .phase-status {
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
  }

  .phase-steps {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .step-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }

  .step-connector {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 16px;
  }

  .step-connector.first {
    display: none;
  }

  .connector-line {
    width: 2px;
    height: 16px;
    background: rgba(107, 114, 128, 0.3);
  }

  .connector-line.completed {
    background: rgba(16, 185, 129, 0.5);
  }

  .step-content {
    flex: 1;
  }

  .step-label {
    font-size: 12px;
    color: #e4e4e7;
    margin-bottom: 1px;
  }

  .step-time {
    font-size: 10px;
    color: #6b7280;
  }

  .step-badge {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .step-badge :global(.spinning) {
    animation: spin 1s linear infinite;
  }

  .run-info {
    margin-top: 16px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    font-size: 12px;
  }

  .run-id {
    font-family: monospace;
    color: #9ca3af;
    margin-bottom: 4px;
  }

  .run-time {
    color: #6b7280;
  }

  .run-error {
    color: #ef4444;
    margin-top: 4px;
  }

  /* ============================================== */
  /* EXPANDABLE DETAIL SECTIONS */
  /* ============================================== */

  .detail-section {
    margin-top: 16px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    overflow: hidden;
  }

  .section-header {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 14px;
    background: transparent;
    border: none;
    color: #e4e4e7;
    cursor: pointer;
    transition: background 0.2s;
  }

  .section-header:hover {
    background: rgba(255, 255, 255, 0.03);
  }

  .section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    font-weight: 600;
  }

  .section-time {
    font-size: 10px;
    color: #6b7280;
    font-weight: normal;
  }

  .section-right {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #6b7280;
  }

  .section-badge {
    font-size: 9px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
  }

  .section-badge.completed {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .section-content {
    padding: 12px 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .empty-section {
    padding: 20px;
    text-align: center;
    color: #6b7280;
    font-size: 12px;
  }

  /* Data Cards */
  .data-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .data-card {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    padding: 12px;
    border: 1px solid rgba(255, 255, 255, 0.04);
  }

  .data-card.full-width {
    grid-column: 1 / -1;
  }

  .data-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    font-size: 12px;
    font-weight: 600;
    color: #9ca3af;
  }

  .data-card-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  /* Metric Rows */
  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  }

  .metric-row:last-child {
    border-bottom: none;
  }

  .metric-label {
    color: #6b7280;
  }

  .metric-value {
    color: #e4e4e7;
    font-weight: 500;
  }

  .metric-value.positive {
    color: #10b981;
  }

  .metric-value.negative {
    color: #ef4444;
  }

  .metric-row.error .metric-value {
    color: #ef4444;
  }

  /* Regime Info */
  .regime-info {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }

  .regime-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }

  .alert-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .alert-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: #f59e0b;
    padding: 6px 8px;
    background: rgba(245, 158, 11, 0.1);
    border-radius: 4px;
  }

  .concern-bots {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
  }

  .concern-label {
    font-size: 11px;
    color: #6b7280;
  }

  .bot-tag {
    padding: 2px 8px;
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border-radius: 4px;
    font-size: 10px;
    font-family: monospace;
  }

  .bot-tag.candidate {
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
  }

  /* Shifts Table */
  .shifts-table {
    font-size: 11px;
  }

  .shifts-header {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 1fr;
    gap: 8px;
    padding: 6px 0;
    color: #6b7280;
    text-transform: uppercase;
    font-size: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }

  .shifts-row {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 1fr;
    gap: 8px;
    padding: 6px 0;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  }

  .shifts-row:last-child {
    border-bottom: none;
  }

  .pair-id {
    font-family: monospace;
    color: #9ca3af;
  }

  .corr-value {
    color: #6b7280;
  }

  .shift-value {
    color: #e4e4e7;
  }

  .shift-value.large-shift {
    color: #f59e0b;
    font-weight: 600;
  }

  /* Candidate List */
  .candidate-list {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  /* WFA Metrics */
  .recommendation-badge {
    margin-left: auto;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 12px;
  }

  .metric-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .metric-item .metric-label {
    font-size: 9px;
    color: #6b7280;
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  .metric-item .metric-value {
    font-size: 14px;
    font-weight: 600;
  }

  .metric-value.good {
    color: #10b981;
  }

  .metric-value.bad {
    color: #ef4444;
  }

  /* Param Changes */
  .param-changes {
    margin-top: 8px;
  }

  .param-label {
    font-size: 11px;
    color: #6b7280;
    display: block;
    margin-bottom: 8px;
  }

  .param-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .param-item {
    display: flex;
    flex-direction: column;
    padding: 6px 8px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
  }

  .param-key {
    font-size: 10px;
    color: #6b7280;
    font-family: monospace;
  }

  .param-val {
    font-size: 11px;
    color: #e4e4e7;
    font-family: monospace;
  }

  /* HMM Stats */
  .pool-stats {
    display: flex;
    gap: 12px;
    margin-left: auto;
    font-size: 10px;
  }

  .pool-stat {
    padding: 2px 6px;
    border-radius: 4px;
  }

  .pool-stat.buffer {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .pool-stat.eligible {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .hmm-stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 12px;
  }

  .hmm-stat-card {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 6px;
    padding: 10px;
    text-align: center;
    border-left: 3px solid;
  }

  .hmm-stat-card.buffer {
    border-color: #f59e0b;
  }

  .hmm-stat-card.eligible {
    border-color: #10b981;
  }

  .hmm-stat-card.trades {
    border-color: #3b82f6;
  }

  .hmm-stat-card.lag {
    border-color: #8b5cf6;
  }

  .hmm-stat-value {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
  }

  .hmm-stat-label {
    font-size: 9px;
    color: #64748b;
    text-transform: uppercase;
    margin-top: 2px;
  }

  .hmm-trades-list {
    font-size: 11px;
  }

  .hmm-trades-header {
    display: grid;
    grid-template-columns: 80px 100px 80px 80px 50px 90px;
    gap: 8px;
    padding: 6px 0;
    color: #6b7280;
    text-transform: uppercase;
    font-size: 9px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }

  .hmm-trade-row {
    display: grid;
    grid-template-columns: 80px 100px 80px 80px 50px 90px;
    gap: 8px;
    padding: 6px 0;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  }

  .hmm-trade-row.in-buffer {
    background: rgba(245, 158, 11, 0.05);
  }

  .trade-id {
    font-family: monospace;
    font-size: 10px;
    color: #9ca3af;
  }

  .trade-bot {
    font-family: monospace;
    font-size: 10px;
    color: #60a5fa;
  }

  .trade-date, .trade-eligible {
    font-size: 10px;
    color: #6b7280;
  }

  .trade-lag {
    font-size: 11px;
    color: #e4e4e7;
    text-align: center;
  }

  .trade-lag.warning {
    color: #f59e0b;
    font-weight: 600;
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: 600;
  }

  .buffer-badge {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
  }

  .eligible-badge {
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
  }

  .json-preview {
    background: rgba(0, 0, 0, 0.3);
    padding: 10px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 10px;
    color: #9ca3af;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 200px;
    overflow-y: auto;
  }

  /* Kelly Sessions */
  .kelly-sessions {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .session-card {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    padding: 12px;
    border: 1px solid rgba(255, 255, 255, 0.04);
  }

  .session-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .session-name {
    font-size: 13px;
    font-weight: 600;
    color: #e4e4e7;
  }

  .session-stats {
    font-size: 10px;
    color: #6b7280;
  }

  .kelly-bars {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .kelly-bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .kelly-label {
    width: 70px;
    font-size: 11px;
    color: #6b7280;
  }

  .kelly-bar-container {
    flex: 1;
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }

  .kelly-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .kelly-value {
    width: 45px;
    text-align: right;
    font-size: 12px;
    font-weight: 600;
    font-family: monospace;
  }

  .session-threshold {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 11px;
  }

  .threshold-label {
    color: #6b7280;
  }

  .threshold-value {
    font-weight: 600;
  }

  /* Baselines */
  .baselines-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 10px;
  }

  .baseline-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 8px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
  }

  .baseline-item.current {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
  }

  .baseline-label {
    font-size: 9px;
    color: #6b7280;
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  .baseline-value {
    font-size: 16px;
    font-weight: 700;
    color: #e4e4e7;
  }

  .baseline-item.current .baseline-value {
    color: #10b981;
  }

  .factor-row {
    display: flex;
    gap: 16px;
    font-size: 10px;
    color: #6b7280;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
  }

  /* Roster */
  .roster-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 6px;
    font-size: 12px;
    color: #10b981;
  }

  .deployment-status {
    margin-left: auto;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
  }

  .deployment-status.deployed {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .deployment-status.failed {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .deployment-info {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  /* Spread Grid */
  .spread-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
  }

  .spread-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 8px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
  }

  .spread-pair {
    font-size: 11px;
    font-weight: 600;
    color: #e4e4e7;
    margin-bottom: 4px;
  }

  .spread-values {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .spread-avg {
    font-size: 14px;
    font-weight: 700;
    color: #10b981;
  }

  .spread-max {
    font-size: 10px;
    color: #6b7280;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .phases-container {
      grid-template-columns: 1fr;
    }

    .data-grid {
      grid-template-columns: 1fr;
    }

    .metrics-grid {
      grid-template-columns: repeat(3, 1fr);
    }

    .hmm-stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .baselines-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .spread-grid {
      grid-template-columns: repeat(3, 1fr);
    }
  }
</style>
