<script lang="ts">
  /**
   * Development Canvas
   *
   * Development Department Head workspace.
   * AlphaForge pipeline stage tracker at top, dense tab-navigated tile grid,
   *
   * Frosted Terminal aesthetic — cyan (#00d4ff) dept accent.
   * Svelte 5 runes only.
   */
  import { onMount } from 'svelte';
  import {
    Code2,
    GitBranch,
    FlaskConical,
    Rocket,
    TestTube2,
    Kanban,
    Inbox,
    Loader2,
    Play,
    Square,
    TrendingUp,
    Clock,
    CheckCircle2,
    CheckCircle,
    XCircle,
    BarChart3,
    AlertCircle,
    Cpu,
    Split,
    FileCode,
    Layers,
    Save,
    Trash2,
    Lightbulb,
    ChevronDown
  } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import DepartmentKanban from '$lib/components/department-kanban/DepartmentKanban.svelte';
  import AgentTilePanel from '$lib/components/AgentTilePanel.svelte';
  import PipelineBoard from '$lib/components/development/PipelineBoard.svelte';
  import VariantBrowser from '$lib/components/development/VariantBrowser.svelte';
  import ABComparisonView from '$lib/components/development/ABComparisonView.svelte';
  import ProvenanceChain from '$lib/components/development/ProvenanceChain.svelte';
  import MonacoEditorStub from '$lib/components/development/MonacoEditorStub.svelte';
  import MonacoEditor from '$lib/components/MonacoEditor.svelte';
  import BacktestRunner from '$lib/components/BacktestRunner.svelte';
  import { alphaForgeStore } from '$lib/stores/alpha-forge';

  // =============================================================================
  // Types
  // =============================================================================

  type DevTab = 'active-eas' | 'variants' | 'backtest' | 'pipeline' | 'workflows' | 'dept-tasks';

  interface ActiveEA {
    id: string;
    name: string;
    symbol: string;
    status: 'RUNNING' | 'STOPPED' | 'ERROR';
    pnl_today: number;
    last_modified: string;
  }

  type StageStatus = 'waiting' | 'running' | 'passed' | 'failed';

  interface BacktestStage {
    status: StageStatus;
    win_rate?: number;
    max_drawdown?: number;
    passed_threshold?: boolean;
  }

  interface BacktestResult {
    strategy_id: string;
    strategy_name: string;
    vanilla?: BacktestStage;
    spiced?: BacktestStage;
    monte_carlo?: BacktestStage;
  }

  interface DeploymentListItem {
    deployment_id: string;
    strategy_id: string;
    status: string;
    started_at: string;
    completed_at?: string | null;
  }

  interface DeploymentListResponse {
    deployments?: DeploymentListItem[];
  }

  // =============================================================================
  // State
  // =============================================================================

  let activeTab = $state<DevTab>('active-eas');
  let loadingTab = $state<DevTab | null>(null);

  // Active EAs
  let activeEAs = $state<ActiveEA[]>([]);

  // Backtest pipeline
  let backtestResults = $state<BacktestResult[]>([]);
  let backtestLoading = $state(false);
  let backtestLoaded = $state(false);

  // Workflow builder state
  let workflowCode = $state('# Select a template or start from scratch\n');

  // Agent insights strip
  let insightsExpanded = $state(false);
  let insightsUnread = $state(0);

  // =============================================================================
  // Tab config
  // =============================================================================

  const tabs: { id: DevTab; label: string; icon: typeof Code2 }[] = [
    { id: 'active-eas',  label: 'Active EAs',  icon: Cpu       },
    { id: 'variants',    label: 'Variants',     icon: GitBranch },
    { id: 'backtest',    label: 'Backtest',     icon: FlaskConical },
    { id: 'pipeline',    label: 'Pipeline',     icon: Kanban    },
    { id: 'workflows',   label: 'Workflows',    icon: GitBranch },
    { id: 'dept-tasks',  label: 'Dept Tasks',   icon: CheckCircle2 },
  ];

  // Workflow Templates (mirrored from FlowForge Builder)
  const WORKFLOW_TEMPLATES = [
    {
      id: 'alpha-research',
      name: 'Alpha Research Pipeline',
      description: 'Hypothesis → TRD → Backtest',
      code: `# Alpha Research Pipeline
# Hypothesis → TRD → Backtest

from prefect import flow, task

@task
def generate_hypothesis(market: str) -> dict:
    """Research agent generates trading hypothesis."""
    return {"market": market, "signal": "momentum", "confidence": 0.75}

@task
def create_trd(hypothesis: dict) -> dict:
    """Convert hypothesis to Technical Requirements Document."""
    return {"hypothesis": hypothesis, "strategy_type": "MA_crossover"}

@task
def run_backtest(trd: dict) -> dict:
    """Run backtest on the strategy specification."""
    return {"trd": trd, "sharpe": 1.42, "max_drawdown": 0.08}

@flow(name="alpha-research-pipeline")
def alpha_research_pipeline(market: str = "EURUSD"):
    hypothesis = generate_hypothesis(market)
    trd = create_trd(hypothesis)
    results = run_backtest(trd)
    return results

if __name__ == "__main__":
    alpha_research_pipeline()
`,
    },
    {
      id: 'ea-deploy',
      name: 'EA Deployment Flow',
      description: 'Code → Review → Deploy to MT5',
      code: `# EA Deployment Flow
# Code → Review → Deploy to MT5

from prefect import flow, task

@task
def generate_ea_code(strategy_id: str) -> str:
    """Development head generates MQL5 EA code."""
    return f"// EA for strategy {strategy_id}\\n// Generated by QuantMindX"

@task
def review_ea(code: str) -> dict:
    """Risk head reviews EA for compliance."""
    return {"approved": True, "notes": "Compliant with prop firm rules"}

@task
def compile_ea(code: str) -> str:
    """Compile MQL5 code via compiler service."""
    return "ea_compiled.ex5"

@task
def deploy_to_mt5(compiled: str, account_id: str) -> bool:
    """Deploy EA to MT5 account."""
    return True

@flow(name="ea-deployment-flow")
def ea_deployment_flow(strategy_id: str, account_id: str = "demo_001"):
    code = generate_ea_code(strategy_id)
    review = review_ea(code)
    if review["approved"]:
        compiled = compile_ea(code)
        deploy_to_mt5(compiled, account_id)

if __name__ == "__main__":
    ea_deployment_flow(strategy_id="strat_001")
`,
    },
    {
      id: 'risk-scan',
      name: 'Risk Scan',
      description: 'Portfolio risk assessment workflow',
      code: `# Risk Scan Workflow
# Portfolio risk assessment

from prefect import flow, task

@task
def fetch_positions() -> list:
    """Fetch current open positions from MT5."""
    return [{"symbol": "EURUSD", "volume": 0.1, "profit": 50.0}]

@task
def calculate_var(positions: list) -> float:
    """Calculate Value at Risk (VaR) using HMM regime data."""
    return 0.02  # 2% VaR

@task
def check_drawdown(positions: list) -> dict:
    """Check current drawdown vs prop firm limits."""
    return {"current_dd": 0.03, "limit": 0.05, "breached": False}

@task
def run_physics_sensors(positions: list) -> dict:
    """Run physics-based risk sensors (volatility regime)."""
    return {"regime": "low_vol", "hmm_state": 1}

@task
def emit_risk_report(var: float, dd: dict, sensors: dict):
    """Emit risk assessment report to Risk Canvas."""
    print(f"VaR: {var:.1%} | DD: {dd['current_dd']:.1%} | Regime: {sensors['regime']}")

@flow(name="risk-scan")
def risk_scan():
    positions = fetch_positions()
    var = calculate_var(positions)
    dd = check_drawdown(positions)
    sensors = run_physics_sensors(positions)
    emit_risk_report(var, dd, sensors)

if __name__ == "__main__":
    risk_scan()
`,
    },
    {
      id: 'news-scan',
      name: 'News Scanner',
      description: 'Geopolitical event monitoring',
      code: `# News Scanner Workflow
# Geopolitical event monitoring

from prefect import flow, task

@task
def fetch_news_feed(sources: list[str]) -> list:
    """Fetch latest news from configured sources."""
    return [{"title": "Fed signals rate hold", "sentiment": -0.2, "impact": "high"}]

@task
def classify_events(articles: list) -> list:
    """Classify news events by market impact level."""
    return [{"article": a, "trading_relevant": True} for a in articles]

@task
def check_blackout_calendar(events: list) -> list:
    """Cross-reference with CalendarGovernor blackout windows."""
    return [e for e in events if not e.get("blackout", False)]

@task
def route_to_research(events: list):
    """Route high-impact events to Research department."""
    for event in events:
        if event.get("trading_relevant"):
            print(f"Routing to Research: {event['article']['title']}")

@flow(name="news-scanner")
def news_scanner(sources: list[str] = ["reuters", "bloomberg"]):
    articles = fetch_news_feed(sources)
    events = classify_events(articles)
    filtered = check_blackout_calendar(events)
    route_to_research(filtered)

if __name__ == "__main__":
    news_scanner()
`,
    },
  ];

  // Load a workflow template into the editor
  function loadWorkflowTemplate(id: string) {
    const template = WORKFLOW_TEMPLATES.find((t) => t.id === id);
    if (template) {
      workflowCode = template.code;
    }
  }

  // Clear the workflow editor
  function clearWorkflowEditor() {
    workflowCode = '# Select a template or start from scratch\n';
  }

  // Run workflow (stub)
  function runWorkflow() {
    console.log('[Dev Workflows] Run workflow triggered', workflowCode);
  }

  // Save as template (stub)
  function saveWorkflowAsTemplate() {
    console.log('[Dev Workflows] Save as template triggered');
  }

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(async () => {
    try {
      await canvasContextService.loadCanvasContext('development');
    } catch {
      // canvas context is optional
    }

    alphaForgeStore.startPolling(10000);

    await loadTab('active-eas');
  });

  // =============================================================================
  // Data loading
  // =============================================================================

  async function loadTab(tab: DevTab) {
    if (tab === 'dept-tasks' || tab === 'variants' || tab === 'pipeline' || tab === 'workflows') return;
    if (loadingTab === tab) return;

    loadingTab = tab;
    try {
      if (tab === 'active-eas') {
        const data = await apiFetch<DeploymentListResponse>('/deployment/list?limit=50');
        const deployments = Array.isArray(data?.deployments) ? data.deployments : [];
        activeEAs = deployments.map((deployment) => {
          const status = deployment.status?.toLowerCase();
          let mappedStatus: ActiveEA['status'] = 'STOPPED';
          if (status === 'deployed' || status === 'running') {
            mappedStatus = 'RUNNING';
          } else if (status === 'failed' || status === 'error') {
            mappedStatus = 'ERROR';
          }
          return {
            id: deployment.deployment_id,
            name: deployment.strategy_id || deployment.deployment_id,
            symbol: '',
            status: mappedStatus,
            pnl_today: 0,
            last_modified: deployment.completed_at || deployment.started_at || '',
          };
        });
      } else if (tab === 'backtest' && !backtestLoaded) {
        backtestLoading = true;
        const data = await apiFetch<BacktestResult[]>('/backtest/results?limit=20');
        backtestResults = Array.isArray(data) ? data : [];
        backtestLoaded = true;
        backtestLoading = false;
      }
    } catch {
      // empty state — backend not available is non-error
      if (tab === 'backtest') {
        backtestResults = [];
        backtestLoaded = true;
        backtestLoading = false;
      }
    } finally {
      loadingTab = null;
    }
  }

  async function handleTabChange(tab: DevTab) {
    activeTab = tab;
    await loadTab(tab);
  }

  // =============================================================================
  // Helpers
  // =============================================================================

  function formatPnl(pnl: number): string {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}$${Math.abs(pnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function getStatusColor(status: ActiveEA['status']): string {
    switch (status) {
      case 'RUNNING': return '#00d4ff';
      case 'STOPPED': return '#6b7280';
      case 'ERROR':   return '#ef4444';
    }
  }

  function getStatusIcon(status: ActiveEA['status']) {
    switch (status) {
      case 'RUNNING': return Play;
      case 'STOPPED': return Square;
      case 'ERROR':   return AlertCircle;
    }
  }

  // =============================================================================
  // Backtest pipeline helpers
  // =============================================================================

  function stageLabel(status: StageStatus | undefined): string {
    switch (status) {
      case 'running': return 'Running';
      case 'passed':  return 'Approved';
      case 'failed':  return 'Archived';
      default:        return 'Waiting';
    }
  }

  function stageBadgeClass(status: StageStatus | undefined): string {
    switch (status) {
      case 'running': return 'badge-running';
      case 'passed':  return 'badge-passed';
      case 'failed':  return 'badge-failed';
      default:        return 'badge-waiting';
    }
  }

  function fmtPct(val: number | undefined): string {
    if (val === undefined || val === null) return '—';
    return `${(val * 100).toFixed(1)}%`;
  }

  function getDevelopmentAttachableResources() {
    const baseResources = [
      {
        id: `development:tab:${activeTab}`,
        label: tabs.find((tab) => tab.id === activeTab)?.label ?? activeTab,
        canvas: 'development',
        resource_type: 'active-tab',
        metadata: {
          active_tab: activeTab,
          loading: loadingTab === activeTab,
        },
      },
    ];

    if (activeTab === 'active-eas') {
      return [
        ...baseResources,
        ...activeEAs.slice(0, 50).map((ea) => ({
          id: ea.id,
          label: ea.name,
          canvas: 'development',
          resource_type: 'active-ea',
          metadata: {
            symbol: ea.symbol,
            status: ea.status,
            pnl_today: ea.pnl_today,
            last_modified: ea.last_modified,
          },
        })),
      ];
    }

    if (activeTab === 'backtest') {
      return [
        ...baseResources,
        ...backtestResults.slice(0, 50).map((result) => ({
          id: result.strategy_id,
          label: result.strategy_name,
          canvas: 'development',
          resource_type: 'backtest-result',
          metadata: {
            strategy_id: result.strategy_id,
            vanilla_status: result.vanilla?.status ?? 'waiting',
            spiced_status: result.spiced?.status ?? 'waiting',
            monte_carlo_status: result.monte_carlo?.status ?? 'waiting',
          },
        })),
      ];
    }

    if (activeTab === 'workflows') {
      return [
        ...baseResources,
        ...WORKFLOW_TEMPLATES.map((template) => ({
          id: template.id,
          label: template.name,
          canvas: 'development',
          resource_type: 'workflow-template',
          metadata: {
            description: template.description,
          },
        })),
        {
          id: 'development:workflow-editor',
          label: 'Workflow Editor',
          canvas: 'development',
          resource_type: 'editor',
          metadata: {
            has_code: workflowCode.trim().length > 0,
            line_count: workflowCode.split('\n').length,
          },
        },
      ];
    }

    if (activeTab === 'pipeline') {
      return [
        ...baseResources,
        {
          id: 'development:pipeline-board',
          label: 'Pipeline Board',
          canvas: 'development',
          resource_type: 'board',
          metadata: {
            active_tab: activeTab,
          },
        },
        {
          id: 'development:department-kanban',
          label: 'Department Task Board',
          canvas: 'development',
          resource_type: 'kanban',
          metadata: {
            department: 'development',
          },
        },
      ];
    }

    if (activeTab === 'variants') {
      return [
        ...baseResources,
        {
          id: 'development:variant-browser',
          label: 'Variant Browser',
          canvas: 'development',
          resource_type: 'browser',
          metadata: {
            active_tab: activeTab,
          },
        },
        {
          id: 'development:variant-editor',
          label: 'Variant Editor',
          canvas: 'development',
          resource_type: 'editor',
          metadata: {
            read_only: true,
          },
        },
      ];
    }

    if (activeTab === 'dept-tasks') {
      return [
        ...baseResources,
        {
          id: 'development:department-kanban',
          label: 'Department Task Board',
          canvas: 'development',
          resource_type: 'kanban',
          metadata: {
            department: 'development',
          },
        },
      ];
    }

    return baseResources;
  }

  $effect(() => {
    canvasContextService.setRuntimeState('development', {
      active_tab: activeTab,
      loading_tab: loadingTab,
      active_ea_count: activeEAs.length,
      backtest_result_count: backtestResults.length,
      attachable_resources: getDevelopmentAttachableResources(),
    });
  });
</script>

<!-- =========================================================
     Shell
     ========================================================= -->
<div class="development-canvas" data-dept="development">

  <!-- =========================================================
       Canvas Header
       ========================================================= -->
  <header class="canvas-header">
    <div class="header-left">
      <Code2 size={18} class="dept-icon" />
      <h1 class="canvas-title">Development</h1>
      <span class="dept-badge">Alpha Forge</span>
    </div>

    <!-- Tab navigation -->
    <nav class="tab-nav">
      {#each tabs as tab}
        <button
          class="tab-btn"
          class:active={activeTab === tab.id}
          onclick={() => handleTabChange(tab.id)}
        >
          <svelte:component this={tab.icon} size={13} />
          <span>{tab.label}</span>
          {#if loadingTab === tab.id}
            <Loader2 size={11} class="spin" />
          {/if}
        </button>
      {/each}
    </nav>
  </header>

  <!-- =========================================================
       Main content area
       ========================================================= -->
  <div class="canvas-body">

    {#if activeTab === 'dept-tasks'}
      <!-- ---- Dept Tasks: inline kanban ---- -->
      <div class="kanban-wrapper">
        <DepartmentKanban department="development" />
      </div>

    {:else if activeTab === 'pipeline'}
      <!-- ---- Pipeline: full PipelineBoard + kanban sub-section ---- -->
      <div class="pipeline-wrapper">
        <div class="pipeline-board-section">
          <PipelineBoard />
        </div>
        <div class="kanban-sub-section">
          <div class="sub-section-label">
            <Kanban size={13} />
            <span>Department Task Board</span>
          </div>
          <DepartmentKanban department="development" />
        </div>
      </div>

    {:else if activeTab === 'variants'}
      <!-- ---- Variants: VariantBrowser + Monaco editor ---- -->
      <div class="variants-wrapper">
        <div class="variant-browser-pane">
          <VariantBrowser />
        </div>
        <div class="monaco-pane">
          <MonacoEditorStub
            strategyName="Selected Strategy"
            variantType="vanilla"
            version="1.0.0"
            readOnly={true}
          />
        </div>
      </div>

    {:else if activeTab === 'backtest'}
      <!-- ---- Backtest: visual pipeline per strategy ---- -->
      <div class="backtest-wrapper">

        <!-- Header row -->
        <div class="backtest-header">
          <BarChart3 size={15} class="backtest-header-icon" />
          <span class="backtest-header-title">Backtest Pipeline</span>
          <span class="backtest-header-sub">Vanilla → Spiced → Monte Carlo</span>
        </div>

        {#if backtestLoading}
          <!-- Loading state -->
          <div class="backtest-empty-state">
            <Loader2 size={28} class="spin" />
            <span>Loading backtest results…</span>
          </div>

        {:else if backtestResults.length === 0}
          <!-- Empty state -->
          <div class="backtest-empty-state">
            <FlaskConical size={32} class="backtest-empty-icon" />
            <span class="backtest-empty-label">No backtest results yet</span>
            <span class="backtest-empty-sub">Run a strategy through the AlphaForge pipeline to see results here</span>
          </div>

        {:else}
          <!-- Strategy cards -->
          <div class="backtest-cards">
            {#each backtestResults as result (result.strategy_id)}
              <div class="backtest-card">

                <!-- Card header: strategy name -->
                <div class="backtest-card-header">
                  <FlaskConical size={13} class="backtest-card-icon" />
                  <span class="backtest-card-name">{result.strategy_name}</span>
                  <span class="backtest-card-id">{result.strategy_id}</span>
                </div>

                <!-- Pipeline stages row -->
                <div class="pipeline-stages">

                  <!-- Vanilla -->
                  <div class="pipeline-stage">
                    <span class="stage-label">Vanilla</span>
                    <span class="stage-badge {stageBadgeClass(result.vanilla?.status)}">
                      {#if result.vanilla?.status === 'running'}
                        <span class="pulse-dot"></span>
                        Running
                      {:else if result.vanilla?.status === 'passed'}
                        <CheckCircle size={11} />
                        Approved
                      {:else if result.vanilla?.status === 'failed'}
                        <XCircle size={11} />
                        Archived
                      {:else}
                        Waiting
                      {/if}
                    </span>
                    {#if result.vanilla?.win_rate !== undefined}
                      <span class="stage-stat">WR {fmtPct(result.vanilla.win_rate)}</span>
                    {/if}
                  </div>

                  <!-- Connector -->
                  <div class="stage-connector" aria-hidden="true">→</div>

                  <!-- Spiced -->
                  <div class="pipeline-stage">
                    <span class="stage-label">Spiced</span>
                    <span class="stage-badge {stageBadgeClass(result.spiced?.status)}">
                      {#if result.spiced?.status === 'running'}
                        <span class="pulse-dot"></span>
                        Running
                      {:else if result.spiced?.status === 'passed'}
                        <CheckCircle size={11} />
                        Approved
                      {:else if result.spiced?.status === 'failed'}
                        <XCircle size={11} />
                        Archived
                      {:else}
                        Waiting
                      {/if}
                    </span>
                    {#if result.spiced?.win_rate !== undefined}
                      <span class="stage-stat">WR {fmtPct(result.spiced.win_rate)}</span>
                    {/if}
                  </div>

                  <!-- Connector -->
                  <div class="stage-connector" aria-hidden="true">→</div>

                  <!-- Monte Carlo -->
                  <div class="pipeline-stage">
                    <span class="stage-label">Monte Carlo</span>
                    <span class="stage-badge {stageBadgeClass(result.monte_carlo?.status)}">
                      {#if result.monte_carlo?.status === 'running'}
                        <span class="pulse-dot"></span>
                        Running
                      {:else if result.monte_carlo?.status === 'passed'}
                        <CheckCircle size={11} />
                        Approved
                      {:else if result.monte_carlo?.status === 'failed'}
                        <XCircle size={11} />
                        Archived
                      {:else}
                        Waiting
                      {/if}
                    </span>
                    <!-- Monte Carlo stats -->
                    {#if result.monte_carlo?.win_rate !== undefined || result.monte_carlo?.max_drawdown !== undefined}
                      <div class="mc-stats">
                        {#if result.monte_carlo?.win_rate !== undefined}
                          <span class="mc-stat" class:mc-stat-pass={result.monte_carlo.win_rate >= 0.48} class:mc-stat-fail={result.monte_carlo.win_rate < 0.48}>
                            WR {fmtPct(result.monte_carlo.win_rate)}
                            <span class="mc-threshold">/ 48% min</span>
                          </span>
                        {/if}
                        {#if result.monte_carlo?.max_drawdown !== undefined}
                          <span class="mc-stat mc-stat-dd">
                            DD {fmtPct(result.monte_carlo.max_drawdown)}
                          </span>
                        {/if}
                      </div>
                    {/if}
                  </div>

                </div>
              </div>
            {/each}
          </div>
        {/if}

      </div>

    {:else if activeTab === 'workflows'}
      <!-- ---- Workflows: template library + Monaco editor ---- -->
      <div class="workflows-builder-layout">
        <!-- Left panel: template library -->
        <div class="workflows-template-panel">
          <div class="workflows-template-header">
            <Layers size={14} />
            <span>Workflow Templates</span>
          </div>
          <div class="workflows-template-list">
            <button class="workflows-template-item blank-btn" onclick={clearWorkflowEditor}>
              <span class="workflows-template-name">Blank</span>
              <span class="workflows-template-desc">Start from scratch</span>
            </button>
            {#each WORKFLOW_TEMPLATES as tpl (tpl.id)}
              <button
                class="workflows-template-item"
                onclick={() => loadWorkflowTemplate(tpl.id)}
              >
                <span class="workflows-template-name">{tpl.name}</span>
                <span class="workflows-template-desc">{tpl.description}</span>
              </button>
            {/each}
          </div>
        </div>

        <!-- Right panel: Monaco editor -->
        <div class="workflows-editor-panel">
          <MonacoEditor bind:content={workflowCode} language="python" filename="workflow.py" />
          <div class="workflows-editor-toolbar">
            <button class="workflows-toolbar-btn run-btn" onclick={runWorkflow}>
              <Play size={13} />
              <span>Run</span>
            </button>
            <button class="workflows-toolbar-btn save-btn" onclick={saveWorkflowAsTemplate}>
              <Save size={13} />
              <span>Save as Template</span>
            </button>
            <button class="workflows-toolbar-btn clear-btn" onclick={clearWorkflowEditor}>
              <Trash2 size={13} />
              <span>Clear</span>
            </button>
          </div>
        </div>
      </div>

    {:else}
      <!-- ---- Active EAs tile grid ---- -->
      <div class="tile-grid-wrapper">
        {#if loadingTab === 'active-eas'}
          <div class="empty-state">
            <Loader2 size={28} class="spin" />
            <span>Loading active EAs…</span>
          </div>

        {:else if activeEAs.length === 0}
          <div class="empty-state">
            <Inbox size={28} />
            <span>No active EAs deployed</span>
            <span class="empty-sub">Deploy a strategy to see it here</span>
          </div>

        {:else}
          <div class="tile-grid">
            {#each activeEAs as ea}
              <div class="tile ea-tile">
                <div class="tile-top">
                  <svelte:component this={getStatusIcon(ea.status)} size={13} style="color: {getStatusColor(ea.status)}" />
                  <span class="ea-status" style="color: {getStatusColor(ea.status)}">{ea.status}</span>
                  <span class="ea-symbol">{ea.symbol}</span>
                </div>
                <h3 class="tile-title">{ea.name}</h3>
                <div class="tile-meta">
                  <span class="pnl" class:positive={ea.pnl_today >= 0} class:negative={ea.pnl_today < 0}>
                    <TrendingUp size={11} />
                    {formatPnl(ea.pnl_today)} today
                  </span>
                  <span class="last-mod">
                    <Clock size={10} />
                    {ea.last_modified}
                  </span>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <!-- =========================================================
       Agent Insights strip (always visible, collapsible)
       ========================================================= -->
  <div class="agent-insights-strip" class:expanded={insightsExpanded}>
    <button class="insights-toggle" onclick={() => { insightsExpanded = !insightsExpanded; }}>
      <Lightbulb size={12} />
      <span>Agent Insights</span>
      {#if insightsUnread > 0}
        <span class="unread-badge">{insightsUnread}</span>
      {/if}
      <span class:rotated={insightsExpanded} style="display:inline-flex;align-items:center;transition:transform 0.2s;"><ChevronDown size={12} /></span>
    </button>
    {#if insightsExpanded}
      <AgentTilePanel
        canvas="development"
        maxHeight="200px"
        showHeader={false}
        onUnreadCount={(n: number) => { insightsUnread = n; }}
      />
    {/if}
  </div>

</div>

<style>
  /* =========================================================
     Shell
     ========================================================= */
  .development-canvas {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--color-bg-surface);
    backdrop-filter: var(--glass-blur);
    overflow: hidden;
  }

  /* =========================================================
     Canvas Header
     ========================================================= */
  .canvas-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.12);
    background: rgba(8, 13, 20, 0.5);
    backdrop-filter: blur(12px);
    flex-shrink: 0;
    flex-wrap: wrap;
    gap: 8px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .canvas-header :global(.dept-icon) {
    color: #00d4ff;
  }

  .canvas-title {
    font-family: var(--font-heading);
    font-size: var(--text-base);
    font-weight: 700;
    color: #00d4ff;
    margin: 0;
    letter-spacing: -0.01em;
  }

  .dept-badge {
    padding: 2px 8px;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    font-family: var(--font-ambient);
    font-size: 10px;
    color: rgba(0, 212, 255, 0.7);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  /* Tab nav */
  .tab-nav {
    display: flex;
    gap: 4px;
    margin-left: auto;
    flex-wrap: wrap;
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: var(--color-text-muted);
    font-family: var(--font-ambient);
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    white-space: nowrap;
  }

  .tab-btn:hover {
    color: var(--color-text-secondary);
    background: rgba(255, 255, 255, 0.04);
  }

  .tab-btn.active {
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.06);
    border-color: rgba(0, 212, 255, 0.2);
  }

  /* =========================================================
     Canvas Body
     ========================================================= */
  .canvas-body {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: height 0.2s ease;
  }

  /* =========================================================
     Tile Grid (Active EAs)
     ========================================================= */
  .tile-grid-wrapper {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  .tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
  }

  .tile {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(12px) saturate(160%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 8px;
    padding: 16px;
    cursor: pointer;
    transition: border-color 0.15s, transform 0.12s;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .tile:hover {
    border-color: rgba(0, 212, 255, 0.4);
    transform: translateY(-1px);
  }

  .tile-top {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .ea-status {
    font-family: var(--font-ambient);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .ea-symbol {
    margin-left: auto;
    font-family: var(--font-ambient);
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-weight: 600;
  }

  .tile-title {
    font-family: var(--font-heading);
    font-size: 13px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    margin: 0;
    line-height: 1.3;
  }

  .tile-meta {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: auto;
  }

  .pnl {
    display: flex;
    align-items: center;
    gap: 4px;
    font-family: var(--font-ambient);
    font-size: 11px;
    font-weight: 600;
  }

  .pnl.positive { color: #22c55e; }
  .pnl.negative { color: #ef4444; }

  .last-mod {
    display: flex;
    align-items: center;
    gap: 4px;
    font-family: var(--font-ambient);
    font-size: 10px;
    color: rgba(255, 255, 255, 0.35);
  }

  /* =========================================================
     Empty State
     ========================================================= */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    height: 200px;
    color: rgba(255, 255, 255, 0.35);
    font-family: var(--font-ambient);
    font-size: 13px;
  }

  .empty-sub {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.2);
  }

  /* =========================================================
     Variants pane (split: browser + Monaco)
     ========================================================= */
  .variants-wrapper {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    overflow: hidden;
  }

  .variant-browser-pane {
    border-right: 1px solid rgba(0, 212, 255, 0.1);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .monaco-pane {
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* =========================================================
     Backtest pane
     ========================================================= */
  .backtest-wrapper {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Backtest header row */
  .backtest-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    flex-shrink: 0;
  }

  .backtest-header :global(.backtest-header-icon) {
    color: #00d4ff;
  }

  .backtest-header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .backtest-header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.35);
    margin-left: 4px;
  }

  /* Empty / loading state */
  .backtest-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    flex: 1;
    min-height: 200px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }

  .backtest-empty-state :global(.backtest-empty-icon) {
    color: rgba(0, 212, 255, 0.25);
  }

  .backtest-empty-label {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.5);
  }

  .backtest-empty-sub {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.2);
    text-align: center;
    max-width: 340px;
    line-height: 1.5;
  }

  /* Strategy cards list */
  .backtest-cards {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  /* Individual strategy card */
  .backtest-card {
    background: rgba(8, 13, 20, 0.92);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 212, 255, 0.14);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    transition: border-color 0.15s;
  }

  .backtest-card:hover {
    border-color: rgba(0, 212, 255, 0.28);
  }

  /* Card header */
  .backtest-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .backtest-card-header :global(.backtest-card-icon) {
    color: #00d4ff;
    flex-shrink: 0;
  }

  .backtest-card-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
  }

  .backtest-card-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.28);
    margin-left: auto;
  }

  /* Pipeline row */
  .pipeline-stages {
    display: flex;
    align-items: flex-start;
    gap: 0;
    flex-wrap: wrap;
  }

  .stage-connector {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: rgba(0, 212, 255, 0.3);
    padding: 0 8px;
    align-self: center;
    margin-top: -8px; /* visual centering against badge */
    flex-shrink: 0;
  }

  .pipeline-stage {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 5px;
    min-width: 100px;
  }

  .stage-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: rgba(255, 255, 255, 0.35);
    font-weight: 600;
  }

  /* Stage badges */
  .stage-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
  }

  .badge-waiting {
    background: rgba(107, 114, 128, 0.15);
    color: rgba(107, 114, 128, 0.7);
    border: 1px solid rgba(107, 114, 128, 0.2);
  }

  .badge-running {
    background: rgba(240, 165, 0, 0.12);
    color: #f0a500;
    border: 1px solid rgba(240, 165, 0, 0.25);
  }

  .badge-passed {
    background: rgba(0, 200, 150, 0.16);
    color: #00c896;
    border: 1px solid rgba(0, 200, 150, 0.3);
  }

  .badge-failed {
    background: rgba(255, 59, 59, 0.14);
    color: #ff3b3b;
    border: 1px solid rgba(255, 59, 59, 0.28);
  }

  /* Pulsing amber dot for in-progress */
  .pulse-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #f0a500;
    animation: pulse-amber 1.2s ease-in-out infinite;
    flex-shrink: 0;
  }

  @keyframes pulse-amber {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.45; transform: scale(0.75); }
  }

  /* Small WR/DD stat below a stage badge */
  .stage-stat {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
  }

  /* Monte Carlo extended stats */
  .mc-stats {
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin-top: 2px;
  }

  .mc-stat {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.45);
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .mc-stat.mc-stat-pass {
    color: #00c896;
  }

  .mc-stat.mc-stat-fail {
    color: #ff3b3b;
  }

  .mc-stat.mc-stat-dd {
    color: rgba(240, 165, 0, 0.8);
  }

  .mc-threshold {
    color: rgba(255, 255, 255, 0.25);
    font-size: 9px;
  }

  /* =========================================================
     Pipeline pane (PipelineBoard + DepartmentKanban sub-section)
     ========================================================= */
  .pipeline-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .pipeline-board-section {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .kanban-sub-section {
    flex-shrink: 0;
    max-height: 320px;
    display: flex;
    flex-direction: column;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
    overflow: hidden;
  }

  .sub-section-label {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: rgba(8, 13, 20, 0.45);
    border-bottom: 1px solid rgba(0, 212, 255, 0.08);
    font-family: var(--font-ambient);
    font-size: 11px;
    color: rgba(0, 212, 255, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    flex-shrink: 0;
  }

  .kanban-sub-section :global(.department-kanban) {
    flex: 1;
    min-height: 0;
  }

  /* =========================================================
     Kanban wrapper (dept-tasks tab)
     ========================================================= */
  .kanban-wrapper {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    width: 100%;
    min-width: 0;
  }

  /* =========================================================
     Workflows builder pane
     ========================================================= */
  .workflows-builder-layout {
    display: flex;
    flex-direction: row;
    flex: 1;
    height: 0; /* forces flex child to respect parent height */
    overflow: hidden;
  }

  .workflows-template-panel {
    width: 280px;
    min-width: 280px;
    display: flex;
    flex-direction: column;
    background: rgba(8, 13, 20, 0.4);
    border-right: 1px solid rgba(0, 212, 255, 0.1);
    overflow: hidden;
  }

  .workflows-template-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    color: #00d4ff;
    font-family: var(--font-ambient);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    background: rgba(0, 212, 255, 0.05);
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    flex-shrink: 0;
  }

  .workflows-template-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    scrollbar-width: thin;
    scrollbar-color: rgba(0, 212, 255, 0.2) transparent;
  }

  .workflows-template-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    width: 100%;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 7px;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s ease, border-color 0.15s ease;
  }

  .workflows-template-item:hover {
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.2);
  }

  .workflows-template-item.blank-btn {
    border-color: rgba(255, 255, 255, 0.1);
    margin-bottom: 4px;
  }

  .workflows-template-item.blank-btn:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.15);
  }

  .workflows-template-name {
    font-family: var(--font-ambient);
    font-size: 12px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .workflows-template-desc {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
    line-height: 1.3;
  }

  .workflows-editor-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: rgba(8, 13, 20, 0.95);
  }

  .workflows-editor-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: rgba(8, 13, 20, 0.6);
    border-top: 1px solid rgba(0, 212, 255, 0.1);
    flex-shrink: 0;
  }

  .workflows-toolbar-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 5px;
    border: 1px solid transparent;
    font-family: var(--font-ambient);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
  }

  .workflows-toolbar-btn.run-btn {
    background: rgba(34, 197, 94, 0.15);
    border-color: rgba(34, 197, 94, 0.3);
    color: #22c55e;
  }
  .workflows-toolbar-btn.run-btn:hover {
    background: rgba(34, 197, 94, 0.25);
    border-color: rgba(34, 197, 94, 0.5);
  }

  .workflows-toolbar-btn.save-btn {
    background: rgba(0, 212, 255, 0.12);
    border-color: rgba(0, 212, 255, 0.25);
    color: #00d4ff;
  }
  .workflows-toolbar-btn.save-btn:hover {
    background: rgba(0, 212, 255, 0.22);
    border-color: rgba(0, 212, 255, 0.45);
  }

  .workflows-toolbar-btn.clear-btn {
    background: rgba(100, 116, 139, 0.1);
    border-color: rgba(100, 116, 139, 0.2);
    color: #6b7280;
  }
  .workflows-toolbar-btn.clear-btn:hover {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.25);
    color: #ef4444;
  }

  /* =========================================================
     Spin utility
     ========================================================= */
  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* =========================================================
     Agent Insights strip
     ========================================================= */

  .agent-insights-strip {
    flex-shrink: 0;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(8, 13, 20, 0.85);
  }

  .insights-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 6px 16px;
    background: transparent;
    border: none;
    color: rgba(224, 224, 224, 0.45);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: color 0.15s, background 0.15s;
  }

  .insights-toggle:hover {
    color: rgba(224, 224, 224, 0.75);
    background: rgba(255, 255, 255, 0.03);
  }

  .agent-insights-strip.expanded .insights-toggle {
    color: #00d4ff;
  }

  .unread-badge {
    padding: 1px 6px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 10px;
    font-size: 10px;
    color: #00d4ff;
    font-weight: 700;
    line-height: 1.4;
  }

  .insights-toggle .rotated {
    transform: rotate(180deg);
  }
</style>
