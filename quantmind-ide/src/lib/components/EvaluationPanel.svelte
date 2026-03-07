<script lang="ts">
  import { onMount } from "svelte";
  import { CheckCircle, XCircle, Loader, Play, Clock, DollarSign, BarChart3, FlaskConical } from "lucide-svelte";
  import * as evaluationApi from "$lib/api/evaluation";

  // State
  let loading = false;
  let error = "";
  let report: evaluationApi.EvaluationReport | null = null;
  let criteriaTypes: evaluationApi.CriteriaInfo[] = [];

  // Form state
  let testCaseName = "Basic Test";
  let inputDataKey = "value";
  let inputDataValue = "5";
  let expectedOutputKey = "result";
  let expectedOutputValue = "10";
  let selectedCriteria = "partial";
  let threshold = 0.8;
  let runParallel = true;

  // Benchmark state
  let benchmarkLoading = false;
  let benchmarkResult: evaluationApi.BenchmarkResult | null = null;
  let benchmarkIterations = 10;

  onMount(async () => {
    await loadCriteriaTypes();
  });

  async function loadCriteriaTypes() {
    try {
      const result = await evaluationApi.getCriteriaTypes();
      criteriaTypes = result.criteria;
    } catch (e) {
      console.error("Failed to load criteria types:", e);
    }
  }

  async function runEvaluation() {
    loading = true;
    error = "";
    report = null;

    try {
      const testCase = evaluationApi.createTestCase(
        `test_${Date.now()}`,
        testCaseName,
        { [inputDataKey]: parseFloat(inputDataValue) || inputDataValue },
        { [expectedOutputKey]: parseFloat(expectedOutputValue) || expectedOutputValue },
        { timeoutSeconds: 30 }
      );

      const request: evaluationApi.EvaluationRequest = {
        test_cases: [testCase],
        criteria_type: selectedCriteria as 'exact' | 'partial' | 'threshold',
        threshold,
        parallel: runParallel,
        cost_per_token: 0.001
      };

      report = await evaluationApi.runEvaluation(request);
    } catch (e) {
      error = e instanceof Error ? e.message : "Evaluation failed";
    } finally {
      loading = false;
    }
  }

  async function runBenchmark() {
    benchmarkLoading = true;
    benchmarkResult = null;

    try {
      const config = evaluationApi.createBenchmarkConfig(
        "latency_test",
        {
          description: "Agent latency benchmark",
          iterations: benchmarkIterations,
          warmupIterations: 2
        }
      );

      const request: evaluationApi.BenchmarkRequest = {
        config,
        workload_type: "latency"
      };

      benchmarkResult = await evaluationApi.runBenchmark(request);
    } catch (e) {
      error = e instanceof Error ? e.message : "Benchmark failed";
    } finally {
      benchmarkLoading = false;
    }
  }

  function clearResults() {
    report = null;
    benchmarkResult = null;
    error = "";
  }
</script>

<div class="evaluation-panel">
  <div class="panel-header-row">
    <h3>Agent Evaluation</h3>
    <button class="refresh-btn" on:click={clearResults}>
      Clear
    </button>
  </div>

  {#if error}
    <div class="error-banner">{error}</div>
  {/if}

  <div class="evaluation-grid">
    <!-- Evaluation Section -->
    <div class="eval-section">
      <div class="section-header">
        <FlaskConical size={14} />
        <span>Run Evaluation</span>
      </div>

      <div class="form-group">
        <label for="testName">Test Name</label>
        <input
          id="testName"
          type="text"
          bind:value={testCaseName}
          placeholder="Test name"
        />
      </div>

      <div class="form-row">
        <div class="form-group">
          <label for="inputKey">Input Key</label>
          <input
            id="inputKey"
            type="text"
            bind:value={inputDataKey}
            placeholder="value"
          />
        </div>
        <div class="form-group">
          <label for="inputValue">Input Value</label>
          <input
            id="inputValue"
            type="text"
            bind:value={inputDataValue}
            placeholder="5"
          />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label for="expectedKey">Expected Key</label>
          <input
            id="expectedKey"
            type="text"
            bind:value={expectedOutputKey}
            placeholder="result"
          />
        </div>
        <div class="form-group">
          <label for="expectedValue">Expected Value</label>
          <input
            id="expectedValue"
            type="text"
            bind:value={expectedOutputValue}
            placeholder="10"
          />
        </div>
      </div>

      <div class="form-group">
        <label for="criteria">Criteria</label>
        <select id="criteria" bind:value={selectedCriteria}>
          {#each criteriaTypes as criteria}
            <option value={criteria.type}>{criteria.type}</option>
          {/each}
          <option value="exact">exact</option>
          <option value="partial">partial</option>
          <option value="threshold">threshold</option>
        </select>
      </div>

      {#if selectedCriteria === 'threshold'}
        <div class="form-group">
          <label for="threshold">Threshold: {threshold}</label>
          <input
            id="threshold"
            type="range"
            min="0"
            max="1"
            step="0.1"
            bind:value={threshold}
          />
        </div>
      {/if}

      <div class="form-group checkbox">
        <label>
          <input type="checkbox" bind:checked={runParallel} />
          Run in parallel
        </label>
      </div>

      <button
        class="run-btn"
        on:click={runEvaluation}
        disabled={loading}
      >
        {#if loading}
          <Loader size={14} class="spinning" />
          Running...
        {:else}
          <Play size={14} />
          Run Evaluation
        {/if}
      </button>
    </div>

    <!-- Benchmark Section -->
    <div class="eval-section">
      <div class="section-header">
        <BarChart3 size={14} />
        <span>Run Benchmark</span>
      </div>

      <div class="form-group">
        <label for="iterations">Iterations</label>
        <input
          id="iterations"
          type="number"
          bind:value={benchmarkIterations}
          min="1"
          max="100"
        />
      </div>

      <button
        class="run-btn"
        on:click={runBenchmark}
        disabled={benchmarkLoading}
      >
        {#if benchmarkLoading}
          <Loader size={14} class="spinning" />
          Running...
        {:else}
          <Play size={14} />
          Run Benchmark
        {/if}
      </button>
    </div>
  </div>

  <!-- Results Section -->
  {#if report}
    <div class="results-section">
      <h4>Evaluation Results</h4>

      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-value">{report.total_tests}</span>
          <span class="stat-label">Total</span>
        </div>
        <div class="stat-card success">
          <span class="stat-value">{report.passed_tests}</span>
          <span class="stat-label">Passed</span>
        </div>
        <div class="stat-card error">
          <span class="stat-value">{report.failed_tests}</span>
          <span class="stat-label">Failed</span>
        </div>
        <div class="stat-card">
          <span class="stat-value">{(report.pass_rate * 100).toFixed(1)}%</span>
          <span class="stat-label">Pass Rate</span>
        </div>
      </div>

      <div class="metrics-row">
        <div class="metric">
          <Clock size={12} />
          <span>Avg Latency: {report.avg_latency_ms.toFixed(2)}ms</span>
        </div>
        <div class="metric">
          <DollarSign size={12} />
          <span>Total Cost: ${report.total_cost.toFixed(4)}</span>
        </div>
      </div>

      <div class="results-list">
        {#each report.results as result}
          <div class="result-item" class:passed={result.passed} class:failed={!result.passed}>
            <div class="result-header">
              {#if result.passed}
                <CheckCircle size={14} class="icon-success" />
              {:else}
                <XCircle size={14} class="icon-error" />
              {/if}
              <span class="result-id">{result.test_case_id}</span>
              <span class="result-latency">{result.latency_ms.toFixed(2)}ms</span>
            </div>
            {#if result.error}
              <div class="result-error">{result.error}</div>
            {:else}
              <div class="result-metrics">
                Accuracy: {(result.metrics.accuracy * 100).toFixed(1)}%
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  {#if benchmarkResult}
    <div class="results-section">
      <h4>Benchmark Results</h4>

      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-value">{benchmarkResult.iterations}</span>
          <span class="stat-label">Iterations</span>
        </div>
        <div class="stat-card">
          <span class="stat-value">{benchmarkResult.avg_latency_ms.toFixed(2)}</span>
          <span class="stat-label">Avg ms</span>
        </div>
        <div class="stat-card">
          <span class="stat-value">{benchmarkResult.min_latency_ms.toFixed(2)}</span>
          <span class="stat-label">Min ms</span>
        </div>
        <div class="stat-card">
          <span class="stat-value">{benchmarkResult.max_latency_ms.toFixed(2)}</span>
          <span class="stat-label">Max ms</span>
        </div>
      </div>

      <div class="metrics-row">
        <div class="metric">
          <span>Std Dev: {benchmarkResult.std_dev_ms.toFixed(2)}ms</span>
        </div>
        <div class="metric">
          <span>Throughput: {benchmarkResult.throughput.toFixed(2)} ops/s</span>
        </div>
        <div class="metric">
          <span>Errors: {benchmarkResult.errors}</span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .evaluation-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 1rem;
    gap: 1rem;
  }

  .panel-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .panel-header-row h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: var(--bg-tertiary, #1e293b);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.375rem;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.75rem;
    cursor: pointer;
  }

  .refresh-btn:hover {
    background: var(--bg-hover, #334155);
  }

  .error-banner {
    padding: 0.5rem 0.75rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.375rem;
    color: #fca5a5;
    font-size: 0.75rem;
  }

  .evaluation-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .eval-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 1rem;
    background: var(--bg-tertiary, #1e293b);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color, #334155);
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-secondary, #94a3b8);
    margin-bottom: 0.25rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .form-group label {
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
  }

  .form-group input[type="text"],
  .form-group input[type="number"],
  .form-group select {
    padding: 0.5rem;
    background: var(--bg-input, #0f172a);
    border: 1px solid var(--border-color, #334155);
    border-radius: 0.375rem;
    color: var(--text-primary, #e2e8f0);
    font-size: 0.75rem;
  }

  .form-group input[type="range"] {
    width: 100%;
  }

  .form-group.checkbox {
    flex-direction: row;
    align-items: center;
  }

  .form-group.checkbox label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-transform: none;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }

  .run-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.625rem 1rem;
    background: var(--accent-primary, #3b82f6);
    border: none;
    border-radius: 0.375rem;
    color: white;
    font-size: 0.75rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .run-btn:hover:not(:disabled) {
    background: var(--accent-hover, #2563eb);
  }

  .run-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .results-section {
    padding: 1rem;
    background: var(--bg-tertiary, #1e293b);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color, #334155);
  }

  .results-section h4 {
    margin: 0 0 0.75rem 0;
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-secondary, #94a3b8);
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }

  .stat-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.5rem;
    background: var(--bg-secondary, #111827);
    border-radius: 0.375rem;
  }

  .stat-card.success .stat-value {
    color: #22c55e;
  }

  .stat-card.error .stat-value {
    color: #ef4444;
  }

  .stat-value {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-primary, #3b82f6);
  }

  .stat-label {
    font-size: 0.625rem;
    color: var(--text-muted, #64748b);
    text-transform: uppercase;
  }

  .metrics-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.75rem;
  }

  .metric {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.6875rem;
    color: var(--text-secondary, #94a3b8);
  }

  .results-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .result-item {
    padding: 0.5rem;
    background: var(--bg-secondary, #111827);
    border-radius: 0.375rem;
    border-left: 3px solid;
  }

  .result-item.passed {
    border-left-color: #22c55e;
  }

  .result-item.failed {
    border-left-color: #ef4444;
  }

  .result-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .result-id {
    font-size: 0.75rem;
    font-weight: 500;
  }

  .result-latency {
    margin-left: auto;
    font-size: 0.6875rem;
    color: var(--text-muted, #64748b);
  }

  .result-metrics {
    font-size: 0.6875rem;
    color: var(--text-secondary, #94a3b8);
    margin-top: 0.25rem;
  }

  .result-error {
    font-size: 0.6875rem;
    color: #fca5a5;
    margin-top: 0.25rem;
  }

  .icon-success {
    color: #22c55e;
  }

  .icon-error {
    color: #ef4444;
  }
</style>
