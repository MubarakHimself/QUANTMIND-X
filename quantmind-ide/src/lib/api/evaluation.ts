/**
 * Evaluation API Client
 * Provides frontend API functions for agent evaluation endpoints
 *
 * Wraps the backend endpoints defined in src/api/evaluation_endpoints.py
 */

const API_BASE = '/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers
    }
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return response.json();
}

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Test case input for evaluation
 */
export interface TestCaseInput {
  id: string;
  name: string;
  input_data: Record<string, unknown>;
  expected_output: unknown;
  metadata?: Record<string, unknown>;
  timeout_seconds?: number;
  tags?: string[];
}

/**
 * Evaluation request parameters
 */
export interface EvaluationRequest {
  test_cases: TestCaseInput[];
  criteria_type?: 'exact' | 'partial' | 'threshold';
  threshold?: number;
  parallel?: boolean;
  stop_on_first_failure?: boolean;
  cost_per_token?: number;
}

/**
 * Single evaluation result
 */
export interface EvaluationResult {
  test_case_id: string;
  passed: boolean;
  actual_output: unknown;
  expected_output: unknown;
  metrics: Record<string, number>;
  latency_ms: number;
  error?: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

/**
 * Evaluation report
 */
export interface EvaluationReport {
  total_tests: number;
  passed_tests: number;
  failed_tests: number;
  pass_rate: number;
  avg_latency_ms: number;
  total_cost: number;
  results: EvaluationResult[];
  custom_metrics: Record<string, number>;
  timestamp: string;
}

/**
 * Benchmark configuration
 */
export interface BenchmarkConfigInput {
  name: string;
  description?: string;
  iterations?: number;
  warmup_iterations?: number;
  timeout_seconds?: number;
  parallel_workers?: number;
}

/**
 * Benchmark request
 */
export interface BenchmarkRequest {
  config: BenchmarkConfigInput;
  workload_type?: 'latency' | 'throughput';
  custom_workload?: Record<string, unknown>;
}

/**
 * Single benchmark result
 */
export interface BenchmarkResult {
  config_name: string;
  iterations: number;
  total_time_ms: number;
  avg_latency_ms: number;
  min_latency_ms: number;
  max_latency_ms: number;
  median_latency_ms: number;
  std_dev_ms: number;
  throughput: number;
  errors: number;
  error_rate: number;
  timestamp: string;
  metadata: Record<string, unknown>;
}

/**
 * Benchmark suite report
 */
export interface BenchmarkSuiteReport {
  suite_name: string;
  benchmarks: BenchmarkResult[];
  total_iterations: number;
  total_time_ms: number;
  avg_throughput: number;
  timestamp: string;
  summary: Record<string, unknown>;
}

/**
 * Criteria type info
 */
export interface CriteriaInfo {
  type: string;
  description: string;
}

/**
 * Health check response
 */
export interface HealthResponse {
  status: string;
  service: string;
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Run an evaluation with the provided test cases
 */
export async function runEvaluation(request: EvaluationRequest): Promise<EvaluationReport> {
  return apiFetch<EvaluationReport>('/evaluation/evaluate', {
    method: 'POST',
    body: JSON.stringify(request)
  });
}

/**
 * Run a single benchmark
 */
export async function runBenchmark(request: BenchmarkRequest): Promise<BenchmarkResult> {
  return apiFetch<BenchmarkResult>('/evaluation/benchmark', {
    method: 'POST',
    body: JSON.stringify(request)
  });
}

/**
 * Run multiple benchmarks as a suite
 */
export async function runBenchmarkSuite(configs: BenchmarkConfigInput[]): Promise<BenchmarkSuiteReport> {
  return apiFetch<BenchmarkSuiteReport>('/evaluation/benchmark/suite', {
    method: 'POST',
    body: JSON.stringify(configs)
  });
}

/**
 * Get available evaluation criteria types
 */
export async function getCriteriaTypes(): Promise<{ criteria: CriteriaInfo[] }> {
  return apiFetch<{ criteria: CriteriaInfo[] }>('/evaluation/criteria');
}

/**
 * Health check for evaluation service
 */
export async function checkEvaluationHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/evaluation/health');
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Create a test case for evaluation
 */
export function createTestCase(
  id: string,
  name: string,
  inputData: Record<string, unknown>,
  expectedOutput: unknown,
  options?: {
    metadata?: Record<string, unknown>;
    timeoutSeconds?: number;
    tags?: string[];
  }
): TestCaseInput {
  return {
    id,
    name,
    input_data: inputData,
    expected_output: expectedOutput,
    metadata: options?.metadata ?? {},
    timeout_seconds: options?.timeoutSeconds ?? 30.0,
    tags: options?.tags ?? []
  };
}

/**
 * Create a benchmark configuration
 */
export function createBenchmarkConfig(
  name: string,
  options?: {
    description?: string;
    iterations?: number;
    warmupIterations?: number;
    timeoutSeconds?: number;
    parallelWorkers?: number;
  }
): BenchmarkConfigInput {
  return {
    name,
    description: options?.description ?? '',
    iterations: options?.iterations ?? 10,
    warmup_iterations: options?.warmupIterations ?? 2,
    timeout_seconds: options?.timeoutSeconds ?? 60.0,
    parallel_workers: options?.parallelWorkers ?? 1
  };
}

/**
 * Format evaluation report for display
 */
export function formatEvaluationReport(report: EvaluationReport): string {
  const lines = [
    `Evaluation Report`,
    `=================`,
    `Total Tests: ${report.total_tests}`,
    `Passed: ${report.passed_tests}`,
    `Failed: ${report.failed_tests}`,
    `Pass Rate: ${(report.pass_rate * 100).toFixed(1)}%`,
    `Avg Latency: ${report.avg_latency_ms.toFixed(2)}ms`,
    `Total Cost: $${report.total_cost.toFixed(4)}`,
    ``,
    `Results:`,
    ...report.results.map(r =>
      `  ${r.passed ? '✓' : '✗'} ${r.test_case_id}: ${r.error ? r.error : `${(r.metrics.accuracy * 100).toFixed(1)}% accuracy, ${r.latency_ms.toFixed(2)}ms`}`
    )
  ];
  return lines.join('\n');
}

/**
 * Format benchmark result for display
 */
export function formatBenchmarkResult(result: BenchmarkResult): string {
  const lines = [
    `Benchmark: ${result.config_name}`,
    `Iterations: ${result.iterations}`,
    `Avg Latency: ${result.avg_latency_ms.toFixed(2)}ms`,
    `Min Latency: ${result.min_latency_ms.toFixed(2)}ms`,
    `Max Latency: ${result.max_latency_ms.toFixed(2)}ms`,
    `Std Dev: ${result.std_dev_ms.toFixed(2)}ms`,
    `Throughput: ${result.throughput.toFixed(2)} ops/sec`,
    `Errors: ${result.errors} (${(result.error_rate * 100).toFixed(1)}%)`
  ];
  return lines.join('\n');
}
