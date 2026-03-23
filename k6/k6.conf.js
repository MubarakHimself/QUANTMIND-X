/**
 * Main k6 configuration for QUANTMINDX load tests.
 *
 * This configuration ties together all scenarios and load profiles
 * for comprehensive API load testing.
 *
 * Usage:
 *   k6 run k6/k6.conf.js
 *   k6 run k6/k6.conf.js --env PROFILE=sustained
 *   k6 run k6/k6.conf.js --env K6_BASE_URL=http://api.example.com
 *   k6 run k6/k6.conf.js --out json=results.json
 */

import { getBaseUrl, isAuthConfigured } from './scripts/shared/auth.js';

// Load profiles
import { rampUpProfile } from './load-profiles/ramp-up.js';
import { sustainedLoadProfile } from './load-profiles/sustained-load.js';
import { spikeTestProfile } from './load-profiles/spike-test.js';

// Scenarios
import { runTradingScenarios } from './scenarios/trading-scenarios.js';
import { runKillSwitchScenarios } from './scenarios/kill-switch-scenarios.js';
import { runHealthScenarios } from './scenarios/health-scenarios.js';
import { runRiskScenarios } from './scenarios/risk-scenarios.js';
import { runFloorManagerScenarios } from './scenarios/floor-manager-scenarios.js';
import { runAgentScenarios } from './scenarios/agent-scenarios.js';

// Determine which profile to use
const PROFILE = __ENV.PROFILE || 'sustained';

// Get base URL for configuration
const BASE_URL = getBaseUrl();
const HAS_AUTH = isAuthConfigured();

console.log(`K6 Load Test Configuration:`);
console.log(`  Base URL: ${BASE_URL}`);
console.log(`  Auth configured: ${HAS_AUTH}`);
console.log(`  Load profile: ${PROFILE}`);

/**
 * Get scenarios configuration based on profile.
 */
function getScenarios() {
    const scenarios = {};

    // Trading scenarios - run on all profiles
    scenarios.trading_scenarios = {
        executor: 'shared-iterations',
        iterations: 10,
        maxDuration: '60s',
        scenarios: runTradingScenarios,
        tags: { category: 'trading' },
    };

    // Kill switch scenarios - run on all profiles
    scenarios.kill_switch_scenarios = {
        executor: 'shared-iterations',
        iterations: 10,
        maxDuration: '60s',
        scenarios: runKillSwitchScenarios,
        tags: { category: 'kill_switch' },
    };

    // Health scenarios - run on all profiles (lightweight)
    scenarios.health_scenarios = {
        executor: 'shared-iterations',
        iterations: 20,
        maxDuration: '60s',
        scenarios: runHealthScenarios,
        tags: { category: 'health' },
    };

    // Risk scenarios - run on all profiles
    scenarios.risk_scenarios = {
        executor: 'shared-iterations',
        iterations: 10,
        maxDuration: '60s',
        scenarios: runRiskScenarios,
        tags: { category: 'risk' },
    };

    // Floor manager scenarios - run on all profiles
    scenarios.floor_manager_scenarios = {
        executor: 'shared-iterations',
        iterations: 5,
        maxDuration: '60s',
        scenarios: runFloorManagerScenarios,
        tags: { category: 'floor_manager' },
    };

    // Agent scenarios - run on all profiles
    scenarios.agent_scenarios = {
        executor: 'shared-iterations',
        iterations: 5,
        maxDuration: '60s',
        scenarios: runAgentScenarios,
        tags: { category: 'agents' },
    };

    return scenarios;
}

/**
 * Get duration-based executor for load profiles.
 */
function getLoadProfileScenarios() {
    switch (PROFILE) {
        case 'rampup':
            return rampUpProfile.scenarios;

        case 'spike':
            return spikeTestProfile.scenarios;

        case 'sustained':
        default:
            return sustainedLoadProfile.scenarios;
    }
}

// Main execution function
export function executeScenarios() {
    // Run trading scenarios
    runTradingScenarios();

    // Run kill switch scenarios
    runKillSwitchScenarios();

    // Run health scenarios
    runHealthScenarios();

    // Run risk scenarios
    runRiskScenarios();

    // Run floor manager scenarios
    runFloorManagerScenarios();

    // Run agent scenarios
    runAgentScenarios();
}

// k6 Options
export const options = {
    // Disable open connection limit warning for WebSocket tests
    discardResponseBodies: false,

    // Scenarios configuration
    scenarios: getScenarios(),

    // Thresholds for pass/fail
    thresholds: {
        // HTTP request duration thresholds
        'http_req_duration': ['p(95)<500'],           // 95% of requests under 500ms
        'http_req_duration': ['p(99)<1000'],          // 99% of requests under 1s

        // Trading endpoint thresholds
        'trading_backtest_run_latency': ['p(95)<30000'],  // Backtest can be slow
        'trading_status_latency': ['p(95)<500'],
        'trading_bot_status_latency': ['p(95)<500'],
        'trading_close_position_latency': ['p(95)<2000'],
        'trading_emergency_stop_latency': ['p(95)<1000'],

        // Kill switch thresholds
        'killswitch_status_latency': ['p(95)<200'],
        'killswitch_trigger_latency': ['p(95)<1000'],
        'killswitch_health_latency': ['p(95)<100'],

        // Health endpoint thresholds
        'health_full_latency': ['p(95)<500'],
        'health_api_latency': ['p(95)<100'],

        // Risk endpoint thresholds
        'risk_regime_latency': ['p(95)<500'],
        'risk_params_latency': ['p(95)<200'],
        'risk_compliance_latency': ['p(95)<500'],
        'risk_physics_latency': ['p(95)<1000'],

        // Floor manager thresholds
        'floor_manager_chat_latency': ['p(95)<5000'],  // AI responses can be slow
        'floor_manager_status_latency': ['p(95)<500'],

        // Error rate thresholds
        'trading_errors': ['rate<0.05'],               // Less than 5% errors
        'killswitch_errors': ['rate<0.05'],
        'health_errors': ['rate<0.05'],
        'risk_errors': ['rate<0.05'],
        'floor_manager_errors': ['rate<0.05'],
        'agent_errors': ['rate<0.05'],
    },

    // Tags for filtering results
    tags: {
        service: 'quantmindx',
        environment: __ENV.K6_ENVIRONMENT || 'test',
    },
};

// Summary writer for results
export function handleSummary(data) {
    return {
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
        `_bmad-output/test-artifacts/k6-results-${Date.now()}.json`: JSON.stringify(data),
    };
}

// Text summary formatter
function textSummary(data, opts) {
    const indent = opts.indent || '';
    let output = '\n';

    output += `${indent}==================================================\n`;
    output += `${indent}  K6 LOAD TEST SUMMARY\n`;
    output += `${indent}==================================================\n\n`;

    // Duration and VUs
    output += `${indent}Test Configuration:\n`;
    output += `${indent}  Duration: ${data.state.testDuration || 'N/A'}\n`;
    output += `${indent}  VUs: ${data.state.vus || 0}\n`;
    output += `${indent}  Base URL: ${BASE_URL}\n`;
    output += `${indent}  Auth: ${HAS_AUTH ? 'Configured' : 'Not configured'}\n`;
    output += `${indent}  Profile: ${PROFILE}\n\n`;

    // Request statistics
    output += `${indent}Request Statistics:\n`;
    const httpStats = data.metrics.http_req_duration;
    if (httpStats) {
        output += `${indent}  Total Requests: ${httpStats.values.count}\n`;
        output += `${indent}  Failed Requests: ${data.metrics.http_req_failed?.values?.passes || 0}\n`;
        output += `${indent}  P50 (Median): ${httpStats.values['p(50)']?.toFixed(2)}ms\n`;
        output += `${indent}  P95: ${httpStats.values['p(95)']?.toFixed(2)}ms\n`;
        output += `${indent}  P99: ${httpStats.values['p(99)']?.toFixed(2)}ms\n`;
        output += `${indent}  Max: ${httpStats.values.max?.toFixed(2)}ms\n\n`;
    }

    // Custom metrics
    output += `${indent}Endpoint Latencies (P95):\n`;
    const p95Metrics = [
        'trading_backtest_run_latency',
        'trading_status_latency',
        'killswitch_status_latency',
        'killswitch_trigger_latency',
        'health_full_latency',
        'risk_regime_latency',
        'floor_manager_chat_latency',
    ];

    for (const metric of p95Metrics) {
        if (data.metrics[metric]) {
            const value = data.metrics[metric].values['p(95)'];
            const status = value < 500 ? 'PASS' : 'FAIL';
            output += `${indent}  ${metric}: ${value?.toFixed(2) || 'N/A'}ms [${status}]\n`;
        }
    }

    // Error rates
    output += `\n${indent}Error Rates:\n`;
    const errorMetrics = [
        'trading_errors',
        'killswitch_errors',
        'health_errors',
        'risk_errors',
    ];

    for (const metric of errorMetrics) {
        if (data.metrics[metric]) {
            const rate = (data.metrics[metric].values.rate * 100).toFixed(2);
            const status = parseFloat(rate) < 5 ? 'PASS' : 'FAIL';
            output += `${indent}  ${metric}: ${rate}% [${status}]\n`;
        }
    }

    output += `\n${indent}==================================================\n`;
    output += `${indent}  END OF SUMMARY\n`;
    output += `${indent}==================================================\n`;

    return output;
}

// Default export for k6
export default executeScenarios;
