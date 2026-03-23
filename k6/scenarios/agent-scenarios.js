/**
 * Agent endpoint load test scenarios.
 *
 * Tests:
 * - GET /api/agents/stream - Agent SSE stream
 * - GET /api/agents/{id}/health - Agent health
 * - GET /api/agents/activity - Agent activity
 * - GET /api/agents/metrics - Agent metrics
 */

import { check } from 'k6';
import http from 'k6/http';
import { Trend, Rate } from 'k6/metrics';
import { getBaseUrl, getDefaultHeaders, getAuthHeaders } from '../scripts/shared/auth.js';
import { checkResponse, checkValidJson, checkLatency } from '../scripts/shared/checks.js';
import { randomBotId, correlationId } from '../scripts/shared/payload-generator.js';

const BASE_URL = getBaseUrl();
const HEADERS = getDefaultHeaders();
const AUTH_HEADERS = getAuthHeaders();

// Custom metrics for agent endpoints
const agentStreamLatency = new Trend('agent_stream_latency');
const agentHealthLatency = new Trend('agent_health_latency');
const agentActivityLatency = new Trend('agent_activity_latency');
const agentMetricsLatency = new Trend('agent_metrics_latency');

const agentErrorRate = new Rate('agent_errors');

/**
 * Test agent stream endpoint (SSE).
 */
export function testAgentStream() {
    const url = `${BASE_URL}/api/agents/stream`;

    const response = http.get(url, {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'agent_stream' }
    });

    agentStreamLatency.add(response.timings.duration);

    // SSE endpoint returns 200 but may have no events in test environment
    const passed = response.status === 200;

    agentErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test agent health endpoint.
 */
export function testAgentHealth() {
    const agentId = randomBotId();
    const url = `${BASE_URL}/api/agents/${agentId}/health`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'agent_health' }
    });

    agentHealthLatency.add(response.timings.duration);

    // 200 = success, 404 = agent not found (acceptable in test env)
    const passed = response.status === 200 || response.status === 404;

    agentErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test agent activity endpoint.
 */
export function testAgentActivity() {
    const url = `${BASE_URL}/api/agents/activity`;

    const response = http.get(url, {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'agent_activity' }
    });

    agentActivityLatency.add(response.timings.duration);

    const passed = response.status === 200 || response.status === 401;

    agentErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test agent metrics endpoint.
 */
export function testAgentMetrics() {
    const url = `${BASE_URL}/api/agents/metrics`;

    const response = http.get(url, {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'agent_metrics' }
    });

    agentMetricsLatency.add(response.timings.duration);

    const passed = response.status === 200 || response.status === 401;

    agentErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Run all agent scenario tests.
 */
export function runAgentScenarios() {
    testAgentStream();
    testAgentHealth();
    testAgentActivity();
    testAgentMetrics();
}

// Default export for k6 scenario executor
export default function() {
    runAgentScenarios();
}
