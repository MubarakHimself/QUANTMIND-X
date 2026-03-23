/**
 * Health endpoint load test scenarios.
 *
 * Tests:
 * - GET /health - Full health check
 * - GET /health/api - API health
 * - GET /health/mt5 - MT5 health
 * - GET /health/database - Database health
 * - GET /health/redis - Redis health
 * - GET /health/prometheus - Prometheus health
 */

import { check } from 'k6';
import http from 'k6/http';
import { Trend, Rate } from 'k6/metrics';
import { getBaseUrl, getDefaultHeaders } from '../scripts/shared/auth.js';
import { checkResponse, checkValidJson, checkLatency } from '../scripts/shared/checks.js';

const BASE_URL = getBaseUrl();
const HEADERS = getDefaultHeaders();

// Custom metrics for health endpoints
const fullHealthLatency = new Trend('health_full_latency');
const apiHealthLatency = new Trend('health_api_latency');
const mt5HealthLatency = new Trend('health_mt5_latency');
const dbHealthLatency = new Trend('health_database_latency');
const redisHealthLatency = new Trend('health_redis_latency');
const prometheusHealthLatency = new Trend('health_prometheus_latency');

const healthErrorRate = new Rate('health_errors');

/**
 * Test full health endpoint.
 */
export function testFullHealth() {
    const url = `${BASE_URL}/health`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'health_full' }
    });

    fullHealthLatency.add(response.timings.duration);

    const passed = checkResponse('health_full_check', response, 200) &&
                   checkValidJson('health_full_json', response) &&
                   checkRequiredFields('health_full_fields', response, [
                       'overall_status',
                       'services'
                   ]);

    healthErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test API health endpoint.
 */
export function testApiHealth() {
    const url = `${BASE_URL}/health/api`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'health_api' }
    });

    apiHealthLatency.add(response.timings.duration);

    const passed = checkResponse('health_api_check', response, 200) &&
                   checkValidJson('health_api_json', response) &&
                   checkRequiredFields('health_api_fields', response, [
                       'status'
                   ]);

    healthErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test MT5 health endpoint.
 */
export function testMt5Health() {
    const url = `${BASE_URL}/health/mt5`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'health_mt5' }
    });

    mt5HealthLatency.add(response.timings.duration);

    const passed = checkResponse('health_mt5_check', response, 200) &&
                   checkValidJson('health_mt5_json', response);

    healthErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test database health endpoint.
 */
export function testDatabaseHealth() {
    const url = `${BASE_URL}/health/database`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'health_database' }
    });

    dbHealthLatency.add(response.timings.duration);

    const passed = checkResponse('health_database_check', response, 200) &&
                   checkValidJson('health_database_json', response);

    healthErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test Redis health endpoint.
 */
export function testRedisHealth() {
    const url = `${BASE_URL}/health/redis`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'health_redis' }
    });

    redisHealthLatency.add(response.timings.duration);

    const passed = checkResponse('health_redis_check', response, 200) &&
                   checkValidJson('health_redis_json', response);

    healthErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test Prometheus health endpoint.
 */
export function testPrometheusHealth() {
    const url = `${BASE_URL}/health/prometheus`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'health_prometheus' }
    });

    prometheusHealthLatency.add(response.timings.duration);

    const passed = checkResponse('health_prometheus_check', response, 200) &&
                   checkValidJson('health_prometheus_json', response);

    healthErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Run all health scenario tests.
 */
export function runHealthScenarios() {
    testFullHealth();
    testApiHealth();
    testMt5Health();
    testDatabaseHealth();
    testRedisHealth();
    testPrometheusHealth();
}

// Default export for k6 scenario executor
export default function() {
    runHealthScenarios();
}
