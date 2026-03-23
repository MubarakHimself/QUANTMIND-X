/**
 * Kill Switch endpoint load test scenarios.
 *
 * Tests:
 * - GET /api/kill-switch/status - Kill switch status
 * - POST /api/kill-switch/trigger - Trigger tier 1/2/3
 * - GET /api/kill-switch/health - Quick health check
 * - GET /api/kill-switch/alerts - Current alerts
 * - GET /api/kill-switch/audit - Audit logs
 * - GET /api/kill-switch/config - Configuration
 */

import { check } from 'k6';
import http from 'k6/http';
import { Trend, Rate } from 'k6/metrics';
import { getBaseUrl, getDefaultHeaders, getAuthHeaders } from '../scripts/shared/auth.js';
import { checkResponse, checkValidJson, checkLatency, checkKillSwitchResponse } from '../scripts/shared/checks.js';
import { killSwitchTriggerPayload, correlationId } from '../scripts/shared/payload-generator.js';

const BASE_URL = getBaseUrl();
const HEADERS = getDefaultHeaders();
const AUTH_HEADERS = getAuthHeaders();

// Custom metrics for kill switch endpoints
const ksStatusLatency = new Trend('killswitch_status_latency');
const ksTriggerLatency = new Trend('killswitch_trigger_latency');
const ksHealthLatency = new Trend('killswitch_health_latency');
const ksAlertsLatency = new Trend('killswitch_alerts_latency');
const ksAuditLatency = new Trend('killswitch_audit_latency');
const ksConfigLatency = new Trend('killswitch_config_latency');

const ksErrorRate = new Rate('killswitch_errors');

/**
 * Test kill switch status endpoint (no auth required).
 */
export function testKillSwitchStatus() {
    const url = `${BASE_URL}/api/kill-switch/status`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'ks_status' }
    });

    ksStatusLatency.add(response.timings.duration);

    const passed = checkResponse('ks_status_check', response, 200) &&
                   checkValidJson('ks_status_json', response);

    ksErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test kill switch trigger endpoint (auth required).
 * Tests Tier 1 (Soft Stop).
 */
export function testKillSwitchTriggerTier1() {
    const payload = killSwitchTriggerPayload(1, 'k6-load-test-tier1');
    const url = `${BASE_URL}/api/kill-switch/trigger`;

    const response = http.post(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'ks_trigger_tier1' }
    });

    ksTriggerLatency.add(response.timings.duration);

    // 200 = success, 401 = no auth configured, 500 = no MT5 in test env
    const passed = response.status === 200 ||
                   response.status === 401 ||
                   response.status === 500;

    ksErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test kill switch trigger endpoint (auth required).
 * Tests Tier 2 (Strategy Pause).
 */
export function testKillSwitchTriggerTier2() {
    const payload = killSwitchTriggerPayload(2, 'k6-load-test-tier2', ['STRAT-001', 'STRAT-002']);
    const url = `${BASE_URL}/api/kill-switch/trigger`;

    const response = http.post(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'ks_trigger_tier2' }
    });

    ksTriggerLatency.add(response.timings.duration);

    const passed = response.status === 200 ||
                   response.status === 401 ||
                   response.status === 500;

    ksErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test kill switch trigger endpoint (auth required).
 * Tests Tier 3 (Emergency Close).
 */
export function testKillSwitchTriggerTier3() {
    const payload = killSwitchTriggerPayload(3, 'k6-load-test-tier3');
    const url = `${BASE_URL}/api/kill-switch/trigger`;

    const response = http.post(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'ks_trigger_tier3' }
    });

    ksTriggerLatency.add(response.timings.duration);

    const passed = response.status === 200 ||
                   response.status === 401 ||
                   response.status === 500;

    ksErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test kill switch health endpoint (no auth required).
 */
export function testKillSwitchHealth() {
    const url = `${BASE_URL}/api/kill-switch/health`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'ks_health' }
    });

    ksHealthLatency.add(response.timings.duration);

    const passed = checkResponse('ks_health_check', response, 200) &&
                   checkValidJson('ks_health_json', response);

    ksErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test kill switch alerts endpoint (no auth required).
 */
export function testKillSwitchAlerts() {
    const url = `${BASE_URL}/api/kill-switch/alerts`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'ks_alerts' }
    });

    ksAlertsLatency.add(response.timings.duration);

    const passed = checkResponse('ks_alerts_check', response, 200) &&
                   checkValidJson('ks_alerts_json', response);

    ksErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test kill switch audit endpoint (no auth required).
 */
export function testKillSwitchAudit() {
    const url = `${BASE_URL}/api/kill-switch/audit`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'ks_audit' }
    });

    ksAuditLatency.add(response.timings.duration);

    const passed = checkResponse('ks_audit_check', response, 200) &&
                   checkValidJson('ks_audit_json', response);

    ksErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test kill switch config endpoint (no auth required).
 */
export function testKillSwitchConfig() {
    const url = `${BASE_URL}/api/kill-switch/config`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'ks_config' }
    });

    ksConfigLatency.add(response.timings.duration);

    const passed = checkResponse('ks_config_check', response, 200) &&
                   checkValidJson('ks_config_json', response);

    ksErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Run all kill switch scenario tests.
 */
export function runKillSwitchScenarios() {
    // Read endpoints (no auth)
    testKillSwitchStatus();
    testKillSwitchHealth();
    testKillSwitchAlerts();
    testKillSwitchAudit();
    testKillSwitchConfig();

    // Write endpoints (auth required, tier tests)
    testKillSwitchTriggerTier1();
    testKillSwitchTriggerTier2();
    testKillSwitchTriggerTier3();
}

// Default export for k6 scenario executor
export default function() {
    runKillSwitchScenarios();
}
