/**
 * Risk endpoint load test scenarios.
 *
 * Tests:
 * - GET /api/risk/regime - Regime classification
 * - GET /api/risk/params/{account_tag} - Risk parameters
 * - PUT /api/risk/params/{account_tag} - Update risk params
 * - GET /api/risk/compliance - Compliance status
 * - GET /api/risk/islamic-status - Islamic compliance
 * - GET /api/risk/physics - Physics sensor outputs
 * - GET /api/risk/calendar/rules - Calendar rules
 * - GET /api/risk/calendar/events - Calendar events
 */

import { check } from 'k6';
import http from 'k6/http';
import { Trend, Rate } from 'k6/metrics';
import { getBaseUrl, getDefaultHeaders, getAuthHeaders } from '../scripts/shared/auth.js';
import { checkResponse, checkValidJson, checkLatency, checkRequiredFields, checkCircuitBreakerState } from '../scripts/shared/checks.js';
import { randomAccountTag, correlationId } from '../scripts/shared/payload-generator.js';

const BASE_URL = getBaseUrl();
const HEADERS = getDefaultHeaders();
const AUTH_HEADERS = getAuthHeaders();

// Custom metrics for risk endpoints
const regimeLatency = new Trend('risk_regime_latency');
const riskParamsLatency = new Trend('risk_params_latency');
const complianceLatency = new Trend('risk_compliance_latency');
const islamicStatusLatency = new Trend('risk_islamic_latency');
const physicsLatency = new Trend('risk_physics_latency');
const calendarRulesLatency = new Trend('risk_calendar_rules_latency');
const calendarEventsLatency = new Trend('risk_calendar_events_latency');

const riskErrorRate = new Rate('risk_errors');

/**
 * Test regime endpoint.
 */
export function testRegime() {
    const url = `${BASE_URL}/api/risk/regime`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'risk_regime' }
    });

    regimeLatency.add(response.timings.duration);

    const passed = checkResponse('risk_regime_check', response, 200) &&
                   checkValidJson('risk_regime_json', response) &&
                   checkRequiredFields('risk_regime_fields', response, [
                       'regime',
                       'confidence_pct'
                   ]);

    riskErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test risk params GET endpoint.
 */
export function testRiskParams() {
    const accountTag = randomAccountTag();
    const url = `${BASE_URL}/api/risk/params/${accountTag}`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'risk_params' }
    });

    riskParamsLatency.add(response.timings.duration);

    const passed = checkResponse('risk_params_check', response, 200) &&
                   checkValidJson('risk_params_json', response);

    riskErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test risk params PUT endpoint (requires auth).
 */
export function testRiskParamsUpdate() {
    const accountTag = randomAccountTag();
    const url = `${BASE_URL}/api/risk/params/${accountTag}`;

    const payload = {
        daily_loss_cap_pct: 5.0,
        max_trades_per_day: 10,
        kelly_fraction: 0.5
    };

    const response = http.put(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'risk_params_update' }
    });

    riskParamsLatency.add(response.timings.duration);

    // 200 = success, 401 = no auth, 422 = validation error (acceptable)
    const passed = response.status === 200 ||
                   response.status === 401 ||
                   response.status === 422;

    riskErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test compliance endpoint.
 */
export function testCompliance() {
    const url = `${BASE_URL}/api/risk/compliance`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'risk_compliance' }
    });

    complianceLatency.add(response.timings.duration);

    const passed = checkResponse('risk_compliance_check', response, 200) &&
                   checkValidJson('risk_compliance_json', response) &&
                   checkRequiredFields('risk_compliance_fields', response, [
                       'overall_status',
                       'account_tags'
                   ]);

    // Check circuit breaker state
    const cbState = checkCircuitBreakerState(response);
    if (cbState.triggered) {
        console.log('Circuit breaker triggered detected in compliance response');
    }

    riskErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test Islamic compliance status endpoint.
 */
export function testIslamicStatus() {
    const url = `${BASE_URL}/api/risk/islamic-status`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'risk_islamic_status' }
    });

    islamicStatusLatency.add(response.timings.duration);

    const passed = checkResponse('risk_islamic_check', response, 200) &&
                   checkValidJson('risk_islamic_json', response);

    riskErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test physics sensor outputs endpoint.
 */
export function testPhysics() {
    const url = `${BASE_URL}/api/risk/physics`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'risk_physics' }
    });

    physicsLatency.add(response.timings.duration);

    const passed = checkResponse('risk_physics_check', response, 200) &&
                   checkValidJson('risk_physics_json', response) &&
                   checkRequiredFields('risk_physics_fields', response, [
                       'ising',
                       'lyapunov',
                       'hmm'
                   ]);

    riskErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test calendar rules list endpoint.
 */
export function testCalendarRules() {
    const url = `${BASE_URL}/api/risk/calendar/rules`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'risk_calendar_rules' }
    });

    calendarRulesLatency.add(response.timings.duration);

    const passed = checkResponse('risk_calendar_rules_check', response, 200);

    riskErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test calendar events list endpoint.
 */
export function testCalendarEvents() {
    const url = `${BASE_URL}/api/risk/calendar/events`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'risk_calendar_events' }
    });

    calendarEventsLatency.add(response.timings.duration);

    const passed = checkResponse('risk_calendar_events_check', response, 200);

    riskErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Run all risk scenario tests.
 */
export function runRiskScenarios() {
    testRegime();
    testRiskParams();
    testRiskParamsUpdate();
    testCompliance();
    testIslamicStatus();
    testPhysics();
    testCalendarRules();
    testCalendarEvents();
}

// Default export for k6 scenario executor
export default function() {
    runRiskScenarios();
}
