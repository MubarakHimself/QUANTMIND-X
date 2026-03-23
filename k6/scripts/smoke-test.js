/**
 * Quick smoke test for QUANTMINDX API.
 *
 * Runs a lightweight test to quickly validate API availability
 * and basic functionality before running full load tests.
 *
 * Usage:
 *   k6 run k6/scripts/smoke-test.js
 */

import { check, group } from 'k6';
import http from 'k6/http';
import { Trend, Rate } from 'k6/metrics';
import { getBaseUrl, getDefaultHeaders, getAuthHeaders, isAuthConfigured } from './scripts/shared/auth.js';
import { checkResponse, checkValidJson } from './scripts/shared/checks.js';

const BASE_URL = getBaseUrl();
const HEADERS = getDefaultHeaders();
const AUTH_HEADERS = getAuthHeaders();

// Metrics
const smokeErrorRate = new Rate('smoke_errors');

/**
 * Smoke test for health endpoints.
 */
function testHealthEndpoints() {
    group('Health Endpoints', function() {
        // API health
        const apiRes = http.get(`${BASE_URL}/health/api`);
        check(apiRes, {
            'api health: status 200': (r) => r.status === 200,
            'api health: has status field': (r) => {
                try { return r.json().status === 'healthy' || r.json().status === 'degraded'; }
                catch (e) { return false; }
            },
        });

        // Kill switch health
        const ksHealth = http.get(`${BASE_URL}/api/kill-switch/health`);
        check(ksHealth, {
            'kill switch health: status 200': (r) => r.status === 200,
        });

        // Prometheus health
        const prom = http.get(`${BASE_URL}/health/prometheus`);
        check(prom, {
            'prometheus health: status 200': (r) => r.status === 200,
        });
    });
}

/**
 * Smoke test for trading endpoints.
 */
function testTradingEndpoints() {
    group('Trading Endpoints', function() {
        // Trading status
        const status = http.get(`${BASE_URL}/api/v1/trading/status`, {
            headers: HEADERS,
        });
        check(status, {
            'trading status: returns 200 or auth error': (r) => r.status === 200 || r.status === 401,
        });

        // Bot status
        const bots = http.get(`${BASE_URL}/api/v1/trading/bots`, {
            headers: HEADERS,
        });
        check(bots, {
            'bot status: returns 200 or auth error': (r) => r.status === 200 || r.status === 401,
        });
    });
}

/**
 * Smoke test for risk endpoints.
 */
function testRiskEndpoints() {
    group('Risk Endpoints', function() {
        // Regime
        const regime = http.get(`${BASE_URL}/api/risk/regime`);
        check(regime, {
            'regime: status 200': (r) => r.status === 200,
            'regime: has regime field': (r) => {
                try { return r.json().regime !== undefined; }
                catch (e) { return false; }
            },
        });

        // Compliance
        const compliance = http.get(`${BASE_URL}/api/risk/compliance`);
        check(compliance, {
            'compliance: status 200': (r) => r.status === 200,
        });

        // Physics
        const physics = http.get(`${BASE_URL}/api/risk/physics`);
        check(physics, {
            'physics: status 200': (r) => r.status === 200,
        });
    });
}

/**
 * Smoke test for floor manager endpoints.
 */
function testFloorManagerEndpoints() {
    group('Floor Manager Endpoints', function() {
        // Status
        const fmStatus = http.get(`${BASE_URL}/api/floor-manager/status`);
        check(fmStatus, {
            'floor manager status: status 200': (r) => r.status === 200,
        });
    });
}

/**
 * Smoke test for kill switch endpoints.
 */
function testKillSwitchEndpoints() {
    group('Kill Switch Endpoints', function() {
        // Status
        const ksStatus = http.get(`${BASE_URL}/api/kill-switch/status`);
        check(ksStatus, {
            'kill switch status: status 200': (r) => r.status === 200,
        });

        // Config
        const ksConfig = http.get(`${BASE_URL}/api/kill-switch/config`);
        check(ksConfig, {
            'kill switch config: status 200': (r) => r.status === 200,
        });
    });
}

/**
 * Main smoke test runner.
 */
export default function() {
    testHealthEndpoints();
    testTradingEndpoints();
    testRiskEndpoints();
    testFloorManagerEndpoints();
    testKillSwitchEndpoints();

    // Summary check
    check(null, {
        'smoke test completed': () => true,
    });
}

// Smoke test configuration
export const options = {
    // Quick smoke test: 1 VU, 30 seconds
    vus: 1,
    duration: '30s',

    thresholds: {
        'smoke_errors': ['rate<0.5'],
        'http_req_duration': ['p(95)<2000'],
    },

    tags: {
        type: 'smoke',
    },
};
