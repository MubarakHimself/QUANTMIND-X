/**
 * Trading endpoint load test scenarios.
 *
 * Tests:
 * - POST /api/v1/backtest/run - Backtest execution
 * - GET /api/v1/backtest/results/{id} - Backtest results
 * - GET /api/v1/trading/status - Trading status
 * - GET /api/v1/trading/bots - Bot status
 * - POST /api/v1/trading/close - Close position
 * - POST /api/v1/trading/emergency_stop - Emergency stop
 */

import { check } from 'k6';
import http from 'k6/http';
import { Trend, Rate } from 'k6/metrics';
import { getBaseUrl, getDefaultHeaders } from '../scripts/shared/auth.js';
import { checkResponse, checkValidJson, checkLatency, checkRequiredFields } from '../scripts/shared/checks.js';
import { closePositionPayload, backtestRunPayload, emergencyStopPayload, randomBotId, correlationId } from '../scripts/shared/payload-generator.js';

const BASE_URL = getBaseUrl();
const HEADERS = getDefaultHeaders();

// Custom metrics for trading endpoints
const backtestRunLatency = new Trend('trading_backtest_run_latency');
const tradingStatusLatency = new Trend('trading_status_latency');
const botStatusLatency = new Trend('trading_bot_status_latency');
const closePositionLatency = new Trend('trading_close_position_latency');
const emergencyStopLatency = new Trend('trading_emergency_stop_latency');

const tradingErrorRate = new Rate('trading_errors');

/**
 * Test backtest run endpoint.
 */
export function testBacktestRun() {
    const payload = backtestRunPayload();
    const url = `${BASE_URL}/api/v1/backtest/run`;
    const corrId = correlationId();

    const response = http.post(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            'X-Correlation-ID': corrId
        },
        tags: { name: 'backtest_run' }
    });

    backtestRunLatency.add(response.timings.duration);

    const passed = checkResponse('backtest_run_status', response, 200) &&
                   checkValidJson('backtest_run_json', response);

    tradingErrorRate.add(passed ? 0 : 1);

    // Return backtest_id for follow-up requests
    if (passed) {
        try {
            const json = response.json();
            return json.backtest_id || null;
        } catch (e) {
            return null;
        }
    }
    return null;
}

/**
 * Test backtest results endpoint.
 */
export function testBacktestResults(backtestId) {
    if (!backtestId) {
        // Generate a fake ID for testing even if we didn't run a backtest
        backtestId = `bt-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
    }

    const url = `${BASE_URL}/api/v1/backtest/results/${backtestId}`;
    const corrId = correlationId();

    const response = http.get(url, {
        headers: {
            ...HEADERS,
            'X-Correlation-ID': corrId
        },
        tags: { name: 'backtest_results' }
    });

    checkLatency('backtest_results', response, 500);

    const passed = checkResponse('backtest_results_status', response, 200);

    tradingErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test trading status endpoint.
 */
export function testTradingStatus() {
    const url = `${BASE_URL}/api/v1/trading/status`;
    const corrId = correlationId();

    const response = http.get(url, {
        headers: {
            ...HEADERS,
            'X-Correlation-ID': corrId
        },
        tags: { name: 'trading_status' }
    });

    tradingStatusLatency.add(response.timings.duration);

    const passed = checkResponse('trading_status_check', response, 200) &&
                   checkValidJson('trading_status_json', response);

    tradingErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test bot status endpoint.
 */
export function testBotStatus() {
    const url = `${BASE_URL}/api/v1/trading/bots`;
    const corrId = correlationId();

    const response = http.get(url, {
        headers: {
            ...HEADERS,
            'X-Correlation-ID': corrId
        },
        tags: { name: 'bot_status' }
    });

    botStatusLatency.add(response.timings.duration);

    const passed = checkResponse('bot_status_check', response, 200) &&
                   checkValidJson('bot_status_json', response);

    tradingErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test close position endpoint.
 */
export function testClosePosition() {
    const payload = closePositionPayload();
    const url = `${BASE_URL}/api/v1/trading/close`;
    const corrId = correlationId();

    const response = http.post(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            'X-Correlation-ID': corrId
        },
        tags: { name: 'close_position' }
    });

    closePositionLatency.add(response.timings.duration);

    // 200 or 400/404 are acceptable (position may not exist in test env)
    const passed = response.status === 200 ||
                   response.status === 400 ||
                   response.status === 404;

    tradingErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test emergency stop endpoint.
 */
export function testEmergencyStop() {
    const payload = emergencyStopPayload();
    const url = `${BASE_URL}/api/v1/trading/emergency_stop`;
    const corrId = correlationId();

    const response = http.post(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            'X-Correlation-ID': corrId
        },
        tags: { name: 'emergency_stop' }
    });

    emergencyStopLatency.add(response.timings.duration);

    // Accept 200 (success) or 500 (no MT5 in test env)
    const passed = response.status === 200 || response.status === 500;

    tradingErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Run all trading scenario tests.
 */
export function runTradingScenarios() {
    // Test backtest and capture ID for results test
    const backtestId = testBacktestRun();

    // Test backtest results (uses captured or generated ID)
    testBacktestResults(backtestId);

    // Test trading status
    testTradingStatus();

    // Test bot status
    testBotStatus();

    // Test close position
    testClosePosition();

    // Test emergency stop
    testEmergencyStop();
}

// Default export for k6 scenario executor
export default function() {
    runTradingScenarios();
}
