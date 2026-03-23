/**
 * Custom k6 checks for QUANTMINDX load tests.
 *
 * Provides standardized checks for:
 * - HTTP status codes
 * - Response latency
 * - JSON payload validation
 * - Error rate tracking
 */

/**
 * Common HTTP status codes
 */
export const Status = {
    OK: 200,
    CREATED: 201,
    NO_CONTENT: 204,
    BAD_REQUEST: 400,
    UNAUTHORIZED: 401,
    FORBIDDEN: 403,
    NOT_FOUND: 404,
    TIMEOUT: 408,
    SERVER_ERROR: 500,
    BAD_GATEWAY: 502,
    SERVICE_UNAVAILABLE: 503,
    GATEWAY_TIMEOUT: 504,
};

/**
 * Create a standard response check.
 *
 * @param {string} name - Check name
 * @param {Object} response - k6 HTTP response
 * @param {number} expectedStatus - Expected HTTP status code
 * @returns {boolean} True if check passes
 */
export function checkResponse(name, response, expectedStatus = Status.OK) {
    const passed = response.status === expectedStatus;

    if (!passed) {
        console.error(`Check "${name}" failed: expected ${expectedStatus}, got ${response.status}`);
        console.error(`  URL: ${response.url}`);
        console.error(`  Body: ${response.body?.substring(0, 500)}`);
    }

    return passed;
}

/**
 * Check response has valid JSON.
 *
 * @param {string} name - Check name
 * @param {Object} response - k6 HTTP response
 * @returns {boolean} True if response is valid JSON
 */
export function checkValidJson(name, response) {
    try {
        const json = response.json();
        return json !== undefined && json !== null;
    } catch (e) {
        console.error(`Check "${name}" failed: invalid JSON - ${e.message}`);
        return false;
    }
}

/**
 * Check response time is within threshold.
 *
 * @param {string} name - Check name
 * @param {Object} response - k6 HTTP response
 * @param {number} maxMs - Maximum acceptable latency in milliseconds
 * @returns {boolean} True if latency is within threshold
 */
export function checkLatency(name, response, maxMs = 500) {
    const latency = response.timings.duration;
    const passed = latency < maxMs;

    if (!passed) {
        console.warn(`Check "${name}" latency warning: ${latency}ms > ${maxMs}ms threshold`);
    }

    return passed;
}

/**
 * Create a response time trend metric.
 *
 * @param {string} name - Metric name
 * @returns {Object} k6 Trend metric
 */
export function createLatencyTrend(name) {
    return new Trend(name);
}

/**
 * Check JSON payload contains required fields.
 *
 * @param {string} name - Check name
 * @param {Object} response - k6 HTTP response
 * @param {string[]} fields - Required field names (supports dot notation)
 * @returns {boolean} True if all fields present
 */
export function checkRequiredFields(name, response, fields) {
    try {
        const json = response.json();

        for (const field of fields) {
            const parts = field.split('.');
            let value = json;

            for (const part of parts) {
                if (value === undefined || value === null) {
                    console.error(`Check "${name}" failed: missing field "${field}"`);
                    return false;
                }
                value = value[part];
            }

            if (value === undefined && parts.length === 1) {
                console.error(`Check "${name}" failed: missing field "${field}"`);
                return false;
            }
        }

        return true;
    } catch (e) {
        console.error(`Check "${name}" failed: ${e.message}`);
        return false;
    }
}

/**
 * Create a rate counter for error tracking.
 *
 * @param {string} name - Metric name
 * @returns {Object} k6 Rate metric
 */
export function createErrorRate(name) {
    return new Rate(name);
}

/**
 * Track error occurrences.
 *
 * @param {Object} errorRate - k6 Rate metric from createErrorRate
 * @param {boolean} isError - Whether an error occurred
 */
export function trackError(errorRate, isError) {
    errorRate.add(isError ? 1 : 0);
}

/**
 * Check circuit breaker state in response.
 *
 * @param {Object} response - k6 HTTP response
 * @returns {Object} Circuit breaker state { triggered: boolean, state: string }
 */
export function checkCircuitBreakerState(response) {
    try {
        const json = response.json();

        // Check for circuit breaker fields in compliance response
        if (json.account_tags) {
            const triggered = json.account_tags.some(
                tag => tag.circuit_breaker_state === 'triggered'
            );
            return { triggered, state: triggered ? 'triggered' : 'normal' };
        }

        return { triggered: false, state: 'unknown' };
    } catch (e) {
        return { triggered: false, state: 'error' };
    }
}

/**
 * Validate kill switch response structure.
 *
 * @param {Object} response - k6 HTTP response
 * @returns {boolean} True if valid kill switch response
 */
export function checkKillSwitchResponse(response) {
    return checkRequiredFields('kill_switch_response', response, [
        'success',
        'tier',
        'activated_at_utc'
    ]);
}

/**
 * Validate trading status response structure.
 *
 * @param {Object} response - k6 HTTP response
 * @returns {boolean} True if valid trading status response
 */
export function checkTradingStatusResponse(response) {
    try {
        const json = response.json();
        // Trading status should have regime and bot information
        return json !== null && typeof json === 'object';
    } catch (e) {
        return false;
    }
}

// k6 built-in metrics
import { Trend, Rate } from 'k6/metrics';
