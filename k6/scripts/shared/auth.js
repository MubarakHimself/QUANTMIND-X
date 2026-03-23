/**
 * Authentication helpers for k6 load tests.
 *
 * Usage:
 *   import { getAuthHeaders } from './scripts/shared/auth.js';
 *
 *   export const options = {
 *     thresholds: { ... }
 *   };
 *
 *   export default function() {
 *     const headers = getAuthHeaders();
 *     http.get(`${BASE_URL}/api/endpoint`, { headers });
 *   }
 */

/**
 * Get authentication headers for protected endpoints.
 * Uses K6_AUTH_TOKEN environment variable.
 *
 * @returns {Object} Headers object with Authorization header if token is set
 */
export function getAuthHeaders() {
    const token = __ENV.K6_AUTH_TOKEN || '';

    if (!token) {
        return {};
    }

    return {
        'Authorization': `Bearer ${token}`
    };
}

/**
 * Get default headers for all requests.
 *
 * @returns {Object} Default headers including content-type and auth
 */
export function getDefaultHeaders() {
    const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    };

    const token = __ENV.K6_AUTH_TOKEN || '';
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    return headers;
}

/**
 * Check if auth is configured.
 *
 * @returns {boolean} True if K6_AUTH_TOKEN is set
 */
export function isAuthConfigured() {
    return !!__ENV.K6_AUTH_TOKEN;
}

/**
 * Get base URL from environment or default.
 *
 * @returns {string} Base URL for API
 */
export function getBaseUrl() {
    return __ENV.K6_BASE_URL || 'http://localhost:8000';
}
