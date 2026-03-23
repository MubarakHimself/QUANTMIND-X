/**
 * Floor Manager endpoint load test scenarios.
 *
 * Tests:
 * - POST /api/floor-manager/chat - Chat with floor manager
 * - GET /api/floor-manager/status - Floor manager status
 * - POST /api/floor-manager/task - Submit task for routing
 * - GET /api/floor-manager/departments - List departments
 */

import { check } from 'k6';
import http from 'k6/http';
import { Trend, Rate } from 'k6/metrics';
import { getBaseUrl, getDefaultHeaders, getAuthHeaders } from '../scripts/shared/auth.js';
import { checkResponse, checkValidJson, checkLatency, checkRequiredFields } from '../scripts/shared/checks.js';
import { chatPayload, taskRequestPayload, correlationId } from '../scripts/shared/payload-generator.js';

const BASE_URL = getBaseUrl();
const HEADERS = getDefaultHeaders();
const AUTH_HEADERS = getAuthHeaders();

// Custom metrics for floor manager endpoints
const chatLatency = new Trend('floor_manager_chat_latency');
const statusLatency = new Trend('floor_manager_status_latency');
const taskLatency = new Trend('floor_manager_task_latency');
const departmentsLatency = new Trend('floor_manager_departments_latency');

const floorManagerErrorRate = new Rate('floor_manager_errors');

/**
 * Test floor manager chat endpoint (requires auth for full functionality).
 */
export function testFloorManagerChat() {
    const payload = chatPayload('What is the current system status?', false);
    const url = `${BASE_URL}/api/floor-manager/chat`;

    const response = http.post(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'floor_manager_chat' }
    });

    chatLatency.add(response.timings.duration);

    // 200 = success, 401 = no auth configured
    const passed = response.status === 200 || response.status === 401;

    floorManagerErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test floor manager status endpoint.
 */
export function testFloorManagerStatus() {
    const url = `${BASE_URL}/api/floor-manager/status`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'floor_manager_status' }
    });

    statusLatency.add(response.timings.duration);

    const passed = checkResponse('floor_manager_status_check', response, 200) &&
                   checkValidJson('floor_manager_status_json', response);

    floorManagerErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test floor manager task submission endpoint.
 */
export function testFloorManagerTask() {
    const payload = taskRequestPayload('Check regime status for EURUSD');
    const url = `${BASE_URL}/api/floor-manager/task`;

    const response = http.post(url, JSON.stringify(payload), {
        headers: {
            ...HEADERS,
            ...AUTH_HEADERS
        },
        tags: { name: 'floor_manager_task' }
    });

    taskLatency.add(response.timings.duration);

    // 200 = success, 401 = no auth configured
    const passed = response.status === 200 || response.status === 401;

    floorManagerErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Test floor manager departments endpoint.
 */
export function testFloorManagerDepartments() {
    const url = `${BASE_URL}/api/floor-manager/departments`;

    const response = http.get(url, {
        headers: HEADERS,
        tags: { name: 'floor_manager_departments' }
    });

    departmentsLatency.add(response.timings.duration);

    const passed = checkResponse('floor_manager_departments_check', response, 200) &&
                   checkValidJson('floor_manager_departments_json', response);

    floorManagerErrorRate.add(passed ? 0 : 1);

    return passed;
}

/**
 * Run all floor manager scenario tests.
 */
export function runFloorManagerScenarios() {
    testFloorManagerStatus();
    testFloorManagerDepartments();
    testFloorManagerChat();
    testFloorManagerTask();
}

// Default export for k6 scenario executor
export default function() {
    runFloorManagerScenarios();
}
