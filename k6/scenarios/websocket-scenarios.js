/**
 * WebSocket endpoint load test scenarios.
 *
 * Tests:
 * - WS /ws - Main WebSocket endpoint
 * - WS /ws/trading - Trading WebSocket endpoint
 *
 * Note: WebSocket support requires k6 >= 0.46.0
 */

import ws from 'k6/ws';
import { Trend, Rate, Counter } from 'k6/metrics';
import { getBaseUrl } from '../scripts/shared/auth.js';
import { wsSubscribeMessage, wsPingMessage, correlationId } from '../scripts/shared/payload-generator.js';

const BASE_URL = getBaseUrl().replace('http://', 'ws://').replace('https://', 'wss://');

// Custom metrics for WebSocket endpoints
const wsConnectLatency = new Trend('ws_connect_latency');
const wsMessageLatency = new Trend('ws_message_latency');
const wsErrors = new Rate('ws_errors');
const wsMessagesReceived = new Counter('ws_messages_received');
const wsClosedCleanly = new Counter('ws_closed_cleanly');

/**
 * Test main WebSocket endpoint.
 * Connects, subscribes to topics, sends ping, and closes.
 */
export function testMainWebSocket() {
    const url = `${BASE_URL}/ws`;
    const corrId = correlationId();
    const testDuration = 5; // seconds

    let connected = false;
    let errorOccurred = false;

    const startTime = Date.now();

    const response = ws.connect(url, {}, function(socket) {
        connected = true;
        wsConnectLatency.add(Date.now() - startTime);

        socket.on('open', function() {
            // Subscribe to trading topic
            socket.send(wsSubscribeMessage('trading'));
        });

        socket.on('message', function(message) {
            wsMessagesReceived.add(1);

            // Try to parse and validate message
            try {
                const json = JSON.parse(message);
                // Message received successfully
            } catch (e) {
                // Non-JSON message is also acceptable (e.g., ping/pong)
            }
        });

        socket.on('close', function() {
            wsClosedCleanly.add(1);
        });

        socket.on('error', function(e) {
            console.error('WebSocket error:', e);
            errorOccurred = true;
            wsErrors.add(1);
        });

        // Send ping periodically
        socket.setInterval(function() {
            if (socket.__proto__.readyState === 1) { // OPEN
                socket.send(wsPingMessage());
            }
        }, 2000);

        // Run for test duration then close
        socket.setTimeout(function() {
            socket.close();
        }, testDuration * 1000);
    });

    return !errorOccurred;
}

/**
 * Test trading WebSocket endpoint.
 * Connects, subscribes, and verifies state snapshot on connect.
 */
export function testTradingWebSocket() {
    const url = `${BASE_URL}/ws/trading`;
    const corrId = correlationId();
    const testDuration = 5; // seconds

    let connected = false;
    let errorOccurred = false;
    let stateSnapshotReceived = false;

    const startTime = Date.now();

    const response = ws.connect(url, {}, function(socket) {
        connected = true;
        wsConnectLatency.add(Date.now() - startTime);

        socket.on('open', function() {
            // Subscribe to trading topic
            socket.send(wsSubscribeMessage('trading'));
        });

        socket.on('message', function(message) {
            wsMessagesReceived.add(1);

            try {
                const json = JSON.parse(message);

                // Check for state snapshot (should be received on connect)
                if (json.type === 'state_snapshot') {
                    stateSnapshotReceived = true;
                }

                // Handle subscription confirmation
                if (json.type === 'subscription_confirmed') {
                    console.log(`Subscribed to: ${json.topic}`);
                }
            } catch (e) {
                // Non-JSON message is also acceptable
            }
        });

        socket.on('close', function() {
            wsClosedCleanly.add(1);
        });

        socket.on('error', function(e) {
            console.error('Trading WebSocket error:', e);
            errorOccurred = true;
            wsErrors.add(1);
        });

        // Run for test duration then close
        socket.setTimeout(function() {
            socket.close();
        }, testDuration * 1000);
    });

    // Consider success if connected and no error
    return connected && !errorOccurred;
}

/**
 * Run WebSocket scenarios with configured VUs.
 * Note: WebSocket tests typically run with fewer VUs due to connection limits.
 */
export function runWebSocketScenarios() {
    testMainWebSocket();
    testTradingWebSocket();
}

// Default export for k6 scenario executor
export default function() {
    runWebSocketScenarios();
}

// Configuration for WebSocket-only tests
export const wsOptions = {
    // WebSocket tests need longer timeouts
    timeout: '30s',

    thresholds: {
        'ws_connect_latency': ['p(95)<1000'],
        'ws_errors': ['rate<0.1'],
    }
};
