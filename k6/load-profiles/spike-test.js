/**
 * Spike test profile.
 *
 * Tests system behavior under sudden load spikes to verify
 * circuit breaker activation and graceful degradation.
 *
 * Profile:
 * - 0-10s: 10 VUs baseline
 * - 10-15s: Spike to 100 VUs (2x target)
 * - 15-25s: Drop to 10 VUs
 * - 25-35s: Another spike to 75 VUs (1.5x target)
 * - 35-45s: Return to baseline
 *
 * Circuit breaker should trigger at:
 * - 5% error rate threshold
 * - P99 latency > 2000ms
 */

export const spikeTestProfile = {
    scenarios: {
        // Spike test
        spike_test: {
            executor: 'ramping-vus',
            startVUs: 10,
            stages: [
                { duration: '10s', target: 10 },   // Baseline: 10 VUs
                { duration: '5s', target: 100 },     // Spike: 10 -> 100 VUs (2x capacity)
                { duration: '10s', target: 10 },    // Recovery: 100 -> 10 VUs
                { duration: '10s', target: 75 },    // Second spike: 10 -> 75 VUs
                { duration: '10s', target: 10 },    // Recovery to baseline
                { duration: '5s', target: 0 },      // Ramp down
            ],
            tags: { profile: 'spike' },
        },
    },
};

export default spikeTestProfile;
