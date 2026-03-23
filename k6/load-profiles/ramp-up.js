/**
 * Ramp-up load profile.
 *
 * Gradually increases load to establish baseline latency under increasing load.
 *
 * Profile:
 * - 0-30s: 1-10 VUs (warm-up)
 * - 30-60s: 10-25 VUs (gradual increase)
 * - 60-90s: 25-50 VUs (approach target)
 * - 90-120s: Hold at 50 VUs (sustained)
 * - 120-150s: Ramp down
 */

export const rampUpProfile = {
    scenarios: {
        // Ramp-up scenario
        ramp_up: {
            executor: 'ramping-vus',
            startVUs: 1,
            stages: [
                { duration: '30s', target: 10 },   // Warm-up: 1 -> 10 VUs
                { duration: '30s', target: 25 },   // Increase: 10 -> 25 VUs
                { duration: '30s', target: 50 },   // Approach target: 25 -> 50 VUs
                { duration: '30s', target: 50 },   // Hold at target
                { duration: '30s', target: 0 },    // Ramp down
            ],
            tags: { profile: 'ramp_up' },
        },
    },
};

export default rampUpProfile;
