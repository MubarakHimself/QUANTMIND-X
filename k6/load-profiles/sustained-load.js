/**
 * Sustained load profile (50-bot capacity test).
 *
 * Tests 50-bot concurrent capacity requirement with steady load.
 *
 * Profile:
 * - 0-30s: Ramp to 50 VUs
 * - 30-120s: Hold at 50 VUs (simulates 50 concurrent bots)
 * - 120-150s: Ramp down
 */

export const sustainedLoadProfile = {
    scenarios: {
        // Sustained 50-bot load
        sustained_load: {
            executor: 'ramping-vus',
            startVUs: 5,
            stages: [
                { duration: '30s', target: 50 },   // Ramp up to 50 VUs
                { duration: '90s', target: 50 },   // Hold at 50 VUs (50-bot capacity)
                { duration: '30s', target: 0 },     // Ramp down
            ],
            tags: { profile: 'sustained' },
        },
    },
};

export default sustainedLoadProfile;
