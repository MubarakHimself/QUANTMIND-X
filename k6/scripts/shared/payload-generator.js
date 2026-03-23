/**
 * Test data payload generators for k6 load tests.
 *
 * Provides realistic test data generation for:
 * - Trading requests
 * - Kill switch triggers
 * - Risk parameter queries
 * - Agent IDs
 */

/**
 * Generate a random trading symbol.
 *
 * @returns {string} Random forex symbol
 */
export function randomSymbol() {
    const symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
        'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP',
        'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY'
    ];
    return symbols[Math.floor(Math.random() * symbols.length)];
}

/**
 * Generate a random bot ID.
 *
 * @param {number} index - Optional bot index for deterministic generation
 * @returns {string} Bot ID
 */
export function randomBotId(index = null) {
    const prefix = 'BOT';
    const suffixes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010',
                      '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
                      '021', '022', '023', '024', '025', '026', '027', '028', '029', '030',
                      '031', '032', '033', '034', '035', '036', '037', '038', '039', '040',
                      '041', '042', '043', '044', '045', '046', '047', '048', '049', '050'];

    if (index !== null && index < suffixes.length) {
        return `${prefix}-${suffixes[index]}`;
    }

    return `${prefix}-${suffixes[Math.floor(Math.random() * suffixes.length)]}`;
}

/**
 * Generate a random account tag.
 *
 * @returns {string} Account tag
 */
export function randomAccountTag() {
    const tags = [
        'prop-firm-001', 'prop-firm-002', 'prop-firm-003',
        'demo-account', 'live-account-001', 'live-account-002'
    ];
    return tags[Math.floor(Math.random() * tags.length)];
}

/**
 * Generate a close position request payload.
 *
 * @param {string} botId - Optional bot ID
 * @param {number} ticket - Optional ticket number
 * @returns {Object} Close position request body
 */
export function closePositionPayload(botId = null, ticket = null) {
    return {
        bot_id: botId || randomBotId(),
        ticket: ticket || Math.floor(Math.random() * 100000) + 10000
    };
}

/**
 * Generate a kill switch trigger request payload.
 *
 * @param {number} tier - Kill switch tier (1, 2, or 3)
 * @param {string} activator - Name of activator
 * @param {string[]} strategyIds - Optional strategy IDs for tier 2
 * @returns {Object} Kill switch trigger request body
 */
export function killSwitchTriggerPayload(tier = 1, activator = 'k6-load-test', strategyIds = null) {
    const payload = {
        tier: tier,
        activator: activator
    };

    if (tier === 2 && strategyIds) {
        payload.strategy_ids = strategyIds;
    }

    return payload;
}

/**
 * Generate a floor manager chat request payload.
 *
 * @param {string} message - Chat message
 * @param {boolean} stream - Whether to stream response
 * @returns {Object} Chat request body
 */
export function chatPayload(message = 'What is the current trading status?', stream = false) {
    return {
        message: message,
        context: null,
        history: [],
        stream: stream
    };
}

/**
 * Generate a backtest run request payload.
 *
 * @returns {Object} Backtest run request body
 */
export function backtestRunPayload() {
    return {
        symbol: randomSymbol(),
        timeframe: 'H1',
        variant: 'vanilla',
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        initial_cash: 10000.0,
        commission: 0.001,
        slippage: 0.0,
        regime_filtering: false,
        chaos_threshold: 0.6,
        banned_regimes: ['NEWS_EVENT', 'HIGH_CHAOS']
    };
}

/**
 * Generate a task request payload for floor manager.
 *
 * @param {string} task - Task description
 * @returns {Object} Task request body
 */
export function taskRequestPayload(task = 'What is the current regime status?') {
    return {
        task: task,
        context: {
            source: 'k6-load-test',
            timestamp: new Date().toISOString()
        }
    };
}

/**
 * Generate emergency stop request payload.
 *
 * @param {string} reason - Reason for emergency stop
 * @returns {Object} Emergency stop request body
 */
export function emergencyStopPayload(reason = 'k6-load-test-emergency') {
    return {
        reason: reason,
        confirm: true
    };
}

/**
 * Generate WebSocket subscribe message.
 *
 * @param {string} topic - Topic to subscribe to
 * @returns {string} JSON subscribe message
 */
export function wsSubscribeMessage(topic = 'trading') {
    return JSON.stringify({
        action: 'subscribe',
        topic: topic
    });
}

/**
 * Generate WebSocket ping message.
 *
 * @returns {string} JSON ping message
 */
export function wsPingMessage() {
    return JSON.stringify({
        action: 'ping'
    });
}

/**
 * Get all bot IDs for batch testing.
 *
 * @param {number} count - Number of bot IDs to generate
 * @returns {string[]} Array of bot IDs
 */
export function getAllBotIds(count = 50) {
    return Array.from({ length: Math.min(count, 50) }, (_, i) => randomBotId(i));
}

/**
 * Generate unique correlation ID for request tracking.
 *
 * @returns {string} Correlation ID
 */
export function correlationId() {
    return `k6-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
