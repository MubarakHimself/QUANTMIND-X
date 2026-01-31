//+------------------------------------------------------------------+
//|                                                    Constants.mqh |
//|                        QuantMind Standard Library (QSL) - Core   |
//|                        System Constants Module                   |
//|                                                                  |
//| Defines system-wide constants used across all QSL modules.      |
//| Includes risk parameters, timeframes, magic numbers, and        |
//| configuration values.                                            |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

#ifndef __QSL_CONSTANTS_MQH__
#define __QSL_CONSTANTS_MQH__

//+------------------------------------------------------------------+
//| System Version Information                                       |
//+------------------------------------------------------------------+
#define QM_VERSION_MAJOR        1
#define QM_VERSION_MINOR        0
#define QM_VERSION_PATCH        0
#define QM_VERSION_STRING       "1.0.0"
#define QM_SYSTEM_NAME          "QuantMindX"

//+------------------------------------------------------------------+
//| Risk Management Constants                                        |
//+------------------------------------------------------------------+
// PropFirm Risk Limits
#define QM_DAILY_LOSS_LIMIT_PCT         5.0     // 5% daily loss limit (standard prop firm)
#define QM_HARD_STOP_BUFFER_PCT         1.0     // 1% buffer before hitting limit
#define QM_EFFECTIVE_LIMIT_PCT          4.0     // Effective stop at 4% (5% - 1% buffer)
#define QM_MAX_DRAWDOWN_PCT             10.0    // Maximum drawdown percentage

// Kelly Criterion Parameters
#define QM_KELLY_THRESHOLD              0.8     // Minimum Kelly score for A+ setups
#define QM_KELLY_FRACTION_MAX           0.25    // Maximum Kelly fraction (25% of capital)
#define QM_KELLY_FRACTION_MIN           0.01    // Minimum Kelly fraction (1% of capital)

// Position Sizing
#define QM_MAX_RISK_PER_TRADE_PCT       2.0     // Maximum risk per trade (2% of capital)
#define QM_MIN_RISK_PER_TRADE_PCT       0.5     // Minimum risk per trade (0.5% of capital)
#define QM_DEFAULT_RISK_PCT             1.0     // Default risk per trade (1% of capital)

// Risk Multiplier Bounds
#define QM_RISK_MULTIPLIER_MIN          0.0     // Minimum risk multiplier (no trading)
#define QM_RISK_MULTIPLIER_MAX          1.0     // Maximum risk multiplier (full risk)
#define QM_RISK_MULTIPLIER_DEFAULT      1.0     // Default risk multiplier

//+------------------------------------------------------------------+
//| Magic Number Ranges                                              |
//+------------------------------------------------------------------+
// Magic number ranges for different agent types
#define QM_MAGIC_BASE                   100000  // Base magic number
#define QM_MAGIC_ANALYST_START          100000  // Analyst agent range: 100000-109999
#define QM_MAGIC_ANALYST_END            109999
#define QM_MAGIC_QUANT_START            110000  // Quant agent range: 110000-119999
#define QM_MAGIC_QUANT_END              119999
#define QM_MAGIC_EXECUTOR_START         120000  // Executor agent range: 120000-129999
#define QM_MAGIC_EXECUTOR_END           129999
#define QM_MAGIC_COINFLIP_START         130000  // Coin Flip Bot range: 130000-139999
#define QM_MAGIC_COINFLIP_END           139999

//+------------------------------------------------------------------+
//| Timeframe Constants                                              |
//+------------------------------------------------------------------+
// Timeframe multipliers (in minutes)
#define QM_TF_M1_MINUTES                1
#define QM_TF_M5_MINUTES                5
#define QM_TF_M15_MINUTES               15
#define QM_TF_M30_MINUTES               30
#define QM_TF_H1_MINUTES                60
#define QM_TF_H4_MINUTES                240
#define QM_TF_D1_MINUTES                1440
#define QM_TF_W1_MINUTES                10080
#define QM_TF_MN1_MINUTES               43200

//+------------------------------------------------------------------+
//| Communication Constants                                          |
//+------------------------------------------------------------------+
// Python Bridge Configuration
#define QM_BRIDGE_HOST                  "localhost"
#define QM_BRIDGE_PORT                  8000
#define QM_BRIDGE_TIMEOUT_MS            5000    // 5 second timeout
#define QM_HEARTBEAT_INTERVAL_SEC       60      // Send heartbeat every 60 seconds
#define QM_HEARTBEAT_ENDPOINT           "/heartbeat"

// File Paths
#define QM_RISK_MATRIX_FILE             "risk_matrix.json"
#define QM_CONFIG_FILE                  "quantmind_config.json"
#define QM_LOG_FILE                     "quantmind_log.txt"

// GlobalVariable Names
#define QM_GV_RISK_MULTIPLIER           "QM_RISK_MULTIPLIER"
#define QM_GV_TRADING_ALLOWED           "QM_TRADING_ALLOWED"
#define QM_GV_HARD_STOP_ACTIVE          "QM_HARD_STOP_ACTIVE"
#define QM_GV_NEWS_GUARD_ACTIVE         "QM_NEWS_GUARD_ACTIVE"
#define QM_GV_LAST_HEARTBEAT            "QM_LAST_HEARTBEAT"

//+------------------------------------------------------------------+
//| Trading Constants                                                |
//+------------------------------------------------------------------+
// Order Types
#define QM_ORDER_TYPE_BUY               0
#define QM_ORDER_TYPE_SELL              1
#define QM_ORDER_TYPE_BUY_LIMIT         2
#define QM_ORDER_TYPE_SELL_LIMIT        3
#define QM_ORDER_TYPE_BUY_STOP          4
#define QM_ORDER_TYPE_SELL_STOP         5

// Slippage
#define QM_SLIPPAGE_POINTS              3       // Maximum slippage in points
#define QM_SLIPPAGE_PIPS                0.3     // Maximum slippage in pips

// Trade Execution
#define QM_MAX_RETRIES                  3       // Maximum order send retries
#define QM_RETRY_DELAY_MS               1000    // Delay between retries (1 second)

//+------------------------------------------------------------------+
//| News Guard Constants                                             |
//+------------------------------------------------------------------+
// News event timing
#define QM_NEWS_GUARD_BEFORE_MIN        30      // Stop trading 30 min before news
#define QM_NEWS_GUARD_AFTER_MIN         30      // Resume trading 30 min after news
#define QM_KILL_ZONE_ACTIVE             true    // Enable kill zone protection

// High-impact news events (example list)
#define QM_NEWS_NFP                     "Non-Farm Payrolls"
#define QM_NEWS_FOMC                    "FOMC Meeting"
#define QM_NEWS_CPI                     "CPI Release"
#define QM_NEWS_GDP                     "GDP Release"

//+------------------------------------------------------------------+
//| Indicator Constants                                              |
//+------------------------------------------------------------------+
// Moving Average Periods
#define QM_MA_FAST_PERIOD               10
#define QM_MA_MEDIUM_PERIOD             50
#define QM_MA_SLOW_PERIOD               200

// RSI Parameters
#define QM_RSI_PERIOD                   14
#define QM_RSI_OVERBOUGHT               70
#define QM_RSI_OVERSOLD                 30

// Bollinger Bands
#define QM_BB_PERIOD                    20
#define QM_BB_DEVIATION                 2.0

// ATR Parameters
#define QM_ATR_PERIOD                   14
#define QM_ATR_MULTIPLIER               2.0

//+------------------------------------------------------------------+
//| Database Constants                                               |
//+------------------------------------------------------------------+
// Database file paths (relative to Python backend)
#define QM_DB_SQLITE_PATH               "data/quantmind.db"
#define QM_DB_CHROMADB_PATH             "data/chromadb/"

// Collection names
#define QM_COLLECTION_STRATEGY_DNA      "strategy_dna"
#define QM_COLLECTION_MARKET_RESEARCH   "market_research"
#define QM_COLLECTION_AGENT_MEMORY      "agent_memory"

//+------------------------------------------------------------------+
//| Error Codes                                                      |
//+------------------------------------------------------------------+
// Custom error codes for QSL modules
#define QM_ERR_SUCCESS                  0
#define QM_ERR_INITIALIZATION_FAILED    -1
#define QM_ERR_INVALID_PARAMETER        -2
#define QM_ERR_SYMBOL_NOT_FOUND         -3
#define QM_ERR_TIMEFRAME_INVALID        -4
#define QM_ERR_TRADING_DISABLED         -5
#define QM_ERR_INSUFFICIENT_MARGIN      -6
#define QM_ERR_ORDER_SEND_FAILED        -7
#define QM_ERR_HARD_STOP_ACTIVE         -8
#define QM_ERR_NEWS_GUARD_ACTIVE        -9
#define QM_ERR_KELLY_THRESHOLD_FAILED   -10
#define QM_ERR_BRIDGE_CONNECTION_FAILED -11
#define QM_ERR_FILE_READ_FAILED         -12
#define QM_ERR_FILE_WRITE_FAILED        -13
#define QM_ERR_JSON_PARSE_FAILED        -14
#define QM_ERR_GLOBAL_VARIABLE_FAILED   -15

//+------------------------------------------------------------------+
//| Logging Constants                                                |
//+------------------------------------------------------------------+
// Log levels
#define QM_LOG_LEVEL_DEBUG              0
#define QM_LOG_LEVEL_INFO               1
#define QM_LOG_LEVEL_WARNING            2
#define QM_LOG_LEVEL_ERROR              3
#define QM_LOG_LEVEL_CRITICAL           4

// Default log level
#define QM_LOG_LEVEL_DEFAULT            QM_LOG_LEVEL_INFO

// Log message prefixes
#define QM_LOG_PREFIX_DEBUG             "[DEBUG]"
#define QM_LOG_PREFIX_INFO              "[INFO]"
#define QM_LOG_PREFIX_WARNING           "[WARNING]"
#define QM_LOG_PREFIX_ERROR             "[ERROR]"
#define QM_LOG_PREFIX_CRITICAL          "[CRITICAL]"

//+------------------------------------------------------------------+
//| Performance Constants                                            |
//+------------------------------------------------------------------+
// Timing thresholds (in milliseconds)
#define QM_PERF_HEARTBEAT_MAX_MS        100     // Max heartbeat response time
#define QM_PERF_RISK_RETRIEVAL_MAX_MS   50      // Max risk multiplier retrieval time
#define QM_PERF_DB_QUERY_MAX_MS         200     // Max database query time
#define QM_PERF_AGENT_WORKFLOW_MAX_MS   30000   // Max agent workflow time (30s)

// Ring buffer sizes
#define QM_RING_BUFFER_SIZE_SMALL       100
#define QM_RING_BUFFER_SIZE_MEDIUM      500
#define QM_RING_BUFFER_SIZE_LARGE       1000

//+------------------------------------------------------------------+
//| Coin Flip Bot Constants                                          |
//+------------------------------------------------------------------+
// Minimum trading days requirement
#define QM_MIN_TRADING_DAYS             5       // Minimum days to meet prop firm requirement
#define QM_COINFLIP_RISK_PCT            0.1     // Ultra-low risk (0.1% per trade)
#define QM_COINFLIP_MAX_TRADES_PER_DAY  2       // Maximum trades per day

//+------------------------------------------------------------------+
//| Quadratic Throttle Constants                                     |
//+------------------------------------------------------------------+
// Throttle calculation parameters
#define QM_THROTTLE_EXPONENT            2.0     // Quadratic exponent
#define QM_THROTTLE_MIN_MULTIPLIER      0.0     // Minimum throttle multiplier
#define QM_THROTTLE_MAX_MULTIPLIER      1.0     // Maximum throttle multiplier

//+------------------------------------------------------------------+
//| System Status Constants                                          |
//+------------------------------------------------------------------+
// Agent status codes
#define QM_STATUS_IDLE                  0
#define QM_STATUS_ACTIVE                1
#define QM_STATUS_PAUSED                2
#define QM_STATUS_ERROR                 3
#define QM_STATUS_STOPPED               4

// Trading mode
#define QM_MODE_LIVE                    0
#define QM_MODE_DEMO                    1
#define QM_MODE_BACKTEST                2

//+------------------------------------------------------------------+
//| Utility Macros                                                   |
//+------------------------------------------------------------------+
// Convert pips to points (for 5-digit brokers)
#define QM_PIPS_TO_POINTS(pips)         ((int)((pips) * 10))

// Convert points to pips
#define QM_POINTS_TO_PIPS(points)       ((double)(points) / 10.0)

// Check if value is within range
#define QM_IN_RANGE(value, min, max)    ((value) >= (min) && (value) <= (max))

// Clamp value to range
#define QM_CLAMP(value, min, max)       (((value) < (min)) ? (min) : (((value) > (max)) ? (max) : (value)))

#endif // __QSL_CONSTANTS_MQH__
//+------------------------------------------------------------------+
