//+------------------------------------------------------------------+
//|                                                        Types.mqh |
//|                        QuantMind Standard Library (QSL) - Core   |
//|                        Custom Data Types Module                  |
//|                                                                  |
//| Defines custom data structures, enums, and types used across    |
//| all QSL modules. Includes trade proposals, account states,      |
//| risk parameters, and agent communication structures.             |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

#ifndef __QSL_TYPES_MQH__
#define __QSL_TYPES_MQH__

//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+

// Trade decision types
enum ENUM_TRADE_DECISION
{
    TRADE_DECISION_REJECT = 0,      // Reject trade proposal
    TRADE_DECISION_APPROVE = 1,     // Approve trade proposal
    TRADE_DECISION_PENDING = 2,     // Pending review
    TRADE_DECISION_MODIFIED = 3     // Approved with modifications
};

// Agent types
enum ENUM_AGENT_TYPE
{
    AGENT_TYPE_ANALYST = 0,         // Market analyst agent
    AGENT_TYPE_QUANT = 1,           // Quantitative strategy agent
    AGENT_TYPE_EXECUTOR = 2,        // Trade execution agent
    AGENT_TYPE_COINFLIP = 3,        // Coin flip bot (minimum days)
    AGENT_TYPE_SENTINEL = 4         // Risk monitoring agent
};

// Risk status
enum ENUM_RISK_STATUS
{
    RISK_STATUS_NORMAL = 0,         // Normal trading conditions
    RISK_STATUS_THROTTLED = 1,      // Risk throttled (reduced position sizing)
    RISK_STATUS_HARD_STOP = 2,      // Hard stop active (no trading)
    RISK_STATUS_NEWS_GUARD = 3,     // News guard active (no trading)
    RISK_STATUS_PRESERVATION = 4    // Preservation mode (A+ setups only)
};

// Account status
enum ENUM_ACCOUNT_STATUS
{
    ACCOUNT_STATUS_ACTIVE = 0,      // Account active and trading
    ACCOUNT_STATUS_PAUSED = 1,      // Account paused
    ACCOUNT_STATUS_BREACHED = 2,    // Account breached limits
    ACCOUNT_STATUS_INACTIVE = 3     // Account inactive
};

// Trade signal type
enum ENUM_SIGNAL_TYPE
{
    SIGNAL_TYPE_NONE = 0,           // No signal
    SIGNAL_TYPE_BUY = 1,            // Buy signal
    SIGNAL_TYPE_SELL = 2,           // Sell signal
    SIGNAL_TYPE_CLOSE_BUY = 3,      // Close buy positions
    SIGNAL_TYPE_CLOSE_SELL = 4,     // Close sell positions
    SIGNAL_TYPE_CLOSE_ALL = 5       // Close all positions
};

// Strategy quality rating
enum ENUM_STRATEGY_QUALITY
{
    STRATEGY_QUALITY_F = 0,         // F grade (Kelly < 0.2)
    STRATEGY_QUALITY_D = 1,         // D grade (Kelly 0.2-0.4)
    STRATEGY_QUALITY_C = 2,         // C grade (Kelly 0.4-0.6)
    STRATEGY_QUALITY_B = 3,         // B grade (Kelly 0.6-0.8)
    STRATEGY_QUALITY_A = 4,         // A grade (Kelly 0.8-0.9)
    STRATEGY_QUALITY_A_PLUS = 5     // A+ grade (Kelly >= 0.9)
};

//+------------------------------------------------------------------+
//| Structure Definitions                                            |
//+------------------------------------------------------------------+

// Trade proposal structure
struct STradeProposal
{
    string            symbol;                // Trading symbol
    ENUM_SIGNAL_TYPE  signalType;           // Signal type (buy/sell)
    double            entryPrice;           // Proposed entry price
    double            stopLoss;             // Stop loss price
    double            takeProfit;           // Take profit price
    double            lotSize;              // Proposed lot size
    double            kellyScore;           // Kelly criterion score
    double            winRate;              // Historical win rate
    double            avgWin;               // Average win amount
    double            avgLoss;              // Average loss amount
    double            sharpeRatio;          // Sharpe ratio
    double            maxDrawdown;          // Maximum drawdown
    int               magicNumber;          // Magic number
    string            strategyName;         // Strategy identifier
    datetime          timestamp;            // Proposal timestamp
    string            comment;              // Additional comments
};

// Account state structure
struct SAccountState
{
    string            accountId;            // Account identifier
    double            balance;              // Account balance
    double            equity;               // Account equity
    double            freeMargin;           // Free margin
    double            highWaterMark;        // Daily high water mark
    double            dailyPnL;             // Daily profit/loss
    double            dailyDrawdown;        // Daily drawdown percentage
    double            dailyLossLimit;       // Daily loss limit
    double            maxDrawdown;          // Maximum drawdown limit
    int               tradesCount;          // Number of trades today
    datetime          lastUpdate;           // Last update timestamp
    ENUM_ACCOUNT_STATUS status;             // Account status
    ENUM_RISK_STATUS  riskStatus;           // Risk status
    bool              isKillZone;           // News guard active
};

// Risk parameters structure
struct SRiskParameters
{
    double            riskMultiplier;       // Current risk multiplier (0.0-1.0)
    double            maxRiskPerTrade;      // Maximum risk per trade (%)
    double            kellyThreshold;       // Kelly threshold for A+ setups
    double            dailyLossLimit;       // Daily loss limit (%)
    double            hardStopThreshold;    // Hard stop threshold (%)
    bool              preservationMode;     // Preservation mode active
    bool              newsGuardActive;      // News guard active
    bool              hardStopActive;       // Hard stop active
    datetime          lastHeartbeat;        // Last heartbeat timestamp
};

// Position information structure
struct SPositionInfo
{
    ulong             ticket;               // Position ticket
    string            symbol;               // Position symbol
    ENUM_POSITION_TYPE type;                // Position type (buy/sell)
    double            volume;               // Position volume (lots)
    double            openPrice;            // Open price
    double            currentPrice;         // Current price
    double            stopLoss;             // Stop loss price
    double            takeProfit;           // Take profit price
    double            profit;               // Current profit/loss
    double            swap;                 // Swap charges
    double            commission;           // Commission charges
    int               magicNumber;          // Magic number
    datetime          openTime;             // Open time
    string            comment;              // Position comment
};

// Order information structure
struct SOrderInfo
{
    ulong             ticket;               // Order ticket
    string            symbol;               // Order symbol
    ENUM_ORDER_TYPE   type;                 // Order type
    double            volume;               // Order volume (lots)
    double            priceOpen;            // Order open price
    double            stopLoss;             // Stop loss price
    double            takeProfit;           // Take profit price
    datetime          timeSetup;            // Order setup time
    datetime          expiration;           // Order expiration time
    int               magicNumber;          // Magic number
    string            comment;              // Order comment
};

// Heartbeat payload structure
struct SHeartbeatPayload
{
    string            eaName;               // EA name
    string            symbol;               // Trading symbol
    int               magicNumber;          // Magic number
    double            riskMultiplier;       // Current risk multiplier
    double            accountBalance;       // Account balance
    double            accountEquity;        // Account equity
    double            dailyPnL;             // Daily P&L
    int               openPositions;        // Number of open positions
    datetime          timestamp;            // Heartbeat timestamp
    string            status;               // EA status
};

// Strategy performance structure
struct SStrategyPerformance
{
    string            strategyName;         // Strategy name
    double            kellyScore;           // Kelly criterion score
    double            sharpeRatio;          // Sharpe ratio
    double            sortinoRatio;         // Sortino ratio
    double            maxDrawdown;          // Maximum drawdown (%)
    double            winRate;              // Win rate (%)
    double            profitFactor;         // Profit factor
    double            avgWin;               // Average win
    double            avgLoss;              // Average loss
    double            totalTrades;          // Total number of trades
    double            totalProfit;          // Total profit
    double            totalLoss;            // Total loss
    ENUM_STRATEGY_QUALITY quality;          // Strategy quality rating
    datetime          lastUpdate;           // Last update timestamp
};

// Agent task structure
struct SAgentTask
{
    string            taskId;               // Task identifier
    ENUM_AGENT_TYPE   agentType;            // Agent type
    string            taskType;             // Task type (research, backtest, deploy)
    string            taskData;             // Task data (JSON string)
    string            status;               // Task status (pending, in_progress, completed)
    datetime          createdAt;            // Creation timestamp
    datetime          completedAt;          // Completion timestamp
    string            result;               // Task result (JSON string)
};

// Market condition structure
struct SMarketCondition
{
    string            symbol;               // Symbol
    ENUM_TIMEFRAMES   timeframe;            // Timeframe
    double            volatility;           // Current volatility (ATR)
    double            trend;                // Trend strength (-1 to 1)
    double            momentum;             // Momentum indicator
    double            volume;               // Current volume
    double            spread;               // Current spread
    bool              isTrending;           // Is market trending
    bool              isVolatile;           // Is market volatile
    datetime          timestamp;            // Timestamp
};

// News event structure
struct SNewsEvent
{
    string            currency;             // Currency affected
    string            eventName;            // Event name
    string            impact;               // Impact level (Low, Medium, High)
    datetime          eventTime;            // Event time
    datetime          guardStartTime;       // Guard start time (30 min before)
    datetime          guardEndTime;         // Guard end time (30 min after)
    bool              isActive;             // Is guard currently active
};

// Ring buffer element structure (for indicators)
struct SRingBufferElement
{
    double            value;                // Indicator value
    datetime          timestamp;            // Timestamp
    int               barIndex;             // Bar index
};

// JSON parse result structure
struct SJsonParseResult
{
    bool              success;              // Parse success flag
    string            errorMessage;         // Error message if failed
    int               errorCode;            // Error code
};

//+------------------------------------------------------------------+
//| Type Aliases                                                     |
//+------------------------------------------------------------------+
// Callback function types
typedef void (*TOnTradeCallback)(const STradeProposal &proposal);
typedef void (*TOnRiskUpdateCallback)(const SRiskParameters &params);
typedef void (*TOnHeartbeatCallback)(const SHeartbeatPayload &payload);
typedef void (*TOnNewsEventCallback)(const SNewsEvent &event);

//+------------------------------------------------------------------+
//| Utility Functions for Type Conversion                           |
//+------------------------------------------------------------------+

// Convert trade decision enum to string
string TradeDecisionToString(ENUM_TRADE_DECISION decision)
{
    switch(decision)
    {
        case TRADE_DECISION_REJECT:     return "REJECT";
        case TRADE_DECISION_APPROVE:    return "APPROVE";
        case TRADE_DECISION_PENDING:    return "PENDING";
        case TRADE_DECISION_MODIFIED:   return "MODIFIED";
        default:                        return "UNKNOWN";
    }
}

// Convert agent type enum to string
string AgentTypeToString(ENUM_AGENT_TYPE agentType)
{
    switch(agentType)
    {
        case AGENT_TYPE_ANALYST:    return "ANALYST";
        case AGENT_TYPE_QUANT:      return "QUANT";
        case AGENT_TYPE_EXECUTOR:   return "EXECUTOR";
        case AGENT_TYPE_COINFLIP:   return "COINFLIP";
        case AGENT_TYPE_SENTINEL:   return "SENTINEL";
        default:                    return "UNKNOWN";
    }
}

// Convert risk status enum to string
string RiskStatusToString(ENUM_RISK_STATUS status)
{
    switch(status)
    {
        case RISK_STATUS_NORMAL:        return "NORMAL";
        case RISK_STATUS_THROTTLED:     return "THROTTLED";
        case RISK_STATUS_HARD_STOP:     return "HARD_STOP";
        case RISK_STATUS_NEWS_GUARD:    return "NEWS_GUARD";
        case RISK_STATUS_PRESERVATION:  return "PRESERVATION";
        default:                        return "UNKNOWN";
    }
}

// Convert signal type enum to string
string SignalTypeToString(ENUM_SIGNAL_TYPE signal)
{
    switch(signal)
    {
        case SIGNAL_TYPE_NONE:          return "NONE";
        case SIGNAL_TYPE_BUY:           return "BUY";
        case SIGNAL_TYPE_SELL:          return "SELL";
        case SIGNAL_TYPE_CLOSE_BUY:     return "CLOSE_BUY";
        case SIGNAL_TYPE_CLOSE_SELL:    return "CLOSE_SELL";
        case SIGNAL_TYPE_CLOSE_ALL:     return "CLOSE_ALL";
        default:                        return "UNKNOWN";
    }
}

// Convert strategy quality enum to string
string StrategyQualityToString(ENUM_STRATEGY_QUALITY quality)
{
    switch(quality)
    {
        case STRATEGY_QUALITY_F:        return "F";
        case STRATEGY_QUALITY_D:        return "D";
        case STRATEGY_QUALITY_C:        return "C";
        case STRATEGY_QUALITY_B:        return "B";
        case STRATEGY_QUALITY_A:        return "A";
        case STRATEGY_QUALITY_A_PLUS:   return "A+";
        default:                        return "UNKNOWN";
    }
}

// Get strategy quality from Kelly score
ENUM_STRATEGY_QUALITY GetStrategyQualityFromKelly(double kellyScore)
{
    if(kellyScore >= 0.9)       return STRATEGY_QUALITY_A_PLUS;
    if(kellyScore >= 0.8)       return STRATEGY_QUALITY_A;
    if(kellyScore >= 0.6)       return STRATEGY_QUALITY_B;
    if(kellyScore >= 0.4)       return STRATEGY_QUALITY_C;
    if(kellyScore >= 0.2)       return STRATEGY_QUALITY_D;
    return STRATEGY_QUALITY_F;
}

#endif // __QSL_TYPES_MQH__
//+------------------------------------------------------------------+
