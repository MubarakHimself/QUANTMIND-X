//+------------------------------------------------------------------+
//|                                                    PropManager.mqh |
//|                        QuantMind Standard Library (QSL) - Risk    |
//|                        Prop Firm Risk Management Module           |
//|                                                                  |
//| Manages daily drawdown tracking, hard stop enforcement, news     |
//| guard functionality, and quadratic throttle calculations for     |
//| prop firm trading accounts.                                      |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

#ifndef __QSL_PROP_MANAGER_MQH__
#define __QSL_PROP_MANAGER_MQH__

// Include dependencies
#include <QuantMind/Core/Constants.mqh>
#include <QuantMind/Core/Types.mqh>

//+------------------------------------------------------------------+
//| CPropManager Class                                                |
//|                                                                   |
//| Comprehensive prop firm risk management including:               |
//| - Daily drawdown tracking with high water mark                   |
//| - Hard stop enforcement (4.5% threshold with 1% buffer)          |
//| - News guard (KILL_ZONE) functionality                           |
//| - Quadratic throttle calculation                                 |
//| - Integration with Python backend for persistence                |
//+------------------------------------------------------------------+
class CPropManager
{
private:
    // Account identification
    string            m_accountId;
    string            m_firmName;
    
    // Daily tracking
    double            m_startBalance;
    double            m_highWaterMark;
    double            m_dailyLossLimit;
    double            m_maxDrawdown;
    datetime          m_lastResetDate;
    
    // Risk status
    bool              m_hardStopActive;
    bool              m_newsGuardActive;
    ENUM_RISK_STATUS  m_riskStatus;
    
    // Statistics
    int               m_tradesCount;
    double            m_dailyPnL;
    double            m_currentDrawdown;
    
public:
    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    CPropManager()
    {
        m_accountId = "";
        m_firmName = "";
        m_startBalance = 0.0;
        m_highWaterMark = 0.0;
        m_dailyLossLimit = QM_DAILY_LOSS_LIMIT_PCT;
        m_maxDrawdown = QM_MAX_DRAWDOWN_PCT;
        m_lastResetDate = 0;
        m_hardStopActive = false;
        m_newsGuardActive = false;
        m_riskStatus = RISK_STATUS_NORMAL;
        m_tradesCount = 0;
        m_dailyPnL = 0.0;
        m_currentDrawdown = 0.0;
    }
    
    //+------------------------------------------------------------------+
    //| Destructor                                                       |
    //+------------------------------------------------------------------+
    ~CPropManager()
    {
    }
    
    //+------------------------------------------------------------------+
    //| Initialize PropManager for a specific account                   |
    //|                                                                  |
    //| @param accountId Account identifier                             |
    //| @param firmName Prop firm name                                  |
    //| @param dailyLossLimit Daily loss limit percentage               |
    //| @return true if initialization successful                       |
    //+------------------------------------------------------------------+
    bool Initialize(string accountId, string firmName, double dailyLossLimit = QM_DAILY_LOSS_LIMIT_PCT)
    {
        m_accountId = accountId;
        m_firmName = firmName;
        m_dailyLossLimit = dailyLossLimit;
        m_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        m_highWaterMark = m_startBalance;
        m_lastResetDate = TimeCurrent();
        
        Print("[CPropManager] Initialized for account ", m_accountId, 
              " (", m_firmName, ") with daily loss limit ", m_dailyLossLimit, "%");
        
        return true;
    }
    
    //+------------------------------------------------------------------+
    //| Update daily state (call on each tick or periodically)          |
    //|                                                                  |
    //| Checks if new trading day has started and resets daily metrics  |
    //+------------------------------------------------------------------+
    void Update()
    {
        // Check if new trading day
        datetime currentDate = TimeCurrent();
        MqlDateTime dt1, dt2;
        TimeToStruct(m_lastResetDate, dt1);
        TimeToStruct(currentDate, dt2);
        
        if(dt1.day != dt2.day || dt1.mon != dt2.mon || dt1.year != dt2.year)
        {
            ResetDailyMetrics();
        }
        
        // Update current state
        double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
        m_dailyPnL = currentEquity - m_startBalance;
        m_currentDrawdown = CalculateDailyDrawdown(m_startBalance, currentEquity);
        
        // Update high water mark
        if(currentEquity > m_highWaterMark)
        {
            m_highWaterMark = currentEquity;
        }
        
        // Check hard stop condition
        CheckHardStop();
        
        // Update risk status
        UpdateRiskStatus();
    }
    
    //+------------------------------------------------------------------+
    //| Reset daily metrics for new trading day                         |
    //+------------------------------------------------------------------+
    void ResetDailyMetrics()
    {
        m_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        m_highWaterMark = m_startBalance;
        m_dailyPnL = 0.0;
        m_currentDrawdown = 0.0;
        m_tradesCount = 0;
        m_hardStopActive = false;
        m_riskStatus = RISK_STATUS_NORMAL;
        m_lastResetDate = TimeCurrent();
        
        Print("[CPropManager] Daily metrics reset for new trading day");
    }
    
    //+------------------------------------------------------------------+
    //| Calculate daily drawdown percentage                             |
    //|                                                                  |
    //| Formula: ((StartBalance - CurrentEquity) / StartBalance) * 100  |
    //|                                                                  |
    //| @param startBalance Daily starting balance                      |
    //| @param currentEquity Current account equity                     |
    //| @return Drawdown percentage (positive when losing)              |
    //+------------------------------------------------------------------+
    double CalculateDailyDrawdown(double startBalance, double currentEquity)
    {
        if(startBalance <= 0.0)
        {
            return 0.0;
        }
        
        double drawdown = ((startBalance - currentEquity) / startBalance) * 100.0;
        return drawdown;
    }
    
    //+------------------------------------------------------------------+
    //| Check if hard stop should be activated                          |
    //|                                                                  |
    //| Hard stop triggers when drawdown >= 4.5% (effective limit)      |
    //| This provides 1% buffer before hitting 5% daily limit           |
    //+------------------------------------------------------------------+
    void CheckHardStop()
    {
        if(m_currentDrawdown >= QM_EFFECTIVE_LIMIT_PCT)
        {
            if(!m_hardStopActive)
            {
                m_hardStopActive = true;
                m_riskStatus = RISK_STATUS_HARD_STOP;
                
                Print("[CPropManager] HARD STOP ACTIVATED - Drawdown: ", 
                      DoubleToString(m_currentDrawdown, 2), "% >= ",
                      DoubleToString(QM_EFFECTIVE_LIMIT_PCT, 2), "%");
            }
        }
    }
    
    //+------------------------------------------------------------------+
    //| Activate/deactivate news guard                                  |
    //|                                                                  |
    //| @param active true to activate news guard, false to deactivate  |
    //+------------------------------------------------------------------+
    void SetNewsGuard(bool active)
    {
        if(m_newsGuardActive != active)
        {
            m_newsGuardActive = active;
            
            if(active)
            {
                m_riskStatus = RISK_STATUS_NEWS_GUARD;
                Print("[CPropManager] NEWS GUARD ACTIVATED (KILL_ZONE)");
            }
            else
            {
                UpdateRiskStatus();
                Print("[CPropManager] NEWS GUARD DEACTIVATED");
            }
        }
    }
    
    //+------------------------------------------------------------------+
    //| Calculate quadratic throttle multiplier                         |
    //|                                                                  |
    //| Formula: Multiplier = ((MaxLoss - CurrentLoss) / MaxLoss) ^ 2   |
    //|                                                                  |
    //| This provides smooth risk reduction as drawdown increases:      |
    //| - 0% drawdown = 1.0 multiplier (full risk)                      |
    //| - 2% drawdown = 0.64 multiplier (64% risk)                      |
    //| - 3% drawdown = 0.36 multiplier (36% risk)                      |
    //| - 4% drawdown = 0.04 multiplier (4% risk)                       |
    //| - 4.5%+ drawdown = 0.0 multiplier (hard stop)                   |
    //|                                                                  |
    //| @return Risk multiplier (0.0 to 1.0)                            |
    //+------------------------------------------------------------------+
    double CalculateQuadraticThrottle()
    {
        // Hard stop or news guard = zero risk
        if(m_hardStopActive || m_newsGuardActive)
        {
            return 0.0;
        }
        
        // Calculate remaining capacity
        double maxLoss = m_dailyLossLimit;
        double currentLoss = m_currentDrawdown;
        
        // Ensure we don't divide by zero
        if(maxLoss <= 0.0)
        {
            return 0.0;
        }
        
        // Ensure current loss doesn't exceed max loss
        if(currentLoss >= maxLoss)
        {
            return 0.0;
        }
        
        // Quadratic throttle formula
        double remainingCapacity = (maxLoss - currentLoss) / maxLoss;
        double multiplier = MathPow(remainingCapacity, QM_THROTTLE_EXPONENT);
        
        // Clamp to valid range
        multiplier = QM_CLAMP(multiplier, QM_THROTTLE_MIN_MULTIPLIER, QM_THROTTLE_MAX_MULTIPLIER);
        
        return multiplier;
    }
    
    //+------------------------------------------------------------------+
    //| Check if trading is allowed                                     |
    //|                                                                  |
    //| @return true if trading allowed, false if stopped               |
    //+------------------------------------------------------------------+
    bool IsTradingAllowed()
    {
        // Update state first
        Update();
        
        // Check hard stop
        if(m_hardStopActive)
        {
            return false;
        }
        
        // Check news guard
        if(m_newsGuardActive)
        {
            return false;
        }
        
        return true;
    }
    
    //+------------------------------------------------------------------+
    //| Get current risk multiplier (combines throttle and status)      |
    //|                                                                  |
    //| @return Current risk multiplier (0.0 to 1.0)                    |
    //+------------------------------------------------------------------+
    double GetRiskMultiplier()
    {
        Update();
        return CalculateQuadraticThrottle();
    }
    
    //+------------------------------------------------------------------+
    //| Update risk status based on current conditions                  |
    //+------------------------------------------------------------------+
    void UpdateRiskStatus()
    {
        if(m_hardStopActive)
        {
            m_riskStatus = RISK_STATUS_HARD_STOP;
        }
        else if(m_newsGuardActive)
        {
            m_riskStatus = RISK_STATUS_NEWS_GUARD;
        }
        else if(m_currentDrawdown > 2.0)
        {
            m_riskStatus = RISK_STATUS_THROTTLED;
        }
        else
        {
            m_riskStatus = RISK_STATUS_NORMAL;
        }
    }
    
    //+------------------------------------------------------------------+
    //| Get current account state                                       |
    //|                                                                  |
    //| @param state Output account state structure                     |
    //+------------------------------------------------------------------+
    void GetAccountState(SAccountState &state)
    {
        Update();
        
        state.accountId = m_accountId;
        state.balance = AccountInfoDouble(ACCOUNT_BALANCE);
        state.equity = AccountInfoDouble(ACCOUNT_EQUITY);
        state.freeMargin = AccountInfoDouble(ACCOUNT_FREEMARGIN);
        state.highWaterMark = m_highWaterMark;
        state.dailyPnL = m_dailyPnL;
        state.dailyDrawdown = m_currentDrawdown;
        state.dailyLossLimit = m_dailyLossLimit;
        state.maxDrawdown = m_maxDrawdown;
        state.tradesCount = m_tradesCount;
        state.lastUpdate = TimeCurrent();
        state.status = ACCOUNT_STATUS_ACTIVE;
        state.riskStatus = m_riskStatus;
        state.isKillZone = m_newsGuardActive;
    }
    
    //+------------------------------------------------------------------+
    //| Get risk status summary string                                  |
    //|                                                                  |
    //| @return Formatted status string                                 |
    //+------------------------------------------------------------------+
    string GetRiskStatusSummary()
    {
        Update();
        
        string summary = "[CPropManager] Risk Status:\n";
        summary += "  Account: " + m_accountId + " (" + m_firmName + ")\n";
        summary += "  Start Balance: " + DoubleToString(m_startBalance, 2) + "\n";
        summary += "  Current Equity: " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + "\n";
        summary += "  High Water Mark: " + DoubleToString(m_highWaterMark, 2) + "\n";
        summary += "  Daily P&L: " + DoubleToString(m_dailyPnL, 2) + "\n";
        summary += "  Daily Drawdown: " + DoubleToString(m_currentDrawdown, 2) + "%\n";
        summary += "  Daily Loss Limit: " + DoubleToString(m_dailyLossLimit, 2) + "%\n";
        summary += "  Risk Multiplier: " + DoubleToString(CalculateQuadraticThrottle(), 4) + "\n";
        summary += "  Risk Status: " + RiskStatusToString(m_riskStatus) + "\n";
        summary += "  Hard Stop: " + (m_hardStopActive ? "ACTIVE" : "INACTIVE") + "\n";
        summary += "  News Guard: " + (m_newsGuardActive ? "ACTIVE" : "INACTIVE") + "\n";
        summary += "  Trades Today: " + IntegerToString(m_tradesCount);
        
        return summary;
    }
    
    //+------------------------------------------------------------------+
    //| Increment trade counter                                         |
    //+------------------------------------------------------------------+
    void IncrementTradeCount()
    {
        m_tradesCount++;
    }
    
    //+------------------------------------------------------------------+
    //| Get current drawdown percentage                                 |
    //|                                                                  |
    //| @return Current drawdown percentage                             |
    //+------------------------------------------------------------------+
    double GetCurrentDrawdown()
    {
        Update();
        return m_currentDrawdown;
    }
    
    //+------------------------------------------------------------------+
    //| Get daily P&L                                                   |
    //|                                                                  |
    //| @return Daily profit/loss                                       |
    //+------------------------------------------------------------------+
    double GetDailyPnL()
    {
        Update();
        return m_dailyPnL;
    }
    
    //+------------------------------------------------------------------+
    //| Check if hard stop is active                                    |
    //|                                                                  |
    //| @return true if hard stop active                                |
    //+------------------------------------------------------------------+
    bool IsHardStopActive()
    {
        Update();
        return m_hardStopActive;
    }
    
    //+------------------------------------------------------------------+
    //| Check if news guard is active                                   |
    //|                                                                  |
    //| @return true if news guard active                               |
    //+------------------------------------------------------------------+
    bool IsNewsGuardActive()
    {
        return m_newsGuardActive;
    }
    
    //+------------------------------------------------------------------+
    //| Get risk status                                                  |
    //|                                                                  |
    //| @return Current risk status enum                                |
    //+------------------------------------------------------------------+
    ENUM_RISK_STATUS GetRiskStatus()
    {
        Update();
        return m_riskStatus;
    }
};

#endif // __QSL_PROP_MANAGER_MQH__
//+------------------------------------------------------------------+
