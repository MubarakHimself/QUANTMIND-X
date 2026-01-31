//+------------------------------------------------------------------+
//|                                                      PropManager.mqh |
//|                        QuantMind Standard Library (QSL) - Core      |
//|                        Prop Firm Risk Management Module            |
//|                                                                  |
//| Manages daily drawdown tracking, hard stop enforcement, and       |
//| high water mark calculations for prop firm trading accounts.      |
//| Integrates with DatabaseManager via Python bridge for persistence.|
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

#ifndef __QSL_PROP_MANAGER_MQH__
#define __QSL_PROP_MANAGER_MQH__

//+------------------------------------------------------------------+
//| Constants                                                         |
//+------------------------------------------------------------------+
// Daily loss limit percentage (5% is standard prop firm limit)
#define QM_DAILY_LOSS_LIMIT_PCT    5.0

// Hard stop buffer percentage (1% buffer from daily limit)
#define QM_HARD_STOP_BUFFER_PCT    1.0

// Effective hard stop limit (5% - 1% = 4%)
// Trading must stop when drawdown reaches 4% to avoid breaching 5% limit
#define QM_EFFECTIVE_LIMIT_PCT     4.0

//+------------------------------------------------------------------+
//| Prop Firm Manager Class                                           |
//|                                                                   |
//| Tracks daily drawdown, enforces hard stop limits, and maintains   |
//| high water mark for prop firm trading accounts.                   |
//+------------------------------------------------------------------+
class QMPropManager
{
private:
    double m_highWaterMark;        // Highest equity achieved today
    string m_accountId;            // Account identifier

public:
    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    QMPropManager()
    {
        m_highWaterMark = 0.0;
        m_accountId = "";
    }

    //+------------------------------------------------------------------+
    //| Destructor                                                       |
    //+------------------------------------------------------------------+
    ~QMPropManager()
    {
    }

    //+------------------------------------------------------------------+
    //| Initialize for a specific account                                |
    //|                                                                   |
    //| @param accountId Account identifier (e.g., account number)       |
    //| @param initialBalance Starting balance for the day               |
    //+------------------------------------------------------------------+
    void Initialize(string accountId, double initialBalance)
    {
        m_accountId = accountId;
        m_highWaterMark = initialBalance;
    }

    //+------------------------------------------------------------------+
    //| Calculate daily drawdown percentage                              |
    //|                                                                   |
    //| Formula: ((StartBalance - CurrentEquity) / StartBalance) * 100   |
    //|                                                                   |
    //| Returns positive value when at loss, negative when in profit.    |
    //|                                                                   |
    //| @param startBalance Daily starting balance                       |
    //| @param currentEquity Current account equity                      |
    //| @return Drawdown percentage                                      |
    //+------------------------------------------------------------------+
    double CalculateDailyDrawdown(double startBalance, double currentEquity)
    {
        // Prevent division by zero
        if(startBalance <= 0.0)
        {
            return 0.0;
        }

        double drawdown = ((startBalance - currentEquity) / startBalance) * 100.0;
        return drawdown;
    }

    //+------------------------------------------------------------------+
    //| Check if hard stop is breached                                   |
    //|                                                                   |
    //| Hard stop triggers when drawdown >= QM_EFFECTIVE_LIMIT_PCT (4%)  |
    //| This provides a 1% buffer before hitting the 5% daily limit.    |
    //|                                                                   |
    //| @param drawdownPct Current drawdown percentage                   |
    //| @return true if hard stop breached, false otherwise             |
    //+------------------------------------------------------------------+
    bool IsHardStopBreached(double drawdownPct)
    {
        return drawdownPct >= QM_EFFECTIVE_LIMIT_PCT;
    }

    //+------------------------------------------------------------------+
    //| Update high water mark                                           |
    //|                                                                   |
    //| The high water mark tracks the highest equity achieved during    |
    //| the trading day. Used for drawdown calculations and risk mgmt.  |
    //|                                                                   |
    //| @param currentEquity Current account equity                      |
    //| @param previousHWM Previous high water mark value                |
    //| @return New high water mark (max of current and previous)        |
    //+------------------------------------------------------------------+
    double UpdateHighWaterMark(double currentEquity, double previousHWM)
    {
        double newHWM = currentEquity;
        if(currentEquity < previousHWM)
        {
            newHWM = previousHWM;
        }
        return newHWM;
    }

    //+------------------------------------------------------------------+
    //| Get current high water mark                                      |
    //|                                                                   |
    //| @return Current high water mark value                            |
    //+------------------------------------------------------------------+
    double GetHighWaterMark()
    {
        return m_highWaterMark;
    }

    //+------------------------------------------------------------------+
    //| Save daily snapshot to database                                  |
    //|                                                                   |
    //| Persists account state to database via Python bridge.            |
    //| This ensures risk rules persist across platform restarts.        |
    //|                                                                   |
    //| @param accountId Account identifier                              |
    //| @param equity Current account equity                             |
    //| @param balance Current account balance                           |
    //| @return true if snapshot saved successfully, false otherwise     |
    //+------------------------------------------------------------------+
    bool SaveSnapshot(string accountId, double equity, double balance)
    {
        // In production, this calls the Python bridge to DatabaseManager
        // For now, we log the snapshot for debugging
        Print("[QMPropManager] Snapshot - Account: ", accountId,
              ", Equity: ", DoubleToString(equity, 2),
              ", Balance: ", DoubleToString(balance, 2),
              ", HWM: ", DoubleToString(m_highWaterMark, 2));

        // TODO: Implement Python bridge call to DatabaseManager.save_daily_snapshot()
        // The Python bridge will use WebRequest or Named Pipes to communicate

        return true;
    }

    //+------------------------------------------------------------------+
    //| Check trading status based on current drawdown                   |
    //|                                                                   |
    //| Provides a single method to check if trading should continue.   |
    //|                                                                   |
    //| @param startBalance Daily starting balance                       |
    //| @param currentEquity Current account equity                      |
    //| @return true if trading allowed, false if hard stop triggered    |
    //+------------------------------------------------------------------+
    bool IsTradingAllowed(double startBalance, double currentEquity)
    {
        double drawdown = CalculateDailyDrawdown(startBalance, currentEquity);

        if(IsHardStopBreached(drawdown))
        {
            Print("[QMPropManager] HARD STOP TRIGGERED - Drawdown: ",
                  DoubleToString(drawdown, 2), "% >= ",
                  DoubleToString(QM_EFFECTIVE_LIMIT_PCT, 2), "%");
            return false;
        }

        // Update HWM if we're at a new high
        if(currentEquity > m_highWaterMark)
        {
            m_highWaterMark = currentEquity;
        }

        return true;
    }

    //+------------------------------------------------------------------+
    //| Get risk status summary for logging                              |
    //|                                                                   |
    //| @param startBalance Daily starting balance                       |
    //| @param currentEquity Current account equity                      |
    //| @return Formatted status string                                  |
    //+------------------------------------------------------------------+
    string GetRiskStatus(double startBalance, double currentEquity)
    {
        double drawdown = CalculateDailyDrawdown(startBalance, currentEquity);
        bool breached = IsHardStopBreached(drawdown);

        string status = "[QMPropManager] Risk Status:\n";
        status += "  Start Balance: " + DoubleToString(startBalance, 2) + "\n";
        status += "  Current Equity: " + DoubleToString(currentEquity, 2) + "\n";
        status += "  High Water Mark: " + DoubleToString(m_highWaterMark, 2) + "\n";
        status += "  Drawdown: " + DoubleToString(drawdown, 2) + "%\n";
        status += "  Hard Stop: " + (breached ? "BREACHED" : "OK");

        return status;
    }
};

#endif // __QSL_PROP_MANAGER_MQH__
//+------------------------------------------------------------------+
