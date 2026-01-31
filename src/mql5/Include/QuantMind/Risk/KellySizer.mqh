//+------------------------------------------------------------------+
//|                                                       KellySizer.mqh |
//|                        QuantMind Standard Library - Risk Module    |
//|                                                                  |
//| Kelly criterion position sizing calculations for optimal risk     |
//| management. Implements the Kelly formula: f* = (bp - q) / b       |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Include Guards                                                    |
//+------------------------------------------------------------------+
#ifndef __QSL_KELLY_SIZER_MQH__
#define __QSL_KELLY_SIZER_MQH__

//+------------------------------------------------------------------+
//| Constants                                                         |
//+------------------------------------------------------------------+
#define QM_KELLY_MAX_FRACTION     0.25    // Maximum 25% of equity at risk

//+------------------------------------------------------------------+
//| Error Codes                                                       |
//+------------------------------------------------------------------+
#define QM_KELLY_OK                 0      // Calculation successful
#define QM_KELLY_INVALID_WIN_RATE   1      // Win rate not in [0, 1]
#define QM_KELLY_ZERO_AVG_WIN       2      // Average win is zero or negative
#define QM_KELLY_ZERO_AVG_LOSS      3      // Average loss is zero or negative
#define QM_KELLY_NEGATIVE_EV        4      // Negative expected value

//+------------------------------------------------------------------+
//| QMKellySizer Class                                                |
//|                                                                   |
//| Provides Kelly criterion calculations for optimal position sizing.|
//|                                                                   |
//| The Kelly criterion determines the optimal fraction of equity to  |
//| risk based on:                                                    |
//|   - Win rate (probability of winning)                            |
//|   - Average win amount                                            |
//|   - Average loss amount                                           |
//|                                                                   |
//| Formula: f* = (bp - q) / b                                        |
//|   where: b = avgWin / avgLoss (payoff ratio)                      |
//|          p = winRate (probability of win)                         |
//|          q = 1 - p (probability of loss)                          |
//|                                                                   |
//| Usage:                                                            |
//|   QMKellySizer kelly;                                             |
//|   double fraction = kelly.CalculateKellyFraction(0.55, 400, 200); |
//|   double lots = kelly.CalculateLotSize(fraction, 10000, 1.0, 10); |
//+------------------------------------------------------------------+
class QMKellySizer
{
public:
    //+------------------------------------------------------------------+
    //| Constructor                                                      |
    //+------------------------------------------------------------------+
    QMKellySizer()
    {
        m_lastError = QM_KELLY_OK;
    }

    //+------------------------------------------------------------------+
    //| Calculate Kelly Fraction                                         |
    //|                                                                   |
    //| Implements the Kelly criterion formula: f* = (bp - q) / b        |
    //|                                                                   |
    //| @param winRate  Probability of winning (0.0 to 1.0)              |
    //| @param avgWin   Average profit amount per winning trade           |
    //| @param avgLoss  Average loss amount per losing trade              |
    //| @return         Optimal fraction of equity to risk (capped at 25%)|
    //+------------------------------------------------------------------+
    double CalculateKellyFraction(double winRate, double avgWin, double avgLoss)
    {
        m_lastError = QM_KELLY_OK;

        // Validate win rate
        if(winRate < 0.0 || winRate > 1.0)
        {
            m_lastError = QM_KELLY_INVALID_WIN_RATE;
            Print("[QMKellySizer] Error: Win rate must be between 0 and 1, got ", winRate);
            return 0.0;
        }

        // Validate average win
        if(avgWin <= 0.0)
        {
            m_lastError = QM_KELLY_ZERO_AVG_WIN;
            Print("[QMKellySizer] Error: Average win must be positive, got ", avgWin);
            return 0.0;
        }

        // Validate average loss
        if(avgLoss <= 0.0)
        {
            m_lastError = QM_KELLY_ZERO_AVG_LOSS;
            Print("[QMKellySizer] Error: Average loss must be positive, got ", avgLoss);
            return 0.0;
        }

        // Calculate payoff ratio (b = avgWin / avgLoss)
        double b = avgWin / avgLoss;

        // Calculate probabilities
        double p = winRate;        // Probability of win
        double q = 1.0 - winRate;  // Probability of loss

        // Kelly formula: f* = (bp - q) / b
        double rawKelly = (b * p - q) / b;

        // Check for negative expected value
        if(rawKelly < 0.0)
        {
            m_lastError = QM_KELLY_NEGATIVE_EV;
            Print("[QMKellySizer] Warning: Negative expected value (", rawKelly, "), returning 0");
            return 0.0;
        }

        // Cap at maximum fraction (25% of equity)
        double finalKelly = MathMin(rawKelly, QM_KELLY_MAX_FRACTION);

        // Log calculation (useful for debugging)
        Print("[QMKellySizer] Kelly: winRate=", winRate, " b=", b,
              " rawKelly=", rawKelly, " finalKelly=", finalKelly);

        return finalKelly;
    }

    //+------------------------------------------------------------------+
    //| Calculate Lot Size                                               |
    //|                                                                   |
    //| Converts Kelly fraction and equity into position size in lots.   |
    //|                                                                   |
    //| Formula: lots = (equity * kellyFraction * riskPct) /             |
    //|                    (stopLossPips * tickValue)                    |
    //|                                                                   |
    //| @param kellyFraction Kelly fraction from CalculateKellyFraction() |
    //| @param equity       Account equity in account currency            |
    //| @param riskPct      Risk percentage (1.0 = 100% of Kelly,         |
    //|                      0.5 = Half-Kelly)                           |
    //| @param stopLossPips Stop loss distance in pips                    |
    //| @param tickValue    Monetary value per pip per lot                |
    //| @return             Position size in lots                         |
    //+------------------------------------------------------------------+
    double CalculateLotSize(double kellyFraction, double equity,
                            double riskPct, double stopLossPips,
                            double tickValue)
    {
        m_lastError = QM_KELLY_OK;

        // Validate inputs
        if(kellyFraction <= 0.0)
        {
            Print("[QMKellySizer] Warning: Kelly fraction is zero or negative, returning 0 lots");
            return 0.0;
        }

        if(equity <= 0.0)
        {
            Print("[QMKellySizer] Error: Equity must be positive, got ", equity);
            return 0.0;
        }

        if(riskPct <= 0.0)
        {
            Print("[QMKellySizer] Warning: Risk percentage is zero or negative, returning 0 lots");
            return 0.0;
        }

        if(stopLossPips <= 0.0)
        {
            Print("[QMKellySizer] Error: Stop loss pips must be positive, got ", stopLossPips);
            return 0.0;
        }

        if(tickValue <= 0.0)
        {
            Print("[QMKellySizer] Error: Tick value must be positive, got ", tickValue);
            return 0.0;
        }

        // Calculate risk amount
        double riskAmount = equity * kellyFraction * riskPct;

        // Calculate risk per lot
        double riskPerLot = stopLossPips * tickValue;

        // Calculate lot size
        double lots = riskAmount / riskPerLot;

        // Log calculation
        Print("[QMKellySizer] LotSize: equity=", equity, " kelly=", kellyFraction,
              " riskAmount=", riskAmount, " riskPerLot=", riskPerLot, " lots=", lots);

        return lots;
    }

    //+------------------------------------------------------------------+
    //| Get Last Error                                                   |
    //|                                                                   |
    //| Returns the error code from the last operation.                   |
    //|                                                                   |
    //| @return Error code (QM_KELLY_OK if no error)                      |
    //+------------------------------------------------------------------+
    int GetLastError(void)
    {
        return m_lastError;
    }

    //+------------------------------------------------------------------+
    //| Get Error Message                                                 |
    //|                                                                   |
    //| Returns a human-readable error message.                           |
    //|                                                                   |
    //| @return Error description string                                  |
    //+------------------------------------------------------------------+
    string GetErrorMessage(void)
    {
        switch(m_lastError)
        {
            case QM_KELLY_OK:
                return "No error";
            case QM_KELLY_INVALID_WIN_RATE:
                return "Invalid win rate (must be 0-1)";
            case QM_KELLY_ZERO_AVG_WIN:
                return "Average win must be positive";
            case QM_KELLY_ZERO_AVG_LOSS:
                return "Average loss must be positive";
            case QM_KELLY_NEGATIVE_EV:
                return "Negative expected value";
            default:
                return "Unknown error";
        }
    }

private:
    //+------------------------------------------------------------------+
    //| Private Members                                                   |
    //+------------------------------------------------------------------+
    int m_lastError;  // Last error code
};

//+------------------------------------------------------------------+
//| Global: Create Kelly Sizer instance                                |
//+------------------------------------------------------------------+
QMKellySizer g_kellySizer;

//+------------------------------------------------------------------+
#endif // __QSL_KELLY_SIZER_MQH__
//+------------------------------------------------------------------+
