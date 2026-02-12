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
//| Input Parameters for Tiered Risk Engine                          |
//+------------------------------------------------------------------+
input double InpGrowthModeCeiling = 1000.0;    // Growth Tier Ceiling ($)
input double InpScalingModeCeiling = 5000.0;   // Scaling Tier Ceiling ($)
input double InpFixedRiskAmount = 5.0;         // Fixed Risk Amount ($)

//+------------------------------------------------------------------+
//| Constants                                                         |
//+------------------------------------------------------------------+
#define QM_KELLY_MAX_FRACTION     0.25    // Maximum 25% of equity at risk

//+------------------------------------------------------------------+
//| V8 Tiered Risk Engine - Risk Tier Enumeration                    |
//+------------------------------------------------------------------+
enum ENUM_RISK_TIER
{
    TIER_GROWTH,      // $100-$1K: Dynamic 3% risk with $2 floor
    TIER_SCALING,     // $1K-$5K: Kelly percentage-based risk
    TIER_GUARDIAN     // $5K+: Kelly + Quadratic Throttle
};

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
        // V8 Tiered Risk Engine - Initialize from Input Parameters
        m_growthModeCeiling = InpGrowthModeCeiling;      // Default: $1,000
        m_scalingModeCeiling = InpScalingModeCeiling;    // Default: $5,000
        m_growthPercent = 3.0;                           // 3% Aggressive Risk for Growth Tier
        m_fixedFloorAmount = InpFixedRiskAmount;         // Default: $5.00
        
        Print("[QMKellySizer] Initialized with: GrowthCeiling=$", m_growthModeCeiling,
              " ScalingCeiling=$", m_scalingModeCeiling, 
              " FixedRisk=$", m_fixedFloorAmount);
    }
    
    //+------------------------------------------------------------------+
    //| V8: Set Tiered Risk Parameters                                   |
    //+------------------------------------------------------------------+
    void SetTieredRiskParams(double growthCeiling, double scalingCeiling, 
                            double growthPercent, double fixedFloor)
    {
        m_growthModeCeiling = growthCeiling;
        m_scalingModeCeiling = scalingCeiling;
        m_growthPercent = growthPercent;
        m_fixedFloorAmount = fixedFloor;
        Print("[QMKellySizer] Tiered Risk Params: Growth=", growthCeiling, 
              " Scaling=", scalingCeiling, " GrowthPct=", growthPercent, 
              "% Floor=$", fixedFloor);
    }

    //+------------------------------------------------------------------+
    //| V8: Get Risk Amount for Growth Tier (Dynamic Aggressive)         |
    //|                                                                   |
    //| Implements Dynamic Aggressive model with hard floor:             |
    //|   RiskAmount = MathMax(AccountEquity * GrowthPercent, FixedFloorAmount) |
    //|                                                                   |
    //| Examples:                                                         |
    //|   - $100 equity: Risk = $3.00 (3%)                               |
    //|   - $50 equity:  Risk = $2.00 (Floor kicks in)                   |
    //|   - $500 equity: Risk = $15.00 (Scales up)                       |
    //|                                                                   |
    //| @param equity  Current account equity                             |
    //| @return        Risk amount in dollars                             |
    //+------------------------------------------------------------------+
    double GetRiskAmount(double equity)
    {
        if(equity <= 0.0)
        {
            Print("[QMKellySizer] Error: Equity must be positive, got ", equity);
            return 0.0;
        }
        
        // Calculate percentage-based risk
        double percentRisk = equity * (m_growthPercent / 100.0);
        
        // Apply hard floor
        double riskAmount = MathMax(percentRisk, m_fixedFloorAmount);
        
        Print("[QMKellySizer] GetRiskAmount: equity=$", equity, 
              " percentRisk=$", percentRisk, " floor=$", m_fixedFloorAmount,
              " finalRisk=$", riskAmount);
        
        return riskAmount;
    }
    
    //+------------------------------------------------------------------+
    //| V8: Determine Risk Tier Based on Equity                          |
    //|                                                                   |
    //| Determines which risk tier to use based on account equity.       |
    //|                                                                   |
    //| @param equity  Current account equity                             |
    //| @return        Risk tier (TIER_GROWTH, TIER_SCALING, TIER_GUARDIAN)|
    //+------------------------------------------------------------------+
    ENUM_RISK_TIER DetermineRiskTier(double equity)
    {
        if(equity < m_growthModeCeiling)
        {
            return TIER_GROWTH;
        }
        else if(equity < m_scalingModeCeiling)
        {
            return TIER_SCALING;
        }
        else
        {
            return TIER_GUARDIAN;
        }
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
    //| V8: Calculate Position Size with Tiered Risk Logic               |
    //|                                                                   |
    //| Implements three-tier risk system based on account equity:       |
    //|   - Growth Tier ($100-$1K): Dynamic 3% risk with $2 floor        |
    //|   - Scaling Tier ($1K-$5K): Kelly percentage-based risk          |
    //|   - Guardian Tier ($5K+): Kelly + Quadratic Throttle             |
    //|                                                                   |
    //| @param equity       Current account equity                        |
    //| @param stopLossPips Stop loss distance in pips                    |
    //| @param tickValue    Monetary value per pip per lot                |
    //| @param currentLoss  Current daily loss (for Guardian tier)        |
    //| @param maxLoss      Maximum daily loss limit (for Guardian tier)  |
    //| @param winRate      Win rate for Kelly calculation                |
    //| @param avgWin       Average win for Kelly calculation             |
    //| @param avgLoss      Average loss for Kelly calculation            |
    //| @return             Position size in lots                         |
    //+------------------------------------------------------------------+
    double CalculateTieredPositionSize(double equity, double stopLossPips, 
                                       double tickValue, double currentLoss = 0.0,
                                       double maxLoss = 0.0, double winRate = 0.55,
                                       double avgWin = 400.0, double avgLoss = 200.0)
    {
        ENUM_RISK_TIER tier = DetermineRiskTier(equity);
        
        Print("[QMKellySizer] Tiered Position Sizing: equity=", equity, 
              " tier=", EnumToString(tier));
        
        switch(tier)
        {
            case TIER_GROWTH:
                // Dynamic 3% risk with $2 floor
                return CalculateGrowthTierLots(equity, stopLossPips, tickValue);
                
            case TIER_SCALING:
                // Standard Kelly Criterion
                return CalculateScalingTierLots(equity, stopLossPips, tickValue,
                                               winRate, avgWin, avgLoss);
                
            case TIER_GUARDIAN:
                // Kelly + Quadratic Throttle
                return CalculateGuardianTierLots(equity, stopLossPips, tickValue,
                                                currentLoss, maxLoss, winRate, 
                                                avgWin, avgLoss);
        }
        
        return 0.0;
    }
    
    //+------------------------------------------------------------------+
    //| V8: Growth Tier Position Sizing (Dynamic Aggressive with Floor)  |
    //|                                                                   |
    //| Uses GetRiskAmount() to calculate dynamic risk with hard floor.  |
    //|                                                                   |
    //| @param equity       Current account equity                        |
    //| @param stopLossPips Stop loss distance in pips                    |
    //| @param tickValue    Monetary value per pip per lot                |
    //| @return             Position size in lots                         |
    //+------------------------------------------------------------------+
    double CalculateGrowthTierLots(double equity, double stopLossPips, double tickValue)
    {
        if(stopLossPips <= 0.0 || tickValue <= 0.0)
        {
            Print("[QMKellySizer] Error: Invalid stop loss or tick value");
            return 0.0;
        }
        
        // Get dynamic risk amount with floor
        double riskAmount = GetRiskAmount(equity);
        
        // Calculate risk per lot
        double riskPerLot = stopLossPips * tickValue;
        
        // Calculate lot size
        double lots = riskAmount / riskPerLot;
        
        Print("[QMKellySizer] Growth Tier: equity=$", equity,
              " riskAmount=$", riskAmount, " riskPerLot=$", riskPerLot, 
              " lots=", lots);
        
        return lots;
    }
    
    //+------------------------------------------------------------------+
    //| V8: Scaling Tier Position Sizing (Kelly Percentage)              |
    //+------------------------------------------------------------------+
    double CalculateScalingTierLots(double equity, double stopLossPips, 
                                    double tickValue, double winRate,
                                    double avgWin, double avgLoss)
    {
        // Calculate Kelly fraction
        double kellyFraction = CalculateKellyFraction(winRate, avgWin, avgLoss);
        
        if(kellyFraction <= 0.0)
        {
            Print("[QMKellySizer] Scaling Tier: Kelly fraction is zero or negative");
            return 0.0;
        }
        
        // Use full Kelly (riskPct = 1.0)
        double lots = CalculateLotSize(kellyFraction, equity, 1.0, 
                                      stopLossPips, tickValue);
        
        Print("[QMKellySizer] Scaling Tier: kelly=", kellyFraction, " lots=", lots);
        
        return lots;
    }
    
    //+------------------------------------------------------------------+
    //| V8: Guardian Tier Position Sizing (Kelly + Quadratic Throttle)   |
    //+------------------------------------------------------------------+
    double CalculateGuardianTierLots(double equity, double stopLossPips,
                                     double tickValue, double currentLoss,
                                     double maxLoss, double winRate,
                                     double avgWin, double avgLoss)
    {
        // Calculate base Kelly position size
        double kellyFraction = CalculateKellyFraction(winRate, avgWin, avgLoss);
        
        if(kellyFraction <= 0.0)
        {
            Print("[QMKellySizer] Guardian Tier: Kelly fraction is zero or negative");
            return 0.0;
        }
        
        double baseLots = CalculateLotSize(kellyFraction, equity, 1.0,
                                          stopLossPips, tickValue);
        
        // Apply Quadratic Throttle
        double throttledLots = ApplyQuadraticThrottle(baseLots, currentLoss, maxLoss);
        
        Print("[QMKellySizer] Guardian Tier: baseLots=", baseLots,
              " throttledLots=", throttledLots, " currentLoss=", currentLoss,
              " maxLoss=", maxLoss);
        
        return throttledLots;
    }
    
    //+------------------------------------------------------------------+
    //| V8: Apply Quadratic Throttle to Position Size                    |
    //|                                                                   |
    //| Implements quadratic throttle formula from PropFirm article:     |
    //|   Multiplier = ((MaxLoss - CurrentLoss) / MaxLoss) ^ 2           |
    //|                                                                   |
    //| @param baseSize     Base position size before throttling          |
    //| @param currentLoss  Current daily loss amount                     |
    //| @param maxLoss      Maximum daily loss limit                      |
    //| @return             Throttled position size                       |
    //+------------------------------------------------------------------+
    double ApplyQuadraticThrottle(double baseSize, double currentLoss, double maxLoss)
    {
        if(maxLoss <= 0.0)
        {
            Print("[QMKellySizer] Warning: Max loss is zero or negative, returning base size");
            return baseSize;
        }
        
        // Calculate remaining capacity
        double remainingCapacity = (maxLoss - currentLoss) / maxLoss;
        
        // Ensure remaining capacity is in [0, 1]
        if(remainingCapacity < 0.0) remainingCapacity = 0.0;
        if(remainingCapacity > 1.0) remainingCapacity = 1.0;
        
        // Apply quadratic throttle
        double multiplier = MathPow(remainingCapacity, 2);
        double throttledSize = baseSize * multiplier;
        
        Print("[QMKellySizer] Quadratic Throttle: remainingCapacity=", remainingCapacity,
              " multiplier=", multiplier, " throttledSize=", throttledSize);
        
        return throttledSize;
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
    
    // V8 Tiered Risk Engine Parameters
    double m_growthModeCeiling;    // Growth tier ceiling ($1,000)
    double m_scalingModeCeiling;   // Scaling tier ceiling ($5,000)
    double m_growthPercent;        // Growth tier percentage risk (3%)
    double m_fixedFloorAmount;     // Minimum risk amount floor ($2)
};

//+------------------------------------------------------------------+
//| Global: Create Kelly Sizer instance                                |
//+------------------------------------------------------------------+
QMKellySizer g_kellySizer;

//+------------------------------------------------------------------+
#endif // __QSL_KELLY_SIZER_MQH__
//+------------------------------------------------------------------+
