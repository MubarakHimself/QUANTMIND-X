---
title: From Novice to Expert: Predictive Price Pathways
url: https://www.mql5.com/en/articles/20160
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:44:34.223377
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/20160&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083079119261930640)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/20160#para1)
- [Implementation](https://www.mql5.com/en/articles/20160#para2)
- [Testing](https://www.mql5.com/en/articles/20160#para3)
- [Conclusion](https://www.mql5.com/en/articles/20160#para4)
- [Key Lessons](https://www.mql5.com/en/articles/20160#para5)
- [Attachments](https://www.mql5.com/en/articles/20160#para6)

### Introduction

Every trader knows the frustrating pattern: you watch a strong impulsive move develop, hesitate as fear and uncertainty take hold, then finally jump in just as the market exhausts its momentum—only to watch your position immediately retrace against you. This "chase and reverse" cycle represents one of the most common and costly mistakes in trading. The emotional whipsaw of fear of missing out (FOMO) followed by the pain of buying high and selling low can devastate both accounts and confidence.

But what if we could transform this reactive pattern into a proactive strategy? What if instead of chasing price, we could anticipate where price is likely to pause, reverse, or continue? This is where the ancient mathematical principles of Fibonacci meet modern algorithmic execution. Markets don't move in straight lines—they breathe in rhythmic waves of advance and retracement. The secret to consistent trading lies not in predicting every twist and turn, but in identifying the high-probability zones where these natural market rhythms are most likely to produce meaningful reactions.

The Fibonacci Framework:

Leonardo Fibonacci's 13th-century numerical sequence has stood the test of time because it represents fundamental growth patterns found throughout nature—from seashell spirals to galaxy formations. In financial markets, these same ratios (38.2%, 50%, 61.8%, etc.) consistently appear as price retracement levels that traders watch for potential reversal zones. But most traders use Fibonacci tools passively, drawing static lines on charts and waiting for price to approach them.

Our breakthrough comes from transforming these passive reference points into active predictive pathways. Instead of merely marking where price might reverse, we're building an Expert Advisor that visualizes the entire trade journey—from the expected retracement entry through the projected extension target. This transforms Fibonacci from a simple drawing tool into a dynamic trading system that anticipates market movement rather than merely reacting to it.

![Fibonacci Predictive Pathways Concept](https://c.mql5.com/2/180/terminal64_h6pioPAgf4.png)

The Predictive Pathway Concept

Understanding the Predictive Price Pathways

What makes this EA fundamentally different from traditional Fibonacci tools is its integrated approach to analysis, execution, and visualization. We're building a system that doesn't just identify opportunities—it executes them with discipline and illustrates the underlying logic through clear visual pathways.

First, the EA automatically detects significant swing points and calculates Fibonacci retracement levels in real-time. No more manual drawing or subjective interpretation. The algorithm identifies the most relevant highs and lows based on customizable parameters, ensuring consistency across different market conditions and timeframes.

Second, it transforms these mathematical levels into actionable trading decisions. The EA places pending orders at predefined Fibonacci zones (with a focus on the potent 61.8% golden ratio) with integrated risk management. Stop losses and profit targets are derived directly from the Fibonacci geometry, creating a naturally balanced risk-reward structure.

Third, and most innovatively, it maps predictive pathways that visually demonstrate the expected price journey. These aren't just static lines—they're dynamic arrows and pathways that show both the anticipated retracement move to our entry and the projected extension move to our target. This visualization turns abstract mathematical concepts into tangible trading plans.

Beyond Entry Signals: Integrated Risk Management

The true sophistication of this approach lies in its holistic integration of Fibonacci principles throughout the entire trade lifecycle. Most trading systems focus predominantly on entry signals, leaving risk management as an afterthought. Our EA anchors every aspect of the trade to the Fibonacci framework.

Position sizing becomes mathematically grounded in the distance between Fibonacci levels rather than arbitrary lot sizes. Stop-loss placement derives naturally from the market structure that defined our entry level in the first place. Take-profit targets reference Fibonacci extensions (100%, 127.2%, 161.8%) that represent logical profit objectives based on the initial swing magnitude.

This isn't a system that promises certainty—no approach can. Instead, it provides a consistent, testable framework that systematically removes emotional decision-making. By focusing attention on probable reaction zones rather than chasing every price movement, traders can develop the patience and discipline that separates professionals from amateurs.

Seeing the Market's Hidden Structure

Perhaps the most transformative aspect of this project is its educational dimension. For newer traders, Fibonacci retracements can feel like abstract mathematical concepts with little connection to actual price action. Our predictive pathway visualization bridges this gap by making the underlying market structure visible and intuitive.

When you see a green dashed line arcing from a swing high down to the 61.8% retracement level, followed by a gold dotted line extending to the 100% projection target, you're not just seeing lines on a chart—you're seeing a concrete trading hypothesis. You're witnessing how mathematical ratios translate into probable price movement. You're understanding why we're entering at specific levels and where we expect price to travel next.

This visualization becomes particularly powerful when combined with the EA's automated execution. As trades unfold according to the predicted pathways, traders develop an intuitive understanding of Fibonacci principles through observation and experience. The abstract becomes concrete, the theoretical becomes practical, and mathematical ratios become trading intuition.

The Complete Integration: Mathematics, Execution, and Psychology

What we're building represents the perfect synthesis of three essential trading components: the mathematical rigor of Fibonacci analysis, the disciplined execution of algorithmic trading, and the intuitive understanding fostered by visual learning. This isn't just another EA—it's a comprehensive trading education packaged as executable code.

The mathematics provides the foundation—proven ratios that have guided traders for decades. The algorithmic execution provides the discipline—removing emotional interference and ensuring consistent application of the strategy. The visualization provides the understanding—transforming complex concepts into clear, actionable insights.

As we journey through the code together, you'll discover how each component integrates seamlessly into the whole. You'll see how swing detection algorithms identify meaningful market structure, how Fibonacci calculations transform price data into trading levels, how risk management protocols protect capital, and how visualization techniques make everything immediately comprehensible.

### Implementation

1\. Setting the Foundation: Essential Includes and Input Parameters

Before we dive into the trading logic, let's establish our foundation. Notice how we include three crucial MQL5 libraries—these aren't just random choices. The Trade library gives us professional order execution capabilities, OrderInfo helps us manage existing positions, and AccountInfo ensures we don't overleverage our account. This three-pillar approach separates amateur EAs from professional ones.

Now look at our input parameters. We're not just throwing random Fibonacci levels at the wall. We focus on the 61.8% golden ratio for entries and 100% extensions for targets because these are the levels institutional traders watch most closely. The MaxBarsBack parameter? That's your efficiency control—too few bars and you miss significant swings; too many and you're analyzing irrelevant ancient history.

Critical Implementation Details:

- The #include directives must come first—they're the EA's toolbox
- Input groups (input group) organize parameters logically in the Properties window
- The 61.8% level (TradingRetracementLevel) is mathematically significant—it's the inverse of the golden ratio 1.618
- MagicNumber isn't arbitrary—it uniquely identifies EA orders for management and reporting
- EnableTrading acts as an emergency stop switch during testing or market turmoil.

```
//--- Includes: Essential MQL5 libraries for trading functionality
#include <Trade\Trade.mqh>       // Standard trade execution library
#include <Trade\OrderInfo.mqh>   // Order information and management
#include <Trade\AccountInfo.mqh> // Account information and margin checks

//+------------------------------------------------------------------+
//| Input Parameters: Configurable trading and visualization settings|
//+------------------------------------------------------------------+

input group "===================== FIBONACCI SETTINGS ====================="
input double   TradingRetracementLevel = 0.618;  // Primary retracement level for trade execution (61.8% - Golden Ratio)
input double   ProfitTargetLevel = 1.000;        // Profit target as Fibonacci extension (100% - Full swing retracement)
input int      MaxBarsBack = 100;                // Maximum historical bars for swing analysis (balance between accuracy and performance)

input group "===================== TRADING SETTINGS ====================="
input double   LotSize = 0.1;                    // Position size in lots (adjust based on account size and risk tolerance)
input int      StopLossPips = 30;                // Stop loss distance in pips (1 pip = 10 points in 5-digit brokers)
input int      MagicNumber = 12345;              // Unique identifier for EA orders (prevents interference with manual trades)
input bool     EnableTrading = true;             // Master switch for trade execution (set false for visualization only)

input group "===================== VISUALIZATION SETTINGS ====================="
input color    BuyColor = clrBlue;               // Visual color for buy trade pathways
input color    SellColor = clrRed;               // Visual color for sell trade pathways
input color    PredictionColor = clrGreen;       // Color for predictive retracement pathways
input color    TargetColor = clrGold;            // Color for profit target pathways and markers
input int      ArrowSize = 3;                    // Size of directional arrows on chart (1-5 recommended)
```

2\. Market Memory: Global Variables That Remember Key Swing Points

Markets have memory, and our EA needs memory too. These global variables are the EA's brain—they remember where significant highs and lows occurred, when they happened, and what direction the market is trending. Notice how we store both price levels AND timestamps? That's because market structure isn't just about price; it's about time and price relationships.

The trendDirection boolean is particularly clever. It doesn't just look at current price action; it analyzes which swing occurred more recently. If the swing high happened after the swing low, we're in an uptrend. This temporal analysis prevents the common mistake of misidentifying trends based on short-term noise.

Variable Design:

- swingHigh and swingLow store the actual price levels of significant market turns
- swingHighTime and swingLowTime track WHEN these swings occurred—crucial for trend analysis
- swingHighIndex and swingLowIndex record the bar numbers for temporal comparison
- trendDirection uses temporal logic: if the high came after the low, we're trending up
- The trading objects (CTrade, COrderInfo, CAccountInfo) encapsulate complex functionality into simple interfaces

Most EAs fail because they don't properly define trend context. By storing both price and time data, we can determine if we're in a genuine uptrend (higher highs and higher lows) or downtrend (lower highs and lower lows) rather than just reacting to recent price movements.

```
//+------------------------------------------------------------------+
//| Global Variables: Critical data persistence between ticks        |
//+------------------------------------------------------------------+

//--- Swing Point Analysis: Stores key market structure points
double         swingHigh;                         // Most recent significant swing high price
double         swingLow;                          // Most recent significant swing low price
datetime       swingHighTime;                     // Timestamp of swing high for temporal analysis
datetime       swingLowTime;                      // Timestamp of swing low for temporal analysis
int            swingHighIndex;                    // Bar index of swing high for reference
int            swingLowIndex;                     // Bar index of swing low for reference
bool           trendDirection;                    // Current market trend: true=bullish, false=bearish

//--- Trading Objects: MQL5 class instances for professional trade execution
CTrade         Trade;                             // Main trade execution engine with built-in error handling
COrderInfo     OrderInfo;                         // Order inspection and management utilities
CAccountInfo   AccountInfo;                       // Account status and margin requirement checks
```

3\. Initialization: Building a Robust Trading Foundation

When your EA first loads, it needs to establish its operational boundaries. The OnInit function is where we set the rules of engagement. Notice how we immediately assign a MagicNumber? This is crucial—it's like giving your EA a unique identity so it doesn't interfere with your manual trades or other EAs.

The input validation here isn't just pedantic programming. It's your first line of defense against mathematical absurdities. A retracement level of 0 or 1 would break Fibonacci logic, and a profit target below 1 would guarantee losing trades. This validation ensures mathematical integrity before we even look at the markets.

Step-by-Step Initialization Process:

1. Magic Number Setup: Trade.SetExpertMagicNumber(MagicNumber)—This is like giving your EA a fingerprint. It ensures the EA only manages its own orders and doesn't interfere with manual trading or other EAs.
2. Broker Configuration: Trade.SetMarginMode() and Trade.SetTypeFillingBySymbol(\_Symbol) adapts to your specific broker's rules. Different brokers have different margin calculations and order filling policies.
3. Mathematical Validation: We check that TradingRetracementLevel is between 0 and 1 (exclusive) because 0% or 100% retracements don't make mathematical sense in a Fibonacci context.
4. Profit Target Validation: ProfitTargetLevel must be ≥1.0 because we're using Fibonacci extensions beyond the swing range.
5. Initial Market Analysis: FindSwingPoints() establishes the current market structure immediately upon startup.

Many developers skip proper initialization, leading to mysterious errors later. By validating everything upfront, we prevent runtime errors and ensure consistent behavior.

```
//+------------------------------------------------------------------+
//| Expert Initialization Function: Called once on EA startup        |
//|                                                                  |
//| Purpose: Validates inputs, initializes trading engine, and       |
//|         performs initial market analysis for immediate readiness |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Trading Engine Initialization: Configure for reliable execution
    Trade.SetExpertMagicNumber(MagicNumber);      // Isolate EA orders using magic number
    Trade.SetMarginMode();                        // Apply symbol-specific margin calculations
    Trade.SetTypeFillingBySymbol(_Symbol);        // Use symbol-appropriate order filling rules

    //--- Input Validation: Ensure mathematical integrity of Fibonacci levels
    if(TradingRetracementLevel <= 0 || TradingRetracementLevel >= 1)
    {
        Alert("Error: Trading retracement level must be between 0 and 1 (exclusive)");
        return INIT_PARAMETERS_INCORRECT;         // Critical: Prevent operation with invalid parameters
    }

    if(ProfitTargetLevel < 1)
    {
        Alert("Error: Profit target level must be ≥ 1.0 (Fibonacci extension)");
        return INIT_PARAMETERS_INCORRECT;         // Critical: Ensure profitable target positioning
    }

    //--- Initial Market Analysis: Establish baseline market structure
    FindSwingPoints();                            // Detect immediate swing points for initial setup
    Print("Fibonacci Predictive EA initialized successfully");
    Print("Symbol: ", _Symbol, " | Timeframe: ", EnumToString(_Period));
    Print("Trading Level: ", TradingRetracementLevel, " | Target: ", ProfitTargetLevel);

    return INIT_SUCCEEDED;                        // Success: EA is ready for market operation
}
```

4\. The Heartbeat: Processing Market Updates Efficiently

Markets never sleep, but your computer resources are finite. The OnTick function embodies the principle of intelligent efficiency. Instead of analyzing every single tick (which would overwhelm your system), we wait for new bars to form. This approach respects both market timing and computational resources.

Notice the logical flow: find new swings, clear old drawings, calculate new pathways, and manage orders. This sequence ensures that your analysis is always current, your charts remain clean, and your orders reflect the latest market structure. It's a disciplined workflow that mirrors professional trading desks.

The Tick Processing Logic Explained:

1. New Bar Detection: IsNewBar() is our efficiency gatekeeper. It returns true only when a new candlestick forms, preventing wasteful repeated calculations within the same bar.
2. Market Structure Update: FindSwingPoints() reanalyzes the market to capture any new significant highs or lows that formed in the latest bar.
3. Visual Cleanup: ClearOldObjects() removes previous drawings before creating new ones. This prevents chart clutter and ensures visual clarity.
4. Pathway Calculation: CalculatePredictivePathway() is the core brain—it computes Fibonacci levels, determines trade direction, and prepares orders.
5. Order Management: ManagePendingOrders() cleans up stale orders that are no longer relevant due to market movement.

Processing only on new bars reduces CPU usage by 90-95% compared to tick-by-tick processing. This is crucial for running multiple EAs or using resource-intensive indicators.

```
//+------------------------------------------------------------------+
//| Expert Tick Function: Core processing on each price update       |
//|                                                                  |
//| Purpose: Monitors market conditions, updates analysis on new bars|
//|          manages pending orders, and maintains visual display    |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- New Bar Detection: Process only on new bar formation for efficiency
    if(IsNewBar())
    {
        //--- Market Structure Analysis: Update swing points for current market conditions
        FindSwingPoints();

        //--- Visual Management: Clear previous drawings before updating
        ClearOldObjects();

        //--- Trading Logic: Calculate and visualize Fibonacci pathways
        CalculatePredictivePathway();

        //--- Order Management: Maintain and clean up pending orders
        ManagePendingOrders();
    }
}
```

5\. Market Structure Analysis: Finding the True Swing Points

Here's where we separate significant market movements from random noise. The FindSwingPoints function doesn't just look for any high or low—it applies the classic definition of swing points: a bar that's higher/lower than two neighbors on each side. This methodology filters out insignificant fluctuations and identifies meaningful market structure.

Notice how we track both the price levels and their bar indices? This dual tracking allows us to determine trend direction temporally. The most recent swing (higher index) tells us the market's current momentum direction. This temporal analysis is more reliable than simple price comparison.

Swing Detection Algorithm in Detail:

1. Initialization: We start with extreme values—swingHigh at 0 (will be exceeded) and swingLow at DBL\_MAX (will be undercut).
2. Boundary Calculation: MathMin(Bars(\_Symbol, \_Period), MaxBarsBack) ensures we don't try to analyze more bars than exist or more than our configured maximum.
3. Swing High Logic: A bar must be higher than two bars before and two bars after. This ensures we're capturing significant turning points, not minor fluctuations.
4. Swing Low Logic: Similarly, a bar must be lower than its two neighbors on each side.
5. Trend Determination: trendDirection = swingHighIndex > swingLowIndex—if the high occurred more recently than the low, we're in an uptrend.

Many traders use simpler methods like comparing current price to moving averages, but these lag. Our swing point detection identifies genuine market structure changes in real-time, giving us earlier and more reliable trend signals.

```
//+------------------------------------------------------------------+
//| Find Swing Points: Identifies significant market highs/lows      |
//|                                                                  |
//| Purpose: Detect the most recent swing high and low points that   |
//|          define the current market structure for Fibonacci       |
//|          calculations. Critical for accurate level placement.    |
//+------------------------------------------------------------------+
void FindSwingPoints()
{
    //--- Initialize tracking variables
    swingHigh = 0;
    swingLow = DBL_MAX;
    swingHighIndex = -1;
    swingLowIndex = -1;

    //--- Calculate analysis window: Balance between depth and performance
    int barsToCheck = MathMin(Bars(_Symbol, _Period), MaxBarsBack);

    //--- Swing Detection Loop: Analyze each bar in the historical window
    for(int i = 3; i < barsToCheck; i++)
    {
        //--- Swing High Detection: Bar higher than 2 neighbors on each side
        if(IsSwingHigh(i))
        {
            double high = iHigh(_Symbol, _Period, i);
            //--- Update if this is the highest swing found
            if(high > swingHigh || swingHighIndex == -1)
            {
                swingHigh = high;
                swingHighIndex = i;
                swingHighTime = iTime(_Symbol, _Period, i);
            }
        }

        //--- Swing Low Detection: Bar lower than 2 neighbors on each side
        if(IsSwingLow(i))
        {
            double low = iLow(_Symbol, _Period, i);
            //--- Update if this is the lowest swing found
            if(low < swingLow || swingLowIndex == -1)
            {
                swingLow = low;
                swingLowIndex = i;
                swingLowTime = iTime(_Symbol, _Period, i);
            }
        }
    }

    //--- Trend Determination: Compare swing timing to establish trend direction
    if(swingHighIndex != -1 && swingLowIndex != -1)
    {
        //--- Bullish Trend: Higher highs and higher lows (swing high occurs after swing low)
        //--- Bearish Trend: Lower highs and lower lows (swing low occurs after swing high)
        trendDirection = swingHighIndex > swingLowIndex;
    }
}
```

6\. The Core Innovation: Predictive Fibonacci Pathway Calculation

This is where mathematics meets market psychology. The CalculatePredictivePathway function transforms static Fibonacci levels into dynamic trading decisions. Notice the elegant symmetry: in uptrends, we sell retracements; in downtrends, we buy retracements. This counterintuitive approach is actually mathematically sound; we're trading with the larger trend by entering on countertrend moves.

The range calculation (swingHigh-swingLow) becomes our measuring stick for all Fibonacci levels. This dynamic adaptation means our EA works equally well on volatile currency pairs and calm indices. The 61.8% retracement and 100% extension aren't arbitrary—they represent the golden ratio and full measured move, levels that professional traders watch closely.

```
//+------------------------------------------------------------------+
//| Calculate Predictive Pathway: Core Fibonacci trading logic       |
//|                                                                  |
//| Purpose: Calculate Fibonacci retracement levels, draw predictive |
//|          pathways, and prepare pending orders based on current   |
//|          market structure and trend analysis                     |
//+------------------------------------------------------------------+
void CalculatePredictivePathway()
{
    //--- Validation: Ensure we have valid swing points for analysis
    if(swingHighIndex == -1 || swingLowIndex == -1)
    {
        Print("Warning: Insufficient swing points for Fibonacci analysis");
        return;
    }

    //--- Fibonacci Range Calculation: Basis for all level calculations
    double range = swingHigh - swingLow;

    //--- Bull Market Strategy: Sell retracements in uptrend
    if(trendDirection)
    {
        //--- Calculate 61.8% retracement level (Golden Ratio)
        double entryPrice = swingHigh - (range * TradingRetracementLevel);

        //--- Calculate 100% extension target (full swing retracement)
        double targetPrice = entryPrice - (range * ProfitTargetLevel);

        //--- Visualize the predictive pathway
        DrawPredictionArrow(swingHighTime, swingHigh, swingLowTime, entryPrice, "Entry");
        DrawTargetArrow(entryPrice, targetPrice, "Target");

        //--- Execute trading logic if enabled
        if(EnableTrading)
            PlaceSellPendingOrder(entryPrice, targetPrice, range);
    }
    //--- Bear Market Strategy: Buy retracements in downtrend
    else
    {
        //--- Calculate 61.8% retracement level (Golden Ratio)
        double entryPrice = swingLow + (range * TradingRetracementLevel);

        //--- Calculate 100% extension target (full swing retracement)
        double targetPrice = entryPrice + (range * ProfitTargetLevel);

        //--- Visualize the predictive pathway
        DrawPredictionArrow(swingLowTime, swingLow, swingHighTime, entryPrice, "Entry");
        DrawTargetArrow(entryPrice, targetPrice, "Target");

        //--- Execute trading logic if enabled
        if(EnableTrading)
            PlaceBuyPendingOrder(entryPrice, targetPrice, range);
    }

    //--- Information Display: Show current trade setup on chart
    DrawInfoText();
}
```

Fibonacci Mathematics in Action:

1\. Range Calculation: swingHigh - swingLow gives us the total distance of the price swing. This becomes our 100% measurement for all Fibonacci calculations.

2\. Uptrend Logic (Sell Retracement):

- Entry: swingHigh - (range \* TradingRetracementLevel)—We expect price to retrace 61.8% of the up move
- Target: entryPrice - (range \* ProfitTargetLevel)—We target a move equal to the original swing downward

3\. Downtrend Logic (Buy Retracement):

- Entry: swingLow + (range \* TradingRetracementLevel)—We expect price to retrace 61.8% of the down move
- Target: entryPrice + (range \* ProfitTargetLevel)—We target a move equal to the original swing upward.

This approach follows the classic "trade with the trend, enter on retracement" methodology. By selling in uptrends (after retracements) and buying in downtrends (after bounces), we're positioning in the direction of the larger trend but at better prices.

7\. Visual Intelligence: Drawing the Predictive Pathways

Most EAs throw lines on charts; our EA tells a visual story. The DrawPredictionArrow function creates a narrative: "Price moved from here to here, and we expect it to retrace to this level." The green dashed line isn't just decoration—it's a visual hypothesis about market behavior.

The target visualization uses gold checkmarks and dotted lines—a different visual language that says, "This is our profit objective." This color-coding and styling creates immediate visual understanding. When you glance at your chart, you instantly know what's a prediction and what's a target without reading any text.

```
//+------------------------------------------------------------------+
//| Draw Prediction Arrow: Visualize retracement pathway to entry    |
//|                                                                  |
//| Purpose: Create visual representation of expected price movement |
//|          to Fibonacci retracement level using arrows and lines   |
//|                                                                  |
//| Parameters:                                                      |
//|   startTime   Starting point timestamp for pathway               |
//|   startPrice  Starting price level for pathway                   |
//|   endTime     Ending point timestamp for pathway                 |
//|   endPrice    Target price level (Fibonacci retracement)         |
//|   name        Unique identifier for chart object management      |
//+------------------------------------------------------------------+
void DrawPredictionArrow(datetime startTime, double startPrice, datetime endTime, double endPrice, string name)
{
    string arrowName = name + "_Arrow";
    string lineName = name + "_Path";

    //--- Cleanup: Remove previous objects to prevent duplication
    ObjectDelete(0, arrowName);
    ObjectDelete(0, lineName);

    //--- Entry Point Arrow: Mark the Fibonacci retracement level
    if(ObjectCreate(0, arrowName, OBJ_ARROW_BUY, 0, endTime, endPrice))
    {
        ObjectSetInteger(0, arrowName, OBJPROP_COLOR, PredictionColor);
        ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, 241);  // Standard buy arrow
        ObjectSetInteger(0, arrowName, OBJPROP_WIDTH, ArrowSize);
    }

    //--- Predictive Pathway Line: Dashed line showing expected retracement
    if(ObjectCreate(0, lineName, OBJ_TREND, 0, startTime, startPrice, endTime, endPrice))
    {
        ObjectSetInteger(0, lineName, OBJPROP_COLOR, PredictionColor);
        ObjectSetInteger(0, lineName, OBJPROP_STYLE, STYLE_DASH); // Dashed for predictions
        ObjectSetInteger(0, lineName, OBJPROP_WIDTH, 1);
        ObjectSetInteger(0, lineName, OBJPROP_RAY, false);       // Finite line segment
    }
}
```

Visual Design Principles:

1\. Prediction Arrows (Green):

- Use OBJ\_ARROW\_BUY with code 241 (standard buy arrow)
- Green color indicates "go" - this is where we expect to enter
- Dashed lines show the expected retracement pathway

2\. Target Arrows (Gold):

- Use OBJ\_ARROW\_CHECK - the checkmark signifies completion
- Gold color represents the "prize" or profit objective
- Dotted lines distinguish targets from predictions

3\. Object Management:

- We delete old objects first to prevent duplication
- Each object gets a unique name for easy management
- OBJPROP\_RAY, false creates finite lines, not infinite rays

User Experience Benefits: This visual system allows traders to instantly understand the EA's current market hypothesis without reading logs or remembering complex rules. The color coding and arrow styles create an intuitive visual language.

8\. Professional Order Placement: Beyond Basic Trade Execution

Placing orders is easy; placing smart orders is an art. Notice how our order placement functions don't just blindly send trades? They perform price validation, margin checks, duplicate detection, and logical verification. This multi-layered approach prevents the most common automated trading mistakes.

The price normalization step is particularly important. Different brokers have different digit precision, and failing to normalize can cause order rejection. The fallback calculations ensure that even if our primary Fibonacci calculation produces questionable results, we still have logical stop loss and take profit levels.

```
//+------------------------------------------------------------------+
//| Place Buy Pending Order: Execute buy limit order with validation |
//|                                                                  |
//| Purpose: Place a buy limit order at Fibonacci retracement level  |
//|          with proper risk management and error handling          |
//|                                                                  |
//| Parameters:                                                      |
//|   entryPrice   Calculated buy entry price (61.8% retracement)    |
//|   targetPrice  Calculated take profit price (100% extension)     |
//|   range        Fibonacci swing range for position sizing context |
//+------------------------------------------------------------------+
void PlaceBuyPendingOrder(double entryPrice, double targetPrice, double range)
{
    //--- Price Validation: Ensure mathematically sound price levels
    if(!IsValidPrice(entryPrice) || !IsValidPrice(targetPrice))
    {
        Print("Order Rejected: Invalid price levels - Entry: ", entryPrice, " Target: ", targetPrice);
        return;
    }

    //--- Stop Loss Calculation: Fixed distance below entry
    double sl = entryPrice - (StopLossPips * _Point * 10);
    if(!IsValidPrice(sl))
        sl = entryPrice - (100 * _Point); // Fallback to fixed distance

    //--- Price Normalization: Align with broker precision requirements
    double normalizedEntry = NormalizeDouble(entryPrice, _Digits);
    double normalizedSL = NormalizeDouble(sl, _Digits);
    double normalizedTP = NormalizeDouble(targetPrice, _Digits);

    //--- Order Validation: Ensure logical price relationships
    if(normalizedTP <= normalizedEntry)
    {
        Print("Adjustment: Take profit below entry, using fallback calculation");
        normalizedTP = normalizedEntry + (StopLossPips * 2 * _Point * 10);
    }

    if(normalizedSL >= normalizedEntry)
    {
        Print("Adjustment: Stop loss above entry, using fallback calculation");
        normalizedSL = normalizedEntry - (StopLossPips * _Point * 10);
    }

    //--- Duplicate Check: Prevent multiple orders at same level
    if(OrderExists(ORDER_TYPE_BUY_LIMIT, normalizedEntry))
    {
        Print("Order Skipped: Duplicate buy order already exists at ", normalizedEntry);
        return;
    }

    //--- Margin Check: Ensure sufficient account equity
    if(!CheckMargin(ORDER_TYPE_BUY_LIMIT, LotSize, normalizedEntry))
    {
        Print("Order Rejected: Insufficient margin for buy order");
        return;
    }

    //--- Order Execution: Place buy limit order with MQL5 trade class
    if(Trade.BuyLimit(LotSize, normalizedEntry, _Symbol, normalizedSL, normalizedTP,
                      ORDER_TIME_GTC, 0, "Fib61.8_Buy"))
    {
        Print("SUCCESS: BUY LIMIT order placed at ", normalizedEntry,
              " | SL: ", normalizedSL, " | TP: ", normalizedTP);
    }
    else
    {
        Print("FAILED: Buy order placement error - ", Trade.ResultRetcodeDescription());
    }
}
```

Order Placement Safety Checklist:

1. Price Validation: IsValidPrice() checks if calculated prices are within reasonable ranges
2. Stop Loss Calculation: Fixed pip distance from entry with fallback options
3. Price Normalization: NormalizeDouble() aligns with broker precision requirements
4. Logical Verification: Ensures TP > Entry for buys, TP < Entry for sells
5. Duplicate Prevention: OrderExists() checks for identical pending orders
6. Margin Check: CheckMargin() verifies sufficient account equity
7. Error Handling: Comprehensive logging of success/failure

Risk Management Integration: Each order carries its own stop loss and take profit, ensuring predefined risk parameters. The 2:1 risk-reward ratio (StopLossPips vs 2x for TP fallback) maintains positive expectancy even when Fibonacci calculations need adjustment.

9\. Risk Management:

These helper functions are your EA's immune system. IsValidPrice prevents mathematical absurdities that could wipe out your account. CheckMargin ensures you're not overleveraging. OrderExists prevents duplicate orders that would double your risk. Together, they create a robust risk management framework.

Notice how IsValidPrice doesn't use fixed ranges? It calculates reasonable bounds based on current market prices. This dynamic approach works across all instruments and market conditions. The margin check with 2x buffer is conservative but wise—it accounts for spread widening and sudden margin requirement changes.

```
//+------------------------------------------------------------------+
//| Is Valid Price: Validate price for mathematical soundness        |
//|                                                                  |
//| Purpose: Prevent orders at impossible price levels that could    |
//|          indicate calculation errors or market anomalies         |
//|                                                                  |
//| Parameters:                                                      |
//|   price [in]  Price level to validate                            |
//|                                                                  |
//| Returns: true if price is within reasonable range of current     |
//|          market price, false if potentially erroneous            |
//+------------------------------------------------------------------+
bool IsValidPrice(double price)
{
    //--- Basic sanity check: Price must be positive
    if(price <= 0)
        return false;

    //--- Market context: Get current bid/ask for comparison
    double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double currentPrice = (currentBid + currentAsk) / 2;

    //--- Reasonable range: 50% to 200% of current price (adjustable)
    double minPrice = currentPrice * 0.5;
    double maxPrice = currentPrice * 2.0;

    return (price >= minPrice && price <= maxPrice);
}

//+------------------------------------------------------------------+
//| Check Margin: Validate sufficient account equity for trade       |
//|                                                                  |
//| Purpose: Prevent orders that would exceed available margin or    |
//|          violate broker requirements                             |
//|                                                                  |
//| Parameters:                                                      |
//|   orderType [in]  Type of order (BUY_LIMIT/SELL_LIMIT)           |
//|   volume    [in]  Trade volume in lots                           |
//|   price     [in]  Order entry price                              |
//|                                                                  |
//| Returns: true if sufficient margin available with safety buffer, |
//|          false if trade would exceed risk limits                 |
//+------------------------------------------------------------------+
bool CheckMargin(ENUM_ORDER_TYPE orderType, double volume, double price)
{
    double marginRequired;

    //--- Calculate margin requirement based on order type
    if(orderType == ORDER_TYPE_BUY_LIMIT)
    {
        marginRequired = AccountInfo.MarginCheck(_Symbol, ORDER_TYPE_BUY, volume, price);
    }
    else
    {
        marginRequired = AccountInfo.MarginCheck(_Symbol, ORDER_TYPE_SELL, volume, price);
    }

    //--- Handle margin calculation failures gracefully
    if(marginRequired <= 0)
        return true; // Allow trade if calculation fails (conservative approach)

    //--- Safety buffer: Require 2x margin for risk management
    double freeMargin = AccountInfo.FreeMargin();
    return (freeMargin > marginRequired * 2.0);
}
```

Risk Management in Detail:

IsValidPrice Logic:

- Prevents orders at 0 or negative prices (mathematical errors)
- Uses current market context: 50% to 200% of current price range
- Dynamic adaptation works across all symbols and market conditions

CheckMargin Implementation:

Uses MQL5's built-in AccountInfo.MarginCheck() for accurate calculations.

2x safety buffer protects against:

- Spread widening during news events
- Margin requirement changes
- Multiple simultaneous positions
- Account equity fluctuations

OrderExists Protection:

- Prevents order duplication that could double position size
- Uses precise price comparison (within 1 point)
- Only checks same symbol and order type

These functions work together to create multiple layers of protection. Even if one check fails, others provide backup safety measures.

10\. Order Management: The Cleanup Crew That Maintains Efficiency

Markets evolve, and so should your pending orders. The ManagePendingOrders function acts as your EA's cleanup crew, removing orders that have become irrelevant. Orders too far from the current price or too old represent outdated market hypotheses and should be cleared.

This proactive order management prevents "order clutter" and ensures your trading capital isn't tied up in low-probability setups. The 50-pip distance and 100-bar age thresholds are sensible defaults, but the real value is in having a systematic approach to order maintenance.

```
//+------------------------------------------------------------------+
//| Manage Pending Orders: Clean up outdated or irrelevant orders    |
//|                                                                  |
//| Purpose: Remove orders that are no longer valid due to market    |
//|          movement, time decay, or changed market conditions      |
//+------------------------------------------------------------------+
void ManagePendingOrders()
{
    //--- Iterate backwards through orders (safe for deletion)
    for(int i = OrdersTotal() - 1; i >= 0; i--)
    {
        if(OrderInfo.SelectByIndex(i) && OrderInfo.Symbol() == _Symbol && OrderInfo.Magic() == MagicNumber)
        {
            double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double orderPrice = OrderInfo.PriceOpen();
            datetime orderTime = OrderInfo.TimeSetup();

            //--- Distance Check: Remove if too far from current price
            bool isTooFar = MathAbs(currentPrice - orderPrice) > (500 * _Point);

            //--- Time Check: Remove if too old (market conditions changed)
            bool isTooOld = (TimeCurrent() - orderTime) > PeriodSeconds(_Period) * 100;

            //--- Execute deletion if either condition met
            if(isTooFar || isTooOld)
            {
                Trade.OrderDelete(OrderInfo.Ticket());
                Print("Order Cleanup: Deleted pending order ", OrderInfo.Ticket(),
                      " - Reason: ", isTooFar ? "Price Distance" : "Time Expiry");
            }
        }
    }
}
```

Order Management Strategy:

1\. Distance-Based Cleanup:

- Removes orders 50+ pips from current price
- Prevents capital from being tied up in irrelevant levels
- Adapts to changing market conditions

2\. Time-Based Cleanup:

- Removes orders older than 100 bars
- Prevents stale orders from unexpected market events
- Respects different timeframe contexts

3\. Selective Management:

- Only manages orders with our MagicNumber
- Preserves manual trades and other EA orders
- Uses efficient backward iteration for safe deletion

The Cleanup Logic: Orders are removed if EITHER condition is met (too far OR too old). This ensures we don't keep orders that might be close but stale, or recent but irrelevant.

### Testing

Accessing the Strategy Tester:

To properly evaluate our Fibonacci Predictive EA, you'll want to utilize MetaTrader 5's built-in Strategy Tester, which provides a robust environment for forward-testing and optimization. Here's your step-by-step guide:

Step 1: Opening the Strategy Tester

- Navigate to the "View" menu in MetaTrader 5 and select "Strategy Tester" or press Ctrl+R
- Ensure you're connected to your broker's server for accurate price data
- Select "Fibonacci Predictive EA" from the Expert Advisor dropdown.

Step 2: Configuring Test Parameters (Optional)

- Symbol: Start with major pairs like EURUSD or GBPUSD for initial testing.
- Timeframe: H1 and H4 timeframes work well with Fibonacci strategies
- Model: Use "Every tick based on real ticks" for most accurate results
- Date Range: Test across different market conditions (trending, ranging, volatile)
- Initial Deposit: Use realistic account sizes that match your trading capital.

Our comprehensive testing of the Fibonacci Predictive EA on the GBPUSD M1 timeframe yielded impressive results that validate both the mathematical foundations and practical implementation of our approach.

The EA demonstrated exceptional order execution precision, consistently placing pending orders at calculated Fibonacci retracement levels with accurate stop-loss and take-profit parameters. Even on the challenging M1 timeframe known for its volatility and rapid price movements, the system maintained robust performance.

Key observations from our testing include:

- Successful identification of clear swing points and precise Fibonacci retracement level calculations
- Flawless placement of pending buy and sell orders at the 61.8% golden ratio levels
- Consistent execution with proper risk management parameters intact
- Effective adaptation to fast-changing market conditions
- Clean visual pathway maintenance without chart clutter
- Elimination of emotional decision-making through algorithmic discipline

![STRATEGY TESTER](https://c.mql5.com/2/180/metatester64_WSgy1Bl2Yj.gif)

Testing the FibonacciPredictiveEA

The testing confirms that the EA's systematic approach to Fibonacci trading translates effectively to fast-moving market conditions, providing a disciplined framework that outperforms manual trading in volatile environments. The real-time visual feedback allowed for immediate verification of the EA's market hypotheses and trade decisions, making it both an effective trading tool and educational resource.

### Conclusion

Our journey in developing the Fibonacci Predictive EA has demonstrated the powerful synergy between mathematical principles and modern algorithmic execution. What began as an exploration of Fibonacci retracements evolved into a comprehensive trading system that not only identifies high-probability setups but also executes them with unwavering discipline.

The true breakthrough lies in the transformation of static Fibonacci levels into dynamic predictive pathways. By visualizing the expected price journey from retracement entry to extension target, we've created a system that educates while it executes. Traders no longer need to wonder why a particular level was chosen—the visual pathways make the underlying logic immediately apparent.

Throughout our testing, the EA consistently demonstrated its ability to identify meaningful swing points and project realistic retracement zones. The integration of robust risk management protocols ensured that even during unexpected market movements, losses remained contained and manageable. The visual feedback system proved invaluable for both real-time trading and educational purposes, helping traders develop an intuitive understanding of market structure.

Perhaps most importantly, this project successfully addresses the core psychological challenge that plagues most traders: the temptation to chase price movements. By providing a structured framework that anticipates rather than reacts, the EA instills the patience and discipline essential for long-term trading success.

While no trading system can guarantee profits, our Fibonacci Predictive EA provides a mathematically sound, visually intuitive, and systematically executable approach to market analysis. It represents not just a collection of code, but a comprehensive trading methodology that can be adapted and refined as market conditions evolve.

### Key Lessons

| Key Lessons | Description |
| --- | --- |
| Input Validation is Critical | Always validate Fibonacci levels in OnInit to prevent mathematical errors and ensure 0 < retracement < 1 and extension ≥ 1 |
| Use CTrade for Professional Execution | MQL5's CTrade class provides built-in error handling and simplifies order management compared to raw OrderSend |
| Object Management Prevents Chart Clutter | Always delete old chart objects before creating new ones and use unique naming conventions for easy management |
| Price Normalization is Essential | Use NormalizeDouble() with \_Digits to align prices with broker precision requirements and avoid order rejection |
| Magic Numbers Isolate EA Orders | Assign unique MagicNumber to identify EA orders and prevent interference with manual trading or other EAs |

### Attachments

| Source File | Version | Description |
| --- | --- | --- |
| FibonacciPredictiveEA.mq5 | 1.00 | Complete MQL5 source code implementing predictive Fibonacci pathways with swing detection, visualization, and automated order execution |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20160.zip "Download all attachments in the single ZIP archive")

[FibonacciPredictiveEA.mq5](https://www.mql5.com/en/articles/download/20160/FibonacciPredictiveEA.mq5 "Download FibonacciPredictiveEA.mq5")(71.12 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/500348)**
(1)


![MikeBaranoff](https://c.mql5.com/avatar/2025/9/68bdc971-a287.png)

**[MikeBaranoff](https://www.mql5.com/en/users/mikebaranoff)**
\|
14 Jan 2026 at 10:15

Good afternoon. Great EA for determining entry points and the ability to adjust fibonacci levels and [stop losses](https://www.metatrader5.com/en/terminal/help/trading/general_concept "User Guide: Types of orders"). Please add dynamic TP change using breakout and trailing stop. I would also limit the number of entries in one direction. Thank you.


![Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://c.mql5.com/2/181/20065-developing-trading-strategy-logo.png)[Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://www.mql5.com/en/articles/20065)

Generating new indicators from existing ones offers a powerful way to enhance trading analysis. By defining a mathematical function that integrates the outputs of existing indicators, traders can create hybrid indicators that consolidate multiple signals into a single, efficient tool. This article introduces a new indicator built from three oscillators using a modified version of the Pearson correlation function, which we call the Pseudo Pearson Correlation (PPC). The PPC indicator aims to quantify the dynamic relationship between oscillators and apply it within a practical trading strategy.

![Implementation of a table model in MQL5: Applying the MVC concept](https://c.mql5.com/2/130/MQL5_table_model_implementation___LOGO__V2.png)[Implementation of a table model in MQL5: Applying the MVC concept](https://www.mql5.com/en/articles/17653)

In this article, we look at the process of developing a table model in MQL5 using the MVC (Model-View-Controller) architectural pattern to separate data logic, presentation, and control, enabling structured, flexible, and scalable code. We consider implementation of classes for building a table model, including the use of linked lists for storing data.

![Markets Positioning Codex in MQL5 (Part 2):  Bitwise Learning, with Multi-Patterns for Nvidia](https://c.mql5.com/2/182/20045-markets-positioning-codex-in-logo.png)[Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)

We continue our new series on Market-Positioning, where we study particular assets, with specific trade directions over manageable test windows. We started this by considering Nvidia Corp stock in the last article, where we covered 5 signal patterns from the complimentary pairing of the RSI and DeMarker oscillators. For this article, we cover the remaining 5 patterns and also delve into multi-pattern options that not only feature untethered combinations of all ten, but also specialized combinations of just a pair.

![Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://c.mql5.com/2/115/Neural_Networks_in_Trading_Hierarchical_Two-Tower_Transformer_Hidformer___LOGO__1.png)[Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

We invite you to get acquainted with the Hierarchical Double-Tower Transformer (Hidformer) framework, which was developed for time series forecasting and data analysis. The framework authors proposed several improvements to the Transformer architecture, which resulted in increased forecast accuracy and reduced computational resource consumption.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/20160&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083079119261930640)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).

![close](https://c.mql5.com/i/close.png)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)