---
title: Developing a Trading Strategy: The Triple Sine Mean Reversion Method
url: https://www.mql5.com/en/articles/20220
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:24:54.954047
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/20220&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049115398353298741)

MetaTrader 5 / Trading


### Introduction

Traditional technical indicators such as the **Moving Average Convergence Divergence** ( **MACD)**, **Money Flow Index (MFI)**, **Stochastic Oscillator**, and **Awesome Oscillator (AO)** have long served as the foundation for traders and analysts seeking to identify market entry and exit points. While diverse in their construction each of these indicators is built upon distinct mathematical formulations that transform price data into meaningful graphical representations, helping traders interpret market behavior and momentum.

In this article, we explore the development of a novel trading strategy that steps beyond these conventional tools. We introduce a **mean-reversion strategy** whose entry criteria are based on a custom-built indicator: the **Triple Sine Oscillator (TSO)**. To achieve this, we employ a specific trigonometric function—the **Sine Cube**—to engineer an oscillator that uses scaled price data as its primary input. The goal is to harness the cyclical nature of trigonometric functions to identify potential mean reversion points in price movements.

### Mathematical Foundation: The Sine Cube Function

At the core of the Triple Sine Indicator lies the _sine cube function_, mathematically expressed as:

![sinecubefxn](https://c.mql5.com/2/180/sinecubefxn.png)

This function, being a higher-order transformation of the basic sine wave, preserves its oscillatory behavior but with more pronounced curvature around the zero line. Like the standard sine function, it oscillates between _-1 and +1_, but with smoother transitions near the peaks and sharper inflection points around zero. These characteristics make it ideal for designing a mean reversion-based signal, as it captures subtle shifts in momentum and a candidate for constructing an overbought/oversold trading signal.

The _sine cube function_ thus serves as a powerful mechanism for smoothing out price fluctuations while retaining sensitivity to directional changes. When applied to scaled price inputs, it produces an indicator that highlights overbought and oversold regions — conditions that are essential for mean reversion strategies.

**Visualizing the Sine Wave Cycle**

At this point, it is important to _visualize the nature of the sine cube function_ to gain a clearer understanding of its behavior and how it can be applied within our trading framework. To achieve this, we plot the function over the range _–2π to 2π._

![SCplot](https://c.mql5.com/2/179/SineCubePlot2.png)

Figure 1: Sine Cube Plot

Figure 1 shows the structure of the sine cube function. The resulting waveform reveals the cyclical and symmetric structure of the sine cube function. Within this interval, the function completes a _full oscillatory cycle_, displaying _two peaks_ and _two troughs_. Each half of the cycle exhibits one positive and one negative extreme — in other words, the negative inputs (-2π  to  0) produce alternating +1 and –1 outputs, and the same pattern is mirrored for the positive inputs (0 to 2π).

This periodic symmetry highlights the balance between bullish and bearish momentum swings, a property that is essential for mean reversion analysis. By observing how the function transitions smoothly between these peaks and troughs, traders can interpret the turning points as potential zones of price exhaustion or reversal, providing a foundation for constructing reliable entry and exit rules in the Triple Sine Mean Reversion strategy.

**Exploring the MQL5 Code Structure for Plotting the Sine Cube**

Before developing the _Triple Sine Oscillator_ indicator, it is essential to understand the foundational MQL5 script structure used to plot the Sine Cube function. This process provides insight into how graphical plots are constructed and displayed within the MetaTrader 5 environment.

Importing the Graphics Library

The script begins by importing the graphics library, which provides the necessary tools for visual rendering:

```
#include <Graphics/Graphic.mqh>

string g_name = "SineCubePlot";
```

Here, the Graphic.mqh file is included to access graphical functions and classes required for plotting. The variable g\_name is defined as "SineCubePlot" , serving as the name of the graphical window that will display the sine cube plot.

Creating the Graphic Object and Window

The next step involves creating an instance of the CGraphic class and initializing the plotting window:

```
   // Create graphic object
   CGraphic graphic;

   // Create graphic window
   if(!graphic.Create(0, g_name, 0, 30, 30, 1000, 600))
   {
      Print("Error creating graphic!");
      return;
   }
```

In this section, a graphic object named graphic is declared. The Create() method is then called to initialize a graphical window with specific parameters such as position and size (30, 30, 1000, 600). If the window creation fails, an error message — “Error creating graphic!” — is printed, and the program terminates gracefully.

Setting Up the Chart Properties for the Sine Cube Plot

It is important to define the appearance and layout of the chart. The function SetupChart() handles this task by configuring the background color, axis limits, labels, font sizes, and grid properties to ensure the final plot is clear and visually appealing.

```
//+------------------------------------------------------------------+
//| Set up the chart properties                                      |
//+------------------------------------------------------------------+
void SetupChart(CGraphic &graphic)
{
   // Set background color
   graphic.BackgroundColor(clrWhite);

   // Configure X axis
   graphic.XAxis().Min(-2*M_PI);
   graphic.XAxis().Max(2*M_PI);
   graphic.XAxis().Name("x");

   // Configure Y axis
   graphic.YAxis().Min(-1.2);
   graphic.YAxis().Max(1.2);
   graphic.YAxis().Name("sin(x)³");

   // Add grid
   graphic.XAxis().Color(clrGray);
   graphic.YAxis().Color(clrGray);

   // Increase font sizes
   graphic.XAxis().ValuesSize(14);
   graphic.YAxis().ValuesSize(14);
   graphic.XAxis().NameSize(20);
   graphic.YAxis().NameSize(20);

   // Add title
   graphic.CurvePlotAll();
   graphic.Update();
}
```

The SetupChart() function initializes the chart environment before plotting begins.

- The background color is set to white for a clean, professional appearance.
- The X-axis ranges from  −2π to +2π, representing two complete sine wave cycles, while the Y-axis spans from  −1.2 to +1.2, slightly beyond the function’s natural amplitude to give space for visual clarity.
- Axis labels are assigned: "x" for the horizontal axis and "sin(x)³" for the vertical axis.
- Grid lines are added in gray to improve readability without overpowering the plotted curve.
- Finally, font sizes for both axis names and numerical values are increased to enhance visibility.

Once all these parameters are configured, the chart is refreshed using _graphic.Update()_ to apply the new settings.

**Generating and Plotting the Sine Cube Function**

After preparing the chart layout, the next step is to generate the data points for the Sine Cube function and display them using the PlotSinCube() function.

```
//+------------------------------------------------------------------+
//| Generate and plot sin(x)^3 data                                  |
//+------------------------------------------------------------------+
void PlotSinCube(CGraphic &graphic)
{
   int points = 1000;
   double x[], y[];
   ArrayResize(x, points);
   ArrayResize(y, points);

   // Generate data points
   for(int i = 0; i < points; i++)
   {
      x[i] = -2*M_PI + (4*M_PI)*i/(points-1);
      y[i] = MathPow(MathSin(x[i]), 3);
   }

   // Get the curve object and set line properties
   CCurve* curve = graphic.CurveAdd(x, y, CURVE_LINES, "sin(x)³");
   if(curve != NULL)
   {
      curve.LinesStyle(STYLE_SOLID);
      curve.LinesWidth(3);
      curve.Color(ColorToARGB(clrBlue, 255));
   }
   graphic.CurvePlotAll();
}
```

This function builds the Sine Cube by:

- Defining 1,000 evenly spaced data points within the −2π to +2π interval.
- Computing each y value as sine cube and storing it alongside its corresponding x value.
- Using the CurveAdd() method to plot the resulting waveform with a solid blue line that has a width of three pixels for better visibility.
- Updating the chart with CurvePlotAll() to render the final curve.

Putting It All Together

The combination of SetupChart() and PlotSinCube() allows the MQL5 script to both configure and visualize mathematical functions efficiently. By separating chart configuration from data generation, the script maintains clarity and modularity.

### Triple Sine Oscillator (TSO) Indicator

Having examined the behavior of the sine cube function, we now proceed to construct the _Triple Sine Oscillator (TSO)_— a technical indicator that applies the mathematical principles of the sine cube function to market price data.

The TSO aims to translate raw price movements into a bounded oscillatory signal that fluctuates between –1 and +1, making it ideal for identifying mean reversion opportunities. However, since the sine cube function operates effectively within the input range –6.284 to +6.284 (approximately equivalent to –2π to +2π for one complete cycle), the raw price data must be scaled appropriately before being applied to the function.

To achieve this, we transform the closing price into a normalized input variable as follows:

![ScaledPriceInput](https://c.mql5.com/2/180/ScaledPrice.png)

Where:

- MA = Moving Average of the price (represents the central tendency)
- Std= Standard deviation of price (measures volatility)
- k= Sensitivity factor (a scalar constant that controls the degree of oscillation)

This transformation ensures that price movements are represented in a standardized, dimensionless form, compatible with the oscillatory range of the sine cube function. This calculation is powerful because it measures how many standard deviations the current price is from its moving average. A value of +2, for instance, indicates the price is two standard deviations above the mean—a potential overbought condition.

The sensitivity factor (k) allows traders to adjust the responsiveness of the indicator — a higher _k_ value increases sensitivity to short-term price fluctuations, while a lower _k_ value smooths out noise for broader trend detection.

By applying the sine cube function to this scaled price input, we derive the Triple Sine Oscillator — TSO = sin³( Scaled\_Price )— a bounded and cyclic indicator capable of detecting overbought and oversold zones, and signaling potential price reversion points in the market.

**TSO Indicator Code Structure**

Before demonstrating how the _Triple Sine Oscillator (TSO)_ performs in real-time trading scenarios, it is essential to first understand the code structure that governs its operation. The TSO indicator is built on a simple yet powerful framework designed to transform market price data into oscillatory signals using the sine cube function. The structure can be summarized in the following key stages:

Initialization Section

The Initialization Section defines the foundational structure and visual layout of the TSO indicator. This includes the indicator’s name, display mode, color scheme, and predefined levels that guide signal interpretation.

```
#property description "Triple-Sine Oscillator — (sin(x))^3"
#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1

//--- plot settings
#property indicator_label1  "Triple-Sine Oscillator"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrYellowGreen
#property indicator_width1  2

//--- indicator levels
#property indicator_level1   0.7
#property indicator_level2  -0.7
#property indicator_level3   0
#property indicator_levelcolor clrGray
```

In this configuration:

- The indicator is labeled “Triple-Sine Oscillator — (sin(x))³” to reflect its mathematical foundation.
- It is displayed in a separate indicator window, making it independent of the main price chart.
- Only one plot bufferis used, which stores and renders the computed TSO values.
- The plotted line appears in YellowGreen with a width of two pixels for clear visibility.
- Three reference levels are added at +0.7 , –0.7 , and 0 , colored gray. These levels visually identify overbought, oversold, and neutral (mean reversion) zones.

Input Parameters

The indicator accepts two key inputs that determine its responsiveness and smoothing behavior:

```
//--- input parameters
input int mPeriod = 20;      // Period for SMA and StdDev
input double Ksense = 0.5;   // Sensitivity multiplier
```

- MA Period (mPeriod)— defines the lookback window for calculating the Simple Moving Average (SMA) and Standard Deviation (StdDev). The default period is 20 bars.
- Sensitivity Factor (Ksense) — controls how strongly the indicator reacts to price deviations. A higher value makes it more responsive, while a lower value smooths the output. By default, k = 0.5.


These parameters allow traders to adjust the indicator based on market volatility and preferred signal frequency.

Defining the Triple Sine Function

At the core of the TSO lies the Triple Sine function, which mathematically transforms normalized price deviations into a smooth oscillatory waveform.

```
//+------------------------------------------------------------------+
//| TripleSine function                                              |
//+------------------------------------------------------------------+
double TripleSine(double x)
  {
   return(MathPow(MathSin(x), 3));
  }
```

Core Calculation Logic

Within the main calculation loop, the indicator computes the scaled price for each bar and transforms it using the Triple Sine function.

```
   int limit =rates_total- MathMax(mPeriod, prev_calculated - 1);

   for(int i = limit; i>=0; i--)
     {
      if(sd[i] == 0) { TripleSineBuffer[i] = 0; continue; }

      double scaled = Ksense * (price[i] - ma[i]) / sd[i];
      TripleSineBuffer[i] = TripleSine(scaled);
     }
```

Here, the moving average and standard deviation are dynamically calculated over the specified period to normalize price action.

The scaled price is then passed into the sine cube function. This transforms the normalized data into a bounded oscillatory signal ranging between –1 and +1.

Finally, the computed TSO values are stored in the TripleSineBuffer and plotted as a continuous line. This graphical output represents the cyclical behavior of market momentum — highlighting overbought, oversold, and mean reversion zones.

TSO indicator Demonstration

At this juncture, it is time to test the newly developed TSO on theMetaTrader 5 platform. This stage allows us to validate the indicator’s performance, visual behavior, and responsiveness to real market data.

![TSOdemo](https://c.mql5.com/2/179/TSO_demo.gif)

Figure 2: Triple Sine Oscillator

In the demonstration of theTSO, a key observation was made regarding the effect of the sensitivity factor (k) on the indicator’s responsiveness to price movement.

When the sensitivity value was increased from 0.5 to 2.3, the oscillator became more reactive to market fluctuations. This heightened sensitivity caused the TSO line to respond sharply to even minor price changes, making it ideal for short-term or highly volatile trading environments where quick signals are essential.

Conversely, when the sensitivity value was reduced to 0.1, the TSO became less responsive to price action. The oscillator appeared smoother, filtering out small market noise and emphasizing broader trends instead. This setting is often preferred in longer-term analysis, where traders aim to capture major market cycles rather than frequent reversals.

By optimizing both the sensitivity factor (k) and the moving average period (mPeriod), the TSO can be fine-tuned to adapt effectively across various financial instruments — from fast-moving currency pairs and commodities to slower, trend-oriented equities.

This flexibility makes the TSO a versatile analytical tool, capable of aligning its behavior with the volatility profile and trading characteristics of different markets.

### TSO Trading Strategy: The Mean Reversion Method

Now that the TSO indicator is fully constructed, we can build upon it to develop a complete Mean Reversion Trading Strategy. Before delving into the strategy itself, it is crucial to understand the underlying concept of mean reversion in financial markets.

Mean reversion is based on the idea that prices and other financial variables tend to move back toward their historical average (mean) over time. In simpler terms, when an asset’s price deviates significantly from its long-term average — either rising too high or falling too low — it is likely to “revert” **or return** toward that average eventually.

This concept assumes that markets exhibit a natural equilibrium, and that extreme price movements are temporary deviations caused by short-term imbalances in buying and selling pressure. Once these pressures subside, prices often normalize, moving closer to their mean.

However, it’s important to note that mean reversion strategies typically perform poorly in strongly trending markets. In such conditions, prices may continue to move away from the average for extended periods, leading to premature or false reversal signals. Therefore, the effectiveness of a mean reversion strategy depends heavily on identifying range-bound or oscillatory market conditions.

In the following section, we will integrate the TSO with key market conditions to formulate a complete mean reversion trading framework, defining the entry and exit rules that guide profitable decision-making.

**Trading Logic and Conditions**

1\. Sell Condition

A sell (short) position is initiated when the oscillator indicates that price momentum has moved excessively above its equilibrium zone. This occurs when:

Previous TSO< 0.7 and Current TSO> 0.7

This condition signals a potential overbought market, suggesting that prices may soon revert downward toward the mean.

2\. Buy Condition

A buy (long) position is triggered when the oscillator identifies a downward deviation from the mean — a potential oversold condition. The criteria are:

Previous TSO > − 0.7 and Current TSO < − 0.7

This crossover suggests that selling pressure may be fading, increasing the likelihood of an upward correction.

3\. Exit Condition

All open positions will be closed when either the Take Profit (TP) or Stop Loss (SL) thresholds are reached. This ensures that profits are secured and downside risks are managed effectively.

4\. Multiple Entry Rule

The strategy supports multiple entries in the direction of an active trade to capitalize on sustained mean reversion moves:

- When a buyposition is active, no sellpositions are opened.
- Additional buy trades may be added only if the buy condition is met again while the initial trade remains open.
- The same rule applies to sell positions — new sells are permitted only if existing ones are active and the sell condition reoccurs.

To prevent overtrading and ensure sufficient market movement between entries, the system enforces a 10-bar interval before checking for a new entry signal.

Furthermore, the maximum number of entries per direction is capped at five (5). This constraint balances profit potential with controlled risk exposure, avoiding excessive position stacking during volatile periods.

**Code Structure of TSO Expert Advisor**

The first operation within the OnTick() function is to determine whether a new bar (candle) has formed. This is done using the custom function IsNewBar() .

This conditional statement ensures that all computations — including indicator updates and signal checks — occur only once per completed bar, rather than on every incoming tick. This approach significantly reduces computational load and prevents repeated trade triggers within the same candle.

```
    if(!IsNewBar()) return;

    // Calculate TSO values
    CalculateTSO();

    // Check entry conditions
    CheckEntryConditions();
```

Once a new bar is confirmed, the EA calls the CalculateTSO() function.

This function computes the most recent Triple Sine Oscillator (TSO) values based on the current price data. It performs the following operations internally:

- Updates the moving average (MA) and standard deviation (Std) of the price.
- Scales the price using the sensitivity factor ( _k_).
- Applies the sine cube function to generate the TSO value.

The computed oscillator values (previous and current) are then stored in buffers for use in the signal evaluation phase.

After calculating the updated TSO readings, the EA proceeds to evaluate the trading logic using the CheckEntryConditions() function.

The _CheckEntryConditions()_ function forms the decision-making core of the Triple Sine Mean Reversion Expert Advisor, responsible for validating trading signals and managing position entries based on the TSO indicator values. It integrates position tracking, trade direction control, and signal verification to ensure that each trade adheres to the system’s logic and risk parameters.

```
//+------------------------------------------------------------------+
//| Check entry conditions                                           |
//+------------------------------------------------------------------+
void CheckEntryConditions()
{
    // Count current positions
    CountPositions();

    // Check if we can open new positions
    if(buyCount + sellCount >= MaxEntries)
    {
        Print("Maximum entries reached: ", MaxEntries);
        return;
    }

    // Check Buy condition: PrevTSO > -0.7 && CurTSO < -0.7
    if(prevTSO > -TSO_Threshold && curTSO < -TSO_Threshold)
    {
        Print("Buy condition met");
        if(CanOpenBuy())
        {
            OpenBuyPosition();
        }
    }

    // Check Sell condition: PrevTSO < 0.7 && CurTSO > 0.7
    if(prevTSO < TSO_Threshold && curTSO > TSO_Threshold)
    {
        Print("Sell condition met");
        if(CanOpenSell())
        {
            OpenSellPosition();
        }
    }
}
```

Below is the operational breakdown of the function:

At the start of the function, the EA counts the number of active buy and sell positions currently open in the market. The information is stored in buyCount and sellCount variables, which are later used to enforce position limits and directional consistency.

Before any new trades can be opened, the EA ensures that the total number of positions does not exceed the predefined maximum.

This condition prevents overexposure by limiting the total number of open positions to five (5), as defined in the strategy. Once this threshold is reached, the EA halts any further entries until one or more positions are closed.

The system then evaluates the buy signal condition, which occurs when the previous TSO value is greater than –0.7 and the current TSO value crosses below –0.7.

If this condition is met, it signals a potential oversold state in the market.

Before executing a new buy order, the EA calls the helper function CanOpenBuy(), which ensures:

- No existing sell positions are open (to maintain one-sided trading).
- The minimum bar interval(10 bars) since the last entry has elapsed.

Once these conditions are verified, the EA executes a new buy order through OpenBuyPosition().

Similarly, the sell condition is checked when the previous TSO value is less than +0.7 and the current TSO value crosses above +0.7.

This indicates an overbought condition, suggesting that the price may revert downward. The CanOpenSell() function ensures that no buy positions are active and that the minimum bar spacing requirement is satisfied before opening a new sell trade.

#### **Trade Filtering and Control**

By combining CountPositions() , CanOpenBuy() , and CanOpenSell() , the system enforces the following safeguards:

- Directional Consistency:Only one trade direction (buy or sell) is active at any time.
- Position Control:No more than five positions can be open simultaneously.
- Timing Discipline: A minimum of 10 bars must elapse before a new position is considered.

**TSO Expert Advisor Demonstration**

Having walked you through the code structure and the internal logic of the EA, we now demonstrate how the system executes trades in a live market environment.

When deployed, the EA continuously monitors market data in real time and evaluating entry conditions at the opening of each new bar. Once a valid signal is detected — based on the crossover behavior of the TSO — the EA automatically triggers buy or sell orders according to the rules defined earlier.

![TSO EA Demo](https://c.mql5.com/2/180/TSO_EA_demo3__1.gif)

Figure 3: TSO EA demonstration

Figure 3 illustrates how the TSO EA executes these orders in an actual market scenario. Each buy and sell execution corresponds precisely with the oscillator’s threshold crossovers.

In addition, multiple entries in the same direction can be observed, reflecting the EA’s ability to add to existing positions when the same condition reoccurs — provided the 10-bar interval and maximum entry limit are respected.

Through this automated execution process, the TSO EA transforms mathematical signal detection into precise market actions, ensuring consistency, speed, and discipline in trade management.

### Conclusion

In this article, we have successfully developed a new trading approach by integrating mathematical modeling with technical analysis. Beginning with the sine cube function, we constructed the Triple Sine Oscillator (TSO) — a unique indicator designed to capture cyclical market behavior and identify mean reversion opportunities. We further extended this concept into a complete Mean Reversion Strategy, detailing its structure, code logic, and real-time trade execution process.

The TSO’s mathematical foundation allows it to effectively detect overbought and oversold conditions, providing timely entry signals in range-bound markets. By incorporating multiple entries, the strategy enables traders to “scale in” and maximize potential profits during strong mean reversion phases. However, this same feature introduces additional exposure, meaning that while it can enhance gains, it also carries a higher risk of compounded losses when market conditions shift unexpectedly.

Through this article, we have demonstrated how a mathematical function can be transformed into a novel oscillator and then built upon to form a structured mean reversion trading strategy. The Triple Sine method showcases how creativity in mathematical design can lead to innovative trading tools with real-world applicability.

In our next article, we will evaluate the performance of the TSO when combined with trend-following indicators, testing its robustness and adaptability across various market conditions to determine its strengths, limitations, and optimization potential.

| FileName | Description |
| --- | --- |
| SineCube.mq5 | The file plots the mathematical sine function in the range of -2 **π** to +2 **π.** Once plotted, the template has to be updated to remove the plot |
| TripleSineOscillator.mq5 | This file generates the oscillator indicator on the chart window. |
| TSO\_EA.mq5 | This file is an expert advisor that executes the trade in real time. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20220.zip "Download all attachments in the single ZIP archive")

[SineCube.mq5](https://www.mql5.com/en/articles/download/20220/SineCube.mq5 "Download SineCube.mq5")(5.61 KB)

[TripleSineOscillator.mq5](https://www.mql5.com/en/articles/download/20220/TripleSineOscillator.mq5 "Download TripleSineOscillator.mq5")(8.02 KB)

[TSO\_EA.mq5](https://www.mql5.com/en/articles/download/20220/TSO_EA.mq5 "Download TSO_EA.mq5")(9.64 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing a Trading Strategy: Using a Volume-Bound Approach](https://www.mql5.com/en/articles/20469)
- [Developing a Trading Strategy: The Flower Volatility Index Trend-Following Approach](https://www.mql5.com/en/articles/20309)
- [Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://www.mql5.com/en/articles/20065)
- [Developing a Trading Strategy: The Butterfly Oscillator Method](https://www.mql5.com/en/articles/20113)
- [Building a Trading System (Part 5): Managing Gains Through Structured Trade Exits](https://www.mql5.com/en/articles/19693)
- [Building a Trading System (Part 4): How Random Exits Influence Trading Expectancy](https://www.mql5.com/en/articles/19211)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/499951)**
(2)


![Paul Afolabi](https://c.mql5.com/avatar/2025/11/6918bfb0-ecd2.jpg)

**[Paul Afolabi](https://www.mql5.com/en/users/aop93)**
\|
15 Nov 2025 at 17:56

Please can I get this for testing


![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
15 Nov 2025 at 18:44

**Paul Afolabi [#](https://www.mql5.com/en/forum/499951#comment_58521135):**

Please can I get this for testing

You will find it inside the article in the first post.


![Risk-Based Trade Placement EA with On-Chart UI (Part 2): Adding Interactivity and Logic](https://c.mql5.com/2/180/20159-risk-based-trade-placement-logo.png)[Risk-Based Trade Placement EA with On-Chart UI (Part 2): Adding Interactivity and Logic](https://www.mql5.com/en/articles/20159)

Learn how to build an interactive MQL5 Expert Advisor with an on-chart control panel. Know how to compute risk-based lot sizes and place trades directly from the chart.

![Automating Trading Strategies in MQL5 (Part 39): Statistical Mean Reversion with Confidence Intervals and Dashboard](https://c.mql5.com/2/180/20167-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 39): Statistical Mean Reversion with Confidence Intervals and Dashboard](https://www.mql5.com/en/articles/20167)

In this article, we develop an MQL5 Expert Advisor for statistical mean reversion trading, calculating moments like mean, variance, skewness, kurtosis, and Jarque-Bera statistics over a specified period to identify non-normal distributions and generate buy/sell signals based on confidence intervals with adaptive thresholds

![MQL5 Trading Tools (Part 10): Building a Strategy Tracker System with Visual Levels and Success Metrics](https://c.mql5.com/2/180/20229-mql5-trading-tools-part-10-logo__1.png)[MQL5 Trading Tools (Part 10): Building a Strategy Tracker System with Visual Levels and Success Metrics](https://www.mql5.com/en/articles/20229)

In this article, we develop an MQL5 strategy tracker system that detects moving average crossover signals filtered by a long-term MA, simulates or executes trades with configurable TP levels and SL in points, and monitors outcomes like TP/SL hits for performance analysis.

![How can century-old functions update your trading strategies?](https://c.mql5.com/2/120/How_100-Year-Old_Features_Can_Update_Your_Trading_Strategies__LOGO.png)[How can century-old functions update your trading strategies?](https://www.mql5.com/en/articles/17252)

This article considers the Rademacher and Walsh functions. We will explore ways to apply these functions to financial time series analysis and also consider various applications for them in trading.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=grzhlrzgcluieekkvddwifdyznsbpgyg&ssn=1769091893175540567&ssn_dr=0&ssn_sr=0&fv_date=1769091893&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20220&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Trading%20Strategy%3A%20The%20Triple%20Sine%20Mean%20Reversion%20Method%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909189390057371&fz_uniq=5049115398353298741&sv=2552)

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