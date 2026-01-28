---
title: Developing a Trading Strategy: The Butterfly Oscillator Method
url: https://www.mql5.com/en/articles/20113
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:25:08.483155
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/20113&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049117537247012159)

MetaTrader 5 / Trading


### Introduction

Indicators are mathematical tools designed to represent market data in a graphical form. They help traders analyze market trends, volatility, and momentum. Common examples include the Relative Strength Index (RSI), Average True Range (ATR), Moving Average (MA), and Relative Vigor Index (RVI). Each serves a unique purpose in identifying entry and exit points on trading charts.

In this article, we introduce a **new technical indicator**—the **Butterfly Oscillator**—based on a famous mathematical curve discovered in 1989 by **Temple H. Fay**. This curve, known as the **Butterfly Curve**, derives its name from the distinct wing-like shape of its graph. The objective here is to construct the Butterfly Oscillator using this mathematical foundation and develop a corresponding trading strategy.

### The Butterfly Curve: A Mathematical Foundation

The butterfly curve is defined by its parametric equations, which generate its distinctive, wing-like shape:

![butterflyEqn](https://c.mql5.com/2/178/butterflyEqn.png)

Where the parameter _t_ ranges over the interval  \[0,12π\].

Figure 1 illustrates the Butterfly Curve.

![butterfly](https://c.mql5.com/2/178/butterflyCurve_2.png)

Figure 1: Butterfly Curve

Notably:

- The _x-axis_ values display symmetry, while the _y-axis_ values are asymmetrical.
- The overall structure is nonlinear and cyclical, meaning it repeats its pattern periodically.
- The parametric equation takes a single input (t) and outputs corresponding _x_ and _y_ values.

Code Structure to Construct the Butterfly Curve:

At this point we show how the butterfly curve was plotted on the chart using script file.

```
#include <Canvas\Canvas.mqh>
```

The first step is to include the Canvas library in MQL5 using the line. This library provides all the necessary classes and functions required to draw and visualize the Butterfly Curve on the chart.

```
   // Create canvas for drawing
   int width = 800;   // Canvas width
   int height = 600;  // Canvas height
   string canvas_name = "ButterflyCurve";

   // Create canvas object
   CCanvas canvas;
   if(!canvas.CreateBitmapLabel(canvas_name, 200, 50, width, height, COLOR_FORMAT_XRGB_NOALPHA))
     {
      Print("Error creating canvas: ", GetLastError());
      return;
     }
```

First, the canvas dimensions are defined — with a width of 800 pixels and a height of 600 pixels — and the canvas name is set as "ButterflyCurve" . These parameters determine the size and label of the drawing area that will appear on the chart.

Next, a Canvas object (CCanvas canvas) is declared, which serves as the graphical surface for rendering all visual elements, such as lines, shapes, and text. The CreateBitmapLabel()function is then used to instantiate the canvas at a specific position on the chart — in this case, 200 pixels from the left and 50 pixels from the top. The function also specifies the color format (COLOR\_FORMAT\_XRGB\_NOALPHA) **,** which defines how colors and transparency are processed.

Finally, the code includes an error-handling mechanism: if the canvas creation fails, the program prints an error message containing the error code (GetLastError() ) and stops execution. This ensures that subsequent drawing operations are only performed when the canvas is successfully created and ready for use.

```
   // Set up drawing parameters
   int points = 1000;            // Number of points
   double t_start = 0;           // Start angle
   double t_end = 12 * M_PI;     // End angle
   double step = (t_end - t_start) / (points - 1);

   // Calculate curve points
   double x[], y[];
   ArrayResize(x, points);
   ArrayResize(y, points);

   for(int i = 0; i < points; i++)
     {
      double t = t_start + i * step;
      double expr = MathExp(MathCos(t)) - 2 * MathCos(4 * t) - MathPow(MathSin(t/12), 5);
      x[i] = MathSin(t) * expr;
      y[i] = MathCos(t) * expr;
     }
```

This portion of the code defines the mathematical setup and data structure needed to generate the Butterfly Curve before plotting it on the canvas.

First, it establishes the drawing parameters:

- points specifies the number of data points (1000) that will be used to draw the curve, ensuring a smooth and continuous line,
- t\_start and t\_end define the range of the variable t, which acts as the parameter for the Butterfly Curve equation — here, from 0 to 12π,
- step determines the incremental change in t between each point, ensuring an even distribution of data along the curve.

Next, two dynamic arrays, x\[\] and y\[\], are created and resized to hold all the computed coordinates of the Butterfly Curve. Each element in these arrays corresponds to one point on the curve.

Within the for loop, the Butterfly Curve equation is calculated for each value of t. These computed values represent the geometric coordinates of the Butterfly Curve, which are later used for scaling and plotting on the canvas.

```
   // Find coordinate bounds for scaling
   double x_min = x[ArrayMinimum(x)];
   double x_max = x[ArrayMaximum(x)];
   double y_min = y[ArrayMinimum(y)];
   double y_max = y[ArrayMaximum(y)];

   // Scale points to canvas coordinates
   int x_px[], y_px[];
   ArrayResize(x_px, points);
   ArrayResize(y_px, points);

   double x_scale = (width - 40) / (x_max - x_min);
   double y_scale = (height - 40) / (y_max - y_min);
   double scale = MathMin(x_scale, y_scale); // Maintain aspect ratio

   for(int i = 0; i < points; i++)
     {
      x_px[i] = (int)((x[i] - x_min) * scale) + 20;
      y_px[i] = height - (int)((y[i] - y_min) * scale) - 20; // Flip Y-axis for canvas coordinates
     }
```

This section of the code handles the scaling and coordinate transformation of the computed Butterfly Curve data points so that they fit neatly within the canvas area.

First, the program determines the coordinate bounds of the curve by finding the minimum and maximum values of both the **x** and **y** arrays. These values define the full range of the Butterfly Curve in both directions and are essential for correctly mapping the data to the screen.

Next, two new integer arrays — x\_px and y\_px — are created and resized to match the number of data points. These arrays will store the pixel-based coordinates that correspond to the scaled version of the curve.

The scaling factors are then computed:

- x\_scale converts the range of x-values to the available canvas width (leaving a 20-pixel margin on each side).
- y\_scale converts the range of y-values to the available canvas height (also with margins). To ensure that the curve maintains its aspect ratio (so it isn’t stretched or distorted), the smaller of these two scaling factors is selected as the final scale value.

Finally, each data point is transformed into canvas coordinates.

The formula translate the mathematical x–y values into screen positions, taking into account the margins and inverting the _Y-axis_ (since canvas coordinates start from the top-left corner). At the end of this process, all Butterfly Curve data points are properly scaled and positioned within the visible canvas area, ready for plotting.

```
   // Draw the curve
   canvas.Erase(ColorToARGB(clrWhite, 255)); // White background

   // Set line color - we'll pass this directly to Line() method
   uint line_color = ColorToARGB(clrBlue, 255);

   // Draw the polyline
   for(int i = 1; i < points; i++)
     {
      canvas.Line(x_px[i-1], y_px[i-1], x_px[i], y_px[i], line_color);
     }

   // Title
   canvas.FontSet("Arial", 20, FW_BOLD);
   canvas.TextOut(300, 20, "Butterfly Curve", ColorToARGB(clrBlack, 255));

   // Update display
   canvas.Update();
```

This section of the code focuses on rendering and styling the final Butterfly Curve on the canvas. We configure the plot's visual style by setting the canvas background to white and the data line to blue. The data points from  x\_px  and  y\_px  are then plotted. Finally, we set the title font to 20-point Arial in black and update the canvas to display the final visualization.

### Developing the Butterfly Oscillator

To construct the Butterfly Oscillator, we use the x-component of the butterfly curve because of its symmetrical property. The x-component is naturally bounded within the range \[-3, +3\], making it ideal for creating an oscillator.

To adapt this for trading, we define the input parameter  t  simply as the elapsed number of price bars.Since t represents bar count, the curve is identical across all timeframes. The underlying equation is cyclical, because of that the oscillator forms wave-like patterns that repeat after each complete cycle.

A complete cycle of the underlying butterfly curve occurs at t=24π, or approximately 75.4 bars. To control the oscillator's sensitivity, we introduce a step size (Δt)as a scaling factor.The number of bars (N) required for a full oscillator cycle is calculated as:  N = 24π / Δt

Examples:

- Step size = 1:     N=24π/1=76 bars (complete cycle)
- Step size = 0.5:  N=24π/0.5=151 bars
- Step size = 5:     N=24π/5=15 bars

The step size directly influences the oscillator’s behavior:

- Smallerstep sizes produce smoother and slower waveforms.
- Largerstep sizes create faster, more frequent oscillations.

This allows traders to customize the oscillator for either long-term pattern detection or short-term cycle analysis.

**Making the Oscillator Price-Reactive:**

While the oscillator can operate purely based on bar count, it can also be made responsive to price movement. To achieve this, we incorporate the difference between the candle’s closing and opening prices. This adjustment allows the oscillator to react to changes in price dynamics, making it more adaptive to real-time market volatility rather than depending solely on time progression.

This hybrid design—combining cyclical behavior with price sensitivity—enables the Butterfly Oscillator to provide both structural timing and momentum-based confirmation.

Before demonstrating how the oscillator performs in practice, let’s briefly outline the code structure used to implement it.

```
//--- plot Butterfly
#property indicator_label1  "Butterfly Oscillator"
#property indicator_type1   DRAW_LINE
#property indicator_color1  DodgerBlue
#property indicator_width1  2
#property indicator_style1  STYLE_SOLID

//--- indicator levels
#property indicator_level1  2.5
#property indicator_level2 -2.5
#property indicator_level3  0.0
```

This section of the code sets the visual and structural properties of the Butterfly Oscillator. It defines how the indicator appears on the chart — using a solid DodgerBlue line as default, labeled “Butterfly Oscillator,” with a thickness of 2 for clear visibility. Additionally, it establishes key reference levels at 2.5, -2.5, and 0.0. Overall, these settings ensure the oscillator is visually distinct and analytically useful for identifying market cycles.

```
//--- input parameters
input bool   UsePriceStep = false; // Use ClosePrice as Step size for t increment
input double tmStep       = 0.05;  // Step size for t increment
```

This section defines user input parameters that control how the Butterfly Oscillator calculates its time increments.

- UsePriceStep: A Boolean option that lets the user choose whether to base the step size on price movement instead of a number of bars.
- tmStep: A numeric input that sets the fixed step size for incrementing the variable _t_.

These inputs give traders flexibility to adjust how smoothly or sensitively the oscillator responds to market changes. The chosen step size greatly influences the oscillator’s behavior and overall dynamics.

```
double CalButterflyValue(int bar_index, double bar_close, double bar_open)
  {
   double tStep = UsePriceStep ? MathMod((bar_close - bar_open) / _Point, tmStep) : tmStep;
   double t = bar_index * tStep;

   // Butterfly curve formula
   double x = MathSin(t) *
              (MathExp(MathCos(t)) - 2.0 * MathCos(4.0 * t) - MathPow(MathSin(t / 12.0), 5));

   return (x);
  }
```

This section defines the Butterfly function, which calculates the oscillator’s value for each bar on the chart.

The function CalButterflyValue() takes three parameters:

- bar\_index – the position of the current bar on the chart,
- bar\_close – the closing price of that bar, and
- bar\_open – the opening price of that bar.

Inside the function, a step size (tStep) is computed. If UsePriceStep is enabled, it dynamically adjusts based on price movement; otherwise, it uses the fixed tmStep value. The variable **t** is then derived from the bar index and the step size.

Finally, the Butterfly curve formula is applied to compute **x**, representing the oscillator’s value at that bar. This value captures the mathematical shape of the Butterfly curve and is returned for plotting on the chart.

```
   int pStart = prev_calculated == 0 ? 0 : prev_calculated - 1;

   for(int i = pStart; i < rates_total; i++)
     {
      double bar_close = close[i];
      double bar_open  = open[i];
      ButterflyBuffer[i] = CalButterflyValue(i, bar_close, bar_open);
     }
```

This section initializes and fills the Butterfly Oscillator buffer with computed values for chart plotting.

The variable pStart determines where the calculation should begin — starting from zero for the first run, or from the last calculated bar on subsequent updates to improve efficiency. Each calculated value is stored in the ButterflyBuffer array, which is later used to plot the oscillator line in the indicator window.

### Demonstrating the Butterfly Oscillator

In our demonstration of the Butterfly Oscillator, we examine its behavior under two conditions: using bar count only and using bar count combined with price action. This comparison allows us to observe how the oscillator responds in each scenario.

Case 1: The Effect of Step Size (Bar Count Only)

Figure 2 illustrates the oscillator when it is driven solely by bar count. In this case, the parameter UsePriceAction is set to false, meaning that only the number of bars determines the oscillator’s progression. Three different step sizes—0.05, 0.5, and 5—are used to study their effects on the waveform.

![butterflyDemo1](https://c.mql5.com/2/178/BfyOsc.gif)

Figure 2: Butterfly Oscillator without Price Action

From the graph, it can be observed that:

- When the step size is small (e.g., 0.05), the oscillator takes longer to complete a full cycle, producing a smoother and more gradual wave.
- When the step size is large (e.g., 5), the oscillator completes its cycle much faster, resulting in a sharper and more irregular pattern.

This demonstrates that the step size directly controls the oscillator’s smoothness and frequency—smaller steps emphasize long-term cyclical movements, while larger steps highlight short-term fluctuations.

Case 2: The Effect of Step Size (With Price Action)

Figure 3 illustrates the oscillator pattern when it is driven by price action. In this case, the oscillator’s behavior is influenced not only by the step size but also by market price movements. Similar to the bar count mode, the oscillator moves faster with larger step sizes and slower with smaller step sizes. However, unlike the previous case, the inclusion of price action adds variability to the curve.

![butterflydemo2](https://c.mql5.com/2/178/BfyOsc2.gif)

Figure 3: Butterfly Oscillator  Driven by Price Action

Because price action tends to move in a chaotic and nonlinear manner, its fluctuations are reflected in the shape of the oscillator. As a result, the Butterfly Oscillator becomes more dynamic, adapting its waveform to the underlying market volatility rather than following a purely mathematical rhythm.

### Butterfly Oscillator Trading Strategy

Now that we have successfully developed the Butterfly Oscillator indicator, we can proceed to build a simple Butterfly Expert Advisor (EA) that uses this indicator as the entry criterion. The trading strategy is designed to detect cyclical turning points in the oscillator and align them with the broader market trend indicated by the Moving Average (MA).

In this approach, we define two strategies for market entry:

Strategy 1

- Buy Signal: When the Moving Average (MA) indicates an uptrendand the Butterfly Oscillator crosses above +2.5, open a buy position
- Sell Signal: When the MA indicates a downtrend and the Butterfly Oscillator crosses above +2.5, open a sell position.

Strategy 2

- Buy Signal: When the MA indicates an uptrend and the Butterfly Oscillator crosses below -2.5, open a buy position.
- Sell Signal: When the MA indicates a downtrend and the Butterfly Oscillator crosses below -2.5, open a sell position.

In both strategies, the MA trend direction serves as the trend filter, while the Butterfly Oscillator acts as the entry trigger.

- Strategy 1 focuses on upper threshold (+2.5) signals, typically identifying the Peak of  oscillator cycles.
- Strategy 2 focuses on lower threshold (-2.5) signals, capturing the trough of the cycle.

The Butterfly EA provides flexibility by allowing traders to choose between Strategy 1 and Strategy 2, or to run both sequentially. The EA also includes an option to determine whether the Butterfly Oscillator should react to price movements or rely solely on bar count progression.

The relevant input parameters are defined as follows:

```
//--- Strategy Selection
input bool UsePriceStep = false;      // Use ClosePrice
input bool EnableStrategy1 = true;    // Enable Strategy 1 Peak
input bool EnableStrategy2 = false;   // Enable Strategy 2 Trough
```

UsePriceStep – Enables or disables the price action mode.

- When set to true, the EA incorporates price movement (close-open difference)into the oscillator’s behavior, making it responsive to market dynamics.
- When set to false, the oscillator relies solely on bar count, maintaining a fixed cyclical pattern.

Strategy Selection:

- Strategy 1 is designed to generate signals based on the peak formations of the oscillator.
- Strategy 2 is designed to generate signals based on its trough formations.

When both Strategy 1 and Strategy 2 are active, the EA gives priority to whichever threshold condition is triggered first during market execution.

To ensure disciplined trade management, the EA is designed to allow only one open position at a time. This prevents conflicting trades from being executed simultaneously, maintaining clarity in strategy performance evaluation.

The core trading logic for the Butterfly EA is implemented as follows:

```
   //--- Strategy 1: Peak
   if(EnableStrategy1 && prevValue < 2.5 && currValue > 2.5)
   {
      if(maFast > maSlow)
         OpenBuy();
      else if(maFast < maSlow)
         OpenSell();
   }

   //--- Strategy 2: Trough
   if(EnableStrategy2 && prevValue > -2.5 && currValue < -2.5)
   {
      if(maFast > maSlow)
         OpenBuy();
      else if(maFast < maSlow)
         OpenSell();
   }
```

Strategy 1 (Peak):

When the previous Butterfly Oscillator value is below +2.5 and the current value crosses above +2.5, a signal is generated.

- If the fast MA is above the slow MA, it indicates an uptrend, and a buy order is executed.
- If the fast MAis below the slow MA, it indicates a downtrend, and a sell order is placed.

Strategy 2 (Trough):

When the previous Butterfly Oscillator value is above –2.5 and the current value crosses below –2.5, a signal is generated.

- If the fast MA is above the slow MA, the EA interprets this as an uptrend, and a buy order is executed.
- If the fast MA is below the slow MA, indicating a downtrend, a sell order is issued.

**Testing the Butterfly Oscillator EA:**

To tailor the EA to individual trading preferences and risk management rules, a comprehensive set of input parameters is provided. These settings are categorized for clarity and allow users to control trade management, indicator behavior, and strategy selection.

![InputVal](https://c.mql5.com/2/178/InpVal.png)

Figure 4: EA Input Parameters

Strategy Testing and Entry Signal Demonstration:

- Step Size : 0.3:

To demonstrate the entry signals generated by the Butterfly EA, we tested Strategy 1 using a step size of 0.3.

Figure 5 illustrates the resulting buy and sell signals produced under this configuration. With a larger step size, the Butterfly Oscillator completes its cycle over fewer bars, allowing it to reach its peaks and troughs more quickly. As a result, the EA generates more frequent entry signals within a shorter time span.

![](https://c.mql5.com/2/178/BtyOsc_demo1.gif)

Figure 5: Strategy 1 with Step Size =0.3

- Step Size : 0.03:

Using a step size of 0.03, we further examined the behavior of the Butterfly Expert Advisor under Strategy 1.

Figure 6 presents the buy and sell signals generated with this smaller step size. When the step size is small, the Butterfly Oscillator requires more bars to complete a full cycle. Consequently, it reaches its peaks and troughs more slowly, resulting in fewer entry signals being produced over the same period.

![](https://c.mql5.com/2/178/BtyOsc_demo2.gif)

Figure 6: Strategy 1 with Step Size =0.03

This behavior highlights the direct influence of the step size on trading signal frequency—larger values lead to faster oscillations and more trades, while smaller values create slower, smoother cycles with fewer but more stable signals.

### Conclusion

In this article, we used the Butterfly Curve concept to build an oscillator and a trading strategy. The Butterfly Oscillator demonstrated how mathematical patterns can be adapted to market dynamics, providing unique insights into price movements. By varying the step size, we observed how the oscillator’s smoothness and signal frequency change, and how these characteristics can influence trade entries when combined with moving average trends.

The developed Butterfly EA successfully applied the oscillator as an entry criterion, offering two distinct strategies—one for peaks and another for troughs—each responsive to both market direction and oscillator thresholds.

In our next article, we will experiment with various instruments to test the strength and weakness of this concept. We will further experiment with this new oscillator in combination with other oscillators to fine-tune our entry strategy. Stay tuned for more entry tips and enhancements to the Butterfly trading system.

| File | Description |
| --- | --- |
| ButterflyPlot.mq5 | This script file is responsible for generating and displaying the Butterfly Curve on the active chart window using its defined mathematical function. When the script is executed, it plots the curve visually on the chart window. Once the Butterfly Curve is drawn, it remains displayed on the chart as part of the current visual layer. The curve generated by this script can only be cleared or removed when the chart’s template is changed or updated to a new one. |
| ButterflyOscillator.mq5 | This file defines the Butterfly Indicator, which is designed to appear in a separate subwindow below the main price chart. |
| ButterflyOscillatorEA.mq5 | This file defines an Expert Advisorthat utilizes the Butterfly Oscillator to automatically open and manage trading positions the platform. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20113.zip "Download all attachments in the single ZIP archive")

[ButterflyPlot.mq5](https://www.mql5.com/en/articles/download/20113/ButterflyPlot.mq5 "Download ButterflyPlot.mq5")(5.79 KB)

[ButterflyOscillator.mq5](https://www.mql5.com/en/articles/download/20113/ButterflyOscillator.mq5 "Download ButterflyOscillator.mq5")(2.84 KB)

[ButterflyOscillatorEA.mq5](https://www.mql5.com/en/articles/download/20113/ButterflyOscillatorEA.mq5 "Download ButterflyOscillatorEA.mq5")(5.74 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing a Trading Strategy: Using a Volume-Bound Approach](https://www.mql5.com/en/articles/20469)
- [Developing a Trading Strategy: The Flower Volatility Index Trend-Following Approach](https://www.mql5.com/en/articles/20309)
- [Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://www.mql5.com/en/articles/20065)
- [Developing a Trading Strategy: The Triple Sine Mean Reversion Method](https://www.mql5.com/en/articles/20220)
- [Building a Trading System (Part 5): Managing Gains Through Structured Trade Exits](https://www.mql5.com/en/articles/19693)
- [Building a Trading System (Part 4): How Random Exits Influence Trading Expectancy](https://www.mql5.com/en/articles/19211)

**[Go to discussion](https://www.mql5.com/en/forum/499521)**

![Price Action Analysis Toolkit Development (Part 49): Integrating Trend, Momentum, and Volatility Indicators into One MQL5 System](https://c.mql5.com/2/179/20168-price-action-analysis-toolkit-logo__1.png)[Price Action Analysis Toolkit Development (Part 49): Integrating Trend, Momentum, and Volatility Indicators into One MQL5 System](https://www.mql5.com/en/articles/20168)

Simplify your MetaTrader  5 charts with the Multi  Indicator  Handler EA. This interactive dashboard merges trend, momentum, and volatility indicators into one real‑time panel. Switch instantly between profiles to focus on the analysis you need most. Declutter with one‑click Hide/Show controls and stay focused on price action. Read on to learn step‑by‑step how to build and customize it yourself in MQL5.

![Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://c.mql5.com/2/179/20173-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)

This article describes two additional scoring criteria used for selection of baskets of stocks to be traded in mean-reversion strategies, more specifically, in cointegration based statistical arbitrage. It complements a previous article where liquidity and strength of the cointegration vectors were presented, along with the strategic criteria of timeframe and lookback period, by including the stability of the cointegration vectors and the time to mean reversion (half-time). The article includes the commented results of a backtest with the new filters applied and the files required for its reproduction are also provided.

![Neural Networks in Trading: Memory Augmented Context-Aware Learning (MacroHFT) for Cryptocurrency Markets](https://c.mql5.com/2/112/Neural_Networks_in_Trading_MacroHFT____LOGO__1.png)[Neural Networks in Trading: Memory Augmented Context-Aware Learning (MacroHFT) for Cryptocurrency Markets](https://www.mql5.com/en/articles/16975)

I invite you to explore the MacroHFT framework, which applies context-aware reinforcement learning and memory to improve high-frequency cryptocurrency trading decisions using macroeconomic data and adaptive agents.

![Optimizing Long-Term Trades: Engulfing Candles and Liquidity Strategies](https://c.mql5.com/2/179/19756-mastering-high-time-frame-trading-logo.png)[Optimizing Long-Term Trades: Engulfing Candles and Liquidity Strategies](https://www.mql5.com/en/articles/19756)

This is a high-timeframe-based EA that makes long-term analyses, trading decisions, and executions based on higher-timeframe analyses of W1, D1, and MN. This article will explore in detail an EA that is specifically designed for long-term traders who are patient enough to withstand and hold their positions during tumultuous lower time frame price action without changing their bias frequently until take-profit targets are hit.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/20113&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049117537247012159)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).