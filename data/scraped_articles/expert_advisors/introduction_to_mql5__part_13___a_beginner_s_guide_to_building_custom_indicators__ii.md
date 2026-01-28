---
title: Introduction to MQL5 (Part 13): A Beginner's Guide to Building Custom Indicators (II)
url: https://www.mql5.com/en/articles/17296
categories: Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:42:27.335611
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/17296&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049318296903330128)

MetaTrader 5 / Expert Advisors


### Introduction

Welcome back to our MQL5 series! [Part 12](https://www.mql5.com/en/articles/17096) of this series explored the fundamentals of building custom indicators in MQL5. We created a Moving Average indicator from scratch, implementing its logic manually instead of relying on built-in functions. Then, we extended this knowledge by transforming it into a Moving Average in candle format, demonstrating how to manipulate graphical elements within an indicator.

Building on that foundation, this article will introduce more interesting concepts in indicator development. We will use a project-based approach as usual to make sure you understand topics by putting them into practice. The creation of a Heikin Ashi indicator and the calculation of a Moving Average utilizing its data will be the main objectives. After these indicators are built, we will develop an Expert Advisor that incorporates the Heikin Ashi and Moving Average indicators. Even those who are new to MQL5 can follow along because this is a beginner-friendly article. To assist you in understanding not only how the implementation functions but also why each step is required, every line of code will be thoroughly explained.

This article will cover a strategy that is solely intended for educational purposes. It is not meant to be a trading strategy that guarantees success or financial advice. Before using strategy in live trading, always test them in a risk-free setting.

**Heikin Ashi (HA) and HA Moving Average**

![Figure 1. Heikin Ashi and MA indicator](https://c.mql5.com/2/121/Figure_1.png)

In this article, you'll learn:

- How to create a custom Heikin Ashi indicator from scratch in MQL5.
- Utilizing Heikin Ashi candle data to calculate Heikin Ashi Moving Average.
- Using the iCustom() function to access non-built-in indicators and integrate their data into trading strategies.
- Defining entry conditions using Heikin Ashi and MA crossovers.
- Managing risk effectively by setting stop-loss and take-profit levels dynamically using Heikin Ashi-based calculations.
- Applying a trailing stop mechanism using Heikin Ashi candle patterns to secure profits as trends develop.

### 1\. Heikin Ashi Indicator

**1.1. Understanding the Heikin Ashi Indicator**

The Heikin Ashi (HA) indicator makes trends easier to see, HA uses a unique technique to determine new values based on averaged historical price data, in contrast to typical candlestick charts, which display the precise open, high, low, and close prices for each period. This helps traders sort through the clutter and concentrate on the important things by producing a clearer, more understandable picture of market movements.

Each candle in a typical candlestick chart shows the movement of the price over a given period. Whereas a red (bearish) candle signifies the contrary, a green (bullish) candle shows that the price closed higher than it opened. A glimpse of market volatility is provided to traders by the thin wicks above and below the candle body, which display the highest and lowest prices attained during that time.

But Heikin Ashi candles do things differently. They use a special computation to smooth out trends rather than accurately reporting price movements. Longer green candles with fewer wicks are indicative of an upswing, which makes it simpler to identify and track bullish momentum. In a similar vein, red candles become more noticeable during a downtrend, amply indicating negative swings. Smaller candles with wicks on both end frequently emerge when the market is in a range or lacks significant momentum, indicating traders' hesitancy or indecision.

The HA indicator is unique in that it modifies conventional candlestick calculations by using an averaging technique. It produces new values that produce a smoother, more consistent depiction of price motion rather than directly plotting the market's open, high, low, and closing prices. By removing the "noise" of small price fluctuations, traders are better able to spot trends and make wiser choices.

**Heikin Ashi Close**

The average of the current period's open, high, low, and close values is used to determine the Heikin Ashi Close price. A more balanced perspective of price change is offered by standard candlesticks, which simply use the closing price.

The formula is:

![Figure 2. H A_Close Formula](https://c.mql5.com/2/122/HA_CLOSE.png)

The Heikin Ashi Close smoothes out price swings by averaging these four values, which makes patterns more obvious visually.

**Heikin Ashi Open**

The preceding Heikin Ashi candle, not the actual market opening, is used to determine the Heikin Ashi Open price. It's determined by averaging the previous Heikin Ashi Open and Close values:

![Figure 3. H A_Open Formula](https://c.mql5.com/2/122/HA_Open.png)

Heikin Ashi reduces the unpredictable jumps that frequently occur in conventional candlestick charts by establishing a continuous flow of price motion by the linking of each new candle's Open value to the one before it.

**Heikin Ashi High**

The Heikin Ashi High price is the highest value attained throughout the time; however, it considers three values: the high of the current period, the Heikin Ashi Open, and the Heikin Ashi Close, rather than just the market's real high. Out of these three, the highest is selected:

![Figure 4. H A_High Formula](https://c.mql5.com/2/122/HA_High.png)

**Heikin Ashi Low**

Similarly, the Heikin Ashi Low price is calculated by selecting the lowest value between the Heikin Ashi Close, Heikin Ashi Open, and the actual low for the period:

![Figure 5. H A_Low Formula](https://c.mql5.com/2/122/HA_Low.png)

This approach maintains consistency with the smoothing methods while guaranteeing that the Heikin Ashi Low captures the lowest point of the price fluctuation. Heikin Ashi removes minor oscillations and gives a more accurate picture of the market's direction by employing these computations. We will use this reasoning to construct our own Heikin Ashi indicator in MQL5 in the following part.

**1.2. Benefits of Using Heikin Ashi**

Because it can eliminate small price swings, the Heikin Ashi indicator is a favorite among traders and makes identifying trends much simpler. Because of their rapid color changes, traditional candlestick charts can be difficult to read and frequently leave traders unsure of whether the market is heading upward or downward. By employing averaged price data to smooth out the chart, Heikin Ashi addresses this issue and helps you avoid becoming bogged down in the details.

A string of red candles indicates a downturn, whereas a series of green candles typically indicates a strong rising trend. Because of this clarity, it is simpler to distinguish between real, long-term market changes and brief declines.  Heikin Ashi helps you cut down on false signals, avoiding unnecessary trades caused by short-term price swings. By using historical data, it filters out market noise, offering more reliable trend confirmation. Many traders combine it with tools like RSI or Moving Averages to improve their strategies. With its clearer view of price action, Heikin Ashi makes it easier to decide when to enter or exit trades.

**1.3. Implementing Heikin Ashi in MQL5**

Implementing the Heikin Ashi indicator in MQL5 is the next step after learning about its operation. We will build our own Heikin Ashi indicator from scratch because MetaTrader 5 lacks one. To do this, the Heikin Ashi formulas must be coded, applied to price data, and the indication must be shown on the chart.

As usual, developing a program begins with drafting a pseudocode that outlines the logic before we begin writing code. This guarantees that we comprehend every phase before putting the program into action and helps us structure it appropriately.

**Pseudocode:**

**SETUP INDICATOR**

- Set the indicator to be plotted in a separate chart window.
- Plot settings for Heikin Ashi candles
- Define 4 buffers to store Heikin Ashi values (Open, High, Low, Close).

**DEFINE BUFFERS**

Create buffers to store the calculated values for:

- Heikin Ashi Open
- Heikin Ashi High
- Heikin Ashi Low
- Heikin Ashi Close

**CALCULATE HEIKIN ASHI VALUES**

Loop through historical price data to compute Heikin Ashi values using the formulas:

- **HA Close** = (Open + High + Low + Close) / 4

- **HA Open** = (Previous HA Open + Previous HA Close) / 2

- **HA High** = Maximum of (High, HA Open, HA Close)

- **HA Low** = Minimum of (Low, HA Open, HA Close)


**1.3.1. Creating and Customizing Heikin Ashi**

We must now move further with the programming after creating a pseudocode. As was covered in the last article, the first step in designing and modifying an indication is to picture how it should look on the chart. The plotting of the indicator, the display of the Heikin Ashi candles, and the inclusion of any other components, such as colors for bullish and bearish trends, should all be decided upon before creating any code.

We must make sure that our custom indicator appropriately substitutes its own calculated values for the normal candlestick charts, since Heikin Ashi alters the appearance of candlesticks. This entails setting up buffers for the open, high, low, and close prices as well as making sure the colors change dynamically to show both bearish and bullish trends. We can specify the indicator structure and start writing the code as soon as we have this clear visualization. We must define the Heikin Ashi indicator's property settings before we can apply its logic. These parameters dictate the style of the plotted elements (such Heikin Ashi candles), the number of buffers the indicator will use, and how the indication will appear on the chart.

**Example:**

```
// PROPERTY SETTINGS
#property indicator_separate_window
#property indicator_buffers 5
#property indicator_plots   1

// PLOT SETTINGS FOR HEIKIN ASHI CANDLES
#property indicator_label1  "Heikin Ashi"
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  clrGreen, clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
```

**Explanation:**

**Property Settings**

```
#property indicator_separate_window
```

This instructs MetaTrader 5 that rather than being superimposed on the main chart, the indicator ought to be shown in a different window. This line would be removed if we wanted it to show up straight on the price chart.

**Analogy**

Consider your trading chart to be a workstation where you examine changes in the market. Essential tools, including as price candles and conventional indicators, are placed on the main chart, which functions similarly to a tabletop. Now, picture yourself working on a meticulous side project that needs a specific area, such as a tiny whiteboard beside your desk. By moving this particular activity to the whiteboard, you can concentrate on it independently rather than clogging the main workspace.

In a similar vein, #property indicator\_separate\_window makes it simpler to examine trends without interfering with standard candlestick data by moving the Heikin Ashi indicator to its own window instead of superimposing it on the main price chart.

```
#property indicator_buffers 5
```

This specifies how many buffers the indicator will utilize. Five buffers are being used in this instance, one for color representation and the other for storing the computed Heikin Ashi values (Open, High, Low, and Close).

**Analogy**

Now that you have a different whiteboard beside your workbench for your side project, picture yourself needing five different trays to keep track of your work. Sketches, measurements, notes, and so on are stored in different trays. This keeps everything organized so you can quickly get the relevant information when you need it.

Similar to these trays, #property indicator\_buffers 5 makes sure that various Heikin Ashi data points are kept apart. Here, we have five buffers: one for color representation and four for the Heikin Ashi values (Open, High, Low, and Close). These buffers keep the indicator's calculations structured, which makes it simpler to display the right data on the chart, much the way the trays maintain your workspace orderly.

```
#property indicator_plots 1
```

This indicates how many charts the indicator will show. We just need one plot because we are charting Heikin Ashi candles as a single unit.

**Analogy**

After arranging your supplies into distinct trays and setting up your whiteboard workplace, the following step is to choose how to display your work. Instead of creating several distinct charts, picture yourself making a single, comprehensive diagram that unifies all the data into a single, understandable visual depiction.

Likewise, MetaTrader 5 is informed by #property indicator\_plots 1 that the Heikin Ashi indicator will be shown as a single plotted element. Similar to your single diagram on the whiteboard, several buffers hold distinct data (Open, High, Low, Close, and color), but they all combine to create a single set of candlesticks. We just need one plot to show the Heikin Ashi candles on the chart because we are only plotting them.

**Plot Settings**

```
#property indicator_label1  "Heikin Ashi"
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  clrGreen, clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
```

**Analogy**

After arranging your materials and setting up your whiteboard workplace, it's important to clearly and understandably convey your findings. To make trends more apparent, you choose to employ color-coded symbols rather than writing plain text or creating abstract designs. You clearly identify what your diagram symbolizes by labeling it "Heikin Ashi" for anyone observing the whiteboard. In the same way, #property indicator\_label1 "Heikin Ashi" gives the indicator a name and guarantees that it shows up in the MetaTrader 5 indication list. In this manner, traders may quickly identify it on their charts amid other indications.

#property indicator\_type1 DRAW\_COLOR\_CANDLES tells MetaTrader 5 to use color-coded candlesticks instead of lines or histograms. The colors are defined by #property indicator\_color1 clrGreen, clrRed, where green represents bullish candles and red represents bearish candles. This visual clarity makes it easier to spot trends at a glance. To keep your whiteboard neat and readable, you decide to use a solid marker stroke rather than dashed or dotted lines.

Similarly, #property indicator\_style1 STYLE\_SOLID ensures the Heikin Ashi candlesticks are filled with a solid color, making them visually distinct. Lastly, just as you avoid making your lines too thick so they don’t clutter your diagram, #property indicator\_width1 1 keeps the candle outlines at a reasonable width for clarity without overwhelming the chart. By setting up the Heikin Ashi indicator this way, we create a clear, structured, and visually intuitive representation of market trends, just as you’ve done with your well-organized whiteboard workspace.

Now that we have set the indicator properties and plot settings, the next step is to define buffers that will store the Heikin Ashi candle prices. Buffers act as storage containers for the calculated values of the indicator, allowing MetaTrader 5 to display them on the chart. In this case, we need buffers to store the Heikin Ashi Open, High, Low, and Close prices, as well as an additional buffer for color representation. We will also set their respective buffer indices to ensure that each buffer correctly corresponds to its intended data.

**Example:**

```
// PROPERTY SETTINGS
#property indicator_separate_window
#property indicator_buffers 5
#property indicator_plots   1

// PLOT SETTINGS FOR HEIKIN ASHI CANDLES
#property indicator_label1  "Heikin Ashi"
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  clrGreen, clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

// INDICATOR BUFFERS
double HA_Open[];
double HA_High[];
double HA_Low[];
double HA_Close[];
double ColorBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
// SET BUFFERS
   SetIndexBuffer(0, HA_Open, INDICATOR_DATA);
   SetIndexBuffer(1, HA_High, INDICATOR_DATA);
   SetIndexBuffer(2, HA_Low, INDICATOR_DATA);
   SetIndexBuffer(3, HA_Close, INDICATOR_DATA);
   SetIndexBuffer(4, ColorBuffer, INDICATOR_COLOR_INDEX);

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {

   return(rates_total);

  }
```

**Explanation:**

We build arrays (buffers) that will keep the Open, High, Low, and Close prices of the modified candlesticks to store the Heikin Ashi indicator's computed values. We also control the color of each candle using a different buffer.

```
double HA_Open[];
double HA_High[];
double HA_Low[];
double HA_Close[];
double ColorBuffer[];
```

The HA\_Open\[\] buffer has the Heikin Ashi open price of each candle, whereas HA\_High\[\] contains the Heikin Ashi candle's highest price. Similarly, the closing price of the Heikin Ashi candle is held by HA\_Close\[\], while the lowest price is recorded by HA\_Low\[\]. Additionally, the ColorBuffer\[\] is used to decide the color of each candle to distinguish between bullish candles (green) and bearish candles (red). The chart can save and display the updated Heikin Ashi candlesticks thanks to these buffers working together.

```
SetIndexBuffer(0, HA_Open, INDICATOR_DATA);
SetIndexBuffer(1, HA_High, INDICATOR_DATA);
SetIndexBuffer(2, HA_Low, INDICATOR_DATA);
SetIndexBuffer(3, HA_Close, INDICATOR_DATA);
SetIndexBuffer(4, ColorBuffer, INDICATOR_COLOR_INDEX);
```

The SetIndexBuffer function in MetaTrader 5 links particular buffers to their corresponding indices, ensuring that Heikin Ashi data is handled and displayed accurately. According to the platform's candlestick structure, the Open price is always allocated to index 0, and the High, Low, and Close values are mapped to indices 1, 2, and 3. In the absence of proper indexing, MetaTrader 5 might not identify the data as legitimate candlesticks, which could result in missing chart elements or display problems.

SetIndexBuffer(4, ColorBuffer, INDICATOR\_COLOR\_INDEX) specifies the color of each candle, indicating bullish (green) or bearish (red) movements, to visually differentiate trends. The Heikin Ashi indication guarantees correct price representation and visual styling by appropriately indexing these buffers, enabling traders to swiftly analyze trends and make well-informed decisions.

We must now carry out the computations that produce Heikin Ashi values after configuring the indicator properties, defining buffers, and connecting them to the appropriate indexes.

**Example:**

```
// PROPERTY SETTINGS
#property indicator_separate_window
#property indicator_buffers 5
#property indicator_plots   1

// PLOT SETTINGS FOR HEIKIN ASHI CANDLES
#property indicator_label1  "Heikin Ashi"
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  clrGreen, clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

// INDICATOR BUFFERS
double HA_Open[];
double HA_High[];
double HA_Low[];
double HA_Close[];
double ColorBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
// SET BUFFERS
   SetIndexBuffer(0, HA_Open, INDICATOR_DATA);
   SetIndexBuffer(1, HA_High, INDICATOR_DATA);
   SetIndexBuffer(2, HA_Low, INDICATOR_DATA);
   SetIndexBuffer(3, HA_Close, INDICATOR_DATA);
   SetIndexBuffer(4, ColorBuffer, INDICATOR_COLOR_INDEX);

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {

   if(rates_total < 2)
      return 0; // ENSURE ENOUGH DATA

   for(int i = 1; i < rates_total; i++)  // START FROM SECOND BAR
     {
      // HEIKIN ASHI CLOSE FORMULA
      HA_Close[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;

      // HEIKIN ASHI OPEN FORMULA
      HA_Open[i] = (HA_Open[i - 1] + HA_Close[i - 1]) / 2.0;

      // HEIKIN ASHI HIGH FORMULA
      HA_High[i] = MathMax(high[i], MathMax(HA_Open[i], HA_Close[i]));

      // HEIKIN ASHI LOW FORMULA
      HA_Low[i] = MathMin(low[i], MathMin(HA_Open[i], HA_Close[i]));

      // SET COLOR: GREEN FOR BULLISH, RED FOR BEARISH
      ColorBuffer[i] = (HA_Close[i] >= HA_Open[i]) ? 0 : 1;
     }

   return(rates_total);

  }
```

Since the first bar does not contain any previous data, this method computes Heikin Ashi values beginning with the second bar. To smooth out price swings, the close price is the mean of the current bar's open, high, low, and close prices. The Open price ensures seamless transitions by averaging the Heikin Ashi Open and Close of the previous bar. The current bar's high/low and Heikin Ashi open/close are the highest and lowest figures from which the high and low prices are calculated. Finally, candles are colored red (bearish) otherwise or green (bullish) if the Close is less than or equal to the Open. This helps traders see trends by cutting down on market noise.

![Figure 6. HA Indicator](https://c.mql5.com/2/121/Figure_2.png)

### 2\. Creating a Moving Average from Heikin Ashi Data

Now that we have successfully generated Heikin Ashi candles, the next step is to create a Moving Average (MA) based on the Heikin Ashi values instead of standard price data.

**Pseudocode:**

**MODIFY INDICATOR PROPERTIES**

- Adjust the buffer count from five to six to make room for the Heikin Ashi Moving Average.

- Change the number of plots from 1 to 2 so that the candles and the Heikin Ashi Moving Average may be seen together.


**DEFINE BUFFER FOR HEIKIN ASHI MOVING AVERAGE**

- Create a buffer to store Heikin Ashi Moving Average values.
- Define an input variable for the Moving Average period (e.g., 20).

**SET BUFFER FOR HEIKIN ASHI MOVING AVERAGE**

- Link the buffer to an index to store calculated MA values.
- Set the plot index to begin at the Moving Average period to ensure proper display.

**CALCULATE HEIKIN ASHI MOVING AVERAGE**

- Start looping from (period - 1) to ensure enough data points.
- Compute the sum of the last 'n' Heikin Ashi Close values.
- Divide the sum by the period and store the result in the buffer.

**Example:**

```
// PROPERTY SETTINGS
#property indicator_separate_window
#property indicator_buffers 6
#property indicator_plots   2

// PLOT SETTINGS FOR HEIKIN ASHI CANDLES
#property indicator_label1  "Heikin Ashi"
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  clrGreen, clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

//PROPERTIES OF THE Heikin MA
#property indicator_label2  "Heikin MA"
#property indicator_type2   DRAW_LINE
#property indicator_style2  STYLE_DASH
#property indicator_width2  1
#property indicator_color2  clrBrown

// INDICATOR BUFFERS
double HA_Open[];
double HA_High[];
double HA_Low[];
double HA_Close[];
double ColorBuffer[];

double Heikin_MA_Buffer[];
int input  heikin_ma_period = 20;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
    // SET BUFFERS
    SetIndexBuffer(0, HA_Open, INDICATOR_DATA);
    SetIndexBuffer(1, HA_High, INDICATOR_DATA);
    SetIndexBuffer(2, HA_Low, INDICATOR_DATA);
    SetIndexBuffer(3, HA_Close, INDICATOR_DATA);
    SetIndexBuffer(4, ColorBuffer, INDICATOR_COLOR_INDEX);

    SetIndexBuffer(5, Heikin_MA_Buffer, INDICATOR_DATA);
    PlotIndexSetInteger(5, PLOT_DRAW_BEGIN, heikin_ma_period);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
    if (rates_total < 2) return 0; // ENSURE ENOUGH DATA

    for (int i = 1; i < rates_total; i++) // START FROM SECOND BAR
    {
        // HEIKIN ASHI CLOSE FORMULA
        HA_Close[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;

        // HEIKIN ASHI OPEN FORMULA
        HA_Open[i] = (HA_Open[i - 1] + HA_Close[i - 1]) / 2.0;

        // HEIKIN ASHI HIGH FORMULA
        HA_High[i] = MathMax(high[i], MathMax(HA_Open[i], HA_Close[i]));

        // HEIKIN ASHI LOW FORMULA
        HA_Low[i] = MathMin(low[i], MathMin(HA_Open[i], HA_Close[i]));

        // SET COLOR: GREEN FOR BULLISH, RED FOR BEARISH
        ColorBuffer[i] = (HA_Close[i] >= HA_Open[i]) ? 0 : 1;
    }

     for(int i = heikin_ma_period - 1; i < rates_total; i++)
     {

      double sum = 0.0;
      for(int j = 0; j < heikin_ma_period; j++)
        {
         sum += HA_Close[i - j];
        }

      Heikin_MA_Buffer[i] = sum / heikin_ma_period;

     }

    return rates_total;
}
```

**Explanation:**

We must first modify the buffer and plot settings to incorporate the Heikin Ashi Moving Average (HA MA) into the indicator. Using #property indicator\_buffers 6, the buffer count is raised from 5 to 6, guaranteeing that there is a spare buffer to hold the Heikin MA data. The Heikin Ashi candles and the Heikin MA can both be seen on the chart by using #property indicator\_plots 2 to alter the plot count from 1 to 2. This guarantees that the indicator can effectively handle both data sets.

The Heikin MA plot's properties are then configured. #property indicator\_label2 is the label. By giving the Moving Average a name, "Heikin MA" makes it noticeable in the list of indicators. To define that the Moving Average would be displayed as a line instead of candlesticks, the type is set using #property indicator\_type2 DRAW\_LINE. We set #property indicator\_style2 STYLE\_DASH to make the line dashed and use #property indicator\_width2 1 to determine its width to increase visibility. Using #property indicator\_color2 clrBrown, the color is set to brown, guaranteeing a dramatic contrast with the Heikin Ashi candles.

After setting the properties, we define an input parameter int input heikin\_ma\_period = 20; that lets users change the period and declare an array double Heikin\_MA\_Buffer\[\]; to hold the Moving Average data. SetIndexBuffer(5, Heikin\_MA\_Buffer, INDICATOR\_DATA); is used to link the buffer to the indicator, allowing MetaTrader 5 to manage and show the values appropriately. PlotIndexSetInteger(5, PLOT\_DRAW\_BEGIN, heikin\_ma\_period); also guarantees that the Moving Average plots only after there are sufficient bars available.

Calculating the Heikin Ashi Moving Average is the last step. To ensure that we have enough data points for the calculation, the loop starts at heikin\_ma\_period -1. The Moving Average is calculated within the loop by adding together the last n Heikin Ashi Close values and dividing the total by the period. The indicator then plots the Moving Average alongside the Heikin Ashi candles after storing the result in Heikin\_MA\_Buffer\[i\], giving traders a smoothed trend-following tool.

![Figure 7. HA Indicator and HA MA](https://c.mql5.com/2/121/Figure_1__2.png)

### 3\. Integrating a Custom Indicator into an Expert Advisor

You may be asking, now that we have created a custom indicator, how we can use it to create an Expert Advisor. Using the Heikin Ashi indication that was created in the previous chapter, we will continue our project-based approach in this chapter and design an EA. Our technique will be realized as a working trading bot thanks to this EA's use of Heikin Ashi signals to automate trading decisions.

A simple crossover strategy based on the Heikin Ashi Moving Average (HA MA) and Heikin Ashi (HA) candles will be our trading strategy. This technique helps identify potential trend reversals and continuation signs by looking at the relationship between the Heikin Ashi candle closure price and the Heikin Ashi Moving Average. An upward trend and potential buying opportunity are indicated by a Heikin Ashi candle closing above the Heikin Ashi Moving Average. Conversely, a Heikin Ashi candle closing below the Heikin Ashi Moving Average generates a sell signal, suggesting a bearish trend reversal and a possible selling opportunity.

![Figure 8. Buy and Sell Logic](https://c.mql5.com/2/121/figure_3.png)

**3.1. Retrieving Indicator Data**

The first thing you should think about when integrating a custom indicator into an EA is how to input the indicator data into your EA. The EA is unable to base trade choices on the indicator if its values are inaccessible. For instance, the four values that comprise a Heikin Ashi candle must be imported into our Heikin Ashi Moving Average crossover strategy:

- HA Open price
- HA High price
- HA Low price
- HA Close price

Our trading signals are based on these parameters, which will be used to assess whether a Heikin Ashi candle has crossed above or below the HA MA.

**Example:**

```
// Declare arrays to store Heikin Ashi data from the custom indicator
double heikin_open[];   // Stores Heikin Ashi Open prices
double heikin_close[];  // Stores Heikin Ashi Close prices
double heikin_low[];    // Stores Heikin Ashi Low prices
double heikin_high[];   // Stores Heikin Ashi High prices
double heikin_ma[];     // Stores Heikin Ashi Moving Average values

int heikin_handle;      // Handle for the custom Heikin Ashi indicator

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
// Ensure arrays store data in a time series format (most recent data first)
   ArraySetAsSeries(heikin_open, true);
   ArraySetAsSeries(heikin_close, true);
   ArraySetAsSeries(heikin_low, true);
   ArraySetAsSeries(heikin_high, true);
   ArraySetAsSeries(heikin_ma, true);

// Load the custom Heikin Ashi indicator and get its handle
   heikin_handle = iCustom(_Symbol, PERIOD_CURRENT, "Project7 Heikin Ashi Indicator.ex5");

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// Nothing to clean up in this case, but can be used for resource management if needed
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Copy the latest 3 values of each buffer from the Heikin Ashi indicator
   CopyBuffer(heikin_handle, 0, 0, 3, heikin_open);  // Get HA Open values
   CopyBuffer(heikin_handle, 1, 0, 3, heikin_high);  // Get HA High values
   CopyBuffer(heikin_handle, 2, 0, 3, heikin_low);   // Get HA Low values
   CopyBuffer(heikin_handle, 3, 0, 3, heikin_close); // Get HA Close values
   CopyBuffer(heikin_handle, 5, 0, 3, heikin_ma);    // Get HA Moving Average values

// Print index 0 values to the terminal
   Print("HA Open: ", heikin_open[0],
         "\nHA High: ", heikin_high[0],
         "\nHA Low: ", heikin_low[0],
         "\nHA Close: ", heikin_close[0],
         "\nHA MA: ", heikin_ma[0]);
  }
```

**Explanation:**

**Declaring Arrays to Store Heikin Ashi Data**

Declaring arrays to hold the data collected from the indicator is the first step in integrating a custom indicator into an Expert Advisor (EA). Five arrays are declared in this instance to store the various Heikin Ashi indicator components:

- heikin\_open\[\]: Stores the Heikin Ashi Open prices.
- heikin\_close\[\]: For the Heikin Ashi Close prices.
- heikin\_low\[\]: Stores the Heikin Ashi Low prices.
- heikin\_high\[\]: Heikin Ashi High prices.
- heikin\_ma\[\]: Stores the Heikin Ashi Moving Average values.

Because they give the EA access to both past and current Heikin Ashi values, these arrays are crucial. The EA may evaluate historical price movements and adjust trading decisions by storing this data. These arrays don't need to be initialized with fixed values since they will be populated with data from the custom indicator; instead, they will be updated dynamically as new data becomes available.

**Declaring the Indicator Handle**

The handle of the customized Heikin Ashi indicator is stored in the variable heikin\_handle. A handle in MQL5 is a distinct reference to a background-running indicator instance. Because it enables the EA to interact with the indicator and request data as needed, this handle is essential. The EA wouldn't have access to the Heikin Ashi values without a handle. Later, when the iCustom() function is run, a value will be set to the handle. The indicator was not loaded correctly if the handle is invalid (returns -1), which prevents the EA from retrieving the necessary data.

**Array Initialization in Time Series Format**

The arrays must be constructed so that the most recent data is always at index 0 when they are declared. The ArraySetAsSeries() function is used for this, arranging array components in descending order such that the most recent data is kept first.

The function is applied to each of the five arrays as follows:

```
 ArraySetAsSeries(heikin_open, true);
 ArraySetAsSeries(heikin_close, true);
 ArraySetAsSeries(heikin_low, true);
 ArraySetAsSeries(heikin_high, true);
 ArraySetAsSeries(heikin_ma, true);
```

By converting these arrays to a time series format, index 0 allows the EA to always access the most recent Heikin Ashi data. When putting trading techniques into practice, this is especially helpful because it guarantees that the EA responds to current market movements rather than historical data.

**Loading the Custom Heikin Ashi Indicator with iCustom()**

By establishing, We require a means of accessing the data to incorporate a custom indicator into an Expert Advisor (EA). We can load a custom indicator and receive a handle that the EA may use to request indicator values using the iCustom() function. The EA wouldn't be able to access the indicator's data without this handle.

```
heikin_handle = iCustom(_Symbol, PERIOD_CURRENT, "Project7 Heikin Ashi Indicator.ex5");
```

- \_Symbol: This tells the function to apply the indicator to the current trading symbol (e.g., EUR/USD, GBP/JPY, etc.). This ensures that the indicator processes data for the same asset the EA is running on.
- PERIOD\_CURRENT: This applies the indicator to the same timeframe as the EA. If the EA is running on an H1 chart, the indicator will also be applied to H1.
- "Project7 Heikin Ashi Indicator.ex5": This specifies the filename of the custom indicator that the EA should use. The .ex5 extension indicates that this is a compiled MQL5 indicator file.

Making sure the indicator file is kept in the appropriate MetaTrader 5 directory is essential. The indicator needs to be in the MQL5 directory's Indicators folder. This directory's whole path is:

**MQL5/Indicators/**

The main function that connects a custom indicator to an EA is iCustom(), to sum up. It gives the EA a handle that enables dynamic extraction of indicator values. The function will not function properly unless the indicator is appropriately placed in the Indicators directory (or a subfolder inside it).

**Copying Indicator Data into the Arrays**

After obtaining the handle, the EA can retrieve the most recent Heikin Ashi values from the indicator using the CopyBuffer() function. Data is copied into the EA's arrays from the internal buffers of an indicator using the CopyBuffer() function. This EA calls the method five times, one for every piece of data:

```
CopyBuffer(heikin_handle, 0, 0, 3, heikin_open);  // Get HA Open values
CopyBuffer(heikin_handle, 1, 0, 3, heikin_high);  // Get HA High values
CopyBuffer(heikin_handle, 2, 0, 3, heikin_low);   // Get HA Low values
CopyBuffer(heikin_handle, 3, 0, 3, heikin_close); // Get HA Close values
CopyBuffer(heikin_handle, 5, 0, 3, heikin_ma);    // Get HA Moving Average values
```

Data is retrieved from the custom indicator using the same structure for every call to CopyBuffer(). The handle retrieved from the iCustom() function is the first parameter, heikin\_handle. The EA can access the custom Heikin Ashi indicator's data by using this handle as a reference. The EA couldn't request indicator values without this handle. The buffer indexes for the various Heikin Ashi indicator components are represented by the following set of parameters (0, 1, 2, 3, 5). In MQL5, indicators use buffers to hold their data, and each buffer is given a unique index. The Heikin Ashi Open price in this instance is represented by buffer 0, the High by buffer 1, the Low by buffer 2, the Close by buffer 3, and the Heikin Ashi Moving Average by buffer 5. We guarantee that the right data is obtained from the indicator by providing these indexes.

Data should be duplicated beginning with the most recent bar (current candle), according to the third option, 0. This guarantees that the EA is always using the most recent market data, which is necessary for trading decisions made in real time. Three data points are to be replicated, according to the fourth parameter, 3. Multiple value retrieval enables the EA to examine both historical and current Heikin Ashi data, which is helpful for discovering patterns or validating trends.

Lastly, the associated arrays — heikin\_open, heikin\_high, heikin\_low, heikin\_close, and heikin\_ma — are used to hold the recovered data. The retrieved values are stored in these arrays so that the EA may process and utilize them in its trading logic. For the EA to make well-informed trading decisions based on price trends, it is imperative that it has current information on the Heikin Ashi indicator.

**3.2. Utilizing Indicator Data in Our Trading Strategy**

After learning how to extract data from the custom indicator, we must now comprehend how to apply this data to our trading strategy. This will enable us to successfully use the Heikin Ashi indicator in an EA and show how to base trading decisions on the numbers that are retrieved. Furthermore, this procedure offers a chance to investigate certain important MQL5 ideas, which enhances our comprehension as we move through this text.

We cannot, however, simply apply the reasoning without considering crucial elements of the EA. Proper commerce administration requires the incorporation of specific features. These include keeping tabs on the quantity of open positions, putting trailing stops in place, controlling risk parameters, and making sure the EA executes trades in a methodical manner. By including these essential elements, we build a more reliable and useful trading system that can function well in a real-world setting.

**3.2.1. Tracking Open Buy and Sell Positions in the Expert Advisor**

We must make sure that just one buy and sell position are open at the same time in order for the trading technique to work properly. This reduces needless risk from many positions in the same direction and aids in maintaining control over position execution. The EA can decide whether to establish a new position or wait for an existing one to close before executing another by monitoring the number of open positions. This method guarantees that the strategy works as planned and enhances position management.

**Example:**

```
int totalPositions = 0;
int position_type_buy = 0;
int position_type_sell = 0;

for(int i = 0; i < PositionsTotal(); i++)
  {
   ulong ticket = PositionGetTicket(i);

   if(PositionSelectByTicket(ticket))
     {
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetString(POSITION_SYMBOL) == ChartSymbol(ChartID())
     {
      totalPositions++;
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
           {
            position_type_buy++;
           }
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
           {
            position_type_sell++;
           }
        }
     }
  }
```

The EA counts the number of buy, sell, and total trades to keep track of open positions. It iterates through open positions, classifies them as buy or sell, and uses the magic number and the chart symbol to confirm that they align with the strategy. By permitting only one purchase and one sell trade at a time, this guarantees controlled position management.

**3.2.2. Tracking Daily Trade Limits in the Expert Advisor**

The sum of trades made during a given trading session is the main subject of this section. The Expert Advisor makes sure the strategy doesn't go over the permitted daily trading limit by establishing a predetermined time period and tallying buy and sell trades. By avoiding over trading during the allotted time, this strategy aids in maintaining controlled trading behavior.

**Example:**

```
input int daily_trades = 6; // Total Daily Trades

// Start and end time for
string start = "00:00";
string end = "23:50";

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

   datetime start_time = StringToTime(start);
   datetime end_time = StringToTime(end);
   bool success = HistorySelect(start_time, end_time);

// Getting total trades
   int totalDeal = 0;
   int deal_type_buy = 0;
   int deal_type_sell = 0;
   if(success)
     {
      for(int i = 0; i < HistoryDealsTotal(); i++)
        {
         ulong ticket = HistoryDealGetTicket(i);

         if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == MagicNumber && HistoryDealGetString(ticket,DEAL_SYMBOL) == ChartSymbol(chart_id))
           {
            if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN)
              {
               totalDeal++;
               if(HistoryDealGetInteger(ticket, DEAL_TYPE) == DEAL_TYPE_BUY)
                 {

                  deal_type_buy++;

                 }
               if(HistoryDealGetInteger(ticket, DEAL_TYPE) == DEAL_TYPE_SELL)
                 {

                  deal_type_sell++;

                 }
              }
           }
        }
     }
  }
```

**Explanation:**

The Expert Advisor initially translates the specified start and ends times into datetime format to keep track of the total number of trades made during a given trading session. The HistorySelect() function is then used to choose the trading history that falls within this time frame. The EA sets up counters to record the total number of performed trades and the number of buy and sell deals independently if the selection is successful.

The EA then extracts the ticket number for each trade by iterating over the trade history. To make sure that only pertinent deals are counted, it verifies if the trade was opened using the same Magic Number and symbol as the current chart. The total number of trades is raised if the trade is determined to be an entering deal. After deciding whether the trade was placed, the EA updates the corresponding counters. This guarantees that the strategy monitors transactions that are executed and keeps the daily trading limit from being exceeded.

**3.2.3. Preventing Repeated Trades within the Same Candlestick**

In some cases, the EA may open a new trade immediately after closing an existing position if you want fresh trades to be completed only at the open price of a new bar. This is especially true if the trading conditions are still the same. Unintentional consecutive trades inside the same candlestick may result from this. We'll look at ways to stop this in this part, including making sure that deals are only executed at the following bar's open price, improving trade control, and preventing pointless reentry.

**Example:**

```
// Declare an array to store time data
datetime time[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
// Ensure the time array is structured as a time series (most recent data first)
   ArraySetAsSeries(time, true);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Retrieve the latest three time values from the current symbol and timeframe
   CopyTime(_Symbol, PERIOD_CURRENT, 0, 3, time);

// Select trading history from the earliest recorded time to the current time
   bool trade_control = HistorySelect(time[0], TimeCurrent());

// Variable to count the number of closed trades
   int total_deals_out = 0;

// If the trading history selection is successful, process the data
   if(trade_control)
     {
      // Loop through all closed trades in the history
      for(int i = 0; i < HistoryDealsTotal(); i++)
        {
         // Get the ticket number of the historical trade
         ulong ticket = HistoryDealGetTicket(i);

         // Check if the trade matches the current EA's Magic Number and symbol
         if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == MagicNumber && HistoryDealGetString(ticket, DEAL_SYMBOL) == ChartSymbol(chart_id))
           {
            // If the trade was an exit trade, increment the counter
            if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
              {
               total_deals_out++;
              }
           }
        }
     }
  }
```

**Explanation:**

This feature stops the EA from opening a trade inside the same candlestick after closing one. Using CopyTime(), which yields the timestamps of the most recent bars, the procedure first retrieves the most recent time values from the chart. Next, from the first recorded bar time (time\[0\]) to the current market time (TimeCurrent()), the EA chooses the trade history within the timeframe.

To find closed trades that match the EA's magic number and symbol, it then iterates through the trading history. The total\_deals\_out counter is increased if a closed trade — more especially, an exit trade (DEAL\_ENTRY\_OUT)—is found. If a trade was recently closed within the same candlestick, this counter might assist you find out. To ensure that trades only execute at the start of a new bar and to avoid undesired immediate re-entry, the EA will refrain from initiating a new trade until a new candlestick begins.

**3.2.4. Implementing Risk Management and Trade Execution with Heikin Ashi**

To guarantee that buy and sell trades are executed in accordance with Heikin Ashi signals, we shall specify the requirements for trade execution in this section. Furthermore, by determining lot sizes according to a predetermined dollar risk per trade and a risk-to-reward ratio (RRR), we will integrate risk management. This strategy avoids undue risk exposure while preserving controlled trading.

**Example:**

```
input  double dollar_risk = 12.0; // How Many Dollars($) Per Trade?
input double RRR = 3;

double ask_price;
double lot_size;
double point_risk;
double take_profit;

// Variable to store the time of the last executed trade
datetime lastTradeBarTime = 0;

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

// Get the opening time of the current bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);

// Check conditions for opening a buy position
// Ensures Heikin Ashi candle crosses above the moving average
// Limits trades per day and prevents multiple trades in the same candlestick
   if(heikin_open[1] < heikin_ma[1] && heikin_close[1] > heikin_ma[1] &&
      deal_type_buy < (daily_trades / 2) && total_deals_out < 1 &&
      totalPositions < 1 && currentBarTime != lastTradeBarTime)
     {
      // Calculate risk in points (distance from entry to stop loss)
      point_risk = ask_price - heikin_low[1];

      // Calculate take profit based on risk-to-reward ratio (RRR)
      take_profit = ((ask_price - heikin_low[1]) * RRR) + ask_price;

      // Determine lot size based on the dollar risk per trade
      lot_size = CalculateLotSize(_Symbol, dollar_risk, point_risk);

      // Execute a buy trade
      trade.Buy(lot_size, _Symbol, ask_price, heikin_low[1], take_profit);

      // Store the current bar time to prevent multiple trades in the same candle
      lastTradeBarTime = currentBarTime;
     }

// Check conditions for opening a sell position
// Ensures Heikin Ashi candle crosses below the moving average
// Limits trades per day and prevents multiple trades in the same candlestick
   if(heikin_open[1] > heikin_ma[1] && heikin_close[1] < heikin_ma[1] &&
      deal_type_sell < (daily_trades / 2) && total_deals_out < 1 &&
      totalPositions < 1 && currentBarTime != lastTradeBarTime)
     {
      // Calculate risk in points (distance from entry to stop loss)
      point_risk = heikin_high[1] - ask_price;

      // Calculate take profit based on risk-to-reward ratio (RRR)
      take_profit = MathAbs(((heikin_high[1] - ask_price) * RRR) - ask_price);

      // Determine lot size based on the dollar risk per trade
      lot_size = CalculateLotSize(_Symbol, dollar_risk, point_risk);

      // Execute a sell trade
      trade.Sell(lot_size, _Symbol, ask_price, heikin_high[1], take_profit);

      // Store the current bar time to prevent multiple trades in the same candle
      lastTradeBarTime = currentBarTime;
     }
  }

//+------------------------------------------------------------------+
//| Function to calculate the lot size based on risk amount and stop loss
//+------------------------------------------------------------------+
double CalculateLotSize(string symbol, double riskAmount, double stopLossPips)
  {
// Get symbol information
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);

// Calculate pip value per lot
   double pipValuePerLot = tickValue / point;

// Calculate the stop loss value in currency
   double stopLossValue = stopLossPips * pipValuePerLot;

// Calculate the lot size
   double lotSize = riskAmount / stopLossValue;

// Round the lot size to the nearest acceptable lot step
   double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   lotSize = MathFloor(lotSize / lotStep) * lotStep;

   return lotSize;
  }
```

**Explanation:**

To validate trends, the trade execution logic depends on Heikin Ashi candles crossing a Moving Average (MA). A bullish reversal is indicated when the open price is below the MA and the close is above, and a sell when the opposite is true. The system restricts daily trades, makes sure there are no open positions, and only conducts trades on fresh candlesticks to stop excessive trading. Stop losses at prior Heikin Ashi highs or lows, profit-taking based on a Risk-to-Reward Ratio (RRR), and dynamic lot sizing are all components of risk management.

**3.2.5. Using Heikin Ashi Candles to Put in Place a Trailing Stop**

The implementation of a trailing stop mechanism using Heikin Ashi candles is the main objective of this section. It ensures that stop-loss levels are adjusted dynamically in reaction to changes in the Heikin Ashi price.

**Example:**

```
input bool    allow_trailing  = false; // Do you Allow Trailing Stop?

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

   // Check if trailing stop is enabled
   if(allow_trailing == true)
     {
      // Variables to store trade-related information
      double positionProfit = 0;
      double positionopen = 0;
      double positionTP = 0;
      double positionSL = 0;

      // Loop through all open positions
      for(int i = 0; i < PositionsTotal(); i++)
        {
         // Get the ticket number of the position
         ulong ticket = PositionGetTicket(i);

         // Select the position using its ticket number
         if(PositionSelectByTicket(ticket))
           {
            // Check if the position belongs to the EA by verifying the magic number and symbol
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetString(POSITION_SYMBOL) == ChartSymbol(chart_id))
              {
               // Retrieve trade details: open price, take profit, profit, and stop loss
               positionopen = PositionGetDouble(POSITION_PRICE_OPEN);
               positionTP = PositionGetDouble(POSITION_TP);
               positionProfit = PositionGetDouble(POSITION_PROFIT);
               positionSL = PositionGetDouble(POSITION_SL);

               // Apply trailing stop logic for buy positions
               if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                 {
                  // Adjust stop loss if Heikin Ashi low is above the entry price and the candle is bullish
                  if(heikin_low[1] > positionopen && heikin_close[1] > heikin_open[1])
                    {
                     trade.PositionModify(ticket, heikin_low[1], positionTP);
                    }
                 }

               // Apply trailing stop logic for sell positions
               if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
                 {
                  // Adjust stop loss if Heikin Ashi high is below the entry price and the candle is bearish
                  if(heikin_high[1] < positionopen && heikin_close[1] < heikin_open[1])
                    {
                     trade.PositionModify(ticket, heikin_high[1], positionTP);
                    }
                 }
              }
           }
        }
     }
  }
```

**Explanation:**

To secure profits and dynamically adjust stop-loss levels, this section uses Heikin Ashi candles to implement a trailing stop system. The allow\_trailing input variable controls whether the trailing stop feature is enabled; if it is set to true, the system loops through all open positions, retrieves their details, and verifies that they belong to the EA by checking the magic number and symbol. To make the trailing stop logic easier, key trade information is extracted, including open price, take profit, profit, and stop loss.

By verifying that the prior Heikin Ashi low is higher than the entry price and that the Heikin Ashi close is higher than its open, the system verifies open positions and confirms a bullish trend. To protect profits while permitting additional upward movement, the stop loss is set at the Heikin Ashi low if it is reached. By ensuring that the preceding Heikin Ashi high is below the entry price and that the Heikin Ashi close is below its open, it validates a downtrend for sell trades. To safeguard the trade and capture more market swings, the stop loss is set to the Heikin Ashi high if these conditions are met.

### **Conclusion**

In this article, we built a Heikin Ashi indicator from scratch, integrated it with a Moving Average that uses Heikin Ashi candle data, and explored how to incorporate a custom indicator into an EA, ensuring seamless integration for automated trading. We implemented risk management techniques such as limiting daily trades, preventing multiple closed trades within the same candle, and dynamically calculating lot sizes. Additionally, we introduced a trailing stop mechanism using Heikin Ashi candles to adjust stop-loss levels. These concepts provide a solid foundation for building and refining automated trading strategies in MQL5.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17296.zip "Download all attachments in the single ZIP archive")

[Project7\_Heikin\_Ashi\_Indicator.mq5](https://www.mql5.com/en/articles/download/17296/project7_heikin_ashi_indicator.mq5 "Download Project7_Heikin_Ashi_Indicator.mq5")(3.55 KB)

[Project7\_Heikin\_Ashi\_EA.mq5](https://www.mql5.com/en/articles/download/17296/project7_heikin_ashi_ea.mq5 "Download Project7_Heikin_Ashi_EA.mq5")(27.51 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Introduction to MQL5 (Part 36): Mastering API and WebRequest Function in MQL5 (X)](https://www.mql5.com/en/articles/20938)
- [Introduction to MQL5 (Part 35): Mastering API and WebRequest Function in MQL5 (IX)](https://www.mql5.com/en/articles/20859)
- [Introduction to MQL5 (Part 34): Mastering API and WebRequest Function in MQL5 (VIII)](https://www.mql5.com/en/articles/20802)
- [Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://www.mql5.com/en/articles/20700)
- [Introduction to MQL5 (Part 32): Mastering API and WebRequest Function in MQL5 (VI)](https://www.mql5.com/en/articles/20591)
- [Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://www.mql5.com/en/articles/20546)
- [Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://www.mql5.com/en/articles/20425)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/482205)**
(2)


![dhermanus](https://c.mql5.com/avatar/avatar_na2.png)

**[dhermanus](https://www.mql5.com/en/users/dhermanus)**
\|
31 May 2025 at 11:08

Hi Isreal,

Thanks for the blog, time and effort.

I would like ask you about your Heikin Ashi [custom indicator](https://www.mql5.com/en/articles/5 "Article: Step on New Rails: Custom Indicators in MQL5 ") code.

On the Heikin Ashi Open formula :

      // HEIKIN ASHI OPEN FORMULA

       HA\_Open\[i\] = (HA\_Open\[i - 1\] + HA\_Close\[i - 1\]) / 2.0;

Since you have not compute the HA\_Open\[i - 1). Wouldn't this be 0?\
\
My suggestion  :\
\
        if (i == 1){\
\
         HA\_Open\[i\] = (open\[i - 1\] + close\[i - 1\])/2.0;  // On HA first bar just use the normal open/close data\
\
        }\
\
        else{\
\
        // HEIKIN ASHI OPEN FORMULA\
\
        HA\_Open\[i\] = (HA\_Open\[i - 1\] + HA\_Close\[i - 1\]) / 2.0;\
\
        }\
\
![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)\
\
**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**\
\|\
31 May 2025 at 12:28\
\
**dhermanus [#](https://www.mql5.com/en/forum/482205#comment_56828411):**\
\
Hi Isreal,\
\
Thanks for the blog, time and effort.\
\
I would like ask you about your Heikin Ashi [custom indicator](https://www.mql5.com/en/articles/5 "Article: Step on New Rails: Custom Indicators in MQL5 ") code.\
\
On the Heikin Ashi Open formula :\
\
      // HEIKIN ASHI OPEN FORMULA\
\
       HA\_Open\[i\] = (HA\_Open\[i - 1\] + HA\_Close\[i - 1\]) / 2.0;\
\
Since you have not compute the HA\_Open\[i - 1). Wouldn't this be 0?\
\
My suggestion  :\
\
        if (i == 1){\
\
         HA\_Open\[i\] = (open\[i - 1\] + close\[i - 1\])/2.0;  // On HA first bar just use the normal open/close data\
\
        }\
\
        else{\
\
        // HEIKIN ASHI OPEN FORMULA\
\
        HA\_Open\[i\] = (HA\_Open\[i - 1\] + HA\_Close\[i - 1\]) / 2.0;\
\
        }\
\
Thank you. I’ll look into that\
\
\
![MQL5 Wizard Techniques you should know (Part 56): Bill Williams Fractals](https://c.mql5.com/2/122/MQL5_Wizard_Techniques_you_should_know_Part_56___LOGO.png)[MQL5 Wizard Techniques you should know (Part 56): Bill Williams Fractals](https://www.mql5.com/en/articles/17334)\
\
The Fractals by Bill Williams is a potent indicator that is easy to overlook when one initially spots it on a price chart. It appears too busy and probably not incisive enough. We aim to draw away this curtain on this indicator by examining what its various patterns could accomplish when examined with forward walk tests on all, with wizard assembled Expert Advisor.\
\
![Artificial Algae Algorithm (AAA)](https://c.mql5.com/2/89/logo-midjourney_image_15565_402_3881__3.png)[Artificial Algae Algorithm (AAA)](https://www.mql5.com/en/articles/15565)\
\
The article considers the Artificial Algae Algorithm (AAA) based on biological processes characteristic of microalgae. The algorithm includes spiral motion, evolutionary process and adaptation, which allows it to solve optimization problems. The article provides an in-depth analysis of the working principles of AAA and its potential in mathematical modeling, highlighting the connection between nature and algorithmic solutions.\
\
![Cycles and Forex](https://c.mql5.com/2/90/logo-midjourney_image_15614_405_3907_1.png)[Cycles and Forex](https://www.mql5.com/en/articles/15614)\
\
Cycles are of great importance in our lives. Day and night, seasons, days of the week and many other cycles of different nature are present in the life of any person. In this article, we will consider cycles in financial markets.\
\
![Neural Network in Practice: Sketching a Neuron](https://c.mql5.com/2/88/Neural_network_in_practice_Sketching_a_neuron___LOGO.png)[Neural Network in Practice: Sketching a Neuron](https://www.mql5.com/en/articles/13744)\
\
In this article we will build a basic neuron. And although it looks simple, and many may consider this code completely trivial and meaningless, I want you to have fun studying this simple sketch of a neuron. Don't be afraid to modify the code, understanding it fully is the goal.\
\
[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/17296&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049318296903330128)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).