---
title: Price Action Analysis Toolkit Development (Part 15): Introducing Quarters Theory (I) — Quarters Drawer Script
url: https://www.mql5.com/en/articles/17250
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:40:57.385414
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/17250&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069653330667440263)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/17250#para2)
- [Concept Overview](https://www.mql5.com/en/articles/17250#para3)
- [MQL5 Implementation](https://www.mql5.com/en/articles/17250#para4)
- [MQL5 Script](https://www.mql5.com/en/articles/17250#para5)
- [Outcomes](https://www.mql5.com/en/articles/17250#para6)
- [Conclusion](https://www.mql5.com/en/articles/17250#para7)

### Introduction

In every market, prices move in cycles. Whether prices are trending up or down, they repeatedly oscillate within defined ranges. Over time traders have developed many theories to explain these patterns. In our ongoing series, we are building a comprehensive price action analysis toolkit. Today we introduce an innovative approach that we have automated using MQL5 to simplify application and achieve impressive results.

The Quarters Theory is the focus of this article. It was developed by Ilian Yotov, a name familiar to thousands of currency traders and Forex strategists from his work on the Traders Television Network. Ilian founded AllThingsForex.com and TraderTape.com and hosts the popular daily show All Things Forex. His theory suggests that significant price moves occur between two Large Quarter Points and progress in increments of 250 PIPs. This method offers clear markers to identify key support and resistance levels while giving traders practical insights.

This article is the first part in our Quarters Theory. In this installment, we focus on constructing the quarters using our " [quarters drawer](https://www.mql5.com/en/articles/17250#para5)" script. By automating the drawing of these crucial levels, we offer a practical starting point for visualizing key reversal and continuation zones. As we advance in this theory, we will develop even more robust tools to support signal generation and advanced market analysis.

### Concept Overview

The Quarters Theory is a technical analysis approach that divides a significant price range into smaller, more meaningful segments. In this framework, a " _MajorStep_"—for example, 0.1000 in currency pairs like EUR/USD, defines the difference between major whole numbers (such as 1.2000 to 1.3000). This range is then subdivided into four equal parts, known as large quarters. Each large quarter represents a critical level where price may pause, reverse, or accelerate, offering traders potential support and resistance areas.

Key components of the Quarters Theory include

- Major Whole Numbers:These are the primary levels (e.g., 1.2000, 1.3000) that frame the trading range. They are used as reference points to build the finer structure of the theory.
- Large Quarter Lines:The interval between two major levels is divided equally into four segments. These lines indicate the intermediate levels that often play an important role in price behavior. Traders use these levels to anticipate potential turning points or areas of consolidation.
- Small Quarter Lines (Optional):For further precision, each 100-pip segment can be subdivided into even smaller intervals, small quarters. Although these lines offer additional granularity, the primary focus remains on the large quarter levels.
- Overshoot/Undershoot Areas:Surrounding each large quarter line, slight offsets (overshoots and undershoots) are drawn to signal zones where price might momentarily exceed or fall short of the expected level. These areas help in identifying potential corrections or reversals.

For further insights, please review the following diagrams

![Quarters](https://c.mql5.com/2/120/Q3.png)

Fig 1. The Quarters

![Quarters](https://c.mql5.com/2/120/Quarters_theory.png)

Fig 2. The QUATERS

By visually marking these key levels on a chart, the Quarters Theory provides traders with a structured way to assess price dynamics. The approach is implemented in the script “ [Quarters Drawer](https://www.mql5.com/en/articles/17250#para5),” which draws distinct lines with varying styles and colors to enhance clarity, ensuring that each group (major, large quarter, and small quarter lines) is easily distinguishable. This visual tool reinforces the theory by clearly highlighting the crucial price segments.

### MQL5 Implementation

The first block of the script includes a header comment that provides essential details such as the script’s name, version, copyright information, and a link to the [author’s profile](https://www.mql5.com/en/users/lynnchris). Following this, property declarations embed important metadata (like copyright and version) and enforce strict compilation settings. This combination establishes the script's identity and ensures it adheres to robust coding standards.

```
//+------------------------------------------------------------------+
//|                                              Quarters Drawer.mql5|
//|                                Copyright 2025, Christian Benjamin|
//|                           https://www.mql5.com/en/users/lynnchris|
//+------------------------------------------------------------------+
#property copyright "Christian Benjamin"
#property link      "https://www.mql5.com/en/users/lynnchris"
#property version   "1.0"
#property strict
```

Next, the input parameters are defined to give users complete control over the script’s behavior and appearance. Here, you set the interval between major price levels with a parameter like _MajorStep_, and use _boolean_ flags to toggle the drawing of large quarter lines, small quarter lines, and overshoot/undershoot markers. For instance, if _DrawLargeQuarters_ is set to true, the script will draw large quarter lines, but switching it to false will prevent them from appearing.

The same applies to _DrawSmallQuarters_ and _DrawOvershootAreas_, if you don’t want to see small quarter lines or overshoot zones, simply toggle their values to false. This makes it easy to customize the display without modifying the core logic. Moreover, color settings and line style/thickness options are provided for each type of line, allowing traders to easily customize the visual output to match their personal preferences and trading strategies.

```
//---- Input parameters -------------------------------------------------
input double MajorStep          = 0.1000;   // Difference between major whole numbers.
input bool   DrawLargeQuarters  = true;     // Draw intermediate large quarter lines.
input bool   DrawSmallQuarters  = true;     // Draw small quarter lines.
input bool   DrawOvershootAreas = true;     // Mark overshoot/undershoot areas for large quarter lines.

//---- Color settings ---------------------------------------------------
input color  MajorColor         = 0x2F4F4F; // Dark Slate Gray for major lines.
input color  LargeQuarterColor  = 0x8B0000; // Dark Red for large quarter lines.
input color  SmallQuarterColor  = 0x00008B; // Dark Blue for small quarter lines.
input color  OvershootColor     = clrRed;   // Red for overshoot/undershoot lines.

//---- Group style and thickness settings -------------------------------
input ENUM_LINE_STYLE MajorLineStyle       = STYLE_SOLID;
input int    MajorLineWidth                 = 4;
input ENUM_LINE_STYLE LargeQuarterLineStyle  = STYLE_DOT;
input int    LargeQuarterLineWidth          = 3;
input ENUM_LINE_STYLE OvershootLineStyle     = STYLE_DASH;
input int    OvershootLineWidth             = 1;
input ENUM_LINE_STYLE SmallQuarterLineStyle  = STYLE_SOLID;
input int    SmallQuarterLineWidth          = 1;
```

When you see hexadecimal color codes in the script, they’re representing specific colors by combining red, green, and blue values. For example:

- 0x2F4F4F – This code corresponds to Dark Slate Gray. The two-digit pairs represent the red, green, and blue components respectively. Here, “2F” (47 in decimal) is for red, and “4F” (79 in decimal) for both green and blue, creating a muted, cool gray tone that's ideal for primary lines on a chart.
- 0x8B0000 – This code represents Dark Red. In this value, “8B” (139 in decimal) is a strong red component, while both the green and blue components are “00” (zero), resulting in a deep, intense red. This color is used in the script to emphasize large quarter lines.
- 0x00008B – This code signifies Dark Blue. In this case, the red and green components are “00”, and the blue component is “8B” (139 in decimal), yielding a deep blue color. This is applied to small quarter lines, providing a distinct visual contrast.

In addition to these hard-coded hexadecimal values, the script also uses a predefined constant called _clrRed_. Instead of specifying a hex code manually, you can simply use this built-in constant, which typically represents the color red with the RGB values (255, 0, 0). For instance, the script sets the overshoot color like this:

```
input color OvershootColor = clrRed;
```

Following the inputs, the script features a dedicated function that handles drawing horizontal lines on the chart. This function first checks whether a line with the specified name already exists, removing it if necessary to avoid duplicates. It then creates a new horizontal line at a specified price level and applies the defined color, style, and width. By ensuring that each line extends across the chart, this modular function simplifies the process of drawing multiple lines consistently throughout the script.

```
//+------------------------------------------------------------------+
//| DrawHorizontalLine: Creates or replaces a horizontal line        |
//+------------------------------------------------------------------+
void DrawHorizontalLine(string name, double price, color lineColor, int width, ENUM_LINE_STYLE style)
{
   // Delete the object if it already exists
   if(ObjectFind(0, name) != -1)
      ObjectDelete(0, name);

   // Create a new horizontal line object
   if(!ObjectCreate(0, name, OBJ_HLINE, 0, 0, price))
   {
      Print("Failed to create line: ", name);
      return;
   }

   // Set properties: color, style, width, and extension to the right
   ObjectSetInteger(0, name, OBJPROP_COLOR, lineColor);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, true);
}
```

The script’s main execution begins with the _OnStart_ function, where it retrieves the current symbol and its bid price. This step is crucial, as all subsequent calculations depend on a valid market price. If the bid price isn’t available, the script exits early to prevent any errors, ensuring that further operations are only performed with valid data.

```
void OnStart()
{
   // Get current symbol price
   string symbol = _Symbol;
   double currentPrice = SymbolInfoDouble(symbol, SYMBOL_BID);
   if(currentPrice == 0)
      return;
```

Once the current price is obtained, the script calculates the major price levels that will serve as the primary reference points. It determines the lower level by rounding down the current price to the nearest interval defined by _MajorStep_, and then computes the upper level by adding that same step. These calculated levels create a structured framework, setting clear boundaries for the rest of the visual grid.

```
   // Calculate the major range based on the current price and MajorStep
   double lowerMajor = MathFloor(currentPrice / MajorStep) * MajorStep;
   double upperMajor = lowerMajor + MajorStep;
```

Using these major price levels, the script then draws two prominent horizontal lines at the lower and upper boundaries. These major lines are styled with specific colors, thicknesses, and line patterns, making them easily identifiable on the chart. They act as the backbone for the chart’s visual structure, helping traders quickly spot significant support and resistance areas.

```
   // Draw Major Whole Number lines at lower and upper boundaries
   DrawHorizontalLine("MajorLower", lowerMajor, MajorColor, MajorLineWidth, MajorLineStyle);
   DrawHorizontalLine("MajorUpper", upperMajor, MajorColor, MajorLineWidth, MajorLineStyle);
```

If the option to draw large quarter lines is enabled, the script subdivides the major interval into four equal parts. It calculates intermediate levels between the major boundaries and draws horizontal lines at these points. Furthermore, if overshoot/undershoot markers are also activated, additional lines are drawn just above and below each large quarter line. These extra markers serve to highlight areas where the price might temporarily extend beyond its expected range, offering traders valuable insights into potential price reversals.

```
   // Draw Large Quarter lines and overshoot/undershoot lines if enabled
   if(DrawLargeQuarters)
   {
      double LQIncrement = MajorStep / 4.0;
      for(int i = 1; i < 4; i++)
      {
         double level = lowerMajor + i * LQIncrement;
         string objName = "LargeQuarter_" + IntegerToString(i);
         DrawHorizontalLine(objName, level, LargeQuarterColor, LargeQuarterLineWidth, LargeQuarterLineStyle);

         if(DrawOvershootAreas)
         {
            double smallQuarter = MajorStep / 40.0;
            DrawHorizontalLine("Overshoot_" + IntegerToString(i) + "_up", level + smallQuarter, OvershootColor, OvershootLineWidth, OvershootLineStyle);
            DrawHorizontalLine("Undershoot_" + IntegerToString(i) + "_down", level - smallQuarter, OvershootColor, OvershootLineWidth, OvershootLineStyle);
         }
      }
   }
```

For even finer detail, the script can also draw small quarter lines when that option is enabled. It divides the major interval into ten segments, and then further splits each segment into smaller subdivisions. This creates a more granular grid that captures subtle price movements, providing traders with a detailed view that can be particularly useful for identifying precise entry and exit points.

```
   // Draw Small Quarter lines if enabled
   if(DrawSmallQuarters)
   {
      double segStep = MajorStep / 10.0;
      double smallQuarter = segStep / 4.0;
      for(int seg = 0; seg < 10; seg++)
      {
         double segStart = lowerMajor + seg * segStep;
         for(int j = 1; j < 4; j++)
         {
            double level = segStart + j * smallQuarter;
            string objName = "SmallQuarter_" + IntegerToString(seg) + "_" + IntegerToString(j);
            DrawHorizontalLine(objName, level, SmallQuarterColor, SmallQuarterLineWidth, SmallQuarterLineStyle);
         }
      }
   }
```

To round off the visual enhancements, the script adds a persistent label, or logo, to the chart. This label displays the script’s name, version, and author information, acting as both a credit and a quick reference for users. Before placing the label, the script checks for any existing instance to avoid duplicates, then positions the label in the top-right corner with carefully chosen font size and color settings for clear visibility.

```
   // Create a persistent label (logo) in the top-right corner
   if(ObjectFind(0, "ScriptLogo") != -1)
      ObjectDelete(0, "ScriptLogo");
   if(ObjectCreate(0, "ScriptLogo", OBJ_LABEL, 0, 0, 0))
   {
      string logoText = "Script: DrawQuarters_DarkBold\nv1.04\nby Christian Benjamin";
      ObjectSetString(0, "ScriptLogo", OBJPROP_TEXT, logoText);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_COLOR, clrYellow);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_FONTSIZE, 14);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_XDISTANCE, 80);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_YDISTANCE, 10);
   }
```

Finally, the script refreshes the chart with a redraw command, ensuring that all the drawn objects (lines and labels) are immediately visible. This final step is crucial as it updates the display to reflect the latest market data and custom settings, presenting a complete and current visual tool for technical analysis.

```
   // Redraw the chart to display all objects
   ChartRedraw();
}
```

MQL5 Code

```
//+------------------------------------------------------------------+
//|                                              Quarters Drawer.mql5|
//|                                Copyright 2025, Christian Benjamin|
//|                           https://www.mql5.com/en/users/lynnchris|
//+------------------------------------------------------------------+
#property copyright "Christian Benjamin"
#property link      "https://www.mql5.com/en/users/lynnchris"
#property version   "1.0"
#property strict

//---- Input parameters -------------------------------------------------
input double MajorStep          = 0.1000;   // Difference between major whole numbers.
input bool   DrawLargeQuarters  = true;     // Draw intermediate large quarter lines.
input bool   DrawSmallQuarters  = true;    // Draw small quarter lines.
input bool   DrawOvershootAreas = true;     // Mark overshoot/undershoot areas for large quarter lines.

//---- Color settings ---------------------------------------------------
input color  MajorColor         = 0x2F4F4F; // Dark Slate Gray for major lines.
input color  LargeQuarterColor  = 0x8B0000; // Dark Red for large quarter lines.
input color  SmallQuarterColor  = 0x00008B; // Dark Blue for small quarter lines.
input color  OvershootColor     = clrRed;   // Red for overshoot/undershoot lines.

//---- Group style and thickness settings -------------------------------
input ENUM_LINE_STYLE MajorLineStyle       = STYLE_SOLID;
input int    MajorLineWidth                 = 4;
input ENUM_LINE_STYLE LargeQuarterLineStyle  = STYLE_DOT;
input int    LargeQuarterLineWidth          = 3;
input ENUM_LINE_STYLE OvershootLineStyle     = STYLE_DASH;
input int    OvershootLineWidth             = 1;
// For small quarter lines, we now use a continuous (solid) style.
input ENUM_LINE_STYLE SmallQuarterLineStyle  = STYLE_SOLID;
input int    SmallQuarterLineWidth          = 1;

//+------------------------------------------------------------------+
//| DrawHorizontalLine: Creates or replaces a horizontal line        |
//+------------------------------------------------------------------+
void DrawHorizontalLine(string name, double price, color lineColor, int width, ENUM_LINE_STYLE style)
  {
   if(ObjectFind(0, name) != -1)
      ObjectDelete(0, name);

   if(!ObjectCreate(0, name, OBJ_HLINE, 0, 0, price))
     {
      Print("Failed to create line: ", name);
      return;
     }

   ObjectSetInteger(0, name, OBJPROP_COLOR, lineColor);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, true);
  }

//+------------------------------------------------------------------+
//| Script entry point                                               |
//+------------------------------------------------------------------+
void OnStart()
  {
// Get current symbol price
   string symbol = _Symbol;
   double currentPrice = SymbolInfoDouble(symbol, SYMBOL_BID);
   if(currentPrice == 0)
      return;

// Calculate the major range
   double lowerMajor = MathFloor(currentPrice / MajorStep) * MajorStep;
   double upperMajor = lowerMajor + MajorStep;

// Draw Major Whole Number lines
   DrawHorizontalLine("MajorLower", lowerMajor, MajorColor, MajorLineWidth, MajorLineStyle);
   DrawHorizontalLine("MajorUpper", upperMajor, MajorColor, MajorLineWidth, MajorLineStyle);

// Draw Large Quarter lines and their overshoot/undershoot lines
   if(DrawLargeQuarters)
     {
      double LQIncrement = MajorStep / 4.0;
      for(int i = 1; i < 4; i++)
        {
         double level = lowerMajor + i * LQIncrement;
         string objName = "LargeQuarter_" + IntegerToString(i);
         DrawHorizontalLine(objName, level, LargeQuarterColor, LargeQuarterLineWidth, LargeQuarterLineStyle);

         if(DrawOvershootAreas)
           {
            double smallQuarter = MajorStep / 40.0;
            DrawHorizontalLine("Overshoot_" + IntegerToString(i) + "_up", level + smallQuarter, OvershootColor, OvershootLineWidth, OvershootLineStyle);
            DrawHorizontalLine("Undershoot_" + IntegerToString(i) + "_down", level - smallQuarter, OvershootColor, OvershootLineWidth, OvershootLineStyle);
           }
        }
     }

// Draw Small Quarter lines if enabled (continuous lines, without overshoot/undershoot)
   if(DrawSmallQuarters)
     {
      double segStep = MajorStep / 10.0;
      double smallQuarter = segStep / 4.0;
      for(int seg = 0; seg < 10; seg++)
        {
         double segStart = lowerMajor + seg * segStep;
         for(int j = 1; j < 4; j++)
           {
            double level = segStart + j * smallQuarter;
            string objName = "SmallQuarter_" + IntegerToString(seg) + "_" + IntegerToString(j);
            DrawHorizontalLine(objName, level, SmallQuarterColor, SmallQuarterLineWidth, SmallQuarterLineStyle);
           }
        }
     }

// Create a persistent label (logo) in the top-right corner
   if(ObjectFind(0, "ScriptLogo") != -1)
      ObjectDelete(0, "ScriptLogo");
   if(ObjectCreate(0, "ScriptLogo", OBJ_LABEL, 0, 0, 0))
     {
      string logoText = "Script: DrawQuarters_DarkBold\nv1.04\nby Christian Benjamin";
      ObjectSetString(0, "ScriptLogo", OBJPROP_TEXT, logoText);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_COLOR, clrYellow);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_FONTSIZE, 14);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_XDISTANCE, 80);
      ObjectSetInteger(0, "ScriptLogo", OBJPROP_YDISTANCE, 10);
     }

   ChartRedraw();
  }
//+------------------------------------------------------------------+
```

### Outcomes

Before exploring the outcomes, here is how to create and compile the script. Open MetaEditor and select "New" then choose "Script." Enter a name for your script and start writing your code. Compile the script. If errors appear, fix them until the compilation is successful. Once compiled, test your script on a chart. Since it is not an EA, you can run it on a live chart in either demo or live mode without affecting your balance.

In this section, I have several diagrams for visual understanding. I will guide you through each diagram step by step. The first diagram shows a test on the AUD/USD pair. In that test, I enabled _DrawLargeQuarters_ along with overshoot and undershoot areas and set DrawSmallQuarters to false.

```
//---- Input parameters -------------------------------------------------
input double MajorStep          = 0.1000;   // Difference between major whole numbers.
input bool   DrawLargeQuarters  = true;     // Draw intermediate large quarter lines.
input bool   DrawSmallQuarters  = false;    // Draw small quarter lines.
input bool   DrawOvershootAreas = true;     // Mark overshoot/undershoot areas for large quarter lines.
```

Let's review Figure 2 below. It shows how I added the script and the outcomes recorded. We see how the price interacts with the Quarters and their overshoot and undershoot.

![Quartile Drawer](https://c.mql5.com/2/120/Quartile_Drawer.gif)

Fig 2. AUDUSD Quarters

Below is a screenshot captured on the same pair. I have highlighted the effect of the quarters on the chart. The results are clear: the market finds support and resistance at our larger quarter levels. I set the overshoot lines and smaller quartile options to false to clearly visualize these large quarters.

```
//---- Input parameters -------------------------------------------------
input double MajorStep          = 0.1000;   // Difference between major whole numbers.
input bool   DrawLargeQuarters  = true;     // Draw intermediate large quarter lines.
input bool   DrawSmallQuarters  = false;    // Draw small quarter lines.
input bool   DrawOvershootAreas = false;     // Mark overshoot/undershoot areas for large quarter lines.
```

See Figure 3 below

![Large Quarters](https://c.mql5.com/2/120/Large_Quarters.png)

Fig 3. Large Quarters

I have zoomed in further so you can clearly see how the price interacts with the large quarter line, including its undershoots and overshoots.

![Quartile Shoots](https://c.mql5.com/2/120/Quartiles_Shoots.png)

Fig 4. Overshoot and Undershoot

Lastly, let's examine the script's performance on EUR/USD. Key levels that every FX trader hunts for are clearly identified by the quarter levels.

![EURUSD](https://c.mql5.com/2/120/EURUSD_Quartiles_Drawer.gif)

Fig 5. EURUSD Quarters

Please refer to the book:  [The\_Quarters\_Theory\_-\_Ilian\_Yotov](https://www.mql5.com/go?link=https://scholar.google.com/scholar?q=quarters+theory+strategy%26hl=en%26as_sdt=0%26as_vis=1%26oi=scholart "https://scholar.google.com/scholar?q=quarters+theory+strategy&hl=en&as_sdt=0&as_vis=1&oi=scholart")

### Conclusion

I am sure you can all see the impact of this script. It draws the quarters and does the calculations that would be time-consuming to do manually. A notable aspect is how the quartile lines serve as support and reversal levels. The large quartiles, the major lower and upper levels, act as significant resistance and support, while the smaller quartiles serve as minor levels. This initial approach is a strong starting point for automated analysis. We look forward to developing more tools as we dive deeper into this theory. If you have any suggestions or recommendations, please share them.

| Date | Tool Name | Description | Version | Updates | Notes |
| --- | --- | --- | --- | --- | --- |
| 01/10/24 | [Chart Projector](https://www.mql5.com/en/articles/16014) | Script to overlay the previous day's price action with ghost effect. | 1.0 | Initial Release | Tool number 1 |
| 18/11/24 | [Analytical Comment](https://www.mql5.com/en/articles/15927) | It provides previous day's information in a tabular format, as well as anticipates the future direction of the market. | 1.0 | Initial Release | Tool number 2 |
| 27/11/24 | [Analytics Master](https://www.mql5.com/en/articles/16434) | Regular Update of market metrics after every two hours | 1.01 | Second Release | Tool number 3 |
| 02/12/24 | [Analytics Forecaster](https://www.mql5.com/en/articles/16559) | Regular Update of market metrics after every two hours with telegram integration | 1.1 | Third Edition | Tool number 4 |
| 09/12/24 | [Volatility Navigator](https://www.mql5.com/en/articles/16560) | The EA analyzes market conditions using the Bollinger Bands, RSI and ATR indicators | 1.0 | Initial Release | Tool Number 5 |
| 19/12/24 | [Mean Reversion Signal Reaper](https://www.mql5.com/en/articles/16700) | Analyzes market using mean reversion strategy and provides signal | 1.0 | Initial Release | Tool number 6 |
| 9/01/25 | [Signal Pulse](https://www.mql5.com/en/articles/16861) | Multiple timeframe analyzer | 1.0 | Initial Release | Tool number 7 |
| 17/01/25 | [Metrics Board](https://www.mql5.com/en/articles/16584) | Panel with button for analysis | 1.0 | Initial Release | Tool number 8 |
| 21/01/25 | [External Flow](https://www.mql5.com/en/articles/16967) | Analytics through external libraries | 1.0 | Initial Release | Tool number 9 |
| 27/01/25 | [VWAP](https://www.mql5.com/en/articles/16984) | Volume Weighted Average Price | 1.3 | Initial Release | Tool number 10 |
| 02/02/25 | [Heikin Ashi](https://www.mql5.com/en/articles/17021) | Trend Smoothening and reversal signal identification | 1.0 | Initial Release | Tool number 11 |
| 04/02/25 | [FibVWAP](https://www.mql5.com/en/articles/17121) | Signal generation through python analysis | 1.0 | Initial Release | Tool number  12 |
| 14/02/25 | [RSI DIVERGENCE](https://www.mql5.com/en/articles/17198) | Price action versus RSI divergences | 1.0 | Initial Release | Tool number 13 |
| 17/02/25 | [Parabolic Stop and Reverse (PSAR)](https://www.mql5.com/en/articles/17234) | Automating PSAR strategy | 1.0 | Initial Release | Tool number 14 |
| 20/02/25 | Quarters Drawer Script | Drawing quarters levels on chart | 1.0 | Initial Release | Tool number 15 |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17250.zip "Download all attachments in the single ZIP archive")

[Quarters\_Drawer.mq5](https://www.mql5.com/en/articles/download/17250/quarters_drawer.mq5 "Download Quarters_Drawer.mq5")(5.67 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Price Action Analysis Toolkit (Part 55): Designing a CPI Mini-Candle Overlay for Intra-bar Pressure](https://www.mql5.com/en/articles/20949)
- [Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://www.mql5.com/en/articles/20851)
- [Price Action Analysis Toolkit Development (Part 53): Pattern Density Heatmap for Support and Resistance Zone Discovery](https://www.mql5.com/en/articles/20390)
- [Price Action Analysis Toolkit Development (Part 52): Master Market Structure with Multi-Timeframe Visual Analysis](https://www.mql5.com/en/articles/20387)
- [Price Action Analysis Toolkit Development (Part 51): Revolutionary Chart Search Technology for Candlestick Pattern Discovery](https://www.mql5.com/en/articles/20313)
- [Price Action Analysis Toolkit Development (Part 50): Developing the RVGI, CCI and SMA Confluence Engine in MQL5](https://www.mql5.com/en/articles/20262)
- [Price Action Analysis Toolkit Development (Part 49): Integrating Trend, Momentum, and Volatility Indicators into One MQL5 System](https://www.mql5.com/en/articles/20168)

**[Go to discussion](https://www.mql5.com/en/forum/482044)**

![From Basic to Intermediate: Operators](https://c.mql5.com/2/88/From_basic_to_intermediate_Operators___LOGO.png)[From Basic to Intermediate: Operators](https://www.mql5.com/en/articles/15305)

In this article we will look at the main operators. Although the topic is simple to understand, there are certain points that are of great importance when it comes to including mathematical expressions in the code format. Without an adequate understanding of these details, programmers with little or no experience eventually give up trying to create their own solutions.

![Trading with the MQL5 Economic Calendar (Part 6): Automating Trade Entry with News Event Analysis and Countdown Timers](https://c.mql5.com/2/121/Trading_with_the_MQL5_Economic_Calendar_Part_6____LOGO.png)[Trading with the MQL5 Economic Calendar (Part 6): Automating Trade Entry with News Event Analysis and Countdown Timers](https://www.mql5.com/en/articles/17271)

In this article, we implement automated trade entry using the MQL5 Economic Calendar by applying user-defined filters and time offsets to identify qualifying news events. We compare forecast and previous values to determine whether to open a BUY or SELL trade. Dynamic countdown timers display the remaining time until news release and reset automatically after a trade.

![William Gann methods (Part II): Creating Gann Square indicator](https://c.mql5.com/2/89/logo-midjourney_image_15566_400_3863__3.png)[William Gann methods (Part II): Creating Gann Square indicator](https://www.mql5.com/en/articles/15566)

We will create an indicator based on the Gann's Square of 9, built by squaring time and price. We will prepare the code and test the indicator in the platform on different time intervals.

![Automating Trading Strategies in MQL5 (Part 9): Building an Expert Advisor for the Asian Breakout Strategy](https://c.mql5.com/2/121/Automating_Trading_Strategies_in_MQL5_Part_9__LOGO.png)[Automating Trading Strategies in MQL5 (Part 9): Building an Expert Advisor for the Asian Breakout Strategy](https://www.mql5.com/en/articles/17239)

In this article, we build an Expert Advisor in MQL5 for the Asian Breakout Strategy by calculating the session's high and low and applying trend filtering with a moving average. We implement dynamic object styling, user-defined time inputs, and robust risk management. Finally, we demonstrate backtesting and optimization techniques to refine the program.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/17250&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069653330667440263)

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