---
title: Introduction to MQL5 (Part 14): A Beginner's Guide to Building Custom Indicators (III)
url: https://www.mql5.com/en/articles/17574
categories: Trading Systems, Indicators
relevance_score: 9
scraped_at: 2026-01-22T17:35:10.810135
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/17574&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049233054687405979)

MetaTrader 5 / Trading systems


### Introduction

Welcome back to our MQL5 series! In the previous articles of this series, we explored how to build custom indicators in MQL5 using buffers and plots. Buffers allow us to store indicator values, while plots help visualize them on the chart. These methods are effective for many indicators, but they have limitations when it comes to creating complex visual representations.

In this article, we'll adopt a new approach by creating indicators using Meta Trader 5 chart objects. Additional flexibility is offered by chart objects, which let us create labels, shapes, and trend lines right on the chart without the need for indication buffers. This technique works well for developing indicators that need unique graphical components, showing patterns, and identifying important price levels.

We will create an indicator that resembles Harmonic Patterns to implement this. The logic we'll use can be modified to identify and depict different Harmonic Patterns, even though we won't be concentrating on any particular one (such as Gartley, Bat, or Butterfly). Instead of building a fully effective Harmonic Pattern detector, the main objective is to learn how to use chart objects in MQL5 to develop indicators. In Part 9 of this series, we looked at how to build and work with objects like trend lines, rectangles, and labels in MQL5, which is where we first covered using chart objects. Building on that knowledge, this piece will apply it to developing of indicators. By the end, you will have a firm grasp on how to develop unique visual indications by working with chart items dynamically.

In this article, you'll learn:

- How to create a custom indicator using MetaTrader 5 chart objects instead of relying on buffers and plots.

- Understanding the structure of Harmonic Patterns and how they are identified in price action.

- How to detect key swing points in the market to form potential Harmonic Patterns.

- Using Fibonacci retracement levels to validate pattern formations.

- How to draw geometric shapes programmatically (such as triangles and lines) to visualize patterns on the chart.

- How to filter out invalid patterns to improve accuracy in trade signals.


### **1\. Harmonic Pattern Indicator**

**1.1. Understanding the Harmonic Pattern Indicator**

Price structures known as harmonic patterns use particular Fibonacci ratios to pinpoint possible market reversal zones. The Gartley, Bat, and Butterfly patterns, among others, depend on price fluctuations creating geometric shapes that signal high-probability trading opportunities.

Harmonic Patterns necessitate identifying critical price points and using Fibonacci ratios to validate their structure, whereas standard indicators create signals using buffers and plots. As a result, they are more difficult to identify and display than more conventional indicators like RSI or Moving Averages.

The logic we use can be modified to identify and depict different Harmonic Patterns, even though we won't be concentrating on any particular one (such Gartley, Bat, or Butterfly). Instead of building a fully effective Harmonic Pattern detector, the main objective is to learn how to use chart objects in MQL5 to develop indicators.

To do this, we will mark important swing points (highs and lows) on the chart, use text objects (XABCD) to indicate important levels, illustrate the pattern structure with different objects, and apply Fibonacci levels to improve pattern accuracy and validation.

**1.2. Setting up the Project**

As I often say, you must first envision how an indicator will appear before you can design one. Knowing exactly how the indicator should be set up facilitates the development process and guarantees that the end product meets our requirements.

In this project, we will use MetaTrader 5 chart objects to build both bullish and bearish harmonic patterns. The indicator will draw several graphical elements to form the pattern structure, locate important swing points, mark them with text objects (XABCD), and add Fibonacci levels to improve the presentation of the pattern.

In addition to creating an indicator that closely resembles harmonic patterns, this method will give us hands-on experience with MQL5 chart objects, which will facilitate the creation of future custom indicators. In addition to discussing how to create these patterns with chart objects, we will also go over potential mistakes that could occur. Problems can arise, such as misaligned objects, misplaced Fibonacci levels, and inaccurate swing point detection. To guarantee precise pattern viewing and seamless indicator operation, we will go over these problems and offer solutions.

**1.2.1. Bullish Pattern**

We will use a methodical technique to identify the important swing points that make up the pattern to develop the Bullish Harmonic Pattern Indicator. The process entails determining particular pricing points and making sure they follow the guidelines for harmonic patterns.

Identifying a swing low (X) as the pattern's beginning point will be our first step. Next, we will identify a swing high (A) and a swing low (B). B needs to retrace between 61.8% and 78.6% of the XA leg for the pattern to stay valid. We will then look for a swing high (C) and validate the swing low (D). The complete structure of the bullish pattern is formed when the last point, D, is lower than X.

When these conditions are satisfied, we will utilize chart objects — such as trend lines joining the swing points, labels for XABCD points, and Fibonacci retracement levels to confirm the structure — to graphically depict the pattern on the chart. This will offer a methodical and transparent approach to identifying possible bullish reversal zones. To make it easier for you to recognize important trading zones inside the pattern, we will also utilize chart objects to indicate possible entry positions, stop-loss (SL), and take-profit (TP) levels. This will help with trade decision-making by offering a methodical and transparent approach to identify possible bullish reversal zones.

![Figure 1.  Bullish Pattern](https://c.mql5.com/2/127/Figure_1__1.png)

Pseudocode:

// Step 1: Identify Swing Points

- Detect swing low (X)
- Identify swing high (A) after X
- Detect swing low (B) after A

IF B retracement is NOT between 61.8% and 78.6% of XA

- Discard pattern and restart detection

IF B retracement is between 61.8% and 78.6% of XA

- Identify swing high (C) after B
- Detect swing low (D) after C

IF D is NOT lower than X

- THEN discard pattern and restart detection

// Step 2: Draw Chart Objects

- Draw objects connecting X → A → B → C → D
-  Label points with text objects: X, A, B, C, D
- Draw Fibonacci retracement from X to A for validation

// Step 3: Define Trade Levels

- Create chart objects to mark Entry, SL, and TP.

**1.2.2. Bearish Pattern**

Similar in structure to the bullish pattern, but in reverse, is the Bearish Harmonic Pattern Indicator. To make sure they follow the guidelines for harmonic patterns, we will pinpoint important swing spots. The pattern will be identified by first identifying a swing high (X), which will be followed by a swing low (A). The swing high (B), which needs to retrace between 61.8% and 78.6% of the XA leg, will then be determined. The final swing high (D) will then be confirmed when we have detected a swing low (C). D must be above X, creating a possible bearish reversal zone, to validate the pattern.

Following confirmation of the structure, we will use chart objects, such as trend lines, XABCD labels, and Fibonacci retracement levels, to graphically depict the pattern. To assist you in locating important trading zones inside the pattern, we will further indicate the entry, stop-loss (SL), and take-profit (TP) levels.

![Figure 2. Bearish Pattern](https://c.mql5.com/2/127/Figure_2.png)

Pseudocode:

// Step 1: Identify Swing Points

- Detect swing high (X)
- Detect swing low (A) after X
- Detect swing high (B) after A

IF B retracement is NOT between 61.8% and 78.6% of XA

- Discard pattern and restart detection

IF B retracement is between 61.8% and 78.6% of XA

- Detect swing low (C) after B
- Detect swing high (D) after C

IF D is NOT above than X

- Discard pattern and restart detection

// Step 2: Draw Chart Objects

- Draw objects connecting X → A → B → C → D
- Label points with text objects: X, A, B, C, D
- Draw Fibonacci retracement from X to A for validation

// Step 3: Define Trade Levels

- Create chart objects to mark Entry, SL, and TP.

### **2\. Building Bullish Harmonic Patterns**

We will start integrating the Harmonic Pattern Indicator into MQL5 in this section. We will specify the functions required to identify swing points, confirm retracement levels, and visualize the pattern on the chart using chart objects. Additionally, we will customize the appearance of the drawn objects to enhance clarity and usability.

**2.1. Identifying Swing Highs and Lows**

A reliable technique for locating swing highs and swing lows on the chart is necessary before we can spot a harmonic pattern. The XABCD structure is plotted using swing points as the base. We determine whether a candlestick's high is the highest among a specified range of nearby candles to identify a swing high. In a similar vein, a swing low is identified when the low of a candlestick is the lowest in a certain range. The size of this range determines the sensitivity of our detection — larger values capture major swings, while smaller values detect minor fluctuations. In the next step, we will develop an algorithm to scan historical price data, locate key swing points, and guarantee they correspond with the pattern formation principles.

**2.1.1. Swing Highs and Lows Functions**

Example:

```
#property indicator_chart_window

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(INIT_SUCCEEDED);
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
//---

//--- return value of prev_calculated for next call
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING LOW                                           |
//+------------------------------------------------------------------+
bool IsSwingLow(const double &low[], int index, int lookback)
  {
   for(int i = 1; i <= lookback; i++)
     {
      if(low[index] > low[index - i] || low[index] > low[index + i])
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING HIGH                                          |
//+------------------------------------------------------------------+
bool IsSwingHigh(const double &high[], int index, int lookback)
  {
   for(int i = 1; i <= lookback; i++)
     {
      if(high[index] < high[index - i] || high[index] < high[index + i])
         return false;
     }
   return true;
  }
```

Explanation:

MetaTrader 5 is informed by the directive #property indicator\_chart\_window that the custom indicator needs to be shown on the primary price chart instead of in a different sub-window. This is helpful when creating pattern-based indications like Harmonic Patterns or indicators that overlay price activity, like trend lines, support and resistance levels, etc. In the absence of this characteristic, the indicator might by default plot in a different indicator window, much as oscillators like MACD or RSI.

The purpose of the IsSwingLow function is to identify price chart swing lows. When a candle's low falls below the lows of the surrounding candles within a certain range, it's known as a swing low. The function accepts a lookback value that establishes how many candles before and after should be considered, an array of low prices, and the index of the candle under evaluation. It determines whether the current low is the lowest by iterating through the nearby candles. It returns false if it discovers a lower value in the surrounding range, proving that the current candle is not a swing low. If not, it returns true, indicating that a legitimate swing low is present.

Similar to this, swing highs — points where a candle's high exceeds the highs of the nearby candles — are found using the IsSwingHigh function. The function uses the same logic as IsSwingLow, except it makes sure that the high at the specified index is the highest value in the lookback range, rather than searching for the lowest value. The function returns false if any of the surrounding candles have a higher value. If not, it confirms a swing high and returns true.

The detection of significant price turning moments, which provide the basis for defining Harmonic Patterns, requires both of these capabilities. Following the identification of swing highs and lows, the XABCD structure of the pattern can be plotted by joining them with trend lines and using Fibonacci retracements for confirmation. With this method, the indicator is guaranteed to dynamically adjust to fresh price information and update the identified patterns appropriately.

Analogy:

Finding the lowest point in a valley is how the IsSwingLow function operates. Consider yourself standing on a hiking trail and trying to locate the path's deepest dip. During the lookback time, you take a few strides forward and a few steps back. The present point is verified as the lowest dip (swing low) if it is lower than every other point in that range. Your current location is only another section of the slope and not the actual lowest point if any neighboring points are lower. Potential reversal zones where prices might begin to rise are identified with the use of this function.

Finding the highest point in a mountain range is similar to using the IsSwingHigh function. It is considered a swing high if you stand at one spot and take a few steps forward and backward to make sure your current position is the highest within that range. How far you check before announcing a peak or a drop depends on the lookback time; if it's too little, slight variations could be confused for important points, and if it's too big, you might miss smaller but major trends. This equilibrium guarantees that the identified swing points are significant and not merely arbitrary fluctuations in the price.

![Figure 3. Swing High and Low](https://c.mql5.com/2/127/figure_3.png)

**2.1.2. Identifying Swing Low (X)**

The next step is to incorporate the IsSwingLow and IsSwingHigh functions into the OnCalculate function when they have been created. Here, real-time price data will be subjected to the logic of swing point detection. Finding the initial swing low (X), which is where the harmonic pattern begins, is the first challenge. Using our created function, we can iteratively examine each candle in OnCalculate's price history to see if it meets the criteria for a swing low.

Example:

```
#property indicator_chart_window

// Input parameters
input int LookbackBars = 10; // Number of bars to look back/forward for swing points
input int bars_check  = 1000; // Number of bars to check for swing points

// CHART ID
long chart_id = ChartID(); // Get the ID of the current chart to manage objects (lines, text, etc.)

//X
double X; // Price of the swing low (X).
datetime X_time; // Time of the swing low (X).
string X_line; // Unique name for the trend line object.
string X_letter; // Unique name for the text label object.

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(INIT_SUCCEEDED);
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
//---

   if(rates_total >= bars_check)
     {

      for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
        {
         if(IsSwingLow(low, i, LookbackBars))
           {
            // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
            X = low[i]; // Price of the swing low (X).
            X_time = time[i]; // Time of the swing low (X).
            X_line = StringFormat("XLow%d", i); // Unique name for the trend line object.
            X_letter = StringFormat("X%d", i); // Unique name for the text label object.

            ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, X_time, X); // Create text object for X
            ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".

            ObjectCreate(chart_id,X_line,OBJ_TREND,0,X_time,X,time[i+LookbackBars],X); // Create line to mark X
           }
        }
     }

//--- return value of prev_calculated for next call
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING LOW                                           |
//+------------------------------------------------------------------+
bool IsSwingLow(const double &low[], int index, int lookback)
  {

   for(int i = 1; i <= lookback; i++)
     {
      if(low[index] > low[index - i] || low[index] > low[index + i])
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING HIGH                                          |
//+------------------------------------------------------------------+
bool IsSwingHigh(const double &high[], int index, int lookback)
  {

   for(int i = 1; i <= lookback; i++)
     {
      if(high[index] < high[index - i] || high[index] < high[index + i])
         return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Output:

![Figure 4. Swing Low (X)](https://c.mql5.com/2/127/Figurw_4.png)

Explanation:

We start by stating some crucial input data, since determining the swing low (X) is the first step in finding a bullish harmonic pattern. The number of bars that should be examined before and after a specific point to make sure it is a legitimate swing point is determined by the LookbackBars variable. This aids in removing small price swings that don't result in notable swing lows. To ensure that the script processes only a manageable number of historical bars to retain efficiency, the bars\_check argument specifies how many bars should be analyzed.

Additionally, we use ChartID() to acquire the chart\_id, which enables us to create and manage chart objects such as trend lines and text labels. We declare variables "X" to hold the price of the detected swing low (X), "X\_time" to record the time it occurred, and "X\_line" and "X\_letter" to give distinct names to the objects that will be used to identify this point on the chart. The text label indicating the swing low will be created using the X\_letter variable, and a trend line will be created for improved display using the X\_line variable.

We then use these preliminary configurations to determine the swing low (X) in the OnCalculate function. We only begin checking when there is sufficient pricing data available, thanks to the condition if(rates\_total >= bars\_check). To avoid checking incomplete pricing data, the loop iterates through historical bars, starting at rates\_total - bars\_check and ending at rates\_total - LookbackBars. The IsSwingLow function in the loop detects whether a given point is a swing low. When a legitimate swing low is discovered, we record its time and price, come up with original object names, and then produce visual components for the chart.

A text object is created using ObjectCreate(chart\_id, X\_letter, OBJ\_TEXT, 0, X\_time, X) to indicate the swing low with the letter "X." To further illustrate the identified swing low, we employ ObjectCreate(chart\_id, X\_line, OBJ\_TREND, 0, X\_time, X, time\[i+LookbackBars\], X) to build a trend line at the X level. This guarantees that the first important point in our harmonic pattern structure is accurately identified and shown on the chart, laying the groundwork for determining the pattern's subsequent points.

**2.1.3. Identifying Swing High (A)**

Finding the swing high (A) is the next stage after successfully identifying the swing low (X). This point is essential to the formation of the harmonic pattern since it is the first notable upward movement following X. In the same way that we discovered X, we will utilize the IsSwingHigh function to find A, but this time we will look for a peak rather than a dip.

We will search for a point in the price data after X where the high exceeds the highs of the surrounding bars within the designated LookbackBars range. After identifying a legitimate swing high, we will record its price and timing, give it a distinctive name, and use chart objects to properly describe it.

Example:

```
#property indicator_chart_window

// Input parameters
input int LookbackBars = 10; // Number of bars to look back/forward for swing points
input int bars_check  = 1000; // Number of bars to check for swing points

// CHART ID
long chart_id = ChartID(); // Get the ID of the current chart to manage objects (lines, text, etc.)

//X
double X; // Price of the swing low (X).
datetime X_time; // Time of the swing low (X).
string X_line; // Unique name for the trend line object.
string X_letter; // Unique name for the text label object.

//A
double A; // Price of the swing high (A).
datetime A_time; // Time of the swing high (A).
string A_line; // Unique name for the trend line object.
string A_letter; // Unique name for the text label object.

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(INIT_SUCCEEDED);
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
//---

   if(rates_total >= bars_check)
     {

      for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
        {
         if(IsSwingLow(low, i, LookbackBars))
           {
            // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
            X = low[i]; // Price of the swing low (X).
            X_time = time[i]; // Time of the swing low (X).
            X_line = StringFormat("XLow%d", i); // Unique name for the trend line object.
            X_letter = StringFormat("X%d", i); // Unique name for the text label object.

            ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, X_time, X); // Create text object for X
            ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
            ObjectCreate(chart_id,X_line,OBJ_TREND,0,X_time,X,time[i+LookbackBars],X); // Create line to mark X

            for(int j = i; j < rates_total - LookbackBars; j++)
              {
               if(IsSwingHigh(high, j, LookbackBars)  && time[j] > X_time)
                 {

                  A = high[j]; // Price of the swing high (A).
                  A_time = time[j]; // Time of the swing high (A).
                  A_line = StringFormat("AHigh%d", j); // Unique name for the trend line object.
                  A_letter = StringFormat("A%d", j); // Unique name for the text label object.

                  ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, A_time, A); // Create text object for A
                  ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
                  ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                  ObjectCreate(chart_id,A_line,OBJ_TREND,0,A_time,A,time[j+LookbackBars],A); // Create line to mark A
                  ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

                  break;
                 }
              }
           }
        }
     }

//--- return value of prev_calculated for next call
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING LOW                                           |
//+------------------------------------------------------------------+
bool IsSwingLow(const double &low[], int index, int lookback)
  {
   for(int i = 1; i <= lookback; i++)
     {
      if(low[index] > low[index - i] || low[index] > low[index + i])
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING HIGH                                          |
//+------------------------------------------------------------------+
bool IsSwingHigh(const double &high[], int index, int lookback)
  {
   for(int i = 1; i <= lookback; i++)
     {
      if(high[index] < high[index - i] || high[index] < high[index + i])
         return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Output:

![Figure 5. Swing High (A)](https://c.mql5.com/2/127/Figure_5.png)

Explanation:

We first define the required variables to determine the swing high (A). A\_time logs the precise moment the swing high happened, while A stores the price value of the swing high. We use A\_line as the name for the trend line object and A\_letter as the text label to guarantee that every detected A point has a distinct visual representation. These variables aid in the creation of chart objects that prominently display the A point, which facilitates the visualization of the harmonic pattern's structure.

Once X has been identified, the code iterates through pricing data, beginning at X's location, to find A. The IsSwingHigh function, which ascertains whether a bar is a local high within the designated LookbackBars range, is used to check each bar. The condition time\[j\] > X\_time is used to guarantee that the swing high A comes after X.

This ensures that the correct sequence in the harmonic pattern is maintained by selecting A only if its timestamp is greater than X\_time. The price and timing of a legitimate swing high are recorded, and the trend line and text label objects are given distinctive identities if one is discovered. A green trend line is constructed to highlight the identified swing high, and a text label with the letter "A" is placed there. As soon as the first legitimate swing high (A) has been identified, the loop is broken using the break; expression. The loop would keep looking for more swing highs without break;, maybe overwriting A with a later swing high. Since we only want the first instance of A following X, the break; makes sure that the loop ends iterating as soon as a swing high is detected, protecting the first valid A and avoiding pointless calculations.

**2.1.4. Identifying Swing Low (B)**

Finding the following swing low, B, is the next step after determining the swing high, A. The hunt for B starts as soon as A is identified because B must come after A. This is accomplished by iterating through the price data, beginning at position A, and using the IsSwingLow function to compare a bar to nearby bars in the LookbackBars range to determine whether it is a swing low. Additionally, B is anticipated to fall under a particular Fibonacci retracement level of XA in harmonic pattern identification. We will not, however, discuss Fibonacci validation at this time; we are only concerned with identifying B.

The price and time of a swing low are recorded in B and B\_time, respectively, as soon as it is identified. We provide the trend line (B\_line) and text label (B\_letter) distinctive names to indicate this location on the chart. A trend line is produced to visually connect and highlight the text object with the label "B" that is created at the swing low point. This stage guarantees that the three main points — X, A, and B — that now serve as the framework for our harmonic pattern structure are in place.

Example:

```
#property indicator_chart_window

// Input parameters
input int LookbackBars = 10; // Number of bars to look back/forward for swing points
input int bars_check  = 1000; // Number of bars to check for swing points

// CHART ID
long chart_id = ChartID(); // Get the ID of the current chart to manage objects (lines, text, etc.)

//X
double X; // Price of the swing low (X).
datetime X_time; // Time of the swing low (X).
string X_line; // Unique name for the trend line object.
string X_letter; // Unique name for the text label object.

//A
double A; // Price of the swing high (A).
datetime A_time; // Time of the swing high (A).
string A_line; // Unique name for the trend line object.
string A_letter; // Unique name for the text label object.

//B
double B; // Price of the swing low (B).
datetime B_time; // Time of the swing low (B).
string B_line; // Unique name for the trend line object.
string B_letter; // Unique name for the text label object.

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// This function is called when the indicator is removed or the chart is closed.
// Delete all objects (lines, text, etc.) from the chart to clean up.
   ObjectsDeleteAll(chart_id);
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
//---

   if(rates_total >= bars_check)
     {

      for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
        {
         if(IsSwingLow(low, i, LookbackBars))
           {
            // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
            X = low[i]; // Price of the swing low (X).
            X_time = time[i]; // Time of the swing low (X).
            X_line = StringFormat("XLow%d", i); // Unique name for the trend line object.
            X_letter = StringFormat("X%d", i); // Unique name for the text label object.

            ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, X_time, X); // Create text object for X
            ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
            ObjectCreate(chart_id,X_line,OBJ_TREND,0,X_time,X,time[i+LookbackBars],X); // Create line to mark X

            for(int j = i; j < rates_total - LookbackBars; j++)
              {
               if(IsSwingHigh(high, j, LookbackBars) && time[j] > X_time)
                 {

                  A = high[j]; // Price of the swing high (A).
                  A_time = time[j]; // Time of the swing high (A).
                  A_line = StringFormat("AHigh%d", j); // Unique name for the trend line object.
                  A_letter = StringFormat("A%d", j); // Unique name for the text label object.

                  ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, A_time, A); // Create text object for A
                  ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
                  ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                  ObjectCreate(chart_id,A_line,OBJ_TREND,0,A_time,A,time[j+LookbackBars],A); // Create line to mark A
                  ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

                  for(int k = j; k < rates_total - LookbackBars; k++)
                    {
                     if(IsSwingLow(low, k, LookbackBars) && time[k] > A_time)
                       {

                        // If a swing low is found, store its price, time, and create a name for the object.
                        B = low[k]; // Price of the swing low (B).
                        B_time = time[k]; // Time of the swing low (B).
                        B_line = StringFormat("BLow%d", k); // Unique name for the trend line object.
                        B_letter = StringFormat("B%d", k); // Unique name for the text label object.

                        ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, B_time, B); // Create text object for B
                        ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
                        ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                        ObjectCreate(chart_id,B_line,OBJ_TREND,0,B_time,B,time[k+LookbackBars],B); // Create line to mark B
                        ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to green

                        break;

                       }
                    }

                  break;
                 }
              }
           }
        }
     }

//--- return value of prev_calculated for next call
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING LOW                                           |
//+------------------------------------------------------------------+
bool IsSwingLow(const double &low[], int index, int lookback)
  {
   for(int i = 1; i <= lookback; i++)
     {
      if(low[index] > low[index - i] || low[index] > low[index + i])
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING HIGH                                          |
//+------------------------------------------------------------------+
bool IsSwingHigh(const double &high[], int index, int lookback)
  {

   for(int i = 1; i <= lookback; i++)
     {
      if(high[index] < high[index - i] || high[index] < high[index + i])
         return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Output:

![Figure 6.Swing Low (B)](https://c.mql5.com/2/127/figure_6.png)

Explanation:

The price of the swing low that follows A is represented by the variable B. A loop that begins at point A and uses the IsSwingLow function to find the next swing low is used to identify it. The price, time, and distinct names for the trend line and text label are saved after they are identified. Then, to indicate B on the chart, the computer generates a trend line in midnight blue and a text label ("B") in green. To maintain the proper swing point sequence for pattern generation, a break statement makes sure that just the first valid B is chosen. Because X and B are both recognized as swing lows throughout the loop iteration, they may occasionally overlap in the current implementation. Given that the program assesses each swing point in turn, this behavior is to be expected. However, it could result in redundant annotations on the chart if X and B are the same.

You can include a condition to make sure that B is a separate swing low that follows A, but B is not the same as X if you wish to keep X and B from overlapping. I will include a choice in the code that lets you choose whether X and B can overlap. The indicator will operate as it does now, where X and B may be the same swing low if overlap is permitted. However, the indicator will make sure that B is a different swing low from X if overlap is prohibited.

Because B must be a separate low rather than possibly being the same as X, preventing overlap will result in fewer harmonic patterns being discovered. Depending on the user's preference for pattern recognition, this option offers flexibility.

Example:

```
input bool overlap = false; //Allow Overlaping
```

```
for(int k = j; k < rates_total - LookbackBars; k++)
  {
   if(IsSwingLow(low, k, LookbackBars) && time[k] > A_time)
     {

      // If a swing low is found, store its price, time, and create a name for the object.
      B = low[k]; // Price of the swing low (B).
      B_time = time[k]; // Time of the swing low (B).
      B_line = StringFormat("BLow%d", k); // Unique name for the trend line object.
      B_letter = StringFormat("B%d", k); // Unique name for the text label object.

      ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, B_time, B); // Create text object for B
      ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
      ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
      ObjectCreate(chart_id,B_line,OBJ_TREND,0,B_time,B,time[k+LookbackBars],B); // Create line to mark B
      ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to green

      if(overlap == false)
        {
         i = k;
        }
      if(overlap == true)
        {
         i = i;
        }

      break;

     }
  }
```

Output:

![Figure 7. X and B Overlap](https://c.mql5.com/2/127/figure_7.png)

Explanation:

Whether X and B can have the same swing low depends on the overlap input parameter. The line i = k; guarantees that the iteration skips forward once B is located, preventing B from being the same as X when overlap is set to false. This limits the amount of harmonic patterns that can be recognized while simultaneously minimizing redundancy by maintaining a distinct separation between swing points.

However, the line i = i; indicates that the loop proceeds regularly without requiring a skip when overlap is true. This makes it possible to discover more harmonic patterns by allowing X and B to overlap when they naturally occur at the same price level. It could, however, also provide some repetition to the pattern markings. By successfully preventing X and B from overlapping, the code has made sure that they are separate swing points. As the picture illustrates, B now manifests independently of X, preserving a more distinct pattern structure.

**2.1.5. Identifying Swing High (C)**

Finding the swing high (C) comes after determining the swing low (X), swing high (A), and following swing low (B). This point is essential for building the structure of a harmonic pattern, since it determines the pattern's overall form and possible reversal zone.

Both the trend line (C\_line) and the text label (C\_letter) are given distinct names to visually indicate this point. A trend line is built to visually connect and highlight a text item with the label "C" that is positioned at the swing high on the chart. This stage guarantees that the four main points — X, A, B, and C — that make up the harmonic pattern's structure are now in place.

Example:

```
//C
double C; // Price of the swing high (C).
datetime C_time; // Time of the swing high (C).
string C_line; // Unique name for the trend line object.
string C_letter; // Unique name for the text label object.
```

```
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
//---

   if(rates_total >= bars_check)
     {

      for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
        {
         if(IsSwingLow(low, i, LookbackBars))
           {
            // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
            X = low[i]; // Price of the swing low (X).
            X_time = time[i]; // Time of the swing low (X).
            X_line = StringFormat("XLow%d", i); // Unique name for the trend line object.
            X_letter = StringFormat("X%d", i); // Unique name for the text label object.

            ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, X_time, X); // Create text object for X
            ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
            ObjectCreate(chart_id,X_line,OBJ_TREND,0,X_time,X,time[i+LookbackBars],X); // Create line to mark X

            for(int j = i; j < rates_total - LookbackBars; j++)
              {
               if(IsSwingHigh(high, j, LookbackBars) && time[j] > X_time)
                 {

                  A = high[j]; // Price of the swing high (A).
                  A_time = time[j]; // Time of the swing high (A).
                  A_line = StringFormat("AHigh%d", j); // Unique name for the trend line object.
                  A_letter = StringFormat("A%d", j); // Unique name for the text label object.

                  ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, A_time, A); // Create text object for A
                  ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
                  ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                  ObjectCreate(chart_id,A_line,OBJ_TREND,0,A_time,A,time[j+LookbackBars],A); // Create line to mark A
                  ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

                  for(int k = j; k < rates_total - LookbackBars; k++)
                    {
                     if(IsSwingLow(low, k, LookbackBars) && time[k] > A_time)
                       {

                        // If a swing low is found, store its price, time, and create a name for the object.
                        B = low[k]; // Price of the swing low (B).
                        B_time = time[k]; // Time of the swing low (B).
                        B_line = StringFormat("BLow%d", k); // Unique name for the trend line object.
                        B_letter = StringFormat("B%d", k); // Unique name for the text label object.

                        ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, B_time, B); // Create text object for B
                        ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
                        ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                        ObjectCreate(chart_id,B_line,OBJ_TREND,0,B_time,B,time[k+LookbackBars],B); // Create line to mark B
                        ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to MidnightBlue

                        for(int l = j ; l < rates_total - LookbackBars; l++)
                          {
                           if(IsSwingHigh(high, l, LookbackBars) && time[l] > B_time)
                             {
                              C = high[l]; // Price of the swing high (C).
                              C_time = time[l]; // Time of the swing high (C).
                              C_line = StringFormat("CHigh%d", l); // Unique name for the trend line object.
                              C_letter = StringFormat("C%d", l); // Unique name for the text label object.

                              ObjectCreate(chart_id, C_letter, OBJ_TEXT, 0, C_time, C); // Create text object for C
                              ObjectSetString(chart_id, C_letter, OBJPROP_TEXT, "C"); // Set the text to "C".
                              ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                              ObjectCreate(chart_id,C_line,OBJ_TREND,0,C_time,C,time[l+LookbackBars],C); // Create line to mark C
                              ObjectSetInteger(chart_id,C_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

                              if(overlap == false)
                                {
                                 i = l;
                                }
                              if(overlap == true)
                                {
                                 i = i;
                                }

                              break;

                             }
                          }

                        break;

                       }
                    }

                  break;
                 }
              }
           }
        }
     }

//--- return value of prev_calculated for next call
   return(rates_total);
  }
```

Output:

![Figure 8. Swing High (C)](https://c.mql5.com/2/127/figure_8.png)

Explanation:

After determining the swing low (B), this part of the code oversees determining the swing high (C), which is the subsequent crucial step. Iterating over the price data, the loop for(int l = j; l < rates\_total - LookbackBars; l++) begins with index j, where the previous swing high (A) was found. To make sure that C comes after B in the sequence, it keeps looking until it finds the most recent bars. To maintain the proper order of the pattern, the condition if(IsSwingHigh(high, l, LookbackBars) && time\[l\] > B\_time) verifies whether the current bar is a swing high within the specified lookback range and makes sure that its timestamp is later than B.

The price and time of a legitimate swing high are saved in C and C\_time, respectively, and the trend line and text label objects are given distinct names (C\_line and C\_letter). Using ObjectCreate(chart\_id, C\_line, OBJ\_TREND, 0, C\_time, C, time\[l+LookbackBars\], C), a trend line is built to highlight the text label "C" at the indicated swing high. To visually differentiate the C trend line from other segments, the ObjectSetInteger(chart\_id, C\_line, OBJPROP\_COLOR, clrSaddleBrown); sets the color of the line to SaddleBrown.

The final conditional block guarantees that overlapping patterns are handled correctly: i = l; stops previous swing points from being used in another pattern if overlap is set to false. Multiple patterns can overlap if overlap is true, as i = i; preserves the current value. After determining the first legitimate swing high (C), break; finally ends the loop, guaranteeing efficiency by avoiding needless repeats.

**2.1.6. Identifying Swing Low (D)**

Identifying the swing low (D) comes after determining the swing high (C). This guarantees that the price action is sequenced correctly for the production of harmonic patterns. To find the next swing low following C in chronological order, the search for D starts from the location where C was identified. Two requirements must be fulfilled to verify a legitimate swing low. First, within the designated lookback period, the current price must be considered a swing low. Second, to maintain the proper pattern order, its timestamp needs to be later than C's.

The price and timing of a legitimate swing low are noted, and distinct identifiers are created for the trend line and text label that correspond to it. To make the swing low easier to spot within the pattern, a trend line is made to emphasize it and a text label is inserted at its location to visually designate it.

Example:

```
//D
double D; // Price of the swing low (D).
datetime D_time; // Time of the swing low (D).
string D_line; // Unique name for the trend line object.
string D_letter; // Unique name for the text label object.

//Trend
string XA_line; // Unique name for XA trend line object.
string AB_line; // Unique name for AB trend line object.
string BC_line; // Unique name for BC trend line object.
string CD_line; // Unique name for CD trend line object.

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
//---

   if(rates_total >= bars_check)
     {

      for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
        {
         if(IsSwingLow(low, i, LookbackBars))
           {
            // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
            X = low[i]; // Price of the swing low (X).
            X_time = time[i]; // Time of the swing low (X).
            X_line = StringFormat("XLow%d", i); // Unique name for the trend line object.
            X_letter = StringFormat("X%d", i); // Unique name for the text label object.

            ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, X_time, X); // Create text object for X
            ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
            ObjectCreate(chart_id,X_line,OBJ_TREND,0,X_time,X,time[i+LookbackBars],X); // Create line to mark X

            for(int j = i; j < rates_total - LookbackBars; j++)
              {
               if(IsSwingHigh(high, j, LookbackBars) && time[j] > X_time)
                 {

                  A = high[j]; // Price of the swing high (A).
                  A_time = time[j]; // Time of the swing high (A).
                  A_line = StringFormat("AHigh%d", j); // Unique name for the trend line object.
                  A_letter = StringFormat("A%d", j); // Unique name for the text label object.

                  ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, A_time, A); // Create text object for A
                  ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
                  ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                  ObjectCreate(chart_id,A_line,OBJ_TREND,0,A_time,A,time[j+LookbackBars],A); // Create line to mark A
                  ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

                  for(int k = j; k < rates_total - LookbackBars; k++)
                    {
                     if(IsSwingLow(low, k, LookbackBars) && time[k] > A_time)
                       {

                        // If a swing low is found, store its price, time, and create a name for the object.
                        B = low[k]; // Price of the swing low (B).
                        B_time = time[k]; // Time of the swing low (B).
                        B_line = StringFormat("BLow%d", k); // Unique name for the trend line object.
                        B_letter = StringFormat("B%d", k); // Unique name for the text label object.

                        ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, B_time, B); // Create text object for B
                        ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
                        ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                        ObjectCreate(chart_id,B_line,OBJ_TREND,0,B_time,B,time[k+LookbackBars],B); // Create line to mark B
                        ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to MidnightBlue

                        for(int l = j ; l < rates_total - LookbackBars; l++)
                          {
                           if(IsSwingHigh(high, l, LookbackBars) && time[l] > B_time)
                             {
                              C = high[l]; // Price of the swing high (C).
                              C_time = time[l]; // Time of the swing high (C).
                              C_line = StringFormat("CHigh%d", l); // Unique name for the trend line object.
                              C_letter = StringFormat("C%d", l); // Unique name for the text label object.

                              ObjectCreate(chart_id, C_letter, OBJ_TEXT, 0, C_time, C); // Create text object for C
                              ObjectSetString(chart_id, C_letter, OBJPROP_TEXT, "C"); // Set the text to "C".
                              ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                              ObjectCreate(chart_id,C_line,OBJ_TREND,0,C_time,C,time[l+LookbackBars],C); // Create line to mark C
                              ObjectSetInteger(chart_id,C_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

                              for(int m = l; m < rates_total - (LookbackBars / 2); m++)
                                {
                                 if(IsSwingLow(low, m, LookbackBars / 2) && time[m] > C_time)
                                   {
                                    D = low[m]; // Price of the swing low (D).
                                    D_time = time[m]; // Time of the swing low (D).
                                    D_line = StringFormat("DLow%d", m); // Unique name for the trend line object.
                                    D_letter = StringFormat("D%d", m); // Unique name for the text label object.

                                    ObjectCreate(chart_id, D_letter, OBJ_TEXT, 0, D_time, D); // Create text object for D
                                    ObjectSetString(chart_id, D_letter, OBJPROP_TEXT, "D"); // Set the text to "D".
                                    ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,D_line,OBJ_TREND,0,D_time,D,time[m+LookbackBars],D); // Create line to mark D
                                    ObjectSetInteger(chart_id,D_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

                                    XA_line = StringFormat("XA Line%d", m); // Unique name for the XA line.
                                    ObjectCreate(chart_id,XA_line,OBJ_TREND,0,X_time,X,A_time,A); // Create line to connect XA
                                    ObjectSetInteger(chart_id,XA_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,XA_line,OBJPROP_WIDTH,3); // Line width

                                    AB_line = StringFormat("AB Line%d", m); // Unique name for the AB line.
                                    ObjectCreate(chart_id,AB_line,OBJ_TREND,0,A_time,A,B_time,B); // Create line to connect AB
                                    ObjectSetInteger(chart_id,AB_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,AB_line,OBJPROP_WIDTH,3); // Line width

                                    BC_line = StringFormat("BC Line%d", m); // Unique name for the BC line.
                                    ObjectCreate(chart_id,BC_line,OBJ_TREND,0,B_time,B,C_time,C); // Create line to connect BC
                                    ObjectSetInteger(chart_id,BC_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,BC_line,OBJPROP_WIDTH,3); // Line width

                                    CD_line = StringFormat("CD Line%d", m); // Unique name for the CD line.
                                    ObjectCreate(chart_id,CD_line,OBJ_TREND,0,C_time,C,D_time,D); // Create line to connect CD
                                    ObjectSetInteger(chart_id,CD_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,CD_line,OBJPROP_WIDTH,3); // Line width

                                    if(overlap == false)
                                      {
                                       i = m;
                                      }
                                    if(overlap == true)
                                      {
                                       i = i;
                                      }
                                    break;

                                   }
                                }

                              break;

                             }
                          }

                        break;

                       }
                    }

                  break;
                 }
              }
           }
        }
     }

//--- return value of prev_calculated for next call
   return(rates_total);
  }
```

Output:

![Figure 9. Swing Low (D)](https://c.mql5.com/2/127/figure_9.png)

Explanation:

The price of the swing low at point D in the recognized pattern is represented by the variable D. D is a crucial point in harmonic pattern trading, where price action may turn around and finish the pattern. D\_time records the timestamp of this low's occurrence in addition to D. To visually identify this swing low, the trend line and text item on the chart are also created and labeled using D\_line and D\_letter.

The program iterates through the pricing data to find point D in the loop that begins with for(int m = l; m < rates\_total - (LookbackBars / 2); m++).... To ascertain whether the current price at index m is a swing low, it makes use of the IsSwingLow() function, making sure that the low is after point C in time. The price and time of a valid swing low are assigned to D and D\_time, respectively.

The indicator produces visual elements on the chart after point D has been located. At the location of the swing low, the ObjectCreate() function creates a text label (D\_letter) with the letter "D" and tints it green for convenience. To keep visual representation consistent, a trend line (D\_line) is also drawn at point D, extending for LookbackBars periods, and given the color brown (clrSaddleBrown).

At this point, the program also creates trend lines that link the swing points that have been identified. To visually connect the corresponding points (X to A, A to B, B to C, and C to D), the XA\_line, AB\_line, BC\_line, and CD\_line objects are made. These lines help traders analyze possible price changes by outlining the harmonic pattern. These lines' width and color settings are adjusted to improve visibility on the chart.

After successfully recognizing the XABCD pattern, it's critical to solve a few crucial concerns. The program's logic is that after identifying point X, it won't update X until the ABCD sequence is finished. This implies that the program won't change X if a new low that is lower than the originally determined X arises between X and A. Consequently, the pattern could not accurately reflect the real market structure.

We must put in place a system that continuously determines if X is still the lowest low between X and A to guarantee the pattern's validity. It is necessary to dynamically update X if a lower low occurs within this range. To ensure that no higher points are overlooked in this segment, A should likewise be the highest high between A and B. Similarly, C ought to be the highest high between C and D, and B ought to be the lowest low between B and C. We can improve pattern detection accuracy and avoid misidentifications brought on by strict point selection by following these requirements. This method preserves the integrity of the harmonic pattern structure while enabling the indication to stay adaptable and responsive to fresh price activity.

Example:

```
//X
int x_a_bars; // Number of bars between XA
int x_lowest_index; // Index of the lowest bar
double x_a_ll; // Price of the lowest bar
datetime x_a_ll_t; // Time of the lowest bar

//A
int a_b_bars; // Number of bars between AB
int a_highest_index; // Index of the highest bar
double a_b_hh; // Price of the highest bar
datetime a_b_hh_t; // Time of the highest bar

//B
int b_c_bars; // Number of bars between BC
int b_lowest_index; // Index of the lowest bar
double b_c_ll; // Price of the lowest bar
datetime b_c_ll_t; // Time of the lowest bar

//C
int c_d_bars; // Number of bars between CD
int c_highest_index; // Index of the highest bar
double c_d_hh; // Price of the highest bar
datetime c_d_hh_t; // Time of the highest bar

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

//---

   if(rates_total >= bars_check)
     {

      for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
        {
         if(IsSwingLow(low, i, LookbackBars))
           {
            // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
            X = low[i]; // Price of the swing low (X).
            X_time = time[i]; // Time of the swing low (X).
            X_line = StringFormat("XLow%d", i); // Unique name for the trend line object.
            X_letter = StringFormat("X%d", i); // Unique name for the text label object.

            for(int j = i; j < rates_total - LookbackBars; j++)
              {
               if(IsSwingHigh(high, j, LookbackBars) && time[j] > X_time)
                 {

                  A = high[j]; // Price of the swing high (A).
                  A_time = time[j]; // Time of the swing high (A).
                  A_line = StringFormat("AHigh%d", j); // Unique name for the trend line object.
                  A_letter = StringFormat("A%d", j); // Unique name for the text label object.

                  for(int k = j; k < rates_total - LookbackBars; k++)
                    {
                     if(IsSwingLow(low, k, LookbackBars) && time[k] > A_time)
                       {

                        // If a swing low is found, store its price, time, and create a name for the object.
                        B = low[k]; // Price of the swing low (B).
                        B_time = time[k]; // Time of the swing low (B).
                        B_line = StringFormat("BLow%d", k); // Unique name for the trend line object.
                        B_letter = StringFormat("B%d", k); // Unique name for the text label object.

                        for(int l = j ; l < rates_total - LookbackBars; l++)
                          {
                           if(IsSwingHigh(high, l, LookbackBars) && time[l] > B_time)
                             {
                              C = high[l]; // Price of the swing high (C).
                              C_time = time[l]; // Time of the swing high (C).
                              C_line = StringFormat("CHigh%d", l); // Unique name for the trend line object.
                              C_letter = StringFormat("C%d", l); // Unique name for the text label object.

                              for(int m = l; m < rates_total - (LookbackBars / 2); m++)
                                {
                                 if(IsSwingLow(low, m, LookbackBars / 2) && time[m] > C_time)
                                   {
                                    D = low[m]; // Price of the swing low (D).
                                    D_time = time[m]; // Time of the swing low (D).
                                    D_line = StringFormat("DLow%d", m); // Unique name for the trend line object.
                                    D_letter = StringFormat("D%d", m); // Unique name for the text label object.

                                    //D
                                    ObjectCreate(chart_id, D_letter, OBJ_TEXT, 0, D_time, D); // Create text object for D
                                    ObjectSetString(chart_id, D_letter, OBJPROP_TEXT, "D"); // Set the text to "D".
                                    ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,D_line,OBJ_TREND,0,D_time,D,time[m+LookbackBars],D); // Create line to mark D
                                    ObjectSetInteger(chart_id,D_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

                                    //C
                                    c_d_bars = Bars(_Symbol,PERIOD_CURRENT,C_time, D_time);
                                    c_highest_index = ArrayMaximum(high,l, c_d_bars);

                                    c_d_hh = high[c_highest_index]; //C - D Highest High and time
                                    c_d_hh_t = time[c_highest_index];

                                    ObjectCreate(chart_id, C_letter, OBJ_TEXT, 0, c_d_hh_t, c_d_hh); // Create text object for C
                                    ObjectSetString(chart_id, C_letter, OBJPROP_TEXT, "C"); // Set the text to "C".
                                    ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,C_line,OBJ_TREND,0,c_d_hh_t,C,time[c_highest_index+LookbackBars],c_d_hh); // Create line to mark C
                                    ObjectSetInteger(chart_id,C_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

                                    //B
                                    b_c_bars = Bars(_Symbol,PERIOD_CURRENT,B_time, c_d_hh_t);
                                    b_lowest_index = ArrayMinimum(low,k, b_c_bars);

                                    b_c_ll = low[b_lowest_index]; //B - C Lowest Low and time
                                    b_c_ll_t = time[b_lowest_index];

                                    ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, b_c_ll_t, b_c_ll); // Create text object for B
                                    ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
                                    ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,B_line,OBJ_TREND,0,b_c_ll_t,b_c_ll,time[b_lowest_index+LookbackBars],b_c_ll); // Create line to mark B
                                    ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to MidnightBlue

                                    //A
                                    a_b_bars = Bars(_Symbol,PERIOD_CURRENT,A_time, b_c_ll_t);
                                    a_highest_index = ArrayMaximum(high,j, a_b_bars);

                                    a_b_hh = high[a_highest_index]; //A - B Highest High and time
                                    a_b_hh_t = time[a_highest_index];

                                    ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, a_b_hh_t, a_b_hh); // Create text object for A
                                    ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
                                    ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,A_line,OBJ_TREND,0,a_b_hh_t,a_b_hh,time[a_highest_index+LookbackBars],a_b_hh); // Create line to mark A
                                    ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

                                    //X
                                    x_a_bars = Bars(_Symbol,PERIOD_CURRENT,X_time, a_b_hh_t);
                                    x_lowest_index = ArrayMinimum(low,i, x_a_bars);

                                    x_a_ll = low[x_lowest_index]; //X - A Lowest Low and time
                                    x_a_ll_t = time[x_lowest_index];

                                    ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, x_a_ll_t, x_a_ll); // Create text object for X
                                    ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
                                    ObjectCreate(chart_id,X_line,OBJ_TREND,0,x_a_ll_t,x_a_ll,time[x_lowest_index+LookbackBars],x_a_ll); // Create line to mark X

                                    XA_line = StringFormat("XA Line%d", m); // Unique name for the XA line.
                                    ObjectCreate(chart_id,XA_line,OBJ_TREND,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh); // Create line to connect XA
                                    ObjectSetInteger(chart_id,XA_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,XA_line,OBJPROP_WIDTH,3); // Line width

                                    AB_line = StringFormat("AB Line%d", m); // Unique name for the AB line.
                                    ObjectCreate(chart_id,AB_line,OBJ_TREND,0,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll); // Create line to connect AB
                                    ObjectSetInteger(chart_id,AB_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,AB_line,OBJPROP_WIDTH,3); // Line width

                                    BC_line = StringFormat("BC Line%d", m); // Unique name for the BC line.
                                    ObjectCreate(chart_id,BC_line,OBJ_TREND,0,b_c_ll_t,b_c_ll,c_d_hh_t,c_d_hh); // Create line to connect BC
                                    ObjectSetInteger(chart_id,BC_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,BC_line,OBJPROP_WIDTH,3); // Line width

                                    CD_line = StringFormat("CD Line%d", m); // Unique name for the CD line.
                                    ObjectCreate(chart_id,CD_line,OBJ_TREND,0,c_d_hh_t,c_d_hh,D_time,D); // Create line to connect CD
                                    ObjectSetInteger(chart_id,CD_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,CD_line,OBJPROP_WIDTH,3); // Line width

                                    if(overlap == false)
                                      {
                                       i = m;
                                      }
                                    if(overlap == true)
                                      {
                                       i = i;
                                      }
                                    break;

                                   }
                                }

                              break;

                             }
                          }

                        break;

                       }
                    }

                  break;
                 }
              }
           }
        }
     }

//--- return value of prev_calculated for next call
   return(rates_total);
  }
```

Output:

![Figure 10. New XABCD](https://c.mql5.com/2/127/figure_10.png)

Explanation:

The X point serves as the initial point of reference for spotting possible reversal patterns in harmonic pattern trading. In a bullish pattern, it signifies a large swing low; in a bearish pattern, it signifies a swing high. Because it lays the groundwork for measuring Fibonacci retracements and extensions — which aid in confirming harmonic formations — this point is essential. The purpose of the code is to recognize X, record its time and price, and visually indicate it on the chart. To create the entire pattern structure, it also joins X to the other swing points (A, B, C, and D).

A number of important variables are utilized to store data about X. The number of bars between X and A is recorded by the variable x\_a\_bars, which aids in determining the relative separations between pattern legs. To pinpoint the precise bar where X occurred, the variable x\_lowest\_index maintains track of the array index of the lowest price inside this range. The timestamp associated with the lowest price is saved in x\_a\_ll\_t, while the price itself is assigned to x\_a\_ll.

Finding the number of bars between X and A using the Bars() function is the first step toward determining X. The range required for analysis is provided by this function, which counts the number of bars between these two swing points. The program utilizes ArrayMinimum() to search through the designated bars and return the index of the lowest price when the range has been determined. Then, using the low\[x\_lowest\_index\] and time\[x\_lowest\_index\] arrays, respectively, the real price and time of X are obtained.

Once X has been identified, the script uses MetaTrader's object to visibly indicate it on the chart. To make sure you can quickly identify this crucial moment, ObjectCreate() is used to produce a text label (X\_letter) at the X swing low. For clarity, the text "X" is given to the text label, which is placed at the time and price of X. To adequately highlight the swing low, a trend line (X\_line) is also formed at X using ObjectCreate(). The line is extended for a predetermined number of bars (LookbackBars).

The program creates trend lines to link X to other important locations (A, B, C, and D) after X has been determined. You can better see the harmonic pattern with these trend lines. The ObjectCreate() function is used to draw the first connection, the XA line, which connects X and A. This line has a width of three pixels and is colored brown (clrSaddleBrown) to guarantee visibility. The remaining pattern legs — the AB line, which connects A to B; the BC line, which connects B to C; and the CD line, which connects C to D — are similarly depicted with extra trend lines. To produce a unified visual representation, each of these lines keeps the same width and color characteristics.

**2.2. Validating Point B Using Fibonacci Retracements of XA**

We will verify swing low B in this section by seeing if it stays above the 78.6% level and drops below the 61.8% retracement level of XA. Because it guarantees that point B complies with the necessary Fibonacci ratios, which aid in defining the pattern's structure, this validation phase is essential in bullish pattern trading. There is a greater chance of a legitimate harmonic arrangement if B is in this range. Otherwise, the pattern might not be genuine or need more confirmation if B is too high or too low.

We first determine the Fibonacci retracement levels for the XA leg to carry out this validation. A popular technical analysis method for spotting possible price action reversals is the Fibonacci retracement.

We will use the graphical tools in MQL5 to add a Fibonacci retracement object to the chart depiction in addition to validating B. The Fibonacci tool will be plotted using the ObjectCreate() function, with X serving as the starting point and A as the final point. Once the object has been created, we will modify it by including additional Fibonacci levels that MetaTrader 5 does not by default support. This guarantees the display of all pertinent retracement zones.

Example:

```
//Fibo
string XA_fibo; // Unique name for XA fibo
double lvl_61_8; // Level 61.8
double lvl_78_6; // Level 78.6
string fibo_618_786; // Unique name for the object that marks 61.8 and 78.6.
string fibo_78_6_txt; // Unique name for text "78.6" cause its not available by default
```

```
for(int m = l; m < rates_total - (LookbackBars / 2); m++)
  {
   if(IsSwingLow(low, m, LookbackBars / 2) && time[m] > C_time)
     {
      D = low[m]; // Price of the swing low (D).
      D_time = time[m]; // Time of the swing low (D).
      D_line = StringFormat("DLow%d", m); // Unique name for the trend line object.
      D_letter = StringFormat("D%d", m); // Unique name for the text label object.

      //D
      ObjectCreate(chart_id, D_letter, OBJ_TEXT, 0, D_time, D); // Create text object for D
      ObjectSetString(chart_id, D_letter, OBJPROP_TEXT, "D"); // Set the text to "D".
      ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
      ObjectCreate(chart_id,D_line,OBJ_TREND,0,D_time,D,time[m+LookbackBars],D); // Create line to mark D
      ObjectSetInteger(chart_id,D_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

      //C
      c_d_bars = Bars(_Symbol,PERIOD_CURRENT,C_time, D_time);
      c_highest_index = ArrayMaximum(high,l, c_d_bars);

      c_d_hh = high[c_highest_index]; //C - D Highest High and time
      c_d_hh_t = time[c_highest_index];

      ObjectCreate(chart_id, C_letter, OBJ_TEXT, 0, c_d_hh_t, c_d_hh); // Create text object for C
      ObjectSetString(chart_id, C_letter, OBJPROP_TEXT, "C"); // Set the text to "C".
      ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
      ObjectCreate(chart_id,C_line,OBJ_TREND,0,c_d_hh_t,C,time[c_highest_index+LookbackBars],c_d_hh); // Create line to mark C
      ObjectSetInteger(chart_id,C_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

      //B
      b_c_bars = Bars(_Symbol,PERIOD_CURRENT,B_time, c_d_hh_t);
      b_lowest_index = ArrayMinimum(low,k, b_c_bars);

      b_c_ll = low[b_lowest_index]; //B - C Lowest Low and time
      b_c_ll_t = time[b_lowest_index];

      ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, b_c_ll_t, b_c_ll); // Create text object for B
      ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
      ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
      ObjectCreate(chart_id,B_line,OBJ_TREND,0,b_c_ll_t,b_c_ll,time[b_lowest_index+LookbackBars],b_c_ll); // Create line to mark B
      ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to MidnightBlue

      //A
      a_b_bars = Bars(_Symbol,PERIOD_CURRENT,A_time, b_c_ll_t);
      a_highest_index = ArrayMaximum(high,j, a_b_bars);

      a_b_hh = high[a_highest_index]; //A - B Highest High and time
      a_b_hh_t = time[a_highest_index];

      ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, a_b_hh_t, a_b_hh); // Create text object for A
      ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
      ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
      ObjectCreate(chart_id,A_line,OBJ_TREND,0,a_b_hh_t,a_b_hh,time[a_highest_index+LookbackBars],a_b_hh); // Create line to mark A
      ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

      //X
      x_a_bars = Bars(_Symbol,PERIOD_CURRENT,X_time, a_b_hh_t);
      x_lowest_index = ArrayMinimum(low,i, x_a_bars);

      x_a_ll = low[x_lowest_index]; //X - A Lowest Low and time
      x_a_ll_t = time[x_lowest_index];

      ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, x_a_ll_t, x_a_ll); // Create text object for X
      ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
      ObjectCreate(chart_id,X_line,OBJ_TREND,0,x_a_ll_t,x_a_ll,time[x_lowest_index+LookbackBars],x_a_ll); // Create line to mark X

      XA_line = StringFormat("XA Line%d", m); // Unique name for the XA line.
      ObjectCreate(chart_id,XA_line,OBJ_TREND,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh); // Create line to connect XA
      ObjectSetInteger(chart_id,XA_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
      ObjectSetInteger(chart_id,XA_line,OBJPROP_WIDTH,3); // Line width

      AB_line = StringFormat("AB Line%d", m); // Unique name for the AB line.
      ObjectCreate(chart_id,AB_line,OBJ_TREND,0,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll); // Create line to connect AB
      ObjectSetInteger(chart_id,AB_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
      ObjectSetInteger(chart_id,AB_line,OBJPROP_WIDTH,3); // Line width

      BC_line = StringFormat("BC Line%d", m); // Unique name for the BC line.
      ObjectCreate(chart_id,BC_line,OBJ_TREND,0,b_c_ll_t,b_c_ll,c_d_hh_t,c_d_hh); // Create line to connect BC
      ObjectSetInteger(chart_id,BC_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
      ObjectSetInteger(chart_id,BC_line,OBJPROP_WIDTH,3); // Line width

      CD_line = StringFormat("CD Line%d", m); // Unique name for the CD line.
      ObjectCreate(chart_id,CD_line,OBJ_TREND,0,c_d_hh_t,c_d_hh,D_time,D); // Create line to connect CD
      ObjectSetInteger(chart_id,CD_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
      ObjectSetInteger(chart_id,CD_line,OBJPROP_WIDTH,3); // Line width

      lvl_61_8 = a_b_hh - ((61.8/100) * (a_b_hh - x_a_ll)); // Calculating level 61.8
      lvl_78_6 = a_b_hh - ((78.6/100) * (a_b_hh - x_a_ll)); // Calculating level 78.6

      //XA FIBO
      XA_fibo = StringFormat("XA FIB0 %d", i);
      ObjectCreate(chart_id,XA_fibo,OBJ_FIBO,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh); // Create XA fibo
      ObjectSetInteger(chart_id,XA_fibo,OBJPROP_LEVELS,6); // Number of default levels
      for(int i = 1; i <= 6; i++)
        {

         ObjectSetInteger(chart_id,XA_fibo,OBJPROP_LEVELCOLOR,i,clrSaddleBrown); // Set each level color to SaddleBrown

        }

      fibo_618_786 = StringFormat("Fibo 78.6 - 61.8 %d", i);
      fibo_78_6_txt = StringFormat("Fibo 78.6 Text %d", i);

      ObjectCreate(chart_id,fibo_618_786,OBJ_RECTANGLE,0,x_a_ll_t, lvl_61_8,b_c_ll_t,lvl_78_6); // Create rectangle object marking 61.8 and 78.6
      ObjectSetInteger(chart_id,fibo_618_786,OBJPROP_COLOR,clrSaddleBrown); // Set color to clrSaddleBrown

      ObjectCreate(chart_id,fibo_78_6_txt,OBJ_TEXT,0,b_c_ll_t,lvl_78_6); // Create text for level 78.6
      ObjectSetString(chart_id,fibo_78_6_txt,OBJPROP_TEXT,"78.6"); // Text to be displayed
      ObjectSetInteger(chart_id,fibo_78_6_txt,OBJPROP_COLOR,clrSaddleBrown); // Set text color
      ObjectSetInteger(chart_id,fibo_78_6_txt,OBJPROP_FONTSIZE,8); // Set text size

      if(overlap == false)
        {
         i = m;
        }
      if(overlap == true)
        {
         i = i;
        }
      break;

     }
  }
```

Output:

![Figure 11. Fibo](https://c.mql5.com/2/127/figure_11.png)

Explanation:

To confirm that point B is located inside a given range, we apply Fibonacci retracement, for the XA leg in this section. In echnical analysis, the Fibonacci retracement is frequently used to pinpoint possible reversal zones. It is essential for verifying if point B corresponds with the anticipated retracement of XA. In particular, B should ideally be in the range of XA's 61.8% to 78.6% retracement levels. It will be simpler to determine whether point B is within the acceptable range if we build a Fibonacci retracement object and highlight this zone on the chart. We first establish a few important variables to accomplish this. To ensure that every Fibonacci drawing is correctly identified, the variable XA\_fibo keeps a unique name for the Fibonacci retracement object.

The Fibonacci retracement object on the XA leg is created using the ObjectCreate() function after the levels have been determined. You may understand how price interacts with Fibonacci levels by looking at this object, which spans from point X (low) to point A (high). We set the number of Fibonacci levels to six and changed their color to SaddleBrown to improve the visualization's clarity and make sure that each level is easily observable and distinct from other chart elements.

We further highlight the 61.8% to 78.6% retracement zone by making a rectangular object that draws attention to it because it is an important area. It is simple to determine whether point B is inside the expected range because the rectangle spans the 61.8% and 78.6% levels. This facilitates rapid pattern validation without requiring laborious retracement level checks. Additionally, the rectangle has the color SaddleBrown, which keeps the chart's visual concept consistent.

We also include a text label for the 78.6% retracement level, which is another significant change. We manually generate a text object to display "78.6" at the relevant price level because MetaTrader 5's Fibonacci tool by default does not contain this level. This increases the precision of identifying harmonic patterns by guaranteeing that traders can observe the precise location of the 78.6% retracement. We make it simpler to verify whether point B is legitimate inside the harmonic pattern by putting these changes into practice. A precise and visually intuitive technique for validating harmonic patterns is ensured by the Fibonacci retracement object, the marked 61.8% – 78.6% zone, and the extra text label.

We must make sure that all chart objects are only drawn when the requirements for a valid pattern are satisfied now that we have everything we need. In particular, point D must be below point X, and point B must be inside the 61.8% to 78.6% retracement of XA. No objects should be added to the chart if these requirements are not met.

Examples:

```
for(int m = l; m < rates_total - (LookbackBars / 2); m++)
  {
   if(IsSwingLow(low, m, LookbackBars / 2) && time[m] > C_time)
     {
      D = low[m]; // Price of the swing low (D).
      D_time = time[m]; // Time of the swing low (D).
      D_line = StringFormat("DLow%d", m); // Unique name for the trend line object.
      D_letter = StringFormat("D%d", m); // Unique name for the text label object.

      //C
      c_d_bars = Bars(_Symbol,PERIOD_CURRENT,C_time, D_time);
      c_highest_index = ArrayMaximum(high,l, c_d_bars);

      c_d_hh = high[c_highest_index]; //C - D Highest High and time
      c_d_hh_t = time[c_highest_index];

      //B
      b_c_bars = Bars(_Symbol,PERIOD_CURRENT,B_time, c_d_hh_t);
      b_lowest_index = ArrayMinimum(low,k, b_c_bars);

      b_c_ll = low[b_lowest_index]; //B - C Lowest Low and time
      b_c_ll_t = time[b_lowest_index];

      //A
      a_b_bars = Bars(_Symbol,PERIOD_CURRENT,A_time, b_c_ll_t);
      a_highest_index = ArrayMaximum(high,j, a_b_bars);

      a_b_hh = high[a_highest_index]; //A - B Highest High and time
      a_b_hh_t = time[a_highest_index];

      //X
      x_a_bars = Bars(_Symbol,PERIOD_CURRENT,X_time, a_b_hh_t);
      x_lowest_index = ArrayMinimum(low,i, x_a_bars);

      x_a_ll = low[x_lowest_index]; //X - A Lowest Low and time
      x_a_ll_t = time[x_lowest_index];

      lvl_61_8 = a_b_hh - ((61.8/100) * (a_b_hh - x_a_ll)); // Calculating level 61.8
      lvl_78_6 = a_b_hh - ((78.6/100) * (a_b_hh - x_a_ll)); // Calculating level 78.6

      if((b_c_ll <= lvl_61_8 && b_c_ll >= lvl_78_6) && (D < x_a_ll))
        {

         //D
         ObjectCreate(chart_id, D_letter, OBJ_TEXT, 0, D_time, D); // Create text object for D
         ObjectSetString(chart_id, D_letter, OBJPROP_TEXT, "D"); // Set the text to "D".
         ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
         ObjectCreate(chart_id,D_line,OBJ_TREND,0,D_time,D,time[m+LookbackBars],D); // Create line to mark D
         ObjectSetInteger(chart_id,D_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

         //C
         ObjectCreate(chart_id, C_letter, OBJ_TEXT, 0, c_d_hh_t, c_d_hh); // Create text object for C
         ObjectSetString(chart_id, C_letter, OBJPROP_TEXT, "C"); // Set the text to "C".
         ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
         ObjectCreate(chart_id,C_line,OBJ_TREND,0,c_d_hh_t,C,time[c_highest_index+LookbackBars],c_d_hh); // Create line to mark C
         ObjectSetInteger(chart_id,C_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

         //B
         ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, b_c_ll_t, b_c_ll); // Create text object for B
         ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
         ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
         ObjectCreate(chart_id,B_line,OBJ_TREND,0,b_c_ll_t,b_c_ll,time[b_lowest_index+LookbackBars],b_c_ll); // Create line to mark B
         ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to MidnightBlue

         //A
         ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, a_b_hh_t, a_b_hh); // Create text object for A
         ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
         ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
         ObjectCreate(chart_id,A_line,OBJ_TREND,0,a_b_hh_t,a_b_hh,time[a_highest_index+LookbackBars],a_b_hh); // Create line to mark A
         ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

         //X
         ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, x_a_ll_t, x_a_ll); // Create text object for X
         ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
         ObjectCreate(chart_id,X_line,OBJ_TREND,0,x_a_ll_t,x_a_ll,time[x_lowest_index+LookbackBars],x_a_ll); // Create line to mark X

         XA_line = StringFormat("XA Line%d", m); // Unique name for the XA line.
         ObjectCreate(chart_id,XA_line,OBJ_TREND,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh); // Create line to connect XA
         ObjectSetInteger(chart_id,XA_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
         ObjectSetInteger(chart_id,XA_line,OBJPROP_WIDTH,3); // Line width

         AB_line = StringFormat("AB Line%d", m); // Unique name for the AB line.
         ObjectCreate(chart_id,AB_line,OBJ_TREND,0,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll); // Create line to connect AB
         ObjectSetInteger(chart_id,AB_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
         ObjectSetInteger(chart_id,AB_line,OBJPROP_WIDTH,3); // Line width

         BC_line = StringFormat("BC Line%d", m); // Unique name for the BC line.
         ObjectCreate(chart_id,BC_line,OBJ_TREND,0,b_c_ll_t,b_c_ll,c_d_hh_t,c_d_hh); // Create line to connect BC
         ObjectSetInteger(chart_id,BC_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
         ObjectSetInteger(chart_id,BC_line,OBJPROP_WIDTH,3); // Line width

         CD_line = StringFormat("CD Line%d", m); // Unique name for the CD line.
         ObjectCreate(chart_id,CD_line,OBJ_TREND,0,c_d_hh_t,c_d_hh,D_time,D); // Create line to connect CD
         ObjectSetInteger(chart_id,CD_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
         ObjectSetInteger(chart_id,CD_line,OBJPROP_WIDTH,3); // Line width

         //XA FIBO
         XA_fibo = StringFormat("XA FIB0 %d", i);
         ObjectCreate(chart_id,XA_fibo,OBJ_FIBO,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh); // Create XA fibo
         ObjectSetInteger(chart_id,XA_fibo,OBJPROP_LEVELS,6); // Number of default levels
         for(int i = 1; i <= 6; i++)
           {
            ObjectSetInteger(chart_id,XA_fibo,OBJPROP_LEVELCOLOR,i,clrSaddleBrown); // Set each level color to SaddleBrown
           }

         fibo_618_786 = StringFormat("Fibo 78.6 - 61.8 %d", i);
         fibo_78_6_txt = StringFormat("Fibo 78.6 Text %d", i);

         ObjectCreate(chart_id,fibo_618_786,OBJ_RECTANGLE,0,x_a_ll_t, lvl_61_8,b_c_ll_t,lvl_78_6); // Create rectangle object marking 61.8 and 78.6
         ObjectSetInteger(chart_id,fibo_618_786,OBJPROP_COLOR,clrSaddleBrown); // Set color to clrSaddleBrown

         ObjectCreate(chart_id,fibo_78_6_txt,OBJ_TEXT,0,b_c_ll_t,lvl_78_6); // Create text for level 78.6
         ObjectSetString(chart_id,fibo_78_6_txt,OBJPROP_TEXT,"78.6"); // Text to be displayed
         ObjectSetInteger(chart_id,fibo_78_6_txt,OBJPROP_COLOR,clrSaddleBrown); // Set text color
         ObjectSetInteger(chart_id,fibo_78_6_txt,OBJPROP_FONTSIZE,8); // Set text size

        }

      if(overlap == false)
        {
         i = m;
        }
      if(overlap == true)
        {
         i = i;
        }
      break;
     }
  }
```

Output:

![Figure 12. B and X](https://c.mql5.com/2/127/Figure_12.png)

Explanation:

By evaluating to true, the condition if ((b\_c\_ll <= lvl\_61\_8 && b\_c\_ll >= lvl\_78\_6) && (D < x\_a\_ll)) guarantees that all chart objects are only drawn when a proper harmonic pattern is found. This keeps extraneous items from overcrowding the chart with erroneous patterns. The script then creates and shows all required components, such as the Fibonacci retracement levels, trend lines joining X, A, B, C, and D, and text labels identifying each important point, when the requirement is met. By ensuring that the indicator only displays correctly formed patterns, this method improves accuracy and preserves a clear, informative chart display.

**2.3. Visualizing the XAB and BCD Triangles**

To better visualize the harmonic pattern, we shall mark the XAB and BCD forms with triangles in this section. As visual assistance, these triangles will make it simpler to recognize and examine the pattern structure on the chart.

This will be accomplished by connecting the points X, A, and B for the first triangle and B, C, and D for the second triangle using OBJ\_TRIANGLE objects. The BCD triangle will depict the last leg of the pattern, strengthening the structure that leads to point D, the possible reversal zone, while the XAB triangle will emphasize the first leg, confirming the relationship between X, A, and B.

Example:

```
string X_A_B;
string B_C_D;
if((b_c_ll <= lvl_61_8 && b_c_ll >= lvl_78_6) && (D < x_a_ll))
  {

//D
   ObjectCreate(chart_id, D_letter, OBJ_TEXT, 0, D_time, D); // Create text object for D
   ObjectSetString(chart_id, D_letter, OBJPROP_TEXT, "D"); // Set the text to "D".
   ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
   ObjectCreate(chart_id,D_line,OBJ_TREND,0,D_time,D,time[m+LookbackBars],D); // Create line to mark D
   ObjectSetInteger(chart_id,D_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

//C
   ObjectCreate(chart_id, C_letter, OBJ_TEXT, 0, c_d_hh_t, c_d_hh); // Create text object for C
   ObjectSetString(chart_id, C_letter, OBJPROP_TEXT, "C"); // Set the text to "C".
   ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
   ObjectCreate(chart_id,C_line,OBJ_TREND,0,c_d_hh_t,C,time[c_highest_index+LookbackBars],c_d_hh); // Create line to mark C
   ObjectSetInteger(chart_id,C_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

//B
   ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, b_c_ll_t, b_c_ll); // Create text object for B
   ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
   ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
   ObjectCreate(chart_id,B_line,OBJ_TREND,0,b_c_ll_t,b_c_ll,time[b_lowest_index+LookbackBars],b_c_ll); // Create line to mark B
   ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to MidnightBlue

//A
   ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, a_b_hh_t, a_b_hh); // Create text object for A
   ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
   ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
   ObjectCreate(chart_id,A_line,OBJ_TREND,0,a_b_hh_t,a_b_hh,time[a_highest_index+LookbackBars],a_b_hh); // Create line to mark A
   ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

//X
   ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, x_a_ll_t, x_a_ll); // Create text object for X
   ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
   ObjectCreate(chart_id,X_line,OBJ_TREND,0,x_a_ll_t,x_a_ll,time[x_lowest_index+LookbackBars],x_a_ll); // Create line to mark X

   XA_line = StringFormat("XA Line%d", m); // Unique name for the XA line.
   ObjectCreate(chart_id,XA_line,OBJ_TREND,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh); // Create line to connect XA
   ObjectSetInteger(chart_id,XA_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
   ObjectSetInteger(chart_id,XA_line,OBJPROP_WIDTH,3); // Line width

   AB_line = StringFormat("AB Line%d", m); // Unique name for the AB line.
   ObjectCreate(chart_id,AB_line,OBJ_TREND,0,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll); // Create line to connect AB
   ObjectSetInteger(chart_id,AB_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
   ObjectSetInteger(chart_id,AB_line,OBJPROP_WIDTH,3); // Line width

   BC_line = StringFormat("BC Line%d", m); // Unique name for the BC line.
   ObjectCreate(chart_id,BC_line,OBJ_TREND,0,b_c_ll_t,b_c_ll,c_d_hh_t,c_d_hh); // Create line to connect BC
   ObjectSetInteger(chart_id,BC_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
   ObjectSetInteger(chart_id,BC_line,OBJPROP_WIDTH,3); // Line width

   CD_line = StringFormat("CD Line%d", m); // Unique name for the CD line.
   ObjectCreate(chart_id,CD_line,OBJ_TREND,0,c_d_hh_t,c_d_hh,D_time,D); // Create line to connect CD
   ObjectSetInteger(chart_id,CD_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
   ObjectSetInteger(chart_id,CD_line,OBJPROP_WIDTH,3); // Line width

//XA FIBO
   XA_fibo = StringFormat("XA FIB0 %d", i);
   ObjectCreate(chart_id,XA_fibo,OBJ_FIBO,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh); // Create XA fibo
   ObjectSetInteger(chart_id,XA_fibo,OBJPROP_LEVELS,6); // Number of default levels
   for(int i = 1; i <= 6; i++)
     {
      ObjectSetInteger(chart_id,XA_fibo,OBJPROP_LEVELCOLOR,i,clrSaddleBrown); // Set each level color to SaddleBrown
     }

   fibo_618_786 = StringFormat("Fibo 78.6 - 61.8 %d", i);
   fibo_78_6_txt = StringFormat("Fibo 78.6 Text %d", i);

   ObjectCreate(chart_id,fibo_618_786,OBJ_RECTANGLE,0,x_a_ll_t, lvl_61_8,b_c_ll_t,lvl_78_6); // Create rectangle object marking 61.8 and 78.6
   ObjectSetInteger(chart_id,fibo_618_786,OBJPROP_COLOR,clrSaddleBrown); // Set color to clrSaddleBrown

   ObjectCreate(chart_id,fibo_78_6_txt,OBJ_TEXT,0,b_c_ll_t,lvl_78_6); // Create text for level 78.6
   ObjectSetString(chart_id,fibo_78_6_txt,OBJPROP_TEXT,"78.6"); // Text to be displayed
   ObjectSetInteger(chart_id,fibo_78_6_txt,OBJPROP_COLOR,clrSaddleBrown); // Set text color
   ObjectSetInteger(chart_id,fibo_78_6_txt,OBJPROP_FONTSIZE,8); // Set text size

   X_A_B = StringFormat("XAB %d", i);
   ObjectCreate(chart_id,X_A_B,OBJ_TRIANGLE,0,x_a_ll_t,x_a_ll, a_b_hh_t, a_b_hh, b_c_ll_t, b_c_ll);
   ObjectSetInteger(chart_id,X_A_B,OBJPROP_COLOR,clrCornflowerBlue);
   ObjectSetInteger(chart_id,X_A_B,OBJPROP_FILL,true);

   B_C_D = StringFormat("BCD %d", i);
   ObjectCreate(chart_id,B_C_D,OBJ_TRIANGLE,0,b_c_ll_t,b_c_ll, c_d_hh_t, c_d_hh, D_time, D);
   ObjectSetInteger(chart_id,B_C_D,OBJPROP_COLOR,clrCornflowerBlue);
   ObjectSetInteger(chart_id,B_C_D,OBJPROP_FILL,true);
  }
```

Output:

![Figure 13. Bullish Pattern](https://c.mql5.com/2/127/Figure_13.png)

Explanation:

Unique names for the triangles that constitute the XAB and BCD segments of the harmonic pattern are stored in the X\_A\_B and B\_C\_D variables. Three points — X, A, and B —are used to generate the X\_A\_B triangle. These points are identified by their respective timestamps and price values (x\_a\_ll\_t, x\_a\_ll, a\_b\_hh\_t, a\_b\_hh, b\_c\_ll\_t, b\_c\_ll). The B\_C\_D triangle, which uses (b\_c\_ll\_t, b\_c\_ll, c\_d\_hh\_t, c\_d\_hh, D\_time, D), similarly joins B, C, and D. StringFormat() is used to give each triangle a unique name, allowing for the drawing of various patterns without naming problems.

ObjectSetInteger() is used to set the triangles' visual characteristics once they have been generated using ObjectCreate(). The OBJPROP\_FILL attribute is set to true, guaranteeing that the triangles are filled rather than outlined, and both triangles have the color Cornflower Blue (clrCornflowerBlue). By enhancing clarity and making it simpler to verify possible trading opportunities, these visual components assist traders in rapidly identifying the pattern structure.

**2.4. Indicating Entry Point, Stop Loss, and Take Profit**

In this section, we will set the entry point, stop loss (SL), and take profit (TP) in to identify the important trade levels for the discovered harmonic pattern. Point D, the pattern's lowest swing low, is where the stop loss is set. Because harmonic patterns predict price reversals, setting the stop loss at D minimizes possible losses by guaranteeing that the pattern is invalidated if the market climbs above this level.

The index of point D plus half of the LookbackBars is the entering point. This allows for a small lag following the identification of D, enabling price action confirmation before making a transaction. Since point C is the swing high before the anticipated upward movement, it serves as the take-profit level. Labels and horizontal lines for the SL, entry, and TP levels are made to represent these levels on the chart. These items ensure clarity when trading based on the pattern by assisting traders in rapidly identifying the trade setup.

Example:

```
//Signal
string buy_txt;
string sell_txt;
string entry_line;
string tp_line;
string sl_line;
string tp_txt;
string sl_txt;
```

```
buy_txt = StringFormat("Buy %d", i);

ObjectCreate(chart_id,buy_txt,OBJ_TEXT,0,time[m + (LookbackBars / 2)],open[m  + (LookbackBars / 2)]);
ObjectSetString(chart_id,buy_txt,OBJPROP_TEXT,"BUY");
ObjectSetInteger(chart_id,buy_txt,OBJPROP_COLOR,clrCornflowerBlue);
ObjectSetInteger(chart_id,buy_txt,OBJPROP_FONTSIZE,10);

entry_line = StringFormat("Buy Entry Line %d", i);
ObjectCreate(chart_id,entry_line,OBJ_TREND,0,time[m  + (LookbackBars / 2)],open[m  + (LookbackBars / 2)], c_d_hh_t,open[m  + (LookbackBars / 2)]);
ObjectSetInteger(chart_id,entry_line,OBJPROP_WIDTH,3);
ObjectSetInteger(chart_id,entry_line,OBJPROP_COLOR,clrCornflowerBlue);

sl_txt = StringFormat("Buy SL %d", i);
ObjectCreate(chart_id,sl_txt,OBJ_TEXT,0,time[m + (LookbackBars / 2)],D);
ObjectSetString(chart_id,sl_txt,OBJPROP_TEXT,"SL");
ObjectSetInteger(chart_id,sl_txt,OBJPROP_COLOR,clrCornflowerBlue);
ObjectSetInteger(chart_id,sl_txt,OBJPROP_FONTSIZE,10);

sl_line = StringFormat("Buy SL Line %d", i);
ObjectCreate(chart_id,sl_line,OBJ_TREND,0,time[m + (LookbackBars / 2)],D,c_d_hh_t,D);
ObjectSetInteger(chart_id,sl_line,OBJPROP_WIDTH,3);
ObjectSetInteger(chart_id,sl_line,OBJPROP_COLOR,clrCornflowerBlue);

tp_txt = StringFormat("Buy TP %d", i);
ObjectCreate(chart_id,tp_txt,OBJ_TEXT,0,time[m +(LookbackBars / 2)],c_d_hh);
ObjectSetString(chart_id,tp_txt,OBJPROP_TEXT,"TP");
ObjectSetInteger(chart_id,tp_txt,OBJPROP_COLOR,clrCornflowerBlue);
ObjectSetInteger(chart_id,tp_txt,OBJPROP_FONTSIZE,10);

tp_line = StringFormat("Buy TP Line %d", i);
ObjectCreate(chart_id,tp_line,OBJ_TREND,0,time[m + (LookbackBars / 2)],c_d_hh,c_d_hh_t,c_d_hh);
ObjectSetInteger(chart_id,tp_line,OBJPROP_WIDTH,3);
ObjectSetInteger(chart_id,tp_line,OBJPROP_COLOR,clrCornflowerBlue);
```

Output:

![Figure 14. Signals](https://c.mql5.com/2/127/Figure_14.png)

Explanation:

The buy signal, entry point, stop loss (SL), and take profit (TP) levels for the harmonic pattern are defined and created as visual objects on the chart in this section. To make sure the items are readily maintained and identified on the chart, the variables buy\_txt, sell\_txt, entry\_line, tp\_line, sl\_line, tp\_txt, and sl\_txt store unique names for the corresponding objects. At the entry point, the buy\_txt object is a text label that reads "BUY" to denote the direction of the trade. The trade setup is clearly visualized thanks to the entry\_line, a horizontal trend line that indicates the entry price at open\[m + (LookbackBars / 2)\].

Likewise, the stop loss label and its associated trend line are denoted by sl\_txt and sl\_line, respectively, and are both situated at D, the pattern's lowest swing low. The stop loss makes sure that the trade is void if the price rises above this threshold. Tp\_txt and tp\_line, which are situated at c\_d\_hh, the preceding swing high, indicate the take profit level. These graphical elements are set to a distinctive color (clrCornflowerBlue) so that you can quickly recognize important trading levels on the chart. These components support the development of an orderly and transparent trading strategy based on the harmonic pattern.

### **3\. Building Bearish Harmonic Patterns**

The same logic that we used to identify bullish harmonic patterns will be modified for bearish patterns in this part. There is no need to overemphasize every feature because this is only the bullish pattern's opposite. A swing high will now be represented by X, a swing low by A, a swing high by B, a swing low by C, and a final swing high by D. This is the main difference. This guarantees that the pattern accurately depicts a bearish scenario, indicating a possible time to sell.

The B point will still be verified using the Fibonacci retracement levels to make sure it is within the appropriate range in relation to the XA leg. Similar to this, the pattern will be delineated by trend lines, triangles, and other chart elements, but their locations will be modified to conform to the bearish structure. The entry will be set at the index of D plus half of the LookbackBars, the take-profit (TP) at point C, and the stop-loss (SL) at point D. The same logic holds true because this is really the bullish pattern's inverse; therefore we won't rehash the arguments in great length.

Example:

```
int x_highest_index;   // Stores the index of the highest price point (X) in the pattern
double x_a_hh;         // Holds the price value of the swing high at point X
datetime x_a_hh_t;     // Stores the timestamp of when the swing high at X occurred

int a_lowest_index;   // Stores the index of the lowest price point (A) in the pattern
double a_b_ll;        // Holds the price value of the swing low at point A
datetime a_b_ll_t;    // Stores the timestamp of when the swing low at A occurred

int b_highest_index;   // Stores the index of the highest price point (B) in the pattern
double b_c_hh;         // Holds the price value of the swing high at point B
datetime b_c_hh_t;     // Stores the timestamp of when the swing high at B occurred
int c_lowest_index;   // Stores the index of the lowest price point (C) in the pattern
double c_d_ll;        // Holds the price value of the swing low at point C
datetime c_d_ll_t;    // Stores the timestamp of when the swing low at C occurred
```

```
if(rates_total >= bars_check)
  {
   for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
     {
      if(IsSwingHigh(high, i, LookbackBars))
        {

         X = high[i];                   // Assign the highest price at index 'i' to X
         X_time = time[i];              // Assign the corresponding time value to X_time
         X_line = StringFormat("XHigh%d", i);  // Create a unique string identifier for the X-high trendline
         X_letter = StringFormat("XB%d", i);   // Create a unique string identifier for the X-high label

         for(int j = i; j < rates_total - LookbackBars; j++)
           {
            if(IsSwingLow(low, j, LookbackBars) && time[j] > X_time)
              {
               A = low[j];                    // Assign the lowest price at index 'j' to A
               A_time = time[j];               // Assign the corresponding time value to A_time
               A_line = StringFormat("ALow%d", j);  // Create a unique string identifier for the A-low trendline
               A_letter = StringFormat("AB%d", j);  // Create a unique string identifier for the A-low label

               for(int k = j; k < rates_total - LookbackBars; k++)
                 {
                  if(IsSwingHigh(high, k, LookbackBars) && time[k] > A_time)
                    {

                     B = high[k];                    // Assign the highest price at index 'k' to B
                     B_time = time[k];                // Assign the corresponding time value to B_time
                     B_line = StringFormat("BHigh%d", k);  // Create a unique string identifier for the B-high trendline
                     B_letter = StringFormat("BB%d", k);   // Create a unique string identifier for the B-high label

                     for(int l = k; l < rates_total - LookbackBars; l++)
                       {
                        if(IsSwingLow(low, l, LookbackBars) && time[l] > B_time)
                          {

                           C = low[l];                      // Assign the lowest price at index 'l' to C
                           C_time = time[l];                // Assign the corresponding time value to C_time
                           C_line = StringFormat("CLow%d", l);  // Create a unique string identifier for the C-low trendline
                           C_letter = StringFormat("CB%d", l);   // Create a unique string identifier for the C-low label

                           for(int m = l; m < rates_total - (LookbackBars / 2); m++)
                             {
                              if(IsSwingHigh(high, m, LookbackBars / 2) && time[m] > C_time)
                                {
                                 D = high[m];                      // Assign the highest price at index 'm' to D
                                 D_time = time[m];                 // Assign the corresponding time value to D_time
                                 D_line = StringFormat("DHigh%d", m);  // Create a unique string identifier for the D-high trendline
                                 D_letter = StringFormat("DB%d", m);   // Create a unique string identifier for the D-high label

                                 // C - D Segment: Find the lowest low between C and D
                                 c_d_bars = Bars(_Symbol, PERIOD_CURRENT, C_time, D_time); // Count the number of bars between C and D
                                 c_lowest_index = ArrayMinimum(low, l, c_d_bars); // Find the index of the lowest low in the range

                                 c_d_ll = low[c_lowest_index]; // Store the lowest low (C - D lowest point)
                                 c_d_ll_t = time[c_lowest_index]; // Store the corresponding time for C - D

                                 // B - C Segment: Find the highest high between B and C
                                 b_c_bars = Bars(_Symbol, PERIOD_CURRENT, B_time, c_d_ll_t); // Count the number of bars between B and C
                                 b_highest_index = ArrayMaximum(high, k, b_c_bars); // Find the index of the highest high in the range

                                 b_c_hh = high[b_highest_index]; // Store the highest high (B - C highest point)
                                 b_c_hh_t = time[b_highest_index]; // Store the corresponding time for B - C

                                 // A - B Segment: Find the lowest low between A and B
                                 a_b_bars = Bars(_Symbol, PERIOD_CURRENT, A_time, b_c_hh_t); // Count the number of bars between A and B
                                 a_lowest_index = ArrayMinimum(low, j, a_b_bars); // Find the index of the lowest low in the range

                                 a_b_ll = low[a_lowest_index]; // Store the lowest low (A - B lowest point)
                                 a_b_ll_t = time[a_lowest_index]; // Store the corresponding time for A - B

                                 // X - A Segment: Find the highest high between X and A
                                 x_a_bars = Bars(_Symbol, PERIOD_CURRENT, X_time, a_b_ll_t); // Count the number of bars between X and A
                                 x_highest_index = ArrayMaximum(high, i, x_a_bars); // Find the index of the highest high in the range

                                 x_a_hh = high[x_highest_index]; // Store the highest high (X - A highest point)
                                 x_a_hh_t = time[x_highest_index]; // Store the corresponding time for X - A

                                 // Fibonacci Retracement Levels: Calculate 61.8% and 78.6% retracement levels from X to A
                                 lvl_61_8 = a_b_ll + ((61.8 / 100) * (x_a_hh - a_b_ll)); // 61.8% retracement level
                                 lvl_78_6 = a_b_ll + ((78.6 / 100) * (x_a_hh - a_b_ll)); // 78.6% retracement level

                                 if((b_c_hh >= lvl_61_8 && b_c_hh <= lvl_78_6) && (D > x_a_hh))
                                   {
                                    //D
                                    ObjectCreate(chart_id, D_letter, OBJ_TEXT, 0, D_time, D); // Create text object for D
                                    ObjectSetString(chart_id, D_letter, OBJPROP_TEXT, "D"); // Set the text to "D".
                                    ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,D_line,OBJ_TREND,0,D_time,D,time[m+LookbackBars],D); // Create line to mark D
                                    ObjectSetInteger(chart_id,D_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

                                    //C
                                    ObjectCreate(chart_id, C_letter, OBJ_TEXT, 0, c_d_ll_t, c_d_ll); // Create text object for C
                                    ObjectSetString(chart_id, C_letter, OBJPROP_TEXT, "C"); // Set the text to "C".
                                    ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,C_line,OBJ_TREND,0,c_d_ll_t,C,time[c_lowest_index+LookbackBars],c_d_ll); // Create line to mark C
                                    ObjectSetInteger(chart_id,C_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown

                                    //B
                                    ObjectCreate(chart_id, B_letter, OBJ_TEXT, 0, b_c_hh_t, b_c_hh); // Create text object for B
                                    ObjectSetString(chart_id, B_letter, OBJPROP_TEXT, "B"); // Set the text to "B".
                                    ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,B_line,OBJ_TREND,0,b_c_hh_t,b_c_hh,time[b_highest_index+LookbackBars],b_c_hh); // Create line to mark B
                                    ObjectSetInteger(chart_id,B_line,OBJPROP_COLOR,clrMidnightBlue); // Set line color to MidnightBlue

                                    //A
                                    ObjectCreate(chart_id, A_letter, OBJ_TEXT, 0, a_b_ll_t, a_b_ll); // Create text object for A
                                    ObjectSetString(chart_id, A_letter, OBJPROP_TEXT, "A"); // Set the text to "A".
                                    ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrGreen); // Set text color to green
                                    ObjectCreate(chart_id,A_line,OBJ_TREND,0,a_b_ll_t,a_b_ll,time[a_lowest_index+LookbackBars],a_b_ll); // Create line to mark A
                                    ObjectSetInteger(chart_id,A_line,OBJPROP_COLOR,clrGreen); // Set line color to green

                                    //X
                                    ObjectCreate(chart_id, X_letter, OBJ_TEXT, 0, x_a_hh_t, x_a_hh); // Create text object for X
                                    ObjectSetString(chart_id, X_letter, OBJPROP_TEXT, "X"); // Set the text to "X".
                                    ObjectCreate(chart_id,X_line,OBJ_TREND,0,x_a_hh_t,x_a_hh,time[x_highest_index+LookbackBars],x_a_hh); // Create line to mark X

                                    XA_line = StringFormat("XA LineB%d", m); // Unique name for the XA line.
                                    ObjectCreate(chart_id,XA_line,OBJ_TREND,0,x_a_hh_t,x_a_hh,a_b_ll_t,a_b_ll); // Create line to connect XA
                                    ObjectSetInteger(chart_id,XA_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,XA_line,OBJPROP_WIDTH,3); // Line width

                                    AB_line = StringFormat("AB LineB%d", m); // Unique name for the AB line.
                                    ObjectCreate(chart_id,AB_line,OBJ_TREND,0,a_b_ll_t,a_b_ll,b_c_hh_t,b_c_hh); // Create line to connect AB
                                    ObjectSetInteger(chart_id,AB_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,AB_line,OBJPROP_WIDTH,3); // Line width

                                    BC_line = StringFormat("BC LineB%d", m); // Unique name for the BC line.
                                    ObjectCreate(chart_id,BC_line,OBJ_TREND,0,b_c_hh_t,b_c_hh,c_d_ll_t,c_d_ll); // Create line to connect BC
                                    ObjectSetInteger(chart_id,BC_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,BC_line,OBJPROP_WIDTH,3); // Line width

                                    CD_line = StringFormat("CD LineB%d", m); // Unique name for the CD line.
                                    ObjectCreate(chart_id,CD_line,OBJ_TREND,0,c_d_ll_t,c_d_ll,D_time,D); // Create line to connect CD
                                    ObjectSetInteger(chart_id,CD_line,OBJPROP_COLOR,clrSaddleBrown); // Set line color to SaddleBrown
                                    ObjectSetInteger(chart_id,CD_line,OBJPROP_WIDTH,3); // Line width

                                    //XA FIBO
                                    XA_fibo = StringFormat("XA FIB0 B%d", i);
                                    ObjectCreate(chart_id,XA_fibo,OBJ_FIBO,0,x_a_hh_t,x_a_hh,a_b_ll_t,a_b_ll); // Create XA fibo
                                    ObjectSetInteger(chart_id,XA_fibo,OBJPROP_LEVELS,6); // Number of default levels
                                    for(int i = 1; i <= 6; i++)
                                      {
                                       ObjectSetInteger(chart_id,XA_fibo,OBJPROP_LEVELCOLOR,i,clrSaddleBrown); // Set each level color to SaddleBrown
                                      }

                                    fibo_618_786 = StringFormat("Fibo 78.6 - 61.8 B%d", i);
                                    fibo_78_6_txt = StringFormat("Fibo 78.6 Text B%d", i);

                                    ObjectCreate(chart_id,fibo_618_786,OBJ_RECTANGLE,0,x_a_hh_t, lvl_61_8,b_c_hh_t,lvl_78_6); // Create rectangle object marking 61.8 and 78.6
                                    ObjectSetInteger(chart_id,fibo_618_786,OBJPROP_COLOR,clrSaddleBrown); // Set color to clrSaddleBrown

                                    ObjectCreate(chart_id,fibo_78_6_txt,OBJ_TEXT,0,b_c_hh_t,lvl_78_6); // Create text for level 78.6
                                    ObjectSetString(chart_id,fibo_78_6_txt,OBJPROP_TEXT,"78.6"); // Text to be displayed
                                    ObjectSetInteger(chart_id,fibo_78_6_txt,OBJPROP_COLOR,clrSaddleBrown); // Set text color
                                    ObjectSetInteger(chart_id,fibo_78_6_txt,OBJPROP_FONTSIZE,8); // Set text size
                                    // Create and format the first bearish harmonic pattern (XAB)
                                    X_A_B = StringFormat("XAB B%d", i);
                                    ObjectCreate(chart_id, X_A_B, OBJ_TRIANGLE, 0, x_a_hh_t, x_a_hh, a_b_ll_t, a_b_ll, b_c_hh_t, b_c_hh);
                                    ObjectSetInteger(chart_id, X_A_B, OBJPROP_COLOR, clrMistyRose); // Set color for the pattern
                                    ObjectSetInteger(chart_id, X_A_B, OBJPROP_FILL, true); // Fill the triangle shape

                                    // Create and format the second bearish harmonic pattern (BCD)
                                    B_C_D = StringFormat("BCD B%d", i);
                                    ObjectCreate(chart_id, B_C_D, OBJ_TRIANGLE, 0, b_c_hh_t, b_c_hh, c_d_ll_t, c_d_ll, D_time, D);
                                    ObjectSetInteger(chart_id, B_C_D, OBJPROP_COLOR, clrMistyRose);
                                    ObjectSetInteger(chart_id, B_C_D, OBJPROP_FILL, true);

                                    // Create and format the SELL text at the entry position
                                    sell_txt = StringFormat("Sell %d", i);
                                    ObjectCreate(chart_id, sell_txt, OBJ_TEXT, 0, time[m + (LookbackBars / 2)], open[m + (LookbackBars / 2)]);
                                    ObjectSetString(chart_id, sell_txt, OBJPROP_TEXT, "SELL");
                                    ObjectSetInteger(chart_id, sell_txt, OBJPROP_COLOR, clrMagenta); // Set text color
                                    ObjectSetInteger(chart_id, sell_txt, OBJPROP_FONTSIZE, 10); // Set font size

                                    // Create and format the SELL entry line
                                    entry_line = StringFormat("Sell Entry Line %d", i);
                                    ObjectCreate(chart_id, entry_line, OBJ_TREND, 0, time[m + (LookbackBars / 2)], open[m + (LookbackBars / 2)], c_d_ll_t, open[m + (LookbackBars / 2)]);
                                    ObjectSetInteger(chart_id, entry_line, OBJPROP_WIDTH, 3); // Set line thickness
                                    ObjectSetInteger(chart_id, entry_line, OBJPROP_COLOR, clrMagenta); // Set line color

                                    // Create and format the Stop Loss (SL) text
                                    sl_txt = StringFormat("Sell SL %d", i);
                                    ObjectCreate(chart_id, sl_txt, OBJ_TEXT, 0, time[m + (LookbackBars / 2)], D);
                                    ObjectSetString(chart_id, sl_txt, OBJPROP_TEXT, "SL");
                                    ObjectSetInteger(chart_id, sl_txt, OBJPROP_COLOR, clrMagenta);
                                    ObjectSetInteger(chart_id, sl_txt, OBJPROP_FONTSIZE, 10);

                                    // Create and format the Stop Loss (SL) line
                                    sl_line = StringFormat("Sell SL Line %d", i);
                                    ObjectCreate(chart_id, sl_line, OBJ_TREND, 0, time[m + (LookbackBars / 2)], D, c_d_ll_t, D);
                                    ObjectSetInteger(chart_id, sl_line, OBJPROP_WIDTH, 3);
                                    ObjectSetInteger(chart_id, sl_line, OBJPROP_COLOR, clrMagenta);

                                    // Create and format the Take Profit (TP) text
                                    tp_txt = StringFormat("Sell TP %d", i);
                                    ObjectCreate(chart_id, tp_txt, OBJ_TEXT, 0, time[m + (LookbackBars / 2)], c_d_ll);
                                    ObjectSetString(chart_id, tp_txt, OBJPROP_TEXT, "TP");
                                    ObjectSetInteger(chart_id, tp_txt, OBJPROP_COLOR, clrMagenta);
                                    ObjectSetInteger(chart_id, tp_txt, OBJPROP_FONTSIZE, 10);

                                    // Create and format the Take Profit (TP) line
                                    tp_line = StringFormat("Sell TP Line %d", i);
                                    ObjectCreate(chart_id, tp_line, OBJ_TREND, 0, time[m + (LookbackBars / 2)], c_d_ll, c_d_ll_t, c_d_ll);
                                    ObjectSetInteger(chart_id, tp_line, OBJPROP_WIDTH, 3);
                                    ObjectSetInteger(chart_id, tp_line, OBJPROP_COLOR, clrMagenta);

                                    if(overlap == false)
                                      {
                                       i = m;
                                      }
                                    if(overlap == true)
                                      {
                                       i = i;
                                      }
                                    break;

                                   }

                                 break;

                                }
                             }

                           break;

                          }
                       }

                     break;

                    }
                 }

               break;

              }
           }

        }
     }
  }
```

Output:

![Figure 15. Bearish Pattern](https://c.mql5.com/2/127/Figure_15.png)

Explanation:

Finding a swing high at X, which signifies the start of the pattern, is the first step in the detection procedure. A swing low (A), a high (B), a low (C), and a high (D) are the next things it looks for. A legitimate pattern structure is ensured by verifying each point according to its relative position to the one before it. Validating the pattern is mostly dependent on the retracement levels, especially the 61.8% and 78.6% Fibonacci retracements from X to A. A possible SELL configuration is found if these requirements are satisfied and D is greater than X.

In-depth explanations won't be repeated because the bearish and bullish harmonic patterns are similar. The main distinction is that the bullish form of this pattern predicts a price gain, whereas this pattern predicts a price decrease. The pattern's orientation and the associated trading direction are the only differences. Otherwise, the fundamental ideas are the same.

### **Conclusion**

In this article, we explored how to build a Harmonic Patterns-like indicator using MetaTrader 5 chart objects. We covered the logic behind detecting key swing points, structuring the pattern, and validating it using Fibonacci retracement levels. By implementing both bullish and bearish patterns, we demonstrated how to identify potential trade opportunities based on price action. With this approach, you can visualize harmonic formations directly on their charts, allowing for better decision-making without relying on traditional indicator buffers. This method enhances flexibility and provides a solid foundation for further customization and refinement of pattern-based trading strategies.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17574.zip "Download all attachments in the single ZIP archive")

[Project8\_Harmonic\_Pattern\_Indicator.mq5](https://www.mql5.com/en/articles/download/17574/project8_harmonic_pattern_indicator.mq5 "Download Project8_Harmonic_Pattern_Indicator.mq5")(43.97 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/483826)**
(6)


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
1 Apr 2025 at 12:03

**Simon Simson [#](https://www.mql5.com/en/forum/483826#comment_56319748):**

Thank you so much for your good works on these MQL5 series.

Hello Simon.

Thank you for your kind words.

![Oluwatosin Mary Babalola](https://c.mql5.com/avatar/2024/2/65cfcb6a-f1c4.jpg)

**[Oluwatosin Mary Babalola](https://www.mql5.com/en/users/excel_om)**
\|
1 Apr 2025 at 12:15

Wow, this is the best so far since I’ve been following your articles. Keep up the good work


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
1 Apr 2025 at 13:02

**Oluwatosin Mary Babalola [#](https://www.mql5.com/en/forum/483826#comment_56322246):**

Wow, this is the best so far since I’ve been following your articles. Keep up the good work

Thank you.


![Louai Habiche](https://c.mql5.com/avatar/2024/2/65DC71BA-56DC.jpg)

**[Louai Habiche](https://www.mql5.com/en/users/louaiboss)**
\|
19 Apr 2025 at 17:03

awesome your doing great with youre articls best regrates

we can put in alert when the sihnal show !

![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
20 Apr 2025 at 10:37

**Louai Habiche [#](https://www.mql5.com/en/forum/483826#comment_56496108):**

awesome your doing great with youre articls best regrates

we can put in alert when the sihnal show !

Hello, Thank you for your kind words. its possible to do that by using "PlaySound()" function

![Master MQL5 from beginner to pro (Part V): Fundamental control flow operators](https://c.mql5.com/2/91/Learning_MQL5_-_From_Beginner_to_Pro_Part_5.___LOGOpng.png)[Master MQL5 from beginner to pro (Part V): Fundamental control flow operators](https://www.mql5.com/en/articles/15499)

This article explores the key operators used to modify the program's execution flow: conditional statements, loops, and switch statements. Utilizing these operators will allow the functions we create to behave more "intelligently".

![Build Self Optimizing Expert Advisors in MQL5 (Part 6): Self Adapting Trading Rules (II)](https://c.mql5.com/2/128/Automating_Trading_Strategies_in_MQL5_Part_5___LOGO3.png)[Build Self Optimizing Expert Advisors in MQL5 (Part 6): Self Adapting Trading Rules (II)](https://www.mql5.com/en/articles/17571)

This article explores optimizing RSI levels and periods for better trading signals. We introduce methods to estimate optimal RSI values and automate period selection using grid search and statistical models. Finally, we implement the solution in MQL5 while leveraging Python for analysis. Our approach aims to be pragmatic and straightforward to help you solve potentially complicated problems, with simplicity.

![MQL5 Wizard Techniques you should know (Part 58): Reinforcement Learning (DDPG) with Moving Average and Stochastic Oscillator Patterns](https://c.mql5.com/2/130/MQL5_Wizard_Techniques_you_should_know_Part_58__LOGO__2.png)[MQL5 Wizard Techniques you should know (Part 58): Reinforcement Learning (DDPG) with Moving Average and Stochastic Oscillator Patterns](https://www.mql5.com/en/articles/17668)

Moving Average and Stochastic Oscillator are very common indicators whose collective patterns we explored in the prior article, via a supervised learning network, to see which “patterns-would-stick”. We take our analyses from that article, a step further by considering the effects' reinforcement learning, when used with this trained network, would have on performance. Readers should note our testing is over a very limited time window. Nonetheless, we continue to harness the minimal coding requirements afforded by the MQL5 wizard in showcasing this.

![Neural Networks in Trading: Hierarchical Vector Transformer (HiVT)](https://c.mql5.com/2/90/logo-15688.png)[Neural Networks in Trading: Hierarchical Vector Transformer (HiVT)](https://www.mql5.com/en/articles/15688)

We invite you to get acquainted with the Hierarchical Vector Transformer (HiVT) method, which was developed for fast and accurate forecasting of multimodal time series.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/17574&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049233054687405979)

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