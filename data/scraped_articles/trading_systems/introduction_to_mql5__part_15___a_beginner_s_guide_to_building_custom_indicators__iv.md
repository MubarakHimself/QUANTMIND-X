---
title: Introduction to MQL5 (Part 15): A Beginner's Guide to Building Custom Indicators (IV)
url: https://www.mql5.com/en/articles/17689
categories: Trading Systems, Indicators
relevance_score: 9
scraped_at: 2026-01-22T17:34:50.111196
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/17689&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049229245051414407)

MetaTrader 5 / Trading systems


### Introduction

Welcome back to the Introduction to MQL5 series! You'll discover that this article will build directly on the ideas and techniques we've already discussed in previous articles. Since we'll be using a lot of what we've learned so far, this part will actually seem more like a continuation than a new beginning. By now, you should have a solid understanding of the MQL5 basics, and in this article, we’ll take things a step further by combining that knowledge to develop more interesting custom indicator.

You're only as good as the projects you've worked on in MQL5, which is why this series always takes a project-based approach. It's the most useful method for learning and developing yourself. We'll be creating an indicator in this part of the series that can recognize trends, use break of structure, and generate buy and sell signals. The entry point, stop loss, and several take profit levels are all included in these signals, providing you with a comprehensive strategy that you can test and build on. In this piece, you will learn how to design custom indicators in MQL5 using price action concepts. To create a trend-following strategy, you will learn how to recognize important market structures such as higher highs, higher lows, and lower highs, lower lows.

In this article, you'll learn:

- How to create price action indicator.
- Recognizing important points such as low (L), high (H), higher low (HL), higher high (HH), lower low (LL), and lower high (LH) to comprehend the structure of a bullish and bearish trend.
- Drawing the premium and discount zone based on key trend points, and marking the 50% retracement level.
- How to apply risk-reward ratio when calculating potential profit targets in a bullish trend setup.
- Calculating and marking the entry point, stop loss (SL), and multiple take profit (TP) levels based on the trend structure.

### **1\. Setting up the Project**

**1.1. How the Indicator Works**

The indicator will identify a low, high, higher low, and higher high to indicate an uptrend for buy signals. The 50% retracement level between the higher low and higher high will then be determined. A break in the structure above the higher high will initiate the entry, and the 50% retracement level will serve as the stop loss. A 1:1 risk-reward ratio will be the goal of Take Profit 1, and a 1:2 ratio will be the goal of Take Profit 2.

![Figure 1. Up Trend](https://c.mql5.com/2/131/Figure_1.png)

To identify a downtrend for sell signals, the indicator will first identify a high, low, lower high, and lower low. It will then compute the 50% retracement between the lower high and lower low. TP1 will be 1:1, TP2 will be 1:2, the stop loss will be at the 50% level, and the entry will take place on a break below the lower low.

![Figure 2. Down Trend](https://c.mql5.com/2/131/Figure_2.png)

### **2\. Building Price Action Indicator**

Every trading strategy can be transformed into an indicator — it just hasn’t been visualized yet. Anything that fits with a set of guidelines can be coded and shown on the chart, whether it be supply and demand, price action, or support and resistance. This is where MQL5 comes in. For algorithmic traders, it is among the greatest and most straightforward programming languages, enabling you to transform any trading logic into a useful and visually appealing tool. We'll start developing an indicator in this section that analyzes price movement, recognizing market structure such as highs, lows, higher highs, and lower lows, and then utilizes that data to produce insightful buy and sell signals that include entry, stop loss, and take profit levels.

In Chapter One, I outlined the project's purpose and how the indicator will spot trends, spot structural breaks, and produce full trade signals that include entry, stop loss, and take profits. We will now start putting everything in MQL5 into practice in this chapter. We'll take the logic we discussed and begin, step by step, to implement it in code.

**2.1. Identifying Highs and Lows**

Finding swing highs and lows is the first stage in creating our price action indicator. These significant market turning moments aid in identifying the trend's structure. By comparing the high or low of the current candle with that of the preceding and subsequent candles, we can identify them in MQL5. The detection of higher highs, higher lows, lower highs, and lower lows — all crucial for identifying patterns and structural breaks — will be based on this.

**Examples:**

```
//+------------------------------------------------------------------+
//| FUNCTION FOR SWING LOWS                                          |
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
//| FUNCTION FOR SWING HIGHS                                         |
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

The two functions were used in the previous article, and it helps to identify both swing highs and lows.

**2.2. Bullish Trend**

Using market structure, this indicator must first verify if there is an uptrend before indicating a buy signal. A low, a high, a higher low, and a higher high are the main price points that must be identified to accomplish this. This pattern denotes a bullish trend, meaning that buyers are in charge and that the market will probably keep rising. Following confirmation of this pattern, the indicator will get ready to produce a legitimate buy signal.

![Figure 3. Bullish Trend](https://c.mql5.com/2/131/figure_3.png)

**Example:**

```
// CHART ID
long chart_id = ChartID();

// Input parameters
input int  LookbackBars = 10;   // Number of bars to look back/forward for swing points
input int  bars_check   = 1000; // Number of bars to check for swing points
input bool show_bullish = true; //Show Buy Signals

// Variables for Bullish Market Structure

double L;              // Low: the starting low point in the up trend
datetime L_time;       // Time of the low
string L_letter;       // Label for the low point (e.g., "L")

double H;              // High: the first high after the low
datetime H_time;       // Time of the high
string H_letter;       // Label for the high point (e.g., "H")

double HL;             // Higher Low: the next low that is higher than the first low
datetime HL_time;      // Time of the higher low
string HL_letter;      // Label for the higher low point (e.g., "HL")

double HH;             // Higher High: the next high that is higher than the first high
datetime HH_time;      // Time of the higher high
string HH_letter;      // Label for the higher high point (e.g., "HH")

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
   if(show_bullish == true) // Check if the bullish trend is to be displayed
     {
      if(rates_total >= bars_check) // Ensure enough bars are available for analysis
        {
         // Loop through the price data starting from a certain point based on bars_check and LookbackBars
         for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
           {
            // Check if the current bar is a swing low
            if(IsSwingLow(low, i, LookbackBars))
              {
               // Store the values for the swing low
               L = low[i];
               L_time = time[i];
               L_letter = StringFormat("Low%d", i);

               // Loop through further to find a swing high after the low
               for(int j = i; j < rates_total - LookbackBars; j++)
                 {
                  // Check if the current bar is a swing high and occurs after the identified swing low
                  if(IsSwingHigh(high, j, LookbackBars) && time[j] > L_time)
                    {
                     // Store the values for the swing high
                     H = high[j];
                     H_time = time[j];
                     H_letter = StringFormat("High%d", j);

                     // Loop further to find a higher low after the swing high
                     for(int k = j; k < rates_total - LookbackBars; k++)
                       {
                        // Check if the current bar is a swing low and occurs after the swing high
                        if(IsSwingLow(low, k, LookbackBars) && time[k] > H_time)
                          {
                           // Store the values for the higher low
                           HL = low[k];
                           HL_time = time[k];
                           HL_letter = StringFormat("Higher Low%d", j);

                           // Loop further to find a higher high after the higher low
                           for(int l = j ; l < rates_total - LookbackBars; l++)
                             {
                              // Check if the current bar is a swing high and occurs after the higher low
                              if(IsSwingHigh(high, l, LookbackBars) && time[l] > HL_time)
                                {
                                 // Store the values for the higher high
                                 HH = high[l];
                                 HH_time = time[l];
                                 HH_letter = StringFormat("Higher High%d", l);

                                 // Check if the pattern follows the expected bullish structure: Low < High, Higher Low < High, Higher High > High
                                 if(L < H && HL < H && HL > L && HH > H)
                                   {
                                    // Create and display text objects for Low, High, Higher Low, and Higher High on the chart
                                    ObjectCreate(chart_id, L_letter, OBJ_TEXT, 0, L_time, L);
                                    ObjectSetString(chart_id, L_letter, OBJPROP_TEXT, "L");
                                    ObjectSetInteger(chart_id, L_letter, OBJPROP_COLOR, clrDarkGreen);
                                    ObjectSetInteger(chart_id, L_letter, OBJPROP_FONTSIZE, 15);

                                    ObjectCreate(chart_id, H_letter, OBJ_TEXT, 0, H_time, H);
                                    ObjectSetString(chart_id, H_letter, OBJPROP_TEXT, "H");
                                    ObjectSetInteger(chart_id, H_letter, OBJPROP_COLOR, clrDarkGreen);
                                    ObjectSetInteger(chart_id, H_letter, OBJPROP_FONTSIZE, 15);

                                    ObjectCreate(chart_id, HL_letter, OBJ_TEXT, 0, HL_time, HL);
                                    ObjectSetString(chart_id, HL_letter, OBJPROP_TEXT, "HL");
                                    ObjectSetInteger(chart_id, HL_letter, OBJPROP_COLOR, clrDarkGreen);
                                    ObjectSetInteger(chart_id, HL_letter, OBJPROP_FONTSIZE, 15);

                                    ObjectCreate(chart_id, HH_letter, OBJ_TEXT, 0, HH_time, HH);
                                    ObjectSetString(chart_id, HH_letter, OBJPROP_TEXT, "HH");
                                    ObjectSetInteger(chart_id, HH_letter, OBJPROP_COLOR, clrDarkGreen);
                                    ObjectSetInteger(chart_id, HH_letter, OBJPROP_FONTSIZE, 15);
                                   }

                                 break; // Exit the loop once the pattern is found
                                }
                             }

                           break; // Exit the loop once the higher low is found
                          }
                       }

                     break; // Exit the loop once the higher high is found
                    }
                 }
              }
           }
        }
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
```

**Output:**

![Figure 4. Up Trend Swings](https://c.mql5.com/2/131/figure_4.png)

**Explanation:**

The input parameters that are set at the beginning of the code are used to examine the market structure. To determine swing highs or lows and guarantee relevance in the current market, LookbackBars specifies the number of bars that are evaluated before and after the current bar. Meanwhile, bars\_check controls the number of bars of price data that will be examined in total, enabling the script to search through up to 1000 bars searching for possible bullish patterns.

Although it will also require more computing power, larger values for bars\_check indicate that the algorithm will consider a wider range of data when searching for these locations. A boolean input called show\_bullish controls whether the buy signals, or bullish signals, should be displayed. The application will proceed to analyze price movement and determine the bullish market structure based on swing points if show\_bullish is set to true. Even if the requirements are satisfied, the script will not plot or highlight any bullish structures if it is set to false.

The check for show\_bullish == true is the first action that takes place in terms of the indicator's logic. This guarantees that only when you have the desire to see buy signals will the bullish structure be identified. The program then checks to see if there is sufficient price data available for the analysis, if the criteria are satisfied. The condition if (rates\_total >= bars\_check) is used to verify this. To prevent errors from inadequate data, the analysis is skipped if there are fewer bars available than necessary. The script then loops over the price data to find swing spots if the criterion is met.

The outer loop begins with i, which looks for legitimate swing lows by scanning the price data backward from the most recent bar. The function IsSwingLow() determines whether the current bar is the lowest in the range specified by LookbackBars, hence identifying a swing low. The price and time of the low are recorded into the variables L and L\_time as soon as a swing low is identified. This lays the groundwork for the subsequent phase of the bullish pattern discovery process.  Following the identification of the swing low, the program searches for another swing high. Using IsSwingHigh(), the second loop, indexed by j, looks for a swing high in each succeeding bar. The values of any swing high that is discovered to follow the low are recorded in H and H\_time. This creates the initial segment of the bullish market structure, which consists of a low and a high.

After the swing high, the third loop, which is indexed by k, searches for a higher low. The script uses IsSwingLow() once more to find a higher low, which is defined as a low that is greater than the initial low L. When a higher low is discovered, HL and HL\_time are updated with its value and time. Following the identification of this higher low, the program keeps looking for the subsequent higher high. The swing high that follows the higher low is checked for in the fourth loop, which is indexed by l. Its values are saved in HH and HH\_time if this greater high is discovered. The code determines whether the four critical points — low, high, higher low, and higher high — follow a legitimate bullish pattern after they have been found. First low should be less than first high, higher low should be less than first high, higher low should be greater than first low, and higher high should be greater than first high. These criteria are checked by the condition if (L < H && HL < H && HL > L && HH > H). This confirms a bullish trend by ensuring that the pattern follows the anticipated sequence of higher highs and higher lows.

The program then creates and displays text objects on the chart to highlight the points that have been identified if all of these requirements are satisfied. The chart shows the points as labels at the appropriate periods and prices: Low (L), High (H), Higher Low (HL), and Higher High (HH). ObjectCreate() is used to create the text objects, while ObjectSetInteger() and ObjectSetString() are used to set their characteristics, including font size and color. With the points prominently indicated for reference, the user may now easily recognize the bullish structure on the chart. In summary, the program's goal is to find a pattern of higher highs and higher lows in price data to assess it for a bullish market structure. It accomplishes this by looking at swing points within a predetermined range of bars, recording the relevant information, and determining whether the structure exhibits the appropriate pattern. If the pattern is verified, the user can see it visually on the chart. The input parameters, which enable modification according to the user's preferences, govern the entire procedure.

**2.2.1. Mapping Premium and Discount Levels from Higher Low to Higher High**

Once the bullish market structure's swing points — the Low (L), High (H), Higher Low (HL), and Higher High (HH) — have been identified, the following step is to draw a box from the Higher Low to the Higher High. The box shows the range of price movement between these two significant swing points visually. This range is then calculated, and the 50% retracement level — a critical boundary between two price zones — is divided in half.

The division aids in defining the terms "premium zone" and "discount zone." Below the 50% mark, the discount zone denotes the range of prices that are thought to be more advantageous or "cheaper" for possible purchases. The premium zone, on the other hand, is above the 50% mark and denotes comparatively "expensive" rates. To optimize risk-to-reward, traders typically choose to purchase from the discount zone in many trading methods; however, for this specific indicator, we adopt a slightly different strategy.

![Figure 5. Premium and Discount](https://c.mql5.com/2/131/Figure_5.png)

In this instance, we are only interested in buying when the market trades above the premium zone or breaks the Higher High structure. This pattern indicates that the bullish structure is still in place and that the price is probably going to keep rising. To stay in line with the trend and lessen the chance of buying into a pullback or reversal, it is recommended to wait for the break above the Higher High or premium zone.

**Example:**

```
// CHART ID
long chart_id = ChartID();

// Input parameters
input int  LookbackBars = 10;   // Number of bars to look back/forward for swing points
input int  bars_check   = 1000; // Number of bars to check for swing points
input bool show_bullish = true; //Show Buy Signals

// Variables for Bullish Market Structure

double L;              // Low: the starting low point in the up trend
datetime L_time;       // Time of the low
string L_letter;       // Label for the low point (e.g., "L")

double H;              // High: the first high after the low
datetime H_time;       // Time of the high
string H_letter;       // Label for the high point (e.g., "H")

double HL;             // Higher Low: the next low that is higher than the first low
datetime HL_time;      // Time of the higher low
string HL_letter;      // Label for the higher low point (e.g., "HL")

double HH;             // Higher High: the next high that is higher than the first high
datetime HH_time;      // Time of the higher high
string HH_letter;      // Label for the higher high point (e.g., "HH")

// Variables for Premium and Discount
string pre_dis_box;     // Name/ID for the premium-discount zone box (rectangle object on chart)
double lvl_50;          // The price level representing the 50% retracement between Higher Low and Higher High
string lvl_50_line;     // Name/ID for the horizontal line marking the 50% level

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
   if(show_bullish == true) // Check if the bullish trend is to be displayed
     {
      if(rates_total >= bars_check) // Ensure enough bars are available for analysis
        {
         // Loop through the price data starting from a certain point based on bars_check and LookbackBars
         for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
           {
            // Check if the current bar is a swing low
            if(IsSwingLow(low, i, LookbackBars))
              {
               // Store the values for the swing low
               L = low[i];
               L_time = time[i];
               L_letter = StringFormat("Low%d", i);

               // Loop through further to find a swing high after the low
               for(int j = i; j < rates_total - LookbackBars; j++)
                 {
                  // Check if the current bar is a swing high and occurs after the identified swing low
                  if(IsSwingHigh(high, j, LookbackBars) && time[j] > L_time)
                    {
                     // Store the values for the swing high
                     H = high[j];
                     H_time = time[j];
                     H_letter = StringFormat("High%d", j);

                     // Loop further to find a higher low after the swing high
                     for(int k = j; k < rates_total - LookbackBars; k++)
                       {
                        // Check if the current bar is a swing low and occurs after the swing high
                        if(IsSwingLow(low, k, LookbackBars) && time[k] > H_time)
                          {
                           // Store the values for the higher low
                           HL = low[k];
                           HL_time = time[k];
                           HL_letter = StringFormat("Higher Low%d", j);

                           // Loop further to find a higher high after the higher low
                           for(int l = j ; l < rates_total - LookbackBars; l++)
                             {
                              // Check if the current bar is a swing high and occurs after the higher low
                              if(IsSwingHigh(high, l, LookbackBars) && time[l] > HL_time)
                                {
                                 // Store the values for the higher high
                                 HH = high[l];
                                 HH_time = time[l];
                                 HH_letter = StringFormat("Higher High%d", l);

                                 // Check if the pattern follows the expected bullish structure: Low < High, Higher Low < High, Higher High > High
                                 if(L < H && HL < H && HL > L && HH > H)
                                   {
                                    // Create and display text objects for Low, High, Higher Low, and Higher High on the chart
                                    ObjectCreate(chart_id, L_letter, OBJ_TEXT, 0, L_time, L);
                                    ObjectSetString(chart_id, L_letter, OBJPROP_TEXT, "L");
                                    ObjectSetInteger(chart_id, L_letter, OBJPROP_COLOR, clrDarkGreen);
                                    ObjectSetInteger(chart_id, L_letter, OBJPROP_FONTSIZE, 15);

                                    ObjectCreate(chart_id, H_letter, OBJ_TEXT, 0, H_time, H);
                                    ObjectSetString(chart_id, H_letter, OBJPROP_TEXT, "H");
                                    ObjectSetInteger(chart_id, H_letter, OBJPROP_COLOR, clrDarkGreen);
                                    ObjectSetInteger(chart_id, H_letter, OBJPROP_FONTSIZE, 15);

                                    ObjectCreate(chart_id, HL_letter, OBJ_TEXT, 0, HL_time, HL);
                                    ObjectSetString(chart_id, HL_letter, OBJPROP_TEXT, "HL");
                                    ObjectSetInteger(chart_id, HL_letter, OBJPROP_COLOR, clrDarkGreen);
                                    ObjectSetInteger(chart_id, HL_letter, OBJPROP_FONTSIZE, 15);

                                    ObjectCreate(chart_id, HH_letter, OBJ_TEXT, 0, HH_time, HH);
                                    ObjectSetString(chart_id, HH_letter, OBJPROP_TEXT, "HH");
                                    ObjectSetInteger(chart_id, HH_letter, OBJPROP_COLOR, clrDarkGreen);
                                    ObjectSetInteger(chart_id, HH_letter, OBJPROP_FONTSIZE, 15);

                                    // Calculate the 50% retracement level between the Higher Low and Higher High
                                    lvl_50 = HL + ((HH - HL)/2);

                                    // Generate unique names for the premium-discount box and the 50% level line using the current loop index
                                    pre_dis_box = StringFormat("Premium and Discount Box%d", i);
                                    lvl_50_line = StringFormat("Level 50 Line%d", i);

                                    // Create a rectangle object representing the premium-discount zone from the Higher Low to the Higher High
                                    ObjectCreate(chart_id, pre_dis_box, OBJ_RECTANGLE, 0, HL_time, HL, time[l + LookbackBars], HH);

                                    // Create a trend line (horizontal line) marking the 50% retracement level
                                    ObjectCreate(chart_id, lvl_50_line, OBJ_TREND, 0, HL_time, lvl_50, time[l + LookbackBars], lvl_50);

                                    // Set the color of the premium-discount box to dark green
                                    ObjectSetInteger(chart_id, pre_dis_box, OBJPROP_COLOR, clrDarkGreen);

                                    // Set the color of the 50% level line to dark green
                                    ObjectSetInteger(chart_id, lvl_50_line, OBJPROP_COLOR, clrDarkGreen);

                                    // Set the width of the premium-discount box for better visibility
                                    ObjectSetInteger(chart_id, pre_dis_box, OBJPROP_WIDTH, 2);

                                    // Set the width of the 50% level line for better visibility
                                    ObjectSetInteger(chart_id, lvl_50_line, OBJPROP_WIDTH, 2);

                                   }

                                 break; // Exit the loop once the pattern is found
                                }
                             }

                           break; // Exit the loop once the higher low is found
                          }
                       }

                     break; // Exit the loop once the higher high is found
                    }
                 }
              }
           }
        }
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
```

**Output:**

![Figure 6. Premium and Discount on chart](https://c.mql5.com/2/131/figure_6.png)

**Explanation:**

The code determines the 50% retracement level, which is the halfway between the Higher Low (HL) and Higher High (HH). The premium (above) and discount (below) zones are separated by this level. The midway point between HL and HH is determined using the formula lvl\_50 = HL + ((HH - HL)/2);. Next, the loop index i is included in the names of two string variables, pre\_dis\_box and lvl\_50\_line. These serve as distinct identifiers for the visual elements we will be drawing on the chart. Each drawing will be unique and won't replace earlier ones if the loop index is included.

A rectangle is created on the chart by the line ObjectCreate(chart\_id, pre\_dis\_box, OBJ\_RECTANGLE, 0, HL\_time, HL, time\[l + LookbackBars\], HH); which graphically depicts the transition from the higher low to the higher high. Traders can quickly determine the range of the most recent bullish swing with the use of this box. The rectangle's end is anchored at a future bar (l + LookbackBars) using the HH price, while its beginning is anchored at the HL's time and price. This keeps the box visible by extending it slightly into the future.

A horizontal line is then drawn across the chart at the 50% level by the line ObjectCreate(chart\_id, lvl\_50\_line, OBJ\_TREND, 0, HL\_time, lvl\_50, time\[l + LookbackBars\], lvl\_50);. This threshold is important because, according to the logic of this indicator, we are only looking for possible buy signals when the market is trading above the premium zone, or over the 50% level of the most recent upswing. The box's and the line's colors are set to clrDarkGreen, and the ObjectSetInteger function is used to boost their thickness (or width) to 2  to make them both visually clear. The market must fully break above the HH for a buy signal to be deemed valid; in other words, the price must close outside and above the whole premium zone. Put another way, we only want to purchase when the market structure is clearly bullish and has beyond the previous swing high (HH).

**2.2.2. Indicating Entry Point, Stop Loss, and Take Profit**

Finding possible buy signals comes next once the premium and discount zones between the Higher Low (HL) and Higher High (HH) have been correctly marked. Waiting for a bullish breakout bar to break above the Higher High (HH), which indicates that the market is still moving strongly upward, is the key to a legitimate buy setup.

But crossing over the HH alone is insufficient. The bullish bar must close above the HH for the entry to be verified because we need to make sure the breakout is real. The fact that the price closed above the HH suggests that there is ongoing purchasing demand and that it will probably keep climbing. The entry point is established at the close of the bullish bar that breaks above the HH after the breakout has been verified. We are certain that the market has demonstrated sufficient strength currently for us to enter the trade.

We set the Stop Loss (SL) at the 50% level (lvl\_50), which is the middle point between the HL and HH, to guard against a possible reversal. To avoid getting caught in a possible retreat into the discount zone (below the 50% level), which can indicate a change in market mood, we have put the SL here. Our method is based on the Risk-to-Reward (R:R) ratio for the Take Profit (TP) levels. The profit objective for the first Take Profit level, TP1, is equal to the risk distance between the entry point and the SL since it is set at a 1:1 R:R. The profit target for the second Take Profit level, or TP2, is twice the distance between the entrance point and the SL, with a 1:2 R:R setting. For traders who would rather lock in partial profits at TP1, these two take-profit levels offer flexibility, allowing them to leave a portion of the trade open to profit from additional gains should the market continue its positive trend.

**Example:**

```
// Variables for Entry, Stop Loss, and Take Profit
string entry_line;        // Line object to represent the entry point on the chart
string entry_txt;         // Text object for displaying "BUY" at the entry point
double lvl_SL;            // Stop Loss level (set at the 50% retracement level)
string lvl_sl_line;       // Line object for representing the Stop Loss level
string lvl_sl_txt;        // Text object for labeling the Stop Loss level
double TP1;               // Take Profit 1 level (1:1 risk-reward ratio)
double TP2;               // Take Profit 2 level (1:2 risk-reward ratio)
string lvl_tp_line;       // Line object for representing the Take Profit 1 level
string lvl_tp2_line;      // Line object for representing the Take Profit 2 level
string lvl_tp_txt;        // Text object for labeling the Take Profit 1 level
string lvl_tp2_txt;       // Text object for labeling the Take Profit 2 level
string buy_object;        // Arrow object to indicate the Buy signal on the chart
```

```
if(show_bullish == true) // Check if the bullish trend is to be displayed
  {
   if(rates_total >= bars_check) // Ensure enough bars are available for analysis
     {
      // Loop through the price data starting from a certain point based on bars_check and LookbackBars
      for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
        {
         // Check if the current bar is a swing low
         if(IsSwingLow(low, i, LookbackBars))
           {
            // Store the values for the swing low
            L = low[i];
            L_time = time[i];
            L_letter = StringFormat("Low%d", i);

            // Loop through further to find a swing high after the low
            for(int j = i; j < rates_total - LookbackBars; j++)
              {
               // Check if the current bar is a swing high and occurs after the identified swing low
               if(IsSwingHigh(high, j, LookbackBars) && time[j] > L_time)
                 {
                  // Store the values for the swing high
                  H = high[j];
                  H_time = time[j];
                  H_letter = StringFormat("High%d", j);

                  // Loop further to find a higher low after the swing high
                  for(int k = j; k < rates_total - LookbackBars; k++)
                    {
                     // Check if the current bar is a swing low and occurs after the swing high
                     if(IsSwingLow(low, k, LookbackBars) && time[k] > H_time)
                       {
                        // Store the values for the higher low
                        HL = low[k];
                        HL_time = time[k];
                        HL_letter = StringFormat("Higher Low%d", j);

                        // Loop further to find a higher high after the higher low
                        for(int l = j ; l < rates_total - LookbackBars; l++)
                          {
                           // Check if the current bar is a swing high and occurs after the higher low
                           if(IsSwingHigh(high, l, LookbackBars) && time[l] > HL_time)
                             {
                              // Store the values for the higher high
                              HH = high[l];
                              HH_time = time[l];
                              HH_letter = StringFormat("Higher High%d", l);

                              // Check if the pattern follows the expected bullish structure: Low < High, Higher Low < High, Higher High > High
                              if(L < H && HL < H && HL > L && HH > H)
                                {
                                 // Create and display text objects for Low, High, Higher Low, and Higher High on the chart
                                 ObjectCreate(chart_id, L_letter, OBJ_TEXT, 0, L_time, L);
                                 ObjectSetString(chart_id, L_letter, OBJPROP_TEXT, "L");
                                 ObjectSetInteger(chart_id, L_letter, OBJPROP_COLOR, clrDarkGreen);
                                 ObjectSetInteger(chart_id, L_letter, OBJPROP_FONTSIZE, 15);

                                 ObjectCreate(chart_id, H_letter, OBJ_TEXT, 0, H_time, H);
                                 ObjectSetString(chart_id, H_letter, OBJPROP_TEXT, "H");
                                 ObjectSetInteger(chart_id, H_letter, OBJPROP_COLOR, clrDarkGreen);
                                 ObjectSetInteger(chart_id, H_letter, OBJPROP_FONTSIZE, 15);

                                 ObjectCreate(chart_id, HL_letter, OBJ_TEXT, 0, HL_time, HL);
                                 ObjectSetString(chart_id, HL_letter, OBJPROP_TEXT, "HL");
                                 ObjectSetInteger(chart_id, HL_letter, OBJPROP_COLOR, clrDarkGreen);
                                 ObjectSetInteger(chart_id, HL_letter, OBJPROP_FONTSIZE, 15);

                                 ObjectCreate(chart_id, HH_letter, OBJ_TEXT, 0, HH_time, HH);
                                 ObjectSetString(chart_id, HH_letter, OBJPROP_TEXT, "HH");
                                 ObjectSetInteger(chart_id, HH_letter, OBJPROP_COLOR, clrDarkGreen);
                                 ObjectSetInteger(chart_id, HH_letter, OBJPROP_FONTSIZE, 15);

                                 // Calculate the 50% retracement level between the Higher Low and Higher High
                                 lvl_50 = HL + ((HH - HL)/2);

                                 // Generate unique names for the premium-discount box and the 50% level line using the current loop index
                                 pre_dis_box = StringFormat("Premium and Discount Box%d", i);
                                 lvl_50_line = StringFormat("Level 50 Line%d", i);

                                 // Create a rectangle object representing the premium-discount zone from the Higher Low to the Higher High
                                 ObjectCreate(chart_id, pre_dis_box, OBJ_RECTANGLE, 0, HL_time, HL, time[l + LookbackBars], HH);

                                 // Create a trend line (horizontal line) marking the 50% retracement level
                                 ObjectCreate(chart_id, lvl_50_line, OBJ_TREND, 0, HL_time, lvl_50, time[l + LookbackBars], lvl_50);

                                 // Set the color of the premium-discount box to dark green
                                 ObjectSetInteger(chart_id, pre_dis_box, OBJPROP_COLOR, clrDarkGreen);

                                 // Set the color of the 50% level line to dark green
                                 ObjectSetInteger(chart_id, lvl_50_line, OBJPROP_COLOR, clrDarkGreen);

                                 // Set the width of the premium-discount box for better visibility
                                 ObjectSetInteger(chart_id, pre_dis_box, OBJPROP_WIDTH, 2);

                                 // Set the width of the 50% level line for better visibility
                                 ObjectSetInteger(chart_id, lvl_50_line, OBJPROP_WIDTH, 2);

                                 for(int m = l; m < rates_total-1; m++)
                                   {

                                    if(close[m] > open[m] && close[m] > HH && time[m] >= time[l+LookbackBars])
                                      {

                                       TP1 = close[m] + (close[m] - lvl_50);
                                       TP2 = TP1 + (close[m] - lvl_50);

                                       entry_line = StringFormat("Entry%d", m);
                                       lvl_sl_line = StringFormat("SL%d", m);
                                       lvl_tp_line =  StringFormat("TP%d", m);
                                       lvl_tp2_line =  StringFormat("TP 2%d", m);

                                       ObjectCreate(chart_id,entry_line,OBJ_TREND,0,HL_time,close[m],time[m],close[m]);
                                       ObjectCreate(chart_id,lvl_sl_line,OBJ_TREND,0,HL_time,lvl_50,time[m],lvl_50);
                                       ObjectCreate(chart_id,lvl_tp_line,OBJ_TREND,0,HL_time,TP1,time[m],TP1);
                                       ObjectCreate(chart_id,lvl_tp2_line,OBJ_TREND,0,HL_time,TP2,time[m],TP2);

                                       ObjectSetInteger(chart_id,entry_line,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,lvl_sl_line,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,lvl_tp_line,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,lvl_tp2_line,OBJPROP_WIDTH,2);

                                       ObjectSetInteger(chart_id,entry_line,OBJPROP_COLOR,clrDarkGreen);
                                       ObjectSetInteger(chart_id,lvl_sl_line,OBJPROP_COLOR,clrDarkGreen);
                                       ObjectSetInteger(chart_id,lvl_tp_line,OBJPROP_COLOR,clrDarkGreen);
                                       ObjectSetInteger(chart_id,lvl_tp2_line,OBJPROP_COLOR,clrDarkGreen);

                                       entry_txt = StringFormat("Entry Text%d", m);
                                       lvl_sl_txt = StringFormat("SL Text%d", m);
                                       lvl_tp_txt = StringFormat("TP 1 Text%d", m);
                                       lvl_tp2_txt = StringFormat("TP 2 Text%d", m);

                                       ObjectCreate(chart_id, lvl_sl_txt, OBJ_TEXT, 0,time[m],lvl_50);
                                       ObjectSetString(chart_id, lvl_sl_txt, OBJPROP_TEXT, "SL");
                                       ObjectSetInteger(chart_id,lvl_sl_txt,OBJPROP_COLOR,clrDarkGreen);
                                       ObjectSetInteger(chart_id,lvl_sl_txt,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, entry_txt, OBJ_TEXT, 0,time[m],close[m]);
                                       ObjectSetString(chart_id, entry_txt, OBJPROP_TEXT, "BUY");
                                       ObjectSetInteger(chart_id,entry_txt,OBJPROP_COLOR,clrDarkGreen);
                                       ObjectSetInteger(chart_id,entry_txt,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, lvl_tp_txt, OBJ_TEXT, 0,time[m],TP1);
                                       ObjectSetString(chart_id, lvl_tp_txt, OBJPROP_TEXT, "TP1");
                                       ObjectSetInteger(chart_id,lvl_tp_txt,OBJPROP_COLOR,clrDarkGreen);
                                       ObjectSetInteger(chart_id,lvl_tp_txt,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, lvl_tp2_txt, OBJ_TEXT, 0,time[m],TP2);
                                       ObjectSetString(chart_id, lvl_tp2_txt, OBJPROP_TEXT, "TP2");
                                       ObjectSetInteger(chart_id,lvl_tp2_txt,OBJPROP_COLOR,clrDarkGreen);
                                       ObjectSetInteger(chart_id,lvl_tp2_txt,OBJPROP_FONTSIZE,15);

                                       buy_object = StringFormat("Buy Object%d", m);
                                       ObjectCreate(chart_id,buy_object,OBJ_ARROW_BUY,0,time[m],close[m]);

                                       break;
                                      }
                                   }

                                }

                              break; // Exit the loop once the pattern is found
                             }
                          }

                        break; // Exit the loop once the higher low is found
                       }
                    }

                  break; // Exit the loop once the higher high is found
                 }
              }
           }
        }
     }
  }
```

**Output:**

![Figure 7. Buy Signal](https://c.mql5.com/2/131/Figure_7.png)

You can see that the premium and discount zones, as well as the buy signal markers, are not precisely drawn in the image above. This is because certain conditions were overlooked before the objects being drawn on the chart. To improve the approach, we must include more verifications to make sure the buy signal is legitimate. Before sketching any objects on the chart, we must first make sure that a candle has truly broken the higher high (HH). A break of the HH signifies a continuation of the bullish trend, which is required for the buy signal to be deemed legitimate, making this a critical criterion. We shouldn't start the entry and risk management computations until this requirement has been satisfied.

The number of bars from the higher low (HL) to the end of the premium and discount box must then be counted. This guarantees that the price action is within an acceptable range and aids in our understanding of how far the market has moved. After this count is finished, we need to confirm that the bullish bar's close price, which broke the higher high (HH), is near to the premium and discount box. This guarantees that the buy signal is not too far from the anticipated market structure and is occurring within a fair price range.

**Example:**

```
// Declare variables to count bars
int n_bars;     // Number of bars from Higher Low to the end of the Premium/Discount box
int n_bars_2;   // Number of bars from the end of the Premium/Discount box to the bullish bar that broke HH
```

```
 if(show_bullish == true) // Check if the bullish trend is to be displayed
     {
      if(rates_total >= bars_check) // Ensure enough bars are available for analysis
        {
         // Loop through the price data starting from a certain point based on bars_check and LookbackBars
         for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
           {
            // Check if the current bar is a swing low
            if(IsSwingLow(low, i, LookbackBars))
              {
               // Store the values for the swing low
               L = low[i];
               L_time = time[i];
               L_letter = StringFormat("Low%d", i);

               // Loop through further to find a swing high after the low
               for(int j = i; j < rates_total - LookbackBars; j++)
                 {
                  // Check if the current bar is a swing high and occurs after the identified swing low
                  if(IsSwingHigh(high, j, LookbackBars) && time[j] > L_time)
                    {
                     // Store the values for the swing high
                     H = high[j];
                     H_time = time[j];
                     H_letter = StringFormat("High%d", j);

                     // Loop further to find a higher low after the swing high
                     for(int k = j; k < rates_total - LookbackBars; k++)
                       {
                        // Check if the current bar is a swing low and occurs after the swing high
                        if(IsSwingLow(low, k, LookbackBars) && time[k] > H_time)
                          {
                           // Store the values for the higher low
                           HL = low[k];
                           HL_time = time[k];
                           HL_letter = StringFormat("Higher Low%d", j);

                           // Loop further to find a higher high after the higher low
                           for(int l = j ; l < rates_total - LookbackBars; l++)
                             {
                              // Check if the current bar is a swing high and occurs after the higher low
                              if(IsSwingHigh(high, l, LookbackBars) && time[l] > HL_time)
                                {
                                 // Store the values for the higher high
                                 HH = high[l];
                                 HH_time = time[l];
                                 HH_letter = StringFormat("Higher High%d", l);

                                 // Loop through the bars to check for the conditions for entry
                                 for(int m = l; m < rates_total-1; m++)
                                   {
                                    // Check if the current bar is a bullish bar and if the price has broken the higher high (HH)
                                    if(close[m] > open[m] && close[m] > HH && time[m] >= time[l+LookbackBars])
                                      {
                                       // Count the bars between HL_time and the end of the Premium/Discount box
                                       n_bars = Bars(_Symbol, PERIOD_CURRENT, HL_time, time[l + LookbackBars]);

                                       // Count the bars between the end of the Premium/Discount box and the candle that broke HH
                                       n_bars_2 = Bars(_Symbol, PERIOD_CURRENT, time[l + LookbackBars], time[m]);

                                       // Check if the pattern follows the expected bullish structure: Low < High, Higher Low < High, Higher High > High
                                       if(L < H && HL < H && HL > L && HH > H && open[l+LookbackBars] <= HH && n_bars_2 < n_bars)
                                         {
                                          // Create and display text objects for Low, High, Higher Low, and Higher High on the chart
                                          ObjectCreate(chart_id, L_letter, OBJ_TEXT, 0, L_time, L);
                                          ObjectSetString(chart_id, L_letter, OBJPROP_TEXT, "L");
                                          ObjectSetInteger(chart_id, L_letter, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, L_letter, OBJPROP_FONTSIZE, 15);

                                          ObjectCreate(chart_id, H_letter, OBJ_TEXT, 0, H_time, H);
                                          ObjectSetString(chart_id, H_letter, OBJPROP_TEXT, "H");
                                          ObjectSetInteger(chart_id, H_letter, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, H_letter, OBJPROP_FONTSIZE, 15);

                                          ObjectCreate(chart_id, HL_letter, OBJ_TEXT, 0, HL_time, HL);
                                          ObjectSetString(chart_id, HL_letter, OBJPROP_TEXT, "HL");
                                          ObjectSetInteger(chart_id, HL_letter, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, HL_letter, OBJPROP_FONTSIZE, 15);

                                          ObjectCreate(chart_id, HH_letter, OBJ_TEXT, 0, HH_time, HH);
                                          ObjectSetString(chart_id, HH_letter, OBJPROP_TEXT, "HH");
                                          ObjectSetInteger(chart_id, HH_letter, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, HH_letter, OBJPROP_FONTSIZE, 15);

                                          // Calculate the 50% retracement level between the Higher Low and Higher High
                                          lvl_50 = HL + ((HH - HL)/2);

                                          // Generate unique names for the premium-discount box and the 50% level line using the current loop index
                                          pre_dis_box = StringFormat("Premium and Discount Box%d", i);
                                          lvl_50_line = StringFormat("Level 50 Line%d", i);

                                          // Create a rectangle object representing the premium-discount zone from the Higher Low to the Higher High
                                          ObjectCreate(chart_id, pre_dis_box, OBJ_RECTANGLE, 0, HL_time, HL, time[l + LookbackBars], HH);

                                          // Create a trend line (horizontal line) marking the 50% retracement level
                                          ObjectCreate(chart_id, lvl_50_line, OBJ_TREND, 0, HL_time, lvl_50, time[l + LookbackBars], lvl_50);

                                          // Set the color of the premium-discount box to dark green
                                          ObjectSetInteger(chart_id, pre_dis_box, OBJPROP_COLOR, clrDarkGreen);

                                          // Set the color of the 50% level line to dark green
                                          ObjectSetInteger(chart_id, lvl_50_line, OBJPROP_COLOR, clrDarkGreen);

                                          // Set the width of the premium-discount box for better visibility
                                          ObjectSetInteger(chart_id, pre_dis_box, OBJPROP_WIDTH, 2);

                                          // Set the width of the 50% level line for better visibility
                                          ObjectSetInteger(chart_id, lvl_50_line, OBJPROP_WIDTH, 2);

                                          // Calculate Take Profit levels based on the 50% retracement
                                          TP1 = close[m] + (close[m] - lvl_50);    // TP1 at 1:1 risk-reward ratio
                                          TP2 = TP1 + (close[m] - lvl_50);          // TP2 at 1:2 risk-reward ratio

                                          // Create unique object names for Entry, Stop Loss, and Take Profit lines and text
                                          entry_line = StringFormat("Entry%d", m);
                                          lvl_sl_line = StringFormat("SL%d", m);
                                          lvl_tp_line =  StringFormat("TP%d", m);
                                          lvl_tp2_line =  StringFormat("TP 2%d", m);

                                          // Create the lines on the chart for Entry, Stop Loss, and Take Profit levels
                                          ObjectCreate(chart_id, entry_line, OBJ_TREND, 0, HL_time, close[m], time[m], close[m]);
                                          ObjectCreate(chart_id, lvl_sl_line, OBJ_TREND, 0, HL_time, lvl_50, time[m], lvl_50);
                                          ObjectCreate(chart_id, lvl_tp_line, OBJ_TREND, 0, HL_time, TP1, time[m], TP1);
                                          ObjectCreate(chart_id, lvl_tp2_line, OBJ_TREND, 0, HL_time, TP2, time[m], TP2);

                                          // Set the properties for the lines (width, color, etc.)
                                          ObjectSetInteger(chart_id, entry_line, OBJPROP_WIDTH, 2);
                                          ObjectSetInteger(chart_id, lvl_sl_line, OBJPROP_WIDTH, 2);
                                          ObjectSetInteger(chart_id, lvl_tp_line, OBJPROP_WIDTH, 2);
                                          ObjectSetInteger(chart_id, lvl_tp2_line, OBJPROP_WIDTH, 2);

                                          ObjectSetInteger(chart_id, entry_line, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, lvl_sl_line, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, lvl_tp_line, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, lvl_tp2_line, OBJPROP_COLOR, clrDarkGreen);

                                          // Create the text labels for Entry, Stop Loss, and Take Profit levels
                                          entry_txt = StringFormat("Entry Text%d", m);
                                          lvl_sl_txt = StringFormat("SL Text%d", m);
                                          lvl_tp_txt = StringFormat("TP 1 Text%d", m);
                                          lvl_tp2_txt = StringFormat("TP 2 Text%d", m);

                                          // Create the text objects for the Entry, Stop Loss, and Take Profit labels
                                          ObjectCreate(chart_id, lvl_sl_txt, OBJ_TEXT, 0, time[m], lvl_50);
                                          ObjectSetString(chart_id, lvl_sl_txt, OBJPROP_TEXT, "SL");
                                          ObjectSetInteger(chart_id, lvl_sl_txt, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, lvl_sl_txt, OBJPROP_FONTSIZE, 15);

                                          ObjectCreate(chart_id, entry_txt, OBJ_TEXT, 0, time[m], close[m]);
                                          ObjectSetString(chart_id, entry_txt, OBJPROP_TEXT, "BUY");
                                          ObjectSetInteger(chart_id, entry_txt, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, entry_txt, OBJPROP_FONTSIZE, 15);

                                          ObjectCreate(chart_id, lvl_tp_txt, OBJ_TEXT, 0, time[m], TP1);
                                          ObjectSetString(chart_id, lvl_tp_txt, OBJPROP_TEXT, "TP1");
                                          ObjectSetInteger(chart_id, lvl_tp_txt, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, lvl_tp_txt, OBJPROP_FONTSIZE, 15);

                                          ObjectCreate(chart_id, lvl_tp2_txt, OBJ_TEXT, 0, time[m], TP2);
                                          ObjectSetString(chart_id, lvl_tp2_txt, OBJPROP_TEXT, "TP2");
                                          ObjectSetInteger(chart_id, lvl_tp2_txt, OBJPROP_COLOR, clrDarkGreen);
                                          ObjectSetInteger(chart_id, lvl_tp2_txt, OBJPROP_FONTSIZE, 15);

                                          // Create a Buy arrow object to indicate the Buy signal on the chart
                                          buy_object = StringFormat("Buy Object%d", m);
                                          ObjectCreate(chart_id, buy_object, OBJ_ARROW_BUY, 0, time[m], close[m]);

                                          break; // Exit the loop once a Buy signal is found
                                         }
                                      }

                                   }

                                 break; // Exit the loop once the pattern is found
                                }
                             }

                           break; // Exit the loop once the higher low is found
                          }
                       }

                     break; // Exit the loop once the higher high is found
                    }
                 }
              }
           }
        }
     }
```

**Output:**

![figure 8. Buy Signal](https://c.mql5.com/2/131/figure_8.png)

**Explanation:**

Two integer variables, n\_bars and n\_bars\_2, were declared in the code's global space. The number of candlesticks (bars) between important points in the bullish market structure pattern is determined by these variables. In particular, n\_bars is the number of bars that separate the Premium/Discount box's end (time\[l + LookbackBars\]) from the Higher Low (HL). However, n\_bars\_2 counts the number of bars that separate the bullish candlestick that broke the Higher High (HH) from the end of the Premium/Discount box. This count is used to assess if the price action has strayed too far from the optimal trading zone or whether the buy signal is still valid.

These variables are employed later in the code as part of an extra validation condition that strengthens the argument for a bullish market structure. An additional condition n\_bars\_2 < n\_bars is checked after determining the Low, High, Higher Low, and Higher High (making sure they adhere to the structure L < H, HL < H, HL > L, and HH > H) and verifying that the opening price of the candle that ended the Premium/Discount box is not higher than the Higher High. This makes sure that the bullish breakout candle (the one that breaks above HH) doesn't show up too far after the formation, which could suggest a weak or invalid setup, and that it shows up fairly close to the pattern.

All the ObjectCreate() and ObjectSet\*() routines that were previously used to draw the chart's Low, High, Higher Low, Higher High, Premium/Discount box, 50% line, and Entry/SL/TP markers were transferred inside this if statement to impose this more stringent inspection. This implies that only when all the bullish structure and time requirements are satisfied will these visual components be produced and shown. By doing this, the chart stays clear and isn't overloaded with erroneous or premature items due to misleading signals.

**2.3. Bearish Trend**

This indicator must first use market structure to validate a downturn before it can indicate a sell. Finding a series of significant pricing points — a high, a low, a lower high, and a lower low — is how this is accomplished. This pattern demonstrates that sellers are in control and that the market is probably going to keep moving lower, which supports bearish momentum. The indicator will start looking for legitimate sell signals as soon as this structure is verified.

![Bearish Trend](https://c.mql5.com/2/131/figure_9.png)

**Example:**

```
// Variables for Bearish Market Structure
double LH;              // Lower High: the high formed after the initial low in a downtrend
datetime LH_time;       // Time of the Lower High
string LH_letter;       // Label used to display the Lower High on the chart (e.g., "LH")
double LL;              // Lower Low: the new low formed after the Lower High in a downtrend
datetime LL_time;       // Time of the Lower Low
string LL_letter;       // Label used to display the Lower Low on the chart (e.g., "LL")
string sell_object; // Arrow object to indicate the Sell signal on the chart
```

```
// BEARISH TREND
if(show_bearish == true)  // Check if the user enabled the bearish trend display
  {
   if(rates_total >= bars_check)  // Ensure enough candles are available for processing
     {
      // Loop through historical bars to find a swing high (potential start of bearish structure)
      for(int i = rates_total - bars_check; i < rates_total - LookbackBars; i++)
        {
         if(IsSwingHigh(high, i, LookbackBars))  // Detect first swing high
           {
            H = high[i];
            H_time = time[i];
            H_letter = StringFormat("High B%d", i);  // Label for the high

            // From the swing high, look for the next swing low
            for(int j = i; j < rates_total - LookbackBars; j++)
              {
               if(IsSwingLow(low, j, LookbackBars) && time[j] > H_time)  // Confirm next swing low
                 {
                  L = low[j];
                  L_time = time[j];
                  L_letter = StringFormat("Low B%d", j);  // Label for the low

                  // From the swing low, look for the Lower High
                  for(int k = j; k < rates_total - LookbackBars; k++)
                    {
                     if(IsSwingHigh(high, k, LookbackBars) && time[k] > L_time)
                       {
                        LH = high[k];
                        LH_time = time[k];
                        LH_letter = StringFormat("Lower High%d", k);  // Label for the Lower High

                        // From the LH, find a Lower Low
                        for(int l = j ; l < rates_total - LookbackBars; l++)
                          {
                           if(IsSwingLow(low, l, LookbackBars) && time[l] > LH_time)
                             {
                              LL = low[l];
                              LL_time = time[l];
                              LL_letter = StringFormat("Lower Low%d", l);  // Label for Lower Low

                              // Calculate 50% retracement level from LH to LL
                              lvl_50 = LL + ((LH - LL)/2);

                              // Prepare object names
                              pre_dis_box = StringFormat("Gan Box B%d", i);
                              lvl_50_line = StringFormat("Level 50 Line B%d", i);

                              // Search for a bearish entry condition
                              for(int m = l; m < rates_total-1; m++)
                                {
                                 // Confirm bearish candle breaking below the LL
                                 if(close[m] < open[m] && close[m] < LL && time[m] >= time[l+LookbackBars])
                                   {
                                    // Count bars for pattern distance validation
                                    n_bars = Bars(_Symbol,PERIOD_CURRENT,LH_time, time[l+LookbackBars]);  // From LH to box end
                                    n_bars_2 = Bars(_Symbol,PERIOD_CURRENT,time[l+LookbackBars], time[m]);  // From box end to break candle

                                    // Confirm valid bearish structure and proximity of break candle
                                    if(H > L && LH > L && LH < H && LL < L && open[l+LookbackBars] >= LL && n_bars_2 < n_bars)
                                      {
                                       // Draw the Premium/Discount box
                                       ObjectCreate(chart_id,pre_dis_box, OBJ_RECTANGLE,0,LH_time,LH, time[l+LookbackBars],LL);
                                       ObjectCreate(chart_id,lvl_50_line, OBJ_TREND,0,LH_time,lvl_50, time[l+LookbackBars],lvl_50);

                                       ObjectSetInteger(chart_id,pre_dis_box,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,lvl_50_line,OBJPROP_WIDTH,2);

                                       // Label the structure points
                                       ObjectCreate(chart_id, H_letter, OBJ_TEXT, 0, H_time, H);
                                       ObjectSetString(chart_id, H_letter, OBJPROP_TEXT, "H");
                                       ObjectSetInteger(chart_id,H_letter,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, L_letter, OBJ_TEXT, 0, L_time, L);
                                       ObjectSetString(chart_id, L_letter, OBJPROP_TEXT, "L");
                                       ObjectSetInteger(chart_id,L_letter,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, LH_letter, OBJ_TEXT, 0, LH_time, LH);
                                       ObjectSetString(chart_id, LH_letter, OBJPROP_TEXT, "LH");
                                       ObjectSetInteger(chart_id,LH_letter,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, LL_letter, OBJ_TEXT, 0, LL_time, LL);
                                       ObjectSetString(chart_id, LL_letter, OBJPROP_TEXT, "LL");
                                       ObjectSetInteger(chart_id,LL_letter,OBJPROP_FONTSIZE,15);

                                       ObjectSetInteger(chart_id,H_letter,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,L_letter,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,LL_letter,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,LH_letter,OBJPROP_WIDTH,2);

                                       // Calculate Take Profits based on 1:1 and 1:2 RR
                                       TP1 = close[m] - (lvl_50 - close[m]);
                                       TP2 = TP1 - (lvl_50 - close[m]);

                                       // Generate entry, SL and TP object names
                                       entry_line = StringFormat("Entry B%d", m);
                                       lvl_sl_line = StringFormat("SL B%d", m);
                                       lvl_tp_line =  StringFormat("TP B%d", m);
                                       lvl_tp2_line =  StringFormat("TP 2 B%d", m);

                                       // Draw entry, SL, TP1, TP2 levels
                                       ObjectCreate(chart_id,entry_line,OBJ_TREND,0,LH_time,close[m],time[m],close[m]);
                                       ObjectCreate(chart_id,lvl_sl_line, OBJ_TREND,0,LH_time,lvl_50, time[m],lvl_50);
                                       ObjectCreate(chart_id,lvl_tp_line, OBJ_TREND,0,LH_time,TP1, time[m],TP1);
                                       ObjectCreate(chart_id,lvl_tp2_line, OBJ_TREND,0,LH_time,TP2, time[m],TP2);

                                       ObjectSetInteger(chart_id,entry_line,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,lvl_sl_line,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,lvl_tp_line,OBJPROP_WIDTH,2);
                                       ObjectSetInteger(chart_id,lvl_tp2_line,OBJPROP_WIDTH,2);

                                       // Generate text labels
                                       entry_txt = StringFormat("Entry Text B%d", m);
                                       lvl_sl_txt = StringFormat("SL Text B%d", m);
                                       lvl_tp_txt = StringFormat("TP Text B%d", m);
                                       lvl_tp2_txt = StringFormat("TP 2 Text B%d", m);

                                       ObjectCreate(chart_id, entry_txt, OBJ_TEXT, 0,time[m],close[m]);
                                       ObjectSetString(chart_id, entry_txt, OBJPROP_TEXT, "SELL");
                                       ObjectSetInteger(chart_id,entry_txt,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, lvl_sl_txt, OBJ_TEXT, 0,time[m],lvl_50);
                                       ObjectSetString(chart_id, lvl_sl_txt, OBJPROP_TEXT, "SL");
                                       ObjectSetInteger(chart_id,lvl_sl_txt,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, lvl_tp_txt, OBJ_TEXT, 0,time[m],TP1);
                                       ObjectSetString(chart_id, lvl_tp_txt, OBJPROP_TEXT, "TP1");
                                       ObjectSetInteger(chart_id,lvl_tp_txt,OBJPROP_FONTSIZE,15);

                                       ObjectCreate(chart_id, lvl_tp2_txt, OBJ_TEXT, 0,time[m],TP2);
                                       ObjectSetString(chart_id, lvl_tp2_txt, OBJPROP_TEXT, "TP2");
                                       ObjectSetInteger(chart_id,lvl_tp2_txt,OBJPROP_FONTSIZE,15);

                                       // Draw sell arrow
                                       sell_object = StringFormat("Sell Object%d", m);
                                       ObjectCreate(chart_id,sell_object,OBJ_ARROW_SELL,0,time[m],close[m]);
                                      }

                                    break;  // Exit loop after valid setup
                                   }
                                }

                              break;  // Exit LL search
                             }
                          }

                        break;  // Exit LH search
                       }
                    }

                  break;  // Exit L search
                 }
              }
           }
        }
     }
  }
```

**Output:**

![Figure 10. Sell Signals](https://c.mql5.com/2/131/figure_10.png)

**Explanation:**

The code begins by using if(show\_bearish == true) to see if the user has activated the bearish trend logic. To find a legitimate bearish market structure, the indicator loops through historical bars if it is enabled and there are enough bars available (rates\_total >= bars\_check). The first step in the process is to determine a swing high (H). After identifying a swing high, the code searches for a swing low (L) that follows the high. If it is discovered, it keeps looking for a lower low (LL) that validates the bearish structure, followed by a lower high (LH), which is a swing high below the first high H. Labels such as "H", "L", "LH", and "LL" are created on the chart using these values and associated timestamps.

Next, the Premium/Discount zone box is drawn from the LH to the LL, and the 50% retracement level between LH and LL is determined (lvl\_50 = LL + ((LH - LL)/2). The indicator searches for a bearish candle (close < open) that closes below the LL before placing any trade-related objects (entry, SL, TP1, TP2). By using two variables, n\_bars, which counts the bars from the Lower High (LH) to the end of the Premium/Discount box, and n\_bars\_2, which counts from the end of the box to the bearish candle that breaks the Lower Low (LL), the code also makes sure that the break of structure happens within a reasonable number of bars. The code only draws the entry line at the candle close, sets the Stop Loss (SL) at the 50% level, and positions TP1 and TP2 at 1:1 and 1:2 risk-reward levels, respectively, when all requirements are met, including proper structure, a valid break, and an appropriate distance.

Additionally, it adds a sell arrow and the words "SELL" to the bearish candle. For a downtrend, the logic is basically the same as for a positive trend pattern, but it is reversed. Since the bearish trend is only the opposite of the bullish trend, which has already been thoroughly detailed, the explanation is brief. The structure and reasoning are the same, except they are reversed to show a downward trend rather than an upward one.

### **Conclusion**

In this article, we built an MQL5 custom indicator that identifies market structure by detecting key points such as Low (L), Lower Low (LL), Higher Low (HL), High (H), and Higher High (HH). Using these points, the indicator determines bullish or bearish trends and automatically draws entry points, stop loss at level 50, and take profit levels (TP1 and TP2) based on a structured pattern. It also marks the premium and discount zones to visually highlight where price is likely to react. All chart objects are drawn only when specific conditions are met, ensuring that the signals remain clean and reliable.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17689.zip "Download all attachments in the single ZIP archive")

[Project9\_Trend\_Indicator.mq5](https://www.mql5.com/en/articles/download/17689/project9_trend_indicator.mq5 "Download Project9_Trend_Indicator.mq5")(29.37 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/484624)**

![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (V): AnalyticsPanel Class](https://c.mql5.com/2/133/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_X_CODEIV___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (V): AnalyticsPanel Class](https://www.mql5.com/en/articles/17397)

In this discussion, we explore how to retrieve real-time market data and trading account information, perform various calculations, and display the results on a custom panel. To achieve this, we will dive deeper into developing an AnalyticsPanel class that encapsulates all these features, including panel creation. This effort is part of our ongoing expansion of the New Admin Panel EA, introducing advanced functionalities using modular design principles and best practices for code organization.

![Developing a Trading System Based on the Order Book (Part I): Indicator](https://c.mql5.com/2/92/Desenvolvendo_um_Trading_System_com_base_no_Livro_de_Ofertas_Parte_I.png)[Developing a Trading System Based on the Order Book (Part I): Indicator](https://www.mql5.com/en/articles/15748)

Depth of Market is undoubtedly a very important element for executing fast trades, especially in High Frequency Trading (HFT) algorithms. In this series of articles, we will look at this type of trading events that can be obtained through a broker on many tradable symbols. We will start with an indicator, where you can customize the color palette, position and size of the histogram displayed directly on the chart. We will also look at how to generate BookEvent events to test the indicator under certain conditions. Other possible topics for future articles include how to store price distribution data and how to use it in a strategy tester.

![Neural Networks in Trading: Hierarchical Feature Learning for Point Clouds](https://c.mql5.com/2/92/Neural_Networks_in_Trading_Hierarchical_Learning_of_Point_Cloud_Features___LOGO.png)[Neural Networks in Trading: Hierarchical Feature Learning for Point Clouds](https://www.mql5.com/en/articles/15789)

We continue to study algorithms for extracting features from a point cloud. In this article, we will get acquainted with the mechanisms for increasing the efficiency of the PointNet method.

![Statistical Arbitrage Through Mean Reversion in Pairs Trading: Beating the Market by Math](https://c.mql5.com/2/132/Statistical_Arbitrage_Through_Mean_Reversion_in_Pairs_Trading__LOGO.png)[Statistical Arbitrage Through Mean Reversion in Pairs Trading: Beating the Market by Math](https://www.mql5.com/en/articles/17735)

This article describes the fundamentals of portfolio-level statistical arbitrage. Its goal is to facilitate the understanding of the principles of statistical arbitrage to readers without deep math knowledge and propose a starting point conceptual framework. The article includes a working Expert Advisor, some notes about its one-year backtest, and the respective backtest configuration settings (.ini file) for the reproduction of the experiment.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/17689&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049229245051414407)

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