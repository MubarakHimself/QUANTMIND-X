---
title: Introduction to MQL5 (Part 22): Building an Expert Advisor for the 5-0 Harmonic Pattern
url: https://www.mql5.com/en/articles/19856
categories: Trading Systems, Indicators, Expert Advisors
relevance_score: 10
scraped_at: 2026-01-22T17:20:27.994051
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/19856&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049057609568330791)

MetaTrader 5 / Trading systems


### Introduction

Welcome back to Part 22 of the Introduction to MQL5 series! In the previous article, we explored how to automate the detection of harmonic patterns such as the Gartley formation in MQL5. In this continuation, we will focus on another fascinating yet less common structure known as the 5-0 Harmonic Pattern. This pattern stands out because, unlike most harmonic setups that start with a retracement, the 5-0 pattern begins with a sharp counter-trend move, followed by a structured correction that provides valuable trading opportunities.

This article will teach you how to recognize the 5-0 pattern automatically in an Expert Advisor and detect it programmatically. You will learn how to determine the 5-0 structure's major turning points (0, X, A, B, C, and D), calculate and validate Fibonacci ratios specific to it, and visually represent the pattern on the chart. By the end of this article, you will know how to convert the technical idea underlying the 5-0 pattern into functional MQL5 code that can analyze price action automatically.

### **5-0 Harmonic Pattern**

The 5-0 Harmonic Pattern is a reversal formation that usually emerges following a significant and prolonged price movement. It is seen as a continuation-to-reversal pattern rather than a pure reversal setup and indicates a possible shift in the trend's direction. Four legs (XA, AB, BC, and CD) are formed by the pattern's six primary points: 0, X, A, B, C, and D. The distinctive beginning point and structure of the 5-0 pattern set it apart from other harmonic patterns. In contrast to most harmonic patterns, which start with a swing high or low, the 5-0 pattern starts with a corrective move, which frequently occurs after a prior trend or pattern has finished. This makes it stand out because it aims to capture the depletion of a countertrend move rather than the beginning of a new impetus.

Point 0, which creates a clear swing high on the chart, is where the bullish 5-0 pattern starts. The market drops from this level to form point X, which is the structure's first significant bearish leg. A corrective retreat of that decline is represented by the subsequent movement from X to A. Even though there is no set Fibonacci level for this retracement, point A needs to be below point 0 to indicate that the trend is still bearish overall.

Following the establishment of point A, the market descends further to produce point B, which is situated between the XA leg's 113% and 161.8% Fibonacci extension. The exhaustion stage of the current downtrend is frequently represented by this leg. The market turns back up from point B to form point C, which usually represents 161.8% to 224% of the AB leg. Within the pattern, this movement demonstrates a powerful corrective reaction. The structure is completed at point D, where the market pulls back once more. It is recommended that Point D retrace 50% to 55% of the BC leg. The possible buying area is this retracement zone.

![Figure 1. Bullish 5-0 Pattern](https://c.mql5.com/2/174/Figure_1.png)

Point 0 establishes a distinct swing low on the chart and initiates the bearish 5-0 pattern. The first bullish leg of the pattern is established when the market rises from this point to produce point X. Although the subsequent movement, from X to A, is a corrective retreat, this leg does not require a precise Fibonacci retracement. To ensure that the overall trend is still positive presently, point A must be higher than point 0.

Point B, which should be between the 113% and 161.8% Fibonacci extension of the XA leg, is created when the price rises once again after point A is formed. Usually, this level indicates that the bullish trend has reached its limit. After that, the market turns back down to form point C, which should represent a significant pattern correction and span between 161.8% and 224% of the AB leg. The structure is completed when the price rises once more to reach point D. It is recommended that Point D retrace 50% to 55% of the BC leg. This area of retracement serves as a possible selling point.

![Figure 3. Bearish 5-0 Pattern](https://c.mql5.com/2/174/Figure_2.png)

### **Identifying 0XAB**

The next step is to programmatically implement the 5-0 pattern in MQL5 now that we are aware of its meaning and essential ratio principles. This entails putting the idea into practice by collecting and updating candle data, accurately identifying swing points, putting together potential sequences for 0, X, A, B, C, and D, verifying each leg in accordance with the ratio requirements of the pattern, and labeling verified patterns on the chart.

In practice, we will follow a straightforward process: copy recent bars into arrays, locate important pivot points using a reliable swing detection technique, and then step through these pivots to create possible 0X, XA, AB, and BC legs. The Fibonacci extension and retracement criteria will then be used to verify that points B, C, and D are genuine. If all requirements are met, we will next connect the detection logic to trade execution and draw the pattern and triangles for visual confirmation. While identifying the bullish 5-0 harmonic pattern will be the primary focus of this article, it should be noted that, with a few slight directional alterations, the same reasoning and detection procedure may also be used to identify the bearish 5-0 pattern.

Copying candle data and locating the 0XAB structure is the first step in developing the detection algorithm for the 5-0 pattern. Using the built-in Copy functions in MQL5, this begins by gathering recent market data, including open, close, high, low, and time. The computer can access historical price movements through these data arrays, which is crucial for identifying significant swing highs and lows.

The program uses swing detection logic to identify possible turning points in price action once the data is available. The EA uses these swing points to determine potential 0, X, A, and B points, which form the basis of the 5-0 pattern. A large swing high is represented by point 0, and a swing low is represented by point X. Then, point B expands beyond X in accordance with the Fibonacci extension range (between 113% and 161.8% of XA), whereas point A emerges as a lower high below point 0. Because they dictate whether the pattern may advance into a legitimate bullish 5-0 setup when points C and D are later confirmed, correctly detecting these first four points is essential.

Example:

```
input ENUM_TIMEFRAMES timeframe = PERIOD_CURRENT;

datetime time_bar;
int bars_check = 400;
int total_symbol_bars;

double open[];
double close[];
double low[];
double high[];
datetime time[];

int z = 4;

double O;
datetime O_time;
string O_line;
string O_letter;

double X;
datetime X_time;
string X_letter;

double A;
datetime A_time;
string A_letter;

double B;
datetime B_time;
string B_line;
string B_letter;

double C;
datetime C_time;
string C_line;
string C_letter;

double D;
datetime D_time;
string D_line;
string D_letter;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

   total_symbol_bars = Bars(_Symbol, timeframe);
   time_bar = iTime(_Symbol, timeframe, 0);
   CopyOpen(_Symbol, timeframe, time_bar, bars_check, open);
   CopyClose(_Symbol, timeframe, time_bar, bars_check, close);
   CopyLow(_Symbol, timeframe, time_bar, bars_check, low);
   CopyHigh(_Symbol, timeframe, time_bar, bars_check, high);
   CopyTime(_Symbol, timeframe, time_bar, bars_check, time);

  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING LOW                                           |
//+------------------------------------------------------------------+
bool IsSwingLow(const double &low_price[], int index, int lookback)
  {
   for(int i = 1; i <= lookback; i++)
     {
      if(low_price[index] > low_price[index - i] || low_price[index] > low_price[index + i])
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| FUNCTION FOR SWING HIGH                                          |
//+------------------------------------------------------------------+
bool IsSwingHigh(const double &high_price[], int index, int lookback)
  {
   for(int i = 1; i <= lookback; i++)
     {
      if(high_price[index] < high_price[index - i] || high_price[index] < high_price[index + i])
         return false; // If the current high is not the highest, return false.
     }
   return true;
  }
```

Explanation:

```
input ENUM_TIMEFRAMES timeframe = PERIOD_CURRENT;
```

The chart timeframe that the Expert Advisor (EA) will examine is specified here. Although it uses the timeframe of the current chart by default, traders may easily change it to test the EA on other timeframes, including M15, H1, or D1, without changing the basic code.

To manage candle data and overall chart information, a number of global variables are then declared:

```
datetime time_bar;
int bars_check = 400;
int total_symbol_bars;

double open[];
double close[];
double low[];
double high[];
datetime time[];
```

Together, these variables store market data. The number of bars (candles) that the EA should examine is specified by the variable bars\_check; in this example, the last 400 candles. The matching candle data from the chart will then be loaded into the arrays. Because they offer the price history that swing highs and lows will be identified from, these data arrays are crucial.

The program defines a number of variables to hold the specifics of each significant point in the 5-0 pattern after declaring the fundamental data containers:

```
double O;
datetime O_time;
string O_line;
string O_letter;

double X;
datetime X_time;
string X_letter;

double A;
datetime A_time;
string A_letter;

double B;
datetime B_time;
string B_line;
string B_letter;

double C;
datetime C_time;
string C_line;
string C_letter;

double D;
datetime D_time;
string D_line;
string D_letter;
```

Every swing point has a unique collection of variables assigned to it. For example, O\_time records the time at which the O point happened, whereas O stores the price of the O point. The point can be labeled and visual markers drawn on the chart using the O\_line and O\_letter variables. With the swing points, the same pattern is repeated. Plotting the entire 5-0 pattern when all legs are verified is made possible by this structure, which enables the software to record the coordinates of each point.

To determine swing lows, the IsSwingLow function looks at whether the low price of an index is lower than the lows of the candles that surround it. The lookback parameter regulates the number of candles that are compared before and after the current one. A swing low is verified if the candle's low is, in fact, the lowest of its neighbors. For spotting turning points that could create the pattern's bases of 0 or X, swing lows are essential. Within the specified lookback range, the IsSwingHigh function additionally verifies whether a candle's high is higher than the highs of nearby candles.

Finding the 0XAB formation is the next step after establishing the fundamental framework for managing market data and identifying swing points. Finding notable swing highs and lows that correspond to the waves of the bullish 5-0 harmonic pattern entails searching through historical candle data. Point X will be recognized as a lower swing low, and Point 0 as a significant swing high. Point B must then extend past the X point within the Fibonacci extension range of 113% to 161.8% of the XA leg, after which point A should form as a lower high beneath point 0.

Example:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

   total_symbol_bars = Bars(_Symbol, timeframe);
   time_bar = iTime(_Symbol, timeframe, 0);
   CopyOpen(_Symbol, timeframe, time_bar, bars_check, open);
   CopyClose(_Symbol, timeframe, time_bar, bars_check, close);
   CopyLow(_Symbol, timeframe, time_bar, bars_check, low);
   CopyHigh(_Symbol, timeframe, time_bar, bars_check, high);
   CopyTime(_Symbol, timeframe, time_bar, bars_check, time);

   if(total_symbol_bars >= bars_check)
     {

      for(int i = z ; i < bars_check - z; i++)
        {
         if(IsSwingHigh(high, i, z))
           {

            for(int j = i; j < bars_check - z; j++)
              {

               if(IsSwingLow(low, j, z) && low[j] < high[i])
                 {

                  X = low[j];
                  X_time = time[j];
                  X_letter = StringFormat("X  %d",j);

                  for(int a = j; a >= i; a--)
                    {
                     if(IsSwingHigh(high, a, z) && high[a] > X)
                       {

                        O = high[a];
                        O_time = time[a];
                        O_letter = StringFormat("0  %d",a);

                        for(int k = j; k < bars_check - z; k++)
                          {

                           if(IsSwingHigh(high, k, z) && high[k] > X)
                             {

                              A = high[k];
                              A_time = time[k];
                              A_letter = StringFormat("A  %d",k);

                              for(int l = k; l < bars_check - z; l++)
                                {

                                 if(IsSwingLow(low, l, z) && low[l] < X)
                                   {
                                    B = low[l];
                                    B_time = time[l];
                                    B_letter = StringFormat("B  %d",l);

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
  }
```

Explanation:

The program starts by checking if there are enough bars to analyze. The search doesn't start if there are fewer bars available overall than what is needed for analysis. This is similar to making sure your bookcase is stocked before searching for a sequence because if you don't, you may end up at the conclusion too quickly and miss the entire pattern.

It then starts looking for a first swing high on the chart. To determine if a bar at a specific location is a local high, the outer loop checks through each bar using the IsSwingHigh() function. That place becomes a possible beginning point for the 5-0 structure when this requirement is met.

The program looks for a swing low that can be used as point X after identifying a swing high. Another loop that begins at the high and proceeds through the bars searching for a low position that is lower than the previous high is used to accomplish this. When the criteria are met, the low price, timestamp, and label for the chart display are all stored in the variable X. You have marked the next book in your potential sequence after finding a shorter one a little further down the shelf than the tall one.

The algorithm locates the right point 0 by moving backward from X after detecting it. To find the highest peak that is above X, it reverses the scan between X and the initial high. With its price, time, and label noted, the first such peak discovered turns into point 0. The search is done backwards to ensure that the 0 point is the most recent high before X.

After identifying points 0 and X, the algorithm proceeds once more to find point A. When the following swing high crosses over X, it looks for it and marks it as A. Point A marks the beginning of a fresh price rally inside the evolving pattern by representing a new peak following the initial fall. This indicates that your series is still forming in the anticipated alternate pattern, much like when you see a moderately tall book after the short one in the bookshelf example.

Finding point B is the next stage. To find a swing low that falls below X, the code keeps scanning forward after A. The price, time, and label of such a bar are noted as B when it is found. The code then marks B as the last point of the 0XAB sequence and stops looking for more lows. The four-book grouping that characterizes the early structure of the 5-0 pattern is now complete, as you have discovered another small book that is even lower than the preceding short one in the bookshelf metaphor.

This sequence's inner loops all employ break commands to terminate when a valid point is reached. This keeps the detection targeted and effective by ensuring the algorithm doesn't generate many overlapping sets from the same region. When you've put together a legitimate stack of books that adheres to the height and order guidelines, you stop scanning that section and proceed to the next one. The program correctly locates and saves the first four crucial points, 0, X, A, and B, along with their times and labels after these nested loops.

Recall as I previously stated, swing high A, which is a retracement of 0X, must be bigger than X and less than 0, and swing high 0 must be greater than X. But for A, no particular Fibonacci retracement % is necessary. Point B must fall between the XA leg's Fibonacci extension levels of 113.0% and 161.8%. We are supposed to create an object on the chart to visually identify and express this pattern development once the algorithm has verified these conditions.

Example:

```
input ENUM_TIMEFRAMES timeframe = PERIOD_CURRENT;
input double b_xa_max = 161.8; // MAX B EXTENSION LEVEL FOR XA
input double b_xa_min = 113.0; // MIN B EXTENSION LEVEL FOR XA
```

```
double fib_ext_b_161;
double fib_ext_b_113;
string fib_xa_ext_obj;
string fib_xa_ext_lvl;

string ox_line;
string xa_line;
string ab_line;
ulong chart_id = ChartID();
```

```
for(int l = k; l < bars_check - z; l++)
  {

   if(IsSwingLow(low, l, z) && low[l] < X)
     {

      B = low[l];
      B_time = time[l];
      B_letter = StringFormat("B  %d",l);

      fib_ext_b_113 =  MathAbs((((A - X) / 100) * (b_xa_min - 100)) - X);
      fib_ext_b_161 =   MathAbs((((A - X) / 100) * (b_xa_max - 100)) - X);

      if(X < O && A > X && A < O && B <= fib_ext_b_113 && B >= fib_ext_b_161)
        {

         ObjectCreate(chart_id,O_letter,OBJ_TEXT,0,O_time,O);
         ObjectSetString(chart_id,O_letter,OBJPROP_TEXT,"0");
         ObjectSetInteger(chart_id,O_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,X_time,X);
         ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
         ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,A_time,A);
         ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
         ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,B_time,B);
         ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
         ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrBlue);

         ox_line = StringFormat("0X Line  %d",i);
         xa_line = StringFormat("XA Line  %d",i);
         ab_line = StringFormat("AB Line  %d",i);

         ObjectCreate(chart_id,ox_line,OBJ_TREND,0, O_time, O,X_time,X);
         ObjectSetInteger(chart_id,ox_line,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,xa_line,OBJ_TREND,0, X_time, X,A_time,A);
         ObjectSetInteger(chart_id,xa_line,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,ab_line,OBJ_TREND,0, A_time, A,B_time,B);
         ObjectSetInteger(chart_id,ab_line,OBJPROP_COLOR,clrBlue);

         fib_xa_ext_obj = StringFormat("XA Expansion  %d",i);
         ObjectCreate(chart_id,fib_xa_ext_obj,OBJ_EXPANSION,0,A_time,A,X_time,X,A_time,A);
         ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_COLOR,clrBlue);
         for(int i = 0; i <= 2; i++)
           {

            ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);

           }

         fib_xa_ext_lvl = StringFormat("XA Expansion Levels %d",i);

         ObjectCreate(chart_id,fib_xa_ext_lvl,OBJ_RECTANGLE,0,X_time,fib_ext_b_113,B_time,fib_ext_b_161);
         ObjectSetInteger(chart_id,fib_xa_ext_lvl,OBJPROP_COLOR,clrBlue);

        }

      break;
     }
  }
```

Output:

![Figure 3. 0XAB](https://c.mql5.com/2/174/figure_3.png)

Explanation:

The code first declares a few input variables. The program will automatically utilize the same time frame as the chart to which it is applied. Next, the Fibonacci extension range of the XA leg is represented by two inputs. For the pattern to be valid, point B must fall inside the range defined by these values.

The calculated values and object names are then stored in a collection of variables. The calculated Fibonacci extension price levels that correspond to 113% and 161.8% of XA will subsequently be held by the variables fib\_ext\_b\_113 and fib\_ext\_b\_161. The various graphical components that will be depicted on the chart to illustrate the identified pattern are named using the variables fib\_xa\_ext\_obj, fib\_xa\_ext\_lvl, ox\_line, xa\_line, and ab\_line. To make sure that all drawing activities occur on the appropriate chart window, the variable chart\_id fetches the ID of the active chart.

Following the declaration of these variables, the computer calculates the Fibonacci extension levels using the mathematical correlations between points A and X. It determines the price points that match XA's 113% and 161.8% extensions. Consider measuring the separation between points X and A on a shelf to gain a better understanding of this. The first XA leg is represented by the base distance (100%), and the next point's extension beyond it is represented by the extensions 113% and 161.8%. Point B must be inside this permissible range, as determined by these calculated levels, for the pattern to be legitimate.

A conditional statement that guarantees all necessary geometric and Fibonacci relationships of the 5-0 pattern are met is introduced in the following line of code. It maintains the bearish-to-bullish price structure by determining if point X is below point 0. Additionally, it confirms that point A stays within a legitimate retracement zone by retracing upward from X without exceeding point 0. Lastly, it verifies that point B is within 113% and 161.8% of XA, which is the permissible Fibonacci extension range.

The algorithm then draws the identified structure on the chart once all of these requirements have been satisfied. It starts by naming the important swing locations (0, X, A, and B) with text objects. To make them easily visible, each text item is given a blue color. These designations make it easier for traders to identify the pattern's essential structure.

The code labels the locations and then connects 0X, XA, and AB with trend lines. The movement of the price swings that serve as the basis for the 5-0 pattern is graphically represented by each of these lines. Each line's designation (such as "0X Line" or "XA Line") guarantees that every item on the chart stays distinct and simple to recognize.

After that, potential extension zones are visually projected between the XA legs using the Fibonacci Extension tool. This expansion strengthens the link that establishes the correctness of point B by demonstrating to traders how far the price may have stretched beyond XA. Lastly, to visually emphasize the area between the 113% and 161.8% extension levels, a rectangle object is drawn. This rectangular area clearly illustrates the expected formation of point B by acting as a highlighted shelf portion. Point B can assist you in visually verifying the harmonic setup on their charts by indicating a suitable extension zone for the possible 5-0 pattern if it falls within this range.

### **Identifying C**

Finding points C and D will complete the 5-0 pattern structure after points 0, X, A, and B have been correctly located. Because they establish whether the detected setup develops into a legitimate harmonic formation, these two points are essential. Depending on how the price responds to B, point C may form as an extension or retracement from the AB leg. C is anticipated to be between 161.8% and 224% Fibonacci extension of AB in the case of the bearish 5-0 pattern. This indicates that the price should move past the end of AB by approximately 1.618 to 2.24 times the distance of AB once it reaches point B.

To put it another way, think of the AB leg as a section of a shelf. Placing a new book (C) farther along that shelf, not too close to B (less than 161.8%) nor too far away (beyond 224%), is what happens when the price goes over B. This zone, which indicates that the market has extended sufficiently to preserve harmonic balance without going beyond, is the ideal area for C to arrive.

Following the identification of point C, the structure will be completed by determining point D as a retracement of the BC leg. Verifying the bearish 5-0 setup before applying any trade confirmation or alert logic requires that both C and D be properly aligned in order for the pattern to retain its harmonic symmetry.

Example:

```
input double c_ab_max = 224.0; // MAX C EXTENSION LEVEL FOR AB
input double c_ab_min = 161.8; // MIN C EXTENSION LEVEL FOR AB
```

```
double fib_ext_c_161;
double fib_ext_c_224;
string fib_ab_ext_obj;
string fib_ab_ext_lvl;
string bc_line;
```

```
if(total_symbol_bars >= bars_check)
  {
   for(int i = z ; i < bars_check - z; i++)
     {
      if(IsSwingHigh(high, i, z))
        {
         for(int j = i; j < bars_check - z; j++)
           {
            if(IsSwingLow(low, j, z) && low[j] < high[i])
              {
               X = low[j];
               X_time = time[j];
               X_letter = StringFormat("X  %d",j);

               for(int a = j; a >= i; a--)
                 {
                  if(IsSwingHigh(high, a, z) && high[a] > X)
                    {
                     O = high[a];
                     O_time = time[a];
                     O_letter = StringFormat("0  %d",a);

                     for(int k = j; k < bars_check - z; k++)
                       {

                        if(IsSwingHigh(high, k, z) && high[k] > X)
                          {
                           A = high[k];
                           A_time = time[k];
                           A_letter = StringFormat("A  %d",k);

                           for(int l = k; l < bars_check - z; l++)
                             {

                              if(IsSwingLow(low, l, z) && low[l] < X)
                                {
                                 B = low[l];
                                 B_time = time[l];
                                 B_letter = StringFormat("B  %d",l);

                                 for(int m = l; m < bars_check - z; m++)
                                   {

                                    if(IsSwingHigh(high, m, z) && high[m] > A)
                                      {

                                       C = high[m];
                                       C_time = time[m];
                                       C_letter = StringFormat("C  %d",m);

                                       fib_ext_b_113 =  MathAbs((((A - X) / 100) * (b_xa_min - 100)) - X);
                                       fib_ext_b_161 =   MathAbs((((A - X) / 100) * (b_xa_max - 100)) - X);

                                       fib_ext_c_161 = MathAbs((((A - B) / 100) * (c_ab_min - 100)) + A);
                                       fib_ext_c_224 =    MathAbs((((A - B) / 100) * (c_ab_max - 100)) + A);

                                       if(X < O && A > X && A < O && B <= fib_ext_b_113 && B >= fib_ext_b_161 && C >= fib_ext_c_161 && C <= fib_ext_c_224)
                                         {
                                          ObjectCreate(chart_id,O_letter,OBJ_TEXT,0,O_time,O);
                                          ObjectSetString(chart_id,O_letter,OBJPROP_TEXT,"0");
                                          ObjectSetInteger(chart_id,O_letter,OBJPROP_COLOR,clrBlue);

                                          ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,X_time,X);
                                          ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
                                          ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,clrBlue);

                                          ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,A_time,A);
                                          ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
                                          ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrBlue);

                                          ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,B_time,B);
                                          ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
                                          ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrBlue);

                                          ox_line = StringFormat("0X Line  %d",i);
                                          xa_line = StringFormat("XA Line  %d",i);
                                          ab_line = StringFormat("AB Line  %d",i);

                                          ObjectCreate(chart_id,ox_line,OBJ_TREND,0, O_time, O,X_time,X);
                                          ObjectSetInteger(chart_id,ox_line,OBJPROP_COLOR,clrBlue);

                                          ObjectCreate(chart_id,xa_line,OBJ_TREND,0, X_time, X,A_time,A);
                                          ObjectSetInteger(chart_id,xa_line,OBJPROP_COLOR,clrBlue);

                                          ObjectCreate(chart_id,ab_line,OBJ_TREND,0, A_time, A,B_time,B);
                                          ObjectSetInteger(chart_id,ab_line,OBJPROP_COLOR,clrBlue);

                                          fib_xa_ext_obj = StringFormat("XA Expansion  %d",i);
                                          ObjectCreate(chart_id,fib_xa_ext_obj,OBJ_EXPANSION,0,A_time,A,X_time,X,A_time,A);
                                          ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_COLOR,clrBlue);
                                          for(int i = 0; i <= 2; i++)
                                            {
                                             ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);
                                            }

                                          fib_xa_ext_lvl = StringFormat("XA Expansion Levels %d",i);

                                          ObjectCreate(chart_id,fib_xa_ext_lvl,OBJ_RECTANGLE,0,X_time,fib_ext_b_113,B_time,fib_ext_b_161);
                                          ObjectSetInteger(chart_id,fib_xa_ext_lvl,OBJPROP_COLOR,clrBlue);

                                          ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,C_time,C);
                                          ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
                                          ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrBlue);

                                          bc_line = StringFormat("BC Line  %d",i);
                                          ObjectCreate(chart_id,bc_line,OBJ_TREND,0, B_time, B,C_time,C);
                                          ObjectSetInteger(chart_id,bc_line,OBJPROP_COLOR,clrBlue);

                                          fib_ab_ext_obj = StringFormat("AB Expansion  %d",i);
                                          ObjectCreate(chart_id,fib_ab_ext_obj,OBJ_EXPANSION,0,B_time,B,A_time,A,B_time,B);
                                          ObjectSetInteger(chart_id,fib_ab_ext_obj,OBJPROP_COLOR,clrBlue);
                                          for(int i = 0; i <= 2; i++)
                                            {

                                             ObjectSetInteger(chart_id,fib_ab_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);

                                            }

                                          fib_ab_ext_lvl = StringFormat("AB Expansion Levels %d",i);

                                          ObjectCreate(chart_id,fib_ab_ext_lvl,OBJ_RECTANGLE,0,A_time,fib_ext_c_161,C_time,fib_ext_c_224);
                                          ObjectSetInteger(chart_id,fib_ab_ext_lvl,OBJPROP_COLOR,clrBlue);

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
               break;
              }
           }
        }
     }
  }
```

Output:

![Figure 4. C](https://c.mql5.com/2/174/figure_4.png)

Explanation:

To prevent point C from extending too far past the AB distance, the maximum permitted Fibonacci extension level for the AB leg is set at 224%. To prevent point C from forming too near point B, the minimum Fibonacci extension level is also set at 161.8%. The legitimate pricing zone where point C is anticipated to occur is defined by these two levels taken together. Additionally, a number of variables are ready for calculation and display. For improved pattern structure clarity, they create the trend line connecting points B and C, save the computed Fibonacci extension levels, and control the Fibonacci expansion and its accompanying visual representation on the chart.

The program logs the price, time, and label for reference when it detects a possible swing high that meets the criteria for point C. To verify the legitimate range for point B and to ascertain the anticipated price zone for point C, the Fibonacci extension levels are then recalculated. These computations determine the 161.8% and 224% extension thresholds that specify where point C should ideally appear depending on the difference between points A and B. The Expert Advisor can verify whether the detected swing indeed falls within the permissible bounds of the 5-0 pattern with this procedure.

A conditional statement is then used by the computer to determine whether all the pattern points line up correctly. A is greater than X but lower than O, B is between 113% and 161.8% of XA, and C is between the 161.8% and 224% Fibonacci extension of AB. It confirms that X is lower than O. A legitimate 0XABC structure for a possible bullish 5-0 pattern is confirmed when all of these geometric and Fibonacci rules are met.

The EA graphically depicts the pattern on the chart after this requirement is satisfied. After creating a text label for C and drawing a line linking B and C, it displays the extension zone for C by adding a rectangle and a Fibonacci expansion object. The rectangle shows traders precisely where C has occurred in relation to AB, highlighting the range of 161.8% to 224%.

### **Identifying D**

Point D is the final wave in the bullish 5-0 harmonic pattern, completing the whole structure. Finding D inside the correct Fibonacci retracement range is the last step after identifying and validating points 0, X, A, B, and C. Point D is the pattern's completion zone and frequently denotes the region where traders expect a possible bullish reversal.

Point D must develop between the 50% and 55% Fibonacci retracement of the BC leg for a bullish 5-0 setup to be considered legitimate. The harmonic balance that establishes the pattern's structure is provided by this retracement zone. D should, in other words, retrace around half of the BC move, indicating that the downward correction from C might be coming to a close.

Example:

```
input double d_bc_max = 55.0; // MAX D RETRACEMENT LEVEL FOR BC
input double d_bc_min = 50.0; // MIN D RETRACEMENT LEVEL FOR BC
```

```
double fib_ret_d_50;
double fib_ret_d_55;
string fib_bc_ret_lvl;
string cd_line;
```

```
for(int n = m; n < bars_check - z; n++)
  {
   if(IsSwingLow(low, n, z) && low[n] < C)
     {

      D = low[n];
      D_time = time[n];
      D_letter = StringFormat("D  %d",l);
      cd_line = StringFormat("CD %d",i);

      fib_ext_b_113 =  MathAbs((((A - X) / 100) * (b_xa_min - 100)) - X);
      fib_ext_b_161 =   MathAbs((((A - X) / 100) * (b_xa_max - 100)) - X);

      fib_ext_c_161 = MathAbs((((A - B) / 100) * (c_ab_min - 100)) + A);
      fib_ext_c_224 =    MathAbs((((A - B) / 100) * (c_ab_max - 100)) + A);

      fib_ret_d_50 = C - ((d_bc_min / 100) * (C - B));
      fib_ret_d_55 = C - ((d_bc_max / 100) * (C - B));

      if(X < O && A > X && A < O && B <= fib_ext_b_113 && B >= fib_ext_b_161 && C >= fib_ext_c_161 && C <= fib_ext_c_224 && D <= fib_ret_d_50 && D >= fib_ret_d_55)
        {
         ObjectCreate(chart_id,O_letter,OBJ_TEXT,0,O_time,O);
         ObjectSetString(chart_id,O_letter,OBJPROP_TEXT,"0");
         ObjectSetInteger(chart_id,O_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,X_time,X);
         ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
         ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,A_time,A);
         ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
         ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,B_time,B);
         ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
         ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrBlue);

         ox_line = StringFormat("0X Line  %d",i);
         xa_line = StringFormat("XA Line  %d",i);
         ab_line = StringFormat("AB Line  %d",i);

         ObjectCreate(chart_id,ox_line,OBJ_TREND,0, O_time, O,X_time,X);
         ObjectSetInteger(chart_id,ox_line,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,xa_line,OBJ_TREND,0, X_time, X,A_time,A);
         ObjectSetInteger(chart_id,xa_line,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,ab_line,OBJ_TREND,0, A_time, A,B_time,B);
         ObjectSetInteger(chart_id,ab_line,OBJPROP_COLOR,clrBlue);

         fib_xa_ext_obj = StringFormat("XA Expansion  %d",i);
         ObjectCreate(chart_id,fib_xa_ext_obj,OBJ_EXPANSION,0,A_time,A,X_time,X,A_time,A);
         ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_COLOR,clrBlue);
         for(int i = 0; i <= 2; i++)
           {
            ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);
           }
         fib_xa_ext_lvl = StringFormat("XA Expansion Levels %d",i);

         ObjectCreate(chart_id,fib_xa_ext_lvl,OBJ_RECTANGLE,0,X_time,fib_ext_b_113,B_time,fib_ext_b_161);
         ObjectSetInteger(chart_id,fib_xa_ext_lvl,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,C_time,C);
         ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
         ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrBlue);

         bc_line = StringFormat("BC Line  %d",i);
         ObjectCreate(chart_id,bc_line,OBJ_TREND,0, B_time, B,C_time,C);
         ObjectSetInteger(chart_id,bc_line,OBJPROP_COLOR,clrBlue);

         fib_ab_ext_obj = StringFormat("AB Expansion  %d",i);
         ObjectCreate(chart_id,fib_ab_ext_obj,OBJ_EXPANSION,0,B_time,B,A_time,A,B_time,B);
         ObjectSetInteger(chart_id,fib_ab_ext_obj,OBJPROP_COLOR,clrBlue);
         for(int i = 0; i <= 2; i++)
           {

            ObjectSetInteger(chart_id,fib_ab_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);

           }

         fib_ab_ext_lvl = StringFormat("AB Expansion Levels %d",i);

         ObjectCreate(chart_id,fib_ab_ext_lvl,OBJ_RECTANGLE,0,A_time,fib_ext_c_161,C_time,fib_ext_c_224);
         ObjectSetInteger(chart_id,fib_ab_ext_lvl,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,D_time,D);
         ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
         ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrBlue);

         cd_line = StringFormat("CD Line  %d",i);
         ObjectCreate(chart_id,cd_line,OBJ_TREND,0, C_time, C,D_time,D);
         ObjectSetInteger(chart_id,cd_line,OBJPROP_COLOR,clrBlue);

         fib_bc_ret_lvl = StringFormat("BC RETRACEMENT Levels %d",i);

         ObjectCreate(chart_id,fib_bc_ret_lvl,OBJ_RECTANGLE,0,B_time,fib_ret_d_50,D_time,fib_ret_d_55);
         ObjectSetInteger(chart_id,fib_bc_ret_lvl,OBJPROP_COLOR,clrBlue);

        }

      break;
     }
  }
```

Output:

![Figure 5. D](https://c.mql5.com/2/174/Figure_5.png)

Explanation:

Fib\_ret\_d\_50 and fib\_ret\_d\_55 are two variables that are used to determine the precise retracement levels at which D may occur. The computation multiplies the difference between C and B by the decimal representation of the retracement percentage, then subtracts the result from C. This provides the price levels that delineate point D's retracement zone. D's proper placement within the anticipated Fibonacci retracement limit is guaranteed by the formula.

The program determines whether the entire 5-0 pattern structure satisfies the necessary harmonic requirements after the retracement levels have been determined. Point D is inside the 50% to 55% retracement range of BC, while points B and C extend between 113% and 161.8% of the XA leg, point C stands between 161.8% and 224% of the AB extension, and point X is lower than O, according to the circumstances. The structure is only considered a legitimate 5-0 pattern when all of these relationships are met.

Following validation, the computer uses the chart to visually depict the structure. At the D position, a text label "D" is made to indicate its location. The last section of the pattern is then obviously displayed when a line is drawn joining C and D to produce the CD leg. To visually indicate the possible completion zone for point D, a rectangle is also created to emphasize the 50% to 55% retracement area. This makes it simple for traders to spot potential reversals or completions in real time.

Verifying that each identified point accurately depicts the prevailing swing inside its corresponding leg is the next step. Because slight price changes can skew the overall pattern structure and produce confusion or misleading signals, simply recording highs and lows is insufficient. Clear guidelines must be developed to verify each swing point to preserve accuracy.

It must be the highest high between points 0 and X, starting at point 0. This guarantees that a genuine swing high, rather than a slight price rise, marks the start of the first downward leg (0X). However, among points X and A, point X must be the lowest low. This demonstrates that the XA leg begins with a true swing low as opposed to a transient drop.

In a similar vein, to guarantee that the AB leg reflects a legitimate swing peak, point A must be the highest high between points A and B. The BC leg must then retrace from a dominant trough if point B is the lowest low between points B and C. To guarantee that the CD leg begins from a true swing high, point C must be the highest high between points C and D.

Example:

```
int c_d_bars;
int c_highest_index;
double c_d_hh;
datetime c_d_hh_t;

int b_c_bars;
int b_lowest_index;
double b_c_ll;
datetime b_c_ll_t;

int a_b_bars;
int a_highest_index;
double a_b_hh;
datetime a_b_hh_t;

int x_a_bars;
int x_lowest_index;
double x_a_ll;
datetime x_a_ll_t;

int o_x_bars;
int o_highest_index;
double o_x_hh;
datetime o_x_hh_t;
```

```
for(int n = m; n < bars_check - z; n++)
  {
   if(IsSwingLow(low, n, z) && low[n] < C)
     {

      D = low[n];
      D_time = time[n];
      D_letter = StringFormat("D  %d",l);
      cd_line = StringFormat("CD %d",i);

      c_d_bars = Bars(_Symbol,PERIOD_CURRENT,C_time,D_time);
      c_highest_index = ArrayMaximum(high,m,c_d_bars);
      c_d_hh = high[c_highest_index];
      c_d_hh_t = time[c_highest_index];

      b_c_bars = Bars(_Symbol,PERIOD_CURRENT,B_time,c_d_hh_t);
      b_lowest_index = ArrayMinimum(low,l,b_c_bars);
      b_c_ll = low[b_lowest_index];
      b_c_ll_t = time[b_lowest_index];

      a_b_bars = Bars(_Symbol,PERIOD_CURRENT,A_time,b_c_ll_t);
      a_highest_index = ArrayMaximum(high,k,a_b_bars);
      a_b_hh = high[a_highest_index];
      a_b_hh_t = time[a_highest_index];

      x_a_bars = Bars(_Symbol,PERIOD_CURRENT,X_time,a_b_hh_t);
      x_lowest_index = ArrayMinimum(low,j,x_a_bars);
      x_a_ll = low[x_lowest_index];
      x_a_ll_t = time[x_lowest_index];

      o_x_bars = Bars(_Symbol,PERIOD_CURRENT,O_time,x_a_ll_t);
      o_highest_index = ArrayMaximum(high,a,o_x_bars);
      o_x_hh = high[o_highest_index];
      o_x_hh_t = time[o_highest_index];

      fib_ext_b_113 =  MathAbs((((a_b_hh - x_a_ll) / 100) * (b_xa_min - 100)) - x_a_ll);
      fib_ext_b_161 =   MathAbs((((a_b_hh - x_a_ll) / 100) * (b_xa_max - 100)) - x_a_ll);

      fib_ext_c_161 = MathAbs((((a_b_hh - b_c_ll) / 100) * (c_ab_min - 100)) + a_b_hh);
      fib_ext_c_224 =    MathAbs((((a_b_hh - b_c_ll) / 100) * (c_ab_max - 100)) + a_b_hh);

      fib_ret_d_50 = c_d_hh - ((d_bc_min / 100) * (c_d_hh - b_c_ll));
      fib_ret_d_55 = c_d_hh - ((d_bc_max / 100) * (c_d_hh - b_c_ll));

      if(x_a_ll < o_x_hh && a_b_hh > x_a_ll && a_b_hh < o_x_hh && b_c_ll <= fib_ext_b_113 && b_c_ll >= fib_ext_b_161 && c_d_hh >= fib_ext_c_161 && c_d_hh <= fib_ext_c_224 && D <= fib_ret_d_50 && D >= fib_ret_d_55)
        {
         ObjectCreate(chart_id,O_letter,OBJ_TEXT,0,o_x_hh_t,o_x_hh);
         ObjectSetString(chart_id,O_letter,OBJPROP_TEXT,"0");
         ObjectSetInteger(chart_id,O_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,x_a_ll_t,x_a_ll);
         ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
         ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,a_b_hh_t,a_b_hh);
         ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
         ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,b_c_ll_t,b_c_ll);
         ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
         ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrBlue);

         ox_line = StringFormat("0X Line  %d",i);
         xa_line = StringFormat("XA Line  %d",i);
         ab_line = StringFormat("AB Line  %d",i);

         ObjectCreate(chart_id,ox_line,OBJ_TREND,0, o_x_hh_t, o_x_hh,x_a_ll_t,x_a_ll);
         ObjectSetInteger(chart_id,ox_line,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,xa_line,OBJ_TREND,0, x_a_ll_t, x_a_ll,a_b_hh_t,a_b_hh);
         ObjectSetInteger(chart_id,xa_line,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,ab_line,OBJ_TREND,0, a_b_hh_t, a_b_hh,b_c_ll_t,b_c_ll);
         ObjectSetInteger(chart_id,ab_line,OBJPROP_COLOR,clrBlue);

         fib_xa_ext_obj = StringFormat("XA Expansion  %d",i);
         ObjectCreate(chart_id,fib_xa_ext_obj,OBJ_EXPANSION,0,a_b_hh_t,a_b_hh,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh);
         ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_COLOR,clrBlue);
         for(int i = 0; i <= 2; i++)
           {
            ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);
           }

         fib_xa_ext_lvl = StringFormat("XA Expansion Levels %d",i);

         ObjectCreate(chart_id,fib_xa_ext_lvl,OBJ_RECTANGLE,0,x_a_ll_t,fib_ext_b_113,b_c_ll_t,fib_ext_b_161);
         ObjectSetInteger(chart_id,fib_xa_ext_lvl,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,c_d_hh_t,c_d_hh);
         ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
         ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrBlue);

         bc_line = StringFormat("BC Line  %d",i);
         ObjectCreate(chart_id,bc_line,OBJ_TREND,0, b_c_ll_t, b_c_ll,c_d_hh_t,c_d_hh);
         ObjectSetInteger(chart_id,bc_line,OBJPROP_COLOR,clrBlue);

         fib_ab_ext_obj = StringFormat("AB Expansion  %d",i);
         ObjectCreate(chart_id,fib_ab_ext_obj,OBJ_EXPANSION,0,b_c_ll_t,b_c_ll,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll);
         ObjectSetInteger(chart_id,fib_ab_ext_obj,OBJPROP_COLOR,clrBlue);
         for(int i = 0; i <= 2; i++)
           {
            ObjectSetInteger(chart_id,fib_ab_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);
           }

         fib_ab_ext_lvl = StringFormat("AB Expansion Levels %d",i);

         ObjectCreate(chart_id,fib_ab_ext_lvl,OBJ_RECTANGLE,0,a_b_hh_t,fib_ext_c_161,c_d_hh_t,fib_ext_c_224);
         ObjectSetInteger(chart_id,fib_ab_ext_lvl,OBJPROP_COLOR,clrBlue);

         ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,D_time,D);
         ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
         ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrBlue);

         cd_line = StringFormat("CD Line  %d",i);
         ObjectCreate(chart_id,cd_line,OBJ_TREND,0, C_time, C,D_time,D);
         ObjectSetInteger(chart_id,cd_line,OBJPROP_COLOR,clrBlue);

         fib_bc_ret_lvl = StringFormat("BC RETRACEMENT Levels %d",i);

         ObjectCreate(chart_id,fib_bc_ret_lvl,OBJ_RECTANGLE,0,b_c_ll_t,fib_ret_d_50,D_time,fib_ret_d_55);
         ObjectSetInteger(chart_id,fib_bc_ret_lvl,OBJPROP_COLOR,clrBlue);

        }
      break;
     }
  }
```

Explanation:

To make sure that every point accurately depicts a market swing, the same validation procedure is used for the remaining legs. The algorithm determines the lowest low between X and A for the XA leg, so validating X as a swing low. To guarantee that 0 represents the dominant swing high at the beginning of the pattern, the 0X leg similarly seeks the highest high between 0 and X. This method of evaluating each point keeps the pattern structure steady and dependable while removing false swings that could skew the accuracy of identification.

The program makes sure that all the recently verified highs and lows are incorporated in later computations after validating each dominating swing point. This improvement makes it possible to more precisely recalculate the Fibonacci extension and retracement levels, guaranteeing that the proportions of each leg correspond to the actual market fluctuations. After this recalibration is finished, the software verifies that every point in the bullish 5-0 pattern still has the proper order and linkages. The method verifies that a legitimate 5-0 structure has developed on the chart once all geometric and Fibonacci requirements have been met.

### **Trade Execution**

After confirming that the bullish 5-0 harmonic pattern has been correctly identified, the next step is to execute trades based on the structure. In this setup, the trade entry occurs when the price reaches point D, which represents the potential completion of the pattern and a possible bullish reversal zone. In MQL5, this can be implemented by checking if the current price is around point D and then opening a buy order with predefined parameters such as lot size, stop loss, and take profit.

Example:

```
#include <Trade/Trade.mqh>
CTrade trade;
input double lot_size = 0.6;
```

```
datetime time_price[];
double ask_price;
double take_p;
datetime last_trade_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   ArraySetAsSeries(time_price,true);

//---
   return(INIT_SUCCEEDED);
  }
```

```
CopyTime(_Symbol, timeframe, 0, 2, time_price);
ask_price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
datetime current_bar_time = iTime(_Symbol,timeframe,0);
```

```
if(x_a_ll < o_x_hh && a_b_hh > x_a_ll && a_b_hh < o_x_hh && b_c_ll <= fib_ext_b_113 && b_c_ll >= fib_ext_b_161 && c_d_hh >= fib_ext_c_161 && c_d_hh <= fib_ext_c_224 && D <= fib_ret_d_50 && D >= fib_ret_d_55)
  {

   ObjectCreate(chart_id,O_letter,OBJ_TEXT,0,o_x_hh_t,o_x_hh);
   ObjectSetString(chart_id,O_letter,OBJPROP_TEXT,"0");
   ObjectSetInteger(chart_id,O_letter,OBJPROP_COLOR,clrBlue);

   ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,x_a_ll_t,x_a_ll);
   ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
   ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,clrBlue);

   ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,a_b_hh_t,a_b_hh);
   ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
   ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,clrBlue);

   ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,b_c_ll_t,b_c_ll);
   ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
   ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,clrBlue);

   ox_line = StringFormat("0X Line  %d",i);
   xa_line = StringFormat("XA Line  %d",i);
   ab_line = StringFormat("AB Line  %d",i);

   ObjectCreate(chart_id,ox_line,OBJ_TREND,0, o_x_hh_t, o_x_hh,x_a_ll_t,x_a_ll);
   ObjectSetInteger(chart_id,ox_line,OBJPROP_COLOR,clrBlue);

   ObjectCreate(chart_id,xa_line,OBJ_TREND,0, x_a_ll_t, x_a_ll,a_b_hh_t,a_b_hh);
   ObjectSetInteger(chart_id,xa_line,OBJPROP_COLOR,clrBlue);

   ObjectCreate(chart_id,ab_line,OBJ_TREND,0, a_b_hh_t, a_b_hh,b_c_ll_t,b_c_ll);
   ObjectSetInteger(chart_id,ab_line,OBJPROP_COLOR,clrBlue);

   fib_xa_ext_obj = StringFormat("XA Expansion  %d",i);
   ObjectCreate(chart_id,fib_xa_ext_obj,OBJ_EXPANSION,0,a_b_hh_t,a_b_hh,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh);
   ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_COLOR,clrBlue);
   for(int i = 0; i <= 2; i++)
     {
      ObjectSetInteger(chart_id,fib_xa_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);
     }

   fib_xa_ext_lvl = StringFormat("XA Expansion Levels %d",i);

   ObjectCreate(chart_id,fib_xa_ext_lvl,OBJ_RECTANGLE,0,x_a_ll_t,fib_ext_b_113,b_c_ll_t,fib_ext_b_161);
   ObjectSetInteger(chart_id,fib_xa_ext_lvl,OBJPROP_COLOR,clrBlue);

   ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,c_d_hh_t,c_d_hh);
   ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
   ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,clrBlue);

   bc_line = StringFormat("BC Line  %d",i);
   ObjectCreate(chart_id,bc_line,OBJ_TREND,0, b_c_ll_t, b_c_ll,c_d_hh_t,c_d_hh);
   ObjectSetInteger(chart_id,bc_line,OBJPROP_COLOR,clrBlue);

   fib_ab_ext_obj = StringFormat("AB Expansion  %d",i);
   ObjectCreate(chart_id,fib_ab_ext_obj,OBJ_EXPANSION,0,b_c_ll_t,b_c_ll,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll);
   ObjectSetInteger(chart_id,fib_ab_ext_obj,OBJPROP_COLOR,clrBlue);
   for(int i = 0; i <= 2; i++)
     {
      ObjectSetInteger(chart_id,fib_ab_ext_obj,OBJPROP_LEVELCOLOR,i,clrBlue);
     }

   fib_ab_ext_lvl = StringFormat("AB Expansion Levels %d",i);

   ObjectCreate(chart_id,fib_ab_ext_lvl,OBJ_RECTANGLE,0,a_b_hh_t,fib_ext_c_161,c_d_hh_t,fib_ext_c_224);
   ObjectSetInteger(chart_id,fib_ab_ext_lvl,OBJPROP_COLOR,clrBlue);

   ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,D_time,D);
   ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
   ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,clrBlue);

   cd_line = StringFormat("CD Line  %d",i);
   ObjectCreate(chart_id,cd_line,OBJ_TREND,0, c_d_hh_t, c_d_hh,D_time,D);
   ObjectSetInteger(chart_id,cd_line,OBJPROP_COLOR,clrBlue);

   fib_bc_ret_lvl = StringFormat("BC RETRACEMENT Levels %d",i);

   ObjectCreate(chart_id,fib_bc_ret_lvl,OBJ_RECTANGLE,0,b_c_ll_t,fib_ret_d_50,D_time,fib_ret_d_55);
   ObjectSetInteger(chart_id,fib_bc_ret_lvl,OBJPROP_COLOR,clrBlue);

   if(time[n+z] == time_price[1] && close[n+z] > D && current_bar_time != last_trade_time)
     {
      take_p = ask_price + (MathAbs(ask_price - D) * 3);

      trade.Buy(lot_size,_Symbol,ask_price,D,take_p);

      last_trade_time = current_bar_time;

     }
  }
```

Output:

![Figure 6. Trade Execution](https://c.mql5.com/2/174/figure_6.png)

Explanation:

Following confirmation of the bullish 5-0 harmonic pattern, trade execution is handled by this section of the software. The MQL5 trading library, which supplies the CTrade class needed to transmit trade orders to the broker, is imported in the first line, #include <Trade/Trade.mqh>. After creating an instance of that class, the line CTrade trade; enables the program to carry out trade operations like buy and sell. Each trade that is opened will employ 0.6 lots, as defined by the input variable lot\_size = 0.6.

To control trade execution and prevent duplicate entries, the program defines a number of variables. Ask\_price saves the current market ask price, take\_p is used to determine the take-profit objective, last\_trade\_time logs the time of the most recent trade, and the array time\_price\[\] tracks candle times. To align with MQL5's handling of time series data, the time\_price array is initialized in the order of newest to oldest. The software retrieves the most recent candle times and market price for each tick.

The current candle time must match the most recent copied time, the close price must be above point D, and no trade should have been executed on the same candle. These criteria are then checked to see if a trade should be opened. To avoid multiple entries within the same bar, the computer makes a purchase transaction if these requirements are met, determines a take-profit target three times the distance between the current ask price and point D, and logs the trade time.

### **Conclusion**

This article explained how to build an Expert Advisor that detects and trades the 5-0 harmonic pattern. It covered how to identify and validate each swing point, confirm the pattern’s structure, execute trades automatically, and display the pattern on the chart using graphical objects such as trend lines and labels. This marks the final part of the Advanced Chart Patterns series. In the next part, new aspects of MQL5 will be explored.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19856.zip "Download all attachments in the single ZIP archive")

[Project\_14\_5-0\_Pattern.mq5](https://www.mql5.com/en/articles/download/19856/Project_14_5-0_Pattern.mq5 "Download Project_14_5-0_Pattern.mq5")(17.6 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/497229)**
(2)


![kimo161](https://c.mql5.com/avatar/2022/2/61F9797C-FA84.jpg)

**[kimo161](https://www.mql5.com/en/users/kimo161)**
\|
24 Dec 2025 at 15:10

Eliot waves?

Yes, they're Eliot waves. And don't invent new entities, William Occam disapproves.

![Silk Road Trading LLC](https://c.mql5.com/avatar/2025/5/68239006-fc9d.png)

**[Ryan L Johnson](https://www.mql5.com/en/users/rjo)**
\|
24 Dec 2025 at 16:32

**kimo161 [#](https://www.mql5.com/en/forum/497229#comment_58799778):**

Eliot waves?

Yes, they're Eliot waves. And don't invent new entities, William Occam disapproves.

It's more nuanced than that. Harmonic patterns, including the 5-0 Pattern ( [Trademark Status & Document Retrieval](https://www.mql5.com/go?link=https://tsdr.uspto.gov/%23caseNumber%3d77001230%26caseSearchType%3dUS_APPLICATION%26caseType%3dDEFAULT%26searchType%3dstatusSearch "https://tsdr.uspto.gov/#caseNumber=77001230&caseSearchType=US_APPLICATION&caseType=DEFAULT&searchType=statusSearch")), were invented after Elliot Waves were invented. Elliot Waves are based on 9 degrees of waves in cycles, while the 5-0 Pattern is based on Fibonacci ratios. There is bound to be some overlap when both patterns are applied to a chart--which can lead to confusion.


![Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://c.mql5.com/2/173/19623-perehodim-na-mql5-algo-forge-logo.png)[Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)

We'll continue developing the Simple Candles and Adwizard projects, while also describing the finer aspects of using the MQL5 Algo Forge version control system and repository.

![From Novice to Expert: Market Periods Synchronizer](https://c.mql5.com/2/174/19841-from-novice-to-expert-market-logo.png)[From Novice to Expert: Market Periods Synchronizer](https://www.mql5.com/en/articles/19841)

In this discussion, we introduce a Higher-to-Lower Timeframe Synchronizer tool designed to solve the problem of analyzing market patterns that span across higher timeframe periods. The built-in period markers in MetaTrader 5 are often limited, rigid, and not easily customizable for non-standard timeframes. Our solution leverages the MQL5 language to develop an indicator that provides a dynamic and visual way to align higher timeframe structures within lower timeframe charts. This tool can be highly valuable for detailed market analysis. To learn more about its features and implementation, I invite you to join the discussion.

![Overcoming The Limitation of Machine Learning (Part 5): A Quick Recap of Time Series Cross Validation](https://c.mql5.com/2/174/19775-overcoming-the-limitation-of-logo__1.png)[Overcoming The Limitation of Machine Learning (Part 5): A Quick Recap of Time Series Cross Validation](https://www.mql5.com/en/articles/19775)

In this series of articles, we look at the challenges faced by algorithmic traders when deploying machine-learning-powered trading strategies. Some challenges within our community remain unseen because they demand deeper technical understanding. Today’s discussion acts as a springboard toward examining the blind spots of cross-validation in machine learning. Although often treated as routine, this step can easily produce misleading or suboptimal results if handled carelessly. This article briefly revisits the essentials of time series cross-validation to prepare us for more in-depth insight into its hidden blind spots.

![Reusing Invalidated Orderblocks As Mitigation Blocks (SMC)](https://c.mql5.com/2/174/19619-reusing-invalidated-orderblocks-logo__1.png)[Reusing Invalidated Orderblocks As Mitigation Blocks (SMC)](https://www.mql5.com/en/articles/19619)

In this article, we explore how previously invalidated orderblocks can be reused as mitigation blocks within Smart Money Concepts (SMC). These zones reveal where institutional traders re-enter the market after a failed orderblock, providing high-probability areas for trade continuation in the dominant trend.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/19856&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049057609568330791)

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