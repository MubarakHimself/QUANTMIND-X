---
title: Introduction to MQL5 (Part 16): Building Expert Advisors Using Technical Chart Patterns
url: https://www.mql5.com/en/articles/18147
categories: Trading, Indicators, Expert Advisors
relevance_score: 10
scraped_at: 2026-01-22T17:18:53.579322
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/18147&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049038539913536457)

MetaTrader 5 / Examples


### Introduction

Welcome back to Part 16 of the Introduction to MQL5 series! This part promises to be especially interesting as we continue building on everything we’ve learned so far, using a project-based approach as usual. Together, we’ll work on a practical project that combines technical analysis patterns and MQL5 coding to create an Expert Advisor, helping you deepen your skills through hands-on experience.

We will focus on the Head and Shoulders pattern — a popular technical pattern used to identify potential trend reversals. Our project will be designed as an Expert Advisor that can automatically recognize this pattern and execute trades accordingly. Additionally, it will serve as an indicator to visually highlight the Head and Shoulders formation on the chart, making it easier for you to spot and understand the pattern in real trading scenarios.

In this article, you'll learn:

- Automating Chart Pattern Trading
- How to Identify the Head and Shoulders Pattern
- Drawing Swing Points Programmatically
- Using Chart Objects in MQL5
- Defining Entry, Stop Loss, and Take Profit
- Avoiding Signal Repetition

### **1\. Understanding Chart Patterns**

Visual forms on price charts that can be used to predict future market moves are called chart patterns. These trends, which are frequently indicators of whether a trend is likely to continue or reverse, are the product of the continuous struggle between buyers and sellers. A Head and Shoulders pattern, for instance, usually signals a possible reversal from an upward trend to a downward trend, whereas a flag pattern during an upward trend frequently implies the trend is likely to continue. Your ability to detect possible trade chances based on past price behavior will improve if you can identify chart patterns such as Triangles, Rectangles, and Double Tops.

**Analogy**

Similar to footprints in the sand, chart patterns are obvious hints left by the continuous conflict between buyers and sellers. Chart patterns help you understand what has happened in the market and predict what might happen next, much like animal trails can help an experienced tracker determine which animal has gone by and where it is going. These graphic patterns on price charts show how market participants behave collectively and frequently indicate whether the present trend is likely to continue or reverse.

**1.1.** **Categories of Patterns**

Chart patterns generally fall into three main categories:

**1.1.1.** **Reversal Patterns**

Chart patterns known as reversal patterns can be used by to spot possible market turning points. They indicate that a new trend may be beginning in the opposite direction and that the existing trend may be ending. For instance, a reversal pattern can suggest that the price is getting ready to go into an uptrend if the market is in a decline. Similarly, these patterns may indicate a potential reversal of the market's rising trajectory.  The Head and Shoulders and the Double Bottom are typical reversal patterns.

**Analogy**

Reversal patterns function as indicators of a possible shift in the direction of the market. These patterns suggest that a trend, whether upward or downward, may be waning and ready to reverse, much like footprints turning around on a beach signal someone has changed direction. They assist traders in identifying potential changes in price movement by pointing out when buyers or sellers are beginning to exert power.

![Figure 1. Reversal Patterns](https://c.mql5.com/2/143/Figure_1.png)

**1.1.2.** **Continuation Patterns**

Chart patterns known as continuation patterns indicate when the market is most likely to stay on its present direction. They typically show up at times of consolidation or short stops in price movement before the trend resumes. For instance, a continuation pattern in an uptrend can suggest that the price will keep increasing following a little sideways dip. In a downtrend, it implies that the price will probably continue to decline following the pause. Rectangles and flags are typical patterns of continuation.

**Analogy**

Similar to footprints, continuation patterns pause briefly but continue to point in the same direction. Consider yourself following a person as they stroll along the shore. Following a continuous line of footsteps, you observe that the prints pause for a moment — possibly to tie their shoelaces or take a look around — but then they resume in the same direction. This indicates that they were just taking a brief pause and never changed their minds. Similarly, continuation patterns indicate that the market is paused for a short time but will probably continue on its current trajectory. These patterns demonstrate that momentum has not reversed and is only momentarily resting, regardless of whether the market is heading higher or lower.

![Figure 2. Continuation Pattern](https://c.mql5.com/2/143/Figure_2.png)

**1.1.3. Neutral Patterns**

Chart patterns that show a period of consolidation during which the market may break out in any direction are known as neutral patterns. With neither side clearly dominating, these patterns show a balance between buyers and sellers. Because of this, traders are forced to wait for a confirmed breakout before acting, and the price moves within a tighter range. Although they can't predict the breakout's path, neutral patterns aid traders in getting ready for a possible move. Neutral patterns are frequently seen, such as the Symmetrical Triangle.

**Analogy**

In trading, neutral patterns are similar to observing someone hesitating on a beach, uncertain of their next move. Neutral patterns, in which buyers and sellers are evenly matched, reflect market hesitancy, much like you can't predict their next move unless they commit. Although there is no obvious bias toward either up or down, these patterns indicate that a breakthrough could occur in either direction.

![Figure 3. Neutral Pattern](https://c.mql5.com/2/143/figure_3.png)

### **2\. Setting up the Project**

**2.1. How the EA Works**

The Expert Advisor (EA) in this project is built to automatically identify the Head and Shoulders pattern in the market and execute trades based on that structure. Whether it is a standard Head and Shoulders or an Inverse Head and Shoulders, the EA will identify a valid pattern, choose the best entry point, and execute a trade in the expected breakout direction.

But the EA doesn't stop there. To make the pattern more visible and understandable on the chart, we will also use graphical objects to mark the left shoulder, head, and right shoulder. These visual markers help confirm the pattern to the trader and improve clarity when reviewing the chart or back testing the strategy. This method offers a visible layer of confirmation in addition to automation and pattern recognition, which can improve debugging, learning, and even real-time monitoring.

**2.1.1. Logic for Buy**

To trigger a buy trade, the EA will identify an Inverse Head and Shoulders pattern by detecting six specific swing points in sequence:

- **Swing High (X):** An initial high, labeled X.
- **Swing Low (A):** A lower low after X, labeled A.
- **Swing High (B):** A rebound forming a lower high than X but higher than A, labeled B.
- **Swing Low (C):** A deeper low than A, labeled C – this is the Head.
- **Swing High (D):** A swing high roughly at the same level as B, labeled D – this forms part of the Neckline.
- **Swing Low (E):** A higher low compared to C, and around the same level as A – this is the second Shoulder.

Once this structure is in place:

- The EA waits for a candle to close above point D (the neckline).
- When that happens, it will execute a buy trade.
- The Stop Loss (SL) will be set at the low of point E.
- The Take Profit (TP) will initially be set at point X (the first swing high).

However, if the distance between the entry point and TP (X) is **less than 1x the stop loss distance** **,** the EA will **ignore X as the target** and instead set a fixed **1:3 risk-reward** target based on the SL distance. This ensures the strategy maintains a minimum risk-reward ratio and avoids taking low-reward trades that aren't worth the risk.

![Figure 4. Long Position](https://c.mql5.com/2/143/figure_4.png)

**2.1.1. Logic for Sell**

To trigger a sell trade, the EA will identify a standard Head and Shoulders pattern by detecting six specific swing points in sequence:

- **Swing Low (X):** An initial low, labeled X.
- **Swing High (A):** A higher high after X, labeled A.
- **Swing Low (B):** A pullback forming a higher low than X but lower than A, labeled B.
- **Swing High (C):** A higher high than A, labeled C — this is the Head.
- **Swing Low (D):** A swing low roughly at the same level as B, labeled D — this forms part of the Neckline.
- **Swing High (E):** A lower high compared to C, and around the same level as A — this is the second Shoulder.

Once this structure is confirmed:

- The EA waits for a candle to close below point D (the neckline).
- When that happens, it executes a sell trade.
- The Stop Loss (SL) is placed at the high of point E.
- The Take Profit (TP) is initially set at point X (the first swing low).

If the distance from the entry to TP (X) is less than one times the stop‑loss distance, the EA overrides X as the target and applies a fixed 1:3 risk‑reward objective based on the SL size.

![Figure 5. Short Position](https://c.mql5.com/2/143/Figure_5.png)

**Note:** _Developing your knowledge of MQL5 programming ideas, particularly how to work with chart patterns and create useful Expert Advisors, is the primary goal of the trading strategy that will be explored in this project. It is not meant to be used with real money or for live trading. Before implementing any technique in a live market, always conduct a comprehensive back-test and get advice from a financial expert._

### **3\. Identifying the Head and Shoulders Patterns on the Chart**

By this stage, I believe you have a firm grasp on the idea of chart patterns and know exactly what we want our Expert Advisor (EA) to perform. Even before the EA makes a trade, it's critical to be able to see the Head and Shoulders pattern on the chart. This helps you identify any runtime faults or logical problems early in the testing process, in addition to confirming that the EA logic matches the real price action.

This chapter will cover how to manually highlight and validate the Head and Shoulders structure on the chart using various chart components, including trend lines, text labels, and shapes. Later on in the EA, this will assist create a strong basis for automating the detection.

**3.1. Retrieving Candlestick Data**

The first step is to obtain candlestick data to locate the Head and Shoulders pattern on the chart. This contains details about each bar's open, high, low, time, and close prices. These values are crucial because the way the price swings throughout several candles defines the pattern's structure. However, we're not merely gathering this information to identify trends. We also want it to act like an indicator, since we want to create an Expert Advisor that can recognize these patterns and make trades on its own. This implies that it must to be able to draw attention to previous Head and Shoulders patterns on the chart so that you can examine and test them again.

In short, the EA we are developing will have two functions: it will be a trading bot that recognizes Head and Shoulders patterns and automatically executes trades, and it will also be a pattern indicator by highlighting similar structures in historical data. In addition to enabling automated trading, this dual functionality enables traders to visually verify signals and examine how the pattern has developed and performed in the past.

**Example:**

```
input ENUM_TIMEFRAMES    timeframe = PERIOD_CURRENT; // MA Time Frame
input int bars_check  = 1000; // Number of bars to check for swing points

// Variable to store how many bars are available on the chart for the selected timeframe
int rates_total;

double open[];   // Array for opening prices
double close[];  // Array for closing prices
double low[];    // Array for lowest prices
double high[];   // Array for highest prices
datetime time[]; // Array for time (timestamps) of each bar

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

// Get the total number of bars available on the selected symbol and timeframe
   rates_total = Bars(_Symbol, timeframe);

// Copy the open prices of the last 'rates_total' bars into the 'open' array
   CopyOpen(_Symbol, timeframe, 0, rates_total, open);

// Copy the close prices of the last 'rates_total' bars into the 'close' array
   CopyClose(_Symbol, timeframe, 0, rates_total, close);

// Copy the low prices of the last 'rates_total' bars into the 'low' array
   CopyLow(_Symbol, timeframe, 0, rates_total, low);

// Copy the high prices of the last 'rates_total' bars into the 'high' array
   CopyHigh(_Symbol, timeframe, 0, rates_total, high);

// Copy the time (timestamps) of the last 'rates_total' bars into the 'time' array
   CopyTime(_Symbol, timeframe, 0, rates_total, time);

  }
```

**Explanation:**

There are two user-defined input variables at the start of the function. The first one, timeframe, lets the user choose the precise time period that the Expert Advisor (EA) will utilize. PERIOD\_CURRENT is the default setting. Thus, the EA will utilize the same timeframe as the chart to which it is linked. You can examine several time periods without changing the code thanks to this flexibility. The EA is instructed by the second parameter, bars\_check, how many historical candlesticks (or bars) to look at while examining price behavior. The EA will look for possible pattern structures in the last 1000 candles, since it is set to 1000 in this instance.

The code declares a few arrays and a variable to hold market data after the input definitions. The total number of bars (candles) that are available for the selected symbol and timeframe will be stored in the rates\_total variable. Each candlestick's related price data is stored in arrays such as open\[\], close\[\], low\[\], and high\[\]. We may also determine the precise time of each candle by using the timestamps for each bar, which are stored in the time\[\] array. Because they give the EA the information they need to examine the chart and find patterns like the Head and Shoulders, these arrays are crucial.

This project requires us to manually replicate candlestick data using functions like CopyOpen(), CopyHigh(), and others because it is designed as an Expert Advisor (EA) rather than a custom indicator. This information would already be supplied automatically through the function parameters if it were an indicator that makes use of the OnCalculate() method, saving us the trouble of copying it.

Three common MQL5 functions — OnInit(), OnDeinit(), and OnTick() — are also included in the EA's structure. When the EA is loaded, the OnInit() method is called once. It simply returns INIT\_SUCCEEDED, signaling that the EA is prepared for execution. The OnDeinit() function, which does not yet have cleanup logic, is called when the terminal shuts down or the EA is removed.

The OnTick() function, which runs each time a new price update (tick) takes place, is where the actual activity takes place. The Bars() function initially ascertains the number of bars that are presently accessible on the chart inside this function. The price and time data are then loaded into the corresponding arrays using five copy functions: Opening prices are filled into the open\[\] array by CopyOpen(), closing prices are filled into the close\[\] array by CopyClose(), the lowest and highest prices are stored by CopyLow() and CopyHigh(), respectively, and the timestamp of each candle is stored by CopyTime(). Because it prepares all the historical data that the EA will utilize to look for chart patterns and examine market activity, this setup is essential.

**3.2. Identifying Swings**

Finding the important swing points — X, A, B, C, D, and E — that comprise the Head and Shoulders or Inverse Head and Shoulders pattern structure comes next once we have successfully collected the historical candlestick data. The EA must accurately identify these swings to determine legitimate chart patterns, which are the turning points in price movement. The EA will assist it map out the pattern step by step by identifying major reversals by examining highs and lows over the chosen number of bars.

**Example:**

```
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

**Explanation:**

We use two functions, IsSwingLow() and IsSwingHigh(), to determine price swing points. These features determine whether a given candlestick, in relation to its nearby candles, creates a swing high or swing low. According to the lookback value, the function guarantees that the low of the current candle is lower than the lows of a predetermined number of candles before and after it in the case of a swing low. Likewise, for a swing high, it verifies that the high of the current candle is greater than the highs of the surrounding candles. Since this idea has previously been covered in great detail in Part 14 of this series, we won't go into too much detail here.

**3.2.1. Identifying XABCDE**

The significance of precisely determining the primary swing points — X, A, B, C, and D — to recognize the Head and Shoulders pattern in both buy and sell settings is emphasized in the essay. These points stand for significant highs and lows that influence the pattern and direct the trading actions of the expert advisor.

**Example:**

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

// Get the total number of bars available on the selected symbol and timeframe
   rates_total = Bars(_Symbol, timeframe);

// Copy the open prices of the last 'rates_total' bars into the 'open' array
   CopyOpen(_Symbol, timeframe, 0, rates_total, open);

// Copy the close prices of the last 'rates_total' bars into the 'close' array
   CopyClose(_Symbol, timeframe, 0, rates_total, close);

// Copy the low prices of the last 'rates_total' bars into the 'low' array
   CopyLow(_Symbol, timeframe, 0, rates_total, low);

// Copy the high prices of the last 'rates_total' bars into the 'high' array
   CopyHigh(_Symbol, timeframe, 0, rates_total, high);

// Copy the time (timestamps) of the last 'rates_total' bars into the 'time' array
   CopyTime(_Symbol, timeframe, 0, rates_total, time);

//FOR SELL
   if(show_sell)
     {
      if(rates_total >= bars_check)
        {

         for(int z = 7; z <= 10; z++)
           {

            for(int i = rates_total - bars_check; i < rates_total - z; i++)
              {

               if(IsSwingLow(low, i, z))
                 {

                  // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
                  X = low[i]; // Price of the swing low (X).
                  X_time = time[i]; // Time of the swing low (X).
                  X_letter = StringFormat("X%d", i); // Unique name for the text label object.

                  for(int j = i; j < rates_total - z; j++)
                    {
                     if(IsSwingHigh(high, j, z) && time[j] > X_time)
                       {

                        A = high[j]; // Price of the swing high (A).
                        A_time = time[j]; // Time of the swing high (A)
                        A_letter = StringFormat("A%d", j); // Unique name for the text label object

                        for(int k = j; k < rates_total - z; k++)
                          {
                           if(IsSwingLow(low, k, z) && time[k] > A_time)
                             {

                              B = low[k]; // Price of the swing low (B).
                              B_time = time[k]; // Time of the swing low (B).
                              B_letter = StringFormat("B%d", k); // Unique name for the text label object.

                              for(int l = k ; l < rates_total - z; l++)
                                {

                                 if(IsSwingHigh(high, l, z) && time[l] > B_time)
                                   {

                                    C = high[l]; // Price of the swing high (C).
                                    C_time = time[l]; // Time of the swing high (C).
                                    C_letter = StringFormat("C%d", l); // Unique name for the text label object.

                                    for(int m = l; m < rates_total - z; m++)
                                      {

                                       if(IsSwingLow(low, m, z) && time[m] > C_time)
                                         {

                                          D = low[m]; // Price of the swing low (D).
                                          D_time = time[m]; // Time of the swing low (D).
                                          D_letter = StringFormat("D%d", m); // Unique name for the text label object.

                                          for(int n = m ; n < rates_total - (z/2) - 1; n++)
                                            {

                                             if(IsSwingHigh(high, n, (z/2)) && time[n] > D_time)
                                               {

                                                E = high[n]; // Price of the swing low (B).
                                                E_time = time[n]; // Time of the swing low (B).
                                                E_letter = StringFormat("E%d", n); // Unique name for the text label object.

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

        }
     }

  }
```

**Explanation:**

```
//X
double X; // Price of the swing low (X).
datetime X_time; // Time of the swing low (X).
string X_letter; // Unique name for the text label object.

//A
double A; // Price of the swing high (A).
datetime A_time; // Time of the swing high (A).
string A_letter; // Unique name for the text label object.

//B
double B; // Price of the swing low (B).
datetime B_time; // Time of the swing low (B).
string B_letter; // Unique name for the text label object.

//C
double C; // Price of the swing low (B).
datetime C_time; // Time of the swing low (B).
string C_letter; // Unique name for the text label object.

//D
double D; // Price of the swing low (B).
datetime D_time; // Time of the swing low (B).
string D_letter; // Unique name for the text label object.

//E
double E; // Price of the swing low (B).
datetime E_time; // Time of the swing low (B).
string E_letter; // Unique name for the text label object.
```

Three different kinds of variables are used in this code to represent each of the pattern's major swing points, which are X, A, B, C, D, and E. The precise price level of that swing point is first stored in a double variable. Double A; keeps the price of the swing high with the label A, whereas double X. holds the price of the swing low with the label X. Each swing point uses a datetime variable to record the time it occurred in addition to the price. This enables the EA to accurately arrange the swing points in chronological sequence on the chart. Datetime X\_time;, for example, records the time of the swing low X, and datetime A\_time; records the time of the swing high A.

Lastly, a distinct label name is generated for every swing point using a string variable. To visually indicate the locations of each swing point, these labels — such as X\_letter or A\_letter — are utilized to construct text objects on the chart. The EA is better able to organize and show these points so that traders can notice the pattern developing thanks to this labeling system. Price, time, and label are the three pieces of information that the EA uses to arrange each swing point so that it can correctly identify and graphically depict the Head and Shoulders pattern on the chart. Pattern identification and the placement of easily interpreted visual cues for traders both depend on this methodical technique.

```
if(show_sell)
  {
   if(rates_total >= bars_check)
     {

      for(int z = 7; z <= 10; z++)
        {

         for(int i = rates_total - bars_check; i < rates_total - z; i++)
           {

            if(IsSwingLow(low, i, z))
              {

               // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
               X = low[i]; // Price of the swing low (X).
               X_time = time[i]; // Time of the swing low (X).
               X_letter = StringFormat("X%d", i); // Unique name for the text label object.

               for(int j = i; j < rates_total - z; j++)
                 {
                  if(IsSwingHigh(high, j, z) && time[j] > X_time)
                    {

                     A = high[j]; // Price of the swing high (A).
                     A_time = time[j]; // Time of the swing high (A)
                     A_letter = StringFormat("A%d", j); // Unique name for the text label object

                     for(int k = j; k < rates_total - z; k++)
                       {
                        if(IsSwingLow(low, k, z) && time[k] > A_time)
                          {

                           B = low[k]; // Price of the swing low (B).
                           B_time = time[k]; // Time of the swing low (B).
                           B_letter = StringFormat("B%d", k); // Unique name for the text label object.

                           for(int l = k ; l < rates_total - z; l++)
                             {

                              if(IsSwingHigh(high, l, z) && time[l] > B_time)
                                {

                                 C = high[l]; // Price of the swing high (C).
                                 C_time = time[l]; // Time of the swing high (C).
                                 C_letter = StringFormat("C%d", l); // Unique name for the text label object.

                                 for(int m = l; m < rates_total - z; m++)
                                   {

                                    if(IsSwingLow(low, m, z) && time[m] > C_time)
                                      {

                                       D = low[m]; // Price of the swing low (D).
                                       D_time = time[m]; // Time of the swing low (D).
                                       D_letter = StringFormat("D%d", m); // Unique name for the text label object.

                                       for(int n = m ; n < rates_total - (z/2) - 1; n++)
                                         {

                                          if(IsSwingHigh(high, n, (z/2)) && time[n] > D_time)
                                            {

                                             E = high[n]; // Price of the swing low (B).
                                             E_time = time[n]; // Time of the swing low (B).
                                             E_letter = StringFormat("E%d", n); // Unique name for the text label object.

                                             ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,X_time,X);
                                             ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
                                             ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,txt_clr);

                                             ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,A_time,A);
                                             ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
                                             ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,txt_clr);

                                             ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,B_time,B);
                                             ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
                                             ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,txt_clr);

                                             ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,C_time,C);
                                             ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
                                             ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,txt_clr);

                                             ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,D_time,D);
                                             ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
                                             ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,txt_clr);

                                             ObjectCreate(chart_id,E_letter,OBJ_TEXT,0,E_time,E);
                                             ObjectSetString(chart_id,E_letter,OBJPROP_TEXT,"E");
                                             ObjectSetInteger(chart_id,E_letter,OBJPROP_COLOR,txt_clr);

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

     }
  }
```

**Output:**

![Figure 6. Identifying Swings](https://c.mql5.com/2/143/figure_6.png)

**Explanation:**

To determine whether the program is now configured to detect sell patterns, the code first determines whether the variable show\_sell is true. When the sale pattern detection is not needed, this simple conditional gate stops pointless calculations. It next checks to see if the total number of bars in the chart (rates\_total) exceeds or equals the bars\_check minimum threshold. This guarantees that there is sufficient previous data for accurate pattern recognition.

Typically used as a window size or lookback time for swing point detection, the variable z is iterated over a limited range of values in the outermost loop. The technique aims to improve the flexibility and precision of swing detection by experimenting with multiple lookback periods by operating between z = 7 and z = 10. This loop enables the algorithm to look for patterns at slightly different resolutions or sensitivities.

From rates\_total - bars\_check to rates\_total - z, the subsequent for loop iterates through the most recent bars on the chart. This range focuses on where possible patterns are more pertinent to the current price action by limiting the search region to the most recent bars. The function IsSwingLow(low, i, z) is used in this loop to search for a swing low and determine whether the bar at position i is a local minimum based on the lookback window z.

When a swing low is detected at position i, the code uses StringFormat("X%d", i) to generate a unique label string called X\_letter, stores its price in the variable X, and its timestamp in X\_time. The EA can more easily visually highlight this spot on the chart and identify it uniquely for processing at a later time thanks to this labeling. The Head and Shoulders pattern begins with this point X as its reference.

To determine the next swing high A that happens after time X, the code then starts another nested loop, beginning at i. To maintain the chronological sequence required for the pattern, it uses IsSwingHigh(high, j, z) to determine whether the bar at j is a swing high and makes sure its time time\[j\] is strictly greater than X\_time. The price, time, and label for A are documented similarly to X if a legitimate swing high is discovered.

The following points, B, C, D, and E, are identified in sequence by this nested loop structure. By iterating through the bars, each swing point is located, verifying that it is a legitimate swing low or high and that its timestamp is strictly greater than the time of the preceding point. The code creates a distinct label string and saves the price and duration for each confirmed swing. The pattern points are kept in the proper sequence by this rigorous sequential check.

The innermost loop uses a somewhat narrower lookback window (z/2) to seek for point E, the pattern's last swing high. The algorithm can adjust the detection sensitivity for the last point thanks to this discrepancy. When E is located, the code instantly exits the loop to save needless additional searching, increasing efficiency. It then assigns its price, time, and label.

The Head and Shoulders pattern is precisely detected in the right order of swing points throughout this process thanks to the nested loops and time checks. Using distinct labels like as "X%d" or "A%d" allows the EA to generate and control text or graphical elements on the chart, giving traders a visual representation of the pattern. The computer can dependably identify intricate price action patterns for use in trading decisions thanks to this methodical, structured methodology.

The purpose of the code sample was to identify the six crucial swing points — X, A, B, C, D, and E — that are necessary for identifying a Head and Shoulders pattern. The current implementation, however, does not yet apply the precise logical constraints necessary to validate the Head and Shoulders structure; instead, it only determines these swing points based on price swings and their chronological order. The algorithm gathers these points' prices, times, and labels, but it doesn't verify if they truly make up the distinctive pattern.

Several important price structure-based criteria will be used to verify the validity of a Head and Shoulders sell pattern. A must be higher than X, B must be in the middle (higher than X but lower than A), and C, the head, must be higher than B. The right shoulder's peak is formed by point E aligning with point A, while the right shoulder's trough is represented by point D being near point B. Furthermore, before sending out any trade signals, a particular pattern's structure must be verified by observing a particular series of highs and lows between these points.

**Example:**

```
input color txt_clr = clrBlue; // Texts color
//X
double X; // Price of the swing low (X).
datetime X_time; // Time of the swing low (X).
string X_letter; // Unique name for the text label object.

int x_a_bars;
int x_lowest_index;
double x_a_ll;
datetime x_a_ll_t;

//A
double A; // Price of the swing high (A).
datetime A_time; // Time of the swing high (A).
string A_letter; // Unique name for the text label object.

int a_b_bars;
int a_highest_index;
double a_b_hh;
datetime a_b_hh_t;

string A_zone;
double A_low;

//B
double B; // Price of the swing low (B).
datetime B_time; // Time of the swing low (B).
string B_letter; // Unique name for the text label object.

int b_c_bars;
int b_lowest_index;
double b_c_ll;
datetime b_c_ll_t;

string B_zone;
double B_high;

//C
double C; // Price of the swing low (B).
datetime C_time; // Time of the swing low (B).
string C_letter; // Unique name for the text label object.

int c_d_bars;
int c_highest_index;
double c_d_hh;
datetime c_d_hh_t;

//D
double D; // Price of the swing low (B).
datetime D_time; // Time of the swing low (B).
string D_letter; // Unique name for the text label object.

int d_e_bars;
int d_lowest_index;
double d_e_ll;
datetime d_e_ll_t;

double D_3bar_high;

//E
double E; // Price of the swing low (B).
datetime E_time; // Time of the swing low (B).
string E_letter; // Unique name for the text label object.
double E_3bar_low;

string xa; // Unique name for the trendline for XA.
string ab; // Unique name for the trendline for AB.
string bc; // Unique name for the trendline for BC.
string cd; // Unique name for the trendline for CD.
string de; // Unique name for the trendline for DE.
string ex; // Unique name for the trendline for EX.

//FOR SELL
if(show_sell)
  {
   if(rates_total >= bars_check)
     {

      for(int z = 7; z <= 10; z++)
        {

         for(int i = rates_total - bars_check; i < rates_total - z; i++)
           {

            if(IsSwingLow(low, i, z))
              {

               // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
               X = low[i]; // Price of the swing low (X).
               X_time = time[i]; // Time of the swing low (X).
               X_letter = StringFormat("X%d", i); // Unique name for the text label object.

               for(int j = i; j < rates_total - z; j++)
                 {
                  if(IsSwingHigh(high, j, z) && time[j] > X_time)
                    {

                     A = high[j]; // Price of the swing high (A).
                     A_time = time[j]; // Time of the swing high (A)
                     A_letter = StringFormat("A%d", j); // Unique name for the text label object

                     for(int k = j; k < rates_total - z; k++)
                       {
                        if(IsSwingLow(low, k, z) && time[k] > A_time)
                          {

                           B = low[k]; // Price of the swing low (B).
                           B_time = time[k]; // Time of the swing low (B).
                           B_letter = StringFormat("B%d", k); // Unique name for the text label object.

                           for(int l = k ; l < rates_total - z; l++)
                             {

                              if(IsSwingHigh(high, l, z) && time[l] > B_time)
                                {

                                 C = high[l]; // Price of the swing high (C).
                                 C_time = time[l]; // Time of the swing high (C).
                                 C_letter = StringFormat("C%d", l); // Unique name for the text label object.

                                 for(int m = l; m < rates_total - z; m++)
                                   {

                                    if(IsSwingLow(low, m, z) && time[m] > C_time)
                                      {

                                       D = low[m]; // Price of the swing low (D).
                                       D_time = time[m]; // Time of the swing low (D).
                                       D_letter = StringFormat("D%d", m); // Unique name for the text label object.

                                       for(int n = m ; n < rates_total - (z/2) - 1; n++)
                                         {

                                          if(IsSwingHigh(high, n, (z/2)) && time[n] > D_time)
                                            {

                                             E = high[n]; // Price of the swing high (E).
                                             E_time = time[n]; // Time of the swing high (E).
                                             E_letter = StringFormat("E%d", n); // Unique name for the text label object.

                                             d_e_bars = Bars(_Symbol, PERIOD_CURRENT, D_time, E_time); // Count the number of bars between D and E
                                             d_lowest_index = ArrayMinimum(low, m, d_e_bars); // Find the index of the lowest low in the range
                                             d_e_ll = low[d_lowest_index]; // Store the lowest low (D - E lowest point)
                                             d_e_ll_t = time[d_lowest_index]; // Store the corresponding time
                                             D_3bar_high = high[d_lowest_index - 3]; // The high price of the third bar before the bar that formed D

                                             c_d_bars = Bars(_Symbol,PERIOD_CURRENT,C_time,d_e_ll_t); // Count the number of bars between C and V
                                             c_highest_index = ArrayMaximum(high,l,c_d_bars); // Find the index of the highest high in the range
                                             c_d_hh = high[c_highest_index];  // Store the lowest high (C - D lowest point)
                                             c_d_hh_t = time[c_highest_index]; // Store the corresponding time

                                             b_c_bars = Bars(_Symbol, PERIOD_CURRENT, B_time, c_d_hh_t); // Count the number of bars between B and C
                                             b_lowest_index = ArrayMinimum(low, k, b_c_bars); // Find the index of the lowest low in the range
                                             b_c_ll = low[b_lowest_index]; // Store the lowest low B - C lowest point)
                                             b_c_ll_t = time[b_lowest_index]; // Store the corresponding time
                                             B_high = high[b_lowest_index];  // The high price of the bar that formed swing low D

                                             a_b_bars = Bars(_Symbol,PERIOD_CURRENT,A_time,b_c_ll_t); // Count the number of bars between A and B
                                             a_highest_index = ArrayMaximum(high,j,a_b_bars); // Find the index of the highest high in the range
                                             a_b_hh = high[a_highest_index]; // Store the lowest low A - B lowest point)
                                             a_b_hh_t = time[a_highest_index];  // Store the corresponding time
                                             A_low = low[a_highest_index];

                                             x_a_bars = Bars(_Symbol, PERIOD_CURRENT, X_time, a_b_hh_t); // Count the number of bars between C and D
                                             x_lowest_index = ArrayMinimum(low, i, x_a_bars); // Find the index of the lowest low in the range
                                             x_a_ll = low[x_lowest_index]; // Store the lowest low (C - D lowest point)
                                             x_a_ll_t = time[x_lowest_index]; // Store the corresponding time for C - D

                                             E_3bar_low = low[n - 3]; // The LOW price of the third bar before the bar that formed E

                                             if(a_b_hh > x_a_ll && b_c_ll < a_b_hh && b_c_ll > x_a_ll && c_d_hh > a_b_hh && E < c_d_hh && d_e_ll > x_a_ll
                                                && d_e_ll <= B_high && D_3bar_high >= B_high && E > A_low && E_3bar_low < a_b_hh)
                                               {

                                                ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,X_time,X);
                                                ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
                                                ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,A_time,A);
                                                ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
                                                ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,B_time,B);
                                                ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
                                                ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,C_time,C);
                                                ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
                                                ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,D_time,D);
                                                ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
                                                ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,E_letter,OBJ_TEXT,0,E_time,E);
                                                ObjectSetString(chart_id,E_letter,OBJPROP_TEXT,"E");
                                                ObjectSetInteger(chart_id,E_letter,OBJPROP_COLOR,txt_clr);

                                                xa = StringFormat("XA line%d", i);
                                                ab = StringFormat("AB line%d", i);
                                                bc = StringFormat("BC line%d", i);
                                                cd = StringFormat("CD line%d", i);
                                                de = StringFormat("DE line%d", i);
                                                ex = StringFormat("EX line%d", i);

                                                ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,x_a_ll_t,x_a_ll);
                                                ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
                                                ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,a_b_hh_t,a_b_hh);
                                                ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
                                                ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,b_c_ll_t,b_c_ll);
                                                ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
                                                ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,c_d_hh_t,c_d_hh);
                                                ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
                                                ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,d_e_ll_t,d_e_ll);
                                                ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
                                                ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,E_letter,OBJ_TEXT,0,E_time,E);
                                                ObjectSetString(chart_id,E_letter,OBJPROP_TEXT,"E");
                                                ObjectSetInteger(chart_id,E_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id, xa,OBJ_TREND,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh);
                                                ObjectSetInteger(chart_id,xa,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,xa,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, xa, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, ab,OBJ_TREND,0,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll);
                                                ObjectSetInteger(chart_id,ab,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,ab,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, ab, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, bc,OBJ_TREND,0,b_c_ll_t,b_c_ll,c_d_hh_t,c_d_hh);
                                                ObjectSetInteger(chart_id,bc,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,bc,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, bc, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, cd,OBJ_TREND,0,c_d_hh_t,c_d_hh,d_e_ll_t,d_e_ll);
                                                ObjectSetInteger(chart_id,cd,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,cd,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, cd, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, de,OBJ_TREND,0,d_e_ll_t,d_e_ll,E_time,E);
                                                ObjectSetInteger(chart_id,de,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,de,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, de, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, ex,OBJ_TREND,0,E_time,E,time[n+(z/2)],x_a_ll);
                                                ObjectSetInteger(chart_id,ex,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,ex,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, ex, OBJPROP_BACK, true);

                                                A_zone = StringFormat("A ZONEe%d", i);
                                                B_zone = StringFormat("B ZONEe%d", i);

                                                ObjectCreate(chart_id,A_zone,OBJ_RECTANGLE,0,a_b_hh_t,a_b_hh,E_time,A_low);
                                                ObjectCreate(chart_id,B_zone,OBJ_RECTANGLE,0,b_c_ll_t,b_c_ll,d_e_ll_t,B_high);

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

     }
  }
```

**Output:**

![Figure 7. Head and Shoulder](https://c.mql5.com/2/143/Figure_7.png)

**Explanation:**

The code contains variable declarations that are used to store important information about the structure and components of a custom XABCDE pattern in price action. Each variable plays a specific role in identifying and visualizing the swing highs and lows that make up the pattern, as well as in labeling and drawing the structure on the chart. The first group of variables is related to the segment from point X to A. x\_a\_bars holds the number of bars between these two points. x\_lowest\_index stores the index of the lowest price bar in that segment, while x\_a\_ll keeps the actual lowest price (the “lower low”) and x\_a\_ll\_t saves the corresponding time this low occurred.

The stretch from A to B is handled by the following group. The number of bars in this section is a\_b\_bars. The index of the highest bar is identified by a\_highest\_index, while the actual highest price, or "higher high," is stored in a\_b\_hh, with a\_b\_hh\_t recording the time it happened. These support the idea that point B is a higher low and point A is a swing high. The variables b\_c\_bars, b\_lowest\_index, b\_c\_ll, and b\_c\_ll\_t then record the bar count, lowest index, lowest price, and the time of that price, respectively, for the segment from B to C. This serves to validate the structure's subsequent lower low.

String variables A\_zone and B\_zone are probably used to hold distinct object names for creating rectangular zones close to locations A and B. The price levels A\_low and B\_high, which serve as visual cues or decision-making areas on the chart, delineate the lower and upper limits of these zones. Similar to the previous variables, d\_e\_bars, d\_lowest\_index, d\_e\_ll, and d\_e\_ll\_t store bar count, index, price, and time data for the segment from D to E, aiding in locating the pattern's last leg.

D\_3bar\_high and E\_3bar\_low are used to store the highest high near point D and the lowest low near point E, typically calculated from 3-bar structures. These help verify the authenticity of swing points and reduce false positives. Lastly, the string IDs for trend lines that will be created between each of the pattern's turning points—from X to A, A to B, and so forth—are the variables xa, ab, bc, cd, de, and ex. These strings guarantee that every trendline object has a distinct name, enabling a precise and harmonious graphical depiction of the entire pattern on the chart.

Examining the price movement between the six crucial locations, which are denoted by the letters X, A, B, C, D, and E. It is hypothesized that these points correspond to swing highs and lows that create particular structural relationships. Finding the lowest low (d\_e\_ll) between points D and E is the first step in the script. It then obtains the time and a reference high (D\_3bar\_high) that is three bars before that low. Similarly, it finds the highest high (c\_d\_hh) between points C and this d\_e\_ll. It then collects all pertinent swing highs and lows along with their corresponding timestamps and repeats the process backward through time to point X. In this manner, the code uses bar analysis to build the entire X-A-B-C-D-E swing structure.

Between two timestamps, the number of candles is counted via the Bars() function. Using ArrayMaximum() and ArrayMinimum(), it aids in separating particular ranges between the swing points so that the code can examine highs and lows inside them. To locate the greatest high or the minimum low, these functions scan a specific number of bars from a given offset (i, j, k, etc.). This aids in locating swing points. ArrayMaximum(high, l, c\_d\_bars), for example, finds the highest high between C and D, yielding point C. Point X is reached by repeating the reasoning throughout the structure.

The structural relationship between the points is verified by evaluating a set of conditions. These criteria compare the highs and lows of the segments; for instance, it determines whether point A is higher than point X, whether point B is lower than A, but still higher than X, whether point C is higher than A, and so on. Before charting the price action, this confirms if it actually forms the anticipated pattern. E > A\_low and E\_3bar\_low < a\_b\_hh are examples of comparisons that guarantee the E-leg is in a legitimate position in relation to the remainder of the structure.

The method uses ObjectCreate() to graphically designate the recognized points on the chart with text labels ("X", "A", "B", etc.) and trend lines connecting them once the pattern satisfies all the requirements. These trend lines use OBJ\_TREND objects to draw the pattern's XA, AB, BC, CD, DE, and EX legs, and OBJ\_TEXT objects to denote the points. Readability is enhanced by colors, line width, and visual layering (via OBJPROP\_BACK).

Lastly, the code highlights the A-zone and B-zone, two crucial price zones, with rectangles. The A-zone extends from the high of point A to the low of point E, whereas the B-zone extends from the low of point B to the high of point D. When making trading decisions like entry, stops, or targets, these rectangles are probably utilized as a visual reference to identify reactions or confluence inside certain regions. Traders can more easily understand intricate patterns that the program automatically finds thanks to this display.

**3.2.2. Highlighting Pattern Structure with Triangle Shapes**

Drawing triangular shapes to highlight the pattern's structure, particularly when formations like Head and Shoulders emerge, comes next after we've located and labeled all the important swing points, which are X, A, B, C, D, and E. These triangles will indicate the peaks and troughs that characterize the "head" and "shoulders" of each pattern rather than following the full price legs. A Head and Shoulders pattern with the line not falling below the neckline is visually represented by the XAB triangle, which joins X, A, and B.

The BCD triangle, which joins B, C, and D and emphasizes the structure's second wave, is the next step in this method. The DEX triangle, which visually completes the shape, connects D, E, and X. By serving as a visual aid, these triangular shapes enable traders to identify crucial turning points and pattern geometry more rapidly without overcrowding the chart with lines.

**Example:**

```
input ENUM_TIMEFRAMES    timeframe = PERIOD_CURRENT; // MA Time Frame
input int bars_check  = 1000; // Number of bars to check for swing points
input bool show_sell = true; // Display sell signals
input bool show_buy = true; // Display buy signals
input color txt_clr = clrBlue; // Texts color
input color head_clr = clrCornflowerBlue; // Head color
input color shoulder_clr = clrLightSeaGreen; // Shoulder color
```

```
if(a_b_hh > x_a_ll && b_c_ll < a_b_hh && b_c_ll > x_a_ll && c_d_hh > a_b_hh && E < c_d_hh && d_e_ll > x_a_ll
   && d_e_ll <= B_high && D_3bar_high >= B_high && E > A_low && E_3bar_low < a_b_hh)
  {

   ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,X_time,X);
   ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
   ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,A_time,A);
   ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
   ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,B_time,B);
   ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
   ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,C_time,C);
   ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
   ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,D_time,D);
   ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
   ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,E_letter,OBJ_TEXT,0,E_time,E);
   ObjectSetString(chart_id,E_letter,OBJPROP_TEXT,"E");
   ObjectSetInteger(chart_id,E_letter,OBJPROP_COLOR,txt_clr);

   xa = StringFormat("XA line%d", i);
   ab = StringFormat("AB line%d", i);
   bc = StringFormat("BC line%d", i);
   cd = StringFormat("CD line%d", i);
   de = StringFormat("DE line%d", i);
   ex = StringFormat("EX line%d", i);

   ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,x_a_ll_t,x_a_ll);
   ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
   ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,a_b_hh_t,a_b_hh);
   ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
   ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,b_c_ll_t,b_c_ll);
   ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
   ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,c_d_hh_t,c_d_hh);
   ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
   ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,d_e_ll_t,d_e_ll);
   ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
   ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id,E_letter,OBJ_TEXT,0,E_time,E);
   ObjectSetString(chart_id,E_letter,OBJPROP_TEXT,"E");
   ObjectSetInteger(chart_id,E_letter,OBJPROP_COLOR,txt_clr);

   ObjectCreate(chart_id, xa,OBJ_TREND,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh);
   ObjectSetInteger(chart_id,xa,OBJPROP_WIDTH,3);
   ObjectSetInteger(chart_id,xa,OBJPROP_COLOR,clrSaddleBrown);
   ObjectSetInteger(chart_id, xa, OBJPROP_BACK, true);

   ObjectCreate(chart_id, ab,OBJ_TREND,0,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll);
   ObjectSetInteger(chart_id,ab,OBJPROP_WIDTH,3);
   ObjectSetInteger(chart_id,ab,OBJPROP_COLOR,clrSaddleBrown);
   ObjectSetInteger(chart_id, ab, OBJPROP_BACK, true);

   ObjectCreate(chart_id, bc,OBJ_TREND,0,b_c_ll_t,b_c_ll,c_d_hh_t,c_d_hh);
   ObjectSetInteger(chart_id,bc,OBJPROP_WIDTH,3);
   ObjectSetInteger(chart_id,bc,OBJPROP_COLOR,clrSaddleBrown);
   ObjectSetInteger(chart_id, bc, OBJPROP_BACK, true);

   ObjectCreate(chart_id, cd,OBJ_TREND,0,c_d_hh_t,c_d_hh,d_e_ll_t,d_e_ll);
   ObjectSetInteger(chart_id,cd,OBJPROP_WIDTH,3);
   ObjectSetInteger(chart_id,cd,OBJPROP_COLOR,clrSaddleBrown);
   ObjectSetInteger(chart_id, cd, OBJPROP_BACK, true);

   ObjectCreate(chart_id, de,OBJ_TREND,0,d_e_ll_t,d_e_ll,E_time,E);
   ObjectSetInteger(chart_id,de,OBJPROP_WIDTH,3);
   ObjectSetInteger(chart_id,de,OBJPROP_COLOR,clrSaddleBrown);
   ObjectSetInteger(chart_id, de, OBJPROP_BACK, true);

   ObjectCreate(chart_id, ex,OBJ_TREND,0,E_time,E,time[n+(z/2)],x_a_ll);
   ObjectSetInteger(chart_id,ex,OBJPROP_WIDTH,3);
   ObjectSetInteger(chart_id,ex,OBJPROP_COLOR,clrSaddleBrown);
   ObjectSetInteger(chart_id, ex, OBJPROP_BACK, true);

   A_zone = StringFormat("A ZONEe%d", i);
   B_zone = StringFormat("B ZONEe%d", i);

   ObjectCreate(chart_id,A_zone,OBJ_RECTANGLE,0,a_b_hh_t,a_b_hh,E_time,A_low);
   ObjectCreate(chart_id,B_zone,OBJ_RECTANGLE,0,b_c_ll_t,b_c_ll,d_e_ll_t,B_high);

   xa_line_t = ObjectGetTimeByValue(chart_id,xa,b_c_ll,0);
   ex_line_t = ObjectGetTimeByValue(chart_id,ex,d_e_ll,0);

   X_A_B = StringFormat("XAB %d", i);
   ObjectCreate(chart_id,X_A_B,OBJ_TRIANGLE,0,xa_line_t,b_c_ll,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll);
   ObjectSetInteger(chart_id, X_A_B, OBJPROP_FILL, true);
   ObjectSetInteger(chart_id, X_A_B, OBJPROP_BACK, true);
   ObjectSetInteger(chart_id, X_A_B, OBJPROP_COLOR, shoulder_clr);

   B_C_D = StringFormat("BCD %d", i);
   ObjectCreate(chart_id, B_C_D, OBJ_TRIANGLE, 0, b_c_ll_t, b_c_ll, c_d_hh_t, c_d_hh, d_e_ll_t, d_e_ll);
   ObjectSetInteger(chart_id, B_C_D, OBJPROP_COLOR, head_clr);
   ObjectSetInteger(chart_id, B_C_D, OBJPROP_FILL, true);
   ObjectSetInteger(chart_id, B_C_D, OBJPROP_BACK, true);

   D_E_X = StringFormat("DEX %d", i);
   ObjectCreate(chart_id, D_E_X, OBJ_TRIANGLE, 0, d_e_ll_t, d_e_ll, E_time, E, ex_line_t, d_e_ll);
   ObjectSetInteger(chart_id, D_E_X, OBJPROP_COLOR, shoulder_clr);
   ObjectSetInteger(chart_id, D_E_X, OBJPROP_FILL, true);
   ObjectSetInteger(chart_id, D_E_X, OBJPROP_BACK, true);

  }
```

**Output:**

![Figure 8. Highlighting Pattern Structure with Triangle Shapes](https://c.mql5.com/2/143/figure_8.png)

**Explanation:**

This section of code focuses on using triangles and rectangles to graphically depict pattern structures, notably XAB, BCD, and DEX, on the MetaTrader 5 chart. To keep dynamic names for the triangle objects that will be produced, three string variables—X\_A\_B, B\_C\_D, and D\_E\_X—are declared. Specific time coordinates are retrieved using two datetime variables (xa\_line\_t and ex\_line\_t) to anchor portions of the triangle shapes at the appropriate locations on the chart.

To improve the trader's understanding, the code starts by establishing two rectangular areas called "A ZONE" and "B ZONE." These rectangles, which highlight significant areas of the price structure, are created using ObjectCreate(). The A Zone extends to the E point, from the high between points A and B (a\_b\_hh) to the low at point A (A\_low). Similarly, the B Zone extends from the low position (b\_c\_ll) between points B and C to the highest point (B\_high), concluding at the time coordinate of the D to E leg.

The following section calls ObjectGetTimeByValue() to determine the time coordinates required to anchor portions of the XAB and DEX triangles. This function makes sure that the visual markers appropriately match the price levels they are intended to highlight by searching for a particular price value along an object's journey and returning the matching time.

Then, each triangle is depicted as a separate pattern leg. Although it does not link the full leg, the XAB triangle indicates the first swing structure. To prevent visual clutter and concentrate on structure, it only draws attention to the important turning points that resemble shoulders. Similar to this, the BCD triangle, which joins a low point (B), a peak (C), and a retracement (D), emphasizes the pattern's major head. The DEX triangle, which depicts the return swing that reflects the original shoulder, finally completes the framework.

Properties like OBJPROP\_FILL and OBJPROP\_BACK are used to fill all three triangles with color and place them behind other chart elements. Traders can more easily recognize the pattern at a glance because of the colors employed, which are kept in variables like shoulder\_clr and head\_clr. To handle and recognize many instances of these patterns on the same chart, the index i is included in the naming of each item.

**3.2.3. Indicating Entry Point, Stop Loss, and Take Profit**

The following stage is to specify the trade parameters, which include the entry point, stop loss (SL), and take profit (TP), after the XAB, BCD, and DEX structures have been marked out with triangles, emphasizing the head and shoulders shape. These components are necessary to transform the recognized pattern into a comprehensive trading strategy. When a candlestick closes below point D, the entry is made. This demonstrates that the price may have rejected the neckline region and is moving in the pattern's predicted direction. The trade is now seen as legitimate, and execution is possible.

Just above point E, the stop loss is set. Placing the SL here helps safeguard the trade if the pattern fails, and the price unexpectedly reverses because E is the most recent swing high before the pattern completes. Additionally, it keeps the setup reasonable by only invalidating the trade when the pattern is violated. Since point X, the pattern's genesis, is a reliable reference level where the price has previously reversed, the take profit is originally put there. The TP is stretched further to reach a minimum of 1:2, though, if the distance from entry to X does not yield a Risk-Reward Ratio (RRR) of at least 1:1. A key element of any sustainable trading strategy is ensuring that the trade has a favorable return potential in relation to the amount risked.

**Example:**

```
int n_bars;
int n_bars_2;

string sl_t;
string tp_t;

double sl_price;
double tp_price;
```

```
for(int o = n; o < rates_total - 1; o++)
  {

   if(close[o] < d_e_ll && time[o] >= time[n+(z/2)])
     {

      n_bars = Bars(_Symbol,PERIOD_CURRENT,x_a_ll_t, E_time);
      n_bars_2 = Bars(_Symbol,PERIOD_CURRENT,time[n+(z/2)], time[o]);
      if(n_bars_2 <= n_bars)
        {

         double sl_zone = MathAbs(E - close[o]);
         double tp_zone = MathAbs(close[o] - x_a_ll);

         bool no_cross = false;

         for(int p = n + (z/2); p < o; p++)
           {

            if(close[p] < d_e_ll)
              {

               no_cross = true;

               break;

              }

           }

         if(no_cross == false)
           {
            if(tp_zone >= sl_zone)
              {

               string loss_zone = StringFormat("Loss %d", i);
               ObjectCreate(chart_id,loss_zone,OBJ_RECTANGLE,0,E_time,E,time[o],close[o]);
               ObjectSetInteger(chart_id, loss_zone, OBJPROP_FILL, true);
               ObjectSetInteger(chart_id, loss_zone, OBJPROP_BACK, true);
               ObjectSetInteger(chart_id, loss_zone, OBJPROP_COLOR, lz_clr);

               string   sell_object = StringFormat("Sell Object%d", i);
               ObjectCreate(chart_id,sell_object,OBJ_ARROW_SELL,0,time[o],close[o]);

               string win_zone = StringFormat("Win %d", i);
               ObjectCreate(chart_id,win_zone,OBJ_RECTANGLE,0,E_time,close[o],time[o],x_a_ll);
               ObjectSetInteger(chart_id, win_zone, OBJPROP_FILL, true);
               ObjectSetInteger(chart_id, win_zone, OBJPROP_BACK, true);
               ObjectSetInteger(chart_id, win_zone, OBJPROP_COLOR, wz_clr);

               sl_price =  E;
               tp_price =  x_a_ll;

               string sl_d_s = DoubleToString(sl_price,_Digits);
               string tp_d_s = DoubleToString(tp_price,_Digits);

               sl_t = StringFormat("sl %d", i);
               tp_t = StringFormat("tp %d", i);

               ObjectCreate(chart_id,sl_t,OBJ_TEXT,0,time[o],sl_price);
               ObjectSetString(chart_id,sl_t,OBJPROP_TEXT,"SL - " + sl_d_s);
               ObjectSetInteger(chart_id,sl_t,OBJPROP_FONTSIZE,8);
               ObjectSetInteger(chart_id,sl_t,OBJPROP_COLOR,txt_clr);

               ObjectCreate(chart_id,tp_t,OBJ_TEXT,0,time[o],x_a_ll);
               ObjectSetString(chart_id,tp_t,OBJPROP_TEXT,"TP - " + tp_d_s);
               ObjectSetInteger(chart_id,tp_t,OBJPROP_FONTSIZE,8);
               ObjectSetInteger(chart_id,tp_t,OBJPROP_COLOR,txt_clr);

              }

            if(tp_zone < sl_zone)
              {
               string loss_zone = StringFormat("Loss %d", i);
               ObjectCreate(chart_id,loss_zone,OBJ_RECTANGLE,0,E_time,E,time[o],close[o]);
               ObjectSetInteger(chart_id, loss_zone, OBJPROP_FILL, true);
               ObjectSetInteger(chart_id, loss_zone, OBJPROP_BACK, true);
               ObjectSetInteger(chart_id, loss_zone, OBJPROP_COLOR, lz_clr);

               string   sell_object = StringFormat("Sell Object%d", i);
               ObjectCreate(chart_id,sell_object,OBJ_ARROW_SELL,0,time[o],close[o]);

               double n_tp = MathAbs(close[o] - (sl_zone * 2));

               string win_zone = StringFormat("Win %d", i);
               ObjectCreate(chart_id,win_zone,OBJ_RECTANGLE,0,E_time,close[o],time[o],n_tp);
               ObjectSetInteger(chart_id, win_zone, OBJPROP_FILL, true);
               ObjectSetInteger(chart_id, win_zone, OBJPROP_BACK, true);
               ObjectSetInteger(chart_id, win_zone, OBJPROP_COLOR, wz_clr);

               sl_price =  E;
               tp_price =  n_tp;

               string sl_d_s = DoubleToString(sl_price,_Digits);
               string tp_d_s = DoubleToString(tp_price,_Digits);

               sl_t = StringFormat("sl %d", i);
               tp_t = StringFormat("tp %d", i);

               ObjectCreate(chart_id,sl_t,OBJ_TEXT,0,time[o],sl_price);;
               ObjectSetString(chart_id,sl_t,OBJPROP_TEXT,"SL - " + sl_d_s);
               ObjectSetInteger(chart_id,sl_t,OBJPROP_FONTSIZE,8);
               ObjectSetInteger(chart_id,sl_t,OBJPROP_COLOR,txt_clr);

               ObjectCreate(chart_id,tp_t,OBJ_TEXT,0,time[o],tp_price);
               ObjectSetString(chart_id,tp_t,OBJPROP_TEXT,"TP - " + tp_d_s);
               ObjectSetInteger(chart_id,tp_t,OBJPROP_FONTSIZE,8);
               ObjectSetInteger(chart_id,tp_t,OBJPROP_COLOR,txt_clr);

              }


           }

        }

      break;

     }
  }
```

**Output:**

![Figure 9. Indicating Entry Point, Stop Loss, and Take Profit](https://c.mql5.com/2/143/figure_9.png)

**Explanation:**

The logic in this area of the code is centered on determining a legitimate trade setting using the head and shoulders structure that was previously highlighted. Graphical objects are then utilized to indicate the entry position graphically, stop loss (SL), and take profit (TP) on the chart. The number of candlesticks between significant points is calculated and compared using two integer variables, n\_bars and n\_bars\_2. This aids in figuring out whether the current price action is still inside the setup's acceptable window. While sl\_price and tp\_price store their respective price levels, the sl\_t and tp\_t strings are used to dynamically name the SL and TP text objects.

From the pattern's completion point (n) onward, the main logic starts in a loop that goes through the bars. A candle that closes below the neckline level (point D), indicating a possible entrance, is looked for by the condition if (close\[o\] < d\_e\_ll && time\[o\] >= time\[n+(z/2)\]). The number of bars between points X and E (n\_bars) and between the pattern's midpoint and the current candle (n\_bars\_2) is then determined if this requirement is satisfied. By doing this, the trade is guaranteed to be valid within a reasonable interval.

The code then compares the entry price to points E and X, respectively, to determine the amount of the possible stop loss (sl\_zone) and the take profit (tp\_zone). To rule out this entry as the initial breakout, it additionally performs a loop to make sure that no prior candle had closed below point D before the present one. The script continues if this check is successful.

To visually depict the stop loss and take profit zones on the chart, the code uses OBJ\_RECTANGLE to produce graphical rectangles if the potential reward (TP) is greater than or equal to the risk (SL). Additionally, there is a sale arrow (OBJ\_ARROW\_SELL) near the entrance point. The sl\_price and tp\_price variables contain the actual SL and TP price values. Labels are made with OBJ\_TEXT to prominently display these levels on the chart using color and font choices. To provide a more favorable RRR of at least 1:2, the script adjusts by computing a new TP zone that is three times the SL distance if the risk-reward ratio is less than 1:1 (i.e., TP is less than SL). Then, for this alternate configuration, the identical graphical objects are made and modified.

### **4\. Executing Trades Based on the Pattern**

Our goal in this part is to do more than just recognize and show trade situations. This section is all about actually carrying out the trade based on the identified pattern, whereas the previous portion concentrated on marking the entry, stop loss (SL), and take profit (TP) levels on the chart. The EA ought to initiate a trade automatically when the requirements are satisfied, such as when the price closes below point D for a sell arrangement. The take-profit should be set at point X or modified for a risk-to-reward ratio of at least 1:2, and the stop loss should be set at point E. By taking this step, the EA is converted from a visual tool into a completely automated system that can enter and manage trades without the need for human participation.

**Example:**

```
#include <Trade/Trade.mqh>
CTrade trade;
int MagicNumber = 5122025;
datetime lastTradeBarTime = 0;
double ask_price;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   trade.SetExpertMagicNumber(MagicNumber);

//---
   return(INIT_SUCCEEDED);
  }
```

```
ask_price = ask_price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
datetime currentBarTime = iTime(_Symbol, timeframe, 0);

//FOR SELL
if(show_sell)
  {
   if(rates_total >= bars_check)
     {

      for(int z = 7; z <= 10; z++)
        {

         for(int i = rates_total - bars_check; i < rates_total - z; i++)
           {

            if(IsSwingLow(low, i, z))
              {

               // If a swing low is found, store its price, time, and create a name for the objects to mark the X.
               X = low[i]; // Price of the swing low (X).
               X_time = time[i]; // Time of the swing low (X).
               X_letter = StringFormat("X%d", i); // Unique name for the text label object.

               for(int j = i; j < rates_total - z; j++)
                 {
                  if(IsSwingHigh(high, j, z) && time[j] > X_time)
                    {

                     A = high[j]; // Price of the swing high (A).
                     A_time = time[j]; // Time of the swing high (A)
                     A_letter = StringFormat("A%d", j); // Unique name for the text label object

                     for(int k = j; k < rates_total - z; k++)
                       {
                        if(IsSwingLow(low, k, z) && time[k] > A_time)
                          {

                           B = low[k]; // Price of the swing low (B).
                           B_time = time[k]; // Time of the swing low (B).
                           B_letter = StringFormat("B%d", k); // Unique name for the text label object.

                           for(int l = k ; l < rates_total - z; l++)
                             {

                              if(IsSwingHigh(high, l, z) && time[l] > B_time)
                                {

                                 C = high[l]; // Price of the swing high (C).
                                 C_time = time[l]; // Time of the swing high (C).
                                 C_letter = StringFormat("C%d", l); // Unique name for the text label object.

                                 for(int m = l; m < rates_total - z; m++)
                                   {

                                    if(IsSwingLow(low, m, z) && time[m] > C_time)
                                      {

                                       D = low[m]; // Price of the swing low (D).
                                       D_time = time[m]; // Time of the swing low (D).
                                       D_letter = StringFormat("D%d", m); // Unique name for the text label object.

                                       for(int n = m ; n < rates_total - (z/2) - 1; n++)
                                         {

                                          if(IsSwingHigh(high, n, (z/2)) && time[n] > D_time)
                                            {

                                             E = high[n]; // Price of the swing high (E).
                                             E_time = time[n]; // Time of the swing high (E).
                                             E_letter = StringFormat("E%d", n); // Unique name for the text label object.

                                             d_e_bars = Bars(_Symbol, PERIOD_CURRENT, D_time, E_time); // Count the number of bars between D and E
                                             d_lowest_index = ArrayMinimum(low, m, d_e_bars); // Find the index of the lowest low in the range
                                             d_e_ll = low[d_lowest_index]; // Store the lowest low (D - E lowest point)
                                             d_e_ll_t = time[d_lowest_index]; // Store the corresponding time
                                             D_3bar_high = high[d_lowest_index - 3]; // The high price of the third bar before the bar that formed D

                                             c_d_bars = Bars(_Symbol,PERIOD_CURRENT,C_time,d_e_ll_t); // Count the number of bars between C and V
                                             c_highest_index = ArrayMaximum(high,l,c_d_bars); // Find the index of the highest high in the range
                                             c_d_hh = high[c_highest_index];  // Store the lowest high (C - D lowest point)
                                             c_d_hh_t = time[c_highest_index]; // Store the corresponding time

                                             b_c_bars = Bars(_Symbol, PERIOD_CURRENT, B_time, c_d_hh_t); // Count the number of bars between B and C
                                             b_lowest_index = ArrayMinimum(low, k, b_c_bars); // Find the index of the lowest low in the range
                                             b_c_ll = low[b_lowest_index]; // Store the lowest low B - C lowest point)
                                             b_c_ll_t = time[b_lowest_index]; // Store the corresponding time
                                             B_high = high[b_lowest_index];  // The high price of the bar that formed swing low D

                                             a_b_bars = Bars(_Symbol,PERIOD_CURRENT,A_time,b_c_ll_t); // Count the number of bars between A and B
                                             a_highest_index = ArrayMaximum(high,j,a_b_bars); // Find the index of the highest high in the range
                                             a_b_hh = high[a_highest_index]; // Store the lowest low A - B lowest point)
                                             a_b_hh_t = time[a_highest_index];  // Store the corresponding time
                                             A_low = low[a_highest_index];

                                             x_a_bars = Bars(_Symbol, PERIOD_CURRENT, X_time, a_b_hh_t); // Count the number of bars between C and D
                                             x_lowest_index = ArrayMinimum(low, i, x_a_bars); // Find the index of the lowest low in the range
                                             x_a_ll = low[x_lowest_index]; // Store the lowest low (C - D lowest point)
                                             x_a_ll_t = time[x_lowest_index]; // Store the corresponding time for C - D

                                             E_3bar_low = low[n - 3]; // The LOW price of the third bar before the bar that formed E

                                             if(a_b_hh > x_a_ll && b_c_ll < a_b_hh && b_c_ll > x_a_ll && c_d_hh > a_b_hh && E < c_d_hh && d_e_ll > x_a_ll
                                                && d_e_ll <= B_high && D_3bar_high >= B_high && E > A_low && E_3bar_low < a_b_hh)
                                               {

                                                ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,X_time,X);
                                                ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
                                                ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,A_time,A);
                                                ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
                                                ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,B_time,B);
                                                ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
                                                ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,C_time,C);
                                                ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
                                                ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,D_time,D);
                                                ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
                                                ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,E_letter,OBJ_TEXT,0,E_time,E);
                                                ObjectSetString(chart_id,E_letter,OBJPROP_TEXT,"E");
                                                ObjectSetInteger(chart_id,E_letter,OBJPROP_COLOR,txt_clr);

                                                xa = StringFormat("XA line%d", i);
                                                ab = StringFormat("AB line%d", i);
                                                bc = StringFormat("BC line%d", i);
                                                cd = StringFormat("CD line%d", i);
                                                de = StringFormat("DE line%d", i);
                                                ex = StringFormat("EX line%d", i);

                                                ObjectCreate(chart_id,X_letter,OBJ_TEXT,0,x_a_ll_t,x_a_ll);
                                                ObjectSetString(chart_id,X_letter,OBJPROP_TEXT,"X");
                                                ObjectSetInteger(chart_id,X_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,A_letter,OBJ_TEXT,0,a_b_hh_t,a_b_hh);
                                                ObjectSetString(chart_id,A_letter,OBJPROP_TEXT,"A");
                                                ObjectSetInteger(chart_id,A_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,B_letter,OBJ_TEXT,0,b_c_ll_t,b_c_ll);
                                                ObjectSetString(chart_id,B_letter,OBJPROP_TEXT,"B");
                                                ObjectSetInteger(chart_id,B_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,C_letter,OBJ_TEXT,0,c_d_hh_t,c_d_hh);
                                                ObjectSetString(chart_id,C_letter,OBJPROP_TEXT,"C");
                                                ObjectSetInteger(chart_id,C_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,D_letter,OBJ_TEXT,0,d_e_ll_t,d_e_ll);
                                                ObjectSetString(chart_id,D_letter,OBJPROP_TEXT,"D");
                                                ObjectSetInteger(chart_id,D_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id,E_letter,OBJ_TEXT,0,E_time,E);
                                                ObjectSetString(chart_id,E_letter,OBJPROP_TEXT,"E");
                                                ObjectSetInteger(chart_id,E_letter,OBJPROP_COLOR,txt_clr);

                                                ObjectCreate(chart_id, xa,OBJ_TREND,0,x_a_ll_t,x_a_ll,a_b_hh_t,a_b_hh);
                                                ObjectSetInteger(chart_id,xa,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,xa,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, xa, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, ab,OBJ_TREND,0,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll);
                                                ObjectSetInteger(chart_id,ab,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,ab,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, ab, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, bc,OBJ_TREND,0,b_c_ll_t,b_c_ll,c_d_hh_t,c_d_hh);
                                                ObjectSetInteger(chart_id,bc,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,bc,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, bc, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, cd,OBJ_TREND,0,c_d_hh_t,c_d_hh,d_e_ll_t,d_e_ll);
                                                ObjectSetInteger(chart_id,cd,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,cd,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, cd, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, de,OBJ_TREND,0,d_e_ll_t,d_e_ll,E_time,E);
                                                ObjectSetInteger(chart_id,de,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,de,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, de, OBJPROP_BACK, true);

                                                ObjectCreate(chart_id, ex,OBJ_TREND,0,E_time,E,time[n+(z/2)],x_a_ll);
                                                ObjectSetInteger(chart_id,ex,OBJPROP_WIDTH,3);
                                                ObjectSetInteger(chart_id,ex,OBJPROP_COLOR,clrSaddleBrown);
                                                ObjectSetInteger(chart_id, ex, OBJPROP_BACK, true);

                                                A_zone = StringFormat("A ZONEe%d", i);
                                                B_zone = StringFormat("B ZONEe%d", i);

                                                ObjectCreate(chart_id,A_zone,OBJ_RECTANGLE,0,a_b_hh_t,a_b_hh,E_time,A_low);
                                                ObjectCreate(chart_id,B_zone,OBJ_RECTANGLE,0,b_c_ll_t,b_c_ll,d_e_ll_t,B_high);

                                                xa_line_t = ObjectGetTimeByValue(chart_id,xa,b_c_ll,0);
                                                ex_line_t = ObjectGetTimeByValue(chart_id,ex,d_e_ll,0);

                                                X_A_B = StringFormat("XAB %d", i);
                                                ObjectCreate(chart_id,X_A_B,OBJ_TRIANGLE,0,xa_line_t,b_c_ll,a_b_hh_t,a_b_hh,b_c_ll_t,b_c_ll);
                                                ObjectSetInteger(chart_id, X_A_B, OBJPROP_FILL, true);
                                                ObjectSetInteger(chart_id, X_A_B, OBJPROP_BACK, true);
                                                ObjectSetInteger(chart_id, X_A_B, OBJPROP_COLOR, shoulder_clr);

                                                B_C_D = StringFormat("BCD %d", i);
                                                ObjectCreate(chart_id, B_C_D, OBJ_TRIANGLE, 0, b_c_ll_t, b_c_ll, c_d_hh_t, c_d_hh, d_e_ll_t, d_e_ll);
                                                ObjectSetInteger(chart_id, B_C_D, OBJPROP_COLOR, head_clr);
                                                ObjectSetInteger(chart_id, B_C_D, OBJPROP_FILL, true);
                                                ObjectSetInteger(chart_id, B_C_D, OBJPROP_BACK, true);

                                                D_E_X = StringFormat("DEX %d", i);
                                                ObjectCreate(chart_id, D_E_X, OBJ_TRIANGLE, 0, d_e_ll_t, d_e_ll, E_time, E, ex_line_t, d_e_ll);
                                                ObjectSetInteger(chart_id, D_E_X, OBJPROP_COLOR, shoulder_clr);
                                                ObjectSetInteger(chart_id, D_E_X, OBJPROP_FILL, true);
                                                ObjectSetInteger(chart_id, D_E_X, OBJPROP_BACK, true);

                                                for(int o = n; o < rates_total - 1; o++)
                                                  {

                                                   if(close[o] < d_e_ll && time[o] >= time[n+(z/2)])
                                                     {

                                                      n_bars = Bars(_Symbol,PERIOD_CURRENT,x_a_ll_t, E_time);
                                                      n_bars_2 = Bars(_Symbol,PERIOD_CURRENT,time[n+(z/2)], time[o]);
                                                      if(n_bars_2 <= n_bars)
                                                        {

                                                         double sl_zone = MathAbs(E - close[o]);
                                                         double tp_zone = MathAbs(close[o] - x_a_ll);

                                                         bool no_cross = false;

                                                         for(int p = n + (z/2); p < o; p++)
                                                           {

                                                            if(close[p] < d_e_ll)
                                                              {

                                                               no_cross = true;

                                                               break;

                                                              }

                                                           }

                                                         if(no_cross == false)
                                                           {
                                                            if(tp_zone >= sl_zone)
                                                              {

                                                               string loss_zone = StringFormat("Loss %d", i);
                                                               ObjectCreate(chart_id,loss_zone,OBJ_RECTANGLE,0,E_time,E,time[o],close[o]);
                                                               ObjectSetInteger(chart_id, loss_zone, OBJPROP_FILL, true);
                                                               ObjectSetInteger(chart_id, loss_zone, OBJPROP_BACK, true);
                                                               ObjectSetInteger(chart_id, loss_zone, OBJPROP_COLOR, lz_clr);

                                                               string   sell_object = StringFormat("Sell Object%d", i);
                                                               ObjectCreate(chart_id,sell_object,OBJ_ARROW_SELL,0,time[o],close[o]);

                                                               string win_zone = StringFormat("Win %d", i);
                                                               ObjectCreate(chart_id,win_zone,OBJ_RECTANGLE,0,E_time,close[o],time[o],x_a_ll);
                                                               ObjectSetInteger(chart_id, win_zone, OBJPROP_FILL, true);
                                                               ObjectSetInteger(chart_id, win_zone, OBJPROP_BACK, true);
                                                               ObjectSetInteger(chart_id, win_zone, OBJPROP_COLOR, wz_clr);

                                                               sl_price =  E;
                                                               tp_price =  x_a_ll;

                                                               string sl_d_s = DoubleToString(sl_price,_Digits);
                                                               string tp_d_s = DoubleToString(tp_price,_Digits);

                                                               sl_t = StringFormat("sl %d", i);
                                                               tp_t = StringFormat("tp %d", i);

                                                               ObjectCreate(chart_id,sl_t,OBJ_TEXT,0,time[o],sl_price);
                                                               ObjectSetString(chart_id,sl_t,OBJPROP_TEXT,"SL - " + sl_d_s);
                                                               ObjectSetInteger(chart_id,sl_t,OBJPROP_FONTSIZE,8);
                                                               ObjectSetInteger(chart_id,sl_t,OBJPROP_COLOR,txt_clr);

                                                               ObjectCreate(chart_id,tp_t,OBJ_TEXT,0,time[o],x_a_ll);
                                                               ObjectSetString(chart_id,tp_t,OBJPROP_TEXT,"TP - " + tp_d_s);
                                                               ObjectSetInteger(chart_id,tp_t,OBJPROP_FONTSIZE,8);
                                                               ObjectSetInteger(chart_id,tp_t,OBJPROP_COLOR,txt_clr);

                                                              }

                                                            if(tp_zone < sl_zone)
                                                              {
                                                               string loss_zone = StringFormat("Loss %d", i);
                                                               ObjectCreate(chart_id,loss_zone,OBJ_RECTANGLE,0,E_time,E,time[o],close[o]);
                                                               ObjectSetInteger(chart_id, loss_zone, OBJPROP_FILL, true);
                                                               ObjectSetInteger(chart_id, loss_zone, OBJPROP_BACK, true);
                                                               ObjectSetInteger(chart_id, loss_zone, OBJPROP_COLOR, lz_clr);

                                                               string   sell_object = StringFormat("Sell Object%d", i);
                                                               ObjectCreate(chart_id,sell_object,OBJ_ARROW_SELL,0,time[o],close[o]);

                                                               double n_tp = MathAbs(close[o] - (sl_zone * 3));

                                                               string win_zone = StringFormat("Win %d", i);
                                                               ObjectCreate(chart_id,win_zone,OBJ_RECTANGLE,0,E_time,close[o],time[o],n_tp);
                                                               ObjectSetInteger(chart_id, win_zone, OBJPROP_FILL, true);
                                                               ObjectSetInteger(chart_id, win_zone, OBJPROP_BACK, true);
                                                               ObjectSetInteger(chart_id, win_zone, OBJPROP_COLOR, wz_clr);

                                                               sl_price =  E;
                                                               tp_price =  n_tp;

                                                               string sl_d_s = DoubleToString(sl_price,_Digits);
                                                               string tp_d_s = DoubleToString(tp_price,_Digits);

                                                               sl_t = StringFormat("sl %d", i);
                                                               tp_t = StringFormat("tp %d", i);

                                                               ObjectCreate(chart_id,sl_t,OBJ_TEXT,0,time[o],sl_price);;
                                                               ObjectSetString(chart_id,sl_t,OBJPROP_TEXT,"SL - " + sl_d_s);
                                                               ObjectSetInteger(chart_id,sl_t,OBJPROP_FONTSIZE,8);
                                                               ObjectSetInteger(chart_id,sl_t,OBJPROP_COLOR,txt_clr);

                                                               ObjectCreate(chart_id,tp_t,OBJ_TEXT,0,time[o],tp_price);
                                                               ObjectSetString(chart_id,tp_t,OBJPROP_TEXT,"TP - " + tp_d_s);
                                                               ObjectSetInteger(chart_id,tp_t,OBJPROP_FONTSIZE,8);
                                                               ObjectSetInteger(chart_id,tp_t,OBJPROP_COLOR,txt_clr);

                                                              }

                                                            if(time[o] == time[rates_total-2] && currentBarTime != lastTradeBarTime)
                                                              {

                                                               trade.Sell(lot_size,_Symbol,ask_price, sl_price,tp_price);
                                                               lastTradeBarTime = currentBarTime;

                                                              }

                                                           }

                                                        }

                                                      break;

                                                     }
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

                     break;
                    }
                 }
              }
           }
        }
     }
  }
```

**Output:**

![Figure 10. Executing Trades Based on the Pattern](https://c.mql5.com/2/143/figure_10.png)

**Explanation:**

This block of code manages trade execution using MQL5's built-in CTrade class, which streamlines trading tasks including placing, editing, and closing orders. The trading library, which provides access to the CTrade class, is included at the top using #include <Trade/Trade.mqh>. Next, an instance of this class is created using CTrade trade;, and int MagicNumber = 5122025; is used to define a unique identification for trades done by this Expert Advisor (EA). Because it aids in differentiating trades made by this EA from those made manually or by other EAs, the MagicNumber is crucial. Trade is used to set it's MagicNumber; SetExpertMagicNumber;.

SymbolInfoDouble(\_Symbol, SYMBOL\_ASK); is used to initialize the variable ask\_price with the symbol's current ask price. An order to sell would be executed at this price. The code then uses time\[o\] == time\[rates\_total-2\] to determine if the current bar is the most recent one. Additionally, the condition currentBarTime!= lastTradeBarTime is used to confirm that no trade has already been placed on the current bar. The EA can't place more than one order atthe same bar thanks to this check.

The EA uses the transaction to issue a sell order if both requirements are met. Sell() function. Ask\_price is the entrance price, sl\_price is the stop loss level, tp\_price is the take profit level, lot\_size is the number of lots to be traded, and \_Symbol is the current trading symbol (such as EURUSD). The EA adds the value of currentBarTime to the lastTradeBarTime variable after the transaction is successfully placed. This update is essential because it makes sure that only one trade is made per legitimate pattern indication by stopping the EA from making numerous transactions based on the same bar.

It's crucial to remember that the same reasoning that is used to identify and display a sell setup can equally be utilized to identify and show a purchase setup by just performing the reverse at each stage. The purchase strategy would wait for a candle to close above the neckline (point D) rather than below it. After that, the take-profit would be positioned at the highest point (X), and the stop loss at the lowest point (E). The risk-reward reasoning, visual cues, and conditions are all the same; they are simply mirrored for bullish structure. The source code that will be included with the post will completely implement this logic for anyone who wants to further examine or alter it.

### **Conclusion**

In this article, we explored how to build an Expert Advisor (EA) in MQL5 that identifies and trades based on technical chart patterns — in particular, the Head and Shoulders pattern. We started by detecting and marking important swing points, then used graphical tools like triangles and rectangles to visually represent the XAB, BCD, and DEX structures. We followed this by defining clear trade parameters, including the entry point, stop loss, and take profit levels, while also ensuring the risk-to-reward ratio met logical standards. Finally, we implemented trade execution logic that places real orders based on confirmed signals, making the EA fully automated.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18147.zip "Download all attachments in the single ZIP archive")

[Project\_10\_Head\_and\_Shoulder\_EA.mq5](https://www.mql5.com/en/articles/download/18147/project_10_head_and_shoulder_ea.mq5 "Download Project_10_Head_and_Shoulder_EA.mq5")(54.57 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/487140)**
(6)


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
22 May 2025 at 17:46

**Oluwatosin Mary Babalola [#](https://www.mql5.com/en/forum/487140#comment_56764290):**

Thank you, this is why I always anticipate your articles. Very explanatory!

You’re welcome


![Miguel Angel Vico Alba](https://c.mql5.com/avatar/2025/10/68e99f33-714e.jpg)

**[Miguel Angel Vico Alba](https://www.mql5.com/en/users/mike_explosion)**
\|
22 May 2025 at 22:31

**Israel Pelumi Abioye [#](https://www.mql5.com/en/forum/487140#comment_56764514):** You’re welcome

You're amazing, Israel! I regularly translate your articles into Spanish officially, and it's always a real pleasure to work on them.

Keep up the fantastic work! ❤️

![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
22 May 2025 at 22:35

**Miguel Angel Vico Alba [#](https://www.mql5.com/en/forum/487140#comment_56766108):**

You're amazing, Israel! I regularly translate your articles into Spanish officially, and it's always a real pleasure to work on them.

Keep up the fantastic work! ❤️

Hello, Miguel.

Thank you for your kind words, it means a lot to me ❤️

![daniels goodness](https://c.mql5.com/avatar/2023/10/652D2F65-2B81.jpg)

**[daniels goodness](https://www.mql5.com/en/users/danielsgoodness)**
\|
26 May 2025 at 14:00

I am not a programmer so understanding this took a while but the way it was broken down and the step by step process has given me motivation. 💯


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
26 May 2025 at 17:24

**daniels goodness [#](https://www.mql5.com/en/forum/487140#comment_56788792):**

I am not a programmer so understanding this took a while but the way it was broken down and the step by step process has given me motivation. 💯

Hello Daniels.

Good to hear that from you! ❤️

![Price Action Analysis Toolkit Development (Part 24): Price Action Quantification Analysis Tool](https://c.mql5.com/2/144/18207-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 24): Price Action Quantification Analysis Tool](https://www.mql5.com/en/articles/18207)

Candlestick patterns offer valuable insights into potential market moves. Some single candles signal continuation of the current trend, while others foreshadow reversals, depending on their position within the price action. This article introduces an EA that automatically identifies four key candlestick formations. Explore the following sections to learn how this tool can enhance your price-action analysis.

![Neural Networks in Trading: Controlled Segmentation (Final Part)](https://c.mql5.com/2/96/Neural_Networks_in_Trading_Controlled_Segmentation___LOGO__1.png)[Neural Networks in Trading: Controlled Segmentation (Final Part)](https://www.mql5.com/en/articles/16057)

We continue the work started in the previous article on building the RefMask3D framework using MQL5. This framework is designed to comprehensively study multimodal interaction and feature analysis in a point cloud, followed by target object identification based on a description provided in natural language.

![From Basic to Intermediate: Array (I)](https://c.mql5.com/2/97/Do_bzsico_ao_intermedikrio__Array_I___LOGO.png)[From Basic to Intermediate: Array (I)](https://www.mql5.com/en/articles/15462)

This article is a transition between what has been discussed so far and a new stage of research. To understand this article, you need to read the previous ones. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Developing a Replay System (Part 69): Getting the Time Right (II)](https://c.mql5.com/2/97/Desenvolvendo_um_sistema_de_Replay_Parte_69___LOGO.png)[Developing a Replay System (Part 69): Getting the Time Right (II)](https://www.mql5.com/en/articles/12317)

Today we will look at why we need the iSpread feature. At the same time, we will understand how the system informs us about the remaining time of the bar when there is not a single tick available for it. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qrhotfeorvmilnlozqlyeeibjscgmkdv&ssn=1769091530068841266&ssn_dr=0&ssn_sr=0&fv_date=1769091530&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18147&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%2016)%3A%20Building%20Expert%20Advisors%20Using%20Technical%20Chart%20Patterns%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909153063665670&fz_uniq=5049038539913536457&sv=2552)

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