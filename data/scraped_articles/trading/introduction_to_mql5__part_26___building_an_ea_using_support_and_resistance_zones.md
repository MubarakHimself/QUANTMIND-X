---
title: Introduction to MQL5 (Part 26): Building an EA Using Support and Resistance Zones
url: https://www.mql5.com/en/articles/20021
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 8
scraped_at: 2026-01-22T17:44:01.162379
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/20021&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049338706587920815)

MetaTrader 5 / Trading


### Introduction

Welcome back to Part 26 of the Introduction to MQL5 series! In this article, we will concentrate on support and resistance zones, which are among the most fundamental ideas in technical analysis, and explore how to create an EA that makes trades based on these crucial price levels. Important trading choices are frequently made by market players at psychological barriers known as support and resistance zones. Resistance indicates a level where selling pressure is powerful enough to stop upward movement, while support indicates a level where purchasing pressure tends to overcome selling pressure and cause prices to rebound.

We looked at how to manually set up support and resistance zones using the rectangle chart object in [Part 24](https://www.mql5.com/en/articles/19912) of this series. This allowed you to create a semi-automated trading system that reacts to the zones you draw. However, what if you want the EA to automatically, without human assistance, identify every support and resistance zone on the chart? This article will specifically address that. As usual, we will cover MQL5 principles in a hands-on, beginner-friendly manner using a project-based approach. You'll discover how to identify these zones programmatically, track price movements surrounding them, and create an EA that responds sensibly to possible reversals.

### **How the EA Works**

In this project, we are creating an EA that automatically searches for support and resistance zones within a specified number of bars. Instead of relying on manually drawn zones, the EA will analyze recent price movements to detect areas where the market has repeatedly reacted, either by bouncing upward (support) or reversing downward (resistance).

Support

To find every swing low, the EA first loops through the specified number of bars. A swing low is a candle that indicates a brief bottom in price action when its low price is lower than the candles that come right before and after it.

After identifying a swing low, the EA discovers the minimal value between the candle's open and close prices as well as the precise low price of the candle that created it. The possible support zone is defined by these two values, although this has not been verified yet. After this swing low, the EA checks ahead in time to see if another swing low has formed in the same zone. The minimum between the open and close of the candle that generated the first swing low must be greater than the second swing low. This recurrence suggests that the market has once more examined the region, enhancing its importance.

After identifying this support zone, the EA confirms that no candle has broken below it. The zone is deemed illegitimate and not **considered** if any candle closes below it. The EA moves to a lower timeframe to keep an eye out for a bullish change of character when the zone has held steady. A buy trade is initiated when such a change in structure is identified, indicating that the market is being taken over by buyers and that the support zone has stayed firm.

![Figure 1. Support](https://c.mql5.com/2/176/figure_1__1.png)

Resistance

To find every swing high, the EA first loops through the chosen number of bars. A swing high is a candle that indicates a brief high in price movement when its high price is higher than the candles that come right before and after it.

Once a swing high has been identified, the EA logs the candle's high price as well as the highest value between its open and closing prices. Although it hasn't been verified yet, these figures indicate the possible resistance zone. To determine whether another swing high has developed inside the same zone, the EA then looks forward from the swing time. The gap between the open and close of the candle that created the first swing high must be less than the second swing high. Strong selling pressure is indicated by this repeated rejection at the same level, which confirms the area's potential as a barrier.

The EA then verifies that no candle has broken above the zone of resistance that has been indicated. The zone is deemed invalid and disregarded if a candle closes above it. The EA moves to a lower timeframe to keep an eye out for a bearish change of character once the zone is still valid. The EA initiates a sell trade when this bearish structural shift takes place, since it indicates that sellers are taking back control from that resistance level.

![Figure 2. Resistance](https://c.mql5.com/2/176/Figure_2__1.png)

### **Identifying First Swing Lows**

Finding the support zone will be our next step now that we have a better understanding of the project. Finding several important swing lows on the chart is the first stage in this process. These swing lows indicate possible regions where buying pressure may be present, since they show points where the market momentarily stopped moving down and turned back higher. The EA will be able to detect possible support zones that could subsequently act as entry points for bullish setups by recognizing these swing lows.

First, we need to duplicate the pertinent candle data. Next, we'll specify how many bars we want the EA to search for when identifying potential support and resistance zones. The function that determines swing highs and swing lows within the selected range will then be developed. This characteristic will serve as the foundation for spotting important price turning points, which are required to determine trustworthy levels of support and resistance.

Example:

```
input ENUM_TIMEFRAMES timeframe = PERIOD_W1; //SUPPORT AND RESISTANCE TIMEFRAME

double open[];
double close[];
double low[];
double high[];
datetime time[];

int bars_check = 200;

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
   CopyOpen(_Symbol, timeframe, TimeCurrent(), bars_check, open);
   CopyClose(_Symbol, timeframe, TimeCurrent(), bars_check, close);
   CopyLow(_Symbol, timeframe, TimeCurrent(), bars_check, low);
   CopyHigh(_Symbol, timeframe, TimeCurrent(), bars_check, high);
   CopyTime(_Symbol, timeframe, TimeCurrent(), bars_check, time);

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

The user can select the timeframe to search for resistance and support zones.  The arrays will hold the candle data. According to the option bars\_check, the EA will look at 200 candles while looking for swing points. Each candlestick on the chart has its price and timing data copied by the program. \_Symbol, the first argument, instructs the EA to transfer the data for the asset that is currently associated with the chart. Timeframe is the second parameter, and it indicates the user-selected timeframe. Any desired timeframe for analysis can be specified by the user because it is defined as an input. The next parameter, from which the data copying should begin, is where we utilized the TimeCurrent() function.

Since this function returns the server time as of right now, the EA starts copying data from the most recent candle. To find out how many candles the EA should copy, use the bars\_check argument. Here, it indicates that the EA will pull information for the previous 200 bars from the present moment. The EA can arrange the data for a further analysis by using the final argument, which indicates the array (open, close, low, high, or time) where the copied data will be kept.

Three arguments are required for the IsSwingLow() function. Each candle's low price is contained in an array called low\_price\[\], which is the first parameter. The current candle under analysis is specified by the second parameter, index. The number of candles that should be compared before and after the current one is determined by the third parameter, lookback. A loop within the function compares the low price of the current candle to the lows of the nearby candles, running from 1 to the lookback value. The function returns false, indicating that the current low is not a swing low, if it is higher than any of the neighboring lows. The function returns true, indicating that it is a legitimate swing low, if it stays the lowest when compared to all nearby candles.

This is the opposite of how the IsSwingHigh() function operates. To determine if the high of the current candle is higher than the highs of adjacent candles within the specified lookback range, it makes use of the high\_price\[\] array. The current high return is false if it is discovered to be lower than any of its neighbors. It is identified as a swing high if it returns true otherwise.

Finding every swing low on the chart comes next after copying the candle data. These swing lows are locations where the price momentarily turned back up, creating possible support zones. Our support zones will be created using each detected swing low as the first reference point. The EA will then verify these zones by looking for more touches and reactions within the same price range.

Example:

```
input ENUM_TIMEFRAMES timeframe = PERIOD_W1; //SUPPORT AND RESISTANCE TIMEFRAME

double open[];
double close[];
double low[];
double high[];
datetime time[];

int bars_check = 200;

double first_sup_price;
double first_sup_min_body_price;
datetime first_sup_time;

string support_object;
ulong chart_id =  ChartID();

int total_symbol_bars;
int z = 7;

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
   ObjectsDeleteAll(chart_id);

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   CopyOpen(_Symbol, timeframe, TimeCurrent(), bars_check, open);
   CopyClose(_Symbol, timeframe, TimeCurrent(), bars_check, close);
   CopyLow(_Symbol, timeframe, TimeCurrent(), bars_check, low);
   CopyHigh(_Symbol, timeframe, TimeCurrent(), bars_check, high);
   CopyTime(_Symbol, timeframe, TimeCurrent(), bars_check, time);

   total_symbol_bars = Bars(_Symbol, timeframe);

   if(total_symbol_bars >= bars_check)
     {

      for(int i = z ; i < bars_check - z; i++)
        {

         if(IsSwingLow(low, i, z))
           {

            first_sup_price = low[i];
            first_sup_min_body_price = MathMin(close[i], open[i]);
            first_sup_time = time[i];

            support_object = StringFormat("SUPPORT %f",first_sup_price);

            ObjectCreate(chart_id,support_object,OBJ_RECTANGLE,0,first_sup_time,first_sup_price,TimeCurrent(),first_sup_min_body_price);
            ObjectSetInteger(chart_id,support_object,OBJPROP_COLOR,clrBlue);
            ObjectSetInteger(chart_id,support_object,OBJPROP_BACK,true);
            ObjectSetInteger(chart_id,support_object,OBJPROP_FILL,true);

           }
        }

     }
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

Output:

![Figure 3. First Swing Low](https://c.mql5.com/2/176/figure_3__2.png)

Explanation:

Every swing low we find has basic information that we preserve by defining variables to keep vital features. It records the actual low price of the swing low candle, saves the time at which the swing low occurred, and represents the lower boundary of the candle body as the smaller value between the open and close prices.

We keep track of important information about every swing low, such as the time it happened, the lower border of the body of the candle, and the actual low price. To manage graphical objects, the chart ID is retrieved, and a variable is also made to save the name of the rectangle object that represents the support zone. Using a seven-candle lookback time, authentic swing lows are identified. After confirming that there are at least 200 candles available on the chart, the computer loops through the data from the seventh candle.

The swing low function is used by the program inside the loop to determine whether each candle makes a swing low pattern. If it occurs, the software logs the candle's timestamp, low price, and the smaller value between the open and close prices. After then, it combines the word "SUPPORT" with the real support fee to give the support zone a distinctive moniker.

By examining each candle to see whether its low is lower than the lows of the seven candles preceding and following it, the computer automatically detects swing lows. When a swing low is identified, the candle's precise time to establish the support zone is noted, along with the lowest price and the smaller value between the open and close prices. This zone is then visually represented on the chart by a rectangle that is stylized with color and fill and extends from the swing low to the present time. By combining the word "SUPPORT" with its price, each support zone is also given a distinctive name that makes it simple to monitor and see on the chart.

A rectangle that spans the period between the price of the swing low and the lower border of its body, from the time of the swing low to the current candle, is used by the program to graphically indicate the support zones on the chart.

### **Identifying Second Swing Lows**

The next stage is to determine whether another swing low has formed within the zone that was previously marked by the first swing low. This supports the authenticity and robustness of the support area. From the moment of the first swing low, the EA will watch for the emergence of a second swing low whose low price is still inside the limits of the previously determined support zone, that is, between the first swing low candle's lowest price and its minimum body price. When this need is satisfied, the market has repeatedly respected the same price level, strengthening the support zone and increasing its dependability for possible buy setups.

Example:

```
double second_sup_price;
datetime second_sup_time;
string first_low_txt;
string second_low_txt;
```

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   CopyOpen(_Symbol, timeframe, TimeCurrent(), bars_check, open);
   CopyClose(_Symbol, timeframe, TimeCurrent(), bars_check, close);
   CopyLow(_Symbol, timeframe, TimeCurrent(), bars_check, low);
   CopyHigh(_Symbol, timeframe, TimeCurrent(), bars_check, high);
   CopyTime(_Symbol, timeframe, TimeCurrent(), bars_check, time);

   total_symbol_bars = Bars(_Symbol, timeframe);

   if(total_symbol_bars >= bars_check)
     {

      for(int i = z ; i < bars_check - z; i++)
        {

         if(IsSwingLow(low, i, z))
           {

            first_sup_price = low[i];
            first_sup_min_body_price = MathMin(close[i], open[i]);
            first_sup_time = time[i];

            for(int j = i+1; j < bars_check - z; j++)
              {

               if(IsSwingLow(low, j, z) && low[j] <= first_sup_min_body_price &&  low[j] >= first_sup_price)
                 {

                  second_sup_price = low[j];
                  second_sup_time = time[j];

                  support_object = StringFormat("SUPPORT %f",first_sup_price);

                  ObjectCreate(chart_id,support_object,OBJ_RECTANGLE,0,first_sup_time,first_sup_price,TimeCurrent(),first_sup_min_body_price);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_COLOR,clrBlue);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_BACK,true);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_FILL,true);

                  first_low_txt = StringFormat("FIRST LOW%d",i);
                  ObjectCreate(chart_id,first_low_txt,OBJ_TEXT,0,first_sup_time,first_sup_price);
                  ObjectSetString(chart_id,first_low_txt,OBJPROP_TEXT,"1");

                  second_low_txt = StringFormat("SECOND LOW%d",i);
                  ObjectCreate(chart_id,second_low_txt,OBJ_TEXT,0,second_sup_time,second_sup_price);
                  ObjectSetString(chart_id,second_low_txt,OBJPROP_TEXT,"2");

                  break;
                 }
              }
           }
        }
     }
  }
```

Output:

![Figure 4. Second Swing Low](https://c.mql5.com/2/176/figure_4__2.png)

Explanation:

We monitor its time and price before searching for a second swing low, and we mark both swing lows on the chart for simple visual identification. A second swing low that comes after the first is then sought after using a for loop. One candle after the first swing low, or i + 1, the loop begins checking and continues through the remaining bars. The swing low function evaluates each candle inside the loop to see if it generates another swing low pattern.

The program simultaneously determines if the second swing low's low falls inside the first support zone, or between the first swing low's lowest price and the minimum body price. The program confirms that a second legitimate low has developed within the support zone by recording the price and time of the new swing low if these criteria are satisfied.

To draw the support zone only when both swing lows affirm the same area, we should now relocate the rectangle object that we previously constructed into this loop. This guarantees that the zone is merely a validated support level and is not plotted too soon.

### **Identifying Support Zone Breakout**

Two swing lows that we have found so far have turned around the same price range, creating what looks to be a possible support region. But not all zones with two reversals are appropriate for our approach. Zones where the market has twice respected the level without falling below it are the only ones we wish to retain. We must exclude any zones that have previously seen a breakout to guarantee this. By limiting the EA's emphasis to robust and active support zones where the price has demonstrated unambiguous respect and rejection, this step will improve the accuracy of future trading signals.

Example:

```
double second_sup_price;
datetime second_sup_time;
string first_low_txt;
string second_low_txt;

int sup_bars;
int sup_min_low_index;
double sup_min_low_price;
```

```
if(total_symbol_bars >= bars_check)
  {

   for(int i = z ; i < bars_check - z; i++)
     {

      if(IsSwingLow(low, i, z))
        {

         first_sup_price = low[i];
         first_sup_min_body_price = MathMin(close[i], open[i]);
         first_sup_time = time[i];

         for(int j = i+1; j < bars_check - z; j++)
           {

            if(IsSwingLow(low, j, z) && low[j] <= first_sup_min_body_price &&  low[j] >= first_sup_price)
              {

               second_sup_price = low[j];
               second_sup_time = time[j];

               sup_bars = Bars(_Symbol,timeframe,first_sup_time,TimeCurrent());
               sup_min_low_index = ArrayMinimum(low,i,sup_bars);
               sup_min_low_price = low[sup_min_low_index];

               if(sup_min_low_price >= first_sup_price)
                 {

                  support_object = StringFormat("SUPPORT %f",first_sup_price);

                  ObjectCreate(chart_id,support_object,OBJ_RECTANGLE,0,first_sup_time,first_sup_price,TimeCurrent(),first_sup_min_body_price);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_COLOR,clrBlue);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_BACK,true);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_FILL,true);

                  first_low_txt = StringFormat("FIRST LOW%d",i);
                  ObjectCreate(chart_id,first_low_txt,OBJ_TEXT,0,first_sup_time,first_sup_price);
                  ObjectSetString(chart_id,first_low_txt,OBJPROP_TEXT,"1");

                  second_low_txt = StringFormat("SECOND LOW%d",i);
                  ObjectCreate(chart_id,second_low_txt,OBJ_TEXT,0,second_sup_time,second_sup_price);
                  ObjectSetString(chart_id,second_low_txt,OBJPROP_TEXT,"2");

                 }
               break;
              }
           }
        }
     }
  }
```

Output:

![Figure 5. Support Breakout](https://c.mql5.com/2/176/Figure_5__2.png)

Explanation:

To ascertain whether the market has broken below the designated support zone or whether the zone is still valid, three variables are introduced in this section of the explanation. Between the time of the initial swing low and the present, the first variable counts the number of candles that have been created. This aids in defining the range of bars that will be examined to confirm the support zone or look for a possible breakout.

The program then determines which candle in that range has the lowest price, indicating the point at which there has been the most downward movement since the first swing low. The real pricing value at that lowest point is then obtained and saved for later examination.

After that, a criterion is applied to determine if the market has broken below or respected the support zone. The support zone is deemed legitimate, and the market is thought to be maintaining that level if the lowest price discovered stays above or equal to the first swing low price. In this situation, the support area is physically represented on the chart by drawing graphical components such as rectangles, lines, or labels. A breakout is indicated if the lowest price drops below the original support level; in this case, no graphical objects are shown because the zone is no longer valid.

### **Identifying Bullish Change of Character**

After confirming a valid support zone, the following stage is to spot a bullish shift of character. The market has twice rejected this support zone, indicating buying activity, and it has already been defined at this point with two swing lows. But having a zone that was refused twice is not enough to establish a possible bullish reversal; we also need to see a noticeable change in the market's structure.

We wait for the market to make another return to the support zone to do this. We start looking for a bullish change of character (CHOCH) pattern when the price returns to the zone. A change from bearish to bullish momentum is reflected in this pattern. A series of market structure points, including a high, a low, a lower high, and a lower low, must be formed by the pattern in this instance. We watch for a candle to break and close above the lower high once the lower low has formed. This breakout indicates a shift in direction and validates that buyers have taken charge.

The trade execution period is the lower timeframe on which the character change is detected. The lower timeframe is used to confirm entry by identifying a bullish change of character within or near the support and resistance zones that were first identified on a higher timeframe. Accurate and timely transaction entries are ensured by this multi-timeframe method.

Identifying Lower Low and Lower High

We will begin our examination with the most recent bar on the chart since we want to determine the most recent change in character. This implies that the lower low, which is the most recent point at which the price established a new low, should be the first thing we look for. Next, we will determine the lower high, then the low, and lastly, the high.

Example:

```
input ENUM_TIMEFRAMES timeframe = PERIOD_M30; //SUPPORT AND RESISTANCE TIMEFRAME
input ENUM_TIMEFRAMES exe_timeframe = PERIOD_M5; //EXECUTION TIMEFRAME
```

```
double exe_open[];
double exe_close[];
double exe_low[];
double exe_high[];
datetime exe_time[];

int exe_total_symbol_bars;

double lower_high;
datetime lower_high_time;
double lower_low;
datetime lower_low_time;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   ArraySetAsSeries(exe_close,true);
   ArraySetAsSeries(exe_open,true);
   ArraySetAsSeries(exe_high,true);
   ArraySetAsSeries(exe_low,true);
   ArraySetAsSeries(exe_time,true);

//---
   return(INIT_SUCCEEDED);
  }
```

```
CopyOpen(_Symbol, exe_timeframe, TimeCurrent(), bars_check, exe_open);
CopyClose(_Symbol, exe_timeframe, TimeCurrent(), bars_check, exe_close);
CopyLow(_Symbol, exe_timeframe, TimeCurrent(), bars_check, exe_low);
CopyHigh(_Symbol, exe_timeframe, TimeCurrent(), bars_check, exe_high);
CopyTime(_Symbol, exe_timeframe, TimeCurrent(), bars_check, exe_time);
```

```
if(total_symbol_bars >= bars_check)
  {

   for(int i = z ; i < bars_check - z; i++)
     {

      if(IsSwingLow(low, i, z))
        {

         first_sup_price = low[i];
         first_sup_min_body_price = MathMin(close[i], open[i]);
         first_sup_time = time[i];

         for(int j = i+1; j < bars_check - z; j++)
           {

            if(IsSwingLow(low, j, z) && low[j] <= first_sup_min_body_price &&  low[j] >= first_sup_price)
              {

               second_sup_price = low[j];
               second_sup_time = time[j];

               sup_bars = Bars(_Symbol,timeframe,first_sup_time,TimeCurrent());
               sup_min_low_index = ArrayMinimum(low,i,sup_bars);
               sup_min_low_price = low[sup_min_low_index];

               if(sup_min_low_price >= first_sup_price)
                 {

                  support_object = StringFormat("SUPPORT %f",first_sup_price);

                  ObjectCreate(chart_id,support_object,OBJ_RECTANGLE,0,first_sup_time,first_sup_price,TimeCurrent(),first_sup_min_body_price);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_COLOR,clrBlue);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_BACK,true);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_FILL,true);

                  first_low_txt = StringFormat("FIRST LOW%d",i);
                  ObjectCreate(chart_id,first_low_txt,OBJ_TEXT,0,first_sup_time,first_sup_price);
                  ObjectSetString(chart_id,first_low_txt,OBJPROP_TEXT,"1");

                  second_low_txt = StringFormat("SECOND LOW%d",i);
                  ObjectCreate(chart_id,second_low_txt,OBJ_TEXT,0,second_sup_time,second_sup_price);
                  ObjectSetString(chart_id,second_low_txt,OBJPROP_TEXT,"2");

                  exe_total_symbol_bars = Bars(_Symbol, exe_timeframe);

                  if(exe_total_symbol_bars >= bars_check)
                    {
                     for(int k = 4; k < bars_check-3; k++)
                       {
                        if(IsSwingLow(exe_low, k, 3))
                          {

                           lower_low = exe_low[k];
                           lower_low_time = exe_time[k];

                           for(int l = k; l < bars_check-3; l++)
                             {

                              if(IsSwingHigh(exe_high,l,3))
                                {
                                 lower_high = exe_high[l];
                                 lower_high_time = exe_time[l];

                                 break;
                                }
                             }

                           break;
                          }
                       }

                    }

                 }
               break;
              }
           }
        }
     }
  }
```

Explanation:

In this program area, the preparation and analysis of data for trade execution based on the change of character (CHoCH) is the main focus. The execution window, which in this instance is a shorter duration like the 5-minute chart, is defined first. To capture more accurate market movements, the actual confirmations for trade entries occur on a lower period, even when the support and resistance zones are indicated on higher timeframes, such as the daily or weekly charts.

After that, arrays are made to hold the execution timeframe's candle price and time data. These arrays are essential for structural analysis since they give the EA precise reference to each candle's highs, lows, opens, and closes. Together with the price and time of the most recent lower high and lower low, other variables are provided to hold the total number of candles accessible.

The program first verifies that there are enough candles on the chosen timeframe to conduct a useful analysis before continuing. It looks for swing lows by going through each candle after confirmation. When the current candle's low is below the lows of a few candles before and after it, it is proven to be a swing low, indicating a local market bottom. The lower low is the price and timing of the swing low when one is discovered.

Following the identification of a lower low, the software proceeds to look for the subsequent swing high. When the high of a candle surpasses the highs of adjacent candles, indicating a transient peak, it is proven to be a swing high. After identifying the lower low and lower high, the analysis phase is over, and the EA is prepared to assess market structure for possible bearish or bullish character shifts.

Identifying Low and High

Finding the new low and the new high that follow is the next stage after determining the lower low and lower high. This phase makes sure that the market structure is set up correctly to validate a legitimate character change. The market was previously in a decline when the lower high was above the lower low.

To indicate a weakening of the bearish move, the newly formed low should be situated between the two, that is, below the lower high but above the lower low. The market has changed momentum from bearish to bullish, indicating the beginning of a possible upward turnaround if the subsequent high, which turns into the higher high, is above all prior highs and lows.

Example:

```
double low_l;
datetime low_time;
double high_h;
datetime high_time;
```

```
for(int m = l; m < bars_check-3; m++)
  {

   if(IsSwingLow(exe_low, m, 3))
     {

      low_l = exe_low[m];
      low_time = exe_time[m];

      for(int n = m; n < bars_check-3; n++)
        {

         if(IsSwingHigh(exe_high,n,3))
           {

            high_h = exe_high[n];
            high_time = exe_time[n];

            break;
           }
        }
      break;
     }
```

Explanation:

This section of the software uses a number of criteria to identify the next significant swing points that support a bullish character shift. These consist of the newly established swing low and swing high price and time. The EA can ascertain whether the market structure has changed from a bearish to a bullish trend by monitoring these parameters.

The program searches recent candles for a new swing bottom before identifying these locations. It quickly looks for the next swing high after detecting the new low, noting its price and timing. Finding the new swing high and storing its details completes the identification of the crucial swing sequence required to assess the structural alteration.

When a bullish candle closes above the lower high that was previously created, it confirms a bullish change of character. This event essentially ends the bearish period by indicating that buyers have taken control. A possible trend reversal and the beginning of a new bullish structure are indicated by the market's formation of higher highs and higher lows.

Example:

```
datetime start_choch_time;
```

```
if(total_symbol_bars >= bars_check)
  {

   for(int i = z ; i < bars_check - z; i++)
     {

      if(IsSwingLow(low, i, z))
        {

         first_sup_price = low[i];
         first_sup_min_body_price = MathMin(close[i], open[i]);
         first_sup_time = time[i];

         for(int j = i+1; j < bars_check - z; j++)
           {

            if(IsSwingLow(low, j, z) && low[j] <= first_sup_min_body_price &&  low[j] >= first_sup_price)
              {

               second_sup_price = low[j];
               second_sup_time = time[j];
               start_choch_time = time[j+z];

               sup_bars = Bars(_Symbol,timeframe,first_sup_time,TimeCurrent());
               sup_min_low_index = ArrayMinimum(low,i,sup_bars);
               sup_min_low_price = low[sup_min_low_index];

               if(sup_min_low_price >= first_sup_price)
                 {

                  support_object = StringFormat("SUPPORT %f",first_sup_price);

                  ObjectCreate(chart_id,support_object,OBJ_RECTANGLE,0,first_sup_time,first_sup_price,TimeCurrent(),first_sup_min_body_price);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_COLOR,clrBlue);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_BACK,true);
                  ObjectSetInteger(chart_id,support_object,OBJPROP_FILL,true);

                  first_low_txt = StringFormat("FIRST LOW%d",i);
                  ObjectCreate(chart_id,first_low_txt,OBJ_TEXT,0,first_sup_time,first_sup_price);
                  ObjectSetString(chart_id,first_low_txt,OBJPROP_TEXT,"1");

                  second_low_txt = StringFormat("SECOND LOW%d",i);
                  ObjectCreate(chart_id,second_low_txt,OBJ_TEXT,0,second_sup_time,second_sup_price);
                  ObjectSetString(chart_id,second_low_txt,OBJPROP_TEXT,"2");

                  exe_total_symbol_bars = Bars(_Symbol, exe_timeframe);

                  if(exe_total_symbol_bars >= bars_check)
                    {
                     for(int k = 4; k < bars_check-3; k++)
                       {
                        if(IsSwingLow(exe_low, k, 3))
                          {

                           lower_low = exe_low[k];
                           lower_low_time = exe_time[k];

                           for(int l = k; l < bars_check-3; l++)
                             {

                              if(IsSwingHigh(exe_high,l,3))
                                {
                                 lower_high = exe_high[l];
                                 lower_high_time = exe_time[l];

                                 for(int m = l; m < bars_check-3; m++)
                                   {

                                    if(IsSwingLow(exe_low, m, 3))
                                      {

                                       low_l = exe_low[m];
                                       low_time = exe_time[m];

                                       for(int n = m; n < bars_check-3; n++)
                                         {

                                          if(IsSwingHigh(exe_high,n,3))
                                            {

                                             high_h = exe_high[n];
                                             high_time = exe_time[n];

                                             for(int o = k; o > 0; o--)
                                               {
                                                if(exe_close[o] > lower_high && exe_open[o] < lower_high)
                                                  {

                                                   if(lower_high > lower_low && low_l < lower_high && low_l > lower_low && high_h > low_l && high_h > lower_high && lower_low <= first_sup_min_body_price && lower_low >= first_sup_price
                                                      && high_time > start_choch_time)
                                                     {

                                                      ObjectCreate(chart_id,"LLLH",OBJ_TREND,0,lower_low_time,lower_low,lower_high_time,lower_high);
                                                      ObjectSetInteger(chart_id,"LLLH",OBJPROP_COLOR,clrRed);
                                                      ObjectSetInteger(chart_id,"LLLH",OBJPROP_WIDTH,2);

                                                      ObjectCreate(chart_id,"LHL",OBJ_TREND,0,lower_high_time,lower_high,low_time,low_l);
                                                      ObjectSetInteger(chart_id,"LHL",OBJPROP_COLOR,clrRed);
                                                      ObjectSetInteger(chart_id,"LHL",OBJPROP_WIDTH,2);

                                                      ObjectCreate(chart_id,"LH",OBJ_TREND,0,low_time,low_l,high_time,high_h);
                                                      ObjectSetInteger(chart_id,"LH",OBJPROP_COLOR,clrRed);
                                                      ObjectSetInteger(chart_id,"LH",OBJPROP_WIDTH,2);

                                                      ObjectCreate(chart_id,"S Cross Line",OBJ_TREND,0,lower_high_time,lower_high,exe_time[o],lower_high);
                                                      ObjectSetInteger(chart_id,"S Cross Line",OBJPROP_COLOR,clrRed);
                                                      ObjectSetInteger(chart_id,"S Cross Line",OBJPROP_WIDTH,2);

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
               break;
              }
           }
        }
     }
  }
```

![Figure 6. Bullish CHOCH](https://c.mql5.com/2/176/figure_6__2.png)

Explanation:

This part first establishes a reference point to indicate the appropriate time to start the change of character (CHOCH) analysis. This guarantees that after verifying the most current support zone, the program will only begin searching for structural changes.

The program then searches through the candles in reverse to identify one that closes above the lower high, indicating the beginning of a bullish shift in sentiment. Strong indications of a breakthrough and a possible market reversal are provided by a candle that closes above the lower high and opens below it.

After locating such a candle, the program runs a number of verification tests to make sure the change of character is valid. The EA connects the major swing points (lower low, lower high, new low, and higher high) on the chart to graphically indicate the structure after all of these requirements are met. Additionally, it draws attention to the breakout candle that closed above the lower high, indicating that buyers have taken back control of the market and acting as the last confirmation of a bullish change of character.

### **Trade Execution**

All the earlier confirmations come together to create a possible trade setup in the logic's last section, which is dedicated to trade execution. The code now determines if every one of the specified requirements for a bullish Change of Character (CHoCH) has been satisfied. The software then marks the structure on the chart and gets ready to execute the deal after confirmation.

Example:

```
#include <Trade/Trade.mqh>
CTrade trade;
```

```
input ENUM_TIMEFRAMES timeframe = PERIOD_W1; //SUPPORT AND RESISTANCE TIMEFRAME
input ENUM_TIMEFRAMES exe_timeframe = PERIOD_M30; //EXECUTION TIMEFRAME
input double lot_size = 0.2; // Lot Size
input double RRR = 3; //RRR

double ask_price;
double take_profit;
datetime lastTradeBarTime = 0;
```

```
datetime currentBarTime = iTime(_Symbol, exe_timeframe, 0);
ask_price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
```

```
for(int o = k; o > 0; o--)
  {
   if(exe_close[o] > lower_high && exe_open[o] < lower_high)
     {

      if(lower_high > lower_low && low_l < lower_high && low_l > lower_low && high_h > low_l && high_h > lower_high && lower_low <= first_sup_min_body_price && lower_low >= first_sup_price
         && high_time > start_choch_time)
        {

         ObjectCreate(chart_id,"LLLH",OBJ_TREND,0,lower_low_time,lower_low,lower_high_time,lower_high);
         ObjectSetInteger(chart_id,"LLLH",OBJPROP_COLOR,clrRed);
         ObjectSetInteger(chart_id,"LLLH",OBJPROP_WIDTH,2);

         ObjectCreate(chart_id,"LHL",OBJ_TREND,0,lower_high_time,lower_high,low_time,low_l);
         ObjectSetInteger(chart_id,"LHL",OBJPROP_COLOR,clrRed);
         ObjectSetInteger(chart_id,"LHL",OBJPROP_WIDTH,2);

         ObjectCreate(chart_id,"LH",OBJ_TREND,0,low_time,low_l,high_time,high_h);
         ObjectSetInteger(chart_id,"LH",OBJPROP_COLOR,clrRed);
         ObjectSetInteger(chart_id,"LH",OBJPROP_WIDTH,2);

         ObjectCreate(chart_id,"S Cross Line",OBJ_TREND,0,lower_high_time,lower_high,exe_time[o],lower_high);
         ObjectSetInteger(chart_id,"S Cross Line",OBJPROP_COLOR,clrRed);
         ObjectSetInteger(chart_id,"S Cross Line",OBJPROP_WIDTH,2);

         if(exe_time[1] == exe_time[o] && currentBarTime != lastTradeBarTime)
           {

            take_profit = MathAbs(ask_price + ((ask_price - lower_low) * RRR));

            trade.Buy(lot_size,_Symbol,ask_price,lower_low,take_profit);
            lastTradeBarTime = currentBarTime;

           }

        }
      break;
     }
  }
```

Output:

![Figure 7. Trade Execution](https://c.mql5.com/2/176/Figure_7__2.png)

Explanation:

After verifying a bullish Change of Character (CHoCH), this section describes the trade setup and execution procedure. For the system to automatically place buy orders when all trading requirements are satisfied, it first enables trade activities. The transaction's parameters are defined by a number of inputs, including the lot size for each trade, the timeframes for support/resistance detection and execution, and the risk-to-reward ratio (RRR), which establishes the desired profit in relation to the risk.

To make sure it only reacts once each candle, the algorithm also monitors the current market price and candle time. This stops people from entering the same pub more than once. The program verifies that all the parameters are in line before placing a trade, such as the anticipated candle timing and the lack of prior trades on that bar. Based on the RRR, it determines the take-profit level after confirmation, places the stop loss at the lower low, and automatically initiates a buy trade. As a result, trade entries are guaranteed to be disciplined, consistent, and grounded in established market mechanisms rather than arbitrary judgments.

### **Identifying Resistance Zone**

The resistance zone, which is just the opposite of the support zone, must also be determined. Swing highs are used to identify the resistance zone, while swing lows are used to identify the support zone. Stated differently, we search for locations where the market has refused upward movement rather than areas where it has rejected downward movement.

The same logic that determines support also determines a resistance zone, but the opposite is true. Before defining the zone surrounding a swing high, we first identify the high. Assessing if a second swing high has developed within the same resistance zone is the next step after determining the first swing high and marking it. The price has tested and rejected the area twice, proving its validity as a resistance zone, if the market makes a second swing high inside the highlighted zone.

The market may find it difficult to break above this resistance zone and may even revert lower, making it a possible selling opportunity. We must wait for a bearish change in character, though, to verify if a bearish setup is actually developing. As a result, the market needs to start producing lower highs and lower lows instead of higher highs and lower lows. The final indication that sellers have gained control and that a downward move may ensue is this bearish shift of character that takes place inside or near the resistance zone.

Example:

```
double first_res_price;
double first_res_max_body_price;
datetime first_res_time;

double second_res_price;
datetime second_res_time;

string resistance_object;
int res_bars;
int res_max_high_index;
double res_max_high_price;

string first_high_txt;
string second_high_txt;

double higher_high;
datetime higher_high_time;
double higher_low;
datetime higher_low_time;
```

```
//RESISTANCE
for(int i = z ; i < bars_check - z; i++)
  {

   if(IsSwingHigh(high, i, z))
     {

      first_res_price = high[i];
      first_res_max_body_price = MathMax(close[i], open[i]);
      first_res_time = time[i];

      for(int j = i+1; j < bars_check - z; j++)
        {
         if(IsSwingHigh(high, j, z) && high[j] >= first_res_max_body_price &&  high[j] <= first_res_price)
           {

            second_res_price = high[j];
            second_res_time = time[j];
            start_choch_time = time[j+z];

            resistance_object = StringFormat("RESISTANCE %f",first_res_price);

            res_bars = Bars(_Symbol,timeframe,first_res_time,TimeCurrent());
            res_max_high_index = ArrayMaximum(high,i,res_bars);
            res_max_high_price = high[res_max_high_index];

            if(res_max_high_price <= first_res_price)
              {
               ObjectCreate(chart_id,resistance_object,OBJ_RECTANGLE,0,first_res_time,first_res_price,TimeCurrent(),first_res_max_body_price);
               ObjectSetInteger(chart_id,resistance_object,OBJPROP_COLOR,clrGreen);
               ObjectSetInteger(chart_id,resistance_object,OBJPROP_BACK,true);
               ObjectSetInteger(chart_id,resistance_object,OBJPROP_FILL,true);

               first_high_txt = StringFormat("FIRST HIGH%d",i);
               ObjectCreate(chart_id,first_high_txt,OBJ_TEXT,0,first_res_time,first_res_price);
               ObjectSetString(chart_id,first_high_txt,OBJPROP_TEXT,"1");

               second_high_txt = StringFormat("SECOND HIGH%d",i);
               ObjectCreate(chart_id,second_high_txt,OBJ_TEXT,0,second_res_time,second_res_price);
               ObjectSetString(chart_id,second_high_txt,OBJPROP_TEXT,"2");

               if(exe_total_symbol_bars >= bars_check)
                 {

                  for(int k = 4; k < bars_check-3; k++)
                    {
                     if(IsSwingHigh(exe_high, k, 3))
                       {

                        higher_high = exe_high[k];
                        higher_high_time = exe_time[k];

                        for(int l = k; l < bars_check-3; l++)
                          {

                           if(IsSwingLow(exe_low,l,3))
                             {

                              higher_low = exe_low[l];
                              higher_low_time = exe_time[l];

                              for(int m = l; m < bars_check-3; m++)
                                {

                                 if(IsSwingHigh(exe_high, m, 3))
                                   {

                                    high_h = exe_high[m];
                                    high_time = exe_time[m];

                                    for(int n = m; n < bars_check-3; n++)
                                      {

                                       if(IsSwingLow(exe_low,n,3))
                                         {

                                          low_l = exe_low[n];
                                          low_time = exe_time[n];

                                          for(int o = k; o > 0; o--)
                                            {
                                             if(exe_close[o] < higher_low && exe_open[o] > higher_low)
                                               {

                                                if(higher_low < higher_high && high_h > higher_low && high_h < higher_high && low_l < high_h && low_l < higher_low && higher_high >= first_res_max_body_price
                                                   && higher_high <= first_res_price && low_time > start_choch_time)
                                                  {

                                                   ObjectCreate(chart_id,"HHHL",OBJ_TREND,0,higher_high_time,higher_high,higher_low_time,higher_low);
                                                   ObjectSetInteger(chart_id,"HHHL",OBJPROP_COLOR,clrRed);
                                                   ObjectSetInteger(chart_id,"HHHL",OBJPROP_WIDTH,2);

                                                   ObjectCreate(chart_id,"HLH",OBJ_TREND,0,higher_low_time,higher_low,high_time,high_h);
                                                   ObjectSetInteger(chart_id,"HLH",OBJPROP_COLOR,clrRed);
                                                   ObjectSetInteger(chart_id,"HLH",OBJPROP_WIDTH,2);

                                                   ObjectCreate(chart_id,"HL",OBJ_TREND,0,high_time,high_h,low_time,low_l);
                                                   ObjectSetInteger(chart_id,"HL",OBJPROP_COLOR,clrRed);
                                                   ObjectSetInteger(chart_id,"HL",OBJPROP_WIDTH,2);

                                                   ObjectCreate(chart_id,"R Cross Line",OBJ_TREND,0,higher_low_time,higher_low,exe_time[o],higher_low);
                                                   ObjectSetInteger(chart_id,"R Cross Line",OBJPROP_COLOR,clrRed);
                                                   ObjectSetInteger(chart_id,"R Cross Line",OBJPROP_WIDTH,2);

                                                   if(exe_time[1] == exe_time[o] && currentBarTime != lastTradeBarTime)
                                                     {

                                                      take_profit = MathAbs(ask_price - ((high_h - ask_price) * RRR));

                                                      trade.Sell(lot_size,_Symbol,ask_price,higher_high,take_profit);
                                                      lastTradeBarTime = currentBarTime;

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
            break;
           }

        }

     }
  }
```

Output:

![Figure 8. Resistance Trade Execution](https://c.mql5.com/2/176/figure_8__1.png)

Explanation:

Finding notable swing highs on the chart by looping through the candles is the first step in determining the resistance zone. Finding a swing high and noting its high price, the greater of its open or close prices, and the time it formed is the first step. The program looks for a second swing high that falls inside the zone that the first high defines after detecting this initial swing high.

The market has tested the region twice, confirming its validity as a resistance zone with its second swing high. Before drawing the visual rectangle for the resistance region on the chart, the computer determines the highest price between the two highs to make sure no candle has broken above the zone. To identify the two locations that make up the zone, the program then uses text objects to designate the first and second swing highs.

The program then looks for a bearish change of character for trade confirmation after noting the resistance zone. It looks for a price action sequence with a higher high, followed by a higher low, then a lower high and lower low, creating a bearish pattern, using a lower period that is designated for execution. The application creates trend lines linking the pertinent highs and lows for visualization, noting them together with their timestamps.

When a bearish candle closes below the pattern's higher low, it serves as the last confirmation. The EA determines the take profit by multiplying the distance between the entry price and the pattern's high by the risk-to-reward ratio, and then executes a sell transaction when this condition coincides with the resistance zone. By comparing the current bar time to the most recent trade that was executed, the method guarantees that each candle only contains one trade. With this method, resistance zones are automatically identified, verified by price action patterns, and trades are only executed when all requirements are met.

Note:

_This article's strategy is entirely project-based and intended to teach readers MQL5 through real-world, hands-on application. It is not a guaranteed method for making profits in live trading._

### **Conclusion**

The EA we built identifies support and resistance zones by detecting swing highs and lows that have been tested at least twice, ensuring the zones are significant. It then monitors for bullish or bearish changes of character on a lower timeframe, confirming the proper entry points. By combining higher timeframe zone detection with lower timeframe price action, the EA can automatically execute trades with defined stop loss and take profit levels.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20021.zip "Download all attachments in the single ZIP archive")

[Project\_18\_Support\_and\_Resistance.mq5](https://www.mql5.com/en/articles/download/20021/Project_18_Support_and_Resistance.mq5 "Download Project_18_Support_and_Resistance.mq5")(19.62 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/498768)**

![Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://c.mql5.com/2/177/20026-statistical-arbitrage-through-logo__1.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)

In this article, we propose a scoring system for mean-reversion strategies based on statistical arbitrage of cointegrated stocks. The article suggests criteria that go from liquidity and transaction costs to the number of cointegration ranks and time to mean-reversion, while taking into account the strategic criteria of data frequency (timeframe) and the lookback period for cointegration tests, which are evaluated before the score ranking properly. The files required for the reproduction of the backtest are provided, and their results are commented on as well.

![From Novice to Expert: Parameter Control Utility](https://c.mql5.com/2/177/19918-from-novice-to-expert-parameter-logo__1.png)[From Novice to Expert: Parameter Control Utility](https://www.mql5.com/en/articles/19918)

Imagine transforming the traditional EA or indicator input properties into a real-time, on-chart control interface. This discussion builds upon our foundational work in the Market Periods Synchronizer indicator, marking a significant evolution in how we visualize and manage higher-timeframe (HTF) market structures. Here, we turn that concept into a fully interactive utility—a dashboard that brings dynamic control and enhanced multi-period price action visualization directly onto the chart. Join us as we explore how this innovation reshapes the way traders interact with their tools.

![Price Action Analysis Toolkit Development (Part 47): Tracking Forex Sessions and Breakouts in MetaTrader 5](https://c.mql5.com/2/177/19944-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 47): Tracking Forex Sessions and Breakouts in MetaTrader 5](https://www.mql5.com/en/articles/19944)

Global market sessions shape the rhythm of the trading day, and understanding their overlap is vital to timing entries and exits. In this article, we’ll build an interactive trading sessions  EA that brings those global hours to life directly on your chart. The EA automatically plots color‑coded rectangles for the Asia, Tokyo, London, and New York sessions, updating in real time as each market opens or closes. It features on‑chart toggle buttons, a dynamic information panel, and a scrolling ticker headline that streams live status and breakout messages. Tested on different brokers, this EA combines precision with style—helping traders see volatility transitions, identify cross‑session breakouts, and stay visually connected to the global market’s pulse.

![From Basic to Intermediate: Template and Typename (V)](https://c.mql5.com/2/116/Do_b8sico_ao_intermediurio_Template_e_Typename____LOGO.png)[From Basic to Intermediate: Template and Typename (V)](https://www.mql5.com/en/articles/15671)

In this article, we'll explore one last simple use case for templates, and discuss the benefits and necessity of using typename in your code. Although this article may seem a bit complicated at first, it is important to understand it properly in order to use templates and typename later.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=xidlqxvejxtthwvtqcsmokryokevyumw&ssn=1769093039715516659&ssn_dr=0&ssn_sr=0&fv_date=1769093039&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20021&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%2026)%3A%20Building%20an%20EA%20Using%20Support%20and%20Resistance%20Zones%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909303964465400&fz_uniq=5049338706587920815&sv=2552)

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