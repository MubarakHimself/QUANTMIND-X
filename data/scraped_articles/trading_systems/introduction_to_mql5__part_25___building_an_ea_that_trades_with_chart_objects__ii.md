---
title: Introduction to MQL5 (Part 25): Building an EA that Trades with Chart Objects (II)
url: https://www.mql5.com/en/articles/19968
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:32:57.162802
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/19968&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062545602174166070)

MetaTrader 5 / Trading systems


### Introduction

Welcome back to Part 25 of the Introduction to MQL5 series! In the previous article, we explored how combining manual chart analysis with automated trading execution using chart objects helps bridge the gap between discretionary and automated trading. That project focused on support and resistance zones drawn with the Rectangle tool.

This article is a continuation of that concept, but we'll be using trend lines this time. Among the most widely used techniques in technical analysis, trend lines assist you in determining the direction of the market as well as possible breakout or reversal points. We will create an EA that can identify trend line objects on the chart and, based on the state of the market, will automatically initiate trades when the price breaks above or below them.

### **How the EA Works**

The user will manually draw two trend lines on the chart, which the EA will use. One line showing a downtrend is drawn from a higher point to a lower point, while another showing an uptrend is drawn from a lower point to a higher position. The EA will use their names to identify the appropriate objects on the chart. This implies that the user will enter the trend lines' precise names in the input settings. Even when there are several trend lines on the chart, this method provides the EA with a clear reference for which things to keep an eye on.

Additionally, the user will have the ability to specify the timeframe that the EA should track, the lot size that they would want to utilize, and the trading style that they would like to employ. Both, where the EA handles both reversal and breakout-and-retest conditions depending on how the price behaves around the trend line, and reversal, where the EA opens a trade when the price touches the trend line and displays a reversal signal, are among the available trading modes. The Breakout and Retest mode waits for the price to break through the trend line before opening a trade.

![Figure 1. How the EA Works](https://c.mql5.com/2/176/Figure_1.png)

For trade management, the EA will automatically set the stop loss (SL) and take profit (TP) levels based on the latest closed candle.

- For a buy trade, the SL will be placed at the low of the most recently closed candle.
- Sell trade, the SL will be placed at the high of the most recently closed candle.
- The take profit (TP) will be set to 1:4 of the distance between the entry price and the stop-loss level.

![Figure 2. Reversal](https://c.mql5.com/2/176/Figure_2.png)

![Figure 3. Breakout and Retest](https://c.mql5.com/2/176/figure_3.png)

### **Identifying Trend Lines**

The next stage is to explore how the EA will recognize the trend lines on the chart now that you have a thorough understanding of how it operates. The EA needs a straightforward method to distinguish between the lines that indicate the uptrend and the downturn because it depends on user-drawn trend lines.

Trend line names will be used to accomplish this. The names the user enters in the input settings will be used by the EA to look for chart items. Through name matching, the EA can accurately locate each trend line on the chart and follow its positions in real time. The EA will only react to the particular trend lines that the user has designated, even if the chart contains other lines or illustrations.

Example:

```
input string down_trend = ""; // Down Trend Line
input string up_trend = ""; // Up Trend Line

ulong chart_id = ChartID();
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
   ObjectSetInteger(chart_id,down_trend,OBJPROP_RAY_RIGHT,true);
   ObjectSetInteger(chart_id,up_trend,OBJPROP_RAY_RIGHT,true);

  }
```

Explanation:

The chart's trend lines must be understood by the EA before it can recognize and manipulate them. To do this, we design input choices that let the user enter the two trend lines' names. One line will be used to indicate the downtrend, while the other will be used to symbolize the rise. Even in the presence of numerous other drawings or signs, this naming technique aids the EA in precisely identifying which objects on the chart are its property.

![Figure 4. Object Name](https://c.mql5.com/2/176/figure_4.png)

The chart that the EA is working on must likewise be known to it. Each chart in MetaTrader 5 includes a chart ID, which is a special identification number. By obtaining this ID, the EA guarantees that it only engages with the appropriate chart and its elements.

Extending the trend lines to the chart's right side is a crucial component of this arrangement. Because of this modification, the trend lines can extend into future price regions and continue past their anchor points. Because it allows the EA to identify price interactions with trend lines even after new candles have formed, it is a crucial step. The lines would stop at their anchor points without this extension, making it impossible for the EA to spot upcoming breakouts or reversals. As the market changes, extending them beyond the anchor points guarantees reliable signal recognition and ongoing monitoring.

![Figure 5. Anchor Price](https://c.mql5.com/2/176/Figure_5.png)

### **Retrieving Candle and Down Trend Line Object Data**

For the EA to examine how the price moves around the trend lines, the next step is to retrieve both candle and trend line data. The candle data for the chart's final five candles will be copied first by the EA. You can tell if a breakout or reversal is possible based on these recent candles alone.

The EA will gather the trend line data within the same five-candle range after getting the candle data. It will only pay attention to the trend line prices that match those recent candles rather than the anchor points. By doing this, the EA will be able to compare the current candle movements with the precise price levels of the trend lines for that same time frame.

The EA can identify when a price breaks above, below, or reverses from a trend line by examining the correlation between the prices of the trend line and the last five candles. This technique guarantees that the EA responds only to the most recent and pertinent market moves and improves the accuracy of the detection process.

Analogy

Consider the EA as a vigilant watchdog on the market, similar to a detective examining a crime scene. Every candle represents what transpired in terms of price at a certain moment, much like a photograph. Conversely, the trend line, which denotes significant boundaries, resembles a rope that is stretched over the chart.

The EA examines each candle separately, paying particular attention to the final five. It examines each candle's properties as well as the precise location of the trend line at that particular time. This procedure is like looking at two photos simultaneously, one of which shows the movement of the candle and the other the location of the trend line during that movement.

The EA starts its analysis after gathering this data. It poses queries like, did the top or low of the candle contact the trend line? Was it above or below it when the torch crossed? Was there a bullish or bearish candle?

Similar to how a trader would visually inspect a chart to see when price contacts or breaks through a line, the EA can comprehend how the candle interacted with the trend line by comparing these two pieces of information side by side.

Example:

```
input string down_trend = ""; // Down Trend Line
input string up_trend = ""; // Up Trend Line
input ENUM_TIMEFRAMES time_frame = PERIOD_CURRENT; // TIME FRAME

ulong chart_id = ChartID();

double close_price[];
double open_price[];
double low_price[];
double high_price[];
datetime time_price[];

double td_line_value;
double td1_line_value;
double td2_line_value;
double td3_line_value;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ArraySetAsSeries(close_price, true);
   ArraySetAsSeries(open_price, true);
   ArraySetAsSeries(low_price, true);
   ArraySetAsSeries(high_price, true);
   ArraySetAsSeries(time_price, true);

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
   ObjectSetInteger(chart_id,down_trend,OBJPROP_RAY_RIGHT,true);
   ObjectSetInteger(chart_id,up_trend,OBJPROP_RAY_RIGHT,true);

   CopyOpen(_Symbol, time_frame, 1, 5, open_price);
   CopyClose(_Symbol, time_frame, 1, 5, close_price);
   CopyLow(_Symbol, time_frame, 1, 5, low_price);
   CopyHigh(_Symbol, time_frame, 1, 5, high_price);
   CopyTime(_Symbol, time_frame, 1, 5, time_price);

//DOWN TREND

   td_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[0],0);
   td1_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[1],0);
   td2_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[2],0);
   td3_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[3],0);

  }
```

Explanation:

The program establishes the time range it will run on and gets ready to store candle information in empty containers at this initial step. The opening, closing, highest, and lowest prices of each candle, along with the precise moment the candle appeared on the chart, will subsequently be stored in these containers. Before beginning analysis, this configuration guarantees that the program can get comprehensive and well-structured price data from the market.

```
double td_line_value;
double td1_line_value;
double td2_line_value;
double td3_line_value;
```

At this stage, storage spaces are set up to accommodate numerical data that shows the trend line's location at particular times. The computer will use these figures to compare the market price's movement with the chart's downward line. The containers are then arranged such that the most recent information appears first. The computer can now analyze recent market activity more easily because it can access the most recent candles instantly, bypassing the need to sort through older data. Furthermore, all containers are ensured to be properly arranged before receiving market values because.

The program then collects information for the last five candles. It gathers their pricing as well as the times at which they were displayed on the chart. The program can monitor price movements around the trend line and determine whether a breakout or reversal pattern is developing by using this brief candle series.

```
td_line_value  = ObjectGetValueByTime(chart_id, down_trend, time_price[0], 0);
td1_line_value = ObjectGetValueByTime(chart_id, down_trend, time_price[1], 0);
td2_line_value = ObjectGetValueByTime(chart_id, down_trend, time_price[2], 0);
td3_line_value = ObjectGetValueByTime(chart_id, down_trend, time_price[3], 0);
```

Every line obtains the trend line's price value at a certain candle time. The function is instructed which chart to examine by the first parameter, which also specifies the name of the object (in this case, the downtrend line), the precise time to check (corresponding to the formation of each candle), and the chart window (with 0 denoting the main price chart) by the second parameter.

In simple terms, this function checks where the trend line was at the exact instant each candle appeared. The EA can then compare the candle's peak, low, or close with the line's position since it returns the line's price level at that moment. This helps identify breakout or reversal situations by allowing one to determine whether the candle touched or crossed the trend line.

### **Identifying Downward Trend Reversal**

Finding a downtrend reversal, which indicates a sell opportunity, is the next stage. In this instance, the EA will keep an eye on the interaction between the downward trend line and the recent candles. The EA will determine whether a candle closes below the line, signaling a potential reversal, when the price approaches or touches it. This conduct is a possible sign for a sell transaction since it indicates that sellers are taking back control of the market.

But it's important to avoid overtrading or duplicating indications. The EA initially verifies that there hasn't been a recent trade signal within the last several bars before doing this. This ensures that the program only reacts to new setups and stops it from opening redundant or unnecessary trades in the same short period of time.

Example:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   ObjectSetInteger(chart_id,down_trend,OBJPROP_RAY_RIGHT,true);
   ObjectSetInteger(chart_id,up_trend,OBJPROP_RAY_RIGHT,true);

   CopyOpen(_Symbol, time_frame, 1, 5, open_price);
   CopyClose(_Symbol, time_frame, 1, 5, close_price);
   CopyLow(_Symbol, time_frame, 1, 5, low_price);
   CopyHigh(_Symbol, time_frame, 1, 5, high_price);
   CopyTime(_Symbol, time_frame, 1, 5, time_price);

//DOWN TREND

   td_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[0],0);
   td1_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[1],0);
   td2_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[2],0);
   td3_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[3],0);

   bool prev_touch_down = false;

   if((high_price[1] > td1_line_value && close_price[1] < open_price[1])
      ||
      (high_price[2] > td2_line_value && close_price[2] < open_price[2])
     )
     {
      prev_touch_down = true;
     }
   int no_bars_down = 0;
   for(int i = 0; i <= 3; i++)
     {
      if(high_price[i] > ObjectGetValueByTime(chart_id,down_trend,time_price[i],0) && open_price[i] < ObjectGetValueByTime(chart_id,down_trend,time_price[i],0))
        {
         for(int j = i; j >= 0; j--)
           {
            if(close_price[j] < open_price[j] && close_price[j] < ObjectGetValueByTime(chart_id,down_trend,time_price[j],0))
              {
               no_bars_down = Bars(_Symbol,time_frame,time_price[j],TimeCurrent());

               break;
              }
           }
         break;
        }
     }
  }
```

Explanation:

```
bool prev_touch_down = false;

if((high_price[1] > td1_line_value && close_price[1] < open_price[1])
   ||
   (high_price[2] > td2_line_value && close_price[2] < open_price[2])
  )
  {
   prev_touch_down = true;
  }
```

A variable is initialized as false in the first line. This variable acts as a marker to indicate whether a bearish candle has recently followed a valid touch on the downtrend line. Since no such condition was found at the beginning, the flag is still false. The concept is that this variable will be updated to true after the system confirms a bearish reaction to the trend line. This variable can then be used to govern further trading actions, including avoiding multiple or repeated sell signals in a short period of time.

The if statement's condition examines two recent candles to see if either of them satisfies the reversal conditions. (high\_price\[1\] > td1\_line\_value && close\_price\[1\] < open\_price\[1\]) is the first part of the condition that looks at the candle that closed right before the current one. It determines if the candle's peak crossed the trend line at that moment, signifying that the price had touched or marginally penetrated the line. The candle is then confirmed as a bearish candle by confirming that it closed lower than it opened. In order to test the trend line, buyers may have driven the price higher, but sellers took back control and drove the market lower, leaving a rejection wick above.

To record previous rejections, the second section examines the candle from two bars ago. It is deemed legitimate if one of the two candles satisfies the rejection criteria.

The code enclosed in braces changes the value of the flag to true when the condition is evaluated as true. According to this report, the market has already reacted negatively to the downward trend line. This flag can be used as a safeguard later in the program to stop overtrading, making sure that repeated touches around the same location don't cause the system to produce additional sell signals.

In conclusion, this code block aids in locating a legitimate downtrend reversal signal. When a candle approaches the trend line from below, touches it, and closes bearish, it is confirmed to have done so. The program lays the groundwork for more disciplined trade execution by identifying this behavior and flagging it. This way, bearish reversals are only confirmed when price movement blatantly violates the trend line, and superfluous duplicate signals are avoided.

```
int no_bars_down = 0;

for(int i = 0; i <= 3; i++)
  {
   if(high_price[i] > ObjectGetValueByTime(chart_id,down_trend,time_price[i],0) && open_price[i] < ObjectGetValueByTime(chart_id,down_trend,time_price[i],0))
     {
      for(int j = i; j >= 0; j--)
        {
         if(close_price[j] < open_price[j] && close_price[j] < ObjectGetValueByTime(chart_id,down_trend,time_price[j],0))
           {
            no_bars_down = Bars(_Symbol,time_frame,time_price[j],TimeCurrent());

            break;
           }
        }
      break;
     }
  }
```

int no\_bars\_down = 0; is used to initialize a counter to zero at the start of the code. The number of bars that have elapsed since the last legitimate touch and rejection of the downward trend line will subsequently be recorded by this counter. The loop looks at the last four candles to see if the opening price is still below the trend line and the high of each candle has crossed over. This makes it easier to spot candles that made contact with the trend line from below, which is the initial indication of a potential reversal.

The program then looks backward after detecting such a candle to locate the first one that closes below the trend line and is bearish. This stage validates the signal for a possible sell opportunity by confirming that the market rejected the trend line. The program determines the number of bars that have elapsed since the confirming candle and up to the present moment. This count aids in avoiding repeated or duplicate alerts. The EA makes sure it only considers fresh trading opportunities after sufficient market movement has taken place by keeping track of the number of bars since the last valid rejection. This significantly lowers overtrading.

### **Adding the Entry Conditions**

Adding the entry condition for the sell trade is the next stage. The EA can now specify the exact circumstances under which it will create a position after verifying that the downtrend reversal has taken place and making sure that no recent signal has previously been triggered. To make sure that only legitimate setups initiate trades, this entails combining the trend line contact, the bearish candle confirmation, and the number of bars since the last signal. The EA can enter the market methodically and prevent overtrading by establishing these entry rules, which guarantee that every trade corresponds with the chart's reversal pattern.

Example:

```
#include <Trade/Trade.mqh>
CTrade trade;

int MagicNumber = 53217;
```

```
datetime lastTradeBarTime = 0;
double ask_price;
double take_profit;
```

```
//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ArraySetAsSeries(close_price, true);
   ArraySetAsSeries(open_price, true);
   ArraySetAsSeries(low_price, true);
   ArraySetAsSeries(high_price, true);
   ArraySetAsSeries(time_price, true);

   trade.SetExpertMagicNumber(MagicNumber);

//---
   return(INIT_SUCCEEDED);
  }
```

```
ask_price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
datetime currentBarTime = iTime(_Symbol, time_frame, 0);

if(((high_price[1] >= td1_line_value && open_price[1] < td1_line_value) || (high_price[2] >= td2_line_value && open_price[2] < td2_line_value)
    || (high_price[3] >= td3_line_value && open_price[3] < td3_line_value) || (high_price[0] >= td_line_value))
   && (close_price[0] < td_line_value && close_price[0] < open_price[0] && open_price[1] < td1_line_value)
   && (no_bars_down < 3)
   && prev_touch_down == false
   && (currentBarTime != lastTradeBarTime)
  )
  {
   take_profit = MathAbs(ask_price - ((high_price[0] - ask_price) * 4));

   trade.Sell(lot_size,_Symbol,ask_price, high_price[0], take_profit);
   lastTradeBarTime = currentBarTime;
  }
```

Explanation:

The standard trading library for MQL5 is included in the first line, #include <Trade/Trade.mqh>. The functions and classes required to conduct trades, alter orders, and maintain positions are made available to the EA by this library. The EA will use the instance of the trade object created by the line CTrade trade; to place orders in a methodical and secure manner.

int MagicNumber = 53217; defines a magic number. This is a special number that is linked to every deal that this EA opens. The EA can distinguish its trades from manual trades or deals opened by other EAs thanks to SetExpertMagicNumber (MagicNumber). To manage positions, change them, or close them later without influencing other trades, this is essential.

The input double lot\_size = 0.6; is used to define the trade size. This enables the user to modify each trade's size straight from the input settings. datetime currentBarTime = iTime(\_Symbol, time\_frame, 0); stores the time of the most recent candle, and ask\_price = SymbolInfoDouble(\_Symbol, SYMBOL\_ASK); retrieves the current market ask price. Setting the entry price, figuring out the take-profit, and preventing multiple trades on the same candle are all done with these values.

```
(high_price[1] >= td1_line_value && open_price[1] < td1_line_value)
|| (high_price[2] >= td2_line_value && open_price[2] < td2_line_value)
|| (high_price[3] >= td3_line_value && open_price[3] < td3_line_value)
|| (high_price[0] >= td_line_value)
```

This line determines if any of the most recent candles have made contact with or even marginally crossed the downward trend line. (high\_price\[1\] >= td1\_line\_value && open\_price\[1\] < td1\_line\_value, for instance) confirms that the candle's high one bar ago was above the trend line, but its opening was below it. This attests to the price's approach to the trend line and the possible reversal situation it produced. Two bars ago, three bars ago, and the current candle are all subject to the same reasoning. By using the logical OR operator, the condition is guaranteed to pass if any of these candles satisfy this requirement.

The condition that confirms a definitive bearish rejection was then added by examining both the current and previous candles. Indicating that sellers have gained strength and that the momentum is declining, it ensures that the most recent candle closes below the trend line. Moreover, the candle must close lower than it began to be deemed bearish. For the price structure to stay stable and the reversal signal to be more accurate, the previous candle also needs to have opened below the trend line.

This requirement guarantees that a bearish reversal is confirmed by the most recent candle. It verifies that the candle closed below the trend line, indicating a decline in momentum and the takeover by sellers. Bearish pressure is confirmed when the candle closes lower than it started. Furthermore, for the previous candle to have maintained a steady downward pattern, it should have opened below the trend line. When combined, these conditions assist in preventing false entry brought on by transient price swings and increase the reversal signal's validity.

By comparing the current candle's time to the last performed trade's time, the condition (currentBarTime!= lastTradeBarTime) makes it so that the EA doesn't open multiple trades on the same candle, hence permitting only one entry per bar. The EA determines the take profit once all entry requirements are met by multiplying the absolute difference between the current ask price and the confirming candle's high by four to determine a positive risk-to-reward ratio.

Using the designated lot size, symbol, entry price, stop loss at the high of the confirming candle, and calculated take profit, it then executes the sell transaction. To ensure disciplined and methodical trade execution, the EA updates the lastTradeBarTime to the time of the current candle once the transaction is placed. This prevents numerous trades on the same bar.

The EA must then be able to decide whether to trade a breakout with retest or a straightforward reversal. Although we have only addressed the downtrend reversal situation thus far, you can see how the EA determines a legitimate reversal signal by going over the conditions we just reviewed. Once the reversal logic is fully established, these identical ideas can be modified to handle a breakout-and-retest technique.

Example:

```
input string down_trend = ""; // Down Trend Line
input string up_trend = ""; // Up Trend Line
input ENUM_TIMEFRAMES time_frame = PERIOD_CURRENT; // TIME FRAME
input double lot_size = 0.6; // LOT SIZE

enum line_type
  {
   reversal = 0, //REVERSAL
   break_out = 1, //BREAK-OUT
   reverse_break = 2 // REVERSAL AND BREAK-OUT
  };
input line_type line_exe =  reversal; // MODE
```

```
if(((high_price[1] >= td1_line_value && open_price[1] < td1_line_value) || (high_price[2] >= td2_line_value && open_price[2] < td2_line_value)
    || (high_price[3] >= td3_line_value && open_price[3] < td3_line_value) || (high_price[0] >= td_line_value))
   && (close_price[0] < td_line_value && close_price[0] < open_price[0] && open_price[1] < td1_line_value)
   && (no_bars_down < 3)
   && prev_touch_down == false
   && (currentBarTime != lastTradeBarTime)
   && (line_exe == reversal || line_exe == reverse_break)
  )
  {
   take_profit = MathAbs(ask_price - ((high_price[0] - ask_price) * 4));

   trade.Sell(lot_size,_Symbol,ask_price, high_price[0], take_profit);
   lastTradeBarTime = currentBarTime;
  }
```

Output:

![Figure 6. Down Trend Reversal](https://c.mql5.com/2/176/figure_6.png)

### **Identifying Downward Trend Breakout and Retest**

The EA first watches for a candle to close above the downtrend line, indicating that the prior resistance has been broken, before executing a breakout and retest setup aimed at a buy. The EA does not, however, start trading right away. It awaits a retest in which the price contacts and returns to the broken trend line, indicating that the previous resistance is now serving as support.

Following confirmation of the retest, the EA searches for a bullish candle that follows the touch. This bullish confirmation confirms the entry point and shows that buyers are taking over. The EA initiates a buy order when these three criteria are satisfied: a breakout, a retest touch, and a confirming bullish candle. By making sure that trades are only made after the breakout is validated by price movement, this strategy helps prevent entering too soon and lowers the possibility of false breakouts.

Example:

```
td_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[0],0);
td1_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[1],0);
td2_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[2],0);
td3_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[3],0);

bool prev_touch_down = false;

if((high_price[1] > td1_line_value && close_price[1] < open_price[1])
   ||
   (high_price[2] > td2_line_value && close_price[2] < open_price[2])
  )
  {
   prev_touch_down = true;
  }
int no_bars_down = 0;
for(int i = 0; i <= 3; i++)
  {
   if(high_price[i] > ObjectGetValueByTime(chart_id,down_trend,time_price[i],0) && open_price[i] < ObjectGetValueByTime(chart_id,down_trend,time_price[i],0))
     {
      for(int j = i; j >= 0; j--)
        {
         if(close_price[j] < open_price[j] && close_price[j] < ObjectGetValueByTime(chart_id,down_trend,time_price[j],0))
           {
            no_bars_down = Bars(_Symbol,time_frame,time_price[j],TimeCurrent());
            break;
           }
        }
      break;
     }
  }

//DOWN TREND

td_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[0],0);
td1_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[1],0);
td2_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[2],0);
td3_line_value = ObjectGetValueByTime(chart_id,down_trend,time_price[3],0);

bool prev_touch_down = false;

if((high_price[1] > td1_line_value && close_price[1] < open_price[1])
   ||
   (high_price[2] > td2_line_value && close_price[2] < open_price[2])
  )
  {
   prev_touch_down = true;
  }
int no_bars_down = 0;
for(int i = 0; i <= 3; i++)
  {
   if(high_price[i] > ObjectGetValueByTime(chart_id,down_trend,time_price[i],0) && open_price[i] < ObjectGetValueByTime(chart_id,down_trend,time_price[i],0))
     {
      for(int j = i; j >= 0; j--)
        {
         if(close_price[j] < open_price[j] && close_price[j] < ObjectGetValueByTime(chart_id,down_trend,time_price[j],0))
           {
            no_bars_down = Bars(_Symbol,time_frame,time_price[j],TimeCurrent());

            break;
           }
        }
      break;
     }

ask_price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
datetime currentBarTime = iTime(_Symbol, time_frame, 0);

if(((high_price[1] >= td1_line_value && open_price[1] < td1_line_value) || (high_price[2] >= td2_line_value && open_price[2] < td2_line_value)
    || (high_price[3] >= td3_line_value && open_price[3] < td3_line_value) || (high_price[0] >= td_line_value))
   && (close_price[0] < td_line_value && close_price[0] < open_price[0] && open_price[1] < td1_line_value)
   && (no_bars_down < 3)
   && prev_touch_down == false
   && (currentBarTime != lastTradeBarTime)
   && (line_exe == reversal || line_exe == reverse_break)
  )
  {
   take_profit = MathAbs(ask_price - ((high_price[0] - ask_price) * 4));

   trade.Sell(lot_size,_Symbol,ask_price, high_price[0], take_profit);
   lastTradeBarTime = currentBarTime;

  }

// DOWNTREND BREAKOUT AMD RETEST
bool prev_touch_break_out_down = false;

if((low_price[1] < td1_line_value && close_price[1] > open_price[1]) ||
   (low_price[2] < td2_line_value && close_price[2] > open_price[2] && open_price[2] > td2_line_value))
  {
   prev_touch_break_out_down = true;
  }

int no_bars_down_breakout = 0;

for(int i = 0; i <= 3; i++)
  {

   if(low_price[i] < ObjectGetValueByTime(chart_id, down_trend, time_price[i], 0) &&
      open_price[i] > ObjectGetValueByTime(chart_id, down_trend, time_price[i], 0))
     {

      for(int j = i; j >= 0; j--)
        {
         if(close_price[j] > open_price[j] &&
            close_price[j] > ObjectGetValueByTime(chart_id, down_trend, time_price[j], 0))
           {

            no_bars_down_breakout = Bars(_Symbol, time_frame, time_price[j], TimeCurrent());
            break;
           }
        }
      break;
     }
  }

if(
   ((low_price[0] < td_line_value && open_price[0] > td_line_value) ||
    (low_price[1] < td1_line_value && open_price[1] > td1_line_value) ||
    (low_price[2] < td2_line_value && open_price[2] > td2_line_value) ||
    (low_price[3] < td3_line_value && open_price[3] > td3_line_value)) &&
   (close_price[0] > open_price[0]) && close_price[0] > td_line_value &&
   (no_bars_down_breakout < 3) &&
   (prev_touch_break_out_down == false) &&
   (currentBarTime != lastTradeBarTime)
   && (line_exe == break_out || line_exe == reverse_break)
)
  {
   take_profit = MathAbs(ask_price + ((ask_price - low_price[0]) * 4));

   trade.Buy(lot_size, _Symbol, ask_price, low_price[0], take_profit);

   lastTradeBarTime = currentBarTime;
  }
```

Explanation:

A boolean flag is defined in the initial section of the code to monitor if a recent breakout-and-retest interaction on the downtrend has already taken place. By using this flag, the EA can prevent the same setup from generating additional trades. The next requirement determines whether a breakout scenario has already been started by looking at the last few candles.

Additionally, it checks to see if the candle opened above the trend line to confirm that the price has broken through and retested it. To stop the EA from opening several trades too frequently, a counter counts the number of bars since the last breakout, and a flag is set to signal that a breakout touch has already occurred.

The breakout point is then determined by iterating through the most recent candles in the main loop. It starts by searching for a candle in which the open was above the trend line and the low was below it. This indicates the trend line's first penetration, which is the precursor to a breakout. Following the discovery of such a candle, a nested loop looks backward to locate a subsequent candle that closes above the trend line and is bullish. This demonstrates that the trend line has effectively moved from resistance to support and that the breakout is legitimate. Both loops are stopped to avoid needless additional checks once this confirmation is obtained, and the counter is adjusted to reflect the number of bars since the breakout.

This breakout-and-retest reasoning differs significantly from a reversal. The EA searches for a bearish candle that touches the trend line from below to indicate a rejection during a reversal. The breakout-and-retest strategy involves the EA waiting for a candle to break above the downward trend line, followed by a retest and a bullish candle that confirms the trend. The breakout setup is appropriate for trades that follow a verified shift in market direction because the emphasis is on gaining momentum after the trend line is breached rather than catching a price rejection.

```
if(
   ((low_price[0] < td_line_value && open_price[0] > td_line_value) ||
    (low_price[1] < td1_line_value && open_price[1] > td1_line_value) ||
    (low_price[2] < td2_line_value && open_price[2] > td2_line_value) ||
    (low_price[3] < td3_line_value && open_price[3] > td3_line_value)) &&
   (close_price[0] > open_price[0]) && close_price[0] > td_line_value &&
   (no_bars_down_breakout < 3) &&
   (prev_touch_break_out_down == false) &&
   (currentBarTime != lastTradeBarTime)
   && (line_exe == break_out || line_exe == reverse_break)
)
  {
   take_profit = MathAbs(ask_price + ((ask_price - low_price[0]) * 4));

   trade.Buy(lot_size, _Symbol, ask_price, low_price[0], take_profit);

   lastTradeBarTime = currentBarTime;
  }
```

The condition's first section determines if the downtrend line has seen a breakout. It looks at the most recent candles to determine if any of them had an open above the trend line and a low that fell below it. This suggests a possible breakthrough since the candle broke through the trend line from above. The mechanism allows for delayed confirmations by guaranteeing that the EA detects breakout touches on the preceding three candles in addition to the current candle.

The condition (close\_price\[0\] > open\_price\[0\]) comes next. The current candle is bullish and closed above the trend line if && close\_price\[0\] > td\_line\_value. The fact that buyers are in charge and the trend line has effectively shifted from resistance to support serves as confirmation for the breakout.

By keeping track of the amount of bars since the last confirmed setup, it makes sure that trades are not made too near a previous breakout. This enables the EA to concentrate on new possibilities and refrain from overtrading. By verifying that no recent breakout touch has been recorded, it also avoids duplicate transactions from the same price action. Lastly, it checks to make sure a deal hasn't already been placed within the same bar, ensuring that only one trade is executed per candle.

It verifies that the EA is in a mode that allows breakout trades, either by itself or with reversals. Using a favorable risk-to-reward ratio, the EA calculates the take profit based on the distance between the current price and the low of the confirming candle once all trade requirements have been satisfied. After that, it executes a purchase transaction with the specified lot size, assigns the calculated take profit, sets the stop loss at the low of the breakout candle, and logs the trade time to avoid duplicate entries on the same candle.

Output:

![Figure 7. Breakout and Retest](https://c.mql5.com/2/176/Figure_7.png)

### **Identifying Upward Trend Reversal**

Finding reversal possibilities on an ascending trend line is the next step after learning how to recognize breakout-and-retest and reversal setups on the declining trend line. The uptrend's reasoning is basically the opposite of the downtrend's. The trend line here indicates support as opposed to opposition. When a recent candle's low touches or approaches the uptrend line and a bullish candle closes above its open, this is a legitimate reversal signal. This suggests a possible buying opportunity since it shows that the price has tested the support and buyers are beginning to gain ground.

To avoid making several trades from the same support test, the EA makes sure that no recent signals have been generated, much like in the downtrend reversal. By comparing the time of the current candle with the time of the most recent trade, it also makes sure that just one trade per candle is permitted. The EA can consistently manage risk and avoid overtrading while methodically identifying uptrend reversals by replicating the downtrend reversal logic.

Example:

```
double t_line_value;
double t1_line_value;
double t2_line_value;
double t3_line_value;
```

```
// UP TREND
t_line_value = ObjectGetValueByTime(chart_id,up_trend,time_price[0],0);
t1_line_value = ObjectGetValueByTime(chart_id,up_trend,time_price[1],0);
t2_line_value = ObjectGetValueByTime(chart_id,up_trend,time_price[2],0);
t3_line_value = ObjectGetValueByTime(chart_id,up_trend,time_price[3],0);

int no_bars_up = 0;

for(int i = 0; i <= 3; i++)
  {

   if(low_price[i] < ObjectGetValueByTime(chart_id, up_trend, time_price[i], 0) &&
      open_price[i] > ObjectGetValueByTime(chart_id, up_trend, time_price[i], 0))
     {

      for(int j = i; j >= 0; j--)
        {

         if(close_price[j] > open_price[j] &&
            close_price[j] > ObjectGetValueByTime(chart_id, up_trend, time_price[j], 0))
           {

            no_bars_up = Bars(_Symbol, time_frame, time_price[j], TimeCurrent());

            break;
           }
        }
      break;
     }
  }

bool prev_touch_up = false;

if((low_price[1] < t1_line_value && close_price[1] > open_price[1]) ||
   (low_price[2] < t2_line_value && close_price[2] > open_price[2]))
  {

   prev_touch_up = true;  // Flag that a recent touch already occurred

  }

if(
   ((low_price[0] < t_line_value && open_price[0] > t_line_value) ||
    (low_price[1] < t1_line_value && open_price[1] > t1_line_value) ||
    (low_price[2] < t2_line_value && open_price[2] > t2_line_value) ||
    (low_price[3] < t3_line_value && open_price[3] > t3_line_value))
   &&
   (close_price[0] > open_price[0]) && close_price[0] > t_line_value
   &&
   (no_bars_up < 3)
   &&
   prev_touch_up == false
   &&
   (currentBarTime != lastTradeBarTime)
   &&
   (line_exe == reversal || line_exe == reverse_break)
)
  {
   take_profit = MathAbs(ask_price + ((ask_price - low_price[0]) * 4));

   trade.Buy(lot_size, _Symbol, ask_price, low_price[0],take_profit);
   lastTradeBarTime = currentBarTime;

  }
```

Explanation:

The uptrend line's value at the time of the most recent candles is stored in variables that are initialized in the first section. The EA may assess price activity against the trend line at various times because each variable is associated with a distinct candle. This guarantees that the EA can watch the most recent four candles and is crucial for identifying interactions between price and the trend line.

The precise trend line values for the final few candles are retrieved in the following step. The EA can precisely ascertain whether a candle's low touched or penetrated the uptrend line, which is the initial need for an uptrend reversal, by obtaining the price of the line at the time of each candle.

The number of bars since the last verified reversal setup is then tracked by an initialized counter. When identifying a candle that may indicate a support test, the EA iterates over the most recent candles until the low falls below the trend line and the open is above it. To verify that buyers are intervening, a nested loop within this looks for a subsequent candle that closes bullish and above the trend line. When avoiding repeated alerts in quick succession, the counter is updated to reflect the amount of bars since the confirmation.

Next, a boolean flag is set to show if the trend line has already touched recently. The EA looks at the last several candles to determine if any of them closed bullish after dipping below the trend line. To make sure that no multiple trades are taken from the same price interaction, the flag is set to true if such a candle is present.

Lastly, before placing a buy order, the EA considers a number of factors. It verifies that the current candle is bullish and closes above the trend line, that no recent touches have already triggered a trade, that the trade has not already been executed on the current candle, that the execution mode permits reversal trades, and that one of the recent candles touched or penetrated the trend line.

The distance between the ask price and the low of the confirming candle, increased by a multiplier for a positive risk-to-reward ratio, is how the EA determines the take profit if all of these conditions are met. After that, it applies the take profit, performs a purchase transaction, changes the last trade bar time to avoid multiple trades on the same candle, and puts the stop loss at the low of the reversal candle.

### **Identifying Upward Trend Breakout and Retest**

Understanding uptrend reversals is followed by identifying breakout-and-retest possibilities on an ascending trend line. A candle falling below the trend line in this instance confirms the breakout by indicating that the support has been breached. The EA waits for a retest after this breakout, which occurs when the high of a subsequent candle crosses back to the trend line. This retest shows that the broken support is now acting as resistance. After this retest has been validated, the EA looks for a bearish candle to signal a sell entry, which would mean that sellers have taken control.

Example:

```
// UP TREND
t_line_value = ObjectGetValueByTime(chart_id,up_trend,time_price[0],0);
t1_line_value = ObjectGetValueByTime(chart_id,up_trend,time_price[1],0);
t2_line_value = ObjectGetValueByTime(chart_id,up_trend,time_price[2],0);
t3_line_value = ObjectGetValueByTime(chart_id,up_trend,time_price[3],0);

int no_bars_up = 0;

for(int i = 0; i <= 3; i++)
  {

   if(low_price[i] < ObjectGetValueByTime(chart_id, up_trend, time_price[i], 0) &&
      open_price[i] > ObjectGetValueByTime(chart_id, up_trend, time_price[i], 0))
     {

      for(int j = i; j >= 0; j--)
        {

         if(close_price[j] > open_price[j] &&
            close_price[j] > ObjectGetValueByTime(chart_id, up_trend, time_price[j], 0))
           {

            no_bars_up = Bars(_Symbol, time_frame, time_price[j], TimeCurrent());

            break;
           }
        }
      break;
     }
  }

bool prev_touch_up = false;

if((low_price[1] < t1_line_value && close_price[1] > open_price[1]) ||
   (low_price[2] < t2_line_value && close_price[2] > open_price[2]))
  {

   prev_touch_up = true;  // Flag that a recent touch already occurred

  }

if(
   ((low_price[0] < t_line_value && open_price[0] > t_line_value) ||
    (low_price[1] < t1_line_value && open_price[1] > t1_line_value) ||
    (low_price[2] < t2_line_value && open_price[2] > t2_line_value) ||
    (low_price[3] < t3_line_value && open_price[3] > t3_line_value))
   &&
   (close_price[0] > open_price[0]) && close_price[0] > t_line_value
   &&
   (no_bars_up < 3)
   &&
   prev_touch_up == false
   &&
   (currentBarTime != lastTradeBarTime)
   &&
   (line_exe == reversal || line_exe == reverse_break)
)
  {
   take_profit = MathAbs(ask_price + ((ask_price - low_price[0]) * 4));

   trade.Buy(lot_size, _Symbol, ask_price, low_price[0],take_profit);
   lastTradeBarTime = currentBarTime; // Update last trade bar time to avoid duplicate signals
  }

// UPTREND BREAKOUT AMD RETEST

bool prev_touch_break_out_up = false;

if((high_price[1] > td1_line_value && close_price[1] < open_price[1])
   ||
   (high_price[2] > td2_line_value && close_price[2] < open_price[2])
  )
  {

   prev_touch_break_out_up = true;

  }

int no_bars_up_break_out = 0;

for(int i = 0; i <= 3; i++)
  {

   if(high_price[i] > ObjectGetValueByTime(chart_id,down_trend,time_price[i],0) && open_price[i] < ObjectGetValueByTime(chart_id,down_trend,time_price[i],0)
     )
     {

      for(int j = i; j >= 0; j--)
        {

         if(close_price[j] < open_price[j] && close_price[j] < ObjectGetValueByTime(chart_id,down_trend,time_price[j],0))
           {

            no_bars_up_break_out = Bars(_Symbol,time_frame,time_price[j],TimeCurrent());

            break;

           }

        }
      break;

     }

  }

if(((high_price[1] >= t1_line_value && open_price[1] < t1_line_value) || (high_price[2] >= t2_line_value && open_price[2] < t2_line_value)
    || (high_price[3] >= t3_line_value && open_price[3] < t3_line_value) || (high_price[0] >= t_line_value))
   && (close_price[0] < t_line_value && close_price[0] < open_price[0] && open_price[1] < t1_line_value)
   && (no_bars_up_break_out < 3)
   && (no_bars_up_break_out == false)
   && (currentBarTime != lastTradeBarTime)
   && (line_exe == break_out || line_exe == reverse_break)
  )
  {

   take_profit = MathAbs(ask_price - ((high_price[0] - ask_price) * 4));
   trade.Sell(lot_size,_Symbol,ask_price,high_price[0], take_profit);
   lastTradeBarTime = currentBarTime;

  }
```

Explanation:

To determine whether a recent breakout touch has already taken place, the first section sets a boolean flag. When the candle closed bearish, the EA looked at the prior candles to see if their peak hit the trend line from below. To stop several signals from coming from the same breakout event, the flag is set to true if such a touch is detected. This guarantees that the same price interaction won't cause the EA to initiate trades again.

The EA then determines how many bars have passed since the last verified breakout. It searches through the most recent candles to identify any where the open was below the trend line and the high was above it. A nested loop within this looks for a follow-up candle that closes below the trend line in a bearish manner. To make sure that trades are not made too near a prior breakout, the EA counts the bars between this candle and the current bar after it has been spotted. To prevent overtrading and keep the frequency of entries under control, this bar count is utilized.

Multiple checks are combined to provide the primary criterion for executing a sell trade. To validate the retest, it first confirms that one of the most recent candles has touched or crossed above the trend line. Next, it determines whether the current candle falls below the trend line and is bearish, indicating that sellers are gaining ground. By comparing the current bar time with the last executed transaction, it also makes sure that there aren't too many bars since the last breakout to permit a new trade, that no recent breakout touch has already triggered a trade, and that only one trade is executed each candle. Lastly, it attests that breakout trades are permitted in the EA mode.

When all of these requirements are met, the EA determines the take profit by multiplying the distance between the current ask price and the high of the confirming candle by a factor to produce a good risk-to-reward ratio. After that, it applies the calculated take profit, puts the stop loss at the high of the breakout candle, executes a sell order with the designated lot size, and updates the last trade bar time. By eliminating duplicate or premature trades, this logic guarantees that the EA consistently captures uptrend breakout-and-retest possibilities.

### **Conclusion**

Continuing from the last study on chart objects, we concentrated on trend lines rather than rectangles. When the price reverses from or breaks through trend line objects on the chart, you learn how to create an EA that can identify them and automatically execute trades. By combining automatic trading with manual trend line analysis, this approach transforms a popular technical tool into useful trading signals in MQL5.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19968.zip "Download all attachments in the single ZIP archive")

[Project\_17\_Trend\_Line\_Object\_EA.mq5](https://www.mql5.com/en/articles/download/19968/Project_17_Trend_Line_Object_EA.mq5 "Download Project_17_Trend_Line_Object_EA.mq5")(10.45 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/498239)**
(1)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
22 Oct 2025 at 16:20

More flexible and powerful solution of this kind has been presented in the article [TradeObjects: Automation of trading based on MetaTrader graphical objects](https://www.mql5.com/en/articles/3442).


![Market Simulation (Part 04): Creating the C_Orders Class (I)](https://c.mql5.com/2/112/Simulat6o_de_mercado_Parte_01_Cross_Order_I_LOGO.png)[Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)

In this article, we will start creating the C\_Orders class to be able to send orders to the trading server. We'll do this little by little, as our goal is to explain in detail how this will happen through the messaging system.

![Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (FinAgent)](https://c.mql5.com/2/108/Neural_Networks_in_Trading_Multimodal_Agent_Augmented_with_Instruments____LOGO.png)[Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (FinAgent)](https://www.mql5.com/en/articles/16850)

We invite you to explore FinAgent, a multimodal financial trading agent framework designed to analyze various types of data reflecting market dynamics and historical trading patterns.

![Mastering Quick Trades: Overcoming Execution Paralysis](https://c.mql5.com/2/176/19576-mastering-quick-trades-overcoming-logo.png)[Mastering Quick Trades: Overcoming Execution Paralysis](https://www.mql5.com/en/articles/19576)

The UT BOT ATR Trailing Indicator is a personal and customizable indicator that is very effective for traders who like to make quick decisions and make money from differences in price referred to as short-term trading (scalpers) and also proves to be vital and very effective for long-term traders (positional traders).

![Dynamic Swing Architecture: Market Structure Recognition from Swings to Automated Execution](https://c.mql5.com/2/176/19793-dynamic-swing-architecture-logo.png)[Dynamic Swing Architecture: Market Structure Recognition from Swings to Automated Execution](https://www.mql5.com/en/articles/19793)

This article introduces a fully automated MQL5 system designed to identify and trade market swings with precision. Unlike traditional fixed-bar swing indicators, this system adapts dynamically to evolving price structure—detecting swing highs and swing lows in real time to capture directional opportunities as they form.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ckygldquyfytfnrnqgifpojqcnvnovqa&ssn=1769157175554519035&ssn_dr=0&ssn_sr=0&fv_date=1769157175&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19968&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%2025)%3A%20Building%20an%20EA%20that%20Trades%20with%20Chart%20Objects%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915717538075664&fz_uniq=5062545602174166070&sv=2552)

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