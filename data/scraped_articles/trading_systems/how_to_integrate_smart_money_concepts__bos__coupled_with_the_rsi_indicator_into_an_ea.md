---
title: How to Integrate Smart Money Concepts (BOS) Coupled with the RSI Indicator into an EA
url: https://www.mql5.com/en/articles/15030
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:41:12.886890
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=timblhcokamcvxduoaxlmlhgmtgjgzbe&ssn=1769157671274193271&ssn_dr=0&ssn_sr=0&fv_date=1769157671&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15030&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Integrate%20Smart%20Money%20Concepts%20(BOS)%20Coupled%20with%20the%20RSI%20Indicator%20into%20an%20EA%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691576717658283&fz_uniq=5062643471593940519&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the fast-paced world of foreign exchange trading, having a reliable and efficient trading system is crucial for success. Trading in general has plenty of terms, concepts, and strategies. Often at times it can be overwhelming, especially for new traders who are still trying to find their footing in the trading industry.  Smart Money Concept(SMC) is one of the trending concepts in forex trading, but it can be difficult at times for new traders or for anyone to use smart money concepts in their trading.

To solve this problem, there should be a robust tool for automating trading decisions based on market structure and price action. The solution is to couple the Smart Money Concept (Break Of Structure) with the popular Relative Strength Index(RSI) indicator. This combination provides a strategic edge by leveraging both price action and momentum analysis, which enhances the accuracy of trade entries and exits, aiming to optimize trading performance.

### The idea of the Expert Advisor example

The idea and functionality of this Expert Advisor example is that the expert will be detecting the swing lows and the swing highs as some smart money concepts utilize the swings. The RSI indicator will still be using the traditional conversion of the 70 level for overbought market and the 30 level for oversold market and the period will be 8. When the market price is above the previously detected high, that will indicate the break of structure on the upside. Similarly, when the market price is below the previously detected low, that will indicate break of structure on the downside.

### Now let's develop the Expert Advisor example

The EA aims to open a buy and a sell orders based on market conditions and RSI levels. Specifically, it:

1. Identifies swing highs and swing lows in the market.
2. Checks if the market is above the previous swing high (for a sell signal) or below the previous swing low (for a buy signals).
3. Then Confirms the signal with RSI levels.

With that being said, basically the Expert Advisor will search for break of structure or break of previous swings (high/low) and then if the RSI value is within the specified settings then a market order will be executed ( **_BUY/SELL_**).

### Code Breakdown

### **1\. Properties and Includes**

```
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Trade/Trade.mqh>
```

These lines define the EA's properties and include the trade library necessary for executing trades.

**2\. Global Variables and Inputs**

```
long MagicNumber = 76543;
double lotsize = 0.01;
input int RSIperiod = 8;
input int RSIlevels = 70;
int stopLoss = 200;
int takeProfit = 500;
bool closeSignal = false;
```

- MagicNumber: A unique identifier for the EA's trades.
- Lotsize: The size of each trade.
- RSIperiod: Period for the RSI calculations.
- RSIlevels: Threshold level for the RSI.
- Stoploss and takeProfit: SL and TP levels in points.
- CloseSignal: Flag for closing positions based on the opposite signal compared to the position which is currently open.


**3\. RSI Variables**

```
int handle;
double buffer[];
MqlTick currentTick;
CTrade trade;
datetime openTimeBuy = 0;
datetime openTimeSell = 0;
```

- Handle: Handle for the RSI indicator.

- Buffer: Array to store the RSI values.
- currentTick: Structure to store current market prices.
- Trade: This is the object for trade operations.
- openTimeBuy and openTimeSell: The timestamps for the last buy and sell signals and operations.

**4\. Initialization Function**

```
int OnInit() {
    if (RSIperiod <= 1) {
        Alert("RSI period <= 1");
        return INIT_PARAMETERS_INCORRECT;
    }
    if (RSIlevels >= 100 || RSIlevels <= 50) {
        Alert("RSI level >= 100 or <= 50");
        return INIT_PARAMETERS_INCORRECT;
    }
    trade.SetExpertMagicNumber(MagicNumber);
    handle = iRSI(_Symbol, PERIOD_CURRENT, RSIperiod, PRICE_CLOSE);
    if (handle == INVALID_HANDLE) {
        Alert("Failed to create indicator handle");
        return INIT_FAILED;
    }
    ArraySetAsSeries(buffer, true);
    return INIT_SUCCEEDED;
}
```

- Validates RSI period and levels.
- Sets the magic number for trade identification.
- Creates an RSI indicator handle.
- Sets the buffer as a series to store RSI values in reverse chronological order.


**5\. Deinitialization Function**

```
void OnDeinit(const int reason) {
    if (handle != INVALID_HANDLE) {
        IndicatorRelease(handle);
    }
}
```

Releases the RSI indicator handle when the EA is removed.

**6\. OnTick Function**

As we all know, the OnTick function contains the core logic for detecting signals and executing trades.

```
void OnTick() {
    static bool isNewBar = false;
    int newbar = iBars(_Symbol, _Period);
    static int prevBars = newbar;
    if (prevBars == newbar) {
        isNewBar = false;
    } else if (prevBars != newbar) {
        isNewBar = true;
        prevBars = newbar;
    }
```

Firstly, since the OnTick function runs on every Tick, we need to make sure that when we do detect a signal or execute a trade we execute it once per bar. We accomplish that by declaring a static boolean variable _isNewBar._ Which we set to false initially, and then declare an int variable _newBar_ which is assigned the function iBars so that we can keep track of every candle bar.

- static bool isNewBar: Keeps track of whether a new bar (candlestick) has formed.
- int newbar = iBars(\_Symbol, \_Period): Gets the current number of bars on the chart.
- static int prevBars = newbar: Initializes the previous bar count.
- The if-else block checks if the bar count has changed, indicating a new bar. If a new bar has formed, 'isNewBar' is set to 'true' otherwise, it's 'false'.

```
    const int length = 10;
    int right_index, left_index;
    int curr_bar = length;
    bool isSwingHigh = true, isSwingLow = true;
    static double swing_H = -1.0, swing_L = -1.0;
```

We then need to set the variables to detect the swing highs and swing lows:

- const int length = 10: Defines the range for detecting the swing highs and the swing lows.
- int right\_index, left\_index: Indices for bars to the right and bars to the left of the current bar.
- int curr\_bar = length: Set the current bar index.
- bool isSwingHigh = true, isSwingLow = true: These are flags to determine if a bar is a swing high or a swing low.
- static double swing\_H = -1.0, swing\_L = -1.0: Stores the latest detected swing high and the latest detected swing low values.

```
    double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits);
    double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits);
```

The variables are of type double and the variable 'Ask' is used to get the current market ASK price, and the variable 'Bid' is used to get the current market BID price:

- double Ask: Gets the current ask price
- double Bid: Gets the current bid price
- NormalizeDouble: Rounds the price to the correct number of decimal places.

```
    if (isNewBar) {
        for (int a = 1; a <= length; a++) {
            right_index = curr_bar - a;
            left_index = curr_bar + a;

            if ((high(curr_bar) <= high(right_index)) || (high(curr_bar) < high(left_index))) {
                isSwingHigh = false;
            }
            if ((low(curr_bar) >= low(right_index)) || (low(curr_bar) > low(left_index))) {
                isSwingLow = false;
            }
        }

        if (isSwingHigh) {
            swing_H = high(curr_bar);
            Print("We do have a swing high at: ", curr_bar, " H: ", high(curr_bar));
            drawswing(TimeToString(time(curr_bar)), time(curr_bar), high(curr_bar), 32, clrBlue, -1);
        }
        if (isSwingLow) {
            swing_L = low(curr_bar);
            Print("We do have a swing low at: ", curr_bar, " L: ", low(curr_bar));
            drawswing(TimeToString(time(curr_bar)), time(curr_bar), low(curr_bar), 32, clrRed, +1);
        }
    }
```

We then proceed to search for swing highs and swings lows, but before that, if a new bar is detected, the function checks the surrounding bars within the defined ' **_length_**' to identify if there are swing highs or swing lows. The variable **_length_** is to define a range of candles to search for swings, for each bar within the range, the function compares the current bar's high/low with adjacent bars.

If the current bar's high is not higher than the surrounding bars, then it is not a swing high. If the current bar's low is not lower than the surrounding bars, it is not a swing low. Given that all those conditions are met then a swing high or a swing low is confirmed, then the respective value is stored, and then a marker is drawn on the chart using the ' **_drawswing_**' function.

```
    int values = CopyBuffer(handle, 0, 0, 2, buffer);
    if (values != 2) {
        Print("Failed to get indicator values");
        return;
    }

    Comment("Buffer[0]: ", buffer[0],
            "\nBuffer[1]: ", buffer[1]);
```

- copyBuffer(handle, 0, 0, 2, buffer): Copies the latest RSI values into the buffer.
- If the function fails to retrieve the RSI values, it exits.
- The RSI values will be displayed as a comment on the chart.

```
    int cntBuy = 0, cntSell = 0;
    if (!countOpenPositions(cntBuy, cntSell)) {
        return;
    }
```

' _**countOpenPositions(cntBuy, cntSell)**_' this function counts the number of open buy and sell positions, if the function fails, ' **_OnTick_**' exits.

```
    if (swing_H > 0 && Ask > swing_H && buffer[0] >= 70) {
        Print("Sell Signal: Market is above previous high and RSI >= 70");
        int swing_H_index = 0;
        for (int i = 0; i <= length * 2 + 1000; i++) {
            if (high(i) == swing_H) {
                swing_H_index = i;
                break;
            }
        }
        drawBreakLevels(TimeToString(time(0)), time(swing_H_index), high(swing_H_index), time(0), high(swing_H_index), clrBlue, -1);

        if (cntSell == 0) {
            double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits);
            double sl = Bid + stopLoss * _Point;
            double tp = Bid - takeProfit * _Point;
            trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, lotsize, currentTick.bid, sl, tp, "RSI EA");
        }

        swing_H = -1.0;
        return;
    }
```

As I have explained previously, we will be checking for a sell signal. The logic is that the ask price must be above the previous swing high. Additionally, the RSI value must be somewhat greater or equals to the 70 level. If the conditions are met, it marks the swing high and draws a break level on the chart. If there are no open sell positions, it then proceeds to open a sell position, this sell trade will be opened with the calculated stop loss and take profit level, and then the swing high value will reset.

```
    if (swing_L > 0 && Bid < swing_L && buffer[0] <= 30) {
        Print("Buy Signal: Market is below previous low and RSI <= 30");
        int swing_L_index = 0;
        for (int i = 0; i <= length * 2 + 1000; i++) {
            if (low(i) == swing_L) {
                swing_L_index = i;
                break;
            }
        }
        drawBreakLevels(TimeToString(time(0)), time(swing_L_index), low(swing_L_index), time(0), low(swing_L_index), clrRed, +1);

        if (cntBuy == 0) {
            double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits);
            double sl = Ask - stopLoss * _Point;
            double tp = Ask + takeProfit * _Point;
            trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, lotsize, currentTick.ask, sl, tp, "RSI EA");
        }

        swing_L = -1.0;
        return;
    }
}
```

For the buy signal the same opposite logic applies, if the bid price is below the previous swing low, and the RSI value is less than or equal to the 30 level, then technically the buy logic is in order and the condition is met. When the condition is met it marks the swing low and then draws a break level on the chart.  And if there are no open buy positions, it will proceed to open a new buy trade with the calculated stop loss and calculated take profit levels, and the lastly it will reset the swing low value.

What we are trying to accomplish is the following:

_\_ A sell Trade with all the conditions met:_

![sell trade](https://c.mql5.com/2/115/2ndSell__1.png)

_\_ A buy Trade with all the conditions met:_

![buy trade](https://c.mql5.com/2/115/2ndbuy__1.png)

### The summary of the OnTick function:

_The Expert performs the following key actions on each market tick_

1. **Detects New Bars**: It first checks if we have a new candle bar which could have formed since the last tick, and also it keeps track of all the candles.
2. **Identifies Swing Highs and Swing Lows**: It then identifies swing points in the market to use as reference levels for later when the market price is above swing High or the market is below swing low.
3. **Retrieves RSI Values**: Obtains the latest RSI values for signal confirmation.
4. **Counts Open Positions**: Keeps track of the current open Buy and Sell positions.
5. **Generates Trading Signals**: Uses the swing points which are the swing high and the swing low and the RSI levels to generate Buy or Sell signals.
6. **Executes Trades**: Opens new positions based on the generated signals if the conditions are all met.

_Custom Functions for High, Low, and Time Values_

```
double high(int index){
       return (iHigh(_Symbol, _Period, index));
}

double low(int index){
       return (iLow(_Symbol, _Period, index));
}

datetime time(int index){
       return (iTime(_Symbol, _Period, index));
}
```

The function _high (int index)_ returns the high price of the bar at the specified _index_. It takes _index_ as a parameter of type _'int'._ The built-in MQL5 function _iHigh(\_symbol, \_Period, index)_ is used to get the high price of the bar at _index_ for the current symbol and period. Then followed by _low (int index)_ this function returns the low price of the bar at the specified _index_, and it also takes _index_ as a parameter which is of type **_int_**, and then we have iLow(\_symbol, \_Period, index). It is also MQL5 built-in function which gets the low price for the bar at _index_ for the current symbol. Finally, we have _time (int index)_ this function returns the time of the bar at the specified index, _iTime(\_symbol, \_Period, index)_ is also MQL5 built-in function to get the time of the bar at _index_ for the current symbol and period iTime's datatype is of type _datetime_.

_The Function to Draw Swing Points_

```
void drawswing(string objName, datetime time, double price, int arrCode, color clr, int direction){
   if(ObjectFind(0, objName) < 0){
      ObjectCreate(0, objName, OBJ_ARROW, 0, time, price);
      ObjectSetInteger(0, objName, OBJPROP_ARROWCODE, arrCode);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, 10);

      if(direction > 0){ObjectSetInteger(0, objName, OBJPROP_ANCHOR, ANCHOR_TOP);}
      if(direction < 0){ObjectSetInteger(0, objName, OBJPROP_ANCHOR, ANCHOR_BOTTOM);}

      string text = "";
      string Descr = objName + text;
      ObjectCreate(0, Descr, OBJ_TEXT, 0, time, price);
      ObjectSetInteger(0, Descr, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, Descr, OBJPROP_FONTSIZE, 10);

      if(direction > 0){
         ObjectSetString(0, Descr, OBJPROP_TEXT,"  "+text);
         ObjectSetInteger(0, Descr, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
      }
      if(direction < 0){
         ObjectSetString(0, Descr, OBJPROP_TEXT,"  "+text);
         ObjectSetInteger(0, Descr, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER);
      }
   }
   ChartRedraw(0);
}
```

_This function creates visual markers for the swing points (highs and lows) on the chart. It takes the following parameters:_

- _ObjName_: Is the name of the object to be created.
- _Time_: The time at which the swing happened or has been detected.
- _Price_: The price at which the swing formed.
- _ArrCode_: Arrow code for the visual representation.
- _Clr_: Color for the arrow.
- _Direction_: Direction of the swing (positive for high, negative for low).

_Functionality_

1\. Object Creation:

- _ObjectFind(0, objName) < 0_: Checks if an object with the given name already exists.
- _ObjectCreate(0, objName, OBJ-ARROW, 0, time, price)_: Creates an arrow object at the specified time and price.
- _ObjectSetInteger(0, objName, OBJPROP-ARROWCODE, arrCode)_: Sets the arrow code.
- _ObjectSetInteger(0, objName, OBJPROP-COLOR, clr)_: Sets the arrow color.
- _ObjectSetInteger(0, objName, OBJPROP-FONTSIZE, 10)_: Sets the font size.

2\. Direction Handling:

- Sets the anchor position based on the direction.
- _OBJPROP-ANCHOR_: Sets the position of the anchor point for the arrow.

3\. Text Object Creation:

- Creates a text object associated with the arrow to display additional information.
- Sets color, font size, and anchor point for the text based on the direction.

4\. Chart Update:

- _ChartRedraw_ **_(0)_**: Redraws the chart to reflect the changes


_Function to Draw Break Levels_

```
void drawBreakLevels(string objName, datetime time1, double price1, datetime time2, double price2, color clr, int direction){
   if(ObjectFind(0, objName) < 0){
         ObjectCreate(0, objName, OBJ_ARROWED_LINE, 0, time1, price1, time2, price2);
         ObjectSetInteger(0, objName, OBJPROP_TIME, 0, time1);
         ObjectSetDouble(0, objName, OBJPROP_PRICE, 0, price1);
         ObjectSetInteger(0, objName, OBJPROP_TIME, 1, time2);
         ObjectSetDouble(0, objName, OBJPROP_PRICE, 1, price2);
         ObjectSetInteger(0, objName, OBJPROP_COLOR, clr);
         ObjectSetInteger(0, objName, OBJPROP_WIDTH, 2);

         string text = "Break";
         string Descr = objName + text;
         ObjectCreate(0, Descr, OBJ_TEXT, 0, time2, price2);
         ObjectSetInteger(0, Descr, OBJPROP_COLOR, clr);
         ObjectSetInteger(0, Descr, OBJPROP_FONTSIZE, 10);

         if(direction > 0){
            ObjectSetString(0, Descr, OBJPROP_TEXT,text+"  ");
            ObjectSetInteger(0, Descr, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
         }
         if(direction < 0){
            ObjectSetString(0, Descr, OBJPROP_TEXT,text+"  ");
            ObjectSetInteger(0, Descr, OBJPROP_ANCHOR, ANCHOR_RIGHT_LOWER);
         }
   }
   ChartRedraw(0);
}
```

This Function creates the visual representations for the break of price levels which are the swing points which would have been previously detected on the chart, it has the following parameters:

- _objName_: Name of the object to be created.
- _time1, time2_: The start time is the time at which the swing formed, and the end time is the time when break occurs.
- _price1, price2_: Start price is the price of the swing high or swing low occurred and price2 is the price at which the break of swing high or swing low happens.
- _Clr_: Color of the arrow.
- _Direction_: Direction for anchoring text.

### _Functionality_

1\. Object Creation:

- _ObjectFind(0, objName) < 0_: Checks if an object with a given name already exists.
- _ObjectCreate(0, objName, OBJ\_ARROWED\_LINE, 0, time1, price1, time2, price2)_: Creates the arrow object at the specified times and prices.
- Sets the times and prices for the start and end points of arrow line.
- _ObjectSetInteger(0, objName, OBJPROP-COLOR, clr)_: Sets the line color.
- _ObjectSetInteger(0, objName, OBJPROP-WIDTH, 2)_: Sets the line width.

2\. Text Object Creation:

- Creates a text object associated with the line to display additional information.
- Sets color, font size, and anchor point for the text based on the direction.

3\. Chart Update:

- _chartRedraw_ _(0)_: Redraws the chart to reflect the changes.

### Conclusion

In summary, the custom functions enhance the trading experience since it provides convenient access, there is simplified access to high, low, and time values of the bars. By coupling the RSI indicator with the SMC concepts, we can visualize the indicators and confirm that the EA follows the instructions. When the function draws the arrow line on the chart, it marks significant points, which are the swing points (Highs and Lows). Additionally, we can observe the break levels. We also have the Dynamic update, which ensures that the chart is up-to-date in real time with the latest markers and indicators, aiding in visual analysis and decision-making.

Through this comprehensive guide, a thorough understanding of how to integrate the SMC concept with the RSI indicator to any Expert Advisor's structure and functionality has been provided. By following the step-by-step explanation, readers should now have a clear grasp of the SMC\_RSI Expert Advisor operates, from initializing variables to executing trades based on calculated signals. Whether you are a seasoned trader or a beginner, this comprehensive insight should equip you with the necessary knowledge to utilize and customize this powerful tools effectively on your trading endeavors.

Here are the back test results below, I can say the EA itself still need some sort of optimization to obtain higher profit factor, now below are the back test results:

![backtest](https://c.mql5.com/2/115/thg__1.png)

And then below we can see the visual representation of the equity curve, I have only tested for 12 months so who can say precisely how it would perform if maybe tested for possibly a period of 12 years.

![equity curve](https://c.mql5.com/2/115/eer__1.png)

### References

Original article: [https://www.mql5.com/en/articles/15017](https://www.mql5.com/en/articles/15017)

YouTube

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15030.zip "Download all attachments in the single ZIP archive")

[SMC\_RSI.mq5](https://www.mql5.com/en/articles/download/15030/smc_rsi.mq5 "Download SMC_RSI.mq5")(12.13 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)
- [Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)
- [Black-Scholes Greeks: Gamma and Delta](https://www.mql5.com/en/articles/20054)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/469769)**
(3)


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
22 Jul 2024 at 23:04

For copyright issues this article copies directly the code in parts from the original article found at [https://www.mql5.com/en/articles/15017](https://www.mql5.com/en/articles/15017) published by [@Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372) which is originally owned by the author. You should have reserved the courtesy of tagging the original author of the content implemented, which is also found specifically here in [YouTube](https://www.youtube.com/watch?v=ZrkrFhY-fxw&t=586s "https://www.youtube.com/watch?v=ZrkrFhY-fxw&t=586s") video.

For example:

Original content:

![ORIGINAL 1](https://c.mql5.com/3/440/Screenshot_2024-07-22_235009.png)

Copyrighted content:

![COPYRIGHT 1](https://c.mql5.com/3/440/Screenshot_2024-07-22_235040.png)

At least you could try to implement your own logic other than just copy and paste not only the logic but also the exact variables used. Here is an example:

Original content:

![ORIGINAL 2](https://c.mql5.com/3/440/Screenshot_2024-07-22_235708.png)

Copyrighted content:

![COPYRIGHT 2](https://c.mql5.com/3/440/Screenshot_2024-07-22_235821.png)

Clearly some credibility is due to [@Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)

![Hlomohang John Borotho](https://c.mql5.com/avatar/2023/9/6505ca3e-1abb.jpg)

**[Hlomohang John Borotho](https://www.mql5.com/en/users/johnhlomohang)**
\|
30 Jul 2024 at 15:31

**Allan Munene Mutiiria [#](https://www.mql5.com/en/forum/469769#comment_54065418):**

For copyright issues this article copies directly the code in parts from the original article found at [https://www.mql5.com/en/articles/15017](https://www.mql5.com/en/articles/15017) published by [@Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372) which is originally owned by the author. You should have reserved the courtesy of tagging the original author of the content implemented, which is also found specifically here in [YouTube](https://www.youtube.com/watch?v=ZrkrFhY-fxw&t=586s "https://www.youtube.com/watch?v=ZrkrFhY-fxw&t=586s") video.

For example:

Original content:

Copyrighted content:

At least you could try to implement your own logic other than just copy and paste not only the logic but also the exact variables used. Here is an example:

Original content:

Copyrighted content:

Clearly some credibility is due to [@Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)

I apologize to you sir

I will tag you under references, originally I got the motivation from YouTube video by forex algo trader, I sincerely didn't know the existence of the original article.

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
7 Aug 2024 at 17:33

**Hlomohang John Borotho [#](https://www.mql5.com/en/forum/469769#comment_54143006):**

I apologize to you sir

I will tag you under references, originally I got the motivation from YouTube video by forex algo trader, I sincerely didn't know the existence of the original article.

No problem.

![Eigenvectors and eigenvalues: Exploratory data analysis in MetaTrader 5](https://c.mql5.com/2/83/Eigenvectors_and_eigenvalues__Exploratory_data_analysis_in_MetaTrader___LOGO.png)[Eigenvectors and eigenvalues: Exploratory data analysis in MetaTrader 5](https://www.mql5.com/en/articles/15229)

In this article we explore different ways in which the eigenvectors and eigenvalues can be applied in exploratory data analysis to reveal unique relationships in data.

![Creating a Daily Drawdown Limiter EA in MQL5](https://c.mql5.com/2/83/Creating_a_Daily_Drawdown_Limiter_EA_in_MQL5___LOGO.png)[Creating a Daily Drawdown Limiter EA in MQL5](https://www.mql5.com/en/articles/15199)

The article discusses, from a detailed perspective, how to implement the creation of an Expert Advisor (EA) based on the trading algorithm. This helps to automate the system in the MQL5 and take control of the Daily Drawdown.

![Using PatchTST Machine Learning Algorithm for Predicting Next 24 Hours of Price Action](https://c.mql5.com/2/83/Using_PatchTST_Machine_Learning_Algorithm_for_Predicting_Next_24_Hours_of_Price_Action__LOGO.png)[Using PatchTST Machine Learning Algorithm for Predicting Next 24 Hours of Price Action](https://www.mql5.com/en/articles/15198)

In this article, we apply a relatively complex neural network algorithm released in 2023 called PatchTST for predicting the price action for the next 24 hours. We will use the official repository, make slight modifications, train a model for EURUSD, and apply it to making future predictions both in Python and MQL5.

![Neural networks made easy (Part 79): Feature Aggregated Queries (FAQ) in the context of state](https://c.mql5.com/2/71/Neural_networks_are_easy_Part_79____LOGO__2.png)[Neural networks made easy (Part 79): Feature Aggregated Queries (FAQ) in the context of state](https://www.mql5.com/en/articles/14394)

In the previous article, we got acquainted with one of the methods for detecting objects in an image. However, processing a static image is somewhat different from working with dynamic time series, such as the dynamics of the prices we analyze. In this article, we will consider the method of detecting objects in video, which is somewhat closer to the problem we are solving.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/15030&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062643471593940519)

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