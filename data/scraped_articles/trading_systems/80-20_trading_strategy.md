---
title: 80-20 trading strategy
url: https://www.mql5.com/en/articles/2785
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:44:36.347713
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/2785&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070525050114742129)

MetaTrader 5 / Trading systems


1. [Introduction](https://www.mql5.com/en/articles/2785#para1)
2. ['80-20' trading system](https://www.mql5.com/en/articles/2785#para2)
3. [Signal module](https://www.mql5.com/en/articles/2785#para3)
4. [Indicator for manual trading](https://www.mql5.com/en/articles/2785#para4)
5. [Expert Advisor for testing the '80-20' trading strategy](https://www.mql5.com/en/articles/2785#para5)
6. [Strategy backtesting](https://www.mql5.com/en/articles/2785#para6)
7. [Conclusion](https://www.mql5.com/en/articles/2785#para7)

### Introduction

'80-20' is a name of one of the trading strategies (TS) described in the book [Street Smarts: High Probability Short-Term Trading Strategies](https://www.mql5.com/go?link=https://www.amazon.co.uk/Street-Smarts-Probability-Trading-Strategies/dp/0965046109 "https://www.amazon.co.uk/Street-Smarts-Probability-Trading-Strategies/dp/0965046109") by Linda Raschke and Laurence Connors. Similar to the strategies discussed in my [previous article](https://www.mql5.com/en/articles/2717), the authors attribute it to the stage when the price tests the range borders. It is also focused on profiting from false breakouts and roll-backs from the borders. But this time, we analyze the price movement on a significantly shorter history interval involving the previous day only. The lifetime of an obtained signal is also relatively short, since the system is meant for intraday trading.

The first objective of the article is to describe the development of the '80-20' trading strategy signal module using MQL5 language. Then, we are going to connect this module to the slightly edited version of the basic trading robot developed in the previous article of the series. Besides, we are going to use the very same module for the development of an indicator for manual trading.

As already said, the code provided in the article series is aimed mainly at slightly advanced novice programmers. Therefore, besides its main objective, the code is designed to help move from the procedural programming to the object-oriented one. The code will not feature classes. Instead, it will fully implement structures that are easier to master.

Yet another objective of the article is to develop tools allowing us to check if the strategy is still viable today, since Raschke and Connors used the market behavior at the end of the last century when creating it. A few EA tests based on the up-to-date history data are presented at the end of the article.

### '80-20' trading system

The authors name George Taylor's [The Taylor Trading Technique](https://www.mql5.com/go?link=https://www.amazon.com/Taylor-Trading-Technique-George-Douglass/dp/0934380244 "https://www.amazon.com/Taylor-Trading-Technique-George-Douglass/dp/0934380244"), as well as Steve Moore's works on the computer analysis of futures markets and Derek Gipson's trading experience as theoretical basis for their own work. The essence of the trading strategy can be briefly described as follows: if the previous day's Open and Close prices are located at the opposite daily range areas, then the probability of a reversal towards the previous day's opening is very high today. The previous day's Open and Close prices should locate close to the range borders. The reversal should start the current day (not before the previous day's candle is closed). The strategy rules for buying are as follows:

1\. Make sure that the market opened in the upper 20% and closed in the lower 20% of the daily range yesterday

2\. Wait till today's Low breaks the previous day's one at least by 5 ticks

3\. Place a buy pending order on the lower border of the yesterday's range

4\. Once the pending order triggers, set its initial StopLoss at the day's Low

5\. Use trailing stop to protect the obtained profit

Sell entry rules are similar, but the yesterday's bar should be bullish, a buy order should be located at the upper border of the bar, while StopLoss should be placed at the today's High.

Yet another important detail is a size of a closed daily bar. According to Linda Raschke, it should be large enough - more than the average size of daily bars. However, she does not specify how many history days should be taken into consideration when calculating the average daily range.

We should also keep in mind that the TS is designed exclusively for intraday trading — examples shown in the book use M15 charts.

The signal block and the indicator making a layout according to the strategy are described below. You can also see a few screenshots with the indicator operation results. They clearly illustrate patterns corresponding to the system rules and trading levels linked to the patterns.

M5 timeframe:

![80-20 TS pattern](https://c.mql5.com/2/25/indicator_1__1.png)

The pattern analysis should result in placing a buy pending order. Appropriate trading levels are better seen on M1 timeframe:

![80-20 TS pattern: trading levels](https://c.mql5.com/2/25/indicator_2__1.png)

A similar pattern with the opposite trading direction on M5 timeframe:

![80-20 TS pattern](https://c.mql5.com/2/25/indicator_3.png)

Its trading levels (M1 timeframe):

![80-20 TS pattern: trading levels](https://c.mql5.com/2/25/indicator_4__4.png)

### Signal module

Let's add Take Profit level calculation to illustrate adding new options to a custom TS. There is no such a level in the original version as only a trailing stop is used to close a position. Let's make Take Profit dependent on the custom minimum breakout level (TS\_8020\_Extremum\_Break) — we will multiply it by the TS\_8020\_Take\_Profit\_Ratio custom ratio.

We will need the following elements of the fe\_Get\_Entry\_Signal signal module's main function: current signal status, calculated entry and exit levels (Stop Loss and Take Profit), as well as yesterday's range borders. All levels are received via links to the variables passed to the function, while the signal's return status uses the list of options from the previous article:

enum ENUM\_ENTRY\_SIGNAL {  // The list of entry signals

ENTRY\_BUY,              // buy signal

ENTRY\_SELL,             // sell signal

ENTRY\_NONE,             // no signal

ENTRY\_UNKNOWN           // status not defined

};

ENUM\_ENTRY\_SIGNAL fe\_Get\_Entry\_Signal( // D1 two-candle pattern analysis

datetime  t\_Time,          // current time

double&    d\_Entry\_Level,  // entry level (link to the variable)

double&    d\_SL,           // StopLoss level (link to the variable)

double&    d\_TP,           // TakeProfit level (link to the variable)

double&    d\_Range\_High,   // High of the pattern's 1 st bar (link to the variable)

double&    d\_Range\_Low     // Low of the pattern's 1 st bar (link to the variable)

) {}

In order to detect a signal, we need to analyze the last two bars of D1 timeframe. Let's start from the first one — if it does not meet the TS criteria, there is no need to check the second bar. There are two criteria:

1\. The bar size (difference between High and Low) should exceed the average value for the last XX days (set by the TS\_8020\_D1\_Average\_Period custom setting)

2\. Bar Open and Close levels should be located at the opposite 20% of the bar range

If these conditions are met, High and Low prices should be saved for further use. Since the first bar parameters do not change within the entire day, there is no point in checking them at each function call. Let's store them in static variables:

// custom settings

inputuint  TS\_8020\_D1\_Average\_Period = 20;  // 80-20: Number of days for calculating the average daily range

inputuint  TS\_8020\_Extremum\_Break = 50;     // 80-20: Minimum breakout of the yesterday's extremum (in points)

static ENUM\_ENTRY\_SIGNAL se\_Possible\_Signal = ENTRY\_UNKNOWN; // pattern's first bar signal direction

staticdouble

// variables for storing calculated levels between ticks

sd\_Entry\_Level = 0,

sd\_SL = 0, sd\_TP = 0,

sd\_Range\_High = 0, sd\_Range\_Low = 0

;

// check the pattern's first bar on D1:

if(se\_Possible\_Signal == ENTRY\_UNKNOWN) { // not carried out yet

st\_Last\_D1\_Bar = t\_Curr\_D1\_Bar; // 1 st bar does not change this day

// average daily range

double d\_Average\_Bar\_Range = fd\_Average\_Bar\_Range(TS\_8020\_D1\_Average\_Period, PERIOD\_D1, t\_Time);

if(ma\_Rates\[0\].high — ma\_Rates\[0\].low <= d\_Average\_Bar\_Range) {

// 1 st bar is not large enough

     se\_Possible\_Signal = ENTRY\_NONE; // means no signal today

return(se\_Possible\_Signal);

}

double d\_20\_Percents = 0.2 \\* (ma\_Rates\[0\].high — ma\_Rates\[0\].low); // 20% of the yesterday's range

if((

// bearish bar:

       ma\_Rates\[0\].open > ma\_Rates\[0\].high — d\_20\_Percents // bar opened in the upper 20%

       &&

       ma\_Rates\[0\].close < ma\_Rates\[0\].low + d\_20\_Percents // and closed in the lower 20%

     ) \|\| (

// bullish:

       ma\_Rates\[0\].close > ma\_Rates\[0\].high — d\_20\_Percents // bar closed in the upper 20%

       &&

       ma\_Rates\[0\].open < ma\_Rates\[0\].low + d\_20\_Percents // and opened in the lower 20%

)) {

// 1 st bar corresponds to the conditions

// define today's trading direction for the pattern's 1 st bar:

     se\_Possible\_Signal = ma\_Rates\[0\].open > ma\_Rates\[0\].close ? ENTRY\_BUY : ENTRY\_SELL;

// market entry level:

     sd\_Entry\_Level = d\_Entry\_Level = se\_Possible\_Signal == ENTRY\_BUY ? ma\_Rates\[0\].low : ma\_Rates\[0\].high;

// pattern's 1 st bar range borders:

     sd\_Range\_High = d\_Range\_High = ma\_Rates\[0\].high;

     sd\_Range\_Low = d\_Range\_Low = ma\_Rates\[0\].low;

} else {

// 1 st bar open/close levels do not match conditions

     se\_Possible\_Signal = ENTRY\_NONE; // means no signal today

return(se\_Possible\_Signal);

}

}

Listing of the function for defining the average bar range within the specified number of bars on the specified timeframe beginning from the specified time function:

double fd\_Average\_Bar\_Range(    // Calculate average bar size

int i\_Bars\_Limit,             // how many bars to consider

ENUM\_TIMEFRAMES e\_TF = PERIOD\_CURRENT,  // bars timeframe

datetime t\_Time = WRONG\_VALUE// when to start calculation

) {

double d\_Average\_Range = 0; // variable for summing values

if(i\_Bars\_Limit < 1) return(d\_Average\_Range);

MqlRates ma\_Rates\[\]; // bar info array

// get bar info from the specified history interval:

if(t\_Time == WRONG\_VALUE) t\_Time = TimeCurrent();

int i\_Price\_Bars = CopyRates(\_Symbol, e\_TF, t\_Time, i\_Bars\_Limit, ma\_Rates);

if(i\_Price\_Bars == WRONG\_VALUE) { // processing CopyRates function error

if(Log\_Level > LOG\_LEVEL\_NONE) PrintFormat("%s: CopyRates: error #%u", \_\_FUNCTION\_\_, \_LastError);

return(d\_Average\_Range);

}

if(i\_Price\_Bars < i\_Bars\_Limit) { // CopyRates function has not retrieved the required data amount

if(Log\_Level > LOG\_LEVEL\_NONE) PrintFormat("%s: CopyRates: copied %u bars of %u", \_\_FUNCTION\_\_, i\_Price\_Bars, i\_Bars\_Limit);

}

// sum of ranges:

int i\_Bar = i\_Price\_Bars;

while(i\_Bar-- > 0)

     d\_Average\_Range += ma\_Rates\[i\_Bar\].high — ma\_Rates\[i\_Bar\].low;

// average value:

return(d\_Average\_Range / double(i\_Price\_Bars));

}

There is only one criterion for the pattern's second (current) bar — breakout of the yesterday's range border should not be less than the one specified in the settings (TS\_8020\_Extremum\_Break). As soon as the level is reached, a signal for placing a pending order appears:

// check the pattern's 2 nd (current) bar on D1:

if(se\_Possible\_Signal == ENTRY\_BUY) {

sd\_SL = d\_SL = ma\_Rates\[1\].low; // StopLoss — to the today's High

if(TS\_8020\_Take\_Profit\_Ratio > 0) sd\_TP = d\_TP = d\_Entry\_Level + \_Point \\* TS\_8020\_Extremum\_Break \* TS\_8020\_Take\_Profit\_Ratio; // TakeProfit

return(

// is the downward breakout clearly seen?

     ma\_Rates\[1\].close < ma\_Rates\[0\].low — \_Point \\* TS\_8020\_Extremum\_Break ?

     ENTRY\_BUY : ENTRY\_NONE

);

}

if(se\_Possible\_Signal == ENTRY\_SELL) {

sd\_SL = d\_SL = ma\_Rates\[1\].high; // StopLoss — to the today's Low

if(TS\_8020\_Take\_Profit\_Ratio > 0) sd\_TP = d\_TP = d\_Entry\_Level — \_Point \\* TS\_8020\_Extremum\_Break \* TS\_8020\_Take\_Profit\_Ratio; // TakeProfit

return(

// is the upward breakout clearly seen?

     ma\_Rates\[1\].close > ma\_Rates\[0\].high + \_Point \\* TS\_8020\_Extremum\_Break ?

     ENTRY\_SELL : ENTRY\_NONE

);

}

Save the two functions mentioned above (fe\_Get\_Entry\_Signal and fd\_Average\_Bar\_Range) and the custom settings related to receiving a signal to the mqh library file. The full listing is attached below. Let's name the file Signal\_80-20.mqh and place it to the appropriate directory of the terminal data folder (MQL5\\Include\\Expert\\Signal).

### Indicator for manual trading

Just like the EA, the indicator is to use the signal module described above. The indicator should inform a trader on receiving a pending order placement signal and provide the calculated levels — order placement, Take Profit and Stop Loss levels. A user can select a notification method — a standard pop-up window, email alert or push notification. It is possible to choose all at once or any combination you like.

Another indicator objective is a trading history layout according to '80-20' TS. The indicator is to highlight daily bars corresponding to the system criteria and plot calculated trading levels. The level lines display how the situation evolved over time. For more clarity, let's do as follows: when the price touches the signal line, the latter is replaced with a pending order line. When the pending order is activated, its line is replaced with Take Profit and Stop Loss lines. These lines are interrupted when the price touches one of them (the order is closed). This layout makes it easier to evaluate the efficiency of the trading system rules and define what can be improved.

Let's start with declaring the buffers and their display parameters. First, we need to declare the two buffers with the vertical area filling (DRAW\_FILLING). The first one is to highlight the full daily bar range of the previous day, while another one is to highlight the inner area only to separate it from the upper and lower 20% of the range used in TS. After that, declare the two buffers for the multi-colored signal line and the pending order line (DRAW\_COLOR\_LINE). Their color depends on the trading direction. There are other two lines (Take Proft and Stop Loss) with their color remaining the same (DRAW\_LINE) — they are to use the same standard colors assigned to them in the terminal. All selected display types, except for a simple line, require two buffers each, therefore the code looks as follows:

#property indicator\_chart\_window

#property indicator\_buffers10

#property indicator\_plots6

#property indicator\_label1  "1 st bar of the pattern"

#property indicator\_type1   DRAW\_FILLING

#property indicator\_color1  clrDeepPink, clrDodgerBlue

#property indicator\_width1  1

#property indicator\_label2  "1 st bar of the pattern"

#property indicator\_type2   DRAW\_FILLING

#property indicator\_color2  clrDeepPink, clrDodgerBlue

#property indicator\_width2  1

#property indicator\_label3  "Signal level"

#property indicator\_type3   DRAW\_COLOR\_LINE

#property indicator\_style3  STYLE\_SOLID

#property indicator\_color3  clrDeepPink, clrDodgerBlue

#property indicator\_width3  2

#property indicator\_label4  "Entry level"

#property indicator\_type4   DRAW\_COLOR\_LINE

#property indicator\_style4  STYLE\_DASHDOT

#property indicator\_color4  clrDeepPink, clrDodgerBlue

#property indicator\_width4  2

#property indicator\_label5  "Stop Loss"

#property indicator\_type5   DRAW\_LINE

#property indicator\_style5  STYLE\_DASHDOTDOT

#property indicator\_color5  clrCrimson

#property indicator\_width5  1

#property indicator\_label6  "Take Profit"

#property indicator\_type6   DRAW\_LINE

#property indicator\_style6  STYLE\_DASHDOTDOT

#property indicator\_color6  clrLime

#property indicator\_width6  1

Let's provide traders with the ability to disable the filling of the daily pattern's first bar, select signal notification options and limit the history layout depth. All trading system settings from the signal module are also included here. To do this, we need to preliminarily enumerate the variables used in the module even if some of them are to be used only in the EA and are of no need in the indicator:

#include <Expert\\Signal\\Signal\_80-20.mqh> // '80-20' TS signal module

inputbool    Show\_Outer = true;      // 1 st bar of the pattern: Show the full range?

inputbool    Show\_Inner = true;      // 1 st bar of the pattern: Show the inner area?

inputbool    Alert\_Popup = true;     // Alert: Show a pop-up window?

inputbool    Alert\_Email = false;    // Alert: Send an eMail?

inputstring  Alert\_Email\_Subj = "";  // Alert: eMail subject

inputbool    Alert\_Push = true;      // Alert: Send a push notification?

inputuint  Bars\_Limit = 2000;  // History layout depth (in the current TF bars)

ENUM\_LOG\_LEVEL  Log\_Level = LOG\_LEVEL\_NONE;  // Logging mode

double

buff\_1st\_Bar\_Outer\[\], buff\_1st\_Bar\_Outer\_Zero\[\], // buffers for plotting the full range of the pattern's 1 st bar

buff\_1st\_Bar\_Inner\[\], buff\_1st\_Bar\_Inner\_Zero\[\], // buffers for plotting the internal 60% of the pattern's 1 st bar

buff\_Signal\[\], buff\_Signal\_Color\[\], // signal line buffers

buff\_Entry\[\], buff\_Entry\_Color\[\], // pending order line buffers

buff\_SL\[\], buff\_TP\[\], // StopLoss and TakeProfit lines' buffers

gd\_Extremum\_Break = 0// TS\_8020\_Extremum\_Break in symbol prices

;

int

gi\_D1\_Average\_Period = 1, // correct value for TS\_8020\_D1\_Average\_Period

gi\_Min\_Bars = WRONG\_VALUE// minimum required number of bars for re-calculation

;

intOnInit() {

// check the entered TS\_8020\_D1\_Average\_Period parameter:

gi\_D1\_Average\_Period = int(fmin(1, TS\_8020\_D1\_Average\_Period));

// converting points to symbol prices:

gd\_Extremum\_Break = TS\_8020\_Extremum\_Break \* \_Point;

// minimum required number of bars for re-calculation = number of bars of the current TF within a day

gi\_Min\_Bars = int(86400 / PeriodSeconds());

// indicator buffers' objective:

// 1 st bar's full range rectangle

SetIndexBuffer(0, buff\_1st\_Bar\_Outer, INDICATOR\_DATA);

PlotIndexSetDouble(0, PLOT\_EMPTY\_VALUE, 0);

SetIndexBuffer(1, buff\_1st\_Bar\_Outer\_Zero, INDICATOR\_DATA);

// 1 st bar's inner area rectangle

SetIndexBuffer(2, buff\_1st\_Bar\_Inner, INDICATOR\_DATA);

PlotIndexSetDouble(1, PLOT\_EMPTY\_VALUE, 0);

SetIndexBuffer(3, buff\_1st\_Bar\_Inner\_Zero, INDICATOR\_DATA);

// signal line

SetIndexBuffer(4, buff\_Signal, INDICATOR\_DATA);

PlotIndexSetDouble(2, PLOT\_EMPTY\_VALUE, 0);

SetIndexBuffer(5, buff\_Signal\_Color, INDICATOR\_COLOR\_INDEX);

// pending order placement line

SetIndexBuffer(6, buff\_Entry, INDICATOR\_DATA);

PlotIndexSetDouble(3, PLOT\_EMPTY\_VALUE, 0);

SetIndexBuffer(7, buff\_Entry\_Color, INDICATOR\_COLOR\_INDEX);

// SL line

SetIndexBuffer(8, buff\_SL, INDICATOR\_DATA);

PlotIndexSetDouble(4, PLOT\_EMPTY\_VALUE, 0);

// TP line

SetIndexBuffer(9, buff\_TP, INDICATOR\_DATA);

PlotIndexSetDouble(5, PLOT\_EMPTY\_VALUE, 0);

IndicatorSetInteger(INDICATOR\_DIGITS, \_Digits);

IndicatorSetString(INDICATOR\_SHORTNAME, "80-20 TS");

return(INIT\_SUCCEEDED);

}

Place the main program's code to the built-in OnCalculate function — arrange the loop for iterating over the current timeframe's bars from the past to the future searching them for a signal using the function from the signal module. Declare and initialize the necessary variables using initial values. Let's define the oldest loop bar for the first calculation considering a user-defined history depth limit (Bars\_Limit). For subsequent calls, all bars of the current day (rather than the last bar) are re-calculated, since the two-bar pattern actually belongs to D1 chart regardless of the current timeframe.

Besides, we should protect against the so-called phantoms: if we do not perform a forced indicator buffers clearing during re-initialization, then no longer relevant filled areas remain on the screen when switching timeframes or symbols. The buffer clearing should be bound to the first OnCalculate function call after the indicator initialization. However, the standard prev\_calculated variable is not enough for defining if the call is the first one, since it may contain zero not only during the first function call but also "when changing the checksum". Let's spend some time to properly solve this issue by creating the structure not affected by setting the prev\_calculated variable to zero. The structure is to store and process data frequently used in the indicators:

> \- flag of the OnCalculate function first launch;
>
> \- the counter of calculated bars that is not set to zero when changing the checksum;
>
> \- flag of changing the checksum;
>
> \- flag of the beginning of a new bar;
>
> \- current bar start time.

The structure combining all these data is to be declared at the global level. It should be able to gather or present data from/to any built-in or custom functions. Let's name this structure Brownie. It can be placed to the end of the indicator code. A single global type structure object named go\_Brownie is to be declared there as well:

struct BROWNIE {                // Brownie: structure for storing and processing data at the global level

datetime  t\_Last\_Bar\_Time;    // time of the last processed bar

int        i\_Prew\_Calculated; // number of calculated bars

bool      b\_First\_Run;        // first launch flag

bool      b\_History\_Updated;  // history update flag

bool      b\_Is\_New\_Bar;       // new bar opening flag

BROWNIE() { // constructor

// default values:

     t\_Last\_Bar\_Time = 0;

     i\_Prew\_Calculated = WRONG\_VALUE;

     b\_First\_Run = b\_Is\_New\_Bar = true;

     b\_History\_Updated = false;

}

void f\_Reset(bool b\_Reset\_First\_Run = true) { // setting variables to zero

// default values:

     t\_Last\_Bar\_Time = 0;

     i\_Prew\_Calculated = WRONG\_VALUE;

if(b\_Reset\_First\_Run) b\_First\_Run = true; // set to zero if there is permission

     b\_Is\_New\_Bar = true;

     b\_History\_Updated = false;

}

void f\_Update(int i\_New\_Prew\_Calculated = WRONG\_VALUE) { // update the variables

// flag of the OnCalculate built-in function first call

if(b\_First\_Run && i\_Prew\_Calculated > 0) b\_First\_Run = false;

// new bar?

datetime t\_This\_Bar\_Time = TimeCurrent() \- TimeCurrent() % PeriodSeconds();

     b\_Is\_New\_Bar = t\_Last\_Bar\_Time == t\_This\_Bar\_Time;

// update the current bar time?

if(b\_Is\_New\_Bar) t\_Last\_Bar\_Time = t\_This\_Bar\_Time;

if(i\_New\_Prew\_Calculated > -1) {

// are there any changes in history?

       b\_History\_Updated = i\_New\_Prew\_Calculated == 0 && i\_Prew\_Calculated > WRONG\_VALUE;

// use prew\_calculated in case of OnCalculate 1 st call

if(i\_Prew\_Calculated == WRONG\_VALUE) i\_Prew\_Calculated = i\_New\_Prew\_Calculated;

// or if there was no history update

elseif(i\_New\_Prew\_Calculated > 0) i\_Prew\_Calculated = i\_New\_Prew\_Calculated;

     }

}

};

BROWNIE go\_Brownie;

Let's inform the Brownie of the indicator de-initialization event:

voidOnDeinit(constint reason) {

go\_Brownie.f\_Reset(); // inform Brownie

}

If necessary, the amount of data stored by the Brownie can be expanded if custom functions or classes need prices, volumes or the current bar's spread value (Open, High, Low, Close, tick\_volume, volume, spread). It is more convenient to use ready-made data from the OnCalculate function and pass them via Brownie rather than using the time series copying functions (CopyOpen, CopyHigh etc. or CopyRates) — this saves the CPU resources and eliminates the necessity to arrange processing of errors of these language functions.

Let's get back to the main indicator function. Declaring variables and preparing the arrays using the go\_Brownie structure look as follows:

go\_Brownie.f\_Update(prev\_calculated); // feed data to Brownie

int

i\_Period\_Bar = 0, // auxiliary counter

i\_Current\_TF\_Bar = rates\_total - int(Bars\_Limit) // bar index of the current TF loop start

;

staticdatetime st\_Last\_D1\_Bar = 0; // time of the last processed bar of the couple of D1 bars (pattern's 2 nd bar)

staticint si\_1st\_Bar\_of\_Day = 0; // index of the current day's first bar

if(go\_Brownie.b\_First\_Run) { // if this is the 1 st launch

// clear the buffers during re-initialization:

ArrayInitialize(buff\_1st\_Bar\_Inner, 0); ArrayInitialize(buff\_1st\_Bar\_Inner\_Zero, 0);

ArrayInitialize(buff\_1st\_Bar\_Outer, 0); ArrayInitialize(buff\_1st\_Bar\_Outer\_Zero, 0);

ArrayInitialize(buff\_Entry, 0); ArrayInitialize(buff\_Entry\_Color, 0);

ArrayInitialize(buff\_Signal, 0); ArrayInitialize(buff\_Signal\_Color, 0);

ArrayInitialize(buff\_TP, 0);

ArrayInitialize(buff\_SL, 0);

st\_Last\_D1\_Bar = 0;

si\_1st\_Bar\_of\_Day = 0;

} else { // this is not the 1 st launch

datetime t\_Time = TimeCurrent();

// minimum re-calculation depth - from the previous day:

i\_Current\_TF\_Bar = rates\_total - Bars(\_Symbol, PERIOD\_CURRENT, t\_Time - t\_Time % 86400, t\_Time) - 1;

}

ENUM\_ENTRY\_SIGNAL e\_Signal = ENTRY\_UNKNOWN; // signal

double

d\_SL = WRONG\_VALUE, // SL level

d\_TP = WRONG\_VALUE, // TP level

d\_Entry\_Level = WRONG\_VALUE, // entry level

d\_Range\_High = WRONG\_VALUE, d\_Range\_Low = WRONG\_VALUE// borders of the pattern's 1 st bar range

;

datetime

t\_Curr\_D1\_Bar = 0, // current D1 bar time (pattern's 2 nd bar)

t\_D1\_Bar\_To\_Fill = 0// D1 bar time to be filled (pattern's 1 st bar)

;

// make sure the initial re-calculation bar index is within acceptable range:

i\_Current\_TF\_Bar = int(fmax(0, fmin(i\_Current\_TF\_Bar, rates\_total - gi\_Min\_Bars)));

while(++i\_Current\_TF\_Bar < rates\_total && !IsStopped()) { // iterate over the current TF bars

// the main program loop is to be located here

}

Check the presence of a signal when iterating over the current timeframe bars:

e\_Signal = fe\_Get\_Entry\_Signal(Time\[i\_Current\_TF\_Bar\], d\_Entry\_Level, d\_SL, d\_TP, d\_Range\_High, d\_Range\_Low);

if(e\_Signal > 1) continue; // no signal during the day the bar belongs to

If there is a signal on a new day's first bar, the range of the previous daily bar should be filled. The value of the t\_D1\_Bar\_To\_Fill variable of datetime type is used as a flag. If it is equal to WRONG\_VALUE, no filling is required on this bar. The signal line should start at the same first bar, but let's extend it to the last bar of the previous day for better layout perception. Since the calculations of a signal line, as well as line and filling colors for bullish and bearish bars are different, let's make two similar blocks:

t\_Curr\_D1\_Bar = Time\[i\_Current\_TF\_Bar\] — Time\[i\_Current\_TF\_Bar\] % 86400; // start of the day the bar belongs to

if(st\_Last\_D1\_Bar < t\_Curr\_D1\_Bar) { // this is a new day bar

t\_D1\_Bar\_To\_Fill = Time\[i\_Current\_TF\_Bar — 1\] — Time\[i\_Current\_TF\_Bar — 1\] % 86400;

si\_1st\_Bar\_of\_Day = i\_Current\_TF\_Bar;

}

else t\_D1\_Bar\_To\_Fill = WRONG\_VALUE; // previous day bar, no new filling required

st\_Last\_D1\_Bar = t\_Curr\_D1\_Bar; // remember

if(t\_D1\_Bar\_To\_Fill != WRONG\_VALUE) { // new D1 bar

// Filling the previous day's D1 bar:

i\_Period\_Bar = i\_Current\_TF\_Bar;

if(d\_Entry\_Level < d\_Range\_High) { // D1 bearish bar

if(Show\_Outer) while(--i\_Period\_Bar > 0) { // full range

if(Time\[i\_Period\_Bar\] < t\_D1\_Bar\_To\_Fill) break;

       buff\_1st\_Bar\_Outer\_Zero\[i\_Period\_Bar\] = d\_Range\_Low;

       buff\_1st\_Bar\_Outer\[i\_Period\_Bar\] = d\_Range\_High;

     }

if(Show\_Inner) { // inner area

       i\_Period\_Bar = i\_Current\_TF\_Bar;

while(--i\_Period\_Bar > 0) {

if(Time\[i\_Period\_Bar\] < t\_D1\_Bar\_To\_Fill) break;

         buff\_1st\_Bar\_Inner\_Zero\[i\_Period\_Bar\] = d\_Range\_Low + 0.2 \\* (d\_Range\_High — d\_Range\_Low);

         buff\_1st\_Bar\_Inner\[i\_Period\_Bar\] = d\_Range\_High — 0.2 \\* (d\_Range\_High — d\_Range\_Low);

       }

     }

// start of the signal line — from the previous day's last bar

     buff\_Signal\[i\_Current\_TF\_Bar\] = buff\_Signal\[i\_Current\_TF\_Bar — 1\] = d\_Range\_Low — gd\_Extremum\_Break;

     buff\_Signal\_Color\[i\_Current\_TF\_Bar\] = buff\_Signal\_Color\[i\_Current\_TF\_Bar — 1\] = 0;

} else { // bullish D1 bar

if(Show\_Outer) while(--i\_Period\_Bar > 0) { // full range

if(Time\[i\_Period\_Bar\] < t\_D1\_Bar\_To\_Fill) break;

       buff\_1st\_Bar\_Outer\_Zero\[i\_Period\_Bar\] = d\_Range\_High;

       buff\_1st\_Bar\_Outer\[i\_Period\_Bar\] = d\_Range\_Low;

     }

if(Show\_Inner) { // inner area

       i\_Period\_Bar = i\_Current\_TF\_Bar;

while(--i\_Period\_Bar > 0) {

if(Time\[i\_Period\_Bar\] < t\_D1\_Bar\_To\_Fill) break;

         buff\_1st\_Bar\_Inner\_Zero\[i\_Period\_Bar\] = d\_Range\_High — 0.2 \\* (d\_Range\_High — d\_Range\_Low);

         buff\_1st\_Bar\_Inner\[i\_Period\_Bar\] = d\_Range\_Low + 0.2 \\* (d\_Range\_High — d\_Range\_Low);

       }

     }

// start of the signal line — from the previous day's last bar

     buff\_Signal\[i\_Current\_TF\_Bar\] = buff\_Signal\[i\_Current\_TF\_Bar — 1\] = d\_Range\_High + gd\_Extremum\_Break;

     buff\_Signal\_Color\[i\_Current\_TF\_Bar\] = buff\_Signal\_Color\[i\_Current\_TF\_Bar — 1\] = 1;

}

} elsecontinue;

All the remaining layout lines are to be plotted inside the current timeframe's bars iteration loop. As already mentioned, the signal line should end at the bar where the price touched it. The pending order line should start at the same bar and end on the bar, at which the contact with the price occurs. Take Profit and Stop Loss lines should start at the same bar. The layout of the pattern is finished at the bar, at which the price touches one of them:

// Signal line till crossed by a bar:

i\_Period\_Bar = i\_Current\_TF\_Bar;

if(d\_Entry\_Level < d\_Range\_High) { // bearish D1 bar

while(++i\_Period\_Bar < rates\_total) {

if(Time\[i\_Period\_Bar\] > t\_Curr\_D1\_Bar + 86399) break;

     buff\_Signal\[i\_Period\_Bar\] = d\_Range\_Low — gd\_Extremum\_Break;

     buff\_Signal\_Color\[i\_Period\_Bar\] = 0;

if(d\_Range\_Low — gd\_Extremum\_Break >= Low\[i\_Period\_Bar\]) break;

}

} else { // bullish D1 bar

while(++i\_Period\_Bar < rates\_total) {

if(Time\[i\_Period\_Bar\] > t\_Curr\_D1\_Bar + 86399) break;

     buff\_Signal\[i\_Period\_Bar\] = d\_Range\_High + gd\_Extremum\_Break;

     buff\_Signal\_Color\[i\_Period\_Bar\] = 1;

if(d\_Range\_High + gd\_Extremum\_Break <= High\[i\_Period\_Bar\]) break;

}

}

// Entry line till crossed by a bar:

if(d\_Entry\_Level < d\_Range\_High) { // bearish D1 bar

while(++i\_Period\_Bar < rates\_total) {

if(Time\[i\_Period\_Bar\] > t\_Curr\_D1\_Bar + 86399) break;

     buff\_Entry\[i\_Period\_Bar\] = d\_Range\_Low;

     buff\_Entry\_Color\[i\_Period\_Bar\] = 0;

if(d\_Range\_Low <= High\[i\_Period\_Bar\]) {

if(buff\_Entry\[i\_Period\_Bar — 1\] == 0.) {

// start and end on a single bar, extend by 1 bar to the past

         buff\_Entry\[i\_Period\_Bar — 1\] = d\_Range\_Low;

         buff\_Entry\_Color\[i\_Period\_Bar — 1\] = 0;

       }

break;

     }

}

} else { // bullish D1 bar

while(++i\_Period\_Bar < rates\_total) {

if(Time\[i\_Period\_Bar\] > t\_Curr\_D1\_Bar + 86399) break;

     buff\_Entry\[i\_Period\_Bar\] = d\_Range\_High;

     buff\_Entry\_Color\[i\_Period\_Bar\] = 1;

if(d\_Range\_High >= Low\[i\_Period\_Bar\]) {

if(buff\_Entry\[i\_Period\_Bar — 1\] == 0.) {

// start and end on a single bar, extend by 1 bar to the past

         buff\_Entry\[i\_Period\_Bar — 1\] = d\_Range\_High;

         buff\_Entry\_Color\[i\_Period\_Bar — 1\] = 1;

       }

break;

     }

}

}

// TP and SL lines till one of them is crossed by a bar:

if(d\_Entry\_Level < d\_Range\_High) { // bearish D1 bar

// SL is equal to the Low since the beginning of a day:

d\_SL = Low\[ArrayMinimum(Low, si\_1st\_Bar\_of\_Day, i\_Period\_Bar — si\_1st\_Bar\_of\_Day)\];

while(++i\_Period\_Bar < rates\_total) {

if(Time\[i\_Period\_Bar\] > t\_Curr\_D1\_Bar + 86399) break;

     buff\_SL\[i\_Period\_Bar\] = d\_SL;

     buff\_TP\[i\_Period\_Bar\] = d\_TP;

if(d\_TP <= High\[i\_Period\_Bar\] \|\| d\_SL >= Low\[i\_Period\_Bar\]) {

if(buff\_SL\[i\_Period\_Bar — 1\] == 0.) {

// start and end on a single bar, extend by 1 bar to the past

         buff\_SL\[i\_Period\_Bar — 1\] = d\_SL;

         buff\_TP\[i\_Period\_Bar — 1\] = d\_TP;

       }

break;

     }

}

} else { // bullish D1 bar

// SL is equal to the High since the beginning of a day:

d\_SL = High\[ArrayMaximum(High, si\_1st\_Bar\_of\_Day, i\_Period\_Bar — si\_1st\_Bar\_of\_Day)\];

while(++i\_Period\_Bar < rates\_total) {

if(Time\[i\_Period\_Bar\] > t\_Curr\_D1\_Bar + 86399) break;

     buff\_SL\[i\_Period\_Bar\] = d\_SL;

     buff\_TP\[i\_Period\_Bar\] = d\_TP;

if(d\_SL <= High\[i\_Period\_Bar\] \|\| d\_TP >= Low\[i\_Period\_Bar\]) {

if(buff\_SL\[i\_Period\_Bar — 1\] == 0.) {

// start and end on a single bar, extend by 1 bar to the past

         buff\_SL\[i\_Period\_Bar — 1\] = d\_SL;

         buff\_TP\[i\_Period\_Bar — 1\] = d\_TP;

       }

break;

     }

}

}

Let's place the call code of the f\_Do\_Alert signal notification function out of the loop. In fact, it has slightly wider opportunities as compared to the ones involved in this indicator — the function is able to work with audio files meaning that this option can be added to custom settings. The same is true for the ability to select separate files for buy and sell signals. Function listing:

void f\_Do\_Alert(                  // Function for sending signals and notifications

string  s\_Message,              // alert message

bool    b\_Alert = true,         // show a pop-up window?

bool    b\_Sound = false,        // play a sound file?

bool    b\_Email = false,        // send an eMail?

bool    b\_Notification = false, // send a push notification?

string  s\_Email\_Subject = "",   // eMail subject

string  s\_Sound = "alert.wav"// sound file

) {

staticstring ss\_Prev\_Message = "there was silence"; // previous alert message

staticdatetime st\_Prev\_Time; // previous alert bar time

datetime t\_This\_Bar\_Time = TimeCurrent() — PeriodSeconds() % PeriodSeconds(); // current bar time

if(ss\_Prev\_Message != s\_Message \|\| st\_Prev\_Time != t\_This\_Bar\_Time) {

// another and/or 1 st at this bar

// remember:

     ss\_Prev\_Message = s\_Message;

     st\_Prev\_Time = t\_This\_Bar\_Time;

// form a message string:

     s\_Message = StringFormat("%s \| %s \| %s \| %s",

TimeToString(TimeLocal(), TIME\_SECONDS), // local time

\_Symbol, // symbol

StringSubstr(EnumToString(ENUM\_TIMEFRAMES(\_Period)), 7), // TF

       s\_Message // message

     );

// activate notification signal:

if(b\_Alert) Alert(s\_Message);

if(b\_Email) SendMail(s\_Email\_Subject + " " \+ \_Symbol, s\_Message);

if(b\_Notification) SendNotification(s\_Message);

if(b\_Sound) PlaySound(s\_Sound);

}

}

The code for checking the need for calling the function and forming the text for it located in the program body before completion of the OnCalculate event handler:

// alert

i\_Period\_Bar = rates\_total — 1; // current bar

if(Alert\_Popup + Alert\_Email + Alert\_Push == 0) return(rates\_total); // all is disabled

if(buff\_Signal\[i\_Period\_Bar\] == 0) return(rates\_total); // nothing to catch yet (or already)

if(

buff\_Signal\[i\_Period\_Bar\] > High\[i\_Period\_Bar\]

\|\|

buff\_Signal\[i\_Period\_Bar\] < Low\[i\_Period\_Bar\]

) return(rates\_total); // no signal line touching

// message text:

string s\_Message = StringFormat("TS 80-20: needed %s @ %s, TP: %s, SL: %s",

buff\_Signal\_Color\[i\_Period\_Bar\] > 0 ? "BuyStop" : "SellStop",

DoubleToString(d\_Entry\_Level, \_Digits),

DoubleToString(d\_TP, \_Digits),

DoubleToString(d\_SL, \_Digits)

);

// notification:

f\_Do\_Alert(s\_Message, Alert\_Popup, false, Alert\_Email, Alert\_Push, Alert\_Email\_Subj);

return(rates\_total); // complete OnCalculate operation

The entire source code of the indicator can be found in the attached files (TS\_80-20.mq5). The trading layout according to the system is best seen on minute charts.

Please note that the indicator uses the bar data rather than tick sequences inside bars. This means if the price crossed several layout lines (for example, Take Profit and Stop Loss lines) on a single bar, you cannot always define which of them was crossed first. Another uncertainty stems from the fact that the start and end lines cannot coincide. Otherwise, the lines from the buffer of DRAW\_LINE and DRAW\_COLOR\_LINE types will simply be invisible to a user. These features reduce the layout accuracy but it still remains quite clear.

### Expert Advisor for testing the '80-20' trading strategy

The basic EA for testing strategies from the book Street Smarts: [High Probability Short-Term Trading Strategies](https://www.mql5.com/go?link=https://www.amazon.co.uk/Street-Smarts-Probability-Trading-Strategies/dp/0965046109 "https://www.amazon.co.uk/Street-Smarts-Probability-Trading-Strategies/dp/0965046109") was described in details in the [first article](https://www.mql5.com/en/articles/2717). Let's insert two significant changes in it. First, the signal module is to be used in the indicator as well meaning it would be reasonable to set trading levels calculation in it. We have already done this above. Apart from the signal status, the fe\_Get\_Entry\_Signal function returns order placement, Stop Loss and Take Profit levels. Therefore, let's remove the appropriate part of the code from the previous EA version adding the variables for accepting levels from the function and edit the function call itself. The listings of the old and new code blocks can be found in the attached file (strings 128-141).

Another significant addition to the basic EA code is due to the fact that, unlike the previous two, this TS deals with a short-term trend. It assumes that the roll-back happens once a day and is unlikely to be repeated. This means that the robot has to make only one entry ignoring the existing signal all the rest of the time until the next day. The easiest way to implement that is to use a special flag — static or global variable of bool type in the program memory. But if the EA operation is interrupted for some reason (the terminal is closed, the EA is removed from the chart, etc.), the flag value is lost as well. Thus, we should have the ability to check if today's signal was activated previously. To do this, we may analyze the history of trades for today or store the date of the last entry in the terminal global variables rather than in the program. Let us use the second option since it is much easier to implement.

Provide users with the ability to manage 'one entry per day' option and set an ID of each launched version of the robot — it is needed to use global variables of the terminal level:

inputbool  One\_Trade = false;    // One position per day

inputuint  Magic\_Number = 2016;  // EA magic number

Let's add the variables necessary to implement 'one entry per day' option to the program's global variables definition block. Initialize them in the OnInit function:

string

gs\_Prefix // identifier of (super)global variables

;

bool

gb\_Position\_Today = false,

gb\_Pending\_Today = false

;

intOnInit() {

...

// Create a prefix of (super)global variable names:

gs\_Prefix = StringFormat("SSB %s %u %s", \_Symbol, Magic\_Number, MQLInfoInteger(MQL\_TESTER) ? "t " : "");

// Has the robot worked with market or pending orders today?

gb\_Position\_Today = int(GlobalVariableGet(gs\_Prefix + "Last\_Position\_Date")) == TimeCurrent() — TimeCurrent() % 86400;

gb\_Pending\_Today = int(GlobalVariableGet(gs\_Prefix + "Last\_Pending\_Date")) == TimeCurrent() — TimeCurrent() % 86400;

...

}

Here the robot reads the values of global variables and compares the written time with the day start time, thus defining if the today's signal has already been processed. Time is written to the variables in two places — let's add the appropriate block to the pending order installation code (additions highlighted):

if(i\_Try != -10) { // placing a pending order failed

if(Log\_Level > LOG\_LEVEL\_NONE) Print("Pending order placing error");

// the distance from the current price is not enough :(

if(Log\_Level > LOG\_LEVEL\_ERR)

PrintFormat("Pending order cannot be placed at the %s level. Bid: %s Ask: %s StopLevel: %s",

DoubleToString(d\_Entry\_Level, \_Digits),

DoubleToString(go\_Tick.bid, \_Digits),

DoubleToString(go\_Tick.ask, \_Digits),

DoubleToString(gd\_Stop\_Level, \_Digits)

     );

}else { // managed

// to update the flag:

GlobalVariableSet( // in the terminal global variables

     gs\_Prefix + "Last\_Pending\_Date",

TimeCurrent() — TimeCurrent() % 86400

);

gb\_Pending\_Today = true; // in the program global variables

}

The second block is placed after the code defining a newly opened position:

if(PositionSelect(\_Symbol)) { // there is an open position

if(PositionGetDouble(POSITION\_SL) == 0.) { // new position

if(!gb\_Position\_Today) { // this is the 1 st position today

// update the flag:

GlobalVariableSet( // in the terminal global variables

                                 gs\_Prefix + "Last\_Position\_Date",

TimeCurrent() — TimeCurrent() % 86400

                         );

                         gb\_Position\_Today = true; // in the program global variables

                 }

...

These are the only significant changes in the previous EA version code. The finalized source code of the new version is attached below.

### Strategy backtesting

In order to illustrate the trading system viability, its authors use patterns detected on the charts from the end of the last century. Therefore, we need to check its relevance in today's market conditions. For testing, I took the most popular Forex pair EURUSD, the most volatile pair USDJPY and one of the metals — XAUUSD. I increased the indents specified by Raschke and Connors 10 times, since four-digit quotes were used when the book was written, while I tested the EA on five-digit ones. Since there is no any guidance concerning the trailing parameters, I have selected the ones that seem to be most appropriate to daily timeframe and instrument volatility. The same applies to the Take Profit calculation algorithm added to the original rules — the ratio for its calculation was chosen arbitrarily, without deep optimization.

The balance chart when testing on the five-year EURUSD history with the original rules (no Take Profit):

![EURUSD D1 5 years](https://c.mql5.com/2/25/8020_A_EURUSD_D1_5yrs.png)

The same settings and Take Profit:

![EURUSD D1 5 years](https://c.mql5.com/2/25/8020_B_EURUSD_D1_5yrs.png)

The balance chart when testing the original rules on the five-year USDJPY history:

![USDJPY D1 5 years](https://c.mql5.com/2/25/8020_A_USDJPY_D1_5yrs.png)

The same settings and Take Profit:

![USDJPY D1 5 years](https://c.mql5.com/2/25/8020_B_USDJPY_D1_5yrs.png)

The balance chart when testing the original rules on the daily gold quotes for the last 4 years:

![XAUUSD D1 4 years](https://c.mql5.com/2/25/8020_A_XAUUSD_D1_4yrs.png)

The full data on the robot settings used in each test can be found in the attached archive containing the complete reports.

### Conclusion

The rules programmed in the signal module match the 80-20 trading system description provided by Linda Raschke and Laurence Connors in their book "Street Smarts: High Probability Short-Term Trading Strategies". However, we have extended the original rules a bit. The tools (the robot and the indicator) are to help traders draw their own conclusions concerning the TS relevance in today's market. In my humble opinion, the TS needs a serious upgrade. In this article, I have tried to make some detailed comments on developing the code of the signal module, as well as the appropriate robot and indicator. I hope, this will help those who decide to do the upgrade. Apart from modifying the rules, it is also possible to find trading instruments that fit better to the system, as well as signal detection and tracking parameters.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2785](https://www.mql5.com/ru/articles/2785)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2785.zip "Download all attachments in the single ZIP archive")

[Reports.zip](https://www.mql5.com/en/articles/download/2785/reports.zip "Download Reports.zip")(607.29 KB)

[MQL5.zip](https://www.mql5.com/en/articles/download/2785/mql5.zip "Download MQL5.zip")(123.59 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Momentum Pinball trading strategy](https://www.mql5.com/en/articles/2825)
- [The 'Turtle Soup' trading system and its 'Turtle Soup Plus One' modification](https://www.mql5.com/en/articles/2717)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/164307)**
(6)


![Hank Scorpio](https://c.mql5.com/avatar/2022/6/629D446D-25E1.jpg)

**[Hank Scorpio](https://www.mql5.com/en/users/clemmo)**
\|
22 Dec 2016 at 12:22

It seems like the criteria for this strategy would rarely be met.

1\. First a strong unhesitating [momentum](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/momentum "MetaTrader 5 Help: Momentum Indicator") candle (with wicks < 20%).

2\. Then price must exceed the high or low of that candle, thus confirming the momentum.

3\. Then price must reverse and cross the border of yesterday's range against its previous momentum.  This is a paradox, and would rarely happen except with news.

If I understood correctly I would guess that this strategy rarely places trades. Is that what your research shows?

Перевод Google

Похоже, что критерии этой стратегии редко будут удовлетворены.

1\. Сначала сильный импульс неколеблющийся свеча (с фитилями <20%).

2\. Тогда цена должна превышать максимума или минимума этой свечи, подтвердив тем самым импульс.

3\. Тогда цена должна развернуться и пересечь границу вчерашнего диапазона от его предыдущего импульса. Это парадокс, и редко случаются, кроме как с новостями.

Если я правильно понял, я бы предположил, что эта стратегия редко ставит сделок. Это то, что ваше исследование показывает?

![tahach](https://c.mql5.com/avatar/avatar_na2.png)

**[tahach](https://www.mql5.com/en/users/tahach)**
\|
10 Feb 2017 at 02:03

**clemmo:**

It seems like the criteria for this strategy would rarely be met.

1\. First a strong unhesitating momentum candle (with wicks < 20%).

2\. Then price must exceed the high or low of that candle, thus confirming the momentum.

3\. Then price must reverse and cross the border of yesterday's range against its previous momentum.  This is a paradox, and would rarely happen except with news.

If I understood correctly I would guess that this strategy rarely places trades. Is that what your research shows?

Перевод Google

Похоже, что критерии этой стратегии редко будут удовлетворены.

1\. Сначала сильный импульс неколеблющийся свеча (с фитилями <20%).

2\. Тогда цена должна превышать максимума или минимума этой свечи, подтвердив тем самым импульс.

3\. Тогда цена должна развернуться и пересечь границу вчерашнего диапазона от его предыдущего импульса. Это парадокс, и редко случаются, кроме как с новостями.

Если я правильно понял, я бы предположил, что эта стратегия редко ставит сделок. Это то, что ваше исследование показывает?

Well, the reports contain the total amount of trades, which are 233 in one case, in almost 6 years. I would not call it a small number, but they are hardly profitable!


![Yudha Adicita](https://c.mql5.com/avatar/2017/5/591EC5B7-85E4.jpg)

**[Yudha Adicita](https://www.mql5.com/en/users/metabotfx)**
\|
19 May 2017 at 10:40

:) interesting, seem work nice on trending market, but on ranging would be difficult to gain pips


![whkh18](https://c.mql5.com/avatar/avatar_na2.png)

**[whkh18](https://www.mql5.com/en/users/whkh18)**
\|
19 Jul 2017 at 03:07

Thanks for sharing, [gold](https://www.mql5.com/en/quotes/metals/XAUUSD "XAUUSD chart: technical analysis") 0.3 U.S. dollars can return to the capital of the broker, there is a need to contact WeChat whkh18 Both  username

![demleitner](https://c.mql5.com/avatar/avatar_na2.png)

**[demleitner](https://www.mql5.com/en/users/demleitner)**
\|
28 Mar 2020 at 20:42

Russian? No problem! I speak German and reasonable English and found my way:

Put Windows on Russian for 8-bit characters. My Windows 10 is in German now, Later I can add, how to do it in English Windows 10. Now for German Windows 10 - this is the way:

Open "Control Panel"

switch to "Large [icons](https://www.mql5.com/en/docs/constants/tradingconstants/enum_series_info_integer "Reference book MQL5 : Information on historical data of an instrument")" or "Small icons"

From there open "Region"

On the top you find 2 pages, named "Formats" and Administration" switch to "Administration"

Now you see two buttons (where you need Administrator-rights) "Copy settings..." and "Change locale"

The latter one is what you need, "Change locale". Here you choose "Russian (Russian Federation"

You need to restart, by clicking OK and your way out of there the machine will tell you.

Now ASCII 7-bit is still correct (English) but the higher letters (above 127) are now Russian.

Then you can copy the Russian from Metaeditor and paste it to a translator i.e. translate.google.com.

I recommend to translate to English - most of the time it looks like any US-programmer did it. to German is very often terrible (just my opinion).

Greetings to the non-Russian-speaking world (and to the Russian spoken authors as well,

Gerhard

![How to build and test a Binary Options strategy with the MetaTrader 4 Strategy Tester](https://c.mql5.com/2/25/Avatar-Binary-Options-strategy-002.png)[How to build and test a Binary Options strategy with the MetaTrader 4 Strategy Tester](https://www.mql5.com/en/articles/2820)

Tutorial to build a Binary Options strategy an test it in Strategy-Tester of MetaTrader 4 with Binary-Options-Strategy-Tester utility from marketplace.

![LifeHack for Trader: A comparative report of several tests](https://c.mql5.com/2/25/life_hacks_02.png)[LifeHack for Trader: A comparative report of several tests](https://www.mql5.com/en/articles/2731)

The article deals with the simultaneous launch of Expert Advisor testing on four different trading instruments. The final comparison of four testing reports is provided in a table similar to how goods are represented in online stores. An additional bonus is that distribution charts will be automatically created for each symbol.

![Universal ZigZag](https://c.mql5.com/2/25/zigzag__1.png)[Universal ZigZag](https://www.mql5.com/en/articles/2774)

ZigZag is one of the most popular indicators among the MetaTrader 5 users. The article analyzes the possibilities for creating various versions of the ZigZag. The result is a universal indicator with ample opportunities to extend its functionality, which is useful in the development of trading experts and other indicators.

![The 'Turtle Soup' trading system and its 'Turtle Soup Plus One' modification](https://c.mql5.com/2/25/turtles.png)[The 'Turtle Soup' trading system and its 'Turtle Soup Plus One' modification](https://www.mql5.com/en/articles/2717)

The article features formalized rules of two trading strategies 'Turtle Soup' and 'Turtle Soup Plus One' from Street Smarts: High Probability Short-Term Trading Strategies by Linda Bradford Raschke and Laurence A. Connors. The strategies described in the book are quite popular. But it is important to understand that the authors have developed them based on the 15...20 year old market behavior.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/2785&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070525050114742129)

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