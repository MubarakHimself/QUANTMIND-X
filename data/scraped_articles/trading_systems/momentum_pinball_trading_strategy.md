---
title: Momentum Pinball trading strategy
url: https://www.mql5.com/en/articles/2825
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:44:15.875867
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=cfngvefzvxdmeixvqlctfeswpdjhydoy&ssn=1769186653165422374&ssn_dr=0&ssn_sr=0&fv_date=1769186653&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2825&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Momentum%20Pinball%20trading%20strategy%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918665399580009&fz_uniq=5070520656363198300&sv=2552)

MetaTrader 5 / Examples


### Introduction

In this article, we continue programming of trading strategies described in a section of the book by L. Raschke and L. Connors [Street Smarts: High Probability Short-Term Trading Strategies](https://www.mql5.com/go?link=https://www.amazon.com/Street-Smarts-Probability-Short-Term-Strategies/dp/0965046109 "Street Smarts: High Probability Short-Term Trading Strategies"), devoted to testing of range limits by price. The last of full-fledged TS in the section is Momentum Pinball, which operates the pattern consisting of two daily bars. By the first bar, trade direction on the second day is defined, and price movement in the beginning of the second bar should specify certain trade levels for entries and exits from the market.

The purpose of this article is to demonstrate to the programmers who have already mastered MQL5, one of the variants for realizing Momentum Pinball TS, in which simplified methods of object-oriented programming will be applied. From the full-fledged OOP, the code will differ by the absence of classes - they will be replaced by structures. As opposed to classes, design in the code and application of objects of this type differs minimally from the procedural programming familiar to most starting coders. On the other hand, features being provided by structures are more than enough to resolve such tasks.

Like in the previous article, first, create a signal block module, then - an indicator for manual trading and history marking, which uses this module. The third program will be Expert Advisor for automated trading; it will also use the signal module. In conclusion, we will test the Expert Advisor on fresh quotes because authors of the book worked with 20-year old quotes.

### Rules of the Momentum Pinball TS

L. Raschke and L. Connors faced uncertainty when using the trading techniques [described by George Taylor](https://www.mql5.com/go?link=https://www.amazon.com/Taylor-Trading-Technique-George-Douglass/dp/0934380244 "The Taylor Trading Technique") which became the reason for compiling rules of this trading strategy. Taylor’s strategy prior to the beginning of another day defines direction of its trade - whether this will be a day of sells or a day of buys. However, the author's actual trading often violates this arrangement which, in the opinion of book authors, would get trading rules tangled.

In order to more definitely determine trading direction of the next day, authors applied ROC (Rate Of Change) indicator. RSI (Relative Strength Index) oscillator was applied to its values and cyclicity of ROC values has become well visible. Finally, TS authors added signal levels - borders of overbought and oversold areas on RSI chart. Presence of the line of such indicator (it was named LBR/RSI, from Linda Bradford Raschke) in a respective area is designated to detect most probable sell days and buy days. LBR/RSI is detailed below.

Complete rules of the Momentum Pinball TS for buy entries are formulated as follows.

1. On D1, the value of LBR/RSI of the last closed day should be within the oversold area - below 30.
2. After closing the first hourly bar of a new day, place the pending buy order higher than that bar maximum.
3. After triggering of the pending order, place Stop Loss to the Low of the first hourly bar.
4. If a position is closed with the loss, once again place the pending sell order at the same level.
5. If by the end of the day, the position remains profitable, leave it for the next day. On the second trading day, the position has to be closed.

Visualization of entry rules with the help of two indicators described below, looks as follows:

— LBR/RSI on daily timeframe is in oversold area (see 30 October, 2017)

![](https://c.mql5.com/2/30/im_1__1.png)

— indicator TS\_Momentum\_Pinball on undefined timeframe (from M1 to D1) displays trading levels and price range of the first hour of the day, on which basis these levels are calculated:

![](https://c.mql5.com/2/30/im2__6.png)

The rules for exiting the market are not clearly defined in the book: authors say about the use of trailing, about closing on the next morning and about exit higher then the first trading day High.

The rules for sell entries are similar - LBR/RSI should be within overbought area (higher than 70), a pending order should be placed at the Low of the first hourly bar.

![](https://c.mql5.com/2/30/im3__3.png)

![](https://c.mql5.com/2/30/im4__2.png)

### LBR/RSI indicator

Of course, all the computations necessary for receiving a signal may be performed in the very signal module, but apart from automated trading the plan of this article provides manual trading as well. Having a separate indicator LBR/RSI with highlighting of overbought/oversold areas will be useful for convenience of visual identification of manual version pattern. And in order to optimize our efforts, we will not program two several versions of LBR/RSI estimation (‘buffer’ one for indicator and ‘bufferless’ for robot). Let’s connect an external indicator to the signal module through the standard [iCustom](https://www.mql5.com/en/docs/indicators/icustom "iCustom") function. This indicator will not carry out resource-intensive assessments and there is no need to question it at each tick - in the TS, the value of the indicator at closed daily bar is used. We do not care about a continuously altering current value. Therefore, there are no substantial hindrances for such solution.

Here, unite computing algorithms [ROC](https://www.mql5.com/en/code/46 "Speed of price change") and [RSI](https://www.mql5.com/en/code/7898 "Relative strength index  "), which will paint the resulting oscillator curve. In order to detect needed values easily, add filling of overbought and oversold areas in various colors. For doing this, we need five buffers to display and four more - for auxiliary computations.

Add standard settings (RSI period and values of borders of two areas) with another one which is not provided by original trading system rules. For calculations, you will be able to use not only daily bar closing price, but also more informative median, typical or weighted average price. Actually, for his experiments the user can choose between any of the seven variants provided by [ENUM\_APPLIED\_PRICE](https://www.mql5.com/en/docs/constants/indicatorconstants/prices "Price constants").

Declaration of buffers, user text boxes and initialization block will look as follows:

```
#property indicator_separate_window
#property indicator_buffers  9
#property indicator_plots    3
#property indicator_label1  “Overbought area"
#property indicator_type1   DRAW_FILLING
#property indicator_color1  C'255,208,234'
#property indicator_width1  1
#property indicator_label2  “Oversold area"
#property indicator_type2   DRAW_FILLING
#property indicator_color2  C'179,217,255'
#property indicator_width2  1
#property indicator_label3  "RSI от ROC"
#property indicator_type3   DRAW_LINE
#property indicator_style3  STYLE_SOLID
#property indicator_color3  clrTeal
#property indicator_width3  2
#property indicator_minimum 0
#property indicator_maximum 100
input ENUM_APPLIED_PRICE  TS_MomPin_Applied_Price = PRICE_CLOSE;  // Price for ROC calculation
input uint    TS_MomPin_RSI_Period = 3;                           // RSI Period
input double  TS_MomPin_RSI_Overbought = 70;                      // RSI Oversold level
input double  TS_MomPin_RSI_Oversold = 30;                        // RSI overbought level
double
  buff_Overbought_High[], buff_Overbought_Low[],                  // overbought area background
  buff_Oversold_High[], buff_Oversold_Low[],                      // oversold area background
  buff_Price[],                                                   // array of bar calculated prices
  buff_ROC[],                                                     // ROC array from calculated prices
  buff_RSI[],                                                     // RSI from ROC
  buff_Positive[], buff_Negative[]                                // auxiliary arrays for RSI calculation
;
int OnInit() {
  // designation of indicator buffers:

  // overbought area
  SetIndexBuffer(0, buff_Overbought_High, INDICATOR_DATA);
    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    PlotIndexSetInteger(0, PLOT_SHOW_DATA, false);
  SetIndexBuffer(1, buff_Overbought_Low, INDICATOR_DATA);

  // oversold area
  SetIndexBuffer(2, buff_Oversold_High, INDICATOR_DATA);
    PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    PlotIndexSetInteger(1, PLOT_SHOW_DATA, false);
  SetIndexBuffer(3, buff_Oversold_Low, INDICATOR_DATA);

  // RSI curve
  SetIndexBuffer(4, buff_RSI, INDICATOR_DATA);
    PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);

  // auxiliary buffers for RSI calculation
  SetIndexBuffer(5, buff_Price, INDICATOR_CALCULATIONS);
  SetIndexBuffer(6, buff_ROC, INDICATOR_CALCULATIONS);
  SetIndexBuffer(7, buff_Negative, INDICATOR_CALCULATIONS);
  SetIndexBuffer(8, buff_Positive, INDICATOR_CALCULATIONS);

  IndicatorSetInteger(INDICATOR_DIGITS, 2);
  IndicatorSetString(INDICATOR_SHORTNAME, "LBR/RSI");

  return(INIT_SUCCEEDED);
  }
```

In the standard event handler OnCalculate, arrange two separate loops: the first one prepares ROC data array, the second one calculates oscillator values based on this array data.

In Rates Of Change version suggested by Linda Raschke, we should compare not the prices of bars following each other, but missing one bar between them. In other words, in the TS, they use price variations of the days standing off from the trading day one and three business days respectively. This is rather easy to do; along the way, in this loop arrange background filling of overbought and oversold areas. Also, make sure to implement the price type selection feature:

```
int
  i_RSI_Period = int(TS_MomPin_RSI_Period),         // transfer of RSI period into int type
  i_Bar, i_Period_Bar                               // two bar indices for simultaneous application
;
double
  d_Sum_Negative, d_Sum_Positive,                   // auxiliary variables for RSI calculation
  d_Change                                          // auxiliary variable for ROC calculation
;
// Fill in ROC buffer and fill areas:
i_Period_Bar = 1;
while(++i_Period_Bar < rates_total && !IsStopped()) {
// calculated bar price:
  switch(TS_MomPin_Applied_Price) {
    case PRICE_CLOSE:     buff_Price[i_Period_Bar] = Close[i_Period_Bar]; break;
    case PRICE_OPEN:      buff_Price[i_Period_Bar] = Open[i_Period_Bar]; break;
    case PRICE_HIGH:      buff_Price[i_Period_Bar] = High[i_Period_Bar]; break;
    case PRICE_LOW:       buff_Price[i_Period_Bar] = Low[i_Period_Bar]; break;
    case PRICE_MEDIAN:    buff_Price[i_Period_Bar] = 0.50000 * (High[i_Period_Bar] + Low[i_Period_Bar]); break;
    case PRICE_TYPICAL:   buff_Price[i_Period_Bar] = 0.33333 * (High[i_Period_Bar] + Low[i_Period_Bar] + Open[i_Period_Bar]); break;
    case PRICE_WEIGHTED:  buff_Price[i_Period_Bar] = 0.25000 * (High[i_Period_Bar] + Low[i_Period_Bar] + Open[i_Period_Bar] + Open[i_Period_Bar]); break;
  }
  // difference of bar calculated prices (ROC value):
  if(i_Period_Bar > 1) buff_ROC[i_Period_Bar] = buff_Price[i_Period_Bar] - buff_Price[i_Period_Bar - 2];

  // background filling:
  buff_Overbought_High[i_Period_Bar] = 100;
  buff_Overbought_Low[i_Period_Bar] = TS_MomPin_RSI_Overbought;
  buff_Oversold_High[i_Period_Bar] = TS_MomPin_RSI_Oversold;
  buff_Oversold_Low[i_Period_Bar] = 0;
    }
```

The second loop (RSI calculation) has no peculiarities, it almost completely repeats the algorithm of a standard oscillator of this type:

```
i_Period_Bar = prev_calculated - 1;
if(i_Period_Bar <= i_RSI_Period) {
  buff_RSI[0] = buff_Positive[0] = buff_Negative[0] = d_Sum_Positive = d_Sum_Negative = 0;
  i_Bar = 0;
  while(i_Bar++ < i_RSI_Period) {
    buff_RSI[0] = buff_Positive[0] = buff_Negative[0] = 0;
    d_Change = buff_ROC[i_Bar] - buff_ROC[i_Bar - 1];
    d_Sum_Positive += (d_Change > 0 ? d_Change : 0);
    d_Sum_Negative += (d_Change < 0 ? -d_Change : 0);
  }
  buff_Positive[i_RSI_Period] = d_Sum_Positive / i_RSI_Period;
  buff_Negative[i_RSI_Period] = d_Sum_Negative / i_RSI_Period;

  if(buff_Negative[i_RSI_Period] != 0)
    buff_RSI[i_RSI_Period] = 100 - (100 / (1. + buff_Positive[i_RSI_Period] / buff_Negative[i_RSI_Period]));
  else
    buff_RSI[i_RSI_Period] = buff_Positive[i_RSI_Period] != 0 ? 100 : 50;

  i_Period_Bar = i_RSI_Period + 1;
}

i_Bar = i_Period_Bar - 1;
while(++i_Bar < rates_total && !IsStopped()) {
  d_Change = buff_ROC[i_Bar] - buff_ROC[i_Bar - 1];

  buff_Positive[i_Bar] = (buff_Positive[i_Bar - 1] * (i_RSI_Period - 1) + (d_Change> 0 ? d_Change : 0)) / i_RSI_Period;
  buff_Negative[i_Bar] = (buff_Negative[i_Bar - 1] * (i_RSI_Period - 1) + (d_Change <0 ? -d_Change : 0)) / i_RSI_Period;

  if(buff_Negative[i_Bar] != 0)
    buff_RSI[i_Bar] = 100 - 100. / (1. + buff_Positive[i_Bar] / buff_Negative[i_Bar]);
  else
    buff_RSI[i_Bar] = buff_Positive[i_Bar] != 0 ? 100 : 50;
}
```

Let’s name the indicator LBR\_RSI.mq5 and place it into a standard indicator folder of the terminal data catalog. It is this name that will be specified in the iCustom function of the signal module, therefore you should not change it.

### Signal module

In the signal module connected to the Expert Advisor and indicator, place user settings of Momentum Pinball trading strategy. The authors provide fixed values for calculating the LBR/RSI indicator (period RSI = 3, overbought level = 30, oversold level = 70). But we will make them changeable for experiments, just like position closing methods - the book mentions three variants. We will program all of them and the user will have a feature of selecting the required option:

- to close position by Stop Loss level trailing;
- to close it in the morning of the following day;
- to wait on the second day for breakthrough of extremum of the position opening day.

“Morning” is a rather intangible notion, to formalize rules a more certain definition is required. Raschke and Connors do not say about it, but it is reasonable to suppose that binding to new day first bar (applied in other TS rules) will point to ‘morning’ label of 24 hrs time scale.

Remember about two more TS settings - offsets from borders of the first hour of a day; the offsets should specify levels of pending order placement and StopLoss level:

```
enum ENUM_EXIT_MODE {     // List of exit methods
  CLOSE_ON_SL_TRAIL,      // only by trailing
  CLOSE_ON_NEW_1ST_CLOSE, // by closing of the 1st bar of the following day
  CLOSE_ON_DAY_BREAK      // by break-through of extremum of the position opening day
};
// user settings
input ENUM_APPLIED_PRICE  TS_MomPin_Applied_Price = PRICE_CLOSE;     // Momentum Pinball: Prices for ROC calculation
input uint    TS_MomPin_RSI_Period = 3;                              // Momentum Pinball: RSI period
input double  TS_MomPin_RSI_Overbought = 70;                         // Momentum Pinball: RSI oversold level
input double  TS_MomPin_RSI_Oversold = 30;                           // Momentum Pinball: RSI overbought level
input uint    TS_MomPin_Entry_Offset = 10;                           // Momentum Pinball: Offset of entry level from borders H1 (in points)
input uint    TS_MomPin_Exit_Offset = 10;                            // Momentum Pinball: Offset of exit level from borders H1 (in points)
  input ENUM_EXIT_MODE  TS_MomPin_Exit_Mode = CLOSE_ON_SL_TRAIL;       // Momentum Pinball: Profitable position closing method
```

The main module function fe\_Get\_Entry\_Signal will be unified with the signal module function [of the previous trading strategy](https://www.mql5.com/en/articles/2785#para3 "TS '80-20'") from Raschke and Connors book, as well with subsequent analogic modules of other TS described in this source. This means that the function should have such package of parameters passed to it, links to variables and the same type of returned value:

```
ENUM_ENTRY_SIGNAL fe_Get_Entry_Signal(      // Two-candle pattern analysis (D1 + H1)
  datetime  t_Time,                         // current time
  double&    d_Entry_Level,                 // entry level (link to the variable)
  double&    d_SL,                          // StopLoss level (link to the variable)
  double&    d_TP,                          // TakeProfit level (link to the variable)
  double&    d_Range_High,                  // high of the range's 1st hourly bar (link to the variable)
  double&    d_Range_Low                    // low of the range's 1st hourly bar (link to the variable)
) {
  // function body
  }
```

Like in the previous version, we will not calculate everything once again at each tick when calling the function from robot. Instead, we will store between ticks the calculated levels in static variables. However, working with this function in manual trading indicator will have substantial differences; and zeroing of static variables at calling the function from indicator should be provided. In order to distinguish between a call from indicator and a call from robot, apply t\_Time variable. The indicator will invert it, i.e. make its value negative:

```
static ENUM_ENTRY_SIGNAL se_Trade_Direction = ENTRY_UNKNOWN;   // trading direction for today
static double
  // variables for storing calculated levels between ticks
  sd_Entry_Level = 0,
  sd_SL = 0, sd_TP = 0,
  sd_Range_High = 0, sd_Range_Low = 0
;
if(t_Time < 0) {                                               // only for call from indicator
  sd_Entry_Level = sd_SL = sd_TP = sd_Range_High = sd_Range_Low = 0;
  se_Trade_Direction = ENTRY_UNKNOWN;
}
// by default apply earlier saved levels of entries/exits:
    d_Entry_Level = sd_Entry_Level; d_SL = sd_SL; d_TP = sd_TP; d_Range_High = sd_Range_High; d_Range_Low = sd_Range_Low;
```

Below, find the code for receiving the LBR/RSI indicator handle when calling the function for the first time:

```
static int si_Indicator_Handle = INVALID_HANDLE;
if(si_Indicator_Handle == INVALID_HANDLE) {
  // to receive indicator handle when calling the function for the first time:
  si_Indicator_Handle = iCustom(_Symbol, PERIOD_D1, "LBR_RSI",
    TS_MomPin_Applied_Price,
    TS_MomPin_RSI_Period,
    TS_MomPin_RSI_Overbought,
    TS_MomPin_RSI_Oversold
  );

  if(si_Indicator_Handle == INVALID_HANDLE) { // indicator handle not received
    if(Log_Level > LOG_LEVEL_NONE) PrintFormat("%s: error of receiving LBR_RSI indicator handle #%u", __FUNCTION__, _LastError);
    return(ENTRY_INTERNAL_ERROR);
  }
  }
```

Once per 24 hours robot should analyze indicator value on the last closed daily bar and establish the trading direction allowed for today. Or it should disable trading if LBR/RSI value is in neutral area. The retrieval code of this value from indicator buffer and its analysis, with logging functions, subject to possible errors and peculiarities of calling from manual trading indicator:

```
static int si_Indicator_Handle = INVALID_HANDLE;
if(si_Indicator_Handle == INVALID_HANDLE) {
  // receiving indicator handle at first function call:
  si_Indicator_Handle = iCustom(_Symbol, PERIOD_D1, "LBR_RSI",
    TS_MomPin_Applied_Price,
    TS_MomPin_RSI_Period,
    TS_MomPin_RSI_Overbought,
    TS_MomPin_RSI_Oversold
  );

  if(si_Indicator_Handle == INVALID_HANDLE) {       // handle not received
    if(Log_Level > LOG_LEVEL_NONE) PrintFormat("%s: error of indicator handle receipt LBR_RSI #%u", __FUNCTION__, _LastError);
    return(ENTRY_INTERNAL_ERROR);
  }
}
// to find out the time of previous day daily bar:
datetime ta_Bar_Time[];
if(CopyTime(_Symbol, PERIOD_D1, fabs(t_Time), 2, ta_Bar_Time) < 2) {
  if(Log_Level > LOG_LEVEL_NONE) PrintFormat("%s: CopyTime: error #%u", __FUNCTION__, _LastError);
  return(ENTRY_INTERNAL_ERROR);
}
// previous day analysis, if this is the 1st call today:
static datetime st_Prev_Day = 0;
if(t_Time < 0) st_Prev_Day = 0;                     // only for call from indicator
if(st_Prev_Day < ta_Bar_Time[0]) {
  // zeroing of previous day parameters:
  se_Trade_Direction = ENTRY_UNKNOWN;
  d_Entry_Level = sd_Entry_Level = d_SL = sd_SL = d_TP = sd_TP = d_Range_High = sd_Range_High = d_Range_Low = sd_Range_Low = 0;

  // retrieve value LBR/RSI of previous day:
  double da_Indicator_Value[];
  if(1 > CopyBuffer(si_Indicator_Handle, 4, ta_Bar_Time[0], 1, da_Indicator_Value)) {
    if(Log_Level > LOG_LEVEL_NONE) PrintFormat("%s: CopyBuffer: error #%u", __FUNCTION__, _LastError);
    return(ENTRY_INTERNAL_ERROR);
  }

  // if anything is wrong with LBR/RSI value:
  if(da_Indicator_Value[0] > 100. || da_Indicator_Value[0] < 0.) {
    if(Log_Level > LOG_LEVEL_NONE) PrintFormat("%s: Indicator buffer value error (%f)", __FUNCTION__, da_Indicator_Value[0]);
    return(ENTRY_UNKNOWN);
  }

  st_Prev_Day = ta_Bar_Time[0];                     // attempt counted

  // remember trading direction for today:
  if(da_Indicator_Value[0] > TS_MomPin_RSI_Overbought) se_Trade_Direction = ENTRY_SELL;
  else se_Trade_Direction = da_Indicator_Value[0] > TS_MomPin_RSI_Oversold ? ENTRY_NONE : ENTRY_BUY;

  // to log:
  if(Log_Level == LOG_LEVEL_DEBUG) PrintFormat("%s: Trading direction for %s: %s. LBR/RSI: (%.2f)",
    __FUNCTION__,
    TimeToString(ta_Bar_Time[1], TIME_DATE),
    StringSubstr(EnumToString(se_Trade_Direction), 6),
    da_Indicator_Value[0]
  );
  }
```

We have clarified the allowed trading direction. The next task will be determining entry levels and loss limitation (Stop Loss). It is also sufficient to do it once per 24 hours - right after closing the date first bar on hourly timeframe. However, subject to peculiarities of manual trading indicator functioning, we will complicate the algorithm a little. This is caused by the fact that indicator should not only detect real-time signal levels, but also to make marks on history:

```
// no signal search today
if(se_Trade_Direction == ENTRY_NONE) return(ENTRY_NONE);
// analysis of today’s first bar H1, unless this is already done:
if(sd_Entry_Level == 0.) {
  // to receive data of last 24 bars H1:
  MqlRates oa_H1_Rates[];
  int i_Price_Bars = CopyRates(_Symbol, PERIOD_H1, fabs(t_Time), 24, oa_H1_Rates);
  if(i_Price_Bars == WRONG_VALUE) {                      // handling of CopyRates function error
    if(Log_Level > LOG_LEVEL_NONE) PrintFormat("%s: CopyRates: error #%u", __FUNCTION__, _LastError);
    return(ENTRY_INTERNAL_ERROR);
  }

  // among 24 bars to find the 1st bar of today and to remember High, Low:
  int i_Bar = i_Price_Bars;
  while(i_Bar-- > 0) {
    if(oa_H1_Rates[i_Bar].time < ta_Bar_Time[1]) break;      // last bar H1 of previous day

    // borders of H1 1st bar range:
    sd_Range_High = d_Range_High = oa_H1_Rates[i_Bar].high;
    sd_Range_Low = d_Range_Low = oa_H1_Rates[i_Bar].low;
  }
  // H1 1st bar is not closed yet:
  if(i_Price_Bars - i_Bar < 3) return(ENTRY_UNKNOWN);

  // to calculate trading levels:

  // level of market entry:
  d_Entry_Level = _Point * TS_MomPin_Entry_Offset;           // auxiliary calculations
  sd_Entry_Level = d_Entry_Level = se_Trade_Direction == ENTRY_SELL ? d_Range_Low - d_Entry_Level : d_Range_High + d_Entry_Level;
  // initial level SL:
  d_SL = _Point * TS_MomPin_Exit_Offset;                     // auxiliary calculations
  sd_SL = d_SL = se_Trade_Direction == ENTRY_BUY ? d_Range_Low - d_SL : d_Range_High + d_SL;
  }
```

After that, we only should stop the function by returning the detected trading direction:

return(se\_Trade\_Direction);

Now, let’s program analysis of conditions for position signal closing. We have three variants, one of which (Stop Loss level trailing) is already realized in the Expert Advisor code of previous versions. Two other variants, in the aggregate, require price and time of entry, position direction for calculations. We will pass them together with the current time and selected closing method to the function fe\_Get\_Exit\_Signal:

```
ENUM_EXIT_SIGNAL fe_Get_Exit_Signal(    // Detection of position closing signal
  double            d_Entry_Level,      // entry level
  datetime          t_Entry_Time,       // entry time
  ENUM_ENTRY_SIGNAL e_Trade_Direction,  // trade direction
  datetime          t_Current_Time,     // current time
  ENUM_EXIT_MODE    e_Exit_Mode         // exit mode
) {
  static MqlRates soa_Prev_D1_Rate[];   // data of D1 bar for previous day
  static int si_Price_Bars = 0;         // auxiliary counter
  if(t_Current_Time < 0) {              // to distinguish a call from indicator and a call from Expert Advisor
    t_Current_Time = -t_Current_Time;
    si_Price_Bars = 0;
  }
  double
    d_Curr_Entry_Level,
    d_SL, d_TP,
    d_Range_High,  d_Range_Low
  ;

  if(e_Trade_Direction < 1) {          // no positions, to zero everything
    si_Price_Bars = 0;
  }

  switch(e_Exit_Mode) {
    case CLOSE_ON_SL_TRAIL:            // only on trail
            return(EXIT_NONE);

    case CLOSE_ON_NEW_1ST_CLOSE:       // on closing of next day 1st bar
            if((t_Current_Time - t_Current_Time % 86400)
              ==
              (t_Entry_Time - t_Current_Time % 86400)
            ) return(EXIT_NONE);       // day of position opening not finished yet

            if(fe_Get_Entry_Signal(t_Current_Time, d_Curr_Entry_Level, d_SL, d_TP, d_Range_High, d_Range_Low)
              < ENTRY_UNKNOWN
            ) {
              if(Log_Level > LOG_LEVEL_ERR) PrintFormat("%s: 1st bar of the following day is closed", __FUNCTION__);
              return(EXIT_ALL);
            }
            return(EXIT_NONE);         // not closed

    case CLOSE_ON_DAY_BREAK:           // upon break-through of extremum of the position opening day
            if((t_Current_Time - t_Current_Time % 86400)
              ==
              (t_Entry_Time - t_Current_Time % 86400)
            ) return(EXIT_NONE);       // position opening day not finished yet

            if(t_Current_Time % 86400 > 36000) return(EXIT_ALL); // time out

            if(si_Price_Bars < 1) {
              si_Price_Bars = CopyRates(_Symbol, PERIOD_D1, t_Current_Time, 2, soa_Prev_D1_Rate);
              if(si_Price_Bars == WRONG_VALUE) { // handling of CopyRates function error
                if(Log_Level > LOG_LEVEL_NONE) PrintFormat("%s: CopyRates: error #%u", __FUNCTION__, _LastError);
                return(EXIT_UNKNOWN);
              }

              if(e_Trade_Direction == ENTRY_BUY) {
                if(soa_Prev_D1_Rate[1].high < soa_Prev_D1_Rate[0].high) return(EXIT_NONE);        // did not break-through

                if(Log_Level > LOG_LEVEL_ERR) PrintFormat("%s: price broke-through yesterday’s High: %s > %s", __FUNCTION__, DoubleToString(soa_Prev_D1_Rate[1].high, _Digits), DoubleToString(soa_Prev_D1_Rate[0].high, _Digits));
                return(EXIT_BUY);
              } else {
                if(soa_Prev_D1_Rate[1].low > soa_Prev_D1_Rate[0].low) return(EXIT_NONE);          // did not break through

                if(Log_Level > LOG_LEVEL_ERR) PrintFormat("%s: price broke through yesterday’s Low: %s < %s", __FUNCTION__, DoubleToString(soa_Prev_D1_Rate[1].low, _Digits), DoubleToString(soa_Prev_D1_Rate[0].low, _Digits));
                return(EXIT_SELL);
              }
            }

            return(EXIT_NONE); // for each
  }

  return(EXIT_UNKNOWN);
  }
```

Here, we have a ‘cap’ in case if ‘trailing exit’ option is selected - the function returns signal absence without any analysis. For two other options, occurrence of events ‘morning has come’ and ‘yesterday’s extremum is broken through’ is identified. Variants of the values of ENUM\_EXIT\_SIGNAL type returned by function are very similar to the analogous list of entry signal values (ENUM\_ENTRY\_SIGNAL):

```
enum ENUM_EXIT_SIGNAL {  // The list of exit signals
  EXIT_UNKNOWN,          // not identified
  EXIT_BUY,              // close buys
  EXIT_SELL,             // close sells
  EXIT_ALL,              // close all
  EXIT_NONE              // close nothing
  };
```

### Indicator for manual trading

The above described signal module should be applied in the robot for automated trading. Let us detail this application method below. First, let us create a tool for more explicit consideration of TS peculiarities at charts in the terminal. This will be an indicator applying the signal module without any changes and displaying the trade levels calculated in it - pending order placement level and Stop Loss level. Closing a profitable deal in this indicator will be provided only by one simplified variant - when the preset level (TakeProfit) is reached. As you remember, in the module we have programmed more complicated algorithms for detecting deal exit signals, but let’s leave them for implementation in robot.

In addition to trade levels, the indicator will fill bars of the first hour of a day, in order to make it clear why it is these levels that are applied. Such marking will help visually evaluate advantages and disadvantages of most rules of the Momentum Pinball strategy - to discover things that cannot be obtained from strategy tester reports. Visual analysis added with tester statistics will facilitate making TS rules more efficient.

In order to apply indicator for ordinary manual trading, let’s add a real-time trader notification system to it. Such notification will contain the entry direction recommended by signal module jointly with placement levels of pending order and emergency exit (Stop Loss). There will be three ways of notice delivery - a standard pop-up window with text and sound signal, e-mail message and push notification to the mobile phone.

All the requirements to indicator are listed. Thus, we may proceed to programming. In order to draw on the chart all the objects we planned, the indicator should have: one buffer of DRAW\_FILLING type (to fill bar range of the first hour of a day) and three buffers to display trade levels (entry level, take profit level, stop loss level). One of them (pending order placement level) should have a feature to change color (DRAW\_COLOR\_LINE type) depending on trading direction, for two others it is enough to have a single-color type DRAW\_LINE:

```
#property indicator_chart_window
#property indicator_buffers  6
#property indicator_plots    4
#property indicator_label1  “1st hour of a day"
#property indicator_type1   DRAW_FILLING
#property indicator_color1  C'255,208,234', C'179,217,255'
#property indicator_width1  1
#property indicator_label2  “Entry level"
#property indicator_type2   DRAW_COLOR_LINE
#property indicator_style2  STYLE_DASHDOT
#property indicator_color2  clrDodgerBlue, clrDeepPink
#property indicator_width2  2
#property indicator_label3  "Stop Loss"
#property indicator_type3   DRAW_LINE
#property indicator_style3  STYLE_DASHDOTDOT
#property indicator_color3  clrCrimson
#property indicator_width3  1
#property indicator_label4  "Take Profit"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrGreen
  #property indicator_width4  1
```

Now, declare lists, part of which are not required in indicator (they are used only by Expert Advisor), but they are engaged in signal module functions. These variables of enum type are required for working with logging and various methods of position closing; we will omit them in indicator also - let me remind that here we have only imitation of simple take profit at preset level (Take Profit). Following declaration of those variables we may connect the external module, list user settings and declare global variables:

```
enum ENUM_LOG_LEVEL {  // The list of logging levels
  LOG_LEVEL_NONE,      // logging disabled
  LOG_LEVEL_ERR,       // only info about errors
  LOG_LEVEL_INFO,      // errors + robot comments
  LOG_LEVEL_DEBUG      // all without exclusions
};
enum ENUM_ENTRY_SIGNAL {  // List of entry signals
  ENTRY_BUY,              // buy signal
  ENTRY_SELL,             // sell signal
  ENTRY_NONE,             // no signal
  ENTRY_UNKNOWN,          // status indefinite
  ENTRY_INTERNAL_ERROR    // internal function error
};
enum ENUM_EXIT_SIGNAL {  // The list of exit signals
  EXIT_UNKNOWN,          // not identified
  EXIT_BUY,              // close buys
  EXIT_SELL,             // close sells
  EXIT_ALL,              // close all
  EXIT_NONE              // close nothing
};
#include <Expert\Signal\Signal_Momentum_Pinball.mqh>     // signal module of ‘Momentum Pinball’ TS
input uint    TS_MomPin_Take_Profit = 10;                // Momentum Pinball: Take Profit (in points)
input bool    Show_1st_H1_Bar = true;                    // Show day 1st hourly bar range?
input bool    Alert_Popup = true;                        // Alert: Show pop-up window?
input bool    Alert_Email = false;                       // Alert: Send e-mail?
input string  Alert_Email_Subj = "";                     // Alert: Subjects of e-mail alert
input bool    Alert_Push = true;                         // Alert: Send push-notification?
input uint  Days_Limit = 7;                              // History layout depth (calendar days)
ENUM_LOG_LEVEL  Log_Level = LOG_LEVEL_DEBUG;             // Logging mode
double
  buff_1st_H1_Bar[], buff_1st_H1_Bar_Zero[],             // buffers for filling day 1st hourly bar range
  buff_Entry[], buff_Entry_Color[],                      // buffers of pending order line
  buff_SL[],                                             // buffer of StopLoss line
  buff_TP[],                                             // buffer of TakeProfit line
  gd_Entry_Offset = 0,                                   // TS_MomPin_Entry_Offset in symbol prices
  gd_Exit_Offset = 0                                     // TS_MomPin_Exit_Offset in symbol prices
  ;
```

Initialization function has nothing remarkable - here, assign indices of indicator buffers to the arrays declared for these buffers. As well, here let’s convert user settings from points to symbol prices; this should be done to reduce resource consumption at least a little and not to perform such conversion thousands of times during the run of the main program:

```
int OnInit() {
  // converting points to symbol prices:
  gd_Entry_Offset = TS_MomPin_Entry_Offset * _Point;
  gd_Exit_Offset = TS_MomPin_Exit_Offset * _Point;

  // designation of indicator buffers:

  // day 1st hourly bar range rectangle
  SetIndexBuffer(0, buff_1st_H1_Bar, INDICATOR_DATA);
    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, 0);
  SetIndexBuffer(1, buff_1st_H1_Bar_Zero, INDICATOR_DATA);
    PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, 0);

  // pending order placement line
  SetIndexBuffer(2, buff_Entry, INDICATOR_DATA);
    PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, 0);
  SetIndexBuffer(3, buff_Entry_Color, INDICATOR_COLOR_INDEX);

  // line SL
  SetIndexBuffer(4, buff_SL, INDICATOR_DATA);
    PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, 0);

  // line TP
  SetIndexBuffer(5, buff_TP, INDICATOR_DATA);
    PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, 0);

  IndicatorSetInteger(INDICATOR_DIGITS, _Digits);
  IndicatorSetString(INDICATOR_SHORTNAME, "Momentum Pinball");

  return(INIT_SUCCEEDED);
  }
```

In the indicator code of [this series' previous article](https://www.mql5.com/en/articles/2785), some program structure was created, its designation is to store information of any type between ticks. You may read there why it is needed and how it is built here. We will just engage it without any modifications. In this indicator version out of the complete function set of ‘Brownie’, only the flag of the beginning of a new bar will be engaged. But if you want to make the manual trading indicator more advanced, other ‘Brownie’ features will be of use. The complete code of go\_Brownie is provided in the end of indicator’s source code file (TS\_Momentum\_Pinball.mq5) attached to this article. As well, there you can see the notification-push function code f\_Do\_Alert — it also has no modifications if compared to the previous indicator of this series of articles, therefore, there is no necessity to consider it in detail.

Inside the standard tick-receiving event handler (OnCalculate), prior to the beginning of the program main loop, the necessary variables must be declared. If this is not the first call of the main loop, the re-calculation range should be limited only by currently actual bars - for this trading strategy these are yesterday’s and today’s bars. If this is the first loop call after initialization, clearing of indicator buffers from remaining data should be arranged. If this is not done, already not-actual areas will remain filled when switching a timeframe. Besides, calling of the main function should be limited once per one bar. It is convenient to do it using the go\_Brownie structure:

```
go_Brownie.f_Update(prev_calculated, prev_calculated);     // “feed” data to Brownie
datetime t_Time = TimeCurrent();                           // last known server time
int
  i_Period_Bar = 0,                                        // auxiliary counter
  i_Current_TF_Bar = 0                                     // loop beginning bar index
;
if(go_Brownie.b_First_Run) {                               // if this is the 1st launch
  i_Current_TF_Bar = rates_total — Bars(_Symbol, PERIOD_CURRENT, t_Time — t_Time % 86400 — 86400 * Days_Limit, t_Time);
  // clearing buffer at re-initialization:
  ArrayInitialize(buff_1st_H1_Bar, 0); ArrayInitialize(buff_1st_H1_Bar_Zero, 0);
  ArrayInitialize(buff_Entry, 0); ArrayInitialize(buff_Entry_Color, 0);
  ArrayInitialize(buff_TP, 0);
  ArrayInitialize(buff_SL, 0);
} else if(!go_Brownie.b_Is_New_Bar) return(rates_total);   // waiting for bar closing
else {                                                     // new bar
  // minimum re-calculation depth - from day beginning:
  i_Current_TF_Bar = rates_total — Bars(_Symbol, PERIOD_CURRENT, t_Time — t_Time % 86400, t_Time);
}
ENUM_ENTRY_SIGNAL e_Entry_Signal = ENTRY_UNKNOWN;          // entry signal
double
  d_SL = WRONG_VALUE,                                      // SL level
  d_TP = WRONG_VALUE,                                      // TP level
  d_Entry_Level = WRONG_VALUE,                             // entry level
  d_Range_High = WRONG_VALUE, d_Range_Low = WRONG_VALUE    // pattern's 1 st bar range borders
;
datetime
  t_Curr_D1_Bar = 0,                                       // current D1 bar time (pattern's 2 nd bar)
  t_Last_D1_Bar = 0,                                       // time of the last bar D1, on which signal was available
  t_Entry_Bar = 0                                          // pending order placement bar time
;
// making sure that initial re-calculation bar index is within allowed frames:
  i_Current_TF_Bar = int(fmax(0, fmin(i_Current_TF_Bar, rates_total — 1)));
```

Now, let us program the main working loop. In the beginning of each iteration, we should receive data from signal module, control performance, availability of errors and arrange transition to the next loop iteration, if no signal is available:

```
while(++i_Current_TF_Bar < rates_total && !IsStopped()) {                // iterate over the current TF bars
  // receiving data from signal module:
  e_Entry_Signal = fe_Get_Entry_Signal(-Time[i_Current_TF_Bar], d_Entry_Level, d_SL, d_TP, d_Range_High, d_Range_Low);
  if(e_Entry_Signal == ENTRY_INTERNAL_ERROR) {                           // error of data copying from external indicator buffer
    // calculations and drawing should be repeated on the next tick:
    go_Brownie.f_Reset();
    return(rates_total);
  }
    if(e_Entry_Signal > 1) continue;                                       // no active signal on this bar
```

If the module detected a signal on the bar in question and returned the estimated entry level, first calculate the take profit level (Take Profit):

t\_Curr\_D1\_Bar = Time\[i\_Current\_TF\_Bar\] - Time\[i\_Current\_TF\_Bar\] % 86400; // start of the day the bar belongs to

And then this trade in process will be plotted, laid out on history, if this is the first bar of a new day:

```
t_Curr_D1_Bar = Time[i_Current_TF_Bar] - Time[i_Current_TF_Bar] % 86400;            // start of the day the bar belongs to
if(t_Last_D1_Bar < t_Curr_D1_Bar) {                                                 // this is 1st bar of the day, on which signal is available
    t_Entry_Bar = Time[i_Current_TF_Bar];                                             // remember the time of trading start
```

Starting with background filling of day’s first hour bars, used in level calculations:

```
// Background filling 1st hour bars:
if(Show_1st_H1_Bar) {
  i_Period_Bar = i_Current_TF_Bar;
  while(Time[--i_Period_Bar] >= t_Curr_D1_Bar && i_Period_Bar > 0)
    if(e_Entry_Signal == ENTRY_BUY) {                               // bullish pattern

      buff_1st_H1_Bar_Zero[i_Period_Bar] = d_Range_High;
      buff_1st_H1_Bar[i_Period_Bar] = d_Range_Low;
    } else {                                                        // bearish pattern
      buff_1st_H1_Bar[i_Period_Bar] = d_Range_High;
      buff_1st_H1_Bar_Zero[i_Period_Bar] = d_Range_Low;
    }
  }
```

Then, draw the pending order placement line until the moment when the pending order becomes an open position, i.e. when the price touches this level:

```
// Entry line till crossed by a bar:
i_Period_Bar = i_Current_TF_Bar - 1;
if(e_Entry_Signal == ENTRY_BUY) {                               // bullish pattern
  while(++i_Period_Bar < rates_total) {
    if(Time[i_Period_Bar] > t_Curr_D1_Bar + 86399) {            // day end
      e_Entry_Signal = ENTRY_NONE;                              // pending order did not trigger
      break;
    }

    // extend line:
    buff_Entry[i_Period_Bar] = d_Entry_Level;
    buff_Entry_Color[i_Period_Bar] = 0;

    if(d_Entry_Level <= High[i_Period_Bar]) break;               // entry was on this bar
  }
} else {                                                         // bearish pattern
  while(++i_Period_Bar < rates_total) {
    if(Time[i_Period_Bar] > t_Curr_D1_Bar + 86399) {             // day end
      e_Entry_Signal = ENTRY_NONE;                               // pending order did not trigger
      break;
    }

    // extend line:
    buff_Entry[i_Period_Bar] = d_Entry_Level;
    buff_Entry_Color[i_Period_Bar] = 1;

    if(d_Entry_Level >= Low[i_Period_Bar]) break;               // entry was on this bar
  }
  }
```

If price failed to achieve the calculated level before the day end, proceed to the following step of the main loop:

```
if(e_Entry_Signal == ENTRY_NONE) {                 // pending order did not trigger before the day end
  i_Current_TF_Bar = i_Period_Bar;                 // this day bars are not interesting to us any more
  continue;
  }
```

If this day is not finished yet and pending order future is still indefinite, there is no sense to continue the main program loop:

if(i\_Period\_Bar >= rates\_total - 1) break;        // current (not finished) day is completed

After these two filters, only one possible variant of events remains - pending order triggered. Let’s find pending order execution bar and, starting with this bar, draw Take Profit and Stop Loss levels until one of them is crossed by the price, i.e. until position closing. If position opening and closing take place on a single bar, the line should be extended by one bar to the past so that it is visible on the chart:

```
// order triggered, find position closing bar:
i_Period_Bar = fmin(i_Period_Bar, rates_total - 1);
buff_SL[i_Period_Bar] = d_SL;
while(++i_Period_Bar < rates_total) {
  if(TS_MomPin_Exit_Mode == CLOSE_ON_SL_TRAIL) {
    if(Time[i_Period_Bar] >= t_Curr_D1_Bar + 86400) break;        // this is the following day bar

    // Lines TP and SL until the bar crossing one of them:
    buff_SL[i_Period_Bar] = d_SL;
    buff_TP[i_Period_Bar] = d_TP;

    if((
      e_Entry_Signal == ENTRY_BUY && d_SL >= Low[i_Period_Bar]
      ) || (
      e_Entry_Signal == ENTRY_SELL && d_SL <= High[i_Period_Bar]
    )) {                                                          // SL exit
      if(buff_SL[int(fmax(0, i_Period_Bar - 1))] == 0.) {
   // beginning and end on a single bar, extend it by 1 bar to the past
        buff_SL[int(fmax(0, i_Period_Bar - 1))] = d_SL;
        buff_TP[int(fmax(0, i_Period_Bar - 1))] = d_TP;
      }
      break;
    }

    if((
      e_Entry_Signal == ENTRY_BUY && d_TP <= High[i_Period_Bar]
      ) || (
      e_Entry_Signal == ENTRY_SELL && d_SL >= Low[i_Period_Bar]
    )) {                                                         // TP exit
      if(buff_TP[int(fmax(0, i_Period_Bar - 1))] == 0.) {
        // beginning and end on a single bar, extend it by 1 bar to the past
        buff_SL[int(fmax(0, i_Period_Bar - 1))] = d_SL;
        buff_TP[int(fmax(0, i_Period_Bar - 1))] = d_TP;
      }
      break;
    }
  }
  }
```

After position closing remaining bars of the day may be omitted in program’s main loop:

```
i_Period_Bar = i_Current_TF_Bar;
t_Curr_D1_Bar = Time[i_Period_Bar] - Time[i_Period_Bar] % 86400;
while(
  ++i_Period_Bar < rates_total
  &&
  t_Curr_D1_Bar == Time[i_Period_Bar] - Time[i_Period_Bar] % 86400
  ) i_Current_TF_Bar = i_Period_Bar;
```

Here the main loop code is finished. Now, notification should be arranged if a signal is detected on the current bar:

```
i_Period_Bar = rates_total - 1;                                            // current bar
if(Alert_Popup + Alert_Email + Alert_Push == 0) return(rates_total);       // all disabled
if(t_Entry_Bar != Time[i_Period_Bar]) return(rates_total);                 // no signal on this bar
// message wording:
string s_Message = StringFormat("ТС Momentum Pinball: needed %s @ %s, SL: %s",
  e_Entry_Signal == ENTRY_BUY ? "BuyStop" : "SellStop",
  DoubleToString(d_Entry_Level, _Digits),
  DoubleToString(d_SL, _Digits)
);
// alert:
  f_Do_Alert(s_Message, Alert_Popup, false, Alert_Email, Alert_Push, Alert_Email_Subj);
```

The full indicator code is provided in TS\_Momentum\_Pinball.mq5 file attached below.

### Expert Advisor for testing the Momentum Pinball TS

Features of the basic Expert Advisor should be somewhat extended when preparing for testing of another trading strategy from Raschke Connors book. You can find the source code taken as the basis of this version together with detailed description in the previous article. Here, we will not repeat ourselves, studying only substantial changes and supplements, and there are two of them.

The first supplement - the list of exit signals, which was not available in the previous version of trade robot. Additionally, ENTRY\_INTERNAL\_ERROR status was added to the list of entry signals. These numbered lists do not differ from the same enum lists in the above studied indicator. In the robot code, we place them before the connection string of standard library trading operations class. In Street\_Smarts\_Bot\_MomPin.mq5 file of attachment to the article, these are strings 24..32.

The second change is connected with the fact that the signal module now provides position closing signals as well. Let’s add corresponding code block to work also with this signal. In the previous robot version, there is a conditional operator ‘if’ for checking whether the existing position is new (string 139); the checkup is used to calculate and place StopLoss initial level. In this version, let’s add to ‘if’ operator through alternative ‘else’ a corresponding code block for calling the signal module. If calling result so requires, the Expert Advisor should close a position:

```
} else {                       // not new position
                               // conditions for position closing ready?
  ENUM_EXIT_SIGNAL e_Exit_Signal = fe_Get_Exit_Signal(d_Entry_Level, datetime(PositionGetInteger(POSITION_TIME)), e_Entry_Signal, TimeCurrent(), TS_MomPin_Exit_Mode);
  if((
      e_Exit_Signal == EXIT_BUY && e_Entry_Signal == ENTRY_BUY
    ) || (
      e_Exit_Signal == EXIT_SELL && e_Entry_Signal == ENTRY_SELL
    ) || e_Exit_Signal == EXIT_ALL
  ) {
                              // it must be closed
    CTrade o_Trade;
    o_Trade.LogLevel(LOG_LEVEL_ERRORS);
    o_Trade.PositionClose(_Symbol);
    return;
  }
  }
```

In robot source code, these are strings 171..186.

There are some changes in the code of the function controlling sufficiency of distance to trade levels fb\_Is\_Acceptable\_Distance (strings 424..434).

### Strategy backtesting

We have created a couple of tools (indicator and Expert Advisor) for studying a trading system which became well-known due to a book by L. Raschke and L. Connors. The principle purpose of the Expert Advisor backtesting is to check up viability of trade robot which is one of those tools. Therefore, I did not optimize parameters, testing was performed with default settings.

Full results of all the runs are provided in the attached archive; here, only balance variation charts will be supplied. Only as illustration of the second (by significance) purpose of testing - rough (without parameter optimization) evaluation of TS performance under modern market conditions. Let me remind, that the authors illustrated the strategy by charts from the late last century.

Balance variation chart at Expert Advisor testing from the beginning of 2014 on quotes of MetaQuotes demo server. Symbol — EURJPY, timeframe — H1:

![](https://c.mql5.com/2/30/MP_EURJPY_H1__1.jpg)

Similar chart for EURUSD symbol, same timeframe and same testing period:

![](https://c.mql5.com/2/30/MP_EURUSD_H1__1.jpg)

When testing without setting modification on quotes of one of metals (XAUUSD) for the same period and on the same timeframe, the balance variation chart looks as follows:

![](https://c.mql5.com/2/30/MP_XAUUSD_H1__1.jpg)

### Conclusion

The rules for the Momentum Pinball trading system listed in [Street Smarts: High Probability Short-Term Trading Strategies](https://www.mql5.com/go?link=https://www.amazon.com/Street-Smarts-Probability-Short-Term-Strategies/dp/0965046109 "/go?link=https://www.amazon.com/Street-Smarts-Probability-Short-Term-Strategies/dp/0965046109") are carried to the code of indicator and Expert Advisor. Unfortunately, description is not so detailed as it might be and provides more than one variant for the rules of position tracking and closing. Therefore, for those who want to research trading system peculiarities in detail, there is a rather wide field for selecting optimal parameters and algorithms of robot activities. The created code enables to do it; besides, hopefully, source codes will be useful at studying object-oriented programming.

Source codes, compiled files and library in MQL5.zip are placed in respective catalogs. Designation of each of them:

| # | File name | Type | Description |
| --- | --- | --- | --- |
| 1 | LBR\_RSI.mq5 | indicator | Indicator which consolidated ROC and RSI. Used to determine trade direction (or its disabled status) of a starting day |
| 2 | TS\_Momentum\_Pinball.mq5 | indicator | Indicator for manual trading by this TS. Displays calculated levels of entries and exits, highlights the first hour range, on which basis calculations are performed |
| 3 | Signal\_Momentum\_Pinball.mqh | library | Library of functions, structures and user settings. Used by indicator and Expert Advisor |
| 4 | Street\_Smarts\_Bot\_MomPin.mq5 | Expert Advisor | Expert Advisor for automated trading by this TS |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2825](https://www.mql5.com/ru/articles/2825)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2825.zip "Download all attachments in the single ZIP archive")

[MPtest.zip](https://www.mql5.com/en/articles/download/2825/mptest.zip "Download MPtest.zip")(744.43 KB)

[MQL5.zip](https://www.mql5.com/en/articles/download/2825/mql5.zip "Download MQL5.zip")(163.27 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [80-20 trading strategy](https://www.mql5.com/en/articles/2785)
- [The 'Turtle Soup' trading system and its 'Turtle Soup Plus One' modification](https://www.mql5.com/en/articles/2717)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/224137)**
(5)


![MaPi7475 Pittner](https://c.mql5.com/avatar/2018/1/5A53EA80-8CBE.jpeg)

**[MaPi7475 Pittner](https://www.mql5.com/en/users/mapi7475)**
\|
20 Jan 2018 at 22:26

Hello,

unfortunately the translation did not [work](https://www.mql5.com/en/articles/180 "Article: Averaging of price series without additional buffers for intermediate calculations") completely, because all the settings and notifications are written in Kirill characters, please change this to English.

![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
20 Jan 2018 at 23:08

**MaPi7475 Pittner:**

Hello,

unfortunately the translation didn't quite work, because all the settings and notifications are written in Kirill characters, please change this to English.

1. You can use the [comments](https://www.mql5.com/en/docs/common/comment "Reference book MQL5 : Comment function") here in the article, but these are certainly not all.
2. The Notepad++ has a plug-in "Translate", which works with MyMemory-Engine and Russian => German (or English), I have already done this several times!
3. I also wrote a language highlighting for NotePad++ (for MT4), which I could also send you...


![MaPi7475 Pittner](https://c.mql5.com/avatar/2018/1/5A53EA80-8CBE.jpeg)

**[MaPi7475 Pittner](https://www.mql5.com/en/users/mapi7475)**
\|
21 Jan 2018 at 17:20

**Carl Schreiber:**

1. You can use the comments here in the article, but these are certainly not all of them.
2. The Notepad++ has a plug-in "Translate", which works with MyMemory-Engine and Russian => German (or English), I have already done this several times!
3. I have also written a language highlighting for NotePad++ (for MT4), I could also send it to you...


Hello Carl,

Thanks for the tip, I will try it with Notepad.

![Vasily Belozerov](https://c.mql5.com/avatar/2022/10/634bb81b-1c89.png)

**[Vasily Belozerov](https://www.mql5.com/en/users/geezer)**
\|
3 Jun 2018 at 17:25

I quote you: "In this article we will continue programming the trading strategies described in the section of the book by L. Raschke and L. Connors, devoted to testing the range boundaries by price". See the reference book. Testing is the process of finding errors. So what is an error, [continuation of](https://www.mql5.com/en/articles/4222 "Article: Continuation pattern - chart search and execution statistics ") price [movement](https://www.mql5.com/en/articles/4222 "Article: Continuation pattern - chart search and execution statistics ") at the range boundary or price return from the range boundary?


![Alexander Puzanov](https://c.mql5.com/avatar/2014/3/53253C30-96FC.png)

**[Alexander Puzanov](https://www.mql5.com/en/users/f2011)**
\|
3 Jun 2018 at 18:56

It is better to change the reference book to one closer to stock exchange topics. Here testing means testing properties, in the context of this section of the book - testing properties of range boundaries by applying to them price movements of different strength. Depending on the ratio of these properties (the value of [price change](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes "MQL5 Documentation: Prices have changed") and resistance of the range boundaries), the price can either break through the levels or break away from them

As for the essence of your question - you can try to profit from the breakout or pullback, but this particular strategy is rather focused on getting profit from the very process of testing the level. At first it is determined which boundary (upper or lower) the price will hit today, and then you try to profit from the movement in the direction of this boundary. It does not matter whether it will be a breakout or a rebound, the main thing is that there should be a movement in the direction of the selected boundary

![Night trading during the Asian session: How to stay profitable](https://c.mql5.com/2/30/timezone.png)[Night trading during the Asian session: How to stay profitable](https://www.mql5.com/en/articles/4102)

The article deals with the concept of night trading, as well as trading strategies and their implementation in MQL5. We perform tests and make appropriate conclusions.

![Creating a custom news feed for MetaTrader 5](https://c.mql5.com/2/30/Creating_a_Custom_news_feed.png)[Creating a custom news feed for MetaTrader 5](https://www.mql5.com/en/articles/4149)

In this article we look at the possibility of creating a flexible news feed that offers more options in terms of the type of news and also its source. The article will show how a web API can be integrated with the MetaTrader 5 terminal.

![Risk Evaluation in the Sequence of Deals with One Asset. Continued](https://c.mql5.com/2/30/Risk_estimation.png)[Risk Evaluation in the Sequence of Deals with One Asset. Continued](https://www.mql5.com/en/articles/3973)

The article develops the ideas proposed in the previous part and considers them further. It describes the problems of yield distributions, plotting and studying statistical regularities.

![The NRTR indicator and trading modules based on NRTR for the MQL5 Wizard](https://c.mql5.com/2/30/qatis21ft_NRTR_2.png)[The NRTR indicator and trading modules based on NRTR for the MQL5 Wizard](https://www.mql5.com/en/articles/3690)

In this article we are going to analyze the NRTR indicator and create a trading system based on this indicator. We are going to develop a module of trading signals that can be used in creating strategies based on a combination of NRTR with additional trend confirmation indicators.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/2825&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070520656363198300)

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