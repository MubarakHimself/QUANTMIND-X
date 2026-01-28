---
title: Building A Candlestick Trend Constraint Model (Part 4): Customizing Display Style For Each Trend Wave
url: https://www.mql5.com/en/articles/14899
categories: Trading Systems, Indicators
relevance_score: 5
scraped_at: 2026-01-23T17:34:02.098405
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/14899&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068388187920857304)

MetaTrader 5 / Trading systems


### Contents

- [Introduction](https://www.mql5.com/en/articles/14899#para1)
- [Recap](https://www.mql5.com/en/articles/14899#para2)
- [MQL5 Display styles](https://www.mql5.com/en/articles/14899#para3)
- [Implementing DRAW\_LINE style in our system.](https://www.mql5.com/en/articles/14899#para4)
- [My D1 Candle status](https://www.mql5.com/en/articles/14899#para5)
- [Understanding the functions and variables of My\_D1\_Candlestatus.mq5](https://www.mql5.com/en/articles/14899#para6)
- [Other alternatives to MQL5 draw styles](https://www.mql5.com/en/articles/14899#para7)
- [Conclusion](https://www.mql5.com/en/articles/14899#para8)

### Introduction

Custom draw styles can enhance the visual appeal of charts, making them more engaging and easier to read. A well-designed chart can improve user experience and reduce eye strain during long trading sessions. By tailoring draw styles to specific needs, traders can create more efficient and effective trading setups. For example, using histograms to represent volume data or lines for moving averages can make it easier to interpret these indicators at a glance. Draw styles like arrows or symbols can be used to mark specific events or signals on the chart, such as buy/sell points, making it easier to spot trading opportunities.

MQL5 boasts a variety of drawing styles for indicators on MetaTrader 5. These visual elements provide traders with an analytical advantage when showcased on the MetaTrader 5 chart, aiding in swift adaptation to market sentiment. Incorporating these diverse drawing styles not only enhances the aesthetic appeal of the charts but also enables traders to make informed decisions based on a comprehensive analysis of market dynamics. Traders can effectively interpret price movements, identify trends, and anticipate potential reversals with greater precision. MQL5 is rich with 18 types of graphical plotting. In this article, we want to delve deeper and explore how to implement one of these display styles into our model.

Instead of just drawing an arrow for our alert we want to create a more advanced visual on the chart to make it even more easier. Remember in this series our goal is to refine our Trend Constraint model such that  it adheres to the sentiment of our D1 candlestick shape as well as present a comprehensive visual signals on chart. MQL5 draw styles can be customized in terms of color, thickness, and style (e.g., dashed or solid lines), helping traders to personalize their charts according to their preferences and improve readability.  Different draw styles allow traders to represent data more clearly and precisely. For instance, using lines, histograms, or candles can make it easier to interpret price movements and market trends.

The various drawing styles available in MQL5 offer traders numerous advantages. They serve to enhance the clarity, precision, and customization of data visualization. These diverse styles expand the capabilities of technical analysis, enabling the utilization of advanced charting techniques and facilitating dynamic real-time updates. Furthermore, the adaptability and ingenuity inherent in MQL5 drawing styles empower traders to craft unique indicators and analysis instruments, thereby amplifying their trading strategies and comprehension of the market landscape. For a comprehensive grasp of this topic, delve into the MQL5 Reference for an in-depth exploration of draw styles.

### Recap

In the previous articles of this series( [Part1](https://www.mql5.com/en/articles/14347), [Part2](https://www.mql5.com/en/articles/14803), and [Part3](https://www.mql5.com/en/articles/14853)) the goal was to confine every signal to the sentiment of the D1 candle. The concept being that if the D1 candle is bullish, typically the day's overall trend will uptrend in lower timeframes. By employing advanced analytical methods at lower timeframes, we can identify entry points and generate signals that align with the current trend. With each phase, we have enhanced our source code, incorporating new features and enhancing our model. In the article series, we utilized arrows as a design element for each iteration of our indicator, exploring the utilization of the Wingdings font for optional display elements.

We added 200 and 100 moving averages to an asset on the MQL5 chart to strategize. By analyzing the behavior of these built-in indicators, we identified a significant periodic crossover event. Subsequently, a personalized crossover indicator with an alert system was created to notify us of such occurrences, indicating a possible trend reversal. Adjusting the MA values to higher levels can help filter out signals during market fluctuations. In Part3, I further refined the approach by introducing customizable MA period input to explore various period values and determine the optimal setup for trend reversals.

```
input int Slow_MA_period = 200;
input int Fast_MA_period = 100;
```

The software provides the capability for instant adjustment of input configurations displayed in the visual representation. To modify its attributes, either utilize the key combination Ctrl + I while viewing the MetaTrader 5 chart or simply right-click the mouse to reveal the indicator menu and locate Trend Constraint V1.03.

> > > ![Optimizing MA for trend reversals](https://c.mql5.com/2/79/Optimizing_MA_for_trend_reversals.gif)
> > >
> > > Fig 1: Optimizing MA period for best trend reversal settings.

After compiling the program using these input configurations, you will discover the source document attached just below the conclusion of this article. Presented below is the most recent code for Trend Constraint V1.03:

```
/// Program after adding Moving Average optimization feature for reversals
///Indicator Name: Trend Constraint
#property copyright "Clemence Benjamin"
#property link      "https://mql5.com"
#property version   "1.03"
#property description "A model that seek to produce sell signal when D1 candle is Bearish only and  buy signal when it is Bullish"

//--- indicator settings
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots 4

#property indicator_type1 DRAW_ARROW
#property indicator_width1 5
#property indicator_color1 0xFF3C00
#property indicator_label1 "Buy"

#property indicator_type2 DRAW_ARROW
#property indicator_width2 5
#property indicator_color2 0x0000FF
#property indicator_label2 "Sell"

#property indicator_type3 DRAW_ARROW
#property indicator_width3 1
#property indicator_color3 0x04CC04
#property indicator_label3 "Buy Reversal"

#property indicator_type4 DRAW_ARROW
#property indicator_width4 1
#property indicator_color4 0xE81AC6
#property indicator_label4 "Sell Reversal"

#define PLOT_MAXIMUM_BARS_BACK 5000
#define OMIT_OLDEST_BARS 50

//--- indicator buffers
double Buffer1[];
double Buffer2[];
double Buffer3[];
double Buffer4[];

input double Oversold = 30;
input double Overbought = 70;
input int Slow_MA_period = 200;
input int Fast_MA_period = 100;
datetime time_alert; //used when sending alert
input bool Audible_Alerts = true;
input bool Push_Notifications = true;
double myPoint; //initialized in OnInit
int RSI_handle;
double RSI[];
double Open[];
double Close[];
int MA_handle;
double MA[];
int MA_handle2;
double MA2[];
int MA_handle3;
double MA3[];
int MA_handle4;
double MA4[];
double Low[];
double High[];

void myAlert(string type, string message)
  {
   if(type == "print")
      Print(message);
   else if(type == "error")
     {
      Print(type+" | Trend Constraint V1.03 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
     }
   else if(type == "order")
     {
     }
   else if(type == "modify")
     {
     }
   else if(type == "indicator")
     {
      if(Audible_Alerts) Alert(type+" | Trend Constraint V1.03 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
      if(Push_Notifications) SendNotification(type+" | Trend Constraint V1.03 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
     }
  }

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, Buffer1);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(0, PLOT_ARROW, 241);
   SetIndexBuffer(1, Buffer2);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(1, PLOT_ARROW, 242);
   SetIndexBuffer(2, Buffer3);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(2, PLOT_ARROW, 236);
   SetIndexBuffer(3, Buffer4);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(3, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(3, PLOT_ARROW, 238);
   //initialize myPoint
   myPoint = Point();
   if(Digits() == 5 || Digits() == 3)
     {
      myPoint *= 10;
     }
   RSI_handle = iRSI(NULL, PERIOD_CURRENT, 14, PRICE_CLOSE);
   if(RSI_handle < 0)
     {
      Print("The creation of iRSI has failed: RSI_handle=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle = iMA(NULL, PERIOD_CURRENT, 7, 0, MODE_SMMA, PRICE_CLOSE);
   if(MA_handle < 0)
     {
      Print("The creation of iMA has failed: MA_handle=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle2 = iMA(NULL, PERIOD_CURRENT, 400, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle2 < 0)
     {
      Print("The creation of iMA has failed: MA_handle2=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle3 = iMA(NULL, PERIOD_CURRENT, Fast_MA_period, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle3 < 0)
     {
      Print("The creation of iMA has failed: MA_handle3=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle4 = iMA(NULL, PERIOD_CURRENT, Slow_MA_period, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle4 < 0)
     {
      Print("The creation of iMA has failed: MA_handle4=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime& time[],
                const double& open[],
                const double& high[],
                const double& low[],
                const double& close[],
                const long& tick_volume[],
                const long& volume[],
                const int& spread[])
  {
   int limit = rates_total - prev_calculated;
   //--- counting from 0 to rates_total
   ArraySetAsSeries(Buffer1, true);
   ArraySetAsSeries(Buffer2, true);
   ArraySetAsSeries(Buffer3, true);
   ArraySetAsSeries(Buffer4, true);
   //--- initial zero
   if(prev_calculated < 1)
     {
      ArrayInitialize(Buffer1, EMPTY_VALUE);
      ArrayInitialize(Buffer2, EMPTY_VALUE);
      ArrayInitialize(Buffer3, EMPTY_VALUE);
      ArrayInitialize(Buffer4, EMPTY_VALUE);
     }
   else
      limit++;
   datetime Time[];

   datetime TimeShift[];
   if(CopyTime(Symbol(), PERIOD_CURRENT, 0, rates_total, TimeShift) <= 0) return(rates_total);
   ArraySetAsSeries(TimeShift, true);
   int barshift_M1[];
   ArrayResize(barshift_M1, rates_total);
   int barshift_D1[];
   ArrayResize(barshift_D1, rates_total);
   for(int i = 0; i < rates_total; i++)
     {
      barshift_M1[i] = iBarShift(Symbol(), PERIOD_M1, TimeShift[i]);
      barshift_D1[i] = iBarShift(Symbol(), PERIOD_D1, TimeShift[i]);
   }
   if(BarsCalculated(RSI_handle) <= 0)
      return(0);
   if(CopyBuffer(RSI_handle, 0, 0, rates_total, RSI) <= 0) return(rates_total);
   ArraySetAsSeries(RSI, true);
   if(CopyOpen(Symbol(), PERIOD_M1, 0, rates_total, Open) <= 0) return(rates_total);
   ArraySetAsSeries(Open, true);
   if(CopyClose(Symbol(), PERIOD_D1, 0, rates_total, Close) <= 0) return(rates_total);
   ArraySetAsSeries(Close, true);
   if(BarsCalculated(MA_handle) <= 0)
      return(0);
   if(CopyBuffer(MA_handle, 0, 0, rates_total, MA) <= 0) return(rates_total);
   ArraySetAsSeries(MA, true);
   if(BarsCalculated(MA_handle2) <= 0)
      return(0);
   if(CopyBuffer(MA_handle2, 0, 0, rates_total, MA2) <= 0) return(rates_total);
   ArraySetAsSeries(MA2, true);
   if(BarsCalculated(MA_handle3) <= 0)
      return(0);
   if(CopyBuffer(MA_handle3, 0, 0, rates_total, MA3) <= 0) return(rates_total);
   ArraySetAsSeries(MA3, true);
   if(BarsCalculated(MA_handle4) <= 0)
      return(0);
   if(CopyBuffer(MA_handle4, 0, 0, rates_total, MA4) <= 0) return(rates_total);
   ArraySetAsSeries(MA4, true);
   if(CopyLow(Symbol(), PERIOD_CURRENT, 0, rates_total, Low) <= 0) return(rates_total);
   ArraySetAsSeries(Low, true);
   if(CopyHigh(Symbol(), PERIOD_CURRENT, 0, rates_total, High) <= 0) return(rates_total);
   ArraySetAsSeries(High, true);
   if(CopyTime(Symbol(), Period(), 0, rates_total, Time) <= 0) return(rates_total);
   ArraySetAsSeries(Time, true);
   //--- main loop
   for(int i = limit-1; i >= 0; i--)
     {
      if (i >= MathMin(PLOT_MAXIMUM_BARS_BACK-1, rates_total-1-OMIT_OLDEST_BARS)) continue; //omit some old rates to prevent "Array out of range" or slow calculation

      if(barshift_M1[i] < 0 || barshift_M1[i] >= rates_total) continue;
      if(barshift_D1[i] < 0 || barshift_D1[i] >= rates_total) continue;

      //Indicator Buffer 1
      if(RSI[i] < Oversold
      && RSI[i+1] > Oversold //Relative Strength Index crosses below fixed value
      && Open[barshift_M1[i]] >= Close[1+barshift_D1[i]] //Candlestick Open >= Candlestick Close
      && MA[i] > MA2[i] //Moving Average > Moving Average
      && MA3[i] > MA4[i] //Moving Average > Moving Average
      )
        {
         Buffer1[i] = Low[1+i]; //Set indicator value at Candlestick Low
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer1[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 2
      if(RSI[i] > Overbought
      && RSI[i+1] < Overbought //Relative Strength Index crosses above fixed value
      && Open[barshift_M1[i]] <= Close[1+barshift_D1[i]] //Candlestick Open <= Candlestick Close
      && MA[i] < MA2[i] //Moving Average < Moving Average
      && MA3[i] < MA4[i] //Moving Average < Moving Average
      )
        {
         Buffer2[i] = High[1+i]; //Set indicator value at Candlestick High
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer2[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 3
      if(MA3[i] > MA4[i]
      && MA3[i+1] < MA4[i+1] //Moving Average crosses above Moving Average
      )
        {
         Buffer3[i] = Low[i]; //Set indicator value at Candlestick Low
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy Reversal"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer3[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 4
      if(MA3[i] < MA4[i]
      && MA3[i+1] > MA4[i+1] //Moving Average crosses below Moving Average
      )
        {
         Buffer4[i] = High[i]; //Set indicator value at Candlestick High
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell Reversal"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer4[i] = EMPTY_VALUE;
        }
     }
   return(rates_total);
  }
//Thank you, friend. You have reached this stage and you can do more+
```

### MQL5 Display styles

MQL5 provides a wide array of drawing styles for indicators. When developing indicators, the main goal is usually to design systems that can notify users through both sound and visual cues. This helps streamline trading operations by reducing the need for traders to constantly watch charts, as the computer takes care of this responsibility. Let me give you a brief overview of some drawing styles that can be utilized in MQL5, such as DRAW\_ARROW, DRAW\_LINE, DRAW\_HISTOGRAM, DRAW\_FILLING, and DRAW\_NONE. Please refer to the table presented below for a concise summary. Check [MQL5 Reference](https://www.mql5.com/en/docs/customind/indicators_examples) for detailed information about draw styles

| DRAW STYLE | DESCPRITION |
| --- | --- |
| [DRAW\_ARROW](https://www.mql5.com/en/docs/customind/indicators_examples/draw_arrow) | Draws arrows at specified points. Often used to highlight buy/sell signals or other important events. |
| [DRAW\_LINE](https://www.mql5.com/en/docs/customind/indicators_examples/draw_line) | Used to draw a line connecting data points. Ideal for moving averages, trend lines, and other line-based indicators |
| [DRAW\_HISTOGRAM](https://www.mql5.com/en/docs/customind/indicators_examples/draw_histogram) | Displays data as bars or histograms. Useful for volume, MACD histograms, and other bar-type indicators. |
| [DRAW\_FILLING](https://www.mql5.com/en/docs/customind/indicators_examples/draw_filling) | Used to fill the area between two lines on a chart, providing a visually intuitive way to represent data ranges, spread, or differences between two indicators |
| [DRAW\_NONE](https://www.mql5.com/en/docs/customind/indicators_examples/draw_none) | Used to define an indicator that does not draw any visual representation on the chart. |

### Implementing DRAW\_LINE style in our system

Let's make use of the DRAW\_LINE function to exhibit our trends in a distinctive manner on the MetaTrader 5 chart. In the past, we managed to set up our indicator effectively to recognize trend reversals via crossovers of higher period moving averages. Our goal now is to improve how we present information visually without overcrowding the chart with too many elements. This fresh addition will allow us to sketch a sole line that showcases trends and automatically alters its color with every new direction. With our current 4 buffers, we are looking to broaden our capabilities by introducing buffer 5 and buffer 6 for version 1.04.

- Buffer 5 : To draw a blue line when MA 100 is above MA 200

```
///properties
#property indicator_type5 DRAW_LINE
#property indicator_style5 STYLE_SOLID
#property indicator_width5 2
#property indicator_color5 0xFFAA00
#property indicator_label5 "Buy Trend"
```

- Buffer 6: To draw a red line when MA 100 is below  MA 200

```
#property indicator_type6 DRAW_LINE
#property indicator_style6 STYLE_SOLID
#property indicator_width6 2
#property indicator_color6 0x0000FF
#property indicator_label6 "Sell Trend"
```

Note: The above section serves as a placeholder for additional information. Further details can be found in the main code below. The buffer layout is not rigid, allowing for variations in values as long as the original intent is maintained consistently.

Main code for Trend Constraint v1.04

```
///Indicator Name: Trend Constraint
#property copyright "Clemence Benjamin"
#property link      "https://mql5.com"
#property version   "1.04"
#property description "A model that seek to produce sell signal when D1 candle is Bearish only and  buy signal when it is Bullish"
//--- indicator settings
#property indicator_chart_window
#property indicator_buffers 6
#property indicator_plots 6

#property indicator_type1 DRAW_ARROW
#property indicator_width1 5
#property indicator_color1 0xFF3C00
#property indicator_label1 "Buy"

#property indicator_type2 DRAW_ARROW
#property indicator_width2 5
#property indicator_color2 0x0000FF
#property indicator_label2 "Sell"

#property indicator_type3 DRAW_ARROW
#property indicator_width3 2
#property indicator_color3 0xE8351A
#property indicator_label3 "Buy Reversal"

#property indicator_type4 DRAW_ARROW
#property indicator_width4 2
#property indicator_color4 0x1A1AE8
#property indicator_label4 "Sell Reversal"

#property indicator_type5 DRAW_LINE
#property indicator_style5 STYLE_SOLID
#property indicator_width5 2
#property indicator_color5 0xFFAA00
#property indicator_label5 "Buy Trend"

#property indicator_type6 DRAW_LINE
#property indicator_style6 STYLE_SOLID
#property indicator_width6 2
#property indicator_color6 0x0000FF
#property indicator_label6 "Sell Trend"

#define PLOT_MAXIMUM_BARS_BACK 5000
#define OMIT_OLDEST_BARS 50

//--- indicator buffers
double Buffer1[];
double Buffer2[];
double Buffer3[];
double Buffer4[];
double Buffer5[];
double Buffer6[];

input double Oversold = 30;
input double Overbought = 70;
input int Slow_MA_period = 200;
input int Fast_MA_period = 100;
datetime time_alert; //used when sending alert
input bool Audible_Alerts = true;
input bool Push_Notifications = true;
double myPoint; //initialized in OnInit
int RSI_handle;
double RSI[];
double Open[];
double Close[];
int MA_handle;
double MA[];
int MA_handle2;
double MA2[];
int MA_handle3;
double MA3[];
int MA_handle4;
double MA4[];
double Low[];
double High[];
int MA_handle5;
double MA5[];
int MA_handle6;
double MA6[];

void myAlert(string type, string message)
  {
   if(type == "print")
      Print(message);
   else if(type == "error")
     {
      Print(type+" | Trend Constraint V1.04 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
     }
   else if(type == "order")
     {
     }
   else if(type == "modify")
     {
     }
   else if(type == "indicator")
     {
      if(Audible_Alerts) Alert(type+" | Trend Constraint V1.04 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
      if(Push_Notifications) SendNotification(type+" | Trend Constraint V1.04 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
     }
  }

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, Buffer1);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(0, PLOT_ARROW, 241);
   SetIndexBuffer(1, Buffer2);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(1, PLOT_ARROW, 242);
   SetIndexBuffer(2, Buffer3);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(2, PLOT_ARROW, 236);
   SetIndexBuffer(3, Buffer4);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(3, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(3, PLOT_ARROW, 238);
   SetIndexBuffer(4, Buffer5);
   PlotIndexSetDouble(4, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(4, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   SetIndexBuffer(5, Buffer6);
   PlotIndexSetDouble(5, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(5, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   //initialize myPoint
   myPoint = Point();
   if(Digits() == 5 || Digits() == 3)
     {
      myPoint *= 10;
     }
   RSI_handle = iRSI(NULL, PERIOD_CURRENT, 14, PRICE_CLOSE);
   if(RSI_handle < 0)
     {
      Print("The creation of iRSI has failed: RSI_handle=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle = iMA(NULL, PERIOD_CURRENT, 7, 0, MODE_SMMA, PRICE_CLOSE);
   if(MA_handle < 0)
     {
      Print("The creation of iMA has failed: MA_handle=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle2 = iMA(NULL, PERIOD_CURRENT, 400, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle2 < 0)
     {
      Print("The creation of iMA has failed: MA_handle2=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle3 = iMA(NULL, PERIOD_CURRENT, 100, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle3 < 0)
     {
      Print("The creation of iMA has failed: MA_handle3=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle4 = iMA(NULL, PERIOD_CURRENT, 200, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle4 < 0)
     {
      Print("The creation of iMA has failed: MA_handle4=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle5 = iMA(NULL, PERIOD_CURRENT, Fast_MA_period, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle5 < 0)
     {
      Print("The creation of iMA has failed: MA_handle5=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle6 = iMA(NULL, PERIOD_CURRENT, Slow_MA_period, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle6 < 0)
     {
      Print("The creation of iMA has failed: MA_handle6=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime& time[],
                const double& open[],
                const double& high[],
                const double& low[],
                const double& close[],
                const long& tick_volume[],
                const long& volume[],
                const int& spread[])
  {
   int limit = rates_total - prev_calculated;
   //--- counting from 0 to rates_total
   ArraySetAsSeries(Buffer1, true);
   ArraySetAsSeries(Buffer2, true);
   ArraySetAsSeries(Buffer3, true);
   ArraySetAsSeries(Buffer4, true);
   ArraySetAsSeries(Buffer5, true);
   ArraySetAsSeries(Buffer6, true);
   //--- initial zero
   if(prev_calculated < 1)
     {
      ArrayInitialize(Buffer1, EMPTY_VALUE);
      ArrayInitialize(Buffer2, EMPTY_VALUE);
      ArrayInitialize(Buffer3, EMPTY_VALUE);
      ArrayInitialize(Buffer4, EMPTY_VALUE);
      ArrayInitialize(Buffer5, EMPTY_VALUE);
      ArrayInitialize(Buffer6, EMPTY_VALUE);
     }
   else
      limit++;
   datetime Time[];

   datetime TimeShift[];
   if(CopyTime(Symbol(), PERIOD_CURRENT, 0, rates_total, TimeShift) <= 0) return(rates_total);
   ArraySetAsSeries(TimeShift, true);
   int barshift_M1[];
   ArrayResize(barshift_M1, rates_total);
   int barshift_D1[];
   ArrayResize(barshift_D1, rates_total);
   for(int i = 0; i < rates_total; i++)
     {
      barshift_M1[i] = iBarShift(Symbol(), PERIOD_M1, TimeShift[i]);
      barshift_D1[i] = iBarShift(Symbol(), PERIOD_D1, TimeShift[i]);
   }
   if(BarsCalculated(RSI_handle) <= 0)
      return(0);
   if(CopyBuffer(RSI_handle, 0, 0, rates_total, RSI) <= 0) return(rates_total);
   ArraySetAsSeries(RSI, true);
   if(CopyOpen(Symbol(), PERIOD_M1, 0, rates_total, Open) <= 0) return(rates_total);
   ArraySetAsSeries(Open, true);
   if(CopyClose(Symbol(), PERIOD_D1, 0, rates_total, Close) <= 0) return(rates_total);
   ArraySetAsSeries(Close, true);
   if(BarsCalculated(MA_handle) <= 0)
      return(0);
   if(CopyBuffer(MA_handle, 0, 0, rates_total, MA) <= 0) return(rates_total);
   ArraySetAsSeries(MA, true);
   if(BarsCalculated(MA_handle2) <= 0)
      return(0);
   if(CopyBuffer(MA_handle2, 0, 0, rates_total, MA2) <= 0) return(rates_total);
   ArraySetAsSeries(MA2, true);
   if(BarsCalculated(MA_handle3) <= 0)
      return(0);
   if(CopyBuffer(MA_handle3, 0, 0, rates_total, MA3) <= 0) return(rates_total);
   ArraySetAsSeries(MA3, true);
   if(BarsCalculated(MA_handle4) <= 0)
      return(0);
   if(CopyBuffer(MA_handle4, 0, 0, rates_total, MA4) <= 0) return(rates_total);
   ArraySetAsSeries(MA4, true);
   if(CopyLow(Symbol(), PERIOD_CURRENT, 0, rates_total, Low) <= 0) return(rates_total);
   ArraySetAsSeries(Low, true);
   if(CopyHigh(Symbol(), PERIOD_CURRENT, 0, rates_total, High) <= 0) return(rates_total);
   ArraySetAsSeries(High, true);
   if(BarsCalculated(MA_handle5) <= 0)
      return(0);
   if(CopyBuffer(MA_handle5, 0, 0, rates_total, MA5) <= 0) return(rates_total);
   ArraySetAsSeries(MA5, true);
   if(BarsCalculated(MA_handle6) <= 0)
      return(0);
   if(CopyBuffer(MA_handle6, 0, 0, rates_total, MA6) <= 0) return(rates_total);
   ArraySetAsSeries(MA6, true);
   if(CopyTime(Symbol(), Period(), 0, rates_total, Time) <= 0) return(rates_total);
   ArraySetAsSeries(Time, true);
   //--- main loop
   for(int i = limit-1; i >= 0; i--)
     {
      if (i >= MathMin(PLOT_MAXIMUM_BARS_BACK-1, rates_total-1-OMIT_OLDEST_BARS)) continue; //omit some old rates to prevent "Array out of range" or slow calculation

      if(barshift_M1[i] < 0 || barshift_M1[i] >= rates_total) continue;
      if(barshift_D1[i] < 0 || barshift_D1[i] >= rates_total) continue;

      //Indicator Buffer 1
      if(RSI[i] < Oversold
      && RSI[i+1] > Oversold //Relative Strength Index crosses below fixed value
      && Open[barshift_M1[i]] >= Close[1+barshift_D1[i]] //Candlestick Open >= Candlestick Close
      && MA[i] > MA2[i] //Moving Average > Moving Average
      && MA3[i] > MA4[i] //Moving Average > Moving Average
      )
        {
         Buffer1[i] = Low[1+i]; //Set indicator value at Candlestick Low
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer1[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 2
      if(RSI[i] > Overbought
      && RSI[i+1] < Overbought //Relative Strength Index crosses above fixed value
      && Open[barshift_M1[i]] <= Close[1+barshift_D1[i]] //Candlestick Open <= Candlestick Close
      && MA[i] < MA2[i] //Moving Average < Moving Average
      && MA3[i] < MA4[i] //Moving Average < Moving Average
      )
        {
         Buffer2[i] = High[1+i]; //Set indicator value at Candlestick High
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer2[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 3
      if(MA5[i] > MA6[i]
      && MA5[i+1] < MA6[i+1] //Moving Average crosses above Moving Average
      )
        {
         Buffer3[i] = Low[i]; //Set indicator value at Candlestick Low
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy Reversal"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer3[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 4
      if(MA5[i] < MA6[i]
      && MA5[i+1] > MA6[i+1] //Moving Average crosses below Moving Average
      )
        {
         Buffer4[i] = High[i]; //Set indicator value at Candlestick High
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell Reversal"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer4[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 5
      if(MA5[i] > MA6[i] //Moving Average > Moving Average
      )
        {
         Buffer5[i] = MA6[i]; //Set indicator value at Moving Average
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy Trend"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer5[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 6
      if(MA5[i] < MA6[i] //Moving Average < Moving Average
      )
        {
         Buffer6[i] = MA6[i]; //Set indicator value at Moving Average
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell Trend"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer6[i] = EMPTY_VALUE;
        }
     }
   return(rates_total);
  }
//You are the best coder
```

Our innovative program now demonstrates advanced capabilities by adjusting the color of a single line whenever a new course is established. This functionality streamlines the chart by incorporating color, thereby enhancing the signaling system alongside other notifications that are both visual and auditory in nature. The images below provide a visual representation of the impact of this latest feature.

![Trend Constraint V1.04 on USDJPY](https://c.mql5.com/2/79/USDJPYmicroM1.png)

Fig 2: Trend Constraint V1.04 on USDJPY

![Trend Constraint V1.04 on Boom 500 Index](https://c.mql5.com/2/79/Boom_500_IndexM1.png)

Fig 3: Trend Constraint V1.04 on Boom 500 index

### My D1 candle status

To add a new feature to our model, we want to quickly check the status of the D1 candlestick as soon as we switch to MetaTrader 5 and start using our model. I've decided to bring in a script here that shows the D1 candle status on the chart, even if the chart is on timeframes other than D1. This improvement helps in quickly identifying the candle status, especially when working on M1 timeframes, where D1 period separators might not be visible at certain zoom levels. Now, let's examine the MQL5 script code given below:

```
//My_D1_candlestatus.mql5
//Author: Clemence Benjamin
//Link: https://www.mql5.com/en/users/billionaire2024/seller
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.00"
#property script_show_inputs
#property strict

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   //--- Get the opening and closing prices of the current D1 candle
   double openPrice = iOpen(NULL, PERIOD_D1, 0);
   double closePrice = iClose(NULL, PERIOD_D1, 0);

   //--- Determine if the candle is bullish or bearish
   string candleStatus;
   if(closePrice > openPrice)
     {
      candleStatus = " D1 candle is bullish.";
     }
   else if(closePrice < openPrice)
     {
      candleStatus = " D1 candle is bearish.";
     }
   else
     {
      candleStatus = " D1 candle is neutral.";// when open price is equal to close price

     }

   //--- Print the status on the chart
   Comment(candleStatus);

   //--- Also print the status in the Experts tab for logging
   Print(candleStatus);
  }
//+------------------------------------------------------------------+
```

### Understanding functions and variables of My\_D1\_Candlestatus.mq5 Script

| Functions and Variable | Descriprition |
| --- | --- |
| OnStart() | This is the main function of the script that executes when the script is run. It retrieves the opening and closing prices of the current D1 candle, determines if the candle is bullish, bearish, or neutral, and then displays this information on the chart using Comment(). |
| iOpen() and iClose() | These functions retrieve the opening and closing prices of the current D1 candle. |
| candleStatus | A string variable to store the status message of the current D1 candle. |
| Comment() | This function displays the status message on the chart. |
| Print() | This function logs the status message to the "Experts" tab for additional logging. |

After you have finished compiling the code, you will be able to locate the script and execute it following the instructions provided in the illustration. The script will display a comment that shows the current day's candle status.

![Running-My D1 Candle status script](https://c.mql5.com/2/79/Running__My_D1_candle_status_script.gif)

Fig 4: Running My\_D1\_candlestatus.mq5 script

### Other alternatives to MQL5 Draw styles

MetaTrader 5 comes with a wide range of sketching aids for hands-on market evaluation, like lines, channels, and figures, all conveniently located in the Meta Trader platform. Through delving into a strong grasp of the MQL5 Object Oriented Programming language and its connections with Python and C++, traders have the ability to devise personalized aids and expansions. Traders can leverage these tools to enhance their technical analysis capabilities and make more informed trading decisions. With the flexibility and customization options available in [MetaTrader 5](https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=www.mql5.com&utm_campaign=0685.mql5.chats.promo "https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=www.mql5.com&utm_campaign=0685.mql5.chats.promo"), traders have the opportunity to tailor their trading experience to suit their preferences and strategies.

### Conclusion

We successfully integrated a new feature into our program, yielding positive outcomes that have the potential to influence the upcoming version as we progress from an indicator to an EA in subsequent articles. The draw styles in MQL5 present notable benefits for traders, enhancing the clarity, precision, and customization of data visualization. These styles bolster technical analysis capabilities, facilitate advanced charting techniques, and support interactive, real-time updates. Moreover, the flexibility and creativity afforded by MQL5 draw styles empower traders to craft distinctive indicators and analytical tools, enriching their trading approaches and overall market comprehension.

Our model is evolving with refined precision, offering instantaneous visual insights at the click of a button. The script promptly conveys the D1 candle status, while the DRAW\_LINE feature accentuates trends at a lower timeframe in a comprehensible manner. I have included the source files for all the features discussed here. I trust you are following the progress and gaining insights; please feel free to share your comments and engage in discussions.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14899.zip "Download all attachments in the single ZIP archive")

[Trend\_Constraint\_V1.03.mq5](https://www.mql5.com/en/articles/download/14899/trend_constraint_v1.03.mq5 "Download Trend_Constraint_V1.03.mq5")(10.75 KB)

[Trend\_Constraint\_V1.04.mq5](https://www.mql5.com/en/articles/download/14899/trend_constraint_v1.04.mq5 "Download Trend_Constraint_V1.04.mq5")(13.46 KB)

[Trend\_Constraint\_V1.04.ex5](https://www.mql5.com/en/articles/download/14899/trend_constraint_v1.04.ex5 "Download Trend_Constraint_V1.04.ex5")(19.16 KB)

[My\_D1\_Candle.mq5](https://www.mql5.com/en/articles/download/14899/my_d1_candle.mq5 "Download My_D1_Candle.mq5")(1.35 KB)

[MY\_D1\_Candle.ex5](https://www.mql5.com/en/articles/download/14899/my_d1_candle.ex5 "Download MY_D1_Candle.ex5")(5.7 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/468550)**
(1)


![4641208](https://c.mql5.com/avatar/avatar_na2.png)

**[4641208](https://www.mql5.com/en/users/4641208)**
\|
13 Jun 2024 at 16:23

**MetaQuotes:**

Check out the new article: [Building A Candlestick Trend Constraint Model (Part 4): Customizing Display Style For Each Trend Wave](https://www.mql5.com/en/articles/14899).

Author: [Clemence Benjamin](https://www.mql5.com/en/users/Billionaire2024 "Billionaire2024")

In the articles:The MQL5 is designed in that it is strong in the algorithm of data analysis and also powerful in drawing various indicator styles on Meta Trader 5.


![A Step-by-Step Guide on Trading the Break of Structure (BoS) Strategy](https://c.mql5.com/2/80/A_Step-by-Step_Guide_on_Trading_the_Break_of_Structure____LOGO_.png)[A Step-by-Step Guide on Trading the Break of Structure (BoS) Strategy](https://www.mql5.com/en/articles/15017)

A comprehensive guide to developing an automated trading algorithm based on the Break of Structure (BoS) strategy. Detailed information on all aspects of creating an advisor in MQL5 and testing it in MetaTrader 5 — from analyzing price support and resistance to risk management

![Integrating Hidden Markov Models in MetaTrader 5](https://c.mql5.com/2/80/Integrating_Hidden_Markov_Models_in_MetaTrader_5_____LOGO.png)[Integrating Hidden Markov Models in MetaTrader 5](https://www.mql5.com/en/articles/15033)

In this article we demonstrate how Hidden Markov Models trained using Python can be integrated into MetaTrader 5 applications. Hidden Markov Models are a powerful statistical tool used for modeling time series data, where the system being modeled is characterized by unobservable (hidden) states. A fundamental premise of HMMs is that the probability of being in a given state at a particular time depends on the process's state at the previous time slot.

![MQL5 Trading Toolkit (Part 1): Developing A Positions Management EX5 Library](https://c.mql5.com/2/80/MQL5_Trading_Toolkit_Part_1___LOGO.png)[MQL5 Trading Toolkit (Part 1): Developing A Positions Management EX5 Library](https://www.mql5.com/en/articles/14822)

Learn how to create a developer's toolkit for managing various position operations with MQL5. In this article, I will demonstrate how to create a library of functions (ex5) that will perform simple to advanced position management operations, including automatic handling and reporting of the different errors that arise when dealing with position management tasks with MQL5.

![Gain An Edge Over Any Market (Part II): Forecasting Technical Indicators](https://c.mql5.com/2/80/Gain_An_Edge_Over_Any_Market_Part_II___LOGO.png)[Gain An Edge Over Any Market (Part II): Forecasting Technical Indicators](https://www.mql5.com/en/articles/14936)

Did you know that we can gain more accuracy forecasting certain technical indicators than predicting the underlying price of a traded symbol? Join us to explore how to leverage this insight for better trading strategies.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/14899&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068388187920857304)

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