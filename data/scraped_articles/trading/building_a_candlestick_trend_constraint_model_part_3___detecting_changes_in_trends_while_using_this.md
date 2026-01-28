---
title: Building A Candlestick Trend Constraint Model(Part 3): Detecting changes in trends while using this system
url: https://www.mql5.com/en/articles/14853
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 5
scraped_at: 2026-01-23T17:32:14.448922
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/14853&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068351727943481453)

MetaTrader 5 / Trading


Contents

- [Introduction](https://www.mql5.com/en/articles/14853#para1)
- [Ways of identifying a change in market trends](https://www.mql5.com/en/articles/14853#para2)
- [Using moving averages to spot trend reversal](https://www.mql5.com/en/articles/14853#para3)
- [Using candlestick patterns to spot trend reversal](https://www.mql5.com/en/articles/14853#para4)
- [Using trendlines to spot trend reversal](https://www.mql5.com/en/articles/14853#para5)
- [Using support and resistance to spot reversal](https://www.mql5.com/en/articles/14853#para6)
- [Problems with our current system](https://www.mql5.com/en/articles/14853#para7).
- [Incorporating a new feature into our program using MQL5](https://www.mql5.com/en/articles/14853#para8)
- [Exploring the results of the final system](https://www.mql5.com/en/articles/14853#para9)
- [Video result explainer](https://www.mql5.com/en/articles/14853#para11)
- [Conclusion](https://www.mql5.com/en/articles/14853#para10)

### Introduction

Generally, markets do not stay static. Whether trending up or down, unexpected shifts can occur when the market alters its trajectory. It is vital for a system to identify and adjust to these changes. Even a prolonged bearish D1 candle can signal a shift in dynamics when a reversal occurs at a lower time frame. This article delves into various methods employed to recognize changes in price action trends. As traders navigate through the complexities of financial markets, the ability to adapt swiftly to evolving conditions becomes paramount. Recognizing the nuances of price movements, understanding the significance of key indicators, and interpreting market sentiment are all crucial components in staying ahead of the curve. By honing the skill of identifying shifts in trends, traders can position themselves strategically to capitalize on opportunities that arise amidst the ever-changing landscape of the financial world.

Various factors influence or drive changes in market trends. Here are a few examples:

- investor behavior, that is buying and selling
- economic news release such as GDP and Non-Farm Payroll
- monetary policies
- global events such as natural disasters
- political events such as war etc.

Through different learning resources, we have acquired extensive knowledge on how to manually detect changes in trends. This includes the concept of trendline analysis, which involves drawing lines on a price chart to connect the highs or lows of an asset's price movement such that traders can gain insight into potential trend changes when the price breaks above or below these lines. Later in this article, we will select one method to detect market trend changes and integrate it into our Trend Constraint indicator using MQL5. First, we will explore various technical analysis tools such as moving averages, candlestick patterns, relative strength index( see [part 2](https://www.mql5.com/en/articles/14803)), and trendlines(see Fig 1) to identify potential trend reversals. We will proceed to modify our Trend Constraint indicator in MQL5 to incorporate this new functionality.

Disclaimer: The strategies and techniques outlined in this article are intended for educational purposes only. Your trades are the outcome of your own actions. I will not be liable for any losses resulting from the use of the information or tools provided here

.

![trend illustration](https://c.mql5.com/2/78/trend_illustration.png)

FIG 1: illustration of trends

Note: The image above was hand-drawn for learning purposes; no real asset is being presented.

The illustration above depicts a typical uptrend with lows A and B connected by a blue trendline, indicating an upward trend. In the following examples, we will explore real chart instances to further understand trends. Our primary objective is to recognize market trend changes and incorporate a suitable method into our system using MQL5 code.

### Ways of detecting change in market trends.

Let us begin by defining market trends as:

Market trend signifies the overall direction a market moves over time, reflecting buyer and seller behavior. Trends can be upward (bullish), downward (bearish), or sideways (consolidation) see  (Fig 1) in [introduction](https://www.mql5.com/en/articles/14853#para1)

Let's now define a trend reversal as:

A trend reversal occurs when the price movement shifts from an uptrend to a downtrend, or vice versa. This shift can be identified by analyzing key technical indicators such as moving averages, trendlines, candlestick patterns and support/resistance levels. Traders and investors closely monitor these changes in market trend to make informed decisions and adjust their strategies accordingly.

### Using Moving Averages to spot trend reversal

In [part 1](https://www.mql5.com/en/articles/14347) of this series, we created a fast-moving average crossover indicator to provide trend continuation signals when a crossover occurs. However, higher period moving averages can indicate a significant trend change during a crossover. In this article, we will examine how moving averages react when the market changes direction. Below is an image

![USDJPYmicroM5 showing how Moving Averages can signai a reversal](https://c.mql5.com/2/78/USDJPYmicroM5.illust.png)

Fig 2: Moving average crossover as a trend reversal signal.

### Using candlestick shapes to spot trend reversals

Candlestick patterns can be effectively used to spot possible reversals. They have been analyzed throughout history for their ability to shift market sentiment significantly. Visionaries like [Honma Munehisa](https://en.wikipedia.org/wiki/Honma_Munehisa "Visit Honma Munehisa wiki"), the creator of the [Candlestick Bible](https://www.mql5.com/go?link=https://www.noor-book.com/en/ebook-the-candlestick-trading-bible-pdf "Get the book"), have enriched our understanding of candlesticks. Here are some of the most prevalent candlestick patterns for market reversals:

| Candlestich name | Description |
| --- | --- |
| hammer | small body and a long lower shadow |
| inverted hammer | small body and a long upper shadow |
| bullish engulfing pattern | a bullish candlestick completely engulfs the previous bearish candlestick |
| bearish engulfing pattern | a bearish candlestick completely engulfs the previous bullish candlestick |
| doji | a small body and long upper and lower shadows |
| shooting star | a small body and a long upper shadow |
| hanging man | a small body and a long lower shadow |
| morning star | a long bearish candlestick, followed by a small-bodied candlestick with a lower low and higher high |
| evening star | a long bullish candlestick, followed by a small-bodied candlestick with a higher high and lower low |

All these candlestick characteristics are programmable using MQL5 because the have all the price levels open close high and low which are crucial when programming.

### Using Trendlines to spot trend reversal

In the [Mt5 platform](https://www.metatrader5.com/ "Downlad MT5") chart, we can use the trendline object tool to plot trends by connecting consecutive troughs in a price series of a digital asset. A broken trendline indicates a trend change. To draw a trendline, simply click on the trendline tool in the toolbar, then click on the first trough and drag the line to the next trough. The trendline will automatically extend to the right side of the chart.

![B300 index trendline broken](https://c.mql5.com/2/78/Boom_300_IndexM5.illust.png)

Fig 3: Trendline as at a tool for detecting trend reversal

### Using Support/resistance levels to spot trend reversal

A horizontal line tool in the [MT5 chart](https://www.metatrader5.com/ "Download MT5") can be utilized to plot support and resistance levels in trends by placing it on price peaks. Observing the price when it breaks those levels might indicate a trend shift. Check the [video explainer](https://www.mql5.com/en/articles/14853/151338#para11) below.

### Problems with our current system.

We have successfully configured our system to align its signal with the D1 trend shape and incorporate indicators such as SMA 400 in [part 2](https://www.mql5.com/en/articles/14803). However, there are significant challenges observed in historical data. We must consider fluctuations in lower timeframes, which can potentially reverse the initial sentiment at the beginning of the day; for instance, a day might commence with a bearish outlook but conclude with a pin bar or bullish sentiment. Reversal signals often emerge in lower timeframes, prompting us to adapt the system's signaling mechanism to account for trend changes, even if they were initially confined to the daily market sentiment.

### Incorporating trend change detection feature with MQL5

I have chosen to use SMA 200 as a slow moving average and EMA 100 as a fast moving average. Typically, these moving averages are further apart during strong trends but closer together when market momentum is weak or the price is moving sideways. When the two averages cross over, it often signals a change in direction. Having our system detect this and alert us would be very beneficial. Our goal is for the system to only provide us with the signal, so that our constraint signals can be adjusted accordingly. With this approach, we aim to capture potential trend reversals and capitalize on profitable trading opportunities. I shall explain some key feature then include the main code after

Our MA handler is declared as follows.

```
MA_handle3 = iMA(NULL, PERIOD_CURRENT, 100, 0, MODE_EMA, PRICE_CLOSE); // For EMA 100

MA_handle4 = iMA(NULL, PERIOD_CURRENT, 200, 0, MODE_SMA, PRICE_CLOSE); // For SMA 200
```

The code below shows the crossover conditions in MQL5 under the iteration function.

```
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
```

The Trend Constraint from the previous article featured 2 buffers, one for buy signals and the other for sells for trend continuation. To achieve our goal, we want to add two more buffers, one for sell and the other for buy, all representing reversal signals when a crossover happens. In the program they are consecutively named Buffer3  and Buffer4. We optimized a new display style for the feature. The indicator's display can be customized by selecting from the mql5 object [Wingdings](https://www.mql5.com/en/docs/constants/objectconstants/wingdings). Here, I utilized object number 236 for my buy reversal signal and object number 238 for the sell reversal sign.

```
// under OnInit() function. The wingding objects can be customized by altering those highlighted values choosing from wingding listing.
  SetIndexBuffer(2, Buffer3);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(2, PLOT_ARROW, 236);
   SetIndexBuffer(3, Buffer4);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(3, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(3, PLOT_ARROW, 238);
```

The other aspect is color code using  MQL5. These colours can be optimised  through the input settings in [metatrader 5](https://www.metatrader5.com/ "Download MT5/"). Each color is represented by a unique code within a program .eg. **"C'0,0,0'"** represents black see code snippet below

```
#property indicator_type3 DRAW_ARROW
#property indicator_width3 1  // with can be adjusted up to 5 times.
#property indicator_color3 0x04CC04 //color for buy reversal
#property indicator_label3 "buy reversal"

#property indicator_type4 DRAW_ARROW
#property indicator_width4 1              //with can be adjusted up to 5 times.
#property indicator_color4 0xE81AC6  // Color code for sell reversal
#property indicator_label4 "sell reversal"
```

More details and comments in the  main code below having combined all the pieces and idea together.

```
///Indicator Name: Trend Constraint
#property copyright "Clemence Benjamin"
#property link      "https://mql5.com"
#property version   "1.03"
#property description "A model that seek to produce sell signal when D1 candle is Bearish only and  buy signal when it is Bullish"

//+------------------------------------------------------------------------------------------------------------------------------+
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
         Buffer1[i] = Low[i]; //Set indicator value at Candlestick Low
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
         Buffer2[i] = High[i]; //Set indicator value at Candlestick High
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
//Thank you for getting this far, you are amazing.
//+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

### Exploring the results of the final system

The results from the system are impressive. We can now limit our signal system to D1 Market sentiment and also receive signals when a reversal impulse happens via the crossover of EMA(100) and SMA(200). Here are images showing the indicator output on actual chart history. The system seems to be performing exceptionally well in capturing market sentiment shifts and identifying potential reversal points. By focusing on D1 Market sentiment and utilizing the EMA(100) and SMA(200) crossover signals, we are able to enhance our trading strategies and make more informed decisions. The indicator output on the historical chart data clearly demonstrates the effectiveness of these signals in predicting market movements.

![USDJPYM1 with Trend Constraint v1.03](https://c.mql5.com/2/78/USDJPY.png)

Fig 4 : Results of Trend Constraint V1.03 on USDJPYmicroM1

Note: If you are experiencing issues with missing signal arrows on your chart after adding the indicator, try refreshing by right-clicking the mouse button (RMB) while on the MT5 chart and selecting "Refresh" from the menu that appears.

The data collected from the developments we are making can later be incorporated into machine learning and artificial intelligence systems for further refinement. These systems can be trained to conduct advanced analysis, which would be beneficial in overcoming the challenges we have faced with the current model. The signals in the result image above align with the idea, but there were also some misleading signals. This is typical for any system and serves as motivation to explore additional methods for improving our current system. Synthetics can provide various results when utilizing this system.

### Video result explainer

View through the video below to explore the performance of our  new version development.

YouTube

### Conclusion

Incorporating trend change detection features into our system has significantly enhanced it. While we confine our signals to the current market trend, we have successfully mitigated potential losses that could arise from signals backing an invalidated trend, despite D1 sentiment supporting it. We have encountered issues with the reversal signals provided by this system during a persistent trend. To address this, I have decided to extend the period of the moving averages utilized. In upcoming articles, I will revisit this topic to delve into how this adjustment unfolded. I trust you found value in this conversation and welcome your thoughts in the comments section below. Our future articles will incorporate advanced visualizations of our indicator system using the versatile MQL5 language.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14853.zip "Download all attachments in the single ZIP archive")

[Trend\_Constraint\_V1.03.ex5](https://www.mql5.com/en/articles/download/14853/trend_constraint_v1.03.ex5 "Download Trend_Constraint_V1.03.ex5")(17.04 KB)

[Trend\_Constraint\_V1.03.mq5](https://www.mql5.com/en/articles/download/14853/trend_constraint_v1.03.mq5 "Download Trend_Constraint_V1.03.mq5")(10.62 KB)

[Trend\_Constraint\_V1.03.zip](https://www.mql5.com/en/articles/download/14853/trend_constraint_v1.03.zip "Download Trend_Constraint_V1.03.zip")(2.4 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/467506)**

![DRAW_ARROW drawing type in multi-symbol multi-period indicators](https://c.mql5.com/2/65/Drawing_type_DRAW_ARROW_in_multi-symbol_multi-period_indicators__LOGO.png)[DRAW\_ARROW drawing type in multi-symbol multi-period indicators](https://www.mql5.com/en/articles/14105)

In this article, we will look at drawing arrow multi-symbol multi-period indicators. We will also improve the class methods for correct display of arrows showing data from arrow indicators calculated on a symbol/period that does not correspond to the symbol/period of the current chart.

![Spurious Regressions in Python](https://c.mql5.com/2/78/Spurious_Regressions_in_Python___LOGO__BIG-transformed.png)[Spurious Regressions in Python](https://www.mql5.com/en/articles/14199)

Spurious regressions occur when two time series exhibit a high degree of correlation purely by chance, leading to misleading results in regression analysis. In such cases, even though variables may appear to be related, the correlation is coincidental and the model may be unreliable.

![Causal inference in time series classification problems](https://c.mql5.com/2/66/Causal_inference_in_time_series_classification_problems___LOGO.png)[Causal inference in time series classification problems](https://www.mql5.com/en/articles/13957)

In this article, we will look at the theory of causal inference using machine learning, as well as the custom approach implementation in Python. Causal inference and causal thinking have their roots in philosophy and psychology and play an important role in our understanding of reality.

![Developing a multi-currency Expert Advisor (Part 1): Collaboration of several trading strategies](https://c.mql5.com/2/65/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 1): Collaboration of several trading strategies](https://www.mql5.com/en/articles/14026)

There are quite a lot of different trading strategies. So, it might be useful to apply several strategies working in parallel to diversify risks and increase the stability of trading results. But if each strategy is implemented as a separate Expert Advisor (EA), then managing their work on one trading account becomes much more difficult. To solve this problem, it would be reasonable to implement the operation of different trading strategies within a single EA.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/14853&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068351727943481453)

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