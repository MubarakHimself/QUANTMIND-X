---
title: DiNapoli trading system
url: https://www.mql5.com/en/articles/3061
categories: Trading, Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T18:18:04.154066
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/3061&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069322961783030570)

MetaTrader 5 / Trading


### Contents

1. [Introduction](https://www.mql5.com/en/articles/3061#z1)
2. [DiNapoli levels: the basics](https://www.mql5.com/en/articles/3061#z2)
3. [Main principles and concepts](https://www.mql5.com/en/articles/3061#z3)
4. [Fibo/DiNapoli extensions](https://www.mql5.com/en/articles/3061#z4)
5. [DiNapoli levels trading technics](https://www.mql5.com/en/articles/3061#z5)
6. [Minesweeper A and Minesweeper B strategies](https://www.mql5.com/en/articles/3061#z6)
7. [DiNapoli levels indicator](https://www.mql5.com/en/articles/3061#z7)
8. [Conclusion](https://www.mql5.com/en/articles/3061#z8)
9. [Reference](https://www.mql5.com/en/articles/3061#z9)

### 1\. Introduction

New trading strategies and modifications of classical trading systems appear every day. One of the most famous methods was created by Joe DiNapoli, trader, economist and author of popular books. His widely used system is based on Fibonacci levels.

In this article, we will have a close look at the system. I have made my best to explain the basic concepts and trading principles that otherwise cannot be described without close acquaintance with Fibo levels. So, what are these levels? How to interpret and apply them in practice? And most importantly, how do they work in trading?

### 2\. DiNapoli levels: the basics

![](https://c.mql5.com/2/28/1_eqj__2.png)

DiNapoli levels basics

Let's consider the general principles of trading DiNapoli levels, including how to correctly plot them on charts, interpret their readings relative to the price movement and apply them to define market entry points.

As I have already said, DiNapoli strategy is based on Fibo levels. Only the horizontal markup is used in this method. Fibo arcs and fans are not applied.

![](https://c.mql5.com/2/28/2_cx4__2.png)

Wrong placement of Fibo levels

![](https://c.mql5.com/2/28/3_dhx.png)

Correct placement of Fibo levels

DiNapoli levels are inherently support and resistance levels, although their interpretation is more deep as compared to conventional levels. They are based on Fibo levels and apply the so-called _market swing_ (see below) or, more simply, a trend. We need only three Fibo levels to build DiNapoli ones: 61.8%, 50% and 38.2%. These are the levels to be used on your chart as support and resistance ones. The Fibo levels are plotted upwards in case of an uptrend and downwards in case of a downtrend.

Ideally, the lines are arranged from 0% to 100% in the direction of the current trend. All Fibo levels exceeding 100% are used to set position closing points.

![](https://c.mql5.com/2/28/4_gai__2.png)

Market swing

DiNapoli levels can be applied in two ways — as an extension and correction.

**Correction** is applied for detecting target entry points, while an **extension** grid is built for detecting market exit points.

### 3\. Main principles and concepts

- _**Fibo node**_ — price chart point, at which correction movement ends. Joe DiNapoli advised to work only with nodes found between 38.2% and 61.8%. All nodes located below 38.2% are too insignificant, while the ones exceeding 61.8% are too strong (meaning a trend may not recover).

- _**Market swing**_ (distance between extremums) — distance from the beginning to the end of the price movement, the Fibo levels are plotted on. If the price changes its High or Low point after the end of the correction, you should also move the extreme point of the levels. In this case, the swing increases.


- _**Accumulation**_ — place on a chart where an accumulation of several Fibo nodes (multiple correction levels close to each other) is detected.


- _**Focus point** (extremum)_ — extreme point of the market swing (level the price correction starts from). When an extremum changes, another focus point appears on a chart. Thus, there may be multiple points on the chart in one swing zone.

- _**Target level**_ — point on a price chart where you want to place a take profit.


- _**Target point**_ — any chart point an action is planned at (market entry, market exit, etc.).

- _**Reaction** —_ finished correction in a trend movement. Several reactions may form in a single swing radius.


![](https://c.mql5.com/2/28/5_l6p__2.png)

Extensions, correction and focus point

A price chart is constantly moving, and each time market swings should be corrected relative to the chart. In this case, the swings are expanded each time leading to changes in the number of focus points on the chart. Price corrections called "reactions" appear in the center of the market swings. Each of them has its own index.

![](https://c.mql5.com/2/28/6_olk__2.png)

Reactions on the chart

**Tips from DiNapoli**

- Signals received from the levels produce results only if correction levels as well as accumulations/support levels are located within the levels range of 38.2%-61.8%. Other signals are ignored.


- Use common Fibo levels reaching the last correction as take profit levels in order to place a stop loss on 161.8% and 261.8%.

- The number of swings on the chart is always equal to the amount of focus numbers.

- Focus numbers are always located to the right of the reactions they are connected with. In other words, price corrections formed on the levels swings should be located to the left of the focus number.

- The higher the timeframe, the lesser the number of reactions. As you may have already noticed, when working on different timeframes, the price movement is quite fast on small time periods, whereas these movements are not even displayed on higher ones.This rule is also used in the reverse order: when the price detects a strong resistance/support, you are able to find necessary reactions and swings that are not displayed on higher timeframes by using lower ones.

### 4\. Fibo/DiNapoli extensions

DiNapoli extensions grid allows us to define market exit points based on Fibo extensions grid. This time, we will use the levels of 100%, 161.8% and 261.8% that are target points for setting a take profit.

![](https://c.mql5.com/2/28/7_coa__2.png)

Fibo (DiNapoli) levels extensions

The Fibo extensions grid is built the following way:

- For a downtrend — starting with a Low price to the price correction peak formed by the price roll-back line beginning from 38.2% and higher.


- For an uptrend — starting with a Low price to the price correction minimum crossing the level of 38.2% and higher.

![](https://c.mql5.com/2/28/8_xbk__3.png)

Placing the Fibo extensions grid on a chart

In his book "Trading with DiNapoli Levels", the author pays much attention to such concepts as "Multiple focus numbers" and "Market swings". These patterns are typical when trading instruments with frequent flats. Such patterns are more complex than simple swings. The main difficulty is sorting out unnecessary levels.

Let me briefly list the conclusions I made after reading the book. The book's indisputable advantages lie in the fact that it was written not by a theorist, but by an experienced trader and manager. Another solid argument in its favor is the use of Fibo levels that passed through a centuries-old time test and are mathematically balanced and verified. Besides, examples provided in the book describe not only commodity and stock markets, but also currency futures trading, and its principles can be applied to the Forex market.

Many new concepts introduced by DiNapoli also require careful study and comprehensive understanding.

### 5\. DiNapoli levels trading technics

Now, let's consider trading using DiNapoli levels. In short, its basic idea described in his book allows us to develop several trading tactics with "aggressive" and "quiet" levels.

- The **aggressive** one features two market entry methods: the Bushes and Bonsai strategies. The trading principle is similar, the only difference being the placement of stop losses.

- The **quiet** strategy also describes the two methods: Minesweeper A and B.

When applying the aggressive trading method, it is assumed that the price rolls back from the level of 38.2% and the already formed swing. The only difference between the Bushes and Bonsai strategies is the Fibo level, after which a stop loss is set.

Let's start from the **Bushes** trading method. According to this method, positions should be opened the moment the price on the generated corrections grid crosses the level of 38.2% in the current trend direction, while a stop loss is placed further than the level of 50%. According to the **Bonsai** method, the market entry is similar to the Bushes one, although a stop loss is set stop short of the Fibo level of 50%.

Following these methods is assumed to be aggressive since there is a risk that the price roll-back on the chart may not happen. It is quite common for the correction to turn into a new trend, or the price may enter flat just for a short while. Therefore, if you decide to apply this method, wait for a full confirmation of the signal to be on the safe side.

![](https://c.mql5.com/2/28/9_86m__2.png)

The Bushes strategy

The book also describes some of the negative features of the Bonsai strategy. DiNapoli emphasizes that a considerable slippage is probable when executing a stop order in that case, since there is no powerful level meaning insufficient number of trades and no solid request matching. Thus, the choice depends on a brokerage company and a trading platform. On the other hand, if you trade highly liquid instruments with a small volume in the market, such situation is unlikely.

### 6\. Minesweeper A and Minesweeper B strategies

The most quiet and less risky strategies are Minesweeper A and B. According to them, a market entry should be performed after a correction, while the trading itself is conducted using safety measures.

- **Minesweeper A**. First, we wait for the initial correction to finish (no market entry), then for the second correction to form. Only after that, we open positions. A stop loss is placed the same way as in the previous strategy, i.e. behind the next Fibo level.


- **Minesweeper B**. Instead of opening positions after the second formed correction, they are opened after the third, fourth or even later ones. In other words, we enter the market only after the trend is thoroughly confirmed meaning the risk of false signal is considerably reduced.

What are the results of the strategy? The market entries are infrequent. A trend is not always long-living enough to provide 3 or more correction moves in a row corresponding to the rules of this trading system (a roll-back end should be on the levels from 38.2% to 61.8%). On the other hand, this allows us to sort out false signals, in which the price correction does not become a continuation of a trend.

### ![](https://c.mql5.com/2/28/10_g5o__3.png)      Minesweeper A strategy

If you come across a rather long-living trend featuring multiple corrections (reactions) and follow all the analysis rules, your chart is quickly littered with many unimportant levels and lines. Most of them can be simply discarded as redundant data that do not fall under the rules for trading DiNapoli levels.

Suppose that you see a powerful uptrend with a few reactions. At some time, you have a correction of all upward movements. As a result, the price starts re-writing the Lows of some reactions. Such reactions should be canceled and their Lows should be discarded.

### 7\. DiNapoli levels indicator

For those unwilling to spend their time plotting DiNapoli levels manually by placing Fibo levels, there is an indicator doing that automatically. The indicator is attached below. It can also be found in the [CodeBase](https://www.mql5.com/en/code/1509). Let's analyze its operation in more details. The indicator name has been changed for more convenience.

The indicator is installed in a usual way by placing the file to the Indicators folder of the MetaTrader 5 root directory. It does not have too much settings most of them being level colors. The colors are customizable, but I do not recommend changing them if you are a novice in order to avoid display and market analysis errors.

The indicator for auto display of DiNapoli levels also includes Zigzag with a ZigZag reversal audio signal. The red line marks the place for placing a stop loss on the chart, while the blue one shows the operation start level. The remaining horizontal lines are price target ones. Besides, the indicator shows the vertical time layout lines (they can be disabled in the indicator settings).

![](https://c.mql5.com/2/28/bandicam_96.png)

Indicator inputs

![](https://c.mql5.com/2/28/11_qk6__3.png)

Displaying DiNapoli Levels on the terminal price chart

**Inputs:**

- Minimum points in a ray (default = 400) – change the width of the vertical time levels;
- Show the vertical lines (default = true) – show/hide vertical time levels;
- Number of history bars (default = 5000) – number of history bars used by the built-in ZigZag indicator;
- Play sound (default = true) – enable audio notifications on ZigZag changing its direction;
- Sound file (default = "expert.wav") – select an audio notification file;
- Start Line color (default = Blue) – start horizontal line color;
- Stop Line color (default = Red) – color of a horizontal line for setting a stop loss;
- Target1 Line color (default = Green) – color of a horizontal line for the target 1;
- Target2 Line color (default = DarkOrange) – color of a horizontal line for the target 2;
- Target3 Line color (default = DarkOrchid) – color of a horizontal line for the target 3;
- Target4 Line color (default = DarkSlateBlue) – color of a horizontal line for the target 4;
- Time Target1 color (default = DarkSlateGray) – vertical time line 1 color;
- Time Target2 color (default = SaddleBrown) – vertical time line 2 color;
- Time Target3 color (default = DarkSlateGray) – vertical time line 3 color;
- Time Target4 color (default = DarkSlateGray) – vertical time line 4 color.

First, let's introduce the basic indicator parameters, on which the entire code is built.

The initial code parameters look as follows:

```
//------------------------------------------------------------------------------------
//                                                                 DiNapoli Levels.mq5
//                                                   The modified indicator FastZZ.mq5
//                                       Added DiNapoli Target Levels and Time Targets
//                                                         victorg, www.mql5.com, 2013
//------------------------------------------------------------------------------------
#property copyright   "Copyright 2012, Yurich"
#property link        "https://login.mql5.com/en/users/Yurich"
#property version     "3.00"
#property description "FastZZ plus DiNapoli Target Levels."
#property description "The modified indicator 'FastZZ.mq5'."
#property description "victorg, www.mql5.com, 2013."
//------------------------------------------------------------------------------------
#property indicator_chart_window // Display the indicator in the chart window
#property indicator_buffers 3    // The number of buffers to calculate the indicator
#property indicator_plots   1    // Number of indicator windows
#property indicator_label1  "DiNapoli Levels" // Set the label for the graphics series
#property indicator_type1   DRAW_COLOR_ZIGZAG // Drawing style of the indicator. N - number of the graphic series
#property indicator_color1  clrTeal,clrOlive  // The color for the N line output, where N is the number of the graphics series
#property indicator_style1  STYLE_SOLID       // Line Style in the Graphics Series
#property indicator_width1  1    // Line thickness in the graphics series
//------------------------------------------------------------------------------------
input int    iDepth=400;              // Minimum of points in the ray
input bool   VLine=true;              // Show Vertical Lines
input int    iNumBars=5000;           // Number of bars in history
input bool   Sound=true;              // Enable sound notifications
input string SoundFile="expert.wav";  // Audio file
input color  cStar=clrBlue;           // Color of the start line
input color  cStop=clrRed;            // Stop line color
input color  cTar1=clrGreen;          // Color of the goal line #1
input color  cTar2=clrDarkOrange;     // Color of the goal line # 2
input color  cTar3=clrDarkOrchid;     // Color of the goal line # 3
input color  cTar4=clrDarkSlateBlue;  // Color of the goal line #4
input color  cTarT1=clrDarkSlateGray; // Color of the time line #1
input color  cTarT2=clrDarkSlateGray; // Color of the time line #2
input color  cTarT3=clrSaddleBrown;   // Color of the time line #3
input color  cTarT4=clrDarkSlateGray; // Color of the time line #4
input color  cTarT5=clrDarkSlateGray; // Color of the time line #5
```

Let's enter the variables in the indicator.

```
//Main variables
double   DiNapoliH[],DiNapoliL[],ColorBuffer[],Depth,A,B,C,Price[6];
int      Last,Direction,Refresh,NumBars;
datetime AT,BT,CT,Time[5];
color    Color[11];
string   Name[11]={"Start Line","Stop Line","Target1 Line","Target2 Line",
                   "Target3 Line","Target4 Line","Time Target1","Time Target2",
                   "Time Target3","Time Target4","Time Target5"};
```

After setting the main parameters and entering the variables, it is time to develop the main part of the indicator.

Main part:

```
// Begin the initialization of the indicator
void OnInit()
  {
  int i;
  string sn,sn2;

// Set the conditions for the points in the ray
  if(iDepth<=0)Depth=500;
  else Depth=iDepth;

// Set the conditions for bars in history
  if(iNumBars<10)NumBars=10;
  else NumBars=iNumBars;

// Set up displaying indicator buffers
  SetIndexBuffer(0,DiNapoliH,INDICATOR_DATA);
  SetIndexBuffer(1,DiNapoliL,INDICATOR_DATA);
  SetIndexBuffer(2,ColorBuffer,INDICATOR_COLOR_INDEX);

// Set the accuracy of displaying the indicator values
  IndicatorSetInteger(INDICATOR_DIGITS,Digits());

// Set up the drawing of lines
  PlotIndexSetDouble(0,PLOT_EMPTY_VALUE,0.0);
  PlotIndexSetDouble(1,PLOT_EMPTY_VALUE,0.0);

// Set up a short name for the indicator
  sn="DiNapoli"; sn2="";
  for(i=1;i<100;i++)
    {
// Set up a chart search
    if(ChartWindowFind(0,sn)<0){break;}
    sn2="_"+(string)i; sn+=sn2;
    }

// Set the symbol display
  IndicatorSetString(INDICATOR_SHORTNAME,sn);
  for(i=0;i<11;i++) Name[i]+=sn2;

// Initialize the buffers with empty values
  ArrayInitialize(DiNapoliH,0); ArrayInitialize(DiNapoliL,0);
```

Let's proceed developing the indicator:

```
// Adjust the color lines of the indicator
  Color[0]=cStar; Color[1]=cStop; Color[2]=cTar1; Color[3]=cTar2;
  Color[4]=cTar3; Color[5]=cTar4; Color[6]=cTarT1; Color[7]=cTarT2;
  Color[8]=cTarT3; Color[9]=cTarT4; Color[10]=cTarT5;
  Depth=Depth*_Point;
  Direction=1; Last=0; Refresh=1;
  for(i=0;i<6;i++)
    {
    if(ObjectFind(0,sn)!=0)
      {

// Set up horizontal and vertical lines
      ObjectCreate(0,Name[i],OBJ_HLINE,0,0,0);
      ObjectSetInteger(0,Name[i],OBJPROP_COLOR,Color[i]);
      ObjectSetInteger(0,Name[i],OBJPROP_WIDTH,1);
      ObjectSetInteger(0,Name[i],OBJPROP_STYLE,STYLE_DOT);
//    ObjectSetString(0,Name[i],OBJPROP_TEXT,Name[i]);// Object description
      }
    }
  if(VLine==true)
    {
    for(i=6;i<11;i++)
      {
      if(ObjectFind(0,sn)!=0)
        {
        ObjectCreate(0,Name[i],OBJ_VLINE,0,0,0);
        ObjectSetInteger(0,Name[i],OBJPROP_COLOR,Color[i]);
        ObjectSetInteger(0,Name[i],OBJPROP_WIDTH,1);
        ObjectSetInteger(0,Name[i],OBJPROP_STYLE,STYLE_DOT);
//      ObjectSetString(0,Name[i],OBJPROP_TEXT,Name[i]);// Object description
        }
      }
    }
  }

// Add function when the indicator is removed from the graph, graphic objects are deleted from the indicator
void OnDeinit(const int reason)
  {
  int i;

  for(i=0;i<11;i++) ObjectDelete(0,Name[i]);
  ChartRedraw();
  return;
  }
```

Now, let's calculate the indicator buffers:

```
// Function of iteration of the indicator
int OnCalculate(const int total,        // Size of the input timeseries
                const int calculated,   // Processed bars call
                const datetime &time[], // Array with time values
                const double &open[],   // Array with opening prices
                const double &high[],   // Array for copying the maximum prices
                const double &low[],    // Array of minimum prices
                const double &close[],  // The closing price array
                const long &tick[],     // Parameter containing the history of the tick volume
                const long &real[],     // Real volume
                const int &spread[])    // An array containing the spreads history

  {
  int i,start;
  bool set;
  double a;

// Set the bar check
  if(calculated<=0)
    {
    start=total-NumBars; if(start<0)start=0;

// Initialize the buffers with empty values
    Last=start; ArrayInitialize(ColorBuffer,0);
    ArrayInitialize(DiNapoliH,0); ArrayInitialize(DiNapoliL,0);
    }

// Calculation of a new bar
  else start=calculated-1;
  for(i=start;i<total-1;i++)
    {
    set=false; DiNapoliL[i]=0; DiNapoliH[i]=0;
    if(Direction>0)
      {
      if(high[i]>DiNapoliH[Last])
        {
        DiNapoliH[Last]=0; DiNapoliH[i]=high[i];
        if(low[i]<high[Last]-Depth)
          {
          if(open[i]<close[i])
            {
            DiNapoliH[Last]=high[Last];
            A=C; B=high[Last]; C=low[i];
            AT=CT; BT=time[Last]; CT=time[i];
            Refresh=1;
            }
          else
            {
            Direction=-1;
            A=B; B=C; C=high[i];
            AT=BT; BT=CT; CT=time[i];
            Refresh=1;
            }
          DiNapoliL[i]=low[i];
          }

// Set the line colors
        ColorBuffer[Last]=0; Last=i; ColorBuffer[Last]=1;
        set=true;
        }
      if(low[i]<DiNapoliH[Last]-Depth&&(!set||open[i]>close[i]))
        {
        DiNapoliL[i]=low[i];
        if(high[i]>DiNapoliL[i]+Depth&&open[i]<close[i])
          {
          DiNapoliH[i]=high[i];
          A=C; B=high[Last]; C=low[i];
          AT=CT; BT=time[Last]; CT=time[i];
          Refresh=1;
          }
        else
          {
          if(Direction>0)
            {
            A=B; B=C; C=high[Last];
            AT=BT; BT=CT; CT=time[Last];
            Refresh=1;
            }
          Direction=-1;
          }

// Set the line colors
        ColorBuffer[Last]=0; Last=i; ColorBuffer[Last]=1;
        }
      }
    else
      {
      if(low[i]<DiNapoliL[Last])
        {
        DiNapoliL[Last]=0; DiNapoliL[i]=low[i];
        if(high[i]>low[Last]+Depth)
          {
          if(open[i]>close[i])
            {
            DiNapoliL[Last]=low[Last];
            A=C; B=low[Last]; C=high[i];
            AT=CT; BT=time[Last]; CT=time[i];
            Refresh=1;
            }
          else
            {
            Direction=1;
            A=B; B=C; C=low[i];
            AT=BT; BT=CT; CT=time[i];
            Refresh=1;
            }
          DiNapoliH[i]=high[i];
          }

// Set the line colors
        ColorBuffer[Last]=0; Last=i; ColorBuffer[Last]=1;
        set=true;
        }
      if(high[i]>DiNapoliL[Last]+Depth&&(!set||open[i]<close[i]))
        {
        DiNapoliH[i]=high[i];
        if(low[i]<DiNapoliH[i]-Depth&&open[i]>close[i])
          {
          DiNapoliL[i]=low[i];
          A=C; B=low[Last]; C=high[i];
          AT=CT; BT=time[Last]; CT=time[i];
          Refresh=1;
          }
        else
          {
          if(Direction<0)
            {
            A=B; B=C; C=low[Last];
            AT=BT; BT=CT; CT=time[Last];
            Refresh=1;
            }
          Direction=1;
          }
// Set the line colors
        ColorBuffer[Last]=0; Last=i; ColorBuffer[Last]=1;
        }
      }
    DiNapoliH[total-1]=0; DiNapoliL[total-1]=0;
    }
//------------
  if(Refresh==1)
    {
```

The final cycle of calculating the indicator:

```
// Check the number of bars for sufficiency for calculation
    Refresh=0; a=B-A;
    Price[0]=NormalizeDouble(a*0.318+C,_Digits);           // Start;
    Price[1]=C;                                            // Stop
    Price[2]=NormalizeDouble(a*0.618+C,_Digits);           // Target№1
    Price[3]=a+C;                                          // Target№2;
    Price[4]=NormalizeDouble(a*1.618+C,_Digits);           // Target№3;
    Price[5]=NormalizeDouble(a*2.618+C,_Digits);           // Target№4;
    for(i=0;i<6;i++) ObjectMove(0,Name[i],0,time[total-1],Price[i]);
    if(VLine==true)
      {

// Return the value rounded to the nearest integer of the specified value
      a=(double)(BT-AT);
      Time[0]=(datetime)MathRound(a*0.318)+CT;             // Temporary goal number №1
      Time[1]=(datetime)MathRound(a*0.618)+CT;             // Temporary goal number №2
      Time[2]=(datetime)MathRound(a)+CT;                   // Temporary goal number №3
      Time[3]=(datetime)MathRound(a*1.618)+CT;             // Temporary goal number №4
      Time[4]=(datetime)MathRound(a*2.618)+CT;             // Temporary goal number №5
      for(i=6;i<11;i++) ObjectMove(0,Name[i],0,Time[i-6],open[total-1]);
      }
    ChartRedraw();

// If the direction is changed then turn on the audio playback
    if(Sound==true&&calculated>0)PlaySound(SoundFile);
    }
  return(total);
  }
//------------------------------------------------------------------------------------
```

The indicator is attached below.

### 8\. Conclusion

I hope, this article has provided you with sufficient data on applying DiNapoli method in trading. The DiNapoli levels offer an original approach to working with standard Fibo levels and extensions. The core principle of working with levels remains the same. DiNapoli has simply introduced a number of new rules yielding reliable results on the market when properly applied.

**Programs used in the article:**

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | DiNapoli  Levels.mq5 | Indicator | The indicator for DiNapoli levels auto calculation and plotting |

### 9\. Reference

1. Joe DiNapoli. DiNapoli Levels: The Practical Application of Fibonacci Analysis to Investment Markets. Coast Investment Software, Inc; Second Edition (1998)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/3061](https://www.mql5.com/ru/articles/3061)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3061.zip "Download all attachments in the single ZIP archive")

[DiNapoli\_Levels.mq5](https://www.mql5.com/en/articles/download/3061/dinapoli_levels.mq5 "Download DiNapoli_Levels.mq5")(22.35 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/208343)**
(19)


![Moneyman31](https://c.mql5.com/avatar/avatar_na2.png)

**[Moneyman31](https://www.mql5.com/en/users/moneyman31)**
\|
21 Nov 2017 at 15:01

**Roman Vashchilin:**

Yes, the screenshot has not been replaced by the correct one, we will soon fix it.

**Roman Vashchilin:**

Correctly set the levels you need as in the first screenshot.

Hi, please is ther any indicator for mt4 for dinapoli levels? I dont like the indicator name dinapoli targets. is it the same, dinapoli levels indicator and dinapoli targets?

![Denis Tikhonov](https://c.mql5.com/avatar/avatar_na2.png)

**[Denis Tikhonov](https://www.mql5.com/en/users/denist)**
\|
28 Feb 2018 at 12:53

You have the wrong indicator showing the deal in the wrong direction )) The deal should be after the price rebound, in the direction of the trend, and the indicator [draws the](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_draw_type "MQL5 Documentation: Drawing Styles") stop loss [line](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_draw_type "MQL5 Documentation: Drawing Styles") at the current maximum ))


![Evgeniy Scherbina](https://c.mql5.com/avatar/2014/4/53426E3A-A025.jpg)

**[Evgeniy Scherbina](https://www.mql5.com/en/users/nume)**
\|
29 Dec 2018 at 10:13

I'll say something and someone will be happy for me.


![Carl Max](https://c.mql5.com/avatar/2018/6/5B32AF40-C723.jpg)

**[Carl Max](https://www.mql5.com/en/users/carlmax)**
\|
3 Jun 2020 at 15:18

This is a very fantastic indicator. Thank you so much for sharing this.

![mschesnokov](https://c.mql5.com/avatar/avatar_na2.png)

**[mschesnokov](https://www.mql5.com/en/users/mschesnokov)**
\|
7 Dec 2021 at 12:22

Hi, I have a question. Why the indicator DiNapoli\_Levels does not make further calculations on time lines? Or I don't understand something?

[![](https://c.mql5.com/3/375/3520360477855__1.png)](https://c.mql5.com/3/375/3520360477855.png "https://c.mql5.com/3/375/3520360477855.png")

![Forecasting market movements using the Bayesian classification and indicators based on Singular Spectrum Analysis](https://c.mql5.com/2/27/MQL5-avatar-SSAtrend-001__1.png)[Forecasting market movements using the Bayesian classification and indicators based on Singular Spectrum Analysis](https://www.mql5.com/en/articles/3172)

The article considers the ideology and methodology of building a recommendatory system for time-efficient trading by combining the capabilities of forecasting with the singular spectrum analysis (SSA) and important machine learning method on the basis of Bayes' Theorem.

![An example of an indicator drawing Support and Resistance lines](https://c.mql5.com/2/28/MQL5-avatar-SupportLines-001.png)[An example of an indicator drawing Support and Resistance lines](https://www.mql5.com/en/articles/3186)

The article provides an example of how to implement an indicator for drawing support and resistance lines based on formalized conditions. In addition to having a ready-to-use indicator, you will see how simple the indicator creation process is. You will also learn how to formulate conditions for drawing any desired line by changing the indicator code.

![Cross-Platform Expert Advisor: Money Management](https://c.mql5.com/2/28/Cross_Platform_Expert_Advisor__1.png)[Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)

This article discusses the implementation of money management method for a cross-platform expert advisor. The money management classes are responsible for the calculation of the lot size to be used for the next trade to be entered by the expert advisor.

![Graphical Interfaces X: Text selection in the Multiline Text box (build 13)](https://c.mql5.com/2/27/MQL5-avatar-XRedHighlight-001__1.png)[Graphical Interfaces X: Text selection in the Multiline Text box (build 13)](https://www.mql5.com/en/articles/3197)

This article will implement the ability to select text using various key combinations and deletion of the selected text, similar to the way it is done in any other text editor. In addition, we will continue to optimize the code and prepare the classes to move on to the final process of the second stage of the library's evolution, where all controls will be rendered as separate images (canvases).

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/3061&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069322961783030570)

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