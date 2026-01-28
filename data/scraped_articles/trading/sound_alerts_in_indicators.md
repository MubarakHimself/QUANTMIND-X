---
title: Sound Alerts in Indicators
url: https://www.mql5.com/en/articles/1448
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:24:39.622100
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=maimxvgupgghbrddebyjukajkngsbfyw&ssn=1769181878786491744&ssn_dr=0&ssn_sr=0&fv_date=1769181878&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1448&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Sound%20Alerts%20in%20Indicators%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176918187834751716&fz_uniq=5069414998637216950&sv=2552)

MetaTrader 4 / Trading


### Introduction

Though automated trading becomes more and more popular, many traders still practice manual trading. So, where an Expert Advisor needs some milliseconds to evaluate the current market situation, a human will spend much time, power and - which is most important - attention.

As a couple of years before, many traders use one or more [Technical Indicators](https://www.metatrader5.com/en/terminal/help/charts_analysis/indicators "https://www.metatrader5.com/en/terminal/help/charts_analysis/indicators"). Some strategies consider indicator values on several timeframes simultaneously.

So, how can one "catch" an important signal? There are several choices:

- write an Expert Advisor that would analyze the market and alert about important events;
- sit in front of the monitor and , switching between tens of charts, try to analyze the information from all of them;
- add an alerting system into all indicators used.

The first choice is, in my opinion, the most proper. But it demands either programming skills or money to pay for realization. The second way is very time consuming, tiring, and inefficient. The third choice is a cross between the former two ways. One needs much fewer skills and less time to implement it, but it can really better the lot of the user trading manually.

It is the implementation of the third choice that the article is devoted to. After having read it, every trader will be able to add convenient alerts into indicators.

### Types of Alerts

There are many ways to interpret indicators. People can differently understand the meaning of even MetaTrader 4 Client Terminal indicators, not to say about various custom indicators...

Somebody buys when the main line of MACD touches the signal line, another trader waits until it intersects the zero line, and somebody opens a long position when MACD is below 0 and starts moving up. I don't feel myself able to count all possible interpreting variations, so I will just describe the principles of how an alerting block can be added into an indicator. Then you will be able to add any kind of alerts into practically all indicators according to your taste.

The most possible alerts are listed below:

- intersection of two lines of an indicator (lie in the example above - the main and the signal line of MACD);
- intersection of the indicator line and a certain level (for example, the main line of MACD and zero line, Stoсhastic and levels of 70 and 30, CCI and levels of -100 and 100);
- reversed moving of the indicator (for example, AC and AO, normal MA);
- changed location towards price (Parabolic SAR);
- appearing arrow above or below the price value (Fractals).

There are probably some other interpretations that are forgotten or even not known to me, so we will describe the five ones listed above.

### Ways of Alerting

MetaTrader 4 and MQL4 allow implementation of several ways of both visual and audio alerting:

- a usual screen message (function Comment);
- a records in the log (function Print);
- a message window plus a sound (function Alert);
- a special sound, a file to be selected and played (function PlaySound).

Besides, there are functions for sending a file to the FTP server (function SendFTP()), displaying a message/dialog box (MessageBox()), and sending mails (SendMail()). Function SendFTP() will hardly be demanded by a regular user, function MessageBox() does not suit for being used in an indicator since it stops its operation until the message box is closed, function SendMail(), though it is good for sending SMS, is rather "dangerous" in use - having drawn a number of indicators in a chart, you will provide yourselves with an endless and uncontrolled stream of messages. The function may be used, but it would be better to use if from an EA, for instance, by sending a message when an alert occurs on several indicators simultaneously, paying much attention to it.

In this article, we will consider only audio and visual ways of alerting in the MetaTrader 4 Client Terminal.

One of the most convenient and the simplest of them is function Alert since it contains both text and sound. Besides, the terminal stores the Alerts history, so it is possible to see what signal came an hour ago.

But tastes differ, it's a common knowledge. So I will make something like a preform for all the above-mentioned methods (except for SendFTP, MessageBox, SendMail), and you will just choose a suitable one.

### Alert Frequency Filter

If you have ever used alerts in indicators, you certainly had to deal with their overfrequency, especially on smaller timeframes. There are some ways to solve this problem:

- To define alerts on bars already formed. This solution would be the most proper.
- Alternate alerts - sell after buy and vice versa (it would be a very logical way, too, that can be used together with other ones).
- Make a pause between alerts (not a good idea).
- Give only one alert per bar (this limitation is rather affected limitation).

Whether to use alerts from a zero, not yet formed bar, is everyone's personal business. I, for instance, suppose this to be wrong. But there are indicators that need instant response - one bar is too much for them. So we will allow users to make their choice. Several alerts to buy would hardly have any sense, so we will alternate all alerts. We will not introduce any artificial pauses I suppose. If they are really necessary, this fact will be known from comments to this article.

Thus, let us start realization.

### Alert One - Intersection of Two Lines of an Indicator

Let us start with the MACD that has been given in examples above.

Our main task is to detect in what arrays the indicator lines are stored. Let us look into the code for this:

```
//---- indicator settings
#property  indicator_separate_window
#property  indicator_buffers 2
#property  indicator_color1  Silver
#property  indicator_color2  Red
#property  indicator_width1  2
//---- indicator parameters
extern int FastEMA = 12;
extern int SlowEMA = 26;
extern int SignalSMA = 9;
//---- indicator buffers
double MacdBuffer[];
double SignalBuffer[];
```

Please note the comment of "indicator buffers" is that what we were looking for. Such arrays mostly have intuitively comprehensive names (MacdBuffer is the MACD main line value buffer, SignalBuffer - buffer of the signal line) and are always located outside of functions init, deinit, start.

If there are many arrays and it is difficult to see which of them is necessary, look into function init - all arrays shown in the chart are anchored to a certain number using function SetIndexBuffer:

```
int init()
  {
//---- drawing settings
   SetIndexStyle(0, DRAW_HISTOGRAM);
   SetIndexStyle(1, DRAW_LINE);
   SetIndexDrawBegin(1, SignalSMA);
   IndicatorDigits(Digits + 1);
//---- indicator buffers mapping
   SetIndexBuffer(0, MacdBuffer);
   SetIndexBuffer(1, SignalBuffer);
//---- name for DataWindow and indicator subwindow label
   IndicatorShortName("sMACD(" + FastEMA + "," + SlowEMA + "," + SignalSMA + ")");
   SetIndexLabel(0, "sMACD");
   SetIndexLabel(1, "sSignal");
//---- initialization done
   return(0);
  }
```

This is the sequence (from 0 to 7), in which the indicator line values are shown in the DataWindow. Names that you can see there are given by function SetIndexLabel - this is the third identification method.

Now, when we know where the necessary data are stored, we can start realization of the alerting block. For this, let's go to the very end of function start - just above the preceding operator return:

```
   for(i = 0; i < limit; i++)
       SignalBuffer[i] = iMAOnArray(MacdBuffer, Bars,S ignalSMA, 0, MODE_SMA, i);
//---- done

// we will add our code here

   return(0);
  }
//+------------------------------------------------------------------+
```

In no case, the alerting block should be added in the indicator's calculating loop - this will slow execution and give no effect.

So, let's start writing our "composition":

```
    //---- Static variables where the last bar time
    //---- and the last alert direction are stored
    static int PrevSignal = 0, PrevTime = 0;

    //---- If the bar selected to be analyzed is not a zero bar,
    //     there is no sense to check the alert
    //---- several times. If no new bar starts to be formed, quit.
    if(SIGNAL_BAR > 0 && Time[0] <= PrevTime )
        return(0);
    //---- Mark that this bar was checked
    PrevTime = Time[0];
```

Every time when function start is executed, our code will be executed, as well. Normal variables are zeroized after each execution of the function. So we have declared two static variables to store the latest alert and the calculated bar number.

Then a simple checking follows: we check whether a new bar has started (it only works if SIGNAL\_BAR is more than 0).

By the way, we declared variable SIGNAL\_BAR itself a bit earlier, before function init:

```
double     SignalBuffer[];

//---- Bar number the alert to be searched by
#define SIGNAL_BAR 1

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int init()
  {
```

Please note directive #define - the compiler will just replace variable SIGNAL\_BAR with the given value (1) throughout the code.

Below is the alert code itself:

```
    //---- If the preceding alert was SELL or this is the first launch (PrevSignal=0)
    if(PrevSignal <= 0)
      {
        //---- Check whether the lines have met in the preceding bar:
        if(MacdBuffer[SIGNAL_BAR] - SignalBuffer[SIGNAL_BAR] > 0 &&
           SignalBuffer[SIGNAL_BAR+1] - MacdBuffer[SIGNAL_BAR+1] >= 0)
          {
            //---- If yes, mark that the last alert was BUY
            PrevSignal = 1;
            //---- and display information:
            Alert("sMACD (", Symbol(), ", ", Period(), ")  -  BUY!!!");
//            Print("sMACD (", Symbol(), ", ", Period(), ")  -  BUY!!!");
//            Comment("sMACD (", Symbol(), ", ", Period(), ")  -  BUY!!!");
//            PlaySound("Alert.wav");
          }
      }
```

This is very simple, too. If the preceding alert was SELL, check intersection of lines:

_if the MACD main line value on bar #1 exceeds that of the signal line on bar # 1_

_AND_

_the siganl line value on bar #2 exceeds that of the MACD line on bar #2,_

_then_

_lines have met._ Then mark that the last alert was for BUY and display the informing message. Note the three commented lines - these are three more alert variations. You can decomment or delete any or all of them. I left Alert by default as the most convenient one.

In function PlaySound, it can be specified what wave file should be played. The file must be located in directory MetaTrader 4\\sounds\ and have extension wav. For example, a special sound can be assigned to the BUY alert, another - for the SELL alert, or there can be different sounds for different indicators, etc.

The SELL alert is absolutely the same:

```
    //---- Completely the same for the SELL alert
    if(PrevSignal >= 0)
      {
        if(SignalBuffer[SIGNAL_BAR] - MacdBuffer[SIGNAL_BAR] > 0 &&
           MacdBuffer[SIGNAL_BAR+1] - SignalBuffer[SIGNAL_BAR+1] >= 0)
          {
            PrevSignal = -1;
            Alert("sMACD (", Symbol(), ", ", Period(), ")  -  SELL!!!");
//            Print("sMACD (", Symbol(), ", ", Period(), ")  -  SELL!!!");
//            Comment("sMACD (", Symbol(), ", ", Period(), ")  -  SELL!!!");
//            PlaySound("Alert.wav");
          }
      }
```

### Other Alerts

Now, when we have known the indicator code, it will be much easier for us to write other alerting blocks. Only the "formula" will be changed, the rest of the code will be just copied and pasted.

Alert that signals about touching a certain level is very similar to that of intersection of lines. I added it to Stochastic, but you can make a similar one for any other indicator:

```
    if(PrevSignal <= 0)
      {
        if(MainBuffer[SIGNAL_BAR] - 30.0 > 0 &&
           30.0 - MainBuffer[SIGNAL_BAR+1] >= 0)
          {
            PrevSignal = 1;
            Alert("sStochastic (", Symbol(), ", ", Period(), ")  -  BUY!!!");
          }
      }
    if(PrevSignal >= 0)
      {
        if(70.0 - MainBuffer[SIGNAL_BAR] > 0 &&
           MainBuffer[SIGNAL_BAR+1] - 70.0 >= 0)
          {
            PrevSignal = -1;
            Alert("sStochastic (", Symbol(), ", ", Period(), ")  -  SELL!!!");
          }
      }
```

As you can see, if line %K (MainBuffer) meets level 30 bottom-up, the indicator will say "Buy", whereas it will say "Sell" if level 70 is met top-down.

The third kind of alert is alert informing about the changed direction of movement. We will consider it on the example of AC. Note that five buffers are used in this indicator:

```
//---- indicator buffers
double     ExtBuffer0[];
double     ExtBuffer1[];
double     ExtBuffer2[];
double     ExtBuffer3[];
double     ExtBuffer4[];
```

ExtBuffer3 and ExtBuffer4 are used for intermediate calculations, ExtBuffer0 always stores the indicator value, ExtBuffer2 and ExtBuffer3 "color" columns in 2 colors. Since we need only indicator value, we will use ExtBuffer0:

```
    if(PrevSignal <= 0)
      {
        if(ExtBuffer0[SIGNAL_BAR] - ExtBuffer0[SIGNAL_BAR+1] > 0 &&
           ExtBuffer0[SIGNAL_BAR+2] - ExtBuffer0[SIGNAL_BAR+1] > 0)
          {
            PrevSignal = 1;
            Alert("sAC (", Symbol(), ", ", Period(), ")  -  BUY!!!");
          }
      }
    if(PrevSignal >= 0)
      {
        if(ExtBuffer0[SIGNAL_BAR+1] - ExtBuffer0[SIGNAL_BAR] > 0 &&
           ExtBuffer0[SIGNAL_BAR+1] - ExtBuffer0[SIGNAL_BAR+2] > 0)
          {
            PrevSignal = -1;
            Alert("sAC (", Symbol(), ", ", Period(), ")  -  SELL!!!");
          }
      }
```

If the indicator value was decreasing and then started to increase, we give a BUY alert. If vice versa - SELL alert.

The fourth kind of alert - informing about changed location towards price - is rather rare.

But it sometimes appears, for example, in Parabolic. We will write the "formula" using it as an example:

```
    if(PrevSignal <= 0)
      {
        if(Close[SIGNAL_BAR] - SarBuffer[SIGNAL_BAR] > 0)
          {
            PrevSignal = 1;
            Alert("sParabolic Sub (", Symbol(), ", ", Period(), ")  -  BUY!!!");
          }
      }
    if(PrevSignal >= 0)
      {
        if(SarBuffer[SIGNAL_BAR] - Close[SIGNAL_BAR] > 0)
          {
            PrevSignal = -1;
            Alert("sParabolic Sub(", Symbol(), ", ", Period(), ")  -  SELL!!!");
          }
      }
```

It is all simple here - we compare the indicator value to the bar close price. Note that, if SIGNAL\_BAR is set for 0, every price touch of the Parabolic will be accompanied with an alert.

The last alert informs about appearance of an arrow in the chart. It appears rather rarely in standard indicators, but it is rather popular in custom "pivot finders". I will consider this kind of alerts using indicator Fractals (its source code written in MQL4 can be found in [Code Base: Fractals](https://www.mql5.com/en/code/7982)).

Such indicators have a common feature: they are not equal to 0 (or EMPTY\_VALUE) in those places where they are drawn on a chart. On all other bars their buffers are empty. It means, you only need to compare the buffer value to zero in order to determine the signal:

```
    if(PrevSignal <= 0 )
      {
        if(ExtDownFractalsBuffer[SIGNAL_BAR] > 0)
          {
            PrevSignal = 1;
            Alert("sFractals (", Symbol(), ", ", Period(), ")  -  BUY!!!");
          }
      }
    if(PrevSignal >= 0)
      {
        if(ExtUpFractalsBuffer[SIGNAL_BAR] > 0)
          {
            PrevSignal = -1;
            Alert("sFractals (", Symbol(), ", ", Period(), ")  -  SELL!!!");
          }
      }
```

But, if you attach an indicator with such a code to the chart, you will never receive any alerts. Fractals have a special feature - they use 2 future bars for analyses, so the arrows appear only on bar#2 (the third bar starting with the zero one). So, for alerts to start working it is necessary to set SIGNAL\_BAR as 2:

```
//---- Bar number to search an alert by
#define SIGNAL_BAR 2
```

That's all, and alerts will work!

### Conclusion

The article gives a description of various methods used to add sound alerts into indicators. Such terms as **alert interpreting method** (type of alert), **way of alerting** and **alert frequency filter** are defined.

The following types of alerts are defined and realized:

- intersection of two lines of an indicator;
- intersection of the indicator line and a certain level;
- reversed moving of the indicator;
- changed location towards price;
- appearing arrow above or below the price value.

The following functions are selected for alerts:

- Comment() - displaying a normal message;
- Print() - showing a message in the log;
- Alert() - showing the message in a special window and a sound alert;
- PlaySound() - playing any wave file.

To decrease the alert frequency:

- use bars already formed when determining an alert;
- all alerts alternate - only buy after sell, and vice versa.

I used five indicators that correspond with five types of alerts to study their alerting blocks. You can download the resulting indicators - they are attached to the article.

I hope you can see that there is nothing complicated in adding an alerting block into indicators - everyone can do this.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1448](https://www.mql5.com/ru/articles/1448)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1448.zip "Download all attachments in the single ZIP archive")

[sAccelerator.mq4](https://www.mql5.com/en/articles/download/1448/sAccelerator.mq4 "Download sAccelerator.mq4")(4.38 KB)

[sFractals.mq4](https://www.mql5.com/en/articles/download/1448/sFractals.mq4 "Download sFractals.mq4")(7.4 KB)

[sMACD.mq4](https://www.mql5.com/en/articles/download/1448/sMACD.mq4 "Download sMACD.mq4")(4.25 KB)

[sParabolic\_Sub.mq4](https://www.mql5.com/en/articles/download/1448/sParabolic_Sub.mq4 "Download sParabolic_Sub.mq4")(8.46 KB)

[sStochastic.mq4](https://www.mql5.com/en/articles/download/1448/sStochastic.mq4 "Download sStochastic.mq4")(5.22 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235)
- [Equivolume Charting Revisited](https://www.mql5.com/en/articles/1504)
- [Testing Visualization: Account State Charts](https://www.mql5.com/en/articles/1487)
- [An Expert Advisor Made to Order. Manual for a Trader](https://www.mql5.com/en/articles/1460)
- [Testing Visualization: Trade History](https://www.mql5.com/en/articles/1452)
- [Filtering by History](https://www.mql5.com/en/articles/1441)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39307)**
(7)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
20 Aug 2007 at 19:24

Andrey,

thank for your indications, I really can't do it because I don't understand the
codes, can you help me to get ADXcrosses with alerts signals????, can you help
me in this way???

Really, thank a lot.

Best regards,

Ricardo

Ricardo Portugau

ricardo.portugau@gmail.com

![Mehmet Bastem](https://c.mql5.com/avatar/avatar_na2.png)

**[Mehmet Bastem](https://www.mql5.com/en/users/mehmet)**
\|
29 Aug 2008 at 19:50

how do using in EA?. for sParabolic Sub.mq4

Please a sample .

string result=iCustom(.....);

if result=="BUY" ...

if result =="sell"..

![nozzgile](https://c.mql5.com/avatar/2011/8/4E3BA14D-2EE8.jpg)

**[nozzgile](https://www.mql5.com/en/users/nozzgile)**
\|
3 Aug 2011 at 09:25

//+------------------------------------------------------------------+

//\| DPC Market Range Alert.mq4 \|

//\| Copyright © 2010, MetaQuotes Software Corp. \|

//\| http://wwwDeltaPrimeCapital.com \|

//+------------------------------------------------------------------+

#property copyright "Copyright © 2010, MetaQuotes Software Corp."

#property link "http://wwwDeltaPrimeCapital.com"

[#property indicator\_chart\_window](https://www.mql5.com/en/docs/basis/preprosessor/compilation "MQL5 Documentation: Program Properties (#property)")

extern int ATR = 26;

extern int ATRtf = PERIOD\_D1;

extern int Range = 100;

int PrevAlertTime = 0;

//+------------------------------------------------------------------+

//\| Custom indicator initialization function \|

//+------------------------------------------------------------------+

int init()

{

//\-\-\-\- indicators

//----

return(0);

}

//+------------------------------------------------------------------+

//\| Custom indicator deinitialization function \|

//+------------------------------------------------------------------+

int deinit()

{

//----

//----

return(0);

}

//+------------------------------------------------------------------+

//\| Custom indicator iteration function \|

//+------------------------------------------------------------------+

int start()

{

int counted\_bars=IndicatorCounted();

//----

int limit=Bars-counted\_bars;

if(counted\_bars>0) limit++;counted\_bars--;

for(int i=0; i<limit; i++)

double ADR = iATR(NULL,ATRtf,ATR,i)/Point;

double DR = ( iHigh(NULL,ATRtf,0) - iLow(NULL,ATRtf,0) )/Point;

double DPR = (DR/ADR)\*100;

double alertTag;

if ( (DPR\*Point) > (Range\*Point) )

{if ( alertTag!=Time\[0\])

{PlaySound("news.wav");// buy wav

Alert(Symbol()," M",Period()," ", Symbol(), " Reach ",Range, " %" );}alertTag = Time\[0\];}

//----

return(0);

}

//+------------------------------------------------------------------+

I have minor coding experience, this indicator always beep per tick, can you help me to fix it? thanks

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
31 Dec 2011 at 07:18

**hi ricardo can you do me a favour do you have an ADX with sound alert..is it done are you willing to share it with me ???**

happy new year!!

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
9 Mar 2013 at 15:14

help   plz     alert

#property copyright "?2007 RickD"

#property link      " [http://www.e2e-fx.net/](https://www.mql5.com/go?link=http://www.e2e-fx.net/)"

#define major   1

#define minor   0

#property indicator\_chart\_window

#property indicator\_buffers 2

#property indicator\_color1 Red

#property indicator\_color2 Yellow

#property indicator\_width1  1

#property indicator\_width2  1

extern int Fr.Period = 6;

extern int MaxBars = 500;

double upper\_fr\[\];

double lower\_fr\[\];

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void init() {

SetIndexBuffer(0, upper\_fr);

SetIndexBuffer(1, lower\_fr);

SetIndexEmptyValue(0, 0);

SetIndexEmptyValue(1, 0);

SetIndexStyle(0, DRAW\_ARROW);

SetIndexArrow(0, 234);

SetIndexStyle(1, DRAW\_ARROW);

SetIndexArrow(1, 233);

}

void start()

{

int counted = IndicatorCounted();

if (counted < 0) return (-1);

if (counted > 0) counted--;

int limit = MathMin(Bars-counted, MaxBars);

//-----

double dy = 0;

for (int i=1; i <= 20; i++) {

    dy += 0.3\*(High\[i\]-Low\[i\])/20;

}

for (i=0+Fr.Period; i <= limit+Fr.Period; i++)

{

    upper\_fr\[i\] = 0;

    lower\_fr\[i\] = 0;

    if (is\_upper\_fr(i, Fr.Period)) upper\_fr\[i\] = High\[i\]+dy;

    if (is\_lower\_fr(i, Fr.Period)) lower\_fr\[i\] = Low\[i\]-dy;

}

}

bool is\_upper\_fr(int bar, int period)

{

for (int i=1; i<=period; i++)

{

    if (bar+i >= Bars \|\| bar-i < 0) return (false);

    if (High\[bar\] < High\[bar+i\]) return (false);

    if (High\[bar\] < High\[bar-i\]) return (false);

}

return (true);

}

bool is\_lower\_fr(int bar, int period)

{

for (int i=1; i<=period; i++)

{

    if (bar+i >= Bars \|\| bar-i < 0) return (false);

    if (Low\[bar\] > Low\[bar+i\]) return (false);

    if (Low\[bar\] > Low\[bar-i\]) return (false);

}

return (true);

}

![Simultaneous Displaying of the Signals of Several Indicators from the Four Timeframes](https://c.mql5.com/2/14/325_3.png)[Simultaneous Displaying of the Signals of Several Indicators from the Four Timeframes](https://www.mql5.com/en/articles/1461)

While manual trading you have to keep an eye on the values of several indicators. It is a little bit different from mechanical trading. If you have two or three indicators and you have chosen a one timeframe for trading, it is not a complicated task. But what will you do if you have five or six indicators and your trading strategy requires considering the signals on the several timeframes?

![Displaying of Support/Resistance Levels](https://c.mql5.com/2/14/237_1.png)[Displaying of Support/Resistance Levels](https://www.mql5.com/en/articles/1440)

The article deals with detecting and indicating Support/Resistance Levels in the MetaTrader 4 program. The convenient and universal indicator is based on a simple algorithm. The article also tackles such a useful topic as creation of a simple indicator that can display results from different timeframes in one workspace.

![Alternative Log File with the Use of HTML and CSS](https://c.mql5.com/2/14/385_10.gif)[Alternative Log File with the Use of HTML and CSS](https://www.mql5.com/en/articles/1432)

In this article we will describe the process of writing a simple but a very powerful library for making the html files, will learn to adjust their displaying and will see how they can be easily implemented and used in your expert or the script.

![A Method of Drawing the Support/Resistance Levels](https://c.mql5.com/2/14/233_1.png)[A Method of Drawing the Support/Resistance Levels](https://www.mql5.com/en/articles/1439)

This article describes the process of creating a simple script for detecting the support/resistance levels. It is written for beginners, so you can find the detailed explanation of every stage of the process. However, though the script is very simple, the article will be also useful for advanced traders and the users of the MetaTrader 4 platform. It contains the examples of the data export into the tabular format, the import of the table to Microsoft Excel and plotting the charts for the further detailed analysis.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qgkfmqqiwkesuwjxehacgtqelniacsht&ssn=1769181878786491744&ssn_dr=0&ssn_sr=0&fv_date=1769181878&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1448&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Sound%20Alerts%20in%20Indicators%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176918187834731068&fz_uniq=5069414998637216950&sv=2552)

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