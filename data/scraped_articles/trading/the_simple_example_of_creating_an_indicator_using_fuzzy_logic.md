---
title: The Simple Example of Creating an Indicator Using Fuzzy Logic
url: https://www.mql5.com/en/articles/178
categories: Trading, Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:29:52.019353
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/178&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062509760672080766)

MetaTrader 5 / Trading


### Introduction

The use of various methods for financial markets analysis becomes more and more popular among traders in recent years. I would like to make my contribution and show how to make a good indicator by writing a couple of dozen code lines. Also, I will briefly reveal you the basics of fuzzy logic.

Anyone who is interested in this issue and wants to explore it more deeply can read the following works:

1.  Leonenkov А. " [Fuzzy Simulation in MATLAB and fuzzyTECH](https://www.mql5.com/go?link=http://www.ozon.ru/context/detail/id/1401809/ "http://www.ozon.ru/context/detail/id/1401809/")" (in Russian).

2\.  Bocharnikov V." [Fuzzy Technology: Mathematical Background. Simulation Practice in Economics](https://www.mql5.com/go?link=http://www.ozon.ru/context/detail/id/120364/ "http://www.ozon.ru/context/detail/id/120364/")" (in Russian).

3.  S.N. Sivanandam, S. Sumathi, S.N. Deepa. [Introduction to Fuzzy Logic using MATLAB.](https://www.mql5.com/go?link=https://www.amazon.com/Introduction-Fuzzy-Logic-using-MATLAB/dp/3540357807/ref=sr_1_1?s=books&ie=UTF8&qid=1336112121&sr=1-1 "http://www.amazon.com/Introduction-Fuzzy-Logic-using-MATLAB/dp/3540357807/ref=sr_1_1?s=books&ie=UTF8&qid=1336112121&sr=1-1")

4.  C. Kahraman. [Fuzzy Engineering Economics with Applications (Studies in Fuzziness and Soft Computing).](https://www.mql5.com/go?link=https://www.amazon.com/Engineering-Economics-Applications-Fuzziness-Computing/dp/354070809X/ref=sr_1_1?s=books&ie=UTF8&qid=1336112303&sr=1-1 "http://www.amazon.com/Engineering-Economics-Applications-Fuzziness-Computing/dp/354070809X/ref=sr_1_1?s=books&ie=UTF8&qid=1336112303&sr=1-1")

### 1\. Fuzzy Logic Basics

How can we explain to our computing machines the meanings of such simple expressions as "...a little more...", "...too fast...", "...almost nothing..."? In fact, it is quite possible by using the fuzzy sets theory elements, or rather the so-called "membership functions". Here is an example from А. Leonenkov's book:

Let's describe the membership function for the phrase "hot coffee": coffee temperature should be deemed to be in the range from 0 to 100 degrees Celsius for the simple reason that at temperatures below 0 degrees, it will turn into ice, while at temperatures above 100 degrees it will evaporate. It is quite obvious that a cup of coffee with a temperature of 20 degrees cannot be called hot, i.e. the membership function in the "hot" category is equal to 0, while a cup of coffee with a temperature of 70 degrees definitely belongs to the "hot" category and, therefore, the function value equals to 1 in this case.

As for the temperature values ​​that are between these two extreme values, the situation is not so definite. Some people may consider a cup of coffee with a temperature of 55 degrees to be "hot", while others may consider it "not so hot". This is the "fuzziness".

Nevertheless, we can imagine the approximate look of the membership function: it is "monotonically increasing":

![](https://c.mql5.com/2/2/01__1.gif)

The figure above shows the "piecewise linear" membership function.

Thus, the function can be defined by the following analytical expression:

![](https://c.mql5.com/2/2/02__1.gif)

We will use such functions for our indicator.

### 2\. Membership Function

In one way or another, the task of any technical indicator is the determining of the current market state (flat, upward trend, downward trend), as well as generation of market entry/exit signals. How can this be done with the help of membership functions? Easy enough.

First of all, we need to define the boundary conditions. Let us have the following boundary conditions: for «100% upward trend» it will be the crossing of EMA having period 2, based on typical price (H+L+C)/3 with Envelopes upper border having parameters 8, 0.08, SMA, Close, while for «100% downward trend» it will be the crossing of the same EMA with Envelopes lower border. Everything that is located between these conditions will be assumed to be flat. Let's add one more envelope having the parameters 32, 0.15, SMA, Close.

As a result, we will get two identical membership functions. The buy signal will be activated when both functions are equal to 1, while the sell signal will be activated when both functions are equal to -1, respectively. Since it is convenient to build charts with the range from -1 to 1, the resulting chart will be obtained as the arithmetic mean of two functions F(x)= (f1(x)+f2(x))/2.

That is how it looks on the chart:

![](https://c.mql5.com/2/2/03__1.gif)

In this case the membership function will have the following graphical representation:

![](https://c.mql5.com/2/2/04__1.gif)

Analytically, it can be written as follows:

![](https://c.mql5.com/2/2/05__1.gif),

where a and b are upper and lower envelope lines, respectively, while х is a value of EMA(2).

With the function defined, we can now move further to writing the indicator code.

### 3\. Creating the Program Code

First of all, we should define what and how we are going to draw.

Membership function calculations results will be displayed as the line - red and blue, respectively.

The arithmetic mean will be displayed as a histogram from the zero line and painted in one of the five colors depending on the resulting function value:

![](https://c.mql5.com/2/2/colors.png)

[DRAW\_COLOR\_HISTOGRAM](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles) drawing style will be used for that.

Let's draw blue and red rectangles as buy/exit signals over the histogram bars, the values of which are equal to 1 or -1.

Now, it's time to run MetaEditor and start. New->Custom Indicator->Next... Fill in the "Parameters" field:

![](https://c.mql5.com/2/2/06.png)

Create the buffers:

![](https://c.mql5.com/2/2/07.png)

After clicking "Finish" button we receive a source code and start improving it.

First of all, let's define the number of buffers. Seven of them have already been created by the Wizard (5 for data, 2 for color). We
need 5 more.

```
#property indicator_minimum -1.4 // Setting fractional values
#property indicator_maximum 1.4  // Expert Advisors wizard ignores fractional parts for some reason
#property indicator_buffers 12   // Changing the value from 7 to 12 (5 more buffers have been added)
```

Let's edit the input parameters:

```
input string txt1="----------";
input int                  Period_Fast=8;
input ENUM_MA_METHOD        Method_Fast = MODE_SMA; /*Smoothing method*/ //moving average smoothing method
input ENUM_APPLIED_PRICE    Price_Fast  = PRICE_CLOSE;
input double               Dev_Fast=0.08;
input string txt2="----------";
input int                  Period_Slow=32;
input ENUM_MA_METHOD        Method_Slow = MODE_SMA;
input ENUM_APPLIED_PRICE    Price_Slow  = PRICE_CLOSE;
input double               Dev_Slow=0.15;  /*Deviation parameter*/
input string txt3="----------";
input int                  Period_Signal=2;
input ENUM_MA_METHOD        Method_Signal = MODE_EMA;
input ENUM_APPLIED_PRICE    Price_Signal  = PRICE_TYPICAL;
input string txt4="----------";
```

Comments following the declared variable are very handy. The text of the comments is inserted into the indicator parameters window.

The possibility to create lists is also very useful:

![](https://c.mql5.com/2/2/08.png)

Reserving the variables for the indicator handles and buffers:

```
int Envelopes_Fast;     // Fast envelope
int Envelopes_Slow;     // Slow envelope
int MA_Signal;          // Signal line

double Env_Fast_Up[];   // Fast envelope upper border
double Env_Fast_Dn[];   // Fast envelope lower border

double Env_Slow_Up[];   // Slow envelope upper border
double Env_Slow_Dn[];   // Slow envelope lower border

double Mov_Sign[];      // Signal line
```

Now, move to [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function.

Let's add some beauty: specify the indicator name and remove extra decimal zeros:

```
IndicatorSetInteger(INDICATOR_DIGITS,1); // setting display accuracy, we do not need some outstanding accuracy values
string name;                           // indicator name
StringConcatenate(name, "FLE ( ", Period_Fast, " , ", Dev_Fast, " | ", Period_Slow, " , ", Dev_Slow, " | ", Period_Signal, " )");
IndicatorSetString(INDICATOR_SHORTNAME,name);
```

and add the missing buffers:

```
SetIndexBuffer(7,Env_Fast_Up,INDICATOR_CALCULATIONS);
SetIndexBuffer(8,Env_Fast_Dn,INDICATOR_CALCULATIONS);
SetIndexBuffer(9,Env_Slow_Up,INDICATOR_CALCULATIONS);
SetIndexBuffer(10,Env_Slow_Dn,INDICATOR_CALCULATIONS);
SetIndexBuffer(11,Mov_Sign,INDICATOR_CALCULATIONS);
```

[INDICATOR\_CALCULATIONS](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties) parameter means that the buffer data is needed only for intermediate calculations. It will not be displayed on the chart.

Note how the indicators with color buffers are declared:

```
SetIndexBuffer(4,SignalBuffer1,INDICATOR_DATA);      // All indicator buffers at first
SetIndexBuffer(5,SignalBuffer2,INDICATOR_DATA);      // as this is Color Histogram2, then it has 2 data buffers
SetIndexBuffer(6,SignalColors,INDICATOR_COLOR_INDEX);// the color buffer comes next.
```

Filling the handles:

```
Envelopes_Fast = iEnvelopes(NULL,0,Period_Fast,0,Method_Fast,Price_Fast,Dev_Fast);
Envelopes_Slow = iEnvelopes(NULL,0,Period_Slow,0,Method_Slow,Price_Slow,Dev_Slow);
MA_Signal      = iMA(NULL,0,Period_Signal,0,Method_Signal,Price_Signal);
```

All works with [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function are over.

Now let's create the function that will calculate the membership function value:

```
double Fuzzy(double x,double a, double c)
{
double F;
     if (a<x)          F=1;                 // 100% uptrend
else if (x<=a && x>=c)  F=(1-2*(a-x)/(a-c));// Flat
else if (x<c)           F=-1;               // 100% downtrend
return (F);
}
```

Preparations are over. Variables and buffers are declared, handles are assigned.

Now it is time to proceed with [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) basic function.

First of all, let's write the values of the necessary indicators into the intermediate buffers. Use [CopyBuffer()](https://www.mql5.com/en/docs/series/copybuffer) function:

```
CopyBuffer(Envelopes_Fast,  // Indicator handle
           UPPER_LINE,      // Indicator buffer
           0,              // The point to start 0 - from the very beginning
           rates_total,    // How many to be copied - All
           Env_Fast_Up);   // The buffer the values are written in
// - the rest are done in a similar way
CopyBuffer(Envelopes_Fast,LOWER_LINE,0,rates_total,Env_Fast_Dn);
CopyBuffer(Envelopes_Slow,UPPER_LINE,0,rates_total,Env_Slow_Up);
CopyBuffer(Envelopes_Slow,LOWER_LINE,0,rates_total,Env_Slow_Dn);
CopyBuffer(MA_Signal,0,0,rates_total,Mov_Sign);
```

Here we must add the code for the calculations optimization (recalculation of only the last bar is performed):

```
// declaring start variable for storing the index of the bar, recalculation of the indicator buffers will be
// carried out from.

int start;
if (prev_calculated==0)  // in case no bars have been calculated
    {
    start = Period_Slow; // not all indicators have been calculated up to this value, therefore, there is no point in executing the code
    }
else start=prev_calculated-1;

for (int i=start;i<rates_total;i++)
      {
      // All remaining code will be written here
      }
```

Not much of the code has left.

Setting x, a, b parameters, performing the calculation of the membership function value and writing it into the appropriate buffer:

```
double x = Mov_Sign[i]; // Signal
// Setting the first membership function parameters:
double a1 = Env_Fast_Up[i]; // Upper border
double b1 = Env_Fast_Dn[i];
// setting the first membership function value and writing it to the buffer
Rule1Buffer[i] = Fuzzy(x,a1,b1);
// Setting the second membership function parameters:
double a2 = Env_Slow_Up[i]; // Upper border
double b2 = Env_Slow_Dn[i];
// setting the second membership function value and writing it to the buffer
Rule2Buffer[i] = Fuzzy(x,a2,b2);
```

Two indicator lines are built.

Now let's calculate the resulting value.

```
ResultBuffer[i] = (Rule1Buffer[i]+Rule2Buffer[i])/2;
```

Then we should paint the histogram bars in appropriate colors: as we have five colors, then ResultColors\[i\] can have any value from 0 to 4.

Generally, the number of possible colors is 64. Therefore, it is a great opportunity for applying one's creative abilities.

```
for (int ColorIndex=0;ColorIndex<=4;ColorIndex++)
    {
    if (MathAbs(ResultBuffer[i])>0.2*ColorIndex && MathAbs(ResultBuffer[i])<=0.2*(ColorIndex+1))
        {
        ResultColors[i] = ColorIndex;
        break;
        }
    }
```

Then we should draw the signal rectangles. We will use [DRAW\_COLOR\_HISTOGRAM2](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles) drawing style.

It has two data buffers with a histogram bar and one color buffer being built between them.

Data buffers values will always be the same: 1.1 and 1.3 for a buy signal, -1.1 and -1.3 for a sell signal, respectively.

[EMPTY\_VALUE](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants) will mean the signal absence.

```
      if (ResultBuffer[i]==1)
        {
        SignalBuffer1[i]=1.1;
        SignalBuffer2[i]=1.3;
        SignalColors[i]=1;
        }
      else if (ResultBuffer[i]==-1)
        {
        SignalBuffer1[i]=-1.1;
        SignalBuffer2[i]=-1.3;
        SignalColors[i]=0;
        }
      else
        {
        SignalBuffer1[i]=EMPTY_VALUE;
        SignalBuffer2[i]=EMPTY_VALUE;
        SignalColors[i]=EMPTY_VALUE;
        }
```

Click "Compile" and voila!

![](https://c.mql5.com/2/2/EURUSDM15.png)

### Conclusion

What else can be added? In this article I touched on the most basic approach to fuzzy logic.

There is enough space here for various experiments. For example, we can use the following function:

![](https://c.mql5.com/2/2/09.png)

I think, it will not be difficult for you to write the analytical expression for it and find suitable conditions.

Good luck!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/178](https://www.mql5.com/ru/articles/178)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/178.zip "Download all attachments in the single ZIP archive")

[fuzzy\_envelopes.mq5](https://www.mql5.com/en/articles/download/178/fuzzy_envelopes.mq5 "Download fuzzy_envelopes.mq5")(9.08 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6654)**
(4)


![Valerii Mazurenko](https://c.mql5.com/avatar/2014/5/536241C3-76F3.jpg)

**[Valerii Mazurenko](https://www.mql5.com/en/users/notused)**
\|
7 Oct 2010 at 19:53

Fuzzy logic does not even smell of fuzzy logic. Not only the [belonging function](https://www.mql5.com/en/articles/178 "Article: A simple example of building an indicator using fuzzy logic") was taken out of the "canonical" range \[0,1\] to \[-1,1\], but the calculation of the belonging function was presented as a fuzzy indicator, which does not correspond to reality. The defuzzification stage has not been passed. Buy/sell conditions are made up. No credit. It would be better to rename the article to "Strength of the indicator based on probability", because there is no fuzzy logic here, but only probability, which also needs to be normalised


![certus](https://c.mql5.com/avatar/avatar_na2.png)

**[certus](https://www.mql5.com/en/users/certus)**
\|
13 Apr 2014 at 21:04

**notused:**

Fuzzy logic does not even smell of fuzzy logic. Not only the belonging function was taken out of the "canonical" range \[0,1\] to \[-1,1\], but the calculation of the belonging function was presented as a fuzzy indicator, which does not correspond to reality. The defuzzification stage has not been passed. Buy/sell conditions are made up. No credit. It would be better to rename the article to "Strength of the indicator based on probability", because there is no fuzzy logic here, but only probability, which also needs to be normalised.

+1

I came to write the same thing, and here is your exhaustive comment :)

![Rasoul Mojtahedzadeh](https://c.mql5.com/avatar/2015/6/558F004E-DFBD.png)

**[Rasoul Mojtahedzadeh](https://www.mql5.com/en/users/rasoul)**
\|
6 Jul 2014 at 19:55

This indicator in not based on [Fuzzy logic](https://www.mql5.com/en/articles/178 "Article: A simple example of building an indicator using fuzzy logic") (or Fuzzy Inference System) nor an indicator based on probability theory. However, I like the idea of identifying 100% uptrend/downtrend of price action which is presented in this article.


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
17 Mar 2018 at 16:47

The title is misleading, there is no [fuzzy logic](https://www.mql5.com/en/articles/178 "Article: A simple example of building an indicator using fuzzy logic") here, I suppose the author didn't read his reference books or misunderstood them. For the rest, it's a classical "commented code" boring article.

Don't waste your time with this article.

![Synthetic Bars - A New Dimension to Displaying Graphical Information on Prices](https://c.mql5.com/2/17/995_16.png)[Synthetic Bars - A New Dimension to Displaying Graphical Information on Prices](https://www.mql5.com/en/articles/1353)

The main drawback of traditional methods for displaying price information using bars and Japanese candlesticks is that they are bound to the time period. It was perhaps optimal at the time when these methods were created but today when the market movements are sometimes too rapid, prices displayed in a chart in this way do not contribute to a prompt response to the new movement. The proposed price chart display method does not have this drawback and provides a quite familiar layout.

![AutoElliottWaveMaker - MetaTrader 5 Tool for Semi-Automatic Analysis of Elliott Waves](https://c.mql5.com/2/0/ElliottWaveMaker2_0.png)[AutoElliottWaveMaker - MetaTrader 5 Tool for Semi-Automatic Analysis of Elliott Waves](https://www.mql5.com/en/articles/378)

The article provides a review of AutoElliottWaveMaker - the first development for Elliott Wave analysis in MetaTrader 5 that represents a combination of manual and automatic wave labeling. The wave analysis tool is written exclusively in MQL5 and does not include external dll libraries. This is another proof that sophisticated and interesting programs can (and should) be developed in MQL5.

![Trader's Kit: Drag Trade Library](https://c.mql5.com/2/17/902_26.png)[Trader's Kit: Drag Trade Library](https://www.mql5.com/en/articles/1354)

The article describes Drag Trade Library that provides functionality for visual trading. The library can easily be integrated into virtually any Expert Advisor. Your Expert Advisor can be transformed from an automat into an automated trading and information system almost effortless on your side by just adding a few lines of code.

![Who Is Who in MQL5.community?](https://c.mql5.com/2/0/whoiswho.png)[Who Is Who in MQL5.community?](https://www.mql5.com/en/articles/386)

The MQL5.com website remembers all of you quite well! How many of your threads are epic, how popular your articles are and how often your programs in the Code Base are downloaded – this is only a small part of what is remembered at MQL5.com. Your achievements are available in your profile, but what about the overall picture? In this article we will show the general picture of all MQL5.community members achievements.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/178&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062509760672080766)

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