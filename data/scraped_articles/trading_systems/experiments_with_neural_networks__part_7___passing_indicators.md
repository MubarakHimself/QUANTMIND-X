---
title: Experiments with neural networks (Part 7): Passing indicators
url: https://www.mql5.com/en/articles/13598
categories: Trading Systems, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T11:42:44.233684
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/13598&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062660282095937153)

MetaTrader 5 / Trading systems


### Introduction

In the current article, we will talk in more detail about the importance of passing meaningful data, the so-called time series, in a neural network. In particular, we will pass our favorite indicators. To achieve this, I will introduce some new concepts that I use while working with neural networks. Although, I think this is not the limit, and over time I will have a new vision in understanding what and how exactly needs to be passed.

### Background and observations

Reading a large number of articles on this topic, I constantly observe a sad picture of the direct results of trading systems based on neural networks. Many good ideas and algorithms do not bring the desired results.

The same picture is always observed while passing input parameters. For example, direct passing of the oscillator values, which, in my opinion, has nothing in common with the price of an asset. Oscillators have a well-known problem - the so-called divergence. These are the values of Open, Close, High and Low prices that, when passed directly, do not carry any meaning, but bring into the system incomprehensible noise. These values are not tied to anything and have a significant spread over time. As an example, open the daily chart of any currency pair and look at the range of Close price fluctuations.

### Theory and concepts

In this section, I will describe some of the concepts that I have classified for myself and use in my experiments.

1. **Distances** – distance of any indicator to another indicator or its zero value. In my opinion, the main principle is to link the current situation to a stable value in history. If we pass the direct value of the MA 1 indicator, we will get an incomprehensible sample spread since the price value changes over time without any certain range, which implies classification or the slightest hint of statistical behavior. It is a different matter when we convey the difference between two indicators (for example, MA 1 and MA 100 in points or the current distance on the zero candle of the MACD indicator with its value 4 candles ago). Thus, by using distances, we can determine whether the market has moved far from its average and whether it is considerably overbought or oversold compared to the history sample at the current moment. Also, using distance, we determine where the market is moving up or down. The value can be either a positive or negative.

2. **Accumulation** – calculation of the total value of the indicator or indicators relative to each other or their zero point or zero value points in some indicators other than zero. Thus, accumulation allows us to determine how long a particular movement in a certain direction takes place. If the accumulation is small, perhaps consolidation is now underway. Conversely, if the accumulation is large, there is a protracted trend. Direction depends on the sign (- or +). In other words, accumulation values do not have large values during consolidation "turbulence", while having them during trends instead.

3. **Slope angles** – current slope angle of the indicator is a very good way to inform the neural network about the current impulse movement or its absence and the attenuation of activity on the instrument. The ability to determine the slope angles of indicators for a certain history, for example, 10 candles, has proven itself well in analyzing the current situation in comparison with the results of sampling on history. I measure slope angles in radians. This is the only method that suits me personally and does not depend on the chart scale.

### Examples

_**Examples of using distances:**_

- Passing distances on the current zero candle. MA 1 indicator relative to MA 100. MACD indicator relative to its current and zero values. The CCI indicator relative to its current and zero values.

```
#property copyright   "2023, Roman Poshtar"
#property link        "https://www.mql5.com/ru/users/romanuch"
#property strict
#property version   "1.0"

input int    x1 = 1;
input int    x2 = 1;
input int    x3 = 1;

int handle_In1S1;
int handle_In2S1;
int handle_In3S1;
int handle_In4S1;

double ind_In1S1[];
double ind_In2S1[];
double ind_In3S1[];
double ind_In4S1[];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   handle_In1S1=iMA(Symbol(),PERIOD_CURRENT,1,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In1S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In2S1=iMA(Symbol(),PERIOD_CURRENT,100,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In2S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In3S1=iMACD(Symbol(),PERIOD_CURRENT,12,26,9,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In3S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In4S1=iCCI(Symbol(),PERIOD_CURRENT,14,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In4S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In1S1,true);
   if(!iGetArray(handle_In1S1,0,0,1010,ind_In1S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In2S1,true);
   if(!iGetArray(handle_In2S1,0,0,1010,ind_In2S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In3S1,true);
   if(!iGetArray(handle_In3S1,0,0,1010,ind_In3S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In4S1,true);
   if(!iGetArray(handle_In4S1,0,0,1010,ind_In4S1))
     {
      return;
     }
//---

   perceptron1();

  }

//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;

   double a1 = ((ind_In1S1[0]-ind_In2S1[0])/Point());

   double a2 = ind_In3S1[0];

   double a3 = ind_In4S1[0];

   Print("a1 = ", a1);
   Print("a2 = ", a2);
   Print("a3 = ", a3);
   Print("Perceptron = ", (w1 * a1 + w2 * a2 + w3 * a3));
   Print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

   return (w1 * a1 + w2 * a2 + w3 * a3);

  }
```

Log data:

![Example of distance 1](https://c.mql5.com/2/59/D1__1.png)

> a1 - difference in currency pair points between MA 1 and MA 100 indicator. It can take both positive and negative values.
>
> a2 - current value of the MACD indicator. It can take both positive and negative values.
>
> a3 - current value of the CCI indicator. It can take both positive and negative values.

- Passing distances N candles back. MA 1 indicator. MA 100 indicator. MACD indicator. CCI indicator.

```
#property copyright   "2023, Roman Poshtar"
#property link        "https://www.mql5.com/ru/users/romanuch"
#property strict
#property version   "1.0"

input int    Candles= 10;

input int    x1 = 1;
input int    x2 = 1;
input int    x3 = 1;
input int    x4 = 1;

int handle_In1S1;
int handle_In2S1;
int handle_In3S1;
int handle_In4S1;

double ind_In1S1[];
double ind_In2S1[];
double ind_In3S1[];
double ind_In4S1[];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   handle_In1S1=iMA(Symbol(),PERIOD_CURRENT,1,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In1S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In2S1=iMA(Symbol(),PERIOD_CURRENT,100,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In2S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In3S1=iMACD(Symbol(),PERIOD_CURRENT,12,26,9,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In3S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In4S1=iCCI(Symbol(),PERIOD_CURRENT,14,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In4S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In1S1,true);
   if(!iGetArray(handle_In1S1,0,0,1010,ind_In1S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In2S1,true);
   if(!iGetArray(handle_In2S1,0,0,1010,ind_In2S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In3S1,true);
   if(!iGetArray(handle_In3S1,0,0,1010,ind_In3S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In4S1,true);
   if(!iGetArray(handle_In4S1,0,0,1010,ind_In4S1))
     {
      return;
     }
//---

   perceptron1();

  }

//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double a1 = ((ind_In1S1[0]-ind_In1S1[Candles])/Point());

   double a2 = ((ind_In2S1[0]-ind_In2S1[Candles])/Point());

   double a3 = ind_In3S1[0]-ind_In3S1[Candles];

   double a4 = ind_In4S1[0]-ind_In4S1[Candles];

   Print("a1 = ", a1);
   Print("a2 = ", a2);
   Print("a3 = ", a3);
   Print("a3 = ", a4);
   Print("Perceptron = ", (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4));
   Print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);

  }
```

Log data:

![Example of distance 2](https://c.mql5.com/2/59/D2__1.png)

> a1 - difference in currency pair points of MA 1 indicator on candle 0 and candle 10. It can take both positive and negative values.
>
> a2 - difference in currency pair points of MA 100 indicator on candle 0 and candle 10. It can take both positive and negative values.
>
> a3 - difference in MACD indicator values on candle 0 and candle 10. It can take both positive and negative values.
>
> a4 - difference in CCI indicator values on candle 0 and candle 10. It can take both positive and negative values.

- Passing distances from the intersection point of the MA1 and MA 100 indicators relative to its zero value in the past. The MACD indicator relative to its zero value in the past. The CCI indicator relative to its zero value in the past.

```
#property copyright   "2023, Roman Poshtar"
#property link        "https://www.mql5.com/ru/users/romanuch"
#property strict
#property version   "1.0"

input int    Candles= 10;

input int    x1 = 1;
input int    x2 = 1;
input int    x3 = 1;
input int    x4 = 1;

int handle_In1S1;
int handle_In2S1;
int handle_In3S1;
int handle_In4S1;

double ind_In1S1[];
double ind_In2S1[];
double ind_In3S1[];
double ind_In4S1[];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   handle_In1S1=iMA(Symbol(),PERIOD_CURRENT,1,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In1S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In2S1=iMA(Symbol(),PERIOD_CURRENT,100,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In2S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In3S1=iMACD(Symbol(),PERIOD_CURRENT,12,26,9,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In3S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In4S1=iCCI(Symbol(),PERIOD_CURRENT,14,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In4S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In1S1,true);
   if(!iGetArray(handle_In1S1,0,0,1010,ind_In1S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In2S1,true);
   if(!iGetArray(handle_In2S1,0,0,1010,ind_In2S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In3S1,true);
   if(!iGetArray(handle_In3S1,0,0,1010,ind_In3S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In4S1,true);
   if(!iGetArray(handle_In4S1,0,0,1010,ind_In4S1))
     {
      return;
     }
//---

   perceptron1();

  }

//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {

  int c1=0;
  int c2=0;
  int c3=0;

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   for(int i=0; i<=1000; i++){

    if(ind_In1S1[0]>ind_In2S1[0]){if (ind_In1S1[i]<ind_In2S1[i]){c1=i; break;}}

    if(ind_In1S1[0]<ind_In2S1[0]){if (ind_In1S1[i]>ind_In2S1[i]){c1=i; break;}}

   }

   double a1 = ((ind_In1S1[0]-ind_In1S1[c1])/Point());

   double a2 = ((ind_In2S1[0]-ind_In2S1[c1])/Point());

   for(int i=0; i<=1000; i++){

    if(ind_In3S1[0]>0){if (ind_In3S1[i]<0){c2=i; break;}}

    if(ind_In3S1[0]<0){if (ind_In3S1[i]>0){c2=i; break;}}

   }

   double a3 = ind_In3S1[0]-ind_In3S1[c2];

   for(int i=0; i<=1000; i++){

    if(ind_In4S1[0]>0){if (ind_In4S1[i]<0){c3=i; break;}}

    if(ind_In4S1[0]<0){if (ind_In4S1[i]>0){c3=i; break;}}

   }

   double a4 = ind_In4S1[0]-ind_In4S1[c3];

   Print("a1 = ", a1);
   Print("a2 = ", a2);
   Print("a3 = ", a3);
   Print("a4 = ", a4);
   Print("Perceptron = ", (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4));
   Print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);

  }
```

Log data:

![Example of distance 3](https://c.mql5.com/2/59/D3.png)

> a1 - difference in currency pair points of MA 1 indicator on candle 0 and the candle that featured the last intersection of MA1 and MA 100 indicators. It can take both positive and negative values.
>
> a2 -  difference in currency pair points of MA 100 indicator on candle 0 and the candle that featured the last intersection of MA1 and MA 100 indicators. It can take both positive and negative values.
>
> a3 - difference in MACD indicator values on candle 0 and the candle that featured the last intersection with the value of 0. It can take both positive and negative values.
>
> a4 - difference in CCI indicator values on candle 0 and the candle that featured the last intersection with the value of 0. It can take both positive and negative values.

The search for the intersection value is first carried out in a loop.

**_Examples of using accumulation for passing to a perceptron:_**

- Passing accumulation for N indicator candles.

```
#property copyright   "2023, Roman Poshtar"
#property link        "https://www.mql5.com/ru/users/romanuch"
#property strict
#property version   "1.0"

input int    Candles= 10;

input int    x1 = 1;
input int    x2 = 1;
input int    x3 = 1;

int handle_In1S1;
int handle_In2S1;
int handle_In3S1;
int handle_In4S1;

double ind_In1S1[];
double ind_In2S1[];
double ind_In3S1[];
double ind_In4S1[];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   handle_In1S1=iMA(Symbol(),PERIOD_CURRENT,1,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In1S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In2S1=iMA(Symbol(),PERIOD_CURRENT,100,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In2S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In3S1=iMACD(Symbol(),PERIOD_CURRENT,12,26,9,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In3S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In4S1=iCCI(Symbol(),PERIOD_CURRENT,14,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In4S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In1S1,true);
   if(!iGetArray(handle_In1S1,0,0,1010,ind_In1S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In2S1,true);
   if(!iGetArray(handle_In2S1,0,0,1010,ind_In2S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In3S1,true);
   if(!iGetArray(handle_In3S1,0,0,1010,ind_In3S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In4S1,true);
   if(!iGetArray(handle_In4S1,0,0,1010,ind_In4S1))
     {
      return;
     }
//---

   perceptron1();

  }

//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double sum1 = 0;
   double sum2 = 0;
   double sum3 = 0;

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;

   for(int i=0; i<=Candles; i++)
     {

      sum1+=ind_In1S1[i]-ind_In2S1[i];

     }

   double a1 = sum1;

   for(int i=0; i<=Candles; i++)
     {

      sum2+=ind_In3S1[i];

     }

   double a2 = sum2;

   for(int i=0; i<=Candles; i++)
     {

      sum3+=ind_In4S1[i];

     }

   double a3 = sum3;

   Print("a1 = ", a1);
   Print("a2 = ", a2);
   Print("a3 = ", a3);
   Print("Perceptron = ", (w1 * a1 + w2 * a2 + w3 * a3));
   Print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

   return (w1 * a1 + w2 * a2 + w3 * a3);

  }
```

Log data:

![Accumulation example 1](https://c.mql5.com/2/59/Ac1.png)

> a1 - accumulation of the difference between the MA1 and MA 100 indicators for the number of candles of the Candles parameter. It can take both positive and negative values.
>
> a2 - accumulation of the MACD indicator for the number of candles of the Candles parameter. It can take both positive and negative values.
>
> a3 - accumulation of the CCI indicator for the number of candles of the Candles parameter. It can take both positive and negative values.

- Passing accumulation from the zero point of the indicator in the past or the last intersection of indicators in the past.

```
#property copyright   "2023, Roman Poshtar"
#property link        "https://www.mql5.com/ru/users/romanuch"
#property strict
#property version   "1.0"

input int    Candles= 10;

input int    x1 = 1;
input int    x2 = 1;
input int    x3 = 1;

int handle_In1S1;
int handle_In2S1;
int handle_In3S1;
int handle_In4S1;

double ind_In1S1[];
double ind_In2S1[];
double ind_In3S1[];
double ind_In4S1[];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   handle_In1S1=iMA(Symbol(),PERIOD_CURRENT,1,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In1S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In2S1=iMA(Symbol(),PERIOD_CURRENT,100,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In2S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In3S1=iMACD(Symbol(),PERIOD_CURRENT,12,26,9,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In3S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In4S1=iCCI(Symbol(),PERIOD_CURRENT,14,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In4S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In1S1,true);
   if(!iGetArray(handle_In1S1,0,0,1010,ind_In1S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In2S1,true);
   if(!iGetArray(handle_In2S1,0,0,1010,ind_In2S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In3S1,true);
   if(!iGetArray(handle_In3S1,0,0,1010,ind_In3S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In4S1,true);
   if(!iGetArray(handle_In4S1,0,0,1010,ind_In4S1))
     {
      return;
     }
//---

   perceptron1();

  }

//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double sum1 = 0;
   double sum2 = 0;
   double sum3 = 0;

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;

   for(int i=0; i<=1000; i++)
     {

      if(ind_In1S1[0]>ind_In2S1[0])
        {
         if(ind_In1S1[i]<ind_In2S1[i])
           {
            break;
           };
         sum1+=(ind_In1S1[i]-ind_In2S1[i]);
        }

      if(ind_In1S1[0]<ind_In2S1[0])
        {
         if(ind_In1S1[i]>ind_In2S1[i])
           {
            break;
           };
         sum1+=(ind_In1S1[i]-ind_In2S1[i]);
        }

     }

   double a1 = sum1;

   for(int i=0; i<=1000; i++)
     {

      if(ind_In3S1[0]>0)
        {
         if(ind_In3S1[i]<0)
           {
            break;
           };
         sum2+=ind_In3S1[i];
        }

      if(ind_In3S1[0]<0)
        {
         if(ind_In3S1[i]>0)
           {
            break;
           };
         sum2+=ind_In3S1[i];
        }

     }

   double a2 = sum2;

   for(int i=0; i<=1000; i++)
     {

      if(ind_In4S1[0]>0)
        {
         if(ind_In4S1[i]<0)
           {
            break;
           };
         sum3+=ind_In4S1[i];
        }

      if(ind_In4S1[0]<0)
        {
         if(ind_In4S1[i]>0)
           {
            break;
           };
         sum3+=ind_In4S1[i];
        }

     }

   double a3 = sum3;

   Print("a1 = ", a1);
   Print("a2 = ", a2);
   Print("a3 = ", a3);
   Print("Perceptron = ", (w1 * a1 + w2 * a2 + w3 * a3));
   Print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

   return (w1 * a1 + w2 * a2 + w3 * a3);

  }
```

Log data:

![Accumulation example 2](https://c.mql5.com/2/59/Ac2__1.png)

> a1 - accumulation of the difference between the MA1 and MA 100 indicators from the last intersection point. It can take both positive and negative values.
>
> a2 - accumulation of the MACD indicator from the point of the last intersection of the value 0. It can take both positive and negative values.
>
> a3 - accumulation of the CCI indicator from the point of the last intersection of the value 0. It can take both positive and negative values.

**_Examples of using indicator slope angles:_**

- Passing slope angles for N indicator candles.

```
#property copyright   "2023, Roman Poshtar"
#property link        "https://www.mql5.com/ru/users/romanuch"
#property strict
#property version   "1.0"

input int    Candles= 10;

input int    x1 = 1;
input int    x2 = 1;
input int    x3 = 1;
input int    x4 = 1;

int handle_In1S1;
int handle_In2S1;
int handle_In3S1;
int handle_In4S1;

double ind_In1S1[];
double ind_In2S1[];
double ind_In3S1[];
double ind_In4S1[];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   handle_In1S1=iMA(Symbol(),PERIOD_CURRENT,1,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In1S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In2S1=iMA(Symbol(),PERIOD_CURRENT,100,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In2S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In3S1=iMACD(Symbol(),PERIOD_CURRENT,12,26,9,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In3S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In4S1=iCCI(Symbol(),PERIOD_CURRENT,14,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In4S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In1S1,true);
   if(!iGetArray(handle_In1S1,0,0,1010,ind_In1S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In2S1,true);
   if(!iGetArray(handle_In2S1,0,0,1010,ind_In2S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In3S1,true);
   if(!iGetArray(handle_In3S1,0,0,1010,ind_In3S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In4S1,true);
   if(!iGetArray(handle_In4S1,0,0,1010,ind_In4S1))
     {
      return;
     }
//---

   perceptron1();

  }

//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double a1 = (((ind_In1S1[0]-ind_In1S1[Candles])/Point())/Candles);

   double a2 = (((ind_In2S1[0]-ind_In2S1[Candles])/Point())/Candles);

   double a3 = ((ind_In3S1[0]-ind_In3S1[Candles])/Candles);

   double a4 = ((ind_In4S1[0]-ind_In4S1[Candles])/Candles);

   Print("a1 = ", a1);
   Print("a2 = ", a2);
   Print("a3 = ", a3);
   Print("a4 = ", a4);
   Print("Perceptron = ", (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4));
   Print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);

  }
```

Log data:

![Example of angles 1](https://c.mql5.com/2/59/A1.png)

> a1 - MA1 indicator slope angle for the number of candles of the Candles parameter. It can take both positive and negative values.
>
> a2 - MA100 indicator slope angle for the number of candles of the Candles parameter. It can take both positive and negative values.
>
> a3 - MACD indicator slope angle for the number of candles of the Candles parameter. It can take both positive and negative values.
>
> a4 - CCI indicator slope angle for the number of candles of the Candles parameter. It can take both positive and negative values.

- Passing slope angles from the zero point of the indicator in the past or the last intersection of indicators in the past.

```
#property copyright   "2023, Roman Poshtar"
#property link        "https://www.mql5.com/ru/users/romanuch"
#property strict
#property version   "1.0"

input int    x1 = 1;
input int    x2 = 1;
input int    x3 = 1;
input int    x4 = 1;

int handle_In1S1;
int handle_In2S1;
int handle_In3S1;
int handle_In4S1;

double ind_In1S1[];
double ind_In2S1[];
double ind_In3S1[];
double ind_In4S1[];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   handle_In1S1=iMA(Symbol(),PERIOD_CURRENT,1,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In1S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In2S1=iMA(Symbol(),PERIOD_CURRENT,100,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In2S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In3S1=iMACD(Symbol(),PERIOD_CURRENT,12,26,9,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In3S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In4S1=iCCI(Symbol(),PERIOD_CURRENT,14,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In4S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In1S1,true);
   if(!iGetArray(handle_In1S1,0,0,1010,ind_In1S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In2S1,true);
   if(!iGetArray(handle_In2S1,0,0,1010,ind_In2S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In3S1,true);
   if(!iGetArray(handle_In3S1,0,0,1010,ind_In3S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In4S1,true);
   if(!iGetArray(handle_In4S1,0,0,1010,ind_In4S1))
     {
      return;
     }
//---

   perceptron1();

  }

//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {

   int c1=0;
   int c2=0;
   int c3=0;

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   for(int i=0; i<=1000; i++)
     {

      if(ind_In1S1[0]>ind_In2S1[0])
        {
         if(ind_In1S1[i]<ind_In2S1[i])
           {
            c1=i;
            break;
           }
        }

      if(ind_In1S1[0]<ind_In2S1[0])
        {
         if(ind_In1S1[i]>ind_In2S1[i])
           {
            c1=i;
            break;
           }
        }

     }

   double a1 = (((ind_In1S1[0]-ind_In1S1[c1])/Point())/c1);

   double a2 = (((ind_In2S1[0]-ind_In2S1[c1])/Point())/c1);

   for(int i=0; i<=1000; i++)
     {

      if(ind_In3S1[0]>0)
        {
         if(ind_In3S1[i]<0)
           {
            c2=i;
            break;
           }
        }

      if(ind_In3S1[0]<0)
        {
         if(ind_In3S1[i]>0)
           {
            c2=i;
            break;
           }
        }

     }

   double a3 = ((ind_In3S1[0]-ind_In3S1[c2])/c2);

   for(int i=0; i<=1000; i++)
     {

      if(ind_In4S1[0]>0)
        {
         if(ind_In4S1[i]<0)
           {
            c3=i;
            break;
           }
        }

      if(ind_In4S1[0]<0)
        {
         if(ind_In4S1[i]>0)
           {
            c3=i;
            break;
           }
        }

     }

   double a4 = ((ind_In4S1[0]-ind_In4S1[c3])/c3);

   Print("a1 = ", a1);
   Print("a2 = ", a2);
   Print("a3 = ", a3);
   Print("a4 = ", a4);
   Print("Perceptron = ", (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4));
   Print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);

  }
```

Log data:

![Example of angles 2](https://c.mql5.com/2/59/A2.png)

> a1 - MA1 indicator slope angle from the point of the last intersection of the MA1 and MA100 indicators. It can take both positive and negative values.
>
> a2 - MA100 indicator slope angle from the point of the last intersection of the MA1 and MA100 indicators. It can take both positive and negative values.
>
> a3 - MACD indicator slope angle from the point of the last intersection of the value of 0 by the indicator. It can take both positive and negative values.
>
> a4 - CCI indicator slope angle from the point of the last intersection of the value of 0 by the indicator. It can take both positive and negative values.

**_Example of using combined values for passing to the perceptron:_**

- Combined passing of the values of MA1 and MA100 indicators.

```
#property copyright   "2023, Roman Poshtar"
#property link        "https://www.mql5.com/ru/users/romanuch"
#property strict
#property version   "1.0"

input int    Candles= 10;

input int    x1 = 1;
input int    x2 = 1;
input int    x3 = 1;
input int    x4 = 1;
input int    x5 = 1;
input int    x6 = 1;
input int    x7 = 1;

int handle_In1S1;
int handle_In2S1;

double ind_In1S1[];
double ind_In2S1[];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   handle_In1S1=iMA(Symbol(),PERIOD_CURRENT,1,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In1S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   handle_In2S1=iMA(Symbol(),PERIOD_CURRENT,100,0,MODE_SMA,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_In2S1==INVALID_HANDLE)
     {
      return(INIT_FAILED);
     }
//---

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In1S1,true);
   if(!iGetArray(handle_In1S1,0,0,1010,ind_In1S1))
     {
      return;
     }
//---

//--- get data from the three buffers of the i-Regr indicator
   ArraySetAsSeries(ind_In2S1,true);
   if(!iGetArray(handle_In2S1,0,0,1010,ind_In2S1))
     {
      return;
     }
//---

   perceptron1();

  }

//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {

   int c1=0;
   double sum1 = 0;
   double sum2 = 0;

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;
   double w5 = x5 - 10.0;
   double w6 = x6 - 10.0;
   double w7 = x7 - 10.0;

   double a1 = ((ind_In1S1[0]-ind_In2S1[0])/Point());

   for(int i=0; i<=Candles; i++)
     {

      sum1+=ind_In1S1[i]-ind_In2S1[i];

     }

   double a2 = sum1;

   for(int i=0; i<=1000; i++)
     {

      if(ind_In1S1[0]>ind_In2S1[0])
        {
         if(ind_In1S1[i]<ind_In2S1[i])
           {
            break;
           };
         sum2+=(ind_In1S1[i]-ind_In2S1[i]);
        }

      if(ind_In1S1[0]<ind_In2S1[0])
        {
         if(ind_In1S1[i]>ind_In2S1[i])
           {
            break;
           };
         sum2+=(ind_In1S1[i]-ind_In2S1[i]);
        }

     }

   double a3 = sum2;

   double a4 = (((ind_In1S1[0]-ind_In1S1[Candles])/Point())/Candles);

   double a5 = (((ind_In2S1[0]-ind_In2S1[Candles])/Point())/Candles);

   for(int i=0; i<=1000; i++)
     {

      if(ind_In1S1[0]>ind_In2S1[0])
        {
         if(ind_In1S1[i]<ind_In2S1[i])
           {
            c1=i;
            break;
           }
        }

      if(ind_In1S1[0]<ind_In2S1[0])
        {
         if(ind_In1S1[i]>ind_In2S1[i])
           {
            c1=i;
            break;
           }
        }

     }

   double a6 = (((ind_In1S1[0]-ind_In1S1[c1])/Point())/c1);

   double a7 = (((ind_In2S1[0]-ind_In2S1[c1])/Point())/c1);

   Print("a1 = ", a1);
   Print("a2 = ", a2);
   Print("a3 = ", a3);
   Print("a4 = ", a4);
   Print("a5 = ", a5);
   Print("a6 = ", a6);
   Print("a7 = ", a7);
   Print("Perceptron = ", (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4 + w5 * a5 + w6 * a6 + w7 * a7));
   Print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4 + w5 * a5 + w6 * a6 + w7 * a7);

  }
```

Log data:

![Combined passing example 1](https://c.mql5.com/2/59/C1.png)

> a1 - distance between MA1 and MA100 on the current candle. It can take both positive and negative values.
>
> a2 - accumulation between indicators MA1 and MA100 for the number of candles of the Candles parameter. It can take both positive and negative values.
>
> a3 - accumulation between indicators MA1 and MA100 from the point of the last intersection of the indicators. It can take both positive and negative values.
>
> a4 - MA1 indicator slope angle for the number of candles of the Candles parameter. It can take both positive and negative values.
>
> a5 - MA100 indicator slope angle for the number of candles of the Candles parameter. It can take both positive and negative values.
>
> a6 - MA1 indicator slope angle from the point of the last intersection of the indicators. It can take both positive and negative values.
>
> a7 - MA100 indicator slope angle from the point of the last intersection of the indicators. It can take both positive and negative values.

### Expert Advisors

Here we will develop two EAs. As an example, we will use the values of the distances between indicators. The first one is implemented for optimization and selection of parameters. The second one is meant for trading using the results obtained from the first one. I have already considered this approach in my previous articles. This method helps sorting through and using all available optimization results in trading. For example, we got 50 good results during optimization. As you might imagine, installing an EA on 50 charts is not very convenient. Therefore, we will use all the results simultaneously with the help of the second EA.

Find out more about the method in my article ["Experiments with neural networks (Part 3): Practical application"](https://www.mql5.com/en/articles/11949)

I will provide the main code of the first EA for optimization below and describe what we passed to the perceptron:

The main code for opening positions.

```
      //SELL++++++++++++++++++++++++++++++++++++++++++++++++
      if((perceptron1()<-Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (SpreadS1<=MaxSpread))
        {
         OpenSell(symbolS1.Name(), LotsXSell, TP, SL, EAComment);
        }

      //BUY++++++++++++++++++++++++++++++++++++++++++++++++
      if((perceptron1()>Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (SpreadS1<=MaxSpread))
        {
         OpenBuy(symbolS1.Name(), LotsXBuy, TP, SL, EAComment);
        }
```

Perceptron code.

1. a1 is the distance between the MA1 indicator on the current candle and the candle in the Candles parameter.
2. a2 is the distance between the MA100 indicator on the current candle and the candle in the Candles parameter.
3. a3 is the distance between the CCI indicator on the current candle and the candle in the Candles parameter.
4. a4 is the distance between the StdDev indicator on the current candle and the candle in the Candles parameter.

The Candles parameter in this case was not optimized and had a value of 8.

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {

   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double a1 = ((ind_In1S1[0]-ind_In1S1[Candles])/PointS1);

   double a2 = ((ind_In2S1[0]-ind_In2S1[Candles])/PointS1);

   double a3 = (ind_In3S1[0]-ind_In3S1[Candles]);

   double a4 = (ind_In4S1[0]-ind_In4S1[Candles]);

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

Here I will provide all the parameters for optimization and testing so as not to repeat them further in the text:

- Forex market;
- EURUSD;
- Timeframe: H1;
- Indicators: MA 1 SMA CLOSE, MA 200 SMA CLOSE, CCI 42, StdDev 60.
- Stop Loss and Take Profit, 400 and 830;
- Optimization and test mode "1 Minute OHLC" and "Maximum Profit". Our EA initially works at M1 open prices. Here we will use the "Maximum Profit" mode as an experiment;
- Optimization range 1 year. From 2021.10.10 to 2022.10.10. 1 year is not some kind of criterion. You can try more or less on your own;
- Forward testing range is 1 year. From 2022.10.10 to 2023.10.10;
- In forward testing, the first 50 best optimization results were used simultaneously;
- Initial deposit 10000;
- Leverage 1:500.

Optimization settings are provided in the screenshot below:

![Optimization settings](https://c.mql5.com/2/59/Opt_1.png)

The optimization results are shown in the screenshots below:

![Optimization result 1](https://c.mql5.com/2/59/Opt_2.png)

![Optimization result 2](https://c.mql5.com/2/59/Opt_3.png)

After optimization is completed, generate a file in CSV format using Excel. Let me remind you that we take the first 50 best results. Paste the resulting file into the code of the second EA for testing. It will look as follows in the code.

```
string EURUSD[][8]=
  {
   {"Profit","Trades","x1","x2","x3","x4","Param"},
   {"266.45","239","2","1","9","8","5000"},
   {"266.45","239","2","1","9","13","5000"},
   {"266.45","239","2","1","9","11","5000"},
   {"266.45","239","2","1","9","10","5000"},
   {"266.45","239","2","1","9","8","5000"},
   {"266.45","239","2","1","9","12","5000"},
   {"266.45","239","2","1","9","20","5000"},
   {"266.45","239","2","1","9","14","5000"},
   {"266.45","239","2","1","9","2","5000"},
   {"266.45","239","2","1","9","3","5000"},
   {"259.69","239","0","0","12","17","5500"},
   {"259.69","239","0","0","12","8","5500"},
   {"259.69","239","0","0","12","1","5500"},
   {"259.69","239","0","0","12","9","5500"},
   {"259.69","239","0","0","12","16","5500"},
   {"259.69","239","0","0","12","18","5500"},
   {"259.69","239","0","0","12","11","5500"},
   {"259.69","239","0","0","12","7","5500"},
   {"259.69","239","0","0","12","15","5500"},
   {"259.69","239","0","0","12","8","5500"},
   {"259.69","239","0","0","12","9","5500"},
   {"259.69","239","0","0","12","17","5500"},
   {"259.69","239","0","0","12","6","5500"},
   {"259.69","239","0","0","12","15","5500"},
   {"259.69","239","0","0","12","4","5500"},
   {"259.69","239","0","0","12","7","5500"},
   {"259.69","239","0","0","12","11","5500"},
   {"259.69","239","0","0","12","14","5500"},
   {"259.69","239","0","0","12","1","5500"},
   {"259.69","239","0","0","12","5","5500"},
   {"259.69","239","0","0","12","3","5500"},
   {"259.69","239","0","0","12","0","5500"},
   {"259.69","239","0","0","12","12","5500"},
   {"259.69","239","0","0","12","16","5500"},
   {"259.69","239","0","0","12","8","5500"},
   {"259.69","239","0","0","12","10","5500"},
   {"259.69","239","0","0","12","16","5500"},
   {"259.69","239","0","0","12","18","5500"},
   {"259.69","239","0","0","12","13","5500"},
   {"259.69","239","0","0","12","9","5500"},
   {"259.69","239","0","0","12","12","5500"},
   {"259.69","239","0","0","12","11","5500"},
   {"259.69","239","0","0","12","7","5500"},
   {"259.69","239","0","0","12","15","5500"},
   {"259.69","239","0","0","12","14","5500"},
   {"259.69","239","0","0","12","3","5500"},
   {"259.69","239","0","0","12","19","5500"},
   {"259.69","239","0","0","12","0","5500"},
   {"259.69","239","0","0","12","17","5500"},
   {"259.69","239","0","0","12","2","5500"}
  };
```

I will also provide the code for the second trading EA. As you can see, we go through all the values obtained during optimization in the loop and compare them with the current results of the perceptron execution. If the values match, a corresponding buy or sell position is opened. The limit on the number of simultaneously open positions is controlled by the MaxSeries parameter and the CalculateSeries function, which calculates positions by comments.

```
      for(int i=1; i<=(ArraySize(EURUSD)/8)-1; i++)
        {

         comm=(EURUSD[i][0]+EURUSD[i][1]);

         x1=(int)StringToInteger(EURUSD[i][2]);
         x2=(int)StringToInteger(EURUSD[i][3]);
         x3=(int)StringToInteger(EURUSD[i][4]);
         x4=(int)StringToInteger(EURUSD[i][5]);

         Param=(int)StringToInteger(EURUSD[i][6]);

         //SELL++++++++++++++++++++++++++++++++++++++++++++++++
         if((NewOpen==true) && (CalculateSeries(Magic)<MaxSeries) && (perceptron1()<-Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment+" En_"+comm)==0) && (SpreadS1<=MaxSpread))
           {
            OpenSell(symbolS1.Name(), LotsXSell, TP, SL, EAComment+" En_"+comm);
           }

         //BUY++++++++++++++++++++++++++++++++++++++++++++++++
         if((NewOpen==true) && (CalculateSeries(Magic)<MaxSeries) && (perceptron1()>Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment+" En_"+comm)==0) && (SpreadS1<=MaxSpread))
           {
            OpenBuy(symbolS1.Name(), LotsXBuy, TP, SL, EAComment+" En_"+comm);
           }

        }
```

```
//+------------------------------------------------------------------+
//| Calculate Positions                                              |
//+------------------------------------------------------------------+
int CalculateSeries(ulong mag)
  {
   int total=0;
   string com="";

   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      if(position.SelectByIndex(i))
        {
         if(position.Magic()==mag)
           {
            if(com!=position.Comment())
              {
               com=position.Comment();
               total++;
              }
           }
        }
     }
//---
   return(total);
  }
```

Backward test result:

![Backward test result](https://c.mql5.com/2/59/B.png)

Forward test result:

![Forward test result](https://c.mql5.com/2/59/F.png)

![Forward test result 2](https://c.mql5.com/2/59/EURUSDH1.png)

### Conclusion

List of files in the attachment:

1. **Distance 1**, **Distance 2**, **Distance 3** \- EAs featuring examples of passing distances to the perceptron;
2. **Accumulation 1**, **Accumulation 2** \- EAs featuring examples of passing the accumulation into the perceptron;
3. **Angle 1**, **Angle 2** \- EAs featuring examples of passing indicator slope angles to the perceptron;
4. **Combo 1** \- EA with an example of passing combined indicator data to the perceptron;
5. **Perceptron – opt** \- optimization EA;
6. **Perceptron – trade** – EA for trading optimized parameters.

As can be seen from the forward test results, our approach to passing indicators works well. For the first six months, the EA pushes the balance up quite confidently. Above, I provided all sorts of examples of passing indicators. You might be interested in testing them. A lot of work has been done this time, but there is always something to strive for.

If you have any questions, feel free to contact me on the forum or in private messages. I will always be happy to help you.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/13598](https://www.mql5.com/ru/articles/13598)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13598.zip "Download all attachments in the single ZIP archive")

[EA.zip](https://www.mql5.com/en/articles/download/13598/ea.zip "Download EA.zip")(23.32 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Experiments with neural networks (Part 6): Perceptron as a self-sufficient tool for price forecast](https://www.mql5.com/en/articles/12515)
- [Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://www.mql5.com/en/articles/12459)
- [Experiments with neural networks (Part 4): Templates](https://www.mql5.com/en/articles/12202)
- [Testing and optimization of binary options strategies in MetaTrader 5](https://www.mql5.com/en/articles/12103)
- [Experiments with neural networks (Part 3): Practical application](https://www.mql5.com/en/articles/11949)
- [Experiments with neural networks (Part 2): Smart neural network optimization](https://www.mql5.com/en/articles/11186)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/463178)**
(5)


![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
24 Oct 2023 at 16:23

New term in trading : **Canbles**:-)

Correct the typos Candles! from the word Candle, through **d**

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
28 Oct 2023 at 14:00

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/456252#comment_50122214):**

New term in trading : **Canbles**:-)

Correct the typos Candles! from the word Candle, through **d**

Corrected. Thanks for the observation )

![Grigori.S.B](https://c.mql5.com/avatar/2020/1/5E24C71E-B11B.png)

**[Grigori.S.B](https://www.mql5.com/en/users/grigori.s.b)**
\|
29 Oct 2023 at 07:42

**Roman Poshtar [#](https://www.mql5.com/ru/forum/456252#comment_50196515):**

Corrected.

Didn't fix a thing. It's still Canbles.

![](https://c.mql5.com/3/421/2023-10-29_09-39-42.png)

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
30 Oct 2023 at 17:19

Definitely got all the places in the article corrected now.

Thanks for the message.

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
30 Oct 2023 at 19:06

**Rashid Umarov [#](https://www.mql5.com/ru/forum/456252#comment_50228577):**

Now definitely all the places in the article have been corrected.

Thanks for the message.

Thank you.

![Working with ONNX models in float16 and float8 formats](https://c.mql5.com/2/71/onnx-float-avatar.png)[Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)

Data formats used to represent machine learning models play a crucial role in their effectiveness. In recent years, several new types of data have emerged, specifically designed for working with deep learning models. In this article, we will focus on two new data formats that have become widely adopted in modern models.

![Neural networks made easy (Part 60): Online Decision Transformer (ODT)](https://c.mql5.com/2/59/Online_Decision_Transformer_logo_up.png)[Neural networks made easy (Part 60): Online Decision Transformer (ODT)](https://www.mql5.com/en/articles/13596)

The last two articles were devoted to the Decision Transformer method, which models action sequences in the context of an autoregressive model of desired rewards. In this article, we will look at another optimization algorithm for this method.

![Quantization in machine learning (Part 1): Theory, sample code, analysis of implementation in CatBoost](https://c.mql5.com/2/59/Quantization_in_machine_learning_logo.png)[Quantization in machine learning (Part 1): Theory, sample code, analysis of implementation in CatBoost](https://www.mql5.com/en/articles/13219)

The article considers the theoretical application of quantization in the construction of tree models and showcases the implemented quantization methods in CatBoost. No complex mathematical equations are used.

![Developing a Replay System (Part 28): Expert Advisor project — C_Mouse class (II)](https://c.mql5.com/2/58/Replay-p28_II_avatar.png)[Developing a Replay System (Part 28): Expert Advisor project — C\_Mouse class (II)](https://www.mql5.com/en/articles/11349)

When people started creating the first systems capable of computing, everything required the participation of engineers, who had to know the project very well. We are talking about the dawn of computer technology, a time when there were not even terminals for programming. As it developed and more people got interested in being able to create something, new ideas and ways of programming emerged which replaced the previous-style changing of connector positions. This is when the first terminals appeared.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/13598&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062660282095937153)

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