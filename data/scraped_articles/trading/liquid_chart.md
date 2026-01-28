---
title: Liquid Chart
url: https://www.mql5.com/en/articles/1208
categories: Trading, Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:38:35.107988
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/1208&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083009046870496040)

MetaTrader 5 / Trading


### Introduction

Once I noticed that charts with the H4 timeframe and higher looked different at every broker's. The reason behind it was that the brokers were located in different time zones. In some cases certain parts of the same charts were significantly different in spite of a small difference between time zones. On one chart there was a distinct reversal pattern and the same part of the other one did not represent any precise pattern.

Then it crossed my mind to write an indicator which would redraw the H1 chart so that there is always a complete closing bar on the right. The M1 period was chosen as a source of prices. As a result, an hourly chart was redrawn every minute and in an hour I had 60 varieties of the same hourly chart. Its form was changing in a smooth and flowing manner revealing hidden patterns that the initial pattern did not even have a hint on.

I called this indicator "liquid chart" for its specific appearance. Depending on the plotting mode, the chart "flows" (gets redrawn) either when a new bar of the basic period appears or when the value of static shift gets changed. In this article we shall consider the principles of plotting a "liquid chart", then write an indicator and compare efficiency of using this technology for Experts trading by indicators and Experts trading by patterns.

Liquid Chart, DSC mode - YouTube

[Photo image of decanium](https://www.youtube.com/channel/UCFL354HHsDZqd1A0VoQ6xNg?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1208)

decanium

36 subscribers

[Liquid Chart, DSC mode](https://www.youtube.com/watch?v=g75kUgbM7go)

decanium

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=g75kUgbM7go&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1208)

0:00

0:00 / 6:20

•Live

•

### 1\. Plotting Principle

Before we start, we are going to define the terms.

**_Shift_** is a difference between opening prices of the resulting chart bars and the opening prices of the source chart bars.

**_Current timeframe_** is the timeframe of the source chart.

**_Basic timeframe_** is a timeframe with prices we are going to use for forming bars of the resulting chart.

Basic period cannot exceed the current one. Current period must be divided by the basic period without remainder. The greater the ratio of the current timeframe to the basic one, the more different variations of the resulting chart we can get. However, if the ratio is too large, then the historical data of the basic timeframe may not be sufficient for drawing necessary number of the resulting chart bars.

There are three types of plotting a chart.

- Chart with a static shift ( _Static Shift_ or **SS**).
- Chart with a dynamic shift in the opening mode ( _Dynamic Shift, just Open_ or **DSO**).
- Chart with a dynamic shift in the closing mode ( _Dynamic Shift, expected Close_ or **DSC**).

In the static shift mode, opening times of the bars are shifted by the set time. Dynamic shift in the opening mode makes it look like a bar has just been opened and in the closing mode as if the bar will be closed soon.

Let us have a closer look.

**1.1. Chart with Static Shift**

In this mode the opening time of every bar is shifted by the number of minutes equivalent to the set number of basic timeframes. We shall call it a **shift**. This way, if the set shift is equal to 0, then the chart is exactly the same as the source one. Shift 1, providing that the basic timeframe is 15 minutes, is equal to 15 minutes. Shift 2 is equal to 30 minutes and so on.

The shift cannot exceed (k-1), where k is a ratio of the current timeframe to the basic one. It means that with the current timeframe H1 and the basic one M1, maximum permissible shift is 60/1 - 1 = 59 of basic timeframes, that is 59 minutes. If the basic timeframe is M5, then maximum permissible shift is 60/5 - 1 = 11 of basic timeframes, that is 55 minutes.

Opening time of the bars for the current timeframe H1 and the shift of 15 minutes, is 00:15, 01:15, 02:15 and so on. For the current timeframe M15 and the shift of 1 minute, the opening time of the bars is 00:16, 00:31, 00:46, 01:01 and so on.

When shifts are close to the limit values, such a chart is rarely different from the source one. Significant differences appear when a shift value is close to the middle of the permissible range.

![Chart with static shift](https://c.mql5.com/2/12/EURUSDM15_H1_SS1__1.png)

Fig. 1. Example of hourly bars formation on the basic timeframe of M15 with the shift value equal to 1

**1.2. Chart with a Dynamic Shift in the Opening Mode**

In this mode the shift is recalculated every time a new bar of the basic timeframe appears. At the same time, the shift is calculated so that the time of the bar existence in the end of the chart (the latest prices) does not exceed the basic timeframe value. If the current timeframe is H1 and the basic one is M5, it will look like as if the the far right bar was opened not earlier than five minutes ago.

![Chart with dynamic shift, beginning of the bar](https://c.mql5.com/2/12/EURUSDM15_H1_DSS__1.png)

Fig. 2. Example of hourly bars formation on the basic timeframe of M15 with dynamic shift in the opening mode

**1.3. Chart with a Dynamic Shift in the Closing Mode**

In this mode, the shift is recalculated every time a new bar of the basic timeframe appears, like in the opening mode. The only difference is that the shift is calculated the way that the existence time of the bar in the end of the chart (the latest prices) was greater or equal to the difference between the current and basic timeframes. On the current timeframe of H1 and the basic one of M5, it looks like the far right bar will close no later than in five minutes.

![Chart with dynamic shift, completion of the bar](https://c.mql5.com/2/12/EURUSDM15_H1_DSE__1.png)

Fig. 3. Example of hourly bars formation on the basic timeframe of M15 and dynamic shift in the closing mode

### 2\. Data Transformation

The **GetRatesLC()** function was written to convert historical data taking into account the set shift. It writes modified historical data to the array of structures of the [MqlRates](https://www.mql5.com/en/docs/constants/structures/mqlrates) type, similar to the [CopyRates()](https://www.mql5.com/en/docs/series/copyrates) function.

```
int GetRatesLC(
   int             start_pos    // source for copying
   int             len,         // amount to copy
   MqlRates&       rates[],     // target array
   ENUM_TIMEFRAMES base_period, // basic timeframe
   int&            shift        // shift
   );
```

**Parameters**

_start\_pos_

\[in\]  Index of the first element in the current timeframe. It is the start point of the data conversion and copying to a buffer.

_len_

\[in\]  Number of copied elements.

_rates\[\]_

\[out\]  Array of the MqlRates type.

_base\_period_

\[in\]  Basic timeframe.

_shift_

\[in\] \[out\]  Shift. It can accept the following values:

| Value | Description |
| --- | --- |
| -2 | Calculate the shift in the opening mode (beginning of the bar formation) |
| -1 | Calculate the shift in the closing mode (end of the bar formation) |
| 0 ... N | Apply the set shift. Can accept the values from 0 to N.<br> N = Tcur/Tbase - 1. Where Tcur is the current timeframe, Tbase is a basic one. |

Table 1. Permissible values of the _shift_ parameter

If the function is implemented successfully, _shift_ will receive the value of the calculated shift if the values of -2 or -1 were passed.

**Returned value**

Number of copied elements or the error code:

| Code | Description |
| --- | --- |
| -1 | Basic timeframe specified incorrectly |
| -2 | Shift specified incorrectly |

Table 2. Returned error codes

Below is the code of the **GetRatesLC()** function from the **liquidchart.mqh** file.

```
int GetRatesLC(int start_pos,int len,MqlRates &rates[],ENUM_TIMEFRAMES base_period,int& shift)
  {
   //--- number of basic timeframes contained in the current one
   int k=PeriodSeconds()/PeriodSeconds(base_period);
   if(k==0)
      return(-1);//basic timeframe specified incorrectly
   //---
   MqlRates r0[];
   ArrayResize(rates,len);
   if(CopyRates(_Symbol,_Period,start_pos,1,r0)<1)
      return(0);// no data
   //---
   int sh;
   if(shift>=0)
     {
      //--- fixed shift
      if(shift<k)
         sh=shift;
      else
         return(-2);//--- shift specified incorrectly
     }
   else if(shift==-1)
     {
      //--- shift to be calculated (dynamic, beginning of the bar formation)
      sh=int((TimeCurrent()-r0[0].time)/PeriodSeconds(base_period));
     }
   else if(shift==-2)
     {
      //--- shift to be calculated (dynamic, end of the bar formation)
      sh=1+int((TimeCurrent()-r0[0].time)/PeriodSeconds(base_period));
      if(sh>=k)
         sh = 0;
     }
   else
      return(-2);//shift specified incorrectly
   //--- opening time of the basic period bar, which is the beginning of the current period bar formation
   //--- synchronization of the time of opening bars takes place relative to the tO time
   datetime tO;
   //--- closing time of the bar under formation, i.e. opening time of the last bar of basic timeframe in the series
   datetime tC;
   tO=r0[0].time+sh*PeriodSeconds(base_period);
   if(tO>TimeCurrent())
      tO-=PeriodSeconds();
   tC=tO+PeriodSeconds()-PeriodSeconds(base_period);
   if(tC>TimeCurrent())
      tC=TimeCurrent();
   int cnt=0;
   while(cnt<len)
     {
      ArrayFree(r0);
      int l=CopyRates(_Symbol,base_period,tC,k,r0);
      if(l<1)
         break;
      //--- time of the bar with the (l-1) index does not have to be equal to tC
      //--- if there is no bar with the tC time, it can be the nearest bar
      //--- in any case its time is assigned to the tC time
      tC=r0[l-1].time;
      //--- check if tO has the correct value and modify if needed.
      while(tO>tC)
         tO-=PeriodSeconds();
      //--- the time values of tO and tC have actual meaning for the bar under formation
      int index=len-1-cnt;
      rates[index].close=0;
      rates[index].open=0;
      rates[index].high=0;
      rates[index].low=0;
      rates[index].time=tO;
      for(int i=0; i<l; i++)
         if(r0[i].time>=tO && r0[i].time<=tC)
           {
            if(rates[index].open==0)
              {
               rates[index].open= r0[i].open;
               rates[index].low = r0[i].low;
               rates[index].high= r0[i].high;
                 }else{
               if(rates[index].low > r0[i].low)
                  rates[index].low=r0[i].low;
               if(rates[index].high < r0[i].high)
                  rates[index].high=r0[i].high;
              }
            rates[index].close=r0[i].close;
           }
      //--- specifying closing time of the next bar in the loop
      tC=tO-PeriodSeconds(base_period);
      //
      cnt++;
     }
   if(cnt<len)
     {
      //-- less data than required, move to the beginning of the buffer
      int d=len-cnt;
      for(int j=0; j<cnt; j++)
         rates[j]=rates[j+d];
      for(int j=cnt;j<len;j++)
        {
         //--- fill unused array elements with zeros
         rates[j].close=0;
         rates[j].open=0;
         rates[j].high=0;
         rates[j].low=0;
         rates[j].time=0;
        }
     }
   shift = sh;
   return(cnt);
  }
```

A few important points should be highlighted.

- The function does not return tick volumes. The reason for this is that in the DSC mode the function never returns the volume equal to one as it happens at the bar opening. It is pretty logical. If your Expert Advisor uses the tick volume equal to one as a signal of a new bar formation, it will never receive that signal. This method is used in the **Moving Average** Expert Advisor. You could add a counting of the tick volumes to the function but it would not be working correctly. To avoid confusion I do not measure tick volumes at all.
- The function returns requested number of bars but it does not mean that the time between the first and the last bar will be commensurate with the correspondent time interval on the source chart. On a continuous segment of historical data though there will be correspondence. If the specified segment contains weekends, "phantom bars" may appear at the borderline.

The figure below represents an example of a "phantom bar". This bar was formed for the first minute of October, the 27th, which got included into the bar together with the opening time of 23:01 on the 26th of October. It should be noted that after such bars the chart of the indicator will be shifted to the left in relation to the source chart. The bars with the time correspondent to the initial time (for example, 21:00 -> 21:01), will have different indexes.

![Phantom bar](https://c.mql5.com/2/12/USDJPYH1_LC_020.png)

Fig. 4. Phantom bar 2014.10.26 at 23:01

### 3\. Indicator Implementation

Let us write an indicator displaying a "liquid chart" in a separate window. The indicator should work in all three modes: the static shift mode, dynamic shift in the bar opening mode and dynamic shift in the bar closing mode. The indicator also has to have control elements for changing modes and the shift value without a necessity to call the indicator parameters dialog.

For a start we shall use the **GetRatesLC()** function from the **liquidchart.mqh** file. We shall call it from the **RefreshBuffers()** function, which, in its turn, is called from the [OnCalculate](https://www.mql5.com/en/docs/basis/function/events#oncalculate) function. It can also be called from [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events#onchartevent), provided that some alterations in the mode or the shift and recalculation of the indicator buffers are required. The [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events#onchartevent) function will be handling pressing the buttons and changing the values of the shift and the mode.

Input parameters of the indicator:

```
input ENUM_TIMEFRAMES   BaseTF=PERIOD_M1;       // LC Base Period
input int               Depth=100;              // Depth, bars
input ENUM_LC_MODE      inp_LC_mode=LC_MODE_SS; // LC mode
input int               inp_LC_shift=0;         // LC shift
```

where Depth is the number of bars of the resulting chart and ENUM\_LC\_MODE is the type describing the plotting modes of the indicator:

```
enum ENUM_LC_MODE
  {//plotting mode
   LC_MODE_SS=0,  // Static Shift
   LC_MODE_DSO=1, // Dynamic Shift, just Open
   LC_MODE_DSC=2  // Dynamic Shift, expected Close
  };
```

The **inp\_LC\_mode** and **inp\_LC\_shift** parameters are duplicated by **LC\_mode** and **LC\_shift** accordingly. This design allows changing their values by pressing the button. Drawing the buttons and handling pressing the buttons are not going to be considered as they are not relevant to the topic of this article. Let us consider the **RefreshBuffers()** function in detail.

```
bool RefreshBuffers(int total,
                    double &buff1[],
                    double &buff2[],
                    double &buff3[],
                    double &buff4[],
                    double &col_buffer[])
  {
   MqlRates rates[];
   ArrayResize(rates,Depth);
//---
   int copied=0;
   int shift=0;
   if(LC_mode==LC_MODE_SS)
      shift = LC_shift; //static shift
   else if(LC_mode==LC_MODE_DSO)
      shift = -1;       //calculate shift (beginning of the bar formation)
   else if(LC_mode==LC_MODE_DSC)
      shift = -2;       //calculate shift (end of the bar formation)
   else
      return(false);
//---
   copied=GetRatesLC(0,Depth,rates,BaseTF,shift);
//---
   if(copied<=0)
     {
      Print("No data");
      return(false);
     }
   LC_shift = shift;
   refr_keys();
//--- initialize buffers with empty values
   ArrayInitialize(buff1,0.0);
   ArrayInitialize(buff2,0.0);
   ArrayInitialize(buff3,0.0);
   ArrayInitialize(buff4,0.0);
//---
   int buffer_index=total-copied;
   for(int i=0;i<copied;i++)
     {
      buff1[buffer_index]=rates[i].open;
      buff2[buffer_index]=rates[i].high;
      buff3[buffer_index]=rates[i].low;
      buff4[buffer_index]=rates[i].close;
      //---
      if(rates[i].open<=rates[i].close)
         col_buffer[buffer_index]=1;//bullish or doji
      else
         col_buffer[buffer_index]=0;//bearish
      //
      buffer_index++;
     }
//
   return(true);
  }
```

At first, a relevant value of the **shift** variable is to be passed to the **GetRatesLC()** function depending on the mode. In the static mode it will be a copy of the **LC\_shift** parameter and in the opening or closing modes of the bar it will be -1 or -2 correspondingly. After successful execution of the function, **GetRatesLC()** returns the current value of the shift to the **shift** variable. It is either recalculated or left as it is. In any case, we assign its value to the **LC\_shift** variable and then call redrawing of the graphic elements by the **refr\_keys()** function.

After that, we renew the OHLC values and the bar colors in the indicator buffers.

The full code of the indicator can be found in the **liquid\_chart.mq5** file. After the launch, the indicator looks like follows:

![The Liquid Chart indicator, shift 0](https://c.mql5.com/2/12/USDJPYH1_LC_021.png)

Fig. 5. The Liquid Chart indicator

A few words about the control elements.

- The **SS** button switches the indicator to the static shift mode. Buttons with arrows are active in this mode and they can be used for setting the required value of the shift.
- The **DSO** button switches the indicator to the dynamic shift mode, beginning of the bar formation. In this mode the shift value is calculated and it cannot be modified manually.
- The **DSC** button switches the indicator the the dynamic shift mode, end of the bar formation. In this mode manual modifying of the shift is unavailable as well.

In the **SS** mode, when the shift value is 0, the indicator is duplicating the values of the initial chart. If you change the shift, you will see the chart redrawing. A noticeable difference already appears at the value of 28. Instead of feeble "rails" there is a distinct "hammer". Is it time to buy?

![The Liquid Chart indicator, shift 28](https://c.mql5.com/2/12/USDJPYH1_LC_022.png)

Fig. 6. The Liquid Chart indicator, static shift 28

Switch the indicator to the **DSO** mode, and a newly formed bar is always going to be on the right. In the **DSC** mode there is a bar going to close in no later than a basic timeframe value.

### 4\. Creating an Expert

We are going to create two Experts. The first one will be trading by Moving Average and the second one by the "pin bar" pattern.

Let's take the **Moving Average** Expert from the standard examples (the _Experts\\Examples\\Moving Average_ folder) as a template. This way we shall be able to compare the optimization results of two essentially different strategies to understand when using the static or dynamic shift is relevant and of some significance.

**4.1. Expert Trading by Moving Average**

At first the input parameters have to be defined. There are four of them:

```
input double MaximumRisk    = 0.1;  // Maximum Risk in percentage
input double DecreaseFactor = 3;    // Decrease factor
input int    MovingPeriod   = 12;   // Moving Average period
input int    MovingShift    = 6;    // Moving Average shift
```

Three more parameters will be added after modernization:

```
input ENUM_TIMEFRAMES  BaseTF=PERIOD_M1;  // LC Base Period
input bool             LC_on = true;      // LC mode ON
input int              LC_shift = 0;      // LC shift
```

The **LC\_on** parameter will be useful for checks if **GetRatesLC()** works correctly. The combination **(LC\_on == true && LC\_shift == 0)** must have the same result as **(LC\_on == false)**.

For modernization of the ready made Expert **Moving Average** with the shift, the **liquidchart.mqh** file has to be included and the [CopyRates()](https://www.mql5.com/en/docs/series/copyrates) functions have to be substituted with the **GetRatesLC()** functions for those cases when the shift feature is enabled (input parameter **LC\_on** is _true_):

```
   int copied;
   if(LC_on==true)
     {
      int shift = LC_shift;
      copied=GetRatesLC(0,2,rt,BaseTF,shift);
     }
   else
      copied = CopyRates(_Symbol,_Period,0,2,rt);
   if(copied!=2)
     {
      Print("CopyRates of ",_Symbol," failed, no history");
      return;
     }
```

It needs to be done both in the **CheckForOpen()** and **CheckForClose()** functions. We have to refuse from using indicator handles and will be calculating Moving Averages manually. For that we added the **CopyMABuffer()** function:

```
int CopyMABuffer(int len,double &ma[])
  {
   if(len<=0)
      return(0);
   MqlRates rates[];
   int l=len-1+MovingPeriod;
   int copied;
   if(LC_on==true)
     {
      int shift = LC_shift;
      ArrayResize(rates,l);
      copied=GetRatesLC(MovingShift,l,rates,BaseTF,shift);
     }
   else
      copied=CopyRates(_Symbol,_Period,MovingShift,l,rates);
//
   if(copied<l)
      return(0);
//
   for(int i=0;i<len;i++)
     {
      double sum=0;
      for(int j=0;j<MovingPeriod;j++)
        {
         if(LC_on==true)
            sum+=rates[j+i].close;
         else
            sum+=rates[copied-1-j-i].close;
        }
      ma[i]=sum/MovingPeriod;
     }
   return(len);
  }
```

It returns a required number of values or 0 in the **ma\[\]** buffer if for some reason they have failed to be obtained.

The control of opening bars is another important point to consider. In the original version of the **Moving Average** Expert Advisor it was implemented through using tick values:

```
   if(rt[1].tick_volume>1)
      return;
```

In our case there are no tick volumes and therefore we are going to write the **newbar()** function to control bar opening:

```
bool newbar(datetime t)
  {
   static datetime t_prev=0;
   if(t!=t_prev)
     {
      t_prev=t;
      return(true);
     }
   return(false);
  }
```

The operating principle lies in comparison of the opening time of the bar with its previous value. Let's substitute the check of the tick volume for the call of the **newbar()** function in the **CheckForOpen()** and **CheckForClose()** functions:

```
   if(newbar(rt[1].time)==false)
      return;
```

Complete code of the ready made Expert can be found in the **moving\_average\_lc.mq5** file.

**4.2. Expert Advisor Trading by the "Pin Bar" Pattern**

**Pin Bar** or **Pinocchio bar** is a pattern that consists of three bars. The middle bar has to have a long shadow or "nose", indicating a probable reversal of the price movement. The bars at the side are called "eyes". Their extreme points should not come out of the shadow of the neighboring bar. This pattern is popular with traders trading by candlestick models.

Our pin bar has to fulfill the following conditions of reversing the price downwards:

- The **r\[0\]** bar must be bullish.
- The **r\[2\]** bar must be bearish.
- The largest value of the **A** and **C** prices should not exceed **B**, where **A** and **C** are **High** values of the **r\[0\]** and **r\[2\]** bars respectively. **B** is the **High** price of the **r\[1\]** bar.
- The body of the middle bar, that is the module of difference between **Open** and **Close** ( **OC** at the figure) of the **r\[1\]** bar, should not exceed the number of points set by the external parameter.
- The shadow of the middle bar, or the difference of the **High** price and the greatest of **Open** and **Close** values of the **r\[1\]** bar, should not be less than the number of points set by the external parameter.
- The ratio of the middle bar shadow to its body should not be less than the value set by the external parameter.


The check of the pattern will be carried out at the time of opening the **r\[3\]** bar.

![Pinbar](https://c.mql5.com/2/12/USDJPYH1_LC_014.png)

Fig. 7. The "Pin Bar" pattern

Code defining presence of the pin bar for reversal downwards should look like:

```
   if(r[0].open<r[0].close && r[2].open>r[2].close && r[1].high>MathMax(r[0].high,r[2].high))
     {
     //--- eyes of the upper pin bar
      double oc=MathAbs(r[1].open-r[1].close)/_Point;
      if(oc>inp_pb_max_OC)
         return(0);
      double shdw=(r[1].high-MathMax(r[1].open,r[1].close))/_Point;
      if(shdw<inp_pb_min_shdw)
         return(0);
      if(oc!=0)
        {
         if((shdw/oc)<inp_pb_min_ratio)
            return(0);
        }
      return(1);
     }
```

Same for the reversal upwards. So, the function of checking for the pin bar presence will look as follows:

```
int IsPinbar(MqlRates &r[])
  {
   //--- there must be 4 values in the r[] array
   if(ArraySize(r)<4)
      return(0);
   if(r[0].open<r[0].close && r[2].open>r[2].close && r[1].high>MathMax(r[0].high,r[2].high))
     {
      //--- eyes of the upper pin bar
      double oc=MathAbs(r[1].open-r[1].close)/_Point;
      if(oc>inp_pb_max_OC)
         return(0);
      double shdw=(r[1].high-MathMax(r[1].open,r[1].close))/_Point;
      if(shdw<inp_pb_min_shdw)
         return(0);
      if(oc!=0)
        {
         if((shdw/oc)<inp_pb_min_ratio)
            return(0);
        }
      return(1);
     }
   else if(r[0].open>r[0].close && r[2].open<r[2].close && r[1].low<MathMin(r[0].low,r[2].low))
     {
      //--- eyes of the lower pin bar
      double oc=MathAbs(r[1].open-r[1].close)/_Point;
      if(oc>inp_pb_max_OC)
         return(0);
      double shdw=(MathMin(r[1].open,r[1].close)-r[1].low)/_Point;
      if(shdw<inp_pb_min_shdw)
         return(0);
      if(oc!=0)
        {
         if((shdw/oc)<inp_pb_min_ratio)
            return(0);
        }
      return(-1);
     }
   return(0);
  }
```

The passed array of historical data should not have less than four elements. In case the upper pin bar is detected (that is pin bar indicating the reversal downwards), the function will return the value of 1. In case there is a lower pin bar (supposed reversal upwards), the function will return the value of -1. The function will return 0 if there were no pin bars at all. The function also uses the following input parameters:

```
input uint   inp_pb_min_shdw     = 40;    // Pin bar min shadow, point
input uint   inp_pb_max_OC       = 20;    // Pin bar max OC, point
input double inp_pb_min_ratio    = 2.0;   // Pin bar shadow to OC min ratio
```

We are going to trade by the simplest trading strategy by the pin bar. We shall sell if a reversal downwards is expected and buy if the reversal is upwards. Normally, indicator confirmations are required but this time we are not going to use them to maintain experimental integrity. We are going to use only the pin bar.

The Expert Advisor trading by the "Pin Bar" pattern will be based on the Expert trading by Moving Average. The **CopyMABuffer()** function has to be deleted from the latter as well as calling this function from the **CheckForOpen()** and **CheckForClose()** functions. The number of requested historical data is to be increased from two to four. Time of the **r\[3\]** bar will be used in the check for the new bar opening.

```
   int copied;
   if(LC_on==true)
   {
      int shift = LC_shift;
      copied=GetRatesLC(0,4,rt,BaseTF,shift);
   }
   else
      copied = CopyRates(_Symbol,_Period,0,4,rt);
   if(copied!=4)
     {
      Print("CopyRates of ",_Symbol," failed, no history");
      return;
     }
   if(newbar(rt[3].time)==false)
      return;
```

The check of a signal for the position opening will look like:

```
   int pb=IsPinbar(rt);
   if(pb==1)       // upper pin bar
      signal=ORDER_TYPE_SELL; // sell conditions
   else if(pb==-1) // lower pin bar
      signal=ORDER_TYPE_BUY;  // buy conditions
```

To close position by the opposite pin bar:

```
   if(type==(long)POSITION_TYPE_BUY && pb==1)
      signal=true;
   if(type==(long)POSITION_TYPE_SELL && pb==-1)
      signal=true;
```

Please note that in strict conditions of input parameters, pin bar **seldom** occurs. Therefore when closing a position only by an opposite pin bar there is a risk to lose profit or close at loss.

In this connection, we add the **Take Profit** and **Stop Loss** levels. They are going to be set by the external parameters **inp\_tp\_pp** and **inp\_sl\_pp** respectively:

```
   double sl=0,tp=0,p=0;
   double ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);
   if(signal==ORDER_TYPE_SELL)
   {
     p=bid;
     if(inp_sl_pp!=0)
       sl=NormalizeDouble(ask+inp_sl_pp*_Point,_Digits);
     if(inp_tp_pp!=0)
       tp=NormalizeDouble(ask-inp_sl_pp*_Point,_Digits);
   } else {
     p=ask;
     if(inp_sl_pp!=0)
       sl=NormalizeDouble(bid-inp_sl_pp*_Point,_Digits);
     if(inp_tp_pp!=0)
       tp=NormalizeDouble(bid+inp_sl_pp*_Point,_Digits);
   }
   CTrade trade;
   trade.PositionOpen(_Symbol,signal,TradeSizeOptimized(),p,sl,tp);
```

If the value of **inp\_tp\_pp** or **inp\_sl\_pp** is zero, then the correspondent level of **Take Profit** or **Stop Loss** will not be set.

Modification is complete. The Expert is ready. The complete code can be found in the **pinbar\_lc.mq5** file.

### 5\. Expert Optimization

To assess the effectiveness of charts with a shift in different strategies, we shall use Expert optimization with further comparison of the best results. The most important parameters here are profit, drawdown and the number of trades. Expert Advisor trading by Moving Average will serve an example of an indicator strategy and the Expert trading by the "Pin Bar" pattern will represent a case of a non-indicator strategy.

Optimization will be carried out by quotes for the last half a year from the **MetaQuotes-Demo** server. The experiment will be conducted for EURUSD, GBPUSD and USDJPY. Start deposit is 3000 USD with leverage of 1:100. Test mode is "All ticks". Optimization mode - fast ( [generic algorithm](https://www.mql5.com/en/articles/1409)), Balance Max.

**5.1. Analysis of Optimization Results of an Expert Trading by Moving Average**

Let us compare results of optimizing an Expert when it is working in different modes: with zero shift, with static shift and with dynamic shift ( **DSO** and **DSC**).

Test will be conducted for EURUSD, GBPUSD and USDJPY in the period of 2014.04.01 - 2014.10.25 (last six months). Period H1.

Input parameters of the Expert:

| Parameter | Value |
| --- | --- |
| Maximum Risk in percentage | 0.1 |
| Descrease factor | 3.0 |
| Moving Average period | 12 |
| Moving Average shift | 6 |
| BaseTF | 1 Minute |
| LC\_on | true |
| LC\_shift | 0 |

Table 3. Input parameters of the Moving Average LC Expert

**5.1.1. Expert Optimization in the Disabled Shift Mode**

Optimized parameters:

| Parameter | Start | Step | Stop |
| --- | --- | --- | --- |
| Moving Average period | 12 | 1 | 90 |
| Moving Average shift | 6 | 1 | 30 |

Table 4. Optimized parameters of the Moving Average LC Expert in the zero shift mode

Graph depicting the Expert optimization in the disables shift mode, EURUSD:

![Optimization in the zero shift mode, EURUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_27.png)

Fig. 8. Optimization of the Moving Average LC Expert in the zero shift mode, EURUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift |
| --- | --- | --- | --- | --- | --- |
| 3796,43 | 796,43 | 16,18 | 111 | 24 | 12 |
| 3776,98 | 776,98 | 17,70 | 77 | 55 | 22 |
| 3767,45 | 767,45 | 16,10 | 74 | 59 | 23 |
| 3740,38 | 740,38 | 15,87 | 78 | 55 | 17 |
| 3641,16 | 641,16 | 15,97 | 105 | 12 | 17 |

Table 5. Best results of the Moving Average LC Expert for EURUSD in the zero shift mode

Graph depicting Expert optimization in the zero shift mode, GBPUSD:

![Optimization in the zero shift mode, GBPUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_27__2.png)

Fig. 9. Optimization of the Moving Average LC Expert in the zero shift mode, GBPUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift |
| --- | --- | --- | --- | --- | --- |
| 4025,75 | 1025,75 | 8,08 | 80 | 18 | 22 |
| 3857,90 | 857,90 | 15,04 | 74 | 55 | 13 |
| 3851,40 | 851,40 | 18,16 | 80 | 13 | 24 |
| 3849,48 | 849,48 | 13,05 | 69 | 34 | 29 |
| 3804,70 | 804,70 | 16,57 | 137 | 25 | 8 |

Table 6. Best results of the Moving Average LC Expert optimization for GBPUSD in the zero shift mode

Graph depicting Expert optimization in the disabled shift mode, USDJPY:

![Optimization in the zero shift mode, USDJPY](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_27__5.png)

Fig. 10. Optimization of the Moving Average LC Expert in the zero shift mode, USDJPY

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift |
| --- | --- | --- | --- | --- | --- |
| 5801,63 | 2801,63 | 11,54 | 48 | 65 | 23 |
| 5789,17 | 2789,17 | 14,03 | 50 | 44 | 27 |
| 5539,06 | 2539,06 | 17,14 | 46 | 67 | 27 |
| 5331,34 | 2331,34 | 15,05 | 61 | 70 | 9 |
| 5045,19 | 2045,19 | 12,61 | 48 | 83 | 15 |

Table 7. Best results of the Moving Average LC Expert for USDJPY in the zero shift mode

**5.1.2. Expert Optimization in the Static Shift Mode**

Optimized parameters:

| Parameter | Start | Step | Stop |
| --- | --- | --- | --- |
| Moving Average period | 12 | 1 | 90 |
| Moving Average shift | 6 | 1 | 30 |
| LC\_shift | 1 | 1 | 59 |

Table 8. Optimized parameters of the Moving Average LC Expert in the static shift mode

Graph depicting Expert optimization in the static shift mode, EURUSD:

![Optimization in the static shift mode, EURUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_26.png)

Fig. 11. Optimization of the Moving Average LC Expert in the static shift mode, EURUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift | LC\_shift |
| --- | --- | --- | --- | --- | --- | --- |
| 4385,06 | 1385,06 | 12,87 | 100 | 32 | 11 | 8 |
| 4149,63 | 1149,63 | 14,22 | 66 | 77 | 25 | 23 |
| 3984,92 | 984,92 | 21,52 | 122 | 12 | 11 | 26 |
| 3969,35 | 969,35 | 16,08 | 111 | 32 | 11 | 24 |
| 3922,95 | 922,95 | 12,29 | 57 | 77 | 25 | 10 |

Table 9. Best optimization results of the Moving Average LC Expert for EURUSD in the static shift mode

Graph depicting Expert optimization in the static shift mode, GBPUSD:

![Optimization in the static shift mode, GBPUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_27__3.png)

Fig. 12. Optimization of the Moving Average LC Expert in the static shift mode, GBPUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift | LC\_shift |
| --- | --- | --- | --- | --- | --- | --- |
| 4571,07 | 1571,07 | 14,90 | 79 | 12 | 25 | 42 |
| 4488,90 | 1488,90 | 15,46 | 73 | 12 | 25 | 47 |
| 4320,31 | 1320,31 | 9,59 | 107 | 12 | 16 | 27 |
| 4113,47 | 1113,47 | 10,96 | 75 | 12 | 25 | 15 |
| 4069,21 | 1069,21 | 15,27 | 74 | 12 | 25 | 50 |

Table 10. Best optimization results of the Moving Average LC Expert for GBPUSD in the static shift mode

Graph depicting Expert optimization in the static shift mode, USDJPY:

![Optimization in the static shift mode, USDJPY](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_26__1.png)

Fig. 13. Optimization of the Moving Average LC Expert in the static shift mode, USDJPY

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift | LC\_shift |
| --- | --- | --- | --- | --- | --- | --- |
| 6051,39 | 3051,39 | 15,94 | 53 | 76 | 12 | 31 |
| 5448,98 | 2448,98 | 10,71 | 54 | 44 | 30 | 2 |
| 5328,15 | 2328,15 | 11,90 | 50 | 82 | 13 | 52 |
| 5162,82 | 2162,82 | 10,46 | 71 | 22 | 26 | 24 |
| 5154,71 | 2154,71 | 14,34 | 54 | 75 | 14 | 58 |

Table 11. Best optimization results of the Moving Average LC Expert for USDJPY in the static shift mode

**5.1.3. Expert Optimization in the Dynamic Shift Mode**

Optimized parameters:

| Parameter | Start | Step | Stop |
| --- | --- | --- | --- |
| Moving Average period | 12 | 1 | 90 |
| Moving Average shift | 6 | 1 | 30 |
| LC\_shift | -2 | 1 | -1 |

Table 12. Optimized parameters of the Moving Average LC Expert in the dynamic shift mode

Graph depicting Expert optimization in the dynamic shift mode, EURUSD:

![Optimization in the dynamic shift mode, EURUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_27__1.png)

Fig. 14. Optimization of the Moving Average LC Expert in the dynamic shift mode, EURUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift | LC\_shift |
| --- | --- | --- | --- | --- | --- | --- |
| 3392,64 | 392,64 | 27,95 | 594 | 15 | 13 | -2 |
| 3140,26 | 140,26 | 23,35 | 514 | 12 | 17 | -2 |
| 2847,12 | -152,88 | 17,04 | 390 | 79 | 23 | -1 |
| 2847,12 | -152,88 | 17,04 | 390 | 79 | 12 | -1 |
| 2826,25 | -173,75 | 20,12 | 350 | 85 | 22 | -1 |

Table 13. Best optimization results of the Moving Average LC Expert for EURUSD in the dynamic shift mode

Graph depicting Expert optimization in the dynamic shift mode, GBPUSD:

![Optimization in the dynamic shift mode, GBPUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_27__4.png)

Fig. 15. Optimization of the Moving Average LC Expert in the dynamic shift mode, GBPUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift | LC\_shift |
| --- | --- | --- | --- | --- | --- | --- |
| 5377,58 | 2377,58 | 19,73 | 391 | 12 | 26 | -2 |
| 3865,50 | 865,50 | 18,18 | 380 | 23 | 23 | -2 |
| 3465,63 | 465,63 | 21,22 | 329 | 48 | 21 | -2 |
| 3428,99 | 428,99 | 24,55 | 574 | 51 | 16 | -1 |
| 3428,99 | 428,99 | 24,55 | 574 | 51 | 15 | -1 |

Table 14. Best optimization results of the Moving Average LC Expert for GBPUSD in the dynamic shift mode

Graph depicting Expert optimization in the dynamic shift mode, USDJPY:

![Optimization in the dynamic shift mode, USDJPY](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_27__6.png)

Fig. 16. Optimization of the Moving Average LC Expert in the dynamic shift mode, USDJPY

Best results:

| Result | Profit | Drawdown % | Number of trades | MovingPeriod | MovingShift | LC\_shift |
| --- | --- | --- | --- | --- | --- | --- |
| 6500,19 | 3500,19 | 17,45 | 244 | 42 | 28 | -2 |
| 6374,18 | 3374,18 | 19,91 | 243 | 54 | 24 | -2 |
| 6293,29 | 3293,29 | 19,30 | 235 | 48 | 27 | -2 |
| 5427,69 | 2427,69 | 17,65 | 245 | 90 | 8 | -2 |
| 5421,83 | 2421,83 | 16,30 | 301 | 59 | 12 | -2 |

Table 15. Best optimization results of the Moving Average LC Expert for USDJPY in the dynamic shift mode

**5.2. Analysis of Optimization Results of the Expert Trading by the "Pin Bar" Pattern**

Let us compare results of optimizing an Expert when it is working in different modes: with zero shift, with static shift and with dynamic shift ( **DSO** and **DSC**). Test will be carried out for EURUSD, GBPUSD and USDJPY, on the period of 2014.04.01 - 2014.10.25. Period H1.

Input parameters of the Expert:

| Parameter | Value |
| --- | --- |
| Maximum Risk in percentage | 0.1 |
| Decrease factor | 3.0 |
| Pin bar min shadow, points | 40 |
| Pin bar max OC, points | 110 |
| Pin bar shadow to OC min ratio | 1.4 |
| SL, points (0 for OFF) | 150 |
| TP, points (0 for OFF) | 300 |
| LC Base Period | 1 Minute |
| LC mode ON | true |
| LC shift | 0 |

Table 16. Input parameters of the Pin Bar LC Expert

We are going to optimize parameters defining the shape of the pin bar: length of the "nose", ratio of the length of the "nose" to the body of the middle bar and maximum size of the body. The levels **Take Profit** and **Stop Loss** are to be optimized too.

**5.2.1. Expert Optimization in the Disabled Shift Mode**

Optimized parameters:

| Parameter | Start | Step | Stop |
| --- | --- | --- | --- |
| Pin bar min shadow, points | 100 | 20 | 400 |
| Pin bar max OC, points | 20 | 20 | 100 |
| Pin bar shadow to OC min ratio | 1 | 0.2 | 3 |
| SL, points (0 for OFF) | 150 | 50 | 500 |
| TP, points (0 for OFF) | 150 | 50 | 500 |

Table 17. Optimized parameters of the Pin Bar LC Expert in the disabled shift mode

Graph depicting the Expert optimization in the disables shift mode, EURUSD:

![Optimization with zero shift, EURUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_30.png)

Fig. 17. Optimization of the Pin Bar LC Expert in the zero shift mode, EURUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin bar min shadow | Pin bar max OC | Pin bar shadow <br>to OC min ratio | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3504,59 | 504,59 | 9,82 | 33 | 100 | 60 | 1.8 | 450 | 500 |
| 3428,89 | 428,89 | 8,72 | 21 | 120 | 60 | 2.8 | 450 | 350 |
| 3392,37 | 392,37 | 9,94 | 30 | 100 | 60 | 2,6 | 450 | 250 |
| 3388,54 | 388,54 | 9,93 | 31 | 100 | 80 | 2,2 | 450 | 300 |
| 3311,84 | 311,84 | 6,84 | 13 | 140 | 60 | 2,2 | 300 | 450 |

Table 18. Best optimization results of the Pin Bar LC Expert for EURUSD in the zero shift mode

Graph depicting Expert optimization in the zero shift mode, GBPUSD:

![Optimization with zero shift, GBPUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_30__2.png)

Fig. 18. Optimization of the Pin Bar LC Expert in the zero shift mode, GBPUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin Bar min shadow | Pin Bar max OC | Pin Bar shadow <br>to OC min ratio | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3187,13 | 187,13 | 11,10 | 13 | 160 | 60 | 2,6 | 500 | 350 |
| 3148,73 | 148,73 | 3,23 | 4 | 220 | 40 | 2,8 | 400 | 400 |
| 3142,67 | 142,67 | 11,27 | 17 | 160 | 100 | 1,8 | 500 | 350 |
| 3140,80 | 140,80 | 11,79 | 13 | 180 | 100 | 2 | 500 | 500 |
| 3094,20 | 94,20 | 1,62 | 1 | 260 | 60 | 1,6 | 500 | 400 |

Table 19. Best optimization results of the Pin Bar LC Expert for GBPUSD in the zero shift mode

Graph depicting Expert optimization in the disabled shift mode, USDJPY:

![Optimization with zero shift, USDJPY](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_29.png)

Fig. 19. Optimization of the Pin Bar LC Expert in the zero shift mode, USDJPY

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin bar min shadow | Pin bar max OC | Pin bar shadow <br>to OC min ratio | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3531,99 | 531,99 | 9,00 | 6 | 160 | 60 | 2.2 | 450 | 500 |
| 3355,91 | 355,91 | 18,25 | 16 | 120 | 60 | 1,6 | 450 | 400 |
| 3241,93 | 241,93 | 9,11 | 4 | 160 | 40 | 2,8 | 450 | 500 |
| 3180,43 | 180,43 | 6,05 | 33 | 100 | 80 | 1,8 | 150 | 450 |
| 3152,97 | 152,97 | 3,14 | 6 | 160 | 80 | 2,8 | 150 | 500 |

Table 20. Best optimization results of the Pin Bar LC Expert for USDJPY in the zero shift mode

**5.2.2. Expert Optimization in the Static Shift Mode**

Optimized parameters:

| Parameter | Start | Step | Stop |
| --- | --- | --- | --- |
| Pin bar min shadow, points | 100 | 20 | 400 |
| Pin bar max OC, points | 20 | 20 | 100 |
| Pin bar shadow to OC min ratio | 1 | 0.2 | 3 |
| SL, points (0 for OFF) | 150 | 50 | 500 |
| TP, points (0 for OFF) | 150 | 50 | 500 |
| LC shift | 1 | 1 | 59 |

Table 21. Optimized parameters of the Pin Bar LC Expert in the static shift mode

Graph depicting Expert optimization in the static shift mode, EURUSD:

![Optimization in the static shift mode, EURUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_31.png)

Fig. 20. Optimization of the Pin Bar LC Expert in the static shift mode, EURUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin bar min shadow | Pin bar max OC | Pin bar shadow <br>to OC min ratio | SL | TP | LC shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4843,54 | 1843,54 | 10,14 | 19 | 120 | 80 | 1,6 | 500 | 500 | 23 |
| 4714,81 | 1714,81 | 10,99 | 28 | 100 | 100 | 1,6 | 500 | 500 | 23 |
| 4672,12 | 1672,12 | 10,16 | 18 | 120 | 80 | 1,8 | 500 | 500 | 23 |
| 4610,13 | 1610,13 | 9,43 | 19 | 120 | 80 | 1,6 | 450 | 450 | 23 |
| 4562,21 | 1562,21 | 13,94 | 27 | 100 | 100 | 1,6 | 500 | 400 | 25 |

Table 22. Best optimization results of the Pin Bar LC Expert for EURUSD in the static shift mode

Graph depicting Expert optimization in the static shift mode, GBPUSD:

![Optimization in the static shift mode, GBPUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_11_01.png)

Fig. 21. Optimization of the Pin Bar LC Expert in the static shift mode, GBPUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin bar min shadow | Pin bar max OC | Pin bar shadow <br>to OC min ratio | SL | TP | LC shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4838,10 | 1838,10 | 5,60 | 34 | 100 | 40 | 2,4 | 450 | 500 | 24 |
| 4797,09 | 1797,09 | 5,43 | 35 | 100 | 40 | 2,6 | 400 | 500 | 24 |
| 4755,57 | 1755,57 | 7,36 | 42 | 100 | 100 | 2 | 400 | 500 | 24 |
| 4725,41 | 1725,41 | 8,35 | 45 | 100 | 80 | 1 | 400 | 500 | 24 |
| 4705,61 | 1705,61 | 8,32 | 41 | 100 | 100 | 2 | 450 | 500 | 24 |

Table 23. Best optimization results of the Pin Bar LC Expert for GBPUSD in the static shift mode

Graph depicting Expert optimization in the static shift mode, USDJPY:

![Optimization in the static shift mode, USDJPY](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_30__4.png)

Fig. 22. Optimization of the Pin Bar LC Expert in the static shift mode, USDJPY

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin Bar min shadow | Pin bar max OC | Pin bar shadow <br>to OC min ratio | SL | TP | LC shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4108,83 | 1108,83 | 6,45 | 9 | 140 | 40 | 1,4 | 500 | 450 | 55 |
| 3966,74 | 966,74 | 7,88 | 12 | 140 | 60 | 2,8 | 450 | 500 | 45 |
| 3955,32 | 955,32 | 9,91 | 21 | 120 | 80 | 2 | 500 | 500 | 45 |
| 3953,80 | 953,80 | 6,13 | 10 | 140 | 60 | 2,8 | 450 | 450 | 47 |
| 3944,33 | 944,33 | 6,42 | 6 | 160 | 100 | 2,6 | 500 | 400 | 44 |

Table 24. Best optimization results of the Pin Bar LC Expert for USDJPY in the static shift mode

**5.2.3. Expert Optimization in the Dynamic Shift Mode**

Optimized parameters:

| Parameter | Start | Step | Stop |
| --- | --- | --- | --- |
| Pin bar min shadow, points | 100 | 20 | 400 |
| Pin bar max OC, points | 20 | 20 | 100 |
| Pin bar shadow to OC min ratio | 1 | 0.2 | 3 |
| SL, points (0 for OFF) | 150 | 50 | 500 |
| TP, points (0 for OFF) | 150 | 50 | 500 |
| LC shift | -2 | 1 | -1 |

Table 25. Optimized parameters of the Pin Bar LC Expert in the dynamic shift mode

Graph depicting Expert optimization in the dynamic shift mode, EURUSD:

![Optimization in the dynamic shift mode, EURUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_30__1.png)

Fig. 23. Optimization of the Pin Bar LC Expert in the dynamic shift mode, EURUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin bar min shadow | Pin bar max OC | Pin bar shadow <br>to OC min ratio | SL | TP | LC shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4185,65 | 1185,65 | 13,22 | 49 | 200 | 100 | 1,8 | 450 | 500 | -2 |
| 4011,80 | 1011,80 | 13,75 | 49 | 200 | 100 | 2 | 400 | 500 | -2 |
| 3989,28 | 989,28 | 12,01 | 76 | 140 | 20 | 1,2 | 350 | 200 | -1 |
| 3979,50 | 979,50 | 16,45 | 157 | 100 | 20 | 1 | 450 | 500 | -1 |
| 3957,25 | 957,25 | 16,68 | 162 | 100 | 20 | 1 | 400 | 500 | -1 |

Table 26. Best optimization results of the Pin Bar LC Expert for EURUSD in the static shift mode

Graph depicting Expert optimization in the dynamic shift mode for GBPUSD:

![Optimization in the dynamic shift mode, GBPUSD](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_30__3.png)

Fig. 24. Optimization of the Pin Bar LC Expert in the dynamic shift mode, GBPUSD

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin bar min shadow | Pin bar max OC | Pin bar shadow <br>to OC min ratio | SL | TP | LC shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4906,84 | 1906,84 | 10,10 | 179 | 120 | 40 | 1,8 | 500 | 500 | -2 |
| 4316,46 | 1316,46 | 10,71 | 151 | 120 | 20 | 2,4 | 450 | 500 | -1 |
| 4250,96 | 1250,96 | 12,40 | 174 | 120 | 40 | 1,8 | 500 | 500 | -1 |
| 4040,82 | 1040,82 | 12,40 | 194 | 120 | 60 | 2 | 500 | 200 | -2 |
| 4032,85 | 1032,85 | 11,70 | 139 | 140 | 40 | 2 | 400 | 200 | -1 |

Table 27. Best optimization results of the Pin Bar LC Expert for GBPUSD in the dynamic shift mode

Graph depicting Expert optimization in the dynamic shift mode, USDJPY:

![Optimization in the dynamic shift mode, USDJPY](https://c.mql5.com/2/12/TesterOptgraphReport2014_10_30__5.png)

Fig. 25. Optimization of the Pin Bar LC Expert in the dynamic shift mode, USDJPY

Best results:

| Result | Profit | Drawdown % | Number of trades | Pin bar min shadow | Pin bar max OC | Pin bar shadow <br>to OC min ratio | SL | TP | LC shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5472,67 | 2472,67 | 13,01 | 138 | 100 | 20 | 2,4 | 500 | 500 | -1 |
| 4319,84 | 1319,84 | 15,87 | 146 | 100 | 20 | 2,2 | 400 | 500 | -1 |
| 4259,54 | 1259,54 | 19,71 | 137 | 100 | 20 | 2,4 | 500 | 500 | -2 |
| 4197,57 | 1197,57 | 15,98 | 152 | 100 | 20 | 1 | 350 | 500 | -1 |
| 3908,19 | 908,19 | 16,79 | 110 | 120 | 40 | 3 | 400 | 400 | -1 |

Table 28. Best optimization results of the Pin Bar LC Expert for USDJPY in the dynamic shift mode

### 6\. Comparison of Optimization Results

For making a comparison table, let us select maximum values of profit, drawdown and a number of trades from every table of best results. Next to the value received in the static or dynamic shift modes, we shall write the change (in percent) or this value in relation to the same value in the zero shift mode.

**6.1. Expert Trading by Moving Average**

Profit:

|  | No shift | Static<br>shift | Dynamic<br>shift |
| --- | --- | --- | --- |
| EURUSD | 796,43 | 1385,06 (+74%) | 392,64 (-51%) |
| GBPUSD | 1025,75 | 1571,07 (+53%) | 2377,58 (+132%) |
| USDJPY | 2801,63 | 3051,39 (+9%) | 3500,19 (+25%) |

Table 29. Comparison of the maximum profit values from the table of the best optimization results of the Moving Average LC Expert

Drawdown:

|  | No shift | Static<br>shift | Dynamic<br>shift |
| --- | --- | --- | --- |
| EURUSD | 17,7 | 21,52 (+22%) | 27,95 (+58%) |
| GBPUSD | 18,16 | 15,46 (-15%) | 24,55 (+35%) |
| USDJPY | 17,14 | 15,94 (-7%) | 19,91 (+16%) |

Table 30. Comparison of the maximum drawdown values from the table of best optimization results of the Moving Average LC Expert

Number of trades:

|  | No shift | Static<br>shift | Dynamic<br>shift |
| --- | --- | --- | --- |
| EURUSD | 111 | 122 (+10%) | 594 (+435%) |
| GBPUSD | 137 | 107 (-22%) | 574 (+319%) |
| USDJPY | 61 | 71 (+16%) | 301 (+393%) |

Table 31. Comparison of maximum values of the number of trades from the table of best optimization results of the Moving Average LC Expert

The first thing that catches the eye straight away is a significant increase of the entry points in the dynamic shift mode. At the same time, however, the drawdown has noticeably increased and profit has decreased two times for EURUSD.

The mode of static shift is more profitable for this Expert Advisor. Here we can see the decrease of drawdown for GBPUSD and USDJPY and a significant increase in profit for EURUSD and GBPUSD.

**6.2. Expert Trading by the "Pin Bar" Pattern**

Profit:

|  | No shift | Static<br>shift | Dynamic<br>shift |
| --- | --- | --- | --- |
| EURUSD | 504,59 | 1843,54 (+265%) | 1185,65 (+135%) |
| GBPUSD | 187,13 | 1838,10 (+882%) | 1906,84 (+919%) |
| USDJPY | 531,99 | 1108,83 (+108%) | 2472,67 (+365%) |

Table 32. Comparison of maximum profit values from the tables of best optimization results of the Pin Bar LC Expert

Drawdown:

|  | No shift | Static<br>shift | Dynamic<br>shift |
| --- | --- | --- | --- |
| EURUSD | 9,94 | 13,94 (+40%) | 16,68 (+68%) |
| GBPUSD | 11,79 | 8,35 (-29%) | 12,4 (+5%) |
| USDJPY | 18,25 | 9,91 (-46%) | 19,71 (+8%) |

Table 33. Comparison of the maximum drawdown values from the table of best optimization results of the Pin Bar LC Expert

Number of trades:

|  | No shift | Static<br>shift | Dynamic<br>shift |
| --- | --- | --- | --- |
| EURUSD | 33 | 28 (-15%) | 162 (+391%) |
| GBPUSD | 17 | 45 (+165%) | 194 (+1041%) |
| USDJPY | 33 | 21 (-36%) | 152 (+361%) |

Table 34. Comparison of maximum values of the number of trades from the table of best optimization results of the Pin Bar LC Expert

Here we also see substantial increase of the number of trades in the dynamic shift mode. In this case though, a significant drawdown increase, like in the similar case with the Moving Average LC Expert, is only for EURUSD. For other pairs the drawdown increases insignificantly, approximately by 5-8 percent.

In the static shift mode, optimization showed limited profit for GBPUSD and USDJPY. Nevertheless, we can see a significant drawdown decrease for the same pairs and its increase for EURUSD. The mode of static shift looks less profitable for this Expert Advisor.

### Conclusion

In this article we considered plotting principles of "liquid chart" and compared the influence of its operation modes on the optimization result of the Expert based on indicator and non-indicator strategies.

Hence the following conclusions:

- For the Experts using indicator strategies (for instance trading by Moving Average) the static shift mode is more suitable. It ensures a more precises market entry, which results in drawdown decrease and increase in profit.
- The dynamic shift mode is better for the Experts trading by patterns. This mode increases the number of entry points though the drawdown increases at the same time.
- The dynamic shift mode together with well organized money management can give impressive results.

- Although looking very promising for indicator strategies, the static shift mode has one significant drawback. The shift value providing the best result is another variable in the list of input parameters that has to be guessed right.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1208](https://www.mql5.com/ru/articles/1208)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1208.zip "Download all attachments in the single ZIP archive")

[liquid\_chart.mq5](https://www.mql5.com/en/articles/download/1208/liquid_chart.mq5 "Download liquid_chart.mq5")(27.15 KB)

[moving\_average\_lc.mq5](https://www.mql5.com/en/articles/download/1208/moving_average_lc.mq5 "Download moving_average_lc.mq5")(16.98 KB)

[pinbar\_lc.mq5](https://www.mql5.com/en/articles/download/1208/pinbar_lc.mq5 "Download pinbar_lc.mq5")(18.61 KB)

[liquidchart.mqh](https://www.mql5.com/en/articles/download/1208/liquidchart.mqh "Download liquidchart.mqh")(8.32 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://www.mql5.com/en/articles/7495)
- [Applying network functions, or MySQL without DLL: Part I - Connector](https://www.mql5.com/en/articles/7117)
- [Using OpenCL to test candlestick patterns](https://www.mql5.com/en/articles/4236)
- [Drawing Dial Gauges Using the CCanvas Class](https://www.mql5.com/en/articles/1699)
- [Working with GSM Modem from an MQL5 Expert Advisor](https://www.mql5.com/en/articles/797)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/37889)**
(41)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
5 Aug 2015 at 10:47

**Serhii Shevchuk:**

We add a shift of 1 unit of the base period. In this case, it is 1 hour. Now all bars of the resulting chart will be rearranged. The day will begin not at 00:00:00:00, but at 01:00:00. So bar N will open at 01:00:00:00 2015.08.03, bar N+1 at 01:00:00:00 2015.08.04, bar N+2 will open at 01:00:00 2015.08.05, and so on. But we have data for the time from 00:00:00:00 to 01:00:00. We can't throw them away, so the bar for Sunday is formed from them.

Everything is logical: if, taking into account the shift, our synthetic "day" now starts at 01:00:00, it should end in 24 hours, i.e. at 00:59:59 of the next calendar day. We cannot add Sunday's data to Friday's bar because the gap between the opening times of the base period bars is more than a day.

This is a philosophical question: either we cut the [synchronisation of bars](https://www.mql5.com/en/articles/239 "Article \"Testing Basics in MetaTrader 5\"") by phantoms, or we need to pump the base bars from day to day without regard to the time gap. Who likes what better.

I have counted by the algorithm described above (counting the shift from the beginning of one bar of the current tf to the next), that is, if we continue the same example and need a shift of 1 hour, then the Friday bar will include everything from 1:00 Friday to 1:00 Monday. The reasoning is simple - Friday and Monday are neighbouring bars in the current period, there are no Sundays there, and therefore cannot be in the indicator.

![Serhii Shevchuk](https://c.mql5.com/avatar/2014/1/52E54A92-12F9.jpg)

**[Serhii Shevchuk](https://www.mql5.com/en/users/decanium)**
\|
5 Aug 2015 at 11:00

**Stanislav Korotky:**

But this does not seem to be the case - here the Sunday bar is in the broker's quotes - both on the daily and H1, so it is not a "phantom" created in the indicator, but a real bar.

And how is the Sunday bar created in the indicator worse than the Sunday bar in the broker's quotes? If we get synthetics by shifting the [opening time](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer "MQL5 Documentation: Position Properties"), the same rules apply to it. If there is data for an hour before Monday, then Sunday is formed from it, and no other way. It is quite normal that the resulting chart crawls away from the corresponding bars of the initial chart.


![Serhii Shevchuk](https://c.mql5.com/avatar/2014/1/52E54A92-12F9.jpg)

**[Serhii Shevchuk](https://www.mql5.com/en/users/decanium)**
\|
5 Aug 2015 at 11:02

**Stanislav Korotky:**

The reasoning is simple - Friday and Monday are neighbouring bars in the current period, there are no Sundays there, and therefore cannot be in the indicator.

Here I strongly disagree, but I have no desire to argue.


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
5 Aug 2015 at 11:18

**Serhii Shevchuk:**

How is a Sunday bar created in the indicator worse than a Sunday bar in the broker's quotes? If we get synthetics by shifting the [opening time](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer "MQL5 Documentation: Position Properties"), the same rules apply to it. If there is data for an hour before Monday, then Sunday is formed from it, and no other way. It is quite normal that the resulting chart crawls away from the corresponding bars of the initial chart.

Quotes are initial data, so the presence of Sundays or Saturdays in them is not discussed and cannot be changed - the broker gives it to us from above.

And the initiator is bound to quotes and should be synchronised with them. The presence of phantoms breaks this binding. It is at least inconvenient.

But let everyone decide for himself what is better.

![Alessandro Pungitore](https://c.mql5.com/avatar/avatar_na2.png)

**[Alessandro Pungitore](https://www.mql5.com/en/users/luxtrader82)**
\|
7 Apr 2023 at 17:34

Thank you for the article and indicator


![MQL5 Wizard: Placing Orders, Stop-Losses and Take Profits on Calculated Prices. Standard Library Extension](https://c.mql5.com/2/10/ava.png)[MQL5 Wizard: Placing Orders, Stop-Losses and Take Profits on Calculated Prices. Standard Library Extension](https://www.mql5.com/en/articles/987)

This article describes the MQL5 Standard Library extension, which allows to create Expert Advisors, place orders, Stop Losses and Take Profits using the MQL5 Wizard by the prices received from included modules. This approach does not apply any additional restrictions on the number of modules and does not cause conflicts in their joint work.

![MQL5 Programming Basics: Global Variables of the Terminal](https://c.mql5.com/2/12/MQL5_Basics_Global_variables_terminal_MetaTrader5.png)[MQL5 Programming Basics: Global Variables of the Terminal](https://www.mql5.com/en/articles/1210)

This article highlights object-oriented capabilities of the MQL5 language for creating objects facilitating work with global variables of the terminal. As a practical example I consider a case when global variables are used as control points for implementation of program stages.

![Random Forests Predict Trends](https://c.mql5.com/2/11/Random_Forest_MetaTrader5.png)[Random Forests Predict Trends](https://www.mql5.com/en/articles/1165)

This article considers using the Rattle package for automatic search of patterns for predicting long and short positions of currency pairs on Forex. This article can be useful both for novice and experienced traders.

![Why Virtual Hosting On The MetaTrader 4 And MetaTrader 5 Is Better Than Usual VPS](https://c.mql5.com/2/11/Virtual_hosting.png)[Why Virtual Hosting On The MetaTrader 4 And MetaTrader 5 Is Better Than Usual VPS](https://www.mql5.com/en/articles/1171)

The Virtual Hosting Cloud network was developed specially for MetaTrader 4 and MetaTrader 5 and has all the advantages of a native solution. Get the benefit of our free 24 hours offer - test out a virtual server right now.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/1208&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083009046870496040)

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