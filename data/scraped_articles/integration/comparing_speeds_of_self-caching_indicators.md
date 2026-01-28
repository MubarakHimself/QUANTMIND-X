---
title: Comparing speeds of self-caching indicators
url: https://www.mql5.com/en/articles/4388
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:15:59.524525
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/4388&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071704018637499613)

MetaTrader 5 / Integration


### Introduction

Suppose that we have suddenly got bored with the classic MQL5 access to indicators. Let's compare the access speed with alternative options. For example, we can compare it with access to indicators in MQL4 style with and without caching. Ideas about MQL4-style access have been derived from the article ["LifeHack for traders: Fast food made of indicators"](https://www.mql5.com/en/articles/4318) and improved along the way.

### Analyzing MQL5 numbering of the indicator handles

Assume that the terminal features the consecutive numbering of the indicator handles starting with zero. To check this assumption, let's create a simple Expert Advisor **iMACD and IndicatorRelease.mq5** — it creates several indicator handles, immediately prints them and accesses them regularly in OnTick():

```
//+------------------------------------------------------------------+
//|                                   iMACD and IndicatorRelease.mq5 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.003"
//--- input parameter
input int   count=6;   // Count MACD indicators

int    handles_array[]; // array for storing the handles of the iMACD indicators
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   int array_resize=ArrayResize(handles_array,count);
   if(array_resize==-1)
     {
      Print("ArrayResize error# ",GetLastError());
      return(INIT_FAILED);
     }
   if(array_resize!=count)
     {
      Print("ArrayResize != \"Count MACD indicators\"");
      return(INIT_FAILED);
     }
   ArrayInitialize(handles_array,0);
   for(int i=0;i<count;i++)
     {
      handles_array[i]=CreateHandleMACD(12+i);
      //--- if the handle is not created
      if(handles_array[i]==INVALID_HANDLE)
        {
         //--- tell about the failure and output the error code
         PrintFormat("Failed to create handle of the iMACD indicator for the symbol %s/%s, error code %d",
                     Symbol(),
                     EnumToString(Period()),
                     GetLastError());
         //--- the indicator is stopped early
         return(INIT_FAILED);
        }
      Print("ChartID: ",ChartID(),": ",Symbol(),",",StringSubstr(EnumToString(Period()),7),
            ", create handle iMACD (",handles_array[i],")");
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Comment("");
   for(int i=0;i<count;i++)
     {
      Print("ChartID: ",ChartID(),": ",Symbol(),",",StringSubstr(EnumToString(Period()),7),
            ", remove handle iMACD (",handles_array[i],"): ",IndicatorRelease(handles_array[i]));
     }
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   string text="";
   for(int i=0;i<count;i++)
     {
      double macd_main_1=iMACDGet(handles_array[i],MAIN_LINE,1);
      if(i<15)
        {
         text+="\n"+"ChartID: "+IntegerToString(ChartID())+": "+Symbol()+
               ", MACD#"+IntegerToString(i)+" "+DoubleToString(macd_main_1,Digits()+1);
         Comment(text);
        }
      else if(i==15)
        {
         text+="\n"+"only the first 15 indicators are displayed ...";
         Comment(text);
        }
     }
  }
//+------------------------------------------------------------------+
//| Get value of buffers for the iMACD                               |
//|  the buffer numbers are the following:                           |
//|   0 - MAIN_LINE, 1 - SIGNAL_LINE                                 |
//+------------------------------------------------------------------+
double iMACDGet(const int handle_iMACD,const int buffer,const int index)
  {
   double MACD[1];
//--- reset error code
   ResetLastError();
//--- fill a part of the iMACDBuffer array with values from the indicator buffer that has 0 index
   if(CopyBuffer(handle_iMACD,buffer,index,1,MACD)<0)
     {
      //--- if the copying fails, tell the error code
      PrintFormat("Failed to copy data from the iMACD indicator, error code %d",GetLastError());
      //--- quit with zero result - it means that the indicator is considered as not calculated
      return(0.0);
     }
   return(MACD[0]);
  }
//+------------------------------------------------------------------+
//| Create handle MACD                                               |
//+------------------------------------------------------------------+
int CreateHandleMACD(const int fast_ema_period)
  {
//--- create handle of the indicator iMACD
   return(iMACD(Symbol(),Period(),fast_ema_period,52,9,PRICE_CLOSE));
  }
//+------------------------------------------------------------------+
```

**Experiment 1**

Source data: the terminal has open AUDJPY M15, USDJPY M15 and EURUSD M15 charts with no indicators and no EAs. **Count MACD indicators** parameter of **iMACD and IndicatorRelease.mq5** is 6.

Attach **iMACD and IndicatorRelease.mq5** to AUDJPY M15 (ChartID 131571247244850509) immediately after restarting the terminal:

```
2018.02.16 09:36:30.240 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (10)
2018.02.16 09:36:30.240 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (11)
2018.02.16 09:36:30.240 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (12)
2018.02.16 09:36:30.240 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (13)
2018.02.16 09:36:30.240 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (14)
2018.02.16 09:36:30.240 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (15)
```

We can see that handles numbering starts from 10 rather than 0.

**Experiment 2**

Source data: **iMACD and IndicatorRelease.mq5** is launched on AUDJPY M15, **Count MACD indicators** is 6.

Attach **iMACD and IndicatorRelease.mq5** to USDJPY, M15 (ChartID 131571247244850510):

```
2018.02.16 09:37:32.118 iMACD and IndicatorRelease (USDJPY,M15) ChartID: 131571247244850510: USDJPY,M15, create handle iMACD (10)
2018.02.16 09:37:32.118 iMACD and IndicatorRelease (USDJPY,M15) ChartID: 131571247244850510: USDJPY,M15, create handle iMACD (11)
2018.02.16 09:37:32.118 iMACD and IndicatorRelease (USDJPY,M15) ChartID: 131571247244850510: USDJPY,M15, create handle iMACD (12)
2018.02.16 09:37:32.118 iMACD and IndicatorRelease (USDJPY,M15) ChartID: 131571247244850510: USDJPY,M15, create handle iMACD (13)
2018.02.16 09:37:32.118 iMACD and IndicatorRelease (USDJPY,M15) ChartID: 131571247244850510: USDJPY,M15, create handle iMACD (14)
2018.02.16 09:37:32.118 iMACD and IndicatorRelease (USDJPY,M15) ChartID: 131571247244850510: USDJPY,M15, create handle iMACD (15)
```

We can see that handles numbering on the chart (USDJPY M15) also starts from 10, not 0.

Conclusion: the numbering of indicator handles in the terminal (the one provided to a user) is NOT consecutive and does NOT start with zero.

**Experiment 3**

Two identical charts AUDJPY, M15 (ChartID 131571247244850509) and AUDJPY, M15 (ChartID 131571247244850510). Each has **iMACD and IndicatorRelease.mq5** with **Count MACD indicators** equal to 6.

The non-consecutive numbering of created indicator handles confirms that MQL5 maintains internal accounting for them (counter for each unique handle). To make sure of this, let's comment out the period expansion:

```
int OnInit()
  {
***
   ArrayInitialize(handles_array,0);
   for(int i=0;i<count;i++)
     {
      handles_array[i]=CreateHandleMACD(12/*+i*/);
      //--- if the handle is not created
```

Thus, we try to create several MACD indicator handles with exactly the same settings.

Remove the charts left after Experiments 1 and 2 and launch **iMACD and IndicatorRelease.mq5** on AUDJPY, M15 (ChartID 131571247244850509):

```
2018.02.18 07:53:13.600 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:13.600 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:13.600 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:13.600 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:13.600 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:13.600 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, create handle iMACD (10)
```

As we can see, the same handle is returned as a response to creating absolutely identical indicators.

Attach **iMACD and IndicatorRelease.mq5** EA (also with the period extension commented out) on AUDJPY, M15 (ChartID 131571247244850510):

```
2018.02.18 07:53:20.218 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:20.218 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:20.218 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:20.218 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:20.218 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, create handle iMACD (10)
2018.02.18 07:53:20.218 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, create handle iMACD (10)
```

The same handle is returned again. Are the "10" handles on the first and second charts one and the same or two different handles? To check this, remove the EA from the charts (as you remember, the EA passes the array of handles in OnDeinit() and removes each one using [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease)).

```
2018.02.18 07:53:26.716 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, remove handle iMACD (10): true
2018.02.18 07:53:26.716 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, remove handle iMACD (10): false
2018.02.18 07:53:26.716 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, remove handle iMACD (10): false
2018.02.18 07:53:26.716 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, remove handle iMACD (10): false
2018.02.18 07:53:26.716 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, remove handle iMACD (10): false
2018.02.18 07:53:26.716 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850509: AUDJPY,M15, remove handle iMACD (10): false

2018.02.18 07:53:36.116 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, remove handle iMACD (10): true
2018.02.18 07:53:36.117 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, remove handle iMACD (10): false
2018.02.18 07:53:36.117 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, remove handle iMACD (10): false
2018.02.18 07:53:36.117 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, remove handle iMACD (10): false
2018.02.18 07:53:36.117 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, remove handle iMACD (10): false
2018.02.18 07:53:36.117 iMACD and IndicatorRelease (AUDJPY,M15) ChartID: 131571247244850510: AUDJPY,M15, remove handle iMACD (10): false
```

The result has turned outto be expected if we consider [Program Running](https://www.mql5.com/en/docs/runtime/running) documentation section:

The EA is executed in its own thread, there are as many threads of execution as there are EAs

This means that if two EAs on the same charts (same symbol and timeframe) create indicators with the same inputs, MQL5 identifies them as **two different handles** in its internal accounting.

**General conclusion concerning the development of indicators in EAs**

The numbering of indicator handles in the terminal (the one provided to a user) is NOT consecutive and does NOT start with zero, while in its internal handle accounting, MQL5 considers:

- technical indicator function (iMA, iAC, iMACD, iIchimoku, etc.);
- indicator inputs;
- symbol the indicator is created on;
- timeframe the indicator is created on;
- ChartID of a chart the EA works on.

### Does caching handles have a point?

Initial data (timeframe, symbol, tested time interval and tick generation type) are as follows:

![Cache test Settings](https://c.mql5.com/2/31/Cache_test_Settings__1.png)

Fig. 1. Settings

Tests with access to indicators in MQL4 style (with and without handle caching) are performed with the help of **Cache test.mq5** EA, while tests featuring MQL5-style access are conducted using **MQL5 test.mq5**:

```
//+------------------------------------------------------------------+
//|                                                    MQL5 test.mq5 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.000"
//--- input parameters
input bool     UseOneIndicator=false;  // Use indicator: "false" -> 9 indicators, "true" - 1 indicator
//---
int            arr_handle_iMACD[];     // array for storing the handles of the iMACD indicators
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(UseOneIndicator)
      ArrayResize(arr_handle_iMACD,1);
   else
      ArrayResize(arr_handle_iMACD,9);
   if(!CreateHandle(arr_handle_iMACD))
      return(INIT_FAILED);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   int arr_size=ArraySize(arr_handle_iMACD);
   for(int i=0;i<arr_size;i++)
     {
      double macd_main_30=iMACDGet(arr_handle_iMACD[i],MAIN_LINE,0);
     }
  }
//+------------------------------------------------------------------+
//| CreateHandle                                                     |
//+------------------------------------------------------------------+
bool CreateHandle(int &arr_handles[])
  {
   int arr_size=ArraySize(arr_handles);
   for(int i=0;i<arr_size;i++)
     {
      int fast_ema_repiod=30+10*i;
      //--- create handle of the indicator iMACD
      arr_handles[i]=iMACD(NULL,0,fast_ema_repiod,26,9,PRICE_CLOSE);
      //--- if the handle is not created
      if(arr_handles[i]==INVALID_HANDLE)
        {
         //--- tell about the failure and output the error code
         PrintFormat("Failed to create handle of the iMACD indicator for the symbol %s/%s, error code %d",
                     Symbol(),
                     EnumToString(Period()),
                     GetLastError());
         //--- the indicator is stopped early
         return(false);
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Get value of buffers for the iMACD                               |
//|  the buffer numbers are the following:                           |
//|   0 - MAIN_LINE, 1 - SIGNAL_LINE                                 |
//+------------------------------------------------------------------+
double iMACDGet(const int handle_iMACD,const int buffer,const int index)
  {
   double MACD[1];
//--- reset error code
   ResetLastError();
//--- fill a part of the iMACDBuffer array with values from the indicator buffer that has 0 index
   if(CopyBuffer(handle_iMACD,buffer,index,1,MACD)<0)
     {
      //--- if the copying fails, tell the error code
      PrintFormat("Failed to copy data from the iMACD indicator, error code %d",GetLastError());
      //--- quit with zero result - it means that the indicator is considered as not calculated
      return(0.0);
     }
   return(MACD[0]);
  }
//+------------------------------------------------------------------+
```

**MQL5 test.mq5** EA parameter:

![MQL5 test 1](https://c.mql5.com/2/31/MQL5_test_1.png)

Fig. 2. MQL5 test.mq5. Nine indicators

**Cache test.mq5** EA parameters:

- **Use Timer** ("0" -> off timer) — use the timer (0 — not used).
- **Use indicator** ("false" -> 9 indicators, "true" - 1 indicator) — number of surveyed indicators (1 or 9).

![Cache test 1](https://c.mql5.com/2/31/Cache_test_1__2.png)

Fig. 3. Cache test.mq5. No timer, nine indicators

**IndicatorsMQL4.mq** file is used to measure "MQL4 style without handle caching". The file is connected using **SimpleCallMQL4.mqh** (see the article ["LifeHack for traders: Blending ForEach with defines (#define)"](https://www.mql5.com/en/articles/4332) ).

```
#include <SimpleCall\SimpleCallMQL4.mqh> // for tests without caching of the handles
//#include <SimpleCall\SimpleCallMQL4Caching.mqh> // for tests with caching of the handles
//#include <SimpleCall\SimpleCallString.mqh> // for tests with string
```

To measure "MQL4 style with handle caching", the handle caching code from the post [#113](https://www.mql5.com/ru/forum/225641/page12#comment_6408259) is added to **IndicatorsMQL4.mqh** (for MACD only, other functions are deleted). The file is saved as **IndicatorsMQL4Caching.mqh** — it is connected by **SimpleCallCaching.mqh**:

```
//#include <SimpleCall\SimpleCallMQL4.mqh> // for tests without caching of the handles
#include <SimpleCall\SimpleCallMQL4Caching.mqh> // for tests with caching of the handles
//#include <SimpleCall\SimpleCallString.mqh> // for tests with string
```

Results of comparing access styles to the nine indicators (settings are provided in Fig. 1):

![MQL5 vs MQL4 9 indicators](https://c.mql5.com/2/32/MQL5_vs_MQL4_9_indicators__1.png)

Fig. 4. Time spent for accessing the nine indicators

When comparing the results, please note that the test EA has considerably complicated the task:

- data are obtained from NINE indicators simultaneously;
- indicators are accessed AT EACH tick;
- M1 timeframe — 26 169 180 ticks and 370 355 bars were generated.

Now let's conduct a test: call only one indicator (for both EAs, **MQL5 test.mq5** and **Cache test.mq5**, **Use indicator...** parameter is "true", while for **Cache test.mq5**, Use Timer is "0")

![MQL5 vs MQL4 1 indicator](https://c.mql5.com/2/32/MQL5_vs_MQL4_1_indicator__1.png)

Fig. 5. Time spent for accessing one indicator

**Conclusion**

MQL4 style with handle caching provides an advantage as compared to MQL4 style without handle caching. However, MQL4 style loses completely to MQL5 one.

**No handle validity control**

Now we should mention the huge drawback of using handles caching: It provides no check for the existence of the handle in the user's cache. In other words, the case of deleted indicator handle is not processed in any way.

Let's consider the following situation: we work with indicators in MQL4 style and cache the handles. After the first access from the EA:

```
   double macd_main_30=iMACD(NULL,0,30,26,9,PRICE_CLOSE,MODE_MAIN,0);
```

the handle is stored in the user cache (this can be an array of structures or an array of strings). After that, all subsequent accesses from the EA

```
   double macd_main_30=iMACD(NULL,0,30,26,9,PRICE_CLOSE,MODE_MAIN,0);
```

are not passed to MQL5 core. Instead, the indicator values by the handle taken from the cache are returned. Now, delete the handle in OnTimer() — suppose that we know it is equal to "10". As a test, let's use **Cache test.mq5** file, to which **SimpleCallMQL4Caching.mqh** file should be included:

```
//#include <SimpleCall\SimpleCallMQL4.mqh> // for tests without caching of the handles
#include <SimpleCall\SimpleCallMQL4Caching.mqh> // for tests with caching of the handles
//#include <SimpleCall\SimpleCallString.mqh> // for tests with string
```

Make sure to set the timer (here, the timer is set for six seconds, we gain access to one indicator)

![Cache test 2](https://c.mql5.com/2/31/Cache_test_2__2.png)

Fig. 6. Test settings with the handle removal

After the very first OnTimer() entry

```
OnTimer, IndicatorRelease(10)=true
iMACD: CopyBuffer error=4807
iMACD: CopyBuffer error=4807
iMACD: CopyBuffer error=4807
iMACD: CopyBuffer error=4807
```

we get the error 4807:

| ERR\_INDICATOR\_WRONG\_HANDLE | 4807 | Invalid indicator handle |

This means the indicator handle validity control is absent.

### Caching the indicator handles. How it works

The general principle of caching the indicator handles is as follows:

- create a custom handles cache;
- when requesting data from the indicator, check if the handle has already been created at the requested settings (symbol, timeframe, averaging period, etc.):
  - if it already exists in the custom cache, return data on it from the indicator;
  - if no such handle exists yet, create it, save it in the cache and return data on it from the indicator.

**Option 1: Array of structures**

Implementation is performed in **IndicatorsMQL4Caching.mqh** (connected to **Cache test.mq5** using **SimpleCallMQL4Caching.mqh**).

In **Cache test.mq5**, include **SimpleCallMQL4Caching** **.mqh**:

```
//#include <SimpleCall\SimpleCallMQL4.mqh> // for tests without caching of the handles
#include <SimpleCall\SimpleCallMQL4Caching.mqh> // for tests with caching of the handles
//#include <SimpleCall\SimpleCallString.mqh> // for tests with string
```

First, let's have a look at a large code block inserted to the file and iMACD function:

```
...
//+------------------------------------------------------------------+
//| Struct CHandle                                                   |
//+------------------------------------------------------------------+
template<typename T>
struct SHandle
  {
private:
   int               Handle;
   T                 Inputs;

public:
   //+------------------------------------------------------------------+
   //| A constructor with an initialization list                        |
   //+------------------------------------------------------------------+
                     SHandle() : Handle(INVALID_HANDLE)
     {
     }
   //+------------------------------------------------------------------+
   //| Operation Overloading "=="                                       |
   //+------------------------------------------------------------------+
   bool operator==(const T &Inputs2) const
     {
      return(this.Inputs == Inputs2);
     }
   //+------------------------------------------------------------------+
   //| Operation Overloading "="                                        |
   //+------------------------------------------------------------------+
   void operator=(const T &Inputs2)
     {
      this.Inputs=Inputs2;
     }
   //+------------------------------------------------------------------+
   //| SHandle::GetHandle                                               |
   //+------------------------------------------------------------------+
   int GetHandle()
     {
      return((this.Handle != INVALID_HANDLE) ? this.Handle : (this.Handle = this.Inputs.GetHandle()));
     }
  };
//+------------------------------------------------------------------+
//| Get Handle                                                       |
//+------------------------------------------------------------------+
template<typename T>
int GetHandle(SHandle<T>&Handles[],const T &Inputs)
  {
   const int Size=ArraySize(Handles);

   for(int i=0; i<Size; i++)
      if(Handles[i]==Inputs)
         return(Handles[i].GetHandle());

   ArrayResize(Handles,Size+1);
   Handles[Size]=Inputs;

   return(Handles[Size].GetHandle());
  }
//+------------------------------------------------------------------+
//| Struct Macd                                                      |
//+------------------------------------------------------------------+
struct SMacd
  {
   string            symbol;
   ENUM_TIMEFRAMES   period;
   int               fast_ema_period;
   int               slow_ema_period;
   int               signal_period;
   ENUM_APPLIED_PRICE applied_price;
   //+------------------------------------------------------------------+
   //| An empty default constructor                                     |
   //+------------------------------------------------------------------+
                     SMacd(void)
     {
     }
   //+------------------------------------------------------------------+
   //| A constructor with an initialization list                        |
   //+------------------------------------------------------------------+
                     SMacd(const string             &isymbol,
                                             const ENUM_TIMEFRAMES    &iperiod,
                                             const int                &ifast_ema_period,
                                             const int                &islow_ema_period,
                                             const int                &isignal_period,
                                             const ENUM_APPLIED_PRICE &iapplied_price) :
                                             symbol((isymbol== NULL)||(isymbol == "") ? Symbol() : isymbol),
                                             period(iperiod == PERIOD_CURRENT ? Period() : iperiod),
                                             fast_ema_period(ifast_ema_period),
                                             slow_ema_period(islow_ema_period),
                                             signal_period(isignal_period),
                                             applied_price(iapplied_price)
     {
     }
   //+------------------------------------------------------------------+
   //| SMacd::GetHandle                                                 |
   //+------------------------------------------------------------------+
   int GetHandle(void) const
     {
      return(iMACD(this.symbol, this.period, this.fast_ema_period, this.slow_ema_period, this.signal_period, this.applied_price));
     }
   //+------------------------------------------------------------------+
   //| Operation Overloading "=="                                       |
   //+------------------------------------------------------------------+
   bool operator==(const SMacd &Inputs) const
     {
      return((this.symbol == Inputs.symbol) &&
             (this.period == Inputs.period) &&
             (this.fast_ema_period == Inputs.fast_ema_period) &&
             (this.slow_ema_period == Inputs.slow_ema_period) &&
             (this.signal_period == Inputs.signal_period) &&
             (this.applied_price == Inputs.applied_price));
     }
  };
//+------------------------------------------------------------------+
//| iMACD2 function in MQL4 notation                                 |
//|   The buffer numbers are the following:                          |
//|      MQL4 0 - MODE_MAIN, 1 - MODE_SIGNAL                         |
//|      MQL5 0 - MAIN_LINE, 1 - SIGNAL_LINE                         |
//+------------------------------------------------------------------+
int iMACD2(const string             symbol,
           const ENUM_TIMEFRAMES    period,
           const int                fast_ema_period,
           const int                slow_ema_period,
           const int                signal_period,
           const ENUM_APPLIED_PRICE applied_price)
  {
   static SHandle<SMacd>Handles[];
   const SMacd Inputs(symbol,period,fast_ema_period,slow_ema_period,signal_period,applied_price);

   return(GetHandle(Handles, Inputs));
  }
//+------------------------------------------------------------------+
//| iAC function in MQL4 notation                                    |
...
//+------------------------------------------------------------------+
//| iMACD function in MQL4 notation                                  |
//|   The buffer numbers are the following:                          |
//|      MQL4 0 - MODE_MAIN, 1 - MODE_SIGNAL                         |
//|      MQL5 0 - MAIN_LINE, 1 - SIGNAL_LINE                         |
//+------------------------------------------------------------------+
double   iMACD(
               string                     symbol,              // symbol name
               ENUM_TIMEFRAMES            timeframe,           // timeframe
               int                        fast_ema_period,     // period for Fast average calculation
               int                        slow_ema_period,     // period for Slow average calculation
               int                        signal_period,       // period for their difference averaging
               ENUM_APPLIED_PRICE         applied_price,       // type of price or handle
               int                        buffer,              // buffer
               int                        shift                // shift
               )
  {
   double result=NaN;
//---
   int handle=iMACD2(symbol,timeframe,fast_ema_period,slow_ema_period,signal_period,applied_price);
   if(handle==INVALID_HANDLE)
...
```

Let's describe its work. First, there is a request for data from MACD:

```
   double macd_main_30=iMACD(NULL,0,30,26,9,PRICE_CLOSE,MODE_MAIN,0);
```

Then we get into iMACD function and go to iMACD2:

```
//+------------------------------------------------------------------+
//| iMACD2 function in MQL4 notation                                 |
//|   The buffer numbers are the following:                          |
//|      MQL4 0 - MODE_MAIN, 1 - MODE_SIGNAL                         |
//|      MQL5 0 - MAIN_LINE, 1 - SIGNAL_LINE                         |
//+------------------------------------------------------------------+
int iMACD2(const string             symbol,
           const ENUM_TIMEFRAMES    period,
           const int                fast_ema_period,
           const int                slow_ema_period,
           const int                signal_period,
           const ENUM_APPLIED_PRICE applied_price)
  {
   static SHandle<SMacd>Handles[];
   const SMacd Inputs(symbol,period,fast_ema_period,slow_ema_period,signal_period,applied_price);

   return(GetHandle(Handles, Inputs));
  }
```

**Handles**\[\] static array with SMacd type is declared here (it is created during the first entry and is not re-created at subsequent entries). Also, **Inputs** object with SMacd type is created and initialized with parameters at once.

After that, use the links to pass **Handles**\[\] array and **Inputs** object to **GetHandle** function (not to SHandle::GetHandle and SMacd::GetHandle) **:**

```
//+------------------------------------------------------------------+
//| Get Handle                                                       |
//+------------------------------------------------------------------+
template<typename T>
int GetHandle(SHandle<T>&Handles[],const T &Inputs)
  {
   const int Size=ArraySize(Handles);

   for(int i=0; i<Size; i++)
      if(Handles[i]==Inputs)
         return(Handles[i].GetHandle());

   ArrayResize(Handles,Size+1);
   Handles[Size]=Inputs;
   return(Handles[Size].GetHandle());
  }
```

In this function, return the found indicator handle in the array or, if the handle is not found, receive itin SHandle::GetHandle.

But since this is the first access and there is no such handle yet,

```
   //+------------------------------------------------------------------+
   //| SHandle::GetHandle                                               |
   //+------------------------------------------------------------------+
   int GetHandle()
     {
      return((this.Handle != INVALID_HANDLE) ? this.Handle : (this.Handle = this.Inputs.GetHandle()));
     }
```

create it in SMacd::GetHandle:

```
   //+------------------------------------------------------------------+
   //| SMacd::GetHandle                                                 |
   //+------------------------------------------------------------------+
   int GetHandle(void) const
     {
      return(iMACD(this.symbol, this.period, this.fast_ema_period, this.slow_ema_period, this.signal_period, this.applied_price));
     }
```

**Option 2: String array**

Implementation is performed in **IndicatorsMQL4String.mqh** file (connected to **Cache test.mq5** using **SimpleCallString.mqh**).

In **Cache test.mq5** EA, include **SimpleCallString.mqh**:

```
//#include <SimpleCall\SimpleCallMQL4.mqh> // for tests without caching of the handles
//#include <SimpleCall\SimpleCallMQL4Caching.mqh> // for tests with caching of the handles
#include <SimpleCall\SimpleCallString.mqh> // for tests with string
```

Working with strings is terribly expensive in terms of speed. We will see that a bit later. So, the idea of saving parameters as a string looks as follows:

```
   string Hashes[];
   static int Handles[];
   string hash=((symbol==NULL) || (symbol=="") ? Symbol() : symbol)+
               (string)(timeframe==PERIOD_CURRENT ? Period() : timeframe)+
               (string)(fast_ema_period)+
               (string)(slow_ema_period)+
               (string)(signal_period)+
               (string)(applied_price);
```

We will access iMACD from the EA with parameters provided above, in Fig. 1.

| NN | Code | Time |
| --- | --- | --- |
| 1 | ```<br>//--- NN2<br>//static string Hashes[];<br>//static int Handles[];<br>//string hash=((symbol==NULL) || (symbol=="") ? Symbol() : symbol)+<br>//            (string)(timeframe==PERIOD_CURRENT ? Period() : timeframe)+<br>//            (string)(fast_ema_period)+<br>//            (string)(slow_ema_period)+<br>//            (string)(signal_period)+<br>//            (string)(applied_price);<br>//--- NN3<br>//static string Hashes[];<br>//static int Handles[];<br>//string hash="";<br>//StringConcatenate(hash,<br>//                  ((symbol==NULL) || (symbol=="") ? Symbol() : symbol),<br>//                  (timeframe==PERIOD_CURRENT ? Period() : timeframe),<br>//                  fast_ema_period,<br>//                  slow_ema_period,<br>//                  signal_period,<br>//                  applied_price);<br>``` | 0:01:40.953 |
| 2 | ```<br>//--- NN2<br>   static string Hashes[];<br>   static int Handles[];<br>   string hash=((symbol==NULL) || (symbol=="") ? Symbol() : symbol)+<br>               (string)(timeframe==PERIOD_CURRENT ? Period() : timeframe)+<br>               (string)(fast_ema_period)+<br>               (string)(slow_ema_period)+<br>               (string)(signal_period)+<br>               (string)(applied_price);<br>//--- NN3<br>//static string Hashes[];<br>//static int Handles[];<br>//string hash="";<br>//StringConcatenate(hash,<br>//                  ((symbol==NULL) || (symbol=="") ? Symbol() : symbol),<br>//                  (timeframe==PERIOD_CURRENT ? Period() : timeframe),<br>//                  fast_ema_period,<br>//                  slow_ema_period,<br>//                  signal_period,<br>//                  applied_price);<br>``` | 0:05:20.953 |
| 3 | ```<br>//--- NN2<br>//static string Hashes[];<br>//static int Handles[];<br>//string hash=((symbol==NULL) || (symbol=="") ? Symbol() : symbol)+<br>//            (string)(timeframe==PERIOD_CURRENT ? Period() : timeframe)+<br>//            (string)(fast_ema_period)+<br>//            (string)(slow_ema_period)+<br>//            (string)(signal_period)+<br>//            (string)(applied_price);<br>//--- NN3<br>   static string Hashes[];<br>   static int Handles[];<br>   string hash="";<br>   StringConcatenate(hash,<br>                     ((symbol==NULL) || (symbol=="") ? Symbol() : symbol),<br>                     (timeframe==PERIOD_CURRENT ? Period() : timeframe),<br>                     fast_ema_period,<br>                     slow_ema_period,<br>                     signal_period,<br>                     applied_price);<br>``` | 0:04:12.672 |

Test 1 is a benchmark test with MQL4-style access to indicators without working with strings. In test 2, we already work with strings and the string is formed using "+". In test 3, the string is formed using [StringConcatenate](https://www.mql5.com/en/docs/strings/stringconcatenate).

According to the time measurements, it is clear that, although [StringConcatenate](https://www.mql5.com/en/docs/strings/stringconcatenate) gives 21% time gain in comparison with test 2, the overall performance is still 2.5 times less than in test 1.

Therefore, the idea of saving the indicator handles as strings can be discarded.

**Option 3 — class caching handles**( **iIndicators.mqh** class is connected to **Cache test.mq5** EA using **SimpleCallMQL4CachingCiIndicators.mqh**).

In **Cache test.mq5** EA, we include **SimpleCallMQL4CachingCiIndicators.mqh**:

```
//#include <SimpleCall\SimpleCallMQL4.mqh> // for tests without caching of the handles
//#include <SimpleCall\SimpleCallMQL4Caching.mqh> // for tests with caching of the handles
//#include <SimpleCall\SimpleCallString.mqh> // for tests with string
#include <SimpleCall\SimpleCallMQL4CachingCiIndicators.mqh>
```

Static object of CHandle class is created (inside the appropriate MQL4-style function) for each indicator. It serves as CiIndicators class object storage — class containing the indicator parameters and settings.

![Scheme](https://c.mql5.com/2/31/Scheme__1.png)

Fig. 7. Structure

**CiIndicators** class is based on five 'private' variables:

```
//+------------------------------------------------------------------+
//| Class iIndicators                                                |
//+------------------------------------------------------------------+
class CiIndicators
  {
private:
   string            m_symbol;                        // symbol name
   ENUM_TIMEFRAMES   m_period;                        // timeframe
   ENUM_INDICATOR    m_indicator_type;                // indicator type from the enumeration ENUM_INDICATOR
   int               m_parameters_cnt;                // number of parameters
   MqlParam          m_parameters_array[];            // array of parameters

public:
```

It is completely corresponds to [IndicatorCreate](https://www.mql5.com/en/docs/series/indicatorcreate) function variables. This is not done for nothing, since we receive the indicator handle via IndicatorCreate.

**CHandle** class is built using two arrays:

```
//+------------------------------------------------------------------+
//| Class CHandle                                                    |
//+------------------------------------------------------------------+
class CHandle
  {
private:
   int               m_handle[];
   CiIndicators      m_indicators[];

public:
```

**m\_handle** array contains created indicator handles, while **m\_indicators** array is **CiIndicators** class array.

Code of working with **CiIndicators** and **CHandle** classes looks as follows using MACD as an example:

```
//+------------------------------------------------------------------+
//| iMACD function in MQL4 notation                                  |
//|   The buffer numbers are the following:                          |
//|      MQL4 0 - MODE_MAIN, 1 - MODE_SIGNAL                         |
//|      MQL5 0 - MAIN_LINE, 1 - SIGNAL_LINE                         |
//+------------------------------------------------------------------+
double   iMACD(
               string                     symbol,              // symbol name
               ENUM_TIMEFRAMES            timeframe,           // timeframe
               int                        fast_ema_period,     // period for Fast average calculation
               int                        slow_ema_period,     // period for Slow average calculation
               int                        signal_period,       // period for their difference averaging
               ENUM_APPLIED_PRICE         applied_price,       // type of price or handle
               int                        buffer,              // buffer
               int                        shift                // shift
               )
  {
//---
   static CHandle Handles_MACD;
//--- fill the structure with parameters of the indicator
   MqlParam pars[4];
//--- period of fast ma
   pars[0].type=TYPE_INT;
   pars[0].integer_value=fast_ema_period;
//--- period of slow ma
   pars[1].type=TYPE_INT;
   pars[1].integer_value=slow_ema_period;
//--- period of averaging of difference between the fast and the slow moving average
   pars[2].type=TYPE_INT;
   pars[2].integer_value=signal_period;
//--- type of price
   pars[3].type=TYPE_INT;
   pars[3].integer_value=applied_price;

   CiIndicators MACD_Indicator;
   MACD_Indicator.Init(Symbol(),Period(),IND_MACD,4);
   int handle=Handles_MACD.GetHandle(MACD_Indicator,Symbol(),Period(),IND_MACD,4,pars);
//---
   double result=NaN;
//---
   if(handle==INVALID_HANDLE)
     {
      Print(__FUNCTION__,": INVALID_HANDLE error=",GetLastError());
      return(result);
     }
   double val[1];
   int copied=CopyBuffer(handle,buffer,shift,1,val);
   if(copied>0)
      result=val[0];
   else
      Print(__FUNCTION__,": CopyBuffer error=",GetLastError());
   return(result);
  }
```

- **Handles\_MACD** static array of **CHandle** class is declared — it is to store generated handles and MACD parameters.
- **MACD\_Indicator** object of **CiIndicators** class is created and initialized.
- The indicator handle is created (or passed if it has already been created for such parameters) in Handles\_MACD::GetHandle function.

**CiIndicators** **.mqh** class operation time with MQL4-style access and handles caching took 2 minutes and 30 seconds.

### Final graph of the access speed to the nine indicators

MQL4 style with and without caching is checked using **Cache test.mq5**, while standard MQL5 style tests are conducted using **MQL5 test.mq5**.

![MQL5 vs MQL4 9 indicators Summary chart](https://c.mql5.com/2/32/MQL5_vs_MQL4_9_indicators_Summary_chart.png)

### Conclusion

We have conducted some interesting experiments that go against the paradigm of correct MQL5 access to indicators. As a result, we have learned more about the internal mechanism of processing the handles inside the MQL5 core:

- about the handle counter;
- about caching and handle management.

The results of testing various methods of accessing the indicators showed that MQL5 access style is much faster than any MQL4 styles (both with and without handle caching).

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/4388](https://www.mql5.com/ru/articles/4388)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4388.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/4388/mql5.zip "Download MQL5.zip")(12.02 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717)
- [Gap - a profitable strategy or 50/50?](https://www.mql5.com/en/articles/5220)
- [Elder-Ray (Bulls Power and Bears Power)](https://www.mql5.com/en/articles/5014)
- [Improving Panels: Adding transparency, changing background color and inheriting from CAppDialog/CWndClient](https://www.mql5.com/en/articles/4575)
- [How to create a graphical panel of any complexity level](https://www.mql5.com/en/articles/4503)
- [LifeHack for traders: Blending ForEach with defines (#define)](https://www.mql5.com/en/articles/4332)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/242752)**
(10)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
7 Mar 2018 at 09:28

**Комбинатор:**

that's the interpretation I'm talking about

"Internal accounting" interpreted as a counter. Frankly speaking, I don't understand why the first part of the article is about handles? It seems that everything has been chewed up more than once before and presented in more convenient formulations. Up to the reasons for the possibility of executing indicators in the Terminal, where there are no [open charts](https://www.mql5.com/en/docs/chart_operations/chartopen "MQL5 documentation: ChartOpen function").

![MetaQuotes](https://c.mql5.com/avatar/2009/11/4AF883AB-83DE.jpg)

**[Renat Fatkhullin](https://www.mql5.com/en/users/renat)**
\|
7 Mar 2018 at 10:45

**Комбинатор:**

This is a completely wrong conclusion. the handle is the same and it is confirmed by the fact that the id matches.

The first true result indicates only that the handle reference count has decreased.

Yes, it's an error in the article.

In general, you should quit inventing and writing "in [MQL4 style".](https://www.mql5.com/en/articles/4318 "Article: LifeHack for trader: cooking fast food from indicators ") MQL5 is faster and more correct. It was the understanding of crutches and limitations of MQL4 that led us to create a new language and refuse compatibility in order not to pull a bad data access scheme.

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
7 Mar 2018 at 10:55

**Renat Fatkhullin:**

In general, it is necessary to invent and write "in MQL4 style". MQL5 is faster and more correct. It was the understanding of crutches and limitations of MQL4 that led us to create a new language and refuse compatibility, so as not to pull a bad data access scheme.

Go!

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/ru/forum)

[Discussion of the article "Comparing the speed of self-caching indicators"](https://www.mql5.com/ru/forum/230765#comment_6740068)

[fxsaber](https://www.mql5.com/en/users/fxsaber), 2018.03.07 08:17 pm.

In general, the TS calls indicators with calculated (rather than hard-coded) input parameters. And here you can't do without MQL4-style+cache.

I think it is not difficult to find an MT4-advisor of this level in KB. It will be impossible to convert it into what is called in the article MQL5-style.

![MetaQuotes](https://c.mql5.com/avatar/2009/11/4AF883AB-83DE.jpg)

**[Renat Fatkhullin](https://www.mql5.com/en/users/renat)**
\|
7 Mar 2018 at 10:57

**fxsaber:**

Let's go!

That's a stretch.

No argument


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
7 Mar 2018 at 10:57

**Renat Fatkhullin:**

It's a stretch.

No argument.

Continue to advocate for primitive TCs.

![Deep Neural Networks (Part V). Bayesian optimization of DNN hyperparameters](https://c.mql5.com/2/48/Deep_Neural_Networks_05.png)[Deep Neural Networks (Part V). Bayesian optimization of DNN hyperparameters](https://www.mql5.com/en/articles/4225)

The article considers the possibility to apply Bayesian optimization to hyperparameters of deep neural networks, obtained by various training variants. The classification quality of a DNN with the optimal hyperparameters in different training variants is compared. Depth of effectiveness of the DNN optimal hyperparameters has been checked in forward tests. The possible directions for improving the classification quality have been determined.

![How to create a graphical panel of any complexity level](https://c.mql5.com/2/31/graph_panel.png)[How to create a graphical panel of any complexity level](https://www.mql5.com/en/articles/4503)

The article features a detailed explanation of how to create a panel on the basis of the CAppDialog class and how to add controls to the panel. It provides the description of the panel structure and a scheme, which shows the inheritance of objects. From this article, you will also learn how events are handled and how they are delivered to dependent controls. Additional examples show how to edit panel parameters, such as the size and the background color.

![Multi-symbol balance graph in MetaTrader 5](https://c.mql5.com/2/31/MultiSymbol.png)[Multi-symbol balance graph in MetaTrader 5](https://www.mql5.com/en/articles/4430)

The article provides an example of an MQL application with its graphical interface featuring multi-symbol balance and deposit drawdown graphs based on the last test results.

![Visualizing trading strategy optimization in MetaTrader 5](https://c.mql5.com/2/31/t3b4bw8nglimc_2v6gmclew41_jdawvaf9_w1x5mnmfb_d_MetaTrader_5.png)[Visualizing trading strategy optimization in MetaTrader 5](https://www.mql5.com/en/articles/4395)

The article implements an MQL application with a graphical interface for extended visualization of the optimization process. The graphical interface applies the last version of EasyAndFast library. Many users may ask why they need graphical interfaces in MQL applications. This article demonstrates one of multiple cases where they can be useful for traders.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/4388&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071704018637499613)

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