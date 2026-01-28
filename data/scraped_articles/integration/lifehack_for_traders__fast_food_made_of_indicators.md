---
title: LifeHack for traders: Fast food made of indicators
url: https://www.mql5.com/en/articles/4318
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:16:19.744928
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/4318&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071708433863879921)

MetaTrader 5 / Integration


_If you want something forbidden really bad, then it is allowed.  Russian proverb_

### Simplicity vs Reliability

Back in 2005, in the [newly released MetaTrader 4](https://www.metaquotes.net/en/company/news/3432 "https://www.metaquotes.net/en/company/news/3294"), the simple MQL-II script language was replaced by MQL4. As funny as it may seem today, many traders met the new C-like language with hostility. There were many furious debates and accusations directed at MetaQuotes Software Corp. Critics claimed that the language was very complicated and it was impossible to master it.

Now, after 12 years, such claims seem strange but the history repeats itself. Just like in 2005, some traders declare that MQL5 is complicated for learning and developing strategies compared to MQL4. This means that the overall level of developing trading robots has grown substantially over the years thanks to the fact that the developer was not afraid to move on providing algorithmic traders with even more powerful tools of the C++ language. The new MQL5 allows programmers to check results of all operations in maximum detail (this is especially important for handling trades) and consume RAM on demand. The old MQL4 provided much less opportunities of that kind before it was improved to MQL5 level. Besides, the syntax itself was less strict.

I believe, that debates about MQL5 complexity will also pass into oblivion after a short while. But since many traders still feel nostalgic about "good old MQL4", we will try to show how familiar MQL4 functions may look if implemented in MQL5.

If you have newly switched to MQL5, then this article will be useful. First, the access to the indicator data and series is done in the usual MQL4 style. Second, this entire simplicity is implemented in MQL5. All functions are as clear as possible and perfectly suited for step-by-step debugging.

### 1\. Is it possible to work with indicator in MQL5 using MQL4 style?

The main difference in working with indicators is that in MQL4, the indicator data retrieval string is, in fact, the indicator creation command ( iMACD(NULL,0,12,26,9,PRICE\_CLOSE ) combined with request for data from the necessary indicator buffer ( MODE\_MAIN ) and index ( 1 ).

```
//+------------------------------------------------------------------+
//|                                                        iMACd.mq4 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

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
   double macd_main_1=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,1);
  }
//+------------------------------------------------------------------+
```

As a result, a single string stands for only a single step.

In MQL5, the equivalent of this code contains several steps:

- declaring the variable where the indicator handle is to be stored;

- creating and checking the indicator handle;
- separate function providing the indicator value.

```
//+------------------------------------------------------------------+
//|                                                        iMACD.mq5 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.000"

int    handle_iMACD;                         // variable for storing the handle of the iMACD indicator
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create handle of the indicator iMACD
   handle_iMACD=iMACD(Symbol(),Period(),12,26,9,PRICE_CLOSE);
//--- if the handle is not created
   if(handle_iMACD==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the iMACD indicator for the symbol %s/%s, error code %d",
                  Symbol(),
                  EnumToString(Period()),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
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

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   double macd_main_1=iMACDGet(MAIN_LINE,1);
  }
//+------------------------------------------------------------------+
//| Get value of buffers for the iMACD                               |
//|  the buffer numbers are the following:                           |
//|   0 - MAIN_LINE, 1 - SIGNAL_LINE                                 |
//+------------------------------------------------------------------+
double iMACDGet(const int buffer,const int index)
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

Let's re-write the code in MQL4 style.

Creating the indicator handle and obtaining the indicator data will be implemented in a single function:

```
//+------------------------------------------------------------------+
//| iMACD function in MQL4 notation                                  |
//+------------------------------------------------------------------+
double iMACD(
             string              symbol,              // symbol name
             ENUM_TIMEFRAMES     period,              // period
             int                 fast_ema_period,     // period for Fast average calculation
             int                 slow_ema_period,     // period for Slow average calculation
             int                 signal_period,       // period for their difference averaging
             ENUM_APPLIED_PRICE  applied_price,       // type of price or handle
             int                 buffer,              // buffer
             int                 shift                // shift
             )
  {
   double result=NULL;
//---
   int handle=iMACD(symbol,period,fast_ema_period,slow_ema_period,signal_period,
                    applied_price);
   double val[1];
   int copied=CopyBuffer(handle,buffer,shift,1,val);
   if(copied>0)
      result=val[0];
   return(result);
  }
```

NOTE! After writing the function, we will create the indicator handle ON EVERY tick. You may say that the documentation does not recommend such "creativity". Let's have a look at the [Technical Indicator Functions](https://www.mql5.com/en/docs/indicators) section:

You can't refer to the indicator data right after it has been created, because calculation of indicator values requires some time. So it's better to create indicator handles in OnInit().

So why does this code work and not consume memory? The answer is in the same section:

Note. Repeated call of the indicator function with the same parameters within one mql5-program does not lead to a multiple increase of the reference counter; the counter will be increased only once by 1. However, it's recommended to get the indicators handles in function [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) or in the class constructor, and further use these handles in other functions. The reference counter decreases when a mql5-program is deinitialized.

In other words, MQL5 is optimally designed: it controls the creation of the handles and does not allow creating the same indicator with the same parameters many times. In case of repeated attempts to create a handle which is a copy of the indicator, you simply get the handle of the previously created indicator with the corresponding settings. Anyway, it is still recommended to receive the handles a single time in OnInit (). The reasons will be provided later.

Note: there is no check for the validity of the generated handle.

Now, the code that receives iMACD indicator values ​​will look like this:

```
//+------------------------------------------------------------------+
//|                                           MACD MQL4 style EA.mq5 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.000"

#define MODE_MAIN 0
#define MODE_SIGNAL 1
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

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
   double macd_main_1=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,1);
  }
//+------------------------------------------------------------------+
//| iMACD function in MQL4 notation                                  |
//+------------------------------------------------------------------+
double iMACD(
             string              symbol,              // symbol name
             ENUM_TIMEFRAMES     period,              // period
             int                 fast_ema_period,     // period for Fast average calculation
             int                 slow_ema_period,     // period for Slow average calculation
             int                 signal_period,       // period for their difference averaging
             ENUM_APPLIED_PRICE  applied_price,       // type of price or handle
             int                 buffer,              // buffer
             int                 shift                // shift
             )
  {
   double result=NULL;
//---
   int handle=iMACD(symbol,period,fast_ema_period,slow_ema_period,signal_period,
                    applied_price);
   double val[1];
   int copied=CopyBuffer(handle,buffer,shift,1,val);
   if(copied>0)
      result=val[0];
   return(result);
  }
//+------------------------------------------------------------------+
```

NOTE: Desire to access indicators in MQL4 style deprives us of the option of checking the return value, since all functions in MQL4 style return ONLY 'double' values. A possible solution will be provided in the section 1.1.

It looks pretty cumbersome so far, therefore let's implement the 'define' block and the double iMACD() function in a separate **IndicatorsMQL5.mqh** include file to be located in a separate folder "\[data folder\]\\MQL5\\Include\ **SimpleCall**". In this case, the code becomes pretty short. Please note that we include the **IndicatorsMQL5.mqh** file. This means that the names of the indicator lines should be transferred in the form of MQL5 MAIN\_LINE rather than MQL4 MODE\_MAIN when accessing the MACD:

```
//+------------------------------------------------------------------+
//|                                     MACD MQL4 style EA short.mq5 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.000"
#include <SimpleCall\IndicatorsMQL5.mqh>
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

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
   double macd_main_1=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MAIN_LINE,1);
   Comment("MACD, main buffer, index 1: ",DoubleToString(macd_main_1,Digits()+1));
  }
//+------------------------------------------------------------------+
```

I have implemented "Comment" solely for verification. You can verify the work in the tester if you launch "MACD MQL4 style EA short.mq5" in visual mode and place the cursor on the bar with the index #1:

!["MACD MQL4 style EA short.mh5" in tester](https://c.mql5.com/2/30/MACD_MQL4_style_EA_short_in_tester.png)

Fig. 1. "MACD MQL4 style EA short.mh5" in tester

#### 1.1. Some nuances when working with "IndicatorsXXXX.mqh"

**Error handling in a return value**

All indicators pass their data as double. This is an issue of sending a message to a user if it has suddenly become impossible to obtain data from the indicator. This may happen if the indicator handle is not created (for example, if a non-existent symbol is specified) or if a copy error occurred while calling CopyBuffer.

Simply passing "0.0" in case of an error is not an option since for most indicators "0.0" is a quite normal value (for example, for MACD). Returning the EMPTY\_VALUE constant (having the value of DBL\_MAX) is not an option either, since the Fractals indicator fills in the buffer indices by EMPTY\_VALUE values meaning this is not an error.

The only remaining option is to pass "not a number" — [NaN](https://en.wikipedia.org/wiki/NaN "https://en.wikipedia.org/wiki/NaN"). To achieve this, the **NaN** variable is created on a global level. The variable is initialized by a "non-number":

```
double NaN=double("nan");
//+------------------------------------------------------------------+
//| iAC function in MQL4 notation                                    |
//+------------------------------------------------------------------+
double   iAC(
             string                       symbol,              // symbol name
             ENUM_TIMEFRAMES              timeframe,           // timeframe
             int                          shift                // shift
             )
  {
   double result=NaN;
//---
   int handle=iAC(symbol,timeframe);
   if(handle==INVALID_HANDLE)
     {
      Print(__FUNCTION__,": INVALID_HANDLE error=",GetLastError());
      return(result);
     }
   double val[1];
   int copied=CopyBuffer(handle,0,shift,1,val);
   if(copied>0)
      result=val[0];
   else
      Print(__FUNCTION__,": CopyBuffer error=",GetLastError());
   return(result);
  }
```

The advantage of this approach is also that NaN is returned in case of an error, and the result of its comparison with any number will be 'false'.

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- example of NaN comparison
   double NaN=double("nan");
   double a=10.3;
   double b=-5;
   double otherNaN=double("nan");
   Print("NaN>10.3=",NaN>a);
   Print("NaN<-5=",NaN<b);
   Print("(NaN==0)=",NaN==0);
   Print("(NaN==NaN)=",NaN==otherNaN);
//--- result
   NaN>10.3=false
   NaN<-5=false
   (NaN==0)=false
   (NaN==NaN)=false
//---
  }
```

Therefore, if we want to use these functions in MQL4 style, then it is necessary to conduct trading operations (as well as any other important actions) only if the result of the comparison is true. Although in this case, I insist on checking the return value using the [MathIsValidNumber](https://www.mql5.com/en/docs/math/mathisvalidnumber) function.

**Identifiers of indicator lines in MQL4 and MQL5**

There is a compatibility issue in the part of the values ​​of constants that describe the indicator lines. For example, let's take iAlligator:

- MQL4: 1 - MODE\_GATORJAW,2 - MODE\_GATORTEETH, 3 - MODE\_GATORLIPS
- MQL5: 0 - GATORJAW\_LINE,   1 - GATORTEETH\_LINE,   2 - GATORLIPS\_LINE

The issue is that the indicator line in the **"IndicatorsXXXX.mqh"** function comes as a number. If this number, for example, is 1, then no one can say what the user meant: either they worked in MQL4 style (and had in mind 1 - MODE\_GATORJAW), or they worked in the MQL5 style (and had in mind a completely different indicator line 1 - GATORTEETH\_LINE).

In this regard, I decided to create two include files - practically twins: " **IndicatorsMQL4.mqh**" and " **IndicatorsMQL5.mqh**". Their difference is that the "IndicatorsMQL4.mqh" file understands indicator lines ONLY in MQL4 style, while the file "IndicatorsMQL5.mqh" understands indicator lines ONLY in MQL5 style. In "IndicatorsMQL4.mqh", transformation of the indicator line in the input parameter is performed directly inside the iADX, iAlligator ... functions — you cannot relocate these transformations to #define.

Let me explain the reason for this on the example of iBands and iEnvelopes:

```
//+------------------------------------------------------------------+
//| iBands function in MQL4 notation                                 |
//|   The buffer numbers are the following:                          |
//|      MQL4 0 - MODE_MAIN, 1 - MODE_UPPER, 2 - MODE_LOWER          |
//|      MQL5 0 - BASE_LINE, 1 - UPPER_BAND, 2 - LOWER_BAND          |
//+------------------------------------------------------------------+
double   iBands(
...
//+------------------------------------------------------------------+
//| iEnvelopes function in MQL4 notation                             |
//|   The buffer numbers are the following:                          |
//|      MQL4 0 - MODE_MAIN,  1 - MODE_UPPER, 2 - MODE_LOWER         | ???
//|      MQL5 0 - UPPER_LINE, 1 - LOWER_LINE,        -/-             |
//+------------------------------------------------------------------+
double   iEnvelopes(
```

In MQL4, MODE\_UPPER for Bands indicator, is transformed into 1, while for Envelopes indicator, it is transformed into 0.

### 2\. What is the memory consumption if we apply indicators in MQL4 style at each tick?

Let's compare the memory consumption of the two EAs: "iMACD.mq5" — the EA with correct access to the indicators and the "MACD MQL4 style EA short.mq5" — with access to MQL4 style indicators. The maximum number of bats in the window is set to "100 000" in the terminal settings. Create two profiles of 14 charts:

- "iMACd" profile — "iMACd.mq5" EA is set on 13 charts, all charts are of M30 timeframe;
- "MACD MQL4 style EA short" profile — "MACD MQL4 style EA short.mq5" EA is set on 13 charts.

" **Terminal memory used.mq5**" indicator is launched on the fourteenth chart. Its objective is to print [TERMINAL\_MEMORY\_USED](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus) identifier every 10 seconds.

We will compare two values: amount of RAM consumed by the terminal (task manager data) the printed [TERMINAL\_MEMORY\_USED](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus) identifier. The observation will be conducted for 10 minutes — we will see if too much memory is consumed. The main condition: after starting the terminal, do nothing in it - do not open new tabs or read the chat.

| Profile | Task manager | TERMINAL\_MEMORY\_USED | Task manager (in 10 minutes) | TERMINAL\_MEMORY\_USED (in 10 minutes) |
| --- | --- | --- | --- | --- |
| iMACd | 279.7 MB | 745 MB | 279.7 MB | 745 MB |
| MACD MQL4 style EA short | 279.9 MB | 745 MB | 280.0 MB | 745 MB |

Now, let's modify the test: after 10 minutes of work, switch the timeframes of all charts to H1.

| Profile | Task manager | TERMINAL\_MEMORY\_USED | Task manager (in 10 minutes) | TERMINAL\_MEMORY\_USED (in 10 minutes) |
| --- | --- | --- | --- | --- |
| iMACd | 398.0 MB | 869 MB | 398.3 MB | 869 MB |
| MACD MQL4 style EA short | 319.2 MB | 874 MB | 330.5 MB | 874 MB |

Summary table for clarity of memory usage:

| Profile | Task manager<br> (M30), MB | TERMINAL\_MEMORY\_USED<br> (M30), MB | Task manager<br> (H1), MB | TERMINAL\_MEMORY\_USED<br> (H1), MB |
| --- | --- | --- | --- | --- |
|  | start | in 10 minutes | start | in 10 minutes | start | in 10 minutes | start | in 10 minutes |
| iMACd | 279.7 | 279.7 | 745 | 745 | 398.0 | 869 | 398.3 | 869 |
| MACD MQL4 style EA short | 279.9 | 280.0 | 745 | 745 | 319.2 | 874 | 330.5 | 874 |

### 3\. The new life of MACD Sample.mq4 EA

Let's check the execution speed, memory consumption and \[data folder\]\\MQL4\\Experts\\MACD Sample.mq4 EA (developed in MQL5 but in MQL4 style like "MACD MQL4 style EA short.mq5") compliance with \[data folder\]\\MQL5\\Experts\\Examples\\MACD\\MACD Sample.mq5 EA.

#### 3.1. Let's change "MACD Sample.mq5" EA, so that it receives one value at a time

"MACD Sample.mq5" from the standard delivery receives two indicator values at once:

```
//+------------------------------------------------------------------+
//| main function returns true if any position processed             |
//+------------------------------------------------------------------+
bool CSampleExpert::Processing(void)
  {
//--- refresh rates
   if(!m_symbol.RefreshRates())
      return(false);
//--- refresh indicators
   if(BarsCalculated(m_handle_macd)<2 || BarsCalculated(m_handle_ema)<2)
      return(false);
   if(CopyBuffer(m_handle_macd,0,0,2,m_buff_MACD_main)  !=2 ||
      CopyBuffer(m_handle_macd,1,0,2,m_buff_MACD_signal)!=2 ||
      CopyBuffer(m_handle_ema,0,0,2,m_buff_EMA)         !=2)
      return(false);
//   m_indicators.Refresh();
//--- to simplify the coding and speed up access
//--- data are put into internal variables
   m_macd_current   =m_buff_MACD_main[0];
   m_macd_previous  =m_buff_MACD_main[1];
   m_signal_current =m_buff_MACD_signal[0];
   m_signal_previous=m_buff_MACD_signal[1];
   m_ema_current    =m_buff_EMA[0];
   m_ema_previous   =m_buff_EMA[1];
```

After that, data from arrays of dimension "2" are assigned to the variables. Why is it done this way? Regardless of whether we copy by one or two values per time, we still use [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer). However, when copying two values at once, we save one operation of writing to the array.

But "MACD Sample.mq4" EA receives one indicator value per time:

```
//--- to simplify the coding and speed up access data are put into internal variables
   MacdCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,0);
   MacdPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,1);
   SignalCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,0);
   SignalPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,1);
   MaCurrent=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,0);
   MaPrevious=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,1);
```

The MACD main line, MACD signal line and Moving Average are surveyed two times each. Therefore, "MACD Sample.mq5" should be brought to the same form. Let's call this EA version "MACD Sample One value at a time.mq5". Here is how it is changed, so that we receive one value at a time:

```
//--- refresh indicators
   if(BarsCalculated(m_handle_macd)<2 || BarsCalculated(m_handle_ema)<2)
      return(false);
//   if(CopyBuffer(m_handle_macd,0,0,2,m_buff_MACD_main)  !=2 ||
//      CopyBuffer(m_handle_macd,1,0,2,m_buff_MACD_signal)!=2 ||
//      CopyBuffer(m_handle_ema,0,0,2,m_buff_EMA)         !=2)
//      return(false);
//   m_indicators.Refresh();
//--- to simplify the coding and speed up access
//--- data are put into internal variables
   CopyBuffer(m_handle_macd,0,0,1,m_buff_MACD_main);
   m_macd_current=m_buff_MACD_main[0];
   CopyBuffer(m_handle_macd,0,1,1,m_buff_MACD_main);
   m_macd_previous=m_buff_MACD_main[0];
   CopyBuffer(m_handle_macd,1,0,1,m_buff_MACD_signal);
   m_signal_current=m_buff_MACD_signal[0];
   CopyBuffer(m_handle_macd,1,1,1,m_buff_MACD_signal);
   m_signal_previous=m_buff_MACD_signal[0];
   CopyBuffer(m_handle_ema,0,0,1,m_buff_EMA);
   m_ema_current=m_buff_EMA[0];
   CopyBuffer(m_handle_ema,0,1,1,m_buff_EMA);
   m_ema_previous=m_buff_EMA[0];
```

This code is saved in "MACD Sample One value at a time.mq5" attached in the end of the article.

#### 3.2. Convert "MACD Sample.mq4" into MQL5 code

To be able to access the indicators in MQL4 style, as well as work with positions and trade, we should include the "IndicatorsMQL4.mqh" file (as you remember, this file understands only MQL4 names of indicator lines) and [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo), [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade), [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) and [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) trading classes. Also, the block of 'defines' — indicator line names — should be added to the EA to properly access the indicators in "IndicatorsMQL4.mqh":

```
#property description " and the indicators are accessed in the style of MQL4"
#define MODE_MAIN    0
#define MODE_SIGNAL  1
#include <SimpleCall\IndicatorsMQL4.mqh>
//---
#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>
CPositionInfo  m_position;                   // trade position object
CTrade         m_trade;                      // trading object
CSymbolInfo    m_symbol;                     // symbol info object
CAccountInfo   m_account;                    // account info wrapper
//---
input double TakeProfit    =50;
```

Also, the special multiplier is required for adjusting to three- and five-digit quotes:

```
input double MACDCloseLevel=2;
input int    MATrendPeriod =26;
//---
double       m_adjusted_point;               // point value adjusted for 3 or 5 points
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
```

To receive the current prices, I use the **_m\_symbol_** object of the CSymbolInfo trading class:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!m_symbol.Name(Symbol())) // sets symbol name
      return(INIT_FAILED);
   RefreshRates();
```

RefreshRates() method updates prices and makes sure there are no prices equal to "0.0":

```
//+------------------------------------------------------------------+
//| Refreshes the symbol quotes data                                 |
//+------------------------------------------------------------------+
bool RefreshRates(void)
  {
//--- refresh rates
   if(!m_symbol.RefreshRates())
     {
      Print("RefreshRates error");
      return(false);
     }
//--- protection against the return value of "zero"
   if(m_symbol.Ask()==0 || m_symbol.Bid()==0)
      return(false);
//---
   return(true);
  }
```

The **_m\_adjusted\_point_** multiplier is initialized in OnInit() after initializing the **_m\_symbol_** object:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!m_symbol.Name(Symbol())) // sets symbol name
      return(INIT_FAILED);
   RefreshRates();
   //--- tuning for 3 or 5 digits
   int digits_adjust=1;
   if(m_symbol.Digits()==3 || m_symbol.Digits()==5)
      digits_adjust=10;
   m_adjusted_point=m_symbol.Point()*digits_adjust;
//---
   return(INIT_SUCCEEDED);
  }
```

In OnTick() we access the indicators in MQL4 style thanks to "IndicatorsMQL4Style.mqh":

```
   if(!RefreshRates())
      return;
//--- to simplify the coding and speed up access data are put into internal variables
   MacdCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MAIN_LINE,0);
   MacdPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MAIN_LINE,1);
   SignalCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,SIGNAL_LINE,0);
   SignalPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,SIGNAL_LINE,1);
   MaCurrent=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,0);
   MaPrevious=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,1);
```

**3.2.1. Working with positions**

For maximum compliance, determine the absence of positions as

```
   total=PositionsTotal();
   if(total<1)
     {
```

Although this approach is not entirely correct, since it does not take into account the presence of positions on other symbols and/or with other identifiers (magic numbers).

**3.2.2. Buy positions are opened** using [Buy](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuy) method of CTrade class, while execution correctness is verified by the [ResultDeal](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctraderesultdeal) method of the same class. ResultDeal returns a deal ticket if it is executed.

```
      //--- check for long position (BUY) possibility
      if(MacdCurrent<0 && MacdCurrent>SignalCurrent && MacdPrevious<SignalPrevious &&
         MathAbs(MacdCurrent)>(MACDOpenLevel*m_adjusted_point) && MaCurrent>MaPrevious)
        {
         m_trade.Buy(Lots,m_symbol.Name(),m_symbol.Ask(),
                     0.0,
                     m_symbol.NormalizePrice(m_symbol.Ask()+TakeProfit*m_adjusted_point),
                     "macd sample");
         if(m_trade.ResultDeal()!=0)
            Print("BUY position opened : ",m_trade.ResultPrice());
         else
            Print("Error opening BUY position : ",m_trade.ResultRetcodeDescription());
         return;
        }
```

Note that the price in a trade request is normalized using the [NormalizePrice](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfonormalizeprice) method of CSymbolInfo trading class. This method allows considering quantization: minimum price change and number of decimal places.

The same methods are used to open a Sell position.

**3.2.3. Positions bypass block:** Closing or modification.

The loop itself is passed from the common number of positions minus one up to zero inclusive. To be able to work with a position, first, we need to select it by index in the general list:

```
   for(int i=PositionsTotal()-1;i>=0;i--)
      if(m_position.SelectByIndex(i)) // selects the position by index for further access to its properties
```

The position is closed using the [PositionClose](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionclose) method, while modification is done by [PositionModify](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionmodify). Note that modification allows using the [NormalizePrice](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfonormalizeprice) method of the CSymbolInfo trading class.

The entire position bypass block:

```
//--- it is important to enter the market correctly, but it is more important to exit it correctly...
   for(int i=PositionsTotal()-1;i>=0;i--)
      if(m_position.SelectByIndex(i)) // selects the position by index for further access to its properties
         if(m_position.Symbol()==m_symbol.Name())
           {
            //--- long position is opened
            if(m_position.PositionType()==POSITION_TYPE_BUY)
              {
               //--- should it be closed?
               if(MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious &&
                  MacdCurrent>(MACDCloseLevel*m_adjusted_point))
                 {
                  //--- close position and exit
                  if(!m_trade.PositionClose(m_position.Ticket()))
                     Print("PositionClose error ",m_trade.ResultRetcodeDescription());
                  return;
                 }
               //--- check for trailing stop
               if(TrailingStop>0)
                 {
                  if(m_position.PriceCurrent()-m_position.PriceOpen()>m_adjusted_point*TrailingStop)
                    {
                     if(m_position.StopLoss()<m_symbol.Bid()-m_adjusted_point*TrailingStop)
                       {
                        //--- modify position and exit
                        if(!m_trade.PositionModify(m_position.Ticket(),
                           m_symbol.NormalizePrice(m_position.PriceCurrent()-m_adjusted_point*TrailingStop),
                           m_position.TakeProfit()))
                           Print("PositionModify error ",m_trade.ResultRetcodeDescription());
                        return;
                       }
                    }
                 }
              }

            if(m_position.PositionType()==POSITION_TYPE_SELL)
              {
               //--- should it be closed?
               if(MacdCurrent<0 && MacdCurrent>SignalCurrent &&
                  MacdPrevious<SignalPrevious && MathAbs(MacdCurrent)>(MACDCloseLevel*m_adjusted_point))
                 {
                  //--- close position and exit
                  if(!m_trade.PositionClose(m_position.Ticket()))
                     Print("PositionClose error ",m_trade.ResultRetcodeDescription());
                  return;
                 }
               //--- check for trailing stop
               if(TrailingStop>0)
                 {
                  if((m_position.PriceOpen()-m_position.PriceCurrent())>(m_adjusted_point*TrailingStop))
                    {
                     if((m_position.StopLoss()>(m_symbol.Ask()+m_adjusted_point*TrailingStop)) || (m_position.StopLoss()==0.0))
                       {
                        //--- modify position and exit
                        if(!m_trade.PositionModify(m_position.Ticket(),
                           m_symbol.NormalizePrice(m_symbol.Ask()+m_adjusted_point*TrailingStop),
                           m_position.TakeProfit()))
                           Print("PositionModify error ",m_trade.ResultRetcodeDescription());
                        return;
                       }
                    }
                 }
              }
           }
```

We are done with all the changes. The final file "MACD Sample 4 to 5 MQL4 style.mq5" is attached below.

#### 3.3. Let's compare the speed of executing MACD-based EAs

The following EAs will be used for comparison:

- "MACD Sample.mq5" — EA from the standard delivery with correct access to indicators
- "MACD Sample One value at a time.mq5" — equivalent of "MACD Sample.mq5", where we obtain one value from the indicators per time
- "MACD Sample 4 to 5 MQL4 style.mq5" — MQL4 EA converted to MQL5 with minimum modifications and access to MQL4 style indicators

The test was performed on USDJPY M30 from 2017.02.01 to 2018.01.16 on MetaQuotes-Demo server. The terminal was reset after each test (whether it was switching EAs or toggling tick generation modes). PC configuration:

```
Windows 10 (build 16299) x64, IE 11, UAC, Intel Core i3-3120M  @ 2.50GHz, Memory: 4217 / 8077 Mb, Disk: 335 / 464 Gb, GMT+2
```

| # | Expert Advisor | Every tick based on real ticks | Every tick | OHLC |
| --- | --- | --- | --- | --- |
|  |  | Test time | Trades | Deals | Test time | Trades | Deals | Test time | Trades | Deals |
| 1 | MACD Sample.mq5 | 0:01:19.485 | 122 | 244 | 0:00:53.750 | 122 | 244 | 0:00:03.735 | 119 | 238 |
| 2 | MACD Sample One value at a time.mq5 | 0:01:20.344 | 122 | 244 | 0:00:56.297 | 122 | 244 | 0:00:03.687 | 119 | 238 |
| 3 | MACD Sample 4 to 5 MQL4 style.mq5 | 0:02:37.422 | 122 | 244 | 0:01:52.171 | 122 | 244 | 0:00:06.312 | 119 | 238 |

All three EAs demonstrated similar charts in "Every tick mode":

![MACD Sample](https://c.mql5.com/2/30/MACD_Sample.png)

Fig. 2. MACD Sample XXXX in the strategy tester

CONCLUSION: "MACD Sample 4 to 5 MQL4 style.mq5" EA having access to indicators in MQL4 style is **twice slower compared to** **similar EAs** having correct access to indicators.

#### 3.4. Let's compare MACD-based EAs' memory consumption

The same 14 charts are used for that, as in point 2. What happens to memory consumption if we apply indicators in MQL4 style at each tick? "Terminal memory used.mq5" indicator is always left on the first chart. It prints [TERMINAL\_MEMORY\_USED](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus) ID every 10 seconds, while the EAs are launched on the remaining 13 ones one-by-one. The terminal is reset before each measurement.

| # | Expert Advisor | Task manager, MB | TERMINAL\_MEMORY\_USED, Мб |
| --- | --- | --- | --- |
| 1 | MACD Sample.mq5 | 334.6 | 813 |
| 2 | MACD Sample One value at a time.mq5 | 335.8 | 813 |
| 3 | MACD Sample 4 to 5 MQL4 style.mq5 | 342.2 | 818 |

CONCLUSION: MACD-based EAs with correct access to the indicators and the MACD-based EA with access to the indicators in MQL4 style are comparable in terms of memory consumption. They consume approximately the same amount of memory.

### 4\. The new life of \[data folder\]\\MQL4\\Experts\\Moving Average.mq4 EA

In the previous section, we converted MQL4 into MQL5. As for Movinge Average.mq4, I suggest to simply change Moving Average.mq5 by including "IndicatorsMQL5.mqh" file

```
#property version   "1.00"
#include <SimpleCall\IndicatorsMQL5.mqh>

#include <Trade\Trade.mqh>
```

and replacing CopyBuffer

```
//--- get current Moving Average
   double   ma[1];
   if(CopyBuffer(ExtHandle,0,0,1,ma)!=1)
     {
      Print("CopyBuffer from iMA failed, no data");
      return;
     }
```

with MQL4 style of accessing the indicators:

```
//--- get Moving Average
   ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
```

This leaves us with just one option of checking the operation result — compare the obtained data with zero. Considering this, the final entry in the "CheckForOpen" and "CheckForClose" blocks looked as follows:

```
//--- get current Moving Average
   double   ma[1];
   if(CopyBuffer(ExtHandle,0,0,1,ma)!=1)
     {
      Print("CopyBuffer from iMA failed, no data");
      return;
     }
```

and is about to look like this:

```
//--- get current Moving Average
   double   ma[1];
   ma[0]=iMA(_Symbol,_Period,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
//if(CopyBuffer(ExtHandle,0,0,1,ma)!=1)
   if(ma[0]==0.0)
     {
      //Print("CopyBuffer from iMA failed, no data");
      Print("Get iMA in MQL4 style failed, no data");
      return;
     }
```

These are the changes we are going to save in the "Moving Average MQL4 style.mq5" EA. The EA is attached below. Let's measure the performance and memory consumption between the standard "Moving Average.mq5" and "Moving Average MQL4 style.mq5".

As you may remember, the tests were performed on the following system

```
Windows 10 (build 16299) x64, IE 11, UAC, Intel Core i3-3120M  @ 2.50GHz, Memory: 4217 / 8077 Mb, Disk: 335 / 464 Gb, GMT+2
```

The terminal was reset after each test. The tests were conducted on EURUSD M15 from 2017.02.01 to 2018.01.16 on MetaQuotes-Demo server.

| # | Expert Advisor | Every tick based on real ticks | Every tick | OHLC |
| --- | --- | --- | --- | --- |
|  |  | Test time | Trades | Deals | Test time | Trades | Deals | Test time | Trades | Deals |
| 1 | Moving Average.mq5 | 0:00:33.359 | 1135 | 2270 | 0:00:22.562 | 1114 | 2228 | 0:00:02.531 | 1114 | 2228 |
| 2 | Moving Average MQL4 style.mq5 | 0:00:34.984 | 1135 | 2270 | 0:00:23.750 | 1114 | 2228 | 0:00:02.578 | 1114 | 2228 |

CONCLUSION: The MQL5 core probably had to search among two handles in MACD Sample at each tick when accessing the indicators in MQL4 style. It was this search that took the most time.

In case of the Moving Average EA, when accessing the indicator in MQL4 style, the MQL5 core spends no time in searching for a necessary candle, since it is the only one.

**Let's compare Moving Average-based EAs' memory consumption**

The same 14 charts are used for that, as in point 2. "Terminal memory used.mq5" indicator is always left on the first chart. It prints [TERMINAL\_MEMORY\_USED](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus) ID every 10 seconds, while the EAs are launched on the remaining 13 ones one-by-one. The terminal is reset before each measurement.

| # | Expert Advisor | Task manager, MB | TERMINAL\_MEMORY\_USED, Мб |
| --- | --- | --- | --- |
| 1 | Moving Average.mq5 | 295.6 | 771 |
| 2 | Moving Average MQL4 style.mq5 | 283.6 | 760 |

CONCLUSION: The memory consumption is almost identical. Small differences can be attributed to the terminal's "internal life": news updates, etc.

### 5\. Equivalents of iXXXX series

Since we executed the obtaining of indicator values in MQL4 style, let's write the functions of the ["Access to Timeseries and Indicator Data"](https://docs.mql4.com/series "https://docs.mql4.com/series") section. The implementation is done in \[data folder\]\\MQL5\\Include\\SimpleCall\ **Series.mqh**.

The list of functions in **Series.mqh** providing access to time series values as in MQL4:

- [Bars](https://docs.mql4.com/series/barsfunction "https://docs.mql4.com/series/barsfunction")
- [iBarShift](https://docs.mql4.com/en/series/ibarshift "https://docs.mql4.com/en/series/ibarshift")
- [iClose](https://docs.mql4.com/en/series/iclose "https://docs.mql4.com/en/series/iclose")
- [iHigh](https://docs.mql4.com/series/ihigh "https://docs.mql4.com/series/ihigh")
- [iHighest](https://docs.mql4.com/series/ihighest "https://docs.mql4.com/series/ihighest")
- [iLow](https://docs.mql4.com/series/ilow "https://docs.mql4.com/series/ilow")
- [iLowest](https://docs.mql4.com/series/ilowest "https://docs.mql4.com/series/ilowest")
- [iOpen](https://docs.mql4.com/series/iopen "https://docs.mql4.com/series/iopen")
- [iTime](https://docs.mql4.com/series/itime "https://docs.mql4.com/series/itime")
- [iVolume](https://docs.mql4.com/series/ivolume "https://docs.mql4.com/series/ivolume")

The predefined IDs of the MODE\_OPEN, MODE\_LOW, MODE\_HIGH, MODE\_CLOSE, MODE\_VOLUME and MODE\_TIME series are available for the iHighest and iLowest functions.

Example of the iClose function implementation:

```
//+------------------------------------------------------------------+
//| iClose function in MQL4 notation                                 |
//+------------------------------------------------------------------+
double   iClose(
                string                    symbol,              // symbol
                ENUM_TIMEFRAMES           timeframe,           // timeframe
                int                       shift                // shift
                )
  {
   double result=0.0;
//---
   double val[1];
   ResetLastError();
   int copied=CopyClose(symbol,timeframe,shift,1,val);
   if(copied>0)
      result=val[0];
   else
      Print(__FUNCTION__,": CopyClose error=",GetLastError());
//---
   return(result);
  }
```

**_shift_** bar Close price value is obtained using [CopyClose](https://www.mql5.com/en/docs/series/copyclose) — the first form of call (accessing by the initial position and the number of required elements):

```
int  CopyClose(
   string           symbol_name,       // symbol name
   ENUM_TIMEFRAMES  timeframe,         // period
   int              start_pos,         // initial position
   int              count,             // copied number
   double           close_array[]      // array for copying Close prices
   );
```

### Conclusion

As we can see, MQL5 allows MQL4 fans to obtain the values ​​of indicators and time series in their favorite style. They say that this code is shorter and easier to read. Platform developers though require more careful work with a code and maximum checks when calling functions (and I fully agree with them). Let's list briefly the pros and cons of the functions described in the article.

**Cons:**

- limitation in processing the returned error when accessing indicators;
- drop in test speed when accessing more than one indicator simultaneously;

- the need to correctly highlight the indicator lines depending on whether IndicatorsMQL5.mqh or IndicatorsMQL4.mqh is connected.


**Pros**

- simplicity of code writing — one string instead of multiple ones;

- visibility and conciseness — the less the code amount, the easier it is for understanding.


However, I remain committed to the classic MQL5 approach in accessing indicators. In this article, I have only tested a possible alternative.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/4318](https://www.mql5.com/ru/articles/4318)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4318.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/4318/mql5.zip "Download MQL5.zip")(25.33 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717)
- [Gap - a profitable strategy or 50/50?](https://www.mql5.com/en/articles/5220)
- [Elder-Ray (Bulls Power and Bears Power)](https://www.mql5.com/en/articles/5014)
- [Improving Panels: Adding transparency, changing background color and inheriting from CAppDialog/CWndClient](https://www.mql5.com/en/articles/4575)
- [How to create a graphical panel of any complexity level](https://www.mql5.com/en/articles/4503)
- [Comparing speeds of self-caching indicators](https://www.mql5.com/en/articles/4388)
- [LifeHack for traders: Blending ForEach with defines (#define)](https://www.mql5.com/en/articles/4332)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/231370)**
(129)


![Marco vd Heijden](https://c.mql5.com/avatar/2021/9/613A9489-6810.png)

**[Marco vd Heijden](https://www.mql5.com/en/users/thecreator1)**
\|
11 Mar 2018 at 23:21

If i run a simple iVolume in a loop, over all available instruments, in MQL5, then the platform just freezes up, the massive [copybuffer calls](https://www.mql5.com/en/docs/series/copybuffer "MQL5 documentation: CopyBuffer function") act like a ddos attack on the operating system, it just trips, where if i do this same process in MQL4, it runs smooth like a pack of hot molten butter.

On the same machine, and that tells me more then i need to know.

Combine that with all the signals and noise coming from the community which tells me that my conclusions were right all along.

![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
17 Mar 2018 at 01:30

Mark only


![Ludovico Mattiuzzo](https://c.mql5.com/avatar/avatar_na2.png)

**[Ludovico Mattiuzzo](https://www.mql5.com/en/users/ludoz)**
\|
25 Jun 2018 at 21:52

The problem is: why do I have to copy the buffer every time I have to read the indicator value? The buffer is there, already computed, why can't I access it's value directly?

This is a non-sense!

I should only have direct access to the [indicator buffer](https://www.mql5.com/en/docs/constants/indicatorconstants/lines "MQL5 documentation: Indicators Lines"), copying it over and over will only degrade the performance.

I really don't understand why Metaquotes has followed this way of work.

CopyBuffer(...) vs   buffer\[i\] ?

![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
25 Jun 2018 at 22:24

**ludoz:**

The problem is: why do I have to copy the buffer every time I have to read the indicator value? The buffer is there, already computed, why can't I access it's value directly?

This is a non-sense!

I should only have direct access to the [indicator buffer](https://www.mql5.com/en/docs/constants/indicatorconstants/lines "MQL5 documentation: Indicators Lines"), copying it over and over will only degrade the performance.

I really don't understand why Metaquotes has followed this way of work.

CopyBuffer(...) vs   buffer\[i\] ?

Because that's not so simple. The indicator run on one thread and an EA on an other thread (and you can have several indicators/several EAs). If you are complaining about such a simple procedure as handle/CopyBuffer, you don't want to proceed with multi-threaded application, believe me.

mql5 provides generic solutions, able to manage most of the "normal" situation. If you have specific issue on your project, there are always solution.

![-whkh18-](https://c.mql5.com/avatar/2018/8/5B8005DA-E909.jpg)

**[-whkh18-](https://www.mql5.com/en/users/-whkh18-)**
\|
6 Sep 2018 at 02:06

It's MQL5 now, but most people still use mt4.


![Controlled optimization: Simulated annealing](https://c.mql5.com/2/31/icon__1.png)[Controlled optimization: Simulated annealing](https://www.mql5.com/en/articles/4150)

The Strategy Tester in the MetaTrader 5 trading platform provides only two optimization options: complete search of parameters and genetic algorithm. This article proposes a new method for optimizing trading strategies — Simulated annealing. The method's algorithm, its implementation and integration into any Expert Advisor are considered. The developed algorithm is tested on the Moving Average EA.

![Custom Strategy Tester based on fast mathematical calculations](https://c.mql5.com/2/30/Custom_math_tester.png)[Custom Strategy Tester based on fast mathematical calculations](https://www.mql5.com/en/articles/4226)

The article describes the way to create a custom strategy tester and a custom analyzer of the optimization passes. After reading it, you will understand how the math calculations mode and the mechanism of so-called frames work, how to prepare and load custom data for calculations and use effective algorithms for their compression. This article will also be interesting to those interested in ways of storing custom information within an expert.

![LifeHack for traders: Blending ForEach with defines (#define)](https://c.mql5.com/2/31/ForEachwdefine.png)[LifeHack for traders: Blending ForEach with defines (#define)](https://www.mql5.com/en/articles/4332)

The article is an intermediate step for those who still writes in MQL4 and has no desire to switch to MQL5. We continue to search for opportunities to write code in MQL4 style. This time, we will look into the macro substitution of the #define preprocessor.

![Automatic construction of support and resistance lines](https://c.mql5.com/2/30/Auto_support_resisitance.png)[Automatic construction of support and resistance lines](https://www.mql5.com/en/articles/3215)

The article deals with automatic construction of support/resistance lines using local tops and bottoms of price charts. The well-known ZigZag indicator is applied to define these extreme values.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/4318&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071708433863879921)

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