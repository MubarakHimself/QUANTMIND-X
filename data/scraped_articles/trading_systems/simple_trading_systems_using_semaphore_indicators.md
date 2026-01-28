---
title: Simple Trading Systems Using Semaphore Indicators
url: https://www.mql5.com/en/articles/358
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:51:32.520847
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/358&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062770581151066258)

MetaTrader 5 / Trading systems


### Introduction

Semaphore or signal indicators are simple detectors that indicate the moments for market entry or exit. In case there is an entry signal at the current bar, an appropriate label appears on a symbol chart. This label can then be used as a condition for performing a deal.

There are a lot of indicators of that kind, but the very essence of the original trading system based on such indicators has not changed at all. Therefore, it is a good idea to implement it in the most simple and universal form. This will allow further use of the obtained result when working with any similar indicators without considerable alterations.

![Fig.1. ASCtrend semaphore signal indicator](https://c.mql5.com/2/3/ASCtrend.png)

Fig.1. ASCtrend semaphore signal indicator

![Fig.2. ASCtrend indicator. Trading signal for performing a deal](https://c.mql5.com/2/3/ASCtrendBuy__1.png)

Fig.2. Trading signal for performing a deal using ASCtrend semaphore signal indicator

### Samples of Typical Semaphore Signal Indicators

Currently there are lots of such indicators in [Code Base](https://www.mql5.com/en/code). In this article I will provide only a few links to the appropriate web pages:

- [BykovTrend](https://www.mql5.com/en/code/497),

- [ASCtrend](https://www.mql5.com/en/code/491),

- [BrainTrend1Sig](https://www.mql5.com/en/code/392),

- [BrainTrend2Sig](https://www.mql5.com/en/code/395),
- [SilverTrend\_Signal](https://www.mql5.com/en/code/459),
- [Stalin](https://www.mql5.com/en/code/487),

- [WPRSI signal](https://www.mql5.com/en/code/599),

- [StepMA\_NRTR](https://www.mql5.com/en/code/559),
- [LeManSignal](https://www.mql5.com/en/code/474),
- [3Parabolic System](https://www.mql5.com/en/code/554),

- [PriceChannel\_Stop](https://www.mql5.com/en/code/417),
- [Arrows&Curves](https://www.mql5.com/en/code/414),
- [Karacatica](https://www.mql5.com/en/code/416),

- [Sidus](https://www.mql5.com/en/code/751).

In addition to the semaphore signal indicators, there is a group of semaphore trend indicators:

![Fig.3. Trading signals using Heiken_Ashi_Smoothed indicator](https://c.mql5.com/2/3/Heiken_Ashi_Smoothed.png)

Fig.3. Semaphore trend indicator

![Fig.4. Trading signal for performing a deal using Heiken Ashi Smoothed semaphore trend indicator](https://c.mql5.com/2/3/Heiken_Ashi_Smoothed_1__1.png)

Fig.4. Trading signal for performing a deal using Heiken Ashi Smoothed semaphore trend indicator

Trading systems using such indicators have slightly different code for getting trading signals, while the Expert Advisor code remains almost unchanged.

### Samples of Typical Semaphore Trend Indicators

Code Base contains plenty of such indicators. In this article I will provide only a few links to the appropriate web pages:

- [FiboCandles](https://www.mql5.com/en/code/682),
- [Parabolic SAR](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/psar),

- [X2MA](https://www.mql5.com/en/code/642),

- [Candles\_Smoothed](https://www.mql5.com/en/code/536),
- [SuperTrend](https://www.mql5.com/en/code/527),

- [Go](https://www.mql5.com/en/code/440),
- [3LineBreak](https://www.mql5.com/en/code/485),

- [Laguerre](https://www.mql5.com/en/code/432),
- [Heiken Ashi Smoothed](https://www.mql5.com/en/code/701),

- [NonLagDot](https://www.mql5.com/en/code/694).

### Basic Data for Creating a Trading System:

1. Semaphore indicator with the input parameters that are to be present in the Expert Advisor;
2. The list of additional input Expert Advisor trading parameters:

   - a share of a deposit financial resources used in a deal;
   - a size of Stop Loss and Take Profit (pending orders must not be used in case of zero values);
   - slippage (maximum allowable difference between set and actual deal prices);
   - index of the bar, from which trading signals will be received;
   - permissions for opening long and short positions;

   - permissions for forced closing of long and short positions according to indicator signals.

Of course, it would be much more convenient to give orders for performing deals using universal trading functions. These functions are quite complex and they should be packed in a separate library file to make the application code as easy as possible.

### The code of the Expert Advisor implementing the semaphore trading system:

```
//+------------------------------------------------------------------+
//|                                                 Exp_ASCtrend.mq5 |
//|                             Copyright © 2011,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2011, Nikolay Kositsin"
#property link      "farria@mail.redcom.ru"
#property version   "1.00"
//+----------------------------------------------+
//| Expert Advisor indicator input parameters    |
//+----------------------------------------------+
input double MM=-0.1;             // Share of a deposit in a deal, negative values - lot size
input int    StopLoss_=1000;      // Stop loss in points
input int    TakeProfit_=2000;    // Take profit in points
input int    Deviation_=10;       // Max. price deviation in points
input bool   BuyPosOpen=true;     // Permission to buy
input bool   SellPosOpen=true;    // Permission to sell
input bool   BuyPosClose=true;    // Permission to exit long positions
input bool   SellPosClose=true;   // Permission to exit short positions
//+----------------------------------------------+
//| ASCtrend indicator input parameters          |
//+----------------------------------------------+
input ENUM_TIMEFRAMES InpInd_Timeframe=PERIOD_H1; // ASCtrend indicator time frame
input int  RISK=4;                               // Risk level
input uint SignalBar=1;                          // Bar index for getting an entry signal
//+----------------------------------------------+

int TimeShiftSec;
//---- declaration of integer variables for the indicators handles
int InpInd_Handle;
//---- declaration of integer variables of the start of data calculation
int min_rates_total;
//+------------------------------------------------------------------+
//| Trading algorithms                                               |
//+------------------------------------------------------------------+
#include <TradeAlgorithms.mqh>
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---- getting ASCtrend indicator handle
   InpInd_Handle=iCustom(Symbol(),InpInd_Timeframe,"ASCtrend",RISK);
   if(InpInd_Handle==INVALID_HANDLE) Print(" Failed to get handle of ASCtrend indicator");

//---- initialization of a variable for storing a chart period in seconds
   TimeShiftSec=PeriodSeconds(InpInd_Timeframe);

//---- initialization of variables of the start of data calculation
   min_rates_total=int(3+RISK*2+SignalBar);
//----
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//----
   GlobalVariableDel_(Symbol());
//----
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---- checking the number of bars to be enough for calculation
   if(BarsCalculated(InpInd_Handle)<min_rates_total) return;

//---- uploading history for IsNewBar() and SeriesInfoInteger() functions normal operation
   LoadHistory(TimeCurrent()-PeriodSeconds(InpInd_Timeframe)-1,Symbol(),InpInd_Timeframe);

//---- declaration of local variables
   double DnVelue[1],UpVelue[1];
//---- declaration of static variables
   static bool Recount=true;
   static bool BUY_Open=false,BUY_Close=false;
   static bool SELL_Open=false,SELL_Close=false;
   static datetime UpSignalTime,DnSignalTime;
   static CIsNewBar NB;

//+----------------------------------------------+
//| Searching for deals performing signals       |
//+----------------------------------------------+
   if(!SignalBar || NB.IsNewBar(Symbol(),InpInd_Timeframe) || Recount) // checking for a new bar
     {
      //---- zeroing out trading signals
      BUY_Open=false;
      SELL_Open=false;
      BUY_Close=false;
      SELL_Close=false;
      Recount=false;

      //---- copy newly appeared data into the arrays
      if(CopyBuffer(InpInd_Handle,1,SignalBar,1,UpVelue)<=0) {Recount=true; return;}
      if(CopyBuffer(InpInd_Handle,0,SignalBar,1,DnVelue)<=0) {Recount=true; return;}

      //---- getting buy signals
      if(UpVelue[0] && UpVelue[0]!=EMPTY_VALUE)
        {
         if(BuyPosOpen) BUY_Open=true;
         if(SellPosClose) SELL_Close=true;
         UpSignalTime=datetime(SeriesInfoInteger(Symbol(),InpInd_Timeframe,SERIES_LASTBAR_DATE))+TimeShiftSec;
        }

      //---- getting sell signals
      if(DnVelue[0] && DnVelue[0]!=EMPTY_VALUE)
        {
         if(SellPosOpen) SELL_Open=true;
         if(BuyPosClose) BUY_Close=true;
         DnSignalTime=datetime(SeriesInfoInteger(Symbol(),InpInd_Timeframe,SERIES_LASTBAR_DATE))+TimeShiftSec;
        }

      //---- searching for the last trading direction for getting positions closing signals
      //if(!MQL5InfoInteger(MQL5_TESTING) && !MQL5InfoInteger(MQL5_OPTIMIZATION)) //if execution is set to "Random delay" in the Strategy Tester
      if((BuyPosOpen && BuyPosClose || SellPosOpen && SellPosClose) && (!BUY_Close && !SELL_Close))
        {
         int Bars_=Bars(Symbol(),InpInd_Timeframe);

         for(int bar=int(SignalBar+1); bar<Bars_; bar++)
           {
            if(SellPosClose)
              {
               if(CopyBuffer(InpInd_Handle,1,bar,1,UpVelue)<=0) {Recount=true; return;}
               if(UpVelue[0]!=0 && UpVelue[0]!=EMPTY_VALUE)
                 {
                  SELL_Close=true;
                  break;
                 }
              }

            if(BuyPosClose)
              {
               if(CopyBuffer(InpInd_Handle,0,bar,1,DnVelue)<=0) {Recount=true; return;}
               if(DnVelue[0]!=0 && DnVelue[0]!=EMPTY_VALUE)
                 {
                  BUY_Close=true;
                  break;
                 }
              }
           }
        }
     }

//+----------------------------------------------+
//| Performing deals                             |
//+----------------------------------------------+
//---- Closing a long position
   BuyPositionClose(BUY_Close,Symbol(),Deviation_);

//---- Closing a short position
   SellPositionClose(SELL_Close,Symbol(),Deviation_);

//---- Buying
   BuyPositionOpen(BUY_Open,Symbol(),UpSignalTime,MM,0,Deviation_,StopLoss_,TakeProfit_);

//---- Selling
   SellPositionOpen(SELL_Open,Symbol(),DnSignalTime,MM,0,Deviation_,StopLoss_,TakeProfit_);
//----
  }
//+------------------------------------------------------------------+
```

The code for realization of such an idea is quite simple and clear, though some details should be clarified.

The chart period used by a signal indicator and an Expert Advisor is fixed in the InpInd\_Timeframe input variable of the Expert Advisor. Therefore, the change of a chart, at which an Expert Advisor is located, does not alter this parameter for the Expert Advisor.

IsNewBar() function needed for determining the moment of a new bar arrival is implemented as a class placed in TradeAlgorithms.mqh file. This allows to use any number of such functions in the code easily by setting an individual static CIsNewBar variable for each of them.

UpSignalTime and DnSignalTime variables are used for storing and transferring the time, after which it is possible to perform the next deal after the previous one, to trading functions. In our case this feature is used to avoid performing several deals in the same direction at the same bar (when performing a deal, the trading function stores the time of the current bar finish and does not perform new deals in the same direction up to that moment).

The block "Searching for the last trading direction to get signals for closing positions" in OnTick() function is needed to receive positions closing signals on the bars with no trading signals. In case of an Expert Advisor normal operation, there is no need in them. But in case of the internet connection failure, it is quite possible that a new trading signal will be missed. It is hardly a good idea to enter the market post factum, but it would be a wise move to close the open positions.

### Using the Trading System with Other Semaphore Signal Indicators

Now, if there is a necessity to use this code with another semaphore signal indicator, the following actions should be performed:

1. Replace the previous indicator data by the necessary parameters of the new one in an Expert Advisor input parameters;
2. Change the code of getting the indicator handle in OnInit() block;
3. Determine the indices for the indicator buffers, used for storing buy and sell trading signals from the indicator code, and enter them appropriately in CopyBuffer() function calls of OnTick() block. In this case zero and first indicator buffers are used;
4. Change the initialization of the data calculation starting point variable (min\_rates\_total) in an Expert Advisor according to the
indicator code;
5. Change the block "Searching for the last trading direction to get signals for closing positions" in OnTick() function according to the indicator code.

### Using the Trading System with Other Semaphore Trend Indicators

When using this trading system with semaphore trend indicator, the Expert Advisor code has changed a bit in the block for determining signals for OnTick() function deals. For example, the code will look as follows for the Expert Advisor based on [FiboCandles](https://www.mql5.com/en/code/682) indicator:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---- checking the number of bars to be enough for calculation
   if(BarsCalculated(InpInd_Handle)<min_rates_total) return;

//---- uploading history for IsNewBar() and SeriesInfoInteger() functions
   LoadHistory(TimeCurrent()-PeriodSeconds(InpInd_Timeframe)-1,Symbol(),InpInd_Timeframe);

//---- declaration of local variables
   double TrendVelue[2];
//---- declaration of static variables
   static bool Recount=true;
   static bool BUY_Open=false,BUY_Close=false;
   static bool SELL_Open=false,SELL_Close=false;
   static datetime UpSignalTime,DnSignalTime;
   static CIsNewBar NB;

//+----------------------------------------------+
//| Searching for deals performing signals       |
//+----------------------------------------------+
   if(!SignalBar || NB.IsNewBar(Symbol(),InpInd_Timeframe) || Recount) // checking for a new bar
     {
      //---- zeroing out trading signals
      BUY_Open=false;
      SELL_Open=false;
      BUY_Close=false;
      SELL_Close=false;
      Recount=false;

      //---- copy the newly obtained data into the arrays
      if(CopyBuffer(InpInd_Handle,4,SignalBar,2,TrendVelue)<=0) {Recount=true; return;}

      //---- getting buy signals
      if(TrendVelue[0]==1 && TrendVelue[1]==0)
        {
         if(BuyPosOpen) BUY_Open=true;
         if(SellPosClose)SELL_Close=true;
         UpSignalTime=datetime(SeriesInfoInteger(Symbol(),InpInd_Timeframe,SERIES_LASTBAR_DATE))+TimeShiftSec;
        }

      //---- getting sell signals
      if(TrendVelue[0]==0 && TrendVelue[1]==1)
        {
         if(SellPosOpen) SELL_Open=true;
         if(BuyPosClose) BUY_Close=true;
         DnSignalTime=datetime(SeriesInfoInteger(Symbol(),InpInd_Timeframe,SERIES_LASTBAR_DATE))+TimeShiftSec;
        }

      //---- searching for the last trading direction for getting positions closing signals
      //if(!MQL5InfoInteger(MQL5_TESTING) && !MQL5InfoInteger(MQL5_OPTIMIZATION)) //if execution is set to "Random delay" in the Strategy Tester
        {
         if(SellPosOpen && SellPosClose  &&  TrendVelue[1]==0) SELL_Close=true;
         if(BuyPosOpen  &&  BuyPosClose  &&  TrendVelue[1]==1) BUY_Close=true;
        }
     }

//+----------------------------------------------+
//| Performing deals                             |
//+----------------------------------------------+
//---- Closing a long position
   BuyPositionClose(BUY_Close,Symbol(),Deviation_);

//---- Closing a short position
   SellPositionClose(SELL_Close,Symbol(),Deviation_);

//---- Buying
   BuyPositionOpen(BUY_Open,Symbol(),UpSignalTime,MM,0,Deviation_,StopLoss_,TakeProfit_);

//---- Selling
   SellPositionOpen(SELL_Open,Symbol(),DnSignalTime,MM,0,Deviation_,StopLoss_,TakeProfit_);
//----
  }
```

In this case the trading signals are received from only one color indicator buffer (containing color indices). The data in this buffer can have only two values: 0 - for ascending market and 1 - for descending one. "Searching for the last trading direction for getting positions closing signals" block code has become as simple as possible, as a trend direction at any bar can be received directly from the appropriate cell of the indicator buffer.

At the "Performing deals" block the functions of positions closing go first, followed by opening functions. In case of the opposite sequence, it will be possible only to close the deals on one bar, you won't be able to open them simultaneously when testing in the "Open prices only" mode! Therefore, the trading results will be seriously disrupted.

### Testing the Trading System

Before proceeding to the trading system testing, one important detail should be clarified. In case SignalBar input variable value is equal to zero, the Expert Advisor will get deals performing signals from the current bar. But the current bar signal is not reliable in indicating the change of the trend that moved against this signal at the previous bar. The signals on the current bar can appear and disappear, while a trend can move against such signals for quite a long time. This can be easily seen, if an Expert Advisor is tested on all ticks with enabled visualization and SignalBar variable being equal to zero. ASCtrend indicator operation visualization presents a very clear evidence of this fact in such case.

Again, only "Every tick" mode is suitable for an Expert Advisor optimization with a signal received from the current bar. In case it is to be received from any other already closed bar, the "Open prices only" mode is quite enough. That greatly accelerates the trading system behavior analysis without any serious losses in its quality.

Therefore, it is better not to use signals from the current bar for testing and optimization of such trading systems!

So, let's test the Expert Advisor with default parameters on EUR/USD since the beginning of the year up to the beginning of December:

![Fig.5. Testing results of Exp_ASCtrend Expert Advisor with default parameters on EUR/USD H1](https://c.mql5.com/2/3/TesterGraphReport2011r12h09__1.png)

Fig.5. Testing results of Exp\_ASCtrend Expert Advisor with default parameters on EUR/USD H1

After changing a bit the Expert Advisor settings in the Strategy Tester, we can find the most suitable combination of the Expert Advisor parameters for existing historical data quite easily:

![Fig.6. Testing results of Exp_ASCtrend Expert Advisor after optimization with better parameters on EUR/USD H1](https://c.mql5.com/2/3/TesterGraphReport2011i12d09_Optim_Longlshort__1.png)

Fig.6. Testing results of Exp\_ASCtrend Expert Advisor after optimization with better parameters on EUR/USD H1

The process of the trading system optimization does not have any peculiarities, that is why I will provide only one link to the article describing this process in detail: " [MQL5: Guide to Testing and Optimizing of Expert Advisors in MQL5](https://www.mql5.com/en/articles/156)".

Of course, it would be naive to expect some outstanding profits from such a simple trading system. But it is quite possible to achieve good results in case this semi-automatic system is skillfully handled and is regularly tuned according to the market current behavior.

For example, there was an upward trend on EUR/USD H12 chart in 2011 from January up to May. And it was easily detectable at early stages:

![Fig.7. EUR/USD H12 chart (January/May 2011)](https://c.mql5.com/2/3/Trend.png)

Fig.7. EUR/USD H12 chart (January/May 2011)

It would be interesting to test the Expert Advisor on this time interval with the default settings, the possibility to buy only and the use of only 5% of a deposit (MM=0.05). Here are the results of the Expert Advisor with such parameters tested on H1 chart:

![Fig.8. Testing results of Exp_ASCtrend Expert Advisor with default parameters on EUR/USD H1 for January/May 2011 (only long positions, MM=0.05)](https://c.mql5.com/2/3/TesterGraphReport2011z12k11Long__1.png)

Fig.8. Testing results of Exp\_ASCtrend Expert Advisor with default parameters on EUR/USD H1 for January/May 2011 (only long positions, MM=0.05)

Of course, in this case a trader is fully responsible for selecting a deal direction. But if we keep in mind that it should be done using large time frame charts, we will hardly face any difficulties.

### Modification of the Trading Module for Using It with Another Indicator

This article could have been finished here but MetaEditor has acquired the [possibility to generate Expert Advisors](https://www.mql5.com/en/articles/275) based on ready-made trading modules. [The process of creating such modules](https://www.mql5.com/en/articles/226) considering all the material presented here is quite complex and requires a separate study. Therefore, I will focus on the already created trading modules that are completely analogous to the trading systems I have suggested. And only after that I will move on to the details of these modules modification according to the specific signal indicators avoiding unnecessary detalization.

Let's assume that we already have the collection of trading modules for semaphore signal systems (MySignals.zip) and want to create the analogous module for any particular indicator. Let it be BykovTrendSignal.mq5 indicator, which is a typical semaphore signal indicator. First of all, we should find the most accurate analogue of the indicator from this collection (Indicators.zip). Visually we determine that the first indicator from this article (ASCtrend) is the most similar to it.  Therefore, we will use the trading module of this indicator for modification.

Considering its use in the required program code, the indicator itself (BykovTrend) has a set of input parameters:

```
//+----------------------------------------------+
//| Indicator input parameters                   |
//+----------------------------------------------+
input int RISK=3;
input int SSP=9;
//+----------------------------------------------+
```

And we need the indices of the indicator buffers used for storing the signals for performing deals. In our case these are: 0 - for sell signals and 1 - for buy signals.

Now that we know, which module should be used for modification, we copy it in \\MQL5\\Include\\Expert\\Signal\\MySignals\ folder with BykovTrendSignal.mqh file name and then open it in MetaEditor. There is a regularly encountered expression "ASCtrend" (the previous indicator name) in the used code. It should be replaced by the name of the new indicator - "BykovTrend". To do this, press "Ctrl" and "H" keys simultaneously and make the necessary change:

![Replacing the indicator name in the trading module code](https://c.mql5.com/2/3/Image_3.png)

Fig.9. Replacing the indicator name in the trading module code

Next stage of our work is the most meticulous one. We have to replace everything that concerns the indicator input parameters in the trading module code. The process is very similar to what was stated in the article " [MQL5 Wizard: How to create a module of trading signals](https://www.mql5.com/en/articles/226)".

First of all, we should make some changes in the commented out block of MQL5 Wizard trading signals class description:

```
//+----------------------------------------------------------------------+
//| Description of the class                                             |
//| Title=The signals based on BykovTrend indicator                      |
//| Type=SignalAdvanced                                                  |
//| Name=BykovTrend                                                      |
//| Class=CBykovTrendSignal                                              |
//| Page=                                                                |
//| Parameter=BuyPosOpen,bool,true,Permission to buy                     |
//| Parameter=SellPosOpen,bool,true,Permission to sell                   |
//| Parameter=BuyPosClose,bool,true,Permission to exit a long position   |
//| Parameter=SellPosClose,bool,true,Permission to exit a short position |
//| Parameter=Ind_Timeframe,ENUM_TIMEFRAMES,PERIOD_H1,Timeframe          |
//| Parameter=RISK,int,4,Risk level                                      |
//| Parameter=SSP,int,9,SSP                                              |
//| Parameter=SignalBar,uint,1,Bar index for entry signal                |
//+----------------------------------------------------------------------+
//--- wizard description end
//+----------------------------------------------------------------------+
//| CBykovTrendSignal class.                                             |
//| Purpose: Class of generator of trade signals based on                |
//| BykovTrend indicator https://www.mql5.com/ru/code/497/.               |
//|             Is derived from the CExpertSignal class.                 |
//+----------------------------------------------------------------------+
```

Both indicators contain the same RISK input variable, therefore, it can be left. But in these indicators its default value is different. In fact, this difference is not critical and can be left unchanged. The comment line about SSP variable has been added:

```
//| Parameter=SSP,int,9,SSP                                    |
```

And the link to the Code Base indicator has been replaced:

```
//| Purpose: Class of generator of trade signals based on      |
//| BykovTrend values https://www.mql5.com/ru/code/497/.        |
```

Now, all that relates to the changes of input parameters should be reflected in the description of CBykovTrendSignal trading signals class. We have the line of the new global m\_SSP class variable declaration in settings parameters:

```
   uint              m_SSP;              // SSP
```

and the line of the new SSP() settings parameters installation method declaration:

```
   void               SSP(uint value)                         { m_SSP=value;              }
```

Everything related to RISK input variable in the trading signals module that we create is equivalent to the input module and, therefore, there are no changes in the current and any other trading module blocks.

Now, we pass to the CBykovTrendSignal::CBykovTrendSignal() class constructor. Initialization of a new variable should be added in this block:

```
   m_SSP=4;
```

Checking of the new variable for correctness should be performed in
CBykovTrendSignal::ValidationSettings() settings parameters verification block:

```
   if(m_SSP<=0)
     {
      printf(__FUNCTION__+": SSP must be above zero");
      return(false);
     }
```

After that we may pass to BykovTrend indicator initialization block - BykovTrendSignal::InitBykovTrend(). The new indicator has a different number of input variables and, therefore, dimension for the declared input parameters array will also be different:

```
//--- setting the indicator parameters
   MqlParam parameters[3];
```

In our case we need one dimension for the indicator string name and two more for its input parameters.

Now we have to initialize a new cell of the input parameters arrays, indicating the type of the variable that will be stored in it:

```
   parameters[2].type=TYPE_INT;
   parameters[2].integer_value=m_SSP;
```

After that change the number of input variables by 3 in this block in the call for the indicator initialization:

```
//--- object initialization
   if(!m_indicator.Create(m_symbol.Name(),m_Ind_Timeframe,IND_CUSTOM,3,parameters))
```

The number of indicator buffers in the indicator remains the same and equal to two, therefore, there is no need to change anything in the indicator buffers number initialization line in our case:

```
//--- number of buffers
   if(!m_indicator.NumBuffers(2))  return(false);
```

ASCtrend and BykovTrend indicators have two indicator buffers each. The functions of the buffers are completely similar. The zero buffer is used for storing sell signals, while the buffer having index 1 is used for storing buy signals. So, there is no need to change anything in the blocks of functions for delivering CBykovTrendSignal::LongCondition() and
CBykovTrendSignal::ShortCondition() trading signals and the work on the trading signals module modification may be considered complete.

But in general, all semaphore indicators are different and, therefore, these blocks for different semaphore indicators can differ from each other considerably. MySignals.zip trading module archive and the appropriate Indicators.zip archive contain sufficient amount of examples for creating various indicators. After some examination it is possible to find out the details of the replacement process and possible code versions for that.

Now, I would like to focus on Ind\_Timeframe input variable of the trading signals module. This variable allows to download an appropriate time frame to the indicator. However, the generated Expert Advisor operates on the time frame it was assigned to. It means that Ind\_Timeframe input variable time frame should never exceed a period of the chart the Expert Advisor operates on to provide the module normal operation.

Finally, I would like to reveal another peculiarity of creating trading signals modules. Sometimes custom enumerations are implemented into the basic indicator code as the types for the module input variables. For example, Smooth\_Method custom enumeration is used as MA\_SMethod variable type for Candles\_Smoothed indicator:

```
//+-----------------------------------+
//|  Declaration of enumerations      |
//+-----------------------------------+
enum Smooth_Method
  {
   MODE_SMA_,  // SMA
   MODE_EMA_,  // EMA
   MODE_SMMA_, // SMMA
   MODE_LWMA_, // LWMA
   MODE_JJMA,  // JJMA
   MODE_JurX,  // JurX
   MODE_ParMA, // ParMA
   MODE_T3,    // T3
   MODE_VIDYA, // VIDYA
   MODE_AMA,   // AMA
  }; */
//+----------------------------------------------+
//| Indicator input parameters                   |
//+----------------------------------------------+
input Smooth_Method MA_SMethod=MODE_LWMA; // Smoothing method
input int MA_Length=30;                   // Smoothing depth
input int MA_Phase=100;                   // Smoothing parameter
                                          // for JJMA varying within the range -100 ... +100,
                                          // for VIDIA it is a CMO period, for AMA it is a slow average period
//+----------------------------------------------+
```

In such case input variables of that kind and all associated elements in the trading signals module (Candles\_SmoothedSignal.mqh) should be modified into the variables of int or uint types. Also, the reverse procedure of custom enumerations up to the Expert Advisor input parameters and replacement of the necessary input variables types (ExpM\_Candles\_Smoothed Expert Advisor) should be carried out for the ease of use of this input variables in the already generated code of the finished Expert Advisor:

```
//+------------------------------------------------------------------+
//|  Declaration of enumerations                                     |
//+------------------------------------------------------------------+
enum Smooth_Method
  {
   MODE_SMA_,  // SMA
   MODE_EMA_,  // EMA
   MODE_SMMA_, // SMMA
   MODE_LWMA_, // LWMA
   MODE_JJMA,  // JJMA
   MODE_JurX,  // JurX
   MODE_ParMA, // ParMA
   MODE_T3,    // T3
   MODE_VIDYA, // VIDYA
   MODE_AMA,   // AMA
  };
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
//--- inputs for expert
input string          Expert_Title         ="Candles_Smoothed"; // Document name
ulong                 Expert_MagicNumber   =29976;              //
bool                  Expert_EveryTick     =false;              //
//--- inputs for main signal
input int             Signal_ThresholdOpen =40;                 // Signal threshold value to open [0...100]
input int             Signal_ThresholdClose=20;                 // Signal threshold value to close [0...100]
input double          Signal_PriceLevel    =0.0;                // Price level to execute a deal
input double          Signal_StopLevel     =50.0;               // Stop Loss level (in points)
input double          Signal_TakeLevel     =50.0;               // Take Profit level (in points)
input int             Signal_Expiration    =1;                  // Expiration of pending orders (in bars)
input bool            Signal__BuyPosOpen   =true;               // Candles_Smoothed() Permission to buy
input bool            Signal__SellPosOpen  =true;               // Candles_Smoothed() Permission to sell
input bool            Signal__BuyPosClose  =true;               // Candles_Smoothed() Permission to exit a long position
input bool            Signal__SellPosClose =true;               // Candles_Smoothed() Permission to exit a short position
input ENUM_TIMEFRAMES Signal__Ind_Timeframe=PERIOD_H1;            // Candles_Smoothed() Timeframe
input Smooth_Method   Signal__MA_SMethod   =4;                  // Candles_Smoothed() Smoothing method (1 - 10)
input uint            Signal__MA_Length    =30;                 // Candles_Smoothed() Smoothing depth
input uint            Signal__MA_Phase     =100;                // Candles_Smoothed() Smoothing parameter
input uint            Signal__SignalBar    =1;                  // Candles_Smoothed() Bar index for the entry signal
input double          Signal__Weight       =1.0;                // Candles_Smoothed() Weight [0...1.0]
//--- inputs for money
input double          Money_FixLot_Percent =10.0;               // Percent
input double          Money_FixLot_Lots    =0.1;                // Fixed volume
```

In our case this was done with Signal\_\_MA\_SMethod input variable.

You can accelerate code modification considerably, if you open both code versions (ASCtrendSignal.mqh and BykovTrendSignal.mqh) simultaneously in the editor (placing one on the left side and the other one on the right side) and compare both code versions carefully.

### Conclusion

I have placed sufficient amount of Expert Advisors based on the semaphore trading system in Experts.zip archive attached to this article to allow novice Expert Advisors creators to easily understand all features of writing such a code or at least work with ready-made Expert Advisors using quite popular indicators.

All attached Expert Advisors are additionally presented as trading modules for those who want to use the trading strategies generator as a base for their own trading systems. These modules are located in MySignals.zip, while the trading systems based on them can be found in Expertsez.zip. The indicators used in the Expert Advisors are placed in Indicators.zip. The paths for extracting the files are as follows:

- Experts.zip: "\\MQL5\\Experts\\";
- Expertsez.zip: "\\MQL5\\Experts\\";
- MySignals.zip: "\\MQL5\\Include\\Expert\\Signal\\MySignals\\";
- Indicators.zip: "\\MQL5\\Indicators\\";
- SmoothAlgorithms.mqh: "\\Include\\";
- TradeAlgorithms.mqh: "\\Include\\".


Restart MetaEditor, open the Navigator window, right click on the MQL5 label and select "Compile" in the pop-up menu.

SmoothAlgorithms.mqh file is necessary for compilation of some indicators from Indicators.zip, while TradeAlgorithms.mqh file is needed for compilation of all Expert Advisors from Experts.zip.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/358](https://www.mql5.com/ru/articles/358)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/358.zip "Download all attachments in the single ZIP archive")

[expertsez.zip](https://www.mql5.com/en/articles/download/358/expertsez.zip "Download expertsez.zip")(31.51 KB)

[smoothalgorithms.mqh](https://www.mql5.com/en/articles/download/358/smoothalgorithms.mqh "Download smoothalgorithms.mqh")(135.4 KB)

[mysignals.zip](https://www.mql5.com/en/articles/download/358/mysignals.zip "Download mysignals.zip")(44.08 KB)

[indicators.zip](https://www.mql5.com/en/articles/download/358/indicators.zip "Download indicators.zip")(39.15 KB)

[experts.zip](https://www.mql5.com/en/articles/download/358/experts.zip "Download experts.zip")(34.67 KB)

[tradealgorithms.mqh](https://www.mql5.com/en/articles/download/358/tradealgorithms.mqh "Download tradealgorithms.mqh")(66.5 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Averaging Price Series for Intermediate Calculations Without Using Additional Buffers](https://www.mql5.com/en/articles/180)
- [Creating an Indicator with Multiple Indicator Buffers for Newbies](https://www.mql5.com/en/articles/48)
- [Creating an Expert Advisor, which Trades on a Number of Instruments](https://www.mql5.com/en/articles/105)
- [The Principles of Economic Calculation of Indicators](https://www.mql5.com/en/articles/109)
- [Practical Implementation of Digital Filters in MQL5 for Beginners](https://www.mql5.com/en/articles/32)
- [Custom Indicators in MQL5 for Newbies](https://www.mql5.com/en/articles/37)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6326)**
(9)


![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
29 Oct 2019 at 23:27

Hello!

I am trying to add the ASCtrendSignal [trading signal module](https://www.mql5.com/en/articles/303 "Using Forward and Inverse Fisher Transforms to Analyse Markets in MetaTrader 5") (located in the compressed file mysignals.zip) created on the basis of the ASCtrend indicator (located in the compressed file indicators.zip) to the MQL5 Wizard, but nothing works. I place the ASCtrendSignal trading signal module in Include\\Expert\\Signal, and place the ASCtrend indicator in the Indicators package, everything seems to be correct, but the module persistently does not want to be displayed in the MQL5 Wizard. Here is the code of the ASCtrendSignal trading signal module:

```
//+------------------------------------------------------------------+
//|ASCtrendSignal.mqh |
//|Copyright © 2011, Nikolay Kositsin |
//|Khabarovsk, farria@mail.redcom.ru |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2011, Nikolay Kositsin."
#property link      "farria@mail.redcom.ru"
//+------------------------------------------------------------------+
//| Included files|
//+------------------------------------------------------------------+
#property tester_indicator "ASCtrend.ex5"
#include <Expert\ExpertSignal.mqh>
//--- wizard description start
//+------------------------------------------------------------------+
//| Declaration of constants|
//+------------------------------------------------------------------+
#define  OPEN_LONG     80  // The constant for returning the buy command to the Expert Advisor
#define  OPEN_SHORT    80  // The constant for returning the sell command to the Expert Advisor
#define  CLOSE_LONG    40  // The constant for returning the command to close a long position to the Expert Advisor
#define  CLOSE_SHORT   40  // The constant for returning the command to close a short position to the Expert Advisor
#define  REVERSE_LONG  100 // The constant for returning the command to reverse a long position to the Expert Advisor
#define  REVERSE_SHORT 100 // The constant for returning the command to reverse a short position to the Expert Advisor
#define  NO_SIGNAL      0  // The constant for returning the absence of a signal to the Expert Advisor
//+----------------------------------------------------------------------+
//| Description of the class|
//| Title=The signals based on ASCtrend indicator |
//| Type=SignalAdvanced.|
//| Name=ASCtrend|
//| Class=CASCtrendSignal|
//| Page=|
//| Parameter=BuyPosOpen,bool,true,Permission to buy |
//| Parameter=SellPosOpen,bool,true,Permission to sell |
//| Parameter=BuyPosClose,bool,true,Permission to exit a long position |
//| Parameter=SellPosClose,bool,true,Permission to exit a short position |
//| Parameter=Ind_Timeframe,ENUM_TIMEFRAMES,PERIOD_H4,Timeframe |
//| Parameter=RISK,int,4,Risk level|
//| Parameter=SignalBar,uint,1,Bar index for entry signal |
//+----------------------------------------------------------------------+
//--- wizard description end
//+----------------------------------------------------------------------+
//| CASCtrendSignal class.|
//| Purpose: Class of generator of trade signals based on |
//| ASCtrend indicator values http://www.mql5.com/ru/code/491/.&nbsp;         |
//| Is derived from the CExpertSignal class. ||
//+----------------------------------------------------------------------+
class CASCtrendSignal : public CExpertSignal
  {
protected:
   CiCustom          m_indicator;      // the object for access to ASCtrend values

   //--- adjusted parameters
   bool              m_BuyPosOpen;       // permission to buy
   bool              m_SellPosOpen;      // permission to sell
   bool              m_BuyPosClose;      // permission to exit a long position
   bool              m_SellPosClose;     // permission to exit a short position
   ENUM_TIMEFRAMES   m_Ind_Timeframe;    // ASCtrend indicator timeframe
   uint              m_RISK;             // Risk level
   uint              m_SignalBar;        // bar index for entry signal

public:
                     CASCtrendSignal();

   //--- methods of setting adjustable parameters
   void               BuyPosOpen(bool value)                  { m_BuyPosOpen=value;       }
   void               SellPosOpen(bool value)                 { m_SellPosOpen=value;      }
   void               BuyPosClose(bool value)                 { m_BuyPosClose=value;      }
   void               SellPosClose(bool value)                { m_SellPosClose=value;     }
   void               Ind_Timeframe(ENUM_TIMEFRAMES value)    { m_Ind_Timeframe=value;    }
   void               RISK(uint value)                        { m_RISK=value;             }
   void               SignalBar(uint value)                   { m_SignalBar=value;        }

   //--- adjustable parameters validation method
   virtual bool      ValidationSettings();
   //--- adjustable parameters validation method
   virtual bool      InitIndicators(CIndicators *indicators); // indicators initialisation
   //--- market entry signals generation method
   virtual int       LongCondition();
   virtual int       ShortCondition();

   bool              InitASCtrend(CIndicators *indicators);   // ASCtrend indicator initializing method

protected:

  };
//+------------------------------------------------------------------+
//|CASCtrendSignal constructor.|
//| INPUT: no.|
//| OUTPUT: no.|
//|| REMARK: no.|
//+------------------------------------------------------------------+
void CASCtrendSignal::CASCtrendSignal()
  {
//--- setting default parameters
   m_BuyPosOpen=true;
   m_SellPosOpen=true;
   m_BuyPosClose=true;
   m_SellPosClose=true;

//--- indicator input parameters
   m_Ind_Timeframe=PERIOD_H4;
   m_RISK=4;
//---
   m_SignalBar=1;
   m_used_series=USE_SERIES_OPEN+USE_SERIES_HIGH+USE_SERIES_LOW+USE_SERIES_CLOSE;
  }
//+------------------------------------------------------------------+
//| Checking adjustable parameters.|
//| INPUT: no.|
//| OUTPUT: true if the settings are valid, false - if not. ||
//|| REMARK: no.|
//+------------------------------------------------------------------+
bool CASCtrendSignal::ValidationSettings()
  {
//--- checking parameters
   if(m_RISK<=0)
     {
      printf(__FUNCTION__+": Risk level must be above zero");
      return(false);
     }

//--- successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialisation of indicators and time series. ||
//| INPUT: indicators - pointer to an object-collection |
//| of indicators and time series.|
//| OUTPUT: true - in case of successful, otherwise - false. || OUTPUT: true - in case of successful, otherwise - false. ||
//|| REMARK: no.|
//+------------------------------------------------------------------+
bool CASCtrendSignal::InitIndicators(CIndicators *indicators)
  {
//--- check of pointer
   if(indicators==NULL) return(false);

//--- indicator initialisation
   if(!InitASCtrend(indicators)) return(false);

//--- successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| ASCtrend indicator initialisation.|
//| INPUT: indicators - pointer to an object-collection |
//| of indicators and time series.|
//| OUTPUT: true - in case of successful, otherwise - false. || OUTPUT: true - in case of successful, otherwise - false. ||
//|| REMARK: no.|
//+------------------------------------------------------------------+
bool CASCtrendSignal::InitASCtrend(CIndicators *indicators)
  {
//--- check of pointer
   if(indicators==NULL) return(false);

//--- adding an object to the collection
   if(!indicators.Add(GetPointer(m_indicator)))
     {
      printf(__FUNCTION__+": error of adding the object");
      return(false);
     }

//--- setting the indicator parameters
   MqlParam parameters[2];

   parameters[0].type=TYPE_STRING;
   parameters[0].string_value="ASCtrend.ex5";

   parameters[1].type=TYPE_INT;
   parameters[1].integer_value=m_RISK;

//--- object initialisation
   if(!m_indicator.Create(m_symbol.Name(),m_Ind_Timeframe,IND_CUSTOM,2,parameters))
     {
      printf(__FUNCTION__+": object initialization error");
      return(false);
     }

//--- number of buffers
   if(!m_indicator.NumBuffers(2)) return(false);

//--- ASCtrend indicator initialized successfully
   return(true);
  }
//+------------------------------------------------------------------+
//| Checking conditions for opening a long position and |
//| and closing a short one|
//| INPUT:no|
//| OUTPUT: Vote weight from 0 to 100|
//|| REMARK: no.|
//+------------------------------------------------------------------+
int CASCtrendSignal::LongCondition()
  {
//--- buy signal is determined by buffer 1 of the ASCtrend indicator
   double Signal=m_indicator.GetData(1,m_SignalBar);

//--- getting a trading signal
   if(Signal && Signal!=EMPTY_VALUE)
     {
      if(m_BuyPosOpen)
        {
         if(m_SellPosClose) return(REVERSE_SHORT);
         else return(OPEN_LONG);
        }
      else
        {
         if(m_SellPosClose) return(CLOSE_SHORT);
        }
     }

//--- searching for signals for closing a short position
   if(!m_SellPosClose) return(NO_SIGNAL);

   int Bars_=Bars(m_symbol.Name(),m_Ind_Timeframe);

   for(int bar=int(m_SignalBar); bar<Bars_; bar++)
     {
      Signal=m_indicator.GetData(0,bar);
      if(Signal && Signal!=EMPTY_VALUE) return(NO_SIGNAL);

      Signal=m_indicator.GetData(1,bar);
      if(Signal && Signal!=EMPTY_VALUE) return(CLOSE_SHORT);
     }

//--- no trading signal
   return(NO_SIGNAL);
  }
//+------------------------------------------------------------------+
//| Checking conditions for opening a short position and |
//| closing a long one|
//| INPUT:no|
//| OUTPUT: Vote weight from 0 to 100|
//|| REMARK: no.|
//+------------------------------------------------------------------+
int CASCtrendSignal::ShortCondition()
  {
//--- sell signal is determined by buffer 0 of the ASCtrend indicator
   double Signal=m_indicator.GetData(0,m_SignalBar);

//--- getting a trading signal
   if(Signal && Signal!=EMPTY_VALUE)
     {
      if(m_SellPosOpen)
        {
         if(m_BuyPosClose) return(REVERSE_LONG);
         else return(OPEN_SHORT);
        }
      else
        {
         if(m_BuyPosClose) return(CLOSE_LONG);
        }
     }

//--- searching for signals for closing a long position
   if(!m_BuyPosClose) return(NO_SIGNAL);

   int Bars_=Bars(m_symbol.Name(),m_Ind_Timeframe); // Здесь код исправлен с учетом подсказки от Владимира Карпутова: Symbol() заменен на m_symbol.Name()
   for(int bar=int(m_SignalBar); bar<Bars_; bar++)
     {
      Signal=m_indicator.GetData(1,bar);
      if(Signal && Signal!=EMPTY_VALUE) return(NO_SIGNAL);

      Signal=m_indicator.GetData(0,bar);
      if(Signal && Signal!=EMPTY_VALUE) return(CLOSE_LONG);
     }

//--- no trading signal
   return(NO_SIGNAL);
  }
//+------------------------------------------------------------------+
```

Can you please tell me what the problem may be?

Regards, Vladimir

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
30 Oct 2019 at 08:32

**MrBrooklin:**

Hello!

I am trying to add the ASCtrendSignal [trading signal module](https://www.mql5.com/en/articles/303 "Using Forward and Inverse Fisher Transforms to Analyse Markets in MetaTrader 5") (located in the compressed file mysignals.zip) created on the basis of the ASCtrend indicator (located in the compressed file indicators.zip) to the MQL5 Wizard, but nothing works. I place the ASCtrendSignal trading signal module in Include\\Expert\\Signal, and place the ASCtrend indicator in the Indicators package, everything seems to be correct, but the module persistently does not want to be displayed in the MQL5 Wizard. Here is the code of the ASCtrendSignal trading signal module:

Can you please tell me what the problem might be?

Regards, Vladimir

I would like to add that this same problem occurs with other modules of trading signals written on the basis of indicators. Apparently, they have the same problem. Please help in solving these problems.

Regards, Vladimir.

P.S. Some **indicators** that I unpacked from the compressed file **indicators.zip** are installed on the terminal and work normally.

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
30 Oct 2019 at 08:50

It's the order that counts:

```
// wizard description start
//+----------------------------------------------------------------------+
//| Description of the class|
//| Title=The signals based on ASCtrend indicator |
//| Type=SignalAdvanced.|
//| Name=ASCtrend|
//| Class=CASCtrendSignal|
//| Page=|
//| Parameter=BuyPosOpen,bool,true,Permission to buy |
//| Parameter=SellPosOpen,bool,true,Permission to sell |
//| Parameter=BuyPosClose,bool,true,Permission to exit a long position |
//| Parameter=SellPosClose,bool,true,Permission to exit a short position |
//| Parameter=Ind_Timeframe,ENUM_TIMEFRAMES,PERIOD_H4,Timeframe |
//| Parameter=RISK,int,4,Risk level|
//| Parameter=SignalBar,uint,1,Bar index for entry signal |
//+----------------------------------------------------------------------+
// wizard description end
```

not

```
//--- wizard description start
//--- wizard description end
```

and between start and end only a service block - no variables or macro substitutions.

This is how the module should start:

```
//+------------------------------------------------------------------+
//|ASCtrendSignal.mqh |
//|Copyright © 2011, Nikolay Kositsin |
//|Khabarovsk, farria@mail.redcom.ru |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2011, Nikolay Kositsin."
#property link      "farria@mail.redcom.ru"
//+------------------------------------------------------------------+
//| Included files|
//+------------------------------------------------------------------+
//#property tester_indicator "ASCtrend.ex5"
#include <Expert\ExpertSignal.mqh>
// wizard description start
//+----------------------------------------------------------------------+
//| Description of the class|
//| Title=The signals based on ASCtrend indicator |
//| Type=SignalAdvanced.|
//| Name=ASCtrend|
//| Class=CASCtrendSignal|
//| Page=|
//| Parameter=BuyPosOpen,bool,true,Permission to buy |
//| Parameter=SellPosOpen,bool,true,Permission to sell |
//| Parameter=BuyPosClose,bool,true,Permission to exit a long position |
//| Parameter=SellPosClose,bool,true,Permission to exit a short position |
//| Parameter=Ind_Timeframe,ENUM_TIMEFRAMES,PERIOD_H4,Timeframe |
//| Parameter=RISK,int,4,Risk level|
//| Parameter=SignalBar,uint,1,Bar index for entry signal |
//+----------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//| Declaration of constants|
//+------------------------------------------------------------------+
#define  OPEN_LONG     80  // The constant for returning the buy command to the Expert Advisor
#define  OPEN_SHORT    80  // The constant for returning the sell command to the Expert Advisor
#define  CLOSE_LONG    40  // The constant for returning the command to close a long position to the Expert Advisor
#define  CLOSE_SHORT   40  // The constant for returning the command to close a short position to the Expert Advisor
#define  REVERSE_LONG  100 // The constant for returning the command to reverse a long position to the Expert Advisor
#define  REVERSE_SHORT 100 // The constant for returning the command to reverse a short position to the Expert Advisor
#define  NO_SIGNAL      0  // The constant for returning the absence of a signal to the Expert Advisor
//+----------------------------------------------------------------------+
//| CASCtrendSignal class.|
//| Purpose: Class of generator of trade signals based on |
//| ASCtrend indicator values http://www.mql5.com/ru/code/491/.&nbsp;         |
//| Is derived from the CExpertSignal class. ||
//+----------------------------------------------------------------------+
class CASCtrendSignal : public CExpertSignal
  {
```

reload MetaEditor after making edits

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
30 Oct 2019 at 09:10

**Vladimir Karputov:**

It's the order that counts:

not

and between start and end only a service block - no variables or macro substitutions.

This is how a module should start:

reload MetaEditor after making edits

Thank you, Vladimir!

Everything worked.

Regards, Vladimir.

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
1 Nov 2019 at 09:46

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/ru/forum)

[Discussion of the article "Simple trading systems using semaphore indicators"](https://www.mql5.com/ru/forum/5891#comment_13717891)

[Vladimir Karputov](https://www.mql5.com/ru/users/barabashkakvn) , 2019.10.30 08:50

The order is important:

```
 // wizard description start
//+----------------------------------------------------------------------+
//| Description of the class                                             |
//| Title=The signals based on ASCtrend indicator                        |
//| Type=SignalAdvanced                                                  |
//| Name=ASCtrend                                                        |
//| Class=CASCtrendSignal                                                |
//| Page=                                                                |
//| Parameter=BuyPosOpen,bool,true,Permission to buy                     |
//| Parameter=SellPosOpen,bool,true,Permission to sell                   |
//| Parameter=BuyPosClose,bool,true,Permission to exit a long position   |
//| Parameter=SellPosClose,bool,true,Permission to exit a short position |
//| Parameter=Ind_Timeframe,ENUM_TIMEFRAMES,PERIOD_H4,Timeframe          |
//| Parameter=RISK,int,4,Risk level                                      |
//| Parameter=SignalBar,uint,1,Bar index for entry signal                |
//+----------------------------------------------------------------------+
// wizard description end
```

but not

```
 //--- wizard description start
//--- wizard description end
```

and between start and end only the service block - no variables and macro substitutions.

This is the beginning the module should have:

```
 //+------------------------------------------------------------------+
//|                                               ASCtrendSignal.mqh |
//|                             Copyright © 2011,   Nikolay Kositsin |
//|                              Khabarovsk,   farria@mail.redcom.ru |
//+------------------------------------------------------------------+
#property  copyright "Copyright © 2011, Nikolay Kositsin"
#property  link        "farria@mail.redcom.ru"
//+------------------------------------------------------------------+
//| Included files                                                   |
//+------------------------------------------------------------------+
//#property tester_indicator "ASCtrend.ex5"
#include  <Expert\ExpertSignal.mqh>
// wizard description start
//+----------------------------------------------------------------------+
//| Description of the class                                             |
//| Title=The signals based on ASCtrend indicator                        |
//| Type=SignalAdvanced                                                  |
//| Name=ASCtrend                                                        |
//| Class=CASCtrendSignal                                                |
//| Page=                                                                |
//| Parameter=BuyPosOpen,bool,true,Permission to buy                     |
//| Parameter=SellPosOpen,bool,true,Permission to sell                   |
//| Parameter=BuyPosClose,bool,true,Permission to exit a long position   |
//| Parameter=SellPosClose,bool,true,Permission to exit a short position |
//| Parameter=Ind_Timeframe,ENUM_TIMEFRAMES,PERIOD_H4,Timeframe          |
//| Parameter=RISK,int,4,Risk level                                      |
//| Parameter=SignalBar,uint,1,Bar index for entry signal                |
//+----------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//|  Declaration of constants                                        |
//+------------------------------------------------------------------+
#define  OPEN_LONG     80    // The constant for returning the buy command to the Expert Advisor
#define  OPEN_SHORT     80    // The constant for returning the sell command to the Expert Advisor
#define  CLOSE_LONG     40    // The constant for returning the command to close a long position to the Expert Advisor
#define  CLOSE_SHORT   40    // The constant for returning the command to close a short position to the Expert Advisor
#define  REVERSE_LONG   100 // The constant for returning the command to reverse a long position to the Expert Advisor
#define  REVERSE_SHORT 100 // The constant for returning the command to reverse a short position to the Expert Advisor
#define  NO_SIGNAL       0    // The constant for returning the absence of a signal to the Expert Advisor
//+----------------------------------------------------------------------+
//| CASCtrendSignal class.                                               |
//| Purpose: Class of generator of trade signals based on                |
//| ASCtrend indicator values http://www.mql5.com/ru/code/491/.&nbsp;         |
//| Is derived from the CExpertSignal class.                             |
//+----------------------------------------------------------------------+
class CASCtrendSignal : public CExpertSignal
  {
```

restart the MetaEditor after making changes

![The Box-Cox Transformation](https://c.mql5.com/2/0/Cox-Box-transformation_MQL5.png)[The Box-Cox Transformation](https://www.mql5.com/en/articles/363)

The article is intended to get its readers acquainted with the Box-Cox transformation. The issues concerning its usage are addressed and some examples are given allowing to evaluate the transformation efficiency with random sequences and real quotes.

![Multiple Regression Analysis. Strategy Generator and Tester in One](https://c.mql5.com/2/0/Multiple_Regression_Analysis_MQL5.png)[Multiple Regression Analysis. Strategy Generator and Tester in One](https://www.mql5.com/en/articles/349)

The article gives a description of ways of use of the multiple regression analysis for development of trading systems. It demonstrates the use of the regression analysis for strategy search automation. A regression equation generated and integrated in an EA without requiring high proficiency in programming is given as an example.

![On Methods of Technical Analysis and Market Forecasting](https://c.mql5.com/2/17/982_30.gif)[On Methods of Technical Analysis and Market Forecasting](https://www.mql5.com/en/articles/1350)

The article demonstrates the capabilities and potential of a well-known mathematical method coupled with visual thinking and an "out of the box" market outlook. On the one hand, it serves to attract the attention of a wide audience as it can get the creative minds to reconsider the trading paradigm as such. And on the other, it can give rise to alternative developments and program code implementations regarding a wide range of tools for analysis and forecasting.

![Time Series Forecasting Using Exponential Smoothing (continued)](https://c.mql5.com/2/0/Exponent_Smoothing2.png)[Time Series Forecasting Using Exponential Smoothing (continued)](https://www.mql5.com/en/articles/346)

This article seeks to upgrade the indicator created earlier on and briefly deals with a method for estimating forecast confidence intervals using bootstrapping and quantiles. As a result, we will get the forecast indicator and scripts to be used for estimation of the forecast accuracy.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=viknndhvxxqieznpihznpeonwcxkxfkr&ssn=1769158290222792409&ssn_dr=0&ssn_sr=0&fv_date=1769158290&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F358&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Simple%20Trading%20Systems%20Using%20Semaphore%20Indicators%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915829077433891&fz_uniq=5062770581151066258&sv=2552)

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