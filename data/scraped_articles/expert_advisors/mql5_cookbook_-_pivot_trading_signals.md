---
title: MQL5 Cookbook - Pivot trading signals
url: https://www.mql5.com/en/articles/2853
categories: Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:50:59.487407
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/2853&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068719420093693137)

MetaTrader 5 / Examples


### Introduction

The current article continues the series describing indicators and setups that generate trading signals. This time, we will have a look at the pivots — reversal levels (points). We will apply the [Standard Library](https://www.mql5.com/en/docs/standardlibrary) again. First, we will consider the reversal level indicator, develop a basic strategy based on it and finally search for the means to improve it.

It is assumed that the reader is familiar with the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) base class for developing trading signal generators.

### 1\. Pivot (reversal level) indicator

For this strategy, we will use the indicator plotting potential reversal levels. Plotting is performed by means of graphical construction only. No graphical objects are applied. The main advantage of this approach is the ability to refer to the indicator in the optimization mode. On the other hand, graphical constructions cannot exceed the indicator buffers meaning there will be no lines in the future.

Levels can be counted in several different ways. Further information on this subject is available in the article ["Trading strategy based on pivot points analysis"](https://www.mql5.com/en/articles/1465).

Let's consider the standard approach for now (the levels are defined using the following equations):

![](https://c.mql5.com/2/25/2__10.png)

RES is an ith resistance level, while SUP is an ith support level. In total, there will be 1 main reversal level (PP), 6 resistance (RES) and 6 support levels (SUP).

So, visually the indicator looks like a set of horizontal levels plotted at different prices. When launched on the chart for the first time, the indicator draws levels for the current day only (Fig.1).

![Fig.1. Pivot indicator: plotting for the current day](https://c.mql5.com/2/25/1__14.png)

Fig.1.Pivot indicator: plotting for the current day

Let's examine the indicator code block by block beginning with the calculation one.

When a new day begins, we need to count all reversal levels.

//\-\-\- in case of a new day

if(gNewDay.isNewBar(today))

      {

PrintFormat("New day: %s",TimeToString(today));

//\-\-\- normalize prices

double d\_high=NormalizeDouble(daily\_rates\[0\].high,\_Digits);

double d\_low=NormalizeDouble(daily\_rates\[0\].low,\_Digits);

double d\_close=NormalizeDouble(daily\_rates\[0\].close,\_Digits);

//\-\-\- save prices

       gYesterdayHigh=d\_high;

       gYesterdayLow=d\_low;

       gYesterdayClose=d\_close;

//\-\-\- 1) pivot: PP = (HIGH + LOW + CLOSE) / 3

       gPivotVal=NormalizeDouble((gYesterdayHigh+gYesterdayLow+gYesterdayClose)/3.,\_Digits);

//\-\-\- 4) RES1.0 = 2\*PP - LOW

       gResVal\_1\_0=NormalizeDouble(2.\*gPivotVal-gYesterdayLow,\_Digits);

//\-\-\- 5) SUP1.0 = 2\*PP – HIGH

       gSupVal\_1\_0=NormalizeDouble(2.\*gPivotVal-gYesterdayHigh,\_Digits);

//\-\-\- 8) RES2.0 = PP + (HIGH -LOW)

       gResVal\_2\_0=NormalizeDouble(gPivotVal+(gYesterdayHigh-gYesterdayLow),\_Digits);

//\-\-\- 9) SUP2.0 = PP - (HIGH – LOW)

       gSupVal\_2\_0=NormalizeDouble(gPivotVal-(gYesterdayHigh-gYesterdayLow),\_Digits);

//\-\-\- 12) RES3.0 = 2\*PP + (HIGH – 2\*LOW)

       gResVal\_3\_0=NormalizeDouble(2.\*gPivotVal+(gYesterdayHigh-2.\*gYesterdayLow),\_Digits);

//\-\-\- 13) SUP3.0 = 2\*PP - (2\*HIGH – LOW)

       gSupVal\_3\_0=NormalizeDouble(2.\*gPivotVal-(2.\*gYesterdayHigh-gYesterdayLow),\_Digits);

//\-\-\- 2) RES0.5 = (PP + RES1.0) / 2

       gResVal\_0\_5=NormalizeDouble((gPivotVal+gResVal\_1\_0)/2.,\_Digits);

//\-\-\- 3) SUP0.5 = (PP + SUP1.0) / 2

       gSupVal\_0\_5=NormalizeDouble((gPivotVal+gSupVal\_1\_0)/2.,\_Digits);

//\-\-\- 6) RES1.5 = (RES1.0 + RES2.0) / 2

       gResVal\_1\_5=NormalizeDouble((gResVal\_1\_0+gResVal\_2\_0)/2.,\_Digits);

//\-\-\- 7) SUP1.5 = (SUP1.0 + SUP2.0) / 2

       gSupVal\_1\_5=NormalizeDouble((gSupVal\_1\_0+gSupVal\_2\_0)/2.,\_Digits);

//\-\-\- 10) RES2.5 = (RES2.0 + RES3.0) / 2

       gResVal\_2\_5=NormalizeDouble((gResVal\_2\_0+gResVal\_3\_0)/2.,\_Digits);

//\-\-\- 11) SUP2.5 = (SUP2.0 + SUP3.0) / 2

       gSupVal\_2\_5=NormalizeDouble((gSupVal\_2\_0+gSupVal\_3\_0)/2.,\_Digits);

//\-\-\- current day start bar

       gDayStart=today;

//\-\-\- find the start bar of the active TF

//\-\-\- as a time series

for(int bar=0;bar<rates\_total;bar++)

         {

//\-\-\- selected bar time

datetime curr\_bar\_time=time\[bar\];

          user\_date.DateTime(curr\_bar\_time);

//\-\-\- selected bar day

datetime curr\_bar\_time\_of\_day=user\_date.DateOfDay();

//\-\-\- if the current bar was the day before

if(curr\_bar\_time\_of\_day<gDayStart)

            {

//\-\-\- save the start bar

             gBarStart=bar-1;

break;

            }

         }

//\-\-\- reset the local counter

       prev\_calc=0;

      }

The red colorhighlights the strings where the levels are re-calculated. Next, we should find the bar for the current timeframe to be used as a starting point for plotting levels. Its value is defined by the **gBarStart** variable. The **SUserDateTime** custom structure (the descendant of the **CDateTime** structure) is used during the search for working with dates and time.

Now, let's focus our attention on the block designed for filling buffer values for the current timeframe bars.

//\-\-\- if the new bar is on the active TF

if(gNewMinute.isNewBar(time\[0\]))

      {

//\-\-\- bar, up to which the calculation is performed

int bar\_limit=gBarStart;

//\-\-\- if this is not the first launch

if(prev\_calc>0)

          bar\_limit=rates\_total-prev\_calc;

//\-\-\- calculate the buffers

for(int bar=0;bar<=bar\_limit;bar++)

         {

//\-\-\- 1) pivot

          gBuffers\[0\].data\[bar\]=gPivotVal;

//\-\-\- 2) RES0.5

if(gToPlotBuffer\[1\])

             gBuffers\[1\].data\[bar\]=gResVal\_0\_5;

//\-\-\- 3) SUP0.5

if(gToPlotBuffer\[2\])

             gBuffers\[2\].data\[bar\]=gSupVal\_0\_5;

//\-\-\- 4) RES1.0

if(gToPlotBuffer\[3\])

             gBuffers\[3\].data\[bar\]=gResVal\_1\_0;

//\-\-\- 5) SUP1.0

if(gToPlotBuffer\[4\])

             gBuffers\[4\].data\[bar\]=gSupVal\_1\_0;

//\-\-\- 6) RES1.5

if(gToPlotBuffer\[5\])

             gBuffers\[5\].data\[bar\]=gResVal\_1\_5;

//\-\-\- 7) SUP1.5

if(gToPlotBuffer\[6\])

             gBuffers\[6\].data\[bar\]=gSupVal\_1\_5;

//\-\-\- 8) RES2.0

if(gToPlotBuffer\[7\])

             gBuffers\[7\].data\[bar\]=gResVal\_2\_0;

//\-\-\- 9) SUP2.0

if(gToPlotBuffer\[8\])

             gBuffers\[8\].data\[bar\]=gSupVal\_2\_0;

//\-\-\- 10) RES2.5

if(gToPlotBuffer\[9\])

             gBuffers\[9\].data\[bar\]=gResVal\_2\_5;

//\-\-\- 11) SUP2.5

if(gToPlotBuffer\[10\])

             gBuffers\[10\].data\[bar\]=gSupVal\_2\_5;

//\-\-\- 12) RES3.0

if(gToPlotBuffer\[11\])

             gBuffers\[11\].data\[bar\]=gResVal\_3\_0;

//\-\-\- 13) SUP3.0

if(gToPlotBuffer\[12\])

             gBuffers\[12\].data\[bar\]=gSupVal\_3\_0;

         }

      }

Calculation of buffers begins when a new bar appears on the chart the indicator is launched at. The yellow color highlights the definition of the bar number, up to which the buffers are calculated. The local counter of calculated bars is used for that. We need it because the beginning of a new day does not reset the **prev\_calculated** constant value to zero, although such a reset is necessary.

The full code of the pivot indicator can be found in the **Pivots.mq5** file.

### 2\. Basic strategy

Let's develop a simple basic strategy based on the described indicator. Let the open signal depend on the Open price location relative to the central pivot. The price touching the pivot level serves as a signal confirmation.

The EURUSD M15 chart (Fig.2) displays the day (January 15, 2015) Open level below the central pivot. However, later during the day, the price touches the pivot level upwards. Thus, there is a sell signal. If neither stop loss nor take profit are activated, the market exit is performed at the beginning of the next day.

![Fig.2. Basic strategy: sell signal](https://c.mql5.com/2/27/3__8.png)

Fig.2. Basic strategy: sell signal

Stop levels are bound to the pivot indicator reversal levels. The intermediate resistance level Res0.5 at $1.18153 serves as a stop loss when selling. The main support level Sup1.0 at $1.17301 is used as a take profit. We will return to the trading day of January 14 later. In the meantime, let's have a look at the code that is to form the essence of the basic strategy.

**2.1** **CSignalPivots signal class**

Let's create a signal class that will generate signals from various patterns formed on the basis of price dynamics and the reversal levels indicator.

//+------------------------------------------------------------------+

//\| Class CSignalPivots                                              \|

//\| Purpose: Class of trading signals based on pivots.               \|

//\| CExpertSignal class descendant.                                  \|

//+------------------------------------------------------------------+

class CSignalPivots : public CExpertSignal

{

//\-\-\- === Data members === ---

protected:

    CiCustom          m\_pivots;            // "Pivots" indicator object

//\-\-\- adjustable parameters

bool              m\_to\_plot\_minor;     // plot secondary levels

double            m\_pnt\_near;          // tolerance

//\-\-\- estimated

double            m\_pivot\_val;         // pivot value

double            m\_daily\_open\_pr;     // current day Open price

    CisNewBar         m\_day\_new\_bar;       // new bar of the daily TF

//\-\-\- market patterns

//\-\-\- 1) Pattern 0 "first touch of the PP level" (top - buy, bottom - sell)

int               m\_pattern\_0;         // weight

bool              m\_pattern\_0\_done;    // sign that a pattern is over

//\-\-\- === Methods === ---

public:

//\-\-\- constructor/destructor

void              CSignalPivots(void);

void             ~CSignalPivots(void){};

//\-\-\- methods of setting adjustable parameters

void              ToPlotMinor(constbool \_to\_plot) {m\_to\_plot\_minor=\_to\_plot;}

void              PointsNear(constuint \_near\_pips);

//\-\-\- methods of adjusting "weights" of market models

void              Pattern\_0(int \_val) {m\_pattern\_0=\_val;m\_pattern\_0\_done=false;}

//\-\-\- method of verification of settings

virtualbool      ValidationSettings(void);

//\-\-\- method of creating the indicator and time series

virtualbool      InitIndicators(CIndicators \*indicators);

//\-\-\- methods of checking if the market models are generated

virtualint       LongCondition(void);

virtualint       ShortCondition(void);

virtualdouble    Direction(void);

//\-\-\- methods for detection of levels of entering the market

virtualbool      OpenLongParams(double &price,double &sl,double &tp,datetime &expiration);

virtualbool      OpenShortParams(double &price,double &sl,double &tp,datetime &expiration);

//---

protected:

//\-\-\- method of the indicator initialization

bool              InitCustomIndicator(CIndicators \*indicators);

//\-\-\- get the pivot level value

double            Pivot(void) {return(m\_pivots.GetData(0,0));}

//\-\-\- get the main resistance level value

double            MajorResistance(uint \_ind);

//\-\-\- get the secondary resistance level value

double            MinorResistance(uint \_ind);

//\-\-\- get the main support level value

double            MajorSupport(uint \_ind);

//\-\-\- get the secondary support level value

double            MinorSupport(uint \_ind);

};

//+------------------------------------------------------------------+

I already used that approach in the article ["MQL5 Cookbook - Trading signals of moving channels"](https://www.mql5.com/en/articles/1863): the price touching a line is confirmed when the price falls into the line area. The **m\_pnt\_near** data member sets the tolerance for a reversal level.

The signal pattern served by the class plays the most important role. The base class is to have a single pattern. Apart from the weight ( **m\_pattern\_0**), it also has a completion property within a trading day ( **m\_pattern\_0\_done**).

The **CExpertSignal** base signal class is rich in virtual methods. This allows for implementing fine-tuning of the derived class.

In particular, I have re-defined the **OpenLongParams()** and **OpenShortParams()** methods for calculating trading levels.

Let's examine the code of the first method — defining values for trading levels when buying.

//+------------------------------------------------------------------+

//\| Define trading levels when buying                                \|

//+------------------------------------------------------------------+

bool CSignalPivots::OpenLongParams(double &price,double &sl,double &tp,datetime &expiration)

{

bool params\_set=false;

    sl=tp=WRONG\_VALUE;

//\-\-\- if the Pattern 0 is considered

if(IS\_PATTERN\_USAGE(0))

//\-\-\- if the Pattern 0 is not complete

if(!m\_pattern\_0\_done)

         {

//\-\-\- Open price - market

double base\_price=m\_symbol.Ask();

          price=m\_symbol.NormalizePrice(base\_price-m\_price\_level\*PriceLevelUnit());

//\-\-\- sl price - Sup0.5 level

          sl=this.MinorSupport(0);

if(sl==DBL\_MAX)

returnfalse;

//\-\-\- if sl price is set

          sl=m\_symbol.NormalizePrice(sl);

//\-\-\- tp price - Res1.0 level

          tp=this.MajorResistance(0);

if(tp==DBL\_MAX)

returnfalse;

//\-\-\- if tp price is set

          tp=m\_symbol.NormalizePrice(tp);

          expiration+=m\_expiration\*PeriodSeconds(m\_period);

//\-\-\- if prices are set

          params\_set=true;

//\-\-\- pattern complete

          m\_pattern\_0\_done=true;

         }

//---

return params\_set;

}

//+------------------------------------------------------------------+

The stop loss price is calculated as the value of the first secondary support level using the **MinorSupport()** method. The profit is set at the price of the first main resistance level using the **MajorResistance()** method. In case of selling, the methods are replaced with **MinorResistance()** and **MajorSupport()** accordingly.

Make the custom signal the main one to let the methods for defining trading levels work properly. Here is how the method for defining the parent class trading levels looks like:

//+------------------------------------------------------------------+

//\| Detecting the levels for buying                                  \|

//+------------------------------------------------------------------+

bool CExpertSignal::OpenLongParams(double &price,double &sl,double &tp,datetime &expiration)

{

CExpertSignal \*general=(m\_general!=-1) ? m\_filters.At(m\_general) : NULL;

//---

if(general==NULL)

      {

//\-\-\- if a base price is not specified explicitly, take the current market price

double base\_price=(m\_base\_price==0.0) ? m\_symbol.Ask() : m\_base\_price;

       price      =m\_symbol.NormalizePrice(base\_price-m\_price\_level\*PriceLevelUnit());

       sl         =(m\_stop\_level==0.0) ? 0.0 : m\_symbol.NormalizePrice(price-m\_stop\_level\*PriceLevelUnit());

       tp         =(m\_take\_level==0.0) ? 0.0 : m\_symbol.NormalizePrice(price+m\_take\_level\*PriceLevelUnit());

       expiration+=m\_expiration\*PeriodSeconds(m\_period);

return(true);

      }

//---

return(general.OpenLongParams(price,sl,tp,expiration));

}

//+------------------------------------------------------------------+

If no main signal index is set, the levels receive default values. In order to avoid this, set the following in the EA code when initializing the signal:

//\-\-\- CSignalPivots filter

    CSignalPivots \*filter0=new CSignalPivots;

if(filter0==NULL)

      {

//\-\-\- error

PrintFormat(\_\_FUNCTION\_\_+": error creating filter0");

returnINIT\_FAILED;

      }

    signal.AddFilter(filter0);

signal.General(0);

The buy condition verification method is present as follows:

//+------------------------------------------------------------------+

//\| Check the buy condition                                          \|

//+------------------------------------------------------------------+

int CSignalPivots::LongCondition(void)

{

int result=0;

//\-\-\- if the Pattern 0 is not considered

if(IS\_PATTERN\_USAGE(0))

//\-\-\- if the Pattern 0 is not complete

if(!m\_pattern\_0\_done)

//\-\-\- if a day has opened above the pivot

if(m\_daily\_open\_pr>m\_pivot\_val)

            {

//\-\-\- minimum price on the current bar

double last\_low=m\_low.GetData(1);

//\-\-\- if the price is received

if((last\_low>WRONG\_VALUE) && (last\_low<DBL\_MAX))

//\-\-\- if there was a touch from above (considering the tolerance)

if(last\_low<=(m\_pivot\_val+m\_pnt\_near))

                  {

                   result=m\_pattern\_0;

//\-\-\- to the Journal

Print("\\n---== The price touches the pivot level from above ==---");

PrintFormat("Price: %0."+IntegerToString(m\_symbol.Digits())+"f",last\_low);

PrintFormat("Pivot: %0."+IntegerToString(m\_symbol.Digits())+"f",m\_pivot\_val);

PrintFormat("Tolerance: %0."+IntegerToString(m\_symbol.Digits())+"f",m\_pnt\_near);

                  }

            }

//---

return result;

}

//+------------------------------------------------------------------+

It is easy to see that the touch from above is checked considering the tolerance last\_low<=(m\_pivot\_val+m\_pnt\_near).

Apart from other things, the **Direction() method for defining the "weighted" direction** checksif the basic pattern is complete.

//+------------------------------------------------------------------+

//\| Define the "weighted" direction                                  \|

//+------------------------------------------------------------------+

double CSignalPivots::Direction(void)

{

double result=0.;

//\-\-\- receive daily history data

MqlRates daily\_rates\[\];

if(CopyRates(\_Symbol,PERIOD\_D1,0,1,daily\_rates)<0)

return0.;

//\-\-\- if the Pattern 0 is complete

if(m\_pattern\_0\_done)

      {

//\-\-\- check for a new day

if(m\_day\_new\_bar.isNewBar(daily\_rates\[0\].time))

         {

//\-\-\- reset the pattern completion flag

          m\_pattern\_0\_done=false;

return0.;

         }

      }

//\-\-\- if the Pattern 0 is not complete

else

      {

//\-\-\- day Open price

if(m\_daily\_open\_pr!=daily\_rates\[0\].open)

          m\_daily\_open\_pr=daily\_rates\[0\].open;

//\-\-\- pivot

double curr\_pivot\_val=this.Pivot();

if(curr\_pivot\_val<DBL\_MAX)

if(m\_pivot\_val!=curr\_pivot\_val)

             m\_pivot\_val=curr\_pivot\_val;

      }

//\-\-\- result

    result=m\_weight\*(this.LongCondition()-this.ShortCondition());

//---

return result;

}

//+------------------------------------------------------------------+

As for exit signals, re-define the parent class methods **CloseLongParams()** and **CloseShortParams()**. Sample buy block code:

//+------------------------------------------------------------------+

//\| Define trading level when buying                                 \|

//+------------------------------------------------------------------+

bool CSignalPivots::CloseLongParams(double &price)

{

    price=0.;

//\-\-\- if the Pattern 0 is considered

if(IS\_PATTERN\_USAGE(0))

//\-\-\- if the Pattern 0 is not complete

if(!m\_pattern\_0\_done)

         {

          price=m\_symbol.Bid();

//\-\-\- to the Journal

Print("\\n---== Signal to close buy ==---");

PrintFormat("Market price: %0."+IntegerToString(m\_symbol.Digits())+"f",price);

returntrue;

         }

//\-\-\- return the result

returnfalse;

}

//+------------------------------------------------------------------+

The exit signal threshold should be reset to zero in the EA code.

signal.ThresholdClose(0);

No direction check is performed in that case.

//+------------------------------------------------------------------+

//\| Generating a signal for closing of a long position               \|

//+------------------------------------------------------------------+

bool CExpertSignal::CheckCloseLong(double &price)

{

bool   result   =false;

//\-\-\- the "prohibition" signal

if(m\_direction==EMPTY\_VALUE)

return(false);

//\-\-\- check of exceeding the threshold value

if(-m\_direction>=m\_threshold\_close)

      {

//\-\-\- there's a signal

       result=true;

//\-\-\- try to get the level of closing

if(!CloseLongParams(price))

          result=false;

      }

//\-\-\- zeroize the base price

    m\_base\_price=0.0;

//\-\-\- return the result

return(result);

}

//+------------------------------------------------------------------+

The question arises: How is the exit signal checked in that case? First, it is checked by the presence of a position (in the **Processing() method**), and second, using the **m\_pattern\_0\_done** property (in the redefined**CloseLongParams()** and **CloseShortParams()** methods). As soon as the EA detects a position while the Pattern 0 is incomplete, it attempts to close it at once. This happens at the beginning of a trading day.

We have examined the basics of the **CSignalPivots** signal class. Now, let's dwell on the strategy class.

**2.2** **CPivotsExpert trading strategy class**

The derived strategy class is similar to the one for moving channels. The first difference is that minute-by-minute trading mode is used instead of tick-by-tick one. This allows you to quickly test the strategy on a fairly deep history. Second, the check for exit is present. We have already defined when the EA can close a position.

The main handler method looks as follows:

//+------------------------------------------------------------------+

//\| Main module                                                      \|

//+------------------------------------------------------------------+

bool CPivotsExpert::Processing(void)

{

//\-\-\- new minute bar

if(!m\_minute\_new\_bar.isNewBar())

returnfalse;

//\-\-\- calculate direction

    m\_signal.SetDirection();

//\-\-\- if there is no position

if(!this.SelectPosition())

      {

//\-\-\- position opening module

if(this.CheckOpen())

returntrue;

      }

//\-\-\- if there is a position

else

      {

//\-\-\- position closing module

if(this.CheckClose())

returntrue;

      }

//\-\-\- if there are no trade operations

returnfalse;

}

//+------------------------------------------------------------------+

That's it. Now, we may launch the basic strategy. Its code is presented in the **BasePivotsTrader.mq5** file.

![Fig.3. Basic strategy: sell](https://c.mql5.com/2/25/4__4.png)

Fig.3. Basic strategy: sell

Let's get back to the day of January 14, 2015. In this case, the model worked out perfectly. We opened short on the pivot and closed on the main support level Sup1.0.

The run was made in the strategy tester from 07.01.2013 to 07.01.2017 on EURUSD M15 with the following parameters:

- Entry signal threshold, \[0...100\] = 10;
- Weight, \[0...1.0\] = 1,0;
- Fixed volume = 0,1;
- Tolerance, points = 15.


As it turns out, the strategy trades with a steady result. A negative one (Fig. 4).

![Fig.4. EURUSD: Results of the first basic strategy for 2013-2016](https://c.mql5.com/2/27/4__8.png)

Fig.4.EURUSD: Results of the first basic strategy for 2013-2016

Judging by the results, we did everything wrong. We should have bought at a sell signal and sold at a buy one. But is it true? Let's check. To do this, we should develop a basic strategy and implement changes in the signals. In this case, a buy condition will look as follows:

//+------------------------------------------------------------------+

//\| Check condition for selling                                      \|

//+------------------------------------------------------------------+

int CSignalPivots::LongCondition(void)

{

int result=0;

//\-\-\- if the Pattern 0 is not considered

if(IS\_PATTERN\_USAGE(0))

//\-\-\- if the Pattern 0 is not complete

if(!m\_pattern\_0\_done)

//\-\-\- if a day has opened below the pivot

if(m\_daily\_open\_pr<m\_pivot\_val)

            {

//\-\-\- maximum price on the current bar

double last\_high=m\_high.GetData(1);

//\-\-\- if the price is received

if((last\_high>WRONG\_VALUE) && (last\_high<DBL\_MAX))

//\-\-\- if there was a touch from above (considering the tolerance)

if(last\_high>=(m\_pivot\_val-m\_pnt\_near))

                  {

                   result=m\_pattern\_0;

//\-\-\- to the Journal

Print("\\n---== The price touches the pivot level from below ==---");

PrintFormat("Price: %0."+IntegerToString(m\_symbol.Digits())+"f",last\_high);

PrintFormat("Pivot: %0."+IntegerToString(m\_symbol.Digits())+"f",m\_pivot\_val);

PrintFormat("Tolerance: %0."+IntegerToString(m\_symbol.Digits())+"f",m\_pnt\_near);

                  }

            }

//---

return result;

}

//+------------------------------------------------------------------+

Let's launch another strategy in the tester and obtain the result:

![Fig.5. EURUSD: Results of the second basic strategy for 2013-2016](https://c.mql5.com/2/27/5__4.png)

Fig.5.EURUSD: Results of the second basic strategy for 2013-2016

Obviously, the mirroring of the first version did not happen. Probably, the reason is the stop loss and take profit values. Besides, positions with no stop levels activated during a trading day are closed when a new day starts.

Let's try to change the second version of the basic strategy, so that a stop loss level is placed farther when buying — before the main support level Sup1.0, while the profit size is limited by the intermediate resistance level Res0.5. When selling, a stop loss is to be placed on Res1.0, while a take profit — on Sup0.5.

In this case, trading levels for buying are defined the following way:

//+------------------------------------------------------------------+

//\| Define trade levels for buying                                   \|

//+------------------------------------------------------------------+

bool CSignalPivots::OpenLongParams(double &price,double &sl,double &tp,datetime &expiration)

{

bool params\_set=false;

    sl=tp=WRONG\_VALUE;

//\-\-\- if the Pattern 0 is considered

if(IS\_PATTERN\_USAGE(0))

//\-\-\- if the Pattern 0 is not complete

if(!m\_pattern\_0\_done)

         {

//\-\-\- Open price - market

double base\_price=m\_symbol.Ask();

          price=m\_symbol.NormalizePrice(base\_price-m\_price\_level\*PriceLevelUnit());

//\-\-\- sl price - Sup1.0 level

          sl=this.MajorSupport(0);

if(sl==DBL\_MAX)

returnfalse;

//\-\-\- if sl price is set

          sl=m\_symbol.NormalizePrice(sl);

//\-\-\- tp price - Res0.5 level

          tp=this.MinorResistance(0);

if(tp==DBL\_MAX)

returnfalse;

//\-\-\- if tp price is set

          tp=m\_symbol.NormalizePrice(tp);

          expiration+=m\_expiration\*PeriodSeconds(m\_period);

//\-\-\- if prices are set

          params\_set=true;

//\-\-\- pattern complete

          m\_pattern\_0\_done=true;

         }

//---

return params\_set;

}

//+------------------------------------------------------------------+

The result of the third version in the tester is as follows:

![Fig.6. EURUSD: Results of the third basic strategy for 2013-2016](https://c.mql5.com/2/27/6__4.png)

Fig.6. EURUSD: Results of the third basic strategy for 2013-2016

The image is more or less similar to the mirrored first version. At first glance, it seems that the Grail is found. But there are some pitfalls we are going to discuss below.

### 3\. Robustness

If we look closely at Fig. 6, we can easily see that the balance curve grew unevenly. There were the segments where the balance accumulated profits steadily. There were also the drawdown segments as well as the ones where the balance curve moved strictly to the right.

_**Robustness** is a stability of a trading system indicating its relative permanence and efficiency over a long period of time._

In general, we can say that the strategy lacks robustness. Is it possible to improve it? Let's try.

**3.1 Trend indicator**

In my opinion, the trading rules described above work better when there is a directional movement in the market — a trend. The strategy showed the best result on EURUSD in 2014 — early 2015 when the pair was in a steady decline.

This means we need a filter allowing us to avoid a flat. There are plenty of materials about determining a stable trend. You can also find them in the [Articles](https://www.mql5.com/en/articles) section on mql5.com. Personally, I like the article ["Several ways of finding a trend in MQL5"](https://www.mql5.com/en/articles/136) most. It offers a convenient and, more importantly, universal way of searching for a trend.

I have developed a similar indicator **MaTrendCatcher**. It compares the fast and slow Moving Averages. If the difference between them is positive, the trend is bullish. The indicator histogram bars are equal to 1. If the difference is negative, the trend is bearish. The bars are equal to minus 1 (Fig. 7).

![Fig. 7. MaTrendCatcher trend indicator](https://c.mql5.com/2/26/7__1.png)

Fig.7. MaTrendCatcher trend indicator

Besides, if the difference between the Moving Averages increases relative to the previous bar (a trend becomes stronger), the bar is green, otherwise it is red.

Another feature added to the indicator: if the difference between MAs is insignificant, the bars are not displayed. The value of the difference, at which the bars are hidden, depends on the "Cutoff, pp" indicator parameter (Fig. 8).

![Fig.8. MaTrendCatcher trend indicator with small differences hidden](https://c.mql5.com/2/26/8__1.png)

Fig.8. MaTrendCatcher trend indicator with small differences hidden

So, let's use the **MaTrendCatcher** indicator for filtration.

To apply the indicator, we need to implement some changes in the code of the project files. Note that **the last version of the EA is to be stored in the Model folder**.

For this strategy, we need to obtain the calculated value of the "weighted" direction. Therefore, we need a custom class descendant from the base signal class.

class CExpertUserSignal : public CExpertSignal

Then, a new model appears in the updated signal class of reversal levels — Model 1 "trend-flat-countertrend".

In essence, it complements Model 0. Therefore, it can be called a sub-pattern. We will note that in the code a bit later.

Now, verification of buy conditions looks as follows:

//+------------------------------------------------------------------+

//\| Check the buy condition                                          \|

//+------------------------------------------------------------------+

int CSignalPivots::LongCondition(void)

{

int result=0;

//\-\-\- if the Pattern 0 is not considered

if(IS\_PATTERN\_USAGE(0))

//\-\-\- if the Pattern 0 is not complete

if(!m\_pattern\_0\_done)

         {

          m\_is\_signal=false;

//\-\-\- if a day has opened below the pivot

if(m\_daily\_open\_pr<m\_pivot\_val)

            {

//\-\-\- maximum price on the past bar

double last\_high=m\_high.GetData(1);

//\-\-\- if the price is received

if(last\_high>WRONG\_VALUE && last\_high<DBL\_MAX)

//\-\-\- if there was a touch from above (considering the tolerance)

if(last\_high>=(m\_pivot\_val-m\_pnt\_near))

                  {

                   result=m\_pattern\_0;

                   m\_is\_signal=true;

//\-\-\- to the Journal

this.Print(last\_high,ORDER\_TYPE\_BUY);

                  }

            }

//\-\-\- if the Pattern 1 is considered

if(IS\_PATTERN\_USAGE(1))

            {

//\-\-\- if there was a bullish trend on the past bar

if(m\_trend\_val>0. && m\_trend\_val!=EMPTY\_VALUE)

               {

//\-\-\- if there is an acceleration

if(m\_trend\_color==0. && m\_trend\_color!=EMPTY\_VALUE)

                   result+=(m\_pattern\_1+m\_speedup\_allowance);

//\-\-\- if there is no acceleration

else

                   result+=(m\_pattern\_1-m\_speedup\_allowance);

               }

            }

         }

//---

return result;

}

‌

The green block highlights where the sub-pattern is applied.

The idea behind the calculation is as follows: if the market entry is performed without considering the sub-pattern, the signal result is equal to the Pattern 0 weight. If the sub-pattern is considered, the following options are possible:

1. entering in the direction of a trend with acceleration (trend and acceleration bonuses);
2. entering in the direction of a trend without acceleration (trend bonus and acceleration penalty);
3. entering against a trend with acceleration (countertrend and acceleration penalties);
4. entering against a trend with acceleration (countertrend penalty and acceleration bonus).

This approach avoids reacting to a weak signal. If the signal weight overcomes a threshold value, it affects the trading volume size. The pivot EA class features the **CPivotsExpert::LotCoefficient()** method:

//+------------------------------------------------------------------+

//\| Lot ratio                                                        \|

//+------------------------------------------------------------------+

double CPivotsExpert::LotCoefficient(void)

{

double lot\_coeff=1.;

//\-\-\- general signal

    CExpertUserSignal \*ptr\_signal=this.Signal();

if(CheckPointer(ptr\_signal)==POINTER\_DYNAMIC)

      {

double dir\_val=ptr\_signal.GetDirection();

       lot\_coeff=NormalizeDouble(MathAbs(dir\_val/100.),2);

      }

//---

return lot\_coeff;

}

//+------------------------------------------------------------------+

For instance, if the signal has gathered 120 grades, the initial volume is adjusted by 1.2, while in case of 70, it is adjusted by 0.7.

To apply the ratio, it is still necessary to re-define the OpenLong() and OpenShort() methods. For example, the buy method is represented as follows:

//+------------------------------------------------------------------+

//\| Long position open or limit/stop order set                       \|

//+------------------------------------------------------------------+

bool CPivotsExpert::OpenLong(double price,double sl,double tp)

{

if(price==EMPTY\_VALUE)

return(false);

//\-\-\- get lot for open

double lot\_coeff=this.LotCoefficient();

double lot=LotOpenLong(price,sl);

lot=this.NormalLot(lot\_coeff\*lot);

//\-\-\- check lot for open

    lot=LotCheck(lot,price,ORDER\_TYPE\_BUY);

if(lot==0.0)

return(false);

//---

return(m\_trade.Buy(lot,price,sl,tp));

}

//+------------------------------------------------------------------+

The idea with the dynamic formation of the lot size is quite simple: the stronger the signal, the greater the risk.

**3.2 Range size**

It is easy to see that reversal levels (pivots) are close to each other indicating a low market volatility. To avoid trading on such days, the "Width limit, pp" parameter has been introduced. The Pattern 0 (together with the sub-pattern) is considered complete if the limit is not exceeded. The limit is verified in the **Direction()** method body. Below is a part of the code:

//\-\-\- if the limit is set

if(m\_wid\_limit>0.)

      {

//\-\-\- estimated upper limit

double norm\_upper\_limit=m\_symbol.NormalizePrice(m\_wid\_limit+m\_pivot\_val);

//\-\-\- actual upper limit

double res1\_val=this.MajorResistance(0);

if(res1\_val>WRONG\_VALUE && res1\_val<DBL\_MAX)

         {

//\-\-\- if the limit is not exceeded

if(res1\_val<norm\_upper\_limit)

            {

//\-\-\- Pattern 0 is complete

             m\_pattern\_0\_done=true;

//\-\-\- to the Journal

Print("\\n---== Upper limit not exceeded ==---");

PrintFormat("Estimated: %0."+IntegerToString(m\_symbol.Digits())+"f",norm\_upper\_limit);

PrintFormat("Actual: %0."+IntegerToString(m\_symbol.Digits())+"f",res1\_val);

//---

return0.;

            }

         }

//\-\-\- estimated lower limit

double norm\_lower\_limit=m\_symbol.NormalizePrice(m\_pivot\_val-m\_wid\_limit);

//\-\-\- actual lower limit

double sup1\_val=this.MajorSupport(0);

if(sup1\_val>WRONG\_VALUE && sup1\_val<DBL\_MAX)

         {

//\-\-\- if the limit is not exceeded

if(norm\_lower\_limit<sup1\_val)

            {

//\-\-\- Pattern 0 is complete

             m\_pattern\_0\_done=true;

//\-\-\- to the Journal

Print("\\n---== Lower limit not exceeded ==---");

PrintFormat("Estimated: %0."+IntegerToString(m\_symbol.Digits())+"f",norm\_lower\_limit);

PrintFormat("Actual: %0."+IntegerToString(m\_symbol.Digits())+"f",sup1\_val);

//---

return0.;

            }

         }

      }

If the signal does not pass the range width verification, the following entry appears in the Journal:

2015.08.1900:01:00   ---== Upper limit not exceeded ==---

2015.08.1900:01:00   Estimated: 1.10745

2015.08.1900:01:00   Actual: 1.10719

In this case, the signal lacked 26 points to become valid.

Launch the strategy in the tester in the optimization mode. I have used the following optimization parameters:

1. "Width limit, pp";
2. "Tolerance, pp";
3. "Fast МА";
4. "Slow МА";
5. "Cut-off, pp".

The most successful run in terms of profitability looks as follows:

![Fig.9. EURUSD: Results of the strategy with the use of filters for 2013-2016](https://c.mql5.com/2/27/9__4.png)

Fig.9.EURUSD: Results of the strategy with the use of filters for 2013-2016

‌

As expected, some signals were sorted out. The balance curve became smoother.

But there are also fails. As seen on the chart, the strategy generates segments where the balance curve fluctuates in a narrow range without a visible increase in profit starting with 2015. The optimization results can be found in the **EURUSD\_model.xml** file.

Let's look at the results on other symbols.

The best run for USDJPY is displayed on Fig.10.

![Fig.10. USDJPY: Results of the strategy with the use of filters for 2013-2016](https://c.mql5.com/2/27/10__4.png)

Fig.10. USDJPY: Results of the strategy with the use of filters for 2013-2016

‌

Now, let's have a look at spot gold. The best result is shown in Fig. 11.

![Fig.11. XAUUSD: Results of the strategy with the use of filters for 2013-2016](https://c.mql5.com/2/27/11.png)

Fig.11.XAUUSD: Results of the strategy with the use of filters for 2013-2016

‌

During this period, the precious metal was trading in a narrow range, so the strategy did not bring a positive result.

As for GBP, the best run is displayed in Fig. 12.

![Fig.12. GBPUSD: Results of the strategy with the use of filters for 2013-2016](https://c.mql5.com/2/27/12__3.png)

Fig.12.GBPUSD: Results of the strategy with the use of filters for 2013-2016

‌

GBP traded quite well in the direction of a trend. But the correction in 2015 spoiled the final result.

In general, the strategy works best during a trend.

‌

**Conclusion**

Trading strategy development consists of several stages. At the initial stage, the trading idea is formulated. In most cases, this is a hypothesis that needs to be formalized in the form of a code and then checked in the tester. It is often necessary to adjust and refine such a hypothesis during the testing process. This is the standard work of a developer. Here we use the same approach to code the pivot strategy. In my opinion, [OOP](https://www.mql5.com/en/docs/basis/oop) greatly simplifies the task.

All tests in the optimization mode were conducted in the [MQL5 Cloud Network](https://www.mql5.com/en/articles/669). The cloud technology allowed me to evaluate the efficiency of the strategies in quick and non-costly manner.

**File location**

![File location](https://c.mql5.com/2/26/Folders__2.png)‌

It is most convenient to put the strategy files to the single Pivots folder. Move the indicator files (Pivots.ex5 and MaTrendCatcher.ex5) to the %MQL5\\Indicators indicator folder after the compilation.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2853](https://www.mql5.com/ru/articles/2853)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2853.zip "Download all attachments in the single ZIP archive")

[Pivots.zip](https://www.mql5.com/en/articles/download/2853/pivots.zip "Download Pivots.zip")(41.01 KB)

[Optimization.zip](https://www.mql5.com/en/articles/download/2853/optimization.zip "Download Optimization.zip")(40.72 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Cookbook — Macroeconomic events database](https://www.mql5.com/en/articles/11977)
- [MQL5 Cookbook — Services](https://www.mql5.com/en/articles/11826)
- [MQL5 Cookbook – Economic Calendar](https://www.mql5.com/en/articles/9874)
- [MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://www.mql5.com/en/articles/7166)
- [MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)
- [MQL5 Cookbook - Trading signals of moving channels](https://www.mql5.com/en/articles/1863)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/192768)**
(6)


![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
6 Mar 2017 at 15:57

**Maxim Dmitrievsky:**

_I've done this, in the end, purely by piwits you get a coin, you need something to dilute and filter more effectively :)._

An anecdote comes to mind.

_\- Doctor, my neighbour, he is already 70, says that during the night can five times._

_\- Open your mouth. Okay, tongue in place... What's stopping you from saying the same thing?_

![Igor Nistor](https://c.mql5.com/avatar/2017/7/597A54A5-D950.jpg)

**[Igor Nistor](https://www.mql5.com/en/users/netmstnet)**
\|
15 Apr 2017 at 23:30

**MetaQuotes Software Corp.:**

Published article [MQL5 Recipes - Pivot Trading Signals](https://www.mql5.com/en/articles/2853):

Author: [Dennis Kirichenko](https://www.mql5.com/en/users/denkir "denkir")

There are strange trades that immediately close on the next bar, and not a word about them in the journal.

[![](https://c.mql5.com/3/123/Strange__2.png)](https://c.mql5.com/3/123/Strange__1.png "https://c.mql5.com/3/123/Strange__1.png")

Also in SignalPivots.mqh from the Model folder there is a discrepancy in LongCondition and ShortCondition:

```
//+------------------------------------------------------------------+
//| Проверка условия на покупку                                      |
//+------------------------------------------------------------------+
int CSignalPivots::LongCondition(void)
  {
   int result=0;
//--- если Модель 0 учитывается
   if(IS_PATTERN_USAGE(0))
      //--- если Модель 0 не отработана
      if(!m_pattern_0_done)
        {
         m_is_signal=false;
         //--- если день открылся ниже пивота
         if(m_daily_open_pr<m_pivot_val)
           {
            //--- максимальная цена на прошлом баре
            double last_high=m_high.GetData(1);
            //--- если цена получена
            if(last_high>WRONG_VALUE && last_high<DBL_MAX)
               //--- если было касание снизу (с учётом допуска)
               if(last_high>=(m_pivot_val-m_pnt_near))
                 {
                  result=m_pattern_0;
                  m_is_signal=true;
                  //--- в Журнал
                  this.Print(last_high,ORDER_TYPE_BUY);
                 }
           }
         //--- если Модель 1 учитывается
         if(IS_PATTERN_USAGE(1))
           {
            //--- если на прошлом баре был бычий тренд
            if(m_trend_val>0. && m_trend_val!=EMPTY_VALUE)
              {
               //--- если есть ускорение
               if(m_trend_color==0. && m_trend_color!=EMPTY_VALUE)
                  result+=(m_pattern_1+m_speedup_allowance);
               //--- если нет ускорения
               else
                  result+=(m_pattern_1-m_speedup_allowance);
              }
           }
        }
//---
   return result;
  }
//+------------------------------------------------------------------+
//| Проверка условия на продажу                                      |
//+------------------------------------------------------------------+
int CSignalPivots::ShortCondition(void)
  {
   int result=0;
//--- если Модель 0 учитывается
   if(IS_PATTERN_USAGE(0))
      //--- если Модель 0 не отработана
      if(!m_pattern_0_done)
        {
         //--- если день открылся выше пивота
         if(m_daily_open_pr>m_pivot_val)
           {
            //--- минимальная цена на прошлом баре
            double last_low=m_low.GetData(1);
            //--- если цена получена
            if(last_low>WRONG_VALUE && last_low<DBL_MAX)
               //--- если было касание сверху (с учётом допуска)
               if(last_low<=(m_pivot_val+m_pnt_near))
                 {
                  result=m_pattern_0;
                  m_is_signal=true;
                  //--- в Журнал
                  this.Print(last_low,ORDER_TYPE_SELL);
                 }
           }
         //--- если Модель 1 учитывается
         if(IS_PATTERN_USAGE(1))
           {
            //--- если на прошлом баре был медвежий тренд
            if(m_trend_val<0. && m_trend_val!=EMPTY_VALUE)
              {
               //--- если есть ускорение
               if(m_trend_color==0. && m_trend_color!=EMPTY_VALUE)
                  result+=(m_pattern_1+m_speedup_allowance);
               //--- если нет ускорения
               else
                  result+=(m_pattern_1-m_speedup_allowance);
              }
           }
        }
//---
   return result;
  }
```

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
13 May 2017 at 16:16

**Igor Nistor:**

_And also in SignalPivots.mqh from the Model folder there is a discrepancy  in LongCondition and ShortCondition:_

It is imaginary :-)

The flag in the CSignalPivots::LongCondition() method is simply reset, because it is called first.

_There are strange trades that are immediately closed on the next bar, and not a word about them in the log...._

I need details from you. Broker, account type, [EA](https://www.metatrader5.com/en/terminal/help/algotrading/trade_robots_indicators "Help: Setting up and running the Expert Advisor in MetaTrader 5 Client Terminal") and Tester [settings](https://www.metatrader5.com/en/terminal/help/algotrading/trade_robots_indicators "Help: Setting up and running the Expert Advisor in MetaTrader 5 Client Terminal").

I have not noticed such behaviour....

![Simon Mungwira](https://c.mql5.com/avatar/2018/2/5A8ACD93-9AAA.JPG)

**[Simon Mungwira](https://www.mql5.com/en/users/tavamanya)**
\|
16 May 2017 at 21:26

Pivots indicator not loading

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
18 May 2017 at 12:18

**Tavamanya:**

_Pivots indicator not loading_

Have a look at the last 2 sentences in the article:

_**It is most convenient to put the strategy files to the single Pivots folder. Move the indicator files (Pivots.ex5 and MaTrendCatcher.ex5) to the %MQL5\\Indicators indicator folder after the compilation.**_

![Comparative Analysis of 10 Trend Strategies](https://c.mql5.com/2/26/MQL5-avatar-sravn-analiz-001__1.png)[Comparative Analysis of 10 Trend Strategies](https://www.mql5.com/en/articles/3074)

The article provides a brief overview of ten trend following strategies, as well as their testing results and comparative analysis. Based on the obtained results, we draw a general conclusion about the appropriateness, advantages and disadvantages of trend following trading.

![Ready-made Expert Advisors from the MQL5 Wizard work in MetaTrader 4](https://c.mql5.com/2/26/MQL5_expert_in_MT4.png)[Ready-made Expert Advisors from the MQL5 Wizard work in MetaTrader 4](https://www.mql5.com/en/articles/3068)

The article offers a simple emulator of the MetaTrader 5 trading environment for MetaTrader 4. The emulator implements migration and adjustment of trade classes of the Standard Library. As a result, Expert Advisors generated in the MetaTrader 5 Wizard can be compiled and executed in MetaTrader 4 without changes.

![How Long Is the Trend?](https://c.mql5.com/2/27/MQL5-avatar-TrendTime-001.png)[How Long Is the Trend?](https://www.mql5.com/en/articles/3188)

The article highlights several methods for trend identification aiming to determine the trend duration relative to the flat market. In theory, the trend to flat rate is considered to be 30% to 70%. This is what we'll be checking.

![Graphical Interfaces X: Word wrapping algorithm in the Multiline Text box (build 12)](https://c.mql5.com/2/27/MQL5-avatar-RedSquare-001.png)[Graphical Interfaces X: Word wrapping algorithm in the Multiline Text box (build 12)](https://www.mql5.com/en/articles/3173)

We continue to develop the Multiline Text box control. This time our task is to implement an automatic word wrapping in case a text box width overflow occurs, or a reverse word wrapping of the text to the previous line if the opportunity arises.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/2853&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068719420093693137)

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