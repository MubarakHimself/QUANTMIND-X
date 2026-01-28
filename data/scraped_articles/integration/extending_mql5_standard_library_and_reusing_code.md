---
title: Extending MQL5 Standard Library and Reusing Code
url: https://www.mql5.com/en/articles/741
categories: Integration, Indicators
relevance_score: 3
scraped_at: 2026-01-23T21:20:09.848581
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/741&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071755201762766296)

MetaTrader 5 / Examples


### Introduction

MQL5 Standard Library is an object-oriented framework composed of a set of ready-to-use classes which makes your life easier as a developer. Nevertheless, it does not implement all the needs of all developers around the world, so if you feel that you need some more custom stuff, you can take a step further and expand. This article takes you through integrating MetaQuotes' Zig-Zag technical indicator into the Standard Library. We will be inspired by MetaQuotes' design philosophy in order to achieve our goal.

In a nutshell, MQL5 API is intended for you to benefit from
code reuse, reliability, flexibility and easy maintenance. This is
what the theory says, but beyond all this, if you plan to continue to
advance in MQL5 and develop more sophisticated things, such as
multi-currency Experts Advisors, first you should be able to code in the
Standard Library way so that your apps are guaranteed a successful
life.

As your EAs and indicators become more and more complex, it is more
necessary to master the concepts involved in a framework development.
As a real life example, it is my personal need to develop a complex
multi-currency EA which dictates the need of strengthening the base of my project from scratch.

![Figure 1. Regular polyhedra are perfect objects. They depict well the approach of building apps on solid concepts.](https://c.mql5.com/2/6/davinci-platonic-solids.jpg)

**Figure 1. Regular polyhedra are perfect objects. They depict well the approach of building apps on solid concepts**

### 1\. ZigZag Download

We start by downloading [MetaQuotes' ZigZag indicator](https://www.mql5.com/en/code/56), which is available in [Code Base](https://www.mql5.com/en/code), from our MetaTrader 5 Terminal. This will create the files **Indicators\\zigzag.mq5** and **Indicators\\zigzag.ex5**.

![Figure 2. We start downloading MetaQuotes' ZigZag from our MetaTrader 5 Terminal](https://c.mql5.com/2/6/downloadingZigZag__2.jpg)

**Figure 2. We start downloading MetaQuotes' ZigZag from our** **MetaTrader 5 Terminal**

I attach here those lines of **Indicators\\zigzag.mq5** containing the indicator's input parameters, global variables and the **OnInit()** handler. I only put this part because the entire file has 298 lines of code. This is simplyfor convenience and the understanding of the bigger picture we are talking about below.

```
//+------------------------------------------------------------------+
//|                                                       ZigZag.mq5 |
//|                        Copyright 2009, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "2009, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 3
#property indicator_plots   1
//---- plot Zigzag
#property indicator_label1  "Zigzag"
#property indicator_type1   DRAW_SECTION
#property indicator_color1  Red
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
//--- input parameters
input int      ExtDepth=12;
input int      ExtDeviation=5;
input int      ExtBackstep=3;
//--- indicator buffers
double         ZigzagBuffer[];      // main buffer
double         HighMapBuffer[];     // highs
double         LowMapBuffer[];      // lows
int            level=3;             // recounting depth
double         deviation;           // deviation in points
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,ZigzagBuffer,INDICATOR_DATA);
   SetIndexBuffer(1,HighMapBuffer,INDICATOR_CALCULATIONS);
   SetIndexBuffer(2,LowMapBuffer,INDICATOR_CALCULATIONS);

//--- set short name and digits
   PlotIndexSetString(0,PLOT_LABEL,"ZigZag("+(string)ExtDepth+","+(string)ExtDeviation+","+(string)ExtBackstep+")");
   IndicatorSetInteger(INDICATOR_DIGITS,_Digits);
//--- set empty value
   PlotIndexSetDouble(0,PLOT_EMPTY_VALUE,0.0);
//--- to use in cycle
   deviation=ExtDeviation*_Point;
//---
   return(INIT_SUCCEEDED);
  }
```

### 2\. Quick Top-Down Overview

Let's now take a [top-down approach](https://en.wikipedia.org/wiki/Top-down_and_bottom-up_design "https://en.wikipedia.org/wiki/Top-down_and_bottom-up_design") to think of our new object-oriented [ZigZag indicator](https://www.mql5.com/en/code/56) that we want to integrate into MQL5 Standard Library. This means that first we have to look at the whole system and then analyze the smaller parts of it. So why don't we code a couple of dummy EAs in order to see the bigger picture? Let's write a procedural styled Expert Advisor together with its object-oriented version.

**2.1. ZigZag Out of the Box**

Intermediate MQL5 developers would probably use the ZigZag indicator in their EAs like this:

```
//+----------------------------------------------------------------------+
//|                                            ExpertOriginalZigZag.mq5  |
//|                   Copyright © 2013, Laplacianlab - Jordi Bassagañas  |
//+----------------------------------------------------------------------+
//--- EA properties
#property copyright     "Copyright © 2013, Laplacianlab - Jordi Bassagañas"
#property link          "https://www.mql5.com/en/articles"
#property version       "1.00"
#property description   "This dummy Expert Advisor is just for showing how to use the original MetaQuotes' ZigZag indicator."
//--- EA inputs
input ENUM_TIMEFRAMES   EAPeriod=PERIOD_H1;
input string            CurrencyPair="EURUSD";
//--- global variables
int      zigZagHandle;
double   zigZagBuffer[];
double   zigZagHigh[];
double   zigZagLow[];
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   zigZagHandle=iCustom(CurrencyPair,EAPeriod,"zigzag",12,5,3);
   ArraySetAsSeries(zigZagBuffer,true);
   ArraySetAsSeries(zigZagHigh,true);
   ArraySetAsSeries(zigZagLow,true);
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   IndicatorRelease(zigZagHandle);
   ArrayFree(zigZagBuffer);
   ArrayFree(zigZagHigh);
   ArrayFree(zigZagLow);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- refresh data
   if(CopyBuffer(zigZagHandle,0,0,2,zigZagBuffer)<0)
   {
      Print("Can't copy ZigZag buffer 0!");
      return;
   }
   if(CopyBuffer(zigZagHandle,1,0,2,zigZagHigh)<0)
   {
      Print("Can't copy ZigZag buffer 1!");
      return;
   }
   if(CopyBuffer(zigZagHandle,2,0,2,zigZagLow)<0)
   {
      Print("Can't copy ZigZag buffer 2!");
      return;
   }
   //--- print values
   if(zigZagBuffer[0]!=0) Print("zigZagBuffer[0]: ", zigZagBuffer[0]);
   if(zigZagHigh[0]!=0) Print("zigZagHigh[0]: ", zigZagHigh[0]);
   if(zigZagLow[0]!=0) Print("zigZagLow[0]: ", zigZagLow[0]);
  }
//+------------------------------------------------------------------+
```

**2.2. ZigZag Integrated into the Standard Library**

On the other hand, advanced MQL5 developers will want to work with ZigZag indicator just as they already do with the Standard Library indicators, this way:

```
//+----------------------------------------------------------------------+
//|                                                  ExpertOOZigZag.mq5  |
//|                   Copyright © 2013, Laplacianlab - Jordi Bassagañas  |
//+----------------------------------------------------------------------+
#include <..\Include\Indicators\Custom\Trend.mqh>
//--- EA properties
#property copyright     "Copyright © 2013, Laplacianlab - Jordi Bassagañas"
#property link          "https://www.mql5.com/en/articles"
#property version       "1.00"
#property description   "This dummy Expert Advisor is just for showing how to use the object-oriented version of MetaQuotes' ZigZag indicator."
//--- EA inputs
input ENUM_TIMEFRAMES   EAPeriod=PERIOD_H1;
input string            CurrencyPair="EURUSD";
//--- global variables
CiZigZag *ciZigZag;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   ciZigZag=new CiZigZag;
   ciZigZag.Create(CurrencyPair,EAPeriod,12,5,3);
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   delete(ciZigZag);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- refresh data
   ciZigZag.Refresh();
   //--- print values
   if(ciZigZag.ZigZag(0)!=0) Print("OO ZigZag buffer: ", ciZigZag.ZigZag(0));
   if(ciZigZag.High(0)!=0) Print("OO ZigZag high: ", ciZigZag.High(0));
   if(ciZigZag.Low(0)!=0) Print("OO ZigZag low: ",ciZigZag.Low(0));
  }
//+------------------------------------------------------------------+
```

**2.3. Conclusion**

The second solution is better because it is object-oriented. Once the OO classes have been developed, it is intuitive to observe that it is much easier interacting with the Zig-Zag's object-oriented funcionality than working with its procedural counterpart. Let's briefly recall, however, the advantages we benefit from when working with an object oriented library:

- OOP makes it easy to model problems.

- OOP makes it easy to reuse code, which in turn benefits cost, reliability, flexibility and maintenance.
- This paradigm enables the creation of ADTs ( [Abstract Data Types](https://en.wikipedia.org/wiki/Abstract_data_type "https://en.wikipedia.org/wiki/Abstract_data_type")). An ADT is an abstraction of the traditional concept of data type, which is present in all programming languages.


**![Figure 3. Regular icosahedron. Building our apps on solid concepts is a quality guarantee that makes our designs persist in time.](https://c.mql5.com/2/6/regular-polyhedra__2.jpg)**

**Figure 3. Regular icosahedron. Building our apps on solid concepts is a quality guarantee that makes our designs persist in time**

### 3\. Integrating our new OO ZigZag into MQL5 Standard Library

As I said in the introduction of this article, we are being inspired by MetaQuotes' object-oriented style to build our new set of classes intended for wrapping the ZigZag downloaded before. This is easy, we just have to take a look at the files inside **Include\\Indicators** and study and understand some of the ideas that lay behind MQL5 Standard Library. When you look at what there is inside MetaQuotes' **Trend.mqh** you will soon realize that it is full of classes representing some technical indicators: ADX, Bollinger Bands, SAR, Moving Averages, etc. All these classes inherit from **CIndicator**. So let's implement this scheme. By the way, extending the new OO indicator from MQL5's class [**CiCustom**](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator) would have been another alternative to implement this exercise.

Let's begin by creating the new folder **Include\\Indicators\\Custom** and, right after, the new file **Include\\Indicators\\Custom\\Trend.mqh** in order for us to code there our own technical indicators, just as MetaQuotes does in its **Include\\Indicators\\Trend.mqh**. Here is our extending file **Include\\Indicators\\Custom\\Trend.mqh** already implemented. I will discuss below some technical aspects needed to code it.

```
//+------------------------------------------------------------------+
//|                              Include\Indicators\Custom\Trend.mqh |
//|                  Copyright 2013, Laplacianlab - Jordi Bassagañas |
//|                     https://www.mql5.com/en/users/laplacianlab |
//+------------------------------------------------------------------+
#include <..\Include\Indicators\Indicator.mqh>
//+------------------------------------------------------------------+
//| Class CiZigZag.                                                  |
//| Purpose: Class of the "ZigZag" indicator.                        |
//|          Derives from class CIndicator.                          |
//+------------------------------------------------------------------+
class CiZigZag : public CIndicator
  {
protected:
   int               m_depth;
   int               m_deviation;
   int               m_backstep;

public:
                     CiZigZag(void);
                    ~CiZigZag(void);
   //--- methods of access to protected data
   int               Depth(void)          const { return(m_depth);      }
   int               Deviation(void)      const { return(m_deviation);  }
   int               Backstep(void)       const { return(m_backstep);   }
   //--- method of creation
   bool              Create(const string symbol,const ENUM_TIMEFRAMES period,
                            const int depth,const int deviation_create,const int backstep);
   //--- methods of access to indicator data
   double            ZigZag(const int index) const;
   double            High(const int index) const;
   double            Low(const int index) const;
   //--- method of identifying
   virtual int       Type(void) const { return(IND_CUSTOM); }

protected:
   //--- methods of tuning
   virtual bool      Initialize(const string symbol,const ENUM_TIMEFRAMES period,const int num_params,const MqlParam &params[]);
   bool              Initialize(const string symbol,const ENUM_TIMEFRAMES period,
                                const int depth,const int deviation_init,const int backstep);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CiZigZag::CiZigZag(void) : m_depth(-1),
                         m_deviation(-1),
                         m_backstep(-1)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CiZigZag::~CiZigZag(void)
  {
  }
//+------------------------------------------------------------------+
//| Create indicator "Zig Zag"                                       |
//+------------------------------------------------------------------+
bool CiZigZag::Create(const string symbol,const ENUM_TIMEFRAMES period,
                      const int depth,const int deviation_create,const int backstep)
  {
//--- check history
   if(!SetSymbolPeriod(symbol,period))
      return(false);
//--- create
   m_handle=iCustom(symbol,period,"zigzag",depth,deviation_create,backstep);
//--- check result
   if(m_handle==INVALID_HANDLE)
      return(false);
//--- indicator successfully created
   if(!Initialize(symbol,period,depth,deviation_create,backstep))
     {
      //--- initialization failed
      IndicatorRelease(m_handle);
      m_handle=INVALID_HANDLE;
      return(false);
     }
//--- ok
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialize the indicator with universal parameters               |
//+------------------------------------------------------------------+
bool CiZigZag::Initialize(const string symbol,const ENUM_TIMEFRAMES period,const int num_params,const MqlParam &params[])
  {
   return(Initialize(symbol,period,(int)params[0].integer_value,(int)params[1].integer_value,(int)params[2].integer_value));
  }
//+------------------------------------------------------------------+
//| Initialize indicator with the special parameters                 |
//+------------------------------------------------------------------+
bool CiZigZag::Initialize(const string symbol,const ENUM_TIMEFRAMES period,
                        const int depth,const int deviation_init,const int backstep)
  {
   if(CreateBuffers(symbol,period,3))
     {
      //--- string of status of drawing
      m_name  ="ZigZag";
      m_status="("+symbol+","+PeriodDescription()+","+
               IntegerToString(depth)+","+IntegerToString(deviation_init)+","+
               IntegerToString(backstep)+") H="+IntegerToString(m_handle);
      //--- save settings
      m_depth=depth;
      m_deviation=deviation_init;
      m_backstep=backstep;
      //--- create buffers
      ((CIndicatorBuffer*)At(0)).Name("ZIGZAG");
      ((CIndicatorBuffer*)At(1)).Name("HIGH");
      ((CIndicatorBuffer*)At(2)).Name("LOW");
      //--- ok
      return(true);
     }
//--- error
   return(false);
  }
//+------------------------------------------------------------------+
//| Access to ZigZag buffer of "Zig Zag"                             |
//+------------------------------------------------------------------+
double CiZigZag::ZigZag(const int index) const
  {
   CIndicatorBuffer *buffer=At(0);
//--- check
   if(buffer==NULL)
      return(EMPTY_VALUE);
//---
   return(buffer.At(index));
  }
//+------------------------------------------------------------------+
//| Access to High buffer of "Zig Zag"                               |
//+------------------------------------------------------------------+
double CiZigZag::High(const int index) const
  {
   CIndicatorBuffer *buffer=At(1);
//--- check
   if(buffer==NULL)
      return(EMPTY_VALUE);
//---
   return(buffer.At(index));
  }
//+------------------------------------------------------------------+
//| Access to Low buffer of "Zig Zag"                                |
//+------------------------------------------------------------------+
double CiZigZag::Low(const int index) const
  {
   CIndicatorBuffer *buffer=At(2);
//--- check
   if(buffer==NULL)
      return(EMPTY_VALUE);
//---
   return(buffer.At(index));
  }
//+------------------------------------------------------------------+
```

**3.1. Object-Oriented Encapsulation**

OO encapsulation is a good programming practice meaning that data member of objects can only be modified by the operations defined for them. All the classes defined in MetaQuotes's **Trend.mqh** implement this idea, so we are doing the same.

On the one hand, there are **CiZigZag**'s specific protected properties:

```
protected:
   int               m_depth;
   int               m_deviation;
   int               m_backstep;
```

Subsequently, there is **CiZigZag**'s public interface for accessing from outside a given object of **CiZigZag** type the protected properties defined above:

```
public:
   //--- methods of access to protected data
   int               Depth(void)          const { return(m_depth);      }
   int               Deviation(void)      const { return(m_deviation);  }
   int               Backstep(void)       const { return(m_backstep);   }
```

This is a security measure for isolating objects. This encapsulation protects against arbitrary modifications carried out by someone or something that is not allowed to access objects data.

**3.2. Accessing ZigZag's Data**

As seen in the first section of this article, the source code file named **zigzag.mq5** creates three buffers:

```
//--- indicator buffers mapping
   SetIndexBuffer(0,ZigzagBuffer,INDICATOR_DATA);
   SetIndexBuffer(1,HighMapBuffer,INDICATOR_CALCULATIONS);
   SetIndexBuffer(2,LowMapBuffer,INDICATOR_CALCULATIONS);
```

Through object-oriented encapsulation, **CiZigZag**'s methods **ZigZag(const int index)**, **High(const int index)** and **Low(const int index)** return the indicator buffers which have previously been created in the initialization method. It is important to note that the object-oriented wrapper **CIndicatorBuffer** is defined in MQL5's class **Include\\Indicators\\Indicator.mqh**. **CIndicatorBuffer** is the core piece of these three methods. We are already immersed in MQL5 API!

As an example here, it is the code for accessing **CiZigZag**'s High buffer:

```
//+------------------------------------------------------------------+
//| Access to High buffer of "Zig Zag"                               |
//+------------------------------------------------------------------+
double CiZigZag::High(const int index) const
  {
   CIndicatorBuffer *buffer=At(1);
//--- check
   if(buffer==NULL)
      return(EMPTY_VALUE);
//---
   return(buffer.At(index));
  }
```

**3.3.** **Polymorphism,** **method overloading and virtual functions**

In the previous section we briefly discussed the topic of encapsulation which is one of the most important features of object-oriented programming. Well, the classes contained in **Include\\Indicators\\Indicator.mqh** and the file **Include\\Indicators\\Custom\\Trend.mqh** deal with another two aspects of the OOP paradigm, polymorphism and method overloading.

Polymorphism has the ability to access a diverse range of methods through the same interface. This way, a given identifier can take several forms depending on the context in which it is located. Polymorphism requires the inheritance mechanism so that it can be implemented. On the other hand, method overloading is another OOP feature that allows creating several methods sharing the same name but with different parameter declarations.

This is a very short introduction. There is not enough space in this article to discuss these subjects, so delving into them is an exercise left for you. Please, read the MQL5 sections [Polymorphism](https://www.mql5.com/en/docs/basis/oop/polymorphism) and [Overload](https://www.mql5.com/en/docs/basis/oop/overload). In any case, we see the Standard Library implement all OOP features, consequently, the better we know them the better we can extend the API to suit our needs.

With all that was said, there is only one more thing to be noted. MQL5 implements polymorphism by a mechanism called [Virtual Functions](https://en.wikipedia.org/wiki/Virtual_function "https://en.wikipedia.org/wiki/Virtual_function"). Once again, please, read the MQL5 section [Virtual Functions](https://www.mql5.com/en/docs/basis/oop/virtual) to understand how it works.

This is why we code **CiZigZag**'s initialization method this way:

```
//+------------------------------------------------------------------+
//| Initialize the indicator with universal parameters               |
//+------------------------------------------------------------------+
bool CiZigZag::Initialize(const string symbol,const ENUM_TIMEFRAMES period,const int num_params,const MqlParam &params[])
  {
   return(Initialize(symbol,period,(int)params[0].integer_value,(int)params[1].integer_value,(int)params[2].integer_value));
  }
//+------------------------------------------------------------------+
//| Initialize indicator with the special parameters                 |
//+------------------------------------------------------------------+
bool CiZigZag::Initialize(const string symbol,const ENUM_TIMEFRAMES period,
                        const int depth,const int deviation_init,const int backstep)
  {
   if(CreateBuffers(symbol,period,3))
     {
      //--- string of status of drawing
      m_name  ="ZigZag";
      m_status="("+symbol+","+PeriodDescription()+","+
               IntegerToString(depth)+","+IntegerToString(deviation_init)+","+
               IntegerToString(backstep)+") H="+IntegerToString(m_handle);
      //--- save settings
      m_depth=depth;
      m_deviation=deviation_init;
      m_backstep=backstep;
      //--- create buffers
      ((CIndicatorBuffer*)At(0)).Name("ZIGZAG");
      ((CIndicatorBuffer*)At(1)).Name("HIGH");
      ((CIndicatorBuffer*)At(2)).Name("LOW");
      //--- ok
      return(true);
     }
//--- error
   return(false);
  }
```

### 4\. Testing the new OO ZigZag, already available in the Standard Library

Of course, before using the extensions developed by you in your OO developments, you should first ensure that they work as expected. It is recommended to run a comprehensive set of tests on your new custom components. For simplicity issues, however, we'll now run a simple test on the three main methods of **CiZigZag**, that is to say, **ZigZag(const int index)**, **High(const int index)** and **Low(const int index)**.

We will just print the values calculated by those three methods on every EA's tick and then compare the output generated by **ExpertOriginalZigZag.ex5**, the dummy procedural EA, with the output generated by **ExpertOOZigZag.ex5**, the dummy object-oriented EA. Whenever both outputs obtained are the same, we can conclude that the new extension is OK, we can take for good our OO ZigZag integrated into MQL5 API.

![Figure 4. We are comparing the output generated by ExpertOriginalZigZag.ex5 with the output generated by ExpertOOZigZag.ex5](https://c.mql5.com/2/6/backtesting__1.jpg)

**Figure 4. We are comparing the output generated by ExpertOriginalZigZag.ex5 with the output generated by ExpertOOZigZag.ex5**

Therefore we run both **ExpertOriginalZigZag.ex5** and **ExpertOOZigZag.ex5**, the two EAs presented at the beginning of this article, in the Strategy Tester with the following parameters set:

- **Symbol**: EURUSD, H1
- **Date**: Custom period, from 2013.08.01 to 2013.08.15
- **Execution**: Normal, 1 Minute OHLC
- **Deposit**: 10000 USD, 1:100
- **Optimization**: None

As both robots print the same results we conclude that our **CiZigZag** is well implemented, so we can use it in our developments from now on.

Log generated by **ExpertOriginalZigZag.ex5**:

```
DE      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:50:40   zigZagBuffer[0]: 1.32657
ML      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:50:40   zigZagLow[0]: 1.32657
FL      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:50:59   zigZagBuffer[0]: 1.32657
GE      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:50:59   zigZagLow[0]: 1.32657
KS      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:51:00   zigZagBuffer[0]: 1.32657
FR      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:51:00   zigZagLow[0]: 1.32657
GK      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:51:20   zigZagBuffer[0]: 1.32653
RJ      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:51:20   zigZagLow[0]: 1.32653
OR      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:51:40   zigZagBuffer[0]: 1.32653
FS      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:51:40   zigZagLow[0]: 1.32653
QJ      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:51:59   zigZagBuffer[0]: 1.32653
PH      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:51:59   zigZagLow[0]: 1.32653
JQ      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:52:00   zigZagBuffer[0]: 1.32653
KP      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:52:00   zigZagLow[0]: 1.32653
RH      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:52:20   zigZagBuffer[0]: 1.32653
GI      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:52:20   zigZagLow[0]: 1.32653
GP      0       18:45:39        ExpertOriginalZigZag (EURUSD,H1)        2013.08.01 08:52:40   zigZagBuffer[0]: 1.32614
// More data here!..
```

Log generated by **ExpertOOZigZag.ex5**:

```
RP      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:50:40   OO ZigZag buffer(0): 1.32657
HQ      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:50:40   OO ZigZag low(0): 1.32657
DI      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:50:59   OO ZigZag buffer(0): 1.32657
RH      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:50:59   OO ZigZag low(0): 1.32657
QR      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:51:00   OO ZigZag buffer(0): 1.32657
GS      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:51:00   OO ZigZag low(0): 1.32657
IK      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:51:20   OO ZigZag buffer(0): 1.32653
GJ      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:51:20   OO ZigZag low(0): 1.32653
EL      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:51:40   OO ZigZag buffer(0): 1.32653
OD      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:51:40   OO ZigZag low(0): 1.32653
OE      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:51:59   OO ZigZag buffer(0): 1.32653
IO      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:51:59   OO ZigZag low(0): 1.32653
DN      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:52:00   OO ZigZag buffer(0): 1.32653
RF      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:52:00   OO ZigZag low(0): 1.32653
PP      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:52:20   OO ZigZag buffer(0): 1.32653
RQ      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:52:20   OO ZigZag low(0): 1.32653
MI      0       18:48:02        ExpertOOZigZag (EURUSD,H1)      2013.08.01 08:52:40   OO ZigZag buffer(0): 1.32614
// More data here!..
```

### Conclusion

[MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) makes your life easier as a developer. Nevertheless, it cannot implement all the needs of all developers around the world, so there will always be some point where you will need to create your custom stuff. As your EAs and indicators become more and more complex, it is more necessary to master the concepts involved in a framework development. Extending MQL5 Standard Library is a quality guarantee for your applications to have a successful life.

We have taken advantage from code reuse by first downloading [the ZigZag indicator](https://www.mql5.com/en/code/56) from Code Base. Once available in our MetaTrader 5 Terminal, we took a [top-down approach](https://en.wikipedia.org/wiki/Top-down_and_bottom-up_design "https://en.wikipedia.org/wiki/Top-down_and_bottom-up_design") in order to start thinking of our new object-oriented ZigZag indicator. We had a general look at the whole system and then continued analyzing. In the first phase of the development we compared a dummy EA using the procedural styled ZigZag indicator with its object-oriented counterpart.

We wrapped the ZigZag indicator into an object-oriented class which was designed according to MetaQuotes's design philosophy, the same applied for building the Standard Library. And finally we run some simple tests concluding that our new CiZigZag wrapper, already integrated into MQL5 API, is well implemented.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/741.zip "Download all attachments in the single ZIP archive")

[expertoriginalzigzag.mq5](https://www.mql5.com/en/articles/download/741/expertoriginalzigzag.mq5 "Download expertoriginalzigzag.mq5")(2.54 KB)

[expertoozigzag.mq5](https://www.mql5.com/en/articles/download/741/expertoozigzag.mq5 "Download expertoozigzag.mq5")(2.03 KB)

[trend.mqh](https://www.mql5.com/en/articles/download/741/trend.mqh "Download trend.mqh")(6.34 KB)

[zigzag.mq5](https://www.mql5.com/en/articles/download/741/zigzag.mq5 "Download zigzag.mq5")(9.34 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building a Social Technology Startup, Part II: Programming an MQL5 REST Client](https://www.mql5.com/en/articles/1044)
- [Building a Social Technology Startup, Part I: Tweet Your MetaTrader 5 Signals](https://www.mql5.com/en/articles/925)
- [Raise Your Linear Trading Systems to the Power](https://www.mql5.com/en/articles/734)
- [Marvel Your MQL5 Customers with a Usable Cocktail of Technologies!](https://www.mql5.com/en/articles/728)
- [Building an Automatic News Trader](https://www.mql5.com/en/articles/719)
- [Another MQL5 OOP Class](https://www.mql5.com/en/articles/703)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/13868)**
(22)


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
18 Jul 2017 at 09:26

**Tango\_X:**

**Am I understanding you correctly???**

It turns out that it is necessary to copy the whole history on each tick?

1\. You can do it at the opening of a [new bar](https://www.mql5.com/en/articles/159 "Article 'Limitations and checks in experts'")

2\. Why do you need to get all indicator values every time, and at the same time take care of the indexing direction? What is the task at all?

![TheXpert](https://c.mql5.com/avatar/2016/7/5783C6E7-AEEE.png)

**[TheXpert](https://www.mql5.com/en/users/thexpert)**
\|
18 Jul 2017 at 10:08

Why wrapping a simple indicator into a class if it is used later either in a chart or via iCustom?

Второе решение лучше, потому что является объектно-ориентированным

OOP for OOP's sake, okay.

![Tango_X](https://c.mql5.com/avatar/avatar_na2.png)

**[Tango\_X](https://www.mql5.com/en/users/tango_x)**
\|
18 Jul 2017 at 11:28

**Rashid Umarov:**

1\. It is possible to open a [new bar](https://www.mql5.com/en/articles/159 "Article 'Limitations and checks in experts'")

2\. Why do you need to get all the indicator values every time, and at the same time care about the indexing direction? What is the task at all?

The problem was solved by the loop conditions, now everything works as desired. thanks!

![Tango_X](https://c.mql5.com/avatar/avatar_na2.png)

**[Tango\_X](https://www.mql5.com/en/users/tango_x)**
\|
18 Jul 2017 at 11:31

**Комбинатор:**

Why wrapping a simple indicator into a class if it is used later either in a chart or via iCustom?

OOP for OOP's sake, okay.

That's right)))

Very convenient, I recommend it to everyone


![Sergio Tarquini](https://c.mql5.com/avatar/2019/11/5DBF5ADD-19E2.png)

**[Sergio Tarquini](https://www.mql5.com/en/users/stitpro)**
\|
17 Jun 2020 at 19:04

I read this article (and this [https://www.mql5.com/en/forum/335975](https://www.mql5.com/en/forum/335975) and this [https://www.mql5.com/en/articles/5](https://www.mql5.com/en/articles/5)) during a bug fix session related to custom indicator buffers, I finally fix it when I realized that only one buffer can set with INDICATOR\_DATA flag, otherwise CopyBuffers/GetData returns always -1 except for the first indicator flagged as INDICATOR\_DATA.

Thank you.

![Creating Neural Network EAs Using MQL5 Wizard and Hlaiman EA Generator](https://c.mql5.com/2/0/HLAIMAN.png)[Creating Neural Network EAs Using MQL5 Wizard and Hlaiman EA Generator](https://www.mql5.com/en/articles/706)

The article describes a method of automated creation of neural network EAs using MQL5 Wizard and Hlaiman EA Generator. It shows you how you can easily start working with neural networks, without having to learn the entire body of theoretical information and writing your own code.

![Expert Advisor for Trading in the Channel](https://c.mql5.com/2/17/834_22.gif)[Expert Advisor for Trading in the Channel](https://www.mql5.com/en/articles/1375)

The Expert Advisor plots the channel lines. The upper and lower channel lines act as support and resistance levels. The Expert Advisor marks datum points, provides sound notification every time the price reaches or crosses the channel lines and draws the relevant marks. Upon fractal formation, the corresponding arrows appear on the last bars. Line breakouts may suggest the possibility of a growing trend. The Expert Advisor is extensively commented throughout.

![How to Make Money from MetaTrader AppStore and Trading Signals Services If You Are Not a Seller or a Provider](https://c.mql5.com/2/0/mql5_share_avatar.png)[How to Make Money from MetaTrader AppStore and Trading Signals Services If You Are Not a Seller or a Provider](https://www.mql5.com/en/articles/756)

It is possible to start making money on MQL5.com right now without having to be a seller of Market applications or a profitable signals provider. Select the products you like and post links to them on various web resources. Attract potential customers and the profit is yours!

![Trading Signal Generator Based on a Custom Indicator](https://c.mql5.com/2/0/icustom_ava.png)[Trading Signal Generator Based on a Custom Indicator](https://www.mql5.com/en/articles/691)

How to create a trading signal generator based on a custom indicator? How to create a custom indicator? How to get access to custom indicator data? Why do we need the IS\_PATTERN\_USAGE(0) structure and model 0?

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/741&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071755201762766296)

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