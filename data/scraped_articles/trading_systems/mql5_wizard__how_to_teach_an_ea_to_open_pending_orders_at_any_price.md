---
title: MQL5 Wizard: How to Teach an EA to Open Pending Orders at Any Price
url: https://www.mql5.com/en/articles/723
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:47:05.844210
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/723&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070557610261813254)

MetaTrader 5 / Examples


### Introduction

An Expert Advisor generated using the MQL5 Wizard can only open pending orders at the fixed distance from the current price. This means that if the market situation changes (e.g. a change in market volatility), the Expert Advisor will have to be run again with new parameters.

This would not be suitable for many trading systems. In most cases, the price level for pending orders is determined dynamically by a trading system. And the distance from the current price is constantly changing. In this article, we will discuss how to modify an Expert Advisor generated using the MQL5 Wizard so that it can open pending orders at varying distances from the current price.

### 1\. The Mechanism of Opening Pending Orders in the Expert Advisor Generated Using the MQL5 Wizard

A generated Expert Advisor would have approximately the same code in its header as provided below:

```
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
//--- inputs for expert
input string             Expert_Title="ExpertMySignalEnvelopes.mq5";      // Document name
ulong                    Expert_MagicNumber        =3915;        //
bool                     Expert_EveryTick          =false;       //
//--- inputs for main signal
input int                Signal_ThresholdOpen      =10;          // Signal threshold value to open [0...100]
input int                Signal_ThresholdClose     =10;          // Signal threshold value to close [0...100]
input double             Signal_PriceLevel         =0.0;         // Price level to execute a deal
input double             Signal_StopLevel          =85.0;        // Stop Loss level (in points)
input double             Signal_TakeLevel          =195.0;       // Take Profit level (in points)
input int                Signal_Expiration         =0;           // Expiration of pending orders (in bars)
input int                Signal_Envelopes_PeriodMA =13;          // Envelopes(13,0,MODE_SMA,...) Period of averaging
input int                Signal_Envelopes_Shift    =0;           // Envelopes(13,0,MODE_SMA,...) Time shift
input ENUM_MA_METHOD     Signal_Envelopes_Method   =MODE_SMA;    // Envelopes(13,0,MODE_SMA,...) Method of averaging
input ENUM_APPLIED_PRICE Signal_Envelopes_Applied  =PRICE_CLOSE; // Envelopes(13,0,MODE_SMA,...) Prices series
input double             Signal_Envelopes_Deviation=0.2;         // Envelopes(13,0,MODE_SMA,...) Deviation
input double             Signal_Envelopes_Weight   =1.0;         // Envelopes(13,0,MODE_SMA,...) Weight [0...1.0]
//--- inputs for money
input double             Money_FixLot_Percent      =10.0;        // Percent
input double             Money_FixLot_Lots         =0.1;         // Fixed volume
//+------------------------------------------------------------------+
```

Please note the **Signal\_PriceLevel** parameter. By default, the Expert Advisor is generated with **Signal\_PriceLevel=0**. This parameter defines the distance from the current price. If it is equal to zero, an order will be opened at the current market price. To open a pending order, you should set a non-zero value for the **Signal\_PriceLevel** parameter, i.e. **Signal\_PriceLevel** can be both negative and positive.

The value of **Signal\_PriceLevel** is usually a quite big number. The difference between negative and positive values is shown below:

**Signal\_PriceLevel=-50**:

![Fig. 1. Signal_PriceLevel=-50](https://c.mql5.com/2/6/Figc12Signal_PriceLevel_-50__1.png)

Fig. 1. Signal\_PriceLevel=-50

**Signal\_PriceLevel=50**:

![Fig. 2. Signal_PriceLevel=50](https://c.mql5.com/2/6/Figk23Signal_PriceLevel_50__1.png)

Fig. 2. Signal\_PriceLevel=50

Thus, if **Signal\_PriceLevel=-50**, a pending order will be opened at the price that is less favorable than the current price, whereas if **Signal\_PriceLevel=50**, a pending order will be opened at the price that is better than the current price.

This version of the Expert Advisor opens Sell Stop and Buy Stop orders.

### 2\. Where Do We Store Data on the Distance From the Price for Opening a Pending Order?

Let's first take a look at the below figure and then proceed to the comments:

![Fig. 3. Storing data on the distance from the current price](https://c.mql5.com/2/6/save_price_level.png)

Fig. 3. Storing data on the distance from the current price

Interpretation of the above figure.

**Expert Advisor** is the Expert Advisor generated using the [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard).

- The **ExtExpert** object of the **[CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert)** class is declared in the Expert Advisor at the global level.
- Then, in the **OnInit**() function of the Expert Advisor, we declare a [pointer](https://www.mql5.com/en/docs/basis/types/object_pointers) to the **signal** object of the **[CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal)** class and the **signal** object is immediately created using the **[new](https://www.mql5.com/en/docs/basis/operators/newoperator)** operator.
- While being in the **OnInit**() function, we call the [InitSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert/cexpertinitsignal) function of the **ExtExpert** object and initialize the **signal object.**
- While being in the **OnInit**() function, we call the [PriceLevel](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalpricelevel) function of the **signal** object which gets the **Signal\_PriceLevel** parameter.

Thus, the **Signal\_PriceLevel** parameter where the distance from the current price is stored and which was declared in the **Expert Advisor** is passed to the **signal** object of the **CExpertSignal** class.

The **CExpertSignal** class stores the value of the distance from the current price in the m\_price\_level variable declared with the protected class scope:

```
class CExpertSignal : public CExpertBase
  {
protected:
   //--- variables
   double            m_base_price;     // base price for detection of level of entering (and/or exit?)
   //--- variables for working with additional filters
   CArrayObj         m_filters;        // array of additional filters (maximum number of fileter is 64)
   //--- Adjusted parameters
   double            m_weight;         // "weight" of a signal in a combined filter
   int               m_patterns_usage; // bit mask of  using of the market models of signals
   int               m_general;        // index of the "main" signal (-1 - no)
   long              m_ignore;         // bit mask of "ignoring" the additional filter
   long              m_invert;         // bit mask of "inverting" the additional filter
   int               m_threshold_open; // threshold value for opening
   int               m_threshold_close;// threshold level for closing
   double            m_price_level;    // level of placing a pending orders relatively to the base price
   double            m_stop_level;     // level of placing of the "stop loss" order relatively to the open price
   double            m_take_level;     // level of placing of the "take profit" order relatively to the open price
   int               m_expiration;     // time of expiration of a pending order in bars
```

### 3\. Structure of the Expert Advisor Generated Using the MQL5 Wizard

The Expert Advisor consists of several blocks with different functionality.

![Fig. 4. Structure of the Expert Advisor](https://c.mql5.com/2/6/structure_expert__1.png)

Fig. 4. Structure of the Expert Advisor

Interpretation of the above figure:

- **Expert Advisor** is the Expert Advisor generated using the MQL5 Wizard.
- **CExpert** is the base class for implementation of trading strategies.
- **CExpertSignal** is the base class for creating trading signal generators.

- **filter** 0... **filter** n are trading signal generators, the CExpertSignal class descendants. It should be noted that [our trading system](https://www.mql5.com/en/articles/723#trade_system) is based on the trading signal generator of the [Envelopes indicator](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes), but the signals within the generator have been modified. We will talk about those changes in section 7.

### 4\. Expert Advisor Blocks Advisable for Modification

As you could see from the [structure of the Expert Advisor](https://www.mql5.com/en/articles/723#structure_expert) generated using the MQL5 Wizard, there are base class blocks. Base classes are part of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary).

The classes per se are descendants of other base classes and they in turn consist of one or more base classes. Below you can find the first few lines of the code of two classes - **CExpert** and **CExpertSignal**:

```
//+------------------------------------------------------------------+
//|                                                       Expert.mqh |
//|                   Copyright 2009-2013, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#include "ExpertBase.mqh"
#include "ExpertTrade.mqh"
#include "ExpertSignal.mqh"
#include "ExpertMoney.mqh"
#include "ExpertTrailing.mqh"
//+------------------------------------------------------------------+
.
.
.
class CExpert : public CExpertBase
```

and

```
//+------------------------------------------------------------------+
//|                                                 ExpertSignal.mqh |
//|                   Copyright 2009-2013, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#include "ExpertBase.mqh"
.
.
.
class CExpertSignal : public CExpertBase
```

**I am strongly against any modifications of the base classes**:

1. When MetaEditor is updated, all changes you make to the base classes are overridden and the base classes are restored to their initial state.
2. [Inheritance](https://www.mql5.com/en/docs/basis/oop/inheritance) would be more appropriate in this case. But then you will have to modify the ENTIRE Standard Library.

Instead, it would be best to modify the block of the Expert Advisor and trading signal generator modules, especially since [our trading system](https://www.mql5.com/en/articles/723#trade_system) will already have one modified module in use - the [trading signal generator of the Envelopes indicator](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_envelopes).

**So, that's settled**: we will make changes to the blocks of the Expert Advisor and the block of the trading signal generator.

### 5\. The Implementation Logic

The pointer will be passed from the Expert Advisor to the trading signal generator.

For this purpose, we need to additionally declare a variable with the protected scope and write a method that stores the pointer from the Expert Advisor in the internal variable:

![ Fig. 5. The Implementation Logic](https://c.mql5.com/2/6/scheme_ideas__1.png)

Fig. 5. The Implementation Logic

### 6\. Trading System

The chart time frame is **D1**. The indicator to be used is [Envelopes](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes) with the averaging period of 13 and Exponential averaging method. Types of orders that the Expert Advisor can open are **Sell Stop** and **Buy Stop**.

If the previous bar was bullish, we set a Sell Stop order. If the previous bar was bearish, we set a Buy Stop order. In other words, we hope for the pullback:

![Fig. 6. Trading System](https://c.mql5.com/2/6/Figl6xtrading_strategy_1.png)

Fig. 6. Trading System

To generate trading signals as required by the trading system, the standard module of the trading signal generator SignalEnvelopes.mqh has been modified.

Note that here you can use any trading signal generator from the Standard Library.

### 7\. Trading Signal Generator Modification. Getting the Bar Price

So, let's start. I need to say that I prefer saving my programs in [MQL5 Storage](https://www.metatrader5.com/en/metaeditor/help/mql5storage).

The first thing we should do in order to start modifying the trading signal generator is to create a blank include file, delete everything from it and paste the entire contents of the [standard trading signal generator of the Envelopes indicator](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_envelopes).

By default the trading signal generator must be located under ... **MQL5\\Include\\Expert\\Signal**. Not to overload the ... **\\Signal** folder of the Standard Library with too much information, let's create a new folder under the ... **\\Expert** folder and call it **\\MySignals**:

![Fig. 7. Creating the MySignals folder](https://c.mql5.com/2/6/MySignals_v2__1.png)

Fig. 7. Creating the MySignals folder

Next, we will create an [include file](https://www.mql5.com/en/docs/basis/preprosessor/include) using the MQL5 Wizard.

In MetaEditor, select 'New' under the File menu and then select 'Include file (\*.mqh)'.

![Fig. 8. MQL5 Wizard. Creating an include file](https://c.mql5.com/2/6/Figh84include_file_v2__1.png)

Fig. 8. MQL5 Wizard. Creating an include file

The name of our signal generator class will be **MySignalEnvelopes**.

And it will be located under: **Include\\Expert\\MySignals\\MySignalEnvelopes**. Let's specify it:

![Fig. 9. MQL5 Wizard. Location of the include file](https://c.mql5.com/2/6/Fig99blocation_of_the_include_file_v2__1.png)

Fig. 9. MQL5 Wizard. Location of the include file

After you click 'Finish', the MQL5 Wizard will generate an empty template.

The generated MySignalEnvelopes.mqh file must then be added to MQL5 Storage:

![Fig. 10. MQL5 Storage. Adding the file](https://c.mql5.com/2/6/add_to_storage__1.png)

Fig. 10. MQL5 Storage. Adding the file

Once the file has been added, we need to commit the changes to MQL5 Storage:

![Fig. 11. MQL5 Storage. Committing the changes](https://c.mql5.com/2/6/fix_file_on_storage__1.png)

Fig. 11. MQL5 Storage. Committing the changes

Having completed the above steps, we can proceed to modifying our trading signal generator.

Since the generator is based on the **\\Include\\Expert\\Signal\\SignalEnvelopes.mqh** file, we copy the entire contents of the file and paste it into the generator file, only leaving the original header:

```
//+------------------------------------------------------------------+
//|                                            MySignalEnvelopes.mqh |
//|                              Copyright © 2013, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2013, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#include <Expert\ExpertSignal.mqh>
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Signals of indicator 'Envelopes'                           |
//| Type=SignalAdvanced                                              |
//| Name=Envelopes                                                   |
//| ShortName=Envelopes                                              |
//| Class=CSignalEnvelopes                                           |
//| Page=signal_envelopes                                            |
//| Parameter=PeriodMA,int,45,Period of averaging                    |
//| Parameter=Shift,int,0,Time shift                                 |
//| Parameter=Method,ENUM_MA_METHOD,MODE_SMA,Method of averaging     |
//| Parameter=Applied,ENUM_APPLIED_PRICE,PRICE_CLOSE,Prices series   |
//| Parameter=Deviation,double,0.15,Deviation                        |
//+------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//| Class CSignalEnvelopes.                                          |
//| Purpose: Class of generator of trade signals based on            |
//|          the 'Envelopes' indicator.                              |
//| Is derived from the CExpertSignal class.                         |
//+------------------------------------------------------------------+
class CSignalEnvelopes : public CExpertSignal
  {
protected:
   CiEnvelopes       m_env;            // object-indicator
   //--- adjusted parameters
   int               m_ma_period;      // the "period of averaging" parameter of the indicator
   int               m_ma_shift;       // the "time shift" parameter of the indicator
   ENUM_MA_METHOD    m_ma_method;      // the "method of averaging" parameter of the indicator
   ENUM_APPLIED_PRICE m_ma_applied;    // the "object of averaging" parameter of the indicator
   double            m_deviation;      // the "deviation" parameter of the indicator
   double            m_limit_in;       // threshold sensitivity of the 'rollback zone'
   double            m_limit_out;      // threshold sensitivity of the 'break through zone'
   //--- "weights" of market models (0-100)
   int               m_pattern_0;      // model 0 "price is near the necessary border of the envelope"
   int               m_pattern_1;      // model 1 "price crossed a border of the envelope"

public:
                     CSignalEnvelopes(void);
                    ~CSignalEnvelopes(void);
   //--- methods of setting adjustable parameters
   void              PeriodMA(int value)                 { m_ma_period=value;        }
   void              Shift(int value)                    { m_ma_shift=value;         }
   void              Method(ENUM_MA_METHOD value)        { m_ma_method=value;        }
   void              Applied(ENUM_APPLIED_PRICE value)   { m_ma_applied=value;       }
   void              Deviation(double value)             { m_deviation=value;        }
   void              LimitIn(double value)               { m_limit_in=value;         }
   void              LimitOut(double value)              { m_limit_out=value;        }
   //--- methods of adjusting "weights" of market models
   void              Pattern_0(int value)                { m_pattern_0=value;        }
   void              Pattern_1(int value)                { m_pattern_1=value;        }
   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are formed
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);

protected:
   //--- method of initialization of the indicator
   bool              InitMA(CIndicators *indicators);
   //--- methods of getting data
   double            Upper(int ind)                      { return(m_env.Upper(ind)); }
   double            Lower(int ind)                      { return(m_env.Lower(ind)); }
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSignalEnvelopes::CSignalEnvelopes(void) : m_ma_period(45),
                                           m_ma_shift(0),
                                           m_ma_method(MODE_SMA),
                                           m_ma_applied(PRICE_CLOSE),
                                           m_deviation(0.15),
                                           m_limit_in(0.2),
                                           m_limit_out(0.2),
                                           m_pattern_0(90),
                                           m_pattern_1(70)
  {
//--- initialization of protected data
   m_used_series=USE_SERIES_OPEN+USE_SERIES_HIGH+USE_SERIES_LOW+USE_SERIES_CLOSE;
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSignalEnvelopes::~CSignalEnvelopes(void)
  {
  }
//+------------------------------------------------------------------+
//| Validation settings protected data.                              |
//+------------------------------------------------------------------+
bool CSignalEnvelopes::ValidationSettings(void)
  {
//--- validation settings of additional filters
   if(!CExpertSignal::ValidationSettings())
      return(false);
//--- initial data checks
   if(m_ma_period<=0)
     {
      printf(__FUNCTION__+": period MA must be greater than 0");
      return(false);
     }
//--- ok
   return(true);
  }
//+------------------------------------------------------------------+
//| Create indicators.                                               |
//+------------------------------------------------------------------+
bool CSignalEnvelopes::InitIndicators(CIndicators *indicators)
  {
//--- check pointer
   if(indicators==NULL)
      return(false);
//--- initialization of indicators and timeseries of additional filters
   if(!CExpertSignal::InitIndicators(indicators))
      return(false);
//--- create and initialize MA indicator
   if(!InitMA(indicators))
      return(false);
//--- ok
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialize MA indicators.                                        |
//+------------------------------------------------------------------+
bool CSignalEnvelopes::InitMA(CIndicators *indicators)
  {
//--- check pointer
   if(indicators==NULL)
      return(false);
//--- add object to collection
   if(!indicators.Add(GetPointer(m_env)))
     {
      printf(__FUNCTION__+": error adding object");
      return(false);
     }
//--- initialize object
   if(!m_env.Create(m_symbol.Name(),m_period,m_ma_period,m_ma_shift,m_ma_method,m_ma_applied,m_deviation))
     {
      printf(__FUNCTION__+": error initializing object");
      return(false);
     }
//--- ok
   return(true);
  }
//+------------------------------------------------------------------+
//| "Voting" that price will grow.                                   |
//+------------------------------------------------------------------+
int CSignalEnvelopes::LongCondition(void)
  {
   int result=0;
   int idx   =StartIndex();
   double close=Close(idx);
   double upper=Upper(idx);
   double lower=Lower(idx);
   double width=upper-lower;
//--- if the model 0 is used and price is in the rollback zone, then there is a condition for buying
   if(IS_PATTERN_USAGE(0) && close<lower+m_limit_in*width && close>lower-m_limit_out*width)
      result=m_pattern_0;
//--- if the model 1 is used and price is above the rollback zone, then there is a condition for buying
   if(IS_PATTERN_USAGE(1) && close>upper+m_limit_out*width)
      result=m_pattern_1;
//--- return the result
   return(result);
  }
//+------------------------------------------------------------------+
//| "Voting" that price will fall.                                   |
//+------------------------------------------------------------------+
int CSignalEnvelopes::ShortCondition(void)
  {
   int result  =0;
   int idx     =StartIndex();
   double close=Close(idx);
   double upper=Upper(idx);
   double lower=Lower(idx);
   double width=upper-lower;
//--- if the model 0 is used and price is in the rollback zone, then there is a condition for selling
   if(IS_PATTERN_USAGE(0) && close>upper-m_limit_in*width && close<upper+m_limit_out*width)
      result=m_pattern_0;
//--- if the model 1 is used and price is above the rollback zone, then there is a condition for selling
   if(IS_PATTERN_USAGE(1) && close<lower-m_limit_out*width)
      result=m_pattern_1;
//--- return the result
   return(result);
  }
//+------------------------------------------------------------------+
```

Now, we will be working on modifications of some parts of the code.

To avoid confusion, the modified code will be highlighted:

```
//+------------------------------------------------------------------+
//|                                                     MySignal.mqh |
//|                              Copyright © 2013, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
```

The modified code is the code that needs to be copied and pasted into the trading signal generator. I hope that such highlighting will help you better understand the code.

Since we are writing our own class of the trading signal generator, its name should be different from the name of the base class. We therefore replace CSignalEnvelopes with CMySignalEnvelopes throughout the entire code:

![Fig. 12. Renaming the class](https://c.mql5.com/2/6/Figt12vreplacement_v2__1.png)

Fig. 12. Renaming the class

To ensure that the trading signal generator class is displayed in the MQL5 Wizard under its name, change the class name in the description block

```
//| Title=Signals of indicator 'Envelopes'                           |
```

to

```
//| Title=Signals of indicator 'MySignalEnvelopes'                   |
```

Change the MA period value

```
//| Parameter=PeriodMA,int,45,Period of averaging                    |
```

to 13 (this is only my suggestion, you can set any value you prefer)

```
//| Parameter=PeriodMA,int,13,Period of averaging                    |
```

In addition, we also modify the Deviation parameter

```
//| Parameter=Deviation,double,0.15,Deviation                        |
```

by setting a greater value

```
//| Parameter=Deviation,double,1.15,Deviation                        |
```

According to our [implementation logic](https://www.mql5.com/en/articles/723#scheme_ideas), we need to declare an internal variable that will store the pointer to the main signal.

Since this must be an internal variable (within the trading signal generator class scope only), it will be added to the following code block:

```
protected:
   CiEnvelopes       m_env;          // object-indicator
   //--- adjusted parameters
   int               m_ma_period;    // the "period of averaging" parameter of the indicator
   int               m_ma_shift;     // the "time shift" parameter of the indicator
   ENUM_MA_METHOD    m_ma_method;     // the "method of averaging" parameter of the indicator
   ENUM_APPLIED_PRICE m_ma_applied;    // the "object of averaging" parameter of the indicator
   double            m_deviation;    // the "deviation" parameter of the indicator
   //--- "weights" of market models (0-100)
   int               m_pattern_0;      // model 0
   CExpertSignal    *m_signal;         // storing the pointer to the main signal
```

Please also note that I deleted the unnecessary variables from the code.

The method for storing the pointer to the main signal will be declared in another code block - the 'method of setting the pointer to the main signal'. Here, I also deleted some irrelevant methods.

```
public:
                     CMySignalEnvelopes(void);
                    ~CMySignalEnvelopes(void);
   //--- methods of setting adjustable parameters
   void              PeriodMA(int value)                 { m_ma_period=value;        }
   void              Shift(int value)                    { m_ma_shift=value;         }
   void              Method(ENUM_MA_METHOD value)        { m_ma_method=value;        }
   void              Applied(ENUM_APPLIED_PRICE value)   { m_ma_applied=value;       }
   void              Deviation(double value)             { m_deviation=value;        }
   //--- methods of adjusting "weights" of market models
   void              Pattern_0(int value)                { m_pattern_0=value;        }
   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are formed
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);
   //--- method of setting the pointer to the main signal
   virtual bool      InitSignal(CExpertSignal *signal=NULL);
```

Let's now specify some modified parameters in the constructor and delete the variables that are no longer needed:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CMySignalEnvelopes::CMySignalEnvelopes(void) : m_ma_period(13),
                                               m_ma_shift(0),
                                               m_ma_method(MODE_SMA),
                                               m_ma_applied(PRICE_CLOSE),
                                               m_deviation(1.15),
                                               m_pattern_0(50)
```

At this point, we can proceed to modifying the trading signal generation logic according to [our trading system](https://www.mql5.com/en/articles/723#trade_system).

The code block responsible for a buy signal:

```
int CMySignalEnvelopes::LongCondition(void)
  {
   int result=0;
   int idx   =StartIndex();
   double close=Close(idx);
   double upper=Upper(idx);
   double lower=Lower(idx);
   double width=upper-lower;
//--- if the model 0 is used and price is in the rollback zone, then there is a condition for buying
   if(IS_PATTERN_USAGE(0) && close<lower+m_limit_in*width && close>lower-m_limit_out*width)
      result=m_pattern_0;
//--- if the model 1 is used and price is above the rollback zone, then there is a condition for buying
   if(IS_PATTERN_USAGE(1) && close>upper+m_limit_out*width)
      result=m_pattern_1;
//--- return the result
   return(result);
  }
```

will be as shown below, following the necessary changes:

```
int CMySignalEnvelopes::LongCondition(void) //---buy
  {
   int result=0;
   int idx   =StartIndex();
   double open=Open(idx);
   double close=Close(idx);
   double prlevel;
      if(IS_PATTERN_USAGE(0) && close<open)
        {
         prlevel=GetPriceLevelStopp(open,Open(0));
         m_signal.PriceLevel(prlevel);
         result=m_pattern_0;
        }
//--- return the result
   return(result);
  }
```

The code block responsible for a sell signal:

```
int CMySignalEnvelopes::ShortCondition(void)
  {
   int result  =0;
   int idx     =StartIndex();
   double close=Close(idx);
   double upper=Upper(idx);
   double lower=Lower(idx);
   double width=upper-lower;
//--- if the model 0 is used and price is in the rollback zone, then there is a condition for selling
   if(IS_PATTERN_USAGE(0) && close>upper-m_limit_in*width && close<upper+m_limit_out*width)
      result=m_pattern_0;
//--- if the model 1 is used and price is above the rollback zone, then there is a condition for selling
   if(IS_PATTERN_USAGE(1) && close<lower-m_limit_out*width)
      result=m_pattern_1;
//--- return the result
   return(result);
  }
```

will be as shown below, following the necessary changes:

```
int CMySignalEnvelopes::ShortCondition(void) //---sell
  {
   int result  =0;
   int idx     =StartIndex();
   double open=Open(idx);
   double close=Close(idx);
   double prlevel;
      if(IS_PATTERN_USAGE(0) && close>open)
        {
         prlevel=GetPriceLevelStopp(Open(0),open);
         m_signal.PriceLevel(prlevel);
         result=m_pattern_0;
        }
//--- return the result
   return(result);
  }
```

### 8\. A Few Comments on the Signal Code Block

If the required condition for a certain signal is met, we call the **GetPriceLevelStopp** method that returns a number like "20" or "15" - the value of the distance from the current price.

This is followed by calling the **PriceLevel** method of the **m\_signal** object (which sets the distance for determining the pending order level price). It should be reminded that m\_signal is the **[CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal)** class object that stores the pointer to the main signal.

The code of the GetPriceLevelStopp method is provided below:

```
double CMySignalEnvelopes::GetPriceLevelStopp(double price_0,double min)
  {
   double level;
   double temp;
   temp-=(price_0-min)/PriceLevelUnit();
   level=NormalizeDouble(temp,0);
   return(level);
  }
```

We need to declare this method in the class header:

```
protected:
   //--- method of initialization of the indicator
   bool              InitMA(CIndicators *indicators);
   //--- methods of getting data
   double            Upper(int ind)                      { return(m_env.Upper(ind)); }
   double            Lower(int ind)                      { return(m_env.Lower(ind)); }
   double            GetPriceLevelStopp(double price,double min);
  };
```

Another method that we will need is the method of passing the pointer to the main signal to the internal variable:

```
bool CMySignalEnvelopes::InitSignal(CExpertSignal *signal)
  {
   m_signal=signal;
   return(true);
  }
```

After that we should create an Expert Advisor in the MQL5 Wizard and include in it the signal module ' **MySignalEnvelopes**'.

We also need to add the InitSignal method call to the code of the Expert Advisor generated using the MQL5 Wizard:

```
//--- Set filter parameters
   filter0.PeriodMA(Signal_Envelopes_PeriodMA);
   filter0.Shift(Signal_Envelopes_Shift);
   filter0.Method(Signal_Envelopes_Method);
   filter0.Applied(Signal_Envelopes_Applied);
   filter0.Deviation(Signal_Envelopes_Deviation);
   filter0.Weight(Signal_Envelopes_Weight);
   filter0.InitSignal(signal);
//...
```

For better visualization of the operation of the Expert Advisor, I have provided a short video:

YouTube

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Full screen is unavailable. [Learn More](https://support.google.com/youtube/answer/6276924)

More videos

## More videos

Video unavailable

This video is unavailable

[Visit YouTube to search for more videos](https://www.youtube.com/)

## More videos on YouTube

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=rtcZI0RVL_A&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F723)

0:00

0:00 / 0:00

•Live

•

The code of the Expert Advisor generated using the MQL5 Wizard, as well as the code of the signal module, is attached to the article.

Below you can see the testing results of the Expert Advisor. It was tested for EURUSD and USDJPY with the following parameters: testing period 2013.01.01 - 2013.09.01, time frame - D1, Stop Loss level = 85, Take Profit level = 195.

![Fig. 13. Testing for EURUSD on D1](https://c.mql5.com/2/6/Fig013eTest_D1_EURUSD__1.png)

Fig. 13. Testing for EURUSD on D1

![Fig. 14. Testing for USDJPY on D1](https://c.mql5.com/2/6/Fig314pTest_D1_USDJPY__1.png)

Fig. 14. Testing for USDJPY on D1

### Conclusion

We have just seen how we can modify the code of the [trading signal module](https://www.mql5.com/en/docs/standardlibrary/expertclasses) for the implementation of the functionality allowing us to set pending orders at any distance from the current price: it may be the Close or Open price of the previous bar or the value of the moving average. There are plenty of options. Important is that you can set any opening price for a pending order.

The article has demonstrated how we can access the pointer to the main signal, and hence the **[CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) class methods.**  I believe that the article will prove useful to traders who trade with pending orders.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/723](https://www.mql5.com/ru/articles/723)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/723.zip "Download all attachments in the single ZIP archive")

[mysignalenvelopes.mqh](https://www.mql5.com/en/articles/download/723/mysignalenvelopes.mqh "Download mysignalenvelopes.mqh")(9.3 KB)

[expertmysignalenvelopes.mq5](https://www.mql5.com/en/articles/download/723/expertmysignalenvelopes.mq5 "Download expertmysignalenvelopes.mq5")(7.42 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/14182)**
(20)


![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
8 Jun 2022 at 15:22

Hi [@Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn) \-\- that was a very instructive article, thanks for that!

I personally find this OOP framework in MQL5 quite interesting for building bots by composing objects representing experts, signals, filters, indicators, risk managers, and so on -- very elegant approach in my opinion, as it favours code reuse and extensibility without apparently sacrificing much power... however due to its complexity the learning curve seems fairly steep.

In any case, as I'm transitioning to MQL5 exclusively and have good experience in OOP concepts/languages, I am really keen to adapt it for my own use in prototyping new trading ideas and developing trading systems studying. I have been studying and playing with the library code and was wondering about your recommended best practice approach for the following:

### QUESTION: How would you integrate a trend filter for signals received in the expert?

The library includes CExpertBase::m\_trend\_type property but unfortunately it is not really used anywhere in the examples provided with the platform. I am divided between two design possibilities... Adding a _trend filtering object_ directly in my subclass of CExpert (see code snippet below), which could offer more control on making trading decisions at the level of the expert. Another way to solve it could involve fiddling with the filters of my main signal object and somehow calculate the trend and make a decision inside my subclass of CSignalExpert, e.g. inside CSignalWithTrendFilter::Direction(). Not quite sure yet what are the advantages and disadvantages of one method versus the other, and which one will provide me with more flexibility for requirements of my future projects, i.e. more code reuse without complications and less tweaking of my base classes.

```
class CExpertWithTrendFilter : public CExpert
{
protected:
   CExpertSignal    *m_trend;   // work in parallel with CExpert::m_signal to filter the signals it generates
// ...
   virtual bool      Processing(void);
   virtual bool      CheckOpen(void);
// ...
};

bool CExpertWithTrendFilter::Processing(void)
{
   CExpertBase::m_trend_type = (ENUM_TYPE_TREND) m_trend.Direction();   // determine current trend based from specialised object derived from CExpertSignal

   m_signal.TrendType(m_trend_type);             // pass trend type/strength as input to m_signal, also subclassed from CExpertSignal
   m_signal.SetDirection();                      // OPTION #1 >>> calculate signal direction, possibly taking trend established above into consideration (or not)

// ...

   if(CheckOpen())                               // OPTION #2 >>> alternatively, trend type/strength can be checked by expert before opening long or short
      return(true);

   retun(false);                                 // return without any operations
}

bool CExpertWithTrendFilter::CheckOpen(void)
{
   if(m_trend_type > TYPE_TREND_FLAT && CheckOpenLong())    // only allow opening long if trend filter direction agrees
      return(true);
   if(m_trend_type < TYPE_TREND_FLAT && CheckOpenShort())   // only allow opening short if trend filter direction agrees
      return(true);

   return(false);                                // return without any operations
}
```

Thanks in advance for you help and recommendations.

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
8 Jun 2022 at 15:28

I switched to my trading engine a long time ago [An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717) \- it's more flexible.


![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
8 Jun 2022 at 15:48

**Vladimir Karputov [#](https://www.mql5.com/en/forum/14182/page2#comment_40050702):**

I switched to my trading engine a long time ago [An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717) \- it's more flexible.

OK, I see -- I'm reading the article and will have a look at the attached code... in any case, I'd still appreciate your comment/opinion on the question above if you don't mind. Thanks a lot!

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
8 Jun 2022 at 16:00

**Dima Diall [#](https://www.mql5.com/en/forum/14182/page2#comment_40050829) :**

OK, I see -- I'm reading the article and will have a look at the attached code... in any case, I'd still appreciate your comment/opinion on the question above if you don't mind. Thanks a lot!

CExpertSignal is the past. No comments.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
8 Jun 2022 at 16:14

**Vladimir Karputov [#](https://www.mql5.com/en/forum/14182/page2#comment_40050927):**

CExpertSignal is the past. No comments.

:-)

![MetaTrader AppStore Results for Q3 2013](https://c.mql5.com/2/0/avatar3.png)[MetaTrader AppStore Results for Q3 2013](https://www.mql5.com/en/articles/769)

Another quarter of the year has passed and we have decided to sum up its results for MetaTrader AppStore - the largest store of trading robots and technical indicators for MetaTrader platforms. More than 500 developers have placed over 1 200 products in the Market by the end of the reported quarter.

![MQL5 Cookbook: Reducing the Effect of Overfitting and Handling the Lack of Quotes](https://c.mql5.com/2/0/Reduce_Overfitting_avatar.png)[MQL5 Cookbook: Reducing the Effect of Overfitting and Handling the Lack of Quotes](https://www.mql5.com/en/articles/652)

Whatever trading strategy you use, there will always be a question of what parameters to choose to ensure future profits. This article gives an example of an Expert Advisor with a possibility to optimize multiple symbol parameters at the same time. This method is intended to reduce the effect of overfitting parameters and handle situations where data from a single symbol are not enough for the study.

![Lite_EXPERT2.mqh: Functional Kit for Developers of Expert Advisors](https://c.mql5.com/2/17/812_123.gif)[Lite\_EXPERT2.mqh: Functional Kit for Developers of Expert Advisors](https://www.mql5.com/en/articles/1380)

This article continues the series of articles "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization". It familiarizes the readers with a more universal function library of the Lite\_EXPERT2.mqh file.

![EA Status SMS Notifications](https://c.mql5.com/2/17/831_34.png)[EA Status SMS Notifications](https://www.mql5.com/en/articles/1376)

Developing a system of SMS notifications that informs you of the status of your EA so that you are always aware of any critical situation, wherever you may be.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/723&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070557610261813254)

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