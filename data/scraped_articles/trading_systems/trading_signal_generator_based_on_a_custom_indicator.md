---
title: Trading Signal Generator Based on a Custom Indicator
url: https://www.mql5.com/en/articles/691
categories: Trading Systems
relevance_score: 4
scraped_at: 2026-01-23T17:47:37.137437
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/691&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068656687801367559)

MetaTrader 5 / Examples


### Introduction

In this article, I will tell you how to create a trading signal generator based on a custom indicator. You will see how you can write your own trading model for a custom indicator. I will also explain the purpose of model 0 and why **IS\_PATTERN\_USAGE(0)**-type structures are used in the trading signal module.

The article will use two types of code: the code we are about to modify and the code we already modified. The modified code will be highlighted as follows:

```
//+------------------------------------------------------------------+
//|                                                     MySignal.mqh |
//|                              Copyright © 2012, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
```

The modified code is the code to be copied and pasted into the trading signal generator. I hope you will understand the code better through the use of highlighting.

### 1\. Custom Indicator

I am sure there must be an indicator not included in the standard delivery that you have been wanting to use for a long time. And that is the indicator based on which you want to build a trading signal module. I will use the MACD indicator from the standard delivery as such an indicator. The location of the indicator is as follows: ... **MQL5\\Indicators\\Examples\\MACD.mq5**.

Each indicator can describe one or more market models. A market model is a certain combination of the indicator value and the price value. The [models available](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_macd) for the MACD indicator are reversal, crossover of the main and the signal line, crossover of the zero level, divergence and double divergence.

**1.1 New Indicator Model**.

Let's assume that we are not happy with the given market models available for the indicator and want to introduce our own **indicator model**. The new indicator model description: if the MACD indicator is below the zero line and its values are increasing, we can expect further growth and open a long position:

![Figure 1: Model of prospective indicator growth](https://c.mql5.com/2/5/growth_is_possible_v2.png)

Figure 1: Model of prospective indicator growth

if the MACD indicator is above the zero line and its values are decreasing, we can expect further decrease and open a short position:

![Figure 2: Model of prospective indicator fall ](https://c.mql5.com/2/5/lowering_is_possible_v2.png)

Figure 2: Model of prospective indicator fall

So, we have decided on the custom indicator and come up with the new trading model for the indicator and its description. Let's proceed with writing the code.

### 2\. Writing the Trading Signal Generator Based on Our Custom Indicator

Our generator is the descendant of the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) base class. The [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) base class is a class for creating trading signal generators. The [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) class contains a set of [public](https://www.mql5.com/en/docs/basis/oop/incapsulation)(i.e. externally accessible) methods which allow an Expert Advisor to see the indication of the trading signal generator regarding the direction of entry to the market.

Since we are working on our own trading signal generator, it should be inherited from the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) class, with the relevant [virtual methods](https://www.mql5.com/en/docs/basis/oop/virtual) redefined (filled with the corresponding code).

### 3\. Creating the Class of the Trading Signal Generator

The trading signal generator should by default be located in **...MQL5\\Include\\Expert\\Signal** folder. Not to overload the **...\\Signal** folder of the Standard Library with too much information, let's create a new folder under the **...\\Expert** folder and call it **\\MySignals**:

![Figure 3. Creating the new MySignals folder ](https://c.mql5.com/2/5/MySignals_v2.png)

Figure 3. Creating the new MySignals folder

Next, we will create an [include file](https://www.mql5.com/en/docs/basis/preprosessor/include) using the MQL5 Wizard. In MetaEditor, select "New" under the File menu and then select "Include File (\*.mqh)".

![Figure 4. MQL5 Wizard. Creating an include file](https://c.mql5.com/2/5/fig1_691__1.png)

Figure 4. MQL5 Wizard. Creating an include file

The name of the class of signal generator will be **MySignal**. It will be located under **Include\\Expert\\MySignals\\MySignal**. Let's specify it:

![Figure 5. MQL5 Wizard. Location of the include file](https://c.mql5.com/2/5/fig2_691__2.png)

Figure 5. MQL5 Wizard. Location of the include file

After clicking "Finish", the MQL5 Wizard will generate an empty template. From this moment on, we will do everything manually and copy/paste data. I would like to draw your attention to the fact that internally, all the signals from the [Standard Library](https://www.mql5.com/en/docs/standardlibrary/expertclasses) are almost identical. They only differ in algorithms used to determine trading models.

Therefore, you can take any file from the **\\Include\\Expert\\Signal folder,** copy its contents and paste it into your template. You can then start editing the resulting file of the trading signal generator.

### 4\. Description of the Class of the Trading Signal Generator

As a template, I took the **\\Include\\Expert\\Signal\\SignalEnvelopes.mqh file** from which I copied everything but the header:

```
//+------------------------------------------------------------------+
//|                                              SignalEnvelopes.mqh |
//|                   Copyright 2009-2013, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
```

and pasted it all into our almost empty **MySignal.mqh** template. This is what I got:

```
//+------------------------------------------------------------------+
//|                                                     MySignal.mqh |
//|                              Copyright © 2012, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
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
//| It is derived from the CExpertSignal class.                      |
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
   //--- method of creating the indicator and time series
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
//--- initialization of indicators and time series of additional filters
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
//| "Voting" that the price will grow.                               |
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
//| "Voting" that the price will fall.                               |
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

Note line 6:

```
#include <Expert\ExpertSignal.mqh>
```

Here we give an order to the preprocessor to include the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) base class for creating trading signal generators in our template.

We will continue editing the template. To ensure that our template is visible later on to the MQL5 Wizard, we need to change the description of our class:

```
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
```

So, let's see. The line

```
//| Title=Signals of indicator 'Envelopes'                           |
```

shows the name of our signal class under which it will be displayed in the MQL5 Wizard. We will change this name to something like this:

```
//| Title=Signal of the 'Custom Indicator' indicator             |
```

The next line:

```
//| Name=Envelopes                                                   |
```

indicates the name for describing variables of our trading signal class. This description will be used by the MQL5 Wizard. Let's modify this line as follows:

```
//| Name=MyCustomIndicator                                           |
```

The next line:

```
//| ShortName=Envelopes                                              |
```

We will give the same name to this parameter:

```
//| ShortName=MyCustomIndicator                                      |
```

The following line sets the class name:

```
//| Class=CSignalEnvelopes                                           |
```

Let's rename this parameter:

```
//| Class=CSignalMyCustInd                                           |
```

Leave the next parameter as is.

```
//| Page=signal_envelopes                                            |
```

The following parameter group is responsible for description of parameters of the indicator underlying the trading signal generator. As I mentioned earlier, I will use **...MQL5\\Indicators\\Examples\\MACD.mq5** as the custom indicator. It has the following parameters:

```
//--- input parameters
input int                InpFastEMA=12;               // Fast EMA period
input int                InpSlowEMA=26;               // Slow EMA period
input int                InpSignalSMA=9;              // Signal SMA period
input ENUM_APPLIED_PRICE  InpAppliedPrice=PRICE_CLOSE; // Applied price
```

**4.1 Parameter Description Block**

Please note that the parameters given above apply only to **MACD.mq5.** Your custom indicator may have completely different parameters. The main thing here is to match the indicator parameters with their descriptions in the trading signal class. The **parameter description block** in the trading signal class for the custom indicator under consideration, **MACD.mq5**, will be as follows:

```
//| Parameter=PeriodFast,int,12,Period of fast EMA                   |
//| Parameter=PeriodSlow,int,24,Period of slow EMA                   |
//| Parameter=PeriodSignal,int,9,Period of averaging of difference   |
//| Parameter=Applied,ENUM_APPLIED_PRICE,PRICE_CLOSE,Prices series   |
```

Take a look at how the parameters in the indicator now match the descriptions in the class description block. Following all the modifications, the description block of our class will be as follows:

```
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Signal of the 'Custom Indicator' indicator                 |
//| Type=SignalAdvanced                                              |
//| Name=MyCustomIndicator                                           |
//| ShortName=MyCustomIndicator                                      |
//| Class=CSignalMyCustInd                                           |
//| Page=signal_envelopes                                            |
//| Parameter=PeriodFast,int,12,Period of fast EMA                   |
//| Parameter=PeriodSlow,int,24,Period of slow EMA                   |
//| Parameter=PeriodSignal,int,9,Period of averaging of difference   |
//| Parameter=Applied,ENUM_APPLIED_PRICE,PRICE_CLOSE,Prices series   |
//+------------------------------------------------------------------+
```

In programming, it is considered good practice to provide comments to one's code, thus making it easier to understand the code, when getting back to it after some time has passed. So, we will modify the following block:

```
//+------------------------------------------------------------------+
//| Class CSignalEnvelopes.                                          |
//| Purpose: Class of generator of trade signals based on            |
//|          the 'Envelopes' indicator.                              |
//| It is derived from the CExpertSignal class.                      |
//+------------------------------------------------------------------+
```

to match the description of our class:

```
//+------------------------------------------------------------------+
//| Class CSignalMyCustInd.                                          |
//| Purpose: Class of the trading signal generator based on          |
//|          the custom indicator.                                   |
//| It is derived from the CExpertSignal class.                      |
//+------------------------------------------------------------------+
```

To avoid confusion, we need to replace all "CSignalEnvelopes" values with "CSignalMyCustInd"

![Figure 6. Replacing CSignalEnvelopes with CSignalMyCustInd ](https://c.mql5.com/2/5/fig3_691.png)

Figure 6. Replacing CSignalEnvelopes with CSignalMyCustInd

Let's now have a look at some theoretical aspects.

### 5\. The CiCustom Class

We will need the [CiCustom](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator) class to continue working on the code of the class of trading indicators of the custom indicator. The [CiCustom](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator) class was created specifically for working with custom indicators. The [CiCustom](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator) class provides creation, setting up and access to custom indicator data.

### 6\. The CIndicators Class.

[CIndicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2) is the class for collecting instances of time series and technical indicator classes. The [CIndicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2) class provides creation, storage and management (data synchronization, handle and memory management) of technical indicator class instances.

We are particularly interested in the [CIndicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2) class because of the [Create](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2/cindicatorscreate) method. This method creates an indicator of a specified type with specified parameters.

### 7\. Continue Writing Our Trading Signal Class

The next code block we are going to modify (lines 28-42) is as follows:

```
class CSignalMyCustInd : public CExpertSignal
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
```

### 8\. Creation of the Custom Indicator in the Trading Signal Generator

Take a look at the code block provided above. The line

```
   CiEnvelopes       m_env;            // object-indicator
```

declares an object - the [CiEnvelopes](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/trendindicators/cienvelopes) class indicator. [CiEnvelopes](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/trendindicators/cienvelopes) is the class for working with the technical indicator from the Standard Library. The [CiEnvelopes](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/trendindicators/cienvelopes) class was created based on the technical indicator from the Standard Library. However, we are writing the code of the generator based on our custom indicator. Therefore there is no ready made class for our or your custom indicator in the Standard Library. What we can do is use the [CiCustom](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator) class.

Let's declare our indicator as the [CiCustom](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator) class:

```
   CiCustom          m_mci;            // indicator object "MyCustomIndicator"
```

**8.1 Four Variables**

Do you remember the [parameter description block](https://www.mql5.com/en/articles/691#parameter) in the class? There were three parameters in that description. In the [protected](https://www.mql5.com/en/docs/basis/variables#protected) area of our generator class, we will now declare **four variables** for passing the values to our four parameters:

```
   //--- adjustable parameters
   int               m_period_fast;    // "fast EMA period"
   int               m_period_slow;    // "slow EMA period"
   int               m_period_signal;  // "difference averaging period"
   ENUM_APPLIED_PRICE m_applied;       // "price type"
```

The following code block:

```
   //--- "weights" of market models (0-100)
   int               m_pattern_0;      // model 0 "price is near the necessary border of the envelope"
   int               m_pattern_1;      // model 1 "price crossed a border of the envelope"
```

This code declares variables that give "weight" to trading models of our trading signal generator. Let's replace the block of "weights" with the following code:

```
   //--- "weights" of the market models (0-100)
   int               m_pattern_0;      // model 0 "the oscillator has required direction"
   int               m_pattern_1;      // model 1 "the indicator is gaining momentum - buy; the indicator is falling - sell"
```

### 9\. Model 0

As you remember, at the beginning of the article it was decided to describe only [one new model](https://www.mql5.com/en/articles/691#new_model) that will be generated by our trading signal generator. However, in the above code I specified two market models (model 0 and model 1). Here, model 0 is an important auxiliary model. It is required when trading with pending orders. When applied, model 0 ensures that pending orders move together with the price. Let's take a look at our trading signal generator and the following conditions:

- the MACD custom indicator is below the zero line,

- and its values are increasing,
- we are trading with pending orders set 50 points from the bar opening price (four-digit price value).

These conditions perfectly describe our trading model. Here is how things will be moving: Our trading model conditions will be checked upon appearing of the bar no. 1. What we have: MACD is below the zero line, yet it is gaining momentum. This corresponds to the buy signal. Therefore, we place a pending Buy Stop order:

![Figure 7. Placing a pending Buy Stop order](https://c.mql5.com/2/5/price_up_1_v2__1.png)

Figure 7. Placing a pending Buy Stop order

Upon appearing of the next bar no. 2, the condition check finds that MACD is below zero and is falling. According to our trading model, there are currently no conditions for buying or selling. However, note: as per the CExpertSignal class logic, since there are no conditions either for buying or selling, all pending orders should be DELETED. In this case, if the price goes up suddenly and dramatically, we will miss the opportunity to enter the market long to our advantage as there will be no pending order.

This is where the auxiliary model 0 appears to be very useful. The auxiliary model 0 will apply, provided that:

- the MACD custom indicator is below the zero line.


So we can place a pending Buy Stop order. Since we place an order 50 points from the bar opening price, we, in fact, simply move the pending Buy Stop order according to the price movement:

![Figure 8. Moving the Buy Stop order down ](https://c.mql5.com/2/5/price_down_2_v2.png)

Figure 8. Moving the Buy Stop order down

Thus, by using the auxiliary model 0 we get the opportunity to move a pending order as per the price movement.

### 10\. Further Modifications of the Template Code

The next code block to be modified is as follows:

```
public:
                     CSignalMyCustInd(void);
                    ~CSignalMyCustInd(void);
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
   //--- method of creating the indicator and time series
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are formed
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);


```

In this block, we declare methods of setting adjustable parameters, methods of adjusting weights of trading models, method of verification of settings, indicator initialization method and methods of checking if the market models are generated.

Taking into consideration that we have declared [four variables](https://www.mql5.com/en/articles/691#four_variables) in adjustable parameters, the block of methods for setting the parameters will be as follows:

```
   //--- methods of setting adjustable parameters
   void              PeriodFast(int value)               { m_period_fast=value;           }
   void              PeriodSlow(int value)               { m_period_slow=value;           }
   void              PeriodSignal(int value)             { m_period_signal=value;         }
   void              Applied(ENUM_APPLIED_PRICE value)   { m_applied=value;               }
```

The next code fragment will remain unchanged:

```
   //--- methods of adjusting "weights" of market models
   void              Pattern_0(int value)                { m_pattern_0=value;        }
   void              Pattern_1(int value)                { m_pattern_1=value;        }
   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and time series
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are generated
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);
```

The next code block to be modified is as follows:

```
protected:
   //--- method of initialization of the indicator
   bool              InitMA(CIndicators *indicators);
   //--- methods of getting data
   double            Upper(int ind)                      { return(m_env.Upper(ind)); }
   double            Lower(int ind)                      { return(m_env.Lower(ind)); }
  };
```

This block will be heavily modified. Please note that I am using the [GetData](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicator/cindicatorgetdata) method of the [CIndicator](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicator) class. Names of the called methods will be provided directly in the code:

```
protected:
   //--- indicator initialization method
   bool              InitMyCustomIndicator(CIndicators *indicators);
   //--- methods for getting data
   //- getting the indicator value
   double            Main(int ind) { return(m_mci.GetData(0,ind));      }
   //- getting the signal line value
   double            Signal(int ind) { return(m_mci.GetData(1,ind));    }
   //- difference between two successive indicator values
   double            DiffMain(int ind) { return(Main(ind)-Main(ind+1)); }
   int               StateMain(int ind);
   double            State(int ind) { return(Main(ind)-Signal(ind)); }
   //- preparing data for the search
   bool              ExtState(int ind);
   //- searching the market model with the specified parameters
   bool              CompareMaps(int map,int count,bool minimax=false,int start=0);
  };
```

The next code block is the constructor.

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSignalMyCustInd::CSignalMyCustInd(void) : m_ma_period(45),
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
```

In the constructor, we will change the names of the variables. Further, we will use only two series: USE\_SERIES\_HIGH+USE\_SERIES\_LOW

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSignalMyCustInd::CSignalMyCustInd(void) : m_period_fast(12),
                                           m_period_slow(24),
                                           m_period_signal(9),
                                           m_applied(PRICE_CLOSE),
                                           m_pattern_0(10),
                                           m_pattern_1(50)
  {
//--- initialization of protected data
   m_used_series=USE_SERIES_HIGH+USE_SERIES_LOW;
  }
```

Let's modify the ValidationSettings method of our class.

```
//+------------------------------------------------------------------+
//| Validation settings protected data.                              |
//+------------------------------------------------------------------+
bool CSignalMyCustInd::ValidationSettings(void)
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
```

In the checking block, we check the main condition for the given custom indicator: m\_period\_fast>=m\_period\_slow

```
//+------------------------------------------------------------------+
//| Checking parameters of protected data                            |
//+------------------------------------------------------------------+
bool CSignalMyCustInd::ValidationSettings(void)
  {
//--- validation settings of additional filters
   if(!CExpertSignal::ValidationSettings())
      return(false);
//--- initial data checks
   if(m_period_fast>=m_period_slow)
     {
      printf(__FUNCTION__+": slow period must be greater than fast period");
      return(false);
     }
//--- ok
   return(true);
  }
```

The next block deals with creation of indicators:

```
//+------------------------------------------------------------------+
//| Create indicators.                                               |
//+------------------------------------------------------------------+
bool CSignalMyCustInd::InitIndicators(CIndicators *indicators)
  {
//--- check pointer
   if(indicators==NULL)
      return(false);
//--- initialization of indicators and time series of additional filters
   if(!CExpertSignal::InitIndicators(indicators))
      return(false);
//--- create and initialize MA indicator
   if(!InitMA(indicators))
      return(false);
//--- ok
   return(true);
  }
```

As applied to our custom indicator:

```
//+------------------------------------------------------------------+
//| Creation of indicators.                                          |
//+------------------------------------------------------------------+
bool CSignalMyCustInd::InitIndicators(CIndicators *indicators)
  {
//--- check of pointer is performed in the method of the parent class
//---
//--- initialization of indicators and time series of additional filters
   if(!CExpertSignal::InitIndicators(indicators))
      return(false);
//--- creation and initialization of the custom indicator
   if(!InitMyCustomIndicator(indicators))
      return(false);
//--- ok
   return(true);
  }
```

The following block is the indicator initialization block:

```
//+------------------------------------------------------------------+
//| Initialize MA indicators.                                        |
//+------------------------------------------------------------------+
bool CSignalMyCustInd::InitMA(CIndicators *indicators)
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
```

First, we add an object to the collection. We then set the parameters of our indicator and create the custom indicator using the [Create](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2/cindicatorscreate) method of the [CIndicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2) class:

```
//+------------------------------------------------------------------+
//| Initialization of indicators.                                    |
//+------------------------------------------------------------------+
bool CSignalMyCustInd::InitMyCustomIndicator(CIndicators *indicators)
  {
//--- add an object to the collection
   if(!indicators.Add(GetPointer(m_mci)))
     {
      printf(__FUNCTION__+": error adding object");
      return(false);
     }
//--- set parameters of the indicator
   MqlParam parameters[4];
//---
   parameters[0].type=TYPE_STRING;
   parameters[0].string_value="Examples\\MACD.ex5";
   parameters[1].type=TYPE_INT;
   parameters[1].integer_value=m_period_fast;
   parameters[2].type=TYPE_INT;
   parameters[2].integer_value=m_period_slow;
   parameters[3].type=TYPE_INT;
   parameters[3].integer_value=m_period_signal;
//--- object initialization
   if(!m_mci.Create(m_symbol.Name(),0,IND_CUSTOM,4,parameters))
     {
      printf(__FUNCTION__+": error initializing object");
      return(false);
     }
//--- number of buffers
   if(!m_mci.NumBuffers(4)) return(false);
//--- ok
   return(true);
  }
```

The next block checks buying conditions:

```
//+------------------------------------------------------------------+
//| "Voting" that the price will grow.                               |
//+------------------------------------------------------------------+
int CSignalMyCustInd::LongCondition(void)
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

According to our [model 0 implementation](https://www.mql5.com/en/articles/691#model0), two models are checked:

```
//+------------------------------------------------------------------+
//| "Voting" that the price will grow.                               |
//+------------------------------------------------------------------+
int CSignalMyCustInd::LongCondition(void)
  {
   int result=0;
   int idx   =StartIndex();
//--- check direction of the main line
   if(DiffMain(idx)>0.0)
     {
      //--- the main line goes upwards, which confirms the possibility of the price growth
      if(IS_PATTERN_USAGE(0))
         result=m_pattern_0;      // "confirming" signal number 0
      //--- if the model 1 is used, look for a reverse of the main line
      if(IS_PATTERN_USAGE(1) && DiffMain(idx+1)<0.0)
         result=m_pattern_1;      // signal number 1
     }
//--- return the result
   return(result);
  }
```

The following block checks selling conditions:

```
//+------------------------------------------------------------------+
//| "Voting" that the price will fall.                               |
//+------------------------------------------------------------------+
int CSignalMyCustInd::ShortCondition(void)
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

According to our [model 0 implementation](https://www.mql5.com/en/articles/691#model0), two models are checked:

```
//+------------------------------------------------------------------+
//| "Voting" that the price will fall.                               |
//+------------------------------------------------------------------+
int CSignalMyCustInd::ShortCondition(void)
  {
   int result=0;
   int idx   =StartIndex();
//--- check direction of the main line
   if(DiffMain(idx)<0.0)
     {
            //--- the main line gown downwards, which confirms the possibility of the price fall
      if(IS_PATTERN_USAGE(0))
         result=m_pattern_0;      // "confirming" signal number 0
      //--- if the model 1 is used, look for a reverse of the main line
      if(IS_PATTERN_USAGE(1) && DiffMain(idx+1)>0.0)
         result=m_pattern_1;      // signal number 1
     }
//--- return the result
   return(result);
  }
```

### Conclusion

I hope this article has helped you to understand how you can create a trading signal generator based on your custom indicator.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/691](https://www.mql5.com/ru/articles/691)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/691.zip "Download all attachments in the single ZIP archive")

[mysignal.mqh](https://www.mql5.com/en/articles/download/691/mysignal.mqh "Download mysignal.mqh")(9.38 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/13311)**
(10)


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
31 Aug 2013 at 14:39

**tyn:**

... is it possible to correctly (without rewriting the library) transfer the opening level of a pending order from a [custom indicator](https://www.mql5.com/en/articles/5 "Article Switching to New Rails: Custom Indicators in MQL5")? ...?

As I understand, you want to open pending orders not strictly at a distance of +-50 points, but depending on the situation?


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
31 Aug 2013 at 14:40

**tyn:**

... The trading system assumes that sometimes there can be 2 pending orders, one to buy and one to sell?

No. The system does not assume the existence of two [pending orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 Documentation: Order Properties").


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
31 Aug 2013 at 15:11

**barabashkakvn:**

As I understand, you want to open pending orders not strictly at a distance of +-50 pips, but depending on the situation?

Yes, exactly like that


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
31 Aug 2013 at 15:14

**barabashkakvn:**

No. The system does not assume the existence of two [pending orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 Documentation: Order Properties").

This condition can be circumvented... by reopening pending orders as they approach the opening price.


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
31 Aug 2013 at 15:33

**tyn:**

**barabashkakvn:**

As I understand, you want to open pending orders not strictly at a distance of +-50 pips, but depending on the situation?

Yes exactly like that

It can be done. How to do it will be described in a new article.


![Expert Advisor for Trading in the Channel](https://c.mql5.com/2/17/834_22.gif)[Expert Advisor for Trading in the Channel](https://www.mql5.com/en/articles/1375)

The Expert Advisor plots the channel lines. The upper and lower channel lines act as support and resistance levels. The Expert Advisor marks datum points, provides sound notification every time the price reaches or crosses the channel lines and draws the relevant marks. Upon fractal formation, the corresponding arrows appear on the last bars. Line breakouts may suggest the possibility of a growing trend. The Expert Advisor is extensively commented throughout.

![Simple Methods of Forecasting Directions of the Japanese Candlesticks](https://c.mql5.com/2/17/836_34.png)[Simple Methods of Forecasting Directions of the Japanese Candlesticks](https://www.mql5.com/en/articles/1374)

Knowing the direction of the price movement is sufficient for getting positive results from trading operations. Some information on the possible direction of the price can be obtained from the Japanese candlesticks. This article deals with a few simple approaches to forecasting the direction of the Japanese candlesticks.

![Extending MQL5 Standard Library and Reusing Code](https://c.mql5.com/2/0/regular-polyhedra-02.png)[Extending MQL5 Standard Library and Reusing Code](https://www.mql5.com/en/articles/741)

MQL5 Standard Library makes your life as a developer easier. Nevertheless, it does not implement all the needs of all developers in the world, so if you feel that you need some more custom stuff you can take a step further and extend. This article walks you through integrating MetaQuotes' Zig-Zag technical indicator into the Standard Library. We get inspired by MetaQuotes' design philosophy to achieve our goal.

![Building an Automatic News Trader](https://c.mql5.com/2/0/cover.png)[Building an Automatic News Trader](https://www.mql5.com/en/articles/719)

This is the continuation of Another MQL5 OOP class article which showed you how to build a simple OO EA from scratch and gave you some tips on object-oriented programming. Today I am showing you the technical basics needed to develop an EA able to trade the news. My goal is to keep on giving you ideas about OOP and also cover a new topic in this series of articles, working with the file system.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fdmezzplrldedktbaeotagxjaeqntfih&ssn=1769179655173701017&ssn_dr=0&ssn_sr=0&fv_date=1769179655&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F691&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20Signal%20Generator%20Based%20on%20a%20Custom%20Indicator%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917965571233019&fz_uniq=5068656687801367559&sv=2552)

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