---
title: MQL5 Wizard: How to Create a Module of Trading Signals
url: https://www.mql5.com/en/articles/226
categories: Trading Systems, Expert Advisors
relevance_score: 4
scraped_at: 2026-01-23T17:47:47.507145
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/226&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068659462350240785)

MetaTrader 5 / Trading systems


### Introduction

[MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") provides a powerful tool for quick checking of trading ideas. This is the generator of trading strategies of the [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate"). The use of the MQL5 Wizard for automatic creation of Expert Advisor codes is described in the article " [MQL5 Wizard: Creating Expert Advisors without Programming](https://www.mql5.com/en/articles/171)". Openness of the code generation system allows you to add your own classes of trading signals, money management systems and trailing modules to the [standard](https://www.mql5.com/en/docs/standardlibrary/expertclasses) ones.

This article describes the principles of writing [modules of trading signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal) to use them when creating Expert Advisors with the MQL5 Wizard.

The Expert Advisor created with [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate"), is based on four pillars - four [base classes](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses):

![Figure 1. The structure of the CExpert base class](https://c.mql5.com/2/2/MQL5_CExpert_structure.png)

Figure 1. The structure of the [CExpert base class](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert)

The [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) class (or its subclass) is the main "engine" of **a trading robot**. An instance of CExpert contains one copy of each class: [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal), [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) and [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) (or their subclasses):

1. [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) is the basis of the trading signals generator. An instance of the CExpertSignal derived class, included in [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert), provides an Expert Advisor with information about the possibility of entering the market, levels of entry and placing of protective orders, based on built-in algorithms. The final decision on **execution of trading operations** is made by the EA.
2. [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) is the basis of the money and risk management systems. An instance of CExpertMoney derived class calculates volumes for opening positions and placing pending orders. The final decision on the volume is made by the EA.

3. [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) \- is the basis of the module of open positions support. An instance of the CExpertTrailing derived class informs an EA about the necessity to modify protective orders of a position. The final decision on the order modification is made by the EA.

In addition, the members of the [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) class are instances of the following classes:

- CExpertTrade (for trading)
- [CIndicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2) (for controlling indicators and timeseries involved in the work of the EA).

- [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) (for getting information about the instrument)
- [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) (for obtaining information on the state of the trading account)
- [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo)(for obtaining information about positions)
- [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) (for obtaining information about pending orders)

Hereinafter, under "expert" we mean an instance of [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) or its subclass.

More details of CExpert and work with it will be described in a separate article.

### 1\. Base Class CExpertSignal

[CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) is the basis of the trading signals generator. For communication with the "outside world", CExpertSignal has a set of public virtual method:

|     |     |
| --- | --- |
| **Initialization** | **Description** |
| virtual [Init](https://www.mql5.com/en/articles/226#Init) | Initialization of the class instance provides synchronization of the module data with the data of the EA |
| virtual [ValidationSettings](https://www.mql5.com/en/articles/226#ValidationSettings) | Validation of set parameters |
| virtual [InitIndicators](https://www.mql5.com/en/articles/226#InitIndicators) | Creating and initializing all indicators and timeseries required for operation of the trading signals generator |
| **Signals of position opening/reversal/closing** |  |
| virtual [CheckOpenLong](https://www.mql5.com/en/articles/226#CheckOpenLong) | Generating the signal of long position opening, defining the levels of entry and placing of protective orders |
| virtual [CheckOpenShort](https://www.mql5.com/en/articles/226#CheckOpenShort) | Generating the signal of a short position opening, defining the levels of entry and placing of protective orders |
| virtual [CheckCloseLong](https://www.mql5.com/en/articles/226#CheckCloseLong) | Generating the signal of long position closing, defining the exit level |
| virtual [CheckCloseShort](https://www.mql5.com/en/articles/226#CheckCloseShort) | Generating the signal of short position closing, defining the exit level |
| virtual [CheckReverseLong](https://www.mql5.com/en/articles/226#CheckReverseLong) | Generating the signal of long position reversal, defining the levels of reversal and placing of protective orders |
| virtual [CheckReverseShort](https://www.mql5.com/en/articles/226#CheckReverseShort) | Generating the signal of short position reversal, defining the levels of reversal and placing of protective orders |
| **Managing pending orders** |  |
| virtual [CheckTrailingOrderLong](https://www.mql5.com/en/articles/226#CheckTrailingOrderLong) | Generating the signal of modification of a pending Buy order, defining the new order price |
| virtual [CheckTrailingOrderShort](https://www.mql5.com/en/articles/226#CheckTrailingOrderShort) | Generating the signal of modification of a pending Sell order, defining the new order price |

### Description of Methods

**1.1. Initialization methods:**

**1.1.1 Init**

The [Init()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertbase/cexpertbaseinit) method is called **automatically** right after a class instance is added to the expert. Method overriding is not required.

```
virtual bool Init(CSymbolInfo* symbol, ENUM_TIMEFRAMES period, double adjusted_point);
```

**1.1.2** **ValidationSettings**

The [ValidationSettings()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalvalidationsettings) method is called right from the expert after all the parameters are set. You must override the method if there are any setup parameters.

```
virtual bool ValidationSettings();
```

The overridden method must return true, if all options are valid (usable). If at least one of the parameters is incorrect, it must return false (further work is impossible).

Base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) has no adjustable parameters, therefore, the base class method always returns true without performing any checks.

**1.1.3** **InitIndicators**

The [InitIndicators ()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalinitindicators) method implements the creation and initialization of all necessary indicators and timeseries. It is called from the expert after all the parameters are set and their correctness is successful verified. The method should be overridden if the trading signal generator uses at least one indicator or timeseries.

```
virtual bool InitIndicators(CIndicators* indicators);
```

Indicators and/or timeseries should be used through the appropriate classes of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary). Pointers of all indicators and/or timeseries should be added to the collection of indicators of an expert (a pointer to which is passed as a parameter).

The overridden method must return true, if all manipulations with the indicators and/or timeseries were successful (they are suitable for use). If at least one operation with the indicators and/or timeseries failed, the method must return false (further work is impossible).

Base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) does not use indicators or timeseries, therefore, the base class method always returns true, without performing any action.

**1.2. Methods of checking the signal of position opening:**

**1.2.1** **CheckOpenLong**

The [CheckOpenLong()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalcheckopenlong) method generates a signal of opening of a long position, defining the entry level and levels of protective orders placing. It is called by an expert to determine whether it is necessary to open a long position. The method must be overridden, if it is expected that a signal of a long position opening will be generated.

```
virtual bool CheckOpenLong(double& price, double& sl, double& tp, datetime& expiration);
```

The method should implement the algorithm of checking the condition of a long position opening. If the condition is met, the variables price, sl, tp, and expiration (references to which are passed as parameters) must be assigned appropriate values and the method should return true. If the condition is not fulfilled, the method must return false.

Base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) has no built-in algorithm for generating a signal of a long position opening, so the base class method always returns false.

**1.2.2** **CheckOpenShort**

The [CheckOpenShort()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalcheckopenshort) method generates a signal of opening of a short position, defining the entry level and levels of protective orders placing. It is called by an expert to determine whether it is necessary to open a short position. The method must be overridden, if it is expected that a signal of a short position opening will be generated.

```
virtual bool CheckOpenShort(double& price, double& sl, double& tp, datetime& expiration);
```

The method must implement the algorithm for checking the condition to open a short position. If the condition is satisfied, the variables price, sl, tp, and expiration (references to which are passed as parameters) must be assigned appropriate values and the method should return true. If the condition is not fulfilled, the method must return false.

Base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) has no built-in algorithm for generating a signal of a short position opening, so the base class method always returns false.

**1.3. Methods of checking the signal of position closing:**

**1.3.1** **CheckCloseLong**

The [CheckCloseLong()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalcheckcloselong) method generates a signal of closing of a long position, defining the exit level. It is called by an expert to determine whether it is necessary to close a long position. The method must be overridden, if it is expected that a signal of a long position closing will be generated.

```
virtual bool CheckCloseLong(double& price);
```

The method must implement the algorithm for checking the condition to close the long position. If the condition is satisfied, the variable price (the reference to which is passed as a parameter) must be assigned the appropriate value and the method should return true. If the condition is not fulfilled, the method must return false.

Base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) has no built-in algorithm for generating a signal of a long position closing, so the base class method always returns false.

**1.3.2** **CheckCloseShort**

The [CheckCloseShort()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalcheckcloseshort) method generates a signal of closing of a short position, defining the exit level. It is called by an expert to determine whether it is necessary to close a short position. The method must be overridden, if it is expected that a signal of a short position closing will be generated.

```
virtual bool CheckCloseShort(double& price);
```

The method must implement the algorithm for checking the condition to close a short position. If the condition is satisfied, the variable price (the reference to which is passed as a parameter) must be assigned the appropriate value and the method should return true. If the condition is not fulfilled, the method must return false.

Base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) has no built-in algorithm for generating a signal of a short position closing, so the base class method always returns false.

**1.4. Methods of checking the signal of position reversal:**

**1.4.1** **CheckReverseLong**

The CheckReverseLong method generates a signal of reversal of a long position, defining the reversal level and levels of protective orders placing. It is called by an expert to determine whether it is necessary to reverse a long position. The method must be overridden, if it is expected that a signal of a long position reversal will be generated.

```
virtual bool CheckReverseLong(double& price, double& sl, double& tp, datetime& expiration);
```

The method must implement the algorithm for checking the condition of long position reversal. If the condition is satisfied, the variables price, sl, tp, and expiration (references to which are passed as parameters) must be assigned appropriate values and the method should return true. If the condition is not fulfilled, the method must return false.

In the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) base class, the following algorithm for generating a long position reversal signal is implemented:

1. Checking for a signal to close a long position.
2. Checking for a signal to open a short position.
3. If both signals are active (the conditions are met) and the close and open prices match, the variables price, sl, tp, and expiration (references to which are passed as parameters) are assigned the appropriate values and the method returns true.

If the condition is not fulfilled, the method returns false.

**1.4.2** **CheckReverseShort**

The CheckReverseShort method generates a signal of reversal of a short position, defining the reversal level and levels of protective orders placing. It is called by an expert to determine whether it is necessary to reverse a short position. The method must be overridden, if it is expected that a signal of a long position reversal will be generated according to the algorithm that differs from the one implemented in the base class.

```
virtual bool CheckReverseShort(double& price, double& sl, double& tp, datetime& expiration);
```

The method must implement the algorithm for checking the condition of short position reversal. If the condition is satisfied, the variables price, sl, tp, and expiration (references to which are passed as parameters) must be assigned appropriate values and the method should return true. If the condition is not fulfilled, the method must return false.

In the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) base class, the following algorithm for generating a short position reversal signal is implemented:

1. Checking for a signal to close a short position.
2. Checking for a signal to open a long position.
3. If both signals are active (the conditions are met) and the close and open prices match, the variables price, sl, tp, and expiration (references to which are passed as parameters) are assigned the appropriate values and the method returns true.

If the condition is not fulfilled, the method returns false.

**1.5. Methods of checking the signal of pending order modification:**

**1.5.1** **CheckTrailingOrderLong**

The [CheckTrailingOrderLong()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalchecktrailingorderlong) method generates the signal of modification of a pending Buy order, defining a new order price. It is called by an expert to determine whether it is necessary to modify a pending Buy order. The method must be overridden, if it is expected that a signal of modification of a pending Buy order will be generated.

```
virtual bool CheckTrailingOrderLong(COrderInfo* order, double& price)
```

The method must implement the algorithm for checking the condition of modification of a pending Buy order. If the condition is satisfied, the variable price (the reference to which is passed as a parameter) must be assigned the appropriate value and the method should return true. If the condition is not fulfilled, the method must return false.

Base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) has no built-in algorithm for generating a signal of modification of a pending Buy order, so the base class method always returns false.

**1.5.2** **CheckTrailingOrderShort**

The [CheckTrailingOrderShort()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalchecktrailingordershort) method generates the signal of modification of a pending Sell order, defining a new order price. It is called by an expert to determine whether it is necessary to modify a pending Sell order. The method must be overridden, if it is expected that a signal of modification of a pending Sell order will be generated.

```
virtual bool CheckTrailingOrderShort(COrderInfo* order, double& price)
```

The method must implement the algorithm for checking the condition of modification of a pending Sell order. If the condition is satisfied, the variable price (the reference to which is passed as a parameter) must be assigned the appropriate value and the method should return true. If the condition is not fulfilled, the method must return false.

Base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) has no built-in algorithm for generating a signal of modification of a pending Sell order, so the base class method always returns false.

### 2\. Develop Your Own Generator of Trading Signals

Now, after we have reviewed the structure of the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) base class, you can start creating your own trading signals generator.

As mentioned above, the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) class is a set of public virtual "ropes" - methods, using which the expert may know the opinion of the trading signals generator about entering the market in one direction or another.

Therefore, our primary goal is to create our own class of trading signals generator, deriving it from the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) class and overriding the appropriate virtual methods, implementing the required algorithms.

Our second problem (which is not less important) - to make our class "visible" to [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate"). But, first things first.

**2.1. Creating the class of the trading signals generator**

Let's begin.

First, we create (for example, using the same [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard")) an include file with the mqh extension.

In the File menu select "Create" (or press Ctrl+N key combination) and indicate the creation of an included file:

![Figure 2. Create an include file using MQL5 Wizard.](https://c.mql5.com/2/2/MQL5_Wizard_Include__1.png)

Figure 2. Create an include file using MQL5 Wizard

It should be noted that in order for the file to be then "detected" by [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") as a signal generator, it should be created in the folder Include\\Expert\\Signal\\.

In order not to trash in [Standard Library](https://www.mql5.com/en/docs/standardlibrary/expertclasses), create our own folder Include\\Expert\\Signal\\MySignals, in which we create file SampleSignal.mqh, specifying these parameters in MQL5 Wizard:

![Figure 3. Setting the location of the include file](https://c.mql5.com/2/2/MQL5_Wizard_Name.png)

Figure 3. Setting the location of the include file

As a result of [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard") operation we have the following pattern:

```
//+------------------------------------------------------------------+
//|                                                 SampleSignal.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
// #define MacrosHello   "Hello, world!"
// #define MacrosYear    2010
//+------------------------------------------------------------------+
//| DLL imports                                                      |
//+------------------------------------------------------------------+
// #import "user32.dll"
//   int      SendMessageA(int hWnd,int Msg,int wParam,int lParam);
// #import "my_expert.dll"
//   int      ExpertRecalculate(int wParam,int lParam);
// #import
//+------------------------------------------------------------------+
//| EX5 imports                                                      |
//+------------------------------------------------------------------+
// #import "stdlib.ex5"
//   string ErrorDescription(int error_code);
// #import
//+------------------------------------------------------------------+
```

The following is only "manual" work. Remove the unnecessary parts and add what is required (include file ExpertSignal.mqh of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) and a class description which is now empty).

```
//+------------------------------------------------------------------+
//|                                                 SampleSignal.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| include files                                                    |
//+------------------------------------------------------------------+
#include <Expert\ExpertSignal.mqh>
//+------------------------------------------------------------------+
//| The CSampleSignal class.                                         |
//| Purpose: Class of trading signal generator.                      |
//|          It is derived from the CExpertSignal class.             |
//+------------------------------------------------------------------+
class CSampleSignal : public CExpertSignal
  {
  };
//+------------------------------------------------------------------+
```

Now, it is necessary to choose the algorithms.

As a basis for our trading signals generator, we take the widespread model "price crosses the moving average". But we make one more assumption: "After crossing the moving average, the price moves back, and only then goes in the right direction." Reflect this in our file.

Generally, when you are writing something, do not skimp on the comments. After some time, reading a carefully commented code will be so comfortable.

```
//+------------------------------------------------------------------+
//|                                                 SampleSignal.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| include files                                                    |
//+------------------------------------------------------------------+
#include <Expert\ExpertSignal.mqh>
//+------------------------------------------------------------------+
//| Class CSampleSignal.                                             |
//| Purpose: Class of trading signal generator when price            |
//|          crosses moving average,                                 |
//|          entering on the subsequent back movement.               |
//|          It is derived from the CExpertSignal class.             |
//+------------------------------------------------------------------+
class CSampleSignal : public CExpertSignal
  {
  };
//+------------------------------------------------------------------+
```

Now let's define what data is needed for making decisions about the generation of trading signals. In our case, this is the open price and the close price of the previous bar, and the value of the moving average on the same previous bar.

To get access to these data, we use the standard library classes [CiOpen](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/timeseries/ciopen), [CiClose](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/timeseries/ciclose) and [CiMA](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/trendindicators/cima). We'll discuss indicators and timeseries later.

In the meantime, let's define a list of settings for our generator. First, we need to set up the moving average. These parameters include the period, the shift along the time axis, the averaging method and the object of averaging. Secondly, we need to set up the entry level and the levels of placing of protective orders, and the lifetime of a pending order, because we are going to work with pending orders.

All settings of the generator will be stored in protected data members of the [class](https://www.mql5.com/en/docs/basis/types/classes#class). Access to the settings will be implemented through appropriate public methods.

Let's include these changes in our file:

```
//+------------------------------------------------------------------+
//|                                                 SampleSignal.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| include files                                                    |
//+------------------------------------------------------------------+
#include <Expert\ExpertSignal.mqh>
//+------------------------------------------------------------------+
//| The CSampleSignal class.                                         |
//| Purpose: Class of trading signal generator when price            |
//|             crosses moving average,                              |
//|             entering on the subsequent back movement.            |
//|             It is derived from the CExpertSignal class.          |
//+------------------------------------------------------------------+
class CSampleSignal : public CExpertSignal
  {
protected:
   //--- Setup parameters
   int                m_period_ma;       // averaging period of the MA
   int                m_shift_ma;        // shift of the MA along the time axis
   ENUM_MA_METHOD     m_method_ma;       // averaging method of the MA
   ENUM_APPLIED_PRICE m_applied_ma;      // averaging object of the MA
   double             m_limit;           // level to place a pending order relative to the MA
   double             m_stop_loss;       // level to place a stop loss order relative to the open price
   double             m_take_profit;     // level to place a take profit order relative to the open price
   int                m_expiration;      // lifetime of a pending order in bars

public:
   //--- Methods to set the parameters
   void               PeriodMA(int value)                 { m_period_ma=value;   }
   void               ShiftMA(int value)                  { m_shift_ma=value;    }
   void               MethodMA(ENUM_MA_METHOD value)      { m_method_ma=value;   }
   void               AppliedMA(ENUM_APPLIED_PRICE value) { m_applied_ma=value;  }
   void               Limit(double value)                 { m_limit=value;       }
   void               StopLoss(double value)              { m_stop_loss=value;   }
   void               TakeProfit(double value)            { m_take_profit=value; }
   void               Expiration(int value)               { m_expiration=value;  }
  };
//+------------------------------------------------------------------+
```

Since we are using protected data members, we need to add [a class constructor](https://www.mql5.com/en/docs/basis/types/classes#constructor), in which we will initialize these data by default values.

To check the parameters, let's override the virtual method [ValidationSettings](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalvalidationsettings) according to the description of the base class.

Description of the class:

```
class CSampleSignal : public CExpertSignal
  {
protected:
   //--- Setup parameters
   int                m_period_ma;       // averaging period of the MA
   int                m_shift_ma;        // shift of the MA along the time axis
   ENUM_MA_METHOD     m_method_ma;       // averaging method of the MA
   ENUM_APPLIED_PRICE m_applied_ma;      // averaging object of the MA
   double             m_limit;            // level to place a pending order relative to the MA
   double             m_stop_loss;        // level to place a stop loss order relative to the open price
   double             m_take_profit;      // level to place a take profit order relative to the open price
   int                m_expiration;       // lifetime of a pending order in bars

public:
                      CSampleSignal();
   //--- Methods to set the parameters
   void               PeriodMA(int value)                 { m_period_ma=value;   }
   void               ShiftMA(int value)                  { m_shift_ma=value;    }
   void               MethodMA(ENUM_MA_METHOD value)      { m_method_ma=value;   }
   void               AppliedMA(ENUM_APPLIED_PRICE value) { m_applied_ma=value;  }
   void               Limit(double value)                 { m_limit=value;       }
   void               StopLoss(double value)              { m_stop_loss=value;   }
   void               TakeProfit(double value)            { m_take_profit=value; }
   void               Expiration(int value)               { m_expiration=value;  }
   //--- Methods to validate the parameters
   virtual bool       ValidationSettings();
  };
```

Implementation of the ValidationSettings() method:

```

//+------------------------------------------------------------------+
//| Validation of the setup parameters.                              |
//| INPUT:  No.                                                      |
//| OUTPUT: true if the settings are correct, otherwise false.       |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::ValidationSettings()
  {
//--- Validation of parameters
   if(m_period_ma<=0)
     {
      printf(__FUNCTION__+": the MA period must be greater than zero");
      return(false);
     }
//--- ok
   return(true);
  }
//+------------------------------------------------------------------+
```

Now, when we've finished the bulk of the preparatory work, we'll talk more about [indicators and timeseries](https://www.mql5.com/en/docs/series).

Indicators and timeseries are the main source of information for decision-making (you can certainly use the coin toss, or phases of the moon, but they are quite hard to formalize).

As we have already defined above, to make decisions, we need the following information: the open price of the previous bar, the close price of the previous bar, and the value of the moving average on the same previous bar.

To gain access to these data, we will use the following classes of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary):

- [CiOpen](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/timeseries/ciopen) \- to access the open price of the previous bar,
- [CiClose](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/timeseries/ciclose) \- to access the close price of the previous bar,

- [CiMA](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/trendindicators/cima)   \- to access the value of the moving average on the previous bar.

You may ask: "Why use the indicator or timeseries, " wrapped " in a class, in order to get a single number?"

There is a hidden meaning, which we are going to reveal now.

How to use the data of an indicator or timeseries?

First, we need to create an indicator.

Second, we need to copy the necessary amount of data into an intermediate buffer.

Third, we need to check whether copying is complete.

Only after these steps, you can use the data.

Using the classes of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary), you avoid the necessity of creating an indicator, of caring about the availability of intermediate buffers and about data loading or release of a handle. The object of an appropriate class will do that for you. All the required indicators will be generated by our signal generator during the initialization stage, and all indicators will be provided with the necessary temporary buffer. And besides, once we add an indicator or timeseries object in the collection (the object of a special class), you can stop caring about the relevance of the data (the data will be updated automatically by the expert).

We'll place the objects of these classes in the protected data members. For each object, we create a method of initialization and data access method.

Let's override the virtual method [InitIndicators](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalinitindicators) (according to the description of the base class).

Description of the class:

```
class CSampleSignal : public CExpertSignal
  {
protected:
   CiMA               m_MA;              // object to access the values om the moving average
   CiOpen             m_open;            // object to access the bar open prices
   CiClose            m_close;           // object to access the bar close prices
   //--- Setup parameters
   int                m_period_ma;       // averaging period of the MA
   int                m_shift_ma;        // shift of the MA along the time axis
   ENUM_MA_METHOD     m_method_ma;       // averaging method of the MA
   ENUM_APPLIED_PRICE m_applied_ma;      // averaging object of the MA
   double             m_limit;            // level to place a pending order relative to the MA
   double             m_stop_loss;        // level to place a stop loss order relative to the open price
   double             m_take_profit;      // level to place a take profit order relative to the open price
   int                m_expiration;      // lifetime of a pending order in bars

public:
                      CSampleSignal();
   //--- Methods to set the parameters
   void               PeriodMA(int value)                 { m_period_ma=value;              }
   void               ShiftMA(int value)                  { m_shift_ma=value;               }
   void               MethodMA(ENUM_MA_METHOD value)      { m_method_ma=value;              }
   void               AppliedMA(ENUM_APPLIED_PRICE value) { m_applied_ma=value;             }
   void               Limit(double value)                 { m_limit=value;                  }
   void               StopLoss(double value)              { m_stop_loss=value;              }
   void               TakeProfit(double value)            { m_take_profit=value;            }
   void               Expiration(int value)               { m_expiration=value;             }
   //--- Method to validate the parameters
   virtual bool       ValidationSettings();
   //--- Method to validate the parameters
   virtual bool       InitIndicators(CIndicators* indicators);

protected:
   //--- Object initialization method
   bool               InitMA(CIndicators* indicators);
   bool               InitOpen(CIndicators* indicators);
   bool               InitClose(CIndicators* indicators);
   //--- Methods to access object data
   double             MA(int index)                       { return(m_MA.Main(index));       }
   double             Open(int index)                     { return(m_open.GetData(index));  }
   double             Close(int index)                    { return(m_close.GetData(index)); }
  };
```

Implementation of methods InitIndicators, InitMA, InitOpen, InitClose:

```
//+------------------------------------------------------------------+
//| Initialization of indicators and timeseries.                     |
//| INPUT:  indicators - pointer to the object - collection of       |
//|                      indicators and timeseries.                  |
//| OUTPUT: true in case of success, otherwise false.                |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::InitIndicators(CIndicators* indicators)
  {
//--- Validation of the pointer
   if(indicators==NULL)       return(false);
//--- Initialization of the moving average
   if(!InitMA(indicators))    return(false);
//--- Initialization of the timeseries of open prices
   if(!InitOpen(indicators))  return(false);
//--- Initialization of the timeseries of close prices
   if(!InitClose(indicators)) return(false);
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialization of the moving average                             |
//| INPUT:  indicators - pointer to the object - collection of       |
//|                      indicators and timeseries.                  |
//| OUTPUT: true in case of success, otherwise false.                |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::InitMA(CIndicators* indicators)
  {
//--- Initialization of the MA object
   if(!m_MA.Create(m_symbol.Name(),m_period,m_period_ma,m_shift_ma,m_method_ma,m_applied_ma))
     {
      printf(__FUNCTION__+": object initialization error");
      return(false);
     }
   m_MA.BufferResize(3+m_shift_ma);
//--- Adding an object to the collection
   if(!indicators.Add(GetPointer(m_MA)))
     {
      printf(__FUNCTION__+": object adding error");
      return(false);
     }
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialization of the timeseries of open prices.                 |
//| INPUT:  indicators - pointer to the object - collection of       |
//|                      indicators and timeseries.                  |
//| OUTPUT: true in case of success, otherwise false.                |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::InitOpen(CIndicators* indicators)
  {
//--- Initialization of the timeseries object
   if(!m_open.Create(m_symbol.Name(),m_period))
     {
      printf(__FUNCTION__+": object initialization error");
      return(false);
     }
//--- Adding an object to the collection
   if(!indicators.Add(GetPointer(m_open)))
     {
      printf(__FUNCTION__+": object adding error");
      return(false);
     }
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialization of the timeseries of close prices.                |
//| INPUT:  indicators - pointer to the object - collection of       |
//|                      indicators and timeseries.                  |
//| OUTPUT: true in case of success, otherwise false.                |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::InitClose(CIndicators* indicators)
  {
//--- Initialization of the timeseries object
   if(!m_close.Create(m_symbol.Name(),m_period))
     {
      printf(__FUNCTION__+": object initialization error");
      return(false);
     }
//--- Adding an object to the collection
   if(!indicators.Add(GetPointer(m_close)))
     {
      printf(__FUNCTION__+": object adding error");
      return(false);
     }
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
```

All the preparatory works are completed. As you can see, our class has grown significantly.

But now we are ready to generate trading signals.

![Figure 4. Trading signals for the price crossing the moving average](https://c.mql5.com/2/2/sample2_en.png)

Figure 4. Trading signals for the price crossing the moving average

Let's consider our algorithms again in more detail.

**1\. The signal to buy appears when the following conditions have been fulfilled on the previous bar:**

- the bar open price is less than the value of the moving average,

- the bar close price is greater than the value of the moving average,
- the moving average is increasing.

In this case, we offer to place a pending Buy order with the parameters defined by the settings. For this purpose, we override the virtual method [CheckOpenLong](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalcheckopenlong) and fill it with the corresponding functional.

**2\. The signal to sell appears when the following conditions have been fulfilled on the previous bar:**

- the bar open price is greater than the value of the moving average,

- the bar close price is less than the value of the moving average,
- the moving average is decreasing.


In this case, we offer to place a pending Sell order with the parameters defined by the settings. For this purpose, we override the virtual method [CheckOpenShort](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalcheckopenshort) and fill it with the corresponding functional.

**3\. We will not generate signals to close positions. Let the positions be closed by Stop Loss/Take Profit ордерам.**

Accordingly, we will not override virtual methods [CheckCloseLong](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalcheckcloselong) and [CheckCloseShort](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalcheckcloseshort).

**4\. We will propose the modification of a pending order along the moving average at the "distance" specified by the settings.**

For this purpose, we override the virtual methods [CheckTrailingOrderLong](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalchecktrailingorderlong) and [CheckTrailingOrderShort](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalchecktrailingordershort), filling them with corresponding functional.

Description of the class:

```
class CSampleSignal : public CExpertSignal
  {
protected:
   CiMA               m_MA;              // object to access the values of the moving average
   CiOpen             m_open;            // object to access the bar open prices
   CiClose            m_close;           // object to access the bar close prices
   //--- Setup parameters
   int                m_period_ma;       // averaging period of the MA
   int                m_shift_ma;        // shift of the MA along the time axis
   ENUM_MA_METHOD     m_method_ma;       // averaging method of the MA
   ENUM_APPLIED_PRICE m_applied_ma;      // averaging object of the MA
   double             m_limit;            // level to place a pending order relative to the MA
   double             m_stop_loss;        // level to place a stop loss order relative to the open price
   double             m_take_profit;      // level to place a take profit order relative to the open price
   int                m_expiration;       // lifetime of a pending order in bars

public:
                      CSampleSignal();
   //--- Methods to set the parameters

   void               PeriodMA(int value)                 { m_period_ma=value;              }
   void               ShiftMA(int value)                  { m_shift_ma=value;               }
   void               MethodMA(ENUM_MA_METHOD value)      { m_method_ma=value;              }
   void               AppliedMA(ENUM_APPLIED_PRICE value) { m_applied_ma=value;             }
   void               Limit(double value)                 { m_limit=value;                  }
   void               StopLoss(double value)              { m_stop_loss=value;              }
   void               TakeProfit(double value)            { m_take_profit=value;            }
   void               Expiration(int value)               { m_expiration=value;             }
   //--- Method to validate the parameters
   virtual bool       ValidationSettings();
   //--- Method to validate the parameters
   virtual bool       InitIndicators(CIndicators* indicators);
   //--- Methods to generate signals to enter the market
   virtual bool      CheckOpenLong(double& price,double& sl,double& tp,datetime& expiration);
   virtual bool      CheckOpenShort(double& price,double& sl,double& tp,datetime& expiration);
   //--- Methods to generate signals of pending order modification
   virtual bool      CheckTrailingOrderLong(COrderInfo* order,double& price);
   virtual bool      CheckTrailingOrderShort(COrderInfo* order,double& price);

protected:
   //--- Object initialization method
   bool               InitMA(CIndicators* indicators);
   bool               InitOpen(CIndicators* indicators);
   bool               InitClose(CIndicators* indicators);
   //--- Methods to access object data
   double             MA(int index)                       { return(m_MA.Main(index));       }
   double             Open(int index)                     { return(m_open.GetData(index));  }
   double             Close(int index)                    { return(m_close.GetData(index)); }
  };
```

Implementation of methods CheckOpenLong, CheckOpenShort, CheckTrailingOrderLong, CheckTrailingOrderShort:

```
//+------------------------------------------------------------------+
//| Check whether a Buy condition is fulfilled                       |
//| INPUT:  price      - variable for open price                     |
//|         sl         - variable for stop loss price,               |
//|         tp         - variable for take profit price              |
//|         expiration - variable for expiration time.               |
//| OUTPUT: true if the condition is fulfilled, otherwise false.     |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::CheckOpenLong(double& price,double& sl,double& tp,datetime& expiration)
  {
//--- Preparing the data
   double spread=m_symbol.Ask()-m_symbol.Bid();
   double ma    =MA(1);
   double unit  =PriceLevelUnit();
//--- Checking the condition
   if(Open(1)<ma && Close(1)>ma && ma>MA(2))
     {
      price=m_symbol.NormalizePrice(ma-m_limit*unit+spread);
      sl   =m_symbol.NormalizePrice(price-m_stop_loss*unit);
      tp   =m_symbol.NormalizePrice(price+m_take_profit*unit);
      expiration+=m_expiration*PeriodSeconds(m_period);
      //--- Condition is fulfilled
      return(true);
     }
//--- Condition is not fulfilled
   return(false);
  }
//+------------------------------------------------------------------+
//| Check whether a Sell condition is fulfilled.                     |
//| INPUT:  price      - variable for open price,                    |
//|         sl         - variable for stop loss,                     |
//|         tp         - variable for take profit                    |
//|         expiration - variable for expiration time.               |
//| OUTPUT: true if the condition is fulfilled, otherwise false.     |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::CheckOpenShort(double& price,double& sl,double& tp,datetime& expiration)
  {
//--- Preparing the data
   double ma  =MA(1);
   double unit=PriceLevelUnit();
//--- Checking the condition
   if(Open(1)>ma && Close(1)<ma && ma<MA(2))
     {
      price=m_symbol.NormalizePrice(ma+m_limit*unit);
      sl   =m_symbol.NormalizePrice(price+m_stop_loss*unit);
      tp   =m_symbol.NormalizePrice(price-m_take_profit*unit);
      expiration+=m_expiration*PeriodSeconds(m_period);
      //--- Condition is fulfilled
      return(true);
     }
//--- Condition is not fulfilled
   return(false);
  }
//+------------------------------------------------------------------+
//| Check whether the condition of modification                      |
//|  of a Buy order is fulfilled.                                    |
//| INPUT:  order - pointer at the object-order,                     |
//|         price - a variable for the new open price.               |
//| OUTPUT: true if the condition is fulfilled, otherwise false.     |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::CheckTrailingOrderLong(COrderInfo* order,double& price)
  {
//--- Checking the pointer
   if(order==NULL) return(false);
//--- Preparing the data
   double spread   =m_symbol.Ask()-m_symbol.Bid();
   double ma       =MA(1);
   double unit     =PriceLevelUnit();
   double new_price=m_symbol.NormalizePrice(ma-m_limit*unit+spread);
//--- Checking the condition
   if(order.PriceOpen()==new_price) return(false);
   price=new_price;
//--- Condition is fulfilled
   return(true);
  }
//+------------------------------------------------------------------+
//| Check whether the condition of modification                      |
//| of a Sell order is fulfilled.                                    |
//| INPUT:  order - pointer at the object-order,                     |
//|         price - a variable for the new open price.               |
//| OUTPUT: true if the condition is fulfilled, otherwise false.     |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::CheckTrailingOrderShort(COrderInfo* order,double& price)
  {
//--- Checking the pointer
   if(order==NULL) return(false);
//--- Preparing the data
   double ma  =MA(1);
   double unit=PriceLevelUnit();
   double new_price=m_symbol.NormalizePrice(ma+m_limit*unit);
//--- Checking the condition
   if(order.PriceOpen()==new_price) return(false);
   price=new_price;
//--- Condition is fulfilled
   return(true);
  }
//+------------------------------------------------------------------+
```

So we've solved the first problem. The above code is a source code of the class of trading signals generator that meets our main task.

**2.2. Preparing a description of the created class of the trading signals for MQL5 Wizard**

We now turn to solving the second problem. Our signal should be "recognized" by the generator of trading strategies MQL5 Wizard.

We've done the first necessary condition: we've placed the file where it will be "found" by the MQL5 Wizard. But this is not enough. The MQL5 Wizard must not only "find" the file, but also "recognize" it. To do this we must add to the original text the **class descriptor** for the [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate").

A class descriptor is a block of comments composed according to certain rules.

Let's consider these rules.

1\. The block of comments should start with the following lines:

```
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
```

2\. The next line is a text descriptor (what we will see in the MQL5 Wizard when choosing the signal) in the format "//\| Title=<Text> \|". If the text is too big for one line, you can add one more line (but not more) after it.

In our case, we have the following:

```
//| Title=Signal on the crossing of a price and the MA               |
//| entering on its back movement                                    |
```

3\. Then comes a line with the class type specified in the format "//\| Type=<Type> \|". The <Type> field must have the Signal value (in addition to signals, the MQL5 Wizard knows other types of classes).

Write:

```
//| Type=Signal                                                      |
```

4\. The following line in the format "//\| Name=<Name> \|" is the short name of the signal (it is used by the MQL5 Wizard for generating the names of the global variables of the expert).

We get the following:

```
//| Name=Sample                                                      |
```

5\. The name of a class is an important element of the description. In the line with the format "//\| Class=<ClassNameа> \|", the <ClassName> parameter must match with the name of our class:

```
//| Class=CSampleSignal                                              |
```

6\. We do not fill in this line, but it must be present (this is a link to the [language reference](https://www.mql5.com/en/docs) section):

```
//| Page=                                                            |
```

7\. Further, there are descriptions of the signal setup parameters.

This is a set of rows (the number of rows is equal to the number of parameters).

The format of each line is "//\| Parameter=<NameOfMethod>,<TypeOfParameter>,<DefaultValue> \|".

Here is our set of parameters:

```
//| Parameter=PeriodMA,int,12                                        |
//| Parameter=ShiftMA,int,0                                          |
//| Parameter=MethodMA,ENUM_MA_METHOD,MODE_EMA                       |
//| Parameter=AppliedMA,ENUM_APPLIED_PRICE,PRICE_CLOSE               |
//| Parameter=Limit,double,0.0                                       |
//| Parameter=StopLoss,double,50.0                                   |
//| Parameter=TakeProfit,double,50.0                                 |
//| Parameter=Expiration,int,10                                      |
```

8\. The block of comment should end with the following lines:

```
//+------------------------------------------------------------------+
// wizard description end
```

Let's add the descriptor to the source code.

```
//+------------------------------------------------------------------+
//|                                                 SampleSignal.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| include files                                                    |
//+------------------------------------------------------------------+
#include <Expert\ExpertSignal.mqh>
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Signal on crossing of the price and the MA                 |
//| entering on the back movement                                    |
//| Type=Signal                                                      |
//| Name=Sample                                                      |
//| Class=CSampleSignal                                              |
//| Page=                                                            |
//| Parameter=PeriodMA,int,12                                        |
//| Parameter=ShiftMA,int,0                                          |
//| Parameter=MethodMA,ENUM_MA_METHOD,MODE_EMA                       |
//| Parameter=AppliedMA,ENUM_APPLIED_PRICE,PRICE_CLOSE               |
//| Parameter=Limit,double,0.0                                       |
//| Parameter=StopLoss,double,50.0                                   |
//| Parameter=TakeProfit,double,50.0                                 |
//| Parameter=Expiration,int,10                                      |
//+------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//| CSampleSignal class.                                             |
//| Purpose: Class of trading signal generator when price            |
//|             crosses moving average,                              |
//|             entering on the subsequent back movement.            |
//|             It is derived from the CExpertSignal class.          |
//+------------------------------------------------------------------+
class CSampleSignal : public CExpertSignal
  {
protected:
   CiMA               m_MA;               // object to access the values of the moving average
   CiOpen             m_open;             // object to access the bar open prices
   CiClose            m_close;            // object to access the bar close prices
   //--- Setup parameters
   int                m_period_ma;        // averaging period of the MA
   int                m_shift_ma;         // shift of the MA along the time axis
   ENUM_MA_METHOD     m_method_ma;        // averaging method of the MA
   ENUM_APPLIED_PRICE m_applied_ma;       // averaging object of the MA
   double             m_limit;            // level to place a pending order relative to the MA
   double             m_stop_loss;        // level to place a stop loss order relative to the open price
   double             m_take_profit;      // level to place a take profit order relative to the open price
   int                m_expiration;       // lifetime of a pending order in bars

public:
                      CSampleSignal();
   //--- Methods to set the parameters
   void               PeriodMA(int value)                 { m_period_ma=value;              }
   void               ShiftMA(int value)                  { m_shift_ma=value;               }
   void               MethodMA(ENUM_MA_METHOD value)      { m_method_ma=value;              }
   void               AppliedMA(ENUM_APPLIED_PRICE value) { m_applied_ma=value;             }
   void               Limit(double value)                 { m_limit=value;                  }
   void               StopLoss(double value)              { m_stop_loss=value;              }
   void               TakeProfit(double value)            { m_take_profit=value;            }
   void               Expiration(int value)               { m_expiration=value;             }
   //---Method to validate the parameters
   virtual bool       ValidationSettings();
   //--- Method to validate the parameters
   virtual bool       InitIndicators(CIndicators* indicators);
   //--- Methods to generate signals to enter the market
   virtual bool      CheckOpenLong(double& price,double& sl,double& tp,datetime& expiration);
   virtual bool      CheckOpenShort(double& price,double& sl,double& tp,datetime& expiration);
   //--- Methods to generate signals of pending order modification
   virtual bool      CheckTrailingOrderLong(COrderInfo* order,double& price);
   virtual bool      CheckTrailingOrderShort(COrderInfo* order,double& price);

protected:
   //--- Object initialization method
   bool               InitMA(CIndicators* indicators);
   bool               InitOpen(CIndicators* indicators);
   bool               InitClose(CIndicators* indicators);
   //--- Methods to access object data
   double             MA(int index)                       { return(m_MA.Main(index));       }
   double             Open(int index)                     { return(m_open.GetData(index));  }
   double             Close(int index)                    { return(m_close.GetData(index)); }
  };
//+------------------------------------------------------------------+
//| CSampleSignal Constructor.                                       |
//| INPUT:  No.                                                      |
//| OUTPUT: No.                                                      |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
void CSampleSignal::CSampleSignal()
  {
//--- Setting the default values
   m_period_ma  =12;
   m_shift_ma   =0;
   m_method_ma  =MODE_EMA;
   m_applied_ma =PRICE_CLOSE;
   m_limit      =0.0;
   m_stop_loss  =50.0;
   m_take_profit=50.0;
   m_expiration =10;
  }
//+------------------------------------------------------------------+
//| Validation of parameters.                                        |
//| INPUT:  No.                                                      |
//| OUTPUT: true if the settings are correct, otherwise false.       |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::ValidationSettings()
  {
//--- Validation of parameters
   if(m_period_ma<=0)
     {
      printf(__FUNCTION__+": the MA period must be greater than zero");
      return(false);
     }
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialization of indicators and timeseries.                     |
//| INPUT:  indicators - pointer to the object - collection of       |
//|                      indicators and timeseries.                  |
//| OUTPUT: true in case of success, otherwise false.                |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::InitIndicators(CIndicators* indicators)
  {
//--- Validation of the pointer
   if(indicators==NULL)       return(false);
//--- Initialization of the moving average
   if(!InitMA(indicators))    return(false);
//--- Initialization of the timeseries of open prices
   if(!InitOpen(indicators))  return(false);
//--- Initialization of the timeseries of close prices
   if(!InitClose(indicators)) return(false);
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialization of the moving average                             |
//| INPUT:  indicators - pointer to the object - collection of       |
//|                      indicators and timeseries.                  |
//| OUTPUT: true in case of success, otherwise false.                |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::InitMA(CIndicators* indicators)
  {
//--- Initialization of the MA object
   if(!m_MA.Create(m_symbol.Name(),m_period,m_period_ma,m_shift_ma,m_method_ma,m_applied_ma))
     {
      printf(__FUNCTION__+": object initialization error");
      return(false);
     }
   m_MA.BufferResize(3+m_shift_ma);
//--- Adding an object to the collection
   if(!indicators.Add(GetPointer(m_MA)))
     {
      printf(__FUNCTION__+": object adding error");
      return(false);
     }
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialization of the timeseries of open prices.                 |
//| INPUT:  indicators - pointer to the object - collection of       |
//|                      indicators and timeseries.                  |
//| OUTPUT: true in case of success, otherwise false.                |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::InitOpen(CIndicators* indicators)
  {
//--- Initialization of the timeseries object
   if(!m_open.Create(m_symbol.Name(),m_period))
     {
      printf(__FUNCTION__+": object initialization error");
      return(false);
     }
//--- Adding an object to the collection
   if(!indicators.Add(GetPointer(m_open)))
     {
      printf(__FUNCTION__+": object adding error");
      return(false);
     }
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialization of the timeseries of close prices.                |
//| INPUT:  indicators - pointer to the object - collection of       |
//|                      indicators and timeseries.                  |
//| OUTPUT: true in case of success, otherwise false.                |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::InitClose(CIndicators* indicators)
  {
//--- Initialization of the timeseries object
   if(!m_close.Create(m_symbol.Name(),m_period))
     {
      printf(__FUNCTION__+": object initialization error");
      return(false);
     }
//--- Adding an object to the collection
   if(!indicators.Add(GetPointer(m_close)))
     {
      printf(__FUNCTION__+": object adding error");
      return(false);
     }
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Check whether a Buy condition is fulfilled                       |
//| INPUT:  price      - variable for open price                     |
//|         sl         - variable for stop loss price,               |
//|         tp         - variable for take profit price              |
//|         expiration - variable for expiration time.               |
//| OUTPUT: true if the condition is fulfilled, otherwise false.     |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::CheckOpenLong(double& price,double& sl,double& tp,datetime& expiration)
  {
//--- Preparing the data
   double spread=m_symbol.Ask()-m_symbol.Bid();
   double ma    =MA(1);
   double unit  =PriceLevelUnit();
//--- Checking the condition
   if(Open(1)<ma && Close(1)>ma && ma>MA(2))
     {
      price=m_symbol.NormalizePrice(ma-m_limit*unit+spread);
      sl   =m_symbol.NormalizePrice(price-m_stop_loss*unit);
      tp   =m_symbol.NormalizePrice(price+m_take_profit*unit);
      expiration+=m_expiration*PeriodSeconds(m_period);
      //--- Condition is fulfilled
      return(true);
     }
//--- Condition is not fulfilled
   return(false);
  }
//+------------------------------------------------------------------+
//| Check whether a Sell condition is fulfilled.                     |
//| INPUT:  price      - variable for open price,                    |
//|         sl         - variable for stop loss,                     |
//|         tp         - variable for take profit                    |
//|         expiration - variable for expiration time.               |
//| OUTPUT: true if the condition is fulfilled, otherwise false.     |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::CheckOpenShort(double& price,double& sl,double& tp,datetime& expiration)
  {
//--- Preparing the data
   double ma  =MA(1);
   double unit=PriceLevelUnit();
//--- Checking the condition
   if(Open(1)>ma && Close(1)<ma && ma<MA(2))
     {
      price=m_symbol.NormalizePrice(ma+m_limit*unit);
      sl   =m_symbol.NormalizePrice(price+m_stop_loss*unit);
      tp   =m_symbol.NormalizePrice(price-m_take_profit*unit);
      expiration+=m_expiration*PeriodSeconds(m_period);
      //--- Condition is fulfilled
      return(true);
     }
//--- Condition is not fulfilled
   return(false);
  }
//+------------------------------------------------------------------+
//| Check whether the condition of modification                      |
//|  of a Buy order is fulfilled.                                    |
//| INPUT:  order - pointer at the object-order,                     |
//|         price - a variable for the new open price.               |
//| OUTPUT: true if the condition is fulfilled, otherwise false.     |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::CheckTrailingOrderLong(COrderInfo* order,double& price)
  {
//--- Checking the pointer
   if(order==NULL) return(false);
//--- Preparing the data
   double spread   =m_symbol.Ask()-m_symbol.Bid();
   double ma       =MA(1);
   double unit     =PriceLevelUnit();
   double new_price=m_symbol.NormalizePrice(ma-m_limit*unit+spread);
//--- Checking the condition
   if(order.PriceOpen()==new_price) return(false);
   price=new_price;
//--- Condition is fulfilled
   return(true);
  }
//+------------------------------------------------------------------+
//| Check whether the condition of modification                      |
//| of a Sell order is fulfilled.                                    |
//| INPUT:  order - pointer at the object-order,                     |
//|         price - a variable for the new open price.               |
//| OUTPUT: true if the condition is fulfilled, otherwise false.     |
//| REMARK: No.                                                      |
//+------------------------------------------------------------------+
bool CSampleSignal::CheckTrailingOrderShort(COrderInfo* order,double& price)
  {
//--- Checking the pointer
   if(order==NULL) return(false);
//--- Preparing the data
   double ma  =MA(1);
   double unit=PriceLevelUnit();
   double new_price=m_symbol.NormalizePrice(ma+m_limit*unit);
//--- Checking the condition
   if(order.PriceOpen()==new_price) return(false);
   price=new_price;
//--- Condition is fulfilled
   return(true);
  }
//+------------------------------------------------------------------+
```

Well, that's all. The signal is ready to use.

For the [generator trading strategies MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") to be able to use our signal, we should restart [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor") (MQL5 Wizard scans the folder Include\\Expert only at boot).

After restarting [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), the created module of trading signals can be used in the MQL5 Wizard:

![Figure 5. The created generator of trading signals in the MQL5 Wizard](https://c.mql5.com/2/2/MQL5_Wizard_Signals_List.png)

Figure 5. The created generator of trading signals in the MQL5 Wizard

The input parameters specified in the section of description of the parameters of the trading signals generator are now available:

![Figure 6. Input parameters of the created generator of trading signals in the MQL5 Wizard](https://c.mql5.com/2/2/MQL5_Wizard_Signals_Parameters_en.png)

Figure 6. Input parameters of the created generator of trading signals in the MQL5 Wizard

The best values of the input parameters of the implemented trading strategy can be found using the [Strategy Tester](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") of the [MetaTrader 5](https://www.metatrader5.com/en/trading-platform "https://www.metatrader5.com/en/trading-platform") terminal.

### Conclusion

[The generator of trading strategies](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") of the MQL5 Wizard greatly simplifies the testing of trading ideas. The code of the generated expert is based on the [classes of trading strategies](https://www.mql5.com/en/docs/standardlibrary/expertclasses) of the Standard Library, which are used for creating certain implementations of trading signal classes, money and risk management classes and position support classes.

The article discusses how to write your own class of trading signals with the implementation of signals on the crossing of the price and the moving average, and how to include it to the generator of trading strategies of the [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate"), as well as describes the structure and format of the description of the generated class for the MQL5 Wizard.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/226](https://www.mql5.com/ru/articles/226)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/226.zip "Download all attachments in the single ZIP archive")

[samplesignal.mqh](https://www.mql5.com/en/articles/download/226/samplesignal.mqh "Download samplesignal.mqh")(15.49 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/2929)**
(77)


![Nikita Gamolin](https://c.mql5.com/avatar/2022/12/638E6C1A-9A2A.png)

**[Nikita Gamolin](https://www.mql5.com/en/users/n0namer)**
\|
6 Jan 2023 at 23:57

How do I finally generate a close signal via CheckCloseLong/Short from the Signal Module? I didn't see how to do it in this article [https://www.mql5.com/en/articles/367](https://www.mql5.com/en/articles/367 "https://www.mql5.com/en/articles/367")

![FINANSE-BOND](https://c.mql5.com/avatar/avatar_na2.png)

**[FINANSE-BOND](https://www.mql5.com/en/users/finanse-bond)**
\|
15 May 2023 at 18:12

I have just downloaded your Signal Code and compiled it. I get these errors. How can I fix them to make it work?

And the same errors come out and your file, which is in the archive and I did not change it in any way, just compiled.

If you look at how these parameters are written in the ExpertBase file there they are with asterisks, if I put asterisks before the name in the code errors will be even more. What is the reason ?

https://photos.app.goo.gl/2rPVRPfBDhb65aZC9

![FINANSE-BOND](https://c.mql5.com/avatar/avatar_na2.png)

**[FINANSE-BOND](https://www.mql5.com/en/users/finanse-bond)**
\|
20 May 2023 at 12:37

Please set the correct Code.

Even after looking through all the answers in this thread, the EA still does not trade on history, only draws a moving line and does not make any other trades.

I am looking for at least some working EA with [buy](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 Documentation: Order Properties") and sell [orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 Documentation: Order Properties") to experiment with settings or add my ideas. I would like to have a simple Template, and just be able to add Conditions1 and Conditions2 to the Code to execute trades based on them. I just used to write in another programme (Easy Lengwich from another platform), here it is very difficult for a simple user to understand how to write his strategies. Even when an error occurs, there is no possibility to right-click on the mouse to find a variant of its correction in the Help manual, so I have to search the Internet and still can't find a solution.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
20 May 2023 at 16:37

**FINANSE-BOND buy and sell [orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 Documentation: Order Properties") to experiment with settings or add my ideas. I would like to have a simple Template, and just be able to add Conditions1 and Conditions2 to the Code to execute trades based on them. I just used to write in another programme (Easy Lengwich from another platform), here it is very difficult for a simple user to understand how to write his strategies. Even when an error occurs, there is no possibility to right-click on the mouse to find a variant of its correction in the Help manual, so I have to search the Internet and still can't find a solution.**

Try this - [https://www.mql5.com/en/code/32107](https://www.mql5.com/en/code/32107 "https://www.mql5.com/en/code/32107")

![farhadmax](https://c.mql5.com/avatar/avatar_na2.png)

**[farhadmax](https://www.mql5.com/en/users/farhadmax)**
\|
7 Jul 2023 at 22:06

### **Important Note:**

In order MetaEditor Wizard to be able to find the signal file (samplesignal.mqh file), The class discriptor should be as following:

// wizard description start

//+------------------------------------------------------------------+

//\| Description of the class                                         \|

//\| Title=Signal on crossing of the price and the MA                 \|

//\| entering on the back movement                                    \|

//\| Type=**SignalAdvanced**                                              \|

//\| Name=Sample                                                      \|

//\| Class=CSampleSignal                                              \|

//\| Page=                                                            \|

//\| Parameter=PeriodMA,int,12                                        \|

//\| Parameter=ShiftMA,int,0                                          \|

//\| Parameter=MethodMA,ENUM\_MA\_METHOD,MODE\_EMA                       \|

//\| Parameter=AppliedMA,ENUM\_APPLIED\_PRICE,PRICE\_CLOSE               \|

//\| Parameter=Limit,double,0.0                                       \|

//\| Parameter=StopLoss,double,50.0                                   \|

//\| Parameter=TakeProfit,double,50.0                                 \|

//\| Parameter=Expiration,int,10                                      \|

//+------------------------------------------------------------------+

// wizard description end

//+------------------------------------------------------------------+

The Type should be **SignalAdvanced** (which is shown by red color), so change **signal** to **SignalAdvanced** in your source code and then MetaEditor Wizard will be able to find the signal file (samplesignal.mqh file).

and finlly [metaquotes](https://www.mql5.com/en/users/metaquotes) should edit this article.

![Create Your Own Expert Advisor in MQL5 Wizard](https://c.mql5.com/2/0/masterMQL5__2.png)[Create Your Own Expert Advisor in MQL5 Wizard](https://www.mql5.com/en/articles/240)

The knowledge of programming languages is no longer a prerequisite for creating trading robots. Earlier lack of programming skills was an impassable obstacle to the implementation of one's own trading strategies, but with the emergence of the MQL5 Wizard, the situation radically changed. Novice traders can stop worrying because of the lack of programming experience - with the new Wizard, which allows you to generate Expert Advisor code, it is not necessary.

![MQL5 Wizard: Creating Expert Advisors without Programming](https://c.mql5.com/2/0/editor_wizard.png)[MQL5 Wizard: Creating Expert Advisors without Programming](https://www.mql5.com/en/articles/171)

Do you want to try out a trading strategy while wasting no time for programming? In MQL5 Wizard you can simply select the type of trading signals, add modules of trailing positions and money management - and your work is done! Create your own implementations of modules or order them via the Jobs service - and combine your new modules with existing ones.

![Creating Multi-Expert Advisors on the basis of Trading Models](https://c.mql5.com/2/0/Multi_Expert_Advisor_MQL5__1.png)[Creating Multi-Expert Advisors on the basis of Trading Models](https://www.mql5.com/en/articles/217)

Using the object-oriented approach in MQL5 greatly simplifies the creation of multi-currency/multi-system /multi-time-frame Expert Advisors. Just imagine, your single EA trades on several dozens of trading strategies, on all of the available instruments, and on all of the possible time frames! In addition, the EA is easily tested in the tester, and for all of the strategies, included in its composition, it has one or several working systems of money management.

![Create your own Market Watch using the Standard Library Classes](https://c.mql5.com/2/0/visual.png)[Create your own Market Watch using the Standard Library Classes](https://www.mql5.com/en/articles/179)

The new MetaTrader 5 client terminal and the MQL5 Language provides new opportunities for presenting visual information to the trader. In this article, we propose a universal and extensible set of classes, which handles all the work of organizing displaying of the arbitrary text information on the chart. The example of Market Watch indicator is presented.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/226&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068659462350240785)

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