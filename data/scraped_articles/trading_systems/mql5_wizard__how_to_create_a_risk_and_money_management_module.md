---
title: MQL5 Wizard: How to Create a Risk and Money Management Module
url: https://www.mql5.com/en/articles/230
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:52:01.803353
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/230&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062776491026065583)

MetaTrader 5 / Trading systems


### Introduction

[MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") provides a powerful tool that allows you to quickly check various trading ideas. This is generation of Expert Advisors using the [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") on the basis of ready trading strategies.

An Expert Advisor created with the [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate"), is based on four pillars - four [base classes](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses):

![Figure 1. The structure of the CExpert base class](https://c.mql5.com/2/2/MQL5_CExpert_structure__1.png)

Figure 1. The structure of the [CExpert base class](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert)

1. The [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) class (or its subclass) is the main "engine" of an Expert Advisor. An instance of [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) contains one copy of each class: [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal), [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) and [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) (or their subclasses):

2. [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) is the basis of the trading signals generator. An instance of the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) derived class, included in [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert), provides an Expert Advisor with information about the possibility of entering the market, levels of entry and placing of protective orders, based on built-in algorithms. The Expert Advisor decides whether to enter the market. More details of the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) class and work with it are described in the article ["MQL5 Wizard: How to create a module of trading signals"](https://www.mql5.com/en/articles/226).
3. The [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) class is the basis of the risk and money management mechanism. An instance of the [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) derived class, included in [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert), provides an Expert Advisor with information about possible volumes for opening positions and placing pending orders, based on built-in algorithms. The Expert Advisor makes a decision about the volume.
4. The [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) class is the basis of the mechanism of open position support. An instance of the [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) derived class, included in [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert), provides an EA with information about the possibility to modify protective orders of the position, based on built-in algorithms. The Expert Advisor makes a decision about the modification of orders. More details of the [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) class and work with it will be described in a separate article.

In addition, the members of the [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) class are instances of the following classes:

- CExpertTrade (for trading)
- [CIndicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2) (for controlling indicators and timeseries involved in the work of the EA).

- [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) (for obtaining information about the instrument)
- [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) (for obtaining information on the state of the trading account)
- [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo)(for obtaining information about positions)
- [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) (for obtaining information about pending orders)

Hereinafter, under "expert" we mean an instance of [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) or its subclass.

More details of [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) and work with it will be described in a separate article.

### 1\. Base Class CExpertMoney

As mentioned above, the CExpertMoney class is the basis of the risk and money management mechanism. For communication with the "outside world", the CExpertMoney class has a set of public virtual method:

|     |     |
| --- | --- |
| **Initialization** | **Description** |
| virtual [Init](https://www.mql5.com/en/articles/230#Init) | Initialization of the class instance provides synchronization of the module data with the data of the EA |
| [Percent](https://www.mql5.com/en/articles/230#Percent) | Setting the value of the parameter "Percent of risk" |
| virtual [ValidationSettings](https://www.mql5.com/en/articles/230#ValidationSettings) | Validating the set parameters |
| virtual [InitIndicators](https://www.mql5.com/en/articles/230#InitIndicators) | Creating and initializing all indicators and timeseries required for operation of the risk and money management mechanism |
| **Methods for checking the necessity to open/turn/close a position** |  |
| virtual [CheckOpenLong](https://www.mql5.com/en/articles/230#CheckOpenLong) | Determining the volume to open a long position |
| virtual [CheckOpenShort](https://www.mql5.com/en/articles/230#CheckOpenShort) | Determining the volume to open a short position |
| virtual [CheckReverse](https://www.mql5.com/en/articles/230#CheckReverse) | Determining the volume for reversing a position |
| virtual [CheckClose](https://www.mql5.com/en/articles/230#CheckClose) | Determining the necessity to close a position |

### Description of Methods

**1.1. Initialization methods**

**1.1.1 Init**

The [Init()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert/cexpertinit) method is called **automatically** right after a class instance is added to the expert. Method overriding is not required.

```
virtual bool Init(CSymbolInfo* symbol, ENUM_TIMEFRAMES period, double adjusted_point);
```

**1.1.2 Percent**

The [Percent()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneypercent) method is called for configuring the appropriate parameter. Its value can be from 0.0 to 100.0, inclusive. The default value is 100.0. Method overriding is not required.

```
void Percent(double percent);
```

**1.1.3** **ValidationSettings**

The [ValidationSettings()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneyvalidationsettings) method is called right from the expert after all the parameters are set. You must override the method if there are additional setup parameters.

```
virtual bool ValidationSettings();
```

The overridden method must return true, if all options are valid (usable). If at least one of the parameters is incorrect, it must return false (further work is impossible). The overridden method must call the base class method with the check of the result.

The [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) base class has the Percent parameter and, accordingly, the base class method, having performed the parameter validation, returns true if the value is within the allowed range, otherwise it returns false.

**1.1.4** **InitIndicators**

The [InitIndicators()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert/cexpertinitindicators) method implements the creation and initialization of all necessary indicators and timeseries. It is called from the expert after all the parameters are set and their correctness is successful verified. The method should be overridden if the risk and money management mechanism uses at least one indicator or timeseries.

```
virtual bool InitIndicators(CIndicators* indicators);
```

Indicators and/or timeseries should be used through the appropriate classes of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary). Pointers of all indicators and/or timeseries should be added to the collection of indicators of an expert (a pointer to which is passed as a parameter).

The overridden method must return true, if all manipulations with the indicators and/or timeseries were successful (they are suitable for use). If at least one operation with the indicators and/or timeseries failed, the method must return false (further work is impossible).

Base class [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) does not use indicators or timeseries, therefore, the base class method always returns true, without performing any action.

**1.2. Methods for determining the volume of a position**

**1.2.1** **CheckOpenLong**

The [CheckOpenLong()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneycheckopenlong) method calculates the volume for opening a long position. It is called by an expert to determine the volume for opening a long position. The method must be overridden, if you expect to calculate long position opening volume using the algorithm that differs from that implemented in the base class.

```
\virtual double CheckOpenLong(double price, double sl);
```

The method must implement the algorithm for calculating the volume for opening a long position. The method must return the calculated volume.

The [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) base class actually has no built-in algorithm for calculating the volume for opening long positions. The base class method always returns the minimum volume possible for a financial instrument.

**1.2.2** **CheckOpenShort**

The [CheckOpenShort()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneycheckopenshort) method calculates the volume for opening a short position. It is called by an expert to determine the volume for opening a short position. The method must be overridden, if you expect to calculate short position opening volume using the algorithm that differs from the one implemented in the base class.

```
virtual double CheckOpenShort(double price, double sl);
```

The method must implement the algorithm for calculating the volume for opening a short position. The method must return the calculated volume.

The [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) base class has no built-in algorithm for calculating the volume for opening short positions. The base class method always returns the minimum volume possible for a financial instrument.

**1.2.3** **CheckReverse**

The [CheckReverse()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneycheckreverse) method calculates the volume for reversing a position. It is called by an expert to determine the volume of a trade operation for reversing a position. The method must be overridden, if you expect to calculate position reversing volume using the algorithm that differs from that implemented in the base class (e.g. reversal with a double volume).

```
virtual double CheckReverse(CPositionInfo* position, double sl);
```

The method must implement the algorithm for calculating the volume to reverse a position, information about which can be obtained by the _position_ pointer. The method must return the calculated volume for the position reversal.

The [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) base class has the following algorithm for calculating the volume for reversing the position - to reverse the position in such a way that the result is an opposite position with the smallest possible volume.

**1.2.4** **CheckClose**

The [CheckClose()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneycheckclose) methods checks whether it is necessary to close a position (in terms of money management and risk management). It is called by an expert to determine whether it is necessary to close a position. The method must be overridden, if you expect to close a position using the algorithm that differs from the one implemented in the base class (e.g. partial closure).

```
virtual double CheckClose(CPositionInfo* position);
```

The method must implement the algorithm for defining the necessity to close a position, information about which can be obtained by the position pointer. The method must return the calculated volume for position closing.

[CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) has the following algorithm to determine whether it is necessary to close the position: the base class method offers to close a position entirely, if the current loss of the position is larger than the specified percent of the deposit.

### 2\. Creating a Mechanism of Money and Risk Management

Now, after we have reviewed the structure of the [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) base class, you can start creating your own risk and money management mechanism. Hereinafter, the risk and money management mechanism will be referred to as "money-manager".

As mentioned above, the [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) class is a set of public virtual "ropes" - methods, using which the expert may know the opinion of the money-manager about the volume of market entering in one direction or another.

Therefore, our primary goal is to create our own class of the money-manager, deriving it from the [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) class and overriding the appropriate virtual methods, implementing the required algorithms.

Our second problem (which is not less important) - to make our class "visible" to [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate"). But, first things first.

**2.1. Creating the class of the trading signals generator**

Let's begin.

First, we create (for example, using the same [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard")) an include file with the mqh extension.

In the File menu select "Create" (or press Ctrl+N key combination) and indicate the creation of an included file:

![Figure 2. Create an include file using MQL5 Wizard](https://c.mql5.com/2/2/MQL5_Wizard_include__2.png)

Figure 2. Create an include file using MQL5 Wizard

It should be noted that in order for the file to be then "detected" by [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") as a money-manager, it should be created in the folder Include\\Expert.

In order not to trash in the [Standard Library](https://www.mql5.com/en/docs/standardlibrary/expertclasses), create our own folder Include\\Expert\\Money\\MyMoneys, in which we create file SampleMoney.mqh, specifying these parameters in the MQL5 Wizard:

![Figure 3. Setting the location of the include file](https://c.mql5.com/2/2/MQL5_Wizard_properties.png)

Figure 3. Setting the location of the include file

As a result of [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard") operation we have the following pattern:

```
//+------------------------------------------------------------------+
//|                                                  SampleMoney.mqh |
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

The following is only "manual" work. Remove the unnecessary parts and add what is required - the include file ExpertMoney.mqh of the [Standard Library](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) with an empty class description.

```
//+------------------------------------------------------------------+
//|                                                  SampleMoney.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Expert\ExpertMoney.mqh>
//+------------------------------------------------------------------+
//| Class CSampleMoney.                                              |
//| Purpose: Class for risk and money management.                    |
//|             It is derived from the CExpertMoney class.           |
//+------------------------------------------------------------------+
class CSampleMoney : public CExpertMoney
  {
  };
//+------------------------------------------------------------------+
```

Now it is necessary to choose the algorithms.

As the basis for our money-manager we take the following algorithm: In the "normal" conditions it is proposed to use a fixed, predetermined deal volume. But if the previous position was closed with a loss, it is proposed to open a position with a doubled volume.

Reflect this in our file.

```
//+------------------------------------------------------------------+
//|                                                  SampleMoney.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Expert\ExpertMoney.mqh>
//+------------------------------------------------------------------+
//| Class CSampleMoney.                                              |
//| Purpose: Class for risk and money management                     |
//|             doubling the volume after a loss deal.               |
//|             It is derived from the CExpertMoney class.           |
//+------------------------------------------------------------------+
class CSampleMoney : public CExpertMoney
  {
  };
//+------------------------------------------------------------------+
```

Define a list of settings for our money-manager. Actually, there will be no list. All settings are included into a single parameter that will determine the volume of a transaction in "normal" conditions.

The parameter will be stored in a protected data member of the class. Access to the parameter will be implemented through an appropriate public method. In the class constructor, the parameter will be initialized by a default value. To check the parameters, let's override the virtual method [ValidationSettings](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneyvalidationsettings) according to the description of the base class.

Let's include these changes in our file:

```
//+------------------------------------------------------------------+
//|                                                  SampleMoney.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Expert\ExpertMoney.mqh>
//+------------------------------------------------------------------+
//| Class CSampleMoney.                                              |
//| Purpose: Class for risk and money management                     |
//|             doubling the volume after a loss deal.               |
//|             It is derived from the CExpertMoney class.           |
//+------------------------------------------------------------------+
class CSampleMoney : public CExpertMoney
  {
protected:
   //--- setup parameters
   double            m_lots;   // deal volume for "normal" conditions

public:
                     CSampleMoney();
   //--- methods to set the parameters
   void              Lots(double lots) { m_lots=lots; }
  };
//+------------------------------------------------------------------+
//| Constructor CSampleMoney.                                        |
//| INPUT:  no.                                                      |
//| OUTPUT: no.                                                      |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
void CSampleMoney::CSampleMoney()
  {
//--- setting the default values
   m_lots=0.1;
  }
//+------------------------------------------------------------------+
```

Separately, let's consider how to implement the ValidationSettings() method. The point is that the base class already has one configuration parameter, which also requires verification.

Therefore, in the overridden method ValidationSettings(), we should call [ValidationSettings()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneyvalidationsettings) of the base class with the check of execution results.

Implementation of the ValidationSettings() method:

```
//+------------------------------------------------------------------+
//| Validation of the setup parameters.                              |
//| INPUT:  no.                                                      |
//| OUTPUT: true if the settings are correct, otherwise false.       |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
bool CSampleMoney::ValidationSettings()
  {
//--- Call the base class method
   if(!CExpertMoney::ValidationSettings()) return(false);
//--- Validation of parameters
   if(m_lots<m_symbol.LotsMin() || m_lots>m_symbol.LotsMax())
     {
      printf(__FUNCTION__+": the deal volume must be in the range %f to %f",m_symbol.LotsMin(),m_symbol.LotsMax());
      return(false);
     }
   if(MathAbs(m_lots/m_symbol.LotsStep()-MathRound(m_lots/m_symbol.LotsStep()))>1.0E-10)
     {
      printf(__FUNCTION__+": the volume of the deal must be multiple of %f",m_symbol.LotsStep());
      return(false);
     }
//--- Successful completion
   return(true);
  }
```

The settings are ready, now let's proceed with the operation of the money manager. We need a method that will determine whether the previous deal was losing and, if necessary, define its volume. Declare it in the class description:

```
class CSampleMoney : public CExpertMoney
  {
protected:
   //--- Setup parameters
   double            m_lots;  // deal volume for "normal" conditions

public:
                    CSampleMoney();
   //--- Methods to set parameters
   void             Lots(double lots) { m_lots=lots; }
   //--- Methods to validate parameters
   virtual bool      ValidationSettings();

protected:
   double            CheckPrevLoss();
  };
```

Implementation of the method:

```
//+------------------------------------------------------------------+
//| Defines whether the prev. deal was losing.                       |
//| INPUT:  no.                                                      |
//| OUTPUT: volume of the prev. deal if it's losing, otherwise 0.0   |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
double CSampleMoney::CheckPrevLoss()
  {
   double lot=0.0;
//--- Request the history of deals and orders
   HistorySelect(0,TimeCurrent());
//--- variables
   int       deals=HistoryDealsTotal();  // Total number of deals in the history
   CDealInfo deal;
//--- Find the previous deal
   for(int i=deals-1;i>=0;i--)
     {
      if(!deal.SelectByIndex(i))
        {
         printf(__FUNCTION__+": Error of deal selection by index");
         break;
        }
      //--- Check the symbol
      if(deal.Symbol()!=m_symbol.Name()) continue;
      //--- Check the profit
      if(deal.Profit()<0.0) lot=deal.Volume();
      break;
     }
//--- Return the volume
   return(lot);
  }
```

Let's consider our algorithms again in more detail (although it is already detailed).

Without going into nuances, we note that our money-manager will propose to increase the volume of a deal upon receipt of loss in the previous deal. If there was no loss in the previous deal, we will offer to open a position with a fixed volume, which is defined by a certain parameter.

For this purpose, we override the virtual methods [CheckOpenLong](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneycheckopenlong) and [CheckOpenShort](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney/cexpertmoneycheckopenshort), filling them with corresponding functionality.

Description of the class:

```
//+------------------------------------------------------------------+
//| Class CSampleMoney.                                              |
//| Purpose: Class for risk and money management                     |
//|             doubling the volume after a loss deal.               |
//|             It is derived from the CExpertMoney class.           |
//+------------------------------------------------------------------+
class CSampleMoney : public CExpertMoney
  {
protected:
   //--- Setup parameters
   double            m_lots;  // Deal volume for "normal" conditions

public:
                    CSampleMoney();
   //--- Methods to set the parameters
   void             Lots(double lots) { m_lots=lots; }
   //--- Methods to validate the parameters
   virtual bool      ValidationSettings();
   //--- Methods to define the volume
   virtual double    CheckOpenLong(double price,double sl);
   virtual double    CheckOpenShort(double price,double sl);

protected:
   double            CheckPrevLoss();
  };
```

Implementations of CheckOpenLong and CheckOpenShort are virtually identical. Both methods determine the necessity of increasing the volume calling the previously implemented CheckPrevLoss method.

Next we need to take into account that we cannot increase the trade volume indefinitely. There are two limitations on the position volume:

1. The maximum volume for a deal for the symbol, specified in the server settings ( [SYMBOL\_VOLUME\_MAX](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double)).
2. Availability of the required amount of free funds on the deposit.

Implementation of methods CheckOpenLong and CheckOpenShort:

```
//+------------------------------------------------------------------+
//| Defining the volume to open a long position.                     |
//| INPUT:  no.                                                      |
//| OUTPUT: lot-if successful, 0.0 otherwise.                        |
//| REMARK: not.                                                     |
//+------------------------------------------------------------------+
double CSampleMoney::CheckOpenLong(double price,double sl)
  {
   if(m_symbol==NULL) return(0.0);
//--- Select the lot size
   double lot=2*CheckPrevLoss();
   if(lot==0.0) lot=m_lots;
//--- Check the limits
   double maxvol=m_symbol.LotsMax();
   if(lot>maxvol) lot=maxvol;
//--- Check the margin requirements
   if(price==0.0) price=m_symbol.Ask();
   maxvol=m_account.MaxLotCheck(m_symbol.Name(),ORDER_TYPE_BUY,price,m_percent);
   if(lot>maxvol) lot=maxvol;
//--- Return the trade volume
   return(lot);
  }
//+------------------------------------------------------------------+
//| Defining the volume to open a short position.                    |
//| INPUT:  no.                                                      |
//| OUTPUT: lot-if successful, 0.0 otherwise.                        |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
double CSampleMoney::CheckOpenShort(double price,double sl)
  {
   if(m_symbol==NULL) return(0.0);
//--- Select the lot size
   double lot=2*CheckPrevLoss();
   if(lot==0.0) lot=m_lots;
//--- Check the limits
   double maxvol=m_symbol.LotsMax();
   if(lot>maxvol) lot=maxvol;
//--- Check the margin requirements
   if(price==0.0) price=m_symbol.Bid();
   maxvol=m_account.MaxLotCheck(m_symbol.Name(),ORDER_TYPE_SELL,price,m_percent);
   if(lot>maxvol) lot=maxvol;
//--- Return the trade volume
   return(lot);
  }
```

So we've solved the first problem. The above code is a "source code" of the money-manager class that satisfies our main task.

**2.2. Creating a description of the generated money-manager class for the MQL5 Wizard**

We now turn to solving the second problem. Our money-manager should be "recognized" by the generator of trading strategies of the MQL5 Wizard.

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
//| Title=Trade with a doubling of lot after a loss                  |
```

3\. Then comes a line with the class type specified in the format "//\| Type=<Type> \|". The <Type> field must have the Money value (in addition to money-managers, the MQL5 Wizard knows other types of classes).

Write:

```
//| Type=Money                                                       |
```

4\. The following line in the format "//\| Name=<Name> \|" is the short name of the signal (it is used by the MQL5 Wizard for generating the names of the global variables of the expert).

We get the following:

```
//| Name=Sample                                                      |
```

5\. The name of a class is an important element of the description. In the line with the format "//\| Class=<ClassNameа> \|", the <ClassName> parameter must match with the name of our class:

```
//| Class=CSampleMoney                                               |
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
//| Parameter=Lots,double,0.1                                        |
//| Parameter=Percent,double,100.0                                   |
```

8\. The block of comment should end with the following lines:

```
//+------------------------------------------------------------------+
// wizard description end
```

2-7 We need to give further explanations to items 2-7. Sections of the class descriptor contain key words (Title, Type, Name, Class, Page, Parameter). Unfortunately, [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") cannot interpret all the possible combinations of characters as part of the class description.

Therefore, to avoid unnecessary errors, write it like this:

\[Slash\]\[Slash\]\[VerticalLine\]\[Space\]<Keyword>\[EqualitySign\]<Description>;

<Description> can contain spaces only for the key word Title. Paragraphs 1 and 8 should be copied "as is".

The class descriptor (first line) must be found in the file no later than the 20th line.

Let's add the descriptor to the source code.

```
//+------------------------------------------------------------------+
//|                                                  SampleMoney.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Expert\ExpertMoney.mqh>
#include <Trade\DealInfo.mqh>
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Trading with lot doubling after a loss                     |
//| Type=Money                                                       |
//| Name=Sample                                                      |
//| Class=CSampleMoney                                               |
//| Page=                                                            |
//| Parameter=Lots,double,0.1                                        |
//| Parameter=Percent,double,100.0                                   |
//+------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//| Class CSampleMoney.                                              |
//| Purpose: Class for risk and money management                     |
//|             doubling the volume after a loss deal.               |
//|             It is derived from the CExpertMoney class.           |
//+------------------------------------------------------------------+
class CSampleMoney : public CExpertMoney
  {
protected:
   //--- Setup parameters
   double            m_lots;  // Deal volume for "normal" conditions

public:
                     CSampleMoney();
   //--- Methods to set the parameters
   void              Lots(double lots) { m_lots=lots; }
   //--- Methods to validate the parameters
   virtual bool      ValidationSettings();
   //--- Methods to define the volume
   virtual double    CheckOpenLong(double price,double sl);
   virtual double    CheckOpenShort(double price,double sl);

protected:
   double            CheckPrevLoss();
  };
//+------------------------------------------------------------------+
//| Constructor CSampleMoney.                                        |
//| INPUT:  no.                                                      |
//| OUTPUT: no.                                                      |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
void CSampleMoney::CSampleMoney()
  {
//--- Setting default values
   m_lots=0.1;
  }
//+------------------------------------------------------------------+
//| Validation of the setup parameters.                              |
//| INPUT:  no.                                                      |
//| OUTPUT: true if the settings are correct, otherwise false.       |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
bool CSampleMoney::ValidationSettings()
  {
//--- Call the base class method
   if(!CExpertMoney::ValidationSettings()) return(false);
//--- Validating the parameters
   if(m_lots<m_symbol.LotsMin() || m_lots>m_symbol.LotsMax())
     {
      printf(__FUNCTION__+": The deal volume must be in the range %f to %f",m_symbol.LotsMin(),m_symbol.LotsMax());
      return(false);
     }
   if(MathAbs(m_lots/m_symbol.LotsStep()-MathRound(m_lots/m_symbol.LotsStep()))>1.0E-10)
     {
      printf(__FUNCTION__+": The deal volume must be multiple of  %f",m_symbol.LotsStep());
      return(false);
     }
//--- Successful completion
   return(true);
  }
//+------------------------------------------------------------------+
//| Defining the volume to open a long position.                     |
//| INPUT:  no.                                                      |
//| OUTPUT: lot-if successful, 0.0 otherwise.                        |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
double CSampleMoney::CheckOpenLong(double price,double sl)
  {
   if(m_symbol==NULL) return(0.0);
//--- Select the lot size
   double lot=2*CheckPrevLoss();
   if(lot==0.0) lot=m_lots;
//--- Check the limits
   double maxvol=m_symbol.LotsMax();
   if(lot>maxvol) lot=maxvol;
//--- Check the margin requirements
   if(price==0.0) price=m_symbol.Ask();
   maxvol=m_account.MaxLotCheck(m_symbol.Name(),ORDER_TYPE_BUY,price,m_percent);
   if(lot>maxvol) lot=maxvol;
//--- Return the trade volume
   return(lot);
  }
//+------------------------------------------------------------------+
//|Defining the volume to open a short position.                     |
//| INPUT:  no.                                                      |
//| OUTPUT: lot-if successful, 0.0 otherwise.                        |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
double CSampleMoney::CheckOpenShort(double price,double sl)
  {
   if(m_symbol==NULL) return(0.0);
//--- Select the lot size
   double lot=2*CheckPrevLoss();
   if(lot==0.0) lot=m_lots;
//--- Check the limits
   double maxvol=m_symbol.LotsMax();
   if(lot>maxvol) lot=maxvol;
//--- Check the margin requirements
   if(price==0.0) price=m_symbol.Bid();
   maxvol=m_account.MaxLotCheck(m_symbol.Name(),ORDER_TYPE_SELL,price,m_percent);
   if(lot>maxvol) lot=maxvol;
//--- Return the trade volume
   return(lot);
  }
//+------------------------------------------------------------------+
//| Defines whether the prev. deal was losing.                       |
//| INPUT:  no.                                                      |
//| OUTPUT: Volume of the prev. deal if it's losing, otherwise 0.0   |
//| REMARK: no.                                                      |
//+------------------------------------------------------------------+
double CSampleMoney::CheckPrevLoss()
  {
   double lot=0.0;
//--- Request the history of deals and orders
   HistorySelect(0,TimeCurrent());
//--- variables
   int       deals=HistoryDealsTotal();  // Total number of deals in the history
   CDealInfo deal;
//--- Find the previous deal
   for(int i=deals-1;i>=0;i--)
     {
      if(!deal.SelectByIndex(i))
        {
         printf(__FUNCTION__+": Error of deal selection by index");
         break;
        }
      //--- Check the symbol
      if(deal.Symbol()!=m_symbol.Name()) continue;
      //---Check the profit
      if(deal.Profit()<0.0) lot=deal.Volume();
      break;
     }
//--- Return the volume
   return(lot);
  }
//+------------------------------------------------------------------+
```

Well that's all. The money-manager is ready to use.

For the [generator of trading strategies of the MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") to be able to use our money-manager, we should restart [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor") (MQL5 Wizard scans the folder Include\\Expert only at boot).

After restarting [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), the created money-manager module can be used in the MQL5 Wizard:

![Figure 5. The created money-manager in the MQL5 Wizard](https://c.mql5.com/2/2/MQL5_Wizard_MM_doubling.png)

Figure 5. The created money-manager in the MQL5 Wizard

The input parameters specified in the section of description of the money-manager parameters are now available:

![Figure 6. Input parameters of the created money-manager in the MQL5 Wizard](https://c.mql5.com/2/2/MQL5_Wizard_MM_parameters__1.png)

Figure 6. Input parameters of the created money-manager in the MQL5 Wizard

The best values of the input parameters of the implemented trading strategy can be found using the [Strategy Tester](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") of the [MetaTrader 5](https://www.metatrader5.com/en/trading-platform "https://www.metatrader5.com/en/trading-platform") terminal.

Figure 7 shows the testing results of the Expert Advisor that trades according to this money management system (EURUSD H1, the testing period: 01.01.2010-05.01.2011).

![Figure 7. Results of testing on the history of the strategy with the money management module with a doubling after a loss](https://c.mql5.com/2/2/BackTesting_Results.png)

Figure 7. Results of testing on the history of the strategy with the money management module with a doubling after a loss

When creating an Expert Advisor, we used the module of trading signals implemented in the article ["MQL5 Wizard: How to create a module of trading signals"](https://www.mql5.com/en/articles/226). The parameters of the Expert Advisor: (PeriodMA=12, ShiftMA=0, MethodMA=MODE\_EMA, AppliedMA=PRICE\_CLOSE, Limit=-70, StopLoss=145, TakeProfit=430, Expiration=10, Lots=0.1, Percent=100).

### Conclusion

[The generator of trading strategies](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") of the MQL5 Wizard greatly simplifies the testing of trading ideas. The code of the generated expert is based on the [classes of trading strategies](https://www.mql5.com/en/docs/standardlibrary/expertclasses) of the Standard Library, which are used for creating certain implementations of trading signal classes, money and risk management classes and position support classes.

The article describes how to develop a custom risk and money management module and enable it in the MQL5 Wizard. As an example we've considered a money management algorithm, in which the size of the trade volume is determined by the results of the previous deal. The structure and format of description of the created class for the MQL5 Wizard are also described.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/230](https://www.mql5.com/ru/articles/230)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/230.zip "Download all attachments in the single ZIP archive")

[samplemoney.mqh](https://www.mql5.com/en/articles/download/230/samplemoney.mqh "Download samplemoney.mqh")(7.15 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/3155)**
(4)


![Dmitiry Ananiev](https://c.mql5.com/avatar/2021/5/60A1913E-6AF5.jpg)

**[Dmitiry Ananiev](https://www.mql5.com/en/users/dimeon)**
\|
18 Jan 2011 at 16:50

```
 HistorySelect(0,TimeCurrent());
```

Such a construction with a large [number of orders in the history](https://www.mql5.com/en/docs/trading/historyorderstotal "MQL5 documentation: HistoryOrdersTotal function") will slow down [a lot](https://www.mql5.com/en/docs/trading/historyorderstotal "MQL5 documentation: HistoryOrdersTotal function"). Recently Roche published an article on how to copy orders for the last 24 hours into the cache. I put it into my Expert Advisor and MM stopped slowing down the tests.

![Victor Kirillin](https://c.mql5.com/avatar/avatar_na2.png)

**[Victor Kirillin](https://www.mql5.com/en/users/unclevic)**
\|
18 Jan 2011 at 17:23

**dimeon:**

Such a construction with a large [number of orders in the history](https://www.mql5.com/en/docs/trading/historyorderstotal "MQL5 documentation: HistoryOrdersTotal function") will slow down [a lot](https://www.mql5.com/en/docs/trading/historyorderstotal "MQL5 documentation: HistoryOrdersTotal function"). Recently Roche published an article on how to copy orders for the last 24 hours into the cache. I put it into my Expert Advisor and MM stopped slowing down the tests.

Thank you for your attention.

The code is given for example.

Since the Expert Advisor is not tied to a specific timeframe, there is no possibility to determine the necessary depth of the trade history query.

For example, when testing (or working) on daily candlesticks, the history for the last 24 hours will hardly help you.

So, choose the depth of history based on the specific situation (as you have done).

![Stephen Njuki](https://c.mql5.com/avatar/avatar_na2.png)

**[Stephen Njuki](https://www.mql5.com/en/users/ssn)**
\|
30 Mar 2011 at 08:24

For those who are slightly adventurous, here is a martingale. To trade with fixed lots simply set the increase factor to 0.


![Эдуард](https://c.mql5.com/avatar/avatar_na2.png)

**[Эдуард](https://www.mql5.com/en/users/47rxkfn)**
\|
4 May 2023 at 11:09

You should write an example of how to initialise at least the base class CExpertMoney, otherwise you can't figure out how to use it!


![MQL5 Wizard: How to Create a Module of Trailing of Open Positions](https://c.mql5.com/2/0/MQL5_Wizard_Trailing_Stop__1.png)[MQL5 Wizard: How to Create a Module of Trailing of Open Positions](https://www.mql5.com/en/articles/231)

The generator of trade strategies MQL5 Wizard greatly simplifies the testing of trading ideas. The article discusses how to write and connect to the generator of trade strategies MQL5 Wizard your own class of managing open positions by moving the Stop Loss level to a lossless zone when the price goes in the position direction, allowing to protect your profit decrease drawdowns when trading. It also tells about the structure and format of the description of the created class for the MQL5 Wizard.

![Exposing C# code to MQL5 using unmanaged exports](https://c.mql5.com/2/0/logo__5.png)[Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)

In this article I presented different methods of interaction between MQL5 code and managed C# code. I also provided several examples on how to marshal MQL5 structures against C# and how to invoke exported DLL functions in MQL5 scripts. I believe that the provided examples may serve as a basis for future research in writing DLLs in managed code. This article also open doors for MetaTrader to use many libraries that are already implemented in C#.

![Trade Events in MetaTrader 5](https://c.mql5.com/2/0/trade_events.png)[Trade Events in MetaTrader 5](https://www.mql5.com/en/articles/232)

A monitoring of the current state of a trade account implies controlling open positions and orders. Before a trade signal becomes a deal, it should be sent from the client terminal as a request to the trade server, where it will be placed in the order queue awaiting to be processed. Accepting of a request by the trade server, deleting it as it expires or conducting a deal on its basis - all those actions are followed by trade events; and the trade server informs the terminal about them.

![Orders, Positions and Deals in MetaTrader 5](https://c.mql5.com/2/0/TradeIndo_MQL5.png)[Orders, Positions and Deals in MetaTrader 5](https://www.mql5.com/en/articles/211)

Creating a robust trading robot cannot be done without an understanding of the mechanisms of the MetaTrader 5 trading system. The client terminal receives the information about the positions, orders, and deals from the trading server. To handle this data properly using the MQL5, it's necessary to have a good understanding of the interaction between the MQL5-program and the client terminal.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=brwisshxjltabbgounbytldkpksaexlv&ssn=1769158319521102235&ssn_dr=0&ssn_sr=0&fv_date=1769158319&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F230&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%3A%20How%20to%20Create%20a%20Risk%20and%20Money%20Management%20Module%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915831990472213&fz_uniq=5062776491026065583&sv=2552)

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