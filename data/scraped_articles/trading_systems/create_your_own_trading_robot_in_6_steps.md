---
title: Create Your Own Trading Robot in 6 Steps!
url: https://www.mql5.com/en/articles/367
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:01:26.661582
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pvvnkquuxrnulwkhctwgimhwmgmhgrom&ssn=1769191285937494690&ssn_dr=0&ssn_sr=0&fv_date=1769191285&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F367&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Create%20Your%20Own%20Trading%20Robot%20in%206%20Steps!%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919128508263248&fz_uniq=5071513854665501044&sv=2552)

MetaTrader 5 / Examples


### One More Time about the MQL5 Wizard

The world around us is changing rapidly, and we try to keep up with it. We do not have time to learn something new, and this is a normal attitude of a normal human being. Traders are people just like everyone else, they want to get maximum results for the minimum of effort. Specially for traders, MetaEditor 5 offers a wonderful [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard"). There are several articles describing how to create an automated trading system using the wizard, including a "light version" [MQL5 Wizard for Dummies](https://www.mql5.com/en/articles/287) and a "version from developers " - [MQL5 Wizard: New Version](https://www.mql5.com/en/articles/275).

It all seems good - a trading robot is created in 5 mouse clicks, you can test it in the Strategy Tester and [optimize the parameters](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization "https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization") of a trading system, you can let the resulting robot trade on your account without the need to do anything else manually. But the problem arises when the a trader/MQL5 developer wants to create something of his own, something unique which has never been described anywhere, and is going to write his own module of trading signals. The trader opens the MQL5 documentation, gets to the Standard Library, and is horrified to see...

### Five Terrible Classes

True, the MQL5 Wizard greatly simplifies the creation of Expert Advisors, but first you need to learn what will be used as input for it. To **automatically** create an Expert Advisor using the MQL5 Wizard, make sure that its components adhere to five basic classes of the section [Base Classes of Expert Advisors](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses):

- [CExpertBase](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertbase) is a base class for four other classes.

- [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) is the class for creating a trading robot; this is the class that trades.

- [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) **is a class for creating a module of trading signals; the article is about this class**.

- [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) is a class for trailing a protecting Stop Loss.
- [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) is the money management class.


Here is the whole force of the "great and terrible" approach that is called [Object-oriented programming](https://www.mql5.com/en/docs/basis/oop) (OOP). But don't be afraid, now almost everyone has a cell phone with lots of function, and almost no one knows how it works. We do not need to study all this, we will only discuss some functions of the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) class.

![](https://c.mql5.com/2/4/four_trade_classes.gif)

In this article we will go through the [stages of creating a module of trading signals](https://www.mql5.com/en/articles/367#module_create_stages), and you will see how to do this without having to learn OOP or the classes. But if you want, you can go a little further then.

### 1\. Creating a Class from Scratch

We will not alter any existing module of trading signals to our needs, because it's the way to get confused. Therefore, we will simply write our own class, but first we will use the [Navigator](https://www.metatrader5.com/en/metaeditor/help/workspace/navigator "https://www.metatrader5.com/en/metaeditor/help/workspace/navigator") to create a new folder to store our signals in **MQL5/Include/Expert/**.

![](https://c.mql5.com/2/4/create_folder.png)

Right-click on the folder we have created, select "New File" and create a new class for our module of trading signals.

![](https://c.mql5.com/2/4/New-Class.gif)

Fill in the fields:

- Class Name - the name of the class. This will be a module for generating signals at the intersection of two moving averages, so let's name it MA\_Cross.

- Base Name is the class from which our class is [derived](https://www.mql5.com/en/docs/basis/oop/inheritance). And we should derive it from the base class [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal).


Click "Finish" and a draft of our module us ready. It's all east so far. We only need to add the [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) declaration to the resulting file so that the compiler knows where to find the base class CExpertSignal

```
#include "..\ExpertSignal.mqh"   // CExpertSignal is in the file ExpertSignal
```

The result:

```
//+------------------------------------------------------------------+
//|                                                     MA_Cross.mqh |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "..\ExpertSignal.mqh"   // CExpertSignal is in the file ExpertSignal
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class MA_Cross : public CExpertSignal
  {
private:

public:
                     MA_Cross();
                    ~MA_Cross();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MA_Cross::MA_Cross()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MA_Cross::~MA_Cross()
  {
  }
//+------------------------------------------------------------------+
```

Check the resulting class (it must be free of [compilation errors](https://www.mql5.com/en/docs/constants/errorswarnings/errorscompile)) and click [F7](https://www.metatrader5.com/en/metaeditor/help/development/compile "https://www.metatrader5.com/en/metaeditor/help/development/compile"). There are no errors and we can move on.

### 2\. A Handle to the Module

Our class is completely empty, it has no errors and we can test it - let's try to create a new Expert Advisor in the MQL5 Wizard based on it. We reach the step of selecting a module of trading signals and see ... that our module is not there.

![](https://c.mql5.com/2/4/Check_new_class__1.png)

And how can it be there? We do not add any indications for the MQL5 Wizard to understand that our class could be something useful. Let's fix this. If you look at the modules of the standard package, you'll see that each of them contains a header at the beginning of the file. This is the handle of the module compiled according to certain rules. And the rules are very simple.

Open, for example, the source code of the module of AMA based trading signals (see the logic description in [Signals of the Adaptive Moving Average](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ama).) And run the MQL5 Wizard choosing this module. Compare:

![](https://c.mql5.com/2/4/2012-02-16_15_09_51.png)

The last block in the handle refers to the module parameters, the first line contains the name of the module to be displayed in the MQL5 Wizard. As you can see, there is nothing complicated. Thus, the handle of each module contains the following entries:

- Title \- the module name to be shown in the MQL5 Wizard.
- Type - the version of the module of signals. It must always be SignalAdvanced.
- Name - the name of the module after its is selected in the MQL5 Wizard and is used in comments for describing internal parameters of the generated Expert Advisor (preferably specified).

- ShortName - a prefix for automatic naming of external parameters in the generated Expert Advisor (in the form of Signal\_<ShortName>\_<ParameterName>).

- Class - the name of the, which is contained in the module.

- Page - a parameter to get Help for this module (only for modules from the [standard delivery](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal)).

Next comes the description of the parameters in the form of Parameter=list\_of\_values, in which the following is specified (comma-separated):

1. The name of the function to set the value of the parameter when starting the Expert Advisor.
2. The parameter type can be enumeration.
3. The default value for the parameter, i.e. the value that will be set to the parameter, if you do not change it in the MQL5 Wizard.
4. Description of the parameter, which you see when you start the Expert Advisor generated in the MQL5 Wizard.

Now, knowing all this, let's create the handle of our module of trading signals. So, we are writing a module for getting trading signals at the intersection of two moving averages. We need to set at least four external parameters:

- FastPeriod - the period of the fast moving average
- FastMethod - the type of smoothing of the fast moving average
- SlowPeriod - the period of the slow moving average
- SlowMethod - the type of smoothing of the slow moving average

You could also add a shift and the type of prices to calculate each of the moving averages, but it does not change anything fundamentally. So the current version is as follows:

```
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Signals at the intersection of two MAs                     |
//| Type=SignalAdvanced                                              |
//| Name=My_MA_Cross                                                 |
//| ShortName=MaCross                                                |
//| Class=MA_Cross                                                   |
//| Page=Not needed                                                  |
//| Parameter=FastPeriod,int,13,Period of fast MA                    |
//| Parameter=FastMethod,ENUM_MA_METHOD,MODE_SMA,Method of fast MA   |
//| Parameter=SlowPeriod,int,21,Period of slow MA                    |
//| Parameter=SlowMethod,ENUM_MA_METHOD,MODE_SMA,Method of slow MA   |
//+------------------------------------------------------------------+
// wizard description end
```

The module handle is ready, and we have described the following in it:

1. The name displayed in the MQL5 Wizard - "Signals at the intersection of two moving averages".
2. Four external parameter to configure the trading signals.

   - FastPeriod - the period of the fast moving average with the default value of 13.
   - FastMethod - the type of smoothing of the fast moving average, simple smoothing by default.
   - SlowPeriod - the period of the slow moving average with the default value of 21.
   - SlowMethod - the type of smoothing of the slow moving average, simple smoothing by default.

Save the changes and compile. There should not be any errors. Run the MQL5 Wizard to check. You see, our module is now available for selection, and it shows all of our parameters!

![](https://c.mql5.com/2/4/description_test__1.gif)

Congratulations, our module of trading signal looks great now!

### 3\. Methods for Setting Parameters

Now it is time to work with the external parameters. Since our trading module is represented by the class MA\_Cross, then its parameters must be stored within the same class as [private](https://www.mql5.com/en/docs/basis/variables#protected) members. Let's add four lines (equal to the number of parameters) to the class declaration. We've already described the parameter in the handle and know the following:

```
class MA_Cross : public CExpertSignal
  {
private:
   //--- Configurable module parameters
   int               m_period_fast;    // Period of the fast MA
   int               m_period_slow;    // Period of the slow MA
   ENUM_MA_METHOD    m_method_fast;    // Type of smoothing of the fast MA
   ENUM_MA_METHOD    m_method_slow;    // Type of smoothing of the slow MA
```

But how do the values ​​of the external parameters of the module appear in the **appropriate** members of our class MA\_Cross? It's all very simple, you only need to declare public methods **of the same name** in the class, namely, to add four lines to the public section:

```
class MA_Cross : public CExpertSignal
  {
private:
   //--- Configurable module parameters
   int               m_period_fast;    // Period of the fast MA
   int               m_period_slow;    // Period of the slow MA
   ENUM_MA_METHOD    m_method_fast;    // Type of smoothing of the fast MA
   ENUM_MA_METHOD    m_method_slow;    // Type of smoothing of the slow MA

public:
   //--- Constructor of class
                     MA_Cross();
   //--- Destructor of class
                    ~MA_Cross();
   //--- Methods for setting
   void              FastPeriod(int value)               { m_period_fast=value;        }
   void              FastMethod(ENUM_MA_METHOD value)    { m_method_fast=value;        }
   void              SlowPeriod(int value)               { m_period_slow=value;        }
   void              SlowMethod(ENUM_MA_METHOD value)    { m_method_slow=value;        }
   };
```

When you generate an Expert Advisor on the basis of this module using the MQL5 Wizard and run it on the chart, these four methods are automatically called when initializing the Expert Advisor. So here is a simple rule:

**The rule of parameter creation in the module -** for each parameter that we have declared in the handle, we should create a private member in the class for storing its value and a public member for setting a value to it. The method name must match the name of the parameter.

And the last moment is to set default values ​​for our parameters that will be used in case the methods of value setting are not called. Each declared variable or class member **must be initialized**. This technique allows to avoid many of hard-to-find errors.

For automatic initialization, the best suiting one is the class constructor; it is always the first one to be called when creating an object. For default values, we will use those written in the module handle.

```
class MA_Cross : public CExpertSignal
  {
private:
   //--- Configurable module parameters
   int               m_period_fast;    // Period of the fast MA
   ENUM_MA_METHOD    m_method_fast;     // Type of smoothing of the fast MA
   int               m_period_slow;    // Period of the slow MA
   ENUM_MA_METHOD    m_method_slow;     // Type of smoothing of the slow MA

public:
   //--- Constructor of class
                     MA_Cross(void);
   //--- Destructor of class
                    ~MA_Cross(void);
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
MA_Cross::MA_Cross(void) : m_period_fast(13),          // Default period of the fast MA is 3
                           m_method_fast(MODE_SMA),    // Default smoothing method of the fast MA
                           m_period_slow(21),          // Default period of the slow MA is 21
                           m_method_slow(MODE_SMA)     // Default smoothing method of the slow MA
  {
  }
```

Here the class members are initialized using the [initialization list](https://www.mql5.com/en/docs/basis/types/classes#initialization_list).

As you can see, we haven't used moving average indicators yet. We found a simple rule - **as many parameters are stated in the handle** of the module, so many **methods and members should be in the class** that implements the module. There is nothing complicated! However, don't forget to set default values of parameters on the constructor.

### 4\. Check the Correctness of Input Parameters

We have created parameters for our trading module, written methods for setting values ​​to them, and now comes the next important phase - the correctness of parameters must be checked. In our case, we must check the periods of moving averages and the type of smoothing for their calculation. For this purpose you should write **your own** [ValidationSettings()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalvalidationsettings) method in the class. This method is defined in the parent class [CExpertBase](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertbase), and in all its [children](https://www.mql5.com/en/docs/basis/oop/inheritance) it is **obligatorily** redefined.

But if you do not know anything about object-oriented programming, just remember - in our class we should write the ValidationSettings() function, which requires no parameters and returns true or false.

```
class MA_Cross : public CExpertSignal
  {
...
   //--- Constructor of class
                     MA_Cross(void);
   //--- Destructor of class
                    ~MA_Cross(void);
   //--- Checking correctness of input data
   bool              ValidationSettings();
...
   };
//+------------------------------------------------------------------+
//| Checks input parameters and returns true if everything is OK     |
//+------------------------------------------------------------------+
bool MA_Cross:: ValidationSettings()
  {
   //--- Call the base class method
   if(!CExpertSignal::ValidationSettings())  return(false);
   //--- Check periods, number of bars for the calculation of the MA >=1
   if(m_period_fast<1 || m_period_slow<1)
     {
      PrintFormat("Incorrect value set for one of the periods! FastPeriod=%d, SlowPeriod=%d",
                  m_period_fast,m_period_slow);
      return false;
     }
//--- Slow MA period must be greater that the fast MA period
   if(m_period_fast>m_period_slow)
     {
      PrintFormat("SlowPeriod=%d must be greater than FastPeriod=%d!",
                  m_period_slow,m_period_fast);
      return false;
     }
//--- Fast MA smoothing type must be one of the four values of the enumeration
   if(m_method_fast!=MODE_SMA && m_method_fast!=MODE_EMA && m_method_fast!=MODE_SMMA && m_method_fast!=MODE_LWMA)
     {
      PrintFormat("Invalid type of smoothing of the fast MA!");
      return false;
     }
//--- Show MA smoothing type must be one of the four values of the enumeration
   if(m_method_slow!=MODE_SMA && m_method_slow!=MODE_EMA && m_method_slow!=MODE_SMMA && m_method_slow!=MODE_LWMA)
     {
      PrintFormat("Invalid type of smoothing of the slow MA!");
      return false;
     }
//--- All checks are completed, everything is ok
   return true;
  }
```

As you can see, in the public part of the MA\_Cross class we've added declaration of the ValidationSettings() method, and then added [the method body](https://www.mql5.com/en/docs/basis/function#function_body) in the following form:

```
bool MA_Cross:: ValidationSettings()
```

First comes the return type, then the class name, then [scope resolution operator](https://www.mql5.com/en/docs/basis/operations/other#context_allow) ::, and all this is followed by the name of the previously declared method. Do not forget that the name and type of parameters must match in the declaration and description of the class method. However, the compiler will warn you of [such an error](https://www.mql5.com/en/docs/constants/errorswarnings/errorscompile#224).

Note that first the base class method is called, and then input parameters are checked.

```
//--- Call the base class method
   if(!CExpertSignal::ValidationSettings())  return(false);
//--- Our code to check the values of parameters
```

If you do not add this line, the generated Expert Advisor will not be able to initialize our module of trading signals.

### 5\. Where Are Our Indicators?

It's time to work with the indicators, since all the preparatory work with the parameters for them have been completed. Each module of trading signals contains the [InitIndicators()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalinitindicators) method, which is automatically called when you run the generated Expert Advisor. In this method, we must provide indicators of moving averages for our module.

First, declare the InitIndicators() method in the class and paste its draft:

```
public:
   //--- Constructor of class
                     MA_Cross(void);
   //--- Destructor of class
                    ~MA_Cross(void);
   //--- Methods for setting
   void              FastPeriod(int value)               { m_period_fast=value;        }
   void              FastMethod(ENUM_MA_METHOD value)    { m_method_fast=value;        }
   void              SlowPeriod(int value)               { m_period_slow=value;        }
   void              SlowMethod(ENUM_MA_METHOD value)    { m_method_slow=value;        }
   //--- Checking correctness of input data
   bool              ValidationSettings();
   //--- Creating indicators and timeseries for the module of signals
   bool              InitIndicators(CIndicators *indicators);
  };
...
//+------------------------------------------------------------------+
//| Creates indicators                                               |
//| Input:  a pointer to a collection of indicators                  |
//| Output: true if successful, otherwise false                      |
//+------------------------------------------------------------------+
bool MA_Сross::InitIndicators(CIndicators* indicators)
  {
//--- Standard check of the collection of indicators for NULL
   if(indicators==NULL)                           return(false);
//--- Initializing indicators and timeseries in additional filters
   if(!CExpertSignal::InitIndicators(indicators)) return(false);
//--- Creating our MA indicators
   ... Some code here
//--- Reached this part, so the function was successful, return true
   return(true);
  }
```

So there is nothing complicated, we declare the method and then simply create the method body, as we have [done for the ValidationSettings() method](https://www.mql5.com/en/articles/367#validate_parameters). Above all, do not forget to insert the class name and the operator :: in the function definition. We have a draft, which we can insert into a code to create moving averages. Let's do this properly - for each indicator we create a separate function in the class, which returns true if successful. The function can have any name, but let it reflect its purpose, so let's call the functions CreateFastMA() and CreateSlowMA().

```
protected:
   //--- Creating MA indicators
   bool              CreateFastMA(CIndicators *indicators);
   bool              CreateSlowMA(CIndicators *indicators);
  };
//+------------------------------------------------------------------+
//| Creates indicators                                               |
//| Input:  a pointer to a collection of indicators                  |
//| Output: true if successful, otherwise false                      |
//+------------------------------------------------------------------+
bool MA_Cross::InitIndicators(CIndicators *indicators)
  {
//--- Standard check of the collection of indicators for NULL
   if(indicators==NULL) return(false);
//--- Initializing indicators and timeseries in additional filters
   if(!CExpertSignal::InitIndicators(indicators)) return(false);
//--- Creating our MA indicators
   if(!CreateFastMA(indicators))                  return(false);
   if(!CreateSlowMA(indicators))                  return(false);
//--- Reached this part, so the function was successful, return true
   return(true);
  }
//+------------------------------------------------------------------+
//| Creates the "Fast MA" indicator                                  |
//+------------------------------------------------------------------+
bool MA_Cross::CreateFastMA(CIndicators *indicators)
  {
... Some code
//--- Reached this part, so the function was successful, return true
   return(true);
  }
//+------------------------------------------------------------------+
//| Creates the "Slow MA" indicator                                  |
//+------------------------------------------------------------------+
bool MA_Cross::CreateSlowMA(CIndicators *indicators)
  {
... Some code
//--- Reached this part, so the function was successful, return true
   return(true);
  }
```

That's all, we only need to write code that generates the MA indicators and somehow integrates the handles of these indicators into the trading module, so that the module can use the values ​​of these indicators. That is why a pointer to a variable of type [CIndicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2) is passed as a parameter. The following is written in Documentation about it:

The CIndicators is a class for collecting instances of timeseries and technical indicators classes. The CIndicators class provides creation of instanced of technical indicator classes, their storage and management (data synchronization, handle and memory management).

This means that we must create our indicators and place them in this collection. Since only indicators of the [CIndicator](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicator) form and its children can be stored in the collection, we should use this fact. We will use [CiCustom](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator), which is the above mentioned child. For each moving average we declare an object of type CiCustom in the private part of the class:

```
class MA_Cross : public CExpertSignal
  {
private:
   CiCustom          m_fast_ma;            // The indicator as an object
   CiCustom          m_slow_ma;            // The indicator as an object
   //--- Configurable module parameters
   int              m_period_fast;   // Period of the fast MA
   ENUM_MA_METHOD    m_method_fast;    // Type of smoothing of the fast MA
   int              m_period_slow;   // Period of the slow MA
   ENUM_MA_METHOD    m_method_slow;    // Type of smoothing of the slow MA
```

Of course, you can create your own indicator class, which will be derived from [CIndicator](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicator), and implement all the necessary methods for use with the MQL5 Wizard. But in this case we want to show how you can use **any custom** indicator in the module of trading signals using [CiCustom](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator).

Here's how it looks in the code:

```
//+------------------------------------------------------------------+
//| Creates the "Fast MA" indicator                                  |
//+------------------------------------------------------------------+
bool MA_Cross::CreateFastMA(CIndicators *indicators)
  {
//--- Checking the pointer
   if(indicators==NULL) return(false);
//--- Adding an object to the collection
   if(!indicators.Add(GetPointer(m_fast_ma)))
     {
      printf(__FUNCTION__+": Error adding an object of the fast MA");
      return(false);
     }
//--- Setting parameters of the fast MA
   MqlParam parameters[4];
//---
   parameters[0].type=TYPE_STRING;
   parameters[0].string_value="Examples\\Custom Moving Average.ex5";
   parameters[1].type=TYPE_INT;
   parameters[1].integer_value=m_period_fast;      // Period
   parameters[2].type=TYPE_INT;
   parameters[2].integer_value=0;                  // Shift
   parameters[3].type=TYPE_INT;
   parameters[3].integer_value=m_method_fast;      // Averaging method
//--- Object initialization
   if(!m_fast_ma.Create(m_symbol.Name(),m_period,IND_CUSTOM,4,parameters))
     {
      printf(__FUNCTION__+": Error initializing the object of the fast MA");
      return(false);
     }
//--- Number of buffers
   if(!m_fast_ma.NumBuffers(1)) return(false);
//--- Reached this part, so the function was successful, return true
   return(true);
  }
```

In the CreateFastMA() method, first check the pointer of the collection of indicators, and then add a pointer of the fast MA m\_fast\_ma to this collection. Then declare the [MqlParam](https://www.mql5.com/en/docs/constants/structures/mqlparam) structure, which is especially designed for storing parameters of custom indicators, and fill it with values.

We use [Custom Moving Average](https://www.mql5.com/en/code/25) from the standard terminal delivery pack as the custom MA indicator. The name of the indicator must be indicated relative to the folder **data\_folder/MQL5/Indicators/**. Since Custom Moving Average.mq5' from the standard package is located in **data\_folder/MQL5/Indicators/** Examples/, we specify its path including the Examples folder:

```
parameters[0].string_value="Examples\\Custom Moving Average.ex5";
```

If you look at the code for this indicator, you can see all the required data:

```
//--- input parameters
input int            InpMAPeriod=13;       // Period
input int            InpMAShift=0;         // Shift
input ENUM_MA_METHOD InpMAMethod=MODE_SMMA;  // Method
```

The values ​​of the structure contain the type-value pairs:

1. parameter type - string (to transfer the name of the indicator)

2. the name of the executable file of the custom indicator - "Custom Moving Averages.exe"
3. parameter type - int (value of the period)
4. period of the moving average
5. parameter type - int (shift value)
6. horizontal shift of the average in bars
7. parameter type - int (enumeration value is an integer)
8. method of averaging


After filling the structure, the indicator is initialized by the [Create()](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicator/cindicatorcreate) method of all the required parameters: symbol name and the timeframe on which it is calculated, the type of the indicator from the [ENUM\_INDICATOR](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_indicator) enumeration, the number of indicator parameters and the MqlParam structure with parameter values. And the last one is specifying the number of indicator buffers using the [NumBuffers()](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/customindicator/cicustomnumbuffers) method.

The CreateSlowMA() method for creating the slow moving average is simple. When using custom indicators in the module, do not forget that the Expert Advisor generated by the MQL5 Wizard will also run in the tester. So at the beginning of our file we add the property [#property tester\_indicator](https://www.mql5.com/en/docs/basis/preprosessor/compilation) that communicates to the tester the location of required indicators:

```
#include "..\ExpertSignal.mqh"   // The CExpertSignal class is in the file ExpertSignal
#property tester_indicator "Examples\\Custom Moving Average.ex5"
```

If we use several different indicators, we should add this line for each of them. So, we have added the indicators. For more convenience, let's provide two methods of receiving MA values:

```
   //--- Checking correctness of input data
   bool              ValidationSettings(void);
   //--- Creating indicators and timeseries for the module of signals
   bool              InitIndicators(CIndicators *indicators);
   //--- Access to indicator data
   double            FastMA(const int index)             const { return(m_fast_ma.GetData(0,index)); }
   double            SlowMA(const int index)             const { return(m_slow_ma.GetData(0,index)); }
```

As you can see, the methods are very simple, they used the [GetData()](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicator/cindicatorgetdata) method of the SIndicator parent class, which returns a value from the specified indicator buffer at the specified position.

If you need classes for working with classical indicators of the standard package, they are available in section [Classes for working with indicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators). We are ready to proceed to the final stage.

### 6\. Define the LongCondition and ShortCondition Methods

Everything is ready to make our module work and generate trading signals. This functionality is provided by two methods that must be described in each child of [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal):

- [LongCondition()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignallongcondition) checks the buy conditions and returns the strength of the Long signal from 0 to 100.

- [ShortCondition()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal/cexpertsignalshortcondition) \-  checks the sell condition and returns  the strength of the Short signal from 0 to 100.

If the function returns a null value, it means that there is no trading signal. If there are conditions for the signal, then you can estimate the strength of the signal and return any value not exceeding 100. Evaluation of the signal strength allows you to flexibly build trading systems based on several modules and market models. Read more about this in [MQL5 Wizard: New Version](https://www.mql5.com/en/articles/275).

Since we are writing a simple module of trading signals, we can agree that the buy and sell signals are valued equally (100). Let's add necessary methods in the class declaration.

```
   ...
   bool              InitIndicators(CIndicators *indicators);
   //--- Access to data of the indicators
   double            FastMA(const int index)             const { return(m_fast_ma.GetData(0,index)); }
   double            SlowMA(const int index)             const { return(m_slow_ma.GetData(0,index)); }
   //--- Checking buy and sell conditions
   virtual int       LongCondition();
   virtual int       ShortCondition();
```

Also, let's create the description of functions. This is how the buy signal is checked (it's all the same with the sell signal):

```
//+------------------------------------------------------------------+
//| Returns the strength of the buy signal                           |
//+------------------------------------------------------------------+
int MA_Cross::LongCondition()
  {
   int signal=0;
//--- For operation with ticks idx=0, for operation with formed bars idx=1
   int idx=StartIndex();
//--- Values of MAs at the last formed bar
   double last_fast_value=FastMA(idx);
   double last_slow_value=SlowMA(idx);
//--- Values of MAs at the last but one formed bar
   double prev_fast_value=FastMA(idx+1);
   double prev_slow_value=SlowMA(idx+1);
//---If the fast MA crossed the slow MA from bottom upwards on the last two closed bars
   if((last_fast_value>last_slow_value) && (prev_fast_value<prev_slow_value))
     {
      signal=100; // There is a signal to buy
     }
//--- Return the signal value
   return(signal);
  }
```

Note that we have declare the idx variable, to which the value returned by the [StartIndex()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertbase/cexpertbasestartindex) function of the parent class CExpertBase is assigned. The [StartIndex()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertbase/cexpertbasestartindex) function returns 0, if the Expert Advisor is designed to work on all ticks, and in this case the analysis starts with the current bar. If the Expert Advisor is designed to work at open prices, [StartIndex()](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertbase/cexpertbasestartindex) returns 1 and the analysis starts with the last formed bar.

**By default** **StartIndex() returns 1**, which means that the Expert Advisor generated by the MQL5 Wizard will only run at the opening of a new bar and will ignore incoming ticks during formation of the current bar.

How to activate this mode and how it can be used will be described later in [the finishing stroke](https://www.mql5.com/en/articles/367#Expert_EveryTick).

The module is ready for use, so let's create a trading robot in the MQL5 Wizard based on this module.

### Checking an Expert Advisor in the Tester

To test the efficiency of our module, let's generate an Expert Advisor based on it in the MQL5 Wizard and run it on the chart. The "Inputs" tab of the appeared start window contains the parameters of the MA\_Cross module.

![](https://c.mql5.com/2/4/input_test_ma.png)

All other parameters have also been added by the MQL5 Wizard while generating the EA based on the selected money management module and position maintenance module (Trailing Stop). Thus, we only had to write a module of trading signals and received a ready solution. This is the main advantage of using the MQL5 Wizard!

Now let's test the trading robot in the MetaTrader 5 Strategy Tester. Let's try to run a quick optimization of key parameters.

![](https://c.mql5.com/2/4/optimization_inputs.png)

In these settings of input parameters, more than half a million of passes is required for full optimization. Therefore, we choose [fast optimization](https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types "https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types") (genetic algorithm) and additionally utilize [MQL5 Cloud Network to accelerate the optimization](https://www.mql5.com/en/articles/341). The optimization has been done in 10 minutes and we have got the results.

![](https://c.mql5.com/2/4/opt_results.png)

As you can see, creating a trading robot in MQL5 and optimization of input parameters have taken much less time than would be required for writing the position management servicing logic, debugging and searching for the best algorithms.

### Finishing Stroke

You can skip this item or go back to it later when you are completely comfortable with the technique of writing a module of trading signals.

If you open the source code of the Expert Advisor generated by the MQL5 Wizard, you will find the [global variable](https://www.mql5.com/en/docs/basis/variables/global) Expert\_EveryTick with the false value. Based on this variable, [the StartIndex() function](https://www.mql5.com/en/articles/367#StartIndex) returns its value. It communicates to the Expert Advisor the mode it should run in.

```
//+------------------------------------------------------------------+
//|                                                 TestMA_Cross.mq5 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include                                                          |
//+------------------------------------------------------------------+
#include <Expert\Expert.mqh>
//--- available signals
#include <Expert\MySignals\MA_Cross.mqh>
//--- available trailing
#include <Expert\Trailing\TrailingNone.mqh>
//--- available money management
#include <Expert\Money\MoneyFixedLot.mqh>
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
//--- inputs for expert
input string         Expert_Title             ="TestMA_Cross";  // Document name
ulong               Expert_MagicNumber       =22655;          // Expert Advisor ID
bool                  Expert_EveryTick             =false;          // Work of the EA inside the bar
//--- inputs for main signal
input int            Signal_ThresholdOpen     =10;             // Signal threshold value to open [0...100]
input int            Signal_ThresholdClose    =10;             // Signal threshold value to close [0...100]
```

If you set Expert\_EveryTick true and compile the code, the trading robot will analyze each incoming tick, and thus make decisions on the values ​​of the current incomplete bar. Do this only if you understand how it works. Not all trading systems are designed to work inside the bar.

You can also add a keyword [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) for the Expert\_EveryTick parameter, and then you will have a new [input parameter of the Expert Advisor](https://www.metatrader5.com/en/terminal/help/algotrading/autotrading#create "https://www.metatrader5.com/en/terminal/help/algotrading/autotrading#create"), which you can set at the EA startup on a chart or in the tester:

```
input bool          Expert_EveryTick         =false;          // Work of the EA inside the bar
```

And now it's time to summarize what we have done.

### 6 Steps to Create a Module of Trading Signals

If you have mastered MQL5, then you no longer need to write an Expert Advisor from scratch. Just create a module of trading signals and, based on this module, automatically generate a trading robot with the enabled trailing and trade volume management modules. And even if you are not familiar with OOP or do not want to delve much into the structure of trade classes, you can just go through 6 steps:

1. [Create a new class](https://www.mql5.com/en/articles/367#create_class) using the MQL5 Wizard in a separate folder MQL5/Include/MySignals/. Our module of trading signals will be stored there.

2. Create [a module handle](https://www.mql5.com/en/articles/367#descriptor) that describes the parameters, their type and default values.
3. Declare [module parameters](https://www.mql5.com/en/articles/367#set_parameters) in the class and add methods for initialization in the constructor.

4. [Check the input parameters](https://www.mql5.com/en/articles/367#validate_parameters) and do not forget to call ValidationSettings() of the CExpertSignal base class.

5. Create [indicator-objects](https://www.mql5.com/en/articles/367#create_indicators) and add a predefined initialization method InitIndicators().

6. Identify [conditions](https://www.mql5.com/en/articles/367#make_conditions) of trading signals in the methods LongCondition() and ShortCondition().

Each step is simple and requires little skill in MQL5 programming. You only need to write your module once, following the instructions, and further verification of any trade idea will take no more than an hour, without tiring hours of coding and debugging.

### From Simple to Complex

Remember that the trading strategy implemented by your trading robot created using the MQL5 Wizard, is as complex as the module of trading signals it uses. But before you start to build a complex trading system based on a set of rules for entry and exit, split it into several simple systems and check each one separately.

Based on simple modules you can create complex trading strategies using the ready-made modules of trading signals, but this is a topic for another article!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/367](https://www.mql5.com/ru/articles/367)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/367.zip "Download all attachments in the single ZIP archive")

[testma\_cross.mq5](https://www.mql5.com/en/articles/download/367/testma_cross.mq5 "Download testma_cross.mq5")(7.14 KB)

[ma\_cross.mqh](https://www.mql5.com/en/articles/download/367/ma_cross.mqh "Download ma_cross.mqh")(11.57 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/6395)**
(82)


![Longsen Chen](https://c.mql5.com/avatar/2021/4/6066B2E5-2923.jpg)

**[Longsen Chen](https://www.mql5.com/en/users/gchen2101)**
\|
29 Aug 2020 at 05:30

Many articles [attached source](https://www.mql5.com/en/articles/24#insert-code "MQL5.community - User Memo: Insert Code") codes, but many of them can not be compiled correctly.

This article is very good because I downloaded it and compiled without an error neighter a warning!

I'm testing it now. It looks awesome.

![Buy_sell](https://c.mql5.com/avatar/2020/9/5F60E408-F78B.JPG)

**[Buy\_sell](https://www.mql5.com/en/users/newwinner2020)**
\|
16 Nov 2020 at 09:46

Very good article, thanks to the authors.


![odunoaki2](https://c.mql5.com/avatar/avatar_na2.png)

**[odunoaki2](https://www.mql5.com/en/users/odunoaki2)**
\|
15 Aug 2021 at 15:30

Thank you for the great article.

I have a question. When I run the code in the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 "), I get

'''

CIndicator::GetData:Invalid Buffer

'''

Do you have any idea how to fix it?

Thank you.

![ali1400](https://c.mql5.com/avatar/avatar_na2.png)

**[ali1400](https://www.mql5.com/en/users/ali1400)**
\|
30 Oct 2021 at 19:00

Very Great Article. Very Helpful !!


![Nikita Gamolin](https://c.mql5.com/avatar/2022/12/638E6C1A-9A2A.png)

**[Nikita Gamolin](https://www.mql5.com/en/users/n0namer)**
\|
7 Jan 2023 at 00:02

**MetaQuotes:**

Published article [Create a trading robot in 6 steps!](https://www.mql5.com/en/articles/367)

Author: [MetaQuotes](https://www.mql5.com/en/users/MetaQuotes)

Colleagues, help.How to generate a closing signal via CheckCloseLong/Short from the Signals Module?I have not found the answer in the article.

![Fractal Analysis of Joint Currency Movements](https://c.mql5.com/2/17/927_11.png)[Fractal Analysis of Joint Currency Movements](https://www.mql5.com/en/articles/1351)

How independent are currency quotes? Are their movements coordinated or does the movement of one currency suggest nothing of the movement of another? The article describes an effort to tackle this issue using nonlinear dynamics and fractal geometry methods.

![On Methods of Technical Analysis and Market Forecasting](https://c.mql5.com/2/17/982_30.gif)[On Methods of Technical Analysis and Market Forecasting](https://www.mql5.com/en/articles/1350)

The article demonstrates the capabilities and potential of a well-known mathematical method coupled with visual thinking and an "out of the box" market outlook. On the one hand, it serves to attract the attention of a wide audience as it can get the creative minds to reconsider the trading paradigm as such. And on the other, it can give rise to alternative developments and program code implementations regarding a wide range of tools for analysis and forecasting.

![Analyzing the Indicators Statistical Parameters](https://c.mql5.com/2/0/Analysis_Indicators.png)[Analyzing the Indicators Statistical Parameters](https://www.mql5.com/en/articles/320)

The technical analysis widely implements the indicators showing the basic quotes "more clearly" and allowing traders to perform analysis and forecast market prices movement. It's quite obvious that there is no sense in using the indicators, let alone applying them in creation of trading systems, unless we can solve the issues concerning initial quotes transformation and the obtained result credibility. In this article we show that there are serious reasons for such a conclusion.

![The Box-Cox Transformation](https://c.mql5.com/2/0/Cox-Box-transformation_MQL5.png)[The Box-Cox Transformation](https://www.mql5.com/en/articles/363)

The article is intended to get its readers acquainted with the Box-Cox transformation. The issues concerning its usage are addressed and some examples are given allowing to evaluate the transformation efficiency with random sequences and real quotes.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/367&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071513854665501044)

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