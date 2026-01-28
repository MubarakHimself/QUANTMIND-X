---
title: Programming EA's Modes Using Object-Oriented Approach
url: https://www.mql5.com/en/articles/1246
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:46:05.847004
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/1246&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070544759719663563)

MetaTrader 5 / Examples


### Introduction

In this article we are going to discuss programming modes, in which an MQL5 EA can work. The objective of this article is to describe the idea that "each mode is implemented in its own way". The author believes that this approach allows completion of tasks at different stages of development of an EA more efficiently.

At first, we consider what stages the development of an EA consists of. Then the modes, in which an EA in [MetaTrader 5](https://www.metatrader5.com/en/trading-platform "https://www.metatrader5.com/en/trading-platform") can work and its helper applications are explored. Development of the hierarchy of classes for implementing the above idea finishes this article.

### 1\. Development Stages

Development of a trading robot (EA) is a multi-aspect process. The key blocks here are algorithmization of the idea and testing it. Notably, both EA's trading logic and the code algorithm get tested.

As a scheme, the stages of this process can be represented as follows (Fig.1).

![Fig.1. Development stages and implementation of an EA](https://c.mql5.com/2/17/Fig1_Development_stages.png)

Fig.1. Development stages and implementation of an EA

The fifth stage "Algorithmic Trading" showcases the work of the developers, programmers, analysts and other specialists involved. It often happens that all these roles are fulfilled by one person. Let us assume, that it is a trader-programmer.

This scheme can be updated and extended. In my opinion, it illustrates the most important points in development of an EA. The cyclical pattern of this scheme allows improving and modifying the EA's code through its lifetime.

It should be noted that every stage requires certain tools, knowledge and skills.

In my opinion, the developer comes across the following simple variant matrix (Fig.2).

![Fig.2. Variant matrix](https://c.mql5.com/2/17/Fig2_Variant_matrix.png)

Fig.2. Variant matrix

Clearly, only the robot implementing a winning trading strategy with high quality code is to make it to the fifth stage "Algorithmic Trading".

### 2\. The Expert Advisor Modes in MQL5

The MQL5 environment allows working with an EA in different modes. There are 7 of them. We are going to consider each of them further down.

From the perspective of the program file type, 2 groups can be distinguished:

1. Modes requiring the source code file and the executable file;
2. Modes requiring the executable file only.

Debug and profiling modes belong to the first group.

Another criterion of mode classification is work of an EA in a stream of real or historical quotes. All testing modes are connected with historical quotes.

6 modes are defined by programming. A conclusion if an EA is working in a standard (release) mode or not can be made based on the results. A ready program (file with the \*.ex5 extension), which was coded for work on financial markets, is supposed to work in this very mode. At the same time, a ready program enables using other modes in the Strategy Tester too.

Let us create an enumeration of the operational modes of the MQL program called ENUM\_MQL\_MODE:

```
//+------------------------------------------------------------------+
//| MQL Mode                                                         |
//+------------------------------------------------------------------+
enum ENUM_MQL_MODE
  {
   MQL_MODE_RELEASE=0,       // Release
   MQL_MODE_DEBUG=1,         // Debugging
   MQL_MODE_PROFILER=2,      // Profiling
   MQL_MODE_TESTER=3,        // Testing
   MQL_MODE_OPTIMIZATION=4,  // Optimization
   MQL_MODE_VISUAL=5,        // Visual testing
   MQL_MODE_FRAME=6,         // Gathering frames
  };
```

Later on, this will be required for recognizing the mode type in which an EA is working.

**2.1. Function of Identifying and Checking the Mode**

Write a simple function that will iterate over all modes and print information in the journal.

```
//+------------------------------------------------------------------+
//| Checking all MQL modes                                           |
//+------------------------------------------------------------------+
void CheckMqlModes(void)
  {
//--- if it is debug mode
   if(MQLInfoInteger(MQL_DEBUG))
      Print("Debug mode: yes");
   else
      Print("Debug mode: no");
//--- if it is code profiling mode
   if(MQLInfoInteger(MQL_PROFILER))
      Print("Profile mode: yes");
   else
      Print("Profile mode: no");
//--- if it is test mode
   if(MQLInfoInteger(MQL_TESTER))
      Print("Tester mode: yes");
   else
      Print("Tester mode: no");
//--- if it is optimization mode
   if(MQLInfoInteger(MQL_OPTIMIZATION))
      Print("Optimization mode: yes");
   else
      Print("Optimization mode: no");
//--- if it is visual test mode
   if(MQLInfoInteger(MQL_VISUAL_MODE))
      Print("Visual mode: yes");
   else
      Print("Visual mode: no");
//--- if it is frame gathering optimization result mode
   if(MQLInfoInteger(MQL_FRAME_MODE))
      Print("Frame mode: yes");
   else
      Print("Frame mode: no");
  }
```

Work of this function in every mode is going to be checked. It can be called in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) event handler.

For the purpose of the test, let us create a template of the EA called Test1\_Modes\_EA.mq5.

An option to specify the mode where the EA is going to work is enabled in input parameters. It is important to make sure that the correct mode is named otherwise information will be inaccurate. That was what happened.

Below is the release mode.

```
CL      0       17:20:38.932    Test1_Modes_EA (EURUSD.e,H1)     Current mode: MQL_MODE_RELEASE
QD      0       17:20:38.932    Test1_Modes_EA (EURUSD.e,H1)     Debug mode: no
KM      0       17:20:38.932    Test1_Modes_EA (EURUSD.e,H1)     Profile mode: no
EK      0       17:20:38.932    Test1_Modes_EA (EURUSD.e,H1)     Tester mode: no
CS      0       17:20:38.932    Test1_Modes_EA (EURUSD.e,H1)     Optimization mode: no
RJ      0       17:20:38.932    Test1_Modes_EA (EURUSD.e,H1)     Visual mode: no
GL      0       17:20:38.932    Test1_Modes_EA (EURUSD.e,H1)     Frame mode: no
```

For release mode, the flags of all other modes were zeroed. So, the function identified that it was neither debug mode (Debug mode: no), nor profiling mode (Profile mode: no) etc. Using the method of negation, we came to the conclusion that we are working in release mode.

Now we are going to see how debug mode was identified.

```
HG      0       17:27:47.709    Test1_Modes_EA (EURUSD.e,H1)     Current mode: MQL_MODE_DEBUG
LD      0       17:27:47.710    Test1_Modes_EA (EURUSD.e,H1)     Debug mode: yes
RS      0       17:27:47.710    Test1_Modes_EA (EURUSD.e,H1)     Profile mode: no
HE      0       17:27:47.710    Test1_Modes_EA (EURUSD.e,H1)     Tester mode: no
NJ      0       17:27:47.710    Test1_Modes_EA (EURUSD.e,H1)     Optimization mode: no
KD      0       17:27:47.710    Test1_Modes_EA (EURUSD.e,H1)     Visual mode: no
RR      0       17:27:47.710    Test1_Modes_EA (EURUSD.e,H1)     Frame mode: no
```

Debug mode was recognized correctly.

Any handbook on programming contains information that debugging facilitates search and error localization in the code. It also highlights peculiarities of the program. More details about debugging in the MQL5 environment can be found in the article ["Debugging MQL5 Programs"](https://www.mql5.com/en/articles/654).

This mode is most commonly used at the stages of formalizing and constructing the algorithm of a trading idea.

In programming, debugging is enabled either using the **IS\_DEBUG\_MODE** macros or the [MQLInfoInteger()](https://www.mql5.com/en/docs/check/mqlinfointeger) function with the **MQL\_DEBUG** identifier.

We are moving on to profiling mode.

```
GS      0       17:30:53.879    Test1_Modes_EA (EURUSD.e,H1)     Current mode: MQL_MODE_PROFILER
OR      0       17:30:53.879    Test1_Modes_EA (EURUSD.e,H1)     Debug mode: no
GE      0       17:30:53.879    Test1_Modes_EA (EURUSD.e,H1)     Profile mode: yes
QM      0       17:30:53.879    Test1_Modes_EA (EURUSD.e,H1)     Tester mode: no
CE      0       17:30:53.879    Test1_Modes_EA (EURUSD.e,H1)     Optimization mode: no
FM      0       17:30:53.879    Test1_Modes_EA (EURUSD.e,H1)     Visual mode: no
GJ      0       17:30:53.879    Test1_Modes_EA (EURUSD.e,H1)     Frame mode: no
```

The function correctly estimated that the [Profiler](https://www.metatrader5.com/en/metaeditor/help/workspace/toolbox#profile "https://www.metatrader5.com/en/metaeditor/help/workspace/toolbox#profile") was involved.

In this mode it can be checked how quickly the program works. The Profiler passes the information on time expenditure to program blocks. This instrument is supposed to point out bottlenecks of an algorithm. They are not always possible to get rid of but nevertheless this information can be useful.

Profiling can be enabled either through the **IS\_PROFILE\_MODE** macros or the [MQLInfoInteger()](https://www.mql5.com/en/docs/check/mqlinfointeger) function with the **MQL\_PROFILER** identifier.

Now let us have a look at test mode. This information will appear in the "Journal" tab of the Strategy Tester.

```
EG      0       17:35:25.397    Core 1  2014.11.03 00:00:00   Current mode: MQL_MODE_TESTER
OS      0       17:35:25.397    Core 1  2014.11.03 00:00:00   Debug mode: no
GJ      0       17:35:25.397    Core 1  2014.11.03 00:00:00   Profile mode: no
ER      0       17:35:25.397    Core 1  2014.11.03 00:00:00   Tester mode: yes
ED      0       17:35:25.397    Core 1  2014.11.03 00:00:00   Optimization mode: no
NL      0       17:35:25.397    Core 1  2014.11.03 00:00:00   Visual mode: no
EJ      0       17:35:25.397    Core 1  2014.11.03 00:00:00   Frame mode: no
```

Test mode was identified correctly.

This is the EA's default mode when the Strategy Tester gets opened.

There are no macros for this mode and therefore in MQL5 we can only determine it using the [MQLInfoInteger()](https://www.mql5.com/en/docs/check/mqlinfointeger) function with the **MQL\_TESTER** identifier.

Now we are moving on to the optimization. A journal with records will be stored in the agent's folder. In my case the path is as follows: %Program Files\\MetaTrader5\\tester\\Agent-127.0.0.1-3000\\logs

```
OH      0       17:48:14.010    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Current mode: MQL_MODE_OPTIMIZATION
KJ      0       17:48:14.010    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Debug mode: no
NO      0       17:48:14.010    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Profile mode: no
FI      0       17:48:14.010    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Tester mode: yes
KE      0       17:48:14.010    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Optimization mode: yes
LS      0       17:48:14.010    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Visual mode: no
QE      0       17:48:14.010    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Frame mode: no
```

If optimization mode is active, test mode is enabled by default.

Optimization mode is active in the Strategy Tester if the "Optimization" field is not disabled in the "Settings" tab.

To find out if the EA is being tested in optimization mode in MQL5, call either the [MQLInfoInteger()](https://www.mql5.com/en/docs/check/mqlinfointeger) function with the **MQL\_OPTIMIZATION** identifier.

We are proceeding to visualization mode.

```
JQ      0       17:53:51.485    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Current mode: MQL_MODE_VISUAL
JK      0       17:53:51.485    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Debug mode: no
KF      0       17:53:51.485    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Profile mode: no
CP      0       17:53:51.485    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Tester mode: yes
HJ      0       17:53:51.485    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Optimization mode: no
LK      0       17:53:51.485    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Visual mode: yes
KS      0       17:53:51.485    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Frame mode: no
```

Here we can see that visual testing mode and standard testing mode are involved.

The EA works in this mode in the Strategy Tester if the "Visualization" field in the "Settings" tab is flagged.

Establishing the fact of testing an MQL5 program in visual testing mode can be done using the [MQLInfoInteger()](https://www.mql5.com/en/docs/check/mqlinfointeger) function with the **MQL\_VISUAL\_MODE** identifier.

The last mode is handling frames mode.

```
HI      0       17:59:10.177    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Current mode: MQL_MODE_FRAME
GR      0       17:59:10.177    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Debug mode: no
JR      0       17:59:10.177    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Profile mode: no
JG      0       17:59:10.177    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Tester mode: yes
GM      0       17:59:10.177    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Optimization mode: yes
HR      0       17:59:10.177    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Visual mode: no
MI      0       17:59:10.177    Test1_Modes_EA (EURUSD.e,H1)     2014.11.03 00:00:00   Frame mode: no
```

Interestingly enough, the function recognized only testing and optimization modes as the flag of frames was zeroed. If the call of the function is transferred to the [OnTesterInit()](https://www.mql5.com/en/docs/basis/function/events#ontesterinit) handler, the "Experts" journal is going to contain the following entries:

```
IO      0       18:04:27.663    Test1_Modes_EA (EURUSD.e,H1)     Current mode: MQL_MODE_FRAME
GE      0       18:04:27.663    Test1_Modes_EA (EURUSD.e,H1)     Debug mode: no
ML      0       18:04:27.663    Test1_Modes_EA (EURUSD.e,H1)     Profile mode: no
CJ      0       18:04:27.663    Test1_Modes_EA (EURUSD.e,H1)     Tester mode: no
QR      0       18:04:27.663    Test1_Modes_EA (EURUSD.e,H1)     Optimization mode: no
PL      0       18:04:27.663    Test1_Modes_EA (EURUSD.e,H1)     Visual mode: no
GS      0       18:04:27.663    Test1_Modes_EA (EURUSD.e,H1)     Frame mode: yes
```

Effectively, now only gathering frames mode was detected.

This mode is used in the Strategy Tester if the "Optimization" field in the "Settings" tab is not disabled. As experience has shown, this mode is defined in the body of the [OnTesterInit()](https://www.mql5.com/en/docs/basis/function/events#ontesterinit), [OnTesterPass()](https://www.mql5.com/en/docs/basis/function/events#ontesterpass) and [OnTesterDeinit()](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) handlers.

The [MQLInfoInteger()](https://www.mql5.com/en/docs/check/mqlinfointeger) function with the **MQL\_FRAME\_MODE** identifier can facilitate identifying the fact of testing an EA in gathering frames mode.

Below is the code of the service function MqlMode(), which automatically specifies the mode in which the EA is working.

```
//+------------------------------------------------------------------+
//| Identify the current MQL mode                                    |
//+------------------------------------------------------------------+
ENUM_MQL_MODE MqlMode(void)
  {
   ENUM_MQL_MODE curr_mode=WRONG_VALUE;

//--- if it is debug mode
   if(MQLInfoInteger(MQL_DEBUG))
      curr_mode=MQL_MODE_DEBUG;
//--- if it is code profiling mode
   else if(MQLInfoInteger(MQL_PROFILER))
      curr_mode=MQL_MODE_PROFILER;
//--- if it is visual test mode
   else if(MQLInfoInteger(MQL_VISUAL_MODE))
      curr_mode=MQL_MODE_VISUAL;
//--- if it is optimization mode
   else if(MQLInfoInteger(MQL_OPTIMIZATION))
      curr_mode=MQL_MODE_OPTIMIZATION;
//--- if it is test mode
   else if(MQLInfoInteger(MQL_TESTER))
      curr_mode=MQL_MODE_TESTER;
//--- if it is frame gathering optimization result mode
   else if(MQLInfoInteger(MQL_FRAME_MODE))
      curr_mode=MQL_MODE_FRAME;
//--- if it is release mode
   else
      curr_mode=MQL_MODE_RELEASE;
//---
   return curr_mode;
  }
```

Since standard testing is identified at optimization and visual testing mode, then standard testing mode is to be checked after optimization and visualization mode.

If we take a look at the work of the function in the second template of the Test2\_Modes\_EA.mq5 EA, we can see that a new entry appears in the journal when the template is launched. For instance, for profiling mode the below entry was made:

```
HG      0       11:23:52.992    Test2_Modes_EA (EURUSD.e,H1)    Current mode: MQL_MODE_PROFILER
```

We discussed the details of operational modes of the MQL5 Expert for creating class models corresponding to a specified mode. We are going to implement it in the next part of the article.

### 3\. Template of the EA Designed to Work in Different Modes

I suggest going over the development stages of an EA again.

At the stage of algorithmization, a programmer most often does debugging and profiling. For testing historical data, they try all modes of the Strategy Tester. The final mode (release mode) is utilized in online trading.

In my opinion, an EA must be multi-faceted in the sense that requirement of the development and testing stages have to be reflected in its code.

At that, the main algorithm will be preserved and, following it, the EA will have different behavior at different modes. The [object - oriented programming](https://www.mql5.com/en/docs/basis/oop) tools set suits perfectly for implementing this idea.

![Fig.2 Class hierarchy for the EA designed to work in different modes](https://c.mql5.com/2/12/2__1.png)

Fig.3. Class hierarchy for the EA designed to work in different modes

Class hierarchy with implementation of different modes is represented on Fig.3.

The basic class CModeBase encapsulating all common things will have two direct descendants: the CModeRelease and the CModeTester classes. The first one will be a parent to the debugging classes and the second one for the classes connected with testing the EA on historical data.

Let us develop the idea of combining a procedural and modular approach when developing class methods in the context of modes. As an example let us consider the following trading logic:

1. Open by a signal if there is no open position;
2. Closing by the signal if there is an open position;
3. Trailing Stop if there is an open position.

The trading signal is detected by the standard indicator [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") when a new bar appears.

A signal to buy appears when the main line is going upwards and crossing the signal one in the negative zone of the MACD indicator (Fig.4).

![Fig.4 Signal to buy](https://c.mql5.com/2/12/3__2.png)

Fig.4. Signal to buy

A signal to sell appears when the main line is going downwards and crosses the signal one in the positive zone of the indicator (Fig. 5).

![Fig.5 Signal to sell](https://c.mql5.com/2/12/4__2.png)

Fig.5. Signal to sell

The position gets closed either when the opposite signal appears or by the Stop Loss, which is placed in case the mode of a position support is enabled.

Then the definition of the basic class CModeBase is as follows:

```
//+------------------------------------------------------------------+
//| Class CModeBase                                                  |
//| Purpose: a base class for MQL-modes                              |
//+------------------------------------------------------------------+
class CModeBase
  {
//--- === Data members === ---
private:
   //--- a macd object & values
   CiMACD            m_macd_obj;
   double            m_macd_main_vals[2];
   double            m_macd_sig_vals[2];

protected:
   long              m_pos_id;
   bool              m_is_new_bar;
   uint              m_trailing_stop;
   uint              m_trail_step;
   //--- trade objects
   CSymbolInfo       m_symbol_info;
   CTrade            m_trade;
   CPositionInfo     m_pos_info;
   CDealInfo         m_deal_info;
   //--- mql mode
   ENUM_MQL_MODE     m_mql_mode;

   //--- a new bar object
   CisNewBar         m_new_bar;
   //--- current tick signal flag
   bool              m_is_curr_tick_signal;
   //--- close order type
   ENUM_ORDER_TYPE   m_close_ord_type;

//--- === Methods === ---
public:
   //--- constructor/destructor
   void              CModeBase();
   void             ~CModeBase(void){};
   //--- initialization
   virtual bool      Init(int _fast_ema,int slow_ema,int _sig,ENUM_APPLIED_PRICE _app_price);
   virtual void      Deinit(void){};

   //--- Modules
   virtual void      Main(void){};

   //--- Procedures
   virtual void      Open(void){};
   virtual void      Close(void){};
   virtual void      Trail(void){};

   //--- Service
   static ENUM_MQL_MODE CheckMqlMode(void);
   ENUM_MQL_MODE     GetMqlMode(void);
   void              SetMqlMode(const ENUM_MQL_MODE _mode);
   void              SetTrailing(const uint _trailing,const uint _trail_step);

protected:
   //--- Functions
   ENUM_ORDER_TYPE   CheckOpenSignal(const ENUM_ORDER_TYPE _open_sig);
   ENUM_ORDER_TYPE   CheckCloseSignal(const ENUM_ORDER_TYPE _close_sig);
   ENUM_ORDER_TYPE   CheckTrailSignal(const ENUM_ORDER_TYPE _trail_sig,double &_sl_pr);
   //---
   double            GetMacdVal(const int _idx,const bool _is_main=true);

private:
   //--- Macros
   bool              RefreshIndicatorData(void);
   //--- Normalization
   double            NormalPrice(double d);
   double            NormalDbl(double d,int n=-1);
   double            NormalSL(const ENUM_ORDER_TYPE _ord_type,double op,double pr,
                              uint SL,double stop);
   double            NormalTP(const ENUM_ORDER_TYPE _ord_type,double op,double pr,
                              uint _TP,double stop);
   double            NormalLot(const double _lot);
  };
```

Anything can be included in a basic class as long as it is going to be used in the inheritor classes.

The [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") data will be unavailable for descendants as they are represented by private members.

It must be noted that among those methods there are virtual ones: Main(), Open(), Close(), Trail(). Their implementation will vastly depend on the mode the EA is currently working in. These methods will stay empty for the basic class.

Besides, the basic class comprises methods that have the same trading logic for all MQL modes. All signal methods belong to them:

- CModeBase::CheckOpenSignal(),
- CModeBase::CheckCloseSignal(),
- CModeBase::CheckTrailSignal().

It should be kept in mind, that this article does not target writing code for all MQL mode types. Standard and visual testing are going to serve an example.

**3.1. Test Mode**

After algorithm has been coded and compiled, I usually try the strategy out on historical data in the Strategy Tester to check if it works as designed.

Most often it is required to check how precisely the system implements trading signals. In any case, the basic aim at this stage for an EA is to launch and trade.

The CModeTester class for regular testing can be implemented as follows:

```
//+------------------------------------------------------------------+
//| Class CModeTester                                                |
//| Purpose: a class for the tester mode                             |
//| Derives from class CModeBase.                                    |
//+------------------------------------------------------------------+
class CModeTester : public CModeBase
  {
//--- === Methods === ---
public:
   //--- constructor/destructor
   void              CModeTester(void){};
   void             ~CModeTester(void){};

   //--- Modules
   virtual void      Main(void);

   //--- Procedures
   virtual void      Open(void);
   virtual void      Close(void);
   virtual void      Trail(void);
  };
```

The main module is implemented like:

```
//+------------------------------------------------------------------+
//| Main module                                                      |
//+------------------------------------------------------------------+
void CModeTester::Main(void)
  {
//--- 1) closure
   this.Close();
//--- 2) opening
   this.Open();
//--- 3) trailing stop
   this.Trail();
  }
```

For the mode of regular testing, create an opportunity to print information about trading signals in the Journal.

Add strings containing indicator values that are considered to be the source of the trading signal.

Below is an extract from the Journal about a signal to open a position followed by a signal to close.

```
HE      0       13:34:04.118    Core 1  2014.11.14 22:15:00   ---=== Signal to open: SELL===---
FI      0       13:34:04.118    Core 1  2014.11.14 22:15:00   A bar before the last one, main: 0.002117; signal: 0.002109
DL      0       13:34:04.118    Core 1  2014.11.14 22:15:00   The last bar, main: 0.002001; signal: 0.002118
LO      0       13:34:04.118    Core 1  2014.11.14 22:15:00   market sell 0.03 EURUSD.e (1.25242 / 1.25251 / 1.25242)
KH      0       13:34:04.118    Core 1  2014.11.14 22:15:00   deal #660 sell 0.03 EURUSD.e at 1.25242 done (based on order #660)
GE      0       13:34:04.118    Core 1  2014.11.14 22:15:00   deal performed [#660 sell 0.03 EURUSD.e at 1.25242]
OD      0       13:34:04.118    Core 1  2014.11.14 22:15:00   order performed sell 0.03 at 1.25242 [#660 sell 0.03 EURUSD.e at 1.25242]
IK      0       13:34:04.118    Core 1  2014.11.14 22:15:00   CTrade::OrderSend: market sell 0.03 EURUSD.e [done at 1.25242]
IL      0       13:34:04.118    Core 1  2014.11.17 13:30:20
CJ      0       13:34:04.118    Core 1  2014.11.17 13:30:20   ---=== Signal to close: SELL===---
GN      0       13:34:04.118    Core 1  2014.11.17 13:30:20   A bar before the last one, main: -0.001218; signal: -0.001148
QL      0       13:34:04.118    Core 1  2014.11.17 13:30:20   The last bar, main: -0.001123; signal: -0.001189
EP      0       13:34:04.118    Core 1  2014.11.17 13:30:20   market buy 0.03 EURUSD.e (1.25039 / 1.25047 / 1.25039)
FG      0       13:34:04.118    Core 1  2014.11.17 13:30:20   deal #661 buy 0.03 EURUSD.e at 1.25047 done (based on order #661)
OJ      0       13:34:04.118    Core 1  2014.11.17 13:30:20   deal performed [#661 buy 0.03 EURUSD.e at 1.25047]
PD      0       13:34:04.118    Core 1  2014.11.17 13:30:20   order performed buy 0.03 at 1.25047 [#661 buy 0.03 EURUSD.e at 1.25047]
HE      0       13:34:04.118    Core 1  2014.11.17 13:30:20   CTrade::OrderSend: market buy 0.03 EURUSD.e [done at 1.25047]
```

Please note that there is no familiar journal "Experts" in the Strategy Tester. All information can be found in the "Journal" tab, which contains records about actions performed by the Strategy Tester during testing and optimization.

That is why one has to search for the required strings. If the entry information is required to be separated, it can be recorded into a file.

Strategy for standard testing is implemented in the code of the **TestMode\_tester.mq5** EA.

**3.2. Visual Testing Mode**

Sometimes it is required to refer to a live chart and see how an EA is handling current situation.

Simple visualization allows not only to see how the trading system is reacting to ticks but also to compare similar price models at the end of testing.

Definition of the CModeVisual class for visual testing can be as follows:

```
//+------------------------------------------------------------------+
//| Class CModeVisual                                                |
//| Purpose: a class for the tester mode                             |
//| Derived from class CModeBase.                                    |
//+------------------------------------------------------------------+
class CModeVisual : public CModeTester
  {
//--- === Data members === ---
private:
   CArrayObj         m_objects_arr;
   double            m_subwindow_max;
   double            m_subwindow_min;

//--- === Methods === ---
public:
   //--- constructor/destructor
   void              CModeVisual(void);
   void             ~CModeVisual(void);

   //--- Procedures
   virtual void      Open(void);
   virtual void      Close(void);

private:
   bool              CreateSignalLine(const bool _is_open_sig,const bool _is_new_bar=true);
   bool              CreateRectangle(const ENUM_ORDER_TYPE _signal);
   void              RefreshRectangles(void);
  };
```

The class contains hidden members. A member of the class **m\_objects\_arr** implements a dynamic array of the [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) type. Graphical objects, for example, lines and rectangles, belong here. Two other class members ( **m\_subwindow\_max**, **m\_subwindow\_min**) control maximum and minimum sizes of the indicator's subwindow.

Private methods are responsible for work with graphical objects.

This class does not contain the Main() and the Trail() methods. Their parent analogues CModeTester::Main() and CModeTester::Trail() are going to be called respectively.

Graphical objects can be created in the visual testing mode. This cannot be done in other modes of the Strategy Tester.

Let a red vertical be drawn on the chart when a signal to enter appears and a blue vertical when a signal to exit is received. Fill the space between the entry and exit points with a rectangle of the relevant color in the indicator's subwindow.

If it is a long position, then the rectangle is light blue. If the position is short, the rectangle is pink (Fig.6).

![Fig.6. Graphical objects in the visual testing mode](https://c.mql5.com/2/12/Fig6_Graphic_objects_in_visual_testing_mode.png)

Fig.6. Graphical objects in the visual testing mode

The height of the rectangle depends on the values of maximum and minimum of the chart's subwindow at the time of creation. To make all rectangles equal in size, a block of changing rectangle coordinates should be added to code in case the coordinates of the chart subwindow change.

So in the subwindow of the [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") indicator we get the following areas: uncolored (no position), pink (short position), light blue (long position).

Strategy for the visual testing mode is implemented in the code of the **TestMode\_visual\_tester.mq5** EA.

### Conclusion

In this article I tried to illustrate mode capabilities of the MetaTrader 5 terminal and the MQL5 language. It has to be said that a multi-mode approach to programming of a trading algorithm involves more costs on the one hand and on the other hand there is an opportunity to consider each development stage one after another. The [object-oriented programming](https://www.mql5.com/en/docs/basis/oop) is a resourceful aid for a programmer in this case.

The optimization and frame gathering modes will be highlighted in the future articles about the statistic properties of a trading system.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1246](https://www.mql5.com/ru/articles/1246)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1246.zip "Download all attachments in the single ZIP archive")

[cisnewbar.mqh](https://www.mql5.com/en/articles/download/1246/cisnewbar.mqh "Download cisnewbar.mqh")(13.74 KB)

[modes.mqh](https://www.mql5.com/en/articles/download/1246/modes.mqh "Download modes.mqh")(62.77 KB)

[test1\_modes\_ea.mq5](https://www.mql5.com/en/articles/download/1246/test1_modes_ea.mq5 "Download test1_modes_ea.mq5")(9.43 KB)

[test2\_modes\_ea.mq5](https://www.mql5.com/en/articles/download/1246/test2_modes_ea.mq5 "Download test2_modes_ea.mq5")(10.01 KB)

[testmode\_tester.mq5](https://www.mql5.com/en/articles/download/1246/testmode_tester.mq5 "Download testmode_tester.mq5")(6.12 KB)

[testmode\_visual\_tester.mq5](https://www.mql5.com/en/articles/download/1246/testmode_visual_tester.mq5 "Download testmode_visual_tester.mq5")(6.32 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Cookbook — Macroeconomic events database](https://www.mql5.com/en/articles/11977)
- [MQL5 Cookbook — Services](https://www.mql5.com/en/articles/11826)
- [MQL5 Cookbook – Economic Calendar](https://www.mql5.com/en/articles/9874)
- [MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://www.mql5.com/en/articles/7166)
- [MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)
- [MQL5 Cookbook - Pivot trading signals](https://www.mql5.com/en/articles/2853)
- [MQL5 Cookbook - Trading signals of moving channels](https://www.mql5.com/en/articles/1863)

**[Go to discussion](https://www.mql5.com/en/forum/40186)**

![Neural Networks Cheap and Cheerful - Link NeuroPro with MetaTrader 5](https://c.mql5.com/2/12/NeuroPro_MetaTrader4_neural_net.png)[Neural Networks Cheap and Cheerful - Link NeuroPro with MetaTrader 5](https://www.mql5.com/en/articles/830)

If specific neural network programs for trading seem expensive and complex or, on the contrary, too simple, try NeuroPro. It is free and contains the optimal set of functionalities for amateurs. This article will tell you how to use it in conjunction with MetaTrader 5.

![Random Forests Predict Trends](https://c.mql5.com/2/11/Random_Forest_MetaTrader5.png)[Random Forests Predict Trends](https://www.mql5.com/en/articles/1165)

This article considers using the Rattle package for automatic search of patterns for predicting long and short positions of currency pairs on Forex. This article can be useful both for novice and experienced traders.

![Third Generation Neural Networks: Deep Networks](https://c.mql5.com/2/12/Deep_neural_network_MetaTrader5__2.png)[Third Generation Neural Networks: Deep Networks](https://www.mql5.com/en/articles/1103)

This article is dedicated to a new and perspective direction in machine learning - deep learning or, to be precise, deep neural networks. This is a brief review of second generation neural networks, the architecture of their connections and main types, methods and rules of learning and their main disadvantages followed by the history of the third generation neural network development, their main types, peculiarities and training methods. Conducted are practical experiments on building and training a deep neural network initiated by the weights of a stacked autoencoder with real data. All the stages from selecting input data to metric derivation are discussed in detail. The last part of the article contains a software implementation of a deep neural network in an Expert Advisor with a built-in indicator based on MQL4/R.

![MQL5 Wizard: Placing Orders, Stop-Losses and Take Profits on Calculated Prices. Standard Library Extension](https://c.mql5.com/2/10/ava.png)[MQL5 Wizard: Placing Orders, Stop-Losses and Take Profits on Calculated Prices. Standard Library Extension](https://www.mql5.com/en/articles/987)

This article describes the MQL5 Standard Library extension, which allows to create Expert Advisors, place orders, Stop Losses and Take Profits using the MQL5 Wizard by the prices received from included modules. This approach does not apply any additional restrictions on the number of modules and does not cause conflicts in their joint work.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nczxsxlfjkpnsxvmsucvdbtzxzychnmm&ssn=1769186764516303831&ssn_dr=0&ssn_sr=0&fv_date=1769186764&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1246&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Programming%20EA%27s%20Modes%20Using%20Object-Oriented%20Approach%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918676417898352&fz_uniq=5070544759719663563&sv=2552)

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