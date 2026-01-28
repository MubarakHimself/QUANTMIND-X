---
title: Universal Expert Advisor: Custom Strategies and Auxiliary Trade Classes (Part 3)
url: https://www.mql5.com/en/articles/2170
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T19:45:25.599424
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/2170&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070535516950042531)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/2170#intro)
- [Logs, CMessage and CLog Classes, Singleton Pattern](https://www.mql5.com/en/articles/2170#c1)
- [Accessing Quotes Using Indexes from MetaTrader 4](https://www.mql5.com/en/articles/2170#c2)
- [Using Object-Oriented Indicators](https://www.mql5.com/en/articles/2170#c3)
- [Methods to Be Overridden by the Custom Expert Advisor](https://www.mql5.com/en/articles/2170#c4)
- [Example of Expert Advisor Trading Two Moving Averages](https://www.mql5.com/en/articles/2170#c5)
- [Example of Expert Advisor Based on Breakthrough of the BollingerBands Channel](https://www.mql5.com/en/articles/2170#c6)
- [Loading Custom Strategies to the Trading Engine](https://www.mql5.com/en/articles/2170#c7)
- [Conclusion](https://www.mql5.com/en/articles/2170#exit)

### Introduction

In this part, we continue to discuss the CStrategy trading engine. Here are the brief contents of the previous two parts. In the first article [Universal Expert Advisor: Trading Modes of Strategies](https://www.mql5.com/en/articles/2166), we have discussed in detail trading modes that allow configuring Expert Advisor logic in accordance with time and day of the week. In the seconds article [Universal Expert Advisors: the Event Model and Trading Strategy Prototype](https://www.mql5.com/en/articles/2169), we have analyzed the event model based on centralized event handling, as well as the main algorithms of the basic CStrategy class underlying custom Expert Advisors.

In the third part of the series, we will describe in detail the examples of Expert Advisors based on the CStrategy trading engine and some auxiliary algorithms that may be needed for the EA development. Particular attention is paid to the logging procedure. Despite its purely supporting role, logging is a very important element of any complex system. A good logger allows you to quickly understand the cause of the problem and find the place where it occurred. This logger is written using a special programming technique called [Singleton pattern](https://en.wikipedia.org/wiki/Singleton_pattern "https://en.wikipedia.org/wiki/Singleton_pattern"). Information about it will be interesting not only for those who organize the trading process, but also for those who create algorithms for performing non-standard tasks.

Also, the article describes the algorithms that allow you to access market data through convenient and intuitive indexes. Indeed, data access through indexes like Close\[1\] and High\[0\] is a popular feature of MetaTrader 4. So why avoid it, if it can also be used in MetaTrader 5? This article explains how to do this, and describes in detail the algorithms that implement the above idea.

To end the conclusion, I would like to use the words from my previous article. The CStrategy trading engine with all its algorithms is a rather complex set. However, full and deep understanding of its operation principle is not required. You should only understand the general principles and the functionality of the trading engine. Therefore, if any of the article parts is not clear, you can skip it. This is one of the fundamental principles of object-oriented approach: you can use a complex system without knowing its structure.

### Logs, CMessage and CLog Classes, Singleton Pattern

Logging is one of the traditional auxiliary tasks. As a rule, simple applications use the common [Print](https://www.mql5.com/en/docs/common/print) or [printf](https://www.mql5.com/en/docs/common/printformat) function that prints an error message to the MetaTrader 5 terminal:

```
...
double closes[];
if(CopyClose(Symbol(), Period(), 0, 100, closes) < 100)
   printf("Not enough data.");
...
```

However, this simple approach is not always sufficient to understand what is happening in large complex programs that contain hundreds of lines of source code. Therefore, the best solution for such tasks is to develop a special logging module — the **CLog** class.

The most obvious method of the logger is AddMessage(). For example, if _Log_ is an object of our CLog, we can write the following construction:

```
Log.AddMessage("Warning! The number of received bars is less than is required");
```

However, the warning sent contains much less useful information than is necessary for debugging. How can you know from this message when it was created? What function has created it? How can you know what important information is contained in it? To avoid this, we need to expand the notion of the message. In addition to text, each message should contain the following attributes:

- creation time
- message source
- message type (information, warning, error message)

It would also be useful, if our message contained some additional details:

- system error ID
- trading error ID (if a trading action was performed)
- trade server time as of the moment of message creation

All this information can be conveniently combined in the special **CMessage** class. Since our message is a class, you can easily add more data and methods to work with logs. Here is the class header:

```
//+------------------------------------------------------------------+
//|                                                         Logs.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "https://www.mql5.com"
#include <Object.mqh>
#include <Arrays\ArrayObj.mqh>

#define UNKNOW_SOURCE "unknown"     // An unknown source of messages
//+------------------------------------------------------------------+
//| Message type                                                     |
//+------------------------------------------------------------------+
enum ENUM_MESSAGE_TYPE
  {
   MESSAGE_INFO,                    // Information message
   MESSAGE_WARNING,                 // Warning message
   MESSAGE_ERROR                    // Error message
  };
//+------------------------------------------------------------------+
//| A message passed to the logger class                             |
//+------------------------------------------------------------------+
class CMessage : public CObject
  {
private:
   ENUM_MESSAGE_TYPE m_type;               // Message type
   string            m_source;             // Message source
   string            m_text;               // Message text
   int               m_system_error_id;    // Creates an ID of a SYSTEM error
   int               m_retcode;            // Contains a trade server return code
   datetime          m_server_time;        // Trade server time at the moment of message creation
   datetime          m_local_time;         // Local time at the moment of message creation
   void              Init(ENUM_MESSAGE_TYPE type,string source,string text);
public:
                     CMessage(void);
                     CMessage(ENUM_MESSAGE_TYPE type);
                     CMessage(ENUM_MESSAGE_TYPE type,string source,string text);
   void              Type(ENUM_MESSAGE_TYPE type);
   ENUM_MESSAGE_TYPE Type(void);
   void              Source(string source);
   string            Source(void);
   void              Text(string text);
   string            Text(void);
   datetime          TimeServer(void);
   datetime          TimeLocal();
   void              SystemErrorID(int error);
   int               SystemErrorID();
   void              Retcode(int retcode);
   int               Retcode(void);
   string            ToConsoleType(void);
   string            ToCSVType(void);
  };
```

First of all the header contains ENUM\_MESSAGE\_TYPE. It defines the type of the message being created. The message can be informational (MESSAGE\_INFO), warning (MESSAGE\_WARNING) and notifying of an error (MESSAGE\_ERROR).

The class consists of various Get/Set methods that set or read various attributes of a message. For easy message creation in one line, CMessage provides a corresponding overloaded constructor that should be called with the parameters defining the text of the message, its type and source. For example, if we need to create a warning message in the OnTick function notifying of a too small amount of loaded data, this can be done the following way:

```
void OnTick(void)
  {
   double closes[];
   if(CopyClose(Symbol(),Period(),0,100,closes)<100)
      CMessage message=new CMessage(MESSAGE_WARNING,__FUNCTION__,"Not enough data");
  }
```

This message contains more information than the previous one. In addition to the message itself, it contains the name of the function that called it, and the message type. Moreover, our message has data that you do not have to fill during creation. For example the _message_ object contains the message creation time and the current code of the trading error if there is any.

Now it is time to consider the **CLog** logger. The class serves as a storage of CMessage messages. One of its most interesting functions is sending push notifications to mobile terminals using the [SendNotification](https://www.mql5.com/en/docs/network/sendnotification) function. This is an extremely useful feature when constant monitoring of Expert Advisor operation is impossible. Instead, we can send pushes to notify the user that something has gone wrong.

The specific feature of logging is that it must be a single process for all parts of the program. It would be strange if each function or class had its own logging mechanism. Therefore the CLog class is implemented using a special programming pattern called **Singleton**. This pattern ensures that there is only one copy of an object of a certain type. For example, if the program uses two pointers, each of which refers to the object of the CLog type, the pointers will point to the same object. An object is actually created and deleted behind the scenes, in private methods of the class.

Let us consider the title of this class and the methods that implement the Singleton pattern:

```
//+------------------------------------------------------------------+
//|                                                         Logs.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "https://www.mql5.com"
#include <Object.mqh>
#include <Arrays\ArrayObj.mqh>
#include "Message.mqh"

//+------------------------------------------------------------------+
//| The class implements logging of messages as Singleton            |
//+------------------------------------------------------------------+
class CLog
{
private:
   static CLog*      m_log;                     // A pointer to the global static sample
   CArrayObj         m_messages;                // The list of saved messages
   bool              m_terminal_enable;         // True if you need to print the received message to the trading terminal
   bool              m_push_enable;             // If true, sends push notifications
   ENUM_MESSAGE_TYPE m_push_priority;           // Contains the specified priority of message display in the terminal window
   ENUM_MESSAGE_TYPE m_terminal_priority;       // Contains the specified priority of sending pushes to mobile devices
   bool              m_recursive;               // A flag indicating the recursive call of the destructor
   bool              SendPush(CMessage* msg);
   void              CheckMessage(CMessage* msg);
                     CLog(void);                // Private constructor
   string            GetName(void);
   void              DeleteOldLogs(int day_history = 30);
   void              DeleteOldLog(string file_name, int day_history);
                     ~CLog(void){;}
public:
   static CLog*      GetLog(void);              // The method to receive a static object
   bool              AddMessage(CMessage* msg);
   void              Clear(void);
   bool              Save(string path);
   CMessage*         MessageAt(int index)const;
   int               Total(void);
   void              TerminalEnable(bool enable);
   bool              TerminalEnable(void);
   void              PushEnable(bool enable);
   bool              PushEnable(void);
   void              PushPriority(ENUM_MESSAGE_TYPE type);
   ENUM_MESSAGE_TYPE PushPriority(void);
   void              TerminalPriority(ENUM_MESSAGE_TYPE type);
   ENUM_MESSAGE_TYPE TerminalPriority(void);
   bool              SaveToFile(void);
   static bool       DeleteLog(void);
};
CLog* CLog::m_log;
```

The CLog class stores a pointer to the static object of itself as a private member. It may seem a strange programming construct, but it makes sense. The only class constructor is private and cannot be called. Instead of calling the constructor, the GetLog method can be used:

```
//+------------------------------------------------------------------+
//| Returns the logger object                                        |
//+------------------------------------------------------------------+
static CLog* CLog::GetLog()
{
   if(CheckPointer(m_log) == POINTER_INVALID)
      m_log = new CLog();
   return m_log;
}
```

It checks whether the static pointer points to an existing CLog object, and if so, returns a reference to it. Otherwise, it creates a new object and associates the internal _m\_log_ pointer with it. This means that the object is only created once. During further calls of the GetLog method, a previously created object will be returned.

Deletion of the object is also performed only once. This is done by using the DeleteLog method:

```
//+------------------------------------------------------------------+
//| Deletes the logger object                                        |
//+------------------------------------------------------------------+
bool CLog::DeleteLog(void)
{
   bool res = CheckPointer(m_log) != POINTER_INVALID;
   if(res)
      delete m_log;
   return res;
}
```

If the _m\_log_ exists, it will be deleted and true will be returned.

It may seem that the described logging system is complex, however its supported features are quite impressive. For example, you can rank messages by type, or send them as push notifications. A user is ultimately to decide whether to use this system. It is implemented in separate modules Message.mqh and Logs.mqh, so you can use it either separately from the described project or inside it.

### Accessing Quotes Using Indexes from MetaTrader 4

One of the major changes in MetaTrader 5 as compared to its predecessor, is a model of accessing quotes and indicator data. For example, if you needed to find out the close price of the current bar, you could do it in MetaTrader 4 by adding the following procedure:

```
double close = Close[0];
```

It means, you could access data almost directly through the indexing of an appropriate timeseries. More actions are required for finding our the close price of the current bar in MetaTrader 5:

1. Define a receiver array to copy the required amount of quotes to.
2. Copy the required quotes using one of the functions of the Copy\* group (functions for accessing timeseries and indicator data).
3. Refer to the required index of the copied array.

To find our the close price of the current bar in MetaTrader 5, the following actions are required:

```
double closes[];
double close = 0.0;
if(CopyClose(Symbol(), Period(), 0, 1, closes))
   close = closes[0];
else
   printf("Failed to copy the close price.");
```

This access to data is more difficult than in MetaTrader 4. However, this approach makes data access universal: the same unified interface and mechanisms are used to access data received from different symbols and indicators.

Although this is not required in everyday tasks as a rule. Often we only need to obtain the last value of the current symbol. This can be the Open or Close price of a bar, as well as its High or Low. Anyway it would be convenient to use the data access model adopted in MetaTrader 4. Due to the object-oriented nature of MQL5, it is possible to create special classes with an indexer that can be used the same way to access trade data, as is used in MetaTrader 4. For example, to be able to obtain the Close price in MetaTrader 5 this way:

```
double close = Close[0];
```

we should add the following wrapper for close prices:

```
//+------------------------------------------------------------------+
//| Access to Close prices of the symbol bar.                        |
//+------------------------------------------------------------------+
class CClose : public CSeries
  {
public:
   double operator[](int index)
     {
      double value[];
      if(CopyClose(m_symbol, m_timeframe, index, 1, value) == 0)return 0.0;
      return value[0];
     }
  };
```

The same code must be written for the other series, including time, volume, as well as open, high and low prices. Of course, in some cases the code may run much slower than one-time copying of the required array of quotes using the [Copy\*](https://www.mql5.com/en/docs/series) system functions. However, as mentioned above, we often need to only access the last element, because all the previous elements are taken into account in the moving window.

This simple set of classes is included into **Series.mqh**. It provides a convenient interface for accessing quotes like in MetaTrader 4 and is actively used in the trading engine.

A distinctive feature of these classes is their platform independence. For example, in MetaTrader 5 an Expert Advisor can call the method of one of these classes "thinking" that it refers to the quotes directly. This access method will also work in MetaTrader 4, but instead of the specialized wrapper it will directly access system series, such as Open, High, Low or Close.

### Using Object-Oriented Indicators

Almost every indicator has a number of settings for configuration. Working with indicators in MetaTrader 5 is similar to working with quotes, with the only difference being the necessity to create the so called _indicator handle_, i.e. a special pointer to some internal MetaTrader 5 object containing calculation values, before copying the indicator data. Parameters of an indicator are set at the time of handle creation. If you need to edit one of the indicator parameters for some reason, you should delete the old indicator handle and create a new one with updated parameters. The indicator parameters should be stored in an external place, for example, in Expert Advisor variables.

As a result, most of operations with indicators are passed to the Expert Advisor. It is not always convenient. Let us consider a simple example: an Expert Advisor trades using the signals of MA crossing. Despite its simplicity, the Moving Average indicator has six parameters to be set:

1. The symbol to calculate the Moving Average on
2. The chart timeframe or period
3. The averaging period
4. The Moving Average type (simple, exponential, weighted, etc.)
5. Indicator shift from the price bar
6. Applied price (one of the OHLC prices of a bar or a calculation buffer of another indicator)

Therefore, if we want to write an Expert Advisor trading the intersection of two moving averages that use a complete list of the MA settings, it will have to contain twelve parameters — six parameters for the fast moving average and six more for the slow one. Moreover, if the user changes the timeframe or symbol of the chart the EA is running on, the handles of used indicators will also need to be reinitialized.

To free the Expert Advisor from tasks related to indicator handles, we should use the object-oriented versions of indicators. By using the object-oriented indicator classes, we can write constructs like:

```
CMovingAverageExp MAExpert;     // Creating EA that trades based on two moving averages.
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Configuring the fast Moving Average of the Expert Advisor
   MAExpert.FastMA.Symbol("EURUSD");
   MAExpert.FastMA.Symbol(PERIOD_M10);
   MAExpert.FastMA.Period(13);
   MAExpert.FastMA.AppliedPrice(PRICE_CLOSE);
   MAExpert.FastMA.MaShift(1);
//--- Configuring the slow Moving Average of the Expert Advisor
   MAExpert.SlowMA.Symbol("EURUSD");
   MAExpert.SlowMA.Symbol(PERIOD_M15);
   MAExpert.SlowMA.Period(15);
   MAExpert.SlowMA.AppliedPrice(PRICE_CLOSE);
   MAExpert.SlowMA.MaShift(1);

   return(INIT_SUCCEEDED);
  }
```

The end user is only to set the parameters of the indicators used by the Expert Advisor. The Expert Advisor will only read data from them.

There is another important advantage of using indicator objects. Object-oriented indicators hide their implementation. This means that they can calculate their values on their own or by calling appropriate handles. In cases where multiple calculation indicators are used and high execution speed is required, it is advisable to add the indicator calculation unit directly into the Expert Advisor. Thanks to the object-oriented approach, this can be done without rewriting the Expert Advisor. You only need to calculate the indicator values inside the appropriate class without using handles.

To illustrate the above, below is the source code of the CIndMovingAverage class that is based on the system [iMA](https://www.mql5.com/en/docs/indicators/ima) indicator:

```
//+------------------------------------------------------------------+
//|                                                MovingAverage.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "https://www.mql5.com"
#include <Strategy\Message.mqh>
#include <Strategy\Logs.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CIndMovingAverage
  {
private:
   int               m_ma_handle;         // Indicator handle
   ENUM_TIMEFRAMES   m_timeframe;         // Timeframe
   int               m_ma_period;         // Period
   int               m_ma_shift;          // Shift
   string            m_symbol;            // Symbol
   ENUM_MA_METHOD    m_ma_method;         // Moving Average method
   uint              m_applied_price;     // The handle of the indicator, on which you want to calculate the Moving Average value,
                                          // or one of price values of ENUM_APPLIED_PRICE
   CLog*             m_log;               // Logging
   void              Init(void);
public:
                     CIndMovingAverage(void);

/*Params*/
   void              Timeframe(ENUM_TIMEFRAMES timeframe);
   void              MaPeriod(int ma_period);
   void              MaShift(int ma_shift);
   void              MaMethod(ENUM_MA_METHOD method);
   void              AppliedPrice(int source);
   void              Symbol(string symbol);

   ENUM_TIMEFRAMES   Timeframe(void);
   int               MaPeriod(void);
   int               MaShift(void);
   ENUM_MA_METHOD    MaMethod(void);
   uint              AppliedPrice(void);
   string            Symbol(void);

/*Out values*/
   double            OutValue(int index);
  };
//+------------------------------------------------------------------+
//| Default constructor.                                             |
//+------------------------------------------------------------------+
CIndMovingAverage::CIndMovingAverage(void) : m_ma_handle(INVALID_HANDLE),
                                             m_timeframe(PERIOD_CURRENT),
                                             m_ma_period(12),
                                             m_ma_shift(0),
                                             m_ma_method(MODE_SMA),
                                             m_applied_price(PRICE_CLOSE)
  {
   m_log=CLog::GetLog();
  }
//+------------------------------------------------------------------+
//| Initialization.                                                  |
//+------------------------------------------------------------------+
CIndMovingAverage::Init(void)
  {
   if(m_ma_handle!=INVALID_HANDLE)
     {
      bool res=IndicatorRelease(m_ma_handle);
      if(!res)
        {
         string text="Realise iMA indicator failed. Error ID: "+(string)GetLastError();
         CMessage *msg=new CMessage(MESSAGE_WARNING,__FUNCTION__,text);
         m_log.AddMessage(msg);
        }
     }
   m_ma_handle=iMA(m_symbol,m_timeframe,m_ma_period,m_ma_shift,m_ma_method,m_applied_price);
   if(m_ma_handle==INVALID_HANDLE)
     {
      string params="(Period:"+(string)m_ma_period+", Shift: "+(string)m_ma_shift+
                    ", MA Method:"+EnumToString(m_ma_method)+")";
      string text="Create iMA indicator failed"+params+". Error ID: "+(string)GetLastError();
      CMessage *msg=new CMessage(MESSAGE_ERROR,__FUNCTION__,text);
      m_log.AddMessage(msg);
     }
  }
//+------------------------------------------------------------------+
//| Setting the timeframe.                                           |
//+------------------------------------------------------------------+
void CIndMovingAverage::Timeframe(ENUM_TIMEFRAMES tf)
  {
   m_timeframe=tf;
   if(m_ma_handle!=INVALID_HANDLE)
      Init();
  }
//+------------------------------------------------------------------+
//| Returns the current timeframe.                                   |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES CIndMovingAverage::Timeframe(void)
  {
   return m_timeframe;
  }
//+------------------------------------------------------------------+
//| Sets the Moving Average averaging period.                        |
//+------------------------------------------------------------------+
void CIndMovingAverage::MaPeriod(int ma_period)
  {
   m_ma_period=ma_period;
   if(m_ma_handle!=INVALID_HANDLE)
      Init();
  }
//+------------------------------------------------------------------+
//| Returns the current averaging period of the Moving Average.      |
//+------------------------------------------------------------------+
int CIndMovingAverage::MaPeriod(void)
  {
   return m_ma_period;
  }
//+------------------------------------------------------------------+
//| Sets the Moving Average type.                                    |
//+------------------------------------------------------------------+
void CIndMovingAverage::MaMethod(ENUM_MA_METHOD method)
  {
   m_ma_method=method;
   if(m_ma_handle!=INVALID_HANDLE)
      Init();
  }
//+------------------------------------------------------------------+
//| Returns the Moving Average type.                                 |
//+------------------------------------------------------------------+
ENUM_MA_METHOD CIndMovingAverage::MaMethod(void)
  {
   return m_ma_method;
  }
//+------------------------------------------------------------------+
//| Returns the Moving Average shift.                                |
//+------------------------------------------------------------------+
int CIndMovingAverage::MaShift(void)
  {
   return m_ma_shift;
  }
//+------------------------------------------------------------------+
//| Sets the Moving Average shift.                                   |
//+------------------------------------------------------------------+
void CIndMovingAverage::MaShift(int ma_shift)
  {
   m_ma_shift=ma_shift;
   if(m_ma_handle!=INVALID_HANDLE)
      Init();
  }
//+------------------------------------------------------------------+
//| Sets the type of the price used for MA calculation.              |
//+------------------------------------------------------------------+
void CIndMovingAverage::AppliedPrice(int price)
  {
   m_applied_price = price;
   if(m_ma_handle != INVALID_HANDLE)
      Init();
  }
//+------------------------------------------------------------------+
//| Returns the type of the price used for MA calculation.           |
//+------------------------------------------------------------------+
uint CIndMovingAverage::AppliedPrice(void)
  {
   return m_applied_price;
  }
//+------------------------------------------------------------------+
//| Sets the symbol to calculate the indicator for                   |
//+------------------------------------------------------------------+
void CIndMovingAverage::Symbol(string symbol)
  {
   m_symbol=symbol;
   if(m_ma_handle!=INVALID_HANDLE)
      Init();
  }
//+------------------------------------------------------------------+
//| Returns the symbol the indicator is calculated for               |
//+------------------------------------------------------------------+
string CIndMovingAverage::Symbol(void)
  {
   return m_symbol;
  }
//+------------------------------------------------------------------+
//| Returns the value of the MA with index 'index'                   |
//+------------------------------------------------------------------+
double CIndMovingAverage::OutValue(int index)
  {
   if(m_ma_handle==INVALID_HANDLE)
      Init();
   double values[];
   if(CopyBuffer(m_ma_handle,0,index,1,values))
      return values[0];
   return EMPTY_VALUE;
  }
```

The class is quite simple. Its main task is to re-initialize the indicator if one of its parameters changes, as well as to return the calculated value by _index_. A handle is re-initialized using the Init method, and the required value is returned using OutValue. Methods that return one of the indicator values start with the _**Out**_ prefix. This facilitates search for the required method when programming in editors that offer intellectual substitution of parameters, such as MetaEditor.

The trading engine package includes a number of object-oriented indicators. This will help you to understand how they work and create your own object-oriented versions of classical indicators. The section on developing custom Expert Advisors illustrates the principles of working with them.

### Methods to Be Overridden by the Custom Expert Advisor

In the first article [Universal Expert Advisor: Trading Modes of Strategies (Part 1)](https://www.mql5.com/en/articles/2166), we have considered in detail trading modes of a strategy and its main methods that need to be overridden. Now, it is time to do some practicing.

Each Expert Advisor created using the CStrategy engine must override virtual methods that are responsible for some properties and behavior of the Expert Advisor. We will list all the methods to be overridden as a table of three columns. The first one contains the name of the virtual method, the second one shows the event or action to be tracked or performed. The third column contains the description of the purpose of using the method. So here is the table:

| Virtual method | Event/Action | Purpose |
| --- | --- | --- |
| OnSymbolChanged | Called when the name of a trading symbol changes | When you change the trading instrument, the EA's indicators should be re-initialized. The event allows performing re-initialization of indicators of the Expert Advisor. |
| OnTimeframeChanged | Change of the working timframe | When you change the working timeframe, the EA's indicators should be re-initialized. The event allows performing re-initialization of indicators of the Expert Advisor. |
| ParseXmlParams | Parsing custom parameters of the strategy loaded via n XML file | The strategy should recognize XML parameters passed to the method and configure its settings accordingly. |
| ExpertNameFull | Returns the full name of the Expert Advisor | The full name of the Expert Advisor consists of the strategy name and, as a rule, a unique set of the strategy parameters. A strategy instance must independently determine its full name. This name is also used in the visual panel, in the drop-down Agent list. |
| OnTradeTransaction | Occurs in case of a trade event | Some strategies need to analyze trade events for proper operation. This event allows passing a trade event to the Expert Advisor and analyze it. |
| **InitBuy** | Initiates a Buy operation | One of the basic methods that must be overridden. In this method you should execute a Buy operation if suitable trading conditions have formed. |
| **InitSell** | Initiates a Sell operation | One of the basic methods that must be overridden. In this method you should execute a Sell operation if suitable trading conditions have formed. |
| **SupportBuy** | Manages a previously opened long position | An open long position needs to be managed. For example, you should set Stop Loss or close the position at a position exit signal. All these steps must be performed in this method. |
| **SupportSell** | Manages a previously opened short position | An open short position needs to be managed. For example, you should set Stop Loss or close the position at a position exit signal. All these steps must be performed in this method. |

Table 1. Virtual methods and their purposes

The most important methods that you must override are **InitBuy**, **InitSell**, **SupportBuy**, and **SupportSell**. They are shown in bold in the table. If you forget to override, for example, InitBuy, the custom strategy will not buy. If you do not override one of the Support methods, an open position can be left open forever. Therefore, when creating an Expert Advisor, be careful overriding these methods.

If you want the trade engine to automatically load a strategy from an XML file and configure its parameters in accordance with the settings provided in the file, you will also need to override the ParseXmlParams method. In this method, a strategy should determine the parameters that are passed to it, and understand how to change its own settings according to these parameters. Working with XML parameters will be described in more detail in the fourth part of the series: **Universal Expert Advisor: Trading in a Group and Managing a Portfolio of Strategies (Part 4)**". An example of ParseXmlParams overriding is included into the listing of strategies based on the Bollinger Bands.

### Example of Expert Advisor Trading Two Moving Averages

Now it is time to create our first Expert Advisor using the possibilities of CStrategy. To make the source code simple and compact, we will not use the logging function in it. Let us briefly describe the actions that need to be performed in our Expert Advisor:

- When switching the timeframe and symbol, change the settings of the fast and slow moving averages by overriding the OnSymbolChanged and OnTimeframeChanged methods.
- Override methods InitBuy, InitSell, SupportBuy and SupportSell. Define the EA's trading logic in these methods (position opening and managing rules).

The rest of the EA's work should be performed by the trading engine and indicators used by the EA. Here is the source code of the Expert Advisor:

```
//+------------------------------------------------------------------+
//|                                                      Samples.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "https://www.mql5.com"
#include <Strategy\Strategy.mqh>
#include <Strategy\Indicators\MovingAverage.mqh>
//+------------------------------------------------------------------+
//| An example of a classical strategy based on two Moving Averages. |
//| If the fast MA crosses the slow one from upside down             |
//| we buy, if from top down - we sell.                              |
//+------------------------------------------------------------------+
class CMovingAverage : public CStrategy
  {
private:
   bool              IsTrackEvents(const MarketEvent &event);
protected:
   virtual void      InitBuy(const MarketEvent &event);
   virtual void      InitSell(const MarketEvent &event);
   virtual void      SupportBuy(const MarketEvent &event,CPosition *pos);
   virtual void      SupportSell(const MarketEvent &event,CPosition *pos);
   virtual void      OnSymbolChanged(string new_symbol);
   virtual void      OnTimeframeChanged(ENUM_TIMEFRAMES new_tf);
public:
   CIndMovingAverage FastMA;        // Fast moving average
   CIndMovingAverage SlowMA;        // Slow moving average
                     CMovingAverage(void);
   virtual string    ExpertNameFull(void);
  };
//+------------------------------------------------------------------+
//| Initialization.                                                  |
//+------------------------------------------------------------------+
CMovingAverage::CMovingAverage(void)
  {
  }
//+------------------------------------------------------------------+
//| Respond to the symbol change                                     |
//+------------------------------------------------------------------+
void CMovingAverage::OnSymbolChanged(string new_symbol)
  {
   FastMA.Symbol(new_symbol);
   SlowMA.Symbol(new_symbol);
  }
//+------------------------------------------------------------------+
//| Respond to the timeframe change                                  |
//+------------------------------------------------------------------+
void CMovingAverage::OnTimeframeChanged(ENUM_TIMEFRAMES new_tf)
  {
   FastMA.Timeframe(new_tf);
   SlowMA.Timeframe(new_tf);
  }
//+------------------------------------------------------------------+
//| We buy when the fast MA is above the slow one.                   |
//+------------------------------------------------------------------+
void CMovingAverage::InitBuy(const MarketEvent &event)
  {
   if(!IsTrackEvents(event))return;                      // Handling only the required event!
   if(positions.open_buy > 0) return;                    // If there is at least one open position, no need to buy, as we have already bought!
   if(FastMA.OutValue(1) > SlowMA.OutValue(1))           // If no open buy positions, check if the fast MA is above the slow one:
      Trade.Buy(MM.GetLotFixed(), ExpertSymbol(), "");   // If above, buy.
  }
//+------------------------------------------------------------------+
//| Close the long position when the fast MA is below the            |
//| slow one.                                                        |
//+------------------------------------------------------------------+
void CMovingAverage::SupportBuy(const MarketEvent &event,CPosition *pos)
  {
   if(!IsTrackEvents(event))return;                      // Handling only the required event!
   if(FastMA.OutValue(1) < SlowMA.OutValue(1))           // If the fast MA is below the slow one -
      pos.CloseAtMarket("Exit by cross over");           // Close the position.
  }
//+------------------------------------------------------------------+
//| We buy when the fast MA is above the slow one.                   |
//+------------------------------------------------------------------+
void CMovingAverage::InitSell(const MarketEvent &event)
  {
   if(!IsTrackEvents(event))return;                      // Handling only the required event!
   if(positions.open_sell > 0) return;                   // If there is at least one short position, no need to sell, as we have already sold!
   if(FastMA.OutValue(1) < SlowMA.OutValue(1))           // If no open buy positions, check if the fast MA is above the slow one:
      Trade.Sell(1.0, ExpertSymbol(), "");               // If above that, we buy.
  }
//+------------------------------------------------------------------+
//| Close the short position when the fast MA is above the           |
//| the slow one.                                                    |
//+------------------------------------------------------------------+
void CMovingAverage::SupportSell(const MarketEvent &event,CPosition *pos)
  {
   if(!IsTrackEvents(event))return;                      // Handling only the required event!
   if(FastMA.OutValue(1) > SlowMA.OutValue(1))           // If the fast MA is above the slow one -
      pos.CloseAtMarket("Exit by cross under");          // Close the position.
  }
//+------------------------------------------------------------------+
//| Filters incoming events. If the passed event is not              |
//| processed by the strategy, returns false; if it is processed     |
//| returns true.                                                    |
//+------------------------------------------------------------------+
bool CMovingAverage::IsTrackEvents(const MarketEvent &event)
  {
//--- We handle only opening of a new bar on the working symbol and timeframe
   if(event.type != MARKET_EVENT_BAR_OPEN)return false;
   if(event.period != Timeframe())return false;
   if(event.symbol != ExpertSymbol())return false;
   return true;
  }
```

The above source code is easy to understand. However, we need to clarify some points. The CStrategy engine calls methods InitBuy, InitSell, SupportBuy and SuportSell (trade logic methods) upon the emergence of any event, such as depth of market changes, arrival of a new tick or timer change. Typically, these methods are called very frequently. However, an Expert Advisor uses a very limited set of events. This one only uses the event of _formation of a new bar_. Therefore, all other events that call the trade logic methods should be ignored. The IsTrackEvents method is used for that. It checks whether the event passed to it is being tracked, and if so — returns true, otherwise returns false.

The **positions** structure is used as an auxiliary variable. It contains the number of long and short positions, which belong to the current strategy. The CStrategy engine calculates the statistics, so the strategy does not need to go through all open positions in order to count them. The Expert Advisor's position opening logic is actually reduced to the verification of the following conditions:

1. A trade event is the opening of a new bar.
2. There is no other open position in the same direction.
3. The fast moving average is above (to buy) or below (to sell) the slow moving average.

The conditions that must be met to close a position are even simpler:

1. A trade event is the opening of a new bar.
2. The fast moving average is below (to close a long position) or above (to close a short position) the slow moving average.

In this case, there is no need to check open positions, because the call of SupportBuy and SupportSell with the current position as a parameter indicates that the EA's position exists and is passed to it.

The actual logic of the Expert Advisor, without taking into account the definitions of methods and its class, is described in 18 lines of code. Moreover, half of these lines (conditions for Sell) is a mirror of the other half (conditions for Buy). This simplification of logic is only possible when using auxiliary libraries like CStrategy.

### Example of Expert Advisor Based on Breakthrough of the BollingerBands Channel

We continue creating strategies using the CStrategy trading engine. In the second example we will create a strategy that trades breakouts of the Bollinger Bands channel. If the current price is above the upper Bollinger band, we will buy. Conversely, if the current bar close price is below the lower Bollinger band, we will sell. We will exit long and short positions once the price reaches the middle line of the indicator.

This time we will use the standard [iBands](https://www.mql5.com/en/docs/indicators/ibands) indicator handle. This is done to show that our trading model allows working with indicator handles directly, i.e. building special object-oriented indicator classes is not required. However, in this case, we will need to specify two main parameters of the indicator — its averaging period and standard deviation value — straight in the Expert Advisor. Here is the source code of the strategy:

```
//+------------------------------------------------------------------+
//|                                                ChannelSample.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "https://www.mql5.com"
#include <Strategy\Strategy.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CChannel : public CStrategy
  {
private:
   int               m_handle;   // The handle of the indicator that we will use
   int               m_period;   // Bollinger period
   double            m_std_dev;  // Standard deviation value
   bool              IsTrackEvents(const MarketEvent &event);
protected:
   virtual void      OnSymbolChanged(string new_symbol);
   virtual void      OnTimeframeChanged(ENUM_TIMEFRAMES new_tf);
   virtual void      InitBuy(const MarketEvent &event);
   virtual void      SupportBuy(const MarketEvent &event,CPosition *pos);
   virtual void      InitSell(const MarketEvent &event);
   virtual void      SupportSell(const MarketEvent &event,CPosition *pos);
   virtual bool      ParseXmlParams(CXmlElement *params);
   virtual string    ExpertNameFull(void);
public:
                     CChannel(void);
                    ~CChannel(void);
   int               PeriodBands(void);
   void              PeriodBands(int period);
   double            StdDev(void);
   void              StdDev(double std);
  };
//+------------------------------------------------------------------+
//| Default constructor                                              |
//+------------------------------------------------------------------+
CChannel::CChannel(void) : m_handle(INVALID_HANDLE)
  {
  }
//+------------------------------------------------------------------+
//| The destructor frees the used handle of the indicator            |
//+------------------------------------------------------------------+
CChannel::~CChannel(void)
  {
   if(m_handle!=INVALID_HANDLE)
      IndicatorRelease(m_handle);
  }
//+------------------------------------------------------------------+
//| Respond to the symbol change                                     |
//+------------------------------------------------------------------+
void CChannel::OnSymbolChanged(string new_symbol)
  {
   if(m_handle!=INVALID_HANDLE)
      IndicatorRelease(m_handle);
   m_handle=iBands(new_symbol,Timeframe(),m_period,0,m_std_dev,PRICE_CLOSE);
  }
//+------------------------------------------------------------------+
//| Respond to the timeframe change                                  |
//+------------------------------------------------------------------+
void CChannel::OnTimeframeChanged(ENUM_TIMEFRAMES new_tf)
  {
   if(m_handle!=INVALID_HANDLE)
      IndicatorRelease(m_handle);
   m_handle=iBands(ExpertSymbol(),Timeframe(),m_period,0,m_std_dev,PRICE_CLOSE);
  }
//+------------------------------------------------------------------+
//| Returns indicator period                                         |
//+------------------------------------------------------------------+
int CChannel::PeriodBands(void)
  {
   return m_period;
  }
//+------------------------------------------------------------------+
//| Sets indicator period                                            |
//+------------------------------------------------------------------+
void CChannel::PeriodBands(int period)
  {
   if(m_period == period)return;
   m_period=period;
   if(m_handle!=INVALID_HANDLE)
      IndicatorRelease(m_handle);
   m_handle=iBands(ExpertSymbol(),Timeframe(),m_period,0,m_std_dev,PRICE_CLOSE);
  }
//+------------------------------------------------------------------+
//| Sets the standard deviation value                                |
//+------------------------------------------------------------------+
double CChannel::StdDev(void)
  {
   return m_std_dev;
  }
//+------------------------------------------------------------------+
//| Sets the standard deviation value                                |
//+------------------------------------------------------------------+
void CChannel::StdDev(double std)
  {
   if(m_std_dev == std)return;
   m_std_dev=std;
   if(m_handle!=INVALID_HANDLE)
      IndicatorRelease(m_handle);
   m_handle=iBands(ExpertSymbol(),Timeframe(),m_period,0,m_std_dev,PRICE_CLOSE);
  }
//+------------------------------------------------------------------+
//| Long position opening rules                                      |
//+------------------------------------------------------------------+
void CChannel::InitBuy(const MarketEvent &event)
  {
   if(IsTrackEvents(event))return;                    // Enable logic only at the opening of a new bar
   if(positions.open_buy > 0)return;                  // Does not open more than one long position
   double bands[];
   if(CopyBuffer(m_handle, UPPER_BAND, 1, 1, bands) == 0)return;
   if(Close[1]>bands[0])
      Trade.Buy(1.0,ExpertSymbol());
  }
//+------------------------------------------------------------------+
//| Long position closing rules                                      |
//+------------------------------------------------------------------+
void CChannel::SupportBuy(const MarketEvent &event,CPosition *pos)
  {
   if(IsTrackEvents(event))return;                    // Enable logic only at the opening of a new bar
   double bands[];
   if(CopyBuffer(m_handle, BASE_LINE, 1, 1, bands) == 0)return;
   double b = bands[0];
   double s = Close[1];
   if(Close[1]<bands[0])
      pos.CloseAtMarket();
  }
//+------------------------------------------------------------------+
//| Long position opening rules                                      |
//+------------------------------------------------------------------+
void CChannel::InitSell(const MarketEvent &event)
  {
   if(IsTrackEvents(event))return;                    // Enable logic only at the opening of a new bar
   if(positions.open_sell > 0)return;                 // Does not open more than one long position
   double bands[];
   if(CopyBuffer(m_handle, LOWER_BAND, 1, 1, bands) == 0)return;
   if(Close[1]<bands[0])
      Trade.Sell(1.0,ExpertSymbol());
  }
//+------------------------------------------------------------------+
//| Long position closing rules                                      |
//+------------------------------------------------------------------+
void CChannel::SupportSell(const MarketEvent &event,CPosition *pos)
  {
   if(IsTrackEvents(event))return;     // Enable logic only at the opening of a new bar
   double bands[];
   if(CopyBuffer(m_handle, BASE_LINE, 1, 1, bands) == 0)return;
   double b = bands[0];
   double s = Close[1];
   if(Close[1]>bands[0])
      pos.CloseAtMarket();
  }
//+------------------------------------------------------------------+
//| Filters incoming events. If the passed event is not              |
//| processed by the strategy, returns false; if it is processed     |
//| returns true.                                                    |
//+------------------------------------------------------------------+
bool CChannel::IsTrackEvents(const MarketEvent &event)
  {
//--- We handle only opening of a new bar on the working symbol and timeframe
   if(event.type != MARKET_EVENT_BAR_OPEN)return false;
   if(event.period != Timeframe())return false;
   if(event.symbol != ExpertSymbol())return false;
   return true;
  }
//+------------------------------------------------------------------+
//| The strategy's specific parameters are parsed inside it in       |
//| this method overridden from CStrategy                            |
//+------------------------------------------------------------------+
bool CChannel::ParseXmlParams(CXmlElement *params)
  {
   bool res=true;
   for(int i=0; i<params.GetChildCount(); i++)
     {
      CXmlElement *param=params.GetChild(i);
      string name=param.GetName();
      if(name=="Period")
         PeriodBands((int)param.GetText());
      else if(name=="StdDev")
         StdDev(StringToDouble(param.GetText()));
      else
         res=false;
     }
   return res;
  }
//+------------------------------------------------------------------+
//| The full unique name of the Expert Advisor                       |
//+------------------------------------------------------------------+
string CChannel::ExpertNameFull(void)
  {
   string name=ExpertName();
   name += "[" + ExpertSymbol();\
   name += "-" + StringSubstr(EnumToString(Timeframe()), 7);\
   name += "-" + (string)Period();\
   name += "-" + DoubleToString(StdDev(), 1);\
   name += "]";
   return name;
  }
```

Now the Expert Advisor performs more operations. The EA contains Bollinger averaging parameters and its standard deviation value. Also the Expert Advisor creates indicator handles and destroys them in appropriate methods. This is due to the direct use of indicators without using wrappers. The rest of the code is similar to the previous Expert Advisor. It waits for the close price of the last bar to be above (to buy) or below (to sell) Bollinger bands, and opens a new position.

Note that in the Expert Advisor we use direct access to bars through special classes of timeseries. For example, this method is used to compare the last known bar close price with the upper Bollinger Band in the Buy section (the InitBuy method):

```
double bands[];
if(CopyBuffer(m_handle, UPPER_BAND, 1, 1, bands) == 0)return;
if(Close[1] > bands[0])
   Trade.Buy(1.0, ExpertSymbol());
```

In addition to the already familiar methods, the Expert Advisor contains overridden methods ExpertNameFull and ParseXmlParams. The first one determines the unique name of the Expert Advisor, which is displayed in the user panel and as the EA's name. The second method loads Bollinger indicator settings form the XML file. The user panel and EA settings stored in XML files will be discussed in the next article. The rest of the EA operation is similar to the previous one. That is the aim of the proposed approach: complete unification of Expert Advisor development.

### Loading Custom Strategies to the Trading Engine

Once all the strategies have been described, we need to create their instances, initialize them with necessary parameters and add them to the trading engine. Any strategy loaded to the engine should have some required attributes (completed properties) that it should return. These attributes include the following properties:

- **The unique identifier of the strategy** (its magic number). Strategy IDs must be unique, even if they are created as instances of the same class. To specify a unique number, use the ExpertMagic() Set-method of the strategy.
- **Strategy timeframe** (or its operating period). Even if a strategy runs on multiple periods at the same time, you still need to specify the working timeframe. In this case, it may be, for example, the most often used timeframe. To specify the period, use the Timeframe Set-method.
- **Strategy symbol** (or its working instrument). If a strategy works with multiple symbols (a multi-currency strategy), you still need to specify the working symbol. This can be one of the symbols used by the strategy.
- **Strategy name**. In addition to the above attributes, each strategy also must have its own string name. The Expert Advisor name is specified using the ExpertName Set-method. This property is required, because it is used for the automatic creation of strategies from the Strategies.xml file. The same property is used to display the strategy in the user panel that will be described in the fourth article.

If at least one of these attributes is not specified, the trading engine will refuse to load the algorithm and will return a warning message specifying the missing parameter.

The trading engine consists of two main parts:

- An external module for manging strategies **CStrategyList**. This module is a manager of strategies and contains algorithms used to control them. We will discuss this module in the next part of the series.
- An internal module of strategies **CStrategy**. This module defines basic functions of the strategy. It was described in detail in this and the previous article: " [Universal Expert Advisor: Event Model and Trading Strategy Prototype (Part 2)](https://www.mql5.com/en/articles/2169)".

Each instance of CStrategy must be loaded into the CStrategyList manager of strategies. The manager of strategies allows loading strategies in two ways:

- **Automatically** using the Strategies.xml configuration file. For example, you can describe a set of strategies and their parameters in this file. Then, when you run an Expert Advisor on a chart, the strategy manager will create required instances of strategies, will initialize their parameters and add to its list. This method will be described in detail in the next article.
- **Manually** by adding the description to the executing module. In this case, the appropriate strategy object is created in the OnInit section of an Expert Advisor using a set of instructions, then it is initialized with required parameters and added to the CStrategyList strategy manager.

Here is the description of the process of manual configuration. We create the **Agent.mq5** file with the following contents:

```
//+------------------------------------------------------------------+
//|                                                        Agent.mq5 |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Strategy\StrategiesList.mqh>
#include <Strategy\Samples\ChannelSample.mqh>
#include <Strategy\Samples\MovingAverage.mqh>
CStrategyList Manager;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Configure and add to the list of strategies CMovingAverage
   CMovingAverage *ma=new CMovingAverage();
   ma.ExpertMagic(1215);
   ma.Timeframe(Period());
   ma.ExpertSymbol(Symbol());
   ma.ExpertName("Moving Average");
   ma.FastMA.MaPeriod(10);
   ma.SlowMA.MaPeriod(23);
   if(!Manager.AddStrategy(ma))
      delete ma;

//--- Configure and add to the list of strategies CChannel
   CChannel *channel=new CChannel();
   channel.ExpertMagic(1216);
   channel.Timeframe(Period());
   channel.ExpertSymbol(Symbol());
   channel.ExpertName("Bollinger Bands");
   channel.PeriodBands(50);
   channel.StdDev(2.0);
   if(!Manager.AddStrategy(channel))
      delete channel;

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   Manager.OnTick();
  }
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
   Manager.OnBookEvent(symbol);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   Manager.OnChartEvent(id,lparam,dparam,sparam);
  }
```

From this listing we see that strategy configuration is performed in the OnInit function. If you forget to specify one of the required parameters of the strategy, the strategy manager will refuse to add the strategy to its list. In this case, the AddStartegy method will return _false_ and the created strategy instance will need to be deleted. The strategy manager generates a warning message to help you understand the potential problem. Let us try to call such a message. To do this, comment out the instruction that sets the magic number:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Configure and add to the list of strategies CMovingAverage
   CMovingAverage *ma=new CMovingAverage();
 //ma.ExpertMagic(1215);
   ma.Timeframe(Period());
   ma.ExpertSymbol(Symbol());
   ma.ExpertName("Moving Average");
   ma.FastMA.MaPeriod(10);
   ma.SlowMA.MaPeriod(23);
   if(!Manager.AddStrategy(ma))
      delete ma;
   return(INIT_SUCCEEDED);
  }
```

After the start of the executing module, the following message will be displayed in the terminal:

```
2016.01.20 14:08:54.995 AgentWrong (FORTS-USDRUB,H1)    WARNING;CStrategyList::AddStrategy;The strategy should have a magic number. Adding strategy Moving Average is impossible;2016.01.20 14:09:01
```

It is clear from the message that the CStrategyList::AddStartegy method could not add the strategy, because its magic number was not set.

In addition to configuring strategies, the Agent.mq5 file includes processing of trade events to be analyzed. This processing includes tracking of events and passing them to the appropriate methods of the CStrategyList class.

Once the executable file is created, it can be compiled. The source code of the analyzed strategies is available in the **Include\\Strategy\\Samples** directory in the zipped attachment. A compiled Expert Advisor will be ready to use and will include the logic of the two trading strategies.

### Conclusion

We have analyzed the examples of custom strategies and the principle of classes that provide access to quotes through simple indexers. Also, we have discussed classes that implement logging and examples of object-oriented indicators. The proposed concept of constructing an Expert Advisor makes the formalization of the trading system logic easier. All that is needed is to define the rules in several overridden methods.

In the fourth part of the series " **Universal Expert Advisor: Trading in a Group and Managing a Portfolio of Strategies (Part 4)**" we will describe the algorithms, using which we can add an unlimited number of trading logics to one executable EA module ex5. In the fourth part we will also consider a simple user panel, using which you can manage Expert Advisors inside the executable module, for example, change their trading modes or buy and sell on their behalf.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2170](https://www.mql5.com/ru/articles/2170)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2170.zip "Download all attachments in the single ZIP archive")

[strategyarticle.zip](https://www.mql5.com/en/articles/download/2170/strategyarticle.zip "Download strategyarticle.zip")(100.23 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing graphical interfaces based on .Net Framework and C# (part 2): Additional graphical elements](https://www.mql5.com/en/articles/6549)
- [Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://www.mql5.com/en/articles/5563)
- [Custom Strategy Tester based on fast mathematical calculations](https://www.mql5.com/en/articles/4226)
- [R-squared as an estimation of quality of the strategy balance curve](https://www.mql5.com/en/articles/2358)
- [Universal Expert Advisor: CUnIndicator and Use of Pending Orders (Part 9)](https://www.mql5.com/en/articles/2653)
- [Implementing a Scalping Market Depth Using the CGraphic Library](https://www.mql5.com/en/articles/3336)
- [Universal Expert Advisor: Accessing Symbol Properties (Part 8)](https://www.mql5.com/en/articles/3270)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/77698)**
(19)


![igorbel](https://c.mql5.com/avatar/avatar_na2.png)

**[igorbel](https://www.mql5.com/en/users/igorbel)**
\|
31 Mar 2016 at 10:16

Why external variables are not made in **Agent.mq5**? How to optimise?

![Metin Kostekci](https://c.mql5.com/avatar/avatar_na2.png)

**[Metin Kostekci](https://www.mql5.com/en/users/doksandokuzprof)**
\|
11 Feb 2018 at 15:15

Thank you for this nice article.


![Alster Low](https://c.mql5.com/avatar/2020/7/5EFBF33F-AD56.jpg)

**[Alster Low](https://www.mql5.com/en/users/alster.low)**
\|
4 Aug 2020 at 11:38

Hello. The " Example of Expert Advisor Trading Two [Moving Averages](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "MetaTrader 5 Help: Moving Average indicator")' cant be complied as it has 70 errors.

![Xiong Luo](https://c.mql5.com/avatar/2021/2/601E1807-5C5B.jpg)

**[Xiong Luo](https://www.mql5.com/en/users/luojixinhao)**
\|
7 Oct 2020 at 09:00

Bookmarked


![fllss](https://c.mql5.com/avatar/avatar_na2.png)

**[fllss](https://www.mql5.com/en/users/fllss)**
\|
20 Sep 2023 at 17:42

**MetaQuotes:**

A new article [General EA: Custom Strategies and Secondary Trading Classes (Chapter 3) has been](https://www.mql5.com/en/articles/2170) released:

Author: [Vasiliy Sokolov](https://www.mql5.com/en/users/C-4 "C-4")

Hello, I downloaded your file and for some reason it reports an error.


![Thomas DeMark's contribution to technical analysis](https://c.mql5.com/2/19/50mlq5_4.png)[Thomas DeMark's contribution to technical analysis](https://www.mql5.com/en/articles/1995)

The article details TD points and TD lines discovered by Thomas DeMark. Their practical implementation is revealed. In addition to that, a process of writing three indicators and two Expert Advisors using the concepts of Thomas DeMark is demonstrated.

![Graphical Interfaces II: Setting Up the Event Handlers of the Library (Chapter 3)](https://c.mql5.com/2/22/Graphic-interface-part2__2.png)[Graphical Interfaces II: Setting Up the Event Handlers of the Library (Chapter 3)](https://www.mql5.com/en/articles/2204)

The previous articles contain the implementation of the classes for creating constituent parts of the main menu. Now, it is time to take a close look at the event handlers in the principle base classes and in the classes of the created controls. We will also pay special attention to managing the state of the chart depending on the location of the mouse cursor.

![Graphical Interfaces II: The Main Menu Element (Chapter 4)](https://c.mql5.com/2/22/Graphic-interface-part2__3.png)[Graphical Interfaces II: The Main Menu Element (Chapter 4)](https://www.mql5.com/en/articles/2207)

This is the final chapter of the second part of the series about graphical interfaces. Here, we are going to consider the creation of the main menu. The development of this control and setting up handlers of the library classes for correct reaction to the user's actions will be demonstrated here. We will also discuss how to attach context menus to the items of the main menu. Adding to that, we will mention blocking currently inactive elements.

![MetaTrader 5 features hedging position accounting system](https://c.mql5.com/2/22/hedge.png)[MetaTrader 5 features hedging position accounting system](https://www.mql5.com/en/articles/2299)

In order to expand possibilities of retail Forex traders, we have added the second accounting system — hedging. Now, it is possible to have multiple positions per symbol, including oppositely directed ones. This paves the way to implementing trading strategies based on the so-called "locking" — if the price moves against a trader, they can open a position in the opposite direction.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hsmqcghlyhintqalvjgfqundmrjtogch&ssn=1769186723558518326&ssn_dr=0&ssn_sr=0&fv_date=1769186723&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2170&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Universal%20Expert%20Advisor%3A%20Custom%20Strategies%20and%20Auxiliary%20Trade%20Classes%20(Part%203)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918672380534548&fz_uniq=5070535516950042531&sv=2552)

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