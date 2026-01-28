---
title: Library for easy and quick development of MetaTrader programs (part XXX): Pending trading requests - managing request objects
url: https://www.mql5.com/en/articles/7481
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:37:23.397380
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/7481&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070433133519640000)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7481#node01)
- [Pause object](https://www.mql5.com/en/articles/7481#node02)
- [A bit of code optimization](https://www.mql5.com/en/articles/7481#node03)
- [The class for managing pending request objects](https://www.mql5.com/en/articles/7481#node04)
- [Testing](https://www.mql5.com/en/articles/7481#node05)
- [What's next?](https://www.mql5.com/en/articles/7481#node06)


### Concept

Starting from the [article 26](https://www.mql5.com/en/articles/7394), we gradually developed and tested the concept
of working with pending trading requests. [In the previous article,](https://www.mql5.com/en/articles/7454) we have
created the classes of pending request objects corresponding to the general concept of library objects. This time, we are going to deal with
the class allowing the management of pending request objects.

Initially, I wanted to make an independent class for managing pending requests featuring all the necessary methods. But it turned out that the main
CTrading class of the library and the created new class of managing pending requests are so interrelated that it would be much easier to let
the new class for managing pending request objects be the descendant of the main trading class.

The entire management of pending request objects is performed in the class timer, therefore we make the base trading class timer virtual,
which means the timer of the pending request management class is to be virtual as well. Then everything that relates to the base trading class
timer is set in the class timer, while everything that should work in the class for managing pending request objects is set in the timer of that
class.

Apart from the class for managing pending request objects, we are going to create a small class to arrange a pause to avoid using the [Sleep()](https://www.mql5.com/en/docs/common/sleep)
function which stops the program execution for a delay time. With the pause object, we will no longer depend on ticks which means we will be able to test a
code requiring waiting on weekends. The pause is to be controlled in the timer.

While arranging [storing some data in the "Magic number" order property](https://www.mql5.com/en/articles/7394#node02), I had to
set identical methods returning IDs of the magic and groups, specified in the order magic value, in different classes requiring that data.
However, I missed the fact that I had already created [the base object of all library objects](https://www.mql5.com/en/articles/7071)
we need to inherit the newly created objects from. In this case, the objects obtain the event functionality of the base object, as well as some
methods that are similar in their principles and objectives repeating from class to class. Therefore, this is the object that is to contain
writing and obtaining data to the magic number, as well as managing displaying messages to the journal — the flag specifying the logging
level of each class containing the display of various messages. It would be reasonable to inherit all these classes from the base object and
obtain the necessary methods, rather than to set the same thing in each new class all over again.

Considering the above, we need to implement the pause object class (to be used in subsequent articles), conduct some optimization of the ready-made
code for transferring repeating methods of different classes to the base object class and create the class for managing pending requests.
Besides, we are going to improve pending request object classes by adding some new properties allowing us to compare the states of two
interconnected objects — the current status of order properties and the status of the corresponding request object properties. This is
necessary to register the fact of the pending request operation completion to timely remove it from the list of active requests, while
waiting to send the request to the server.

As usual, first of all, we add **new message indices in the**
**Datas.mqh file:**

```
   MSG_LIB_TEXT_PEND_REQUEST_STATUS_REMOVE,           // Pending request to delete a pending order
   MSG_LIB_TEXT_PEND_REQUEST_STATUS_MODIFY,           // Pending request to modify pending order parameters

   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_VOLUME,           // Actual volume
   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_PRICE,            // Actual order price
   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_STOPLIMIT,        // Actual StopLimit order price
   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_SL,               // Actual StopLoss order price
   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_TP,               // Actual TakeProfit order price
   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_TYPE_FILLING,     // Actual order filling type
   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_TYPE_TIME,        // Actual order expiration type
   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_EXPIRATION,       // Actual order lifetime

  };
//+------------------------------------------------------------------+
```

**and the texts corresponding to these new indices:**

```
   {"Отложенный запрос на удаление отложенного ордера","Pending request to remove pending order"},
   {"Отложенный запрос на модификацию параметров отложенного ордера","Pending request to modify pending order parameters"},

   {"Фактический объем","Actual volume"},
   {"Фактическая цена установки ордера","Actual order placement price"},
   {"Фактическая цена установки StopLimit-ордера","Actual StopLimit order placement price"},
   {"Фактическая цена установки StopLoss-ордера","Actual StopLoss order placement price"},
   {"Фактическая цена установки TakeProfit-ордера","Actual TakeProfit order placement price"},
   {"Фактический тип заливки ордера","Actual order filling type"},
   {"Фактический тип экспирации ордера","Actual order expiration type"},
   {"Фактическое время жизни ордера","Actual order lifetime"},

  };
//+---------------------------------------------------------------------+
```

### Pause object

in the folder of library service classes and functions \\MQL5\\Include\\DoEasy\ **Services\**, create the new **CPause** class
in the **Pause.mqh** file and add everything we need to it right away:

```
//+------------------------------------------------------------------+
//|                                                        Pause.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "DELib.mqh"
//+------------------------------------------------------------------+
//| Pause class                                                      |
//+------------------------------------------------------------------+
class CPause
  {
private:
   ulong             m_start;                               // Countdown start
   ulong             m_time_begin;                          // Pause countdown start time
   ulong             m_wait_msc;                            // Pause in milliseconds
public:
//--- Set the new (1) countdown start time and (2) pause in milliseconds
   void              SetTimeBegin(const ulong time)         { this.m_time_begin=time; this.m_start=::GetTickCount();    }
   void              SetWaitingMSC(const ulong pause)       { this.m_wait_msc=pause;                                    }

//--- Return (1) the time passed from the countdown start in milliseconds, (2) waiting completion flag
//--- (3) pause countdown start time, (4) pause in milliseconds
   ulong             Passed(void)                     const { return ::GetTickCount()-this.m_start;                     }
   bool              IsCompleted(void)                const { return this.Passed()>this.m_wait_msc;                     }
   ulong             TimeBegin(void)                  const { return this.m_time_begin;                                 }
   ulong             TimeWait(void)                   const { return this.m_wait_msc;                                   }

//--- Return the description (1) of the time passed till the countdown starts in milliseconds,
//--- (2) pause countdown start time, (3) pause in milliseconds
   string            PassedDescription(void)          const { return ::TimeToString(this.Passed()/1000,TIME_SECONDS);   }
   string            TimeBeginDescription(void)       const { return ::TimeMSCtoString(this.m_time_begin);              }
   string            WaitingMSCDescription(void)      const { return (string)this.m_wait_msc;                           }
   string            WaitingSECDescription(void)      const { return ::TimeToString(this.m_wait_msc/1000,TIME_SECONDS); }

//--- Constructor
                     CPause(void) : m_start(::GetTickCount()){;}
  };
//+------------------------------------------------------------------+
```

Include the file of service functions **DELib.mqh**
featuring the display of time in milliseconds — it will be required for the auxiliary class method displaying the pause start time in the
journal.

In the private class section, declare three class member variables necessary for the pause calculation:

- The **m\_start** variable is necessary for specifying a reference number at the moment of creating the pause object, as well as for
writing a new reference number of the new countdown start in a previously created pause object.

- The **m\_time\_begin** variable stores the specified pause countdown start time and is used solely for displaying info messages.
- The **m\_wait\_msc** variable contains the number of pause milliseconds — upon passing of the specified number of milliseconds, the
pause is deemed over.


I believe, the class methods need no explanations.

- First, the **SetTimeBegin()** method setting the new countdown start time writes the passed time to the **m\_time\_begin** variable.
Then it sets a new countdown start reference number to the **m\_start** variable. The result returned by the [GetTickCount()](https://www.mql5.com/en/docs/common/gettickcount)
function is used as the reference number of any coundown in the class.
- The **Passed()** method returning the number of passed pause milliseconds returns the difference in milliseconds between
the current reference time measurement and the value set in **m\_start** when launching the pause countdown.

The remaining class methods are quite obvious and need no explanations.

We will need this class later.

To make the pause class accessible from anywhere in the library and the program based on the
library, **include the CPause class file to the file of the DELib.mqh library service**
**functions:**

```
//+------------------------------------------------------------------+
//|                                                        DELib.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property strict  // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\Defines.mqh"
#include "Message.mqh"
#include "TimerCounter.mqh"
#include "Pause.mqh"
//+------------------------------------------------------------------+
```

### A bit of code optimization

It is time to move the methods repeating from class to class to the base object of all library objects and inherit these classes from the base
object (if it has not already been done).

**First of all, write the new variables and methods to the class of the base object of all library objects:**

```
//+------------------------------------------------------------------+
//| Base object class for all library objects                        |
//+------------------------------------------------------------------+
#define  CONTROLS_TOTAL    (10)
class CBaseObj : public CObject
  {
private:
   int               m_long_prop_total;
   int               m_double_prop_total;
   //--- Fill in the object property array
   template<typename T> bool  FillPropertySettings(const int index,T &array[][CONTROLS_TOTAL],T &array_prev[][CONTROLS_TOTAL],int &event_id);
protected:
   CArrayObj         m_list_events_base;                       // Object base event list
   CArrayObj         m_list_events;                            // Object event list
   ENUM_LOG_LEVEL    m_log_level;                              // Logging level
   MqlTick           m_tick;                                   // Tick structure for receiving quote data
   double            m_hash_sum;                               // Object data hash sum
   double            m_hash_sum_prev;                          // Object data hash sum during the previous check
   int               m_digits_currency;                        // Number of decimal places in an account currency
   int               m_global_error;                           // Global error code
   long              m_chart_id;                               // Control program chart ID
   bool              m_is_event;                               // Object event flag
   int               m_event_code;                             // Object event code
   int               m_event_id;                               // Event ID (equal to the object property value)
   string            m_name;                                   // Object name
   string            m_folder_name;                            // Name of the folder storing CBaseObj descendant objects
   bool              m_first_start;                            // First launch flag
   int               m_type;                                   // Object type (corresponds to the collection IDs)

//--- Data for storing, controlling and returning tracked properties:
//--- [Property index][0] Controlled property increase value
//--- [Property index][1] Controlled property decrease value
//--- [Property index][2] Controlled property value level
//--- [Property index][3] Property value
//--- [Property index][4] Property value change
//--- [Property index][5] Flag of a property change exceeding the increase value
//--- [Property index][6] Flag of a property change exceeding the decrease value
//--- [Property index][7] Flag of a property increase exceeding the control level
//--- [Property index][8] Flag of a property decrease being less than the control level
//--- [Property index][9] Flag of a property value being equal to the control level
   long              m_long_prop_event[][CONTROLS_TOTAL];         // The array for storing object's integer properties values and controlled property change values
   double            m_double_prop_event[][CONTROLS_TOTAL];       // The array for storing object's real properties values and controlled property change values
   long              m_long_prop_event_prev[][CONTROLS_TOTAL];    // The array for storing object's controlled integer properties values during the previous check
   double            m_double_prop_event_prev[][CONTROLS_TOTAL];  // The array for storing object's controlled real properties values during the previous check

//--- Return (1) time in milliseconds, (2) milliseconds from the MqlTick time value
   long              TickTime(void)                            const { return #ifdef __MQL5__ this.m_tick.time_msc #else this.m_tick.time*1000 #endif ;  }
   ushort            MSCfromTime(const long time_msc)          const { return #ifdef __MQL5__ ushort(this.TickTime()%1000) #else 0 #endif ;              }
//--- return the flag of the event code presence in the event object
   bool              IsPresentEventFlag(const int change_code) const { return (this.m_event_code & change_code)==change_code; }
//--- Return the number of decimal places of the account currency
   int               DigitsCurrency(void)                      const { return this.m_digits_currency; }
//--- Returns the number of decimal places in the 'double' value
   int               GetDigits(const double value)             const;

//--- Set the size of the array of controlled (1) integer and (2) real object properties
   bool              SetControlDataArraySizeLong(const int size);
   bool              SetControlDataArraySizeDouble(const int size);
//--- Check the array size of object properties
   bool              CheckControlDataArraySize(bool check_long=true);

//--- Check the list of object property changes and create an event
   void              CheckEvents(void);

//--- (1) Pack a 'ushort' number to a passed 'long' number
   long              UshortToLong(const ushort ushort_value,const uchar to_byte,long &long_value);

protected:
//--- (1) convert a 'ushort' value to a specified 'long' number byte
   long              UshortToByte(const ushort value,const uchar to_byte)  const;

public:
//--- Set the value of the pbject property controlled (1) increase, (2) decrease, (3) control level
   template<typename T> void  SetControlledValueINC(const int property,const T value);
   template<typename T> void  SetControlledValueDEC(const int property,const T value);
   template<typename T> void  SetControlledValueLEVEL(const int property,const T value);

//--- (1) Set, (2) return the error logging level
   void              SetLogLevel(const ENUM_LOG_LEVEL level)                  { this.m_log_level=level;                                               }
   ENUM_LOG_LEVEL    GetLogLevel(void)                                  const { return this.m_log_level;                                              }
//--- Return the set value of the controlled (1) integer and (2) real object properties increase
   long              GetControlledLongValueINC(const int property)      const { return this.m_long_prop_event[property][0];                           }
   double            GetControlledDoubleValueINC(const int property)    const { return this.m_double_prop_event[property-this.m_long_prop_total][0];  }
//--- Return the set value of the controlled (1) integer and (2) real object properties decrease
   long              GetControlledLongValueDEC(const int property)      const { return this.m_long_prop_event[property][1];                           }
   double            GetControlledDoubleValueDEC(const int property)    const { return this.m_double_prop_event[property-this.m_long_prop_total][1];  }
//--- Return the specified control level of object's (1) integer and (2) real properties
   long              GetControlledLongValueLEVEL(const int property)    const { return this.m_long_prop_event[property][2];                           }
   double            GetControlledDoubleValueLEVEL(const int property)  const { return this.m_double_prop_event[property-this.m_long_prop_total][2];  }

//--- Return the current value of the object (1) integer and (2) real property
   long              GetPropLongValue(const int property)               const { return this.m_long_prop_event[property][3];                           }
   double            GetPropDoubleValue(const int property)             const { return this.m_double_prop_event[property-this.m_long_prop_total][3];  }
//--- Return the change value of the controlled (1) integer and (2) real object property
   long              GetPropLongChangedValue(const int property)        const { return this.m_long_prop_event[property][4];                           }
   double            GetPropDoubleChangedValue(const int property)      const { return this.m_double_prop_event[property-this.m_long_prop_total][4];  }
//--- Return the flag of an (1) integer and (2) real property value change exceeding the increase value
   long              GetPropLongFlagINC(const int property)             const { return this.m_long_prop_event[property][5];                           }
   double            GetPropDoubleFlagINC(const int property)           const { return this.m_double_prop_event[property-this.m_long_prop_total][5];  }
//--- Return the flag of an (1) integer and (2) real property value change exceeding the decrease value
   long              GetPropLongFlagDEC(const int property)             const { return this.m_long_prop_event[property][6];                           }
   double            GetPropDoubleFlagDEC(const int property)           const { return this.m_double_prop_event[property-this.m_long_prop_total][6];  }
//--- Return the flag of an (1) integer and (2) real property value increase exceeding the control level
   long              GetPropLongFlagMORE(const int property)            const { return this.m_long_prop_event[property][7];                           }
   double            GetPropDoubleFlagMORE(const int property)          const { return this.m_double_prop_event[property-this.m_long_prop_total][7];  }
//--- Return the flag of an (1) integer and (2) real property value decrease being less than the control level
   long              GetPropLongFlagLESS(const int property)            const { return this.m_long_prop_event[property][8];                           }
   double            GetPropDoubleFlagLESS(const int property)          const { return this.m_double_prop_event[property-this.m_long_prop_total][8];  }
//--- Return the flag of an (1) integer and (2) real property being equal to the control level
   long              GetPropLongFlagEQUAL(const int property)           const { return this.m_long_prop_event[property][9];                           }
   double            GetPropDoubleFlagEQUAL(const int property)         const { return this.m_double_prop_event[property-this.m_long_prop_total][9];  }

//--- Reset the variables of (1) tracked and (2) controlled object data (can be reset in the descendants)
   void              ResetChangesParams(void);
   virtual void      ResetControlsParams(void);
//--- Add the (1) object event and (2) the object event reason to the list
   bool              EventAdd(const ushort event_id,const long lparam,const double dparam,const string sparam);
   bool              EventBaseAdd(const int event_id,const ENUM_BASE_EVENT_REASON reason,const double value);
//--- Set/return the occurred event flag to the object data
   void              SetEvent(const bool flag)                       { this.m_is_event=flag;                   }
   bool              IsEvent(void)                             const { return this.m_is_event;                 }
//--- Return (1) the list of events, (2) the object event code and (3) the global error code
   CArrayObj        *GetListEvents(void)                             { return &this.m_list_events;             }
   int               GetEventCode(void)                        const { return this.m_event_code;               }
   int               GetError(void)                            const { return this.m_global_error;             }
//--- Return (1) an event object and (2) a base event by its number in the list
   CEventBaseObj    *GetEvent(const int shift=WRONG_VALUE,const bool check_out=true);
   CBaseEvent       *GetEventBase(const int index);
//--- Return the number of (1) object events
   int               GetEventsTotal(void)                      const { return this.m_list_events.Total();      }
//--- (1) Set and (2) return the chart ID of the control program
   void              SetChartID(const long id)                       { this.m_chart_id=id;                     }
   long              GetChartID(void)                          const { return this.m_chart_id;                 }
//--- (1) Set the sub-folder name, (2) return the folder name for storing descendant object files
   void              SetSubFolderName(const string name)             { this.m_folder_name=DIRECTORY+name;      }
   string            GetFolderName(void)                       const { return this.m_folder_name;              }
//--- Return the object name
   string            GetName(void)                             const { return this.m_name;                     }
//--- Update the object data to search for changes (Calling from the descendants: CBaseObj::Refresh())
   virtual void      Refresh(void);
//--- Return an object type
   virtual int       Type(void)                                const { return this.m_type;                     }
//--- Return an object event description
   string            EventDescription(const int property,
                                      const ENUM_BASE_EVENT_REASON reason,
                                      const int source,
                                      const string value,
                                      const string property_descr,
                                      const int digits);

//--- Data location in the magic number int value
      //-----------------------------------------------------------
      //  bit   32|31       24|23       16|15        8|7         0|
      //-----------------------------------------------------------
      //  byte    |     3     |     2     |     1     |     0     |
      //-----------------------------------------------------------
      //  data    |   uchar   |   uchar   |         ushort        |
      //-----------------------------------------------------------
      //  descr   |pend req id| id2 | id1 |          magic        |
      //-----------------------------------------------------------

//--- Set the ID of the (1) first group, (2) second group, (3) pending request to the magic number value
   void              SetGroupID1(const uchar group,uint &magic)            { magic &=0xFFF0FFFF; magic |= uint(this.ConvToXX(group,0)<<16);  }
   void              SetGroupID2(const uchar group,uint &magic)            { magic &=0xFF0FFFFF; magic |= uint(this.ConvToXX(group,1)<<16);  }
   void              SetPendReqID(const uchar id,uint &magic)              { magic &=0x00FFFFFF; magic |= (uint)id<<24;                      }
//--- Convert the value of 0 - 15 into the necessary uchar number bits (0 - lower, 1 - upper ones)
   uchar             ConvToXX(const uchar number,const uchar index)  const { return((number>15 ? 15 : number)<<(4*(index>1 ? 1 : index)));   }
//--- Return (1) the specified magic number, the ID of (2) the first group, (3) second group, (4) pending request from the magic number value
   ushort            GetMagicID(const uint magic)                    const { return ushort(magic & 0xFFFF);                                  }
   uchar             GetGroupID1(const uint magic)                   const { return uchar(magic>>16) & 0x0F;                                 }
   uchar             GetGroupID2(const uint magic)                   const { return uchar((magic>>16) & 0xF0)>>4;                            }
   uchar             GetPendReqID(const uint magic)                  const { return uchar(magic>>24) & 0xFF;                                 }

//--- Constructor
                     CBaseObj();
  };
//+------------------------------------------------------------------+
```

The protected variable (available to descendants but not programs) stores the
logging level values for each of the base object descendant objects requiring displaying info in the journal. It receives the
logging level from the **ENUM\_LOG\_LEVEL** enumeration set in **Defines.mqh**:

```
//+------------------------------------------------------------------+
//|  Logging level                                                   |
//+------------------------------------------------------------------+
enum ENUM_LOG_LEVEL
  {
   LOG_LEVEL_NO_MSG,                                        // Trading logging disabled
   LOG_LEVEL_ERROR_MSG,                                     // Only trading errors
   LOG_LEVEL_ALL_MSG                                        // Full logging
  };
//+------------------------------------------------------------------+
```

The variable values are set and returned using the **SetLogLevel()**
and **GetLogLevel()** public methods respectively.

The methods of setting and returning the values of various IDs stored in the
order/position magic number are made public as well.

We [have \\
already considered](https://www.mql5.com/en/articles/7394#node02) these methods before, so it makes no sense to dwell on them here.

**The symbol's base trading object** is in the \\MQL5\\Include\\DoEasy\\Objects\\Trade\ **TradeObj.mqh** file.

Let's make the necessary improvements: **include the library base object**
**file to it and make the CBaseObj object class its parent class:**

```
//+------------------------------------------------------------------+
//|                                                     TradeObj.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\DELib.mqh"
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Trading object class                                             |
//+------------------------------------------------------------------+
class CTradeObj : public CBaseObj
  {
```

After making these improvements, the symbol's trading object class becomes a descendant of the base object of all library objects. Now we need
to remove all variables and methods already present in the CBaseObj parent class.

**Remove unnecessary variables present in the parent class**
**from the private section of the class:**

```
   SActions                   m_datas;
   MqlTick                    m_tick;                                            // Tick structure for receiving prices
   MqlTradeRequest            m_request;                                         // Trade request structure
   MqlTradeResult             m_result;                                          // trade request execution result
   ENUM_SYMBOL_CHART_MODE     m_chart_mode;                                      // Price type for constructing bars
   ENUM_ACCOUNT_MARGIN_MODE   m_margin_mode;                                     // Margin calculation mode
   ENUM_ORDER_TYPE_FILLING    m_type_filling;                                    // Filling policy
   ENUM_ORDER_TYPE_TIME       m_type_time;                                       // Order type per expiration
   int                        m_symbol_expiration_flags;                         // Flags of order expiration modes for a trading object symbol
   ulong                      m_magic;                                           // Magic number
   string                     m_symbol;                                          // Symbol
   string                     m_comment;                                         // Comment
   ulong                      m_deviation;                                       // Slippage in points
   double                     m_volume;                                          // Volume
   datetime                   m_expiration;                                      // Order expiration time (for ORDER_TIME_SPECIFIED type order)
   bool                       m_async_mode;                                      // Flag of asynchronous sending of a trade request
   ENUM_LOG_LEVEL             m_log_level;                                       // Logging level
   int                        m_stop_limit;                                      // Distance of placing a StopLimit order in points
   bool                       m_use_sound;                                       // The flag of using sounds of the object trading events
   uint                       m_multiplier;                                      // The spread multiplier to adjust levels relative to StopLevel
```

**Remove the methods of setting and returning a logging level from the public section of the class:**

```
//--- (1) Return the margin calculation mode, (2) hedge account flag
   ENUM_ACCOUNT_MARGIN_MODE   GetMarginMode(void)                                const { return this.m_margin_mode;           }
   bool                       IsHedge(void) const { return this.GetMarginMode()==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING;          }
//--- (1) Set, (2) return the error logging level
   void                       SetLogLevel(const ENUM_LOG_LEVEL level)                  { this.m_log_level=level;              }
   ENUM_LOG_LEVEL             GetLogLevel(void)                                  const { return this.m_log_level;             }
//--- (1) Set, (2) return the filling policy
   void                       SetTypeFilling(const ENUM_ORDER_TYPE_FILLING type)       { this.m_type_filling=type;            }
   ENUM_ORDER_TYPE_FILLING    GetTypeFilling(void)                               const { return this.m_type_filling;          }
```

**Remove the logging level initialization from the**
**initialization list of the constructor class:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTradeObj::CTradeObj(void) : m_magic(0),
                             m_deviation(5),
                             m_stop_limit(0),
                             m_expiration(0),
                             m_async_mode(false),
                             m_type_filling(ORDER_FILLING_FOK),
                             m_type_time(ORDER_TIME_GTC),
                             m_comment(::MQLInfoString(MQL_PROGRAM_NAME)+" by DoEasy"),
                             m_log_level(LOG_LEVEL_ERROR_MSG)
```

**Add logging level initialization to** **the method**
**body:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTradeObj::CTradeObj(void) : m_magic(0),
                             m_deviation(5),
                             m_stop_limit(0),
                             m_expiration(0),
                             m_async_mode(false),
                             m_type_filling(ORDER_FILLING_FOK),
                             m_type_time(ORDER_TIME_GTC),
                             m_comment(::MQLInfoString(MQL_PROGRAM_NAME)+" by DoEasy")

  {
   //--- Margin calculation mode
   this.m_margin_mode=
     (
      #ifdef __MQL5__ (ENUM_ACCOUNT_MARGIN_MODE)::AccountInfoInteger(ACCOUNT_MARGIN_MODE)
      #else /* MQL4 */ ACCOUNT_MARGIN_MODE_RETAIL_HEDGING #endif
     );
   //--- Spread multiplier
   this.m_multiplier=1;
   //--- Set default sounds and flags of using sounds
   this.m_use_sound=false;
   this.m_log_level=LOG_LEVEL_ERROR_MSG;
   this.InitSounds();
  }
//+------------------------------------------------------------------+
```

**The base abstract order object** is in the \\MQL5\\Include\\DoEasy\\Objects\\Orders\ **Order.mqh** file.

Let's make the necessary improvements: **include the library base object**
**file to it and make the CBaseObj object class its parent class:**

```
//+------------------------------------------------------------------+
//|                                                        Order.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "..\..\Services\DELib.mqh"
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Abstract order class                                             |
//+------------------------------------------------------------------+
class COrder : public CBaseObj
  {
```

**Remove the methods of receiving IDs written in the order**
**magic number from the private section of the class**(leave the plate reminding of
the location of IDs in the magic number value):

```
//+------------------------------------------------------------------+
//| Abstract order class                                             |
//+------------------------------------------------------------------+
class COrder : public CObject
  {
private:
   ulong             m_ticket;                                    // Selected order/deal ticket (MQL5)
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];      // String properties

//--- Return the index of the array the order's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_ORDER_PROP_DOUBLE property)      const { return(int)property-ORDER_PROP_INTEGER_TOTAL;                         }
   int               IndexProp(ENUM_ORDER_PROP_STRING property)      const { return(int)property-ORDER_PROP_INTEGER_TOTAL-ORDER_PROP_DOUBLE_TOTAL; }

//--- Data location in the magic number int value
      //-----------------------------------------------------------
      //  bit   32|31       24|23       16|15        8|7         0|
      //-----------------------------------------------------------
      //  byte    |     3     |     2     |     1     |     0     |
      //-----------------------------------------------------------
      //  data    |   uchar   |   uchar   |         ushort        |
      //-----------------------------------------------------------
      //  descr   |pend req id| id2 | id1 |          magic        |
      //-----------------------------------------------------------
//--- Return (1) the specified magic number, the ID of (2) the first group, (3) second group, (4) pending request from the magic number value
   ushort            GetMagicID(void)                                const { return ushort(this.Magic() & 0xFFFF);                                 }
   uchar             GetGroupID1(void)                               const { return uchar(this.Magic()>>16) & 0x0F;                                }
   uchar             GetGroupID2(void)                               const { return uchar((this.Magic()>>16) & 0xF0)>>4;                           }
   uchar             GetPendReqID(void)                              const { return uchar(this.Magic()>>24) & 0xFF;                                }

public:
```

**In the closed class constructor, add specification**
**of a magic number to the methods of receiving IDs:**

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
COrder::COrder(ENUM_ORDER_STATUS order_status,const ulong ticket)
  {
//--- Save integer properties
   this.m_ticket=ticket;
   this.m_long_prop[ORDER_PROP_STATUS]                               = order_status;
   this.m_long_prop[ORDER_PROP_MAGIC]                                = this.OrderMagicNumber();
   this.m_long_prop[ORDER_PROP_TICKET]                               = this.OrderTicket();
   this.m_long_prop[ORDER_PROP_TIME_EXP]                             = this.OrderExpiration();
   this.m_long_prop[ORDER_PROP_TYPE_FILLING]                         = this.OrderTypeFilling();
   this.m_long_prop[ORDER_PROP_TYPE_TIME]                            = this.OrderTypeTime();
   this.m_long_prop[ORDER_PROP_TYPE]                                 = this.OrderType();
   this.m_long_prop[ORDER_PROP_STATE]                                = this.OrderState();
   this.m_long_prop[ORDER_PROP_DIRECTION]                            = this.OrderTypeByDirection();
   this.m_long_prop[ORDER_PROP_POSITION_ID]                          = this.OrderPositionID();
   this.m_long_prop[ORDER_PROP_REASON]                               = this.OrderReason();
   this.m_long_prop[ORDER_PROP_DEAL_ORDER_TICKET]                    = this.DealOrderTicket();
   this.m_long_prop[ORDER_PROP_DEAL_ENTRY]                           = this.DealEntry();
   this.m_long_prop[ORDER_PROP_POSITION_BY_ID]                       = this.OrderPositionByID();
   this.m_long_prop[ORDER_PROP_TIME_OPEN]                            = this.OrderOpenTimeMSC();
   this.m_long_prop[ORDER_PROP_TIME_CLOSE]                           = this.OrderCloseTimeMSC();
   this.m_long_prop[ORDER_PROP_TIME_UPDATE]                          = this.PositionTimeUpdateMSC();

//--- Save real properties
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_OPEN)]         = this.OrderOpenPrice();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_CLOSE)]        = this.OrderClosePrice();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PROFIT)]             = this.OrderProfit();
   this.m_double_prop[this.IndexProp(ORDER_PROP_COMMISSION)]         = this.OrderCommission();
   this.m_double_prop[this.IndexProp(ORDER_PROP_SWAP)]               = this.OrderSwap();
   this.m_double_prop[this.IndexProp(ORDER_PROP_VOLUME)]             = this.OrderVolume();
   this.m_double_prop[this.IndexProp(ORDER_PROP_SL)]                 = this.OrderStopLoss();
   this.m_double_prop[this.IndexProp(ORDER_PROP_TP)]                 = this.OrderTakeProfit();
   this.m_double_prop[this.IndexProp(ORDER_PROP_VOLUME_CURRENT)]     = this.OrderVolumeCurrent();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_STOP_LIMIT)]   = this.OrderPriceStopLimit();

//--- Save string properties
   this.m_string_prop[this.IndexProp(ORDER_PROP_SYMBOL)]             = this.OrderSymbol();
   this.m_string_prop[this.IndexProp(ORDER_PROP_COMMENT)]            = this.OrderComment();
   this.m_string_prop[this.IndexProp(ORDER_PROP_EXT_ID)]             = this.OrderExternalID();

//--- Save additional integer properties
   this.m_long_prop[ORDER_PROP_PROFIT_PT]                            = this.ProfitInPoints();
   this.m_long_prop[ORDER_PROP_TICKET_FROM]                          = this.OrderTicketFrom();
   this.m_long_prop[ORDER_PROP_TICKET_TO]                            = this.OrderTicketTo();
   this.m_long_prop[ORDER_PROP_CLOSE_BY_SL]                          = this.OrderCloseByStopLoss();
   this.m_long_prop[ORDER_PROP_CLOSE_BY_TP]                          = this.OrderCloseByTakeProfit();
   this.m_long_prop[ORDER_PROP_MAGIC_ID]                             = this.GetMagicID((uint)this.GetProperty(ORDER_PROP_MAGIC));
   this.m_long_prop[ORDER_PROP_GROUP_ID1]                            = this.GetGroupID1((uint)this.GetProperty(ORDER_PROP_MAGIC));
   this.m_long_prop[ORDER_PROP_GROUP_ID2]                            = this.GetGroupID2((uint)this.GetProperty(ORDER_PROP_MAGIC));
   this.m_long_prop[ORDER_PROP_PEND_REQ_ID]                          = this.GetPendReqID((uint)this.GetProperty(ORDER_PROP_MAGIC));

//--- Save additional real properties
   this.m_double_prop[this.IndexProp(ORDER_PROP_PROFIT_FULL)]        = this.ProfitFull();

//--- Save additional string properties
   this.m_string_prop[this.IndexProp(ORDER_PROP_COMMENT_EXT)]        = "";
  }
//+------------------------------------------------------------------+
```

**The base trading class** is in \\MQL5\\Include\\DoEasy **\\Trading.mqh**.

Let's make the necessary improvements. Since the method has become the parent one for the future class of managing pending requests, **move**
**the pointers to collection objects and the list of pointers to pending requests to the protected section out of the private one:**

```
//+------------------------------------------------------------------+
//| Trading class                                                    |
//+------------------------------------------------------------------+
class CTrading : public CBaseObj
  {
protected:
   CAccount            *m_account;                       // Pointer to the current account object
   CSymbolsCollection  *m_symbols;                       // Pointer to the symbol collection list
   CMarketCollection   *m_market;                        // Pointer to the list of the collection of market orders and positions
   CHistoryCollection  *m_history;                       // Pointer to the list of the collection of historical orders and deals
   CEventsCollection   *m_events;                        // Pointer to the event collection list
   CArrayObj            m_list_request;                  // List of pending requests
private:
```

**Remove the variable for storing the logging level from**
**the private section:**

```
private:
   CArrayInt            m_list_errors;                   // Error list
   bool                 m_is_trade_disable;              // Flag disabling trading
   bool                 m_use_sound;                     // The flag of using sounds of the object trading events
   uchar                m_total_try;                     // Number of trading attempts
   ENUM_LOG_LEVEL       m_log_level;                     // Logging level
   MqlTradeRequest      m_request;                       // Trading request prices
   ENUM_TRADE_REQUEST_ERR_FLAGS m_error_reason_flags;    // Flags of error source in a trading method
   ENUM_ERROR_HANDLING_BEHAVIOR m_err_handling_behavior; // Behavior when handling error
```

**In the public section of the class, add the method returning the**
**entire trading class object and make the class timer virtual:**

```
public:
//--- Return itself
   CTrading            *GetObject(void)    { return &this;   }
//--- Constructor
                        CTrading();
//--- Timer
   virtual void         OnTimer(void);
//--- Get the pointers to the lists (make sure to call the method in program's OnInit() since the symbol collection list is created there)
```

**In the method of creating a pending request, add passing an order**
**object to the method and remove the methods for working with IDs set in the**
**order magic number:**

```
//--- Create a pending request
   bool                 CreatePendingRequest(const ENUM_PEND_REQ_STATUS status,
                                             const uchar id,
                                             const uchar attempts,
                                             const ulong wait,
                                             const MqlTradeRequest &request,
                                             const int retcode,
                                             CSymbol *symbol_obj,
                                             COrder *order);

//--- Data location in the magic number int value
      //-----------------------------------------------------------
      //  bit   32|31       24|23       16|15        8|7         0|
      //-----------------------------------------------------------
      //  byte    |     3     |     2     |     1     |     0     |
      //-----------------------------------------------------------
      //  data    |   uchar   |   uchar   |         ushort        |
      //-----------------------------------------------------------
      //  descr   |pend req id| id2 | id1 |          magic        |
      //-----------------------------------------------------------

//--- Set the ID of the (1) first group, (2) second group, (3) pending request to the magic number value
   void              SetGroupID1(const uchar group,uint &magic)            { magic &=0xFFF0FFFF; magic |= uint(ConvToXX(group,0)<<16);    }
   void              SetGroupID2(const uchar group,uint &magic)            { magic &=0xFF0FFFFF; magic |= uint(ConvToXX(group,1)<<16);    }
   void              SetPendReqID(const uchar id,uint &magic)              { magic &=0x00FFFFFF; magic |= (uint)id<<24;                   }
//--- Convert the value of 0 - 15 into the necessary uchar number bits (0 - lower, 1 - upper ones)
   uchar             ConvToXX(const uchar number,const uchar index)  const { return((number>15 ? 15 : number)<<(4*(index>1 ? 1 : index)));}
//--- Return (1) the specified magic number, the ID of (2) the first group, (3) second group, (4) pending request from the magic number value
   ushort            GetMagicID(const uint magic)                    const { return ushort(magic & 0xFFFF);                               }
   uchar             GetGroupID1(const uint magic)                   const { return uchar(magic>>16) & 0x0F;                              }
   uchar             GetGroupID2(const uint magic)                   const { return uchar((magic>>16) & 0xF0)>>4;                         }
   uchar             GetPendReqID(const uint magic)                  const { return uchar(magic>>24) & 0xFF;                              }

  };
//+------------------------------------------------------------------+
```

**Remove everything from the class timer implementation leaving it empty:**

```
//+------------------------------------------------------------------+
//| Timer                                                            |
//+------------------------------------------------------------------+
void CTrading::OnTimer(void)
  {
  }
//+------------------------------------------------------------------+
```

Here we will write the code in case we need to handle something only in the timer of this class, which is a parent one for other trading classes (in
the current implementation, for the future class of managing pending requests).

**In the CheckTradeConstraints() method of checking limitations for trading, namely**
**in the block for checking the volume validity during trading operations requiring a lot value (condition**
**supplemented), check the volume validity only if its value passed by the**
**input exceeds zero:**

```
//--- If closing or not removing/modifying
   if(action_type<ACTION_TYPE_CLOSE_BY || action_type==ACTION_TYPE_CLOSE)
     {
      //--- In case of close-only, write the error code to the list and return 'false' - there is no point in further checks
      if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY)
        {
         this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_CLOSEONLY);
         return false;
        }
      //--- Check the minimum volume
      if(volume>0)
        {
         if(volume<symbol_obj.LotsMin())
           {
            //--- The volume in a request is less than the minimum allowed one.
            //--- add the error code to the list
            this.m_error_reason_flags |=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME);
            //--- If the EA behavior during the trading error is set to "abort trading operation",
            //--- return 'false' - there is no point in further checks
            if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
               return false;
            //--- If the EA behavior during a trading error is set to
            //--- "correct parameters" or "create a pending request",
            //--- write 'false' to the result
            else res &=false;
           }
         //--- Check the maximum volume
         else if(volume>symbol_obj.LotsMax())
           {
            //--- The volume in the request exceeds the maximum acceptable one.
            //--- add the error code to the list
            this.m_error_reason_flags |=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME);
            //--- If the EA behavior during the trading error is set to "abort trading operation",
            //--- return 'false' - there is no point in further checks
            if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
               return false;
            //--- If the EA behavior during a trading error is set to
            //--- "correct parameters" or "create a pending request",
            //--- write 'false' to the result
            else res &=false;
           }
         //--- Check the minimum volume gradation
         double step=symbol_obj.LotsStep();
         if(fabs((int)round(volume/step)*step-volume)>0.0000001)
           {
            //--- The volume in the request is not a multiple of the minimum gradation of the lot change step
            //--- add the error code to the list
            this.m_error_reason_flags |=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_LIB_TEXT_INVALID_VOLUME_STEP);
            //--- If the EA behavior during the trading error is set to "abort trading operation",
            //--- return 'false' - there is no point in further checks
            if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
               return false;
            //--- If the EA behavior during a trading error is set to
            //--- "correct parameters" or "create a pending request",
            //--- write 'false' to the result
            else res &=false;
           }
        }
     }

//--- When opening a position
```

Next:

A method receives an order object in case creation of a pending request object
is required. In case there is no such object (in the methods of opening a position and setting pending orders), pass NULL
to the method.

**In the method of opening a position, pass NULL**
**as the last parameter:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false' (OpenPosition)
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the trading request magic number, has no pending request ID
         if(this.GetPendReqID((uint)magic)==0)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the pending request object ID to the magic number and fill in the remaining unfilled fields of the trading request structure
            uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
            this.SetPendReqID((uchar)id,mn);
            this.m_request.magic=mn;
            this.m_request.action=TRADE_ACTION_DEAL;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.type=order_type;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_OPEN,(uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj,NULL);
           }
         return false;
        }
```

In the method of setting a pending order, pass NULL as an order object to the
method of creating a pending request, while in the methods working with the already existing orders and positions, pass
the order object to the method of creating a pending request, for example, pass the order object **in the method of modifying**
**stop orders of the existing position:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false' (ModifyPosition)
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the position ticket is not present in the list
         if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true));
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write a type of a conducted operation, as well as a symbol and a ticket of a modified position to the request structure
            this.m_request.action=TRADE_ACTION_SLTP;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.position=ticket;
            this.m_request.type=order_type;
            this.m_request.volume=order.Volume();
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_SLTP,(uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj,order);
           }
         return false;
        }
```

In the remaining methods with the known order or position, pass the order object to the method of creating a pending request.

During the test, it turned out that the **ClosePosition()** position closure method sometimes received an invalid volume
of a closed position in case of a partial closure. A zero volume was sent and the verification was skipped to never be repeated again. To fix
this, we have already added checking the volume when closing a position to the method defining if trading is enabled. Now let's add writing
the correct volume when creating a pending request, as well as add sending it to
the verification method.

**To do this, add a single string to the**
**method:**

```
//--- Write a deviation and a comment to the request structure
   this.m_request.deviation=(deviation==ULONG_MAX ? trade_obj.GetDeviation() : deviation);
   this.m_request.comment=(comment==NULL ? trade_obj.GetComment() : comment);
   this.m_request.volume=(volume==WRONG_VALUE || volume>order.Volume() ? order.Volume() : symbol_obj.NormalizedLot(volume));
//--- If there are trading and volume limitations
//--- there are limitations on FreezeLevel - play the error sound and exit
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(volume,0,action,order_type,symbol_obj,trade_obj,DFUN,0,0,0,ticket);
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
```

Here we add the volume to the trading request structure the following way: if -1 is passed to the method or the volume passed to the method exceeds
the one of the closed position, write the full position volume (full closure) to the structure. Otherwise, write the normalized volume
passed to the method (it may be invalid, for example when dividing the volume of 0.05 by 2 when partially closing a position, the method
receives 0.025 causing an error).

**In the method of modifying the ModifyOrder() pending order, add the**
**magic number ID and the order volume to the trading request**
**structure:**

```
//--- Write the magic number, volume, filling type, as well as expiration date and type to the request structure
   this.m_request.magic=order.GetMagicID((uint)order.Magic());
   this.m_request.volume=order.Volume();
   this.m_request.type_filling=(type_filling>WRONG_VALUE ? type_filling : order.TypeFilling());
   this.m_request.expiration=(expiration>WRONG_VALUE ? expiration : order.TimeExpiration());
   this.m_request.type_time=(type_time>WRONG_VALUE ? type_time : order.TypeTime());

//--- If there are trading limitations,
//--- StopLevel or FreezeLevel limitations, play the error sound and exit
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(0,this.m_request.price,action,order_type,symbol_obj,trade_obj,DFUN,this.m_request.stoplimit,this.m_request.sl,this.m_request.tp,ticket);
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
```

This is completely useless for real trading since an order is modified by ticket. However, in case of pending orders, this provides us with the
correct display of entries about a modified order in the journal.

**The method of creating a pending request has undergone minor changes:**

```
//+------------------------------------------------------------------+
//| Create a pending request                                         |
//+------------------------------------------------------------------+
bool CTrading::CreatePendingRequest(const ENUM_PEND_REQ_STATUS status,
                                    const uchar id,
                                    const uchar attempts,
                                    const ulong wait,
                                    const MqlTradeRequest &request,
                                    const int retcode,
                                    CSymbol *symbol_obj,
                                    COrder *order)
  {
   //--- Create a new pending request object depending on a request status
   CPendRequest *req_obj=NULL;
   switch(status)
     {
      case PEND_REQ_STATUS_OPEN     : req_obj=new CPendReqOpen(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);    break;
      case PEND_REQ_STATUS_CLOSE    : req_obj=new CPendReqClose(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);   break;
      case PEND_REQ_STATUS_SLTP     : req_obj=new CPendReqSLTP(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);    break;
      case PEND_REQ_STATUS_PLACE    : req_obj=new CPendReqPlace(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);   break;
      case PEND_REQ_STATUS_REMOVE   : req_obj=new CPendReqRemove(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);  break;
      case PEND_REQ_STATUS_MODIFY   : req_obj=new CPendReqModify(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);  break;
      default: req_obj=NULL;
        break;
     }
   if(req_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_FAILING_CREATE_PENDING_REQ));
      return false;
     }
   //--- If failed to add the request to the list, display the appropriate message,
   //--- remove the created object and return 'false'
   if(!this.m_list_request.Add(req_obj))
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_FAILING_CREATE_PENDING_REQ));
      delete req_obj;
      return false;
     }
   //--- Fill in the properties of a successfully created object by the values passed to the method
   req_obj.SetTimeActivate(symbol_obj.Time()+wait);
   req_obj.SetWaitingMSC(wait);
   req_obj.SetCurrentAttempt(0);
   req_obj.SetTotalAttempts(attempts);
   if(order!=NULL)
     {
      req_obj.SetActualVolume(order.Volume());
      req_obj.SetActualPrice(order.PriceOpen());
      req_obj.SetActualStopLimit(order.PriceStopLimit());
      req_obj.SetActualSL(order.StopLoss());
      req_obj.SetActualTP(order.TakeProfit());
      req_obj.SetActualTypeFilling(order.TypeFilling());
      req_obj.SetActualTypeTime(order.TypeTime());
      req_obj.SetActualExpiration(order.TimeExpiration());
     }
   else
     {
      req_obj.SetActualVolume(request.volume);
      req_obj.SetActualPrice(request.price);
      req_obj.SetActualStopLimit(request.stoplimit);
      req_obj.SetActualSL(request.sl);
      req_obj.SetActualTP(request.tp);
      req_obj.SetActualTypeFilling(request.type_filling);
      req_obj.SetActualTypeTime(request.type_time);
      req_obj.SetActualExpiration(request.expiration);
     }
   //--- Display a brief description of a created pending request
   if(this.m_log_level>LOG_LEVEL_NO_MSG)
     {
      ::Print(CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_CREATED)," #",req_obj.ID(),":");
      req_obj.PrintShort();
     }
   //--- successful
   return true;
  }
//+------------------------------------------------------------------+
```

Looking ahead (since we are now editing the base trading class), it should be noted that when we finalize the pending request object, we will
introduce the methods for setting its actual values, which will enable us to compare the object status - whether it has changed or whether the
request has already been activated. We compare the status of the order properties with the values set in the pending request object
properties. If they are completely equal to each other, we assume that the pending request has fulfilled its function.

We check what is passed as an order object in the method of
creating a pending trading request. If something other than NULL is passed (the order
object exists), the initial parameters from the order object are set in the pending
request object. Otherwise, the
parameters are set from the trading request structure.

**These are all the changes of the base trading class.**

Now let's improve the pending request object class.

We now have the new properties of the pending request object. Write them to the **Defines.mqh** file.

**Add the new properties to the pending request integer**
**properties and change the number of integer properties from 19 to 22:**

```
//+------------------------------------------------------------------+
//| Integer properties of a pending trading request                  |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_PROP_INTEGER
  {
   PEND_REQ_PROP_STATUS = 0,                                // Trading request status (from the ENUM_PEND_REQ_STATUS enumeration)
   PEND_REQ_PROP_TYPE,                                      // Trading request type (from the ENUM_PEND_REQ_TYPE enumeration)
   PEND_REQ_PROP_ID,                                        // Trading request ID
   PEND_REQ_PROP_RETCODE,                                   // Result a request is based on
   PEND_REQ_PROP_TIME_CREATE,                               // Request creation time
   PEND_REQ_PROP_TIME_ACTIVATE,                             // Next attempt activation time
   PEND_REQ_PROP_WAITING,                                   // Waiting time between requests
   PEND_REQ_PROP_CURRENT_ATTEMPT,                           // Current attempt index
   PEND_REQ_PROP_TOTAL,                                     // Number of attempts
   PEND_REQ_PROP_ACTUAL_TYPE_FILLING,                       // Actual order filling type
   PEND_REQ_PROP_ACTUAL_TYPE_TIME,                          // Actual order expiration type
   PEND_REQ_PROP_ACTUAL_EXPIRATION,                         // Actual order lifetime
   //--- MqlTradeRequest
   PEND_REQ_PROP_MQL_REQ_ACTION,                            // Type of a performed action in the request structure
   PEND_REQ_PROP_MQL_REQ_TYPE,                              // Order type in the request structure
   PEND_REQ_PROP_MQL_REQ_MAGIC,                             // EA stamp (magic number ID) in the request structure
   PEND_REQ_PROP_MQL_REQ_ORDER,                             // Order ticket in the request structure
   PEND_REQ_PROP_MQL_REQ_POSITION,                          // Position ticket in the request structure
   PEND_REQ_PROP_MQL_REQ_POSITION_BY,                       // Opposite position ticket in the request structure
   PEND_REQ_PROP_MQL_REQ_DEVIATION,                         // Maximum acceptable deviation from a requested price in the request structure
   PEND_REQ_PROP_MQL_REQ_EXPIRATION,                        // Order expiration time (for ORDER_TIME_SPECIFIED type orders) in the request structure
   PEND_REQ_PROP_MQL_REQ_TYPE_FILLING,                      // Order filling type in the request structure
   PEND_REQ_PROP_MQL_REQ_TYPE_TIME,                         // Order lifetime type in the request structure
  };
#define PEND_REQ_PROP_INTEGER_TOTAL (22)                    // Total number of integer event properties
#define PEND_REQ_PROP_INTEGER_SKIP  (0)                     // Number of request properties not used in sorting
//+------------------------------------------------------------------+
```

**Add the new properties and change**
**their number from 6 to 11:**

```
//+------------------------------------------------------------------+
//| Real properties of a pending trading request                     |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_PROP_DOUBLE
  {
   PEND_REQ_PROP_PRICE_CREATE = PEND_REQ_PROP_INTEGER_TOTAL,// Price at the moment of a request generation
   PEND_REQ_PROP_ACTUAL_VOLUME,                             // Actual volume
   PEND_REQ_PROP_ACTUAL_PRICE,                              // Actual order price
   PEND_REQ_PROP_ACTUAL_STOPLIMIT,                          // Actual stoplimit order price
   PEND_REQ_PROP_ACTUAL_SL,                                 // Actual stoploss order price
   PEND_REQ_PROP_ACTUAL_TP,                                 // Actual takeprofit order price
   //--- MqlTradeRequest
   PEND_REQ_PROP_MQL_REQ_VOLUME,                            // Requested volume of a deal in lots in the request structure
   PEND_REQ_PROP_MQL_REQ_PRICE,                             // Price in the request structure
   PEND_REQ_PROP_MQL_REQ_STOPLIMIT,                         // StopLimit level in the request structure
   PEND_REQ_PROP_MQL_REQ_SL,                                // Stop Loss level in the request structure
   PEND_REQ_PROP_MQL_REQ_TP,                                // Take Profit level in the request structure
  };
#define PEND_REQ_PROP_DOUBLE_TOTAL  (11)                    // Total number of event's real properties
#define PEND_REQ_PROP_DOUBLE_SKIP   (0)                     // Number of order properties not used in sorting
//+------------------------------------------------------------------+
```

**Add sorting by new properties to the list of possible sorting**
**criteria:**

```
//+------------------------------------------------------------------+
//| Possible pending request sorting criteria                        |
//+------------------------------------------------------------------+
#define FIRST_PREQ_DBL_PROP         (PEND_REQ_PROP_INTEGER_TOTAL-PEND_REQ_PROP_INTEGER_SKIP)
#define FIRST_PREQ_STR_PROP         (PEND_REQ_PROP_INTEGER_TOTAL-PEND_REQ_PROP_INTEGER_SKIP+PEND_REQ_PROP_DOUBLE_TOTAL-PEND_REQ_PROP_DOUBLE_SKIP)
enum ENUM_SORT_PEND_REQ_MODE
  {
//--- Sort by integer properties
   SORT_BY_PEND_REQ_STATUS = 0,                             // Sort by a trading request status (from the ENUM_PEND_REQ_STATUS enumeration)
   SORT_BY_PEND_REQ_TYPE,                                   // Sort by a trading request type (from the ENUM_PEND_REQ_TYPE enumeration)
   SORT_BY_PEND_REQ_ID,                                     // Sort by a trading request ID
   SORT_BY_PEND_REQ_RETCODE,                                // Sort by a result a request is based on
   SORT_BY_PEND_REQ_TIME_CREATE,                            // Sort by a request generation time
   SORT_BY_PEND_REQ_TIME_ACTIVATE,                          // Sort by next attempt activation time
   SORT_BY_PEND_REQ_WAITING,                                // Sort by a waiting time between requests
   SORT_BY_PEND_REQ_CURENT,                                 // Sort by the current attempt index
   SORT_BY_PEND_REQ_TOTAL,                                  // Sort by a number of attempts
   SORT_BY_PEND_REQ_ACTUAL_TYPE_FILLING,                    // Sort by actual order filling type
   SORT_BY_PEND_REQ_ACTUAL_TYPE_TIME,                       // Sort by actual order expiration type
   SORT_BY_PEND_REQ_ACTUAL_EXPIRATION,                      // Sort by actual order lifetime
   //--- MqlTradeRequest
   SORT_BY_PEND_REQ_MQL_REQ_ACTION,                         // Sort by a type of a performed action in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_TYPE,                           // Sort by an order type in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_MAGIC,                          // Sort by an EA stamp (magic number ID) in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_ORDER,                          // Sort by an order ticket in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_POSITION,                       // Sort by a position ticket in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_POSITION_BY,                    // Sort by an opposite position ticket in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_DEVIATION,                      // Sort by a maximum acceptable deviation from a requested price in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_EXPIRATION,                     // Sort by an order expiration time (for ORDER_TIME_SPECIFIED type orders) in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_TYPE_FILLING,                   // Sort by an order filling type in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_TYPE_TIME,                      // Sort by order lifetime type in the request structure
//--- Sort by real properties
   SORT_BY_PEND_REQ_PRICE_CREATE = FIRST_PREQ_DBL_PROP,     // Sort by a price at the moment of a request generation
   SORT_BY_PEND_REQ_ACTUAL_VOLUME,                          // Sort by initial volume
   SORT_BY_PEND_REQ_ACTUAL_PRICE,                           // Sort by actual order price
   SORT_BY_PEND_REQ_ACTUAL_STOPLIMIT,                       // Sort by actual stoplimit order price
   SORT_BY_PEND_REQ_ACTUAL_SL,                              // Sort by actual stoploss order price
   SORT_BY_PEND_REQ_ACTUAL_TP,                              // Sort by actual takeprofit order price
   //--- MqlTradeRequest
   SORT_BY_PEND_REQ_MQL_REQ_VOLUME,                         // Sort by a requested volume of a deal in lots in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_PRICE,                          // Sort by a price in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_STOPLIMIT,                      // Sort by StopLimit order level in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_SL,                             // Sort by StopLoss order level in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_TP,                             // Sort by TakeProfit order level in the request structure
//--- Sort by string properties
   //--- MqlTradeRequest
   SORT_BY_PEND_REQ_MQL_SYMBOL = FIRST_PREQ_STR_PROP,       // Sort by a trading instrument name in the request structure
   SORT_BY_PEND_REQ_MQL_COMMENT                             // Sort by an order comment in the request structure
  };
//+------------------------------------------------------------------+
```

Now let's focus on the objects.

Open the file of the abstract pending request base object class \\MQL5\\Include\\DoEasy\\Objects\\PendRequest\ **PendRequest.mqh**
and make the necessary changes.

**In the private section of the class, declare the pause class object,**
**while in the protected one,** **declare two overloaded methods of comparing object**
**property values with the value passed to the method and the set of methods**
**returning the flags indicating the pending request has completed changing each of the order/position parameters:**

```
//+------------------------------------------------------------------+
//| Abstract pending trading request class                           |
//+------------------------------------------------------------------+
class CPendRequest : public CObject
  {
private:
   MqlTradeRequest   m_request;                                         // Trade request structure
   CPause            m_pause;                                           // Pause class object
//--- Copy trading request data
   void              CopyRequest(const MqlTradeRequest &request);
//--- Return the index of the array the request (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_PEND_REQ_PROP_DOUBLE property)const { return(int)property-PEND_REQ_PROP_INTEGER_TOTAL;                               }
   int               IndexProp(ENUM_PEND_REQ_PROP_STRING property)const { return(int)property-PEND_REQ_PROP_INTEGER_TOTAL-PEND_REQ_PROP_DOUBLE_TOTAL;    }
protected:
   int               m_digits;                                          // Number of decimal places in a quote
   int               m_digits_lot;                                      // Number of decimal places in the symbol lot value
   bool              m_is_hedge;                                        // Hedging account flag
   long              m_long_prop[PEND_REQ_PROP_INTEGER_TOTAL];          // Request integer properties
   double            m_double_prop[PEND_REQ_PROP_DOUBLE_TOTAL];         // Request real properties
   string            m_string_prop[PEND_REQ_PROP_STRING_TOTAL];         // Request string properties
//--- Protected parametric constructor
                     CPendRequest(const ENUM_PEND_REQ_STATUS status,
                                  const uchar id,
                                  const double price,
                                  const ulong time,
                                  const MqlTradeRequest &request,
                                  const int retcode);
//--- Return (1) the magic number specified in the settings, (2) hedging account flag, (3) flag indicating the real property is equal to the value
   ushort            GetMagicID(void)                                      const { return ushort(this.GetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC) & 0xFFFF);}
   bool              IsHedge(void)                                         const { return this.m_is_hedge;                                               }
   bool              IsEqualByMode(const int mode,const double value)      const;
   bool              IsEqualByMode(const int mode,const long value)        const;
//--- Return the flags indicating the pending request has completed changing each of the order/position parameters
   bool              IsCompletedVolume(void)                               const;
   bool              IsCompletedPrice(void)                                const;
   bool              IsCompletedStopLimit(void)                            const;
   bool              IsCompletedStopLoss(void)                             const;
   bool              IsCompletedTakeProfit(void)                           const;
   bool              IsCompletedTypeFilling(void)                          const;
   bool              IsCompletedTypeTime(void)                             const;
   bool              IsCompletedExpiration(void)                           const;
public:
```

**In the public class section, declare the virtual method returning the flag**
**of the pending request work completion, the method returning the current**
**request object in full and the methods setting and returning the pause**
**object parameters:**

```
public:
//--- Default constructor
                     CPendRequest(){;}
//--- Set request (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_PEND_REQ_PROP_INTEGER property,long value) { this.m_long_prop[property]=value;                                     }
   void              SetProperty(ENUM_PEND_REQ_PROP_DOUBLE property,double value){ this.m_double_prop[this.IndexProp(property)]=value;                   }
   void              SetProperty(ENUM_PEND_REQ_PROP_STRING property,string value){ this.m_string_prop[this.IndexProp(property)]=value;                   }
//--- Return (1) integer, (2) real and (3) string request properties from the properties array
   long              GetProperty(ENUM_PEND_REQ_PROP_INTEGER property)      const { return this.m_long_prop[property];                                    }
   double            GetProperty(ENUM_PEND_REQ_PROP_DOUBLE property)       const { return this.m_double_prop[this.IndexProp(property)];                  }
   string            GetProperty(ENUM_PEND_REQ_PROP_STRING property)       const { return this.m_string_prop[this.IndexProp(property)];                  }

//--- Return the flag of the request supporting the property
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)        { return true; }
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)         { return true; }
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property)         { return true; }

//--- Return the flag indicating the pending request has completed its work
   virtual bool      IsCompleted(void)                                     const { return false;}
//--- Return itself
   CPendRequest     *GetObject(void)                                             { return &this;}

//--- Compare CPendRequest objects by a specified property (to sort the lists by a specified request object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CPendRequest objects by all properties (to search for equal request objects)
   bool              IsEqual(CPendRequest* compared_obj);
//--- Return (1) the elapsed number of milliseconds, (2) waiting time completion from the pause object,
//--- time of the (3) pause countdown start in milliseconds and (4) waiting time
   ulong             PausePassed(void)                                     const { return this.m_pause.Passed();                 }
   bool              PauseIsCompleted(void)                                const { return this.m_pause.IsCompleted();            }
   ulong             PauseTimeBegin(void)                                  const { return this.m_pause.TimeBegin();              }
   ulong             PauseTimeWait(void)                                   const { return this.m_pause.TimeWait();               }
//--- Return the description (1) of an elapsed number of pause waiting seconds,
//--- (2) pause countdown start time, as well as waiting time (3) in milliseconds and (4) in seconds
   string            PausePassedDescription(void)                          const { return this.m_pause.PassedDescription();      }
   string            PauseTimeBeginDescription(void)                       const { return this.m_pause.TimeBeginDescription();   }
   string            PauseWaitingMSCDescription(void)                      const { return this.m_pause.WaitingMSCDescription();  }
   string            PauseWaitingSecDescription(void)                      const { return this.m_pause.WaitingSECDescription();  }

//+------------------------------------------------------------------+
```

Here, the **IsCompleted()** method in the parent object always returns 'false' and is implemented for each descendant object
individually. The methods of working with the pause object simply call the corresponding methods of the object we examined at the very
beginning of the article.

**Add the methods returning the actual order properties to**
**the block of methods for simplified access to pending request object properties:**

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the request object properties  |
//+------------------------------------------------------------------+
//--- Return (1) request structure, (2) status, (3) type, (4) price at the moment of the request generation,
//--- (5) request generation time, (6) next attempt activation time,
//--- (7) waiting time between requests, (8) current attempt index,
//--- (9) number of attempts, (10) request ID
//--- (11) result a request is based on,
//--- (12) order ticket, (13) position ticket, (14) trading operation type
   MqlTradeRequest      MqlRequest(void)                                   const { return this.m_request;                                                }
   ENUM_PEND_REQ_STATUS Status(void)                                       const { return (ENUM_PEND_REQ_STATUS)this.GetProperty(PEND_REQ_PROP_STATUS);  }
   ENUM_PEND_REQ_TYPE   TypeRequest(void)                                  const { return (ENUM_PEND_REQ_TYPE)this.GetProperty(PEND_REQ_PROP_TYPE);      }
   double               PriceCreate(void)                                  const { return this.GetProperty(PEND_REQ_PROP_PRICE_CREATE);                  }
   ulong                TimeCreate(void)                                   const { return this.GetProperty(PEND_REQ_PROP_TIME_CREATE);                   }
   ulong                TimeActivate(void)                                 const { return this.GetProperty(PEND_REQ_PROP_TIME_ACTIVATE);                 }
   ulong                WaitingMSC(void)                                   const { return this.GetProperty(PEND_REQ_PROP_WAITING);                       }
   uchar                CurrentAttempt(void)                               const { return (uchar)this.GetProperty(PEND_REQ_PROP_CURRENT_ATTEMPT);        }
   uchar                TotalAttempts(void)                                const { return (uchar)this.GetProperty(PEND_REQ_PROP_TOTAL);                  }
   uchar                ID(void)                                           const { return (uchar)this.GetProperty(PEND_REQ_PROP_ID);                     }
   int                  Retcode(void)                                      const { return (int)this.GetProperty(PEND_REQ_PROP_RETCODE);                  }
   ulong                Order(void)                                        const { return this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER);                 }
   ulong                Position(void)                                     const { return this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION);              }
   ENUM_TRADE_REQUEST_ACTIONS Action(void)                                 const { return (ENUM_TRADE_REQUEST_ACTIONS)this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION);   }

//--- Return the actual (1) volume, (2) order, (3) limit order,
//--- (4) stoploss order and (5) takeprofit order prices, (6) order filling type,
//--- (7) order expiration type and (8) order lifetime
   double               ActualVolume(void)                                 const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_VOLUME);                 }
   double               ActualPrice(void)                                  const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_PRICE);                  }
   double               ActualStopLimit(void)                              const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_STOPLIMIT);              }
   double               ActualSL(void)                                     const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_SL);                     }
   double               ActualTP(void)                                     const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_TP);                     }
   ENUM_ORDER_TYPE_FILLING ActualTypeFilling(void)                         const { return (ENUM_ORDER_TYPE_FILLING)this.GetProperty(PEND_REQ_PROP_ACTUAL_TYPE_FILLING); }
   ENUM_ORDER_TYPE_TIME ActualTypeTime(void)                               const { return (ENUM_ORDER_TYPE_TIME)this.GetProperty(PEND_REQ_PROP_ACTUAL_TYPE_TIME);       }
   datetime             ActualExpiration(void)                             const { return (datetime)this.GetProperty(PEND_REQ_PROP_ACTUAL_EXPIRATION);   }
```

Each time the next pending request object is checked, the appropriate order/position properties the request object is created for are
written to these properties. These methods allow receiving the current properties of the appropriate order from the request object.

**In the same block, improve the method for setting the pending request**
**creation time and the one for setting its waiting time:**

```
//--- Set (1) the price when creating a request, (2) request creation time,
//--- (3) current attempt time, (4) waiting time between requests,
//--- (5) current attempt index, (6) number of attempts, (7) ID,
//--- (8) order ticket, (9) position ticket
   void                 SetPriceCreate(const double price)                       { this.SetProperty(PEND_REQ_PROP_PRICE_CREATE,price);                   }
   void                 SetTimeCreate(const ulong time)
                          {
                           this.SetProperty(PEND_REQ_PROP_TIME_CREATE,time);
                           this.m_pause.SetTimeBegin(time);
                          }
   void                 SetTimeActivate(const ulong time)                        { this.SetProperty(PEND_REQ_PROP_TIME_ACTIVATE,time);                   }
   void                 SetWaitingMSC(const ulong miliseconds)
                          {
                           this.SetProperty(PEND_REQ_PROP_WAITING,miliseconds);
                           this.m_pause.SetWaitingMSC(miliseconds);
                          }
```

In addition to setting the passed values to the pending request object,
these methods will now simultaneously set these values to the pause object as well.

**In the same block, write the methods for setting the actual order object**
**properties to the request object:**

```
//--- Set the actual (1) volume, (2) order, (3) limit order,
//--- (4) stoploss order and (5) takeprofit order prices, (6) order filling type,
//--- (7) order expiration type and (8) order lifetime
   void                 SetActualVolume(const double volume)                     { this.SetProperty(PEND_REQ_PROP_ACTUAL_VOLUME,volume);                 }
   void                 SetActualPrice(const double price)                       { this.SetProperty(PEND_REQ_PROP_ACTUAL_PRICE,price);                   }
   void                 SetActualStopLimit(const double price)                   { this.SetProperty(PEND_REQ_PROP_ACTUAL_STOPLIMIT,price);               }
   void                 SetActualSL(const double price)                          { this.SetProperty(PEND_REQ_PROP_ACTUAL_SL,price);                      }
   void                 SetActualTP(const double price)                          { this.SetProperty(PEND_REQ_PROP_ACTUAL_TP,price);                      }
   void                 SetActualTypeFilling(const ENUM_ORDER_TYPE_FILLING type) { this.SetProperty(PEND_REQ_PROP_ACTUAL_TYPE_FILLING,type);             }
   void                 SetActualTypeTime(const ENUM_ORDER_TYPE_TIME type)       { this.SetProperty(PEND_REQ_PROP_ACTUAL_TYPE_TIME,type);                }
   void                 SetActualExpiration(const datetime expiration)           { this.SetProperty(PEND_REQ_PROP_ACTUAL_EXPIRATION,expiration);         }

//+------------------------------------------------------------------+
```

**In the block of displaying object properties, add defining the**
**methods returning descriptions of actual order properties. At the very end, add**
**the virtual method returning the "header" (brief object description) describing the pending request object:**

```
//+------------------------------------------------------------------+
//| Descriptions of request object properties                        |
//+------------------------------------------------------------------+
//--- Get description of a request (1) integer, (2) real and (3) string property
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_INTEGER property);
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_DOUBLE property);
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_STRING property);

//--- Return the names of pending request object parameters
   string               StatusDescription(void)                const;
   string               TypeRequestDescription(void)           const;
   string               IDDescription(void)                    const;
   string               RetcodeDescription(void)               const;
   string               TimeCreateDescription(void)            const;
   string               TimeActivateDescription(void)          const;
   string               TimeWaitingDescription(void)           const;
   string               CurrentAttemptDescription(void)        const;
   string               TotalAttemptsDescription(void)         const;
   string               PriceCreateDescription(void)           const;

   string               TypeFillingActualDescription(void)     const;
   string               TypeTimeActualDescription(void)        const;
   string               ExpirationActualDescription(void)      const;
   string               VolumeActualDescription(void)          const;
   string               PriceActualDescription(void)           const;
   string               StopLimitActualDescription(void)       const;
   string               StopLossActualDescription(void)        const;
   string               TakeProfitActualDescription(void)      const;

//--- Return the names of trading request structures parameters in the request object
   string               MqlReqActionDescription(void)          const;
   string               MqlReqMagicDescription(void)           const;
   string               MqlReqOrderDescription(void)           const;
   string               MqlReqSymbolDescription(void)          const;
   string               MqlReqVolumeDescription(void)          const;
   string               MqlReqPriceDescription(void)           const;
   string               MqlReqStopLimitDescription(void)       const;
   string               MqlReqStopLossDescription(void)        const;
   string               MqlReqTakeProfitDescription(void)      const;
   string               MqlReqDeviationDescription(void)       const;
   string               MqlReqTypeOrderDescription(void)       const;
   string               MqlReqTypeFillingDescription(void)     const;
   string               MqlReqTypeTimeDescription(void)        const;
   string               MqlReqExpirationDescription(void)      const;
   string               MqlReqCommentDescription(void)         const;
   string               MqlReqPositionDescription(void)        const;
   string               MqlReqPositionByDescription(void)      const;

//--- Display (1) description of request properties (full_prop=true - all properties, false - only supported ones),
//--- (2) short message about the request, (3) short request name (2 and 3 - implementation in the class descendants)
   void                 Print(const bool full_prop=false);
   virtual void         PrintShort(void){;}
   virtual string       Header(void){return NULL;}
  };
//+------------------------------------------------------------------+
```

The methods of describing the actual order object properties are to be described later. The virtual method of the pending request header
description returns NULL here and is to be implemented in descendant objects.

**In the class constructor, add pause object initialization using the**
**values of the request object generation timeand its waiting time:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CPendRequest::CPendRequest(const ENUM_PEND_REQ_STATUS status,
                           const uchar id,
                           const double price,
                           const ulong time,
                           const MqlTradeRequest &request,
                           const int retcode)
  {
   this.CopyRequest(request);
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_digits=(int)::SymbolInfoInteger(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL),SYMBOL_DIGITS);
   int dg=(int)DigitsLots(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL));
   this.m_digits_lot=(dg==0 ? 1 : dg);
   this.SetProperty(PEND_REQ_PROP_STATUS,status);
   this.SetProperty(PEND_REQ_PROP_ID,id);
   this.SetProperty(PEND_REQ_PROP_RETCODE,retcode);
   this.SetProperty(PEND_REQ_PROP_TYPE,this.GetProperty(PEND_REQ_PROP_RETCODE)>0 ? PEND_REQ_TYPE_ERROR : PEND_REQ_TYPE_REQUEST);
   this.SetProperty(PEND_REQ_PROP_TIME_CREATE,time);
   this.SetProperty(PEND_REQ_PROP_PRICE_CREATE,price);
   this.m_pause.SetTimeBegin(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
   this.m_pause.SetWaitingMSC(this.GetProperty(PEND_REQ_PROP_WAITING));
  }
//+------------------------------------------------------------------+
```

Thus, immediately after creating a new pending request object, the waiting start time and pause duration are to be immediately set to its pause
object.

**Outside the class body, implement the methods returning the flags indicating the request object's real and integer properties are equal to**
**the value passed to the methods:**

```
//+---------------------------------------------------------------------+
//| Return the flag indicating the real value is equal to the passed one|
//+---------------------------------------------------------------------+
bool CPendRequest::IsEqualByMode(const int mode,const double value) const
  {
   CPendRequest *req=new CPendRequest();
   if(req==NULL)
      return false;
   req.SetProperty((ENUM_PEND_REQ_PROP_DOUBLE)mode,value);
   bool res=!this.Compare(req,mode);
   delete req;
   return res;
  }
//+------------------------------------------------------------------------+
//| Return the flag indicating the integer value is equal to the passed one|
//+------------------------------------------------------------------------+
bool CPendRequest::IsEqualByMode(const int mode,const long value) const
  {
   CPendRequest *req=new CPendRequest();
   if(req==NULL)
      return false;
   req.SetProperty((ENUM_PEND_REQ_PROP_INTEGER)mode,value);
   bool res=!this.Compare(req,mode);
   delete req;
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the mode used to compare the object property
and the value passed to the method.

The mode involves
indication of one of the object property values from the enumeration of its real (ENUM\_PEND\_REQ\_PROP\_DOUBLE) or integer properties
(ENUM\_PEND\_REQ\_PROP\_INTEGER). This is the property used in the comparison with the value passed to the method.

In the method, create
a new temporary pending request object. The corresponding
property receives the value passed to the method.


The [Compare() method of the CObject base object](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectcompare) of the
standard library returns 0, which means equality. However, the method is virtual and is implemented in the class descendants. Here, the
Compare() method is implemented so that it returns 1 if the current object value exceeds the same value of the compared one. -1 is returned if
the value is lower and 0 in case they are equal.

Thus, if the compared values are equal, the **res** value receives false.

However, we also
add the logical negation of a verification result there. Thus, if the comparison method returns 0, then !false
equals to true. In case of 1 or -1, !true
is equal to false.

After writing the comparison result, the
temporary object is removed and the comparison result is returned.

**The methods returning comparison flags of actual order and request object property values by their types:**

```
//+------------------------------------------------------------------+
//| Return the flag of a successful volume change                    |
//+------------------------------------------------------------------+
bool CPendRequest::IsCompletedVolume(void) const
  {
   return this.IsEqualByMode(PEND_REQ_PROP_ACTUAL_VOLUME,this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME));
  }
//+------------------------------------------------------------------+
//| Return the flag of a successful price modification               |
//+------------------------------------------------------------------+
bool CPendRequest::IsCompletedPrice(void) const
  {
   return this.IsEqualByMode(PEND_REQ_PROP_ACTUAL_PRICE,this.GetProperty(PEND_REQ_PROP_MQL_REQ_PRICE));
  }
//+-------------------------------------------------------------------+
//| Return the flag of a successful StopLimit order price modification|
//+-------------------------------------------------------------------+
bool CPendRequest::IsCompletedStopLimit(void) const
  {
   return this.IsEqualByMode(PEND_REQ_PROP_ACTUAL_STOPLIMIT,this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT));
  }
//+------------------------------------------------------------------+
//| Return the flag of a successful StopLoss order modification      |
//+------------------------------------------------------------------+
bool CPendRequest::IsCompletedStopLoss(void) const
  {
   return this.IsEqualByMode(PEND_REQ_PROP_ACTUAL_SL,this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL));
  }
//+------------------------------------------------------------------+
//| Return the flag of a successful TakeProfit order modification    |
//+------------------------------------------------------------------+
bool CPendRequest::IsCompletedTakeProfit(void) const
  {
   return this.IsEqualByMode(PEND_REQ_PROP_ACTUAL_TP,this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP));
  }
//+------------------------------------------------------------------+
//| Return the flag of a successful order filling type modification  |
//+------------------------------------------------------------------+
bool CPendRequest::IsCompletedTypeFilling(void) const
  {
   return this.IsEqualByMode(PEND_REQ_PROP_ACTUAL_TYPE_FILLING,this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE_FILLING));
  }
//+-------------------------------------------------------------------+
//| Return the flag of a successful order expiration type modification|
//+-------------------------------------------------------------------+
bool CPendRequest::IsCompletedTypeTime(void) const
  {
   return this.IsEqualByMode(PEND_REQ_PROP_ACTUAL_TYPE_TIME,this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE_TIME));
  }
//+------------------------------------------------------------------+
//| Return the flag of a successful order lifetime modification      |
//+------------------------------------------------------------------+
bool CPendRequest::IsCompletedExpiration(void) const
  {
   return this.IsEqualByMode(PEND_REQ_PROP_ACTUAL_EXPIRATION,this.GetProperty(PEND_REQ_PROP_MQL_REQ_EXPIRATION));
  }
//+------------------------------------------------------------------+
```

The methods return the result of comparing two appropriate properties — thw actual and the one set in the object using the comparison method we
have just considered.

**Add the new properties into the implementation of methods returning the descriptions of object properties:**

```
//+------------------------------------------------------------------+
//| Return the description of a request integer property             |
//+------------------------------------------------------------------+
string CPendRequest::GetPropertyDescription(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   return
     (
      property==PEND_REQ_PROP_STATUS               ?  this.StatusDescription()            :
      property==PEND_REQ_PROP_TYPE                 ?  this.TypeRequestDescription()       :
      property==PEND_REQ_PROP_ID                   ?  this.IDDescription()                :
      property==PEND_REQ_PROP_RETCODE              ?  this.RetcodeDescription()           :
      property==PEND_REQ_PROP_TIME_CREATE          ?  this.TimeCreateDescription()        :
      property==PEND_REQ_PROP_TIME_ACTIVATE        ?  this.TimeActivateDescription()      :
      property==PEND_REQ_PROP_WAITING              ?  this.TimeWaitingDescription()       :
      property==PEND_REQ_PROP_CURRENT_ATTEMPT      ?  this.CurrentAttemptDescription()    :
      property==PEND_REQ_PROP_TOTAL                ?  this.TotalAttemptsDescription()     :
      property==PEND_REQ_PROP_ACTUAL_TYPE_FILLING  ?  this.TypeFillingActualDescription() :
      property==PEND_REQ_PROP_ACTUAL_TYPE_TIME     ?  this.TypeTimeActualDescription()    :
      property==PEND_REQ_PROP_ACTUAL_EXPIRATION    ?  this.ExpirationActualDescription()  :
      //--- MqlTradeRequest
      property==PEND_REQ_PROP_MQL_REQ_ACTION       ?  this.MqlReqActionDescription()      :
      property==PEND_REQ_PROP_MQL_REQ_TYPE         ?  this.MqlReqTypeOrderDescription()   :
      property==PEND_REQ_PROP_MQL_REQ_MAGIC        ?  this.MqlReqMagicDescription()       :
      property==PEND_REQ_PROP_MQL_REQ_ORDER        ?  this.MqlReqOrderDescription()       :
      property==PEND_REQ_PROP_MQL_REQ_POSITION     ?  this.MqlReqPositionDescription()    :
      property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  ?  this.MqlReqPositionByDescription()  :
      property==PEND_REQ_PROP_MQL_REQ_DEVIATION    ?  this.MqlReqDeviationDescription()   :
      property==PEND_REQ_PROP_MQL_REQ_EXPIRATION   ?  this.MqlReqExpirationDescription()  :
      property==PEND_REQ_PROP_MQL_REQ_TYPE_FILLING ?  this.MqlReqTypeFillingDescription() :
      property==PEND_REQ_PROP_MQL_REQ_TYPE_TIME    ?  this.MqlReqTypeTimeDescription()    :
      ::EnumToString(property)
     );
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Return the description of the request real property              |
//+------------------------------------------------------------------+
string CPendRequest::GetPropertyDescription(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   return
     (
      property==PEND_REQ_PROP_PRICE_CREATE      ?  this.PriceCreateDescription()          :
      property==PEND_REQ_PROP_ACTUAL_VOLUME     ?  this.VolumeActualDescription()         :
      property==PEND_REQ_PROP_ACTUAL_PRICE      ?  this.PriceActualDescription()          :
      property==PEND_REQ_PROP_ACTUAL_STOPLIMIT  ?  this.StopLimitActualDescription()      :
      property==PEND_REQ_PROP_ACTUAL_SL         ?  this.StopLossActualDescription()       :
      property==PEND_REQ_PROP_ACTUAL_TP         ?  this.TakeProfitActualDescription()     :
      //--- MqlTradeRequest
      property==PEND_REQ_PROP_MQL_REQ_VOLUME    ?  this.MqlReqVolumeDescription()         :
      property==PEND_REQ_PROP_MQL_REQ_PRICE     ?  this.MqlReqPriceDescription()          :
      property==PEND_REQ_PROP_MQL_REQ_STOPLIMIT ?  this.MqlReqStopLimitDescription()      :
      property==PEND_REQ_PROP_MQL_REQ_SL        ?  this.MqlReqStopLossDescription()       :
      property==PEND_REQ_PROP_MQL_REQ_TP        ?  this.MqlReqTakeProfitDescription()     :
      ::EnumToString(property)
     );
  }
//+------------------------------------------------------------------+
```

The object property passed to it is checked in the methods and its description is returned using the appropriate methods.

**The methods returning descriptions of actual order properties written in the request object:**

```
//+------------------------------------------------------------------+
//| Return the description of the actual order filling mode          |
//+------------------------------------------------------------------+
string CPendRequest::TypeFillingActualDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_TYPE_FILLING)+": "+OrderTypeFillingDescription((ENUM_ORDER_TYPE_FILLING)this.GetProperty(PEND_REQ_PROP_ACTUAL_TYPE_FILLING));
  }
//+------------------------------------------------------------------+
//| Return the actual order lifetime type                            |
//+------------------------------------------------------------------+
string CPendRequest::TypeTimeActualDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_TYPE_TIME)+": "+OrderTypeTimeDescription((ENUM_ORDER_TYPE_TIME)this.GetProperty(PEND_REQ_PROP_ACTUAL_TYPE_TIME));
  }
//+------------------------------------------------------------------+
//| Return the description of the actual order lifetime              |
//+------------------------------------------------------------------+
string CPendRequest::ExpirationActualDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_EXPIRATION)+": "+
          (this.GetProperty(PEND_REQ_PROP_ACTUAL_EXPIRATION)>0 ?
           ::TimeToString(this.GetProperty(PEND_REQ_PROP_ACTUAL_EXPIRATION)) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Return the description of the actual order volume                |
//+------------------------------------------------------------------+
string CPendRequest::VolumeActualDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_VOLUME)+": "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_ACTUAL_VOLUME),this.m_digits_lot);
  }
//+------------------------------------------------------------------+
//| Return the description of the actual order price                 |
//+------------------------------------------------------------------+
string CPendRequest::PriceActualDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_PRICE)+": "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_ACTUAL_PRICE),this.m_digits);
  }
//+------------------------------------------------------------------+
//| Return the description of the actual StopLimit order price       |
//+------------------------------------------------------------------+
string CPendRequest::StopLimitActualDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_STOPLIMIT)+": "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_ACTUAL_STOPLIMIT),this.m_digits);
  }
//+------------------------------------------------------------------+
//| Return the description of the actual StopLoss order price        |
//+------------------------------------------------------------------+
string CPendRequest::StopLossActualDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_SL)+": "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_ACTUAL_SL),this.m_digits);
  }
//+------------------------------------------------------------------+
//| Return the description of the actual TakeProfit order price      |
//+------------------------------------------------------------------+
string CPendRequest::TakeProfitActualDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_TP)+": "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_ACTUAL_TP),this.m_digits);
  }
//+------------------------------------------------------------------+
```

The methods are used to prepare and return the message text corresponding to the property returned by the method.

**These are all the changes of the abstract pending request's base object.**

Now let's make changes to descendant classes of the base request object.

Open the file of the class of position opening pending request object \\MQL5\\Include\\DoEasy\\Objects\\PendRequest\ **PendReqOpen.mqh**
and make changes.

**Add defining the virtual method returning the short request name:**

```
//+------------------------------------------------------------------+
//| Pending request for opening a position                           |
//+------------------------------------------------------------------+
class CPendReqOpen : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqOpen(const uchar id,
                                  const double price,
                                  const ulong time,
                                  const MqlTradeRequest &request,
                                  const int retcode) : CPendRequest(PEND_REQ_STATUS_OPEN,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data and (2) a short request name in the journal
   virtual void      PrintShort(void);
   virtual string    Header(void);
  };
//+------------------------------------------------------------------+
```

**And its implementation:**

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqOpen::Header(void)
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_OPEN)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The class of the pending request for modifying position stop orders (PendReqSLTP.mqh file):**

```
//+------------------------------------------------------------------+
//| Pending request to modify position stop orders                   |
//+------------------------------------------------------------------+
class CPendReqSLTP : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqSLTP(const uchar id,
                                  const double price,
                                  const ulong time,
                                  const MqlTradeRequest &request,
                                  const int retcode) : CPendRequest(PEND_REQ_STATUS_SLTP,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Return the flag indicating the pending request has completed its work
   virtual bool      IsCompleted(void) const;
//--- Display a brief message with request data and (2) a short request name in the journal
   virtual void      PrintShort(void);
   virtual string    Header(void);
  };
//+------------------------------------------------------------------+
```

Add the new properties to the methods returning the flag indicating the object supports some of its properties:

```
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqSLTP::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if(property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  ||
      property==PEND_REQ_PROP_MQL_REQ_ORDER        ||
      property==PEND_REQ_PROP_MQL_REQ_EXPIRATION   ||
      property==PEND_REQ_PROP_MQL_REQ_DEVIATION    ||
      property==PEND_REQ_PROP_MQL_REQ_TYPE_FILLING ||
      property==PEND_REQ_PROP_MQL_REQ_TYPE_TIME    ||
      property==PEND_REQ_PROP_ACTUAL_EXPIRATION    ||
      property==PEND_REQ_PROP_ACTUAL_TYPE_TIME     ||
      property==PEND_REQ_PROP_ACTUAL_TYPE_FILLING

     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CPendReqSLTP::SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   if(property==PEND_REQ_PROP_PRICE_CREATE         ||
      property==PEND_REQ_PROP_ACTUAL_SL            ||
      property==PEND_REQ_PROP_ACTUAL_TP            ||
      property==PEND_REQ_PROP_MQL_REQ_SL           ||
      property==PEND_REQ_PROP_MQL_REQ_TP
     ) return true;
   return false;
  }
//+------------------------------------------------------------------+
```

The virtual method returning the flag indicating the pending request has completed its work:

```
//+----------------------------------------------------------------------+
//| Return the flag indicating the pending request has completed its work|
//+----------------------------------------------------------------------+
bool CPendReqSLTP::IsCompleted(void) const
  {
   bool res=true;
   res &= this.IsCompletedStopLoss();
   res &= this.IsCompletedTakeProfit();
   return res;
  }
//+------------------------------------------------------------------+
```

Here, add the results of verifying StopLoss and TakeProfit modification completion to the **res** variable. If at least one of
the verifications returns false, the overall result is false.
Return the final result from the method.

The method returning a short request name:

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqSLTP::Header(void)
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_SLTP)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The class of the pending request for closing a position (PendReqClose.mqh file):**

```
//+------------------------------------------------------------------+
//| Pending request to close a position                              |
//+------------------------------------------------------------------+
class CPendReqClose : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqClose(const uchar id,
                                   const double price,
                                   const ulong time,
                                   const MqlTradeRequest &request,
                                   const int retcode) : CPendRequest(PEND_REQ_STATUS_CLOSE,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Return the flag indicating the pending request has completed its work
   virtual bool      IsCompleted(void) const;
//--- Display a brief message with request data and (2) a short request name in the journal
   virtual void      PrintShort(void);
   virtual string    Header(void);
  };
//+------------------------------------------------------------------+
```

The method returning the flag indicating the request has completed its work:

```
//+----------------------------------------------------------------------+
//| Return the flag indicating the pending request has completed its work|
//+----------------------------------------------------------------------+
bool CPendReqClose::IsCompleted(void) const
  {
   return this.IsCompletedVolume();
  }
//+------------------------------------------------------------------+
```

The method returning a short request name:

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqClose::Header(void)
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_CLOSE)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The class of the pending request for placing a pending order (PendReqPlace.mqh file):**

```
//+------------------------------------------------------------------+
//| Pending request to place a pending order                         |
//+------------------------------------------------------------------+
class CPendReqPlace : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqPlace(const uchar id,
                                   const double price,
                                   const ulong time,
                                   const MqlTradeRequest &request,
                                   const int retcode) : CPendRequest(PEND_REQ_STATUS_PLACE,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data and (2) a short request name in the journal
   virtual void      PrintShort(void);
   virtual string    Header(void);
  };
//+------------------------------------------------------------------+
```

The method returning a short request name:

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqPlace::Header(void)
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_PLACE)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The class of the pending request for changing pending order parameters (PendReqModify.mqh file):**

```
//+------------------------------------------------------------------+
//| Pending request to modify pending order parameters               |
//+------------------------------------------------------------------+
class CPendReqModify : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqModify(const uchar id,
                                    const double price,
                                    const ulong time,
                                    const MqlTradeRequest &request,
                                    const int retcode) : CPendRequest(PEND_REQ_STATUS_MODIFY,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Return the flag indicating the pending request has completed its work
   virtual bool      IsCompleted(void) const;
//--- Display a brief message with request data and (2) a short request name in the journal
   virtual void      PrintShort(void);
   virtual string    Header(void);
  };
//+------------------------------------------------------------------+
```

The method returning the flag indicating the request has completed its work:

```
//+----------------------------------------------------------------------+
//| Return the flag indicating the pending request has completed its work|
//+----------------------------------------------------------------------+
bool CPendReqModify::IsCompleted(void) const
  {
   bool res=true;
   res &= this.IsCompletedPrice();
   res &= this.IsCompletedStopLimit();
   res &= this.IsCompletedStopLoss();
   res &= this.IsCompletedTakeProfit();
   res &= this.IsCompletedTypeFilling();
   res &= this.IsCompletedTypeTime();
   res &= this.IsCompletedExpiration();
   return res;
  }
//+------------------------------------------------------------------+
```

The method returning a short request name:

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqModify::Header(void)
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_MODIFY)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The class of a pending request for removing a pending order (PendReqRemove.mqh file):**

```
//+------------------------------------------------------------------+
//| Pending request to remove a pending order                        |
//+------------------------------------------------------------------+
class CPendReqRemove : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqRemove(const uchar id,
                                    const double price,
                                    const ulong time,
                                    const MqlTradeRequest &request,
                                    const int retcode) : CPendRequest(PEND_REQ_STATUS_REMOVE,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data and (2) a short request name in the journal
   virtual void      PrintShort(void);
   virtual string    Header(void);
  };
//+------------------------------------------------------------------+
```

The method returning a short request name:

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqRemove::Header(void)
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_REMOVE)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

As we can see in the class listings and their appropriate methods, all is quite trivial here — each method uses only its inherent abstract
pending request object properties. A short request name consists of the appropriate text messages.

**This concludes the improvement of the descendant classes of the abstract pending trading request object's base class.**

### The class for managing pending request objects

Now let’s finally get down to the class for managing pending request objects.

As already mentioned above, the class is to be derived from the CTrading trading class since all data and methods are closely interrelated in
the CTrading, CPendRequest and newly added CTradingControl classes. In the current implementation, the class is quite small and works in
the timer. The entire code is to be moved to the timer from the CTrading class timer code with minor changes.

Create the new CTradingControl class in \\MQL5\\Include\\DoEasy\ **TradingControl.mqh**. Set
the CTrading class as a base class when creating it. Make sure to include the
Trading.mqh file to it right after creating the class:

```
//+------------------------------------------------------------------+
//|                                               PendReqControl.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Trading.mqh"
//+------------------------------------------------------------------+
//| Class for managing pending trading requests                      |
//+------------------------------------------------------------------+
class CTradingControl : public CTrading
  {
private:
//--- Set actual order/position data to a pending request object
   void                 SetActualProperties(CPendRequest *req_obj,const COrder *order);
public:
//--- Return itself
   CTradingControl     *GetObject(void)            { return &this;   }
//--- Timer
   virtual void         OnTimer(void);
//--- Constructor
                        CTradingControl();
  };
//+------------------------------------------------------------------+
```

The class features the **SetActualProperties()** private method setting actual order/position data to a pending
request object.

In the public section, the **GetObject()** method returns the pointer to the entire object of managing
pending requests, while the **OnTimer()** virtual method is a timer of the class for managing pending requests.

**In the class constructor, clear the list of pending requests**
**and set the sorted list flag for it:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTradingControl::CTradingControl()
  {
   this.m_list_request.Clear();
   this.m_list_request.Sort();
  }
//+------------------------------------------------------------------+
```

The list belongs to the CTrading class since this is the trading class where pending trading requests are created.

**Implement the method for setting actual order data to a request object beyond the class body:**

```
//+------------------------------------------------------------------+
//| Set order/position data to a pending request object              |
//+------------------------------------------------------------------+
void CTradingControl::SetActualProperties(CPendRequest *req_obj,const COrder *order)
  {
   req_obj.SetActualExpiration(order.TimeExpiration());
   req_obj.SetActualPrice(order.PriceOpen());
   req_obj.SetActualSL(order.StopLoss());
   req_obj.SetActualStopLimit(order.PriceStopLimit());
   req_obj.SetActualTP(order.TakeProfit());
   req_obj.SetActualTypeFilling(order.TypeFilling());
   req_obj.SetActualTypeTime(order.TypeTime());
   req_obj.SetActualVolume(order.Volume());
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to a pending request object
and the pointer to an order object. Next, the appropriate order
property values are set in the request object using the methods considered above.

The class timer tracks all existing request objects in the pending request list. A lifetime of each pending request is checked. If it is over,
the request is removed. If the request has already been activated (there is an appropriate trading event in the account history or an
order/position with a specific pending request ID in the magic number is present), such requests are considered completely executed and
removed from the list.

If the request is not activated yet and its activation time has come, a trading order set in the pending request is
sent to the server.

The method features the verification for the ability to perform trading operations on the terminal side — the AutoTrading button and
Allow Auto Trading option in the EA settings. If a pending trading request has been created according to the 10027 error code (auto trading is
disabled by the terminal) and the error cause has been eliminated by a trader (by clicking the AutoTrading button or checking the Allow Auto
Trading option in the EA settings), the pending request is handled immediately. A new pending request activation time is set so that the
activation is performed since the user has fixed the error, and there is no need to wait. Instead, the order should be sent to the server
immediately to avoid subsequent requotes.

Since the method is quite large, I did my best to thoroughly describe in the code comments all the actions conducted inside the timer of the class
for managing pending requests.

**The timer of the class for managing pending requests:**

```
//+------------------------------------------------------------------+
//| Timer                                                            |
//+------------------------------------------------------------------+
void CTradingControl::OnTimer(void)
  {
   //--- In a loop by the list of pending requests
   int total=this.m_list_request.Total();
   for(int i=total-1;i>WRONG_VALUE;i--)
     {
      //--- receive the next request object
      CPendRequest *req_obj=this.m_list_request.At(i);
      if(req_obj==NULL)
         continue;

      //--- get the request structure and the symbol object a trading operation should be performed for
      MqlTradeRequest request=req_obj.MqlRequest();
      CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(request.symbol);
      if(symbol_obj==NULL || !symbol_obj.RefreshRates())
         continue;

      //--- Set the flag disabling trading in the terminal by two properties simultaneously
      //--- (the AutoTrading button in the terminal and the Allow Automated Trading option in the EA settings)
      //--- If any of the two properties is 'false', the flag is 'false' as well
      bool terminal_trade_allowed=::TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
      terminal_trade_allowed &=::MQLInfoInteger(MQL_TRADE_ALLOWED);
      //--- If a request object is based on the error code
      if(req_obj.TypeRequest()==PEND_REQ_TYPE_ERROR)
        {
         //--- if the error has been caused by trading disabled on the terminal side and has been eliminated
         if(req_obj.Retcode()==10027 && terminal_trade_allowed)
           {
            //--- if the current attempt has not exceeded the defined number of trading attempts yet
            if(req_obj.CurrentAttempt()<req_obj.TotalAttempts()+1)
              {
               //--- Set the request creation time equal to its creation time minus waiting time, i.e. send the request immediately
               //--- Also, decrease the number of a successful attempt since during the next attempt, its number is increased, and if this is the last attempt,
               //--- it is not executed. However, this is related to fixing the error cause by a user, which means we need to give more time for the last attempt
               req_obj.SetTimeCreate(req_obj.TimeCreate()-req_obj.WaitingMSC());
               req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()>0 ? req_obj.CurrentAttempt()-1 : 0));
              }
           }
        }

      //--- if the current attempt exceeds the defined number of trading attempts,
      //--- or the current time exceeds the waiting time of all attempts
      //--- remove the current request object and move on to the next one
      if(req_obj.CurrentAttempt()>req_obj.TotalAttempts() || req_obj.CurrentAttempt()>=UCHAR_MAX ||
         (long)symbol_obj.Time()>long(req_obj.TimeCreate()+req_obj.WaitingMSC()*req_obj.TotalAttempts()))
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_DELETED));
         this.m_list_request.Delete(i);
         continue;
        }

      //--- If this is a position opening or placing a pending order
      if((req_obj.Action()==TRADE_ACTION_DEAL && req_obj.Position()==0) || req_obj.Action()==TRADE_ACTION_PENDING)
        {
         //--- Get the pending request ID
         uchar id=this.GetPendReqID((uint)request.magic);
         //--- Get the list of orders/positions containing the order/position with the pending request ID
         CArrayObj *list=this.m_market.GetList(ORDER_PROP_PEND_REQ_ID,id,EQUAL);
         if(::CheckPointer(list)==POINTER_INVALID)
            continue;
         //--- If the order/position is present, the request is handled: remove it and proceed to the next
         if(list.Total()>0)
           {
            if(this.m_log_level>LOG_LEVEL_NO_MSG)
               ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
            this.m_list_request.Delete(i);
            continue;
           }
        }
      //--- Otherwise: full and partial position closure, removing an order, modifying order parameters and position stop orders
      else
        {
         CArrayObj *list=NULL;
         //--- if this is a position closure, including a closure by an opposite one
         if((req_obj.Action()==TRADE_ACTION_DEAL && req_obj.Position()>0) || req_obj.Action()==TRADE_ACTION_CLOSE_BY)
           {
            //--- Get a position with the necessary ticket from the list of open positions
            list=this.m_market.GetList(ORDER_PROP_TICKET,req_obj.Position(),EQUAL);
            if(::CheckPointer(list)==POINTER_INVALID)
               continue;
            //--- If the market has no such position - the request is handled: remove it and proceed to the next one
            if(list.Total()==0)
              {
               if(this.m_log_level>LOG_LEVEL_NO_MSG)
                  ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
               this.m_list_request.Delete(i);
               continue;
              }
            //--- Otherwise, if the position still exists, this is a partial closure
            else
              {
               //--- Get the list of all account trading events
               list=this.m_events.GetList();
               if(list==NULL)
                  continue;
               //--- In the loop from the end of the account trading event list
               int events_total=list.Total();
               for(int j=events_total-1; j>WRONG_VALUE; j--)
                 {
                  //--- get the next trading event
                  CEvent *event=list.At(j);
                  if(event==NULL)
                     continue;
                  //--- If this event is a partial closure or there was a partial closure when closing by an opposite one
                  if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL || event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS)
                    {
                     //--- If a position ticket in a trading event coincides with the ticket in a pending trading request
                     if(event.PositionID()==req_obj.Position())
                       {
                        //--- Get a position object from the list of market positions
                        CArrayObj *list_orders=this.m_market.GetList(ORDER_PROP_POSITION_ID,req_obj.Position(),EQUAL);
                        if(list_orders==NULL || list_orders.Total()==0)
                           break;
                        COrder *order=list_orders.At(list_orders.Total()-1);
                        if(order==NULL)
                           break;
                        //--- Set actual position data to the pending request object
                        this.SetActualProperties(req_obj,order);
                        //--- If (executed request volume + unexecuted request volume) is equal to the requested volume in a pending request -
                        //--- the request is handled: remove it and break the loop by the list of account trading events
                        if(req_obj.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME)==event.VolumeOrderExecuted()+event.VolumeOrderCurrent())
                          {
                           if(this.m_log_level>LOG_LEVEL_NO_MSG)
                              ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                           this.m_list_request.Delete(i);
                           break;
                          }
                       }
                    }
                 }
               //--- If a handled pending request object was removed by the trading event list in the loop, move on to the next one
               if(::CheckPointer(req_obj)==POINTER_INVALID)
                  continue;
              }
           }
         //--- If this is a modification of position stop orders
         if(req_obj.Action()==TRADE_ACTION_SLTP)
           {
            //--- Get the list of all account trading events
            list=this.m_events.GetList();
            if(list==NULL)
               continue;
            //--- In the loop from the end of the account trading event list
            int events_total=list.Total();
            for(int j=events_total-1; j>WRONG_VALUE; j--)
              {
               //--- get the next trading event
               CEvent *event=list.At(j);
               if(event==NULL)
                  continue;
               //--- If this is a change of the position's stop orders
               if(event.TypeEvent()>TRADE_EVENT_MODIFY_ORDER_TAKE_PROFIT)
                 {
                  //--- If a position ticket in a trading event coincides with the ticket in a pending trading request
                  if(event.PositionID()==req_obj.Position())
                    {
                     //--- Get a position object from the list of market positions
                     CArrayObj *list_orders=this.m_market.GetList(ORDER_PROP_POSITION_ID,req_obj.Position(),EQUAL);
                     if(list_orders==NULL || list_orders.Total()==0)
                        break;
                     COrder *order=list_orders.At(list_orders.Total()-1);
                     if(order==NULL)
                        break;
                     //--- Set actual position data to the pending request object
                     this.SetActualProperties(req_obj,order);
                     //--- If all modifications have worked out -
                     //--- the request is handled: remove it and break the loop by the list of account trading events
                     if(req_obj.IsCompleted())
                       {
                        if(this.m_log_level>LOG_LEVEL_NO_MSG)
                           ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                        this.m_list_request.Delete(i);
                        break;
                       }
                    }
                 }
              }
            //--- If a handled pending request object was removed by the trading event list in the loop, move on to the next one
            if(::CheckPointer(req_obj)==POINTER_INVALID)
               continue;
           }
         //--- If this is a pending order removal
         if(req_obj.Action()==TRADE_ACTION_REMOVE)
           {
            //--- Get the list of removed pending orders from the historical list
            list=this.m_history.GetList(ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_PENDING,EQUAL);
            if(::CheckPointer(list)==POINTER_INVALID)
               continue;
            //--- Leave a single order with the necessary ticket in the list
            list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,req_obj.Order(),EQUAL);
            //--- If the order is present, the request is handled: remove it and proceed to the next
            if(list.Total()>0)
              {
               if(this.m_log_level>LOG_LEVEL_NO_MSG)
                  ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
               this.m_list_request.Delete(i);
               continue;
              }
           }
         //--- If this is a pending order modification
         if(req_obj.Action()==TRADE_ACTION_MODIFY)
           {
            //--- Get the list of all account trading events
            list=this.m_events.GetList();
            if(list==NULL)
               continue;
            //--- In the loop from the end of the account trading event list
            int events_total=list.Total();
            for(int j=events_total-1; j>WRONG_VALUE; j--)
              {
               //--- get the next trading event
               CEvent *event=list.At(j);
               if(event==NULL)
                  continue;
               //--- If this event involves any change of modified pending order parameters
               if(event.TypeEvent()>TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER && event.TypeEvent()<TRADE_EVENT_MODIFY_POSITION_STOP_LOSS_TAKE_PROFIT)
                 {
                  //--- If an order ticket in a trading event coincides with the ticket in a pending trading request
                  if(event.TicketOrderEvent()==req_obj.Order())
                    {
                     //--- Get an order object from the list
                     CArrayObj *list_orders=this.m_market.GetList(ORDER_PROP_TICKET,req_obj.Order(),EQUAL);
                     if(list_orders==NULL || list_orders.Total()==0)
                        break;
                     COrder *order=list_orders.At(0);
                     if(order==NULL)
                        break;
                     //--- Set actual order data to the pending request object
                     this.SetActualProperties(req_obj,order);
                     //--- If all modifications have worked out -
                     //--- the request is handled: remove it and break the loop by the list of account trading events
                     if(req_obj.IsCompleted())
                       {
                        if(this.m_log_level>LOG_LEVEL_NO_MSG)
                           ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                        this.m_list_request.Delete(i);
                        break;
                       }
                    }
                 }
              }
           }
        }

      //--- Exit if the pending request object has been removed after checking its operation
      if(::CheckPointer(req_obj)==POINTER_INVALID)
         return;
      //--- Set the request activation time in the request object
      req_obj.SetTimeActivate(req_obj.TimeCreate()+req_obj.WaitingMSC()*(req_obj.CurrentAttempt()+1));

      //--- If the current time is less than the request activation time,
      //--- this is not the request time - move on to the next request in the list
      if((long)symbol_obj.Time()<(long)req_obj.TimeActivate())
         continue;

      //--- Set the attempt number in the request object
      req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()+1));

      //--- Display the number of a trading attempt in the journal

      if(this.m_log_level>LOG_LEVEL_NO_MSG)
        {
         ::Print(CMessage::Text(MSG_LIB_TEXT_RE_TRY_N)+(string)req_obj.CurrentAttempt()+":");
         req_obj.PrintShort();
        }

      //--- Depending on the type of action performed in the trading request
      switch(request.action)
        {
         //--- Opening/closing a position
         case TRADE_ACTION_DEAL :
            //--- If no ticket is present in the request structure - this is opening a position
            if(request.position==0)
               this.OpenPosition((ENUM_POSITION_TYPE)request.type,request.volume,request.symbol,request.magic,request.sl,request.tp,request.comment,request.deviation,request.type_filling);
            //--- If the ticket is present in the request structure - this is a position closure
            else
               this.ClosePosition(request.position,request.volume,request.comment,request.deviation);
            break;
         //--- Modify StopLoss/TakeProfit position
         case TRADE_ACTION_SLTP :
            this.ModifyPosition(request.position,request.sl,request.tp);
            break;
         //--- Close by an opposite one
         case TRADE_ACTION_CLOSE_BY :
            this.ClosePositionBy(request.position,request.position_by);
            break;
         //---
         //--- Place a pending order
         case TRADE_ACTION_PENDING :
            this.PlaceOrder(request.type,request.volume,request.symbol,request.price,request.stoplimit,request.sl,request.tp,request.magic,request.comment,request.expiration,request.type_time,request.type_filling);
            break;
         //--- Modify a pending order
         case TRADE_ACTION_MODIFY :
            this.ModifyOrder(request.order,request.price,request.sl,request.tp,request.stoplimit,request.expiration,request.type_time,request.type_filling);
            break;
         //--- Remove a pending order
         case TRADE_ACTION_REMOVE :
            this.DeleteOrder(request.order);
            break;
         //---
         default:
            break;
        }
     }
  }
//+------------------------------------------------------------------+
```

For now, this is all we need to do in the class for managing pending trading requests. If you have any questions regarding the class timer
operation, ask them in the comments.

Now let's make changes in the **CEngine** library base object class.

Since now a descendant (the class for managing pending requests) is used rather than the trading class itself, **instead of**
**including in the trading class file listing:**

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Services\TimerCounter.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\ResourceCollection.mqh"
#include "Trading.mqh"
//+------------------------------------------------------------------+
```

**include the file of the class for managing pending trading requests:**

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Services\TimerCounter.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\ResourceCollection.mqh"
#include "TradingControl.mqh"
//+------------------------------------------------------------------+
```

**While instead of the trading class object**

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CSymbolsCollection   m_symbols;                       // Symbol collection
   CResourceCollection  m_resource;                      // Resource list
   CTrading             m_trading;                       // Trading class object
   CArrayObj            m_list_counters;                 // List of timer counters
```

**we are going to use the object of the class for managing trading requests:**

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CSymbolsCollection   m_symbols;                       // Symbol collection
   CResourceCollection  m_resource;                      // Resource list
   CTradingControl      m_trading;                       // Trading management object
   CArrayObj            m_list_counters;                 // List of timer counters
```

These are all the changes in the library for now.

### Testing

To test the new library version, let's use [the EA from the previous \\
article](https://www.mql5.com/en/articles/7454#node04) and save it in \\MQL5\\Experts\\TestDoEasy\ **Part30\** under the name **TestDoEasyPart30.mq5**.

Compile the EA and launch it on a demo account chart.

**We need to check the behavior of pending request objects in case the AutoTrading button in the terminal is disabled.**

In the EA
settings, set the distances of StopLoss and TakeProfit to 0.

The following two customizable parameters are used for that:

- **StopLoss in points**

- **TakeProfit in points**.


After that, disable auto trading by clicking the AutoTrading button in the
terminal and attempt to send a trading request (for example, to set a pending SellLimit order) using the button on the EA trading
panel.

This results in the error caused by trading being disabled by the
terminal, the pending request is created.

Click
AutoTrading immediately to fix the error.

The newly created
trading request should be activated right away. It is to send the trading request to the server, and after its
execution, the request object is removed:

```
2019.12.27 04:16:28.894 automated trading is disabled
2019.12.27 04:16:33.177 CTrading::PlaceOrder<uint,int,uint,uint>: Invalid request:
2019.12.27 04:16:33.177 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.27 04:16:33.177 Correction of trade request parameters ...
2019.12.27 04:16:33.178 Pending request created #1:
2019.12.27 04:16:33.178 Pending request to place a pending order:
2019.12.27 04:16:33.178 - GBPUSD 0.10 Pending order Sell Limit, Price 1.30088
2019.12.27 04:16:33.178 - Pending request ID: #1, Created 2019.12.26 23:16:30.003, Attempts 5, Wait 00:00:20, End 2019.12.26 23:18:10.003
2019.12.27 04:16:33.178
2019.12.27 04:16:37.397 automated trading is enabled
2019.12.27 04:16:37.472 Retry trading attempt #1:
2019.12.27 04:16:37.472 Pending request to place a pending order:
2019.12.27 04:16:37.472 - GBPUSD 0.10 Pending order Sell Limit, Price 1.30088
2019.12.27 04:16:37.472 - Pending request ID: #1, Created 2019.12.26 23:16:10.003, Attempts 5, Wait 00:00:20, End 2019.12.26 23:17:50.003
2019.12.27 04:16:37.472
2019.12.27 04:16:37.977 - Pending order placed: 2019.12.26 23:16:38.325 -
2019.12.27 04:16:37.977 GBPUSD Placed 0.10 Pending order Sell Limit #500442708 at price 1.30088, Magic number 26148987 (123), G1: 15, G2: 8, ID: 1
2019.12.27 04:16:37.979 OnDoEasyEvent: Pending order placed
2019.12.27 04:16:38.024 Pending request to place a pending order, ID #1: Deleted due completed
```

**Now we need to check the ability to modify stop orders by the newly placed pending order** (since it has been placed
without them).

Disable auto trading again and click Set
StopLoss on the test EA trading panel.

Get the error caused by trading
operations being disabled by the terminal.

A pending request to
change the pending order parameters is created.

After another trading attempt sent by a pending request, enable
auto trading in the terminal. StopLoss is set for the pending order,
while the pending request itself is removed due to its execution:

```
2019.12.27 04:24:11.671 automated trading is disabled
2019.12.27 04:24:16.653 CTrading::ModifyOrder<int,int,double,int>: Invalid request:
2019.12.27 04:24:16.653 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.27 04:24:16.653 Correction of trade request parameters ...
2019.12.27 04:24:16.653 Pending request created #1:
2019.12.27 04:24:16.653 Pending request to modify pending order parameters:
2019.12.27 04:24:16.653 - GBPUSD 0.10 Pending order Sell Limit #500442708, Price 1.30088, StopLoss order level 1.30208
2019.12.27 04:24:16.653 - Order execution type: The order is executed at an available volume, unfulfilled remains in the market (Return)
2019.12.27 04:24:16.653 - Order expiration type: Good till cancel order
2019.12.27 04:24:16.653 - Pending request ID: #1, Created 2019.12.26 23:24:00.270, Attempts 5, Wait 00:00:20, End 2019.12.26 23:25:40.270
2019.12.27 04:24:16.653
2019.12.27 04:24:25.803 Retry trading attempt #1:
2019.12.27 04:24:25.803 Pending request to modify pending order parameters:
2019.12.27 04:24:25.803 - GBPUSD 0.10 Pending order Sell Limit #500442708, Price 1.30088, StopLoss order level 1.30208
2019.12.27 04:24:25.803 - Order execution type: The order is executed at an available volume, unfulfilled remains in the market (Return)
2019.12.27 04:24:25.803 - Order expiration type: Good till cancel order
2019.12.27 04:24:25.803 - Pending request ID: #1, Created 2019.12.26 23:24:00.270, Attempts 5, Wait 00:00:20, End 2019.12.26 23:25:40.270
2019.12.27 04:24:25.803
2019.12.27 04:24:29.770 automated trading is enabled
2019.12.27 04:24:30.022 Retry trading attempt #1:
2019.12.27 04:24:30.022 Pending request to modify pending order parameters:
2019.12.27 04:24:30.022 - GBPUSD 0.10 Pending order Sell Limit #500442708, Price 1.30088, StopLoss order level 1.30208
2019.12.27 04:24:30.022 - Order execution type: The order is executed at an available volume, unfulfilled remains in the market (Return)
2019.12.27 04:24:30.022 - Order expiration type: Good till cancel order
2019.12.27 04:24:30.022 - Pending request ID: #1, Created 2019.12.26 23:23:40.270, Attempts 5, Wait 00:00:20, End 2019.12.26 23:25:20.270
2019.12.27 04:24:30.022
2019.12.27 04:24:30.405 - Modified order StopLoss: 2019.12.26 23:16:38.325 -
2019.12.27 04:24:30.405 GBPUSD Pending order Sell Limit #500442708: Modified order StopLoss: [0.00000 --> 1.30208], Magic number 26148987 (123), G1: 15, G2: 8, ID: 1
2019.12.27 04:24:30.405 OnDoEasyEvent: Modified order StopLoss
2019.12.27 04:24:30.601 Pending request to modify pending order parameters, ID #1: Deleted due completed
```

As we can see, after enabling auto trading, a repeated request from the
pending request object has the same number it had during the first attempt of a repeated trading request, i.e. the logic of adding yet
another attempt when the error cause is eliminated by a user (by enabling auto trading in the terminal) has worked correctly.

**Now let's do the same for setting TakeProfit**.

All has worked correctly again (after the second trading attempt
performed by the pending request object), TakeProfit has been set by the pending request object and that request has been removed:

```
2019.12.27 04:32:46.843 automated trading is disabled
2019.12.27 04:32:50.810 CTrading::ModifyOrder<int,int,int,double>: Invalid request:
2019.12.27 04:32:50.810 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.27 04:32:50.810 Correction of trade request parameters ...
2019.12.27 04:32:50.810 Pending request created #1:
2019.12.27 04:32:50.810 Pending request to modify pending order parameters:
2019.12.27 04:32:50.810 - GBPUSD 0.10 Pending order Sell Limit #500442708, Price 1.30088, StopLoss order level 1.30208, TakeProfit order level 1.29888
2019.12.27 04:32:50.810 - Order execution type: The order is executed at an available volume, unfulfilled remains in the market (Return)
2019.12.27 04:32:50.810 - Order expiration type: Good till cancel order
2019.12.27 04:32:50.810 - Pending request ID: #1, Created 2019.12.26 23:32:47.943, Attempts 5, Wait 00:00:20, End 2019.12.26 23:34:27.943
2019.12.27 04:32:50.810
2019.12.27 04:33:08.782 Retry trading attempt #1:
2019.12.27 04:33:08.782 Pending request to modify pending order parameters:
2019.12.27 04:33:08.782 - GBPUSD 0.10 Pending order Sell Limit #500442708, Price 1.30088, StopLoss order level 1.30208, TakeProfit order level 1.29888
2019.12.27 04:33:08.782 - Order execution type: The order is executed at an available volume, unfulfilled remains in the market (Return)
2019.12.27 04:33:08.782 - Order expiration type: Good till cancel order
2019.12.27 04:33:08.782 - Pending request ID: #1, Created 2019.12.26 23:32:47.943, Attempts 5, Wait 00:00:20, End 2019.12.26 23:34:27.943
2019.12.27 04:33:08.782
2019.12.27 04:33:29.984 Retry trading attempt #2:
2019.12.27 04:33:29.984 Pending request to modify pending order parameters:
2019.12.27 04:33:29.984 - GBPUSD 0.10 Pending order Sell Limit #500442708, Price 1.30088, StopLoss order level 1.30208, TakeProfit order level 1.29888
2019.12.27 04:33:29.984 - Order execution type: The order is executed at an available volume, unfulfilled remains in the market (Return)
2019.12.27 04:33:29.984 - Order expiration type: Good till cancel order
2019.12.27 04:33:29.984 - Pending request ID: #1, Created 2019.12.26 23:32:47.943, Attempts 5, Wait 00:00:20, End 2019.12.26 23:34:27.943
2019.12.27 04:33:29.984
2019.12.27 04:33:31.999 automated trading is enabled
2019.12.27 04:33:32.250 Retry trading attempt #2:
2019.12.27 04:33:32.250 Pending request to modify pending order parameters:
2019.12.27 04:33:32.250 - GBPUSD 0.10 Pending order Sell Limit #500442708, Price 1.30088, StopLoss order level 1.30208, TakeProfit order level 1.29888
2019.12.27 04:33:32.250 - Order execution type: The order is executed at an available volume, unfulfilled remains in the market (Return)
2019.12.27 04:33:32.250 - Order expiration type: Good till cancel order
2019.12.27 04:33:32.250 - Pending request ID: #1, Created 2019.12.26 23:32:27.943, Attempts 5, Wait 00:00:20, End 2019.12.26 23:34:07.943
2019.12.27 04:33:32.250
2019.12.27 04:33:32.352 - Modified order TakeProfit: 2019.12.26 23:24:26.509 -
2019.12.27 04:33:32.352 GBPUSD Pending order Sell Limit #500442708: Modified order TakeProfit: [0.00000 --> 1.29888], Magic number 26148987 (123), G1: 15, G2: 8, ID: 1
2019.12.27 04:33:32.352 OnDoEasyEvent: Modified order TakeProfit
2019.12.27 04:33:32.754 Pending request to modify pending order parameters, ID #1: Deleted due completed
```

The tests show that we are now able to conduct different trading operations with the same order or position (trading operations with the same
ticket). In the previous versions, we were able to conduct a single trading operation with one ticket. After that, a pending request object
always assumed the work was complete. We have now fixed that behavior.

We will gradually check and debug the remaining trading operations related to pending requests during the further library development
since this requires careful and lengthy tests to detect and eliminate various abnormal situations.

### What's next?

In the next article, we will continue the development of the pending trading request concept.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7481#node00)

**Previous articles within the series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part 2. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part \\
3\. Collection of market orders and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. \\
Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. \\
Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part 6. Netting account events](https://www.mql5.com/en/articles/6383)

[Part \\
7\. StopLimit order activation events, preparing the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part \\
8\. Order and position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 — \\
Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and \\
activating pending orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure \\
events](https://www.mql5.com/en/articles/6921)

[Part 12. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part 13. Account object events](https://www.mql5.com/en/articles/6995)

[Part \\
14\. Symbol object](https://www.mql5.com/en/articles/7014)

[Part 15. Symbol object collection](https://www.mql5.com/en/articles/7041)

[Part \\
16\. Symbol collection events](https://www.mql5.com/en/articles/7071)

[Part 17. Interactivity of library objects](https://www.mql5.com/en/articles/7124)

[Part 18. Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

[Part \\
19\. Class of library messages](https://www.mql5.com/en/articles/7176)

[Part 20. Creating and storing program resources](https://www.mql5.com/en/articles/7195)

[Part 21. Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

[Part \\
22\. Trading classes - Base trading class, verification of limitations](https://www.mql5.com/en/articles/7258)

[Part 23. \\
Trading classes - Base trading class, verification of valid parameters](https://www.mql5.com/en/articles/7286)

[Part 24. \\
Trading classes - Base trading class, auto correction of invalid parameters](https://www.mql5.com/en/articles/7326)

[Part \\
25\. Trading classes - Base trading class, handling errors returned by the trade server](https://www.mql5.com/en/articles/7365)

[Part \\
26\. Working with pending trading requests - First implementation (opening positions)](https://www.mql5.com/en/articles/7394)

[Part \\
27\. Working with pending trading requests - Placing pending orders](https://www.mql5.com/en/articles/7418)

[Part 28. \\
Working with pending trading requests - Closure, removal and modification](https://www.mql5.com/en/articles/7438)

[Part \\
29\. Working with pending trading requests - Request object classes](https://www.mql5.com/en/articles/7454)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7481](https://www.mql5.com/ru/articles/7481)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7481.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7481/mql5.zip "Download MQL5.zip")(3642.78 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7481/mql4.zip "Download MQL4.zip")(3642.77 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/335372)**
(8)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
9 Jan 2020 at 13:11

**Сергей Романов:**

Artem, good afternoon. Your work is invaluable! Thank you for the opportunity to develop and the time you spend so that people like me can save it qualitatively.

I have a suggestion: should we publish a repository with your libs on github? What it would do:

1. Well, a version control system, of course.
2. Opportunity to participate in the development of the DoEasy library for everyone.
3. The ability to quickly search the code and access the most up-to-date version at any time.

For example, while studying the library I found and fixed some bugs and optimised some code. And in such a case I could now send you a special request within the framework of the version control system, by which you would decide whether to accept my improvements into the current version of the library or not.

You are doing a great job, and we together (there are many people like me) could invest our own time for the common good in order to improve DoEasy. I am ready to help in realisation of the proposed.

Hello. Thank you for your feedback.

At the moment it's too early to talk and think about the repository - the library is under active development, and until I create and publish everything I've planned, I don't want to deviate from the plan. And its numerous fixes and revisions by third-party users will only distract the author from following the plans.

But it is better to report about the found errors and ways of their elimination directly in discussions of those articles where errors were found - it will just help the development of the library and elimination of the found errors.

![MQL_User](https://c.mql5.com/avatar/avatar_na2.png)

**[MQL\_User](https://www.mql5.com/en/users/mql_user)**
\|
5 Aug 2022 at 13:37

When compiling the TradingControl.mqh file, two errors occur:

'CTrading::OpenPosition<double,d...' - cannot access private member function TradingControl.mqh 328 21

see declaration of 'CTrading::OpenPosition<double,double>' Trading.mqh 146 24

'CTrading::PlaceOrder<double,double...' - cannot access private member function TradingControl.mqh 344 18

see declaration of 'CTrading::PlaceOrder<double,double,double,double>' Trading.mqh 156 26

These methods are in the private section of the CTrading class. The errors go away if you move them to the public section of this class. But in the attached files (and as I understand, they are working) these methods are also in the private section of the CTrading class, and these two errors occur when compiling the TradingControl.mqh file.

Artyom, then how does it work for you? Either there is an error here or I misunderstand something.

P.S. I have downloaded the attached files for the next part - 31, and there too these methods are located in the private section of the CTrading class and also these two errors occur when compiling.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
5 Aug 2022 at 13:51

**MQL\_User [#](https://www.mql5.com/ru/forum/329669#comment_41240204):**

Two errors occur when compiling the TradingControl.mqh file:

'CTrading::OpenPosition<double,d...' - cannot access private member function TradingControl.mqh 328 21

see declaration of 'CTrading::OpenPosition<double,double>' Trading.mqh 146 24

'CTrading::PlaceOrder<double,double...' - cannot access private member function TradingControl.mqh 344 18

see declaration of 'CTrading::PlaceOrder<double,double,double,double>' Trading.mqh 156 26

These methods are in the private section of the CTrading class. The errors go away if you move them to the public section of this class. But in the attached files (and as I understand, they are working) these methods are also in the private section of the CTrading class, and these two errors occur when compiling the TradingControl.mqh file.

Artyom, then how does it work for you? Either there is an error here or I misunderstand something.

P.S. I have downloaded the attached files for the next part - 31, and there too these methods are located in the private section of the CTrading class and also these two errors occur when compiling.

This became so after a recent update. Try to make them in a protected section so that they cannot be accessed from outside. Public ones are too bad

![MQL_User](https://c.mql5.com/avatar/avatar_na2.png)

**[MQL\_User](https://www.mql5.com/en/users/mql_user)**
\|
5 Aug 2022 at 14:39

**Artyom Trishkin [#](https://www.mql5.com/ru/forum/329669#comment_41240835):**

This became so after a recent update. Try making them in a protected section so they can't be accessed from outside. Public is too bad

Ok, I'll do that.

One more question: there are methods in the public section ClosePosition, PlaceBuyStop, PlaceBuyLimit, etc. Is it not critical for them that they are in the public section?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
5 Aug 2022 at 14:57

**MQL\_User [#](https://www.mql5.com/ru/forum/329669#comment_41241050):**

All right, I will.

And there is another question: there are methods in the public section ClosePosition, PlaceBuyStop, PlaceBuyLimit, etc. Is it not critical for them to be in the public section?

They should be there - they are one of the methods for working with orders and positions

![Continuous Walk-Forward Optimization (Part 3): Adapting a Robot to Auto Optimizer](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization__2.png)[Continuous Walk-Forward Optimization (Part 3): Adapting a Robot to Auto Optimizer](https://www.mql5.com/en/articles/7490)

The third part serves as a bridge between the previous two parts: it describes the mechanism of interaction with the DLL considered in the first article and the objects for report downloading, which were described in the second article. We will analyze the process of wrapper creation for a class which is imported from DLL and which forms an XML file with the trading history. We will also consider a method for interacting with this wrapper.

![Library for easy and quick development of MetaTrader programs (part XXIX): Pending trading requests - request object classes](https://c.mql5.com/2/37/MQL5-avatar-doeasy__17.png)[Library for easy and quick development of MetaTrader programs (part XXIX): Pending trading requests - request object classes](https://www.mql5.com/en/articles/7454)

In the previous articles, we checked the concept of pending trading requests. A pending request is, in fact, a common trading order executed by a certain condition. In this article, we are going to create full-fledged classes of pending request objects — a base request object and its descendants.

![Econometric approach to finding market patterns: Autocorrelation, Heat Maps and Scatter Plots](https://c.mql5.com/2/37/jlp_0d3zw11j.png)[Econometric approach to finding market patterns: Autocorrelation, Heat Maps and Scatter Plots](https://www.mql5.com/en/articles/5451)

The article presents an extended study of seasonal characteristics: autocorrelation heat maps and scatter plots. The purpose of the article is to show that "market memory" is of seasonal nature, which is expressed through maximized correlation of increments of arbitrary order.

![Neural Networks Made Easy](https://c.mql5.com/2/48/Neural_networks_made_easy_001.png)[Neural Networks Made Easy](https://www.mql5.com/en/articles/7447)

Artificial intelligence is often associated with something fantastically complex and incomprehensible. At the same time, artificial intelligence is increasingly mentioned in everyday life. News about achievements related to the use of neural networks often appear in different media. The purpose of this article is to show that anyone can easily create a neural network and use the AI achievements in trading.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hbyrhcapczcdfqttumhihnrxbswtuiny&ssn=1769186240559393302&ssn_dr=0&ssn_sr=0&fv_date=1769186240&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7481&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XXX)%3A%20Pending%20trading%20requests%20-%20managing%20request%20objects%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918624060850585&fz_uniq=5070433133519640000&sv=2552)

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