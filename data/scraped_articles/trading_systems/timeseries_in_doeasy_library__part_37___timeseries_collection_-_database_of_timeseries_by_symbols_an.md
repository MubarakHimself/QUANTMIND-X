---
title: Timeseries in DoEasy library (part 37): Timeseries collection - database of timeseries by symbols and periods
url: https://www.mql5.com/en/articles/7663
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:51:55.908564
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/7663&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083167183271368265)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7663#node01)
- [Improving previously created timeseries objects](https://www.mql5.com/en/articles/7663#node02)
- [Collection class of timeseries objects by symbols and periods](https://www.mql5.com/en/articles/7663#node03)
- [Testing](https://www.mql5.com/en/articles/7663#node04)
- [What's next?](https://www.mql5.com/en/articles/7663#node05)


### Concept

- At the start of the series, we created the [bar object](https://www.mql5.com/en/articles/7594) containing data of a single bar of a specified chart symbol and period.
- We have created the [bar collection](https://www.mql5.com/en/articles/7594#node04) — timeseries object of a specified chart symbol and period.
- All timeseries objects of a single symbol have been combined into a single [timeseries object of a single symbol](https://www.mql5.com/en/articles/7627).

Today, we are going to create the collection object of timeseries of the symbols used in the program, each of which contains data of specified timeframes of one symbol. As a result, we get one object that contains all the data for a given number of bars for each timeseries of each symbol.

The timeseries collection is to store all required historical data for each of the symbols used in the program and for all timeframes to be set in the program settings.

In addition, the collection allows setting the required data for each timeframe of each symbol separately.

Since the description of the timeseries collection functionality is quite large, its real-time updates and obtaining all possible data from it are to be implemented in the next article.

### Improving previously created timeseries objects

Most library objects are derived from the [base object of all library objects](https://www.mql5.com/en/articles/7071), which in turn is derived from [the basic class for building MQL5 Standard library](https://www.mql5.com/en/docs/standardlibrary/cobject).

With the growing library needs, the CBaseObj class of the base library object has become larger as well. Now, if we inherit new objects from it, then they get additional and often completely unnecessary methods.

To solve this issue, let's divide the base object class into two:

- the first one (CBaseObj) is to contain the baseline minimum set of properties and methods for each library object,
- the second one is an extended (CBaseObjExt) CBaseObj descendant, which is to contain properties and methods for interaction with a user and the event functionality of descendant objects.

Thus, the objects that need base properties and methods are to be derived from CBaseObj, while objects that need the event functionality are to be inherited from CBaseObjExt.

First, simply rename the CBaseObj base object class in \\MQL5\\Include\\DoEasy\\Objects\\BaseObj.mqh to CBaseObjExt and compile the file of the CEngine library main object in \\MQL5\\Include\\DoEasy\\Engine.mqh causing a large list of compilation errors (since we renamed the library base object class).

Go through the list of all errors indicating the absence of the CBaseObj class and replace all instances of "CBaseObj" strings with "CBaseObjExt" in the class listings. Re-compilation with the corrected base object class names should be successful.

Now in the base object class listing, add the new class CBaseObj, derive it from the MQL5 library base object and move from the CBaseObjExt class all variables and methods to be present in the new class of the base object. The CBaseObjExt class is derived from CBaseObj.

It all seems complicated but if we have a look at the class listing, all becomes clear (there is no point in displaying the full listing here, you can find the entire code in the attached files):

```
//+------------------------------------------------------------------+
//| Base object class for all library objects                        |
//+------------------------------------------------------------------+
class CBaseObj : public CObject
  {
protected:
   ENUM_LOG_LEVEL    m_log_level;                              // Logging level
   ENUM_PROGRAM_TYPE m_program;                                // Program type
   bool              m_first_start;                            // First launch flag
   bool              m_use_sound;                              // Flag of playing the sound set for an object
   bool              m_available;                              // Flag of using a descendant object in the program
   int               m_global_error;                           // Global error code
   long              m_chart_id_main;                          // Control program chart ID
   long              m_chart_id;                               // Chart ID
   string            m_name;                                   // Object name
   string            m_folder_name;                            // Name of the folder storing CBaseObj descendant objects
   string            m_sound_name;                             // Object sound file name
   int               m_type;                                   // Object type (corresponds to the collection IDs)

public:
//--- (1) Set, (2) return the error logging level
   void              SetLogLevel(const ENUM_LOG_LEVEL level)         { this.m_log_level=level;                 }
   ENUM_LOG_LEVEL    GetLogLevel(void)                         const { return this.m_log_level;                }
//--- (1) Set and (2) return the chart ID of the control program
   void              SetMainChartID(const long id)                   { this.m_chart_id_main=id;                }
   long              GetMainChartID(void)                      const { return this.m_chart_id_main;            }
//--- (1) Set and (2) return chart ID
   void              SetChartID(const long id)                       { this.m_chart_id=id;                     }
   long              GetChartID(void)                          const { return this.m_chart_id;                 }
//--- (1) Set the sub-folder name, (2) return the folder name for storing descendant object files
   void              SetSubFolderName(const string name)             { this.m_folder_name=DIRECTORY+name;      }
   string            GetFolderName(void)                       const { return this.m_folder_name;              }
//--- (1) Set and (2) return the name of the descendant object sound file
   void              SetSoundName(const string name)                 { this.m_sound_name=name;                 }
   string            GetSoundName(void)                        const { return this.m_sound_name;               }
//--- (1) Set and (2) return the flag of playing descendant object sounds
   void              SetUseSound(const bool flag)                    { this.m_use_sound=flag;                  }
   bool              IsUseSound(void)                          const { return this.m_use_sound;                }
//--- (1) Set and (2) return the flag of using the descendant object in the program
   void              SetAvailable(const bool flag)                   { this.m_available=flag;                  }
   bool              IsAvailable(void)                         const { return this.m_available;                }
//--- Return the global error code
   int               GetError(void)                            const { return this.m_global_error;             }
//--- Return the object name
   string            GetName(void)                             const { return this.m_name;                     }
//--- Return an object type
   virtual int       Type(void)                                const { return this.m_type;                     }
//--- Constructor
                     CBaseObj() : m_program((ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE)),
                                  m_global_error(ERR_SUCCESS),
                                  m_log_level(LOG_LEVEL_ERROR_MSG),
                                  m_chart_id_main(::ChartID()),
                                  m_chart_id(::ChartID()),
                                  m_folder_name(DIRECTORY),
                                  m_sound_name(""),
                                  m_name(__FUNCTION__),
                                  m_type(0),
                                  m_use_sound(false),
                                  m_available(true),
                                  m_first_start(true) {}
  };
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Extended base object class for all library objects               |
//+------------------------------------------------------------------+
#define  CONTROLS_TOTAL    (10)
class CBaseObjExt : public CBaseObj
  {
private:
   int               m_long_prop_total;
   int               m_double_prop_total;
   //--- Fill in the object property array
   template<typename T> bool  FillPropertySettings(const int index,T &array[][CONTROLS_TOTAL],T &array_prev[][CONTROLS_TOTAL],int &event_id);
protected:
   CArrayObj         m_list_events_base;                       // Object base event list
   CArrayObj         m_list_events;                            // Object event list
   MqlTick           m_tick;                                   // Tick structure for receiving quote data
   double            m_hash_sum;                               // Object data hash sum
   double            m_hash_sum_prev;                          // Object data hash sum during the previous check
   int               m_digits_currency;                        // Number of decimal places in an account currency
   bool              m_is_event;                               // Object event flag
   int               m_event_code;                             // Object event code
   int               m_event_id;                               // Event ID (equal to the object property value)

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
//--- Return (1) an event object and (2) a base event by its number in the list
   CEventBaseObj    *GetEvent(const int shift=WRONG_VALUE,const bool check_out=true);
   CBaseEvent       *GetEventBase(const int index);
//--- Return the number of (1) object events
   int               GetEventsTotal(void)                      const { return this.m_list_events.Total();      }
//--- Update the object data to search for changes (Calling from the descendants: CBaseObj::Refresh())
   virtual void      Refresh(void);
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
                     CBaseObjExt();
  };
//+------------------------------------------------------------------+
```

Three new class member variables are now declared in the new base object class of all library objects:

```
   bool              m_use_sound;                              // Flag of playing the sound set for an object
   bool              m_available;                              // Flag of using a descendant object in the program
   string            m_sound_name;                             // Object sound file name
```

The appropriate methods of setting and returning the values of the variables are also added:

```
//--- (1) Set and (2) return the name of the descendant object sound file
   void              SetSoundName(const string name)                 { this.m_sound_name=name;                 }
   string            GetSoundName(void)                        const { return this.m_sound_name;               }
//--- (1) Set and (2) return the flag of playing descendant object sounds
   void              SetUseSound(const bool flag)                    { this.m_use_sound=flag;                  }
   bool              IsUseSound(void)                          const { return this.m_use_sound;                }
//--- (1) Set and (2) return the flag of using the descendant object in the program
   void              SetAvailable(const bool flag)                   { this.m_available=flag;                  }
   bool              IsAvailable(void)                         const { return this.m_available;                }
```

The name of the sound file for the descendant object allows setting (SetSoundName()) or receiving (GetSoundName()) the name of the object sound file that can be played if there is any condition that controls the object properties.

Besides the fact that the name can be assigned to the object, it will be possible to enable/disable (SetUseSound()) the permission to play the file and get the flag of the set permission to play the file (IsUseSound()).

So, what is actually "the flag of using the descendant object" and why do we need to set and receive it?

For example, we have symbol timeseries objects for М5, М30, Н1 and D1. But at some point in time, we do not want to handle the M5 timeseries. By enabling/disabling the flag, we are able to regulate the necessity to manage the event library, for example the new bar for the M5 timeseries.

The presence of such a flag in the base object of all library objects allows us to flexibly control the need to handle the state of the properties of such objects. In other words, if you need to handle and use the object in the program, set the flag. If you no longer need to do that, remove the flag.

Naturally, the class constructors have been changed as well — the variables moved to the new class have been removed, while all the variables in the new class have been initialized. See the changes in detail in the attached files.

The classes that are now derived from the extended base object **CBaseObjExt**:

- **CAccountsCollection** in \\MQL5\\Include\\DoEasy\\Collections\\AccountsCollection.mqh
- **CEventsCollection** in \\MQL5\\Include\\DoEasy\\Collections\\EventsCollection.mqh

(replaced "CBaseObj::EventAdd" with "CBaseObjExt::EventAdd")

- **CSymbolsCollection** in \\MQL5\\Include\\DoEasy\\Collections\\SymbolsCollection.mqh
- **CAccount** in \\MQL5\\Include\\DoEasy\\Objects\\Accounts\\Account.mqh

(replaced "CBaseObj::Refresh()" with "CBaseObjExt::Refresh()")
- **COrder** in \\MQL5\\Include\\DoEasy\\Objects\\Orders\\Order.mqh
- **CPendRequest** in \\MQL5\\Include\\DoEasy\\Objects\\PendRequest\\PendRequest.mqh

(replaced "return CBaseObj::GetMagicID" with "return CBaseObjExt::GetMagicID",

    CBaseObj::GetGroupID1" with "return CBaseObjExt::GetGroupID1" and CBaseObj::GetGroupID2" with "return CBaseObjExt::GetGroupID2")
- **CSymbol** in \\MQL5\\Include\\DoEasy\\Objects\\Symbols\\Symbol.mqh

(replaced "CBaseObj::Refresh()" with "CBaseObjExt::Refresh()")
- **CTradeObj** in \\MQL5\\Include\\DoEasy\\Objects\\Trade\\TradeObj.mqh

(removed the bool m\_use\_sound variable, as well as the SetUseSound() and IsUseSound() methods — they are in the base class now)
- **CTrading** in \\MQL5\\Include\\DoEasy\\Trading.mqh

(removed the bool m\_use\_sound variable and the IsUseSounds() method — they are in the base class now)

Before improving the already created timeseries object classes, let's add the necessary data to the **Datas.mqh** file — the new macro substitution specifying the separator in the string of the list of used symbols and timeframes in the program inputs, the enumeration of timeframe operation modes, as well as new message indices and message texts corresponding to declared indices:

```
//+------------------------------------------------------------------+
//|                                                        Datas.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define INPUT_SEPARATOR                (",")          // Separator in the inputs string
#define TOTAL_LANG                     (2)            // Number of used languages
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Modes of working with symbols                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOLS_MODE
  {
   SYMBOLS_MODE_CURRENT,                              // Work with the current symbol only
   SYMBOLS_MODE_DEFINES,                              // Work with the specified symbol list
   SYMBOLS_MODE_MARKET_WATCH,                         // Work with the Market Watch window symbols
   SYMBOLS_MODE_ALL                                   // Work with the full symbol list
  };
//+------------------------------------------------------------------+
//| Mode of working with timeframes                                  |
//+------------------------------------------------------------------+
enum ENUM_TIMEFRAMES_MODE
  {
   TIMEFRAMES_MODE_CURRENT,                           // Work with the current timeframe only
   TIMEFRAMES_MODE_LIST,                              // Work with the specified timeframe list
   TIMEFRAMES_MODE_ALL                                // Work with the full timeframe list
  };
//+------------------------------------------------------------------+

   MSG_LIB_SYS_ERROR_EMPTY_SYMBOLS_STRING,            // Error. Predefined symbols string empty, to be used
   MSG_LIB_SYS_FAILED_PREPARING_SYMBOLS_ARRAY,        // Failed to prepare array of used symbols. Error
   MSG_LIB_SYS_ERROR_EMPTY_PERIODS_STRING,            // Error. The string of predefined periods is empty and is to be used
   MSG_LIB_SYS_FAILED_PREPARING_PERIODS_ARRAY,        // Failed to prepare array of used periods. Error
   MSG_LIB_SYS_INVALID_ORDER_TYPE,                    // Invalid order type:
```

...

```
//--- CTimeSeries
   MSG_LIB_TEXT_TS_TEXT_FIRS_SET_SYMBOL,              // First, set a symbol using SetSymbol()
   MSG_LIB_TEXT_TS_TEXT_UNKNOWN_TIMEFRAME,            // Unknown timeframe
   MSG_LIB_TEXT_TS_FAILED_GET_SERIES_OBJ,             // Failed to receive the timeseries object
   MSG_LIB_TEXT_TS_REQUIRED_HISTORY_DEPTH,            // Requested history depth
   MSG_LIB_TEXT_TS_ACTUAL_DEPTH,                      // Actual history depth
   MSG_LIB_TEXT_TS_AMOUNT_HISTORY_DATA,               // Created historical data
   MSG_LIB_TEXT_TS_HISTORY_BARS,                      // Number of history bars on the server
   MSG_LIB_TEXT_TS_TEXT_SYMBOL_TIMESERIES,            // Symbol timeseries
   MSG_LIB_TEXT_TS_TEXT_TIMESERIES,                   // Timeseries
   MSG_LIB_TEXT_TS_TEXT_REQUIRED,                     // Requested
   MSG_LIB_TEXT_TS_TEXT_ACTUAL,                       // Actual
   MSG_LIB_TEXT_TS_TEXT_CREATED,                      // Created
   MSG_LIB_TEXT_TS_TEXT_HISTORY_BARS,                 // On the server
   MSG_LIB_TEXT_TS_TEXT_SYMBOL_FIRSTDATE,             // The very first date by a period symbol
   MSG_LIB_TEXT_TS_TEXT_SYMBOL_LASTBAR_DATE,          // Time of opening the last bar by period symbol
   MSG_LIB_TEXT_TS_TEXT_SYMBOL_SERVER_FIRSTDATE,      // The very first date in history by a server symbol
   MSG_LIB_TEXT_TS_TEXT_SYMBOL_TERMINAL_FIRSTDATE,    // The very first date in history by a symbol in the client terminal

};
//+------------------------------------------------------------------+
```

...

```
   {"Ошибка. Строка предопределённых символов пустая, будет использоваться ","Error. String of predefined symbols is empty, the Symbol will be used: "},
   {"Не удалось подготовить массив используемых символов. Ошибка ","Failed to create an array of used symbols. Error "},
   {"Ошибка. Строка предопределённых периодов пустая, будет использоваться ","Error. String of predefined periods is empty, the Period will be used: "},
   {"Не удалось подготовить массив используемых периодов. Ошибка ","Failed to create an array of used periods. Error "},
   {"Неправильный тип ордера: ","Invalid order type: "},
```

...

```
   {"Сначала нужно установить символ при помощи SetSymbol()","First you need to set the Symbol using SetSymbol()"},
   {"Неизвестный таймфрейм","Unknown timeframe"},
   {"Не удалось получить объект-таймсерию ","Failed to get timeseries object "},
   {"Запрошенная глубина истории: ","Required history depth: "},
   {"Фактическая глубина истории: ","Actual history depth: "},
   {"Создано исторических данных: ","Total historical data created: "},
   {"Баров истории на сервере: ","Server history Bars number: "},
   {"Таймсерия символа","Symbol time series"},
   {"Таймсерия","Timeseries"},
   {"Запрошено","Required"},
   {"Фактически","Actual"},
   {"Создано","Created"},
   {"На сервере","On server"},
   {"Самая первая дата по символу-периоду","The very first date for the symbol-period"},
   {"Время открытия последнего бара по символу-периоду","Open time of the last bar of the symbol-period"},
   {"Самая первая дата в истории по символу на сервере","The very first date in the history of the symbol on the server"},
   {"Самая первая дата в истории по символу в клиентском терминале","The very first date in the history of the symbol in the client terminal"},

};
//+---------------------------------------------------------------------+
```

We have prepared all the data necessary for making improvements in the previously developed timeseries classes and creating the collection of all timeseries.

Let's improve the symbol timeseries object class for a single [CSeries](https://www.mql5.com/en/articles/7594#node04) timeframe.

In the private section of the class, add four new variables and one method for setting timeseries dates:

```
//+------------------------------------------------------------------+
//| Timeseries class                                                 |
//+------------------------------------------------------------------+
class CSeries : public CBaseObj
  {
private:
   ENUM_TIMEFRAMES   m_timeframe;                                       // Timeframe
   string            m_symbol;                                          // Symbol
   string            m_period_description;                              // Timeframe string description
   datetime          m_firstdate;                                       // The very first date by a period symbol at the moment
   datetime          m_lastbar_date;                                    // Time of opening the last bar by period symbol
   uint              m_amount;                                          // Amount of applied timeseries data
   uint              m_required;                                        // Required amount of applied timeseries data
   uint              m_bars;                                            // Number of bars in history by symbol and timeframe
   bool              m_sync;                                            // Synchronized data flag
   CArrayObj         m_list_series;                                     // Timeseries list
   CNewBarObj        m_new_bar_obj;                                     // "New bar" object
//--- Set the very first date by a period symbol at the moment and the new time of opening the last bar by a period symbol
   void              SetServerDate(void)
                       {
                        this.m_firstdate=(datetime)::SeriesInfoInteger(this.m_symbol,this.m_timeframe,SERIES_FIRSTDATE);
                        this.m_lastbar_date=(datetime)::SeriesInfoInteger(this.m_symbol,this.m_timeframe,SERIES_LASTBAR_DATE);
                       }
public:
```

Write the description of the timeseries chart period to the **m\_period\_description** variable immediately when creating the class object in the constructor and in the methods setting a timeframe for the timeseries object. This is done in order to avoid constant access to the TimeframeDescription() function from the library's DELib.mqh service function file — the function looks for a substring in the timeframe string description from the ENUM\_TIMEFRAMES enumeration hampering the execution. Therefore, it would be better to execute "slow" functions immediately when constructing objects in case the data is not to be changed or to be changed rarely upon a request from a program.

The **m\_firstdate** variable stores the very first date by a symbol period at the moment obtained by the [SeriesInfoInteger()](https://www.mql5.com/en/docs/series/seriesinfointeger) function with the [SERIES\_FIRSTDATE](https://www.mql5.com/en/docs/constants/tradingconstants/enum_series_info_integer) property ID.

The **m\_lastbar\_date** variable stores the last bar open time by a symbol period obtained by the [SeriesInfoInteger()](https://www.mql5.com/en/docs/series/seriesinfointeger) function with the [SERIES\_LASTBAR\_DATE](https://www.mql5.com/en/docs/constants/tradingconstants/enum_series_info_integer) property ID.

Both variables are set by calling the SetServerDate() method only at the time of creating a class object or changing data on a new bar, as well as when setting a new symbol or timeframe for the timeseries object.

The **m\_required** variable is to store the required (last requested) number of used timeseries data. When requesting the necessary number of timeseries bars, it may turn out that the requested amount of data for creating the timeseries is not present on the server. In this case, the timeseries is created in the amount equal to the amount of available server history. The variable always stores the last requested amount of data regardless of how much data was actually obtained and created. Considering that we use the concept of "requested data", the name of methods in the class containing "Amount" has changed — now it has been replaced with "Required".

In the public class section, the new methods have been added as well:

```
public:
//--- Return (1) oneself and (2) the timeseries list
   CSeries          *GetObject(void)                                    { return &this;         }
   CArrayObj        *GetList(void)                                      { return &m_list_series;}
//--- Return the list of bars by selected (1) double, (2) integer and (3) string property fitting a compared condition
   CArrayObj        *GetList(ENUM_BAR_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByBarProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_BAR_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByBarProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_BAR_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByBarProperty(this.GetList(),property,value,mode); }

//--- Set (1) symbol, (2) timeframe, (3) symbol and timeframe, (4) amount of applied timeseries data
   void              SetSymbol(const string symbol);
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe);
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe);
   bool              SetRequiredUsedData(const uint required,const uint rates_total);

//--- Return (1) symbol, (2) timeframe, number of (3) used and (4) requested timeseries data,
//--- (5) number of bars in the timeseries, (6) the very first date, (7) time of opening the last bar by a symbol period,
//--- new bar flag with (8) automatic and (9) manual time management
   string            Symbol(void)                                          const { return this.m_symbol;                            }
   ENUM_TIMEFRAMES   Timeframe(void)                                       const { return this.m_timeframe;                         }
   ulong             AvailableUsedData(void)                               const { return this.m_amount;                            }
   ulong             RequiredUsedData(void)                                const { return this.m_required;                          }
   ulong             Bars(void)                                            const { return this.m_bars;                              }
   datetime          FirstDate(void)                                       const { return this.m_firstdate;                         }
   datetime          LastBarDate(void)                                     const { return this.m_lastbar_date;                      }
   bool              IsNewBar(const datetime time)                               { return this.m_new_bar_obj.IsNewBar(time);        }
   bool              IsNewBarManual(const datetime time)                         { return this.m_new_bar_obj.IsNewBarManual(time);  }
//--- Return the bar object by index (1) in the list and (2) in the timeseries, as well as (3) the real list size
   CBar             *GetBarByListIndex(const uint index);
   CBar             *GetBarBySeriesIndex(const uint index);
   int               DataTotal(void)                                       const { return this.m_list_series.Total();               }
//--- Return (1) Open, (2) High, (3) Low, (4) Close, (5) time, (6) tick volume, (7) real volume, (8) bar spread by index
   double            Open(const uint index,const bool from_series=true);
   double            High(const uint index,const bool from_series=true);
   double            Low(const uint index,const bool from_series=true);
   double            Close(const uint index,const bool from_series=true);
   datetime          Time(const uint index,const bool from_series=true);
   long              TickVolume(const uint index,const bool from_series=true);
   long              RealVolume(const uint index,const bool from_series=true);
   int               Spread(const uint index,const bool from_series=true);

//--- (1) Set and (2) return the sound of a sound file of the "New bar" timeseries event
   void              SetNewBarSoundName(const string name)                       { this.m_new_bar_obj.SetSoundName(name);           }
   string            NewBarSoundName(void)                                 const { return this.m_new_bar_obj.GetSoundName();        }

//--- Save the new bar time during the manual time management
   void              SaveNewBarTime(const datetime time)                         { this.m_new_bar_obj.SaveNewBarTime(time);         }
//--- Synchronize symbol and timeframe data with server data
   bool              SyncData(const uint required,const uint rates_total);
//--- (1) Create and (2) update the timeseries list
   int               Create(const uint required=0);
   void              Refresh(const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0);

//--- Return the timeseries name
   string            Header(void);
//--- Display (1) the timeseries description and (2) the brief timeseries description in the journal
   void              Print(void);
   void              PrintShort(void);

//--- Constructors
                     CSeries(void);
                     CSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0);
  };
//+------------------------------------------------------------------+
```

The **GetObject()** method returns the pointer to the entire timeseries object to the control program. It allows obtaining the entire timeseries object and work with it in a custom program.

The **RequiredUsedData()** method returns the value of the **m\_required** variable mentioned above to the calling program.

The **FirstDate()** and **LastBarDate()** methods return the values of **m\_firstdate** and **m\_lastbar\_date** variables that have also been described above.

The **SetNewBarSoundName()** method sets the sound file name for CNewBarObj "New bar" object which is part of the timeseries object.

The **NewBarSoundName()** method returns the sound file name assigned to the CNewBarObj "New bar" object which is part of the timeseries object.

The methods allows assigning the sound to any timeseries object. The sound is to be played when the "New bar" event is detected.

The **Header()** method creates and returns a short name of the timeseries object:

```
//+------------------------------------------------------------------+
//| Return the timeseries name                                       |
//+------------------------------------------------------------------+
string CSeries::Header(void)
  {
   return CMessage::Text(MSG_LIB_TEXT_TS_TEXT_TIMESERIES)+" \""+this.m_symbol+"\" "+this.m_period_description;
  }
//+------------------------------------------------------------------+
```

The string description of the timeseries is returned from the method in the following form

```
Timeseries "SYMBOL" TIMEFRAME_DESCRIPTION
```

For example:

```
Timeseries "AUDUSD" M15
```

The **Print()** method displays the full timeseries description in the journal:

```
//+------------------------------------------------------------------+
//| Display the timeseries description in the journal                |
//+------------------------------------------------------------------+
void CSeries::Print(void)
  {
   string txt=
     (
      CMessage::Text(MSG_LIB_TEXT_TS_REQUIRED_HISTORY_DEPTH)+(string)this.RequiredUsedData()+", "+
      CMessage::Text(MSG_LIB_TEXT_TS_ACTUAL_DEPTH)+(string)this.AvailableUsedData()+", "+
      CMessage::Text(MSG_LIB_TEXT_TS_AMOUNT_HISTORY_DATA)+(string)this.DataTotal()+", "+
      CMessage::Text(MSG_LIB_TEXT_TS_HISTORY_BARS)+(string)this.Bars()
     );
   ::Print(this.Header(),": ",txt);
  }
//+------------------------------------------------------------------+
```

Print the timeseries data to the journal in the following form

```
HEADER: HISTORY_DEPTH: XXXX, ACTUAL_DEPTH: XXXX, AMOUNT_HISTORY_DATA: XXXX, HISTORY_BARS: XXXX
```

For example:

```
Timeseries "AUDUSD" W1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 1400

Timeseries "AUDUSD" MN1: Requested history depth: 1000, Actual history depth: 322, Historical data created: 322, History bars on the server: 322
```

The **PrintShort()** method displays the short timeseries description in the journal:

```
//+------------------------------------------------------------------+
//| Display a short timeseries description in the journal            |
//+------------------------------------------------------------------+
void CSeries::PrintShort(void)
  {
   string txt=
     (
      CMessage::Text(MSG_LIB_TEXT_TS_TEXT_REQUIRED)+": "+(string)this.RequiredUsedData()+", "+
      CMessage::Text(MSG_LIB_TEXT_TS_TEXT_ACTUAL)+": "+(string)this.AvailableUsedData()+", "+
      CMessage::Text(MSG_LIB_TEXT_TS_TEXT_CREATED)+": "+(string)this.DataTotal()+", "+
      CMessage::Text(MSG_LIB_TEXT_TS_TEXT_HISTORY_BARS)+": "+(string)this.Bars()
     );
   ::Print(this.Header(),": ",txt);
  }
//+------------------------------------------------------------------+
```

Print the timeseries data to the journal in the following form

```
HEADER: REQUIRED: XXXX, ACTUAL: XXXX, CREATED: XXXX, HISTORY_BARS: XXXX
```

For example:

```
Timeseries "USDJPY" W1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2562

Timeseries "USDJPY" MN1: Requested: 1000, Actual: 589, Created: 589, On the server: 589
```

Add saving the timeframe description and setting the timeseries dates to both class constructors:

```
//+------------------------------------------------------------------+
//| Constructor 1 (current symbol and period timeseries)             |
//+------------------------------------------------------------------+
CSeries::CSeries(void) : m_bars(0),m_amount(0),m_required(0),m_sync(false)
  {
   this.m_program=(ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE);
   this.m_list_series.Clear();
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
   this.SetSymbolPeriod(NULL,(ENUM_TIMEFRAMES)::Period());
   this.m_period_description=TimeframeDescription(this.m_timeframe);
   this.SetServerDate();
  }
//+------------------------------------------------------------------+
//| Constructor 2 (specified symbol and period timeseries)           |
//+------------------------------------------------------------------+
CSeries::CSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0) : m_bars(0), m_amount(0),m_required(0),m_sync(false)
  {
   this.m_program=(ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE);
   this.m_list_series.Clear();
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
   this.SetSymbolPeriod(symbol,timeframe);
   this.m_sync=this.SetRequiredUsedData(required,0);
   this.m_period_description=TimeframeDescription(this.m_timeframe);
   this.SetServerDate();
  }
//+------------------------------------------------------------------+
```

**In the symbol setting method, add the check for the same symbol and setting timeseries dates:**

```
//+------------------------------------------------------------------+
//| Set a symbol                                                     |
//+------------------------------------------------------------------+
void CSeries::SetSymbol(const string symbol)
  {
   if(this.m_symbol==symbol)
      return;
   this.m_symbol=(symbol==NULL || symbol==""   ? ::Symbol() : symbol);
   this.m_new_bar_obj.SetSymbol(this.m_symbol);
   this.SetServerDate();
  }
//+------------------------------------------------------------------+
```

Here, if a symbol already used in the object is passed to the method, then nothing needs to be set — just leave the method.

**In the timeframe setting method, add the check for the same timeframe and setting timeseries dates:**

```
//+------------------------------------------------------------------+
//| Set a timeframe                                                  |
//+------------------------------------------------------------------+
void CSeries::SetTimeframe(const ENUM_TIMEFRAMES timeframe)
  {
   if(this.m_timeframe==timeframe)
      return;
   this.m_timeframe=(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe);
   this.m_new_bar_obj.SetPeriod(this.m_timeframe);
   this.m_period_description=TimeframeDescription(this.m_timeframe);
   this.SetServerDate();
  }
//+------------------------------------------------------------------+
```

Here, if a timeframe already used in the object is passed to the method, then nothing needs to be set — just leave the method.

Let's change the method of setting a symbol and timeframe:

```
//+------------------------------------------------------------------+
//| Set a symbol and timeframe                                       |
//+------------------------------------------------------------------+
void CSeries::SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   if(this.m_symbol==symbol && this.m_timeframe==timeframe)
      return;
   this.SetSymbol(symbol);
   this.SetTimeframe(timeframe);
  }
//+------------------------------------------------------------------+
```

Here, if the same symbol and timeframe are passed to the method, then nothing needs to be changed — just leave the method.

Next, call the methods of setting a symbol and timeframe.

**In the method of setting the required timeseries history depth, save the requested history depth:**

```
//+------------------------------------------------------------------+
//| Set the number of required data                                  |
//+------------------------------------------------------------------+
bool CSeries::SetRequiredUsedData(const uint required,const uint rates_total)
  {
   this.m_required=(required==0 ? SERIES_DEFAULT_BARS_COUNT : required);
//--- Set the number of available timeseries bars
   this.m_bars=(uint)
     (
      //--- If this is an indicator and the work is performed on the current symbol and timeframe,
      //--- add the rates_total value passed to the method,
      //--- otherwise, get the number from the environment
      this.m_program==PROGRAM_INDICATOR &&
      this.m_symbol==::Symbol() && this.m_timeframe==::Period() ? rates_total :
      ::SeriesInfoInteger(this.m_symbol,this.m_timeframe,SERIES_BARS_COUNT)
     );
//--- If managed to set the number of available history, set the amount of data in the list:
   if(this.m_bars>0)
     {
      //--- if zero 'required' value is passed,
      //--- use either the default value (1000 bars) or the number of available history bars - the least one of them
      //--- if non-zero 'required' value is passed,
      //--- use either the 'required' value or the number of available history bars - the least one of them
      this.m_amount=(required==0 ? ::fmin(SERIES_DEFAULT_BARS_COUNT,this.m_bars) : ::fmin(required,this.m_bars));
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

If the zero **required** value is passed, request history in the default amount (1000 bars) specified by the SERIES\_DEFAULT\_BARS\_COUNT macro substitution in the Define.mqh file, otherwise - the value passed from 'required'.

**The method of updating the timeseries receives the check for using the timeseries in a program. I** f the timeseries is not used, then nothing needs to be updated:

```
//+------------------------------------------------------------------+
//| Update timeseries list and data                                  |
//+------------------------------------------------------------------+
void CSeries::Refresh(const datetime time=0,
                      const double open=0,
                      const double high=0,
                      const double low=0,
                      const double close=0,
                      const long tick_volume=0,
                      const long volume=0,
                      const int spread=0)
  {
//--- If the timeseries is not used, exit
   if(!this.m_available)
      return;
   MqlRates rates[1];
//--- Set the flag of sorting the list of bars by index
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
//--- If a new bar is present on a symbol and period,
   if(this.IsNewBarManual(time))
     {
      //--- create a new bar object and add it to the end of the list
      CBar *new_bar=new CBar(this.m_symbol,this.m_timeframe,0);
      if(new_bar==NULL)
         return;
      if(!this.m_list_series.Add(new_bar))
        {
         delete new_bar;
         return;
        }
      //--- Write the very first date by a period symbol at the moment and the new time of opening the last bar by a period symbol
      this.SetServerDate();
      //--- if the timeseries exceeds the requested number of bars, remove the earliest bar
      if(this.m_list_series.Total()>(int)this.m_required)
         this.m_list_series.Delete(0);
      //--- save the new bar time as the previous one for the subsequent new bar check
      this.SaveNewBarTime(time);
     }
//--- Get the index of the last bar in the list and the object bar by the index
   int index=this.m_list_series.Total()-1;
   CBar *bar=this.m_list_series.At(index);
//--- if the work is performed in an indicator and the timeseries belongs to the current symbol and timeframe,
//--- copy price parameters (passed to the method from the outside) to the bar price structure
   int copied=1;
   if(this.m_program==PROGRAM_INDICATOR && this.m_symbol==::Symbol() && this.m_timeframe==::Period())
     {
      rates[0].time=time;
      rates[0].open=open;
      rates[0].high=high;
      rates[0].low=low;
      rates[0].close=close;
      rates[0].tick_volume=tick_volume;
      rates[0].real_volume=volume;
      rates[0].spread=spread;
     }
//--- otherwise, get data to the bar price structure from the environment
   else
      copied=::CopyRates(this.m_symbol,this.m_timeframe,0,1,rates);
//--- If the prices are obtained, set the new properties from the price structure for the bar object
   if(copied==1)
      bar.SetProperties(rates[0]);
  }
//+------------------------------------------------------------------+
```

Additionally, if a new bar appears in the timeseries, update timeseries dates.

These are the main changes in the class. I do not consider minor changes in the names of methods here. You can find the complete class listing in the attached files.

This completes the work on CSeries class at the current stage.

Let's finalize the [CTimeSeries](https://www.mql5.com/en/articles/7627#node02) class containing the [CSeries](https://www.mql5.com/en/articles/7594#node04) objects for all possible chart periods of a single symbol.

The class listing features the IndexTimeframe() and TimeframeByIndex() methods for receiving a timeframe index in the list storing the timeseries of the appropriate chart and timeframe periods by the list index. The methods are quite specific, since they are based on the fact that the index of the minimum possible timeframe (PERIOD\_M1) is contained in the zero index of the list. Within the ENUM\_TIMEFRAMES enumeration, М1 index is already equal to one, since the zero index contains the PERIOD\_CURRENT constant. In other words, all indices are offset by 1 relative to zero.

We will need functions returning the chart period constant index within the ENUM\_TIMEFRAMES enumeration, and, vice versa, return its constant from the enumeration by the chart period. We are going to create the appropriate functions in the file of service functions — IndexEnumTimeframe() and TimeframeByEnumIndex(), while in the CTimeSeries class listing, remove implementation of the IndexTimeframe() and TimeframeByIndex() methods adding the call of the IndexEnumTimeframe() and TimeframeByEnumIndex() functions with the offset of one to the class body.

Add three functions to the DELib.mqh file of service functions:

```
//+------------------------------------------------------------------+
//| Return the timeframe index in the ENUM_TIMEFRAMES enumeration    |
//+------------------------------------------------------------------+
char IndexEnumTimeframe(ENUM_TIMEFRAMES timeframe)
  {
   int statement=(timeframe==PERIOD_CURRENT ? Period() : timeframe);
   switch(statement)
     {
      case PERIOD_M1    :  return 1;
      case PERIOD_M2    :  return 2;
      case PERIOD_M3    :  return 3;
      case PERIOD_M4    :  return 4;
      case PERIOD_M5    :  return 5;
      case PERIOD_M6    :  return 6;
      case PERIOD_M10   :  return 7;
      case PERIOD_M12   :  return 8;
      case PERIOD_M15   :  return 9;
      case PERIOD_M20   :  return 10;
      case PERIOD_M30   :  return 11;
      case PERIOD_H1    :  return 12;
      case PERIOD_H2    :  return 13;
      case PERIOD_H3    :  return 14;
      case PERIOD_H4    :  return 15;
      case PERIOD_H6    :  return 16;
      case PERIOD_H8    :  return 17;
      case PERIOD_H12   :  return 18;
      case PERIOD_D1    :  return 19;
      case PERIOD_W1    :  return 20;
      case PERIOD_MN1   :  return 21;
      default           :  Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_UNKNOWN_TIMEFRAME)); return WRONG_VALUE;
     }
  }
//+------------------------------------------------------------------+
//| Return the timeframe by the ENUM_TIMEFRAMES enumeration index    |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES TimeframeByEnumIndex(const uchar index)
  {
   if(index==0) return(ENUM_TIMEFRAMES)Period();
   switch(index)
     {
      case 1   :  return PERIOD_M1;
      case 2   :  return PERIOD_M2;
      case 3   :  return PERIOD_M3;
      case 4   :  return PERIOD_M4;
      case 5   :  return PERIOD_M5;
      case 6   :  return PERIOD_M6;
      case 7   :  return PERIOD_M10;
      case 8   :  return PERIOD_M12;
      case 9   :  return PERIOD_M15;
      case 10  :  return PERIOD_M20;
      case 11  :  return PERIOD_M30;
      case 12  :  return PERIOD_H1;
      case 13  :  return PERIOD_H2;
      case 14  :  return PERIOD_H3;
      case 15  :  return PERIOD_H4;
      case 16  :  return PERIOD_H6;
      case 17  :  return PERIOD_H8;
      case 18  :  return PERIOD_H12;
      case 19  :  return PERIOD_D1;
      case 20  :  return PERIOD_W1;
      case 21  :  return PERIOD_MN1;
      default  :  Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_GET_DATAS),"... ",CMessage::Text(MSG_SYM_STATUS_INDEX),": ",(string)index); return WRONG_VALUE;
     }
  }
//+------------------------------------------------------------------+
//| Return the timeframe by its description                          |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES TimeframeByDescription(const string timeframe)
  {
   return
     (
      timeframe=="M1"   ?  PERIOD_M1   :
      timeframe=="M2"   ?  PERIOD_M2   :
      timeframe=="M3"   ?  PERIOD_M3   :
      timeframe=="M4"   ?  PERIOD_M4   :
      timeframe=="M5"   ?  PERIOD_M5   :
      timeframe=="M6"   ?  PERIOD_M6   :
      timeframe=="M10"  ?  PERIOD_M10  :
      timeframe=="M12"  ?  PERIOD_M12  :
      timeframe=="M15"  ?  PERIOD_M15  :
      timeframe=="M20"  ?  PERIOD_M20  :
      timeframe=="M30"  ?  PERIOD_M30  :
      timeframe=="H1"   ?  PERIOD_H1   :
      timeframe=="H2"   ?  PERIOD_H2   :
      timeframe=="H3"   ?  PERIOD_H3   :
      timeframe=="H4"   ?  PERIOD_H4   :
      timeframe=="H6"   ?  PERIOD_H6   :
      timeframe=="H8"   ?  PERIOD_H8   :
      timeframe=="H12"  ?  PERIOD_H12  :
      timeframe=="D1"   ?  PERIOD_D1   :
      timeframe=="W1"   ?  PERIOD_W1   :
      timeframe=="MN1"  ?  PERIOD_MN1  :
      PERIOD_CURRENT
     );
  }
//+------------------------------------------------------------------+
```

**In the CTimeSeries class file, remove the implementation of the IndexTimeframe() and TimeframeByIndex() methods from the listing:**

```
//+------------------------------------------------------------------+
//| Return the timeframe index in the list                           |
//+------------------------------------------------------------------+
char CTimeSeries::IndexTimeframe(ENUM_TIMEFRAMES timeframe) const
  {
   int statement=(timeframe==PERIOD_CURRENT ? ::Period() : timeframe);
   switch(statement)
     {
      case PERIOD_M1    :  return 0;
      case PERIOD_M2    :  return 1;
      case PERIOD_M3    :  return 2;
      case PERIOD_M4    :  return 3;
      case PERIOD_M5    :  return 4;
      case PERIOD_M6    :  return 5;
      case PERIOD_M10   :  return 6;
      case PERIOD_M12   :  return 7;
      case PERIOD_M15   :  return 8;
      case PERIOD_M20   :  return 9;
      case PERIOD_M30   :  return 10;
      case PERIOD_H1    :  return 11;
      case PERIOD_H2    :  return 12;
      case PERIOD_H3    :  return 13;
      case PERIOD_H4    :  return 14;
      case PERIOD_H6    :  return 15;
      case PERIOD_H8    :  return 16;
      case PERIOD_H12   :  return 17;
      case PERIOD_D1    :  return 18;
      case PERIOD_W1    :  return 19;
      case PERIOD_MN1   :  return 20;
      default           :  ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TS_TEXT_UNKNOWN_TIMEFRAME)); return WRONG_VALUE;
     }
  }
//+------------------------------------------------------------------+
//| Return a timeframe by index                                      |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES CTimeSeries::TimeframeByIndex(const uchar index) const
  {
   switch(index)

     {
      case 0   :  return PERIOD_M1;
      case 1   :  return PERIOD_M2;
      case 2   :  return PERIOD_M3;
      case 3   :  return PERIOD_M4;
      case 4   :  return PERIOD_M5;
      case 5   :  return PERIOD_M6;
      case 6   :  return PERIOD_M10;
      case 7   :  return PERIOD_M12;
      case 8   :  return PERIOD_M15;
      case 9   :  return PERIOD_M20;
      case 10  :  return PERIOD_M30;
      case 11  :  return PERIOD_H1;
      case 12  :  return PERIOD_H2;
      case 13  :  return PERIOD_H3;
      case 14  :  return PERIOD_H4;
      case 15  :  return PERIOD_H6;
      case 16  :  return PERIOD_H8;
      case 17  :  return PERIOD_H12;
      case 18  :  return PERIOD_D1;
      case 19  :  return PERIOD_W1;
      case 20  :  return PERIOD_MN1;
      default  :  ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_GET_DATAS),"... ",CMessage::Text(MSG_SYM_STATUS_INDEX),": ",(string)index); return WRONG_VALUE;
     }
  }
//+------------------------------------------------------------------+
```

Instead of the methods, we are now going to call the functions from the DELib.mqh file of service functions:

```
//--- Return (1) the timeframe index in the list and (2) the timeframe by the list index
   char              IndexTimeframe(const ENUM_TIMEFRAMES timeframe) const { return IndexEnumTimeframe(timeframe)-1;                            }
   ENUM_TIMEFRAMES   TimeframeByIndex(const uchar index)             const { return TimeframeByEnumIndex(uchar(index+1));                       }
```

Since the list of the class timeseries contains all possible chart periods, the zero list index contains the timeseries of M1 chart period. The ENUM\_TIMEFRAMES enumeration contains PERIOD\_CURRENT in the zero index, while М1 — in the first one, therefore we need to shift the index value by 1 to get the correct index in the list. This is exactly what we are doing here.

The private class section receives two class member variables for setting the very first date in history on the server and in the terminal, as well as the method setting the values of these dates in the variables:

```
//+------------------------------------------------------------------+
//| Symbol timeseries class                                          |
//+------------------------------------------------------------------+
class CTimeSeries : public CBaseObj
  {
private:
   string            m_symbol;                                             // Timeseries symbol
   CArrayObj         m_list_series;                                        // List of timeseries by timeframes
   datetime          m_server_firstdate;                                   // The very first date in history by a server symbol
   datetime          m_terminal_firstdate;                                 // The very first date in history by a symbol in the client terminal
//--- Return (1) the timeframe index in the list and (2) the timeframe by the list index
   char              IndexTimeframe(const ENUM_TIMEFRAMES timeframe) const { return IndexEnumTimeframe(timeframe)-1;                            }
   ENUM_TIMEFRAMES   TimeframeByIndex(const uchar index)             const { return TimeframeByEnumIndex(uchar(index+1));                       }
//--- Set the very first date in history by symbol on the server and in the client terminal
   void              SetTerminalServerDate(void)
                       {
                        this.m_server_firstdate=(datetime)::SeriesInfoInteger(this.m_symbol,::Period(),SERIES_SERVER_FIRSTDATE);
                        this.m_terminal_firstdate=(datetime)::SeriesInfoInteger(this.m_symbol,::Period(),SERIES_TERMINAL_FIRSTDATE);
                       }
public:
```

The **m\_server\_firstdate** variable will store the very first date in history by symbol on the server obtained by the [SeriesInfoInteger()](https://www.mql5.com/en/docs/series/seriesinfointeger) function with the [SERIES\_SERVER\_FIRSTDATE](https://www.mql5.com/en/docs/constants/tradingconstants/enum_series_info_integer) property ID.

The **m\_terminal\_firstdate** variable will store the very first data in history by symbol in the terminal obtained by the [SeriesInfoInteger()](https://www.mql5.com/en/docs/series/seriesinfointeger) function with the [SERIES\_TERMINAL\_FIRSTDATE](https://www.mql5.com/en/docs/constants/tradingconstants/enum_series_info_integer) property ID.

**Six new methods and the parametric constructor have been added in the public class section:**

```
public:
//--- Return (1) oneself, (2) the full list of timeseries, (3) specified timeseries object and (4) timeseries object by index
   CTimeSeries      *GetObject(void)                                       { return &this;                                                      }
   CArrayObj        *GetListSeries(void)                                   { return &this.m_list_series;                                        }
   CSeries          *GetSeries(const ENUM_TIMEFRAMES timeframe)            { return this.m_list_series.At(this.IndexTimeframe(timeframe));      }
   CSeries          *GetSeriesByIndex(const uchar index)                   { return this.m_list_series.At(index);                               }
//--- Set/return timeseries symbol
   void              SetSymbol(const string symbol)                        { this.m_symbol=(symbol==NULL || symbol=="" ? ::Symbol() : symbol);  }
   string            Symbol(void)                                    const { return this.m_symbol;                                              }
//--- Set the history depth (1) of a specified timeseries and (2) of all applied symbol timeseries
   bool              SetRequiredUsedData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool              SetRequiredAllUsedData(const uint required=0,const int rates_total=0);
//--- Return the flag of data synchronization with the server data of the (1) specified timeseries, (2) all timeseries
   bool              SyncData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool              SyncAllData(const uint required=0,const int rates_total=0);
//--- Return the very first date in history by symbol (1) on the server and (2) in the client terminal
   datetime          ServerFirstDate(void)                           const { return this.m_server_firstdate;                                    }
   datetime          TerminalFirstDate(void)                         const { return this.m_terminal_firstdate;                                  }
//--- Create (1) the specified timeseries list and (2) all timeseries lists
   bool              Create(const ENUM_TIMEFRAMES timeframe,const uint required=0);
   bool              CreateAll(const uint required=0);
//--- Update (1) the specified timeseries list and (2) all timeseries lists
   void              Refresh(const ENUM_TIMEFRAMES timeframe,
                             const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0);
   void              RefreshAll(const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0);
//--- Compare CTimeSeries objects (by symbol)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Display (1) description and (2) short symbol timeseries description in the journal
   void              Print(const bool created=true);
   void              PrintShort(const bool created=true);

//--- Constructors
                     CTimeSeries(void){;}
                     CTimeSeries(const string symbol);
  };
//+------------------------------------------------------------------+
```

The **GetObject()** method returns the pointer to the class object. It allows receiving the symbol timeseries class object and working with it in a custom program.

The **ServerFirstDate()** and **TerminalFirstDate()** methods return the values of the **m\_server\_firstdate** and **m\_terminal\_firstdate** variables discussed above.

The **Compare()** virtual method allows comparing two timeseries objects by the timeseries symbol name:

```
//+------------------------------------------------------------------+
//| Compare CTimeSeries objects                                      |
//+------------------------------------------------------------------+
int CTimeSeries::Compare(const CObject *node,const int mode=0) const
  {
   const CTimeSeries *compared_obj=node;
   return(this.Symbol()>compared_obj.Symbol() ? 1 : this.Symbol()<compared_obj.Symbol() ? -1 : 0);
  }
//+------------------------------------------------------------------+
```

The method returns zero in case the symbols of two compared timeseries objects are equal. Otherwise, it returns +/- 1. The method [is declared in the CObject class](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectcompare) of the standard library and should be redefined in its descendants.

The **Print()** method displays the full descriptions of all symbol timeseries in the journal:

```
//+------------------------------------------------------------------+
//| Display descriptions of all symbol timeseries in the journal     |
//+------------------------------------------------------------------+
void CTimeSeries::Print(const bool created=true)
  {
   ::Print(CMessage::Text(MSG_LIB_TEXT_TS_TEXT_SYMBOL_TIMESERIES)," ",this.m_symbol,": ");
   for(int i=0;i<this.m_list_series.Total();i++)
     {
      CSeries *series=this.m_list_series.At(i);
      if(series==NULL || (created && series.DataTotal()==0))
         continue;
      series.Print();
     }
  }
//+------------------------------------------------------------------+
```

The journal displays the list of all created (created=true) or created and declared (created=false) symbol timeseries in the appropriate format, for example

created=true:

```
GBPUSD symbol timeseries:
Timeseries "GBPUSD" M1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 6296
Timeseries "GBPUSD" M5: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 3921
Timeseries "GBPUSD" M15: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 3227
Timeseries "GBPUSD" M30: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 3053
Timeseries "GBPUSD" H1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 6187
Timeseries "GBPUSD" H4: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 5298
Timeseries "GBPUSD" D1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 5288
Timeseries "GBPUSD" W1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 1398
Timeseries "GBPUSD" MN1: Requested history depth: 1000, Actual history depth: 321, Historical data created: 321, History bars on the server: 321
```

created=false:

```
GBPUSD symbol timeseries:
Timeseries "GBPUSD" M1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 6296
Timeseries "GBPUSD" M2: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 5483
Timeseries "GBPUSD" M3: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 4616
Timeseries "GBPUSD" M4: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 4182
Timeseries "GBPUSD" M5: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 3921
Timeseries "GBPUSD" M6: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 3748
Timeseries "GBPUSD" M10: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 3401
Timeseries "GBPUSD" M12: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 3314
Timeseries "GBPUSD" M15: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 3227
Timeseries "GBPUSD" M20: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 3140
Timeseries "GBPUSD" M30: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 3053
Timeseries "GBPUSD" H1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 6187
Timeseries "GBPUSD" H2: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 5047
Timeseries "GBPUSD" H3: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 5031
Timeseries "GBPUSD" H4: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 5298
Timeseries "GBPUSD" H6: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 6324
Timeseries "GBPUSD" H8: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 6301
Timeseries "GBPUSD" H12: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 0, History bars on the server: 5762
Timeseries "GBPUSD" D1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 5288
Timeseries "GBPUSD" W1: Requested history depth: 1000, Actual history depth: 1000, Historical data created: 1000, History bars on the server: 1398
Timeseries "GBPUSD" MN1: Requested history depth: 1000, Actual history depth: 321, Historical data created: 321, History bars on the server: 321
```

The **PrintShort()** method displays short descriptions of all symbol timeseries in the journal:

```
//+-------------------------------------------------------------------+
//| Display short descriptions of all symbol timeseries in the journal|
//+-------------------------------------------------------------------+
void CTimeSeries::PrintShort(const bool created=true)
  {
   ::Print(CMessage::Text(MSG_LIB_TEXT_TS_TEXT_SYMBOL_TIMESERIES)," ",this.m_symbol,": ");
   for(int i=0;i<this.m_list_series.Total();i++)
     {
      CSeries *series=this.m_list_series.At(i);
      if(series==NULL || (created && series.DataTotal()==0))
         continue;
      series.PrintShort();
     }
  }
//+------------------------------------------------------------------+
```

The journal displays the list of all created (created=true) or created and declared (created=false) symbol timeseries in the appropriate format, for example

created=true:

```
USDJPY symbol timeseries:
Timeseries "USDJPY" M1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2880
Timeseries "USDJPY" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3921
Timeseries "USDJPY" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3227
Timeseries "USDJPY" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3053
Timeseries "USDJPY" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5095
Timeseries "USDJPY" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5023
Timeseries "USDJPY" D1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5305
Timeseries "USDJPY" W1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2562
Timeseries "USDJPY" MN1: Requested: 1000, Actual: 589, Created: 589, On the server: 589
```

created=false:

```
USDJPY symbol timeseries:
Timeseries "USDJPY" M1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2880
Timeseries "USDJPY" M2: Requested: 1000, Actual: 1000, Created: 0, On the server: 3608
Timeseries "USDJPY" M3: Requested: 1000, Actual: 1000, Created: 0, On the server: 4616
Timeseries "USDJPY" M4: Requested: 1000, Actual: 1000, Created: 0, On the server: 4182
Timeseries "USDJPY" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3921
Timeseries "USDJPY" M6: Requested: 1000, Actual: 1000, Created: 0, On the server: 3748
Timeseries "USDJPY" M10: Requested: 1000, Actual: 1000, Created: 0, On the server: 3401
Timeseries "USDJPY" M12: Requested: 1000, Actual: 1000, Created: 0, On the server: 3314
Timeseries "USDJPY" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3227
Timeseries "USDJPY" M20: Requested: 1000, Actual: 1000, Created: 0, On the server: 3140
Timeseries "USDJPY" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3053
Timeseries "USDJPY" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5095
Timeseries "USDJPY" H2: Requested: 1000, Actual: 1000, Created: 0, On the server: 5047
Timeseries "USDJPY" H3: Requested: 1000, Actual: 1000, Created: 0, On the server: 5031
Timeseries "USDJPY" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5023
Timeseries "USDJPY" H6: Requested: 1000, Actual: 1000, Created: 0, On the server: 6390
Timeseries "USDJPY" H8: Requested: 1000, Actual: 1000, Created: 0, On the server: 6352
Timeseries "USDJPY" H12: Requested: 1000, Actual: 1000, Created: 0, On the server: 5796
Timeseries "USDJPY" D1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5305
Timeseries "USDJPY" W1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2562
Timeseries "USDJPY" MN1: Requested: 1000, Actual: 589, Created: 589, On the server: 589
```

**The class constructor has received setting timeseries dates:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTimeSeries::CTimeSeries(const string symbol) : m_symbol(symbol)
  {
   this.m_list_series.Clear();
   this.m_list_series.Sort();
   for(int i=0;i<21;i++)
     {
      ENUM_TIMEFRAMES timeframe=this.TimeframeByIndex((uchar)i);
      CSeries *series_obj=new CSeries(this.m_symbol,timeframe);
      this.m_list_series.Add(series_obj);
     }
   this.SetTerminalServerDate();
  }
//+------------------------------------------------------------------+
```

**The specified timeseries update method features setting timeseries dates when the "New bar" event of the updated timeseries is detected:**

```
//+------------------------------------------------------------------+
//| Update a specified timeseries list                               |
//+------------------------------------------------------------------+
void CTimeSeries::Refresh(const ENUM_TIMEFRAMES timeframe,
                          const datetime time=0,
                          const double open=0,
                          const double high=0,
                          const double low=0,
                          const double close=0,
                          const long tick_volume=0,
                          const long volume=0,
                          const int spread=0)
  {
   CSeries *series_obj=this.m_list_series.At(this.IndexTimeframe(timeframe));
   if(series_obj==NULL || series_obj.DataTotal()==0)
      return;
   series_obj.Refresh(time,open,high,low,close,tick_volume,volume,spread);
   if(series_obj.IsNewBar(time))
      this.SetTerminalServerDate();
  }
//+------------------------------------------------------------------+
```

The method of updating all timeseries also features updating timeseries dates:

```
//+------------------------------------------------------------------+
//| Update all timeseries lists                                      |
//+------------------------------------------------------------------+
void CTimeSeries::RefreshAll(const datetime time=0,
                          const double open=0,
                          const double high=0,
                          const double low=0,
                          const double close=0,
                          const long tick_volume=0,
                          const long volume=0,
                          const int spread=0)
  {
   bool upd=false;
   for(int i=0;i<21;i++)
     {
      CSeries *series_obj=this.m_list_series.At(i);
      if(series_obj==NULL || series_obj.DataTotal()==0)
         continue;
      series_obj.Refresh(time,open,high,low,close,tick_volume,volume,spread);
      if(series_obj.IsNewBar(time))
         upd &=true;
     }
   if(upd)
      this.SetTerminalServerDate();
  }
//+------------------------------------------------------------------+
```

However, here we should update the dates only once — when detecting the "New bar" event in any of the updated timeseries present in the list (there are 21 timeseries, and we do not want to set the same dates 21 times). Therefore, the flag of the need to update the dates is set to true when the "New bar" event is met in a loop on all timeseries. Upon the loop completion and in case of the enabled flag, update the dates.

**This concludes the improvement of the CTimeSeries class. We do not consider minor corrections here. You can see all of them in the attached files.**

Currently, we have three classes containing all the necessary data for creating the timeseries collection:

1. CBar "single-period bar of a single symbol" includes the data of a single bar of specified symbol on a specified period;

2. CSeries "single-period timeseries of a single symbol" enables the bar list collection (1) of a single period of a single symbol;
3. CTimeSeries "all-periods timeseries of a single symbol" contains the list of timeseries (2) for each period of a single symbol


Now let's create the timeseries collection which is a timeseries list collection (3) of each symbol used in the program.

### Collection class of timeseries objects by symbols and periods

The timeseries collection is to consist of the [dynamic](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) [array of pointers to CObject class instances and its descendants](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) (the pointers to the [CTimeSeries](https://www.mql5.com/en/articles/7627#node02) class objects).

In the \\MQL5\\Include\\DoEasy\ **Collections\** library folder, create the file **TimeSeriesCollection.mqh** of the CTimeSeriesCollection class.

The base class object is a [base object for constructing the CObject standard library](https://www.mql5.com/en/docs/standardlibrary/cobject).

Let's have a look at the class listing:

```
//+------------------------------------------------------------------+
//|                                         TimeSeriesCollection.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\Objects\Series\TimeSeries.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
//+------------------------------------------------------------------+
//| Symbol timeseries collection                                     |
//+------------------------------------------------------------------+
class CTimeSeriesCollection : public CObject
  {
private:
   CArrayObj               m_list;                    // List of applied symbol timeseries
//--- Return the timeseries index by symbol name
   int                     IndexTimeSeries(const string symbol);
public:
//--- Return (1) oneself and (2) the timeseries list
   CTimeSeriesCollection  *GetObject(void)            { return &this;         }
   CArrayObj              *GetList(void)              { return &this.m_list;  }

//--- Create the symbol timeseries list collection
   bool                    CreateCollection(const CArrayObj *list_symbols);
//--- Set the flag of using (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   void                    SetAvailable(const string symbol,const ENUM_TIMEFRAMES timeframe,const bool flag=true);
   void                    SetAvailable(const ENUM_TIMEFRAMES timeframe,const bool flag=true);
   void                    SetAvailable(const string symbol,const bool flag=true);
   void                    SetAvailable(const bool flag=true);
//--- Get the flag of using (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                    IsAvailable(const string symbol,const ENUM_TIMEFRAMES timeframe);
   bool                    IsAvailable(const ENUM_TIMEFRAMES timeframe);
   bool                    IsAvailable(const string symbol);
   bool                    IsAvailable(void);

//--- Set the history depth of (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                    SetRequiredUsedData(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool                    SetRequiredUsedData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool                    SetRequiredUsedData(const string symbol,const uint required=0,const int rates_total=0);
   bool                    SetRequiredUsedData(const uint required=0,const int rates_total=0);
//--- Return the flag of data synchronization with the server data of the (1) specified timeseries of the specified symbol,
//--- (2) the specified timeseries of all symbols, (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                    SyncData(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool                    SyncData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0);
   bool                    SyncData(const string symbol,const uint required=0,const int rates_total=0);
   bool                    SyncData(const uint required=0,const int rates_total=0);
//--- Create (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                    CreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0);
   bool                    CreateSeries(const ENUM_TIMEFRAMES timeframe,const uint required=0);
   bool                    CreateSeries(const string symbol,const uint required=0);
   bool                    CreateSeries(const uint required=0);
//--- Update (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   void                    Refresh(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                   const datetime time=0,
                                   const double open=0,
                                   const double high=0,
                                   const double low=0,
                                   const double close=0,
                                   const long tick_volume=0,
                                   const long volume=0,
                                   const int spread=0);
   void                    Refresh(const ENUM_TIMEFRAMES timeframe,
                                   const datetime time=0,
                                   const double open=0,
                                   const double high=0,
                                   const double low=0,
                                   const double close=0,
                                   const long tick_volume=0,
                                   const long volume=0,
                                   const int spread=0);
   void                    Refresh(const string symbol,
                                   const datetime time=0,
                                   const double open=0,
                                   const double high=0,
                                   const double low=0,
                                   const double close=0,
                                   const long tick_volume=0,
                                   const long volume=0,
                                   const int spread=0);
   void                    Refresh(const datetime time=0,
                                   const double open=0,
                                   const double high=0,
                                   const double low=0,
                                   const double close=0,
                                   const long tick_volume=0,
                                   const long volume=0,
                                   const int spread=0);

//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(const bool created=true);
   void                    PrintShort(const bool created=true);


//--- Constructor
                           CTimeSeriesCollection();
  };
//+------------------------------------------------------------------+
```

At the moment, the class is a list of timeseries objects and methods of creating/setting parameters and updating the necessary timeseries by symbol and period:

The array of pointers to **m\_list** CObject class objects is to contain the pointers to the CTimeSeries class objects. This is a list, from which we get the data of the necessary timeseries to work with it.

The **IndexTimeSeries()** method returning the timeseries index by symbol name allows gaining access to the necessary CTimeSeries object by symbol name:

```
//+------------------------------------------------------------------+
//| Return the timeseries index by symbol name                       |
//+------------------------------------------------------------------+
int CTimeSeriesCollection::IndexTimeSeries(const string symbol)
  {
   CTimeSeries *tmp=new CTimeSeries(symbol);
   if(tmp==NULL)
      return WRONG_VALUE;
   this.m_list.Sort();
   int index=this.m_list.Search(tmp);
   delete tmp;
   return index;
  }
//+------------------------------------------------------------------+
```

Create a new temporary timeseries object with the symbol value passed to the method, set the sorted list flag for the **m\_list** list and use the [**Search()**](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) method to obtain the object index in the list.

Make sure to delete the temporary timeseries object and return the obtained index.

If such an object with a specified symbol is not in the list, the method returns -1, otherwise — the detected object index value.

The **GetObject()** method returns the pointer to the timeseries collection object to the control program. It allows obtaining the entire collection object and working with it in a custom program.

The **GetList()** method returns the pointer to the CTimeSeries timeseries collection list. It allows obtaining the list of all-symbol timeseries and working with it in a custom program.

The **CreateCollection()** method creates an empty collection of timeseries objects:

```
//+------------------------------------------------------------------+
//| Create the symbol timeseries collection list                     |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::CreateCollection(const CArrayObj *list_symbols)
  {
//--- If an empty list of symbol objects is passed, exit
   if(list_symbols==NULL)
      return false;
//--- Get the number of symbol objects in the passed list
   int total=list_symbols.Total();
//--- Clear the timeseries collection list
   this.m_list.Clear();
//--- In a loop by all symbol objects
   for(int i=0;i<total;i++)
     {
      //--- get the next symbol object
      CSymbol *symbol_obj=list_symbols.At(i);
      //--- if failed to get a symbol object, move on to the next one in the list
      if(symbol_obj==NULL)
         continue;
      //--- Create a new timeseries object with the current symbol name
      CTimeSeries *timeseries=new CTimeSeries(symbol_obj.Name());
      //--- If failed to create the timeseries object, move on to the next symbol in the list
      if(timeseries==NULL)
         continue;
      //--- Set the sorted list flag for the timeseries collection list
      this.m_list.Sort();
      //--- If the object with the same symbol name is already present in the timeseries collection list, remove the timeseries object
      if(this.m_list.Search(timeseries)>WRONG_VALUE)
         delete timeseries;
      //--- if failed to add the timeseries object to the collection list, remove the timeseries object
      else
         if(!this.m_list.Add(timeseries))
            delete timeseries;
     }
//--- Return the flag indicating that the created collection list has a size greater than zero
   return this.m_list.Total()>0;
  }
//+-----------------------------------------------------------------------+
```

Each method string is accompanied with comments in its listing.

The previously created list of all used symbols in the program is passed to the method, and the [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries objects are created in a loop along the list. Symbol names are specified for them immediately. Thus, we obtain the timeseries collection list by the number of symbols used in the program. All the remaining data of the created timeseries objects remains empty. It should be set separately.

This is done so that the library always has the collection list of empty timeseries in the amount equal to the number of symbols specified for work. Timeframes, as well as their timeseries objects (users are to work with in the program), are set at the next step or whenever required.

It is better to create them in the OnInit() handler of the program or immediately after its launch since creating a large number of timeseries takes time, especially in case of a "cold" program launch.

The four overloaded **SetAvailable()** methods are used to set the flag indicating the need to work in the program with the specified timeseries.

**The method for setting the flag of using the specified timeseries of the specified symbol:**

```
//+-----------------------------------------------------------------------+
//|Set the flag of using the specified timeseries of the specified symbol |
//+-----------------------------------------------------------------------+
void CTimeSeriesCollection::SetAvailable(const string symbol,const ENUM_TIMEFRAMES timeframe,const bool flag=true)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return;
   CSeries *series=timeseries.GetSeries(timeframe);
   if(series==NULL)
      return;
   series.SetAvailable(flag);
  }
//+------------------------------------------------------------------+
```

The method receives the symbol, the timeframe and the flag to be set for the timeseries corresponding to the timeframe symbol.

First, get the [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries index in the m\_list list by a symbol using the IndexTimeSeries() method discussed above. Get the timeseries from the list by the index. From the obtained timeseries object, get the required [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries of the specified chart period using the GetSeries() method described [in the previous article](https://www.mql5.com/en/articles/7627#node02) and set the flag passed to the method for it using the SetAvailable() method of the CBaseObj class.

**The method of setting the flag for using the specified timeseries for all symbols:**

```
//+------------------------------------------------------------------+
//|Set the flag of using the specified timeseries of all symbols     |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::SetAvailable(const ENUM_TIMEFRAMES timeframe,const bool flag=true)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      CSeries *series=timeseries.GetSeries(timeframe);
      if(series==NULL)
         continue;
      series.SetAvailable(flag);
     }
  }
//+------------------------------------------------------------------+
```

The method receives the timeframe and flag to be set for the specified timeseries of all symbols.

In the loop by the list of all symbol timeseries,get the next [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries by the loop index. From the obtained timeseries object, get the specified [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries of the specified chart period using the GetSeries() method we have considered [in the previous article](https://www.mql5.com/en/articles/7627#node02) and set the flag passed to the method for it using the SetAvailable() method of the CBaseObj class.

**The method for setting the flag of using all timeseries for the** **specified symbol:**

```
//+------------------------------------------------------------------+
//|Set the flag of using all timeseries of the specified symbol      |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::SetAvailable(const string symbol,const bool flag=true)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return;
   CArrayObj *list=timeseries.GetListSeries();
   if(list==NULL)
      return;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CSeries *series=list.At(i);
      if(series==NULL)
         continue;
      series.SetAvailable(flag);
     }
  }
//+------------------------------------------------------------------+
```

The method receives the symbol and the flag to be set for all timeseries of the specified symbol.

First, get the [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries index in the m\_list list by a symbol using the IndexTimeSeries() method discussed above. Get the CTimeSeries timeseries from the list by the index. From the obtained timeseries object, get the complete list of all [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries using the GetListSeries() method. In the loop by the obtained list, get the next CSeries timeseries and set the flag passed to the method for it using the SetAvailable() method of the CBaseObj class.

**The method of setting the flag of using all timeseries of all symbols** **:**

```
//+------------------------------------------------------------------+
//| Set the flag of using all timeseries of all symbols              |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::SetAvailable(const bool flag=true)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      CArrayObj *list=timeseries.GetListSeries();
      if(list==NULL)
         continue;
      int total_series=list.Total();
      for(int j=0;j<total_series;j++)
        {
         CSeries *series=list.At(j);
         if(series==NULL)
            continue;
         series.SetAvailable(flag);
        }
     }
  }
//+--------------------------------------------------------------------+
```

The method receives the flag that should be additionally set for all timeseries of all symbols.

In the loop by the timeseries list,get the next [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries object by the loop index. Use the GetListSeries() method to obtain the list of all its [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries from the obtained object. In the loop by the CSeries timeseries list, get the next timeseries by the loop index and set the flag passed to the method for it using the SetAvailable() method of the CBaseObj class.

**The four methods returning the flag of using specified or all timeseries:**

```
//+-------------------------------------------------------------------------+
//|Return the flag of using the specified timeseries of the specified symbol|
//+-------------------------------------------------------------------------+
bool CTimeSeriesCollection::IsAvailable(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
   CSeries *series=timeseries.GetSeries(timeframe);
   if(series==NULL)
      return false;
   return series.IsAvailable();
  }
//+------------------------------------------------------------------+
//| Return the flag of using the specified timeseries of all symbols |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::IsAvailable(const ENUM_TIMEFRAMES timeframe)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      CSeries *series=timeseries.GetSeries(timeframe);
      if(series==NULL)
         continue;
      res &=series.IsAvailable();
     }
   return res;
  }
//+------------------------------------------------------------------+
//| Return the flag of using all timeseries of the specified symbol  |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::IsAvailable(const string symbol)
  {
   bool res=true;
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
   CArrayObj *list=timeseries.GetListSeries();
   if(list==NULL)
      return false;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CSeries *series=list.At(i);
      if(series==NULL)
         continue;
      res &=series.IsAvailable();
     }
   return res;
  }
//+------------------------------------------------------------------+
//| Return the flag of using all timeseries of all symbols           |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::IsAvailable(void)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      CArrayObj *list=timeseries.GetListSeries();
      if(list==NULL)
         continue;
      int total_series=list.Total();
      for(int j=0;j<total_series;j++)
        {
         CSeries *series=list.At(j);
         if(series==NULL)
            continue;
         res &=series.IsAvailable();
        }
     }
   return res;
  }
//+--------------------------------------------------------------------+
```

The methods work similarly to the methods of setting the flag for timeseries I have just described above. The difference is in using the **res** local variable having the initial state of true to get the "collective" flag from all timeseries in the methods returning the common flag of using multiple timeseries. In the loop by the [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries, the flag status of each checked timeseries is written to the variable. If at least one of the timeseries is equal to false, the false flag is set in the variable. The value of the variable is returned from the method upon completion of all loops in all timeseries.

**The four methods for setting the required depth history either for specified timeseries or for all timeseries at once:**

```
//+--------------------------------------------------------------------------+
//|Set the history depth for the specified timeseries of the specified symbol|
//+--------------------------------------------------------------------------+
bool CTimeSeriesCollection::SetRequiredUsedData(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
   CSeries *series=timeseries.GetSeries(timeframe);
   if(series==NULL)
      return false;
   return series.SetRequiredUsedData(required,rates_total);
  }
//+------------------------------------------------------------------+
//| Set the history depth of the specified timeseries of all symbols |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::SetRequiredUsedData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      CSeries *series=timeseries.GetSeries(timeframe);
      if(series==NULL)
         continue;
      res &=series.SetRequiredUsedData(required,rates_total);
     }
   return res;
  }
//+------------------------------------------------------------------+
//| Set the history depth for all timeseries of the specified symbol |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::SetRequiredUsedData(const string symbol,const uint required=0,const int rates_total=0)
  {
   bool res=true;
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
   CArrayObj *list=timeseries.GetListSeries();
   if(list==NULL)
      return false;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CSeries *series=list.At(i);
      if(series==NULL)
         continue;
      res &=series.SetRequiredUsedData(required,rates_total);
     }
   return res;
  }
//+------------------------------------------------------------------+
//| Set the history depth for all timeseries of all symbols          |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::SetRequiredUsedData(const uint required=0,const int rates_total=0)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      CArrayObj *list=timeseries.GetListSeries();
      if(list==NULL)
         continue;
      int total_series=list.Total();
      for(int j=0;j<total_series;j++)
        {
         CSeries *series=list.At(j);
         if(series==NULL)
            continue;
         res &=series.SetRequiredUsedData(required,rates_total);
        }
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The methods work similarly to the methods returning timeseries usage flags. They return the result of setting the requested amount of data for [CSeries](https://www.mql5.com/en/articles/7594#node04) timeseries objects using the SetRequiredUsedData() methods returning the boolean value. Therefore, the common flag used when setting the history depth for multiple CSeries timeseries is applied here as well.

**The four methods returning the flag of synchronizing either a specified or all timeseries:**

```
//+------------------------------------------------------------------+
//| Return the flag of data synchronization with the server data     |
//| for a specified timeseries of a specified symbol                 |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::SyncData(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
   CSeries *series=timeseries.GetSeries(timeframe);
   if(series==NULL)
      return false;
   return series.SyncData(required,rates_total);
  }
//+------------------------------------------------------------------+
//| Return the flag of data synchronization with the server data     |
//| for a specified timeseries of all symbols                        |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::SyncData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      CSeries *series=timeseries.GetSeries(timeframe);
      if(series==NULL)
         continue;
      res &=series.SyncData(required,rates_total);
     }
   return res;
  }
//+------------------------------------------------------------------+
//| Return the flag of data synchronization with the server data     |
//| for all timeseries of a specified symbol                         |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::SyncData(const string symbol,const uint required=0,const int rates_total=0)
  {
   bool res=true;
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
   CArrayObj *list=timeseries.GetListSeries();
   if(list==NULL)
      return false;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      CSeries *series=list.At(i);
      if(series==NULL)
         continue;
      res &=series.SyncData(required,rates_total);
     }
   return res;
  }
//+------------------------------------------------------------------+
//| Return the flag of data synchronization with the server data     |
//| for all timeseries of all symbols                                |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::SyncData(const uint required=0,const int rates_total=0)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      CArrayObj *list=timeseries.GetListSeries();
      if(list==NULL)
         continue;
      int total_series=list.Total();
      for(int j=0;j<total_series;j++)
        {
         CSeries *series=list.At(j);
         if(series==NULL)
            continue;
         res &=series.SyncData(required,rates_total);
        }
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The methods work similarly to the above ones. They return the result of checking the timeseries synchronization using the SyncData() method of the [CSeries](https://www.mql5.com/en/articles/7594#node04) class.

The four methods for creating specified or all timeseries.

**The method for creating the specified timeseries of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Create the specified timeseries of the specified symbol          |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::CreateSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
   return timeseries.Create(timeframe,required);
  }
//+------------------------------------------------------------------+
```

The method receives a symbol the timeseries should be created for. The timeseries period is also passed to the method.

Get the timeseries index in the list by a symbol name using the IndexTimeSeries() method. Using the obtained index, get the [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries from the list and return the result of creating a specified timeseries using the Create() method of the CTimeSeries class.

**The method for creating the specified timeseries of all symbols:**

```
//+------------------------------------------------------------------+
//| Create the specified timeseries of all symbols                   |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::CreateSeries(const ENUM_TIMEFRAMES timeframe,const uint required=0)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      res &=timeseries.Create(timeframe,required);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the timeframe whose timeseries should be created for all symbols.

In the loop by all objects of all-object timeseries, get the next [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries object by the loop index. To the **res** variable, add the result of creating a specified timeseries using the Create() method of the CTimeSeries class. Upon the loop completion, return the result of creating a specified timeseries for all symbols. If at least one timeseries has not been created, the result is false.

**The method for creating all timeseries of the specified symbol:**

```
//+------------------------------------------------------------------+
//| Create all timeseries of the specified symbol                    |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::CreateSeries(const string symbol,const uint required=0)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return false;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return false;
   return timeseries.CreateAll(required);
  }
//+------------------------------------------------------------------+
```

The method receives the symbol all timeseries should be created for.

Get the timeseries index in the list by a symbol name using the IndexTimeSeries() method. Using the obtained index, get the [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries from the list and return the result of creating all timeseries using the CreateAll() method of the CTimeSeries class.

**The method for creating all timeseries of all symbols:**

```
//+------------------------------------------------------------------+
//| Create all timeseries of all symbols                             |
//+------------------------------------------------------------------+
bool CTimeSeriesCollection::CreateSeries(const uint required=0)
  {
   bool res=true;
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      res &=timeseries.CreateAll(required);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives only the number of created history depth (as in all the above methods). Zero is passed by default. This means the history depth of 1000 bars set by the SERIES\_DEFAULT\_BARS\_COUNT macro substitution in the Defines.mqh file.

In the loop by the timeseries list,get the next [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries object by the loop index. The **res** variable receives the result of creating all timeseries for the CTimeSeries current object symbol using the CreateAll() method returning the flag of creating all symbol timeseries of the CTimeSeries object.

Upon the loop completion, return the result of creating all timeseries for all symbols. If at least one timeseries has not been created, the result is false.

**The four methods for updating the specified timeseries of the specified symbol or all timeseries:**

```
//+------------------------------------------------------------------+
//| Update the specified timeseries of the specified symbol          |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                    const datetime time=0,
                                    const double open=0,
                                    const double high=0,
                                    const double low=0,
                                    const double close=0,
                                    const long tick_volume=0,
                                    const long volume=0,
                                    const int spread=0)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return;
   timeseries.Refresh(timeframe,time,open,high,low,close,tick_volume,volume,spread);
  }
//+------------------------------------------------------------------+
//| Update the specified timeseries of all symbols                   |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const ENUM_TIMEFRAMES timeframe,
                                    const datetime time=0,
                                    const double open=0,
                                    const double high=0,
                                    const double low=0,
                                    const double close=0,
                                    const long tick_volume=0,
                                    const long volume=0,
                                    const int spread=0)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      timeseries.Refresh(timeframe,time,open,high,low,close,tick_volume,volume,spread);
     }
  }
//+------------------------------------------------------------------+
//| Update all timeseries of the specified symbol                    |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const string symbol,
                                    const datetime time=0,
                                    const double open=0,
                                    const double high=0,
                                    const double low=0,
                                    const double close=0,
                                    const long tick_volume=0,
                                    const long volume=0,
                                    const int spread=0)
  {
   int index=this.IndexTimeSeries(symbol);
   if(index==WRONG_VALUE)
      return;
   CTimeSeries *timeseries=this.m_list.At(index);
   if(timeseries==NULL)
      return;
   timeseries.RefreshAll(time,open,high,low,close,tick_volume,volume,spread);
  }
//+------------------------------------------------------------------+
//| Update all timeseries of all symbols                             |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Refresh(const datetime time=0,
                                    const double open=0,
                                    const double high=0,
                                    const double low=0,
                                    const double close=0,
                                    const long tick_volume=0,
                                    const long volume=0,
                                    const int spread=0)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      timeseries.RefreshAll(time,open,high,low,close,tick_volume,volume,spread);
     }
  }
//+------------------------------------------------------------------+
```

Here, we get the required [CTimeSeries](https://www.mql5.com/en/articles/7627) timeseries object in one way or another (all ways have been considered in the previous methods) and call the methods of updating a single specified Refresh() timeseries or all RefreshAll() timeseries of the CTimeSeries class at once.

[The current data from the timeseries arrays](https://www.mql5.com/en/docs/event_handlers/oncalculate) are passed to all methods. This is necessary to work in indicators on the current period of the current symbol. In the remaining cases, the passed values are of no importance. Therefore, set their default values to 0.

**The method returning the full collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display complete collection description to the journal           |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::Print(const bool created=true)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      timeseries.Print(created);
     }
  }
//+------------------------------------------------------------------+
```

The method receives the flag indicating the necessity to display only created timeseries in the journal.

In the loop by the list of timeseries objects,get the next CTimeSeries timeseries object and call its same-name method whose operation result has been considered above. As a result, the data of all present timeseries of all collection symbols are available in the journal.

**The method returning the short collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display the short collection description in the journal          |
//+------------------------------------------------------------------+
void CTimeSeriesCollection::PrintShort(const bool created=true)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=this.m_list.At(i);
      if(timeseries==NULL)
         continue;
      timeseries.PrintShort(created);
     }
  }
//+------------------------------------------------------------------+
```

The method receives the flag indicating the necessity to display only created timeseries in the journal.

In the loop by the list of timeseries objects,get the next CTimeSeries timeseries object and call its same-name method whose operation result has been considered above. As a result, the data of all present timeseries of all collection symbols are available in the journal in the brief form.

**The current version of the timeseries collection class is ready. See the full class listing in the attached files.**

Now we need to arrange the external access to the created timeseries collection and convenient setting of parameters of created timeseries.

Any program gets access to the library methods from the [CEngine](https://www.mql5.com/en/articles/5687#node02) library main object.

Add the ability to access the timeseries collection to the CEngine class file.

Open the **Engine.mqh** file from the \\MQL5\\Include\ **DoEasy\** library directory and make the necessary changes.

Since the CTimeSeries class file was included to the CEngine class in the previous version only to check its operation, remove it from the list of included files

```
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
#include "Objects\Series\TimeSeries.mqh"
//+------------------------------------------------------------------+
```

and include the file of the timeseries collection class:

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
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
#include "Collections\TimeSeriesCollection.mqh"
#include "TradingControl.mqh"
//+------------------------------------------------------------------+
```

**In the private class section, declare the variable with the timeseries collection class type:**

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
   CTimeSeriesCollection m_series;                       // Timeseries collection
   CResourceCollection  m_resource;                      // Resource list
   CTradingControl      m_trading;                       // Trading management object
   CArrayObj            m_list_counters;                 // List of timer counters
```

In the public section of the class, change implementation of the method for setting the list of used symbols:

```
//--- Set the list of (1) used symbols
   bool                 SetUsedSymbols(const string &array_symbols[])   { return this.m_symbols.SetUsedSymbols(array_symbols);}
```

In the current implementation, the method calls the same-name method of setting the list of symbols for symbol collection class. Here, it would be most convenient for us to create the timeseries collection based on the created symbol collection list immediately after setting the symbol list in the symbol collection.

Leave only the method declaration here:

```
//--- Set the list of used symbols in the symbol collection and create the collection of symbol timeseries
   bool                 SetUsedSymbols(const string &array_symbols[]);
```

and write its implementation beyond the class body:

```
//+------------------------------------------------------------------+
//| Set the list of used symbols in the symbol collection            |
//| and create the symbol timeseries collection                      |
//+------------------------------------------------------------------+
bool CEngine::SetUsedSymbols(const string &array_symbols[])
  {
   bool res=this.m_symbols.SetUsedSymbols(array_symbols);
   CArrayObj *list=this.GetListAllUsedSymbols();
   if(list==NULL)
      return false;
   res&=this.m_series.CreateCollection(list);
   return res;
  }
//+------------------------------------------------------------------+
```

Here, declare the **res** variable and initialize it by the result of the method for setting the symbol list in the symbol collection.

Next, get the list of used symbols from the symbol collection class. If the list is not created, return false, otherwise, add the result of creating the timeseries collection list to the **res** variable based on the symbol collection list.

Return the final result from the method.

**Add the methods for working with the timeseries collection to the public class section:**

```
//--- Return the list of pending requests
   CArrayObj           *GetListPendingRequests(void)                          { return this.m_trading.GetListRequests();                        }

//--- Return (1) the timeseries collection and (2) the list of timeseries from the timeseries collection
   CTimeSeriesCollection *GetTimeSeriesCollection(void)                       { return &this.m_series;                                          }
   CArrayObj           *GetListTimeSeries(void)                               { return this.m_series.GetList();                                 }
//--- Set the flag of using (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   void                 SeriesSetAvailable(const string symbol,const ENUM_TIMEFRAMES timeframe,const bool flag=true)
                          { this.m_series.SetAvailable(symbol,timeframe,flag);}
   void                 SeriesSetAvailable(const ENUM_TIMEFRAMES timeframe,const bool flag=true)
                          { this.m_series.SetAvailable(timeframe,flag);       }
   void                 SeriesSetAvailable(const string symbol,const bool flag=true)
                          { this.m_series.SetAvailable(symbol,flag);          }
   void                 SeriesSetAvailable(const bool flag=true)
                          { this.m_series.SetAvailable(flag);                 }
//--- Set the history depth of (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                 SeriesSetRequiredUsedData(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
                          { return this.m_series.SetRequiredUsedData(symbol,timeframe,required,rates_total);}
   bool                 SeriesSetRequiredUsedData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
                          { return this.m_series.SetRequiredUsedData(timeframe,required,rates_total);       }
   bool                 SeriesSetRequiredUsedData(const string symbol,const uint required=0,const int rates_total=0)
                          { return this.m_series.SetRequiredUsedData(symbol,required,rates_total);          }
   bool                 SeriesSetRequiredUsedData(const uint required=0,const int rates_total=0)
                          { return this.m_series.SetRequiredUsedData(required,rates_total);                 }
//--- Return the flag of data synchronization with the server data of the (1) specified timeseries of the specified symbol,
//--- (2) the specified timeseries of all symbols, (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                 SeriesSyncData(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
                          { return this.m_series.SyncData(symbol,timeframe,required,rates_total);  }
   bool                 SeriesSyncData(const ENUM_TIMEFRAMES timeframe,const uint required=0,const int rates_total=0)
                          { return this.m_series.SyncData(timeframe,required,rates_total);         }
   bool                 SeriesSyncData(const string symbol,const uint required=0,const int rates_total=0)
                          { return this.m_series.SyncData(symbol,required,rates_total);            }
   bool                 SeriesSyncData(const uint required=0,const int rates_total=0)
                          { return this.m_series.SyncData(required,rates_total);                   }
//--- Create (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   bool                 SeriesCreate(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0)
                          { return this.m_series.CreateSeries(symbol,timeframe,required);          }
   bool                 SeriesCreate(const ENUM_TIMEFRAMES timeframe,const uint required=0)
                          { return this.m_series.CreateSeries(timeframe,required);                 }
   bool                 SeriesCreate(const string symbol,const uint required=0)
                          { return this.m_series.CreateSeries(symbol,required);                    }
   bool                 SeriesCreate(const uint required=0)
                          { return this.m_series.CreateSeries(required);                           }
//--- Update (1) the specified timeseries of the specified symbol, (2) the specified timeseries of all symbols,
//--- (3) all timeseries of the specified symbol and (4) all timeseries of all symbols
   void                 SeriesRefresh(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                      const datetime time=0,
                                      const double open=0,
                                      const double high=0,
                                      const double low=0,
                                      const double close=0,
                                      const long tick_volume=0,
                                      const long volume=0,
                                      const int spread=0)
                          { this.m_series.Refresh(symbol,timeframe,time,open,high,low,close,tick_volume,volume,spread); }
   void                 SeriesRefresh(const ENUM_TIMEFRAMES timeframe,
                                      const datetime time=0,
                                      const double open=0,
                                      const double high=0,
                                      const double low=0,
                                      const double close=0,
                                      const long tick_volume=0,
                                      const long volume=0,
                                      const int spread=0)
                          { this.m_series.Refresh(timeframe,time,open,high,low,close,tick_volume,volume,spread);        }
   void                 SeriesRefresh(const string symbol,
                                      const datetime time=0,
                                      const double open=0,
                                      const double high=0,
                                      const double low=0,
                                      const double close=0,
                                      const long tick_volume=0,
                                      const long volume=0,
                                      const int spread=0)
                          { this.m_series.Refresh(symbol,time,open,high,low,close,tick_volume,volume,spread);           }
   void                 SeriesRefresh(const datetime time=0,
                                      const double open=0,
                                      const double high=0,
                                      const double low=0,
                                      const double close=0,
                                      const long tick_volume=0,
                                      const long volume=0,
                                      const int spread=0)
                          { this.m_series.Refresh(time,open,high,low,close,tick_volume,volume,spread);                  }
```

The **GetTimeSeriesCollection()** method returns the pointer to the timeseries collection object to the calling program. It allows receiving the pointer to the collection in the program and work with it efficiently.

The **GetListTimeSeries()** method returns the pointer to the collection timeseries list to the calling program. It allows receiving the pointer to the CTimeSeries timeseries object list and work with it efficiently.

The **SeriesSetAvailable()** overloaded methods provide access to the SetAvailable() methods of the CTimeSeriesCollection() timeseries collection class we have considered above.

The **SeriesSetRequiredUsedData()** overloaded methods provide access to the SetRequiredUsedData() methods of the CTimeSeriesCollection() timeseries collection class we have considered above.

The **SeriesSyncData()** overloaded methods provide access to the SyncData() methods of the CTimeSeriesCollection() timeseries collection class we have considered above.

The **SeriesCreate()** overloaded methods provide access to the CreateSeries() methods of the CTimeSeriesCollection() timeseries collection class we have considered above.

The **SeriesRefresh()** overloaded methods provide access to the Refresh() methods of the CTimeSeriesCollection() timeseries collection class we have considered above.

**These are all the currently necessary tasks of handling all classes for testing a new timeseries collection class.**

### Testing

Let's use [the EA from the previous article](https://www.mql5.com/en/articles/7627#node03) to test creation and filling in the timeseries collection. Save it in the new folder \\MQL5\\Experts\\TestDoEasy\ **Part37\** under the name **TestDoEasyPart37.mq5**.

To select the mode of working with symbols, we have the enumeration in the Datas.mqh file.

```
//+------------------------------------------------------------------+
//| Modes of working with symbols                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOLS_MODE
  {
   SYMBOLS_MODE_CURRENT,                              // Work with the current symbol only
   SYMBOLS_MODE_DEFINES,                              // Work with the specified symbol list
   SYMBOLS_MODE_MARKET_WATCH,                         // Work with the Market Watch window symbols
   SYMBOLS_MODE_ALL                                   // Work with the full symbol list
  };
//+------------------------------------------------------------------+
```

The EA inputs feature the variable allowing us to select symbols to work with them:

```
sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;            // Mode of used symbols list
```

Depending on the variable value, the array of symbols is created and passed to the library during its initialization in the OnInitDoEasy() EA function. The list of working symbols is then created in the library.

We need to perform the same operations to select and create the list of working EA timeframes.

**Create a new enumeration in the Datas.mqh file to select operation modes with symbol chart periods:**

```
//+------------------------------------------------------------------+
//|                                                        Datas.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#define INPUT_SEPARATOR                (",")          // Separator in the inputs string
#define TOTAL_LANG                     (2)            // Number of used languages
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Modes of working with symbols                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOLS_MODE
  {
   SYMBOLS_MODE_CURRENT,                              // Work with the current symbol only
   SYMBOLS_MODE_DEFINES,                              // Work with the specified symbol list
   SYMBOLS_MODE_MARKET_WATCH,                         // Work with the Market Watch window symbols
   SYMBOLS_MODE_ALL                                   // Work with the full symbol list
  };
//+------------------------------------------------------------------+
//| Mode of working with timeframes                                  |
//+------------------------------------------------------------------+
enum ENUM_TIMEFRAMES_MODE
  {
   TIMEFRAMES_MODE_CURRENT,                           // Work with the current timeframe only
   TIMEFRAMES_MODE_LIST,                              // Work with the specified timeframe list
   TIMEFRAMES_MODE_ALL                                // Work with the full timeframe list
  };
//+------------------------------------------------------------------+
```

The file of service functions features the function for preparing the array of used library symbols:

```
//+------------------------------------------------------------------+
//| Prepare the symbol array for a symbol collection                 |
//+------------------------------------------------------------------+
bool CreateUsedSymbolsArray(const ENUM_SYMBOLS_MODE mode_used_symbols,string defined_used_symbols,string &used_symbols_array[])
  {
   //--- When working with the current symbol
   if(mode_used_symbols==SYMBOLS_MODE_CURRENT)
     {
      //--- Write the name of the current symbol to the only array cell
      ArrayResize(used_symbols_array,1);
      used_symbols_array[0]=Symbol();
      return true;
     }
   //--- If working with a predefined symbol set (from the defined_used_symbols string)
   else if(mode_used_symbols==SYMBOLS_MODE_DEFINES)
     {
      //--- Set a comma as a separator
      string separator=",";
      //--- Replace erroneous separators with correct ones
      if(StringFind(defined_used_symbols,";")>WRONG_VALUE)  StringReplace(defined_used_symbols,";",separator);
      if(StringFind(defined_used_symbols,":")>WRONG_VALUE)  StringReplace(defined_used_symbols,":",separator);
      if(StringFind(defined_used_symbols,"|")>WRONG_VALUE)  StringReplace(defined_used_symbols,"|",separator);
      if(StringFind(defined_used_symbols,"/")>WRONG_VALUE)  StringReplace(defined_used_symbols,"/",separator);
      if(StringFind(defined_used_symbols,"\\")>WRONG_VALUE) StringReplace(defined_used_symbols,"\\",separator);
      if(StringFind(defined_used_symbols,"'")>WRONG_VALUE)  StringReplace(defined_used_symbols,"'",separator);
      if(StringFind(defined_used_symbols,"-")>WRONG_VALUE)  StringReplace(defined_used_symbols,"-",separator);
      if(StringFind(defined_used_symbols,"`")>WRONG_VALUE)  StringReplace(defined_used_symbols,"`",separator);
      //--- Delete as long as there are spaces
      while(StringFind(defined_used_symbols," ")>WRONG_VALUE && !IsStopped())
         StringReplace(defined_used_symbols," ","");
      //--- As soon as there are double separators (after removing spaces between them), replace them with a separator
      while(StringFind(defined_used_symbols,separator+separator)>WRONG_VALUE && !IsStopped())
         StringReplace(defined_used_symbols,separator+separator,separator);
      //--- If a single separator remains before the first symbol in the string, replace it with a space
      if(StringFind(defined_used_symbols,separator)==0)
         StringSetCharacter(defined_used_symbols,0,32);
      //--- If a single separator remains after the last symbol in the string, replace it with a space
      if(StringFind(defined_used_symbols,separator)==StringLen(defined_used_symbols)-1)
         StringSetCharacter(defined_used_symbols,StringLen(defined_used_symbols)-1,32);
      //--- Remove all redundant things to the left and right
      #ifdef __MQL5__
         StringTrimLeft(defined_used_symbols);
         StringTrimRight(defined_used_symbols);
      //---  __MQL4__
      #else
         defined_used_symbols=StringTrimLeft(defined_used_symbols);
         defined_used_symbols=StringTrimRight(defined_used_symbols);
      #endif
      //--- Prepare the array
      ArrayResize(used_symbols_array,0);
      ResetLastError();
      //--- divide the string by separators (comma) and add all found substrings to the array
      int n=StringSplit(defined_used_symbols,StringGetCharacter(separator,0),used_symbols_array);
      //--- if nothing is found, display the appropriate message (working with the current symbol is selected automatically)
      if(n<1)
        {
         string err=
           (n==0  ?
            DFUN_ERR_LINE+CMessage::Text(MSG_LIB_SYS_ERROR_EMPTY_STRING)+Symbol() :
            DFUN_ERR_LINE+CMessage::Text(MSG_LIB_SYS_FAILED_PREPARING_SYMBOLS_ARRAY)+(string)GetLastError()
           );
         Print(err);
         return false;
        }
     }
   //--- If working with the Market Watch window or the full list
   else
     {
      //--- Add the (mode_used_symbols) working mode to the only array cell
      ArrayResize(used_symbols_array,1);
      used_symbols_array[0]=EnumToString(mode_used_symbols);
     }
   return true;
  }
//+------------------------------------------------------------------+
```

We should have a similar function to prepare the array of used timeframes. It becomes clear that we will have the same code block (highlighted by color in the provided listing) in two similar functions.

**Let's move this code block into a separate function:**

```
//+------------------------------------------------------------------+
//| Prepare the passed string of parameters                          |
//+------------------------------------------------------------------+
int StringParamsPrepare(string defined_used,string separator,string &array[])
  {
//--- Replace erroneous separators with correct ones
   if(separator!=";" && StringFind(defined_used,";")>WRONG_VALUE)  StringReplace(defined_used,";",separator);
   if(separator!=":" && StringFind(defined_used,":")>WRONG_VALUE)  StringReplace(defined_used,":",separator);
   if(separator!="|" && StringFind(defined_used,"|")>WRONG_VALUE)  StringReplace(defined_used,"|",separator);
   if(separator!="/" && StringFind(defined_used,"/")>WRONG_VALUE)  StringReplace(defined_used,"/",separator);
   if(separator!="\\"&& StringFind(defined_used,"\\")>WRONG_VALUE) StringReplace(defined_used,"\\",separator);
   if(separator!="'" && StringFind(defined_used,"'")>WRONG_VALUE)  StringReplace(defined_used,"'",separator);
   if(separator!="-" && StringFind(defined_used,"-")>WRONG_VALUE)  StringReplace(defined_used,"-",separator);
   if(separator!="`" && StringFind(defined_used,"`")>WRONG_VALUE)  StringReplace(defined_used,"`",separator);
//--- Delete as long as there are spaces
   while(StringFind(defined_used," ")>WRONG_VALUE && !IsStopped())
      StringReplace(defined_used," ","");
//--- As soon as there are double separators (after removing spaces between them), replace them with a separator
   while(StringFind(defined_used,separator+separator)>WRONG_VALUE && !IsStopped())
      StringReplace(defined_used,separator+separator,separator);
//--- If a single separator remains before the first symbol in the string, replace it with a space
   if(StringFind(defined_used,separator)==0)
      StringSetCharacter(defined_used,0,32);
//--- If a single separator remains after the last symbol in the string, replace it with a space
   if(StringFind(defined_used,separator)==StringLen(defined_used)-1)
      StringSetCharacter(defined_used,StringLen(defined_used)-1,32);
//--- Remove all redundant things to the left and right
   #ifdef __MQL5__
      StringTrimLeft(defined_used);
      StringTrimRight(defined_used);
//---  __MQL4__
   #else
      defined_used=StringTrimLeft(defined_used);
      defined_used=StringTrimRight(defined_used);
   #endif
//--- Prepare the array
   ArrayResize(array,0);
   ResetLastError();
//--- divide the string by separators (comma), write all detected substrings into the array and return the number of obtained substrings
   return StringSplit(defined_used,StringGetCharacter(separator,0),array);
  }
//+------------------------------------------------------------------+
```

In this case, the function for creating the array of used symbols will look as follows:

```
//+------------------------------------------------------------------+
//| Prepare the symbol array for a symbol collection                 |
//+------------------------------------------------------------------+
bool CreateUsedSymbolsArray(const ENUM_SYMBOLS_MODE mode_used_symbols,string defined_used_symbols,string &used_symbols_array[])
  {
//--- When working with the current symbol
   if(mode_used_symbols==SYMBOLS_MODE_CURRENT)
     {
      //--- Write the name of the current symbol to the only array cell
      ArrayResize(used_symbols_array,1);
      used_symbols_array[0]=Symbol();
      return true;
     }
//--- If working with a predefined symbol set (from the defined_used_symbols string)
   else if(mode_used_symbols==SYMBOLS_MODE_DEFINES)
     {
      //--- Set comma as a separator (defined in the Datas.mqh file, page 11)
      string separator=INPUT_SEPARATOR;
      int n=StringParamsPrepare(defined_used_symbols,separator,used_symbols_array);
      //--- if nothing is found, display the appropriate message (working with the current symbol is selected automatically)
      if(n<1)
        {
         int err_code=GetLastError();
         string err=
           (n==0  ?
            DFUN_ERR_LINE+CMessage::Text(MSG_LIB_SYS_ERROR_EMPTY_SYMBOLS_STRING)+Symbol() :
            DFUN_ERR_LINE+CMessage::Text(MSG_LIB_SYS_FAILED_PREPARING_SYMBOLS_ARRAY)+(string)err_code+": "+CMessage::Text(err_code)
           );
         Print(err);
         return false;
        }
     }
//--- If working with the Market Watch window or the full list
   else
     {
      //--- Add the (mode_used_symbols) working mode to the only array cell
      ArrayResize(used_symbols_array,1);
      used_symbols_array[0]=EnumToString(mode_used_symbols);
     }
//--- All is successful
   return true;
  }
//+------------------------------------------------------------------+
```

The code block moved into a separate function is replaced with the function for preparing the string of parameters.

Let's create the function for preparing the array of timeframes used in the program in similar fashion:

```
//+------------------------------------------------------------------+
//| Prepare the array of timeframes for the timeseries collection    |
//+------------------------------------------------------------------+
bool CreateUsedTimeframesArray(const ENUM_TIMEFRAMES_MODE mode_used_periods,string defined_used_periods,string &used_periods_array[])
  {
//--- If working with the current chart period, set the current timeframe flag
   if(mode_used_periods==TIMEFRAMES_MODE_CURRENT)
     {
      ArrayResize(used_periods_array,1,21);
      used_periods_array[0]=TimeframeDescription((ENUM_TIMEFRAMES)Period());
      return true;
     }
//--- If working with a predefined set of chart periods (from the defined_used_periods string)
   else if(mode_used_periods==TIMEFRAMES_MODE_LIST)
     {
      //--- Set comma as a separator (defined in the Datas.mqh file, page 11)
      string separator=INPUT_SEPARATOR;
      //--- Fill in the array of parameters from the string with predefined timeframes
      int n=StringParamsPrepare(defined_used_periods,separator,used_periods_array);
      //--- if nothing is found, display the appropriate message (working with the current period is selected automatically)
      if(n<1)
        {
         int err_code=GetLastError();
         string err=
           (n==0  ?
            DFUN_ERR_LINE+CMessage::Text(MSG_LIB_SYS_ERROR_EMPTY_PERIODS_STRING)+TimeframeDescription((ENUM_TIMEFRAMES)Period()) :
            DFUN_ERR_LINE+CMessage::Text(MSG_LIB_SYS_FAILED_PREPARING_PERIODS_ARRAY)+(string)err_code+": "+CMessage::Text(err_code)
           );
         Print(err);
         //--- Set the current period to the array
         ArrayResize(used_periods_array,1,21);
         used_periods_array[0]=TimeframeDescription((ENUM_TIMEFRAMES)Period());
         return false;
        }
     }
//--- If working with the full list of timeframes, fill in the array with strings describing all timeframes
   else
     {
      ArrayResize(used_periods_array,21,21);
      for(int i=0;i<21;i++)
         used_periods_array[i]=TimeframeDescription(TimeframeByEnumIndex(uchar(i+1)));
     }
//--- All is successful
   return true;
  }
//+------------------------------------------------------------------+
```

The function for preparing the parameter string is called in similar fashion here. The appropriate code is identical for the CreateUsedSymbolsArray() and CreateUsedTimeframesArray() functions.

The DELib.mqh file also features the function displaying time with milliseconds:

```
//+------------------------------------------------------------------+
//| Return time with milliseconds                                    |
//+------------------------------------------------------------------+
string TimeMSCtoString(const long time_msc)
  {
   return TimeToString(time_msc/1000,TIME_DATE|TIME_MINUTES|TIME_SECONDS)+"."+IntegerToString(time_msc%1000,3,'0');
  }
//+------------------------------------------------------------------+
```

The function always displays time in the following format: YYYY.MM.DD HH:MM:SS.MSC

Let's add the ability to select the time display format:

```
//+------------------------------------------------------------------+
//| Return time with milliseconds                                    |
//+------------------------------------------------------------------+
string TimeMSCtoString(const long time_msc,int flags=TIME_DATE|TIME_MINUTES|TIME_SECONDS)
  {
   return TimeToString(time_msc/1000,flags)+"."+IntegerToString(time_msc%1000,3,'0');
  }
//+------------------------------------------------------------------+
```

Now it is possible to set the time display format + milliseconds.

Add selection of applied timeframes to the block of EA file inputs:

```
sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;            // Mode of used symbols list
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list
sinput   string            InpUsedTFs           =  "M1,M5,M15,M30,H1,H4,D1,W1,MN1"; // List of used timeframes (comma - separator)
sinput   bool              InpUseSounds         =  true; // Use sounds
```

**InpModeUsedTFs** allows selecting the timeframe usage mode

- Working with the current timeframe only
- Working with the specified timeframe list
- Working with the full timeframe list

When selecting the second mode, the program uses the list of timeframes described in the string specified by the **InpUsedTFs** variable input.

From the block of EA global variables, remove the variable declaring the CTimeSeries class object. It is no longer necessary. Now the access to timeseries is performed by accessing the engine library object.

```
//--- global variables
CEngine        engine;
CTimeSeries    timeseries;
SDataButt      butt_data[TOTAL_BUTT];
```

The same block of global variables receives the new array the names of timeframes used by the program are to be written to:

```
//--- global variables
CEngine        engine;
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ushort         magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           distance_pending_request;
uint           bars_delay_pending_request;
uint           slippage;
bool           trailing_on;
bool           pressed_pending_buy;
bool           pressed_pending_buy_limit;
bool           pressed_pending_buy_stop;
bool           pressed_pending_buy_stoplimit;
bool           pressed_pending_close_buy;
bool           pressed_pending_close_buy2;
bool           pressed_pending_close_buy_by_sell;
bool           pressed_pending_sell;
bool           pressed_pending_sell_limit;
bool           pressed_pending_sell_stop;
bool           pressed_pending_sell_stoplimit;
bool           pressed_pending_close_sell;
bool           pressed_pending_close_sell2;
bool           pressed_pending_close_sell_by_buy;
bool           pressed_pending_delete_all;
bool           pressed_pending_close_all;
bool           pressed_pending_sl;
bool           pressed_pending_tp;
double         trailing_stop;
double         trailing_step;
uint           trailing_start;
uint           stoploss_to_modify;
uint           takeprofit_to_modify;
int            used_symbols_mode;
string         array_used_symbols[];
string         array_used_periods[];
bool           testing;
uchar          group1;
uchar          group2;
double         g_point;
int            g_digits;
//+------------------------------------------------------------------+
```

From the EA's OnInit() handler, remove the code of creating two timeseries remaining from the previous tests:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();

//--- Set EA global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   testing=engine.IsTester();
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      butt_data[i].name=prefix+EnumToString((ENUM_BUTTONS)i);
      butt_data[i].text=EnumToButtText((ENUM_BUTTONS)i);
     }
   lot=NormalizeLot(Symbol(),fmax(InpLots,MinimumLots(Symbol())*2.0));
   magic_number=InpMagic;
   stoploss=InpStopLoss;
   takeprofit=InpTakeProfit;
   distance_pending=InpDistance;
   distance_stoplimit=InpDistanceSL;
   slippage=InpSlippage;
   trailing_stop=InpTrailingStop*Point();
   trailing_step=InpTrailingStep*Point();
   trailing_start=InpTrailingStart;
   stoploss_to_modify=InpStopLossModify;
   takeprofit_to_modify=InpTakeProfitModify;
   distance_pending_request=(InpDistancePReq<5 ? 5 : InpDistancePReq);
   bars_delay_pending_request=(InpBarsDelayPReq<1 ? 1 : InpBarsDelayPReq);
   g_point=SymbolInfoDouble(NULL,SYMBOL_POINT);
   g_digits=(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS);
//--- Initialize random group numbers
   group1=0;
   group2=0;
   srand(GetTickCount());

//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Check and remove remaining EA graphical objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);
//--- Reset states of the buttons for working using pending requests
   for(int i=0;i<14;i++)
     {
      ButtonState(butt_data[i].name+"_PRICE",false);
      ButtonState(butt_data[i].name+"_TIME",false);
     }

//--- Check playing a standard sound by macro substitution and a custom sound by description
   engine.PlaySoundByDescription(SND_OK);
   Sleep(600);
   engine.PlaySoundByDescription(TextByLanguage("Звук упавшей монетки 2","Falling coin 2"));

//--- Set a symbol for created timeseries
   timeseries.SetSymbol(Symbol());
//#define TIMESERIES_ALL
//--- Create two timeseries
   #ifndef TIMESERIES_ALL
      timeseries.SyncData(PERIOD_CURRENT,10);
      timeseries.Create(PERIOD_CURRENT);
      timeseries.SyncData(PERIOD_M15,2);
      timeseries.Create(PERIOD_M15);
//--- Create all timeseries
   #else
      timeseries.SyncAllData();
      timeseries.CreateAll();
   #endif
//--- Check created timeseries
   CArrayObj *list=timeseries.GetList();
   Print(TextByLanguage("Данные созданных таймсерий:","Data of created timeseries:"));
   for(int i=0;i<list.Total();i++)
     {
      CSeries *series_obj=timeseries.GetSeriesByIndex((uchar)i);
      if(series_obj==NULL || series_obj.AmountUsedData()==0 || series_obj.DataTotal()==0)
         continue;
      Print(
            DFUN,i,": ",series_obj.Symbol()," ",TimeframeDescription(series_obj.Timeframe()),
            ": AmountUsedData=",series_obj.AmountUsedData(),", DataTotal=",series_obj.DataTotal(),", Bars=",series_obj.Bars()
           );
     }
   Print("");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

From the EA's OnTick() handler, remove the code of updating created timeseries (the timeseries are to be updated in the next article):

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();       // Working in the timer
      PressButtonsControl();  // Button pressing control
      EventsHandling();       // Working with events
     }
//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();    // Trailing positions
      TrailingOrders();       // Trailing of pending orders
     }
//--- Update created timeseries
   CArrayObj *list=timeseries.GetList();
   for(int i=0;i<list.Total();i++)
     {
      CSeries *series_obj=timeseries.GetSeriesByIndex((uchar)i);
      if(series_obj==NULL || series_obj.DataTotal()==0)
         continue;
      series_obj.Refresh();
      if(series_obj.IsNewBar(0))
        {
         Print(TextByLanguage("Новый бар на ","New bar on "),series_obj.Symbol()," ",TimeframeDescription(series_obj.Timeframe())," ",TimeToString(series_obj.Time(0)));
         if(series_obj.Timeframe()==Period())
            engine.PlaySoundByDescription(SND_NEWS);
        }
     }
  }
//+------------------------------------------------------------------+
```

In the OnInitDoEasy() library initialization function, add creating and displaying the list of used timeframes in the EA and displaying the time of the library initialization in the journal.

Below is the full listing with the changes highlighted in color and strings accompanied by comments for better understanding:

```
//+------------------------------------------------------------------+
//| Initializing DoEasy library                                      |
//+------------------------------------------------------------------+
void OnInitDoEasy()
  {
//--- Check if working with the full list is selected
   used_symbols_mode=InpModeUsedSymbols;
   if((ENUM_SYMBOLS_MODE)used_symbols_mode==SYMBOLS_MODE_ALL)
     {
      int total=SymbolsTotal(false);
      string ru_n="\nКоличество символов на сервере "+(string)total+".\nМаксимальное количество: "+(string)SYMBOLS_COMMON_TOTAL+" символов.";
      string en_n="\nNumber of symbols on server "+(string)total+".\nMaximum number: "+(string)SYMBOLS_COMMON_TOTAL+" symbols.";
      string caption=TextByLanguage("Внимание!","Attention!");
      string ru="Выбран режим работы с полным списком.\nВ этом режиме первичная подготовка списков коллекций символов и таймсерий может занять длительное время."+ru_n+"\nПродолжить?\n\"Нет\" - работа с текущим символом \""+Symbol()+"\"";
      string en="Full list mode selected.\nIn this mode, the initial preparation of lists of symbol collections and timeseries can take a long time."+en_n+"\nContinue?\n\"No\" - working with the current symbol \""+Symbol()+"\"";
      string message=TextByLanguage(ru,en);
      int flags=(MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2);
      int mb_res=MessageBox(message,caption,flags);
      switch(mb_res)
        {
         case IDNO :
           used_symbols_mode=SYMBOLS_MODE_CURRENT;
           break;
         default:
           break;
        }
     }
//--- Set the counter start point to measure the approximate library initialization time
   ulong begin=GetTickCount();
   Print(TextByLanguage("--- Инициализация библиотеки \"DoEasy\" ---","--- Initializing the \"DoEasy\" library ---"));
//--- Fill in the array of used symbols
   CreateUsedSymbolsArray((ENUM_SYMBOLS_MODE)used_symbols_mode,InpUsedSymbols,array_used_symbols);
//--- Set the type of the used symbol list in the symbol collection and fill in the list of symbol timeseries
   engine.SetUsedSymbols(array_used_symbols);
//--- Displaying the selected mode of working with the symbol object collection in the journal
   string num=
     (
      used_symbols_mode==SYMBOLS_MODE_CURRENT ? ": \""+Symbol()+"\"" :
      TextByLanguage(". Количество используемых символов: ",". The number of symbols used: ")+(string)engine.GetSymbolsCollectionTotal()
     );
   Print(engine.ModeSymbolsListDescription(),num);
//--- Implement displaying the list of used symbols only for MQL5 - MQL4 has no ArrayPrint() function
#ifdef __MQL5__
   if(InpModeUsedSymbols!=SYMBOLS_MODE_CURRENT)
     {
      string array_symbols[];
      CArrayObj* list_symbols=engine.GetListAllUsedSymbols();
      for(int i=0;i<list_symbols.Total();i++)
        {
         CSymbol *symbol=list_symbols.At(i);
         if(symbol==NULL)
            continue;
         ArrayResize(array_symbols,ArraySize(array_symbols)+1,1000);
         array_symbols[ArraySize(array_symbols)-1]=symbol.Name();
        }
      ArrayPrint(array_symbols);
     }
#endif
//--- Set used timeframes
   CreateUsedTimeframesArray(InpModeUsedTFs,InpUsedTFs,array_used_periods);
//--- Display the selected mode of working with the timeseries object collection
   string mode=
     (
      InpModeUsedTFs==TIMEFRAMES_MODE_CURRENT   ?
         TextByLanguage("Работа только с текущим таймфреймом: ","Work only with the current Period: ")+TimeframeDescription((ENUM_TIMEFRAMES)Period())   :
      InpModeUsedTFs==TIMEFRAMES_MODE_LIST      ?
         TextByLanguage("Работа с заданным списком таймфреймов:","Work with a predefined list of Periods:")                                              :
      TextByLanguage("Работа с полным списком таймфреймов:","Work with the full list of all Periods:")
     );
   Print(mode);
//--- Implement displaying the list of used timeframes only for MQL5 - MQL4 has no ArrayPrint() function
#ifdef __MQL5__
   if(InpModeUsedTFs!=TIMEFRAMES_MODE_CURRENT)
      ArrayPrint(array_used_periods);
#endif
//--- Create timeseries of all used symbols
   CArrayObj *list_timeseries=engine.GetListTimeSeries();
   int total=list_timeseries.Total();
   for(int i=0;i<total;i++)
     {
      CTimeSeries *timeseries=list_timeseries.At(i);
      if(list_timeseries==NULL)
         continue;
      int total_periods=ArraySize(array_used_periods);
      for(int j=0;j<total_periods;j++)
        {
         ENUM_TIMEFRAMES timeframe=TimeframeByDescription(array_used_periods[j]);
         engine.SeriesSyncData(timeseries.Symbol(),timeframe);
         engine.SeriesCreate(timeseries.Symbol(),timeframe);
        }
     }
//--- Check created timeseries - display descriptions of all created timeseries in the journal
//--- (true - only created ones, false - created and declared ones)
   engine.GetTimeSeriesCollection().PrintShort(true); // Short descriptions
   //engine.GetTimeSeriesCollection().Print(false);   // Full descriptions

//--- Create resource text files
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_01",TextByLanguage("Звук упавшей монетки 1","Falling coin 1"),sound_array_coin_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_02",TextByLanguage("Звук упавших монеток","Falling coins"),sound_array_coin_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_03",TextByLanguage("Звук монеток","Coins"),sound_array_coin_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_04",TextByLanguage("Звук упавшей монетки 2","Falling coin 2"),sound_array_coin_04);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_01",TextByLanguage("Звук щелчка по кнопке 1","Button click 1"),sound_array_click_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_02",TextByLanguage("Звук щелчка по кнопке 2","Button click 2"),sound_array_click_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_03",TextByLanguage("Звук щелчка по кнопке 3","Button click 3"),sound_array_click_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_cash_machine_01",TextByLanguage("Звук кассового аппарата","Cash machine"),sound_array_cash_machine_01);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_green",TextByLanguage("Изображение \"Зелёный светодиод\"","Image \"Green Spot lamp\""),img_array_spot_green);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_red",TextByLanguage("Изображение \"Красный светодиод\"","Image \"Red Spot lamp\""),img_array_spot_red);

//--- Pass all existing collections to the trading class
   engine.TradingOnInit();

//--- Set the default magic number for all used symbols
   engine.TradingSetMagic(engine.SetCompositeMagicNumber(magic_number));
//--- Set synchronous passing of orders for all used symbols
   engine.TradingSetAsyncMode(false);
//--- Set the number of trading attempts in case of an error
   engine.TradingSetTotalTry(InpTotalAttempts);
//--- Set correct order expiration and filling types to all trading objects
   engine.TradingSetCorrectTypeExpiration();
   engine.TradingSetCorrectTypeFilling();

//--- Set standard sounds for trading objects of all used symbols
   engine.SetSoundsStandart();
//--- Set the general flag of using sounds
   engine.SetUseSounds(InpUseSounds);
//--- Set the spread multiplier for symbol trading objects in the symbol collection
   engine.SetSpreadMultiplier(InpSpreadMultiplier);

//--- Set controlled values for symbols
   //--- Get the list of all collection symbols
   CArrayObj *list=engine.GetListAllUsedSymbols();
   if(list!=NULL && list.Total()!=0)
     {
      //--- In a loop by the list, set the necessary values for tracked symbol properties
      //--- By default, the LONG_MAX value is set to all properties, which means "Do not track this property"
      //--- It can be enabled or disabled (by setting the value less than LONG_MAX or vice versa - set the LONG_MAX value) at any time and anywhere in the program
      /*
      for(int i=0;i<list.Total();i++)
        {
         CSymbol* symbol=list.At(i);
         if(symbol==NULL)
            continue;
         //--- Set control of the symbol price increase by 100 points
         symbol.SetControlBidInc(100000*symbol.Point());
         //--- Set control of the symbol price decrease by 100 points
         symbol.SetControlBidDec(100000*symbol.Point());
         //--- Set control of the symbol spread increase by 40 points
         symbol.SetControlSpreadInc(400);
         //--- Set control of the symbol spread decrease by 40 points
         symbol.SetControlSpreadDec(400);
         //--- Set control of the current spread by the value of 40 points
         symbol.SetControlSpreadLevel(400);
        }
      */
     }
//--- Set controlled values for the current account
   CAccount* account=engine.GetAccountCurrent();
   if(account!=NULL)
     {
      //--- Set control of the profit increase to 10
      account.SetControlledValueINC(ACCOUNT_PROP_PROFIT,10.0);
      //--- Set control of the funds increase to 15
      account.SetControlledValueINC(ACCOUNT_PROP_EQUITY,15.0);
      //--- Set profit control level to 20
      account.SetControlledValueLEVEL(ACCOUNT_PROP_PROFIT,20.0);
     }
//--- Get the end of the library initialization time counting and display it in the journal
   ulong end=GetTickCount();
   Print(TextByLanguage("Время инициализации библиотеки: ","Library initialization time: "),TimeMSCtoString(end-begin,TIME_MINUTES|TIME_SECONDS));
  }
//+------------------------------------------------------------------+
```

**These are all the improvements of the test EA.**

Compile and launch it specifying the usage of the current symbol and the current timeframe in the parameters.

The following messages are displayed in the journal:

```
--- Initializing "DoEasy" library ---
Working with the current symbol only: "EURUSD"
Working with the current timeframe only: H4
EURUSD symbol timeseries:
Timeseries "EURUSD" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5330
Library initialization time: 00:00:00.141
```

Configure using the current symbol and the specified timeframe list in the settings (the list features the main timeframes).

The following messages are displayed in the journal:

```
--- Initializing "DoEasy" library ---
Working with the current symbol only: "EURUSD"
Working with the specified timeframe list:
"M1"  "M5"  "M15" "M30" "H1"  "H4"  "D1"  "W1"  "MN1"
EURUSD symbol timeseries:
Timeseries "EURUSD" M1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3286
Timeseries "EURUSD" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3566
Timeseries "EURUSD" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3109
Timeseries "EURUSD" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2894
Timeseries "EURUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5505
Timeseries "EURUSD" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5330
Timeseries "EURUSD" D1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5087
Timeseries "EURUSD" W1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2564
Timeseries "EURUSD" MN1: Requested: 1000, Actual: 590, Created: 590, On the server: 590
Library initialization time: 00:00:00.032
```

Configure using the current symbol and the full list of timeframes.

The following messages are displayed in the journal:

```
--- Initializing "DoEasy" library ---
Working with the current symbol only: "EURUSD"
Working with the full list of timeframes:
"M1"  "M2"  "M3"  "M4"  "M5"  "M6"  "M10" "M12" "M15" "M20" "M30" "H1"  "H2"  "H3"  "H4"  "H6"  "H8"  "H12" "D1"  "W1"  "MN1"
EURUSD symbol timeseries:
Timeseries "EURUSD" M1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3390
Timeseries "EURUSD" M2: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5626
Timeseries "EURUSD" M3: Requested: 1000, Actual: 1000, Created: 1000, On the server: 4713
Timeseries "EURUSD" M4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 4254
Timeseries "EURUSD" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3587
Timeseries "EURUSD" M6: Requested: 1000, Actual: 1000, Created: 1000, On the server: 4805
Timeseries "EURUSD" M10: Requested: 1000, Actual: 1000, Created: 1000, On the server: 4035
Timeseries "EURUSD" M12: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3842
Timeseries "EURUSD" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3116
Timeseries "EURUSD" M20: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3457
Timeseries "EURUSD" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2898
Timeseries "EURUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5507
Timeseries "EURUSD" H2: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6303
Timeseries "EURUSD" H3: Requested: 1000, Actual: 1000, Created: 1000, On the server: 6263
Timeseries "EURUSD" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5331
Timeseries "EURUSD" H6: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5208
Timeseries "EURUSD" H8: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5463
Timeseries "EURUSD" H12: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5205
Timeseries "EURUSD" D1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5087
Timeseries "EURUSD" W1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2564
Timeseries "EURUSD" MN1: Requested: 1000, Actual: 590, Created: 590, On the server: 590
Library initialization time: 00:00:00.094
```

Configure using the specified symbol list and set three symbols EURUSD,AUDUSD,EURAUD in the settings. Also, configure using the specified timeframe list (main timeframes are specified in the list).

The following messages are displayed in the journal:

```
--- Initializing "DoEasy" library ---
Working with predefined symbol list. The number of used symbols: 3
"AUDUSD" "EURUSD" "EURAUD"
Working with the specified timeframe list:
"M1"  "M5"  "M15" "M30" "H1"  "H4"  "D1"  "W1"  "MN1"
AUDUSD symbol timeseries:
Timeseries "AUDUSD" M1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3394
Timeseries "AUDUSD" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 4024
Timeseries "AUDUSD" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3262
Timeseries "AUDUSD" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3071
Timeseries "AUDUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5104
Timeseries "AUDUSD" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5026
Timeseries "AUDUSD" D1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5289
Timeseries "AUDUSD" W1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 1401
Timeseries "AUDUSD" MN1: Requested: 1000, Actual: 323, Created: 323, On the server: 323
EURAUD symbol timeseries:
Timeseries "EURAUD" M1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3393
Timeseries "EURAUD" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 4025
Timeseries "EURAUD" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3262
Timeseries "EURAUD" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3071
Timeseries "EURAUD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5104
Timeseries "EURAUD" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5026
Timeseries "EURAUD" D1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 4071
Timeseries "EURAUD" W1: Requested: 1000, Actual: 820, Created: 820, On the server: 820
Timeseries "EURAUD" MN1: Requested: 1000, Actual: 189, Created: 189, On the server: 189
EURUSD symbol timeseries:
Timeseries "EURUSD" M1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3394
Timeseries "EURUSD" M5: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3588
Timeseries "EURUSD" M15: Requested: 1000, Actual: 1000, Created: 1000, On the server: 3116
Timeseries "EURUSD" M30: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2898
Timeseries "EURUSD" H1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5507
Timeseries "EURUSD" H4: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5331
Timeseries "EURUSD" D1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 5087
Timeseries "EURUSD" W1: Requested: 1000, Actual: 1000, Created: 1000, On the server: 2564
Timeseries "EURUSD" MN1: Requested: 1000, Actual: 590, Created: 590, On the server: 590
Library initialization time: 00:00:00.266
```

As we can see, the required timeseries are created depending on the symbols and timeframes specified in the EA settings. Timeseries creation time depends on the EA launch (cold/hot), as well as on whether the selected symbols and their timeframes were used before.

### What's next?

In the next article, we will create the functionality for real time updates of created timeseries and for sending messages about "New bar" events to the program for all used timeseries, as well as for receiving required data from the existing timeseries.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7663#node00)

**Previous articles within the series:**

[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7663](https://www.mql5.com/ru/articles/7663)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7663.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7663/mql5.zip "Download MQL5.zip")(3693.82 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7663/mql4.zip "Download MQL4.zip")(3693.83 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/345124)**
(2)


![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
7 Apr 2022 at 23:59

Hi there... Can you please clarify where and how the spread multiplier is used in the library? I have looked around and it seems to be used primarily for [error correction](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development") in CTrading::RequestErrorsCorrecting() and functions called from there. What is the actual purpose of the multiplier in this process?

```
engine.SetSpreadMultiplier(InpSpreadMultiplier);
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
8 Apr 2022 at 02:08

**Dima Diall [#](https://www.mql5.com/en/forum/345124#comment_28806093) :**

Hi there... Can you please clarify where and how the spread multiplier is used in the library? I have looked around and it seems to be used primarily for error correction in CTrading::RequestErrorsCorrecting() and functions called from there. What is the actual purpose of the multiplier in this process?

If the StopLevel is set to zero on the server, then this does not mean that it does not exist. This indicates that this level is floating. And usually it is equal to two spread sizes. The multiplier is just needed in order to indicate for such a situation by how much the spread value needs to be multiplied in order to get the correct StopLevel level.

![Applying OLAP in trading (part 4): Quantitative and visual analysis of tester reports](https://c.mql5.com/2/38/OLAP_in_trading.png)[Applying OLAP in trading (part 4): Quantitative and visual analysis of tester reports](https://www.mql5.com/en/articles/7656)

The article offers basic tools for the OLAP analysis of tester reports relating to single passes and optimization results. The tool can work with standard format files (tst and opt), and it also provides a graphical interface. MQL source codes are attached below.

![Continuous Walk-Forward Optimization (Part 5): Auto Optimizer project overview and creation of a GUI](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__1.png)[Continuous Walk-Forward Optimization (Part 5): Auto Optimizer project overview and creation of a GUI](https://www.mql5.com/en/articles/7583)

This article provides further description of the walk-forward optimization in the MetaTrader 5 terminal. In previous articles, we considered methods for generating and filtering the optimization report and started analyzing the internal structure of the application responsible for the optimization process. The Auto Optimizer is implemented as a C# application and it has its own graphical interface. The fifth article is devoted to the creation of this graphical interface.

![Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://c.mql5.com/2/38/Article_Logo.png)[Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://www.mql5.com/en/articles/7600)

In the previous article, we developed the visual part of the application, as well as the basic interaction of GUI elements. This time we are going to add internal logic and the algorithm of trading signal data preparation, as well us the ability to set up signals, to search them and to visualize them in the monitor.

![Forecasting Time Series (Part 2): Least-Square Support-Vector Machine (LS-SVM)](https://c.mql5.com/2/38/mql5-avatar-lssvm.png)[Forecasting Time Series (Part 2): Least-Square Support-Vector Machine (LS-SVM)](https://www.mql5.com/en/articles/7603)

This article deals with the theory and practical application of the algorithm for forecasting time series, based on support-vector method. It also proposes its implementation in MQL and provides test indicators and Expert Advisors. This technology has not been implemented in MQL yet. But first, we have to get to know math for it.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=eohfmeglxinzrjinowzyvfpqhrjejdns&ssn=1769251913089999883&ssn_dr=0&ssn_sr=0&fv_date=1769251913&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7663&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2037)%3A%20Timeseries%20collection%20-%20database%20of%20timeseries%20by%20symbols%20and%20periods%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925191365354664&fz_uniq=5083167183271368265&sv=2552)

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