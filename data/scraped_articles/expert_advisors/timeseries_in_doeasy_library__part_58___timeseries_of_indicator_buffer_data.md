---
title: Timeseries in DoEasy library (part 58): Timeseries of indicator buffer data
url: https://www.mql5.com/en/articles/8787
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:28:15.058329
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=yplqqldwhliqwieyootirvfpjnvilfqa&ssn=1769192893329572980&ssn_dr=0&ssn_sr=0&fv_date=1769192893&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8787&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2058)%3A%20Timeseries%20of%20indicator%20buffer%20data%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919289336454279&fz_uniq=5071855777011937213&sv=2552)

MetaTrader 5 / Examples


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8787#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8787#node02)
- [Timeseries class of indicator buffer data](https://www.mql5.com/en/articles/8787#node03)
- [Testing](https://www.mql5.com/en/articles/8787#node04)
- [What's next?](https://www.mql5.com/en/articles/8787#node05)


### Concept

In conclusion of the topic of working with timeseries organise storage, search and sort of data stored in indicator buffers which will allow to further perform the analysis based on values of the indicators to be created on the library basis in programs.

The concept of indicator data timeseries construction is similar to the concept of timeseries collection construction: for each indicator create a list to store all data of all its buffers. Quantity of data in the indicator buffer list will correspond to the quantity of data of corresponding timeseries on which indicators were calculated. The general concept of all collection classes of the library allows to easily find necessary data in the corresponding collection. Respectively, the same will be possible in the class created today.

In furtherance, I will create classes to perform analysis on all data stored in the library. This will provide a very flexible tool for performing statistical studies at full-scope comparison of any data of library collections.

### Improving library classes

Collection class of indicator buffer data will automatically update data of all created indicators on the current bar; and when a new bar appears it will create new data of indicator buffers and place them to collection.

In **NewBarObj.mqh** file we already have “New bar” class which was created for timeseries collection.

It is stored in timeseries class folder in library directory \\MQL5\\Include\\DoEasyPart57\\Objects\\Series\\.

Now, it is required also to track the new bar in collection class of indicator buffer data.

Therefore, move it to the folder of service functions and classes of the library \\MQL5\\Include\\DoEasy\ **Services\**.

Delete the file in its previous location.

Since “Timeseries” class uses “New bar” class the old path to this class must be changed in “Timeseries” class file \\MQL5\\Include\\DoEasy\\Objects\\Series\ **SeriesDE.mqh** in the included file block:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\Select.mqh"
#include "NewBarObj.mqh"
#include "Bar.mqh"
//+------------------------------------------------------------------+
```

to a new path:

```
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\Select.mqh"
#include "..\..\Services\NewBarObj.mqh"
#include "Bar.mqh"
//+------------------------------------------------------------------+
```

Add new message indices to file \\MQL5\\Include\\DoEasy\ **Data.mqh**:

```
   MSG_LIB_SYS_ERROR_FAILED_GET_PRICE_ASK,            // Failed to get Ask price. Error
   MSG_LIB_SYS_ERROR_FAILED_GET_PRICE_BID,            // Failed to get Bid price. Error
   MSG_LIB_SYS_ERROR_FAILED_GET_TIME,                 // Failed to get time. Error
   MSG_LIB_SYS_ERROR_FAILED_OPEN_BUY,                 // Failed to open Buy position. Error
```

...

```
//--- CDataInd
   MSG_LIB_TEXT_IND_DATA_IND_BUFFER_NUM,              // Indicator buffer number
   MSG_LIB_TEXT_IND_DATA_BUFFER_VALUE,                // Indicator buffer value

//--- CSeriesDataInd
   MSG_LIB_TEXT_METHOD_NOT_FOR_INDICATORS,            // The method is not intended to handle indicator programs
   MSG_LIB_TEXT_IND_DATA_FAILED_GET_SERIES_DATA,      // Failed to get indicator data timeseries
   MSG_LIB_TEXT_IND_DATA_FAILED_GET_CURRENT_DATA,     // Failed to get current data of indicator buffer
   MSG_LIB_SYS_FAILED_CREATE_IND_DATA_OBJ,            // Failed to create indicator data object
   MSG_LIB_TEXT_IND_DATA_FAILED_ADD_TO_LIST,          // Failed to add indicator data object to list

  };
//+------------------------------------------------------------------+
```

and message texts corresponding to newly added indices:

```
   {"Could not get Ask price. Error "},
   {"Could not get Bid price. Error "},
   {"Could not get time. Error "},
   {"Failed to open a Buy position. Error "},
```

...

```
   {"Indicator buffer number"},
   {"Indicator buffer value"},

   {"The method is not intended for working with indicator programs"},
   {"Failed to get indicator data timeseries"},
   {"Failed to get the current data of the indicator buffer"},
   {"Failed to create indicator data object"},
   {"Failed to add indicator data object to the list"},

  };
//+---------------------------------------------------------------------+
```

Since today I will create a new collection, let’s set collection ID for it. Apart from that, create constants of parameters of new collection timer because indicator buffer data will be updated automatically in the timer.

Add new data to file \\MQL5\\Include\\DoEasy\ **Defines.mqh**:

```
//--- Parameters of the timeseries collection timer
#define COLLECTION_TS_PAUSE            (64)                       // Timeseries collection timer pause in milliseconds
#define COLLECTION_TS_COUNTER_STEP     (16)                       // Timeseries timer counter increment
#define COLLECTION_TS_COUNTER_ID       (6)                        // Timeseries timer counter ID
//--- Parameters of the timer of indicator data timeseries collection
#define COLLECTION_IND_TS_PAUSE        (64)                       // Pause of the timer of indicator data timeseries collection in milliseconds
#define COLLECTION_IND_TS_COUNTER_STEP (16)                       // Increment of indicator data timeseries timer counter
#define COLLECTION_IND_TS_COUNTER_ID   (7)                        // ID of indicator data timeseries timer counter
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x777A)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777B)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777C)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777D)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777E)                   // Symbol collection list ID
#define COLLECTION_SERIES_ID           (0x777F)                   // Timeseries collection list ID
#define COLLECTION_BUFFERS_ID          (0x7780)                   // Indicator buffer collection list ID
#define COLLECTION_INDICATORS_ID       (0x7781)                   // Indicator collection list ID
#define COLLECTION_INDICATORS_DATA_ID  (0x7782)                   // Indicator data collection list ID
```

To search indicator buffer data in their collection, data by indicator handle will have to be additionally searched (because data will be affiliated with this indicator and for searching data it would be strange not to use exact ID of their affiliation with the indicator).

To integer properties of indicator data add a new property and increase the quantity of these data from 5 to **6**:

```
//+------------------------------------------------------------------+
//| Integer properties of indicator data                             |
//+------------------------------------------------------------------+
enum ENUM_IND_DATA_PROP_INTEGER
  {
   IND_DATA_PROP_TIME = 0,                                  // Start time of indicator data bar period
   IND_DATA_PROP_PERIOD,                                    // Indicator data period (timeframe)
   IND_DATA_PROP_INDICATOR_TYPE,                            // Indicator type
   IND_DATA_PROP_IND_BUFFER_NUM,                            // Indicator data buffer number
   IND_DATA_PROP_IND_ID,                                    // Indicator ID
   IND_DATA_PROP_IND_HANDLE,                                // Indicator handle
  };
#define IND_DATA_PROP_INTEGER_TOTAL (6)                     // Total number of indicator data integer properties
#define IND_DATA_PROP_INTEGER_SKIP  (0)                     // Number of indicator data properties not used in sorting
//+------------------------------------------------------------------+
```

Respectively, since we added a new integer property add it to the list of possible sort criteria to be able to search and sort by this property:

```
//+------------------------------------------------------------------+
//| Possible criteria for indicator data sorting                     |
//+------------------------------------------------------------------+
#define FIRST_IND_DATA_DBL_PROP          (IND_DATA_PROP_INTEGER_TOTAL-IND_DATA_PROP_INTEGER_SKIP)
#define FIRST_IND_DATA_STR_PROP          (IND_DATA_PROP_INTEGER_TOTAL-IND_DATA_PROP_INTEGER_SKIP+IND_DATA_PROP_DOUBLE_TOTAL-IND_DATA_PROP_DOUBLE_SKIP)
enum ENUM_SORT_IND_DATA_MODE
  {
//--- Sort by integer properties
   SORT_BY_IND_DATA_TIME = 0,                               // Sort by bar period start time of indicator data
   SORT_BY_IND_DATA_PERIOD,                                 // Sort by indicator data period (timeframe)
   SORT_BY_IND_DATA_INDICATOR_TYPE,                         // Sort by indicator type
   SORT_BY_IND_DATA_IND_BUFFER_NUM,                         // Sort by indicator data buffer number
   SORT_BY_IND_DATA_IND_ID,                                 // Sort by indicator ID
   SORT_BY_IND_DATA_IND_HANDLE,                             // Sort by indicator handle
//--- Sort by real properties
   SORT_BY_IND_DATA_BUFFER_VALUE = FIRST_IND_DATA_DBL_PROP, // Sort by indicator data value
//--- Sort by string properties
   SORT_BY_IND_DATA_SYMBOL = FIRST_IND_DATA_STR_PROP,       // Sort by indicator data symbol
   SORT_BY_IND_DATA_IND_NAME,                               // Sort by indicator name
   SORT_BY_IND_DATA_IND_SHORTNAME,                          // Sort by indicator short name
  };
//+------------------------------------------------------------------+
```

Indicator buffer data stored in the collection are presented by [indicator buffer data class created in the previous article](https://www.mql5.com/en/articles/8705#node03).

Now, add to it (in file \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **DataInd.mqh**) an installation method of indicator handle, data of which buffer are stored in the object of this class. Also, to class constructor add the passing of indicator handle parameters and values of its buffer. This will allow to set values of these parameters during creation of the object:

```
//--- Set (1) symbol, timeframe and time for the object, (2) indicator type, (3) number of buffers, (4) number of data buffer,
//--- (5) ID, (6) handle, (7) data value, (8) name, (9) indicator short name
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   void              SetIndicatorType(const ENUM_INDICATOR type)              { this.SetProperty(IND_DATA_PROP_INDICATOR_TYPE,type);               }
   void              SetBufferNum(const int num)                              { this.SetProperty(IND_DATA_PROP_IND_BUFFER_NUM,num);                }
   void              SetIndicatorID(const int id)                             { this.SetProperty(IND_DATA_PROP_IND_ID,id);                         }
   void              SetIndicatorHandle(const int handle)                     { this.SetProperty(IND_DATA_PROP_IND_HANDLE,handle);                 }
   void              SetBufferValue(const double value)                       { this.SetProperty(IND_DATA_PROP_BUFFER_VALUE,value);                }
   void              SetIndicatorName(const string name)                      { this.SetProperty(IND_DATA_PROP_IND_NAME,name);                     }
   void              SetIndicatorShortname(const string shortname)            { this.SetProperty(IND_DATA_PROP_IND_SHORTNAME,shortname);           }

//--- Compare CDataInd objects with each other by all possible properties (for sorting the lists by a specified property of data object )
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CDataInd objects with each other by all properties (to search equal objects)
   bool              IsEqual(CDataInd* compared_data) const;
//--- Constructors
                     CDataInd(){;}
                     CDataInd(const ENUM_INDICATOR ind_type,
                              const int ind_handle,
                              const int ind_id,
                              const int buffer_num,
                              const string symbol,
                              const ENUM_TIMEFRAMES timeframe,
                              const datetime time,
                              const double value);
```

To the block of methods for simplified access to object properties add a method which returns indicator handle, and rename the method which returns the value of indicator buffer data (earlier the method was called PriceValue()):

```
//+------------------------------------------------------------------+
//| Methods of simplified access to object properties                |
//+------------------------------------------------------------------+
//--- Return (1) bar period start time, (2) timeframe, (3) indicator type,
//--- (4) number of buffers, (5) buffer number, (6) ID, (7) indicator handle
   datetime          Time(void)                                         const { return (datetime)this.GetProperty(IND_DATA_PROP_TIME);                   }
   ENUM_TIMEFRAMES   Timeframe(void)                                    const { return (ENUM_TIMEFRAMES)this.GetProperty(IND_DATA_PROP_PERIOD);          }
   ENUM_INDICATOR    IndicatorType(void)                                const { return (ENUM_INDICATOR)this.GetProperty(IND_DATA_PROP_INDICATOR_TYPE);   }
   int               BufferNum(void)                                    const { return (ENUM_INDICATOR)this.GetProperty(IND_DATA_PROP_IND_BUFFER_NUM);   }
   int               IndicatorID(void)                                  const { return (ENUM_INDICATOR)this.GetProperty(IND_DATA_PROP_IND_ID);           }
   int               IndicatorHandle(void)                              const { return (ENUM_INDICATOR)this.GetProperty(IND_DATA_PROP_IND_HANDLE);       }

//--- Return the value of indicator buffer data
   double            BufferValue(void)                                  const { return this.GetProperty(IND_DATA_PROP_BUFFER_VALUE);                     }
```

**In the body of class constructor set the values of new properties to be passed to it when creating a new object:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CDataInd::CDataInd(const ENUM_INDICATOR ind_type,
                   const int ind_handle,
                   const int ind_id,
                   const int buffer_num,
                   const string symbol,
                   const ENUM_TIMEFRAMES timeframe,
                   const datetime time,
                   const double value)
  {
   this.m_type=COLLECTION_INDICATORS_DATA_ID;
   this.m_digits=(int)::SymbolInfoInteger(symbol,SYMBOL_DIGITS)+1;
   this.m_period_description=TimeframeDescription(timeframe);
   this.SetSymbolPeriod(symbol,timeframe,time);
   this.SetIndicatorType(ind_type);
   this.SetIndicatorHandle(ind_handle);
   this.SetBufferNum(buffer_num);
   this.SetIndicatorID(ind_id);
   this.SetBufferValue(value);
  }
//+------------------------------------------------------------------+
```

Add code block for display of indicator handle description in the method returning the description of integer properties:

```
//+------------------------------------------------------------------+
//| Return description of object's integer property                  |
//+------------------------------------------------------------------+
string CDataInd::GetPropertyDescription(ENUM_IND_DATA_PROP_INTEGER property)
  {
   return
     (
      property==IND_DATA_PROP_TIME           ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==IND_DATA_PROP_PERIOD         ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_TIMEFRAME)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.m_period_description
         )  :
      property==IND_DATA_PROP_INDICATOR_TYPE ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_TYPE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.IndicatorTypeDescription()
         )  :
      property==IND_DATA_PROP_IND_BUFFER_NUM ?  CMessage::Text(MSG_LIB_TEXT_IND_DATA_IND_BUFFER_NUM)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==IND_DATA_PROP_IND_ID         ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_ID)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==IND_DATA_PROP_IND_HANDLE     ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_HANDLE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

### Timeseries class of indicator buffer data

All collection classes in the library are created based on the [class](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) [of dynamic array of pointers to instances of CObject class and its descendants](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) of the Standard library. The collection class of indicator buffer data will not be an exclusion.

This class will allow to store [objects of buffer data](https://www.mql5.com/en/articles/8705#node03) of one indicator in the list of CArrayObj objects, to get data of any indicator buffer which correspond to the timeseries bar open time. Naturally, the list will be able to automatically update, search and sort by properties of the objects stored in it.

In folder \\MQL5\\Include\\DoEasy\\Objects\ **Indicators\** create a new  class **CSeriesDataInd** in file **SeriesDataInd.mqh**.

The class object must be derived from the base object of all objects of CBaseObj library.

Analyze a class body with all its variables and methods:

```
//+------------------------------------------------------------------+
//|                                                SeriesDataInd.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\Select.mqh"
#include "..\..\Services\NewBarObj.mqh"
#include "DataInd.mqh"
//+------------------------------------------------------------------+
//| “Indicator data list” class                                      |
//+------------------------------------------------------------------+
class CSeriesDataInd : public CBaseObj
  {
private:
   ENUM_TIMEFRAMES   m_timeframe;                                       // Timeframe
   ENUM_INDICATOR    m_ind_type;                                        // Indicator type
   string            m_symbol;                                          // Symbol
   string            m_period_description;                              // Timeframe string description
   int               m_ind_handle;                                      // Indicator handle
   int               m_ind_id;                                          // Indicator ID
   int               m_buffers_total;                                   // Number of indicator buffers
   uint              m_amount;                                          // Amount of applied timeseries data
   uint              m_required;                                        // Required amount of applied timeseries data
   uint              m_bars;                                            // Number of bars in history by symbol and timeframe
   bool              m_sync;                                            // Synchronized data flag
   CArrayObj         m_list_data;                                       // Indicator data list
   CNewBarObj        m_new_bar_obj;                                     // "New bar" object
//--- Create a new indicator data object, return pointer to it
   CDataInd         *CreateNewDataInd(const int buffer_num,const datetime time,const double value);

public:
//--- Return (1) oneself, (2) the full list of indicator data
   CSeriesDataInd   *GetObject(void)                                    { return &this;               }
   CArrayObj        *GetList(void)                                      { return &this.m_list_data;   }

//--- Return the list of indicator data by selected (1) double, (2) integer and (3) string property fitting a compared condition
   CArrayObj        *GetList(ENUM_IND_DATA_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByIndicatorDataProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_IND_DATA_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByIndicatorDataProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_IND_DATA_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByIndicatorDataProperty(this.GetList(),property,value,mode); }

//--- Return indicator data object by (1) time, (2) bar number and buffer number
   CDataInd         *GetIndDataByTime(const int buffer_num,const datetime time);
   CDataInd         *GetIndDataByBar(const int buffer_num,const uint shift);

//--- Set (1) symbol, (2) timeframe, (3) symbol and timeframe, (4) amount of applied timeseries data,
//--- (5) handle, (6) ID, (7) number of indicator buffers
   void              SetSymbol(const string symbol);
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe);
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe);
   bool              SetRequiredUsedData(const uint required);
   void              SetIndHandle(const int handle)                              { this.m_ind_handle=handle;   }
   void              SetIndID(const int id)                                      { this.m_ind_id=id;           }
   void              SetIndBuffersTotal(const int total)                         { this.m_buffers_total=total; }

//--- Return (1) symbol, (2) timeframe, number of (3) used and (4) requested timeseries data,
//--- (5) number of bars in timeseries, (6) handle, (7) ID, (8) number of indicator buffers
   string            Symbol(void)                                          const { return this.m_symbol;                            }
   ENUM_TIMEFRAMES   Timeframe(void)                                       const { return this.m_timeframe;                         }
   ulong             AvailableUsedData(void)                               const { return this.m_amount;                            }
   ulong             RequiredUsedData(void)                                const { return this.m_required;                          }
   ulong             Bars(void)                                            const { return this.m_bars;                              }
   int               IndHandle(void)                                       const { return this.m_ind_handle;                        }
   int               IndID(void)                                           const { return this.m_ind_id;                            }
   int               IndBuffersTotal(void)                                 const { return this.m_buffers_total;                     }
   bool              IsNewBar(const datetime time)                               { return this.m_new_bar_obj.IsNewBar(time);        }
   bool              IsNewBarManual(const datetime time)                         { return this.m_new_bar_obj.IsNewBarManual(time);  }

//--- Return real list size
   int               DataTotal(void)                                       const { return this.m_list_data.Total();                 }

//--- Save the new bar time during the manual time management
   void              SaveNewBarTime(const datetime time)                         { this.m_new_bar_obj.SaveNewBarTime(time);         }

//--- (1) Create, (2) update the indicator data list
   int               Create(const uint required=0);
   void              Refresh(void);

//--- Return data of specified indicator buffer by (1) open time, (2) bar index
   double            BufferValue(const int buffer_num,const datetime time);
   double            BufferValue(const int buffer_num,const uint shift);

//--- Constructors
                     CSeriesDataInd(void){;}
                     CSeriesDataInd(const int handle,
                                    const ENUM_INDICATOR ind_type,
                                    const int ind_id,
                                    const int buffers_total,
                                    const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0);
  };
//+------------------------------------------------------------------+
```

In the object, already known methods inherent to all objects of the library are seen, as well as a set of class member variables for storing object parameters.

In the class public section, declare the methods for simplified access to object properties and for their installation from the outside.

Let's have a look at the implementation of the class methods.

To the **parametric class constructor** all values of its properties necessary to create an object are passed:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CSeriesDataInd::CSeriesDataInd(const int handle,
                               const ENUM_INDICATOR ind_type,
                               const int ind_id,
                               const int buffers_total,
                               const string symbol,const ENUM_TIMEFRAMES timeframe,const uint required=0) :
                               m_bars(0), m_amount(0),m_required(0),m_sync(false)
  {
//--- To the object set indicator data collection list ID
   this.m_type=COLLECTION_INDICATORS_DATA_ID;
//--- Clear the list and set for it the flag of the list sorted by time
   this.m_list_data.Clear();
   this.m_list_data.Sort(SORT_BY_IND_DATA_TIME);
//--- Set values for all object variables
   this.SetSymbolPeriod(symbol,timeframe);
   this.m_sync=this.SetRequiredUsedData(required);
   this.m_period_description=TimeframeDescription(this.m_timeframe);
   this.m_ind_handle=handle;
   this.m_ind_type=ind_type;
   this.m_ind_id=ind_id;
   this.m_buffers_total=buffers_total;
  }
//+------------------------------------------------------------------+
```

For the object set ID of timeseries collection of indicator buffer data, clear the list where [objects of CDataInd class](https://www.mql5.com/en/articles/8705#node03) will be stored, and the flag of sorted list is set for the list. The most optimal sort mode for these objects is sorting by time: to ensure correspondence of their location in the list with data location in indicator’s physical buffer.

**Methods for setting a symbol and timeframe** require no comments:

```
//+------------------------------------------------------------------+
//| Set a symbol                                                     |
//+------------------------------------------------------------------+
void CSeriesDataInd::SetSymbol(const string symbol)
  {
   if(this.m_symbol==symbol)
      return;
   this.m_symbol=(symbol==NULL || symbol=="" ? ::Symbol() : symbol);
  }
//+------------------------------------------------------------------+
//| Set a timeframe                                                  |
//+------------------------------------------------------------------+
void CSeriesDataInd::SetTimeframe(const ENUM_TIMEFRAMES timeframe)
  {
   if(this.m_timeframe==timeframe)
      return;
   this.m_timeframe=(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe);
  }
//+------------------------------------------------------------------+
//| Set a symbol and timeframe                                       |
//+------------------------------------------------------------------+
void CSeriesDataInd::SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   this.SetSymbol(symbol);
   this.SetTimeframe(timeframe);
  }
//+------------------------------------------------------------------+
```

**Method for setting the necessary amount of indicator buffer data:**

```
//+------------------------------------------------------------------+
//| Set the amount of necessary data                                 |
//+------------------------------------------------------------------+
bool CSeriesDataInd::SetRequiredUsedData(const uint required)
  {
//--- If this is indicator program - report and leave
   if(this.m_program==PROGRAM_INDICATOR)
     {
      ::Print(CMessage::Text(MSG_LIB_TEXT_METHOD_NOT_FOR_INDICATORS));
      return false;
     }
//--- Set the necessary amount of data - if less than 1 is passed use by default (1000), or else - the passed value
   this.m_required=(required<1 ? SERIES_DEFAULT_BARS_COUNT : required);
//--- Launch download of historical data (relevant for the “cold” start)
   datetime array[1];
   ::CopyTime(this.m_symbol,this.m_timeframe,0,1,array);
//--- Set the number of available timeseries bars
   this.m_bars=(uint)::SeriesInfoInteger(this.m_symbol,this.m_timeframe,SERIES_BARS_COUNT);
//--- If succeeded to set the number of available history bars, set the amount of data in the list:
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

To work within indicators other classes are already available, therefore the type of started program is checked at once. And if this is an indicator, a message thereof is displayed and false is returned. Work of a similar method was analyzed when timeseries classes were created in [article 35](https://www.mql5.com/en/articles/7594#node04).

**A method creating a new indicator data object:**

```
//+------------------------------------------------------------------+
//| Create a new indicator data object, return the pointer           |
//+------------------------------------------------------------------+
CDataInd* CSeriesDataInd::CreateNewDataInd(const int buffer_num,const datetime time,const double value)
  {
   CDataInd* data_ind=new CDataInd(this.m_ind_type,this.m_ind_handle,this.m_ind_id,buffer_num,this.m_symbol,this.m_timeframe,time,value);
   return data_ind;
  }
//+------------------------------------------------------------------+
```

Create a new object of CDataInd class and return the pointer to the newly created object or NULL in case if an object was not created.

**Method for creation of indicator data timeseries list:**

```
//+------------------------------------------------------------------+
//| Create indicator data timeseries list                            |
//+------------------------------------------------------------------+
int CSeriesDataInd::Create(const uint required=0)
  {
//--- If this is indicator program - report and leave
   if(this.m_program==PROGRAM_INDICATOR)
     {
      ::Print(CMessage::Text(MSG_LIB_TEXT_METHOD_NOT_FOR_INDICATORS));
      return false;
     }
//--- If the required history depth is not set for the list yet,
//--- display the appropriate message and return zero,
   if(this.m_amount==0)
     {
      ::Print(DFUN,this.m_symbol," ",TimeframeDescription(this.m_timeframe),": ",CMessage::Text(MSG_LIB_TEXT_BAR_TEXT_FIRS_SET_AMOUNT_DATA));
      return 0;
     }
//--- otherwise, if the passed 'required' value exceeds zero and is not equal to the one already set,
//--- while the passed ‘required’ value is less than the available bar number,
//--- set the new value of the required history depth for the list
   else if(required>0 && this.m_amount!=required && required<this.m_bars)
     {
      //--- If failed to set a new value, return zero
      if(!this.SetRequiredUsedData(required))
         return 0;
     }
//--- For the data[] array we are to receive historical data to,
//--- set the flag of direction like in the timeseries,
//--- clear the list of indicator data objects and set the flag of sorting by timeseries bar time
   double data[];
   ::ArraySetAsSeries(data,true);
   this.m_list_data.Clear();
   this.m_list_data.Sort(SORT_BY_IND_DATA_TIME);
   ::ResetLastError();

   int err=ERR_SUCCESS;
//--- In a loop by the number of indicator data buffers
   for(int i=0;i<this.m_buffers_total;i++)
     {
      //--- Get historical data of i buffer of indicator to data[] array  starting from the current bar in the amount of m_amount,
      //--- if failed to get data, display the appropriate message and return zero
      int copied=::CopyBuffer(this.m_ind_handle,i,0,this.m_amount,data);
      if(copied<1)
        {
         err=::GetLastError();
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_IND_DATA_FAILED_GET_SERIES_DATA)," ",this.m_symbol," ",TimeframeDescription(this.m_timeframe),". ",
                      CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err));
         return 0;
        }
      //--- In the loop by amount of required data
      for(int j=0;j<(int)this.m_amount;j++)
        {
         //--- Get time of j bar
         ::ResetLastError();
         datetime time=::iTime(this.m_symbol,this.m_timeframe,j);
         //--- If failed to get time
        //--- display the appropriate message with the error description in the journal and move on to the next loop iteration by data (j)
         if(time==0)
           {
            err=::GetLastError();
            ::Print( DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TIME),CMessage::Text(err),CMessage::Retcode(err));
            continue;
           }
         //--- If failed to create a new indicator data object from i buffer
        //--- display the appropriate message with the error description in the journal and move on to the next loop iteration by data (j)
         CDataInd* data_ind=CreateNewDataInd(i,time,data[j]);
         if(data_ind==NULL)
           {
            err=::GetLastError();
            ::Print
              (
               DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_IND_DATA_OBJ)," ",::TimeToString(time),". ",
               CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err)
              );
            continue;
           }
         //--- If failed to add a new indicator data object to list
         //--- display the appropriate message with the error description in the journal, remove data object
         //--- and move on to the next loop iteration by data
         if(!this.m_list_data.Add(data_ind))
           {
            err=::GetLastError();
            ::Print
              (
               DFUN,CMessage::Text(MSG_LIB_TEXT_IND_DATA_FAILED_ADD_TO_LIST)," ",::TimeToString(time),". ",
               CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err)
              );
            delete data_ind;
            continue;
           }
        }
     }
//--- Return the size of the created list of indicator data objects
   return this.m_list_data.Total();
  }
//+------------------------------------------------------------------+
```

All method logic is described in detail in its listing. The method copies data from all indicator buffers to the array. On these data it creates new indicator data objects CDataInd and puts them to collection list.

**Method for updating indicator data collection list:**

```
//+------------------------------------------------------------------+
//| Update indicator data timeseries list and data                   |
//+------------------------------------------------------------------+
void CSeriesDataInd::Refresh(void)
  {
//--- If this is indicator program - report and leave
   if(this.m_program==PROGRAM_INDICATOR)
     {
      ::Print(CMessage::Text(MSG_LIB_TEXT_METHOD_NOT_FOR_INDICATORS));
      return;
     }
//--- If indicator data timeseries is not used, exit
   if(!this.m_available)
      return;
   double data[1];

//--- Get the current bar time
   int err=ERR_SUCCESS;
   ::ResetLastError();
   datetime time=::iTime(this.m_symbol,this.m_timeframe,0);
   //--- If failed to get time
   //--- display the appropriate message with the error description in the journal and exit
   if(time==0)
     {
      err=::GetLastError();
      ::Print( DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TIME),CMessage::Text(err),CMessage::Retcode(err));
      return;
     }

//--- Set the flag of sorting the data list by time
   this.m_list_data.Sort(SORT_BY_IND_DATA_TIME);
//--- If a new bar is present on a symbol and period
   if(this.IsNewBarManual(time))
     {
      //--- create new indicator data objects by the number of buffers and add them to the list end
      for(int i=0;i<this.m_buffers_total;i++)
        {
         //--- Get data of the current bar of indicator’s i buffer to data[] array,
         //--- if failed to get data, display the appropriate message and exit
         int copied=::CopyBuffer(this.m_ind_handle,i,0,1,data);
         if(copied<1)
           {
            err=::GetLastError();
            ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_IND_DATA_FAILED_GET_CURRENT_DATA)," ",this.m_symbol," ",TimeframeDescription(this.m_timeframe),". ",
                         CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err));
            return;
           }
         //--- Create a new data object of indicator’s i buffer
         CDataInd* data_ind=CreateNewDataInd(i,time,data[0]);
         if(data_ind==NULL)
            return;
         //--- If the created object was not added to the list - remove this object and exit from the method
         if(!this.m_list_data.InsertSort(data_ind))
           {
            delete data_ind;
            return;
           }
        }
      //--- if the size of indicator data timeseries has ecxeeded the requested number of bars, remove the earliest data object
      if(this.m_list_data.Total()>(int)this.m_required)
         this.m_list_data.Delete(0);
      //--- save the new bar time as the previous one for the subsequent new bar check
      this.SaveNewBarTime(time);
     }

//--- Get the list of indicator data objects by the time of current bar start
   CArrayObj *list=CSelect::ByIndicatorDataProperty(this.GetList(),IND_DATA_PROP_TIME,time,EQUAL);
//--- In the loop, by the number of indicator buffers get indicator data objects from the list by loop index
   for(int i=0;i<this.m_buffers_total;i++)
     {
      //--- If failed to get object from the list - exit
      CDataInd *data_ind=this.GetIndDataByTime(i,time);
      if(data_ind==NULL)
         return;
      //--- Copy data of indicator’s i buffer from current bar
      int copied=::CopyBuffer(this.m_ind_handle,i,0,1,data);
      if(copied<1)
        {
         err=::GetLastError();
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_IND_DATA_FAILED_GET_CURRENT_DATA)," ",this.m_symbol," ",TimeframeDescription(this.m_timeframe),". ",
                      CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err));
         return;
        }
      //--- Add received value of indicator’s i buffer to indicator data object
      data_ind.SetBufferValue(data[0]);
     }
  }
//+------------------------------------------------------------------+
```

Logic of method operation is described in detail in its listing. The method updates data of all indicator buffers on the current bar. While on a new bar it creates new objects for the current bar and adds them to collection list. When collection list size exceeds the required value the oldest objects shall be removed from the list.

**Method which returns indicator data object by time and buffer number:**

```
//+------------------------------------------------------------------+
//| Return indicator data object by time and buffer number           |
//+------------------------------------------------------------------+
CDataInd* CSeriesDataInd::GetIndDataByTime(const int buffer_num,const datetime time)
  {
   CArrayObj *list=GetList(IND_DATA_PROP_TIME,time,EQUAL);
   list=CSelect::ByIndicatorDataProperty(list,IND_DATA_PROP_IND_BUFFER_NUM,buffer_num,EQUAL);
   return list.At(0);
  }
//+------------------------------------------------------------------+
```

Here: get the list which contains indicator data objects corresponding to the specified time,

then filter the received list so that only the object of the specified indicator buffer remains therein.

The list will feature only one object. Return it.

**Method which returns indicator data object by bar and buffer number:**

```
//+------------------------------------------------------------------+
//| Return indicator data object by bar and buffer number            |
//+------------------------------------------------------------------+
CDataInd* CSeriesDataInd::GetIndDataByBar(const int buffer_num,const uint shift)
  {
   datetime time=::iTime(this.m_symbol,this.m_timeframe,(int)shift);
   if(time==0)
      return NULL;
   return this.GetIndDataByTime(buffer_num,time);
  }
//+------------------------------------------------------------------+
```

Here: get the bar open time by the specified index, then return the object received with the use of the above considered method.

**Method which returns data of specified indicator buffer by time:**

```
//+------------------------------------------------------------------+
//| Return data of specified indicator buffer by time                |
//+------------------------------------------------------------------+
double CSeriesDataInd::BufferValue(const int buffer_num,const datetime time)
  {
   CDataInd *data_ind=this.GetIndDataByTime(buffer_num,time);
   return(data_ind==NULL ? EMPTY_VALUE : data_ind.BufferValue());
  }
//+------------------------------------------------------------------+
```

Here: get data object using GetIndDataByTime() method and return buffer value written in the object or “empty value” if the object was not received.

**Method which returns data of specified indicator buffer by bar index:**

```
//+------------------------------------------------------------------+
//| Return data of specified indicator buffer by bar index           |
//+------------------------------------------------------------------+
double CSeriesDataInd::BufferValue(const int buffer_num,const uint shift)
  {
   CDataInd *data_ind=this.GetIndDataByBar(buffer_num,shift);
   return(data_ind==NULL ? EMPTY_VALUE : data_ind.BufferValue());
  }
//+------------------------------------------------------------------+
```

Here: get data object using GetIndDataByBar() method and return buffer value written in the object or “empty value” if the object was not received.

All indicators created in the program shall be moved [to indicator object collection created in article 54](https://www.mql5.com/en/articles/8508#node04). Each such object features all data of the standard or custom indicator. However, some data must be supplemented. So that it is always known which number of buffers is possessed by the indicator to be described by the object of this class, a variable for storing the number of indicator buffers must be added to it. This is specifically relevant for custom indicators where you cannot know in advance the number of buffers to be drawn. Also, I must add the indicator buffer data collection list which class is just created. In such case indicator object will store also data lists of its all buffers. From them data on each bar of each indicator buffer may be received.

Open the indicator object class file \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **IndicatorDE.mqh** and make all the necessary improvements in it:

```
//+------------------------------------------------------------------+
//|                                                          Ind.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\Select.mqh"
#include "..\..\Objects\BaseObj.mqh"
#include "..\..\Objects\Indicators\SeriesDataInd.mqh"
//+------------------------------------------------------------------+
//| Abstract indicator class                                         |
//+------------------------------------------------------------------+
class CIndicatorDE : public CBaseObj
  {
protected:
   MqlParam          m_mql_param[];                                              // Array of indicator parameters
   CSeriesDataInd    m_series_data;                                              // Timeseries object of indicator buffer data
private:
   long              m_long_prop[INDICATOR_PROP_INTEGER_TOTAL];                  // Integer properties
   double            m_double_prop[INDICATOR_PROP_DOUBLE_TOTAL];                 // Real properties
   string            m_string_prop[INDICATOR_PROP_STRING_TOTAL];                 // String properties
   string            m_ind_type_description;                                     // Indicator type description
   int               m_buffers_total;                                            // Total number of indicator buffers

//--- Return the index of the array the buffer's (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_INDICATOR_PROP_DOUBLE property)        const { return(int)property-INDICATOR_PROP_INTEGER_TOTAL;                           }
   int               IndexProp(ENUM_INDICATOR_PROP_STRING property)        const { return(int)property-INDICATOR_PROP_INTEGER_TOTAL-INDICATOR_PROP_DOUBLE_TOTAL;}

//--- Comare (1) structures MqlParam, (2) array of structures MqlParam between each other
   bool              IsEqualMqlParams(MqlParam &struct1,MqlParam &struct2) const;
   bool              IsEqualMqlParamArrays(MqlParam &compared_struct[])    const;

protected:
//--- Protected parametric constructor
                     CIndicatorDE(ENUM_INDICATOR ind_type,
                                  string symbol,
                                  ENUM_TIMEFRAMES timeframe,
                                  ENUM_INDICATOR_STATUS status,
                                  ENUM_INDICATOR_GROUP group,
                                  string name,
                                  string shortname,
                                  MqlParam &mql_params[]);
public:
//--- Default constructor
                     CIndicatorDE(void){;}
//--- Destructor
                    ~CIndicatorDE(void);

//--- Set buffer's (1) integer, (2) real and (3) string property
   void              SetProperty(ENUM_INDICATOR_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                                        }
   void              SetProperty(ENUM_INDICATOR_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;                      }
   void              SetProperty(ENUM_INDICATOR_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;                      }
//--- Return buffer’s (1) integer, (2) real and (3) string property from the properties array
   long              GetProperty(ENUM_INDICATOR_PROP_INTEGER property)        const { return this.m_long_prop[property];                                       }
   double            GetProperty(ENUM_INDICATOR_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];                     }
   string            GetProperty(ENUM_INDICATOR_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];                     }
//--- Return description of buffer's (1) integer, (2) real and (3) string property
   string            GetPropertyDescription(ENUM_INDICATOR_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_INDICATOR_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_INDICATOR_PROP_STRING property);
//--- Return the flag of the buffer supporting the property
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_INTEGER property)          { return true;       }
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_DOUBLE property)           { return true;       }
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_STRING property)           { return true;       }

//--- Compare CIndicatorDE objects by all possible properties (for sorting the lists by a specified indicator object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CIndicatorDE objects by all properties (to search equal indicator objects)
   bool              IsEqual(CIndicatorDE* compared_obj) const;

//--- Set indicator’s (1) group, (2) empty value of buffers, (3) name, (4) short name, (5) ID
   void              SetGroup(const ENUM_INDICATOR_GROUP group)      { this.SetProperty(INDICATOR_PROP_GROUP,group);                         }
   void              SetEmptyValue(const double value)               { this.SetProperty(INDICATOR_PROP_EMPTY_VALUE,value);                   }
   void              SetName(const string name)                      { this.SetProperty(INDICATOR_PROP_NAME,name);                           }
   void              SetShortName(const string shortname)            { this.SetProperty(INDICATOR_PROP_SHORTNAME,shortname);                 }
   void              SetID(const int id)                             { this.SetProperty(INDICATOR_PROP_ID,id);                               }
   void              SetBuffersTotal(const int buffers_total)        { this.m_buffers_total=buffers_total;                                   }

//--- Return indicator’s (1) status, (2) group, (3) timeframe, (4) type, (5) handle, (6) ID,
//--- (7) empty value of buffers, (8) name, (9) short name, (10) symbol, (11) number of buffers
   ENUM_INDICATOR_STATUS Status(void)                          const { return (ENUM_INDICATOR_STATUS)this.GetProperty(INDICATOR_PROP_STATUS);}
   ENUM_INDICATOR_GROUP  Group(void)                           const { return (ENUM_INDICATOR_GROUP)this.GetProperty(INDICATOR_PROP_GROUP);  }
   ENUM_TIMEFRAMES   Timeframe(void)                           const { return (ENUM_TIMEFRAMES)this.GetProperty(INDICATOR_PROP_TIMEFRAME);   }
   ENUM_INDICATOR    TypeIndicator(void)                       const { return (ENUM_INDICATOR)this.GetProperty(INDICATOR_PROP_TYPE);         }
   int               Handle(void)                              const { return (int)this.GetProperty(INDICATOR_PROP_HANDLE);                  }
   int               ID(void)                                  const { return (int)this.GetProperty(INDICATOR_PROP_ID);                      }
   double            EmptyValue(void)                          const { return this.GetProperty(INDICATOR_PROP_EMPTY_VALUE);                  }
   string            Name(void)                                const { return this.GetProperty(INDICATOR_PROP_NAME);                         }
   string            ShortName(void)                           const { return this.GetProperty(INDICATOR_PROP_SHORTNAME);                    }
   string            Symbol(void)                              const { return this.GetProperty(INDICATOR_PROP_SYMBOL);                       }
   int               BuffersTotal(void)                        const { return this.m_buffers_total;                                          }

//--- Return description of (1) type, (2) status, (3) group, (4) timeframe, (5) empty value of indicator, (6) parameter of m_mql_param array
   string            GetTypeDescription(void)                  const { return m_ind_type_description;                                        }
   string            GetStatusDescription(void)                const;
   string            GetGroupDescription(void)                 const;
   string            GetTimeframeDescription(void)             const;
   string            GetEmptyValueDescription(void)            const;
   string            GetMqlParamDescription(const int index)   const;

//--- Return indicator data timeseries
   CSeriesDataInd   *GetSeriesData(void)                             { return &this.m_series_data;  }

//--- Display the description of indicator object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display (1) a short description, (2) description of indicator object parameters in the journal (implementation in the descendants)
   virtual void      PrintShort(void) {;}
   virtual void      PrintParameters(void) {;}

//--- Return data of specified buffer from specified bar (1) by index, (2) by bar time
   double            GetDataBuffer(const int buffer_num,const int index);
   double            GetDataBuffer(const int buffer_num,const datetime time);
  };
//+------------------------------------------------------------------+
```

To let the class see the object of indicator data collection class I included its file **SeriesDataInd.mqh** in the section of included files.

In the protected area of the class declare the object of indicator data collection class **m\_series\_data**.

**m\_buffers\_total** variable will store the total number of indicator buffers drawn. Data objects of all these buffers will be stored in indicator buffer data collection.

Methods  **SetBuffersTotal()** and **BuffersTotal()** set/return the total number of indicator buffers drawn.

Method **GetSeriesData()** returns a pointer to indicator buffer data collection.

Now, open the file of indicator collection class \\MQL5\\Include\\DoEasy\\Collections\ **IndicatorsCollection.mqh** and add necessary improvements to it.

During indicator creation, the object of abstract indicator class is created with clarification of all data in correspondence with the type of indicator created. Then, this indicator object is added to collection list. In the moment when the created indicator object is added to the collection using AddIndicatorToList() method the necessary parameters of the indicator are additionally set for its exact identification.

Add to this method specification of the number of indicator buffers drawn and required amount of its buffer data:

```
//--- (1) Create, (2) add to collection list a new indicator object and set an ID for it
   CIndicatorDE           *CreateIndicator(const ENUM_INDICATOR ind_type,MqlParam &mql_param[],const string symbol_name=NULL,const ENUM_TIMEFRAMES period=PERIOD_CURRENT);
   int                     AddIndicatorToList(CIndicatorDE *indicator,const int id,const int buffers_total,const uint required=0);
```

In the method of custom indicator creation additionally pass the total number of its buffers drawn:

```
   int                     CreateCustom(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id,const int buffers_total,
                                       ENUM_INDICATOR_GROUP group,
                                       MqlParam &mql_param[]);
```

Declare the method which returns the pointer to the data object of indicator buffer timeseries by specified time, and methods for creation, update and return of data from indicator buffer data collection:

```
//--- Return (1) indicator object by its ID, (2) the data object of indicator buffer timeseries by time
   CIndicatorDE           *GetIndByID(const uint id);
   CDataInd               *GetDataIndObj(const uint ind_id,const int buffer_num,const datetime time);

//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(void);
   void                    PrintShort(void);

//--- Create (1) timeseries of specified indicator data, (2) all timeseries used of all collection indicators
   bool                    SeriesCreate(CIndicatorDE *indicator,const uint required=0);
   bool                    SeriesCreateAll(const uint required=0);
//--- Update buffer data of all indicators
   void                    SeriesRefreshAll(void);
   void                    SeriesRefresh(const int ind_id);
//--- Return by bar (1) time, (2) number the value of the buffer specified by indicator ID
   double                  GetBufferValue(const uint ind_id,const int buffer_num,const datetime time);
   double                  GetBufferValue(const uint ind_id,const int buffer_num,const uint shift);

//--- Constructor
                           CIndicatorsCollection();

  };
//+------------------------------------------------------------------+
```

In implementation of AddIndicatorToList() method add a setting of a number of indicator buffers  and creation of its timeseries:

```
//+------------------------------------------------------------------+
//| Add a new indicator object to collection list                    |
//+------------------------------------------------------------------+
int CIndicatorsCollection::AddIndicatorToList(CIndicatorDE *indicator,const int id,const int buffers_total,const uint required=0)
  {
//--- If invalid indicator is passed to the object - return INVALID_HANDLE
   if(indicator==NULL)
      return INVALID_HANDLE;
//--- If such indicator is already in the list
   int index=this.Index(indicator);
   if(index!=WRONG_VALUE)
     {
      //--- Remove the earlier created object, get indicator object from the list and return indicator handle
      delete indicator;
      indicator=this.m_list.At(index);
     }
//--- If indicator object is not in the list yet
   else
     {
      //--- If failed to add indicator object to the list - display a corresponding message,
      //--- remove object and return INVALID_HANDLE
      if(!this.m_list.Add(indicator))
        {
         ::Print(CMessage::Text(MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST));
         delete indicator;
         return INVALID_HANDLE;
        }
     }
//--- If indicator is successfully added to the list or is already there...
//--- If indicator with specified ID (not -1) is not in the list - set ID
   if(id>WRONG_VALUE && !this.CheckID(id))
      indicator.SetID(id);
//--- Set the total number of buffers and create data timeseries of all indicator buffers
   indicator.SetBuffersTotal(buffers_total);
   this.SeriesCreate(indicator,required);
//--- Return handle of a new indicator added to the list
   return indicator.Handle();
  }
//+------------------------------------------------------------------+
```

Since now the total number of indicator buffers is also passed to AddIndicatorToList() method, in all methods of indicator object creation the passing of the value of buffer number must be added. For standard indicators their exact number is known, while for the custom indicator this number is passed in the method of its creation.

Changes have already been made in all such class methods. Let's consider only some of them.

**Method of Accelerator Oscillator indicator creation:**

```
//+------------------------------------------------------------------+
//| Create a new indicator object Accelerator Oscillator             |
//| and place it to the collection list                              |
//+------------------------------------------------------------------+
int CIndicatorsCollection::CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id)
  {
//--- AC indicator possesses no parameters - resize the array of parameter structures
   ::ArrayResize(this.m_mql_param,0);
//--- Create indicator object
   CIndicatorDE *indicator=this.CreateIndicator(IND_AC,this.m_mql_param,symbol,timeframe);
//--- Return indicator handle received as a result of adding the object to collection list
   return this.AddIndicatorToList(indicator,id,1);
  }
//+------------------------------------------------------------------+
```

When calling the method for adding indicator object to collection list I additionally specify that this indicator has one drawn buffer.

**Method for creation of Average Directional Movement Index indicator:**

```
//+------------------------------------------------------------------+
//| Create new indicator object                                      |
//| Average Directional Movement Index                               |
//| and place it to the collection list                              |
//+------------------------------------------------------------------+
int CIndicatorsCollection::CreateADX(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id,const int adx_period=14)
  {
//--- Add required indicator parameters to the array of parameter structures
   ::ArrayResize(this.m_mql_param,1);
   this.m_mql_param[0].type=TYPE_INT;
   this.m_mql_param[0].integer_value=adx_period;
//--- Create indicator object
   CIndicatorDE *indicator=this.CreateIndicator(IND_ADX,this.m_mql_param,symbol,timeframe);
//--- Return indicator handle received as a result of adding the object to collection list
   return this.AddIndicatorToList(indicator,id,3);
  }
//+------------------------------------------------------------------+
```

When calling the method for adding indicator object to collection list I additionally specify that this indicator has three drawn buffers.

**Method of custom indicator creation:**

```
//+------------------------------------------------------------------+
//| Create a new object - custom indicator                           |
//| and place it to the collection list                              |
//+------------------------------------------------------------------+
int CIndicatorsCollection::CreateCustom(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id,
                                        const int buffers_total,
                                        ENUM_INDICATOR_GROUP group,
                                        MqlParam &mql_param[])
  {
//--- Create indicator object
   CIndicatorDE *indicator=this.CreateIndicator(IND_CUSTOM,mql_param,symbol,timeframe);
   if(indicator==NULL)
      return INVALID_HANDLE;
//--- Set a group for indicator object
   indicator.SetGroup(group);
//--- Return indicator handle received as a result of adding the object to collection list
   return this.AddIndicatorToList(indicator,id,buffers_total);
  }
//+------------------------------------------------------------------+
```

In method’s input variables pass the value of the number of drawn buffers of the indicator, while when calling the method for adding indicator object to collection listspecify the number of drawn buffers passed to the method.

**Method which returns pointer to the data object of indicator buffer timeseries by time:**

```
//+------------------------------------------------------------------+
//| Return data object of indicator buffer timeseries by time        |
//+------------------------------------------------------------------+
CDataInd *CIndicatorsCollection::GetDataIndObj(const uint ind_id,const int buffer_num,const datetime time)
  {
   CIndicatorDE *indicator=this.GetIndByID(ind_id);
   if(indicator==NULL) return NULL;
   CSeriesDataInd *buffers_data=indicator.GetSeriesData();
   if(buffers_data==NULL) return NULL;
   return buffers_data.GetIndDataByTime(buffer_num,time);
  }
//+------------------------------------------------------------------+
```

Here: get indicator object from the collection by its ID,

get the pointer to collection list of its buffer data,

return data of the specified buffer by timeseries bar open time.

**Method which creates data timeseries of the specified indicator:**

```
//+------------------------------------------------------------------+
//| Create data timeseries of the specified indicator                |
//+------------------------------------------------------------------+
bool CIndicatorsCollection::SeriesCreate(CIndicatorDE *indicator,const uint required=0)
  {
   if(indicator==NULL)
      return false;
   CSeriesDataInd *buffers_data=indicator.GetSeriesData();
   if(buffers_data!=NULL)
     {
      buffers_data.SetSymbolPeriod(indicator.Symbol(),indicator.Timeframe());
      buffers_data.SetIndHandle(indicator.Handle());
      buffers_data.SetIndID(indicator.ID());
      buffers_data.SetIndBuffersTotal(indicator.BuffersTotal());
      buffers_data.SetRequiredUsedData(required);
     }
   return(buffers_data!=NULL ? buffers_data.Create(required)>0 : false);
  }
//+------------------------------------------------------------------+
```

Here: method receives the pointer to indicator object and required number of created bars of indicator buffer data.

From indicator object get the pointer to its buffer data collection,

set all necessary parameters of data collection and

return the result of creation of the requested number of indicator buffer data.

**Method which creates all timeseries used of all collection indicators:**

```
//+------------------------------------------------------------------+
//| Create all timeseries used of all collection indicators          |
//+------------------------------------------------------------------+
bool CIndicatorsCollection::SeriesCreateAll(const uint required=0)
  {
   bool res=true;
   for(int i=0;i<m_list.Total();i++)
     {
      CIndicatorDE *indicator=m_list.At(i);
      if(!this.SeriesCreate(indicator,required))
         res&=false;
     }
   return res;
  }
//+------------------------------------------------------------------+
```

Here: in the loop by the total number of indicator objects in the collectionget the pointer to further indicator object and to **res** variable add the result of creating the timeseries of its buffer data using the above considered method. Upon the loop completion, return the value of variable **res**. If at least one buffer data timeseries was not created this variable will have the false value.

**Update method for buffer data of all collection indicators:**

```
//+------------------------------------------------------------------+
//| Update buffer data of all indicators                             |
//+------------------------------------------------------------------+
void CIndicatorsCollection::SeriesRefreshAll(void)
  {
   for(int i=0;i<m_list.Total();i++)
     {
      CIndicatorDE *indicator=m_list.At(i);
      if(indicator==NULL) continue;
      CSeriesDataInd *buffers_data=indicator.GetSeriesData();
      if(buffers_data==NULL) continue;
      buffers_data.Refresh();
     }
  }
//+------------------------------------------------------------------+
```

Here: in the loop by the total number of indicator objects in the collectionget the pointer to further indicator object, get the pointer to its data object of buffer timeseries  and update the timeseries using Refresh() method.

**Update method for buffer data of the specified indicator:**

```
//+------------------------------------------------------------------+
//| Update buffer data of the specified indicator                    |
//+------------------------------------------------------------------+
void CIndicatorsCollection::SeriesRefresh(const int ind_id)
  {
   CIndicatorDE *indicator=this.GetIndByID(ind_id);
   if(indicator==NULL) return;
   CSeriesDataInd *buffers_data=indicator.GetSeriesData();
   if(buffers_data==NULL) return;
   buffers_data.Refresh();
  }
//+------------------------------------------------------------------+
```

The method receives the ID of the required indicator. Using GetIndByID() method get the pointer to the required indicator, get its timeseries object of buffer data and update the timeseries using Refresh() method.

**Methods which return the value of the specified buffer of the specified indicator by time or bar index:**

```
//+------------------------------------------------------------------+
//| Return buffer value by bar time                                  |
//|  specified by indicator ID                                       |
//+------------------------------------------------------------------+
double CIndicatorsCollection::GetBufferValue(const uint ind_id,const int buffer_num,const datetime time)
  {
   CIndicatorDE *indicator=GetIndByID(ind_id);
   if(indicator==NULL) return EMPTY_VALUE;
   CSeriesDataInd *series=indicator.GetSeriesData();
   return(series!=NULL && series.DataTotal()>0 ? series.BufferValue(buffer_num,time) : EMPTY_VALUE);
  }
//+------------------------------------------------------------------+
//| Return by bar number the buffer value                            |
//|  specified by indicator ID                                       |
//+------------------------------------------------------------------+
double CIndicatorsCollection::GetBufferValue(const uint ind_id,const int buffer_num,const uint shift)
  {
   CIndicatorDE *indicator=GetIndByID(ind_id);
   if(indicator==NULL) return EMPTY_VALUE;
   CSeriesDataInd *series=indicator.GetSeriesData();
   return(series!=NULL && series.DataTotal()>0 ? series.BufferValue(buffer_num,shift) : EMPTY_VALUE);
  }
//+------------------------------------------------------------------+
```

Methods are identical to each other except for the fact that to the first one the bar open time is passed, while bar index is passed to the second one.

Get the pointer to the necessary indicator in the collection, get the pointer to its collection object of buffer data and return the value of the specified buffer using overloaded method BufferValue().

To connect library classes with the “external world” the main library class **CEngine** is used. It is located in file \\MQL5\\Include\\DoEasy\ **Engine.mqh** where methods of access to all library methods from programs are located.

To the public class section add the methods for handling collections of indicator buffer data:

```
//--- Return (1) the indicator collection, (2) the indicator list from the collection
   CIndicatorsCollection *GetIndicatorsCollection(void)                                { return &this.m_indicators;              }
   CArrayObj           *GetListIndicators(void)                                        { return this.m_indicators.GetList();     }

//--- Create all timeseries used of all collection indicators
   bool                 IndicatorSeriesCreateAll(void)                                 { return this.m_indicators.SeriesCreateAll();}
//--- Update buffer data of all indicators
   void                 IndicatorSeriesRefreshAll(void)                                { this.m_indicators.SeriesRefreshAll();   }
   void                 IndicatorSeriesRefresh(const int ind_id)                       { this.m_indicators.SeriesRefresh(ind_id);}

//--- Return the value of the specified buffer by (1) time, (2) number of the bar specified by indicator ID
   double               IndicatorGetBufferValue(const uint ind_id,const int buffer_num,const datetime time)
                          { return this.m_indicators.GetBufferValue(ind_id,buffer_num,time); }
   double               IndicatorGetBufferValue(const uint ind_id,const int buffer_num,const uint shift)
                          { return this.m_indicators.GetBufferValue(ind_id,buffer_num,shift); }
```

All these methods call corresponding methods added to collection class of the indicators above.

In the class constructor, create a new timer for updating collections of indicator buffer data:

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),
                     m_last_trade_event(TRADE_EVENT_NO_EVENT),
                     m_last_account_event(WRONG_VALUE),
                     m_last_symbol_event(WRONG_VALUE),
                     m_global_error(ERR_SUCCESS)
  {
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_is_tester=::MQLInfoInteger(MQL_TESTER);
   this.m_program=(ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE);
   this.m_name=::MQLInfoString(MQL_PROGRAM_NAME);

   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_ORD_COUNTER_ID,COLLECTION_ORD_COUNTER_STEP,COLLECTION_ORD_PAUSE);
   this.CreateCounter(COLLECTION_ACC_COUNTER_ID,COLLECTION_ACC_COUNTER_STEP,COLLECTION_ACC_PAUSE);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID1,COLLECTION_SYM_COUNTER_STEP1,COLLECTION_SYM_PAUSE1);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID2,COLLECTION_SYM_COUNTER_STEP2,COLLECTION_SYM_PAUSE2);
   this.CreateCounter(COLLECTION_REQ_COUNTER_ID,COLLECTION_REQ_COUNTER_STEP,COLLECTION_REQ_PAUSE);
   this.CreateCounter(COLLECTION_TS_COUNTER_ID,COLLECTION_TS_COUNTER_STEP,COLLECTION_TS_PAUSE);
   this.CreateCounter(COLLECTION_IND_TS_COUNTER_ID,COLLECTION_IND_TS_COUNTER_STEP,COLLECTION_IND_TS_PAUSE);

   ::ResetLastError();
   #ifdef __MQL5__
      if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_TIMER),(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   //---__MQL4__
   #else
      if(!this.IsTester() && !::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_TIMER),(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   #endif
   //---
  }
//+------------------------------------------------------------------+
```

In class timer add code blocks for working with timeseries of indicator buffer data by timer and by tick (in tester):

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(SDataCalculate &data_calculate)
  {
//--- If this is not a tester, work with collection events by timer
   if(!this.IsTester())
     {
   //--- Timer of the collections of historical orders and deals, as well as of market orders and positions
      int index=this.CounterIndex(COLLECTION_ORD_COUNTER_ID);
      CTimerCounter* cnt1=this.m_list_counters.At(index);
      if(cnt1!=NULL)
        {
         //--- If unpaused, work with the order, deal and position collections events
         if(cnt1.IsTimeDone())
            this.TradeEventsControl();
        }
   //--- Account collection timer
      index=this.CounterIndex(COLLECTION_ACC_COUNTER_ID);
      CTimerCounter* cnt2=this.m_list_counters.At(index);
      if(cnt2!=NULL)
        {
         //--- If unpaused, work with the account collection events
         if(cnt2.IsTimeDone())
            this.AccountEventsControl();
        }
   //--- Timer 1 of the symbol collection (updating symbol quote data in the collection)
      index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID1);
      CTimerCounter* cnt3=this.m_list_counters.At(index);
      if(cnt3!=NULL)
        {
         //--- If the pause is over, update quote data of all symbols in the collection
         if(cnt3.IsTimeDone())
            this.m_symbols.RefreshRates();
        }
   //--- Timer 2 of the symbol collection (updating all data of all symbols in the collection and tracking symbl and symbol search events in the market watch window)
      index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID2);
      CTimerCounter* cnt4=this.m_list_counters.At(index);
      if(cnt4!=NULL)
        {
         //--- If the pause is over
         if(cnt4.IsTimeDone())
           {
            //--- update data and work with events of all symbols in the collection
            this.SymbolEventsControl();
            //--- When working with the market watch list, check the market watch window events
            if(this.m_symbols.ModeSymbolsList()==SYMBOLS_MODE_MARKET_WATCH)
               this.MarketWatchEventsControl();
           }
        }
   //--- Trading class timer
      index=this.CounterIndex(COLLECTION_REQ_COUNTER_ID);
      CTimerCounter* cnt5=this.m_list_counters.At(index);
      if(cnt5!=NULL)
        {
         //--- If unpaused, work with the list of pending requests
         if(cnt5.IsTimeDone())
            this.m_trading.OnTimer();
        }
   //--- Timeseries collection timer
      index=this.CounterIndex(COLLECTION_TS_COUNTER_ID);
      CTimerCounter* cnt6=this.m_list_counters.At(index);
      if(cnt6!=NULL)
        {
         //--- If the pause is over, work with the timeseries list (update all except the current one)
         if(cnt6.IsTimeDone())
            this.SeriesRefreshAllExceptCurrent(data_calculate);
        }

   //--- Timer of timeseries collection of indicator buffer data
      index=this.CounterIndex(COLLECTION_IND_TS_COUNTER_ID);
      CTimerCounter* cnt7=this.m_list_counters.At(index);
      if(cnt7!=NULL)
        {
         //--- If the pause is over, work with the timeseries list of indicator data (update all except for the current one)
         if(cnt7.IsTimeDone())
            this.IndicatorSeriesRefreshAll();
        }
     }
//--- If this is a tester, work with collection events by tick
   else
     {
      //--- work with events of collections of orders, deals and positions by tick
      this.TradeEventsControl();
      //--- work with events of collections of accounts by tick
      this.AccountEventsControl();
      //--- update quote data of all collection symbols by tick
      this.m_symbols.RefreshRates();
      //--- work with events of all symbols in the collection by tick
      this.SymbolEventsControl();
      //--- work with the list of pending orders by tick
      this.m_trading.OnTimer();
      //--- work with the timeseries list by tick
      this.SeriesRefresh(data_calculate);
      //--- work with the timeseries list of indicator buffers by tick
      this.IndicatorSeriesRefreshAll();
     }
  }
//+------------------------------------------------------------------+
```

**These are all the improvements for now.**

### Testing

To perform the test, let's use [the EA from the previous article](https://www.mql5.com/en/articles/8705#node04) and

save it in the new folder \\MQL5\\Experts\\TestDoEasy\ **Part58\** under the new name **TestDoEasyPart58.mq5**.

Perform testing in the same manner as I did in the previous EA: four indicators, two of which are standard and two are custom ones.

The difference is that in the previous EA I created objects of all classes immediately in it, while now I will use the objects of buffer data collection classes of created indicators and which objects were created in the library.

From the global area remove pointers to indicator data objects:

```
//--- Pointers to indicator data objects
CDataInd      *data_ma1_0=NULL;
CDataInd      *data_ma1_1=NULL;
CDataInd      *data_ma2_0=NULL;
CDataInd      *data_ma2_1=NULL;
CDataInd      *data_ama1_0=NULL;
CDataInd      *data_ama1_1=NULL;
CDataInd      *data_ama2_0=NULL;
CDataInd      *data_ama2_1=NULL;
//+------------------------------------------------------------------+
```

In handler **OnInit()** of EA add the number of buffers of custom indicators created:

```
//--- Create indicators
   ArrayResize(param_ma1,4);
   //--- Name of indicator 1
   param_ma1[0].type=TYPE_STRING;
   param_ma1[0].string_value="Examples\\Custom Moving Average.ex5";
   //--- Calculation period
   param_ma1[1].type=TYPE_INT;
   param_ma1[1].integer_value=13;
   //--- Horizontal shift
   param_ma1[2].type=TYPE_INT;
   param_ma1[2].integer_value=0;
   //--- Smoothing method
   param_ma1[3].type=TYPE_INT;
   param_ma1[3].integer_value=MODE_SMA;
   //--- Create indicator 1
   engine.GetIndicatorsCollection().CreateCustom(NULL,PERIOD_CURRENT,MA1,1,INDICATOR_GROUP_TREND,param_ma1);

   ArrayResize(param_ma2,5);
   //--- Name of indicator 2
   param_ma2[0].type=TYPE_STRING;
   param_ma2[0].string_value="Examples\\Custom Moving Average.ex5";
   //--- Calculation period
   param_ma2[1].type=TYPE_INT;
   param_ma2[1].integer_value=13;
   //--- Horizontal shift
   param_ma2[2].type=TYPE_INT;
   param_ma2[2].integer_value=0;
   //--- Smoothing method
   param_ma2[3].type=TYPE_INT;
   param_ma2[3].integer_value=MODE_SMA;
   //--- Calculation price
   param_ma2[4].type=TYPE_INT;
   param_ma2[4].integer_value=PRICE_OPEN;
   //--- Create indicator 2
   engine.GetIndicatorsCollection().CreateCustom(NULL,PERIOD_CURRENT,MA2,1,INDICATOR_GROUP_TREND,param_ma2);
```

In handler **OnDeinit()** of EA delete removal of created indicator objects:

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Remove EA graphical objects by an object name prefix
   ObjectsDeleteAll(0,prefix);
   Comment("");
//--- Remove created data objects of MA indicators
   if(CheckPointer(data_ma1_0)==POINTER_DYNAMIC)
      delete data_ma1_0;
   if(CheckPointer(data_ma1_1)==POINTER_DYNAMIC)
      delete data_ma1_1;
   if(CheckPointer(data_ma2_0)==POINTER_DYNAMIC)
      delete data_ma2_0;
   if(CheckPointer(data_ma2_1)==POINTER_DYNAMIC)
      delete data_ma2_1;
//--- Remove created data objects of AMA indicators
   if(CheckPointer(data_ama1_0)==POINTER_DYNAMIC)
      delete data_ama1_0;
   if(CheckPointer(data_ama1_1)==POINTER_DYNAMIC)
      delete data_ama1_1;
   if(CheckPointer(data_ama2_0)==POINTER_DYNAMIC)
      delete data_ama2_0;
   if(CheckPointer(data_ama2_1)==POINTER_DYNAMIC)
      delete data_ama2_1;
//--- Deinitialize library
   engine.OnDeinit();
  }
//+------------------------------------------------------------------+
```

Compared to the previous EA, now **OnTick()** handler has become much shorter:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Handle the NewTick event in the library
   engine.OnTick(rates_data);

//--- If work in tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Work in timer
      PressButtonsControl();        // Button press control
      engine.EventsHandling();      // Work with events
     }

//--- Get indicator values by their IDs from data collections of their buffers from bars 0 and 1
   double ma1_value0=engine.IndicatorGetBufferValue(MA1,0,0), ma1_value1=engine.IndicatorGetBufferValue(MA1,0,1);
   double ma2_value0=engine.IndicatorGetBufferValue(MA2,0,0), ma2_value1=engine.IndicatorGetBufferValue(MA2,0,1);
   double ama1_value0=engine.IndicatorGetBufferValue(AMA1,0,0), ama1_value1=engine.IndicatorGetBufferValue(AMA1,0,1);
   double ama2_value0=engine.IndicatorGetBufferValue(AMA2,0,0), ama2_value1=engine.IndicatorGetBufferValue(AMA2,0,1);

//--- Display the values of indicator buffers to comment on the chart from data objects
   Comment
     (
      "ma1(1)=",DoubleToString(ma1_value1,6),", ma1(0)=",DoubleToString(ma1_value0,6),"  ",
      "ma2(1)=",DoubleToString(ma2_value1,6),", ma2(0)=",DoubleToString(ma2_value0,6),", \n",
      "ama1(1)=",DoubleToString(ama1_value1,6),", ama1(0)=",DoubleToString(ama1_value0,6),"  ",
      "ama1(1)=",DoubleToString(ama2_value1,6),", ama1(0)=",DoubleToString(ama2_value0,6)
     );

//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();          // Trailing positions
      TrailingOrders();             // Trailing of pending orders
     }
  }
//+------------------------------------------------------------------+
```

Compile the EA and launch it on the chart having set in settings to use only current symbol and current timeframe. In comments on the chart, data of the first and zero (current) bars of all created indicators will be displayed:

![](https://c.mql5.com/2/41/terminal64_fvXxN7t8xw.png)

Same indicators with the same settings are plotted on the chart for more clarity - indicator data in comments on the chart and in data window (Ctrl+D) match and values on the current bar update.

### What's next?

The following article will start preparation for creation of classes to handle ticks and market depth.

All files of the current library version are attached below together with the test EA file for MQL5. You can download them and test everything.

Leave your comments, questions and suggestions in the comments to the article.

[Back to contents](https://www.mql5.com/en/articles/8787#node00)

**Previous articles within the series:**

[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)

[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)

[Timeseries in DoEasy library (part 37): Timeseries collection - database of timeseries by symbols and periods](https://www.mql5.com/en/articles/7663)

[Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program](https://www.mql5.com/en/articles/7695)

[Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://www.mql5.com/en/articles/7724)

[Timeseries in DoEasy library (part 40): Library-based indicators - updating data in real time](https://www.mql5.com/en/articles/7771)

[Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://www.mql5.com/en/articles/7804)

[Timeseries in DoEasy library (part 42): Abstract indicator buffer object class](https://www.mql5.com/en/articles/7821)

[Timeseries in DoEasy library (part 43): Classes of indicator buffer objects](https://www.mql5.com/en/articles/7868)

[Timeseries in DoEasy library (part 44): Collection class of indicator buffer objects](https://www.mql5.com/en/articles/7886)

[Timeseries in DoEasy library (part 45): Multi-period indicator buffers](https://www.mql5.com/en/articles/8023)

[Timeseries in DoEasy library (part 46): Multi-period multi-symbol indicator buffers](https://www.mql5.com/en/articles/8115)

[Timeseries in DoEasy library (part 47): Multi-period multi-symbol standard indicators](https://www.mql5.com/en/articles/8207)

[Timeseries in DoEasy library (part 48): Multi-period multi-symbol indicators on one buffer in subwindow](https://www.mql5.com/en/articles/8257)

[Timeseries in DoEasy library (part 49): Multi-period multi-symbol multi-buffer standard indicators](https://www.mql5.com/en/articles/8292)

[Timeseries in DoEasy library (part 50): Multi-period multi-symbol standard indicators with a shift](https://www.mql5.com/en/articles/8331)

[Timeseries in DoEasy library (part 51): Composite multi-period multi-symbol standard indicators](https://www.mql5.com/en/articles/8354)

[Timeseries in DoEasy library (part 52): Cross-platform nature of multi-period multi-symbol single-buffer standard indicators](https://www.mql5.com/en/articles/8399)

[Timeseries in DoEasy library (part 53): Abstract base indicator class](https://www.mql5.com/en/articles/8464)

[Timeseries in DoEasy library (part 54): Descendant classes of abstract base indicator](https://www.mql5.com/en/articles/8508)

[Timeseries in DoEasy library (part 55): Indicator collection class](https://www.mql5.com/en/articles/8576/)

[Timeseries in DoEasy library (part 56): Custom indicator object, get data from indicator objects in the collection](https://www.mql5.com/en/articles/8646)

[Timeseries in DoEasy library (part 57): Indicator buffer data object](https://www.mql5.com/en/articles/8705)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8787](https://www.mql5.com/ru/articles/8787)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8787.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8787/mql5.zip "Download MQL5.zip")(3868 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/361146)**
(18)


![Aleksey Mavrin](https://c.mql5.com/avatar/avatar_na2.png)

**[Aleksey Mavrin](https://www.mql5.com/en/users/alex_all)**
\|
12 Dec 2020 at 20:52

**Artyom Trishkin:**

What's stopping you from writing in and picking up this medal yourself?

I have no claim. Let you have it. You deserve it ;)

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
12 Dec 2020 at 21:41

**Aleksey Mavrin:**

I have no claim. Let you have it. You deserve it ;)

Thank you. I don't want it. We'll leave it to others.

![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
13 Dec 2020 at 00:06

It's a pity, but there is no strong side of MT5 programming community as mutual [improvement of code](https://www.mql5.com/en/articles/2661 "Article \"How to quickly develop and debug a trading strategy in MetaTrader 5\"") and creation of help for public works. Those who write help, for example, always improve their understanding of the code.

Maybe it is for the best! I think I will still read all the articles from the DoEasy cycle. I've reached article 34, I'll tell you that it's still a witty coboeuvre)). Pohodu reading think that it's bad that there is no diagram of construction, this software art, **but I think maybe for the best that you need to think a little.**

Conclusions from the articles are very much, especially for amateur programmers, which most traders novadais. Thank you! I hope there will be more.

![Aleksey Mavrin](https://c.mql5.com/avatar/avatar_na2.png)

**[Aleksey Mavrin](https://www.mql5.com/en/users/alex_all)**
\|
13 Dec 2020 at 11:01

Shit, why does everyone keep asking for this certificate? There is a help for SB, so what? I use it only if I forget the specific name of a method, a little faster than opening the class description).

In SB in timeseries there is a method that changes the buffer size. Without it the work with [time series](https://www.mql5.com/en/articles/252 "Article: Forecasting Time Series in MetaTrader 5 Using the ENCOG Machine Learning Library ") is limited, and there is no word about it in the help, about default values too.

So why such a flawed help if there is a code.

In any case, to start serious use you should go through the code, starting with interfaces and further on.

So don't distract Artyom with the help, let him do what he can).

![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
18 Dec 2020 at 06:57

**Aleksey Mavrin:**

Shit, why does everyone keep asking for this certificate? There is a help for SB, so what? I use it only if I forget the specific name of a method, a little faster than opening the class description).

In SB in timeseries there is a method that changes the buffer size. Without it the work with time series is limited, and there is no word about it in the help, and about default values too.

So why such a flawed help, if there is a code.

In any case, to start serious use you should go through the code, starting with interfaces and further on.

So don't distract Artyom with the help, let him do what he can do)

The work is still large and it is necessary to have a certificate and registration on [gitHub](https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development").

![Using spreadsheets to build trading strategies](https://c.mql5.com/2/41/spread_sheets_strategy.png)[Using spreadsheets to build trading strategies](https://www.mql5.com/en/articles/8699)

The article describes the basic principles and methods that allow you to analyze any strategy using spreadsheets (Excel, Calc, Google). The obtained results are compared with MetaTrader 5 tester.

![Manual charting and trading toolkit (Part II). Chart graphics drawing tools](https://c.mql5.com/2/40/MQL5-set_of_tools.png)[Manual charting and trading toolkit (Part II). Chart graphics drawing tools](https://www.mql5.com/en/articles/7908)

This is the next article within the series, in which I show how I created a convenient library for manual application of chart graphics by utilizing keyboard shortcuts. The tools used include straight lines and their combinations. In this part, we will view how the drawing tools are applied using the functions described in the first part. The library can be connected to any Expert Advisor or indicator which will greatly simplify the charting tasks. This solution DOES NOT use external dlls, while all the commands are implemented using built-in MQL tools.

![Prices in DoEasy library (part 59): Object to store data of one tick](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__3.png)[Prices in DoEasy library (part 59): Object to store data of one tick](https://www.mql5.com/en/articles/8818)

From this article on, start creating library functionality to work with price data. Today, create an object class which will store all price data which arrived with yet another tick.

![Brute force approach to pattern search (Part II): Immersion](https://c.mql5.com/2/41/Back-to-the-Future-Part-II-1.png)[Brute force approach to pattern search (Part II): Immersion](https://www.mql5.com/en/articles/8660)

In this article we will continue discussing the brute force approach. I will try to provide a better explanation of the pattern using the new improved version of my application. I will also try to find the difference in stability using different time intervals and timeframes.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/8787&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071855777011937213)

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