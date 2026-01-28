---
title: Prices in DoEasy library (part 59): Object to store data of one tick
url: https://www.mql5.com/en/articles/8818
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:51:07.031132
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/8818&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083157038558615065)

MetaTrader 5 / Examples


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8818#node01)
- [Preparing data](https://www.mql5.com/en/articles/8818#node02)
- [Tick data object class](https://www.mql5.com/en/articles/8818#node03)
- [Testing tick data object](https://www.mql5.com/en/articles/8818#node04)
- [What's next?](https://www.mql5.com/en/articles/8818#node05)


### Concept

From this article on, start development of library functionality to work with tick data.

The concept of storage and use of tick data will be similar to [the concept of storage of timeseries data](https://www.mql5.com/en/articles/7594) where the minimal data unit is the bar. Bar objects are stored in the lists which belong to corresponding timeframes which in their turn are stored in the lists which belong to symbols.

In the concept of tick data storage, the minimal unit of data volume shall be values of price structure on one tick. Such values are described with the use of a structure for storing the last prices by symbol [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick). An object to store such values will possess additional properties: spread - the difference of Ask and Bid prices and the symbol the data of which one tick are described by the object.

For more convenient list structuring, own lists of objects with tick data will be created for each of the symbols. Of course, each of the lists of tick data objects will automatically update, new data of newly incoming ticks will be added to it. While, the size of lists will be supported in the volume determined by library user.

The concept of data storage in the library provides search and sorting by any of object properties stored in the lists, which allows for receiving required data for their subsequent use or performing analytical studies. Thus, tick data will possess the same feature. This will allow the user to quickly analyse tick data flows for their tracking; for example, any changes in the nature of their occurrence, or for searching set patterns, etc.

### Preparing data

Like with any library objects, text messages must be added to display description of their parameters and to create enumeration of object properties by which their search in the lists or sorting objects by these properties will be possible.

In file \\MQL5\\Include\\DoEasy\ **Data.mqh** add new message indices:

```
//--- CSeriesDataInd
   MSG_LIB_TEXT_METHOD_NOT_FOR_INDICATORS,            // The method is not intended to work with indicator programs
   MSG_LIB_TEXT_IND_DATA_FAILED_GET_SERIES_DATA,      // Failed to get indicator data timeseries
   MSG_LIB_TEXT_IND_DATA_FAILED_GET_CURRENT_DATA,     // Failed to get current data of indicator buffer
   MSG_LIB_SYS_FAILED_CREATE_IND_DATA_OBJ,            // Failed to create indicator data object
   MSG_LIB_TEXT_IND_DATA_FAILED_ADD_TO_LIST,          // Failed to add indicator data object to list

//--- CTick
   MSG_TICK_TEXT_TICK,                                // Tick
   MSG_TICK_TIME_MSC,                                 // Time of the last update of prices in milliseconds
   MSG_TICK_TIME,                                     // Time of the last update of prices
   MSG_TICK_VOLUME,                                   // Volume for the current Last price
   MSG_TICK_FLAGS,                                    // Flags
   MSG_TICK_VOLUME_REAL,                              // Volume for the current Last price with greater accuracy
   MSG_TICK_SPREAD,                                   // Spread
   MSG_LIB_TEXT_TICK_CHANGED_DATA,                    // Changed data on tick:
   MSG_LIB_TEXT_TICK_FLAG_BID,                        // Bid price change
   MSG_LIB_TEXT_TICK_FLAG_ASK,                        // Ask price change
   MSG_LIB_TEXT_TICK_FLAG_LAST,                       // Last deal price change
   MSG_LIB_TEXT_TICK_FLAG_VOLUME,                     // Volume change

  };
//+------------------------------------------------------------------+
```

and message texts corresponding to newly added indices:

```
//--- CSeriesDataInd
   {"The method is not intended for working with indicator programs"},
   {"Failed to get indicator data timeseries"},
   {"Failed to get the current data of the indicator buffer"},
   {"Failed to create indicator data object"},
   {"Failed to add indicator data object to the list"},

//--- CTick
   {"Tick"},
   {"Last price update time in milliseconds"},
   {"Last price update time"},
   {"Volume for the current Last price"},
   {"Flags"},
   {"Volume for the current \"Last\" price with increased accuracy"},
   {"Spread"},
   {"Changed data on a tick:"},
   {"Bid price change"},
   {"Ask price change"},
   {"Last price change"},
   {"Volume change"},

  };
//+---------------------------------------------------------------------+
```

In the \\MQL5\\Include\\DoEasy\ **Defines.mqh** file add enumerations for specifying integer, real and string properties of tick data object:

```
//+------------------------------------------------------------------+
//| Data for working with tick data                                  |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Tick integer properties                                          |
//+------------------------------------------------------------------+
enum ENUM_TICK_PROP_INTEGER
  {
   TICK_PROP_TIME_MSC = 0,                                  // Time of the last price update in milliseconds
   TICK_PROP_TIME,                                          // Time of the last update
   TICK_PROP_VOLUME,                                        // Volume for the current Last price
   TICK_PROP_FLAGS,                                         // Tick flags
  };
#define TICK_PROP_INTEGER_TOTAL (4)                         // Total number of tick integer properties
#define TICK_PROP_INTEGER_SKIP  (0)                         // Number of tick properties not used in sorting
//+------------------------------------------------------------------+
//| Real tick properties                                             |
//+------------------------------------------------------------------+
enum ENUM_TICK_PROP_DOUBLE
  {
   TICK_PROP_BID = TICK_PROP_INTEGER_TOTAL,                 // Tick Bid price
   TICK_PROP_ASK,                                           // Tick Ask price
   TICK_PROP_LAST,                                          // Current price of the last trade (Last)
   TICK_PROP_VOLUME_REAL,                                   // Volume for the current Last price with greater accuracy
   TICK_PROP_SPREAD,                                        // Tick spread (Ask - Bid)
  };
#define TICK_PROP_DOUBLE_TOTAL  (5)                         // Total number of real tick properties
#define TICK_PROP_DOUBLE_SKIP   (0)                         // Number of tick properties not used in sorting
//+------------------------------------------------------------------+
//| String tick properties                                           |
//+------------------------------------------------------------------+
enum ENUM_TICK_PROP_STRING
  {
   TICK_PROP_SYMBOL = (TICK_PROP_INTEGER_TOTAL+TICK_PROP_DOUBLE_TOTAL), // Tick symbol
  };
#define TICK_PROP_STRING_TOTAL  (1)                         // Total number of string tick properties
//+------------------------------------------------------------------+
```

For each of the enumerations specify the total number of the properties used and not used in sorting.

For enabling search and sorting by the above specified properties add yet another enumeration where possible criteria of tick data sorting will be specified:

```
//+------------------------------------------------------------------+
//| Possible tick sorting criteria                                   |
//+------------------------------------------------------------------+
#define FIRST_TICK_DBL_PROP          (TICK_PROP_INTEGER_TOTAL-TICK_PROP_INTEGER_SKIP)
#define FIRST_TICK_STR_PROP          (TICK_PROP_INTEGER_TOTAL-TICK_PROP_INTEGER_SKIP+TICK_PROP_DOUBLE_TOTAL-TICK_PROP_DOUBLE_SKIP)
enum ENUM_SORT_TICK_MODE
  {
//--- Sort by integer properties
   SORT_BY_TICK_TIME_MSC = 0,                               // Sort by the time of the last price update in milliseconds
   SORT_BY_TICK_TIM,                                        // Sort by the time of the last price update
   SORT_BY_TICK_VOLUME,                                     // Sort by volume for the current Last price
   SORT_BY_TICK_FLAGS,                                      // Sort by tick flags
//--- Sort by real properties
   SORT_BY_TICK_BID = FIRST_TICK_DBL_PROP,                  // Sort by tick Bid price
   SORT_BY_TICK_ASK,                                        // Sort by tick Ask price
   SORT_BY_TICK_LAST,                                       // Sort by current price of the last trade (Last)
   SORT_BY_TICK_VOLUME_REAL,                                // Sort by volume for the current Last price with greater accuracy
   SORT_BY_TICK_SPREAD,                                     // Sort by tick spread
//--- Sort by string properties
   SORT_BY_TICK_SYMBOL = FIRST_TICK_STR_PROP,               // Sort by tick symbol
  };
//+------------------------------------------------------------------+
```

Earlier, we already have created [“New tick” object in article 38](https://www.mql5.com/en/articles/7695#node03) which enables to track the arrival of a new tick on the specified symbol.

The object is in \\MQL5\\Include\\DoEasy\\Objects\ **Ticks\** folder of library directory in file **NewTickObj.mqh**.

Improve object class so that NULL  value (or empty string) can be passed to the method of symbol setting for specifying the current symbol for the object:

```
//--- Set a symbol
   void              SetSymbol(const string symbol)   { this.m_symbol=(symbol==NULL || symbol=="" ? ::Symbol() : symbol);  }
```

And in the class constructor's initialization list set false value for **m\_new\_tick** variable which stores new tick flag:

```
//+------------------------------------------------------------------+
//| Parametric constructor CNewTickObj                               |
//+------------------------------------------------------------------+
CNewTickObj::CNewTickObj(const string symbol) : m_symbol(symbol),m_new_tick(false)
  {
//--- Reset the structures of the new and previous ticks
   ::ZeroMemory(this.m_tick);
   ::ZeroMemory(this.m_tick_prev);
//--- If managed to get the current prices to the tick structure -
//--- copy data of the obtained tick to the previous tick data and reset the first launch flag
  if(::SymbolInfoTick(this.m_symbol,this.m_tick))
     {
      this.m_tick_prev=this.m_tick;
      this.m_first_start=false;
     }
  }
//+------------------------------------------------------------------+
```

Before that, this variable was not initialized anywhere by the initial value. This is not correct.

### Tick data object class

In the class location folder of “New tick” object \\MQL5\\Include\\DoEasy\\Objects\ **Ticks\** create a new file of tick data object class **DataTick.mqh**.

The object will not be something new for library classes. Everything is standard. Let’s analyze the class body:

```
//+------------------------------------------------------------------+
//|                                                     DataTick.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/ru/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/ru/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\BaseObj.mqh"
#include "..\..\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| “Tick” class                                                     |
//+------------------------------------------------------------------+
class CDataTick : public CBaseObj
  {
private:
   MqlTick           m_tick;                                      // Structure for obtaining current prices
   int               m_digits;                                    // Symbol's digits value
   long              m_long_prop[TICK_PROP_INTEGER_TOTAL];        // Integer properties
   double            m_double_prop[TICK_PROP_DOUBLE_TOTAL];       // Real properties
   string            m_string_prop[TICK_PROP_STRING_TOTAL];       // String properties

//--- Return the index of the array the tick’s (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_TICK_PROP_DOUBLE property)       const { return(int)property-TICK_PROP_INTEGER_TOTAL;                       }
   int               IndexProp(ENUM_TICK_PROP_STRING property)       const { return(int)property-TICK_PROP_INTEGER_TOTAL-TICK_PROP_DOUBLE_TOTAL;}

public:
//--- Set tick’s (1) integer, (2) real and (3) string property
   void              SetProperty(ENUM_TICK_PROP_INTEGER property,long value)  { this.m_long_prop[property]=value;                               }
   void              SetProperty(ENUM_TICK_PROP_DOUBLE property,double value) { this.m_double_prop[this.IndexProp(property)]=value;             }
   void              SetProperty(ENUM_TICK_PROP_STRING property,string value) { this.m_string_prop[this.IndexProp(property)]=value;             }
//--- Return tick’s (1) integer, (2) real and (3) string property from the properties array
   long              GetProperty(ENUM_TICK_PROP_INTEGER property)    const { return this.m_long_prop[property];                                 }
   double            GetProperty(ENUM_TICK_PROP_DOUBLE property)     const { return this.m_double_prop[this.IndexProp(property)];               }
   string            GetProperty(ENUM_TICK_PROP_STRING property)     const { return this.m_string_prop[this.IndexProp(property)];               }

//--- Return the flag of the tick supporting this property
   virtual bool      SupportProperty(ENUM_TICK_PROP_INTEGER property)      { return true; }
   virtual bool      SupportProperty(ENUM_TICK_PROP_DOUBLE property)       { return true; }
   virtual bool      SupportProperty(ENUM_TICK_PROP_STRING property)       { return true; }
//--- Return itself
   CDataTick        *GetObject(void)                                       { return &this;}

//--- Compare CDataTick objects with each other by the specified property (for sorting the lists by a specified object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CDataTick objects with each other by all properties (to search equal objects)
   bool              IsEqual(CDataTick* compared_obj) const;
//--- Constructors
                     CDataTick(){;}
                     CDataTick(const string symbol,const MqlTick &tick);

//+------------------------------------------------------------------+
//| Descriptions of object tick data properties                      |
//+------------------------------------------------------------------+
//--- Return description of tick's (1) integer, (2) real and (3) string property
   string            GetPropertyDescription(ENUM_TICK_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_TICK_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_TICK_PROP_STRING property);

//--- Display the description of tick properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of the tick in the journal
   virtual void      PrintShort(void);
//---  Return the (1) short name and (2) description of tick data object flags
   virtual string    Header(void);
   string            FlagsDescription(void);

//+------------------------------------------------------------------+
//| Methods of simplified access to tick data object properties      |
//+------------------------------------------------------------------+
//--- Return tick’s (1) Time, (2) time in milliseconds, (3) volume, (4) flags
   datetime          Time(void)                                         const { return (datetime)this.GetProperty(TICK_PROP_TIME);           }
   long              TimeMSC(void)                                      const { return this.GetProperty(TICK_PROP_TIME_MSC);                 }
   long              Volume(void)                                       const { return this.GetProperty(TICK_PROP_VOLUME);                   }
   uint              Flags(void)                                        const { return (uint)this.GetProperty(TICK_PROP_FLAGS);              }

//--- Return tick’s (1) Bid, (2) Ask, (3) Last price, (4) volume with greater accuracy, (5) spread of the tick
//--- size of the (9) candle upper, (10) lower wick
   double            Bid(void)                                          const { return this.GetProperty(TICK_PROP_BID);                      }
   double            Ask(void)                                          const { return this.GetProperty(TICK_PROP_ASK);                      }
   double            Last(void)                                         const { return this.GetProperty(TICK_PROP_LAST);                     }
   double            VolumeReal(void)                                   const { return this.GetProperty(TICK_PROP_VOLUME_REAL);              }
   double            Spread(void)                                       const { return this.GetProperty(TICK_PROP_SPREAD);                   }

//--- Return tick symbol
   string            Symbol(void)                                       const { return this.GetProperty(TICK_PROP_SYMBOL);                   }

//--- Return bar (1) time, (2) index on the specified timeframe the tick time falls into
   datetime          TimeBar(const ENUM_TIMEFRAMES timeframe)const
                       { return ::iTime(this.Symbol(),timeframe,this.Index(timeframe));                                                      }
   int               Index(const ENUM_TIMEFRAMES timeframe)  const
                       { return ::iBarShift(this.Symbol(),(timeframe==PERIOD_CURRENT ? ::Period() : timeframe),this.Time());                 }

//--- Return the flag of (1) Bid, (2) Ask, (3) Last price, (4) volume change; (5) buy and (6) sell trades
   bool              IsChangeBid()                                      const { return((this.Flags() & TICK_FLAG_BID)==TICK_FLAG_BID);       }
   bool              IsChangeAsk()                                      const { return((this.Flags() & TICK_FLAG_ASK)==TICK_FLAG_ASK);       }
   bool              IsChangeLast()                                     const { return((this.Flags() & TICK_FLAG_LAST)==TICK_FLAG_LAST);     }
   bool              IsChangeVolume()                                   const { return((this.Flags() & TICK_FLAG_VOLUME)==TICK_FLAG_VOLUME); }
   bool              IsChangeBuy()                                      const { return((this.Flags() & TICK_FLAG_BUY)==TICK_FLAG_BUY);       }
   bool              IsChangeSell()                                     const { return((this.Flags( )& TICK_FLAG_SELL)==TICK_FLAG_SELL);     }
//---
  };
//+------------------------------------------------------------------+
```

The composition of library objects and their creation was discussed in detail in [the very first article](https://www.mql5.com/en/articles/6595). Now, let’s look through the main variables and methods, and further analyze their implementation in class.

Private class section contains auxiliary variables, arrays for storing values of object integer, real and string properties; as well as the methods returning actual property indices at which they are physically located in arrays.

Public class section contains methods for setting and return of the values of specified properties to corresponding object property arrays; methods returning the flags of object supporting one or another property; methods for searching, comparing and sorting the objects, as well as class constructors.

Methods for displaying object property descriptions and methods for simplified access to object properties are located there, too.

**Now, let's have a look at the implementation of the class methods.**

**The symboland filled structure with the current tick data shall be passed to parametric class constructor:**

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CDataTick::CDataTick(const string symbol,const MqlTick &tick)
  {
//--- Save symbol’s Digits
   this.m_digits=(int)::SymbolInfoInteger(symbol,SYMBOL_DIGITS);
//--- Save integer tick properties
   this.SetProperty(TICK_PROP_TIME,tick.time);
   this.SetProperty(TICK_PROP_TIME_MSC,tick.time_msc);
   this.SetProperty(TICK_PROP_VOLUME,tick.volume);
   this.SetProperty(TICK_PROP_FLAGS,tick.flags);
//--- Save real tick properties
   this.SetProperty(TICK_PROP_BID,tick.bid);
   this.SetProperty(TICK_PROP_ASK,tick.ask);
   this.SetProperty(TICK_PROP_LAST,tick.last);
   this.SetProperty(TICK_PROP_VOLUME_REAL,tick.volume_real);
//--- Save additional tick properties
   this.SetProperty(TICK_PROP_SPREAD,tick.ask-tick.bid);
   this.SetProperty(TICK_PROP_SYMBOL,(symbol==NULL || symbol=="" ? ::Symbol() : symbol));
  }
//+------------------------------------------------------------------+
```

Within the constructor, simply fill in object properties with corresponding values from tick structure and the symbol from which tick data are obtained.

**The method of comparing two parameters of objects by the specified property:**

```
//+----------------------------------------------------------------------+
//| Compare CDataTick objects with each other by the specified property  |
//+----------------------------------------------------------------------+
int CDataTick::Compare(const CObject *node,const int mode=0) const
  {
   const CDataTick *obj_compared=node;
//--- compare integer properties of two objects
   if(mode<TICK_PROP_INTEGER_TOTAL)
     {
      long value_compared=obj_compared.GetProperty((ENUM_TICK_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_TICK_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two objects
   else if(mode<TICK_PROP_DOUBLE_TOTAL+TICK_PROP_INTEGER_TOTAL)
     {
      double value_compared=obj_compared.GetProperty((ENUM_TICK_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_TICK_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<TICK_PROP_DOUBLE_TOTAL+TICK_PROP_INTEGER_TOTAL+TICK_PROP_STRING_TOTAL)
     {
      string value_compared=obj_compared.GetProperty((ENUM_TICK_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_TICK_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

The pointer to the object with which the current object must be compared and the type of the property by which two objects will be compared, shall be passed to the method.

Depending on the passed mode of object comparison, compare these properties of the two objects and return 1/-1/0 if the current object’s value of its property is more/less or equal to the value of property of the object compared, respectively.

**Method for comparison of two objects for identity:**

```
//+------------------------------------------------------------------+
//| Compare CDataTick objects with each other by all properties      |
//+------------------------------------------------------------------+
bool CDataTick::IsEqual(CDataTick *compared_obj) const
  {
   int beg=0, end=TICK_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_TICK_PROP_INTEGER prop=(ENUM_TICK_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=TICK_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_TICK_PROP_DOUBLE prop=(ENUM_TICK_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=TICK_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_TICK_PROP_STRING prop=(ENUM_TICK_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Here: the pointer to the object for comparison is passed to the method. Then, in three loops by each property group compare each successive property of the two objects. If at least one of the properties is not equal to the same property of the object compared return false. Upon completion of all three loops true returns — all properties of the two objects are equal. This means that the objects are identical.

**The method for displaying all object properties in the journal:**

```
//+------------------------------------------------------------------+
//| Display tick properties in the journal                           |
//+------------------------------------------------------------------+
void CDataTick::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG)," (",this.Header(),") =============");
   int beg=0, end=TICK_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_TICK_PROP_INTEGER prop=(ENUM_TICK_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=TICK_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_TICK_PROP_DOUBLE prop=(ENUM_TICK_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=TICK_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_TICK_PROP_STRING prop=(ENUM_TICK_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_END)," (",this.Header(),") =============\n");
  }
//+------------------------------------------------------------------+
```

Here: in three loops by each of object property groups the description of each successive property is displayed in the journal using corresponding method GetPropertyDescription(). If in method parameters false value is passed in **full\_prop** variable, only the properties supported by the object are displayed. For an unsupported property, the journal displays that the property is not supported, although all properties are supported in this object. But this can be changed in class descendant objects.

**Methods returning the description of the specified object’s integer, real and string property:**

```
//+------------------------------------------------------------------+
//| Return description of tick's integer property                    |
//+------------------------------------------------------------------+
string CDataTick::GetPropertyDescription(ENUM_TICK_PROP_INTEGER property)
  {
   return
     (
      property==TICK_PROP_TIME_MSC           ?  CMessage::Text(MSG_TICK_TIME_MSC)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+TimeMSCtoString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==TICK_PROP_TIME               ?  CMessage::Text(MSG_TICK_TIME)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==TICK_PROP_VOLUME             ?  CMessage::Text(MSG_TICK_VOLUME)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==TICK_PROP_FLAGS              ?  CMessage::Text(MSG_TICK_FLAGS)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)+"\n"+CMessage::Text(MSG_LIB_TEXT_TICK_CHANGED_DATA)+this.FlagsDescription()
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of tick's real property                       |
//+------------------------------------------------------------------+
string CDataTick::GetPropertyDescription(ENUM_TICK_PROP_DOUBLE property)
  {
   int dg=(this.m_digits>0 ? this.m_digits : 1);
   return
     (
      property==TICK_PROP_BID                ?  CMessage::Text(MSG_LIB_PROP_BID)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==TICK_PROP_ASK                ?  CMessage::Text(MSG_LIB_PROP_ASK)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==TICK_PROP_LAST               ?  CMessage::Text(MSG_LIB_PROP_LAST)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==TICK_PROP_VOLUME_REAL        ?  CMessage::Text(MSG_TICK_VOLUME_REAL)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),2)
         )  :
      property==TICK_PROP_SPREAD             ?  CMessage::Text(MSG_TICK_SPREAD)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of tick's string property                     |
//+------------------------------------------------------------------+
string CDataTick::GetPropertyDescription(ENUM_TICK_PROP_STRING property)
  {
   return(property==TICK_PROP_SYMBOL ? CMessage::Text(MSG_LIB_PROP_SYMBOL)+": \""+this.GetProperty(property)+"\"" : "");
  }
//+------------------------------------------------------------------+
```

Here: depending on the property passed to the method its string description is returned.

**Method returning the string with the description of all flags of the tick:**

```
//+------------------------------------------------------------------+
//| Display the description of flags                                 |
//+------------------------------------------------------------------+
string CDataTick::FlagsDescription(void)
  {
   string flags=
     (
      (this.IsChangeAsk()     ? "\n - "+CMessage::Text(MSG_LIB_TEXT_TICK_FLAG_ASK)     : "")+
      (this.IsChangeBid()     ? "\n - "+CMessage::Text(MSG_LIB_TEXT_TICK_FLAG_BID)     : "")+
      (this.IsChangeLast()    ? "\n - "+CMessage::Text(MSG_LIB_TEXT_TICK_FLAG_LAST)    : "")+
      (this.IsChangeVolume()  ? "\n - "+CMessage::Text(MSG_LIB_TEXT_TICK_FLAG_VOLUME)  : "")+
      (this.IsChangeBuy()     ? "\n - "+CMessage::Text(MSG_DEAL_TO_BUY)                : "")+
      (this.IsChangeSell()    ? "\n - "+CMessage::Text(MSG_DEAL_TO_SELL)               : "")
     );
  return flags;
  }
//+------------------------------------------------------------------+
```

In its properties each tick has a variable which contains a set of flags. These flags describe the events which incurred this tick:

**From [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick) user-guide**

To find out which specific data have changed with the current tick make analysis of its flags:

- TICK\_FLAG\_BID — tick changed Bid price
- TICK\_FLAG\_ASK — tick changed Ask price
- TICK\_FLAG\_LAST — tick changed Last trade price
- TICK\_FLAG\_VOLUME — tick changed volume
- TICK\_FLAG\_BUY — tick resulted from Buy trade
- TICK\_FLAG\_SELL — tick resulted from Sell trade

We have six methods (by the number of flags) returning the presence flag within the variable of each of the flags:

```
//--- Return the flag of (1) Bid, (2) Ask, (3) Last price, (4) volume change; (5) buy and (6) sell trades
   bool IsChangeBid()    const { return((this.Flags() & TICK_FLAG_BID)==TICK_FLAG_BID);       }
   bool IsChangeAsk()    const { return((this.Flags() & TICK_FLAG_ASK)==TICK_FLAG_ASK);       }
   bool IsChangeLast()   const { return((this.Flags() & TICK_FLAG_LAST)==TICK_FLAG_LAST);     }
   bool IsChangeVolume() const { return((this.Flags() & TICK_FLAG_VOLUME)==TICK_FLAG_VOLUME); }
   bool IsChangeBuy()    const { return((this.Flags() & TICK_FLAG_BUY)==TICK_FLAG_BUY);       }
   bool IsChangeSell()   const { return((this.Flags() & TICK_FLAG_SELL)==TICK_FLAG_SELL);     }
```

In the method described, first the flag of flag presence within the variable is checked. Depending on whether such change is available or not the resulted text string shall be added with newline character + flag description or an empty string. After checking all flags, finally we have the compiled string which contains descriptions of the flags present in the variable. If several flags are available, the description of each flag will start from a new line in the journal.

**Method displaying the short description of tick data object in the journal:**

```
//+------------------------------------------------------------------+
//| Display a short description of the tick in the journal           |
//+------------------------------------------------------------------+
void CDataTick::PrintShort(void)
  {
   ::Print(this.Header());
  }
//+------------------------------------------------------------------+
```

The method simply displays in the journal **a short name** of itself which is returned by the following method:

```
//+------------------------------------------------------------------+
//| Return a short name of tick data object                          |
//+------------------------------------------------------------------+
string CDataTick::Header(void)
  {
   return
     (
      CMessage::Text(MSG_TICK_TEXT_TICK)+" \""+this.Symbol()+"\" "+TimeMSCtoString(TimeMSC())
     );
  }
//+------------------------------------------------------------------+
```

**At this phase, creation of tick data object is finished.**

### Testing tick data object

To perform the test, let's use [the EA from the previous article](https://www.mql5.com/en/articles/8787#node04) and save it in the new folder \\MQL5\\Experts\\TestDoEasy\ **Part59\** under the new name **TestDoEasyPart59.mq5**.

In this EA, indicators and their data with their timeseries are not necessary any more. Here, simply create tick data object, display its full description in the journal and the short description in the comment on the chart. With each new tick, its description on the chart will change; while in the journal the first tick which came after EA launch will be displayed.

Since there is no connection of EA with this new library object yet, I will connect the file of its class to the EA:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart59.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/ru/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/ru/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Ticks\DataTick.mqh>
```

In the area of EA’s global variables instead of custom indicator parameter arrays

```
//--- Arrays of custom indicator parameters
MqlParam       param_ma1[];
MqlParam       param_ma2[];
//+------------------------------------------------------------------+
```

declare an object of “New tick” class — it will be required to simulate work within future collection classes of library tick data:

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

//--- "New tick" object
CNewTickObj    check_tick;
//+------------------------------------------------------------------+
```

In **OnInit()** handler remove the indicator creation block:

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

   //--- Create indicator 3
   engine.GetIndicatorsCollection().CreateAMA(NULL,PERIOD_CURRENT,AMA1);
   //--- Create indicator 4
   engine.GetIndicatorsCollection().CreateAMA(NULL,PERIOD_CURRENT,AMA2,14);

   //--- Display descriptions of created indicators
   engine.GetIndicatorsCollection().Print();
   engine.GetIndicatorsCollection().PrintShort();
```

In the very end of **OnInit()** set for “New tick” object the current symbol as the working one:

```
//--- Set the current symbol for "New tick" object
   check_tick.SetSymbol(Symbol());
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

In **OnTick()** handler of EA add a code block for determining a new tick (as simulation of work within the future collection class of ticks) and for creating a new tick data object with the display of its description on the chart and in the journal:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Handle the NewTick event in the library
   engine.OnTick(rates_data);

//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Working in the timer
      PressButtonsControl();        // Button pressing control
      engine.EventsHandling();      // Working with events
     }

//--- Create a temporary list for storing “Tick data” objects,
//--- a variable for obtaining tick data and
//--- a variable for calculating incoming ticks
   static int tick_count=0;
   CArrayObj list;
   MqlTick tick_struct;
//--- Check a new tick on the current symbol
   if(check_tick.IsNewTick())
     {
      //--- If failed to get the price - exit
      if(!SymbolInfoTick(Symbol(),tick_struct))
         return;
      //--- Create a new tick data object
      CDataTick *tick_obj=new CDataTick(Symbol(),tick_struct);
      if(tick_obj==NULL)
         return;
      //--- Increase tick counter (simply to display on the screen, no other purpose is provided)
      tick_count++;
      //--- Limit the number of ticks in the counting as one hundred thousand (again, no purpose is provided)
      if(tick_count>100000) tick_count=1;
      //--- In the comment on the chart display the tick number and its short description
      Comment("--- №",IntegerToString(tick_count,5,'0'),": ",tick_obj.Header());
      //--- If this is the first tick (which follows the first launch of EA) display its full description in the journal
      if(tick_count==1)
         tick_obj.Print();
      //--- Remove if failed to put the created tick data object in the list
      if(!list.Add(tick_obj))
         delete tick_obj;
     }

//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();          // Trailing positions
      TrailingOrders();             // Trailing of pending orders
     }
  }
//+------------------------------------------------------------------+
```

Logic is specified in detail in comments in the listing; I believe it is clear and simple.

Compile the EA and launch it on the chart having preliminary set in settings to use the current symbol and timeframe. After the launching and arrival of the new tick, the description of tick data object (arrived tick) will be displayed in the journal:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10426.13 USD, 1:100, Hedge, Demo account MetaTrader 5
--- Initialize "DoEasy" library ---
Work with the current symbol only: "EURUSD"
Work with the current timeframe only: H1
EURUSD symbol timeseries:
- "EURUSD" H1 timeseries: Requested: 1000, Actually: 1000, Created: 1000, On the server: 5153
Library initialize time: 00:00:00.000
============= Beginning of parameter list (Tick "EURUSD" 2020.12.16 13:22:32.822) =============
Last price update time in milliseconds: 2020.12.16 13:22:32.822
Last price update time: 2020.12.16 13:22:32
Volume for the current Last price: 0
Flags: 6
Changed data on the tick:
 - Ask price change
 - Bid price change
------
Bid price: 1.21927
Ask price: 1.21929
Last price: 0.00000
Volume for the current Last price with greater accuracy: 0.00
Spread: 0.00002
------
Symbol: "EURUSD"
============= End of parameter list (Tick "EURUSD" 2020.12.16 13:22:32.822) =============
```

and with each new tick a comment with its short description will be displayed on the chart:

![](https://c.mql5.com/2/41/terminal64_4LnVWfQomY.png)

### What's next?

In the next article we will start creating the tick data collection for one symbol.

All files of the current library version are attached below together with the test EA file for MQL5. You can download them and test everything.

Leave your comments, questions and suggestions in the comments to the article.

[Back to contents](https://www.mql5.com/en/articles/8818#node00)

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

[Timeseries in DoEasy library (part 58): Timeseries of indicator buffer data](https://www.mql5.com/en/articles/8787)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8818](https://www.mql5.com/ru/articles/8818)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8818.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8818/mql5.zip "Download MQL5.zip")(3872.72 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/361846)**
(1)


![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
9 Jan 2021 at 18:44

It remains to be added, when will the clonal war go ?

![Neural networks made easy (Part 8): Attention mechanisms](https://c.mql5.com/2/48/Neural_networks_made_easy_008.png)[Neural networks made easy (Part 8): Attention mechanisms](https://www.mql5.com/en/articles/8765)

In previous articles, we have already tested various options for organizing neural networks. We also considered convolutional networks borrowed from image processing algorithms. In this article, I suggest considering Attention Mechanisms, the appearance of which gave impetus to the development of language models.

![Using spreadsheets to build trading strategies](https://c.mql5.com/2/41/spread_sheets_strategy.png)[Using spreadsheets to build trading strategies](https://www.mql5.com/en/articles/8699)

The article describes the basic principles and methods that allow you to analyze any strategy using spreadsheets (Excel, Calc, Google). The obtained results are compared with MetaTrader 5 tester.

![Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://c.mql5.com/2/41/50_percents__1.png)[Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://www.mql5.com/en/articles/8616)

In the upcoming series of articles, I will demonstrate the development of self-adapting algorithms considering most market factors, as well as show how to systematize these situations, describe them in logic and take them into account in your trading activity. I will start with a very simple algorithm that will gradually acquire theory and evolve into a very complex project.

![Timeseries in DoEasy library (part 58): Timeseries of indicator buffer data](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__2.png)[Timeseries in DoEasy library (part 58): Timeseries of indicator buffer data](https://www.mql5.com/en/articles/8787)

In conclusion of the topic of working with timeseries organise storage, search and sort of data stored in indicator buffers which will allow to further perform the analysis based on values of the indicators to be created on the library basis in programs. The general concept of all collection classes of the library allows to easily find necessary data in the corresponding collection. Respectively, the same will be possible in the class created today.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/8818&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083157038558615065)

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