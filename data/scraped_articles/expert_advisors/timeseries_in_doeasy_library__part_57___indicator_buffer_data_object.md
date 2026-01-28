---
title: Timeseries in DoEasy library (part 57): Indicator buffer data object
url: https://www.mql5.com/en/articles/8705
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:28:34.386198
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/8705&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071859797101326288)

MetaTrader 5 / Examples


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8705#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8705#node02)
- [Indicator buffer data object](https://www.mql5.com/en/articles/8705#node03)
- [Testing](https://www.mql5.com/en/articles/8705#node04)
- [What's next?](https://www.mql5.com/en/articles/8705#node05)


### Concept

All concept of data construction and storage in the library is based on collection lists which contain a set of same-type data. They can be selected, sorted and filtered in the required sequence. This enables to get necessary sets of data for their further comparison and analysis. By their structure, indicator buffers do not differ greatly from earlier created serial objects also and they can be contained in their collection lists where we will be able to quickly find necessary objects with data. But to be able to create a collection you must have a data object for that collection. The full set of those data will be stored in collection list.

Today, I will create a data object which contains all information about the indicator buffer on one bar and, respectively, it contains information about the indicator to which buffer belongs, data of one bar of which are described by the object being created.

For each separate buffer of one indicator and for each bar of timeseries own data object will be created and organized into the collection list which belongs to the symbol and timeframe of that indicator. Thus, for one copy of indicator I will have a data set for each bar of timeseries of each buffer of that indicator.

At the moment, such concept of data storage seems excessive because you always can get necessary data of the required bar of indicator buffer timeseries using a direct request from indicator. But further, when collections of those data are created (in the following articles) we will be able to quickly find necessary data of any indicators for which collections will be created; as well, to perform benchmarking analysis. For such situation storage of data in collection cache seems appropriate.

### Improving library classes

First, as usual, add the new library messages.

In file \\MQL5\\Include\\DoEasy\ **Data.mqh** add new message indices:

```
//--- CIndicatorsCollection
   MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST,                // Error. Failed to add indicator object to the list
   MSG_LIB_SYS_INVALID_IND_POINTER,                   // Error. Invalid pointer to indicator object is passed
   MSG_LIB_SYS_IND_ID_EXIST,                          // Error. Indicator object with ID already exists

//--- CDataInd
   MSG_LIB_TEXT_IND_DATA_IND_BUFFER_NUM,              // Indicator buffer number
   MSG_LIB_TEXT_IND_DATA_BUFFER_VALUE,                // Indicator buffer value

  };
//+------------------------------------------------------------------+
```

And further - message texts corresponding to newly added indices:

```
   {"Error. Failed to add indicator object to list"},
   {"Error. Invalid pointer to indicator object passed"},
   {"Error. There is already exist an indicator object with ID"},

   {"Indicator buffer number"},
   {"Indicator buffer value"},

  };
//+---------------------------------------------------------------------+
```

Since indicator buffer data object will be stored in the collection list, in order to search and sort this object must receive all properties inherent to other library objects which are stored in lists.

In file \\MQL5\\Include\\DoEasy\ **Defines.mqh** describe all necessary properties of a new object — **object integer properties:**

```
//+------------------------------------------------------------------+
//|  Data for working with indicator data                            |
//+------------------------------------------------------------------+
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
  };
#define IND_DATA_PROP_INTEGER_TOTAL (5)                     // Total number of indicator data integer properties
#define IND_DATA_PROP_INTEGER_SKIP  (0)                     // Number of indicator data properties not used in sorting
//+------------------------------------------------------------------+
```

- Sort by time is the main sort type where all data will be organized in the order of their sequence to indicator buffer data in terminal.

- Timeframe value is included to integer properties so that further, values of those two indicator buffers could be compared on different timeframes.

- Indicator type will contain type value from enumeration [ENUM\_INDICATOR](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_indicator).

- Indicator buffer number - sequence number from zero and further according to the number of indicator buffers.

- Indicator ID - by this property data of the necessary indicator can be found for which ID was set in the program. We discussed it [in the previous article](https://www.mql5.com/en/articles/8646).


**Object real properties:**

```
//+------------------------------------------------------------------+
//| Indicator data real properties                                   |
//+------------------------------------------------------------------+
enum ENUM_IND_DATA_PROP_DOUBLE
  {
//--- bar data
   IND_DATA_PROP_BUFFER_VALUE = IND_DATA_PROP_INTEGER_TOTAL,// Indicator data value
  };
#define IND_DATA_PROP_DOUBLE_TOTAL  (1)                     // Total number of indicator data real properties
#define IND_DATA_PROP_DOUBLE_SKIP   (0)                     // Number of indicator data properties not used in sorting
//+------------------------------------------------------------------+
```

Here, only one property is available - a value in indicator buffer which corresponds to the bar for which indicator data object is created.

**Object string properties:**

```
//+------------------------------------------------------------------+
//| Indicator data string properties                                 |
//+------------------------------------------------------------------+
enum ENUM_IND_DATA_PROP_STRING
  {
   IND_DATA_PROP_SYMBOL = (IND_DATA_PROP_INTEGER_TOTAL+IND_DATA_PROP_DOUBLE_TOTAL), // Indicator data symbol
   IND_DATA_PROP_IND_NAME,                                  // Indicator name
   IND_DATA_PROP_IND_SHORTNAME,                             // Indicator short name
  };
#define IND_DATA_PROP_STRING_TOTAL  (3)                     // Total number of string properties of indicator data
//+------------------------------------------------------------------+
```

By values of these properties it will be possible to select and sort collection data by the symbol on which the indicator is calculated, as well as by indicator name and short name.

Now, add all created properties of the object to enumeration of possible sort criteria:

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
//--- Sort by real properties
   SORT_BY_IND_DATA_BUFFER_VALUE = FIRST_IND_DATA_DBL_PROP, // Sort by indicator data value
//--- Sort by string properties
   SORT_BY_IND_DATA_SYMBOL = FIRST_IND_DATA_STR_PROP,       // Sort by indicator data symbol
   SORT_BY_IND_DATA_IND_NAME,                               // Sort by indicator name
   SORT_BY_IND_DATA_IND_SHORTNAME,                          // Sort by indicator short name
  };
//+------------------------------------------------------------------+
```

### Indicator buffer data object

Object properties are ready. Now, create a new object which will store data of one buffer for one indicator.

In library folder  \\MQL5\\Include\ **DoEasy** in folder \\Objects\ **Indicators\** create a new class CDataInd in file **DataInd.mqh**:

The class is derived from the base object of all objects of CBaseObj library.

In fact, the object contains fields and methods standard for library objects and is identical to the bar object analyzed in [article 35](https://www.mql5.com/en/articles/7594#node02), but in contrast with the bar object it has less properties (all properties of object- indicator buffer data were described in enumerations above).

Let’s analyze the class body of indicator buffer data object:

```
//+------------------------------------------------------------------+
//|                                                      DataInd.mqh |
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
#include "..\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Indicator data class                                             |
//+------------------------------------------------------------------+
class CDataInd : public CBaseObj
  {
private:
   int               m_digits;                                    // Digits value of indicator data
   int               m_index;                                     // Bar index
   string            m_period_description;                        // Timeframe string description
   long              m_long_prop[IND_DATA_PROP_INTEGER_TOTAL];    // Integer properties
   double            m_double_prop[IND_DATA_PROP_DOUBLE_TOTAL];   // Real properties
   string            m_string_prop[IND_DATA_PROP_STRING_TOTAL];   // String properties

//--- Return the index of the array the object's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_IND_DATA_PROP_DOUBLE property)   const { return(int)property-IND_DATA_PROP_INTEGER_TOTAL;                            }
   int               IndexProp(ENUM_IND_DATA_PROP_STRING property)   const { return(int)property-IND_DATA_PROP_INTEGER_TOTAL-IND_DATA_PROP_DOUBLE_TOTAL; }

public:
//--- Set (1) integer, (2) real and (3) string properties of indicator data
   void              SetProperty(ENUM_IND_DATA_PROP_INTEGER property,long value) { this.m_long_prop[property]=value;                               }
   void              SetProperty(ENUM_IND_DATA_PROP_DOUBLE property,double value){ this.m_double_prop[this.IndexProp(property)]=value;             }
   void              SetProperty(ENUM_IND_DATA_PROP_STRING property,string value){ this.m_string_prop[this.IndexProp(property)]=value;             }
//--- Return (1) integer, (2) real and (3) string property of indicator data from the properties array
   long              GetProperty(ENUM_IND_DATA_PROP_INTEGER property)   const { return this.m_long_prop[property];                                 }
   double            GetProperty(ENUM_IND_DATA_PROP_DOUBLE property)    const { return this.m_double_prop[this.IndexProp(property)];               }
   string            GetProperty(ENUM_IND_DATA_PROP_STRING property)    const { return this.m_string_prop[this.IndexProp(property)];               }

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_IND_DATA_PROP_INTEGER property)     { return true; }
   virtual bool      SupportProperty(ENUM_IND_DATA_PROP_DOUBLE property)      { return true; }
   virtual bool      SupportProperty(ENUM_IND_DATA_PROP_STRING property)      { return true; }
//--- Return itself
   CDataInd         *GetObject(void)                                          { return &this;}
//--- Set (1) symbol, timeframe and time for the object, (2) indicator type, (3) number of buffers, (4) number of data buffer,
//--- (5) ID, (6) data value, (7) name, (8) indicator short name
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time);
   void              SetIndicatorType(const ENUM_INDICATOR type)              { this.SetProperty(IND_DATA_PROP_INDICATOR_TYPE,type);               }
   void              SetBufferNum(const int num)                              { this.SetProperty(IND_DATA_PROP_IND_BUFFER_NUM,num);                }
   void              SetIndicatorID(const int id)                             { this.SetProperty(IND_DATA_PROP_IND_ID,id);                         }
   void              SetBufferValue(const double value)                       { this.SetProperty(IND_DATA_PROP_BUFFER_VALUE,value);                }
   void              SetIndicatorName(const string name)                      { this.SetProperty(IND_DATA_PROP_IND_NAME,name);                     }
   void              SetIndicatorShortname(const string shortname)            { this.SetProperty(IND_DATA_PROP_IND_SHORTNAME,shortname);           }

//--- Compare CDataInd objects with each other by all possible properties (for sorting the lists by a specified object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CDataInd objects with each other by all properties (to search equal objects)
   bool              IsEqual(CDataInd* compared_data) const;
//--- Constructors
                     CDataInd(){;}
                     CDataInd(const ENUM_INDICATOR ind_type,
                              const int ind_id,
                              const int buffer_num,
                              const string symbol,
                              const ENUM_TIMEFRAMES timeframe,
                              const datetime time);

//+------------------------------------------------------------------+
//| Methods of simplified access to object properties                |
//+------------------------------------------------------------------+
//--- Return (1) bar period start time, (2) timeframe, (3) indicator type, (4) number of buffers, (5) buffer number, (6) indicator ID
   datetime          Time(void)                                         const { return (datetime)this.GetProperty(IND_DATA_PROP_TIME);                   }
   ENUM_TIMEFRAMES   Timeframe(void)                                    const { return (ENUM_TIMEFRAMES)this.GetProperty(IND_DATA_PROP_PERIOD);          }
   ENUM_INDICATOR    IndicatorType(void)                                const { return (ENUM_INDICATOR)this.GetProperty(IND_DATA_PROP_INDICATOR_TYPE);   }
   int               BufferNum(void)                                    const { return (ENUM_INDICATOR)this.GetProperty(IND_DATA_PROP_IND_BUFFER_NUM);   }
   int               IndicatorID(void)                                  const { return (ENUM_INDICATOR)this.GetProperty(IND_DATA_PROP_IND_ID);           }

//--- Return the price of indicator buffer data
   double            PriceValue(void)                                   const { return this.GetProperty(IND_DATA_PROP_BUFFER_VALUE);                     }

//--- Return (1) data symbol, (2) name, (3) indicator short name
   string            Symbol(void)                                       const { return this.GetProperty(IND_DATA_PROP_SYMBOL);                           }
   string            IndicatorName(void)                                const { return this.GetProperty(IND_DATA_PROP_IND_NAME);                         }
   string            IndicatorShortName(void)                           const { return this.GetProperty(IND_DATA_PROP_IND_SHORTNAME);                    }
//--- Return bar index on the specified timeframe the object bar time falls into
   int               Index(const ENUM_TIMEFRAMES timeframe)  const
                       { return ::iBarShift(this.Symbol(),(timeframe==PERIOD_CURRENT ? ::Period() : timeframe),this.Time());                             }
//--- Return Digits set for the object
   int               Digits(void)                                       const { return this.m_digits;                                                    }
//+------------------------------------------------------------------+
//| Description of properties of the object - indicator data         |
//+------------------------------------------------------------------+
//--- Return description of object's (1) integer, (2) real and (3) string property
   string            GetPropertyDescription(ENUM_IND_DATA_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_IND_DATA_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_IND_DATA_PROP_STRING property);

//--- Return indicator type description
   string            IndicatorTypeDescription(void)                     const { return ::IndicatorTypeDescription(this.IndicatorType());                 }
//--- Display the description of object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of the object in the journal
   virtual void      PrintShort(void);
//---
  };
//+------------------------------------------------------------------+
```

Let’s briefly analyze class contents.

The private class section features:

Three arrays storing the corresponding object properties — integer, real and string ones.

The methods calculating the true index of the object property in the appropriate array.

Variables- class members to store values of additional object properties.

The public section of the class features:

The methods writing the passed object property value to the arrays of integer, real and string properties.

The methods returning the value of a requested  integer, real or string property from the arrays.

The virtual methods returning the flag of supporting the property by the object for each of the properties. The methods are meant to be implemented in descendant objects of the object and should return false in case the descendant object does not support the specified property. In the object of indicator buffer data all properties are supported and the methods return true.

We discussed the entire structure of the library objects [in the first article](https://www.mql5.com/en/articles/5654#node02). Here, we will briefly consider the implementation of the remaining class methods.

All additional methods for setting and returning object properties are for additional convenience only when writing a program; they just duplicate the actions of setting methods and receiving object properties — so that library user doesn’t have to remember names of constants from enumerations of object properties but, instead, to set and receive those properties being guided by the name of those additional methods.

**The Compare() virtual method** is meant for comparing two objects by the specified property. It is defined in the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectcompare) base object class of the standard library and should return zero if the values are equal and 1 or -1 if one of the values compared is respectively higher or lower. It is used in [Search()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) method of the standard library for searching and sorting. The method should be redefined in the descendant classes:

```
//+-------------------------------------------------------------------+
//| Compare CDataInd objects with each other by the specified property|
//+-------------------------------------------------------------------+
int CDataInd::Compare(const CObject *node,const int mode=0) const
  {
   const CDataInd *obj_compared=node;
//--- compare integer properties of two objects
   if(mode<IND_DATA_PROP_INTEGER_TOTAL)
     {
      long value_compared=obj_compared.GetProperty((ENUM_IND_DATA_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_IND_DATA_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two objects
   else if(mode<IND_DATA_PROP_DOUBLE_TOTAL+IND_DATA_PROP_INTEGER_TOTAL)
     {
      double value_compared=obj_compared.GetProperty((ENUM_IND_DATA_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_IND_DATA_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<IND_DATA_PROP_DOUBLE_TOTAL+IND_DATA_PROP_INTEGER_TOTAL+IND_DATA_PROP_STRING_TOTAL)
     {
      string value_compared=obj_compared.GetProperty((ENUM_IND_DATA_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_IND_DATA_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

**The method for determining two identical objects of indicator buffer data** serves to compare two data objects and returns trueonly if all fields of the two compared objects are equal:

```
//+------------------------------------------------------------------+
//|Compare CDataInd objects with each other by all properties        |
//+------------------------------------------------------------------+
bool CDataInd::IsEqual(CDataInd *compared_obj) const
  {
   int beg=0, end=BAR_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_IND_DATA_PROP_INTEGER prop=(ENUM_IND_DATA_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=IND_DATA_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_IND_DATA_PROP_DOUBLE prop=(ENUM_IND_DATA_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=IND_DATA_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_IND_DATA_PROP_STRING prop=(ENUM_IND_DATA_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

**The method of placing a symbol, timeframe and data object index in the timeseries**:

```
//+------------------------------------------------------------------+
//| Set symbol, timeframe and object bar start time                  |
//+------------------------------------------------------------------+
void CDataInd::SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe,const datetime time)
  {
   this.SetProperty(IND_DATA_PROP_TIME,time);
   this.SetProperty(IND_DATA_PROP_SYMBOL,symbol);
   this.SetProperty(IND_DATA_PROP_PERIOD,timeframe);
  }
//+------------------------------------------------------------------+
```

**The method displaying descriptions of all object properties in the journal**:

```
//+------------------------------------------------------------------+
//| Display object properties in the journal                         |
//+------------------------------------------------------------------+
void CDataInd::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG)," (",this.IndicatorShortName(),") =============");
   int beg=0, end=IND_DATA_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_IND_DATA_PROP_INTEGER prop=(ENUM_IND_DATA_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=IND_DATA_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_IND_DATA_PROP_DOUBLE prop=(ENUM_IND_DATA_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=IND_DATA_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_IND_DATA_PROP_STRING prop=(ENUM_IND_DATA_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_END)," (",this.IndicatorShortName(),") =============\n");
  }
//+------------------------------------------------------------------+
```

Description of each next property is displayed by object property arrays in three loops. If the property is not supported, it is not displayed in the journal if the input parameter of full\_prop method is false (by default).

**Virtual method displaying the short object description in the journal**:

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CDataInd::PrintShort(void)
  {
   ::Print
     (
      this.IndicatorShortName(),
      " [",CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_BUFFER)," ",this.BufferNum(),\
      ", ",CMessage::Text(MSG_SYM_STATUS_INDEX)," ",this.Index(this.Timeframe()),"]"
     );
  }
//+------------------------------------------------------------------+
```

The method displays standard indicator buffer data description in the following format:

```
AMA(EURUSD,H1) [Buffer 0, Index 0]
```

for custom indicator:

```
Examples\Custom Moving Average.ex5(EURUSD,H1) [Buffer 0, Index 1]
```

Short name of the indicator included into buffer data description can be changed using method **SetIndicatorShortname()**. The method may be changed in descendant objects to display other descriptions of data object which correspond to the data implemented in descendant object.

**The methods returning descriptions of object’s integer, real and string properties**:

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
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of object's real property                     |
//+------------------------------------------------------------------+
string CDataInd::GetPropertyDescription(ENUM_IND_DATA_PROP_DOUBLE property)
  {
   int dg=(this.m_digits>0 ? this.m_digits : 1);
   return
     (
      property==IND_DATA_PROP_BUFFER_VALUE   ?  CMessage::Text(MSG_LIB_TEXT_IND_DATA_BUFFER_VALUE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of object's string property                   |
//+------------------------------------------------------------------+
string CDataInd::GetPropertyDescription(ENUM_IND_DATA_PROP_STRING property)
  {
   return
     (
      property==IND_DATA_PROP_SYMBOL         ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_SYMBOL)+": \""+this.GetProperty(property)+"\""     :
      property==IND_DATA_PROP_IND_NAME       ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_NAME)+": \""+this.GetProperty(property)+"\""       :
      property==IND_DATA_PROP_IND_SHORTNAME  ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_SHORTNAME)+": \""+this.GetProperty(property)+"\""  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

The methods receive corresponding properties and depending on their value their text descriptions which are set in file **Datas.mqh** added above are returned.

**The class possesses two constructors.**

The first constructor is by default and creates an empty data object which after creation must be filled with all necessary data.

The second constructor is parametric; it receives necessary data for creation of an object with specified basic properties:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CDataInd::CDataInd(const ENUM_INDICATOR ind_type,
                   const int ind_id,
                   const int buffer_num,
                   const string symbol,
                   const ENUM_TIMEFRAMES timeframe,
                   const datetime time)
  {
   this.m_type=COLLECTION_INDICATORS_ID;
   this.m_digits=(int)::SymbolInfoInteger(symbol,SYMBOL_DIGITS)+1;
   this.m_period_description=TimeframeDescription(timeframe);
   this.SetSymbolPeriod(symbol,timeframe,time);
   this.SetIndicatorType(ind_type);
   this.SetBufferNum(buffer_num);
   this.SetIndicatorID(ind_id);
  }
//+------------------------------------------------------------------+
```

At once, specify for the object: indicator type, which buffer data are described by the object, indicator ID \- by this property we can get buffer data of earlier created indicator object for which the ID for its quick search among other created indicator objects is set. Indicator buffer number, which data are described by the object, symbol and timeframe on which indicator object is created, and bar start time, which buffer data are described by the object created.

Apart from adding the above specified parameters the constructor sets a default number of decimal places for displaying indicator buffer values (Digits of symbol + 1 digit), and adds the description of timeframe to variable **m\_period\_description** — a description may be set once when creating an object. To variable **m\_type**, which is declared [in CObject parent class of standard library](https://www.mql5.com/en/docs/standardlibrary/cobject), write temporarily the ID of indicator collection. In furtherance, when creating indicator buffer data collections, I will add the ID of this new collection to this variable.

Now, to be able to sort data objects in their collection (I will start making collections from the next article), to file \\MQL5\\Include\\DoEasy\\Services\ **Select.mqh** add methods of work with the new object to select and to sort by its properties.

Include the just created class of indicator buffer data object to the file:

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\Event.mqh"
#include "..\Objects\Accounts\Account.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\PendRequest\PendRequest.mqh"
#include "..\Objects\Series\SeriesDE.mqh"
#include "..\Objects\Indicators\Buffer.mqh"
#include "..\Objects\Indicators\IndicatorDE.mqh"
#include "..\Objects\Indicators\DataInd.mqh"
//+------------------------------------------------------------------+
```

In the end of class body declare work methods with just created class of indicator data object:

```
//+------------------------------------------------------------------+
//| Methods of work with indicator data                              |
//+------------------------------------------------------------------+
   //--- Return the list of indicator data with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByIndicatorDataProperty(CArrayObj *list_source,ENUM_IND_DATA_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByIndicatorDataProperty(CArrayObj *list_source,ENUM_IND_DATA_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByIndicatorDataProperty(CArrayObj *list_source,ENUM_IND_DATA_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the indicator data index in the list with the maximum value of (1) integer, (2) real and (3) string property of data
   static int        FindIndDataMax(CArrayObj *list_source,ENUM_IND_DATA_PROP_INTEGER property);
   static int        FindIndDataMax(CArrayObj *list_source,ENUM_IND_DATA_PROP_DOUBLE property);
   static int        FindIndDataMax(CArrayObj *list_source,ENUM_IND_DATA_PROP_STRING property);
   //--- Return the indicator data index in the list with the minimum value of (1) integer, (2) real and (3) string property of data
   static int        FindIndDataMin(CArrayObj *list_source,ENUM_IND_DATA_PROP_INTEGER property);
   static int        FindIndDataMin(CArrayObj *list_source,ENUM_IND_DATA_PROP_DOUBLE property);
   static int        FindIndDataMin(CArrayObj *list_source,ENUM_IND_DATA_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

And implement all declared methods to file end:

```
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Methods of work with indicator data lists                        |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of indicators data with one of integer           |
//| properties meeting the specified criterion                       |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByIndicatorDataProperty(CArrayObj *list_source,ENUM_IND_DATA_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CDataInd *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of indicators data with one of real              |
//| properties meeting the specified criterion                       |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByIndicatorDataProperty(CArrayObj *list_source,ENUM_IND_DATA_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CDataInd *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of indicators data with one of string            |
//| properties meeting the specified criterion                       |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByIndicatorDataProperty(CArrayObj *list_source,ENUM_IND_DATA_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CDataInd *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the indicator data index in the list                      |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindIndDataMax(CArrayObj *list_source,ENUM_IND_DATA_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CDataInd *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CDataInd *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator data index in the list                      |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindIndDataMax(CArrayObj *list_source,ENUM_IND_DATA_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CDataInd *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CDataInd *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator data index in the list                      |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindIndDataMax(CArrayObj *list_source,ENUM_IND_DATA_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CDataInd *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CDataInd *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator data index in the list                      |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindIndDataMin(CArrayObj* list_source,ENUM_IND_DATA_PROP_INTEGER property)
  {
   int index=0;
   CDataInd *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CDataInd *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator data index in the list                      |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindIndDataMin(CArrayObj* list_source,ENUM_IND_DATA_PROP_DOUBLE property)
  {
   int index=0;
   CDataInd *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CDataInd *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator data index in the list                      |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindIndDataMin(CArrayObj* list_source,ENUM_IND_DATA_PROP_STRING property)
  {
   int index=0;
   CDataInd *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CDataInd *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

Functioning of these methods was analyzed in detail [in the third article of the library creation description](https://www.mql5.com/en/articles/5687).

So, the new indicator buffer data object is created and now it can be tested.

### Testing

To perform the test, let's use [the EA from the previous article](https://www.mql5.com/en/articles/8646#node04) and save it in the new folder

\\MQL5\\Experts\\TestDoEasy\ **Part57\** under the new name **TestDoEasyPart57.mq5**.

In the previous EA I created four indicator objects: two standard and two custom ones. Identical indicators differed from each other only by other parameters. Here, create the same four indicators, but to display their data create for each of them two buffer data objects - for the current (zero) and the previous (first) bar of timeseries. Data of all objects will be displayed in the comment on symbol chart.

Until indicator buffer data collections are created let’s organize the access to the created class directly from the EA.

To do this, include this class to EA file:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart57.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Indicators\DataInd.mqh>
//--- enums
```

In the same place, in the area of global variables of the program add variable pointers to indicator data objects:

```
//--- Arrays of custom indicator parameters
MqlParam       param_ma1[];
MqlParam       param_ma2[];
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

When creating new data objects, into those variables I will save pointers to newly created objects so that later access to them can be received.

Since all objects will be created using operator [new](https://www.mql5.com/en/docs/basis/operators/newoperator) they all must be removed individually. I will do it in OnDeinit() handler of EA (after creation of the collection of indicator buffer data there will be no necessity in those actions in EA):

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

In OnTick() handler create new objects (unless they are created already), fill them with all necessary data and values and display descriptions of objects in the journal; whereas, display on chart the buffer values of indicators described by objects:

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
//--- Get custom indicator objects
   CIndicatorDE *ma1=engine.GetIndicatorsCollection().GetIndByID(MA1);
   CIndicatorDE *ma2=engine.GetIndicatorsCollection().GetIndByID(MA2);
   CIndicatorDE *ama1=engine.GetIndicatorsCollection().GetIndByID(AMA1);
   CIndicatorDE *ama2=engine.GetIndicatorsCollection().GetIndByID(AMA2);

//--- Write bar start time 0 and 1 to specify it in properties of further created objects
   datetime time0=iTime(ma1.Symbol(),ma1.Timeframe(),0);
   datetime time1=iTime(ma1.Symbol(),ma1.Timeframe(),1);
   if(time0==0 || time1==0)
      return;

//--- Create data objects of МА1 and МА2 for bars 0 and 1 (unless the objects are created)
   if(data_ma1_0==NULL) data_ma1_0=new CDataInd(ma1.TypeIndicator(),ma1.ID(),0,ma1.Symbol(),ma1.Timeframe(),time0);
   if(data_ma1_1==NULL) data_ma1_1=new CDataInd(ma1.TypeIndicator(),ma1.ID(),0,ma1.Symbol(),ma1.Timeframe(),time1);
   if(data_ma2_0==NULL) data_ma2_0=new CDataInd(ma2.TypeIndicator(),ma2.ID(),0,ma2.Symbol(),ma2.Timeframe(),time0);
   if(data_ma2_1==NULL) data_ma2_1=new CDataInd(ma2.TypeIndicator(),ma2.ID(),0,ma2.Symbol(),ma2.Timeframe(),time1);
   if(data_ma1_0==NULL || data_ma1_1==NULL || data_ma2_0==NULL || data_ma2_1==NULL) return;

//--- Set parameters of data object of indicator МА1, bar 0
//--- and add indicator buffer data to the object
   data_ma1_0.SetIndicatorType(ma1.TypeIndicator());
   data_ma1_0.SetIndicatorName(ma1.Name());
   data_ma1_0.SetIndicatorShortname(ma1.ShortName());
   data_ma1_0.SetBufferValue(ma1.GetDataBuffer(0,time0));
//--- Set parameters of data object of indicator МА1, bar 1
//--- and add indicator buffer data to the object
   data_ma1_1.SetIndicatorType(ma1.TypeIndicator());
   data_ma1_1.SetIndicatorName(ma1.Name());
   data_ma1_1.SetIndicatorShortname(ma1.ShortName());
   data_ma1_1.SetBufferValue(ma1.GetDataBuffer(0,time1));
//--- Set parameters of data object of indicator МА2, bar 0
//--- and add indicator buffer data to the object
   data_ma2_0.SetIndicatorType(ma2.TypeIndicator());
   data_ma2_0.SetIndicatorName(ma2.Name());
   data_ma2_0.SetIndicatorShortname(ma2.ShortName());
   data_ma2_0.SetBufferValue(ma2.GetDataBuffer(0,time0));
//--- Set parameters of data object of indicator МА2, bar 1
//--- and add indicator buffer data to the object
   data_ma2_1.SetIndicatorType(ma2.TypeIndicator());
   data_ma2_1.SetIndicatorName(ma2.Name());
   data_ma2_1.SetIndicatorShortname(ma2.ShortName());
   data_ma2_1.SetBufferValue(ma2.GetDataBuffer(0,time1));

//--- Create data objects of АМА1 and АМА2 for bars 0 and 1 (unless the objects are already created)
   if(data_ama1_0==NULL) data_ama1_0=new CDataInd(ama1.TypeIndicator(),ama1.ID(),0,ama1.Symbol(),ama1.Timeframe(),time0);
   if(data_ama1_1==NULL) data_ama1_1=new CDataInd(ama1.TypeIndicator(),ama1.ID(),0,ama1.Symbol(),ama1.Timeframe(),time1);
   if(data_ama2_0==NULL) data_ama2_0=new CDataInd(ama2.TypeIndicator(),ama2.ID(),0,ama2.Symbol(),ama2.Timeframe(),time0);
   if(data_ama2_1==NULL) data_ama2_1=new CDataInd(ama2.TypeIndicator(),ama2.ID(),0,ama2.Symbol(),ama2.Timeframe(),time1);
   if(data_ama1_0==NULL || data_ama1_1==NULL || data_ama2_0==NULL || data_ama2_1==NULL) return;
//--- Set parameters of data object of indicator АМА1, bar 0
//--- and add indicator buffer data to the object
   data_ama1_0.SetIndicatorType(ama1.TypeIndicator());
   data_ama1_0.SetIndicatorName(ama1.Name());
   data_ama1_0.SetIndicatorShortname(ama1.ShortName());
   data_ama1_0.SetBufferValue(ama1.GetDataBuffer(0,time0));
//--- Set parameters of data object of indicator АМА1, bar 1
//--- and add indicator buffer data to the object
   data_ama1_1.SetIndicatorType(ama1.TypeIndicator());
   data_ama1_1.SetIndicatorName(ama1.Name());
   data_ama1_1.SetIndicatorShortname(ama1.ShortName());
   data_ama1_1.SetBufferValue(ama1.GetDataBuffer(0,time1));
//--- Set parameters of data object of indicator АМА2, bar 0
//--- and add indicator buffer data to the object
   data_ama2_0.SetIndicatorType(ama2.TypeIndicator());
   data_ama2_0.SetIndicatorName(ama2.Name());
   data_ama2_0.SetIndicatorShortname(ama2.ShortName());
   data_ama2_0.SetBufferValue(ama2.GetDataBuffer(0,time0));
//--- Set parameters of data object of indicator АМА2, bar 1
//--- and add indicator buffer data to the object
   data_ama2_1.SetIndicatorType(ama2.TypeIndicator());
   data_ama2_1.SetIndicatorName(ama2.Name());
   data_ama2_1.SetIndicatorShortname(ama2.ShortName());
   data_ama2_1.SetBufferValue(ama2.GetDataBuffer(0,time1));

//--- During the first launch, print full and short data of created indicator data objects
   static bool first_start=true;
   if(first_start)
     {
      //--- Full data of buffers МА1 and МА2
      data_ma1_0.Print();
      data_ma1_1.Print();
      data_ma2_0.Print();
      data_ma2_1.Print();
      //--- Full data of buffers АМА1 and АМА2
      data_ama1_0.Print();
      data_ama1_1.Print();
      data_ama2_0.Print();
      data_ama2_1.Print();
      //--- Short data of buffers МА1 and МА2
      data_ma1_0.PrintShort();
      data_ma1_1.PrintShort();
      data_ma2_0.PrintShort();
      data_ma2_1.PrintShort();
      //--- Short data of buffers АМА1 and АМА2
      data_ama1_0.PrintShort();
      data_ama1_1.PrintShort();
      data_ama2_0.PrintShort();
      data_ama2_1.PrintShort();
      //---
      first_start=false;
     }
//--- Display the values of indicator buffers to comment on the chart from data objects
   Comment
     (
      "ma1(1)=",DoubleToString(data_ma1_1.PriceValue(),6),", ma1(0)=",DoubleToString(data_ma1_0.PriceValue(),data_ma1_0.Digits()),", ",
      "ma2(1)=",DoubleToString(data_ma2_1.PriceValue(),6),", ma2(0)=",DoubleToString(data_ma2_0.PriceValue(),data_ma2_0.Digits()),"\n",
      "ama1(1)=",DoubleToString(data_ama1_1.PriceValue(),6),", ama1(0)=",DoubleToString(data_ama1_0.PriceValue(),data_ama1_0.Digits()),", ",
      "ama2(1)=",DoubleToString(data_ama2_1.PriceValue(),6),", ama2(0)=",DoubleToString(data_ama2_0.PriceValue(),data_ama2_0.Digits())
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

In furtherance, all actions as to creation of objects and filling of their properties with values will be performed in collection class of indicator buffer data. Meanwhile, function of the objects created today must be simply checked.

Compile the EA and launch it on symbol chart having preliminary set in settings to use only current symbol and period of the chart. The journal will display data of all created indicator objects and data objects:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10425.23 USD, 1:100, Hedge, Demo account MetaTrader 5
--- Initializing the "DoEasy" library ---
Work only with the current symbol: "EURUSD"
Work only with the current Period: H1
Symbol time series EURUSD:
- Timeseries "EURUSD" H1: Required: 1000, Actual: 1000, Created: 1000, On server: 6351
Library initialization time: 00:00:00.000
============= The beginning of the parameter list: "Custom indicator" =============
Indicator status: Custom indicator
Indicator type: CUSTOM
Indicator timeframe: H1
Indicator handle: 10
Indicator group: Trend indicator
Indicator ID: 1
------
Empty value for plotting, for which there is no drawing: EMPTY_VALUE
------
Indicator symbol: EURUSD
Indicator name: "Examples\Custom Moving Average.ex5"
Indicator shortname: "Examples\Custom Moving Average.ex5(EURUSD,H1)"
 --- Indicator parameters ---
 - [1] Type int: 13
 - [2] Type int: 0
 - [3] Type int: 0
================== End of the parameter list: "Custom indicator" ==================

============= The beginning of the parameter list: "Custom indicator" =============
Indicator status: Custom indicator
Indicator type: CUSTOM
Indicator timeframe: H1
Indicator handle: 11
Indicator group: Trend indicator
Indicator ID: 2
------
Empty value for plotting, for which there is no drawing: EMPTY_VALUE
------
Indicator symbol: EURUSD
Indicator name: "Examples\Custom Moving Average.ex5"
Indicator shortname: "Examples\Custom Moving Average.ex5(EURUSD,H1)"
 --- Indicator parameters ---
 - [1] Type int: 13
 - [2] Type int: 0
 - [3] Type int: 0
 - [4] Type int: 2
================== End of the parameter list: "Custom indicator" ==================

============= The beginning of the parameter list: "Standard indicator" =============
Indicator status: Standard indicator
Indicator type: AMA
Indicator timeframe: H1
Indicator handle: 12
Indicator group: Trend indicator
Indicator ID: 3
------
Empty value for plotting, for which there is no drawing: EMPTY_VALUE
------
Indicator symbol: EURUSD
Indicator name: "Adaptive Moving Average"
Indicator shortname: "AMA(EURUSD,H1)"
 --- Indicator parameters ---
 - Averaging period: 9
 - Fast MA period: 2
 - Slow MA period: 30
 - Horizontal shift of the indicator: 0
 - Price type or handle: CLOSE
================== End of the parameter list: "Standard indicator" ==================

============= The beginning of the parameter list: "Standard indicator" =============
Indicator status: Standard indicator
Indicator type: AMA
Indicator timeframe: H1
Indicator handle: 13
Indicator group: Trend indicator
Indicator ID: 4
------
Empty value for plotting, for which there is no drawing: EMPTY_VALUE
------
Indicator symbol: EURUSD
Indicator name: "Adaptive Moving Average"
Indicator shortname: "AMA(EURUSD,H1)"
 --- Indicator parameters ---
 - Averaging period: 14
 - Fast MA period: 2
 - Slow MA period: 30
 - Horizontal shift of the indicator: 0
 - Price type or handle: CLOSE
================== End of the parameter list: "Standard indicator" ==================

Custom indicator Examples\Custom Moving Average.ex5 EURUSD H1 [handle 10, id #1]
Custom indicator Examples\Custom Moving Average.ex5 EURUSD H1 [handle 11, id #2]
Standard indicator Adaptive Moving Average EURUSD H1 [handle 12, id #3]
Standard indicator Adaptive Moving Average EURUSD H1 [handle 13, id #4]

============= The beginning of the parameter list (Examples\Custom Moving Average.ex5(EURUSD,H1)) =============
Period start time: 2020.11.18 10:00:00
Indicator timeframe: H1
Indicator type: CUSTOM
Indicator buffer number: 0
Indicator ID: 1
------
Indicator buffer value: 1.186694
------
Indicator symbol: "EURUSD"
Indicator name: "Examples\Custom Moving Average.ex5"
Indicator shortname: "Examples\Custom Moving Average.ex5(EURUSD,H1)"
============= End of the parameter list (Examples\Custom Moving Average.ex5(EURUSD,H1)) =============

============= The beginning of the parameter list (Examples\Custom Moving Average.ex5(EURUSD,H1)) =============
Period start time: 2020.11.18 09:00:00
Indicator timeframe: H1
Indicator type: CUSTOM
Indicator buffer number: 0
Indicator ID: 1
------
Indicator buffer value: 1.186535
------
Indicator symbol: "EURUSD"
Indicator name: "Examples\Custom Moving Average.ex5"
Indicator shortname: "Examples\Custom Moving Average.ex5(EURUSD,H1)"
============= End of the parameter list (Examples\Custom Moving Average.ex5(EURUSD,H1)) =============

============= The beginning of the parameter list (Examples\Custom Moving Average.ex5(EURUSD,H1)) =============
Period start time: 2020.11.18 10:00:00
Indicator timeframe: H1
Indicator type: CUSTOM
Indicator buffer number: 0
Indicator ID: 2
------
Indicator buffer value: 1.186552
------
Indicator symbol: "EURUSD"
Indicator name: "Examples\Custom Moving Average.ex5"
Indicator shortname: "Examples\Custom Moving Average.ex5(EURUSD,H1)"
============= End of the parameter list (Examples\Custom Moving Average.ex5(EURUSD,H1)) =============

============= The beginning of the parameter list (Examples\Custom Moving Average.ex5(EURUSD,H1)) =============
Period start time: 2020.11.18 09:00:00
Indicator timeframe: H1
Indicator type: CUSTOM
Indicator buffer number: 0
Indicator ID: 2
------
Indicator buffer value: 1.186438
------
Indicator symbol: "EURUSD"
Indicator name: "Examples\Custom Moving Average.ex5"
Indicator shortname: "Examples\Custom Moving Average.ex5(EURUSD,H1)"
============= End of the parameter list (Examples\Custom Moving Average.ex5(EURUSD,H1)) =============

============= The beginning of the parameter list (AMA(EURUSD,H1)) =============
Period start time: 2020.11.18 10:00:00
Indicator timeframe: H1
Indicator type: AMA
Indicator buffer number: 0
Indicator ID: 3
------
Indicator buffer value: 1.186992
------
Indicator symbol: "EURUSD"
Indicator name: "Adaptive Moving Average"
Indicator shortname: "AMA(EURUSD,H1)"
============= End of the parameter list (AMA(EURUSD,H1)) =============

============= The beginning of the parameter list (AMA(EURUSD,H1)) =============
Period start time: 2020.11.18 09:00:00
Indicator timeframe: H1
Indicator type: AMA
Indicator buffer number: 0
Indicator ID: 3
------
Indicator buffer value: 1.186725
------
Indicator symbol: "EURUSD"
Indicator name: "Adaptive Moving Average"
Indicator shortname: "AMA(EURUSD,H1)"
============= End of the parameter list (AMA(EURUSD,H1)) =============

============= The beginning of the parameter list (AMA(EURUSD,H1)) =============
Period start time: 2020.11.18 10:00:00
Indicator timeframe: H1
Indicator type: AMA
Indicator buffer number: 0
Indicator ID: 4
------
Indicator buffer value: 1.186548
------
Indicator symbol: "EURUSD"
Indicator name: "Adaptive Moving Average"
Indicator shortname: "AMA(EURUSD,H1)"
============= End of the parameter list (AMA(EURUSD,H1)) =============

============= The beginning of the parameter list (AMA(EURUSD,H1)) =============
Period start time: 2020.11.18 09:00:00
Indicator timeframe: H1
Indicator type: AMA
Indicator buffer number: 0
Indicator ID: 4
------
Indicator buffer value: 1.186403
------
Indicator symbol: "EURUSD"
Indicator name: "Adaptive Moving Average"
Indicator shortname: "AMA(EURUSD,H1)"
============= End of the parameter list (AMA(EURUSD,H1)) =============

Examples\Custom Moving Average.ex5(EURUSD,H1) [Buffer 0, Index 0]
Examples\Custom Moving Average.ex5(EURUSD,H1) [Buffer 0, Index 1]
Examples\Custom Moving Average.ex5(EURUSD,H1) [Buffer 0, Index 0]
Examples\Custom Moving Average.ex5(EURUSD,H1) [Buffer 0, Index 1]
AMA(EURUSD,H1) [Buffer 0, Index 0]
AMA(EURUSD,H1) [Buffer 0, Index 1]
AMA(EURUSD,H1) [Buffer 0, Index 0]
AMA(EURUSD,H1) [Buffer 0, Index 1]
```

while the chart (in comment) will display data which correspond to data in indicator buffers on the first and zero bars:

![](https://c.mql5.com/2/41/terminal64_A5qXxhH382.png)

### What's next?

In the next article we will create the collection class of indicator buffer data.

All files of the current library version are attached below together with the test EA file for MQL5. You can download them and test everything.

Note, that at the moment indicator collection class is under development, therefore  it is strictly recommended not to use it in your programs.

Leave your comments, questions and suggestions in the comments to the article.

[Back to contents](https://www.mql5.com/en/articles/8705#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8705](https://www.mql5.com/ru/articles/8705)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8705.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8705/mql5.zip "Download MQL5.zip")(3862.22 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/359804)**

![Optimal approach to the development and analysis of trading systems](https://c.mql5.com/2/40/optimal-approach.png)[Optimal approach to the development and analysis of trading systems](https://www.mql5.com/en/articles/8410)

In this article, I will show the criteria to be used when selecting a system or a signal for investing your funds, as well as describe the optimal approach to the development of trading systems and highlight the importance of this matter in Forex trading.

![Timeseries in DoEasy library (part 56): Custom indicator object, get data from indicator objects in the collection](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library.png)[Timeseries in DoEasy library (part 56): Custom indicator object, get data from indicator objects in the collection](https://www.mql5.com/en/articles/8646)

The article considers creation of the custom indicator object for the use in EAs. Let’s slightly improve library classes and add methods to get data from indicator objects in EAs.

![Neural networks made easy (Part 6): Experimenting with the neural network learning rate](https://c.mql5.com/2/48/Neural_networks_made_easy_006.png)[Neural networks made easy (Part 6): Experimenting with the neural network learning rate](https://www.mql5.com/en/articles/8485)

We have previously considered various types of neural networks along with their implementations. In all cases, the neural networks were trained using the gradient decent method, for which we need to choose a learning rate. In this article, I want to show the importance of a correctly selected rate and its impact on the neural network training, using examples.

![Practical application of neural networks in trading. Python (Part I)](https://c.mql5.com/2/40/neural_python.png)[Practical application of neural networks in trading. Python (Part I)](https://www.mql5.com/en/articles/8502)

In this article, we will analyze the step-by-step implementation of a trading system based on the programming of deep neural networks in Python. This will be performed using the TensorFlow machine learning library developed by Google. We will also use the Keras library for describing neural networks.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/8705&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071859797101326288)

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