---
title: Timeseries in DoEasy library (part 53): Abstract base indicator class
url: https://www.mql5.com/en/articles/8464
categories: Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T19:34:12.927921
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ztgbbivmiydqhibdqzlesdlchktzapgr&ssn=1769186050023647855&ssn_dr=0&ssn_sr=0&fv_date=1769186050&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8464&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2053)%3A%20Abstract%20base%20indicator%20class%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918605040257255&fz_uniq=5070391622660723969&sv=2552)

MetaTrader 5 / Trading systems


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8464#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8464#node02)
- [Abstract indicator class](https://www.mql5.com/en/articles/8464#node03)
- [Testing](https://www.mql5.com/en/articles/8464#node04)
- [What's next?](https://www.mql5.com/en/articles/8464#node05)


### Concept

As DoEasy library is developed, we have come to a necessity to create an indicator object. Existence of such object will allow convenient storage and use of all indicators created and used in the program. Concept of indicator object construction does not differ from the concept of main library objects, namely: base abstract object and its descendants clarifying object affiliation by its status (for indicators - custom and standard). I told about creation of such objects in [the very first articles](https://www.mql5.com/en/articles/5654).

Today, create the object of base abstract indicator and check results of its creation. In the following articles I will create objects of standard and custom indicators.

In addition to affiliation by status (standard and custom) each of the indicator objects created will be affiliated by indicator type (group):

- Trend indicator

- Oscillator

- Volumes

- Arrow indicator


Thus, we will be able to sort indicators by groups in programs. We will not enter Bill Williams’ indicators in a separate group because each of them has its affiliation with one of the specified groups. Therefore, I consider it unnecessary to enter another separate group which will include indicators of all above listed groups.

### Improving library classes

First, add necessary library text messages for indicator objects.

In file \\MQL5\\Include\\DoEasy\ **Data.mqh** add new message indices:

```
//--- CBuffer
//--- removed for the sake of space
//--- ...
//--- ...
//--- ...
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_SOLID,              // Solid line
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASH,               // Dashed line
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DOT,                // Dotted line
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASHDOT,            // Dot-dash line
   MSG_LIB_TEXT_BUFFER_TEXT_STYLE_DASHDOTDOT,         // Dash - two dots

//--- CIndicatorDE
   MSG_LIB_TEXT_IND_TEXT_STATUS,                      // Indicator status
   MSG_LIB_TEXT_IND_TEXT_STATUS_STANDART,             // Standard indicator
   MSG_LIB_TEXT_IND_TEXT_STATUS_CUSTOM,               // Custom indicator

   MSG_LIB_TEXT_IND_TEXT_TIMEFRAME,                   // Indicator timeframe
   MSG_LIB_TEXT_IND_TEXT_HANDLE,                      // Indicator handle

   MSG_LIB_TEXT_IND_TEXT_GROUP,                       // Indicator group
   MSG_LIB_TEXT_IND_TEXT_GROUP_TREND,                 // Trend indicator
   MSG_LIB_TEXT_IND_TEXT_GROUP_OSCILLATOR,            // Oscillator
   MSG_LIB_TEXT_IND_TEXT_GROUP_VOLUMES,               // Volumes
   MSG_LIB_TEXT_IND_TEXT_GROUP_ARROWS,                // Arrow indicator

   MSG_LIB_TEXT_IND_TEXT_EMPTY_VALUE,                 // Empty value for plotting where nothing will be drawn:
   MSG_LIB_TEXT_IND_TEXT_SYMBOL,                      // Indicator symbol
   MSG_LIB_TEXT_IND_TEXT_NAME,                        // Indicator name
   MSG_LIB_TEXT_IND_TEXT_SHORTNAME,                   // Indicator short name

  };
//+------------------------------------------------------------------+
```

... and further in the same file - text messages corresponding to newly added indices:

```
   {"Solid line"},
   {"Broken line"},
   {"Dotted line"},
   {"Dash-dot line"},
   {"Dash - two points"},

   {"Indicator status"},
   {"Standard indicator"},
   {"Custom indicator"},
   {"Indicator timeframe"},
   {"Indicator handle"},
   {"Indicator group"},
   {"Trend indicator"},
   {"Solid lineOscillator"},
   {"Volumes"},
   {"Arrow indicator"},
   {"Empty value for plotting, for which there is no drawing"},
   {"Indicator symbol"},
   {"Indicator name"},
   {"Indicator shortname"},

  };
//+---------------------------------------------------------------------+
```

In file E:\\MetaQuotes\\MetaTrader 5\\MQL5\\Include\\DoEasy\ **Defines.mqh** add for library objects the indicator object parameters which already became standard.

Since all these objects will be stored in indicator buffers collection list in the end, let's introduce own ID for them:

```
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x777A)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777B)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777C)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777D)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777E)                   // Symbol collection list ID
#define COLLECTION_SERIES_ID           (0x777F)                   // Timeseries collection list ID
#define COLLECTION_BUFFERS_ID          (0x7780)                   // Indicator buffer collection list ID
#define COLLECTION_INDICATORS_ID       (0x7781)                   // Indicator collection list ID

//--- Data parameters for file operations
```

And in the end of listing file add all necessary enumerations of properties and sort criteria for indicator objects in collection list:

```
//+------------------------------------------------------------------+
//| Data for working with indicators                                 |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Abstract indicator status                                        |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_STATUS
  {
   INDICATOR_STATUS_STANDART,                               // Standard indicator
   INDICATOR_STATUS_CUSTOM,                                 // Custom indicator
  };
//+------------------------------------------------------------------+
//| Indicator group                                                  |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_GROUP
  {
   INDICATOR_GROUP_TREND,                                   // Trend indicator
   INDICATOR_GROUP_OSCILLATOR,                              // Oscillator
   INDICATOR_GROUP_VOLUMES,                                 // Volumes
   INDICATOR_GROUP_ARROWS,                                  // Arrow indicator
  };
//+------------------------------------------------------------------+
//| Indicator integer properties                                     |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_PROP_INTEGER
  {
   INDICATOR_PROP_STATUS = 0,                               // Indicator status (from enumeration ENUM_INDICATOR_STATUS)
   INDICATOR_PROP_TIMEFRAME,                                // Indicator timeframe
   INDICATOR_PROP_HANDLE,                                   // Indicator handle
   INDICATOR_PROP_GROUP,                                    // Indicator group
  };
#define INDICATOR_PROP_INTEGER_TOTAL (4)                    // Total number of indicator integer properties
#define INDICATOR_PROP_INTEGER_SKIP  (0)                    // Number of indicator properties not used in sorting
//+------------------------------------------------------------------+
//| Indicator real properties                                        |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_PROP_DOUBLE
  {
   INDICATOR_PROP_EMPTY_VALUE = INDICATOR_PROP_INTEGER_TOTAL,// Empty value for plotting where nothing will be drawn
  };
#define INDICATOR_PROP_DOUBLE_TOTAL  (1)                    // Total number of real indicator properties
#define INDICATOR_PROP_DOUBLE_SKIP   (0)                    // Number of indicator properties not used in sorting
//+------------------------------------------------------------------+
//| Indicator string properties                                      |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_PROP_STRING
  {
   INDICATOR_PROP_SYMBOL = (INDICATOR_PROP_INTEGER_TOTAL+INDICATOR_PROP_DOUBLE_TOTAL), // Indicator symbol
   INDICATOR_PROP_NAME,                                     // Indicator name
   INDICATOR_PROP_SHORTNAME,                                // Indicator short name
  };
#define INDICATOR_PROP_STRING_TOTAL  (3)                    // Total number of indicator string properties
//+------------------------------------------------------------------+
//| Possible indicator sorting criteria                              |
//+------------------------------------------------------------------+
#define FIRST_INDICATOR_DBL_PROP          (INDICATOR_PROP_INTEGER_TOTAL-INDICATOR_PROP_INTEGER_SKIP)
#define FIRST_INDICATOR_STR_PROP          (INDICATOR_PROP_INTEGER_TOTAL-INDICATOR_PROP_INTEGER_SKIP+INDICATOR_PROP_DOUBLE_TOTAL-INDICATOR_PROP_DOUBLE_SKIP)
enum ENUM_SORT_INDICATOR_MODE
  {
//--- Sort by integer properties
   SORT_BY_INDICATOR_INDEX_STATUS = 0,                      // Sort by indicator status
   SORT_BY_INDICATOR_TIMEFRAME,                             // Sort by indicator timeframe
   SORT_BY_INDICATOR_HANDLE,                                // Sort by indicator handle
   SORT_BY_INDICATOR_GROUP,                                 // Sort by indicator group
//--- Sort by real properties
   SORT_BY_INDICATOR_EMPTY_VALUE = FIRST_INDICATOR_DBL_PROP,// Sort by the empty value for plotting where nothing will be drawn
//--- Sort by string properties
   SORT_BY_INDICATOR_SYMBOL = FIRST_INDICATOR_STR_PROP,     // Sort by indicator symbol
   SORT_BY_INDICATOR_NAME,                                  // Sort by indicator name
   SORT_BY_INDICATOR_SHORTNAME,                             // Sort by indicator short name
  };
//+------------------------------------------------------------------+
```

No explanations are needed here.

### Abstract indicator class

Proceed to creation of base abstract indicator class.

In folder \\MQL5\\Include\\DoEasy\\Objects\ **Indicators\** create a new class CIndicatorDE in file named **IndicatorDE.mqh**. Since in the standard library [class CIndicator](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicator) already exists I added acronym DE ( **D** o **E** asy) to class and file name.

The new class is derived from [base object class of all objects of CBaseObj library](https://www.mql5.com/en/articles/7071).

Class body contains methods for setting and getting object properties which we considered earlier. Simply look through their listing and further analyze some class methods:

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
//+------------------------------------------------------------------+
//| Abstract indicator class                                         |
//+------------------------------------------------------------------+
class CIndicatorDE : public CBaseObj
  {
private:
   long              m_long_prop[INDICATOR_PROP_INTEGER_TOTAL];                  // Integer properties
   double            m_double_prop[INDICATOR_PROP_DOUBLE_TOTAL];                 // Real properties
   string            m_string_prop[INDICATOR_PROP_STRING_TOTAL];                 // String properties
   MqlParam          m_mql_params[];                                             // Array of indicator parameters

//--- Return the index of the array the buffer's (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_INDICATOR_PROP_DOUBLE property)        const { return(int)property-INDICATOR_PROP_INTEGER_TOTAL;                           }
   int               IndexProp(ENUM_INDICATOR_PROP_STRING property)        const { return(int)property-INDICATOR_PROP_INTEGER_TOTAL-INDICATOR_PROP_DOUBLE_TOTAL;}

//--- Comare (1) structures MqlParam, (2) array of structures MqlParam between each other
   bool              IsEqualMqlParams(MqlParam &struct1,MqlParam &struct2) const;
   bool              IsEqualMqlParamArrays(MqlParam &compared_struct[])    const;

protected:
public:
//--- Protected parametric constructor
                     CIndicatorDE(ENUM_INDICATOR ind_type,
                                  string symbol,
                                  ENUM_TIMEFRAMES timeframe,
                                  ENUM_INDICATOR_STATUS status,
                                  ENUM_INDICATOR_GROUP group,
                                  string name,
                                  string shortname,
                                  MqlParam &indicator_params[]);
public:
//--- Default constructor
                     CIndicatorDE(void){;}
//--- Destructor
                    ~CIndicatorDE(void);

//--- Set buffer's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_INDICATOR_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                                        }
   void              SetProperty(ENUM_INDICATOR_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;                      }
   void              SetProperty(ENUM_INDICATOR_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;                      }
//--- Return (1) integer, (2) real and (3) string buffer properties from the properties array
   long              GetProperty(ENUM_INDICATOR_PROP_INTEGER property)        const { return this.m_long_prop[property];                                       }
   double            GetProperty(ENUM_INDICATOR_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];                     }
   string            GetProperty(ENUM_INDICATOR_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];                     }
//--- Return description of buffer's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_INDICATOR_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_INDICATOR_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_INDICATOR_PROP_STRING property);
//--- Return the flag of the buffer supporting the property
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_INTEGER property)          { return true;       }
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_DOUBLE property)           { return true;       }
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_STRING property)           { return true;       }

//--- Compare CIndicatorDE objects by all possible properties (for sorting the lists by a specified indicator object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CIndicatorDE objects by all properties (to search for equal indicator objects)
   bool              IsEqual(CIndicatorDE* compared_obj) const;

//--- Set indicator’s (1) group, (2) empty value of buffers, (3) name, (4) short name
   void              SetGroup(const ENUM_INDICATOR_GROUP group)      { this.SetProperty(INDICATOR_PROP_GROUP,group);                         }
   void              SetEmptyValue(const double value)               { this.SetProperty(INDICATOR_PROP_EMPTY_VALUE,value);                   }
   void              SetName(const string name)                      { this.SetProperty(INDICATOR_PROP_NAME,name);                           }
   void              SetShortName(const string shortname)            { this.SetProperty(INDICATOR_PROP_SHORTNAME,shortname);                 }

//--- Return indicator’s (1) status, (2) group, (3) timeframe, (4) handle, (5) empty value of buffers, (6) name, (7) short name, (8) symbol
   ENUM_INDICATOR_STATUS Status(void)                          const { return (ENUM_INDICATOR_STATUS)this.GetProperty(INDICATOR_PROP_STATUS);}
   ENUM_INDICATOR_GROUP  Group(void)                           const { return (ENUM_INDICATOR_GROUP)this.GetProperty(INDICATOR_PROP_GROUP);  }
   ENUM_TIMEFRAMES   Timeframe(void)                           const { return (ENUM_TIMEFRAMES)this.GetProperty(INDICATOR_PROP_TIMEFRAME);   }
   int               Handle(void)                              const { return (int)this.GetProperty(INDICATOR_PROP_HANDLE);                  }
   double            EmptyValue(void)                          const { return this.GetProperty(INDICATOR_PROP_EMPTY_VALUE);                  }
   string            Name(void)                                const { return this.GetProperty(INDICATOR_PROP_NAME);                         }
   string            ShortName(void)                           const { return this.GetProperty(INDICATOR_PROP_SHORTNAME);                    }
   string            Symbol(void)                              const { return this.GetProperty(INDICATOR_PROP_SYMBOL);                       }

//--- Return description of indicator’s (1) status, (2) group, (3) timeframe, (4) empty value
   string            GetStatusDescription(void)                const;
   string            GetGroupDescription(void)                 const;
   string            GetTimeframeDescription(void)             const;
   string            GetEmptyValueDescription(void)            const;

//--- Display the description of indicator object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of indicator object in the journal (implementation in the descendants)
   virtual void      PrintShort(void) {;}

  };
//+------------------------------------------------------------------+
```

Make the closed parametric class constructor temporarily public — today we want to test if creation of a single class object is successful. For that reason this constructor must be open for external calls.

As you see, in their logic all class methods repeat all earlier considered library objects. Therefore, readers must be familiar with their purpose and work. As opposed to all other objects where destructor is not available and it is created implicitly, here I added destructor because the indicator created for the object must be destructed. Let’s analyze implementation of methods which is performed outside the class body.

**In class destructor** destruct the created indicator object:

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CIndicatorDE::~CIndicatorDE(void)
  {
   ::IndicatorRelease(this.Handle());
  }
//+------------------------------------------------------------------+
```

**Closed parametric class constructor:**

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CIndicatorDE::CIndicatorDE(ENUM_INDICATOR ind_type,
                           string symbol,
                           ENUM_TIMEFRAMES timeframe,
                           ENUM_INDICATOR_STATUS status,
                           ENUM_INDICATOR_GROUP group,
                           string name,
                           string shortname,
                           MqlParam &indicator_params[])
  {
//--- Set collection ID to the object
   this.m_type=COLLECTION_INDICATORS_ID;
//--- If parameter array value passed to constructor is more than zero
//--- fill in the array of object parameters with data from the array passed to constructor
   int count=::ArrayResize(m_mql_params,::ArraySize(indicator_params));
   for(int i=0;i<count;i++)
     {
      this.m_mql_params[i].type=indicator_params[i].type;
      this.m_mql_params[i].double_value=indicator_params[i].double_value;
      this.m_mql_params[i].integer_value=indicator_params[i].integer_value;
      this.m_mql_params[i].string_value=indicator_params[i].string_value;
     }
//--- Create indicator handle
   int handle=::IndicatorCreate(symbol,timeframe,ind_type,count,indicator_params);

//--- Save integer properties
   this.m_long_prop[INDICATOR_PROP_STATUS]                     = status;
   this.m_long_prop[INDICATOR_PROP_GROUP]                      = group;
   this.m_long_prop[INDICATOR_PROP_TIMEFRAME]                  = timeframe;
   this.m_long_prop[INDICATOR_PROP_HANDLE]                     = handle;

//--- Save real properties
   this.m_double_prop[this.IndexProp(INDICATOR_PROP_EMPTY_VALUE)]=EMPTY_VALUE;
//--- Save string properties
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_SYMBOL)]   = (symbol==NULL || symbol=="" ? ::Symbol() : symbol);
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_NAME)]     = name;
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_SHORTNAME)]= shortname;
  }
//+------------------------------------------------------------------+
```

All necessary parameters to create indicator object are passed to constructor.

If the array of parameters MqlParam &indicator\_params\[\] has zero value, when creating an indicator [IndicatorCreate()](https://www.mql5.com/en/docs/series/indicatorcreate) function will not use the parameters which must be contained in array.

Further, simply save in object property fields all values passed to the method.

To compare two indicator objects between each other to sort and search two identical objects all fields of the two objects must be compared. Indicator object contains the array of [MqlParam structures](https://www.mql5.com/en/docs/constants/structures/mqlparam) — they must also be compared with each other. They will be compared element by element. To do it we have two methods: a method to compare two MqlParam structures with each other and a method to compare two arrays of those structures.

**//\-\-\- Method to compare two MqlParam structures with each other:**

```
//+------------------------------------------------------------------+
//| Compare MqlParam structures with each other                      |
//+------------------------------------------------------------------+
bool CIndicatorDE::IsEqualMqlParams(MqlParam &struct1,MqlParam &struct2) const
  {
   bool res=
     (struct1.type!=struct2.type ? false :
      (struct1.type==TYPE_STRING && struct1.string_value==struct2.string_value) ||
      (struct1.type<TYPE_STRING && struct1.type>TYPE_ULONG && struct1.double_value==struct2.double_value) ||
      (struct1.type<TYPE_FLOAT && struct1.integer_value==struct2.integer_value) ? true : false
     );
   return res;
  }
//+------------------------------------------------------------------+
```

It is simple here.

Check fields of types of structure data and if data types of these compared structures do not correspond - return false.

If data type is string and the data of the two structures is identical — return true.

If data type is real and the data of the two structures is identical — return true.

If data type is integer and the data of the two structures is identical — return true.

In any other case — return false.

**Method comparing arrays of MqlParam structures between each other:**

```
//+------------------------------------------------------------------+
//| Compare array of MqlParam structures with each other             |
//+------------------------------------------------------------------+
bool CIndicatorDE::IsEqualMqlParamArrays(MqlParam &compared_struct[]) const
  {
   int total=::ArraySize(this.m_mql_params);
   int size=::ArraySize(compared_struct);
   if(total!=size || total==0 || size==0)
      return false;
   for(int i=0;i<total;i++)
     {
      if(!this.IsEqualMqlParams(this.m_mql_params[i],compared_struct[i]))
         return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

If values of two arrays are not equal or any of them has a zero value — return false.

Then, in a loop by the number of structures in arraycompare every following structures of the two arrays and if they are not equal — return false.

After successful check of all structures contained in two arrays return true.

**Method of comparison of CIndicatorDE objects by all possible properties:**

```
//+------------------------------------------------------------------+
//| Compare CIndicatorDE objects with each other                     |
//| by all possible properties                                       |
//+------------------------------------------------------------------+
int CIndicatorDE::Compare(const CObject *node,const int mode=0) const
  {
   const CIndicatorDE *compared_obj=node;
//--- compare integer properties of two indicators
   if(mode<INDICATOR_PROP_INTEGER_TOTAL)
     {
      long value_compared=compared_obj.GetProperty((ENUM_INDICATOR_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_INDICATOR_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two indicators
   else if(mode<INDICATOR_PROP_INTEGER_TOTAL+INDICATOR_PROP_DOUBLE_TOTAL)
     {
      double value_compared=compared_obj.GetProperty((ENUM_INDICATOR_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_INDICATOR_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two indicators
   else if(mode<INDICATOR_PROP_INTEGER_TOTAL+INDICATOR_PROP_DOUBLE_TOTAL+INDICATOR_PROP_STRING_TOTAL)
     {
      string value_compared=compared_obj.GetProperty((ENUM_INDICATOR_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_INDICATOR_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

**Comparison method of CIndicatorDE objects by all properties:**

```
//+------------------------------------------------------------------+
//| Compare CIndicatorDE objects with each other by all properties   |
//+------------------------------------------------------------------+
bool CIndicatorDE::IsEqual(CIndicatorDE *compared_obj) const
  {
   if(!IsEqualMqlParamArrays(compared_obj.m_mql_params))
      return false;
   int beg=0, end=INDICATOR_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_INDICATOR_PROP_INTEGER prop=(ENUM_INDICATOR_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=INDICATOR_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_INDICATOR_PROP_DOUBLE prop=(ENUM_INDICATOR_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=INDICATOR_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_INDICATOR_PROP_STRING prop=(ENUM_INDICATOR_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Two comparison methods of CIndicatorDE objects with each other are identical in terms of logic to methods of the same name of other library objects and we considered them repeatedly. The only difference in the second method (IsEqual) — is that first, comparison method of two arrays of MqlParam structures of objects compared is called. If they are not equal the objects are not equal - return false. Further, compare objects by all their fields.

**The methods returning description of indicator’s integer, real and string property:**

```
//+------------------------------------------------------------------+
//| Return description of indicator's integer property               |
//+------------------------------------------------------------------+
string CIndicatorDE::GetPropertyDescription(ENUM_INDICATOR_PROP_INTEGER property)
  {
   return
     (
      property==INDICATOR_PROP_STATUS        ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_STATUS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetStatusDescription()
         )  :
      property==INDICATOR_PROP_GROUP          ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_GROUP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetGroupDescription()
         )  :
      property==INDICATOR_PROP_TIMEFRAME     ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_TIMEFRAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTimeframeDescription()
         )  :
      property==INDICATOR_PROP_HANDLE        ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_HANDLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of indicator's real property                  |
//+------------------------------------------------------------------+
string CIndicatorDE::GetPropertyDescription(ENUM_INDICATOR_PROP_DOUBLE property)
  {
   return
     (
      property==INDICATOR_PROP_EMPTY_VALUE    ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_EMPTY_VALUE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetEmptyValueDescription()
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of indicator's string property                |
//+------------------------------------------------------------------+
string CIndicatorDE::GetPropertyDescription(ENUM_INDICATOR_PROP_STRING property)
  {
   return
     (
      property==INDICATOR_PROP_SYMBOL     ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_SYMBOL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.Symbol()
         )  :
      property==INDICATOR_PROP_NAME       ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_NAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.Name()==NULL || this.Name()=="" ? CMessage::Text(MSG_LIB_PROP_NOT_SET) : "\""+this.Name()+"\"")
         )  :
      property==INDICATOR_PROP_SHORTNAME  ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_SHORTNAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.ShortName()==NULL || this.ShortName()=="" ? CMessage::Text(MSG_LIB_PROP_NOT_SET) : "\""+this.ShortName()+"\"")
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Each library object has the same methods and we considered them earlier, as well.

**Remaining methods for displaying descriptions of various indicator object properties** are also identical to the same methods in other library objects. Therefore, let’s simply look at their listing for individual studying:

```
//+------------------------------------------------------------------+
//| Return indicator status description                              |
//+------------------------------------------------------------------+
string CIndicatorDE::GetStatusDescription(void) const
  {
   return
     (
      this.Status()==INDICATOR_STATUS_CUSTOM    ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_STATUS_CUSTOM)    :
      this.Status()==INDICATOR_STATUS_STANDART  ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_STATUS_STANDART)  :
      "Unknown"
     );
  }
//+------------------------------------------------------------------+
//| Return indicator group description                               |
//+------------------------------------------------------------------+
string CIndicatorDE::GetGroupDescription(void) const
  {
   return
     (
      this.Group()==INDICATOR_GROUP_TREND       ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_GROUP_TREND)      :
      this.Group()==INDICATOR_GROUP_OSCILLATOR  ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_GROUP_OSCILLATOR) :
      this.Group()==INDICATOR_GROUP_VOLUMES     ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_GROUP_VOLUMES)    :
      this.Group()==INDICATOR_GROUP_ARROWS      ? CMessage::Text(MSG_LIB_TEXT_IND_TEXT_GROUP_ARROWS)     :
      "Any"
     );
  }
//+------------------------------------------------------------------+
//| Return description of timeframe used                             |
//+------------------------------------------------------------------+
string CIndicatorDE::GetTimeframeDescription(void) const
  {
   string timeframe=TimeframeDescription(this.Timeframe());
   return(this.Timeframe()==PERIOD_CURRENT ? CMessage::Text(MSG_LIB_TEXT_PERIOD_CURRENT)+" ("+timeframe+")" : timeframe);
  }
//+------------------------------------------------------------------+
//| Return description of the set empty value                        |
//+------------------------------------------------------------------+
string CIndicatorDE::GetEmptyValueDescription(void) const
  {
   double value=fabs(this.EmptyValue());
   return(value<EMPTY_VALUE ? ::DoubleToString(this.EmptyValue(),(this.EmptyValue()==0 ? 1 : 8)) : (this.EmptyValue()>0 ? "EMPTY_VALUE" : "-EMPTY_VALUE"));
  }
//+------------------------------------------------------------------+
//| Display indicator properties in the journal                      |
//+------------------------------------------------------------------+
void CIndicatorDE::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG),": \"",this.GetStatusDescription(),"\" =============");
   int beg=0, end=INDICATOR_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_INDICATOR_PROP_INTEGER prop=(ENUM_INDICATOR_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=INDICATOR_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_INDICATOR_PROP_DOUBLE prop=(ENUM_INDICATOR_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=INDICATOR_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_INDICATOR_PROP_STRING prop=(ENUM_INDICATOR_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("================== ",CMessage::Text(MSG_LIB_PARAMS_LIST_END),": \"",this.GetStatusDescription(),"\" ==================\n");
  }
//+------------------------------------------------------------------+
```

**This is the complete composition of indicator object. It is quite probably that we will improve it further. But for now, this is sufficient to check object creation.**

Now, we should enable to work efficiently with the list of new indicator objects in the library: to sort them and select required ones by preset criteria. To do this, in file \\MQL5\\Include\\DoEasy\\Services\ **Select.mqh** add inclusion of class file of base abstract indicator

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
//+------------------------------------------------------------------+
```

and in the end of class body declare methods for work with indicator object list:

```
//+------------------------------------------------------------------+
//| Methods of working with indicators                               |
//+------------------------------------------------------------------+
   //--- Return the list of indicators with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByIndicatorProperty(CArrayObj *list_source,ENUM_INDICATOR_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByIndicatorProperty(CArrayObj *list_source,ENUM_INDICATOR_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByIndicatorProperty(CArrayObj *list_source,ENUM_INDICATOR_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the indicator index in the list with the maximum value of the indicator's (1) integer, (2) real and (3) string property
   static int        FindIndicatorMax(CArrayObj *list_source,ENUM_INDICATOR_PROP_INTEGER property);
   static int        FindIndicatorMax(CArrayObj *list_source,ENUM_INDICATOR_PROP_DOUBLE property);
   static int        FindIndicatorMax(CArrayObj *list_source,ENUM_INDICATOR_PROP_STRING property);
   //--- Return the indicator index in the list with the minimum value of the indicator's (1) integer, (2) real and (3) string property
   static int        FindIndicatorMin(CArrayObj *list_source,ENUM_INDICATOR_PROP_INTEGER property);
   static int        FindIndicatorMin(CArrayObj *list_source,ENUM_INDICATOR_PROP_DOUBLE property);
   static int        FindIndicatorMin(CArrayObj *list_source,ENUM_INDICATOR_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

All methods of work with indicator objects are also absolutely standard and identical to earlier set methods for search and sort in collection lists of other library objects. They all [were considered](https://www.mql5.com/en/articles/5687#node01) earlier. Let’s simply analyze their implementation for individual studying and review of studied material:

```
//+------------------------------------------------------------------+
//| Methods of working with indicator lists                          |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of indicators with one of integer                |
//| properties meeting the specified criterion                       |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByIndicatorProperty(CArrayObj *list_source,ENUM_INDICATOR_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of indicators with one of real                   |
//| properties meeting the specified criterion                       |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByIndicatorProperty(CArrayObj *list_source,ENUM_INDICATOR_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of indicators with one of string                 |
//| properties meeting the specified criterion                       |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByIndicatorProperty(CArrayObj *list_source,ENUM_INDICATOR_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the indicator index in the list                           |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindIndicatorMax(CArrayObj *list_source,ENUM_INDICATOR_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CIndicatorDE *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator index in the list                           |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindIndicatorMax(CArrayObj *list_source,ENUM_INDICATOR_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CIndicatorDE *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator index in the list                           |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindIndicatorMax(CArrayObj *list_source,ENUM_INDICATOR_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CIndicatorDE *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator index in the list                           |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindIndicatorMin(CArrayObj* list_source,ENUM_INDICATOR_PROP_INTEGER property)
  {
   int index=0;
   CIndicatorDE *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator index in the list                           |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindIndicatorMin(CArrayObj* list_source,ENUM_INDICATOR_PROP_DOUBLE property)
  {
   int index=0;
   CIndicatorDE *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the indicator index in the list                           |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindIndicatorMin(CArrayObj* list_source,ENUM_INDICATOR_PROP_STRING property)
  {
   int index=0;
   CIndicatorDE *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CIndicatorDE *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

At the moment, in our library indicators are created and used inside the collection class of CBuffersCollection indicator buffer. I create a separate class for indicator objects which, in turn, will be collected into their collection class. Access to indicator objects will be available from this collection. In the meantime, work inside CBuffersCollection class since today we only must create indicator object and check if it is successful.

In following articles, when descendant classes of abstract indicator are created, I will collect them into collection and in CBuffersCollection class I will work not directly with indicators (as we do now), but using indicator collection class.

For today, our purpose is to create and check the fact of indicator object creation. Therefore, improvements will concern only one method of buffer collection class, which is the method for creation of Accelerator Oscillator indicator.

Enter changes in the method for cross-platform nature and create indicator object; print its data in the journal and delete this object at once. This is all that needs to be done to check creation of the abstract indicator object.

Open indicator buffer collection class file \\MQL5\\Include\\DoEasy\\Collections\ **BuffersCollection.mqh** and make necessary improvements.

To create indicator object use IndicatorCreate() function to which required parameters for many indicators must be passed. Such parameters are passed using the array of [MqlParam](https://www.mql5.com/en/docs/constants/structures/mqlparam), a structure specifically developed for this purpose.

Declare such array in private class section:

```
//+------------------------------------------------------------------+
//| Collection of indicator buffers                                  |
//+------------------------------------------------------------------+
class CBuffersCollection : public CObject
  {
private:
   CListObj                m_list;                       // Buffer object list
   CTimeSeriesCollection  *m_timeseries;                 // Pointer to the timeseries collection object
   MqlParam                m_mql_param[];                // Array of indicator parameters
//--- Return the index of the (1) last, (2) next drawn and (3) basic buffer
   int                     GetIndexLastPlot(void);
   int                     GetIndexNextPlot(void);
   int                     GetIndexNextBase(void);
//--- Create a new buffer object and place it to the collection list
   bool                    CreateBuffer(ENUM_BUFFER_STATUS status);
//--- Get data of the necessary timeseries and bars for working with a single buffer bar, and return the number of bars
   int                     GetBarsData(CBuffer *buffer,const int series_index,int &index_bar_period);

public:
```

Since creation of indicator object will be checked only in one method namely, creation method of Accelerator Oscillator CreateAC() indicator all improvements will concern only this method. All remaining methods for creation of indicator buffers will be improved in following articles.

Separate the method into two blocks namely, for MQL5 and for MQL4 for cross-platform nature. But in the meantime, for MQL4 only make creation of the second buffer for displaying the second color (Accelerator Oscillator is two-color). But I will not make a change of histogram color, which in fact means alternation of display of two different buffers for visual display of different indicator colors in MQL4. Today, I do a completely different thing.

In the very beginning of the method add strings for indicator object creation, output of data for created object to the journal and object removal:

```
//+------------------------------------------------------------------+
//| Create multi-symbol multi-period AC                              |
//+------------------------------------------------------------------+
int CBuffersCollection::CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id=WRONG_VALUE)
  {
//--- To check it, create indicator object, print its data and remove it at once
   ::ArrayResize(this.m_mql_param,0);
   CIndicatorDE *indicator=new CIndicatorDE(IND_AC,symbol,timeframe,INDICATOR_STATUS_STANDART,INDICATOR_GROUP_OSCILLATOR,"Accelerator Oscillator","AC("+symbol+","+TimeframeDescription(timeframe)+")",this.m_mql_param);
   indicator.Print();
   delete indicator;

//--- Create indicator handle and set default ID
   int handle= #ifdef __MQL5__ ::iAC(symbol,timeframe) #else 0 #endif ;
   int identifier=(id==WRONG_VALUE ? IND_AC : id);
   color array_colors[3]={clrGreen,clrRed,clrGreen};
   CBuffer *buff=NULL;
   if(handle!=INVALID_HANDLE)
     {
      //--- Create histogram buffer from the zero line
      this.CreateHistogram();
      //--- Get the last created buffer object (drawn) and set to it all necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_AC);
      buff.SetLineMode(INDICATOR_LINE_MODE_MAIN);
      buff.SetShowData(true);
      buff.SetLabel("AC("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Accelerator Oscillator");
      buff.SetIndicatorShortName("AC("+symbol+","+TimeframeDescription(timeframe)+")");
      #ifdef __MQL5__
         buff.SetColors(array_colors);
      #else
         buff.SetColor(array_colors[0]);
         buff.SetIndicatorLineAdditionalNumber(0);
      #endif

      //--- MQL5
      #ifdef __MQL5__
         //--- Create calculated buffer, in which standard indicator data will be stored
         this.CreateCalculate();
         //--- Get the last created buffer object (calculated) and set to it all necessary parameters
         buff=this.GetLastCreateBuffer();
         if(buff==NULL)
            return INVALID_HANDLE;
         buff.SetSymbol(symbol);
         buff.SetTimeframe(timeframe);
         buff.SetID(identifier);
         buff.SetIndicatorHandle(handle);
         buff.SetIndicatorType(IND_AC);
         buff.SetLineMode(INDICATOR_LINE_MODE_MAIN);
         buff.SetEmptyValue(EMPTY_VALUE);
         buff.SetLabel("AC("+symbol+","+TimeframeDescription(timeframe)+")");
         buff.SetIndicatorName("Accelerator Oscillator");
         buff.SetIndicatorShortName("AC("+symbol+","+TimeframeDescription(timeframe)+")");

      //--- MQL4
      #else
        //--- Create histogram buffer from the zero line for buffer of the second color
         this.CreateHistogram();
         //--- Get the last created buffer object (drawn) and set to it all necessary parameters
         buff=this.GetLastCreateBuffer();
         if(buff==NULL)
            return INVALID_HANDLE;
         buff.SetSymbol(symbol);
         buff.SetTimeframe(timeframe);
         buff.SetID(identifier);
         buff.SetIndicatorHandle(handle);
         buff.SetIndicatorType(IND_AC);
         buff.SetLineMode(INDICATOR_LINE_MODE_MAIN);
         buff.SetShowData(true);
         buff.SetLabel("AC("+symbol+","+TimeframeDescription(timeframe)+")");
         buff.SetIndicatorName("Accelerator Oscillator");
         buff.SetIndicatorShortName("AC("+symbol+","+TimeframeDescription(timeframe)+")");
         #ifdef __MQL5__
            buff.SetColors(array_colors);
         #else
            buff.SetColor(array_colors[1]);
            buff.SetIndicatorLineAdditionalNumber(1);
         #endif
      #endif
     }
   return handle;
  }
//+------------------------------------------------------------------+
```

Since Accelerator Oscillator standard indicator possesses no parameters, I will not use the array of indicator parameters when creating indicator handle by IndicatorCreate() function.

Reset the size of parameter array.

Create a new indicator object having passed to its constructor all data required for object creation,

at once print data of the newly created object (I will not check if creation of the object is successful because this is just a test).

And remove this object — to avoid memory leak.

In following articles, after creation of descendant objects of base abstract indicator and their placement to indicator collection at their creation, add remaining methods for indicator buffer creation. Today, even such check will be sufficient.

In the method of setting a value for the current chart to buffers of the specified standard indicator by the timeseries index in accordance with buffer object symbol/period make a workpiece to organize cross-platform nature having added separation into code blocks for MQL5 and MQL4 only for single-buffer standard indicators (I will not implement the code for MQL4 because I do not need it today):

```
//+------------------------------------------------------------------+
//| Set values for the current chart to buffers of the specified     |
//| standard indicator by the timeseries index in accordance         |
//| with buffer object symbol/period                                 |
//+------------------------------------------------------------------+
bool CBuffersCollection::SetDataBufferStdInd(const ENUM_INDICATOR ind_type,const int id,const int series_index,const datetime series_time,const char color_index=WRONG_VALUE)
  {
//--- Get the list of buffer objects by type and ID
   CArrayObj *list=this.GetListBufferByTypeID(ind_type,id);
   if(list==NULL || list.Total()==0)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_NO_BUFFER_OBJ));
      return false;
     }

//--- Get the list of drawn buffers with ID
   CArrayObj *list_data=CSelect::ByBufferProperty(list,BUFFER_PROP_TYPE,BUFFER_TYPE_DATA,EQUAL);
   list_data=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_TYPE,ind_type,EQUAL);
//--- Get the list of calculated buffers with ID
   CArrayObj *list_calc=CSelect::ByBufferProperty(list,BUFFER_PROP_TYPE,BUFFER_TYPE_CALCULATE,EQUAL);
   list_calc=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_TYPE,ind_type,EQUAL);

//--- Leave if any of the lists is empty
   if(list_data.Total()==0 #ifdef __MQL5__ || list_calc.Total()==0 #endif )
      return false;

//--- Declare necessary objects and variables
   CBuffer *buffer_data0=NULL,*buffer_data1=NULL,*buffer_data2=NULL,*buffer_data3=NULL,*buffer_data4=NULL,*buffer_tmp0=NULL,*buffer_tmp1=NULL;
   CBuffer *buffer_calc0=NULL,*buffer_calc1=NULL,*buffer_calc2=NULL,*buffer_calc3=NULL,*buffer_calc4=NULL;
   #ifdef __MQL4__ CBuffer *buff_add=NULL; #endif

   double value00=EMPTY_VALUE, value01=EMPTY_VALUE;
   double value10=EMPTY_VALUE, value11=EMPTY_VALUE;
   double value20=EMPTY_VALUE, value21=EMPTY_VALUE;
   double value30=EMPTY_VALUE, value31=EMPTY_VALUE;
   double value40=EMPTY_VALUE, value41=EMPTY_VALUE;
   double value_tmp0=EMPTY_VALUE,value_tmp1=EMPTY_VALUE;
   long vol0=0,vol1=0;
   int series_index_start=series_index,index_period=0, index=0,num_bars=1;
   uchar clr=0;
//--- Depending on standard indicator type

   switch((int)ind_type)
     {
   //--- Single-buffer standard indicators
      case IND_AC       :
      case IND_AD       :
      case IND_AMA      :
      case IND_AO       :
      case IND_ATR      :
      case IND_BEARS    :
      case IND_BULLS    :
      case IND_BWMFI    :
      case IND_CCI      :
      case IND_CHAIKIN  :
      case IND_DEMA     :
      case IND_DEMARKER :
      case IND_FORCE    :
      case IND_FRAMA    :
      case IND_MA       :
      case IND_MFI      :
      case IND_MOMENTUM :
      case IND_OBV      :
      case IND_OSMA     :
      case IND_RSI      :
      case IND_SAR      :
      case IND_STDDEV   :
      case IND_TEMA     :
      case IND_TRIX     :
      case IND_VIDYA    :
      case IND_VOLUMES  :
      case IND_WPR      :
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer_data0=list.At(0);
      #ifdef __MQL5__
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer_calc0=list.At(0);
      #endif

        if(buffer_data0==NULL #ifdef __MQL5__ || buffer_calc0==NULL || buffer_calc0.GetDataTotal(0)==0 #endif )
           return false;

        series_index_start=PreparingSetDataStdInd(buffer_data0,buffer_data1,buffer_data2,buffer_data3,buffer_data4,
                                                  buffer_calc0,buffer_calc1,buffer_calc2,buffer_calc3,buffer_calc4,
                                                  ind_type,series_index,series_time,index_period,num_bars,
                                                  value00,value01,value10,value11,value20,value21,value30,value31,value40,value41);

        if(series_index_start==WRONG_VALUE)
           return false;
        //--- In a loop, by the number of bars in  num_bars fill in the drawn buffer with a value from the calculated buffer taken by index_period index
        //--- and set the drawn buffer color depending on a proportion of value00 and value01 values
        for(int i=0;i<num_bars;i++)
          {
           index=series_index_start-i;
           buffer_data0.SetBufferValue(0,index,value00);
           if(ind_type!=IND_BWMFI)
              clr=(color_index==WRONG_VALUE ? uchar(value00>value01 ? 0 : value00<value01 ? 1 : 2) : color_index);
           else
             {
              vol0=::iVolume(buffer_data0.Symbol(),buffer_data0.Timeframe(),index_period);
              vol1=::iVolume(buffer_data0.Symbol(),buffer_data0.Timeframe(),index_period+1);
              clr=
                (
                 value00>value01 && vol0>vol1 ? 0 :
                 value00<value01 && vol0<vol1 ? 1 :
                 value00>value01 && vol0<vol1 ? 2 :
                 value00<value01 && vol0>vol1 ? 3 : 4
                );
             }
           #ifdef __MQL5__
              buffer_data0.SetBufferColorIndex(index,clr);
           #else

           #endif
          }
        return true;

   //--- Multi-buffer standard indicators
      case IND_ENVELOPES :
      case IND_FRACTALS  :
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer_data0=list.At(0);
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer_data1=list.At(0);

        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer_calc0=list.At(0);
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer_calc1=list.At(0);

        if(buffer_calc0==NULL || buffer_data0==NULL || buffer_calc0.GetDataTotal(0)==0)
           return false;
        if(buffer_calc1==NULL || buffer_data1==NULL || buffer_calc1.GetDataTotal(0)==0)
           return false;

        series_index_start=PreparingSetDataStdInd(buffer_data0,buffer_data1,buffer_data2,buffer_data3,buffer_data4,
                                                  buffer_calc0,buffer_calc1,buffer_calc2,buffer_calc3,buffer_calc4,
                                                  ind_type,series_index,series_time,index_period,num_bars,
                                                  value00,value01,value10,value11,value20,value21,value30,value31,value40,value41);
        if(series_index_start==WRONG_VALUE)
           return false;
        //--- In a loop, by the number of bars in  num_bars fill in the drawn buffer with a value from the calculated buffer taken by index_period index
        //--- and set the drawn buffer color depending on a proportion of value00 and value01 values
        for(int i=0;i<num_bars;i++)
          {
           index=series_index_start-i;
           buffer_data0.SetBufferValue(0,index,value00);
           buffer_data1.SetBufferValue(1,index,value10);
           buffer_data0.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value00>value01 ? 0 : value00<value01 ? 1 : 2) : color_index);
           buffer_data1.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value10>value11 ? 0 : value10<value11 ? 1 : 2) : color_index);
          }
        return true;

      case IND_ADX         :
      case IND_ADXW        :
      case IND_BANDS       :
      case IND_MACD        :
      case IND_RVI         :
      case IND_STOCHASTIC  :
      case IND_ALLIGATOR   :
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer_data0=list.At(0);
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer_data1=list.At(0);
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,2,EQUAL);
        buffer_data2=list.At(0);

        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer_calc0=list.At(0);
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer_calc1=list.At(0);
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,2,EQUAL);
        buffer_calc2=list.At(0);

        if(buffer_calc0==NULL || buffer_data0==NULL || buffer_calc0.GetDataTotal(0)==0)
           return false;
        if(buffer_calc1==NULL || buffer_data1==NULL || buffer_calc1.GetDataTotal(0)==0)
           return false;
        if(buffer_calc2==NULL || buffer_data2==NULL || buffer_calc2.GetDataTotal(0)==0)
           return false;

        series_index_start=PreparingSetDataStdInd(buffer_data0,buffer_data1,buffer_data2,buffer_data3,buffer_data4,
                                                  buffer_calc0,buffer_calc1,buffer_calc2,buffer_calc3,buffer_calc4,
                                                  ind_type,series_index,series_time,index_period,num_bars,
                                                  value00,value01,value10,value11,value20,value21,value30,value31,value40,value41);
        if(series_index_start==WRONG_VALUE)
           return false;
        //--- In a loop, by the number of bars in  num_bars fill in the drawn buffer with a value from the calculated buffer taken by index_period index
        //--- and set the drawn buffer color depending on a proportion of value00 and value01 values
        for(int i=0;i<num_bars;i++)
          {
           index=series_index_start-i;
           buffer_data0.SetBufferValue(0,index,value00);
           buffer_data1.SetBufferValue(0,index,value10);
           buffer_data2.SetBufferValue(0,index,value20);
           buffer_data0.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value00>value01 ? 0 : value00<value01 ? 1 : 2) : color_index);
           buffer_data1.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value10>value11 ? 0 : value10<value11 ? 1 : 2) : color_index);
           buffer_data2.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value20>value21 ? 0 : value20<value21 ? 1 : 2) : color_index);
          }
        return true;

      case IND_ICHIMOKU :
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_TENKAN_SEN,EQUAL);
        buffer_data0=list.At(0);
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_KIJUN_SEN,EQUAL);
        buffer_data1=list.At(0);
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_SENKOU_SPANA,EQUAL);
        buffer_data2=list.At(0);
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_SENKOU_SPANB,EQUAL);
        buffer_data3=list.At(0);
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_CHIKOU_SPAN,EQUAL);
        buffer_data4=list.At(0);

        //--- Get the list of buffer objects which have ID of auxiliary line, and from it - buffer object with line number as 0
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_ADDITIONAL,EQUAL);
        list=CSelect::ByBufferProperty(list,BUFFER_PROP_IND_LINE_ADDITIONAL_NUM,0,EQUAL);
        buffer_tmp0=list.At(0);
        //--- Get the list of buffer objects which have ID of auxiliary line, and from it - buffer object with line number as 1
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_ADDITIONAL,EQUAL);
        list=CSelect::ByBufferProperty(list,BUFFER_PROP_IND_LINE_ADDITIONAL_NUM,1,EQUAL);
        buffer_tmp1=list.At(0);

        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_TENKAN_SEN,EQUAL);
        buffer_calc0=list.At(0);
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_KIJUN_SEN,EQUAL);
        buffer_calc1=list.At(0);
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_SENKOU_SPANA,EQUAL);
        buffer_calc2=list.At(0);
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_SENKOU_SPANB,EQUAL);
        buffer_calc3=list.At(0);
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_CHIKOU_SPAN,EQUAL);
        buffer_calc4=list.At(0);

        if(buffer_calc0==NULL || buffer_data0==NULL || buffer_calc0.GetDataTotal(0)==0)
           return false;
        if(buffer_calc1==NULL || buffer_data1==NULL || buffer_calc1.GetDataTotal(0)==0)
           return false;
        if(buffer_calc2==NULL || buffer_data2==NULL || buffer_calc2.GetDataTotal(0)==0)
           return false;
        if(buffer_calc3==NULL || buffer_data3==NULL || buffer_calc3.GetDataTotal(0)==0)
           return false;
        if(buffer_calc4==NULL || buffer_data4==NULL || buffer_calc4.GetDataTotal(0)==0)
           return false;

        series_index_start=PreparingSetDataStdInd(buffer_data0,buffer_data1,buffer_data2,buffer_data3,buffer_data4,
                                                  buffer_calc0,buffer_calc1,buffer_calc2,buffer_calc3,buffer_calc4,
                                                  ind_type,series_index,series_time,index_period,num_bars,
                                                  value00,value01,value10,value11,value20,value21,value30,value31,value40,value41);
        if(series_index_start==WRONG_VALUE)
           return false;
        //--- In a loop, by the number of bars in  num_bars fill in the drawn buffer with a value from the calculated buffer taken by index_period index
        //--- and set the drawn buffer color depending on a proportion of value00 and value01 values
        for(int i=0;i<num_bars;i++)
          {
           index=series_index_start-i;
           buffer_data0.SetBufferValue(0,index,value00);
           buffer_data1.SetBufferValue(0,index,value10);
           buffer_data2.SetBufferValue(0,index,value20);
           buffer_data3.SetBufferValue(0,index,value30);
           buffer_data4.SetBufferValue(0,index,value40);
           buffer_data0.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value00>value01 ? 0 : value00<value01 ? 1 : 2) : color_index);
           buffer_data1.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value10>value11 ? 0 : value10<value11 ? 1 : 2) : color_index);
           buffer_data2.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value20>value21 ? 0 : value20<value21 ? 1 : 2) : color_index);
           buffer_data3.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value30>value31 ? 0 : value30<value31 ? 1 : 2) : color_index);
           buffer_data4.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value40>value41 ? 0 : value40<value41 ? 1 : 2) : color_index);

           //--- Set values for indicator auxiliary lines depending on mutual position of  Senkou Span A and Senkou Span B lines
           value_tmp0=buffer_data2.GetDataBufferValue(0,index);
           value_tmp1=buffer_data3.GetDataBufferValue(0,index);
           if(value_tmp0<value_tmp1)
             {
              buffer_tmp0.SetBufferValue(0,index,buffer_tmp0.EmptyValue());
              buffer_tmp0.SetBufferValue(1,index,buffer_tmp0.EmptyValue());

              buffer_tmp1.SetBufferValue(0,index,value_tmp0);
              buffer_tmp1.SetBufferValue(1,index,value_tmp1);
             }
           else
             {
              buffer_tmp0.SetBufferValue(0,index,value_tmp0);
              buffer_tmp0.SetBufferValue(1,index,value_tmp1);

              buffer_tmp1.SetBufferValue(0,index,buffer_tmp1.EmptyValue());
              buffer_tmp1.SetBufferValue(1,index,buffer_tmp1.EmptyValue());
             }

          }
        return true;

      case IND_GATOR    :
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer_data0=list.At(0);
        list=CSelect::ByBufferProperty(list_data,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer_data1=list.At(0);

        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer_calc0=list.At(0);
        list=CSelect::ByBufferProperty(list_calc,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer_calc1=list.At(0);

        if(buffer_calc0==NULL || buffer_data0==NULL || buffer_calc0.GetDataTotal(0)==0)
           return false;
        if(buffer_calc1==NULL || buffer_data1==NULL || buffer_calc1.GetDataTotal(0)==0)
           return false;

        series_index_start=PreparingSetDataStdInd(buffer_data0,buffer_data1,buffer_data2,buffer_data3,buffer_data4,
                                                  buffer_calc0,buffer_calc1,buffer_calc2,buffer_calc3,buffer_calc4,
                                                  ind_type,series_index,series_time,index_period,num_bars,
                                                  value00,value01,value10,value11,value20,value21,value30,value31,value40,value41);
        if(series_index_start==WRONG_VALUE)
           return false;
        //--- In a loop, by the number of bars in  num_bars fill in the drawn buffer with a value from the calculated buffer taken by index_period index
        //--- and set the drawn buffer color depending on a proportion of value00 and value01 values
        for(int i=0;i<num_bars;i++)
          {
           index=series_index_start-i;
           buffer_data0.SetBufferValue(0,index,value00);
           buffer_data1.SetBufferValue(1,index,value10);
           buffer_data0.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value00>value01 ? 0 : value00<value01 ? 1 : 2) : color_index);
           buffer_data1.SetBufferColorIndex(index,color_index==WRONG_VALUE ? uchar(value10<value11 ? 0 : value10>value11 ? 1 : 2) : color_index);
          }
        return true;

      default:
        break;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

These are all changes we had to make today.

### Testing

To check creation of indicator object take the [test indicator from the previous article](https://www.mql5.com/en/articles/8399#node03),

save it in a new folder \\MQL5\\Indicators\\TestDoEasy\ **Part53\** under a new name **TestDoEasyPart53.mq5** and replace the strings indicating work with AD indicator by those indicating work with AC indicator:

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Write the name of the working timeframe selected in the settings to InpUsedTFs variable
   InpUsedTFs=TimeframeDescription(InpPeriod);
//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Set indicator global variables
   prefix=engine.Name()+"_";
   //--- Calculate the number of bars of the current period fitting in the maximum used period
   //--- Use the obtained value if it exceeds 2, otherwise use 2
   int num_bars=NumberBarsInTimeframe(InpPeriod);
   min_bars=(num_bars>2 ? num_bars : 2);

//--- Check and remove remaining indicator graphical objects
   if(IsPresentObectByPrefix(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel

//--- Check playing a standard sound using macro substitutions
   engine.PlaySoundByDescription(SND_OK);
//--- Wait for 600 milliseconds
   engine.Pause(600);
   engine.PlaySoundByDescription(SND_NEWS);

//--- indicator buffers mapping
//--- Create all necessary buffer objects to construct the selected standard indicator
   if(!engine.BufferCreateAC(InpUsedSymbols,InpPeriod,1))
     {
      Print(TextByLanguage("Error. Indicator not created"));
      return INIT_FAILED;
     }
//--- Check the number of buffers specified in the 'properties' block
   engine.CheckIndicatorsBuffers(indicator_buffers,indicator_plots);

//--- Create the color array and set non-default colors to all buffers within the collection
//--- (commented out since default colors are already set in methods of standard indicator creation)
//--- (we can always set required colors either for all indicators like here or for each one individually)
   //color array_colors[]={clrGreen,clrRed,clrGray};
   //engine.BuffersSetColors(array_colors);

//--- Display short descriptions of created indicator buffers
   engine.BuffersPrintShort();

//--- Set a short name for the indicator, data capacity and levels
   string label=engine.BufferGetIndicatorShortNameByTypeID(IND_AC,1);
   IndicatorSetString(INDICATOR_SHORTNAME,label);
   SetIndicatorLevels(InpUsedSymbols,IND_AC);

//--- Successful
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Remove indicator graphical objects by an object name prefix
   ObjectsDeleteAll(0,prefix);
   Comment("");
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//+------------------------------------------------------------------+
//| OnCalculate code block for working with the library:             |
//+------------------------------------------------------------------+
//--- Pass the current symbol data from OnCalculate() to the price structure and set the "as timeseries" flag to the arrays
   CopyDataAsSeries(rates_total,prev_calculated,time,open,high,low,close,tick_volume,volume,spread);

//--- Check for the minimum number of bars for calculation
   if(rates_total<min_bars || Point()==0) return 0;
//--- Handle the Calculate event in the library
//--- If the OnCalculate() method of the library returns zero, not all timeseries are ready - leave till the next tick
   if(engine.OnCalculate(rates_data)==0)
      return 0;

//--- If work in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Work in the library timer
      engine.EventsHandling();      // Work with library events
     }
//+------------------------------------------------------------------+
//| OnCalculate code block for working with the indicator:           |
//+------------------------------------------------------------------+
//--- Check and calculate the number of calculated bars
//--- If limit = 0, there are no new bars - calculate the current one
//--- If limit = 1, a new bar has appeared - calculate the first and the current ones
//--- If limit > 1 means the first launch or changes in history - the full recalculation of all data
   int limit=rates_total-prev_calculated;

//--- Recalculate the entire history
   if(limit>1)
     {
      limit=rates_total-1;
      engine.BuffersInitPlots();
      engine.BuffersInitCalculates();
     }
//--- Prepare data
//--- Fill in calculated buffers of all created standard indicators with data
   int bars_total=engine.SeriesGetBarsTotal(InpUsedSymbols,InpPeriod);
   int total_copy=(limit<min_bars ? min_bars : fmin(limit,bars_total));
   if(!engine.BufferPreparingDataAllBuffersStdInd())
      return 0;

//--- Calculate the indicator
//--- Main calculation loop of the indicator
   for(int i=limit; i>WRONG_VALUE && !IsStopped(); i--)
     {
      engine.GetBuffersCollection().SetDataBufferStdInd(IND_AC,1,i,time[i]);
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

This is what we need at the moment to check function of the newly created class of abstract indicator object in test indicator.

The full indicator code is provided in the files attached below.

Compile the indicator and launch it. “Experts” journal displays data on the created indicator object:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10425.23 USD, 1:100, Hedge, Demo account MetaTrader 5
--- Initializing "DoEasy" library ---
Working with the current symbol only. Number of used symbols: 1
"EURUSD"
Working with the specified timeframe list:
"H4" "H1"
EURUSD symbol timeseries:
- "EURUSD" H1 timeseries: Requested: 1000, Actually: 0, Created: 0, On the server: 0
- "EURUSD" H4 timeseries: Requested: 1000, Actually: 1000, Created: 1000, On the server: 6237
Time of library initializing: 00:00:00.156

============= Beginning of the parameter list: "Standard indicator" =============
Indicator status: Standard indicator
Indicator timeframe: H4
Indicator handle: 10
Indicator group: Oscillator
------
Empty value for plotting where nothing will be drawn: EMPTY_VALUE
------
Indicator symbol: EURUSD
Indicator name: "Accelerator Oscillator"
Indicator short name: "AC(EURUSD,H4)"
================== End of the parameter list "Standard indicator" ==================

Buffer(P0/B0/C1): Histogram from the zero line EURUSD H4
Buffer[P0/B2/C2]: Calculated buffer
"EURUSD" H1 timeseries created successfully:
- "EURUSD" H1 timeseries: Requested: 1000, Actually: 1000, Created: 1000, On the server: 6256
```

### What's next?

In the following article I will start creating classes of descendant objects of abstract indicator base object created today.

All files of the current version of the library are attached below together with the test indicator file for MQL5. You can download them and test everything.

Leave your comments, questions and suggestions in the comments to the article.

[Back to contents](https://www.mql5.com/en/articles/8464#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8464](https://www.mql5.com/ru/articles/8464)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8464.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8464/mql5.zip "Download MQL5.zip")(3783.3 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/358021)**

![Brute force approach to pattern search](https://c.mql5.com/2/40/back_to_the_future_part1.png)[Brute force approach to pattern search](https://www.mql5.com/en/articles/8311)

In this article, we will search for market patterns, create Expert Advisors based on the identified patterns, and check how long these patterns remain valid, if they ever retain their validity.

![Timeseries in DoEasy library (part 52): Cross-platform nature of multi-period multi-symbol  single-buffer standard indicators](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library__4.png)[Timeseries in DoEasy library (part 52): Cross-platform nature of multi-period multi-symbol single-buffer standard indicators](https://www.mql5.com/en/articles/8399)

In the article, consider creation of multi-symbol multi-period standard indicator Accumulation/Distribution. Slightly improve library classes with respect to indicators so that, the programs developed for outdated platform MetaTrader 4 based on this library could work normally when switching over to MetaTrader 5.

![Grid and martingale: what are they and how to use them?](https://c.mql5.com/2/40/mql5_martin_grid.png)[Grid and martingale: what are they and how to use them?](https://www.mql5.com/en/articles/8390)

In this article, I will try to explain in detail what grid and martingale are, as well as what they have in common. Besides, I will try to analyze how viable these strategies really are. The article features mathematical and practical sections.

![Parallel Particle Swarm Optimization](https://c.mql5.com/2/40/parallel_optimization_2.png)[Parallel Particle Swarm Optimization](https://www.mql5.com/en/articles/8321)

The article describes a method of fast optimization using the particle swarm algorithm. It also presents the method implementation in MQL, which is ready for use both in single-threaded mode inside an Expert Advisor and in a parallel multi-threaded mode as an add-on that runs on local tester agents.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/8464&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070391622660723969)

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