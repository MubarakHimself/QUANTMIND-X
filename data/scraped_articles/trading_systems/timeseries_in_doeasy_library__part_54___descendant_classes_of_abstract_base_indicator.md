---
title: Timeseries in DoEasy library (part 54): Descendant classes of abstract base indicator
url: https://www.mql5.com/en/articles/8508
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:34:01.980077
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ghorbdnuohpkjfxwfdmdahvxlvhgbjra&ssn=1769186039820391769&ssn_dr=0&ssn_sr=0&fv_date=1769186039&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8508&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2054)%3A%20Descendant%20classes%20of%20abstract%20base%20indicator%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918603974876469&fz_uniq=5070389299083416823&sv=2552)

MetaTrader 5 / Trading systems


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8508#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8508#node02)
- [Indicator object classes](https://www.mql5.com/en/articles/8508#node03)
- [Indicator object collection](https://www.mql5.com/en/articles/8508#node04)
- [Testing](https://www.mql5.com/en/articles/8508#node05)
- [What's next?](https://www.mql5.com/en/articles/8508#node06)


### Concept

[In the previous article](https://www.mql5.com/en/articles/8464) creation of base abstract indicator object was considered. Today, create its descendant objects in which information about a specific indicator object created will be specified. Place all those objects to indicator collection, from which getting data and properties of each created indicator will be possible.

Concept of descendant objects fully corresponds to the concept of object construction in the library and to their interconnections. Whereas, indicator collection will allow to quickly add event functionality to all indicator objects in furtherance. This will allow to easily set indicator events tracked and to use these events in our programs.

### Improving library classes

As usual, start with adding necessary text messages of the library.

In file \\MQL5\\Include\\DoEasy\ **Data.mqh** add new message indices:

```
//--- CIndicatorDE
   MSG_LIB_TEXT_IND_TEXT_STATUS,                      // Indicator status
   MSG_LIB_TEXT_IND_TEXT_STATUS_STANDART,             // Standard indicator
   MSG_LIB_TEXT_IND_TEXT_STATUS_CUSTOM,               // Custom indicator

   MSG_LIB_TEXT_IND_TEXT_TYPE,                        // Indicator type
   MSG_LIB_TEXT_IND_TEXT_TIMEFRAME,                   // Indicator timeframe
   MSG_LIB_TEXT_IND_TEXT_HANDLE,                      // Indicator handle

   MSG_LIB_TEXT_IND_TEXT_GROUP,                       // Indicator group
   MSG_LIB_TEXT_IND_TEXT_GROUP_TREND,                 // Trend indicator
   MSG_LIB_TEXT_IND_TEXT_GROUP_OSCILLATOR,            // Oscillator
   MSG_LIB_TEXT_IND_TEXT_GROUP_VOLUMES,               // Volumes
   MSG_LIB_TEXT_IND_TEXT_GROUP_ARROWS,                // Arrow indicator

   MSG_LIB_TEXT_IND_TEXT_EMPTY_VALUE,                 // Empty value for plotting where nothing will be drawn
   MSG_LIB_TEXT_IND_TEXT_SYMBOL,                      // Indicator symbol
   MSG_LIB_TEXT_IND_TEXT_NAME,                        // Indicator name
   MSG_LIB_TEXT_IND_TEXT_SHORTNAME,                   // Indicator short name

//--- CIndicatorsCollection
   MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST,                // Error. Failed to add indicator object to list

  };
//+------------------------------------------------------------------+
```

and next, add new messages corresponding to the newly added indices:

```
   {"Indicator status"},
   {"Standard indicator"},
   {"Custom indicator"},
   {"Indicator type"},
   {"Indicator timeframe"},
   {"Indicator handle"},
   {"Indicator group"},
   {"Trend indicator"},
   {"Solid lineOscillator"},
   {"Volumes"},
   {"Arrow indicator"},
   {,"Empty value for plotting, for which there is no drawing"},
   {"Indicator symbol"},
   {"Indicator name"},
   {"Indicator shortname"},

   {"Error. Failed to add indicator object to list"},

  };
//+---------------------------------------------------------------------+
```

During creation of abstract indicator object class in the previous article we didn’t add one of object’s integer properties - standard indicator type. This type will correspond to enumeration types [ENUM\_INDICATOR](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_indicator) and it will be required to search a specific indicator.

If we want to find all MACD indicators which are stored in collection, first, we must get the list of all MACD indicators located in collection. To do this, we must sort the full list of indicator collection by type IND\_MACD. And then, in the resulted list containing only IND\_MACD, the choice by other target properties will be made.

Add a new property of indicator object to file \\MQL5\\Include\\DoEasy\ **Defines.mqh**:

```
//+------------------------------------------------------------------+
//| Indicator integer properties                                     |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_PROP_INTEGER
  {
   INDICATOR_PROP_STATUS = 0,                               // Indicator status (from enumeration  ENUM_INDICATOR_STATUS)
   INDICATOR_PROP_TYPE,                                     // Indicator type (from enumeration ENUM_INDICATOR)
   INDICATOR_PROP_TIMEFRAME,                                // Indicator timeframe
   INDICATOR_PROP_HANDLE,                                   // Indicator handle
   INDICATOR_PROP_GROUP,                                    // Indicator group
  };
#define INDICATOR_PROP_INTEGER_TOTAL (5)                    // Total number of indicator integer properties
#define INDICATOR_PROP_INTEGER_SKIP  (0)                    // Number of indicator properties not used in sorting
//+------------------------------------------------------------------+
```

and increase the total number of integer properties from 4 to **5**.

When adding a new property for the object we must set the ability to search and sort objects by this property.

Add a new sorting criterion for enumeration:

```
//+------------------------------------------------------------------+
//| Possible indicator sorting criteria                              |
//+------------------------------------------------------------------+
#define FIRST_INDICATOR_DBL_PROP          (INDICATOR_PROP_INTEGER_TOTAL-INDICATOR_PROP_INTEGER_SKIP)
#define FIRST_INDICATOR_STR_PROP          (INDICATOR_PROP_INTEGER_TOTAL-INDICATOR_PROP_INTEGER_SKIP+INDICATOR_PROP_DOUBLE_TOTAL-INDICATOR_PROP_DOUBLE_SKIP)
enum ENUM_SORT_INDICATOR_MODE
  {
//--- Sort by integer properties
   SORT_BY_INDICATOR_INDEX_STATUS = 0,                      // Sort by indicator status
   SORT_BY_INDICATOR_TYPE,                                  // Sort by indicator type
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

Slightly improve the abstract indicator object class in \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **IndicatorDE.mqh**.

Move the array of indicator parameters structures from the private class section to protected one (this array now will become available for descendant classes), and in private section declare a variable to store indicator type description:

```
//+------------------------------------------------------------------+
//| Abstract indicator class                                         |
//+------------------------------------------------------------------+
class CIndicatorDE : public CBaseObj
  {
protected:
   MqlParam          m_mql_param[];                                              // Array of indicator parameters
private:
   long              m_long_prop[INDICATOR_PROP_INTEGER_TOTAL];                  // Integer properties
   double            m_double_prop[INDICATOR_PROP_DOUBLE_TOTAL];                 // Real properties
   string            m_string_prop[INDICATOR_PROP_STRING_TOTAL];                 // String properties
   string            m_ind_type;                                                 // Indicator type description

```

To public class section add two methods — for return of indicator type and for return of indicator type description:

```
public:
//--- Default constructor
                     CIndicatorDE(void){;}
//--- Destructor
                    ~CIndicatorDE(void);

//--- Set buffer's (1) integer, (2) real and (3) string property
   void              SetProperty(ENUM_INDICATOR_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                                        }
   void              SetProperty(ENUM_INDICATOR_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;                      }
   void              SetProperty(ENUM_INDICATOR_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;                      }
//--- Return (1) integer, (2) real and (3) string buffer property from the properties array
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
//--- Compare CIndicatorDE objects by all properties (to search for equal indicator objects)
   bool              IsEqual(CIndicatorDE* compared_obj) const;

//--- Set indicator’s (1) group, (2) empty value of buffers, (3) name, (4) short name
   void              SetGroup(const ENUM_INDICATOR_GROUP group)      { this.SetProperty(INDICATOR_PROP_GROUP,group);                         }
   void              SetEmptyValue(const double value)               { this.SetProperty(INDICATOR_PROP_EMPTY_VALUE,value);                   }
   void              SetName(const string name)                      { this.SetProperty(INDICATOR_PROP_NAME,name);                           }
   void              SetShortName(const string shortname)            { this.SetProperty(INDICATOR_PROP_SHORTNAME,shortname);                 }

//--- Return indicator’s (1) status, (2) group, (3) timeframe, (4) handle, (5) empty value of buffers, (6) name, (7) short name, (8) symbol, (9) type
   ENUM_INDICATOR_STATUS Status(void)                          const { return (ENUM_INDICATOR_STATUS)this.GetProperty(INDICATOR_PROP_STATUS);}
   ENUM_INDICATOR_GROUP  Group(void)                           const { return (ENUM_INDICATOR_GROUP)this.GetProperty(INDICATOR_PROP_GROUP);  }
   ENUM_TIMEFRAMES   Timeframe(void)                           const { return (ENUM_TIMEFRAMES)this.GetProperty(INDICATOR_PROP_TIMEFRAME);   }
   ENUM_INDICATOR    TypeIndicator(void)                       const { return (ENUM_INDICATOR)this.GetProperty(INDICATOR_PROP_TYPE);         }
   int               Handle(void)                              const { return (int)this.GetProperty(INDICATOR_PROP_HANDLE);                  }
   double            EmptyValue(void)                          const { return this.GetProperty(INDICATOR_PROP_EMPTY_VALUE);                  }
   string            Name(void)                                const { return this.GetProperty(INDICATOR_PROP_NAME);                         }
   string            ShortName(void)                           const { return this.GetProperty(INDICATOR_PROP_SHORTNAME);                    }
   string            Symbol(void)                              const { return this.GetProperty(INDICATOR_PROP_SYMBOL);                       }

//--- Return description of indicator’s (1) type, (2) status, (3) group, (4) timeframe, (5) empty value
   string            GetTypeDescription(void)                  const { return m_ind_type;                                                    }
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

In closed parametric constructor retrieve substring from indicator type input to leave indicator name only (for example, only MACD remains from IND\_MACD which will be indicator type description) and write indicator type in object properties:

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
                           MqlParam &mql_params[])
  {
//--- Set collection ID to the object
   this.m_type=COLLECTION_INDICATORS_ID;
//--- Write description of indicator type
   this.m_ind_type=::StringSubstr(::EnumToString(ind_type),4);
//--- If parameter array size passed to constructor is more than zero
//--- fill in the array of object parameters with data from the array passed to constructor
   int count=::ArrayResize(m_mql_param,::ArraySize(mql_params));
   for(int i=0;i<count;i++)
     {
      this.m_mql_param[i].type=mql_params[i].type;
      this.m_mql_param[i].double_value=mql_params[i].double_value;
      this.m_mql_param[i].integer_value=mql_params[i].integer_value;
      this.m_mql_param[i].string_value=mql_params[i].string_value;
     }
//--- Create indicator handle
   int handle=::IndicatorCreate(symbol,timeframe,ind_type,count,this.m_mql_param);

//--- Save integer properties
   this.m_long_prop[INDICATOR_PROP_STATUS]                     = status;
   this.m_long_prop[INDICATOR_PROP_TYPE]                       = ind_type;
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

Add code block for return of indicator type description in method returning description of indicator integer property:

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
      property==INDICATOR_PROP_TYPE        ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTypeDescription()
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
```

### Indicator object classes

Now, create descendant objects of base abstract indicator which will specify all information on indicator object created. And those classes will serve to create different type indicators and to get data from them.

In directory \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ create a new folder **Standart**, and in it create a new file **IndAC.mqh** of CIndAC class of Accelerator Oscillator standard indicator, inherited from abstract indicator base classwhich file is connected to the listing of this class.

Since the class will be not big, let’s analyze its full listing:

```
//+------------------------------------------------------------------+
//|                                                        IndAC.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\\IndicatorDE.mqh"
//+------------------------------------------------------------------+
//| Standard indicator Accelerator Oscillator                        |
//+------------------------------------------------------------------+
class CIndAC : public CIndicatorDE
  {
private:

public:
   //--- Constructor
                     CIndAC(const string symbol,const ENUM_TIMEFRAMES timeframe,MqlParam &mql_param[]) :
                        CIndicatorDE(IND_AC,symbol,timeframe,
                                     INDICATOR_STATUS_STANDART,
                                     INDICATOR_GROUP_OSCILLATOR,
                                     "Accelerator Oscillator",
                                     "AC("+symbol+","+TimeframeDescription(timeframe)+")",mql_param) {}
   //--- Supported indicator properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_DOUBLE property);


//--- Display a short description of indicator object in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if indicator supports a passed                     |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CIndAC::SupportProperty(ENUM_INDICATOR_PROP_INTEGER property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if indicator supports a passed                     |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CIndAC::SupportProperty(ENUM_INDICATOR_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//--- Display a short description of indicator object in the journal |
//+------------------------------------------------------------------+
void CIndAC::PrintShort(void)
  {
   ::Print(GetStatusDescription()," ",this.Name()," ",this.Symbol()," ",TimeframeDescription(this.Timeframe()));
  }
//+------------------------------------------------------------------+
```

Totally, we must make 38 such classes - according to the number of standard indicators (I do not make custom indicator yet, since its implementation will slightly differ).

All these classes will have the same methods and they will differ only by the parameters being passed to the parent class from its constructor:

```
   //--- Constructor
                     CIndAC(const string symbol,const ENUM_TIMEFRAMES timeframe,MqlParam &mql_param[]) :
                        CIndicatorDE(IND_AC,symbol,timeframe,
                                     INDICATOR_STATUS_STANDART,
                                     INDICATOR_GROUP_OSCILLATOR,
                                     "Accelerator Oscillator",
                                     "AC("+symbol+","+TimeframeDescription(timeframe)+")",mql_param) {}
```

In class inputs I pass the name of the symbol, timeframe and already filled structure of indicator parameters (in this example - Accelerator Oscillator).

In initializing list pass to the closed parametric constructor of parent class (in order):

- indicator type — IND\_AC, symbol name, timeframe
- indicator status - standard
- indicator group - oscillator
- indicator name - Accelerator Oscillator
- indicator short name - AC (symbol, timeframe) and filled structure of indicator parameters


We already know all remaining methods by previous library objects we created and they perform the same tasks.

Methods, which return the flag of supporting the integer and real properties by the object, return true  — all properties are supported by indicator objects.

The method returning short indicator description returns the following string type:

```
Standard indicator Accelerator Oscillator EURUSD H4
```

In all remaining files of similar indicator objects classes the difference will be only in class constructor - to parent class constructor the parameters corresponding to the indicator will be passed.

For example, for indicator Accumulation/Distribution the class constructor will look as follows:

```
   //--- Constructor
                     CIndAD(const string symbol,const ENUM_TIMEFRAMES timeframe,MqlParam &mql_param[]) :
                        CIndicatorDE(IND_AD,symbol,timeframe,
                                     INDICATOR_STATUS_STANDART,
                                     INDICATOR_GROUP_VOLUMES,
                                     "Accumulation/Distribution",
                                     "AD("+symbol+","+TimeframeDescription(timeframe)+")",mql_param) {}
```

As we see, here parameters corresponding to standard indicator AD are passed:

- indicator type — IND\_AD, symbol name, timeframe
- indicator status - standard
- indicator group - volumes
- indicator name - Accumulation/Distribution
- indicator short name - AC (symbol, timeframe) and filled structure of indicator parameters


All files of descendant classes of abstract indicator base class are already implemented and they may be viewed in the files attached to the article in \\MQL5\\Include\\DoEasy\\Objects\\Indicators\\Standart folder.

### Indicator objects collection

In accordance with the general concept of library objects construction and storage now, we must put into collection list all the indicator objects created. From this list we can always get a pointer to the required indicator by specified properties or the list of indicators which have common same properties. By the received pointer to the indicator we will be able to take all data returned by the indicator for further calculations.

In folder \\MQL5\\Include\\DoEasy\ **Collections\** create a new class in the file named **IndicatorsCollection.mqh**.

Apart from the methods standard for the library which return required object lists from the collection, the class will contain one private method for creation of indicator object and multiple methods to create specific indicator objects in accordance with their type; as well as multiple methods to get pointers to created indicator objects also by their types. All methods of each group are identical to each other in accordance with their logic, therefore, consider only some of them as examples.

So that the collection class of indicators had access to indicator classes that we created above, they must be connected to file listing:

```
//+------------------------------------------------------------------+
//|                                         IndicatorsCollection.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Objects\Indicators\Standart\IndAC.mqh"
#include "..\Objects\Indicators\Standart\IndAD.mqh"
#include "..\Objects\Indicators\Standart\IndADX.mqh"
#include "..\Objects\Indicators\Standart\IndADXW.mqh"
#include "..\Objects\Indicators\Standart\IndAlligator.mqh"
#include "..\Objects\Indicators\Standart\IndAMA.mqh"
#include "..\Objects\Indicators\Standart\IndAO.mqh"
#include "..\Objects\Indicators\Standart\IndATR.mqh"
#include "..\Objects\Indicators\Standart\IndBands.mqh"
#include "..\Objects\Indicators\Standart\IndBears.mqh"
#include "..\Objects\Indicators\Standart\IndBulls.mqh"
#include "..\Objects\Indicators\Standart\IndBWMFI.mqh"
#include "..\Objects\Indicators\Standart\IndCCI.mqh"
#include "..\Objects\Indicators\Standart\IndChaikin.mqh"
#include "..\Objects\Indicators\Standart\IndDEMA.mqh"
#include "..\Objects\Indicators\Standart\IndDeMarker.mqh"
#include "..\Objects\Indicators\Standart\IndEnvelopes.mqh"
#include "..\Objects\Indicators\Standart\IndForce.mqh"
#include "..\Objects\Indicators\Standart\IndFractals.mqh"
#include "..\Objects\Indicators\Standart\IndFRAMA.mqh"
#include "..\Objects\Indicators\Standart\IndGator.mqh"
#include "..\Objects\Indicators\Standart\IndIchimoku.mqh"
#include "..\Objects\Indicators\Standart\IndMA.mqh"
#include "..\Objects\Indicators\Standart\IndMACD.mqh"
#include "..\Objects\Indicators\Standart\IndMFI.mqh"
#include "..\Objects\Indicators\Standart\IndMomentum.mqh"
#include "..\Objects\Indicators\Standart\IndOBV.mqh"
#include "..\Objects\Indicators\Standart\IndOsMA.mqh"
#include "..\Objects\Indicators\Standart\IndRSI.mqh"
#include "..\Objects\Indicators\Standart\IndRVI.mqh"
#include "..\Objects\Indicators\Standart\IndSAR.mqh"
#include "..\Objects\Indicators\Standart\IndStDev.mqh"
#include "..\Objects\Indicators\Standart\IndStoch.mqh"
#include "..\Objects\Indicators\Standart\IndTEMA.mqh"
#include "..\Objects\Indicators\Standart\IndTRIX.mqh"
#include "..\Objects\Indicators\Standart\IndVIDYA.mqh"
#include "..\Objects\Indicators\Standart\IndVolumes.mqh"
#include "..\Objects\Indicators\Standart\IndWPR.mqh"
//+------------------------------------------------------------------+
```

Further, have a look at the full listing of the class body and then analyze two methods of each group.

```
//+------------------------------------------------------------------+
//| Indicator collection                                             |
//+------------------------------------------------------------------+
class CIndicatorsCollection : public CObject
  {
private:
   CListObj                m_list;                       // Indicator object list
   MqlParam                m_mql_param[];                // Array of indicator parameters

//--- Create a new indicator object
   CIndicatorDE           *CreateIndicator(const ENUM_INDICATOR ind_type,MqlParam &mql_param[],const string symbol_name=NULL,const ENUM_TIMEFRAMES period=PERIOD_CURRENT);

public:
//--- Return (1) itself, (2) indicator list, (3) list of indicators by type
   CIndicatorsCollection  *GetObject(void)               { return &this;                                       }
   CArrayObj              *GetList(void)                 { return &this.m_list;                                }
//--- Return indicator list by (1) status, (2) type, (3) timeframe, (4) group, (5) symbol, (6) name, (7) short name
   CArrayObj              *GetListIndByStatus(const ENUM_INDICATOR_STATUS status)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_STATUS,status,EQUAL);        }
   CArrayObj              *GetListIndByType(const ENUM_INDICATOR type)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_TYPE,type,EQUAL);            }
   CArrayObj              *GetListIndByTimeframe(const ENUM_TIMEFRAMES timeframe)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_TIMEFRAME,timeframe,EQUAL);  }
   CArrayObj              *GetListIndByGroup(const ENUM_INDICATOR_GROUP group)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_GROUP,group,EQUAL);          }
   CArrayObj              *GetListIndBySymbol(const string symbol)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_SYMBOL,symbol,EQUAL);        }
   CArrayObj              *GetListIndByName(const string name)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_NAME,name,EQUAL);            }
   CArrayObj              *GetListIndByShortname(const string shortname)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_SHORTNAME,shortname,EQUAL);  }

//--- Return the list of indicator objects by type of indicator, symbol and timeframe
   CArrayObj              *GetListAC(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListAD(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListADX(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListADXWilder(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListAlligator(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListAMA(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListAO(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListATR(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListBands(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListBearsPower(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListBullsPower(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListChaikin(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListCCI(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListDEMA(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListDeMarker(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListEnvelopes(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListForce(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListFractals(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListFrAMA(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListGator(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListIchimoku(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListBWMFI(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListMomentum(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListMFI(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListMA(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListOsMA(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListMACD(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListOBV(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListSAR(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListRSI(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListRVI(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListStdDev(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListStochastic(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListTEMA(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListTriX(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListWPR(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListVIDYA(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CArrayObj              *GetListVolumes(const string symbol,const ENUM_TIMEFRAMES timeframe);

//--- Return the pointer to indicator object in the collection by indicator type and by its parameters
   CIndicatorDE           *GetIndAC(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CIndicatorDE           *GetIndAD(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume);
   CIndicatorDE           *GetIndADX(const string symbol,const ENUM_TIMEFRAMES timeframe,const int adx_period);
   CIndicatorDE           *GetIndADXWilder(const string symbol,const ENUM_TIMEFRAMES timeframe,const int adx_period);
   CIndicatorDE           *GetIndAlligator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int jaw_period,
                                       const int jaw_shift,
                                       const int teeth_period,
                                       const int teeth_shift,
                                       const int lips_period,
                                       const int lips_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndAMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ama_period,
                                       const int fast_ema_period,
                                       const int slow_ema_period,
                                       const int ama_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndAO(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CIndicatorDE           *GetIndATR(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   CIndicatorDE           *GetIndBands(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const double deviation,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndBearsPower(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   CIndicatorDE           *GetIndBullsPower(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   CIndicatorDE           *GetIndChaikin(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ma_period,
                                       const int slow_ma_period,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   CIndicatorDE           *GetIndCCI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndDEMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndDeMarker(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   CIndicatorDE           *GetIndEnvelopes(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const double deviation);
   CIndicatorDE           *GetIndForce(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   CIndicatorDE           *GetIndFractals(const string symbol,const ENUM_TIMEFRAMES timeframe);
   CIndicatorDE           *GetIndFrAMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndGator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int jaw_period,
                                       const int jaw_shift,
                                       const int teeth_period,
                                       const int teeth_shift,
                                       const int lips_period,
                                       const int lips_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndIchimoku(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int tenkan_sen,
                                       const int kijun_sen,
                                       const int senkou_span_b);
   CIndicatorDE           *GetIndBWMFI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   CIndicatorDE           *GetIndMomentum(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int mom_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndMFI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   CIndicatorDE           *GetIndMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndOsMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ema_period,
                                       const int slow_ema_period,
                                       const int signal_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndMACD(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ema_period,
                                       const int slow_ema_period,
                                       const int signal_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndOBV(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   CIndicatorDE           *GetIndSAR(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const double step,
                                       const double maximum);
   CIndicatorDE           *GetIndRSI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndRVI(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   CIndicatorDE           *GetIndStdDev(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndStochastic(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int Kperiod,
                                       const int Dperiod,
                                       const int slowing,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_STO_PRICE price_field);
   CIndicatorDE           *GetIndTEMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndTriX(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndWPR(const string symbol,const ENUM_TIMEFRAMES timeframe,const int calc_period);
   CIndicatorDE           *GetIndVIDYA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int cmo_period,
                                       const int ema_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   CIndicatorDE           *GetIndVolumes(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume);

//--- Create a new indicator object by indicator type and places it to collection list
   int                     CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe);
   int                     CreateAD(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume);
   int                     CreateADX(const string symbol,const ENUM_TIMEFRAMES timeframe,const int adx_period);
   int                     CreateADXWilder(const string symbol,const ENUM_TIMEFRAMES timeframe,const int adx_period);
   int                     CreateAlligator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int jaw_period,
                                       const int jaw_shift,
                                       const int teeth_period,
                                       const int teeth_shift,
                                       const int lips_period,
                                       const int lips_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateAMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ama_period,
                                       const int fast_ema_period,
                                       const int slow_ema_period,
                                       const int ama_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateAO(const string symbol,const ENUM_TIMEFRAMES timeframe);
   int                     CreateATR(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   int                     CreateBands(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const double deviation,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateBearsPower(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   int                     CreateBullsPower(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   int                     CreateChaikin(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ma_period,
                                       const int slow_ma_period,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   int                     CreateCCI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateDEMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateDeMarker(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   int                     CreateEnvelopes(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price,
                                       const double deviation);
   int                     CreateForce(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   int                     CreateFractals(const string symbol,const ENUM_TIMEFRAMES timeframe);
   int                     CreateFrAMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateGator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int jaw_period,
                                       const int jaw_shift,
                                       const int teeth_period,
                                       const int teeth_shift,
                                       const int lips_period,
                                       const int lips_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateIchimoku(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int tenkan_sen,
                                       const int kijun_sen,
                                       const int senkou_span_b);
   int                     CreateBWMFI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   int                     CreateMomentum(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int mom_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateMFI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   int                     CreateMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateOsMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ema_period,
                                       const int slow_ema_period,
                                       const int signal_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateMACD(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int fast_ema_period,
                                       const int slow_ema_period,
                                       const int signal_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateOBV(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const ENUM_APPLIED_VOLUME applied_volume);
   int                     CreateSAR(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const double step,
                                       const double maximum);
   int                     CreateRSI(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateRVI(const string symbol,const ENUM_TIMEFRAMES timeframe,const int ma_period);
   int                     CreateStdDev(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateStochastic(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int Kperiod,
                                       const int Dperiod,
                                       const int slowing,
                                       const ENUM_MA_METHOD ma_method,
                                       const ENUM_STO_PRICE price_field);
   int                     CreateTEMA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateTriX(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int ma_period,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateWPR(const string symbol,const ENUM_TIMEFRAMES timeframe,const int calc_period);
   int                     CreateVIDYA(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int cmo_period,
                                       const int ema_period,
                                       const int ma_shift,
                                       const ENUM_APPLIED_PRICE applied_price);
   int                     CreateVolumes(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume);

//--- Constructor
                           CIndicatorsCollection();

  };
//+------------------------------------------------------------------+
```

Listing seems impressive, but, in fact, we have here only several groups of similar-type methods which differ from each other by the type of required indicator.

The methods which return lists with required indicator types are standard for the library and we analyzed them repeatedly:

```
//--- Return (1) itself, (2) indicator list, (3) list of indicators by type
   CIndicatorsCollection  *GetObject(void)               { return &this;                                       }
   CArrayObj              *GetList(void)                 { return &this.m_list;                                }
//--- Return indicator list by (1) status, (2) type, (3) timeframe, (4) group, (5) symbol, (6) name, (7) short name
   CArrayObj              *GetListIndByStatus(const ENUM_INDICATOR_STATUS status)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_STATUS,status,EQUAL);        }
   CArrayObj              *GetListIndByType(const ENUM_INDICATOR type)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_TYPE,type,EQUAL);            }
   CArrayObj              *GetListIndByTimeframe(const ENUM_TIMEFRAMES timeframe)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_TIMEFRAME,timeframe,EQUAL);  }
   CArrayObj              *GetListIndByGroup(const ENUM_INDICATOR_GROUP group)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_GROUP,group,EQUAL);          }
   CArrayObj              *GetListIndBySymbol(const string symbol)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_SYMBOL,symbol,EQUAL);        }
   CArrayObj              *GetListIndByName(const string name)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_NAME,name,EQUAL);            }
   CArrayObj              *GetListIndByShortname(const string shortname)
                             { return CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_SHORTNAME,shortname,EQUAL);  }
```

In class constructor reset the array of indicator parameter structures, clear collection list, set sorted list flag for the list and assign indicator collection ID to it:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CIndicatorsCollection::CIndicatorsCollection()
  {
   ::ArrayResize(this.m_mql_param,0);
   this.m_list.Clear();
   this.m_list.Sort();
   this.m_list.Type(COLLECTION_INDICATORS_ID);
  }
//+------------------------------------------------------------------+
```

The private method for creation of indicator object simply creates a new indicator object depending on the type of indicator created passed to the method and return the pointer to created object:

```
//+------------------------------------------------------------------+
//| Create a new indicator object                                    |
//+------------------------------------------------------------------+
CIndicatorDE *CIndicatorsCollection::CreateIndicator(const ENUM_INDICATOR ind_type,MqlParam &mql_param[],const string symbol_name=NULL,const ENUM_TIMEFRAMES period=PERIOD_CURRENT)
  {
   string symbol=(symbol_name==NULL || symbol_name=="" ? ::Symbol() : symbol_name);
   ENUM_TIMEFRAMES timeframe=(period==PERIOD_CURRENT ? ::Period() : period);
   CIndicatorDE *indicator=NULL;
   switch(ind_type)
     {
      case IND_AC          : indicator=new CIndAC(symbol,timeframe,mql_param);         break;
      case IND_AD          : indicator=new CIndAD(symbol,timeframe,mql_param);         break;
      case IND_ADX         : indicator=new CIndADX(symbol,timeframe,mql_param);        break;
      case IND_ADXW        : indicator=new CIndADXW(symbol,timeframe,mql_param);       break;
      case IND_ALLIGATOR   : indicator=new CIndAlligator(symbol,timeframe,mql_param);  break;
      case IND_AMA         : indicator=new CIndAMA(symbol,timeframe,mql_param);        break;
      case IND_AO          : indicator=new CIndAO(symbol,timeframe,mql_param);         break;
      case IND_ATR         : indicator=new CIndATR(symbol,timeframe,mql_param);        break;
      case IND_BANDS       : indicator=new CIndBands(symbol,timeframe,mql_param);      break;
      case IND_BEARS       : indicator=new CIndBears(symbol,timeframe,mql_param);      break;
      case IND_BULLS       : indicator=new CIndBulls(symbol,timeframe,mql_param);      break;
      case IND_BWMFI       : indicator=new CIndBWMFI(symbol,timeframe,mql_param);      break;
      case IND_CCI         : indicator=new CIndCCI(symbol,timeframe,mql_param);        break;
      case IND_CHAIKIN     : indicator=new CIndCHO(symbol,timeframe,mql_param);        break;
      case IND_DEMA        : indicator=new CIndDEMA(symbol,timeframe,mql_param);       break;
      case IND_DEMARKER    : indicator=new CIndDeMarker(symbol,timeframe,mql_param);   break;
      case IND_ENVELOPES   : indicator=new CIndEnvelopes(symbol,timeframe,mql_param);  break;
      case IND_FORCE       : indicator=new CIndForce(symbol,timeframe,mql_param);      break;
      case IND_FRACTALS    : indicator=new CIndFractals(symbol,timeframe,mql_param);   break;
      case IND_FRAMA       : indicator=new CIndFRAMA(symbol,timeframe,mql_param);      break;
      case IND_GATOR       : indicator=new CIndGator(symbol,timeframe,mql_param);      break;
      case IND_ICHIMOKU    : indicator=new CIndIchimoku(symbol,timeframe,mql_param);   break;
      case IND_MA          : indicator=new CIndMA(symbol,timeframe,mql_param);         break;
      case IND_MACD        : indicator=new CIndMACD(symbol,timeframe,mql_param);       break;
      case IND_MFI         : indicator=new CIndMFI(symbol,timeframe,mql_param);        break;
      case IND_MOMENTUM    : indicator=new CIndMomentum(symbol,timeframe,mql_param);   break;
      case IND_OBV         : indicator=new CIndOBV(symbol,timeframe,mql_param);        break;
      case IND_OSMA        : indicator=new CIndOsMA(symbol,timeframe,mql_param);       break;
      case IND_RSI         : indicator=new CIndRSI(symbol,timeframe,mql_param);        break;
      case IND_RVI         : indicator=new CIndRVI(symbol,timeframe,mql_param);        break;
      case IND_SAR         : indicator=new CIndSAR(symbol,timeframe,mql_param);        break;
      case IND_STDDEV      : indicator=new CIndStDev(symbol,timeframe,mql_param);      break;
      case IND_STOCHASTIC  : indicator=new CIndStoch(symbol,timeframe,mql_param);      break;
      case IND_TEMA        : indicator=new CIndTEMA(symbol,timeframe,mql_param);       break;
      case IND_TRIX        : indicator=new CIndTRIX(symbol,timeframe,mql_param);       break;
      case IND_VIDYA       : indicator=new CIndVIDYA(symbol,timeframe,mql_param);      break;
      case IND_VOLUMES     : indicator=new CIndVolumes(symbol,timeframe,mql_param);    break;
      case IND_WPR         : indicator=new CIndWPR(symbol,timeframe,mql_param);        break;

      case IND_CUSTOM : break;
      default: break;
     }
   return indicator;
  }
//+------------------------------------------------------------------+
```

By default, as well temporarily and for custom indicator (its creation will be implemented in following articles) the method returns NULL.

**The method creating indicator object Accelerator Oscillator:**

```
//+------------------------------------------------------------------+
//| Create a new indicator object Accelerator Oscillator             |
//| and place it to the collection list                              |
//+------------------------------------------------------------------+
int CIndicatorsCollection::CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
//--- AC indicator possesses no parameters - resize the array of parameter structures
   ::ArrayResize(this.m_mql_param,0);
//--- Create indicator object
   CIndicatorDE *indicator=this.CreateIndicator(IND_AC,this.m_mql_param,symbol,timeframe);
   if(indicator==NULL)
      return INVALID_HANDLE;
   int index=this.m_list.Search(indicator);
//--- If such indicator is already in the list
   if(index!=WRONG_VALUE)
     {
      //--- Get indicator object from the list and return indicator handle
      indicator=this.m_list.At(index);
      return indicator.Handle();
     }
//--- If such indicator is not in the list
   else
     {
      //--- If failed to add indicator object to the list
      //--- display the appropriate message and return INVALID_HANDLE
      if(!this.m_list.Add(indicator))
        {
         Print(CMessage::Text(MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST));
         return INVALID_HANDLE;
        }
      //--- Return the handle of a new indicator added to the list
      return indicator.Handle();
     }
//--- Return INVALID_HANDLE
   return INVALID_HANDLE;
  }
//+------------------------------------------------------------------+
```

The method logic is described in its listing and it must not cause questions. Note that indicator object is created in the method of its creation CreateIndicator(), analyzed above. The method receives a type of IND\_AC indicator. Since AC indicator possesses no inputs the array of indicator parameter structures is not required here. Therefore, the array is reset which points that we do not need to use it when creating an indicator in CIndicatorDE class which we considered [in the previous article](https://www.mql5.com/en/articles/8464#node03).

Remaining methods for creation of other standard indicators are identical to that just analyzed in their logic and they differ only by specification of the required indicator and by filling of the array of indicator parameter structures where it is necessary.

As example, consider the method of Alligator indicator creation where eight inputs are required and all of them are added to the array of indicator parameters in accordance with the order of consequence of those parameters with Alligator indicator and IND\_ALLIGATOR type is passed to the method of indicator object creation:

```
//+------------------------------------------------------------------+
//| Create new indicator object Alligator                            |
//| and place it to the collection list                              |
//+------------------------------------------------------------------+
int CIndicatorsCollection::CreateAlligator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                           const int jaw_period,
                                           const int jaw_shift,
                                           const int teeth_period,
                                           const int teeth_shift,
                                           const int lips_period,
                                           const int lips_shift,
                                           const ENUM_MA_METHOD ma_method,
                                           const ENUM_APPLIED_PRICE applied_price)
  {
//--- Add required indicator parameters to the array of parameter structures
   ::ArrayResize(this.m_mql_param,8);
   this.m_mql_param[0].type=TYPE_INT;
   this.m_mql_param[0].integer_value=jaw_period;
   this.m_mql_param[1].type=TYPE_INT;
   this.m_mql_param[1].integer_value=jaw_shift;
   this.m_mql_param[2].type=TYPE_INT;
   this.m_mql_param[2].integer_value=teeth_period;
   this.m_mql_param[3].type=TYPE_INT;
   this.m_mql_param[3].integer_value=teeth_shift;
   this.m_mql_param[4].type=TYPE_INT;
   this.m_mql_param[4].integer_value=lips_period;
   this.m_mql_param[5].type=TYPE_INT;
   this.m_mql_param[5].integer_value=lips_shift;
   this.m_mql_param[6].type=TYPE_INT;
   this.m_mql_param[6].integer_value=ma_method;
   this.m_mql_param[7].type=TYPE_INT;
   this.m_mql_param[7].integer_value=applied_price;
//--- Create indicator object
   CIndicatorDE *indicator=this.CreateIndicator(IND_ALLIGATOR,this.m_mql_param,symbol,timeframe);
   if(indicator==NULL)
      return INVALID_HANDLE;
   int index=this.m_list.Search(indicator);
//--- If such indicator is already in the list
   if(index!=WRONG_VALUE)
     {
      //--- Get indicator object from the list and return indicator handle
      indicator=this.m_list.At(index);
      return indicator.Handle();
     }
//--- If such indicator is not in the list
   else
     {
      //--- If failed to add indicator object to the list
      //--- display the appropriate message and return INVALID_HANDLE
      if(!this.m_list.Add(indicator))
        {
         Print(CMessage::Text(MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST));
         return INVALID_HANDLE;
        }
      //--- Return the handle of a new indicator added to the list
      return indicator.Handle();
     }
//--- Return INVALID_HANDLE
   return INVALID_HANDLE;
  }
//+------------------------------------------------------------------+
```

These are all differences of each method from each other. Remaining methods for indicator creation will not be analyzed. They are available in the files attached to the article.

**The method returning the list of all indicator objects Accelerator Oscillator which are in the collection list, by symbol and timeframe:**

```
//+------------------------------------------------------------------+
//| Return the list of indicator objects Accelerator Oscillator      |
//| by symbol and timeframe                                          |
//+------------------------------------------------------------------+
CArrayObj *CIndicatorsCollection::GetListAC(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   CArrayObj *list=GetListIndByType(IND_AC);
   list=CSelect::ByIndicatorProperty(list,INDICATOR_PROP_SYMBOL,symbol,EQUAL);
   return CSelect::ByIndicatorProperty(list,INDICATOR_PROP_TIMEFRAME,timeframe,EQUAL);
  }
//+------------------------------------------------------------------+
```

First, get the list of all indicator objects Accelerator Oscillator which are in the collection list,

then sort the received list by symbol

and return the received list which is sorted once again, this time by timeframe.

Remaining methods are fully identical to the above considered, save for the fact that first, we get the list of all required indicators in accordance with method setting.

For example, to get the list of all indicator objects Accumulation/Distribution which are in the collection list, by symbol and timeframe, the method will be as follows:

```
//+------------------------------------------------------------------+
//| Return the list of indicator objects  Accumulation/Distribution  |
//| by symbol and timeframe                                          |
//+------------------------------------------------------------------+
CArrayObj *CIndicatorsCollection::GetListAD(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   CArrayObj *list=GetListIndByType(IND_AD);
   list=CSelect::ByIndicatorProperty(list,INDICATOR_PROP_SYMBOL,symbol,EQUAL);
   return CSelect::ByIndicatorProperty(list,INDICATOR_PROP_TIMEFRAME,timeframe,EQUAL);
  }
//+------------------------------------------------------------------+
```

Here, first get the list of all AD indicators, being in the collection and further, sort the list by symbol and timeframe as in the above considered method.

Today implement only one **method which returns the pointer to indicator object in the collection by indicator type and by its parameters** \- for Accelerator Oscillator indicator:

```
//+------------------------------------------------------------------+
//| Return pointer to indicator object Accelerator Oscillator        |
//+------------------------------------------------------------------+
CIndicatorDE *CIndicatorsCollection::GetIndAC(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   CArrayObj *list=GetListAC(symbol,timeframe);
   return(list==NULL || list.Total()==0 ? NULL : list.At(0));
  }
//+------------------------------------------------------------------+
```

Since this indicator (and some other) possesses no inputs, when selecting it from collection list by symbol and timeframe, the list will contain only one indicator object (by index 0), which corresponds to type IND\_AC and to requested symbol and timeframe.

Whereas, the indicators which possess inputs require additional search methods by the array of indicator parameter structures to implement their search by specified parameters. This will be beyond the size of this article. Therefore, such methods will be analyzed in the next article.

I will test only creation of one indicator object Accelerator Oscillator since this article has rather training purposes and doesn’t claim to be complete. In the previous article I already performed such test on creation of AC indicator object in collection class of buffers:

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
```

Today, I will do the same thing: in the same method create Accelerator Oscillator indicator object, but this time using newly added classes.

Looking ahead: we must “see” collection list in collection class of timeseries so that storage function for data of all indicators created in bar objects which “lie” in timeseries lists, open for us.

Therefore, preliminary we need to include collection class of indicators to timeseries collection class in file \\MQL5\\Include\\DoEasy\\Collections\ **TimeSeriesCollection.mqh**:

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
#include "ListObj.mqh"
#include "..\Objects\Series\TimeSeriesDE.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "IndicatorsCollection.mqh"
//+------------------------------------------------------------------+
```

Further, in the private section of the class declare pointer to collection class of indicators:

```
//+------------------------------------------------------------------+
//| Symbol timeseries collection                                     |
//+------------------------------------------------------------------+
class CTimeSeriesCollection : public CBaseObjExt
  {
private:
   CListObj                m_list;                    // List of applied symbol timeseries
   CIndicatorsCollection  *m_indicators;              // Pointer to collection object of indicators
//--- Return the timeseries index by symbol name
   int                     IndexTimeSeries(const string symbol);
public:
```

And in the end of class listing create initializing method through which the pointer to indicator collection object will be passed to the class in the main object of CEngine library:

```
//--- Constructor
                           CTimeSeriesCollection();
//--- Get pointers to the indicator collection (the method is called in CollectionOnInit() method of the CEngine object)
   void                    OnInit(CIndicatorsCollection *indicators) { this.m_indicators=indicators;  }
  };
//+------------------------------------------------------------------+
```

In file of buffer collection class \\MQL5\\Include\\DoEasy\\Collections\ **BuffersCollection.mqh** also declare the pointer to indicator collection object:

```
//+------------------------------------------------------------------+
//| Collection of indicator buffers                                  |
//+------------------------------------------------------------------+
class CBuffersCollection : public CObject
  {
private:
   CListObj                m_list;                       // Buffer object list
   CTimeSeriesCollection  *m_timeseries;                 // Pointer to the timeseries collection object
   CIndicatorsCollection  *m_indicators;                 // Pointer to collection object of indicators
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

To method of OnInit() class add setting for this pointer a value being passed by input parameter:

```
//--- Constructor
                           CBuffersCollection();
//--- Get pointers to collections of timeseries and indicators (the method is called in CollectionOnInit() method of the CEngine object)
   void                    OnInit(CTimeSeriesCollection *timeseries,CIndicatorsCollection *indicators)
                             { this.m_timeseries=timeseries; this.m_indicators=indicators;   }
  };
//+------------------------------------------------------------------+
```

And in the method of creation of AC indicator change creation of indicator object made in the previous article to its creation and getting with the use of collection class of indicators:

```
//+------------------------------------------------------------------+
//| Create multi-symbol multi-period AC                              |
//+------------------------------------------------------------------+
int CBuffersCollection::CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id=WRONG_VALUE)
  {
//--- To check it, create indicator object, print its data and remove it at once
//--- Parameters are not needed for AC, therefore, reset the array of indicator parameter structures
   ::ArrayResize(this.m_mql_param,0);
//--- Create AC indicator and add it to collection
   this.m_indicators.CreateAC(symbol,timeframe);
//--- Get from collection of AC indicator object
   CIndicatorDE *indicator=this.m_indicators.GetIndAC(symbol,timeframe);
//--- Display all data of the created indicator in the journal, display its short description and remove indicator object
   indicator.Print();
   indicator.PrintShort();
   delete indicator;

//--- Create indicator handle and set default ID
```

Now let's improve the CEngine class of library main object.

In file \\MQL5\\Include\\DoEasy\ **Engine.mqh** add inclusion of indicator collection file to class listing:

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
#include "Collections\BuffersCollection.mqh"
#include "Collections\IndicatorsCollection.mqh"
#include "TradingControl.mqh"
//+------------------------------------------------------------------+
```

In the private section of the class declare object of indicator collection class:

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
   CTimeSeriesCollection m_time_series;                  // Timeseries collection
   CBuffersCollection   m_buffers;                       // Collection of indicator buffers
   CIndicatorsCollection m_indicators;                   // Indicator collection
   CResourceCollection  m_resource;                      // Resource list
   CTradingControl      m_trading;                       // Trading control object
   CPause               m_pause;                         // Pause object
   CArrayObj            m_list_counters;                 // List of timer counters
   int                  m_global_error;                  // Global error code
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_tester;                     // Flag of working in the tester
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   bool                 m_is_account_event;              // Account change event flag
   bool                 m_is_symbol_event;               // Symbol change event flag
   ENUM_TRADE_EVENT     m_last_trade_event;              // Last account trading event
   int                  m_last_account_event;            // Last event in the account properties
   int                  m_last_symbol_event;             // Last event in the symbol properties
   ENUM_PROGRAM_TYPE    m_program;                       // Program type
   string               m_name;                          // Program name
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return the first launch flag
   bool                 IsFirstStart(void);
//--- Work with (1) order, deal and position, (2) account events
   void                 TradeEventsControl(void);
   void                 AccountEventsControl(void);
//--- (1) Working with a symbol collection and (2) symbol list events in the market watch window
   void                 SymbolEventsControl(void);
   void                 MarketWatchEventsControl(void);
//--- Return the last (1) market pending order, (2) market order, (3) last position, (4) position by ticket
   COrder              *GetLastMarketPending(void);
   COrder              *GetLastMarketOrder(void);
   COrder              *GetLastPosition(void);
   COrder              *GetPosition(const ulong ticket);
//--- Return the last (1) removed pending order, (2) historical market order, (3) historical order (market or pending one) by its ticket
   COrder              *GetLastHistoryPending(void);
   COrder              *GetLastHistoryOrder(void);
   COrder              *GetHistoryOrder(const ulong ticket);
//--- Return the (1) first and the (2) last historical market orders from the list of all position orders, (3) the last deal
   COrder              *GetFirstOrderPosition(const ulong position_id);
   COrder              *GetLastOrderPosition(const ulong position_id);
   COrder              *GetLastDeal(void);
//--- Retrieve a necessary 'ushort' number from the packed 'long' value
   ushort               LongToUshortFromByte(const long source_value,const uchar index) const;

public:
```

In public section write two methods to return pointer to indicator collection objects and pointer to indicator list of this collection:

```
//--- Return the bar index on the specified timeframe chart by the current chart's bar index
   int                  IndexBarPeriodByBarCurrent(const int series_index,const string symbol,const ENUM_TIMEFRAMES timeframe)
                          { return this.m_time_series.IndexBarPeriodByBarCurrent(series_index,symbol,timeframe);  }

//--- Return (1) the indicator collection, (2) the indicator list from the collection
   CIndicatorsCollection *GetIndicatorsCollection(void)                                { return &this.m_indicators;           }
   CArrayObj           *GetListIndicators(void)                                        { return this.m_indicators.GetList();  }
```

These methods will be useful to us for calling indicator collection from our programs.

Add to method CollectionOnInit() passing of the pointer to indicator collection to collection classes of buffersand timeseries:

```
//--- Pass the pointers to all the necessary collections to the trading class and the indicator buffer collection class
   void                 CollectionOnInit(void)
                          {
                           this.m_trading.OnInit(this.GetAccountCurrent(),m_symbols.GetObject(),m_market.GetObject(),m_history.GetObject(),m_events.GetObject());
                           this.m_buffers.OnInit(this.m_time_series.GetObject(),this.m_indicators.GetObject());
                           this.m_time_series.OnInit(this.m_indicators.GetObject());
                          }
```

Now, in case of initializing of library the pointer to indicator collection object will be passed to all classes where access to indicator collection is necessary. They will be able to work with this collection.

**For now, these are all improvements necessary to create indicator collection class.**

### Testing

For testing we need the [indicator from the previous article](https://www.mql5.com/en/articles/8464#node04) without any changes.

Simply save it in a new folder \\MQL5\\Indicators\\TestDoEasy\ **Part54\** under a new name **TestDoEasyPart54.mq5**.

Compile the indicator and launch it on the chart.

The following will be displayed in the journal: all parameters of created indicator Accelerator Oscillator in full , and then its short description:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10425.23 USD, 1:100, Hedge, Demo account MetaTrader 5
--- Initializing "DoEasy" library ---
Working with the current symbol only. Number of used symbols: 1
"EURUSD"
Working with the specified timeframe list:
"H4" "H1"
EURUSD symbol timeseries:
- "EURUSD" H1 timeseries: Requested: 1000, Actually: 0, Created: 0, On the server: 0
- "EURUSD" H4 timeseries: Requested: 1000, Actually: 1000, Created: 1000, On the server: 6231
Time of library initializing: 00:00:00.156

============= Beginning of the parameter list: "Standard indicator" =============
Indicator status: Standard indicator
Indicator type: AC
Indicator timeframe: H4
Indicator handle: 10
Indicator group: Oscillator
------
Empty value for plotting where nothing will be drawn: EMPTY_VALUE
------
Indicator symbol: EURUSD
Indicator name: "Accelerator Oscillator"
Indicator short name: "AC(EURUSD,H4)"
================== End of the parameter list: "Standard indicator" ==================

Standard indicator Accelerator Oscillator EURUSD H4
Buffer (P0/B0/C1): Histogram from zero line EURUSD H4
Buffer [P0/B2/C2]: Calculated buffer
"EURUSD" H1 timeseries created successfully:
- "EURUSD" H1 timeseries: Requested: 1000, Actually: 1000, Created: 1000, On the server: 6256
```

### What's next?

In the next article we will continue working on indicator collection class.

All files of the current version of the library are attached below together with the test indicator file for MQL5. You can download them and test everything.

Note, that at the moment indicator collection class is under development, therefore  it is strictly recommended not to use it in your programs.

Leave your comments, questions and suggestions in the comments to the article.

[Back to contents](https://www.mql5.com/en/articles/8508#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8508](https://www.mql5.com/ru/articles/8508)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8508.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8508/mql5.zip "Download MQL5.zip")(3835.27 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/358525)**
(8)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
24 Oct 2020 at 17:08

**Dima Diall:**

Could you please show an example?

From article 40 onwards


![Alexey Viktorov](https://c.mql5.com/avatar/2017/4/58E3DFDD-D3B2.jpg)

**[Alexey Viktorov](https://www.mql5.com/en/users/alexeyvik)**
\|
24 Oct 2020 at 17:50

**Artyom Trishkin:**

From Article 40 onwards

Why so little? You have to start from the first one. Otherwise there may be misunderstandings.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
24 Oct 2020 at 20:26

**Alexey Viktorov:**

Why so little? You have to start from the first one. Otherwise, there may be misunderstandings.

This man has studied the entire library thoroughly. Unlike you, my friend.)

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
25 Oct 2020 at 20:14

**Artyom Trishkin:**

From article 40 onwards

Maybe I misunderstood, but for now I don't want to write my own indicator using the DoEasy library (from article 40 onwards)... I meant creating an indicator object (as shown in articles 53-55) that loads an existing indicator of IND\_CUSTOM type, for example **Indicators/ZigZag.mq4** or **Indicators/Examples/ZigZag.mq5**; or even any indicator available [at](https://www.mql5.com/en/code/mt4/indicators) https://www.mql5.com/en/code/mt4/indicators [or https://www.mql5.com/en/code/mt5/indicators](https://www.mql5.com/en/code/mt5/indicators).

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
25 Oct 2020 at 23:45

**Dima Diall:**

Maybe I misunderstood, but for now I don't want to write my own indicator using DoEasy library (starting from article 40)... I meant creating an indicator object (as shown in articles 53-55) that loads an existing indicator of IND\_CUSTOM type, for example **Indicators/ZigZag.mq4** or **Indicators/Examples/ZigZag.mq5**; or even any indicator available [at](https://www.mql5.com/en/code/mt4/indicators) https://www.mql5.com/en/code/mt4/indicators [or https://www.mql5.com/en/code/mt5/indicators](https://www.mql5.com/en/code/mt5/indicators).

Roger. Yes, it will be in the next article (#56).

![Neural networks made easy (Part 4): Recurrent networks](https://c.mql5.com/2/48/Neural_networks_made_easy_004.png)[Neural networks made easy (Part 4): Recurrent networks](https://www.mql5.com/en/articles/8385)

We continue studying the world of neural networks. In this article, we will consider another type of neural networks, recurrent networks. This type is proposed for use with time series, which are represented in the MetaTrader 5 trading platform by price charts.

![Grid and martingale: what are they and how to use them?](https://c.mql5.com/2/40/mql5_martin_grid.png)[Grid and martingale: what are they and how to use them?](https://www.mql5.com/en/articles/8390)

In this article, I will try to explain in detail what grid and martingale are, as well as what they have in common. Besides, I will try to analyze how viable these strategies really are. The article features mathematical and practical sections.

![Timeseries in DoEasy library (part 55): Indicator collection class](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library__7.png)[Timeseries in DoEasy library (part 55): Indicator collection class](https://www.mql5.com/en/articles/8576)

The article continues developing indicator object classes and their collections. For each indicator object create its description and correct collection class for error-free storage and getting indicator objects from the collection list.

![Brute force approach to pattern search](https://c.mql5.com/2/40/back_to_the_future_part1.png)[Brute force approach to pattern search](https://www.mql5.com/en/articles/8311)

In this article, we will search for market patterns, create Expert Advisors based on the identified patterns, and check how long these patterns remain valid, if they ever retain their validity.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/8508&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070389299083416823)

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