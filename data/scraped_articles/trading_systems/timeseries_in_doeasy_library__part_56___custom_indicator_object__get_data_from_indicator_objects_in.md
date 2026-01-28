---
title: Timeseries in DoEasy library (part 56): Custom indicator object, get data from indicator objects in the collection
url: https://www.mql5.com/en/articles/8646
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:33:21.425294
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/8646&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070380271062160591)

MetaTrader 5 / Examples


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8646#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8646#node02)

- [Custom indicator object](https://www.mql5.com/en/articles/8646#node03)
- [Testing](https://www.mql5.com/en/articles/8646#node04)
- [What's next?](https://www.mql5.com/en/articles/8646#node05)


### Concept

Continue creating functionality to handle indicators in EAs created on library data basis.

We have already created standard indicator object classes and started placing them in collection list. To have the complete “set” only custom indicator object is missing. Today, such object will be created. The concept of its construction will differ from the concept of creation of standard indicator objects.

Since we have the ultimate set of standard indicators in the terminal, I can create standard indicator objects based on definitely and beforehand known data for each of the indicators. That’s what I do: specify a strictly set parameter combination which corresponds to the indicator created. Meanwhile, the custom indicator may have any parameter set which cannot be known beforehand.

For this reason, the concept of creation of the custom indicator object will differ from creation of the standard indicator object.

If, to create the standard indicator object it is sufficient to create own creation method for each of them which will contain all its necessary properties in its inputs, at the same time, to create the custom indicator with preliminary unknown number and types of parameters we will have to pass the preliminary filled [array of indicator inputs structures](https://www.mql5.com/en/docs/constants/structures/mqlparam) to its creation method. All parameters and properties necessary to create the indicator will be taken from the method. Therefore, unfortunately, library user will have to independently fill such array of inputs structures in order to create custom indicator in the program.

Since I try to create a library to simplify creation of pograms, I have chosen the following variant of calling the created indicators in my program out of several variants: each indicator created will be marked with its own unique ID. I will be able to call each of created indicators by this ID. Fortunately, I create them, I set IDs for them and I am well aware which of the set IDs corresponds to one or another indicator created in the program.

Apart from the ID, there will be methods of indicator calling by all its parameters. I.e. it will be possible to get indicator object having specified the same parameters with which that indicator object was created and then to handle the received object - to get data from the indicator (methods for data receiving will be created today) and to copy them to arrays for statistical research (this will be in the following articles).

### Improving library classes

As usual, first, enter a new library messages in file \\MQL5\\Include\\DoEasy\ **Data.mqh**.

Add indices of new messages:

```
   MSG_LIB_TEXT_IND_TEXT_GROUP,                       // Indicator group
   MSG_LIB_TEXT_IND_TEXT_GROUP_TREND,                 // Trend indicator
   MSG_LIB_TEXT_IND_TEXT_GROUP_OSCILLATOR,            // Oscillator
   MSG_LIB_TEXT_IND_TEXT_GROUP_VOLUMES,               // Volumes
   MSG_LIB_TEXT_IND_TEXT_GROUP_ARROWS,                // Arrow indicator
   MSG_LIB_TEXT_IND_TEXT_ID,                          // Indicator ID

   MSG_LIB_TEXT_IND_TEXT_EMPTY_VALUE,                 // Empty value for plotting where nothing will be drawn
   MSG_LIB_TEXT_IND_TEXT_SYMBOL,                      // Indicator symbol
   MSG_LIB_TEXT_IND_TEXT_NAME,                        // Indicator name
   MSG_LIB_TEXT_IND_TEXT_SHORTNAME,                   // Indicator short name

   MSG_LIB_TEXT_IND_TEXT_IND_PARAMETERS,              // Indicator parameters
   MSG_LIB_TEXT_IND_TEXT_APPLIED_VOLUME,              // Volume type for calculation
   MSG_LIB_TEXT_IND_TEXT_PERIOD,                      // Averaging period
   MSG_LIB_TEXT_IND_TEXT_FAST_PERIOD,                 // Fast MA period
   MSG_LIB_TEXT_IND_TEXT_SLOW_PERIOD,                 // Slow MA period
   MSG_LIB_TEXT_IND_TEXT_SIGNAL,                      // Difference averaging period
   MSG_LIB_TEXT_IND_TEXT_TENKAN_PERIOD,               // Tenkan-sen period
   MSG_LIB_TEXT_IND_TEXT_KIJUN_PERIOD,                // Kijun-sen period
   MSG_LIB_TEXT_IND_TEXT_SPANB_PERIOD,                // Senkou Span B period
   MSG_LIB_TEXT_IND_TEXT_JAW_PERIOD,                  // Period for jaw line calculation
   MSG_LIB_TEXT_IND_TEXT_TEETH_PERIOD,                // Period for teeth line calculation
   MSG_LIB_TEXT_IND_TEXT_LIPS_PERIOD,                 // Period for lips line calculation
   MSG_LIB_TEXT_IND_TEXT_JAW_SHIFT,                   // Horizontal shift of jaws line
   MSG_LIB_TEXT_IND_TEXT_TEETH_SHIFT,                 // Horizontal shift of teeth line
   MSG_LIB_TEXT_IND_TEXT_LIPS_SHIFT,                  // Horizontal shift of lips line
   MSG_LIB_TEXT_IND_TEXT_SHIFT,                       // Horizontal shift of the indicator
   MSG_LIB_TEXT_IND_TEXT_MA_METHOD,                   // Smoothing type
   MSG_LIB_TEXT_IND_TEXT_APPLIED_PRICE,               // Price type or handle
   MSG_LIB_TEXT_IND_TEXT_STD_DEVIATION,               // Number of standard deviations
   MSG_LIB_TEXT_IND_TEXT_DEVIATION,                   // Deviation of channel borders from the central line
   MSG_LIB_TEXT_IND_TEXT_STEP,                        // Price change step — acceleration factor
   MSG_LIB_TEXT_IND_TEXT_MAXIMUM,                     // Maximum step
   MSG_LIB_TEXT_IND_TEXT_KPERIOD,                     // K-period (number of bars for calculation)
   MSG_LIB_TEXT_IND_TEXT_DPERIOD,                     // D-period (primary smoothing period)
   MSG_LIB_TEXT_IND_TEXT_SLOWING,                     // Final smoothing
   MSG_LIB_TEXT_IND_TEXT_PRICE_FIELD,                 // Stochastic calculation method
   MSG_LIB_TEXT_IND_TEXT_CMO_PERIOD,                  // Chande Momentum period
   MSG_LIB_TEXT_IND_TEXT_SMOOTHING_PERIOD,            // Smoothing factor period
   MSG_LIB_TEXT_IND_TEXT_CUSTOM_PARAM,                // Input parameter

//--- CIndicatorsCollection
   MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST,                // Error. Failed to add indicator object to the list
   MSG_LIB_SYS_INVALID_IND_POINTER,                   // Error. Invalid pointer to indicator object is passed
   MSG_LIB_SYS_IND_ID_EXIST,                          // Error. Indicator object with ID already exists

  };
//+------------------------------------------------------------------+
```

and text messages corresponding to newly added indices:

```
   {"Arrow indicator"},
   {"Indicator ID"},
   {"Empty value for plotting, for which there is no drawing"},
   {"Indicator symbol"},
   {"Indicator name"},
   {"Indicator shortname"},

   {"Indicator parameters"},
   {"Volume type for calculation"},
   {"Averaging period"},
   {"Fast MA period"},
   {"Slow MA period"},
   {"Averaging period for their difference"},
   {"Tenkan-sen period"},
   {"Kijun-sen period"},
   {"Senkou Span B period"},
   {"Period for the calculation of jaws"},
   {"Period for the calculation of teeth"},
   {"Period for the calculation of lips"},
   {"Horizontal shift of jaws"},
   {"Horizontal shift of teeth"},
   {"Horizontal shift of lips"},
   {"Horizontal shift of the indicator"},
   {"Smoothing type"},
   {"Price type or handle"},
   {"Number of standard deviations"},
   {"Deviation of boundaries from the midline"},
   {"Price increment step - acceleration factor"},
   {"Maximum value of step"},
   {,"K-period (number of bars for calculations)"},
   {"D-period (period of first smoothing)"},
   {"Final smoothing"},
   {"Stochastic calculation method"},
   {"Chande Momentum period"},
   {"Smoothing factor period"},
   {"Input parameter"},

   {"Error. Failed to add indicator object to list"},
   {"Error. Invalid pointer to indicator object passed"},
   {"Error. There is already exist an indicator object with ID"},

  };
//+---------------------------------------------------------------------+
```

When creating standard indicator object, in its constructor at once set the group which corresponds to indicator type: trend, arrow, oscillator or volume indicator. Since the type of indicator created is known in advance, I can do it for standard indicators. For the custom indicator, you can’t know its type (the group to which the indicator belongs) in advance and set it in object class constructor. Therefore, I will create one more indicator group “any” which will be automatically set for a newly created object. The customer will be provided with a function of setting the custom indicator affiliation group after its object is created.

In file \\MQL5\\Include\\DoEasy\ **Defines.mqh** in indicator group enumeration add a new constant:

```
//+------------------------------------------------------------------+
//| Indicator group                                                  |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_GROUP
  {
   INDICATOR_GROUP_TREND,                                   // Trend indicator
   INDICATOR_GROUP_OSCILLATOR,                              // Oscillator
   INDICATOR_GROUP_VOLUMES,                                 // Volumes
   INDICATOR_GROUP_ARROWS,                                  // Arrow indicator
   INDICATOR_GROUP_ANY,                                     // Any indicator
  };
//+------------------------------------------------------------------+
```

Since we decided to call created indicators by unique IDs, we should add a new property to enumeration of indicator integer properties:

```
//+------------------------------------------------------------------+
//| Indicator integer properties                                     |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_PROP_INTEGER
  {
   INDICATOR_PROP_STATUS = 0,                               // Indicator status (from enumeration ENUM_INDICATOR_STATUS)
   INDICATOR_PROP_TYPE,                                     // Indicator type (from enumeration ENUM_INDICATOR)
   INDICATOR_PROP_TIMEFRAME,                                // Indicator timeframe
   INDICATOR_PROP_HANDLE,                                   // Indicator handle
   INDICATOR_PROP_GROUP,                                    // Indicator group
   INDICATOR_PROP_ID,                                       // Indicator ID
  };
#define INDICATOR_PROP_INTEGER_TOTAL (6)                    // Total number of indicator integer properties
#define INDICATOR_PROP_INTEGER_SKIP  (0)                    // Number of indicator properties not used in sorting
//+------------------------------------------------------------------+
```

and respectively, increase the number of indicator integer properties from 5 to **6**.

To be able to search an indicator in the list by its ID add a new sort criterion for indicators by IDs:

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
   SORT_BY_INDICATOR_ID,                                    // Sort by indicator ID
//--- Sort by real properties
   SORT_BY_INDICATOR_EMPTY_VALUE = FIRST_INDICATOR_DBL_PROP,// Sort by the empty value for plotting where nothing will be drawn
//--- Sort by string properties
   SORT_BY_INDICATOR_SYMBOL = FIRST_INDICATOR_STR_PROP,     // Sort by indicator symbol
   SORT_BY_INDICATOR_NAME,                                  // Sort by indicator name
   SORT_BY_INDICATOR_SHORTNAME,                             // Sort by indicator short name
  };
//+------------------------------------------------------------------+
```

Since new properties were added for the indicator object, now slightly improve the abstract indicator object class in \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **IndicatorDE.mqh**.

In the class public section write two methods for settingand getting “indicator ID” property:

```
//--- Set indicator’s (1) group, (2) empty value of buffers, (3) name, (4) short name, (5) indicator ID
   void              SetGroup(const ENUM_INDICATOR_GROUP group)      { this.SetProperty(INDICATOR_PROP_GROUP,group);                         }
   void              SetEmptyValue(const double value)               { this.SetProperty(INDICATOR_PROP_EMPTY_VALUE,value);                   }
   void              SetName(const string name)                      { this.SetProperty(INDICATOR_PROP_NAME,name);                           }
   void              SetShortName(const string shortname)            { this.SetProperty(INDICATOR_PROP_SHORTNAME,shortname);                 }
   void              SetID(const int id)                             { this.SetProperty(INDICATOR_PROP_ID,id);                               }

//--- Return indicator’s (1) status, (2) group, (3) timeframe, (4) type, (5) handle, (6) ID,
//--- (7) empty value of buffers, (8) name, (9) short name, (10) symbol
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
```

In standard indicator objects we could add methods for displaying the description of each indicator parameter since we definitely know which parameter of a specific indicator is described. To display the description of custom indicator parameters we can’t know definitely the purpose of each parameter of an indicator unknown beforehand. Therefore, simply display descriptions of each subsequent parameter from the parameter array of [MqlParam](https://www.mql5.com/en/docs/constants/structures/mqlparam) indicator.

In the public section of abstract indicator object class declare a method to display the description of MqlParam structure parameters and two more methods: to get data from indicator object by indexand bar time:

```
//--- Return description of (1) type, (2) status, (3) group, (4) timeframe, (5) empty value of indicator, (6) parameter of m_mql_param array
   string            GetTypeDescription(void)                  const { return m_ind_type_description;                                        }
   string            GetStatusDescription(void)                const;
   string            GetGroupDescription(void)                 const;
   string            GetTimeframeDescription(void)             const;
   string            GetEmptyValueDescription(void)            const;
   string            GetMqlParamDescription(const int index)   const;

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

Let's implement these methods outside the class body:

**The method returning the description of parameter of MqlParam structures array:**

```
//+------------------------------------------------------------------+
//| Return the description of parameter of m_mql_param array         |
//+------------------------------------------------------------------+
string CIndicatorDE::GetMqlParamDescription(const int index) const
  {
   return "["+(string)index+"] "+MqlParameterDescription(this.m_mql_param[index]);
  }
//+------------------------------------------------------------------+
```

To method, pass the parameter index in the array and further, create a string from array index and the parameter description in correspondence with the data stored in the structure by the specified array index. Below, I will write MqlParameterDescription() function to return the description of data of MqlParam structure.

**Methods returning indicator data by index and bar time:**

```
//+------------------------------------------------------------------+
//| Return data of specified buffer from specified bar by index      |
//+------------------------------------------------------------------+
double CIndicatorDE::GetDataBuffer(const int buffer_num,const int index)
  {
   double array[1]={EMPTY_VALUE};
   int copied=::CopyBuffer(this.Handle(),buffer_num,index,1,array);
   return(copied==1 ? array[0] : this.EmptyValue());
  }
//+------------------------------------------------------------------+
//| Return data of specified buffer from specified bar by time       |
//+------------------------------------------------------------------+
double CIndicatorDE::GetDataBuffer(const int buffer_num,const datetime time)
  {
   double array[1]={EMPTY_VALUE};
   int copied=::CopyBuffer(this.Handle(),buffer_num,time,1,array);
   return(copied==1 ? array[0] : this.EmptyValue());
  }
//+------------------------------------------------------------------+
```

The method receives index or bar time which data must be received from indicator. Using [CopyBuffer()](https://www.mql5.com/en/docs/series/copybuffer) function request one bar by index or time and return the received result written in the array. If for any reason data are not received the methods return the empty value set for indicator object.

**In the class constructor, set a default indicator ID (-1):**

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
//--- Set collection ID for the object
   this.m_type=COLLECTION_INDICATORS_ID;
//--- Write description of indicator type
   this.m_ind_type_description=IndicatorTypeDescription(ind_type);
//--- If parameter array size passed to constructor is more than zero
//--- fill in the array of object parameters with data from the array passed to constructor
   int count=::ArrayResize(this.m_mql_param,::ArraySize(mql_params));
   for(int i=0;i<count;i++)
     {
      this.m_mql_param[i].type         = mql_params[i].type;
      this.m_mql_param[i].double_value = mql_params[i].double_value;
      this.m_mql_param[i].integer_value= mql_params[i].integer_value;
      this.m_mql_param[i].string_value = mql_params[i].string_value;
     }
//--- Create indicator handle
   int handle=::IndicatorCreate(symbol,timeframe,ind_type,count,this.m_mql_param);

//--- Save integer properties
   this.m_long_prop[INDICATOR_PROP_STATUS]                     = status;
   this.m_long_prop[INDICATOR_PROP_TYPE]                       = ind_type;
   this.m_long_prop[INDICATOR_PROP_GROUP]                      = group;
   this.m_long_prop[INDICATOR_PROP_TIMEFRAME]                  = timeframe;
   this.m_long_prop[INDICATOR_PROP_HANDLE]                     = handle;
   this.m_long_prop[INDICATOR_PROP_ID]                         = WRONG_VALUE;

//--- Save real properties
   this.m_double_prop[this.IndexProp(INDICATOR_PROP_EMPTY_VALUE)]=EMPTY_VALUE;
//--- Save string properties
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_SYMBOL)]   = (symbol==NULL || symbol=="" ? ::Symbol() : symbol);
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_NAME)]     = name;
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_SHORTNAME)]= shortname;
  }
//+------------------------------------------------------------------+
```

Since each indicator’s handle is unique as well as its ID which we will set at own discretion, when comparing the two indicators for identity by their parameters, the two above said parameters must be skipped. Otherwise, each indicator will be identified as unique and we will fail to correctly compare parameters of the two indicators for identity.

To avoid it, skip “handle” and “ID” properties in the method of comparison of the two indicator objects:

```
//+------------------------------------------------------------------+
//| Compare CIndicatorDE objects with each other by all properties   |
//+------------------------------------------------------------------+
bool CIndicatorDE::IsEqual(CIndicatorDE *compared_obj) const
  {
   if(!IsEqualMqlParamArrays(compared_obj.m_mql_param))
      return false;
   int beg=0, end=INDICATOR_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_INDICATOR_PROP_INTEGER prop=(ENUM_INDICATOR_PROP_INTEGER)i;
      if(prop==INDICATOR_PROP_HANDLE || prop==INDICATOR_PROP_ID) continue;
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

Add the display of “ID” indicator property description in the method returning the description of indicator integer property:

```
//+------------------------------------------------------------------+
//| Return description of indicator's integer property               |
//+------------------------------------------------------------------+
string CIndicatorDE::GetPropertyDescription(ENUM_INDICATOR_PROP_INTEGER property)
  {
   return
     (
      property==INDICATOR_PROP_STATUS     ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_STATUS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetStatusDescription()
         )  :
      property==INDICATOR_PROP_TYPE       ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTypeDescription()
         )  :
      property==INDICATOR_PROP_GROUP      ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_GROUP)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetGroupDescription()
         )  :
      property==INDICATOR_PROP_ID         ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_ID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==INDICATOR_PROP_TIMEFRAME  ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_TIMEFRAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTimeframeDescription()
         )  :
      property==INDICATOR_PROP_HANDLE     ?  CMessage::Text(MSG_LIB_TEXT_IND_TEXT_HANDLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Now in service functions file \\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh** add a function for displaying MqlParam structure description:

```
//+------------------------------------------------------------------+
//| Return the description of parameter MqlParam array               |
//+------------------------------------------------------------------+
string MqlParameterDescription(const MqlParam &mql_param)
  {
   int type=mql_param.type;
   string res=CMessage::Text(MSG_ORD_TYPE)+" "+typename(type)+": ";
   //--- type of parameter string
   if(type==TYPE_STRING)
      res+=mql_param.string_value;
   //--- type of parameter datetime
   else if(type==TYPE_DATETIME)
      res+=TimeToString(mql_param.integer_value,TIME_DATE|TIME_MINUTES|TIME_SECONDS);
   //--- type of parameter color
   else if(type==TYPE_COLOR)
      res+=ColorToString((color)mql_param.integer_value,true);
   //--- type of parameter bool
   else if(type==TYPE_BOOL)
      res+=(string)(bool)mql_param.integer_value;
   //--- integer types
   else if(type>TYPE_BOOL && type<TYPE_FLOAT)
      res+=(string)mql_param.integer_value;
   //--- real types
   else
      res+=DoubleToString(mql_param.double_value,8);
   return res;
  }
//+------------------------------------------------------------------+
```

MqlParam structure contains several fields. One of which contains the data type stored in the structure. By this data type I can understand from which structure field the data will be received and which function of data string presentation will be used for data display in the journal.

Get data type and start forming the string consisting of word "Type " +" "\+ data type string description \+ ": ".

And further, depending on data type, to already created string add description of the value stored in structure field, to a corresponding type, using standard functions for displaying string presentation of the required data type.

In the end, the parameters description of the custom indicator with four parameters when using the method of display of MqlParam structure parameters of the abstract indicator object class and the above analyzed service function in terminal journal will look as follows:

```
--- Indicator parameters ---
- [1] Type int: 13
- [2] Type int: 0
- [3] Type int: 0
- [4] Type int: 2
```

Since a new property is added to indicator object - its ID, to all classes of all indicator objects which files are in folder \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **Standart\** make a little addition to methods of short indicator name display. Simply add ID value to the short name:

```
//+------------------------------------------------------------------+
//| Display a short description of indicator object in the journal   |
//+------------------------------------------------------------------+
void CIndAC::PrintShort(void)
  {
   string id=(this.ID()>WRONG_VALUE ? ", id #"+(string)this.ID()+"]" : "]");
   ::Print(GetStatusDescription()," ",this.Name()," ",this.Symbol()," ",TimeframeDescription(this.Timeframe())," [handle ",this.Handle(),id);\
  }\
//+------------------------------------------------------------------+\
```\
\
Here: create ID description. Whereas, if ID value is more than -1, ID will be displayed, otherwise, if ID is missing (its value is -1) it will not be displayed in the description (just a closing square bracket). And further, add the received string to short indicator description.\
\
Such improvements are already entered into all indicator object classes.\
\
### Custom indicator object\
\
Now, let’s add custom indicator object class. Place it to the standard indicator folder of library \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **Standart\**. I will do it just because the terminal indicator list contains custom indicator, also. It means that it also belongs to terminal indicator list.\
\
**Analyze the whole class in general:**\
\
```\
//+------------------------------------------------------------------+\
//|                                                    IndCustom.mqh |\
//|                        Copyright 2020, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2020, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
//+------------------------------------------------------------------+\
//| Include files                                                    |\
//+------------------------------------------------------------------+\
#include "..\\IndicatorDE.mqh"\
//+------------------------------------------------------------------+\
//| Custom indicator                                                 |\
//+------------------------------------------------------------------+\
class CIndCustom : public CIndicatorDE\
  {\
private:\
\
public:\
   //--- Constructor\
                     CIndCustom(const string symbol,const ENUM_TIMEFRAMES timeframe,MqlParam &mql_param[]) :\
                        CIndicatorDE(IND_CUSTOM,symbol,timeframe,\
                                     INDICATOR_STATUS_CUSTOM,\
                                     INDICATOR_GROUP_ANY,\
                                     mql_param[0].string_value,\
                                     mql_param[0].string_value+"("+(symbol==NULL || symbol=="" ? ::Symbol() : symbol)+\
                                                               ","+TimeframeDescription(timeframe)+")",mql_param) {}\
   //--- Supported indicator properties (1) real, (2) integer\
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_DOUBLE property);\
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_INTEGER property);\
\
//--- Display (1) a short description, (2) description of indicator object parameters in the journal\
   virtual void      PrintShort(void);\
   virtual void      PrintParameters(void);\
  };\
//+------------------------------------------------------------------+\
//| Return 'true' if indicator supports a passed                     |\
//| integer property, otherwise return 'false'                       |\
//+------------------------------------------------------------------+\
bool CIndCustom::SupportProperty(ENUM_INDICATOR_PROP_INTEGER property)\
  {\
   return true;\
  }\
//+------------------------------------------------------------------+\
//| Return 'true' if indicator supports a passed                     |\
//| real property, otherwise return 'false'                          |\
//+------------------------------------------------------------------+\
bool CIndCustom::SupportProperty(ENUM_INDICATOR_PROP_DOUBLE property)\
  {\
   return true;\
  }\
//+------------------------------------------------------------------+\
//| Display a short description of indicator object in the journal   |\
//+------------------------------------------------------------------+\
void CIndCustom::PrintShort(void)\
  {\
   string id=(this.ID()>WRONG_VALUE ? ", id #"+(string)this.ID()+"]" : "]");
   ::Print(GetStatusDescription()," ",this.Name()," ",this.Symbol()," ",TimeframeDescription(this.Timeframe())," [handle ",this.Handle(),id);\
  }\
//+------------------------------------------------------------------+\
//| Display parameter description of indicator object in the journal |\
//+------------------------------------------------------------------+\
void CIndCustom::PrintParameters(void)\
  {\
   ::Print(" --- ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_IND_PARAMETERS)," --- ");\
\
   int total=::ArraySize(this.m_mql_param);\
   for(int i=1;i<total;i++)\
     {\
      ::Print(" - ",this.GetMqlParamDescription(i));\
     }\
  }\
//+------------------------------------------------------------------+\
```\
\
We know all methods already due to standard indicator object classes. But in contrast with them, here to class constructor we get parameters of the indicator created in preliminary created array of MqlParam inputs structures but not in the inputs. And to closed constructor of abstract indicator object class pass “custom indicator” status, “any indicator” group, and as a name pass the very first element of parameters array, which obligatory has the type TYPE\_STRING when creating the custom indicator and value of field **string\_value** contains custom indicator name. In the same manner create short indicator name but together with the description of symbol and timeframe. Further, the indicator name and short name may be changed using methods of parent class SetName() and SetShortName(). But indicator name already contains the path to it. Therefore (at least, at this phase of library development) the name of already created custom indicator cannot be changed.\
\
In the method displaying the description of indicator object parameters in the journal do the following: first, display the header, then in the loop by indicator parameters array display each following parameter using earlier analyzed methods (in particular, the method of parent class GetMqlParamDescription()).\
\
All indicator objects are stored in collection list of class CIndicatorsCollection in \\MQL5\\Include\\DoEasy\\Collections\ **IndicatorsCollection.mqh**.\
\
When adding the indicator to collection list, uniqueness of its ID must be additionally checked. And add visibility of custom indicator class so that collection class of indicators could handle them.\
\
Include the file of custom indicator class to indicator object collection class:\
\
```\
//+------------------------------------------------------------------+\
//|                                         IndicatorsCollection.mqh |\
//|                        Copyright 2020, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2020, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
#property version   "1.00"\
//+------------------------------------------------------------------+\
//| Include files                                                    |\
//+------------------------------------------------------------------+\
#include "ListObj.mqh"\
#include "..\Objects\Indicators\Standart\IndAC.mqh"\
#include "..\Objects\Indicators\Standart\IndAD.mqh"\
#include "..\Objects\Indicators\Standart\IndADX.mqh"\
#include "..\Objects\Indicators\Standart\IndADXW.mqh"\
#include "..\Objects\Indicators\Standart\IndAlligator.mqh"\
#include "..\Objects\Indicators\Standart\IndAMA.mqh"\
#include "..\Objects\Indicators\Standart\IndAO.mqh"\
#include "..\Objects\Indicators\Standart\IndATR.mqh"\
#include "..\Objects\Indicators\Standart\IndBands.mqh"\
#include "..\Objects\Indicators\Standart\IndBears.mqh"\
#include "..\Objects\Indicators\Standart\IndBulls.mqh"\
#include "..\Objects\Indicators\Standart\IndBWMFI.mqh"\
#include "..\Objects\Indicators\Standart\IndCCI.mqh"\
#include "..\Objects\Indicators\Standart\IndChaikin.mqh"\
#include "..\Objects\Indicators\Standart\IndCustom.mqh"\
#include "..\Objects\Indicators\Standart\IndDEMA.mqh"\
#include "..\Objects\Indicators\Standart\IndDeMarker.mqh"\
#include "..\Objects\Indicators\Standart\IndEnvelopes.mqh"\
#include "..\Objects\Indicators\Standart\IndForce.mqh"\
#include "..\Objects\Indicators\Standart\IndFractals.mqh"\
#include "..\Objects\Indicators\Standart\IndFRAMA.mqh"\
#include "..\Objects\Indicators\Standart\IndGator.mqh"\
#include "..\Objects\Indicators\Standart\IndIchimoku.mqh"\
#include "..\Objects\Indicators\Standart\IndMA.mqh"\
#include "..\Objects\Indicators\Standart\IndMACD.mqh"\
#include "..\Objects\Indicators\Standart\IndMFI.mqh"\
#include "..\Objects\Indicators\Standart\IndMomentum.mqh"\
#include "..\Objects\Indicators\Standart\IndOBV.mqh"\
#include "..\Objects\Indicators\Standart\IndOsMA.mqh"\
#include "..\Objects\Indicators\Standart\IndRSI.mqh"\
#include "..\Objects\Indicators\Standart\IndRVI.mqh"\
#include "..\Objects\Indicators\Standart\IndSAR.mqh"\
#include "..\Objects\Indicators\Standart\IndStDev.mqh"\
#include "..\Objects\Indicators\Standart\IndStoch.mqh"\
#include "..\Objects\Indicators\Standart\IndTEMA.mqh"\
#include "..\Objects\Indicators\Standart\IndTRIX.mqh"\
#include "..\Objects\Indicators\Standart\IndVIDYA.mqh"\
#include "..\Objects\Indicators\Standart\IndVolumes.mqh"\
#include "..\Objects\Indicators\Standart\IndWPR.mqh"\
//+------------------------------------------------------------------+\
```\
\
In private class section declare a method for adding indicator object to the collection and a method which checks availability of indicator object with set ID in the list:\
\
```\
//+------------------------------------------------------------------+\
//| Indicator collection                                             |\
//+------------------------------------------------------------------+\
class CIndicatorsCollection : public CObject\
  {\
private:\
   CListObj                m_list;                       // Indicator object list\
   MqlParam                m_mql_param[];                // Array of indicator parameters\
\
//--- (1) Create, (2) add to collection list a new indicator object and set an ID for it\
   CIndicatorDE           *CreateIndicator(const ENUM_INDICATOR ind_type,MqlParam &mql_param[],const string symbol_name=NULL,const ENUM_TIMEFRAMES period=PERIOD_CURRENT);\
   int                     AddIndicatorToList(CIndicatorDE *indicator,const int id);\
//--- Return the indicator index in the list\
   int                     Index(CIndicatorDE *compared_obj);\
//--- Check presence of indicator object with specified id in the list\
   bool                    CheckID(const int id);\
\
public:\
```\
\
In public class section declare a method which returns the pointer to custom indicator objectby its groupand parameters specified in the array MqlParam (in contrast with standard indicators, parameters may be specified only through their passing in such array):\
\
```\
   CIndicatorDE           *GetIndCCI(const string symbol,const ENUM_TIMEFRAMES timeframe,\
                                       const int ma_period,\
                                       const ENUM_APPLIED_PRICE applied_price);\
   CIndicatorDE           *GetIndCustom(const string symbol,const ENUM_TIMEFRAMES timeframe,ENUM_INDICATOR_GROUP group,MqlParam &param[]);\
   CIndicatorDE           *GetIndDEMA(const string symbol,const ENUM_TIMEFRAMES timeframe,\
                                       const int ma_period,\
                                       const int ma_shift,\
                                       const ENUM_APPLIED_PRICE applied_price);\
```\
\
In the declaration block of indicator creation methods, firstly, add to parameters of each method indication of indicator ID and, secondly, add declaration of method for creation of custom indicator. I will not provide the full declaration list of these methods. Instead, I will provide only three methods (all methods are already improved and you can find them in attached files):\
\
```\
   int                     CreateCCI(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id,\
                                       const int ma_period=14,\
                                       const ENUM_APPLIED_PRICE applied_price=PRICE_TYPICAL);\
   int                     CreateCustom(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id,\
                                       ENUM_INDICATOR_GROUP group,\
                                       MqlParam &mql_param[]);\
   int                     CreateDEMA(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id,\
                                       const int ma_period=14,\
                                       const int ma_shift=0,\
                                       const ENUM_APPLIED_PRICE applied_price=PRICE_CLOSE);\
```\
\
In all methods after symbol and timeframe inputs, obligatory specification of indicator ID was added at its creation.\
\
In the method of custom indicator creation additionally indicator group is specified and the array of indicator parameters created and filled beforehand is passed. The custom indicator will be created based on them.\
\
In the end of class body listing declare the method for setting ID for the specified indicator and the method which returns indicator object by the specified ID:\
\
```\
//--- Set ID for the specified indicator\
   void                    SetID(CIndicatorDE *indicator,const int id);\
\
//--- Return indicator object by its ID\
   CIndicatorDE           *GetIndByID(const uint id);\
\
//--- Display (1) the complete and (2) short collection description in the journal\
   void                    Print(void);\
   void                    PrintShort(void);\
\
//--- Constructor\
                           CIndicatorsCollection();\
\
  };\
//+------------------------------------------------------------------+\
```\
\
Now, let’s analyze all declared methods.\
\
**In the private method of new indicator object creation add creation of a new object - custom indicator:**\
\
```\
//+------------------------------------------------------------------+\
//| Create new indicator object                                      |\
//+------------------------------------------------------------------+\
CIndicatorDE *CIndicatorsCollection::CreateIndicator(const ENUM_INDICATOR ind_type,MqlParam &mql_param[],const string symbol_name=NULL,const ENUM_TIMEFRAMES period=PERIOD_CURRENT)\
  {\
   string symbol=(symbol_name==NULL || symbol_name=="" ? ::Symbol() : symbol_name);\
   ENUM_TIMEFRAMES timeframe=(period==PERIOD_CURRENT ? ::Period() : period);\
   CIndicatorDE *indicator=NULL;\
   switch(ind_type)\
     {\
      case IND_AC          : indicator=new CIndAC(symbol,timeframe,mql_param);         break;\
      case IND_AD          : indicator=new CIndAD(symbol,timeframe,mql_param);         break;\
      case IND_ADX         : indicator=new CIndADX(symbol,timeframe,mql_param);        break;\
      case IND_ADXW        : indicator=new CIndADXW(symbol,timeframe,mql_param);       break;\
      case IND_ALLIGATOR   : indicator=new CIndAlligator(symbol,timeframe,mql_param);  break;\
      case IND_AMA         : indicator=new CIndAMA(symbol,timeframe,mql_param);        break;\
      case IND_AO          : indicator=new CIndAO(symbol,timeframe,mql_param);         break;\
      case IND_ATR         : indicator=new CIndATR(symbol,timeframe,mql_param);        break;\
      case IND_BANDS       : indicator=new CIndBands(symbol,timeframe,mql_param);      break;\
      case IND_BEARS       : indicator=new CIndBears(symbol,timeframe,mql_param);      break;\
      case IND_BULLS       : indicator=new CIndBulls(symbol,timeframe,mql_param);      break;\
      case IND_BWMFI       : indicator=new CIndBWMFI(symbol,timeframe,mql_param);      break;\
      case IND_CCI         : indicator=new CIndCCI(symbol,timeframe,mql_param);        break;\
      case IND_CHAIKIN     : indicator=new CIndCHO(symbol,timeframe,mql_param);        break;\
      case IND_DEMA        : indicator=new CIndDEMA(symbol,timeframe,mql_param);       break;\
      case IND_DEMARKER    : indicator=new CIndDeMarker(symbol,timeframe,mql_param);   break;\
      case IND_ENVELOPES   : indicator=new CIndEnvelopes(symbol,timeframe,mql_param);  break;\
      case IND_FORCE       : indicator=new CIndForce(symbol,timeframe,mql_param);      break;\
      case IND_FRACTALS    : indicator=new CIndFractals(symbol,timeframe,mql_param);   break;\
      case IND_FRAMA       : indicator=new CIndFRAMA(symbol,timeframe,mql_param);      break;\
      case IND_GATOR       : indicator=new CIndGator(symbol,timeframe,mql_param);      break;\
      case IND_ICHIMOKU    : indicator=new CIndIchimoku(symbol,timeframe,mql_param);   break;\
      case IND_MA          : indicator=new CIndMA(symbol,timeframe,mql_param);         break;\
      case IND_MACD        : indicator=new CIndMACD(symbol,timeframe,mql_param);       break;\
      case IND_MFI         : indicator=new CIndMFI(symbol,timeframe,mql_param);        break;\
      case IND_MOMENTUM    : indicator=new CIndMomentum(symbol,timeframe,mql_param);   break;\
      case IND_OBV         : indicator=new CIndOBV(symbol,timeframe,mql_param);        break;\
      case IND_OSMA        : indicator=new CIndOsMA(symbol,timeframe,mql_param);       break;\
      case IND_RSI         : indicator=new CIndRSI(symbol,timeframe,mql_param);        break;\
      case IND_RVI         : indicator=new CIndRVI(symbol,timeframe,mql_param);        break;\
      case IND_SAR         : indicator=new CIndSAR(symbol,timeframe,mql_param);        break;\
      case IND_STDDEV      : indicator=new CIndStDev(symbol,timeframe,mql_param);      break;\
      case IND_STOCHASTIC  : indicator=new CIndStoch(symbol,timeframe,mql_param);      break;\
      case IND_TEMA        : indicator=new CIndTEMA(symbol,timeframe,mql_param);       break;\
      case IND_TRIX        : indicator=new CIndTRIX(symbol,timeframe,mql_param);       break;\
      case IND_VIDYA       : indicator=new CIndVIDYA(symbol,timeframe,mql_param);      break;\
      case IND_VOLUMES     : indicator=new CIndVolumes(symbol,timeframe,mql_param);    break;\
      case IND_WPR         : indicator=new CIndWPR(symbol,timeframe,mql_param);        break;\
      case IND_CUSTOM      : indicator=new CIndCustom(symbol,timeframe,mql_param);     break;\
      default: break;\
     }\
   return indicator;\
  }\
//+------------------------------------------------------------------+\
```\
\
**Change the method adding a new indicator object to collection list:**\
\
```\
//+------------------------------------------------------------------+\
//| Add a new indicator object to collection list                    |\
//+------------------------------------------------------------------+\
int CIndicatorsCollection::AddIndicatorToList(CIndicatorDE *indicator,const int id)\
  {\
//--- If invalid indicator is passed to the object - return INVALID_HANDLE\
   if(indicator==NULL)\
      return INVALID_HANDLE;\
//--- If such indicator is already in the list\
   int index=this.Index(indicator);\
   if(index!=WRONG_VALUE)\
     {\
      //--- Remove the earlier created object, get indicator object from the list and return indicator handle\
      delete indicator;\
      indicator=this.m_list.At(index);\
     }\
//--- If indicator object is not in the list yet\
   else\
     {\
      //--- If failed to add indicator object to the list - display a corresponding message,\
      //--- remove object and return INVALID_HANDLE\
      if(!this.m_list.Add(indicator))\
        {\
         ::Print(CMessage::Text(MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST));\
         delete indicator;\
         return INVALID_HANDLE;\
        }\
     }\
//--- If indicator is successfully added to the list or is already there...\
//--- If indicator with specified ID (not -1) is not in the list - set ID\
   if(id>WRONG_VALUE && !this.CheckID(id))\
      indicator.SetID(id);\
//--- Return handle of a new indicator added to the list\
   return indicator.Handle();\
  }\
//+------------------------------------------------------------------+\
```\
\
Now, indicator ID is also passed to the method. If indicator with such ID is not in collection list yet, the specified ID is set for indicator object. Otherwise, indicator ID will be set by default -1.\
\
Now, all methods for indicator object creation have become shorter.\
\
**Let’s analyze it using creation of AC and Alligator indicator objects as an example:**\
\
```\
//+------------------------------------------------------------------+\
//| Create a new indicator object Accelerator Oscillator             |\
//| and place it to the collection list                              |\
//+------------------------------------------------------------------+\
int CIndicatorsCollection::CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id)\
  {\
//--- AC indicator possesses no parameters - resize the array of parameter structures\
   ::ArrayResize(this.m_mql_param,0);\
//--- Create indicator object\
   CIndicatorDE *indicator=this.CreateIndicator(IND_AC,this.m_mql_param,symbol,timeframe);\
//--- Return indicator handle received as a result of adding the object to collection list\
   return this.AddIndicatorToList(indicator,id);\
  }\
//+------------------------------------------------------------------+\
```\
\
```\
//+------------------------------------------------------------------+\
//| Create new indicator object Alligator                            |\
//| and place it to the collection list                              |\
//+------------------------------------------------------------------+\
int CIndicatorsCollection::CreateAlligator(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id,\
                                           const int jaw_period=13,\
                                           const int jaw_shift=8,\
                                           const int teeth_period=8,\
                                           const int teeth_shift=5,\
                                           const int lips_period=5,\
                                           const int lips_shift=3,\
                                           const ENUM_MA_METHOD ma_method=MODE_SMMA,\
                                           const ENUM_APPLIED_PRICE applied_price=PRICE_MEDIAN)\
  {\
//--- Add required indicator parameters to the array of parameter structures\
   ::ArrayResize(this.m_mql_param,8);\
   this.m_mql_param[0].type=TYPE_INT;\
   this.m_mql_param[0].integer_value=jaw_period;\
   this.m_mql_param[1].type=TYPE_INT;\
   this.m_mql_param[1].integer_value=jaw_shift;\
   this.m_mql_param[2].type=TYPE_INT;\
   this.m_mql_param[2].integer_value=teeth_period;\
   this.m_mql_param[3].type=TYPE_INT;\
   this.m_mql_param[3].integer_value=teeth_shift;\
   this.m_mql_param[4].type=TYPE_INT;\
   this.m_mql_param[4].integer_value=lips_period;\
   this.m_mql_param[5].type=TYPE_INT;\
   this.m_mql_param[5].integer_value=lips_shift;\
   this.m_mql_param[6].type=TYPE_INT;\
   this.m_mql_param[6].integer_value=ma_method;\
   this.m_mql_param[7].type=TYPE_INT;\
   this.m_mql_param[7].integer_value=applied_price;\
//--- Create indicator object\
   CIndicatorDE *indicator=this.CreateIndicator(IND_ALLIGATOR,this.m_mql_param,symbol,timeframe);\
//--- Return indicator handle received as a result of adding the object to collection list\
   return this.AddIndicatorToList(indicator,id);\
  }\
//+------------------------------------------------------------------+\
```\
\
Now, it is sufficient only to fill the inputs structure, create an indicator and call the method of its adding to the collection list. Such changes were performed in all methods of indicator object creation. I will not analyze them, except for the custom indicator creation method:\
\
```\
//+------------------------------------------------------------------+\
//| Create a new object - custom indicator                           |\
//| and place it to the collection list                              |\
//+------------------------------------------------------------------+\
int CIndicatorsCollection::CreateCustom(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id,\
                                        ENUM_INDICATOR_GROUP group,\
                                        MqlParam &mql_param[])\
  {\
//--- Create indicator object\
   CIndicatorDE *indicator=this.CreateIndicator(IND_CUSTOM,mql_param,symbol,timeframe);\
   if(indicator==NULL)\
      return INVALID_HANDLE;\
//--- Set a group for indicator object\
   indicator.SetGroup(group);\
//--- Return indicator handle received as a result of adding the object to collection list\
   return this.AddIndicatorToList(indicator,id);\
  }\
//+------------------------------------------------------------------+\
```\
\
Here, it is a little different. Here, apart from ID the creation method gets indicator group, also. And all parameters of the indicator created are passed at once within the array of MqlParam parameters for the reason that we can’t know beforehand about parameters of custom indicator created.\
\
Standard default values for each input parameter were added to absolutely all methods of standard indicator creation. Thus, to create a standard indicator with default parameters it will be sufficient to specify the symbol, timeframe and ID.\
\
**Implementing the method returning the pointer to the object- custom indicator:**\
\
```\
//+------------------------------------------------------------------+\
//| Return pointer to object- custom indicator                       |\
//+------------------------------------------------------------------+\
CIndicatorDE *CIndicatorsCollection::GetIndCustom(const string symbol,const ENUM_TIMEFRAMES timeframe,ENUM_INDICATOR_GROUP group,MqlParam &param[])\
  {\
   CIndicatorDE *tmp=new CIndCustom(symbol,timeframe,param);\
   if(tmp==NULL)\
      return NULL;\
   tmp.SetGroup(group);\
   int index=this.Index(tmp);\
   delete tmp;\
   return(index>WRONG_VALUE ? this.m_list.At(index) : NULL);\
  }\
//+------------------------------------------------------------------+\
```\
\
Here: create a temporary indicator object to search the same object in the collection list, set a group for it and get the index of the found object in the collection list. Then, remove the temporary object and return either a pointer to the object by the found index of the list, or NULL if such object is not found in the list.\
\
**Method which checks presence of indicator object with specified id in the list:**\
\
```\
//+------------------------------------------------------------------+\
//| Check presence of indicator object with specified id in the list |\
//+------------------------------------------------------------------+\
bool CIndicatorsCollection::CheckID(const int id)\
  {\
   CArrayObj *list=CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_ID,id,EQUAL);\
   return(list!=NULL && list.Total()!=0);\
  }\
//+------------------------------------------------------------------+\
```\
\
Get the list of indicator objects with the specified ID and return the flag checking if the list is valid and not empty (the list must contain one object).\
\
**Method which sets ID for the specified indicator:**\
\
```\
//+------------------------------------------------------------------+\
//| Set ID for the specified indicator                               |\
//+------------------------------------------------------------------+\
void CIndicatorsCollection::SetID(CIndicatorDE *indicator,const int id)\
  {\
   if(indicator==NULL)\
     {\
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_INVALID_IND_POINTER));\
      return;\
     }\
   if(id>WRONG_VALUE)\
     {\
      if(CheckID(id))\
        {\
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_IND_ID_EXIST)," #",(string)id);\
         return;\
        }\
     }\
   indicator.SetID(id);\
  }\
//+------------------------------------------------------------------+\
```\
\
The method receives a pointer to indicator object to which the ID passed to the method by the parameter must be set.\
\
If invalid pointer is passed, notify and leave the method.\
\
If ID value is more than -1check presence of indicator object with the same ID and if it is available - inform and leave.\
\
If all checks are successful set ID for the object. In case if ID value passed to the method is less than zero such ID is set for the object without any checks and such ID of indicator object means its absence.\
\
**Method which returns indicator object by the specified ID:**\
\
```\
//+------------------------------------------------------------------+\
//| Return indicator object by its ID                                |\
//+------------------------------------------------------------------+\
CIndicatorDE *CIndicatorsCollection::GetIndByID(const uint id)\
  {\
   CArrayObj *list=CSelect::ByIndicatorProperty(this.GetList(),INDICATOR_PROP_ID,id,EQUAL);\
   return(list==NULL || list.Total()==0 ? NULL : list.At(list.Total()-1));\
  }\
//+------------------------------------------------------------------+\
```\
\
Here: get the list of indicator objects with the specified ID and return either NULL (if failed to get the list or the list is empty), or a pointer to the object with the necessary ID. Since there can be only one object with the specified ID it doesn’t matter which index will be specified in the received list: the first or the last one. Here the last one is specified.\
\
Testing of indicator creation in EA demonstrated a problem: when changing a timeframe the same additional indicators are created but with another timeframe. And it is true: indicators with the same inputs but calculated on different timeframes are two different indicators. To simply avoid this problem it is enough to clear the list of created indicators during program deinitialization. For this purpose, in CEngine library main object, in file \\MQL5\\Include\\DoEasy\ **Engine.mqh** declare a new handler OnDeinit():\
\
```\
//--- (1) Timer, event handler (2) NewTick, (3) Calculate, Deinit\
   void                 OnTimer(SDataCalculate &data_calculate);\
   void                 OnTick(SDataCalculate &data_calculate,const uint required=0);\
   int                  OnCalculate(SDataCalculate &data_calculate,const uint required=0);\
   void                 OnDeinit(void);\
```\
\
And write its implementation outside the class body:\
\
```\
//+------------------------------------------------------------------+\
//| Deinitialize library                                             |\
//+------------------------------------------------------------------+\
void CEngine::OnDeinit(void)\
  {\
   this.m_indicators.GetList().Clear();\
  }\
//+------------------------------------------------------------------+\
```\
\
This class method will be called from OnDeinit() handler of the program. What we have here is calling the collection list clearing method in indicator collection object.\
\
### Testing\
\
Let's use [the EA from the previous article](https://www.mql5.com/en/articles/8576#node03) to test creation of various indicators and getting data from created indicator objects.\
\
Save it in a new folder \\MQL5\\Experts\\TestDoEasy\ **Part56\** under a new name **TestDoEasyPart56.mq5**.\
\
In EA, create two custom indicators Moving Average, but with different parameters (take indicators in indicator sample folder from the terminal standard delivery \\MQL5\\Indicators\\Examples\\). And create two standard indicators Adaptive Moving Average also with different inputs.\
\
In the global area set macro substitutions to simplify calling of indicators by their ID and declare two arrays of parameters for creation of two custom indicators:\
\
```\
//+------------------------------------------------------------------+\
//|                                             TestDoEasyPart56.mq5 |\
//|                        Copyright 2020, MetaQuotes Software Corp. |\
//|                             https://mql5.com/en/users/artmedia70 |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2020, MetaQuotes Software Corp."\
#property link      "https://mql5.com/en/users/artmedia70"\
#property version   "1.00"\
//--- includes\
#include <DoEasy\Engine.mqh>\
//--- enums\
enum ENUM_BUTTONS\
  {\
   BUTT_BUY,\
   BUTT_BUY_LIMIT,\
   BUTT_BUY_STOP,\
   BUTT_BUY_STOP_LIMIT,\
   BUTT_CLOSE_BUY,\
   BUTT_CLOSE_BUY2,\
   BUTT_CLOSE_BUY_BY_SELL,\
   BUTT_SELL,\
   BUTT_SELL_LIMIT,\
   BUTT_SELL_STOP,\
   BUTT_SELL_STOP_LIMIT,\
   BUTT_CLOSE_SELL,\
   BUTT_CLOSE_SELL2,\
   BUTT_CLOSE_SELL_BY_BUY,\
   BUTT_DELETE_PENDING,\
   BUTT_CLOSE_ALL,\
   BUTT_SET_STOP_LOSS,\
   BUTT_SET_TAKE_PROFIT,\
   BUTT_PROFIT_WITHDRAWAL,\
   BUTT_TRAILING_ALL\
  };\
#define TOTAL_BUTT   (20)\
#define MA1          (1)\
#define MA2          (2)\
#define AMA1         (3)\
#define AMA2         (4)\
//--- structures\
struct SDataButt\
  {\
   string      name;\
   string      text;\
  };\
//--- input variables\
input    ushort            InpMagic             =  123;  // Magic number\
input    double            InpLots              =  0.1;  // Lots\
input    uint              InpStopLoss          =  150;  // StopLoss in points\
input    uint              InpTakeProfit        =  150;  // TakeProfit in points\
input    uint              InpDistance          =  50;   // Pending orders distance (points)\
input    uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)\
input    uint              InpDistancePReq      =  50;   // Distance for Pending Request's activate (points)\
input    uint              InpBarsDelayPReq     =  5;    // Bars delay for Pending Request's activate (current timeframe)\
input    uint              InpSlippage          =  5;    // Slippage in points\
input    uint              InpSpreadMultiplier  =  1;    // Spread multiplier for adjusting stop-orders by StopLevel\
input    uchar             InpTotalAttempts     =  5;    // Number of trading attempts\
sinput   double            InpWithdrawal        =  10;   // Withdrawal funds (in tester)\
sinput   uint              InpButtShiftX        =  0;    // Buttons X shift\
sinput   uint              InpButtShiftY        =  10;   // Buttons Y shift\
input    uint              InpTrailingStop      =  50;   // Trailing Stop (points)\
input    uint              InpTrailingStep      =  20;   // Trailing Step (points)\
input    uint              InpTrailingStart     =  0;    // Trailing Start (points)\
input    uint              InpStopLossModify    =  20;   // StopLoss for modification (points)\
input    uint              InpTakeProfitModify  =  60;   // TakeProfit for modification (points)\
sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;            // Mode of used symbols list\
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)\
sinput   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list\
sinput   string            InpUsedTFs           =  "M1,M5,M15,M30,H1,H4,D1,W1,MN1"; // List of used timeframes (comma - separator)\
sinput   bool              InpUseSounds         =  true; // Use sounds\
//--- global variables\
CEngine        engine;\
SDataButt      butt_data[TOTAL_BUTT];\
string         prefix;\
double         lot;\
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);\
ushort         magic_number;\
uint           stoploss;\
uint           takeprofit;\
uint           distance_pending;\
uint           distance_stoplimit;\
uint           distance_pending_request;\
uint           bars_delay_pending_request;\
uint           slippage;\
bool           trailing_on;\
bool           pressed_pending_buy;\
bool           pressed_pending_buy_limit;\
bool           pressed_pending_buy_stop;\
bool           pressed_pending_buy_stoplimit;\
bool           pressed_pending_close_buy;\
bool           pressed_pending_close_buy2;\
bool           pressed_pending_close_buy_by_sell;\
bool           pressed_pending_sell;\
bool           pressed_pending_sell_limit;\
bool           pressed_pending_sell_stop;\
bool           pressed_pending_sell_stoplimit;\
bool           pressed_pending_close_sell;\
bool           pressed_pending_close_sell2;\
bool           pressed_pending_close_sell_by_buy;\
bool           pressed_pending_delete_all;\
bool           pressed_pending_close_all;\
bool           pressed_pending_sl;\
bool           pressed_pending_tp;\
double         trailing_stop;\
double         trailing_step;\
uint           trailing_start;\
uint           stoploss_to_modify;\
uint           takeprofit_to_modify;\
int            used_symbols_mode;\
string         array_used_symbols[];\
string         array_used_periods[];\
bool           testing;\
uchar          group1;\
uchar          group2;\
double         g_point;\
int            g_digits;\
//--- Arrays of custom indicator parameters\
MqlParam       param_ma1[];\
MqlParam       param_ma2[];\
//+------------------------------------------------------------------+\
```\
\
Substantially, the declared macro substitutions are the description of numerical values of indicators. Calling the indicator ID by name is more simple than calling it by its value.\
\
In handler OnInit() of EA create all four indicators and at once display in the journal data of all created indicators:\
\
```\
//+------------------------------------------------------------------+\
//| Expert initialization function                                   |\
//+------------------------------------------------------------------+\
int OnInit()\
  {\
//--- Calling the function displays the list of enumeration constants in the journal,\
//--- (the list is set in strings 22 and 25 of  DELib.mqh file) for checking the constants validity\
   //EnumNumbersTest();\
\
//--- Set EA global variables\
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";\
   testing=engine.IsTester();\
   for(int i=0;i<TOTAL_BUTT;i++)\
     {\
      butt_data[i].name=prefix+EnumToString((ENUM_BUTTONS)i);\
      butt_data[i].text=EnumToButtText((ENUM_BUTTONS)i);\
     }\
   lot=NormalizeLot(Symbol(),fmax(InpLots,MinimumLots(Symbol())*2.0));\
   magic_number=InpMagic;\
   stoploss=InpStopLoss;\
   takeprofit=InpTakeProfit;\
   distance_pending=InpDistance;\
   distance_stoplimit=InpDistanceSL;\
   slippage=InpSlippage;\
   trailing_stop=InpTrailingStop*Point();\
   trailing_step=InpTrailingStep*Point();\
   trailing_start=InpTrailingStart;\
   stoploss_to_modify=InpStopLossModify;\
   takeprofit_to_modify=InpTakeProfitModify;\
   distance_pending_request=(InpDistancePReq<5 ? 5 : InpDistancePReq);\
   bars_delay_pending_request=(InpBarsDelayPReq<1 ? 1 : InpBarsDelayPReq);\
   g_point=SymbolInfoDouble(NULL,SYMBOL_POINT);\
   g_digits=(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS);\
//--- Initialize random group numbers\
   group1=0;\
   group2=0;\
   srand(GetTickCount());\
\
//--- Initialize DoEasy library\
   OnInitDoEasy();\
\
//--- Create indicators\
   ArrayResize(param_ma1,4);\
   //--- Name of indicator 1\
   param_ma1[0].type=TYPE_STRING;\
   param_ma1[0].string_value="Examples\\Custom Moving Average.ex5";\
   //--- Calculation period\
   param_ma1[1].type=TYPE_INT;\
   param_ma1[1].integer_value=13;\
   //--- Horizontal shift\
   param_ma1[2].type=TYPE_INT;\
   param_ma1[2].integer_value=0;\
   //--- Smoothing method\
   param_ma1[3].type=TYPE_INT;\
   param_ma1[3].integer_value=MODE_SMA;\
   //--- Create indicator 1\
   engine.GetIndicatorsCollection().CreateCustom(NULL,PERIOD_CURRENT,MA1,INDICATOR_GROUP_TREND,param_ma1);\
\
   ArrayResize(param_ma2,5);\
   //--- Name of indicator 2\
   param_ma2[0].type=TYPE_STRING;\
   param_ma2[0].string_value="Examples\\Custom Moving Average.ex5";\
   //--- Calculation period\
   param_ma2[1].type=TYPE_INT;\
   param_ma2[1].integer_value=13;\
   //--- Horizontal shift\
   param_ma2[2].type=TYPE_INT;\
   param_ma2[2].integer_value=0;\
   //--- Smoothing method\
   param_ma2[3].type=TYPE_INT;\
   param_ma2[3].integer_value=MODE_SMA;\
   //--- Calculation price\
   param_ma2[4].type=TYPE_INT;\
   param_ma2[4].integer_value=PRICE_OPEN;\
   //--- Create indicator 2\
   engine.GetIndicatorsCollection().CreateCustom(NULL,PERIOD_CURRENT,MA2,INDICATOR_GROUP_TREND,param_ma2);\
\
   //--- Create indicator 3\
   engine.GetIndicatorsCollection().CreateAMA(NULL,PERIOD_CURRENT,AMA1);\
   //--- Create indicator 4\
   engine.GetIndicatorsCollection().CreateAMA(NULL,PERIOD_CURRENT,AMA2,14);\
\
   //--- Display descriptions of created indicators\
   engine.GetIndicatorsCollection().Print();\
   engine.GetIndicatorsCollection().PrintShort();\
\
//--- Check and remove remaining EA graphical objects\
   if(IsPresentObectByPrefix(prefix))\
      ObjectsDeleteAll(0,prefix);\
\
//--- Create the button panel\
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))\
      return INIT_FAILED;\
//--- Set trailing activation button status\
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);\
//--- Reset states of the buttons for working using pending requests\
   for(int i=0;i<14;i++)\
     {\
      ButtonState(butt_data[i].name+"_PRICE",false);\
      ButtonState(butt_data[i].name+"_TIME",false);\
     }\
\
//--- Check playing a standard sound by macro substitution and a custom sound by description\
   engine.PlaySoundByDescription(SND_OK);\
//--- Wait for 600 milliseconds\
   engine.Pause(600);\
   engine.PlaySoundByDescription(TextByLanguage("The sound of a falling coin 2"));\
\
//---\
   return(INIT_SUCCEEDED);\
  }\
//+------------------------------------------------------------------+\
```\
\
Set calculation price PRICE\_OPEN for the second custom indicator MA. If calculation price is not specified clearly, by default (in MA first indicator) PRICE\_CLOSE price is used to calculate the indicator.\
\
When creating AMA indicators the first one of them receives calculation period as 9 (set by default). The second one explicitly receives the value of 14.\
\
Thus, all four indicators created possess different input values of their parameters.\
\
In handler OnDeinit() of EA call handler OnDeinit() of the library:\
\
```\
//+------------------------------------------------------------------+\
//| Expert deinitialization function                                 |\
//+------------------------------------------------------------------+\
void OnDeinit(const int reason)\
  {\
//--- Remove EA graphical objects by an object name prefix\
   ObjectsDeleteAll(0,prefix);\
   Comment("");\
//--- Deinitialize library\
   engine.OnDeinit();\
  }\
//+------------------------------------------------------------------+\
```\
\
This will clear the collection list of indicators during EA deinitialization when switching over timeframes. This will exclude a necessity to create unnecessary additional indicator objects.\
\
In handler OnTick() of EA get access to each of created indicator objects and display data of the current bar of each indicator having displayed them in the comment on chart:\
\
```\
//+------------------------------------------------------------------+\
//| Expert tick function                                             |\
//+------------------------------------------------------------------+\
void OnTick()\
  {\
//--- Handle the NewTick event in the library\
   engine.OnTick(rates_data);\
\
//--- If work in tester\
   if(MQLInfoInteger(MQL_TESTER))\
     {\
      engine.OnTimer(rates_data);   // Work in timer\
      PressButtonsControl();        // Button press control\
      engine.EventsHandling();      // Work with events\
     }\
//--- Get custom indicator objects\
   CIndicatorDE *ma1=engine.GetIndicatorsCollection().GetIndByID(MA1);\
   CIndicatorDE *ma2=engine.GetIndicatorsCollection().GetIndByID(MA2);\
   CIndicatorDE *ama1=engine.GetIndicatorsCollection().GetIndByID(AMA1);\
   CIndicatorDE *ama2=engine.GetIndicatorsCollection().GetIndByID(AMA2);\
   Comment\
     (\
      "ma1=",DoubleToString(ma1.GetDataBuffer(0,0),6),\
      ", ma2=",DoubleToString(ma2.GetDataBuffer(0,0),6),\
      "\nama1=",DoubleToString(ama1.GetDataBuffer(0,0),6),\
      ", ama2=",DoubleToString(ama2.GetDataBuffer(0,0),6)\
     );\
\
\
//--- If the trailing flag is set\
   if(trailing_on)\
     {\
      TrailingPositions();          // Trailing positions\
      TrailingOrders();             // Trailing of pending orders\
     }\
  }\
//+------------------------------------------------------------------+\
```\
\
In the previous article we temporarily created indicator objects in library initializing function OnInitDoEasy(). Remove these strings from function:\
\
```\
//--- Create timeseries of all symbols used\
   engine.SeriesCreateAll(array_used_periods);\
//--- Check created timeseries - display descriptions of all created timeseries in the journal\
//--- (true - only created ones, false - created and declared ones)\
   engine.GetTimeSeriesCollection().PrintShort(false); // Short descriptions\
   //engine.GetTimeSeriesCollection().Print(true);      // Full descriptions\
\
//--- Create indicators\
   engine.GetIndicatorsCollection().CreateAMA(Symbol(),Period(),9,2,30,0,PRICE_CLOSE);\
   engine.GetIndicatorsCollection().CreateAMA(Symbol(),Period(),10,3,32,5,PRICE_CLOSE);\
   engine.GetIndicatorsCollection().Print();\
   engine.GetIndicatorsCollection().PrintShort();\
\
//--- Create resource text files\
```\
\
Compile the EA and launch it on the chart having preliminary set in settings to use only current symbol and timeframe.\
\
The journal will display descriptions of parameters of all created indicators:\
\
```\
--- Initialize "DoEasy" library ---\
Work with the current symbol only: "EURUSD"\
Work with the current timeframe only: H1\
EURUSD symbol timeseries:\
- "EURUSD" H1 timeseries: Requested: 1000, Actually: 1000, Created: 1000, On the server: 6284\
Library initialize time: 00:00:00.141\
============= Parameter list start: “Custom indicator" =============\
Indicator status: Custom indicator\
Indicator type: CUSTOM\
Indicator timeframe: H1\
Indicator handle: 10\
Indicator group: Trend indicator\
Indicator ID: 1\
------\
Empty value for plotting where nothing will be drawn: EMPTY_VALUE\
------\
Indicator symbol: EURUSD\
Indicator name: "Examples\Custom Moving Average.ex5"\
Indicator short name: "Examples\Custom Moving Average.ex5(EURUSD,H1)"\
 --- Indicator parameters ---\
 - [1] Type int: 13\
 - [2] Type int: 0\
 - [3] Type int: 0\
================== Parameter list end: "Custom indicator" ==================\
\
============= Parameter list start: “Custom indicator" =============\
Indicator status: Custom indicator\
Indicator type: CUSTOM\
Indicator timeframe: H1\
Indicator handle: 11\
Indicator group: Trend indicator\
Indicator ID: 2\
------\
Empty value for plotting where nothing will be drawn: EMPTY_VALUE\
------\
Indicator symbol: EURUSD\
Indicator name: "Examples\Custom Moving Average.ex5"\
Indicator short name: "Examples\Custom Moving Average.ex5(EURUSD,H1)"\
 --- Indicator parameters ---\
 - [1] Type int: 13\
 - [2] Type int: 0\
 - [3] Type int: 0\
 - [4] Type int: 2\
================== Parameter list end: "Custom indicator" ==================\
\
============= Parameter list start: "Standard indicator" =============\
Indicator status: Standard indicator\
Indicator type: AMA\
Indicator timeframe: H1\
Indicator handle: 12\
Indicator group: Trend indicator\
Indicator ID: 3\
------\
Empty value for plotting where nothing will be drawn: EMPTY_VALUE\
------\
Indicator symbol: EURUSD\
Indicator name: "Adaptive Moving Average"\
Indicator short name: "AMA(EURUSD,H1)"\
 --- Indicator parameters ---\
 - Averaging period: 9\
 - Fast MA period: 2\
 - Slow MA period: 30\
 - Horizontal shift of the indicator: 0\
 - Price type or handle: CLOSE\
================== Parameter list end: "Standard indicator" ==================\
\
============= Parameter list start: "Standard indicator" =============\
Indicator status: Standard indicator\
Indicator type: AMA\
Indicator timeframe: H1\
Indicator handle: 13\
Indicator group: Trend indicator\
Indicator ID: 4\
------\
Empty value for plotting where nothing will be drawn: EMPTY_VALUE\
------\
Indicator symbol: EURUSD\
Indicator name: "Adaptive Moving Average"\
Indicator short name: "AMA(EURUSD,H1)"\
 --- Indicator parameters ---\
 - Averaging period: 14\
 - Fast MA period: 2\
 - Slow MA period: 30\
 - Horizontal shift of the indicator: 0\
 - Price type or handle: CLOSE\
================== Parameter list end: "Standard indicator" ==================\
\
Custom indicator Examples\Custom Moving Average.ex5 EURUSD H1 [handle 10, id #1]\
Custom indicator Examples\Custom Moving Average.ex5 EURUSD H1 [handle 11, id #2]\
Standard indicator Adaptive Moving Average EURUSD H1 [handle 12, id #3]\
Standard indicator Adaptive Moving Average EURUSD H1 [handle 13, id #4]\
```\
\
Symbol chart will display data from buffers of all created indicators:\
\
![](https://c.mql5.com/2/41/terminal64_wbOoTfdugK.png)\
\
To the chart necessary indicators may be added which correspond by parameters to those created in EA. You can check the match of those indicators in the comment on chart and in data window - they will match.\
\
### What's next?\
\
In the following article I will continue creating functionality to handle indicators in EAs. In the nearest articles I plan to create binding of indicator data to timeseries class bars and to get data from indicators for various statistical research.\
\
All files of the current library version are attached below together with the test EA file for MQL5. You can download them and test everything.\
\
Note, that at the moment indicator collection class is under development, therefore  it is strictly recommended not to use it in your programs.\
\
Leave your comments, questions and suggestions in the comments to the article.\
\
[Back to contents](https://www.mql5.com/en/articles/8646#node00)\
\
**Previous articles within the series:**\
\
[Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list](https://www.mql5.com/en/articles/7594)\
\
[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)\
\
[Timeseries in DoEasy library (part 37): Timeseries collection - database of timeseries by symbols and periods](https://www.mql5.com/en/articles/7663)\
\
[Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program](https://www.mql5.com/en/articles/7695)\
\
[Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://www.mql5.com/en/articles/7724)\
\
[Timeseries in DoEasy library (part 40): Library-based indicators - updating data in real time](https://www.mql5.com/en/articles/7771)\
\
[Timeseries in DoEasy library (part 41): Sample multi-symbol multi-period indicator](https://www.mql5.com/en/articles/7804)\
\
[Timeseries in DoEasy library (part 42): Abstract indicator buffer object class](https://www.mql5.com/en/articles/7821)\
\
[Timeseries in DoEasy library (part 43): Classes of indicator buffer objects](https://www.mql5.com/en/articles/7868)\
\
[Timeseries in DoEasy library (part 44): Collection class of indicator buffer objects](https://www.mql5.com/en/articles/7886)\
\
[Timeseries in DoEasy library (part 45): Multi-period indicator buffers](https://www.mql5.com/en/articles/8023)\
\
[Timeseries in DoEasy library (part 46): Multi-period multi-symbol indicator buffers](https://www.mql5.com/en/articles/8115)\
\
[Timeseries in DoEasy library (part 47): Multi-period multi-symbol standard indicators](https://www.mql5.com/en/articles/8207)\
\
[Timeseries in DoEasy library (part 48): Multi-period multi-symbol indicators on one buffer in subwindow](https://www.mql5.com/en/articles/8257)\
\
[Timeseries in DoEasy library (part 49): Multi-period multi-symbol multi-buffer standard indicators](https://www.mql5.com/en/articles/8292)\
\
[Timeseries in DoEasy library (part 50): Multi-period multi-symbol standard indicators with a shift](https://www.mql5.com/en/articles/8331)\
\
[Timeseries in DoEasy library (part 51): Composite multi-period multi-symbol standard indicators](https://www.mql5.com/en/articles/8354)\
\
[Timeseries in DoEasy library (part 52): Cross-platform nature of multi-period multi-symbol single-buffer standard indicators](https://www.mql5.com/en/articles/8399)\
\
[Timeseries in DoEasy library (part 53): Abstract base indicator class](https://www.mql5.com/en/articles/8464)\
\
[Timeseries in DoEasy library (part 54): Descendant classes of abstract base indicator](https://www.mql5.com/en/articles/8508)\
\
[Timeseries in DoEasy library (part 55): Indicator collection class](https://www.mql5.com/en/articles/8576/)\
\
Translated from Russian by MetaQuotes Ltd.\
\
Original article: [https://www.mql5.com/ru/articles/8646](https://www.mql5.com/ru/articles/8646)\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/8646.zip "Download all attachments in the single ZIP archive")\
\
[MQL5.zip](https://www.mql5.com/en/articles/download/8646/mql5.zip "Download MQL5.zip")(3865.91 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)\
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)\
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)\
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)\
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)\
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)\
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/359579)**\
(2)\
\
\
![Mustafa Nail Sertoglu](https://c.mql5.com/avatar/2021/11/618E1649-9997.PNG)\
\
**[Mustafa Nail Sertoglu](https://www.mql5.com/en/users/nail_mql5)**\
\|\
7 Jan 2021 at 11:29\
\
Thanks for sharing these valuable [PROJECT](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") ; I will be examining all pieces of codes to learn more deep coding tricks..   :) :)\
\
\
![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)\
\
**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**\
\|\
7 Jan 2021 at 12:05\
\
**nail sertoglu:**\
\
Thanks for sharing these valuable PROJECT ; I will be examining all pieces of codes to learn more deep coding tricks..   :) :)\
\
Are you welcome\
\
\
![Timeseries in DoEasy library (part 57): Indicator buffer data object](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__1.png)[Timeseries in DoEasy library (part 57): Indicator buffer data object](https://www.mql5.com/en/articles/8705)\
\
In the article, develop an object which will contain all data of one buffer for one indicator. Such objects will be necessary for storing serial data of indicator buffers. With their help, it will be possible to sort and compare buffer data of any indicators, as well as other similar data with each other.\
\
![Practical application of neural networks in trading. Python (Part I)](https://c.mql5.com/2/40/neural_python.png)[Practical application of neural networks in trading. Python (Part I)](https://www.mql5.com/en/articles/8502)\
\
In this article, we will analyze the step-by-step implementation of a trading system based on the programming of deep neural networks in Python. This will be performed using the TensorFlow machine learning library developed by Google. We will also use the Keras library for describing neural networks.\
\
![Optimal approach to the development and analysis of trading systems](https://c.mql5.com/2/40/optimal-approach.png)[Optimal approach to the development and analysis of trading systems](https://www.mql5.com/en/articles/8410)\
\
In this article, I will show the criteria to be used when selecting a system or a signal for investing your funds, as well as describe the optimal approach to the development of trading systems and highlight the importance of this matter in Forex trading.\
\
![Neural networks made easy (Part 5): Multithreaded calculations in OpenCL](https://c.mql5.com/2/48/Neural_networks_made_easy_0065.png)[Neural networks made easy (Part 5): Multithreaded calculations in OpenCL](https://www.mql5.com/en/articles/8435)\
\
We have earlier discussed some types of neural network implementations. In the considered networks, the same operations are repeated for each neuron. A logical further step is to utilize multithreaded computing capabilities provided by modern technology in an effort to speed up the neural network learning process. One of the possible implementations is described in this article.\
\
[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/8646&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070380271062160591)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)