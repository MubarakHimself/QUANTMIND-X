---
title: Timeseries in DoEasy library (part 55): Indicator collection class
url: https://www.mql5.com/en/articles/8576
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:33:42.105715
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/8576&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070384892446971107)

MetaTrader 5 / Examples


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8576#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8576#node02)
- [Test EA](https://www.mql5.com/en/articles/8576#node03)
- [What's next?](https://www.mql5.com/en/articles/8576#node04)


### Concept

In this article, we are going to finish descendant classes of the abstract indicator base object which we started developing [in the previous article](https://www.mql5.com/en/articles/8508).

Following the general concept of library object construction and so that organisation of indicator objects doesn’t differ from other library objects, into indicator objects we must add their description. Along the way, improve storage of these objects in their collection, namely eliminate shortcomings that we made in the previous article when creating indicator collections and adding objects to the collection.

Note that, abstract indicator base object and indicator objects being descendants of this object are independent subjects which don’t cross with multi-symbol multi-period indicators which we created earlier to use when creating our custom indicators with the use of the library.

Abstract indicator object and its descendants are indicator objects which we must use for indicator EAs and for searching various combinations of data and statuses of various indicator values.

### Improving library classes

As usual, when developing library object classes first, add required text messages used by objects when their descriptions are displayed. For indicator objects, messages for displaying all possible paramaters of all standard indicators are required.

To file\\MQL5\\Include\\DoEasy\ **Data.mqh**  add indices of new messages:

```
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

//--- CIndicatorsCollection
   MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST,                // Error. Failed to add indicator object to the list

  };
//+------------------------------------------------------------------+
```

and text messages corresponding to newly added indices:

```
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
   {"K-period (number of bars for calculations)"},
   {"D-period (period of first smoothing)"},
   {"Final smoothing"},
   {"Stochastic calculation method"},
   {"Chande Momentum period"},
   {"Smoothing factor period"},

   {"Error. Failed to add indicator object to list"},

  };
//+---------------------------------------------------------------------+
```

To display certain indicator parameters such as method of averaging, type of price and volume for calculation, etc., add some functions in the file of library’s service functions in \\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh**:

```
//+------------------------------------------------------------------+
//| Return timeframe description                                     |
//+------------------------------------------------------------------+
string TimeframeDescription(const ENUM_TIMEFRAMES timeframe)
  {
   return StringSubstr(EnumToString((timeframe>PERIOD_CURRENT ? timeframe : (ENUM_TIMEFRAMES)Period())),7);
  }
//+------------------------------------------------------------------+
//| Return volume description for calculation                        |
//+------------------------------------------------------------------+
string AppliedVolumeDescription(const ENUM_APPLIED_VOLUME volume)
  {
   return StringSubstr(EnumToString(volume),7);
  }
//+------------------------------------------------------------------+
//| Return indicator type description                                |
//+------------------------------------------------------------------+
string IndicatorTypeDescription(const ENUM_INDICATOR indicator)
  {
   return StringSubstr(EnumToString(indicator),4);
  }
//+------------------------------------------------------------------+
//| Return averaging method description                              |
//+------------------------------------------------------------------+
string AveragingMethodDescription(const ENUM_MA_METHOD method)
  {
   return StringSubstr(EnumToString(method),5);
  }
//+------------------------------------------------------------------+
//| Return applied price description                                 |
//+------------------------------------------------------------------+
string AppliedPriceDescription(const ENUM_APPLIED_PRICE price)
  {
   return StringSubstr(EnumToString(price),6);
  }
//+------------------------------------------------------------------+
//| Return stochastic price calculation description                  |
//+------------------------------------------------------------------+
string StochPriceDescription(const ENUM_STO_PRICE price)
  {
   return StringSubstr(EnumToString(price),4);
  }
//+------------------------------------------------------------------+
```

It is simple: retrieve the substring from the required position from text representation of an enumeration value and finally get the name of indicator, calculation method or a type of volume and price.

Each indicator possesses its definite set of parameters. These parameters may be set for the indicator with the help of the array of indicator parameter structures [MqlParam](https://www.mql5.com/ru/docs/constants/structures/mqlparam). This is what we do during creation of each indicator object. Respectively, for each indicator all values of the array of these structures may be displayed in the journal. Data of value of the parameters inherent only to this indicator type will be available for different indicators in each array cell. But for several indicators of the same type the properties equal by their purpose and differing only by their value will be specified in each array cell.

Thus, for each indicator a method may be written which will display in the journal a set of indicator parameters with the values set for them. This is true only for standard indicators since for them we definitely know a set of parameters of each specific indicator.

This will be a virtual method. Write it in the abstract indicator object class in \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **IndicatorDE.mqh**:

```
//--- Display the description of indicator object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display (1) a short description, (2) description of indicator object parameters in the journal (implementation in the descendants)
   virtual void      PrintShort(void) {;}
   virtual void      PrintParameters(void) {;}

  };
//+------------------------------------------------------------------+
```

In this class, this method does nothing. And implementation of indicator data display method in the journal will be done in descendant objects.

Each descendant class will possess its own method since its own set of parameters is inherent to each indicator.

In the closed parametric constructor replace the string of getting the indicator type description

```
   this.m_ind_type=::StringSubstr(::EnumToString(ind_type),4);
```

by getting description with the use of a new service function described above:

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

//--- Save real properties
   this.m_double_prop[this.IndexProp(INDICATOR_PROP_EMPTY_VALUE)]=EMPTY_VALUE;
//--- Save string properties
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_SYMBOL)]   = (symbol==NULL || symbol=="" ? ::Symbol() : symbol);
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_NAME)]     = name;
   this.m_string_prop[this.IndexProp(INDICATOR_PROP_SHORTNAME)]= shortname;
  }
//+------------------------------------------------------------------+
```

In the method which displays indicator properties in the journal, in the end of the listing after display of all indicator object properties add the call of the method displaying in the journal the set of indicator parameters with the values set for them:

```
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
   this.PrintParameters();
   ::Print("================== ",CMessage::Text(MSG_LIB_PARAMS_LIST_END),": \"",this.GetStatusDescription(),"\" ==================\n");
  }
//+------------------------------------------------------------------+
```

Now, in case if PrintPatameters() method is present in descendant class of the abstract indicator base object, when Print() method is called the PrintPatameters() virtual method will be called from descendant class in which the display of indicator parameters will be implemented in the journal.

Since each indicator type possesses its own parameter set, in each descendant class we must implement its own PrintPatameters() method.

Such methods are already written for each indicator object. They are all of the same type in terms of logic, but differ in contents. Methods are written for all indicators except for one - the custom indicator because method implementation for it will differ for the reason that we cannot know in advance the set of indicator parameters by contrast with all standard indicators.

Let's analyze these methods for each of descendant objects.

**The class of indicator object Accelerator Oscillator:**

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
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_INTEGER property);

//--- Display (1) a short description, (2) description of indicator object parameters in the journal
   virtual void      PrintShort(void);
   virtual void      PrintParameters(void) {;}
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
//| Display a short description of indicator object in the journal   |
//+------------------------------------------------------------------+
void CIndAC::PrintShort(void)
  {
   ::Print(GetStatusDescription()," ",this.Name()," ",this.Symbol()," ",TimeframeDescription(this.Timeframe())," [",this.Handle(),"]");
  }
//+------------------------------------------------------------------+
```

Here, virtual method for parameter display is only declared and it is empty since AC indicator has no inputs. It could be ommitted here because if the virtual method in descendant class is missing, the parent class virtual method will be called. But here we wrote it so that all descendant classes have the same structure of their methods.

In the display method of indicator short description add the display of created indicator handle for this object. Such changes in this method are made for all indicator objects. We will not analyze them further. Instead, we will analyze only description display methods for indicator parameters.

**The class of indicator object Accumulation/Distribution and its **method of description display for indicator parameters**:**

```
//+------------------------------------------------------------------+
//|                                                        IndAD.mqh |
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
//| Standard indicator Accumulation/Distribution                     |
//+------------------------------------------------------------------+
class CIndAD : public CIndicatorDE
  {
private:

public:
   //--- Constructor
                     CIndAD(const string symbol,const ENUM_TIMEFRAMES timeframe,MqlParam &mql_param[]) :
                        CIndicatorDE(IND_AD,symbol,timeframe,
                                     INDICATOR_STATUS_STANDART,
                                     INDICATOR_GROUP_VOLUMES,
                                     "Accumulation/Distribution",
                                     "AD("+symbol+","+TimeframeDescription(timeframe)+")",mql_param) {}
   //--- Supported indicator properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_INDICATOR_PROP_INTEGER property);

//--- Display (1) a short description, (2) description of indicator object parameters in the journal
   virtual void      PrintShort(void);
   virtual void      PrintParameters(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if indicator supports a passed                     |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CIndAD::SupportProperty(ENUM_INDICATOR_PROP_INTEGER property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if indicator supports a passed                     |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CIndAD::SupportProperty(ENUM_INDICATOR_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a short description of indicator object in the journal   |
//+------------------------------------------------------------------+
void CIndAD::PrintShort(void)
  {
   ::Print(GetStatusDescription()," ",this.Name()," ",this.Symbol()," ",TimeframeDescription(this.Timeframe())," [",this.Handle(),"]");
  }
//+------------------------------------------------------------------+
//| Display parameter description of indicator object in the journal |
//+------------------------------------------------------------------+
void CIndAD::PrintParameters(void)
  {
   ::Print(" --- ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_IND_PARAMETERS)," --- ");
   //--- applied_volume
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_APPLIED_VOLUME),": ",AppliedVolumeDescription((ENUM_APPLIED_VOLUME)m_mql_param[0].integer_value));
  }
//+------------------------------------------------------------------+
```

Accumulation/Distribution indicator possesses only one input - volume type for calculation. Therefore, the array of inputs structures possesses only one cell which stores this parameter. Respectively, to display it in the journal we get it from the array by index 0 from integer data of the structure and send to the journal using the service function described above. Before displaying the parameter value describe it from earlier written text messages of the library.

Further, analyze only methods of description display of indicator inputs.

**The method of description display for class indicator parameters of indicator object Average Directional Index:**

```
//+------------------------------------------------------------------+
//| Display parameter description of indicator object in the journal |
//+------------------------------------------------------------------+
void CIndADX::PrintParameters(void)
  {
   ::Print(" --- ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_IND_PARAMETERS)," --- ");
   //--- adx_period
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_PERIOD),": ",(string)m_mql_param[0].integer_value);
  }
//+------------------------------------------------------------------+
```

ADX indicator possesses one input namely, calculation period. Therefore, in the same way display description of only one array cell by index 0 from the integer data of the structure.

**The method of description display for class indicator parameters of indicator object Alligator:**

```
//+------------------------------------------------------------------+
//| Display parameter description of indicator object in the journal |
//+------------------------------------------------------------------+
void CIndAlligator::PrintParameters(void)
  {
   ::Print(" --- ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_IND_PARAMETERS)," --- ");
   //--- jaw_period
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_JAW_PERIOD),": ",(string)m_mql_param[0].integer_value);
   //--- jaw_shift
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_JAW_SHIFT),": ",(string)m_mql_param[1].integer_value);
   //--- teeth_period
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_TEETH_PERIOD),": ",(string)m_mql_param[2].integer_value);
   //--- teeth_shift
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_TEETH_SHIFT),": ",(string)m_mql_param[3].integer_value);
   //--- lips_period
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_LIPS_PERIOD),": ",(string)m_mql_param[4].integer_value);
   //--- lips_shift
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_LIPS_SHIFT),": ",(string)m_mql_param[5].integer_value);
   //--- ma_method
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_MA_METHOD),": ",AveragingMethodDescription((ENUM_MA_METHOD)m_mql_param[6].integer_value));
   //--- applied_price
   ::Print(
           " - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_APPLIED_PRICE),": ",
           (m_mql_param[7].integer_value<10 ? AppliedPriceDescription((ENUM_APPLIED_PRICE)m_mql_param[7].integer_value) : (string)m_mql_param[7].integer_value)
          );
  }
//+------------------------------------------------------------------+
```

Indicator Alligator possesses eight inputs, therefore display them one by one in correspondence with their sequence order at creation of indicator:

- “Jaws” line calculation period - index in the array 0

- Horizontal shift of “Jaws” line - index in the array 1

- “Teeth” line calculation period - index in the array 2

- Horizontal shift of “Teeth” line - index in the array 3

- “Lips” line calculation period - index in the array 4

- Horizontal shift of “Lips” line - index in the array 5

- Smoothing type - index in the array 6
- Price type or indicator handle - index in the array 7


All data possess the integer type therefore, display the description of structure integer data from corresponding array cells.

The last parameter can store either the type of the price by which the indicator is constructed or the indicator handle on which data Alligator is constructed. Therefore, first check the value in array cell. If the value is less than 10 it means that the indicator is constructed on one of possible price types and display the description of price type. If the value is 10 and more it means that the indicator is constructed on data of another indicator which handle is written in the array - display the handle value of this indicator.

**The method of description display for class indicator parameters of indicator object Envelopes:**

```
//+------------------------------------------------------------------+
//| Display parameter description of indicator object in the journal |
//+------------------------------------------------------------------+
void CIndEnvelopes::PrintParameters(void)
  {
   ::Print(" --- ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_IND_PARAMETERS)," --- ");
   //--- ma_period
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_PERIOD),": ",(string)m_mql_param[0].integer_value);
   //--- ma_shift
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_SHIFT),": ",(string)m_mql_param[1].integer_value);
   //--- ma_method
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_MA_METHOD),": ",AveragingMethodDescription((ENUM_MA_METHOD)m_mql_param[2].integer_value));
   //--- applied_price
   ::Print(
           " - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_APPLIED_PRICE),": ",
           (m_mql_param[3].integer_value<10 ? AppliedPriceDescription((ENUM_APPLIED_PRICE)m_mql_param[3].integer_value) : (string)m_mql_param[3].integer_value)
          );
   //--- deviation
   ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_IND_TEXT_DEVIATION),": ",::DoubleToString(m_mql_param[4].double_value,3));

  }
//+------------------------------------------------------------------+
```

Envelopes indicator possesses five inputs. The last parameter "Deviation of channel borders from the central line" has the real type. Therefore, to display the description of this parameter, take the value from real data  of inputs structure.

All remaining methods of indicator object classes repeat logic of above considered ones and we will not analyze their methods. You can study them in the files attached to this article.

Place all indicator objects created to the collection list of indicators. In the terminal, when creating any number of indicators with identical parameters one indicator is created in fact and all calls proceed to it. Therefore, to create indicator objects and to place them to the collection list we must control if the collection list has the same indicator with the same type, symbol/timeframe and parameters as that one we want to place in the list. This is simply because absolutely same indicator objects will have the same handle of the created indicator for all of them. It means that this is one and the same indicator object.

Add the necessary changes into indicator object collection class in file \\MQL5\\Include\\DoEasy\\Collections\ **IndicatorsCollection.mqh** which we started implementing [in the previous article](https://www.mql5.com/en/articles/8508#node04). To search and compare two indicator objects we used the [Search()](https://www.mql5.com/ru/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) method of the class of the dynamic array of pointers to instances of the CObject class and its descendants [from library standard delivery](https://www.mql5.com/ru/docs/standardlibrary/datastructures/carrayobj). But this method cannot definitely determine the equality of the two objects which include structures. The purpose of this method is to compare one specified property of two same-type objects. In indicator objects actively use the array of parameter structures of [MqlParam](https://www.mql5.com/ru/docs/constants/structures/mqlparam) indicator in which each property of the structure in the array must be compared element by element. Fortunately, in all library objects we have a default method IsEqual() for precise comparison of two same-type objects. To compare two same-type objects for equality this method will be used.

In the private class section declare the method returning the indicator object index in the collection list:

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
//--- Return the indicator index in the list
   int                     Index(CIndicatorDE *compared_obj);

public:
```

In the very end of class body listing declare two public methods - to display full and short descriptions of indicator objects which are in the collection list:

```
//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(void);
   void                    PrintShort(void);

//--- Constructor
                           CIndicatorsCollection();

  };
//+------------------------------------------------------------------+
```

Implement declared methods outside the class body.

**The method displaying the full collection description in the journal:**

```
//+------------------------------------------------------------------+
//| Display full collection description in the journal               |
//+------------------------------------------------------------------+
void CIndicatorsCollection::Print(void)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CIndicatorDE *ind=m_list.At(i);
      if(ind==NULL)
         continue;
      ind.Print();
     }
  }
//+------------------------------------------------------------------+
```

In a loop by collection listget another indicator object and display its full description in the journal.

**The method displaying the short collection description in the journal:**

```
//+------------------------------------------------------------------+
//| Display the short collection description in the journal          |
//+------------------------------------------------------------------+
void CIndicatorsCollection::PrintShort(void)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CIndicatorDE *ind=m_list.At(i);
      if(ind==NULL)
         continue;
      ind.PrintShort();
     }
  }
//+------------------------------------------------------------------+
```

In a loop by collection listget another indicator object and display its short description in the journal.

**The method returning the indicator object index in the collection list**

:

```
//+------------------------------------------------------------------+
//| Return the indicator index in the list                           |
//+------------------------------------------------------------------+
int CIndicatorsCollection::Index(CIndicatorDE *compared_obj)
  {
   int total=this.m_list.Total();
   for(int i=0;i<total;i++)
     {
      CIndicatorDE *indicator=m_list.At(i);
      if(indicator==NULL)
         continue;
      if(indicator.IsEqual(compared_obj))
         return i;
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

In a loop by collection listget another indicator object, compare it with the indicator object, pointer to which is passed to the method and return loop index if the objects are equal. Upon the loop completion (if all objects are unequal) return -1.

In all methods of creation of new indicator objects and their placement to the collection list the changes identical for each method were made. This is made to exclude possible memory leakages at unsuccessful creation of the object or its unsuccessful placement to the list.

**Let's use the method of Accelerator Oscillator indicator creation as an example:**

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
//--- If such indicator is already in the list
   int index=this.Index(indicator);
   if(index!=WRONG_VALUE)
     {
      //--- Remove created object, get indicator object from the list and return indicator handle
      delete indicator;
      indicator=this.m_list.At(index);
      return indicator.Handle();
     }
//--- If such indicator is not in the list
   else
     {
      //--- If failed to add indicator object to the list
      //--- display the appropriate message, remove object and return INVALID_HANDLE
      if(!this.m_list.Add(indicator))
        {
         ::Print(CMessage::Text(MSG_LIB_SYS_FAILED_ADD_IND_TO_LIST));
         delete indicator;
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

Check availability of indicator object in the list by its index.

If the index exceeds -1 it means that it is in the list and the newly created object must be removed.

If such indicator is not in the list and we failed to add it into the collection list for some reasonremove the newly created object.

This will exclude memory leakages in case of unsuccessful placement of the new object into the collection list.

Such changes were made in all methods of indicator object creation and we will not analyze them. You can study them in the files attached to this article.

To search indicator object in the collection list and to get a pointer to it from this list we need the methods which return pointers to the required object. To the method pass the type of the necessary indicator, its symbol, timeframe and parameters (for each indicator its parameters correspond to indicator type). In the end, the pointer to the indicator object found in the list must be received.

In the previous article I wrote such a method to get the pointer to Accelerator Oscillator indicator. It is the simplest one since AC indicator possesses no inputs and we need only to find the necessary object by symbol and timeframe:

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

To search indicators possessing inputs, I will create a temporary indicator objectwith set parameters and search a match in collection list:

```
//+------------------------------------------------------------------+
//| Return pointer to indicator object                               |
//| Accumulation/Distribution                                        |
//+------------------------------------------------------------------+
CIndicatorDE *CIndicatorsCollection::GetIndAD(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume)
  {
   MqlParam param[1];
   param[0].type=TYPE_INT;
   param[0].integer_value=applied_volume;
   CIndicatorDE *tmp=this.CreateIndicator(IND_AD,param,symbol,timeframe);
   if(tmp==NULL)
      return NULL;
   int index=this.Index(tmp);
   delete tmp;
   return(index>WRONG_VALUE ? this.m_list.At(index) : NULL);
  }
//+------------------------------------------------------------------+
```

If such object is found in the list, its index is returned. If not - return -1.

Remaining methods of pointers return to indicator objects in the list are identical to those considered above, but they possess other parameters for creating the indicator object. For example, to return the pointer to Alligator indicator object create the array consisting of eight parameters:

```
//+------------------------------------------------------------------+
//| Return pointer to indicator object Alligator                     |
//+------------------------------------------------------------------+
CIndicatorDE *CIndicatorsCollection::GetIndAlligator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                                     const int jaw_period,
                                                     const int jaw_shift,
                                                     const int teeth_period,
                                                     const int teeth_shift,
                                                     const int lips_period,
                                                     const int lips_shift,
                                                     const ENUM_MA_METHOD ma_method,
                                                     const ENUM_APPLIED_PRICE applied_price)
  {
   MqlParam param[8];
   param[0].type=TYPE_INT;
   param[0].integer_value=jaw_period;
   param[1].type=TYPE_INT;
   param[1].integer_value=jaw_shift;
   param[2].type=TYPE_INT;
   param[2].integer_value=teeth_period;
   param[3].type=TYPE_INT;
   param[3].integer_value=teeth_shift;
   param[4].type=TYPE_INT;
   param[4].integer_value=lips_period;
   param[5].type=TYPE_INT;
   param[5].integer_value=lips_shift;
   param[6].type=TYPE_INT;
   param[6].integer_value=ma_method;
   param[7].type=TYPE_INT;
   param[7].integer_value=applied_price;
   CIndicatorDE *tmp=this.CreateIndicator(IND_ALLIGATOR,param,symbol,timeframe);
   if(tmp==NULL)
      return NULL;
   int index=this.Index(tmp);
   delete tmp;
   return(index>WRONG_VALUE ? this.m_list.At(index) : NULL);
  }
//+------------------------------------------------------------------+
```

All remaining things are identical to the above considered method of return of the pointer to indicator object Accumulation/Distribution.

In each of the methods temporary indicator object is removed obligatory. It serves as the reference for searching a match in the collection list.

I will not analyze remaining similar methods. They are identical to the two methods we have just considered.

**This concludes the improvement of classes within the frames of this articles.**

### Test EA

To perform the test of indicator creation in EAs take the test EA [from article 39](https://www.mql5.com/en/articles/7724)

and save it in a new folder \\MQL5\\Experts\\TestDoEasy\ **Part55\** under a new name **TestDoEasyPart55.mq5**.

Mainly, the improvements will be minor. In one of the previous articles I moved the function of work with events in tester EventsHandling() to the library - to file Engine.mqh. Therefore, remove this function from EA code and in handler OnTick() replace its call from EA file

```
//--- If work in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Work in the timer
      PressButtonsControl();        // Button press control
      EventsHandling();             // Work with events
     }
```

with the call from the library:

```
//--- If work in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Work in the timer
      PressButtonsControl();        // Button press control
      engine.EventsHandling();      // Work with events
     }
```

Remove the code block from handler OnTick() which displays a comment with data on the current bar to chart:

```
//--- Get the zero bar of the current timeseries
   CBar *bar=engine.SeriesGetBar(NULL,PERIOD_CURRENT,0);
   if(bar==NULL)
      return;
//--- Create parameters string of the current bar similar to the one
//--- displayed by the bar object description:
//--- bar.Header()+": "+bar.ParameterDescription()
   string parameters=
     (TextByLanguage("Bar \"")+Symbol()+"\" "+TimeframeDescription((ENUM_TIMEFRAMES)Period())+"[0]: "+TimeToString(bar.Time(),TIME_DATE|TIME_MINUTES|TIME_SECONDS)+
      ", O: "+DoubleToString(engine.SeriesOpen(NULL,PERIOD_CURRENT,0),Digits())+
      ", H: "+DoubleToString(engine.SeriesHigh(NULL,PERIOD_CURRENT,0),Digits())+
      ", L: "+DoubleToString(engine.SeriesLow(NULL,PERIOD_CURRENT,0),Digits())+
      ", C: "+DoubleToString(engine.SeriesClose(NULL,PERIOD_CURRENT,0),Digits())+
      ", V: "+(string)engine.SeriesTickVolume(NULL,PERIOD_CURRENT,0)+
      ", Real: "+(string)engine.SeriesRealVolume(NULL,PERIOD_CURRENT,0)+
      ", Spread: "+(string)engine.SeriesSpread(NULL,PERIOD_CURRENT,0)
     );
//--- Display the data received from the bar object in the first line of the chart comment,
//--- while the second line contains the methods of receiving timeseries price data
   Comment(bar.Header(),": ",bar.ParameterDescription(),"\n",parameters);
```

Thus, “new tick” event handler will be as follows:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Handle the NewTick event in the library
   engine.OnTick(rates_data);

//--- If work in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Work in the timer
      PressButtonsControl();        // Button press control
      engine.EventsHandling();      // Work with events
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

In library initializing function OnInitDoEasy() in the block of display of used symbol list replace a number of symbols, set by value

```
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
```

with a number of symbols specified in the macro substitution:

```
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
         ArrayResize(array_symbols,ArraySize(array_symbols)+1,SYMBOLS_COMMON_TOTAL);
         array_symbols[ArraySize(array_symbols)-1]=symbol.Name();
        }
      ArrayPrint(array_symbols);
     }
#endif
```

since in MetaTrader 5, starting with version 2430 the total number of working symbols is changed and this number is checked by the library and is set automatically to macro substitution SYMBOLS\_COMMON\_TOTAL, declared in file \\MQL5\\Include\\DoEasy\\Defines.mqh.

Temporarily and only to check the creation of indicator objects create two indicators of the same type but with different parameters. Until normal creation of indicators in our programs is not implemented simply create them in the library initializing function:

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
      string en_n="\nThe number of symbols on server "+(string)total+".\nMaximal number: "+(string)SYMBOLS_COMMON_TOTAL+" symbols.";
      string caption=TextByLanguage("Attention!");
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
   Print(TextByLanguage("--- Initializing the \"DoEasy\" library ---"));
//--- Fill in the array of used symbols
   CreateUsedSymbolsArray((ENUM_SYMBOLS_MODE)used_symbols_mode,InpUsedSymbols,array_used_symbols);
//--- Set the type of the used symbol list in the symbol collection and fill in the list of symbol timeseries
   engine.SetUsedSymbols(array_used_symbols);
//--- Displaying the selected mode of working with the symbol object collection in the journal
   string num=
     (
      used_symbols_mode==SYMBOLS_MODE_CURRENT ? ": \""+Symbol()+"\"" :
      TextByLanguage(". Number of used symbols: ",". The number of symbols used: ")+(string)engine.GetSymbolsCollectionTotal()
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
         ArrayResize(array_symbols,ArraySize(array_symbols)+1,SYMBOLS_COMMON_TOTAL);
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
         TextByLanguage("Work only with the current Period: ")+TimeframeDescription((ENUM_TIMEFRAMES)Period())   :
      InpModeUsedTFs==TIMEFRAMES_MODE_LIST      ?
         TextByLanguage("Work with a predefined list of Periods:")                                              :
      TextByLanguage("Work with the full list of all Periods:")
     );
   Print(mode);
//--- Implement displaying the list of used timeframes only for MQL5 - MQL4 has no ArrayPrint() function
#ifdef __MQL5__
   if(InpModeUsedTFs!=TIMEFRAMES_MODE_CURRENT)
      ArrayPrint(array_used_periods);
#endif
//--- Create timeseries of all used symbols
   engine.SeriesCreateAll(array_used_periods);
//--- Check created timeseries - display descriptions of all created timeseries in the journal
//--- (true - only created ones, false - created and declared ones)
   engine.GetTimeSeriesCollection().PrintShort(false); // Short descriptions
   //engine.GetTimeSeriesCollection().Print(true);      // Full descriptions

//--- Create indicators
   engine.GetIndicatorsCollection().CreateAMA(Symbol(),Period(),9,2,30,0,PRICE_CLOSE);
   engine.GetIndicatorsCollection().CreateAMA(Symbol(),Period(),10,3,32,5,PRICE_CLOSE);
   engine.GetIndicatorsCollection().Print();
   engine.GetIndicatorsCollection().PrintShort();

//--- Create resource text files
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_01",TextByLanguage("The sound of a falling coin 1"),sound_array_coin_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_02",TextByLanguage("Sound fallen coins"),sound_array_coin_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_03",TextByLanguage("Sound of coins"),sound_array_coin_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_04",TextByLanguage("The sound of a falling coin 2"),sound_array_coin_04);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_01",TextByLanguage("Click on the button sound 1"),sound_array_click_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_02",TextByLanguage("Click on the button sound 1"),sound_array_click_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_03",TextByLanguage("Click on the button sound 1"),sound_array_click_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_cash_machine_01",TextByLanguage("The sound of the cash machine"),sound_array_cash_machine_01);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_green",TextByLanguage("Image \"Green Spot lamp\""),img_array_spot_green);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_red",TextByLanguage("Image \"Red Spot lamp\""),img_array_spot_red);

//--- Pass all existing collections to the main library class
   engine.CollectionOnInit();

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
      //--- By default, the LONG_MAX value is set to all properties, which means "Do not track this property”
      //--- It can be enabled or disabled (set the value less than LONG_MAX or vice versa - set the LONG_MAX value) at any time and anywhere in the program
      /*
      for(int i=0;i<list.Total();i++)
        {
         CSymbol* symbol=list.At(i);
         if(symbol==NULL)
            continue;
        //--- Set control of the symbol price increase to 100 points
         symbol.SetControlBidInc(100000*symbol.Point());
        //--- Set control of the symbol price decrease to 100 points
         symbol.SetControlBidDec(100000*symbol.Point());
        //--- Set control of the symbol spread increase to 40 points
         symbol.SetControlSpreadInc(400);
        //--- Set control of the symbol spread decrease to 40 points
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
   Print(TextByLanguage("Library initialization time: "),TimeMSCtoString(end-begin,TIME_MINUTES|TIME_SECONDS));
  }
//+------------------------------------------------------------------+
```

Here we created two indicators Adaptive Moving Average, on the current symbol and timeframe, but with different input values.

Compile the EA and launch it on the chart in the terminal.

After its initializing, “Experts” journal displays library messages about initialization. Among them there are full and short lists of parameters of two created indicators:

```
Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.) 10425.23 USD, 1:100, Hedge, Demo account MetaTrader 5
--- Initializing the "DoEasy" library ---
Work only with the current symbol: "EURUSD"
Work with a predefined list of Periods:
"H1" "H4"
Symbol time series EURUSD:
- Timeseries "EURUSD" H1: Required: 1000, Actual: 1000, Created: 1000, On server: 6350
- Timeseries "EURUSD" H4: Required: 1000, Actual: 1000, Created: 1000, On server: 6255
============= The beginning of the event parameter list: "Standard indicator" =============
Indicator status: Standard indicator
Indicator type: AMA
Indicator timeframe: H1
Indicator handle: 10
Indicator group: Trend indicator
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

============= The beginning of the event parameter list: "Standard indicator" =============
Indicator status: Standard indicator
Indicator type: AMA
Indicator timeframe: H1
Indicator handle: 11
Indicator group: Trend indicator
------
Empty value for plotting, for which there is no drawing: EMPTY_VALUE
------
Indicator symbol: EURUSD
Indicator name: "Adaptive Moving Average"
Indicator shortname: "AMA(EURUSD,H1)"
--- Indicator parameters ---
- Averaging period: 10
- Fast MA period: 3
- Slow MA period: 32
- Horizontal shift of the indicator: 5
- Price type or handle: CLOSE
================== End of the parameter list: "Standard indicator" ==================

Standard indicator Adaptive Moving Average EURUSD H1 [10]
Standard indicator Adaptive Moving Average EURUSD H1 [11]
Library initialization time: 00:00:00.000
```

Although, the indicator type is one - AMA, but two handles of this indicator are created since parameters of indicators created were different. Therefore, these are two different indicators - each having its handle. Respectively, two indicator objects are created and placed in indicator collection.

Meanwhile, we can only create different indicators with different parameters. But to apply them in EAs, area for storing their data must be prepared. From that area, data in any required combinations of parameters may be received and used in programs to make decisions or to get statistical data. I will start doing all these things from the next article.

### What's next?

In the following article, implementing data storage and receiving from indicator objects in EAs will be started.

All files of the current version of the library are attached below together with the test EA file for MQL5. You can download them and test everything.

Note, that at the moment indicator collection class is under development, therefore  it is strictly recommended not to use it in your programs.

Leave your comments, questions and suggestions in the comments to the article.

[Back to contents](https://www.mql5.com/en/articles/8576#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8576](https://www.mql5.com/ru/articles/8576)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8576.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8576/mql5.zip "Download MQL5.zip")(3860.21 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/359140)**

![Neural networks made easy (Part 5): Multithreaded calculations in OpenCL](https://c.mql5.com/2/48/Neural_networks_made_easy_0065.png)[Neural networks made easy (Part 5): Multithreaded calculations in OpenCL](https://www.mql5.com/en/articles/8435)

We have earlier discussed some types of neural network implementations. In the considered networks, the same operations are repeated for each neuron. A logical further step is to utilize multithreaded computing capabilities provided by modern technology in an effort to speed up the neural network learning process. One of the possible implementations is described in this article.

![Neural networks made easy (Part 4): Recurrent networks](https://c.mql5.com/2/48/Neural_networks_made_easy_004.png)[Neural networks made easy (Part 4): Recurrent networks](https://www.mql5.com/en/articles/8385)

We continue studying the world of neural networks. In this article, we will consider another type of neural networks, recurrent networks. This type is proposed for use with time series, which are represented in the MetaTrader 5 trading platform by price charts.

![Practical application of neural networks in trading. Python (Part I)](https://c.mql5.com/2/40/neural_python.png)[Practical application of neural networks in trading. Python (Part I)](https://www.mql5.com/en/articles/8502)

In this article, we will analyze the step-by-step implementation of a trading system based on the programming of deep neural networks in Python. This will be performed using the TensorFlow machine learning library developed by Google. We will also use the Keras library for describing neural networks.

![Timeseries in DoEasy library (part 54): Descendant classes of abstract base indicator](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library__6.png)[Timeseries in DoEasy library (part 54): Descendant classes of abstract base indicator](https://www.mql5.com/en/articles/8508)

The article considers creation of classes of descendant objects of base abstract indicator. Such objects will provide access to features of creating indicator EAs, collecting and getting data value statistics of various indicators and prices. Also, create indicator object collection from which getting access to properties and data of each indicator created in the program will be possible.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/8576&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070384892446971107)

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