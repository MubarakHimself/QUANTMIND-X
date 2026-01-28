---
title: Other classes in DoEasy library (Part 69): Chart object collection class
url: https://www.mql5.com/en/articles/9260
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:12:33.809723
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/9260&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083407465216744223)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9260#node01)
- [Improving library classes](https://www.mql5.com/en/articles/9260#node02)
- [Chart object collection class](https://www.mql5.com/en/articles/9260#node03)
- [Test](https://www.mql5.com/en/articles/9260#node04)
- [What's next?](https://www.mql5.com/en/articles/9260#node05)


### Concept

I have already prepared the functionality for working with chart objects. Now we have the chart object with the lists of chart window objects, which in turn contain the lists of indicators attached to them. It is time to combine this into a single chart object collection that will provide a convenient access to any chart open in the terminal. The collection will allow us to search, sort and obtain the chart object lists by any property and handle a selected chart or several charts conveniently.

### Improving library classes

First, let's add new messages to the library.

In \\MQL5\\Include\\DoEasy\ **Data.mqh**, add new message indices:

```
   MSG_CHART_OBJ_CHART_WINDOW,                        // Main chart window
   MSG_CHART_OBJ_CHART_SUBWINDOW,                     // Chart subwindow
   MSG_CHART_OBJ_CHART_SUBWINDOWS_NUM,                // Subwindows
   MSG_CHART_OBJ_INDICATORS_MW_NAME_LIST,             // Indicators in the main chart window
   MSG_CHART_OBJ_INDICATORS_SW_NAME_LIST,             // Indicators in the chart window
   MSG_CHART_OBJ_INDICATOR,                           // Indicator
   MSG_CHART_OBJ_INDICATORS_TOTAL,                    // Indicators
   MSG_CHART_OBJ_WINDOW_N,                            // Window
   MSG_CHART_OBJ_INDICATORS_NONE,                     // No indicators

//--- CChartObjCollection
   MSG_CHART_COLLECTION_TEXT_CHART_COLLECTION,        // Chart collection
   MSG_CHART_COLLECTION_ERR_FAILED_CREATE_CHART_OBJ,  // Failed to create a new chart object
   MSG_CHART_COLLECTION_ERR_FAILED_ADD_CHART,         // Failed to add a chart object to the collection

  };
//+------------------------------------------------------------------+
```

and the message texts corresponding to the newly added indices:

```
   {"Главное окно графика","Main chart window"},
   {"Подокно графика","Chart subwindow"},
   {"Подокон","Subwindows"},
   {"Индикаторы в главном окне графика","Indicators in the main chart window"},
   {"Индикаторы в окне графика","Indicators in the chart window"},
   {"Индикатор","Indicator"},
   {"Индикаторов","Indicators total"},
   {"Окно","Window"},
   {"Отсутствуют","No indicators"},

//--- CChartObjCollection
   {"Коллекция чартов","Chart collection"},
   {"Не удалось создать новый объект-чарт","Failed to create new chart object"},
   {"Не удалось добавить объект-чарт в коллекцию","Failed to add chart object to collection"},

  };
//+---------------------------------------------------------------------+
```

The chart object collection should track changes in the number of open charts, as well as changes in the number of open windows in each chart to promptly update its lists of charts and windows that are opened in them. Some changes can be tracked in the [OnChartEvent()](https://www.mql5.com/en/docs/event_handlers/onchartevent) handler. However, according to the tests, the handler mainly specifies that the chart has undergone certain changes — chart change event ( [CHARTEVENT\_CHART\_CHANGE](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)), i.e. there are no specifics here. Therefore, I will work in the program timer and track the changes in the number of open charts and windows in them. Other changes that may occur on the chart can be tracked either via the OnChartEvent() handler described above or by inheriting the chart object and the chart window object from the CBaseObjExt library object, which in turn is a descendant of the base object of all [CBaseObj](https://www.mql5.com/en/articles/7071#node01) library object and provides an additional event functionality to its descendant objects. This is necessary in case we will require such a functionality for working with charts later.

Charts are handled mainly in semi-automatic mode. So in order to define changes in the number of open charts and windows, it will be sufficient to check the current number of charts and windows comparing it with the previous number twice a second. If there were no changes, then nothing needs to be done. If there are changes in the number of windows and charts, we are going to update data in our collection.

To let the chart objects work in the timer, we will need yet another chart object collection timer counter. Each collection working in the timer features its own timer counter allowing to track the specified collection update frequency. Apart from the counter parameters, we need to add the new collection ID since each each object collection features its own ID allowing us to define the collection a certain object list belongs to.

In the "Macro substitutions" section of \\MQL5\\Include\\DoEasy\ **Defines.mqh**, set the parameters of the chart object collection timer counter and the ID of the chart object collection list:

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- Describe the function with the error line number
#define DFUN_ERR_LINE                  (__FUNCTION__+(TerminalInfoString(TERMINAL_LANGUAGE)=="Russian" ? ", Page " : ", Line ")+(string)__LINE__+": ")
#define DFUN                           (__FUNCTION__+": ")        // "Function description"
#define COUNTRY_LANG                   ("Russian")                // Country language
#define END_TIME                       (D'31.12.3000 23:59:59')   // End date for account history data requests
#define TIMER_FREQUENCY                (16)                       // Minimal frequency of the library timer in milliseconds
#define TOTAL_TRADE_TRY                (5)                        // Default number of trading attempts
#define IND_COLORS_TOTAL               (64)                       // Maximum possible number of indicator buffer colors
#define IND_BUFFERS_MAX                (512)                      // Maximum possible number of indicator buffers
//--- Standard sounds
#define SND_ALERT                      "alert.wav"
#define SND_ALERT2                     "alert2.wav"
#define SND_CONNECT                    "connect.wav"
#define SND_DISCONNECT                 "disconnect.wav"
#define SND_EMAIL                      "email.wav"
#define SND_EXPERT                     "expert.wav"
#define SND_NEWS                       "news.wav"
#define SND_OK                         "ok.wav"
#define SND_REQUEST                    "request.wav"
#define SND_STOPS                      "stops.wav"
#define SND_TICK                       "tick.wav"
#define SND_TIMEOUT                    "timeout.wav"
#define SND_WAIT                       "wait.wav"
//--- Parameters of the orders and deals collection timer
#define COLLECTION_ORD_PAUSE           (250)                      // Orders and deals collection timer pause in milliseconds
#define COLLECTION_ORD_COUNTER_STEP    (16)                       // Increment of the orders and deals collection timer counter
#define COLLECTION_ORD_COUNTER_ID      (1)                        // Orders and deals collection timer counter ID
//--- Parameters of the account collection timer
#define COLLECTION_ACC_PAUSE           (1000)                     // Account collection timer pause in milliseconds
#define COLLECTION_ACC_COUNTER_STEP    (16)                       // Account timer counter increment
#define COLLECTION_ACC_COUNTER_ID      (2)                        // Account timer counter ID
//--- Symbol collection timer 1 parameters
#define COLLECTION_SYM_PAUSE1          (100)                      // Pause of the symbol collection timer 1 in milliseconds (for scanning market watch symbols)
#define COLLECTION_SYM_COUNTER_STEP1   (16)                       // Increment of the symbol timer 1 counter
#define COLLECTION_SYM_COUNTER_ID1     (3)                        // Symbol timer 1 counter ID
//--- Symbol collection timer 2 parameters
#define COLLECTION_SYM_PAUSE2          (300)                      // Pause of the symbol collection timer 2 in milliseconds (for events of the market watch symbol list)
#define COLLECTION_SYM_COUNTER_STEP2   (16)                       // Increment of the symbol timer 2 counter
#define COLLECTION_SYM_COUNTER_ID2     (4)                        // Symbol timer 2 counter ID
//--- Trading class timer parameters
#define COLLECTION_REQ_PAUSE           (300)                      // Trading class timer pause in milliseconds
#define COLLECTION_REQ_COUNTER_STEP    (16)                       // Trading class timer counter increment
#define COLLECTION_REQ_COUNTER_ID      (5)                        // Trading class timer counter ID
//--- Parameters of the timeseries collection timer
#define COLLECTION_TS_PAUSE            (64)                       // Timeseries collection timer pause in milliseconds
#define COLLECTION_TS_COUNTER_STEP     (16)                       // Account timer counter increment
#define COLLECTION_TS_COUNTER_ID       (6)                        // Timeseries timer counter ID
//--- Parameters of the timer of indicator data timeseries collection
#define COLLECTION_IND_TS_PAUSE        (64)                       // Pause of the timer of indicator data timeseries collection in milliseconds
#define COLLECTION_IND_TS_COUNTER_STEP (16)                       // Increment of indicator data timeseries timer counter
#define COLLECTION_IND_TS_COUNTER_ID   (7)                        // ID of indicator data timeseries timer counter
//--- Parameters of the tick series collection timer
#define COLLECTION_TICKS_PAUSE         (64)                       // Tick series collection timer pause in milliseconds
#define COLLECTION_TICKS_COUNTER_STEP  (16)                       // Tick series timer counter increment step
#define COLLECTION_TICKS_COUNTER_ID    (8)                        // Tick series timer counter ID
//--- Parameters of the chart collection timer
#define COLLECTION_CHARTS_PAUSE        (500)                      // Chart collection timer pause in milliseconds
#define COLLECTION_CHARTS_COUNTER_STEP (16)                       // Chart timer counter increment
#define COLLECTION_CHARTS_COUNTER_ID   (9)                        // Chart timer counter ID
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
#define COLLECTION_TICKSERIES_ID       (0x7783)                   // Tick series collection list ID
#define COLLECTION_MBOOKSERIES_ID      (0x7784)                   // DOM series collection list ID
#define COLLECTION_MQL5_SIGNALS_ID     (0x7785)                   // MQL5 signals collection list ID
#define COLLECTION_CHARTS_ID           (0x7786)                   // Chart collection list ID
//--- Data parameters for file operations
```

In the enumeration of the chart integer properties, remove the CHART\_PROP\_WINDOW\_IS\_VISIBLE constant, since I have not found a practical use of this property for an object. Thus, the number of integer properties is reduced by 1 ( **from 67 to 66**):

```
#define CHART_PROP_INTEGER_TOTAL (66)                 // Total number of integer properties
```

Let's fix chart object sorting criteria by their properties by adding the object sorting properties:

```
//+------------------------------------------------------------------+
//| Possible chart sorting criteria                                  |
//+------------------------------------------------------------------+
#define FIRST_CHART_DBL_PROP  (CHART_PROP_INTEGER_TOTAL-CHART_PROP_INTEGER_SKIP)
#define FIRST_CHART_STR_PROP  (CHART_PROP_INTEGER_TOTAL-CHART_PROP_INTEGER_SKIP+CHART_PROP_DOUBLE_TOTAL-CHART_PROP_DOUBLE_SKIP)
enum ENUM_SORT_CHART_MODE
  {
//--- Sort by integer properties
   SORT_BY_CHART_ID = 0,                              // Sort by chart ID
   SORT_BY_CHART_TIMEFRAME,                           // Sort by chart timeframe
   SORT_BY_CHART_SHOW,                                // Sort by the price chart drawing attribute
   SORT_BY_CHART_IS_OBJECT,                           // Sort by chart object (OBJ_CHART) identification attribute
```

Also, remove the SORT\_BY\_CHART\_WINDOW\_IS\_VISIBLE constant from the constant list of the enumeration, since I will not use this property in the object.

All objects whose collections are present in the library feature their own lists that can be sorted by different object properties. The chart object collection list will also have the sorting ability for searching and selecting objects with required property values. For each such object, we create our own sorting methods set in \\MQL5\\Include\\DoEasy\\Services\ **Select.mqh**.

Add new methods to the CSelect class file for arranging the search and sorting of chart objects.

Include the chart object class file to the CSelect class file:

```
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
#include "..\Objects\Ticks\DataTick.mqh"
#include "..\Objects\Book\MarketBookOrd.mqh"
#include "..\Objects\MQLSignalBase\MQLSignal.mqh"
#include "..\Objects\Chart\ChartObj.mqh"
//+------------------------------------------------------------------+
```

Declare the new methods at the end of the class body listing:

```
//+------------------------------------------------------------------+
//| Methods of working with chart data                               |
//+------------------------------------------------------------------+
   //--- Return the list of charts with one of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByChartProperty(CArrayObj *list_source,ENUM_CHART_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByChartProperty(CArrayObj *list_source,ENUM_CHART_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByChartProperty(CArrayObj *list_source,ENUM_CHART_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the chart index with the maximum value of the (1) integer, (2) real and (3) string properties
   static int        FindChartMax(CArrayObj *list_source,ENUM_CHART_PROP_INTEGER property);
   static int        FindChartMax(CArrayObj *list_source,ENUM_CHART_PROP_DOUBLE property);
   static int        FindChartMax(CArrayObj *list_source,ENUM_CHART_PROP_STRING property);
   //--- Return the chart index with the minimum value of the (1) integer, (2) real and (3) string properties
   static int        FindChartMin(CArrayObj *list_source,ENUM_CHART_PROP_INTEGER property);
   static int        FindChartMin(CArrayObj *list_source,ENUM_CHART_PROP_DOUBLE property);
   static int        FindChartMin(CArrayObj *list_source,ENUM_CHART_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

Let's write their implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Methods of working with chart data                               |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of charts with one integer                       |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByChartProperty(CArrayObj *list_source,ENUM_CHART_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CChartObj *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of charts with one real                          |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByChartProperty(CArrayObj *list_source,ENUM_CHART_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CChartObj *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of charts with one string                        |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByChartProperty(CArrayObj *list_source,ENUM_CHART_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CChartObj *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the chart index in the list                               |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindChartMax(CArrayObj *list_source,ENUM_CHART_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CChartObj *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CChartObj *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the chart index in the list                               |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindChartMax(CArrayObj *list_source,ENUM_CHART_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CChartObj *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CChartObj *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the chart index in the list                               |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindChartMax(CArrayObj *list_source,ENUM_CHART_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CChartObj *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CChartObj *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the chart index in the list                               |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindChartMin(CArrayObj* list_source,ENUM_CHART_PROP_INTEGER property)
  {
   int index=0;
   CChartObj *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CChartObj *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the chart index in the list                               |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindChartMin(CArrayObj* list_source,ENUM_CHART_PROP_DOUBLE property)
  {
   int index=0;
   CChartObj *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CChartObj *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the chart index in the list                               |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindChartMin(CArrayObj* list_source,ENUM_CHART_PROP_STRING property)
  {
   int index=0;
   CChartObj *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CChartObj *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

I have considered these methods in detail [in the "Arranging the search" section of the article 3](https://www.mql5.com/en/articles/5687#node01).

When changing the number of windows attached to the chart, window indices change. Therefore, we may sometimes need the new window index. In the public section of the class in the \\MQL5\\Include\\DoEasy\\Objects\\Chart\ **ChartWnd.mqh** chart window object class file, write the new method for setting the window index:

```
//+------------------------------------------------------------------+
//| Chart window indicator object class                              |
//+------------------------------------------------------------------+
class CWndInd : public CObject
  {
private:
   long              m_chart_id;                         // Chart ID
   string            m_name;                             // Indicator short name
   int               m_index;                            // Window index on the chart
   int               m_handle;                           // Indicator handle
public:
//--- Return itself
   CWndInd          *GetObject(void)                     { return &this;         }
//--- Return (1) indicator name, (2) window index and (3) indicator handle
   string            Name(void)                    const { return this.m_name;   }
   int               Index(void)                   const { return this.m_index;  }
   int               Handle(void)                  const { return this.m_handle; }
//--- Set the index of the on-chart window
   void              SetIndex(const int index)           { this.m_index=index;   }

//--- Display the description of object properties in the journal (dash=true - hyphen before the description, false - description only)
   void              Print(const bool dash=false)        { ::Print((dash ? "- " : "")+this.Header());                      }
//--- Return the object short name
   string            Header(void)                  const { return CMessage::Text(MSG_CHART_OBJ_INDICATOR)+" "+this.Name(); }

//--- Compare CWndInd objects with each other by the specified property
   virtual int       Compare(const CObject *node,const int mode=0) const;

//--- Constructors
                     CWndInd(void);
                     CWndInd(const int handle,const string name,const int index) : m_handle(handle),m_name(name),m_index(index) {}
  };
//+------------------------------------------------------------------+
```

Let's slightly improve the chart object class in \\MQL5\\Include\\DoEasy\\Objects\\Chart\ **ChartObj.mqh**.

Remove the declaration of the SetVisible() method from the list of private methods for setting the object properties since I have decided to get rid of the property set by the method as it is of no help here:

```
//--- The methods of setting property values
   bool              SetMode(const string source,const ENUM_CHART_MODE mode,const bool redraw=false);
   bool              SetScale(const string source,const int scale,const bool redraw=false);
   bool              SetModeVolume(const string source,const ENUM_CHART_VOLUME_MODE mode,const bool redraw=false);
   void              SetVisibleBars(void);
   void              SetWindowsTotal(void);
   void              SetVisible(void);
   void              SetFirstVisibleBars(void);
   void              SetWidthInBars(void);
   void              SetWidthInPixels(void);
   void              SetMaximizedFlag(void);
   void              SetMinimizedFlag(void);
   void              SetExpertName(void);
   void              SetScriptName(void);
```

Beyond the class body, find its implementation and remove it as well:

```
//+-------------------------------------------------------------------+
//|  Set the property                                                 |
//| "The total number of chart windows including indicator subwindows"|
//+-------------------------------------------------------------------+
void CChartObj::SetWindowsTotal(void)
  {
   this.SetProperty(CHART_PROP_WINDOWS_TOTAL,::ChartGetInteger(this.ID(),CHART_WINDOWS_TOTAL));
  }
//+------------------------------------------------------------------+
//| Set the "Subwindow visibility" property                          |
//+------------------------------------------------------------------+
void CChartObj::SetVisible(void)
  {
   this.SetProperty(CHART_PROP_WINDOW_IS_VISIBLE,::ChartGetInteger(this.ID(),CHART_WINDOW_IS_VISIBLE,0));
  }
//+---------------------------------------------------------------------+
//| Set the property "The number of the first visible bar on the chart" |
//+---------------------------------------------------------------------+
void CChartObj::SetFirstVisibleBars(void)
  {
   this.SetProperty(CHART_PROP_FIRST_VISIBLE_BAR,::ChartGetInteger(this.ID(),CHART_FIRST_VISIBLE_BAR));
  }
//+------------------------------------------------------------------+
```

The method simply placed the visibility property of the main chart window to the already removed integer property. Since we are not going to use it, let's get rid of all related methods. In any case, the value of the window property can still be obtained directly from the environment rather than from the properties of the chart window object.

In the private section of the class, declare the method for creating the list of windows attached to the chart:

```
//--- The methods of setting property values
   bool              SetMode(const string source,const ENUM_CHART_MODE mode,const bool redraw=false);
   bool              SetScale(const string source,const int scale,const bool redraw=false);
   bool              SetModeVolume(const string source,const ENUM_CHART_VOLUME_MODE mode,const bool redraw=false);
   void              SetVisibleBars(void);
   void              SetWindowsTotal(void);
   void              SetFirstVisibleBars(void);
   void              SetWidthInBars(void);
   void              SetWidthInPixels(void);
   void              SetMaximizedFlag(void);
   void              SetMinimizedFlag(void);
   void              SetExpertName(void);
   void              SetScriptName(void);

//--- Create the list of chart windows
   void              CreateWindowsList(void);

public:
```

In the public section of the class, declare the method for updating the properties of the chart object and its subwindows:

```
//--- Update the chart object and its list of indicator windows
   void              Refresh(void);

//--- Constructors
                     CChartObj(){;}
                     CChartObj(const long chart_id);
```

While arranging the work with the method returning the total number of chart windows, I came across the fact that I have to set the data in the object properties immediately after requesting it. Therefore, I decided to combine the data request and its inclusion to the object property.

In the WindowsTotal() method, first of all, add request and setting the number of chart windows in the SetWindowsTotal() method. Next, implement the return of the newly obtained value saved in the object properties. Remove the implementation of the Visible() method:

```
//--- Return the number of bars on a chart that are available for display
   int               VisibleBars(void)                               const { return (int)this.GetProperty(CHART_PROP_VISIBLE_BARS);       }

//--- Return the total number of chart windows including indicator subwindows
   int               WindowsTotal(void)
                       {
                        this.SetWindowsTotal();
                        return (int)this.GetProperty(CHART_PROP_WINDOWS_TOTAL);
                       }

//--- Return the window visibility
   bool              Visible(void)                                   const { return (bool)this.GetProperty(CHART_PROP_WINDOW_IS_VISIBLE); }
```

Let's add yet another method returning the flag indicating that the chart object is the main one — the chart the library-based program is attached to:

```
//--- Emulate a tick (chart updates - similar to the terminal Refresh command)
   void              EmulateTick(void)                                     { ::ChartSetSymbolPeriod(this.ID(),this.Symbol(),this.Timeframe());}

//--- Return the flag indicating that the chart object belongs to the program chart
   bool              IsMainChart(void)                               const { return(this.m_chart_id==CBaseObj::GetMainChartID());            }
//--- Return the chart window specified by index
   CChartWnd        *GetWindowByIndex(const int index)               const { return this.m_list_wnd.At(index);                               }
```

The **CBaseObj** base object of all library objects features the **m\_chart\_id\_main** variable storing an ID of the chart the program is launched at. In the constructor, the variable receives the value returned by the [ChartID()](https://www.mql5.com/en/docs/chart_operations/chartid) function. The current chart ID is returned using the GetMainChartID() method of the CBaseObj class, which returns the value set in the **m\_chart\_id\_main** variable. Thus, we return the flag indicating that the current chart IDmatches the program main window ID. If there is a match, the method returns true, otherwise — false.

From the parametric constructor, remove the string, in which the current chart windows visibility value is set:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CChartObj::CChartObj(const long chart_id)
  {
//--- Set chart ID to the base object
   CBaseObj::SetChartID(chart_id);
//--- Set integer properties
   this.SetProperty(CHART_PROP_ID,chart_id);                                                             // Chart ID
   this.SetProperty(CHART_PROP_TIMEFRAME,::ChartPeriod(this.ID()));                                      // Chart timeframe
   this.SetProperty(CHART_PROP_SHOW,::ChartGetInteger(this.ID(),CHART_SHOW));                            // Price chart drawing attribute
   this.SetProperty(CHART_PROP_IS_OBJECT,::ChartGetInteger(this.ID(),CHART_IS_OBJECT));                  // Chart object identification attribute
   this.SetProperty(CHART_PROP_BRING_TO_TOP,false);                                                      // Show chart above all others
   this.SetProperty(CHART_PROP_CONTEXT_MENU,::ChartGetInteger(this.ID(),CHART_CONTEXT_MENU));            // Access to the context menu using the right click
   this.SetProperty(CHART_PROP_CROSSHAIR_TOOL,::ChartGetInteger(this.ID(),CHART_CROSSHAIR_TOOL));        // Access the Crosshair tool by pressing the middle mouse button
   this.SetProperty(CHART_PROP_MOUSE_SCROLL,::ChartGetInteger(this.ID(),CHART_MOUSE_SCROLL));            // Scroll the chart horizontally using the left mouse button
   this.SetProperty(CHART_PROP_EVENT_MOUSE_WHEEL,::ChartGetInteger(this.ID(),CHART_EVENT_MOUSE_WHEEL));  // Send messages about mouse wheel events to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_MOUSE_MOVE,::ChartGetInteger(this.ID(),CHART_EVENT_MOUSE_MOVE));    // Send messages about mouse button click and movement events to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_OBJECT_CREATE,::ChartGetInteger(this.ID(),CHART_EVENT_OBJECT_CREATE)); // Send messages about the graphical object creation event to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_OBJECT_DELETE,::ChartGetInteger(this.ID(),CHART_EVENT_OBJECT_DELETE)); // Send messages about the graphical object destruction event to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_MODE,::ChartGetInteger(this.ID(),CHART_MODE));                            // Type of the chart (candlesticks, bars or line)
   this.SetProperty(CHART_PROP_FOREGROUND,::ChartGetInteger(this.ID(),CHART_FOREGROUND));                // Price chart in the foreground
   this.SetProperty(CHART_PROP_SHIFT,::ChartGetInteger(this.ID(),CHART_SHIFT));                          // Mode of shift of the price chart from the right border
   this.SetProperty(CHART_PROP_AUTOSCROLL,::ChartGetInteger(this.ID(),CHART_AUTOSCROLL));                // The mode of automatic shift to the right border of the chart
   this.SetProperty(CHART_PROP_KEYBOARD_CONTROL,::ChartGetInteger(this.ID(),CHART_KEYBOARD_CONTROL));    // Allow managing the chart using a keyboard
   this.SetProperty(CHART_PROP_QUICK_NAVIGATION,::ChartGetInteger(this.ID(),CHART_QUICK_NAVIGATION));    // Allow the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   this.SetProperty(CHART_PROP_SCALE,::ChartGetInteger(this.ID(),CHART_SCALE));                          // Scale
   this.SetProperty(CHART_PROP_SCALEFIX,::ChartGetInteger(this.ID(),CHART_SCALEFIX));                    // Fixed scale mode
   this.SetProperty(CHART_PROP_SCALEFIX_11,::ChartGetInteger(this.ID(),CHART_SCALEFIX_11));              // 1:1 scale mode
   this.SetProperty(CHART_PROP_SCALE_PT_PER_BAR,::ChartGetInteger(this.ID(),CHART_SCALE_PT_PER_BAR));    // Mode for specifying the scale in points per bar
   this.SetProperty(CHART_PROP_SHOW_TICKER,::ChartGetInteger(this.ID(),CHART_SHOW_TICKER));              // Display a symbol ticker in the upper left corner
   this.SetProperty(CHART_PROP_SHOW_OHLC,::ChartGetInteger(this.ID(),CHART_SHOW_OHLC));                  // Display OHLC values in the upper left corner
   this.SetProperty(CHART_PROP_SHOW_BID_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_BID_LINE));          // Display Bid value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_ASK_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_ASK_LINE));          // Display Ask value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_LAST_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_LAST_LINE));        // Display Last value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_PERIOD_SEP,::ChartGetInteger(this.ID(),CHART_SHOW_PERIOD_SEP));      // Display vertical separators between adjacent periods
   this.SetProperty(CHART_PROP_SHOW_GRID,::ChartGetInteger(this.ID(),CHART_SHOW_GRID));                  // Display the chart grid
   this.SetProperty(CHART_PROP_SHOW_VOLUMES,::ChartGetInteger(this.ID(),CHART_SHOW_VOLUMES));            // Display volumes on the chart
   this.SetProperty(CHART_PROP_SHOW_OBJECT_DESCR,::ChartGetInteger(this.ID(),CHART_SHOW_OBJECT_DESCR));  // Display text descriptions of the objects
   this.SetProperty(CHART_PROP_VISIBLE_BARS,::ChartGetInteger(this.ID(),CHART_VISIBLE_BARS));            // Number of bars on a chart that are available for display
   this.SetProperty(CHART_PROP_WINDOWS_TOTAL,::ChartGetInteger(this.ID(),CHART_WINDOWS_TOTAL));          // The total number of chart windows including indicator subwindows
   this.SetProperty(CHART_PROP_WINDOW_IS_VISIBLE,::ChartGetInteger(this.ID(),CHART_WINDOW_IS_VISIBLE,0));// Window visibility
   this.SetProperty(CHART_PROP_WINDOW_HANDLE,::ChartGetInteger(this.ID(),CHART_WINDOW_HANDLE));          // Chart window handle
   this.SetProperty(CHART_PROP_WINDOW_YDISTANCE,::ChartGetInteger(this.ID(),CHART_WINDOW_YDISTANCE,0));  // Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   this.SetProperty(CHART_PROP_FIRST_VISIBLE_BAR,::ChartGetInteger(this.ID(),CHART_FIRST_VISIBLE_BAR));  // Number of the first visible bar on the chart
   this.SetProperty(CHART_PROP_WIDTH_IN_BARS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_BARS));          // Chart width in bars
   this.SetProperty(CHART_PROP_WIDTH_IN_PIXELS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_PIXELS));      // Chart width in pixels
   this.SetProperty(CHART_PROP_HEIGHT_IN_PIXELS,::ChartGetInteger(this.ID(),CHART_HEIGHT_IN_PIXELS,0));  // Chart height in pixels
   this.SetProperty(CHART_PROP_COLOR_BACKGROUND,::ChartGetInteger(this.ID(),CHART_COLOR_BACKGROUND));    // Chart background color
   this.SetProperty(CHART_PROP_COLOR_FOREGROUND,::ChartGetInteger(this.ID(),CHART_COLOR_FOREGROUND));    // Color of axes, scale and OHLC line
   this.SetProperty(CHART_PROP_COLOR_GRID,::ChartGetInteger(this.ID(),CHART_COLOR_GRID));                // Grid color
   this.SetProperty(CHART_PROP_COLOR_VOLUME,::ChartGetInteger(this.ID(),CHART_COLOR_VOLUME));            // Color of volumes and position opening levels
   this.SetProperty(CHART_PROP_COLOR_CHART_UP,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_UP));        // Color for the up bar, shadows and body borders of bullish candlesticks
   this.SetProperty(CHART_PROP_COLOR_CHART_DOWN,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_DOWN));    // Color for the down bar, shadows and body borders of bearish candlesticks
   this.SetProperty(CHART_PROP_COLOR_CHART_LINE,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_LINE));    // Color of the chart line and the Doji candlesticks
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BULL,::ChartGetInteger(this.ID(),CHART_COLOR_CANDLE_BULL));  // Color of the bullish candle body
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BEAR,::ChartGetInteger(this.ID(),CHART_COLOR_CANDLE_BEAR));  // Color of the bearish candle body
   this.SetProperty(CHART_PROP_COLOR_BID,::ChartGetInteger(this.ID(),CHART_COLOR_BID));                  // Bid price line color
   this.SetProperty(CHART_PROP_COLOR_ASK,::ChartGetInteger(this.ID(),CHART_COLOR_ASK));                  // Ask price line color
   this.SetProperty(CHART_PROP_COLOR_LAST,::ChartGetInteger(this.ID(),CHART_COLOR_LAST));                // Color of the last performed deal's price line (Last)
   this.SetProperty(CHART_PROP_COLOR_STOP_LEVEL,::ChartGetInteger(this.ID(),CHART_COLOR_STOP_LEVEL));    // Color of stop order levels (Stop Loss and Take Profit)
   this.SetProperty(CHART_PROP_SHOW_TRADE_LEVELS,::ChartGetInteger(this.ID(),CHART_SHOW_TRADE_LEVELS));  // Display trade levels on the chart (levels of open positions, Stop Loss, Take Profit and pending orders)
   this.SetProperty(CHART_PROP_DRAG_TRADE_LEVELS,::ChartGetInteger(this.ID(),CHART_DRAG_TRADE_LEVELS));  // Enable the ability to drag trading levels on a chart using mouse
   this.SetProperty(CHART_PROP_SHOW_DATE_SCALE,::ChartGetInteger(this.ID(),CHART_SHOW_DATE_SCALE));      // Display the time scale on the chart
   this.SetProperty(CHART_PROP_SHOW_PRICE_SCALE,::ChartGetInteger(this.ID(),CHART_SHOW_PRICE_SCALE));    // Display the price scale on the chart
   this.SetProperty(CHART_PROP_SHOW_ONE_CLICK,::ChartGetInteger(this.ID(),CHART_SHOW_ONE_CLICK));        // Display the quick trading panel on the chart
   this.SetProperty(CHART_PROP_IS_MAXIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MAXIMIZED));            // Chart window maximized
   this.SetProperty(CHART_PROP_IS_MINIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MINIMIZED));            // Chart window minimized
   this.SetProperty(CHART_PROP_IS_DOCKED,::ChartGetInteger(this.ID(),CHART_IS_DOCKED));                  // Chart window docked
   this.SetProperty(CHART_PROP_FLOAT_LEFT,::ChartGetInteger(this.ID(),CHART_FLOAT_LEFT));                // Left coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_TOP,::ChartGetInteger(this.ID(),CHART_FLOAT_TOP));                  // Upper coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_RIGHT,::ChartGetInteger(this.ID(),CHART_FLOAT_RIGHT));              // Right coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_BOTTOM,::ChartGetInteger(this.ID(),CHART_FLOAT_BOTTOM));            // Bottom coordinate of the undocked chart window relative to the virtual screen
//--- Set real properties
   this.SetProperty(CHART_PROP_SHIFT_SIZE,::ChartGetDouble(this.ID(),CHART_SHIFT_SIZE));                 // Shift size of the zero bar from the right border in %
   this.SetProperty(CHART_PROP_FIXED_POSITION,::ChartGetDouble(this.ID(),CHART_FIXED_POSITION));         // Chart fixed position from the left border in %
   this.SetProperty(CHART_PROP_FIXED_MAX,::ChartGetDouble(this.ID(),CHART_FIXED_MAX));                   // Fixed chart maximum
   this.SetProperty(CHART_PROP_FIXED_MIN,::ChartGetDouble(this.ID(),CHART_FIXED_MIN));                   // Fixed chart minimum
   this.SetProperty(CHART_PROP_POINTS_PER_BAR,::ChartGetDouble(this.ID(),CHART_POINTS_PER_BAR));         // Scale in points per bar
   this.SetProperty(CHART_PROP_PRICE_MIN,::ChartGetDouble(this.ID(),CHART_PRICE_MIN));                   // Chart minimum
   this.SetProperty(CHART_PROP_PRICE_MAX,::ChartGetDouble(this.ID(),CHART_PRICE_MAX));                   // Chart maximum
//--- Set string properties
   this.SetProperty(CHART_PROP_COMMENT,::ChartGetString(this.ID(),CHART_COMMENT));                       // Comment text on the chart
   this.SetProperty(CHART_PROP_EXPERT_NAME,::ChartGetString(this.ID(),CHART_EXPERT_NAME));               // name of an EA launched on the chart
   this.SetProperty(CHART_PROP_SCRIPT_NAME,::ChartGetString(this.ID(),CHART_SCRIPT_NAME));               // name of a script launched on the chart
   this.SetProperty(CHART_PROP_SYMBOL,::ChartSymbol(this.ID()));                                         // Chart symbol

   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
```

Instead of the loop featuring the window objects belonging to the chart,

```
   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
   int total=this.WindowsTotal();
   for(int i=0;i<total;i++)
     {
      CChartWnd *wnd=new CChartWnd(m_chart_id,i);
      if(wnd==NULL)
         continue;
      m_list_wnd.Sort();
      if(!m_list_wnd.Add(wnd))
         delete wnd;
     }
```

add calling the method, in which the list of windows belonging to the chart is created:

```
   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
   this.CreateWindowsList();
```

As a result, the parametric constructor now looks like this:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CChartObj::CChartObj(const long chart_id)
  {
//--- Set chart ID to the base object
   CBaseObj::SetChartID(chart_id);
//--- Set integer properties
   this.SetProperty(CHART_PROP_ID,chart_id);                                                             // Chart ID
   this.SetProperty(CHART_PROP_TIMEFRAME,::ChartPeriod(this.ID()));                                      // Chart timeframe
   this.SetProperty(CHART_PROP_SHOW,::ChartGetInteger(this.ID(),CHART_SHOW));                            // Price chart drawing attribute
   this.SetProperty(CHART_PROP_IS_OBJECT,::ChartGetInteger(this.ID(),CHART_IS_OBJECT));                  // Chart object identification attribute
   this.SetProperty(CHART_PROP_BRING_TO_TOP,false);                                                      // Show chart above all others
   this.SetProperty(CHART_PROP_CONTEXT_MENU,::ChartGetInteger(this.ID(),CHART_CONTEXT_MENU));            // Access to the context menu using the right click
   this.SetProperty(CHART_PROP_CROSSHAIR_TOOL,::ChartGetInteger(this.ID(),CHART_CROSSHAIR_TOOL));        // Access the Crosshair tool by pressing the middle mouse button
   this.SetProperty(CHART_PROP_MOUSE_SCROLL,::ChartGetInteger(this.ID(),CHART_MOUSE_SCROLL));            // Scroll the chart horizontally using the left mouse button
   this.SetProperty(CHART_PROP_EVENT_MOUSE_WHEEL,::ChartGetInteger(this.ID(),CHART_EVENT_MOUSE_WHEEL));  // Send messages about mouse wheel events to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_MOUSE_MOVE,::ChartGetInteger(this.ID(),CHART_EVENT_MOUSE_MOVE));    // Send messages about mouse button click and movement events to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_OBJECT_CREATE,::ChartGetInteger(this.ID(),CHART_EVENT_OBJECT_CREATE)); // Send messages about the graphical object creation event to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_EVENT_OBJECT_DELETE,::ChartGetInteger(this.ID(),CHART_EVENT_OBJECT_DELETE)); // Send messages about the graphical object destruction event to all MQL5 programs on a chart
   this.SetProperty(CHART_PROP_MODE,::ChartGetInteger(this.ID(),CHART_MODE));                            // Type of the chart (candlesticks, bars or line)
   this.SetProperty(CHART_PROP_FOREGROUND,::ChartGetInteger(this.ID(),CHART_FOREGROUND));                // Price chart in the foreground
   this.SetProperty(CHART_PROP_SHIFT,::ChartGetInteger(this.ID(),CHART_SHIFT));                          // Mode of shift of the price chart from the right border
   this.SetProperty(CHART_PROP_AUTOSCROLL,::ChartGetInteger(this.ID(),CHART_AUTOSCROLL));                // The mode of automatic shift to the right border of the chart
   this.SetProperty(CHART_PROP_KEYBOARD_CONTROL,::ChartGetInteger(this.ID(),CHART_KEYBOARD_CONTROL));    // Allow managing the chart using a keyboard
   this.SetProperty(CHART_PROP_QUICK_NAVIGATION,::ChartGetInteger(this.ID(),CHART_QUICK_NAVIGATION));    // Allow the chart to intercept Space and Enter key strokes to activate the quick navigation bar
   this.SetProperty(CHART_PROP_SCALE,::ChartGetInteger(this.ID(),CHART_SCALE));                          // Scale
   this.SetProperty(CHART_PROP_SCALEFIX,::ChartGetInteger(this.ID(),CHART_SCALEFIX));                    // Fixed scale mode
   this.SetProperty(CHART_PROP_SCALEFIX_11,::ChartGetInteger(this.ID(),CHART_SCALEFIX_11));              // 1:1 scale mode
   this.SetProperty(CHART_PROP_SCALE_PT_PER_BAR,::ChartGetInteger(this.ID(),CHART_SCALE_PT_PER_BAR));    // Mode for specifying the scale in points per bar
   this.SetProperty(CHART_PROP_SHOW_TICKER,::ChartGetInteger(this.ID(),CHART_SHOW_TICKER));              // Display a symbol ticker in the upper left corner
   this.SetProperty(CHART_PROP_SHOW_OHLC,::ChartGetInteger(this.ID(),CHART_SHOW_OHLC));                  // Display OHLC values in the upper left corner
   this.SetProperty(CHART_PROP_SHOW_BID_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_BID_LINE));          // Display Bid value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_ASK_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_ASK_LINE));          // Display Ask value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_LAST_LINE,::ChartGetInteger(this.ID(),CHART_SHOW_LAST_LINE));        // Display Last value as a horizontal line on the chart
   this.SetProperty(CHART_PROP_SHOW_PERIOD_SEP,::ChartGetInteger(this.ID(),CHART_SHOW_PERIOD_SEP));      // Display vertical separators between adjacent periods
   this.SetProperty(CHART_PROP_SHOW_GRID,::ChartGetInteger(this.ID(),CHART_SHOW_GRID));                  // Display the chart grid
   this.SetProperty(CHART_PROP_SHOW_VOLUMES,::ChartGetInteger(this.ID(),CHART_SHOW_VOLUMES));            // Display volumes on the chart
   this.SetProperty(CHART_PROP_SHOW_OBJECT_DESCR,::ChartGetInteger(this.ID(),CHART_SHOW_OBJECT_DESCR));  // Display text descriptions of the objects
   this.SetProperty(CHART_PROP_VISIBLE_BARS,::ChartGetInteger(this.ID(),CHART_VISIBLE_BARS));            // Number of bars on a chart that are available for display
   this.SetProperty(CHART_PROP_WINDOWS_TOTAL,::ChartGetInteger(this.ID(),CHART_WINDOWS_TOTAL));          // The total number of chart windows including indicator subwindows
   this.SetProperty(CHART_PROP_WINDOW_HANDLE,::ChartGetInteger(this.ID(),CHART_WINDOW_HANDLE));          // Chart window handle
   this.SetProperty(CHART_PROP_WINDOW_YDISTANCE,::ChartGetInteger(this.ID(),CHART_WINDOW_YDISTANCE,0));  // Distance in Y axis pixels between the upper frame of the indicator subwindow and the upper frame of the chart main window
   this.SetProperty(CHART_PROP_FIRST_VISIBLE_BAR,::ChartGetInteger(this.ID(),CHART_FIRST_VISIBLE_BAR));  // Number of the first visible bar on the chart
   this.SetProperty(CHART_PROP_WIDTH_IN_BARS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_BARS));          // Chart width in bars
   this.SetProperty(CHART_PROP_WIDTH_IN_PIXELS,::ChartGetInteger(this.ID(),CHART_WIDTH_IN_PIXELS));      // Chart width in pixels
   this.SetProperty(CHART_PROP_HEIGHT_IN_PIXELS,::ChartGetInteger(this.ID(),CHART_HEIGHT_IN_PIXELS,0));  // Chart height in pixels
   this.SetProperty(CHART_PROP_COLOR_BACKGROUND,::ChartGetInteger(this.ID(),CHART_COLOR_BACKGROUND));    // Chart background color
   this.SetProperty(CHART_PROP_COLOR_FOREGROUND,::ChartGetInteger(this.ID(),CHART_COLOR_FOREGROUND));    // Color of axes, scale and OHLC line
   this.SetProperty(CHART_PROP_COLOR_GRID,::ChartGetInteger(this.ID(),CHART_COLOR_GRID));                // Grid color
   this.SetProperty(CHART_PROP_COLOR_VOLUME,::ChartGetInteger(this.ID(),CHART_COLOR_VOLUME));            // Color of volumes and position opening levels
   this.SetProperty(CHART_PROP_COLOR_CHART_UP,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_UP));        // Color for the up bar, shadows and body borders of bullish candlesticks
   this.SetProperty(CHART_PROP_COLOR_CHART_DOWN,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_DOWN));    // Color for the down bar, shadows and body borders of bearish candlesticks
   this.SetProperty(CHART_PROP_COLOR_CHART_LINE,::ChartGetInteger(this.ID(),CHART_COLOR_CHART_LINE));    // Color of the chart line and the Doji candlesticks
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BULL,::ChartGetInteger(this.ID(),CHART_COLOR_CANDLE_BULL));  // Color of the bullish candle body
   this.SetProperty(CHART_PROP_COLOR_CANDLE_BEAR,::ChartGetInteger(this.ID(),CHART_COLOR_CANDLE_BEAR));  // Color of the bearish candle body
   this.SetProperty(CHART_PROP_COLOR_BID,::ChartGetInteger(this.ID(),CHART_COLOR_BID));                  // Bid price line color
   this.SetProperty(CHART_PROP_COLOR_ASK,::ChartGetInteger(this.ID(),CHART_COLOR_ASK));                  // Ask price line color
   this.SetProperty(CHART_PROP_COLOR_LAST,::ChartGetInteger(this.ID(),CHART_COLOR_LAST));                // Color of the last performed deal's price line (Last)
   this.SetProperty(CHART_PROP_COLOR_STOP_LEVEL,::ChartGetInteger(this.ID(),CHART_COLOR_STOP_LEVEL));    // Color of stop order levels (Stop Loss and Take Profit)
   this.SetProperty(CHART_PROP_SHOW_TRADE_LEVELS,::ChartGetInteger(this.ID(),CHART_SHOW_TRADE_LEVELS));  // Display trade levels on the chart (levels of open positions, Stop Loss, Take Profit and pending orders)
   this.SetProperty(CHART_PROP_DRAG_TRADE_LEVELS,::ChartGetInteger(this.ID(),CHART_DRAG_TRADE_LEVELS));  // Enable the ability to drag trading levels on a chart using mouse
   this.SetProperty(CHART_PROP_SHOW_DATE_SCALE,::ChartGetInteger(this.ID(),CHART_SHOW_DATE_SCALE));      // Display the time scale on the chart
   this.SetProperty(CHART_PROP_SHOW_PRICE_SCALE,::ChartGetInteger(this.ID(),CHART_SHOW_PRICE_SCALE));    // Display the price scale on the chart
   this.SetProperty(CHART_PROP_SHOW_ONE_CLICK,::ChartGetInteger(this.ID(),CHART_SHOW_ONE_CLICK));        // Display the quick trading panel on the chart
   this.SetProperty(CHART_PROP_IS_MAXIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MAXIMIZED));            // Chart window maximized
   this.SetProperty(CHART_PROP_IS_MINIMIZED,::ChartGetInteger(this.ID(),CHART_IS_MINIMIZED));            // Chart window minimized
   this.SetProperty(CHART_PROP_IS_DOCKED,::ChartGetInteger(this.ID(),CHART_IS_DOCKED));                  // Chart window docked
   this.SetProperty(CHART_PROP_FLOAT_LEFT,::ChartGetInteger(this.ID(),CHART_FLOAT_LEFT));                // Left coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_TOP,::ChartGetInteger(this.ID(),CHART_FLOAT_TOP));                  // Upper coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_RIGHT,::ChartGetInteger(this.ID(),CHART_FLOAT_RIGHT));              // Right coordinate of the undocked chart window relative to the virtual screen
   this.SetProperty(CHART_PROP_FLOAT_BOTTOM,::ChartGetInteger(this.ID(),CHART_FLOAT_BOTTOM));            // Bottom coordinate of the undocked chart window relative to the virtual screen
//--- Set real properties
   this.SetProperty(CHART_PROP_SHIFT_SIZE,::ChartGetDouble(this.ID(),CHART_SHIFT_SIZE));                 // Shift size of the zero bar from the right border in %
   this.SetProperty(CHART_PROP_FIXED_POSITION,::ChartGetDouble(this.ID(),CHART_FIXED_POSITION));         // Chart fixed position from the left border in %
   this.SetProperty(CHART_PROP_FIXED_MAX,::ChartGetDouble(this.ID(),CHART_FIXED_MAX));                   // Fixed chart maximum
   this.SetProperty(CHART_PROP_FIXED_MIN,::ChartGetDouble(this.ID(),CHART_FIXED_MIN));                   // Fixed chart minimum
   this.SetProperty(CHART_PROP_POINTS_PER_BAR,::ChartGetDouble(this.ID(),CHART_POINTS_PER_BAR));         // Scale in points per bar
   this.SetProperty(CHART_PROP_PRICE_MIN,::ChartGetDouble(this.ID(),CHART_PRICE_MIN));                   // Chart minimum
   this.SetProperty(CHART_PROP_PRICE_MAX,::ChartGetDouble(this.ID(),CHART_PRICE_MAX));                   // Chart maximum
//--- Set string properties
   this.SetProperty(CHART_PROP_COMMENT,::ChartGetString(this.ID(),CHART_COMMENT));                       // Comment text on the chart
   this.SetProperty(CHART_PROP_EXPERT_NAME,::ChartGetString(this.ID(),CHART_EXPERT_NAME));               // name of an EA launched on the chart
   this.SetProperty(CHART_PROP_SCRIPT_NAME,::ChartGetString(this.ID(),CHART_SCRIPT_NAME));               // name of a script launched on the chart
   this.SetProperty(CHART_PROP_SYMBOL,::ChartSymbol(this.ID()));                                         // Chart symbol

   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
   this.CreateWindowsList();
  }
//+------------------------------------------------------------------+
```

From the GetPropertyDescription() method returning the description of an object integer property, remove the code block, in which the window visibility parameter description is created since I have removed this property from the object:

```
      property==CHART_PROP_WINDOWS_TOTAL  ?  CMessage::Text(MSG_CHART_OBJ_WINDOWS_TOTAL)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==CHART_PROP_WINDOW_IS_VISIBLE  ?  CMessage::Text(MSG_CHART_OBJ_WINDOW_IS_VISIBLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==CHART_PROP_WINDOW_HANDLE  ?  CMessage::Text(MSG_CHART_OBJ_WINDOW_HANDLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
```

In the methods, displaying data of all indicators of all chart windows and the properties of all chart windows in the journal, replace the loop size with the number of chart window objects in the lists:

```
//+-------------------------------------------------------------------+
//| Display data of all indicators of all chart windows in the journal|
//+-------------------------------------------------------------------+
void CChartObj::PrintWndIndicators(void)
  {
   for(int i=0;i<this.m_list_wnd.Total();i++)
     {
      CChartWnd *wnd=m_list_wnd.At(i);
      if(wnd==NULL)
         continue;
      wnd.PrintIndicators(true);
     }
  }
//+------------------------------------------------------------------+
//| Display the properties of all chart windows in the journal       |
//+------------------------------------------------------------------+
void CChartObj::PrintWndParameters(void)
  {
   for(int i=0;i<this.m_list_wnd.Total();i++)
     {
      CChartWnd *wnd=m_list_wnd.At(i);
      if(wnd==NULL)
         continue;
      wnd.PrintParameters(true);
     }
  }
//+------------------------------------------------------------------+
```

Previously, the loops were counted till the value returned by the WindowsTotal() method, which is not entirely correct. The method returns the actual value obtained from the chart, while the lists may contain either more or fewer objects compared to their actual number. This happens when changing their number in the terminal chart. Therefore, the loop based on the incorrect number of objects in the list is fraught with errors. This is exactly what I have fixed here.

**Let's implement the method updating the hart object and the list of its windows:**

```
//+------------------------------------------------------------------+
//| Update the chart object and its window list                      |
//+------------------------------------------------------------------+
void CChartObj::Refresh(void)
  {
   int change=this.WindowsTotal()-this.m_list_wnd.Total();
   if(change==0)
      return;
   this.CreateWindowsList();
  }
//+------------------------------------------------------------------+
```

Here we count the difference between the real number of chart windows and the number of window objects in the list. If there is no difference, there are no changes — leave the method, otherwise create anew the full list of all chart window objects belonging to the chart object. It is much easier than searching for differences between objects or missing chart window objects in order to remove unnecessary ones or add new ones to the list.

**Let's implement the method creating the list of chart window objects belonging to the chart:**

```
//+------------------------------------------------------------------+
//| Create the list of chart windows                                 |
//+------------------------------------------------------------------+
void CChartObj::CreateWindowsList(void)
  {
   this.m_list_wnd.Clear();
   int total=this.WindowsTotal();
   for(int i=0;i<total;i++)
     {
      CChartWnd *wnd=new CChartWnd(m_chart_id,i);
      if(wnd==NULL)
         continue;
      this.m_list_wnd.Sort();
      if(!m_list_wnd.Add(wnd))
         delete wnd;
     }
  }
//+------------------------------------------------------------------+
```

Here we clear the list of window objects, get the total number of chart windows from its terminal parameters and, in the loop by the obtained number of windows, create a new chart window object and add it to the list. If failed to add the object to the list, remove it to avoid memory leaks.

**This concludes the improvement of the library classes.**

Let's start creating the chart object collection class.

### Chart object collection class

Similar to the chart object where we check the changes in the number of chart windows to activate the update of their number in the object, in the chart object collection, we are going to check the changes in the number of open charts to launch re-building the chart object collection list.

First, we will check the number of windows of the existing chart objects. Next, we will check the change in the number of open charts. In case of a non-zero check result, we will launch re-building of the chart object collection.

In the \\MQL5\\Include\\DoEasy\\Collections\ **ChartObjCollection.mqh** library folder, create the new file of the **CChartObjCollection** class. The class should be derived from the class of the base object of all CBaseObj library objects. The chart object file should be included to the file of the chart object collection class:

```
//+------------------------------------------------------------------+
//|                                           ChartObjCollection.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Chart\ChartObj.mqh"
//+------------------------------------------------------------------+
//| MQL5 signal object collection                                    |
//+------------------------------------------------------------------+
class CChartObjCollection : public CBaseObj
  {
  }
```

In the private section of the class, declare the CListObj list storing chart objects, the variable for stroing the previous number of open charts in the terminal and auxiliary methods for arranging the class operation:

```
//+------------------------------------------------------------------+
//| MQL5 signal object collection                                    |
//+------------------------------------------------------------------+
class CChartObjCollection : public CBaseObj
  {
private:
   CListObj                m_list;                                   // List of chart objects
   int                     m_charts_total_prev;                      // Previous number of charts in the terminal
   //--- Return the number of charts in the terminal
   int                     ChartsTotal(void) const;
   //--- Return the flag indicating the existence of (1) a chart object and (2) a chart
   bool                    IsPresentChartObj(const long chart_id);
   bool                    IsPresentChart(const long chart_id);
   //--- Create a new chart object and add it to the list
   bool                    CreateNewChartObj(const long chart_id,const string source);
   //--- Find the missing chart object, create it and add it to the collection list
   bool                    FindAndCreateMissingChartObj(void);
   //--- Find a chart object not present in the terminal and remove it from the list
   void                    FindAndDeleteExcessChartObj(void);
public:
  }
```

Set the standard methods in the public section of the class:

```
public:
//--- Return (1) itself and (2) chart object collection list
   CChartObjCollection    *GetObject(void)                                 { return &this;                  }
   CArrayObj              *GetList(void)                                   { return &this.m_list;           }
   //--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj              *GetList(ENUM_CHART_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByChartProperty(this.GetList(),property,value,mode);  }
   CArrayObj              *GetList(ENUM_CHART_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByChartProperty(this.GetList(),property,value,mode);  }
   CArrayObj              *GetList(ENUM_CHART_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByChartProperty(this.GetList(),property,value,mode);  }
//--- Return the number of chart objects in the list
   int                     DataTotal(void)                           const { return this.m_list.Total();    }
//--- Display (1) the complete and (2) short collection description in the journal
   void                    Print(void);
   void                    PrintShort(void);
//--- Constructor
                           CChartObjCollection();
```

and add additional methods:

```
//--- Return the pointer to the chart object (1) by ID and (2) by an index in the list
   CChartObj              *GetChart(const long id);
   CChartObj              *GetChart(const int index)                       { return this.m_list.At(index);  }
//--- Return the list of chart objects by (1) symbol and (2) timeframe
   CArrayObj              *GetChartsList(const string symbol)              { return this.GetList(CHART_PROP_SYMBOL,symbol,EQUAL);         }
   CArrayObj              *GetChartsList(const ENUM_TIMEFRAMES timeframe)  { return this.GetList(CHART_PROP_TIMEFRAME,timeframe,EQUAL);   }
//--- Return the chart ID with the program
   long                    GetMainChartID(void)                      const { return CBaseObj::GetMainChartID();   }
//--- Create the collection list of chart objects
   bool                    CreateCollection(void);
//--- Update (1) the chart object collection list and (2) the specified chart object
   void                    Refresh(void);
   void                    Refresh(const long chart_id);

  };
//+------------------------------------------------------------------+
```

Let's have a look at the implementation of some class methods.

**In the class constructor**, clear the list of collection objects, set the sorted list flag for the list, assign the chart object collection ID to the list and save the number of the current open charts in the terminal:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CChartObjCollection::CChartObjCollection()
  {
   this.m_list.Clear();
   this.m_list.Sort();
   this.m_list.Type(COLLECTION_CHARTS_ID);
   this.m_charts_total_prev=this.ChartsTotal();
  }
//+------------------------------------------------------------------+
```

MQL features no function returning the number of open charts. Besides, we cannot go through the ready-made array of open charts in a loop getting each subsequent chart by the loop index. However, there is a function returning the ID of the very first chart [ChartFirst()](https://www.mql5.com/en/docs/chart_operations/chartfirst) and the function returning the ID of the next chart after the specified [ChartNext()](https://www.mql5.com/en/docs/chart_operations/chartnext). Thus, we are able to create a loop by all open charts getting each subsequent chart one by one based on the ID of the previous one. The info on ChartNext() in the help contains an example of arranging such a loop:

```
//--- variables for chart identifiers
   long currChart,prevChart=ChartFirst();
   int i=0,limit=100;
   Print("ChartFirst = ",ChartSymbol(prevChart)," ID = ",prevChart);
   while(i<limit)// we probably have no more than 100 open charts
     {
      currChart=ChartNext(prevChart); // get new chart on the basis of the previous one
      if(currChart<0) break;          // the end of chart list is reached
      Print(i,ChartSymbol(currChart)," ID =",currChart);
      prevChart=currChart;// memorize identifier of the current chart for ChartNext()
      i++;// do not forget to increase the counter
     }
```

Based on such a loop, I will create custom methods that require working with open charts of the client terminal.

**The method creating the collection list of chart objects:**

```
//+------------------------------------------------------------------+
//| Create the collection list of chart objects                      |
//+------------------------------------------------------------------+
bool CChartObjCollection::CreateCollection(void)
  {
   //--- Clear the list and set the flag of sorting by the chart ID
   m_list.Clear();
   m_list.Sort(SORT_BY_CHART_ID);
   //--- Declare the variables and get the first chart ID
   long curr_chart,prev_chart=::ChartFirst();
   int i=0;
   //--- Create the first chart object and add it to the list
   if(!this.CreateNewChartObj(prev_chart,DFUN))
      return false;
   //--- In the loop by the total number of terminal charts (not more than 100)
   while(i<CHARTS_MAX)
     {
      //--- based on the previous one, get the new chart
      curr_chart=::ChartNext(prev_chart);
      //--- When reaching the end of the chart list, complete the loop
      if(curr_chart<0) break;
      //--- Create the chart object based on the current chart ID in the loop and add it to the list
      if(!this.CreateNewChartObj(curr_chart,DFUN))
         return false;
      //--- remember the current chart ID for ChartNext() and increase the loop counter
      prev_chart=curr_chart;
      i++;
     }
   //--- Filled in the list successfully
   return true;
  }
//+------------------------------------------------------------------+
```

The entire method logic here is described in the comments to the code. In short, first we clear the collection list removing previously added objects. Then in the above loop, get each subsequent open chart and use its ID to create a new chart object and add it to the collection list.

**The method updating the chart object collection list:**

```
//+------------------------------------------------------------------+
//| Update the collection list of chart objects                      |
//+------------------------------------------------------------------+
void CChartObjCollection::Refresh(void)
  {
   //--- Get the number of open charts in the terminal and
   int charts_total=this.ChartsTotal();
   //--- calculate the difference between the number of open charts in the terminal
   //--- and chart objects in the collection list. These values are displayed in the chart comment
   int change=charts_total-this.m_list.Total();
   Comment(DFUN,", list total=",DataTotal(),", charts total=",charts_total,", change=",change);
   //--- If there are no changes, leave
   if(change==0)
      return;
   //--- If a chart is added in the terminal
   if(change>0)
     {
      //--- Find the missing chart object, create and add it to the collection list
      this.FindAndCreateMissingChartObj();
      //--- Get the current chart and return to it since
      //--- adding a new chart switches the focus to it
      CChartObj *chart=this.GetChart(GetMainChartID());
      if(chart!=NULL)
         chart.SetBringToTopON(true);
     }
   //--- If a chart is removed from the terminal
   else if(change<0)
    {
     //--- Find an extra chart object in the collection list and remove it from the list
     this.FindAndDeleteExcessChartObj();
    }
   //--- In the loop by the number of chart objects in the list,
   for(int i=0;i<this.m_list.Total();i++)
     {
      //--- get the next chart object and
      CChartObj *chart=this.m_list.At(i);
      if(chart==NULL)
         continue;
      //--- update it
      chart.Refresh();
     }
  }
//+------------------------------------------------------------------+
```

The entire method logic is described in the comments to the code. In short, first we check the change in the number of open charts in the client terminal. If a chart has been added, call the FindAndCreateMissingChartObj() method, in which a missing chart object is searched for, created and added to the collection list. After that, the focus is switched to the current chart with the program (since adding a new chart in the terminal switches the focus to it). If the chart is removed from the client terminal, call the method searching for the extra chart object and removing it from the list. Finally, all chart objects are updated — the Refresh() method of the chart object checks changes in the number of windows attached to the chart and alters their number in its own window list if some changes have been detected.

**The method updating the chart object defined by the ID:**

```
//+------------------------------------------------------------------+
//| Update the specified chart object                                |
//+------------------------------------------------------------------+
void CChartObjCollection::Refresh(const long chart_id)
  {
   CChartObj *chart=this.GetChart(chart_id);
   if(chart==NULL)
      return;
   chart.Refresh();
  }
//+------------------------------------------------------------------+
```

Here, we obtain the chart object from the collection list by ID and update it.

**The method returning the full collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display complete collection description to the journal           |
//+------------------------------------------------------------------+
void CChartObjCollection::Print(void)
  {
   //--- Display the header in the journal and print all chart objects in full
   ::Print(CMessage::Text(MSG_CHART_COLLECTION_TEXT_CHART_COLLECTION),":");
   for(int i=0;i<this.m_list.Total();i++)
     {
      CChartObj *chart=this.m_list.At(i);
      if(chart==NULL)
         continue;
      chart.Print();
     }
  }
//+------------------------------------------------------------------+
```

Here, we first print the header, then, in the loop by the total number of objects in the collection list, get the next chart object and display its full description.

**The method returning the short collection list to the journal:**

```
//+------------------------------------------------------------------+
//| Display the short collection description in the journal          |
//+------------------------------------------------------------------+
void CChartObjCollection::PrintShort(void)
  {
   //--- Display the header in the journal and print short descriptions of chart objects
   ::Print(CMessage::Text(MSG_CHART_COLLECTION_TEXT_CHART_COLLECTION),":");
   for(int i=0;i<this.m_list.Total();i++)
     {
      CChartObj *chart=this.m_list.At(i);
      if(chart==NULL)
         continue;
      chart.PrintShort(true);
     }
  }
//+------------------------------------------------------------------+
```

The method logic is identical to the one considered above except that the journal displays short descriptions of chart objects.

**The method returning the number of charts in the terminal:**

```
//+------------------------------------------------------------------+
//| Return the number of charts in the terminal                      |
//+------------------------------------------------------------------+
int CChartObjCollection::ChartsTotal(void) const
  {
   //--- Declare the variables and get the first chart ID
   long currChart,prevChart=::ChartFirst();
   int res=1; // We definitely have one chart - the current one
   //--- In the loop by the total number of terminal charts (not more than 100)
   while(res<CHARTS_MAX)
     {
      //--- based on the previous one, get the new chart
      currChart=::ChartNext(prevChart);
      //--- When reaching the end of the chart list, complete the loop
      if(currChart<0) break;
      prevChart=currChart;
      res++;
     }
   //--- Return the obtained loop counter
   return res;
  }
//+------------------------------------------------------------------+
```

The method logic is described in the code comments. In short, we have at least one open chart — the one the program is launched at. Therefore, the loop counter starts from one. Next, in the loop, receive each subsequent chart based on the previous one, while increasing the counter. As a result, after the end of the loop, we return the counter value obtained in the loop.

**The method creating a new chart object and adding it to the collection list:**

```
//+------------------------------------------------------------------+
//| Create a new chart object and add it to the list                 |
//+------------------------------------------------------------------+
bool CChartObjCollection::CreateNewChartObj(const long chart_id,const string source)
  {
   ::ResetLastError();
   CChartObj *chart_obj=new CChartObj(chart_id);
   if(chart_obj==NULL)
     {
      CMessage::ToLog(source,MSG_CHART_COLLECTION_ERR_FAILED_CREATE_CHART_OBJ,true);
      return false;
     }
   this.m_list.Sort(SORT_BY_CHART_ID);
   if(!this.m_list.InsertSort(chart_obj))
     {
      CMessage::ToLog(source,MSG_CHART_COLLECTION_ERR_FAILED_ADD_CHART,true);
      delete chart_obj;
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Here the method receives the ID ( **chart\_id**) of the chart whose chart object should be created and the name ( **source**) of the method the method is called from. Create a new chart object. If failed to create it, display the appropriate message and return false. If the object is successfully created, set the flag of the list sorted by chart IDs to the collection list and try to add the object to the sorted list. If failed to add the object to the list, display the appropriate message, remove the newly created object and return false. If successful, return true.

**The method returning the pointer to the chart object by chart ID:**

```
//+------------------------------------------------------------------+
//| Return the pointer to the chart object by ID                     |
//+------------------------------------------------------------------+
CChartObj *CChartObjCollection::GetChart(const long id)
  {
   CArrayObj *list=CSelect::ByChartProperty(GetList(),CHART_PROP_ID,id,EQUAL);
   return(list!=NULL ? list.At(0) : NULL);
  }
//+------------------------------------------------------------------+
```

Here we obtain the list of chart objects having the "Chart ID" property equal to the value passed to the method (the list may contain a single chart object since the chart IDs are unique). If the object is obtained in the new list from a single element, return it. In all other cases, return NULL  — the object is not received.

**The method returning the flag indicating the existence of a chart object:**

```
//+------------------------------------------------------------------+
//| Return the flag indicating the existence of a chart object       |
//+------------------------------------------------------------------+
bool CChartObjCollection::IsPresentChartObj(const long chart_id)
  {
   return(this.GetChart(chart_id)!=NULL);
  }
//+------------------------------------------------------------------+
```

Here, if it is possible to obtain an object with a specified ID from the collection list (the request result **is** not NULL), return true. Otherwise, return false.

**The method returning the flag indicating the existence of a chart in the client terminal:**

```
//+------------------------------------------------------------------+
//| Return the flag indicating the existence of a chart              |
//+------------------------------------------------------------------+
bool CChartObjCollection::IsPresentChart(const long chart_id)
  {
   //--- Declare the variables and get the first chart ID
   long curr_chart,prev_chart=::ChartFirst();
   //--- If the IDs match, return 'true'
   if(prev_chart==chart_id)
      return true;
   int i=0;
   //--- In the loop by the total number of terminal charts (not more than 100)
   while(i<CHARTS_MAX)
     {
      //--- based on the previous one, get the new chart
      curr_chart=::ChartNext(prev_chart);
      //--- When reaching the end of the chart list, complete the loop
      if(curr_chart<0) break;
      //--- If the IDs match, return 'true'
      if(curr_chart==chart_id)
         return true;
      //--- remember the current chart ID for ChartNext() and increase the loop counter
      prev_chart=curr_chart;
      i++;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

Here in the loop by the terminal charts obtained based on the previous chart ID, check if the ID of the currently selected chart in the loop matches the value passed to the method. If the values match, the chart with such an ID is present, return true.

Upon the loop completion, return false — the chart with the specified ID was not found.

**The method searching for a missing chart object, as well as creating and adding it to the collection list:**

```
//+------------------------------------------------------------------+
//| Find a missing chart object,                                     |
//| create it and add to the collection list                         |
//+------------------------------------------------------------------+
bool CChartObjCollection::FindAndCreateMissingChartObj(void)
  {
   //--- Declare the variables and get the first chart ID
   long curr_chart,prev_chart=::ChartFirst();
   int i=0;
   //--- If the first chart object is not in the list, attempt to create and add it to the list
   if(!this.IsPresentChartObj(prev_chart) && !this.CreateNewChartObj(prev_chart,DFUN))
      return false;
   //--- In the loop by the total number of terminal charts (not more than 100), look for the rest of the charts
   while(i<CHARTS_MAX)
     {
      //--- based on the previous one, get the new chart
      curr_chart=::ChartNext(prev_chart);
      //--- When reaching the end of the chart list, complete the loop
      if(curr_chart<0) break;
      //--- If the object is not in the list, attempt to create and add it to the list
      if(!this.IsPresentChartObj(curr_chart) && !this.CreateNewChartObj(curr_chart,DFUN))
         return false;
      //--- remember the current chart ID for ChartNext() and increase the loop counter
      prev_chart=curr_chart;
      i++;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Here we also use the loop by charts based on the previous chart ID. First, determine the presence/absence of the chart object with the program in the collection list, then search for missing objects of the charts present in the terminal by all remaining charts in the loop and add them to the collection list.

**The method searching and removing a chart object missing in the terminal but present in the collection list:**

```
//+-----------------------------------------------------------------------------+
//|Find a chart object not present in the terminal and remove it from the list  |
//+-----------------------------------------------------------------------------+
void CChartObjCollection::FindAndDeleteExcessChartObj(void)
  {
   for(int i=this.m_list.Total()-1;i>WRONG_VALUE;i--)
     {
      CChartObj *chart=this.m_list.At(i);
      if(chart==NULL)
         continue;
      if(!this.IsPresentChart(chart.ID()))
        {
         m_list.Delete(i);
        }
     }
  }
//+------------------------------------------------------------------+
```

Here, in a reverse loop through all chart objects in the collection list, get another chart object. If the client terminal contains no chart with an ID matching the chart object's one, remove the object from the collection list.

**This completes the creation of the first version of the chart object collection class.**

Now we need to include the file of the created chart object collection class into the main class of the CEngine library and create the methods for accessing the methods of chart object collection class so that we are able to work with them from our programs.

In \\MQL5\\Include\\DoEasy\ **Engine.mqh** of the **CEngine** class, include the file of a newly created chart object collection class:

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
#include "Collections\TickSeriesCollection.mqh"
#include "Collections\BookSeriesCollection.mqh"
#include "Collections\MQLSignalsCollection.mqh"
#include "Collections\ChartObjCollection.mqh"
#include "TradingControl.mqh"
//+------------------------------------------------------------------+
```

In the private section of the class, declare the object of the chart collection class:

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
   CTickSeriesCollection m_tick_series;                  // Collection of tick series
   CMBookSeriesCollection m_book_series;                 // Collection of DOM series
   CMQLSignalsCollection m_signals_mql5;                 // Collection of MQL5.com Signals service signals
   CChartObjCollection  m_charts;                        // Chart collection
   CResourceCollection  m_resource;                      // Resource list
   CTradingControl      m_trading;                       // Trading management object
   CPause               m_pause;                         // Pause object
   CArrayObj            m_list_counters;                 // List of timer counters
```

In the public class section, place the methods for accessing the chart object collection class:

```
//--- Display (1) the complete, (2) short collection description in the journal and (3) parameters of the signal copying settings
   void                 SignalsMQL5Print(void)                                         { m_signals_mql5.Print();                             }
   void                 SignalsMQL5PrintShort(const bool list=false,const bool paid=true,const bool free=true)
                                                                                       { m_signals_mql5.PrintShort(list,paid,free);          }
   void                 SignalsMQL5CurrentSubscriptionParameters(void)                 { this.m_signals_mql5.CurrentSubscriptionParameters();}

//--- Current the chart collection
   bool                 ChartCreateCollection(void)                                    { return this.m_charts.CreateCollection();            }
//--- Return (1) the chart collection and (2) the list of charts from the chart collection
   CChartObjCollection *GetChartObjCollection(void)                                    { return &this.m_charts;                              }
   CArrayObj           *GetListCharts(void)                                            { return this.m_charts.GetList();                     }

//--- Return (1) the specified chart object and (2) the chart object with the program
   CChartObj           *ChartGetChartObj(const long chart_id)                          { return this.m_charts.GetChart(chart_id);            }
   CChartObj           *ChartGetMainChart(void)                                        { return this.m_charts.GetChart(this.m_charts.GetMainChartID());}

//--- Update (1) the chart specified by ID and (2) all charts
   void                 ChartRefresh(const long chart_id)                              { this.m_charts.Refresh(chart_id);                    }
   void                 ChartsRefreshAll(void)                                         { this.m_charts.Refresh();                            }

//--- Return the list of chart objects by (1) symbol and (2) timeframe
   CArrayObj           *ChartGetChartsList(const string symbol)                        { return this.m_charts.GetChartsList(symbol);         }
   CArrayObj           *ChartGetChartsList(const ENUM_TIMEFRAMES timeframe)            { return this.m_charts.GetChartsList(timeframe);      }

//--- Return (1) the buffer collection and (2) the buffer list from the collection
```

All newly updated methods simply return the result of calling the appropriate chart object collection methods I have just considered above. All these methods will be visible and available for use in our library-based programs.

In the class constructor, add creation of a new counter for the chart object collection class:

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
   this.CreateCounter(COLLECTION_TICKS_COUNTER_ID,COLLECTION_TICKS_COUNTER_STEP,COLLECTION_TICKS_PAUSE);
   this.CreateCounter(COLLECTION_CHARTS_COUNTER_ID,COLLECTION_CHARTS_COUNTER_STEP,COLLECTION_CHARTS_PAUSE);

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

In the class timer, add the code block for working with the chart object collection:

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
   //--- Tick series collection timer
      index=this.CounterIndex(COLLECTION_TICKS_COUNTER_ID);
      CTimerCounter* cnt8=this.m_list_counters.At(index);
      if(cnt8!=NULL)
        {
         //--- If the pause is over, work with the tick series list (update all except the current one)
         if(cnt8.IsTimeDone())
            this.TickSeriesRefreshAllExceptCurrent();
        }
   //--- Chart collection timer
      index=this.CounterIndex(COLLECTION_CHARTS_COUNTER_ID);
      CTimerCounter* cnt9=this.m_list_counters.At(index);
      if(cnt9!=NULL)
        {
         //--- If unpaused, work with the chart list
         if(cnt9.IsTimeDone())
            this.ChartsRefreshAll();
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
      //--- work with the list of tick series by tick
      this.TickSeriesRefreshAll();
     }
  }
//+------------------------------------------------------------------+
```

As soon as the collection timer counter counts half a second, the method of updating all chart objects in their collections is called. Thus, I have automated tracking changes in the number of charts in the terminal and windows attached to each open chart.

**Test the work of the class created today.**

### Test

To perform the test, let's use [the EA from the previous article](https://www.mql5.com/en/articles/9236#node03) and save it to \\MQL5\\Experts\\TestDoEasy\ **Part69\** as **TestDoEasyPart69.mq5**.

In fact, I will not change much in the logic of the previous Expert Advisor — I will display brief descriptions of all chart objects whose charts are open in the terminal, while the full description will be displayed for the main chart. Besides, the method of updating the collection list of the chart object collection class features displaying a chart message informing of the number of chart objects in the collection, the number of the appropriate open charts and the difference of these numbers. I will remove this debugging comment later. In the current article, I will use it to check if the chart object collection class works correctly.

First, remove inclusion of the chart object class file from the EA code:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart68.mq5 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Chart\ChartObj.mqh>
//--- enums
```

Now the chart object collection class is connected to the library main object and all chart classes and their collection are available from it.

In the input block, add the variable specifying the flag indicating the necessity to use the chart object collection class in the program:

```
//--- input variables
input    ushort            InpMagic             =  123;  // Magic number
input    double            InpLots              =  0.1;  // Lots
input    uint              InpStopLoss          =  150;  // StopLoss in points
input    uint              InpTakeProfit        =  150;  // TakeProfit in points
input    uint              InpDistance          =  50;   // Pending orders distance (points)
input    uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)
input    uint              InpDistancePReq      =  50;   // Distance for Pending Request's activate (points)
input    uint              InpBarsDelayPReq     =  5;    // Bars delay for Pending Request's activate (current timeframe)
input    uint              InpSlippage          =  5;    // Slippage in points
input    uint              InpSpreadMultiplier  =  1;    // Spread multiplier for adjusting stop-orders by StopLevel
input    uchar             InpTotalAttempts     =  5;    // Number of trading attempts
sinput   double            InpWithdrawal        =  10;   // Withdrawal funds (in tester)

sinput   uint              InpButtShiftX        =  0;    // Buttons X shift
sinput   uint              InpButtShiftY        =  10;   // Buttons Y shift

input    uint              InpTrailingStop      =  50;   // Trailing Stop (points)
input    uint              InpTrailingStep      =  20;   // Trailing Step (points)
input    uint              InpTrailingStart     =  0;    // Trailing Start (points)
input    uint              InpStopLossModify    =  20;   // StopLoss for modification (points)
input    uint              InpTakeProfitModify  =  60;   // TakeProfit for modification (points)

sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;            // Mode of used symbols list
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;            // Mode of used timeframes list
sinput   string            InpUsedTFs           =  "M1,M5,M15,M30,H1,H4,D1,W1,MN1"; // List of used timeframes (comma - separator)

sinput   ENUM_INPUT_YES_NO InpUseBook           =  INPUT_YES;                       // Use Depth of Market
sinput   ENUM_INPUT_YES_NO InpUseMqlSignals     =  INPUT_YES;                       // Use signal service
sinput   ENUM_INPUT_YES_NO InpUseCharts         =  INPUT_YES;                       // Use Charts control
sinput   ENUM_INPUT_YES_NO InpUseSounds         =  INPUT_YES;                       // Use sounds

//--- global variables
```

From the EA's OnTick() handler, remove the code block, in which I created the list of chart objects and displayed their descriptions in the journal:

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

//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();          // Trailing positions
      TrailingOrders();             // Trailing pending orders
     }

//--- If it is the first launch
   static bool done=false;
   if(!done)
     {
      //--- Create the list object for storing chart objects
      CArrayObj *list=new CArrayObj();
      if(list==NULL)
         return;
      //--- Declare the variables and get the first chart ID
      long currChart,prevChart=ChartFirst();
      int i=0;
      //--- Create the chart object and add it to the list
      CChartObj *chart_first=new CChartObj(prevChart);
      list.Add(chart_first);
      //--- In the loop by the total number of terminal charts (not more than 100)
      while(i<CHARTS_MAX)
        {
         //--- based on the previous one, get the new chart
         currChart=ChartNext(prevChart);
         //--- When reaching the end of the chart list, complete the loop
         if(currChart<0) break;
         //--- Create the chart object based on the current chart ID in the loop and add it to the list
         CChartObj *chart=new CChartObj(currChart);
         list.Add(chart);
         //--- remember the current chart ID for ChartNext() and increase the loop counter
         prevChart=currChart;
         i++;
        }
      Print("");
      //--- From the filled list in the loop, receive the next chart object and display its short description
      int total=list.Total();
      for(int j=0;j<total;j++)
        {
         CChartObj *chart_obj=list.At(j);
         if(chart_obj!=NULL)
            chart_obj.PrintShort();
        }
      Print("");
      //--- Display the full description of the current chart: in the loop by all objects of the created list
      for(int j=0;j<total;j++)
        {
         //--- get the next chart object and
         CChartObj *chart_obj=list.At(j);
         //--- if its symbol matches the current chart symbol, display its full description in the journal
         if(chart_obj!=NULL && chart_obj.Symbol()==Symbol())
            chart_obj.Print();
        }
      //--- Destroy the list of chart objects
      delete list;
      done=true;
     }
//---
  }
//+------------------------------------------------------------------+
```

Now all this is done using the following code strings:

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

//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();          // Trailing positions
      TrailingOrders();             // Trailing pending orders
     }

//--- If it is the first launch
   static bool done=false;
   if(!done)
     {
      //--- Get the chart with the EA from the collection and display its full description
      CChartObj *chart_main=engine.ChartGetMainChart();
      Print("");
      chart_main.Print();
      done=true;
     }
//---
  }
//+------------------------------------------------------------------+
```

Here we can see only the full description of a chart an EA is launched on. Short descriptions of all chart objects within the collection are to be displayed in the following code block of the OnInitDoEasy() function:

```
//--- If working with signals is not enabled or failed to create the signal collection,
//--- disable copying deals by subscription
   else
      engine.SignalsMQL5CurrentSetSubscriptionEnableOFF();

//--- Create the chart collection
//--- If working with charts and the chart collection is created
   if(InpUseCharts && engine.ChartCreateCollection())
     {
      //--- Check created chart objects - display the short collection description in the journal
      engine.GetChartObjCollection().PrintShort();
     }

//--- Create resource text files
```

In the EA's OnBookEvent() handler, comment out displaying the debugging data from the Market Depth collection class on the chart since we do not need this data now (for now, its is present due to occasional strange terminal behavior when working with the Market Depth — the EA sometimes does not respond for quite a long time, and this information, which often changes its values, allows us to see its work). Besides, the comment will prevent us from seeing the changes detected by the chart object collection class.

```
//+------------------------------------------------------------------+
//| OnBookEvent function                                             |
//+------------------------------------------------------------------+
void OnBookEvent(const string& symbol)
  {
   static bool first=true;
   //--- Leave if failed to update the symbol snapshot series
   if(!engine.OnBookEvent(symbol))
      return;
   //--- Work by the current symbol
   if(symbol==Symbol())
     {
      //--- Get the DOM snapshot series of the current symbol
      CMBookSeries *book_series=engine.GetMBookSeries(symbol);
      if(book_series==NULL)
         return;
      //--- Get the last DOM snapshot object from the DOM snapshot series object
      CMBookSnapshot *book=book_series.GetLastMBook();
      if(book==NULL)
         return;
      //--- Get the very first and last DOM order objects from the DOM snapshot object
      CMarketBookOrd *ord_0=book.GetMBookByListIndex(0);
      CMarketBookOrd *ord_N=book.GetMBookByListIndex(book.DataTotal()-1);
      if(ord_0==NULL || ord_N==NULL) return;
      //--- Display the time of the current DOM snapshot in the chart comment,
      //--- the maximum number of displayed orders in DOM for a symbol,
      //--- the obtained number of orders in the current DOM snapshot,
      //--- the total number of DOM snapshots set in the series list and
      //--- the highest and lowest orders of the current DOM snapshot
      //Comment
      //  (
      //   DFUN,book.Symbol(),": ",TimeMSCtoString(book.Time()),
      //   //", symbol book size=",sym.TicksBookdepth(),
      //   ", last book data total: ",book.DataTotal(),
      //   ", series books total: ",book_series.DataTotal(),
      //   "\nMax: ",ord_N.Header(),"\nMin: ",ord_0.Header()
      //  );
      //--- Display the first DOM snapshot in the journal
      if(first)
        {
         //--- series description
         book_series.Print();
         //--- snapshot description
         book.Print();
         first=false;
        }
     }
  }
//+------------------------------------------------------------------+
```

Let's compile the EA. Open any four charts in the terminal (launch the EA on the first one) and create the environment matching the one during the test in the previous article:

Add the fractals indicator to the chart with the EA + add the indicator window, for example, DeMarker containing another indicator, for example AMA based on DeMarker data.

Place Stochastic window on the second chart...

After launching the EA, data on the created chart object collection objects and the chart with the program is displayed in the journal:

```
Chart collection:
- Main chart window EURUSD H4 ID: 131733844391938630, HWND: 920114, Subwindows: 1
- Main chart window GBPUSD H4 ID: 131733844391938633, HWND: 592798, Subwindows: No
- Main chart window AUDUSD H1 ID: 131733844391938634, HWND: 527424, Subwindows: 2
- Main chart window USDRUB H4 ID: 131733844391938635, HWND: 920468, Subwindows: No
Subscribed to Depth of Market EURUSD
Library initialization time: 00:00:05.922

============= The beginning of the parameter list (Main chart window EURUSD H4) =============
Chart ID: 131733844391938630
Timeframe: H4
Drawing attributes of a price chart: Yes
Object "Chart": No
Chart on top of other charts: No
Accessing the context menu by pressing the right mouse button: Yes
Accessing the "Crosshair tool" by pressing the middle mouse button: Yes
Scrolling the chart horizontally using the left mouse button: Yes
Sending messages about mouse wheel events to all mql5 programs on a chart: No
Send notifications of mouse move and mouse click events to all mql5 programs on a chart: No
Send a notification of an event of new object creation to all mql5-programs on a chart: No
Send a notification of an event of object deletion to all mql5-programs on a chart: No
Chart type: Display as Japanese candlesticks
Price chart in the foreground: No
Price chart indent from the right border: Yes
Automatic moving to the right border of the chart: Yes
Managing the chart using a keyboard: Yes
Allowed to intercept Space and Enter key presses on the chart to activate the quick navigation bar: Yes
Scale: 2
Fixed scale mode: No
Scale 1:1 mode: No
Scale to be specified in points per bar: No
Display a symbol ticker in the upper left corner: Yes
Display OHLC values in the upper left corner: Yes
Display Bid values as a horizontal line in a chart: Yes
Display Ask values as a horizontal line in a chart: Yes
Display Last values as a horizontal line in a chart: No
Display vertical separators between adjacent periods: No
Display grid in the chart: No
Display volume in the chart: Trade volumes
Display textual descriptions of objects: Yes
The number of bars on the chart that can be displayed: 137
The total number of chart windows, including indicator subwindows: 2
Chart window handle: 920114
Number of the first visible bar in the chart: 136
Chart width in bars: 168
Chart width in pixels: 670
 Main chart window:
 - Chart height in pixels: 293
 Chart subwindow 1:
 - The distance between the upper frame of the indicator subwindow and the upper frame of the main chart window: 295
 - Chart height in pixels: 21
Chart background color: clrWhite
Color of axes, scales and OHLC line: clrBlack
Grid color: clrSilver
Color of volumes and position opening levels: clrGreen
Color for the up bar, shadows and body borders of bull candlesticks: clrBlack
Color for the down bar, shadows and body borders of bear candlesticks: clrBlack
Line chart color and color of "Doji" Japanese candlesticks: clrBlack
Body color of a bull candlestick: clrWhite
Body color of a bear candlestick: clrBlack
Bid price level color: clrLightSkyBlue
Ask price level color: clrCoral
Line color of the last executed deal price (Last): clrSilver
Color of stop order levels (Stop Loss and Take Profit): clrOrangeRed
Displaying trade levels in the chart (levels of open positions, Stop Loss, Take Profit and pending orders): Yes
Permission to drag trading levels on a chart with a mouse: Yes
Showing the time scale on a chart: Yes
Showing the price scale on a chart: Yes
Showing the "One click trading" panel on a chart: No
Chart window is maximized: Yes
Chart window is minimized: No
The chart window is docked: Yes
The left coordinate of the undocked chart window relative to the virtual screen: 2028
The top coordinate of the undocked chart window relative to the virtual screen: 100
The right coordinate of the undocked chart window relative to the virtual screen: 2654
The bottom coordinate of the undocked chart window relative to the virtual screen: 329
------
The size of the zero bar indent from the right border in percents: 18.93
Chart fixed position from the left border in percent value: 0.00
Fixed  chart maximum: 1.21320
Fixed  chart minimum : 1.16950
Scale in points per bar: 1.00
Chart minimum: 1.16950
Chart maximum: 1.21320
------
Text of a comment in a chart: ""
The name of the Expert Advisor running on the chart: "TestDoEasyPart69"
The name of the script running on the chart: ""
Indicators in the main chart window:
- Indicator Fractals
Indicators in the chart window 1:
- Indicator DeM(14)
- Indicator AMA(14,2,30)
Symbol: "EURUSD"
============= End of the parameter list (Main chart window EURUSD H4) =============
```

First, add several charts and remove them afterwards. The chart comments reflect the ongoing changes in the number of chart objects within the collection, the number of charts opened in the terminal and the difference between the number of charts and objects:

![](https://c.mql5.com/2/42/pdu27qfw1L.gif)

### What's next?

In the next article, I will continue expanding the functionality of the chart object collection.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9260#node00)

**\*Previous articles within the series:**

[Other classes in DoEasy library (Part 67): Chart object class](https://www.mql5.com/en/articles/9213)

[Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window](https://www.mql5.com/en/articles/9236)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9260](https://www.mql5.com/ru/articles/9260)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9260.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9260/mql5.zip "Download MQL5.zip")(3950.16 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/369603)**

![MVC design pattern and its possible application](https://c.mql5.com/2/42/MVC.png)[MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)

The article discusses a popular MVC pattern, as well as the possibilities, pros and cons of its usage in MQL programs. The idea is to split an existing code into three separate components: Model, View and Controller.

![Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__6.png)[Other classes in DoEasy library (Part 68): Chart window object class and indicator object classes in the chart window](https://www.mql5.com/en/articles/9236)

In this article, I will continue the development of the chart object class. I will add the list of chart window objects featuring the lists of available indicators.

![Combination scalping: analyzing trades from the past to increase the performance of future trades](https://c.mql5.com/2/42/logo_01.png)[Combination scalping: analyzing trades from the past to increase the performance of future trades](https://www.mql5.com/en/articles/9231)

The article provides the description of the technology aimed at increasing the effectiveness of any automated trading system. It provides a brief explanation of the idea, as well as its underlying basics, possibilities and disadvantages.

![Neural networks made easy (Part 13): Batch Normalization](https://c.mql5.com/2/48/Neural_networks_made_easy_013.png)[Neural networks made easy (Part 13): Batch Normalization](https://www.mql5.com/en/articles/9207)

In the previous article, we started considering methods aimed at improving neural network training quality. In this article, we will continue this topic and will consider another approach — batch data normalization.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iuaykrucbmufuowwygzmwypbcsxwxytz&ssn=1769253151743177348&ssn_dr=0&ssn_sr=0&fv_date=1769253151&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9260&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Other%20classes%20in%20DoEasy%20library%20(Part%2069)%3A%20Chart%20object%20collection%20class%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925315170871329&fz_uniq=5083407465216744223&sv=2552)

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