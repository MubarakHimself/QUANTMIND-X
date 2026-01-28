---
title: Library for easy and quick development of MetaTrader programs (part XVI): Symbol collection events
url: https://www.mql5.com/en/articles/7071
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:39:42.366542
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/7071&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070462373656991305)

MetaTrader 5 / Examples


### Contents

- [Base object class for all library objects](https://www.mql5.com/en/articles/7071#node01)
- [Symbol collection events](https://www.mql5.com/en/articles/7071#node02)
- [Improving the account event class](https://www.mql5.com/en/articles/7071#node03)
- [Putting the symbol event class and the improved account class to work](https://www.mql5.com/en/articles/7071#node04)
- [Testing symbol and account events](https://www.mql5.com/en/articles/7071#node05)
- [What's next?](https://www.mql5.com/en/articles/7071#node06)


When creating the account object and the account collection in the [part 12](https://www.mql5.com/en/articles/6952), as well as when tracking the current account events in the [part 13](https://www.mql5.com/en/articles/6995) of the library description, we observed the necessity to create a new type of objects sending their events to the **Engine** object.

The account events tracking principles are different from the ones applied when tracking trading events [we started considering in the article 4 and later](https://www.mql5.com/en/articles/5724). Trading events are defined and sent to the collection of trading events for a full-fledged access to any of the previously occurred events, while account events simply work in real time — "here and now": the event is defined and sent by [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) to the chart with a program before being sent to the library base object. After that, you can obtain access to the list of simultaneously occurred events from the library base object and handle them in your program. This is how events are arranged at the account object.

Events in the symbol object collection are to be arranged the same way. The difference is only in quantity — we track account events only for the current account we are connected to at the moment, but we need to track symbol events for each symbol of the collection — be it just one current symbol, two, three, or all symbols available on the server.

Here we come to an understanding that almost every object is endowed with a certain amount of properties repeating from one object to another, and we set each of these properties over and over again in each new object during its development.

This leads to an unambiguous decision — we should create a base object all library objects are to be inherited from. Currently, they are inherited from the [CObject base object](https://www.mql5.com/en/docs/standardlibrary/cobject) of the standard library. We are going to create yet another object derived from CObject and inherit all objects of our library from it. Thus, all the properties that are common for each of the objects can be set once in the base object. All descendant objects are automatically endowed with these properties.

Most interestingly, we are now able to create an event object and write it in the base object, while all library objects will be able to send their events to the program. This is exactly what we need in the library concept — the objects should be able to inform program of their status themselves, while the library or the program should handle the appropriate messages from the objects and decide (program) or handle object events (library). This will increase the interactivity of the library and greatly simplify the development of programs by end users, since the library will take over all actions directed at handling any object events.

The object event structure is to repeat data necessary for sending event ID, long, double and string parameters by the [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) function. This simplifies sending events to the program since all event data to be sent to the program is immediately filled in the class of the object where the event occurred during the event registration. We only need to obtain it and re-direct to the program for further handling.

Let's start the development. First, we are going to create the event object followed by the base object. After that, we will implement tracking symbol collection events and make adjustments keeping the [account events class](https://www.mql5.com/en/articles/6995) concept in mind. Trading events will remain intact, as their structure is completely different and contradicts the concept. Besides, they are already complete and send their data to the program.

### Base object class for all library objects

In the \\MQL5\\Include\ **DoEasy\\Objects\** library folder, create the new class **CBaseObj** in the **BaseObj.mqh** file. The CObject standard library base class is to be used as the class base object. The class is pretty small, so I will display its full listing here. Then we will analyze it by its members and methods.

To avoid creating a new file for the object event class, write that class before the base one in the same file:

```
//+------------------------------------------------------------------+
//|                                                      BaseObj.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| Base object event class for all library objects                  |
//+------------------------------------------------------------------+
class CEventBaseObj : public CObject
  {
private:
   long              m_time;
   long              m_chart_id;
   ushort            m_event_id;
   long              m_lparam;
   double            m_dparam;
   string            m_sparam;
public:
   void              Time(const long time)               { this.m_time=time;           }
   long              Time(void)                    const { return this.m_time;         }
   void              ChartID(const long chart_id)        { this.m_chart_id=chart_id;   }
   long              ChartID(void)                 const { return this.m_chart_id;     }
   void              ID(const ushort id)                 { this.m_event_id=id;         }
   ushort            ID(void)                      const { return this.m_event_id;     }
   void              LParam(const long lparam)           { this.m_lparam=lparam;       }
   long              LParam(void)                  const { return this.m_lparam;       }
   void              DParam(const double dparam)         { this.m_dparam=dparam;       }
   double            DParam(void)                  const { return this.m_dparam;       }
   void              SParam(const string sparam)         { this.m_sparam=sparam;       }
   string            SParam(void)                  const { return this.m_sparam;       }
public:
//--- Constructor
                     CEventBaseObj(const ushort event_id,const long lparam,const double dparam,const string sparam) : m_chart_id(::ChartID())
                       {
                        this.m_event_id=event_id;
                        this.m_lparam=lparam;
                        this.m_dparam=dparam;
                        this.m_sparam=sparam;
                       }
//--- Comparison method to search for identical event objects
   virtual int       Compare(const CObject *node,const int mode=0) const
                       {
                        const CEventBaseObj *compared=node;
                        return
                          (
                           this.ID()>compared.ID()          ?  1  :
                           this.ID()<compared.ID()          ? -1  :
                           this.LParam()>compared.LParam()  ?  1  :
                           this.LParam()<compared.LParam()  ? -1  :
                           this.DParam()>compared.DParam()  ?  1  :
                           this.DParam()<compared.DParam()  ? -1  :
                           this.SParam()>compared.SParam()  ?  1  :
                           this.SParam()<compared.SParam()  ? -1  :  0
                          );
                       }
  };
//+------------------------------------------------------------------+
//| Base object class for all library objects                        |
//+------------------------------------------------------------------+
class CBaseObj : public CObject
  {
private:

protected:
   CArrayObj         m_list_events;                            // Object event list
   MqlTick           m_tick;                                   // Tick structure for receiving quote data
   double            m_hash_sum;                               // Object data hash sum
   double            m_hash_sum_prev;                          // Object data hash sum during the previous check
   int               m_digits_currency;                        // Number of decimal places in an account currency
   int               m_global_error;                           // Global error code
   long              m_chart_id;                               // Control program chart ID
   bool              m_is_event;                               // Object event flag
   int               m_event_code;                             // Object event code
   string            m_name;                                   // Object name
   string            m_folder_name;                            // Name of the folder storing CBaseObj descendant objects

//--- Return time in milliseconds from MqlTick
   long              TickTime(void)                            const { return #ifdef __MQL5__ this.m_tick.time_msc #else this.m_tick.time*1000 #endif ;}
//--- return the flag of the event code presence in the event object
   bool              IsPresentEventFlag(const int change_code) const { return (this.m_event_code & change_code)==change_code; }
//--- Return the number of decimal places of the account currency
   int               DigitsCurrency(void)                      const { return this.m_digits_currency; }
//--- Returns the number of decimal places in the 'double' value
   int               GetDigits(const double value)             const;
//--- Initialize the variables of (1) tracked, (2) controlled object data (implementation in the descendants)
   virtual void      InitChangesParams(void);
   virtual void      InitControlsParams(void);
//--- (1) Check the object change, return the change code, (2) set the event type and fill in the list of events (implementation in the descendants)
   virtual int       SetEventCode(void);
   virtual void      SetTypeEvent(void);
public:
//--- Add the event object to the list
   bool              EventAdd(const ushort event_id,const long lparam,const double dparam,const string sparam);
//--- Return the occurred event flag to the object data
   bool              IsEvent(void)                             const { return this.m_is_event;              }
//--- Return (1) the list of events, (2) the object event code and (3) the global error code
   CArrayObj        *GetListEvents(void)                             { return &this.m_list_events;          }
   int               GetEventCode(void)                        const { return this.m_event_code;            }
   int               GetError(void)                            const { return this.m_global_error;          }
//--- Return the event object by its number in the list
   CEventBaseObj    *GetEvent(const int shift=WRONG_VALUE,const bool check_out=true);
//--- Return the number of object events
   int               GetEventsTotal(void)                      const { return this.m_list_events.Total();   }
//--- (1) Set and (2) return the chart ID of the control program
   void              SetChartID(const long id)                       { this.m_chart_id=id;                  }
   long              GetChartID(void)                          const { return this.m_chart_id;              }
//--- (1) Set the sub-folder name, (2) return the folder name for storing descendant object files
   void              SetSubFolderName(const string name)             { this.m_folder_name=DIRECTORY+name;   }
   string            GetFolderName(void)                       const { return this.m_folder_name;           }
//--- Return the object name
   string            GetName(void)                             const { return this.m_name;                  }
//--- Update the object data (implementation in the descendants)
   virtual void      Refresh(void);

//--- Constructor
                     CBaseObj();
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CBaseObj::CBaseObj() : m_global_error(ERR_SUCCESS),
                       m_hash_sum(0),m_hash_sum_prev(0),
                       m_is_event(false),m_event_code(0),
                       m_chart_id(::ChartID()),
                       m_folder_name(DIRECTORY),
                       m_name("")
  {
   ::ZeroMemory(this.m_tick);
   this.m_digits_currency=(#ifdef __MQL5__ (int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif);
   this.m_list_events.Clear();
   this.m_list_events.Sort();
  }
//+------------------------------------------------------------------+
//| Add the event object to the list                                 |
//+------------------------------------------------------------------+
bool CBaseObj::EventAdd(const ushort event_id,const long lparam,const double dparam,const string sparam)
  {
   CEventBaseObj *event=new CEventBaseObj(event_id,lparam,dparam,sparam);
   if(event==NULL)
      return false;
   this.m_list_events.Sort();
   if(this.m_list_events.Search(event)>WRONG_VALUE)
     {
      delete event;
      return false;
     }
   return this.m_list_events.Add(event);
  }
//+------------------------------------------------------------------+
//| Return the object event by its index in the list                 |
//+------------------------------------------------------------------+
CEventBaseObj *CBaseObj::GetEvent(const int shift=WRONG_VALUE,const bool check_out=true)
  {
   int total=this.m_list_events.Total();
   if(total==0 || (!check_out && shift>total-1))
      return NULL;
   int index=(shift<=0 ? total-1 : shift>total-1 ? 0 : total-shift-1);
   CEventBaseObj *event=this.m_list_events.At(index);
   return(event!=NULL ? event : NULL);
  }
//+------------------------------------------------------------------+
//| Return the number of decimal places in the 'double' value        |
//+------------------------------------------------------------------+
int CBaseObj::GetDigits(const double value) const
  {
   string val_str=(string)value;
   int len=::StringLen(val_str);
   int n=len-::StringFind(val_str,".",0)-1;
   if(::StringSubstr(val_str,len-1,1)=="0")
      n--;
   return n;
  }
//+------------------------------------------------------------------+
```

**Let's have a look at the base object event class**.

In the private section of the base object event class, class member variables are declared for storing all event properties:

```
private:
   long              m_time;
   long              m_chart_id;
   ushort            m_event_id;
   long              m_lparam;
   double            m_dparam;
   string            m_sparam;
```

Event time, ID of the chart the event is sent to, event ID, event long, double and string values to be passed to the control program chart.

**The values for most of these variables are passed to the class constructor:**

```
//--- Constructor
                     CEventBaseObj(const ushort event_id,const long lparam,const double dparam,const string sparam) : m_chart_id(::ChartID())
                       {
                        this.m_event_id=event_id;
                        this.m_lparam=lparam;
                        this.m_dparam=dparam;
                        this.m_sparam=sparam;
                       }
```

where the assigned values are assigned to them.

Also, in the class initialization list, the chart ID variable receives the current chart ID.

**The method of comparing two object event classes:**

```
//--- Comparison method to search for identical event objects
   virtual int       Compare(const CObject *node,const int mode=0) const
                       {
                        const CEventBaseObj *compared=node;
                        return
                          (
                           this.ID()>compared.ID()          ?  1  :
                           this.ID()<compared.ID()          ? -1  :
                           this.LParam()>compared.LParam()  ?  1  :
                           this.LParam()<compared.LParam()  ? -1  :
                           this.DParam()>compared.DParam()  ?  1  :
                           this.DParam()<compared.DParam()  ? -1  :
                           this.SParam()>compared.SParam()  ?  1  :
                           this.SParam()<compared.SParam()  ? -1  :  0
                          );
                       }
```

compares all fields of two classes (the current one and the one passed to the method by the pointer) element by element. If all fields are equal, the method returns 0, which is necessary to perform a search for the same object in the [dynamic list of the standard library pointers](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) since these objects are stored in the CArrayObj list, while its Search() method meant for searching for similar objects in the list:

```
//+------------------------------------------------------------------+
//| Search of position of element in a sorted array                  |
//+------------------------------------------------------------------+
int CArrayObj::Search(const CObject *element) const
  {
   int pos;
//--- check
   if(m_data_total==0 || !CheckPointer(element) || m_sort_mode==-1)
      return(-1);
//--- search
   pos=QuickSearch(element);
   if(m_data[pos].Compare(element,m_sort_mode)==0)
      return(pos);
//--- not found
   return(-1);
  }
//+------------------------------------------------------------------+
```

requires the Compare() virtual method present in the object. The method returns 0 in case all properties of the two objects match.

**In the class public section, declare the methods for setting and returning all object properties:**

```
public:
   void              Time(const long time)               { this.m_time=time;           }
   long              Time(void)                    const { return this.m_time;         }
   void              ChartID(const long chart_id)        { this.m_chart_id=chart_id;   }
   long              ChartID(void)                 const { return this.m_chart_id;     }
   void              ID(const ushort id)                 { this.m_event_id=id;         }
   ushort            ID(void)                      const { return this.m_event_id;     }
   void              LParam(const long lparam)           { this.m_lparam=lparam;       }
   long              LParam(void)                  const { return this.m_lparam;       }
   void              DParam(const double dparam)         { this.m_dparam=dparam;       }
   double            DParam(void)                  const { return this.m_dparam;       }
   void              SParam(const string sparam)         { this.m_sparam=sparam;       }
   string            SParam(void)                  const { return this.m_sparam;       }
```

Here all is clear, and no comments are required. This is an entire object event class.

**Let's consider the base object class for all library objects.**

The class member variables we have already encountered while dealing with the previous objects are declared in the protected class section:

```
protected:
   CArrayObj         m_list_events;                            // Object event list
   MqlTick           m_tick;                                   // Tick structure for receiving quote data
   double            m_hash_sum;                               // Object data hash sum
   double            m_hash_sum_prev;                          // Object data hash sum during the previous check
   int               m_digits_currency;                        // Number of decimal places in an account currency
   int               m_global_error;                           // Global error code
   long              m_chart_id;                               // Control program chart ID
   bool              m_is_event;                               // Object event flag
   int               m_event_code;                             // Object event code
   string            m_name;                                   // Object name
   string            m_folder_name;                            // Name of the folder storing CBaseObj descendant objects
```

- The list of object events **m\_list\_events** is to store objects of the event class we considered above. The object may have a few events at a single time, therefore we need to define all of them and place them in the list. This will allow us to extract the list of all events from the CEngine library main object and handle it.
- The structure of **m\_tick** ticks serves for obtaining prices and an event time.
- The **m\_hash\_sum** hash sum is necessary to define changes occurred in the object properties.

A change in the object properties is defined by comparing the current and previous ( **m\_hash\_sum\_prev**) hash sums.
- The number of decimal places for the **m\_digits\_currency** account currency is necessary for correct display of data on some events related to changing monetary values.
- The number of the error obtained during an erroneous return of any function is written to the **m\_global\_error** global error code. This code is passed to the calling program for its further handling.
- The control program chart ID **m\_chart\_id** is used to specify the chart an object event is to be sent to.
- The **m\_is\_event** object event flag is necessary for notifying the program of the event occurred in the surveyed object for a timely reaction to the event.

- The **m\_event\_code** object event code stores the flags of all simultaneously occurred events. The presence of these flags allows us to define the list of object events that occurred simultaneously.
- The **m\_name** object name is necessary to notify the program of some text properties of the object the events are to be received from. For example, for an account, this is the account number+client name+server name, while for a symbol, this is a symbol name.
- The name of the folder for storing object files **m\_folder\_name** is necessary for saving the object to the file: the name of the subfolder keeping the object files of the same type is stored here.

The base directory for subfolder is the directory of common files of all client terminals+library folder name: "DoEasy\\\". We have already discussed storing files when creating account collections in the [part 12 of the library description](https://www.mql5.com/en/articles/6952#node03).


We have already created all these properties in the library objects and discussed them in the various library description parts corresponding to the discussed objects. Now we placed them in a single class — the base object of all library objects **CBaseObj** removing the definitions of these class members from the descendant object classes (see the attached files).

**Let's have a look at the methods located in the class private section:**

```
//--- Return time in milliseconds from MqlTick
   long              TickTime(void)                            const { return #ifdef __MQL5__ this.m_tick.time_msc #else this.m_tick.time*1000 #endif ;}
//--- return the flag of the event code presence in the event object
   bool              IsPresentEventFlag(const int change_code) const { return (this.m_event_code & change_code)==change_code; }
//--- Return the number of decimal places of the account currency
   int               DigitsCurrency(void)                      const { return this.m_digits_currency; }
//--- Returns the number of decimal places in the 'double' value
   int               GetDigits(const double value)             const;
//--- Initialize the variables of (1) tracked, (2) controlled object data (implementation in the descendants)
   virtual void      InitChangesParams(void);
   virtual void      InitControlsParams(void);
//--- (1) Check the object change, return the change code, (2) set the event type and fill in the list of events (implementation in the descendants)
   virtual int       SetEventCode(void);
   virtual void      SetTypeEvent(void);
```

- The **TickTime()** method returns the event time in milliseconds. For MQL4, the time in seconds \* 1000 is returned due to the absence of milliseconds in the structure
- The **IsPresentEventFlag()** method returns the presence of a certain event code in the **m\_event\_code** variable.
- The **DigitsCurrency()** method returns the number of decimal places in the account currency value from the **m\_digits\_currency** variable.
- The **GetDigits()** method calculates and returns the number of decimal places in the double value passed to it.
- The **InitChangesParams()** virtual method initializes the parameters of all editable object properties.
- The **InitControlsParams()** virtual method initializes the parameters of tracked object properties.
- The **SetEventCode()** virtual method checks the changes in the object properties and returns the code of occurred changes.
- The **SetTypeEvent()** virtual method sets the type of an occurred event based on the event code and places all events to the object event list.

All these methods have already been developed in the previous articles. So, there is no point in describing them here. I only want to clarify that all virtual methods do not do anything here and should be implemented in the base object descendant classes.

**The following methods are declared in the public class section:**

```
public:
//--- Add the event object to the list
   bool              EventAdd(const ushort event_id,const long lparam,const double dparam,const string sparam);
//--- Return the occurred event flag to the object data
   bool              IsEvent(void)                             const { return this.m_is_event;              }
//--- Return (1) the list of events, (2) the object event code and (3) the global error code
   CArrayObj        *GetListEvents(void)                             { return &this.m_list_events;          }
   int               GetEventCode(void)                        const { return this.m_event_code;            }
   int               GetError(void)                            const { return this.m_global_error;          }
//--- Return the event object by its number in the list
   CEventBaseObj    *GetEvent(const int shift=WRONG_VALUE,const bool check_out=true);
//--- Return the number of object events
   int               GetEventsTotal(void)                      const { return this.m_list_events.Total();   }
//--- (1) Set and (2) return the chart ID of the control program
   void              SetChartID(const long id)                       { this.m_chart_id=id;                  }
   long              GetChartID(void)                          const { return this.m_chart_id;              }
//--- (1) Set the sub-folder name, (2) return the folder name for storing descendant object files
   void              SetSubFolderName(const string name)             { this.m_folder_name=DIRECTORY+name;   }
   string            GetFolderName(void)                       const { return this.m_folder_name;           }
//--- Return the object name
   string            GetName(void)                             const { return this.m_name;                  }
//--- Update the object data (implementation in the descendants)
   virtual void      Refresh(void);
```

In the initialization list of the **class constructor**, the initial values are assigned to the class member variables:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CBaseObj::CBaseObj() : m_global_error(ERR_SUCCESS),
                       m_hash_sum(0),m_hash_sum_prev(0),
                       m_is_event(false),m_event_code(0),
                       m_chart_id(::ChartID()),
                       m_folder_name(DIRECTORY),
                       m_name("")
  {
   ::ZeroMemory(this.m_tick);
   this.m_digits_currency=(#ifdef __MQL5__ (int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif);
   this.m_list_events.Clear();
   this.m_list_events.Sort();
  }
//+------------------------------------------------------------------+
```

The tick structure is then set to zero, the number of decimal places for the account currency is assigned, the event list is cleared and the sorted list flag is set for the object event list.

**The method of adding an event to the list:**

```
//+------------------------------------------------------------------+
//| Add the event object to the list                                 |
//+------------------------------------------------------------------+
bool CBaseObj::EventAdd(const ushort event_id,const long lparam,const double dparam,const string sparam)
  {
   CEventBaseObj *event=new CEventBaseObj(event_id,lparam,dparam,sparam);
   if(event==NULL)
      return false;
   this.m_list_events.Sort();
   if(this.m_list_events.Search(event)>WRONG_VALUE)
     {
      delete event;
      return false;
     }
   return this.m_list_events.Add(event);
  }
//+------------------------------------------------------------------+
```

Event ID, as well as long, double and string values of the event properties are passed to the method. A new event with these parameters is then created. If the same event is already present in the list, the event object and method return false, otherwise the method returns the result of adding an event object to the list.

**The method returning an event object by its index in the list:**

```
//+------------------------------------------------------------------+
//| Return the object event by its index in the list                 |
//+------------------------------------------------------------------+
CEventBaseObj *CBaseObj::GetEvent(const int shift=WRONG_VALUE,const bool check_out=true)
  {
   int total=this.m_list_events.Total();
   if(total==0 || (!check_out && shift>total-1))
      return NULL;
   int index=(shift<=0 ? total-1 : shift>total-1 ? 0 : total-shift-1);
   CEventBaseObj *event=this.m_list_events.At(index);
   return(event!=NULL ? event : NULL);
  }
//+------------------------------------------------------------------+
```

We have already considered the method before. Here, we have only added the flag defining the necessity to check and correct the index when its value moves beyond the list. By default, the index -1 is passed to the method and the check for going outside the list is performed. In this case, the method returns the most recent event object from the list. To obtain an object by its index, we need to pass the required index to the method and set the outside-the-list check flag to false. In that case, an object (or an index within the list) is returned. If the index is outside the list, NULL is returned.

**The method returning the number of decimal places in the double value** has also been considered before:

```
//+------------------------------------------------------------------+
//| Return the number of decimal places in the 'double' value        |
//+------------------------------------------------------------------+
int CBaseObj::GetDigits(const double value) const
  {
   string val_str=(string)value;
   int len=::StringLen(val_str);
   int n=len-::StringFind(val_str,".",0)-1;
   if(::StringSubstr(val_str,len-1,1)=="0")
      n--;
   return n;
  }
//+------------------------------------------------------------------+
```

**The new base object is ready**.

Now we simply need to replace the CObject base class with **CBaseObj** in each of the library base objects. These objects are:

**CAccount class object:**

```
//+------------------------------------------------------------------+
//| Account class                                                    |
//+------------------------------------------------------------------+
class CAccount : public CBaseObj
  {
```

**CSymbol class object:**

```
//+------------------------------------------------------------------+
//| Abstract symbol class                                            |
//+------------------------------------------------------------------+
class CSymbol : public CBaseObj
  {
```

The collection classes are also endowed with the general object properties:

**trading event collections:**

```
//+------------------------------------------------------------------+
//| Collection of account trading events                             |
//+------------------------------------------------------------------+
class CEventsCollection : public CBaseObj
  {
```

**account collection:**

```
//+------------------------------------------------------------------+
//| Account collection                                               |
//+------------------------------------------------------------------+
class CAccountsCollection : public CBaseObj
  {
```

**symbol collection:**

```
//+------------------------------------------------------------------+
//| Symbol collection                                                |
//+------------------------------------------------------------------+
class CSymbolsCollection : public CBaseObj
  {
```

Now, all CBaseObj-based objects have a similar set of some properties, and we do not have to reset them anew for each newly created object. Moreover, we are now able to add any properties similar for all objects and created on the basis of this basic object. Most interestingly, each of the objects now has the tool for working with events. Each of the object events has the same set of parameters as the function for sending events to the [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) program chart. Thus, we have greatly simplified further development of new objects and improvement of the ready-made ones.

Now we are ready to create symbol collection events.

### Symbol collection events

As usual, all starts with the definition of constants and enumerations. Open the **Defines.mqh** file and add data necessary for tracking symbol events.

Since symbols should be present in the Market Watch window to work with them, while their number in the window is limited,

**add the string with the macro substitution indicating the maximum possible number of symbols simultaneously located at the Market Watch window to the symbol parameters:**

```
//--- Symbol parameters
#define CLR_DEFAULT                    (0xFF000000)               // Default color
#define SYMBOLS_COMMON_TOTAL           (1000)                     // Total number of working symbols
```

**Move enumeration of modes of working with symbols** from the section of data for working with symbols

```
//+------------------------------------------------------------------+
//| Data for working with symbols                                    |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Modes of working with symbols                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOLS_MODE
  {
   SYMBOLS_MODE_CURRENT,                                    // Work with the current symbol only
   SYMBOLS_MODE_DEFINES,                                    // Work with the specified symbol list
   SYMBOLS_MODE_MARKET_WATCH,                               // Work with the Market Watch window symbols
   SYMBOLS_MODE_ALL                                         // Work with the full symbol list
  };
//+------------------------------------------------------------------+
```

**to the Datas.mqh file:**

```
//+------------------------------------------------------------------+
//|                                                        Datas.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Modes of working with symbols                                    |
//+------------------------------------------------------------------+
enum ENUM_SYMBOLS_MODE
  {
   SYMBOLS_MODE_CURRENT,                                    // Work with the current symbol only
   SYMBOLS_MODE_DEFINES,                                    // Work with the specified symbol list
   SYMBOLS_MODE_MARKET_WATCH,                               // Work with the Market Watch window symbols
   SYMBOLS_MODE_ALL                                         // Work with the full symbol list
  };
//+------------------------------------------------------------------+
//| Data sets                                                        |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Major Forex symbols                                              |
//+------------------------------------------------------------------+
string DataSymbolsFXMajors[]=
  {
```

The decision was prompted by the fact that this data is necessary not only for the library, but also for the library-based programs. This enumeration refers more to general program data than to library one. For example, the data is to be used along with many other enumerations as program inputs, which means it will be converted into the necessary compilation language (to be implemented later). Therefore, let it be in Datas.mqh.

Instead of the enumeration moved from **Defines.mqh**, **add the enumeration with the list of symbol event flags and enumeration with the list of possible symbol events:**

```
//+------------------------------------------------------------------+
//| Data for working with symbols                                    |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of symbol event flags                                       |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_EVENT_FLAGS
  {
   SYMBOL_EVENT_FLAG_NO_EVENT                =  0,          // No event
   SYMBOL_EVENT_FLAG_TRADE_MODE              =  1,          // Change order execution permissions
   SYMBOL_EVENT_FLAG_SESSION_DEALS           =  2,          // Change the number of deals in the current session
   SYMBOL_EVENT_FLAG_SESSION_BUY_ORDERS      =  4,          // Change the total number of the current buy orders
   SYMBOL_EVENT_FLAG_SESSION_SELL_ORDERS     =  8,          // Change the total number of the current sell orders
   SYMBOL_EVENT_FLAG_VOLUME                  =  16,         // Change in the last deal volume exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_VOLUME_HIGH_DAY         =  32,         // Change of the maximum volume per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_VOLUME_LOW_DAY          =  64,         // Change of the minimum volume per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_SPREAD                  =  128,        // Change of a spread exceeds the specified change value in +/-
   SYMBOL_EVENT_FLAG_STOPLEVEL               =  256,        // Change of a Stop order level exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_FREEZELEVEL             =  512,        // Change of the freeze level exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_BID_LAST                =  1024,       // Change of the Bid or Last price exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_BID_LAST_HIGH           =  2048,       // Change of the maximum Bid or Last price per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_BID_LAST_LOW            =  4096,       // Change of the minimum Bid or Last price per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_ASK                     =  8192,       // Change of the Ask price exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_ASK_HIGH                =  16384,      // Change of the maximum Ask price per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_ASK_LOW                 =  32768,      // Change of the minimum Ask price per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_VOLUME_REAL_DAY         =  65536,      // Change of the real volume per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_VOLUME_HIGH_REAL_DAY    =  131072,     // Change of the maximum real volume per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_VOLUME_LOW_REAL_DAY     =  262144,     // Change of the minimum real volume per day exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_OPTION_STRIKE           =  524288,     // Change of the strike price exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_VOLUME_LIMIT            =  1048576,    // Change of the maximum available total position volume and pending orders in one direction
   SYMBOL_EVENT_FLAG_SWAP_LONG               =  2097152,    // Change swap long
   SYMBOL_EVENT_FLAG_SWAP_SHORT              =  4194304,    // Change swap short
   SYMBOL_EVENT_FLAG_SESSION_VOLUME          =  8388608,    // Change of the total volume of deals in the current session exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_SESSION_TURNOVER        =  16777216,   // Change of the total turnover in the current session exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_SESSION_INTEREST        =  33554432,   // Change of the total volume of open positions in the current session exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_SESSION_BUY_ORD_VOLUME  =  67108864,   // Change of the total volume of buy orders exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_SESSION_SELL_ORD_VOLUME =  134217728,  // Change of the total volume of sell orders exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_SESSION_OPEN            =  268435456,  // Change of the session open price exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_SESSION_CLOSE           =  536870912,  // Change of the session close price exceeds the specified value in +/-
   SYMBOL_EVENT_FLAG_SESSION_AW              =  1073741824  // Change of the average weighted session price exceeds the specified value in +/-
  };
//+------------------------------------------------------------------+
//| List of possible symbol events                                   |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_EVENT
  {
   SYMBOL_EVENT_NO_EVENT = ACCOUNT_EVENTS_NEXT_CODE,        // No event
   SYMBOL_EVENT_MW_ADD,                                     // Adding a symbol to the Market Watch window
   SYMBOL_EVENT_MW_DEL,                                     // Removing a symbol from the Market Watch window
   SYMBOL_EVENT_MW_SORT,                                    // Sorting symbols in the Market Watch window
   SYMBOL_EVENT_TRADE_DISABLE,                              // Disable order execution
   SYMBOL_EVENT_TRADE_LONGONLY,                             // Allow buy only
   SYMBOL_EVENT_TRADE_SHORTONLY,                            // Allow sell only
   SYMBOL_EVENT_TRADE_CLOSEONLY,                            // Enable close only
   SYMBOL_EVENT_TRADE_FULL,                                 // No trading limitations
   SYMBOL_EVENT_SESSION_DEALS_INC,                          // The increase in the number of deals in the current session exceeds the specified value
   SYMBOL_EVENT_SESSION_DEALS_DEC,                          // The decrease in the number of deals in the current session exceeds the specified value
   SYMBOL_EVENT_SESSION_BUY_ORDERS_INC,                     // The increase in the total number of buy orders currently exceeds the specified value
   SYMBOL_EVENT_SESSION_BUY_ORDERS_DEC,                     // The decrease in the total number of buy orders currently exceeds the specified value
   SYMBOL_EVENT_SESSION_SELL_ORDERS_INC,                    // The increase in the total number of sell orders currently exceeds the specified value
   SYMBOL_EVENT_SESSION_SELL_ORDERS_DEC,                    // The decrease in the total number of sell orders currently exceeds the specified value
   SYMBOL_EVENT_VOLUME_INC,                                 // Volume increase in the last deal exceeds the specified value
   SYMBOL_EVENT_VOLUME_DEC,                                 // Volume decrease in the last deal exceeds the specified value
   SYMBOL_EVENT_VOLUME_HIGH_DAY_INC,                        // The increase in the maximum volume per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_HIGH_DAY_DEC,                        // The decrease in the maximum volume per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_LOW_DAY_INC,                         // The increase in the minimum volume per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_LOW_DAY_DEC,                         // The decrease in the minimum volume per day exceeds the specified value
   SYMBOL_EVENT_SPREAD_INC,                                 // The increase in a spread exceeds the specified change
   SYMBOL_EVENT_SPREAD_DEC,                                 // The decrease in a spread exceeds the specified change
   SYMBOL_EVENT_STOPLEVEL_INC,                              // The increase of a Stop order level exceeds the specified value
   SYMBOL_EVENT_STOPLEVEL_DEC,                              // The decrease of a Stop order level exceeds the specified value
   SYMBOL_EVENT_FREEZELEVEL_INC,                            // The increase in the freeze level exceeds the specified value
   SYMBOL_EVENT_FREEZELEVEL_DEC,                            // The decrease in the freeze level exceeds the specified value
   SYMBOL_EVENT_BID_LAST_INC,                               // The increase in the Bid or Last price exceeds the specified value
   SYMBOL_EVENT_BID_LAST_DEC,                               // The decrease in the Bid or Last price exceeds the specified value
   SYMBOL_EVENT_BID_LAST_HIGH_INC,                          // The increase in the maximum Bid or Last price per day exceeds the specified value
   SYMBOL_EVENT_BID_LAST_HIGH_DEC,                          // The decrease in the maximum Bid or Last price per day exceeds the specified value relative to the specified price
   SYMBOL_EVENT_BID_LAST_LOW_INC,                           // The increase in the minimum Bid or Last price per day exceeds the specified value relative to the specified price
   SYMBOL_EVENT_BID_LAST_LOW_DEC,                           // The decrease in the minimum Bid or Last price per day exceeds the specified value
   SYMBOL_EVENT_ASK_INC,                                    // The increase in the Ask price exceeds the specified value
   SYMBOL_EVENT_ASK_DEC,                                    // The decrease in the Ask price exceeds the specified value
   SYMBOL_EVENT_ASK_HIGH_INC,                               // The increase in the maximum Ask price per day exceeds the specified value
   SYMBOL_EVENT_ASK_HIGH_DEC,                               // The decrease in the maximum Ask price per day exceeds the specified value
   SYMBOL_EVENT_ASK_LOW_INC,                                // The increase in the minimum Ask price per day exceeds the specified value
   SYMBOL_EVENT_ASK_LOW_DEC,                                // The decrease in the minimum Ask price per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_REAL_DAY_INC,                        // The increase in the real volume per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_REAL_DAY_DEC,                        // The decrease in the real volume per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_HIGH_REAL_DAY_INC,                   // The increase in the maximum real volume per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_HIGH_REAL_DAY_DEC,                   // The decrease in the maximum real volume per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_LOW_REAL_DAY_INC,                    // The increase in the minimum real volume per day exceeds the specified value
   SYMBOL_EVENT_VOLUME_LOW_REAL_DAY_DEC,                    // The decrease in the minimum real volume per day exceeds the specified value
   SYMBOL_EVENT_OPTION_STRIKE_INC,                          // The increase in the strike price exceeds the specified value
   SYMBOL_EVENT_OPTION_STRIKE_DEC,                          // The decrease in the strike price exceeds the specified value
   SYMBOL_EVENT_VOLUME_LIMIT_INC,                           // The increase in the maximum available total position volume and pending orders in one direction
   SYMBOL_EVENT_VOLUME_LIMIT_DEC,                           // The decrease in the maximum available total position volume and pending orders in one direction
   SYMBOL_EVENT_SWAP_LONG_INC,                              // The increase in the swap long
   SYMBOL_EVENT_SWAP_LONG_DEC,                              // The decrease in the swap long
   SYMBOL_EVENT_SWAP_SHORT_INC,                             // The increase in the swap short
   SYMBOL_EVENT_SWAP_SHORT_DEC,                             // The decrease in the swap short
   SYMBOL_EVENT_SESSION_VOLUME_INC,                         // The increase in the total volume of deals in the current session exceeds the specified value
   SYMBOL_EVENT_SESSION_VOLUME_DEC,                         // The decrease in the total volume of deals in the current session exceeds the specified value
   SYMBOL_EVENT_SESSION_TURNOVER_INC,                       // The increase in the total turnover in the current session exceeds the specified value
   SYMBOL_EVENT_SESSION_TURNOVER_DEC,                       // The decrease in the total turnover in the current session exceeds the specified value
   SYMBOL_EVENT_SESSION_INTEREST_INC,                       // The increase in the total volume of open positions in the current session exceeds the specified value
   SYMBOL_EVENT_SESSION_INTEREST_DEC,                       // The decrease in the total volume of open positions in the current session exceeds the specified value
   SYMBOL_EVENT_SESSION_BUY_ORD_VOLUME_INC,                 // The increase in the total volume of buy orders exceeds the specified value
   SYMBOL_EVENT_SESSION_BUY_ORD_VOLUME_DEC,                 // The decrease in the total volume of buy orders exceeds the specified value
   SYMBOL_EVENT_SESSION_SELL_ORD_VOLUME_INC,                // The increase in the total volume of sell orders exceeds the specified value
   SYMBOL_EVENT_SESSION_SELL_ORD_VOLUME_DEC,                // The decrease in the total volume of sell orders exceeds the specified value
   SYMBOL_EVENT_SESSION_OPEN_INC,                           // The increase in the session open price exceeds the specified value relative to the specified price
   SYMBOL_EVENT_SESSION_OPEN_DEC,                           // The decrease in the session open price exceeds the specified value relative to the specified price
   SYMBOL_EVENT_SESSION_CLOSE_INC,                          // The increase in the session close price exceeds the specified value relative to the specified price
   SYMBOL_EVENT_SESSION_CLOSE_DEC,                          // The decrease in the session close price exceeds the specified value relative to the specified price
   SYMBOL_EVENT_SESSION_AW_INC,                             // The increase in the average weighted session price exceeds the specified value
   SYMBOL_EVENT_SESSION_AW_DEC,                             // The decrease in the average weighted session price exceeds the specified value
  };
#define SYMBOL_EVENTS_NEXT_CODE       (SYMBOL_EVENT_SESSION_AW_DEC+1)   // The code of the next event after the last symbol event code
//+------------------------------------------------------------------+
```

Here all is similar to the flag enumerations and possible account and trading events. We have considered event flags and event IDs in the [fourth part of the library description](https://www.mql5.com/en/articles/5724#node01).

**To sort symbols by their location in the Market Watch window, add yet another integer symbol property:**

```
//+------------------------------------------------------------------+
//| Symbol integer properties                                        |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_PROP_INTEGER
  {
   SYMBOL_PROP_STATUS = 0,                                  // Symbol status
   SYMBOL_PROP_INDEX_MW,                                    // Symbol index in the Market Watch window
   SYMBOL_PROP_CUSTOM,                                      // Custom symbol flag
   SYMBOL_PROP_CHART_MODE,                                  // The price type used for generating bars – Bid or Last (from the ENUM_SYMBOL_CHART_MODE enumeration)
   SYMBOL_PROP_EXIST,                                       // Flag indicating that the symbol under this name exists
   SYMBOL_PROP_SELECT,                                      // The indication that the symbol is selected in Market Watch
   SYMBOL_PROP_VISIBLE,                                     // The indication that the symbol is displayed in Market Watch
   SYMBOL_PROP_SESSION_DEALS,                               // The number of deals in the current session
   SYMBOL_PROP_SESSION_BUY_ORDERS,                          // The total number of Buy orders at the moment
   SYMBOL_PROP_SESSION_SELL_ORDERS,                         // The total number of Sell orders at the moment
   SYMBOL_PROP_VOLUME,                                      // Last deal volume
   SYMBOL_PROP_VOLUMEHIGH,                                  // Maximum volume within a day
   SYMBOL_PROP_VOLUMELOW,                                   // Minimum volume within a day
   SYMBOL_PROP_TIME,                                        // Latest quote time
   SYMBOL_PROP_DIGITS,                                      // Number of decimal places
   SYMBOL_PROP_DIGITS_LOTS,                                 // Number of decimal places for a lot
   SYMBOL_PROP_SPREAD,                                      // Spread in points
   SYMBOL_PROP_SPREAD_FLOAT,                                // Floating spread flag
   SYMBOL_PROP_TICKS_BOOKDEPTH,                             // Maximum number of orders displayed in the Depth of Market
   SYMBOL_PROP_TRADE_CALC_MODE,                             // Contract price calculation method (from the ENUM_SYMBOL_CALC_MODE enumeration)
   SYMBOL_PROP_TRADE_MODE,                                  // Order execution type (from the ENUM_SYMBOL_TRADE_MODE enumeration)
   SYMBOL_PROP_START_TIME,                                  // Symbol trading start date (usually used for futures)
   SYMBOL_PROP_EXPIRATION_TIME,                             // Symbol trading end date (usually used for futures)
   SYMBOL_PROP_TRADE_STOPS_LEVEL,                           // Minimum distance in points from the current close price for setting Stop orders
   SYMBOL_PROP_TRADE_FREEZE_LEVEL,                          // Freeze distance for trading operations (in points)
   SYMBOL_PROP_TRADE_EXEMODE,                               // Deal execution mode (from the ENUM_SYMBOL_TRADE_EXECUTION enumeration)
   SYMBOL_PROP_SWAP_MODE,                                   // Swap calculation model (from the ENUM_SYMBOL_SWAP_MODE enumeration)
   SYMBOL_PROP_SWAP_ROLLOVER3DAYS,                          // Triple-day swap (from the ENUM_DAY_OF_WEEK enumeration)
   SYMBOL_PROP_MARGIN_HEDGED_USE_LEG,                       // Calculating hedging margin using the larger leg (Buy or Sell)
   SYMBOL_PROP_EXPIRATION_MODE,                             // Flags of allowed order expiration modes
   SYMBOL_PROP_FILLING_MODE,                                // Flags of allowed order filling modes
   SYMBOL_PROP_ORDER_MODE,                                  // Flags of allowed order types
   SYMBOL_PROP_ORDER_GTC_MODE,                              // Expiration of Stop Loss and Take Profit orders if SYMBOL_EXPIRATION_MODE=SYMBOL_EXPIRATION_GTC (from the ENUM_SYMBOL_ORDER_GTC_MODE enumeration)
   SYMBOL_PROP_OPTION_MODE,                                 // Option type (from the ENUM_SYMBOL_OPTION_MODE enumeration)
   SYMBOL_PROP_OPTION_RIGHT,                                // Option right (Call/Put) (from the ENUM_SYMBOL_OPTION_RIGHT enumeration)
   //--- skipped property
   SYMBOL_PROP_BACKGROUND_COLOR                             // The color of the background used for the symbol in Market Watch
  };
#define SYMBOL_PROP_INTEGER_TOTAL    (36)                   // Total number of integer properties
#define SYMBOL_PROP_INTEGER_SKIP     (1)                    // Number of symbol integer properties not used in sorting
//+------------------------------------------------------------------+
```

Since we have added the new property, we should increase the total number of integer properties to **36**instead of 35.

Finally, add the new property **to the list of possible symbol sorting criteria:**

```
//+------------------------------------------------------------------+
//| Possible symbol sorting criteria                                 |
//+------------------------------------------------------------------+
#define FIRST_SYM_DBL_PROP          (SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_INTEGER_SKIP)
#define FIRST_SYM_STR_PROP          (SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_INTEGER_SKIP+SYMBOL_PROP_DOUBLE_TOTAL-SYMBOL_PROP_DOUBLE_SKIP)
enum ENUM_SORT_SYMBOLS_MODE
  {
//--- Sort by integer properties
   SORT_BY_SYMBOL_STATUS = 0,                               // Sort by symbol status
   SORT_BY_SYMBOL_INDEX_MW,                                 // Sort by index in the Market Watch window
   SORT_BY_SYMBOL_CUSTOM,                                   // Sort by custom symbol property
   SORT_BY_SYMBOL_CHART_MODE,                               // Sort by price type for constructing bars – Bid or Last (from the ENUM_SYMBOL_CHART_MODE enumeration)
   SORT_BY_SYMBOL_EXIST,                                    // Sort by the flag that a symbol with such a name exists
   SORT_BY_SYMBOL_SELECT,                                   // Sort by the flag indicating that a symbol is selected in Market Watch
   SORT_BY_SYMBOL_VISIBLE,                                  // Sort by the flag indicating that a selected symbol is displayed in Market Watch
   SORT_BY_SYMBOL_SESSION_DEALS,                            // Sort by the number of deals in the current session
   SORT_BY_SYMBOL_SESSION_BUY_ORDERS,                       // Sort by the total number of current buy orders
   SORT_BY_SYMBOL_SESSION_SELL_ORDERS,                      // Sort by the total number of current sell orders
   SORT_BY_SYMBOL_VOLUME,                                   // Sort by last deal volume
   SORT_BY_SYMBOL_VOLUMEHIGH,                               // Sort by maximum volume for a day
   SORT_BY_SYMBOL_VOLUMELOW,                                // Sort by minimum volume for a day
   SORT_BY_SYMBOL_TIME,                                     // Sort by the last quote time
   SORT_BY_SYMBOL_DIGITS,                                   // Sort by a number of decimal places
   SORT_BY_SYMBOL_DIGITS_LOT,                               // Sort by a number of decimal places in a lot
   SORT_BY_SYMBOL_SPREAD,                                   // Sort by spread in points
   SORT_BY_SYMBOL_SPREAD_FLOAT,                             // Sort by floating spread
   SORT_BY_SYMBOL_TICKS_BOOKDEPTH,                          // Sort by a maximum number of requests displayed in the market depth
   SORT_BY_SYMBOL_TRADE_CALC_MODE,                          // Sort by contract price calculation method (from the ENUM_SYMBOL_CALC_MODE enumeration)
   SORT_BY_SYMBOL_TRADE_MODE,                               // Sort by order execution type (from the ENUM_SYMBOL_TRADE_MODE enumeration)
   SORT_BY_SYMBOL_START_TIME,                               // Sort by an instrument trading start date (usually used for futures)
   SORT_BY_SYMBOL_EXPIRATION_TIME,                          // Sort by an instrument trading end date (usually used for futures)
   SORT_BY_SYMBOL_TRADE_STOPS_LEVEL,                        // Sort by the minimum indent from the current close price (in points) for setting Stop orders
   SORT_BY_SYMBOL_TRADE_FREEZE_LEVEL,                       // Sort by trade operation freeze distance (in points)
   SORT_BY_SYMBOL_TRADE_EXEMODE,                            // Sort by trade execution mode (from the ENUM_SYMBOL_TRADE_EXECUTION enumeration)
   SORT_BY_SYMBOL_SWAP_MODE,                                // Sort by swap calculation model (from the ENUM_SYMBOL_SWAP_MODE enumeration)
   SORT_BY_SYMBOL_SWAP_ROLLOVER3DAYS,                       // Sort by week day for accruing a triple swap (from the ENUM_DAY_OF_WEEK enumeration)
   SORT_BY_SYMBOL_MARGIN_HEDGED_USE_LEG,                    // Sort by the calculation mode of a hedged margin using the larger leg (Buy or Sell)
   SORT_BY_SYMBOL_EXPIRATION_MODE,                          // Sort by flags of allowed order expiration modes
   SORT_BY_SYMBOL_FILLING_MODE,                             // Sort by flags of allowed order filling modes
   SORT_BY_SYMBOL_ORDER_MODE,                               // Sort by flags of allowed order types
   SORT_BY_SYMBOL_ORDER_GTC_MODE,                           // Sort by StopLoss and TakeProfit orders lifetime
   SORT_BY_SYMBOL_OPTION_MODE,                              // Sort by option type (from the ENUM_SYMBOL_OPTION_MODE enumeration)
   SORT_BY_SYMBOL_OPTION_RIGHT,                             // Sort by option right (Call/Put) (from the ENUM_SYMBOL_OPTION_RIGHT enumeration)
//--- Sort by real properties
   SORT_BY_SYMBOL_BID = FIRST_SYM_DBL_PROP,                 // Sort by Bid
   SORT_BY_SYMBOL_BIDHIGH,                                  // Sort by maximum Bid for a day
   SORT_BY_SYMBOL_BIDLOW,                                   // Sort by minimum Bid for a day
   SORT_BY_SYMBOL_ASK,                                      // Sort by Ask
   SORT_BY_SYMBOL_ASKHIGH,                                  // Sort by maximum Ask for a day
   SORT_BY_SYMBOL_ASKLOW,                                   // Sort by minimum Ask for a day
   SORT_BY_SYMBOL_LAST,                                     // Sort by the last deal price
   SORT_BY_SYMBOL_LASTHIGH,                                 // Sort by maximum Last for a day
   SORT_BY_SYMBOL_LASTLOW,                                  // Sort by minimum Last for a day
   SORT_BY_SYMBOL_VOLUME_REAL,                              // Sort by Volume for a day
   SORT_BY_SYMBOL_VOLUMEHIGH_REAL,                          // Sort by maximum Volume for a day
   SORT_BY_SYMBOL_VOLUMELOW_REAL,                           // Sort by minimum Volume for a day
   SORT_BY_SYMBOL_OPTION_STRIKE,                            // Sort by an option execution price
   SORT_BY_SYMBOL_POINT,                                    // Sort by a single point value
   SORT_BY_SYMBOL_TRADE_TICK_VALUE,                         // Sort by SYMBOL_TRADE_TICK_VALUE_PROFIT value
   SORT_BY_SYMBOL_TRADE_TICK_VALUE_PROFIT,                  // Sort by a calculated tick price for a profitable position
   SORT_BY_SYMBOL_TRADE_TICK_VALUE_LOSS,                    // Sort by a calculated tick price for a loss-making position
   SORT_BY_SYMBOL_TRADE_TICK_SIZE,                          // Sort by a minimum price change
   SORT_BY_SYMBOL_TRADE_CONTRACT_SIZE,                      // Sort by a trading contract size
   SORT_BY_SYMBOL_TRADE_ACCRUED_INTEREST,                   // Sort by accrued interest
   SORT_BY_SYMBOL_TRADE_FACE_VALUE,                         // Sort by face value
   SORT_BY_SYMBOL_TRADE_LIQUIDITY_RATE,                     // Sort by liquidity rate
   SORT_BY_SYMBOL_VOLUME_MIN,                               // Sort by a minimum volume for performing a deal
   SORT_BY_SYMBOL_VOLUME_MAX,                               // Sort by a maximum volume for performing a deal
   SORT_BY_SYMBOL_VOLUME_STEP,                              // Sort by a minimum volume change step for deal execution
   SORT_BY_SYMBOL_VOLUME_LIMIT,                             // Sort by a maximum allowed aggregate volume of an open position and pending orders in one direction
   SORT_BY_SYMBOL_SWAP_LONG,                                // Sort by a long swap value
   SORT_BY_SYMBOL_SWAP_SHORT,                               // Sort by a short swap value
   SORT_BY_SYMBOL_MARGIN_INITIAL,                           // Sort by an initial margin
   SORT_BY_SYMBOL_MARGIN_MAINTENANCE,                       // Sort by a maintenance margin for an instrument
   SORT_BY_SYMBOL_MARGIN_LONG_INITIAL,                      // Sort by initial margin requirement applicable to Long orders
   SORT_BY_SYMBOL_MARGIN_BUY_STOP_INITIAL,                  // Sort by initial margin requirement applicable to BuyStop orders
   SORT_BY_SYMBOL_MARGIN_BUY_LIMIT_INITIAL,                 // Sort by initial margin requirement applicable to BuyLimit orders
   SORT_BY_SYMBOL_MARGIN_BUY_STOPLIMIT_INITIAL,             // Sort by initial margin requirement applicable to BuyStopLimit orders
   SORT_BY_SYMBOL_MARGIN_LONG_MAINTENANCE,                  // Sort by maintenance margin requirement applicable to Long orders
   SORT_BY_SYMBOL_MARGIN_BUY_STOP_MAINTENANCE,              // Sort by maintenance margin requirement applicable to BuyStop orders
   SORT_BY_SYMBOL_MARGIN_BUY_LIMIT_MAINTENANCE,             // Sort by maintenance margin requirement applicable to BuyLimit orders
   SORT_BY_SYMBOL_MARGIN_BUY_STOPLIMIT_MAINTENANCE,         // Sort by maintenance margin requirement applicable to BuyStopLimit orders
   SORT_BY_SYMBOL_MARGIN_SHORT_INITIAL,                     // Sort by initial margin requirement applicable to Short orders
   SORT_BY_SYMBOL_MARGIN_SELL_STOP_INITIAL,                 // Sort by initial margin requirement applicable to SellStop orders
   SORT_BY_SYMBOL_MARGIN_SELL_LIMIT_INITIAL,                // Sort by initial margin requirement applicable to SellLimit orders
   SORT_BY_SYMBOL_MARGIN_SELL_STOPLIMIT_INITIAL,            // Sort by initial margin requirement applicable to SellStopLimit orders
   SORT_BY_SYMBOL_MARGIN_SHORT_MAINTENANCE,                 // Sort by maintenance margin requirement applicable to Short orders
   SORT_BY_SYMBOL_MARGIN_SELL_STOP_MAINTENANCE,             // Sort by maintenance margin requirement applicable to SellStop orders
   SORT_BY_SYMBOL_MARGIN_SELL_LIMIT_MAINTENANCE,            // Sort by maintenance margin requirement applicable to SellLimit orders
   SORT_BY_SYMBOL_MARGIN_SELL_STOPLIMIT_MAINTENANCE,        // Sort by maintenance margin requirement applicable to SellStopLimit orders
   SORT_BY_SYMBOL_SESSION_VOLUME,                           // Sort by summary volume of the current session deals
   SORT_BY_SYMBOL_SESSION_TURNOVER,                         // Sort by the summary turnover of the current session
   SORT_BY_SYMBOL_SESSION_INTEREST,                         // Sort by the summary open interest
   SORT_BY_SYMBOL_SESSION_BUY_ORDERS_VOLUME,                // Sort by the current volume of Buy orders
   SORT_BY_SYMBOL_SESSION_SELL_ORDERS_VOLUME,               // Sort by the current volume of Sell orders
   SORT_BY_SYMBOL_SESSION_OPEN,                             // Sort by a session Open price
   SORT_BY_SYMBOL_SESSION_CLOSE,                            // Sort by a session Close price
   SORT_BY_SYMBOL_SESSION_AW,                               // Sort by an average weighted price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_SETTLEMENT,                 // Sort by a settlement price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_LIMIT_MIN,                  // Sort by a minimum price of the current session
   SORT_BY_SYMBOL_SESSION_PRICE_LIMIT_MAX,                  // Sort by a maximum price of the current session
   SORT_BY_SYMBOL_MARGIN_HEDGED,                            // Sort by a contract size or a margin value per one lot of hedged positions
//--- Sort by string properties
   SORT_BY_SYMBOL_NAME = FIRST_SYM_STR_PROP,                // Sort by a symbol name
   SORT_BY_SYMBOL_BASIS,                                    // Sort by an underlying asset of a derivative
   SORT_BY_SYMBOL_CURRENCY_BASE,                            // Sort by a base currency of a symbol
   SORT_BY_SYMBOL_CURRENCY_PROFIT,                          // Sort by a profit currency
   SORT_BY_SYMBOL_CURRENCY_MARGIN,                          // Sort by a margin currency
   SORT_BY_SYMBOL_BANK,                                     // Sort by a feeder of the current quote
   SORT_BY_SYMBOL_DESCRIPTION,                              // Sort by a symbol string description
   SORT_BY_SYMBOL_FORMULA,                                  // Sort by the formula used for custom symbol pricing
   SORT_BY_SYMBOL_ISIN,                                     // Sort by the name of a symbol in the ISIN system
   SORT_BY_SYMBOL_PAGE,                                     // Sort by an address of the web page containing symbol information
   SORT_BY_SYMBOL_PATH                                      // Sort by a path in the symbol tree
  };
//+------------------------------------------------------------------+
```

**We are done with the Defines.mqh file.**

Now we need to improve the symbol object class. Since we will track the changes in some symbol properties, we need to compare their current value with the previous one either in absolute terms or for detecting any increase in a certain predetermined threshold value. To achieve this, we need to create the structure of symbol properties and compare the fields of the current values' structure with the fields of the previous values' structure. We have already discussed the logic of defining the object events [when implementing tracking the account events](https://www.mql5.com/en/articles/6995#node01). Here we will do the same, except that the base object already features the methods for storing and sending events to the program.

**Connect the base object file to the CSymbol class. In the private class section, create the structure of tracked symbol properties and declare the two variables of the structure for storing the current and previous states of symbol properties:**

```
//+------------------------------------------------------------------+
//|                                                       Symbol.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Abstract symbol class                                            |
//+------------------------------------------------------------------+
class CSymbol : public CBaseObj
  {
private:
   struct MqlDataSymbol
     {
      //--- Symbol integer properties
      ENUM_SYMBOL_TRADE_MODE trade_mode;     // SYMBOL_TRADE_MODE Order filling modes
      long session_deals;                    // SYMBOL_SESSION_DEALS The number of deals in the current session
      long session_buy_orders;               // SYMBOL_SESSION_BUY_ORDERS The total number of current buy orders
      long session_sell_orders;              // SYMBOL_SESSION_SELL_ORDERS The total number of current sell orders
      long volume;                           // SYMBOL_VOLUME Last deal volume
      long volume_high_day;                  // SYMBOL_VOLUMEHIGH Maximum volume within a day
      long volume_low_day;                   // SYMBOL_VOLUMELOW Minimum volume within a day
      int spread;                            // SYMBOL_SPREAD Spread in points
      int stops_level;                       // SYMBOL_TRADE_STOPS_LEVEL Minimum distance in points from the current close price for setting Stop orders
      int freeze_level;                      // SYMBOL_TRADE_FREEZE_LEVEL Freeze distance for trading operations (in points)

      //--- Symbol real properties
      double bid_last;                       // SYMBOL_BID/SYMBOL_LAST Bid - the best sell offer/Last deal price
      double bid_last_high;                  // SYMBOL_BIDHIGH/SYMBOL_LASTHIGH Maximum Bid within the day/Maximum Last per day
      double bid_last_low;                   // SYMBOL_BIDLOW/SYMBOL_LASTLOW Minimum Bid within the day/Minimum Last per day
      double ask;                            // SYMBOL_ASK Ask - nest buy offer
      double ask_high;                       // SYMBOL_ASKHIGH Maximum Ask of the day
      double ask_low;                        // SYMBOL_ASKLOW Minimum Ask of the day
      double volume_real_day;                // SYMBOL_VOLUME_REAL Real Volume of the day
      double volume_high_real_day;           // SYMBOL_VOLUMEHIGH_REAL Maximum real Volume of the day
      double volume_low_real_day;            // SYMBOL_VOLUMELOW_REAL Minimum real Volume of the day
      double option_strike;                  // SYMBOL_OPTION_STRIKE Strike price
      double volume_limit;                   // SYMBOL_VOLUME_LIMIT Maximum permissible total volume for a position and pending orders in one direction
      double swap_long;                      // SYMBOL_SWAP_LONG Long swap value
      double swap_short;                     // SYMBOL_SWAP_SHORT Short swap value
      double session_volume;                 // SYMBOL_SESSION_VOLUME The total volume of deals in the current session
      double session_turnover;               // SYMBOL_SESSION_TURNOVER The total turnover in the current session
      double session_interest;               // SYMBOL_SESSION_INTEREST The total volume of open positions
      double session_buy_ord_volume;         // SYMBOL_SESSION_BUY_ORDERS_VOLUME The total volume of Buy orders at the moment
      double session_sell_ord_volume;        // SYMBOL_SESSION_SELL_ORDERS_VOLUME The total volume of Sell orders at the moment
      double session_open;                   // SYMBOL_SESSION_OPEN Session open price
      double session_close;                  // SYMBOL_SESSION_CLOSE Session close price
      double session_aw;                     // SYMBOL_SESSION_AW The average weighted price of the session
     };
   MqlDataSymbol    m_struct_curr_symbol;    // Current symbol data
   MqlDataSymbol    m_struct_prev_symbol;    // Previous symbol data
//---
```

Next, **declare class member variables for storing specified controlled property change values, occurred change values and flags indicating the presence of tracked events:**

```
   //--- Execution
   bool              m_is_change_trade_mode;                   // Flag of changing trading mode for a symbol
   //--- Current session deals
   long              m_control_session_deals_inc;              // Controlled value of the growth of the number of deals
   long              m_control_session_deals_dec;              // Controlled value of the decrease in the number of deals
   long              m_changed_session_deals_value;            // Value of change in the number of deals
   bool              m_is_change_session_deals_inc;            // Flag of a change in the number of deals exceeding the growth value
   bool              m_is_change_session_deals_dec;            // Flag of a change in the number of deals exceeding the decrease value
   //--- Buy orders of the current session
   long              m_control_session_buy_ord_inc;            // Controlled value of the growth of the number of Buy orders
   long              m_control_session_buy_ord_dec;            // Controlled value of the decrease in the number of Buy orders
   long              m_changed_session_buy_ord_value;          // Buy orders change value
   bool              m_is_change_session_buy_ord_inc;          // Flag of a change in the number of Buy orders exceeding the growth value
   bool              m_is_change_session_buy_ord_dec;          // Flag of a change in the number of Buy orders being less than the growth value
   //--- Sell orders of the current session
   long              m_control_session_sell_ord_inc;           // Controlled value of the growth of the number of Sell orders
   long              m_control_session_sell_ord_dec;           // Controlled value of the decrease in the number of Sell orders
   long              m_changed_session_sell_ord_value;         // Sell orders change value
   bool              m_is_change_session_sell_ord_inc;         // Flag of a change in the number of Sell orders exceeding the growth value
   bool              m_is_change_session_sell_ord_dec;         // Flag of a change in the number of Sell orders exceeding the decrease value
   //--- Volume of the last deal
   long              m_control_volume_inc;                     // Controlled value of the volume growth in the last deal
   long              m_control_volume_dec;                     // Controlled value of the volume decrease in the last deal
   long              m_changed_volume_value;                   // Value of the volume change in the last deal
   bool              m_is_change_volume_inc;                   // Flag of the volume change in the last deal exceeding the growth value
   bool              m_is_change_volume_dec;                   // Flag of the volume change in the last deal being less than the growth value
   //--- Maximum volume within a day
   long              m_control_volume_high_day_inc;            // Controlled value of the maximum volume growth for a day
   long              m_control_volume_high_day_dec;            // Controlled value of the maximum volume decrease for a day
   long              m_changed_volume_high_day_value;          // Maximum volume change value within a day
   bool              m_is_change_volume_high_day_inc;          // Flag of the maximum day volume exceeding the growth value
   bool              m_is_change_volume_high_day_dec;          // Flag of the maximum day volume exceeding the decrease value
   //--- Minimum volume within a day
   long              m_control_volume_low_day_inc;             // Controlled value of the minimum volume growth for a day
   long              m_control_volume_low_day_dec;             // Controlled value of the minimum volume decrease for a day
   long              m_changed_volume_low_day_value;           // Minimum volume change value within a day
   bool              m_is_change_volume_low_day_inc;           // Flag of the minimum day volume exceeding the growth value
   bool              m_is_change_volume_low_day_dec;           // Flag of the minimum day volume exceeding the decrease value
   //--- Spread
   int               m_control_spread_inc;                     // Controlled spread growth value in points
   int               m_control_spread_dec;                     // Controlled spread decrease value in points
   int               m_changed_spread_value;                   // Spread change value in points
   bool              m_is_change_spread_inc;                   // Flag of spread change in points exceeding the growth value
   bool              m_is_change_spread_dec;                   // Flag of spread change in points exceeding the decrease value
   //--- StopLevel
   int               m_control_stops_level_inc;                // Controlled StopLevel growth value in points
   int               m_control_stops_level_dec;                // Controlled StopLevel decrease value in points
   int               m_changed_stops_level_value;              // StopLevel change value in points
   bool              m_is_change_stops_level_inc;              // Flag of StopLevel change in points exceeding the growth value
   bool              m_is_change_stops_level_dec;              // Flag of StopLevel change in points exceeding the decrease value
   //--- Freeze distance
   int               m_control_freeze_level_inc;               // Controlled FreezeLevel growth value in points
   int               m_control_freeze_level_dec;               // Controlled FreezeLevel decrease value in points
   int               m_changed_freeze_level_value;             // FreezeLevel change value in points
   bool              m_is_change_freeze_level_inc;             // Flag of FreezeLevel change in points exceeding the growth value
   bool              m_is_change_freeze_level_dec;             // Flag of FreezeLevel change in points exceeding the decrease value

   //--- Bid/Last
   double            m_control_bid_last_inc;                   // Controlled value of Bid or Last price growth
   double            m_control_bid_last_dec;                   // Controlled value of Bid or Last price decrease
   double            m_changed_bid_last_value;                 // Bid or Last price change value
   bool              m_is_change_bid_last_inc;                 // Flag of Bid or Last price change exceeding the growth value
   bool              m_is_change_bid_last_dec;                 // Flag of Bid or Last price change exceeding the decrease value
   //--- Maximum Bid/Last of the day
   double            m_control_bid_last_high_inc;              // Controlled growth value of the maximum Bid or Last price of the day
   double            m_control_bid_last_high_dec;              // Controlled decrease value of the maximum Bid or Last price of the day
   double            m_changed_bid_last_high_value;            // Maximum Bid or Last change value for the day
   bool              m_is_change_bid_last_high_inc;            // Flag of the maximum Bid or Last price change for the day exceeding the growth value
   bool              m_is_change_bid_last_high_dec;            // Flag of the maximum Bid or Last price change for the day exceeding the decrease value
   //--- Minimum Bid/Last of the day
   double            m_control_bid_last_low_inc;               // Controlled growth value of the minimum Bid or Last price of the day
   double            m_control_bid_last_low_dec;               // Controlled decrease value of the minimum Bid or Last price of the day
   double            m_changed_bid_last_low_value;             // Minimum Bid or Last change value for the day
   bool              m_is_change_bid_last_low_inc;             // Flag of the minimum Bid or Last price change for the day exceeding the growth value
   bool              m_is_change_bid_last_low_dec;             // Flag of the minimum Bid or Last price change for the day exceeding the decrease value
   //--- Ask
   double            m_control_ask_inc;                        // Controlled value of the Ask price growth
   double            m_control_ask_dec;                        // Controlled value of the Ask price decrease
   double            m_changed_ask_value;                      // Ask price change value
   bool              m_is_change_ask_inc;                      // Flag of the Ask price change exceeding the growth value
   bool              m_is_change_ask_dec;                      // Flag of the Ask price change exceeding the decrease value
   //--- Maximum Ask price for the day
   double            m_control_ask_high_inc;                   // Controlled growth value of the maximum Ask price of the day
   double            m_control_ask_high_dec;                   // Controlled decrease value of the maximum Ask price of the day
   double            m_changed_ask_high_value;                 // Maximum Ask price change value for the day
   bool              m_is_change_ask_high_inc;                 // Flag of the maximum Ask price change for the day exceeding the growth value
   bool              m_is_change_ask_high_dec;                 // Flag of the maximum Ask price change for the day exceeding the decrease value
   //--- Minimum Ask price for the day
   double            m_control_ask_low_inc;                    // Controlled growth value of the minimum Ask price of the day
   double            m_control_ask_low_dec;                    // Controlled decrease value of the minimum Ask price of the day
   double            m_changed_ask_low_value;                  // Minimum Ask price change value for the day
   bool              m_is_change_ask_low_inc;                  // Flag of the minimum Ask price change for the day exceeding the growth value
   bool              m_is_change_ask_low_dec;                  // Flag of the minimum Ask price change for the day exceeding the decrease value
   //--- Real Volume for the day
   double            m_control_volume_real_inc;                // Controlled value of the real volume growth of the day
   double            m_control_volume_real_dec;                // Controlled value of the real volume decrease of the day
   double            m_changed_volume_real_value;              // Real volume change value of the day
   bool              m_is_change_volume_real_inc;              // Flag of the real volume change for the day exceeding the growth value
   bool              m_is_change_volume_real_dec;              // Flag of the real volume change for the day exceeding the decrease value
   //--- Maximum real volume for the day
   double            m_control_volume_high_real_day_inc;       // Controlled value of the maximum real volume growth of the day
   double            m_control_volume_high_real_day_dec;       // Controlled value of the maximum real volume decrease of the day
   double            m_changed_volume_high_real_day_value;     // Maximum real volume change value of the day
   bool              m_is_change_volume_high_real_day_inc;     // Flag of the maximum real volume change for the day exceeding the growth value
   bool              m_is_change_volume_high_real_day_dec;     // Flag of the maximum real volume change for the day exceeding the decrease value
   //--- Minimum real volume for the day
   double            m_control_volume_low_real_day_inc;        // Controlled value of the minimum real volume growth of the day
   double            m_control_volume_low_real_day_dec;        // Controlled value of the minimum real volume decrease of the day
   double            m_changed_volume_low_real_day_value;      // Minimum real volume change value of the day
   bool              m_is_change_volume_low_real_day_inc;      // Flag of the minimum real volume change for the day exceeding the growth value
   bool              m_is_change_volume_low_real_day_dec;      // Flag of the minimum real volume change for the day exceeding the decrease value
   //--- Strike price
   double            m_control_option_strike_inc;              // Controlled value of the strike price growth
   double            m_control_option_strike_dec;              // Controlled value of the strike price decrease
   double            m_changed_option_strike_value;            // Strike price change value
   bool              m_is_change_option_strike_inc;            // Flag of the strike price change exceeding the growth value
   bool              m_is_change_option_strike_dec;            // Flag of the strike price change exceeding the decrease value
   //--- Total volume of positions and orders
   double            m_changed_volume_limit_value;             // Minimum total volume change value
   bool              m_is_change_volume_limit_inc;             // Flag of the minimum total volume increase
   bool              m_is_change_volume_limit_dec;             // Flag of the minimum total volume decrease
   //---  Swap long
   double            m_changed_swap_long_value;                // Swap long change value
   bool              m_is_change_swap_long_inc;                // Flag of the swap long increase
   bool              m_is_change_swap_long_dec;                // Flag of the swap long decrease
   //---  Swap short
   double            m_changed_swap_short_value;               // Swap short change value
   bool              m_is_change_swap_short_inc;               // Flag of the swap short increase
   bool              m_is_change_swap_short_dec;               // Flag of the swap short decrease
   //--- The total volume of deals in the current session
   double            m_control_session_volume_inc;             // Controlled value of the total trade volume growth in the current session
   double            m_control_session_volume_dec;             // Controlled value of the total trade volume decrease in the current session
   double            m_changed_session_volume_value;           // The total deal volume change value in the current session
   bool              m_is_change_session_volume_inc;           // Flag of total trade volume change in the current session exceeding the growth value
   bool              m_is_change_session_volume_dec;           // Flag of total trade volume change in the current session exceeding the decrease value
   //--- The total turnover in the current session
   double            m_control_session_turnover_inc;           // Controlled value of the total turnover growth in the current session
   double            m_control_session_turnover_dec;           // Controlled value of the total turnover decrease in the current session
   double            m_changed_session_turnover_value;         // Total turnover change value in the current session
   bool              m_is_change_session_turnover_inc;         // Flag of total turnover change in the current session exceeding the growth value
   bool              m_is_change_session_turnover_dec;         // Flag of total turnover change in the current session exceeding the decrease value
   //--- The total volume of open positions
   double            m_control_session_interest_inc;           // Controlled value of the total open position volume growth in the current session
   double            m_control_session_interest_dec;           // Controlled value of the total open position volume decrease in the current session
   double            m_changed_session_interest_value;         // Change value of the open positions total volume in the current session
   bool              m_is_change_session_interest_inc;         // Flag of total open positions' volume change in the current session exceeding the growth value
   bool              m_is_change_session_interest_dec;         // Flag of total open positions' volume change in the current session exceeding the decrease value
   //--- The total volume of Buy orders at the moment
   double            m_control_session_buy_ord_volume_inc;     // Controlled value of the current total buy order volume growth
   double            m_control_session_buy_ord_volume_dec;     // Controlled value of the current total buy order volume decrease
   double            m_changed_session_buy_ord_volume_value;   // Change value of the current total buy order volume
   bool              m_is_change_session_buy_ord_volume_inc;   // Flag of changing the current total buy orders volume exceeding the growth value
   bool              m_is_change_session_buy_ord_volume_dec;   // Flag of changing the current total buy orders volume exceeding the decrease value
   //--- The total volume of Sell orders at the moment
   double            m_control_session_sell_ord_volume_inc;    // Controlled value of the current total sell order volume growth
   double            m_control_session_sell_ord_volume_dec;    // Controlled value of the current total sell order volume decrease
   double            m_changed_session_sell_ord_volume_value;  // Change value of the current total sell order volume
   bool              m_is_change_session_sell_ord_volume_inc;  // Flag of changing the current total sell orders volume exceeding the growth value
   bool              m_is_change_session_sell_ord_volume_dec;  // Flag of changing the current total sell orders volume exceeding the decrease value
   //--- Session open price
   double            m_control_session_open_inc;               // Controlled value of the session open price growth
   double            m_control_session_open_dec;               // Controlled value of the session open price decrease
   double            m_changed_session_open_value;             // Session open price change value
   bool              m_is_change_session_open_inc;             // Flag of the session open price change exceeding the growth value
   bool              m_is_change_session_open_dec;             // Flag of the session open price change exceeding the decrease value
   //--- Session close price
   double            m_control_session_close_inc;              // Controlled value of the session close price growth
   double            m_control_session_close_dec;              // Controlled value of the session close price decrease
   double            m_changed_session_close_value;            // Session close price change value
   bool              m_is_change_session_close_inc;            // Flag of the session close price change exceeding the growth value
   bool              m_is_change_session_close_dec;            // Flag of the session close price change exceeding the decrease value
   //--- The average weighted session price
   double            m_control_session_aw_inc;                 // Controlled value of the average weighted session price growth
   double            m_control_session_aw_dec;                 // Controlled value of the average weighted session price decrease
   double            m_changed_session_aw_value;               // The average weighted session price change value
   bool              m_is_change_session_aw_inc;               // Flag of the average weighted session price change value exceeding the growth value
   bool              m_is_change_session_aw_dec;               // Flag of the average weighted session price change value exceeding the decrease value

```

**In the private section, declare the virtual methods (already declared in the CBaseObj base class) for initializing tracked variables and symbol properties, as well as the method for checking the changes in the properties returning the event code and the method setting an event type by its code and writing it to the event list:**

```
//--- Initialize the variables of (1) tracked, (2) controlled symbol data
   virtual void      InitChangesParams(void);
   virtual void      InitControlsParams(void);
//--- Check symbol changes, return a change code
   virtual int       SetEventCode(void);
//--- Set an event type and fill in the event list
   virtual void      SetTypeEvent(void);

```

Let's write its implementation outside the class body.

**The method of initializing tracked symbol properties:**

```
//+------------------------------------------------------------------+
//| Initialize the variables of tracked symbol data                  |
//+------------------------------------------------------------------+
void CSymbol::InitChangesParams(void)
  {
//--- List and code of changes
   this.m_list_events.Clear();                           // Clear the change list
   this.m_list_events.Sort();                            // Sort the change list
//--- Execution
   this.m_is_change_trade_mode=false;                    // Flag of changing trading mode for a symbol
//--- Current session deals
   this.m_changed_session_deals_value=0;                 // Value of change in the number of deals
   this.m_is_change_session_deals_inc=false;             // Flag of a change in the number of deals exceeding the growth value
   this.m_is_change_session_deals_dec=false;             // Flag of a change in the number of deals exceeding the decrease value
//--- Buy orders of the current session
   this.m_changed_session_buy_ord_value=0;               // Buy orders change value
   this.m_is_change_session_buy_ord_inc=false;           // Flag of a change in the number of Buy orders exceeding the growth value
   this.m_is_change_session_buy_ord_dec=false;           // Flag of a change in the number of Buy orders exceeding the decrease value
//--- Sell orders of the current session
   this.m_changed_session_sell_ord_value=0;              // Sell orders change value
   this.m_is_change_session_sell_ord_inc=false;          // Flag of a change in the number of Sell orders exceeding the growth value
   this.m_is_change_session_sell_ord_dec=false;          // Flag of a change in the number of Sell orders exceeding the decrease value
//--- Volume of the last deal
   this.m_changed_volume_value=0;                        // Value of the volume change in the last deal
   this.m_is_change_volume_inc=false;                    // Flag of the volume change in the last deal exceeding the growth value
   this.m_is_change_volume_dec=false;                    // Flag of the volume change in the last deal exceeding the decrease value
//--- Maximum volume within a day
   this.m_changed_volume_high_day_value=0;               // Maximum volume change value within a day
   this.m_is_change_volume_high_day_inc=false;           // Flag of the maximum day volume exceeding the growth value
   this.m_is_change_volume_high_day_dec=false;           // Flag of the maximum day volume exceeding the decrease value
//--- Minimum volume within a day
   this.m_changed_volume_low_day_value=0;                // Minimum volume change value within a day
   this.m_is_change_volume_low_day_inc=false;            // Flag of the minimum day volume exceeding the growth value
   this.m_is_change_volume_low_day_dec=false;            // Flag of the minimum day volume exceeding the decrease value
//--- Spread
   this.m_changed_spread_value=0;                        // Spread change value in points
   this.m_is_change_spread_inc=false;                    // Flag of spread change in points exceeding the growth value
   this.m_is_change_spread_dec=false;                    // Flag of spread change in points exceeding the decrease value
//--- StopLevel
   this.m_changed_stops_level_value=0;                   // StopLevel change value in points
   this.m_is_change_stops_level_inc=false;               // Flag of StopLevel change in points exceeding the growth value
   this.m_is_change_stops_level_dec=false;               // Flag of StopLevel change in points exceeding the decrease value
//--- Freeze distance
   this.m_changed_freeze_level_value=0;                  // FreezeLevel change value in points
   this.m_is_change_freeze_level_inc=false;              // Flag of FreezeLevel change in points exceeding the growth value
   this.m_is_change_freeze_level_dec=false;              // Flag of FreezeLevel change in points exceeding the decrease value

//--- Bid/Last
   this.m_changed_bid_last_value=0;                      // Bid or Last price change value
   this.m_is_change_bid_last_inc=false;                  // Flag of Bid or Last price change exceeding the growth value
   this.m_is_change_bid_last_dec=false;                  // Flag of Bid or Last price change exceeding the decrease value
//--- Maximum Bid/Last of the day
   this.m_changed_bid_last_high_value=0;                 // Maximum Bid or Last change value for the day
   this.m_is_change_bid_last_high_inc=false;             // Flag of the maximum Bid or Last price change for the day exceeding the growth value
   this.m_is_change_bid_last_high_dec=false;             // Flag of the maximum Bid or Last price change for the day exceeding the decrease value
//--- Minimum Bid/Last of the day
   this.m_changed_bid_last_low_value=0;                  // Minimum Bid or Last change value for the day
   this.m_is_change_bid_last_low_inc=false;              // Flag of the minimum Bid or Last price change for the day exceeding the growth value
   this.m_is_change_bid_last_low_dec=false;              // Flag of the minimum Bid or Last price change for the day exceeding the decrease value
//--- Ask
   this.m_changed_ask_value=0;                           // Ask price change value
   this.m_is_change_ask_inc=false;                       // Flag of the Ask price change exceeding the growth value
   this.m_is_change_ask_dec=false;                       // Flag of the Ask price change exceeding the decrease value
//--- Maximum Ask price for the day
   this.m_changed_ask_high_value=0;                      // Maximum Ask price change value for the day
   this.m_is_change_ask_high_inc=false;                  // Flag of the maximum day Ask exceeding the growth value
   this.m_is_change_ask_high_dec=false;                  // Flag of the maximum day Ask exceeding the decrease value
//--- Minimum Ask price for the day
   this.m_changed_ask_low_value=0;                       // Minimum Ask price change value for the day
   this.m_is_change_ask_low_inc=false;                   // Flag of the minimum Ask volume exceeding the growth value
   this.m_is_change_ask_low_dec=false;                   // Flag of the minimum Ask volume exceeding the decrease value
//--- Real Volume for the day
   this.m_changed_volume_real_value=0;                   // Real volume change value of the day
   this.m_is_change_volume_real_inc=false;               // Flag of the real volume change for the day exceeding the growth value
   this.m_is_change_volume_real_dec=false;               // Flag of the real volume change for the day exceeding the decrease value
//--- Maximum real volume for the day
   this.m_changed_volume_high_real_day_value=0;          // Maximum real volume change value of the day
   this.m_is_change_volume_high_real_day_inc=false;      // Flag of the maximum real volume change for the day exceeding the growth value
   this.m_is_change_volume_high_real_day_dec=false;      // Flag of the maximum real volume change for the day exceeding the decrease value
//--- Minimum real volume for the day
   this.m_changed_volume_low_real_day_value=0;           // Minimum real volume change value of the day
   this.m_is_change_volume_low_real_day_inc=false;       // Flag of the minimum real volume change for the day exceeding the growth value
   this.m_is_change_volume_low_real_day_dec=false;       // Flag of the minimum real volume change for the day exceeding the decrease value
//--- Strike price
   this.m_changed_option_strike_value=0;                 // Strike price change value
   this.m_is_change_option_strike_inc=false;             // Flag of the strike price change exceeding the growth value
   this.m_is_change_option_strike_dec=false;             // Flag of the strike price change exceeding the decrease value
//--- Total volume of positions and orders
   this.m_changed_volume_limit_value=0;                  // Minimum total volume change value
   this.m_is_change_volume_limit_inc=false;              // Flag of the minimum total volume increase
   this.m_is_change_volume_limit_dec=false;              // Flag of the minimum total volume decrease
//---  Swap long
   this.m_changed_swap_long_value=0;                     // Swap long change value
   this.m_is_change_swap_long_inc=false;                 // Flag of the swap long increase
   this.m_is_change_swap_long_dec=false;                 // Flag of the swap long decrease
//---  Swap short
   this.m_changed_swap_short_value=0;                    // Swap short change value
   this.m_is_change_swap_short_inc=false;                // Flag of the swap short increase
   this.m_is_change_swap_short_dec=false;                // Flag of the swap short decrease
//--- The total volume of deals in the current session
   this.m_changed_session_volume_value=0;                // The total deal volume change value in the current session
   this.m_is_change_session_volume_inc=false;            // Flag of total trade volume change in the current session exceeding the growth value
   this.m_is_change_session_volume_dec=false;            // Flag of total trade volume change in the current session exceeding the decrease value
//--- The total turnover in the current session
   this.m_changed_session_turnover_value=0;              // Total turnover change value in the current session
   this.m_is_change_session_turnover_inc=false;          // Flag of total turnover change in the current session exceeding the growth value
   this.m_is_change_session_turnover_dec=false;          // Flag of total turnover change in the current session exceeding the decrease value
//--- The total volume of open positions
   this.m_changed_session_interest_value=0;              // Change value of the open positions total volume in the current session
   this.m_is_change_session_interest_inc=false;          // Flag of total open positions' volume change in the current session exceeding the growth value
   this.m_is_change_session_interest_dec=false;          // Flag of total open positions' volume change in the current session exceeding the decrease value
//--- The total volume of Buy orders at the moment
   this.m_changed_session_buy_ord_volume_value=0;        // Change value of the current total buy order volume
   this.m_is_change_session_buy_ord_volume_inc=false;    // Flag of changing the current total buy orders volume exceeding the growth value
   this.m_is_change_session_buy_ord_volume_dec=false;    // Flag of changing the current total buy orders volume exceeding the decrease value
//--- The total volume of Sell orders at the moment
   this.m_changed_session_sell_ord_volume_value=0;       // Change value of the current total sell order volume
   this.m_is_change_session_sell_ord_volume_inc=false;   // Flag of changing the current total sell orders volume exceeding the growth value
   this.m_is_change_session_sell_ord_volume_dec=false;   // Flag of changing the current total sell orders volume exceeding the decrease value
//--- Session open price
   this.m_changed_session_open_value=0;                  // Session open price change value
   this.m_is_change_session_open_inc=false;              // Flag of the session open price change exceeding the growth value
   this.m_is_change_session_open_dec=false;              // Flag of the session open price change exceeding the decrease value
//--- Session close price
   this.m_changed_session_close_value=0;                 // Session close price change value
   this.m_is_change_session_close_inc=false;             // Flag of the session close price change exceeding the growth value
   this.m_is_change_session_close_dec=false;             // Flag of the session close price change exceeding the decrease value
//--- The average weighted session price
   this.m_changed_session_aw_value=0;                    // The average weighted session price change value
   this.m_is_change_session_aw_inc=false;                // Flag of the average weighted session price change value exceeding the growth value
   this.m_is_change_session_aw_dec=false;                // Flag of the average weighted session price change value exceeding the decrease value
  }
//+------------------------------------------------------------------+
```

Initial values are simply assigned to variables of tracked symbol properties in the method. These are the initial variables that have been previously declared in the class private section.

**The method of initializing controlled values of symbol properties:**

```
//+------------------------------------------------------------------+
//| Initialize the variables of controlled symbol data               |
//+------------------------------------------------------------------+
void CSymbol::InitControlsParams(void)
  {
//--- Current session deals
   this.m_control_session_deals_inc=10;                  // Controlled value of the growth of the number of deals
   this.m_control_session_deals_dec=10;                  // Controlled value of the decrease in the number of deals
//--- Buy orders of the current session
   this.m_control_session_buy_ord_inc=10;                // Controlled value of the growth of the number of Buy orders
   this.m_control_session_buy_ord_dec=10;                // Controlled value of the decrease in the number of Buy orders
//--- Sell orders of the current session
   this.m_control_session_sell_ord_inc=10;               // Controlled value of the growth of the number of Sell orders
   this.m_control_session_sell_ord_dec=10;               // Controlled value of the decrease in the number of Sell orders
//--- Volume of the last deal
   this.m_control_volume_inc=10;                         // Controlled value of the volume growth in the last deal
   this.m_control_volume_dec=10;                         // Controlled value of the volume decrease in the last deal
//--- Maximum volume within a day
   this.m_control_volume_high_day_inc=50;                // Controlled value of the maximum volume growth for a day
   this.m_control_volume_high_day_dec=50;                // Controlled value of the maximum volume decrease for a day
//--- Minimum volume within a day
   this.m_control_volume_low_day_inc=50;                 // Controlled value of the minimum volume growth for a day
   this.m_control_volume_low_day_dec=50;                 // Controlled value of the minimum volume decrease for a day
//--- Spread
   this.m_control_spread_inc=2;                          // Controlled spread growth value in points
   this.m_control_spread_dec=2;                          // Controlled spread decrease value in points
//--- StopLevel
   this.m_control_stops_level_inc=2;                     // Controlled StopLevel growth value in points
   this.m_control_stops_level_dec=2;                     // Controlled StopLevel decrease value in points
//--- Freeze distance
   this.m_control_freeze_level_inc=2;                    // Controlled FreezeLevel growth value in points
   this.m_control_freeze_level_dec=2;                    // Controlled FreezeLevel decrease value in points

//--- Bid/Last
   this.m_control_bid_last_inc=DBL_MAX;                  // Controlled value of Bid or Last price growth
   this.m_control_bid_last_dec=DBL_MAX;                  // Controlled value of Bid or Last price decrease
//--- Maximum Bid/Last of the day
   this.m_control_bid_last_high_inc=DBL_MAX;             // Controlled growth value of the maximum Bid or Last price of the day
   this.m_control_bid_last_high_dec=DBL_MAX;             // Controlled decrease value of the maximum Bid or Last price of the day
//--- Minimum Bid/Last of the day
   this.m_control_bid_last_low_inc=DBL_MAX;              // Controlled growth value of the minimum Bid or Last price of the day
   this.m_control_bid_last_low_dec=DBL_MAX;              // Controlled decrease value of the minimum Bid or Last price of the day
//--- Ask
   this.m_control_ask_inc=DBL_MAX;                       // Controlled value of the Ask price growth
   this.m_control_ask_dec=DBL_MAX;                       // Controlled value of the Ask price decrease
//--- Maximum Ask price for the day
   this.m_control_ask_high_inc=DBL_MAX;                  // Controlled growth value of the maximum Ask price of the day
   this.m_control_ask_high_dec=DBL_MAX;                  // Controlled decrease value of the maximum Ask price of the day
//--- Minimum Ask price for the day
   this.m_control_ask_low_inc=DBL_MAX;                   // Controlled growth value of the minimum Ask price of the day
   this.m_control_ask_low_dec=DBL_MAX;                   // Controlled decrease value of the minimum Ask price of the day
//--- Real Volume for the day
   this.m_control_volume_real_inc=50;                    // Controlled value of the real volume growth of the day
   this.m_control_volume_real_dec=50;                    // Controlled value of the real volume decrease of the day
//--- Maximum real volume for the day
   this.m_control_volume_high_real_day_inc=20;           // Controlled value of the maximum real volume growth of the day
   this.m_control_volume_high_real_day_dec=20;           // Controlled value of the maximum real volume decrease of the day
//--- Minimum real volume for the day
   this.m_control_volume_low_real_day_inc=10;            // Controlled value of the minimum real volume growth of the day
   this.m_control_volume_low_real_day_dec=10;            // Controlled value of the minimum real volume decrease of the day
//--- Strike price
   this.m_control_option_strike_inc=0;                   // Controlled value of the strike price growth
   this.m_control_option_strike_dec=0;                   // Controlled value of the strike price decrease
//--- The total volume of deals in the current session
   this.m_control_session_volume_inc=10;                 // Controlled value of the total trade volume growth in the current session
   this.m_control_session_volume_dec=10;                 // Controlled value of the total trade volume decrease in the current session
//--- The total turnover in the current session
   this.m_control_session_turnover_inc=1000;             // Controlled value of the total turnover growth in the current session
   this.m_control_session_turnover_dec=500;              // Controlled value of the total turnover decrease in the current session
//--- The total volume of open positions
   this.m_control_session_interest_inc=50;               // Controlled value of the total open position volume growth in the current session
   this.m_control_session_interest_dec=20;               // Controlled value of the total open position volume decrease in the current session
//--- The total volume of Buy orders at the moment
   this.m_control_session_buy_ord_volume_inc=50;         // Controlled value of the current total buy order volume growth
   this.m_control_session_buy_ord_volume_dec=20;         // Controlled value of the current total buy order volume decrease
//--- The total volume of Sell orders at the moment
   this.m_control_session_sell_ord_volume_inc=50;        // Controlled value of the current total sell order volume growth
   this.m_control_session_sell_ord_volume_dec=20;        // Controlled value of the current total sell order volume decrease
//--- Session open price
   this.m_control_session_open_inc=0;                    // Controlled value of the session open price growth
   this.m_control_session_open_dec=0;                    // Controlled value of the session open price decrease
//--- Session close price
   this.m_control_session_close_inc=0;                   // Controlled value of the session close price growth
   this.m_control_session_close_dec=0;                   // Controlled value of the session close price decrease
//--- The average weighted session price
   this.m_control_session_aw_inc=0;                      // Controlled value of the average weighted session price growth
   this.m_control_session_aw_dec=0;                      // Controlled value of the average weighted session price decrease
  }
//+------------------------------------------------------------------+
```

Initial values are simply assigned to variables of controlled symbol properties in the method. These are the initial variables that have been previously declared in the class private section. If the properties exceed the values set in the variables, an appropriate account event is generated. To disable a property control, the value of DBL\_MAX should be assigned to its variable. To track any changes, assign 0 to the variable.

**The method checking the occurred symbol property change and returning the occurred change code:**

```
//+------------------------------------------------------------------+
//| Check symbol changes, return a change code                       |
//+------------------------------------------------------------------+
int CSymbol::SetEventCode(void)
  {
   this.m_event_code=SYMBOL_EVENT_FLAG_NO_EVENT;

   if(this.m_struct_curr_symbol.trade_mode!=this.m_struct_prev_symbol.trade_mode)
      this.m_event_code+=SYMBOL_EVENT_FLAG_TRADE_MODE;
   if(this.m_struct_curr_symbol.session_deals!=this.m_struct_prev_symbol.session_deals)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_DEALS;
   if(this.m_struct_curr_symbol.session_buy_orders!=this.m_struct_prev_symbol.session_buy_orders)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_BUY_ORDERS;
   if(this.m_struct_curr_symbol.session_sell_orders!=this.m_struct_prev_symbol.session_sell_orders)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_SELL_ORDERS;
   if(this.m_struct_curr_symbol.volume!=this.m_struct_prev_symbol.volume)
      this.m_event_code+=SYMBOL_EVENT_FLAG_VOLUME;
   if(this.m_struct_curr_symbol.volume_high_day!=this.m_struct_prev_symbol.volume_high_day)
      this.m_event_code+=SYMBOL_EVENT_FLAG_VOLUME_HIGH_DAY;
   if(this.m_struct_curr_symbol.volume_low_day!=this.m_struct_prev_symbol.volume_low_day)
      this.m_event_code+=SYMBOL_EVENT_FLAG_VOLUME_LOW_DAY;
   if(this.m_struct_curr_symbol.spread!=this.m_struct_prev_symbol.spread)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SPREAD;
   if(this.m_struct_curr_symbol.stops_level!=this.m_struct_prev_symbol.stops_level)
      this.m_event_code+=SYMBOL_EVENT_FLAG_STOPLEVEL;
   if(this.m_struct_curr_symbol.freeze_level!=this.m_struct_prev_symbol.freeze_level)
      this.m_event_code+=SYMBOL_EVENT_FLAG_FREEZELEVEL;

   if(this.m_struct_curr_symbol.bid_last!=this.m_struct_prev_symbol.bid_last)
      this.m_event_code+=SYMBOL_EVENT_FLAG_BID_LAST;
   if(this.m_struct_curr_symbol.bid_last_high!=this.m_struct_prev_symbol.bid_last_high)
      this.m_event_code+=SYMBOL_EVENT_FLAG_BID_LAST_HIGH;
   if(this.m_struct_curr_symbol.bid_last_low!=this.m_struct_prev_symbol.bid_last_low)
      this.m_event_code+=SYMBOL_EVENT_FLAG_BID_LAST_LOW;
   if(this.m_struct_curr_symbol.ask!=this.m_struct_prev_symbol.ask)
      this.m_event_code+=SYMBOL_EVENT_FLAG_ASK;
   if(this.m_struct_curr_symbol.ask_high!=this.m_struct_prev_symbol.ask_high)
      this.m_event_code+=SYMBOL_EVENT_FLAG_ASK_HIGH;
   if(this.m_struct_curr_symbol.ask_low!=this.m_struct_prev_symbol.ask_low)
      this.m_event_code+=SYMBOL_EVENT_FLAG_ASK_LOW;
   if(this.m_struct_curr_symbol.volume_real_day!=this.m_struct_prev_symbol.volume_real_day)
      this.m_event_code+=SYMBOL_EVENT_FLAG_VOLUME_REAL_DAY;
   if(this.m_struct_curr_symbol.volume_high_real_day!=this.m_struct_prev_symbol.volume_high_real_day)
      this.m_event_code+=SYMBOL_EVENT_FLAG_VOLUME_HIGH_REAL_DAY;
   if(this.m_struct_curr_symbol.volume_low_real_day!=this.m_struct_prev_symbol.volume_low_real_day)
      this.m_event_code+=SYMBOL_EVENT_FLAG_VOLUME_LOW_REAL_DAY;
   if(this.m_struct_curr_symbol.option_strike!=this.m_struct_prev_symbol.option_strike)
      this.m_event_code+=SYMBOL_EVENT_FLAG_OPTION_STRIKE;
   if(this.m_struct_curr_symbol.volume_limit!=this.m_struct_prev_symbol.volume_limit)
      this.m_event_code+=SYMBOL_EVENT_FLAG_VOLUME_LIMIT;
   if(this.m_struct_curr_symbol.swap_long!=this.m_struct_prev_symbol.swap_long)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SWAP_LONG;
   if(this.m_struct_curr_symbol.swap_short!=this.m_struct_prev_symbol.swap_short)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SWAP_SHORT;
   if(this.m_struct_curr_symbol.session_volume!=this.m_struct_prev_symbol.session_volume)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_VOLUME;
   if(this.m_struct_curr_symbol.session_turnover!=this.m_struct_prev_symbol.session_turnover)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_TURNOVER;
   if(this.m_struct_curr_symbol.session_interest!=this.m_struct_prev_symbol.session_interest)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_INTEREST;
   if(this.m_struct_curr_symbol.session_buy_ord_volume!=this.m_struct_prev_symbol.session_buy_ord_volume)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_BUY_ORD_VOLUME;
   if(this.m_struct_curr_symbol.session_sell_ord_volume!=this.m_struct_prev_symbol.session_sell_ord_volume)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_SELL_ORD_VOLUME;
   if(this.m_struct_curr_symbol.session_open!=this.m_struct_prev_symbol.session_open)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_OPEN;
   if(this.m_struct_curr_symbol.session_close!=this.m_struct_prev_symbol.session_close)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_CLOSE;
   if(this.m_struct_curr_symbol.session_aw!=this.m_struct_prev_symbol.session_aw)
      this.m_event_code+=SYMBOL_EVENT_FLAG_SESSION_AW;
//---
   return this.m_event_code;
  }
//+------------------------------------------------------------------+
```

First, an event code is reset in the method. Next, the values of the controlled symbol parameters are compared in the current data structure and the previous data structure. If the data is not similar, the appropriate flag is added to the event code.

**The method setting an event type and writing the occurred event to the event list:**

```
//+------------------------------------------------------------------+
//| Set a symbol object event type                                   |
//+------------------------------------------------------------------+
void CSymbol::SetTypeEvent(void)
  {
   this.InitChangesParams();
   ENUM_SYMBOL_EVENT event_id=SYMBOL_EVENT_NO_EVENT;
//--- Change of trading modes on a symbol
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_TRADE_MODE))
     {
      event_id=
        (
         this.TradeMode()==SYMBOL_TRADE_MODE_DISABLED  ? SYMBOL_EVENT_TRADE_DISABLE    :
         this.TradeMode()==SYMBOL_TRADE_MODE_LONGONLY  ? SYMBOL_EVENT_TRADE_LONGONLY   :
         this.TradeMode()==SYMBOL_TRADE_MODE_SHORTONLY ? SYMBOL_EVENT_TRADE_SHORTONLY  :
         this.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY ? SYMBOL_EVENT_TRADE_CLOSEONLY  :
         SYMBOL_EVENT_TRADE_FULL
        );
      this.m_is_change_trade_mode=true;
      if(this.EventAdd(event_id,this.TickTime(),this.TradeMode(),this.Name()))
         this.m_struct_prev_symbol.trade_mode=this.m_struct_curr_symbol.trade_mode;
     }
//--- Change of the number of deals in the current session
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_DEALS))
     {
      this.m_changed_session_deals_value=this.m_struct_curr_symbol.session_deals-this.m_struct_prev_symbol.session_deals;
      if(this.m_changed_session_deals_value>this.m_control_session_deals_inc)
        {
         this.m_is_change_session_deals_inc=true;
         event_id=SYMBOL_EVENT_SESSION_DEALS_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_deals_value,this.Name()))
            this.m_struct_prev_symbol.session_deals=this.m_struct_curr_symbol.session_deals;
        }
      else if(this.m_changed_session_deals_value<-this.m_control_session_deals_dec)
        {
         this.m_is_change_session_deals_dec=true;
         event_id=SYMBOL_EVENT_SESSION_DEALS_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_deals_value,this.Name()))
            this.m_struct_prev_symbol.session_deals=this.m_struct_curr_symbol.session_deals;
        }
     }
//--- Change of the total number of the current buy orders
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_BUY_ORDERS))
     {
      this.m_changed_session_buy_ord_value=this.m_struct_curr_symbol.session_buy_orders-this.m_struct_prev_symbol.session_buy_orders;
      if(this.m_changed_session_buy_ord_value>this.m_control_session_buy_ord_inc)
        {
         this.m_is_change_session_buy_ord_inc=true;
         event_id=SYMBOL_EVENT_SESSION_BUY_ORDERS_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_buy_ord_value,this.Name()))
            this.m_struct_prev_symbol.session_buy_orders=this.m_struct_curr_symbol.session_buy_orders;
        }
      else if(this.m_changed_session_buy_ord_value<-this.m_control_session_buy_ord_dec)
        {
         this.m_is_change_session_buy_ord_dec=true;
         event_id=SYMBOL_EVENT_SESSION_BUY_ORDERS_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_buy_ord_value,this.Name()))
            this.m_struct_prev_symbol.session_buy_orders=this.m_struct_curr_symbol.session_buy_orders;
        }
     }
//--- Change of the total number of the current sell orders
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_SELL_ORDERS))
     {
      this.m_changed_session_sell_ord_value=this.m_struct_curr_symbol.session_sell_orders-this.m_struct_prev_symbol.session_sell_orders;
      if(this.m_changed_session_sell_ord_value>this.m_control_session_sell_ord_inc)
        {
         this.m_is_change_session_sell_ord_inc=true;
         event_id=SYMBOL_EVENT_SESSION_SELL_ORDERS_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_sell_ord_value,this.Name()))
            this.m_struct_prev_symbol.session_sell_orders=this.m_struct_curr_symbol.session_sell_orders;
        }
      else if(this.m_changed_session_sell_ord_value<-this.m_control_session_sell_ord_dec)
        {
         this.m_is_change_session_sell_ord_dec=true;
         event_id=SYMBOL_EVENT_SESSION_SELL_ORDERS_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_sell_ord_value,this.Name()))
            this.m_struct_prev_symbol.session_sell_orders=this.m_struct_curr_symbol.session_sell_orders;
        }
     }
//--- Volume change in the last deal
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_VOLUME))
     {
      this.m_changed_volume_value=this.m_struct_curr_symbol.volume-this.m_struct_prev_symbol.volume;
      if(this.m_changed_volume_value>this.m_control_volume_inc)
        {
         this.m_is_change_volume_inc=true;
         event_id=SYMBOL_EVENT_VOLUME_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_value,this.Name()))
            this.m_struct_prev_symbol.volume=this.m_struct_curr_symbol.volume;
        }
      else if(this.m_changed_volume_value<-this.m_control_volume_dec)
        {
         this.m_is_change_volume_dec=true;
         event_id=SYMBOL_EVENT_VOLUME_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_value,this.Name()))
            this.m_struct_prev_symbol.volume=this.m_struct_curr_symbol.volume;
        }
     }
//--- Maximum volume change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_VOLUME_HIGH_DAY))
     {
      this.m_changed_volume_high_day_value=this.m_struct_curr_symbol.volume_high_day-this.m_struct_prev_symbol.volume_high_day;
      if(this.m_changed_volume_high_day_value>this.m_control_volume_high_day_inc)
        {
         this.m_is_change_volume_high_day_inc=true;
         event_id=SYMBOL_EVENT_VOLUME_HIGH_DAY_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_high_day_value,this.Name()))
            this.m_struct_prev_symbol.volume_high_day=this.m_struct_curr_symbol.volume_high_day;
        }
      else if(this.m_changed_volume_high_day_value<-this.m_control_volume_high_day_dec)
        {
         this.m_is_change_volume_high_day_dec=true;
         event_id=SYMBOL_EVENT_VOLUME_HIGH_DAY_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_high_day_value,this.Name()))
            this.m_struct_prev_symbol.volume_high_day=this.m_struct_curr_symbol.volume_high_day;
        }
     }
//--- Minimum volume change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_VOLUME_LOW_DAY))
     {
      this.m_changed_volume_low_day_value=this.m_struct_curr_symbol.volume_low_day-this.m_struct_prev_symbol.volume_low_day;
      if(this.m_changed_volume_low_day_value>this.m_control_volume_low_day_inc)
        {
         this.m_is_change_volume_low_day_inc=true;
         event_id=SYMBOL_EVENT_VOLUME_LOW_DAY_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_low_day_value,this.Name()))
            this.m_struct_prev_symbol.volume_low_day=this.m_struct_curr_symbol.volume_low_day;
        }
      else if(this.m_changed_volume_low_day_value<-this.m_control_volume_low_day_dec)
        {
         this.m_is_change_volume_low_day_dec=true;
         event_id=SYMBOL_EVENT_VOLUME_LOW_DAY_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_low_day_value,this.Name()))
            this.m_struct_prev_symbol.volume_low_day=this.m_struct_curr_symbol.volume_low_day;
        }
     }
//--- Spread value change in points
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SPREAD))
     {
      this.m_changed_spread_value=this.m_struct_curr_symbol.spread-this.m_struct_prev_symbol.spread;
      if(this.m_changed_spread_value>this.m_control_spread_inc)
        {
         this.m_is_change_spread_inc=true;
         event_id=SYMBOL_EVENT_SPREAD_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_spread_value,this.Name()))
            this.m_struct_prev_symbol.spread=this.m_struct_curr_symbol.spread;
        }
      else if(this.m_changed_spread_value<-this.m_control_spread_dec)
        {
         this.m_is_change_spread_dec=true;
         event_id=SYMBOL_EVENT_SPREAD_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_spread_value,this.Name()))
            this.m_struct_prev_symbol.spread=this.m_struct_curr_symbol.spread;
        }
     }
//--- StopLevel change in points
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_STOPLEVEL))
     {
      this.m_changed_stops_level_value=this.m_struct_curr_symbol.stops_level-this.m_struct_prev_symbol.stops_level;
      if(this.m_changed_stops_level_value>this.m_control_stops_level_inc)
        {
         this.m_is_change_stops_level_inc=true;
         event_id=SYMBOL_EVENT_STOPLEVEL_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_stops_level_value,this.Name()))
            this.m_struct_prev_symbol.stops_level=this.m_struct_curr_symbol.stops_level;
        }
      else if(this.m_changed_stops_level_value<-this.m_control_stops_level_dec)
        {
         this.m_is_change_stops_level_dec=true;
         event_id=SYMBOL_EVENT_STOPLEVEL_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_stops_level_value,this.Name()))
            this.m_struct_prev_symbol.stops_level=this.m_struct_curr_symbol.stops_level;
        }
     }
//--- FreezeLevel change in points
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_FREEZELEVEL))
     {
      this.m_changed_freeze_level_value=this.m_struct_curr_symbol.freeze_level-this.m_struct_prev_symbol.freeze_level;
      if(this.m_changed_freeze_level_value>this.m_control_freeze_level_inc)
        {
         this.m_is_change_freeze_level_inc=true;
         event_id=SYMBOL_EVENT_FREEZELEVEL_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_freeze_level_value,this.Name()))
            this.m_struct_prev_symbol.freeze_level=this.m_struct_curr_symbol.freeze_level;
        }
      else if(this.m_changed_freeze_level_value<-this.m_control_freeze_level_dec)
        {
         this.m_is_change_freeze_level_dec=true;
         event_id=SYMBOL_EVENT_FREEZELEVEL_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_freeze_level_value,this.Name()))
            this.m_struct_prev_symbol.freeze_level=this.m_struct_curr_symbol.freeze_level;
        }
     }
//--- Bid/Last price change
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_BID_LAST))
     {
      this.m_changed_bid_last_value=this.m_struct_curr_symbol.bid_last-this.m_struct_prev_symbol.bid_last;
      if(this.m_changed_bid_last_value>this.m_control_bid_last_inc)
        {
         this.m_is_change_bid_last_inc=true;
         event_id=SYMBOL_EVENT_BID_LAST_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_bid_last_value,this.Name()))
            this.m_struct_prev_symbol.bid_last=this.m_struct_curr_symbol.bid_last;
        }
      else if(this.m_changed_bid_last_value<-this.m_control_bid_last_dec)
        {
         this.m_is_change_bid_last_dec=true;
         event_id=SYMBOL_EVENT_BID_LAST_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_bid_last_value,this.Name()))
            this.m_struct_prev_symbol.bid_last=this.m_struct_curr_symbol.bid_last;
        }
     }
//--- Maximum Bid/Last change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_BID_LAST_HIGH))
     {
      this.m_changed_bid_last_high_value=this.m_struct_curr_symbol.bid_last_high-this.m_struct_prev_symbol.bid_last_high;
      if(this.m_changed_bid_last_high_value>this.m_control_bid_last_high_inc)
        {
         this.m_is_change_bid_last_high_inc=true;
         event_id=SYMBOL_EVENT_BID_LAST_HIGH_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_bid_last_high_value,this.Name()))
            this.m_struct_prev_symbol.bid_last_high=this.m_struct_curr_symbol.bid_last_high;
        }
      else if(this.m_changed_bid_last_high_value<-this.m_control_bid_last_high_dec)
        {
         this.m_is_change_bid_last_high_dec=true;
         event_id=SYMBOL_EVENT_BID_LAST_HIGH_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_bid_last_high_value,this.Name()))
            this.m_struct_prev_symbol.bid_last_high=this.m_struct_curr_symbol.bid_last_high;
        }
     }
//--- Minimum Bid/Last change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_BID_LAST_LOW))
     {
      this.m_changed_bid_last_low_value=this.m_struct_curr_symbol.bid_last_low-this.m_struct_prev_symbol.bid_last_low;
      if(this.m_changed_bid_last_low_value>this.m_control_bid_last_low_inc)
        {
         this.m_is_change_bid_last_low_inc=true;
         event_id=SYMBOL_EVENT_BID_LAST_LOW_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_bid_last_low_value,this.Name()))
            this.m_struct_prev_symbol.bid_last_low=this.m_struct_curr_symbol.bid_last_low;
        }
      else if(this.m_changed_bid_last_low_value<-this.m_control_bid_last_low_dec)
        {
         this.m_is_change_bid_last_low_dec=true;
         event_id=SYMBOL_EVENT_BID_LAST_LOW_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_bid_last_low_value,this.Name()))
            this.m_struct_prev_symbol.bid_last_low=this.m_struct_curr_symbol.bid_last_low;
        }
     }
//--- Ask price change
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_ASK))
     {
      this.m_changed_ask_value=this.m_struct_curr_symbol.ask-this.m_struct_prev_symbol.ask;
      if(this.m_changed_ask_value>this.m_control_ask_inc)
        {
         this.m_is_change_ask_inc=true;
         event_id=SYMBOL_EVENT_ASK_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_ask_value,this.Name()))
            this.m_struct_prev_symbol.ask=this.m_struct_curr_symbol.ask;
        }
      else if(this.m_changed_ask_value<-this.m_control_ask_dec)
        {
         this.m_is_change_ask_dec=true;
         event_id=SYMBOL_EVENT_ASK_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_ask_value,this.Name()))
            this.m_struct_prev_symbol.ask=this.m_struct_curr_symbol.ask;
        }
     }
//--- Maximum As change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_ASK_HIGH))
     {
      this.m_changed_ask_high_value=this.m_struct_curr_symbol.ask_high-this.m_struct_prev_symbol.ask_high;
      if(this.m_changed_ask_high_value>this.m_control_ask_high_inc)
        {
         this.m_is_change_ask_high_inc=true;
         event_id=SYMBOL_EVENT_ASK_HIGH_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_ask_high_value,this.Name()))
            this.m_struct_prev_symbol.ask_high=this.m_struct_curr_symbol.ask_high;
        }
      else if(this.m_changed_ask_high_value<-this.m_control_ask_high_dec)
        {
         this.m_is_change_ask_high_dec=true;
         event_id=SYMBOL_EVENT_ASK_HIGH_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_ask_high_value,this.Name()))
            this.m_struct_prev_symbol.ask_high=this.m_struct_curr_symbol.ask_high;
        }
     }
//--- Minimum Ask change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_ASK_LOW))
     {
      this.m_changed_ask_low_value=this.m_struct_curr_symbol.ask_low-this.m_struct_prev_symbol.ask_low;
      if(this.m_changed_ask_low_value>this.m_control_ask_low_inc)
        {
         this.m_is_change_ask_low_inc=true;
         event_id=SYMBOL_EVENT_ASK_LOW_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_ask_low_value,this.Name()))
            this.m_struct_prev_symbol.ask_low=this.m_struct_curr_symbol.ask_low;
        }
      else if(this.m_changed_ask_low_value<-this.m_control_ask_low_dec)
        {
         this.m_is_change_ask_low_dec=true;
         event_id=SYMBOL_EVENT_ASK_LOW_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_ask_low_value,this.Name()))
            this.m_struct_prev_symbol.ask_low=this.m_struct_curr_symbol.ask_low;
        }
     }
//--- Real volume change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_VOLUME_REAL_DAY))
     {
      this.m_changed_volume_real_value=this.m_struct_curr_symbol.volume_real_day-this.m_struct_prev_symbol.volume_real_day;
      if(this.m_changed_volume_real_value>this.m_control_volume_real_inc)
        {
         this.m_is_change_volume_real_inc=true;
         event_id=SYMBOL_EVENT_VOLUME_REAL_DAY_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_real_value,this.Name()))
            this.m_struct_prev_symbol.volume_real_day=this.m_struct_curr_symbol.volume_real_day;
        }
      else if(this.m_changed_volume_real_value<-this.m_control_volume_real_dec)
        {
         this.m_is_change_volume_real_dec=true;
         event_id=SYMBOL_EVENT_VOLUME_REAL_DAY_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_real_value,this.Name()))
            this.m_struct_prev_symbol.volume_real_day=this.m_struct_curr_symbol.volume_real_day;
        }
     }
//--- Maximum real volume change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_VOLUME_HIGH_REAL_DAY))
     {
      this.m_changed_volume_high_real_day_value=this.m_struct_curr_symbol.volume_high_real_day-this.m_struct_prev_symbol.volume_high_real_day;
      if(this.m_changed_volume_high_real_day_value>this.m_control_volume_high_real_day_inc)
        {
         this.m_is_change_volume_high_real_day_inc=true;
         event_id=SYMBOL_EVENT_VOLUME_HIGH_REAL_DAY_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_high_real_day_value,this.Name()))
            this.m_struct_prev_symbol.volume_high_real_day=this.m_struct_curr_symbol.volume_high_real_day;
        }
      else if(this.m_changed_volume_high_real_day_value<-this.m_control_volume_high_real_day_dec)
        {
         this.m_is_change_volume_high_real_day_dec=true;
         event_id=SYMBOL_EVENT_VOLUME_HIGH_REAL_DAY_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_high_real_day_value,this.Name()))
            this.m_struct_prev_symbol.volume_high_real_day=this.m_struct_curr_symbol.volume_high_real_day;
        }
     }
//--- Minimum real volume change for a day
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_VOLUME_LOW_REAL_DAY))
     {
      this.m_changed_volume_low_real_day_value=this.m_struct_curr_symbol.volume_low_real_day-this.m_struct_prev_symbol.volume_low_real_day;
      if(this.m_changed_volume_low_real_day_value>this.m_control_volume_low_real_day_inc)
        {
         this.m_is_change_volume_low_real_day_inc=true;
         event_id=SYMBOL_EVENT_VOLUME_LOW_REAL_DAY_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_low_real_day_value,this.Name()))
            this.m_struct_prev_symbol.volume_low_real_day=this.m_struct_curr_symbol.volume_low_real_day;
        }
      else if(this.m_changed_volume_low_real_day_value<-this.m_control_volume_low_real_day_dec)
        {
         this.m_is_change_volume_low_real_day_dec=true;
         event_id=SYMBOL_EVENT_VOLUME_LOW_REAL_DAY_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_low_real_day_value,this.Name()))
            this.m_struct_prev_symbol.volume_low_real_day=this.m_struct_curr_symbol.volume_low_real_day;
        }
     }
//--- Strike price change
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_OPTION_STRIKE))
     {
      this.m_changed_option_strike_value=this.m_struct_curr_symbol.option_strike-this.m_struct_prev_symbol.option_strike;
      if(this.m_changed_option_strike_value>this.m_control_option_strike_inc)
        {
         this.m_is_change_option_strike_inc=true;
         event_id=SYMBOL_EVENT_OPTION_STRIKE_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_option_strike_value,this.Name()))
            this.m_struct_prev_symbol.option_strike=this.m_struct_curr_symbol.option_strike;
        }
      else if(this.m_changed_option_strike_value<-this.m_control_option_strike_dec)
        {
         this.m_is_change_option_strike_dec=true;
         event_id=SYMBOL_EVENT_OPTION_STRIKE_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_option_strike_value,this.Name()))
            this.m_struct_prev_symbol.option_strike=this.m_struct_curr_symbol.option_strike;
        }
     }
//--- Change of the maximum available total position volume and pending orders in one direction
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_VOLUME_LIMIT))
     {
      this.m_changed_volume_limit_value=this.m_struct_curr_symbol.volume_limit-this.m_struct_prev_symbol.volume_limit;
      if(this.m_changed_volume_limit_value>0)
        {
         this.m_is_change_volume_limit_inc=true;
         event_id=SYMBOL_EVENT_VOLUME_LIMIT_INC;
        }
      else
        {
         this.m_is_change_volume_limit_dec=true;
         event_id=SYMBOL_EVENT_VOLUME_LIMIT_DEC;
        }
      if(this.EventAdd(event_id,this.TickTime(),this.m_changed_volume_limit_value,this.Name()))
         this.m_struct_prev_symbol.volume_limit=this.m_struct_curr_symbol.volume_limit;
     }
//--- Swap long change
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SWAP_LONG))
     {
      this.m_changed_swap_long_value=this.m_struct_curr_symbol.swap_long-this.m_struct_prev_symbol.swap_long;
      if(this.m_changed_swap_long_value>0)
        {
         this.m_is_change_swap_long_inc=true;
         event_id=SYMBOL_EVENT_SWAP_LONG_INC;
        }
      else
        {
         this.m_is_change_swap_long_dec=true;
         event_id=SYMBOL_EVENT_SWAP_LONG_DEC;
        }
      if(this.EventAdd(event_id,this.TickTime(),this.m_changed_swap_long_value,this.Name()))
         this.m_struct_prev_symbol.swap_long=this.m_struct_curr_symbol.swap_long;
     }
//--- Swap short change
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SWAP_SHORT))
     {
      this.m_changed_swap_short_value=this.m_struct_curr_symbol.swap_short-this.m_struct_prev_symbol.swap_short;
      if(this.m_changed_swap_short_value>0)
        {
         this.m_is_change_swap_short_inc=true;
         event_id=SYMBOL_EVENT_SWAP_SHORT_INC;
        }
      else
        {
         this.m_is_change_swap_short_dec=true;
         event_id=SYMBOL_EVENT_SWAP_SHORT_DEC;
        }
      if(this.EventAdd(event_id,this.TickTime(),this.m_changed_swap_short_value,this.Name()))
         this.m_struct_prev_symbol.swap_short=this.m_struct_curr_symbol.swap_short;
     }
//--- Change of the total volume of deals during the current session
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_VOLUME))
     {
      this.m_changed_session_volume_value=this.m_struct_curr_symbol.session_volume-this.m_struct_prev_symbol.session_volume;
      if(this.m_changed_session_volume_value>this.m_control_session_volume_inc)
        {
         this.m_is_change_session_volume_inc=true;
         event_id=SYMBOL_EVENT_SESSION_VOLUME_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_volume_value,this.Name()))
            this.m_struct_prev_symbol.session_volume=this.m_struct_curr_symbol.session_volume;
        }
      else if(this.m_changed_session_volume_value<-this.m_control_session_volume_dec)
        {
         this.m_is_change_session_volume_dec=true;
         event_id=SYMBOL_EVENT_SESSION_VOLUME_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_volume_value,this.Name()))
            this.m_struct_prev_symbol.session_volume=this.m_struct_curr_symbol.session_volume;
        }
     }
//--- Change of the total turnover during the current session
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_TURNOVER))
     {
      this.m_changed_session_turnover_value=this.m_struct_curr_symbol.session_turnover-this.m_struct_prev_symbol.session_turnover;
      if(this.m_changed_session_turnover_value>this.m_control_session_turnover_inc)
        {
         this.m_is_change_session_turnover_inc=true;
         event_id=SYMBOL_EVENT_SESSION_TURNOVER_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_turnover_value,this.Name()))
            this.m_struct_prev_symbol.session_turnover=this.m_struct_curr_symbol.session_turnover;
        }
      else if(this.m_changed_session_turnover_value<-this.m_control_session_turnover_dec)
        {
         this.m_is_change_session_turnover_dec=true;
         event_id=SYMBOL_EVENT_SESSION_TURNOVER_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_turnover_value,this.Name()))
            this.m_struct_prev_symbol.session_turnover=this.m_struct_curr_symbol.session_turnover;
        }
     }
//--- Change of the total volume of open positions during the current session
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_INTEREST))
     {
      this.m_changed_session_interest_value=this.m_struct_curr_symbol.session_interest-this.m_struct_prev_symbol.session_interest;
      if(this.m_changed_session_interest_value>this.m_control_session_interest_inc)
        {
         this.m_is_change_session_interest_inc=true;
         event_id=SYMBOL_EVENT_SESSION_INTEREST_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_interest_value,this.Name()))
            this.m_struct_prev_symbol.session_interest=this.m_struct_curr_symbol.session_interest;
        }
      else if(this.m_changed_session_interest_value<-this.m_control_session_interest_dec)
        {
         this.m_is_change_session_interest_dec=true;
         event_id=SYMBOL_EVENT_SESSION_INTEREST_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_interest_value,this.Name()))
            this.m_struct_prev_symbol.session_interest=this.m_struct_curr_symbol.session_interest;
        }
     }
//--- Change of the current total volume of buy orders
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_BUY_ORD_VOLUME))
     {
      this.m_changed_session_buy_ord_volume_value=this.m_struct_curr_symbol.session_buy_ord_volume-this.m_struct_prev_symbol.session_buy_ord_volume;
      if(this.m_changed_session_buy_ord_volume_value>this.m_control_session_buy_ord_volume_inc)
        {
         this.m_is_change_session_buy_ord_volume_inc=true;
         event_id=SYMBOL_EVENT_SESSION_BUY_ORD_VOLUME_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_buy_ord_volume_value,this.Name()))
            this.m_struct_prev_symbol.session_buy_ord_volume=this.m_struct_curr_symbol.session_buy_ord_volume;
        }
      else if(this.m_changed_session_buy_ord_volume_value<-this.m_control_session_buy_ord_volume_dec)
        {
         this.m_is_change_session_buy_ord_volume_dec=true;
         event_id=SYMBOL_EVENT_SESSION_BUY_ORD_VOLUME_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_buy_ord_volume_value,this.Name()))
            this.m_struct_prev_symbol.session_buy_ord_volume=this.m_struct_curr_symbol.session_buy_ord_volume;
        }
     }
//--- Change of the current total volume of sell orders
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_SELL_ORD_VOLUME))
     {
      this.m_changed_session_sell_ord_volume_value=this.m_struct_curr_symbol.session_sell_ord_volume-this.m_struct_prev_symbol.session_sell_ord_volume;
      if(this.m_changed_session_sell_ord_volume_value>this.m_control_session_sell_ord_volume_inc)
        {
         this.m_is_change_session_sell_ord_volume_inc=true;
         event_id=SYMBOL_EVENT_SESSION_SELL_ORD_VOLUME_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_sell_ord_volume_value,this.Name()))
            this.m_struct_prev_symbol.session_sell_ord_volume=this.m_struct_curr_symbol.session_sell_ord_volume;
        }
      else if(this.m_changed_session_sell_ord_volume_value<-this.m_control_session_sell_ord_volume_dec)
        {
         this.m_is_change_session_sell_ord_volume_dec=true;
         event_id=SYMBOL_EVENT_SESSION_SELL_ORD_VOLUME_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_sell_ord_volume_value,this.Name()))
            this.m_struct_prev_symbol.session_sell_ord_volume=this.m_struct_curr_symbol.session_sell_ord_volume;
        }
     }
//--- Session open price change
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_OPEN))
     {
      this.m_changed_session_open_value=this.m_struct_curr_symbol.session_open-this.m_struct_prev_symbol.session_open;
      if(this.m_changed_session_open_value>this.m_control_session_open_inc)
        {
         this.m_is_change_session_open_inc=true;
         event_id=SYMBOL_EVENT_SESSION_OPEN_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_open_value,this.Name()))
            this.m_struct_prev_symbol.session_open=this.m_struct_curr_symbol.session_open;
        }
      else if(this.m_changed_session_open_value<-this.m_control_session_open_dec)
        {
         this.m_is_change_session_open_dec=true;
         event_id=SYMBOL_EVENT_SESSION_OPEN_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_open_value,this.Name()))
            this.m_struct_prev_symbol.session_open=this.m_struct_curr_symbol.session_open;
        }
     }
//--- Session close price change
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_CLOSE))
     {
      this.m_changed_session_close_value=this.m_struct_curr_symbol.session_close-this.m_struct_prev_symbol.session_close;
      if(this.m_changed_session_close_value>this.m_control_session_close_inc)
        {
         this.m_is_change_session_close_inc=true;
         event_id=SYMBOL_EVENT_SESSION_CLOSE_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_close_value,this.Name()))
            this.m_struct_prev_symbol.session_close=this.m_struct_curr_symbol.session_close;
        }
      else if(this.m_changed_session_close_value<-this.m_control_session_close_dec)
        {
         this.m_is_change_session_close_dec=true;
         event_id=SYMBOL_EVENT_SESSION_CLOSE_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_close_value,this.Name()))
            this.m_struct_prev_symbol.session_close=this.m_struct_curr_symbol.session_close;
        }
     }
//--- Average weighted session price change
   if(this.IsPresentEventFlag(SYMBOL_EVENT_FLAG_SESSION_AW))
     {
      this.m_changed_session_aw_value=this.m_struct_curr_symbol.session_aw-this.m_struct_prev_symbol.session_aw;
      if(this.m_changed_session_aw_value>this.m_control_session_aw_inc)
        {
         this.m_is_change_session_aw_inc=true;
         event_id=SYMBOL_EVENT_SESSION_AW_INC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_aw_value,this.Name()))
            this.m_struct_prev_symbol.session_aw=this.m_struct_curr_symbol.session_aw;
        }
      else if(this.m_changed_session_aw_value<-this.m_control_session_aw_dec)
        {
         this.m_is_change_session_aw_dec=true;
         event_id=SYMBOL_EVENT_SESSION_AW_DEC;
         if(this.EventAdd(event_id,this.TickTime(),this.m_changed_session_aw_value,this.Name()))
            this.m_struct_prev_symbol.session_aw=this.m_struct_curr_symbol.session_aw;
        }
     }
  }
//+------------------------------------------------------------------+
```

The method is quite "bulky" due to the blocks of identical checks of changes in the tracked symbol properties. [In the part 13](https://www.mql5.com/en/articles/6995), we have already considered a similar method when tracking account changes. Here the logic is similar, except that the entire functionality of saving events is now located in the **CBaseObj** base object.

**Let's consider it using changing the average weighted session price as an example.**

At the very start of the method, reset the values of tracked properties using the **InitChangesParams()** method and set the event status as "No event".

- Using the **IsPresentEventFlag()** method also located in the base object, check if the flag of the [SYMBOL\_SESSION\_AW](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) symbol value change is present in the event code set in the **m\_event\_code** variable.

If the flag of the checked value change is present in the event code,


  - calculate the property value and check if the value exceeds the controlled one.


    - If the calculated value exceeds the controlled one,


      - set the flag of the weighted average price event,

      - write to the "average weighted price increase" event ID and

      - call the method of adding the event to the EventAdd() list of the CBaseObj base class.

         If the event has been successfully added to the list,


        - save the current property value as a previous one for further check.

To register the event of a decrease in the property value, all checks are performed in the same way. The only difference is that we check the value falling below the controlled value.

Send data to the method of adding an event to the EventAdd() list:

1. event ID (event\_id filled when checking the event)

2. the current time in milliseconds (TickTime() method of the CBaseObj base class)
3. calculated value, by which the symbol property changed (m\_changed\_session\_aw\_value)
4. object name (here it is symbol name)

Let's also make a slight change in the protected class constructor: in order to fill in the object symbol's new property "symbol index in the Market Watch window", we need to pass this index when scanning market watch symbols. The index is to be passed directly to the class constructor:

```
protected:
//--- Protected parametric constructor
                     CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name,const int index);
```

In the same protected class section, add yet another method returning the number of decimal places by swap charge method:

```
//--- Get and return integer properties of a selected symbol from its parameters
   bool              SymbolExists(const string name)     const;
   long              SymbolExists(void)                  const;
   long              SymbolCustom(void)                  const;
   long              SymbolChartMode(void)               const;
   long              SymbolMarginHedgedUseLEG(void)      const;
   long              SymbolOrderFillingMode(void)        const;
   long              SymbolOrderMode(void)               const;
   long              SymbolExpirationMode(void)          const;
   long              SymbolOrderGTCMode(void)            const;
   long              SymbolOptionMode(void)              const;
   long              SymbolOptionRight(void)             const;
   long              SymbolBackgroundColor(void)         const;
   long              SymbolCalcMode(void)                const;
   long              SymbolSwapMode(void)                const;
   long              SymbolDigitsLot(void);
   int               SymbolDigitsBySwap(void);
//--- Get and return real properties of a selected symbol from its parameters
```

Swaps are charged in money, points and percentage. For each of these swap charge types, we need to return the appropriate number of decimal places. This is exactly what the method does. Let's write it outside the class body:

```
//+------------------------------------------------------------------+
//| Return the number of decimal places                              |
//| depending on the swap calculation method                         |
//+------------------------------------------------------------------+
int CSymbol::SymbolDigitsBySwap(void)
  {
   return
     (
      this.SwapMode()==SYMBOL_SWAP_MODE_POINTS           ||
      this.SwapMode()==SYMBOL_SWAP_MODE_REOPEN_CURRENT   ||
      this.SwapMode()==SYMBOL_SWAP_MODE_REOPEN_BID       ?  this.Digits() :
      this.SwapMode()==SYMBOL_SWAP_MODE_CURRENCY_SYMBOL  ||
      this.SwapMode()==SYMBOL_SWAP_MODE_CURRENCY_MARGIN  ||
      this.SwapMode()==SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT ?  this.DigitsCurrency():
      this.SwapMode()==SYMBOL_SWAP_MODE_INTEREST_CURRENT ||
      this.SwapMode()==SYMBOL_SWAP_MODE_INTEREST_OPEN    ?  1  :  0
     );
  }
//+------------------------------------------------------------------+
```

Let's make the **Refresh()** method virtual since it is also defined in the base class of all **CBaseObj** objects. Also, replace the **RefreshRates()** quote data refresh method type from void to bool. Since the RefreshRates() method is to be called in the very beginning of the Refresh() method. If no data obtained in it, the method returns false. Accordingly, an exit from the Refresh() method is performed immediately.

**Add defining the method returning the symbol event description:**

```
//--- Update all symbol data
   virtual void      Refresh(void);
//--- Update quote data by a symbol
   bool              RefreshRates(void);
//--- Return description of symbol events
   string            EventDescription(const ENUM_SYMBOL_EVENT event);
```

In the section of simplified access methods of the class public section, in the methods returning symbol integer properties,

**add the method of returning a symbol index in the Market Watch window:**

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
//+------------------------------------------------------------------+
//--- Integer properties
   long              Status(void)                                 const { return this.GetProperty(SYMBOL_PROP_STATUS);                                      }
   int               IndexInMarketWatch(void)                     const { return (int)this.GetProperty(SYMBOL_PROP_INDEX_MW);                               }
   bool              IsCustom(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_CUSTOM);                                }
   color             ColorBackground(void)                        const { return (color)this.GetProperty(SYMBOL_PROP_BACKGROUND_COLOR);                     }
   ENUM_SYMBOL_CHART_MODE ChartMode(void)                         const { return (ENUM_SYMBOL_CHART_MODE)this.GetProperty(SYMBOL_PROP_CHART_MODE);          }
   bool              IsExist(void)                                const { return (bool)this.GetProperty(SYMBOL_PROP_EXIST);                                 }
   bool              IsExist(const string name)                   const { return this.SymbolExists(name);                                                   }
   bool              IsSelect(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_SELECT);                                }
   bool              IsVisible(void)                              const { return (bool)this.GetProperty(SYMBOL_PROP_VISIBLE);                               }
   long              SessionDeals(void)                           const { return this.GetProperty(SYMBOL_PROP_SESSION_DEALS);                               }
   long              SessionBuyOrders(void)                       const { return this.GetProperty(SYMBOL_PROP_SESSION_BUY_ORDERS);                          }
   long              SessionSellOrders(void)                      const { return this.GetProperty(SYMBOL_PROP_SESSION_SELL_ORDERS);                         }
   long              Volume(void)                                 const { return this.GetProperty(SYMBOL_PROP_VOLUME);                                      }
   long              VolumeHigh(void)                             const { return this.GetProperty(SYMBOL_PROP_VOLUMEHIGH);                                  }
   long              VolumeLow(void)                              const { return this.GetProperty(SYMBOL_PROP_VOLUMELOW);                                   }
   datetime          Time(void)                                   const { return (datetime)this.GetProperty(SYMBOL_PROP_TIME);                              }
   int               Digits(void)                                 const { return (int)this.GetProperty(SYMBOL_PROP_DIGITS);                                 }
   int               DigitsLot(void)                              const { return (int)this.GetProperty(SYMBOL_PROP_DIGITS_LOTS);                            }
   int               Spread(void)                                 const { return (int)this.GetProperty(SYMBOL_PROP_SPREAD);                                 }
   bool              IsSpreadFloat(void)                          const { return (bool)this.GetProperty(SYMBOL_PROP_SPREAD_FLOAT);                          }
   int               TicksBookdepth(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_TICKS_BOOKDEPTH);                        }
   ENUM_SYMBOL_CALC_MODE TradeCalcMode(void)                      const { return (ENUM_SYMBOL_CALC_MODE)this.GetProperty(SYMBOL_PROP_TRADE_CALC_MODE);      }
   ENUM_SYMBOL_TRADE_MODE TradeMode(void)                         const { return (ENUM_SYMBOL_TRADE_MODE)this.GetProperty(SYMBOL_PROP_TRADE_MODE);          }
   datetime          StartTime(void)                              const { return (datetime)this.GetProperty(SYMBOL_PROP_START_TIME);                        }
   datetime          ExpirationTime(void)                         const { return (datetime)this.GetProperty(SYMBOL_PROP_EXPIRATION_TIME);                   }
   int               TradeStopLevel(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_TRADE_STOPS_LEVEL);                      }
   int               TradeFreezeLevel(void)                       const { return (int)this.GetProperty(SYMBOL_PROP_TRADE_FREEZE_LEVEL);                     }
   ENUM_SYMBOL_TRADE_EXECUTION TradeExecutionMode(void)           const { return (ENUM_SYMBOL_TRADE_EXECUTION)this.GetProperty(SYMBOL_PROP_TRADE_EXEMODE);  }
   ENUM_SYMBOL_SWAP_MODE SwapMode(void)                           const { return (ENUM_SYMBOL_SWAP_MODE)this.GetProperty(SYMBOL_PROP_SWAP_MODE);            }
   ENUM_DAY_OF_WEEK  SwapRollover3Days(void)                      const { return (ENUM_DAY_OF_WEEK)this.GetProperty(SYMBOL_PROP_SWAP_ROLLOVER3DAYS);        }
   bool              IsMarginHedgedUseLeg(void)                   const { return (bool)this.GetProperty(SYMBOL_PROP_MARGIN_HEDGED_USE_LEG);                 }
   int               ExpirationModeFlags(void)                    const { return (int)this.GetProperty(SYMBOL_PROP_EXPIRATION_MODE);                        }
   int               FillingModeFlags(void)                       const { return (int)this.GetProperty(SYMBOL_PROP_FILLING_MODE);                           }
   int               OrderModeFlags(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_ORDER_MODE);                             }
   ENUM_SYMBOL_ORDER_GTC_MODE OrderModeGTC(void)                  const { return (ENUM_SYMBOL_ORDER_GTC_MODE)this.GetProperty(SYMBOL_PROP_ORDER_GTC_MODE);  }
   ENUM_SYMBOL_OPTION_MODE OptionMode(void)                       const { return (ENUM_SYMBOL_OPTION_MODE)this.GetProperty(SYMBOL_PROP_OPTION_MODE);        }
   ENUM_SYMBOL_OPTION_RIGHT OptionRight(void)                     const { return (ENUM_SYMBOL_OPTION_RIGHT)this.GetProperty(SYMBOL_PROP_OPTION_RIGHT);      }
//--- Real properties
```

While in the methods returning real properties,

**add the method of returning Bid or Last price, the method of returning the maximum Bid or Last price for a day and the method returning the minimum Bid or Last price for a day:**

```
//--- Real properties
   double            Bid(void)                                    const { return this.GetProperty(SYMBOL_PROP_BID);                                         }
   double            BidHigh(void)                                const { return this.GetProperty(SYMBOL_PROP_BIDHIGH);                                     }
   double            BidLow(void)                                 const { return this.GetProperty(SYMBOL_PROP_BIDLOW);                                      }
   double            Ask(void)                                    const { return this.GetProperty(SYMBOL_PROP_ASK);                                         }
   double            AskHigh(void)                                const { return this.GetProperty(SYMBOL_PROP_ASKHIGH);                                     }
   double            AskLow(void)                                 const { return this.GetProperty(SYMBOL_PROP_ASKLOW);                                      }
   double            Last(void)                                   const { return this.GetProperty(SYMBOL_PROP_LAST);                                        }
   double            LastHigh(void)                               const { return this.GetProperty(SYMBOL_PROP_LASTHIGH);                                    }
   double            LastLow(void)                                const { return this.GetProperty(SYMBOL_PROP_LASTLOW);                                     }
   double            VolumeReal(void)                             const { return this.GetProperty(SYMBOL_PROP_VOLUME_REAL);                                 }
   double            VolumeHighReal(void)                         const { return this.GetProperty(SYMBOL_PROP_VOLUMEHIGH_REAL);                             }
   double            VolumeLowReal(void)                          const { return this.GetProperty(SYMBOL_PROP_VOLUMELOW_REAL);                              }
   double            OptionStrike(void)                           const { return this.GetProperty(SYMBOL_PROP_OPTION_STRIKE);                               }
   double            Point(void)                                  const { return this.GetProperty(SYMBOL_PROP_POINT);                                       }
   double            TradeTickValue(void)                         const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE);                            }
   double            TradeTickValueProfit(void)                   const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT);                     }
   double            TradeTickValueLoss(void)                     const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS);                       }
   double            TradeTickSize(void)                          const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_SIZE);                             }
   double            TradeContractSize(void)                      const { return this.GetProperty(SYMBOL_PROP_TRADE_CONTRACT_SIZE);                         }
   double            TradeAccuredInterest(void)                   const { return this.GetProperty(SYMBOL_PROP_TRADE_ACCRUED_INTEREST);                      }
   double            TradeFaceValue(void)                         const { return this.GetProperty(SYMBOL_PROP_TRADE_FACE_VALUE);                            }
   double            TradeLiquidityRate(void)                     const { return this.GetProperty(SYMBOL_PROP_TRADE_LIQUIDITY_RATE);                        }
   double            LotsMin(void)                                const { return this.GetProperty(SYMBOL_PROP_VOLUME_MIN);                                  }
   double            LotsMax(void)                                const { return this.GetProperty(SYMBOL_PROP_VOLUME_MAX);                                  }
   double            LotsStep(void)                               const { return this.GetProperty(SYMBOL_PROP_VOLUME_STEP);                                 }
   double            VolumeLimit(void)                            const { return this.GetProperty(SYMBOL_PROP_VOLUME_LIMIT);                                }
   double            SwapLong(void)                               const { return this.GetProperty(SYMBOL_PROP_SWAP_LONG);                                   }
   double            SwapShort(void)                              const { return this.GetProperty(SYMBOL_PROP_SWAP_SHORT);                                  }
   double            MarginInitial(void)                          const { return this.GetProperty(SYMBOL_PROP_MARGIN_INITIAL);                              }
   double            MarginMaintenance(void)                      const { return this.GetProperty(SYMBOL_PROP_MARGIN_MAINTENANCE);                          }
   double            MarginLongInitial(void)                      const { return this.GetProperty(SYMBOL_PROP_MARGIN_LONG_INITIAL);                         }
   double            MarginBuyStopInitial(void)                   const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL);                     }
   double            MarginBuyLimitInitial(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL);                    }
   double            MarginBuyStopLimitInitial(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL);                }
   double            MarginLongMaintenance(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE);                     }
   double            MarginBuyStopMaintenance(void)               const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE);                 }
   double            MarginBuyLimitMaintenance(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE);                }
   double            MarginBuyStopLimitMaintenance(void)          const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE);            }
   double            MarginShortInitial(void)                     const { return this.GetProperty(SYMBOL_PROP_MARGIN_SHORT_INITIAL);                        }
   double            MarginSellStopInitial(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL);                    }
   double            MarginSellLimitInitial(void)                 const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL);                   }
   double            MarginSellStopLimitInitial(void)             const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL);               }
   double            MarginShortMaintenance(void)                 const { return this.GetProperty(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE);                    }
   double            MarginSellStopMaintenance(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE);                }
   double            MarginSellLimitMaintenance(void)             const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE);               }
   double            MarginSellStopLimitMaintenance(void)         const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE);           }
   double            SessionVolume(void)                          const { return this.GetProperty(SYMBOL_PROP_SESSION_VOLUME);                              }
   double            SessionTurnover(void)                        const { return this.GetProperty(SYMBOL_PROP_SESSION_TURNOVER);                            }
   double            SessionInterest(void)                        const { return this.GetProperty(SYMBOL_PROP_SESSION_INTEREST);                            }
   double            SessionBuyOrdersVolume(void)                 const { return this.GetProperty(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);                   }
   double            SessionSellOrdersVolume(void)                const { return this.GetProperty(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);                  }
   double            SessionOpen(void)                            const { return this.GetProperty(SYMBOL_PROP_SESSION_OPEN);                                }
   double            SessionClose(void)                           const { return this.GetProperty(SYMBOL_PROP_SESSION_CLOSE);                               }
   double            SessionAW(void)                              const { return this.GetProperty(SYMBOL_PROP_SESSION_AW);                                  }
   double            SessionPriceSettlement(void)                 const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT);                    }
   double            SessionPriceLimitMin(void)                   const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN);                     }
   double            SessionPriceLimitMax(void)                   const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX);                     }
   double            MarginHedged(void)                           const { return this.GetProperty(SYMBOL_PROP_MARGIN_HEDGED);                               }
   double            NormalizedPrice(const double price)          const;
   double            BidLast(void)                                const;
   double            BidLastHigh(void)                            const;
   double            BidLastLow(void)                             const;
//--- String properties
```

**Let's write their implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Return Bid or Last price                                         |
//| depending on the chart construction method                       |
//+------------------------------------------------------------------+
double CSymbol::BidLast(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetProperty(SYMBOL_PROP_BID) : this.GetProperty(SYMBOL_PROP_LAST));
  }
//+------------------------------------------------------------------+
//| Return maximum Bid or Last price for a day                       |
//| depending on the chart construction method                       |
//+------------------------------------------------------------------+
double CSymbol::BidLastHigh(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetProperty(SYMBOL_PROP_BIDHIGH) : this.GetProperty(SYMBOL_PROP_LASTHIGH));
  }
//+------------------------------------------------------------------+
//| Return minimum Bid or Last price for a day                       |
//| depending on the chart construction method                       |
//+------------------------------------------------------------------+
double CSymbol::BidLastLow(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetProperty(SYMBOL_PROP_BIDLOW) : this.GetProperty(SYMBOL_PROP_LASTLOW));
  }
//+------------------------------------------------------------------+
```

Here all is simple: an appropriate symbol property (with Bid or Last prices) is returned depending on prices used to construct the price chart.

Also, **write the methods of setting controlled properties and returning property change values and symbol event flags in the public section:**

```
//+------------------------------------------------------------------+
//| Get and set the parameters of tracked changes                    |
//+------------------------------------------------------------------+
   //--- Execution
   //--- Flag of changing the trading mode for a symbol
   bool              IsChangedTradeMode(void)                              const { return this.m_is_change_trade_mode;                       }
   //--- Current session deals
   //--- setting the controlled value of (1) growth, (2) decrease in the number of deals during the current session
   //--- getting (3) the number of deals change value during the current session,
   //--- getting the flag of the number of deals change during the current session exceeding the (4) growth, (5) decrease value
   void              SetControlSessionDealsInc(const long value)                 { this.m_control_session_deals_inc=::fabs(value);           }
   void              SetControlSessionDealsDec(const long value)                 { this.m_control_session_deals_dec=::fabs(value);           }
   long              GetValueChangedSessionDeals(void)                     const { return this.m_changed_session_deals_value;                }
   bool              IsIncreaseSessionDeals(void)                          const { return this.m_is_change_session_deals_inc;                }
   bool              IsDecreaseSessionDeals(void)                          const { return this.m_is_change_session_deals_dec;                }
   //--- Buy orders of the current session
   //--- setting the controlled value of (1) growth, (2) decrease in the current number of Buy orders
   //--- getting (3) the current number of Buy orders change value,
   //--- getting the flag of the current Buy orders' number change exceeding the (4) growth, (5) decrease value
   void              SetControlSessionBuyOrdInc(const long value)                { this.m_control_session_buy_ord_inc=::fabs(value);         }
   void              SetControlSessionBuyOrdDec(const long value)                { this.m_control_session_buy_ord_dec=::fabs(value);         }
   long              GetValueChangedSessionBuyOrders(void)                 const { return this.m_changed_session_buy_ord_value;              }
   bool              IsIncreaseSessionBuyOrders(void)                      const { return this.m_is_change_session_buy_ord_inc;              }
   bool              IsDecreaseSessionBuyOrders(void)                      const { return this.m_is_change_session_buy_ord_dec;              }
   //--- Sell orders of the current session
   //--- setting the controlled value of (1) growth, (2) decrease in the current number of Sell orders
   //--- getting (3) the current number of Sell orders change value,
   //--- getting the flag of the current Sell orders' number change exceeding the (4) growth, (5) decrease value
   void              SetControlSessionSellOrdInc(const long value)               { this.m_control_session_sell_ord_inc=::fabs(value);        }
   void              SetControlSessionSellOrdDec(const long value)               { this.m_control_session_sell_ord_dec=::fabs(value);        }
   long              GetValueChangedSessionSellOrders(void)                const { return this.m_changed_session_sell_ord_value;             }
   bool              IsIncreaseSessionSellOrders(void)                     const { return this.m_is_change_session_sell_ord_inc;             }
   bool              IsDecreaseSessionSellOrders(void)                     const { return this.m_is_change_session_sell_ord_dec;             }
   //--- Volume of the last deal
   //--- setting the last deal volume controlled (1) growth, (2) decrease value
   //--- getting (3) volume change values in the last deal,
   //--- getting the flag of the volume change in the last deal exceeding the (4) growth, (5) decrease value
   void              SetControlVolumeInc(const long value)                       { this.m_control_volume_inc=::fabs(value);                  }
   void              SetControlVolumeDec(const long value)                       { this.m_control_volume_dec=::fabs(value);                  }
   long              GetValueChangedVolume(void)                           const { return this.m_changed_volume_value;                       }
   bool              IsIncreaseVolume(void)                                const { return this.m_is_change_volume_inc;                       }
   bool              IsDecreaseVolume(void)                                const { return this.m_is_change_volume_dec;                       }
   //--- Maximum volume within a day
   //--- setting the maximum day volume controlled (1) growth, (2) decrease value
   //--- getting (3) the maximum volume change value within a day,
   //--- getting the flag of the maximum day volume change exceeding the (4) growth, (5) decrease value
   void              SetControlVolumeHighInc(const long value)                   { this.m_control_volume_high_day_inc=::fabs(value);         }
   void              SetControlVolumeHighDec(const long value)                   { this.m_control_volume_high_day_dec=::fabs(value);         }
   long              GetValueChangedVolumeHigh(void)                       const { return this.m_changed_volume_high_day_value;              }
   bool              IsIncreaseVolumeHigh(void)                            const { return this.m_is_change_volume_high_day_inc;              }
   bool              IsDecreaseVolumeHigh(void)                            const { return this.m_is_change_volume_high_day_dec;              }
   //--- Minimum volume within a day
   //--- setting the minimum day volume controlled (1) growth, (2) decrease value
   //--- getting (3) the minimum volume change value within a day,
   //--- getting the flag of the minimum day volume change exceeding the (4) growth, (5) decrease value
   void              SetControlVolumeLowInc(const long value)                    { this.m_control_volume_low_day_inc=::fabs(value);          }
   void              SetControlVolumeLowDec(const long value)                    { this.m_control_volume_low_day_dec=::fabs(value);          }
   long              GetValueChangedVolumeLow(void)                        const { return this.m_changed_volume_low_day_value;               }
   bool              IsIncreaseVolumeLow(void)                             const { return this.m_is_change_volume_low_day_inc;               }
   bool              IsDecreaseVolumeLow(void)                             const { return this.m_is_change_volume_low_day_dec;               }
   //--- Spread
   //--- setting the controlled spread decrease (1) growth, (2) decrease value in points
   //--- getting (3) spread change value in points,
   //--- getting the flag of the spread change in points exceeding the (4) growth, (5) decrease value
   void              SetControlSpreadInc(const int value)                        { this.m_control_spread_inc=::fabs(value);                  }
   void              SetControlSpreadDec(const int value)                        { this.m_control_spread_dec=::fabs(value);                  }
   int               GetValueChangedSpread(void)                           const { return this.m_changed_spread_value;                       }
   bool              IsIncreaseSpread(void)                                const { return this.m_is_change_spread_inc;                       }
   bool              IsDecreaseSpread(void)                                const { return this.m_is_change_spread_dec;                       }
   //--- StopLevel
   //--- setting the controlled StopLevel decrease (1) growth, (2) decrease value in points
   //--- getting (3) StopLevel change value in points,
   //--- getting the flag of StopLevel change in points exceeding the (4) growth, (5) decrease value
   void              SetControlStopLevelInc(const int value)                     { this.m_control_stops_level_inc=::fabs(value);             }
   void              SetControlStopLevelDec(const int value)                     { this.m_control_stops_level_dec=::fabs(value);             }
   int               GetValueChangedStopLevel(void)                        const { return this.m_changed_stops_level_value;                  }
   bool              IsIncreaseStopLevel(void)                             const { return this.m_is_change_stops_level_inc;                  }
   bool              IsDecreaseStopLevel(void)                             const { return this.m_is_change_stops_level_dec;                  }
   //--- Freeze distance
   //--- setting the controlled FreezeLevel decrease (1) growth, (2) decrease value in points
   //--- getting (3) FreezeLevel change value in points,
   //--- getting the flag of FreezeLevel change in points exceeding the (4) growth, (5) decrease value
   void              SetControlFreezeLevelInc(const int value)                   { this.m_control_freeze_level_inc=::fabs(value);            }
   void              SetControlFreezeLevelDec(const int value)                   { this.m_control_freeze_level_dec=::fabs(value);            }
   int               GetValueChangedFreezeLevel(void)                      const { return this.m_changed_freeze_level_value;                 }
   bool              IsIncreaseFreezeLevel(void)                           const { return this.m_is_change_freeze_level_inc;                 }
   bool              IsDecreaseFreezeLevel(void)                           const { return this.m_is_change_freeze_level_dec;                 }

   //--- Bid/Last
   //--- setting the Bid or Last price controlled (1) growth, (2) decrease value
   //--- getting (3) Bid or Last price change value,
   //--- getting the flag of the Bid or Last price change exceeding the (4) growth, (5) decrease value
   void              SetControlBidLastInc(const double value)                    { this.m_control_bid_last_inc=::fabs(value);                }
   void              SetControlBidLastDec(const double value)                    { this.m_control_bid_last_dec=::fabs(value);                }
   double            GetValueChangedBidLast(void)                          const { return this.m_changed_bid_last_value;                     }
   bool              IsIncreaseBidLast(void)                               const { return this.m_is_change_bid_last_inc;                     }
   bool              IsDecreaseBidLast(void)                               const { return this.m_is_change_bid_last_dec;                     }
   //--- Maximum Bid/Last of the day
   //--- setting the maximum Bid or Last price controlled (1) growth, (2) decrease value
   //--- getting the (3) maximum Bid or Last price change value,
   //--- getting the flag of the maximum Bid or Last price change exceeding the (4) growth, (5) decrease value
   void              SetControlBidLastHighInc(const double value)                { this.m_control_bid_last_high_inc=::fabs(value);           }
   void              SetControlBidLastHighDec(const double value)                { this.m_control_bid_last_high_dec=::fabs(value);           }
   double            GetValueChangedBidLastHigh(void)                      const { return this.m_changed_bid_last_high_value;                }
   bool              IsIncreaseBidLastHigh(void)                           const { return this.m_is_change_bid_last_high_inc;                }
   bool              IsDecreaseBidLastHigh(void)                           const { return this.m_is_change_bid_last_high_dec;                }
   //--- Minimum Bid/Last of the day
   //--- setting the minimum Bid or Last price controlled (1) growth, (2) decrease value
   //--- getting the (3) minimum Bid or Last price change value,
   //--- getting the flag of the minimum Bid or Last price change exceeding the (4) growth, (5) decrease value
   void              SetControlBidLastLowInc(const double value)                 { this.m_control_bid_last_low_inc=::fabs(value);            }
   void              SetControlBidLastLowDec(const double value)                 { this.m_control_bid_last_low_dec=::fabs(value);            }
   double            GetValueChangedBidLastLow(void)                       const { return this.m_changed_bid_last_low_value;                 }
   bool              IsIncreaseBidLastLow(void)                            const { return this.m_is_change_bid_last_low_inc;                 }
   bool              IsDecreaseBidLastLow(void)                            const { return this.m_is_change_bid_last_low_dec;                 }
   //--- Ask
   //--- setting the Ask price controlled (1) growth, (2) decrease value
   //--- getting (3) Ask price change value,
   //--- getting the flag of the Ask price change exceeding the (4) growth, (5) decrease value
   void              SetControlAskInc(const double value)                        { this.m_control_ask_inc=::fabs(value);                     }
   void              SetControlAskDec(const double value)                        { this.m_control_ask_dec=::fabs(value);                     }
   double            GetValueChangedAsk(void)                              const { return this.m_changed_ask_value;                          }
   bool              IsIncreaseAsk(void)                                   const { return this.m_is_change_ask_inc;                          }
   bool              IsDecreaseAsk(void)                                   const { return this.m_is_change_ask_dec;                          }
   //--- Maximum Ask price for the day
   //--- setting the maximum day Ask controlled (1) growth, (2) decrease value
   //--- getting (3) the maximum Ask change value within a day,
   //--- getting the flag of the maximum day Ask change exceeding the (4) growth, (5) decrease value
   void              SetControlAskHighInc(const double value)                    { this.m_control_ask_high_inc=::fabs(value);                }
   void              SetControlAskHighDec(const double value)                    { this.m_control_ask_high_dec=::fabs(value);                }
   double            GetValueChangedAskHigh(void)                          const { return this.m_changed_ask_high_value;                     }
   bool              IsIncreaseAskHigh(void)                               const { return this.m_is_change_ask_high_inc;                     }
   bool              IsDecreaseAskHigh(void)                               const { return this.m_is_change_ask_high_dec;                     }
   //--- Minimum Ask price for the day
   //--- setting the minimum day Ask controlled (1) growth, (2) decrease value
   //--- getting (3) the minimum Ask change value within a day,
   //--- getting the flag of the minimum day Ask change exceeding the (4) growth, (5) decrease value
   void              SetControlAskLowInc(const double value)                     { this.m_control_ask_low_inc=::fabs(value);                 }
   void              SetControlAskLowDec(const double value)                     { this.m_control_ask_low_dec=::fabs(value);                 }
   double            GetValueChangedAskLow(void)                           const { return this.m_changed_ask_low_value;                      }
   bool              IsIncreaseAskLow(void)                                const { return this.m_is_change_ask_low_inc;                      }
   bool              IsDecreaseAskLow(void)                                const { return this.m_is_change_ask_low_dec;                      }
   //--- Real Volume for the day
   //--- setting the real day volume controlled (1) growth, (2) decrease value
   //--- getting (3) the change value of the real day volume,
   //--- getting the flag of the real day volume change exceeding the (4) growth, (5) decrease value
   void              SetControlVolumeRealInc(const double value)                 { this.m_control_volume_real_inc=::fabs(value);             }
   void              SetControlVolumeRealDec(const double value)                 { this.m_control_volume_real_dec=::fabs(value);             }
   double            GetValueChangedVolumeReal(void)                       const { return this.m_changed_volume_real_value;                  }
   bool              IsIncreaseVolumeReal(void)                            const { return this.m_is_change_volume_real_inc;                  }
   bool              IsDecreaseVolumeReal(void)                            const { return this.m_is_change_volume_real_dec;                  }
   //--- Maximum real volume for the day
   //--- setting the maximum real day volume controlled (1) growth, (2) decrease value
   //--- getting (3) the change value of the maximum real day volume,
   //--- getting the flag of the maximum real day volume change exceeding the (4) growth, (5) decrease value
   void              SetControlVolumeHighRealInc(const double value)             { this.m_control_volume_high_real_day_inc=::fabs(value);    }
   void              SetControlVolumeHighRealDec(const double value)             { this.m_control_volume_high_real_day_dec=::fabs(value);    }
   double            GetValueChangedVolumeHighReal(void)                   const { return this.m_changed_volume_high_real_day_value;         }
   bool              IsIncreaseVolumeHighReal(void)                        const { return this.m_is_change_volume_high_real_day_inc;         }
   bool              IsDecreaseVolumeHighReal(void)                        const { return this.m_is_change_volume_high_real_day_dec;         }
   //--- Minimum real volume for the day
   //--- setting the minimum real day volume controlled (1) growth, (2) decrease value
   //--- getting (3) the change value of the minimum real day volume,
   //--- getting the flag of the minimum real day volume change exceeding the (4) growth, (5) decrease value
   void              SetControlVolumeLowRealInc(const double value)              { this.m_control_volume_low_real_day_inc=::fabs(value);     }
   void              SetControlVolumeLowRealDec(const double value)              { this.m_control_volume_low_real_day_dec=::fabs(value);     }
   double            GetValueChangedVolumeLowReal(void)                    const { return this.m_changed_volume_low_real_day_value;          }
   bool              IsIncreaseVolumeLowReal(void)                         const { return this.m_is_change_volume_low_real_day_inc;          }
   bool              IsDecreaseVolumeLowReal(void)                         const { return this.m_is_change_volume_low_real_day_dec;          }
   //--- Strike price
   //--- setting the strike price controlled (1) growth, (2) decrease value
   //--- getting (3) the change value of the strike price,
   //--- getting the flag of the strike price change exceeding the (4) growth, (5) decrease value
   void              SetControlOptionStrikeInc(const double value)               { this.m_control_option_strike_inc=::fabs(value);           }
   void              SetControlOptionStrikeDec(const double value)               { this.m_control_option_strike_dec=::fabs(value);           }
   double            GetValueChangedOptionStrike(void)                     const { return this.m_changed_option_strike_value;                }
   bool              IsIncreaseOptionStrike(void)                          const { return this.m_is_change_option_strike_inc;                }
   bool              IsDecreaseOptionStrike(void)                          const { return this.m_is_change_option_strike_dec;                }
   //--- Maximum allowed total volume of unidirectional positions and orders
   //--- (1) getting the change value of the maximum allowed total volume of unidirectional positions and orders,
   //--- getting the flag of (2) increasing, (3) decreasing the maximum allowed total volume of unidirectional positions and orders
   double            GetValueChangedVolumeLimit(void)                      const { return this.m_changed_volume_limit_value;                 }
   bool              IsIncreaseVolumeLimit(void)                           const { return this.m_is_change_volume_limit_inc;                 }
   bool              IsDecreaseVolumeLimit(void)                           const { return this.m_is_change_volume_limit_dec;                 }
   //---  Swap long
   //--- (1) getting the swap long change value,
   //--- getting the flag of (2) increasing, (3) decreasing the swap long
   double            GetValueChangedSwapLong(void)                         const { return this.m_changed_swap_long_value;                    }
   bool              IsIncreaseSwapLong(void)                              const { return this.m_is_change_swap_long_inc;                    }
   bool              IsDecreaseSwapLong(void)                              const { return this.m_is_change_swap_long_dec;                    }
   //---  Swap short
   //--- (1) getting the swap short change value,
   //--- getting the flag of (2) increasing, (3) decreasing the swap short
   double            GetValueChangedSwapShort(void)                        const { return this.m_changed_swap_short_value;                   }
   bool              IsIncreaseSwapShort(void)                             const { return this.m_is_change_swap_short_inc;                   }
   bool              IsDecreaseSwapShort(void)                             const { return this.m_is_change_swap_short_dec;                   }
   //--- The total volume of deals in the current session
   //--- setting the controlled value of (1) growth, (2) decrease in the total volume of deals during the current session
   //--- getting (3) the total deal volume change value in the current session,
   //--- getting the flag of the total deal volume change during the current session exceeding the (4) growth, (5) decrease value
   void              SetControlSessionVolumeInc(const double value)              { this.m_control_session_volume_inc=::fabs(value);          }
   void              SetControlSessionVolumeDec(const double value)              { this.m_control_session_volume_dec=::fabs(value);          }
   double            GetValueChangedSessionVolume(void)                    const { return this.m_changed_session_volume_value;               }
   bool              IsIncreaseSessionVolume(void)                         const { return this.m_is_change_session_volume_inc;               }
   bool              IsDecreaseSessionVolume(void)                         const { return this.m_is_change_session_volume_dec;               }
   //--- The total turnover in the current session
   //--- setting the controlled value of the turnover (1) growth, (2) decrease during the current session
   //--- getting (3) the total turnover change value in the current session,
   //--- getting the flag of the total turnover change during the current session exceeding the (4) growth, (5) decrease value
   void              SetControlSessionTurnoverInc(const double value)            { this.m_control_session_turnover_inc=::fabs(value);        }
   void              SetControlSessionTurnoverDec(const double value)            { this.m_control_session_turnover_dec=::fabs(value);        }
   double            GetValueChangedSessionTurnover(void)                  const { return this.m_changed_session_turnover_value;             }
   bool              IsIncreaseSessionTurnover(void)                       const { return this.m_is_change_session_turnover_inc;             }
   bool              IsDecreaseSessionTurnover(void)                       const { return this.m_is_change_session_turnover_dec;             }
   //--- The total volume of open positions
   //--- setting the controlled value of (1) growth, (2) decrease in the total volume of open positions during the current session
   //--- getting (3) the change value of the open positions total volume in the current session,
   //--- getting the flag of the open positions total volume change during the current session exceeding the (4) growth, (5) decrease value
   void              SetControlSessionInterestInc(const double value)            { this.m_control_session_interest_inc=::fabs(value);        }
   void              SetControlSessionInterestDec(const double value)            { this.m_control_session_interest_dec=::fabs(value);        }
   double            GetValueChangedSessionInterest(void)                  const { return this.m_changed_session_interest_value;             }
   bool              IsIncreaseSessionInterest(void)                       const { return this.m_is_change_session_interest_inc;             }
   bool              IsDecreaseSessionInterest(void)                       const { return this.m_is_change_session_interest_dec;             }
   //--- The total volume of Buy orders at the moment
   //--- setting the controlled value of (1) growth, (2) decrease in the current total buy order volume
   //--- getting (3) the change value of the current total buy order volume,
   //--- getting the flag of the current total buy orders' volume change exceeding the (4) growth, (5) decrease value
   void              SetControlSessionBuyOrdVolumeInc(const double value)        { this.m_control_session_buy_ord_volume_inc=::fabs(value);  }
   void              SetControlSessionBuyOrdVolumeDec(const double value)        { this.m_control_session_buy_ord_volume_dec=::fabs(value);  }
   double            GetValueChangedSessionBuyOrdVolume(void)              const { return this.m_changed_session_buy_ord_volume_value;       }
   bool              IsIncreaseSessionBuyOrdVolume(void)                   const { return this.m_is_change_session_buy_ord_volume_inc;       }
   bool              IsDecreaseSessionBuyOrdVolume(void)                   const { return this.m_is_change_session_buy_ord_volume_dec;       }
   //--- The total volume of Sell orders at the moment
   //--- setting the controlled value of (1) growth, (2) decrease in the current total sell order volume
   //--- getting (3) the change value of the current total sell order volume,
   //--- getting the flag of the current total sell orders' volume change exceeding the (4) growth, (5) decrease value
   void              SetControlSessionSellOrdVolumeInc(const double value)       { this.m_control_session_sell_ord_volume_inc=::fabs(value); }
   void              SetControlSessionSellOrdVolumeDec(const double value)       { this.m_control_session_sell_ord_volume_dec=::fabs(value); }
   double            GetValueChangedSessionSellOrdVolume(void)             const { return this.m_changed_session_sell_ord_volume_value;      }
   bool              IsIncreaseSessionSellOrdVolume(void)                  const { return this.m_is_change_session_sell_ord_volume_inc;      }
   bool              IsDecreaseSessionSellOrdVolume(void)                  const { return this.m_is_change_session_sell_ord_volume_dec;      }
   //--- Session open price
   //--- setting the session open price controlled (1) growth, (2) decrease value
   //--- getting (3) the change value of the session open price,
   //--- getting the flag of the session open price change exceeding the (4) growth, (5) decrease value
   void              SetControlSessionPriceOpenInc(const double value)           { this.m_control_session_open_inc=::fabs(value);            }
   void              SetControlSessionPriceOpenDec(const double value)           { this.m_control_session_open_dec=::fabs(value);            }
   double            GetValueChangedSessionPriceOpen(void)                 const { return this.m_changed_session_open_value;                 }
   bool              IsIncreaseSessionPriceOpen(void)                      const { return this.m_is_change_session_open_inc;                 }
   bool              IsDecreaseSessionPriceOpen(void)                      const { return this.m_is_change_session_open_dec;                 }
   //--- Session close price
   //--- setting the session close price controlled (1) growth, (2) decrease value
   //--- getting (3) the change value of the session close price,
   //--- getting the flag of the session close price change exceeding the (4) growth, (5) decrease value
   void              SetControlSessionPriceCloseInc(const double value)          { this.m_control_session_close_inc=::fabs(value);           }
   void              SetControlSessionPriceCloseDec(const double value)          { this.m_control_session_close_dec=::fabs(value);           }
   double            GetValueChangedSessionPriceClose(void)                const { return this.m_changed_session_close_value;                }
   bool              IsIncreaseSessionPriceClose(void)                     const { return this.m_is_change_session_close_inc;                }
   bool              IsDecreaseSessionPriceClose(void)                     const { return this.m_is_change_session_close_dec;                }
   //--- The average weighted session price
   //--- setting the average weighted session price controlled (1) growth, (2) decrease value
   //--- getting (3) the change value of the average weighted session price,
   //--- getting the flag of the average weighted session price change exceeding the (4) growth, (5) decrease value
   void              SetControlSessionPriceAWInc(const double value)             { this.m_control_session_aw_inc=::fabs(value);              }
   void              SetControlSessionPriceAWDec(const double value)             { this.m_control_session_aw_dec=::fabs(value);              }
   double            GetValueChangedSessionPriceAW(void)                   const { return this.m_changed_session_aw_value;                   }
   bool              IsIncreaseSessionPriceAW(void)                        const { return this.m_is_change_session_aw_inc;                   }
   bool              IsDecreaseSessionPriceAW(void)                        const { return this.m_is_change_session_aw_dec;                   }
//---
```

Here, the methods for setting a controlled property change value are provided for each of the controlled properties. When the value is exceeded, an event is formed. The event flag can be received using the method returning the event flag, while the change value can be obtained using the appropriate method as well. We have discussed the similar methods when implementing the account events tracking [in the part 13 of the library description](https://www.mql5.com/en/articles/6995).

Some changes have been made in the class constructor:

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CSymbol::CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name,const int index)
  {
   this.m_name=name;
   if(!this.Exist())
     {
      ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\"",": ",TextByLanguage("Ошибка. Такого символа нет на сервере","Error. There is no such symbol on the server"));
      this.m_global_error=ERR_MARKET_UNKNOWN_SYMBOL;
     }
   bool select=::SymbolInfoInteger(this.m_name,SYMBOL_SELECT);
   ::ResetLastError();
   if(!select)
     {
      if(!this.SetToMarketWatch())
        {
         this.m_global_error=::GetLastError();
         ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\": ",TextByLanguage("Не удалось поместить в обзор рынка. Ошибка: ","Failed to put in the market watch. Error: "),this.m_global_error);
        }
     }
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\": ",TextByLanguage("Не удалось получить текущие цены. Ошибка: ","Could not get current prices. Error: "),this.m_global_error);
     }
//--- Initialize data
   this.Reset();
   this.InitMarginRates();
   ::ZeroMemory(this.m_struct_prev_symbol);
   this.m_struct_prev_symbol.trade_mode=WRONG_VALUE;
   this.InitChangesParams();
   this.InitControlsParams();
#ifdef __MQL5__
   ::ResetLastError();
   if(!this.MarginRates())
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,this.Name(),": ",TextByLanguage("Не удалось получить коэффициенты взимания маржи. Ошибка: ","Failed to get margin rates. Error: "),this.m_global_error);
      return;
     }
#endif

//--- Save integer properties
   this.m_long_prop[SYMBOL_PROP_STATUS]                                             = symbol_status;
   this.m_long_prop[SYMBOL_PROP_INDEX_MW]                                           = index;
   this.m_long_prop[SYMBOL_PROP_VOLUME]                                             = (long)this.m_tick.volume;
   this.m_long_prop[SYMBOL_PROP_SELECT]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_SELECT);
   this.m_long_prop[SYMBOL_PROP_VISIBLE]                                            = ::SymbolInfoInteger(this.m_name,SYMBOL_VISIBLE);
   this.m_long_prop[SYMBOL_PROP_SESSION_DEALS]                                      = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_DEALS);
   this.m_long_prop[SYMBOL_PROP_SESSION_BUY_ORDERS]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_BUY_ORDERS);
   this.m_long_prop[SYMBOL_PROP_SESSION_SELL_ORDERS]                                = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_SELL_ORDERS);
   this.m_long_prop[SYMBOL_PROP_VOLUMEHIGH]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_VOLUMEHIGH);
   this.m_long_prop[SYMBOL_PROP_VOLUMELOW]                                          = ::SymbolInfoInteger(this.m_name,SYMBOL_VOLUMELOW);
   this.m_long_prop[SYMBOL_PROP_DIGITS]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_DIGITS);
   this.m_long_prop[SYMBOL_PROP_SPREAD]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_SPREAD);
   this.m_long_prop[SYMBOL_PROP_SPREAD_FLOAT]                                       = ::SymbolInfoInteger(this.m_name,SYMBOL_SPREAD_FLOAT);
   this.m_long_prop[SYMBOL_PROP_TICKS_BOOKDEPTH]                                    = ::SymbolInfoInteger(this.m_name,SYMBOL_TICKS_BOOKDEPTH);
   this.m_long_prop[SYMBOL_PROP_TRADE_MODE]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_MODE);
   this.m_long_prop[SYMBOL_PROP_START_TIME]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_START_TIME);
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_TIME]                                    = ::SymbolInfoInteger(this.m_name,SYMBOL_EXPIRATION_TIME);
   this.m_long_prop[SYMBOL_PROP_TRADE_STOPS_LEVEL]                                  = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_STOPS_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_FREEZE_LEVEL]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_FREEZE_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_EXEMODE]                                      = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_EXEMODE);
   this.m_long_prop[SYMBOL_PROP_SWAP_ROLLOVER3DAYS]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_SWAP_ROLLOVER3DAYS);
   this.m_long_prop[SYMBOL_PROP_TIME]                                               = this.TickTime();
   this.m_long_prop[SYMBOL_PROP_EXIST]                                              = this.SymbolExists();
   this.m_long_prop[SYMBOL_PROP_CUSTOM]                                             = this.SymbolCustom();
   this.m_long_prop[SYMBOL_PROP_MARGIN_HEDGED_USE_LEG]                              = this.SymbolMarginHedgedUseLEG();
   this.m_long_prop[SYMBOL_PROP_ORDER_MODE]                                         = this.SymbolOrderMode();
   this.m_long_prop[SYMBOL_PROP_FILLING_MODE]                                       = this.SymbolOrderFillingMode();
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_MODE]                                    = this.SymbolExpirationMode();
   this.m_long_prop[SYMBOL_PROP_ORDER_GTC_MODE]                                     = this.SymbolOrderGTCMode();
   this.m_long_prop[SYMBOL_PROP_OPTION_MODE]                                        = this.SymbolOptionMode();
   this.m_long_prop[SYMBOL_PROP_OPTION_RIGHT]                                       = this.SymbolOptionRight();
   this.m_long_prop[SYMBOL_PROP_BACKGROUND_COLOR]                                   = this.SymbolBackgroundColor();
   this.m_long_prop[SYMBOL_PROP_CHART_MODE]                                         = this.SymbolChartMode();
   this.m_long_prop[SYMBOL_PROP_TRADE_CALC_MODE]                                    = this.SymbolCalcMode();
   this.m_long_prop[SYMBOL_PROP_SWAP_MODE]                                          = this.SymbolSwapMode();
//--- Save real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKHIGH)]                          = ::SymbolInfoDouble(this.m_name,SYMBOL_ASKHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKLOW)]                           = ::SymbolInfoDouble(this.m_name,SYMBOL_ASKLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTHIGH)]                         = ::SymbolInfoDouble(this.m_name,SYMBOL_LASTHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTLOW)]                          = ::SymbolInfoDouble(this.m_name,SYMBOL_LASTLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_POINT)]                            = ::SymbolInfoDouble(this.m_name,SYMBOL_POINT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE_PROFIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS)]            = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE_LOSS);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_SIZE)]                  = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_CONTRACT_SIZE)]              = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_CONTRACT_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MIN)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MAX)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_STEP)]                      = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_STEP);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_LIMIT)]                     = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_LIMIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_LONG)]                        = ::SymbolInfoDouble(this.m_name,SYMBOL_SWAP_LONG);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_SHORT)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_SWAP_SHORT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_INITIAL)]                   = ::SymbolInfoDouble(this.m_name,SYMBOL_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_MAINTENANCE)]               = ::SymbolInfoDouble(this.m_name,SYMBOL_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_VOLUME)]                   = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_TURNOVER)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_TURNOVER);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_INTEREST)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_INTEREST);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME)]        = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_BUY_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME)]       = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_SELL_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_OPEN)]                     = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_OPEN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_CLOSE)]                    = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_CLOSE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_AW)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_AW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT)]         = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_SETTLEMENT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_LIMIT_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_LIMIT_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BID)]                              = this.m_tick.bid;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASK)]                              = this.m_tick.ask;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LAST)]                             = this.m_tick.last;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDHIGH)]                          = this.SymbolBidHigh();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDLOW)]                           = this.SymbolBidLow();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_REAL)]                      = this.SymbolVolumeReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMEHIGH_REAL)]                  = this.SymbolVolumeHighReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMELOW_REAL)]                   = this.SymbolVolumeLowReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_OPTION_STRIKE)]                    = this.SymbolOptionStrike();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_ACCRUED_INTEREST)]           = this.SymbolTradeAccruedInterest();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_FACE_VALUE)]                 = this.SymbolTradeFaceValue();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_LIQUIDITY_RATE)]             = this.SymbolTradeLiquidityRate();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_HEDGED)]                    = this.SymbolMarginHedged();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_INITIAL)]              = this.m_margin_rate.Long.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL)]          = this.m_margin_rate.BuyStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL)]         = this.m_margin_rate.BuyLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL)]     = this.m_margin_rate.BuyStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE)]          = this.m_margin_rate.Long.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE)]      = this.m_margin_rate.BuyStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE)]     = this.m_margin_rate.BuyLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE)] = this.m_margin_rate.BuyStopLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_INITIAL)]             = this.m_margin_rate.Short.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL)]         = this.m_margin_rate.SellStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL)]        = this.m_margin_rate.SellLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL)]    = this.m_margin_rate.SellStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE)]         = this.m_margin_rate.Short.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE)]     = this.m_margin_rate.SellStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE)]    = this.m_margin_rate.SellLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE)]= this.m_margin_rate.SellStopLimit.Maintenance;
//--- Save string properties
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_NAME)]                             = this.m_name;
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_BASE)]                    = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_BASE);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_PROFIT)]                  = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_PROFIT);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_MARGIN)]                  = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_MARGIN);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_DESCRIPTION)]                      = ::SymbolInfoString(this.m_name,SYMBOL_DESCRIPTION);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_PATH)]                             = ::SymbolInfoString(this.m_name,SYMBOL_PATH);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_BASIS)]                            = this.SymbolBasis();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_BANK)]                             = this.SymbolBank();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_ISIN)]                             = this.SymbolISIN();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_FORMULA)]                          = this.SymbolFormula();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_PAGE)]                             = this.SymbolPage();
//--- Save additional integer properties
   this.m_long_prop[SYMBOL_PROP_DIGITS_LOTS]                                        = this.SymbolDigitsLot();
//---
   if(!select)
      this.RemoveFromMarketWatch();
  }
//+------------------------------------------------------------------+
```

Now the constructor receives the Market Watch symbol index, the symbol name is assigned to the object name and the structure of the previous symbol data is reset. The previous data structure field **trade\_mode** features WRONG\_VALUE (the presence of this value in the symbol trading mode field allows us to define the first launch). The variables of changed and controlled symbol properties are initialized. The index passed to the constructor is written to the "Market Watch window index" symbol property.

Since now we have the **m\_name** variable in the **CBaseObj** base class to store the object name (in our case, it is a symbol name), the **CSymbol** class should no longer have the **m\_symbol\_name** variable. All its occurrences should be replaced with **m\_name**(a symbol name is assigned in the class constructor).

Highlight any of the this.m\_symbol\_name text occurences in the **CSymbol** class listing and press **Ctrl+H**. The search and replace window appears with the highlighted occurrence already inserted in the search field. Enter **this.m\_name** to the replace field and replace all detected this.m\_symbol\_name occurrences with this.m\_name in the listing.

We should also remove the remaining class member variables that are now located in the CBaseObj base object. Their presence in CSymbol triggers the variable duplication warning during the compilation. In the files attached below, all such variables have already been removed. Simply download the files and have a look at their content.

**Add the description of the new "Market Watch window index" symbol property to the method returning the description of an integer property:**

```
//+------------------------------------------------------------------+
//| Return the description of the symbol integer property            |
//+------------------------------------------------------------------+
string CSymbol::GetPropertyDescription(ENUM_SYMBOL_PROP_INTEGER property)
  {
   return
     (
      property==SYMBOL_PROP_STATUS              ?  TextByLanguage("Статус","Status")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_INDEX_MW            ?  TextByLanguage("Индекс в окне \"Обзор рынка\"","Index in the \"Market Watch window\"")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetStatusDescription()
         )  :
      property==SYMBOL_PROP_CUSTOM              ?  TextByLanguage("Пользовательский символ","Custom symbol")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_CHART_MODE          ?  TextByLanguage("Тип цены для построения баров","Price type used for generating symbols bars")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetChartModeDescription()
         )  :
      property==SYMBOL_PROP_EXIST               ?  TextByLanguage("Символ с таким именем существует","Symbol with this name exists")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_SELECT  ?  TextByLanguage("Символ выбран в Market Watch","Symbol selected in Market Watch")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_VISIBLE ?  TextByLanguage("Символ отображается в Market Watch","Symbol visible in Market Watch")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_SESSION_DEALS       ?  TextByLanguage("Количество сделок в текущей сессии","Number of deals in the current session")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_BUY_ORDERS  ?  TextByLanguage("Общее число ордеров на покупку в текущий момент","Number of Buy orders at the moment")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_SESSION_SELL_ORDERS ?  TextByLanguage("Общее число ордеров на продажу в текущий момент","Number of Sell orders at the moment")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUME              ?  TextByLanguage("Объем в последней сделке","Volume of the last deal")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUMEHIGH          ?  TextByLanguage("Максимальный объём за день","Maximal day volume")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_VOLUMELOW           ?  TextByLanguage("Минимальный объём за день","Minimal day volume")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_TIME                ?  TextByLanguage("Время последней котировки","Time of the last quote")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)==0 ? TextByLanguage("(Ещё не было тиков)","(No ticks yet)") : TimeMSCtoString(this.GetProperty(property)))
         )  :
      property==SYMBOL_PROP_DIGITS              ?  TextByLanguage("Количество знаков после запятой","Digits after decimal point")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_DIGITS_LOTS         ?  TextByLanguage("Количество знаков после запятой в значении лота","Digits after decimal point in the value of the lot")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_SPREAD              ?  TextByLanguage("Размер спреда в пунктах","Spread value in points")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_SPREAD_FLOAT        ?  TextByLanguage("Плавающий спред","Spread is floating")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_TICKS_BOOKDEPTH     ?  TextByLanguage("Максимальное количество показываемых заявок в стакане","Maximal number of requests shown in Depth of Market")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+ #ifdef __MQL5__(string)this.GetProperty(property) #else TextByLanguage("Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      property==SYMBOL_PROP_TRADE_CALC_MODE     ?  TextByLanguage("Способ вычисления стоимости контракта","Contract price calculation mode")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetCalcModeDescription()
         )  :
      property==SYMBOL_PROP_TRADE_MODE ?  TextByLanguage("Тип исполнения ордеров","Order execution type")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetTradeModeDescription()
         )  :
      property==SYMBOL_PROP_START_TIME          ?  TextByLanguage("Дата начала торгов по инструменту","Date of symbol trade beginning")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)==0  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": "+TimeMSCtoString(this.GetProperty(property)*1000))
         )  :
      property==SYMBOL_PROP_EXPIRATION_TIME     ?  TextByLanguage("Дата окончания торгов по инструменту","Date of symbol trade end")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)==0  ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": "+TimeMSCtoString(this.GetProperty(property)*1000))
         )  :
      property==SYMBOL_PROP_TRADE_STOPS_LEVEL   ?  TextByLanguage("Минимальный отступ от цены закрытия для установки Stop ордеров","Minimal indention from close price to place Stop orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_TRADE_FREEZE_LEVEL  ?  TextByLanguage("Дистанция заморозки торговых операций","Distance to freeze trade operations in points")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_TRADE_EXEMODE       ?  TextByLanguage("Режим заключения сделок","Deal execution mode")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetTradeExecDescription()
         )  :
      property==SYMBOL_PROP_SWAP_MODE           ?  TextByLanguage("Модель расчета свопа","Swap calculation model")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetSwapModeDescription()
         )  :
      property==SYMBOL_PROP_SWAP_ROLLOVER3DAYS  ?  TextByLanguage("День недели для начисления тройного свопа","Day of week to charge 3 days swap rollover")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+DayOfWeekDescription(this.SwapRollover3Days())
         )  :
      property==SYMBOL_PROP_MARGIN_HEDGED_USE_LEG  ?  TextByLanguage("Расчет хеджированной маржи по наибольшей стороне","Calculating hedging margin using the larger leg")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==SYMBOL_PROP_EXPIRATION_MODE     ?  TextByLanguage("Флаги разрешенных режимов истечения ордера","Flags of allowed order expiration modes")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetExpirationModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_FILLING_MODE        ?  TextByLanguage("Флаги разрешенных режимов заливки ордера","Flags of allowed order filling modes")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetFillingModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_ORDER_MODE          ?  TextByLanguage("Флаги разрешённых типов ордеров","Flags of allowed order types")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetOrderModeFlagsDescription()
         )  :
      property==SYMBOL_PROP_ORDER_GTC_MODE      ?  TextByLanguage("Срок действия StopLoss и TakeProfit ордеров","Expiration of Stop Loss and Take Profit orders")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetOrderGTCModeDescription()
         )  :
      property==SYMBOL_PROP_OPTION_MODE         ?  TextByLanguage("Тип опциона","Option type")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetOptionTypeDescription()
         )  :
      property==SYMBOL_PROP_OPTION_RIGHT        ?  TextByLanguage("Право опциона","Option right")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetOptionRightDescription()
         )  :
      property==SYMBOL_PROP_BACKGROUND_COLOR    ?  TextByLanguage("Цвет фона символа в Market Watch","Background color of symbol in Market Watch")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         #ifdef __MQL5__
         (this.GetProperty(property)==CLR_DEFAULT || this.GetProperty(property)==CLR_NONE ?  TextByLanguage(": (Отсутствует)",": (Not set)") : ": "+::ColorToString((color)this.GetProperty(property),true))
         #else TextByLanguage(": Свойство не поддерживается в MQL4","Property not supported in MQL4") #endif
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Remove the second form of the Exist() method and the method returning the number of decimal places in the number value from the CSymbol class methods implementation listing:

```
//+------------------------------------------------------------------+
bool CSymbol::Exist(const string name) const
  {
   int total=::SymbolsTotal(false);
   for(int i=0;i<total;i++)
      if(::SymbolName(i,false)==name)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return the number of decimal places in the 'double' value        |
//+------------------------------------------------------------------+
int CSymbol::GetDigits(const double value) const
  {
   string val_str=(string)value;
   int len=::StringLen(val_str);
   int n=len-::StringFind(val_str,".",0)-1;
   if(::StringSubstr(val_str,len-1,1)=="0")
      n--;
   return n;
  }
//+------------------------------------------------------------------+
```

These methods are simply redundant here — **Exist()** with the input is not needed in this class. Therefore, I have moved it to the **DELib.mqh** file of service functions, which is the right place for it, while **GetDigits()** is now located in the **CBaseObj** base class.

The Refresh() method of the CSymbol class is launched from the timer and updates all symbol data. We are going to search for symbol property changes in the same method. We have another method — RefreshRates(), which is also launched from the timer, but with a higher refresh rate. Only symbol quote data is updated in this method. If we implement the search for symbol property changes in both methods, this will lead to duplication of event messages.

The possible solution is as follows: the RefreshRates() method updates quote data and returns the flag of its successful receipt. The method will be called from its timer as before. But the ability to call it from the Refresh() method will be added as well. Thus, both methods are called as before — each in its timer, while the search for symbol property changes is performed only in the Refresh() method.

**Let's add the necessary changes to the RefreshRates() and Refresh() methods:**

```
//+------------------------------------------------------------------+
//| Update quote data by a symbol                                    |
//+------------------------------------------------------------------+
bool CSymbol::RefreshRates(void)
  {
//--- Get quote data
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();
      return false;
     }
//--- Update integer properties
   this.m_long_prop[SYMBOL_PROP_VOLUME]                                             = (long)this.m_tick.volume;
   this.m_long_prop[SYMBOL_PROP_TIME]                                               = this.TickTime();
//--- Update real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASK)]                              = this.m_tick.ask;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKHIGH)]                          = ::SymbolInfoDouble(this.m_name,SYMBOL_ASKHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_ASKLOW)]                           = ::SymbolInfoDouble(this.m_name,SYMBOL_ASKLOW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BID)]                              = this.m_tick.bid;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDHIGH)]                          = this.SymbolBidHigh();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_BIDLOW)]                           = this.SymbolBidLow();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LAST)]                             = this.m_tick.last;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTHIGH)]                         = ::SymbolInfoDouble(this.m_name,SYMBOL_LASTHIGH);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_LASTLOW)]                          = ::SymbolInfoDouble(this.m_name,SYMBOL_LASTLOW);
   return true;
  }
//+------------------------------------------------------------------+
```

First, get the symbol quote data. If failed to obtain it, write the error code and exit the method returning false. If data has been obtained, fill in the necessary symbol properties and return true.

```
//+------------------------------------------------------------------+
//| Update all symbol data                                           |
//+------------------------------------------------------------------+
void CSymbol::Refresh(void)
  {
//--- Update quote data
   if(!this.RefreshRates())
      return;
#ifdef __MQL5__
   ::ResetLastError();
   if(!this.MarginRates())
     {
      this.m_global_error=::GetLastError();
      return;
     }
#endif
//--- Prepare event data
   this.m_is_event=false;
   ::ZeroMemory(this.m_struct_curr_symbol);
   this.m_hash_sum=0;
//--- Update integer properties
   this.m_long_prop[SYMBOL_PROP_SELECT]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_SELECT);
   this.m_long_prop[SYMBOL_PROP_VISIBLE]                                            = ::SymbolInfoInteger(this.m_name,SYMBOL_VISIBLE);
   this.m_long_prop[SYMBOL_PROP_SESSION_DEALS]                                      = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_DEALS);
   this.m_long_prop[SYMBOL_PROP_SESSION_BUY_ORDERS]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_BUY_ORDERS);
   this.m_long_prop[SYMBOL_PROP_SESSION_SELL_ORDERS]                                = ::SymbolInfoInteger(this.m_name,SYMBOL_SESSION_SELL_ORDERS);
   this.m_long_prop[SYMBOL_PROP_VOLUMEHIGH]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_VOLUMEHIGH);
   this.m_long_prop[SYMBOL_PROP_VOLUMELOW]                                          = ::SymbolInfoInteger(this.m_name,SYMBOL_VOLUMELOW);
   this.m_long_prop[SYMBOL_PROP_SPREAD]                                             = ::SymbolInfoInteger(this.m_name,SYMBOL_SPREAD);
   this.m_long_prop[SYMBOL_PROP_TICKS_BOOKDEPTH]                                    = ::SymbolInfoInteger(this.m_name,SYMBOL_TICKS_BOOKDEPTH);
   this.m_long_prop[SYMBOL_PROP_START_TIME]                                         = ::SymbolInfoInteger(this.m_name,SYMBOL_START_TIME);
   this.m_long_prop[SYMBOL_PROP_EXPIRATION_TIME]                                    = ::SymbolInfoInteger(this.m_name,SYMBOL_EXPIRATION_TIME);
   this.m_long_prop[SYMBOL_PROP_TRADE_STOPS_LEVEL]                                  = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_STOPS_LEVEL);
   this.m_long_prop[SYMBOL_PROP_TRADE_FREEZE_LEVEL]                                 = ::SymbolInfoInteger(this.m_name,SYMBOL_TRADE_FREEZE_LEVEL);
   this.m_long_prop[SYMBOL_PROP_BACKGROUND_COLOR]                                   = this.SymbolBackgroundColor();

//--- Update real properties
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE_PROFIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS)]            = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_VALUE_LOSS);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_TICK_SIZE)]                  = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_TICK_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_CONTRACT_SIZE)]              = ::SymbolInfoDouble(this.m_name,SYMBOL_TRADE_CONTRACT_SIZE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MIN)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_MAX)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_STEP)]                      = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_STEP);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_LIMIT)]                     = ::SymbolInfoDouble(this.m_name,SYMBOL_VOLUME_LIMIT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_LONG)]                        = ::SymbolInfoDouble(this.m_name,SYMBOL_SWAP_LONG);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SWAP_SHORT)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_SWAP_SHORT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_INITIAL)]                   = ::SymbolInfoDouble(this.m_name,SYMBOL_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_MAINTENANCE)]               = ::SymbolInfoDouble(this.m_name,SYMBOL_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_VOLUME)]                   = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_TURNOVER)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_TURNOVER);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_INTEREST)]                 = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_INTEREST);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME)]        = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_BUY_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME)]       = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_SELL_ORDERS_VOLUME);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_OPEN)]                     = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_OPEN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_CLOSE)]                    = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_CLOSE);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_AW)]                       = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_AW);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT)]         = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_SETTLEMENT);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_LIMIT_MIN);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX)]          = ::SymbolInfoDouble(this.m_name,SYMBOL_SESSION_PRICE_LIMIT_MAX);
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUME_REAL)]                      = this.SymbolVolumeReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMEHIGH_REAL)]                  = this.SymbolVolumeHighReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_VOLUMELOW_REAL)]                   = this.SymbolVolumeLowReal();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_OPTION_STRIKE)]                    = this.SymbolOptionStrike();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_ACCRUED_INTEREST)]           = this.SymbolTradeAccruedInterest();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_FACE_VALUE)]                 = this.SymbolTradeFaceValue();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_TRADE_LIQUIDITY_RATE)]             = this.SymbolTradeLiquidityRate();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_HEDGED)]                    = this.SymbolMarginHedged();
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_INITIAL)]              = this.m_margin_rate.Long.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL)]          = this.m_margin_rate.BuyStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL)]         = this.m_margin_rate.BuyLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL)]     = this.m_margin_rate.BuyStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE)]          = this.m_margin_rate.Long.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE)]      = this.m_margin_rate.BuyStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE)]     = this.m_margin_rate.BuyLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE)] = this.m_margin_rate.BuyStopLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_INITIAL)]             = this.m_margin_rate.Short.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL)]         = this.m_margin_rate.SellStop.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL)]        = this.m_margin_rate.SellLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL)]    = this.m_margin_rate.SellStopLimit.Initial;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE)]         = this.m_margin_rate.Short.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE)]     = this.m_margin_rate.SellStop.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE)]    = this.m_margin_rate.SellLimit.Maintenance;
   this.m_double_prop[this.IndexProp(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE)]= this.m_margin_rate.SellStopLimit.Maintenance;

//--- Fill in the current symbol data structure
   this.m_struct_curr_symbol.ask                                                    = this.Ask();
   this.m_struct_curr_symbol.ask_high                                               = this.AskHigh();
   this.m_struct_curr_symbol.ask_low                                                = this.AskLow();
   this.m_struct_curr_symbol.bid_last                                               = (this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.Bid() : this.Last());
   this.m_struct_curr_symbol.bid_last_high                                          = (this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.BidHigh() : this.LastHigh());
   this.m_struct_curr_symbol.bid_last_low                                           = (this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.BidLow() : this.LastLow());
   this.m_struct_curr_symbol.volume                                                 = this.Volume();
   this.m_struct_curr_symbol.session_deals                                          = this.SessionDeals();
   this.m_struct_curr_symbol.session_buy_orders                                     = this.SessionBuyOrders();
   this.m_struct_curr_symbol.session_sell_orders                                    = this.SessionSellOrders();
   this.m_struct_curr_symbol.volume_high_day                                        = this.VolumeHigh();
   this.m_struct_curr_symbol.volume_low_day                                         = this.VolumeLow();
   this.m_struct_curr_symbol.spread                                                 = this.Spread();
   this.m_struct_curr_symbol.stops_level                                            = this.TradeStopLevel();
   this.m_struct_curr_symbol.freeze_level                                           = this.TradeFreezeLevel();
   this.m_struct_curr_symbol.volume_limit                                           = this.VolumeLimit();
   this.m_struct_curr_symbol.swap_long                                              = this.SwapLong();
   this.m_struct_curr_symbol.swap_short                                             = this.SwapShort();
   this.m_struct_curr_symbol.session_volume                                         = this.SessionVolume();
   this.m_struct_curr_symbol.session_turnover                                       = this.SessionTurnover();
   this.m_struct_curr_symbol.session_interest                                       = this.SessionInterest();
   this.m_struct_curr_symbol.session_buy_ord_volume                                 = this.SessionBuyOrdersVolume();
   this.m_struct_curr_symbol.session_sell_ord_volume                                = this.SessionSellOrdersVolume();
   this.m_struct_curr_symbol.session_open                                           = this.SessionOpen();
   this.m_struct_curr_symbol.session_close                                          = this.SessionClose();
   this.m_struct_curr_symbol.session_aw                                             = this.SessionAW();
   this.m_struct_curr_symbol.volume_real_day                                        = this.VolumeReal();
   this.m_struct_curr_symbol.volume_high_real_day                                   = this.VolumeHighReal();
   this.m_struct_curr_symbol.volume_low_real_day                                    = this.VolumeLowReal();
   this.m_struct_curr_symbol.option_strike                                          = this.OptionStrike();
   this.m_struct_curr_symbol.trade_mode                                             = this.TradeMode();

//--- Hash sum calculation
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.volume;
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.session_deals;
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.session_buy_orders;
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.session_sell_orders;
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.volume_high_day;
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.volume_low_day;
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.spread;
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.stops_level;
   this.m_hash_sum+=(double)this.m_struct_curr_symbol.freeze_level;
   this.m_hash_sum+=this.m_struct_curr_symbol.ask;
   this.m_hash_sum+=this.m_struct_curr_symbol.ask_high;
   this.m_hash_sum+=this.m_struct_curr_symbol.ask_low;
   this.m_hash_sum+=this.m_struct_curr_symbol.bid_last;
   this.m_hash_sum+=this.m_struct_curr_symbol.bid_last_high;
   this.m_hash_sum+=this.m_struct_curr_symbol.bid_last_low;
   this.m_hash_sum+=this.m_struct_curr_symbol.volume_limit;
   this.m_hash_sum+=this.m_struct_curr_symbol.swap_long;
   this.m_hash_sum+=this.m_struct_curr_symbol.swap_short;
   this.m_hash_sum+=this.m_struct_curr_symbol.session_volume;
   this.m_hash_sum+=this.m_struct_curr_symbol.session_turnover;
   this.m_hash_sum+=this.m_struct_curr_symbol.session_interest;
   this.m_hash_sum+=this.m_struct_curr_symbol.session_buy_ord_volume;
   this.m_hash_sum+=this.m_struct_curr_symbol.session_sell_ord_volume;
   this.m_hash_sum+=this.m_struct_curr_symbol.session_open;
   this.m_hash_sum+=this.m_struct_curr_symbol.session_close;
   this.m_hash_sum+=this.m_struct_curr_symbol.session_aw;
   this.m_hash_sum+=this.m_struct_curr_symbol.volume_real_day;
   this.m_hash_sum+=this.m_struct_curr_symbol.volume_high_real_day;
   this.m_hash_sum+=this.m_struct_curr_symbol.volume_low_real_day;
   this.m_hash_sum+=this.m_struct_curr_symbol.option_strike;
   this.m_hash_sum+=this.m_struct_curr_symbol.trade_mode;

//--- First launch
   if(this.m_struct_prev_symbol.trade_mode==WRONG_VALUE)
     {
      this.m_struct_prev_symbol=this.m_struct_curr_symbol;
      this.m_hash_sum_prev=this.m_hash_sum;
      return;
     }
//--- If symbol's hash sum changed
   if(this.m_hash_sum!=this.m_hash_sum_prev)
     {
      this.m_event_code=this.SetEventCode();
      this.SetTypeEvent();
      CEventBaseObj *event=this.GetEvent(WRONG_VALUE,false);
      if(event!=NULL)
        {
         ENUM_SYMBOL_EVENT event_id=(ENUM_SYMBOL_EVENT)event.ID();
         if(event_id!=SYMBOL_EVENT_NO_EVENT)
           {
            this.m_is_event=true;
           }
        }
      this.m_hash_sum_prev=this.m_hash_sum;
     }
  }
//+------------------------------------------------------------------+
```

Here, we first call the RefreshRates() method. If failed to obtain the quote data, there is no point in obtaining the remaining data. Exit the method.

Next, reset the event flag, the structure of the current symbol properties states and the current hash sum. Then, fill all symbol properties with current data and write these data to the structure of the current symbol properties states. After filling in the structure, calculate the hash sum.

If this is the first launch (the [SYMBOL\_TRADE\_MODE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_trade_mode) symbol property is set to WRONG\_VALUE), save the values of the current status structure to the previous status structure, write the current hash sum to the current one and exit the method.

If this is not the first launch, we need to check the previous and current hash sums for inequality.

If the hash sum has changed, a symbol property has changed. Call the method for placing an event code, the method for receiving an event from the event code and event entries in the list of occurred events and the last event from the newly added events. Check the event type. If it is not "No event", activate the event flag. Finally, save the current hash sum as the previous one for further checks.

**This completes the improvement of the CSymbol class for searching events and working with a new base object.**

The symbol object class is ready. Now each symbol is capable of tracking its events and placing them to its event list. Since we use the symbol collection, we need to cycle through all collection symbols getting an event list from each of them. All these events should be placed to the collection event list (now the collection features the same list since it is located in the CBaseObj base object). So, it only remains to survey the obtained event list to define a fact an event or several events have occurred at each of the collection symbols.

Besides, we can work with the Market Watch window. Let's implement the event tracking for it as well, such as adding/removing a symbol and sorting symbols. To do this, we need to save the Market Watch window snapshot and the hash sum of the symbols in it. When the hash sum changes, we now understand that a market watch event has occurred. It remains only to determine the event by comparing the current Market Watch window status with the snapshot and send an identified event to the control program.

Open the **SymbolsCollection.mqh** file of the symbol collection class and make the necessary changes:

```
//+------------------------------------------------------------------+
//|                                            SymbolsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayString.mqh>
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\Symbols\SymbolFX.mqh"
#include "..\Objects\Symbols\SymbolFXMajor.mqh"
#include "..\Objects\Symbols\SymbolFXMinor.mqh"
#include "..\Objects\Symbols\SymbolFXExotic.mqh"
#include "..\Objects\Symbols\SymbolFXRub.mqh"
#include "..\Objects\Symbols\SymbolMetall.mqh"
#include "..\Objects\Symbols\SymbolIndex.mqh"
#include "..\Objects\Symbols\SymbolIndicative.mqh"
#include "..\Objects\Symbols\SymbolCrypto.mqh"
#include "..\Objects\Symbols\SymbolCommodity.mqh"
#include "..\Objects\Symbols\SymbolExchange.mqh"
#include "..\Objects\Symbols\SymbolFutures.mqh"
#include "..\Objects\Symbols\SymbolCFD.mqh"
#include "..\Objects\Symbols\SymbolStocks.mqh"
#include "..\Objects\Symbols\SymbolBonds.mqh"
#include "..\Objects\Symbols\SymbolOption.mqh"
#include "..\Objects\Symbols\SymbolCollateral.mqh"
#include "..\Objects\Symbols\SymbolCustom.mqh"
#include "..\Objects\Symbols\SymbolCommon.mqh"
//+------------------------------------------------------------------+
//| Symbol collection                                                |
//+------------------------------------------------------------------+
class CSymbolsCollection : public CBaseObj
  {
private:
   CListObj          m_list_all_symbols;                          // The list of all symbol objects
   CArrayString      m_list_names;                                // Symbol name control list
   ENUM_SYMBOLS_MODE m_mode_list;                                 // Mode of working with symbol lists
   ENUM_SYMBOL_EVENT m_last_event;                                // The last event
   string            m_array_symbols[];                           // The array of used symbols passed from the program
   int               m_delta_symbol;                              // Difference in the number of symbols compared to the previous check
   int               m_total_symbols;                             // Number of symbols in the Market Watch window
   int               m_total_symbol_prev;                         // Number of symbols in the Market Watch window during the previous check
//--- Return the flag of a symbol object presence by its name in the (1) list of all symbols, (2) Market Watch window, (3) control list
   bool              IsPresentSymbolInList(const string symbol_name);
   bool              IsPresentSymbolInMW(const string symbol_name);
   bool              IsPresentSymbolInControlList(const string symbol_name);
//--- Create the symbol object and place it to the list
   bool              CreateNewSymbol(const ENUM_SYMBOL_STATUS symbol_status,const string name,const int index);
//--- Return the (1) type of a used symbol list (Market watch/Server),
//--- (2) the number of visible symbols, (3) symbol index in the Market Watch window
   ENUM_SYMBOLS_MODE TypeSymbolsList(const string &symbol_used_array[]);
   int               SymbolsTotalVisible(void)                    const;
   int               SymbolIndexInMW(const string name)           const;

//--- Define a symbol affiliation with a group by name and return it
   ENUM_SYMBOL_STATUS SymbolStatus(const string symbol_name)      const;
//--- Return a symbol affiliation with a category by custom criteria
   ENUM_SYMBOL_STATUS StatusByCustomPredefined(const string symbol_name)  const;
//--- Return a symbol affiliation with categories by margin calculation
   ENUM_SYMBOL_STATUS StatusByCalcMode(const string symbol_name)  const;
//--- Return a symbol affiliation with pre-defined (1) majors, (2) minors, (3) exotics, (4) RUB,
//--- (5) indicatives, (6) metals, (7) commodities, (8) indices, (9) cryptocurrency, (10) options
   bool              IsPredefinedFXMajor(const string name)       const;
   bool              IsPredefinedFXMinor(const string name)       const;
   bool              IsPredefinedFXExotic(const string name)      const;
   bool              IsPredefinedFXRUB(const string name)         const;
   bool              IsPredefinedIndicative(const string name)    const;
   bool              IsPredefinedMetall(const string name)        const;
   bool              IsPredefinedCommodity(const string name)     const;
   bool              IsPredefinedIndex(const string name)         const;
   bool              IsPredefinedCrypto(const string name)        const;
   bool              IsPredefinedOption(const string name)        const;

public:
//--- Return the full collection list 'as is'
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_all_symbols;                                      }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)    { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }
   CArrayObj        *GetList(ENUM_SYMBOL_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::BySymbolProperty(this.GetList(),property,value,mode); }
//--- Return the (1) symbol object, (2) the symbol object index from the list by a name
   CSymbol          *GetSymbolByName(const string name);
   int               GetSymbolIndexByName(const string name);
//--- Return the number of new symbols in the Market Watch window
   int               NewSymbols(void)    const                                                              { return this.m_delta_symbol;                                           }
//--- Return (1) the mode of working with symbol lists, (2) the event flag and (3) the event of one of the collection symbols
   ENUM_SYMBOLS_MODE ModeSymbolsList(void)                        const                                     { return this.m_mode_list;                                              }
   bool              IsEvent(void)                                const                                     { return this.m_is_event;                                               }
   int               GetLastEventsCode(void)                      const                                     { return this.m_event_code;                                             }
   ENUM_SYMBOL_EVENT GetLastEvent(void)                           const                                     { return this.m_last_event;                                             }
//--- Return the number of symbols in the collection
   int               GetSymbolsCollectionTotal(void)              const                                     { return this.m_list_all_symbols.Total();                               }

//--- Constructor
                     CSymbolsCollection();

//--- (1) Set the list of used symbols, (2) creating the symbol list (Market Watch or the complete list)
   bool              SetUsedSymbols(const string &symbol_used_array[]);
   bool              CreateSymbolsList(const bool flag);
//--- Save names of used Market Watch symbols
   void              CopySymbolsNames(void);
//--- Update (1) all, (2) quote data of the collection symbols
   virtual void      Refresh(void);
   void              RefreshRates(void);
//--- Working with the events of the (1) collection symbol list, (2) market watch window
   void              SymbolsEventsControl(void);
   void              MarketWatchEventsControl(const bool send_events=true);
//--- Return the description of the (1) Market Watch window event, (2) mode of working with symbols
   string            EventDescription(const ENUM_SYMBOL_EVENT event);
   string            ModeSymbolsListDescription(void);
  };
//+------------------------------------------------------------------+
```

The [CArrayString](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraystring) class file included from the standard library is to be used to create a snapshot of the Market Watch window. This list is to store the copy of the symbol set from the market watch and compare the current status of the symbol list in the window with the one in the snapshot. In case of any changes, we should react to them for creating the Market Watch window event.

The last event that happened to any of the collection symbols is added to the **m\_last\_event** variable. Thus, the variable stores the latest event occurred at one of the used symbols.

The **m\_array\_symbols** array passes the array of used symbols from the control program to the symbol collection class.

The current and previous number of symbols in the Market Watch window is to be stored in the **m\_total\_symbols** and **m\_total\_symbols\_prev** class member variables. Comparing the values of the variables allows us to define the events of adding and removing symbols from the market watch.

The **IsPresentSymbolInMW()** and **IsPresentSymbolInControlList()** private class methods return the symbol presence flags by its name in the Market Watch window and the snapshot list of the market watch window accordingly. The **SymbolsTotalVisible()** and **SymbolIndexInMW()** methods return the number of visible symbols in the Market Watch window and the symbol index in the window list accordingly.

Add the following methods to the class public section:

**IsEvent()** — return the flag of an event presence in the symbol collection or the Market Watch window,

**GetLastEventsCode()** — return the code of the last event in the symbol collection or the Market Watch window,

**GetLastEvent()** — return the last event in the symbol collection or the Market Watch window,

**GetSymbolsCollectionTotal()** — return the total number of symbols in the collection,

**CreateSymbolsList()** — create the collection list when working with the Market Watch window or the full list of symbols on the server (not more than 1000),

**CopySymbolsNames()** — create a snapshot of Market Watch window symbols from the list of all collection symbols,

**Refresh()** — now the method is declared as a virtual one, since it is declared as virtual in the base class object (while the symbol collection is now created based on the CBaseObj base object class),

**SymbolsEventsControl()** — method of working with the symbol collection list for defining the symbol collection events,

**MarketWatchEventsControl()** — method of working with the Market Watch window list for defining events in the market watch window,

**EventDescription()** — return a description of the symbol collection event,

**ModeSymbolsListDescription()** — return a description of the mode of working with symbols.

Let's have a look at the implementation of the methods.

In the initialization list of the **class constructor**, initialize the variables for working with the Market Watch window. In the class body, change sorting the list of collection symbols from sorting by name to sorting by index in the Market Watch window and clear the list of the market watch window snapshot.

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSymbolsCollection::CSymbolsCollection(void) : m_total_symbol_prev(0),
                                               m_delta_symbol(0),
                                               m_mode_list(SYMBOLS_MODE_CURRENT)
  {
   this.m_list_all_symbols.Sort(SORT_BY_SYMBOL_INDEX_MW);
   this.m_list_all_symbols.Clear();
   this.m_list_all_symbols.Type(COLLECTION_SYMBOLS_ID);
   this.m_list_names.Clear();
  }
//+------------------------------------------------------------------+
```

Since some symbols may be not visible in the Market Watch window but still be present there ( [SYMBOL\_VISIBLE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) symbol property), some symbols (generally, these are crosses necessary for calculation of margin requirements and profit in the deposit currency) are selected automatically and are not displayed in the Market Watch. Therefore, in order to find out the number of visible symbols only, we need to calculate only symbols having this property set in a loop by window symbols.

**The SymbolsTotalVisible() method returns the number of visible symbols in the Market Watch window:**

```
//+------------------------------------------------------------------+
//| Return the number of visible symbols in the Market Watch window  |
//+------------------------------------------------------------------+
int CSymbolsCollection::SymbolsTotalVisible(void) const
  {
   int total=::SymbolsTotal(true),n=0;
   for(int i=0;i<total;i++)
     {
      if(!::SymbolInfoInteger(::SymbolName(i,true),SYMBOL_VISIBLE))
         continue;
      n++;
     }
   return n;
  }
//+------------------------------------------------------------------+
```

**The method returning the symbol index in the Market Watch window list:**

```
//+------------------------------------------------------------------+
//| Return symbol index in the Market Watch window                   |
//+------------------------------------------------------------------+
int CSymbolsCollection::SymbolIndexInMW(const string name) const
  {
   int total=::SymbolsTotal(true);
   for(int i=0;i<total;i++)
     {
      if(!::SymbolInfoInteger(::SymbolName(i,true),SYMBOL_VISIBLE))
         continue;
      if(SymbolName(i,true)==name)
         return i;
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

When working with the symbol list from the Market Watch window, we need to find out the index of each of the symbols in the window list to be able to synchronize the location of symbols in the collection symbol list with the location of symbols in the market watch window. For example, this may be useful for creating a custom symbol list window fully synchronized with the terminal window. The index is one of the symbol properties allowing us to sort the list by it. The index is to be passed to the abstract symbol class constructor.

**The method returning the flag of a visible symbol presence in the Market Watch window:**

```
//+------------------------------------------------------------------+
//| Return the visible symbol object presence flag                   |
//| by its name in the Market Watch window                           |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPresentSymbolInMW(const string symbol_name)
  {
   int total=SymbolsTotal(true);
   for(int i=0;i<total;i++)
     {
      string name=::SymbolName(i,true);
      if(!::SymbolInfoInteger(name,SYMBOL_VISIBLE))
         continue;
      if(name==symbol_name)
         return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the symbol name, then, in the loop by the full list of symbolsselected in the market watch window, we skip the invisible symbol, compare the name of the next symbol with the one passed to the method. Return true if the names match. If the symbol is not found in the list, return false.

**The method returning the flag of a symbol present in the snapshot list of the market watch window:**

```
//+------------------------------------------------------------------+
//| Return the flag of a symbol object presence in the control list  |
//+------------------------------------------------------------------+
bool CSymbolsCollection::IsPresentSymbolInControlList(const string symbol_name)
  {
   int total=this.m_list_names.Total();
   for(int i=0;i<total;i++)
     {
      string name=this.m_list_names.At(i);
      if(name==NULL)
         continue;
      if(name==symbol_name)
         return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

A necessary symbol name is passed to the method. By the market watch window snapshot list in a loop,receive the necessary symbol, and if its name matches the one passed to the method, return true, otherwise — false.

**The method of creating the list when working with the Market Watch window or the full list of symbols on the server:**

```
//+------------------------------------------------------------------+
//| Creating the symbol list (Market Watch or the complete list)     |
//+------------------------------------------------------------------+
bool CSymbolsCollection::CreateSymbolsList(const bool flag)
  {
   bool res=true;
   int total=::SymbolsTotal(flag);
   for(int i=0;i<total && i<SYMBOLS_COMMON_TOTAL;i++)
     {
      string name=::SymbolName(i,flag);
      if(flag && !::SymbolInfoInteger(name,SYMBOL_VISIBLE))
         continue;
      ENUM_SYMBOL_STATUS status=this.SymbolStatus(name);
      bool add=this.CreateNewSymbol(status,name,i);
      res &=add;
      if(!add)
         continue;
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the flag setting the search mode: true — working with selected symbols in the market watch window, false — working with the complete list of symbols present on the server. Get the total number of symbols depending on the flag — either in the market watch window, or on the server, and, in a loop by the list, but not more than the value set by the SYMBOLS\_COMMON\_TOTAL constant of the **Defines.mqh** file (1000 symbols), get the name of the next symbol from the list, check its visibility property if working with the market watch window and skip it if it is invisible.

Then obtain the symbol object status by its name and add the symbol to the list of all collection symbols using the CreateNewSymbol() method [considered in the previous article](https://www.mql5.com/en/articles/7041#node03). (Note that the method has been slightly changed due to adding the new symbol property — its index in the Market Watch window symbol list. Now the index is also passed to the symbol object).

The results of adding each symbol to the list of all collection symbols are added to the variable returning the method operation result, and the total value of this variable is returned from the method upon completion of the entire symbol handling cycle.

**Let's consider the improved method of creating a symbol object and its placement to the list:**

```
//+------------------------------------------------------------------+
//| Create a symbol object and place it to the list                  |
//+------------------------------------------------------------------+
bool CSymbolsCollection::CreateNewSymbol(const ENUM_SYMBOL_STATUS symbol_status,const string name,const int index)
  {
   if(this.IsPresentSymbolInList(name))
     {
      return true;
     }
   if(#ifdef __MQL5__ !::SymbolInfoInteger(name,SYMBOL_EXIST) #else !Exist(name) #endif )
     {
      string t1=TextByLanguage("Ошибка входных данных: нет символа ","Input error: no ");
      string t2=TextByLanguage(" на сервере"," symbol on the server");
      ::Print(DFUN,t1,name,t2);
      this.m_global_error=ERR_MARKET_UNKNOWN_SYMBOL;
      return false;
     }
   CSymbol *symbol=NULL;
   switch(symbol_status)
     {
      case SYMBOL_STATUS_FX         :  symbol=new CSymbolFX(name,index);         break;   // Forex symbol
      case SYMBOL_STATUS_FX_MAJOR   :  symbol=new CSymbolFXMajor(name,index);    break;   // Major Forex symbol
      case SYMBOL_STATUS_FX_MINOR   :  symbol=new CSymbolFXMinor(name,index);    break;   // Minor Forex symbol
      case SYMBOL_STATUS_FX_EXOTIC  :  symbol=new CSymbolFXExotic(name,index);   break;   // Exotic Forex symbol
      case SYMBOL_STATUS_FX_RUB     :  symbol=new CSymbolFXRub(name,index);      break;   // Forex symbol/RUR
      case SYMBOL_STATUS_METAL      :  symbol=new CSymbolMetall(name,index);     break;   // Metal
      case SYMBOL_STATUS_INDEX      :  symbol=new CSymbolIndex(name,index);      break;   // Index
      case SYMBOL_STATUS_INDICATIVE :  symbol=new CSymbolIndicative(name,index); break;   // Indicative
      case SYMBOL_STATUS_CRYPTO     :  symbol=new CSymbolCrypto(name,index);     break;   // Cryptocurrency symbol
      case SYMBOL_STATUS_COMMODITY  :  symbol=new CSymbolCommodity(name,index);  break;   // Commodity
      case SYMBOL_STATUS_EXCHANGE   :  symbol=new CSymbolExchange(name,index);   break;   // Exchange symbol
      case SYMBOL_STATUS_FUTURES    :  symbol=new CSymbolFutures(name,index);    break;   // Futures
      case SYMBOL_STATUS_CFD        :  symbol=new CSymbolCFD(name,index);        break;   // CFD
      case SYMBOL_STATUS_STOCKS     :  symbol=new CSymbolStocks(name,index);     break;   // Stock
      case SYMBOL_STATUS_BONDS      :  symbol=new CSymbolBonds(name,index);      break;   // Bond
      case SYMBOL_STATUS_OPTION     :  symbol=new CSymbolOption(name,index);     break;   // Option
      case SYMBOL_STATUS_COLLATERAL :  symbol=new CSymbolCollateral(name,index); break;   // Non-tradable asset
      case SYMBOL_STATUS_CUSTOM     :  symbol=new CSymbolCustom(name,index);     break;   // Custom symbol
      default                       :  symbol=new CSymbolCommon(name,index);     break;   // The rest
     }
   if(symbol==NULL)
     {
      ::Print(DFUN,TextByLanguage("Не удалось создать объект-символ ","Failed to create symbol object "),name);
      return false;
     }
   if(!this.m_list_all_symbols.Add(symbol))
     {
      string t1=TextByLanguage("Не удалось добавить символ ","Failed to add ");
      string t2=TextByLanguage(" в список"," symbol to the list");
      ::Print(DFUN,t1,name,t2);
      delete symbol;
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

As can be seen from the listing, here we additionally pass the index sending it with a symbol name to each of the symbol object class constructors. This means we need to refine each of the classes derived from the CSymbol abstract symbol class.

**Let's consider this refinement using the CSymbolFX class as an example:**

```
//+------------------------------------------------------------------+
//|                                                     SymbolFX.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Symbol.mqh"
//+------------------------------------------------------------------+
//| Forex symbol                                                     |
//+------------------------------------------------------------------+
class CSymbolFX : public CSymbol
  {
public:
//--- Constructor
                     CSymbolFX(const string name,const int index) : CSymbol(SYMBOL_STATUS_FX,name,index) {}
//--- Supported integer properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_INTEGER property);
//--- Supported real properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_DOUBLE property);
//--- Supported string properties of a symbol
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_STRING property);
//--- Display a short symbol description in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
```

Here, in addition to a symbol name, the class constructor also receives its index, and the index is passed to the CSymbol parent class constructor in the initialization list.

This is all that needs to be changed in all classes derived from the abstract symbol. All changes to the symbol object classes have already been made in the files attached below.

**The improved method setting the list of used symbols in the collection:**

```
//+------------------------------------------------------------------+
//| Set the list of used symbols                                     |
//+------------------------------------------------------------------+
bool CSymbolsCollection::SetUsedSymbols(const string &symbol_used_array[])
  {
   ::ArrayCopy(this.m_array_symbols,symbol_used_array);
   this.m_mode_list=this.TypeSymbolsList(this.m_array_symbols);
   this.m_list_all_symbols.Clear();
   this.m_list_all_symbols.Sort(SORT_BY_SYMBOL_INDEX_MW);
   //--- Use only the current symbol
   if(this.m_mode_list==SYMBOLS_MODE_CURRENT)
     {
      string name=::Symbol();
      ENUM_SYMBOL_STATUS status=this.SymbolStatus(name);
      return this.CreateNewSymbol(status,name,this.SymbolIndexInMW(name));
     }
   else
     {
      bool res=true;
      //--- Use the pre-defined symbol list
      if(this.m_mode_list==SYMBOLS_MODE_DEFINES)
        {
         int total=::ArraySize(this.m_array_symbols);
         for(int i=0;i<total;i++)
           {
            string name=this.m_array_symbols[i];
            ENUM_SYMBOL_STATUS status=this.SymbolStatus(name);
            bool add=this.CreateNewSymbol(status,name,this.SymbolIndexInMW(name));
            res &=add;
            if(!add)
               continue;
           }
         return res;
        }
      //--- Use the full list of the server symbols
      else if(this.m_mode_list==SYMBOLS_MODE_ALL)
        {
         return this.CreateSymbolsList(false);
        }
      //--- Use the symbol list from the Market Watch window
      else if(this.m_mode_list==SYMBOLS_MODE_MARKET_WATCH)
        {
         this.MarketWatchEventsControl(false);
         return true;
        }
     }
   return false;
  }
//+------------------------------------------------------------------+
```

Here, copy the array of used symbols passed from the control program to the custom array. Save the mode of working with symbols and set the sorting of the list of all symbols to sorting by index. Now, the method of creating a new symbol object now receives its index in addition to the symbol name.

When using the full list of all symbols on the server, call the method of constructing the collection symbol list with the flag = false indicating the construction of the full list of symbols on the server.

When using the list from the Market Watch window, call the method of working with the market watch window with the flag = false, which indicates the requirement to create the list and fill in its data, rather than the complete ban on working with events.

**The method of working with events of all collection symbols:**

```
//+------------------------------------------------------------------+
//| Working with the events of the collection symbol list            |
//+------------------------------------------------------------------+
void CSymbolsCollection::SymbolsEventsControl(void)
  {
   this.m_is_event=false;
   this.m_list_events.Clear();
   this.m_list_events.Sort();
   //--- The full update of all collection symbols
   int total=this.m_list_all_symbols.Total();
   for(int i=0;i<total;i++)
     {
      CSymbol *symbol=this.m_list_all_symbols.At(i);
      if(symbol==NULL)
         continue;
      symbol.Refresh();
      if(!symbol.IsEvent())
         continue;
      this.m_is_event=true;
      CArrayObj *list=symbol.GetListEvents();
      if(list==NULL)
         continue;
      this.m_event_code=symbol.GetEventCode();
      int n=list.Total();
      for(int j=0; j<n; j++)
        {
         CEventBaseObj *event=list.At(j);
         if(event==NULL)
            continue;
         ENUM_SYMBOL_EVENT event_id=(ENUM_SYMBOL_EVENT)event.ID();
         if(event_id==SYMBOL_EVENT_NO_EVENT)
            continue;
         this.m_last_event=event_id;
         if(this.EventAdd((ushort)event.ID(),event.LParam(),event.DParam(),event.SParam()))
           {
            ::EventChartCustom(this.m_chart_id,(ushort)event_id,event.LParam(),event.DParam(),event.SParam());
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The method works in the timer. The data is initialized first:

the symbol collection event flag is reset,

the list of events in the symbol collection is cleared and

the sorted list flag is set for the list.

Next, in the loop by all symbols in the collection symbol list, receive the next symbol, update all its data and check the presence of the event flag set for the symbol. If there is no event, check the next symbol in the collection.

If the symbol features the event flag, set the event flag for the entire collection (presence of an event at least at one of the symbols means the event is present for the entire collection). Get the list of all current symbol events, set the code of the last collection event equal to the event code of the current symbol from the list. If the event is present, the value set in the variable storing the code of the last collection event is updated for the next symbol, while in a loop by the list of all current symbol events, we get a new event and save the event ID as the last collection event. In the same way, the event of the next symbol updates the last symbol collection event.

Next, create the collection event with the symbol event parameters set for it ( event ID, long, double and string) and save the symbol event to the list of symbol collection events.

In case a symbol event is successfully saved to the collection event list, the event is sent to the program chart using the [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) function with the same event parameters for further handling of the event in the calling program.

Thus, receiving the event list from each of the symbols in the collection list, we go through the list of events of each symbols sending all its events to the collection event list. Upon completion of the cycle, the list of collection events features all events of all collection symbols. A custom event is created for each of the events and sent to the control program chart.

To distinguish events in the market watch window, we need to calculate the hash sum of all market watch symbols. Its change indicates an event has occurred. The first thing that comes to mind is a simple count of the number of symbols in the window. However, adding or removing a symbol increases or decreases the size of the list, while sorting symbols by the mouse does not change the number of symbols. This means the number of symbols in the Market Watch window is not suitable for calculating the hash sum.

Let's do as follows: a name of each symbol stored in the list can be represented as a number (symbol code) comprising of a sum of uchar values of symbol (character) codes the symbol name consists of with an index of a symbol in the market watch window added to it. The sum of all these symbol codes forms the hash sum.

- Adding a symbol to the list changes the hash sum (the new code of the added symbol is added to it).
- Removing a symbol from the list changes the hash sum as well (the code of the removed symbol is subtracted from the hash sum).
- Sorting the symbol list changes the hash sum (codes of sorted symbols are changed since their indices are altered)


Implementing the **method of working with the Market Watch window events:**

```
//+------------------------------------------------------------------+
//| Working with market watch window events                          |
//+------------------------------------------------------------------+
void CSymbolsCollection::MarketWatchEventsControl(const bool send_events=true)
  {
   ::ResetLastError();
//--- If no current prices are received, exit
   if(!::SymbolInfoTick(::Symbol(),this.m_tick))
     {
      this.m_global_error=::GetLastError();
      return;
     }
   uchar array[];
   int sum=0;
   this.m_hash_sum=0;
//--- Calculate the hash sum of all visible symbols in the Market Watch window
   this.m_total_symbols=this.SymbolsTotalVisible();
   //--- In the loop by all Market Watch window symbols
   int total_symbols=::SymbolsTotal(true);
   for(int i=0;i<total_symbols;i++)
     {
      //--- get a symbol name by index
      string name=::SymbolName(i,true);
      //--- skip if invisible
      if(!::SymbolInfoInteger(name,SYMBOL_VISIBLE))
         continue;
      //--- write symbol name (characters) codes to the uchar array
      ::StringToCharArray(name,array);
      //--- in a loop by the resulting array, sum up the values of all array cells creating the symbol code
      for(int j=::ArraySize(array)-1;j>WRONG_VALUE;j--)
         sum+=array[j];
      //--- add the symbol code and the loop index specifying the symbol index in the market watch list to the hash sum
      m_hash_sum+=i+sum;
     }
//--- If sending events is disabled, create the collection list and exit saving the current hash some as the previous one
   if(!send_events)
     {
      //--- Clear the list
      this.m_list_all_symbols.Clear();
      //--- Clear the collection list
      this.CreateSymbolsList(true);
      //--- Clear the market watch window snapshot
      this.CopySymbolsNames();
      //--- save the current hash some as the previous one
      this.m_hash_sum_prev=this.m_hash_sum;
      //--- save the current number of visible symbols as the previous one
      this.m_total_symbol_prev=this.m_total_symbols;
      return;
     }

//--- If the hash sum of symbols in the Market Watch window has changed
   if(this.m_hash_sum!=this.m_hash_sum_prev)
     {
      //--- Define the Market Watch window event
      this.m_delta_symbol=this.m_total_symbols-this.m_total_symbol_prev;
      ENUM_SYMBOL_EVENT event_id=
        (
         this.m_total_symbols>this.m_total_symbol_prev ? SYMBOL_EVENT_MW_ADD :
         this.m_total_symbols<this.m_total_symbol_prev ? SYMBOL_EVENT_MW_DEL :
         SYMBOL_EVENT_MW_SORT
        );
      //--- Adding a symbol to the Market Watch window
      if(event_id==SYMBOL_EVENT_MW_ADD)
        {
         string name="";
         //--- In the loop by all Market Watch window symbols
         int total=::SymbolsTotal(true), index=WRONG_VALUE;
         for(int i=0;i<total;i++)
           {
            //--- get the symbol name and check its "visibility". Skip it if invisible
            name=::SymbolName(i,true);
            if(!::SymbolInfoInteger(name,SYMBOL_VISIBLE))
               continue;
            //--- If there is no symbol in the collection symbol list yet
            if(!this.IsPresentSymbolInList(name))
              {
               //--- clear the collection list
               this.m_list_all_symbols.Clear();
               //--- recreate the collection list
               this.CreateSymbolsList(true);
               //--- create the symbol collection snapshot
               this.CopySymbolsNames();
               //--- get a new symbol index in the Market Watch window
               index=this.GetSymbolIndexByName(name);
               //--- If the "Adding a new symbol" event is successfully added to the event list
               if(this.EventAdd(event_id,this.TickTime(),index,name))
                 {
                  //--- send the event to the chart:
                  //--- long value = event time in milliseconds, double value = symbol index, string value = added symbol name
                  ::EventChartCustom(this.m_chart_id,(ushort)event_id,this.TickTime(),index,name);
                 }
              }
           }
         //--- Save the new number of visible symbols in the market watch window
         this.m_total_symbols=this.SymbolsTotalVisible();
        }
      //--- Remove a symbol from the Market Watch window
      else if(event_id==SYMBOL_EVENT_MW_DEL)
        {
         //--- clear the collection list
         this.m_list_all_symbols.Clear();
         //--- recreate the collection list
         this.CreateSymbolsList(true);
         //--- In a loop by the market watch window snapshot
         int total=this.m_list_names.Total();
         for(int i=0; i<total;i++)
           {
            //--- get a symbol name
            string name=this.m_list_names.At(i);
            if(name==NULL)
               continue;
            //--- if no symbol with such a name exists in the collection symbol list
            if(!this.IsPresentSymbolInList(name))
              {
               //--- If the "Removing a symbol" event is successfully added to the event list
               if(this.EventAdd(event_id,this.TickTime(),WRONG_VALUE,name))
                 {
                  //--- send the event to the chart:
                  //--- long value = event tine in milliseconds, double value = -1 for an absent symbol, string value = a removed symbol name
                  ::EventChartCustom(this.m_chart_id,(ushort)event_id,this.TickTime(),WRONG_VALUE,name);
                 }
              }
           }
         //--- Recreate the market watch snapshot
         this.CopySymbolsNames();
         //--- Save the new number of visible symbols in the market watch window
         this.m_total_symbols=this.SymbolsTotalVisible();
        }
      //--- Sorting symbols in the Market Watch window
      else if(event_id==SYMBOL_EVENT_MW_SORT)
        {
         //--- clear the collection list
         this.m_list_all_symbols.Clear();
         //--- set sorting of the collection list as sorting by index
         this.m_list_all_symbols.Sort(SORT_BY_SYMBOL_INDEX_MW);
         //--- recreate the collection list
         this.CreateSymbolsList(true);
         //--- get the current symbol index in the Market Watch window
         int index=this.GetSymbolIndexByName(Symbol());
         //--- send the event to the chart:
         //--- long value = event time in milliseconds, double value = current symbol index, string value = current symbol name
         ::EventChartCustom(this.m_chart_id,(ushort)event_id,this.TickTime(),index,::Symbol());
        }
      //--- save the current number of visible symbols as the previous one
      this.m_total_symbol_prev=this.m_total_symbols;
      //--- save the current hash some as the previous one
      this.m_hash_sum_prev=this.m_hash_sum;
     }
  }
//+------------------------------------------------------------------+
```

To avoid describing all branches in the method code, I have placed the code in the listing block by block. Each of the blocks is accompanied by detailed comments. I hope, everything is clear there. If you have any questions, ask them in the comments to the article.

When working with the Market Watch window to track events occurring in it, working with the hash sum only is insufficient. If we want to know what happened before an event, we need to have a copy of the market watch symbol list (market watch snapshot). For instance, this copy allows us to find out what symbol has been removed. Without the market watch snapshot, we cannot know the name of the removed symbol.

**The method of creating the Market Watch window snapshot:**

```
//+------------------------------------------------------------------+
//| Save names of used Market Watch symbols                          |
//+------------------------------------------------------------------+
void CSymbolsCollection::CopySymbolsNames(void)
  {
   this.m_list_names.Clear();
   int total=this.m_list_all_symbols.Total();
   for(int i=0;i<total;i++)
     {
      CSymbol *symbol=this.m_list_all_symbols.At(i);
      if(symbol==NULL)
         continue;
      this.m_list_names.Add(symbol.Name());
     }
  }
//+------------------------------------------------------------------+
```

Here, we clear the list of symbol names. In a loop by the collection symbol list,receive a new symbol and add it to the list of symbol names.

Upon completion of the loop, we will have the list of names of all symbols in the symbol collection list.

**The method returning a symbol object by name:**

```
//+------------------------------------------------------------------+
//| Return an object symbol from the list by a name                  |
//+------------------------------------------------------------------+
CSymbol *CSymbolsCollection::GetSymbolByName(const string name)
  {
   CArrayObj *list=this.GetList(SYMBOL_PROP_NAME,name,EQUAL);
   if(list==NULL || list.Total()==0)
      return NULL;
   CSymbol *symbol=list.At(0);
   return(symbol!=NULL ? symbol : NULL);
  }
//+------------------------------------------------------------------+
```

A symbol name is passed to the method. Next, using the method of receiving the list of GetList() objects by the "Symbol name" property, create a new list a single object symbol should be located at provided that its namematchesthe one passed to the method.

Get the symbol object from the list and  return it if the search is successful or return NULL if there is no symbol with such a name in the symbol collection list.

**The method returning a symbol index in the symbol collection list:**

```
//+------------------------------------------------------------------+
//| Return the symbol object index from the list by a name           |
//+------------------------------------------------------------------+
int CSymbolsCollection::GetSymbolIndexByName(const string name)
  {
   int total=this.m_list_all_symbols.Total();
   for(int i=0;i<total;i++)
     {
      CSymbol *symbol=this.m_list_all_symbols.At(i);
      if(symbol==NULL)
         continue;
      if(symbol.Name()==name)
         return i;
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

Here, the necessary symbol name is passed to the method. Then, in a loop by all symbols in the collection symbol list, we get another symbol object from the list. If its name matches the necessary one, return the loop index. Otherwise, return -1.

**The method returning a description of an event from the Market Watch window:**

```
//+------------------------------------------------------------------+
//| Return the Market Watch window event description                 |
//+------------------------------------------------------------------+
string CSymbolsCollection::EventDescription(const ENUM_SYMBOL_EVENT event)
  {
   return
     (
      event==SYMBOL_EVENT_MW_ADD    ?  TextByLanguage("В окно \"Обзор рынка\" добавлен символ","Added symbol to \"Market Watch\" window")                                     :
      event==SYMBOL_EVENT_MW_DEL    ?  TextByLanguage("Из окна \"Обзор рынка\" удалён символ","Removed from \"Market Watch\" window")                                       :
      event==SYMBOL_EVENT_MW_SORT   ?  TextByLanguage("Изменено расположение символов в окне \"Обзор рынка\"","Changed arrangement of symbols in \"Market Watch\" window")  :
      EnumToString(event)
     );
  }
//+------------------------------------------------------------------+
```

The method receives an event and returns its text description depending on what event has been received.

**The method returning the mode of working with symbols:**

```
//+------------------------------------------------------------------+
//| Return a description of the mode of working with symbols         |
//+------------------------------------------------------------------+
string CSymbolsCollection::ModeSymbolsListDescription(void)
  {
   return
     (
      this.m_mode_list==SYMBOLS_MODE_CURRENT       ? TextByLanguage("Работа только с текущим символом","Work only with current symbol")                                :
      this.m_mode_list==SYMBOLS_MODE_DEFINES       ? TextByLanguage("Работа с предопределённым списком символов","Work with predefined list of symbols")                 :
      this.m_mode_list==SYMBOLS_MODE_MARKET_WATCH  ? TextByLanguage("Работа с символами из окна \"Обзор рынка\"","Working with symbols from \"Market Watch\" window")  :
      TextByLanguage("Работа с полным списком всех доступных символов","Work with full list of all available symbols")
     );
  }
//+------------------------------------------------------------------+
```

Here, the presence of the **m\_mode\_list** variable is checked and the text description of the operation mode according to the variable value is returned.

**This completes creating the symbol event class.**

Before putting the symbol events class to work, let's keep in mind that we can also arrange event tracking in it. The class now features the CBaseObj base object and the account class is now based on CBaseObj, which means that both classes can use the already prepared event search functionality. All subsequent objects to be derived from CBaseObj will also be endowed with the properties allowing you to track the object events.

### Improving the account event class

Open the **Account.mqh** account class file and make the necessary changes.

**Replace including the Object.mqh file with including BaseObj.mqh:**

```
//+------------------------------------------------------------------+
//|                                                      Account.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\BaseObj.mqh"
#include "..\..\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| Account class                                                    |
//+------------------------------------------------------------------+
class CAccount : public CBaseObj
  {
private:
```

Let's set CBaseObj as the base class instead of CObject.

Since we are now able to assign a name to an object inherited from CBaseObj, use that feature and set the name for the account object.

**At the very end of the CAccount class constructor, add the line setting the account object name:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccount::CAccount(void)
  {
//--- Save integer properties
   this.m_long_prop[ACCOUNT_PROP_LOGIN]                              = ::AccountInfoInteger(ACCOUNT_LOGIN);
   this.m_long_prop[ACCOUNT_PROP_TRADE_MODE]                         = ::AccountInfoInteger(ACCOUNT_TRADE_MODE);
   this.m_long_prop[ACCOUNT_PROP_LEVERAGE]                           = ::AccountInfoInteger(ACCOUNT_LEVERAGE);
   this.m_long_prop[ACCOUNT_PROP_LIMIT_ORDERS]                       = ::AccountInfoInteger(ACCOUNT_LIMIT_ORDERS);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_SO_MODE]                     = ::AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
   this.m_long_prop[ACCOUNT_PROP_TRADE_ALLOWED]                      = ::AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   this.m_long_prop[ACCOUNT_PROP_TRADE_EXPERT]                       = ::AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_MODE]                        = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_MARGIN_MODE) #else ACCOUNT_MARGIN_MODE_RETAIL_HEDGING #endif ;
   this.m_long_prop[ACCOUNT_PROP_CURRENCY_DIGITS]                    = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif ;
   this.m_long_prop[ACCOUNT_PROP_SERVER_TYPE]                        = (::TerminalInfoString(TERMINAL_NAME)=="MetaTrader 5" ? 5 : 4);

//--- Save real properties
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_BALANCE)]          = ::AccountInfoDouble(ACCOUNT_BALANCE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_CREDIT)]           = ::AccountInfoDouble(ACCOUNT_CREDIT);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_PROFIT)]           = ::AccountInfoDouble(ACCOUNT_PROFIT);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_EQUITY)]           = ::AccountInfoDouble(ACCOUNT_EQUITY);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN)]           = ::AccountInfoDouble(ACCOUNT_MARGIN);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_FREE)]      = ::AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_LEVEL)]     = ::AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_SO_CALL)]   = ::AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_SO_SO)]     = ::AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_INITIAL)]   = ::AccountInfoDouble(ACCOUNT_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_MAINTENANCE)]=::AccountInfoDouble(ACCOUNT_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_ASSETS)]           = ::AccountInfoDouble(ACCOUNT_ASSETS);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_LIABILITIES)]      = ::AccountInfoDouble(ACCOUNT_LIABILITIES);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_COMMISSION_BLOCKED)]=::AccountInfoDouble(ACCOUNT_COMMISSION_BLOCKED);

//--- Save string properties
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_NAME)]             = ::AccountInfoString(ACCOUNT_NAME);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_SERVER)]           = ::AccountInfoString(ACCOUNT_SERVER);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_CURRENCY)]         = ::AccountInfoString(ACCOUNT_CURRENCY);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_COMPANY)]          = ::AccountInfoString(ACCOUNT_COMPANY);

//--- Account object name
   this.m_name=TextByLanguage("Счёт ","Account ")+(string)this.Login()+": "+this.Name()+" ("+this.Company()+")";
  }
//+-------------------------------------------------------------------+
```

As we can see, the object name consists of the Account text, account number, client name and name of the company serving the account.

For example, when connecting to one of the accounts on MetaQuotes-Demo under my name, the account object name looks as follows: _"Account 8550475: Artyom Trishkin (MetaQuotes Software Corp.)"_

**Enter the 'names' variable value to the method for displaying a brief account name (previously, the variable was set the same way as we have just set the account object):**

```
//+------------------------------------------------------------------+
//| Display a short account description in the journal               |
//+------------------------------------------------------------------+
void CAccount::PrintShort(void)
  {
   string mode=(this.MarginMode()==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING ? ", Hedge" : this.MarginMode()==ACCOUNT_MARGIN_MODE_EXCHANGE ? ", Exhange" : "");
   string names=this.m_name+" ";
   string values=::DoubleToString(this.Balance(),(int)this.CurrencyDigits())+" "+this.Currency()+", 1:"+(string)+this.Leverage()+mode+", "+this.TradeModeDescription()+" "+this.ServerTypeDescription();
   ::Print(names,values);
  }
//+------------------------------------------------------------------+
```

**This concludes the improvement of the CAccount class.**

Now let's improve the account collection class. Open the **AccountsCollection.mqh** file and add the necessary changes.

Let's assign the role of the account collection class **base object to the CBaseObj class:**

```
//+------------------------------------------------------------------+
//| Account collection                                               |
//+------------------------------------------------------------------+
class CAccountsCollection : public CBaseObj
  {
private:
```

Since the class is inherited from the base object featuring the functionality for tracking object events, remove the duplicated variables and methods from the account collection class.

**In the account data structure, remove the hash sum field:**

```
   struct MqlDataAccount
     {
      double         hash_sum;               // Account data hash sum
      //--- Account integer properties
      long           login;                  // ACCOUNT_LOGIN (Account number)
      long           leverage;               // ACCOUNT_LEVERAGE (Leverage)
      int            limit_orders;           // ACCOUNT_LIMIT_ORDERS (Maximum allowed number of active pending orders)
      bool           trade_allowed;          // ACCOUNT_TRADE_ALLOWED (Permission to trade for the current account from the server side)
      bool           trade_expert;           // ACCOUNT_TRADE_EXPERT (Permission to trade for an EA from the server side)
      //--- Account real properties
      double         balance;                // ACCOUNT_BALANCE (Account balance in a deposit currency)
      double         credit;                 // ACCOUNT_CREDIT (Credit in a deposit currency)
      double         profit;                 // ACCOUNT_PROFIT (Current profit on an account in the account currency)
      double         equity;                 // ACCOUNT_EQUITY (Equity on an account in the deposit currency)
      double         margin;                 // ACCOUNT_MARGIN (Reserved margin on an account in a deposit currency)
      double         margin_free;            // ACCOUNT_MARGIN_FREE (Free funds available for opening a position in a deposit currency)
      double         margin_level;           // ACCOUNT_MARGIN_LEVEL (Margin level on an account in %)
      double         margin_so_call;         // ACCOUNT_MARGIN_SO_CALL (MarginCall)
      double         margin_so_so;           // ACCOUNT_MARGIN_SO_SO (StopOut)
      double         margin_initial;         // ACCOUNT_MARGIN_INITIAL (Funds reserved on an account to ensure a guarantee amount for all pending orders)
      double         margin_maintenance;     // ACCOUNT_MARGIN_MAINTENANCE (Funds reserved on an account to ensure a minimum amount for all open positions)
      double         assets;                 // ACCOUNT_ASSETS (Current assets on an account)
      double         liabilities;            // ACCOUNT_LIABILITIES (Current liabilities on an account)
      double         comission_blocked;      // ACCOUNT_COMMISSION_BLOCKED (Current sum of blocked commissions on an account)
     };
```

**Remove private class member variables:**

```
   MqlTick           m_tick;                             // Tick structure
   string            m_symbol;                           // Current symbol
   long              m_chart_id;                         // Control program chart ID
   CListObj          m_list_accounts;                    // Account object list
   CArrayInt         m_list_changes;                     // Account change list
   string            m_folder_name;                      // Name of a folder account objects are stored
   int               m_index_current;                    // Index of an account object featuring the current account data
//--- Tracking account changes
   bool              m_is_account_event;                 // Account data event flag
   int               m_change_code;                      // Account change code
```

Rename the **SetChangeCode()** method to **SetEventCode()** so that the names of same-type methods remain the same in different classes.

The **SetTypeEvent()** method should be made virtual since it has already been declared in the CBaseObj class and should be implemented in the descendants.

Remove the **IsPresentEventFlag()** method from the class since it has already been implemented in CBaseObj.

Remove the duplicated methods in the class public section.

Remove the **GetEventCode()**, **GetListChanges()** and **SetChartID()** methods, while the **ENUM\_ACCOUNT\_EVENT GetEvent(const int shift=WRONG\_VALUE)** method should look as follows:

```
ENUM_ACCOUNT_EVENT GetEventID(const int shift=WRONG_VALUE,const bool check_out=true);
```

Let's consider its implementation outside the class body right away:

```
//+------------------------------------------------------------------+
//| Return the account event by its number in the list               |
//+------------------------------------------------------------------+
ENUM_ACCOUNT_EVENT CAccountsCollection::GetEventID(const int shift=WRONG_VALUE,const bool check_out=true)
  {
   CEventBaseObj *event=this.GetEvent(shift,check_out);
   if(event==NULL)
      return ACCOUNT_EVENT_NO_EVENT;
   return (ENUM_ACCOUNT_EVENT)event.ID();
  }
//+------------------------------------------------------------------+
```

The method receives the necessary event index (-1 for the last one) and the flag of detecting the index going beyond the event list boundaries.

Get the event object using the **GetEvent()** CBaseObj base object method we have considered in the beginning of the article. If there is no event, return "No event", otherwise return the event ID.

**In the initialization list of the class constructor, remove the initialization of all parameters except the symbol setting and set the name of the subfolder for storing the account object files:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccountsCollection::CAccountsCollection(void) : m_symbol(::Symbol())
  {
   this.m_list_accounts.Clear();
   this.m_list_accounts.Sort(SORT_BY_ACCOUNT_LOGIN);
   this.m_list_accounts.Type(COLLECTION_ACCOUNT_ID);
   ::ZeroMemory(this.m_struct_prev_account);
   ::ZeroMemory(this.m_tick);
   this.InitChangesParams();
   this.InitControlsParams();
//--- Create the folder for storing account files
   this.SetSubFolderName("Accounts");
   ::ResetLastError();
   if(!::FolderCreate(this.m_folder_name,FILE_COMMON))
      ::Print(DFUN,TextByLanguage("Не удалось создать папку хранения файлов. Ошибка ","Could not create file storage folder. Error "),::GetLastError());
//--- Create the current account object and add it to the list
   CAccount* account=new CAccount();
   if(account!=NULL)
     {
      if(!this.AddToList(account))
        {
         ::Print(DFUN_ERR_LINE,TextByLanguage("Ошибка. Не удалось добавить текущий объект-аккаунт в список-коллекцию.","Error. Failed to add current account object to collection list."));
         delete account;
        }
      else
         account.PrintShort();
     }
   else
      ::Print(DFUN,TextByLanguage("Ошибка. Не удалось создать объект-аккаунт с данными текущего счёта.","Error. Failed to create an account object with current account data."));

//--- Download account objects from the files to the collection
   this.LoadObjects();
//--- Save the current account index
   this.m_index_current=this.Index();
  }
//+------------------------------------------------------------------+
```

The **Refresh()** method for updating the account data should be made virtual since it is declared in the CBaseObj class and implemented in its descendants.

**Some changes are made in the method implementation:**

```
//+------------------------------------------------------------------+
//| Update the current account data                                  |
//+------------------------------------------------------------------+
void CAccountsCollection::Refresh(void)
  {
   ::ResetLastError();
   if(!::SymbolInfoTick(::Symbol(),this.m_tick))
     {
      this.m_global_error=::GetLastError();
      return;
     }
   if(this.m_index_current==WRONG_VALUE)
      return;
   CAccount* account=this.m_list_accounts.At(this.m_index_current);
   if(account==NULL)
      return;
//--- Prepare event data
   this.m_is_event=false;
   ::ZeroMemory(this.m_struct_curr_account);
   this.m_hash_sum=0;
   this.SetAccountsParams(account);

//--- First launch
   if(!this.m_struct_prev_account.login)
     {
      this.m_struct_prev_account=this.m_struct_curr_account;
      this.m_hash_sum_prev=this.m_hash_sum;
      return;
     }
//--- If the account hash sum changed
   if(this.m_hash_sum!=this.m_hash_sum_prev)
     {
      this.m_list_events.Clear();
      this.m_event_code=this.SetEventCode();
      this.SetTypeEvent();
      int total=this.m_list_events.Total();
      if(total>0)
        {
         this.m_is_event=true;
         for(int i=0;i<total;i++)
           {
            CEventBaseObj *event=this.GetEvent(i,false);
            if(event==NULL)
               continue;
            ENUM_ACCOUNT_EVENT event_id=(ENUM_ACCOUNT_EVENT)event.ID();
            if(event_id==ACCOUNT_EVENT_NO_EVENT)
               continue;
            long lparam=event.LParam();
            double dparam=event.DParam();
            string sparam=event.SParam();
            ::EventChartCustom(this.m_chart_id,(ushort)event_id,lparam,dparam,sparam);
           }
        }
      this.m_hash_sum_prev=this.m_hash_sum;
     }
  }
//+------------------------------------------------------------------+
```

**Let's consider the changes we have made.**

First, receive quote data by symbol (to define a millisecond time). If failed to receive, exit the method.

Reset the account event flag and the current hash sum value. During the first launch, save the current hash sum as the previous one.

When changing the hash sum, clear the account events list and perform the actions for receiving events from the account events list similar to the ones performed when receiving the symbol events list (discussed above).

Now, for any objects inherited from CBaseObj, the actions performed to receive their events will be similar. Therefore, it is better to once again familiarize yourself with obtaining them, so that everything remains clear in subsequent articles, and there is no need to return to the description of the actions taken to obtain the object events list.

In the method of saving account properties in the account object and account data structure, replace accessing the hash sum structure field with the variable of the hash sum CBaseObj class and save the object name:

```
//+------------------------------------------------------------------+
//| Write the current account data to the account object properties  |
//+------------------------------------------------------------------+
void CAccountsCollection::SetAccountsParams(CAccount *account)
  {
   if(account==NULL)
      return;
//--- Name
   this.m_name=account.GetName();
//--- Account number
   this.m_struct_curr_account.login=account.Login();
//--- Leverage
   account.SetProperty(ACCOUNT_PROP_LEVERAGE,::AccountInfoInteger(ACCOUNT_LEVERAGE));
   this.m_struct_curr_account.leverage=account.Leverage();
   this.m_hash_sum+=(double)this.m_struct_curr_account.leverage;
//--- Maximum allowed number of active pending orders
   account.SetProperty(ACCOUNT_PROP_LIMIT_ORDERS,::AccountInfoInteger(ACCOUNT_LIMIT_ORDERS));
   this.m_struct_curr_account.limit_orders=(int)account.LimitOrders();
   this.m_hash_sum+=(double)this.m_struct_curr_account.limit_orders;
//--- Permission to trade for the current account from the server side
   account.SetProperty(ACCOUNT_PROP_TRADE_ALLOWED,::AccountInfoInteger(ACCOUNT_TRADE_ALLOWED));
   this.m_struct_curr_account.trade_allowed=account.TradeAllowed();
   this.m_hash_sum+=(double)this.m_struct_curr_account.trade_allowed;
//--- Permission to trade for an EA from the server side
   account.SetProperty(ACCOUNT_PROP_TRADE_EXPERT,::AccountInfoInteger(ACCOUNT_TRADE_EXPERT));
   this.m_struct_curr_account.trade_expert=account.TradeExpert();
   this.m_hash_sum+=(double)this.m_struct_curr_account.trade_expert;
//--- Account balance in a deposit currency
   account.SetProperty(ACCOUNT_PROP_BALANCE,::AccountInfoDouble(ACCOUNT_BALANCE));
   this.m_struct_curr_account.balance=account.Balance();
   this.m_hash_sum+=(double)this.m_struct_curr_account.balance;
//--- Credit in a deposit currency
   account.SetProperty(ACCOUNT_PROP_CREDIT,::AccountInfoDouble(ACCOUNT_CREDIT));
   this.m_struct_curr_account.credit=account.Credit();
   this.m_hash_sum+=(double)this.m_struct_curr_account.credit;
//--- Current profit on an account in the account currency
   account.SetProperty(ACCOUNT_PROP_PROFIT,::AccountInfoDouble(ACCOUNT_PROFIT));
   this.m_struct_curr_account.profit=account.Profit();
   this.m_hash_sum+=(double)this.m_struct_curr_account.profit;
//--- Equity on an account in the deposit currency
   account.SetProperty(ACCOUNT_PROP_EQUITY,::AccountInfoDouble(ACCOUNT_EQUITY));
   this.m_struct_curr_account.equity=account.Equity();
   this.m_hash_sum+=(double)this.m_struct_curr_account.equity;
//--- Reserved margin on an account in the deposit currency
   account.SetProperty(ACCOUNT_PROP_MARGIN,::AccountInfoDouble(ACCOUNT_MARGIN));
   this.m_struct_curr_account.margin=account.Margin();
   this.m_hash_sum+=(double)this.m_struct_curr_account.margin;
//--- Free funds available for opening a position on an account in the deposit currency
   account.SetProperty(ACCOUNT_PROP_MARGIN_FREE,::AccountInfoDouble(ACCOUNT_MARGIN_FREE));
   this.m_struct_curr_account.margin_free=account.MarginFree();
   this.m_hash_sum+=(double)this.m_struct_curr_account.margin_free;
//--- Margin level on an account in %
   account.SetProperty(ACCOUNT_PROP_MARGIN_LEVEL,::AccountInfoDouble(ACCOUNT_MARGIN_LEVEL));
   this.m_struct_curr_account.margin_level=account.MarginLevel();
   this.m_hash_sum+=(double)this.m_struct_curr_account.margin_level;
//--- Margin Call level
   account.SetProperty(ACCOUNT_PROP_MARGIN_SO_CALL,::AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL));
   this.m_struct_curr_account.margin_so_call=account.MarginSOCall();
   this.m_hash_sum+=(double)this.m_struct_curr_account.margin_so_call;
//--- Stop Out level
   account.SetProperty(ACCOUNT_PROP_MARGIN_SO_SO,::AccountInfoDouble(ACCOUNT_MARGIN_SO_SO));
   this.m_struct_curr_account.margin_so_so=account.MarginSOSO();
   this.m_hash_sum+=(double)this.m_struct_curr_account.margin_so_so;
//--- Funds reserved on an account to ensure a guarantee amount for all pending orders
   account.SetProperty(ACCOUNT_PROP_MARGIN_INITIAL,::AccountInfoDouble(ACCOUNT_MARGIN_INITIAL));
   this.m_struct_curr_account.margin_initial=account.MarginInitial();
   this.m_hash_sum+=(double)this.m_struct_curr_account.margin_initial;
//--- Funds reserved on an account to ensure a minimum amount for all open positions
   account.SetProperty(ACCOUNT_PROP_MARGIN_MAINTENANCE,::AccountInfoDouble(ACCOUNT_MARGIN_MAINTENANCE));
   this.m_struct_curr_account.margin_maintenance=account.MarginMaintenance();
   this.m_hash_sum+=(double)this.m_struct_curr_account.margin_maintenance;
//--- Current assets on an account
   account.SetProperty(ACCOUNT_PROP_ASSETS,::AccountInfoDouble(ACCOUNT_ASSETS));
   this.m_struct_curr_account.assets=account.Assets();
   this.m_hash_sum+=(double)this.m_struct_curr_account.assets;
//--- Current liabilities on an account
   account.SetProperty(ACCOUNT_PROP_LIABILITIES,::AccountInfoDouble(ACCOUNT_LIABILITIES));
   this.m_struct_curr_account.liabilities=account.Liabilities();
   this.m_hash_sum+=(double)this.m_struct_curr_account.liabilities;
//--- Current sum of blocked commissions on an account
   account.SetProperty(ACCOUNT_PROP_COMMISSION_BLOCKED,::AccountInfoDouble(ACCOUNT_COMMISSION_BLOCKED));
   this.m_struct_curr_account.comission_blocked=account.ComissionBlocked();
   this.m_hash_sum+=(double)this.m_struct_curr_account.comission_blocked;
  }
//+------------------------------------------------------------------+
```

The **SetTypeEvent()** method of the account collection class has also been improved and changed due to the ability to define events of any object. The method is large, although all actions for defining account event types are of the same type. We have already considered them when analyzing the definition of symbol event types. Therefore, I will only provide an example of the event of enabling trading on an account. The full listing of the method is available in the files attached to the article:

```
//+------------------------------------------------------------------+
//| Set the account object event type                                |
//+------------------------------------------------------------------+
void CAccountsCollection::SetTypeEvent(void)
  {
   this.InitChangesParams();
   ENUM_ACCOUNT_EVENT event_id=ACCOUNT_EVENT_NO_EVENT;
//--- Changing permission to trade for the account
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_TRADE_ALLOWED))
     {
      if(!this.m_struct_curr_account.trade_allowed)
        {
         this.m_is_change_trade_allowed_off=true;
         event_id=ACCOUNT_EVENT_TRADE_ALLOWED_OFF;
         if(this.EventAdd(event_id,this.TickTime(),this.m_is_change_trade_allowed_off,this.m_name))
            this.m_struct_prev_account.trade_allowed=this.m_struct_curr_account.trade_allowed;
        }
      else
        {
         this.m_is_change_trade_allowed_on=true;
         event_id=ACCOUNT_EVENT_TRADE_ALLOWED_ON;
         if(this.EventAdd(event_id,this.TickTime(),this.m_is_change_trade_allowed_on,this.m_name))
            this.m_struct_prev_account.trade_allowed=this.m_struct_curr_account.trade_allowed;
        }
     }
//--- Changing permission for auto trading for the account
```

**This concludes the changes and improvements of the account collection class. Now it is time to put the updated symbol and account event classes to work.**

As you may remember, all control starts from the CEngine class, and all data is sent to it as well. Symbol and account event classes are no exceptions.

### Putting the symbol event class and the improved account class to work

Open the **Engine.mqh** file and add the necessary changes.

**In the private section of the class, declare the symbol events flag and a value of a last event on a symbol:**

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CSymbolsCollection   m_symbols;                       // Symbol collection
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
   ENUM_ACCOUNT_EVENT   m_last_account_event;            // Last event in the account properties
   ENUM_SYMBOL_EVENT    m_last_symbol_event;             // Last event in the symbol properties
//--- Return the counter index by id
```

**In the public section of the class, declare the method returning the description of the last trading event:**

```
//--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals,
//--- (4) all market orders of a position by its ID, (5) description of the last trading event
   CArrayObj           *GetListHistoryOrders(void);
   CArrayObj           *GetListHistoryPendings(void);
   CArrayObj           *GetListDeals(void);
   CArrayObj           *GetListAllOrdersByPosID(const ulong position_id);
   string               GetLastTradeEventDescription(void);
```

and **the new methods for working with symbol events. Also, change the method returning the account event flag:**

```
//--- Return the list of (1) used symbols, (2) symbol events, (3) the last symbol change event
//--- (4) the current symbol, (5) symbol event description, (6) Market Watch event description
   CArrayObj           *GetListAllUsedSymbols(void)                     { return this.m_symbols.GetList();                    }
   CArrayObj           *GetListSymbolsEvents(void)                      { return this.m_symbols.GetListEvents();              }
   ENUM_SYMBOL_EVENT    GetLastSymbolsEvent()                           { return this.m_symbols.GetLastEvent();               }
   CSymbol             *GetSymbolCurrent(void);
   string               GetSymbolEventDescription(ENUM_SYMBOL_EVENT event);
   string               GetMWEventDescription(ENUM_SYMBOL_EVENT event)  { return this.m_symbols.EventDescription(event);      }
   string               ModeSymbolsListDescription(void)                { return this.m_symbols.ModeSymbolsListDescription(); }

//--- Return the list of order, deal and position events
   CArrayObj           *GetListAllOrdersEvents(void)                    { return this.m_events.GetList();                     }
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_events.ResetLastTradeEvent(); }
//--- Return the (1) last trading event, (2) the last event in the account properties, (3) hedging account flag, (4) flag of working in the tester
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_last_trade_event;                     }
   ENUM_ACCOUNT_EVENT   LastAccountEvent(void)                    const { return this.m_last_account_event;                   }
   ENUM_SYMBOL_EVENT    LastSymbolsEvent(void)                    const { return this.m_last_symbol_event;                    }
//--- Return the (1) hedge account, (2) working in the tester, (3) account event and (4) symbol event flag
   bool                 IsHedge(void)                             const { return this.m_is_hedge;                             }
   bool                 IsTester(void)                            const { return this.m_is_tester;                            }
   bool                 IsAccountsEvent(void)                     const { return this.m_accounts.IsEvent();                   }
   bool                 IsSymbolsEvent(void)                      const { return this.m_symbols.IsEvent();                    }
//--- Return the (1) symbol object by name, as well as the code of the last event of (2) an account and (3) a symbol
   CSymbol             *GetSymbolObjByName(const string name)           { return this.m_symbols.GetSymbolByName(name);        }
   int                  GetAccountEventsCode(void)                const { return this.m_accounts.GetEventCode();              }
   int                  GetSymbolsEventsCode(void)                const { return this.m_symbols.GetLastEventsCode();          }
//--- Return the number of (1) symbols, (2) events in the symbol collection
   int                  GetSymbolsCollectionTotal(void)           const { return this.m_symbols.GetSymbolsCollectionTotal();  }
   int                  GetSymbolsCollectionEventsTotal(void)     const { return this.m_symbols.GetEventsTotal();             }
```

**In the initialization list of the class constructor, add initialization of the last event in the symbol collection:**

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),
                     m_last_trade_event(TRADE_EVENT_NO_EVENT),
                     m_last_account_event(ACCOUNT_EVENT_NO_EVENT),
                     m_last_symbol_event(SYMBOL_EVENT_NO_EVENT),
                     m_global_error(ERR_SUCCESS)
  {
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_is_tester=::MQLInfoInteger(MQL_TESTER);

   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_ORD_COUNTER_ID,COLLECTION_ORD_COUNTER_STEP,COLLECTION_ORD_PAUSE);
   this.CreateCounter(COLLECTION_ACC_COUNTER_ID,COLLECTION_ACC_COUNTER_STEP,COLLECTION_ACC_PAUSE);

   this.CreateCounter(COLLECTION_SYM_COUNTER_ID1,COLLECTION_SYM_COUNTER_STEP1,COLLECTION_SYM_PAUSE1);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID2,COLLECTION_SYM_COUNTER_STEP2,COLLECTION_SYM_PAUSE2);

   ::ResetLastError();
   #ifdef __MQL5__
      if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   //---__MQL4__
   #else
      if(!this.IsTester() && !::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   #endif
  }
//+------------------------------------------------------------------+
```

**In the class timer handler, add changes to the timer 1 and timer 2 handling blocks of the symbol collection:**

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of the collections of historical orders and deals, as well as of market orders and positions
   int index=this.CounterIndex(COLLECTION_ORD_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the order, deal and position collections events
            if(counter.IsTimeDone())
               this.TradeEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
            this.TradeEventsControl();
        }
     }
//--- Account collection timer
   index=this.CounterIndex(COLLECTION_ACC_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the account collection events
            if(counter.IsTimeDone())
               this.AccountEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
            this.AccountEventsControl();
        }
     }

//--- Timer 1 of the symbol collection (updating symbol quote data in the collection)
   index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID1);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If the pause is over, update quote data of all symbols in the collection
            if(counter.IsTimeDone())
               this.m_symbols.RefreshRates();
           }
         //--- In case of a tester, update quote data of all collection symbols by tick
         else
            this.m_symbols.RefreshRates();
        }
     }
//--- Timer 2 of the symbol collection (updating all data of all symbols in the collection and tracking symbl and symbol search events in the market watch window)
   index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID2);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If the pause is over
            if(counter.IsTimeDone())
              {
               //--- update data and work with events of all symbols in the collection
               this.SymbolEventsControl();
               //--- When working with the market watch list, check the market watch window events
               if(this.m_symbols.ModeSymbolsList()==SYMBOLS_MODE_MARKET_WATCH)
                  this.MarketWatchEventsControl();
              }
           }
         //--- If this is a tester, work with events of all symbols in the collection by tick
         else
            this.SymbolEventsControl();
        }
     }
  }
//+------------------------------------------------------------------+
```

Here, after the timer 1 counter completes its work, we need to simply update quote data of all collection symbols, therefore call the **RefreshRates()** method of the symbol collection.

Upon completion of the timer 2 counter operation, we need to fully update all collection symbols and track the occurred events of both the symbol collection and the Market Watch symbol list, therefore call the CEngine class methods **SymbolEventsControl()** and, when not working in the tester, — **MarketWatchEventsControl()**.

**Implementing the method of working with symbol collection events:**

```
//+------------------------------------------------------------------+
//| Working with symbol collection events                            |
//+------------------------------------------------------------------+
void CEngine::SymbolEventsControl(void)
  {
   this.m_symbols.SymbolsEventsControl();
   this.m_is_symbol_event=this.m_symbols.IsEvent();
//--- If there are changes in symbol properties
   if(this.m_is_symbol_event)
     {
      //--- Get the last event of the symbol property change
      this.m_last_symbol_event=this.m_symbols.GetLastEvent();
     }
  }
//+------------------------------------------------------------------+
```

Here, we call the symbol collection method **SymbolsEventsControl()** we have considered above when discussing the symbol collection events class. After the method completes its work, the event flag is enabled in the symbol collection class provided that an event was detected at any collection symbol. The flag status is set using the **IsEvent()** method of the **CBaseObj** base object class to the **m\_is\_symbol\_event** symbol collection event flag variable, whose value can be tracked in the calling program. If an event in the symbol collection has been registered, write the last event to the **m\_last\_symbol\_event** variable. Its value can also be tracked in the calling program.

**Implementing the method of working with the Market Watch window events:**

```
//+------------------------------------------------------------------+
//| Working with symbol list events in the market watch window       |
//+------------------------------------------------------------------+
void CEngine::MarketWatchEventsControl(void)
  {
   if(this.IsTester())
      return;
   this.m_symbols.MarketWatchEventsControl();
  }
//+------------------------------------------------------------------+
```

Here, if this is the tester, exit, otherwise, call the **MarketWatchEventsControl()** method of the symbol collection class for handling the market watch window events. We have already considered the method above when discussing tracking the symbol collection class events.

**Implementing the method returning the description of the last trading event:**

```
//+------------------------------------------------------------------+
//| Return the description of the last trading event                 |
//+------------------------------------------------------------------+
string CEngine::GetLastTradeEventDescription(void)
  {
   CArrayObj *list=this.m_events.GetList();
   if(list!=NULL)
     {
      if(list.Total()==0)
         return TextByLanguage("С момента последнего запуска ЕА торговых событий не было","There have been no trade events since the last launch of EA");
      CEvent *event=list.At(list.Total()-1);
      if(event!=NULL)
         return event.TypeEventDescription();
     }
   return DFUN_ERR_LINE+TextByLanguage("Не удалось получить описание последнего торгового события","Failed to get the description of the last trading event");
  }
//+------------------------------------------------------------------+
```

Here, we get the full list of trading events on an account. If the list is obtained but its size is zero, return the 'no trading events yet' message, otherwise, receive the last event from the list and return its description. In the contrary case, return the message about unsuccessful obtaining of a trading event.

**The method returning the description of the last event in the symbol collection:**

```
//+------------------------------------------------------------------+
//| Return the symbol event description                              |
//+------------------------------------------------------------------+
string CEngine::GetSymbolEventDescription(ENUM_SYMBOL_EVENT event)
  {
   CArrayObj *list=this.m_symbols.GetList();
   if(list!=NULL)
     {
      if(list.Total()==0)
         return TextByLanguage("С момента последнего запуска ЕА не было никаких событий символов","There have been no events of symbols since the last launch of EA");
      CSymbol *symbol=list.At(list.Total()-1);
      if(symbol!=NULL)
         return symbol.EventDescription(event);
     }
   return DFUN_ERR_LINE+TextByLanguage("Не удалось получить описание события символа","Failed to get symbol's event description");
  }
//+------------------------------------------------------------------+
```

The method works similarly to the method of returning the last trading event we have just considered.

**Improvement of the CEngine class is complete. All is ready for testing the symbol events, as well as the updated account class and account events.**

Some additional changes have been made to the classes. They have not been considered here as they are mostly related to names of some methods. They have been introduced to ensure that the same-type methods of different classes have the same names whenever appropriate. I believe, there is no point in describing all these minor improvements in the article since the library code is constantly evolving. You can always find them in the files attached to the articles.

### Testing symbol and account events

For testing, we will use the test EA [from the previous article](https://www.mql5.com/en/articles/7041#node04), save it in \\MQL5\\Experts\\TestDoEasy\ under the name **Part16\\TestDoEasyPart16.mq5** and add all the necessary changes.

**In the list of global variables, add the variable for storing the mode of working with the symbol lists:**

```
//--- global variables
CEngine        engine;
#ifdef __MQL5__
CTrade         trade;
#endif
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ulong          magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           slippage;
bool           trailing_on;
double         trailing_stop;
double         trailing_step;
uint           trailing_start;
uint           stoploss_to_modify;
uint           takeprofit_to_modify;
int            used_symbols_mode;
string         used_symbols;
string         array_used_symbols[];
```

When selecting the "Working with the full list of symbols on the server" mode of working with symbols, the first launch may take quite a long time because the symbol collection needs to collect all data on all existing symbols. Therefore, we need to warn users about that. There is no point in doing that in the library itself since it only does what users ask it to do. Thus, the warning should be made in the OnInit() handler of the program.

Let's do it the following way. If working with the full list of symbols available on the server is selected in the EA settings, the program displays the [MessageBox()](https://www.mql5.com/en/docs/common/messagebox) function standard window containing the warning

![](https://c.mql5.com/2/37/terminal64_qng6KJBD14.png)

prompting to select Yes for downloading the full list of symbols and No for working with the current symbol only. A user will only have to make a choice: click Yes and wait for the collection to be created from all available symbols, or click No and work with the current one.

**Let's prepare the check with the ability to send the question in the EA's OnInit() handler:**

```
//--- Check if working with the full list is selected
   used_symbols_mode=InpModeUsedSymbols;
   if((ENUM_SYMBOLS_MODE)used_symbols_mode==SYMBOLS_MODE_ALL)
     {
      int total=SymbolsTotal(false);
      string ru_n="\nКоличество символов на сервере "+(string)total+".\nМаксимальное количество: "+(string)SYMBOLS_COMMON_TOTAL+" символов.";
      string en_n="\nThe number of symbols on server "+(string)total+".\nMaximal number: "+(string)SYMBOLS_COMMON_TOTAL+" symbols.";
      string caption=TextByLanguage("Внимание!","Attention!");
      string ru="Выбран режим работы с полным списком.\nВ этом режиме первичная подготовка списка коллекции символов может занять длительное время."+ru_n+"\nПродолжить?\n\"Нет\" - работа с текущим символом \""+Symbol()+"\"";
      string en="Full list mode selected.\nIn this mode, the initial preparation of the collection symbols list may take a long time."+en_n+"\nContinue?\n\"No\" - working with the current symbol \""+Symbol()+"\"";
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
```

Here, the mode of working with symbols selected by a user in the EA settings is assigned to the **used\_symbols\_mode** global variable.

If working with the full list is selected, create the warning window text and display that window on the screen. Next, check what button was clicked by the user. If No, the mode of working with the current symbol is assigned to the **used\_symbols\_mode** variable.

In the remaining cases (Yes button or Esc), leave the mode of working with the full list of available symbols.

Next, create the array of used symbols (send the **used\_symbols\_mode** variable to the array creation function):

```
//--- Fill in the array of used symbols
   used_symbols=InpUsedSymbols;
   CreateUsedSymbolsArray((ENUM_SYMBOLS_MODE)used_symbols_mode,used_symbols,array_used_symbols);
```

set the type of the used list (mode of working with symbols) in the library and send the message about the applied mode of working with symbols to the journal:

```
//--- Set the type of the used symbol list in the symbol collection
   engine.SetUsedSymbols(array_used_symbols);
//--- Displaying the selected mode of working with the symbol object collection
   Print(engine.ModeSymbolsListDescription(),TextByLanguage(". Количество используемых символов: ",". Number of symbols used: "),engine.GetSymbolsCollectionTotal());
```

**The block of code for the fast check of a symbol collection is removed from the OnInit() handler since it is not needed in this test EA:**

```
//--- Fast check of the symbol object collection
   CArrayObj *list=engine.GetListAllUsedSymbols();
   CSymbol *symbol=NULL;
   if(list!=NULL)
     {
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         symbol.Refresh();
         symbol.RefreshRates();
         symbol.PrintShort();
         if(InpModeUsedSymbols<SYMBOLS_MODE_MARKET_WATCH)
            symbol.Print();
        }
     }
```

**Add the variable for storing the last event in the symbol collection to the OnTick() handler and write (or change in case of account events) the account and symbol collection events handling blocks:**

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Initializing the last events
   static ENUM_TRADE_EVENT last_trade_event=WRONG_VALUE;
   static ENUM_ACCOUNT_EVENT last_account_event=WRONG_VALUE;
   static ENUM_SYMBOL_EVENT last_symbol_event=WRONG_VALUE;
//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();
      PressButtonsControl();
     }
//--- If the last trading event changed
   if(engine.LastTradeEvent()!=last_trade_event)
     {
      last_trade_event=engine.LastTradeEvent();
      Comment("\nLast trade event: ",engine.GetLastTradeEventDescription());
      engine.ResetLastTradeEvent();
     }
//--- If there is an account event
   if(engine.IsAccountsEvent())
     {
      //--- the last account event
      last_account_event=engine.LastAccountEvent();
      //--- If this is a tester
      if(MQLInfoInteger(MQL_TESTER))
        {
         //--- Get the list of all account events occurred simultaneously
         CArrayObj* list=engine.GetListAccountEvents();
         if(list!=NULL)
           {
            //--- Get the next event in a loop
            int total=list.Total();
            for(int i=0;i<total;i++)
              {
               //--- take an event from the list
               CEventBaseObj *event=list.At(i);
               if(event==NULL)
                  continue;
               //--- Send an event to the event handler
               long lparam=event.LParam();
               double dparam=event.DParam();
               string sparam=event.SParam();
               OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
              }
           }
        }
     }
//--- If there is a symbol collection event
   if(engine.IsSymbolsEvent())
     {
      //--- the last event in the symbol collection
      last_symbol_event=engine.LastSymbolsEvent();
      //--- If this is a tester
      if(MQLInfoInteger(MQL_TESTER))
        {
         //--- Get the list of all symbol events occurred simultaneously
         CArrayObj* list=engine.GetListSymbolsEvents();
         if(list!=NULL)
           {
            //--- Get the next event in a loop
            int total=list.Total();
            for(int i=0;i<total;i++)
              {
               //--- take an event from the list
               CEventBaseObj *event=list.At(i);
               if(event==NULL)
                  continue;
               //--- Send an event to the event handler
               long lparam=event.LParam();
               double dparam=event.DParam();
               string sparam=event.SParam();
               OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
              }
           }
        }
     }
//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();
      TrailingOrders();
     }
  }
//+------------------------------------------------------------------+
```

Here all is simple. All the necessary actions for handling account and symbol collection events are commented in the listing.

As you can see, event handling is now similar for all objects — both for an account and a symbol collection. It all boils down to receiving the list of events and sending each new event from the list to the EA's OnDoEasyEvent() handler processing the event libraries. This has been made possible due to changes in inheriting the library objects from the CBaseObj base object, in which handling the events of descendant objects has been implemented.

**In the OnDoEasyEvent() EA handler, add the code of handling symbol collection events:**

```
//+------------------------------------------------------------------+
//| Handling DoEasy library events                                   |
//+------------------------------------------------------------------+
void OnDoEasyEvent(const int id,
                   const long &lparam,
                   const double &dparam,
                   const string &sparam)
  {
   int idx=id-CHARTEVENT_CUSTOM;
   string event="::"+string(idx);
   int digits=Digits();
//--- Handling trading events
   if(idx>TRADE_EVENT_NO_EVENT && idx<TRADE_EVENTS_NEXT_CODE)
     {
      event=EnumToString((ENUM_TRADE_EVENT)ushort(idx));
      digits=(int)SymbolInfoInteger(sparam,SYMBOL_DIGITS);
     }
//--- Handling account events
   else if(idx>ACCOUNT_EVENT_NO_EVENT && idx<ACCOUNT_EVENTS_NEXT_CODE)
     {
      Print(TimeMSCtoString(lparam)," ",sparam,": ",engine.GetAccountEventDescription((ENUM_ACCOUNT_EVENT)idx));

      //--- if this is an equity increase
      if((ENUM_ACCOUNT_EVENT)idx==ACCOUNT_EVENT_EQUITY_INC)
        {
         //--- Close a position with the highest profit exceeding zero when the equity exceeds the value,
         //--- specified in the CAccountsCollection::InitControlsParams() method for
         //--- the m_control_equity_inc variable tracking the equity growth by 15 units (by default)
         //--- AccountCollection file, InitControlsParams() method, string 1199

         //--- Get the list of all open positions
         CArrayObj* list_positions=engine.GetListMarketPosition();
         //--- Select positions with the profit exceeding zero
         list_positions=CSelect::ByOrderProperty(list_positions,ORDER_PROP_PROFIT_FULL,0,MORE);
         if(list_positions!=NULL)
           {
            //--- Sort the list by profit considering commission and swap
            list_positions.Sort(SORT_BY_ORDER_PROFIT_FULL);
            //--- Get the position index with the highest profit
            int index=CSelect::FindOrderMax(list_positions,ORDER_PROP_PROFIT_FULL);
            if(index>WRONG_VALUE)
              {
               COrder* position=list_positions.At(index);
               if(position!=NULL)
                 {
                  //--- Get a ticket of a position with the highest profit and close the position by a ticket
                  #ifdef __MQL5__
                     trade.PositionClose(position.Ticket());
                  #else
                     PositionClose(position.Ticket(),position.Volume());
                  #endif
                 }
              }
           }
        }
     }

//--- Handling symbol events
   else if(idx>SYMBOL_EVENT_NO_EVENT && idx<SYMBOL_EVENTS_NEXT_CODE)
     {
      string name="";
      //--- Market Watch window event
      if(idx<SYMBOL_EVENT_TRADE_DISABLE)
        {
         string descr=engine.GetMWEventDescription((ENUM_SYMBOL_EVENT)idx);
         name=(idx==SYMBOL_EVENT_MW_SORT ? "" : ": "+sparam);
         Print(TimeMSCtoString(lparam)," ",descr,name);
        }
      //--- Symbol event
      else
        {
         CSymbol *symbol=engine.GetSymbolObjByName(sparam);
         if(symbol!=NULL)
           {
            string descr=": "+symbol.EventDescription((ENUM_SYMBOL_EVENT)ushort(idx));
            Print(TimeMSCtoString(lparam)," ",sparam,descr);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

Here we have only two options: handling Marlet Watch window events and handling collection symbols events.

For market watch window events,

create the necessary message and send it to the journal,

while for symbol events,

receive a symbol from the list by its name taken from the 'sparam' string event parameter, get the string description of an event from a symbol object, add it to the created text and send the text to the journal.

We will not implement any other additional actions for the tests.

**This concludes the changes of the test EA.**

Find the full EA listing in the attached files.

When launching the EA on a demo account, you will see entries related to the symbol properties changes after some time. For example, when launching the EA on the eve of opening a trading session on Monday, multiple entries about a changed spread of various symbols start appearing in the journal.

In the example below, the following messages about symbol spread changes appear in the journal within an hour for only four Market Watch symbols:

```
2019.07.15 04:02:24.167 TestDoEasyPart16 (EURUSD,H4)    Working with symbols from the "Market Watch" window. The number of symbols used: 4
2019.07.15 04:02:25.762 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:02:27.316 GBPUSD: Spread value in points decreased by -7 (351)
2019.07.15 04:02:31.259 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:02:32.676 USDCHF: Spread value in points increased by 4 (287)
2019.07.15 04:02:33.761 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:02:35.218 USDCHF: Spread value in points decreased by -4 (283)
2019.07.15 04:02:46.261 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:02:47.680 USDCHF: Spread value in points increased by 4 (287)
2019.07.15 04:02:48.761 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:02:50.222 USDCHF: Spread value in points decreased by -4 (283)
2019.07.15 04:02:53.760 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:02:55.305 USDCHF: Spread value in points increased by 4 (287)
2019.07.15 04:02:56.760 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:02:58.221 USDCHF: Spread value in points decreased by -4 (283)
2019.07.15 04:03:01.261 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:02.683 USDCHF: Spread value in points increased by 4 (287)
2019.07.15 04:03:03.760 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:05.226 USDCHF: Spread value in points decreased by -4 (283)
2019.07.15 04:03:16.260 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:17.673 USDCHF: Spread value in points increased by 4 (287)
2019.07.15 04:03:18.789 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:20.219 USDCHF: Spread value in points decreased by -4 (283)
2019.07.15 04:03:30.832 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:32.686 USDCHF: Spread value in points increased by 4 (287)
2019.07.15 04:03:33.819 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:35.219 USDCHF: Spread value in points decreased by -4 (283)
2019.07.15 04:03:38.820 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:39.926 USDCHF: Spread value in points increased by 4 (287)
2019.07.15 04:03:41.821 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:43.221 USDCHF: Spread value in points decreased by -4 (283)
2019.07.15 04:03:45.820 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:47.673 USDCHF: Spread value in points increased by 4 (287)
2019.07.15 04:03:48.836 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:50.234 USDCHF: Spread value in points decreased by -4 (283)
2019.07.15 04:03:50.865 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:03:52.598 USDCHF: Spread value in points increased by 51 (334)
2019.07.15 04:03:58.867 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:04:00.450 EURUSD: Spread value in points decreased by -42 (50)
2019.07.15 04:03:58.868 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:04:00.430 USDCHF: Spread value in points decreased by -96 (238)
2019.07.15 04:03:59.417 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:04:00.934 USDCHF: Spread value in points increased by 22 (260)
2019.07.15 04:03:59.912 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:04:01.431 USDCHF: Spread value in points decreased by -5 (255)
2019.07.15 04:04:35.445 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:04:36.984 GBPUSD: Spread value in points decreased by -112 (239)
2019.07.15 04:04:35.445 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:04:36.985 EURUSD: Spread value in points decreased by -7 (43)
2019.07.15 04:04:35.445 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:04:36.984 USDCHF: Spread value in points decreased by -127 (128)
2019.07.15 04:04:58.460 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:05:00.102 GBPUSD: Spread value in points decreased by -207 (32)
2019.07.15 04:04:58.959 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:05:00.696 EURUSD: Spread value in points decreased by -4 (39)
2019.07.15 04:05:01.006 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:05:02.697 EURUSD: Spread value in points increased by 3 (42)
2019.07.15 04:05:02.037 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:05:03.686 EURUSD: Spread value in points decreased by -32 (10)
```

... skipped multiple strings ...

```
2019.07.15 04:55:09.780 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:11.578 GBPUSD: Spread value in points decreased by -3 (29)
2019.07.15 04:55:09.780 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:11.478 USDCHF: Spread value in points increased by 4 (32)
2019.07.15 04:55:10.482 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:11.681 USDCHF: Spread value in points decreased by -3 (29)
2019.07.15 04:55:11.623 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:13.477 USDCHF: Spread value in points increased by 3 (32)
2019.07.15 04:55:12.111 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:13.884 USDCHF: Spread value in points decreased by -5 (27)
2019.07.15 04:55:13.626 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:15.275 USDCHF: Spread value in points increased by 4 (31)
2019.07.15 04:55:19.628 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:21.381 USDCHF: Spread value in points decreased by -3 (28)
2019.07.15 04:55:20.126 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:21.882 USDCHF: Spread value in points increased by 3 (31)
2019.07.15 04:55:28.659 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:30.292 EURUSD: Spread value in points increased by 3 (20)
2019.07.15 04:55:33.690 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:35.298 EURUSD: Spread value in points decreased by -3 (17)
2019.07.15 04:55:53.298 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:55.137 EURUSD: Spread value in points increased by 3 (20)
2019.07.15 04:55:53.826 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:55.643 EURUSD: Spread value in points decreased by -3 (17)
2019.07.15 04:55:54.906 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:56.632 USDCHF: Spread value in points decreased by -3 (28)
2019.07.15 04:55:55.912 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:57.536 USDCHF: Spread value in points increased by 4 (32)
2019.07.15 04:55:56.907 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:58.636 USDCHF: Spread value in points decreased by -4 (28)
2019.07.15 04:55:57.434 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:55:58.832 USDCHF: Spread value in points increased by 4 (32)
2019.07.15 04:55:59.949 TestDoEasyPart16 (EURUSD,H4)    2019.07.15 00:56:01.538 USDCHF: Spread value in points decreased by -3 (29)
```

Now let's launch the EA in the tester with two symbols and see what entries it will show.

In the tester settings, for the **Mode of used symbols list** EA input parameter, select _"Working with the specified symbol list"_ from the drop-down list, while in the **List of used symbols (comma - separator)** parameter, enter two comma-separated symbols: _EURUSD,GBPUSD_ and launch the visual EA test:

![](https://c.mql5.com/2/36/VH1c2L43b2.gif)

The entries about the events of both symbols (in particular, the ones related to spread changes in the used symbols) are sent to the journal. When changing the account properties (in our case, it is an increase of the current profit), the appropriate entries are sent to the journal and profitable positions are closed.

### What's next?

In the next article, we will implement a convenient access to changing the values of controlled and tracked object properties from the program based on the CBaseObj base object class.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7071#node00)

**Previous articles within the series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part 2. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part 6. Netting account events](https://www.mql5.com/en/articles/6383)

[Part 7. StopLimit order activation events, preparing the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part 8. Order and position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 - Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and activating pending orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure events](https://www.mql5.com/en/articles/6921)

[Part 12. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part 13. Account object events](https://www.mql5.com/en/articles/6995)

[Part 14. Symbol object](https://www.mql5.com/en/articles/7014)

[Part 15. Symbol object collection](https://www.mql5.com/en/articles/7041)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7071](https://www.mql5.com/ru/articles/7071)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7071.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7071/mql5.zip "Download MQL5.zip")(224.58 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7071/mql4.zip "Download MQL4.zip")(224.58 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/325531)**
(3)


![Dmitiry Ananiev](https://c.mql5.com/avatar/2021/5/60A1913E-6AF5.jpg)

**[Dmitiry Ananiev](https://www.mql5.com/en/users/dimeon)**
\|
17 Jul 2019 at 09:11

Maybe someone will learn something from these codes. But what is the use of this bible?

Maybe it would be better to attach some test robot that it would be clear why all this clutter?


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
17 Jul 2019 at 09:21

**Dmitiry Ananiev:**

Maybe someone will learn something from these codes. But what is the use of this bible?

Maybe it would be better to attach a test robot so that it would be clear why all this clutter?

And the application will be later. Now the collection of necessary data is being realised. And test advisors are attached to each article. Further it will be organised simple access to any available data, and this data can also report about changes in its state. The user's task is to react to the message about the change of state. Or to request the necessary information and immediately receive and process it.

You see, in order to "attach a test robot", you need to know exactly what to show in it out of the many possibilities. You didn't ask "how to get", you asked "why".....

![MQL_User](https://c.mql5.com/avatar/avatar_na2.png)

**[MQL\_User](https://www.mql5.com/en/users/mql_user)**
\|
12 Jul 2022 at 20:50

Tried it on a [real account](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 documentation: Account Properties"). When working with the current symbol (BR-8.22) works fine. But what is strange is that when working with the specified list, I took the names of symbols from the "Market Watch" window and BR-8.22 futures does not recognise (Number of symbols used is 0), but @BR does.


![Strategy builder based on Merrill patterns](https://c.mql5.com/2/37/Article_Logo.png)[Strategy builder based on Merrill patterns](https://www.mql5.com/en/articles/7218)

In the previous article, we considered application of Merrill patterns to various data, such as to a price value on a currency symbol chart and values of standard MetaTrader 5 indicators: ATR, WPR, CCI, RSI, among others. Now, let us try to create a strategy construction set based on Merrill patterns.

![Developing a cross-platform grid EA (Last part): Diversification as a way to increase profitability](https://c.mql5.com/2/37/mql5_ea_adviser_grid.png)[Developing a cross-platform grid EA (Last part): Diversification as a way to increase profitability](https://www.mql5.com/en/articles/7219)

In previous articles within this series, we tried various methods for creating a more or less profitable grid Expert Advisor. Now we will try to increase the EA profitability through diversification. Our ultimate goal is to reach 100% profit per year with the maximum balance drawdown no more than 20%.

![Library for easy and quick development of MetaTrader programs (part XVII): Interactivity of library objects](https://c.mql5.com/2/36/MQL5-avatar-doeasy__12.png)[Library for easy and quick development of MetaTrader programs (part XVII): Interactivity of library objects](https://www.mql5.com/en/articles/7124)

In this article, we are going to finish the development of the base object of all library objects, so that any library object based on it is able to interact with a user. For example, users will be able to set the maximum acceptable size of a spread for opening a position and a price level, upon reaching which an event from a symbol object is sent to the program with the spread or price level-based signal.

![MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://c.mql5.com/2/37/custom_stress_test.png)[MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://www.mql5.com/en/articles/7166)

The article considers an approach to stress testing of a trading strategy using custom symbols. A custom symbol class is created for this purpose. This class is used to receive tick data from third-party sources, as well as to change symbol properties. Based on the results of the work done, we will consider several options for changing trading conditions, under which a trading strategy is being tested.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=bivbyhvkfiutxgqljwafhogsasfjnxqn&ssn=1769186378443834121&ssn_dr=0&ssn_sr=0&fv_date=1769186378&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7071&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XVI)%3A%20Symbol%20collection%20events%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918637838148563&fz_uniq=5070462373656991305&sv=2552)

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