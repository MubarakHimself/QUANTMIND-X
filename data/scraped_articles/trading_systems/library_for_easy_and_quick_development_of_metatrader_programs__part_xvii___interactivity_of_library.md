---
title: Library for easy and quick development of MetaTrader programs (part XVII): Interactivity of library objects
url: https://www.mql5.com/en/articles/7124
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:39:29.611344
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=mxydgzriykzwughtkhlrynlpwgnkeppo&ssn=1769186366753207090&ssn_dr=0&ssn_sr=0&fv_date=1769186366&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7124&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XVII)%3A%20Interactivity%20of%20library%20objects%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918636667487538&fz_uniq=6383576078116394558&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Control methods of the library base object events](https://www.mql5.com/en/articles/7124#node01)
- [Revamping the symbol class and the symbol collection](https://www.mql5.com/en/articles/7124#node02)
- [Testing the event functionality of the base object of all library objects](https://www.mql5.com/en/articles/7124#node03)
- [What's next?](https://www.mql5.com/en/articles/7124#node04)


In the [previous article](https://www.mql5.com/en/articles/7071), we created the base object of all library objects.
Now, any object inherited from the base one receives the event functionality, which allows us to easily track events occurring in the base
object's descendant class properties.

Today, we will go a bit further and endow the object (and therefore, all other library objects) with the ability to set what properties are
to be controlled externally in terms of their changes, change size and object property value level. Thus, all library objects will receive
the functionality allowing users to interact with library objects.

For example, suppose that we want to check a spread and a price level to open a position. We can easily set the controlled spread size, track
the price reaching the specified level and open a position. All we have to do is to programmatically set the spread size, below which trading
is possible, and a price level, upon reaching which an event from a symbol object is sent to the program allowing trading by the spread and the
price level.

Another important thing is that we get rid of the need to use event flags (which imposes limitations on tracking events and demands storing
enumeration lists of all possible event types for each object). Now the number of possible events will correspond to the number of object
properties — integer and real ones. The properties that should not be tracked are initialized by the

[LONG\_MAX](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants) value and do not participate in the
search for object events.

Since library objects are stored in their collections, updating object properties in the collection is performed in the library timer using
the Refresh() methods for the collections where Refresh() methods of objects stored in the collection list are called. If we track changes
of the descendant object in the Refresh() method of the base object, we are able to create a simple event model for each of the library objects.
Each of the objects send the list of its events to the

[CEngine](https://www.mql5.com/en/articles/5687#node02) library main object.

Thus, the library-based program is
always aware of all events that occurred in any object of any collection. Besides, we are always able to programmatically set and change the
size of a controlled value of any property for each object of any collection.

All this is achieved by the [simple class of the base object of all library objects](https://www.mql5.com/en/articles/7071).

### Control methods of the library base object events

Working with events of the library base objects is to be arranged as follows: previously, in order to define events of a certain class, we
implemented separate event control methods for it, created flags of events and enumeration of possible object events. Now that control for
descendant class events is to be arranged in their only base class, we need to implement a universal event control regardless of whether this
is a symbol event, an account event or an event of any other class that will be created later. Therefore, this is a suitable place for
controlling changes in object properties states (integer and real ones). Their list for each descendant class is unique and represents an

**event ID**. Also, we need to consider a property change direction — increasing or decreasing property values (let's call this an **event**
**reason**), as well as the **object property change value**. Event ID, reason and change value are to be written to a simple
class of the object base event and saved in the list of events that occurred simultaneously.

We have already agreed upon using an event with strictly specified parameters (event ID, 'long', 'double' and 'string' values) to send
events to the program. In the 'long' parameter, we sent an event time in milliseconds. Now we need to accurately define the event by some of its
parameters due to changes in definition of events:

1. Event ID — changed property of an object. Each object features its unique properties. The program knows nothing about an object the
    property has been changed in, as well as it knows nothing about the status of the changed property (whether it is integer or real),
    therefore it is impossible to accurately define it by an event ID.
2. Event reason — increasing or decreasing a property value or crossing a controlled level. This value also does not allow us to accurately
    determine an event. But the event ID and reason allow us to define that a certain object property has been increased or decreased, or has
    crossed a specified controlled value. Therefore, we need to specify the ID of the class in whose object the event has occurred to
    identify the event accurately.



    The collection list is ideally suited for that as it accurately defines the class the object belongs to — symbol, account or some
    other collection object created in the future. Therefore, the following data should additionally be sent to the event:



3. Collection ID — so that three above mentioned IDs allow us to define the event accurately.
4. Event string property — name of an object the event has occurred in.

Thus, in order to define an event, we need to obtain three integer parameters, as well as get the event name, which is also passed by the 'long'
value. We have only one 'long' event property. What should we do? The solution is simple: we are going to pass three 'ushort' type integer
events in a single 'long' parameter. The 'long' type has eight bytes, while 'ushort' has two bytes. So, the 'long' container allows us to
store three 'ushort' numbers written in bytes 0,1, bytes 2,3, bytes 4,5 of the 'long' number, and we still have two more bytes 6 and 7 for
passing yet another 'ushort' value if required later.

To define an event time, we only need to pass the time milliseconds in bytes 0 and 1 of the 'long' parameter.

- Event date and time can be taken from [TimeCurrent()](https://www.mql5.com/en/docs/dateandtime/timecurrent)
when obtaining an event and add the number of milliseconds passed in bytes 0 and 1 of the event 'long' value.

- The event reason is set in bytes 2 and 3 of the event 'long' parameter, while

- the class ID is set in the bytes 4 and 5 of the event 'long' parameter.


Thus, when receiving an event, we retrieve three 'ushort' values from the 'long' parameter to define an event time and get additional data for
the accurate event identification by event ID passed as the

**custom\_event\_id**'ushort' parameter to [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom), as
well as construct an accurate ID of the occurred event using the event ID and two values additionally retrieved from

**lparam**.

In the parent object timer, check the current state of each of the object properties and compare it with the previous state to define events in
the descendant object properties. First of all, check if the value to be compared with the property change one has been set. If the checked
value is not set (LONG\_MAX is set for it), this property is ignored.

Since we check the lists of object properties of 'long' and 'double' types, it is more reasonable to use two-dimensional arrays rather than the
structure to store the current and previous states of object properties. The array's first dimension is to store object property indices,
while the second one is to store the values of the property whose index has been set in the first dimension, property change value, as well as
controlled values and flags of the property events.

**Let me explain why it is more convenient to use arrays rather than the structure:**

We do not know in advance the
type of the property we are going to check. However, we can see its type in the property index ('double' object properties are always located
after 'long' ones). This means we will not need to duplicate fields in the structure for 'long' and 'double' values of the same object
property value. We will simply write the necessary data for controlling the object properties' states with their correct types to the array
of the necessary type (corresponding to the property type defined from the property index). Thus, there is no need to select the structure
field the passed value should be set into (long or double).

As soon as a change of any of the object properties is defined, add it to the list of basic object events (since the search is conducted in the
base object, the event is to be basic, it should not be confused with the descendant class event that is to be defined by the list of base events
and created out of base events, the pointers to which are stored in the list).

The lists of property changes (base event lists) are checked in the timer in the Refresh() methods of each base object's descendant class. If
the object lists contain these events, each event is transformed to the library event and sent to the controlling program.

To make the picture complete, we need to create the methods allowing us to programmatically set controlled change values for any property of
any library object based on the base object. This will allow us to quickly change the essential conditions of generating events from the
necessary objects at any time.

The set of all measures taken here and directed at improving the library base object allows us not to think about creating event control for all
subsequently created objects. Instead, we will use the ready-made functionality.

Let's get started.

Since we are to work with events in the base object of all library objects, we need to create the event reason enumeration to identify events.


In the \\MQL5\\Include\\DoEasy\

**Defines.mqh** file, after time options, **add the enumeration of possible**
**base object event reasons:**

```
//+------------------------------------------------------------------+
//| Possible options of selecting by time                            |
//+------------------------------------------------------------------+
enum ENUM_SELECT_BY_TIME
  {
   SELECT_BY_TIME_OPEN,                                     // By open time (in milliseconds)
   SELECT_BY_TIME_CLOSE,                                    // By close time (in milliseconds)
  };
//+------------------------------------------------------------------+
//| Possible event reasons of the object library base object         |
//+------------------------------------------------------------------+
enum ENUM_BASE_EVENT_REASON
  {
   BASE_EVENT_REASON_INC,                                   // Increase in the object property value
   BASE_EVENT_REASON_DEC,                                   // Decrease in the object property value
   BASE_EVENT_REASON_MORE_THEN,                             // Object property value exceeds the control value
   BASE_EVENT_REASON_LESS_THEN,                             // Object property value is less than the control value
   BASE_EVENT_REASON_EQUALS                                 // Object property value is equal to the control value
  };
//+------------------------------------------------------------------+
```

Since we no longer need event flags, replace the symbol event flag lists **with the list**
**of possible symbol events in the Market Watch window:**

```
//+------------------------------------------------------------------+
//| Data for working with symbols                                    |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of possible symbol events in the Market Watch window        |
//+------------------------------------------------------------------+
enum ENUM_MW_EVENT
  {
   MARKET_WATCH_EVENT_NO_EVENT = ACCOUNT_EVENTS_NEXT_CODE,  // No event
   MARKET_WATCH_EVENT_SYMBOL_ADD,                           // Adding a symbol to the Market Watch window
   MARKET_WATCH_EVENT_SYMBOL_DEL,                           // Removing a symbol from the Market Watch window
   MARKET_WATCH_EVENT_SYMBOL_SORT,                          // Sorting symbols in the Market Watch window
  };
#define SYMBOL_EVENTS_NEXT_CODE  (MARKET_WATCH_EVENT_SYMBOL_SORT+1)  // The code of the next event after the last symbol event code
//+------------------------------------------------------------------+
```

Remove **the list of possible symbol events since it is no**
**longer needed:**

```
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

The code of the next event has been replaced with the value
following the

MARKET\_WATCH\_EVENT\_SYMBOL\_SORT constant from the ENUM\_MW\_EVENT
enumeration.

**Now let's implement the planned functionality.**

In the \\MQL5\\Include\\DoEasy\\Objects\ **BaseObj.mqh** base object file, add the base event's new class:

```
//+------------------------------------------------------------------+
//| Library object's base event class                                |
//+------------------------------------------------------------------+
class CBaseEvent : public CObject
  {
private:
   ENUM_BASE_EVENT_REASON  m_reason;
   int                     m_event_id;
   double                  m_value;
public:
   ENUM_BASE_EVENT_REASON  Reason(void)   const { return this.m_reason;    }
   int                     ID(void)       const { return this.m_event_id;  }
   double                  Value(void)    const { return this.m_value;     }
//--- Constructor
                           CBaseEvent(const int event_id,const ENUM_BASE_EVENT_REASON reason,const double value) : m_reason(reason),
                                                                                                                   m_event_id(event_id),
                                                                                                                   m_value(value){}
//--- Comparison method to search for identical event objects
   virtual int             Compare(const CObject *node,const int mode=0) const
                             {
                              const CBaseEvent *compared=node;
                              return
                                (
                                 this.Reason()>compared.Reason()  ?  1  :
                                 this.Reason()<compared.Reason()  ? -1  :
                                 this.ID()>compared.ID()          ?  1  :
                                 this.ID()<compared.ID()          ? -1  : 0
                                );
                             }
  };
//+------------------------------------------------------------------+
```

The private class section features the variables for storing event
reasons, event ID (matches the index of the changed object
property) and

event property change value.

The public class section features
the methods for returning the class member variables listed above.

The formal parameters of the class constructor receive the
property values. The passed values are immediately assigned to the appropriate class member variables in the initialization list.

Also, the class features the method of comparing two class objects
for searching in the

[list of dynamic pointers to objects](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) we
have discussed more than once.

Since we are to store the list of controlled object properties in two-dimensional arrays, let's add
the macro substitution pointing at the size of the second array dimension. In the private section of the class, declare the two
variables, in which we are to store the

number of integer and real
properties of the object to be inherited from the class (since the base class knows nothing about the numbers of the properties its
descendants have, and these numbers should be displayed explicitly).

Declare the method for filling in the property arrays and
searching for changes in the descendant object properties.

```
//+------------------------------------------------------------------+
//| Base object class for all library objects                        |
//+------------------------------------------------------------------+
#define  CONTROLS_TOTAL    (10)
class CBaseObj : public CObject
  {
private:
   int               m_long_prop_total;
   int               m_double_prop_total;
   //--- Fill in the object property array
   template<typename T> bool  FillPropertySettings(const int index,T &array[][CONTROLS_TOTAL],T &array_prev[][CONTROLS_TOTAL],int &event_id);
protected:
```

In the protected class section, declare the list for storing the pointers to
instances of the object base events, the variable for storing the event ID,

the first launch flag and the
variable for storing the descendant object type.

We have also added four
two-dimensional arrays for storing the properties and controlling their changes (the current and previous integer and real
descendant object properties), as well as the

method returning only milliseconds stored in the event time
(for MQL4, return 0, while for MQL5, return the remainder of dividing the time 'long' value by 1000).

Since the base class knows nothing on the number of descendant object properties, while the size should be set from the descendant
classes (where they are known), declare the

methods for setting and checking
sizes of the arrays:

```
protected:
   CArrayObj         m_list_events_base;                       // Object base event list
   CArrayObj         m_list_events;                            // Object event list
   MqlTick           m_tick;                                   // Tick structure for receiving quote data
   double            m_hash_sum;                               // Object data hash sum
   double            m_hash_sum_prev;                          // Object data hash sum during the previous check
   int               m_digits_currency;                        // Number of decimal places in an account currency
   int               m_global_error;                           // Global error code
   long              m_chart_id;                               // Control program chart ID
   bool              m_is_event;                               // Object event flag
   int               m_event_code;                             // Object event code
   int               m_event_id;                               // Event ID (equal to the object property value)
   string            m_name;                                   // Object name
   string            m_folder_name;                            // Name of the folder storing CBaseObj descendant objects
   bool              m_first_start;                            // First launch flag
   int               m_type;                                   // Object type (corresponds to the collection IDs)
//--- Data in the array cells
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
   bool              CheckControlDataArraySize(bool check_long=true);

//--- Set the (1) controlled value and (2) object property change value
   template<typename T> void  SetControlledValue(const int property,const T value);
   template<typename T> void  SetControlledChangedValue(const int property,const T value);

//--- Set the value of the pbject property controlled (1) increase, (2) decrease, (3) control level
   template<typename T> void  SetControlledValueINC(const int property,const T value);
   template<typename T> void  SetControlledValueDEC(const int property,const T value);
   template<typename T> void  SetControlledValueLEVEL(const int property,const T value);

//--- Set the flag of a property change exceeding the (1) increase and (2) decrease values
   template<typename T> void  SetControlledFlagINC(const int property,const T value);
   template<typename T> void  SetControlledFlagDEC(const int property,const T value);
//--- Set the flag of a property change (1) exceeding, (2) being less than the control level, (3) being equal to the level
   template<typename T> void  SetControlledFlagMORE(const int property,const T value);
   template<typename T> void  SetControlledFlagLESS(const int property,const T value);
   template<typename T> void  SetControlledFlagEQUAL(const int property,const T value);

//--- Return the set value of the controlled (1) integer and (2) real object properties increase
   long              GetControlledValueLongINC(const int property)         const { return this.m_long_prop_event[property][0];                           }
   double            GetControlledValueDoubleINC(const int property)       const { return this.m_double_prop_event[property-this.m_long_prop_total][0];  }
//--- Return the set value of the controlled (1) integer and (2) real object properties decrease
   long              GetControlledValueLongDEC(const int property)         const { return this.m_long_prop_event[property][1];                           }
   double            GetControlledValueDoubleDEC(const int property)       const { return this.m_double_prop_event[property-this.m_long_prop_total][1];  }
//--- Return the control level of object's (1) integer and (2) real properties
   long              GetControlledValueLongLEVEL(const int property)       const { return this.m_long_prop_event[property][2];                           }
   double            GetControlledValueDoubleLEVEL(const int property)     const { return this.m_double_prop_event[property-this.m_long_prop_total][2];  }
//--- Return the value of the object (1) integer and (2) real property
   long              GetControlledValueLong(const int property)            const { return this.m_long_prop_event[property][3];                           }
   double            GetControlledValueDouble(const int property)          const { return this.m_double_prop_event[property-this.m_long_prop_total][3];  }
//--- Return the change value of the controlled (1) integer and (2) real object property
   long              GetControlledChangedValueLong(const int property)     const { return this.m_long_prop_event[property][4];                           }
   double            GetControlledChangedValueDouble(const int property)   const { return this.m_double_prop_event[property-this.m_long_prop_total][4];  }

//--- Return the flag of an (1) integer and (2) real property value change exceeding the increase value
   long              GetControlledFlagLongINC(const int property)          const { return this.m_long_prop_event[property][5];                           }
   double            GetControlledFlagDoubleINC(const int property)        const { return this.m_double_prop_event[property-this.m_long_prop_total][5];  }
//--- Return the flag of an (1) integer and (2) real property value change exceeding the decrease value
   long              GetControlledFlagLongDEC(const int property)          const { return this.m_long_prop_event[property][6];                           }
   double            GetControlledFlagDoubleDEC(const int property)        const { return this.m_double_prop_event[property-this.m_long_prop_total][6];  }
//--- Return the flag of an (1) integer and (2) real property value increase exceeding the control level
   long              GetControlledFlagLongMORE(const int property)         const { return this.m_long_prop_event[property][7];                           }
   double            GetControlledFlagDoubleMORE(const int property)       const { return this.m_double_prop_event[property-this.m_long_prop_total][7];  }
//--- Return the flag of an (1) integer and (2) real property value decrease being less than the control level
   long              GetControlledFlagLongLESS(const int property)         const { return this.m_long_prop_event[property][8];                           }
   double            GetControlledFlagDoubleLESS(const int property)       const { return this.m_double_prop_event[property-this.m_long_prop_total][8];  }
//--- Return the flag of an (1) integer and (2) real property being equal to the control level
   long              GetControlledFlagLongEQUAL(const int property)        const { return this.m_long_prop_event[property][9];                           }
   double            GetControlledFlagDoubleEQUAL(const int property)      const { return this.m_double_prop_event[property-this.m_long_prop_total][9];  }

//--- (1) Pack a 'ushort' number to a passed 'long' number
//--- (2) convert a 'ushort' value to a specified 'long' number byte
   long              UshortToLong(const ushort ushort_value,const uchar index,long &long_value);
   long              UshortToByte(const ushort value,const uchar index) const;
public:
```

The methods for setting and returning controlled properties and their values,
and the

methods for packing a 'ushort' number to specified 'long' container bytes by an index are
declared in the private section of the class as well. ( **Index 0** =\> bytes **0-1**, **Index 1** =\>
bytes

**2-3**, **Index 2** =\> bytes **4-5**)

**In the public section of the class, declare the methods** of
resetting the values of changed properties and the values of controlled
object properties, the method of adding a base event to the list, the
method of receiving the base object from the list by an index, the method
returning the number of base objects in the list, the virtual method
returning an object type and the method returning a string description
of a base event:

```
public:
//--- Reset the variables of (1) tracked and (2) controlled object data (can be reset in the descendants)
   void              ResetChangesParams(void);
   virtual void      ResetControlsParams(void);
//--- Add the (1) object event and (2) the object event reason to the list
   bool              EventAdd(const ushort event_id,const long lparam,const double dparam,const string sparam);
   bool              EventBaseAdd(const int event_id,const ENUM_BASE_EVENT_REASON reason,const double value);
//--- Return the occurred event flag to the object data
   bool              IsEvent(void)                             const { return this.m_is_event;                 }
//--- Return (1) the list of events, (2) the object event code and (3) the global error code
   CArrayObj        *GetListEvents(void)                             { return &this.m_list_events;             }
   int               GetEventCode(void)                        const { return this.m_event_code;               }
   int               GetError(void)                            const { return this.m_global_error;             }
//--- Return (1) an event object and (2) a base event by its number in the list
   CEventBaseObj    *GetEvent(const int shift=WRONG_VALUE,const bool check_out=true);
   CBaseEvent       *GetEventBase(const int index);
//--- Return the number of (1) object events
   int               GetEventsTotal(void)                      const { return this.m_list_events.Total();      }
//--- (1) Set and (2) return the chart ID of the control program
   void              SetChartID(const long id)                       { this.m_chart_id=id;                     }
   long              GetChartID(void)                          const { return this.m_chart_id;                 }
//--- (1) Set the sub-folder name, (2) return the folder name for storing descendant object files
   void              SetSubFolderName(const string name)             { this.m_folder_name=DIRECTORY+name;      }
   string            GetFolderName(void)                       const { return this.m_folder_name;              }
//--- Return the object name
   string            GetName(void)                             const { return this.m_name;                     }
//--- Update the object data to search for changes (Calling from the descendants: CBaseObj::Refresh())
   virtual void      Refresh(void);
//--- Return an object type
   virtual int       Type(void)                                const { return this.m_type;                     }
//--- Return an object event description
   string            EventDescription(const int property,
                                      const ENUM_BASE_EVENT_REASON reason,
                                      const int source,
                                      const string value,
                                      const string property_descr,
                                      const int digits);
//--- Constructor
                     CBaseObj();
  };
//+------------------------------------------------------------------+
```

Now let's briefly review all the methods declared above.

**Class constructor:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CBaseObj::CBaseObj() : m_global_error(ERR_SUCCESS),
                       m_hash_sum(0),m_hash_sum_prev(0),
                       m_is_event(false),m_event_code(WRONG_VALUE),
                       m_chart_id(::ChartID()),
                       m_folder_name(DIRECTORY),
                       m_name(__FUNCTION__),
                       m_long_prop_total(0),
                       m_double_prop_total(0),
                       m_first_start(true)
  {
   ::ArrayResize(this.m_long_prop_event,0,100);
   ::ArrayResize(this.m_double_prop_event,0,100);
   ::ArrayResize(this.m_long_prop_event_prev,0,100);
   ::ArrayResize(this.m_double_prop_event_prev,0,100);
   ::ZeroMemory(this.m_tick);
   this.m_digits_currency=(#ifdef __MQL5__ (int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif);
   this.m_list_events.Clear();
   this.m_list_events.Sort();
   this.m_list_events_base.Clear();
   this.m_list_events_base.Sort();
  }
//+------------------------------------------------------------------+
```

Since we no longer have event codes we have previously gathered from flags where the zero value indicated the absence of an event, we need to
replace the event code initialization with a non-zero one in the class initialization list (since zero stands for the very first property in
the enumeration of object's integer properties, while positive values indicate the next events in the object property list, but we cannot
define their number in the parent class).

Let's set the event code to -1.

The
object name is initialized by a class name (the name is re-assigned in the descendants). Initialize the number
of integer and real properties of the descendant object by
zero values and

set the flag of the first launch.

In the class body, set
the size of arrays of integer and real properties to the zero value, clear the
list of base events and set the sorted list flag.

Previously, the **Refresh() virtual method** of the class was simply declared, and its implementation was to be performed by the
descendant classes. Now let's create and implement the method for the base object class to track changes of the descendant object
properties. When an event is defined, the base events are created and added to the list of base events for their subsequent handling and
creating object events to be sent to the program:

```
//+------------------------------------------------------------------+
//| Update the object data to search changes in them                 |
//| Call from descendants: CBaseObj::Refresh()                       |
//+------------------------------------------------------------------+
void CBaseObj::Refresh(void)
  {
//--- Check the size of the arrays, Exit if it is zero
   if(!this.CheckControlDataArraySize() || !this.CheckControlDataArraySize(false))
      return;
//--- Reset the event flag and clear all lists
   this.m_is_event=false;
   this.m_list_events.Clear();
   this.m_list_events.Sort();
   this.m_list_events_base.Clear();
   this.m_list_events_base.Sort();
//--- Fill in the array of integer properties and control their changes
   for(int i=0;i<this.m_long_prop_total;i++)
      if(!this.FillPropertySettings(i,this.m_long_prop_event,this.m_long_prop_event_prev,this.m_event_id))
         continue;
//--- Fill in the array of real properties and control their changes
   for(int i=0;i<this.m_double_prop_total;i++)
      if(!this.FillPropertySettings(i,this.m_double_prop_event,this.m_double_prop_event_prev,this.m_event_id))
         continue;
//--- First launch
   if(this.m_first_start)
     {
      ::ArrayCopy(this.m_long_prop_event_prev,this.m_long_prop_event);
      ::ArrayCopy(this.m_double_prop_event_prev,this.m_double_prop_event);
      this.m_hash_sum_prev=this.m_hash_sum;
      this.m_first_start=false;
      this.m_is_event=false;
      this.m_list_events_base.Clear();
      this.m_list_events_base.Sort();
      return;
     }
  }
//+------------------------------------------------------------------+
```

Here, all actions are described in the code comments, including the ones performed in the method, like preliminary clearing of event lists and
calling the methods for filling in the arrays of integer and real descendant object properties and checking their changes.

If this is the first launch, the current status of property arrays is copied to the previous status (to avoid the difference between them
leading to events registration), the first launch flag is reset and the list of base events possibly created while calling the

**FillPropertySettings()** methods is cleared.

**Implementing the method filling in the arrays of the descendant object properties and controlling their changes:**

```
//+------------------------------------------------------------------+
//| Fill in the object property array                                |
//+------------------------------------------------------------------+
template<typename T> bool CBaseObj::FillPropertySettings(const int index,T &array[][CONTROLS_TOTAL],T &array_prev[][CONTROLS_TOTAL],int &event_id)
  {
   //--- Data in the array cells
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

   //--- If controlled values are not set, exit with 'false'
   if(this.m_first_start)
      return false;
   //--- Set the shift of the 'double' property index and the event ID
   event_id=index+(typename(T)=="double" ? this.m_long_prop_total : 0);
   //--- Reset all event flags
   for(int j=5;j<CONTROLS_TOTAL;j++)
      array[index][j]=false;
   //--- Property change value
   T value=array[index][3]-array_prev[index][3];
   array[index][4]=value;
   //--- If the controlled property increase value is set
   if(array[index][0]<LONG_MAX)
     {
      //--- If the property change value exceeds the controlled increase value - there is an event,
      //--- add the event to the list, set the flag and save the new property value size
      if(value>0 && value>array[index][0])
        {
         if(this.EventBaseAdd(event_id,BASE_EVENT_REASON_INC,value))
           {
            array[index][5]=true;
            array_prev[index][4]=value;
           }
        }
     }
   //--- If the controlled property decrease value is set
   if(array[index][1]<LONG_MAX)
     {
      //--- If the property change value exceeds the controlled decrease value - there is an event,
      //--- add the event to the list, set the flag and save the new property value size
      if(value<0 && fabs(value)>array[index][1])
        {
         if(this.EventBaseAdd(event_id,BASE_EVENT_REASON_DEC,value))
           {
            array[index][6]=true;
            array_prev[index][4]=value;
           }
        }
     }
   //--- If the controlled level value is set
   if(array[index][2]<LONG_MAX)
     {
      value=array[index][3]-array[index][2];
      //--- If a property value exceeds the control level, there is an event
      //--- add the event to the list and set the flag
      if(value>0 && array_prev[index][3]<=array[index][2])
        {
         if(this.EventBaseAdd(event_id,BASE_EVENT_REASON_MORE_THEN,array[index][2]))
            array[index][7]=true;
        }
      //--- If a property value is less than the control level, there is an event,
      //--- add the event to the list and set the flag
      else if(value<0 && array_prev[index][3]>=array[index][2])
        {
         if(this.EventBaseAdd(event_id,BASE_EVENT_REASON_LESS_THEN,array[index][2]))
            array[index][8]=true;
        }
      //--- If a property value is equal to the control level, there is an event,
      //--- add the event to the list and set the flag
      else if(value==0 && array_prev[index][3]!=array[index][2])
        {
         if(this.EventBaseAdd(event_id,BASE_EVENT_REASON_EQUALS,array[index][2]))
            array[index][9]=true;
        }
     }
   //--- Save the current property value as a previous one
   array_prev[index][3]=array[index][3];
   return true;
  }
//+------------------------------------------------------------------+
```

Here, all actions are described in the code comments. The only thing I want to clarify is the
selection of an object 'double' property index shiftfor receiving an
event ID. Since the real properties of all objects are located after the integer properties, the beginning of the first real property
is equal to the number of integer properties (if the number of 'long' properties is equal to three, the first real property has the index of 3
(0,1,2,

**3**)). In arrays, the counting starts from zero. Therefore, when
working with 'double' properties, we need to add the number of object integer properties to the
array index.

**The methods of setting a size of arrays of integer and real descendant object properties:**

```
//+------------------------------------------------------------------+
//| Set the size of the arrays of the object integer properties      |
//+------------------------------------------------------------------+
bool CBaseObj::SetControlDataArraySizeLong(const int size)
  {
   int x=(#ifdef __MQL4__ CONTROLS_TOTAL #else 1 #endif );
   this.m_long_prop_total=::ArrayResize(this.m_long_prop_event,size,100)/x;
   return((::ArrayResize(this.m_long_prop_event_prev,size,100)/x)==size && this.m_long_prop_total==size ? true : false);
  }
//+------------------------------------------------------------------+
//| Set the size of the arrays of the object real properties         |
//+------------------------------------------------------------------+
bool CBaseObj::SetControlDataArraySizeDouble(const int size)
  {
   int x=(#ifdef __MQL4__ CONTROLS_TOTAL #else 1 #endif );
   this.m_double_prop_total=::ArrayResize(this.m_double_prop_event,size,100)/x;
   return((::ArrayResize(this.m_double_prop_event_prev,size,100)/x)==size && this.m_double_prop_total==size ? true : false);
  }
//+------------------------------------------------------------------+
```

The methods return the result of changing the size of arrays by the value passed to the method.

I should highlight one feature of changing a multidimensional array size
in MQL4. The [ArrayResize()](https://www.mql5.com/en/docs/array/arrayresize) function in MQL4 returns the
total size of all array dimensions. In MQL5, it returns the size of the first dimension, which is subject to change. For example, if the
second dimension's size is two, then, if the first dimension changes its size to 10, the function returns 20, which is not logical (since
we change the size of the first dimension only). In MQL5, the function returns the correct value. For the above example, it returns 10, as
expected.

Therefore, the methods feature the separator
the values returned by the function should be divided by in MQL4 (the
second dimension size).

**The method for checking the size of the array of the integer and real descendant object properties:**

```
//+------------------------------------------------------------------+
//| Check the array size of object properties                        |
//+------------------------------------------------------------------+
bool CBaseObj::CheckControlDataArraySize(bool check_long=true)
  {
   string txt1="";
   string txt2="";
   string txt3="";
   string txt4="";
   bool res=true;
   if(check_long)
     {
      if(this.m_long_prop_total==0)
        {
         txt1=TextByLanguage("Массив данных контролируемых integer-свойств имеет нулевой размер","Controlled integer properties data array has zero size");
         txt2=TextByLanguage("Необходимо сначала установить размер массива равным количеству integer-свойств объекта","You should first set size of array equal to number of object integer properties");
         txt3=TextByLanguage("Для этого используйте метод CBaseObj::SetControlDataArraySizeLong()","To do this, use CBaseObj::SetControlDataArraySizeLong() method");
         txt4=TextByLanguage("со значением количества integer-свойств объекта в параметре \"size\"","with value of number of integer properties of object in \"size\" parameter");
         res=false;
        }
     }
   else
     {
      if(this.m_double_prop_total==0)
        {
         txt1=TextByLanguage("Массив данных контролируемых double-свойств имеет нулевой размер","Controlled double properties data array has zero size");
         txt2=TextByLanguage("Необходимо сначала установить размер массива равным количеству double-свойств объекта","You should first set size of array equal to number of object double properties");
         txt3=TextByLanguage("Для этого используйте метод CBaseObj::SetControlDataArraySizeDouble()","To do this, use CBaseObj::SetControlDataArraySizeDouble() method");
         txt4=TextByLanguage("со значением количества double-свойств объекта в параметре \"size\"","with value of number of double properties of object in \"size\" parameter");
         res=false;
        }
     }
   if(res)
      return true;
   #ifdef __MQL5__
      ::Print(DFUN,"\n",txt1,"\n",txt2,"\n",txt3,"\n",txt4);
   #else
      ::Print(DFUN);
      ::Print(txt1);
      ::Print(txt2);
      ::Print(txt3);
      ::Print(txt4);
   #endif
   this.m_global_error=ERR_ZEROSIZE_ARRAY;
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the flag indicating the size of the checked
array.

If true, the array of 'long' properties is checked. If false,
the array of 'double' properties is checked instead.

If a size of the checked array is not set, the message text is generated, the message is displayed in the journal and false
is returned. If the array size has already been set, true is returned.

**The methods of resetting controlled values and change**
**values of tracked data of object properties:**

```
//+------------------------------------------------------------------+
//| Reset the variables of controlled object data values             |
//+------------------------------------------------------------------+
void CBaseObj::ResetControlsParams(void)
  {
   if(!this.CheckControlDataArraySize(true) || !this.CheckControlDataArraySize(false))
      return;
//--- Data in the array cells
//--- [Property index][0] Controlled property increase value
//--- [Property index][1] Controlled property decrease value
//--- [Property index][2] Controlled property value level
   for(int i=this.m_long_prop_total-1;i>WRONG_VALUE;i--)
      for(int j=0; j<3; j++)
         this.m_long_prop_event[i][j]=LONG_MAX;
   for(int i=this.m_double_prop_total-1;i>WRONG_VALUE;i--)
      for(int j=0; j<3; j++)
         this.m_double_prop_event[i][j]=(double)LONG_MAX;
  }
//+------------------------------------------------------------------+
//| Reset the variables of tracked object data                       |
//+------------------------------------------------------------------+
void CBaseObj::ResetChangesParams(void)
  {
   if(!this.CheckControlDataArraySize(true) || !this.CheckControlDataArraySize(false))
      return;
   this.m_list_events.Clear();
   this.m_list_events.Sort();
   this.m_list_events_base.Clear();
   this.m_list_events_base.Sort();
//--- Data in the array cells
//--- [Property index][3] Property value
//--- [Property index][4] Property value change
//--- [Property index][5] Flag of a property change exceeding the increase value
//--- [Property index][6] Flag of a property change exceeding the decrease value
//--- [Property index][7] Flag of a property increase exceeding the control level
//--- [Property index][8] Flag of a property decrease being less than the control level
//--- [Property index][9] Flag of a property value being equal to the controlled value
   for(int i=this.m_long_prop_total-1;i>WRONG_VALUE;i--)
      for(int j=3; j<CONTROLS_TOTAL; j++)
         this.m_long_prop_event[i][j]=(j<5 ? LONG_MAX : 0);
   for(int i=this.m_double_prop_total-1;i>WRONG_VALUE;i--)
      for(int j=3; j<CONTROLS_TOTAL; j++)
         this.m_double_prop_event[i][j]=(j<5 ? (double)LONG_MAX : 0);
  }
//+------------------------------------------------------------------+
```

The initializing values are set in the necessary cells of the arrays' second dimensions in the methods in two loops by arrays of the descendant
object integer and real properties. The initialized cells are set in the code comments.

**The method adding the base event to the list of the object base events:**

```
//+------------------------------------------------------------------+
//| Add the object base event to the list                            |
//+------------------------------------------------------------------+
bool CBaseObj::EventBaseAdd(const int event_id,const ENUM_BASE_EVENT_REASON reason,const double value)
  {
   CBaseEvent* event=new CBaseEvent(event_id,reason,value);
   if(event==NULL)
      return false;
   this.m_list_events_base.Sort();
   if(this.m_list_events_base.Search(event)>WRONG_VALUE)
     {
      delete event;
      return false;
     }
   return this.m_list_events_base.Add(event);
  }
//+------------------------------------------------------------------+
```

The method receives the event ID, event
reason and the descendant object property change value.

Next, a new base event is created, and if
the same event is already present in the list of base events, it is removed and

false is returned — the event
is not added. Otherwise,

the result of adding a new event to the list of base object events is returned.

**The method returning the base event by its index in the list of base object events:**

```
//+------------------------------------------------------------------+
//| Return a base event by its index in the list                     |
//+------------------------------------------------------------------+
CBaseEvent *CBaseObj::GetEventBase(const int index)
  {
   int total=this.m_list_events_base.Total();
   if(total==0 || index<0 || index>total-1)
      return NULL;
   CBaseEvent *event=this.m_list_events_base.At(index);
   return(event!=NULL ? event : NULL);
  }
//+------------------------------------------------------------------+
```

The method receives the index of the required event. If
the list has the zero size or the index extends beyond the base event list,

NULL is returned. Otherwise, receive
the event from the list by an index and return the pointer to the obtained
object.

We need to create the methods for the object base class to quickly set the necessary property changes. Exceeding the changes leads to events
generation. We also need the methods for setting new values of descendant object properties and returning flags about occurred
"controlled" object events. Since the base class knows nothing about the properties of its descendants, we need to create the universal
methods allowing us to make changes to the necessary descendant object property. Since we are going to indicate the number of integer and
real properties for each of the descendant classes, it is easy to define the property we set the value for. We should simply check the changed
property index. If the index is less than the number of integer properties, the changes are made to the object integer property, otherwise
the changes are made to the real property.

**Implementing the methods of setting descendant object controlled properties:**

```
//+------------------------------------------------------------------+
//| Methods of setting controlled parameters                         |
//+------------------------------------------------------------------+
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
//+------------------------------------------------------------------+
//| Set the value of the controlled increase of object properties    |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledValueINC(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][0]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][0]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the value of the controlled decrease of object properties    |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledValueDEC(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][1]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][1]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the control level of object properties                       |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledValueLEVEL(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][2]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][2]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the object property value                                    |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledValue(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][3]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][3]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the object property change value			     |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledChangedValue(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][4]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][4]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the flag of the property value change                        |
//| exceeding the increase value                                     |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledFlagINC(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][5]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][5]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the flag of the property value change                        |
//| exceeding the decrease value                                     |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledFlagDEC(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][6]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][6]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the flag of the property value increase                      |
//| exceeding the control level                                      |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledFlagMORE(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][7]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][7]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the flag of the property value decrease                      |
//| being less than the control level                                |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledFlagLESS(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][8]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][8]=(double)value;
  }
//+------------------------------------------------------------------+
//| Set the flag of the property value being equal to the            |
//| control level                                                    |
//+------------------------------------------------------------------+
template<typename T> void CBaseObj::SetControlledFlagEQUAL(const int property,const T value)
  {
   if(property<this.m_long_prop_total)
      this.m_long_prop_event[property][9]=(long)value;
   else
      this.m_double_prop_event[property-this.m_long_prop_total][9]=(double)value;
  }
//+------------------------------------------------------------------+
```

Let's have a look at the last method receiving the property
the template

T value is to be added to. If
the property index is less than the number of descendant object integer properties, the
T value is added to the necessary cell of the array of object integer properties, otherwise calculate
the index, by which the property is stored in the real properties array (the 'double' index of the property always exceeds the index
of the same property by the number of object integer properties) and

add the T value to the necessary cell of the array of object real properties.
The required cells of the array's second dimension for each of the methods are listed before the method list.

**The method converting a 'ushort' value to a 'long' one shifted by the necessary number of bytes for its subsequent packing to a 'long'**
**container:**

```
//+------------------------------------------------------------------+
//| Convert a 'ushort' value to a specified 'long' number byte       |
//+------------------------------------------------------------------+
long CBaseObj::UshortToByte(const ushort value,const uchar index) const
  {
   if(index>3)
     {
      ::Print(DFUN,TextByLanguage("Ошибка. Значение \"index\" должно быть в пределах 0 - 3","Error. \"index\" value should be between 0 - 3"));
      return 0;
     }
   return(long)value<<(16*index);
  }
//+------------------------------------------------------------------+
```

Suppose that we have an eight-byte 'long' value divided into cells of two bytes each (a unique index is assigned to each cell of that kind):

| Bytes 6-7 (index 3) | Bytes 4-5 (index 2) | Bytes 2-3 (index 1) | Bytes 0-1 (index 0) |
| --- | --- | --- | --- |
| ushort 4 | ushort 3 | ushort 2 | ushort 1 |

We can place four 'ushort' number to it. Each subsequent number should be shifted to the left by 16 bits \* index (1 byte = 8 bits). Next, the
obtained value is added to the 'long' number. Thus, we obtain a few 'ushort' values packed into the 'long' container.

The method receives a 'ushort' number and the index,
by which a 'ushort' value should be stored in the 'long' container.


The index is checked, and if it exceeds 3, the incorrect index message is displayed and 0 is returned.

If
the index is correct, a

'ushort' number is shifted to the left by 16
bits \* index (one byte contains 8 bits, while we need to shift the two-byte 'ushort' number) and
the shift result is returned from the method.

**The method packing the 'ushort' value shifted by the necessary number of bytes to a 'long' container:**

```
//+------------------------------------------------------------------+
//| Pack a 'ushort' number to a passed 'long' number                 |
//+------------------------------------------------------------------+
long CBaseObj::UshortToLong(const ushort ushort_value,const uchar index,long &long_value)
  {
   if(index>3)
     {
      ::Print(DFUN,TextByLanguage("Ошибка. Значение \"index\" должно быть в пределах 0 - 3","Error. \"index\" value should be between 0 - 3"));
      return 0;
     }
   return(long_value |= UshortToByte(ushort_value,index));
  }
//+------------------------------------------------------------------+
```

The method receives a 'ushort' number that should be packed to a 'long'
container passed to the method by the link and the index of bytes the
'ushort' value in the 'long' container should be placed to.

Like in the method described above, the
index is checked, and if the check is successful, the 'ushort'
value

shifted by the necessary number of bytes using the UshortToByte() method is
added to the 'long' number

using bitwise "OR", and the result is returned to the calling program.

**The method returning the string description of the descendant object's event:**

```
//+------------------------------------------------------------------+
//| Return an object event description                               |
//+------------------------------------------------------------------+
string CBaseObj::EventDescription(const int property,
                                  const ENUM_BASE_EVENT_REASON reason,
                                  const int source,
                                  const string value,
                                  const string property_descr,
                                  const int digits)
  {
//--- Depending on the collection ID, create th object type description
   string type=
     (
      this.Type()==COLLECTION_SYMBOLS_ID ? TextByLanguage("символа: ","symbol property: ")   :
      this.Type()==COLLECTION_ACCOUNT_ID ? TextByLanguage("аккаунта: ","account property: ")   :
      ""
     );
//--- Depending on the property type, create the property change value description
   string level=
     (
      property<this.m_long_prop_total ?
      ::DoubleToString(this.GetControlledValueLongLEVEL(property),digits) :
      ::DoubleToString(this.GetControlledValueDoubleLEVEL(property),digits)
     );
//--- Depending on the event reason, create the event description text
   string res=
     (
      reason==BASE_EVENT_REASON_INC       ?  TextByLanguage("Значение свойства ","Value of the ")+type+property_descr+TextByLanguage(" увеличено на "," increased by ")+value       :
      reason==BASE_EVENT_REASON_DEC       ?  TextByLanguage("Значение свойства ","Value of the ")+type+property_descr+TextByLanguage(" уменьшено на "," decreased by ")+value       :
      reason==BASE_EVENT_REASON_MORE_THEN ?  TextByLanguage("Значение свойства ","Value of the ")+type+property_descr+TextByLanguage(" стало больше "," became more than ")+level   :
      reason==BASE_EVENT_REASON_LESS_THEN ?  TextByLanguage("Значение свойства ","Value of the ")+type+property_descr+TextByLanguage(" стало меньше "," became less than ")+level   :
      reason==BASE_EVENT_REASON_EQUALS    ?  TextByLanguage("Значение свойства ","Value of the ")+type+property_descr+TextByLanguage(" равно "," is equal to ")+level               :
      TextByLanguage("Неизвестное событие ","Unknown ")+type
     );
//--- Return the object name+created event description text
   return this.m_name+": "+res;
  }
//+------------------------------------------------------------------+
```

Since the base object class knows nothing about its descendants, we need to specify the descendant objects the event has occurred at to describe
an event in the descendant class.

To achieve this, the method receives

- a property of the object the event has been detected
in,


- an event reason — increase/decrease of a property value by a
specified value/crossing of a specified level by the property value,



- event source — ID of the collection in whose object an event has
occurred,



- the value, by which the object property was changed,
- the text description of the descendant object property
(available in the descendant) and



- the number of decimal places in the numerical
representation of the changed property (also available in the descendant).

All steps of creating a descendant object are described in the code comments. I believe, they are comprehensive enough.

**We have made all the necessary changes in the base object class** (during the further library development and creating
new collections, new collection IDs are to be added to the last method to create a correct event description).

### Revamping the symbol class and the symbol collection

Here, we are going to revise the symbol class and the symbol collection operation with the new base object events in mind. Let's make some changes
in the symbol and symbol collection classes.

Open \\MQL5\\Include\\DoEasy\\Objects\\Symbols\ **Symbol.mqh** and add the changes.

Since all descendant object events of the base object are now defined in the parent class, there is no need to control the changes of object
properties in the descendant class. Therefore, the tracked object properties data structure is unnecessary now.

**Let's remove the structure and two objects with the structure type from the symbol object class:**

```
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

**Remove all class member variables for storing controlled and changed symbol object properties** **— now all this**
**data is stored in the base object class arrays:**

```
   //--- Current session deals
   long              m_control_session_deals_inc;              // Controlled value of the growth of the number of deals
   long              m_control_session_deals_dec;              // Controlled value of the decrease in the number of deals
   long              m_changed_session_deals_value;            // Value of change in the number of deals
   bool              m_is_change_session_deals_inc;            // Flag of a change in the number of deals exceeding the growth value
   bool              m_is_change_session_deals_dec;            // Flag of a change in the number of deals exceeding the decrease value
   //--- Buy orders of the current session
   long              m_control_session_buy_ord_inc;            // Controlled value of the increase of the number of Buy orders
   long              m_control_session_buy_ord_dec;            // Controlled value of the decrease in the number of Buy orders
   long              m_changed_session_buy_ord_value;          // Buy orders change value
   bool              m_is_change_session_buy_ord_inc;          // Flag of a change in the number of Buy orders exceeding the increase value
   bool              m_is_change_session_buy_ord_dec;          // Flag of a change in the number of Buy orders being less than the increase value
   //--- Sell orders of the current session
   long              m_control_session_sell_ord_inc;           // Controlled value of the increase of the number of Sell orders
   long              m_control_session_sell_ord_dec;           // Controlled value of the decrease in the number of Sell orders
   long              m_changed_session_sell_ord_value;         // Sell orders change value
   bool              m_is_change_session_sell_ord_inc;         // Flag of a change in the number of Sell orders exceeding the increase value
   bool              m_is_change_session_sell_ord_dec;         // Flag of a change in the number of Sell orders exceeding the decrease value
   //--- Volume of the last deal
   long              m_control_volume_inc;                     // Controlled value of the volume increase in the last deal
   long              m_control_volume_dec;                     // Controlled value of the volume decrease in the last deal
   long              m_changed_volume_value;                   // Value of the volume change in the last deal
   bool              m_is_change_volume_inc;                   // Flag of the volume change in the last deal exceeding the increase value
   bool              m_is_change_volume_dec;                   // Flag of the volume change in the last deal being less than the increase value
   //--- Maximum volume within a day
   long              m_control_volume_high_day_inc;            // Controlled value of the maximum volume increase for a day
   long              m_control_volume_high_day_dec;            // Controlled value of the maximum volume decrease for a day
   long              m_changed_volume_high_day_value;          // Maximum volume change value within a day
   bool              m_is_change_volume_high_day_inc;          // Flag of the maximum day volume exceeding the increase value
   bool              m_is_change_volume_high_day_dec;          // Flag of the maximum day volume exceeding the decrease value
   //--- Minimum volume within a day
   long              m_control_volume_low_day_inc;             // Controlled value of the minimum volume increase for a day
   long              m_control_volume_low_day_dec;             // Controlled value of the minimum volume decrease for a day
   long              m_changed_volume_low_day_value;           // Minimum volume change value within a day
   bool              m_is_change_volume_low_day_inc;           // Flag of the minimum day volume exceeding the increase value
   bool              m_is_change_volume_low_day_dec;           // Flag of the minimum day volume exceeding the decrease value
   //--- Spread
   int               m_control_spread_inc;                     // Controlled spread increase value in points
   int               m_control_spread_dec;                     // Controlled spread decrease value in points
   int               m_changed_spread_value;                   // Spread change value in points
   bool              m_is_change_spread_inc;                   // Flag of spread change in points exceeding the increase value
   bool              m_is_change_spread_dec;                   // Flag of spread change in points exceeding the decrease value
   //--- StopLevel
   int               m_control_stops_level_inc;                // Controlled StopLevel increase value in points
   int               m_control_stops_level_dec;                // Controlled StopLevel decrease value in points
   int               m_changed_stops_level_value;              // StopLevel change value in points
   bool              m_is_change_stops_level_inc;              // Flag of StopLevel change in points exceeding the increase value
   bool              m_is_change_stops_level_dec;              // Flag of StopLevel change in points exceeding the decrease value
   //--- Freeze distance
   int               m_control_freeze_level_inc;               // Controlled FreezeLevel increase value in points
   int               m_control_freeze_level_dec;               // Controlled FreezeLevel decrease value in points
   int               m_changed_freeze_level_value;             // FreezeLevel change value in points
   bool              m_is_change_freeze_level_inc;             // Flag of FreezeLevel change in points exceeding the increase value
   bool              m_is_change_freeze_level_dec;             // Flag of FreezeLevel change in points exceeding the decrease value

   //--- Bid/Last
   double            m_control_bid_last_inc;                   // Controlled value of Bid or Last price increase
   double            m_control_bid_last_dec;                   // Controlled value of Bid or Last price decrease
   double            m_changed_bid_last_value;                 // Bid or Last price change value
   bool              m_is_change_bid_last_inc;                 // Flag of Bid or Last price change exceeding the increase value
   bool              m_is_change_bid_last_dec;                 // Flag of Bid or Last price change exceeding the decrease value
   //--- Maximum Bid/Last of the day
   double            m_control_bid_last_high_inc;              // Controlled increase value of the maximum Bid or Last price of the day
   double            m_control_bid_last_high_dec;              // Controlled decrease value of the maximum Bid or Last price of the day
   double            m_changed_bid_last_high_value;            // Maximum Bid or Last change value for the day
   bool              m_is_change_bid_last_high_inc;            // Flag of the maximum Bid or Last price change for the day exceeding the increase value
   bool              m_is_change_bid_last_high_dec;            // Flag of the maximum Bid or Last price change for the day exceeding the decrease value
   //--- Minimum Bid/Last of the day
   double            m_control_bid_last_low_inc;               // Controlled increase value of the minimum Bid or Last price of the day
   double            m_control_bid_last_low_dec;               // Controlled decrease value of the minimum Bid or Last price of the day
   double            m_changed_bid_last_low_value;             // Minimum Bid or Last change value for the day
   bool              m_is_change_bid_last_low_inc;             // Flag of the minimum Bid or Last price change for the day exceeding the increase value
   bool              m_is_change_bid_last_low_dec;             // Flag of the minimum Bid or Last price change for the day exceeding the decrease value
   //--- Ask
   double            m_control_ask_inc;                        // Controlled value of the Ask price increase
   double            m_control_ask_dec;                        // Controlled value of the Ask price decrease
   double            m_changed_ask_value;                      // Ask price change value
   bool              m_is_change_ask_inc;                      // Flag of the Ask price change exceeding the increase value
   bool              m_is_change_ask_dec;                      // Flag of the Ask price change exceeding the decrease value
   //--- Maximum Ask price for the day
   double            m_control_ask_high_inc;                   // Controlled increase value of the maximum Ask price of the day
   double            m_control_ask_high_dec;                   // Controlled decrease value of the maximum Ask price of the day
   double            m_changed_ask_high_value;                 // Maximum Ask price change value for the day
   bool              m_is_change_ask_high_inc;                 // Flag of the maximum Ask price change for the day exceeding the increase value
   bool              m_is_change_ask_high_dec;                 // Flag of the maximum Ask price change for the day exceeding the decrease value
   //--- Minimum Ask price for the day
   double            m_control_ask_low_inc;                    // Controlled increase value of the minimum Ask price of the day
   double            m_control_ask_low_dec;                    // Controlled decrease value of the minimum Ask price of the day
   double            m_changed_ask_low_value;                  // Minimum Ask price change value for the day
   bool              m_is_change_ask_low_inc;                  // Flag of the minimum Ask price change for the day exceeding the increase value
   bool              m_is_change_ask_low_dec;                  // Flag of the minimum Ask price change for the day exceeding the decrease value
   //--- Real Volume for the day
   double            m_control_volume_real_inc;                // Controlled value of the real volume increase of the day
   double            m_control_volume_real_dec;                // Controlled value of the real volume decrease of the day
   double            m_changed_volume_real_value;              // Real volume change value of the day
   bool              m_is_change_volume_real_inc;              // Flag of the real volume change for the day exceeding the increase value
   bool              m_is_change_volume_real_dec;              // Flag of the real volume change for the day exceeding the decrease value
   //--- Maximum real volume for the day
   double            m_control_volume_high_real_day_inc;       // Controlled value of the maximum real volume increase of the day
   double            m_control_volume_high_real_day_dec;       // Controlled value of the maximum real volume decrease of the day
   double            m_changed_volume_high_real_day_value;     // Maximum real volume change value of the day
   bool              m_is_change_volume_high_real_day_inc;     // Flag of the maximum real volume change for the day exceeding the increase value
   bool              m_is_change_volume_high_real_day_dec;     // Flag of the maximum real volume change for the day exceeding the decrease value
   //--- Minimum real volume for the day
   double            m_control_volume_low_real_day_inc;        // Controlled value of the minimum real volume increase of the day
   double            m_control_volume_low_real_day_dec;        // Controlled value of the minimum real volume decrease of the day
   double            m_changed_volume_low_real_day_value;      // Minimum real volume change value of the day
   bool              m_is_change_volume_low_real_day_inc;      // Flag of the minimum real volume change for the day exceeding the increase value
   bool              m_is_change_volume_low_real_day_dec;      // Flag of the minimum real volume change for the day exceeding the decrease value
   //--- Strike price
   double            m_control_option_strike_inc;              // Controlled value of the strike price increase
   double            m_control_option_strike_dec;              // Controlled value of the strike price decrease
   double            m_changed_option_strike_value;            // Strike price change value
   bool              m_is_change_option_strike_inc;            // Flag of the strike price change exceeding the increase value
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
   double            m_control_session_volume_inc;             // Controlled value of the total trade volume increase in the current session
   double            m_control_session_volume_dec;             // Controlled value of the total trade volume decrease in the current session
   double            m_changed_session_volume_value;           // The total deal volume change value in the current session
   bool              m_is_change_session_volume_inc;           // Flag of total trade volume change in the current session exceeding the increase value
   bool              m_is_change_session_volume_dec;           // Flag of total trade volume change in the current session exceeding the decrease value
   //--- The total turnover in the current session
   double            m_control_session_turnover_inc;           // Controlled value of the total turnover increase in the current session
   double            m_control_session_turnover_dec;           // Controlled value of the total turnover decrease in the current session
   double            m_changed_session_turnover_value;         // Total turnover change value in the current session
   bool              m_is_change_session_turnover_inc;         // Flag of total turnover change in the current session exceeding the increase value
   bool              m_is_change_session_turnover_dec;         // Flag of total turnover change in the current session exceeding the decrease value
   //--- The total volume of open positions
   double            m_control_session_interest_inc;           // Controlled value of the total open position volume increase in the current session
   double            m_control_session_interest_dec;           // Controlled value of the total open position volume decrease in the current session
   double            m_changed_session_interest_value;         // Change value of the open positions total volume in the current session
   bool              m_is_change_session_interest_inc;         // Flag of total open positions' volume change in the current session exceeding the increase value
   bool              m_is_change_session_interest_dec;         // Flag of total open positions' volume change in the current session exceeding the decrease value
   //--- The total volume of Buy orders at the moment
   double            m_control_session_buy_ord_volume_inc;     // Controlled value of the current total buy order volume increase
   double            m_control_session_buy_ord_volume_dec;     // Controlled value of the current total buy order volume decrease
   double            m_changed_session_buy_ord_volume_value;   // Change value of the current total buy order volume
   bool              m_is_change_session_buy_ord_volume_inc;   // Flag of changing the current total buy orders volume exceeding the increase value
   bool              m_is_change_session_buy_ord_volume_dec;   // Flag of changing the current total buy orders volume exceeding the decrease value
   //--- The total volume of Sell orders at the moment
   double            m_control_session_sell_ord_volume_inc;    // Controlled value of the current total sell order volume increase
   double            m_control_session_sell_ord_volume_dec;    // Controlled value of the current total sell order volume decrease
   double            m_changed_session_sell_ord_volume_value;  // Change value of the current total sell order volume
   bool              m_is_change_session_sell_ord_volume_inc;  // Flag of changing the current total sell orders volume exceeding the increase value
   bool              m_is_change_session_sell_ord_volume_dec;  // Flag of changing the current total sell orders volume exceeding the decrease value
   //--- Session open price
   double            m_control_session_open_inc;               // Controlled value of the session open price increase
   double            m_control_session_open_dec;               // Controlled value of the session open price decrease
   double            m_changed_session_open_value;             // Session open price change value
   bool              m_is_change_session_open_inc;             // Flag of the session open price change exceeding the increase value
   bool              m_is_change_session_open_dec;             // Flag of the session open price change exceeding the decrease value
   //--- Session close price
   double            m_control_session_close_inc;              // Controlled value of the session close price increase
   double            m_control_session_close_dec;              // Controlled value of the session close price decrease
   double            m_changed_session_close_value;            // Session close price change value
   bool              m_is_change_session_close_inc;            // Flag of the session close price change exceeding the increase value
   bool              m_is_change_session_close_dec;            // Flag of the session close price change exceeding the decrease value
   //--- The average weighted session price
   double            m_control_session_aw_inc;                 // Controlled value of the average weighted session price increase
   double            m_control_session_aw_dec;                 // Controlled value of the average weighted session price decrease
   double            m_changed_session_aw_value;               // The average weighted session price change value
   bool              m_is_change_session_aw_inc;               // Flag of the average weighted session price change value exceeding the increase value
   bool              m_is_change_session_aw_dec;               // Flag of the average weighted session price change value exceeding the decrease value
```

**Remove the highlighted methods due to their redundancy:**

```
//--- Initialize the variables of (1) tracked, (2) controlled symbol data
   virtual void      InitChangesParams(void);
   virtual void      InitControlsParams(void);
//--- Check symbol changes, return a change code
   virtual int       SetEventCode(void);
//--- Set an event type and fill in the event list
   virtual void      SetTypeEvent(void);

//--- Return description of symbol events
   string            EventDescription(const ENUM_SYMBOL_EVENT event);
```

**Instead of the virtual method of placing the code for changing a symbol property, declare**
**the method for checking the changes in the symbol properties and creating an event:**

```
//--- Initialize the variables of controlled symbol data
   virtual void      InitControlsParams(void);
//--- Check the list of symbol property changes and create an event
   void              CheckEvents(void);
```

**In the public section of the class, add declarations of methods setting tracked values and returning controlled values of tracked**
**properties, property change values and flags:**

```
public:
//--- Set the change value of the controlled symbol property
   template<typename T> void  SetControlChangedValue(const int property,const T value);
//--- Set the value of the controlled symbol property (1) increase, (2) decrease and (3) control level
   template<typename T> void  SetControlPropertyINC(const int property,const T value);
   template<typename T> void  SetControlPropertyDEC(const int property,const T value);
   template<typename T> void  SetControlPropertyLEVEL(const int property,const T value);
//--- Set the flag of a symbol property change exceeding the (1) increase and (2) decrease values
   template<typename T> void  SetControlFlagINC(const int property,const T value);
   template<typename T> void  SetControlFlagDEC(const int property,const T value);

//--- Return the set value of the (1) integer and (2) real symbol property controlled increase
   long              GetControlParameterINC(const ENUM_SYMBOL_PROP_INTEGER property)   const { return this.GetControlledValueLongINC(property);             }
   double            GetControlParameterINC(const ENUM_SYMBOL_PROP_DOUBLE property)    const { return this.GetControlledValueDoubleINC(property);           }
//--- Return the set value of the (1) integer and (2) real symbol property controlled decrease
   long              GetControlParameterDEC(const ENUM_SYMBOL_PROP_INTEGER property)   const { return this.GetControlledValueLongDEC(property);             }
   double            GetControlParameterDEC(const ENUM_SYMBOL_PROP_DOUBLE property)    const { return this.GetControlledValueDoubleDEC(property);           }
//--- Return the flag of an (1) integer and (2) real symbol property value change exceeding the increase value
   long              GetControlFlagINC(const ENUM_SYMBOL_PROP_INTEGER property)        const { return this.GetControlledFlagLongINC(property);              }
   double            GetControlFlagINC(const ENUM_SYMBOL_PROP_DOUBLE property)         const { return this.GetControlledFlagDoubleINC(property);            }
//--- Return the flag of an (1) integer and (2) real symbol property value change exceeding the decrease value
   bool              GetControlFlagDEC(const ENUM_SYMBOL_PROP_INTEGER property)        const { return (bool)this.GetControlledFlagLongDEC(property);        }
   bool              GetControlFlagDEC(const ENUM_SYMBOL_PROP_DOUBLE property)         const { return (bool)this.GetControlledFlagDoubleDEC(property);      }
//--- Return the change value of the controlled (1) integer and (2) real object property
   long              GetControlChangedValue(const ENUM_SYMBOL_PROP_INTEGER property)   const { return this.GetControlledChangedValueLong(property);         }
   double            GetControlChangedValue(const ENUM_SYMBOL_PROP_DOUBLE property)    const { return this.GetControlledChangedValueDouble(property);       }

//+------------------------------------------------------------------+
```

Each method is made in the form of two overloaded methods calling the methods of the base object corresponding to the type of a set/checked
symbol object property.

Previously, we have already developed the methods of a simplified access to some symbol object properties. Let's add the
methods of placing the values of properties' controlled levels there as well and declare the
methods of setting/receiving data for Bid/Last and related parameters (previously, Bid or Last was selected automatically
depending on the prices the chart is based on. Now we need to create the methods for working with this data):

```
//+------------------------------------------------------------------+
//| Get and set the parameters of tracked property changes           |
//+------------------------------------------------------------------+
   //--- Execution
   //--- Flag of changing the trading mode for a symbol
   bool              IsChangedTradeMode(void)                              const { return this.m_is_change_trade_mode;                                      }
   //--- Current session deals
   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the number of deals during the current session
   //--- getting (3) the number of deals change value during the current session,
   //--- getting the flag of the number of deals change during the current session exceeding the (4) increase, (5) decrease value
   void              SetControlSessionDealsInc(const long value)                 { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_DEALS,(long)::fabs(value));        }
   void              SetControlSessionDealsDec(const long value)                 { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_DEALS,(long)::fabs(value));        }
   void              SetControlSessionDealsLevel(const long value)               { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_DEALS,(long)::fabs(value));      }
   long              GetValueChangedSessionDeals(void)                     const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_DEALS);                    }
   bool              IsIncreasedSessionDeals(void)                         const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_DEALS);                   }
   bool              IsDecreasedSessionDeals(void)                         const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_DEALS);                   }
   //--- Buy orders of the current session
   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the current number of Buy orders
   //--- getting (4) the current number of Buy orders change value,
   //--- getting the flag of the current Buy orders' number change exceeding the (5) growth, (6) decrease value
   void              SetControlSessionBuyOrdInc(const long value)                { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_BUY_ORDERS,(long)::fabs(value));   }
   void              SetControlSessionBuyOrdDec(const long value)                { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_BUY_ORDERS,(long)::fabs(value));   }
   void              SetControlSessionBuyOrdLevel(const long value)              { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_BUY_ORDERS,(long)::fabs(value)); }
   long              GetValueChangedSessionBuyOrders(void)                 const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_BUY_ORDERS);               }
   bool              IsIncreasedSessionBuyOrders(void)                     const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_BUY_ORDERS);              }
   bool              IsDecreasedSessionBuyOrders(void)                     const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_BUY_ORDERS);              }
   //--- Sell orders of the current session
   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the current number of Sell orders
   //--- getting (4) the current number of Sell orders change value,
   //--- getting the flag of the current Sell orders' number change exceeding the (5) growth, (6) decrease value
   void              SetControlSessionSellOrdInc(const long value)               { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_SELL_ORDERS,(long)::fabs(value));  }
   void              SetControlSessionSellOrdDec(const long value)               { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_SELL_ORDERS,(long)::fabs(value));  }
   void              SetControlSessionSellOrdLevel(const long value)             { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_SELL_ORDERS,(long)::fabs(value));}
   long              GetValueChangedSessionSellOrders(void)                const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_SELL_ORDERS);              }
   bool              IsIncreasedSessionSellOrders(void)                    const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_SELL_ORDERS);             }
   bool              IsDecreasedSessionSellOrders(void)                    const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_SELL_ORDERS);             }
   //--- Volume of the last deal
   //--- setting the last deal volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) volume change values in the last deal,
   //--- getting the flag of the volume change in the last deal exceeding the (5) growth, (6) decrease value
   void              SetControlVolumeInc(const long value)                       { this.SetControlPropertyINC(SYMBOL_PROP_VOLUME,(long)::fabs(value));               }
   void              SetControlVolumeDec(const long value)                       { this.SetControlPropertyDEC(SYMBOL_PROP_VOLUME,(long)::fabs(value));               }
   void              SetControlVolumeLevel(const long value)                     { this.SetControlPropertyLEVEL(SYMBOL_PROP_VOLUME,(long)::fabs(value));             }
   long              GetValueChangedVolume(void)                           const { return this.GetControlChangedValue(SYMBOL_PROP_VOLUME);                           }
   bool              IsIncreasedVolume(void)                               const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_VOLUME);                          }
   bool              IsDecreasedVolume(void)                               const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_VOLUME);                          }
   //--- Maximum volume within a day
   //--- setting the maximum day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the maximum volume change value within a day,
   //--- getting the flag of the maximum day volume change exceeding the (5) growth, (6) decrease value
   void              SetControlVolumeHighInc(const long value)                   { this.SetControlPropertyINC(SYMBOL_PROP_VOLUMEHIGH,(long)::fabs(value));           }
   void              SetControlVolumeHighDec(const long value)                   { this.SetControlPropertyDEC(SYMBOL_PROP_VOLUMEHIGH,(long)::fabs(value));           }
   void              SetControlVolumeHighLevel(const long value)                 { this.SetControlPropertyLEVEL(SYMBOL_PROP_VOLUMEHIGH,(long)::fabs(value));         }
   long              GetValueChangedVolumeHigh(void)                       const { return this.GetControlChangedValue(SYMBOL_PROP_VOLUMEHIGH);                       }
   bool              IsIncreasedVolumeHigh(void)                           const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_VOLUMEHIGH);                      }
   bool              IsDecreasedVolumeHigh(void)                           const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_VOLUMEHIGH);                      }
   //--- Minimum volume within a day
   //--- setting the minimum day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the minimum volume change value within a day,
   //--- getting the flag of the minimum day volume change exceeding the (5) growth, (6) decrease value
   void              SetControlVolumeLowInc(const long value)                    { this.SetControlPropertyINC(SYMBOL_PROP_VOLUMELOW,(long)::fabs(value));            }
   void              SetControlVolumeLowDec(const long value)                    { this.SetControlPropertyDEC(SYMBOL_PROP_VOLUMELOW,(long)::fabs(value));            }
   void              SetControlVolumeLowLevel(const long value)                  { this.SetControlPropertyLEVEL(SYMBOL_PROP_VOLUMELOW,(long)::fabs(value));          }
   long              GetValueChangedVolumeLow(void)                        const { return this.GetControlChangedValue(SYMBOL_PROP_VOLUMELOW);                        }
   bool              IsIncreasedVolumeLow(void)                            const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_VOLUMELOW);                       }
   bool              IsDecreasedVolumeLow(void)                            const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_VOLUMELOW);                       }
   //--- Spread
   //--- setting the controlled spread (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) spread change value in points,
   //--- getting the flag of the spread change in points exceeding the (5) growth, (6) decrease value
   void              SetControlSpreadInc(const int value)                        { this.SetControlPropertyINC(SYMBOL_PROP_SPREAD,(long)::fabs(value));               }
   void              SetControlSpreadDec(const int value)                        { this.SetControlPropertyDEC(SYMBOL_PROP_SPREAD,(long)::fabs(value));               }
   void              SetControlSpreadLevel(const int value)                      { this.SetControlPropertyLEVEL(SYMBOL_PROP_SPREAD,(long)::fabs(value));             }
   int               GetValueChangedSpread(void)                           const { return (int)this.GetControlChangedValue(SYMBOL_PROP_SPREAD);                      }
   bool              IsIncreasedSpread(void)                               const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SPREAD);                          }
   bool              IsDecreasedSpread(void)                               const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SPREAD);                          }
   //--- StopLevel
   //--- setting the controlled StopLevel (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) StopLevel change value in points,
   //--- getting the flag of StopLevel change in points exceeding the (5) growth, (6) decrease value
   void              SetControlStopLevelInc(const int value)                     { this.SetControlPropertyINC(SYMBOL_PROP_TRADE_STOPS_LEVEL,(long)::fabs(value));    }
   void              SetControlStopLevelDec(const int value)                     { this.SetControlPropertyDEC(SYMBOL_PROP_TRADE_STOPS_LEVEL,(long)::fabs(value));    }
   void              SetControlStopLevelLevel(const int value)                   { this.SetControlPropertyLEVEL(SYMBOL_PROP_TRADE_STOPS_LEVEL,(long)::fabs(value));  }
   int               GetValueChangedStopLevel(void)                        const { return (int)this.GetControlChangedValue(SYMBOL_PROP_TRADE_STOPS_LEVEL);           }
   bool              IsIncreasedStopLevel(void)                            const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_TRADE_STOPS_LEVEL);               }
   bool              IsDecreasedStopLevel(void)                            const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_TRADE_STOPS_LEVEL);               }
   //--- Freeze distance
   //--- setting the controlled FreezeLevel (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) FreezeLevel change value in points,
   //--- getting the flag of FreezeLevel change in points exceeding the (5) growth, (6) decrease value
   void              SetControlFreezeLevelInc(const int value)                   { this.SetControlPropertyINC(SYMBOL_PROP_TRADE_FREEZE_LEVEL,(long)::fabs(value));   }
   void              SetControlFreezeLevelDec(const int value)                   { this.SetControlPropertyDEC(SYMBOL_PROP_TRADE_FREEZE_LEVEL,(long)::fabs(value));   }
   void              SetControlFreezeLevelLevel(const int value)                 { this.SetControlPropertyLEVEL(SYMBOL_PROP_TRADE_FREEZE_LEVEL,(long)::fabs(value)); }
   int               GetValueChangedFreezeLevel(void)                      const { return (int)this.GetControlChangedValue(SYMBOL_PROP_TRADE_FREEZE_LEVEL);          }
   bool              IsIncreasedFreezeLevel(void)                          const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_TRADE_FREEZE_LEVEL);              }
   bool              IsDecreasedFreezeLevel(void)                          const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_TRADE_FREEZE_LEVEL);              }

   //--- Bid
   //--- setting the controlled Bid price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) Bid or Last price change value,
   //--- getting the flag of the Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlBidInc(const double value)                        { this.SetControlPropertyINC(SYMBOL_PROP_BID,::fabs(value));                        }
   void              SetControlBidDec(const double value)                        { this.SetControlPropertyDEC(SYMBOL_PROP_BID,::fabs(value));                        }
   void              SetControlBidLevel(const double value)                      { this.SetControlPropertyLEVEL(SYMBOL_PROP_BID,::fabs(value));                      }
   double            GetValueChangedBid(void)                              const { return this.GetControlChangedValue(SYMBOL_PROP_BID);                              }
   bool              IsIncreasedBid(void)                                  const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_BID);                             }
   bool              IsDecreasedBid(void)                                  const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_BID);                             }
   //--- The highest Bid price of the day
   //--- setting the controlled maximum Bid price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) maximum Bid or Last price change value,
   //--- getting the flag of the maximum Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlBidHighInc(const double value)                    { this.SetControlPropertyINC(SYMBOL_PROP_BIDHIGH,::fabs(value));                    }
   void              SetControlBidHighDec(const double value)                    { this.SetControlPropertyDEC(SYMBOL_PROP_BIDHIGH,::fabs(value));                    }
   void              SetControlBidHighLevel(const double value)                  { this.SetControlPropertyLEVEL(SYMBOL_PROP_BIDHIGH,::fabs(value));                  }
   double            GetValueChangedBidHigh(void)                          const { return this.GetControlChangedValue(SYMBOL_PROP_BIDHIGH);                          }
   bool              IsIncreasedBidHigh(void)                              const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_BIDHIGH);                         }
   bool              IsDecreasedBidHigh(void)                              const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_BIDHIGH);                         }
   //--- The lowest Bid price of the day
   //--- setting the controlled minimum Bid price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) minimum Bid or Last price change value,
   //--- getting the flag of the minimum Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlBidLowInc(const double value)                     { this.SetControlPropertyINC(SYMBOL_PROP_BIDLOW,::fabs(value));                     }
   void              SetControlBidLowDec(const double value)                     { this.SetControlPropertyDEC(SYMBOL_PROP_BIDLOW,::fabs(value));                     }
   void              SetControlBidLowLevel(const double value)                   { this.SetControlPropertyLEVEL(SYMBOL_PROP_BIDLOW,::fabs(value));                   }
   double            GetValueChangedBidLow(void)                           const { return this.GetControlChangedValue(SYMBOL_PROP_BIDLOW);                           }
   bool              IsIncreasedBidLow(void)                               const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_BIDLOW);                          }
   bool              IsDecreasedBidLow(void)                               const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_BIDLOW);                          }

   //--- Last
   //--- setting the controlled Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) Bid or Last price change value,
   //--- getting the flag of the Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlLastInc(const double value)                       { this.SetControlPropertyINC(SYMBOL_PROP_LAST,::fabs(value));                       }
   void              SetControlLastDec(const double value)                       { this.SetControlPropertyDEC(SYMBOL_PROP_LAST,::fabs(value));                       }
   void              SetControlLastLevel(const double value)                     { this.SetControlPropertyLEVEL(SYMBOL_PROP_LAST,::fabs(value));                     }
   double            GetValueChangedLast(void)                             const { return this.GetControlChangedValue(SYMBOL_PROP_LAST);                             }
   bool              IsIncreasedLast(void)                                 const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_LAST);                            }
   bool              IsDecreasedLast(void)                                 const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_LAST);                            }
   //--- The highest Last price of the day
   //--- setting the controlled maximum Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) maximum Bid or Last price change value,
   //--- getting the flag of the maximum Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlLastHighInc(const double value)                   { this.SetControlPropertyINC(SYMBOL_PROP_LASTHIGH,::fabs(value));                   }
   void              SetControlLastHighDec(const double value)                   { this.SetControlPropertyDEC(SYMBOL_PROP_LASTHIGH,::fabs(value));                   }
   void              SetControlLastHighLevel(const double value)                 { this.SetControlPropertyLEVEL(SYMBOL_PROP_LASTHIGH,::fabs(value));                 }
   double            GetValueChangedLastHigh(void)                         const { return this.GetControlChangedValue(SYMBOL_PROP_LASTHIGH);                         }
   bool              IsIncreasedLastHigh(void)                             const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_LASTHIGH);                        }
   bool              IsDecreasedLastHigh(void)                             const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_LASTHIGH);                        }
   //--- The lowest Last price of the day
   //--- setting the controlled minimum Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) minimum Bid or Last price change value,
   //--- getting the flag of the minimum Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlLastLowInc(const double value)                    { this.SetControlPropertyINC(SYMBOL_PROP_LASTLOW,::fabs(value));                    }
   void              SetControlLastLowDec(const double value)                    { this.SetControlPropertyDEC(SYMBOL_PROP_LASTLOW,::fabs(value));                    }
   void              SetControlLastLowLevel(const double value)                  { this.SetControlPropertyLEVEL(SYMBOL_PROP_LASTLOW,::fabs(value));                  }
   double            GetValueChangedLastLow(void)                          const { return this.GetControlChangedValue(SYMBOL_PROP_LASTLOW);                          }
   bool              IsIncreasedLastLow(void)                              const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_LASTLOW);                         }
   bool              IsDecreasedLastLow(void)                              const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_LASTLOW);                         }

   //--- Bid/Last
   //--- setting the controlled Bid or Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) Bid or Last price change value,
   //--- getting the flag of the Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlBidLastInc(const double value);
   void              SetControlBidLastDec(const double value);
   void              SetControlBidLastLevel(const double value);
   double            GetValueChangedBidLast(void)                          const;
   bool              IsIncreasedBidLast(void)                              const;
   bool              IsDecreasedBidLast(void)                              const;
   //--- Maximum Bid/Last of the day
   //--- setting the controlled maximum Bid or Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) maximum Bid or Last price change value,
   //--- getting the flag of the maximum Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlBidLastHighInc(const double value);
   void              SetControlBidLastHighDec(const double value);
   void              SetControlBidLastHighLevel(const double value);
   double            GetValueChangedBidLastHigh(void)                      const;
   bool              IsIncreasedBidLastHigh(void)                          const;
   bool              IsDecreasedBidLastHigh(void)                          const;
   //--- Minimum Bid/Last of the day
   //--- setting the controlled minimum Bid or Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) minimum Bid or Last price change value,
   //--- getting the flag of the minimum Bid or Last price change exceeding the (5) growth, (6) decrease value
   void              SetControlBidLastLowInc(const double value);
   void              SetControlBidLastLowDec(const double value);
   void              SetControlBidLastLowLevev(const double value);
   double            GetValueChangedBidLastLow(void)                       const;
   bool              IsIncreasedBidLastLow(void)                           const;
   bool              IsDecreasedBidLastLow(void)                           const;

   //--- Ask
   //--- setting the controlled Ask price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) Ask price change value,
   //--- getting the flag of the Ask price change exceeding the (5) growth, (6) decrease value
   void              SetControlAskInc(const double value)                        { this.SetControlPropertyINC(SYMBOL_PROP_ASK,::fabs(value));                        }
   void              SetControlAskDec(const double value)                        { this.SetControlPropertyDEC(SYMBOL_PROP_ASK,::fabs(value));                        }
   void              SetControlAskLevel(const double value)                      { this.SetControlPropertyLEVEL(SYMBOL_PROP_ASK,::fabs(value));                      }
   double            GetValueChangedAsk(void)                              const { return this.GetControlChangedValue(SYMBOL_PROP_ASK);                              }
   bool              IsIncreasedAsk(void)                                  const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_ASK);                             }
   bool              IsDecreasedAsk(void)                                  const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_ASK);                             }
   //--- Maximum Ask price for the day
   //--- setting the maximum day Ask controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the maximum Ask change value within a day,
   //--- getting the flag of the maximum day Ask change exceeding the (5) growth, (6) decrease value
   void              SetControlAskHighInc(const double value)                    { this.SetControlPropertyINC(SYMBOL_PROP_ASKHIGH,::fabs(value));                    }
   void              SetControlAskHighDec(const double value)                    { this.SetControlPropertyDEC(SYMBOL_PROP_ASKHIGH,::fabs(value));                    }
   void              SetControlAskHighLevel(const double value)                  { this.SetControlPropertyLEVEL(SYMBOL_PROP_ASKHIGH,::fabs(value));                  }
   double            GetValueChangedAskHigh(void)                          const { return this.GetControlChangedValue(SYMBOL_PROP_ASKHIGH);                          }
   bool              IsIncreasedAskHigh(void)                              const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_ASKHIGH);                         }
   bool              IsDecreasedAskHigh(void)                              const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_ASKHIGH);                         }
   //--- Minimum Ask price for the day
   //--- setting the minimum day Ask controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the minimum Ask change value within a day,
   //--- getting the flag of the minimum day Ask change exceeding the (5) growth, (6) decrease value
   void              SetControlAskLowInc(const double value)                     { this.SetControlPropertyINC(SYMBOL_PROP_ASKLOW,::fabs(value));                     }
   void              SetControlAskLowDec(const double value)                     { this.SetControlPropertyDEC(SYMBOL_PROP_ASKLOW,::fabs(value));                     }
   void              SetControlAskLowLevel(const double value)                   { this.SetControlPropertyLEVEL(SYMBOL_PROP_ASKLOW,::fabs(value));                   }
   double            GetValueChangedAskLow(void)                           const { return this.GetControlChangedValue(SYMBOL_PROP_ASKLOW);                           }
   bool              IsIncreasedAskLow(void)                               const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_ASKLOW);                          }
   bool              IsDecreasedAskLow(void)                               const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_ASKLOW);                          }
   //--- Real Volume for the day
   //--- setting the real day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the change value of the real day volume,
   //--- getting the flag of the real day volume change exceeding the (5) growth, (6) decrease value
   void              SetControlVolumeRealInc(const double value)                 { this.SetControlPropertyINC(SYMBOL_PROP_VOLUME_REAL,::fabs(value));                }
   void              SetControlVolumeRealDec(const double value)                 { this.SetControlPropertyDEC(SYMBOL_PROP_VOLUME_REAL,::fabs(value));                }
   void              SetControlVolumeRealLevel(const double value)               { this.SetControlPropertyLEVEL(SYMBOL_PROP_VOLUME_REAL,::fabs(value));              }
   double            GetValueChangedVolumeReal(void)                       const { return this.GetControlChangedValue(SYMBOL_PROP_VOLUME_REAL);                      }
   bool              IsIncreasedVolumeReal(void)                           const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_VOLUME_REAL);                     }
   bool              IsDecreasedVolumeReal(void)                           const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_VOLUME_REAL);                     }
   //--- Maximum real volume for the day
   //--- setting the maximum real day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the change value of the maximum real day volume,
   //--- getting the flag of the maximum real day volume change exceeding the (5) growth, (6) decrease value
   void              SetControlVolumeHighRealInc(const double value)             { this.SetControlPropertyINC(SYMBOL_PROP_VOLUMEHIGH_REAL,::fabs(value));            }
   void              SetControlVolumeHighRealDec(const double value)             { this.SetControlPropertyDEC(SYMBOL_PROP_VOLUMEHIGH_REAL,::fabs(value));            }
   void              SetControlVolumeHighRealLevel(const double value)           { this.SetControlPropertyLEVEL(SYMBOL_PROP_VOLUMEHIGH_REAL,::fabs(value));          }
   double            GetValueChangedVolumeHighReal(void)                   const { return this.GetControlChangedValue(SYMBOL_PROP_VOLUMEHIGH_REAL);                  }
   bool              IsIncreasedVolumeHighReal(void)                       const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_VOLUMEHIGH_REAL);                 }
   bool              IsDecreasedVolumeHighReal(void)                       const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_VOLUMEHIGH_REAL);                 }
   //--- Minimum real volume for the day
   //--- setting the minimum real day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the change value of the minimum real day volume,
   //--- getting the flag of the minimum real day volume change exceeding the (5) growth, (6) decrease value
   void              SetControlVolumeLowRealInc(const double value)              { this.SetControlPropertyINC(SYMBOL_PROP_VOLUMELOW_REAL,::fabs(value));             }
   void              SetControlVolumeLowRealDec(const double value)              { this.SetControlPropertyDEC(SYMBOL_PROP_VOLUMELOW_REAL,::fabs(value));             }
   void              SetControlVolumeLowRealLevel(const double value)            { this.SetControlPropertyLEVEL(SYMBOL_PROP_VOLUMELOW_REAL,::fabs(value));           }
   double            GetValueChangedVolumeLowReal(void)                    const { return this.GetControlChangedValue(SYMBOL_PROP_VOLUMELOW_REAL);                   }
   bool              IsIncreasedVolumeLowReal(void)                        const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_VOLUMELOW_REAL);                  }
   bool              IsDecreasedVolumeLowReal(void)                        const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_VOLUMELOW_REAL);                  }
   //--- Strike price
   //--- setting the controlled strike price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) the change value of the strike price,
   //--- getting the flag of the strike price change exceeding the (5) growth, (6) decrease value
   void              SetControlOptionStrikeInc(const double value)               { this.SetControlPropertyINC(SYMBOL_PROP_OPTION_STRIKE,::fabs(value));              }
   void              SetControlOptionStrikeDec(const double value)               { this.SetControlPropertyDEC(SYMBOL_PROP_OPTION_STRIKE,::fabs(value));              }
   void              SetControlOptionStrikeLevel(const double value)             { this.SetControlPropertyLEVEL(SYMBOL_PROP_OPTION_STRIKE,::fabs(value));            }
   double            GetValueChangedOptionStrike(void)                     const { return this.GetControlChangedValue(SYMBOL_PROP_OPTION_STRIKE);                    }
   bool              IsIncreasedOptionStrike(void)                         const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_OPTION_STRIKE);                   }
   bool              IsDecreasedOptionStrike(void)                         const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_OPTION_STRIKE);                   }
   //--- Maximum allowed total volume of unidirectional positions and orders
   //--- (1) Setting the control level
   //--- (2) getting the change value of the maximum allowed total volume of unidirectional positions and orders,
   //--- getting the flag of (3) increasing, (4) decreasing the maximum allowed total volume of unidirectional positions and orders
   void              SetControlVolumeLimitLevel(const double value)              { this.SetControlPropertyLEVEL(SYMBOL_PROP_VOLUME_LIMIT,::fabs(value));             }
   double            GetValueChangedVolumeLimit(void)                      const { return this.GetControlChangedValue(SYMBOL_PROP_VOLUME_LIMIT);                     }
   bool              IsIncreasedVolumeLimit(void)                          const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_VOLUME_LIMIT);                    }
   bool              IsDecreasedVolumeLimit(void)                          const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_VOLUME_LIMIT);                    }
   //---  Swap long
   //--- (1) Setting the control level
   //--- (2) getting the swap long change value,
   //--- getting the flag of (3) increasing, (4) decreasing the swap long
   void              SetControlSwapLongLevel(const double value)                 { this.SetControlPropertyLEVEL(SYMBOL_PROP_SWAP_LONG,::fabs(value));                }
   double            GetValueChangedSwapLong(void)                         const { return this.GetControlChangedValue(SYMBOL_PROP_SWAP_LONG);                        }
   bool              IsIncreasedSwapLong(void)                             const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SWAP_LONG);                       }
   bool              IsDecreasedSwapLong(void)                             const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SWAP_LONG);                       }
   //---  Swap short
   //--- (1) Setting the control level
   //--- (2) getting the swap short change value,
   //--- getting the flag of (3) increasing, (4) decreasing the swap short
   void              SetControlSwapShortLevel(const double value)                { this.SetControlPropertyLEVEL(SYMBOL_PROP_SWAP_SHORT,::fabs(value));               }
   double            GetValueChangedSwapShort(void)                        const { return this.GetControlChangedValue(SYMBOL_PROP_SWAP_SHORT);                       }
   bool              IsIncreasedSwapShort(void)                            const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SWAP_SHORT);                      }
   bool              IsDecreasedSwapShort(void)                            const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SWAP_SHORT);                      }
   //--- The total volume of deals in the current session
   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the total volume of deals during the current session
   //--- getting (4) the total deal volume change value in the current session,
   //--- getting the flag of the total deal volume change during the current session exceeding the (5) growth, (6) decrease value
   void              SetControlSessionVolumeInc(const double value)              { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_VOLUME,::fabs(value));             }
   void              SetControlSessionVolumeDec(const double value)              { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_VOLUME,::fabs(value));             }
   void              SetControlSessionVolumeLevel(const double value)            { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_VOLUME,::fabs(value));           }
   double            GetValueChangedSessionVolume(void)                    const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_VOLUME);                   }
   bool              IsIncreasedSessionVolume(void)                        const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_VOLUME);                  }
   bool              IsDecreasedSessionVolume(void)                        const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_VOLUME);                  }
   //--- The total turnover in the current session
   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the total turnover during the current session
   //--- getting (4) the total turnover change value in the current session,
   //--- getting the flag of the total turnover change during the current session exceeding the (5) growth, (6) decrease value
   void              SetControlSessionTurnoverInc(const double value)            { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_TURNOVER,::fabs(value));           }
   void              SetControlSessionTurnoverDec(const double value)            { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_TURNOVER,::fabs(value));           }
   void              SetControlSessionTurnoverLevel(const double value)          { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_TURNOVER,::fabs(value));         }
   double            GetValueChangedSessionTurnover(void)                  const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_TURNOVER);                 }
   bool              IsIncreasedSessionTurnover(void)                      const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_TURNOVER);                }
   bool              IsDecreasedSessionTurnover(void)                      const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_TURNOVER);                }
   //--- The total volume of open positions
   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the total volume of open positions during the current session
   //--- getting (4) the change value of the open positions total volume in the current session,
   //--- getting the flag of the open positions total volume change during the current session exceeding the (5) growth, (6) decrease value
   void              SetControlSessionInterestInc(const double value)            { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_INTEREST,::fabs(value));           }
   void              SetControlSessionInterestDec(const double value)            { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_INTEREST,::fabs(value));           }
   void              SetControlSessionInterestLevel(const double value)          { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_INTEREST,::fabs(value));         }
   double            GetValueChangedSessionInterest(void)                  const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_INTEREST);                 }
   bool              IsIncreasedSessionInterest(void)                      const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_INTEREST);                }
   bool              IsDecreasedSessionInterest(void)                      const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_INTEREST);                }
   //--- The total volume of Buy orders at the moment
   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the current total buy order volume
   //--- getting (4) the change value of the current total buy order volume,
   //--- getting the flag of the current total buy orders' volume change exceeding the (5) growth, (6) decrease value
   void              SetControlSessionBuyOrdVolumeInc(const double value)        { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME,::fabs(value));  }
   void              SetControlSessionBuyOrdVolumeDec(const double value)        { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME,::fabs(value));  }
   void              SetControlSessionBuyOrdVolumeLevel(const double value)      { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME,::fabs(value));}
   double            GetValueChangedSessionBuyOrdVolume(void)              const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);        }
   bool              IsIncreasedSessionBuyOrdVolume(void)                  const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);       }
   bool              IsDecreasedSessionBuyOrdVolume(void)                  const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);       }
   //--- The total volume of Sell orders at the moment
   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the current total sell order volume
   //--- getting (4) the change value of the current total sell order volume,
   //--- getting the flag of the current total sell orders' volume change exceeding the (5) growth, (6) decrease value
   void              SetControlSessionSellOrdVolumeInc(const double value)       { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME,::fabs(value)); }
   void              SetControlSessionSellOrdVolumeDec(const double value)       { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME,::fabs(value)); }
   void              SetControlSessionSellOrdVolumeLevel(const double value)     { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME,::fabs(value));}
   double            GetValueChangedSessionSellOrdVolume(void)             const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);       }
   bool              IsIncreasedSessionSellOrdVolume(void)                 const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);      }
   bool              IsDecreasedSessionSellOrdVolume(void)                 const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);      }
   //--- Session open price
   //--- setting the controlled session open price (1) increase, (2) decrease and (3) control value
   //--- getting (4) the change value of the session open price,
   //--- getting the flag of the session open price change exceeding the (5) growth, (6) decrease value
   void              SetControlSessionPriceOpenInc(const double value)           { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_OPEN,::fabs(value));               }
   void              SetControlSessionPriceOpenDec(const double value)           { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_OPEN,::fabs(value));               }
   void              SetControlSessionPriceOpenLevel(const double value)         { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_OPEN,::fabs(value));             }
   double            GetValueChangedSessionPriceOpen(void)                 const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_OPEN);                     }
   bool              IsIncreasedSessionPriceOpen(void)                     const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_OPEN);                    }
   bool              IsDecreasedSessionPriceOpen(void)                     const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_OPEN);                    }
   //--- Session close price
   //--- setting the controlled session close price (1) increase, (2) decrease and (3) control value
   //--- getting (4) the change value of the session close price,
   //--- getting the flag of the session close price change exceeding the (5) growth, (6) decrease value
   void              SetControlSessionPriceCloseInc(const double value)          { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_CLOSE,::fabs(value));              }
   void              SetControlSessionPriceCloseDec(const double value)          { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_CLOSE,::fabs(value));              }
   void              SetControlSessionPriceCloseLevel(const double value)        { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_CLOSE,::fabs(value));            }
   double            GetValueChangedSessionPriceClose(void)                const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_CLOSE);                    }
   bool              IsIncreasedSessionPriceClose(void)                    const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_CLOSE);                   }
   bool              IsDecreasedSessionPriceClose(void)                    const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_CLOSE);                   }
   //--- The average weighted session price
   //--- setting the controlled session average weighted price (1) increase, (2) decrease and (3) control value
   //--- getting (4) the change value of the average weighted session price,
   //--- getting the flag of the average weighted session price change exceeding the (5) growth, (6) decrease value
   void              SetControlSessionPriceAWInc(const double value)             { this.SetControlPropertyINC(SYMBOL_PROP_SESSION_AW,::fabs(value));                 }
   void              SetControlSessionPriceAWDec(const double value)             { this.SetControlPropertyDEC(SYMBOL_PROP_SESSION_AW,::fabs(value));                 }
   void              SetControlSessionPriceAWLevel(const double value)           { this.SetControlPropertyLEVEL(SYMBOL_PROP_SESSION_AW,::fabs(value));               }
   double            GetValueChangedSessionPriceAW(void)                   const { return this.GetControlChangedValue(SYMBOL_PROP_SESSION_AW);                       }
   bool              IsIncreasedSessionPriceAW(void)                       const { return (bool)this.GetControlFlagINC(SYMBOL_PROP_SESSION_AW);                      }
   bool              IsDecreasedSessionPriceAW(void)                       const { return (bool)this.GetControlFlagDEC(SYMBOL_PROP_SESSION_AW);                      }
//---
```

Let's consider the implementation of declared methods and changes in the already existing ones.

**Make some small changes in the class constructor:**

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CSymbol::CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name,const int index)
  {
   this.m_name=name;
   this.m_type=COLLECTION_SYMBOLS_ID;
   if(!this.Exist())
     {
      ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\"",": ",TextByLanguage("Ошибка. Такого символа нет на сервере","Error. No such symbol on the server"));
      this.m_global_error=ERR_MARKET_UNKNOWN_SYMBOL;
     }
   bool select=::SymbolInfoInteger(this.m_name,SYMBOL_SELECT);
   ::ResetLastError();
   if(!select)
     {
      if(!this.SetToMarketWatch())
        {
         this.m_global_error=::GetLastError();
         ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\": ",TextByLanguage("Не удалось поместить в обзор рынка. Ошибка: ","Failed to put in market watch. Error: "),this.m_global_error);
        }
     }
   ::ResetLastError();
   if(!::SymbolInfoTick(this.m_name,this.m_tick))
     {
      this.m_global_error=::GetLastError();
      ::Print(DFUN_ERR_LINE,"\"",this.m_name,"\": ",TextByLanguage("Не удалось получить текущие цены. Ошибка: ","Could not get current prices. Error: "),this.m_global_error);
     }
//--- Initialize control data
   this.SetControlDataArraySizeLong(SYMBOL_PROP_INTEGER_TOTAL);
   this.SetControlDataArraySizeDouble(SYMBOL_PROP_DOUBLE_TOTAL);
   this.ResetChangesParams();
   this.ResetControlsParams();

//--- Initialize symbol data
   this.Reset();
   this.InitMarginRates();
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
//+------------------------------------------------------------------------------------------------------------+
//|Compare CSymbol objects by all possible properties (for sorting lists by a specified symbol object property)|
//+------------------------------------------------------------------------------------------------------------+
```

To accurately define object events in the base object class, assign the
symbol collection ID to the symbol object type and set the size of arrays
of integer and real data to track events in symbol object
properties by the base parent object. Next,

initialize editable and control
parameters in the arrays of integer and real properties.

**The Refresh() method of the symbol object has also been changed:**

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
//--- Initialize event data
   this.m_is_event=false;

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

//--- Fill in the symbol current data
   for(int i=0;i<SYMBOL_PROP_INTEGER_TOTAL;i++)
      this.m_long_prop_event[i][3]=this.m_long_prop[i];
   for(int i=0;i<SYMBOL_PROP_DOUBLE_TOTAL;i++)
      this.m_double_prop_event[i][3]=this.m_double_prop[i];

   CBaseObj::Refresh();
   this.CheckEvents();
  }
//+------------------------------------------------------------------+
```

Since we no longer need to create structures for storing the current and previous symbol property states, filling in the structure of the
current symbol status data has been removed. Instead, we arranged

filling in the arrays of integer and real
properties in the base object.

After the arrays are filled in, we need to call
the Refresh() method of the

CBaseObj base object, in which the search for occurred changes
is performed and the list of descendant object base events is created.

After creating the list of base events in the parent class (if there are event generation criteria), check
the base events using the CheckEvents() method. If they are present, create the list of symbol events.

**Implementing the method of checking events:**

```
//+------------------------------------------------------------------+
//| Check the list of symbol property changes and create an event    |
//+------------------------------------------------------------------+
void CSymbol::CheckEvents(void)
  {
   int total=this.m_list_events_base.Total();
   if(total==0)
      return;
  for(int i=0;i<total;i++)
     {
      CBaseEvent *event=this.GetEventBase(i);
      if(event==NULL)
         continue;
      long lvalue=0;
      this.UshortToLong(this.MSCfromTime(this.TickTime()),0,lvalue);
      this.UshortToLong(event.Reason(),1,lvalue);
      this.UshortToLong(COLLECTION_SYMBOLS_ID,2,lvalue);
      if(this.EventAdd((ushort)event.ID(),lvalue,event.Value(),this.Name()))
         this.m_is_event=true;
     }
  }
//+------------------------------------------------------------------+
```

If the list of base events is empty, exit.

In the loop by the list of base events, receive
the next event. If the event is received, create a symbol event:

- receive only milliseconds
from the current time in milliseconds and add themto **the first two bytes** of the event 'long' parameter
- get an event reason (increase/decrease/above
level/below level) and

add itto
the

**second two bytes** of the event 'long' parameter
- symbol collection IDis
added to the **third two bytes** of the event 'long'
parameter
- add the symbol event to the list of symbol events and set
the flag indicating an event presence on a symbol

**The method of initializing the variables of controlled symbol data:**

```
//+------------------------------------------------------------------+
//| Initialize the variables of controlled symbol data               |
//+------------------------------------------------------------------+
void CSymbol::InitControlsParams(void)
  {
   this.ResetControlsParams();
  }
//+------------------------------------------------------------------+
```

Simply call the above-mentioned method of resetting the variables of controlled object data values.

**The methods of setting controlled values and flags of occurred changes and the methods for receiving a size of occurred changes and**
**flags:**

```
//+------------------------------------------------------------------+
//| Set the value of the controlled property increase                |
//+------------------------------------------------------------------+
template<typename T> void CSymbol::SetControlPropertyINC(const int property,const T value)
  {
   if(property<SYMBOL_PROP_INTEGER_TOTAL)
      this.SetControlledValueINC(property,(long)value);
   else
      this.SetControlledValueINC(property,(double)value);
  }
//+------------------------------------------------------------------+
//| Set the value of the controlled property decrease                |
//+------------------------------------------------------------------+
template<typename T> void CSymbol::SetControlPropertyDEC(const int property,const T value)
  {
   if(property<SYMBOL_PROP_INTEGER_TOTAL)
      this.SetControlledValueDEC(property,(long)value);
   else
      this.SetControlledValueDEC(property,(double)value);
  }
//+------------------------------------------------------------------+
//| Set the value of the controlled property level                   |
//+------------------------------------------------------------------+
template<typename T> void CSymbol::SetControlPropertyLEVEL(const int property,const T value)
  {
   if(property<SYMBOL_PROP_INTEGER_TOTAL)
      this.SetControlledValueLEVEL(property,(long)value);
   else
      this.SetControlledValueLEVEL(property,(double)value);
  }
//+------------------------------------------------------------------+
//| Set the flag of the symbol property value change                 |
//| exceeding the increase value                                     |
//+------------------------------------------------------------------+
template<typename T> void CSymbol::SetControlFlagINC(const int property,const T value)
  {
   if(property<SYMBOL_PROP_INTEGER_TOTAL)
      this.SetControlledFlagINC(property,(long)value);
   else
      this.SetControlledFlagINC(property,(double)value);
  }
//+------------------------------------------------------------------+
//| Set the flag of the symbol property value change                 |
//| exceeding the decrease value                                     |
//+------------------------------------------------------------------+
template<typename T> void CSymbol::SetControlFlagDEC(const int property,const T value)
  {
   if(property<SYMBOL_PROP_INTEGER_TOTAL)
      this.SetControlledFlagDEC(property,(long)value);
   else
      this.SetControlledFlagDEC(property,(double)value);
  }
//+------------------------------------------------------------------+
//| Set the change value of the controlled symbol property           |
//+------------------------------------------------------------------+
template<typename T> void CSymbol::SetControlChangedValue(const int property,const T value)
  {
   if(property<SYMBOL_PROP_INTEGER_TOTAL)
      this.SetControlledChangedValue(property,(long)value);
   else
      this.SetControlledChangedValue(property,(double)value);
  }
//+------------------------------------------------------------------+
//| Set the Bid or Last price controlled increase                    |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastInc(const double value)
  {
   this.SetControlPropertyINC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BID : SYMBOL_PROP_LAST),::fabs(value));
  }
//+------------------------------------------------------------------+
//|Set the Bid or Last price controlled decrease                     |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastDec(const double value)
  {
   this.SetControlPropertyDEC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BID : SYMBOL_PROP_LAST),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the Bid or Last price control level                          |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastLevel(const double value)
  {
   this.SetControlPropertyLEVEL((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BID : SYMBOL_PROP_LAST),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Return the Bid or Last price change value                        |
//+------------------------------------------------------------------+
double CSymbol::GetValueChangedBidLast(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetControlChangedValue(SYMBOL_PROP_BID) : this.GetControlChangedValue(SYMBOL_PROP_LAST));
  }
//+------------------------------------------------------------------+
//| Return the flag of the Bid or Last price change                  |
//| exceeding the increase value                                     |
//+------------------------------------------------------------------+
bool CSymbol::IsIncreasedBidLast(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetControlFlagINC(SYMBOL_PROP_BID) : (bool)this.GetControlFlagINC(SYMBOL_PROP_LAST));
  }
//+------------------------------------------------------------------+
//| Return the flag of the Bid or Last price change                  |
//| exceeding the decrease value                                     |
//+------------------------------------------------------------------+
bool CSymbol::IsDecreasedBidLast(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetControlFlagDEC(SYMBOL_PROP_BID) : (bool)this.GetControlFlagDEC(SYMBOL_PROP_LAST));
  }
//+------------------------------------------------------------------+
//| Set the controlled increase value                                |
//| of the maximum Bid or Last price                                 |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastHighInc(const double value)
  {
   this.SetControlPropertyINC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDHIGH : SYMBOL_PROP_LASTHIGH),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the controlled decrease value                                |
//| of the maximum Bid or Last price                                 |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastHighDec(const double value)
  {
   this.SetControlPropertyDEC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDHIGH : SYMBOL_PROP_LASTHIGH),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the maximum Bid or Last price control level                  |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastHighLevel(const double value)
  {
   this.SetControlPropertyLEVEL((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDHIGH : SYMBOL_PROP_LASTHIGH),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Return the maximum Bid or Last price change value                |
//+------------------------------------------------------------------+
double CSymbol::GetValueChangedBidLastHigh(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetControlChangedValue(SYMBOL_PROP_BIDHIGH) : this.GetControlChangedValue(SYMBOL_PROP_LASTHIGH));
  }
//+------------------------------------------------------------------+
//| Return the flag of a change of the maximum                       |
//| Bid or Last price exceeding the increase value                   |
//+------------------------------------------------------------------+
bool CSymbol::IsIncreasedBidLastHigh(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetControlFlagINC(SYMBOL_PROP_BIDHIGH) : (bool)this.GetControlFlagINC(SYMBOL_PROP_LASTHIGH));
  }
//+------------------------------------------------------------------+
//| Return the flag of a change of the maximum                       |
//| Bid or Last price exceeding the decrease value                   |
//+------------------------------------------------------------------+
bool CSymbol::IsDecreasedBidLastHigh(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetControlFlagDEC(SYMBOL_PROP_BIDHIGH) : (bool)this.GetControlFlagDEC(SYMBOL_PROP_LASTHIGH));
  }
//+------------------------------------------------------------------+
//| Set the controlled increase value                                |
//| of the minimum Bid or Last price                                 |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastLowInc(const double value)
  {
   this.SetControlPropertyINC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDLOW : SYMBOL_PROP_LASTLOW),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the controlled decrease value                                |
//| of the minimum Bid or Last price                                 |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastLowDec(const double value)
  {
   this.SetControlPropertyDEC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDLOW : SYMBOL_PROP_LASTLOW),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the minimum Bid or Last price control level                  |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastLowLevev(const double value)
  {
   this.SetControlPropertyLEVEL((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDLOW : SYMBOL_PROP_LASTLOW),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Return the minimum Bid or Last price change value                |
//+------------------------------------------------------------------+
double CSymbol::GetValueChangedBidLastLow(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetControlChangedValue(SYMBOL_PROP_BIDLOW) : this.GetControlChangedValue(SYMBOL_PROP_LASTLOW));
  }
//+------------------------------------------------------------------+
//| Return the flag of a change of the minimum                       |
//| Bid or Last price exceeding the increase value                   |
//+------------------------------------------------------------------+
bool CSymbol::IsIncreasedBidLastLow(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetControlFlagINC(SYMBOL_PROP_BIDLOW) : (bool)this.GetControlFlagINC(SYMBOL_PROP_LASTLOW));
  }
//+------------------------------------------------------------------+
//| Return the flag of a change of the minimum                       |
//| Bid or Last price exceeding the decrease value                   |
//+------------------------------------------------------------------+
bool CSymbol::IsDecreasedBidLastLow(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetControlFlagDEC(SYMBOL_PROP_BIDLOW) : (bool)this.GetControlFlagDEC(SYMBOL_PROP_LASTLOW));
  }
//+------------------------------------------------------------------+
```

We have already considered the similar methods when improving the base object class. The considered methods are called here depending
on a required symbol object property.

**This concludes the improvement of the symbol object class.**

Now it only remains to slightly refine the symbol collection class.

Open the \\MQL5\\Include\\DoEasy\\Collections\ **SymbolsCollection.mqh**
file and make the necessary changes.

Since we no longer need to create separate event enumerations for each object, set 'int' type for the **"last**
**symbol event" variable and the GetLastEvent()**
**method instead of the previous ENUM\_SYMBOL\_EVENT type:**

```
int m_last_event; // The last event
```

```
int GetLastEvent(void) const { return this.m_last_event; }
```

Since all symbol events (as well as events of any descendant object) are now handled in the base object class, rename the **EventDescription() method into**
**EventMWDescription() and pass the variable with the enumeration type**
**of the Market Watch window events to the method:**

```
//--- Return the description of the (1) Market Watch window event, (2) mode of working with symbols
   string            EventMWDescription(const ENUM_MW_EVENT event);
   string            ModeSymbolsListDescription(void);
```

Since the enumeration names have changed, **the method of working with the Market Watch window has undergone minor changes (enumeration**
**names and the event variable type have been**
**altered):**

```
//+------------------------------------------------------------------+
//| Working with market watch window events                          |
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
      ushort event_id=
        (ushort(
         this.m_total_symbols>this.m_total_symbol_prev ? MARKET_WATCH_EVENT_SYMBOL_ADD :
         this.m_total_symbols<this.m_total_symbol_prev ? MARKET_WATCH_EVENT_SYMBOL_DEL :
         MARKET_WATCH_EVENT_SYMBOL_SORT)
        );
      //--- Adding a symbol to the Market Watch window
      if(event_id==MARKET_WATCH_EVENT_SYMBOL_ADD)
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
      else if(event_id==MARKET_WATCH_EVENT_SYMBOL_DEL)
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
      else if(event_id==MARKET_WATCH_EVENT_SYMBOL_SORT)
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

**The event variable type has also been changed in the**
**method of working with the symbol collection event list:**

```
//+------------------------------------------------------------------+
//| Working with the events of the collection symbol list            |
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
         ushort event_id=event.ID();
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

**The names of the event enumeration constants have**
**also been changed in the method returning the string description of the market watch window events:**

```
//+------------------------------------------------------------------+
//| Return the Market Watch window event description                 |
//+------------------------------------------------------------------+
string CSymbolsCollection::EventMWDescription(const ENUM_MW_EVENT event)
  {
   return
     (
      event==MARKET_WATCH_EVENT_SYMBOL_ADD   ?  TextByLanguage("В окно \"Обзор рынка\" добавлен символ","Added symbol to \"Market Watch\" window")                                     :
      event==MARKET_WATCH_EVENT_SYMBOL_DEL   ?  TextByLanguage("Из окна \"Обзор рынка\" удалён символ","Removed symbol from \"Market Watch\" window")                                       :
      event==MARKET_WATCH_EVENT_SYMBOL_SORT  ?  TextByLanguage("Изменено расположение символов в окне \"Обзор рынка\"","Changed arrangement of symbols in \"Market Watch\" window")  :
      EnumToString(event)
     );
  }
//+------------------------------------------------------------------+
```

Now let's improve the CEngine class. Open the \\MQL5\\Include\\DoEasy\ **Engine.mqh** file and make the necessary
changes there:

**The variable storing the last event in symbol properties**
**and the method returning the value of the variable will**
**also be of 'int' type:**

```
   int m_last_symbol_event;  // Last event in the symbol properties
```

```
   int LastSymbolsEvent(void) const { return this.m_last_symbol_event; }
```

**In the public section of the class, add declaration of the method retrieving a 'ushort' number from the 'long' container at the**
**specified storage index in the 'long' parameter of the 'ushort' number:**

```
//--- Retrieve a necessary 'ushort' number from the packed 'long' value
   ushort               LongToUshortFromByte(const long source_value,const uchar index) const;
```

**Also, write the three methods immediately returning event milliseconds,**


**reason and source**
**from the event 'long' parameter:**

```
//--- Return event (1) milliseconds, (2) reason and (3) source from its 'long' value
   ushort               EventMSC(const long lparam)               const { return this.LongToUshortFromByte(lparam,0);         }
   ushort               EventReason(const long lparam)            const { return this.LongToUshortFromByte(lparam,1);         }
   ushort               EventSource(const long lparam)            const { return this.LongToUshortFromByte(lparam,2);         }
```

Since the zero value is the very first integer property of any object, **change**
**the initializing value for the variable storing the last symbol event** **in**
**the initialization list of the class constructor — now it will be initialized with a negative value:**

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),
                     m_last_trade_event(TRADE_EVENT_NO_EVENT),
                     m_last_account_event(ACCOUNT_EVENT_NO_EVENT),
                     m_last_symbol_event(WRONG_VALUE),
                     m_global_error(ERR_SUCCESS)
  {
```

**Implementing the method retrieving a 'ushort' number from the 'long' container by the byte index of its location in the 'long' container:**

```
//+------------------------------------------------------------------+
//| Retrieve a necessary 'ushort' number from the packed 'long' value|
//+------------------------------------------------------------------+
ushort CEngine::LongToUshortFromByte(const long source_value,const uchar index) const
  {
   if(index>3)
     {
      ::Print(DFUN,TextByLanguage("Ошибка. Значение \"index\" должно быть в пределах 0 - 3","Error. \"index\" value should be between 0 - 3"));
      return 0;
     }
   long res=source_value>>(16*index);
   return ushort(res &=0xFFFF);
  }
//+------------------------------------------------------------------+
```

The method receives a 'long' value a 'ushort' number should be retrieved from
and the byte index where the number is located (the table of
'ushort' numbers location in the 'long' container has been considered above). Next,

the validity of the index specification is checked. If the
index is invalid, an error message is displayed and

0 is returned.

Next, shift
'long' number bits by 16 \* index bits to the right, apply a mask for
"extinguishing" the remaining high bits and return a 'ushort' number
retrieved in such a manner.

To work in MQL4, we need to inform the compiler of the [ERR\_ZEROSIZE\_ARRAY](https://www.mql5.com/en/docs/constants/errorswarnings/errorcodes)
array zero size error.

The error most suitable for the array zero size out of the ones known to the MQL4 compiler is "invalid
array". Let's set it as an alternative to the array zero size error.



Open the \\MQL5\\Include\\DoEasy\ **ToMQL4.mqh** file and add the error code unknown for the MQL4 compiler:

```
//+------------------------------------------------------------------+
//|                                                       ToMQL4.mqh |
//|              Copyright 2017, Artem A. Trishkin, Skype artmedia70 |
//|                         https://www.mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Artem A. Trishkin, Skype artmedia70"
#property link      "https://www.mql5.com/en/users/artmedia70"
#property strict
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| Error codes                                                      |
//+------------------------------------------------------------------+
#define ERR_SUCCESS                       (ERR_NO_ERROR)
#define ERR_MARKET_UNKNOWN_SYMBOL         (ERR_UNKNOWN_SYMBOL)
#define ERR_ZEROSIZE_ARRAY                (ERR_ARRAY_INVALID)
//+------------------------------------------------------------------+
```

**These are all the changes that we needed in order to launch symbols to work with the new event functionality provided by the CBaseObj**
**object to all its descendants.**

### Testing the event functionality of the base object of all library objects

To test the new event functionality of the base object, take the [EA \\
from the previous article](https://www.mql5.com/en/articles/7071#node05) and save it under the name **TestDoEasyPart17.mq5** in \\MQL5\\Experts\\TestDoEasy\ **Part17**.

Let's test the spread change of the current symbol by 4 points (increase and decrease), as well as control a spread size by 15 points. For the
Bid price, control increase/decrease of its value by +/- 10 points and track the price crossing the level of 1.13700.

**To set the aforementioned monitored values, simply add the**
**following strings in the OnInit() handler in this example:**

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
//--- Fill in the array of used symbols
   used_symbols=InpUsedSymbols;
   CreateUsedSymbolsArray((ENUM_SYMBOLS_MODE)used_symbols_mode,used_symbols,array_used_symbols);

//--- Set the type of the used symbol list in the symbol collection
   engine.SetUsedSymbols(array_used_symbols);
//--- Displaying the selected mode of working with the symbol object collection
   Print(engine.ModeSymbolsListDescription(),TextByLanguage(". Количество используемых символов: ",". Number of symbols used: "),engine.GetSymbolsCollectionTotal());

//--- Set controlled values for the current symbol
   CSymbol* symbol=engine.GetSymbolCurrent();
   if(symbol!=NULL)
     {
      //--- Set control of the current symbol price increase by 10 points
      symbol.SetControlBidInc(10*Point());
      //--- Set control of the current symbol price decrease by 10 points
      symbol.SetControlBidDec(10*Point());
      //--- Set control of the current symbol spread increase by 4 points
      symbol.SetControlSpreadInc(4);
      //--- Set control of the current symbol spread decrease by 4 points
      symbol.SetControlSpreadDec(4);
      //--- Set control of the current spread by the value of 15 points
      symbol.SetControlSpreadLevel(15);
      //--- Set control of the price crossing the level of 1.13700
      symbol.SetControlBidLevel(1.13700);
     }

//--- Check and remove remaining EA graphical objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);

//--- Set CTrade trading class parameters
#ifdef __MQL5__
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
#endif
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

This is a test example of setting tracked symbol parameters, therefore we immediately set the required control values in OnInit().


However, nothing prevents us from quickly changing tracked symbol values based on some current criteria during the operation since
all the methods are present in the base object. It only remains to gain access to any of the objects inherited from CBaseObj to obtain the
methods for setting controlled parameters and methods for receiving changed parameters, as well as change controlled parameters
according to the logic built into the program — either programmatically or from the library graphical shell to be subsequently
created.

**From the EA's OnTick() handler, remove the variable storing the last**
**symbol event**. We have other tools for tracking symbol events rather than a simple comparison of the current and previous states.

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
```

**Change the library event handler regarding handling the symbol collection**
**events:**

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

//--- Retrieve (1) event time milliseconds, (2) reason and (3) source from lparam, as well as (4) set the exact event time
   ushort msc=engine.EventMSC(lparam);
   ushort reason=engine.EventReason(lparam);
   ushort source=engine.EventSource(lparam);
   long time=TimeCurrent()*1000+msc;

//--- Handling market watch window events
   if(idx>MARKET_WATCH_EVENT_NO_EVENT && idx<SYMBOL_EVENTS_NEXT_CODE)
     {
      string name="";
      //--- Market Watch window event
      string descr=engine.GetMWEventDescription((ENUM_MW_EVENT)idx);
      name=(idx==MARKET_WATCH_EVENT_SYMBOL_SORT ? "" : ": "+sparam);
      Print(TimeMSCtoString(lparam)," ",descr,name);
     }
//--- Handling symbol events
   if(source==COLLECTION_SYMBOLS_ID)
     {
      CSymbol *symbol=engine.GetSymbolObjByName(sparam);
      if(symbol==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=(idx<SYMBOL_PROP_INTEGER_TOTAL ? 0 : symbol.Digits());
      //--- Event text description
      string id_descr=(idx<SYMBOL_PROP_INTEGER_TOTAL ? symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_INTEGER)idx) : symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=DoubleToString(dparam,digits);
      //--- Check event reasons and display its description in the journal
      if(reason==BASE_EVENT_REASON_INC)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_DEC)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling trading events
   if(idx>TRADE_EVENT_NO_EVENT && idx<TRADE_EVENTS_NEXT_CODE)
     {
      event=EnumToString((ENUM_TRADE_EVENT)ushort(idx));
      int digits=(int)SymbolInfoInteger(sparam,SYMBOL_DIGITS);
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
         //--- the m_control_equity_inc variable tracking the equity increase by 15 units (by default)
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
  }
//+------------------------------------------------------------------+
```

All changes are commented on in the code and related only to obtaining an event description from the symbol object and displaying it in the
journal depending on the event reason. In the non-test handler, add the normal event handler instead of displaying a message in the journal.

**Compile and launch the EA in the tester:**

![](https://c.mql5.com/2/37/Jol5eSxMb5.gif)

As we can see, when a spread is increased or decreased beyond the specified control values, the appropriate entries are sent to the journal.
Changes in the Bid price (its increase or decrease by more than 10 points) are also accompanied by journal entries. Finally, when the Bid
price crosses the specified control level, an event is sent as well and the journal entry appears.

Thus, we have created the base object allowing us to track events of any of its descendant objects and send them to the control program where the
program can track them and react according to its built-in logic, as well as set new tracked values and levels enabling flexible management
of the program operation logic.

### What's next?

In the next article, we will implement the work of the account object and its events based on the event functionality of the CBaseObj base
object class.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7124#node00)

**Previous articles within the series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part \\
6\. Netting account events](https://www.mql5.com/en/articles/6383)

[Part 7. StopLimit order activation events, preparing \\
the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part 8. Order and \\
position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 - Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and activating pending orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure events](https://www.mql5.com/en/articles/6921)

[Part \\
12\. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part 13. Account object \\
events](https://www.mql5.com/en/articles/6995)

[Part 14. Symbol object](https://www.mql5.com/en/articles/7014)

[Part \\
15\. Symbol object collection](https://www.mql5.com/en/articles/7041)

[Part 16. Symbol collection events](https://www.mql5.com/en/articles/7071)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7124](https://www.mql5.com/ru/articles/7124)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7124.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7124/mql5.zip "Download MQL5.zip")(218.23 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7124/mql4.zip "Download MQL4.zip")(218.22 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/326278)**
(37)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
13 Nov 2019 at 05:09

**Marcin Rutkowski:**

Thank You  for all the work You put in to those articles. I can learn a lot from them :)

You are welcome.


![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
22 Sep 2020 at 16:13

Hello - I very much like the potential of the event-driven capability you implemented in CSymbol, such as SetControlBidDec, SetControlAskLevel, etc... I haven\`t studied your later articles on indicators yet (after part 38), but would like to understand if you intend to add the ability to set similar events on indicators (e.g. price touching/crossing a certain moving average value, etc)?

What is your recommended approach to implement such [checks](https://www.mql5.com/en/docs/common/GetTickCount64 "MQL5 Documentation: GetTickCount64 function") in the current version of the library? Is there an alternative to checking every time in OnTick()? What about if I'm working with multiple symbols?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
22 Sep 2020 at 21:24

**Dima Diall :**

Hello - I very much like the potential of the event-driven capability you implemented in CSymbol, such as SetControlBidDec, SetControlAskLevel, etc... I haven\`t studied your later articles on indicators yet (after part 38), but would like to understand if you intend to add the ability to set similar events on indicators (e.g. price touching/crossing a certain moving average value, etc)?

What is your recommended approach to implement such checks in the current version of the library? Is there an alternative to checking every time in OnTick()? What about if I'm working with multiple symbols?

Hi. I have not thought about implementing the same [event model](https://www.mql5.com/en/articles/2169 "Article: Universal Expert Advisor: Event Model and Prototype Trading Strategy (Part 2) ") for indicators yet. I am not very happy with the implementation of indicators. Therefore, I will add indicator objects, and they will already refer to the required bars in the timeseries for the required indicator data. And there, the implementation of the event model will already be easier.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
23 Sep 2020 at 13:26

**Artyom Trishkin:**

Hi. I have not thought about implementing the same event model for indicators yet. I am not very happy with the implementation of indicators. Therefore, I will add indicator objects, and they will already refer to the required bars in the timeseries for the required indicator data. And there, the implementation of the event model will already be easier.

Is the main purpose of your indicators support in DoEasy to help the library user to implement their own indicators? Do you also plan to allow multi-platform access to any indicator data in MT4/MT5from other programs, for example EAs? As you know, currently [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") such as iMACD(), iBands(), etc work very differently between MQL4 and MQL5 so I want to write some wrapper functions to enable my EAs coded with DoEasy can run on both versions.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
23 Sep 2020 at 13:31

**Dima Diall :**

Is the main purpose of your indicators support in DoEasy to help the library user to implement their own indicators? Do you also plan to allow multi-platform access to any indicator data in MT4/MT5 from other programs, for example EAs? As you know, currently [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") such as iMACD(), iBands(), etc work very differently between MQL4 and MQL5 so I want to write some wrapper functions to enable my EAs coded with DoEasy can run on both versions.

Yes. I have already started writing an article about indicator objects. It is with their help that everything will be simple and, I hope, the way most users of the library will be satisfied.

![Library for easy and quick development of MetaTrader programs (part XVIII): Interactivity of account and any other library objects](https://c.mql5.com/2/37/MQL5-avatar-doeasy.png)[Library for easy and quick development of MetaTrader programs (part XVIII): Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

The article arranges the work of an account object on a new base object of all library objects, improves the CBaseObj base object and tests setting tracked parameters, as well as receiving events for any library objects.

![Strategy builder based on Merrill patterns](https://c.mql5.com/2/37/Article_Logo.png)[Strategy builder based on Merrill patterns](https://www.mql5.com/en/articles/7218)

In the previous article, we considered application of Merrill patterns to various data, such as to a price value on a currency symbol chart and values of standard MetaTrader 5 indicators: ATR, WPR, CCI, RSI, among others. Now, let us try to create a strategy construction set based on Merrill patterns.

![Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages](https://c.mql5.com/2/37/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages](https://www.mql5.com/en/articles/7176)

In this article, we will consider the class of displaying text messages. Currently, we have a sufficient number of different text messages. It is time to re-arrange the methods of their storage, display and translation of Russian or English messages to other languages. Besides, it would be good to introduce convenient ways of adding new languages to the library and quickly switching between them.

![Library for easy and quick development of MetaTrader programs (part XVI): Symbol collection events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__11.png)[Library for easy and quick development of MetaTrader programs (part XVI): Symbol collection events](https://www.mql5.com/en/articles/7071)

In this article, we will create a new base class of all library objects adding the event functionality to all its descendants and develop the class for tracking symbol collection events based on the new base class. We will also change account and account event classes for developing the new base object functionality.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ghdrzhtmczdmvxxnhirhprbwpmhfrqkd&ssn=1769186366753207090&ssn_dr=0&ssn_sr=0&fv_date=1769186366&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7124&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XVII)%3A%20Interactivity%20of%20library%20objects%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918636667489296&fz_uniq=6383576078116394558&sv=2552)

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