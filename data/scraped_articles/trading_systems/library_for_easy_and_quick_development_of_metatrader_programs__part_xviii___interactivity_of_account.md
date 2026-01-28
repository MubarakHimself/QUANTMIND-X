---
title: Library for easy and quick development of MetaTrader programs (part XVIII): Interactivity of account and any other library objects
url: https://www.mql5.com/en/articles/7149
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:39:18.374774
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/7149&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070457902596036146)

MetaTrader 5 / Examples


### Contents

- [Improving the base and symbol objects](https://www.mql5.com/en/articles/7149#node01)
- [Revamping the account object](https://www.mql5.com/en/articles/7149#node02)
- [Testing the setting of the tracking parameters and receiving object events](https://www.mql5.com/en/articles/7149#node03)
- [What's next?](https://www.mql5.com/en/articles/7149#node04)


We have created the base object all other library objects in need of the event functionality are to be inherited from. Now it has become much
easier to set the parameters of tracked events and receive them from objects — all is done in the base class, and for all of its descendant
objects, we need to carry out the same steps to set the monitored properties and values we want to track in the object properties.

Today we will slightly improve the base object and connect the account object to it. We will also test the ability to set the tracked object
properties and their values using a test EA as an example.

[In the previous article,](https://www.mql5.com/en/articles/7124) when creating the methods of the CBaseObj base
object and its descendant (the symbol object), I got carried away and implemented a set of methods in two classes that actually duplicate
each other. Let's fix this — simply remove the methods for setting and receiving the properties from the CSymbol descendant object class and
arrange the CBaseObj base object methods.

### Improving the base and symbol objects

Open the file \\MQL5\\Include\\DoEasy\\Objects\ **BaseObj.mqh** and make some changes to it.

Move the **CheckEvents()** method from the **CSymbol** class to the **CBaseObj** base class since this method is
absolutely identical for any descendant objects of the base class, which means the base class is the right place for it.

**Let's add its definition to the protected section of the**
**CBaseObj class:**

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
   int               m_event_id;                               // Event ID (equal to the object property value)
   string            m_name;                                   // Object name
   string            m_folder_name;                            // Name of the folder storing CBaseObj descendant objects
   bool              m_first_start;                            // First launch flag
   int               m_type;                                   // Object type (corresponds to the collection IDs)

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
   bool              CheckControlDataArraySize(bool check_long=true);

//--- Check the list of symbol property changes and create an event
   void              CheckEvents(void);

//--- (1) Pack a 'ushort' number to a passed 'long' number
//--- (2) convert a 'ushort' value to a specified 'long' number byte
   long              UshortToLong(const ushort ushort_value,const uchar to_byte,long &long_value);

protected:
   long              UshortToByte(const ushort value,const uchar index) const;

public:
```

**From the public section, remove the methods setting the values to event**
**flags** — events occur regardless of our discretion, and the flags are set without user intervention. Therefore, I believe, it is
reasonable to have the methods that forcibly set event flags:

```
//--- Set the flag of a property change exceeding the (1) increase and (2) decrease values
   template<typename T> void  SetControlledFlagINC(const int property,const T value);
   template<typename T> void  SetControlledFlagDEC(const int property,const T value);
//--- Set the flag of a property change (1) exceeding, (2) being less than the control level, (3) being equal to the level
   template<typename T> void  SetControlledFlagMORE(const int property,const T value);
   template<typename T> void  SetControlledFlagLESS(const int property,const T value);
   template<typename T> void  SetControlledFlagEQUAL(const int property,const T value);
```

The methods returning controlled object property values set by users, the current object property value and the values of object property
changes

**will now have more readable and meaningful names:**

```
//--- Return the set value of the controlled (1) integer and (2) real object properties increase
   long              GetControlledLongValueINC(const int property)         const { return this.m_long_prop_event[property][0];                           }
   double            GetControlledDoubleValueINC(const int property)       const { return this.m_double_prop_event[property-this.m_long_prop_total][0];  }
//--- Return the set value of the controlled (1) integer and (2) real object properties decrease
   long              GetControlledLongValueDEC(const int property)         const { return this.m_long_prop_event[property][1];                           }
   double            GetControlledDoubleValueDEC(const int property)       const { return this.m_double_prop_event[property-this.m_long_prop_total][1];  }
//--- Return the specified control level of object's (1) integer and (2) real properties
   long              GetControlledLongValueLEVEL(const int property)       const { return this.m_long_prop_event[property][2];                           }
   double            GetControlledDoubleValueLEVEL(const int property)     const { return this.m_double_prop_event[property-this.m_long_prop_total][2];  }

//--- Return the current value of the object (1) integer and (2) real property
   long              GetPropLongValue(const int property)            const { return this.m_long_prop_event[property][3];                           }
   double            GetPropDoubleValue(const int property)          const { return this.m_double_prop_event[property-this.m_long_prop_total][3];  }
//--- Return the change value of the controlled (1) integer and (2) real object property
   long              GetPropLongChangedValue(const int property)     const { return this.m_long_prop_event[property][4];                           }
   double            GetPropDoubleChangedValue(const int property)   const { return this.m_double_prop_event[property-this.m_long_prop_total][4];  }

//--- Return the flag of an (1) integer and (2) real property value change exceeding the increase value
   long              GetPropLongFlagINC(const int property)          const { return this.m_long_prop_event[property][5];                           }
   double            GetPropDoubleFlagINC(const int property)        const { return this.m_double_prop_event[property-this.m_long_prop_total][5];  }
//--- Return the flag of an (1) integer and (2) real property value change exceeding the decrease value
   long              GetPropLongFlagDEC(const int property)          const { return this.m_long_prop_event[property][6];                           }
   double            GetPropDoubleFlagDEC(const int property)        const { return this.m_double_prop_event[property-this.m_long_prop_total][6];  }
//--- Return the flag of an (1) integer and (2) real property value increase exceeding the control level
   long              GetPropLongFlagMORE(const int property)         const { return this.m_long_prop_event[property][7];                           }
   double            GetPropDoubleFlagMORE(const int property)       const { return this.m_double_prop_event[property-this.m_long_prop_total][7];  }
//--- Return the flag of an (1) integer and (2) real property value decrease being less than the control level
   long              GetPropLongFlagLESS(const int property)         const { return this.m_long_prop_event[property][8];                           }
   double            GetPropDoubleFlagLESS(const int property)       const { return this.m_double_prop_event[property-this.m_long_prop_total][8];  }
//--- Return the flag of an (1) integer and (2) real property being equal to the control level
   long              GetPropLongFlagEQUAL(const int property)        const { return this.m_long_prop_event[property][9];                           }
   double            GetPropDoubleFlagEQUAL(const int property)      const { return this.m_double_prop_event[property-this.m_long_prop_total][9];  }
```

When conducting the in-depth testing of the **CBaseObj** base object event functionality, the event definitions errors
were detected. The method for filling in the base object properties and searching for the

**FillPropertySettings()** events featured an inaccuracy — at the very end of the method, the current property status was written to the
previous one in any case. This prevented from defining a property change value, since a newly received property value was immediately
written to the previous one. In case of a small controlled change value, this error did not occur since the property managed to change by an
amount exceeding the one specified for event generation in one go.

But when it was necessary to track considerable changes occurring in more than one tick, that was impossible.

**This**
**has been fixed**— **the current state is written to the previous one only when**
**registering an event:**

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
         //--- Save the current property value as a previous one
         array_prev[index][3]=array[index][3];
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
         //--- Save the current property value as a previous one
         array_prev[index][3]=array[index][3];
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
         //--- Save the current property value as a previous one
         array_prev[index][3]=array[index][3];
        }
      //--- If a property value is less than the control level, there is an event,
      //--- add the event to the list and set the flag
      else if(value<0 && array_prev[index][3]>=array[index][2])
        {
         if(this.EventBaseAdd(event_id,BASE_EVENT_REASON_LESS_THEN,array[index][2]))
            array[index][8]=true;
         //--- Save the current property value as a previous one
         array_prev[index][3]=array[index][3];
        }
      //--- If a property value is equal to the control level, there is an event,
      //--- add the event to the list and set the flag
      else if(value==0 && array_prev[index][3]!=array[index][2])
        {
         if(this.EventBaseAdd(event_id,BASE_EVENT_REASON_EQUALS,array[index][2]))
            array[index][9]=true;
         //--- Save the current property value as a previous one
         array_prev[index][3]=array[index][3];
        }
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Previously, the properties were initialized using LONG\_MAX values in the base
object properties initialization method,

**now they are to be initialized using zero:**

```
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
         this.m_long_prop_event[i][j]=0;
   for(int i=this.m_double_prop_total-1;i>WRONG_VALUE;i--)
      for(int j=3; j<CONTROLS_TOTAL; j++)
         this.m_double_prop_event[i][j]=0;
  }
//+------------------------------------------------------------------+
```

**Add the implementation of the CheckEvents() method transferred from CSymbol:**

```
//+------------------------------------------------------------------+
//| Check the list of object property changes and create an event    |
//+------------------------------------------------------------------+
void CBaseObj::CheckEvents(void)
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
      this.UshortToLong((ushort)this.m_type,2,lvalue);
      if(this.EventAdd((ushort)event.ID(),lvalue,event.Value(),this.m_name))
         this.m_is_event=true;
     }
  }
//+------------------------------------------------------------------+
```

Make corrections in the CSymbol symbol object class.

Open \\MQL5\\Include\\DoEasy\\Objects\\Symbols\ **Symbol.mqh**
file and make the changes.

**Remove two methods from the private class section:**

```
private:
   struct MqlMarginRate
     {
      double         Initial;                                  // initial margin rate
      double         Maintenance;                              // maintenance margin rate
     };
   struct MqlMarginRateMode
     {
      MqlMarginRate  Long;                                     // MarginRate of long positions
      MqlMarginRate  Short;                                    // MarginRate of short positions
      MqlMarginRate  BuyStop;                                  // MarginRate of BuyStop orders
      MqlMarginRate  BuyLimit;                                 // MarginRate of BuyLimit orders
      MqlMarginRate  BuyStopLimit;                             // MarginRate of BuyStopLimit orders
      MqlMarginRate  SellStop;                                 // MarginRate of SellStop orders
      MqlMarginRate  SellLimit;                                // MarginRate of SellLimit orders
      MqlMarginRate  SellStopLimit;                            // MarginRate of SellStopLimit orders
     };
   MqlMarginRateMode m_margin_rate;                            // Margin ratio structure
   MqlBookInfo       m_book_info_array[];                      // Array of the market depth data structures

   long              m_long_prop[SYMBOL_PROP_INTEGER_TOTAL];   // Integer properties
   double            m_double_prop[SYMBOL_PROP_DOUBLE_TOTAL];  // Real properties
   string            m_string_prop[SYMBOL_PROP_STRING_TOTAL];  // String properties
   bool              m_is_change_trade_mode;                   // Flag of changing trading mode for a symbol

//--- Initialize the variables of controlled symbol data
   virtual void      InitControlsParams(void);
//--- Check the list of symbol property changes and create an event
   void              CheckEvents(void);
```

Controlled data initialization method is not needed since all
properties and parameters of any class necessary for tracking should be performed by explicit specification of tracked parameter values.
By default, no changes of descendant object properties are tracked. We have already moved the

CheckEvents() method from here to the CBaseObj base class.

**From the public class section, remove methods that**
**actually duplicated the base object methods:**

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
```

The methods of receiving and setting parameters of tracked symbol properties now directly use the base class methods to return the values.
Below is the

**method list:**

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
   void              SetControlSessionDealsInc(const long value)                 { this.SetControlledValueINC(SYMBOL_PROP_SESSION_DEALS,(long)::fabs(value));        }
   void              SetControlSessionDealsDec(const long value)                 { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_DEALS,(long)::fabs(value));        }
   void              SetControlSessionDealsLevel(const long value)               { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_DEALS,(long)::fabs(value));      }
   long              GetValueChangedSessionDeals(void)                     const { return this.GetPropLongChangedValue(SYMBOL_PROP_SESSION_DEALS);                   }
   bool              IsIncreasedSessionDeals(void)                         const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_SESSION_DEALS);                  }
   bool              IsDecreasedSessionDeals(void)                         const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_SESSION_DEALS);                  }
   //--- Buy orders of the current session
   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the current number of Buy orders
   //--- getting (4) the current number of Buy orders change value,
   //--- getting the flag of the current Buy orders' number change exceeding the (5) increase, (6) decrease value
   void              SetControlSessionBuyOrdInc(const long value)                { this.SetControlledValueINC(SYMBOL_PROP_SESSION_BUY_ORDERS,(long)::fabs(value));   }
   void              SetControlSessionBuyOrdDec(const long value)                { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_BUY_ORDERS,(long)::fabs(value));   }
   void              SetControlSessionBuyOrdLevel(const long value)              { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_BUY_ORDERS,(long)::fabs(value)); }
   long              GetValueChangedSessionBuyOrders(void)                 const { return this.GetPropLongChangedValue(SYMBOL_PROP_SESSION_BUY_ORDERS);              }
   bool              IsIncreasedSessionBuyOrders(void)                     const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_SESSION_BUY_ORDERS);             }
   bool              IsDecreasedSessionBuyOrders(void)                     const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_SESSION_BUY_ORDERS);             }
   //--- Sell orders of the current session
   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the current number of Sell orders
   //--- getting (4) the current number of Sell orders change value,
   //--- getting the flag of the current Sell orders' number change exceeding the (5) increase, (6) decrease value
   void              SetControlSessionSellOrdInc(const long value)               { this.SetControlledValueINC(SYMBOL_PROP_SESSION_SELL_ORDERS,(long)::fabs(value));  }
   void              SetControlSessionSellOrdDec(const long value)               { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_SELL_ORDERS,(long)::fabs(value));  }
   void              SetControlSessionSellOrdLevel(const long value)             { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_SELL_ORDERS,(long)::fabs(value));}
   long              GetValueChangedSessionSellOrders(void)                const { return this.GetPropLongChangedValue(SYMBOL_PROP_SESSION_SELL_ORDERS);             }
   bool              IsIncreasedSessionSellOrders(void)                    const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_SESSION_SELL_ORDERS);            }
   bool              IsDecreasedSessionSellOrders(void)                    const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_SESSION_SELL_ORDERS);            }
   //--- Volume of the last deal
   //--- setting the last deal volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) volume change values in the last deal,
   //--- getting the flag of the volume change in the last deal exceeding the (5) increase, (6) decrease value
   void              SetControlVolumeInc(const long value)                       { this.SetControlledValueINC(SYMBOL_PROP_VOLUME,(long)::fabs(value));               }
   void              SetControlVolumeDec(const long value)                       { this.SetControlledValueDEC(SYMBOL_PROP_VOLUME,(long)::fabs(value));               }
   void              SetControlVolumeLevel(const long value)                     { this.SetControlledValueLEVEL(SYMBOL_PROP_VOLUME,(long)::fabs(value));             }
   long              GetValueChangedVolume(void)                           const { return this.GetPropLongChangedValue(SYMBOL_PROP_VOLUME);                          }
   bool              IsIncreasedVolume(void)                               const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_VOLUME);                         }
   bool              IsDecreasedVolume(void)                               const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_VOLUME);                         }
   //--- Maximum volume within a day
   //--- setting the maximum day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the maximum volume change value within a day,
   //--- getting the flag of the maximum day volume change exceeding the (5) increase, (6) decrease value
   void              SetControlVolumeHighInc(const long value)                   { this.SetControlledValueINC(SYMBOL_PROP_VOLUMEHIGH,(long)::fabs(value));           }
   void              SetControlVolumeHighDec(const long value)                   { this.SetControlledValueDEC(SYMBOL_PROP_VOLUMEHIGH,(long)::fabs(value));           }
   void              SetControlVolumeHighLevel(const long value)                 { this.SetControlledValueLEVEL(SYMBOL_PROP_VOLUMEHIGH,(long)::fabs(value));         }
   long              GetValueChangedVolumeHigh(void)                       const { return this.GetPropLongChangedValue(SYMBOL_PROP_VOLUMEHIGH);                      }
   bool              IsIncreasedVolumeHigh(void)                           const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_VOLUMEHIGH);                     }
   bool              IsDecreasedVolumeHigh(void)                           const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_VOLUMEHIGH);                     }
   //--- Minimum volume within a day
   //--- setting the minimum day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the minimum volume change value within a day,
   //--- getting the flag of the minimum day volume change exceeding the (5) increase, (6) decrease value
   void              SetControlVolumeLowInc(const long value)                    { this.SetControlledValueINC(SYMBOL_PROP_VOLUMELOW,(long)::fabs(value));            }
   void              SetControlVolumeLowDec(const long value)                    { this.SetControlledValueDEC(SYMBOL_PROP_VOLUMELOW,(long)::fabs(value));            }
   void              SetControlVolumeLowLevel(const long value)                  { this.SetControlledValueLEVEL(SYMBOL_PROP_VOLUMELOW,(long)::fabs(value));          }
   long              GetValueChangedVolumeLow(void)                        const { return this.GetPropLongChangedValue(SYMBOL_PROP_VOLUMELOW);                       }
   bool              IsIncreasedVolumeLow(void)                            const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_VOLUMELOW);                      }
   bool              IsDecreasedVolumeLow(void)                            const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_VOLUMELOW);                      }
   //--- Spread
   //--- setting the controlled spread (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) spread change value in points,
   //--- getting the flag of the spread change in points exceeding the (5) increase, (6) decrease value
   void              SetControlSpreadInc(const int value)                        { this.SetControlledValueINC(SYMBOL_PROP_SPREAD,(long)::fabs(value));               }
   void              SetControlSpreadDec(const int value)                        { this.SetControlledValueDEC(SYMBOL_PROP_SPREAD,(long)::fabs(value));               }
   void              SetControlSpreadLevel(const int value)                      { this.SetControlledValueLEVEL(SYMBOL_PROP_SPREAD,(long)::fabs(value));             }
   int               GetValueChangedSpread(void)                           const { return (int)this.GetPropLongChangedValue(SYMBOL_PROP_SPREAD);                     }
   bool              IsIncreasedSpread(void)                               const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_SPREAD);                         }
   bool              IsDecreasedSpread(void)                               const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_SPREAD);                         }
   //--- StopLevel
   //--- setting the controlled StopLevel (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) StopLevel change value in points,
   //--- getting the flag of StopLevel change in points exceeding the (5) increase, (6) decrease value
   void              SetControlStopLevelInc(const int value)                     { this.SetControlledValueINC(SYMBOL_PROP_TRADE_STOPS_LEVEL,(long)::fabs(value));    }
   void              SetControlStopLevelDec(const int value)                     { this.SetControlledValueDEC(SYMBOL_PROP_TRADE_STOPS_LEVEL,(long)::fabs(value));    }
   void              SetControlStopLevelLevel(const int value)                   { this.SetControlledValueLEVEL(SYMBOL_PROP_TRADE_STOPS_LEVEL,(long)::fabs(value));  }
   int               GetValueChangedStopLevel(void)                        const { return (int)this.GetPropLongChangedValue(SYMBOL_PROP_TRADE_STOPS_LEVEL);          }
   bool              IsIncreasedStopLevel(void)                            const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_TRADE_STOPS_LEVEL);              }
   bool              IsDecreasedStopLevel(void)                            const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_TRADE_STOPS_LEVEL);              }
   //--- Freeze distance
   //--- setting the controlled FreezeLevel (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) FreezeLevel change value in points,
   //--- getting the flag of FreezeLevel change in points exceeding the (5) increase, (6) decrease value
   void              SetControlFreezeLevelInc(const int value)                   { this.SetControlledValueINC(SYMBOL_PROP_TRADE_FREEZE_LEVEL,(long)::fabs(value));   }
   void              SetControlFreezeLevelDec(const int value)                   { this.SetControlledValueDEC(SYMBOL_PROP_TRADE_FREEZE_LEVEL,(long)::fabs(value));   }
   void              SetControlFreezeLevelLevel(const int value)                 { this.SetControlledValueLEVEL(SYMBOL_PROP_TRADE_FREEZE_LEVEL,(long)::fabs(value)); }
   int               GetValueChangedFreezeLevel(void)                      const { return (int)this.GetPropLongChangedValue(SYMBOL_PROP_TRADE_FREEZE_LEVEL);         }
   bool              IsIncreasedFreezeLevel(void)                          const { return (bool)this.GetPropLongFlagINC(SYMBOL_PROP_TRADE_FREEZE_LEVEL);             }
   bool              IsDecreasedFreezeLevel(void)                          const { return (bool)this.GetPropLongFlagDEC(SYMBOL_PROP_TRADE_FREEZE_LEVEL);             }

   //--- Bid
   //--- setting the controlled Bid price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) Bid or Last price change value,
   //--- getting the flag of the Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlBidInc(const double value)                        { this.SetControlledValueINC(SYMBOL_PROP_BID,::fabs(value));                        }
   void              SetControlBidDec(const double value)                        { this.SetControlledValueDEC(SYMBOL_PROP_BID,::fabs(value));                        }
   void              SetControlBidLevel(const double value)                      { this.SetControlledValueLEVEL(SYMBOL_PROP_BID,::fabs(value));                      }
   double            GetValueChangedBid(void)                              const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_BID);                           }
   bool              IsIncreasedBid(void)                                  const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_BID);                          }
   bool              IsDecreasedBid(void)                                  const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_BID);                          }
   //--- The highest Bid price of the day
   //--- setting the controlled maximum Bid price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) maximum Bid or Last price change value,
   //--- getting the flag of the maximum Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlBidHighInc(const double value)                    { this.SetControlledValueINC(SYMBOL_PROP_BIDHIGH,::fabs(value));                    }
   void              SetControlBidHighDec(const double value)                    { this.SetControlledValueDEC(SYMBOL_PROP_BIDHIGH,::fabs(value));                    }
   void              SetControlBidHighLevel(const double value)                  { this.SetControlledValueLEVEL(SYMBOL_PROP_BIDHIGH,::fabs(value));                  }
   double            GetValueChangedBidHigh(void)                          const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_BIDHIGH);                       }
   bool              IsIncreasedBidHigh(void)                              const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_BIDHIGH);                      }
   bool              IsDecreasedBidHigh(void)                              const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_BIDHIGH);                      }
   //--- The lowest Bid price of the day
   //--- setting the controlled minimum Bid price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) minimum Bid or Last price change value,
   //--- getting the flag of the minimum Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlBidLowInc(const double value)                     { this.SetControlledValueINC(SYMBOL_PROP_BIDLOW,::fabs(value));                     }
   void              SetControlBidLowDec(const double value)                     { this.SetControlledValueDEC(SYMBOL_PROP_BIDLOW,::fabs(value));                     }
   void              SetControlBidLowLevel(const double value)                   { this.SetControlledValueLEVEL(SYMBOL_PROP_BIDLOW,::fabs(value));                   }
   double            GetValueChangedBidLow(void)                           const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_BIDLOW);                        }
   bool              IsIncreasedBidLow(void)                               const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_BIDLOW);                       }
   bool              IsDecreasedBidLow(void)                               const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_BIDLOW);                       }

   //--- Last
   //--- setting the controlled Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) Bid or Last price change value,
   //--- getting the flag of the Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlLastInc(const double value)                       { this.SetControlledValueINC(SYMBOL_PROP_LAST,::fabs(value));                       }
   void              SetControlLastDec(const double value)                       { this.SetControlledValueDEC(SYMBOL_PROP_LAST,::fabs(value));                       }
   void              SetControlLastLevel(const double value)                     { this.SetControlledValueLEVEL(SYMBOL_PROP_LAST,::fabs(value));                     }
   double            GetValueChangedLast(void)                             const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_LAST);                          }
   bool              IsIncreasedLast(void)                                 const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_LAST);                         }
   bool              IsDecreasedLast(void)                                 const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_LAST);                         }
   //--- The highest Last price of the day
   //--- setting the controlled maximum Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) maximum Bid or Last price change value,
   //--- getting the flag of the maximum Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlLastHighInc(const double value)                   { this.SetControlledValueINC(SYMBOL_PROP_LASTHIGH,::fabs(value));                   }
   void              SetControlLastHighDec(const double value)                   { this.SetControlledValueDEC(SYMBOL_PROP_LASTHIGH,::fabs(value));                   }
   void              SetControlLastHighLevel(const double value)                 { this.SetControlledValueLEVEL(SYMBOL_PROP_LASTHIGH,::fabs(value));                 }
   double            GetValueChangedLastHigh(void)                         const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_LASTHIGH);                      }
   bool              IsIncreasedLastHigh(void)                             const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_LASTHIGH);                     }
   bool              IsDecreasedLastHigh(void)                             const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_LASTHIGH);                     }
   //--- The lowest Last price of the day
   //--- setting the controlled minimum Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) minimum Bid or Last price change value,
   //--- getting the flag of the minimum Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlLastLowInc(const double value)                    { this.SetControlledValueINC(SYMBOL_PROP_LASTLOW,::fabs(value));                    }
   void              SetControlLastLowDec(const double value)                    { this.SetControlledValueDEC(SYMBOL_PROP_LASTLOW,::fabs(value));                    }
   void              SetControlLastLowLevel(const double value)                  { this.SetControlledValueLEVEL(SYMBOL_PROP_LASTLOW,::fabs(value));                  }
   double            GetValueChangedLastLow(void)                          const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_LASTLOW);                       }
   bool              IsIncreasedLastLow(void)                              const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_LASTLOW);                      }
   bool              IsDecreasedLastLow(void)                              const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_LASTLOW);                      }

   //--- Bid/Last
   //--- setting the controlled Bid or Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) Bid or Last price change value,
   //--- getting the flag of the Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlBidLastInc(const double value);
   void              SetControlBidLastDec(const double value);
   void              SetControlBidLastLevel(const double value);
   double            GetValueChangedBidLast(void)                          const;
   bool              IsIncreasedBidLast(void)                              const;
   bool              IsDecreasedBidLast(void)                              const;
   //--- Maximum Bid/Last of the day
   //--- setting the controlled maximum Bid or Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) maximum Bid or Last price change value,
   //--- getting the flag of the maximum Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlBidLastHighInc(const double value);
   void              SetControlBidLastHighDec(const double value);
   void              SetControlBidLastHighLevel(const double value);
   double            GetValueChangedBidLastHigh(void)                      const;
   bool              IsIncreasedBidLastHigh(void)                          const;
   bool              IsDecreasedBidLastHigh(void)                          const;
   //--- Minimum Bid/Last of the day
   //--- setting the controlled minimum Bid or Last price (1) increase, (2) decrease value and (3) control level in points
   //--- getting the (4) minimum Bid or Last price change value,
   //--- getting the flag of the minimum Bid or Last price change exceeding the (5) increase, (6) decrease value
   void              SetControlBidLastLowInc(const double value);
   void              SetControlBidLastLowDec(const double value);
   void              SetControlBidLastLowLevev(const double value);
   double            GetValueChangedBidLastLow(void)                       const;
   bool              IsIncreasedBidLastLow(void)                           const;
   bool              IsDecreasedBidLastLow(void)                           const;

   //--- Ask
   //--- setting the controlled Ask price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) Ask price change value,
   //--- getting the flag of the Ask price change exceeding the (5) increase, (6) decrease value
   void              SetControlAskInc(const double value)                        { this.SetControlledValueINC(SYMBOL_PROP_ASK,::fabs(value));                        }
   void              SetControlAskDec(const double value)                        { this.SetControlledValueDEC(SYMBOL_PROP_ASK,::fabs(value));                        }
   void              SetControlAskLevel(const double value)                      { this.SetControlledValueLEVEL(SYMBOL_PROP_ASK,::fabs(value));                      }
   double            GetValueChangedAsk(void)                              const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_ASK);                           }
   bool              IsIncreasedAsk(void)                                  const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_ASK);                          }
   bool              IsDecreasedAsk(void)                                  const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_ASK);                          }
   //--- Maximum Ask price for the day
   //--- setting the maximum day Ask controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the maximum Ask change value within a day,
   //--- getting the flag of the maximum day Ask change exceeding the (5) increase, (6) decrease value
   void              SetControlAskHighInc(const double value)                    { this.SetControlledValueINC(SYMBOL_PROP_ASKHIGH,::fabs(value));                    }
   void              SetControlAskHighDec(const double value)                    { this.SetControlledValueDEC(SYMBOL_PROP_ASKHIGH,::fabs(value));                    }
   void              SetControlAskHighLevel(const double value)                  { this.SetControlledValueLEVEL(SYMBOL_PROP_ASKHIGH,::fabs(value));                  }
   double            GetValueChangedAskHigh(void)                          const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_ASKHIGH);                       }
   bool              IsIncreasedAskHigh(void)                              const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_ASKHIGH);                      }
   bool              IsDecreasedAskHigh(void)                              const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_ASKHIGH);                      }
   //--- Minimum Ask price for the day
   //--- setting the minimum day Ask controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the minimum Ask change value within a day,
   //--- getting the flag of the minimum day Ask change exceeding the (5) increase, (6) decrease value
   void              SetControlAskLowInc(const double value)                     { this.SetControlledValueINC(SYMBOL_PROP_ASKLOW,::fabs(value));                     }
   void              SetControlAskLowDec(const double value)                     { this.SetControlledValueDEC(SYMBOL_PROP_ASKLOW,::fabs(value));                     }
   void              SetControlAskLowLevel(const double value)                   { this.SetControlledValueLEVEL(SYMBOL_PROP_ASKLOW,::fabs(value));                   }
   double            GetValueChangedAskLow(void)                           const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_ASKLOW);                        }
   bool              IsIncreasedAskLow(void)                               const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_ASKLOW);                       }
   bool              IsDecreasedAskLow(void)                               const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_ASKLOW);                       }
   //--- Real Volume for the day
   //--- setting the real day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the change value of the real day volume,
   //--- getting the flag of the real day volume change exceeding the (5) increase, (6) decrease value
   void              SetControlVolumeRealInc(const double value)                 { this.SetControlledValueINC(SYMBOL_PROP_VOLUME_REAL,::fabs(value));                }
   void              SetControlVolumeRealDec(const double value)                 { this.SetControlledValueDEC(SYMBOL_PROP_VOLUME_REAL,::fabs(value));                }
   void              SetControlVolumeRealLevel(const double value)               { this.SetControlledValueLEVEL(SYMBOL_PROP_VOLUME_REAL,::fabs(value));              }
   double            GetValueChangedVolumeReal(void)                       const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_VOLUME_REAL);                   }
   bool              IsIncreasedVolumeReal(void)                           const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_VOLUME_REAL);                  }
   bool              IsDecreasedVolumeReal(void)                           const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_VOLUME_REAL);                  }
   //--- Maximum real volume for the day
   //--- setting the maximum real day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the change value of the maximum real day volume,
   //--- getting the flag of the maximum real day volume change exceeding the (5) increase, (6) decrease value
   void              SetControlVolumeHighRealInc(const double value)             { this.SetControlledValueINC(SYMBOL_PROP_VOLUMEHIGH_REAL,::fabs(value));            }
   void              SetControlVolumeHighRealDec(const double value)             { this.SetControlledValueDEC(SYMBOL_PROP_VOLUMEHIGH_REAL,::fabs(value));            }
   void              SetControlVolumeHighRealLevel(const double value)           { this.SetControlledValueLEVEL(SYMBOL_PROP_VOLUMEHIGH_REAL,::fabs(value));          }
   double            GetValueChangedVolumeHighReal(void)                   const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_VOLUMEHIGH_REAL);               }
   bool              IsIncreasedVolumeHighReal(void)                       const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_VOLUMEHIGH_REAL);              }
   bool              IsDecreasedVolumeHighReal(void)                       const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_VOLUMEHIGH_REAL);              }
   //--- Minimum real volume for the day
   //--- setting the minimum real day volume controlled (1) increase, (2) decrease and (3) control level
   //--- getting (4) the change value of the minimum real day volume,
   //--- getting the flag of the minimum real day volume change exceeding the (5) increase, (6) decrease value
   void              SetControlVolumeLowRealInc(const double value)              { this.SetControlledValueINC(SYMBOL_PROP_VOLUMELOW_REAL,::fabs(value));             }
   void              SetControlVolumeLowRealDec(const double value)              { this.SetControlledValueDEC(SYMBOL_PROP_VOLUMELOW_REAL,::fabs(value));             }
   void              SetControlVolumeLowRealLevel(const double value)            { this.SetControlledValueLEVEL(SYMBOL_PROP_VOLUMELOW_REAL,::fabs(value));           }
   double            GetValueChangedVolumeLowReal(void)                    const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_VOLUMELOW_REAL);                }
   bool              IsIncreasedVolumeLowReal(void)                        const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_VOLUMELOW_REAL);               }
   bool              IsDecreasedVolumeLowReal(void)                        const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_VOLUMELOW_REAL);               }
   //--- Strike price
   //--- setting the controlled strike price (1) increase, (2) decrease value and (3) control level in points
   //--- getting (4) the change value of the strike price,
   //--- getting the flag of the strike price change exceeding the (5) increase, (6) decrease value
   void              SetControlOptionStrikeInc(const double value)               { this.SetControlledValueINC(SYMBOL_PROP_OPTION_STRIKE,::fabs(value));              }
   void              SetControlOptionStrikeDec(const double value)               { this.SetControlledValueDEC(SYMBOL_PROP_OPTION_STRIKE,::fabs(value));              }
   void              SetControlOptionStrikeLevel(const double value)             { this.SetControlledValueLEVEL(SYMBOL_PROP_OPTION_STRIKE,::fabs(value));            }
   double            GetValueChangedOptionStrike(void)                     const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_OPTION_STRIKE);                 }
   bool              IsIncreasedOptionStrike(void)                         const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_OPTION_STRIKE);                }
   bool              IsDecreasedOptionStrike(void)                         const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_OPTION_STRIKE);                }
   //--- Maximum allowed total volume of unidirectional positions and orders
   //--- (1) Setting the control level
   //--- (2) getting the change value of the maximum allowed total volume of unidirectional positions and orders,
   //--- getting the flag of (3) increasing, (4) decreasing the maximum allowed total volume of unidirectional positions and orders
   void              SetControlVolumeLimitLevel(const double value)              { this.SetControlledValueLEVEL(SYMBOL_PROP_VOLUME_LIMIT,::fabs(value));             }
   double            GetValueChangedVolumeLimit(void)                      const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_VOLUME_LIMIT);                  }
   bool              IsIncreasedVolumeLimit(void)                          const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_VOLUME_LIMIT);                 }
   bool              IsDecreasedVolumeLimit(void)                          const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_VOLUME_LIMIT);                 }
   //---  Swap long
   //--- (1) Setting the control level
   //--- (2) getting the swap long change value,
   //--- getting the flag of (3) increasing, (4) decreasing the swap long
   void              SetControlSwapLongLevel(const double value)                 { this.SetControlledValueLEVEL(SYMBOL_PROP_SWAP_LONG,::fabs(value));                }
   double            GetValueChangedSwapLong(void)                         const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SWAP_LONG);                     }
   bool              IsIncreasedSwapLong(void)                             const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SWAP_LONG);                    }
   bool              IsDecreasedSwapLong(void)                             const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SWAP_LONG);                    }
   //---  Swap short
   //--- (1) Setting the control level
   //--- (2) getting the swap short change value,
   //--- getting the flag of (3) increasing, (4) decreasing the swap short
   void              SetControlSwapShortLevel(const double value)                { this.SetControlledValueLEVEL(SYMBOL_PROP_SWAP_SHORT,::fabs(value));               }
   double            GetValueChangedSwapShort(void)                        const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SWAP_SHORT);                    }
   bool              IsIncreasedSwapShort(void)                            const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SWAP_SHORT);                   }
   bool              IsDecreasedSwapShort(void)                            const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SWAP_SHORT);                   }
   //--- The total volume of deals in the current session
   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the total volume of deals during the current session
   //--- getting (4) the total deal volume change value in the current session,
   //--- getting the flag of the total deal volume change during the current session exceeding the (5) increase, (6) decrease value
   void              SetControlSessionVolumeInc(const double value)              { this.SetControlledValueINC(SYMBOL_PROP_SESSION_VOLUME,::fabs(value));             }
   void              SetControlSessionVolumeDec(const double value)              { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_VOLUME,::fabs(value));             }
   void              SetControlSessionVolumeLevel(const double value)            { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_VOLUME,::fabs(value));           }
   double            GetValueChangedSessionVolume(void)                    const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_VOLUME);                }
   bool              IsIncreasedSessionVolume(void)                        const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_VOLUME);               }
   bool              IsDecreasedSessionVolume(void)                        const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_VOLUME);               }
   //--- The total turnover in the current session
   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the total turnover during the current session
   //--- getting (4) the total turnover change value in the current session,
   //--- getting the flag of the total turnover change during the current session exceeding the (5) increase, (6) decrease value
   void              SetControlSessionTurnoverInc(const double value)            { this.SetControlledValueINC(SYMBOL_PROP_SESSION_TURNOVER,::fabs(value));           }
   void              SetControlSessionTurnoverDec(const double value)            { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_TURNOVER,::fabs(value));           }
   void              SetControlSessionTurnoverLevel(const double value)          { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_TURNOVER,::fabs(value));         }
   double            GetValueChangedSessionTurnover(void)                  const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_TURNOVER);              }
   bool              IsIncreasedSessionTurnover(void)                      const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_TURNOVER);             }
   bool              IsDecreasedSessionTurnover(void)                      const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_TURNOVER);             }
   //--- The total volume of open positions
   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the total volume of open positions during the current session
   //--- getting (4) the change value of the open positions total volume in the current session,
   //--- getting the flag of the open positions total volume change during the current session exceeding the (5) increase, (6) decrease value
   void              SetControlSessionInterestInc(const double value)            { this.SetControlledValueINC(SYMBOL_PROP_SESSION_INTEREST,::fabs(value));           }
   void              SetControlSessionInterestDec(const double value)            { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_INTEREST,::fabs(value));           }
   void              SetControlSessionInterestLevel(const double value)          { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_INTEREST,::fabs(value));         }
   double            GetValueChangedSessionInterest(void)                  const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_INTEREST);              }
   bool              IsIncreasedSessionInterest(void)                      const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_INTEREST);             }
   bool              IsDecreasedSessionInterest(void)                      const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_INTEREST);             }
   //--- The total volume of Buy orders at the moment
   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the current total buy order volume
   //--- getting (4) the change value of the current total buy order volume,
   //--- getting the flag of the current total buy orders' volume change exceeding the (5) increase, (6) decrease value
   void              SetControlSessionBuyOrdVolumeInc(const double value)        { this.SetControlledValueINC(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME,::fabs(value));  }
   void              SetControlSessionBuyOrdVolumeDec(const double value)        { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME,::fabs(value));  }
   void              SetControlSessionBuyOrdVolumeLevel(const double value)      { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME,::fabs(value));}
   double            GetValueChangedSessionBuyOrdVolume(void)              const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);     }
   bool              IsIncreasedSessionBuyOrdVolume(void)                  const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);    }
   bool              IsDecreasedSessionBuyOrdVolume(void)                  const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);    }
   //--- The total volume of Sell orders at the moment
   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the current total sell order volume
   //--- getting (4) the change value of the current total sell order volume,
   //--- getting the flag of the current total sell orders' volume change exceeding the (5) increase, (6) decrease value
   void              SetControlSessionSellOrdVolumeInc(const double value)       { this.SetControlledValueINC(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME,::fabs(value)); }
   void              SetControlSessionSellOrdVolumeDec(const double value)       { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME,::fabs(value)); }
   void              SetControlSessionSellOrdVolumeLevel(const double value)     { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME,::fabs(value));}
   double            GetValueChangedSessionSellOrdVolume(void)             const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);    }
   bool              IsIncreasedSessionSellOrdVolume(void)                 const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);   }
   bool              IsDecreasedSessionSellOrdVolume(void)                 const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);   }
   //--- Session open price
   //--- setting the controlled session open price (1) increase, (2) decrease and (3) control value
   //--- getting (4) the change value of the session open price,
   //--- getting the flag of the session open price change exceeding the (5) increase, (6) decrease value
   void              SetControlSessionPriceOpenInc(const double value)           { this.SetControlledValueINC(SYMBOL_PROP_SESSION_OPEN,::fabs(value));               }
   void              SetControlSessionPriceOpenDec(const double value)           { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_OPEN,::fabs(value));               }
   void              SetControlSessionPriceOpenLevel(const double value)         { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_OPEN,::fabs(value));             }
   double            GetValueChangedSessionPriceOpen(void)                 const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_OPEN);                  }
   bool              IsIncreasedSessionPriceOpen(void)                     const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_OPEN);                 }
   bool              IsDecreasedSessionPriceOpen(void)                     const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_OPEN);                 }
   //--- Session close price
   //--- setting the controlled session close price (1) increase, (2) decrease and (3) control value
   //--- getting (4) the change value of the session close price,
   //--- getting the flag of the session close price change exceeding the (5) increase, (6) decrease value
   void              SetControlSessionPriceCloseInc(const double value)          { this.SetControlledValueINC(SYMBOL_PROP_SESSION_CLOSE,::fabs(value));              }
   void              SetControlSessionPriceCloseDec(const double value)          { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_CLOSE,::fabs(value));              }
   void              SetControlSessionPriceCloseLevel(const double value)        { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_CLOSE,::fabs(value));            }
   double            GetValueChangedSessionPriceClose(void)                const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_CLOSE);                 }
   bool              IsIncreasedSessionPriceClose(void)                    const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_CLOSE);                }
   bool              IsDecreasedSessionPriceClose(void)                    const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_CLOSE);                }
   //--- The average weighted session price
   //--- setting the controlled session average weighted price (1) increase, (2) decrease and (3) control value
   //--- getting (4) the change value of the average weighted session price,
   //--- getting the flag of the average weighted session price change exceeding the (5) increase, (6) decrease value
   void              SetControlSessionPriceAWInc(const double value)             { this.SetControlledValueINC(SYMBOL_PROP_SESSION_AW,::fabs(value));                 }
   void              SetControlSessionPriceAWDec(const double value)             { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_AW,::fabs(value));                 }
   void              SetControlSessionPriceAWLevel(const double value)           { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_AW,::fabs(value));               }
   double            GetValueChangedSessionPriceAW(void)                   const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_AW);                    }
   bool              IsIncreasedSessionPriceAW(void)                       const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_AW);                   }
   bool              IsDecreasedSessionPriceAW(void)                       const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_AW);                   }
//---
```

The **methods implemented beyond the class body now also use base class methods directly:**

```
//+------------------------------------------------------------------+
//| Set the Bid or Last price controlled increase                    |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastInc(const double value)
  {
   this.SetControlledValueINC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BID : SYMBOL_PROP_LAST),::fabs(value));
  }
//+------------------------------------------------------------------+
//|Set the Bid or Last price controlled decrease                     |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastDec(const double value)
  {
   this.SetControlledValueDEC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BID : SYMBOL_PROP_LAST),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the Bid or Last price control level                          |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastLevel(const double value)
  {
   this.SetControlledValueLEVEL((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BID : SYMBOL_PROP_LAST),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Return the Bid or Last price change value                        |
//+------------------------------------------------------------------+
double CSymbol::GetValueChangedBidLast(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetPropDoubleChangedValue(SYMBOL_PROP_BID) : this.GetPropDoubleChangedValue(SYMBOL_PROP_LAST));
  }
//+------------------------------------------------------------------+
//| Return the flag of the Bid or Last price change                  |
//| exceeding the increase value                                     |
//+------------------------------------------------------------------+
bool CSymbol::IsIncreasedBidLast(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_BID) : (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_LAST));
  }
//+------------------------------------------------------------------+
//| Return the flag of the Bid or Last price change                  |
//| exceeding the decrease value                                     |
//+------------------------------------------------------------------+
bool CSymbol::IsDecreasedBidLast(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_BID) : (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_LAST));
  }
//+------------------------------------------------------------------+
//| Set the controlled increase value                                |
//| of the maximum Bid or Last price                                 |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastHighInc(const double value)
  {
   this.SetControlledValueINC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDHIGH : SYMBOL_PROP_LASTHIGH),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the controlled decrease value                                |
//| of the maximum Bid or Last price                                 |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastHighDec(const double value)
  {
   this.SetControlledValueDEC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDHIGH : SYMBOL_PROP_LASTHIGH),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the maximum Bid or Last price control level                  |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastHighLevel(const double value)
  {
   this.SetControlledValueLEVEL((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDHIGH : SYMBOL_PROP_LASTHIGH),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Return the maximum Bid or Last price change value                |
//+------------------------------------------------------------------+
double CSymbol::GetValueChangedBidLastHigh(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetPropDoubleChangedValue(SYMBOL_PROP_BIDHIGH) : this.GetPropDoubleChangedValue(SYMBOL_PROP_LASTHIGH));
  }
//+------------------------------------------------------------------+
//| Return the flag of a change of the maximum                       |
//| Bid or Last price exceeding the increase value                   |
//+------------------------------------------------------------------+
bool CSymbol::IsIncreasedBidLastHigh(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_BIDHIGH) : (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_LASTHIGH));
  }
//+------------------------------------------------------------------+
//| Return the flag of a change of the maximum                       |
//| Bid or Last price exceeding the decrease value                   |
//+------------------------------------------------------------------+
bool CSymbol::IsDecreasedBidLastHigh(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_BIDHIGH) : (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_LASTHIGH));
  }
//+------------------------------------------------------------------+
//| Set the controlled increase value                                |
//| of the minimum Bid or Last price                                 |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastLowInc(const double value)
  {
   this.SetControlledValueINC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDLOW : SYMBOL_PROP_LASTLOW),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the controlled decrease value                                |
//| of the minimum Bid or Last price                                 |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastLowDec(const double value)
  {
   this.SetControlledValueDEC((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDLOW : SYMBOL_PROP_LASTLOW),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Set the minimum Bid or Last price control level                  |
//+------------------------------------------------------------------+
void CSymbol::SetControlBidLastLowLevev(const double value)
  {
   this.SetControlledValueLEVEL((this.ChartMode()==SYMBOL_CHART_MODE_BID ? SYMBOL_PROP_BIDLOW : SYMBOL_PROP_LASTLOW),::fabs(value));
  }
//+------------------------------------------------------------------+
//| Return the minimum Bid or Last price change value                |
//+------------------------------------------------------------------+
double CSymbol::GetValueChangedBidLastLow(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetPropDoubleChangedValue(SYMBOL_PROP_BIDLOW) : this.GetPropDoubleChangedValue(SYMBOL_PROP_LASTLOW));
  }
//+------------------------------------------------------------------+
//| Return the flag of a change of the minimum                       |
//| Bid or Last price exceeding the increase value                   |
//+------------------------------------------------------------------+
bool CSymbol::IsIncreasedBidLastLow(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_BIDLOW) : (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_LASTLOW));
  }
//+------------------------------------------------------------------+
//| Return the flag of a change of the minimum                       |
//| Bid or Last price exceeding the decrease value                   |
//+------------------------------------------------------------------+
bool CSymbol::IsDecreasedBidLastLow(void) const
  {
   return(this.ChartMode()==SYMBOL_CHART_MODE_BID ? (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_BIDLOW) : (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_LASTLOW));
  }
//+------------------------------------------------------------------+
```

Besides, I made a logical error in the class constructor. The base object data was not filled in immediately after creating a symbol object
preventing events from being tracked during the fist launch. The filling in started only after a symbol property change event occurred with
the change value between two neighboring ticks being insignificant enough.

Let's fix this — **immediately after filling in the symbol object properties, add filling**
**in its base object properties:**

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
//--- Initializing base object data arrays
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

//--- Fill in the symbol current data
   for(int i=0;i<SYMBOL_PROP_INTEGER_TOTAL;i++)
      this.m_long_prop_event[i][3]=this.m_long_prop[i];
   for(int i=0;i<SYMBOL_PROP_DOUBLE_TOTAL;i++)
      this.m_double_prop_event[i][3]=this.m_double_prop[i];

//--- Update the base object data and search for changes
   CBaseObj::Refresh();
//---
   if(!select)
      this.RemoveFromMarketWatch();
  }
//+------------------------------------------------------------------+
```

In the public section of the symbol collection class in \\MQL5\\Include\\DoEasy\\Collections\ **SymbolsCollection.mqh**,
also change the name of the method for updating collection symbols and searching for SymbolsEventsControl() events.

**Make the method name more suitable for its tasks** **— updating data and searching for events:**

```
//--- Working with the events of the (1) collection symbol list, (2) market watch window
   void              RefreshAndEventsControl(void);
   void              MarketWatchEventsControl(const bool send_events=true);
```

**These are all the changes to be implemented in the classes of the base object, its descendant** **(symbol object) and**
**the symbol collection class.**

Now let's start improving account object classes to make it a descendant of the CBaseObj base object as well and let it get the event
functionality for easy tracking of the account object property changes.

### Revamping the account object

As you may remember from the previous article, we no longer need to create event flags and create event IDs from the flag combinations when
using the base object as the event generation source. Now the event functionality of the base object is more flexible. This means we need to

**remove unnecessary enumerations from the library's Defines.mqh**
**file:**

```
//+------------------------------------------------------------------+
//| Data for working with accounts                                   |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of account event flags                                      |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_EVENT_FLAGS
  {
   ACCOUNT_EVENT_FLAG_NO_EVENT            =  0,             // No event
   ACCOUNT_EVENT_FLAG_LEVERAGE            =  1,             // Changing the leverage
   ACCOUNT_EVENT_FLAG_LIMIT_ORDERS        =  2,             // Changing permission for auto trading for the account
   ACCOUNT_EVENT_FLAG_TRADE_ALLOWED       =  4,             // Changing permission to trade for the account
   ACCOUNT_EVENT_FLAG_TRADE_EXPERT        =  8,             // Changing permission for auto trading for the account
   ACCOUNT_EVENT_FLAG_BALANCE             =  16,            // The balance exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_EQUITY              =  32,            // The equity exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_PROFIT              =  64,            // The profit exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_CREDIT              =  128,           // Change a credit in a deposit currency
   ACCOUNT_EVENT_FLAG_MARGIN              =  256,           // The reserved margin on an account in the deposit currency change exceeds the specified value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_FREE         =  512,           // The free funds available for opening a position in a deposit currency exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_LEVEL        =  1024,          // The margin level on an account in % exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_INITIAL      =  2048,          // The funds reserved on an account to ensure a guarantee amount for all pending orders exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_MAINTENANCE  =  4096,          // The funds reserved on an account to ensure a minimum amount for all open positions exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_SO_CALL      =  8192,          // Changing the Margin Call level
   ACCOUNT_EVENT_FLAG_MARGIN_SO_SO        =  16384,         // Changing the Stop Out level
   ACCOUNT_EVENT_FLAG_ASSETS              =  32768,         // The current assets on an account exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_LIABILITIES         =  65536,         // The current liabilities on an account exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_COMISSION_BLOCKED   =  131072,        // The current sum of blocked commissions on an account exceeds the specified change value +/-
  };
//+------------------------------------------------------------------+
//| List of possible account events                                  |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_EVENT
  {
   ACCOUNT_EVENT_NO_EVENT = TRADE_EVENTS_NEXT_CODE,         // No event
   ACCOUNT_EVENT_LEVERAGE_INC,                              // Increasing the leverage
   ACCOUNT_EVENT_LEVERAGE_DEC,                              // Decreasing the leverage
   ACCOUNT_EVENT_LIMIT_ORDERS_INC,                          // Increasing the maximum allowed number of active pending orders
   ACCOUNT_EVENT_LIMIT_ORDERS_DEC,                          // Decreasing the maximum allowed number of active pending orders
   ACCOUNT_EVENT_TRADE_ALLOWED_ON,                          // Enabling trading for the account
   ACCOUNT_EVENT_TRADE_ALLOWED_OFF,                         // Disabling trading for the account
   ACCOUNT_EVENT_TRADE_EXPERT_ON,                           // Enabling auto trading for the account
   ACCOUNT_EVENT_TRADE_EXPERT_OFF,                          // Disabling auto trading for the account
   ACCOUNT_EVENT_BALANCE_INC,                               // The balance exceeds the specified value
   ACCOUNT_EVENT_BALANCE_DEC,                               // The balance falls below the specified value
   ACCOUNT_EVENT_EQUITY_INC,                                // The equity exceeds the specified value
   ACCOUNT_EVENT_EQUITY_DEC,                                // The equity falls below the specified value
   ACCOUNT_EVENT_PROFIT_INC,                                // The profit exceeds the specified value
   ACCOUNT_EVENT_PROFIT_DEC,                                // The profit falls below the specified value
   ACCOUNT_EVENT_CREDIT_INC,                                // The credit exceeds the specified value in a deposit currency
   ACCOUNT_EVENT_CREDIT_DEC,                                // The credit falls below the specified value in a deposit currency
   ACCOUNT_EVENT_MARGIN_INC,                                // Increasing the reserved margin on an account in a deposit currency
   ACCOUNT_EVENT_MARGIN_DEC,                                // Decreasing the reserved margin on an account in a deposit currency
   ACCOUNT_EVENT_MARGIN_FREE_INC,                           // Increasing the free funds available for opening a position in a deposit currency
   ACCOUNT_EVENT_MARGIN_FREE_DEC,                           // Decreasing the free funds available for opening a position in a deposit currency
   ACCOUNT_EVENT_MARGIN_LEVEL_INC,                          // Increasing the margin level on an account in %
   ACCOUNT_EVENT_MARGIN_LEVEL_DEC,                          // Decreasing the margin level on an account in %
   ACCOUNT_EVENT_MARGIN_INITIAL_INC,                        // Increasing the funds reserved on an account to ensure a guarantee amount for all pending orders
   ACCOUNT_EVENT_MARGIN_INITIAL_DEC,                        // Decreasing the funds reserved on an account to ensure a guarantee amount for all pending orders
   ACCOUNT_EVENT_MARGIN_MAINTENANCE_INC,                    // Increasing the funds reserved on an account to ensure a minimum amount for all open positions
   ACCOUNT_EVENT_MARGIN_MAINTENANCE_DEC,                    // Decreasing the funds reserved on an account to ensure a minimum amount for all open positions
   ACCOUNT_EVENT_MARGIN_SO_CALL_INC,                        // Increasing the Margin Call level
   ACCOUNT_EVENT_MARGIN_SO_CALL_DEC,                        // Decreasing the Margin Call level
   ACCOUNT_EVENT_MARGIN_SO_SO_INC,                          // Increasing the Stop Out level
   ACCOUNT_EVENT_MARGIN_SO_SO_DEC,                          // Decreasing the Stop Out level
   ACCOUNT_EVENT_ASSETS_INC,                                // Increasing the current asset size on the account
   ACCOUNT_EVENT_ASSETS_DEC,                                // Decreasing the current asset size on the account
   ACCOUNT_EVENT_LIABILITIES_INC,                           // Increasing the current liabilities on the account
   ACCOUNT_EVENT_LIABILITIES_DEC,                           // Decreasing the current liabilities on the account
   ACCOUNT_EVENT_COMISSION_BLOCKED_INC,                     // Increasing the current sum of blocked commissions on an account
   ACCOUNT_EVENT_COMISSION_BLOCKED_DEC,                     // Decreasing the current sum of blocked commissions on an account
  };
```

All that remains from these enumerations is a **macro substitution**
**indicating the next event code:**

```
//+------------------------------------------------------------------+
//| Data for working with accounts                                   |
//+------------------------------------------------------------------+
#define ACCOUNT_EVENTS_NEXT_CODE       (TRADE_EVENTS_NEXT_CODE)   // The code of the next event after the last account event code
//+------------------------------------------------------------------+
//| Account integer properties                                       |
//+------------------------------------------------------------------+
```

Open \\MQL5\\Include\\DoEasy\\Objects\\Accounts\ **Account.mqh** file and make the necessary changes.

**Declare the Refresh() virtual method** in the public
class section:

```
public:
//--- Constructor
                     CAccount(void);
//--- Set account's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_ACCOUNT_PROP_INTEGER property,long value)        { this.m_long_prop[property]=value;                                  }
   void              SetProperty(ENUM_ACCOUNT_PROP_DOUBLE property,double value)       { this.m_double_prop[this.IndexProp(property)]=value;                }
   void              SetProperty(ENUM_ACCOUNT_PROP_STRING property,string value)       { this.m_string_prop[this.IndexProp(property)]=value;                }
//--- Return (1) integer, (2) real and (3) string order properties from the account string property
   long              GetProperty(ENUM_ACCOUNT_PROP_INTEGER property)             const { return this.m_long_prop[property];                                 }
   double            GetProperty(ENUM_ACCOUNT_PROP_DOUBLE property)              const { return this.m_double_prop[this.IndexProp(property)];               }
   string            GetProperty(ENUM_ACCOUNT_PROP_STRING property)              const { return this.m_string_prop[this.IndexProp(property)];               }

//--- Return the flag of calculating MarginCall and StopOut levels in %
   bool              IsPercentsForSOLevels(void)                                 const { return this.MarginSOMode()==ACCOUNT_STOPOUT_MODE_PERCENT;          }
//--- Return the flag of supporting the property by the account object
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_INTEGER property)               { return true; }
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_DOUBLE property)                { return true; }
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_STRING property)                { return true; }

//--- Compare CAccount objects by all possible properties (for sorting the lists by a specified account object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CAccount objects by account properties (to search for equal account objects)
   bool              IsEqual(CAccount* compared_account) const;
//--- Update all account data
   virtual void      Refresh(void);
//--- (1) Save the account object to the file, (2), download the account object from the file
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
```

To update the current account data, use the Refresh() method the same way as in the CSymbol class, as well as in all subsequent classes based on
the CBaseObj base object. Previously, we updated the current account data from the account collection class. But in order for all classes to
have the same structure, we will do everything the same way as we did in CSymbol, as well as the same way as we are going to do in other future
classes.

We have already created the methods in the CSymbol class to obtain and set the parameters of tracked symbol properties.

**Let's create the**
**same methods for the account object class as well:**

```
//+------------------------------------------------------------------+
//| Get and set the parameters of tracked property changes           |
//+------------------------------------------------------------------+
   //--- setting the controlled leverage (1) increase, (2) decrease value and (3) control level
   //--- getting the (3) leverage change value,
   //--- getting the flag of the leverage change exceeding the (4) increase, (5) decrease value
   void              SetControlLeverageInc(const long value)               { this.SetControlledValueINC(ACCOUNT_PROP_LEVERAGE,(long)::fabs(value));               }
   void              SetControlLeverageDec(const long value)               { this.SetControlledValueDEC(ACCOUNT_PROP_LEVERAGE,(long)::fabs(value));               }
   void              SetControlLeverageLevel(const long value)             { this.SetControlledValueLEVEL(ACCOUNT_PROP_LEVERAGE,(long)::fabs(value));             }
   long              GetValueChangedLeverage(void)                   const { return this.GetPropLongChangedValue(ACCOUNT_PROP_LEVERAGE);                          }
   bool              IsIncreasedLeverage(void)                       const { return (bool)this.GetPropLongFlagINC(ACCOUNT_PROP_LEVERAGE);                         }
   bool              IsDecreasedLeverage(void)                       const { return (bool)this.GetPropLongFlagDEC(ACCOUNT_PROP_LEVERAGE);                         }

   //--- setting the controlled value of (1) growth, (2) decrease and (3) control level of the number of active pending orders
   //--- getting (3) the change value of the number of active pending orders,
   //--- getting the flag of the number of active pending orders exceeding the (4) increase, (5) decrease value
   void              SetControlLimitOrdersInc(const long value)            { this.SetControlledValueINC(ACCOUNT_PROP_LIMIT_ORDERS,(long)::fabs(value));           }
   void              SetControlLimitOrdersDec(const long value)            { this.SetControlledValueDEC(ACCOUNT_PROP_LIMIT_ORDERS,(long)::fabs(value));           }
   void              SetControlLimitOrdersLevel(const long value)          { this.SetControlledValueLEVEL(ACCOUNT_PROP_LIMIT_ORDERS,(long)::fabs(value));         }
   long              GetValueChangedLimitOrders(void)                const { return this.GetPropLongChangedValue(ACCOUNT_PROP_LIMIT_ORDERS);                      }
   bool              IsIncreasedLimitOrders(void)                    const { return (bool)this.GetPropLongFlagINC(ACCOUNT_PROP_LIMIT_ORDERS);                     }
   bool              IsDecreasedLimitOrders(void)                    const { return (bool)this.GetPropLongFlagDEC(ACCOUNT_PROP_LIMIT_ORDERS);                     }

   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the permission to trade for the current account from the server side
   //--- getting (3) the value of the permission to trade for the current account from the server side,
   //--- getting the flag of a change of the permission to trade for the current account from the server side exceeding the (4) increase and (5) decrease values
   void              SetControlTradeAllowedInc(const long value)           { this.SetControlledValueINC(ACCOUNT_PROP_TRADE_ALLOWED,(long)::fabs(value));          }
   void              SetControlTradeAllowedDec(const long value)           { this.SetControlledValueDEC(ACCOUNT_PROP_TRADE_ALLOWED,(long)::fabs(value));          }
   void              SetControlTradeAllowedLevel(const long value)         { this.SetControlledValueLEVEL(ACCOUNT_PROP_TRADE_ALLOWED,(long)::fabs(value));        }
   long              GetValueChangedTradeAllowed(void)               const { return this.GetPropLongChangedValue(ACCOUNT_PROP_TRADE_ALLOWED);                     }
   bool              IsIncreasedTradeAllowed(void)                   const { return (bool)this.GetPropLongFlagINC(ACCOUNT_PROP_TRADE_ALLOWED);                    }
   bool              IsDecreasedTradeAllowed(void)                   const { return (bool)this.GetPropLongFlagDEC(ACCOUNT_PROP_TRADE_ALLOWED);                    }

   //--- setting the controlled value of (1) increase, (2) decrease and (3) control level of the permission to trade for an EA from the server side
   //--- getting (3) the value of the permission to trade for an EA from the server side,
   //--- getting the flag of a change of the permission to trade for an EA from the server side exceeding the (4) increase and (5) decrease values
   void              SetControlTradeExpertInc(const long value)            { this.SetControlledValueINC(ACCOUNT_PROP_TRADE_EXPERT,(long)::fabs(value));           }
   void              SetControlTradeExpertDec(const long value)            { this.SetControlledValueDEC(ACCOUNT_PROP_TRADE_EXPERT,(long)::fabs(value));           }
   void              SetControlTradeExpertLevel(const long value)          { this.SetControlledValueLEVEL(ACCOUNT_PROP_TRADE_EXPERT,(long)::fabs(value));         }
   long              GetValueChangedTradeExpert(void)                const { return this.GetPropLongChangedValue(ACCOUNT_PROP_TRADE_EXPERT);                      }
   bool              IsIncreasedTradeExpert(void)                    const { return (bool)this.GetPropLongFlagINC(ACCOUNT_PROP_TRADE_EXPERT);                     }
   bool              IsDecreasedTradeExpert(void)                    const { return (bool)this.GetPropLongFlagDEC(ACCOUNT_PROP_TRADE_EXPERT);                     }

   //--- setting the controlled balance (1) increase, (2) decrease value and (3) control level
   //--- getting (3) the balance change value,
   //--- getting the flag of the balance change exceeding the (4) increase value, (5) decrease value
   void              SetControlBalanceInc(const long value)                { this.SetControlledValueINC(ACCOUNT_PROP_BALANCE,(double)::fabs(value));              }
   void              SetControlBalanceDec(const long value)                { this.SetControlledValueDEC(ACCOUNT_PROP_BALANCE,(double)::fabs(value));              }
   void              SetControlBalanceLevel(const long value)              { this.SetControlledValueLEVEL(ACCOUNT_PROP_BALANCE,(double)::fabs(value));            }
   double            GetValueChangedBalance(void)                    const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_BALANCE);                         }
   bool              IsIncreasedBalance(void)                        const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_BALANCE);                        }
   bool              IsDecreasedBalance(void)                        const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_BALANCE);                        }

   //--- setting the controlled credit (1) increase, (2) decrease value and (3) control level
   //--- getting the (3) credit change value,
   //--- getting the flag of the credit change exceeding the (4) increase, (5) decrease value
   void              SetControlCreditInc(const long value)                 { this.SetControlledValueINC(ACCOUNT_PROP_CREDIT,(double)::fabs(value));               }
   void              SetControlCreditDec(const long value)                 { this.SetControlledValueDEC(ACCOUNT_PROP_CREDIT,(double)::fabs(value));               }
   void              SetControlCreditLevel(const long value)               { this.SetControlledValueLEVEL(ACCOUNT_PROP_CREDIT,(double)::fabs(value));             }
   double            GetValueChangedCredit(void)                     const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_CREDIT);                          }
   bool              IsIncreasedCredit(void)                         const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_CREDIT);                         }
   bool              IsDecreasedCredit(void)                         const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_CREDIT);                         }

   //--- setting the controlled profit (1) increase, (2) decrease value and (3) control level
   //--- getting the (3) profit change value,
   //--- getting the flag of the profit change exceeding the (4) increase, (5) decrease value
   void              SetControlProfitInc(const long value)                 { this.SetControlledValueINC(ACCOUNT_PROP_PROFIT,(double)::fabs(value));               }
   void              SetControlProfitDec(const long value)                 { this.SetControlledValueDEC(ACCOUNT_PROP_PROFIT,(double)::fabs(value));               }
   void              SetControlProfitLevel(const long value)               { this.SetControlledValueLEVEL(ACCOUNT_PROP_PROFIT,(double)::fabs(value));             }
   double            GetValueChangedProfit(void)                     const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_PROFIT);                          }
   bool              IsIncreasedProfit(void)                         const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_PROFIT);                         }
   bool              IsDecreasedProfit(void)                         const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_PROFIT);                         }

   //--- setting the controlled equity (1) increase, (2) decrease value and (3) control level
   //--- getting the (3) equity change value,
   //--- getting the flag of the equity change exceeding the (4) increase, (5) decrease value
   void              SetControlEquityInc(const long value)                 { this.SetControlledValueINC(ACCOUNT_PROP_EQUITY,(double)::fabs(value));               }
   void              SetControlEquityDec(const long value)                 { this.SetControlledValueDEC(ACCOUNT_PROP_EQUITY,(double)::fabs(value));               }
   void              SetControlEquityLevel(const long value)               { this.SetControlledValueLEVEL(ACCOUNT_PROP_EQUITY,(double)::fabs(value));             }
   double            GetValueChangedEquity(void)                     const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_EQUITY);                          }
   bool              IsIncreasedEquity(void)                         const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_EQUITY);                         }
   bool              IsDecreasedEquity(void)                         const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_EQUITY);                         }

   //--- setting the controlled margin (1) increase, (2) decrease value and (3) control level
   //--- getting the (3) margin change value,
   //--- getting the flag of the margin change exceeding the (4) increase, (5) decrease value
   void              SetControlMarginInc(const long value)                 { this.SetControlledValueINC(ACCOUNT_PROP_MARGIN,(double)::fabs(value));               }
   void              SetControlMarginDec(const long value)                 { this.SetControlledValueDEC(ACCOUNT_PROP_MARGIN,(double)::fabs(value));               }
   void              SetControlMarginLevel(const long value)               { this.SetControlledValueLEVEL(ACCOUNT_PROP_MARGIN,(double)::fabs(value));             }
   double            GetValueChangedMargin(void)                     const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_MARGIN);                          }
   bool              IsIncreasedMargin(void)                         const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_MARGIN);                         }
   bool              IsDecreasedMargin(void)                         const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_MARGIN);                         }

   //--- setting the controlled free margin (1) increase, (2) decrease value and (3) control level
   //--- getting the (3) free margin change value,
   //--- getting the flag of the free margin change exceeding the (4) increase, (5) decrease value
   void              SetControlMarginFreeInc(const long value)             { this.SetControlledValueINC(ACCOUNT_PROP_MARGIN_FREE,(double)::fabs(value));          }
   void              SetControlMarginFreeDec(const long value)             { this.SetControlledValueDEC(ACCOUNT_PROP_MARGIN_FREE,(double)::fabs(value));          }
   void              SetControlMarginFreeLevel(const long value)           { this.SetControlledValueLEVEL(ACCOUNT_PROP_MARGIN_FREE,(double)::fabs(value));        }
   double            GetValueChangedMarginFree(void)                 const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_MARGIN_FREE);                     }
   bool              IsIncreasedMarginFree(void)                     const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_MARGIN_FREE);                    }
   bool              IsDecreasedMarginFree(void)                     const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_MARGIN_FREE);                    }

   //--- setting the controlled margin level (1) increase, (2) decrease value and (3) control level
   //--- getting the (3) margin level change value,
   //--- getting the flag of the margin level change exceeding the (4) increase, (5) decrease value
   void              SetControlMarginLevelInc(const long value)            { this.SetControlledValueINC(ACCOUNT_PROP_MARGIN_LEVEL,(double)::fabs(value));         }
   void              SetControlMarginLevelDec(const long value)            { this.SetControlledValueDEC(ACCOUNT_PROP_MARGIN_LEVEL,(double)::fabs(value));         }
   void              SetControlMarginLevelLevel(const long value)          { this.SetControlledValueLEVEL(ACCOUNT_PROP_MARGIN_LEVEL,(double)::fabs(value));       }
   double            GetValueChangedMarginLevel(void)                const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_MARGIN_LEVEL);                    }
   bool              IsIncreasedMarginLevel(void)                    const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_MARGIN_LEVEL);                   }
   bool              IsDecreasedMarginLevel(void)                    const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_MARGIN_LEVEL);                   }

   //--- setting the controlled Margin Call (1) increase, (2) decrease value and (3) control level in points
   //--- getting (3) Margin Call level change value,
   //--- getting the flag of the Margin Call level change exceeding the (4) increase, (5) decrease value
   void              SetControlMarginCallInc(const long value)             { this.SetControlledValueINC(ACCOUNT_PROP_MARGIN_SO_CALL,(double)::fabs(value));       }
   void              SetControlMarginCallDec(const long value)             { this.SetControlledValueDEC(ACCOUNT_PROP_MARGIN_SO_CALL,(double)::fabs(value));       }
   void              SetControlMarginCallLevel(const long value)           { this.SetControlledValueLEVEL(ACCOUNT_PROP_MARGIN_SO_CALL,(double)::fabs(value));     }
   double            GetValueChangedMarginCall(void)                 const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_MARGIN_SO_CALL);                  }
   bool              IsIncreasedMarginCall(void)                     const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_MARGIN_SO_CALL);                 }
   bool              IsDecreasedMarginCall(void)                     const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_MARGIN_SO_CALL);                 }

   //--- setting the controlled Margin StopOut (1) increase, (2) decrease value and (3) control level
   //--- getting (3) Margin StopOut level change value,
   //--- getting the flag of the Margin StopOut level change exceeding the (4) increase, (5) decrease value
   void              SetControlMarginStopOutInc(const long value)          { this.SetControlledValueINC(ACCOUNT_PROP_MARGIN_SO_SO,(double)::fabs(value));         }
   void              SetControlMarginStopOutDec(const long value)          { this.SetControlledValueDEC(ACCOUNT_PROP_MARGIN_SO_SO,(double)::fabs(value));         }
   void              SetControlMarginStopOutLevel(const long value)        { this.SetControlledValueLEVEL(ACCOUNT_PROP_MARGIN_SO_SO,(double)::fabs(value));       }
   double            GetValueChangedMarginStopOut(void)              const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_MARGIN_SO_SO);                    }
   bool              IsIncreasedMarginStopOut(void)                  const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_MARGIN_SO_SO);                   }
   bool              IsDecreasedMarginStopOut(void)                  const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_MARGIN_SO_SO);                   }

   //--- setting the controlled value of the growth of the reserved funds for providing the guarantee sum for pending orders (1) increase, (2) decrease
   //--- getting (3) the change value of the growth of the reserved funds for providing the guarantee sum for pending orders,
   //--- getting the flag of the growth change of the reserved funds for providing the guarantee sum for pending orders exceeding the (4) increase, (5) decrease value
   void              SetControlMarginInitialInc(const long value)          { this.SetControlledValueINC(ACCOUNT_PROP_MARGIN_INITIAL,(double)::fabs(value));       }
   void              SetControlMarginInitialDec(const long value)          { this.SetControlledValueDEC(ACCOUNT_PROP_MARGIN_INITIAL,(double)::fabs(value));       }
   void              SetControlMarginInitialLevel(const long value)        { this.SetControlledValueLEVEL(ACCOUNT_PROP_MARGIN_INITIAL,(double)::fabs(value));     }
   double            GetValueChangedMarginInitial(void)              const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_MARGIN_INITIAL);                  }
   bool              IsIncreasedMarginInitial(void)                  const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_MARGIN_INITIAL);                 }
   bool              IsDecreasedMarginInitial(void)                  const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_MARGIN_INITIAL);                 }

   //--- setting the controlled value of the growth of the reserved funds for providing the minimum sum for all open positions (1) increase, (2) decrease value and (3) control level
   //--- getting (3) the change value of the growth of the reserved funds for providing the minimum sum for all open positions,
   //--- getting the flag of the growth change of the reserved funds for providing the minimum sum for all open positions exceeding the (4) increase, (5) decrease value
   void              SetControlMarginMaintenanceInc(const long value)      { this.SetControlledValueINC(ACCOUNT_PROP_MARGIN_MAINTENANCE,(double)::fabs(value));   }
   void              SetControlMarginMaintenanceDec(const long value)      { this.SetControlledValueDEC(ACCOUNT_PROP_MARGIN_MAINTENANCE,(double)::fabs(value));   }
   void              SetControlMarginMaintenanceLevel(const long value)    { this.SetControlledValueLEVEL(ACCOUNT_PROP_MARGIN_MAINTENANCE,(double)::fabs(value)); }
   double            GetValueChangedMarginMaintenance(void)          const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_MARGIN_MAINTENANCE);              }
   bool              IsIncreasedMarginMaintenance(void)              const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_MARGIN_MAINTENANCE);             }

   bool              IsDecreasedMarginMaintenance(void)              const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_MARGIN_MAINTENANCE);             }

   //--- setting the controlled assets (1) increase, (2) decrease value and (3) control level
   //--- getting (3) the assets change value,
   //--- getting the flag of the assets change exceeding the (4) increase, (5) decrease value
   void              SetControlAssetsInc(const long value)                 { this.SetControlledValueINC(ACCOUNT_PROP_ASSETS,(double)::fabs(value));               }
   void              SetControlAssetsDec(const long value)                 { this.SetControlledValueDEC(ACCOUNT_PROP_ASSETS,(double)::fabs(value));               }
   void              SetControlAssetsLevel(const long value)               { this.SetControlledValueLEVEL(ACCOUNT_PROP_ASSETS,(double)::fabs(value));             }
   double            GetValueChangedAssets(void)                     const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_ASSETS);                          }
   bool              IsIncreasedAssets(void)                         const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_ASSETS);                         }
   bool              IsDecreasedAssets(void)                         const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_ASSETS);                         }

   //--- setting the controlled liabilities (1) increase, (2) decrease value and (3) control level
   //--- getting (3) the liabilities change value,
   //--- getting the flag of the liabilities change exceeding the (4) increase, (5) decrease value
   void              SetControlLiabilitiesInc(const long value)            { this.SetControlledValueINC(ACCOUNT_PROP_LIABILITIES,(double)::fabs(value));          }
   void              SetControlLiabilitiesDec(const long value)            { this.SetControlledValueDEC(ACCOUNT_PROP_LIABILITIES,(double)::fabs(value));          }
   void              SetControlLiabilitiesLevel(const long value)          { this.SetControlledValueLEVEL(ACCOUNT_PROP_LIABILITIES,(double)::fabs(value));        }
   double            GetValueChangedLiabilities(void)                const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_LIABILITIES);                     }
   bool              IsIncreasedLiabilities(void)                    const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_LIABILITIES);                    }
   bool              IsDecreasedLiabilities(void)                    const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_LIABILITIES);                    }

   //--- setting the controlled blocked commissions (1) increase, (2) decrease value and (3) control level in points
   //--- getting (3) the blocked commissions change value,
   //--- getting the flag of the blocked commissions change exceeding the (4) increase, (5) decrease value
   void              SetControlComissionBlockedInc(const long value)       { this.SetControlledValueINC(ACCOUNT_PROP_COMMISSION_BLOCKED,(double)::fabs(value));   }
   void              SetControlComissionBlockedDec(const long value)       { this.SetControlledValueDEC(ACCOUNT_PROP_COMMISSION_BLOCKED,(double)::fabs(value));   }
   void              SetControlComissionBlockedLevel(const long value)     { this.SetControlledValueLEVEL(ACCOUNT_PROP_COMMISSION_BLOCKED,(double)::fabs(value)); }
   double            GetValueChangedComissionBlocked(void)           const { return this.GetPropDoubleChangedValue(ACCOUNT_PROP_COMMISSION_BLOCKED);              }
   bool              IsIncreasedComissionBlocked(void)               const { return (bool)this.GetPropDoubleFlagINC(ACCOUNT_PROP_COMMISSION_BLOCKED);             }
   bool              IsDecreasedComissionBlocked(void)               const { return (bool)this.GetPropDoubleFlagDEC(ACCOUNT_PROP_COMMISSION_BLOCKED);             }

//+------------------------------------------------------------------+
```

**In the class constructor, first specify data array size, initialize**
**all controlled data in the CBaseObj base object and then,** **after filling in all account object properties, fill**
**in the properties in the base object as well. Finally, update all account**
**data in the CBaseObj base object:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccount::CAccount(void)
  {
//--- Initialize control data
   this.SetControlDataArraySizeLong(ACCOUNT_PROP_INTEGER_TOTAL);
   this.SetControlDataArraySizeDouble(ACCOUNT_PROP_DOUBLE_TOTAL);
   this.ResetChangesParams();
   this.ResetControlsParams();

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
   this.m_type=COLLECTION_ACCOUNT_ID;

//--- Filling in the current account data
   for(int i=0;i<ACCOUNT_PROP_INTEGER_TOTAL;i++)
      this.m_long_prop_event[i][3]=this.m_long_prop[i];
   for(int i=0;i<ACCOUNT_PROP_DOUBLE_TOTAL;i++)
      this.m_double_prop_event[i][3]=this.m_double_prop[i];

//--- Update the base object data and search for changes
   CBaseObj::Refresh();
  }
//+-------------------------------------------------------------------+
```

**Beyond the class body, write the implementation of the virtual method of updating account data:**

```
//+------------------------------------------------------------------+
//| Update all account data                                          |
//+------------------------------------------------------------------+
void CAccount::Refresh(void)
  {
//--- Initialize event data
   this.m_is_event=false;
   this.m_hash_sum=0;
//--- Update integer properties
   this.m_long_prop[ACCOUNT_PROP_LOGIN]                                 = ::AccountInfoInteger(ACCOUNT_LOGIN);
   this.m_long_prop[ACCOUNT_PROP_TRADE_MODE]                            = ::AccountInfoInteger(ACCOUNT_TRADE_MODE);
   this.m_long_prop[ACCOUNT_PROP_LEVERAGE]                              = ::AccountInfoInteger(ACCOUNT_LEVERAGE);
   this.m_long_prop[ACCOUNT_PROP_LIMIT_ORDERS]                          = ::AccountInfoInteger(ACCOUNT_LIMIT_ORDERS);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_SO_MODE]                        = ::AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
   this.m_long_prop[ACCOUNT_PROP_TRADE_ALLOWED]                         = ::AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   this.m_long_prop[ACCOUNT_PROP_TRADE_EXPERT]                          = ::AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_MODE]                           = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_MARGIN_MODE) #else ACCOUNT_MARGIN_MODE_RETAIL_HEDGING #endif ;
   this.m_long_prop[ACCOUNT_PROP_CURRENCY_DIGITS]                       = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif ;
   this.m_long_prop[ACCOUNT_PROP_SERVER_TYPE]                           = (::TerminalInfoString(TERMINAL_NAME)=="MetaTrader 5" ? 5 : 4);

//--- Update real properties
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_BALANCE)]             = ::AccountInfoDouble(ACCOUNT_BALANCE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_CREDIT)]              = ::AccountInfoDouble(ACCOUNT_CREDIT);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_PROFIT)]              = ::AccountInfoDouble(ACCOUNT_PROFIT);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_EQUITY)]              = ::AccountInfoDouble(ACCOUNT_EQUITY);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN)]              = ::AccountInfoDouble(ACCOUNT_MARGIN);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_FREE)]         = ::AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_LEVEL)]        = ::AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_SO_CALL)]      = ::AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_SO_SO)]        = ::AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_INITIAL)]      = ::AccountInfoDouble(ACCOUNT_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_MAINTENANCE)]  =::AccountInfoDouble(ACCOUNT_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_ASSETS)]              = ::AccountInfoDouble(ACCOUNT_ASSETS);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_LIABILITIES)]         = ::AccountInfoDouble(ACCOUNT_LIABILITIES);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_COMMISSION_BLOCKED)]  =::AccountInfoDouble(ACCOUNT_COMMISSION_BLOCKED);

//--- Filling in the current account data in the base object
   for(int i=0;i<ACCOUNT_PROP_INTEGER_TOTAL;i++)
      this.m_long_prop_event[i][3]=this.m_long_prop[i];
   for(int i=0;i<ACCOUNT_PROP_DOUBLE_TOTAL;i++)
      this.m_double_prop_event[i][3]=this.m_double_prop[i];

//--- Update the base object data and search for changes
   CBaseObj::Refresh();
   this.CheckEvents();
  }
//+------------------------------------------------------------------+
```

Here, the account event flag is reset and the hash
sum is set to zero (most probably, the hash sum will be removed in the following versions if it turns out to be unnecessary for other
CBaseObj-based objects).

Next, all account object properties are filled in, the account data
in the base object is filled (like in the class constructor) and the
method of updating the base object is called. In the base object, the current data is updated and the search for object property values
is performed. If the change of the values set for searching for events is exceeded, the object base events are generated.

Next, using the **CheckEvents()** method of the parent class, check
the presence of base events in the list of the CBaseObj object base events. If the events are present, the method creates the list of
events of its descendant. In this case, this is the list of account events.

**The CAccount class has been improved.**

Now let's make corrections in the account collection class.

Open \\MQL5\\Include\\DoEasy\\Collections\ **AccountsCollection.mqh** and make
the necessary changes.

**Remove all unnecessary elements:**

```
//+------------------------------------------------------------------+
//| Account collection                                               |
//+------------------------------------------------------------------+
class CAccountsCollection : public CBaseObj
  {
private:
   struct MqlDataAccount
     {
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
   MqlDataAccount    m_struct_curr_account;              // Account current data
   MqlDataAccount    m_struct_prev_account;              // Account previous data

   string            m_symbol;                           // Current symbol
   CListObj          m_list_accounts;                    // Account object list
   int               m_index_current;                    // Index of an account object featuring the current account data
   //--- Leverage
   long              m_changed_leverage_value;           // Leverage change value
   bool              m_is_change_leverage_inc;           // Leverage increase flag
   bool              m_is_change_leverage_dec;           // Leverage decrease flag
   //--- Number of active pending orders
   int               m_changed_limit_orders_value;       // Change value of the maximum allowed number of active pending orders
   bool              m_is_change_limit_orders_inc;       // Increase flag of the maximum allowed number of active pending orders
   bool              m_is_change_limit_orders_dec;       // Decrease flag of the maximum allowed number of active pending orders
   //--- Trading on an account
   bool              m_is_change_trade_allowed_on;       // The flag allowing to trade for the current account from the server side
   bool              m_is_change_trade_allowed_off;      // The flag prohibiting trading for the current account from the server side
   //--- Auto trading on an account
   bool              m_is_change_trade_expert_on;        // The flag allowing to trade for an EA from the server side
   bool              m_is_change_trade_expert_off;       // The flag prohibiting trading for an EA from the server side
   //--- Balance
   double            m_control_balance_inc;              // Tracked balance increase value
   double            m_control_balance_dec;              // Tracked balance decrease value
   double            m_changed_balance_value;            // Balance change value
   bool              m_is_change_balance_inc;            // The flag of the balance change exceeding the increase value
   bool              m_is_change_balance_dec;            // The flag of the balance change exceeding the decrease value
   //--- Credit
   double            m_changed_credit_value;             // Credit change value
   bool              m_is_change_credit_inc;             // Credit increase flag
   bool              m_is_change_credit_dec;             // Credit decrease flag
   //--- Profit
   double            m_control_profit_inc;               // Controlled profit growth value
   double            m_control_profit_dec;               // Controlled profit decrease value
   double            m_changed_profit_value;             // Profit change value
   bool              m_is_change_profit_inc;             // The flag of the profit change exceeding the increase value
   bool              m_is_change_profit_dec;             // The flag of the profit change exceeding the decrease value
   //--- Funds (equity)
   double            m_control_equity_inc;               // Controlled funds increase value
   double            m_control_equity_dec;               // Controlled funds decrease value
   double            m_changed_equity_value;             // Funds change value
   bool              m_is_change_equity_inc;             // The flag of the funds change exceeding the increase value
   bool              m_is_change_equity_dec;             // The flag of the funds change exceeding the decrease value
   //--- Margin
   double            m_control_margin_inc;               // Controlled margin increase value
   double            m_control_margin_dec;               // Controlled margin decrease value
   double            m_changed_margin_value;             // Margin change value
   bool              m_is_change_margin_inc;             // The flag of the margin change exceeding the increase value
   bool              m_is_change_margin_dec;             // The flag of the margin change exceeding the decrease value
   //--- Free margin
   double            m_control_margin_free_inc;          // Controlled free margin increase value
   double            m_control_margin_free_dec;          // Controlled free margin decrease value
   double            m_changed_margin_free_value;        // Free margin change valu
   bool              m_is_change_margin_free_inc;        // The flag of the free margin change exceeding the increase value
   bool              m_is_change_margin_free_dec;        // The flag of the free margin change exceeding the decrease value
   //--- Margin level
   double            m_control_margin_level_inc;         // Controlled margin level increase value
   double            m_control_margin_level_dec;         // Controlled margin level decrease value
   double            m_changed_margin_level_value;       // Margin level change value
   bool              m_is_change_margin_level_inc;       // The flag of the free margin change exceeding the increase value
   bool              m_is_change_margin_level_dec;       // The flag of the margin level change exceeding the decrease value
   //--- Margin Call
   double            m_changed_margin_so_call_value;     // Margin Call level change value
   bool              m_is_change_margin_so_call_inc;     // Margin Call level increase value
   bool              m_is_change_margin_so_call_dec;     // Margin Call level decrease value
   //--- MarginStopOut
   double            m_changed_margin_so_so_value;       // Margin StopOut level change value
   bool              m_is_change_margin_so_so_inc;       // Margin StopOut level increase flag
   bool              m_is_change_margin_so_so_dec;       // Margin StopOut level decrease flag
   //--- Guarantee sum for pending orders
   double            m_control_margin_initial_inc;       // Controlled increase value of the reserved funds for providing the guarantee sum for pending orders
   double            m_control_margin_initial_dec;       // Controlled decrease value of the reserved funds for providing the guarantee sum for pending orders
   double            m_changed_margin_initial_value;     // The change value of the reserved funds for providing the guarantee sum for pending orders
   bool              m_is_change_margin_initial_inc;     // The flag of the reserved funds for providing the guarantee sum for pending orders exceeding the increase value
   bool              m_is_change_margin_initial_dec;     // The flag of the reserved funds for providing the guarantee sum for pending orders exceeding the decrease value
   //--- Guarantee sum for open positions
   double            m_control_margin_maintenance_inc;   // Controlled increase value of the funds reserved on an account to ensure a minimum amount for all open positions
   double            m_control_margin_maintenance_dec;   // Controlled decrease value of the funds reserved on an account to ensure a minimum amount for all open positions
   double            m_changed_margin_maintenance_value; // The change value of the funds reserved on an account to ensure a minimum amount for all open positions
   bool              m_is_change_margin_maintenance_inc; // The flag of the funds reserved on an account to ensure a minimum amount for all open positions exceeding the increase value
   bool              m_is_change_margin_maintenance_dec; // The flag of the funds reserved on an account to ensure a minimum amount for all open positions exceeding the decrease value
   //--- Assets
   double            m_control_assets_inc;               // Controlled assets increase value
   double            m_control_assets_dec;               // Controlled assets decrease value
   double            m_changed_assets_value;             // Assets change value
   bool              m_is_change_assets_inc;             // The flag of the assets change exceeding the increase value
   bool              m_is_change_assets_dec;             // The flag of the assets change exceeding the decrease value
   //--- Liabilities
   double            m_control_liabilities_inc;          // Controlled liabilities increase value
   double            m_control_liabilities_dec;          // Controlled liabilities decrease value
   double            m_changed_liabilities_value;        // Liabilities change values
   bool              m_is_change_liabilities_inc;        // The flag of the liabilities change exceeding the increase value
   bool              m_is_change_liabilities_dec;        // The flag of the liabilities change exceeding the decrease value
   //--- Blocked commissions
   double            m_control_comission_blocked_inc;    // Controlled blocked commissions increase value
   double            m_control_comission_blocked_dec;    // Controlled blocked commissions decrease value
   double            m_changed_comission_blocked_value;  // Blocked commissions changed value
   bool              m_is_change_comission_blocked_inc;  // The flag of the blocked commissions change exceeding the increase value
   bool              m_is_change_comission_blocked_dec;  // The flag of the blocked commissions change exceeding the decrease value

//--- Initialize the variables of (1) tracked, (2) controlled account data
   void              InitChangesParams(void);
   void              InitControlsParams(void);
//--- Set an event type and fill in the event list
   virtual void      SetTypeEvent(void);
//--- Write the current account data to the account object properties
   void              SetAccountsParams(CAccount* account);
//--- Check the account object presence in the collection list
   bool              IsPresent(CAccount* account);
//--- Find and return the account object index with the current account data
   int               Index(void);
public:
//--- Return the full account collection list "as is"
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_accounts;                                         }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
//--- Return (1) the current account object index and (2) event ID by its number in the list
   int               IndexCurrentAccount(void)                                                        const { return this.m_index_current;                                          }
   ENUM_ACCOUNT_EVENT GetEventID(const int shift=WRONG_VALUE,const bool check_out=true);
//--- (1) Set and (2) return the current symbol
   void              SetSymbol(const string symbol)                                                         { this.m_symbol=symbol;                                  }
   string            GetSymbol(void)                                                                  const { return this.m_symbol;                                  }

//--- Constructor, destructor
                     CAccountsCollection();
                    ~CAccountsCollection();
//--- Add the account object to the list
   bool              AddToList(CAccount* account);
//--- (1) Save account objects from the list to the files
//--- (2) Save account objects from the files to the list
   bool              SaveObjects(void);
   bool              LoadObjects(void);
//--- Return the account event description
   string            EventDescription(const ENUM_ACCOUNT_EVENT event);
//--- Update the current account data
   virtual void      Refresh(void);

//--- Get and set the parameters of tracked changes
   //--- Leverage:
   //--- (1) Leverage change value, (2) Leverage increase flag, (3) Leverage decrease flag
   long              GetValueChangedLeverage(void)                                                    const { return this.m_changed_leverage_value;                  }
   bool              IsIncreaseLeverage(void)                                                         const { return this.m_is_change_leverage_inc;                  }
   bool              IsDecreaseLeverage(void)                                                         const { return this.m_is_change_leverage_dec;                  }
   //--- Number of active pending orders:
   //--- (1) Change value, (2) Increase flag, (3) Decrease flag
   int               GetValueChangedLimitOrders(void)                                                 const { return this.m_changed_limit_orders_value;              }
   bool              IsIncreaseLimitOrders(void)                                                      const { return this.m_is_change_limit_orders_inc;              }
   bool              IsDecreaseLimitOrders(void)                                                      const { return this.m_is_change_limit_orders_dec;              }
   //--- Trading on an account:
   //--- (1) The flag allowing to trade for the current account, (2) The flag prohibiting trading for the current account from the server side
   bool              IsOnTradeAllowed(void)                                                           const { return this.m_is_change_trade_allowed_on;              }
   bool              IsOffTradeAllowed(void)                                                          const { return this.m_is_change_trade_allowed_off;             }
   //--- Auto trading on an account:
   //--- (1) The flag allowing to trade for an EA, (2) The flag prohibiting trading for an EA from the server side
   bool              IsOnTradeExpert(void)                                                            const { return this.m_is_change_trade_expert_on;               }
   bool              IsOffTradeExpert(void)                                                           const { return this.m_is_change_trade_expert_off;              }
   //--- Balance:
   //--- setting the controlled value of the balance (1) increase, (2) decrease
   //--- getting (3) the balance change value,
   //--- getting the flag of the balance change exceeding the (4) increase value, (5) decrease value
   void              SetControlBalanceInc(const double value)                                               { this.m_control_balance_inc=::fabs(value);              }
   void              SetControlBalanceDec(const double value)                                               { this.m_control_balance_dec=::fabs(value);              }
   double            GetValueChangedBalance(void)                                                     const { return this.m_changed_balance_value;                   }
   bool              IsIncreaseBalance(void)                                                          const { return this.m_is_change_balance_inc;                   }
   bool              IsDecreaseBalance(void)                                                          const { return this.m_is_change_balance_dec;                   }
   //--- Credit:
   //--- getting (1) the credit change value, (2) credit increase flag, (3) decrease flag
   double            GetValueChangedCredit(void)                                                      const { return this.m_changed_credit_value;                    }
   bool              IsIncreaseCredit(void)                                                           const { return this.m_is_change_credit_inc;                    }
   bool              IsDecreaseCredit(void)                                                           const { return this.m_is_change_credit_dec;                    }
   //--- Profit:
   //--- setting the controlled profit (1) increase, (2) decrease value
   //--- getting the (3) profit change value,
   //--- getting the flag of the profit change exceeding the (4) increase, (5) decrease value
   void              SetControlProfitInc(const double value)                                                { this.m_control_profit_inc=::fabs(value);               }
   void              SetControlProfitDec(const double value)                                                { this.m_control_profit_dec=::fabs(value);               }
   double            GetValueChangedProfit(void)                                                      const { return this.m_changed_profit_value;                    }
   bool              IsIncreaseProfit(void)                                                           const { return this.m_is_change_profit_inc;                    }
   bool              IsDecreaseProfit(void)                                                           const { return this.m_is_change_profit_dec;                    }
   //--- Equity:
   //--- setting the controlled equity (1) increase, (2) decrease value
   //--- getting the (3) equity change value,
   //--- getting the flag of the equity change exceeding the (4) increase, (5) decrease value
   void              SetControlEquityInc(const double value)                                                { this.m_control_equity_inc=::fabs(value);               }
   void              SetControlEquityDec(const double value)                                                { this.m_control_equity_dec=::fabs(value);               }
   double            GetValueChangedEquity(void)                                                      const { return this.m_changed_equity_value;                    }
   bool              IsIncreaseEquity(void)                                                           const { return this.m_is_change_equity_inc;                    }
   bool              IsDecreaseEquity(void)                                                           const { return this.m_is_change_equity_dec;                    }
   //--- Margin:
   //--- setting the controlled margin (1) increase, (2) decrease value
   //--- getting the (3) margin change value,
   //--- getting the flag of the margin change exceeding the (4) increase, (5) decrease value
   void              SetControlMarginInc(const double value)                                                { this.m_control_margin_inc=::fabs(value);               }
   void              SetControlMarginDec(const double value)                                                { this.m_control_margin_dec=::fabs(value);               }
   double            GetValueChangedMargin(void)                                                      const { return this.m_changed_margin_value;                    }
   bool              IsIncreaseMargin(void)                                                           const { return this.m_is_change_margin_inc;                    }
   bool              IsDecreaseMargin(void)                                                           const { return this.m_is_change_margin_dec;                    }
   //--- Free margin:
   //--- setting the controlled free margin (1) increase, (2) decrease value
   //--- getting the (3) free margin change value,
   //--- getting the flag of the free margin change exceeding the (4) increase, (5) decrease value
   void              SetControlMarginFreeInc(const double value)                                            { this.m_control_margin_free_inc=::fabs(value);          }
   void              SetControlMarginFreeDec(const double value)                                            { this.m_control_margin_free_dec=::fabs(value);          }
   double            GetValueChangedMarginFree(void)                                                  const { return this.m_changed_margin_free_value;               }
   bool              IsIncreaseMarginFree(void)                                                       const { return this.m_is_change_margin_free_inc;               }
   bool              IsDecreaseMarginFree(void)                                                       const { return this.m_is_change_margin_free_dec;               }
   //--- Margin level:
   //--- setting the controlled margin level (1) increase, (2) decrease value
   //--- getting the (3) margin level change value,
   //--- getting the flag of the margin level change exceeding the (4) increase, (5) decrease value
   void              SetControlMarginLevelInc(const double value)                                           { this.m_control_margin_level_inc=::fabs(value);         }
   void              SetControlMarginLevelDec(const double value)                                           { this.m_control_margin_level_dec=::fabs(value);         }
   double            GetValueChangedMarginLevel(void)                                                 const { return this.m_changed_margin_level_value;              }
   bool              IsIncreaseMarginLevel(void)                                                      const { return this.m_is_change_margin_level_inc;              }
   bool              IsDecreaseMarginLevel(void)                                                      const { return this.m_is_change_margin_level_dec;              }
   //--- Margin Call:
   //--- getting (1) Margin Call change value, (2) increase flag, (3) decrease flag
   double            GetValueChangedMarginCall(void)                                                  const { return this.m_changed_margin_so_call_value;            }
   bool              IsIncreaseMarginCall(void)                                                       const { return this.m_is_change_margin_so_call_inc;            }
   bool              IsDecreaseMarginCall(void)                                                       const { return this.m_is_change_margin_so_call_dec;            }
   //--- Margin StopOut:
   //--- getting (1) Margin StopOut change value, (2) increase flag, (3) decrease flag
   double            GetValueChangedMarginStopOut(void)                                               const { return this.m_changed_margin_so_so_value;              }
   bool              IsIncreaseMarginStopOut(void)                                                    const { return this.m_is_change_margin_so_so_inc;              }
   bool              IsDecreasMarginStopOute(void)                                                    const { return this.m_is_change_margin_so_so_dec;              }
   //--- Guarantee sum for pending orders:
   //--- setting the controlled value of the reserved funds for providing the guarantee sum for pending orders (1) increase, (2) decrease
   //--- getting the change value of the (3) reserved funds for providing the guarantee sum,
   //--- getting the flag of the reserved funds for providing the guarantee sum for pending orders exceeding the (4) increase, (5) decrease value
   void              SetControlMarginInitialInc(const double value)                                         { this.m_control_margin_initial_inc=::fabs(value);       }
   void              SetControlMarginInitialDec(const double value)                                         { this.m_control_margin_initial_dec=::fabs(value);       }
   double            GetValueChangedMarginInitial(void)                                               const { return this.m_changed_margin_initial_value;            }
   bool              IsIncreaseMarginInitial(void)                                                    const { return this.m_is_change_margin_initial_inc;            }
   bool              IsDecreaseMarginInitial(void)                                                    const { return this.m_is_change_margin_initial_dec;            }
   //--- Guarantee sum for open positions:
   //--- setting the controlled value of the funds reserved on an account to ensure a minimum amount for all open positions (1) increase, (2) decrease
   //--- getting the change value of the (3) reserved funds for providing the guarantee sum,
   //--- getting the flag of the reserved funds for providing the guarantee sum for pending orders exceeding the (4) increase, (5) decrease value
   void              SetControlMarginMaintenanceInc(const double value)                                     { this.m_control_margin_maintenance_inc=::fabs(value);   }
   void              SetControlMarginMaintenanceDec(const double value)                                     { this.m_control_margin_maintenance_dec=::fabs(value);   }
   double            GetValueChangedMarginMaintenance(void)                                           const { return this.m_changed_margin_maintenance_value;        }
   bool              IsIncreaseMarginMaintenance(void)                                                const { return this.m_is_change_margin_maintenance_inc;        }
   bool              IsDecreaseMarginMaintenance(void)                                                const { return this.m_is_change_margin_maintenance_dec;        }
   //--- Assets:
   //--- setting the controlled value of the assets (1) increase, (2) decrease
   //--- getting (3) the assets change value,
   //--- getting the flag of the change exceeding the (4) increase, (5) decrease value
   void              SetControlAssetsInc(const double value)                                                { this.m_control_assets_inc=::fabs(value);               }
   void              SetControlAssetsDec(const double value)                                                { this.m_control_assets_dec=::fabs(value);               }
   double            GetValueChangedAssets(void)                                                      const { return this.m_changed_assets_value;                    }
   bool              IsIncreaseAssets(void)                                                           const { return this.m_is_change_assets_inc;                    }
   bool              IsDecreaseAssets(void)                                                           const { return this.m_is_change_assets_dec;                    }
   //--- Liabilities:
   //--- setting the controlled value of the liabilities (1) increase, (2) decrease
   //--- getting (3) the liabilities change value,
   //--- getting the flag of the liabilities change exceeding the (4) increase, (5) decrease value
   void              SetControlLiabilitiesInc(const double value)                                           { this.m_control_liabilities_inc=::fabs(value);          }
   void              SetControlLiabilitiesDec(const double value)                                           { this.m_control_liabilities_dec=::fabs(value);          }
   double            GetValueChangedLiabilities(void)                                                 const { return this.m_changed_liabilities_value;               }
   bool              IsIncreaseLiabilities(void)                                                      const { return this.m_is_change_liabilities_inc;               }
   bool              IsDecreaseLiabilities(void)                                                      const { return this.m_is_change_liabilities_dec;               }
   //--- Blocked commissions:
   //--- setting the controlled blocked commissions (1) increase, (2) decrease value
   //--- getting (3) the blocked commissions change value,
   //--- getting the flag of the blocked commissions change exceeding the (4) increase, (5) decrease value
   void              SetControlComissionBlockedInc(const double value)                                      { this.m_control_comission_blocked_inc=::fabs(value);    }
   void              SetControlComissionBlockedDec(const double value)                                      { this.m_control_comission_blocked_dec=::fabs(value);    }
   double            GetValueChangedComissionBlocked(void)                                            const { return this.m_changed_comission_blocked_value;         }
   bool              IsIncreaseComissionBlocked(void)                                                 const { return this.m_is_change_comission_blocked_inc;         }
   bool              IsDecreaseComissionBlocked(void)                                                 const { return this.m_is_change_comission_blocked_dec;         }
  };
//+------------------------------------------------------------------+
```

**Change types of methods and add declarations of some**
**necessary**

**variables and methods:**

```
//+------------------------------------------------------------------+
//| Account collection                                               |
//+------------------------------------------------------------------+
class CAccountsCollection : public CBaseObj
  {
private:
   string            m_symbol;                                    // Current symbol
   CListObj          m_list_accounts;                             // Account object list
   int               m_index_current;                             // Index of an account object featuring the current account data
   int               m_last_event;                                // The last event

//--- Check the account object presence in the collection list
   bool              IsPresent(CAccount* account);
//--- Find and return the account object index with the current account data
   int               Index(void);
public:
//--- Return the full account collection list "as is"
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_accounts;                                         }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
//--- Return (1) the current account object index and (2) event ID by its number in the list
   int               IndexCurrentAccount(void)                                                        const { return this.m_index_current;                                          }
   int               GetEventID(const int shift=WRONG_VALUE,const bool check_out=true);
//--- (1) Set and (2) return the current symbol
   void              SetSymbol(const string symbol)                                                         { this.m_symbol=symbol;                                  }
   string            GetSymbol(void)                                                                  const { return this.m_symbol;                                  }
//--- (1) Update data, (2) working with events of the current account
   virtual void      Refresh(void);
   void              RefreshAndEventsControl(void);

//--- Constructor, destructor
                     CAccountsCollection();
                    ~CAccountsCollection();
//--- Add the account object to the list
   bool              AddToList(CAccount* account);
//--- (1) Save account objects from the list to the files
//--- (2) Save account objects from the files to the list
   bool              SaveObjects(void);
   bool              LoadObjects(void);
  };
```

**Remove calling two removed methods and clearing the remote**
**structure from the class constructor:**

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

All these variables, structures and methods have now been replaced with the ready-made base object functionality. There is no need to
recreate them again for descendant classes.

**The event functionality should also be deleted in the account data update method:**

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

**All removed elements are replaced with calling the method for updating the**
**base class:**

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
   account.Refresh();
  }
//+------------------------------------------------------------------+
```

**The method updating object data and searching for property changes to generate events:**

```
//+------------------------------------------------------------------+
//| Working with the events of the collection symbol list            |
//+------------------------------------------------------------------+
void CAccountsCollection::RefreshAndEventsControl(void)
  {
   ::ResetLastError();
   if(!::SymbolInfoTick(::Symbol(),this.m_tick))
     {
      this.m_global_error=::GetLastError();
      return;
     }
   if(this.m_index_current==WRONG_VALUE)
      return;
   this.m_is_event=false;
   this.m_list_events.Clear();
   this.m_list_events.Sort();
   CAccount* account=this.m_list_accounts.At(this.m_index_current);
   if(account==NULL)
      return;
   account.Refresh();
   if(!account.IsEvent())
      return;
   CArrayObj *list=account.GetListEvents();
   if(list==NULL)
      return;
   this.m_is_event=true;
   this.m_event_code=account.GetEventCode();
   int n=list.Total();
   for(int j=0; j<n; j++)
     {
      CEventBaseObj *event=list.At(j);
      if(event==NULL)
         continue;
      this.m_last_event=event.ID();
      if(this.EventAdd((ushort)event.ID(),event.LParam(),event.DParam(),event.SParam()))
        {
         ::EventChartCustom(this.m_chart_id,(ushort)event.ID(),event.LParam(),event.DParam(),event.SParam());
        }
     }
  }
//+------------------------------------------------------------------+
```

Here, in case of an unsuccessful attempt to obtain the current symbol tick,
save the error code and exit the method. If the current account in the account collection list is not found for some reason and its
index is negative, exit the method.

Reset the account event flag,

clear the list of account events and set the sorted list flag to it.

Get the current account object from the account collection list
and update account data.


If there is no account event at the moment, exit the method.


Otherwise,

get the list of account base events from the base object, set
the account event flag and get the last event code (most
likely, this remnant of the past will be removed later as well).

In a loop by the list of base events, get
the next event from the list, save the last account event, add
it to the list of account events and send the event to the control program
chart.

Remove **SetAccountsParams()**, **SetEventCode()**, **EventDescription()**, **InitChangesParams()** and **InitControlsParams()**
from the method implementation class listing.

The method that previously returned the enumeration value now **returns the account event by its number in the list**.
The enumeration has been removed, so the method returns an

'int' value
now. If the event is not found, return -1:

```
//+------------------------------------------------------------------+
//| Return the account event by its number in the list               |
//+------------------------------------------------------------------+
int CAccountsCollection::GetEventID(const int shift=WRONG_VALUE,const bool check_out=true)
  {
   CEventBaseObj *event=this.GetEvent(shift,check_out);
   if(event==NULL)
      return WRONG_VALUE;
   return (int)event.ID();
  }
//+------------------------------------------------------------------+
```

**These are all the necessary changes in the account collection class.**

Now it only remains to make slight changes in the library's **CEngine** base object class.

Open
\\MQL5\\Include\\DoEasy\

**Engine.mqh** and make the changes.

Previously, the variable storing the **m\_last\_account\_event** account's last event, as well as the **GetAccountEventByIndex()** and **LastAccountEvent()**
methods returning it, had the ENUM\_ACCOUNT\_EVENT enumeration type but we got rid of it. **Let's**
**introduce them with 'int' type:**

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
   int                  m_last_account_event;            // Last event in the account properties

//--- Return the list of (1) accounts, (2) account events, (3) account change event by its index in the list
//--- (4) the current account, (5) event description
   CArrayObj           *GetListAllAccounts(void)                        { return this.m_accounts.GetList();                   }
   CArrayObj           *GetListAccountEvents(void)                      { return this.m_accounts.GetListEvents();             }
   int                  GetAccountEventByIndex(const int index=-1)      { return this.m_accounts.GetEventID(index);           }
   CAccount            *GetAccountCurrent(void);

//--- Return the (1) last trading event, (2) the last event in the account properties, (3) hedging account flag, (4) flag of working in the tester
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_last_trade_event;                     }
   int                  LastAccountEvent(void)                    const { return this.m_last_account_event;                   }
   int                  LastSymbolsEvent(void)                    const { return this.m_last_symbol_event;                    }
```

**Initialize the m\_last\_account\_event variable with -1**
in the class constructor's initialization list. Previously, we initialized it using the ACCOUNT\_EVENT\_NO\_EVENT constant of the removed
ENUM\_ACCOUNT\_EVENT enumeration.

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

Since we replaced the Refresh() method name of the symbol collection class with RefreshAndEventsControl() and created the same method in the
account collection class,

**in the methods of working with symbol events and accounts,**
**replace the name of the called method:**

```
//+------------------------------------------------------------------+
//| Working with symbol collection events                            |
//+------------------------------------------------------------------+
void CEngine::SymbolEventsControl(void)
  {
   this.m_symbols.RefreshAndEventsControl();
   this.m_is_symbol_event=this.m_symbols.IsEvent();
//--- If there are changes in symbol properties
   if(this.m_is_symbol_event)
     {
      //--- Get the last event of the symbol property change
      this.m_last_symbol_event=this.m_symbols.GetLastEvent();
     }
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Check account events                                             |
//+------------------------------------------------------------------+
void CEngine::AccountEventsControl(void)
  {
//--- Check account property changes and set the flag of the account change events
   this.m_accounts.RefreshAndEventsControl();
   this.m_is_account_event=this.m_accounts.IsEvent();
//--- If there are account property changes
   if(this.m_is_account_event)
     {
      //--- Get the last event of the account property change
      this.m_last_account_event=this.m_accounts.GetEventID();
     }
  }
//+------------------------------------------------------------------+
```

**These are all the changes in the CEngine class.**

Now we are able to programmatically set the properties we want to track for any of the classes based on the CBaseObj base object. Besides, we are
able to set property change values, upon exceeding which, events of the base class descendants are generated.

Let's see how all this can be done.

### Testing the setting of the tracking parameters and receiving object events

To perform the test, we will use the [test EA from the previous article](https://www.mql5.com/en/articles/7124#node03)
and save it

in the new folder under a
new name \\MQL5\\Experts\\TestDoEasy\**Part18**\**TestDoEasyPart18.mq5**.

We need to test setting the parameters whose custom changes we want to track in two different classes. Now this can be done in a unified way.

**Let's track for the CSymbol class**

- Increase of the Bid price of all applied symbols by 10 points
- Decrease of the Bid price of all applied symbols by 10 points
- Increase of the spread of all applied symbols by 4 points
- Decrease of the spread of all applied symbols by 4 points
- Control a spread value exceeding 15 points for all used symbols
- Control the current symbol Bid price crossing the value of 1.10300

**For the CAccount class, we will track**

- Increase of the current profit by 10 account currency units
- Increase of the funds by 15 account currency units
- Control the increase of the current profit above 20 account currency units


When the funds increase exceeds 15 units, close the most profitable position if it is present and its profit is greater than zero.

**Set all the necessary values in the OnInit() handler for the**
**testing purposes:**

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

//--- Set controlled values for symbols
   string ru1="",ru2="",ru3="",en1="",en2="",en3="";
   //--- Get the list of all collection symbols
   CArrayObj *list=engine.GetListAllUsedSymbols();
   if(list!=NULL && list.Total()!=0)
     {
      //--- In a loop by the list, set the necessary values for tracked symbol properties
      //--- By default, the LONG_MAX value is set to all properties, which means "Do not track this property"
      //--- It can be enabled or disabled (by setting the value less than LONG_MAX or vice versa - set the LONG_MAX value) at any time and anywhere in the program
      for(int i=0;i<list.Total();i++)
        {
         CSymbol* symbol=list.At(i);
         if(symbol==NULL)
            continue;
         //--- Set control of the symbol price increase by 10 points
         symbol.SetControlBidInc(10*symbol.Point());
         ru1="Контролируем увеличение цены Bid для символа ";
         ru2=" на ";
         ru3=" пунктов";
         en1="Bid price increase control for symbol ";
         en2=" by ";
         en3=" points";
         Print(TextByLanguage(ru1,en1),symbol.Name(),TextByLanguage(ru2,en2),DoubleToString(symbol.GetControlledDoubleValueINC(SYMBOL_PROP_BID),symbol.Digits()));

         //--- Set control of the symbol price decrease by 10 points
         symbol.SetControlBidDec(10*symbol.Point());
         ru1="Контролируем уменьшение цены Bid для символа ";
         ru2=" на ";
         ru3=" пунктов";
         en1="Bid price decrease control for symbol ";
         en2=" by ";
         en3=" points";
         Print(TextByLanguage(ru1,en1),symbol.Name(),TextByLanguage(ru2,en2),DoubleToString(symbol.GetControlledDoubleValueINC(SYMBOL_PROP_BID),symbol.Digits()));

         //--- Set control of the symbol spread increase by 4 points
         symbol.SetControlSpreadInc(4);
         ru1="Контролируем увеличение спреда для символа ";
         ru2=" на ";
         ru3=" пунктов";
         en1="Spread value increase control for symbol ";
         en2=" by ";
         en3=" points";
         Print(TextByLanguage(ru1,en1),symbol.Name(),TextByLanguage(ru2,en2),(string)symbol.GetControlledLongValueINC(SYMBOL_PROP_SPREAD),TextByLanguage(ru3,en3));

         //--- Set control of the symbol spread decrease by 4 points
         symbol.SetControlSpreadDec(4);
         ru1="Контролируем уменьшение спреда для символа ";
         ru2=" на ";
         ru3=" пунктов";
         en1="Spread value decrease control for symbol ";
         en2=" by ";
         en3=" points";
         Print(TextByLanguage(ru1,en1),symbol.Name(),TextByLanguage(ru2,en2),(string)symbol.GetControlledLongValueDEC(SYMBOL_PROP_SPREAD),TextByLanguage(ru3,en3));

         //--- Set control of the current spread by the value of 15 points
         symbol.SetControlSpreadLevel(15);
         ru1="Контролируем значение спреда для символа ";
         ru2=" в ";
         ru3=" пунктов";
         en1="Control the spread value for the symbol ";
         en2=" at ";
         en3=" points";
         Print(TextByLanguage(ru1,en1),symbol.Name(),TextByLanguage(ru2,en2),(string)symbol.GetControlledLongValueLEVEL(SYMBOL_PROP_SPREAD),TextByLanguage(ru3,en3));
         Print("------");

         //--- Set control of the price crossing the level of 1.10700 for the current symbol
         if(symbol.Name()==Symbol())
           {
            symbol.SetControlBidLevel(1.10300);
            ru1="Контролируемый уровень цены Bid для символа ";
            ru2=" установлен в значение ";
            en1="Controlled level of Bid price for the symbol ";
            en2=" is set to ";
            Print(TextByLanguage(ru1,en1),symbol.Name(),TextByLanguage(ru2,en2),DoubleToString(symbol.GetControlledDoubleValueLEVEL(SYMBOL_PROP_BID),symbol.Digits()));
           }
        }
     }
//--- Set controlled values for the current account
   Print("------");
   CAccount* account=engine.GetAccountCurrent();
   if(account!=NULL)
     {
      //--- Set control of the profit increase
      account.SetControlledValueINC(ACCOUNT_PROP_PROFIT,10.0);
      Print(TextByLanguage("Контролируем увеличение прибыли аккаунта на ","Controlling account profit increase by "),DoubleToString(account.GetControlledDoubleValueINC(ACCOUNT_PROP_PROFIT),(int)account.CurrencyDigits())," ",account.Currency());
      //--- Set control of the funds increase
      account.SetControlledValueINC(ACCOUNT_PROP_EQUITY,15.0);
      Print(TextByLanguage("Контролируем увеличение средств аккаунта на ","Controlling account equity increase by "),DoubleToString(account.GetControlledDoubleValueINC(ACCOUNT_PROP_EQUITY),(int)account.CurrencyDigits())," ",account.Currency());
      //--- Set profit control level
      account.SetControlledValueLEVEL(ACCOUNT_PROP_PROFIT,20.0);
      Print(TextByLanguage("Контролируем уровень прибыли аккаунта в ","Controlling the account profit level of "),DoubleToString(account.GetControlledDoubleValueLEVEL(ACCOUNT_PROP_PROFIT),(int)account.CurrencyDigits())," ",account.Currency());
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

The listing features all the necessary comments. I believe, all is easy to understand there. After setting a tracked value for a property, it
is immediately displayed in the journal (as an example of receiving a set tracked property value).

From the **OnTick()** handler, remove the **last\_account\_event** variable for storing the last account event, as it was
needed earlier for defining a new event.

**Now the handler looks as follows:**

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Initializing the last events
   static ENUM_TRADE_EVENT last_trade_event=WRONG_VALUE;
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

Now all new event flags can be obtained from the library's **CEngine** main object. More importantly, we can obtain even
two similar events from different symbols. This is related to defining event symbols, while the account always remains the same (the
current one). On the contrary, this was impossible when using variables since the current and previous events were supposedly the same and,
therefore, there was no event. That was not right.

**The library event handler has been improved to define**
**account events and a response to increase in the amount of funds by a specified value:**

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

//--- Handling account events
   else if(source==COLLECTION_ACCOUNT_ID)
     {
      CAccount *account=engine.GetAccountCurrent();
      if(account==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=int(idx<ACCOUNT_PROP_INTEGER_TOTAL ? 0 : account.CurrencyDigits());
      //--- Event text description
      string id_descr=(idx<ACCOUNT_PROP_INTEGER_TOTAL ? account.GetPropertyDescription((ENUM_ACCOUNT_PROP_INTEGER)idx) : account.GetPropertyDescription((ENUM_ACCOUNT_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=DoubleToString(dparam,digits);

      //--- Checking event reasons and handling the increase of funds by a specified value,
      //--- for other events, simply display their descriptions in the journal

      //--- In case of a property value increase
      if(reason==BASE_EVENT_REASON_INC)
        {
         //--- Display an event in the journal
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
         //--- if this is an equity increase
         if(idx==ACCOUNT_PROP_EQUITY)
           {
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
      //--- Other events are simply displayed in the journal
      if(reason==BASE_EVENT_REASON_DEC)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling trading events
   else if(idx>TRADE_EVENT_NO_EVENT && idx<TRADE_EVENTS_NEXT_CODE)
     {
      event=EnumToString((ENUM_TRADE_EVENT)ushort(idx));
      int digits=(int)SymbolInfoInteger(sparam,SYMBOL_DIGITS);
     }

//--- Handling market watch window events
   else if(idx>MARKET_WATCH_EVENT_NO_EVENT && idx<SYMBOL_EVENTS_NEXT_CODE)
     {
      string name="";
      //--- Market Watch window event
      string descr=engine.GetMWEventDescription((ENUM_MW_EVENT)idx);
      name=(idx==MARKET_WATCH_EVENT_SYMBOL_SORT ? "" : ": "+sparam);
      Print(TimeMSCtoString(lparam)," ",descr,name);
     }
  }
//+------------------------------------------------------------------+
```

All actions related to defining events feature the necessary code comments making them easy to understand. The blocks for handling events
from different classes were swapped (simply for maintaining order) and separated by the 'if - else' conditions. This is important because
we have several event types: symbol and accounts events are handled the same way (using collection IDs), while trading and Market Watch
window events are handled by a value of the events' enumerations. To avoid conflicts between different event definition methods, the
blocks were divided by the conditional statements.

The full EA listing is provided in the files attached below.

Compile the EA, set zero values in the tester settings for the **StopLoss in points** and **TakeProfit in points**
parameters. For the **Mode of used symbols list** parameter, select "Work only with the current symbol" and launch the **M15 Last**
**month** visual EA test:

![](https://c.mql5.com/2/37/KrbEgtVtav.gif)

Before launching the test, we can see that the journal features the specified values for tracked symbol and account properties. During the
visual testing, the messages about obtained events from the properties whose changes we are tracking are displayed in the journal. If the
increase in the funds exceeds the controlled value, profitable positions are closed.

Thus, we have created the base object for all library objects, which provides its descendants with the event functionality and the methods for
setting and receiving tracking parameters for any properties of any object at any time.

Going forward, this will greatly simplify the development of new classes for new objects.

### What's next?

In the next article, we will implement the class of library messages including both internal (messages from library methods) and external
(error and other messages from the terminal) ones.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7149#node00)

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

[Part 17. Interactivity of library objects](https://www.mql5.com/en/articles/7124)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7149](https://www.mql5.com/ru/articles/7149)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7149.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7149/mql5.zip "Download MQL5.zip")(210.04 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7149/mql4.zip "Download MQL4.zip")(210.03 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/326549)**
(5)


![Shuai Fu](https://c.mql5.com/avatar/2019/11/5DD41289-A104.jpg)

**[Shuai Fu](https://www.mql5.com/en/users/baocangxiaowangzi)**
\|
2 Dec 2019 at 04:06

I hope you can reply me: why can't I [insert pictures](https://www.mql5.com/en/articles/24#insert-image "Article: MQL5.community - User Memo ") when I edit documents, user links, youtube videos, tables, code, the only thing missing in the middle is a function to insert pictures ,,, why?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
2 Dec 2019 at 04:30

**Shuai Fu :**

Please write your question in English. I hope you can reply me: Why can't I insert pictures when editing documents, user links, youtube videos, tables, codes, and the only thing missing is the function of inserting pictures, why?

Please write your question in English.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
15 Oct 2020 at 21:01

Hello Artyom - I need a suggestion for overcoming an apparent limitation in DoEasy library... I am looking at the mechanism that allows to trigger events upon price reaching a specific level with e.g. **CSymbol::SetControlBidLevel()**, or increasing/decreasing by a certain number of points with e.g. **CSymbol::SetControlBidInc()** and **CSymbol::SetControlBidDec()** \-\- it seems to me that, at any given point in time, I can only set one bid price level (or increase/decrease) per symbol. Is my understanding correct here?

If I have a multi-symbol, multi-timeframe EA that needs to [control](https://www.mql5.com/en/articles/310 "Article: Custom Graphic Controls Part 1. Creating a simple control") for events at multiple values (for level/increase/decrease) **for the same symbol** i.e. because of different timeframes, is there a simple and elegant way to do it with this library?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
16 Oct 2020 at 01:51

**Dima Diall :**

Hello Artyom - I need a suggestion for overcoming an apparent limitation in DoEasy library... I am looking at the mechanism that allows to trigger events upon price reaching a specific level with e.g. **CSymbol::SetControlBidLevel()** ,  or increasing/decreasing by a certain number of points with e.g. **CSymbol::SetControlBidInc()** and **CSymbol::SetControlBidDec()**  \-\- it seems to me that, at any given point in time, I can only set one bid price level (or increase/decrease) per symbol. Is my understanding correct here?

If I have a multi-symbol, multi-timeframe EA that needs to control for events at multiple values (for level/increase/decrease) **for the same symbol** i.e. because of different timeframes, is there a simple and elegant way to do it with this library?

Thanks for the suggestion. I'll see how this can be implemented.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
16 Oct 2020 at 14:04

**Artyom Trishkin:**

Thanks for the suggestion. I'll see how this can be implemented.

Great! Can you help me to think of a workaround I could try for now with the current DoEasy implementation?

![Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages](https://c.mql5.com/2/37/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages](https://www.mql5.com/en/articles/7176)

In this article, we will consider the class of displaying text messages. Currently, we have a sufficient number of different text messages. It is time to re-arrange the methods of their storage, display and translation of Russian or English messages to other languages. Besides, it would be good to introduce convenient ways of adding new languages to the library and quickly switching between them.

![Library for easy and quick development of MetaTrader programs (part XVII): Interactivity of library objects](https://c.mql5.com/2/36/MQL5-avatar-doeasy__12.png)[Library for easy and quick development of MetaTrader programs (part XVII): Interactivity of library objects](https://www.mql5.com/en/articles/7124)

In this article, we are going to finish the development of the base object of all library objects, so that any library object based on it is able to interact with a user. For example, users will be able to set the maximum acceptable size of a spread for opening a position and a price level, upon reaching which an event from a symbol object is sent to the program with the spread or price level-based signal.

![Building an Expert Advisor using separate modules](https://c.mql5.com/2/37/mql5_avatar_adviser_modules.png)[Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)

When developing indicators, Expert Advisors and scripts, developers often need to create various pieces of code, which are not directly related to the trading strategy. In this article, we consider a way to create Expert Advisors using earlier created blocks, such as trailing, filtering and scheduling code, among others. We will see the benefits of this programming approach.

![Strategy builder based on Merrill patterns](https://c.mql5.com/2/37/Article_Logo.png)[Strategy builder based on Merrill patterns](https://www.mql5.com/en/articles/7218)

In the previous article, we considered application of Merrill patterns to various data, such as to a price value on a currency symbol chart and values of standard MetaTrader 5 indicators: ATR, WPR, CCI, RSI, among others. Now, let us try to create a strategy construction set based on Merrill patterns.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/7149&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070457902596036146)

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