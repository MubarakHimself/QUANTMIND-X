---
title: Prices in DoEasy library (part 63): Depth of Market and its abstract request class
url: https://www.mql5.com/en/articles/9010
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:50:36.617857
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=vfexxchsstqsaamlearolgvupxtudwth&ssn=1769251835093689665&ssn_dr=0&ssn_sr=0&fv_date=1769251835&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9010&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Prices%20in%20DoEasy%20library%20(part%2063)%3A%20Depth%20of%20Market%20and%20its%20abstract%20request%20class%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925183507481318&fz_uniq=5083151682734396922&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/9010#node01)
- [Class of the abstract order object in the Depth of Market](https://www.mql5.com/en/articles/9010#node02)
- [Descendant classes of the abstract order object](https://www.mql5.com/en/articles/9010#node03)
- [Test](https://www.mql5.com/en/articles/9010#node04)
- [What's next?](https://www.mql5.com/en/articles/9010#node05)


### Concept

In this article, I will start implementing the functionality for working with the Depth of Market (DOM). Conceptually, classes for working with DOM will not differ from all previously implemented library classes. At the same time, we will have a mold of DOM featuring data about orders stored in DOM. The data is obtained by the [MarketBookGet()](https://www.mql5.com/en/docs/marketinformation/marketbookget) function when the [OnBookEvent()](https://www.mql5.com/en/docs/event_handlers/onbookevent) handler is activated. In case of any change in DOM, an event is activated for each of the symbols in the handler having the active subscription to DOM events.

Thus, the DOM class structure is to be as follows:

1. DOM order object class — the object describing data of one order out of multiple orders obtained from DOM when OnBookEvent() handler is triggered for one symbol;
2. DOM mold object class — the object describing data on all orders obtained from DOM simultaneously at a single activation of the OnBookEvent() handler for a single symbol — p1 set of objects making up the current DOM mold;
3. Timeseries class consisting of the p2 object sequence entered into the timeseries list at each OnBookEvent() activation for a single symbol;

4. Timeseries collection class of DOM data of all used symbols with enabled subscription to DOM events.

Today I will implement the order object class (1) and test obtaining DOM data when OnBookEvent() is activated for the current symbol.

The properties of each order are set in the [MqlBookInfo](https://www.mql5.com/en/docs/constants/structures/mqlbookinfo) structure providing data in DOM:

- order type from the [ENUM\_BOOK\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/enum_book_type) enumeration
- order price

- order volume

- extended accuracy order volume


DOM may feature four order types (from the [ENUM\_BOOK\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/enum_book_type) enumeration):

- Sell order
- Sell order by Market
- Buy order
- Buy order by Market

As we can see, there are four order types — two Buy and two Sell ones. To divide all types of orders into two sides, we should add one more property to the already existing ones — order status indicating its direction — Buy or Sell order. This will allow us to quickly divide all orders into their sides — supply and demand.

The object of a single DOM request will be made similar to order objects (as well as many other library objects) — we will have a basic object of the DOM abstract order and four descendant objects with the order type specification. The concept of constructing such objects was considered at the very beginning of the library development in the [first](https://www.mql5.com/en/articles/5654) and [second](https://www.mql5.com/en/articles/5669) articles.

Before implementing the classes for working with DOM, add new library messages and slightly improve the tick data object classes. Add new message indices in \\MQL5\\Include\\DoEasy\ **Data.mqh**:

```
   MSG_SYM_EVENT_SYMBOL_ADD,                          // Added symbol to Market Watch window
   MSG_SYM_EVENT_SYMBOL_DEL,                          // Symbol removed from Market Watch window
   MSG_SYM_EVENT_SYMBOL_SORT,                         // Changed location of symbols in Market Watch window
   MSG_SYM_SYMBOLS_MODE_CURRENT,                      // Work with current symbol only
   MSG_SYM_SYMBOLS_MODE_DEFINES,                      // Work with predefined symbol list
   MSG_SYM_SYMBOLS_MODE_MARKET_WATCH,                 // Work with Market Watch window symbols
   MSG_SYM_SYMBOLS_MODE_ALL,                          // Work with full list of all available symbols
   MSG_SYM_SYMBOLS_BOOK_ADD,                          // Subscribed to Depth of Market
   MSG_SYM_SYMBOLS_BOOK_DEL,                          // Unsubscribed from Depth of Market
   MSG_SYM_SYMBOLS_MODE_BOOK,                         // Subscription to Depth of Market
   MSG_SYM_SYMBOLS_ERR_BOOK_ADD,                      // Error subscribing to DOM
   MSG_SYM_SYMBOLS_ERR_BOOK_DEL,                      // Error unsubscribing from DOM

//--- CAccount
```

...

```
//--- CTickSeries
   MSG_TICKSERIES_TEXT_TICKSERIES,                    // Tick series
   MSG_TICKSERIES_ERR_GET_TICK_DATA,                  // Failed to get tick data
   MSG_TICKSERIES_FAILED_CREATE_TICK_DATA_OBJ,        // Failed to create tick data object
   MSG_TICKSERIES_FAILED_ADD_TO_LIST,                 // Failed to add tick data object to list
   MSG_TICKSERIES_TEXT_IS_NOT_USE,                    // Tick series not used. Set the flag using SetAvailable()
   MSG_TICKSERIES_REQUIRED_HISTORY_DAYS,              // Requested number of days

//--- CMarketBookOrd
   MSG_MBOOK_ORD_TEXT_MBOOK_ORD,                      // Order in DOM
   MSG_MBOOK_ORD_VOLUME,                              // Volume
   MSG_MBOOK_ORD_VOLUME_REAL,                         // Extended accuracy volume
   MSG_MBOOK_ORD_STATUS_BUY,                          // Buy side
   MSG_MBOOK_ORD_STATUS_SELL,                         // Sell side
   MSG_MBOOK_ORD_TYPE_SELL,                           // Sell order
   MSG_MBOOK_ORD_TYPE_BUY,                            // Buy order
   MSG_MBOOK_ORD_TYPE_SELL_MARKET,                    // Sell order by Market
   MSG_MBOOK_ORD_TYPE_BUY_MARKET,                     // Buy order by Market

  };
//+------------------------------------------------------------------+
```

and message texts corresponding to newly added indices:

```
   {"В окно \"Обзор рынка\" добавлен символ","Added symbol to \"Market Watch\" window"},
   {"Из окна \"Обзор рынка\" удалён символ","Removed from \"Market Watch\" window"},
   {"Изменено расположение символов в окне \"Обзор рынка\"","Changed arrangement of symbols in \"Market Watch\" window"},
   {"Работа только с текущим символом","Work only with the current symbol"},
   {"Работа с предопределённым списком символов","Work with predefined list of symbols"},
   {"Работа с символами из окна \"Обзор рынка\"","Working with symbols from \"Market Watch\" window"},
   {"Работа с полным списком всех доступных символов","Work with full list of all available symbols"},
   {"Осуществлена подписка на стакан цен ","Subscribed to Depth of Market"},
   {"Осуществлена отписка от стакан цен ","Unsubscribed from Depth of Market"},
   {"Подписка на стакан цен","Subscription to Depth of Market"},
   {"Ошибка при подписке на стакан цен",""},
   {"Ошибка при отписке от стакан цен",""},

//--- CAccount
```

...

```
//--- CMarketBookOrd
   {"Заявка в стакане цен","Order in Depth of Market"},
   {"Объем","Volume"},
   {"Объем c повышенной точностью","Volume Real"},
   {"Сторона Buy","Buy side"},
   {"Сторона Sell","Sell side"},
   {"Заявка на продажу","Sell order"},
   {"Заявка на покупку","Buy order"},
   {"Заявка на продажу по рыночной цене","Sell order at market price"},
   {"Заявка на покупку по рыночной цене","Buy order at market price"},

  };
//+---------------------------------------------------------------------+
```

Add displaying messages about the error when subscribing to DOM in \\MQL5\\Include\\DoEasy\\Objects\\Symbols\ **Symbol.mqh** file of the symbol object class:

```
//+------------------------------------------------------------------+
//| Subscribe to the Depth of Market                                 |
//+------------------------------------------------------------------+
bool CSymbol::BookAdd(void)
  {
   this.m_book_subscribed=(#ifdef __MQL5__ ::MarketBookAdd(this.m_name) #else false #endif);
   this.m_long_prop[SYMBOL_PROP_BOOKDEPTH_STATE]=this.m_book_subscribed;
   if(this.m_book_subscribed)
      ::Print(CMessage::Text(MSG_SYM_SYMBOLS_BOOK_ADD)+" "+this.m_name);
   else
      ::Print(CMessage::Text(MSG_SYM_SYMBOLS_ERR_BOOK_ADD)+": "+CMessage::Text(::GetLastError()));
   return this.m_book_subscribed;
  }
//+------------------------------------------------------------------+
```

and do the same when unsubscribing from it:

```
//+------------------------------------------------------------------+
//| Close the market depth                                           |
//+------------------------------------------------------------------+
bool CSymbol::BookClose(void)
  {
//--- If the DOM subscription flag is off, subscription is disabled (or not enabled yet). Return 'true'
   if(!this.m_book_subscribed)
      return true;
//--- Save the result of unsubscribing from the DOM
   bool res=( #ifdef __MQL5__ ::MarketBookRelease(this.m_name) #else true #endif );
//--- If unsubscribed successfully, reset the DOM subscription flag and write the status to the object property
   if(res)
     {
      this.m_long_prop[SYMBOL_PROP_BOOKDEPTH_STATE]=this.m_book_subscribed=false;
      ::Print(CMessage::Text(MSG_SYM_SYMBOLS_BOOK_DEL)+" "+this.m_name);
     }
   else
     {
      this.m_long_prop[SYMBOL_PROP_BOOKDEPTH_STATE]=this.m_book_subscribed=true;
      ::Print(CMessage::Text(MSG_SYM_SYMBOLS_ERR_BOOK_DEL)+": "+CMessage::Text(::GetLastError()));
     }
//--- Return the result of unsubscribing from DOM
   return res;
  }
//+------------------------------------------------------------------+
```

From the tick series update method of the tick series class in \\MQL5\\Include\\DoEasy\\Objects\\Ticks\ **TickSeries.mqh**, remove displaying debugging comments on the symbol chart we left for tests [in the previous article](https://www.mql5.com/en/articles/8988):

```
//+------------------------------------------------------------------+
//| Update the tick series list                                      |
//+------------------------------------------------------------------+
void CTickSeries::Refresh(void)
  {
   MqlTick ticks_array[];
   if(IsNewTick())
     {
      //--- Copy ticks from m_last_time time+1 ms to the end of history
      int err=ERR_SUCCESS;
      int total=::CopyTicksRange(this.Symbol(),ticks_array,COPY_TICKS_ALL,this.m_last_time+1,0);
      //--- If the ticks have been copied, create new tick data objects and add them to the list in the loop by their number
      if(total>0)
        {
         for(int i=0;i<total;i++)
           {
            //--- Create the tick object and add it to the list
            CDataTick *tick_obj=this.CreateNewTickObj(ticks_array[i]);
            if(tick_obj==NULL)
               break;
            //--- Write the last tick time for subsequent copying of newly arrived ticks
            long end_time=ticks_array[::ArraySize(ticks_array)-1].time_msc;
            if(this.Symbol()=="AUDUSD")
               Comment(DFUN,this.Symbol(),", copied=",total,", m_last_time=",TimeMSCtoString(m_last_time),", end_time=",TimeMSCtoString(end_time),", total=",DataTotal());
            this.m_last_time=end_time;
           }
         //--- If the number of ticks in the list exceeds the default maximum number,
         //--- remove the calculated number of tick objects from the end of the list
         if(this.DataTotal()>TICKSERIES_MAX_DATA_TOTAL)
           {
            int total_del=m_list_ticks.Total()-TICKSERIES_MAX_DATA_TOTAL;
            for(int j=0;j<total_del;j++)
               this.m_list_ticks.Delete(j);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The last tick time is now immediately set in the **m\_last\_time** variable meant for this purpose because I needed to display verification data as a symbol chart comment featuring the previous and current tick times in the previous article. Now we do not need it and the time is immediately saved in the variable:

```
//+------------------------------------------------------------------+
//| Update the tick series list                                      |
//+------------------------------------------------------------------+
void CTickSeries::Refresh(void)
  {
   MqlTick ticks_array[];
   if(IsNewTick())
     {
      //--- Copy ticks from m_last_time time+1 ms to the end of history
      int err=ERR_SUCCESS;
      int total=::CopyTicksRange(this.Symbol(),ticks_array,COPY_TICKS_ALL,this.m_last_time+1,0);
      //--- If the ticks have been copied, create new tick data objects and add them to the list in the loop by their number
      if(total>0)
        {
         for(int i=0;i<total;i++)
           {
            //--- Create the tick object and add it to the list
            CDataTick *tick_obj=this.CreateNewTickObj(ticks_array[i]);
            if(tick_obj==NULL)
               break;
            //--- Write the last tick time for subsequent copying of newly arrived ticks
            this.m_last_time=ticks_array[::ArraySize(ticks_array)-1].time_msc;
           }
         //--- If the number of ticks in the list exceeds the default maximum number,
         //--- remove the calculated number of tick objects from the end of the list
         if(this.DataTotal()>TICKSERIES_MAX_DATA_TOTAL)
           {
            int total_del=m_list_ticks.Total()-TICKSERIES_MAX_DATA_TOTAL;
            for(int j=0;j<total_del;j++)
               this.m_list_ticks.Delete(j);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

### Class of the abstract order object in the Depth of Market

Like with all library objects having enumeration sets for defining object property constants, we need to create enumerations of integer, real and string object properties for DOM orders as well.

Add enumerations of DOM order object properties and parameters in \\MQL5\\Include\\DoEasy\ **Defines.mqh**. Since I am not going to implement an event model of working with each of orders in DOM (at one moment in time the order book displays the current state of all orders, and their change leads to its next state and is processed at the next activation of OnBookEvent()), simply add the constant specifying the code of the next event after the last code of the DOM event simply to maintain the identity of the constants of all objects to bring them to the same form:

```
//+------------------------------------------------------------------+
//| Data for working with DOM                                        |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of possible DOM events                                      |
//+------------------------------------------------------------------+
#define MBOOK_ORD_EVENTS_NEXT_CODE  (SERIES_EVENTS_NEXT_CODE+1)   // The code of the next event after the last DOM event code
//+------------------------------------------------------------------+
```

Define the enumeration featuring two possible states of a single DOM order — Buy or Sell side:

```
//+------------------------------------------------------------------+
//| Abstract DOM type (status)                                       |
//+------------------------------------------------------------------+
enum ENUM_MBOOK_ORD_STATUS
  {
   MBOOK_ORD_STATUS_BUY,                              // Buy side
   MBOOK_ORD_STATUS_SELL,                             // Sell side
  };
//+------------------------------------------------------------------+
```

Sorting out all orders in DOM by these properties allows us to quickly select all orders in DOM belonging either to demand or to supply by these properties.

Next, add enumerations of the integer, real and string properties of DOM order object properties:

```
//+------------------------------------------------------------------+
//| Integer properties of DOM order                                  |
//+------------------------------------------------------------------+
enum ENUM_MBOOK_ORD_PROP_INTEGER
  {
   MBOOK_ORD_PROP_STATUS = 0,                         // Order status
   MBOOK_ORD_PROP_TYPE,                               // Order type
   MBOOK_ORD_PROP_VOLUME,                             // Order volume
  };
#define MBOOK_ORD_PROP_INTEGER_TOTAL (3)              // Total number of integer properties
#define MBOOK_ORD_PROP_INTEGER_SKIP  (0)              // Number of integer DOM properties not used in sorting
//+------------------------------------------------------------------+
//| Real properties of DOM order                                     |
//+------------------------------------------------------------------+
enum ENUM_MBOOK_ORD_PROP_DOUBLE
  {
   MBOOK_ORD_PROP_PRICE = MBOOK_ORD_PROP_INTEGER_TOTAL, // Order price
   MBOOK_ORD_PROP_VOLUME_REAL,                        // Extended accuracy order volume
  };
#define MBOOK_ORD_PROP_DOUBLE_TOTAL  (2)              // Total number of real properties
#define MBOOK_ORD_PROP_DOUBLE_SKIP   (0)              // Number of real properties not used in sorting
//+------------------------------------------------------------------+
//| String properties of DOM order                                   |
//+------------------------------------------------------------------+
enum ENUM_MBOOK_ORD_PROP_STRING
  {
   MBOOK_ORD_PROP_SYMBOL = (MBOOK_ORD_PROP_INTEGER_TOTAL+MBOOK_ORD_PROP_DOUBLE_TOTAL), // Order symbol name
  };
#define MBOOK_ORD_PROP_STRING_TOTAL  (1)              // Total number of string properties
//+------------------------------------------------------------------+
```

Let's implement the enumeration of possible criteria of sorting orders in DOM according to created properties:

```
//+------------------------------------------------------------------+
//| Possible sorting criteria of DOM orders                          |
//+------------------------------------------------------------------+
#define FIRST_MB_DBL_PROP  (MBOOK_ORD_PROP_INTEGER_TOTAL-MBOOK_ORD_PROP_INTEGER_SKIP)
#define FIRST_MB_STR_PROP  (MBOOK_ORD_PROP_INTEGER_TOTAL-MBOOK_ORD_PROP_INTEGER_SKIP+MBOOK_ORD_PROP_DOUBLE_TOTAL-MBOOK_ORD_PROP_DOUBLE_SKIP)
enum ENUM_SORT_MBOOK_ORD_MODE
  {
//--- Sort by integer properties
   SORT_BY_MBOOK_ORD_STATUS = 0,                      // Sort by order status
   SORT_BY_MBOOK_ORD_TYPE,                            // Sort by order type
   SORT_BY_MBOOK_ORD_VOLUME,                          // Sort by order volume
//--- Sort by real properties
   SORT_BY_MBOOK_ORD_PRICE = FIRST_MB_DBL_PROP,       // Sort by order price
   SORT_BY_MBOOK_ORD_VOLUME_REAL,                     // Sort by extended accuracy order volume
//--- Sort by string properties
   SORT_BY_MBOOK_ORD_SYMBOL = FIRST_MB_STR_PROP,      // Sort by symbol name
  };
//+------------------------------------------------------------------+
```

**Now it is possible to create the abstract order object class in DOM.**

In \\MQL5\\Include\\DoEasy\ **Objects\**, create the new **Book\** folder containing the **MarketBookOrd.mqh** file of the CMarketBookOrd class inherited from the basic object of all CBaseObj library objects:

```
//+------------------------------------------------------------------+
//|                                                MarketBookOrd.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\DELib.mqh"
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| DOM abstract order class                                         |
//+------------------------------------------------------------------+
class CMarketBookOrd : public CBaseObj
  {
private:
   int               m_digits;                                       // Number of decimal places
   long              m_long_prop[MBOOK_ORD_PROP_INTEGER_TOTAL];      // Integer properties
   double            m_double_prop[MBOOK_ORD_PROP_DOUBLE_TOTAL];     // Real properties
   string            m_string_prop[MBOOK_ORD_PROP_STRING_TOTAL];     // String properties

//--- Return the index of the array the (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_MBOOK_ORD_PROP_DOUBLE property)  const { return(int)property-MBOOK_ORD_PROP_INTEGER_TOTAL;                              }
   int               IndexProp(ENUM_MBOOK_ORD_PROP_STRING property)  const { return(int)property-MBOOK_ORD_PROP_INTEGER_TOTAL-MBOOK_ORD_PROP_DOUBLE_TOTAL;  }

public:
//--- Set object's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_MBOOK_ORD_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                      }
   void              SetProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;    }
   void              SetProperty(ENUM_MBOOK_ORD_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;    }
//--- Return object’s (1) integer, (2) real and (3) string property from the properties array
   long              GetProperty(ENUM_MBOOK_ORD_PROP_INTEGER property)        const { return this.m_long_prop[property];                     }
   double            GetProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];   }
   string            GetProperty(ENUM_MBOOK_ORD_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];   }
//--- Return itself
   CMarketBookOrd   *GetObject(void)                                                { return &this;}

//--- Return the flag of the object supporting this property
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property)          { return true; }
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property)           { return true; }
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_STRING property)           { return true; }

//--- Get description of (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_MBOOK_ORD_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_MBOOK_ORD_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_MBOOK_ORD_PROP_STRING property);

//--- Display the description of object properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short description of the object in the journal
   virtual void      PrintShort(void);
//--- Return the object short name
   virtual string    Header(void);

//--- Compare CMarketBookOrd objects by all possible properties (to sort the lists by a specified order object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CMarketBookOrd objects by all properties (to search for equal request objects)
   bool              IsEqual(CMarketBookOrd* compared_req) const;

//--- Default constructor
                     CMarketBookOrd(){;}
protected:
//--- Protected parametric constructor
                     CMarketBookOrd(const ENUM_MBOOK_ORD_STATUS status,const MqlBookInfo &book_info,const string symbol);

public:
//+-------------------------------------------------------------------+
//|Methods of a simplified access to the DOM request object properties|
//+-------------------------------------------------------------------+
//--- Return order (1) status, (2) type and (3) order volume
   ENUM_MBOOK_ORD_STATUS Status(void)        const { return (ENUM_MBOOK_ORD_STATUS)this.GetProperty(MBOOK_ORD_PROP_STATUS);   }
   ENUM_BOOK_TYPE    TypeOrd(void)           const { return (ENUM_BOOK_TYPE)this.GetProperty(MBOOK_ORD_PROP_TYPE);            }
   long              Volume(void)            const { return this.GetProperty(MBOOK_ORD_PROP_VOLUME);                          }
//--- Return (1) the price and (2) extended accuracy order volume
   double            Price(void)             const { return this.GetProperty(MBOOK_ORD_PROP_PRICE);                           }
   double            VolumeReal(void)        const { return this.GetProperty(MBOOK_ORD_PROP_VOLUME_REAL);                     }
//--- Return (1) order symbol and (2) symbol's Digits
   string            Symbol(void)            const { return this.GetProperty(MBOOK_ORD_PROP_SYMBOL);                          }
   int               Digits()                const { return this.m_digits;                                                    }

//--- Return the description of order  (1) type (ENUM_BOOK_TYPE) and (2) status (ENUM_MBOOK_ORD_STATUS)
   virtual string    TypeDescription(void)   const { return this.StatusDescription();                                         }
   string            StatusDescription(void) const;

  };
//+------------------------------------------------------------------+
```

The composition of the class is absolutely identical to other classes of library objects. I mentioned them quite often. You can find the detailed descriptions in the [first and subsequent articles](https://www.mql5.com/en/articles/5654).

**Let's have a look at the implementation of the class methods.**

**In the closed class parametric constructor,** set all object properties from the order structure passed from DOM to the constructor:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CMarketBookOrd::CMarketBookOrd(const ENUM_MBOOK_ORD_STATUS status,const MqlBookInfo &book_info,const string symbol)
  {
//--- Save symbol’s Digits
   this.m_digits=(int)::SymbolInfoInteger(symbol,SYMBOL_DIGITS);
//--- Save integer object properties
   this.SetProperty(MBOOK_ORD_PROP_STATUS,status);
   this.SetProperty(MBOOK_ORD_PROP_TYPE,book_info.type);
   this.SetProperty(MBOOK_ORD_PROP_VOLUME,book_info.volume);
//--- Save real object properties
   this.SetProperty(MBOOK_ORD_PROP_PRICE,book_info.price);
   this.SetProperty(MBOOK_ORD_PROP_VOLUME_REAL,book_info.volume_real);
//--- Save additional object properties
   this.SetProperty(MBOOK_ORD_PROP_SYMBOL,(symbol==NULL || symbol=="" ? ::Symbol() : symbol));
  }
//+------------------------------------------------------------------+
```

The constructor also receives the order status specified in descendant objects of the class when creating a new DOM order object.

**The method of comparing two CMarketBookOrd objects by a specified property** for defining the equality of the specified properties of two objects:

```
//+------------------------------------------------------------------+
//| Compare CMarketBookOrd objects                                   |
//| by a specified property                                          |
//+------------------------------------------------------------------+
int CMarketBookOrd::Compare(const CObject *node,const int mode=0) const
  {
   const CMarketBookOrd *obj_compared=node;
//--- compare integer properties of two objects
   if(mode<MBOOK_ORD_PROP_INTEGER_TOTAL)
     {
      long value_compared=obj_compared.GetProperty((ENUM_MBOOK_ORD_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_MBOOK_ORD_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two objects
   else if(mode<MBOOK_ORD_PROP_DOUBLE_TOTAL+MBOOK_ORD_PROP_INTEGER_TOTAL)
     {
      double value_compared=obj_compared.GetProperty((ENUM_MBOOK_ORD_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_MBOOK_ORD_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<MBOOK_ORD_PROP_DOUBLE_TOTAL+MBOOK_ORD_PROP_INTEGER_TOTAL+MBOOK_ORD_PROP_STRING_TOTAL)
     {
      string value_compared=obj_compared.GetProperty((ENUM_MBOOK_ORD_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_MBOOK_ORD_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

The method receives the object whose property should be compared with the same property of the current object. If the specified property value of the compared object is lower than the current one, -1 is returned, if larger — +1, if the properties are equal — 0.

**The method for comparing two CMarketBookOrd objects by all properties**. It allows determining the complete identity of two compared objects:

```
//+------------------------------------------------------------------+
//| Compare CMarketBookOrd objects by all properties                 |
//+------------------------------------------------------------------+
bool CMarketBookOrd::IsEqual(CMarketBookOrd *compared_obj) const
  {
   int beg=0, end=MBOOK_ORD_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_MBOOK_ORD_PROP_INTEGER prop=(ENUM_MBOOK_ORD_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=MBOOK_ORD_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_MBOOK_ORD_PROP_DOUBLE prop=(ENUM_MBOOK_ORD_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=MBOOK_ORD_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_MBOOK_ORD_PROP_STRING prop=(ENUM_MBOOK_ORD_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

Each subsequent property of two objects is compared one by one here. If the objects are not equal, false is returned. After checking the equality of all properties of two objects is complete and no false is obtained, return true  — both objects are completely identical.

**The method displaying all object properties in the journal:**

```
//+------------------------------------------------------------------+
//| Display object properties in the journal                         |
//+------------------------------------------------------------------+
void CMarketBookOrd::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG)," (",this.Header(),") =============");
   int beg=0, end=MBOOK_ORD_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_MBOOK_ORD_PROP_INTEGER prop=(ENUM_MBOOK_ORD_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=MBOOK_ORD_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_MBOOK_ORD_PROP_DOUBLE prop=(ENUM_MBOOK_ORD_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=MBOOK_ORD_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_MBOOK_ORD_PROP_STRING prop=(ENUM_MBOOK_ORD_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_END)," (",this.Header(),") =============\n");
  }
//+------------------------------------------------------------------+
```

String descriptions of each subsequent property are displayed in three loops by integer, real and string object properties.

**The methods returning the descriptions of the specified integer, real and string object properties:**

```
//+------------------------------------------------------------------+
//| Return description of object's integer property                  |
//+------------------------------------------------------------------+
string CMarketBookOrd::GetPropertyDescription(ENUM_MBOOK_ORD_PROP_INTEGER property)
  {
   return
     (
      property==MBOOK_ORD_PROP_STATUS        ?  CMessage::Text(MSG_ORD_STATUS)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.StatusDescription()
         )  :
      property==MBOOK_ORD_PROP_TYPE          ?  CMessage::Text(MSG_ORD_TYPE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.TypeDescription()
         )  :
      property==MBOOK_ORD_PROP_VOLUME        ?  CMessage::Text(MSG_MBOOK_ORD_VOLUME)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of object's real property                     |
//+------------------------------------------------------------------+
string CMarketBookOrd::GetPropertyDescription(ENUM_MBOOK_ORD_PROP_DOUBLE property)
  {
   int dg=(this.m_digits>0 ? this.m_digits : 1);
   return
     (
      property==MBOOK_ORD_PROP_PRICE         ?  CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==MBOOK_ORD_PROP_VOLUME_REAL   ?  CMessage::Text(MSG_MBOOK_ORD_VOLUME_REAL)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return description of object's string property                   |
//+------------------------------------------------------------------+
string CMarketBookOrd::GetPropertyDescription(ENUM_MBOOK_ORD_PROP_STRING property)
  {
   return(property==MBOOK_ORD_PROP_SYMBOL ? CMessage::Text(MSG_LIB_PROP_SYMBOL)+": \""+this.GetProperty(property)+"\"" : "");
  }
//+------------------------------------------------------------------+
```

Each of the methods receives the property whose description should be returned. Depending on the property passed to the method, a string to be eventually returned from the method is created.

**The method returning a short object name:**

```
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookOrd::Header(void)
  {
   return this.TypeDescription()+" \""+this.Symbol()+"\"";
  }
//+------------------------------------------------------------------+
```

The method returns the string consisting of the description of an order type and its symbol.

**The method displaying the short object description in the journal:**

```
//+------------------------------------------------------------------+
//| Display a short description of the object in the journal         |
//+------------------------------------------------------------------+
void CMarketBookOrd::PrintShort(void)
  {
   ::Print(this.Header());
  }
//+------------------------------------------------------------------+
```

The method simply displays the string created by the previous method in the journal.

**The method returning the order status description in DOM:**

```
//+------------------------------------------------------------------+
//| Return the order status description in DOM                       |
//+------------------------------------------------------------------+
string CMarketBookOrd::StatusDescription(void) const
  {
   return
     (
      Status()==MBOOK_ORD_STATUS_SELL  ?  CMessage::Text(MSG_MBOOK_ORD_STATUS_SELL) :
      Status()==MBOOK_ORD_STATUS_BUY   ?  CMessage::Text(MSG_MBOOK_ORD_STATUS_BUY)  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Depending on the order "status", the string with the status description is returned.

**This is the entire order object class in DOM.**

Now we need to create four classes that inherit from this abstract order object. The descendant classes will be used to create new order objects from DOM. The status of the created order object will be specified in the initialization list of the descendant class constructor depending on the order type.

### Descendant classes of the abstract order object

In \\MQL5\\Include\\DoEasy\\Objects\ **Book\**, create the **MarketBookBuy.mqh** file of the CMarketBookBuy class. The newly created CMarketBookOrd abstract order class is to be a parent class:

```
//+------------------------------------------------------------------+
//|                                                MarketBookBuy.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "MarketBookOrd.mqh"
//+------------------------------------------------------------------+
//| Buy order in DOM                                                 |
//+------------------------------------------------------------------+
class CMarketBookBuy : public CMarketBookOrd
  {
private:

public:
   //--- Constructor
                     CMarketBookBuy(const string symbol,const MqlBookInfo &book_info) :
                        CMarketBookOrd(MBOOK_ORD_STATUS_BUY,book_info,symbol) {}
   //--- Supported order properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property);
//--- Return the object short name
   virtual string    Header(void);
//--- Return the description of order type (ENUM_BOOK_TYPE)
   virtual string    TypeDescription(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CMarketBookBuy::SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CMarketBookBuy::SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookBuy::Header(void)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_BUY)+" \""+this.Symbol()+
          "\": "+::DoubleToString(this.Price(),this.Digits())+" ["+::DoubleToString(this.VolumeReal(),2)+"]";
  }
//+------------------------------------------------------------------+
//| Return the description of order type                             |
//+------------------------------------------------------------------+
string CMarketBookBuy::TypeDescription(void)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_BUY);
  }
//+------------------------------------------------------------------+
```

When creating a new DOM order object, set the "Buy side" in the parent class constructor.

In the virtual methods returning the flag of supporting the integer and real properties, return true  — each property is supported by the object.

In the virtual method returning the short name of the DOM order object, return the string in the following format

```
Type "Symbol": Price [VolumeReal]
```

For example:

```
"EURUSD" buy order: 1.20123 [10.00]
```

In the virtual method returning the description of the DOM order object type, return the "Buy order" string.

The remaining three classes inherited from the DOM abstract order base class are identical to the considered one except for the order status. Each class constructor features its status corresponding to the described order object and its virtual methods returning the strings corresponding to the type of the DOM order described by each of the objects. All these classes are located in the same folder as the one described above. I will show their listings here allowing you to analyze and compare their virtual methods.

**MarketBookBuyMarket.mqh:**

```
//+------------------------------------------------------------------+
//|                                          MarketBookBuyMarket.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "MarketBookOrd.mqh"
//+------------------------------------------------------------------+
//| Buy order by Market in DOM                                       |
//+------------------------------------------------------------------+
class CMarketBookBuyMarket : public CMarketBookOrd
  {
private:

public:
   //--- Constructor
                     CMarketBookBuyMarket(const string symbol,const MqlBookInfo &book_info) :
                        CMarketBookOrd(MBOOK_ORD_STATUS_BUY,book_info,symbol) {}
   //--- Supported order properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property);
//--- Return the object short name
   virtual string    Header(void);
//--- Return the description of order type (ENUM_BOOK_TYPE)
   virtual string    TypeDescription(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CMarketBookBuyMarket::SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CMarketBookBuyMarket::SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookBuyMarket::Header(void)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_BUY_MARKET)+" \""+this.Symbol()+
          "\": "+::DoubleToString(this.Price(),this.Digits())+" ["+::DoubleToString(this.VolumeReal(),2)+"]";
  }
//+------------------------------------------------------------------+
//| Return the description of order type                             |
//+------------------------------------------------------------------+
string CMarketBookBuyMarket::TypeDescription(void)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_BUY_MARKET);
  }
//+------------------------------------------------------------------+
```

**MarketBookSell.mqh:**

```
//+------------------------------------------------------------------+
//|                                               MarketBookSell.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "MarketBookOrd.mqh"
//+------------------------------------------------------------------+
//| Sell order in DOM                                                |
//+------------------------------------------------------------------+
class CMarketBookSell : public CMarketBookOrd
  {
private:

public:
   //--- Constructor
                     CMarketBookSell(const string symbol,const MqlBookInfo &book_info) :
                        CMarketBookOrd(MBOOK_ORD_STATUS_SELL,book_info,symbol) {}
   //--- Supported order properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property);
//--- Return the object short name
   virtual string    Header(void);
//--- Return the description of order type (ENUM_BOOK_TYPE)
   virtual string    TypeDescription(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CMarketBookSell::SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CMarketBookSell::SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookSell::Header(void)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_SELL)+" \""+this.Symbol()+
          "\": "+::DoubleToString(this.Price(),this.Digits())+" ["+::DoubleToString(this.VolumeReal(),2)+"]";
  }
//+------------------------------------------------------------------+
//| Return the description of order type                             |
//+------------------------------------------------------------------+
string CMarketBookSell::TypeDescription(void)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_SELL);
  }
//+------------------------------------------------------------------+
```

**MarketBookSellMarket.mqh:**

```
//+------------------------------------------------------------------+
//|                                         MarketBookSellMarket.mqh |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "MarketBookOrd.mqh"
//+------------------------------------------------------------------+
//| Sell order by Market in DOM                                      |
//+------------------------------------------------------------------+
class CMarketBookSellMarket : public CMarketBookOrd
  {
private:

public:
   //--- Constructor
                     CMarketBookSellMarket(const string symbol,const MqlBookInfo &book_info) :
                        CMarketBookOrd(MBOOK_ORD_STATUS_SELL,book_info,symbol) {}
   //--- Supported order properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property);
//--- Return the object short name
   virtual string    Header(void);
//--- Return the description of order type (ENUM_BOOK_TYPE)
   virtual string    TypeDescription(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CMarketBookSellMarket::SupportProperty(ENUM_MBOOK_ORD_PROP_INTEGER property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CMarketBookSellMarket::SupportProperty(ENUM_MBOOK_ORD_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return the object short name                                     |
//+------------------------------------------------------------------+
string CMarketBookSellMarket::Header(void)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_SELL_MARKET)+" \""+this.Symbol()+
          "\": "+::DoubleToString(this.Price(),this.Digits())+" ["+::DoubleToString(this.VolumeReal(),2)+"]";
  }
//+------------------------------------------------------------------+
//| Return the description of order type                             |
//+------------------------------------------------------------------+
string CMarketBookSellMarket::TypeDescription(void)
  {
   return CMessage::Text(MSG_MBOOK_ORD_TYPE_SELL_MARKET);
  }
//+------------------------------------------------------------------+
```

**This is all I wanted to do in the current article.**

### Test

To perform the test, I will use the [EA from the previous article](https://www.mql5.com/en/articles/8988#node05) and save it in \\MQL5\\Experts\\TestDoEasy\ **Part63\** as **TestDoEasyPart63.mq5**.

After launching the EA, we subscribe to DOMs of symbols specified in the settings for work. All events with DOMs are registered in the [OnBookEvent()](https://www.mql5.com/en/docs/event_handlers/onbookevent) handler. Accordingly, in this handler, we make sure that the event has occurred on the current symbol. We also get the DOM snapshot and save all existing orders to the list sorted by price values. Next, display the very first and last orders from the list in the chart comments. Thus, we will display two extreme DOM orders — sell and buy ones. In the journal, display the list of all obtained DOM orders at the very first OnBookEvent() activation.

To enable the EA to see the newly created classes, include them to the EA file (currently, they cannot be accessed from the CEngine library main object):

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart63.mq5 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <DoEasy\Objects\Book\MarketBookBuy.mqh>
#include <DoEasy\Objects\Book\MarketBookSell.mqh>
#include <DoEasy\Objects\Book\MarketBookBuyMarket.mqh>
#include <DoEasy\Objects\Book\MarketBookSellMarket.mqh>
//--- enums
```

Now we need to create the OnBookEvent() handler in the EA and implement handling a DOM event in it:

```
//+------------------------------------------------------------------+
//| OnBookEvent function                                             |
//+------------------------------------------------------------------+
void OnBookEvent(const string& symbol)
  {
   static bool first=true;
   //--- Get a symbol object
   CSymbol *sym=engine.GetSymbolCurrent();
   //--- If failed to get a symbol object or it is not subscribed to DOM, exit
   if(sym==NULL || !sym.BookdepthSubscription()) return;
   //--- create the list for storing DOM order objects
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return;
   //--- Work by the current symbol
   if(symbol==sym.Name())
     {
      //--- Declare the DOM structure array
      MqlBookInfo book_array[];
      //--- Get DOM entries to the structure array
      if(!MarketBookGet(sym.Name(),book_array))
         return;
      //--- clear the list
      list.Clear();
      //--- In the loop by the structure array
      int total=ArraySize(book_array);
      for(int i=0;i<total;i++)
        {
         //--- Create order objects of the current DOM snapshot depending on the order type
         CMarketBookOrd *mbook_ord=NULL;
         switch(book_array[i].type)
           {
            case BOOK_TYPE_BUY         : mbook_ord=new CMarketBookBuy(sym.Name(),book_array[i]);         break;
            case BOOK_TYPE_SELL        : mbook_ord=new CMarketBookSell(sym.Name(),book_array[i]);        break;
            case BOOK_TYPE_BUY_MARKET  : mbook_ord=new CMarketBookBuyMarket(sym.Name(),book_array[i]);   break;
            case BOOK_TYPE_SELL_MARKET : mbook_ord=new CMarketBookSellMarket(sym.Name(),book_array[i]);  break;
            default: break;
           }
         if(mbook_ord==NULL)
            continue;
         //--- Set the sorted list flag for the list (by the price value) and add the current order object to it
         list.Sort(SORT_BY_MBOOK_ORD_PRICE);
         if(!list.InsertSort(mbook_ord))
            delete mbook_ord;
        }
      //--- Get the very first and last DOM order objects from the list
      CMarketBookOrd *ord_0=list.At(0);
      CMarketBookOrd *ord_N=list.At(list.Total()-1);
      if(ord_0==NULL || ord_N==NULL) return;
      //--- Display the size of the current DOM snapshot in the chart comment,
      //--- the maximum number of displayed orders in DOM for a symbol and
      //--- the highest and lowest orders of the current DOM snapshot
      Comment
        (
         DFUN,sym.Name(),": ",TimeMSCtoString(sym.Time()),", array total=",total,", book size=",sym.TicksBookdepth(),", list.Total: ",list.Total(),"\n",
         "Max: ",ord_N.Header(),"\nMin: ",ord_0.Header()
        );
      //--- Display the first DOM snapshot in the journal
      if(first)
        {
         for(int i=list.Total()-1;i>WRONG_VALUE;i--)
           {
            CMarketBookOrd *ord=list.At(i);
            ord.PrintShort();
           }
         first=false;
        }
     }
   //--- Delete the created list
   delete list;
  }
//+------------------------------------------------------------------+
```

The code comments contain all the details. If you have any questions, feel free to ask them in the comments.

Compile the EA and launch it on a symbol chart having preliminarily defined in the settings to use the two specified symbols and the current timeframe.

![](https://c.mql5.com/2/42/ferlYZx9y0.png)

After the EA is launched and the first DOM change event arrives, the parameters of the current DOM snapshot list are displayed in the chart comments together with two orders — the highest Buy and the lowest Sell one:

![](https://c.mql5.com/2/42/terminal64_5xmPhqo6fN.png)

The journal displays the list of all orders of the current DOM snapshot:

```
Subscribed to Depth of Market  AUDUSD
Subscribed to Depth of Market  EURUSD
Library initialization time: 00:00:11.391
"EURUSD" sell order: 1.20250 [250.00]
"EURUSD" sell order: 1.20245 [100.00]
"EURUSD" sell order: 1.20244 [50.00]
"EURUSD" sell order: 1.20242 [36.00]
"EURUSD" buy order: 1.20240 [16.00]
"EURUSD" buy order: 1.20239 [20.00]
"EURUSD" buy order: 1.20238 [50.00]
"EURUSD" buy order: 1.20236 [100.00]
"EURUSD" buy order: 1.20232 [250.00]
```

### What's next?

In the next article, we will continue creating the functionality for working with DOM.

All files of the current version of the library are attached below together with the test EA file for MQL5 for you to test and download.

The classes for working with DOM are under development, therefore their use in custom programs at this stage is strongly not recommended.

Leave your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/9010#node00)

**\*Previous articles within the series:**

[Prices in DoEasy library (part 59): Object to store data of one tick](https://www.mql5.com/en/articles/8818)

[Prices in DoEasy library (part 60): Series list of symbol tick data](https://www.mql5.com/en/articles/8912)

[Prices in DoEasy library (part 61): Collection of symbol tick series](https://www.mql5.com/en/articles/8952)

[Prices in DoEasy library (part 62): Updating tick series in real time, preparation for working with Depth of Market](https://www.mql5.com/en/articles/8988)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9010](https://www.mql5.com/ru/articles/9010)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9010.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/9010/mql5.zip "Download MQL5.zip")(3891.18 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/366114)**
(8)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
24 Jun 2021 at 20:55

**André Dias de Oliveira:**

Hello Artyom,

First of all congratulations on the article, simply fantastic!!  One question, from what I could see you do not touch on the position of a specific order (limit order) at a specific price level...For example, if my order is in front of the queue (first) at that level, half the way or really behind all orders.... I am tying to automate a strategy for a very liquid instrument and very low cost of trading where I could enter a position and potentially exit at the same price, for that I would need to have access to the position or my order in the queue of a specific price level...Don't seem to find this discussed anywhere.

Do you know how I would go about retrieving that information, provided the exchange does support that info?

Best Regards

André Oliveira

Thank you.

I didn't understand the question a little - probably the language barrier ...

Here, the library reads all available data that it can read from the Depth of Market using the capabilities provided by MQL.

Try to explain your question with examples, please.

![André Dias de Oliveira](https://c.mql5.com/avatar/avatar_na2.png)

**[André Dias de Oliveira](https://www.mql5.com/en/users/andredeoliveira)**
\|
27 Jun 2021 at 15:42

**Artyom Trishkin:**

Thank you.

I didn't understand the question a little - probably the language barrier ...

Here, the library reads all available data that it can read from the Depth of Market using the capabilities provided by MQL.

Try to explain your question with examples, please.

Thanks for the reply Artyom, sure, let me try to better explain.... To simplify and make it easier to understand, let's imagine a hypothetical and very simple "Order Book"  with just one depth of price level, meaning limit orders in the bid side and limit orders in the ask side ... For the example let's imaigne high volume of orders in both sides  (let\`s say bid price 1,34 and ask price 1,35). For this example let's imagine this "Order Book" has orders only in this two prices...Nothing else.

I then place a single order in both sides (ask and bid) and my orders will be placed at the very end of the queue in each side (last buy order at 1,34 side and last [sell order](https://www.mql5.com/en/docs/constants/tradingconstants/enum_book_type "MQL5 documentation: Trade Orders in Depth Of Market") at 1,35 side)

As the orders in front of mine are consumed or cancelled, my orders will make progress in the queue, and aditional limit orders MIGHT be placed behind my orders at the same price level....I wanted to understand if there is a way to retrieve the position of my orders in the queue, at any given time. See picture I have attached.

Really appreciate your attention and effort to understand my question Artyom, let me know if this is clear, I can try to think of additional examples if this is not a good one.

Best Regards and once again, really appreciate your comments on this one.

André Oliveira

André Oliveira

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
30 Jun 2021 at 20:16

**André Dias de Oliveira :**

Thanks for the reply Artyom, sure, let me try to better explain.... To simplify and make it easier to understand, let's imagine a hypothetical and very simple "Order Book"  with just one depth of price level, meaning limit orders in the bid side and limit orders in the ask side ... For the example let's imaigne high volume of orders in both sides  (let\`s say bid price 1,34 and ask price 1,35). For this example let's imagine this "Order Book" has orders only in this two prices...Nothing else.

I then place a single order in both sides (ask and bid) and my orders will be placed at the very end of the queue in each side (last buy order at 1,34 side and last [sell order](https://www.mql5.com/en/docs/constants/tradingconstants/enum_book_type "MQL5 documentation: Trade Orders in Depth Of Market") at 1,35 side)

As the orders in front of mine are consumed or cancelled, my orders will make progress in the queue, and aditional limit orders MIGHT be placed behind my orders at the same price level....I wanted to understand if there is a way to retrieve the position of my orders in the queue, at any given time. See picture I have attached.

Really appreciate your attention and effort to understand my question Artyom, let me know if this is clear, I can try to think of additional examples if this is not a good one.

Best Regards and once again, really appreciate your comments on this one.

André Oliveira

André Oliveira

I am afraid that we cannot see the order queue in the Depth of Market. Correct me if I am wrong.

![André Dias de Oliveira](https://c.mql5.com/avatar/avatar_na2.png)

**[André Dias de Oliveira](https://www.mql5.com/en/users/andredeoliveira)**
\|
30 Jun 2021 at 20:41

**Artyom Trishkin:**

I am afraid that we cannot see the order queue in the Depth of Market. Correct me if I am wrong.

Once again thanks for looking at the question Artyom.... See, I am very new to mql5 programming, but at least in our Brazillian Exchange apparently this is possible, as this has been implemented in the "Order Book" and "List of Orders" of a [trading platform](https://www.mql5.com/en/trading "Web terminal for the MetaTrader trading platform") called Profit and by another one called Tryd.. Both these trading platforms are oriented to manual traders and do not emphazise automated trading.

See attached screenshot, they expose "my order" in yellow and also show all other orders in front and behind... in fact they expose all the brokers and order sizes... it is a very transparent process.

This is probably not very usual for other exchanges (I am guessing,  as I don't have a lot of experience in other exchanges) and for this reason maybe this is not explored in the mql5 language... I will try to find out how this is exported to these trading platforms (there must be some sort of API), I just thought that MAYBE this was already explored in mql5 too.

Artyom, thank you very much for the comments you have made, much appreciated. Congratulations on your articules, they have extremely high quality content and information.

Best Regards

André Oliveira

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
30 Jun 2021 at 20:53

**André Dias de Oliveira :**

Once again thanks for looking at the question Artyom.... See, I am very new to mql5 programming, but at least in our Brazillian Exchange apparently this is possible, as this has been implemented in the "Order Book" and "List of Orders" of a [trading platform](https://www.mql5.com/en/trading "Web terminal for the MetaTrader trading platform") called Profit and by another one called Tryd.. Both these trading platforms are oriented to manual traders and do not emphazise automated trading.

See attached screenshot, they expose "my order" in yellow and also show all other orders in front and behind... in fact they expose all the brokers and order sizes... it is a very transparent process.

This is probably not very usual for other exchanges (I am guessing,  as I don't have a lot of experience in other exchanges) and for this reason maybe this is not explored in the mql5 language... I will try to find out how this is exported to these trading platforms (there must be some sort of API), I just thought that MAYBE this was already explored in mql5 too.

Artyom, thank you very much for the comments you have made, much appreciated. Congratulations on your articules, they have extremely high quality content and information.

Best Regards

André Oliveira

I will try to consider this issue in more detail. But as soon as there is time. Unfortunately, I don't have much time.

![Useful and exotic techniques for automated trading](https://c.mql5.com/2/42/exotic.png)[Useful and exotic techniques for automated trading](https://www.mql5.com/en/articles/8793)

In this article I will demonstrate some very interesting and useful techniques for automated trading. Some of them may be familiar to you. I will try to cover the most interesting methods and will explain why they are worth using. Furthermore, I will show what these techniques are apt to in practice. We will create Expert Advisors and test all the described techniques using historic quotes.

![Neural networks made easy (Part 11): A take on GPT](https://c.mql5.com/2/48/Neural_networks_made_easy_011.png)[Neural networks made easy (Part 11): A take on GPT](https://www.mql5.com/en/articles/9025)

Perhaps one of the most advanced models among currently existing language neural networks is GPT-3, the maximal variant of which contains 175 billion parameters. Of course, we are not going to create such a monster on our home PCs. However, we can view which architectural solutions can be used in our work and how we can benefit from them.

![Self-adapting algorithm (Part IV): Additional functionality and tests](https://c.mql5.com/2/41/50_percents__4.png)[Self-adapting algorithm (Part IV): Additional functionality and tests](https://www.mql5.com/en/articles/8859)

I continue filling the algorithm with the minimum necessary functionality and testing the results. The profitability is quite low but the articles demonstrate the model of the fully automated profitable trading on completely different instruments traded on fundamentally different markets.

![Prices in DoEasy library (part 62): Updating tick series in real time, preparation for working with Depth of Market](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library.png)[Prices in DoEasy library (part 62): Updating tick series in real time, preparation for working with Depth of Market](https://www.mql5.com/en/articles/8988)

In this article, I will implement updating tick data in real time and prepare the symbol object class for working with Depth of Market (DOM itself is to be implemented in the next article).

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/9010&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083151682734396922)

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