---
title: Timeseries in DoEasy library (part 35): Bar object and symbol timeseries list
url: https://www.mql5.com/en/articles/7594
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:52:25.879850
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/7594&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083175902054979175)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7594#node01)
- [Bar object](https://www.mql5.com/en/articles/7594#node02)
- [New bar object](https://www.mql5.com/en/articles/7594#node03)

- [List of bar objects, search and sorting](https://www.mql5.com/en/articles/7594#node04)
- [Testing](https://www.mql5.com/en/articles/7594#node05)
- [What's next?](https://www.mql5.com/en/articles/7594#node06)


This article starts the new section in the description of the library for convenient development of programs for MetaTrader 5 and 4 terminals. [The\\
first series (34 articles)](https://www.mql5.com/en/articles/7594#node06) was devoted to the concept of library objects and their interconnections. The concept was used to develop
the functionality for working with an account — its current status and history.

Currently, the library features the following functionality:

- searching, sorting and comparing data of any order or position,
- providing quick and convenient access to any order and position properties,
- tracking any account event,
- obtaining and comparing account and symbol data,
- responding to property changes of all existing objects in databases (collections) and sending notifications of occurred events to the
program.

We can also tell the library what events it should respond to and send notifications about the monitored events.

Besides, we have implemented the ability to work with the terminal trading functions. We have also developed new types of trading requests —
pending trading requests that allow us to create the program behavior logic in various trading conditions, as well as provide the full set of
features for creating new types of pending orders.

All this and much more was created and described [in\\
the previous series](https://www.mql5.com/en/articles/5654) of the library development descriptions.

The second series of the library description considers many aspects of working with price data, symbol charts, market depths, indicators,
etc. In general, the new series of articles is to be devoted to the development of the library functionality for quick access, search,
comparison and sorting of any price data arrays, as well as their storage objects and sources.

### Concept

The concept of arranging the library objects, their storage and working with them were described in detail in the first articles of the
previous series. In the current article, I will briefly repeat the basic principles of arranging and storing the library objects.

Any set of same-type data can be represented as a list of objects having identical properties inherent to this data type. Each object from the
list of same-type objects features a similar set of properties, although each object of the same sorted list has different values.

Each list
storing a set of same-type objects can be sorted by any of the properties the list objects have.

The object properties have three main types — integer, real, and string properties.

Sorting such objects in sorted lists allow
us to quickly find any of the objects, get any data from it and use it in custom programs.

Almost all library objects are able to independently track their own properties — changing their values by a certain amount that can be set for
each object we are interested in. When object properties are changed by a specified value, the tracked object sends the message to the
control program informing of reaching the threshold value set for tracking.

The basis of all library objects is a [base standard library\\
object](https://www.mql5.com/en/docs/standardlibrary/cobject) supplied with the trading platform, while the object list (object collection) is a [dynamic\\
array of pointers to CObject class instances and its](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) Standard library descendants.

In the current article, I am going to create the Bar object that is to store the entire data inherent to a single bar on a symbol chart, as well as
the list of all bar objects for a single symbol and timeframe.

Each bar of a symbol chart has a certain set of parameters described in the [MqlRates](https://www.mql5.com/en/docs/constants/structures/mqlrates)
structure:

- period start time
- open price
- highest price for the period
- lowest price for the period
- close price
- tick volume
- spread
- exchange volume


Apart from the main bar object properties, we are able to set its additional properties right when creating an object:

- A year the bar belongs to.
- A month the bar belongs to.
- Bar week day.

- Bar serial number in the year.
- Day of month (number).
- Bar hour.
- Bar minute.
- Bar index in a symbol timeseries.
- Bar size (from High to Low).
- Candle body top (Max(Open,Close)).
- Candle body bottom (Min(Open,Close)).
- Candle body size (from top to bottom).
- Candle upper wick (from High to candle body top).
- Candle lower wick (from candle body bottom to Low).

All these properties allow us to look for any combinations of bars and candles (stored in the bar list) within the range. The desired range of
stored data can be set for the list of bar objects. By default, we are going to store a thousand of bars in the list (or the entire available
history of symbol bars if it is less than thousand bars or another specified range).

### Bar object

Let's create enumerations of all bar object properties. To do this, open the **Defines.mqh** file stored in
\\MQL5\\Include\\DoEasy\\Defines.mqh and add the enumerations of the bar type,
integer, real
and string bar object properties, as well as the methods
of sorting the bar object list to the end of the file:

```
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Info for working with serial data                                |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Bar type (candle body)                                           |
//+------------------------------------------------------------------+
enum ENUM_BAR_BODY_TYPE
  {
   BAR_BODY_TYPE_BULLISH,                                   // Bullish bar
   BAR_BODY_TYPE_BEARISH,                                   // Bearish bar
   BAR_BODY_TYPE_NULL,                                      // Zero bar
   BAR_BODY_TYPE_CANDLE_ZERO_BODY,                          // Candle with a zero body
  };
//+------------------------------------------------------------------+
//| Bar integer properties                                           |
//+------------------------------------------------------------------+
enum ENUM_BAR_PROP_INTEGER
  {
   BAR_PROP_INDEX = 0,                                      // Bar index in timeseries
   BAR_PROP_TYPE,                                           // Bar type (from the ENUM_BAR_BODY_TYPE enumeration)
   BAR_PROP_PERIOD,                                         // Bar period (timeframe)
   BAR_PROP_SPREAD,                                         // Bar spread
   BAR_PROP_VOLUME_TICK,                                    // Bar tick volume
   BAR_PROP_VOLUME_REAL,                                    // Bar exchange volume
   BAR_PROP_TIME,                                           // Bar period start time
   BAR_PROP_TIME_DAY_OF_YEAR,                               // Bar day serial number in a year
   BAR_PROP_TIME_YEAR,                                      // A year the bar belongs to
   BAR_PROP_TIME_MONTH,                                     // A month the bar belongs to
   BAR_PROP_TIME_DAY_OF_WEEK,                               // Bar week day
   BAR_PROP_TIME_DAY,                                       // Bar day of month (number)
   BAR_PROP_TIME_HOUR,                                      // Bar hour
   BAR_PROP_TIME_MINUTE,                                    // Bar minute
  };
#define BAR_PROP_INTEGER_TOTAL (14)                         // Total number of integer bar properties
#define BAR_PROP_INTEGER_SKIP  (0)                          // Number of bar properties not used in sorting
//+------------------------------------------------------------------+
//| Real bar properties                                              |
//+------------------------------------------------------------------+
enum ENUM_BAR_PROP_DOUBLE
  {
//--- bar data
   BAR_PROP_OPEN = BAR_PROP_INTEGER_TOTAL,                  // Bar open price
   BAR_PROP_HIGH,                                           // Highest price for the bar period
   BAR_PROP_LOW,                                            // Lowest price for the bar period
   BAR_PROP_CLOSE,                                          // Bar close price
//--- candle data
   BAR_PROP_CANDLE_SIZE,                                    // Candle size
   BAR_PROP_CANDLE_SIZE_BODY,                               // Candle body size
   BAR_PROP_CANDLE_BODY_TOP,                                // Candle body top
   BAR_PROP_CANDLE_BODY_BOTTOM,                             // Candle body bottom
   BAR_PROP_CANDLE_SIZE_SHADOW_UP,                          // Candle upper wick size
   BAR_PROP_CANDLE_SIZE_SHADOW_DOWN,                        // Candle lower wick size
  };
#define BAR_PROP_DOUBLE_TOTAL  (10)                         // Total number of real bar properties
#define BAR_PROP_DOUBLE_SKIP   (0)                          // Number of bar properties not used in sorting
//+------------------------------------------------------------------+
//| Bar string properties                                            |
//+------------------------------------------------------------------+
enum ENUM_BAR_PROP_STRING
  {
   BAR_PROP_SYMBOL = (BAR_PROP_INTEGER_TOTAL+BAR_PROP_DOUBLE_TOTAL), // Bar symbol
  };
#define BAR_PROP_STRING_TOTAL  (1)                          // Total number of string bar properties
//+------------------------------------------------------------------+
//| Possible bar sorting criteria                                    |
//+------------------------------------------------------------------+
#define FIRST_BAR_DBL_PROP          (BAR_PROP_INTEGER_TOTAL-BAR_PROP_INTEGER_SKIP)
#define FIRST_BAR_STR_PROP          (BAR_PROP_INTEGER_TOTAL-BAR_PROP_INTEGER_SKIP+BAR_PROP_DOUBLE_TOTAL-BAR_PROP_DOUBLE_SKIP)
enum ENUM_SORT_BAR_MODE
  {
//--- Sort by integer properties
   SORT_BY_BAR_INDEX = 0,                                   // Sort by bar index
   SORT_BY_BAR_TYPE,                                        // Sort by bar type (from the ENUM_BAR_BODY_TYPE enumeration)
   SORT_BY_BAR_PERIOD,                                      // Sort by bar period (timeframe)
   SORT_BY_BAR_SPREAD,                                      // Sort by bar spread
   SORT_BY_BAR_VOLUME_TICK,                                 // Sort by bar tick volume
   SORT_BY_BAR_VOLUME_REAL,                                 // Sort by bar exchange volume
   SORT_BY_BAR_TIME,                                        // Sort by bar period start time
   SORT_BY_BAR_TIME_DAY_OF_YEAR,                            // Sort by bar day number in a year
   SORT_BY_BAR_TIME_YEAR,                                   // Sort by a year the bar belongs to
   SORT_BY_BAR_TIME_MONTH,                                  // Sort by a month the bar belongs to
   SORT_BY_BAR_TIME_DAY_OF_WEEK,                            // Sort by a bar week day
   SORT_BY_BAR_TIME_DAY,                                    // Sort by a bar day
   SORT_BY_BAR_TIME_HOUR,                                   // Sort by a bar hour
   SORT_BY_BAR_TIME_MINUTE,                                 // Sort by a bar minute
//--- Sort by real properties
   SORT_BY_BAR_OPEN = FIRST_BAR_DBL_PROP,                   // Sort by bar open price
   SORT_BY_BAR_HIGH,                                        // Sort by the highest price for the bar period
   SORT_BY_BAR_LOW,                                         // Sort by the lowest price for the bar period
   SORT_BY_BAR_CLOSE,                                       // Sort by a bar close price
   SORT_BY_BAR_CANDLE_SIZE,                                 // Sort by a candle price
   SORT_BY_BAR_CANDLE_SIZE_BODY,                            // Sort by a candle body size
   SORT_BY_BAR_CANDLE_BODY_TOP,                             // Sort by a candle body top
   SORT_BY_BAR_CANDLE_BODY_BOTTOM,                          // Sort by a candle body bottom
   SORT_BY_BAR_CANDLE_SIZE_SHADOW_UP,                       // Sort by candle upper wick size
   SORT_BY_BAR_CANDLE_SIZE_SHADOW_DOWN,                     // Sort by candle lower wick size
//--- Sort by string properties
   SORT_BY_BAR_SYMBOL = FIRST_BAR_STR_PROP,                 // Sort by a bar symbol
  };
//+------------------------------------------------------------------+
```

A bar type can be bullish, bearish and zero. A separate bar type
property is used to set a candle body type because a candle body
size is one of the parameters used to define various candle formations. Therefore, the zero candle body size should not be considered
equivalent to the zero bar. The zero bar has only one price for all its four OHLC prices, while the zero body candle may have wicks, while
the location and wick size are considered when defining candle formations.

To display descriptions of bar properties and some other library messages, we will need new text messages.

The **Datas.mqh** file located at \\MQL5\\Include\\DoEasy\\Datas.mqh receives indices
of new messages:

```
   MSG_LIB_SYS_ERROR_CODE_OUT_OF_RANGE,               // Return code out of range of error codes
   MSG_LIB_SYS_FAILED_CREATE_PAUSE_OBJ,               // Failed to create the \"Pause\" object
   MSG_LIB_SYS_FAILED_CREATE_BAR_OBJ,                 // Failed to create the \"Bar\" object
   MSG_LIB_SYS_FAILED_SYNC_DATA,                      // Failed to synchronize data with the server
```

...

```
   MSG_LIB_TEXT_TIME_UNTIL_THE_END_DAY,               // Order lifetime till the end of the current day to be used

   MSG_LIB_TEXT_JANUARY,                              // January
   MSG_LIB_TEXT_FEBRUARY,                             // February
   MSG_LIB_TEXT_MARCH,                                // March
   MSG_LIB_TEXT_APRIL,                                // April
   MSG_LIB_TEXT_MAY,                                  // May
   MSG_LIB_TEXT_JUNE,                                 // June
   MSG_LIB_TEXT_JULY,                                 // July
   MSG_LIB_TEXT_AUGUST,                               // August
   MSG_LIB_TEXT_SEPTEMBER,                            // September
   MSG_LIB_TEXT_OCTOBER,                              // October
   MSG_LIB_TEXT_NOVEMBER,                             // November
   MSG_LIB_TEXT_DECEMBER,                             // December

   MSG_LIB_TEXT_SUNDAY,                               // Sunday
```

...

```
   MSG_LIB_TEXT_PEND_REQUEST_ADD_CRITERIONS,          // Added pending request activation conditions

//--- CBar
   MSG_LIB_TEXT_BAR_FAILED_GET_BAR_DATA,              // Failed to receive bar data
   MSG_LIB_TEXT_BAR_FAILED_GET_SERIES_DATA,           // Failed to receive timeseries data
   MSG_LIB_TEXT_BAR_FAILED_ADD_TO_LIST,               // Could not add bar object to the list
   MSG_LIB_TEXT_BAR,                                  // Bar
   MSG_LIB_TEXT_BAR_PERIOD,                           // Timeframe
   MSG_LIB_TEXT_BAR_SPREAD,                           // Spread
   MSG_LIB_TEXT_BAR_VOLUME_TICK,                      // Tick volume
   MSG_LIB_TEXT_BAR_VOLUME_REAL,                      // Exchange volume
   MSG_LIB_TEXT_BAR_TIME,                             // Period start time
   MSG_LIB_TEXT_BAR_TIME_YEAR,                        // Year
   MSG_LIB_TEXT_BAR_TIME_MONTH,                       // Month
   MSG_LIB_TEXT_BAR_TIME_DAY_OF_YEAR,                 // Day serial number in a year
   MSG_LIB_TEXT_BAR_TIME_DAY_OF_WEEK,                 // Week day
   MSG_LIB_TEXT_BAR_TIME_DAY,                         // Two months
   MSG_LIB_TEXT_BAR_TIME_HOUR,                        // Hour
   MSG_LIB_TEXT_BAR_TIME_MINUTE,                      // Minute
   MSG_LIB_TEXT_BAR_INDEX,                            // Index in timeseries
   MSG_LIB_TEXT_BAR_HIGH,                             // Highest price for the period
   MSG_LIB_TEXT_BAR_LOW,                              // Lowest price for the period
   MSG_LIB_TEXT_BAR_CANDLE_SIZE,                      // Candle size
   MSG_LIB_TEXT_BAR_CANDLE_SIZE_BODY,                 // Candle body size
   MSG_LIB_TEXT_BAR_CANDLE_SIZE_SHADOW_UP,            // Candle upper wick size
   MSG_LIB_TEXT_BAR_CANDLE_SIZE_SHADOW_DOWN,          // Candle lower wick size
   MSG_LIB_TEXT_BAR_CANDLE_BODY_TOP,                  // Candle body top
   MSG_LIB_TEXT_BAR_CANDLE_BODY_BOTTOM,               // Candle body bottom
   MSG_LIB_TEXT_BAR_TYPE_BULLISH,                     // Bullish bar
   MSG_LIB_TEXT_BAR_TYPE_BEARISH,                     // Bearish bar
   MSG_LIB_TEXT_BAR_TYPE_NULL,                        // Zero bar
   MSG_LIB_TEXT_BAR_TYPE_CANDLE_ZERO_BODY,            // Candle with a zero body
   MSG_LIB_TEXT_BAR_TEXT_FIRS_SET_AMOUNT_DATA,        // First, we need to set the required amount of data using SetAmountUsedData()

  };
//+------------------------------------------------------------------+
```

and text messages corresponding to newly added indices:

```
   {"Код возврата вне заданного диапазона кодов ошибок","Out of range of error codes return code"},
   {"Не удалось создать объект \"Пауза\"","Failed to create \"Pause\" object"},
   {"Не удалось создать объект \"Бар\"","Failed to create \"Bar\" object"},
   {"Не удалось синхронизировать данные с сервером","Failed to sync data with server"},
```

...

```
   {"Будет использоваться время действия ордера до конца текущего дня","Order validity time until the end of the current day will be used"},

   {"Январь","January"},
   {"Февраль","February"},
   {"Март","March"},
   {"Апрель","April"},
   {"Май","May"},
   {"Июнь","June"},
   {"Июль","July"},
   {"Август","August"},
   {"Сентябрь","September"},
   {"Октябрь","October"},
   {"Ноябрь","November"},
   {"Декабрь","December"},

   {"Воскресение","Sunday"},
```

...

```
   {"Добавлены условия активации отложенного запроса","Pending request activation conditions added"},

   {"Не удалось получить данные бара","Failed to get bar data"},
   {"Не удалось получить данные таймсерии","Failed to get timeseries data"},
   {"Не удалось добавить объект-бар в список","Failed to add bar object to list"},
   {"Бар","Bar"},
   {"Таймфрейм","Timeframe"},
   {"Спред","Spread"},
   {"Тиковый объём","Tick volume"},
   {"Биржевой объём","Real volume"},
   {"Время начала периода","Period start time"},
   {"Год","Year"},
   {"Месяц","Month"},
   {"Порядковый номер дня в году","Sequence day number in a year"},
   {"День недели","Day of week"},
   {"День месяца","Day of month"},
   {"Час","Hour"},
   {"Минута","Minute"},
   {"Индекс в таймсерии","Timeseries index"},
   {"Наивысшая цена за период","Highest price for the period"},
   {"Наименьшая цена за период","Lowest price for the period"},
   {"Размер свечи","Candle size"},
   {"Размер тела свечи","Candle body size"},
   {"Размер верхней тени свечи","Candle upper shadow size"},
   {"Размер нижней тени свечи","Candle lower shadow size"},
   {"Верх тела свечи","Top of candle body"},
   {"Низ тела свечи","Bottom of candle body"},

   {"Бычий бар","Bullish bar"},
   {"Медвежий бар","Bearish bar"},
   {"Нулевой бар","Zero bar"},
   {"Свеча с нулевым телом","Candle with zero body"},
   {"Сначала нужно установить требуемое количество данных при помощи SetAmountUsedData()","First you need to set required amount of data using SetAmountUsedData()"},

  };
//+---------------------------------------------------------------------+
```

The file of service functions **DELib.mqh** located at \\MQL5\\Include\\DoEasy\\Services\\DELib.mqh receives the
function returning a month name and the function returning a
timeframe description:

```
//+------------------------------------------------------------------+
//| Return month names                                               |
//+------------------------------------------------------------------+
string MonthDescription(const int month)
  {
   return
     (
      month==1    ?  CMessage::Text(MSG_LIB_TEXT_JANUARY)   :
      month==2    ?  CMessage::Text(MSG_LIB_TEXT_FEBRUARY)  :
      month==3    ?  CMessage::Text(MSG_LIB_TEXT_MARCH)     :
      month==4    ?  CMessage::Text(MSG_LIB_TEXT_APRIL)     :
      month==5    ?  CMessage::Text(MSG_LIB_TEXT_MAY)       :
      month==6    ?  CMessage::Text(MSG_LIB_TEXT_JUNE)      :
      month==7    ?  CMessage::Text(MSG_LIB_TEXT_JULY)      :
      month==8    ?  CMessage::Text(MSG_LIB_TEXT_AUGUST)    :
      month==9    ?  CMessage::Text(MSG_LIB_TEXT_SEPTEMBER) :
      month==10   ?  CMessage::Text(MSG_LIB_TEXT_OCTOBER)   :
      month==11   ?  CMessage::Text(MSG_LIB_TEXT_NOVEMBER)  :
      month==12   ?  CMessage::Text(MSG_LIB_TEXT_DECEMBER)  :
      (string)month
     );
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Return timeframe description                                     |
//+------------------------------------------------------------------+
string TimeframeDescription(const ENUM_TIMEFRAMES timeframe)
  {
   return StringSubstr(EnumToString(timeframe),7);
  }
//+------------------------------------------------------------------+
```

The function returning a month name receives a month number, and its text description is returned according to it.

The function returning a timeframe name receives a timeframe. Next, [a\\
text representation of an enumeration value](https://www.mql5.com/en/docs/convert/enumtostring) of the timeframe is used to [retrieve\\
a substring](https://www.mql5.com/en/docs/strings/stringsubstr) starting from the position 7 up to the end of the string. The obtained result is returned as a text. For example, the value
of H1 is retrieved from the text representation of PERIOD\_H1 hour timeframe.

To store bar object classes, create a new folder
in the library's object catalog \\MQL5\\Include\\DoEasy\Objects\Series\.
In turn, create the **Bar.mqh** file of the CBar class in the new folder.

Let's consider the class body listing and implementation of its methods:

```
//+------------------------------------------------------------------+
//|                                                          Bar.mqh |
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
#include "..\..\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| bar class                                                        |
//+------------------------------------------------------------------+
class CBar : public CObject
  {
private:
   MqlDateTime       m_dt_struct;                                 // Date structure
   int               m_digits;                                    // Symbol's digits value
   string            m_period_description;                        // Timeframe string description
   long              m_long_prop[BAR_PROP_INTEGER_TOTAL];         // Integer properties
   double            m_double_prop[BAR_PROP_DOUBLE_TOTAL];        // Real properties
   string            m_string_prop[BAR_PROP_STRING_TOTAL];        // String properties

//--- Return the index of the array the bar's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_BAR_PROP_DOUBLE property)     const { return(int)property-BAR_PROP_INTEGER_TOTAL;                        }
   int               IndexProp(ENUM_BAR_PROP_STRING property)     const { return(int)property-BAR_PROP_INTEGER_TOTAL-BAR_PROP_DOUBLE_TOTAL;  }

//--- Return the bar type (bullish/bearish/zero)
   ENUM_BAR_BODY_TYPE BodyType(void)                              const;
//--- Calculate and return the size of (1) candle, (2) candle body,
//--- (3) upper, (4) lower candle wick,
//--- (5) candle body top and (6) bottom
   double            CandleSize(void)                             const { return(this.High()-this.Low());                                    }
   double            BodySize(void)                               const { return(this.BodyHigh()-this.BodyLow());                            }
   double            ShadowUpSize(void)                           const { return(this.High()-this.BodyHigh());                               }
   double            ShadowDownSize(void)                         const { return(this.BodyLow()-this.Low());                                 }
   double            BodyHigh(void)                               const { return ::fmax(this.Close(),this.Open());                           }
   double            BodyLow(void)                                const { return ::fmin(this.Close(),this.Open());                           }

//--- Return the (1) year and (2) month the bar belongs to, (3) week day,
//--- (4) bar serial number in a year, (5) day, (6) hour, (7) minute,
   int               TimeYear(void)                               const { return this.m_dt_struct.year;                                      }
   int               TimeMonth(void)                              const { return this.m_dt_struct.mon;                                       }
   int               TimeDayOfWeek(void)                          const { return this.m_dt_struct.day_of_week;                               }
   int               TimeDayOfYear(void)                          const { return this.m_dt_struct.day_of_year;                               }
   int               TimeDay(void)                                const { return this.m_dt_struct.day;                                       }
   int               TimeHour(void)                               const { return this.m_dt_struct.hour;                                      }
   int               TimeMinute(void)                             const { return this.m_dt_struct.min;                                       }

public:
//--- Set bar's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_BAR_PROP_INTEGER property,long value) { this.m_long_prop[property]=value;                              }
   void              SetProperty(ENUM_BAR_PROP_DOUBLE property,double value){ this.m_double_prop[this.IndexProp(property)]=value;            }
   void              SetProperty(ENUM_BAR_PROP_STRING property,string value){ this.m_string_prop[this.IndexProp(property)]=value;            }
//--- Return (1) integer, (2) real and (3) string bar properties from the properties array
   long              GetProperty(ENUM_BAR_PROP_INTEGER property)  const { return this.m_long_prop[property];                                 }
   double            GetProperty(ENUM_BAR_PROP_DOUBLE property)   const { return this.m_double_prop[this.IndexProp(property)];               }
   string            GetProperty(ENUM_BAR_PROP_STRING property)   const { return this.m_string_prop[this.IndexProp(property)];               }

//--- Return the flag of the bar supporting the property
   virtual bool      SupportProperty(ENUM_BAR_PROP_INTEGER property)    { return true; }
   virtual bool      SupportProperty(ENUM_BAR_PROP_DOUBLE property)     { return true; }
   virtual bool      SupportProperty(ENUM_BAR_PROP_STRING property)     { return true; }
//--- Return itself
   CBar             *GetObject(void)                                    { return &this;}
//--- Set (1) bar symbol, timeframe and index, (2) bar object parameters
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
   void              SetProperties(const MqlRates &rates);

//--- Compare CBar objects by all possible properties (for sorting the lists by a specified bar object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CBar objects by all properties (to search for equal bar objects)
   bool              IsEqual(CBar* compared_bar) const;
//--- Constructors
                     CBar(){;}
                     CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index);
                     CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index,const MqlRates &rates);

//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
//+------------------------------------------------------------------+
//--- Return the (1) type, (2) period, (3) spread, (4) tick, (5) exchange volume,
//--- (6) bar period start time, (7) year, (8) month the bar belongs to
//--- (9) week number since the year start, (10) week number since the month start
//--- (11) bar's day, (12) hour, (13) minute, (14) index
   ENUM_BAR_BODY_TYPE   TypeBody(void)                                  const { return (ENUM_BAR_BODY_TYPE)this.GetProperty(BAR_PROP_TYPE);  }
   ENUM_TIMEFRAMES   Period(void)                                       const { return (ENUM_TIMEFRAMES)this.GetProperty(BAR_PROP_PERIOD);   }
   int               Spread(void)                                       const { return (int)this.GetProperty(BAR_PROP_SPREAD);               }
   long              VolumeTick(void)                                   const { return this.GetProperty(BAR_PROP_VOLUME_TICK);               }
   long              VolumeReal(void)                                   const { return this.GetProperty(BAR_PROP_VOLUME_REAL);               }
   datetime          Time(void)                                         const { return (datetime)this.GetProperty(BAR_PROP_TIME);            }
   long              Year(void)                                         const { return this.GetProperty(BAR_PROP_TIME_YEAR);                 }
   long              Month(void)                                        const { return this.GetProperty(BAR_PROP_TIME_MONTH);                }
   long              DayOfWeek(void)                                    const { return this.GetProperty(BAR_PROP_TIME_DAY_OF_WEEK);          }
   long              DayOfYear(void)                                    const { return this.GetProperty(BAR_PROP_TIME_DAY_OF_YEAR);          }
   long              Day(void)                                          const { return this.GetProperty(BAR_PROP_TIME_DAY);                  }
   long              Hour(void)                                         const { return this.GetProperty(BAR_PROP_TIME_HOUR);                 }
   long              Minute(void)                                       const { return this.GetProperty(BAR_PROP_TIME_MINUTE);               }
   long              Index(void)                                        const { return this.GetProperty(BAR_PROP_INDEX);                     }

//--- Return bar's (1) Open, (2) High, (3) Low, (4) Close price,
//--- size of the (5) candle, (6) body, (7) candle top, (8) bottom,
//--- size of the (9) candle upper, (10) lower wick
   double            Open(void)                                         const { return this.GetProperty(BAR_PROP_OPEN);                      }
   double            High(void)                                         const { return this.GetProperty(BAR_PROP_HIGH);                      }
   double            Low(void)                                          const { return this.GetProperty(BAR_PROP_LOW);                       }
   double            Close(void)                                        const { return this.GetProperty(BAR_PROP_CLOSE);                     }
   double            Size(void)                                         const { return this.GetProperty(BAR_PROP_CANDLE_SIZE);               }
   double            SizeBody(void)                                     const { return this.GetProperty(BAR_PROP_CANDLE_SIZE_BODY);          }
   double            TopBody(void)                                      const { return this.GetProperty(BAR_PROP_CANDLE_BODY_TOP);           }
   double            BottomBody(void)                                   const { return this.GetProperty(BAR_PROP_CANDLE_BODY_BOTTOM);        }
   double            SizeShadowUp(void)                                 const { return this.GetProperty(BAR_PROP_CANDLE_SIZE_SHADOW_UP);     }
   double            SizeShadowDown(void)                               const { return this.GetProperty(BAR_PROP_CANDLE_SIZE_SHADOW_DOWN);   }

//--- Return bar symbol
   string            Symbol(void)                                       const { return this.GetProperty(BAR_PROP_SYMBOL);                    }

//+------------------------------------------------------------------+
//| Descriptions of bar object properties                            |
//+------------------------------------------------------------------+
//--- Get description of a bar's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_BAR_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_BAR_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_BAR_PROP_STRING property);

//--- Return the bar type description
   string            BodyTypeDescription(void)  const;
//--- Send description of bar properties to the journal (full_prop=true - all properties, false - only supported ones)
   void              Print(const bool full_prop=false);
//--- Display a short bar description in the journal
   virtual void      PrintShort(void);
//--- Return the bar object short name
   virtual string    Header(void);
//---
  };
//+------------------------------------------------------------------+
```

Let's consider the class contents to refresh memory about the [previously\\
described material](https://www.mql5.com/en/articles/5654#node02).

The private class section features:

Three arrays storing the corresponding bar object properties
— integer, real and string ones.

The methods calculating the true
index of the object property in the appropriate array.

The
methods calculating and returning additional bar object properties.

The public section of the class features:

The methods writing the passed object property value to the arrays of
integer, real and string properties.

The methods returning the
value of a requested integer, real or string property from the arrays.

The
virtual methods returning the flag of supporting a property by the object for each of the properties. The methods are meant to be
implemented in descendant objects of the bar object and should return false in case
the descendant object does not support the specified property. In the Bar object, all properties are supported and the methods return true.

We discussed the entire structure of the library objects [in the\\
first article](https://www.mql5.com/en/articles/5654#node02). Here, we will briefly consider the implementation of the remaining class methods.

The class features three constructors:

1\. By default, the constructor without parameters is used for a simple declaration of an object class with subsequent setting of all the
necessary parameters for the created object.

**2\. The first parametric constructor** receives three parameters — symbol, timeframe and bar index. Based on these three
parameters, it gets all the properties of a single bar object from the timeseries using the first form of the [CopyRates()](https://www.mql5.com/en/docs/series/copyrates)
function:

```
//+------------------------------------------------------------------+
//| Constructor 1                                                    |
//+------------------------------------------------------------------+
CBar::CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   MqlRates rates_array[1];
   this.SetSymbolPeriod(symbol,timeframe,index);
   ::ResetLastError();
//--- If failed to write bar data to the array by index or set the time to the time structure
//--- display an error message, create and fill the structure with zeros, and write it to the rates_array array
   if(::CopyRates(symbol,timeframe,index,1,rates_array)<1 || !::TimeToStruct(rates_array[0].time,this.m_dt_struct))
     {
      int err_code=::GetLastError();
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_GET_BAR_DATA),". ",CMessage::Text(MSG_LIB_SYS_ERROR)," ",CMessage::Text(err_code)," ",CMessage::Retcode(err_code));
      MqlRates err={0};
      rates_array[0]=err;
     }
//--- Set the bar properties
   this.SetProperties(rates_array[0]);
  }
//+------------------------------------------------------------------+
```

This constructor is used for one-time data acquisition from the timeseries immediately when creating the bar object.

**3\. The second parametric constructor** is used to create a bar object from the already prepared array of [MqlRates](https://www.mql5.com/en/docs/constants/structures/mqlrates)
structures.

In other words, this implies passing through the MqlRates structure array in a loop and creating an object by the array
index:

```
//+------------------------------------------------------------------+
//| Constructor 2                                                    |
//+------------------------------------------------------------------+
CBar::CBar(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index,const MqlRates &rates)
  {
   this.SetSymbolPeriod(symbol,timeframe,index);
   ::ResetLastError();
//--- If failed to set time to the time structure, display the error message,
//--- create and fill the structure with zeros, set the bar properties from this structure and exit
   if(!::TimeToStruct(rates.time,this.m_dt_struct))
     {
      int err_code=::GetLastError();
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_GET_BAR_DATA),". ",CMessage::Text(MSG_LIB_SYS_ERROR)," ",CMessage::Text(err_code)," ",CMessage::Retcode(err_code));
      MqlRates err={0};
      this.SetProperties(err);
      return;
     }
//--- Set the bar properties
   this.SetProperties(rates);
  }
//+------------------------------------------------------------------+
```

Here, the link to the MqlRates structure is passed to the constructor apart from the symbol, timeframe and index. The bar object is created
based on this data.

**The Compare() virtual method** is meant for comparing two objects by the specified property. It is defined in the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectcompare)
base object class of the standard library and should return zero if the values are equal and 1/-1 if one of the compared values is higher/lower.
The Search() method of the Standard library is used for searching and sorting. The method should be redefined in the descendant
classes:

```
//+------------------------------------------------------------------+
//| Compare CBar objects by all possible properties                  |
//+------------------------------------------------------------------+
int CBar::Compare(const CObject *node,const int mode=0) const
  {
   const CBar *bar_compared=node;
//--- compare integer properties of two bars
   if(mode<BAR_PROP_INTEGER_TOTAL)
     {
      long value_compared=bar_compared.GetProperty((ENUM_BAR_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_BAR_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare real properties of two bars
   else if(mode<BAR_PROP_DOUBLE_TOTAL+BAR_PROP_INTEGER_TOTAL)
     {
      double value_compared=bar_compared.GetProperty((ENUM_BAR_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_BAR_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two bars
   else if(mode<BAR_PROP_DOUBLE_TOTAL+BAR_PROP_INTEGER_TOTAL+BAR_PROP_STRING_TOTAL)
     {
      string value_compared=bar_compared.GetProperty((ENUM_BAR_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_BAR_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

**The method of defining two similar bar objects** is used to compare two object bars. It returns true
only if all the fields of the two compared objects are identical:

```
//+------------------------------------------------------------------+
//| Compare CBar objects by all properties                           |
//+------------------------------------------------------------------+
bool CBar::IsEqual(CBar *compared_bar) const
  {
   int beg=0, end=BAR_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BAR_PROP_INTEGER prop=(ENUM_BAR_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_bar.GetProperty(prop)) return false;
     }
   beg=end; end+=BAR_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BAR_PROP_DOUBLE prop=(ENUM_BAR_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_bar.GetProperty(prop)) return false;
     }
   beg=end; end+=BAR_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BAR_PROP_STRING prop=(ENUM_BAR_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_bar.GetProperty(prop)) return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

**The method of placing a symbol, timeframe and bar object index in the timeseries**:

```
//+------------------------------------------------------------------+
//| Set bar symbol, timeframe and index                              |
//+------------------------------------------------------------------+
void CBar::SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe,const int index)
  {
   this.SetProperty(BAR_PROP_INDEX,index);
   this.SetProperty(BAR_PROP_SYMBOL,symbol);
   this.SetProperty(BAR_PROP_PERIOD,timeframe);
   this.m_digits=(int)::SymbolInfoInteger(symbol,SYMBOL_DIGITS);
   this.m_period_description=TimeframeDescription(timeframe);
  }
//+------------------------------------------------------------------+
```

Apart from setting the three mentioned properties, the method sets the number of decimal places in a symbol price to the **m\_digits** variable
and the timeframe text description to the **m\_period\_description** variable. It is sufficient to specify them once when creating the bar
object.

**The method for setting all bar object parameters** simply writes the values from the MqlRates structure passed to the
method to the object properties, as well as calculates the parameters of additional object properties using the appropriate methods:

```
//+------------------------------------------------------------------+
//| Set bar object parameters                                        |
//+------------------------------------------------------------------+
void CBar::SetProperties(const MqlRates &rates)
  {
   this.SetProperty(BAR_PROP_SPREAD,rates.spread);
   this.SetProperty(BAR_PROP_VOLUME_TICK,rates.tick_volume);
   this.SetProperty(BAR_PROP_VOLUME_REAL,rates.real_volume);
   this.SetProperty(BAR_PROP_TIME,rates.time);
   this.SetProperty(BAR_PROP_TIME_YEAR,this.TimeYear());
   this.SetProperty(BAR_PROP_TIME_MONTH,this.TimeMonth());
   this.SetProperty(BAR_PROP_TIME_DAY_OF_YEAR,this.TimeDayOfYear());
   this.SetProperty(BAR_PROP_TIME_DAY_OF_WEEK,this.TimeDayOfWeek());
   this.SetProperty(BAR_PROP_TIME_DAY,this.TimeDay());
   this.SetProperty(BAR_PROP_TIME_HOUR,this.TimeHour());
   this.SetProperty(BAR_PROP_TIME_MINUTE,this.TimeMinute());
//---
   this.SetProperty(BAR_PROP_OPEN,rates.open);
   this.SetProperty(BAR_PROP_HIGH,rates.high);
   this.SetProperty(BAR_PROP_LOW,rates.low);
   this.SetProperty(BAR_PROP_CLOSE,rates.close);
   this.SetProperty(BAR_PROP_CANDLE_SIZE,this.CandleSize());
   this.SetProperty(BAR_PROP_CANDLE_SIZE_BODY,this.BodySize());
   this.SetProperty(BAR_PROP_CANDLE_BODY_TOP,this.BodyHigh());
   this.SetProperty(BAR_PROP_CANDLE_BODY_BOTTOM,this.BodyLow());
   this.SetProperty(BAR_PROP_CANDLE_SIZE_SHADOW_UP,this.ShadowUpSize());
   this.SetProperty(BAR_PROP_CANDLE_SIZE_SHADOW_DOWN,this.ShadowDownSize());
//---
   this.SetProperty(BAR_PROP_TYPE,this.BodyType());
  }
//+------------------------------------------------------------------+
```

**The method displaying descriptions of all bar object properties**:

```
//+------------------------------------------------------------------+
//| Display bar properties in the journal                            |
//+------------------------------------------------------------------+
void CBar::Print(const bool full_prop=false)
  {
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_BEG)," (",this.Header(),") =============");
   int beg=0, end=BAR_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BAR_PROP_INTEGER prop=(ENUM_BAR_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=BAR_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BAR_PROP_DOUBLE prop=(ENUM_BAR_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=BAR_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_BAR_PROP_STRING prop=(ENUM_BAR_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("============= ",CMessage::Text(MSG_LIB_PARAMS_LIST_END)," (",this.Header(),") =============\n");
  }
//+------------------------------------------------------------------+
```

Description of each next property is displayed by object property arrays in three loops. If the property is not supported, it is not displayed in
the journal if the full\_prop method input is false (by default).

**The method displaying a short description of the bar object in the journal**:

```
//+------------------------------------------------------------------+
//| Display a short bar description in the journal                   |
//+------------------------------------------------------------------+
void CBar::PrintShort(void)
  {
   int dg=(this.m_digits>0 ? this.m_digits : 1);
   string params=
     (
      ::TimeToString(this.Time(),TIME_DATE|TIME_MINUTES|TIME_SECONDS)+", "+
      "O: "+::DoubleToString(this.Open(),dg)+", "+
      "H: "+::DoubleToString(this.High(),dg)+", "+
      "L: "+::DoubleToString(this.Low(),dg)+", "+
      "C: "+::DoubleToString(this.Close(),dg)+", "+
      "V: "+(string)this.VolumeTick()+", "+
      (this.VolumeReal()>0 ? "R: "+(string)this.VolumeReal()+", " : "")+
      this.BodyTypeDescription()
     );
   ::Print(this.Header(),": ",params);
  }
//+------------------------------------------------------------------+
```

The method displays the bar description in the following format

```
Bar "SYMBOL" H4[INDEX]: YYYY.MM.DD HH:MM:SS, O: X.XXXXX, H: X.XXXXX, L: X.XXXXX, C: X.XXXXX, V: XXXX, BAR_TYPE
```

For example:

```
Bar "EURUSD" H4[6]: 2020.02.06 20:00:00, O: 1.09749, H: 1.09828, L: 1.09706, C: 1.09827, V: 3323, Bullish bar
```

**The method displaying a short bar name**:

```
//+------------------------------------------------------------------+
//| Return the bar object short name                                 |
//+------------------------------------------------------------------+
string CBar::Header(void)
  {
   return
     (
      CMessage::Text(MSG_LIB_TEXT_BAR)+" \""+this.GetProperty(BAR_PROP_SYMBOL)+"\" "+
      TimeframeDescription((ENUM_TIMEFRAMES)this.GetProperty(BAR_PROP_PERIOD))+"["+(string)this.GetProperty(BAR_PROP_INDEX)+"]"
     );
  }
//+------------------------------------------------------------------+
```

Display the bar name in the following format

```
Bar "SYMBOL" H4[INDEX]
```

For example:

```
Bar "EURUSD" H4[6]
```

**The method returning the description of the bar object integer property**:

```
//+------------------------------------------------------------------+
//| Return the description of the bar integer property               |
//+------------------------------------------------------------------+
string CBar::GetPropertyDescription(ENUM_BAR_PROP_INTEGER property)
  {
   return
     (
      property==BAR_PROP_INDEX               ?  CMessage::Text(MSG_LIB_TEXT_BAR_INDEX)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BAR_PROP_TYPE                ?  CMessage::Text(MSG_ORD_TYPE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.BodyTypeDescription()
         )  :
      property==BAR_PROP_PERIOD              ?  CMessage::Text(MSG_LIB_TEXT_BAR_PERIOD)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.m_period_description
         )  :
      property==BAR_PROP_SPREAD              ?  CMessage::Text(MSG_LIB_TEXT_BAR_SPREAD)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BAR_PROP_VOLUME_TICK         ?  CMessage::Text(MSG_LIB_TEXT_BAR_VOLUME_TICK)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BAR_PROP_VOLUME_REAL         ?  CMessage::Text(MSG_LIB_TEXT_BAR_VOLUME_REAL)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BAR_PROP_TIME                ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==BAR_PROP_TIME_YEAR           ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_YEAR)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.Year()
         )  :
      property==BAR_PROP_TIME_MONTH          ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_MONTH)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+MonthDescription((int)this.Month())
         )  :
      property==BAR_PROP_TIME_DAY_OF_YEAR    ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_DAY_OF_YEAR)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)::IntegerToString(this.DayOfYear(),3,'0')
         )  :
      property==BAR_PROP_TIME_DAY_OF_WEEK    ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_DAY_OF_WEEK)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+DayOfWeekDescription((ENUM_DAY_OF_WEEK)this.DayOfWeek())
         )  :
      property==BAR_PROP_TIME_DAY ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_DAY)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)::IntegerToString(this.Day(),2,'0')
         )  :
      property==BAR_PROP_TIME_HOUR           ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_HOUR)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)::IntegerToString(this.Hour(),2,'0')
         )  :
      property==BAR_PROP_TIME_MINUTE         ?  CMessage::Text(MSG_LIB_TEXT_BAR_TIME_MINUTE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)::IntegerToString(this.Minute(),2,'0')
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

The method passes the integer property. Depending on its value, its text description set in the Datas.mqh file is returned.

**The methods returning descriptions of bar object real and string properties** are similar to the method returning
the description of the bar object integer property:

```
//+------------------------------------------------------------------+
//| Return the description of the bar's real property                |
//+------------------------------------------------------------------+
string CBar::GetPropertyDescription(ENUM_BAR_PROP_DOUBLE property)
  {
   int dg=(this.m_digits>0 ? this.m_digits : 1);
   return
     (
      property==BAR_PROP_OPEN                ?  CMessage::Text(MSG_ORD_PRICE_OPEN)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_HIGH                ?  CMessage::Text(MSG_LIB_TEXT_BAR_HIGH)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_LOW                 ?  CMessage::Text(MSG_LIB_TEXT_BAR_LOW)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_CLOSE               ?  CMessage::Text(MSG_ORD_PRICE_CLOSE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_CANDLE_SIZE         ?  CMessage::Text(MSG_LIB_TEXT_BAR_CANDLE_SIZE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_CANDLE_SIZE_BODY    ?  CMessage::Text(MSG_LIB_TEXT_BAR_CANDLE_SIZE_BODY)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_CANDLE_SIZE_SHADOW_UP  ?  CMessage::Text(MSG_LIB_TEXT_BAR_CANDLE_SIZE_SHADOW_UP)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_CANDLE_SIZE_SHADOW_DOWN   ?  CMessage::Text(MSG_LIB_TEXT_BAR_CANDLE_SIZE_SHADOW_DOWN)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_CANDLE_BODY_TOP     ?  CMessage::Text(MSG_LIB_TEXT_BAR_CANDLE_BODY_TOP)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      property==BAR_PROP_CANDLE_BODY_BOTTOM  ?  CMessage::Text(MSG_LIB_TEXT_BAR_CANDLE_BODY_BOTTOM)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::DoubleToString(this.GetProperty(property),dg)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the bar string property                |
//+------------------------------------------------------------------+
string CBar::GetPropertyDescription(ENUM_BAR_PROP_STRING property)
  {
   return(property==BAR_PROP_SYMBOL ? CMessage::Text(MSG_LIB_PROP_SYMBOL)+": \""+this.GetProperty(property)+"\"" : "");
  }
//+------------------------------------------------------------------+
```

**The method returning the bar type**:

```
//+------------------------------------------------------------------+
//| Return the bar type (bullish/bearish/zero)                       |
//+------------------------------------------------------------------+
ENUM_BAR_BODY_TYPE CBar::BodyType(void) const
  {
   return
     (
      this.Close()>this.Open() ? BAR_BODY_TYPE_BULLISH :
      this.Close()<this.Open() ? BAR_BODY_TYPE_BEARISH :
      (this.ShadowUpSize()+this.ShadowDownSize()==0 ? BAR_BODY_TYPE_NULL : BAR_BODY_TYPE_CANDLE_ZERO_BODY)
     );
  }
//+------------------------------------------------------------------+
```

Here all is simple: if the bar close price exceeds the open price, this
is a bullish bar, if the bar close price is lower than the open one,
this is a bearish price. If both candle wicks are equal to zero, this
is a bar with a zero body, otherwise this
is a candle with a zero body.

**The method returning a bar type description**:

```
//+------------------------------------------------------------------+
//| Return the bar type description                                  |
//+------------------------------------------------------------------+
string CBar::BodyTypeDescription(void) const
  {
   return
     (
      this.BodyType()==BAR_BODY_TYPE_BULLISH          ? CMessage::Text(MSG_LIB_TEXT_BAR_TYPE_BULLISH)          :
      this.BodyType()==BAR_BODY_TYPE_BEARISH          ? CMessage::Text(MSG_LIB_TEXT_BAR_TYPE_BEARISH)          :
      this.BodyType()==BAR_BODY_TYPE_CANDLE_ZERO_BODY ? CMessage::Text(MSG_LIB_TEXT_BAR_TYPE_CANDLE_ZERO_BODY) :
      CMessage::Text(MSG_LIB_TEXT_BAR_TYPE_NULL)
     );
  }
//+------------------------------------------------------------------+
```

Depending on the bar type, the method returns its text description set in the Datas.mqh file.

The Bar object class is ready. Now we can create a bar object for each necessary bar of a required timeseries. However, the object alone
cannot give us any significant advantage over the usual receiving of data via requesting a timeseries bar using CopyRates().

To be able to manage timeseries data as we want, we need to create the list of bar objects corresponding to the required timeseries in the
required quantity. Then we can already analyze the list data and search for all the data necessary for analysis.

This means we need to create a timeseries list to store bar objects.

Besides, we will need to detect the opening of a
new bar to add another bar object to the list and always have a tool notifying of opening a new bar on any symbol and timeframe regardless of
their amount.

Before creating a list of storing bar objects, write the "New bar" class since the class object is one of the bar list properties.

### New bar object

To detect the event of opening a new bar, compare the time of opening the current bar with the time of the previous opening. If they do not
match, a new bar has been opened. In this case, we need to save the new opening time as the previous one for subsequent comparison:

```
NewBar = false;
if(PrevTime != Time)
  {
   NewBar = true;
   PrevTime = Time;
  }
```

In this case, the new bar event is opened once, while all subsequent commands are performed on the new bar already.

However, sometimes we require certain commands to be executed precisely after the "New bar". The event should remain relevant till all
commands are completed. To achieve this, we need to execute all the commands that should be completed by the time a new bar appears before
assigning a new value to the previous one:

```
NewBar = false;
if(PrevTime != Time)
  {
   NewBar = true;
   // ... commands to be
   // ... executed
   // ... when a new bar appears
   PrevTime = Time;
  }
```

Thus, the first option can be performed as an independent function returning the new bar opening flag. The second option in the presented
version should be part of the OnTick() handler, preferably at the very beginning, so that all commands to be executed at the moment the
new bar appears are executed first. They are then followed by all commands executed at all times.

In the simplest case, this is sufficient to control the opening of a new bar.

However, this is not enough for the
library needs — we require a separate new bar definition for each symbol and timeframe in two described options:

1. auto control and saving time for a specified symbol and timeframe (returning the "New bar" flag event for each symbol and timeframe
    separately),
2. time management for a specified symbol and timeframe with manual control of saving its new value


    (defining the "New bar" event and allowing a user to determine when to save the new time for subsequent control of the next new bar for
    each symbol and timeframe separately).

[It is not recommended to access the timeseries update functions\\
if data is requested on the current symbol and timeframe](https://www.mql5.com/en/docs/series/timeseries_access). Such a request is likely to cause a conflict since historical data
is updated in the same thread the indicator works in. Therefore, be sure to check _whether you are trying to get the timeseries_
_data from the current symbol or timeframe_. If yes, use another method for such a request instead of applying [SeriesInfoInteger()](https://www.mql5.com/en/docs/series/seriesinfointeger)
and [other functions returning serial data](https://www.mql5.com/en/docs/series), accessing which initiates
loading of history.

Such a method exists and it is quite simple:

The [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) handler parameters of the
indicators already feature the necessary variables:

```
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
```

- **rates\_total**— number of available history in timeseries (similar to the [Bars()](https://www.mql5.com/en/docs/series/bars)
function without parameters),
- **prev\_calculated**— amount of the already calculated data during the last call,
- **time\[\]** — timeseries array with data on the bar time.


The indicators allow you to track changes of historical data using a simple calculation:

- if (rates\_total - prev\_calculated) exceeds 1, this means the history is loaded and the indicator should be redrawn anew,
- if (rates\_total - prev\_calculated) is 1, a new bar is opened on the current timeframe symbol.
- normally, the (rates\_total - prev\_calculated) expression is 0 on each new tick.

In the indicators, the bar time from the time\[\] array is passed to the class methods for the current symbol and timeframe to define a new
bar and specify its time. In other cases, we receive time inside the class methods.

In \\MQL5\\Include\\DoEasy\\Objects\\Series\\, create the new file **NewBarObj.mqh** of the CNewBarObj class:

```
//+------------------------------------------------------------------+
//|                                                    NewBarObj.mqh |
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

//+------------------------------------------------------------------+
//| "New bar" object class                                           |
//+------------------------------------------------------------------+
class CNewBarObj
  {
private:
   string            m_symbol;                                    // Symbol
   ENUM_TIMEFRAMES   m_timeframe;                                 // Timeframe
   datetime          m_new_bar_time;                              // New bar time for auto time management
   datetime          m_prev_time;                                 // Previous time for auto time management
   datetime          m_new_bar_time_manual;                       // New bar time for manual time management
   datetime          m_prev_time_manual;                          // Previous time for manual time management
//--- Return the current bar data
   datetime          GetLastBarDate(const datetime time);
public:
//--- Set (1) symbol and (2) timeframe
   void              SetSymbol(const string symbol)               { this.m_symbol=(symbol==NULL || symbol==""   ? ::Symbol() : symbol);                     }
   void              SetPeriod(const ENUM_TIMEFRAMES timeframe)   { this.m_timeframe=(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe); }
//--- Save the new bar time during the manual time management
   void              SaveNewBarTime(const datetime time)          { this.m_prev_time_manual=this.GetLastBarDate(time);                                      }
//--- Return (1) symbol and (2) timeframe
   string            Symbol(void)                           const { return this.m_symbol;       }
   ENUM_TIMEFRAMES   Period(void)                           const { return this.m_timeframe;    }
//--- Return the (1) new bar time
   datetime          TimeNewBar(void)                       const { return this.m_new_bar_time; }
//--- Return the new bar opening flag during the time (1) auto, (2) manual management
   bool              IsNewBar(const datetime time);
   bool              IsNewBarManual(const datetime time);
//--- Constructors
                     CNewBarObj(void) : m_symbol(::Symbol()),
                                        m_timeframe((ENUM_TIMEFRAMES)::Period()),
                                        m_prev_time(0),m_new_bar_time(0),
                                        m_prev_time_manual(0),m_new_bar_time_manual(0) {}
                     CNewBarObj(const string symbol,const ENUM_TIMEFRAMES timeframe);
  };
//+------------------------------------------------------------------+
```

I think, all is quite simple here:

The class member variables
are declared in the private section for storing a symbol and timeframe, for which the object is to define the "New bar" event,

the variables for storing the new bar open time and the previous open time are separate for
auto and manual time management (we have already
discussed why we need this).

The **GetLastBarDate()** method returns the new bar open time and is to be considered below.

The **SaveNewBarTime()** method for saving the new bar time as the previous time for manual time management is declared and
implemented in the public class section. It allows library users to save the new bar time after all the necessary actions on it are
completed.

The remaining methods are self-explanatory and there is no point in dwelling on them here.

The class features two constructors. The first constructor has no parameters. The current symbol and timeframe are set in its
initialization list, while all new bar time values and previous bar opening time are reset. After creating such an object, we need to
independently call the methods of setting the necessary symbol and timeframe for the created class object.

The second
constructor is parametric. The necessary symbol and timeframe are immediately sent to it:

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CNewBarObj::CNewBarObj(const string symbol,const ENUM_TIMEFRAMES timeframe) : m_symbol(symbol),m_timeframe(timeframe)
  {
   this.m_prev_time=this.m_prev_time_manual=this.m_new_bar_time=this.m_new_bar_time_manual=0;
  }
//+------------------------------------------------------------------+
```

A symbol and timeframe passed to the created class object parameters are set in its initialization list. Zero values are then set to all
time variables in the class body.

**The method returning the new bar opening flag during auto time management**:

```
//+------------------------------------------------------------------+
//| Return new bar opening flag                                      |
//+------------------------------------------------------------------+
bool CNewBarObj::IsNewBar(const datetime time)
  {
//--- Get the current bar time
   datetime tm=this.GetLastBarDate(time);
//--- If the previous and current time are equal to zero, this is the first launch
   if(this.m_prev_time==0 && this.m_new_bar_time==0)
     {
      //--- set the new bar opening time,
      //--- set the previous bar time as the current one and return 'false'
      this.m_new_bar_time=this.m_prev_time=tm;
      return false;
     }
//--- If the previous time is not equal to the current bar open time, this is a new bar
   if(this.m_prev_time!=tm)
     {
      //--- set the new bar opening time,
      //--- set the previous time as the current one and return 'true'
      this.m_new_bar_time=this.m_prev_time=tm;
      return true;
     }
//--- in other cases, return 'false'
   return false;
  }
//+------------------------------------------------------------------+
```

The method returns true once each time a new bar is opened on a symbol or
timeframe set for the object.

**The method returning the new bar opening flag during manual time management**:

```
//+------------------------------------------------------------------+
//| Return the new bar opening flag during the manual management     |
//+------------------------------------------------------------------+
bool CNewBarObj::IsNewBarManual(const datetime time)
  {
//--- Get the current bar time
   datetime tm=this.GetLastBarDate(time);
//--- If the previous and current time are equal to zero, this is the first launch
   if(this.m_prev_time_manual==0 && this.m_new_bar_time_manual==0)
     {
      //--- set the new bar opening time,
      //--- set the previous bar time as the current one and return 'false'
      this.m_new_bar_time_manual=this.m_prev_time_manual=tm;
      return false;
     }
//--- If the previous time is not equal to the current bar open time, this is a new bar
   if(this.m_prev_time_manual!=tm)
     {
      //--- set the new bar opening time and return 'true'
      //--- Save the previous time as the current one from the program using the SaveNewBarTime() method
      //--- Till the previous time is forcibly set as the current one from the program,
      //--- the method returns the new bar flag allowing the completion of all the necessary actions on the new bar.
      this.m_new_bar_time=tm;
      return true;
     }
//--- in other cases, return 'false'
   return false;
  }
//+------------------------------------------------------------------+
```

Unlike the auto time management, the method does not set the current new bar opening time to the variable storing the bar's previous time.
This makes it possible to send the flag of opening a new bar each time on a new tick after detecting the "New bar" event till the user decides
that all actions to be performed when opening a new bar have been completed and it is necessary to save the new bar opening time as the
previous one.

The time of a new bar is passed to the inputs of both methods. The GetLastBarDate() method is then used to define which
time to use:

```
//+------------------------------------------------------------------+
//| Return the current bar time                                      |
//+------------------------------------------------------------------+
datetime CNewBarObj::GetLastBarDate(const datetime time)
  {
   return
     (
      ::MQLInfoInteger(MQL_PROGRAM_TYPE)==PROGRAM_INDICATOR && this.m_symbol==::Symbol() && this.m_timeframe==::Period() ? time :
      (datetime)::SeriesInfoInteger(this.m_symbol,this.m_timeframe,SERIES_LASTBAR_DATE)
     );
  }
//+------------------------------------------------------------------+
```

If this is an indicator,both
a symbol and timeframe of the "New bar" object coincide with the current symbol and timeframe, the
time passed to the method is returned (in indicators, the time is present in the OnCalculate() parameters, namely in the time\[\]
array — this is the array, from which the time should be passed to the methods defining a new bar), otherwise
receive the last bar time using SeriesInfoInteger()
— in this case, any value can be passed instead of time.

The "New bar" object class is ready for our current needs. Now it is time to create the list of bar objects.

The list of bar objects is essentially a segment of historical timeseries data which is a part of MqlRates. Why do we need a separate list?

First, we obtain the fast sorting, search and comparison feature. Second, apart from the MqlRates structure fields, the bar objects,
whose specified number is stored in the list, provide additional fields with values that are to simplify the search for different
candle formations in the future.

### List of bar objects, search and sorting

For the timeseries list (list of bar objects), we are going to use the standard library's[class\\
of dynamic array of pointers to instances of the CObject class and its descendants](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj). In the current article, I will implement the only list
object of bar objects to store bars of only one timeseries by symbol and timeframe. In the future articles, I am going to use the list as a basis
for the collection of timeseries by timeframes for each separate symbol applied in the user program. Thus, we will have multiple same-type
collections of timeseries lists, in which it will be possible to quickly search for the necessary information for analysis and comparison
with other lists of timeseries collections available to library users.

Each list of bar objects is to have a user-defined number of bar objects (history depth). By default, the history depth of all the lists is to have
a dimension of 1000 bars. Before its construction, each list should consider synchronization of data with the trade server. Each list of
each timeseries is to have a value specifying the number of available history bars. This value is returned by the Bars() function without
parameters. If it returns zero, the history is not synchronized yet. We will make several attempts with little break time between them while
waiting for data synchronization with the server.

In the **Defines.mqh** file, create the macro substitution defining
the depth of the applied history, the number of pause milliseconds
between the attempts to synchronize history with the server and the number
of attempts to obtain the synchronization event:

```
//--- Pending request type IDs
#define PENDING_REQUEST_ID_TYPE_ERR    (1)                        // Type of a pending request created based on the server return code
#define PENDING_REQUEST_ID_TYPE_REQ    (2)                        // Type of a pending request created by request
//--- Timeseries parameters
#define SERIES_DEFAULT_BARS_COUNT      (1000)                     // Required default amount of timeseries data
#define PAUSE_FOR_SYNC_ATTEMPTS        (16)                       // Amount of pause milliseconds between synchronization attempts
#define ATTEMPTS_FOR_SYNC              (5)                        // Number of attempts to receive synchronization with the server
//+------------------------------------------------------------------+
//| Structures                                                       |
//+------------------------------------------------------------------+
```

To quickly search and sort the collection lists, we have already created the functionality in the CSelect class selected in the
\\MQL5\\Include\\DoEasy\ **Services\** folder of service functions and classes of the **Select.mqh** file.

Add the methods for searching and sorting in the bar object lists.

Include
the Bar object class file to the listing:

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
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
#include "..\Objects\Series\Bar.mqh"
//+------------------------------------------------------------------+
```

Add the definitions of the methods for searching and sorting by the Bar object
properties to the class body:

```
//+------------------------------------------------------------------+
//| Class for sorting objects meeting the criterion                  |
//+------------------------------------------------------------------+
class CSelect
  {
private:
   //--- Method for comparing two values
   template<typename T>
   static bool       CompareValues(T value1,T value2,ENUM_COMPARER_TYPE mode);
public:
//+------------------------------------------------------------------+
//| Methods of working with orders                                   |
//+------------------------------------------------------------------+
   //--- Return the list of orders with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the order index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property);
   //--- Return the order index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with events                                   |
//+------------------------------------------------------------------+
   //--- Return the list of events with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the event index with the maximum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property);
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property);
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property);
   //--- Return the event index with the minimum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property);
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property);
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with accounts                                 |
//+------------------------------------------------------------------+
   //--- Return the list of accounts with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the event index with the maximum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property);
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property);
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property);
   //--- Return the event index with the minimum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property);
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property);
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with symbols                                  |
//+------------------------------------------------------------------+
   //--- Return the list of symbols with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the symbol index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property);
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property);
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property);
   //--- Return the symbol index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property);
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property);
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with pending requests                         |
//+------------------------------------------------------------------+
   //--- Return the list of pending requests with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the pending request index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_INTEGER property);
   static int        FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_DOUBLE property);
   static int        FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_STRING property);
   //--- Return the pending request index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindPendReqMin(CArrayObj *list_source,ENUM_PEND_REQ_PROP_INTEGER property);
   static int        FindPendReqMin(CArrayObj *list_source,ENUM_PEND_REQ_PROP_DOUBLE property);
   static int        FindPendReqMin(CArrayObj *list_source,ENUM_PEND_REQ_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with timeseries bars                          |
//+------------------------------------------------------------------+
   //--- Return the list of bars with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByBarProperty(CArrayObj *list_source,ENUM_BAR_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByBarProperty(CArrayObj *list_source,ENUM_BAR_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByBarProperty(CArrayObj *list_source,ENUM_BAR_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the pending request index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindBarMax(CArrayObj *list_source,ENUM_BAR_PROP_INTEGER property);
   static int        FindBarMax(CArrayObj *list_source,ENUM_BAR_PROP_DOUBLE property);
   static int        FindBarMax(CArrayObj *list_source,ENUM_BAR_PROP_STRING property);
   //--- Return the pending request index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindBarMin(CArrayObj *list_source,ENUM_BAR_PROP_INTEGER property);
   static int        FindBarMin(CArrayObj *list_source,ENUM_BAR_PROP_DOUBLE property);
   static int        FindBarMin(CArrayObj *list_source,ENUM_BAR_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

Implement added methods outside the class body:

```
//+------------------------------------------------------------------+
//| Methods of working with timeseries bar lists                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of bars with one integer                         |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByBarProperty(CArrayObj *list_source,ENUM_BAR_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CBar *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of bars with one real                            |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByBarProperty(CArrayObj *list_source,ENUM_BAR_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CBar *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of bars with one string                          |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByBarProperty(CArrayObj *list_source,ENUM_BAR_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CBar *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the listed bar index                                      |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindBarMax(CArrayObj *list_source,ENUM_BAR_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CBar *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CBar *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed bar index                                      |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindBarMax(CArrayObj *list_source,ENUM_BAR_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CBar *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CBar *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed bar index                                      |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindBarMax(CArrayObj *list_source,ENUM_BAR_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CBar *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CBar *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed bar index                                      |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindBarMin(CArrayObj* list_source,ENUM_BAR_PROP_INTEGER property)
  {
   int index=0;
   CBar *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CBar *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed bar index                                      |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindBarMin(CArrayObj* list_source,ENUM_BAR_PROP_DOUBLE property)
  {
   int index=0;
   CBar *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CBar *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed bar index                                      |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindBarMin(CArrayObj* list_source,ENUM_BAR_PROP_STRING property)
  {
   int index=0;
   CBar *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CBar *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

We have considered similar methods [in the third article of the previous\\
series](https://www.mql5.com/en/articles/5687#node01), so there is no point in dwelling on them here. If you have any questions, feel free to ask them in the comments.

In \\MQL5\\Include\\DoEasy\\Objects\ **Series\**, create the **Series.mqh** file of the CSeries class and connect
the CSelect class file to it together with the newly created "New Bar"
and "Bar" object classes:

```
//+------------------------------------------------------------------+
//|                                                       Series.mqh |
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
#include "NewBarObj.mqh"
#include "Bar.mqh"
//+------------------------------------------------------------------+
```

Now, add all the necessary class member variables and declare class methods in the class body:

```
//+------------------------------------------------------------------+
//|                                                       Series.mqh |
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
#include "NewBarObj.mqh"
#include "Bar.mqh"
//+------------------------------------------------------------------+
//| Timeseries class                                                 |
//+------------------------------------------------------------------+
class CSeries : public CObject
  {
private:
   ENUM_PROGRAM_TYPE m_program;                                         // Program type
   ENUM_TIMEFRAMES   m_timeframe;                                       // Timeframe
   string            m_symbol;                                          // Symbol
   uint              m_amount;                                          // Amount of applied timeseries data
   uint              m_bars;                                            // Number of bars in history by symbol and timeframe
   bool              m_sync;                                            // Synchronized data flag
   CArrayObj         m_list_series;                                     // Timeseries list
   CNewBarObj        m_new_bar_obj;                                     // "New bar" object
public:
//--- Return the timeseries list
   CArrayObj*        GetList(void)                                      { return &m_list_series;}
//--- Return the list of bars by selected (1) double, (2) integer and (3) string property fitting a compared condition
   CArrayObj*        GetList(ENUM_BAR_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByBarProperty(this.GetList(),property,value,mode); }
   CArrayObj*        GetList(ENUM_BAR_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByBarProperty(this.GetList(),property,value,mode); }
   CArrayObj*        GetList(ENUM_BAR_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL){ return CSelect::ByBarProperty(this.GetList(),property,value,mode); }

//--- Set (1) symbol and timeframe and (2) the number of applied timeseries data
   void              SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe);
   bool              SetAmountUsedData(const uint amount,const uint rates_total);

//--- Return (1) symbol, (2) timeframe, (3) number of applied timeseries data,
//--- (4) number of bars in the timeseries, the new bar flag with the (5) auto, (6) manual time management
   string            Symbol(void)                                          const { return this.m_symbol;                            }
   ENUM_TIMEFRAMES   Period(void)                                          const { return this.m_timeframe;                         }
   uint              AmountUsedData(void)                                  const { return this.m_amount;                            }
   uint              Bars(void)                                            const { return this.m_bars;                              }
   bool              IsNewBar(const datetime time)                               { return this.m_new_bar_obj.IsNewBar(time);        }
   bool              IsNewBarManual(const datetime time)                         { return this.m_new_bar_obj.IsNewBarManual(time);  }
//--- Return the bar object by index (1) in the list and (2) in the timeseries
   CBar             *GetBarByListIndex(const uint index);
   CBar             *GetBarBySeriesIndex(const uint index);
//--- Return (1) Open, (2) High, (3) Low, (4) Close, (5) time, (6) tick volume, (7) real volume, (8) bar spread by index
   double            Open(const uint index,const bool from_series=true);
   double            High(const uint index,const bool from_series=true);
   double            Low(const uint index,const bool from_series=true);
   double            Close(const uint index,const bool from_series=true);
   datetime          Time(const uint index,const bool from_series=true);
   long              TickVolume(const uint index,const bool from_series=true);
   long              RealVolume(const uint index,const bool from_series=true);
   int               Spread(const uint index,const bool from_series=true);

//--- Save the new bar time during the manual time management
   void              SaveNewBarTime(const datetime time)                         { this.m_new_bar_obj.SaveNewBarTime(time);         }
//--- Synchronize symbol data with server data
   bool              SyncData(const uint amount,const uint rates_total);
//--- (1) Create and (2) update the timeseries list
   int               Create(const uint amount=0);
   void              Refresh(const datetime time=0,
                             const double open=0,
                             const double high=0,
                             const double low=0,
                             const double close=0,
                             const long tick_volume=0,
                             const long volume=0,
                             const int spread=0);

//--- Constructors
                     CSeries(void);
                     CSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint amount=0);
  };
//+------------------------------------------------------------------+
```

The methods of receiving the list are present in all object
collection classes, the development of these methods was described [in the third\\
part of the previous series](https://www.mql5.com/en/articles/5687#node01).

Let’s go through the list of declared methods and analyze their implementation.

**The first class constructor** has no parameters and is used for creating the list for the current symbol and timeframe:

```
//+------------------------------------------------------------------+
//| Constructor 1 (current symbol and period timeseries)             |
//+------------------------------------------------------------------+
CSeries::CSeries(void) : m_bars(0),m_amount(0),m_sync(false)
  {
   this.m_program=(ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE);
   this.m_list_series.Clear();
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
   this.SetSymbolPeriod(::Symbol(),(ENUM_TIMEFRAMES)::Period());
  }
//+------------------------------------------------------------------+
```

The number of available timeseries bars, the number of bars stored in the list and
the flag of data synchronization with the server are reset in the initialization list.

The
program type is then set, the list of bar objects is cleared
and the flag of sorting by bar index is set for it. The
current symbol and timeframe are set for the list afterwards.

After creating the bar object list, make sure to set the number of
used bars for it using the SetAmountUsedData() or SyncData() method, which includes the SetAmountUsedData() method. In case of
indicators, make sure to pass **rates\_total** as a second parameter to the method.

**The second constructor of the class** features three inputs (symbol, timeframe and the bar object list size) and is used to
create the list for the specified symbol and timeframe considering the necessary history depth:

```
//+------------------------------------------------------------------+
//| Constructor 2 (specified symbol and period timeseries)           |
//+------------------------------------------------------------------+
CSeries::CSeries(const string symbol,const ENUM_TIMEFRAMES timeframe,const uint amount=0) : m_bars(0), m_amount(0),m_sync(false)
  {
   this.m_program=(ENUM_PROGRAM_TYPE)::MQLInfoInteger(MQL_PROGRAM_TYPE);
   this.m_list_series.Clear();
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
   this.SetSymbolPeriod(symbol,timeframe);
   this.m_sync=this.SetAmountUsedData(amount,0);
  }
//+------------------------------------------------------------------+
```

The number of available timeseries bars, the number of bars stored in the list and
the flag of data synchronization with the server are reset in the initialization list.

The
program type is then set, the list of bar objects is cleared
and the flag of sorting by bar index is set for it.

The
current symbol and timeframe are set for the list afterwards.

Finally, the
result of the method of setting the number of required bars for the SetAmountUsedData() bar object list is set for the synchronization flag.
The list receives the required history depth specified by the constructor's **amount** input.

After creating the list of bar
objects, make sure to check synchronization with the server using the SyncData() method from the program.

**The method for setting a symbol and timeframe:**

```
//+------------------------------------------------------------------+
//| Set a symbol and timeframe                                       |
//+------------------------------------------------------------------+
void CSeries::SetSymbolPeriod(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   this.m_symbol=(symbol==NULL || symbol==""   ? ::Symbol() : symbol);
   this.m_timeframe=(timeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)::Period() : timeframe);
   this.m_new_bar_obj.SetSymbol(this.m_symbol);
   this.m_new_bar_obj.SetPeriod(this.m_timeframe);
  }
//+------------------------------------------------------------------+
```

The method passes a symbol and timeframe and the validity of passed
values is checked. Eventually, either the current symbol and timeframe, or the ones passed to the method are set. Next, the
symbol and timeframe newly saved in the variables are set for the "New bar" object of the bar list.

**The method setting the number of used timeseries data for the bar object list**:

```
//+------------------------------------------------------------------+
//| Set the number of required data                                  |
//+------------------------------------------------------------------+
bool CSeries::SetAmountUsedData(const uint amount,const uint rates_total)
  {
//--- Set the number of available timeseries bars
   this.m_bars=
     (
      //--- If this is an indicator and the work is performed on the current symbol and timeframe,
      //--- add the rates_total value passed to the method,
      //--- otherwise, get the number from the environment
      this.m_program==PROGRAM_INDICATOR &&
      this.m_symbol==::Symbol() && this.m_timeframe==::Period() ? rates_total :
      ::Bars(this.m_symbol,this.m_timeframe)
     );
//--- If managed to set the number of available history, set the amount of data in the list:
   if(this.m_bars>0)
     {
      //--- if zero 'amount' value is passed,
      //--- use either the default value (1000 bars) or the number of available history bars - the least one of them
      //--- if non-zero 'amount' value is passed,
      //--- use either the 'amount' value or the number of available history bars - the least one of them
      this.m_amount=(amount==0 ? ::fmin(SERIES_DEFAULT_BARS_COUNT,this.m_bars) : ::fmin(amount,this.m_bars));
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the necessary amount of data for the bar object list
and the total number of bars of the current timeseries (for
indicators).

The program type is then checked and the source of the available history amount for the **m\_bars**
variable is defined — either from the value passed to the method
(for the indicator on the current symbol and timeframe) or from the environment.
Next, define what value should be set for the **m\_amount** variable based on the amount of available and required history.

**The method of synchronizing symbol and timeframe data with the server data**:

```
//+------------------------------------------------------------------+
//|Synchronize symbol and timeframe data with server data            |
//+------------------------------------------------------------------+
bool CSeries::SyncData(const uint amount,const uint rates_total)
  {
//--- If managed to obtain the available number of bars in the timeseries
//--- and return the size of the bar object list, return 'true'
   this.m_sync=this.SetAmountUsedData(amount,rates_total);
   if(this.m_sync)
      return true;

//--- Data is not yet synchronized with the server
//--- Create a pause object
   CPause *pause=new CPause();
   if(pause==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_PAUSE_OBJ));
      return false;
     }
//--- Set the pause duration of 16 milliseconds (PAUSE_FOR_SYNC_ATTEMPTS) and initialize the tick counter
   pause.SetWaitingMSC(PAUSE_FOR_SYNC_ATTEMPTS);
   pause.SetTimeBegin(0);
//--- Make five (ATTEMPTS_FOR_SYNC) attempts to obtain the available number of bars in the timeseries
//--- and set the bar object list size
   int attempts=0;
   while(attempts<ATTEMPTS_FOR_SYNC && !::IsStopped())
     {
      //--- If data is currently synchronized with the server
      if(::SeriesInfoInteger(this.m_symbol,this.m_timeframe,SERIES_SYNCHRONIZED))
        {
         //--- if managed to obtain the available number of bars in the timeseries
         //--- and set the size of the bar object list, break the loop
         this.m_sync=this.SetAmountUsedData(amount,rates_total);
         if(this.m_sync)
            break;
        }
      //--- Data is not yet synchronized.
      //--- If the pause of 16 ms is over
      if(pause.IsCompleted())
        {
         //--- set the new start of the next waiting for the pause object
         //--- and increase the attempt counter
         pause.SetTimeBegin(0);
         attempts++;
        }
     }
//--- Remove the pause object and return the m_sync value
   delete pause;
   return this.m_sync;
  }
//+------------------------------------------------------------------+
```

The method logic is described in the code comments. I believe, all is clear there.

The methods of creating and updating
the bar object list are commented in detail in the listing. Let's consider them in their entirety in order not to take up much space for
their description. If you have any questions regarding these methods, feel free to ask them in the comments:

```
//+------------------------------------------------------------------+
//| Create the timeseries list                                       |
//+------------------------------------------------------------------+
int CSeries::Create(const uint amount=0)
  {
//--- If the required history depth is not set for the list yet,
//--- display the appropriate message and return zero,
   if(this.m_amount==0)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_TEXT_FIRS_SET_AMOUNT_DATA));
      return 0;
     }
//--- otherwise, if the passed 'amount' value exceeds zero and is not equal to the one already set,
//--- while being lower than the available bar number,
//--- set the new value of the required history depth for the list
   else if(amount>0 && this.m_amount!=amount && amount<this.m_bars)
     {
      //--- If failed to set a new value, return zero
      if(!this.SetAmountUsedData(amount,0))
         return 0;
     }
//--- For the rates[] array we are to receive historical data to,
//--- set the flag of direction like in the timeseries,
//--- clear the bar object list and set the flag of sorting by bar index
   MqlRates rates[];
   ::ArraySetAsSeries(rates,true);
   this.m_list_series.Clear();
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
   ::ResetLastError();
//--- Get historical data of the MqlRates structure to the rates[] array starting from the current bar in the amount of m_amount,
//--- if failed to get data, display the appropriate message and return zero
   int copied=::CopyRates(this.m_symbol,this.m_timeframe,0,this.m_amount,rates),err=ERR_SUCCESS;
   if(copied<1)
     {
      err=::GetLastError();
      ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_GET_SERIES_DATA)," ",this.m_symbol," ",TimeframeDescription(this.m_timeframe),". ",
                   CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err));
      return 0;
     }

//--- Historical data is received in the rates[] array
//--- In the rates[] array loop,
   for(int i=0; i<copied; i++)
     {
      //--- create a new bar object out of the current MqlRates structure by the loop index
      ::ResetLastError();
      CBar* bar=new CBar(this.m_symbol,this.m_timeframe,i,rates[i]);
      if(bar==NULL)
        {
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_BAR_OBJ),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(::GetLastError()));
         continue;
        }
      //--- If failed to add bar object to the list,
      //--- display the appropriate message with the error description in the journal
      if(!this.m_list_series.Add(bar))
        {
         err=::GetLastError();
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_BAR_FAILED_ADD_TO_LIST)," ",bar.Header(),". ",
                      CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(err),CMessage::Retcode(err));
        }
     }
//--- Return the size of the created bar object list
   return this.m_list_series.Total();
  }
//+------------------------------------------------------------------+
//| Update timeseries list and data                                  |
//+------------------------------------------------------------------+
void CSeries::Refresh(const datetime time=0,
                      const double open=0,
                      const double high=0,
                      const double low=0,
                      const double close=0,
                      const long tick_volume=0,
                      const long volume=0,
                      const int spread=0)
  {
   MqlRates rates[1];
//--- Set the flag of sorting the list of bars by index
   this.m_list_series.Sort(SORT_BY_BAR_INDEX);
//--- If a new bar is present on a symbol and period,
   if(this.IsNewBarManual(time))
     {
      //--- create a new bar object and add it to the end of the list
      CBar *new_bar=new CBar(this.m_symbol,this.m_timeframe,0);
      if(new_bar==NULL)
         return;
      if(!this.m_list_series.Add(new_bar))
        {
         delete new_bar;
         return;
        }
      //--- if the specified timeseries size exceeds one bar, remove the earliest bar
      if(this.m_list_series.Total()>1)
         this.m_list_series.Delete(0);
      //--- save the new bar time as the previous one for the subsequent new bar check
      this.SaveNewBarTime(time);
     }
//--- Get the index of the last bar in the list and the object bar by the index
   int index=this.m_list_series.Total()-1;
   CBar *bar=this.m_list_series.At(index);
//--- if the work is performed in an indicator and the timeseries belongs to the current symbol and timeframe,
//--- copy price parameters (passed to the method from the outside) to the bar price structure
   int copied=1;
   if(this.m_program==PROGRAM_INDICATOR && this.m_symbol==::Symbol() && this.m_timeframe==::Period())
     {
      rates[0].time=time;
      rates[0].open=open;
      rates[0].high=high;
      rates[0].low=low;
      rates[0].close=close;
      rates[0].tick_volume=tick_volume;
      rates[0].real_volume=volume;
      rates[0].spread=spread;
     }
//--- otherwise, get data to the bar price structure from the environment
   else
      copied=::CopyRates(this.m_symbol,this.m_timeframe,0,1,rates);
//--- If the prices are obtained, set the new properties from the price structure for the bar object
   if(copied==1)
      bar.SetProperties(rates[0]);
  }
//+------------------------------------------------------------------+
```

To create the timeseries list, first, set the size of the required history and get the flag of synchronization with the server using the
SyncData() method. This is followed by calling the Create() method. To update the bar object list data, make sure to call the Refresh()
method on each tick. The method detects a new bar opening on its own and adds the new bar object to the timeseries list. The earliest bar object
is removed from the list of bar objects so that the list size always remains at the level set by the SyncData() method.

We need to get Bar objects from the timeseries list to manage the data. If the flag of sorting by index (SORT\_BY\_BAR\_INDEX) is set for the
timeseries list, the sequence of bar objects in the list corresponds to the location of real bars in the timeseries. But if we set another
sorting flag for the list, the order of objects in the list no longer corresponds to the location of real bars in the timeseries — they are
arranged in ascending order of the property the list is sorted by. Therefore, we have two methods to select objects from the bar object list:
the method returning an object bar by its index in the timeseries and the method returning an object bar by its index in the bar object list.

Let's consider these two methods.

**The method returning the bar object by its index in the bar object list**:

```
//+------------------------------------------------------------------+
//| Return the bar object by index in the list                       |
//+------------------------------------------------------------------+
CBar *CSeries::GetBarByListIndex(const uint index)
  {
   return this.m_list_series.At(this.m_list_series.Total()-index-1);
  }
//+------------------------------------------------------------------+
```

The index of the necessary bar object is passed to the method. The passed index implies the same direction as in the timeseries: the zero index
indicates the last object in the list. However, the objects in the list are saved from the zero index up to list.Total()-1, i.e. in order to get
the latest bar in the list, request it by the list.Total()-1 index, while in order to get the rightmost bar on the chart, request it by index 0
(reverse indexing).

Therefore, the index is re-calculated in the appropriate method: the number of the passed index-1 is subtracted
from the list size and the bar object is returned based on the calculated index according to the indexing direction as in the timeseries.

**The method returning a bar object by its index in the timeseries**:

```
//+------------------------------------------------------------------+
//| Return the bar object by index in the timeseries                 |
//+------------------------------------------------------------------+
CBar *CSeries::GetBarBySeriesIndex(const uint index)
  {
   CArrayObj *list=this.GetList(BAR_PROP_INDEX,index);
   return(list==NULL || list.Total()==0 ? NULL : list.At(0));
  }
//+------------------------------------------------------------------+
```

The index of the necessary bar object is passed to the method. The passed index implies the same direction as in the timeseries.

To get the
object with the same bar index as in the timeseries, it should be selected by the
BAR\_PROP\_INDEX property. If the list of bar objects features the bar with the necessary index, **list**
features a single object that should be returned.

If there is
no such object, NULL is returned. In case of an error, both of these methods return NULL.

**The methods returning basic properties of the bar object from the list of bar objects by the index**:

```
//+------------------------------------------------------------------+
//| Return bar's Open by the index                                   |
//+------------------------------------------------------------------+
double CSeries::Open(const uint index,const bool from_series=true)
  {
   CBar *bar=(from_series ? this.GetBarBySeriesIndex(index) : this.GetBarByListIndex(index));
   return(bar!=NULL ? bar.Open() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar's High by the timeseries index or the list of bars    |
//+------------------------------------------------------------------+
double CSeries::High(const uint index,const bool from_series=true)
  {
   CBar *bar=(from_series ? this.GetBarBySeriesIndex(index) : this.GetBarByListIndex(index));
   return(bar!=NULL ? bar.High() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar's Low by the timeseries index or the list of bars     |
//+------------------------------------------------------------------+
double CSeries::Low(const uint index,const bool from_series=true)
  {
   CBar *bar=(from_series ? this.GetBarBySeriesIndex(index) : this.GetBarByListIndex(index));
   return(bar!=NULL ? bar.Low() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar's Close by the timeseries index or the list of bars   |
//+------------------------------------------------------------------+
double CSeries::Close(const uint index,const bool from_series=true)
  {
   CBar *bar=(from_series ? this.GetBarBySeriesIndex(index) : this.GetBarByListIndex(index));
   return(bar!=NULL ? bar.Close() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar time by the timeseries index or the list of bars      |
//+------------------------------------------------------------------+
datetime CSeries::Time(const uint index,const bool from_series=true)
  {
   CBar *bar=(from_series ? this.GetBarBySeriesIndex(index) : this.GetBarByListIndex(index));
   return(bar!=NULL ? bar.Time() : 0);
  }
//+-------------------------------------------------------------------+
//|Return bar tick volume by the timeseries index or the list of bars |
//+-------------------------------------------------------------------+
long CSeries::TickVolume(const uint index,const bool from_series=true)
  {
   CBar *bar=(from_series ? this.GetBarBySeriesIndex(index) : this.GetBarByListIndex(index));
   return(bar!=NULL ? bar.VolumeTick() : WRONG_VALUE);
  }
//+--------------------------------------------------------------------+
//|Return bar real volume by the timeseries index or the list of bars  |
//+--------------------------------------------------------------------+
long CSeries::RealVolume(const uint index,const bool from_series=true)
  {
   CBar *bar=(from_series ? this.GetBarBySeriesIndex(index) : this.GetBarByListIndex(index));
   return(bar!=NULL ? bar.VolumeReal() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//| Return bar spread by the timeseries index or the list of bars    |
//+------------------------------------------------------------------+
int CSeries::Spread(const uint index,const bool from_series=true)
  {
   CBar *bar=(from_series ? this.GetBarBySeriesIndex(index) : this.GetBarByListIndex(index));
   return(bar!=NULL ? bar.Spread() : WRONG_VALUE);
  }
//+------------------------------------------------------------------+
```

The methods receive the bar index and the
flag indicating that the requested index corresponds (true) the indexing direction as in
the timeseries.

Based on the flag value,
get the bar object either using the GetBarBySeriesIndex() or GetBarByListIndex() method.
Then return the value of the property requested in the method.

The remaining CSeries class methods not considered here are used to set or return class member variable values.

To check what we have done in the current article, we need to let the external program know about the created classes. To achieve this, simply include
the CSeries class file to the CEngine library main object file:

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
#include "TradingControl.mqh"
#include "Objects\Series\Series.mqh"
//+------------------------------------------------------------------+
```

Now we are able to define the variable of the CSeries class type in the test EA to create and use timeseries lists with the specified number of bar
objects (in the current implementation, the lists are used for testing only). Let's do this now.

### Testing

To perform the test, use [the EA from the last article of the previous series](https://www.mql5.com/en/articles/7569#node03)
and save it in \\MQL5\\Experts\\TestDoEasy\ **Part35\** under the name **TestDoEasyPart35.mq5**.

Create two variables (for testing) of the CSeries class object — for M1 (2 bars) and the current timeframe (10 bars). Set all the
necessary parameters in the OnInit() handler and display three lists:

1. the list of all bars of the current timeseries sorted by candle size (from bar's High to Low) — short descriptions of bar objects;
2. the list of all bars of the current timeseries sorted by bar indices (from bar 0 to 9) — short descriptions of bar objects;
3. the list of all bar object properties corresponding to the appropriate index 1 (previous bar) of the current timeseries — full list of
    all bar properties.


In the OnTick() handler, update both timeseries on each tick and display the entry about opening a new bar on each of the two timeseries of
these two timeseries lists — М1 and the current one. Apart from the journal entry, play the standard "news.wav" sound when opening a new bar
for the current timeseries.

**In the list of global EA variables, define the two variables of the CSeries class type** — the
timeseries list variable of the current timeframe and the variable of
the M1 timeseries list:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart35.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
//--- enums
enum ENUM_BUTTONS
  {
   BUTT_BUY,
   BUTT_BUY_LIMIT,
   BUTT_BUY_STOP,
   BUTT_BUY_STOP_LIMIT,
   BUTT_CLOSE_BUY,
   BUTT_CLOSE_BUY2,
   BUTT_CLOSE_BUY_BY_SELL,
   BUTT_SELL,
   BUTT_SELL_LIMIT,
   BUTT_SELL_STOP,
   BUTT_SELL_STOP_LIMIT,
   BUTT_CLOSE_SELL,
   BUTT_CLOSE_SELL2,
   BUTT_CLOSE_SELL_BY_BUY,
   BUTT_DELETE_PENDING,
   BUTT_CLOSE_ALL,
   BUTT_SET_STOP_LOSS,
   BUTT_SET_TAKE_PROFIT,
   BUTT_PROFIT_WITHDRAWAL,
   BUTT_TRAILING_ALL
  };
#define TOTAL_BUTT   (20)
//--- structures
struct SDataButt
  {
   string      name;
   string      text;
  };
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
sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;   // Mode of used symbols list
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   bool              InpUseSounds         =  true; // Use sounds
//--- global variables
CEngine        engine;
CSeries        series;
CSeries        series_m1;
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ushort         magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           distance_pending_request;
uint           bars_delay_pending_request;
uint           slippage;
bool           trailing_on;
bool           pressed_pending_buy;
bool           pressed_pending_buy_limit;
bool           pressed_pending_buy_stop;
bool           pressed_pending_buy_stoplimit;
bool           pressed_pending_close_buy;
bool           pressed_pending_close_buy2;
bool           pressed_pending_close_buy_by_sell;
bool           pressed_pending_sell;
bool           pressed_pending_sell_limit;
bool           pressed_pending_sell_stop;
bool           pressed_pending_sell_stoplimit;
bool           pressed_pending_close_sell;
bool           pressed_pending_close_sell2;
bool           pressed_pending_close_sell_by_buy;
bool           pressed_pending_delete_all;
bool           pressed_pending_close_all;
bool           pressed_pending_sl;
bool           pressed_pending_tp;
double         trailing_stop;
double         trailing_step;
uint           trailing_start;
uint           stoploss_to_modify;
uint           takeprofit_to_modify;
int            used_symbols_mode;
string         used_symbols;
string         array_used_symbols[];
bool           testing;
uchar          group1;
uchar          group2;
double         g_point;
int            g_digits;
//+------------------------------------------------------------------+
```

In the EA's OnInit() handler, set the necessary properties for both
variables of timeseries objects and display the entire data on the created
list of the current timeframe's bar objects. For М1, simply display the
message about the successful list creation:

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
   testing=engine.IsTester();
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
   distance_pending_request=(InpDistancePReq<5 ? 5 : InpDistancePReq);
   bars_delay_pending_request=(InpBarsDelayPReq<1 ? 1 : InpBarsDelayPReq);
   g_point=SymbolInfoDouble(NULL,SYMBOL_POINT);
   g_digits=(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS);
//--- Initialize random group numbers
   group1=0;
   group2=0;
   srand(GetTickCount());

//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Check and remove remaining EA graphical objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);
//--- Reset states of the buttons for working using pending requests
   for(int i=0;i<14;i++)
     {
      ButtonState(butt_data[i].name+"_PRICE",false);
      ButtonState(butt_data[i].name+"_TIME",false);
     }

//--- Check playing a standard sound by macro substitution and a custom sound by description
   engine.PlaySoundByDescription(SND_OK);
   Sleep(600);
   engine.PlaySoundByDescription(TextByLanguage("Звук упавшей монетки 2","Falling coin 2"));

//--- Set the M1 timeseries object parameters
   series_m1.SetSymbolPeriod(Symbol(),PERIOD_M1);
//--- If symbol and M1 timeframe data are synchronized
   if(series_m1.SyncData(2,0))
     {
      //--- create the timeseries list of two bars (the current and previous ones),
      //--- if the timeseries is created, display the appropriate message in the journal
      int total=series_m1.Create(2);
      if(total>0)
         Print(TextByLanguage("Создана таймсерия М1 с размером ","Created timeseries M1 with size "),(string)total);
     }
//--- Check filling price data on the current symbol and timeframe
   series.SetSymbolPeriod(Symbol(),(ENUM_TIMEFRAMES)Period());
//--- If symbol and the current timeframe data are synchronized
   if(series.SyncData(10,0))
     {
      //--- create the timeseries list of ten bars (bars 0 - 9),
      //--- if the timeseries is created, display three lists:
      //--- 1. the list of bars sorted by candle size (from bars' High to Low)
      //--- 2. the list of bars sorted by bar indices (according to their sequence in the timeseries)
      //--- 3. the full list of all properties of the previous bar object (bar properties with the timeseries index of 1)
      int total=series.Create(10);
      if(total>0)
        {
         CArrayObj *list=series.GetList();
         CBar *bar=NULL;
         //--- Display short properties of the bar list by the candle size
         Print("\n",TextByLanguage("Бары, сортированные по размеру свечи от High до Low:","Bars sorted by candle size from High to Low:"));
         list.Sort(SORT_BY_BAR_CANDLE_SIZE);
         for(int i=0;i<total;i++)
           {
            bar=series.GetBarByListIndex(i);
            if(bar==NULL)
               continue;
            bar.PrintShort();
           }
         //--- Display short properties of the bar list by the timeseries index
         Print("\n",TextByLanguage("Бары, сортированные по индексу таймсерии:","Bars sorted by timeseries index:"));
         list.Sort(SORT_BY_BAR_INDEX);
         for(int i=0;i<total;i++)
           {
            bar=series.GetBarByListIndex(i);
            if(bar==NULL)
               continue;
            bar.PrintShort();
           }
         //--- Display all bar 1 properties
         Print("");
         list=CSelect::ByBarProperty(list,BAR_PROP_INDEX,1,EQUAL);
         if(list.Total()==1)
           {
            bar=list.At(0);
            bar.Print();
           }
        }
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

In the OnTick() handler, update the timeseries lists of CSeries class objects
on each tick, while during the "New bar" event, display
the messages about this event for each of the two lists. Additionally, play the
sound of the new bar opening event on the current timeframe:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();       // Working in the timer
      PressButtonsControl();  // Button pressing control
      EventsHandling();       // Working with events
     }
//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();    // Trailing positions
      TrailingOrders();       // Trailing of pending orders
     }
//--- Check the update of the current and M1 timeseries
   series.Refresh();
   if(series.IsNewBar(0))
     {
      Print("New bar on ",series.Symbol()," ",TimeframeDescription(series.Period())," ",TimeToString(series.Time(0)));
      engine.PlaySoundByDescription(SND_NEWS);
     }
   series_m1.Refresh();
   if(series_m1.IsNewBar(0))
     {
      Print("New bar on ",series_m1.Symbol()," ",TimeframeDescription(series_m1.Period())," ",TimeToString(series_m1.Time(0)));
     }
  }
//+------------------------------------------------------------------+
```

Compile the EA and launch it on a symbol chart.

First, the journal displays entries on the
start of creating the M1 timeseries list of two bars, then it displays the
list of bars sorted by candle size, next, it shows the list of bars sorted in
the order of bar indices in the timeseries, and finally, all object bar
properties with the index of 1 in the timeseries:

```
Account 15585535: Artyom Trishkin (MetaQuotes Software Corp.) 9999.40 USD, 1:100, Demo account MetaTrader 5
Work only with the current symbol. The number of symbols used: 1
Created timeseries M1 with size 2

Bars, sorted by size candle from High to Low:
Bar "EURUSD" H1[2]: 2020.02.12 10:00:00, O: 1.09145, H: 1.09255, L: 1.09116, C: 1.09215, V: 2498, Bullish bar
Bar "EURUSD" H1[3]: 2020.02.12 09:00:00, O: 1.09057, H: 1.09150, L: 1.09022, C: 1.09147, V: 1773, Bullish bar
Bar "EURUSD" H1[1]: 2020.02.12 11:00:00, O: 1.09215, H: 1.09232, L: 1.09114, C: 1.09202, V: 1753, Bearish bar
Bar "EURUSD" H1[9]: 2020.02.12 03:00:00, O: 1.09130, H: 1.09197, L: 1.09129, C: 1.09183, V: 1042, Bullish bar
Bar "EURUSD" H1[4]: 2020.02.12 08:00:00, O: 1.09108, H: 1.09108, L: 1.09050, C: 1.09057, V: 581, Bearish bar
Bar "EURUSD" H1[8]: 2020.02.12 04:00:00, O: 1.09183, H: 1.09197, L: 1.09146, C: 1.09159, V: 697, Bearish bar
Bar "EURUSD" H1[5]: 2020.02.12 07:00:00, O: 1.09122, H: 1.09143, L: 1.09096, C: 1.09108, V: 591, Bearish bar
Bar "EURUSD" H1[6]: 2020.02.12 06:00:00, O: 1.09152, H: 1.09159, L: 1.09121, C: 1.09122, V: 366, Bearish bar
Bar "EURUSD" H1[7]: 2020.02.12 05:00:00, O: 1.09159, H: 1.09177, L: 1.09149, C: 1.09152, V: 416, Bearish bar
Bar "EURUSD" H1[0]: 2020.02.12 12:00:00, O: 1.09202, H: 1.09204, L: 1.09181, C: 1.09184, V: 63, Bearish bar

Bars, sorted by timeseries index:
Bar "EURUSD" H1[9]: 2020.02.12 03:00:00, O: 1.09130, H: 1.09197, L: 1.09129, C: 1.09183, V: 1042, Bullish bar
Bar "EURUSD" H1[8]: 2020.02.12 04:00:00, O: 1.09183, H: 1.09197, L: 1.09146, C: 1.09159, V: 697, Bearish bar
Bar "EURUSD" H1[7]: 2020.02.12 05:00:00, O: 1.09159, H: 1.09177, L: 1.09149, C: 1.09152, V: 416, Bearish bar
Bar "EURUSD" H1[6]: 2020.02.12 06:00:00, O: 1.09152, H: 1.09159, L: 1.09121, C: 1.09122, V: 366, Bearish bar
Bar "EURUSD" H1[5]: 2020.02.12 07:00:00, O: 1.09122, H: 1.09143, L: 1.09096, C: 1.09108, V: 591, Bearish bar
Bar "EURUSD" H1[4]: 2020.02.12 08:00:00, O: 1.09108, H: 1.09108, L: 1.09050, C: 1.09057, V: 581, Bearish bar
Bar "EURUSD" H1[3]: 2020.02.12 09:00:00, O: 1.09057, H: 1.09150, L: 1.09022, C: 1.09147, V: 1773, Bullish bar
Bar "EURUSD" H1[2]: 2020.02.12 10:00:00, O: 1.09145, H: 1.09255, L: 1.09116, C: 1.09215, V: 2498, Bullish bar
Bar "EURUSD" H1[1]: 2020.02.12 11:00:00, O: 1.09215, H: 1.09232, L: 1.09114, C: 1.09202, V: 1753, Bearish bar
Bar "EURUSD" H1[0]: 2020.02.12 12:00:00, O: 1.09202, H: 1.09204, L: 1.09181, C: 1.09184, V: 63, Bearish bar

============= The beginning of the event parameter list (Bar "EURUSD" H1[1]) =============
Timeseries index: 1
Type: Bearish bar
Timeframe: H1
Spread: 1
Tick volume: 1753
Real volume: 0
Period start time: 2020.02.12 11:00:00
Sequence day number in the year: 042
Year: 2020
Month: February
Day of week: Wednesday
Day od month: 12
Hour: 11
Minute: 00
------
Price open: 1.09215
Highest price for the period: 1.09232
Lowest price for the period: 1.09114
Price close: 1.09202
Candle size: 0.00118
Candle body size: 0.00013
Top of the candle body: 1.09215
Bottom of the candle body: 1.09202
Candle upper shadow size: 0.00017
Candle lower shadow size: 0.00088
------
Symbol: "EURUSD"
============= End of the parameter list (Bar "EURUSD" H1[1]) =============
```

Now let's launch the EA in tester visual mode on М5 and have a look at the tester journal messages about opening new bars:

![](https://c.mql5.com/2/38/X8RG2DFd4Y.gif)

As we can see, every fifth message is about opening a new bar on М5 intermittent with the messages about opening a new bar on М1.

### What's next?

In the next article, we are going to create the collection class of bar lists.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7594#node00)

**Articles of the previous series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part\\
2\. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders\\
and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part\\
6\. Netting account events](https://www.mql5.com/en/articles/6383)

[Part 7. StopLimit order activation events, preparing\\
the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part 8. Order and\\
position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 — Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and activating pending\\
orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure events](https://www.mql5.com/en/articles/6921)

[Part 12. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part\\
13\. Account object events](https://www.mql5.com/en/articles/6995)

[Part 14. Symbol object](https://www.mql5.com/en/articles/7014)

[Part\\
15\. Symbol object collection](https://www.mql5.com/en/articles/7041)

[Part 16. Symbol collection events](https://www.mql5.com/en/articles/7071)

[Part 17. Interactivity of library objects](https://www.mql5.com/en/articles/7124)

[Part\\
18\. Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

[Part 19. Class of\\
library messages](https://www.mql5.com/en/articles/7176)

[Part 20. Creating and storing program resources](https://www.mql5.com/en/articles/7195)

[Part 21. Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

[Part\\
22\. Trading classes - Base trading class, verification of limitations](https://www.mql5.com/en/articles/7258)

[Part 23.\\
Trading classes - Base trading class, verification of valid parameters](https://www.mql5.com/en/articles/7286)

[Part 24.\\
Trading classes - Base trading class, auto correction of invalid parameters](https://www.mql5.com/en/articles/7326)

[Part\\
25\. Trading classes - Base trading class, handling errors returned by the trade server](https://www.mql5.com/en/articles/7365)

[Part\\
26\. Working with pending trading requests - First implementation (opening positions)](https://www.mql5.com/en/articles/7394)

[Part\\
27\. Working with pending trading requests - Placing pending orders](https://www.mql5.com/en/articles/7418)

[Part 28.\\
Working with pending trading requests - Closure, removal and modification](https://www.mql5.com/en/articles/7438)

[Part\\
29\. Working with pending trading requests - request object classes](https://www.mql5.com/en/articles/7454)

[Part 30.\\
Pending trading requests - managing request objects](https://www.mql5.com/en/articles/7481)

[Part 31. Pending trading\\
requests - opening positions under certain conditions](https://www.mql5.com/en/articles/7521)

[Part 32. Pending trading\\
requests - placing pending orders under certain conditions](https://www.mql5.com/en/articles/7536)

[Part 33. Pending\\
trading requests - closing positions (full, partial or by an opposite one) under certain conditions](https://www.mql5.com/en/articles/7554)

[Part\\
34\. Pending trading requests - removing and modifying orders and positions under certain conditions](https://www.mql5.com/en/articles/7569)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7594](https://www.mql5.com/ru/articles/7594)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7594.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7594/mql5.zip "Download MQL5.zip")(3683.65 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7594/mql4.zip "Download MQL4.zip")(3683.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/342208)**
(8)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
25 Jun 2020 at 20:56

**Rosimir Mateev:**

A very important parameter of each bar is the time of the High price and the time of the Low price, or at least a flag indicating which of the two High or Low prices occurred first in time. This information can be obtained by analysing bars from a lower timeframe or directly from M1. Please add this functionality.

Such functionality can be done by yourself - there is all data on the lower timeframes, and there is all data on their bars. This is not the last article, and in the next ones there is a complete list of all timeframes of any symbol - so you should not overload the already existing one with unnecessary functionality. Especially since it will cause an increase in timeseries update time. Therefore, it is better to get data from already available ones on request than to do such analysis where it is not required.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
4 Sep 2020 at 17:20

Hi Artyom -- using DoEasy instead of the Standard Library, what is the best approach to create a class for detecting different candle patterns, similar to what is **CCandlePattern** as in [https://www.mql5.com/en/code/316](https://www.mql5.com/en/code/316)?

I am asking to avoid doing something that be completely contrary to the future direction you're taking with this library... I was thinking to either (1) _modify_ or _extend_ classes **CBar** & **CSeries** to do what I need; or (2) or use the DoEasy events to detect new bars on the symbols/periods I am interested in, from there implement my code to check for my candle patterns and use DoEasy again to place orders or open positions. Maybe option #2 is cleaner and less risky relative to your future developments, but let me know if you have another suggestion please.

**acandlepatterns.mqh**

classCCandlePattern :publicCExpertSignal

{

protected:

// ...

//\-\-\- methods, used for check of the candlestick pattern formation

double            AvgBody(int ind);

double            MA(int ind)                const { return(m\_MA.Main(ind));             }

double            Open(int ind)              const { return(m\_open.GetData(ind));        }

double            High(int ind)              const { return(m\_high.GetData(ind));        }

double            Low(int ind)               const { return(m\_low.GetData(ind));         }

double            Close(int ind)             const { return(m\_close.GetData(ind));       }

double            CloseAvg(int ind)          const { return(MA(ind));                    }

double            MidPoint(int ind)          const { return(0.5\*(High(ind)+Low(ind)));   }

double            MidOpenClose(int ind)      const { return(0.5\*(Open(ind)+Close(ind))); }

//\-\-\- methods for checking of candlestick patterns

bool              CheckPatternThreeBlackCrows();

bool              CheckPatternThreeWhiteSoldiers();

bool              CheckPatternDarkCloudCover();

bool              CheckPatternPiercingLine();

bool              CheckPatternMorningDoji();

bool              CheckPatternEveningDoji();

bool              CheckPatternBearishEngulfing();

bool              CheckPatternBullishEngulfing();

bool              CheckPatternEveningStar();

bool              CheckPatternMorningStar();

bool              CheckPatternHammer();

bool              CheckPatternHangingMan();

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
4 Sep 2020 at 19:12

**ddiall :**

Hi Artyom --  using DoEasy instead of the Standard Library,  what is the best approach to create a class for detecting different candle patterns, similar to what is **CCandlePattern** as in [https://www.mql5.com/en/code/316](https://www.mql5.com/en/code/316) ?

I am asking to avoid doing something that be completely contrary to the future direction you're taking with this library... I was thinking to either (1) _modify_ or _extend_  classes **CBar** & **CSeries** to do what I need; or (2)  or use the DoEasy events to detect new bars on the symbols/periods I am interested in, from there implement my code to check for my candle patterns and use DoEasy again to place orders or open positions. Maybe option #2 is cleaner and less risky relative to your future developments, but let me know if you have another suggestion please.

**acandlepatterns.mqh**

classCCandlePattern :publicCExpertSignal

{

protected:

// ...

//\-\-\- methods, used for check of the candlestick pattern formation

double             AvgBody( int  ind);

double             MA( int  ind)                 const  {  return (m\_MA.Main(ind));             }

double             Open( int  ind)               const  {  return (m\_open.GetData(ind));        }

double             High( int  ind)               const  {  return (m\_high.GetData(ind));        }

double             Low( int  ind)                const  {  return (m\_low.GetData(ind));         }

double             Close( int  ind)              const  {  return (m\_close.GetData(ind));       }

double             CloseAvg( int  ind)           const  {  return ( MA(ind) );                    }

double             MidPoint( int  ind)           const  {  return ( 0.5 \*(High(ind)+Low(ind)) );   }

double             MidOpenClose( int  ind)       const  {  return ( 0.5 \*(Open(ind)+Close(ind)) ); }

//\-\-\- methods for checking of candlestick patterns

bool               CheckPatternThreeBlackCrows();

bool               CheckPatternThreeWhiteSoldiers();

bool               CheckPatternDarkCloudCover();

bool               CheckPatternPiercingLine();

bool               CheckPatternMorningDoji();

bool               CheckPatternEveningDoji();

bool               CheckPatternBearishEngulfing();

bool               CheckPatternBullishEngulfing();

bool               CheckPatternEveningStar();

bool               CheckPatternMorningStar();

bool               CheckPatternHammer();

bool               CheckPatternHangingMan();

The library will contain a class for searching for PriceAction candlestick patterns and various Japanese candlestick formations.

I find it most prudent to add a new property to the CBar object class. Accordingly, the collection of timeseries will be able to search for the desired candles and patterns.

So it will be in the library.

For now, you can search for the patterns and candles you need in the lists of the collection of timeseries - there are all the bars, and you can use them to determine the pattern.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
4 Sep 2020 at 21:31

**Artyom Trishkin:**

The library will contain a class for searching for PriceAction candlestick patterns and various Japanese candlestick formations.

I find it most prudent to add a new property to the CBar object class. Accordingly, the collection of timeseries will be able to search for the desired candles and patterns.

So it will be in the library.

For now, you can search for the patterns and candles you need in the lists of the collection of timeseries - there are all the bars, and you can use them to determine the pattern.

That sounds great! Any idea when you will start implementing all this good stuff? Are you interested in any code contributions?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
4 Sep 2020 at 22:42

**Dima Diall :**

That sounds great! Any idea when you will start implementing all this good stuff? Are you interested in any code contributions?

As the time comes :)

![Forecasting Time Series (Part 1): Empirical Mode Decomposition (EMD) Method](https://c.mql5.com/2/38/mql5-avatar-emd.png)[Forecasting Time Series (Part 1): Empirical Mode Decomposition (EMD) Method](https://www.mql5.com/en/articles/7601)

This article deals with the theory and practical use of the algorithm for forecasting time series, based on the empirical decomposition mode. It proposes the MQL implementation of this method and presents test indicators and Expert Advisors.

![Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://c.mql5.com/2/37/Article_Logo__3.png)[Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://www.mql5.com/en/articles/7528)

In the previous article, we created the application framework, which we will use as the basis for all further work. In this part, we will proceed with the development: we will create the visual part of the application and will configure basic interaction of interface elements.

![Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__1.png)[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)

In this article, we will consider combining the lists of bar objects for each used symbol period into a single symbol timeseries object. Thus, each symbol will have an object storing the lists of all used symbol timeseries periods.

![Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://c.mql5.com/2/37/kisspng-computer-icons-application-programming-interface-c-database-administrator-icon-free-download__1.png)[Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://www.mql5.com/en/articles/7495)

In the previous part, we considered the implementation of the MySQL connector. In this article, we will consider its application by implementing the service for collecting signal properties and the program for viewing their changes over time. The implemented example has practical sense if users need to observe changes in properties that are not displayed on the signal's web page.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fchrvxclnurpiqjxciljqxqvgrpvkmtb&ssn=1769251943076044646&ssn_dr=0&ssn_sr=0&fv_date=1769251943&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7594&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2035)%3A%20Bar%20object%20and%20symbol%20timeseries%20list%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17692519438021596&fz_uniq=5083175902054979175&sv=2552)

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