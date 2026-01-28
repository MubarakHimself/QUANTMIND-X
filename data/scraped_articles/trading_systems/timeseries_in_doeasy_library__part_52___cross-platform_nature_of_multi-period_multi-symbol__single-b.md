---
title: Timeseries in DoEasy library (part 52): Cross-platform nature of multi-period multi-symbol  single-buffer standard indicators
url: https://www.mql5.com/en/articles/8399
categories: Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T19:34:24.507377
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=deabwegrapahisplivhvpcmoaisppoki&ssn=1769186062068437459&ssn_dr=0&ssn_sr=0&fv_date=1769186062&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8399&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2052)%3A%20Cross-platform%20nature%20of%20multi-period%20multi-symbol%20single-buffer%20standard%20indicators%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918606210018217&fz_uniq=5070393615525549325&sv=2552)

MetaTrader 5 / Examples


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8399#node01)
- [Improving library classes](https://www.mql5.com/en/articles/8399#node02)
- [Testing](https://www.mql5.com/en/articles/8399#node03)
- [What's next?](https://www.mql5.com/en/articles/8399#node04)


### Concept

Starting with [article 39](https://www.mql5.com/en/articles/7724), step by step we have created functionality for constructing own multi-symbol multi-period indicators. On the created basis it was natural to let standard indicators work in multi-modes, as well. And starting with [article 47](https://www.mql5.com/en/articles/8207) we created such functionality (there are shortcomings to gradually locate and fix). But everything we did was done for [platform MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/").

Slightly improve library classes with respect to indicators so, that the programs developed for outdated platform MetaTrader 4 based on this library could work normally when switching over to MetaTrader 5.

As opposed to MQL5, in MQL4 we cannot have several colors of drawing for a single buffer. In MetaTrader 4 all indicator buffers are monochrome. This restriction also effects the concept of multi-buffer construction for MetaTrader 4. Here, we cannot specify its drawing color for a specific bar. Whereas, this is easy in MetaTrader 5. Here we will have to use one monochrome indicator buffer for each color. Whereas, we don’t have to create additional calculated buffers to store indicator data from the specified symbol/period of chart - all functions of request to indicator in MQL4 return the value on the specified indicator bar from the specified symbol/period. And in MQL5 one has to create indicator handle and from that handle to request data by copying the necessary amount to target array. For indicators it is its calculated buffer. And after that, get data of the specified indicator from that array by required bar index. Whereas, we get acceleration of access to indicator data in MQL5.

Thus, construction of buffer object for standard indicator in MQL4 differs. There is no need to create additional calculated buffers to store information with data of the indicator to be displayed on the current chart. But at seeming simplification we loose in flexibility: to create a colored buffer for each color we must have own one-color indicator buffer and now, when specifying the necessary bar color the buffer which corresponds to color should be displayed. The remaining buffers will be hidden. And this is a complication.

Based on the above specified the concept of multi-buffer construction for MQL4 will be as follows:

- we will have an individual indicator buffer for each color of one indicator line,
- to switch over line color one indicator line (buffer) which corresponds to necessary color will be displayed; whereas, all remaining lines of the same indicator which refer to other colors will be hidden


To sum up the above said, get that for colored multi-symbol multi-period indicator Moving Average which has three colors for displaying its line:

**In MQL5 we will have three data arrays (three buffers):**

1. Drawn buffer (data is diplayed in Data window)
2. Color buffer (not displayed in Data window but it specifies with which color buffer 1 line on each bar will be drawn)
3. Calculated buffer for data storage Moving Average from the specified symbol/period (not displayed in Data window)


**In MQL4 we will have three data arrays (three buffers):**

1. Drawn buffer for color 1 (data is diplayed in Data window)
2. Drawn buffer for color 2 (data is diplayed in Data window)

3. Drawn buffer for color 3 (data is diplayed in Data window)


When number of colors is reduced, number of buffers for MQL4 will be reduced, at an increase the number will increase. In MQL5, number of buffers for this example will always equal 3. Whereas, MQL5 enables to dynamically change the number of colors up to 64. In MQL4, painting over indicator lines is not that simple. The reason is, 64 buffer will have to be created for 64 colors. And this is only for one line. If an indicator has 4 lines 256 indicator buffer arrays will be necessary. For eight lines 512 buffers will be necessary, which is the limit. As for MQL5, for each bar the index of corresponding color is specified and on that bar the line is painted into the specified color. For MQL4, the buffer corresponding to the color will be displayed, others will be hidden. And in MQL4, all buffers for each color will be visible in the terminal data window. In MQL5, as for this example, one buffer will be visible in the terminal data window which is correct: one indicator line = one value in the terminal data window.

We are not going to correct at once what is already done. Instead, gradually, step by step from simple to complex, improve library classes to reach compatibility of certain aspects of work with indicators in library with MQL4. Today, using [Accumulation/Distribution](https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/volume_indicators/accumulation_distribution "https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/volume_indicators/accumulation_distribution") indicator as an example check creation of single-buffer monochrome multi-symbol multi-period standard indicator in MQL4 with the use of library.

### Improving library classes

As usually, first, add necessary text messages. Earlier, immediately in the final indicator program we checked correspondence of indicator buffers created by the library with entries in indicator code about necessary number of indicator buffers:

```
#property indicator_buffers 3
#property indicator_plots   1
```

Further, in the code after the library has created all necessary buffers, a check was performed:

```
//--- Check the number of buffers specified in the 'properties' block
   if(engine.BuffersPropertyPlotsTotal()!=indicator_plots)
      Alert(TextByLanguage("Attention! Value of \"indicator_plots\" should be "),engine.BuffersPropertyPlotsTotal());
   if(engine.BuffersPropertyBuffersTotal()!=indicator_buffers)
      Alert(TextByLanguage("Attention! Value of \"indicator_buffers\" should be "),engine.BuffersPropertyBuffersTotal());
```

Move this check which is slightly improved for MQL4 compatibility to the library. Put the texts displayed within the check, in the corresponding place in the library - in file \\MQL5\\Include\\DoEasy\ **Data.mqh**. And add indices of new messages to it:

```
//--- CEngine
   MSG_ENG_NO_TRADE_EVENTS,                           // There have been no trade events since the last launch of EA
   MSG_ENG_FAILED_GET_LAST_TRADE_EVENT_DESCR,         // Failed to get description of the last trading event
   MSG_ENG_FAILED_GET_MARKET_POS_LIST,                // Failed to get the list of open positions
   MSG_ENG_FAILED_GET_PENDING_ORD_LIST,               // Failed to get the list of placed orders
   MSG_ENG_NO_OPEN_POSITIONS,                         // No open positions
   MSG_ENG_NO_PLACED_ORDERS,                          // No placed orders
   MSG_ENG_ERR_VALUE_PLOTS,                           // Attention! Value \"indicator_plots\" must be
   MSG_ENG_ERR_VALUE_ORDERS,                          // Attention! Value \"indicator_buffers\" must be
```

and message texts corresponding to newly added indices:

```
//--- CEngine
   {"There have been no trade events since the last launch of EA"},
   {"Failed to get the description of the last trading event"},
   {"Failed to get open positions list"},
   {"Failed to get pending orders list"},
   {"No open positions"},
   {"No placed orders"},
   {"Attention! Value of \"indicator_plots\" should be "},
   {"Attention! Value of \"indicator_buffers\" should be "},
```

The file containing data for program inputs is called InpDatas.mqh... Change it to a correct name in terms of English grammar (I made a mistake when I named the file). Now, this file will be called as follows: \\MQL5\\Include\\DoEasy\ **InpData.mqh**.

Simply re-name it in the folder where it is.

This file is connected to the library in file Data.mqh (which we are editing now), adjust the inclusion string:

```
//+------------------------------------------------------------------+
//|                                                         Data.mqh |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "InpData.mqh"
//+------------------------------------------------------------------+
```

**Let's start implementation of cross-platform nature.**

If try and compile the library in MetaEditor by MetaTrader 4 (F7 on file Engine.mqh), we will get a plenty of errors:

![](https://c.mql5.com/2/40/a7CsJB61wt.gif)

Well, it is no surprise. Simply, start in order. First, it becomes obvious that MQL4 does not know some constants and enumerations.

Add new constants and enumerations in file \\MQL5\\Include\\DoEasy\ **ToMQL4.mqh**:

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
#define ERR_MAIL_SEND_FAILED              (ERR_SEND_MAIL_ERROR)
#define ERR_NOTIFICATION_SEND_FAILED      (ERR_NOTIFICATION_ERROR)
#define ERR_FTP_SEND_FAILED               (ERR_FTP_ERROR)
//+------------------------------------------------------------------+
//| Order types, filling policy, period, reasons                     |
//+------------------------------------------------------------------+
#define ORDER_TYPE_CLOSE_BY               (8)
#define ORDER_TYPE_BUY_STOP_LIMIT         (9)
#define ORDER_TYPE_SELL_STOP_LIMIT        (10)
#define ORDER_REASON_EXPERT               (3)
#define ORDER_REASON_SL                   (4)
#define ORDER_REASON_TP                   (5)
#define ORDER_REASON_BALANCE              (6)
#define ORDER_REASON_CREDIT               (7)
//+------------------------------------------------------------------+
//| Flags of allowed order expiration modes                          |
//+------------------------------------------------------------------+
#define SYMBOL_EXPIRATION_GTC             (1)
#define SYMBOL_EXPIRATION_DAY             (2)
#define SYMBOL_EXPIRATION_SPECIFIED       (4)
#define SYMBOL_EXPIRATION_SPECIFIED_DAY   (8)
//+------------------------------------------------------------------+
//| Flags of allowed order filling modes                             |
//+------------------------------------------------------------------+
#define SYMBOL_FILLING_FOK                (1)
#define SYMBOL_FILLING_IOC                (2)
//+------------------------------------------------------------------+
//| The flags of allowed order types                                 |
//+------------------------------------------------------------------+
#define SYMBOL_ORDER_MARKET               (1)
#define SYMBOL_ORDER_LIMIT                (2)
#define SYMBOL_ORDER_STOP                 (4)
#define SYMBOL_ORDER_STOP_LIMIT           (8)
#define SYMBOL_ORDER_SL                   (16)
#define SYMBOL_ORDER_TP                   (32)
#define SYMBOL_ORDER_CLOSEBY              (64)
//+------------------------------------------------------------------+
//| Indicator lines IDs                                              |
//+------------------------------------------------------------------+
#define TENKANSEN_LINE                    (0)
#define KIJUNSEN_LINE                     (1)
#define SENKOUSPANA_LINE                  (2)
#define SENKOUSPANB_LINE                  (3)
#define CHIKOUSPAN_LINE                   (4)
//+------------------------------------------------------------------+
//| Deal types of MQL5                                               |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Position change method                                           |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Open position direction                                          |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Order status                                                     |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Margin calculation mode                                          |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Prices on which basis chart by symbol is constructed             |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Life span of pending orders and                                  |
//| set levels of StopLoss/TakeProfit                                |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Options types                                                    |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Right provided by an option                                      |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Method of margin value calculation for an instrument             |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Swap charge methods at position move                             |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Trading operations types                                         |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Order filling policies                                           |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Order life span                                                  |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Integer properties of selected position                          |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Real properties of selected position                             |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| String properties of selected position                           |
//+------------------------------------------------------------------+
//--- the code has been removed for the sake of space
//+------------------------------------------------------------------+
//| Technical indicator types                                        |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR
  {
   IND_AC         = 5,
   IND_AD         = 6,
   IND_ALLIGATOR  = 7,
   IND_ADX        = 8,
   IND_ADXW       = 9,
   IND_ATR        = 10,
   IND_AO         = 11,
   IND_BEARS      = 12,
   IND_BANDS      = 13,
   IND_BULLS      = 14,
   IND_CCI        = 15,
   IND_DEMARKER   = 16,
   IND_ENVELOPES  = 17,
   IND_FORCE      = 18,
   IND_FRACTALS   = 19,
   IND_GATOR      = 20,
   IND_ICHIMOKU   = 21,
   IND_BWMFI      = 22,
   IND_MACD       = 23,
   IND_MOMENTUM   = 24,
   IND_MFI        = 25,
   IND_MA         = 26,
   IND_OSMA       = 27,
   IND_OBV        = 28,
   IND_SAR        = 29,
   IND_RSI        = 30,
   IND_RVI        = 31,
   IND_STDDEV     = 32,
   IND_STOCHASTIC = 33,
   IND_VOLUMES    = 34,
   IND_WPR        = 35,
   IND_DEMA       = 36,
   IND_TEMA       = 37,
   IND_TRIX       = 38,
   IND_FRAMA      = 39,
   IND_AMA        = 40,
   IND_CHAIKIN    = 41,
   IND_VIDYA      = 42,
   IND_CUSTOM     = 43,
  };
//+------------------------------------------------------------------+
//| Drawing style                                                    |
//+------------------------------------------------------------------+
enum ENUM_DRAW_TYPE
  {
   DRAW_COLOR_LINE = DRAW_LINE,              // MQL5 = 1, MQL4 = 0
   DRAW_COLOR_HISTOGRAM = DRAW_HISTOGRAM,    // MQL5 = 2, MQL4 = 2
   DRAW_COLOR_ARROW = DRAW_ARROW,            // MQL5 = 3, MQL4 = 3
   DRAW_COLOR_SECTION = DRAW_SECTION,        // MQL5 = 4, MQL4 = 1
   DRAW_COLOR_HISTOGRAM2 = DRAW_NONE,        // MQL5 = 0, MQL4 = 12
   DRAW_COLOR_ZIGZAG = DRAW_ZIGZAG,          // MQL5 = 6, MQL4 = 4
   DRAW_COLOR_BARS = DRAW_NONE,              // MQL5 = 0, MQL4 = 12
   DRAW_COLOR_CANDLES = DRAW_NONE,           // MQL5 = 0, MQL4 = 12
// DRAW_FILLING                                           MQL4 = 5
   DRAW_COLOR_NONE = DRAW_NONE,              // MQL5 = 0, MQL4 = 12
  };
//+------------------------------------------------------------------+
//| Volume type                                                      |
//+------------------------------------------------------------------+
enum ENUM_APPLIED_VOLUME
  {
   VOLUME_TICK,
   VOLUME_REAL
  };
//+------------------------------------------------------------------+
//| Trade request structure                                          |
//+------------------------------------------------------------------+
//--- the code has been further removed for the sake of space
#endif
```

Further, at the following compilation get to the error of the absence of mql5-functions in MQL4. In particular, [BarsCalculated()](https://www.mql5.com/en/docs/series/barscalculated). This function returns the number of bars calculated by indicator by its handle. For MQL4 this is unknown. [mql4-function Bars()](https://docs.mql4.com/en/series/barsfunction "https://docs.mql4.com/en/series/barsfunction") which returns the number of available bars of the specified timeseries will be the closest function to BarsCalculated() in terms of its meaning.

Since in MQL4 it is considered that the indicator is already calculated during request to it we can substitute the amount of calculated data of indicator (MQL5 BarsCalculated() ) by the number of available bars of timeseries (MQL4 Bars() ). In any case, when getting indicator data the library methods return received data and verify them. Therefore, let’s consider that specification of available timeseries bars can substitute the amount of calculated data of indicator in MQL4.

Method IndicatorBarsCalculated() which uses function BarsCalculated() is located in file \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **Buffer.mqh**. And at once in the same place we will have to make a big number of improvements to other methods of working with indicators.

Earlier, the method was fully written in the class body where the amount of calculated data was returned immediately:

```
   ENUM_INDICATOR    IndicatorType(void)                       const { return (ENUM_INDICATOR)this.GetProperty(BUFFER_PROP_IND_TYPE);        }
   string            IndicatorName(void)                       const { return this.GetProperty(BUFFER_PROP_IND_NAME);                        }
   string            IndicatorShortName(void)                  const { return this.GetProperty(BUFFER_PROP_IND_NAME_SHORT);                  }
   int               IndicatorBarsCalculated(void)             const { return ::BarsCalculated((int)this.GetProperty(BUFFER_PROP_IND_HANDLE));}
   int               IndicatorLineAdditionalNumber(void)       const { return (int)this.GetProperty(BUFFER_PROP_IND_LINE_ADDITIONAL_NUM);    }
   int               IndicatorLineMode(void)                   const { return (int)this.GetProperty(BUFFER_PROP_IND_LINE_MODE);              }
```

Now, leave the method declaration only

```
   ENUM_INDICATOR    IndicatorType(void)                       const { return (ENUM_INDICATOR)this.GetProperty(BUFFER_PROP_IND_TYPE);        }
   string            IndicatorName(void)                       const { return this.GetProperty(BUFFER_PROP_IND_NAME);                        }
   string            IndicatorShortName(void)                  const { return this.GetProperty(BUFFER_PROP_IND_NAME_SHORT);                  }
   int               IndicatorLineAdditionalNumber(void)       const { return (int)this.GetProperty(BUFFER_PROP_IND_LINE_ADDITIONAL_NUM);    }
   int               IndicatorLineMode(void)                   const { return (int)this.GetProperty(BUFFER_PROP_IND_LINE_MODE);              }
   int               IndicatorBarsCalculated(void);
```

... and move its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Return the number of standard indicator calculated bars          |
//+------------------------------------------------------------------+
int CBuffer::IndicatorBarsCalculated(void)
  {
   return(#ifdef __MQL5__ ::BarsCalculated((int)this.GetProperty(BUFFER_PROP_IND_HANDLE)) #else ::Bars(this.Symbol(),this.Timeframe()) #endif);
  }
//+------------------------------------------------------------------+
```

Here, for MQL5 return the amount of calculated indicator data, and for MQL4 — the quantity of available timeseries data.

Divide the closed parametric class constructor into two parts.

The first part — the one already available, will remain only for MQL5, and for MQL4 make a copy of mql5 code and delete unnecessary from it:

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
CBuffer::CBuffer(ENUM_BUFFER_STATUS buffer_status,
                 ENUM_BUFFER_TYPE buffer_type,
                 const uint index_plot,
                 const uint index_base_array,
                 const int num_datas,
                 const uchar total_arrays,
                 const int width,
                 const string label)
  {
#ifdef __MQL5__
   this.m_type=COLLECTION_BUFFERS_ID;
   this.m_act_state_trigger=true;
   this.m_total_arrays=total_arrays;
//--- Save integer properties
   this.m_long_prop[BUFFER_PROP_STATUS]                        = buffer_status;
   this.m_long_prop[BUFFER_PROP_TYPE]                          = buffer_type;
   this.m_long_prop[BUFFER_PROP_ID]                            = WRONG_VALUE;
   this.m_long_prop[BUFFER_PROP_IND_LINE_MODE]                 = INDICATOR_LINE_MODE_MAIN;
   this.m_long_prop[BUFFER_PROP_IND_HANDLE]                    = INVALID_HANDLE;
   this.m_long_prop[BUFFER_PROP_IND_TYPE]                      = WRONG_VALUE;
   this.m_long_prop[BUFFER_PROP_IND_LINE_ADDITIONAL_NUM]       = WRONG_VALUE;
   ENUM_DRAW_TYPE type=
     (
      !this.TypeBuffer() || !this.Status() ? DRAW_NONE      :
      this.Status()==BUFFER_STATUS_FILLING ? DRAW_FILLING   :
      ENUM_DRAW_TYPE(this.Status()+8)
     );
   this.m_long_prop[BUFFER_PROP_DRAW_TYPE]                     = type;
   this.m_long_prop[BUFFER_PROP_TIMEFRAME]                     = PERIOD_CURRENT;
   this.m_long_prop[BUFFER_PROP_ACTIVE]                        = true;
   this.m_long_prop[BUFFER_PROP_ARROW_CODE]                    = 0x9F;
   this.m_long_prop[BUFFER_PROP_ARROW_SHIFT]                   = 0;
   this.m_long_prop[BUFFER_PROP_DRAW_BEGIN]                    = 0;
   this.m_long_prop[BUFFER_PROP_SHOW_DATA]                     = (buffer_type>BUFFER_TYPE_CALCULATE ? true : false);
   this.m_long_prop[BUFFER_PROP_SHIFT]                         = 0;
   this.m_long_prop[BUFFER_PROP_LINE_STYLE]                    = STYLE_SOLID;
   this.m_long_prop[BUFFER_PROP_LINE_WIDTH]                    = width;
   this.m_long_prop[BUFFER_PROP_COLOR_INDEXES]                 = (this.Status()>BUFFER_STATUS_NONE ? (this.Status()!=BUFFER_STATUS_FILLING ? 1 : 2) : 0);
   this.m_long_prop[BUFFER_PROP_COLOR]                         = clrRed;
   this.m_long_prop[BUFFER_PROP_NUM_DATAS]                     = num_datas;
   this.m_long_prop[BUFFER_PROP_INDEX_PLOT]                    = index_plot;
   this.m_long_prop[BUFFER_PROP_INDEX_BASE]                    = index_base_array;
   this.m_long_prop[BUFFER_PROP_INDEX_COLOR]                   = this.GetProperty(BUFFER_PROP_INDEX_BASE)+
                                                                   (this.TypeBuffer()!=BUFFER_TYPE_CALCULATE ? this.GetProperty(BUFFER_PROP_NUM_DATAS) : 0);
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT_BASE]               = index_base_array+this.m_total_arrays;
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT_PLOT]               = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? index_plot+1 : index_plot);

//--- Save real properties
   this.m_double_prop[this.IndexProp(BUFFER_PROP_EMPTY_VALUE)] = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? EMPTY_VALUE : 0);
//--- Save string properties
   this.m_string_prop[this.IndexProp(BUFFER_PROP_SYMBOL)]      = ::Symbol();
   this.m_string_prop[this.IndexProp(BUFFER_PROP_LABEL)]       = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? label : NULL);
   this.m_string_prop[this.IndexProp(BUFFER_PROP_IND_NAME)]    = NULL;
   this.m_string_prop[this.IndexProp(BUFFER_PROP_IND_NAME_SHORT)]=NULL;

//--- If failed to change the size of the indicator buffer array, display the appropriate message indicating the string
   if(::ArrayResize(this.DataBuffer,(int)this.GetProperty(BUFFER_PROP_NUM_DATAS))==WRONG_VALUE)
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_DRAWING_ARRAY_RESIZE),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",(string)::GetLastError());

//--- If failed to change the size of the color array (only for a non-calculated buffer), display the appropriate message indicating the string
   if(this.TypeBuffer()>BUFFER_TYPE_CALCULATE)
      if(::ArrayResize(this.ArrayColors,(int)this.ColorsTotal())==WRONG_VALUE)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_COLORS_ARRAY_RESIZE),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",(string)::GetLastError());

//--- For DRAW_FILLING, fill in the color array with two default colors
   if(this.Status()==BUFFER_STATUS_FILLING)
     {
      this.SetColor(clrBlue,0);
      this.SetColor(clrRed,1);
     }

//--- Bind indicator buffers with arrays
//--- In a loop by the number of indicator buffers
   int total=::ArraySize(DataBuffer);
   for(int i=0;i<total;i++)
     {
      //--- calculate the index of the next array and
      //--- bind the indicator buffer by the calculated index with the dynamic array
      //--- located by the i loop index in the DataBuffer array
      int index=(int)this.GetProperty(BUFFER_PROP_INDEX_BASE)+i;
      ::SetIndexBuffer(index,this.DataBuffer[i].Array,(this.TypeBuffer()==BUFFER_TYPE_DATA ? INDICATOR_DATA : INDICATOR_CALCULATIONS));
      //--- Set indexation flag as in the timeseries to all buffer arrays
      ::ArraySetAsSeries(this.DataBuffer[i].Array,true);
     }
//--- Bind the color buffer with the array (only for a non-calculated buffer and not for the filling buffer)
   if(this.Status()!=BUFFER_STATUS_FILLING && this.TypeBuffer()!=BUFFER_TYPE_CALCULATE)
     {
      ::SetIndexBuffer((int)this.GetProperty(BUFFER_PROP_INDEX_COLOR),this.ColorBufferArray,INDICATOR_COLOR_INDEX);
      ::ArraySetAsSeries(this.ColorBufferArray,true);
     }
//--- Done if this is a calculated buffer
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
//--- Set integer parameters of the graphical series
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_TYPE,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_DRAW_TYPE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_ARROW,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_ARROW_CODE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_ARROW_SHIFT,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_ARROW_SHIFT));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_BEGIN,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_DRAW_BEGIN));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHOW_DATA,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_SHOW_DATA));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHIFT,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_SHIFT));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_STYLE,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_LINE_STYLE));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_WIDTH,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_LINE_WIDTH));
   this.SetColor((color)this.GetProperty(BUFFER_PROP_COLOR));
//--- Set real parameters of the graphical series
   ::PlotIndexSetDouble((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_EMPTY_VALUE,this.GetProperty(BUFFER_PROP_EMPTY_VALUE));
//--- Set string parameters of the graphical series
   ::PlotIndexSetString((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LABEL,this.GetProperty(BUFFER_PROP_LABEL));

//--- MQL4
#else
   this.m_type=COLLECTION_BUFFERS_ID;
   this.m_act_state_trigger=true;
   this.m_total_arrays=1;
//--- Save integer properties
   this.m_long_prop[BUFFER_PROP_STATUS]                        = buffer_status;
   this.m_long_prop[BUFFER_PROP_TYPE]                          = buffer_type;
   this.m_long_prop[BUFFER_PROP_ID]                            = WRONG_VALUE;
   this.m_long_prop[BUFFER_PROP_IND_LINE_MODE]                 = INDICATOR_LINE_MODE_MAIN;
   this.m_long_prop[BUFFER_PROP_IND_HANDLE]                    = INVALID_HANDLE;
   this.m_long_prop[BUFFER_PROP_IND_TYPE]                      = WRONG_VALUE;
   this.m_long_prop[BUFFER_PROP_IND_LINE_ADDITIONAL_NUM]       = WRONG_VALUE;

   ENUM_DRAW_TYPE type=DRAW_COLOR_NONE;
   switch((int)this.Status())
     {
      case BUFFER_STATUS_LINE       :  type=DRAW_COLOR_LINE;      break;
      case BUFFER_STATUS_HISTOGRAM  :  type=DRAW_COLOR_HISTOGRAM; break;
      case BUFFER_STATUS_ARROW      :  type=DRAW_COLOR_ARROW;     break;
      case BUFFER_STATUS_SECTION    :  type=DRAW_COLOR_SECTION;   break;
      case BUFFER_STATUS_ZIGZAG     :  type=DRAW_COLOR_ZIGZAG;    break;
      case BUFFER_STATUS_NONE       :  type=DRAW_COLOR_NONE;      break;
      case BUFFER_STATUS_FILLING    :  type=DRAW_COLOR_NONE;      break;
      case BUFFER_STATUS_HISTOGRAM2 :  type=DRAW_COLOR_NONE;      break;
      case BUFFER_STATUS_BARS       :  type=DRAW_COLOR_NONE;      break;
      case BUFFER_STATUS_CANDLES    :  type=DRAW_COLOR_NONE;      break;
      default                       :  type=DRAW_COLOR_NONE;      break;
     }
   this.m_long_prop[BUFFER_PROP_DRAW_TYPE]                     = type;
   this.m_long_prop[BUFFER_PROP_TIMEFRAME]                     = PERIOD_CURRENT;
   this.m_long_prop[BUFFER_PROP_ACTIVE]                        = true;
   this.m_long_prop[BUFFER_PROP_ARROW_CODE]                    = 0x9F;
   this.m_long_prop[BUFFER_PROP_ARROW_SHIFT]                   = 0;
   this.m_long_prop[BUFFER_PROP_DRAW_BEGIN]                    = 0;
   this.m_long_prop[BUFFER_PROP_SHOW_DATA]                     = (buffer_type>BUFFER_TYPE_CALCULATE ? true : false);
   this.m_long_prop[BUFFER_PROP_SHIFT]                         = 0;
   this.m_long_prop[BUFFER_PROP_LINE_STYLE]                    = STYLE_SOLID;
   this.m_long_prop[BUFFER_PROP_LINE_WIDTH]                    = width;
   this.m_long_prop[BUFFER_PROP_COLOR_INDEXES]                 = (this.Status()>BUFFER_STATUS_NONE ? (this.Status()!=BUFFER_STATUS_FILLING ? 1 : 2) : 0);
   this.m_long_prop[BUFFER_PROP_COLOR]                         = clrRed;
   this.m_long_prop[BUFFER_PROP_NUM_DATAS]                     = num_datas;
   this.m_long_prop[BUFFER_PROP_INDEX_PLOT]                    = index_plot;
   this.m_long_prop[BUFFER_PROP_INDEX_BASE]                    = index_base_array;
   this.m_long_prop[BUFFER_PROP_INDEX_COLOR]                   = this.GetProperty(BUFFER_PROP_INDEX_BASE);
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT_BASE]               = index_base_array+this.m_total_arrays;
   this.m_long_prop[BUFFER_PROP_INDEX_NEXT_PLOT]               = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? index_plot+1 : index_plot);

//--- Save real properties
   this.m_double_prop[this.IndexProp(BUFFER_PROP_EMPTY_VALUE)] = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? EMPTY_VALUE : 0);
//--- Save string properties
   this.m_string_prop[this.IndexProp(BUFFER_PROP_SYMBOL)]      = ::Symbol();
   this.m_string_prop[this.IndexProp(BUFFER_PROP_LABEL)]       = (this.TypeBuffer()>BUFFER_TYPE_CALCULATE ? label : NULL);
   this.m_string_prop[this.IndexProp(BUFFER_PROP_IND_NAME)]    = NULL;
   this.m_string_prop[this.IndexProp(BUFFER_PROP_IND_NAME_SHORT)]=NULL;

//--- If failed to change the size of the indicator buffer array, display the appropriate message indicating the string
   if(::ArrayResize(this.DataBuffer,(int)this.GetProperty(BUFFER_PROP_NUM_DATAS))==WRONG_VALUE)
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_DRAWING_ARRAY_RESIZE),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",(string)::GetLastError());

//--- Bind indicator buffers with arrays
//--- In a loop by the number of indicator buffers
   int total=::ArraySize(DataBuffer);
   for(int i=0;i<total;i++)
     {
      //--- calculate the index of the next array and
      //--- bind the indicator buffer by the calculated index with the dynamic array
      //--- located by the i loop index in the DataBuffer array
      int index=(int)this.GetProperty(BUFFER_PROP_INDEX_BASE)+i;
      ::SetIndexBuffer(index,this.DataBuffer[i].Array,(this.TypeBuffer()==BUFFER_TYPE_DATA ? INDICATOR_DATA : INDICATOR_CALCULATIONS));
      //--- Set indexation flag as in the timeseries to all buffer arrays
      ::ArraySetAsSeries(this.DataBuffer[i].Array,true);
     }

//--- Done if this is a calculated buffer
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
//--- Set integer parameters of the graphical series
   this.SetDrawType(type);
   ::SetIndexStyle((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),
                   (ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_DRAW_TYPE),
                   (ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_LINE_STYLE),
                   (ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_LINE_WIDTH),
                   (ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_COLOR));
   ::SetIndexArrow((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_ARROW_CODE));
   ::SetIndexShift((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_ARROW_SHIFT));
   ::SetIndexDrawBegin((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_DRAW_BEGIN));
   ::SetIndexShift((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_SHIFT));
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHOW_DATA,(ENUM_PLOT_PROPERTY_INTEGER)this.GetProperty(BUFFER_PROP_SHOW_DATA));

//--- Set real parameters of the graphical series
   ::SetIndexEmptyValue((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),this.GetProperty(BUFFER_PROP_EMPTY_VALUE));

//--- Set string parameters of the graphical series
   ::SetIndexLabel((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),this.GetProperty(BUFFER_PROP_LABEL));

#endif
  }
//+------------------------------------------------------------------+
```

Here, the main difference is in calculation of drawing type. For MQL5 we calculate it from buffer type (its status) whereas, here setting the corresponding values was easier. To set necessary values to indicator buffer use corresponding mql4 functions, since although mql5 functions [PlotIndexSetInteger()](https://www.mql5.com/en/docs/customind/plotindexsetinteger), [PlotIndexSetDouble()](https://www.mql5.com/en/docs/customind/plotindexsetdouble) and [PlotIndexSetString()](https://www.mql5.com/en/docs/customind/plotindexsetstring) cause no compilation errors, but at the same time they do not set necessary values to indicator buffer in MQL4.

In the same manner, in methods of setting certain properties for indicator buffer make a separation into mql5- and mql4-code with the use of corresponding functions for each of the languages:

```
//+------------------------------------------------------------------+
//--- Set the graphical construction type by type and status         |
//+------------------------------------------------------------------+
void CBuffer::SetDrawType(void)
  {
   ENUM_DRAW_TYPE type=(!this.TypeBuffer() || !this.Status() ? (ENUM_DRAW_TYPE)DRAW_NONE : this.Status()==BUFFER_STATUS_FILLING ? (ENUM_DRAW_TYPE)DRAW_FILLING : ENUM_DRAW_TYPE(this.Status()+8));
   this.SetProperty(BUFFER_PROP_DRAW_TYPE,type);
   #ifdef __MQL5__
      ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_TYPE,type);
   #else
      ::SetIndexStyle((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),type,EMPTY,EMPTY,EMPTY);
   #endif
  }
//+------------------------------------------------------------------+
//| Set the passed graphical construction type                       |
//+------------------------------------------------------------------+
void CBuffer::SetDrawType(const ENUM_DRAW_TYPE draw_type)
  {
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   this.SetProperty(BUFFER_PROP_DRAW_TYPE,draw_type);
   #ifdef __MQL5__
      ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_TYPE,draw_type);
   #else
      ::SetIndexStyle((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),draw_type,EMPTY,EMPTY,EMPTY);
   #endif
  }
//+------------------------------------------------------------------+
//| Set the number of initial bars                                   |
//| without drawing and values in DataWindow                         |
//+------------------------------------------------------------------+
void CBuffer::SetDrawBegin(const int value)
  {
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   this.SetProperty(BUFFER_PROP_DRAW_BEGIN,value);
   #ifdef __MQL5__
      ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_DRAW_BEGIN,value);
   #else
      ::SetIndexDrawBegin((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),value);
   #endif
  }
//+------------------------------------------------------------------+
//| Set the flag of displaying                                       |
//| construction values in DataWindow                                |
//+------------------------------------------------------------------+
void CBuffer::SetShowData(const bool flag)
  {
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   this.SetProperty(BUFFER_PROP_SHOW_DATA,flag);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHOW_DATA,flag);
  }
//+------------------------------------------------------------------+
//| Set the indicator graphical construction shift                   |
//+------------------------------------------------------------------+
void CBuffer::SetShift(const int shift)
  {
   this.SetProperty(BUFFER_PROP_SHIFT,shift);
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   #ifdef __MQL5__
      ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_SHIFT,shift);
   #else
      ::SetIndexShift((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),shift);
   #endif
  }
//+------------------------------------------------------------------+
//| Set the line style                                               |
//+------------------------------------------------------------------+
void CBuffer::SetStyle(const ENUM_LINE_STYLE style)
  {
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   this.SetProperty(BUFFER_PROP_LINE_STYLE,style);
   #ifdef __MQL5__
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_STYLE,style);
   #else
      ::SetIndexStyle((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),this.DrawType(),style,EMPTY,EMPTY);
   #endif
  }
//+------------------------------------------------------------------+
//| Set the line width                                               |
//+------------------------------------------------------------------+
void CBuffer::SetWidth(const int width)
  {
   if(this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   this.SetProperty(BUFFER_PROP_LINE_WIDTH,width);
   #ifdef __MQL5__
      ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_WIDTH,width);
   #else
      ::SetIndexStyle((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),this.DrawType(),EMPTY,width,EMPTY);
   #endif
  }
//+------------------------------------------------------------------+
//| Set the number of colors                                         |
//+------------------------------------------------------------------+
void CBuffer::SetColorNumbers(const int number)
  {
   if(number>IND_COLORS_TOTAL || this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   int n=(this.Status()!=BUFFER_STATUS_FILLING ? number : 2);
   this.SetProperty(BUFFER_PROP_COLOR_INDEXES,n);
   ::ArrayResize(this.ArrayColors,n);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_COLOR_INDEXES,n);
  }
//+------------------------------------------------------------------+
//| Set a single specified drawing color for the buffer              |
//+------------------------------------------------------------------+
void CBuffer::SetColor(const color colour)
  {
   if(this.Status()==BUFFER_STATUS_FILLING || this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   this.SetColorNumbers(1);
   this.SetProperty(BUFFER_PROP_COLOR,colour);
   this.ArrayColors[0]=colour;
   #ifdef __MQL5__
      ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_COLOR,0,this.ArrayColors[0]);
   #else
      ::SetIndexStyle((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),this.DrawType(),EMPTY,EMPTY,this.ArrayColors[0]);
   #endif
  }
//+------------------------------------------------------------------+
//| Set the drawing color to the specified color index               |
//+------------------------------------------------------------------+
void CBuffer::SetColor(const color colour,const uchar index)
  {
#ifdef __MQL5__
   if(index>IND_COLORS_TOTAL-1 || this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
   if(index>this.ColorsTotal()-1)
      this.SetColorNumbers(index+1);
   this.ArrayColors[index]=colour;
   if(index==0)
      this.SetProperty(BUFFER_PROP_COLOR,(color)this.ArrayColors[0]);
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_COLOR,index,this.ArrayColors[index]);
#else
#endif
  }
//+------------------------------------------------------------------+
//| Set drawing colors from the color array                          |
//+------------------------------------------------------------------+
void CBuffer::SetColors(const color &array_colors[])
  {
#ifdef __MQL5__
//--- Exit if the passed array is empty
   if(::ArraySize(array_colors)==0 || this.TypeBuffer()==BUFFER_TYPE_CALCULATE)
      return;
//--- Copy the passed array to the array of buffer object colors
   ::ArrayCopy(this.ArrayColors,array_colors,0,0,IND_COLORS_TOTAL);
//--- Exit if the color array was empty and not copied for some reason
   int total=::ArraySize(this.ArrayColors);
   if(total==0)
      return;
//--- If the drawing style is not DRAW_FILLING
   if(this.Status()!=BUFFER_STATUS_FILLING)
     {
      //---  if the new number of colors exceeds the currently set one,
      //--- set the new value for the number of colors
      if(total>this.ColorsTotal())
         this.SetColorNumbers(total);
     }
   //--- If the drawing style is DRAW_FILLING, set the number of colors equal to 2
   else
      total=2;
//--- Set the very first color from the color array (for a single color) to the buffer object color property
   this.SetProperty(BUFFER_PROP_COLOR,(color)this.ArrayColors[0]);
//--- Set the new number of colors for the indicator buffer
   ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_COLOR_INDEXES,total);
//--- In the loop by the new number of colors, set all colors by their indices for the indicator buffer
   for(int i=0;i<total;i++)
      ::PlotIndexSetInteger((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LINE_COLOR,i,this.ArrayColors[i]);
#else
#endif
  }
//+------------------------------------------------------------------+
//| Set the "empty" value for construction                           |
//| without drawing                                                  |
//+------------------------------------------------------------------+
void CBuffer::SetEmptyValue(const double value)
  {
   this.SetProperty(BUFFER_PROP_EMPTY_VALUE,value);
   if(this.TypeBuffer()!=BUFFER_TYPE_CALCULATE)
      #ifdef __MQL5__
         ::PlotIndexSetDouble((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_EMPTY_VALUE,value);
      #else
         ::SetIndexEmptyValue((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),value);
      #endif
  }
//+------------------------------------------------------------------+
//| Set the name for the graphical indicator series                  |
//+------------------------------------------------------------------+
void CBuffer::SetLabel(const string label)
  {
   this.SetProperty(BUFFER_PROP_LABEL,label);
   if(this.TypeBuffer()!=BUFFER_TYPE_CALCULATE)
      #ifdef __MQL5__
         ::PlotIndexSetString((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),PLOT_LABEL,label);
      #else
         ::SetIndexLabel((int)this.GetProperty(BUFFER_PROP_INDEX_PLOT),label);
      #endif
  }
//+------------------------------------------------------------------
```

```
//+------------------------------------------------------------------+
//| Set the color index to the specified timeseries index            |
//| of the color buffer array                                        |
//+------------------------------------------------------------------+
void CBuffer::SetBufferColorIndex(const uint series_index,const uchar color_index)
  {
   #ifdef __MQL4__
      return;
   #endif
   if(this.GetDataTotal(0)==0 || color_index>this.ColorsTotal()-1 || this.Status()==BUFFER_STATUS_FILLING || this.Status()==BUFFER_STATUS_NONE)
      return;
   int data_total=this.GetDataTotal(0);
   int data_index=((int)series_index<data_total ? (int)series_index : data_total-1);
   if(::ArraySize(this.ColorBufferArray)==0)
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INVALID_PROPERTY_BUFF));
   if(data_index<0)
      return;
   this.ColorBufferArray[data_index]=color_index;
  }
//+------------------------------------------------------------------+
```

In calculated buffer, in class CBufferCalculate in file \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **BufferCalculate.mqh** we have three methods which copy data from indicator handle to data array of calculated buffer object. Methods return the amount of copied data. Since for MQL4 copying data from indicator handle is not necessary and we will simply get them using corresponding standard mql4-functions with specification of symbol, timeframe and bar number, in these methods we must return a certain dummy value showing successful copying.

In the method pass the number of bars necessary for copying and return the flag indicating that copied data is equal to that value.

For MQL4 simply return it:

```
//+------------------------------------------------------------------+
//| Copy data of the specified indicator to the buffer object array  |
//+------------------------------------------------------------------+
int CBufferCalculate::FillAsSeries(const int indicator_handle,const int buffer_num,const int start_pos,const int count)
  {
   return(#ifdef __MQL5__ ::CopyBuffer(indicator_handle,buffer_num,-start_pos,count,this.DataBuffer[0].Array) #else count #endif );
  }
//+------------------------------------------------------------------+
int CBufferCalculate::FillAsSeries(const int indicator_handle,const int buffer_num,const datetime start_time,const int count)
  {
   return(#ifdef __MQL5__ ::CopyBuffer(indicator_handle,buffer_num,start_time,count,this.DataBuffer[0].Array) #else count #endif );
  }
//+------------------------------------------------------------------+
int CBufferCalculate::FillAsSeries(const int indicator_handle,const int buffer_num,const datetime start_time,const datetime stop_time)
  {
   return
     (
      #ifdef __MQL5__ ::CopyBuffer(indicator_handle,buffer_num,start_time,stop_time,this.DataBuffer[0].Array)
      #else int(::fabs(start_time-stop_time)/::PeriodSeconds(this.Timeframe())+1)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

For the very last method where we do not specify amount of data copied, but specify the start time and end time of required data,

for MQL4 calculate the number of bars between values of start time and end time of required data and return the calculated value.

Create all the buffer objects for standard indicators in indicator buffer collection class

in file \\MQL5\\Include\\DoEasy\\Collections\ **BuffersCollection.mqh**.

This class also was improved for MQL4 compatibility.

In MQL5, in methods for creation of standard indicator objects a handle of the required indicator is created first, and in case of its successful creation the object itself is constructed. In MQL4 no handle has to be created, therefore, add a dummy handle of the created indicator to all these methods:

```
//+------------------------------------------------------------------+
//| Create multi-symbol multi-period AC                              |
//+------------------------------------------------------------------+
int CBuffersCollection::CreateAC(const string symbol,const ENUM_TIMEFRAMES timeframe,const int id=WRONG_VALUE)
  {
//--- Create indicator handle and set default ID
   int handle= #ifdef __MQL5__ ::iAC(symbol,timeframe) #else 0 #endif ;
   int identifier=(id==WRONG_VALUE ? IND_AC : id);
   color array_colors[3]={clrGreen,clrRed,clrGreen};
   CBuffer *buff=NULL;
   if(handle!=INVALID_HANDLE)
     {
      //--- Create histogram buffer from the zero line
      this.CreateHistogram();
```

In the meantime, we added zero as handle value. Further, probably, we will emulate creation of indicator handles so that to exclude creation of two same standard indicator objects with the same inputs. However, practice will demonstrate whether this should be done factually.

A string with handle creation emulation is already added to all methods of creation of standard indicator objects.

We will not dwell on them here. Instead, consider the changes necessary to create single-buffer monochrome standard indicator for MQL4 using the AD indicator creation method as an example:

```
//+------------------------------------------------------------------+
//| Create multi-symbol multi-period AD                              |
//+------------------------------------------------------------------+
int CBuffersCollection::CreateAD(const string symbol,const ENUM_TIMEFRAMES timeframe,const ENUM_APPLIED_VOLUME applied_volume,const int id=WRONG_VALUE)
  {
//--- Create indicator handle and set default ID
   int handle= #ifdef __MQL5__ ::iAD(symbol,timeframe,applied_volume) #else 0 #endif ;
   int identifier=(id==WRONG_VALUE ? IND_AD : id);
   color array_colors[1]={clrLightSeaGreen};
   CBuffer *buff=NULL;
   if(handle!=INVALID_HANDLE)
     {
      //--- Create line buffer
      this.CreateLine();
      //--- Get the last created buffer object (drawn) and set to it all necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_AD);
      buff.SetLineMode(INDICATOR_LINE_MODE_MAIN);
      buff.SetShowData(true);
      buff.SetLabel("A/D("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Accumulation/Distribution");
      buff.SetIndicatorShortName("A/D("+symbol+","+TimeframeDescription(timeframe)+")");
      #ifdef __MQL5__ buff.SetColors(array_colors); #else buff.SetColor(array_colors[0]); #endif

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
         buff.SetIndicatorType(IND_AD);
         buff.SetLineMode(INDICATOR_LINE_MODE_MAIN);
         buff.SetEmptyValue(EMPTY_VALUE);
         buff.SetLabel("A/D("+symbol+","+TimeframeDescription(timeframe)+")");
         buff.SetIndicatorName("Accumulation/Distribution");
         buff.SetIndicatorShortName("A/D("+symbol+","+TimeframeDescription(timeframe)+")");
      #endif
     }
   return handle;
  }
//+------------------------------------------------------------------+
```

Here, for MQL5 set for the buffer a set of its colors by the method of their passing to object using color array, and for MQL4 set only one color — the first one in color array. We need calculated buffer only for MQL5: it will store data of created indicator AD on the specified symbol and chart period. For MQL4, such buffer is not needed since we will get all data directly from the function of call to indicator [iAD()](https://docs.mql4.com/en/indicators/iad "https://docs.mql4.com/en/indicators/iad").

The method which prepares data of the specified standard indicator for setting values on the current symbol chart, for MQL5 it reads data from calculated buffer. For MQL4 it is quite simple to get requested data [from the function of call to standard indicator data](https://docs.mql4.com/en/indicators "https://docs.mql4.com/en/indicators"):

```
//+------------------------------------------------------------------+
//| Prepare data of the specified standard indicator                 |
//| for setting values on the current symbol chart                   |
//+------------------------------------------------------------------+
int CBuffersCollection::PreparingSetDataStdInd(CBuffer *buffer_data0,CBuffer *buffer_data1,CBuffer *buffer_data2,CBuffer *buffer_data3,CBuffer *buffer_data4,
                                               CBuffer *buffer_calc0,CBuffer *buffer_calc1,CBuffer *buffer_calc2,CBuffer *buffer_calc3,CBuffer *buffer_calc4,
                                               const ENUM_INDICATOR ind_type,
                                               const int series_index,
                                               const datetime series_time,
                                               int &index_period,
                                               int &num_bars,
                                               double &value00,
                                               double &value01,
                                               double &value10,
                                               double &value11,
                                               double &value20,
                                               double &value21,
                                               double &value30,
                                               double &value31,
                                               double &value40,
                                               double &value41)
  {
     //--- Find bar index on a period which corresponds to the time of current bar start
     index_period=::iBarShift(buffer_data0.Symbol(),buffer_data0.Timeframe(),series_time,true);
     if(index_period==WRONG_VALUE || #ifdef __MQL5__ index_period>buffer_calc0.GetDataTotal()-1 #else index_period>buffer_data0.GetDataTotal()-1 #endif )
        return WRONG_VALUE;

     //--- For MQL5
     #ifdef __MQL5__
        //--- Get the value by this index from indicator buffer
        if(buffer_calc0!=NULL)
           value00=buffer_calc0.GetDataBufferValue(0,index_period);
        if(buffer_calc1!=NULL)
           value10=buffer_calc1.GetDataBufferValue(0,index_period);
        if(buffer_calc2!=NULL)
           value20=buffer_calc2.GetDataBufferValue(0,index_period);
        if(buffer_calc3!=NULL)
           value30=buffer_calc3.GetDataBufferValue(0,index_period);
        if(buffer_calc4!=NULL)
           value40=buffer_calc4.GetDataBufferValue(0,index_period);

     //--- for MQL4
     #else
        switch((int)ind_type)
          {
           //--- Single-buffer standard indicators
           case IND_AC           :  value00=::iAC(buffer_data0.Symbol(),buffer_data0.Timeframe(),index_period);      break;
           case IND_AD           :  value00=::iAD(buffer_data0.Symbol(),buffer_data0.Timeframe(),index_period);      break;
           case IND_AMA          :  break;
           case IND_AO           :  value00=::iAO(buffer_data0.Symbol(),buffer_data0.Timeframe(),index_period);      break;
           case IND_ATR          :  break;
           case IND_BEARS        :  break;
           case IND_BULLS        :  break;
           case IND_BWMFI        :  value00=::iBWMFI(buffer_data0.Symbol(),buffer_data0.Timeframe(),index_period);   break;
           case IND_CCI          :  break;
           case IND_CHAIKIN      :  break;
           case IND_DEMA         :  break;
           case IND_DEMARKER     :  break;
           case IND_FORCE        :  break;
           case IND_FRAMA        :  break;
           case IND_MA           :  break;
           case IND_MFI          :  break;
           case IND_MOMENTUM     :  break;
           case IND_OBV          :  break;
           case IND_OSMA         :  break;
           case IND_RSI          :  break;
           case IND_SAR          :  break;
           case IND_STDDEV       :  break;
           case IND_TEMA         :  break;
           case IND_TRIX         :  break;
           case IND_VIDYA        :  break;
           case IND_VOLUMES      :  value00=(double)::iVolume(buffer_data0.Symbol(),buffer_data0.Timeframe(),index_period);  break;
           case IND_WPR          :  break;

           //--- Multi-buffer standard indicators
           case IND_ENVELOPES    :  break;
           case IND_FRACTALS     :  break;

           case IND_ADX          :  break;
           case IND_ADXW         :  break;
           case IND_BANDS        :  break;
           case IND_MACD         :  break;
           case IND_RVI          :  break;
           case IND_STOCHASTIC   :  break;
           case IND_ALLIGATOR    :  break;

           case IND_ICHIMOKU     :  break;
           case IND_GATOR        :  break;

           default:
             break;
          }
     #endif

     int series_index_start=series_index;
     //--- For the current chart we don’t need to calculate the number of bars processed - only one bar is available
     if(buffer_data0.Symbol()==::Symbol() && buffer_data0.Timeframe()==::Period())
       {
        series_index_start=series_index;
        num_bars=1;
       }
     else
       {
        //--- Get the bar time which the bar with index_period index falls into on a period and symbol of calculated buffer
        datetime time_period=::iTime(buffer_data0.Symbol(),buffer_data0.Timeframe(),index_period);
        if(time_period==0) return false;
        //--- Get the current chart bar which corresponds to the time
        series_index_start=::iBarShift(::Symbol(),::Period(),time_period,true);
        if(series_index_start==WRONG_VALUE) return WRONG_VALUE;
        //--- Calculate the number of bars on the current chart which are to be filled in with calculated buffer data
        num_bars=::PeriodSeconds(buffer_data0.Timeframe())/::PeriodSeconds(PERIOD_CURRENT);
        if(num_bars==0) num_bars=1;
       }
     //--- Take values for color calculation
     if(buffer_data0!=NULL)
        value01=(series_index_start+num_bars>buffer_data0.GetDataTotal()-1 ? value00 : buffer_data0.GetDataBufferValue(0,series_index_start+num_bars));
     if(buffer_data1!=NULL)
        value11=(series_index_start+num_bars>buffer_data1.GetDataTotal()-1 ? value10 : buffer_data1.GetDataBufferValue(0,series_index_start+num_bars));
     if(buffer_data2!=NULL)
        value21=(series_index_start+num_bars>buffer_data2.GetDataTotal()-1 ? value20 : buffer_data2.GetDataBufferValue(0,series_index_start+num_bars));
     if(buffer_data3!=NULL)
        value31=(series_index_start+num_bars>buffer_data3.GetDataTotal()-1 ? value30 : buffer_data3.GetDataBufferValue(0,series_index_start+num_bars));
     if(buffer_data4!=NULL)
        value41=(series_index_start+num_bars>buffer_data4.GetDataTotal()-1 ? value40 : buffer_data4.GetDataBufferValue(0,series_index_start+num_bars));

   return series_index_start;
  }
//+------------------------------------------------------------------+
```

For now, for MQL4 getting data only from single-buffer standard indicators which have no inputs except of symbol and chart period, is implemented. The remaining standard indicators will be implemented in the following articles.

Minor changes on exclusion of check for MQL4 are made in the method of setting a value for the current chart to buffers of the specified standard indicator by the timeseries index in accordance with buffer object symbol/period:

```
//+------------------------------------------------------------------+
//| Set values for the current chart to buffers of the specified     |
//| standard indicator by the timeseries index in accordance         |
//| with buffer object symbol/period                                 |
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
              vol0=::iVolume(buffer_calc0.Symbol(),buffer_calc0.Timeframe(),index_period);
              vol1=::iVolume(buffer_calc0.Symbol(),buffer_calc0.Timeframe(),index_period+1);
              clr=
                (
                 value00>value01 && vol0>vol1 ? 0 :
                 value00<value01 && vol0<vol1 ? 1 :
                 value00>value01 && vol0<vol1 ? 2 :
                 value00<value01 && vol0>vol1 ? 3 : 4
                );
             }
           buffer_data0.SetBufferColorIndex(index,clr);
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

Since we decided to move the correspondence check for the number of buffer objects created by the library with values in #property of the program, add such method to the class of library main object CEngine in file \\MQL5\\Include\\DoEasy\ **Engine.mqh**.

Declare the method in the public section of the class:

```
//--- Display short description of all indicator buffers of the buffer collection
   void                 BuffersPrintShort(void);

//--- Specify the required number of buffers for indicators
   void                 CheckIndicatorsBuffers(const int buffers,const int plots #ifdef __MQL4__ =1 #endif );

//--- Return the bar index on the specified timeframe chart by the current chart's bar index
```

Write its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Specify the required number of buffers for indicators            |
//+------------------------------------------------------------------+
void CEngine::CheckIndicatorsBuffers(const int buffers,const int plots #ifdef __MQL4__ =1 #endif )
  {
   #ifdef __MQL5__
      if(this.BuffersPropertyPlotsTotal()!=plots)
         ::Alert(CMessage::Text(MSG_ENG_ERR_VALUE_PLOTS),this.BuffersPropertyPlotsTotal());
      if(this.BuffersPropertyBuffersTotal()!=buffers)
         ::Alert(CMessage::Text(MSG_ENG_ERR_VALUE_ORDERS),this.BuffersPropertyBuffersTotal());
   #else
      if(buffers!=this.BuffersPropertyPlotsTotal())
         ::Alert(CMessage::Text(MSG_ENG_ERR_VALUE_ORDERS),this.BuffersPropertyPlotsTotal());
      ::IndicatorBuffers(this.BuffersPropertyBuffersTotal());
   #endif
  }
//+------------------------------------------------------------------+
```

For MQL5 simply display warning notices about noncorrespondence of created number of indicator buffers (drawn and calculated) with specified value in #property of indicator program.

For MQL4 in case of noncorrespondence of the value specified in #property indicator\_buffers, display a notice on that and [set the total number of all indicator buffers](https://docs.mql4.com/en/customind/indicatorbuffers "https://docs.mql4.com/en/customind/indicatorbuffers") (drawn and calculated) according to the total number of all buffers created by the library.

Now, set capacity of data displayed for indicators in MQL4. To do this, improve function for setting the capacity and levels for standard indicators in file of library service functions \\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh**:

```
//+------------------------------------------------------------------+
//| Set capacity and levels for standard indicator                   |
//+------------------------------------------------------------------+
void SetIndicatorLevels(const string symbol,const ENUM_INDICATOR ind_type)
  {
   int digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
   switch(ind_type)
     {
      case IND_AD          :
      case IND_CHAIKIN     :
      case IND_OBV         :
      case IND_VOLUMES     : digits=0;    break;

      case IND_AO          :
      case IND_BEARS       :
      case IND_BULLS       :
      case IND_FORCE       :
      case IND_STDDEV      :
      case IND_AMA         :
      case IND_DEMA        :
      case IND_FRAMA       :
      case IND_MA          :
      case IND_TEMA        :
      case IND_VIDYA       :
      case IND_BANDS       :
      case IND_ENVELOPES   :
      case IND_MACD        : digits+=1;   break;

      case IND_AC          :
      case IND_OSMA        : digits+=2;   break;

      case IND_MOMENTUM    : digits=2;    break;

      case IND_CCI         :
        IndicatorSetInteger(INDICATOR_LEVELS,2);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,0,100);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,1,-100);
        digits=2;
        break;
      case IND_DEMARKER    :
        IndicatorSetInteger(INDICATOR_LEVELS,2);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,0,0.7);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,1,0.3);
        digits=3;
        break;
      case IND_MFI         :
        IndicatorSetInteger(INDICATOR_LEVELS,2);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,0,80);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,1,20);
        break;
      case IND_RSI         :
        IndicatorSetInteger(INDICATOR_LEVELS,3);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,0,70);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,1,50);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,2,30);
        digits=2;
        break;
      case IND_STOCHASTIC  :
        IndicatorSetInteger(INDICATOR_LEVELS,2);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,0,80);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,1,20);
        digits=2;
        break;
      case IND_WPR         :
        IndicatorSetInteger(INDICATOR_LEVELS,2);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,0,-80);
        IndicatorSetDouble(INDICATOR_LEVELVALUE,1,-20);
        digits=2;
        break;

      case IND_ATR         :              break;
      case IND_SAR         :              break;
      case IND_TRIX        :              break;

      default:
        IndicatorSetInteger(INDICATOR_LEVELS,0);
        break;
     }
   #ifdef __MQL5__
      IndicatorSetInteger(INDICATOR_DIGITS,digits);
   #else
      IndicatorDigits(digits);
   #endif
  }
//+------------------------------------------------------------------+
```

Here, for MQL4 to set capacity of displayed indicator data use [standard mql4-function IndicatorDigits()](https://docs.mql4.com/en/customind/indicatordigits "https://docs.mql4.com/en/customind/indicatordigits").

**This concludes the improvement of library classes for creation of single-buffer multi-symbol multi-period standard indicators.**

### Testing

To perform the test [take the second indicator (TestDoEasyPart51\_2.mq5) from the previous article](https://www.mql5.com/en/articles/8354#node03) and

save it in the folder of terminal indicators MetaTrader 4 \\MQL4\\Indicators\\TestDoEasy\ **Part52\** named as **TestDoEasyPart52.mq4**.

The previous test indicator created Gator Oscillator multi-symbol multi-period standard indicator. Whereas, we want to create Accumulation/Distribution indicator.

In file header set the number of indicator buffers required for MQL4:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart52.mq4 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
//--- properties
#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1

//--- classes

//--- enums

//--- defines

//--- structures

//--- input variables
sinput   string               InpUsedSymbols    =  "EURUSD";      // Used symbol (one only)
sinput   ENUM_TIMEFRAMES      InpPeriod         =  PERIOD_H4;     // Used chart period
//---
sinput   bool                 InpUseSounds      =  true;          // Use sounds
//--- indicator buffers

//--- global variables
ENUM_SYMBOLS_MODE    InpModeUsedSymbols=  SYMBOLS_MODE_DEFINES;   // Mode of used symbols list
ENUM_TIMEFRAMES_MODE InpModeUsedTFs    =  TIMEFRAMES_MODE_LIST;   // Mode of used timeframes list
string               InpUsedTFs;                                  //  List of used timeframes
CEngine              engine;                                      // CEngine library main object
string               prefix;                                      // Prefix of graphical object names
int                  min_bars;                                    // The minimum number of bars for the indicator calculation
int                  used_symbols_mode;                           // Mode of working with symbols
string               array_used_symbols[];                        // The array for passing used symbols to the library
string               array_used_periods[];                        // The array for passing used timeframes to the library
//+------------------------------------------------------------------+
```

In handler OnInit() create standard indicator object Accumulation/Distribution, and specify where AD indicator type is required:

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
//--- Create all the necessary buffer objects for constructing the selected standard indicator
   if(!engine.BufferCreateAD(InpUsedSymbols,InpPeriod,VOLUME_TICK,1))
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
   string label=engine.BufferGetIndicatorShortNameByTypeID(IND_AD,1);
   IndicatorSetString(INDICATOR_SHORTNAME,label);
   SetIndicatorLevels(InpUsedSymbols,IND_AD);

//--- Successful
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Earlier, correspondence of number of specified and created indicator buffers was checked in handler OnInit():

```
//--- Check the number of buffers specified in the 'properties' block
   if(engine.BuffersPropertyPlotsTotal()!=indicator_plots)
      Alert(TextByLanguage("Attention! Value of \"indicator_plots\" should be "),engine.BuffersPropertyPlotsTotal());
   if(engine.BuffersPropertyBuffersTotal()!=indicator_buffers)
      Alert(TextByLanguage("Attention! Value of \"indicator_buffers\" should be "),engine.BuffersPropertyBuffersTotal());
```

Now, it is substituted by the call of corresponding library method.

In handler OnCalculate(), it is enough to only substitute writing of Gator Oscillator data by writing of Accumulation/Distribution data in the main program loop:

```
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

//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer(rates_data);   // Working in the library timer
      engine.EventsHandling();      // Working with library events
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
      engine.GetBuffersCollection().SetDataBufferStdInd(IND_AD,1,i,time[i]);
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

For indicator’s mql5-version as opposed to mql4-version we will need to change the number of drawn and calculated buffers specified in #property:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart52.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
//--- properties
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   1
```

Compile the indicator and launch it on EURUSD H1 chart in MetaTrader 4 terminal with values in indicator inputs for EURUSD symbol and for H4 period. Thus, display AD indicator calculated for EURUSD H4 on hour chart EURUSD in MetaTrader 4 terminal:

![](https://c.mql5.com/2/40/CDopGjpCpF.gif)

### What's next?

In the next article, continue working with indicators in MetaTrader 5 and work unhurriedly upon library’s cross-platform nature.

All files of the current version of the library are attached below together with test indicator files. You can download them and test everything.

Leave your comments, questions and suggestions in the comments to the article.

**Please, pay attention that all work on compatibility with the previous platform is being done only to support multi-platform nature of the library which originally was created for MQL5 and where it has more advantages and functionality**.

No separate articles on work in MQL4 with this library were planned. And none will be written. Whereas, each reader can develop individually, on a “tailor made” basis everything which occurs to be insufficient when using MetaTrader 4 with this library. I will continue making the library compatible with both platforms, but only for a reason that library user is able to easily move all his library-based programs to work in MetaTrader 4.

[Back to contents](https://www.mql5.com/en/articles/8399#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8399](https://www.mql5.com/ru/articles/8399)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8399.zip "Download all attachments in the single ZIP archive")

[MQL4.zip](https://www.mql5.com/en/articles/download/8399/mql4.zip "Download MQL4.zip")(3777.3 KB)

[MQL5.zip](https://www.mql5.com/en/articles/download/8399/mql5.zip "Download MQL5.zip")(3777.31 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/357949)**

![Timeseries in DoEasy library (part 53): Abstract base indicator class](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library__5.png)[Timeseries in DoEasy library (part 53): Abstract base indicator class](https://www.mql5.com/en/articles/8464)

The article considers creation of an abstract indicator which further will be used as the base class to create objects of library’s standard and custom indicators.

![Parallel Particle Swarm Optimization](https://c.mql5.com/2/40/parallel_optimization_2.png)[Parallel Particle Swarm Optimization](https://www.mql5.com/en/articles/8321)

The article describes a method of fast optimization using the particle swarm algorithm. It also presents the method implementation in MQL, which is ready for use both in single-threaded mode inside an Expert Advisor and in a parallel multi-threaded mode as an add-on that runs on local tester agents.

![Brute force approach to pattern search](https://c.mql5.com/2/40/back_to_the_future_part1.png)[Brute force approach to pattern search](https://www.mql5.com/en/articles/8311)

In this article, we will search for market patterns, create Expert Advisors based on the identified patterns, and check how long these patterns remain valid, if they ever retain their validity.

![Neural networks made easy (Part 3): Convolutional networks](https://c.mql5.com/2/48/Neural_networks_made_easy_003.png)[Neural networks made easy (Part 3): Convolutional networks](https://www.mql5.com/en/articles/8234)

As a continuation of the neural network topic, I propose considering convolutional neural networks. This type of neural network are usually applied to analyzing visual imagery. In this article, we will consider the application of these networks in the financial markets.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/8399&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070393615525549325)

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