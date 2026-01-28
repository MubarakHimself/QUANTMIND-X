---
title: Timeseries in DoEasy library (part 51): Composite multi-period multi-symbol standard indicators
url: https://www.mql5.com/en/articles/8354
categories: Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T19:34:45.596191
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=idnpxnvzoxipavwiqfbwufpnorzyrhpd&ssn=1769186083526865780&ssn_dr=0&ssn_sr=0&fv_date=1769186083&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8354&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Timeseries%20in%20DoEasy%20library%20(part%2051)%3A%20Composite%20multi-period%20multi-symbol%20standard%20indicators%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918608349383989&fz_uniq=5070398318514738467&sv=2552)

MetaTrader 5 / Examples


### Table of contents

- [Concept](https://www.mql5.com/en/articles/8354#node01)
- [Improving library classes and methods](https://www.mql5.com/en/articles/8354#node02)
- [Test](https://www.mql5.com/en/articles/8354#node03)
- [What's next?](https://www.mql5.com/en/articles/8354#node04)


### Concept

Today, I finish creation of objects of multi-symbol multi-period standard indicators. And tо top it off, I left a very good example of creating [Ichimoku Kinko Hyo](https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/trend_indicators/ichimoku_kinko_hyo "https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/trend_indicators/ichimoku_kinko_hyo") indicator. To draw it we need not only to create all of its significant buffers displayed in terminal data window, but also to add two additional buffers for painting over the area between its two lines 'Senkou Span A' and 'Senkou Span B' using two histograms drawn between two values. Whereas, each of the histograms must repeat the style and the color of the line it relates to.

Creation of such indicator will be a good illustrative example of how one can create own compound custom indicators using this library.

Finally, I will create the last object of a multi-symbol multi-period indicator out of the complete set of standard indicators of MetaTrader5 terminal — Bill Williams’ indicator [Gator Oscillator](https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/bw_indicators/gator_oscillator "https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/bw_indicators/gator_oscillator"), constructed based on his another indicator [Alligator](https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/bw_indicators/alligator "https://www.metatrader5.com/en/mobile-trading/iphone/help/chart/indicators/bw_indicators/alligator"), considered [in the previous article](https://www.mql5.com/en/articles/8331).

Since we can create indicators of any complexity, which may contain various lines of various drawing types with the buffer objects corresponding to them, but all these buffers will belong to a single indicator object, we must introduce another property for the buffer object - a number of additional indicator line (auxiliary buffer for drawing additional indicator lines which serve for its design). Thus, by the number of that line we will exactly determine the required auxiliary line (buffer object) of any indicator object.

For example, if we want the indicator which draws the moving average to display certain statuses of its main line, such as line crossing by price, line interaction with other indicators, etc. we can add to our custom indicator one or several buffer objects and by this buffer display on the chart the required data in the required moments - to put arrows, to paint over areas, etc.

### Improving library classes and methods

First, enter a new library text message in file \\MQL5\\Include\\DoEasy\ **Data.mqh**.

Add new message index:

```
//--- CBuffer
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_BASE,               // Base data buffer index
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_PLOT,               // Plotted buffer serial number
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_COLOR,              // Color buffer index
   MSG_LIB_TEXT_BUFFER_TEXT_NUM_DATAS,                // Number of data buffers
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_BASE,          // Index of the array to be assigned as the next indicator buffer
   MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_PLOT,          // Index of the next drawn buffer
   MSG_LIB_TEXT_BUFFER_TEXT_ID,                       // Indicator buffers ID
   MSG_LIB_TEXT_BUFFER_TEXT_IND_LINE_MODE,            // Indicator line
   MSG_LIB_TEXT_BUFFER_TEXT_IND_HANDLE,               // Indicator handle that uses the buffer
   MSG_LIB_TEXT_BUFFER_TEXT_IND_HANDLE,                 // Indicator type that uses the buffer
   MSG_LIB_TEXT_BUFFER_TEXT_IND_LINE_ADDITIONAL_NUM,  // Number of additional line
   MSG_LIB_TEXT_BUFFER_TEXT_TIMEFRAME,                // Buffer data period (timeframe)
```

and the text corresponding to the newly added index:

```
   {"Index of Base data buffer"},
   {"Plot buffer sequence number"},
   {"Color buffer index"},
   {"Number of data buffers"},
   {"Array index for assignment as the next indicator buffer"},
   {"Index of the next drawable buffer"},
   {"Indicator Buffer Id"},
   {"Indicator line"},
   {"Indicator handle that uses the buffer"},
   {"Indicator type that uses the buffer"},
   {"Additional line number"},
   {"Buffer data Period (Timeframe)"},
```

In \\MQL5\\Include\\DoEasy\ **Defines.mqh** file, to enumeration of indicator line types add new lines for Ichimoku Kinko Hyo indicator and another line as auxiliary one — for drawing our indicators:

```
//+------------------------------------------------------------------+
//| Values of indicator lines in enumeration                         |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_LINE_MODE
  {
   INDICATOR_LINE_MODE_MAIN         =  0,                   // Main line
   INDICATOR_LINE_MODE_SIGNAL       =  1,                   // Signal line
   INDICATOR_LINE_MODE_UPPER        =  0,                   // Upper line
   INDICATOR_LINE_MODE_LOWER        =  1,                   // Lower line
   INDICATOR_LINE_MODE_MIDDLE       =  2,                   // Middle line
   INDICATOR_LINE_MODE_JAWS         =  0,                   // Jaws line
   INDICATOR_LINE_MODE_TEETH        =  1,                   // Teeth line
   INDICATOR_LINE_MODE_LIPS         =  2,                   // Lips line
   INDICATOR_LINE_MODE_DI_PLUS      =  1,                   // Line +DI
   INDICATOR_LINE_MODE_DI_MINUS     =  2,                   // Line -DI
   INDICATOR_LINE_MODE_TENKAN_SEN   =  0,                   // Line Tenkan-sen
   INDICATOR_LINE_MODE_KIJUN_SEN    =  1,                   // Line Kijun-sen
   INDICATOR_LINE_MODE_SENKOU_SPANA =  2,                   // Line Senkou Span A
   INDICATOR_LINE_MODE_SENKOU_SPANB =  3,                   // Line Senkou Span B
   INDICATOR_LINE_MODE_CHIKOU_SPAN  =  4,                   // Line Chikou Span
   INDICATOR_LINE_MODE_ADDITIONAL   =  5,                   // Additional line
  };
//+------------------------------------------------------------------+
```

Since in the previous article I set identical values to same-type indicator lines in this enumeration and united handlers of different objects of standard indicators into a single one, now when displaying line-type description from a buffer object we see descriptions of the lines which correspond to the very first encountered values of this enumeration. For example, if I display description of Jaws line of Alligator standard indicator and the value of INDICATOR\_LINE\_MODE\_JAWS constant in this enumeration equals to zero, description of the first constant out of this enumeration is displayed which value is also equal to zero - constants INDICATOR\_LINE\_MODE\_MAIN.

This is not an error but an unpleasant confusion. To avoid it we must have its unique value for each enumeration constant. But in this case handlers will have to be divided again, which is much worse. Thus, we will do as follows: add another enumeration and display description of buffer object line by checking exactly which line of which indicator exactly the buffer displays and display the value of the new enumeration which corresponds to the buffer object.

Add this enumeration:

```
//+------------------------------------------------------------------+
//| Enumeration of indicator lines                                   |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_LINE
  {
   INDICATOR_LINE_MAIN,                                     // Main line
   INDICATOR_LINE_SIGNAL,                                   // Signal line
   INDICATOR_LINE_UPPER,                                    // Upper line
   INDICATOR_LINE_LOWER,                                    // Lower line
   INDICATOR_LINE_MIDDLE,                                   // Middle line
   INDICATOR_LINE_JAWS,                                     // Jaws line
   INDICATOR_LINE_TEETH,                                    // Teeth line
   INDICATOR_LINE_LIPS,                                     // Lips line
   INDICATOR_LINE_DI_PLUS,                                  // Line +DI
   INDICATOR_LINE_DI_MINUS,                                 // Line -DI
   INDICATOR_LINE_TENKAN_SEN,                               // Line Tenkan-sen
   INDICATOR_LINE_KIJUN_SEN,                                // Line Kijun-sen
   INDICATOR_LINE_SENKOU_SPANA,                             // Line Senkou Span A
   INDICATOR_LINE_SENKOU_SPANB,                             // Line Senkou Span B
   INDICATOR_LINE_CHIKOU_SPAN,                              // Line Chikou Span
   INDICATOR_LINE_ADDITIONAL,                               // Additional line
  };
//+------------------------------------------------------------------+
```

In this enumeration each constant has its own unique value ranging from 0 to 15 and now we can easily display required values for each specific line of a specific indicator. Let’s do it below.

In the same file add another buffer object integer property, at the same time having increased the number of buffer object integer properties from 24 to **25**:

```
//+------------------------------------------------------------------+
//| Buffer integer properties                                        |
//+------------------------------------------------------------------+
enum ENUM_BUFFER_PROP_INTEGER
  {
   BUFFER_PROP_INDEX_PLOT = 0,                              // Plotted buffer serial number
   BUFFER_PROP_STATUS,                                      // Buffer status (by drawing style) (from the ENUM_BUFFER_STATUS enumeration)
   BUFFER_PROP_TYPE,                                        // Buffer type (from the ENUM_BUFFER_TYPE enumeration)
   BUFFER_PROP_TIMEFRAME,                                   // Buffer data period (timeframe)
   BUFFER_PROP_ACTIVE,                                      // Buffer usage flag
   BUFFER_PROP_DRAW_TYPE,                                   // Graphical construction type (from the ENUM_DRAW_TYPE enumeration)
   BUFFER_PROP_ARROW_CODE,                                  // Arrow code for DRAW_ARROW style
   BUFFER_PROP_ARROW_SHIFT,                                 // The vertical shift of the arrows for DRAW_ARROW style
   BUFFER_PROP_LINE_STYLE,                                  // Line style
   BUFFER_PROP_LINE_WIDTH,                                  // Line width
   BUFFER_PROP_DRAW_BEGIN,                                  // The number of initial bars that are not drawn and values in DataWindow
   BUFFER_PROP_SHOW_DATA,                                   // Flag of displaying construction values in DataWindow
   BUFFER_PROP_SHIFT,                                       // Indicator graphical construction shift by time axis in bars
   BUFFER_PROP_COLOR_INDEXES,                               // Number of colors
   BUFFER_PROP_COLOR,                                       // Drawing color
   BUFFER_PROP_INDEX_BASE,                                  // Base data buffer index
   BUFFER_PROP_INDEX_NEXT_BASE,                             // Index of the array to be assigned as the next indicator buffer
   BUFFER_PROP_INDEX_NEXT_PLOT,                             // Index of the next plotted buffer
   BUFFER_PROP_ID,                                          // ID of multiple buffers of the same indicator
   BUFFER_PROP_IND_LINE_MODE,                               // Indicator line
   BUFFER_PROP_IND_HANDLE,                                  // Indicator handle that uses the buffer
   BUFFER_PROP_IND_TYPE,                                    // Indicator type that uses the buffer
   BUFFER_PROP_IND_LINE_ADDITIONAL_NUM,                     // Number of indicator additional line
   BUFFER_PROP_NUM_DATAS,                                   // Number of data buffers
   BUFFER_PROP_INDEX_COLOR,                                 // Color buffer index
  };
#define BUFFER_PROP_INTEGER_TOTAL (25)                      // Total number of buffer integer properties
#define BUFFER_PROP_INTEGER_SKIP  (2)                       // Number of buffer properties not used in sorting
//+------------------------------------------------------------------+
```

To be able to search and sort buffer objects by a new property add this property to enumeration of possible sorting criteria:

```
//+------------------------------------------------------------------+
//| Possible buffer sorting criteria                                 |
//+------------------------------------------------------------------+
#define FIRST_BUFFER_DBL_PROP          (BUFFER_PROP_INTEGER_TOTAL-BUFFER_PROP_INTEGER_SKIP)
#define FIRST_BUFFER_STR_PROP          (BUFFER_PROP_INTEGER_TOTAL-BUFFER_PROP_INTEGER_SKIP+BUFFER_PROP_DOUBLE_TOTAL-BUFFER_PROP_DOUBLE_SKIP)
enum ENUM_SORT_BUFFER_MODE
  {
//--- Sort by integer properties
   SORT_BY_BUFFER_INDEX_PLOT = 0,                           // Sort by the plotted buffer serial number
   SORT_BY_BUFFER_STATUS,                                   // Sort by buffer drawing style (status) (from the ENUM_BUFFER_STATUS enumeration)
   SORT_BY_BUFFER_TYPE,                                     // Sort by buffer type (from the ENUM_BUFFER_TYPE enumeration)
   SORT_BY_BUFFER_TIMEFRAME,                                // Sort by the buffer data period (timeframe)
   SORT_BY_BUFFER_ACTIVE,                                   // Sort by the buffer usage flag
   SORT_BY_BUFFER_DRAW_TYPE,                                // Sort by graphical construction type (from the ENUM_DRAW_TYPE enumeration)
   SORT_BY_BUFFER_ARROW_CODE,                               // Sort by the arrow code for DRAW_ARROW style
   SORT_BY_BUFFER_ARROW_SHIFT,                              // Sort by the vertical shift of the arrows for DRAW_ARROW style
   SORT_BY_BUFFER_LINE_STYLE,                               // Sort by the line style
   SORT_BY_BUFFER_LINE_WIDTH,                               // Sort by the line width
   SORT_BY_BUFFER_DRAW_BEGIN,                               // Sort by the number of initial bars that are not drawn and values in DataWindow
   SORT_BY_BUFFER_SHOW_DATA,                                // Sort by the flag of displaying construction values in DataWindow
   SORT_BY_BUFFER_SHIFT,                                    // Sort by the indicator graphical construction shift by time axis in bars
   SORT_BY_BUFFER_COLOR_INDEXES,                            // Sort by a number of attempts
   SORT_BY_BUFFER_COLOR,                                    // Sort by the drawing color
   SORT_BY_BUFFER_INDEX_BASE,                               // Sort by the basic data buffer index
   SORT_BY_BUFFER_INDEX_NEXT_BASE,                          // Sort by the index of the array to be assigned as the next indicator buffer
   SORT_BY_BUFFER_INDEX_NEXT_PLOT,                          // Sort by the index of the next drawn buffer
   SORT_BY_BUFFER_ID,                                       // Sort by ID of multiple buffers of the same indicator
   SORT_BY_BUFFER_IND_LINE_MODE,                            // Sort by indicator line
   SORT_BY_BUFFER_IND_HANDLE,                               // Sort by indicator handle that uses the buffer
   SORT_BY_BUFFER_IND_TYPE,                                 // Sort by indicator type that uses the buffer
   SORT_BY_BUFFER_IND_LINE_ADDITIONAL_NUM,                  // Sort by number of additional indicator line
//--- Sort by real properties
   SORT_BY_BUFFER_EMPTY_VALUE = FIRST_BUFFER_DBL_PROP,      // Sort by the empty value for plotting where nothing will be drawn
//--- Sort by string properties
   SORT_BY_BUFFER_SYMBOL = FIRST_BUFFER_STR_PROP,           // Sort by the buffer symbol
   SORT_BY_BUFFER_LABEL,                                    // Sort by the name of the graphical indicator series displayed in DataWindow
   SORT_BY_BUFFER_IND_NAME,                                 // Sort by indicator name that uses the buffer
   SORT_BY_BUFFER_IND_NAME_SHORT,                           // Sort by a short name of indicator that uses the buffer
  };
//+------------------------------------------------------------------+
```

Now, slightly improve the abstract buffer object class in \\MQL5\\Include\\DoEasy\\Objects\\Indicators\ **Buffer.mqh**.

In the class public section write the methods for settingand returning the additional line number:

```
public:
//--- Display the description of the buffer properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short buffer description in the journal (implementation in the descendants)
   virtual void      PrintShort(void) {;}

//--- Set (1) the arrow code, (2) vertical shift of arrows, (3) symbol, (4) timeframe, (5) buffer activity flag
//--- (6) drawing type, (7) number of initial bars without drawing, (8) flag of displaying construction values in DataWindow,
//--- (9) shift of the indicator graphical construction along the time axis, (10) line style, (11) line width,
//--- (12) total number of colors, (13) one drawing color, (14) color of drawing in the specified color index,
//--- (15) drawing colors from the color array, (16) empty value, (17) name of the graphical series displayed in DataWindow
   virtual void      SetArrowCode(const uchar code)                  { return;                                                               }
   virtual void      SetArrowShift(const int shift)                  { return;                                                               }
   void              SetSymbol(const string symbol)                  { this.SetProperty(BUFFER_PROP_SYMBOL,symbol);                          }
   void              SetTimeframe(const ENUM_TIMEFRAMES timeframe)   { this.SetProperty(BUFFER_PROP_TIMEFRAME,timeframe);                    }
   void              SetActive(const bool flag)                      { this.SetProperty(BUFFER_PROP_ACTIVE,flag);                            }
   void              SetDrawType(const ENUM_DRAW_TYPE draw_type);
   void              SetDrawBegin(const int value);
   void              SetShowData(const bool flag);
   void              SetShift(const int shift);
   void              SetStyle(const ENUM_LINE_STYLE style);
   void              SetWidth(const int width);
   void              SetColorNumbers(const int number);
   void              SetColor(const color colour);
   void              SetColor(const color colour,const uchar index);
   void              SetColors(const color &array_colors[]);
   void              SetEmptyValue(const double value);
   virtual void      SetLabel(const string label);
   void              SetID(const int id)                             { this.SetProperty(BUFFER_PROP_ID,id);                                  }
   void              SetIndicatorHandle(const int handle)            { this.SetProperty(BUFFER_PROP_IND_HANDLE,handle);                      }
   void              SetIndicatorType(const ENUM_INDICATOR type)     { this.SetProperty(BUFFER_PROP_IND_TYPE,type);                          }
   void              SetIndicatorName(const string name)             { this.SetProperty(BUFFER_PROP_IND_NAME,name);                          }
   void              SetIndicatorShortName(const string name)        { this.SetProperty(BUFFER_PROP_IND_NAME_SHORT,name);                    }
   void              SetLineMode(const ENUM_INDICATOR_LINE_MODE mode){ this.SetProperty(BUFFER_PROP_IND_LINE_MODE,mode);                     }
   void              SetIndicatorLineAdditionalNumber(const int number){this.SetProperty(BUFFER_PROP_IND_LINE_ADDITIONAL_NUM,number);        }

//--- Return (1) the serial number of the drawn buffer, (2) bound array index, (3) color buffer index,
//--- (4) index of the first free bound array, (5) index of the next drawn buffer, (6) buffer data period, (7) buffer status,
//--- (8) buffer type, (9) buffer usage flag, (10) arrow code, (11) arrow shift for DRAW_ARROW style,
//--- (12) number of initial bars that are not drawn and values in DataWindow, (13) graphical construction type,
//--- (14) flag of displaying construction values in DataWindow, (15) indicator graphical construction shift along the time axis,
//--- (16) drawing line style, (17) drawing line width, (18) number of colors, (19) drawing color, (20) number of buffers for construction
//--- (21) set empty value, (22) buffer symbol, (23) name of the indicator graphical series displayed in DataWindow
//--- (24) buffer ID, (25) indicator handle, (26) standard indicator type, (27) standard indicator name,
//--- (28) number of standard indicator calculated bars, (29) number of additional indicator line, (30) line type (main, signal, etc.)
   int               IndexPlot(void)                           const { return (int)this.GetProperty(BUFFER_PROP_INDEX_PLOT);                 }
   int               IndexBase(void)                           const { return (int)this.GetProperty(BUFFER_PROP_INDEX_BASE);                 }
   int               IndexColor(void)                          const { return (int)this.GetProperty(BUFFER_PROP_INDEX_COLOR);                }
   int               IndexNextBaseBuffer(void)                 const { return (int)this.GetProperty(BUFFER_PROP_INDEX_NEXT_BASE);            }
   int               IndexNextPlotBuffer(void)                 const { return (int)this.GetProperty(BUFFER_PROP_INDEX_NEXT_PLOT);            }
   ENUM_TIMEFRAMES   Timeframe(void)                           const { return (ENUM_TIMEFRAMES)this.GetProperty(BUFFER_PROP_TIMEFRAME);      }
   ENUM_BUFFER_STATUS Status(void)                             const { return (ENUM_BUFFER_STATUS)this.GetProperty(BUFFER_PROP_STATUS);      }
   ENUM_BUFFER_TYPE  TypeBuffer(void)                          const { return (ENUM_BUFFER_TYPE)this.GetProperty(BUFFER_PROP_TYPE);          }
   bool              IsActive(void)                            const { return (bool)this.GetProperty(BUFFER_PROP_ACTIVE);                    }
   uchar             ArrowCode(void)                           const { return (uchar)this.GetProperty(BUFFER_PROP_ARROW_CODE);               }
   int               ArrowShift(void)                          const { return (int)this.GetProperty(BUFFER_PROP_ARROW_SHIFT);                }
   int               DrawBegin(void)                           const { return (int)this.GetProperty(BUFFER_PROP_DRAW_BEGIN);                 }
   ENUM_DRAW_TYPE    DrawType(void)                            const { return (ENUM_DRAW_TYPE)this.GetProperty(BUFFER_PROP_DRAW_TYPE);       }
   bool              IsShowData(void)                          const { return (bool)this.GetProperty(BUFFER_PROP_SHOW_DATA);                 }
   int               Shift(void)                               const { return (int)this.GetProperty(BUFFER_PROP_SHIFT);                      }
   ENUM_LINE_STYLE   LineStyle(void)                           const { return (ENUM_LINE_STYLE)this.GetProperty(BUFFER_PROP_LINE_STYLE);     }
   int               LineWidth(void)                           const { return (int)this.GetProperty(BUFFER_PROP_LINE_WIDTH);                 }
   int               ColorsTotal(void)                         const { return (int)this.GetProperty(BUFFER_PROP_COLOR_INDEXES);              }
   color             Color(void)                               const { return (color)this.GetProperty(BUFFER_PROP_COLOR);                    }
   int               BuffersTotal(void)                        const { return (int)this.GetProperty(BUFFER_PROP_NUM_DATAS);                  }
   double            EmptyValue(void)                          const { return this.GetProperty(BUFFER_PROP_EMPTY_VALUE);                     }
   string            Symbol(void)                              const { return this.GetProperty(BUFFER_PROP_SYMBOL);                          }
   string            Label(void)                               const { return this.GetProperty(BUFFER_PROP_LABEL);                           }
   int               ID(void)                                  const { return (int)this.GetProperty(BUFFER_PROP_ID);                         }
   int               IndicatorHandle(void)                     const { return (int)this.GetProperty(BUFFER_PROP_IND_HANDLE);                 }
   ENUM_INDICATOR    IndicatorType(void)                       const { return (ENUM_INDICATOR)this.GetProperty(BUFFER_PROP_IND_TYPE);        }
   string            IndicatorName(void)                       const { return this.GetProperty(BUFFER_PROP_IND_NAME);                        }
   string            IndicatorShortName(void)                  const { return this.GetProperty(BUFFER_PROP_IND_NAME_SHORT);                  }
   int               IndicatorBarsCalculated(void)             const { return ::BarsCalculated((int)this.GetProperty(BUFFER_PROP_IND_HANDLE));}
   int               IndicatorLineAdditionalNumber(void)       const { return (int)this.GetProperty(BUFFER_PROP_IND_LINE_ADDITIONAL_NUM);    }
   int               IndicatorLineMode(void)                   const { return (int)this.GetProperty(BUFFER_PROP_IND_LINE_MODE);              }
```

Since now I will display description of indicator line from another enumeration (implemented above) now IndicatorLineMode() method will return an integer value instead of enumeration value ENUM\_INDICATOR\_LINE\_MODE.

In the same public section of the class, declare the method returning the indicator line description:

```
//--- Return descriptions of the (1) buffer status, (2) buffer type, (3) buffer usage flag, (4) flag of displaying construction values in DataWindow,
//--- (5) drawing line style, (6) set empty value, (7) graphical construction type, (8) used timeframe and (9) specified colors
   string            GetStatusDescription(bool draw_type=false)const;
   string            GetTypeBufferDescription(void)            const;
   string            GetActiveDescription(void)                const;
   string            GetShowDataDescription(void)              const;
   string            GetLineStyleDescription(void)             const;
   string            GetEmptyValueDescription(void)            const;
   string            GetDrawTypeDescription(void)              const;
   string            GetTimeframeDescription(void)             const;
   string            GetColorsDescription(void)                const;
   string            GetIndicatorLineModeDescription(void)     const;
```

and let's write its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Return description of indicator buffer line                      |
//+------------------------------------------------------------------+
string CBuffer::GetIndicatorLineModeDescription(void) const
  {
   uchar shift=0;
   switch(this.IndicatorType())
     {
      case IND_ENVELOPES   :
      case IND_FRACTALS    :
      case IND_GATOR       :
      case IND_BANDS       :  shift=2; break;
      case IND_ALLIGATOR   :  shift=5; break;
      case IND_ADX         :
      case IND_ADXW        :  shift=8; break;
      case IND_ICHIMOKU    :  shift=10;break;
      default              :  shift=0; break;
     }
   return ::StringSubstr(::EnumToString(ENUM_INDICATOR_LINE(this.GetProperty(BUFFER_PROP_IND_LINE_MODE)+shift)),10);
  }
//+------------------------------------------------------------------+
```

Here: declare the variable which stores shift value by which the enumeration constant values ENUM\_INDICATOR\_LINE\_MODE must be increased to get to the beginning of declare of line constants of the corresponding indicator in enumeration ENUM\_INDICATOR\_LINE.

For example, if we need to display description of Teeth line of Allegator indicator, the shift value is equal to 5, which points to constant INDICATOR\_LINE\_JAWS of enumeration ENUM\_INDICATOR\_LINE:

```
//+------------------------------------------------------------------+
//| Enumeration of indicator lines                                   |
//+------------------------------------------------------------------+
enum ENUM_INDICATOR_LINE
  {
   INDICATOR_LINE_MAIN,                                     // Main line
   INDICATOR_LINE_SIGNAL,                                   // Signal line
   INDICATOR_LINE_UPPER,                                    // Upper line
   INDICATOR_LINE_LOWER,                                    // Lower line
   INDICATOR_LINE_MIDDLE,                                   // Middle line
   INDICATOR_LINE_JAWS,                                     // Jaws line
   INDICATOR_LINE_TEETH,                                    // Teeth line
   INDICATOR_LINE_LIPS,                                     // Lips line
   INDICATOR_LINE_DI_PLUS,                                  // Line +DI
   INDICATOR_LINE_DI_MINUS,                                 // Line -DI
   INDICATOR_LINE_TENKAN_SEN,                               // Line Tenkan-sen
   INDICATOR_LINE_KIJUN_SEN,                                // Line Kijun-sen
   INDICATOR_LINE_SENKOU_SPANA,                             // Line Senkou Span A
   INDICATOR_LINE_SENKOU_SPANB,                             // Line Senkou Span B
   INDICATOR_LINE_CHIKOU_SPAN,                              // Line Chikou Span
   INDICATOR_LINE_ADDITIONAL,                               // Additional line
  };
//+------------------------------------------------------------------+
```

And since our buffer returns the value of Teeth line from GetProperty(BUFFER\_PROP\_IND\_LINE\_MODE) method and this value equals to one, adding the value of 5 to one results in constant index which is equal to 6, which points out to constant INDICATOR\_LINE\_TEETH.

As a result, the method returns string description of the received constant reduced to value "LINE\_TEETH".

In the class closed constructor set the default value to a new property of buffer object as -1 which will mean **non** additional indicator line:

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
```

In the method which returns description of buffer object integer property add a code block for return of new property description and improve the code block which displays indicator line description (now, we will display description using a new method written specifically for this purpose):

```
//+------------------------------------------------------------------+
//| Return description of a buffer's integer property                |
//+------------------------------------------------------------------+
string CBuffer::GetPropertyDescription(ENUM_BUFFER_PROP_INTEGER property)
  {
   return
     (
      property==BUFFER_PROP_INDEX_PLOT    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_PLOT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_STATUS        ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_STATUS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetStatusDescription()
         )  :
      property==BUFFER_PROP_TYPE          ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTypeBufferDescription()
         )  :
      property==BUFFER_PROP_TIMEFRAME     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_TIMEFRAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetTimeframeDescription()
         )  :
      property==BUFFER_PROP_ACTIVE        ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ACTIVE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetActiveDescription()
         )  :
      property==BUFFER_PROP_DRAW_TYPE     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_DRAW_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetDrawTypeDescription()
         )  :
      property==BUFFER_PROP_ARROW_CODE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ARROW_CODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_ARROW_SHIFT   ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ARROW_SHIFT)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_LINE_STYLE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_LINE_STYLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetLineStyleDescription()
         )  :
      property==BUFFER_PROP_LINE_WIDTH    ?
         (this.Status()==BUFFER_STATUS_ARROW ? CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ARROW_SIZE) :
          CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_LINE_WIDTH))+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_DRAW_BEGIN    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_DRAW_BEGIN)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_SHOW_DATA     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_SHOW_DATA)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetShowDataDescription()
         )  :
      property==BUFFER_PROP_SHIFT         ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_SHIFT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_COLOR_INDEXES ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_COLOR_NUM)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_COLOR   ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_COLOR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_BASE    ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_BASE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_NEXT_BASE ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_BASE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_INDEX_NEXT_PLOT ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_INDEX_NEXT_PLOT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_ID ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_ID)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_IND_LINE_MODE ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_IND_LINE_MODE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetIndicatorLineModeDescription()
         )  :
      property==BUFFER_PROP_IND_HANDLE ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_IND_HANDLE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_IND_TYPE ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_IND_TYPE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+::StringSubstr(::EnumToString((ENUM_INDICATOR)this.GetProperty(property)),4)
         )  :
      property==BUFFER_PROP_IND_LINE_ADDITIONAL_NUM ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_IND_LINE_ADDITIONAL_NUM)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)==WRONG_VALUE ? CMessage::Text(MSG_LIB_PROP_NOT_SET) : (string)this.GetProperty(property))
         )  :
      property==BUFFER_PROP_NUM_DATAS     ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_NUM_DATAS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==BUFFER_PROP_COLOR         ?  CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_COLOR)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetColorsDescription()
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Since we added one new integer property, in all classes of descendant objects of the abstract buffer object make improvement into virtual method which returns the flag of supporting by the object of this new property (using CBufferArrow class method as an example):

```
//+------------------------------------------------------------------+
//| Return 'true' if a buffer supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CBufferArrow::SupportProperty(ENUM_BUFFER_PROP_INTEGER property)
  {
   if(property==BUFFER_PROP_LINE_STYLE ||
      (
       this.TypeBuffer()==BUFFER_TYPE_CALCULATE &&
       property!=BUFFER_PROP_TYPE &&
       property!=BUFFER_PROP_INDEX_NEXT_BASE &&
       property!=BUFFER_PROP_IND_LINE_MODE &&
       property!=BUFFER_PROP_IND_HANDLE &&
       property!=BUFFER_PROP_IND_TYPE &&
       property!=BUFFER_PROP_IND_LINE_ADDITIONAL_NUM &&
       property!=BUFFER_PROP_ID
      )
     )
      return false;
   return true;
  }
//+------------------------------------------------------------------+
```

The same changes were made in all files of all buffer objects, such as **BufferArrow.mqh**, **BufferBars.mqh**, **BufferCandles.mqh**, **BufferFilling.mqh**, **BufferHistogram.mqh**, **BufferHistogram2.mqh**, **BufferLine.mqh**, **BufferSection.mqh** and **BufferZigZag.mqh**.

All methods for creation of standard indicator objects are located in buffer object collection class

in file \\MQL5\\Include\\DoEasy\\Collections\ **BuffersCollection.mqh**. Before adding two methods for creation of objects of standard indicators Ichimoku Kinko Hyo and Gator Oscillator let’s slightly improve the method for preparing data of the specified standard indicator for setting values on the current symbol chart. Since Ichimoku Kinko Hyo indicator has five drawn buffers but only three pointers at standard indicator buffer objects are passed to the method and, respectively, six variables - by the link (two for each buffer) for writing values of indicator lines into them, to the method it is necessary to add passing into it of four additional pointers at buffer objects (two for each drawn and calculated one) and passing of four more more variables by the link.

In the class body, add the method declaration with necessary values:

```
//--- Set values for the current chart to buffers of the specified standard indicator by the timeseries index in accordance with buffer object symbol/period
   bool                    SetDataBufferStdInd(const ENUM_INDICATOR std_ind,const int id,const int series_index,const datetime series_time,const char color_index=WRONG_VALUE);
private:
//--- Prepare data of the specified standard indicator for setting values on the current symbol chart
   int                     PreparingSetDataStdInd(CBuffer *buffer_data0,CBuffer *buffer_data1,CBuffer *buffer_data2,CBuffer *buffer_data3,CBuffer *buffer_data4,
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
                                                  double &value41);
public:

//--- Return the buffer (1) by the graphical series name, (2) timeframe,
```

In implementing the method written outside the class body add necessary changes:

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
     //--- Find bar index on a period which corresponds to the time of current bar beginning
     index_period=::iBarShift(buffer_calc0.Symbol(),buffer_calc0.Timeframe(),series_time,true);
     if(index_period==WRONG_VALUE || index_period>buffer_calc0.GetDataTotal()-1)
        return WRONG_VALUE;

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

     int series_index_start=series_index;
     //--- For the current chart we don’t need to calculate a number of bars processed - only one bar is available
     if(buffer_calc0.Symbol()==::Symbol() && buffer_calc0.Timeframe()==::Period())
       {
        series_index_start=series_index;
        num_bars=1;
       }
     else
       {
        //--- Get the bar time which the bar with index_period index falls into on a period and symbol of calculated buffer
        datetime time_period=::iTime(buffer_calc0.Symbol(),buffer_calc0.Timeframe(),index_period);
        if(time_period==0) return false;
        //--- Get the current chart bar which corresponds to the time
        series_index_start=::iBarShift(::Symbol(),::Period(),time_period,true);
        if(series_index_start==WRONG_VALUE) return WRONG_VALUE;
        //--- Calculate the number of bars on the current chart which are to be filled in with calculated buffer data
        num_bars=::PeriodSeconds(buffer_calc0.Timeframe())/::PeriodSeconds(PERIOD_CURRENT);
        if(num_bars==0) num_bars=1;
       }
     //--- Take values for color calculation
     if(buffer_calc0!=NULL)
        value01=(series_index_start+num_bars>buffer_data0.GetDataTotal()-1 ? value00 : buffer_data0.GetDataBufferValue(0,series_index_start+num_bars));
     if(buffer_calc1!=NULL)
        value11=(series_index_start+num_bars>buffer_data1.GetDataTotal()-1 ? value10 : buffer_data1.GetDataBufferValue(0,series_index_start+num_bars));
     if(buffer_calc2!=NULL)
        value21=(series_index_start+num_bars>buffer_data2.GetDataTotal()-1 ? value20 : buffer_data2.GetDataBufferValue(0,series_index_start+num_bars));
     if(buffer_calc3!=NULL)
        value31=(series_index_start+num_bars>buffer_data3.GetDataTotal()-1 ? value30 : buffer_data3.GetDataBufferValue(0,series_index_start+num_bars));
     if(buffer_calc4!=NULL)
        value41=(series_index_start+num_bars>buffer_data4.GetDataTotal()-1 ? value40 : buffer_data4.GetDataBufferValue(0,series_index_start+num_bars));

   return series_index_start;
  }
//+------------------------------------------------------------------+
```

We simply added the same handling, as for buffer objects which are already available in the method, to two new buffer objects pointers to which are passed to the method; and writing of values to the variables corresponding to buffers which are passed to the method by the link.

Since between two lines in Ichimoku Kinko Hyo standard indicator hatching is drawn, whereas, its type of hatching and color which corresponds to type and color of line is set for each line:

![](https://c.mql5.com/2/40/prnkdb04aZ.gif)

... we need a method which will return the pointer to standard indicator buffer object by indicator type, its ID and line so that we can take all parameters from it set for drawing its lines and can set them for auxiliary buffer object which serves for designing the appearance of our custom indicator.

In the public section of the class declare this method

```
public:

//--- Return the buffer (1) by the graphical series name, (2) timeframe,
//--- (3) Plot index, (4) object index in the collection list, (5) the last created,
//--- (6) standard indicator buffer by indicator type, its ID and line
   CBuffer                *GetBufferByLabel(const string plot_label);
   CBuffer                *GetBufferByTimeframe(const ENUM_TIMEFRAMES timeframe);
   CBuffer                *GetBufferByPlot(const int plot_index);
   CBuffer                *GetBufferByListIndex(const int index_list);
   CBuffer                *GetLastCreateBuffer(void);
   CBuffer                *GetBufferStdInd(const ENUM_INDICATOR indicator_type,const int id,const ENUM_INDICATOR_LINE_MODE line_mode,const char additional_id=WRONG_VALUE);
//--- Return buffer list (1) by ID, (2) standard indicator type, (3) type and ID
```

and let's write its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Return standard indicator buffer                                 |
//| by indicator type, its ID and line                               |
//+------------------------------------------------------------------+
CBuffer *CBuffersCollection::GetBufferStdInd(const ENUM_INDICATOR indicator_type,const int id,const ENUM_INDICATOR_LINE_MODE line_mode,const char additional_id=WRONG_VALUE)
  {
   CArrayObj *list=this.GetListBufferByTypeID(indicator_type,id);
   list=CSelect::ByBufferProperty(list,BUFFER_PROP_IND_LINE_MODE,line_mode,EQUAL);
   list=CSelect::ByBufferProperty(list,BUFFER_PROP_IND_LINE_ADDITIONAL_NUM,additional_id,EQUAL);
   if(list==NULL)
      return NULL;
   return list.At(0);
  }
//+------------------------------------------------------------------+
```

Here, all is simple: first, get the list of buffer objects by standard indicator type and its ID, then sort the obtained list by the type of standard indicator line and finally, from the remaining objects leave in the list only those having ID of auxiliary buffer (by default, all buffers are assigned the value of -1 which means absence of such ID).

From the method, return the first object from the sorted list.

**Supplement a method which prepares calculated buffer data of the specified standard indicator for handling Ichimoku Kinko Hyo and Gator Oscillator indicators.**

We need only to add the handling which corresponds to indicator type. Such handling is identical for all indicators and is already implemented in the method. But there is a little difference for standard indicator Gator Oscillator  — its second buffer has index 2instead of 1 like other two-buffer indicators do because buffer with index 1 belongs to color buffer of data buffer with index 0. I will not use standard indicator color buffers since I already created handling and setting of color for indicator lines, plus I can assign the required color for each bar, therefore, by default the color of columns of Gator Oscillator indicator will be calculated automatically by the library. And such calculation is already made. Where suitable, the user can pass the necessary color for each bar to data drawing methods of standard indicators in order to create own coloring of the indicator.

For Ichimoku Kinko Hyo indicator data preparation is identical to data preparation of other standard indicators - only two buffers more.

```
//+------------------------------------------------------------------+
//| Prepare calculated buffer data                                   |
//| of the specified standard indicator                              |
//+------------------------------------------------------------------+
int CBuffersCollection::PreparingDataBufferStdInd(const ENUM_INDICATOR std_ind,const int id,const int total_copy)
  {
   CArrayObj *list_ind=this.GetListBufferByTypeID(std_ind,id);
   CArrayObj *list;
   list_ind=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_TYPE,BUFFER_TYPE_CALCULATE,EQUAL);
   if(list_ind==NULL || list_ind.Total()==0)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_TEXT_BUFFER_TEXT_NO_BUFFER_OBJ));
      return 0;
     }
   CBufferCalculate *buffer=NULL;
   int copied=WRONG_VALUE;
   int idx0=0,idx1=1,idx2=2;
   switch((int)std_ind)
     {
   //--- Single-buffer standard indicators
      case IND_AC          :
      case IND_AD          :
      case IND_AMA         :
      case IND_AO          :
      case IND_ATR         :
      case IND_BEARS       :
      case IND_BULLS       :
      case IND_BWMFI       :
      case IND_CCI         :
      case IND_CHAIKIN     :
      case IND_DEMA        :
      case IND_DEMARKER    :
      case IND_FORCE       :
      case IND_FRAMA       :
      case IND_MA          :
      case IND_MFI         :
      case IND_MOMENTUM    :
      case IND_OBV         :
      case IND_OSMA        :
      case IND_RSI         :
      case IND_SAR         :
      case IND_STDDEV      :
      case IND_TEMA        :
      case IND_TRIX        :
      case IND_VIDYA       :
      case IND_VOLUMES     :
      case IND_WPR         :

        buffer=list_ind.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),0,buffer.Shift(),total_copy);
        return copied;

   //--- Multi-buffer standard indicators
      case IND_ENVELOPES   :
      case IND_FRACTALS    :
      case IND_MACD        :
      case IND_RVI         :
      case IND_STOCHASTIC  :
      case IND_GATOR       :
        if(std_ind==IND_GATOR)
          {
           idx0=0;
           idx1=2;
          }
        else
          {
           idx0=0;
           idx1=1;
          }
        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),idx0,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),idx1,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;
        return copied;

      case IND_ALLIGATOR   :
      case IND_ADX         :
      case IND_ADXW        :
      case IND_BANDS       :
        if(std_ind==IND_BANDS)
          {
           idx0=1;
           idx1=2;
           idx2=0;
          }
        else
          {
           idx0=0;
           idx1=1;
           idx2=2;
          }
        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),idx0,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),idx1,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,2,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),idx2,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;
        return copied;

      case IND_ICHIMOKU    :
        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_TENKAN_SEN,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),TENKANSEN_LINE,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_KIJUN_SEN,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),KIJUNSEN_LINE,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_SENKOU_SPANA,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),SENKOUSPANA_LINE,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_SENKOU_SPANB,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),SENKOUSPANB_LINE,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_CHIKOU_SPAN,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return 0;
        copied=buffer.FillAsSeries(buffer.IndicatorHandle(),CHIKOUSPAN_LINE,buffer.Shift(),total_copy);
        if(copied<total_copy) return 0;
        return copied;

      default:
        break;
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

**To buffer data clearing method of the specified standard indicator by the timeseries index add buffers handling for Ichimoku Kinko Hyo indicator:**

```
//+------------------------------------------------------------------+
//| Clear buffer data of the specified standard indicator            |
//| by the timeseries index                                          |
//+------------------------------------------------------------------+
void CBuffersCollection::ClearDataBufferStdInd(const ENUM_INDICATOR std_ind,const int id,const int series_index)
  {
//--- Get the list of buffer objects by type and ID
   CArrayObj *list_ind=this.GetListBufferByTypeID(std_ind,id);
   CArrayObj *list=NULL;
   if(list_ind==NULL || list_ind.Total()==0)
      return;
   list_ind=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_TYPE,BUFFER_TYPE_DATA,EQUAL);
   if(list_ind.Total()==0)
      return;
   CBuffer *buffer=NULL;
   switch((int)std_ind)
     {
   //--- Single-buffer standard indicators
      case IND_AC          :
      case IND_AD          :
      case IND_AMA         :
      case IND_AO          :
      case IND_ATR         :
      case IND_BEARS       :
      case IND_BULLS       :
      case IND_BWMFI       :
      case IND_CCI         :
      case IND_CHAIKIN     :
      case IND_DEMA        :
      case IND_DEMARKER    :
      case IND_FORCE       :
      case IND_FRAMA       :
      case IND_MA          :
      case IND_MFI         :
      case IND_MOMENTUM    :
      case IND_OBV         :
      case IND_OSMA        :
      case IND_RSI         :
      case IND_SAR         :
      case IND_STDDEV      :
      case IND_TEMA        :
      case IND_TRIX        :
      case IND_VIDYA       :
      case IND_VOLUMES     :
      case IND_WPR         :
        buffer=list_ind.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());
        break;

   //--- Multi-buffer standard indicators
      case IND_ENVELOPES   :
      case IND_FRACTALS    :
      case IND_MACD        :
      case IND_RVI         :
      case IND_STOCHASTIC  :
      case IND_GATOR       :
        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());
        break;

      case IND_ALLIGATOR   :
      case IND_ADX         :
      case IND_ADXW        :
      case IND_BANDS       :
        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,0,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,1,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,2,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());
        break;

      case IND_ICHIMOKU :
        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_TENKAN_SEN,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_KIJUN_SEN,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_SENKOU_SPANA,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_SENKOU_SPANB,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_CHIKOU_SPAN,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_ADDITIONAL,EQUAL);
        list=CSelect::ByBufferProperty(list,BUFFER_PROP_IND_LINE_ADDITIONAL_NUM,0,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());

        list=CSelect::ByBufferProperty(list_ind,BUFFER_PROP_IND_LINE_MODE,INDICATOR_LINE_MODE_ADDITIONAL,EQUAL);
        list=CSelect::ByBufferProperty(list,BUFFER_PROP_IND_LINE_ADDITIONAL_NUM,1,EQUAL);
        buffer=list.At(0);
        if(buffer==NULL) return;
        buffer.SetBufferValue(0,series_index,buffer.EmptyValue());
        break;

      default:
        break;
     }
  }
//+------------------------------------------------------------------+
```

Since further I will create the object of Ichimoku Kinko Hyo standard indicator and it will have two additional histogram buffers for designing indicator appearance, to the method code blocks for data clearing of these two auxiliary buffers are already added.

The rest is identical to buffer data clearing of other standard indicators implemented in previous articles.

**Improve the method which sets values for the current chart to buffers of the specified standard indicator by the timeseries index in accordance with buffer object symbol/period.**

Now the method will have more buffer objects, since Ichimoku Kinko Hyo has five of them, plus two auxiliary histogram buffers for designing indicator appearance. Respectively, the number of variables for storing values of all buffer object lines has increased:

```
//+------------------------------------------------------------------+
//| Sets values for the current chart to buffers of the specified    |
//| standard indicator by the timeseries index in accordance         |
//| with buffer object symbol/period                                 |
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
   if(list_data.Total()==0 || list_calc.Total()==0)
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
```

In each code block which handles its set of standard indicators, now pass more values to buffer data preparation method PreparingSetDataStdInd():

```
        series_index_start=PreparingSetDataStdInd(buffer_data0,buffer_data1,buffer_data2,buffer_data3,buffer_data4,
                                                  buffer_calc0,buffer_calc1,buffer_calc2,buffer_calc3,buffer_calc4,
                                                  ind_type,series_index,series_time,index_period,num_bars,
                                                  value00,value01,value10,value11,value20,value21,value30,value31,value40,value41);
```

In method end add handlers of standard indicators Ichimoku Kinko Hyo and Gator Oscillator:

```
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
```

The code is provided with comments on the actions which differ from the same code blocks for handling of other standard indicators considered in previous articles. This has concerned only getting and handling of data for design of Ichimoku Kinko Hyo indicator. For Gator Oscillator indicator all logic remained the same as for other standard indicators.

Now, write methods for creation of objects of standard indicators Ichimoku Kinko Hyo and Gator Oscillator.

**Method for Gator Oscillator creation:**

```
//+------------------------------------------------------------------+
//| Create multi-symbol multi-period Gator                           |
//+------------------------------------------------------------------+
int CBuffersCollection::CreateGator(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                    const int jaw_period,
                                    const int jaw_shift,
                                    const int teeth_period,
                                    const int teeth_shift,
                                    const int lips_period,
                                    const int lips_shift,
                                    const ENUM_MA_METHOD ma_method,
                                    const ENUM_APPLIED_PRICE applied_price,
                                    const int id=WRONG_VALUE)
  {
//--- Create indicator handle and set default ID
   int num_bars=::PeriodSeconds(timeframe)/::PeriodSeconds(PERIOD_CURRENT);
   int shift=::fmin(jaw_shift,teeth_shift);
   int handle=::iGator(symbol,timeframe,jaw_period,jaw_shift,teeth_period,teeth_shift,lips_period,lips_shift,ma_method,applied_price);
   int identifier=(id==WRONG_VALUE ? IND_GATOR : id);
   color array_colors[3]={clrGreen,clrRed,clrGreen};
   CBuffer *buff=NULL;
   if(handle!=INVALID_HANDLE)
     {
      //--- Create histogram buffer from the zero line
      this.CreateHistogram();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters of Up
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(shift*num_bars);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_GATOR);
      buff.SetShowData(true);
      buff.SetLineMode(INDICATOR_LINE_MODE_UPPER);
      buff.SetIndicatorName("Gator Oscillator");
      buff.SetIndicatorShortName("Gator("+symbol+","+TimeframeDescription(timeframe)+": "+(string)jaw_period+","+(string)teeth_period+","+(string)lips_period+")");
      buff.SetLabel("Gator("+symbol+","+TimeframeDescription(timeframe)+": "+(string)jaw_period+","+(string)teeth_period+","+(string)lips_period+","+") Up");
      buff.SetColors(array_colors);

      //--- Create histogram buffer from the zero line
      this.CreateHistogram();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters of Down
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(shift*num_bars);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_GATOR);
      buff.SetShowData(true);
      buff.SetLineMode(INDICATOR_LINE_MODE_LOWER);
      buff.SetIndicatorName("Gator Oscillator");
      buff.SetIndicatorShortName("Gator("+symbol+","+TimeframeDescription(timeframe)+": "+(string)jaw_period+","+(string)teeth_period+","+(string)lips_period+")");
      buff.SetLabel("Gator("+symbol+","+TimeframeDescription(timeframe)+": "+(string)jaw_period+","+(string)teeth_period+","+(string)lips_period+","+") Down");
      buff.SetColors(array_colors);

      //--- Create calculated buffer of Up, in which standard indicator data will be stored
      this.CreateCalculate();
      //--- Get the last created buffer object (calculated) and set to it all the necessary parameters of Up
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(shift);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_GATOR);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLineMode(INDICATOR_LINE_MODE_UPPER);
      buff.SetIndicatorName("Gator Oscillator");
      buff.SetLabel("Gator("+symbol+","+TimeframeDescription(timeframe)+": "+(string)jaw_period+","+(string)teeth_period+","+(string)lips_period+","+") Up");

      //--- Create calculated buffer of Teeth in which standard indicator data will be stored
      this.CreateCalculate();
      //--- Get the last created buffer object (calculated) and set to it all the necessary parameters of Teeth
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(shift);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_GATOR);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLineMode(INDICATOR_LINE_MODE_LOWER);
      buff.SetIndicatorName("Gator Oscillator");
      buff.SetLabel("Gator("+symbol+","+TimeframeDescription(timeframe)+": "+(string)jaw_period+","+(string)teeth_period+","+(string)lips_period+","+") Down");
     }
   return handle;
  }
//+------------------------------------------------------------------+
```

At creation of indicator handle pass to it data of shift of Alligator indicator lines on which data Gator is calculated, as they are in inputs irrespective of the fact that the indicator may be drawn on “non-native” timeframe - all these data are required for indicator’s internal calculation. Visual shift of lines of Gator standard indicator is calculated as the minimal value from values of shift of Jaw and Teeth lines of Alligator indicator on which Gator is calculated. And this visual shift must be multiplied by a number of bars which should be displayed on the current timeframe. And we do it setting the shift value for indicator’s plotted buffer objects. For calculated buffer objects set the shift without multiplication by a number of bars.

**Method for Ichimoku Kinko Hyo creation:**

```
//+------------------------------------------------------------------+
//| Create multi-symbol multi-period Ichimoku                        |
//+------------------------------------------------------------------+
int CBuffersCollection::CreateIchimoku(const string symbol,const ENUM_TIMEFRAMES timeframe,
                                       const int tenkan_sen,
                                       const int kijun_sen,
                                       const int senkou_span_b,
                                       const int id=WRONG_VALUE)
  {
//--- Create indicator handle and set default ID
   int num_bars=::PeriodSeconds(timeframe)/::PeriodSeconds(PERIOD_CURRENT);
   int handle=::iIchimoku(symbol,timeframe,tenkan_sen,kijun_sen,senkou_span_b);
   int identifier=(id==WRONG_VALUE ? IND_ICHIMOKU : id);
   color array_colors[1]={clrRed};
   CBuffer *buff=NULL;
   if(handle!=INVALID_HANDLE)
     {
      //--- Create line buffer
      this.CreateLine();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters of Tenkan-Sen
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_TENKAN_SEN);
      buff.SetShowData(true);
      buff.SetLabel("Tenkan-Sen("+symbol+","+TimeframeDescription(timeframe)+": "+(string)tenkan_sen+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetColors(array_colors);

      //--- Create line buffer
      this.CreateLine();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters of Kijun-Sen
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_KIJUN_SEN);
      buff.SetShowData(true);
      buff.SetLabel("Kijun-Sen("+symbol+","+TimeframeDescription(timeframe)+": "+(string)kijun_sen+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");
      array_colors[0]=clrBlue;
      buff.SetColors(array_colors);

      //--- Create line buffer
      this.CreateLine();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters of Senkou Span A
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(kijun_sen*num_bars);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_SENKOU_SPANA);
      buff.SetShowData(true);
      buff.SetLabel("Senkou Span A("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");
      array_colors[0]=clrSandyBrown;
      buff.SetColors(array_colors);
      buff.SetStyle(STYLE_DOT);

      //--- Create line buffer
      this.CreateLine();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters of Senkou Span B
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(kijun_sen*num_bars);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_SENKOU_SPANB);
      buff.SetShowData(true);
      buff.SetLabel("Senkou Span B("+symbol+","+TimeframeDescription(timeframe)+": "+(string)senkou_span_b+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");
      array_colors[0]=clrThistle;
      buff.SetColors(array_colors);
      buff.SetStyle(STYLE_DOT);

      //--- Create line buffer
      this.CreateLine();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters of Chikou Span
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_CHIKOU_SPAN);
      buff.SetShowData(true);
      buff.SetLabel("Chikou Span("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");
      array_colors[0]=clrLime;
      buff.SetColors(array_colors);

      //--- Create histogram buffer on two lines for displaying the histogram of Senkou Span A
      this.CreateHistogram2();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(kijun_sen*num_bars);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_ADDITIONAL);
      buff.SetShowData(false);
      buff.SetLabel("Senkou Span A("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorLineAdditionalNumber(0);
      //--- Get buffer data of Senkou Span A and set values of line color, width and style to the histogram
      CBuffer *tmp=GetBufferStdInd(IND_ICHIMOKU,identifier,INDICATOR_LINE_MODE_SENKOU_SPANA);
      array_colors[0]=(tmp!=NULL ? tmp.Color() : clrSandyBrown);
      buff.SetColors(array_colors);
      buff.SetWidth(tmp!=NULL ? tmp.LineWidth() : 1);
      buff.SetStyle(tmp!=NULL ? tmp.LineStyle() : STYLE_DOT);

      //--- Create histogram buffer on two lines for displaying the histogram of Senkou Span B
      this.CreateHistogram2();
      //--- Get the last created buffer object (drawn) and set to it all the necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(kijun_sen*num_bars);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_ADDITIONAL);
      buff.SetShowData(false);
      buff.SetLabel("Senkou Span B("+symbol+","+TimeframeDescription(timeframe)+": "+(string)senkou_span_b+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorLineAdditionalNumber(1);
      //--- Get buffer data of Senkou Span B and set values of line color, width and style to the histogram
      tmp=GetBufferStdInd(IND_ICHIMOKU,identifier,INDICATOR_LINE_MODE_SENKOU_SPANB);
      array_colors[0]=(tmp!=NULL ? tmp.Color() : clrThistle);
      buff.SetColors(array_colors);
      buff.SetWidth(tmp!=NULL ? tmp.LineWidth() : 1);
      buff.SetStyle(tmp!=NULL ? tmp.LineStyle() : STYLE_DOT);

      //--- Create calculated buffer in which data of Tenkan-Sen line will be stored
      this.CreateCalculate();
      //--- Get the last created buffer object (calculated) and set to it all the necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_TENKAN_SEN);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLabel("Tenkan-Sen("+symbol+","+TimeframeDescription(timeframe)+": "+(string)tenkan_sen+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");

      //--- Create calculated buffer in which data of Kijun-Sen line will be stored
      this.CreateCalculate();
      //--- Get the last created buffer object (calculated) and set to it all the necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_KIJUN_SEN);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLabel("Kijun-Sen("+symbol+","+TimeframeDescription(timeframe)+": "+(string)kijun_sen+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");

      //--- Create calculated buffer in which data of Senkou Span A line will be stored
      this.CreateCalculate();
      //--- Get the last created buffer object (calculated) and set to it all the necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(kijun_sen);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_SENKOU_SPANA);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLabel("Senkou Span A("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");

      //--- Create calculated buffer in which data of Senkou Span B line will be stored
      this.CreateCalculate();
      //--- Get the last created buffer object (calculated) and set to it all the necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetShift(kijun_sen);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_SENKOU_SPANB);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLabel("Senkou Span B("+symbol+","+TimeframeDescription(timeframe)+": "+(string)senkou_span_b+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");

      //--- Create calculated buffer in which data of Chikou Span line will be stored
      this.CreateCalculate();
      //--- Get the last created buffer object (calculated) and set to it all the necessary parameters
      buff=this.GetLastCreateBuffer();
      if(buff==NULL)
         return INVALID_HANDLE;
      buff.SetSymbol(symbol);
      buff.SetTimeframe(timeframe);
      buff.SetID(identifier);
      buff.SetIndicatorHandle(handle);
      buff.SetIndicatorType(IND_ICHIMOKU);
      buff.SetLineMode(INDICATOR_LINE_MODE_CHIKOU_SPAN);
      buff.SetEmptyValue(EMPTY_VALUE);
      buff.SetLabel("Chikou Span("+symbol+","+TimeframeDescription(timeframe)+")");
      buff.SetIndicatorName("Ichimoku Kinko Hyo");
      buff.SetIndicatorShortName("Ichimoku("+symbol+","+TimeframeDescription(timeframe)+")");
     }
   return handle;
  }
//+------------------------------------------------------------------+
```

Here, like with creation of the object of Gator standard indicator when creating the handle pass to it inputs without change and when creating buffer objects for displaying Senkou Span A and Senkou Span B lines and additional buffers for designing area between these two lines as histograms, multiply the shift to get the necessary number of bars which must be displayed on the current chart. When creating objects of calculated buffers set the shift without multiplication by a number of bars of the current chart.

Calculation period of Kijun-Sen line serves as the shift for Senkou Span A and Senkou Span B lines.

When creating additional histogram buffers first, set all indicator’s standard parameters to it and then get buffer object of the line which corresponds to the histogram and set drawing parameters for the histogram the same as for indicator line getting values from taken buffer of the line.

For buffer objects of histograms set a property **not to** display its line in the data window \- these buffers are necessary only for design and their values fully correspond to those indicator lines from which they get their data.

This concludes the improvement of classes and methods of the library for creation of multi-symbol multi-period standard indicators. Now, we have the full set of methods for creation of any standard and custom multi-indicators in custom programs. Certainly, there are still shortcomings to be gradually fixed at further development of library functionality.

### Test

To perform the test, let's take the indicator from the [previous article](https://www.mql5.com/en/articles/8331#node03) and create two new indicators in a new folder \\MQL5\\Indicators\\TestDoEasy\ **Part51\**

under names **TestDoEasyPart51\_1.mq5** and **TestDoEasyPart51\_2.mq5**.

They will differ only by parameter #property indicator\_chart\_window, or #property indicator\_separate\_window, since one of them will be drawn on the main chart and the second one - in a subwindow. Also, in one case we will create Ichimoku Kinko Hyo indicator, and in the second case — Gator Oscillator.

Delete from inputs strings setting indicators type and a shift of lines:

```
sinput   ENUM_INDICATOR       InpIndType        =  IND_AC;        // Type standard indicator
sinput   int                  InpShift          =  0;             // Indicator line shift
```

In handler OnInit() in file TestDoEasyPart51\_1.mq5 create the object of Ichimoku Kinko Hyo:

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Write the name of the working timeframe selected in the settings to the InpUsedTFs variable
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
   if(!engine.BufferCreateIchimoku(InpUsedSymbols,InpPeriod,9,26,52,1))
     {
      Print(TextByLanguage("Error. Indicator not created"));
      return INIT_FAILED;
     }
//--- Check the number of buffers specified in the 'properties' block
   if(engine.BuffersPropertyPlotsTotal()!=indicator_plots)
      Alert(TextByLanguage("Attention! Value of \"indicator_plots\" should be "),engine.BuffersPropertyPlotsTotal());
   if(engine.BuffersPropertyBuffersTotal()!=indicator_buffers)
      Alert(TextByLanguage("Attention! Value of \"indicator_buffers\" should be "),engine.BuffersPropertyBuffersTotal());

//--- Create the color array and set non-default colors to all buffers within the collection
//--- (commented out since default colors are already set in methods of standard indicator creation)
//--- (we can always set required colors either for all indicators like here or for each one individually)
   //color array_colors[]={clrGreen,clrRed,clrGray};
   //engine.BuffersSetColors(array_colors);

//--- Display short descriptions of created indicator buffers
   engine.BuffersPrintShort();

//--- Set a short name for the indicator, data capacity and levels
   string label=engine.BufferGetIndicatorShortNameByTypeID(IND_ICHIMOKU,1);
   IndicatorSetString(INDICATOR_SHORTNAME,label);
   SetIndicatorLevels(InpUsedSymbols,IND_ICHIMOKU);

//--- Succeeded
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

In handler OnCalculate() handle only one created indicator:

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
      engine.GetBuffersCollection().SetDataBufferStdInd(IND_ICHIMOKU,1,i,time[i]);
     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

In file TestDoEasyPart51\_2.mq5, make the same changes, but instead of creating and handling Ichimoku Kinko Hyo create and handle Gator Oscillator. And set a hint for the preprocessor to create an indicator working in a subwindow:

```
//+------------------------------------------------------------------+
//|                                           TestDoEasyPart51_2.mq5 |
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
#property indicator_buffers 6
#property indicator_plots   2
```

Launch both indicators on EURUSD H1 chart having indicated in indicator settings to use data of EURUSD H4 for calculation. And compare them to standard indicators:

![](https://c.mql5.com/2/40/8TLHJbMiyE.gif)

As you can see, data of both indicators match data of standard indicators.

The full codes of both indicators are provided in the files attached hereto.

### What's next?

In the next article we will start working on compatibility of standard indicator object classes with MQL4.

All files of the current version of the library are attached below together with test indicator files. You can download them and test everything.

Leave your comments, questions and suggestions in the comments to the article.

Please keep in mind that here I have developed MQL5 test indicators for MetaTrader 5.

The attached files are intended only for MetaTrader 5. The current library version has not been tested in MetaTrader 4.

[Back to contents](https://www.mql5.com/en/articles/8354#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8354](https://www.mql5.com/ru/articles/8354)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8354.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8354/mql5.zip "Download MQL5.zip")(3770.76 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/357384)**

![Advanced resampling and selection of CatBoost models by brute-force method](https://c.mql5.com/2/41/yandex_catboost__1.png)[Advanced resampling and selection of CatBoost models by brute-force method](https://www.mql5.com/en/articles/8662)

This article describes one of the possible approaches to data transformation aimed at improving the generalizability of the model, and also discusses sampling and selection of CatBoost models.

![A scientific approach to the development of trading algorithms](https://c.mql5.com/2/40/algorithm_2.png)[A scientific approach to the development of trading algorithms](https://www.mql5.com/en/articles/8231)

The article considers the methodology for developing trading algorithms, in which a consistent scientific approach is used to analyze possible price patterns and to build trading algorithms based on these patterns. Development ideals are demonstrated using examples.

![Basic math behind Forex trading](https://c.mql5.com/2/40/56.png)[Basic math behind Forex trading](https://www.mql5.com/en/articles/8274)

The article aims to describe the main features of Forex trading as simply and quickly as possible, as well as share some basic ideas with beginners. It also attempts to answer the most tantalizing questions in the trading community along with showcasing the development of a simple indicator.

![CatBoost machine learning algorithm from Yandex with no Python or R knowledge required](https://c.mql5.com/2/41/yandex_catboost_2.png)[CatBoost machine learning algorithm from Yandex with no Python or R knowledge required](https://www.mql5.com/en/articles/8657)

The article provides the code and the description of the main stages of the machine learning process using a specific example. To obtain the model, you do not need Python or R knowledge. Furthermore, basic MQL5 knowledge is enough — this is exactly my level. Therefore, I hope that the article will serve as a good tutorial for a broad audience, assisting those interested in evaluating machine learning capabilities and in implementing them in their programs.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/8354&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070398318514738467)

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