---
title: Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics
url: https://www.mql5.com/en/articles/20632
categories: Trading Systems, Indicators
relevance_score: 15
scraped_at: 2026-01-22T17:09:25.660066
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/20632&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048826759371136877)

MetaTrader 5 / Trading systems


### Introduction

In our [previous article (Part 1)](https://www.mql5.com/en/articles/20610), we built a Pivot-Based Trend Indicator in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). It calculated fast and slow pivot lines over user-defined periods, detected trend direction relative to these lines, and signaled trend starts with arrows, optionally extending the lines beyond the current bar. In Part 2, we create a Gauge-Style [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI) Display that uses canvas and needle mechanics. This model shows Relative Strength Index values on a circular gauge with a dynamic needle, color-coded ranges for overbought and oversold levels, and customizable legends. It also integrates traditional line plotting for comprehensive momentum analysis. We will cover the following topics:

1. [Understanding the Gauge-Style RSI Indicator Framework](https://www.mql5.com/en/articles/20632#para2)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/20632#para3)
3. [Backtesting](https://www.mql5.com/en/articles/20632#para4)
4. [Conclusion](https://www.mql5.com/en/articles/20632#para5)

By the end, you’ll have a functional MQL5 indicator for gauge-based Relative Strength Index visualization, ready for customization—let’s dive in!

### Understanding the Gauge-Style RSI Indicator Framework

The gauge-style [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") indicator reimagines the standard Relative Strength Index as a circular dial, where a needle dynamically points to the current momentum value on a scale from 0 to 100, highlighting overbought conditions above 70 and oversold below 30 through distinct color zones for quick visual assessment. It incorporates tick marks for precise reading, legends for context, like the indicator name and value display, and a traditional line plot in a separate window to complement the gauge with historical data trends. We thought of this [dial gauge](https://en.wikipedia.org/wiki/Indicator_(distance_amplifying_instrument) "https://en.wikipedia.org/wiki/Indicator_(distance_amplifying_instrument)") approach because it is intuitive and appealing to do analysis and display the results, and we used a standard, well-known calculation approach, but in the future can incorporate complex data for rendering and annotations.

For now, we aim to build a modular framework that separates graphical layers for the scale and needle, allowing independent transparency and updates for efficiency. We will start by outlining input parameters for customization, such as angle ranges, colors, and tick intervals, then define structures for elements like arcs, pies, and labels to organize drawing logic. From there, we will create a base class to handle creation, parameter setting, and redrawing, ensuring the gauge initializes properly and responds to new Relative Strength Index values from the market. Our plan is to leverage canvas drawing for all visual components, integrate with the built-in Relative Strength Index calculation, and manage event handlers for seamless operation across chart updates. In brief, here is a visual representation of our objectives. We have detailed most of the elements for ease in understanding.

![DIAL GAUGE FRAMEWORK](https://c.mql5.com/2/186/Screenshot_2025-12-14_005812.png)

### Implementation in MQL5

To create the indicator in MQL5, just open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is created, in the coding environment, we will define the indicator properties and settings, such as the number of [buffers](https://www.mql5.com/en/docs/series/bufferdirection), plots, and individual line properties, such as the color, width, and label.

```
//+------------------------------------------------------------------+
//|                           1. Gauge-Based RSI Indicator Part1.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"

#property indicator_separate_window
#property indicator_buffers 2
#property indicator_plots 1
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrDodgerBlue
#property indicator_style1 STYLE_SOLID
#property indicator_width1 2
#property indicator_label1 "RSI"
#property indicator_minimum 0
#property indicator_maximum 100
#property indicator_level1 30
#property indicator_level2 70
#property indicator_levelcolor clrGray
#property indicator_levelstyle STYLE_DOT
```

We begin the implementation by defining the indicator's metadata with [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) directives, specifying that it draws in a separate subwindow with [indicator\_separate\_window](https://www.mql5.com/en/docs/basis/preprosessor/compilation), allocating 2 buffers with "indicator\_buffers", and configuring 1 plot with "indicator\_plots". For the plot, we set its type to [DRAW\_LINE](https://www.mql5.com/en/docs/customind/indicators_examples/draw_line), color to dodger blue, style to solid, width to 2, and labeled it "RSI". We also fix the vertical scale from 0 to 100 using "indicator\_minimum" and "indicator\_maximum", and add dotted gray levels at 30 and 70 via "indicator\_level1", "indicator\_level2", "indicator\_levelcolor", and "indicator\_levelstyle" to highlight oversold and overbought zones. These properties create a clean RSI line visualization in its own window for momentum analysis. Then, we can include the canvas library for custom drawing elements.

```
#include <Canvas\Canvas.mqh>
```

We include the [Canvas](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) library with " [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Canvas\\Canvas.mqh>" to incorporate built-in tools for custom graphical rendering, such as the [CCanvas](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) class that enables bitmap creation and drawing operations on the chart. This prepares the program for constructing visual elements like arcs, circles, and text in the gauge display. Then, we can define structures to organize the graphical components.

```
//+------------------------------------------------------------------+
//| Circle Structure                                                 |
//+------------------------------------------------------------------+
struct Struct_Circle {                     // Define circle structure
   int centerX;                            // Store center X coordinate
   int centerY;                            // Store center Y coordinate
   int radius;                             // Store radius
   color clr;                              // Store color
   bool display;                           // Store display flag
};

//+------------------------------------------------------------------+
//| Arc Structure                                                    |
//+------------------------------------------------------------------+
struct Struct_Arc {                        // Define arc structure
   int centerX;                            // Store center X coordinate
   int centerY;                            // Store center Y coordinate
   int radius;                             // Store radius
   double startAngle;                      // Store start angle in radians
   double endAngle;                        // Store end angle in radians
   color clr;                              // Store color
   bool display;                           // Store display flag
};

//+------------------------------------------------------------------+
//| Line Structure                                                   |
//+------------------------------------------------------------------+
struct Struct_Line {                       // Define line structure
   int startX;                             // Store start X coordinate
   int startY;                             // Store start Y coordinate
   int endX;                               // Store end X coordinate
   int endY;                               // Store end Y coordinate
   color clr;                              // Store color
};

//+------------------------------------------------------------------+
//| Dot Structure                                                    |
//+------------------------------------------------------------------+
struct Struct_Dot {                        // Define dot structure
   int x;                                  // Store X coordinate
   int y;                                  // Store Y coordinate
   color clr;                              // Store color
};

//+------------------------------------------------------------------+
//| Pie/Sector Structure                                             |
//+------------------------------------------------------------------+
struct Struct_Pie {                        // Define pie structure
   int centerX;                            // Store center X coordinate
   int centerY;                            // Store center Y coordinate
   int radius;                             // Store radius
   int eraseRadius;                        // Store erase radius
   double startAngle;                      // Store start angle in radians
   double endAngle;                        // Store end angle in radians
   double eraseStartAngle;                 // Store erase start angle in radians
   double eraseEndAngle;                   // Store erase end angle in radians
   color clr;                              // Store color
   color eraseClr;                         // Store erase color
};

//+------------------------------------------------------------------+
//| Range Structure                                                  |
//+------------------------------------------------------------------+
struct Struct_Range {                      // Define range structure
   bool active;                            // Store active status
   double startValue;                      // Store start value
   double endValue;                        // Store end value
   color clr;                              // Store color
   Struct_Pie pie;                         // Store pie structure
};

//+------------------------------------------------------------------+
//| Case Structure                                                   |
//+------------------------------------------------------------------+
struct Struct_Case {                       // Define case structure
   bool display;                           // Store display flag
   Struct_Circle circle;                   // Store circle structure
};

//+------------------------------------------------------------------+
//| Scale Marks Structure                                            |
//+------------------------------------------------------------------+
struct Struct_ScaleMarks {                 // Define scale marks structure
   double minValue;                        // Store minimum value
   double maxValue;                        // Store maximum value
   double valueRange;                      // Store value range
   bool forwardDirection;                  // Store forward direction flag
   int nullMarkPosition;                   // Store null mark position
   double nullMarkAngle;                   // Store null mark angle
   int decimalPlaces;                      // Store decimal places
   int majorTickLength;                    // Store major tick length
   int mediumTickLength;                   // Store medium tick length
   int minorTickLength;                    // Store minor tick length
   double minAngle;                        // Store minimum angle
   double maxAngle;                        // Store maximum angle
   double angleRange;                      // Store angle range
   double multiplier;                      // Store multiplier
   string gaugeName;                       // Store gauge name
   string currentValue;                    // Store current value
   string units;                           // Store units
   int tickFontSize;                       // Store tick font size
   string tickFontName;                    // Store tick font name
   uint tickFontFlags;                     // Store tick font flags
   int tickFontGap;                        // Store tick font gap
};

//+------------------------------------------------------------------+
//| Label Area Size Structure                                        |
//+------------------------------------------------------------------+
struct Struct_LabelAreaSize {              // Define label area size structure
   int height;                             // Store height
   int width;                              // Store width
   int diagonal;                           // Store diagonal
};

//+------------------------------------------------------------------+
//| Gauge Legend Parameters Structure                                |
//+------------------------------------------------------------------+
struct Struct_GaugeLegendParams {          // Define gauge legend parameters structure
   bool enable;                            // Store enable flag
   string text;                            // Store text
   uint radius;                            // Store radius
   double angle;                           // Store angle
   uint fontSize;                          // Store font size
   string fontName;                        // Store font name
   bool italic;                            // Store italic flag
   bool bold;                              // Store bold flag
   color textColor;                        // Store text color
};

//+------------------------------------------------------------------+
//| Gauge Legend String Structure                                    |
//+------------------------------------------------------------------+
struct Struct_GaugeLegendString {          // Define gauge legend string structure
   string text;                            // Store text
   int radius;                             // Store radius
   double angle;                           // Store angle
   int fontSize;                           // Store font size
   string fontName;                        // Store font name
   uint fontFlags;                         // Store font flags
   color textColor;                        // Store text color
   color backgroundColor;                  // Store background color
   uint decimalPlaces;                     // Store decimal places
   uint x;                                 // Store x coordinate
   uint y;                                 // Store y coordinate
   bool draw;                              // Store draw flag
};

//+------------------------------------------------------------------+
//| Gauge Label Structure                                            |
//+------------------------------------------------------------------+
struct Struct_GaugeLabel {                 // Define gauge label structure
   Struct_GaugeLegendString description;   // Store description
   Struct_GaugeLegendString units;         // Store units
   Struct_GaugeLegendString multiplier;    // Store multiplier
   Struct_GaugeLegendString value;         // Store value
};

//+------------------------------------------------------------------+
//| Scale Layer Structure                                            |
//+------------------------------------------------------------------+
struct Struct_ScaleLayer {                 // Define scale layer structure
   string objectName;                      // Store object name
   CCanvas obj_Canvas;                     // Store canvas object
   uchar transparency;                     // Store transparency
   color caseColor;                        // Store case color
   Struct_Case externalCase;               // Store external case
   int borderSize;                         // Store border size
   Struct_Case internalCase;               // Store internal case
   int borderGap;                          // Store border gap
   int externalLabelArea;                  // Store external label area
   int externalScaleGap;                   // Store external scale gap
   Struct_Arc scaleArc;                    // Store scale arc
   int internalScaleGap;                   // Store internal scale gap
   int internalLabelArea;                  // Store internal label area
   Struct_ScaleMarks scaleMarks;           // Store scale marks
   Struct_GaugeLabel gaugeLabel;           // Store gauge label
   Struct_Range ranges[4];                 // Store ranges array
};

//+------------------------------------------------------------------+
//| Needle Structure                                                 |
//+------------------------------------------------------------------+
struct Struct_Needle {                     // Define needle structure
   int tipRadius;                          // Store tip radius
   int tailRadius;                         // Store tail radius
   int x[4];                               // Store x coordinates array
   int y[4];                               // Store y coordinates array
   int fillStyle;                          // Store fill style
   color clr;                              // Store color
};

//+------------------------------------------------------------------+
//| Needle Layer Structure                                           |
//+------------------------------------------------------------------+
struct Struct_NeedleLayer {                // Define needle layer structure
   string objectName;                      // Store object name
   CCanvas obj_Canvas;                     // Store canvas object
   uchar transparency;                     // Store transparency
   Struct_Arc needleCenter;                // Store needle center
   Struct_Needle needle;                   // Store needle
};

//+------------------------------------------------------------------+
//| Range Parameters Structure                                       |
//+------------------------------------------------------------------+
struct Struct_RangeParams {                // Define range parameters structure
   bool enable;                            // Store enable flag
   double start;                           // Store start value
   double end;                             // Store end value
   color clr;                              // Store color
};

//+------------------------------------------------------------------+
//| Gauge Input Parameters Structure                                 |
//+------------------------------------------------------------------+
struct Struct_GaugeInputParams {           // Define gauge input parameters structure
   int xOffset;                            // Store x offset
   int yOffset;                            // Store y offset
   int anchorCorner;                       // Store anchor corner
   int relativeMode;                       // Store relative mode
   string relativeObjectName;              // Store relative object name
   int scaleAngleRange;                    // Store scale angle range
   int rotationAngle;                      // Store rotation angle
   color scaleColor;                       // Store scale color
   int scaleStyle;                         // Store scale style
   bool displayScaleArc;                   // Store display scale arc flag
   double minScaleValue;                   // Store minimum scale value
   double maxScaleValue;                   // Store maximum scale value
   int scaleMultiplier;                    // Store scale multiplier
   int tickStyle;                          // Store tick style
   int tickSize;                           // Store tick size
   double majorTickInterval;               // Store major tick interval
   int mediumTicksPerMajor;                // Store medium ticks per major
   int minorTicksPerInterval;              // Store minor ticks per interval
   int tickFontSize;                       // Store tick font size
   string tickFontName;                    // Store tick font name
   bool tickFontItalic;                    // Store tick font italic flag
   bool tickFontBold;                      // Store tick font bold flag
   color tickFontColor;                    // Store tick font color
   Struct_RangeParams ranges[4];           // Store ranges array
   color caseColor;                        // Store case color
   int borderStyle;                        // Store border style
   color borderColor;                      // Store border color
   int borderGapSize;                      // Store border gap size
   Struct_GaugeLegendParams description;   // Store description
   Struct_GaugeLegendParams units;         // Store units
   Struct_GaugeLegendParams multiplier;    // Store multiplier
   Struct_GaugeLegendParams value;         // Store value
   int needleCenterStyle;                  // Store needle center style
   color needleCenterColor;                // Store needle center color
   color needleColor;                      // Store needle color
   int needleFillStyle;                    // Store needle fill style
};
```

For the structures, we first define the "Struct\_Circle" structure using the [struct](https://www.mql5.com/en/docs/basis/types/classes) keyword to hold properties for circular elements, including center coordinates, radius, color, and a display flag. We have added comments for clarity. Next, we create the "Struct\_Arc" structure for arc shapes, storing center points, radius, start and end angles in radians, color, and display status. We then set up the "Struct\_Line" structure for lines, with start and end coordinates and color. The "Struct\_Dot" structure is defined for points, containing x and y positions and color. For sector or pie slices, we define the "Struct\_Pie" structure with center, radii for drawing and erasing, angle ranges for both, and colors for filling and erasing. We establish the "Struct\_Range" structure to manage value ranges, including active status, start and end values, color, and an embedded "Struct\_Pie" for visualization.

The "Struct\_Case" structure is created for gauge casing, with a display flag and an included "Struct\_Circle". We define the "Struct\_ScaleMarks" structure to organize scale details like value limits, ranges, directions, tick lengths, angles, multipliers, font properties, and gaps. For label sizing, we create the "Struct\_LabelAreaSize" structure holding height, width, and diagonal measurements. The "Struct\_GaugeLegendParams" structure is set for legend configurations, including the enable flag, text, radius, angle, font details, and color. We then define "Struct\_GaugeLegendString" for specific legend strings, with text, position, font flags, colors, decimal places, coordinates, and draw flag. The "Struct\_GaugeLabel" structure groups multiple legend strings for description, units, multiplier, and value. For layering, we create "Struct\_ScaleLayer" to manage the scale component, including object name, canvas instance, transparency, case color, border details, label areas, scale arc, marks, labels, and an array of ranges.

We define "Struct\_Needle" for the pointer, with tip and tail radii, coordinate arrays, fill style, and color. The "Struct\_NeedleLayer" structure handles the needle layer, with object name, canvas, transparency, center arc, and needle data. For range settings, we set "Struct\_RangeParams" with enable flag, start and end values, and color. Finally, we define "Struct\_GaugeInputParams" to consolidate all input options, covering offsets, anchors, scale angles, colors, display flags, value bounds, multipliers, tick configurations, font flags, range arrays, case properties, legends, and needle styles. With that done, we can now move on to creating the standard RSI indicator, which is the easiest, then we will create the complex gauge-based indicator later, since we will need these standard RSI data either way. This is the approach we used to achieve that.

```
int rsiHandle;                             //--- Declare RSI handle
double rsiBuffer[];                        //--- Declare RSI buffer

//+------------------------------------------------------------------+
//| Initialize Indicator                                             |
//+------------------------------------------------------------------+
int OnInit() {
   rsiHandle = iRSI(_Symbol, _Period, 14, 4);       //--- Get RSI handle
   if(rsiHandle == INVALID_HANDLE)                  //--- Check handle
      return(INIT_FAILED);                          //--- Return failed
   SetIndexBuffer(0, rsiBuffer, INDICATOR_DATA);    //--- Set index buffer
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE); //--- Set plot empty
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, 14 - 1); //--- Set draw begin
   IndicatorSetInteger(INDICATOR_LEVELS, 2);        //--- Set levels
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 0, 30); //--- Set level 0
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 1, 70); //--- Set level 1
   return(INIT_SUCCEEDED);                          //--- Return succeeded
}

//+------------------------------------------------------------------+
//| Calculate Indicator                                              |
//+------------------------------------------------------------------+
int OnCalculate(const int ratesTotal,
                const int prevCalculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tickVolume[],
                const long &volume[],
                const int &spread[]) {
   if(CopyBuffer(rsiHandle, 0, 0, ratesTotal, rsiBuffer) < 0) { //--- Copy buffer
      Print("RSI CopyBuffer error for plot"); //--- Print error
      return(0);                           //--- Return 0
   }
   return(ratesTotal);                     //--- Return rates total
}
```

To create the standard RSI indicator, on the global scope, we declare the "rsiHandle" as an [integer](https://www.mql5.com/en/docs/basis/types/integer) to store the reference to it, and "rsiBuffer" as a double array to hold the calculated Relative Strength Index values.

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we initialize "rsiHandle" using the [iRSI](https://www.mql5.com/en/docs/indicators/irsi) function with the current symbol, timeframe, a period of 14, and close prices. If the handle is invalid, we return [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to halt initialization. We then bind buffer index 0 to "rsiBuffer" for indicator data with [SetIndexBuffer](https://www.mql5.com/en/docs/customind/setindexbuffer), set empty plot values to "EMPTY\_VALUE" via [PlotIndexSetDouble](https://www.mql5.com/en/docs/customind/plotindexsetdouble), and specify the drawing start at bar 13 using [PlotIndexSetInteger](https://www.mql5.com/en/docs/customind/plotindexsetinteger) to account for the Relative Strength Index calculation period. Additionally, we configure two indicator levels with "IndicatorSetInteger", setting them to 30 and 70 through "IndicatorSetDouble" for oversold and overbought thresholds, before returning [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode).

In the [OnCalculate](https://www.mql5.com/en/docs/event_handlers/oncalculate) event handler, we copy data from the Relative Strength Index handle's buffer 0 into "rsiBuffer" for the total available rates using the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function. If the copy fails, we print an error message and return 0 to stop processing; otherwise, we return the rates total to indicate successful calculation. It is always a good programming language to run your program on every milestone to ascertain everything is going on smoothly. When we run the program, we get the following outcome.

![TRADITIONAL RSI INDICATOR](https://c.mql5.com/2/186/Screenshot_2025-12-14_014628.png)

From the image, we can see that we have the standard indicator set. That was quite easy and straightforward. What we need to do now is move on to the next point, where we define a class for drawing the gauge so we can reuse it later with ease when we need to create more gauges, but first, we will define some helper functions.

```
//+------------------------------------------------------------------+
//| Degrees to Radians                                               |
//+------------------------------------------------------------------+
double DegreesToRadians(double degrees) {
   return((M_PI * degrees) / 180.0);       //--- Convert degrees to radians
}

//+------------------------------------------------------------------+
//| Normalize Radians                                                |
//+------------------------------------------------------------------+
double NormalizeRadians(double angle) {
   while(angle < 0.0) angle += 2.0 * M_PI; //--- Adjust negative
   while(angle >= 2.0 * M_PI) angle -= 2.0 * M_PI; //--- Adjust positive
   return(angle);                          //--- Return normalized
}

//+------------------------------------------------------------------+
//| Get Tick Font Gap                                                |
//+------------------------------------------------------------------+
int GetTickFontGap(Struct_ScaleMarks &scaleMarks, int stringLength) {
   int gap = 0;                            //--- Initialize gap
   Struct_LabelAreaSize areaSize;          //--- Declare area size
   CCanvas obj_Canvas_temp;                //--- Declare temp canvas
   if(!obj_Canvas_temp.FontSet(scaleMarks.tickFontName, scaleMarks.tickFontSize, scaleMarks.tickFontFlags, 0)) //--- Set font
      return gap;                          //--- Return gap
   string str = "000";                     //--- Set str
   obj_Canvas_temp.TextSize(str, areaSize.width, areaSize.height); //--- Get text size
   if(areaSize.width == 0 || areaSize.height == 0) //--- Check size
      return gap;                          //--- Return gap
   areaSize.diagonal = (int)MathCeil(MathSqrt((double)(areaSize.width * areaSize.width + areaSize.height * areaSize.height))); //--- Calculate diagonal
   gap = (int)(areaSize.diagonal * 0.5);   //--- Set gap
   return gap;                             //--- Return gap
}

//+------------------------------------------------------------------+
//| Get Tick Label Area Size                                         |
//+------------------------------------------------------------------+
bool GetTickLabelAreaSize(int &areaSize, Struct_ScaleMarks &scaleMarks, int stringLength) {
   CCanvas obj_Canvas_temp;                //--- Declare temp canvas
   int width = 0, height = 0;              //--- Initialize width height
   if(!obj_Canvas_temp.FontSet(scaleMarks.tickFontName, scaleMarks.tickFontSize, scaleMarks.tickFontFlags, 0)) //--- Set font
      return false;                        //--- Return false
   string str = "000";                     //--- Set str
   obj_Canvas_temp.TextSize(str, width, height); //--- Get text size
   if(width == 0 || height == 0)           //--- Check size
      return false;                        //--- Return false
   areaSize = (int)MathCeil(MathSqrt((double)(width * width + height * height))); //--- Calculate area size
   return true;                            //--- Return true
}
```

First, we define the "DegreesToRadians" function to convert an input degree value to radians by multiplying it with [M\_PI](https://www.mql5.com/en/docs/constants/namedconstants/mathsconstants) and dividing by 180. Then, we create the "NormalizeRadians" function to ensure an angle is within the 0 to 2pi range, adding 2pi for negative values or subtracting 2pi for those exceeding 2pi. The "GetTickFontGap" function computes a spacing gap for tick labels using a temporary [CCanvas](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) object; it sets the font from the "Struct\_ScaleMarks" parameter, measures the text size of "000", calculates the diagonal of the width and height using [MathCeil](https://www.mql5.com/en/docs/math/mathceil) and [MathSqrt](https://www.mql5.com/en/docs/math/mathsqrt), and returns half that diagonal as the gap, defaulting to 0 if font setting fails or sizes are zero.

We also define the "GetTickLabelAreaSize" function to determine the area needed for tick labels with a temp "CCanvas", setting the font, measuring "000" text dimensions, and computing the ceiling of the diagonal sqrt for the output area size, returning false on failure or zero sizes. Armed with these functions, we can create a complete class with full method definitions. Let us first declare a base class with all the methods that we need and then define them later.

```
//+------------------------------------------------------------------+
//| Base Gauge Class                                                 |
//+------------------------------------------------------------------+
class CGaugeBase                           // Define base gauge class
{
private:
   int relativeX;                          //--- Store relative X
   int relativeY;                          //--- Store relative Y
   int centerX;                            //--- Store center X
   int centerY;                            //--- Store center Y
   double currentValue;                    //--- Store current value
   bool initializationComplete;            //--- Store initialization complete flag
   void Draw();                            //--- Declare draw method
   void CalculateNeedle();                 //--- Declare calculate needle method
   void RedrawNeedle(double value);        //--- Declare redraw needle method
   void CalculateAndDrawLegends();         //--- Declare calculate and draw legends method
   void CalculateAndDrawLegendString(Struct_GaugeLegendString &legendString); //--- Declare calculate and draw legend string method
   void RedrawScaleMarks(Struct_Case &internalCase, Struct_Arc &scaleArc, int borderGap); //--- Declare redraw scale marks method
   void CalculateRanges(int borderGap);    //--- Declare calculate ranges method
   bool IsValidRange(int index);           //--- Declare check valid range method
   void NormalizeRangeValues(double &minValue, double &maxValue, double val0, double val1); //--- Declare normalize range values method
   void CalculateRangePie(Struct_Range &range, int innerRadius, int radialGap, int outerRadius, double rangeStart, double rangeEnd, color rangeClr, color caseClr); //--- Declare calculate range pie method
   void DrawRanges();                      //--- Declare draw ranges method
   void DrawRange(Struct_Range &range);    //--- Declare draw range method
   void CalculateInnerOuterRadii(int &innerRadius, int &outerRadius, int baseRadius, int tickLength, int tickStyle); //--- Declare calculate inner outer radii method
   bool DrawTick(double angle, int length, Struct_Arc &scaleArc); //--- Declare draw tick method
   double CalculateAngleDelta(double angle1, double angle2, int direction); //--- Declare calculate angle delta method
   bool GetLabelAreaSize(Struct_LabelAreaSize &areaSize, Struct_GaugeLegendString &legendString); //--- Declare get label area size method
   bool EraseLegendString(Struct_GaugeLegendString &legendString, color eraseClr); //--- Declare erase legend string method
   bool RedrawValueDisplay(double value);  //--- Declare redraw value display method
   void SetLegendStringParams(Struct_GaugeLegendString &legendString, Struct_GaugeLegendParams &param, int minRadius, int radiusDelta); //--- Declare set legend string params method
   void CalculateCaseElements(Struct_Case &externalCase, Struct_Case &internalCase, int borderSize, int borderGap); //--- Declare calculate case elements method
   void DrawCaseElements(Struct_Case &externalCase, Struct_Case &internalCase); //--- Declare draw case elements method
protected:
   Struct_GaugeInputParams inputParams;    //--- Store input parameters
   Struct_ScaleLayer scaleLayer;           //--- Store scale layer
   Struct_NeedleLayer needleLayer;         //--- Store needle layer
   int m_radius;                           //--- Store radius
public:
   bool Create(string name, int x, int y, int size, string relativeObjectName, int relativeMode, int corner, bool background, uchar scaleTransparency, uchar needleTransparency); //--- Declare create method
   bool CalculateLocation();               //--- Declare calculate location method
   void Redraw();                          //--- Declare redraw method
   void NewValue(double value);            //--- Declare new value method
   void Delete();                          //--- Declare delete method
   void SetScaleParameters(int angleRange, int rotation, double minValue, double maxValue, int multiplier, int style, color scaleClr, bool displayArc = false); //--- Declare set scale parameters method
   void SetTickParameters(int style, int size, double majorInterval, int mediumPerMajor, int minorPerInterval); //--- Declare set tick parameters method
   void SetTickLabelFont(int fontSize, string fontName, bool italic, bool bold, color fontClr = clrBlack); //--- Declare set tick label font method
   void SetCaseParameters(color caseClr, int borderStyle, color borderClr, int borderGapSize); //--- Declare set case parameters method
   void SetLegendParameters(int legendType, bool enable, string text, int radius, double angle, uint fontSize, string fontName, bool italic, bool bold, color textClr = clrDarkGray); //--- Declare set legend parameters method
   void SetLegendParam(Struct_GaugeLegendParams &legendParam, bool enable, string text, int radius, double angle, uint fontSize, string fontName, bool italic, bool bold, color textClr = clrDarkGray); //--- Declare set legend param method
   void SetRangeParameters(int index, bool enable, double start, double end, color rangeClr); //--- Declare set range parameters method
   void SetNeedleParameters(int centerStyle, color centerClr, color needleClr, int fillStyle); //--- Declare set needle parameters method
};
```

We define the "CGaugeBase" class to serve as the core for building and managing the gauge visualization, encapsulating all logic for creation, drawing, and updates. In the [private](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_access_rights) section, we declare variables to track relative and center positions, the current displayed value, and an initialization flag. We also declare methods such as "Draw" for rendering the gauge, "CalculateNeedle" for pointer positioning, "RedrawNeedle" to update the needle based on a value, "CalculateAndDrawLegends" and "CalculateAndDrawLegendString" for handling text elements, "RedrawScaleMarks" for ticks and labels, "CalculateRanges" and related helpers like "IsValidRange", "NormalizeRangeValues", "CalculateRangePie", "DrawRanges", and "DrawRange" for color zones, "CalculateInnerOuterRadii" for radius computations, "DrawTick" for individual marks, "CalculateAngleDelta" for angular differences, "GetLabelAreaSize" and "EraseLegendString" for label management, "RedrawValueDisplay" for updating shown values, "SetLegendStringParams" for legend setup, and "CalculateCaseElements" with "DrawCaseElements" for the gauge casing.

The [protected section](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_access_rights) includes structures for input parameters, scale layer, needle layer, and the radius to allow derived classes access while keeping them encapsulated.

In the [public section](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_access_rights), we provide methods like "Create" to initialize the gauge with name, position, size, relativity, corner, background, and transparencies; "CalculateLocation" for positioning; "Redraw" for full refresh; "NewValue" to update with a new value; "Delete" for cleanup; and setter methods such as "SetScaleParameters" for angle and value ranges, "SetTickParameters" for tick intervals, "SetTickLabelFont" for font styling, "SetCaseParameters" for borders, "SetLegendParameters" and "SetLegendParam" for legends, "SetRangeParameters" for zones, and "SetNeedleParameters" for pointer styles. We can now define the gauge creation and location logic.

```
//+------------------------------------------------------------------+
//| Create Gauge                                                     |
//+------------------------------------------------------------------+
bool CGaugeBase::Create(string name, int x, int y, int size, string relativeObjectName, int relativeMode, int corner, bool background, uchar scaleTransparency, uchar needleTransparency) {
   initializationComplete = false;         //--- Set initialization complete flag to false
   m_radius = size / 2;                    //--- Calculate radius
   inputParams.xOffset = x;                //--- Set x offset
   inputParams.yOffset = y;                //--- Set y offset
   inputParams.anchorCorner = corner;      //--- Set anchor corner
   inputParams.relativeMode = relativeMode;//--- Set relative mode
   inputParams.relativeObjectName = relativeObjectName; //--- Set relative object name
   if(!CalculateLocation())                //--- Check location calculation
      return false;                        //--- Return false if failed
   int canvasWidthHeight = (m_radius + 5) * 2; //--- Calculate canvas size
   scaleLayer.objectName = name + "_s";    //--- Set scale layer object name
   ObjectDelete(0, scaleLayer.objectName); //--- Delete scale layer object
   if(!scaleLayer.obj_Canvas.CreateBitmapLabel(scaleLayer.objectName, centerX, centerY, canvasWidthHeight, canvasWidthHeight, COLOR_FORMAT_ARGB_NORMALIZE)) //--- Create scale canvas
      return false;                        //--- Return false if failed
   ObjectSetInteger(0, scaleLayer.objectName, OBJPROP_CORNER, inputParams.anchorCorner); //--- Set corner property
   ObjectSetInteger(0, scaleLayer.objectName, OBJPROP_ANCHOR, ANCHOR_CENTER); //--- Set anchor property
   ObjectSetInteger(0, scaleLayer.objectName, OBJPROP_BACK, background); //--- Set back property
   needleLayer.objectName = name + "_n";   //--- Set needle layer object name
   ObjectDelete(0, needleLayer.objectName);//--- Delete needle layer object
   if(!needleLayer.obj_Canvas.CreateBitmapLabel(needleLayer.objectName, centerX, centerY, canvasWidthHeight, canvasWidthHeight, COLOR_FORMAT_ARGB_NORMALIZE)) //--- Create needle canvas
      return false;                        //--- Return false if failed
   ObjectSetInteger(0, needleLayer.objectName, OBJPROP_CORNER, inputParams.anchorCorner); //--- Set corner property
   ObjectSetInteger(0, needleLayer.objectName, OBJPROP_ANCHOR, ANCHOR_CENTER); //--- Set anchor property
   ObjectSetInteger(0, needleLayer.objectName, OBJPROP_BACK, background); //--- Set back property
   scaleLayer.transparency = 255 - scaleTransparency; //--- Set scale transparency
   needleLayer.transparency = 255 - needleTransparency; //--- Set needle transparency
   return true;                            //--- Return true
}

//+------------------------------------------------------------------+
//| Calculate Gauge Center Location                                  |
//+------------------------------------------------------------------+
bool CGaugeBase::CalculateLocation() {
   bool locationChanged = false;           //--- Initialize location changed flag
   int cX = m_radius;                      //--- Set initial X
   int cY = m_radius;                      //--- Set initial Y
   cX += inputParams.xOffset;              //--- Add X offset
   cY += inputParams.yOffset;              //--- Add Y offset
   if(centerX != cX || centerY != cY) {    //--- Check if position changed
      centerX = cX;                        //--- Update center X
      centerY = cY;                        //--- Update center Y
      locationChanged = true;              //--- Set changed flag
   }
   return locationChanged;                 //--- Return changed flag
}
```

Here, we implement the "Create" [method](https://www.mql5.com/en/docs/basis/types/classes) in the "CGaugeBase" class to initialize the gauge, starting by resetting the "initializationComplete" flag to false and computing "m\_radius" as half the provided size. We store the input parameters, including x and y offsets, anchor corner, relative mode, and relative object name. If the "CalculateLocation" method fails, we return false. We then determine the canvas dimensions as twice the sum of radius plus 5, assign the scale layer object name by appending "\_s" to the base name, delete any existing object with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete), and create a new bitmap label for the scale canvas using "CreateBitmapLabel" at the center position with [ARGB](https://www.mql5.com/en/docs/common/resourcecreate) normalization. We configure its properties with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) for corner, anchor to center, and background status. Similarly, for the needle layer, we append "\_n" to the name, delete if it exists, create the bitmap label, and set the same properties. Finally, we adjust transparencies by subtracting the input values from 255 and return true on success.

We also define the "CalculateLocation" method to update the gauge's center position, initializing a change flag to false, setting temporary cX and cY to the radius, adding the stored offsets, and checking if they differ from the current center values. If different, we update the center coordinates and set the flag to true before returning it. To create the gauge before setting its parameters, we will need to declare a global object from our class and use it to get access to the class members and methods as below.

```
//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CGaugeBase gauge;                          //--- Declare gauge object

// In the initialization, call the class method
if(!gauge.Create("rsi_gauge", 30, 30, 250, "", 0, 0, false, 0, 0)) //--- Create gauge
   return(INIT_FAILED);                 //--- Return failed
```

We declare a global instance of the "CGaugeBase" class named "gauge" to manage the overall gauge visualization throughout the program. In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we call the "Create" method on this instance with parameters for name "rsi\_gauge", x and y positions at 30, size of 250, empty relative object, relative mode 0, corner 0, background false, and both transparencies at 0. If creation fails, we return [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to stop the indicator setup. On compilation, we get the following outcome.

![INITIAL GAUGE CREATION](https://c.mql5.com/2/186/Screenshot_2025-12-14_021344.png)

We can see that we have created the initial gauge silhouette. What now needs to be done is create the casing and the other parameters on the base that we have currently by defining the rest of the methods. We will define the gauge parameters first.

```
//+------------------------------------------------------------------+
//| Set Scale Parameters                                             |
//+------------------------------------------------------------------+
void CGaugeBase::SetScaleParameters(int angleRange, int rotation, double minValue, double maxValue, int multiplier, int style, color scaleClr, bool displayArc) {
   inputParams.scaleAngleRange = angleRange; //--- Set scale angle range
   inputParams.rotationAngle = rotation;   //--- Set rotation angle
   inputParams.minScaleValue = minValue;   //--- Set minimum scale value
   inputParams.maxScaleValue = maxValue;   //--- Set maximum scale value
   inputParams.scaleMultiplier = multiplier; //--- Set scale multiplier
   inputParams.scaleStyle = style;         //--- Set scale style
   inputParams.scaleColor = scaleClr;      //--- Set scale color
   inputParams.displayScaleArc = displayArc; //--- Set display scale arc flag
}

//+------------------------------------------------------------------+
//| Set Tick Parameters                                              |
//+------------------------------------------------------------------+
void CGaugeBase::SetTickParameters(int style, int size, double majorInterval, int mediumPerMajor, int minorPerInterval) {
   inputParams.tickStyle = style;          //--- Set tick style
   inputParams.tickSize = size;            //--- Set tick size
   inputParams.majorTickInterval = majorInterval; //--- Set major tick interval
   inputParams.mediumTicksPerMajor = mediumPerMajor; //--- Set medium ticks per major
   inputParams.minorTicksPerInterval = minorPerInterval; //--- Set minor ticks per interval
}

//+------------------------------------------------------------------+
//| Set Tick Label Font                                              |
//+------------------------------------------------------------------+
void CGaugeBase::SetTickLabelFont(int fontSize, string fontName, bool italic, bool bold, color fontClr) {
   inputParams.tickFontSize = fontSize;    //--- Set tick font size
   inputParams.tickFontName = fontName;    //--- Set tick font name
   inputParams.tickFontItalic = italic;    //--- Set tick font italic flag
   inputParams.tickFontBold = bold;        //--- Set tick font bold flag
   inputParams.tickFontColor = fontClr;    //--- Set tick font color
}

//+------------------------------------------------------------------+
//| Set Case Parameters                                              |
//+------------------------------------------------------------------+
void CGaugeBase::SetCaseParameters(color caseClr, int borderStyle, color borderClr, int borderGapSize) {
   inputParams.caseColor = caseClr;        //--- Set case color
   inputParams.borderStyle = borderStyle;  //--- Set border style
   inputParams.borderColor = borderClr;    //--- Set border color
   inputParams.borderGapSize = borderGapSize; //--- Set border gap size
}

//+------------------------------------------------------------------+
//| Set Legend Parameters                                            |
//+------------------------------------------------------------------+
void CGaugeBase::SetLegendParameters(int legendType, bool enable, string text, int radius, double angle, uint fontSize, string fontName, bool italic, bool bold, color textClr) {
   switch(legendType) {                    //--- Switch on legend type
   case 0:                                 //--- Handle description
      SetLegendParam(inputParams.description, enable, text, radius, angle, fontSize, fontName, italic, bold, textClr); //--- Set description param
      break;                               //--- Break
   case 1:                                 //--- Handle units
      SetLegendParam(inputParams.units, enable, text, radius, angle, fontSize, fontName, italic, bold, textClr); //--- Set units param
      break;                               //--- Break
   case 2:                                 //--- Handle multiplier
      SetLegendParam(inputParams.multiplier, enable, text, radius, angle, fontSize, fontName, italic, bold, textClr); //--- Set multiplier param
      break;                               //--- Break
   case 3:                                 //--- Handle value
      SetLegendParam(inputParams.value, enable, text, radius, angle, fontSize, fontName, italic, bold, textClr); //--- Set value param
      break;                               //--- Break
   }
}

//+------------------------------------------------------------------+
//| Set Individual Legend Parameter                                  |
//+------------------------------------------------------------------+
void CGaugeBase::SetLegendParam(Struct_GaugeLegendParams &legendParam, bool enable, string text, int radius, double angle, uint fontSize, string fontName, bool italic, bool bold, color textClr) {
   legendParam.enable = enable;            //--- Set enable flag
   legendParam.text = text;                //--- Set text
   legendParam.radius = radius;            //--- Set radius
   legendParam.angle = angle;              //--- Set angle
   legendParam.fontSize = fontSize;        //--- Set font size
   legendParam.fontName = fontName;        //--- Set font name
   legendParam.italic = italic;            //--- Set italic flag
   legendParam.bold = bold;                //--- Set bold flag
   legendParam.textColor = textClr;        //--- Set text color
}

//+------------------------------------------------------------------+
//| Set Range Parameters                                             |
//+------------------------------------------------------------------+
void CGaugeBase::SetRangeParameters(int index, bool enable, double start, double end, color rangeClr) {
   if(index >= 0 && index < 4) {           //--- Check index range
      inputParams.ranges[index].enable = enable; //--- Set enable flag
      inputParams.ranges[index].start = start; //--- Set start
      inputParams.ranges[index].end = end; //--- Set end
      inputParams.ranges[index].clr = rangeClr; //--- Set color
   }
}

//+------------------------------------------------------------------+
//| Set Needle Parameters                                            |
//+------------------------------------------------------------------+
void CGaugeBase::SetNeedleParameters(int centerStyle, color centerClr, color needleClr, int fillStyle) {
   inputParams.needleCenterStyle = centerStyle; //--- Set needle center style
   inputParams.needleCenterColor = centerClr; //--- Set needle center color
   inputParams.needleColor = needleClr;    //--- Set needle color
   inputParams.needleFillStyle = fillStyle;//--- Set needle fill style
}
```

Here, we define the "SetScaleParameters" method in the "CGaugeBase" [class](https://www.mql5.com/en/docs/basis/types/classes) to configure the gauge's scale by assigning the provided angle range, rotation angle, minimum and maximum values, multiplier index, style, color, and arc display flag to the corresponding fields in "inputParams". We define the "SetTickParameters" method to set tick-related options, storing the style, size, major interval, number of medium ticks per major, and minor ticks per interval in "inputParams". The "SetTickLabelFont" method handles font settings for tick labels, updating "inputParams" with font size, name, italic and bold flags, and color, defaulting to black if unspecified. We create the "SetCaseParameters" method to define the gauge's casing, assigning case color, border style, border color, and gap size to "inputParams".

For legends, we implement the "SetLegendParameters" method, which uses a switch statement based on legend type (0 for description, 1 for units, 2 for multiplier, 3 for value) to route parameters to the appropriate "Struct\_GaugeLegendParams" via the "SetLegendParam" helper method, with default text color as dark gray. The "SetLegendParam" method directly sets the enable flag, text, radius, angle, font size, name, italic, bold, and text color in the passed legend parameter structure. We add the "SetRangeParameters" method to configure up to four ranges, validating the index between 0 and 3 before setting enable, start, end, and color in the "inputParams.ranges" array. Finally, the "SetNeedleParameters" method assigns the center style, center color, needle color, and fill style to "inputParams" for the pointer configuration. For drawing the other elements, we adopt the following logic.

```
//+------------------------------------------------------------------+
//| Calculate Case Elements                                          |
//+------------------------------------------------------------------+
void CGaugeBase::CalculateCaseElements(Struct_Case &externalCase, Struct_Case &internalCase, int borderSize, int borderGap) {
   if(borderSize > 0) {                    //--- Check border size
      externalCase.circle.centerX = scaleLayer.scaleArc.centerX; //--- Set external center X
      externalCase.circle.centerY = scaleLayer.scaleArc.centerY; //--- Set external center Y
      externalCase.circle.radius = m_radius; //--- Set external radius
      externalCase.circle.clr = inputParams.borderColor; //--- Set external color
      externalCase.display = true;         //--- Set display flag
   } else                                  //--- Handle no border
      externalCase.display = false;        //--- Set display flag false
   internalCase.circle.centerX = scaleLayer.scaleArc.centerX; //--- Set internal center X
   internalCase.circle.centerY = scaleLayer.scaleArc.centerY; //--- Set internal center Y
   internalCase.circle.radius = m_radius - borderSize; //--- Set internal radius
   internalCase.circle.clr = inputParams.caseColor; //--- Set internal color
   internalCase.display = true;            //--- Set display flag
}

//+------------------------------------------------------------------+
//| Draw Case Elements                                               |
//+------------------------------------------------------------------+
void CGaugeBase::DrawCaseElements(Struct_Case &externalCase, Struct_Case &internalCase) {
   if(externalCase.display)                //--- Check external display
      scaleLayer.obj_Canvas.FillCircle(externalCase.circle.centerX, externalCase.circle.centerY, externalCase.circle.radius, ColorToARGB(externalCase.circle.clr, scaleLayer.transparency)); //--- Fill external circle
   if(internalCase.display)                //--- Check internal display
      scaleLayer.obj_Canvas.FillCircle(internalCase.circle.centerX, internalCase.circle.centerY, internalCase.circle.radius, ColorToARGB(internalCase.circle.clr, scaleLayer.transparency)); //--- Fill internal circle
}

//+------------------------------------------------------------------+
//| Calculate Needle                                                 |
//+------------------------------------------------------------------+
void CGaugeBase::CalculateNeedle() {
   int innerRadius = 0, outerRadius = 0;   //--- Initialize radii
   if(inputParams.minorTicksPerInterval > 0) //--- Check minor ticks
      CalculateInnerOuterRadii(innerRadius, outerRadius, scaleLayer.scaleArc.radius, scaleLayer.scaleMarks.minorTickLength, inputParams.tickStyle); //--- Calculate for minor
   else if(inputParams.mediumTicksPerMajor > 0) //--- Check medium ticks
      CalculateInnerOuterRadii(innerRadius, outerRadius, scaleLayer.scaleArc.radius, scaleLayer.scaleMarks.mediumTickLength, inputParams.tickStyle); //--- Calculate for medium
   else if(inputParams.majorTickInterval > 0) //--- Check major ticks
      CalculateInnerOuterRadii(innerRadius, outerRadius, scaleLayer.scaleArc.radius, scaleLayer.scaleMarks.majorTickLength, inputParams.tickStyle); //--- Calculate for major
   needleLayer.needle.tipRadius = outerRadius; //--- Set tip radius
   needleLayer.needle.clr = inputParams.needleColor; //--- Set needle color
   needleLayer.needle.fillStyle = inputParams.needleFillStyle; //--- Set fill style
   needleLayer.needle.tailRadius = needleLayer.needleCenter.radius * 2; //--- Set tail radius
}

//+------------------------------------------------------------------+
//| Redraw Needle                                                    |
//+------------------------------------------------------------------+
void CGaugeBase::RedrawNeedle(double value) {
   needleLayer.obj_Canvas.Erase();         //--- Erase canvas
   double normalizedValue = 0;             //--- Initialize normalized value
   if(scaleLayer.scaleMarks.minValue < scaleLayer.scaleMarks.maxValue) { //--- Check direct order
      if(value < scaleLayer.scaleMarks.minValue) //--- Check min value
         value = scaleLayer.scaleMarks.minValue; //--- Clamp to min
      if(value > scaleLayer.scaleMarks.maxValue) //--- Check max value
         value = scaleLayer.scaleMarks.maxValue; //--- Clamp to max
      normalizedValue = value - scaleLayer.scaleMarks.minValue; //--- Normalize
   } else {                                //--- Handle inverse order
      if(value > scaleLayer.scaleMarks.minValue) //--- Check min value
         value = scaleLayer.scaleMarks.minValue; //--- Clamp to min
      if(value < scaleLayer.scaleMarks.maxValue) //--- Check max value
         value = scaleLayer.scaleMarks.maxValue; //--- Clamp to max
      normalizedValue = scaleLayer.scaleMarks.minValue - value; //--- Normalize
   }
   if(scaleLayer.scaleMarks.valueRange == 0) //--- Check value range
      return;                              //--- Return if zero
   double currentAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - ((normalizedValue * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange)); //--- Calculate current angle
   needleLayer.needle.x[0] = (int)(scaleLayer.scaleArc.centerX - needleLayer.needle.tipRadius * MathCos(M_PI - currentAngle)); //--- Set x0
   needleLayer.needle.y[0] = (int)(scaleLayer.scaleArc.centerY - needleLayer.needle.tipRadius * MathSin(M_PI - currentAngle)); //--- Set y0
   double bufferX[3], bufferY[3];          //--- Declare buffers
   bufferX[0] = scaleLayer.scaleArc.centerX - needleLayer.needle.tipRadius * MathCos(M_PI - currentAngle); //--- Set bufferX0
   bufferY[0] = scaleLayer.scaleArc.centerY - needleLayer.needle.tipRadius * MathSin(M_PI - currentAngle); //--- Set bufferY0
   double tailX = scaleLayer.scaleArc.centerX - needleLayer.needle.tailRadius * MathCos(2 * M_PI - currentAngle); //--- Calculate tail X
   double tailY = scaleLayer.scaleArc.centerY - needleLayer.needle.tailRadius * MathSin(2 * M_PI - currentAngle); //--- Calculate tail Y
   int r = (int)(needleLayer.needle.tailRadius / 3.0); //--- Calculate r
   bufferX[1] = tailX - r * MathCos(0.5 * M_PI - currentAngle); //--- Set bufferX1
   bufferY[1] = tailY - r * MathSin(0.5 * M_PI - currentAngle); //--- Set bufferY1
   bufferX[2] = tailX - r * MathCos(1.5 * M_PI - currentAngle); //--- Set bufferX2
   bufferY[2] = tailY - r * MathSin(1.5 * M_PI - currentAngle); //--- Set bufferY2
   uint clr = ColorToARGB(needleLayer.needle.clr, needleLayer.transparency); //--- Get color
   needleLayer.obj_Canvas.LineAA((int)bufferX[0], (int)bufferY[0], (int)bufferX[1], (int)bufferY[1], clr); //--- Draw line AA 0-1
   needleLayer.obj_Canvas.LineAA((int)bufferX[1], (int)bufferY[1], (int)bufferX[2], (int)bufferY[2], clr); //--- Draw line AA 1-2
   needleLayer.obj_Canvas.LineAA((int)bufferX[2], (int)bufferY[2], (int)bufferX[0], (int)bufferY[0], clr); //--- Draw line AA 2-0
   double centroidX = (bufferX[0] + bufferX[1] + bufferX[2]) / 3.0; //--- Calculate centroid X
   double centroidY = (bufferY[0] + bufferY[1] + bufferY[2]) / 3.0; //--- Calculate centroid Y
   needleLayer.obj_Canvas.Fill((int)centroidX, (int)centroidY, clr); //--- Fill
   needleLayer.obj_Canvas.LineAA(scaleLayer.scaleArc.centerX, scaleLayer.scaleArc.centerY, (int)bufferX[0], (int)bufferY[0], clr); //--- Draw line AA center to 0
   if(needleLayer.needleCenter.display)    //--- Check display
      needleLayer.obj_Canvas.FillCircle(needleLayer.needleCenter.centerX, needleLayer.needleCenter.centerY, needleLayer.needleCenter.radius, ColorToARGB(needleLayer.needleCenter.clr, needleLayer.transparency)); //--- Fill needle center
}

//+------------------------------------------------------------------+
//| Calculate and Draw Legends                                       |
//+------------------------------------------------------------------+
void CGaugeBase::CalculateAndDrawLegends() {
   if(inputParams.description.enable)      //--- Check description enable
      CalculateAndDrawLegendString(scaleLayer.gaugeLabel.description); //--- Calculate and draw description
   if(inputParams.units.enable)            //--- Check units enable
      CalculateAndDrawLegendString(scaleLayer.gaugeLabel.units); //--- Calculate and draw units
   if(inputParams.multiplier.enable) {     //--- Check multiplier enable
      scaleLayer.gaugeLabel.multiplier.text = scaleMultiplierStrings[inputParams.scaleMultiplier]; //--- Set multiplier text
      CalculateAndDrawLegendString(scaleLayer.gaugeLabel.multiplier); //--- Calculate and draw multiplier
   }
   if(inputParams.value.enable) {          //--- Check value enable
      scaleLayer.gaugeLabel.value.decimalPlaces = 0; //--- Set decimal places
      if(inputParams.value.text != "" ) {  //--- Check text
         int digits = (int)StringToInteger(inputParams.value.text); //--- Get digits
         if(digits >= 1 && digits <= 8)    //--- Check digits range
            scaleLayer.gaugeLabel.value.decimalPlaces = (uint)digits; //--- Set decimal places
      }
      scaleLayer.gaugeLabel.value.text = " "; //--- Set text
      CalculateAndDrawLegendString(scaleLayer.gaugeLabel.value); //--- Calculate and draw value
   }
}

//+------------------------------------------------------------------+
//| Calculate and Draw Legend String                                 |
//+------------------------------------------------------------------+
void CGaugeBase::CalculateAndDrawLegendString(Struct_GaugeLegendString &legendString) {
   if(legendString.text != "") {           //--- Check text
      legendString.draw = true;            //--- Set draw flag
      scaleLayer.obj_Canvas.FontSet(legendString.fontName, legendString.fontSize, legendString.fontFlags, 0); //--- Set font
      double normalizedAngle = NormalizeRadians(DegreesToRadians(legendString.angle + 90)); //--- Normalize angle
      legendString.x = (uint)(scaleLayer.scaleArc.centerX - legendString.radius * MathCos(M_PI - normalizedAngle)); //--- Set x
      legendString.y = (uint)(scaleLayer.scaleArc.centerY - legendString.radius * MathSin(M_PI - normalizedAngle)); //--- Set y
      legendString.backgroundColor = scaleLayer.caseColor; //--- Set background color
      scaleLayer.obj_Canvas.TextOut(legendString.x, legendString.y, legendString.text, ColorToARGB(legendString.textColor, scaleLayer.transparency), TA_CENTER | TA_VCENTER); //--- Draw text
   }
}
```

We define the "CalculateCaseElements" method in the "CGaugeBase" class to prepare the external and internal casing for the gauge. If the border size is positive, we configure the external case's circle with the scale arc's center coordinates, the full radius, border color, and set its display flag to true; otherwise, we disable its display. For the internal case, we always set its center to match the scale arc, reduce the radius by the border size, apply the case color, and enable display. The "DrawCaseElements" method renders these cases on the scale layer canvas. If the external case is displayed, we fill its circle using "FillCircle" with the ARGB-converted color adjusted for transparency via the [ColorToARGB](https://www.mql5.com/en/docs/convert/colortoargb) function. Similarly, for the internal case, we fill its circle with the appropriate ARGB color if enabled.

We define the "CalculateNeedle" method to determine the needle's dimensions based on tick configurations. We initialize inner and outer radii, then conditionally call "CalculateInnerOuterRadii" using the minor, medium, or major tick length depending on which intervals are set, passing the scale arc radius and tick style. We assign the outer radius to the needle's tip, set its color and fill style from input parameters, and calculate the tail radius as twice the needle center's radius.

In the "RedrawNeedle" method, we first erase the needle layer canvas with "Erase". We normalize the input value by clamping it to the scale's min and max, then compute a normalized value based on whether the scale is ascending or descending. If the value range is zero, we exit early. We calculate the current angle using "NormalizeRadians", adjusting for the normalized value's proportion of the angle range. We set the needle's tip coordinates using cosine and sine functions with pi adjustments, prepare buffer arrays for the tail triangle by computing tail positions and offset points with a radius third of the tail, convert the color to ARGB, draw anti-aliased lines for the triangle edges with "LineAA", find the centroid for filling the triangle via "Fill", draw a line from center to tip, and if the center is displayed, fill it as a circle with ARGB color.

We create the "CalculateAndDrawLegends" method to handle various legend elements if enabled. For the description and units, we directly call "CalculateAndDrawLegendString". For the multiplier, we set its text from a predefined array based on the scale multiplier index before drawing. For the value, we default decimal places to 0, parse any input text for digits between 1 and 8 to override decimals, set initial text to a space, and draw it. The "CalculateAndDrawLegendString" method processes individual legends if text is present, setting the draw flag, configuring the font with "FontSet", normalizing the angle by adding 90 degrees and converting via "DegreesToRadians" and "NormalizeRadians", computing x and y positions using cosine and sine, assigning the case color as background, and rendering the text centered with "TextOut" using ARGB color and alignment flags. For the scale marks and ranges, we use the following logic.

```
//+------------------------------------------------------------------+
//| Redraw Scale Marks                                               |
//+------------------------------------------------------------------+
void CGaugeBase::RedrawScaleMarks(Struct_Case &internalCase, Struct_Arc &scaleArc, int borderGap) {
   int majorIndex, mediumIndex, minorIndex;           //--- Declare indices
   double angle = 0, mediumAngle = 0, minorAngle = 0; //--- Declare angles
   scaleLayer.scaleMarks.multiplier = scaleMultipliers[inputParams.scaleMultiplier]; //--- Set multiplier
   if(scaleLayer.scaleMarks.multiplier <= 0)          //--- Check multiplier
      scaleLayer.scaleMarks.multiplier = 1.0;         //--- Set default multiplier
   scaleLayer.scaleMarks.minValue = inputParams.minScaleValue; //--- Set min value
   scaleLayer.scaleMarks.maxValue = inputParams.maxScaleValue; //--- Set max value
   scaleLayer.scaleMarks.decimalPlaces = 0;           //--- Set decimal places
   if(scaleLayer.scaleMarks.maxValue > scaleLayer.scaleMarks.minValue) { //--- Check direct order
      scaleLayer.scaleMarks.forwardDirection = true;  //--- Set forward direction
      scaleLayer.scaleMarks.valueRange = scaleLayer.scaleMarks.maxValue - scaleLayer.scaleMarks.minValue; //--- Set value range
   } else {                                           //--- Handle inverse order
      scaleLayer.scaleMarks.forwardDirection = false; //--- Set forward direction false
      scaleLayer.scaleMarks.valueRange = scaleLayer.scaleMarks.minValue - scaleLayer.scaleMarks.maxValue; //--- Set value range
   }
   scaleLayer.scaleMarks.nullMarkPosition = 1;         //--- Set null mark position
   scaleLayer.scaleMarks.minAngle = scaleArc.endAngle; //--- Set min angle
   scaleLayer.scaleMarks.maxAngle = scaleArc.startAngle; //--- Set max angle
   if(scaleArc.endAngle > scaleArc.startAngle)         //--- Check angles
      scaleLayer.scaleMarks.angleRange = NormalizeRadians(scaleArc.endAngle - scaleArc.startAngle); //--- Set angle range
   else                                                //--- Handle wrap around
      scaleLayer.scaleMarks.angleRange = NormalizeRadians(scaleArc.endAngle + (2 * M_PI - scaleArc.startAngle)); //--- Set angle range
   int leftMarkCount = 0;                              //--- Initialize left mark count
   int rightMarkCount = 0;                             //--- Initialize right mark count
   double markBuffer[361][2];                          //--- Declare mark buffer
   int bufferCenterIndex = (int)(361 / 2);             //--- Set buffer center index
   double tempValue = 0;                               //--- Initialize temp value
   int sign = 0;                                       //--- Initialize sign
   double multiplier = scaleMultipliers[inputParams.scaleMultiplier]; //--- Set multiplier
   markBuffer[bufferCenterIndex][0] = 0;               //--- Set zero value
   markBuffer[bufferCenterIndex][1] = scaleLayer.scaleMarks.minAngle; //--- Set zero angle
   tempValue = 0;                                      //--- Reset temp value
   sign = scaleLayer.scaleMarks.forwardDirection ? 1 : -1; //--- Set sign
   for(majorIndex = 1; majorIndex < (int)(361 / 2); majorIndex++) { //--- Loop major indices
      tempValue = majorIndex * inputParams.majorTickInterval; //--- Calculate temp value
      if(tempValue <= scaleLayer.scaleMarks.valueRange) { //--- Check range
         markBuffer[bufferCenterIndex + majorIndex][0] = (majorIndex * inputParams.majorTickInterval * sign) / multiplier; //--- Set mark value
         markBuffer[bufferCenterIndex + majorIndex][1] = NormalizeRadians(scaleLayer.scaleMarks.minAngle - ((majorIndex * inputParams.majorTickInterval * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange)); //--- Set mark angle
         rightMarkCount++;                             //--- Increment right count
      } else                                           //--- Handle out of range
         break;                                        //--- Break loop
   }
   double majorAngleStep, mediumAngleStep, minorAngleStep; //--- Declare angle steps
   majorAngleStep = (inputParams.majorTickInterval * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange; //--- Set major step
   mediumAngleStep = 0;                                //--- Initialize medium step
   if(inputParams.mediumTicksPerMajor != 0)            //--- Check medium ticks
      mediumAngleStep = ((inputParams.majorTickInterval / (inputParams.mediumTicksPerMajor + 1)) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange; //--- Set medium step
   minorAngleStep = 0;                                 //--- Initialize minor step
   if(inputParams.minorTicksPerInterval != 0) {        //--- Check minor ticks
      if(mediumAngleStep != 0)                         //--- Check medium step
         minorAngleStep = (((inputParams.majorTickInterval / (inputParams.mediumTicksPerMajor + 1)) / (inputParams.minorTicksPerInterval + 1)) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange; //--- Set minor step with medium
      else                                             //--- Handle no medium
         minorAngleStep = ((inputParams.majorTickInterval / (inputParams.minorTicksPerInterval + 1)) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange; //--- Set minor step without medium
   }
   CalculateRanges(borderGap);             //--- Calculate ranges
   DrawRanges();                           //--- Draw ranges
   int innerR, outerR;                     //--- Declare radii
   double startX, startY, endX, endY;      //--- Declare coordinates
   int textX, textY;                       //--- Declare text coordinates
   string markText;                        //--- Declare mark text
   int digits;                             //--- Declare digits
   scaleLayer.obj_Canvas.FontSet(scaleLayer.scaleMarks.tickFontName, scaleLayer.scaleMarks.tickFontSize, scaleLayer.scaleMarks.tickFontFlags, 0); //--- Set font
   rightMarkCount++;                       //--- Increment right count
   for(majorIndex = 0; majorIndex < rightMarkCount; majorIndex++) { //--- Loop right marks
      angle = markBuffer[bufferCenterIndex + majorIndex][1];        //--- Get angle
      CalculateInnerOuterRadii(innerR, outerR, (int)scaleArc.radius, scaleLayer.scaleMarks.majorTickLength, inputParams.tickStyle); //--- Calculate radii
      startX = scaleArc.centerX - innerR * MathCos(M_PI - angle);   //--- Set start X
      startY = scaleArc.centerY - innerR * MathSin(M_PI - angle);   //--- Set start Y
      endX = scaleArc.centerX - outerR * MathCos(M_PI - angle);     //--- Set end X
      endY = scaleArc.centerY - outerR * MathSin(M_PI - angle);     //--- Set end Y
      textX = (int)(scaleArc.centerX - (outerR - scaleLayer.scaleMarks.tickFontGap) * MathCos(M_PI - angle)); //--- Set text X
      textY = (int)(scaleArc.centerY - (outerR - scaleLayer.scaleMarks.tickFontGap) * MathSin(M_PI - angle)); //--- Set text Y
      scaleLayer.obj_Canvas.LineAA((int)startX, (int)startY, (int)endX, (int)endY, ColorToARGB(scaleArc.clr, scaleLayer.transparency)); //--- Draw line AA
      digits = (markBuffer[bufferCenterIndex + majorIndex][0] == 0) ? 0 : scaleLayer.scaleMarks.decimalPlaces; //--- Set digits
      markText = DoubleToString(markBuffer[bufferCenterIndex + majorIndex][0], digits); //--- Get mark text
      scaleLayer.obj_Canvas.TextOut(textX, textY, markText, ColorToARGB(inputParams.tickFontColor, scaleLayer.transparency), TA_CENTER | TA_VCENTER); //--- Draw text
      if(mediumAngleStep != 0) {           //--- Check medium step
         mediumAngle = angle;              //--- Set medium angle
         for(minorIndex = 1; minorIndex <= inputParams.minorTicksPerInterval; minorIndex++) { //--- Loop minor
            minorAngle = NormalizeRadians(mediumAngle - minorAngleStep * minorIndex);         //--- Calculate minor angle
            if(!DrawTick(minorAngle, scaleLayer.scaleMarks.minorTickLength, scaleArc))        //--- Draw minor tick
               break;                      //--- Break if failed
         }
         for(mediumIndex = 1; mediumIndex <= inputParams.mediumTicksPerMajor; mediumIndex++) { //--- Loop medium
            mediumAngle = NormalizeRadians(angle - mediumAngleStep * mediumIndex);            //--- Calculate medium angle
            if(!DrawTick(mediumAngle, scaleLayer.scaleMarks.mediumTickLength, scaleArc))      //--- Draw medium tick
               break;                      //--- Break if failed
            for(minorIndex = 1; minorIndex <= inputParams.minorTicksPerInterval; minorIndex++) { //--- Loop minor
               minorAngle = NormalizeRadians(mediumAngle - minorAngleStep * minorIndex);      //--- Calculate minor angle
               if(!DrawTick(minorAngle, scaleLayer.scaleMarks.minorTickLength, scaleArc))     //--- Draw minor tick
                  break;                   //--- Break if failed
            }
         }
      } else {                             //--- Handle no medium
         for(minorIndex = 1; minorIndex <= inputParams.minorTicksPerInterval; minorIndex++) { //--- Loop minor
            minorAngle = NormalizeRadians(angle - minorAngleStep * minorIndex); //--- Calculate minor angle
            if(!DrawTick(minorAngle, scaleLayer.scaleMarks.minorTickLength, scaleArc)) //--- Draw minor tick
               break;                      //--- Break if failed
         }
      }
   }
   for(majorIndex = 0; majorIndex < (leftMarkCount + 1); majorIndex++) { //--- Loop left marks
      angle = markBuffer[bufferCenterIndex - majorIndex][1];      //--- Get angle
      CalculateInnerOuterRadii(innerR, outerR, (int)scaleArc.radius, scaleLayer.scaleMarks.majorTickLength, inputParams.tickStyle); //--- Calculate radii
      startX = scaleArc.centerX - innerR * MathCos(M_PI - angle); //--- Set start X
      startY = scaleArc.centerY - innerR * MathSin(M_PI - angle); //--- Set start Y
      endX = scaleArc.centerX - outerR * MathCos(M_PI - angle);   //--- Set end X
      endY = scaleArc.centerY - outerR * MathSin(M_PI - angle);   //--- Set end Y
      textX = (int)(scaleArc.centerX - (outerR - scaleLayer.scaleMarks.tickFontGap) * MathCos(M_PI - angle));  //--- Set text X
      textY = (int)(scaleArc.centerY - (outerR - scaleLayer.scaleMarks.tickFontGap) * MathSin(M_PI - angle));  //--- Set text Y
      digits = (markBuffer[bufferCenterIndex - majorIndex][0] == 0) ? 0 : scaleLayer.scaleMarks.decimalPlaces; //--- Set digits
      markText = DoubleToString(markBuffer[bufferCenterIndex - majorIndex][0], digits);        //--- Get mark text
      if(majorIndex > 0 || (majorIndex == 0 && scaleLayer.scaleMarks.nullMarkPosition == 3)) { //--- Check condition
         scaleLayer.obj_Canvas.LineAA((int)startX, (int)startY, (int)endX, (int)endY, ColorToARGB(scaleArc.clr, scaleLayer.transparency)); //--- Draw line AA
         scaleLayer.obj_Canvas.TextOut(textX, textY, markText, ColorToARGB(inputParams.tickFontColor, scaleLayer.transparency), TA_CENTER | TA_VCENTER); //--- Draw text
      }
      if(mediumAngleStep != 0) {           //--- Check medium step
         mediumAngle = angle;              //--- Set medium angle
         for(minorIndex = 1; minorIndex <= inputParams.minorTicksPerInterval; minorIndex++) { //--- Loop minor
            minorAngle = NormalizeRadians(mediumAngle + minorAngleStep * minorIndex);         //--- Calculate minor angle
            if(!DrawTick(minorAngle, scaleLayer.scaleMarks.minorTickLength, scaleArc))        //--- Draw minor tick
               break;                      //--- Break if failed
         }
         for(mediumIndex = 1; mediumIndex <= inputParams.mediumTicksPerMajor; mediumIndex++) { //--- Loop medium
            mediumAngle = NormalizeRadians(angle + mediumAngleStep * mediumIndex);             //--- Calculate medium angle
            if(!DrawTick(mediumAngle, scaleLayer.scaleMarks.mediumTickLength, scaleArc))       //--- Draw medium tick
               break;                      //--- Break if failed
            for(minorIndex = 1; minorIndex <= inputParams.minorTicksPerInterval; minorIndex++) { //--- Loop minor
               minorAngle = NormalizeRadians(mediumAngle + minorAngleStep * minorIndex);       //--- Calculate minor angle
               if(!DrawTick(minorAngle, scaleLayer.scaleMarks.minorTickLength, scaleArc))      //--- Draw minor tick
                  break;                   //--- Break if failed
            }
         }
      } else {                             //--- Handle no medium
         for(minorIndex = 1; minorIndex <= inputParams.minorTicksPerInterval; minorIndex++) {  //--- Loop minor
            minorAngle = NormalizeRadians(angle + minorAngleStep * minorIndex);                //--- Calculate minor angle
            if(!DrawTick(minorAngle, scaleLayer.scaleMarks.minorTickLength, scaleArc))         //--- Draw minor tick
               break;                      //--- Break if failed
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate Ranges                                                 |
//+------------------------------------------------------------------+
void CGaugeBase::CalculateRanges(int borderGap) {
   int innerR, outerR;                     //--- Declare radii
   CalculateInnerOuterRadii(innerR, outerR, scaleLayer.scaleArc.radius, scaleLayer.scaleMarks.majorTickLength, inputParams.tickStyle); //--- Calculate radii
   for(int rangeIndex = 0; rangeIndex < 4; rangeIndex++) { //--- Loop ranges
      if(IsValidRange(rangeIndex))         //--- Check valid range
         CalculateRangePie(scaleLayer.ranges[rangeIndex], innerR, borderGap, outerR, inputParams.ranges[rangeIndex].start, inputParams.ranges[rangeIndex].end, inputParams.ranges[rangeIndex].clr, scaleLayer.caseColor); //--- Calculate pie
   }
}

//+------------------------------------------------------------------+
//| Check Valid Range                                                |
//+------------------------------------------------------------------+
bool CGaugeBase::IsValidRange(int index) {
   if(!inputParams.ranges[index].enable)   //--- Check enable
      return false;                        //--- Return false
   if(inputParams.ranges[index].start == inputParams.ranges[index].end) //--- Check start end
      return false;                        //--- Return false
   double paramMin, paramMax, rangeMin, rangeMax; //--- Declare mins maxs
   NormalizeRangeValues(paramMin, paramMax, inputParams.minScaleValue, inputParams.maxScaleValue); //--- Normalize param
   NormalizeRangeValues(rangeMin, rangeMax, inputParams.ranges[index].start, inputParams.ranges[index].end); //--- Normalize range
   if(rangeMin < paramMin && rangeMax < paramMin) //--- Check below param
      return false;                        //--- Return false
   if(rangeMin > paramMax && rangeMax > paramMax) //--- Check above param
      return false;                        //--- Return false
   if(rangeMin < paramMin)                 //--- Check min
      rangeMin = paramMin;                 //--- Clamp min
   if(rangeMax > paramMax)                 //--- Check max
      rangeMax = paramMax;                 //--- Clamp max
   inputParams.ranges[index].start = rangeMin; //--- Set start
   inputParams.ranges[index].end = rangeMax;   //--- Set end
   return true;                                //--- Return true
}

//+------------------------------------------------------------------+
//| Normalize Range Values                                           |
//+------------------------------------------------------------------+
void CGaugeBase::NormalizeRangeValues(double &minValue, double &maxValue, double val0, double val1) {
   if(val0 < val1) {                       //--- Check val0 < val1
      minValue = val0;                     //--- Set min
      maxValue = val1;                     //--- Set max
   } else {                                //--- Handle val0 >= val1
      minValue = val1;                     //--- Set min
      maxValue = val0;                     //--- Set max
   }
}

//+------------------------------------------------------------------+
//| Calculate Range Pie                                              |
//+------------------------------------------------------------------+
void CGaugeBase::CalculateRangePie(Struct_Range &range, int innerRadius, int radialGap, int outerRadius, double rangeStart, double rangeEnd, color rangeClr, color caseClr) {
   range.startValue = rangeStart;          //--- Set start value
   range.endValue = rangeEnd;              //--- Set end value
   double rangeStartNorm, rangeEndNorm;    //--- Declare norms
   if(range.startValue > range.endValue) { //--- Check start > end
      rangeStartNorm = range.startValue;   //--- Set start norm
      rangeEndNorm = range.endValue;       //--- Set end norm
   } else if(range.startValue < range.endValue) { //--- Check start < end
      rangeEndNorm = range.startValue;     //--- Set end norm
      rangeStartNorm = range.endValue;     //--- Set start norm
   } else                                  //--- Handle equal
      return;                              //--- Return
   if(scaleLayer.scaleMarks.minValue > scaleLayer.scaleMarks.maxValue) { //--- Check inverse
      double temp = rangeStartNorm;        //--- Temp start
      rangeStartNorm = -rangeEndNorm;      //--- Set start norm
      rangeEndNorm = -temp;                //--- Set end norm
   }
   range.active = true;                    //--- Set active
   range.clr = rangeClr;                   //--- Set color
   range.pie.centerX = scaleLayer.scaleArc.centerX; //--- Set center X
   range.pie.centerY = scaleLayer.scaleArc.centerY; //--- Set center Y
   range.pie.radius = innerRadius;         //--- Set radius
   range.pie.eraseRadius = outerRadius;    //--- Set erase radius
   double angularOffset = MathArcsin(((double)radialGap / (double)range.pie.radius) / 2.0); //--- Calculate offset
   if(scaleLayer.scaleMarks.minValue < scaleLayer.scaleMarks.maxValue) { //--- Check direct
      range.pie.startAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - (((rangeStartNorm - scaleLayer.scaleMarks.minValue) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange)); //--- Set start angle
      range.pie.endAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - (((rangeEndNorm - scaleLayer.scaleMarks.minValue) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange)); //--- Set end angle
      range.pie.eraseStartAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - (((rangeStartNorm - scaleLayer.scaleMarks.minValue) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange) - angularOffset); //--- Set erase start
      range.pie.eraseEndAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - (((rangeEndNorm - scaleLayer.scaleMarks.minValue) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange) + angularOffset); //--- Set erase end
   } else {                                //--- Handle inverse
      range.pie.startAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - (((scaleLayer.scaleMarks.minValue + rangeStartNorm) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange)); //--- Set start angle
      range.pie.endAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - (((scaleLayer.scaleMarks.minValue + rangeEndNorm) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange)); //--- Set end angle
      range.pie.eraseStartAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - (((rangeStartNorm + scaleLayer.scaleMarks.minValue) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange) - angularOffset); //--- Set erase start
      range.pie.eraseEndAngle = NormalizeRadians(scaleLayer.scaleMarks.minAngle - (((rangeEndNorm + scaleLayer.scaleMarks.minValue) * scaleLayer.scaleMarks.angleRange) / scaleLayer.scaleMarks.valueRange) + angularOffset); //--- Set erase end
   }
   range.pie.clr = rangeClr;               //--- Set pie color
   range.pie.eraseClr = caseClr;           //--- Set erase color
}

//+------------------------------------------------------------------+
//| Draw Ranges                                                      |
//+------------------------------------------------------------------+
void CGaugeBase::DrawRanges() {
   for(int i = 0; i < 4; i++)              //--- Loop indices
      DrawRange(scaleLayer.ranges[i]);     //--- Draw range
}

//+------------------------------------------------------------------+
//| Draw Range                                                       |
//+------------------------------------------------------------------+
void CGaugeBase::DrawRange(Struct_Range &range) {
   if(!range.active)                       //--- Check active
      return;                              //--- Return
   int r_min = MathMin(range.pie.radius, range.pie.eraseRadius); //--- Get min r
   int r_max = MathMax(range.pie.radius, range.pie.eraseRadius); //--- Get max r
   for(int r = r_min + 1; r <= r_max; r++) { //--- Loop radii
      double frac = (double)(r - r_min) / (r_max - r_min); //--- Calculate frac
      uchar alpha = (uchar)(scaleLayer.transparency * frac); //--- Calculate alpha
      uint col = ColorToARGB(range.pie.clr, alpha); //--- Get color
      scaleLayer.obj_Canvas.Arc(range.pie.centerX, range.pie.centerY, r, r, range.pie.startAngle, range.pie.endAngle, col); //--- Draw arc
      uint erase_col = ColorToARGB(range.pie.eraseClr, scaleLayer.transparency); //--- Get erase color
      scaleLayer.obj_Canvas.Arc(range.pie.centerX, range.pie.centerY, r, r, range.pie.eraseStartAngle, range.pie.startAngle, erase_col); //--- Draw left erase
      scaleLayer.obj_Canvas.Arc(range.pie.centerX, range.pie.centerY, r, r, range.pie.endAngle, range.pie.eraseEndAngle, erase_col); //--- Draw right erase
   }
}

//+------------------------------------------------------------------+
//| Calculate Inner Outer Radii                                      |
//+------------------------------------------------------------------+
void CGaugeBase::CalculateInnerOuterRadii(int &innerRadius, int &outerRadius, int baseRadius, int tickLength, int tickStyle) {
   innerRadius = baseRadius;               //--- Set inner radius
   outerRadius = baseRadius - tickLength;  //--- Set outer radius
}

//+------------------------------------------------------------------+
//| Draw Tick                                                        |
//+------------------------------------------------------------------+
bool CGaugeBase::DrawTick(double angle, int length, Struct_Arc &scaleArc) {
   int innerR, outerR;                     //--- Declare radii
   double startX, startY, endX, endY;      //--- Declare coordinates
   double arcStartAngle = scaleArc.startAngle; //--- Get start angle
   double arcEndAngle = scaleArc.endAngle; //--- Get end angle
   double deltaToStart = CalculateAngleDelta(arcStartAngle, angle, -1); //--- Calculate delta to start
   double deltaToEnd = CalculateAngleDelta(arcEndAngle, angle, 1); //--- Calculate delta to end
   double totalArcDelta = CalculateAngleDelta(arcStartAngle, arcEndAngle, -1); //--- Calculate total delta
   if(MathAbs(totalArcDelta - (deltaToEnd + deltaToStart)) < (M_PI / 180.0)) //--- Check within arc
      return false;                        //--- Return false
   CalculateInnerOuterRadii(innerR, outerR, scaleArc.radius, length, inputParams.tickStyle); //--- Calculate radii
   startX = scaleArc.centerX - innerR * MathCos(M_PI - angle); //--- Set start X
   startY = scaleArc.centerY - innerR * MathSin(M_PI - angle); //--- Set start Y
   endX = scaleArc.centerX - outerR * MathCos(M_PI - angle); //--- Set end X
   endY = scaleArc.centerY - outerR * MathSin(M_PI - angle); //--- Set end Y
   scaleLayer.obj_Canvas.LineAA((int)startX, (int)startY, (int)endX, (int)endY, ColorToARGB(scaleArc.clr, scaleLayer.transparency)); //--- Draw line AA
   return true;                            //--- Return true
}

//+------------------------------------------------------------------+
//| Calculate Angle Delta                                            |
//+------------------------------------------------------------------+
double CGaugeBase::CalculateAngleDelta(double angle1, double angle2, int direction) {
   double normAngle1 = NormalizeRadians(angle1); //--- Normalize angle1
   double normAngle2 = NormalizeRadians(angle2); //--- Normalize angle2
   double delta1, delta2;                  //--- Declare deltas
   if(normAngle1 == normAngle2)            //--- Check equal
      return 0;                            //--- Return 0
   if(normAngle1 > normAngle2) {           //--- Check angle1 > angle2
      delta1 = normAngle1 - normAngle2;    //--- Set delta1
      delta2 = normAngle2 + (2 * M_PI - normAngle1); //--- Set delta2
   } else {                                //--- Handle angle1 <= angle2
      delta1 = normAngle1 + (2 * M_PI - normAngle2); //--- Set delta1
      delta2 = normAngle2 - normAngle1;    //--- Set delta2
   }
   if(direction < 0)                       //--- Check direction
      return delta1;                       //--- Return delta1
   return delta2;                          //--- Return delta2
}

//+------------------------------------------------------------------+
//| Set Legend String Params                                         |
//+------------------------------------------------------------------+
void CGaugeBase::SetLegendStringParams(Struct_GaugeLegendString &legendString, Struct_GaugeLegendParams &param, int minRadius, int radiusDelta) {
   if(param.enable && param.fontName != "") { //--- Check enable and font
      legendString.text = param.text;      //--- Set text
      legendString.angle = param.angle;    //--- Set angle
      legendString.radius = minRadius + (int)((radiusDelta * param.radius * 10) / 100.0); //--- Set radius
      legendString.fontName = param.fontName; //--- Set font name
      legendString.fontFlags = 0;          //--- Initialize flags
      if(param.italic)                     //--- Check italic
         legendString.fontFlags |= FONT_ITALIC; //--- Add italic
      if(param.bold)                       //--- Check bold
         legendString.fontFlags |= FW_BOLD; //--- Add bold
      legendString.fontSize = (int)(((param.fontSize + 2) * radiusDelta) / 64); //--- Set font size
      legendString.textColor = param.textColor; //--- Set text color
   }
}

//+------------------------------------------------------------------+
//| Delete Gauge                                                     |
//+------------------------------------------------------------------+
void CGaugeBase::Delete() {
   ObjectDelete(0, scaleLayer.objectName); //--- Delete scale object
   ObjectDelete(0, needleLayer.objectName);//--- Delete needle object
}

//+------------------------------------------------------------------+
//| Set New Value                                                    |
//+------------------------------------------------------------------+
void CGaugeBase::NewValue(double value) {
   if(!initializationComplete)             //--- Check initialization
      return;                              //--- Return
   currentValue = value;                   //--- Set current value
   if(scaleLayer.gaugeLabel.value.draw)    //--- Check draw
      RedrawValueDisplay(currentValue);    //--- Redraw value
   RedrawNeedle(currentValue);             //--- Redraw needle
   needleLayer.obj_Canvas.Update(true);    //--- Update canvas
}

//+------------------------------------------------------------------+
//| Get Label Area Size                                              |
//+------------------------------------------------------------------+
bool CGaugeBase::GetLabelAreaSize(Struct_LabelAreaSize &areaSize, Struct_GaugeLegendString &legendString) {
   if(!scaleLayer.obj_Canvas.FontSet(legendString.fontName, legendString.fontSize, legendString.fontFlags, 0)) //--- Set font
      return false;                        //--- Return false
   scaleLayer.obj_Canvas.TextSize(legendString.text, areaSize.width, areaSize.height); //--- Get text size
   if(areaSize.width == 0 || areaSize.height == 0) //--- Check size
      return false;                        //--- Return false
   areaSize.diagonal = (int)MathCeil(MathSqrt((double)(areaSize.width * areaSize.width + areaSize.height * areaSize.height))); //--- Calculate diagonal
   return true;                            //--- Return true
}

//+------------------------------------------------------------------+
//| Erase Legend String                                              |
//+------------------------------------------------------------------+
bool CGaugeBase::EraseLegendString(Struct_GaugeLegendString &legendString, color eraseClr) {
   Struct_LabelAreaSize areaSize;          //--- Declare area size
   if(!GetLabelAreaSize(areaSize, legendString)) //--- Get area size
      return false;                        //--- Return false
   scaleLayer.obj_Canvas.FillRectangle((int)legendString.x - (areaSize.width / 2) - 4, (int)legendString.y - (areaSize.height / 2), (int)legendString.x + (areaSize.width / 2) + 4, (int)legendString.y + (areaSize.height / 2), ColorToARGB(eraseClr, scaleLayer.transparency)); //--- Fill rectangle
   return true;                            //--- Return true
}

//+------------------------------------------------------------------+
//| Redraw Value Display                                             |
//+------------------------------------------------------------------+
bool CGaugeBase::RedrawValueDisplay(double value) {
   if(StringLen(scaleLayer.gaugeLabel.value.text) > 0) { //--- Check text length
      if(!EraseLegendString(scaleLayer.gaugeLabel.value, scaleLayer.gaugeLabel.value.backgroundColor)) //--- Erase string
         return false;                     //--- Return false
   }
   scaleLayer.gaugeLabel.value.text = DoubleToString(value, (int)scaleLayer.gaugeLabel.value.decimalPlaces); //--- Set text
   if(!scaleLayer.obj_Canvas.FontSet(scaleLayer.gaugeLabel.value.fontName, scaleLayer.gaugeLabel.value.fontSize, scaleLayer.gaugeLabel.value.fontFlags, 0)) //--- Set font
      return false;                        //--- Return false
   scaleLayer.obj_Canvas.TextOut(scaleLayer.gaugeLabel.value.x, scaleLayer.gaugeLabel.value.y, scaleLayer.gaugeLabel.value.text, ColorToARGB(scaleLayer.gaugeLabel.value.textColor, scaleLayer.transparency), TA_CENTER | TA_VCENTER); //--- Draw text
   scaleLayer.obj_Canvas.Update(true);     //--- Update canvas
   return true;                            //--- Return true
}
```

First, we implement the "RedrawScaleMarks" method to regenerate the scale's ticks and labels. We declare indices and angles, set the multiplier from a predefined array or default to 1 if invalid, assign min and max values from inputs, and determine decimal places as 0. We check if the scale is ascending to set the forward direction and calculate the value range accordingly. We fix the null mark position to 1, assign min and max angles from the scale arc (swapping start and end), and compute the angle range with "NormalizeRadians", handling wrap-around if needed. We initialize left and right mark counts to 0, create a 361-entry buffer array for marks, center it at index 180, and set the zero mark's value and angle.

We prepare a sign based on direction and loop to populate right-side major marks, calculating values adjusted by multiplier and angles proportionally, incrementing the count until exceeding the range. We compute angle steps for major, medium (if per major is set), and minor ticks (adjusting for medium presence). We call "CalculateRanges" and "DrawRanges" to handle color zones first.

We set the font on the canvas with "FontSet" and increment the right count. For right marks, we loop to get each angle, compute inner and outer radii for major ticks, calculate start, end, and text positions using [MathCos](https://www.mql5.com/en/docs/math/mathcos) and [MathSin](https://www.mql5.com/en/docs/math/mathsin) with pi adjustment, draw the tick line with "LineAA" in ARGB color, determine digits (0 for zero value), convert the mark value to string with [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring), and draw the label with "TextOut" centered. If medium steps exist, we draw minor ticks before each medium, then medium ticks, and minors after; otherwise, just minors, using "DrawTick" and breaking on failure. For left marks (though count is 0 here, logic is symmetric), we similarly loop, but adjust angles positively and include a condition to draw only if not the zero mark unless position is 3, with minor/medium drawing in the opposite direction.

We define the "CalculateRanges" method to prepare color zones, computing inner and outer radii for major ticks, then looping through four ranges, calling "CalculateRangePie" if "IsValidRange" returns true, passing adjusted parameters, including border gap as radial gap. The other methods are straightforward. We have added comments for clarity. Finally, we just need to call these methods to do the heavy lifting as below.

```
//+------------------------------------------------------------------+
//| Redraw Gauge                                                     |
//+------------------------------------------------------------------+
void CGaugeBase::Redraw() {
   Draw();                                 //--- Call draw
   initializationComplete = true;          //--- Set initialization complete
}

//+------------------------------------------------------------------+
//| Draw Scale and Needle                                            |
//+------------------------------------------------------------------+
void CGaugeBase::Draw() {
   double diameter = m_radius * 2.0;       //--- Calculate diameter
   scaleLayer.scaleMarks.majorTickLength = (int)((diameter * 10.0) / 100.0); //--- Set major tick length
   scaleLayer.scaleMarks.mediumTickLength = (int)((diameter * 7.5) / 100.0); //--- Set medium tick length
   scaleLayer.scaleMarks.minorTickLength = (int)((diameter * 5.0) / 100.0); //--- Set minor tick length
   scaleLayer.scaleMarks.tickFontName = inputParams.tickFontName; //--- Set tick font name
   scaleLayer.scaleMarks.tickFontFlags = 0; //--- Initialize tick font flags
   if(inputParams.tickFontItalic)          //--- Check italic flag
      scaleLayer.scaleMarks.tickFontFlags |= FONT_ITALIC; //--- Add italic flag
   if(inputParams.tickFontBold)            //--- Check bold flag
      scaleLayer.scaleMarks.tickFontFlags |= FW_BOLD; //--- Add bold flag
   scaleLayer.scaleMarks.tickFontSize = (int)((diameter * 6.5) / 100.0); //--- Set tick font size
   scaleLayer.scaleMarks.tickFontGap = GetTickFontGap(scaleLayer.scaleMarks, 3); //--- Set tick font gap
   scaleLayer.externalLabelArea = 0;       //--- Set external label area
   scaleLayer.internalLabelArea = 0;       //--- Set internal label area
   GetTickLabelAreaSize(scaleLayer.internalLabelArea, scaleLayer.scaleMarks, 3); //--- Get tick label area size
   scaleLayer.borderSize = (int)((diameter * 2) / 100.0); //--- Set border size
   scaleLayer.borderGap = (int)((diameter * 3.0) / 100.0); //--- Set border gap
   scaleLayer.externalScaleGap = 0;        //--- Set external scale gap
   scaleLayer.internalScaleGap = scaleLayer.scaleMarks.majorTickLength; //--- Set internal scale gap
   if(inputParams.scaleAngleRange < 30)    //--- Check min angle range
      inputParams.scaleAngleRange = 30;    //--- Set min angle range
   if(inputParams.scaleAngleRange > 320)   //--- Check max angle range
      inputParams.scaleAngleRange = 320;   //--- Set max angle range
   int halfAngleRange = inputParams.scaleAngleRange / 2; //--- Calculate half angle range
   int startAngle = 90 + halfAngleRange + inputParams.rotationAngle; //--- Calculate start angle
   int endAngle = 90 - halfAngleRange + inputParams.rotationAngle; //--- Calculate end angle
   scaleLayer.scaleArc.centerX = m_radius + 5; //--- Set scale arc center X
   scaleLayer.scaleArc.centerY = m_radius + 5; //--- Set scale arc center Y
   scaleLayer.scaleArc.radius = m_radius - (scaleLayer.borderSize + scaleLayer.borderGap + scaleLayer.externalLabelArea + scaleLayer.externalScaleGap); //--- Set scale arc radius
   scaleLayer.scaleArc.startAngle = NormalizeRadians(DegreesToRadians(endAngle)); //--- Set start angle
   scaleLayer.scaleArc.endAngle = NormalizeRadians(DegreesToRadians(startAngle) - 0.0001); //--- Set end angle
   scaleLayer.scaleArc.clr = inputParams.scaleColor; //--- Set scale arc color
   needleLayer.needleCenter.radius = (int)((diameter * 5) / 100.0); //--- Set needle center radius
   needleLayer.needleCenter.display = true; //--- Set display flag
   needleLayer.needleCenter.centerX = scaleLayer.scaleArc.centerX; //--- Set needle center X
   needleLayer.needleCenter.centerY = scaleLayer.scaleArc.centerY; //--- Set needle center Y
   needleLayer.needleCenter.clr = inputParams.needleCenterColor; //--- Set needle center color
   int maxLegendRadius = m_radius - (scaleLayer.borderSize + scaleLayer.borderGap); //--- Calculate max legend radius
   int minLegendRadius = needleLayer.needleCenter.radius; //--- Set min legend radius
   int legendRadiusDelta = maxLegendRadius - minLegendRadius; //--- Calculate legend radius delta
   SetLegendStringParams(scaleLayer.gaugeLabel.description, inputParams.description, minLegendRadius, legendRadiusDelta); //--- Set description params
   SetLegendStringParams(scaleLayer.gaugeLabel.units, inputParams.units, minLegendRadius, legendRadiusDelta); //--- Set units params
   SetLegendStringParams(scaleLayer.gaugeLabel.multiplier, inputParams.multiplier, minLegendRadius, legendRadiusDelta); //--- Set multiplier params
   SetLegendStringParams(scaleLayer.gaugeLabel.value, inputParams.value, minLegendRadius, legendRadiusDelta); //--- Set value params
   CalculateCaseElements(scaleLayer.externalCase, scaleLayer.internalCase, scaleLayer.borderSize, scaleLayer.borderGap); //--- Calculate case elements
   scaleLayer.caseColor = inputParams.caseColor; //--- Set case color
   DrawCaseElements(scaleLayer.externalCase, scaleLayer.internalCase); //--- Draw case elements
   if(inputParams.displayScaleArc)         //--- Check display scale arc
      scaleLayer.obj_Canvas.Arc(scaleLayer.scaleArc.centerX, scaleLayer.scaleArc.centerY, scaleLayer.scaleArc.radius, scaleLayer.scaleArc.radius, scaleLayer.scaleArc.startAngle, scaleLayer.scaleArc.endAngle, ColorToARGB(scaleLayer.scaleArc.clr, scaleLayer.transparency)); //--- Draw scale arc
   RedrawScaleMarks(scaleLayer.internalCase, scaleLayer.scaleArc, scaleLayer.borderGap); //--- Redraw scale marks
   CalculateAndDrawLegends();              //--- Calculate and draw legends
   CalculateNeedle();                      //--- Calculate needle
   scaleLayer.obj_Canvas.Update(true);     //--- Update scale canvas
   needleLayer.obj_Canvas.Update(true);    //--- Update needle canvas
}
```

We implement the "Redraw" method in the "CGaugeBase" class to refresh the gauge visualization by calling the "Draw" method and then setting the "initializationComplete" flag to true, indicating the setup is finished. We then define the "Draw" method to handle the full rendering of the scale and needle layers.

We calculate the diameter as twice the radius, then set tick lengths proportionally: major at 10% of the diameter, medium at 7.5%, and minor at 5%. We assign the tick font name from inputs, initialize font flags to 0, and add [FONT\_ITALIC](https://www.mql5.com/en/docs/objects/textsetfont#font_type_flags) or "FW\_BOLD" if the respective flags are set. We scale the tick font size to 6.5% of the diameter and compute the font gap with "GetTickFontGap" using a string length of 3. We reset external and internal label areas to 0, then update the internal area with "GetTickLabelAreaSize". We set the border size to 2% and the gap to 3% of the diameter, the external scale gap to 0, and the internal to the major tick length. We clamp the scale angle range between 30 and 320 degrees if outside bounds, compute half the range, and derive start and end angles centered around 90 degrees plus rotation.

We position the scale arc center at radius plus 5 for both x and y, calculate its radius by subtracting border, gap, external label, and scale gap from the main radius, set start and end angles in normalized radians via "NormalizeRadians" and "DegreesToRadians" (with a small adjustment to end), and assign the scale color. For the needle center, we set its radius to 5% of the diameter, enable display, match its center to the scale arc, and apply the input center color. We determine the max legend radius by subtracting border and gap from the main radius, min as the needle center radius, and delta as their difference, then configure each legend string (description, units, multiplier, value) with "SetLegendStringParams" using these radii.

We prepare case elements with "CalculateCaseElements" passing border size and gap, set the case color, and render them with "DrawCaseElements". If the scale arc display is enabled, we draw it on the canvas with "Arc" using ARGB color. We redraw marks with "RedrawScaleMarks" passing internal case, scale arc, and border gap, calculate and draw legends with "CalculateAndDrawLegends", prepare the needle with "CalculateNeedle", and update both scale and needle canvases with "Update" set to true. With that, now our class is complete, and we can use it to set the properties as needed by calling the respective methods, but first, we will need to set the scale multipliers and string arrays on the global scope as below.

```
double scaleMultipliers[9] = {10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001}; //--- Define scale multipliers array
string scaleMultiplierStrings[9] = {"x10k", "x1k", "x100", "x10", " ", "/10", "/100", "/1k", "/10k"}; //--- Define multiplier strings array
```

We define the "scaleMultipliers" array as a global double array with 9 elements containing scaling factors: 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, and 0.0001, used for adjusting mark values on the gauge scale. We also define the "scaleMultiplierStrings" array as a global string array with 9 elements providing display labels: "x10k", "x1k", "x100", "x10", a space, "/10", "/100", "/1k", and "/10k", corresponding to the multipliers for visual representation in legends. We can now proceed to the initialization function and draw our gauge with updated properties.

```
gauge.SetCaseParameters(clrMintCream, 1, clrLightSkyBlue, 1); //--- Set case parameters
gauge.SetScaleParameters(250, 0, 0, 100, 4, 0, clrBlack, false); //--- Set scale parameters
gauge.SetTickParameters(0, 2, 10, 1, 4); //--- Set tick parameters
gauge.SetTickLabelFont(1, "Arial", false, false, clrBlack); //--- Set tick label font
gauge.SetRangeParameters(0, true, 0, 30, clrLimeGreen); //--- Set range 0
gauge.SetRangeParameters(1, true, 70, 100, clrCoral); //--- Set range 1
gauge.SetRangeParameters(2, true, 30, 70, clrYellow); //--- Set range 2
gauge.SetRangeParameters(3, false, 0, 0, clrGray); //--- Set range 3
gauge.SetLegendParameters(0, true, "RSI", 8, -180, 20, "Arial", false, false, clrBlueViolet); //--- Set legend 0
gauge.SetLegendParameters(3, true, "2", 4, 180, 13, "Arial", true, false, clrRed); //--- Set legend 3
gauge.SetNeedleParameters(1, clrBlack, clrDimGray, 1); //--- Set needle parameters
gauge.Redraw();                         //--- Redraw gauge
gauge.NewValue(0);                      //--- Set new value 0
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we configure the gauge's casing by calling "gauge.SetCaseParameters" with mint cream as the case color, border style 1, light sky blue border color, and gap size 1. We set the scale properties using "gauge.SetScaleParameters" with an angle range of 250 degrees, no rotation, min value 0, max 100, multiplier index 4 (corresponding to 1), style 0, black color, and arc display false. For ticks, we invoke "gauge.SetTickParameters" with style 0, size 2, major interval 10, 1 medium per major, and 4 minors per interval. We apply tick label font via "gauge.SetTickLabelFont" with size 1, "Arial" name, no italic or bold, and black color. We define ranges with "gauge.SetRangeParameters": index 0 enabled from 0 to 30 in lime green, index 1 from 70 to 100 in coral, index 2 from 30 to 70 in yellow, and index 3 disabled with dummy values in gray.

For legends, we use "gauge.SetLegendParameters" to set description (type 0) enabled with text "RSI", radius 8, angle -180, font size 20, "Arial", no italic or bold, blue violet color; and value (type 3) enabled with text "2" (for decimals), radius 4, angle 180, size 13, "Arial", italic true, no bold, red color. We configure the needle with "gauge.SetNeedleParameters" using center style 1, black center color, dim gray needle color, and fill style 1. Finally, we call "gauge.Redraw" to render the gauge and "gauge.NewValue" with 0 to initialize the pointer position. Upon compilation, we get the following outcome.

![GAUGE INITIALIZATION](https://c.mql5.com/2/186/GAUGE_1_GIF.gif)

From the visualization, we can see that we set the gauge with all the properties. What remains is breathing life to it so that it responds to data as new values are calculated. We will achieve that by calling the respective function in the [OnCalculate](https://www.mql5.com/en/docs/event_handlers/oncalculate) event handler.

```
static datetime lastBarTime = 0;        //--- Declare last bar time
bool isNewBar = (ratesTotal > 0 && time[ratesTotal - 1] != lastBarTime); //--- Check new bar
if(isNewBar)                            //--- If new bar
   lastBarTime = time[ratesTotal - 1];  //--- Update last bar time
if(isNewBar) {                          //--- If new bar
   int barsCalculated = BarsCalculated(rsiHandle); //--- Get bars calculated
   if(barsCalculated > 0) {             //--- Check calculated
      double currentRsiValue[1];        //--- Declare current value
      if(CopyBuffer(rsiHandle, 0, 0, 1, currentRsiValue) < 0) //--- Copy buffer
         Print("RSI CopyBuffer error for gauge"); //--- Print error
      else                              //--- Else
         gauge.NewValue(currentRsiValue[0]); //--- Set new value
   }
}
```

Here, we declare a static [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) variable "lastBarTime" initialized to 0 to track the timestamp of the most recently processed bar across calls to the [OnCalculate](https://www.mql5.com/en/docs/event_handlers/oncalculate) function. We determine if a new bar has formed by setting "isNewBar" to true if "ratesTotal" is greater than 0 and the timestamp at "time\[ratesTotal - 1\]" differs from "lastBarTime". If "isNewBar" is true, we update "lastBarTime" to the current bar's timestamp. In a separate check for "isNewBar", we retrieve the number of calculated bars for the Relative Strength Index handle using "BarsCalculated". If this is greater than 0, we create a single-element double array "currentRsiValue", attempt to copy the latest value from the handle's buffer 0 starting at position 0 with [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer), print an error if the copy fails, or otherwise pass the value to "gauge.NewValue" to update the gauge display. If you want to display the values per tick, you can ignore the new bar logic. Upon compilation, we get the following outcome.

![LIVE RSI GAUGE](https://c.mql5.com/2/186/GAUGE_1_GIF_2.gif)

After breathing life into the gauge, what remains is deleting the gauge to remove the rendered objects, and that is all.

```
//+------------------------------------------------------------------+
//| Deinitialize Indicator                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   gauge.Delete();                         //--- Delete gauge
   ChartRedraw();                          //--- Redraw chart
}
```

In the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, which is called when the indicator is removed from the chart or during terminal shutdown, we invoke the "gauge.Delete" method to remove the gauge's scale and needle layer objects, ensuring the cleanup of graphical resources. We then call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to refresh the chart display, removing any remnants of the gauge visualization. We end up with the following outcome.

![GAUGE DELETION GIF](https://c.mql5.com/2/186/gauge_3.gif)

From the visualization, we can see that we calculate the indicator, draw the gauge, set parameters, and delete it when not needed, hence achieving our objectives. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

We did the testing, and below is the compiled visualization in a single [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF "https://en.wikipedia.org/wiki/GIF") (GIF) bitmap image format.

![RSI CANVAS GAUGE BACKTEST GIF](https://c.mql5.com/2/186/GAUGE_BACKTEST_GIF.gif)

### Conclusion

In conclusion, we’ve created a gauge-style [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") indicator in [MQL5](https://www.mql5.com/) that visualizes momentum values on a circular dial with a dynamic needle, color-coded ranges for overbought and oversold zones, tick marks for precision, and legends for context, while integrating a traditional line plot and optimizing updates on new bars via the built-in [iRSI](https://www.mql5.com/en/docs/indicators/irsi) function. This indicator offers an engaging tool for market analysis, with flexible parameters for scales, fonts, and visuals. In upcoming parts, we will delve into modularizing the code so we can use it to create more advanced and stylish gauges. Stay tuned.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20632.zip "Download all attachments in the single ZIP archive")

[1\_\_Gauge-Based\_Indicator\_Part1.mq5](https://www.mql5.com/en/articles/download/20632/1__Gauge-Based_Indicator_Part1.mq5 "Download 1__Gauge-Based_Indicator_Part1.mq5")(85.47 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/502168)**
(3)


![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
18 Dec 2025 at 20:36

Thanks Allan for a creative idea.


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
20 Dec 2025 at 15:56

**Clemence Benjamin [#](https://www.mql5.com/en/forum/502168#comment_58763751):**

Thanks Allan for a creative idea.

Welcome [@Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024) for the kind feedback.


![Brian Mutuku Mwanthi](https://c.mql5.com/avatar/2026/1/696f3640-f635.png)

**[Brian Mutuku Mwanthi](https://www.mql5.com/en/users/gurbpipanalytica)**
\|
22 Dec 2025 at 05:50

Thank you for this Mr allan

Unrelated, the chart seems to have another EA taking multiple trades

Each candle has multiple trades, which EA is that

![From Novice to Expert: Navigating Market Irregularities](https://c.mql5.com/2/186/20645-from-novice-to-expert-navigating-logo.png)[From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)

Market rules are continuously evolving, and many once-reliable principles gradually lose their effectiveness. What worked in the past no longer works consistently over time. Today’s discussion focuses on probability ranges and how they can be used to navigate market irregularities. We will leverage MQL5 to develop an algorithm capable of trading effectively even in the choppiest market conditions. Join this discussion to find out more.

![Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://c.mql5.com/2/186/20657-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 9): Backtesting Portfolio Weights Updates](https://www.mql5.com/en/articles/20657)

This article describes the use of CSV files for backtesting portfolio weights updates in a mean-reversion-based strategy that uses statistical arbitrage through cointegrated stocks. It goes from feeding the database with the results of a Rolling Windows Eigenvector Comparison (RWEC) to comparing the backtest reports. In the meantime, the article details the role of each RWEC parameter and its impact in the overall backtest result, showing how the comparison of the relative drawdown can help us to further improve those parameters.

![Reimagining Classic Strategies (Part 20): Modern Stochastic Oscillators](https://c.mql5.com/2/186/20530-reimagining-classic-strategies-logo.png)[Reimagining Classic Strategies (Part 20): Modern Stochastic Oscillators](https://www.mql5.com/en/articles/20530)

This article demonstrates how the stochastic oscillator, a classical technical indicator, can be repurposed beyond its conventional use as a mean-reversion tool. By viewing the indicator through a different analytical lens, we show how familiar strategies can yield new value and support alternative trading rules, including trend-following interpretations. Ultimately, the article highlights how every technical indicator in the MetaTrader 5 terminal holds untapped potential, and how thoughtful trial and error can uncover meaningful interpretations hidden from view.

![Pure implementation of RSA encryption in MQL5](https://c.mql5.com/2/185/20273-pure-implementation-of-rsa-logo__1.png)[Pure implementation of RSA encryption in MQL5](https://www.mql5.com/en/articles/20273)

MQL5 lacks built-in asymmetric cryptography, making secure data exchange over insecure channels like HTTP difficult. This article presents a pure MQL5 implementation of RSA using PKCS#1 v1.5 padding, enabling safe transmission of AES session keys and small data blocks without external libraries. This approach provides HTTPS-like security over standard HTTP and even more, it fills an important gap in secure communication for MQL5 applications.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/20632&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048826759371136877)

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