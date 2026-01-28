---
title: Visualize this! MQL5 graphics library similar to 'plot' of R language
url: https://www.mql5.com/en/articles/2866
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:18:00.296303
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=yevzvslhtmzoqpjcyewjlctozlmhqzrm&ssn=1769192279909604851&ssn_dr=0&ssn_sr=0&fv_date=1769192279&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2866&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Visualize%20this!%20MQL5%20graphics%20library%20similar%20to%20%27plot%27%20of%20R%20language%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919227905474536&fz_uniq=5071729552218074454&sv=2552)

MetaTrader 5 / Examples


When studying trading logic, visual representation in the form of graphs is of great importance. A number of programming languages popular among the scientific community (such as R and Python) feature the special 'plot' function used for visualization. It allows drawing lines, point distributions and histograms to visualize patterns.

The important advantage of the 'plot' function is that you need only a few lines of code to plot any graph. Simply pass the data array as a parameter, specify the graph type and you are ready to go! The 'plot' function performs all the routine operations of calculating a scale, building axes, selecting a color, etc.

In MQL5, all features of the function are represented by the [graphics library](https://www.mql5.com/en/docs/standardlibrary/graphics) method from the Standard Library. The sample code and the result of its execution are shown below:

#include <Graphics\\Graphic.mqh>

#define RESULT\_OR\_NAN(x,expression) ((x==0)?(double)"nan":expression)

//\-\-\- Functions

double BlueFunction(double x)   { return(RESULT\_OR\_NAN(x,10\*x\*sin(1/x)));      }

double RedFunction(double x)    { return(RESULT\_OR\_NAN(x,sin(100\*x)/sqrt(x))); }

double OrangeFunction(double x) { return(RESULT\_OR\_NAN(x,sin(100\*x)/sqrt(-x)));}

//+------------------------------------------------------------------+

//\| Script program start function                                    \|

//+------------------------------------------------------------------+

voidOnStart()

{

double from=-1.2;

double to=1.2;

double step=0.005;

    CGraphic graphic;

    graphic.Create(0,"G",0,30,30,780,380);

//\-\-\- colors

    CColorGenerator generator;

uint blue= generator.Next();

uint red = generator.Next();

uint orange=generator.Next();

//\-\-\- plot all curves

    graphic.CurveAdd(RedFunction,from,to,step,red,CURVE\_LINES,"Red");

    graphic.CurveAdd(OrangeFunction,from,to,step,orange,CURVE\_LINES,"Orange");

    graphic.CurveAdd(BlueFunction,from,to,step,blue,CURVE\_LINES,"Blue");

    graphic.CurvePlotAll();

    graphic.Update();

}

![](https://c.mql5.com/2/25/3functions_graphic.png)

### CCanvas base class and its development

The standard library contains the [CCanvas](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) base class designed for fast and convenient plotting of images directly on price charts. The class is based on creating a graphical [resource](https://www.mql5.com/en/docs/runtime/resources) and plotting simple primitives (dots, straight lines and polylines, circles, triangles and polygons) on the canvas. The class implements the functions for filling shapes and displaying text using the necessary font, color and size.

Initially, CCanvas contained only two modes of displaying graphical primitives — with antialiasing (AA) and without it. Then, the new functions were added for plotting the primitives based on the [Wu's algorithm](https://en.wikipedia.org/wiki/Xiaolin_Wu "https://ru.wikipedia.org/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%92%D1%83"):

- [LineWu](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaslinewu)— straight line
- [PolylineWu](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaspolylinewu)— polyline

- [PolygonWu](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvaspolygonwu)— polygon

- [TriangleWu](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvastrianglewu)— triangle

- [CircleWu](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvascirclewu)— circle
- [EllipseWu](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasellipsewu)— ellipse

The Wu's algorithm combines high-quality elimination of aliasing with the operation speed close to the [Bresenham's algorithm](https://en.wikipedia.org/wiki/Bresenham "Bresenham's algorithm") one without anti-aliasing. It also visually differs from the standard anti-aliasing algorithm (AA) implemented in CCanvas. Below is an example of plotting a circle using three different functions:

#include<Canvas\\Canvas.mqh>

CCanvas canvas;

//+------------------------------------------------------------------+

//\| Script program start function                                    \|

//+------------------------------------------------------------------+

voidOnStart()

{

int      Width=800;

int      Height=600;

//\-\-\- create canvas

if(!canvas.CreateBitmapLabel(0,0,"CirclesCanvas",30,30,Width,Height))

      {

Print("Error creating canvas: ",GetLastError());

      }

//\-\-\- draw

    canvas.Erase(clrWhite);

    canvas.Circle(70,70,25,clrBlack);

    canvas.CircleAA(120,70,25,clrBlack);

    canvas.CircleWu(170,70,25,clrBlack);

//\-\-\-

    canvas.Update();

}

![](https://c.mql5.com/2/25/3Circles.png)

As we can see, [CircleAA()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvascircleaa) with the standard smoothing algorithm draws a thicker line as compared to the [CircleWu()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvascirclewu) function according to the Wu's algorithm. Due to its smaller thickness and better calculation of transitional shades, CircleWu looks more neat and natural.

There are also other improvements in the CCanvas class:

1. Added the new [ellipse](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasellipse) primitive with two anti-aliasing options — [EllipseAA()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasellipseaa) and [EllipseWu()](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasellipsewu)

2. Added the [filling area function](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas/ccanvasfill) overload with the new parameter responsible for the "filling sensitivity" (the threshould parameter).

### The algorithm of working with the library

**1\.** After connecting the library, we should create the CGraphic class object. The curves to be drawn will be added to it.

Next, we should call the Create() method for the created object. The method contains the main graph parameters:

1. Graph ID
2. Object name
3. Window index

4. Graph anchor point
5. Graph width and height

The method applies the defined parameters to create a chart object and a graphical resource to be used when plotting a graph.

//\-\-\- object for creating graphs

    CGraphic graphic;

//\-\-\- create canvas

    graphic.Create(0,"Graphic",0,30,30,830,430);

As a result, we have a ready-made canvas.

**2.** Now, let's fill our object with curves. Adding is performed using the [CurveAdd()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphiccurveadd) method able to plot curves in four different ways:

1. Based on the double type one-dimensional array. In this case, values from the array are displayed on Y axis, while array indices serve as X coordinates.
2. Based on two x\[\] and y\[\] double type arrays.
3. Based on the CPoint2D array.
4. Based on the CurveFunction() pointer and three values for building the function arguments: initial, final and increment by argument.

The CurveAdd() method returns the pointer to the [CCurve](https://www.mql5.com/en/docs/standardlibrary/graphics/ccurve) class providing fast access to the newly created curve and ability to change its properties.

double x\[\]={-10,-4,-1,2,3,4,5,6,7,8};

double y\[\]={-5,4,-10,23,17,18,-9,13,17,4};

    CCurve \*curve=graphic.CurveAdd(x,y,CURVE\_LINES);

**3\.** Any of the added curves can then be displayed on the chart. This can be done in three ways.

1. By using the [CurvePlotAll()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphiccurveplotall) method that automatically draws all curves added to the chart.

graphic.CurvePlotAll();

2. By using the [CurvePlot()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphiccurveplot) method that draws a curve by the specified index.

graphic.CurvePlot(0);

3. By using the Redraw() method and setting the curve's Visible property to 'true'.

curve.Visible(true);


    graphic.Redraw();


**4\.** In order to plot a graph on the chart, call the [Update()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphicupdate) method. As a result, we obtain the entire code of the script for plotting a simple graph:

#include <Graphics\\Graphic.mqh>

//+------------------------------------------------------------------+

//\| Script program start function                                    \|

//+------------------------------------------------------------------+

voidOnStart()

{

    CGraphic graphic;

    graphic.Create(0,"Graphic",0,30,30,780,380);

double x\[\]={-10,-4,-1,2,3,4,5,6,7,8};

double y\[\]={-5,4,-10,23,17,18,-9,13,17,4};

    CCurve \*curve=graphic.CurveAdd(x,y,CURVE\_LINES);

    graphic.CurvePlotAll();

    graphic.Update();

}

Below is the resulting graph:

![](https://c.mql5.com/2/25/1__15.png)

The properties of the graph and any of its functions can be changed at any moment. For example, we can add labels to the graph axes, change the name of the curve and enable the mode of spline approximation for it:

#include <Graphics\\Graphic.mqh>

//+------------------------------------------------------------------+

//\| Script program start function                                    \|

//+------------------------------------------------------------------+

voidOnStart()

{

    CGraphic graphic;

    graphic.Create(0,"Graphic",0,30,30,780,380);

double x\[\]={-10,-4,-1,2,3,4,5,6,7,8};

double y\[\]={-5,4,-10,23,17,18,-9,13,17,4};

    CCurve \*curve=graphic.CurveAdd(x,y,CURVE\_LINES);

curve.Name("Example");

curve.LinesIsSmooth(true);

graphic.XAxis().Name("X - axis");

graphic.XAxis().NameSize(12);

graphic.YAxis().Name("Y - axis");

graphic.YAxis().NameSize(12);

    graphic.YAxis().ValuesWidth(15);

    graphic.CurvePlotAll();

    graphic.Update();

DebugBreak();

}

![](https://c.mql5.com/2/25/curve.png)

If the changes had been set after calling CurvePlotAll, we would have had to additionally call the Redraw method to see them.

Like many modern libraries, Graphics contains various ready-made algorithms considerably simplifying plotting charts:

1. The library is capable of auto generating contrasting colors of curves if they are not explicitly specified.
2. Graph axes feature the parametric auto scaling mode that can be disabled if necessary.
3. Curve names are generated automatically depending on their type and order of addition.
4. The graph's working area is automatically lined and actual axes are set.
5. It is possible to smooth curves when using lines.

The Graphics library also has a few additional methods for adding new elements to the chart:

1. [TextAdd()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphictextadd) — add a text to an arbitrary position on the chart. Coordinates should be set in real scale. Use the FontSet method for precise configuring of a displayed text.

2. [LineAdd()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphiclineadd) — add a line to an arbitrary position on the chart. Coordinates should be set in real scale.
3. [MarksToAxisAdd()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphicmarkstoaxisadd) — add new labels to a specified coordinate axis.

Data on adding the elements is not stored anywhere. After plotting a new curve on the chart or re-drawing a previous one, they disappear.

### Graph types

The Graphics library supports basic types of curves plotting types. All of them are specified in the ENUM\_CURVE\_TYPE enumeration:

1. [CURVE\_POINTS](https://www.mql5.com/en/articles/2866#CURVE_POINTS) — draw a point curve
2. [CURVE\_LINES](https://www.mql5.com/en/articles/2866#CURVE_LINES) — draw a line curve
3. [CURVE\_POINTS\_AND\_LINES](https://www.mql5.com/en/articles/2866#CURVE_POINTS_AND_LINES) — draw both point and line curves
4. [CURVE\_STEPS](https://www.mql5.com/en/articles/2866#CURVE_STEPS) — draw a stepped curve

5. [CURVE\_HISTOGRAM](https://www.mql5.com/en/articles/2866#CURVE_HISTOGRAM) — draw a histogram curve
6. [CURVE\_NONE](https://www.mql5.com/en/articles/2866#CURVE_NONE) — do not draw a curve


Each of these modes has its own properties affecting the display of a curve on the chart. The CCurve pointer to a curve allows fast modification of these properties. Therefore, it is recommended to remember all the pointers returned by the CurveAdd method. A property name always starts with a curve drawing mode it is used in.

Let's have a more detailed look at the properties of each of the types.

1\. **CURVE\_POINTS** is the fastest and simplest mode. Each curve coordinate is displayed as a point having specified properties:

- PointsSize — point size
- PointsFill — flag indicating if the filling is present

- PointsColor — filling color

- PointsType — points type


In this case, the color of the curve itself defines the color of the points' borders.

CCurve \*curve=graphic.CurveAdd(x,y,ColorToARGB(clrBlue,255),CURVE\_POINTS);

    curve.PointsSize(20);

    curve.PointsFill(true);

    curve.PointsColor(ColorToARGB(clrRed,255));

![](https://c.mql5.com/2/25/1__16.png)

The type of points defines a certain geometric shape from the ENUM\_POINT\_TYPE enumeration. This shape is to be used for displaying all curve points. In total, ENUM\_POINT\_TYPE includes ten main geometric shapes:

01. POINT\_CIRCLE — circle (used by default)

02. POINT\_SQUARE — square

03. POINT\_DIAMOND — diamond

04. POINT\_TRIANGLE — triangle

05. POINT\_TRIANGLE\_DOWN — inverted triangle

06. POINT\_X\_CROSS — cross

07. POINT\_PLUS — plus

08. POINT\_STAR — star

09. POINT\_HORIZONTAL\_DASH — horizontal line

10. POINT\_VERTICAL\_DASH — vertical line


![](https://c.mql5.com/2/25/points__1.png)

Below is an example of a visual representation of different kinds of iris (see the article ["Using self-organizing feature maps (Kohonen Maps) in MetaTrader 5"](https://www.mql5.com/en/articles/283)) from the attached _IrisSample.mq_ 5 script.

![](https://c.mql5.com/2/25/Iris_sample.png)

2\. **CURVE\_LINES** display mode is the main mode for visualizing the curves, in which each pair of dots is connected by one or several (in case of smoothing) straight lines. The mode properties are as follows:

- LinesStyle — line style from the [ENUM\_LINE\_STYLE](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_line_style) enumeration
- LinesSmooth — flag indicating if smoothing should be performed

- LinesSmoothTension — smoothing degree

- LinesSmoothStep — length of approximating lines when smoothing


Graphics features the standard parametric curve smoothing algorithm. It consists of two stages:

1. Two reference points are defined for each pair of points on the basis of their derivatives
2. A [Bezier curve](https://en.wikipedia.org/wiki/B%C3%A9zier_curve "https://ru.wikipedia.org/wiki/%D0%9A%D1%80%D0%B8%D0%B2%D0%B0%D1%8F_%D0%91%D0%B5%D0%B7%D1%8C%D0%B5") with a specified approximation step is plotted based on these four points

LinesSmoothTension parameter takes the values (0.0; 1.0\]. If LinesSmoothTension is set to 0.0, no smoothing happens. By increasing this parameter, we receive more and more smoothed curve.

CCurve \*curve=graphic.CurveAdd(x,y,ColorToARGB(clrBlue,255),CURVE\_LINES);

    curve.LinesStyle(STYLE\_DOT);

    curve.LinesSmooth(true);

    curve.LinesSmoothTension(0.8);

    curve.LinesSmoothStep(0.2);

![](https://c.mql5.com/2/25/2__11.png)

3\. **CURVE\_POINTS\_AND\_LINES** combines the previous two display modes and their properties.

4\. In the **CURVE\_STEPS** mode, each pair of points is connected by two lines as a step. The mode has two properties:

- LinesStyle — this property is taken from CURVE\_POINTS and defines the line style
- StepsDimension — step dimension: 0 — x (a horizontal line followed by a vertical one) or 1 — y (a vertical line followed by a horizontal one).


CCurve \*curve=graphic.CurveAdd(x,y,ColorToARGB(clrBlue,255),CURVE\_STEPS);

    curve.LinesStyle(STYLE\_DASH);

    curve.StepsDimension(1);

![](https://c.mql5.com/2/25/3__9.png)

5\. The **CURVE\_HISTOGRAM** mode draws a standard bar histogram. The mode features a single property:

- HistogramWidth — bar width

If the value is too large, the bars may overlap, and the bars with greater Y value "absorb" the adjacent bars having smaller values.

![](https://c.mql5.com/2/25/4__5.png)

6\. The **CURVE\_NONE** mode disables graphical representation of curves regardless of its visibility.

When auto scaling, all curves added to the chart have certain values. Therefore, even if the curve is not plotted or set to the CURVE\_NONE mode, its values are still taken into account.

### Graphs on functions — fast generation in a few lines

Another advantage of the library is working with CurveFunction pointers to functions. In MQL5, pointers to functions accept only global or static functions, while the function syntax should fully correspond to the pointer one. In our case, CurveFunction is configured for the functions receiving a double type parameter receiving double as well.

In order to construct a curve by a pointer to a function, we also need to accurately set the initial (from) and final (to) argument values, as well as its increment (step). The less the increment value, the more function points we have for constructing it. To create a data series, use [CurveAdd()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphiccurveadd), while to plot a function, apply [CurvePlot()](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphiccurveplot) or [CurvePlotAll().](https://www.mql5.com/en/docs/standardlibrary/graphics/cgraphic/cgraphiccurveplotall)

For example, let's create a parabolic function and draw it in various increments:

#include <Graphics\\Graphic.mqh>

//+------------------------------------------------------------------+

//\| Parabola                                                         \|

//+------------------------------------------------------------------+

double Parabola(double x) { returnMathPow(x,2); }

//+------------------------------------------------------------------+

//\| Script program start function                                    \|

//+------------------------------------------------------------------+

voidOnStart()

{

double from1=-5;

double to1=0;

double step1=1;

double from2=0;

double to2=5;

double step2=0.2;

    CurveFunction function = Parabola;

    CGraphic graph;

    graph.Create(0,"Graph",0,30,30,780,380);

    graph.CurveAdd(function,from1,to1,step1,CURVE\_LINES);

    graph.CurveAdd(function,from2,to2,step2,CURVE\_LINES);

    graph.CurvePlotAll();

    graph.Update();

}

![](https://c.mql5.com/2/25/parabola.png)

The library works with functions having break points (one of the coordinates has the value of plus or minus infinity or is non-numeric). Increment by function should be considered, since sometimes, we can simply miss a break point. In this case, a graph does not meet expectations. For example, let's draw two hyperbolic functions within the \[-5.0; 5.0\] segment with the first function having a step of 0.7 and the second one — 0.1. The result is displayed below:

![](https://c.mql5.com/2/25/Hyperbola.png)

As we can see in the image above, we have simply missed the break point when using a step of 0.7. As a result, the resulting curve has almost nothing to do with the real hyperbolic function.

A zero divide error may occur when using the functions. There are two ways to handle this issue:

- disable the check for zero divide in metaeditor.ini


\[Experts\]

FpNoZeroCheckOnDivision=1

- or analyze an equation used in a function and return a valid value for such instances. An example of such handling using a macro can be found in the attached _3Functions.mq5_ and _bat.mq5_ files.





![](https://c.mql5.com/2/25/bat.png)


### Quick plotting functions

The Graphics library also includes a number of [GraphPlot()](https://www.mql5.com/en/docs/standardlibrary/graphics/graphplot) global functions that perform all graph plotting stages based on available data and return an object name on the chart as a result. These functions are similar to 'plot' of R or Phyton languages allowing you to instantly visualize the data available in different formats.

The GraphPlot function has 10 various overloads allowing to plot a different number of curves on a single chart and set them in different ways. All you need to do is form data for plotting a curve using one of the available methods. For example, the source code for quick plotting of the x\[\] and y\[\] arrays looks as follows:

voidOnStart()

{

double x\[\]={-10,-4,-1,2,3,4,5,6,7,8};

double y\[\]={-5,4,-10,23,17,18,-9,13,17,4};

    GraphPlot(x,y);

}

It looks similar on R:

\> x<-c(-10,-4,-1,2,3,4,5,6,7,8)

> y<-c(-5,4,-10,23,17,18,-9,13,17,4)

> plot(x,y)

Results of comparing graphs by three main display modes built by the GraphPlot function on MQL5 and the plot function on R:

1\. Point curves

![](https://c.mql5.com/2/25/points__2.png)

2\. Lines

![](https://c.mql5.com/2/25/lines__1.png)

3\. Histogram

![](https://c.mql5.com/2/25/histograms.png)

Apart from quite significant visual differences of the GraphPlot() and plot() functions operation, they apply different input parameters. While the plot() function allows setting specific curve parameters (for example, 'lwd' that changes the lines width), the GraphPlot() function includes only key parameters necessary for building data.

Let's name them:

1. Curve data in one of the [four formats](https://www.mql5.com/en/articles/2866#DATA_TYPE) described above.

2. [Plotting type](https://www.mql5.com/en/articles/2866#ENUM_CURVE_TYPE) (default is CURVE\_POINTS).

3. Object name (default is NULL).

Each graph created using the Graphics library consists of a chart object and a graphical resource assigned to it. The graphical resource name is formed based on the object name by simply adding "::" before it. For example, if the object name is "SomeGraphic", the name of its graphical resource is "::SomeGraphic".

The GraphPlot() function has a fixed anchor point on a chart x=65 and y=45. The width and height of the graph are calculated based on the chart size: the width comprises 60% of the chart one, while the height is 65% of the chart's height. Thus, if the current chart dimensions are less than 65 to 45, the GraphPlot() function is not able to work correctly.

If you apply a name of an already created object while creating a graph, the Graphics library attempts to display the graph on that object after checking its resource type. If the resource type is [OBJ\_BITMAP\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_bitmap_label), plotting is performed on the same object-resource pair.

If an object name is explicitly passed to the GraphPlot() function, an attempt is made to find that object and display the graph on it. If the object is not found, a new object-resource pair is automatically created based on the specified name. When using the GraphPlot() function without an explicitly specified object name, the "Graphic" standard name is used.

In this case, you are able to specify a graph anchor point and its size. To do this, create an object-resource pair with necessary parameters and pass the name of the created object to the GraphPlot() function. By creating a pair with the Graphic object name, we redefine and fix the standard canvas for the GraphPlot function eliminating the necessity to pass the object name at each call.

For example, let's take the data from the above example and set a new graph size of 750х350. Also, let's move the [anchor point](https://www.mql5.com/en/docs/constants/objectconstants/enum_anchorpoint) to the upper left corner:

voidOnStart()

{

//\-\-\- create object on chart and dynamic resource

string name="Graphic";

long x=0;

long y=0;

int width=750;

int height=350;

int data\[\];

ArrayResize(data,width\*height);

ZeroMemory(data);

ObjectCreate(0,name,OBJ\_BITMAP\_LABEL,0,0,0);

ResourceCreate("::"+name,data,width,height,0,0,0,COLOR\_FORMAT\_XRGB\_NOALPHA);

ObjectSetInteger(0,name,OBJPROP\_XDISTANCE,x);

ObjectSetInteger(0,name,OBJPROP\_YDISTANCE,y);

ObjectSetString(0,name,OBJPROP\_BMPFILE,"::"+name);

//\-\-\- create x and y array

double arr\_x\[\]={-10,-4,-1,2,3,4,5,6,7,8};

double arr\_y\[\]={-5,4,-10,23,17,18,-9,13,17,4};

//\-\-\- plot x and y array

    GraphPlot(arr\_x,arr\_y,CURVE\_LINES);

}

![](https://c.mql5.com/2/25/2graphics_gif.gif)

### Sample scientific graphs

The Standard Library includes the [Statistics](https://www.mql5.com/en/docs/standardlibrary/mathematics/stat) section featuring the functions for working with multiple [statistical distributions](https://www.mql5.com/en/articles/2742) from the probability theory. Each distribution is accompanied by a sample graph and a code to retrieve it. Here, we simply display these graphs in a single GIF. The source codes of examples are attached in the MQL5.zip file. Unpack them to MQL5\\Scripts.

All these examples have a price chart disabled by the [CHART\_SHOW](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer) property:

//\-\-\- disable a price chart

ChartSetInteger(0,CHART\_SHOW,false);

This allows us to turn a chart window into a single large [canvas](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) and draw objects of any complexity applying the [graphical resources](https://www.mql5.com/en/docs/runtime/resources).

![](https://c.mql5.com/2/26/StatisticalDistributions.gif)
Read the article " [An example of developing a spread strategy for Moscow Exchange futures](https://www.mql5.com/en/articles/2739)" that demonstrates applying the [graphics library](https://www.mql5.com/en/docs/standardlibrary/graphics) to display the training sample and the linear regression results obtained by the Alglib library.


### Key advantages of the graphics library

The MQL5 language allows developers not only to create trading robots and technical indicators but also perform complex mathematical calculations using the [ALGLIB](https://www.mql5.com/en/code/1146), [Fuzzy](https://www.mql5.com/en/code/13697) and [Statistics](https://www.mql5.com/en/docs/standardlibrary/mathematics/stat) libraries. Obtained data is then easily visualized by provided graphics library. Most operations are automated, and the library offers the extensive functionality:

- 5 graph [display types](https://www.mql5.com/en/articles/2866#ENUM_CURVE_TYPE)
- 10 chart [marker types](https://www.mql5.com/en/articles/2866#enum_point_type)

- auto scaling of charts by X and Y axes
- color auto selection even if a graph features several constructions

- smoothing lines using the standard anti-aliasing or the more advanced [Bresenham's algorithm](https://en.wikipedia.org/wiki/Bresenham "Bresenham's algorithm")
- ability to set [spline approximation](https://www.mql5.com/en/docs/standardlibrary/graphics/ccurve/ccurvelinesissmooth) parameters for displaying lines

- ability to plot a graph using a single line of code based on the x\[\] and y\[\] arrays
- ability to plot graphs using pointers to functions


The graphics library simplifies plotting scientific graphs and raises the development of trading applications to a new level. The MetaTrader 5 platform allows you to perform mathematical calculations of any complexity and display results directly in the terminal window in a professional way.

Try the attached codes. You no more need third-party packages!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2866](https://www.mql5.com/ru/articles/2866)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2866.zip "Download all attachments in the single ZIP archive")

[3Functions.mq5](https://www.mql5.com/en/articles/download/2866/3functions.mq5 "Download 3Functions.mq5")(1.71 KB)

[Bat.mq5](https://www.mql5.com/en/articles/download/2866/bat.mq5 "Download Bat.mq5")(2.94 KB)

[iris.txt](https://www.mql5.com/en/articles/download/2866/iris.txt "Download iris.txt")(4.59 KB)

[IrisSample.mq5](https://www.mql5.com/en/articles/download/2866/irissample.mq5 "Download IrisSample.mq5")(7.23 KB)

[MQL5.zip](https://www.mql5.com/en/articles/download/2866/mql5.zip "Download MQL5.zip")(44.97 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/187978)**
(63)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
1 Aug 2023 at 06:15

**Nikolai Semko [#](https://www.mql5.com/ru/forum/169286/page6#comment_48481238):**

We need a new class with fewer functions than CCanvas. It's better to make this class as close as possible to [JS canvas](https://www.mql5.com/go?link=https://www.w3schools.com/tags/ref_canvas.asp "https://www.w3schools.com/tags/ref_canvas.asp"), which doesn't have many functions in essence, for ease of learning and adoption, as the whole IT is moving towards web.

but this is ideal.

I could probably write something like this myself, but it would take at least half a year full-time

although personally I don't like everything in JS Canvas, but for standardisation it would be right to implement something similar with small changes to do without string parsing.

But for normal work we need to redo MT5 event model, as it is just awful

It is clear that everything is "under the knife", but still the question was a bit different. What it is desirable to change/refine in the existing class...

![BeeXXI Corporation](https://c.mql5.com/avatar/2024/9/66dbee89-a47e.png)

**[Nikolai Semko](https://www.mql5.com/en/users/nikolay7ko)**
\|
1 Aug 2023 at 06:20

**Artyom Trishkin [#](https://www.mql5.com/ru/forum/169286/page6#comment_48481267):**

It is clear that everything is "under the knife", but still the question was a bit different. What is it desirable to change/improve in the existing class...

smooth methods to bring them to life. Now I think there are even functions in CCanvas that are undocumented.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
1 Aug 2023 at 06:42

**Nikolai Semko [#](https://www.mql5.com/ru/forum/169286/page6#comment_48481272):**

smoothed methods to bring them to life. Now it seems there are even functions in CCanvas that are undocumented.

Really? Didn't pay attention to that... I'll have to take a look at my leisure....

![Aleksey Vyazmikin](https://c.mql5.com/avatar/2024/6/6678986f-2caa.png)

**[Aleksey Vyazmikin](https://www.mql5.com/en/users/-aleks-)**
\|
1 Aug 2023 at 13:15

I need the "drawers with moustache" type charts, are there any plans to add them?

Also missing is the ability to use subwindows/basements, i.e. when you want to use two different scales.

![Vitaliy Davydov](https://c.mql5.com/avatar/2019/3/5C98AE5B-7DD0.jpg)

**[Vitaliy Davydov](https://www.mql5.com/en/users/viteck116)**
\|
16 Jul 2025 at 07:58

Hello.

Does the CGraphic class  have inbuilt methods for drawing [heat maps](https://www.mql5.com/en/articles/5451 "Article: Econometric approach to finding market patterns: autocorrelation, heat maps and scatter plots ")?

![Graphical Interfaces X: Updates for the Rendered table and code optimization (build 10)](https://c.mql5.com/2/26/MQL5-avatar-X-Auto-table-001.png)[Graphical Interfaces X: Updates for the Rendered table and code optimization (build 10)](https://www.mql5.com/en/articles/3042)

We continue to complement the Rendered table (CCanvasTable) with new features. The table will now have: highlighting of the rows when hovered; ability to add an array of icons for each cell and a method for switching them; ability to set or modify the cell text during the runtime, and more.

![Graphical interfaces X: New features for the Rendered table (build 9)](https://c.mql5.com/2/26/MQL5-avatar-X-table-003-1.png)[Graphical interfaces X: New features for the Rendered table (build 9)](https://www.mql5.com/en/articles/3030)

Until today, the CTable was the most advanced type of tables among all presented in the library. This table is assembled from edit boxes of the OBJ\_EDIT type, and its further development becomes problematic. Therefore, in terms of maximum capabilities, it is better to develop rendered tables of the CCanvasTable type even at the current development stage of the library. Its current version is completely lifeless, but starting from this article, we will try to fix the situation.

![Calculating the Hurst exponent](https://c.mql5.com/2/26/22.png)[Calculating the Hurst exponent](https://www.mql5.com/en/articles/2930)

The article thoroughly explains the idea behind the Hurst exponent, as well as the meaning of its values and the calculation algorithm. A number of financial market segments are analyzed and the method of working with MetaTrader 5 products implementing the fractal analysis is described.

![Graphical Interfaces X: The Multiline Text box control (build 8)](https://c.mql5.com/2/26/MQL5-avatar-graphic-interface.png)[Graphical Interfaces X: The Multiline Text box control (build 8)](https://www.mql5.com/en/articles/3004)

The Multiline Text box control is discussed. Unlike the graphical objects of the OBJ\_EDIT type, the presented version will not have restrictions on the number of input characters. It also adds the mode for turning the text box into a simple text editor, where the cursor can be moved using the mouse or keys.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dmsdhokhwkqidhpnpbqthascxrbwjuja&ssn=1769192279909604851&ssn_dr=0&ssn_sr=0&fv_date=1769192279&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2866&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Visualize%20this!%20MQL5%20graphics%20library%20similar%20to%20%27plot%27%20of%20R%20language%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919227905488964&fz_uniq=5071729552218074454&sv=2552)

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