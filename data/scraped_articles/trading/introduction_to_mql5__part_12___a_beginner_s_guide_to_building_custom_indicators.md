---
title: Introduction to MQL5 (Part 12): A Beginner's Guide to Building Custom Indicators
url: https://www.mql5.com/en/articles/17096
categories: Trading, Trading Systems, Indicators
relevance_score: 9
scraped_at: 2026-01-22T17:27:35.736071
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/17096&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049147842536252883)

MetaTrader 5 / Trading


### Introduction

Welcome back to our MQL5 series! So far, we’ve covered a lot, including dealing with built-in indicators, creating Expert Advisors, exploring fundamental MQL5 concepts, and putting our knowledge to use through practical projects. It's time to advance by learning how to create a custom indicator from scratch. We'll gain a  more in-depth understanding of how indicators operate internally, allowing us complete control over their operation and design rather than depending on built-in features. Have you ever wondered how the Moving Average or MACD, two of MQL5's built-in indicators, are created? If there were no such functions as iRSI or iMA, could you still build indicators?

Using a project-based approach, we will divide the process into two main parts. First, without utilizing the iMA function, we will build a Moving Average indicator entirely from scratch. Next, we'll go one step further and transform the Moving Average from the conventional line shape into a candle-style indicator. In addition, this practical method will open up new avenues for developing trading tools that are specifically suited to your requirements.

**Moving Average in Line Format**

![Figure 1. MA Line Format](https://c.mql5.com/2/116/Figure_1.png)

**Moving Average in Candle Format**

![Figure 2. MA Candle Format](https://c.mql5.com/2/116/Figure_2.png)

In this article, you'll learn:

- How to create a custom indicators from scratch in MQL5.
- The difference between plotting indicators in line format and candle format.
- Using indicator buffers to store and display calculated values.
- Set indicator properties correctly.
- Creating a custom Moving Average indicator in MQL5.
- Mapping indicator buffers using SetIndexBuffer().
- Processing price data using the OnCalculate() function.
- How to determine candle colors based on price movement.

Even if you're new to MQL5, you can follow along without feeling overwhelmed because this article is meant to be beginner-friendly. Each line of code will be thoroughly explained, dissecting intricate ideas into manageable steps. By the end of this tutorial, you should have a firm understanding of how custom indicators function in MQL5, since I'll keep things straightforward and useful.

### 1\. Custom Indicators

**1.1. What Are Custom Indicators?**

Custom indicators are those that are not available in MQL5 by default. Unlike the built-in indicators that MetaTrader 5 comes with, such as Moving Averages (MA), Relative Strength Index (RSI), or MACD, users can create their own indicators to perform particular computations, provide signals, or present market data however they see fit.

It is simple to incorporate built-in indicators into trading strategies because they can be accessible directly using functions like iMA(), iRSI(), and iMACD(). But when it comes to customizing, they are limited. On the other hand, custom indicators provide complete control over the calculations, logic, and chart display, enabling you to create tools that are tailored to their particular trading requirements. Traders who wish to do market analysis in methods not supported by built-in indicators will find these indicators especially helpful. You can design unique indicators that provide a more thorough comprehension of price movement, trends, and possible trading opportunities by utilizing MQL5 programming.

**Analogy**

Consider arranging a bookshelf in your living space. Built-in indicators are comparable to store with pre-assembled bookshelves. These bookshelves have predetermined sizes, forms, and features and are available in standard designs. For the majority of individuals, they are easy to use, convenient, and effective. However, the pre-assembled bookshelf could not fit precisely or satisfy your demands if your space has an odd shape or if you require special shelving for rare volumes.

On the other hand, Custom indicators are like a bookshelf that you make yourself. The exact measurements, number of shelves, and materials can be chosen to create a bookshelf that fits your space and arranges your books the way you choose. Likewise, in trading, Custom indicators give you the precision and flexibility that built-in indicators cannot, allowing you to design solutions that suit your unique strategy.

**1.2. Differences between Expert Advisors and Indicators**

Although indicators and Expert Advisors evaluate market data in somewhat similar ways, their functions are quite distinct. Because EAs are made to automate trading, in addition to analyzing the market, they also carry out deals in accordance with preset rules. They are capable of managing risk, modifying stop-loss, determining take-profit levels, and opening and closing orders. Every time there is a new price update, OnTick(), one of the most important functions in an EA, is called. With the help of this feature, the EA may keep an eye on price movements in real time and check if trading criteria are satisfied before placing an order.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//

  }
```

In contrast, indicators do not have the power to execute trades; instead, they only concentrate on market analysis. Indicators use the OnCalculate() method, which analyzes historical and current data to produce trading indications, rather than OnTick(). Calculating indicator values, updating chart elements, and presenting visual signals like trend lines or color-coded candles are all handled by OnCalculate().

Indicators merely offer insightful information; traders are left to make the ultimate decision, unlike EAs, which are made to act. Although they both examine market data, indicators and EAs serve different purposes. While indicators assist traders in manually interpreting the market and making well-informed trading decisions, EAs act on trading opportunities.

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
//
  }
```

**Analogy**

Consider indicators and Expert Advisors as two distinct trading tools. An EA analyzes the market, makes choices, and executes trades autonomously, much like a self-driving car. Without human input, it applies trading techniques, continuously monitors market fluctuations, and controls risk.

An indicator, on the other hand, functions more like a GPS in that it offers direction but accomplishes nothing. It analyzes market data, identifies patterns, and aids traders in comprehending changes in price. A moving average, for instance, can indicate whether the market is heading upward or downward, but it cannot initiate or terminate trades. An indicator only offers information; the final choice is up to you, but an EA trades for you.

### **2\. Setting up the Project**

Now that we know what custom indicators are and how they are different from Expert Advisors and built-in indicators, let's look at how to make and modify our indicators in MQL5. We will explore the procedure in detail in this section to make sure you have a good idea of how to create an indicator from the start. Using no pre-existing indicator routines, such as MA, we will construct two unique indicators from the scratch in this tutorial.

**2.1. Line MA**

The first indicator will be a simple moving average indicator that calculates and displays three different moving averages:

- Period 200 applied to the High price.
- Period 100 applied to the Close price.
- Period 50 applied to the Open price.

To build a custom indicator, it is essential to first understand the logic behind its functionality. Before writing any code, we must define how the indicator will process market data and display relevant information. This step ensures that our approach is well-structured and easy to implement. Drafting a pseudocode is an essential step in addition to comprehending its logic. Making a pseudocode facilitates the division of the implementation into more manageable, smaller parts. It reduces the possibility of errors and increases the efficiency of the coding process by clearly outlining what must be done.

The logic needed to determine and show three moving averages utilizing various price sources will be examined first in this section. Before writing the actual code, we will first draft a pseudocode to serve as a guide for our implementation. This structured approach will help us build a well-defined custom indicator while reinforcing key programming concepts in MQL5.

**Pseudocode:**

// SETUP INDICATOR

1\. Set the indicator to display on the chart window.

2\. Define 3 buffers to store data for the moving averages (MA200, MA100, MA50).

3\. Define 3 plots to display the moving averages on the chart.

// SETUP PLOT PROPERTIES

4\. For MA200:

- Label: "MA 200"
- Type: Line
- Style: Dash-Dot-Dot
- Width: 1
- Color: Blue

5\. For MA100:

- Label: "MA 100"
- Type: Line
- Style: Dash
- Width: 1
- Color: Brown

6\. For MA50:

- Label: "MA 50"
- Type: Line
- Style: Dot
- Width: 1
- Color: Purple

// DEFINE BUFFERS

7\. Create buffers to store the calculated values for:

- MA200
- MA100
- MA50

// DEFINE INPUT PARAMETERS

8\. Allow the user to set the period for each moving average:

- MA200: Default period = 200
- MA100: Default period = 100
- MA50: Default period = 50

// INITIALIZATION FUNCTION

9\. When the indicator is initialized:

- Assign each buffer to its corresponding plot.
- Set the number of initial bars to skip for each moving average:
- MA200: Skip the first 200 bars
- MA100: Skip the first 100 bars
- MA50: Skip the first 50 bars

// CALCULATION FUNCTION

10\. For each new candle or when the chart is updated:

The high, close, and open prices of the previous 200, 100, and 50 bars will be added up for each bar, divided by the corresponding time, and the results will be stored in the appropriate buffers to determine the MA200, MA100, and MA50.

**2.2. Candle MA**

Next, we will create a different moving average indicator that represents price trends in candle format. This indicator will use a moving average with a period of 5 and will visually represent its values using candles instead of lines. By displaying moving average data in this way, we can highlight short-term trends and filter out market noise more effectively. This will provide a unique perspective on price movement while maintaining the principles of moving average calculations.

Every project will be thoroughly explained, making sure that every line of code is simple enough for beginners to understand. You will have practical experience building and modifying indicators from scratch by the end of this part, giving you useful abilities for further projects.

**Pseudocode:**

// SETUP INDICATOR

1\. Set the indicator to display in a separate window.

2\. Create 5 arrays (buffers) to store:

- open prices (OpenBuffer)
- high prices (HighBuffer)
- low prices (LowBuffer)
- close prices (CloseBuffer)
- Candle colors (ColorBuffer: 0 = green, 1 = red)

3\. Define 1 plot to display the candles.

// SETUP PLOT PROPERTIES

4\. For the plot:

- Label: "Candles"
- Type: Color Candles (DRAW\_COLOR\_CANDLES)
- Colors: Green for bullish, Red for bearish
- Style: Solid
- Width: 1

// DEFINE INPUT PARAMETERS

5\. Allow the user to set the smoothing period (default = 5).

// INITIALIZATION FUNCTION

6\. When the indicator is initialized:

- Assign each buffer to its corresponding data or color index.
- Set the number of initial bars to skip (equal to the smoothing period).

// CALCULATION FUNCTION

7\. For each new candle or when the chart is updated:

- Check if there are enough bars to calculate (at least "period" bars).
- If not, print an error message and stop.

8\. For each bar from the "period-1" bar to the latest bar:

Calculate the smoothed open, high, low, and close prices:

- Sum the open, high, low, and close prices over the last "period" bars.
- Divide each sum by "period" to get the average.
- Store the results in OpenBuffer, HighBuffer, LowBuffer, and CloseBuffer.

Set the candle color:

- If the smoothed close price >= smoothed open price, set the color to green (0).
- Otherwise, set the color to red (1).

### **3\. Creating and Customizing Custom Indicators**

We can now create and modify our custom indicators, since we have written a pseudocode. With a well-defined strategy in place, we will begin putting the logic into practice gradually, making sure the indicator performs as intended. We may efficiently visualize market movements and modify the indicator to suit our needs by properly organizing our code.

**3.1. Building a Moving Average Indicator in Line Format**

The first step in creating an indicator is to picture how you want it to look and work. It is crucial to ask yourself a few crucial questions to help you understand the design and implementation before creating any code:

- Should the indicator be displayed on the main chart window or in a separate window?
- How many buffers will the indicator require storing and display data?
- What type of plot should be used — line, histogram, candles, or something else?
- What colors should be assigned to different elements for better visibility?
- Will the indicator use multiple data sources, such as high, low, open, or close prices?

By answering these questions, you can create a clear blueprint for your indicator, making the development process smoother and more structured.

**Example:**

```
//INDICATOR IN CHART WINDOW
#property indicator_chart_window

//SET INDICATOR BUFFER TO STORE DATA
#property indicator_buffers 3

//SET NUMBER FOR INDICATOR PLOTS
#property indicator_plots   3
```

**Explanation:**

**Indicator in Chart Window**

```
#property indicator_chart_window
```

Instead of displaying the custom indicator in a separate window, this directive instructs MetaTrader 5 to display it immediately on the main chart, which displays the price action. To plot the indicator in a different window (such as the RSI or MACD), we would use:

```
#property indicator_separate_window
```

This directive ensures that the indicator is displayed in a separate window below the main chart, rather than being plotted directly on the price chart.

**Setting Indicator Buffers to Store Data**

```
#property indicator_buffers 3
```

Calculated values that will be plotted on the chart are stored in arrays called indicator buffers. A distinct element of the indicator is represented by each buffer. In this instance, we will store three distinct sets of values, since we construct three indicator buffers. Three buffers are required to hold the values for each moving average independently because we are building a moving average indicator with three distinct periods (50, 100, and 200).

**Setting the Number of Indicator Plots**

```
#property indicator_plots   3
```

This specifies how many plots the indicator will show. A distinct graphical representation on the chart is called a plot. Three plots are required since we are drawing three moving averages: one for each of the 50-, 100-, and 200-period moving averages.

Color, style, and format (such as line, histogram, or candles) can be customized for each plot. Here, each of the three plots will be shown as a line that represents a distinct moving average.

Summary

- The indicator will be displayed on the main chart (#property indicator\_chart\_window).
- It will store three sets of values using indicator buffers (#property indicator\_buffers 3).
- It will plot three different moving averages (#property indicator\_plots 3).

This setup ensures that the indicator correctly calculates and displays the three moving averages on the price chart.

![Figure 3. Plots and Window Chart](https://c.mql5.com/2/116/figure_3.png)

Setting plot properties comes next after defining the fundamental characteristics of our indicator. To specify how the indicator will appear on the chart, it is essential to provide plot attributes when constructing a custom indicator in MQL5. The type of drawing, line style, thickness, color, and labels are all determined by these parameters. Understanding these parameters will assist you in creating and modifying indicators for various use cases, since this article focuses on teaching how to design custom indicators using a project-based approach.

**Examples:**

```
//INDICATOR IN CHART WINDOW
#property indicator_chart_window

//SET INDICATOR BUFFER TO STORE DATA
#property indicator_buffers 3

//SET NUMBER FOR INDICATOR PLOTS
#property indicator_plots   3

//SETTING PLOTS PROPERTIES
//PROPERTIES OF THE FIRST MA (MA200)
#property indicator_label1  "MA 200"    //GIVE PLOT ONE A NAME
#property indicator_type1   DRAW_LINE  //TYPE OF PLOT THE FIRST MA
#property indicator_style1  STYLE_DASHDOTDOT  //STYLE OF SPOT THE FIRST MA
#property indicator_width1  1         //LINE THICKNESS THE FIRST MA
#property indicator_color1  clrBlue    //LINE COLOR THE FIRST MA

//PROPERTIES OF THE SECOND MA (MA100)
#property indicator_label2  "MA 100"   //GIVE PLOT TWO A NAME
#property indicator_type2   DRAW_LINE  //TYPE OF PLOT THE SECOND MA
#property indicator_style2  STYLE_DASH  //STYLE OF SPOT THE SECOND MA
#property indicator_width2  1          //LINE THICKNESS THE SECOND MA
#property indicator_color2  clrBrown    //LINE COLOR THE SECOND MA

//PROPERTIES OF THE THIRD MA (MA50)
#property indicator_label3  "MA 50"    //GIVE PLOT TWO A NAME
#property indicator_type3   DRAW_LINE  //TYPE OF PLOT THE THIRD MA
#property indicator_style3  STYLE_DOT  //STYLE OF SPOT THE THIRD MA
#property indicator_width3  1          //LINE THICKNESS THE THIRD MA
#property indicator_color3  clrPurple    //LINE COLOR THE THIRD MA
```

**Explanation:**

**First-Moving Average (MA 200)**

```
#property indicator_label1  "MA 200"    //GIVE PLOT ONE A NAME
#property indicator_type1   DRAW_LINE  //TYPE OF PLOT THE FIRST MA
#property indicator_style1  STYLE_DASHDOTDOT  //STYLE OF SPOT THE FIRST MA
#property indicator_width1  1         //LINE THICKNESS THE FIRST MA
#property indicator_color1  clrBlue    //LINE COLOR THE FIRST MA
```

The first-Moving Average (MA200) will appear as a blue dashed dot line with a thickness of 1.

**Second Moving Average (MA 100)**

```
#property indicator_label2  "MA 100"   //GIVE PLOT TWO A NAME
#property indicator_type2   DRAW_LINE  //TYPE OF PLOT THE SECOND MA
#property indicator_style2  STYLE_DASH  //STYLE OF SPOT THE SECOND MA
#property indicator_width2  1          //LINE THICKNESS THE SECOND MA
#property indicator_color2  clrBrown    //LINE COLOR THE SECOND MA
```

The second Moving Average (MA100) will be displayed as a brown dashed line with a thickness of 1.

**Third Moving Average (MA 50)**

```
#property indicator_label3  "MA 50"    //GIVE PLOT TWO A NAME
#property indicator_type3   DRAW_LINE  //TYPE OF PLOT THE THIRD MA
#property indicator_style3  STYLE_DOT  //STYLE OF SPOT THE THIRD MA
#property indicator_width3  1          //LINE THICKNESS THE THIRD MA
#property indicator_color3  clrPurple    //LINE COLOR THE THIRD MA
```

The third Moving Average (MA50) will be displayed as a purple dotted line with a thickness of 1.

**Understanding the Relationship between Plot Properties in MQL5**

Every plot (line, histogram, arrow, etc.) requires a set of characteristics to govern how it looks when determining how an indicator should be displayed on a chart. Numbered suffixes, such as indicator\_label1, indicator\_type1, indicator\_style1, etc., are used in MQL5 to assign the properties.

All the properties for a certain plot are guaranteed to belong together thanks to this numbering. Let's explore it:

- indicator\_label1 assigns a name to the first plot, making it easier to identify.
- indicator\_type1 specifies how this first plot will be drawn (e.g., as a line).
- indicator\_style1 determines the line style for this first plot (solid, dashed, dotted, etc.).
- indicator\_width1 controls the thickness of the first plot.
- indicator\_color1 sets the color for this first plot.

**Example to Clarify the Relationship**

Think of these properties as a matching set for each moving average:

| Property | **What It Controls** | **First MA (200)** | Second MA (100) | **Third MA (50)** |
| --- | --- | --- | --- | --- |
| indicator\_labelX | Name of the MA | "MA 200" | "MA 100" | "MA 50" |
| indicator\_typeX | Plot Type | DRAW\_LINE | DRAW\_LINE | DRAW\_LINE |
| indicator\_styleX | Line Style | STYLE\_DASHDOTDOT | STYLE\_DASH | STYLE\_DOT |
| indicator\_widthX | Thickness | 1 | 1 | 1 |
| indicator\_colorX | Color | clrBlue | clrBrown | clrPurple |

This approach guarantees that every property matches the appropriate moving average and enables us to specify many plots in an organized manner. Therefore, only the Second MA will be impacted if indicator\_style2 is changed. Only the Third MA's color will change if indicator\_color3 is changed. We can manage each moving average's appearance on the chart without creating confusion by maintaining these characteristics' consistency and organization.

In addition to the many plot types, line styles, and colors we employed in our Moving Average indicator, MQL5 offers an array of other customization possibilities for indicator plots. Developers can design various indication types using these parameters, ranging from simple lines to more intricate visualizations like bands, arrows, and histograms. We are employing a project-based approach, concentrating on a single indicator while presenting essential ideas that may be used to other customized indicators because there are too many variables to discuss in a single article. It's helpful to know that there are further choices, though.

Generally, different plot styles can be analyzed using the same logic used in this article. But this isn't always the case for all kinds of plots. For instance, whereas DRAW\_LINE only needs one buffer per plot, DRAW\_CANDLES needs four buffers to plot a single candle. You can consult the [MQL5 documentation](https://www.mql5.com/en/docs) for a comprehensive description of the many plot kinds that are available, along with their particular needs.

Declaring double variables to hold the indicator buffers comes after configuring the plot attributes. Because they store the computed values that will be shown on the chart, these buffers are crucial. The SetIndexBuffer() function must be used to link the variables to the indicator buffers after they have been declared. This stage guarantees that the data kept in these buffers may be shown accurately and is appropriately assigned to the corresponding plots.

**Example:**

```
//INDICATOR IN CHART WINDOW
#property indicator_chart_window

//SET INDICATOR BUFFER TO STORE DATA
#property indicator_buffers 3

//SET NUMBER FOR INDICATOR PLOTS
#property indicator_plots   3

//SETTING PLOTS PROPERTIES
//PROPERTIES OF THE FIRST MA (MA200)
#property indicator_label1  "MA 200"   //GIVE PLOT ONE A NAME
#property indicator_type1   DRAW_LINE  //TYPE OF PLOT THE FIRST MA
#property indicator_style1  STYLE_DASHDOTDOT  //STYLE OF SPOT THE FIRST MA
#property indicator_width1  1         //LINE THICKNESS THE FIRST MA
#property indicator_color1  clrBlue    //LINE COLOR THE FIRST MA

//PROPERTIES OF THE SECOND MA (MA100)
#property indicator_label2  "MA 100"   //GIVE PLOT TWO A NAME
#property indicator_type2   DRAW_LINE  //TYPE OF PLOT THE SECOND MA
#property indicator_style2  STYLE_DASH  //STYLE OF SPOT THE SECOND MA
#property indicator_width2  1          //LINE THICKNESS THE SECOND MA
#property indicator_color2  clrBrown    //LINE COLOR THE SECOND MA

//PROPERTIES OF THE THIRD MA (MA50)
#property indicator_label3  "MA 50"    //GIVE PLOT TWO A NAME
#property indicator_type3   DRAW_LINE  //TYPE OF PLOT THE THIRD MA
#property indicator_style3  STYLE_DOT  //STYLE OF SPOT THE THIRD MA
#property indicator_width3  1          //LINE THICKNESS THE THIRD MA
#property indicator_color3  clrPurple    //LINE COLOR THE THIRD MA

//SET MA BUFFER TO STORE DATA
double buffer_mA200[];  //BUFFER FOR THE FIRST MA
double buffer_mA100[];  //BUFFER FOR THE SECOND MA
double buffer_mA50[];   //BUFFER FOR THE THIRD MA

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {

//SETTING BUFFER
   SetIndexBuffer(0, buffer_mA200, INDICATOR_DATA);  //INDEX 0 FOR MA200
   SetIndexBuffer(1, buffer_mA100, INDICATOR_DATA);  //INDEX 1 FOR MA100
   SetIndexBuffer(2, buffer_mA50, INDICATOR_DATA);   //INDEX 1 FOR MA50

//---
   return(INIT_SUCCEEDED);
  }
```

**Explanation:**

```
//SET MA BUFFER TO STORE DATA
double buffer_mA200[];  //BUFFER FOR THE FIRST MA
double buffer_mA100[];  //BUFFER FOR THE SECOND MA
double buffer_mA50[];   //BUFFER FOR THE THIRD MA
```

These arrays (buffer\_mA200, buffer\_mA100, buffer\_mA50) serve as storage for the calculated values of each Moving Average (MA).

**Each buffer corresponds to a specific MA:**

- buffer\_mA200 → Stores values for MA 200
- buffer\_mA100 → Stores values for MA 100
- buffer\_mA50 → Stores values for MA 50

When the indicator is shown, the values kept in these buffers will be plotted on the chart. To store the calculated values of an indicator in MQL5, indicator buffers are necessary. It would be impossible to save and show the findings without buffers.

```
//SETTING BUFFER
SetIndexBuffer(0, buffer_mA200, INDICATOR_DATA);  //INDEX 0 FOR MA200
SetIndexBuffer(1, buffer_mA100, INDICATOR_DATA);  //INDEX 1 FOR MA100
SetIndexBuffer(2, buffer_mA50, INDICATOR_DATA);   //INDEX 2 FOR MA50
```

[SetIndexBuffer()](https://docs.mql4.com/customind/setindexbuffer "https://docs.mql4.com/customind/setindexbuffer") is used to link each buffer to the indicator’s plotting system.

**It takes three parameters:**

- Buffer index → Determines the order in which buffers are used (starting from 0).
- The Buffer variable → The buffer that will store the values for that index.
- Buffer type → INDICATOR\_DATA specifies that the buffer holds indicator values.

**Breakdown of each line:**

- SetIndexBuffer(0, buffer\_mA200, INDICATOR\_DATA); → Assigns buffer\_mA200 to index 0 (First MA).
- SetIndexBuffer(1, buffer\_mA100, INDICATOR\_DATA); → Assigns buffer\_mA100 to index 1 (Second MA).
- SetIndexBuffer(2, buffer\_mA50, INDICATOR\_DATA); → Assigns buffer\_mA50 to index 2 (Third MA).

This guarantees that the MQL5 indicator system correctly recognizes and plots each buffer. The indicator wouldn't work properly without this step.

Having a thorough understanding of the mathematics or reasoning underlying an indicator is crucial to its creation. Only when an indicator accurately evaluates price changes using a sound mathematical or logical basis can it be considered useful. You must comprehend how an indicator functions and what inputs are needed to calculate it efficiently before putting it into practice in MQL5.

For example, let's take the Moving Average:

By reducing price swings, the moving average makes it simpler to spot market patterns without being influenced by transient changes.

To calculate a Moving Average, we need to consider:

**Period**

The number of historical candles that are used to compute the moving average depends on the time frame. For instance, the latest 200 candles are averaged by a 200-period MA. While the MA is more responsive to recent moves when the period is shorter, a longer period produces a smoother line that responds to price changes more slowly.

**Price Type**

MA can be calculated using different price types, such as:

- Close Price → Uses the closing price of each candle.
- Open Price → Uses the opening price of each candle.
- High Price → Uses the highest price of each candle.
- Low Price → Uses the lowest price of each candle.

In this project, we will be working with three different Moving Averages:

- MA 200 (applied to the High price)
- MA 100 (applied to the Close price)
- MA 50 (applied to the Open price)

Understanding these core concepts ensures that we can correctly calculate and visualize the Moving Average without relying on built-in MQL5 functions like iMA().

**Example:**

```
//INDICATOR IN CHART WINDOW
#property indicator_chart_window

//SET INDICATOR BUFFER TO STORE DATA
#property indicator_buffers 3

//SET NUMBER FOR INDICATOR PLOTS
#property indicator_plots   3

//SETTING PLOTS PROPERTIES
//PROPERTIES OF THE FIRST MA (MA200)
#property indicator_label1  "MA 200"   //GIVE PLOT ONE A NAME
#property indicator_type1   DRAW_LINE  //TYPE OF PLOT THE FIRST MA
#property indicator_style1  STYLE_DASHDOTDOT  //STYLE OF SPOT THE FIRST MA
#property indicator_width1  1         //LINE THICKNESS THE FIRST MA
#property indicator_color1  clrBlue    //LINE COLOR THE FIRST MA

//PROPERTIES OF THE SECOND MA (MA100)
#property indicator_label2  "MA 100"   //GIVE PLOT TWO A NAME
#property indicator_type2   DRAW_LINE  //TYPE OF PLOT THE SECOND MA
#property indicator_style2  STYLE_DASH  //STYLE OF SPOT THE SECOND MA
#property indicator_width2  1          //LINE THICKNESS THE SECOND MA
#property indicator_color2  clrBrown    //LINE COLOR THE SECOND MA

//PROPERTIES OF THE THIRD MA (MA50)
#property indicator_label3  "MA 50"    //GIVE PLOT TWO A NAME
#property indicator_type3   DRAW_LINE  //TYPE OF PLOT THE THIRD MA
#property indicator_style3  STYLE_DOT  //STYLE OF SPOT THE THIRD MA
#property indicator_width3  1          //LINE THICKNESS THE THIRD MA
#property indicator_color3  clrPurple    //LINE COLOR THE THIRD MA

//SET MA BUFFER TO STORE DATA
double buffer_mA200[];  //BUFFER FOR THE FIRST MA
double buffer_mA100[];  //BUFFER FOR THE SECOND MA
double buffer_mA50[];   //BUFFER FOR THE THIRD MA

//SET MA PERIOD
input int period_ma200 = 200;  //PERIOD FOR THE FIRST MA
input int period_ma100 = 100;  //PERIOD FOR THE SECOND MA
input int period_ma50  = 50;   //PERIOD FOR THE THIRD MA

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {

//SETTING BUFFER
   SetIndexBuffer(0, buffer_mA200, INDICATOR_DATA);  //INDEX 0 FOR MA200
   SetIndexBuffer(1, buffer_mA100, INDICATOR_DATA);  //INDEX 1 FOR MA100
   SetIndexBuffer(2, buffer_mA50, INDICATOR_DATA);   //INDEX 1 FOR MA50

//SETTING BARS TO START PLOTTING
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, period_ma200);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, period_ma100);
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, period_ma50);

//---
   return(INIT_SUCCEEDED);
  }
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

//CALCULATE THE MOVING AVERAGE FOR THE THE First MA (MA200)
   for(int i = period_ma200 - 1; i < rates_total; i++)
     {

      double sum = 0.0;
      for(int j = 0; j < period_ma200; j++)
        {
         sum += high[i - j];
        }

      buffer_mA200[i] = sum / period_ma200;

     }

//CALCULATE THE MOVING AVERAGE FOR THE THE SECOND MA (MA100)
   for(int i = period_ma100 - 1; i < rates_total; i++)
     {

      double sum = 0.0;
      for(int j = 0; j < period_ma100; j++)
        {
         sum += close[i - j];
        }
      buffer_mA100[i] = sum / period_ma100;

     }

//CALCULATE THE MOVING AVERAGE FOR THE THE THIRD MA (MA50)
   for(int i = period_ma50 - 1; i < rates_total; i++)
     {
      double sum = 0.0;
      for(int j = 0; j < period_ma50; j++)
        {
         sum += open[i - j];
        }
      buffer_mA50[i] = sum / period_ma50;
     }

//--- return value of prev_calculated for next call
   return(rates_total);
  }
```

**Explanation:**

```
//SET MA PERIOD
input int period_ma200 = 200;  //PERIOD FOR THE FIRST MA
input int period_ma100 = 100;  //PERIOD FOR THE SECOND MA
input int period_ma50  = 50;   //PERIOD FOR THE THIRD MA
```

- These input variables define the period of each Moving Average (MA).
- The period determines how many past candles the MA will use for its calculation.
- period\_ma200 = 200 means the MA200 will use the last 200 candles to calculate the average price.
- period\_ma100 = 100 means the MA100 will use the last 100 candles to calculate the average.
- period\_ma50 = 50 means the MA50 will use the last 50 candles to calculate the average.
- Since these are input variables, traders can modify them in the indicator settings without changing the code.

```
//SETTING BARS TO START PLOTTING
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, period_ma200);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, period_ma100);
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, period_ma50);
```

**Why is this necessary?**

- Since a Moving Average with period 200 needs 200 previous candles to compute its value, it cannot plot anything before the first 200 candles.
- PlotIndexSetInteger(0, PLOT\_DRAW\_BEGIN, period\_ma200); tells MetaTrader to start drawing MA200 only after 200 candles are available.
- PlotIndexSetInteger(1, PLOT\_DRAW\_BEGIN, period\_ma100); does the same for MA100 (starts after 100 candles).
- PlotIndexSetInteger(2, PLOT\_DRAW\_BEGIN, period\_ma50); ensures that MA50 starts after 50 candles.

![Figure 4. Plot Begin](https://c.mql5.com/2/116/figure_4.png)

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
```

Every custom indicator's central function, OnCalculate(), is automatically called whenever a new tick is received or historical data is updated. It is crucial for indicator calculations since it processes price data and modifies the indicator buffers. OnCalculate() is made especially for indicators and offers direct access to historical price data without the need for other procedures like CopyOpen(), CopyClose(), or ArraySetAsSeries(), in contrast to OnTick(), which is utilized in Expert Advisors.

**Difference Between OnCalculate() and OnTick()**

| Feature | OnCalculate() | OnTick() |
| --- | --- | --- |
| Used in | Indicators | Expert Advisors |
| Called when | New tick arrives or chart updates | New tick arrives |
| Access to price data | Uses built-in arrays (open\[\], close\[\], etc.) | Requires functions like CopyClose() to fetch data |
| Purpose | Calculates and updates indicator values | Executes trading logic, places orders |

**Understanding the Parameters in OnCalculate()**

| Parameter | Description |
| --- | --- |
| rates\_total | The total number of available candles (bars) on the chart. |
| prev\_calculated | The number of previously calculated bars (helps optimize performance). |
| time\[\] | The timestamp of each candle (e.g., 2024.02.02 12:30). |
| open\[\] | The opening price of each candle. |
| high\[\] | The highest price of each candle |
| low\[\] | The lowest price of each candle. |
| close\[\] | The closing price of each candle. |
| tick\_volume\[\] | The number of ticks (price updates) within a candle. |
| volume\[\] | The total volume traded within a candle. |
| spread\[\] | The spread (difference between bid and ask price) for each candle. |

```
// CALCULATE THE MOVING AVERAGE FOR THE FIRST MA (MA200)
for(int i = period_ma200 - 1; i < rates_total; i++)
  {
   double sum = 0.0;
   for(int j = 0; j < period_ma200; j++)
     {
      sum += high[i - j];  // SUM OF HIGH PRICES OVER THE PERIOD
     }
   buffer_mA200[i] = sum / period_ma200;  // CALCULATE THE AVERAGE
  }
```

A for loop that iterates through the price data is used to calculate the Moving Average (MA) for the first MA (MA200). We have enough previous data points to calculate the average because the loop begins at period\_ma200 - 1. It continues until rates\_total, which is the total amount of price data points that are provided. This method avoids mistakes that could happen if we try to access nonexistent data points (such negative index values). Every bar in the chart has its moving average systematically determined by the loop, which then updates the indicator buffer appropriately.

To hold the cumulative sum of the high prices over the given time period (in this case, 200), we initialize a variable sum inside the loop. A nested for loop iterates across the final 200 bars from 0 to period\_ma200. Sum += high\[i - j\]; is used in each iteration to add the high price of a certain bar to the sum variable. By advancing backward from the current index i, the formula i - j guarantees that we are adding together the high prices of the last 200 bars. This procedure efficiently adds up all the high prices for the specified time frame.

Once we have the total of the high prices across the 200 periods, we divide that amount by the length of the period to get the average: buffer\_mA200\[i\] = sum / period\_ma200; The computed MA value for each bar is subsequently represented by this final value, which is then saved in the appropriate index of buffer\_mA200. By taking the average of the last 200 high prices, the moving average evens out price swings and enables traders to spot long-term patterns. Although the other moving averages employ different price types (close and open) and periods (100 and 50), the same reasoning holds true for them.

![Figure 5. Line MA](https://c.mql5.com/2/116/Figure_5.png)

**3.2. Building a Moving Average Indicator in Candle Format**

In this section, we will develop a different type of moving average indicator that uses candle format to show price movements. This indicator will employ a five-period moving average and use candles rather than lines to graphically show its values. We can more successfully filter out market noise and highlight short-term trends by presenting moving average data in this manner. This will preserve the fundamentals of moving average calculations while offering a distinctive viewpoint on price movement.

The open price, closing price, high price, and low price are the four essential price components that make up a candlestick. The candle format necessitates independent moving average computations for these four values, in contrast to the line style, which only requires one price value per period (such as the close or open). Accordingly, we require a minimum of four buffers for a single plot, one for each of the moving average values that represent the open, close, high, and low prices. This indicator's structure guarantees that the moving average data is presented in a manner that closely mimics actual candlestick patterns, which facilitates price trend analysis using well-known visual cues.

**Example:**

```
#property indicator_separate_window
#property indicator_buffers 5
#property indicator_plots   1

//---- plot ColorCandles
#property indicator_label1  "Candles"
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  clrGreen, clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

//--- indicator buffers
double OpenBuffer[];
double HighBuffer[];
double LowBuffer[];
double CloseBuffer[];
double ColorBuffer[];

int period = 5;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0, OpenBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, HighBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, LowBuffer, INDICATOR_DATA);
   SetIndexBuffer(3, CloseBuffer, INDICATOR_DATA);
   SetIndexBuffer(4, ColorBuffer, INDICATOR_COLOR_INDEX);

   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, period);

   return INIT_SUCCEEDED;
  }

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

   if(rates_total < period)
     {
      Print("Error: Not enough bars to calculate.");
      return 0;
     }

   for(int i = period - 1; i < rates_total; i++)
     {

      if(i - period + 1 < 0)
         continue;

      double sumClose = 0.0;
      double sumOpen = 0.0;
      double sumHigh = 0.0;
      double sumLow = 0.0;

      for(int j = 0; j < period; j++)
        {
         int index = i - j;
         if(index < 0)
            continue; // Prevent out-of-bounds access

         sumClose += close[index];
         sumOpen += open[index];
         sumHigh += high[index];
         sumLow += low[index];
        }

      if(period > 0)
        {
         OpenBuffer[i] = sumOpen / period;
         HighBuffer[i] = sumHigh / period;
         LowBuffer[i] = sumLow / period;
         CloseBuffer[i] = sumClose / period;
        }
      else
        {
         Print("Error: Division by zero prevented.");
         return 0;
        }

      ColorBuffer[i] = (CloseBuffer[i] >= OpenBuffer[i]) ? 0 : 1;
     }

   return rates_total;
  }
```

**Explanation:**

```
#property indicator_separate_window
#property indicator_buffers 5
#property indicator_plots   1

//---- plot ColorCandles
#property indicator_label1  "Candles"
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  clrGreen, clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

//--- indicator buffers
double OpenBuffer[];
double HighBuffer[];
double LowBuffer[];
double CloseBuffer[];
double ColorBuffer[];

int period = 5;
```

The properties and buffers for a custom indicator that displays a moving average in candlestick format are defined by this code. A candlestick typically needs four buffers: one for the Open, High, Low, and Close price components. However, because an extra buffer (ColorBuffer) is needed to identify and store each candle's color, this indicator needs five buffers.

**Indicator Properties**

- _#property indicator\_separate\_window_ ensures that the indicator is displayed in a separate window rather than being plotted on the main chart.
- _#property indicator\_buffers 5_ defines the number of buffers used. Even though a candle requires only four buffers (Open, High, Low, and Close), a fifth buffer is needed to assign a color to each candle.
- _#property indicator\_plots 1_ specifies that the indicator has only one plot, which will use colored candles.

**Plot and Visualization Settings**

- indicator\_label1 "Candles" names the plot.
- indicator\_type1 DRAW\_COLOR\_CANDLES sets the plot type to colored candlesticks. Unlike DRAW\_LINE, this type requires Open, High, Low, and Close values, plus a color index.
- indicator\_color1 clrGreen, clrRed assigns green to bullish candles and red to bearish candles.
- indicator\_style1 STYLE\_SOLID ensures solid candle edges.
- indicator\_width1 1 defines the thickness of the candle outlines.

**Why Five Buffers Instead of Four?**

A typical candlestick structure requires four buffers:

- OpenBuffer\[\]: Stores the moving average open prices.
- HighBuffer\[\]: Stores the moving average high prices.
- LowBuffer\[\]: Stores the moving average low prices.
- CloseBuffer\[\]: Stores the moving average close prices.

However, since the indicator uses colored candles, it also needs a fifth buffer:

- ColorBuffer\[\]: Stores the color index (0 for green, 1 for red).  By providing an additional buffer, the moving average is graphically represented by bearish (red) and bullish (green) candles, which facilitates the identification of price patterns.

```
//--- indicator buffers mapping
SetIndexBuffer(0, OpenBuffer, INDICATOR_DATA);
SetIndexBuffer(1, HighBuffer, INDICATOR_DATA);
SetIndexBuffer(2, LowBuffer, INDICATOR_DATA);
SetIndexBuffer(3, CloseBuffer, INDICATOR_DATA);
SetIndexBuffer(4, ColorBuffer, INDICATOR_COLOR_INDEX);

PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, period);
```

To save the determined moving average values in candlestick format, this section of code maps the indicator buffers. This candlestick variant of the moving average uses four buffers (OpenBuffer, HighBuffer, LowBuffer, and CloseBuffer) to store the moving average equivalent of open, high, low, and close prices, in contrast to the conventional line-based version, which only needs one buffer to hold calculated values for plotting.

A fifth buffer, called ColorBuffer, is also utilized to identify each candle's color, differentiating between bearish (red) and bullish (green) candles. To guarantee that the indicator processes and displays the data appropriately, the SetIndexBuffer() function associates each buffer with a distinct plot index. To prevent incomplete or deceptive graphics, PlotIndexSetInteger(0, PLOT\_DRAW\_BEGIN, period); makes sure that the indicator only begins plotting after there are sufficient bars available.

```
if(rates_total < period)
  {
   Print("Error: Not enough bars to calculate.");
   return 0;
  }

for(int i = period - 1; i < rates_total; i++)
  {

   if(i - period + 1 < 0)
      continue;

   double sumClose = 0.0;
   double sumOpen = 0.0;
   double sumHigh = 0.0;
   double sumLow = 0.0;

   for(int j = 0; j < period; j++)
     {
      int index = i - j;
      if(index < 0)
         continue; // Prevent out-of-bounds access

      sumClose += close[index];
      sumOpen += open[index];
      sumHigh += high[index];
      sumLow += low[index];
     }

   if(period > 0)
     {
      OpenBuffer[i] = sumOpen / period;
      HighBuffer[i] = sumHigh / period;
      LowBuffer[i] = sumLow / period;
      CloseBuffer[i] = sumClose / period;
     }
   else
     {
      Print("Error: Division by zero prevented.");
      return 0;
     }

   ColorBuffer[i] = (CloseBuffer[i] >= OpenBuffer[i]) ? 0 : 1;
  }

return rates_total;
```

This code segment starts by determining if there are sufficient price bars for the computation to continue. There are enough historical data points (or "bars") to compute the moving average based on the user-defined period thanks to the if (rates\_total < period) condition. The function returns 0 and writes an error message if there are insufficient bars. This step is essential since a meaningful moving average calculation cannot be performed without sufficient data, and trying to do so may lead to inaccurate or misleading statistics.

The for loop iterates over the price data, starting at the period - 1 index and going up to rates\_total, after it has been verified that there are enough bars. Each bar is processed by the loop, which begins at the designated time and proceeds. In this loop, the current bar index is represented by the variable i. Using the price information from the bars from the prior period, this loop determines the moving average for every bar. The loop ensures that the calculation only begins when the necessary number of bars are available by continuing to the next iteration if the index i - period + 1 is less than 0.

Four variables are initialized to 0.0 inside the loop: sumClose, sumOpen, sumHigh, and sumLow. The cumulative sums of the corresponding price data for the specified time period will be stored in these variables. The price information from each bar inside the period is gathered using a second for loop, this time iterating from 0 to period -1. By deducting j from the current index i, the inner loop retrieves each of the previous bars and computes the sum of the close, open, high, and low prices during the specified period. The loop avoids faulty data by using if(index < 0) condition to prevent accessing out-of-bounds entries in the arrays.

To avoid division by zero problems, the code determines whether the period is bigger than zero after gathering the price data for the period. The code determines the average of the open, high, low, and close prices over the period and saves the result in the OpenBuffer, HighBuffer, LowBuffer, and CloseBuffer buffers, if the period is valid. The computed values for the candlestick plot are stored in these buffers. Lastly, the color of the candle is set by updating the ColorBuffer. The candle is colored green (value 0) when the close price is larger than or equal to the open price, signifying a bullish trend; if the close price is less than or equal to the open price, the candle is colored red (value 1), signifying a bearish trend.

The value of rates\_total, which indicates to MetaTrader how many bars were successfully processed, is returned at the end of the OnCalculate function. For later calls to the OnCalculate method, this value is crucial because it keeps track of the number of bars that have been processed and guarantees that real-time data updates are handled correctly.

![Figure 6. Candlestick MA](https://c.mql5.com/2/116/figure_6.png)

By working with candle formats, we also laid the foundation for more advanced indicators like Heikin Ashi charts, which will be explored in future articles.

### **Conclusion**

In conclusion, this article introduced basic concepts such as creating custom indicators, using moving averages, and visualizing data in different formats like line and candle styles. We also explored how to use buffers and set up plots to represent data. Moving forward, we’ll focus on more interesting projects in future articles. The best way to learn as a beginner is through a project-based approach, which breaks down learning into manageable steps rather than overwhelming you with unnecessary details at this stage. This method ensures gradual progress and a more in-depth understanding of key concepts.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17096.zip "Download all attachments in the single ZIP archive")

[Project6\_CandleMA.mq5](https://www.mql5.com/en/articles/download/17096/project6_candlema.mq5 "Download Project6_CandleMA.mq5")(10.18 KB)

[Project6\_LineMA.mq5](https://www.mql5.com/en/articles/download/17096/project6_linema.mq5 "Download Project6_LineMA.mq5")(4.57 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Introduction to MQL5 (Part 36): Mastering API and WebRequest Function in MQL5 (X)](https://www.mql5.com/en/articles/20938)
- [Introduction to MQL5 (Part 35): Mastering API and WebRequest Function in MQL5 (IX)](https://www.mql5.com/en/articles/20859)
- [Introduction to MQL5 (Part 34): Mastering API and WebRequest Function in MQL5 (VIII)](https://www.mql5.com/en/articles/20802)
- [Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://www.mql5.com/en/articles/20700)
- [Introduction to MQL5 (Part 32): Mastering API and WebRequest Function in MQL5 (VI)](https://www.mql5.com/en/articles/20591)
- [Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://www.mql5.com/en/articles/20546)
- [Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://www.mql5.com/en/articles/20425)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/481084)**
(2)


![dhermanus](https://c.mql5.com/avatar/avatar_na2.png)

**[dhermanus](https://www.mql5.com/en/users/dhermanus)**
\|
30 May 2025 at 07:12

Thanks for the effort. Isreal. Appreciate it.

I'd like to comment in this part of the series for [customize indicator](https://www.mql5.com/en/articles/5 "Article: Step on New Rails: Custom Indicators in MQL5 ").

You didn't utilize the prev\_calculated. That means for every running tick

your code would recalculate every previous calculated bar again.

![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
30 May 2025 at 09:27

**dhermanus [#](https://www.mql5.com/en/forum/481084#comment_56820240):**

Thanks for the effort. Isreal. Appreciate it.

I'd like to comment in this part of the series for [customize indicator](https://www.mql5.com/en/articles/5 "Article: Step on New Rails: Custom Indicators in MQL5 ").

You didn't utilize the prev\_calculated. That means for every running tick

your code would recalculate every previous calculated bar again.

Hello.

Your comment is highly appreciated, I will look into that.

Thank you.

![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (I)](https://c.mql5.com/2/117/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_IX___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (I)](https://www.mql5.com/en/articles/16539)

This discussion delves into the challenges encountered when working with large codebases. We will explore the best practices for code organization in MQL5 and implement a practical approach to enhance the readability and scalability of our Trading Administrator Panel source code. Additionally, we aim to develop reusable code components that can potentially benefit other developers in their algorithm development. Read on and join the conversation.

![Artificial Bee Hive Algorithm (ABHA): Tests and results](https://c.mql5.com/2/88/Artificial_Bee_Hive_Algorithm_ABHA__Final__LOGO.png)[Artificial Bee Hive Algorithm (ABHA): Tests and results](https://www.mql5.com/en/articles/15486)

In this article, we will continue exploring the Artificial Bee Hive Algorithm (ABHA) by diving into the code and considering the remaining methods. As you might remember, each bee in the model is represented as an individual agent whose behavior depends on internal and external information, as well as motivational state. We will test the algorithm on various functions and summarize the results by presenting them in the rating table.

![Price Action Analysis Toolkit Development (Part 12): External Flow (III) TrendMap](https://c.mql5.com/2/118/Price_Action_Analysis_Toolkit_Development_Part_12___LOGO.png)[Price Action Analysis Toolkit Development (Part 12): External Flow (III) TrendMap](https://www.mql5.com/en/articles/17121)

The flow of the market is determined by the forces between bulls and bears. There are specific levels that the market respects due to the forces acting on them. Fibonacci and VWAP levels are especially powerful in influencing market behavior. Join me in this article as we explore a strategy based on VWAP and Fibonacci levels for signal generation.

![Feature Engineering With Python And MQL5 (Part III): Angle Of Price (2) Polar Coordinates](https://c.mql5.com/2/117/Feature_Engineering_With_Python_And_MQL5_Part_III_Angle_Of_Price_2__LOGO.png)[Feature Engineering With Python And MQL5 (Part III): Angle Of Price (2) Polar Coordinates](https://www.mql5.com/en/articles/17085)

In this article, we take our second attempt to convert the changes in price levels on any market, into a corresponding change in angle. This time around, we selected a more mathematically sophisticated approach than we selected in our first attempt, and the results we obtained suggest that our change in approach may have been the right decision. Join us today, as we discuss how we can use Polar coordinates to calculate the angle formed by changes in price levels, in a meaningful way, regardless of which market you are analyzing.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/17096&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049147842536252883)

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