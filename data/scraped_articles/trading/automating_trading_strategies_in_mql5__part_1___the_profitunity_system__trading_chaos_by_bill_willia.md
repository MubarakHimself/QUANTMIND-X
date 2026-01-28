---
title: Automating Trading Strategies in MQL5 (Part 1): The Profitunity System (Trading Chaos by Bill Williams)
url: https://www.mql5.com/en/articles/16365
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:59:31.598597
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/16365&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068885291730665169)

MetaTrader 5 / Trading


### Introduction

In this article, we explore the [Profitunity System](https://www.mql5.com/go?link=https://profitunity.com/bill-williams/ "https://profitunity.com/bill-williams/"), a trading strategy developed by [Bill Williams](https://en.wikipedia.org/wiki/Bill_Williams_(trader) "https://en.wikipedia.org/wiki/Bill_Williams_(trader)") that aims to profit from market "chaos" by utilizing a set of key indicators and show how to automate it in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). We start with an overview of the strategy and its key principles. Then, we walk through the implementation process in MQL5, focusing on coding the essential indicators and automating entry and exit signals. Next, we test and optimize the system to ensure performance across various market conditions. Finally, we conclude by discussing the potential and effectiveness of the Profitunity System in automated trading. The sections we cover in this article include:

1. Overview of the Profitunity System
2. Strategy Implementation in MQL5
3. Strategy Testing and Optimization
4. Conclusion

By the end of this article, you will have a clear understanding of how to automate the Profitunity System using [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5"), from implementing its key indicators to optimizing its performance. This will equip you with the tools to enhance your trading strategy and leverage market "chaos" for potential trading performances. Let’s dive in.

### Overview of the Profitunity System

The Profitunity System, crafted by Bill Williams, utilizes a set of specialized indicators that allow us to understand and act on the chaotic movements in the market. The strategy combines the power of trend-following and momentum indicators to create a dynamic, highly responsive trading methodology. The system identifies trend reversals and market acceleration, helping us find high-probability trade setups. The key indicators used in the strategy are:

- Fractals
- Alligator
- Awesome Oscillator (AO)
- Accelerator Oscillator (AC)

Each of these indicators works together, providing critical insights into market conditions and offering entry and exit signals. Let us have a deeper look at the individual indicator settings that apply to the strategy.

**Indicator Settings**

Fractal indicator. It identifies points of reversal in the market. The fractals are formed when there are five consecutive bars with the middle bar being the highest or lowest. They signal the potential start of a new trend or a price reversal, which helps in marking local highs or lows, giving an early indication of a possible trend change.As for the settings, the default period for the fractals is 2 or 5. This means it checks for patterns where a single bar is surrounded by two bars on either side that are lower (for fractal downs) or higher (for fractal ups). Here is how it looks like on the chart.

![FRACTALS](https://c.mql5.com/2/102/Screenshot_2024-11-15_190953.png)

Alligator indicator. The indicator is a combination of three smoothed moving averages—known as the Jaw, Teeth, and Lips—that work together to determine the market's trend. The interaction between these lines helps us recognize whether the market is in a trend or consolidating. When the lines start to spread apart, it signals a trend; when they converge, it suggests the market is in a phase of consolidation.

Settings:

- Jaw (Blue Line): 13-period, smoothed by 8 bars
- Teeth (Red Line): 8-period, smoothed by 5 bars
- Lips (Green Line): 5-period, smoothed by 3 bars

The indicator will help us identify trend direction and timing, as well as market entries and exits. Here is its settings on the chart.

![ALLIGATOR](https://c.mql5.com/2/102/Screenshot_2024-11-15_191144.png)

Awesome Oscillator (AO) indicator. This indicator is a momentum indicator that calculates the difference between a 34-period and a 5-period simple moving average of the median price. It helps us to gauge the strength and direction of a trend by plotting the difference between these two moving averages as a histogram. The settings for this indicator are default.

The AO is typically used to identify bullish or bearish momentum in the market and to spot trend shifts. A histogram above the zero line indicates upward momentum, while a histogram below the zero line signals downward momentum.

Accelerator Oscillator (AC) indicator. Derived from the AO indicator, it measures the acceleration of the market’s momentum. It provides insight into whether market momentum is accelerating or decelerating, which is vital for detecting trend changes before they fully develop. The AC oscillates around a zero line, moving in green (positive) or red (negative) zones. Its settings are also the default.

The AC indicator is used in conjunction with the AO indicator to provide confirmation of market strength and momentum shifts, ensuring that the market is moving strongly in one direction before entering a trade. Now let us have a look at the entry and exit conditions as utilized by the system. The AO and AC indicators would resemble the following illustration.

![AO AND AC](https://c.mql5.com/2/102/Screenshot_2024-11-15_191255.png)

**Entry and Exit Conditions**

The system uses a set of specific conditions to enter and exit trades, which are based on the sequential signals from the Fractal, Alligator, Accelerator Oscillator (AC), and Awesome Oscillator (AO) indicators. The signals work together to ensure that trades are initiated only when the market provides strong confirmation of direction, reducing the risk of false signals.

Buy Entry Conditions:

1. Fractal Signal: A Fractal Down signal occurs when the price action forms a series of lower highs, suggesting a potential upward reversal in price.
2. Alligator Line Breakdown: The Alligator’s Blue Line (Jaw) is broken from bottom to top, indicating the start of an uptrend.
3. Accelerator Oscillator (AC) Confirmation: The AC is in the green zone, indicating bullish momentum and supporting the trend's strength.
4. Awesome Oscillator (AO) Confirmation: The AO histogram crosses above the zero line from bottom to top, further confirming the upward momentum.

Buy Condition:

A Buy entry is triggered after the AO indicator histogram crosses above the zero line from below, confirming a rise in bullish momentum. This is an indication that a strong uptrend is developing and is the point we open a market buy position. Below is an example of a buy signal in the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") chart.

![MT5 BUY SIGNAL](https://c.mql5.com/2/101/Screenshot_2024-11-12_201639.png)

Sell Entry Conditions:

1. Fractal Signal: A Fractal Up signal occurs when the price action forms a series of higher lows, suggesting a potential downward reversal in price.
2. Alligator Line Breakdown: The Alligator’s Blue Line (Jaw) is broken from top to bottom, signaling the start of a downtrend.
3. Accelerator Oscillator (AC) Confirmation: The AC is in the red zone, confirming strong bearish momentum and indicating a high probability of continued downside movement.
4. Awesome Oscillator (AO) Confirmation: The AO histogram crosses below the zero line from top to bottom, signaling a bearish trend.

Sell Condition:

A Sell entry is triggered after the AO histogram crosses below the zero line from above, confirming downward momentum. This indicates that the market is likely to continue moving downward, and is the point we open a market sell position.

**Exit or Reversal Conditions:**

1. Alligator Line Reversal: A reversal of the Alligator’s Green Line (Lips) occurs, suggesting the end of the current trend. The Lips reversing direction indicates that the price may now be reversing or consolidating.
2. Accelerator Oscillator (AC) Reversal: The AC crosses from the green zone to the red zone (or vice versa), signaling a potential change in momentum. This is an early indicator that market momentum is shifting and the current trend may be coming to an end.
3. Awesome Oscillator (AO) Reversal: The AO histogram crossing the zero line in the opposite direction further confirms that a trend reversal is likely to occur.

Exit Condition:

Any or all of the above exit conditions can be used, but in our case we will choose to exit the positions when the AO indicator histograms are crossing into the opposite zone, indicating a shift in market momentum.

By utilizing the combination of the aforementioned indicators, the Profitunity System offers a powerful approach for identifying market reversals and strong trending opportunities. In the next section, we will discuss how to implement these entry and exit conditions in MQL5, enabling full automation of this strategy.

### Strategy Implementation in MQL5

After learning all the theories about the Bill William's Profitunity trading strategy, let us then automate the theory and craft an Expert Advisor (EA) in MetaQuotes Language 5 (MQL5) for [MetaTrader 5](https://www.metatrader5.com/en "https://www.metatrader5.com/en").

To create an expert advisor (EA), on your MetaTrader 5 terminal, click the Tools tab and check MetaQuotes Language Editor, or simply press F4 on your keyboard. Alternatively, you can click the IDE (Integrated Development Environment) icon on the tools bar. This will open the [MetaQuotes Language Editor](https://www.mql5.com/en/book/intro/edit_compile_run) environment, which allows the writing of trading robots, technical indicators, scripts, and libraries of functions.

![OPEN METAEDITOR](https://c.mql5.com/2/101/f._IDE__1.png)

Once the MetaEditor is opened, on the tools bar, navigate to the File tab and check New File, or simply press CTRL + N, to create a new document. Alternatively, you can click on the New icon on the tools tab. This will result in a MQL Wizard pop-up.

![CREATE NEW EA ](https://c.mql5.com/2/101/g._NEW_EA_CREATE__1.png)

On the Wizard that pops, check Expert Advisor (template) and click Next.

![MQL WIZARD](https://c.mql5.com/2/101/h._MQL_Wizard__1.png)

On the general properties of the Expert Advisor, under the name section, provide your expert's file name. Note that to specify or create a folder if it doesn't exist, you use the backslash before the name of the EA. For example, here we have "Experts\\" by default. That means that our EA will be created in the Experts folder and we can find it there. The other sections are pretty much straightforward, but you can follow the link at the bottom of the Wizard to know how to precisely undertake the process.

![NEW EA NAME](https://c.mql5.com/2/101/i._NEW_EA_NAME__1.png)

After providing your desired Expert Advisor file name, click on Next, click Next, and then click Finish. After doing all that, we are now ready to code and program our strategy.

First, we start by defining some metadata about the Expert Advisor (EA). This includes the name of the EA, the copyright information, and a link to the MetaQuotes website. We also specify the version of the EA, which is set to "1.00".

```
//+------------------------------------------------------------------+
//|              1. PROFITUNITY (TRADING CHAOS BY BILL WILLIAMS).mq5 |
//|      Copyright 2024, ALLAN MUNENE MUTIIRIA. #@Forex Algo-Trader. |
//|                                     https://forexalgo-trader.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, ALLAN MUNENE MUTIIRIA. #@Forex Algo-Trader"
#property link      "https://forexalgo-trader.com"
#property description "1. PROFITUNITY (TRADING CHAOS BY BILL WILLIAMS)"
#property version   "1.00"
```

This will display the system metadata when loading the program. We can then move on to adding some global variables that we will use within the program. First, we include a trade instance by using [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) at the beginning of the source code. This gives us access to the "CTrade class", which we will use to create a trade object. This is crucial as we need it to open trades.

```
#include <Trade/Trade.mqh>
CTrade obj_Trade;
```

The preprocessor will replace the line #include <Trade/Trade.mqh> with the content of the file Trade.mqh. Angle brackets indicate that the Trade.mqh file will be taken from the standard directory (usually it is terminal\_installation\_directory\\MQL5\\Include). The current directory is not included in the search. The line can be placed anywhere in the program, but usually, all inclusions are placed at the beginning of the source code, for a better code structure and easier reference. Declaration of the obj\_Trade object of the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class will give us access to the methods contained in that class easily, thanks to the MQL5 developers.

![CTRADE CLASS](https://c.mql5.com/2/101/j._INCLUDE_CTRADE_CLASS.png)

After that, we need to declare several important indicator handles that we will use in the trading system.

```
int handle_Fractals = INVALID_HANDLE; //--- Initialize fractals indicator handle with an invalid handle value
int handle_Alligator = INVALID_HANDLE; //--- Initialize alligator indicator handle with an invalid handle value
int handle_AO = INVALID_HANDLE; //--- Initialize Awesome Oscillator (AO) handle with an invalid handle value
int handle_AC = INVALID_HANDLE; //--- Initialize Accelerator Oscillator (AC) handle with an invalid handle value
```

Here, we set up initial variables to hold the handles for each technical indicator in the program. Specifically, we initialize four [integer](https://www.mql5.com/en/docs/basis/types/integer) variables—"handle\_Fractals", "handle\_Alligator", "handle\_AO", and "handle\_AC"—with the [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) value.

Each of these handles will act as a reference to access the respective indicators throughout the code. By assigning the initial "INVALID\_HANDLE" value, we ensure that each handle variable clearly shows an invalid state until proper initialization takes place later in the code. This setup prevents errors from using an uninitialized handle and helps detect if any indicator fails to load during the initialization process.

In summary, this is what the indicators will do individually:

- "handle\_Fractals" will store the handle for the Fractals indicator.
- "handle\_Alligator" will store the handle for the Alligator indicator.
- "handle\_AO" will store the handle for the Awesome Oscillator.
- "handle\_AC" will store the handle for the Accelerator Oscillator.

Next, we need to define and initialize arrays and constants required to store and process data from the indicators used in this Expert Advisor, which we specifically retrieve from initialized indicator handles. We will do this sequentially to keep everything straight and simple for referencing.

```
double fractals_up[]; //--- Array to store values for upward fractals
double fractals_down[]; //--- Array to store values for downward fractals

double alligator_jaws[]; //--- Array to store values for Alligator's Jaw line
double alligator_teeth[]; //--- Array to store values for Alligator's Teeth line
double alligator_lips[]; //--- Array to store values for Alligator's Lips line

double ao_values[]; //--- Array to store values of the Awesome Oscillator (AO)

double ac_color[]; //--- Array to store color status of the Accelerator Oscillator (AC)
#define AC_COLOR_UP 0 //--- Define constant for upward AC color state
#define AC_COLOR_DOWN 1 //--- Define constant for downward AC color state
```

We begin by creating two [arrays](https://www.mql5.com/en/docs/basis/variables#array_define), "fractals\_up" and "fractals\_down", which will store values for upward and downward fractals, respectively. These arrays will allow us to keep track of specific fractal points, helping us identify significant price reversals or patterns.

Next, we set up three arrays—"alligator\_jaws", "alligator\_teeth", and "alligator\_lips"—to store the values of the Alligator indicator's different lines. By keeping these values in separate arrays, we can efficiently track the state of each Alligator line and use them in cross-referencing for trading signals.

We then define the "ao\_values" array to store the values of the Awesome Oscillator (AO). The AO will help us identify market momentum and trends, and storing these values will allow us to analyze changes over time and apply them to our trading conditions.

Lastly, we define the "ac\_color" array to capture the color status of the Accelerator Oscillator (AC). This array will hold information about the AC's upward or downward movement, which we will store as a color state. To facilitate this, we define two constants: "AC\_COLOR\_UP" (set to 0) and "AC\_COLOR\_DOWN" (set to 1). These constants will represent the AC's color states, with green (upward) indicating momentum growth and red (downward) indicating a slowing trend. This setup will simplify our logic when we later check the AC status for trading signals.

You may have noticed that we can easily store the values of other indicators directly other than the fractals. This is because their values are readily available on every bar. However, as for the fractals, they form on specific swing points that are at least 3 bars away from the current bar. Thus, we can't just retrieve any of the fractal bars as they form conditionally. Thus, we need logic to keep track of the previous fractal value as well as its direction. Here is the logic we adopt.

```
double lastFractal_value = 0.0; //--- Variable to store the value of the last detected fractal
enum fractal_direction {FRACTAL_UP, FRACTAL_DOWN, FRACTAL_NEUTRAL}; //--- Enum for fractal direction states
fractal_direction lastFractal_direction = FRACTAL_NEUTRAL; //--- Variable to store the direction of the last fractal
```

Here, we define variables and an [enumeration](https://www.mql5.com/en/book/basis/builtin_types/enums) to store and manage the data of the most recent fractal detected in our trading analysis. We start by declaring the "lastFractal\_value" variable and initialize it to "0.0". This variable will store the numerical value of the last fractal we detect on the chart. By keeping track of this value, we can use it to compare against the current price and analyze fractal formations for potential trade signals.

Next, we define an enumeration called "fractal\_direction" with three possible states: "FRACTAL\_UP", "FRACTAL\_DOWN", and "FRACTAL\_NEUTRAL". These states represent the direction of the last fractal:

- "FRACTAL\_UP" will indicate an upward fractal, signaling potential bearish conditions.
- "FRACTAL\_DOWN" will indicate a downward fractal, signaling potential bullish conditions.
- "FRACTAL\_NEUTRAL" represents a state where no specific fractal direction has been confirmed.

Finally, we declare a variable, "lastFractal\_direction", of type "fractal\_direction" and initialize it to "FRACTAL\_NEUTRAL". This variable will hold the direction of the last fractal detected, allowing us to make directional assessments within the trading logic based on the most recent fractal data.

From that, we can now graduate to the actual code-processing logic. We will need to initialize our indicators and thus we will dive directly into the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, which is called and executed on every instance the program is initialized.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){
   //---

   //---
   return(INIT_SUCCEEDED); //--- Return successful initialization status
}
```

This is simply the default initialization event handler that we will use to initialize our program's control logic. Next, we need to initialize the indicators so that they are attached to the chart and we can use them for data retrieval and trading decisions. We will first initialize the fractals indicator as below.

```
   handle_Fractals = iFractals(_Symbol,_Period); //--- Initialize the fractals indicator handle
   if (handle_Fractals == INVALID_HANDLE){ //--- Check if the fractals indicator failed to initialize
      Print("ERROR: UNABLE TO INITIALIZE THE FRACTALS INDICATOR. REVERTING NOW!"); //--- Print error if fractals initialization failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
```

Here, we initialize the "handle\_Fractals" variable by calling the [iFractals](https://www.mql5.com/en/docs/indicators/ifractals) function. This function creates a handle for the Fractals indicator, applied to the specified symbol (represented by [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol)) and the current chart period ( [\_Period](https://www.mql5.com/en/docs/predefined/_period)). By setting "handle\_Fractals" to the return value of [iFractals](https://www.mql5.com/en/docs/indicators/ifractals), we enable access to the indicator data, which we can later use to analyze fractal formations in our strategy.

After attempting to initialize the fractals indicator, we verify if it succeeded by checking if "handle\_Fractals" equals INVALID\_HANDLE. An "INVALID\_HANDLE" value indicates that the indicator could not be initialized, which could happen for various reasons, such as lack of system resources or incorrect parameters.

If initialization fails, we use the [Print](https://www.mql5.com/en/docs/common/print) function to output an error message, "ERROR: UNABLE TO INITIALIZE THE FRACTALS INDICATOR. REVERTING NOW!", to the journal. This message will serve as a clear notification of the issue, making troubleshooting easier. We then return [INIT\_FAILED](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) to exit the OnInit function, signaling that the initialization process could not be completed successfully. This check helps ensure that the Expert Advisor does not proceed with an incomplete setup, which could lead to errors during execution. We do the same for the Alligator indicator initialization.

```
   handle_Alligator = iAlligator(_Symbol,_Period,13,8,8,5,5,3,MODE_SMMA,PRICE_MEDIAN); //--- Initialize the alligator indicator with specific settings
   if (handle_Alligator == INVALID_HANDLE){ //--- Check if the alligator indicator failed to initialize
      Print("ERROR: UNABLE TO INITIALIZE THE ALLIGATOR INDICATOR. REVERTING NOW!"); //--- Print error if alligator initialization failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
```

We initialize the "handle\_Alligator" variable by calling the [iAlligator](https://www.mql5.com/en/docs/indicators/ialligator) function. The Alligator indicator requires several parameters to define its three lines (Jaws, Teeth, and Lips), each of which responds to market trends. We specify these settings as follows: "13" for the Jaws period, "8" for the Teeth period, and "5" for the Lips period. Additionally, we define shift values of "8", "5", and "3" for each line, set the calculation method to [MODE\_SMMA](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_ma_method) (smoothed moving average), and use [PRICE\_MEDIAN](https://www.mql5.com/en/docs/constants/indicatorconstants/prices#enum_applied_price_enum) as the price type.

After attempting to initialize the Alligator indicator, we check if "handle\_Alligator" has a valid handle. If it equals [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants), it indicates the initialization process failed. This could happen due to insufficient resources or incorrect parameters, preventing the Alligator indicator from functioning correctly.

If initialization fails, we call the Print function to display an error message: "ERROR: UNABLE TO INITIALIZE THE ALLIGATOR INDICATOR. REVERTING NOW!" This message helps alert us to the problem, making it easier to diagnose and resolve. After printing the message, we return [INIT\_FAILED](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants), which exits the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function and indicates that the initialization did not complete successfully.

We use a similar approach to initialize the AO and AC indicators as follows.

```
   handle_AO = iAO(_Symbol,_Period); //--- Initialize the Awesome Oscillator (AO) indicator handle
   if (handle_AO == INVALID_HANDLE){ //--- Check if AO indicator failed to initialize
      Print("ERROR: UNABLE TO INITIALIZE THE AO INDICATOR. REVERTING NOW!"); //--- Print error if AO initialization failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
   handle_AC = iAC(_Symbol,_Period); //--- Initialize the Accelerator Oscillator (AC) indicator handle
   if (handle_AC == INVALID_HANDLE){ //--- Check if AC indicator failed to initialize
      Print("ERROR: UNABLE TO INITIALIZE THE AC INDICATOR. REVERTING NOW!"); //--- Print error if AC initialization failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
```

After all the indicators are initialized successfully, we can then add them to the chart automatically when the program is loaded up. We use the following logic to achieve that.

```
   if (!ChartIndicatorAdd(0,0,handle_Fractals)){ //--- Add the fractals indicator to the main chart window and check for success
      Print("ERROR: UNABLE TO ADD THE FRACTALS INDICATOR TO CHART. REVERTING NOW!"); //--- Print error if fractals addition failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
```

Here, we attempt to add the Fractals indicator to the main chart window using the [ChartIndicatorAdd](https://www.mql5.com/en/docs/chart_operations/chartindicatoradd) function. We pass "0" as the chart ID (indicating the current chart) and specify "0" as the window ID, targeting the main chart window. The "handle\_Fractals" variable, which we previously initialized to store the Fractals indicator handle, is passed in to add this specific indicator.

After calling the ChartIndicatorAdd function, we check if the function call was successful. If it returns "false", represented by "!", this indicates that the Fractals indicator could not be added to the chart. Failure here could occur due to chart limitations or insufficient resources. In this case, we display an error message using Print to alert us: "ERROR: UNABLE TO ADD THE FRACTALS INDICATOR TO CHART. REVERTING NOW!" This message will allow us to quickly identify the source of the issue during debugging.

If the addition fails, we return [INIT\_FAILED](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) to exit the OnInit function with a failure status, ensuring that the Expert Advisor will not run if the Fractals indicator is missing from the chart, helping prevent execution errors later on by confirming the indicator's visual availability. A similar logic is used to add the Alligator indicator since it is also in the main window, as below.

```
   if (!ChartIndicatorAdd(0,0,handle_Alligator)){ //--- Add the alligator indicator to the main chart window and check for success
      Print("ERROR: UNABLE TO ADD THE ALLIGATOR INDICATOR TO CHART. REVERTING NOW!"); //--- Print error if alligator addition failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
```

To add the other indicators, a similar approach is used, only that the subwindows now change since we create a new subwindow for every indicator respectively as follows.

```
   if (!ChartIndicatorAdd(0,1,handle_AO)){ //--- Add the AO indicator to a separate subwindow and check for success
      Print("ERROR: UNABLE TO ADD THE AO INDICATOR TO CHART. REVERTING NOW!"); //--- Print error if AO addition failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
   if (!ChartIndicatorAdd(0,2,handle_AC)){ //--- Add the AC indicator to a separate subwindow and check for success
      Print("ERROR: UNABLE TO ADD THE AC INDICATOR TO CHART. REVERTING NOW!"); //--- Print error if AC addition failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
```

We add the Awesome Oscillator (AO) and Accelerator Oscillator (AC) indicators to separate subwindows on the chart, ensuring they each have their dedicated view. To achieve this, we use the [ChartIndicatorAdd](https://www.mql5.com/en/docs/chart_operations/chartindicatoradd) function for each indicator. We specify "0" for the chart ID (indicating the current chart) and use distinct window IDs: "1" for the AO indicator and "2" for the AC indicator, directing each to appear in a unique subwindow. The numbering here is crucial as every indicator has to be on its window, so you need to keep track of the subwindow indexing properly.

We then check the success of each addition individually. If ChartIndicatorAdd returns "false" for either the AO or AC indicators, it signals that the addition process failed. In case of failure, we output an error message with "Print" to clarify which specific indicator failed to load. For example, if the AO indicator cannot be added, we print "ERROR: UNABLE TO ADD THE AO INDICATOR TO CHART. REVERTING NOW!" Similarly, we output an error message if the AC indicator addition fails.

If either indicator addition fails, we immediately return INIT\_FAILED, exiting the OnInit function and preventing further execution. To ensure everything is okay, we can print the indicator handles to the journal.

```
   Print("HANDLE ID FRACTALS = ",handle_Fractals); //--- Print the handle ID for fractals
   Print("HANDLE ID ALLIGATOR = ",handle_Alligator); //--- Print the handle ID for alligator
   Print("HANDLE ID AO = ",handle_AO); //--- Print the handle ID for AO
   Print("HANDLE ID AC = ",handle_AC); //--- Print the handle ID for AC
```

Upon running the program, we get the following initialization data.

![INDICATOR HANDLES DATA](https://c.mql5.com/2/101/Screenshot_2024-11-13_000334.png)

From the image, we can see that the handle IDs start from 10, sequentially to 13. Handle IDs are critical in MQL5 because they allow us to reference each indicator throughout the Expert Advisor’s lifecycle. When a function like [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) retrieves values from an indicator, it relies on these handles to access the correct data. Here, the integers represent specific identifiers for each initialized indicator. Each ID functions as a unique “pointer” within the MQL5 environment, linking each handle to its designated indicator. This helps the EA know which indicator data to pull from memory, supporting efficient execution and clear organization of indicator-based tasks.

From that, all that we now need to do is set the data holders as time series.

```
   ArraySetAsSeries(fractals_up,true); //--- Set the fractals_up array as a time series
   ArraySetAsSeries(fractals_down,true); //--- Set the fractals_down array as a time series

   ArraySetAsSeries(alligator_jaws,true); //--- Set the alligator_jaws array as a time series
   ArraySetAsSeries(alligator_teeth,true); //--- Set the alligator_teeth array as a time series
   ArraySetAsSeries(alligator_lips,true); //--- Set the alligator_lips array as a time series

   ArraySetAsSeries(ao_values,true); //--- Set the ao_values array as a time series

   ArraySetAsSeries(ac_color,true); //--- Set the ac_color array as a time series
```

Here, we set each array to operate as a time series, meaning the data within each array will be organized from the newest to the oldest values. We do this by applying the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) function to each array and passing "true" as the second argument. This setting ensures that the most recent data point always appears at index 0, which is particularly useful in trading applications, where accessing the latest value is essential for real-time decision-making.

We start by setting the "fractals\_up" and "fractals\_down" arrays as time series, enabling us to track the latest upward and downward fractal values efficiently. Similarly, we apply this organization to the "alligator\_jaws", "alligator\_teeth", and "alligator\_lips" arrays, which represent the three lines of the Alligator indicator. This allows us to access the latest values of each line in real-time, making it easier to detect any changes in market trends.

We also configure the "ao\_values" array, which stores the data for the Awesome Oscillator, in the same way. By setting it as a time series, we ensure that the most recent oscillator value is always readily available for our calculations. Finally, we apply this structure to the "ac\_color" array, which tracks the color state of the Accelerator Oscillator, so that the most recent color status can be accessed immediately. The full initialization code snippet responsible for ensuring smooth and clean initialization is as below.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){
   //---

   handle_Fractals = iFractals(_Symbol,_Period); //--- Initialize the fractals indicator handle
   if (handle_Fractals == INVALID_HANDLE){ //--- Check if the fractals indicator failed to initialize
      Print("ERROR: UNABLE TO INITIALIZE THE FRACTALS INDICATOR. REVERTING NOW!"); //--- Print error if fractals initialization failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
   handle_Alligator = iAlligator(_Symbol,_Period,13,8,8,5,5,3,MODE_SMMA,PRICE_MEDIAN); //--- Initialize the alligator indicator with specific settings
   if (handle_Alligator == INVALID_HANDLE){ //--- Check if the alligator indicator failed to initialize
      Print("ERROR: UNABLE TO INITIALIZE THE ALLIGATOR INDICATOR. REVERTING NOW!"); //--- Print error if alligator initialization failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
   handle_AO = iAO(_Symbol,_Period); //--- Initialize the Awesome Oscillator (AO) indicator handle
   if (handle_AO == INVALID_HANDLE){ //--- Check if AO indicator failed to initialize
      Print("ERROR: UNABLE TO INITIALIZE THE AO INDICATOR. REVERTING NOW!"); //--- Print error if AO initialization failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
   handle_AC = iAC(_Symbol,_Period); //--- Initialize the Accelerator Oscillator (AC) indicator handle
   if (handle_AC == INVALID_HANDLE){ //--- Check if AC indicator failed to initialize
      Print("ERROR: UNABLE TO INITIALIZE THE AC INDICATOR. REVERTING NOW!"); //--- Print error if AC initialization failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }

   if (!ChartIndicatorAdd(0,0,handle_Fractals)){ //--- Add the fractals indicator to the main chart window and check for success
      Print("ERROR: UNABLE TO ADD THE FRACTALS INDICATOR TO CHART. REVERTING NOW!"); //--- Print error if fractals addition failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
   if (!ChartIndicatorAdd(0,0,handle_Alligator)){ //--- Add the alligator indicator to the main chart window and check for success
      Print("ERROR: UNABLE TO ADD THE ALLIGATOR INDICATOR TO CHART. REVERTING NOW!"); //--- Print error if alligator addition failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
   if (!ChartIndicatorAdd(0,1,handle_AO)){ //--- Add the AO indicator to a separate subwindow and check for success
      Print("ERROR: UNABLE TO ADD THE AO INDICATOR TO CHART. REVERTING NOW!"); //--- Print error if AO addition failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }
   if (!ChartIndicatorAdd(0,2,handle_AC)){ //--- Add the AC indicator to a separate subwindow and check for success
      Print("ERROR: UNABLE TO ADD THE AC INDICATOR TO CHART. REVERTING NOW!"); //--- Print error if AC addition failed
      return (INIT_FAILED); //--- Exit initialization with failed status
   }

   Print("HANDLE ID FRACTALS = ",handle_Fractals); //--- Print the handle ID for fractals
   Print("HANDLE ID ALLIGATOR = ",handle_Alligator); //--- Print the handle ID for alligator
   Print("HANDLE ID AO = ",handle_AO); //--- Print the handle ID for AO
   Print("HANDLE ID AC = ",handle_AC); //--- Print the handle ID for AC

   ArraySetAsSeries(fractals_up,true); //--- Set the fractals_up array as a time series
   ArraySetAsSeries(fractals_down,true); //--- Set the fractals_down array as a time series

   ArraySetAsSeries(alligator_jaws,true); //--- Set the alligator_jaws array as a time series
   ArraySetAsSeries(alligator_teeth,true); //--- Set the alligator_teeth array as a time series
   ArraySetAsSeries(alligator_lips,true); //--- Set the alligator_lips array as a time series

   ArraySetAsSeries(ao_values,true); //--- Set the ao_values array as a time series

   ArraySetAsSeries(ac_color,true); //--- Set the ac_color array as a time series

   //---
   return(INIT_SUCCEEDED); //--- Return successful initialization status
}
```

Next, we can graduate to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler where we will house the control logic.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
//---
}
```

This is simply the default tick event handler that we will use to base our control logic. Next, we need to retrieve the data values from the indicator handles for further analysis.

```
   if (CopyBuffer(handle_Fractals,0,2,3,fractals_up) < 3){ //--- Copy upward fractals data; check if copying is successful
      Print("ERROR: UNABLE TO COPY THE FRACTALS UP DATA. REVERTING!"); //--- Print error message if failed
      return;
   }
   if (CopyBuffer(handle_Fractals,1,2,3,fractals_down) < 3){ //--- Copy downward fractals data; check if copying is successful
      Print("ERROR: UNABLE TO COPY THE FRACTALS DOWN DATA. REVERTING!"); //--- Print error message if failed
      return;
   }
```

Here, we are copying data from the "fractals\_up" and "fractals\_down" buffers of the "handle\_Fractals" indicator into their respective arrays. We use the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function to retrieve data from the indicator handle. Specifically, we are attempting to copy 3 data points starting from the third most recent bar (index 2) for both the upward and downward fractals.

First, we check if the function returns a value that is less than 3, which would indicate that fewer than 3 values were successfully copied into the "fractals\_up" array. If this happens, we print an error message ("ERROR: UNABLE TO COPY THE FRACTALS UP DATA. REVERTING!") and exit the function to prevent any further processing with incomplete data.

Similarly, we attempt to copy the downward fractal data into the "fractals\_down" array using the same CopyBuffer function. Again, if the copying fails (less than 3 values are returned), we print a corresponding error message ("ERROR: UNABLE TO COPY THE FRACTALS DOWN DATA. REVERTING!") and exit the function to avoid further issues. This approach ensures that our program does not proceed with invalid or incomplete data, thus maintaining the integrity of our trading logic. By verifying that the correct number of values has been copied, we can prevent potential errors in the analysis of fractals, which are crucial for detecting market reversal points.

However, you could have noticed that the indicator buffer numbers vary, 0 and 1. These are crucial indices that you need to be keen on since they represent the actual mapping buffers of the indicator values. Here is an illustration for us to understand why we use the specific indices.

![BUFFER INDICES](https://c.mql5.com/2/101/Screenshot_2024-11-13_003102.png)

From the image, we can see that the fractal up is the first, hence being index 0, and the fractal down is the second at index 1. Via the same logic, we map the alligator lines.

```
   if (CopyBuffer(handle_Alligator,0,0,3,alligator_jaws) < 3){ //--- Copy Alligator's Jaw data
      Print("ERROR: UNABLE TO COPY THE ALLIGATOR JAWS DATA. REVERTING!");
      return;
   }
   if (CopyBuffer(handle_Alligator,1,0,3,alligator_teeth) < 3){ //--- Copy Alligator's Teeth data
      Print("ERROR: UNABLE TO COPY THE ALLIGATOR TEETH DATA. REVERTING!");
      return;
   }
   if (CopyBuffer(handle_Alligator,2,0,3,alligator_lips) < 3){ //--- Copy Alligator's Lips data
      Print("ERROR: UNABLE TO COPY THE ALLIGATOR LIPS DATA. REVERTING!");
      return;
   }
```

Here, since the buffers are in 3 lines, you can see the buffer indices start from 0 through 1 to 2. In the case where there is only one buffer, as for the case of the AO indicator, we would have only a buffer index 0 for the prices as below.

```
   if (CopyBuffer(handle_AO,0,0,3,ao_values) < 3){ //--- Copy AO data
      Print("ERROR: UNABLE TO COPY THE AO DATA. REVERTING!");
      return;
   }
```

The same case would apply to the AC indicator if we would be interested in its values. However, we only need to know the color of the histogram formed. These can be acquired by using the same buffer number logic, but in this case, the color buffers are mostly mapped in the next unavailable buffer index in the data window. Thus, in our case, it is 0+1=1 as follows.

```
   if (CopyBuffer(handle_AC,1,0,3,ac_color) < 3){ //--- Copy AC color data
      Print("ERROR: UNABLE TO COPY THE AC COLOR DATA. REVERTING!");
      return;
   }
```

Actually to get the color buffers you can just open the indicator property window and the colors index will appear on the parameters tab.

![COLOR BUFFER INDEX](https://c.mql5.com/2/101/Screenshot_2024-11-13_004654.png)

After retrieving and storing the data, we can then use it to make trading decisions. To save resources, we will run checks on every bar and not on every tick that is generated. Thus, we need logic to detect new bar formations.

```
if (isNewBar()){ //--- Check if a new bar has formed

//---

}
```

Here, we utilize a custom function whose code snippet is as below.

```
//+------------------------------------------------------------------+
//|   IS NEW BAR FUNCTION                                            |
//+------------------------------------------------------------------+
bool isNewBar(){
   static int prevBars = 0; //--- Store previous bar count
   int currBars = iBars(_Symbol,_Period); //--- Get current bar count for the symbol and period
   if (prevBars == currBars) return (false); //--- If bars haven't changed, return false
   prevBars = currBars; //--- Update previous bar count
   return (true); //--- Return true if new bar is detected
}
```

Here, we define a [boolean](https://www.mql5.com/en/book/basis/builtin_types/booleans) function called "isNewBar" that checks if a new bar has appeared on the chart, which will help us detect when a new candle or bar is formed, which is essential for updating or recalculating trading conditions. We start by declaring a static [integer](https://www.mql5.com/en/docs/basis/types/integer) variable "prevBars" and initializing it to 0. The keyword " [static](https://www.mql5.com/en/docs/basis/variables/static)" ensures that the value of "prevBars" is retained across multiple calls to the function, instead of being reset each time the function is called. This allows us to store the number of bars from the previous function call.

Next, we define a local integer variable "currBars" and use the built-in function [iBars](https://www.mql5.com/en/docs/series/ibars) to retrieve the current number of bars for the selected symbol ( [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol)) and period ( [\_Period](https://www.mql5.com/en/docs/predefined/_period)). This function counts the total number of bars available for the given timeframe and stores it in "currBars".

We then compare "prevBars" with "currBars". If the two values are equal, this means no new bar has been formed since the last time the function was called, so we return "false" to indicate that no new bar is detected. If the number of bars has changed (i.e., a new bar has formed), the condition fails, and we update "prevBars" with the current bar count ("currBars") to keep track of the new value. Finally, we return "true" to signal that a new bar has been detected.

Now within this function, we can determine and store the fractal data by checking and updating the value and direction of the last detected fractal based on the data stored in the "fractals\_up" and "fractals\_down" arrays.

```
const int index_fractal = 0;
if (fractals_up[index_fractal] != EMPTY_VALUE){ //--- Detect upward fractal presence
   lastFractal_value = fractals_up[index_fractal]; //--- Store fractal value
   lastFractal_direction = FRACTAL_UP; //--- Set last fractal direction as up
}
if (fractals_down[index_fractal] != EMPTY_VALUE){ //--- Detect downward fractal presence
   lastFractal_value = fractals_down[index_fractal];
   lastFractal_direction = FRACTAL_DOWN;
}
```

We start by defining a constant integer "index\_fractal" and setting it to 0. This constant will represent the index of the current fractal data that we want to check. In this case, we're checking the first fractal in the arrays.

Next, we check if the value at "fractals\_up\[index\_fractal\]" is not equal to [EMPTY\_VALUE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants), which indicates that there is a valid upward fractal present. If this condition is true, we proceed to store the value of the upward fractal in the "lastFractal\_value" variable. We also updated the "lastFractal\_direction" to "FRACTAL\_UP" to indicate that the last detected fractal was an upward fractal.

Similarly, we check if the value at "fractals\_down\[index\_fractal\]" is not equal to EMPTY\_VALUE, which indicates the presence of a downward fractal. If this condition is true, we store the downward fractal value in the "lastFractal\_value" variable and set "lastFractal\_direction" to "FRACTAL\_DOWN" to reflect that the last fractal detected was downward. We can then log the data acquired and check its validity.

```
if (lastFractal_value != 0.0 && lastFractal_direction != FRACTAL_NEUTRAL){ //--- Ensure fractal is valid
   Print("FRACTAL VALUE = ",lastFractal_value);
   Print("FRACTAL DIRECTION = ",getLastFractalDirection());
}
```

This will log the Fractals data. We used a custom function to get the fractal direction and its code snippet is as below.

```
//+------------------------------------------------------------------+
//|     FUNCTION TO GET FRACTAL DIRECTION                            |
//+------------------------------------------------------------------+

string getLastFractalDirection(){
   string direction_fractal = "NEUTRAL"; //--- Default direction set to NEUTRAL

   if (lastFractal_direction == FRACTAL_UP) return ("UP"); //--- Return UP if last fractal was up
   else if (lastFractal_direction == FRACTAL_DOWN) return ("DOWN"); //--- Return DOWN if last fractal was down

   return (direction_fractal); //--- Return NEUTRAL if no specific direction
}
```

Here, we define the function "getLastFractalDirection" to determine and return the direction of the last detected fractal. The function works by checking the value of the "lastFractal\_direction" variable, which keeps track of the most recent fractal direction (either upward or downward). We begin by initializing a [string](https://www.mql5.com/en/docs/strings) variable "direction\_fractal" and setting it to "NEUTRAL" by default. This means that, if no valid direction is found or the fractal direction has not been updated, the function will return "NEUTRAL" as the result.

Next, we check the value of the "lastFractal\_direction" variable. If it equals "FRACTAL\_UP" (indicating the last detected fractal was an upward fractal), the function returns the string "UP". If "lastFractal\_direction" equals "FRACTAL\_DOWN" (indicating the last detected fractal was a downward fractal), the function returns the string "DOWN". If neither of these conditions are met (meaning that no upward or downward fractal was detected or the direction is still neutral), the function returns the default "NEUTRAL" value, indicating there is no specific direction at the moment.

We can also log the other indicators' data as follows.

```
Print("ALLIGATOR JAWS = ",NormalizeDouble(alligator_jaws[1],_Digits));
Print("ALLIGATOR TEETH = ",NormalizeDouble(alligator_teeth[1],_Digits));
Print("ALLIGATOR LIPS = ",NormalizeDouble(alligator_lips[1],_Digits));

Print("AO VALUE = ",NormalizeDouble(ao_values[1],_Digits+1));

if (ac_color[1] == AC_COLOR_UP){
   Print("AC COLOR UP GREEN = ",AC_COLOR_UP);
}
else if (ac_color[1] == AC_COLOR_DOWN){
   Print("AC COLOR DOWN RED = ",AC_COLOR_DOWN);
}
```

Upon run, we get the following output:

**Fractals Confirmation:**

![FRACTALS CONFIRMATION](https://c.mql5.com/2/101/Screenshot_2024-11-13_012602.png)

**Other Indicators Confirmation:**

![OTHER INDICATORS CONFIRMATION](https://c.mql5.com/2/101/Screenshot_2024-11-13_012730.png)

From the visualization, we can see that the retrieved data rhymes with the actual data as seen on the data window, which is a success. We can then continue to use this data for trading purposes. First, we define some necessary functions that we will use to make the analysis as follows.

```
//+------------------------------------------------------------------+
//|        FUNCTION TO GET CLOSE PRICES                              |
//+------------------------------------------------------------------+

double getClosePrice(int bar_index){
   return (iClose(_Symbol, _Period, bar_index)); //--- Retrieve the close price of the specified bar
}

//+------------------------------------------------------------------+
//|        FUNCTION TO GET ASK PRICES                                |
//+------------------------------------------------------------------+

double getAsk(){
   return (NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits)); //--- Get and normalize the Ask price
}

//+------------------------------------------------------------------+
//|        FUNCTION TO GET BID PRICES                                |
//+------------------------------------------------------------------+

double getBid(){
   return (NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits)); //--- Get and normalize the Bid price
}
```

Here, we define 3 functions to get the closing, ask, and bid prices respectively. Next, we need to define boolean variables to check for potential trading signals based on the "Alligator's Jaw" line as below.

```
bool isBreakdown_jaws_buy = alligator_jaws[1] < getClosePrice(1) //--- Check if breakdown for buy
                            && alligator_jaws[2] > getClosePrice(2);
bool isBreakdown_jaws_sell = alligator_jaws[1] > getClosePrice(1) //--- Check if breakdown for sell
                            && alligator_jaws[2] < getClosePrice(2);
```

First, we define "isBreakdown\_jaws\_buy" to detect a breakdown condition for a potential buy signal. The condition is that the value of the "alligator\_jaws" array at index 1 (representing the previous bar) should be less than the close price of the previous bar, which is retrieved by calling the "getClosePrice(1)" function. Additionally, the value of the "alligator\_jaws" array at index 2 (representing the bar before that) should be greater than the close price of the bar before that, which is retrieved by calling the "getClosePrice(2)" function. This combination suggests that the Alligator's Jaw line has crossed below the close price of the previous bar but was above the close price of the bar before that, which could be interpreted as a potential setup for a buy trade.

Next, we define "isBreakdown\_jaws\_sell" to detect a breakdown condition for a potential sell signal. In this case, the "alligator\_jaws" value at index 1 should be greater than the close price of the previous bar, and the "alligator\_jaws" value at index 2 should be less than the close price of the bar before that. This scenario indicates that the Alligator's Jaw line has crossed above the close price of the previous bar but was below the close price of the bar before that, suggesting a potential setup for a sell trade. From here we can define the rest of the conditions for open positions.

```
if (lastFractal_direction == FRACTAL_DOWN //--- Conditions for Buy signal
   && isBreakdown_jaws_buy
   && ac_color[1] == AC_COLOR_UP
   && (ao_values[1] > 0 && ao_values[2] < 0)){
   Print("BUY SIGNAL GENERATED");
   obj_Trade.Buy(0.01,_Symbol,getAsk()); //--- Execute Buy order
}
else if (lastFractal_direction == FRACTAL_UP //--- Conditions for Sell signal
   && isBreakdown_jaws_sell
   && ac_color[1] == AC_COLOR_DOWN
   && (ao_values[1] < 0 && ao_values[2] > 0)){
   Print("SELL SIGNAL GENERATED");
   obj_Trade.Sell(0.01,_Symbol,getBid()); //--- Execute Sell order
}
```

Here, we implement the logic for executing buy and sell signals based on the combination of indicators, specifically the fractal direction, Alligator jaw breakdown, Accelerator Oscillator (AC) color state, and Awesome Oscillator (AO) values.

First, we check if the conditions for a buy signal are met. We verify that the "lastFractal\_direction" is set to "FRACTAL\_DOWN", meaning the last detected fractal was a downward fractal. Then, we check if the "isBreakdown\_jaws\_buy" condition is true, which signals that the Alligator's Jaw line has crossed below the price and is now set up for a potential buy.

Additionally, we ensure that the "ac\_color\[1\]" is equal to "AC\_COLOR\_UP", meaning the Accelerator Oscillator is in an upward color state, indicating a bullish market sentiment. Finally, we check the values of the Awesome Oscillator: "ao\_values\[1\]" should be greater than zero (indicating a positive momentum), and "ao\_values\[2\]" should be less than zero (indicating a previous negative momentum). This combination suggests that there is a reversal in momentum, with the market shifting from negative to positive. If all these conditions are satisfied, a buy signal is generated, and we execute a buy order with the specified lot size (0.01) at the asking price.

On the other hand, we check if the conditions for a sell signal are met. We verify that the "lastFractal\_direction" is set to "FRACTAL\_UP", meaning the last detected fractal was an upward fractal. Then, we check if the "isBreakdown\_jaws\_sell" condition is true, which signals that the Alligator's Jaw line has crossed above the price and is now set up for a potential sell order.

Additionally, we ensure that the "ac\_color\[1\]" is equal to "AC\_COLOR\_DOWN", meaning the Accelerator Oscillator is in a downward color state, indicating a bearish market sentiment. Finally, we check the values of the Awesome Oscillator: "ao\_values\[1\]" should be less than zero (indicating a negative momentum), and "ao\_values\[2\]" should be greater than zero (indicating a previous positive momentum). This combination suggests that the market is reversing from positive to negative momentum. If all these conditions are satisfied, a sell signal is generated, and we execute a sell order with the specified lot size (0.01) at the bid price.

This will open the positions. However, you can see that there are no exit orders placed and thus the positions will hang. We thus need logic to close the positions based on AO indicator reversals.

```
if (ao_values[1] < 0 && ao_values[2] > 0){ //--- Condition to close all Buy positions
   if (PositionsTotal() > 0){
      Print("CLOSE ALL BUY POSITIONS");
      for (int i=0; i<PositionsTotal(); i++){
         ulong pos_ticket = PositionGetTicket(i); //--- Get position ticket
         if (pos_ticket > 0 && PositionSelectByTicket(pos_ticket)){ //--- Check if ticket is valid
            ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            if (pos_type == POSITION_TYPE_BUY){ //--- Close Buy positions
               obj_Trade.PositionClose(pos_ticket);
            }
         }
      }
   }
}
else if (ao_values[1] > 0 && ao_values[2] < 0){ //--- Condition to close all Sell positions
   if (PositionsTotal() > 0){
      Print("CLOSE ALL SELL POSITIONS");
      for (int i=0; i<PositionsTotal(); i++){
         ulong pos_ticket = PositionGetTicket(i); //--- Get position ticket
         if (pos_ticket > 0 && PositionSelectByTicket(pos_ticket)){ //--- Check if ticket is valid
            ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            if (pos_type == POSITION_TYPE_SELL){ //--- Close Sell positions
               obj_Trade.PositionClose(pos_ticket);
            }
         }
      }
   }
}
```

Here, we implement a logic to close all active positions (buy or sell) based on the values of the Awesome Oscillator (AO). First, we check for a condition to close all buy positions. If "ao\_values\[1\]" is less than 0 and "ao\_values\[2\]" is greater than 0, it indicates a potential shift from negative momentum to positive momentum. This is the condition for closing all buy positions. To do this, we first check if any open positions are in existence using the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function.

If there are positions, we loop through each position, retrieving the ticket number of each one using the [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) function. For each position, we validate the ticket with the [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) function, ensuring it is a valid position. Then, we retrieve the position type using [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) and cast it to the [ENUM\_POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type) enum. If the position type is [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type), meaning it is a buy position, we close it using "obj\_Trade.PositionClose(pos\_ticket)" and print "CLOSE ALL BUY POSITIONS" for confirmation.

Next, we check for a condition to close all Sell positions. If "ao\_values\[1\]" is greater than 0 and "ao\_values\[2\]" is less than 0, it indicates a potential shift from positive momentum to negative momentum, signaling the need to close all sell positions. Similarly, we first check if there are any open positions. If there are, we loop through them, retrieve the position tickets, validate the tickets, and check the position type. If the position type is [POSITION\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type), meaning it is a sell position, we close it using "obj\_Trade.PositionClose(pos\_ticket)" and print "CLOSE ALL SELL POSITIONS" for confirmation.

Once we run the program, we get the following output.

![BUY POSITION](https://c.mql5.com/2/101/Screenshot_2024-11-13_020125.png)

That was a success. We can see that we confirmed and opened the buy position when all the entry conditions were met. That is all for the strategy implementation. We now need to test the program in the Strategy Tester and optimize it if necessary for it to adapt to the current market conditions. That is done in the next section below.

### Strategy Testing and Optimization

After completing the core implementation, the next phase now involves testing our Expert Advisor (EA) using the [MetaTrader 5](https://www.metatrader5.com/en/download "https://www.metatrader5.com/en/download") Strategy Tester to evaluate its performance accurately within various market scenarios. This testing stage is to verify the strategy's behavior in line with our expectations and identify any adjustments needed to optimize outcomes. Here, we have already completed an initial optimization, focusing specifically on parameters integral to our strategy.

We have paid particular attention to the fractal and alligator line thresholds to evaluate the EA’s responsiveness in different trading sessions and conditions. This thorough testing has allowed us to validate that the program adheres to the expected buy and sell signals with efficient trade handling, enhancing reliability and performance while minimizing potential errors. Here are the results that we acquired from the testing process.

**Backtest Results:**

![RESULTS](https://c.mql5.com/2/101/Screenshot_2024-11-13_112721.png)

**Backtest Graph:**

![GRAPH](https://c.mql5.com/2/101/Screenshot_2024-11-13_112643.png)

### Conclusion

In this article, we explore the process of creating an [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") Expert Advisor (EA) using the Profitunity trading strategy, integrating fractals, the Alligator indicator, and oscillators like the Awesome and Accelerator Oscillators to identify strategic buy and sell signals. Starting with the core indicators and threshold-based conditions, we automate trade signals that leverage market momentum and price breakouts. Each step involves careful code construction, configuring indicator handles, and implementing logic to trigger buy and sell trades according to the strategy’s criteria. After completing the implementation, we rigorously tested the EA using MetaTrader 5’s Strategy Tester to validate its responsiveness and reliability in various market conditions, emphasizing accuracy in trade execution through optimized parameters.

Disclaimer: This article is an educational guide for building a custom program based on indicator-driven trade signals using the Profitunity trading strategy. The strategies and methods shared do not guarantee specific trading results and should be used cautiously. Always perform thorough testing and validation, adapting automated trading solutions to live market conditions and your personal risk tolerance.

This article provides a structured approach to automating trading signals in [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") using the Profitunity strategy. We hope it encourages you to further explore MQL5 development, inspiring the creation of more sophisticated and profitable trading systems. Happy coding and successful trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16365.zip "Download all attachments in the single ZIP archive")

[1.\_PROFITUNITY\_tTRADING\_CHAOS\_BY\_BILL\_WILLIAMSd.mq5](https://www.mql5.com/en/articles/download/16365/1._profitunity_ttrading_chaos_by_bill_williamsd.mq5 "Download 1._PROFITUNITY_tTRADING_CHAOS_BY_BILL_WILLIAMSd.mq5")(14.36 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)
- [Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/476890)**
(8)


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
5 Aug 2025 at 09:04

**MrBrooklin [#](https://www.mql5.com/en/forum/476890#comment_57730774):**

I have a question for the author of the article regarding this part of the text:

As a [beginner in](https://www.mql5.com/en/articles/447 "Article: Quick Dive into MQL5") MQL5 [programming](https://www.mql5.com/en/articles/447 "Article: Quick Dive into MQL5"), it is not quite clear to me why it is necessary to initialise handles of all indicators with INVALID\_HANDLE value at once? What will happen if we declare indicator handles without initialisation? The Expert Advisor will not work or what?

Regards, Vladimir.

Thanks for the kind feedback. It is not a must you initialize the handles but it is a good programing practice doing so in that you can check if they were initialized after you define them to avoid potential errors. It is just for a security check. For example, you can do this:

```
//--- on a global scope
int m_handleRsi; // HANDLE NOT INITIALIZED
OR
int m_handleRsi = INVALID_HANDLE; // HANDLE INITIALIZED

//--- on initialization
m_handleRsi = iRSI(m_symbol, RSI_TF, RSI_PERIOD, RSI_APP_PRICE); // YOU COULD JUST INITIALIZE AND MOVE ON
OR
m_handleRsi = iRSI(m_symbol, RSI_TF, RSI_PERIOD, RSI_APP_PRICE); // YOU COULD INITIALIZE AND CHECK. THIS IS BETTER
if (m_handleRsi == INVALID_HANDLE) {
   Print("Failed to initialize RSI indicator");
   return false;
}

// So now any will work. Let's take an instance where the indicator initialization fails, though rare.
// If there was no check, no indicator will be added and thus strategy logic will be tampered with.
// For the one who checked, the program will terminate, avoiding false strategy. In the OnInit event handler, it will return initialization failed and the program will not run.
// So the user will know something failed and needs to be checked. If you did not check, the program will run but where it needs the failed indicator, the logic will fail. You get it now?
// The Initialization logic looks like this:

int OnInit() {
   if (!(YOUR LOGIC) e.g. m_handleRsi == INVALID_HANDLE) {
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}
```

Does this make sense now? Thanks.

![Zhang Ming](https://c.mql5.com/avatar/avatar_na2.png)

**[Zhang Ming](https://www.mql5.com/en/users/youmin_1979)**
\|
30 Aug 2025 at 09:14

Very detailed content, thanks for sharing it wonderfully!


![peter matty](https://c.mql5.com/avatar/avatar_na2.png)

**[peter matty](https://www.mql5.com/en/users/petermatty)**
\|
31 Aug 2025 at 06:53

Quote: In this article, we examine the Profitunity System by Bill Williams, breaking down its core components and unique approach to trading within market chaos.

Answer: Profit and loss columns will only exist if your back tested [product](https://www.metatrader5.com/en/terminal/help/fundamental/economic_indicators_usa/usa_productivity "Productivity") or the flat market is as good as the forward market you are using against the subsequence portfolio or basket of index that will follow this line of order.

There are some index and newly foundered ETF\`s coming out, or that are produced on an increasing basis, as for this intended usage, and will produce these results, profit margins such as the dowjones 30 index as well many other index's which have been created for this intended use. Peter Matty

![Miguel Angel Vico Alba](https://c.mql5.com/avatar/2025/10/68e99f33-714e.jpg)

**[Miguel Angel Vico Alba](https://www.mql5.com/en/users/mike_explosion)**
\|
31 Aug 2025 at 11:43

**peter matty [#](https://www.mql5.com/en/forum/476890#comment_57923748):**

The article is not about profit/loss "columns" or market indices/ETFs. It focuses on the Profitunity System by Bill Williams and how to implement its indicators (Fractals, Alligator, AO, AC) in MQL5.

The discussion here is around coding practices and strategy automation, so keeping to those points will be most helpful for readers.

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
2 Sep 2025 at 05:57

**Miguel Angel Vico Alba [#](https://www.mql5.com/en/forum/476890#comment_57924738):**

The article is not about profit/loss "columns" or market indices/ETFs. It focuses on the Profitunity System by Bill Williams and how to implement its indicators (Fractals, Alligator, AO, AC) in MQL5.

The discussion here is around coding practices and strategy automation, so keeping to those points will be most helpful for readers.

Sure

![Neural Networks Made Easy (Part 93): Adaptive Forecasting in Frequency and Time Domains (Final Part)](https://c.mql5.com/2/80/Neural_networks_are_easy_Part_93____LOGO.png)[Neural Networks Made Easy (Part 93): Adaptive Forecasting in Frequency and Time Domains (Final Part)](https://www.mql5.com/en/articles/15024)

In this article, we continue the implementation of the approaches of the ATFNet model, which adaptively combines the results of 2 blocks (frequency and time) within time series forecasting.

![Connexus Observer (Part 8): Adding a Request Observer](https://c.mql5.com/2/101/http60x60__1.png)[Connexus Observer (Part 8): Adding a Request Observer](https://www.mql5.com/en/articles/16377)

In this final installment of our Connexus library series, we explored the implementation of the Observer pattern, as well as essential refactorings to file paths and method names. This series covered the entire development of Connexus, designed to simplify HTTP communication in complex applications.

![Data Science and ML (Part 32): Keeping your AI models updated, Online Learning](https://c.mql5.com/2/102/Data_Science_and_ML_Part_32___LOGO.png)[Data Science and ML (Part 32): Keeping your AI models updated, Online Learning](https://www.mql5.com/en/articles/16390)

In the ever-changing world of trading, adapting to market shifts is not just a choice—it's a necessity. New patterns and trends emerge everyday, making it harder even the most advanced machine learning models to stay effective in the face of evolving conditions. In this article, we’ll explore how to keep your models relevant and responsive to new market data by automatically retraining.

![Chemical reaction optimization (CRO) algorithm (Part I): Process chemistry in optimization](https://c.mql5.com/2/81/Algorithm_for_optimization_by_chemical_reactions__LOGO___2.png)[Chemical reaction optimization (CRO) algorithm (Part I): Process chemistry in optimization](https://www.mql5.com/en/articles/15041)

In the first part of this article, we will dive into the world of chemical reactions and discover a new approach to optimization! Chemical reaction optimization (CRO) uses principles derived from the laws of thermodynamics to achieve efficient results. We will reveal the secrets of decomposition, synthesis and other chemical processes that became the basis of this innovative method.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pmduyctmffkyjfbednsqmvlidurdifcq&ssn=1769180369067822004&ssn_dr=0&ssn_sr=0&fv_date=1769180369&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16365&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%201)%3A%20The%20Profitunity%20System%20(Trading%20Chaos%20by%20Bill%20Williams)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918036989082320&fz_uniq=5068885291730665169&sv=2552)

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