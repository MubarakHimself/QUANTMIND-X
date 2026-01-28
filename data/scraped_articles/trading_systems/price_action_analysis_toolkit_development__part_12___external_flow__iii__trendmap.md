---
title: Price Action Analysis Toolkit Development (Part 12): External Flow (III) TrendMap
url: https://www.mql5.com/en/articles/17121
categories: Trading Systems, Integration, Indicators
relevance_score: 4
scraped_at: 2026-01-23T17:44:56.335808
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/17121&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068600836046650214)

MetaTrader 5 / Trading systems


### Introduction

During the market volatility of 2020 amid the COVID-19 crisis, traders looked to various technical tools to help gauge when a recovery might occur. Some even experimented with Fibonacci time zones to identify potential turning points based on historical price behavior. Although the alignment between Fibonacci projections and major rebounds is still a subject of debate, these tools offered one of several frameworks for navigating an uncertain environment as governments began implementing stimulus measures and economies gradually reopened.

In our previous [Price Action Analysis Toolkit Development](https://www.mql5.com/en/articles/16984) series, we examined a VWAP-based strategy that focused on how VWAP levels could influence market decisions—signaling a buy when the price was above VWAP and a sell when it was below. However, relying solely on VWAP can be problematic, particularly during periods of extreme market conditions when reversals may occur.

In this article, we take our analysis one step further by combining VWAP with Fibonacci levels to generate trading signals. Fibonacci retracement levels help identify potential areas of support and resistance, and when paired with VWAP, they can enhance the robustness of your trading strategy. We begin by explaining the underlying concepts, then outline the key functions in both the MQL5 EA and the Python script. Next, we delve into additional function details and discuss the expected outcomes, before summarizing the key insights. Please refer to the table of contents below.

- [Concept](https://www.mql5.com/en/articles/17121#para2)
- [Main Functions](https://www.mql5.com/en/articles/17121#para3)
- [Other Functions](https://www.mql5.com/en/articles/17121#para4)
- [Outcomes](https://www.mql5.com/en/articles/17121#para5)
- [Conclusion](https://www.mql5.com/en/articles/17121#para6)

### Concept

Fibonacci retracement is based on ratios derived from the Fibonacci sequence, introduced by Leonardo Fibonacci in the 13th century, and is widely used in technical analysis to pinpoint potential support and resistance levels where price reversals might occur. Combining Fibonacci retracement with VWAP (Volume Weighted Average Price) provides us with a more robust trading tool, as VWAP incorporates both price and volume data to reflect the true market sentiment and liquidity. This integration enhances decision-making by confirming key reversal zones with volume-backed evidence, reducing false signals and offering a dynamic perspective on market behavior.

The _TrendMap_ system is a trading signal generation framework that integrates MQL5 (MetaTrader 5 Expert Advisor) with a Python-based analytical server. The system processes Fibonacci retracement levels, VWAP (Volume Weighted Average Price), and price data to generate trading signals (Buy or Sell).

- If VWAP is below the 50% Fibonacci level (mid-price), Buy signal is generated.
- If VWAP is above the 50% Fibonacci level (mid-price), Sell signal is generated.

This ensures that signals are provided based on price action relative to both VWAP and Fibonacci retracement levels. Below, I have provided a diagram illustrating how the market interacts with Fibonacci levels and VWAP. This diagram represents the necessary conditions that our Python-based system monitors for signal generation.

- Fib 0 - 0%
- Fib 1 - 24%
- Fib 2 - 38%
- Fib 3 - 50%
- Fib 4 - 62%
- Fib 5 -  79%
- Fib 6 - 100%

Fib 3 represents the 50% retracement level, and we observe that the VWAP is above this level, indicating a potential reversal. Consequently, a sell signal is generated. I have also circled the levels where the price interacts with both the Fibonacci levels and the VWAP.

![CONCEPT](https://c.mql5.com/2/117/FVWAP__1.png)

Fig 1. Price vs Fibonacci and VWAP

The MQL5 expert advisor integrates Fibonacci retracement levels and VWAP analysis to generate trading signals without executing trades. It calculates Fibonacci levels from a specified number of past bars, determines swing highs and lows, and computes VWAP to assess market trends. The EA sends this data to a Python server via _HTTP_, which analyzes the market conditions and returns a buy or sell signal. Upon receiving a new signal, the EA updates the chart by displaying the signal and drawing Fibonacci lines, the VWAP line, and arrows for visual representation.

The system operates on a timer-based approach, ensuring periodic updates while logging and tracking market changes. This script sets up a Flask server that takes in market data from the MQL5 EA, processes it, and returns a trading signal.

When the EA sends JSON data containing VWAP, Fibonacci retracement levels, swing high/low, and price data, the script evaluates whether VWAP is below or above the mid-price of the swing range to determine a "Buy" or "Sell" signal. At the same time, it generates a _Matplotlib_ chart plotting VWAP, Fibonacci levels, and price data, saving the image for reference. The plotting runs in a separate thread to keep things smooth.

Everything is logged for tracking, and the server runs on _127.0.0.1:5110_. Let's go through the flowchart below to get a full picture of the process.

![Main Logic](https://c.mql5.com/2/116/Flowchart.png)

Fig 2. Flowchart

### Main Functions

**MQL5 Expert Advisor**

In this part, we’ll walk through the core functions that power our Fibonacci-VWAP EA in MQL5. The goal is to:

> 1. Calculate Fibonacci retracement levels using swing highs and lows
> 2. Determine VWAP (Volume Weighted Average Price) for trend confirmation
> 3. Send market data to Python, which will perform more profound analysis and return a buy/sell signal
> 4. Draw signal arrows on the chart once all conditions are met

Let’s break it down step by step.

1\. Updating Fibonacci and VWAP Indicators

To begin, we must ensure that our Fibonacci retracement levels and VWAP are up-to-date before making generating any signal. We achieve this by calling two key functions. First, the _CalculateFibonacciLevels()_ function is responsible for identifying the swing high and swing low points, as well as calculating the relevant Fibonacci retracement levels.

These levels are crucial for understanding potential support and resistance zones. The second function, _CalculateVWAP()_, computes the Volume Weighted Average Price (VWAP), which serves as an indicator for determining the overall market trend based on volume and price. By keeping these indicators updated, we make sure we are working with the most recent market data, which is essential for accurate analysis and decision-making.

```
void UpdateIndicators()
  {
   CalculateFibonacciLevels(InpTimeFrame, InpNumBars, g_SwingHigh, g_SwingLow, g_FibLevels);
   g_VWAP = CalculateVWAP(InpTimeFrame, InpNumBars);
   Print("Updated Indicators: SwingHigh=" + DoubleToString(g_SwingHigh, Digits()) +
         ", SwingLow=" + DoubleToString(g_SwingLow, Digits()) +
         ", VWAP=" + DoubleToString(g_VWAP, Digits()));
  }
```

2\. Sending Market Data to Python

Once the Fibonacci and VWAP indicators are updated, the next step is to send the market data to Python for more advanced analysis. This is done by creating a JSON payload that includes all the essential information needed for further processing. The payload is structured to contain key data points such as the symbol and timeframe of the chart, the swing high and low points, the VWAP value, and the calculated Fibonacci levels. Additionally, we include recent price data for further market analysis. Once the data is structured into this JSON format, it is sent to the Python server through an HTTP request, where Python will process it and return a trading signal based on the calculations.

```
string BuildJSONPayload()
  {
   string jsonPayload = "{";
   jsonPayload += "\"symbol\":\"" + Symbol() + "\",";
   jsonPayload += "\"timeframe\":\"" + EnumToString(InpTimeFrame) + "\",";
   jsonPayload += "\"swingHigh\":" + DoubleToString(g_SwingHigh, Digits()) + ",";
   jsonPayload += "\"swingLow\":" + DoubleToString(g_SwingLow, Digits()) + ",";
   jsonPayload += "\"vwap\":" + DoubleToString(g_VWAP, Digits()) + ",";

   jsonPayload += "\"fibLevels\":[";\
   for(int i = 0; i < 7; i++)\
     {\
      jsonPayload += DoubleToString(g_FibLevels[i], 3);\
      if(i < 6) jsonPayload += ",";\
     }\
   jsonPayload += "],";

   jsonPayload += "\"priceData\":[";\
   for(int i = 0; i < InpNumBars; i++)\
     {\
      jsonPayload += DoubleToString(iClose(Symbol(), InpTimeFrame, i), Digits());\
      if(i < InpNumBars - 1) jsonPayload += ",";\
     }\
   jsonPayload += "]}";

   return jsonPayload;
  }
```

3\. Communicating with Python and Receiving Signals

At this stage, the HTTP request is made, sending the JSON payload to the Python script. The Python server will then analyze the data and return a response containing a buy or sell signal. The HTTP request is built in a way that handles both the sending of data and the reception of responses, checking for any errors during the process. If the request is successful (indicated by a response code of 200), the response is parsed, and the relevant signal (buy, sell, or hold) is extracted from the JSON data returned by Python. This allows the MQL5 EA to integrate external computational power into its decision-making process, enabling more robust trading signals.

```
string SendDataToPython(string payload)
  {
   string headers = "Content-Type: application/json\r\n";
   char postData[];
   StringToCharArray(payload, postData);
   char result[];
   string resultHeaders;
   int resCode = WebRequest("POST", InpPythonURL, headers, InpHTTPTimeout, postData, result, resultHeaders);

   if(resCode == 200)
     {
      string response = CharArrayToString(result);
      Print("HTTP Response: " + response);
      string signal = ParseSignalFromJSON(response);
      return signal;
     }
   else
     {
      Print("Error: WebRequest failed with code " + IntegerToString(resCode));
      return "";
     }
  }
```

4\. Plotting Buy/Sell Signals on the Chart

After receiving the trading signal from Python, the next step is to visually represent that signal on the chart. This is done by drawing an arrow at the current market price to indicate the suggested action. If the signal is a buy signal, an upward-pointing arrow (green in color) is placed, and for a sell signal, a downward-pointing arrow (red) is drawn. This visual cue is crucial for traders to quickly interpret the trading suggestion without needing to analyze numbers.

The arrows are dynamically created using the ObjectCreate function, and their appearance (such as color and size) can be adjusted for better visibility. The use of these arrows makes the trading signals clear and accessible, even for those who might not be following every detail of the system's analysis.

```
void DrawSignalArrow(string signal)
  {
   int arrowCode = 0;
   color arrowColor = clrWhite;

   string lowerSignal = MyStringToLower(signal);
   if(lowerSignal == "buy")
     {
      arrowCode = 233;  // Upward arrow
      arrowColor = clrLime;
     }
   else if(lowerSignal == "sell")
     {
      arrowCode = 234;  // Downward arrow
      arrowColor = clrRed;
     }
   else
     {
      Print("Invalid signal: " + signal);
      return;
     }

   string arrowName = "SignalArrow_" + IntegerToString(TimeCurrent());
   ObjectCreate(0, arrowName, OBJ_ARROW, 0, TimeCurrent(), iClose(Symbol(), PERIOD_CURRENT, 0));
   ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, arrowCode);
   ObjectSetInteger(0, arrowName, OBJPROP_COLOR, arrowColor);
   ObjectSetInteger(0, arrowName, OBJPROP_WIDTH, 2);
  }
```

**Python Script**

1\. Plotting VWAP & Fibonacci vs Price

The _plot\_vwap\_fib\_vs\_price()_ function visualizes the price movement, VWAP, and Fibonacci retracement levels. It is used to help understand the relationship between these three indicators. The function accepts the following parameters:

- symbol: The asset being analyzed
- _vwap_: The calculated Volume Weighted Average Price
- _swingHigh_ and _swingLow_: The high and low points used to calculate Fibonacci retracement levels
- _fibLevels:_ The Fibonacci retracement levels (default values include common retracements such as 0.236, 0.382, etc.)
- _price\_data_: A pandas Series containing the price data over time

Function Breakdown:

- Plotting Price Data: The function starts by plotting the price data over time using sns.set for style and ax.plot for visualization
- Drawing Fibonacci Levels: For each Fibonacci level provided in the list, the function calculates the corresponding price and draws a horizontal line using ax.axhline
- Drawing VWAP: The VWAP is plotted as a yellow horizontal line
- Saving Plot: The plot is saved with a timestamp to avoid overwriting previous charts

The plot is saved as an image file, and this helps visualize how the price interacts with key Fibonacci retracement levels and the VWAP.

```
def plot_vwap_fib_vs_price(symbol: str, vwap: float, swingHigh: float, swingLow: float, fibLevels: list, price_data: pd.Series) -> str:
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=100)
    ax.plot(price_data.index, price_data.values, label="Price", color='blue', marker='o', markersize=4)

    for level in fibLevels:
        level_price = swingLow + (swingHigh - swingLow) * level
        ax.axhline(y=level_price, linestyle='--', linewidth=1.5, label=f'Fib {level*100:.1f}%: {level_price:.5f}')

    ax.axhline(y=vwap, color='yellow', linestyle='-', linewidth=2, label=f'VWAP: {vwap:.5f}')
    ax.set_title(f'VWAP & Fibonacci vs Price for {symbol}')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Price')
    ax.legend()

    filename = f"vwap_fib_plot_{symbol}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close(fig)

    return filename
```

2\. Flask Route for Signal and Analysis

The _get\_signal()_ function is the core route that handles POST requests and is responsible for receiving market data, performing analysis, and returning trading signals. The data is sent to this route as a JSON payload from MQL5.

Function Breakdown:

- Receive Data: The function receives market data in raw format, decodes it, and parses it into a JSON object.
- Extract Data: Key data points like symbol, _swingHigh_, _swingLow_, _vwap_, _fibLevels_, and _priceData_ are extracted from the incoming JSON.
- Plotting: It starts a new thread that calls the plotting function to generate the visual representation of the market conditions (VWAP and Fibonacci).
- Signal Generation: The buy/sell signal is generated by comparing the VWAP with the midpoint between the swing high and swing low. If the VWAP is lower than the midpoint, a "Buy" signal is generated; otherwise, a "Sell" signal is issued.
- Response: A response is returned with the generated signal ("Buy", "Sell", or "None") and an explanation.

The signal generation logic is simple but effective:

- Buy Signal: The VWAP is below the midpoint of the swing high and low, indicating potential upward momentum.
- Sell Signal: The VWAP is above the midpoint, suggesting a downtrend or bearish conditions.

```
@app.route('/getSignal', methods=['POST'])
def get_signal():
    try:
        # Get the raw data from the request
        raw_data = request.data.decode('utf-8').strip()
        logging.debug("Raw data received: " + raw_data)

        # Parse JSON
        decoder = json.JSONDecoder()
        data, idx = decoder.raw_decode(raw_data)
        if idx != len(raw_data):
            logging.error(f"Extra data found after valid JSON: index {idx} of {len(raw_data)}")

        logging.info("Data received from MQL5: " + json.dumps(data, indent=2))

    except Exception as e:
        logging.error("Error parsing JSON: " + str(e))
        return jsonify({"signal": "None", "error": str(e)})

    try:
        # Extract parameters
        symbol = data.get('symbol', 'Unknown')
        swingHigh = float(data.get('swingHigh', 0))
        swingLow = float(data.get('swingLow', 0))
        vwap = float(data.get('vwap', 0))
        fibLevels = data.get('fibLevels', [0.236, 0.382, 0.5, 0.618, 1.618])  # Default levels if not provided

        # Convert priceData list into a pandas Series
        price_data = pd.Series(data.get('priceData', []))

        # Start thread for visualization
        threading.Thread(target=plot_vwap_fib_vs_price, args=(symbol, vwap, swingHigh, swingLow, fibLevels, price_data)).start()

        # Determine signal based on VWAP & Fibonacci
        mid_price = np.mean([swingHigh, swingLow])

        if vwap < mid_price and price_data.iloc[-1] < swingLow + (swingHigh - swingLow) * 0.382:
            signal = "Buy"
        elif vwap > mid_price and price_data.iloc[-1] > swingLow + (swingHigh - swingLow) * 0.618:
            signal = "Sell"
        else:
            signal = "None"

        explanation = f"Signal: {signal} based on VWAP and Fibonacci analysis."

    except Exception as e:
        logging.error("Error processing data: " + str(e))
        signal = "None"
        explanation = "Error processing the signal."

    # Build response
    response = {
        "signal": signal,
        "explanation": explanation,
        "received_data": data
    }

    logging.debug("Sending response to MQL5: " + json.dumps(response))
    return jsonify(response)
```

In the code, the signal generation process takes both VWAP and Fibonacci into account, with the logic explicitly relying on both to determine whether to generate a Buy or Sell signal. The Buy or Sell signal is directly based on the alignment of the VWAP and Fibonacci levels. For example:

- Buy Signal: When the VWAP is below the midpoint and the price is approaching a support level (Fibonacci retracement), it's a strong indication for a Buy.
- Sell Signal: If the VWAP is above the midpoint and the price is at a resistance Fibonacci level, the Sell signal is stronger.

### Other Functions

MQL5 EA

- Helper Functions

Let's start with the helper functions. These are like the small tools in our toolbox that simplify our work later on. For example, the function _BoolToString_ converts a true/false value into a string ("true" or "false") so that when you log or display these values, they're easy to read. Then there's _CharToStr_, which takes a character code (an unsigned short) and converts it to a string—a handy function when you need to work with text data. Finally, _MyStringToLower_ goes through a given string character by character, converting any uppercase letters to lowercase. This is especially useful when you want to compare strings without worrying about case differences. Here’s the exact code:

```
//+------------------------------------------------------------------+
//| Helper: Convert bool to string                                   |
//+------------------------------------------------------------------+
string BoolToString(bool val)
  {
   return(val ? "true" : "false");
  }

//+------------------------------------------------------------------+
//| Helper: Convert ushort (character code) to string                |
//+------------------------------------------------------------------+
string CharToStr(ushort ch)
  {
   return(StringFormat("%c", ch));
  }

//+------------------------------------------------------------------+
//| Helper: Convert a string to lower case                           |
//| This custom function avoids the implicit conversion warning.     |
//+------------------------------------------------------------------+
string MyStringToLower(string s)
  {
   string res = "";
   int len = StringLen(s);
   for(int i = 0; i < len; i++)
     {
      ushort ch = s[i];
      // Check if character is uppercase A-Z.
      if(ch >= 'A' && ch <= 'Z')
         ch = ch + 32;
      res += CharToStr(ch);
     }
   return res;
  }
```

In essence, these functions help ensure our text and data handling is consistent throughout the EA.

- Initialization and Cleanup

Next, we have the initialization and cleanup routines. Think of these as setting the stage before the play begins and tidying up afterward. In the _OnInit_ function, the EA starts by printing a welcome message. It then sets a timer using _EventSetTimer_ (which is critical for periodic tasks) and calls _InitializeFibonacciArray_ to preload our Fibonacci levels. Additionally, it records the current time for managing subsequent updates. If the timer fails, the EA stops immediately to prevent further issues. Conversely, the _OnDeinit_ function is called when the EA is removed from the chart; it kills the timer with _EventKillTimer_ and logs the reason. Here's what that looks like:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("FibVWAP No-Trade EA initializing...");
   if(!EventSetTimer(InpTimerInterval))
     {
      Print("Error: Unable to set timer.");
      return(INIT_FAILED);
     }
   InitializeFibonacciArray();
   g_LastUpdateTime = TimeCurrent();
   Print("FibVWAP No-Trade EA successfully initialized.");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   Print("FibVWAP No-Trade EA deinitialized, reason code: " + IntegerToString(reason));
  }

//+------------------------------------------------------------------+
//| Initialize Fibonacci levels array                                |
//+------------------------------------------------------------------+
void InitializeFibonacciArray()
  {
   g_FibLevels[0] = 0.000;
   g_FibLevels[1] = 0.236;
   g_FibLevels[2] = 0.382;
   g_FibLevels[3] = 0.500;
   g_FibLevels[4] = 0.618;
   g_FibLevels[5] = 0.786;
   g_FibLevels[6] = 1.000;
  }
```

By carefully setting up and tearing down our environment, we ensure that our EA runs smoothly without leaving any loose ends.

- Periodic Execution and Event Handling

Now, let's look at how the EA manages its tasks during runtime. The _OnTimer_ function is key here; it fires at intervals defined by the user (set by _InpTimerInterval_). Every time this timer goes off, the EA updates its indicators, builds a JSON payload, sends that payload to our Python endpoint, and processes any returned signal. If there's a new signal, the EA even draws an arrow on the chart. While _OnTick_ gets called with every market update (or tick), we use it only to call _MainProcessingLoop_ so that heavy processing doesn’t happen on every tick—only at controlled intervals. This design helps in keeping the EA efficient. Check out the code:

```
//+------------------------------------------------------------------+
//| Expert timer function: periodic update                           |
//+------------------------------------------------------------------+
void OnTimer()
  {
   UpdateIndicators();

   string payload = BuildJSONPayload();
   Print("Payload sent to Python: " + payload);

   string signal = SendDataToPython(payload);
   if(signal != "")
     {
      if(signal != g_LastSignal)
        {
         g_LastSignal = signal;
         Print("New signal received: " + signal);
         Comment("ML Signal: " + signal);
         // Draw an arrow on the chart for the new signal.
         DrawSignalArrow(signal);
        }
      else
        {
         Print("Signal unchanged: " + signal);
        }
     }
   else
     {
      Print("Warning: No valid signal received.");
     }

   UpdateChartObjects();
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   MainProcessingLoop();
  }

//+------------------------------------------------------------------+
//| Main processing loop: can be called from OnTick                  |
//+------------------------------------------------------------------+
void MainProcessingLoop()
  {
   datetime currentTime = TimeCurrent();
   if(currentTime - g_LastUpdateTime >= InpTimerInterval)
     {
      UpdateChartObjects();
      g_LastUpdateTime = currentTime;
     }
  }
```

This setup ensures that our EA performs its updates at the right pace without being overwhelmed by every tick.

- Chart Object Management

Visual feedback is significant in trading, and this is where our chart object management comes into play. The _DrawChartObjects_ function is responsible for drawing visual elements on the chart, such as horizontal lines representing our Fibonacci levels and the VWAP. It first clears any previous objects to avoid clutter, then creates new objects with the appropriate colors, styles, and text labels. The helper function _UpdateChartObjects_ simply calls _DrawChartObjects_ to keep things modular, and _ExtendedProcessing_ is available if we need to perform any extra updates or debugging actions. Here is the code:

```
//+------------------------------------------------------------------+
//| Draw objects on the chart for visual indicator levels            |
//+------------------------------------------------------------------+
void DrawChartObjects()
  {
   string objPrefix = "FibVWAP_";
   // Remove previous objects with the given prefix.
   ObjectsDeleteAll(0, objPrefix);
   double range = g_SwingHigh - g_SwingLow;
   for(int i = 0; i < 7; i++)
     {
      double levelPrice = g_SwingLow + range * g_FibLevels[i];
      string name = objPrefix + "FibLevel_" + IntegerToString(i);
      if(!ObjectCreate(0, name, OBJ_HLINE, 0, 0, levelPrice))
         Print("Error creating object: " + name);
      else
        {
         ObjectSetInteger(0, name, OBJPROP_COLOR, clrDeepSkyBlue);
         ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
         ObjectSetString(0, name, OBJPROP_TEXT, "Fib " + DoubleToString(g_FibLevels[i]*100, 0) + "%");
        }
     }
   string vwapName = objPrefix + "VWAP";
   if(!ObjectCreate(0, vwapName, OBJ_HLINE, 0, 0, g_VWAP))
      Print("Error creating VWAP object.");
   else
     {
      ObjectSetInteger(0, vwapName, OBJPROP_COLOR, clrYellow);
      ObjectSetInteger(0, vwapName, OBJPROP_STYLE, STYLE_SOLID);
      ObjectSetString(0, vwapName, OBJPROP_TEXT, "VWAP");
     }
  }

//+------------------------------------------------------------------+
//| Periodically update chart objects                                |
//+------------------------------------------------------------------+
void UpdateChartObjects()
  {
   DrawChartObjects();
  }

//+------------------------------------------------------------------+
//| Extended processing: additional updates and chart redraw         |
//+------------------------------------------------------------------+
void ExtendedProcessing()
  {
   Print("Extended processing executed.");
   UpdateChartObjects();
  }
```

- Additional Utility Functions

Finally, let’s look at the utility functions. These are the unsung heroes that support our EA behind the scenes. The _CharArrayToString_ function takes an array of characters—often received from web requests—and converts it into a usable string. _LogDebugInfo_ prints out messages with a timestamp so you can track the EA’s activity over time, which is very helpful during debugging. And _PauseExecution_ allows you to halt the EA for a given number of seconds, which can be useful if you need to slow things down to observe behavior during testing. Here’s the code for these utilities:

```
//+------------------------------------------------------------------+
//| Convert char array to string                                     |
//+------------------------------------------------------------------+
string CharArrayToString(const char &arr[])
  {
   string ret = "";
   for(int i = 0; i < ArraySize(arr); i++)
      ret += CharToStr(arr[i]);
   return(ret);
  }

//+------------------------------------------------------------------+
//| Custom logging function for detailed debug information           |
//+------------------------------------------------------------------+
void LogDebugInfo(string info)
  {
   string logMessage = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + " | " + info;
   Print(logMessage);
  }

//+------------------------------------------------------------------+
//| Additional utility: Pause execution (for debugging)              |
//+------------------------------------------------------------------+
void PauseExecution(int seconds)
  {
   datetime endTime = TimeCurrent() + seconds;
   while(TimeCurrent() < endTime)
     {
      Sleep(100);
     }
  }
```

These functions might not be the star of the EA, but they make a big difference when it comes to troubleshooting and ensuring everything runs as expected.

Python Script

- Key Imports in Our Python Trading System

Let’s break down the key libraries used in this system. Each plays a crucial role, whether in handling data, performing calculations, or visualizing market insights. Below, I’ll provide a code snippet showing these imports, followed by a table explaining their purpose and how they contribute to the system.

```
import datetime
import json
import logging
import threading

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify
```

| Library | Purpose | How It’s Used in Our System |
| --- | --- | --- |
| _datetime_ | Time Management | Used to timestamp saved trading charts and log events. Every time we generate a chart, we attach a timestamp to keep track of different market conditions over time. |
| _json_ | Data Exchange | Since our system communicates with MQL5, we send and receive data in JSON format. This module helps us encode and decode JSON messages between Python and MQL5. |
| _logging_ | Debugging and monitoring | Keeps track of system events. We use it to log every step—when a signal is received, processed, or if an error occurs—making debugging easier. |
| _threading_ | Running Tasks in Background | When we generate a VWAP-Fibonacci chart, we run it in a separate thread to ensure the server remains responsive and doesn’t delay sending signals back to MQL5. |
| _numpy_ | Numerical Computation | We use _NumPy_ to calculate the mid-price and perform other mathematical operations quickly and efficiently. Speed matters when processing real-time market data. |
| _pandas_ | Data Handling | Since market data is structured in rows and columns (like a spreadsheet), pandas makes it easy to store, filter, and manipulate this data for analysis. |
| _pandas\_ta_ | Technical Indicators | Though not a core part of our current system, it allows for additional technical analysis tools if needed in the future. |
| _matplotlib.pyplot_ | Charting & Visualization | The backbone of our visual analysis. We use it to plot price movements, VWAP levels, and Fibonacci retracement zones. |
| _seanborn_ | Enhanced Chart Styling | Helps make our charts visually appealing, making it easier to spot trends and key levels. |
| _flask_ | Communication Between MQL5 and Python | This is what makes real-time signal exchange possible. Flask creates an API that allows our MQL5 Expert Advisor to send price data and receive trading signals instantly. |

- Flask Server Initialization (if \_\_name\_\_ == '\_\_main\_\_')

This block initializes and runs the Flask web server, allowing the MQL5 EA to communicate with the Python script via HTTP requests.

How It Works:

- Binds the server to port 5110
- Runs in debug mode for real-time logging and troubleshooting
- Ensures the server doesn't auto-restart unnecessarily (use\_reloader=False)

```
if __name__ == '__main__':
    port = 5110
    logging.info(f"Starting Flask server on 127.0.0.1:{port} (debug mode ON)")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
```

### Outcomes

To test the system, start by running the Python script to activate the port and server for connection. If you're unsure how to run the script, refer to the [External Flow article](https://www.mql5.com/en/articles/16967) for guidance. Once the server is active, run the MQL5 EA. Now, let's examine the results after successfully running the system.

Command Prompt Logs

```
2025-02-06 12:45:32,384 DEBUG: Sending response to MQL5: {"signal": "Sell", "explanation": "Signal:
Sell based on VWAP and Fibonacci analysis.", "received_data": {"symbol": "EURUSD", "timeframe":
"PERIOD_H1", "swingHigh": 1.05331, "swingLow": 1.02099, "vwap": 1.03795, "fibLevels": [0.0, 0.236,\
0.382, 0.5, 0.618, 0.786, 1.0], "priceData": [1.03622, 1.03594, 1.03651, 1.03799, 1.03901, 1.03865,\
1.03871, 1.0392, 1.03951, 1.04025, 1.03974, 1.03986, 1.04041, 1.04017, 1.04049, 1.04072, 1.04156,\
1.04197, 1.04181, 1.04223, 1.04224, 1.04079, 1.04192, 1.04147, 1.04251, 1.04039, 1.04039, 1.04026,\
1.0385, 1.03778, 1.03786, 1.03796, 1.03845, 1.03734, 1.03786, 1.03714, 1.0378, 1.03786, 1.03854,\
1.03817, 1.03811, 1.03767, 1.03784, 1.03801, 1.03593, 1.03404, 1.03254, 1.03223, 1.0335, 1.03386,\
1.03344, 1.03115, 1.03067, 1.02932, 1.0306, 1.03147, 1.0322, 1.03221, 1.03178, 1.03281, 1.0342, 1.03441,\
1.02955, 1.02846, 1.02785, 1.02795, 1.02761, 1.03162, 1.0251, 1.0264, 1.02577, 1.02522, 1.02438, 1.0231,\
1.02436, 1.02249, 1.02431, 1.02404, 1.02265, 1.02216, 1.02235, 1.02377, 1.02314, 1.0247, 1.02504, 1.03583,\
1.03733, 1.03763, 1.03698, 1.042, 1.03998, 1.03964, 1.03687, 1.03822, 1.03825, 1.03759, 1.03765, 1.03836,\
1.03845, 1.0404, 1.03946, 1.03888, 1.03875, 1.0385, 1.03897, 1.03884, 1.03867, 1.03969, 1.03873, 1.03885,\
1.04076, 1.0428, 1.0425, 1.0417, 1.04197, 1.04314, 1.0428, 1.04562, 1.04033, 1.03918, 1.04007, 1.04076,\
1.04217, 1.04239, 1.04139, 1.0422, 1.04191, 1.04253, 1.0423, 1.042, 1.04259, 1.04247, 1.04216, 1.04209,\
1.04105, 1.04164, 1.042, 1.04213, 1.04157, 1.04194, 1.04013, 1.03878, 1.0404, 1.04016, 1.04037, 1.04038,\
1.04244, 1.04161, 1.04372, 1.04403, 1.04386, 1.04374, 1.0434, 1.04272, 1.04304, 1.04272, 1.04286, 1.04301,\
1.04315, 1.0435, 1.04264, 1.04279, 1.04262, 1.04241, 1.04314, 1.04249, 1.04203, 1.04234, 1.0425, 1.04352, 1.04252, 1.04342, 1.04376, 1.04364, 1.04336, 1.04291, 1.04336, 1.04378, 1.04453, 1.0437, 1.04886, 1.04916, 1.04881, 1.04926, 1.04849, 1.04888, 1.04908, 1.04992, 1.05094, 1.05199, 1.05212, 1.0513, 1.05054, 1.04888, 1.04875, 1.04571, 1.04591, 1.0463, 1.04633, 1.04686]}}
2025-02-06 12:39:32,397 INFO: 127.0.0.1 - - [06/Feb/2025 12:39:32] "POST /getSignal HTTP/1.1" 200 -
2025-02-06 12:39:32,805 DEBUG: VWAP & Fibonacci vs Price graph saved as vwap_fib_plot_EURUSD_20250206_123932.png
```

MQL5 Logs

```
2025.02.06 12:45:32.331 FIBVWAP (EURUSD,M5)     Fib Level 0 (0%): 1.02099
2025.02.06 12:45:32.331 FIBVWAP (EURUSD,M5)     Fib Level 1 (24%): 1.02862
2025.02.06 12:45:32.331 FIBVWAP (EURUSD,M5)     Fib Level 2 (38%): 1.03334
2025.02.06 12:45:32.331 FIBVWAP (EURUSD,M5)     Fib Level 3 (50%): 1.03715
2025.02.06 12:45:32.331 FIBVWAP (EURUSD,M5)     Fib Level 4 (62%): 1.04096
2025.02.06 12:45:32.331 FIBVWAP (EURUSD,M5)     Fib Level 5 (79%): 1.04639
2025.02.06 12:45:32.332 FIBVWAP (EURUSD,M5)     Fib Level 6 (100%): 1.05331
2025.02.06 12:45:32.332 FIBVWAP (EURUSD,M5)     Updated Indicators: SwingHigh=1.05331, SwingLow=1.02099, VWAP=1.03795
2025.02.06 12:45:32.332 FIBVWAP (EURUSD,M5)     Payload sent to Python: {"symbol":"EURUSD","timeframe":"PERIOD_H1","swingHigh":1.05331,"swingLow":1.02099,"vwap":1.03795,"fibLevels":[0.000,0.236,0.382,0.500,0.618,0.786,1.000],"priceData":[1.03583,1.03594,1.03651,1.03799,1.03901,1.03865,1.03871,1.03920,1.03951,1.04025,1.03974,1.03986,1.04041,1.04017,1.04049,1.04072,1.04156,1.04197,1.04181,1.04223,1.04224,1.04079,1.04192,1.04147,1.04251,1.04039,1.04039,1.04026,1.03850,1.03778,1.03786,1.03796,1.03845,1.03734,1.03786,1.03714,1.03780,1.03786,1.03854,1.03817,1\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)     HTTP Response: {\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)       "explanation": "Signal: Sell based on VWAP and Fibonacci analysis.",\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)       "received_data": {\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)         "fibLevels": [\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           0.0,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           0.236,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           0.382,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           0.5,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           0.618,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           0.786,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.0\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)         ],\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)         "priceData": [\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03583,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03594,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03651,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03799,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03901,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03865,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03871,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.0392,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03951,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04025,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03974,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03986,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04041,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04017,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04049,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04072,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04156,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04197,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04181,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04223,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04224,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04079,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04192,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04147,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04251,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04039,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04039,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.04026,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.0385,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03778,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03786,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03796,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03845,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03734,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03786,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03714,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.0378,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03786,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03854,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03817,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03811,\
2025.02.06 12:45:32.441 FIBVWAP (EURUSD,M5)           1.03767,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03784,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03801,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03593,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03404,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03254,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03223,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0335,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03386,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03344,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03115,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03067,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02932,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0306,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03147,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0322,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03221,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03178,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03281,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0342,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03441,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02955,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02846,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02785,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02795,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02761,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03162,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0251,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0264,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02577,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02522,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02438,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0231,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02436,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02249,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02431,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02404,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02265,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02216,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02235,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02377,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02314,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0247,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.02504,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03583,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03733,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03763,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03698,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.042,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03998,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03964,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03687,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03822,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03825,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03759,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03765,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03836,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03845,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0404,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03946,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03888,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03875,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0385,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03897,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03884,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03867,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03969,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03873,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03885,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04076,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0428,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0425,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0417,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04197,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04314,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0428,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04562,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04033,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03918,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04007,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04076,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04217,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04239,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04139,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0422,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04191,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04253,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0423,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.042,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04259,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04247,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04216,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04209,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04105,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04164,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.042,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04213,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04157,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04194,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04013,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.03878,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0404,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04016,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04037,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04038,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04244,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04161,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04372,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04403,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04386,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04374,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0434,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04272,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04304,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04272,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04286,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04301,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04315,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0435,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04264,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04279,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04262,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04241,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04314,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04249,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04203,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04234,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0425,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04352,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04252,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04342,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04376,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04364,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04336,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04291,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04336,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04378,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04453,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0437,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04886,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04916,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04881,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04926,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04849,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04888,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04908,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04992,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.05094,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.05199,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.05212,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0513,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.05054,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04888,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04875,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04571,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04591,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.0463,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04633,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)           1.04686\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)         ],\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)         "swingHigh": 1.05331,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)         "swingLow": 1.02099,\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)         "symbol": "EURUSD",\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)         "timeframe": "PERIOD_H1",\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)         "vwap": 1.03795\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)       },\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)       "signal": "Sell"\
2025.02.06 12:45:32.442 FIBVWAP (EURUSD,M5)     }\
```\
\
The VWAP-Fibonacci vs. Price graph was also plotted and saved in the same directory as the Python script. This plot helps visualize how price interacts with VWAP and Fibonacci levels.\
\
![MATPLOTLIB](https://c.mql5.com/2/117/FIBVWAP_PLOT__1.png)\
\
Fig 3. Matplotlib Plot\
\
In the diagram below, the MQL5 EA on the MetaTrader 5 chart logs and displays the response from the Python server. The chart shows two buy signals, indicated by arrows, which are also logged in the Experts tab. We can see that the market followed the direction of these signals, confirming their validity. I chose to view the M1 chart for a clearer analysis.\
\
![B300](https://c.mql5.com/2/117/B300.gif)\
\
Fig 4. Test on Boom 300 Index\
\
### Conclusion\
\
This system turns out to be highly useful for traders. Besides providing signals, it allows for a clear visualization of how price interacts with Fibonacci levels and VWAP during signal generation. The visualization is achieved in two ways: Python plots a graph using the _Matplotlib_ library, while the MQL5 EA displays VWAP and Fibonacci lines directly on the chart for better analysis. This enhanced visual representation can assist traders in making informed decisions.\
\
| Date | Tool Name | Description | Version | Updates | Notes |\
| --- | --- | --- | --- | --- | --- |\
| 01/10/24 | [Chart Projector](https://www.mql5.com/en/articles/16014) | Script to overlay the previous day's price action with ghost effect. | 1.0 | Initial Release | First tool in Lynnchris Tool Chest |\
| 18/11/24 | [Analytical Comment](https://www.mql5.com/en/articles/15927) | It provides previous day's information in a tabular format, as well as anticipates the future direction of the market. | 1.0 | Initial Release | Second tool in the Lynnchris Tool Chest |\
| 27/11/24 | [Analytics Master](https://www.mql5.com/en/articles/16434) | Regular Update of market metrics after every two hours | 1.01 | Second Release | Third tool in the Lynnchris Tool Chest |\
| 02/12/24 | [Analytics Forecaster](https://www.mql5.com/en/articles/16559) | Regular Update of market metrics after every two hours with telegram integration | 1.1 | Third Edition | Tool number 4 |\
| 09/12/24 | [Volatility Navigator](https://www.mql5.com/en/articles/16560) | The EA analyzes market conditions using the Bollinger Bands, RSI and ATR indicators | 1.0 | Initial Release | Tool Number 5 |\
| 19/12/24 | [Mean Reversion Signal Reaper](https://www.mql5.com/en/articles/16700) | Analyzes market using mean reversion strategy and provides signal | 1.0 | Initial Release | Tool number 6 |\
| 9/01/25 | [Signal Pulse](https://www.mql5.com/en/articles/16861) | Multiple timeframe analyzer | 1.0 | Initial Release | Tool number 7 |\
| 17/01/25 | [Metrics Board](https://www.mql5.com/en/articles/16584) | Panel with button for analysis | 1.0 | Initial Release | Tool number 8 |\
| 21/01/25 | [External Flow](https://www.mql5.com/en/articles/16967) | Analytics through external libraries | 1.0 | Initial Release | Tool number 9 |\
| 27/01/25 | [VWAP](https://www.mql5.com/en/articles/16984) | Volume Weighted Average Price | 1.3 | Initial Release | Tool number 10 |\
| 02/02/25 | [Heikin Ashi](https://www.mql5.com/en/articles/17021) | Trend Smoothening and reversal signal identification | 1.0 | Initial Release | Tool number 11 |\
| 04/02/25 | FibVWAP | Signal generation through python analysis | 1.0 | Initial Release | Tool number  12 |\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/17121.zip "Download all attachments in the single ZIP archive")\
\
[server.py](https://www.mql5.com/en/articles/download/17121/server.py "Download server.py")(4.53 KB)\
\
[FIBVWAP.mq5](https://www.mql5.com/en/articles/download/17121/fibvwap.mq5 "Download FIBVWAP.mq5")(17.22 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Price Action Analysis Toolkit (Part 55): Designing a CPI Mini-Candle Overlay for Intra-bar Pressure](https://www.mql5.com/en/articles/20949)\
- [Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://www.mql5.com/en/articles/20851)\
- [Price Action Analysis Toolkit Development (Part 53): Pattern Density Heatmap for Support and Resistance Zone Discovery](https://www.mql5.com/en/articles/20390)\
- [Price Action Analysis Toolkit Development (Part 52): Master Market Structure with Multi-Timeframe Visual Analysis](https://www.mql5.com/en/articles/20387)\
- [Price Action Analysis Toolkit Development (Part 51): Revolutionary Chart Search Technology for Candlestick Pattern Discovery](https://www.mql5.com/en/articles/20313)\
- [Price Action Analysis Toolkit Development (Part 50): Developing the RVGI, CCI and SMA Confluence Engine in MQL5](https://www.mql5.com/en/articles/20262)\
- [Price Action Analysis Toolkit Development (Part 49): Integrating Trend, Momentum, and Volatility Indicators into One MQL5 System](https://www.mql5.com/en/articles/20168)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/481278)**\
(2)\
\
\
![linfo2](https://c.mql5.com/avatar/2023/4/6438c14d-e2f0.png)\
\
**[linfo2](https://www.mql5.com/en/users/neilhazelwood)**\
\|\
21 Mar 2025 at 18:52\
\
Thank You Chris for all your ideas and publications, definitely helpful stuff. For this article. to get it to work I need to make some setting appropriate to my environment after rechecking  [Price Action Analysis Toolkit Development (Part 9): External Flow - MQL5 Articles](https://www.mql5.com/en/articles/16967) Which Maybe are helpful to others\
\
1  update the scripts to use your address or local host 127.0.0.1\
\
2  in tools options web request url add the address you  are using\
\
4 make sure all of the Python parts are installed\
\
install matplotlib\
\
pip install pandas\
\
pip install Flask seaborn and Numpty if necessary\
\
3  start the flask app on that address by running the python file  with the address you are using\
\
![Christian Benjamin](https://c.mql5.com/avatar/2025/10/68fd3661-daee.png)\
\
**[Christian Benjamin](https://www.mql5.com/en/users/lynnchris)**\
\|\
22 Mar 2025 at 14:59\
\
**linfo2 [#](https://www.mql5.com/en/forum/481278#comment_56236266):**\
\
Thank You Chris for all your ideas and publications, definitely helpful stuff. For this article. to get it to work I need to make some setting appropriate to my environment after rechecking  [Price Action Analysis Toolkit Development (Part 9): External Flow - MQL5 Articles](https://www.mql5.com/en/articles/16967) Which Maybe are helpful to others\
\
1  update the scripts to use your address or local host 127.0.0.1\
\
2  in tools options web request url add the address you  are using\
\
4 make sure all of the Python parts are installed\
\
install matplotlib\
\
pip install pandas\
\
pip install Flask seaborn and Numpty if necessary\
\
3  start the flask app on that address by running the python file  with the address you are using\
\
Hello linfo2\
\
You are absolutely right on the additional notes you provided. Many thanks to you🔥.\
\
![Neural Networks in Trading: Lightweight Models for Time Series Forecasting](https://c.mql5.com/2/86/Neural_networks_in_trading_____Easy_time_series_forecasting_models___LOGO.png)[Neural Networks in Trading: Lightweight Models for Time Series Forecasting](https://www.mql5.com/en/articles/15392)\
\
Lightweight time series forecasting models achieve high performance using a minimum number of parameters. This, in turn, reduces the consumption of computing resources and speeds up decision-making. Despite being lightweight, such models achieve forecast quality comparable to more complex ones.\
\
![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (I)](https://c.mql5.com/2/117/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_IX___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (I)](https://www.mql5.com/en/articles/16539)\
\
This discussion delves into the challenges encountered when working with large codebases. We will explore the best practices for code organization in MQL5 and implement a practical approach to enhance the readability and scalability of our Trading Administrator Panel source code. Additionally, we aim to develop reusable code components that can potentially benefit other developers in their algorithm development. Read on and join the conversation.\
\
![Automating Trading Strategies in MQL5 (Part 6): Mastering Order Block Detection for Smart Money Trading](https://c.mql5.com/2/118/Automating_Trading_Strategies_in_MQL5_Part_6__LOGO.png)[Automating Trading Strategies in MQL5 (Part 6): Mastering Order Block Detection for Smart Money Trading](https://www.mql5.com/en/articles/17135)\
\
In this article, we automate order block detection in MQL5 using pure price action analysis. We define order blocks, implement their detection, and integrate automated trade execution. Finally, we backtest the strategy to evaluate its performance.\
\
![Introduction to MQL5 (Part 12): A Beginner's Guide to Building Custom Indicators](https://c.mql5.com/2/117/Introduction_to_MQL5_Part_12___LOGO.png)[Introduction to MQL5 (Part 12): A Beginner's Guide to Building Custom Indicators](https://www.mql5.com/en/articles/17096)\
\
Learn how to build a custom indicator in MQL5. With a project-based approach. This beginner-friendly guide covers indicator buffers, properties, and trend visualization, allowing you to learn step-by-step.\
\
[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/17121&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068600836046650214)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).