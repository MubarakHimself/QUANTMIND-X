---
title: Implementing a Rapid-Fire Trading Strategy Algorithm with Parabolic SAR and Simple Moving Average (SMA) in MQL5
url: https://www.mql5.com/en/articles/15698
categories: Trading Systems, Indicators, Expert Advisors
relevance_score: 18
scraped_at: 2026-01-22T17:08:29.034483
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=vgsfhxhmrlvmexlwcikfrcltriwzrtyh&ssn=1769090379404731416&ssn_dr=0&ssn_sr=0&fv_date=1769090379&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15698&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Implementing%20a%20Rapid-Fire%20Trading%20Strategy%20Algorithm%20with%20Parabolic%20SAR%20and%20Simple%20Moving%20Average%20(SMA)%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909037911770343&fz_uniq=5048819994797645642&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

This article will walk you through the vital processes of testing and optimizing our trading algorithm. We will use MQL5's Strategy Tester to backtest our EA against historical data. In layman's terms, we will check how well our algorithm would have performed in the past compared to real market conditions. As we discuss our EA's performance, we will also cover how to interpret various performance metrics (e.g., profit factor, drawdown, win rate, etc.) and what they tell us about our algorithm's reliability and profitability. By the end of this article, we will have a much clearer idea of how to implement a rapid-fire trading strategy and what it takes to ensure its success. The following topics will be effective throughout the article:

1. [Introduction to Rapid-Fire Trading Strategy](https://www.mql5.com/en/articles/15698#para1)
2. [Understanding the Indicators: Parabolic SAR and SMA](https://www.mql5.com/en/articles/15698#para2)
3. [Implementation in MQL5](https://www.mql5.com/en/articles/15698#para3)
4. [Testing and Optimization](https://www.mql5.com/en/articles/15698#para4)
5. [Conclusion](https://www.mql5.com/en/articles/15698#para5)

A rapid-fire trading strategy focuses on profiting from quick and frequent market movements by executing multiple trades in a short timeframe, often holding positions for less than an hour. This approach contrasts with traditional strategies that target longer-term trends, aiming instead to exploit small price changes in fast-moving securities. Success in this strategy relies on the ability to process and respond to market data almost instantly, making automated systems, like Expert Advisors, crucial for executing trades efficiently.

Key to the effectiveness of this strategy is the use of technical indicators such as the Parabolic SAR and Simple Moving Average, which help identify trend reversals and smooth out price data. These indicators, integrated into a high-frequency trading algorithm, enable the strategy to adapt quickly to market changes, maintaining accuracy in signal generation and trade execution. When properly implemented, rapid-fire trading can yield quick profits, though it requires careful management of costs and risks.

### Understanding the Indicators: Parabolic SAR and SMA

An effective rapid-fire trading strategy requires an understanding of the important technical indicators that direct trading decisions. Two guides that are useful in this respect are the Parabolic SAR (Stop and Reverse) and the Simple Moving Average (SMA). The SMA is one of the oldest and most often-used trend-following indicators. The Parabolic SAR is relatively new in comparison but is certainly not a lesser-known tool; both of these indicators are very handy in determining market conditions and signaling potential trading opportunities.

Designed to follow trends, the Parabolic SAR indicator aims to find and identify potential price direction reversals. It works by plotting a series of dots either above or below the price, relative to where the price is about the SAR (stop and reverse). If we incorporate the Parabolic SAR into our trading strategy, we can interpret it relative to the series of dots to decide to buy, sell, or short a position in the market. If the price is above the SAR dots, we're in a bullish trend; if the price is below the SAR dots, we're in a bearish trend. Default indicator settings are to be used. This is as illustrated below:

![PARABOLIC SAR SETTINGS](https://c.mql5.com/2/134/Screenshot_2024-08-25_112003.png)

Parabolic SAR indicator setup:

![PARABOLIC SAR SETUP](https://c.mql5.com/2/134/Screenshot_2024-08-25_112711.png)

Another essential part of our trading strategy is the Simple Moving Average (SMA). The SMA takes price data over a specified period and "smooths" it out. It gives us a much easier-to-read look at what the overall trend is. We could also use the SMA to get an even simpler read on the trend. If the SMA is sloping up, then we could say, in very simple terms, that the market is in an uptrend. If the SMA is sloping down, we could say the market is in a downtrend. The SMA is a trend filter. It tells us whether we should be looking for long (uptrend), short (downtrend), or no trades (when the price is hovering around the SMA's flatline). A simple moving average of period 60 is to be used. This is as illustrated below:

![SMA SETTINGS](https://c.mql5.com/2/134/Screenshot_2024-08-25_112939.png)

Simple Moving Average indicator setup:

![SMA SETUP](https://c.mql5.com/2/134/Screenshot_2024-08-25_113210.png)

When used together, the Parabolic SAR and simple moving average yield a complete picture of market conditions. The SAR can be read for immediate signals about what might be happening to the current trend. It can and does quite often signal trend reversals before they happen. The SMA looks at more data over a longer period and, therefore, confirms the trend direction with more certainty. When combined, we have them as shown below:

![PARABOLIC SAR & SMA COMBINED SETUP](https://c.mql5.com/2/134/Screenshot_2024-08-25_113401.png)

With the provided strategy overview, let us craft the strategy's Expert Advisor in MQL5.

### Implementation in MQL5

After learning all the theories about the Rapid-Fire trading strategy, let us then automate the theory and craft an Expert Advisor (EA) in MetaQuotes Language 5 (MQL5) for MetaTrader 5 (MT5).

To create an expert advisor (EA), on your MetaTrader 5 terminal, click the Tools tab and check MetaQuotes Language Editor, or simply press F4 on your keyboard. Alternatively, you can click the IDE (Integrated Development Environment) icon on the tools bar. This will open the MetaQuotes Language Editor environment, which allows the writing of trading robots, technical indicators, scripts, and libraries of functions.

![OPEN METAEDITOR](https://c.mql5.com/2/134/f._IDE.png)

Once the MetaEditor is opened, on the tools bar, navigate to the File tab and check New File, or simply press CTRL + N, to create a new document. Alternatively, you can click on the New icon on the tools tab. This will result in a MQL Wizard pop-up.

![CREATE NEW EA ](https://c.mql5.com/2/134/g._NEW_EA_CREATE.png)

On the Wizard that pops, check Expert Advisor (template) and click Next.

![MQL WIZARD](https://c.mql5.com/2/134/h._MQL_Wizard.png)

On the general properties of the Expert Advisor, under the name section, provide your expert's file name. Note that to specify or create a folder if it doesn't exist, you use the backslash before the name of the EA. For example, here we have "Experts\\" by default. That means that our EA will be created in the Experts folder and we can find it there. The other sections are pretty much straightforward, but you can follow the link at the bottom of the Wizard to know how to precisely undertake the process.

![NEW EA NAME](https://c.mql5.com/2/134/i._NEW_EA_NAME.png)

After providing your desired Expert Advisor file name, click on Next, click Next, and then click Finish. After doing all that, we are now ready to code and program our strategy.

First, we start by defining some metadata about the Expert Advisor (EA). This includes the name of the EA, the copyright information, and a link to the MetaQuotes website. We also specify the version of the EA, which is set to "1.00".

```
//+------------------------------------------------------------------+
//|                                               #1. RAPID FIRE.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

#property copyright "Copyright 2024, MetaQuotes Ltd."   // Set the copyright property for the Expert Advisor
#property link      "https://www.mql5.com"              // Set the link property for the Expert Advisor
#property version   "1.00"                              // Set the version of the Expert Advisor
```

When loading the program, information that depicts the one shown below is realized.

![EA METADATA](https://c.mql5.com/2/134/Screenshot_2024-08-25_125643.png)

[First, we include a trade instance by using](https://www.mql5.com/en/articles/14105#para1) [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) at the beginning of the source code. This gives us access to the CTrade class, which we will use to create a trade object. This is crucial as we need it to open trades.

```
#include <Trade/Trade.mqh>   // Include the MQL5 standard library for trading operations
CTrade obj_Trade;            // Create an instance of the CTrade class to handle trade operations
```

The preprocessor will replace the line [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Trade/Trade.mqh> with the content of the file Trade.mqh. Angle brackets indicate that the Trade.mqh file will be taken from the standard directory (usually it is terminal\_installation\_directory\\MQL5\\Include). The current directory is not included in the search. The line can be placed anywhere in the program, but usually, all inclusions are placed at the beginning of the source code, for a better code structure and easier reference. Declaration of the obj\_Trade object of the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class will give us access to the methods contained in that class easily, thanks to the MQL5 developers.

![CTRADE CLASS](https://c.mql5.com/2/134/j._INCLUDE_CTRADE_CLASS__2.png)

We will need to create indicator handles so that we can include the necessary indicators in the strategy.

```
int handleSMA = INVALID_HANDLE, handleSAR = INVALID_HANDLE;  // Initialize handles for SMA and SAR indicators
```

Here, we first declare and initialize two [integer](https://www.mql5.com/en/docs/basis/types/integer) variables, "handleSMA" and "handleSAR". These will serve as handles for two of the technical indicators we plan to use in our Expert Advisor. In MQL5, a handle is a unique identifier assigned to an indicator when we create it using specific MQL5 functions. We will use the identifiers to reference the non-painted indicators anywhere in the program. We have also assigned both identifiers to an [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) constant, which ensures that the program won't reference an invalid handle if we get lazy and forget to check the program's flow logic after it has been compiled. Both indicators will do their work without any problems if we handle them correctly and avoid referencing an invalid handle or doing anything stupid without checking the handles first.

Next, we will need to create arrays where we store the values or data that we retrieve from the indicators.

```
double sma_data[], sar_data[];  // Arrays to store SMA and SAR indicator data
```

Here, we define two [dynamic](https://www.mql5.com/en/book/common/arrays/arrays_dynamic) arrays, "sma\_data" and "sar\_data", to which we will assign the data points generated by the SMA (Simple Moving Average) and the SAR (Parabolic SAR) indicators. We assign the most recent values calculated by each indicator to these arrays, leaving them free to reference and analyze the data when generating trading signals. Trading signals can be generated from a signal line crossover. To generate a signal trading with the current trend or a reversal of the current trend, we can make use of the past values held in these arrays. By making use of these past values, we should be able to more accurately interpret our indicators and the price action of the asset being traded.

Next, we need the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler. The handler is essential because it is automatically called when the Expert Advisor (EA) is initialized on a chart. This function is responsible for setting up the EA, including creating necessary indicator handles, initializing variables, and preparing resources. In other words, [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) is an in-built function that ensures that everything is properly configured before the EA begins processing market data. It is as follows.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){
   // OnInit is called when the EA is initialized on the chart
//...
}
```

On the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we need to initialize the indicator handles so that data values are assigned to them.

```
   handleSMA = iMA(_Symbol,PERIOD_M1,60,0,MODE_SMA,PRICE_CLOSE);  // Create an SMA (Simple Moving Average) indicator handle for the M1 timeframe
```

Here, we use the [iMA](https://www.mql5.com/en/docs/indicators/ima) function to create a handle for the Simple Moving Average (SMA) indicator. The handle will give us access to the SMA values for whatever trading symbol and timeframe we specify. Here, we set it up for the current trading symbol ( [\_Symbol](https://www.mql5.com/en/docs/check/symbol)) on a 1-minute chart ( [PERIOD\_M1](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes)), over a period of 60 bars. The parameter "0" indicates that there is no shift in the SMA calculation, while the demand to calculate the mode is specified with the constant [MODE\_SMA](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_ma_method); hence, we will be calculating a simple moving average. Finally, we specify the price type [PRICE\_CLOSE](https://www.mql5.com/en/docs/constants/indicatorconstants/prices), which means the indicator will be based on the closing prices of each bar. We store the resulting handle in the variable "handleSMA". Next, we create the handle for the Parabolic SAR indicator.

```
   handleSAR = iSAR(_Symbol,PERIOD_M1,0.02,0.2);                  // Create a SAR (Parabolic SAR) indicator handle for the M1 timeframe
```

Similarly, we create a handle for the Parabolic SAR (SAR) indicator using the [iSAR](https://www.mql5.com/en/docs/indicators/ISAR) function. The [iSAR](https://www.mql5.com/en/docs/indicators/ISAR) function returns the values of the Parabolic SAR for a specific trading symbol and timeframe. In our case, we use it for the current trading symbol ( [\_Symbol](https://www.mql5.com/en/docs/check/symbol)) on the 1-minute chart ( [PERIOD\_M1](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes)). The parameters 0.02 and 0.2 define the step and maximum value of the SAR, which means they control how sensitive the indicator will be to price movements. The resulting handle will let us access SAR data that will be central to our first trading strategy, which relies on market trends and price reversals.

By default, the indicator handle starts from 10 and progresses with an interval of 1. So if we print the indicator values, we should have 10 and 11 since we have two handles. To visualize their values, let us log them.

```
   Print("SMA Handle = ",handleSMA);
   Print("SAR Handle = ",handleSAR);
```

The values we get are as below:

![HANDLE VALUES](https://c.mql5.com/2/134/Screenshot_2024-08-25_162948.png)

Now we log the values correctly. Next, let us make sure that the indicator handles are indeed not empty.

```
   // Check if the handles for either the SMA or SAR are invalid (indicating failure)
   if (handleSMA == INVALID_HANDLE || handleSAR == INVALID_HANDLE){
      Print("ERROR: FAILED TO CREATE SMA/SAR HANDLE. REVERTING NOW");  // Print error message in case of failure
      return (INIT_FAILED);  // Return failure code, stopping the EA from running
   }
```

In this section, we set up a mechanism to catch critical errors that occur during the SMA and SAR indicator handle creation. If these handles are not created successfully, the Expert Advisor cannot function correctly. Thus, we check to see whether "handleSMA" or "handleSAR" has remained equal to [INVALID\_HANDLE](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants), which would indicate an unsuccessful initialization of either of these two indicators. If we find that either of these handles is invalid, we print an error message ("ERROR: FAILED TO CREATE SMA/SAR HANDLE. REVERTING NOW.") and return the [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events) value from the initialization function. These actions together constitute a graceful error-handling routine and serve to ensure that the EA can make unsafe trading decisions. Then, we need to set the storage arrays as time series. This is achieved via the following code snippet:

```
   // Configure the SMA and SAR data arrays to work as series, with the newest data at index 0
   ArraySetAsSeries(sma_data,true);
   ArraySetAsSeries(sar_data,true);
```

We set up our storage arrays to operate as time series data, with the most recent information located at index 0. We accomplish this using the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) function and the two arrays ("sma\_data" and "sar\_data") as the first argument, with true as the second argument to ensure the arrays are treated as series. This is a necessary step for accessing the latest indicator values at the ordinary decision times in the trading logic when the data represented by the indicator would have been spit out to the trading decision-making routine. The next part of the logic references both a series of current and past values for the two indicators. Finally, if we reach this point, it means that everything is initialized correctly and thus we return a successful initialization instance.

```
   return(INIT_SUCCEEDED);  // Return success code to indicate successful initialization
```

Up to this point, everything in the initialization section worked correctly. The full source code responsible for the program initialization is as follows:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){
   // OnInit is called when the EA is initialized on the chart

   handleSMA = iMA(_Symbol,PERIOD_M1,60,0,MODE_SMA,PRICE_CLOSE);  // Create an SMA (Simple Moving Average) indicator handle for the M1 timeframe
   handleSAR = iSAR(_Symbol,PERIOD_M1,0.02,0.2);                  // Create a SAR (Parabolic SAR) indicator handle for the M1 timeframe

   Print("SMA Handle = ",handleSMA);
   Print("SAR Handle = ",handleSAR);

   // Check if the handles for either the SMA or SAR are invalid (indicating failure)
   if (handleSMA == INVALID_HANDLE || handleSAR == INVALID_HANDLE){
      Print("ERROR: FAILED TO CREATE SMA/SAR HANDLE. REVERTING NOW");  // Print error message in case of failure
      return (INIT_FAILED);  // Return failure code, stopping the EA from running
   }

   // Configure the SMA and SAR data arrays to work as series, with the newest data at index 0
   ArraySetAsSeries(sma_data,true);
   ArraySetAsSeries(sar_data,true);

   return(INIT_SUCCEEDED);  // Return success code to indicate successful initialization
}
```

Next, we move on to the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, which is a function called when the program is deinitialized.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason){
   // OnDeinit is called when the EA is removed from the chart or terminated
//...
}
```

The [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function gets invoked when the Expert Advisor (EA) is removed from the chart or when the terminal shuts down. We have to use this event handler to ensure correct upkeep and resource management. When the EA terminates, we must release any handles to indicators we created in the initialization phase. If we didn't do this, we could be leaving behind memory locations that we used, which would be inefficient; we certainly did not want to risk leaving behind any resources that we didn't need. This is why [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) is important and why cleanup steps are critical in any programming environment.

```
   // Release the indicator handles to free up resources
   IndicatorRelease(handleSMA);
   IndicatorRelease(handleSAR);
```

We clean up every resource we set aside for use. To perform this cleanup, we use the [IndicatorRelease](https://www.mql5.com/en/docs/series/indicatorrelease) function, which we call separately for each handle we allocated and initialized, just as we stored them away when we first began using them. This ensures we free up the resources in the reverse order of their allocation. Here, we specifically look at the SMA and SAR indicators. The cleanup is crucial to maintaining the platform's performance, especially if you are using multiple Expert Advisors or running the platform for extended periods. Thus, the full source code for the resources free-up is as follows:

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason){
   // OnDeinit is called when the EA is removed from the chart or terminated

   // Release the indicator handles to free up resources
   IndicatorRelease(handleSMA);
   IndicatorRelease(handleSAR);
}
```

Next, we need to check for trading opportunities whenever there are price updates. This is achieved on the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
   // OnTick is called whenever there is a new market tick (price update)

//...

}
```

The event-handler function, [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick), executes and processes recent price information every time there is a new tick or a change in market conditions. It is an essential part of the operation of our Expert Advisor (EA) because it is where we run our trading logic, the trading conditions of which are, hopefully, structured to yield profitable trades. When the market data changes, we assess the current state of the market and make decisions regarding whether to open or close a position. The function executes as often as market conditions change, ensuring that our strategy operates in real-time and is responsive to current prices and changes in the values of our market indicators.

To stay updated with the current market conditions, we need to get the values of the current price quotes.

```
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);  // Get and normalize the Ask price
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);  // Get and normalize the Bid price
```

Here, we obtain the most current Ask and Bid prices for the traded symbol. To get these prices, we use the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function. For the Ask price, we specify [SYMBOL\_ASK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants), and for the Bid price, we specify [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants). After we obtain the prices, we use the [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) function to round the prices to the number of decimal places defined by [\_Digits](https://www.mql5.com/en/docs/check/digits). This step is crucial because it ensures that our trading operations are performed using prices that are both standardized and accurate. If we didn't round the prices, floating-point inaccuracies could yield misleading results in operation price calculations. We then copy the indicator values for use in analysis and trade operations.

```
   // Retrieve the last 3 values of the SMA indicator into the sma_data array
   if (CopyBuffer(handleSMA,0,0,3,sma_data) < 3){
      Print("ERROR: NOT ENOUGH DATA FROM SMA FOR FURTHER ANALYSIS. REVERTING");  // Print error if insufficient SMA data
      return;  // Exit the function if not enough data is available
   }
```

We use the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function to retrieve the three most recent values of the Simple Moving Average indicator. The [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function takes as input the SMA indicator handle, the SMA data buffer index, the starting position (in this case, the most recent value), the number of values to retrieve (three), and the array to store the data (sma\_data). After we have copied the SMA values into the array, we check to see if they copied correctly. If there is an error at this point, we print an error message to the log and exit the function to avoid proceeding with our trading logic based on potentially faulty or incomplete data.

Similarly, we use the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function to retrieve the SAR indicator data. The following code snippet is employed.

```
   // Retrieve the last 3 values of the SAR indicator into the sar_data array
   if (CopyBuffer(handleSAR,0,0,3,sar_data) < 3){
      Print("ERROR: NOT ENOUGH DATA FROM SAR FOR FURTHER ANALYSIS. REVERTING");  // Print error if insufficient SAR data
      return;  // Exit the function if not enough data is available
   }
```

All that you need to know right now is the starting index of the bar to store data from and the buffer index. Let us visualize them as below:

![BAR AND BUFFER INDICES](https://c.mql5.com/2/134/Screenshot_2024-08-25_173215.png)

Let us print their data to the journal so we can see if we correctly get their data.

```
   ArrayPrint(sma_data,_Digits," , ");
   ArrayPrint(sar_data,_Digits," , ");
```

We use the [ArrayPrint](https://www.mql5.com/en/docs/array/arrayprint) function to show the contents of the "sma\_data" and "sar\_data" arrays. [ArrayPrint](https://www.mql5.com/en/docs/array/arrayprint) is a basic function that prints the contents of an array to the terminal. It’s good for preparing simple visualizations of the data being worked with. To use the function, you need to pass it three pieces of information:

- The array you want to print ("sma\_data" or "sar\_data").
- The precision you want for the displayed values ( [\_Digits](https://www.mql5.com/en/docs/check/digits)), determines how many decimal places will be shown.
- The delimiter is to be used between the printed values (", ") in the output.

By printing the contents of the SMA and SAR arrays, we can keep track of the actual numbers our algorithm is using, which is helpful for debugging purposes. We can also check this data in a backtest to ascertain that we indeed get the required data.

![DATA CONFIRMATION](https://c.mql5.com/2/134/Screenshot_2024-08-25_175256.png)

From the picture, we can see that the data is correctly retrieved and stored. The data in red color represents the Moving Average data while the data in blue color represents the SAR data. From the crosshair, we are referencing the second bar and index 1. It is now clear that we get data 3 data values retrieved, just as requested. We can proceed to get the necessary bar values.

```
   // Get the low prices for the current and previous bars on the M1 timeframe
   double low0 = iLow(_Symbol,PERIOD_M1,0);  // Low of the current bar
   double low1 = iLow(_Symbol,PERIOD_M1,1);  // Low of the previous bar
```

Here, we extract the low prices from the current and previous bars on the 1-minute timeframe using the [iLow](https://www.mql5.com/en/docs/series/ilow) function. The first parameter, [\_Symbol](https://www.mql5.com/en/docs/check/symbol), automatically adjusts to the trading instrument for which the Expert Advisor is operating (for instance, EUR/USD). The second parameter, [PERIOD\_M1](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes), tells the system that we are working on the 1-minute chart; this is vital because the rapid-fire strategy is predicated on making short-term trades based on rapid price movements. The third parameter, 0, in "iLow(\_Symbol, PERIOD\_M1, 0)", indicates that we want the low price of the current bar that is in the process of being formed. Similarly, in "iLow(\_Symbol, PERIOD\_M1, 1)", the 1 signals that we want the low price of the previous completed bar. We store the retrieved values in the "double" data type variables "low0" and "low1" respectively.

Similarly, to retrieve the high prices for the current and previous bars, a similar logic is adopted.

```
   // Get the high prices for the current and previous bars on the M1 timeframe
   double high0 = iHigh(_Symbol,PERIOD_M1,0);  // High of the current bar
   double high1 = iHigh(_Symbol,PERIOD_M1,1);  // High of the previous bar
```

Here, the only thing that changes is the function used, [iHigh](https://www.mql5.com/en/docs/series/ihigh), which retrieves the high data for the specified bar index and timeframe. To confirm the data that we get is correct, let us print it again using the [Print](https://www.mql5.com/en/docs/common/print) function.

```
   Print("High bar 0 = ",high0,", High bar 1 = ",high1);
   Print("Low bar 0 = ",low0,", Low bar 1 = ",low1);
```

Data confirmation is as below:

![HIGH & LOW DATA CONFIRMATION](https://c.mql5.com/2/134/Screenshot_2024-08-25_182414.png)

That was a success. We can now move on to crafting the trading logic. However, we don't need to execute the trading logic on every tick. Thus, let us craft a mechanism to ensure we trade only once per bar.

```
   // Define a static variable to track the last time a signal was generated
   static datetime signalTime = 0;
   datetime currTime0 = iTime(_Symbol,PERIOD_M1,0);  // Get the time of the current bar
```

In this section, we create a variable, "signalTime", to keep track of when the last trading signal was generated. We make this variable [static](https://www.mql5.com/en/docs/basis/variables/static) to ensure that it maintains its value between calls to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function. If "signalTime" did not have static storage duration, it would be allocated each time the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) was called, and its value would be gone after [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) finished executing. We want the signal time to be set only once for each bar, and we want it to keep its value until the next bar starts. Thus, we set it to 0 at the beginning so that it has a value only after the first signal is generated. To prevent it from generating a second (or multiple) signal during the same bar, we will compare the time of the current bar with the value stored in the variable. Afterward, we can now define the trading logic.

```
   // Check for BUY signal conditions:
   // - Current SAR is below the current low (bullish)
   // - Previous SAR was above the previous high (bullish reversal)
   // - SMA is below the current Ask price (indicating upward momentum)
   // - No other positions are currently open (PositionsTotal() == 0)
   // - The signal hasn't already been generated on the current bar
   if (sar_data[0] < low0 && sar_data[1] > high1 && signalTime != currTime0
      && sma_data[0] < Ask && PositionsTotal() == 0){
      Print("BUY SIGNAL @ ",TimeCurrent());  // Print buy signal with timestamp
      signalTime = currTime0;  // Update the signal time to the current bar time
   }
```

Here, we verify the conditions for issuing a buy signal based on the simultaneous behavior of the Parabolic SAR and SMA indicators, alongside a couple of other constraints. To qualify for a bullish signal from the Parabolic SAR, the current SAR value ("sar\_data\[0\]") must be below the current low price ("low0"). At the same time, the previous SAR value ("sar\_data\[1\]") must have been above the previous high price ("high1"). To qualify for a bullish SMA signal, the SMA must be below the current Ask price ("sma\_data\[0\] < Ask").

In addition, we verify that there aren't any other open positions ( [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) == 0). We wouldn't want to open multiple trades at the same time. We then make sure that the signal hasn't been generated for the current bar by comparing "signalTime" with "currTime0". If all of these checks pass, we print a message announcing the generation of a buy signal. We also update the signal time variable to the current time. That neat little trick is what allows us to limit the signal generation to just one time per bar. In the end, what we're doing is making sure that the EA is only reacting to certain market conditions and isn't doing any unnecessary work. The results we get are as below:

![BUY SIGNAL](https://c.mql5.com/2/134/Screenshot_2024-08-25_185035.png)

Since we get the signal, let us then proceed to open buy positions.

```
      obj_Trade.Buy(0.01,_Symbol,Ask,Ask-150*_Point,Ask+100*_Point);  // Execute a buy order with a lot size of 0.01, stop loss and take profit
```

Here, we execute a buy order using the "Buy" method from the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class (represented by the "obj\_Trade" object that we created). Here's a breakdown of the parameters:

- 0.01: This is the lot size of the trade. Here, we are specifying a small position size of 0.01 lots.
- \_Symbol: This represents the trading symbol (currency pair, stock, etc.) that the Expert Advisor is attached to. The symbol is automatically detected by using the built-in [\_Symbol](https://www.mql5.com/en/docs/check/symbol) variable.
- Ask: This is the price at which the buy order will be executed, which is the current Ask price in the market. The Ask price is typically used for buying trades.
- Ask - 150 \* \_Point: This is the stop loss level. We set the stop loss 150 points below the Ask price to limit potential losses in case the market moves against the trade.
- Ask + 100 \* \_Point: This is the take profit level. We set the take profit 100 points above the Ask price, meaning that the trade will automatically close when the market reaches this profit level.

By placing this order, we are instructing the Expert Advisor to open a buy trade at the current Ask price, with predefined stop loss and take profit levels to manage risk and reward. The final code snippet for confirming buy signals and opening buy orders is as follows:

```
   // Check for BUY signal conditions:
   // - Current SAR is below the current low (bullish)
   // - Previous SAR was above the previous high (bullish reversal)
   // - SMA is below the current Ask price (indicating upward momentum)
   // - No other positions are currently open (PositionsTotal() == 0)
   // - The signal hasn't already been generated on the current bar
   if (sar_data[0] < low0 && sar_data[1] > high1 && signalTime != currTime0
      && sma_data[0] < Ask && PositionsTotal() == 0){
      Print("BUY SIGNAL @ ",TimeCurrent());  // Print buy signal with timestamp
      signalTime = currTime0;  // Update the signal time to the current bar time
      obj_Trade.Buy(0.01,_Symbol,Ask,Ask-150*_Point,Ask+100*_Point);  // Execute a buy order with a lot size of 0.01, stop loss and take profit
   }
```

To confirm a sell signal and open a sell order, similar logic applies.

```
   // Check for SELL signal conditions:
   // - Current SAR is above the current high (bearish)
   // - Previous SAR was below the previous low (bearish reversal)
   // - SMA is above the current Bid price (indicating downward momentum)
   // - No other positions are currently open (PositionsTotal() == 0)
   // - The signal hasn't already been generated on the current bar
   else if (sar_data[0] > high0 && sar_data[1] < low1 && signalTime != currTime0
      && sma_data[0] > Bid && PositionsTotal() == 0){
      Print("SELL SIGNAL @ ",TimeCurrent());  // Print sell signal with timestamp
      signalTime = currTime0;  // Update the signal time to the current bar time
      obj_Trade.Sell(0.01,_Symbol,Bid,Bid+150*_Point,Bid-100*_Point);  // Execute a sell order with a lot size of 0.01, stop loss and take profit
   }
```

Here, we implement a sell trade logic that will execute when specific market conditions are present. We first check whether the Parabolic SAR (SAR) indicator suggests a bearish trend. We do this by comparing the most recent SAR data to the bar's prices. The SAR should be above the bar's high for the current candle ("sar\_data\[0\] > high0"), and the previous SAR value was below the previous bar's low price ("sar\_data\[1\] < low1"), which indicates a possible bearish reversal. The SAR's trend indication can be "safe to trade" momentum. Then we check the SMA. It should be above the Bid price ("sma\_data\[0\] > Bid"), which indicates a "safe to trade" possible downtrend.

When all the aforementioned conditions have been met, we signal a sell and log it ("Print("SELL SIGNAL @ ", TimeCurrent())"). The "signalTime" variable records the current bar's time; this is important because we must ensure that we do not issue more than one sell signal per bar. To execute the trade, we use the object function "obj\_Trade.Sell". We systematically enter the market at points in time when potential bearish reversals may occur. We do this with an eye toward the risk that we are taking. And we manage that risk using stop loss and take profit orders. The sell confirmation setup is as below:

![SELL SETUP CONFIRMATION](https://c.mql5.com/2/134/Screenshot_2024-08-25_191532.png)

Up to this point, it is now clear that we have correctly crafted the Rapid-Fire Expert Advisor algorithm. Thus, the full [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) source code responsible for confirming trades and opening positions is as follows:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
   // OnTick is called whenever there is a new market tick (price update)

   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);  // Get and normalize the Ask price
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);  // Get and normalize the Bid price

   // Retrieve the last 3 values of the SMA indicator into the sma_data array
   if (CopyBuffer(handleSMA,0,0,3,sma_data) < 3){
      Print("ERROR: NOT ENOUGH DATA FROM SMA FOR FURTHER ANALYSIS. REVERTING");  // Print error if insufficient SMA data
      return;  // Exit the function if not enough data is available
   }

   // Retrieve the last 3 values of the SAR indicator into the sar_data array
   if (CopyBuffer(handleSAR,0,0,3,sar_data) < 3){
      Print("ERROR: NOT ENOUGH DATA FROM SAR FOR FURTHER ANALYSIS. REVERTING");  // Print error if insufficient SAR data
      return;  // Exit the function if not enough data is available
   }

   //ArrayPrint(sma_data,_Digits," , ");
   //ArrayPrint(sar_data,_Digits," , ");

   // Get the low prices for the current and previous bars on the M1 timeframe
   double low0 = iLow(_Symbol,PERIOD_M1,0);  // Low of the current bar
   double low1 = iLow(_Symbol,PERIOD_M1,1);  // Low of the previous bar

   // Get the high prices for the current and previous bars on the M1 timeframe
   double high0 = iHigh(_Symbol,PERIOD_M1,0);  // High of the current bar
   double high1 = iHigh(_Symbol,PERIOD_M1,1);  // High of the previous bar

   //Print("High bar 0 = ",high0,", High bar 1 = ",high1);
   //Print("Low bar 0 = ",low0,", Low bar 1 = ",low1);

   // Define a static variable to track the last time a signal was generated
   static datetime signalTime = 0;
   datetime currTime0 = iTime(_Symbol,PERIOD_M1,0);  // Get the time of the current bar

   // Check for BUY signal conditions:
   // - Current SAR is below the current low (bullish)
   // - Previous SAR was above the previous high (bullish reversal)
   // - SMA is below the current Ask price (indicating upward momentum)
   // - No other positions are currently open (PositionsTotal() == 0)
   // - The signal hasn't already been generated on the current bar
   if (sar_data[0] < low0 && sar_data[1] > high1 && signalTime != currTime0
      && sma_data[0] < Ask && PositionsTotal() == 0){
      Print("BUY SIGNAL @ ",TimeCurrent());  // Print buy signal with timestamp
      signalTime = currTime0;  // Update the signal time to the current bar time
      obj_Trade.Buy(0.01,_Symbol,Ask,Ask-150*_Point,Ask+100*_Point);  // Execute a buy order with a lot size of 0.01, stop loss and take profit
   }

   // Check for SELL signal conditions:
   // - Current SAR is above the current high (bearish)
   // - Previous SAR was below the previous low (bearish reversal)
   // - SMA is above the current Bid price (indicating downward momentum)
   // - No other positions are currently open (PositionsTotal() == 0)
   // - The signal hasn't already been generated on the current bar
   else if (sar_data[0] > high0 && sar_data[1] < low1 && signalTime != currTime0
      && sma_data[0] > Bid && PositionsTotal() == 0){
      Print("SELL SIGNAL @ ",TimeCurrent());  // Print sell signal with timestamp
      signalTime = currTime0;  // Update the signal time to the current bar time
      obj_Trade.Sell(0.01,_Symbol,Bid,Bid+150*_Point,Bid-100*_Point);  // Execute a sell order with a lot size of 0.01, stop loss and take profit
   }
}
//+------------------------------------------------------------------+
```

Cheers to us for creating the program. Next, we need to do the testing of the Expert Advisor and optimize it for maximum gains. This is done in the next section.

### Testing and Optimization

This section concentrates on the testing and optimization of our rapid-fire trading strategy using the MetaTrader 5 Strategy Tester. Testing is essential because we need to ensure that our Expert Advisor (EA) is functioning as it should and that it is performing well in a variety of market conditions. To begin testing our EA, we need to load it into the Strategy Tester, choose the desired trading symbol and timeframe (we have specified the M1 timeframe in our code), and set the testing period. By backtesting our EA, we can get a historical simulation of how our EA would have performed with the specified strategy and parameters. Of course, we try to identify any potential problems that could cause the EA to misfunction in a live trading scenario. To open the strategy tester, click on view, and select Strategy Tester.

![OPEN STRATEGY TESTER](https://c.mql5.com/2/134/Screenshot_2024-08-25_202238.png)

Once it is opened, navigate to the overview tab, select "Visualize", switch to the settings tab, load the program, and set the desired settings. For our case, we will have the testing period for just this month so that we work with small data.

![TESTER SETTINGS](https://c.mql5.com/2/134/Screenshot_2024-08-25_203204.png)

Upon testing, we have the following results:

![TESTER RESULTS](https://c.mql5.com/2/134/Screenshot_2024-08-25_203754.png)

We need to take a close look at the results from the testing phase, concentrating on vital performance metrics like total net profit, maximum drawdown, and percentage of winning trades. These numbers will tell us how well (and how poorly) our strategy performs and give us some idea of its overall robustness. We'll also want to examine trade reports that the Strategy Tester produces. These reports will allow us to see the sorts of trades our EA is making (or not making) and give us a chance to understand its "reasoning" when it encounters various market conditions. If we're not happy with the results, we'll have to rework the strategy to look for better-performing parameter sets for the SMA and SAR indicators. Or we'll have to dig deeper into the strategy's overall logic to find out why it isn't performing any better than it does.

To find the best trading strategy, we have to tune it to the market. To do that, we use a method called "strategy optimization." Strategy optimization is not something we do once and forget about. We have to see how the system behaves under different configurations and then pick the configuration that gives us the most consistent results over all the tests we perform. When testing a strategy in this way, we are effectively finding the optimal parameters and using them as keys to unlock the trading strategy we're examining. After we find these keys, we have to run the strategy again and test it over a walkforward period. Otherwise, we're no better than someone who just guesses at the market's future direction. To optimize the system, we will need to have some inputs from where the iterations can take place. The following logic will be implemented.

```
input int sl_points = 150; // Stoploss points
input int tp_points = 100; // Takeprofit points
```

Upon optimization, we have the following results:

![OPTIMIZATION RESULTS](https://c.mql5.com/2/134/Screenshot_2024-08-25_210042.png)

The optimized results graph resembles the one below:

![OPTIMIZATION GRAPH](https://c.mql5.com/2/134/Screenshot_2024-08-25_210244.png)

Finally, the test report is as shown below:

![FINAL TEST REPORT](https://c.mql5.com/2/134/Screenshot_2024-08-25_211112.png)

### Conclusion

In this article, we discussed the development of a rapid-fire trading Expert Advisor (EA) in MQL5, which executes short, quick trades. The strategy relies on two main technical indicators: the Parabolic SAR, which signals trend reversals, and the Simple Moving Average (SMA), which gauges market momentum. These indicators form the core of the EA's trading logic.

We covered the implementation details, including configuring the indicators, setting up data handles, and managing trades to ensure the EA interacts effectively with the market. Finally, we emphasized the importance of testing and optimizing the EA using MetaTrader 5's Strategy Tester to ensure it meets our trading objectives. Despite being an automated system, the development process closely mirrors that of manual trading strategies.

Disclaimer: The information illustrated in this article is only for educational purposes. It is just intended to show insights on how to create a Rapid-Fire Strategy Expert Advisor (EA) based on the Price Action approach and thus should be used as a base for creating a better expert advisor with more optimization and data extraction taken into account. The information presented does not guarantee any trading results.

We do hope that you found the article helpful, fun, and easy to understand, in a way that you can make use of the presented knowledge in your development of future expert advisors. Technically, this eases your way of analyzing the market based on the Price Action approach and particularly the Rapid-Fire strategy. Enjoy.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15698.zip "Download all attachments in the single ZIP archive")

[RAPID\_FIRE\_EA.mq5](https://www.mql5.com/en/articles/download/15698/rapid_fire_ea.mq5 "Download RAPID_FIRE_EA.mq5")(6.6 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/472207)**
(9)


![viper2k16](https://c.mql5.com/avatar/avatar_na2.png)

**[viper2k16](https://www.mql5.com/en/users/viper2k16)**
\|
19 Sep 2024 at 14:51

Very interesting, found this strategy after testing it manually!

Would be interesting to add the possibility of having risk management settings, by fixed lot size or a percentage of balance, thanks!


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
22 Sep 2024 at 20:52

**viper2k16 [#](https://www.mql5.com/en/forum/472207#comment_54618843):**

Very interesting, found this strategy after testing it manually!

Would be interesting to add the possibility of having risk management settings, by fixed lot size or a percentage of balance, thanks!

We're glad to hear that. We'll consider the idea. Thanks.

![Line00](https://c.mql5.com/avatar/2021/1/60049DE9-DE73.png)

**[Line00](https://www.mql5.com/en/users/line00)**
\|
20 May 2025 at 12:35

Great article!!! After reading and applying it, there are a lot of ideas for EA development based on this strategy.


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
20 May 2025 at 19:12

**Line00 [#](https://www.mql5.com/en/forum/472207#comment_56742425):**

Great article!!! After reading and applying it, there are a lot of ideas for EA development based on this strategy.

Thanks very much and welcome.

![Vadim Perovskikh](https://c.mql5.com/avatar/2014/6/53AC31D8-9161.jpg)

**[Vadim Perovskikh](https://www.mql5.com/en/users/perovskikh)**
\|
19 Oct 2025 at 08:35

Very often [parabolic](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/psar "MetaTrader 5 Help: Parabolic SAR Indicator") sar breaks through the lp on the news. And the price goes further.


![Developing a Replay System (Part 45): Chart Trade Project (IV)](https://c.mql5.com/2/74/Desenvolvendo_um_sistema_de_Replay_Parte_45___LOGO.png)[Developing a Replay System (Part 45): Chart Trade Project (IV)](https://www.mql5.com/en/articles/11701)

The main purpose of this article is to introduce and explain the C\_ChartFloatingRAD class. We have a Chart Trade indicator that works in a rather interesting way. As you may have noticed, we still have a fairly small number of objects on the chart, and yet we get the expected functionality. The values present in the indicator can be edited. The question is, how is this possible? This article will start to make things clearer.

![MQL5 Wizard Techniques you should know (Part 35): Support Vector Regression](https://c.mql5.com/2/91/MQL5_Wizard_Techniques_you_should_know_Part_35__LOGO.png)[MQL5 Wizard Techniques you should know (Part 35): Support Vector Regression](https://www.mql5.com/en/articles/15692)

Support Vector Regression is an idealistic way of finding a function or ‘hyper-plane’ that best describes the relationship between two sets of data. We attempt to exploit this in time series forecasting within custom classes of the MQL5 wizard.

![Reimagining Classic Strategies (Part VII) : Forex Markets And Sovereign Debt Analysis on the USDJPY](https://c.mql5.com/2/91/Reimagining_Classic_Strategies_Part_VII___LOGO.png)[Reimagining Classic Strategies (Part VII) : Forex Markets And Sovereign Debt Analysis on the USDJPY](https://www.mql5.com/en/articles/15719)

In today's article, we will analyze the relationship between future exchange rates and government bonds. Bonds are among the most popular forms of fixed income securities and will be the focus of our discussion.Join us as we explore whether we can improve a classic strategy using AI.

![Developing a Replay System (Part 44): Chart Trade Project (III)](https://c.mql5.com/2/73/Desenvolvendo_um_sistema_de_Replay_Parte_44___LOGO.png)[Developing a Replay System (Part 44): Chart Trade Project (III)](https://www.mql5.com/en/articles/11690)

In the previous article I explained how you can manipulate template data for use in OBJ\_CHART. In that article, I only outlined the topic without going into details, since in that version the work was done in a very simplified way. This was done to make it easier to explain the content, because despite the apparent simplicity of many things, some of them were not so obvious, and without understanding the simplest and most basic part, you would not be able to truly understand the entire picture.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/15698&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048819994797645642)

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