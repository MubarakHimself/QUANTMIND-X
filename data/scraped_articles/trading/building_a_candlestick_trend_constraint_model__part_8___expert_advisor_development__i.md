---
title: Building a Candlestick Trend Constraint Model (Part 8): Expert Advisor Development (I)
url: https://www.mql5.com/en/articles/15321
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 8
scraped_at: 2026-01-22T17:45:02.678138
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ommlldbhwfgemolduryigqgmzrahjmbk&ssn=1769093100115708426&ssn_dr=0&ssn_sr=0&fv_date=1769093100&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15321&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Building%20a%20Candlestick%20Trend%20Constraint%20Model%20(Part%208)%3A%20Expert%20Advisor%20Development%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909310099147854&fz_uniq=5049351423986084332&sv=2552)

MetaTrader 5 / Tester


### Contents:

- [Introduction](https://www.mql5.com/en/articles/15321#para2)
- [Solution to the prior challenges on drawing risk and reward rectangles](https://www.mql5.com/en/articles/15321#para3)
- [Creating an Expert Advisor that works based an indicato](https://www.mql5.com/en/articles/15321#para4) r
- [Testing](https://www.mql5.com/en/articles/15321#para5)
- [Conclusion](https://www.mql5.com/en/articles/15321#para6)

### Introduction

MetaEditor software includes a compiler that effectively manages errors detected during profiling attempts. This tool helped me uncover why the previous version failed to display the risk-reward rectangles as intended. Although the program compiled successfully, the issue was not with the code itself. Instead, the challenge lay in the fact that nothing was displayed within the range of the look back candlesticks, primarily due to specific technicalities.

1. The look back candle value was set too high at 5,000 bars by default.
2. Having multiple buffers in a single program increases the complexity of calculations, which can slow down the display of the indicator chart window.

After briefly discussing how we resolved the issues encountered, we'll move on to the main goal of this article: developing an Expert Advisor based on the refined Trend Constraint Indicator. Below is an image showing how a separate script successfully addressed the problem we originally intended to solve with the main indicator.

![Risk-Reward ratio drawn for Moving Average crossover ](https://c.mql5.com/2/88/R-R_rectangle_drawn.png)

Risk and Reward Rectangles automatically drawn using rectangle

### Solution to the prior challenges on drawing risk and reward rectangles

To address the challenges in the indicator program:

1. We reduced the look-back period from 5000 bars to 1000 bars, therefore significantly decreased the amount of data to be calculated.
2. We aimed to reduce the program's workload by creating a standalone script as part of the tool set. This script specifically checks the conditions handled by Buffer 6 and Buffer 7 in the indicator. Once these conditions are met, the script draws the necessary risk-reward rectangles and places lines with price labels for the entry price, Stop Loss, and Take Profit. However, it’s important to note that the script performs a one-time task and does not run continuously. The user must manually add the script to the chart to visualize the trade levels as represented by the drawn objects and price markings.

Below is an image showing the launch of the script program:

![Trend Constraint  R-R script launch](https://c.mql5.com/2/88/ShareX_dIM1CGrYDN.gif)

Trend Constraint R-R script: For Drawing Risk and Reward Rectangle when a Moving Average crossover happens

By isolating this feature, we ensured that our indicator runs smoothly, avoiding any freezing of the computer or trading terminal. Incorporating risk-reward rectangles and marking the exit levels allows traders to visually assess the trade direction and targets in advance, enabling manual trading without the need for an Expert Advisor. The script program with the required logic worked flawlessly, perfectly. Here is our full script program.

```
//+------------------------------------------------------------------+
//|                                        Trend Constraint R-R.mq5  |
//|                                                  Script program  |
//+------------------------------------------------------------------+
#property strict
#property script_show_inputs
#property copyright "2024 Clemence Benjamin"
#property version "1.00"
#property link "https://www.mql5.com/en/users/billionaire2024/seller"
#property description "A script program for drawing risk and rewars rectangles based on Moving Averaage crossover."

//--- input parameters
input int FastMAPeriod = 14;
input int SlowMAPeriod = 50;
input double RiskHeightPoints = 5000.0; // Default height of the risk rectangle in points
input double RewardHeightPoints = 15000.0; // Default height of the reward rectangle in points
input color RiskColor = clrIndianRed; // Default risk color
input color RewardColor = clrSpringGreen; // Default reward color
input int MaxBars = 500; // Maximum bars to process
input int RectangleWidth = 10; // Width of the rectangle in bars
input bool FillRectangles = true; // Option to fill rectangles
input int FillTransparency = 128; // Transparency level (0-255), 128 is 50% transparency

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  //--- delete existing rectangles and lines
  DeleteExistingObjects();

  //--- declare and initialize variables
  int i, limit;
  double FastMA[], SlowMA[];
  double closePrice, riskLevel, rewardLevel;

  //--- calculate moving averages
  if (iMA(NULL, 0, FastMAPeriod, 0, MODE_SMA, PRICE_CLOSE) < 0 || iMA(NULL, 0, SlowMAPeriod, 0, MODE_SMA, PRICE_CLOSE) < 0)
  {
    Print("Error in calculating moving averages.");
    return;
  }

  ArraySetAsSeries(FastMA, true);
  ArraySetAsSeries(SlowMA, true);

  CopyBuffer(iMA(NULL, 0, FastMAPeriod, 0, MODE_SMA, PRICE_CLOSE), 0, 0, MaxBars, FastMA);
  CopyBuffer(iMA(NULL, 0, SlowMAPeriod, 0, MODE_SMA, PRICE_CLOSE), 0, 0, MaxBars, SlowMA);

  limit = MathMin(ArraySize(FastMA), ArraySize(SlowMA));

  for (i = 1; i < limit - 1; i++)
  {
    //--- check for crossover
    if (FastMA[i] > SlowMA[i] && FastMA[i - 1] <= SlowMA[i - 1])
    {
      //--- long position entry point (bullish crossover)
      closePrice = iClose(NULL, 0, i);
      riskLevel = closePrice + RiskHeightPoints * Point();
      rewardLevel = closePrice - RewardHeightPoints * Point();

      //--- draw risk rectangle
      DrawRectangle("Risk_" + IntegerToString(i), i, closePrice, i - RectangleWidth, riskLevel, RiskColor);

      //--- draw reward rectangle
      DrawRectangle("Reward_" + IntegerToString(i), i, closePrice, i - RectangleWidth, rewardLevel, RewardColor);

      //--- draw entry, stop loss, and take profit lines
      DrawPriceLine("Entry_" + IntegerToString(i), i, closePrice, clrBlue, "Entry: " + DoubleToString(closePrice, _Digits));
      DrawPriceLine("StopLoss_" + IntegerToString(i), i, riskLevel, clrRed, "Stop Loss: " + DoubleToString(riskLevel, _Digits));
      DrawPriceLine("TakeProfit_" + IntegerToString(i), i, rewardLevel, clrGreen, "Take Profit: " + DoubleToString(rewardLevel, _Digits));
    }
    else if (FastMA[i] < SlowMA[i] && FastMA[i - 1] >= SlowMA[i - 1])
    {
      //--- short position entry point (bearish crossover)
      closePrice = iClose(NULL, 0, i);
      riskLevel = closePrice - RiskHeightPoints * Point();
      rewardLevel = closePrice + RewardHeightPoints * Point();

      //--- draw risk rectangle
      DrawRectangle("Risk_" + IntegerToString(i), i, closePrice, i - RectangleWidth, riskLevel, RiskColor);

      //--- draw reward rectangle
      DrawRectangle("Reward_" + IntegerToString(i), i, closePrice, i - RectangleWidth, rewardLevel, RewardColor);

      //--- draw entry, stop loss, and take profit lines
      DrawPriceLine("Entry_" + IntegerToString(i), i, closePrice, clrBlue, "Entry: " + DoubleToString(closePrice, _Digits));
      DrawPriceLine("StopLoss_" + IntegerToString(i), i, riskLevel, clrRed, "Stop Loss: " + DoubleToString(riskLevel, _Digits));
      DrawPriceLine("TakeProfit_" + IntegerToString(i), i, rewardLevel, clrGreen, "Take Profit: " + DoubleToString(rewardLevel, _Digits));
    }
  }
}

//+------------------------------------------------------------------+
//| Function to delete existing rectangles and lines                 |
//+------------------------------------------------------------------+
void DeleteExistingObjects()
{
  int totalObjects = ObjectsTotal(0, 0, -1);
  for (int i = totalObjects - 1; i >= 0; i--)
  {
    string name = ObjectName(0, i, 0, -1);
    if (StringFind(name, "Risk_") >= 0 || StringFind(name, "Reward_") >= 0 ||
        StringFind(name, "Entry_") >= 0 || StringFind(name, "StopLoss_") >= 0 ||
        StringFind(name, "TakeProfit_") >= 0)
    {
      ObjectDelete(0, name);
    }
  }
}

//+------------------------------------------------------------------+
//| Function to draw rectangles                                      |
//+------------------------------------------------------------------+
void DrawRectangle(string name, int startBar, double startPrice, int endBar, double endPrice, color rectColor)
{
  if (ObjectFind(0, name) >= 0)
    ObjectDelete(0, name);

  datetime startTime = iTime(NULL, 0, startBar);
  datetime endTime = (endBar < 0) ? (TimeCurrent() + (PeriodSeconds() * (-endBar))) : iTime(NULL, 0, endBar);

  if (!ObjectCreate(0, name, OBJ_RECTANGLE, 0, startTime, startPrice, endTime, endPrice))
    Print("Failed to create rectangle: ", name);

  // Set the color with transparency (alpha value)
  int alphaValue = FillTransparency; // Adjust transparency level (0-255)
  color fillColor = rectColor & 0x00FFFFFF | (alphaValue << 24); // Combine alpha with RGB

  ObjectSetInteger(0, name, OBJPROP_COLOR, rectColor);
  ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
  ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
  ObjectSetInteger(0, name, OBJPROP_BACK, true); // Set to background

  if (FillRectangles)
  {
    ObjectSetInteger(0, name, OBJPROP_COLOR, fillColor); // Fill color with transparency
  }
  else
  {
    ObjectSetInteger(0, name, OBJPROP_COLOR, rectColor & 0x00FFFFFF); // No fill color
  }
}

//+------------------------------------------------------------------+
//| Function to draw price lines                                     |
//+------------------------------------------------------------------+
void DrawPriceLine(string name, int barIndex, double price, color lineColor, string labelText)
{
  datetime time = iTime(NULL, 0, barIndex);
  datetime endTime = (barIndex - 2 * RectangleWidth < 0) ? (TimeCurrent() + (PeriodSeconds() * (-barIndex - 2 * RectangleWidth))) : iTime(NULL, 0, barIndex - 2 * RectangleWidth); // Extend line to the right

  if (!ObjectCreate(0, name, OBJ_TREND, 0, time, price, endTime, price))
    Print("Failed to create price line: ", name);

  ObjectSetInteger(0, name, OBJPROP_COLOR, lineColor);
  ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
  ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
  ObjectSetInteger(0, name, OBJPROP_BACK, true); // Set to background

  // Create text label
  string labelName = name + "_Label";
  if (ObjectFind(0, labelName) >= 0)
    ObjectDelete(0, labelName);

  if (!ObjectCreate(0, labelName, OBJ_TEXT, 0, endTime, price))
    Print("Failed to create label: ", labelName);

  ObjectSetInteger(0, labelName, OBJPROP_COLOR, lineColor);
  ObjectSetInteger(0, labelName, OBJPROP_ANCHOR, ANCHOR_LEFT);
  ObjectSetString(0, labelName, OBJPROP_TEXT, labelText);
  ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 10);
  ObjectSetInteger(0, labelName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
  ObjectSetInteger(0, labelName, OBJPROP_XOFFSET, 5);
  ObjectSetInteger(0, labelName, OBJPROP_YOFFSET, 0);
}
```

Let's thoroughly discuss the script's performance before proceeding with the development of the Expert Advisor:

- We, allowed for the customization of key aspects of the trading strategy by defining input parameters. We provided options to adjust the periods of the fast and slow-moving averages, set the dimensions and colors of the risk and reward rectangles, determine the maximum number of bars to process, and choose whether to fill the rectangles with color. This setup enables the script to be tailored to different trading strategies according to user preferences and that formed the input parameter section of our program:

```
//---- Input Parameters
input int FastMAPeriod = 14;
input int SlowMAPeriod = 50;
input double RiskHeightPoints = 5000.0;
input double RewardHeightPoints = 15000.0;
input color RiskColor = clrIndianRed;
input color RewardColor = clrSpringGreen;
input int MaxBars = 500;
input int RectangleWidth = 10;
input bool FillRectangles = true;
input int FillTransparency = 128;
```

- In the (OnStart) function, we made sure the chart stayed clean by programming the script to first delete any existing risk/reward rectangles and price lines. We then made it to calculate the fast and slow-moving averages using the (iMA) function and stored these values in arrays for further processing. As the script loops through the bars on the chart, we established conditions for detecting bullish crossovers, where the fast-moving average crossed above the slow one. When these conditions are met, the script calculates the entry price, risk level (stop loss), and reward level (take profit). It then draws rectangles and price lines on the chart, effectively marking these critical trading levels, as we will explain sub-code snippets further below this:

```
//----Onstart Function
void OnStart()
{
  //--- delete existing rectangles and lines
  DeleteExistingObjects();

  //--- declare and initialize variables
  int i, limit;
  double FastMA[], SlowMA[];
  double closePrice, riskLevel, rewardLevel;

  //--- calculate moving averages
  if (iMA(NULL, 0, FastMAPeriod, 0, MODE_SMA, PRICE_CLOSE) < 0 || iMA(NULL, 0, SlowMAPeriod, 0, MODE_SMA, PRICE_CLOSE) < 0)
  {
    Print("Error in calculating moving averages.");
    return;
  }
```

- To maintain clarity on the chart, we developed the "Delete Existing Objects" function, which the script uses to remove any previously drawn objects related to trading signals. By checking the names of all objects on the chart, the script ensured that only the most recent and relevant information was displayed, keeping the chart focused and free of clutter:

```
//---- DeleteAllExistingObjects Function
void DeleteExistingObjects()
{
  int totalObjects = ObjectsTotal(0, 0, -1);
  for (int i = totalObjects - 1; i >= 0; i--)
  {
    string name = ObjectName(0, i, 0, -1);
    if (StringFind(name, "Risk_") >= 0 || StringFind(name, "Reward_") >= 0 ||
        StringFind(name, "Entry_") >= 0 || StringFind(name, "StopLoss_") >= 0 ||
        StringFind(name, "TakeProfit_") >= 0)
    {
      ObjectDelete(0, name);
    }
  }
}
```

- In the "Draw Rectangle" function, we ensured that the script could visually represent risk and reward levels by first removing any existing rectangles with the same name to avoid duplication. We then had the script calculate the start and end times for the rectangles based on the bar indices, and we carefully set the colors and transparency levels. This allowed the rectangles to stand out on the chart without obscuring other important details:

```
///---Draw rectangle function
void DrawRectangle(string name, int startBar, double startPrice, int endBar, double endPrice, color rectColor)
{
  if (ObjectFind(0, name) >= 0)
    ObjectDelete(0, name);

  datetime startTime = iTime(NULL, 0, startBar);
  datetime endTime = (endBar < 0) ? (TimeCurrent() + (PeriodSeconds() * (-endBar))) : iTime(NULL, 0, endBar);

  if (!ObjectCreate(0, name, OBJ_RECTANGLE, 0, startTime, startPrice, endTime, endPrice))
    Print("Failed to create rectangle: ", name);

  int alphaValue = FillTransparency;
  color fillColor = rectColor & 0x00FFFFFF | (alphaValue << 24);

  ObjectSetInteger(0, name, OBJPROP_COLOR, rectColor);
  ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
  ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
  ObjectSetInteger(0, name, OBJPROP_BACK, true);

  if (FillRectangles)
  {
    ObjectSetInteger(0, name, OBJPROP_COLOR, fillColor);
  }
  else
  {
    ObjectSetInteger(0, name, OBJPROP_COLOR, rectColor & 0x00FFFFFF);
  }
}
```

- Finally, we implemented the "Draw Price Line" function to instruct the script to add horizontal lines at the entry, stop loss, and take profit levels. The script extended these lines across the chart and added text labels that displayed the corresponding price levels. This provided a visual reference, allowing users to quickly identify and manage key points for trades based on the signals generated by the moving averages:

```
///---- Draw Price Lines Function
void DrawPriceLine(string name, int barIndex, double price, color lineColor, string labelText)
{
  datetime time = iTime(NULL, 0, barIndex);
  datetime endTime = (barIndex - 2 * RectangleWidth < 0) ? (TimeCurrent() + (PeriodSeconds() * (-barIndex - 2 * RectangleWidth))) : iTime(NULL, 0, barIndex - 2 * RectangleWidth);

  if (!ObjectCreate(0, name, OBJ_TREND, 0, time, price, endTime, price))
    Print("Failed to create price line: ", name);

  ObjectSetInteger(0, name, OBJPROP_COLOR, lineColor);
  ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
  ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
  ObjectSetInteger(0, name, OBJPROP_BACK, true);

  string labelName = name + "_Label";
  if (ObjectFind(0, labelName) >= 0)
    ObjectDelete(0, labelName);

  if (!ObjectCreate(0, labelName, OBJ_TEXT, 0, endTime, price))
    Print("Failed to create label: ", labelName);

  ObjectSetInteger(0, labelName, OBJPROP_COLOR, lineColor);
  ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 10);
  ObjectSetString(0, labelName, OBJPROP_TEXT, labelText);
  ObjectSetInteger(0, labelName, OBJPROP_BACK, true);
}
```

Now as part of the trading kit, this script can be launched regularly to see the visualized trade levels, of the past and current time. We now move own to create our unique Expert Advisor. I will focus on explaining the whole development to the final working EA. In this article, we will just focus on making it work along the indicator that we made previous, Trend Constraint V1.09.

### Creating an Expert Advisor that works based on an indicator:

To create an expert advisor (EA) in MQL5 using a custom indicator we need to ensure that the custom indicator, file (.ex5) is available in the "Indicators" folder within the MetaTrader 5 platform, in this case it's Trend Constraint V1.09. Using the MetaEditor we can write our EA, incorporating MQL5 functions to access the indicator's buffer values. We utilize the (iCustom()) function to call the custom indicator within the EA, specifying the necessary parameters, such as symbol and timeframe.

To extract data from the indicator buffers, we use the (CopyBuffer()) function, which retrieves the buffer values you intend to analyze for trading signals. We then implement our trading logic based on these buffer values, defining conditions to open, close, or modify orders according to our strategy. Integrate risk management features, like stop-loss and take-profit, for prudent trade management. On back-testing the EA using MetaTrader 5 Strategy Tester to evaluate its performance and fine-tune its parameters. Finally, we will verify the EA's functionality in a demo account environment before transitioning to live trading to ensure it operates effectively under real market conditions.

We can start by launching an Expert Advisor template in MetaEditor and then develop or modify it based on the example shown in this image:

![Launch an EA template in MetaEditor](https://c.mql5.com/2/88/launch_ea_tmplate.png)

Launching an Expert Advisor template in MetaEditor

To guide you through the construction of our Expert Advisor (EA), we'll break down the process into six stages. As we progress, I recommend typing the code snippets directly into MetaEditor. This hands-on approach will help you better understand and internalize the steps, especially if you're new to EA development.

**1\. Header and Metadata**

In the header section, we define the name and purpose of our Expert Advisor (EA). By including copyright information, a link to our profile, and specifying the version, we ensure that our EA is easily identifiable and traceable. This metadata helps us and others understand the EA's origin and purpose, especially when it's shared or modified:

```
//You can replace the author details with yours.
//+------------------------------------------------------------------+
//|                                    Trend Constraint Expert.mq5   |
//|                                Copyright 2024, Clemence Benjamin |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property strict
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.0"
#property description "An Expert based on the buffer6 and buffer7 of Trend Constraint V1.09"
```

**2\. Input Parameters**

Here, we define the key input parameters that allow us to customize the EA's behavior without altering the code itself. By setting parameters such as (Lots, Slippage, Stop Loss, Take Profit, and Magic Number), we make the EA flexible and adaptable to different trading strategies. The magic number is particularly important, as it allows us to uniquely identify trades made by this EA, which is crucial when multiple EAs or manual trades are involved:

```
///----------- EA inputs parameters for customizations
input double Lots = 0.1;          // Lot size
input int Slippage = 3;           // Slippage
input double StopLoss = 50;       // Stop Loss in points
input double TakeProfit = 100;    // Take Profit in points
input int MagicNumber = 123456;   // Magic number for orders
```

**3\. Initialization Function (OnInit)**

In the OnInit function, we set the stage for the EA’s operation by initializing the necessary components. We begin by attempting to obtain a handle for our custom indicator, "Trend Constraint V1.09." This handle allows us to interact with the indicator programmatically. If the handle is successfully obtained, we proceed to set up the buffer arrays (Buffer6 and Buffer7) in series, enabling us to store and manipulate the indicator values. If, however, the handle cannot be retrieved, we ensure that the EA returns an initialization failure, with an error message to help us diagnose the issue:

```
////-------Initialization Function
int OnInit()
  {
   //--- Get the indicator handle
   indicator_handle = iCustom(Symbol(), PERIOD_CURRENT, "Trend Constraint V1.09");
   if (indicator_handle < 0)
     {
      Print("Failed to get the indicator handle. Error: ", GetLastError());
      return(INIT_FAILED);
     }

   //--- Set the buffer arrays as series
   ArraySetAsSeries(Buffer6, true);
   ArraySetAsSeries(Buffer7, true);

   return(INIT_SUCCEEDED);
  }
```

**4\. Initialization Function (OnDeinit)**

When our EA is removed from the chart or when the platform is closed, the (OnDeinit) function is executed. Here, we take care to release the indicator handle, ensuring that we free up any resources that were allocated during the EA’s operation. This cleanup step is crucial for maintaining the efficiency and stability of our trading environment, as it prevents unnecessary resource consumption:

```
///------Deinitialization Function(OnDeinit)
void OnDeinit(const int reason)
  {
   //--- Release the indicator handle
   IndicatorRelease(indicator_handle);
  }
```

**5\. Main Execution Function (OnTick)**

The (OnTick) function is where the real trading action happens. Every time a new market tick is received, this function is called. We start by checking if there is already an open position with the same magic number, ensuring that we don't place duplicate trades. Next, we copy the latest values from the indicator buffers (Buffer6 and Buffer7) to make trading decisions. If our conditions for a buy or sell signal are met, we construct and send the appropriate trade request. We take care to specify all necessary parameters, such as the order type, price, stop loss, take profit, and slippage, to execute our trading strategy effectively:

```
///---Main Execution Function(OnTick)
void OnTick()
  {
   //--- Check if there is already an open position with the same MagicNumber
   if (PositionSelect(Symbol()))
     {
      if (PositionGetInteger(POSITION_MAGIC) == MagicNumber)
        {
         return; // Exit OnTick if there's an open position with the same MagicNumber
        }
     }

   //--- Calculate the indicator
   if (CopyBuffer(indicator_handle, 5, 0, 2, Buffer6) <= 0 || CopyBuffer(indicator_handle, 6, 0, 2, Buffer7) <= 0)
     {
      Print("Failed to copy buffer values. Error: ", GetLastError());
      return;
     }

   //--- Check for a buy signal
   if (Buffer7[0] != EMPTY_VALUE)
     {
      double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
      double sl = NormalizeDouble(ask - StopLoss * _Point, _Digits);
      double tp = NormalizeDouble(ask + TakeProfit * _Point, _Digits);

      //--- Prepare the buy order request
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action = TRADE_ACTION_DEAL;
      request.symbol = Symbol();
      request.volume = Lots;
      request.type = ORDER_TYPE_BUY;
      request.price = ask;
      request.sl = sl;
      request.tp = tp;
      request.deviation = Slippage;
      request.magic = MagicNumber;
      request.comment = "Buy Order";

      //--- Send the buy order
      if (!OrderSend(request, result))
        {
         Print("Error opening buy order: ", result.retcode);
        }
      else
        {
         Print("Buy order opened successfully! Ticket: ", result.order);
        }
     }

   //--- Check for a sell signal
   if (Buffer6[0] != EMPTY_VALUE)
     {
      double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
      double sl = NormalizeDouble(bid + StopLoss * _Point, _Digits);
      double tp = NormalizeDouble(bid - TakeProfit * _Point, _Digits);

      //--- Prepare the sell order request
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action = TRADE_ACTION_DEAL;
      request.symbol = Symbol();
      request.volume = Lots;
      request.type = ORDER_TYPE_SELL;
      request.price = bid;
      request.sl = sl;
      request.tp = tp;
      request.deviation = Slippage;
      request.magic = MagicNumber;
      request.comment = "Sell Order";

      //--- Send the sell order
      if (!OrderSend(request, result))
        {
         Print("Error opening sell order: ", result.retcode);
        }
      else
        {
         Print("Sell order opened successfully! Ticket: ", result.order);
        }
     }
  }
```

**6\. Other Functions**

We also include several other functions to handle different events that our EA might encounter, and they came as part of an EA template and current we do not use them for simplicity:

- (OnTrade): Here, we can handle any specific actions that need to occur when a trade event happens. While currently empty, this function provides a space for us to add logic if needed.
- (OnTester): This function is used for custom calculations during back-testing. By returning a value, we can optimize our strategy based on specific metrics.
- (OnTesterInit, OnTesterPass, OnTesterDeinit): These functions are involved in the optimization process within the strategy tester. They allow us to initialize settings, perform actions after each optimization pass, and clean up resources afterward.
- (OnChartEvent): This function allows us to handle various chart events, such as mouse clicks or key presses. Although it’s currently empty, we can use this space to add interactive features to our EA:

```
///----Other Template functions available
void OnTrade()
  {
   //--- Handle trade events if necessary
  }

double OnTester()
  {
   double ret = 0.0;
   //--- Custom calculations for strategy tester
   return (ret);
  }

void OnTesterInit()
  {
   //--- Initialization for the strategy tester
  }

void OnTesterPass()
  {
   //--- Code executed after each pass in optimization
  }

void OnTesterDeinit()
  {
   //--- Cleanup after tester runs
  }

void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   //--- Handle chart events here
  }
```

Our final program put together becomes like this:

```
//+------------------------------------------------------------------+
//|                                    Trend Constraint Expert.mq5   |
//|                                Copyright 2024, Clemence Benjamin |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property strict
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.0"
#property description "An Expert based on the buffer6 and buffer7 of Trend Constraint V1.09"

//--- Input parameters for the EA
input double Lots = 0.1;          // Lot size
input int Slippage = 3;           // Slippage
input double StopLoss = 50;       // Stop Loss in points
input double TakeProfit = 100;    // Take Profit in points
input int MagicNumber = 123456;   // Magic number for orders

//--- Indicator handle
int indicator_handle;
double Buffer6[];
double Buffer7[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Get the indicator handle
   indicator_handle = iCustom(Symbol(), PERIOD_CURRENT, "Trend Constraint V1.09");
   if (indicator_handle < 0)
     {
      Print("Failed to get the indicator handle. Error: ", GetLastError());
      return(INIT_FAILED);
     }

   //--- Set the buffer arrays as series
   ArraySetAsSeries(Buffer6, true);
   ArraySetAsSeries(Buffer7, true);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   //--- Release the indicator handle
   IndicatorRelease(indicator_handle);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- Check if there is already an open position with the same MagicNumber
   if (PositionSelect(Symbol()))
     {
      if (PositionGetInteger(POSITION_MAGIC) == MagicNumber)
        {
         return; // Exit OnTick if there's an open position with the same MagicNumber
        }
     }

   //--- Calculate the indicator
   if (CopyBuffer(indicator_handle, 5, 0, 2, Buffer6) <= 0 || CopyBuffer(indicator_handle, 6, 0, 2, Buffer7) <= 0)
     {
      Print("Failed to copy buffer values. Error: ", GetLastError());
      return;
     }

   //--- Check for a buy signal
   if (Buffer7[0] != EMPTY_VALUE)
     {
      double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
      double sl = NormalizeDouble(ask - StopLoss * _Point, _Digits);
      double tp = NormalizeDouble(ask + TakeProfit * _Point, _Digits);

      //--- Prepare the buy order request
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action = TRADE_ACTION_DEAL;
      request.symbol = Symbol();
      request.volume = Lots;
      request.type = ORDER_TYPE_BUY;
      request.price = ask;
      request.sl = sl;
      request.tp = tp;
      request.deviation = Slippage;
      request.magic = MagicNumber;
      request.comment = "Buy Order";

      //--- Send the buy order
      if (!OrderSend(request, result))
        {
         Print("Error opening buy order: ", result.retcode);
        }
      else
        {
         Print("Buy order opened successfully! Ticket: ", result.order);
        }
     }

   //--- Check for a sell signal
   if (Buffer6[0] != EMPTY_VALUE)
     {
      double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
      double sl = NormalizeDouble(bid + StopLoss * _Point, _Digits);
      double tp = NormalizeDouble(bid - TakeProfit * _Point, _Digits);

      //--- Prepare the sell order request
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action = TRADE_ACTION_DEAL;
      request.symbol = Symbol();
      request.volume = Lots;
      request.type = ORDER_TYPE_SELL;
      request.price = bid;
      request.sl = sl;
      request.tp = tp;
      request.deviation = Slippage;
      request.magic = MagicNumber;
      request.comment = "Sell Order";

      //--- Send the sell order
      if (!OrderSend(request, result))
        {
         Print("Error opening sell order: ", result.retcode);
        }
      else
        {
         Print("Sell order opened successfully! Ticket: ", result.order);
        }
     }
  }

//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
   //--- Handle trade events if necessary
  }

//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
   double ret = 0.0;
   //--- Custom calculations for strategy tester
   return (ret);
  }

//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
void OnTesterInit()
  {
   //--- Initialization for the strategy tester
  }

//+------------------------------------------------------------------+
//| TesterPass function                                              |
//+------------------------------------------------------------------+
void OnTesterPass()
  {
   //--- Code executed after each pass in optimization
  }

//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
   //--- Cleanup after tester runs
  }

//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   //--- Handle chart events here
  }
//+------------------------------------------------------------------+
```

In the next segment after a successful compile we are going to test our Program.

### Testing:

In MetaEditor, we can use the "Compile" or "Run" button to prepare our program for testing. In this case, the compilation was successful, and we launched the test on the Boom 500 Index using the Strategy Tester.

![Launching strategy tester](https://c.mql5.com/2/88/terminal64_5wEBv7v7jZ.gif)

Launching the Tester from the navigator

A Strategy Tester panel opens up, allowing you to adjust some settings before clicking the "Start" button in the lower-right corner. For example, the default lot size in the EA is set to 0.1 lots, but for the Boom 500 Index, I had to increase it to a minimum of 0.2 lots in this case.

![Strategy Testor](https://c.mql5.com/2/88/Strategy_Tester_Window.PNG.png)

Strategy Tester Panel

Amazingly, our system performed well in the Strategy Tester, as shown in the image below:

![MetaTester Replay](https://c.mql5.com/2/88/metatester64_1eWQlsjfC6.gif)

Strategy Tester Visualization: Trend Constraint Expert

### Conclusion

The addition of risk and reward rectangles provides traders with a clear, graphical representation of their trades, making it easier to monitor and manage open positions. This visual aid is particularly useful in fast-moving markets, where quick decisions are necessary. The rectangles serve as a constant reminder of the trade's potential outcomes, helping traders stay aligned with their original trading plan.

The successful collaboration between the (Trend Constraint V1.09) indicator and the Expert Advisor highlights the importance of synergy between tools in a trading strategy. The indicator identifies potential trends and reversals, while the EA executes trades and manages risk based on this information. This integrated approach leads to a more cohesive and effective trading strategy.

Attached below are the indicator, script, and EA used. There is still room for improvement and modification. I hope you found this information valuable. You are welcome to share your thoughts in the comments section. Happy trading!

| File Attached | Description |
| --- | --- |
| (Trend\_Constraint V1.09.mq5) | Source Code for the Indicator to work with EA. |
| (Trend Constraint R-R.mq5) | Source code for Risk-Reward Rectangle Script. |
| (Trend Constraint Expert.mq5) | Source code for the Expert Advisor working strictly with (Trend Constraint V1.09) |

[Back to Contents](https://www.mql5.com/en/articles/15321#para1)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15321.zip "Download all attachments in the single ZIP archive")

[Trend\_Constraint\_V1.09.mq5](https://www.mql5.com/en/articles/download/15321/trend_constraint_v1.09.mq5 "Download Trend_Constraint_V1.09.mq5")(20.95 KB)

[Trend\_Constraint\_R-R.mq5](https://www.mql5.com/en/articles/download/15321/trend_constraint_r-r.mq5 "Download Trend_Constraint_R-R.mq5")(7.94 KB)

[Trend\_Constraint\_Expert.mq5](https://www.mql5.com/en/articles/download/15321/trend_constraint_expert.mq5 "Download Trend_Constraint_Expert.mq5")(6.79 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/471437)**
(2)


![Helga Gustana Argita](https://c.mql5.com/avatar/2020/1/5E2FE6D1-E5DE.jpg)

**[Helga Gustana Argita](https://www.mql5.com/en/users/argatafx28)**
\|
14 Aug 2024 at 19:50

hello i have some error how to fix it?

2024.08.15 00:47:15.1232024.08.01 00:00:00   cannot load [custom indicator](https://www.mql5.com/en/articles/5 "Article: Step on New Rails: Custom Indicators in MQL5 ") 'Trend Constraint V1.09' \[4802\]

2024.08.15 00:47:15.1232024.08.01 00:00:00   indicator create error in 'Trend\_Constraint\_Expert.mq5' (1,1)

![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
15 Aug 2024 at 22:43

**argatafx28 [#](https://www.mql5.com/en/forum/471437#comment_54300272):**

hello i have some error how to fix it?

2024.08.15 00:47:15.1232024.08.01 00:00:00   cannot load [custom indicator](https://www.mql5.com/en/articles/5 "Article: Step on New Rails: Custom Indicators in MQL5 ") 'Trend Constraint V1.09' \[4802\]

2024.08.15 00:47:15.1232024.08.01 00:00:00   indicator create error in 'Trend\_Constraint\_Expert.mq5' (1,1)

Hi [@argatafx28](https://www.mql5.com/en/users/argatafx28)

1. Ensure that the indicator Trend Constraint V1.09  is installed for the EA to function.
2. Trend Constraint V1.09 uses [ShellExecute](https://www.mql5.com/en/articles/14968#para2) to integrate telegram, could you make sure to allow DLL on dependencies when launching the indicator?

![Enable Allow DLL](https://c.mql5.com/3/442/solution.PNG)

Which operating system are you using by the way?

![MQL5 Integration: Python](https://c.mql5.com/2/89/logo-midjourney_image_14135_392_3769__1.png)[MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)

Python is a well-known and popular programming language with many features, especially in the fields of finance, data science, Artificial Intelligence, and Machine Learning. Python is a powerful tool that can be useful in trading as well. MQL5 allows us to use this powerful language as an integration to get our objectives done effectively. In this article, we will share how we can use Python as an integration in MQL5 after learning some basic information about Python.

![Reimagining Classic Strategies (Part IV): SP500 and US Treasury Notes](https://c.mql5.com/2/90/logo-15531_385_3705.png)[Reimagining Classic Strategies (Part IV): SP500 and US Treasury Notes](https://www.mql5.com/en/articles/15531)

In this series of articles, we analyze classical trading strategies using modern algorithms to determine whether we can improve the strategy using AI. In today's article, we revisit a classical approach for trading the SP500 using the relationship it has with US Treasury Notes.

![Pattern Recognition Using Dynamic Time Warping in MQL5](https://c.mql5.com/2/89/logo-midjourney_image_15572_396_3823.png)[Pattern Recognition Using Dynamic Time Warping in MQL5](https://www.mql5.com/en/articles/15572)

In this article, we discuss the concept of dynamic time warping as a means of identifying predictive patterns in financial time series. We will look into how it works as well as present its implementation in pure MQL5.

![Population optimization algorithms: Boids Algorithm](https://c.mql5.com/2/74/Population_optimization_algorithms_Boyd_algorithmp_or_flock_algorithm___LOGO.png)[Population optimization algorithms: Boids Algorithm](https://www.mql5.com/en/articles/14576)

The article considers Boids algorithm based on unique examples of animal flocking behavior. In turn, the Boids algorithm serves as the basis for the creation of the whole class of algorithms united under the name "Swarm Intelligence".

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/15321&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049351423986084332)

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