---
title: William Gann methods (Part I): Creating Gann Angles indicator
url: https://www.mql5.com/en/articles/15556
categories: Trading, Trading Systems, Indicators
relevance_score: 6
scraped_at: 2026-01-22T17:58:08.099792
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/15556&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049522587022765310)

MetaTrader 5 / Trading


### Introduction

William Delbert Gann is a legendary trader and technical analyst whose innovative methods of market analysis still attract interest from modern traders. One of the key tools in Gann's arsenal were his famous angles, used to forecast price movements and identify potential support and resistance levels.

In this article, we will dive into the world of William Gann's trading methods, starting with creating Gann Angles indicator in MQL5. We will go through the theory behind this tool and implement it step by step as a custom indicator for the MetaTrader 5 platform.

Whether you are an experienced trader looking for new analysis tools or a beginner looking to expand your arsenal of technical indicators, this article will help you better understand and apply one of William Gann's most intriguing methods in your trading.

### History of creating Gann angles

![](https://c.mql5.com/2/120/metody-analiza__1.jpg)

William Delbert Gann (1878-1955) developed his system of angles in the early 20th century based on years of research into market movements and his unique understanding of the relationship between time and price.

Gann believed that markets moved in predictable geometric patterns and that these movements could be predicted using a combination of mathematics, geometry and astrology. He developed the concept of angles, which are diagonal lines on a chart that reflect the perfect balance between time and price movement.

Key to Gann's theory was the idea that a 45 degree angle (known as the 1x1 angle) represented the perfect balance between time and price. He believed that when the price moves up or down at this angle, it indicates a balanced and stable trend.

Gann also developed other angles such as 2x1, 3x1, 4x1 and their reciprocals which represent different relationships between time and price. These angles became the basis for his trading system and market analysis.

While some aspects of Gann's work remain controversial, his methods, including Gann angles, continue to attract the attention of traders and analysts around the world and remain relevant in modern trading.

### Gann Angle theory and its importance in technical analysis

Gann Angle theory is based on the concept that market movements follow predictable geometric patterns that can be identified using special angle plots on a chart. The theory is based on the idea of a balance between time and price, where a 1x1 angle (45 degrees) represents a perfect balance, assuming a price change of one unit per time period.

Gann developed a system of different angles such as 2x1, 3x1, 4x1 and their reciprocals, each of which reflects a specific relationship between time and price movement. These angles serve as dynamic support and resistance levels, helping traders determine the strength and direction of a trend. Steeper angles indicate a stronger trend, while flatter angles indicate a weaker movement.

### Basic principles of constructing Gann angles

![](https://c.mql5.com/2/120/Gannb__1.png)

The construction of Gann angles is based on several key principles that allow traders to effectively use this tool in technical analysis. The choice of the starting point (which is usually a significant minimum or maximum on the chart) is of primary importance. It is from this point that the construction of angles begins.

The base angle is considered to be the 1x1 angle that forms a 45 degree line on the chart. This angle reflects the perfect balance between time and price, where the price changes by one unit in one period of time. Other angles, such as 2x1, 3x1, 4x1 and their reciprocals, are constructed relative to this base angle.

When plotting, it is important to consider the chart scale. Traders often use special templates or tools to ensure accurate angles. The angle lines continue into the future, allowing one to forecast potential support and resistance levels.

### Types of Gann angles and their interpretation

William Gann developed a system of angles, each with its own unique meaning and interpretation in technical analysis. 1x1, or 45 degrees, is considered to be the main angle, which reflects the balance between time and price. This angle serves as a basic guideline for assessing the strength of the trend.

The 2x1 angle (63.75 degrees) indicates a stronger price move where the price is rising twice as fast as time. This is often interpreted as a sign of a strong bullish trend. In contrast, a 1x2 angle (26.25 degrees) indicates a slower price increase relative to time, which may indicate a weakening trend.

The 3x1 (71.25 degrees) and 4x1 (75 degrees) angles represent even more aggressive price action and are usually associated with very strong trends or potential market overheating. Their reciprocals - 1x3 (18.75 degrees) and 1x4 (15 degrees) - can indicate strong resistance or support.

The interpretation of Gann angles is not limited to their slope alone. It is also important to consider how price interacts with these lines. A price crossing the angle line may signal a potential change in trend. If the price moves along the angle line, it is often interpreted as confirmation of the strength of the current trend.

### Practical application of Gann angles in trading

Traders use these tools for a variety of purposes, from identifying trends to choosing entry and exit points for positions.

To determine a trend, one usually starts by plotting a 1x1 angle from a significant minimum or maximum. If the price moves above this line, it is interpreted as an uptrend, and if below, as a downtrend. Steeper angles, such as 2x1 or 3x1, are used to confirm the strength of the trend.

When choosing entry points, many traders look for moments when the price bounces off or breaks through the Gann angle line. For example, a bounce off the 1x1 line in the direction of the trend can be seen as a potential opportunity to enter a position.

For risk management, Gann angles are often used to set stop losses. A trader can place a stop loss just below the nearest angle line for a long position or just above it for a short position.

In long-term trading, Gann angles help determine the overall direction of the market. We can use shallower angles, such as 1x2 or 1x4, to assess long-term trends and make strategic decisions.

### Examples of trading strategies using Gann angles

Gann angles provide traders with a wide range of options to create a variety of trading strategies. Here are some examples of how they can be used effectively in real trading.

The Angle Bounce strategy is based on the assumption that Gann angle lines often act as support or resistance levels. A trader looks for situations where the price approaches a Gann angle line (especially 1x1 or 2x1) and bounces off it. Entry into a position is carried out after confirmation of a rebound, for example, the formation of a candlestick reversal pattern.

Another popular strategy is Angle Breakout. Here the trader waits for the price to break through an important Gann angle line, especially if this is accompanied by an increase in trading volume. An upward breakout may signal a potential long position, while a downward breakout may signal a short position.

The Gann Fan strategy uses multiple angles radiating from a single point to form a fan-shaped structure. The trader analyzes how price interacts with the various fan lines, using this to determine support, resistance levels and potential reversal points.

Combining Angles and Time Cycles is a more complex strategy where the trader combines Gann angles with the concept of time cycles, also developed by Gann. Here the intersection points of important time lines with Gann angles are analyzed to determine critical moments for entering or exiting the market.

The multi-timeframe strategy involves analyzing Gann angles on different timeframes. For example, a trader might use a 1x1 angle on a daily chart to determine the overall trend, and then switch to an hourly chart to look for entry points using steeper angles.

### Creating Gann Angles indicator in MQL5: Basic steps

![](https://c.mql5.com/2/120/7l9ykvfg_12-08-2024_142158__1.jpg)

Creating a Gann Angles indicator in MQL5 involves several key steps. This process requires an understanding of both the principles of constructing Gann angles and the specifics of programming in the MetaTrader 5 environment.

The first step is to define the indicator structure. Here we set the basic parameters, such as the indicator name, input parameters for setting angles and the required libraries.

The main logic for constructing Gann angles is located in the OnCalculate() function. Here we define the starting point for plotting the angles, calculate the coordinates for each angle, and draw the lines on the chart.

An important aspect is the correct calculation of the coordinates of the angles, taking into account the scale of the chart and the selected time interval. This requires a precise mathematical approach and an understanding of the geometry of Gann angles.

The final stage is testing and debugging the indicator. It is necessary to check the validity of the construction of angles on different timeframes and instruments.

### Indicator code structure

Here is the basic code of the Gann Angles indicator that can be used in MetaTrader 5:

```
#property copyright "Copyright 2024, Evgeniy Shtenco"
#property link      "https://www.mql5.com/en/users/koshtenko"
#property version   "1.00"
#property indicator_chart_window

// Input parameters
input datetime StartDate = D'2023.01.01 00:00'; // Start date for analysis
input datetime EndDate = D'2023.12.31 23:59';   // End date for analysis
input color GannFanColor = clrBlue;             // Color for Gann Fan lines

// Global variables
double extremumPrice;
datetime extremumTime;
double oppositeExtremumPrice;
datetime oppositeExtremumTime;
bool isTrendUp;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    ObjectsDeleteAll(0, "GannFan_");
    ObjectsDeleteAll(0, "OppositeGannFan_");
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
    if(rates_total < 1) return(0);

    // Clear previous objects
    if (prev_calculated == 0)
    {
        ObjectsDeleteAll(0, "GannFan_");
        ObjectsDeleteAll(0, "OppositeGannFan_");
    }

    // Find extremums within the specified date range
    FindExtremums(rates_total, high, low, time);

    // Draw Gann Fans
    DrawGannFan(extremumPrice, extremumTime);
    DrawOppositeGannFan(oppositeExtremumPrice, oppositeExtremumTime);

    return(rates_total);
}

//+------------------------------------------------------------------+
//| Find both extremums within the specified date range              |
//+------------------------------------------------------------------+
void FindExtremums(const int rates_total, const double &high[], const double &low[], const datetime &time[])
{
    int startIndex = -1;
    int endIndex = -1;

    for (int i = 0; i < rates_total; i++)
    {
        if (time[i] >= StartDate && startIndex == -1)
        {
            startIndex = i;
        }
        if (time[i] <= EndDate)
        {
            endIndex = i;
        }
    }

    if (startIndex == -1 || endIndex == -1 || startIndex > endIndex)
    {
        Print("Error: Invalid date range or no data available in the specified range");
        return;
    }

    int highestIndex = ArrayMaximum(high, startIndex, endIndex - startIndex + 1);
    int lowestIndex = ArrayMinimum(low, startIndex, endIndex - startIndex + 1);

    // Determine the most recent extremum
    if (time[highestIndex] > time[lowestIndex])
    {
        extremumPrice = high[highestIndex];
        extremumTime = time[highestIndex];
        oppositeExtremumPrice = low[lowestIndex];
        oppositeExtremumTime = time[lowestIndex];
        isTrendUp = false;
    }
    else
    {
        extremumPrice = low[lowestIndex];
        extremumTime = time[lowestIndex];
        oppositeExtremumPrice = high[highestIndex];
        oppositeExtremumTime = time[highestIndex];
        isTrendUp = true;
    }
}

//+------------------------------------------------------------------+
//| Draw Gann Fan                                                    |
//+------------------------------------------------------------------+
void DrawGannFan(double extremum, datetime extremumTime)
{
    double angles[] = {82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5};
    string angleNames[] = {"1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"};

    datetime endTime = extremumTime + PeriodSeconds() * 300;

    for(int i = 0; i < ArraySize(angles); i++)
    {
        string label = "GannFan_" + angleNames[i];
        double angle = angles[i];

        double priceShift = MathTan(angle * M_PI / 180.0) * 300 * _Point;
        double endPrice;

        if(isTrendUp)
        {
            endPrice = extremum + priceShift;
        }
        else
        {
            endPrice = extremum - priceShift;
            angle = -angle; // Invert the angle for a downtrend
        }

        if(ObjectCreate(0, label, OBJ_TREND, 0, extremumTime, extremum, endTime, endPrice))
        {
            ObjectSetInteger(0, label, OBJPROP_COLOR, GannFanColor);
            ObjectSetInteger(0, label, OBJPROP_STYLE, STYLE_SOLID);
            ObjectSetInteger(0, label, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, label, OBJPROP_RAY_RIGHT, true);
            ObjectSetString(0, label, OBJPROP_TOOLTIP, "Gann Fan " + angleNames[i]);
        }
        else
        {
            Print("Failed to create Gann Fan line: ", GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//| Draw Opposite Gann Fan                                           |
//+------------------------------------------------------------------+
void DrawOppositeGannFan(double extremum, datetime extremumTime)
{
    double angles[] = {82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5};
    string angleNames[] = {"1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"};

    datetime endTime = extremumTime + PeriodSeconds() * 300;

    for(int i = 0; i < ArraySize(angles); i++)
    {
        string label = "OppositeGannFan_" + angleNames[i];
        double angle = angles[i];

        double priceShift = MathTan(angle * M_PI / 180.0) * 300 * _Point;
        double endPrice;

        if(!isTrendUp) // Opposite trend
        {
            endPrice = extremum + priceShift;
        }
        else
        {
            endPrice = extremum - priceShift;
            angle = -angle; // Invert the angle for a downtrend
        }

        if(ObjectCreate(0, label, OBJ_TREND, 0, extremumTime, extremum, endTime, endPrice))
        {
            ObjectSetInteger(0, label, OBJPROP_COLOR, GannFanColor);
            ObjectSetInteger(0, label, OBJPROP_STYLE, STYLE_SOLID);
            ObjectSetInteger(0, label, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, label, OBJPROP_RAY_RIGHT, true);
            ObjectSetString(0, label, OBJPROP_TOOLTIP, "Opposite Gann Fan " + angleNames[i]);
        }
        else
        {
            Print("Failed to create Opposite Gann Fan line: ", GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
    // Redraw objects when chart changes
    if(id == CHARTEVENT_CHART_CHANGE)
    {
        // Find extremums and redraw Gann Fans
        int rates_total = Bars(_Symbol, PERIOD_CURRENT);
        double high[], low[];
        datetime time[];
        ArraySetAsSeries(high, true);
        ArraySetAsSeries(low, true);
        ArraySetAsSeries(time, true);
        CopyHigh(_Symbol, PERIOD_CURRENT, 0, rates_total, high);
        CopyLow(_Symbol, PERIOD_CURRENT, 0, rates_total, low);
        CopyTime(_Symbol, PERIOD_CURRENT, 0, rates_total, time);

        FindExtremums(rates_total, high, low, time);
        DrawGannFan(extremumPrice, extremumTime);
        DrawOppositeGannFan(oppositeExtremumPrice, oppositeExtremumTime);
    }
}
```

### Main functions and their purpose

The code is a MetaTrader 5 indicator that draws Gann Fan angles on the chart. Here is a description of the main functions and their purpose:

1. OnInit() is the indicator initialization function. In this case, it simply returns a successful initialization result.
2. OnDeinit() is a deinitialization function that removes all objects created by the indicator when it is removed from the chart.
3. OnCalculate() is the main function of the indicator. It is called at every tick. It clears previous objects, finds extremes in a given date range and draws Gann angles.
4. FindExtremums() is a function for finding extremes (maximum and minimum) of a price in a given date range. It determines which extreme is more recent and establishes the direction of the trend.
5. DrawGannFan() is a function for drawing the main Gann fan from the found extremum. It creates lines for nine different angles.
6. DrawOppositeGannFan() is a function for drawing the opposite Gann fan from another extremum. It also creates lines for the nine angles, but in the opposite direction.
7. OnChartEvent() is a function that responds to chart events. In this case, it redraws the Gann fans when the chart changes.

The indicator uses inputs to set the analysis date range (StartDate and EndDate) and the color of the lines (GannFanColor). It finds price extremes in this range, determines the trend direction and draws two Gann fans - one from the last extreme, the other from the opposite one. Each fan consists of nine lines corresponding to different Gann angles (82.5°, 75°, 71.25°, 63.75°, 45°, 26.25°, 18.75°, 15° and 7.5°).

The code also includes error handling and dynamic updating when the chart changes, making it quite robust and adaptive to different market conditions. Using MQL5 functions, such as ArrayMaximum() and ArrayMinimum(), ensures efficient data handling.

![](https://c.mql5.com/2/120/vhfphifb_12-08-2024_142252__1.jpg)

### Gann angle calculation algorithm

The algorithm for calculating Gann angles in this indicator is based on the geometric construction of lines from the extremum point. Here are the key aspects of this algorithm:

- Determining extremes: The indicator finds the maximum and minimum price in a given time range. The later of these extremes is used as the starting point for the main Gann fan, and the earlier one for the opposite fan.
- Angle set: A fixed angle set is used: 82.5°, 75°, 71.25°, 63.75°, 45°, 26.25°, 18.75°, 15° and 7.5°. These angles correspond to the traditional Gann ratios: 1x8, 1x4, 1x3, 1x2, 1x1, 2x1, 3x1, 4x1 and 8x1.
- End point calculation: For each angle, the end point of the line is calculated. The calculation is based on the equation:

```
endPrice = extremum + MathTan(angle * M_PI / 180.0) * 300 * Point
```

- Here 300 is the conventional number of bars ahead used to construct the line.
- Inversion for a downtrend: If the trend is down, the angle is inverted (changed sign) so that the lines point downwards from the extreme point.
- Line construction: For each angle, a line object (OBJ\_TREND) is created from the extreme point to the calculated end point. Lines are extended to the right to cover future data.
- Double fan: the algorithm is applied twice - once for the main fan from the last extremum, the second time for the opposite fan from the early extremum. This allows us to visualize potential support and resistance levels on both sides.
- Dynamic update: When the chart changes (for example, when new data arrives), the algorithm recalculates the extremes and rebuilds the fans, ensuring that the analysis is up-to-date.

The algorithm allows us to visualize classic Gann angles on a chart, providing the trader with a tool to analyze potential support, resistance levels and trend direction.

### Is it possible to trade profitably?

Is it possible to trade profitably using Gann angles? This question does not have a clear answer, but with the right approach, profitable trading is quite possible. The key to success lies in the comprehensive use of this tool.

However, even with the most careful and meticulous analysis, one must not forget about risk management. This is the cornerstone of profitable trading. No method guarantees 100% success, so proper capital and risk management play a major role in long-term profitability.

### Prospects for creating a Gann Angles-based EA

![](https://c.mql5.com/2/120/xnik59u7_12-08-2024_142922__1.jpg)

Creating an EA based on Gann Angles is an interesting and promising task for developers of algorithmic trading systems. Such an EA could automate the analysis and trading using the principles laid down in Gann's theory.

The basis for such an EA can be the already implemented Gann Angles indicator. The EA algorithm could include the analysis of the interaction of price with Gann lines, determining entry and exit points based on the intersection of these lines, as well as taking into account additional factors such as trading volume and market volatility.

One of the key benefits of an automated system based on Gann angles would be to remove the emotional factor from the trading decision making process. The EA could strictly follow the given strategy, without giving in to fear or greed, which often influences the decisions of a human trader.

However, developing such an EA faces a number of challenges. First of all, there is a need for accurate interpretation of the signals generated by Gann angles. Gann Theory is largely subjective and requires a deep understanding of the market, which is difficult to fully implement in the form of an algorithm.

Of course, the success of such an EA will largely depend on the depth of understanding of the Gann theory by the developer, as well as on their ability to effectively combine this theory with modern methods of market analysis and risk management.

### Biggest challenge: Scaling angles

One of the most serious problems with creating and using indicators and EAs based on Gann angles is the difficulty in scaling them. This problem significantly affects the accuracy and efficiency of trading systems based on the Gann method.

The essence of the problem is that Gann angles, being geometric constructions, are highly dependent on the chart scale. When changing the scale of the time axis or price scale, the visual representation of angles may become significantly distorted. What looks like a perfect 1x1 (45 degree) angle at one scale may turn into a completely different angle when you change the scale.

This creates serious difficulties for software implementation. The indicator or EA should constantly take into account the current scale of the chart and adjust the angle calculations accordingly. Moreover, the very concept of "unit of time equal to unit of price" Gann angles are based on becomes ambiguous when working with different financial instruments and time frames.

Attempts to solve this problem often lead to compromises. Some developers fix the scale, which limits the applicability of the tool. Others introduce complex algorithms for recalculating angles, which can reduce the speed of the indicator or EA.

In addition, the scaling issue makes it difficult to compare analysis results across different tools or timeframes. An angle that works effectively on a daily chart of one currency pair may produce completely different results on an hourly chart, or for another trading instrument.

This issue calls into question the universal applicability of Gann angles in automated trading. It requires developers to have a deep understanding of not only Gann theory, but also the specifics of working with charts in trading platforms.

### Conclusion

Gann angles are a unique technical analysis tool that continues to attract the attention of traders and trading system developers. Our review showed that creating Gann angles indicator in MQL5 is quite feasible, and the provided code serves as a good starting point for further development.

However, using Gann angles in automated trading comes with a number of significant challenges. The main one is the scaling problem, which makes it much more difficult to create a universal and reliable trading EA based on this method.

Despite these difficulties, the potential of Gann angles as a market analysis tool remains high. When used correctly, in combination with other technical analysis methods and proper risk management, they can be an effective addition to a trader's toolkit.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15556](https://www.mql5.com/ru/articles/15556)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15556.zip "Download all attachments in the single ZIP archive")

[GannAngleIndicator.mq5](https://www.mql5.com/en/articles/download/15556/gannangleindicator.mq5 "Download GannAngleIndicator.mq5")(16.11 KB)

[GannAngleIndicator.mq4](https://www.mql5.com/en/articles/download/15556/gannangleindicator.mq4 "Download GannAngleIndicator.mq4")(15.42 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forex arbitrage trading: Analyzing synthetic currencies movements and their mean reversion](https://www.mql5.com/en/articles/17512)
- [Forex arbitrage trading: A simple synthetic market maker bot to get started](https://www.mql5.com/en/articles/17424)
- [Build a Remote Forex Risk Management System in Python](https://www.mql5.com/en/articles/17410)
- [Currency pair strength indicator in pure MQL5](https://www.mql5.com/en/articles/17303)
- [Capital management in trading and the trader's home accounting program with a database](https://www.mql5.com/en/articles/17282)
- [Analyzing all price movement options on the IBM quantum computer](https://www.mql5.com/en/articles/17171)
- [Fibonacci in Forex (Part I): Examining the Price-Time Relationship](https://www.mql5.com/en/articles/17168)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/481724)**
(18)


![Bogard_11](https://c.mql5.com/avatar/avatar_na2.png)

**[Bogard\_11](https://www.mql5.com/en/users/bogard_11)**
\|
15 Aug 2024 at 18:56

**Eva-oren [#](https://www.mql5.com/ru/forum/471438/page2#comment_54305813):**

Gunn was a brilliant trader, and astrologer, and numerologist, a mathematician who believed in magic numbers. A great soothsayer who predicted the beginning and end of the First and Second World Wars. He had a huge number of students, but as his followers claim, he did not give his strategy to anyone. Had a huge turnover of money, and by the end of his life came with a modest sum of **$100000**. I will look forward to the next articles about Gunn's genius personality, very interesting.

Forgot to mention, he had a private jet, which was a rarity in those days. Plus **$100,000** back then, that's at least a multh of green now, if not **10** mulths.

**P.S.** \- he passed on the strategy. In his writings. Who wanted, collected from different chapters of the complete TC. Gunn scattered it all so no one could get it for free.

![Ivan Butko](https://c.mql5.com/avatar/2017/1/58797422-0BFA.png)

**[Ivan Butko](https://www.mql5.com/en/users/capitalplus)**
\|
21 Aug 2024 at 17:11

Found a way to trade on google

[![](https://c.mql5.com/3/442/5965935208656__1.png)](https://c.mql5.com/3/442/5965935208656.png "https://c.mql5.com/3/442/5965935208656.png")[![](https://c.mql5.com/3/442/6485080701675__1.png)](https://c.mql5.com/3/442/6485080701675.png "https://c.mql5.com/3/442/6485080701675.png")

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
22 Aug 2024 at 03:16

**Ivan Butko [#](https://www.mql5.com/ru/forum/471438/page2#comment_54360435):**

Found a way of trading on Google

in the second picture - I don't understand the principle :-(

![Bogard_11](https://c.mql5.com/avatar/avatar_na2.png)

**[Bogard\_11](https://www.mql5.com/en/users/bogard_11)**
\|
22 Aug 2024 at 06:58

I'll give you a picture. Who understands Gunn, will immediately understand the principle of working with corners, where to look for a long entry and where to roll over.

[![](https://c.mql5.com/3/442/GBPUSD.ffxM5__1.png)](https://c.mql5.com/3/442/GBPUSD.ffxM5.png "https://c.mql5.com/3/442/GBPUSD.ffxM5.png")

![Avatar2025](https://c.mql5.com/avatar/2025/6/685bc77a-8ff4.jpg)

**[Avatar2025](https://www.mql5.com/en/users/avatar2025)**
\|
25 Jun 2025 at 09:50

Disclaimer: I don't know MQL5 programming syntax for indicators.

But I think the angle line algorithm is not right (see attached angle line indicator I wrote in another software), here is my dialogue with DeepSeek:

```
下面代码是通达信的代码，里面计算斜率的方法才是正确的，请阅读理解后，把上面priceShift 的计算方法改为下列代码正确的范式:

涨周期数:= IF(低点k位置 > 高点k位置, 低点k位置 - 高点k位置 + 1, 低点k位置), NODRAW;
跌周期数:= IF(高点k位置 > 低点k位置, 高点k位置 - 低点k位置 + 1, 高点k位置), NODRAW;

天线:= CONST(IF(高点k位置 = 1, H, REF(H, 高点k位置 - 1))), NODRAW;
地线:= CONST(IF(低点k位置 = 1, L, REF(L, 低点k位置 - 1))), NODRAW;

上涨天数:= IF(ISVALID(地线), BARSLAST(L = 地线), DRAWNULL), NODRAW;
下跌天数:= IF(ISVALID(天线), BARSLAST(H = 天线), DRAWNULL), NODRAW;

上涨高度:= IF(低点k位置 > 高点k位置, 天线 - 地线, HHV(H, 上涨天数) - 地线), NODRAW;
下跌高度:= IF(高点k位置 > 低点k位置, 天线 - 地线, 天线 - LLV(L, 下跌天数)), NODRAW;

上涨斜率:= IF(上涨天数 > 0, ROUND2(上涨高度 / 涨周期数, 4), 0), NODRAW;
下跌斜率:= IF(下跌天数 > 0, ROUND2(下跌高度 / 跌周期数, 4), 0), NODRAW;
```

However, DeepSeek gave me a modification code that didn't seem to work for me:

```
//+------------------------------------------------------------------+
//| Plotting the Venn sector line & nbsp; |
//+------------------------------------------------------------------+
void DrawGannFan(double extremum, datetime extremumTime)
{
    // Define the Gann angle and the corresponding name.
    double angles[] = {82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5};
    string angleNames[] = {"1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"};

    // Get the number of periods from the extreme point to the current K-line
    int extremumBar = iBarShift(_Symbol, PERIOD_CURRENT, extremumTime);
    int currentBar = 0; // Current K-line is 0
    int barDiff = currentBar - extremumBar; // Cycle number difference

    // Calculate the price height difference
    double priceDiff;
    if(isTrendUp)
    {
        // Uptrend: calculates the height from the extreme low to the current maximum price
        double currentHigh = iHigh(_Symbol, PERIOD_CURRENT, 0);
        priceDiff = currentHigh - extremum;
    }
    else
    {
        // Downtrend: calculates the height from the extreme high to the current low price
        double currentLow = iLow(_Symbol, PERIOD_CURRENT, 0);
        priceDiff = extremum - currentLow;
    }

    // Calculate the base slope (slope of the 1x1 line)
    double baseSlope = (barDiff > 0) ? priceDiff / barDiff : 0;

    // Draw sector lines for each angle
    for(int i = 0; i < ArraySize(angles); i++)
    {
        string label = "GannFan_" + angleNames[i];  // Object labels
        double angle = angles[i];                  // Current angle

        // Calculate the slope multiplier based on the angle
        double slopeMultiplier = MathTan(angle * M_PI / 180.0) / MathTan(45 * M_PI / 180.0);

        // Calculate price offset (based on underlying slope and angle multiplier)
        double priceShift = baseSlope * barDiff * slopeMultiplier * _Point;
        double endPrice;

        // Calculation of closing price based on trend direction
        if(isTrendUp)
        {
            endPrice = extremum + priceShift;
        }
        else
        {
            endPrice = extremum - priceShift;
            angle = -angle; // Reversal angle in downtrend
        }

        // Create a trendline object
        if(ObjectCreate(0, label, OBJ_TREND, 0, extremumTime, extremum, TimeCurrent(), endPrice))
        {
            // Set trendline properties
            ObjectSetInteger(0, label, OBJPROP_COLOR, GannFanColor);
            ObjectSetInteger(0, label, OBJPROP_STYLE, STYLE_SOLID);
            ObjectSetInteger(0, label, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, label, OBJPROP_RAY_RIGHT, true);
            ObjectSetString(0, label, OBJPROP_TOOLTIP, "Gann Fan " + angleNames[i]);
        }
        else
        {
            Print("Failed to create Gann Fan line: ", GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//|| Plotting a Reverse Gann Fan Line & nbsp; &nbsp ; ||
//+------------------------------------------------------------------+
void DrawOppositeGannFan(double extremum, datetime extremumTime)
{
    // Define the Gann angle and the corresponding name.
    double angles[] = {82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5};
    string angleNames[] = {"1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"};

    // Get the number of periods from the extreme point to the current K-line
    int extremumBar = iBarShift(_Symbol, PERIOD_CURRENT, extremumTime);
    int currentBar = 0; // Current K-line is 0
    int barDiff = currentBar - extremumBar; // Cycle number difference

    // Calculate the price height difference
    double priceDiff;
    if(!isTrendUp)
    {
        // Reverse trend (up): calculates the height from the extreme low to the current maximum price
        double currentHigh = iHigh(_Symbol, PERIOD_CURRENT, 0);
        priceDiff = currentHigh - extremum;
    }
    else
    {
        // Reverse trend (downward): calculates the height from the extreme high to the current low price
        double currentLow = iLow(_Symbol, PERIOD_CURRENT, 0);
        priceDiff = extremum - currentLow;
    }

    // Calculate the base slope (slope of the 1x1 line)
    double baseSlope = (barDiff > 0) ? priceDiff / barDiff : 0;

    // Draw sector lines for each angle
    for(int i = 0; i < ArraySize(angles); i++)
    {
        string label = "OppositeGannFan_" + angleNames[i];  // Object labels
        double angle = angles[i];                          // Current angle

        // Calculate the slope multiplier based on the angle
        double slopeMultiplier = MathTan(angle * M_PI / 180.0) / MathTan(45 * M_PI / 180.0);

        // Calculate price offset (based on underlying slope and angle multiplier)
        double priceShift = baseSlope * barDiff * slopeMultiplier * _Point;
        double endPrice;

        // Calculation of the closing price based on the direction of the reverse trend
        if(!isTrendUp)
        {
            endPrice = extremum + priceShift;
        }
        else
        {
            endPrice = extremum - priceShift;
            angle = -angle; // Reversal angle in downtrend
        }

        // Create a trendline object
        if(ObjectCreate(0, label, OBJ_TREND, 0, extremumTime, extremum, TimeCurrent(), endPrice))
        {
            // Set trendline properties
            ObjectSetInteger(0, label, OBJPROP_COLOR, GannFanColor);
            ObjectSetInteger(0, label, OBJPROP_STYLE, STYLE_SOLID);
            ObjectSetInteger(0, label, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, label, OBJPROP_RAY_RIGHT, true);
            ObjectSetString(0, label, OBJPROP_TOOLTIP, "Opposite Gann Fan " + angleNames[i]);
        }
        else
        {
            Print("Failed to create Opposite Gann Fan line: ", GetLastError());
        }
    }
}
```

In the end, I had to ask DeepSeek to tweak your source code (it works fine and draws lines automatically):

```
#property copyright "Copyright 2024, Evgeniy Shtenco"  // Copyright information
#property link      "https://www.mql5.com/en/users/koshtenko"  // Author Links
#property version   "1.00"  // Indicator version number
#property indicator_chart_window  // Indicators are displayed in the chart window

// Input parameters
input int LookBackBars = 300;      // Number of K-lines analysed back
input color GannFanColor = clrBlue; // Vaughan's fan line colours

// Global variables
double extremumPrice;        // Extreme point price
datetime extremumTime;       // Extreme point time
double oppositeExtremumPrice; // Reverse Extreme Point Price
datetime oppositeExtremumTime; // Reverse polarity point time
bool isTrendUp;              // Trend direction flag (true for uptrend)

//+------------------------------------------------------------------+
//| Custom Indicator Initialisation Functions & nbsp; &nbsp ; ||
//+------------------------------------------------------------------+
int OnInit()
{
    return (INIT_SUCCEEDED);  // Successful initialisation
}

//+------------------------------------------------------------------+
//| Custom Indicator Reinitialisation Functions & nbsp; &nbsp ; ||
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
    {
        // Delete all Gann sector objects
        ObjectsDeleteAll(0, "GannFan_");
ObjectsDeleteAll(0, "OppositeGannFan_");
}

//+------------------------------------------------------------------+
//| Custom Indicator Calculation Functions & nbsp; &nbsp ; | | | | | | | |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,          // Current number of cycles
const int prev_calculated,      // Number of previously calculated cycles
const datetime & time[],         // Time arrays
                const double & open[],          // Array of opening prices
                const double & high[],           // Array of highest prices
                const double & low[],            // Array of lowest prices
                const double & close[],         // Closing Price Array
                const long & tick_volume[],      // Volume array
                const long & volume[],           // Real Volume Array
                const int & spread[])            // Array of point spreads
{
    if (rates_total < LookBackBars) return (0);  // Return if there is not enough data

    // Clear the previously drawn objects
    if (prev_calculated == 0) {
        ObjectsDeleteAll(0, "GannFan_");
        ObjectsDeleteAll(0, "OppositeGannFan_");
    }

    // Finding the extreme points in the last 300 K-lines
    FindExtremums(rates_total, high, low, time);

    // Drawing the Venn Fan Line
    DrawGannFan(extremumPrice, extremumTime);
    DrawOppositeGannFan(oppositeExtremumPrice, oppositeExtremumTime);

    return (rates_total);  // Returns the total number of columns processed
}

//+------------------------------------------------------------------+
//| Find extreme points within a specified number of K-lines. |
//+------------------------------------------------------------------+
void FindExtremums(const int rates_total, const double & high[], const double & low[], const datetime & time[])
{
    int startIndex = rates_total - LookBackBars;  // Start indexing (300 K-lines ago)
    int endIndex = rates_total - 1;               // End index (latest K-line)

    // Finding highs and lows
    int highestIndex = ArrayMaximum(high, startIndex, LookBackBars);
    int lowestIndex = ArrayMinimum(low, startIndex, LookBackBars);

    // Determine the direction of the trend (compare the timing of highs and lows)
    if (time[highestIndex] > time[lowestIndex]) {
        // Downtrend if the highest point is after the lowest point
        extremumPrice = high[highestIndex];
        extremumTime = time[highestIndex];
        oppositeExtremumPrice = low[lowestIndex];
        oppositeExtremumTime = time[lowestIndex];
        isTrendUp = false;
    }
    else {
        // otherwise uptrend
        extremumPrice = low[lowestIndex];
        extremumTime = time[lowestIndex];
        oppositeExtremumPrice = high[highestIndex];
        oppositeExtremumTime = time[highestIndex];
        isTrendUp = true;
    }
}

//+------------------------------------------------------------------+
//| Plotting the Venn sector line & nbsp; |
//+------------------------------------------------------------------+
void DrawGannFan(double extremum, datetime extremumTime)
{
    // Define the Gann angle and the corresponding name.
    double angles[] = { 82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5};
    string angleNames[] = { "1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"};

    // Calculate the end time of the sector line (current time + 300 cycles)
    datetime endTime = TimeCurrent();  // Use the current time as the end time

    // Draw sector lines for each angle
    for (int i = 0; i < ArraySize(angles); i++)
    {
        string label = "GannFan_" + angleNames[i];  // Object labels
        double angle = angles[i];                  // Current angle

        // Calculate price offset (based on time difference)
        double secondsDiff = endTime - extremumTime;
        double priceShift = MathTan(angle * M_PI / 180.0) * secondsDiff / PeriodSeconds();
        double endPrice;

        // Calculation of closing price based on trend direction
        if (isTrendUp) {
            endPrice = extremum + priceShift;
        }
        else {
            endPrice = extremum - priceShift;
            angle = -angle; // Reversal angle in downtrend
        }

        // Create a trendline object
        if (ObjectCreate(0, label, OBJ_TREND, 0, extremumTime, extremum, endTime, endPrice)) {
            // Set trendline properties
            ObjectSetInteger(0, label, OBJPROP_COLOR, GannFanColor);
            ObjectSetInteger(0, label, OBJPROP_STYLE, STYLE_SOLID);
            ObjectSetInteger(0, label, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, label, OBJPROP_RAY_RIGHT, true);
            ObjectSetString(0, label, OBJPROP_TOOLTIP, "Gann Fan " + angleNames[i]);
        }
        else {
            Print("Failed to create Gann Fan line: ", GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//|| Plotting a Reverse Gann Fan Line & nbsp; &nbsp ; ||
//+------------------------------------------------------------------+
void DrawOppositeGannFan(double extremum, datetime extremumTime)
{
    // Define the Gann angle and the corresponding name.
    double angles[] = { 82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5};
    string angleNames[] = { "1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"};

    // Calculate the end time of the sector line (current time)
    datetime endTime = TimeCurrent();

    // Draw sector lines for each angle
    for (int i = 0; i < ArraySize(angles); i++)
    {
        string label = "OppositeGannFan_" + angleNames[i];  // Object labels
        double angle = angles[i];                          // Current angle

        // Calculate price offset (based on time difference)
        double secondsDiff = endTime - extremumTime;
        double priceShift = MathTan(angle * M_PI / 180.0) * secondsDiff / PeriodSeconds();
        double endPrice;

        // Calculation of the ending price based on the direction of the opposite trend
        if (!isTrendUp) // Reverse trend
        {
            endPrice = extremum + priceShift;
        }
        else {
            endPrice = extremum - priceShift;
            angle = -angle; // Reversal angle in downtrend
        }

        // Create a trendline object
        if (ObjectCreate(0, label, OBJ_TREND, 0, extremumTime, extremum, endTime, endPrice)) {
            // Set trendline properties
            ObjectSetInteger(0, label, OBJPROP_COLOR, GannFanColor);
            ObjectSetInteger(0, label, OBJPROP_STYLE, STYLE_SOLID);
            ObjectSetInteger(0, label, OBJPROP_WIDTH, 1);
            ObjectSetInteger(0, label, OBJPROP_RAY_RIGHT, true);
            ObjectSetString(0, label, OBJPROP_TOOLTIP, "Opposite Gann Fan " + angleNames[i]);
        }
        else {
            Print("Failed to create Opposite Gann Fan line: ", GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//| Chart Event Handler Functions &nbsp nbsp; & & nbsp; |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,                  // Event ID
const long & lparam,            // Long integer parameters
                  const double & dparam,          // Double precision parameters
                  const string & sparam)         // String parameters
{
    // Redraw objects when the chart changes
    if (id == CHARTEVENT_CHART_CHANGE || id == CHARTEVENT_CLICK) {
        // Re-find the extremes and plot the Venn fan line.
        int rates_total = Bars(_Symbol, PERIOD_CURRENT);
        double high[], low[];
        datetime time[];
        ArraySetAsSeries(high, true);
        ArraySetAsSeries(low, true);
        ArraySetAsSeries(time, true);
        CopyHigh(_Symbol, PERIOD_CURRENT, 0, rates_total, high);
        CopyLow(_Symbol, PERIOD_CURRENT, 0, rates_total, low);
        CopyTime(_Symbol, PERIOD_CURRENT, 0, rates_total, time);

        FindExtremums(rates_total, high, low, time);
        DrawGannFan(extremumPrice, extremumTime);
        DrawOppositeGannFan(oppositeExtremumPrice, oppositeExtremumTime);
    }
}
```

![Price Action Analysis Toolkit Development (Part 14): Parabolic Stop and Reverse Tool](https://c.mql5.com/2/120/Price_Action_Analysis_Toolkit_Development_Part_14___LOGO.png)[Price Action Analysis Toolkit Development (Part 14): Parabolic Stop and Reverse Tool](https://www.mql5.com/en/articles/17234)

Embracing technical indicators in price action analysis is a powerful approach. These indicators often highlight key levels of reversals and retracements, offering valuable insights into market dynamics. In this article, we demonstrate how we developed an automated tool that generates signals using the Parabolic SAR indicator.

![Master MQL5 from beginner to pro (Part IV): About Arrays, Functions and Global Terminal Variables](https://c.mql5.com/2/87/Learning_MQL5_-_from_beginner_to_pro_Part_IV___LOGO.png)[Master MQL5 from beginner to pro (Part IV): About Arrays, Functions and Global Terminal Variables](https://www.mql5.com/en/articles/15357)

The article is a continuation of the series for beginners. It covers in detail data arrays, the interaction of data and functions, as well as global terminal variables that allow data exchange between different MQL5 programs.

![Neural Networks in Trading: Practical Results of the TEMPO Method](https://c.mql5.com/2/87/Neural_networks_in_trading__Practical_results_of_the_TEMPO_method___LOGO.png)[Neural Networks in Trading: Practical Results of the TEMPO Method](https://www.mql5.com/en/articles/15469)

We continue our acquaintance with the TEMPO method. In this article we will evaluate the actual effectiveness of the proposed approaches on real historical data.

![Animal Migration Optimization (AMO) algorithm](https://c.mql5.com/2/90/logo-amo_15543.png)[Animal Migration Optimization (AMO) algorithm](https://www.mql5.com/en/articles/15543)

The article is devoted to the AMO algorithm, which models the seasonal migration of animals in search of optimal conditions for life and reproduction. The main features of AMO include the use of topological neighborhood and a probabilistic update mechanism, which makes it easy to implement and flexible for various optimization tasks.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=rtdmxfcbmpnuaglgcbeeniysuohpynzv&ssn=1769093886721292116&ssn_dr=0&ssn_sr=0&fv_date=1769093886&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15556&back_ref=https%3A%2F%2Fwww.google.com%2F&title=William%20Gann%20methods%20(Part%20I)%3A%20Creating%20Gann%20Angles%20indicator%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909388679937442&fz_uniq=5049522587022765310&sv=2552)

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