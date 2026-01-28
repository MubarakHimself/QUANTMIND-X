---
title: From Novice to Expert: Higher Probability Signals
url: https://www.mql5.com/en/articles/20658
categories: Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:43:46.710912
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/20658&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5072049432792347489)

MetaTrader 5 / Examples


### Contents

- [Introduction and Concept Overview](https://www.mql5.com/en/articles/20658#para2)
- [Implementation](https://www.mql5.com/en/articles/20658#para3)
- [Testing](https://www.mql5.com/en/articles/20658#para4)
- [Conclusion](https://www.mql5.com/en/articles/20658#para5)
- [Key Lessons](https://www.mql5.com/en/articles/20658#para6)
- [Attachments](https://www.mql5.com/en/articles/20658#para7)

### Introduction and Concept Overview

In the [previous publication](https://www.mql5.com/en/articles/20645), we explored the idea of navigating market irregularities by using zones of probability rather than fixed price levels. That discussion laid an important foundation, but it also revealed a critical gap in our solution. While we successfully designed an algorithm capable of identifying high-probability support and resistance zones, we stopped short of building true trading intelligence around those zones.

Yes, we delivered a working indicator. However, from an algorithmic trading perspective, that solution remains incomplete. An indicator—despite being a form of automation—still transfers most of the workload to the trader. Decisions must be observed, interpreted, and executed manually. Full automation was not achieved, and that limitation matters.

In this article, we take the next decisive step forward.

The zones themselves are extremely valuable, but they represent only the context of a trade, not the trade decision itself. To progress toward a complete algorithmic system, we must enhance those zones with decision-making logic—intelligence that understands when and how to act inside them.

Before moving into implementation, it is important to clearly acknowledge the limitations of our previous solution:

1. No accurate signal-identification intelligence within the identified zones
2. No notification or alerting mechanism
3. No automatic or semi-automatic trade execution

In this phase of development, we address the first two limitations directly. Automatic trading will be deliberately reserved for a future stage, where the foundation we build today can be safely extended. Let us now establish the conceptual framework before translating it into MQL5 code.

Understanding Higher-Probability Entry Signals

Once high-probability support and resistance zones are plotted on the chart, the next challenge becomes precision. A zone alone is not a signal. Entering a trade simply because price has reached a zone is neither professional nor sustainable—especially for retail traders operating under limited capital.

For smaller accounts, precision is non-negotiable. Entries must be justified, filtered, and confirmed. Larger market participants may tolerate imprecision due to capital depth, but retail traders cannot afford prolonged drawdowns caused by poorly timed entries. This is where confirmation logic becomes essential.

Candlestick patterns and indicator-based confirmations provide that precision. When reversal or continuation patterns emerge within a predefined probability zone, the zone transforms from a passive area of interest into an active decision framework. In this sense, the zone validates the signal—and the signal validates the zone.

Many readers will already be familiar with classic candlestick formations. [Earlier](https://www.mql5.com/en/articles/download/17525) in this series, we developed a dedicated [candlestick pattern library](https://www.mql5.com/en/articles/download/17525/infinity_candlestick_pattern.mqh), and that work now becomes directly applicable. By observing price behavior inside the zones and analyzing the recurring candlestick characteristics that form there, we establish repeatable, programmable entry logic.

Why Automation and Notifications Matter

Manually monitoring charts for candle formations inside zones is both time-consuming and unreliable. Even experienced traders miss valid setups due to fatigue, distraction, or simple unavailability. This is precisely why automation is not a luxury—it is a necessity.

In this upgrade, we automate the process of watching for qualifying candlestick patterns inside probability zones. Even when manual execution is preferred, the system must be complete enough to notify the trader in real time. Alerts transform the workflow: instead of watching charts continuously, the trader responds only when the market presents a valid opportunity—much like responding to a phone notification.

This flexibility dramatically improves efficiency while reducing missed trades.

Trading with Structure Awareness

Although these zones are classified as higher probability, they are not immune to failure. Support and resistance can be invalidated by structural breakouts, and ignoring market context leads to unnecessary losses. Therefore, trade selection must respect the dominant market direction preceding the structure.

If a consolidation forms after a strong bullish impulse, we prioritize long opportunities. Conversely, after a strong bearish move, we favor short setups. Directional bias is not optional—it is a filter that protects capital.

At this point, the theoretical foundation of today’s advancement is complete. In the next section, we present visual illustrations of candlestick trigger patterns observed within the zones. From there, we transition into the MQL5 implementation, where these concepts are transformed into working, intelligent code.

![Manually identifying patterns inside zones](https://c.mql5.com/2/186/Inside_S_and_R_zones.png)

Fig. 1: Manually identifying patterns inside zones

As mentioned earlier, I worked directly with the tool to observe and identify candlestick formations occurring inside the predefined probability zones. If you refer to the results section of our previous article, you will notice that the same zone labeling appears in the illustration above (Fig. 1). This consistency is intentional and allows us to evaluate the behavior of price under identical structural conditions.

What stands out clearly in the highlighted example is the formation of strong and convincing reversal setups. We observe a bullish engulfing pattern followed by pin bars forming within the support zone. Shortly after these formations, price respected the zone and initiated a bullish move toward the previously identified resistance zone. Upon reaching that area, we also observe signs of resistance in price action, further validating both the zone-detection logic and the relevance of the candlestick confirmations.

At this point, the remaining gap becomes evident: the responsibility now lies with us to elevate the algorithm from a passive analytical tool into an intelligent system—one that actively informs us when a zone begins to produce high-quality candlestick signals. The zones already define where we should pay attention; the next task is to define when an actionable opportunity emerges.

To support what we observe in Fig. 1 above, let us expand the definitions of the two candlestick patterns involved so their significance is clearly understood.

A _bullish engulfing_ pattern is defined by a clear transition of control from sellers to buyers across two consecutive candles. The first candle is bearish, meaning its close is below its open. The second candle is bullish, with its close above its open. The defining characteristic of this pattern is that the body of the second candlestick engulfs the body of the first candle—its open forms below the previous candle’s close, and its close rises above the previous candle’s open. This structure reflects a decisive shift in market sentiment, where buying pressure not only cancels the prior bearish move but fully overwhelms it within a single trading period.

A _pin bar_, by contrast, is a single-candle rejection signal. It is characterized by a small real body—where the open and close are positioned very close to each other—and a long wick extending in one direction. In the case of a bullish pin bar, the lower wick is significantly longer than the candle body, while the close occurs near the upper end of the candle’s range. This formation indicates that price was aggressively pushed lower during the period but was strongly rejected, forcing the candle to close near its open or higher. The small body reflects temporary indecision, while the extended wick records a failed attempt to break to lower prices, ultimately resolving in favor of buyers. These definitions are critical because they rely solely on measurable price components: open, close, high, low, and body size. This makes them ideal candidates for algorithmic detection.

These precise structural characteristics are undoubtedly what we observe in the presented image. The patterns do not appear randomly; they emerge within a predefined support zone, reinforcing the idea that zones provide context while candlestick formations provide timing. Together, they form a complete decision framework rather than isolated signals.

This interaction between probabilistic zones and structurally defined candlestick confirmations forms the conceptual backbone of the intelligent confirmation system we now aim to encode in MQL5.

In the next section, we translate these visual and structural observations into clear, programmable rules and demonstrate how the algorithm detects and reacts to such formations in real time.

### Implementation

At this stage, all the conceptual building blocks are already in place. What remains is to connect them into a coherent, working solution. Our objective is no longer theoretical validation, but practical implementation.

From our previous work, we already possess three critical components of a professional trading system. First, the zone context, provided by [SRProbabilityZones.mq5](https://www.mql5.com/en/articles/download/20645/SRProbabilityZones.mq5), defines where price action deserves attention by identifying high-probability support and resistance areas. Second, the pattern intelligence, encapsulated in [infinity\_candlestick\_pattern.mqh](https://www.mql5.com/en/articles/download/17525/infinity_candlestick_pattern.mqh) from our [old article](https://www.mql5.com/en/articles/17525) as well, defines what qualifies as a meaningful market signal through well-structured candlestick recognition logic. Finally, we will introduce a notification layer, which determines when the trader is informed that a valid opportunity has emerged.

The missing link between these components is a zone-aware signal engine—a layer of logic that evaluates candlestick behavior only when price is interacting with a relevant zone. This is the point where analysis becomes intelligence.

At an implementation level, the logic is deliberately simple and explicit:

1. If price is trading inside a predefined support or resistance zone.
2. _And_ if a valid candlestick pattern is detected within that zone.
3. Then the system immediately notifies the trader using native MQL5 mechanisms such as alerts, push notifications, emails, or sound events.

In the sections that follow, we develop this signal engine step by step. We will integrate zone awareness with candlestick detection, implement robust notification handling, and ensure the system remains modular, efficient, and ready for future automation. Let's get started.

1\. File Header and Project Identity

We begin by defining the identity of the Expert Advisor. This step establishes authorship, versioning, and the official MQL5 Market reference. By clearly declaring ownership and metadata at the top of the file, we ensure that the project is traceable, publishable, and easy to maintain as it evolves. This is a standard but essential practice for professional-grade MQL5 development.

```
//+---------------------------------------------------------------------------------+
//|                                                 Pattern Zone Notification EA.mq5|
//|                                                               Clemence Benjamin |
//|https://www.mql5.com/go?link=https://www.mql5.com/en/users/billionaire2024/seller|
//+---------------------------------------------------------------------------------+

#property copyright "Clemence Benjamin"
#property link      "https://www.mql5.com/go?link=https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.00"
```

2\. Integrating the Candlestick Pattern Intelligence Layer

At this stage, we integrate the candlestick intelligence engine into the EA. Rather than embedding pattern-detection logic directly in the main file, we rely on an external pattern library. This approach keeps the system modular and scalable. All candlestick definitions, measurements, and validation rules remain isolated in a dedicated include file, allowing us to upgrade or extend pattern logic without modifying the EA’s core structure.

```
#include <infinity_Candlestick_Pattern.mqh>
```

3\. User-Configurable Alert and Behavior Settings

We then expose a set of input parameters that allow traders to control how the EA behaves. These settings define notification channels, alert limits, visual feedback, and execution timing. By placing these options in the input section, we make the system flexible and trader-friendly, enabling customization without requiring code changes.

```
input bool   EnableEmailNotifications = true;
input bool   EnablePushNotifications  = true;
input bool   EnableAlertSounds        = true;
input bool   EnableOnScreenAlerts     = true;
input string AlertSoundFile           = "alert.wav";
input bool   CheckOnNewBar            = true;
input int    MaxAlertsPerDay          = 10;
input bool   DrawPatternArrows        = true;
input color  BullishArrowColor        = clrGreen;
input color  BearishArrowColor        = clrRed;
```

4\. Zone Configuration Parameters and Context Alignment

To maintain alignment with the support and resistance framework, we replicate the same configuration parameters used by the zone indicator. This ensures that the EA evaluates price action using the same structural context displayed on the chart. Consistency here is critical: the EA must calculate the same zones the trader sees to  generate meaningful confirmations.

```
input int    Zone_LookBackPeriod   = 150;
input int    Zone_SwingSensitivity = 5;
input int    Zone_ZonePadding      = 20;
input color  Zone_SupportColor     = clrRoyalBlue;
input color  Zone_ResistanceColor  = clrCrimson;
input bool   Zone_ShowZoneLabels   = true;
```

5\. Global State Management and Data Buffers

We declare global variables to manage execution state, alert tracking, indicator handles, and price buffers. These variables persist across ticks and allow the EA to retain context, such as the last processed bar, the number of alerts sent during the day, and historical price data required for multi-candle pattern detection. This layer forms the backbone of the EA’s internal memory.

```
datetime lastBarTime = 0;
int alertsToday = 0;
datetime lastAlertDate = 0;

MqlRates rates[];
double highs[], lows[], opens[], closes[];
double atrValues[];

int atrHandle = INVALID_HANDLE;
int zoneIndicatorHandle = INVALID_HANDLE;
```

6\. Zone Indicator Buffer Mapping

We define explicit mappings for the support and resistance indicator buffers. This allows the EA to read zone boundaries directly from the indicator when it is available on the chart. By formalizing these buffer indices, we create a clean and reliable communication bridge between the EA and the indicator.

```
#define ZONE_SUPPORT_HIGH_BUFFER 0
#define ZONE_SUPPORT_LOW_BUFFER  1
#define ZONE_RESISTANCE_HIGH_BUFFER 2
#define ZONE_RESISTANCE_LOW_BUFFER  3
```

7\. Pattern Name Registry

To produce clear and human-readable alerts, we initialize a registry of pattern names. Each detected candlestick formation is mapped to a descriptive label. This ensures that notifications are informative and immediately understandable, rather than exposing internal function names or cryptic identifiers to the trader.

```
string patternNames[24];
```

```
void InitializePatternNames()
{
    patternNames[0]  = "Morning Star";
    patternNames[1]  = "Evening Star";
    patternNames[2]  = "Three White Soldiers";
    ...
    patternNames[23] = "Spinning Top";
}
```

8\. Expert Advisor Initialization Sequence

During the initialization phase, we prepare all core components. We load the ATR indicator for volatility-aware pattern evaluation, attempt to connect to the zone indicator, initialize all arrays as time series, and establish the reference for the most recent bar. If on-screen alerts are enabled, we also construct the notification panel at this stage. This ensures the EA starts in a synchronized and fully operational state.

```
int OnInit()
{
    InitializePatternNames();

    atrHandle = iATR(_Symbol, _Period, 14);
    if(atrHandle == INVALID_HANDLE)
        return(INIT_FAILED);

    zoneIndicatorHandle = iCustom(_Symbol, _Period, "SRProbabilityZones",
                                  Zone_LookBackPeriod,
                                  Zone_SwingSensitivity,
                                  Zone_ZonePadding,
                                  Zone_SupportColor,
                                  Zone_ResistanceColor,
                                  Zone_ShowZoneLabels);

    ArraySetAsSeries(rates, true);
    ArraySetAsSeries(opens, true);
    ArraySetAsSeries(highs, true);
    ArraySetAsSeries(lows, true);
    ArraySetAsSeries(closes, true);
    ArraySetAsSeries(atrValues, true);

    lastBarTime = iTime(_Symbol, _Period, 0);
    lastAlertDate = TimeCurrent();

    if(EnableOnScreenAlerts)
        CreateNotificationPanel();

    Print("Pattern Zone Notification EA initialized successfully");
    return(INIT_SUCCEEDED);
}
```

9\. Robust Zone Data Acquisition Strategy

To ensure reliability, we design a multi-layered zone acquisition process. The EA first attempts to read zone values from global variables, then from indicator buffers, and finally falls back to direct zone calculation if necessary. This redundancy guarantees that the EA continues functioning even if one data source becomes unavailable, which is especially important during testing and deployment.

```
bool GetZoneData(double &supportHigh, double &supportLow,
                 double &resistanceHigh, double &resistanceLow)
{
    ...
}
```

10\. Tick Processing and Execution Control

Within the tick-processing logic, we control when the EA evaluates the market. We optionally restrict execution to new bars only, reset daily alert counters when a new trading day begins, and enforce a maximum alert limit. These safeguards prevent over-notification and unnecessary computation while keeping the system responsive and efficient.

```
void OnTick()
{
    if(CheckOnNewBar)
    {
        datetime currentBarTime = iTime(_Symbol, _Period, 0);
        if(currentBarTime == lastBarTime)
            return;
        lastBarTime = currentBarTime;
    }

    CheckDailyReset();

    if(alertsToday >= MaxAlertsPerDay)
        return;

    ...
    CheckPatternsAtBar(0, supportHigh, supportLow, resistanceHigh, resistanceLow);
}
```

11\. Zone-Aware Pattern Evaluation Logic

This step represents the core intelligence of the system. Once zone boundaries are available, we evaluate whether the current price lies inside a support or resistance zone. Only when this condition is met do we proceed to analyze candlestick patterns. Bullish patterns are evaluated exclusively within support zones, while bearish patterns are evaluated within resistance zones. This contextual filtering prevents signals from appearing randomly and enforces disciplined, location-based trading logic.

```
void CheckPatternsAtBar(int index, double supHigh, double supLow,
                        double resHigh, double resLow)
{
    double currentPrice = closes[index];
    datetime patternTime = rates[index].time;

    bool inSupportZone = (currentPrice >= supLow && currentPrice <= supHigh);
    bool inResistanceZone = (currentPrice >= resLow && currentPrice <= resHigh);

    if(inSupportZone)
        CheckBullishPatterns(index, patternTime, currentPrice, "Support Zone");

    if(inResistanceZone)
        CheckBearishPatterns(index, patternTime, currentPrice, "Resistance Zone");
}
```

12\. Pattern Confirmation, Alerts, and Visual Feedback

When a valid pattern is detected within its appropriate zone, the EA notifies the trader through all enabled channels. Alerts may be delivered via terminal pop-ups, sounds, emails, push notifications, and on-screen messages. In addition, the EA can draw arrows and labels directly on the chart, providing immediate visual confirmation of where and why the signal occurred.

```
void SendAlert(string message, datetime time, double price,
               string zone, bool isBullish)
{
    Alert(message);
    if(EnableOnScreenAlerts)
        ShowOnScreenNotification(message, isBullish);

    if(EnableAlertSounds)
        PlaySound(AlertSoundFile);

    if(EnableEmailNotifications)
        SendMail("Pattern Alert", message);

    if(EnablePushNotifications)
        SendNotification(message);

    alertsToday++;
    lastAlertDate = TimeCurrent();
}
```

With this implementation, we move beyond passive zone visualization and introduce an intelligent, context-aware confirmation engine. Support and resistance zones define where attention is required, candlestick patterns define when opportunity emerges, and the EA connects both into a structured, automated decision-support system. This architecture lays a solid foundation for further evolution toward fully automated trading in subsequent stages. The complete source code is provided at the end of this article. In addition, we have included references to the resources and prior work on which this implementation builds, making it easier for readers to understand the design choices and extend or adapt the solution. In the next section, I will share my strategy tester results.

### Testing

With the Expert Advisor successfully compiled in MetaEditor, we proceeded to evaluate its performance using the Strategy Tester. This phase allowed us to simulate historical market conditions, observe how the EA reacts to price movements within SUPPORT and RESISTANCE zones, and verify that pattern detection and notification mechanisms function as intended. By collecting detailed logs during the tests, we can analyze signal accuracy, timing, and the consistency of alerts, providing valuable insights for further refinement and optimization.  The following animated screen capture and log illustrate the EA in action, highlighting its behavior and alerts throughout the testing process.

![Testing higher probability signals](https://c.mql5.com/2/187/ShareX_V61j2cVnRr.gif)

Fig. 2: Testing

```
2025.12.15 14:08:31.087   program file added: \Indicators\SRProbabilityZones.ex5. 13720 bytes loaded
2025.12.15 14:08:31.089   2024.01.01 00:00:00   SRProbabilityZones indicator handle created successfully
2025.12.15 14:08:31.128   2024.01.01 00:00:00   Pattern Zone Notification EA initialized successfully

2025.12.15 14:08:38.093   2024.01.02 00:18:30   Alert: GBPUSD - Bullish Marubozu detected in Support Zone at 2024.01.02 00:10 Price: 1.27216
2025.12.15 14:08:38.093   GBPUSD - Bullish Marubozu detected in Support Zone at 2024.01.02 00:10 Price: 1.27216

2025.12.15 14:08:38.096   2024.01.02 00:20:30   Alert: GBPUSD - Hammer detected in Support Zone at 2024.01.02 00:15 Price: 1.27236
2025.12.15 14:08:38.096   GBPUSD - Hammer detected in Support Zone at 2024.01.02 00:15 Price: 1.27236

2025.12.15 14:08:38.128   2024.01.02 00:30:00   Alert: GBPUSD - Bullish Marubozu detected in Support Zone at 2024.01.02 00:25 Price: 1.27253
2025.12.15 14:08:38.128   GBPUSD - Bullish Marubozu detected in Support Zone at 2024.01.02 00:25 Price: 1.27253

2025.12.15 14:08:38.291   2024.01.02 00:45:30   Alert: GBPUSD - Bullish Marubozu detected in Support Zone at 2024.01.02 00:40 Price: 1.27289
2025.12.15 14:08:38.291   GBPUSD - Bullish Marubozu detected in Support Zone at 2024.01.02 00:40 Price: 1.27289

2025.12.15 14:08:38.291   2024.01.02 00:45:30   Alert: GBPUSD - Tweezer Bottom detected in Support Zone at 2024.01.02 00:35 Price: 1.27254
2025.12.15 14:08:38.291   GBPUSD - Tweezer Bottom detected in Support Zone at 2024.01.02 00:35 Price: 1.27254

2025.12.15 14:10:36.075   2024.01.02 01:35:00   Alert: GBPUSD - Tweezer Bottom detected in Support Zone at 2024.01.02 01:25 Price: 1.27303
2025.12.15 14:10:36.075   GBPUSD - Tweezer Bottom detected in Support Zone at 2024.01.02 01:25 Price: 1.27303

2025.12.15 14:12:07.303   2024.01.03 00:25:00   Alert: GBPUSD - Bullish Marubozu detected in Support Zone at 2024.01.03 00:20 Price: 1.26149
2025.12.15 14:12:07.303   GBPUSD - Bullish Marubozu detected in Support Zone at 2024.01.03 00:20 Price: 1.26149

2025.12.15 14:12:07.444   2024.01.03 01:40:00   Alert: GBPUSD - Bearish Harami detected in Resistance Zone at 2024.01.03 01:30 Price: 1.26210
2025.12.15 14:12:07.444   GBPUSD - Bearish Harami detected in Resistance Zone at 2024.01.03 01:30 Price: 1.26210

2025.12.15 14:12:08.088   2024.01.03 03:05:00   Alert: GBPUSD - Shooting Star detected in Resistance Zone at 2024.01.03 03:00 Price: 1.26294
2025.12.15 14:12:08.089   GBPUSD - Shooting Star detected in Resistance Zone at 2024.01.03 03:00 Price: 1.26294
```

The log shows that both the SRProbabilityZones indicator and the Pattern Zone Notification EA initialized successfully without errors, confirming that the system setup is correct. The EA accurately detects candlestick patterns, distinguishing bullish patterns in support zones and bearish patterns in resistance zones. Alerts are logged with precise timestamps and prices, indicating that the EA correctly captures market conditions and maps historical data accurately for replay or testing.

Multiple alerts appear close together in time, suggesting the EA is sensitive to consecutive pattern formations on nearby bars, which can be useful for spotting rapid market shifts but may also generate overlapping notifications. The logging is consistent, with each alert appearing in both terminal and expert logs, and the on-screen notifications and counters function as intended. Overall, the system demonstrates reliable pattern detection and effective alerting within defined support and resistance zones.

### Conclusion

Today, we took a significant step forward by developing an EA that identifies higher-probability trading signals within defined SUPPORT and RESISTANCE zones. The modular design of the EA was made possible by integrating our previous work into the project—a flexibility that the MQL5 platform offers. By leveraging an already-prepared candlestick pattern library, we were able to streamline the development process and focus on the final solution efficiently. Importantly, as observed during testing, signals were generated exclusively within these zones, promoting systematic trading and reducing uncertainty for traders.

We enhanced the EA further by integrating a comprehensive notification system that alerts us whenever a signal is produced, ensuring timely awareness of key market opportunities. To consolidate the learning experience, I have prepared a summary of key lessons for everyone, and the complete source codes are attached below. You are encouraged to experiment, modify, and discuss these ideas in the comments. Until our next publication, stay tuned for more insights and practical implementations.

### Key Lessons

| Key Lessons | Description |
| --- | --- |
| 1\. Global variables for inter-component communication. | Use GlobalVariableSet()/GlobalVariableGet() to share data between indicators and EAs, enabling modular design where components work independently but share data. |
| 2\. Proper time comparison in MQL5. | Always use TimeToStruct() for date comparisons instead of TimeDay()/TimeMonth(). The struct method is more reliable and avoids issues with uninitialized datetime variables. |
| 3\. Variable naming conflicts. | When including libraries, ensure your variable names don't conflict with library parameter names. Rename local variables (patternATR → atrValues) to avoid "hides global variable" warnings. |
| 4\. Strategy Tester limitations. | The MT5 Strategy Tester can only run one expert at a time. For systems requiring multiple components, either combine them into one EA or test on live charts. |
| 5\. Multiple data source fallbacks. | Implement fallback mechanisms: first check global variables, then indicator buffers, then calculate internally. This makes your EA robust in different environments. |
| 6\. Comprehensive initialization. | Always initialize all global variables in OnInit() with valid values. Don't rely on global scope initialization for datetime variables that need TimeCurrent(). |
| 7\. Modular function design. | Separate concerns into dedicated functions (CheckDailyReset(), GetZoneData(), CheckPatternsAtBar()). This improves readability, testing, and maintenance. |
| 8\. Multi-channel alert systems. | Implement multiple notification methods (terminal alert, email, push, sound, on-screen) to ensure traders don't miss important signals. |
| 9\. Resource management. | Always release indicator handles (IndicatorRelease()) and delete chart objects in OnDeinit(). Prevent memory leaks and clean up chart visual elements. |
| 10\. User configuration options. | Provide input parameters for key settings (alert limits, notification toggles, visual preferences). This allows users to customize without code changes. |
| 11\. Error handling with graceful degradation. | When components fail (indicator not found, data unavailable), log errors but continue running with reduced functionality rather than crashing. |
| 12\. Visual feedback for users. | Provide on-screen panels showing system status, counters, and temporary alerts. Visual feedback helps users understand what the system is doing in real-time. |

### Attachments

| Source File Name | Version | Description |
| --- | --- | --- |
| [PatternZoneNotificationEA.mq5](https://www.mql5.com/en/articles/download/20658/PatternZoneNotificationEA.mq5 ".mq5") | 1.00 | Expert Advisor that detects candlestick patterns within support and resistance zones, sends alerts via terminal, email, push notifications, and displays on-chart notifications with optional arrows. |
| [SRProbabiblityZones.mq5](https://www.mql5.com/en/articles/download/20658/SRProbabilityZones.mq5) | 1.00 | Indicator that calculates dynamic support and resistance zones using swing points and zone padding; used by the EA for pattern alerting. |
| I [nfinity\_Candlestick\_Pattern.mqh](https://www.mql5.com/en/articles/17525 ".mqh") | 1.00 | Library containing functions for detecting multiple candlestick patterns, including single- and multi-candle formations, used by the EA for pattern recognition. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20658.zip "Download all attachments in the single ZIP archive")

[PatternZoneNotificationEA.mq5](https://www.mql5.com/en/articles/download/20658/PatternZoneNotificationEA.mq5 "Download PatternZoneNotificationEA.mq5")(29.71 KB)

[SRProbabilityZones.mq5](https://www.mql5.com/en/articles/download/20658/SRProbabilityZones.mq5 "Download SRProbabilityZones.mq5")(13.91 KB)

[infinity\_candlestick\_pattern.mqh](https://www.mql5.com/en/articles/download/20658/infinity_candlestick_pattern.mqh "Download infinity_candlestick_pattern.mqh")(13.91 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/502971)**

![Creating a mean-reversion strategy based on machine learning](https://c.mql5.com/2/124/Creating_a_Mean_Reversion_Strategy_Based_on_Machine_Learning__LOGO.png)[Creating a mean-reversion strategy based on machine learning](https://www.mql5.com/en/articles/16457)

This article proposes another original approach to creating trading systems based on machine learning, using clustering and trade labeling for mean reversion strategies.

![Billiards Optimization Algorithm (BOA)](https://c.mql5.com/2/123/Billiards_Optimization_Algorithm__LOGO__4.png)[Billiards Optimization Algorithm (BOA)](https://www.mql5.com/en/articles/17325)

The BOA method is inspired by the classic game of billiards and simulates the search for optimal solutions as a game with balls trying to fall into pockets representing the best results. In this article, we will consider the basics of BOA, its mathematical model, and its efficiency in solving various optimization problems.

![Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://c.mql5.com/2/188/20571-data-science-and-ml-part-47-logo.png)[Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)

In this article, we will attempt to predict the market with a decent model for time series forecasting named DeepAR. A model that is a combination of deep neural networks and autoregressive properties found in models like ARIMA and Vector Autoregressive (VAR).

![Larry Williams Market Secrets (Part 2): Automating a Market Structure Trading System](https://c.mql5.com/2/187/20512-larry-williams-market-secrets-logo.png)[Larry Williams Market Secrets (Part 2): Automating a Market Structure Trading System](https://www.mql5.com/en/articles/20512)

Learn how to automate Larry Williams market structure concepts in MQL5 by building a complete Expert Advisor that reads swing points, generates trade signals, manages risk, and applies a dynamic trailing stop strategy.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ybyljngkscvvammyxtjuvotgootghzki&ssn=1769193825239061651&ssn_dr=0&ssn_sr=0&fv_date=1769193825&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20658&back_ref=https%3A%2F%2Fwww.google.com%2F&title=From%20Novice%20to%20Expert%3A%20Higher%20Probability%20Signals%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919382541711574&fz_uniq=5072049432792347489&sv=2552)

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