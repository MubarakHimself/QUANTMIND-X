---
title: From Novice to Expert: Forex Market Periods
url: https://www.mql5.com/en/articles/20005
categories: Trading Systems, Integration
relevance_score: 0
scraped_at: 2026-01-24T13:44:53.780417
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=igjpgidrqyuodhlzuaznbukamqwxftmm&ssn=1769251492820967898&ssn_dr=0&ssn_sr=0&fv_date=1769251492&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20005&back_ref=https%3A%2F%2Fwww.google.com%2F&title=From%20Novice%20to%20Expert%3A%20Forex%20Market%20Periods%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925149221663453&fz_uniq=5083082293242762403&sv=2552)

MetaTrader 5 / Trading systems


### Contents

- [Introduction](https://www.mql5.com/en/articles/20005#para1)
- [Implementation](https://www.mql5.com/en/articles/20005#para2)
- [Testing](https://www.mql5.com/en/articles/20005#para3)
- [Conclusion](https://www.mql5.com/en/articles/20005#para4)
- [Key Lessons](https://www.mql5.com/en/articles/20005#para5)
- [Attachments](https://www.mql5.com/en/articles/20005#para6)

### Introduction

In my town, I once took time to observe how the local vegetable market operates daily—and what I discovered was quite enlightening. Every morning, farmers arrive early with their fresh produce, and retailers flock in just as early to secure the best wholesale deals. During those first few hours after the market opens, activity reaches its peak—transactions happen rapidly, and prices often shift dynamically. As the day progresses toward midday and lunch hours, the pace gradually slows down; fewer retailers arrive, and many farmers begin packing up. Yet, as evening approaches and the market nears closing time, activity surges once more, with late buyers and sellers making their final moves before sunset.

This simple marketplace behavior strikingly mirrors how the Forex market functions. Just like the farmers and retailers, traders across the world operate within distinct sessions—periods of high and low activity that define the rhythm of the market. There are peak hours when major transactions take place, shaping the character and momentum of the trading day.

Today’s challenge is to leverage MQL5 to visualize these market session periods, aligning them seamlessly with higher- and lower-timeframe structures. By synchronizing the open and close of these sessions, we aim to better understand how each one influences the next—and how timing truly powers price movement. Another fascinating mystery to uncover lies in the realization that financial candlesticks themselves represent time-based bars—typically denoting hourly, daily, or weekly periods. By applying the same concept to market sessions, we can create session-based candlesticks, each reflecting the unique characteristics of that trading window. Just like traditional timeframes, every session possesses its own Open, High, Low, and Close (OHLC) structure, telling a distinct story of activity, sentiment, and volatility within that period.

The most notable difference, however, is that session bars often overlap, especially for global markets that share trading hours. For example, the London and New York sessions Fig.1 intersect for several hours, creating periods of intensified volatility and liquidity. Unlike standard timeframe bars that follow one after another in sequence, session bars coexist across time, reflecting the simultaneous participation of multiple regions. This overlap is precisely what gives rise to dynamic transitions in market sentiment, making session-based analysis an essential complement to traditional time-based structures.

![London and New York Forex Market Sessions Conceptual](https://c.mql5.com/2/178/sessions.png)

Fig. 1. Conceptual Overlap Between the London and New York Trading Sessions

Through this development, we aim to visualize session bars in the same analytical language as standard candlesticks—allowing traders to recognize session-specific behavior, identify where liquidity peaks, and compare how one session’s sentiment flows into the next. Ultimately, this feature bridges the understanding between conventional timeframe analysis and the hidden rhythm of market sessions, fostering a deeper connection between time, structure, and price dynamics.

To bring this idea to life, I will outline the full implementation below—explaining the underlying concept in detail and then presenting the complete MQL5 code in the following section of this discussion.

### Implementation

Our exploration of market periods continues to evolve as we introduce another powerful feature—one that unites conventional timeframe analysis with the dynamics of global Forex trading sessions. This enhancement bridges the gap between structured time periods and real-world session behavior, offering a more realistic view of how markets breathe throughout the day. To implement this new feature, we will first develop and test it in isolation, ensuring its accuracy and performance, before integrating it into the existing [Market Periods Synchronizer.](https://www.mql5.com/en/articles/19919) The result will be a more advanced and insightful version of the tool—one that captures both temporal structure and session-driven market behavior in perfect harmony.

Another reason for implementing this feature in isolation is to ensure that every reader can fully grasp the concept before integration. It also serves as good development practice, especially as the program grows more complex with each new feature. By isolating components, we make the system easier to test, debug, and understand—both for learners following along and for developers extending the tool in future versions.

To put this concept into practice, we’ll follow a two-step workflow: first, developing the feature in isolation, and then integrating the validated component into the [Market Periods Synchronizer EA.](https://www.mql5.com/en/articles/download/19919/MarketPeriodsSynchronizer_EA.mq5) This approach ensures that the logic remains modular, easy to test, and adaptable for future enhancements.

- Step 1—Development in Isolation: We begin by creating a standalone CSessionVisualizer class responsible for defining trading sessions, computing their OHLC values, and rendering them as filled rectangles that mimic candlesticks (optionally including wicks and labels). This standalone version should be tested independently—such as by plotting sessions on an EURUSD H1 chart—to verify the accuracy of session boundaries and data visualization.

- Step 2—Integration Phase: Once verified, the session visualizer can be integrated into the main EA. This involves adding new UI controls such as a “Show Sessions: ON/OFF” toggle, a session color picker, and a lookback setting to determine how many historical sessions to display. The RefreshLines() function will be extended to call the session visualizer after drawing major and minor lines, ensuring all elements stay synchronized. Shared global variables (like g\_Lookback) will maintain consistency, while visual overlaps between sessions will be managed using semi-transparent layering for clarity.

**Step1—Development in Isolation**

In this section, we’ll build a modular class called CSessionVisualizer, designed to detect, calculate, and visualize Forex market sessions on a chart—complete with candlestick-style bodies, optional wicks, and labels. The approach emphasizes modularity, readability, and reusability, allowing the feature to be integrated later into larger frameworks such as the Market Periods Synchronizer EA.

1\. Header Setup and Purpose

Every good MQL5 module begins with a proper header that clearly defines what the script is, who wrote it, and what it’s for. This helps others understand your code and maintains development hygiene for version control.

We also define a small but powerful [ARGB](https://www.mql5.com/en/docs/common/resourcecreate) macro, which allows us to easily create colors with transparency—essential for blending multiple visual layers (like sessions overlapping on a chart).

```
//+------------------------------------------------------------------+
//| SessionVisualizerTest.mqh                                        |
//| Modular class for Forex session OHLC visualization               |
//| Author: Clemence Benjamin                                        |
//| Version: 1.00                                                    |
//+------------------------------------------------------------------+
#property strict

// ARGB Macro: Creates a color with alpha (transparency)
#define ARGB(a,r,g,b) ((color)(((uchar)(a))<<24)|(((uchar)(r))<<16)|(((uchar)(g))<<8)|((uchar)(b)))

#include <Object.mqh>  // Provides access to chart object manipulation
```

Here, we’re using [Object.mqh](https://www.mql5.com/en/docs/standardlibrary/cobject) because the visualizer creates and manages multiple chart objects (rectangles, labels, lines). The macro ARGB() allows us to produce semi-transparent colors like ARGB(120,255,0,0) for a see-through red fill—vital for visual layering. But later we will remove ARGB to use one from CCanvas.

2\. Session Definitions and Struct Setup

Before drawing anything, we must define what a “session” means in our context. Each Forex session—Sydney, Tokyo, London, and New York—has its unique open and close hours. We use an enum to define them symbolically and a struct to store the properties of each session, such as its name, timing, color, and whether it’s currently active.

```
enum SESSION_TYPE {
   SESSION_SYDNEY = 0,  // 22:00-07:00 GMT
   SESSION_TOKYO   = 1,  // 00:00-09:00 GMT
   SESSION_LONDON  = 2,  // 08:00-17:00 GMT
   SESSION_NEWYORK = 3   // 13:00-22:00 GMT
};

struct SessionInfo {
   SESSION_TYPE type;
   string       name;       // Short code e.g., "LON"
   int          open_hour;  // GMT open hour
   int          close_hour; // GMT close hour
   color        sess_color; // Visual color on chart
   bool         enabled;    // Whether to render this session
};
```

Using enums ensures that our code is easy to read and extend later. For instance, adding a “Custom Session” would only require one extra entry and struct initialization.

3\. Input Parameters and Class Initialization

Next, we define input parameters and prepare our class constructor. This setup allows traders or developers to customize lookback periods, colors, and wick behavior directly from the EA’s input dialog.

```
input int    InpSessionLookback = 10;  // Days of historical sessions
input bool   InpShowSessionWicks = false;
input int    InpWickAlpha = 120;
input color  InpFillBull = clrLime;
input color  InpFillBear = clrPink;

class CSessionVisualizer : public CObject {
private:
   SessionInfo m_sessions[4];
   int         m_gmt_offset;
   string      m_prefix;

public:
   CSessionVisualizer(string prefix = "SESS_") : m_prefix(prefix) {
      // Initialize all session parameters (GMT-based)
      m_sessions[SESSION_SYDNEY] = {SESSION_SYDNEY, "SYD", 22, 7, clrAqua, true};
      m_sessions[SESSION_TOKYO]  = {SESSION_TOKYO,  "TOK", 0, 9, clrYellow, true};
      m_sessions[SESSION_LONDON] = {SESSION_LONDON, "LON", 8, 17, clrRed, true};
      m_sessions[SESSION_NEWYORK]= {SESSION_NEWYORK,"NY", 13, 22, clrBlue, true};

      // Basic GMT offset estimation
      MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
      m_gmt_offset = -dt.hour % 24;
   }

   ~CSessionVisualizer() {
      DeleteAllSessionObjects();
   }
```

Here, m\_prefix helps prevent object naming conflicts on the chart, especially when multiple tools are used simultaneously. For example, each object may be labeled like "SESS\_LON\_O\_1730764800."

4\. Refreshing and Drawing Sessions

The RefreshSessions() method is the heart of the visualizer. It wipes any previous session drawings and regenerates them over a lookback period defined by the user. This modular design makes it possible to refresh visuals on every new bar or timer tick without cluttering the chart.

```
void RefreshSessions(int lookback_days = 10) {
   DeleteAllSessionObjects();  // Clear previously drawn sessions
   datetime end_time = TimeCurrent();
   datetime start_time = end_time - (lookback_days * 86400);

   for (int day = 0; day < lookback_days; day++) {
      datetime day_start = start_time + (day * 86400);
      for (int s = 0; s < 4; s++) {
         if (!m_sessions[s].enabled) continue;
         DrawSessionForDay(m_sessions[s], day_start);
      }
   }
   ChartRedraw();
}
```

This structure is loop-based for efficiency and modularity. Each day and each session gets processed individually—a technique that avoids overlapping session artifacts or missing data ranges.

5\. Drawing Individual Sessions

The DrawSessionForDay() function takes care of actually rendering each session. It determines open and close times (adjusted for GMT offset), retrieves price data for that window, and visualizes the session as a candlestick-like rectangle with optional wicks.

```
void DrawSessionForDay(const SessionInfo &sess, datetime day_start) {
   MqlDateTime dt_open, dt_close;
   TimeToStruct(day_start, dt_open);
   dt_open.hour = sess.open_hour + m_gmt_offset;
   datetime t_open = StructToTime(dt_open);

   TimeToStruct(day_start, dt_close);
   dt_close.hour = sess.close_hour + m_gmt_offset;
   datetime t_close = StructToTime(dt_close);

   if (sess.close_hour < sess.open_hour) t_close += 86400;  // Overnight fix

   double o,h,l,c;
   if (!GetOHLCInTimeRange(t_open, t_close, o,h,l,c)) return;

   color body_col = (c > o) ? InpFillBull : InpFillBear;
   string rect = m_prefix + sess.name + "_B_" + IntegerToString((int)t_open);
   ObjectCreate(0, rect, OBJ_RECTANGLE, 0, t_open, MathMin(o,c), t_close, MathMax(o,c));
   ObjectSetInteger(0, rect, OBJPROP_BGCOLOR, body_col);
   ObjectSetInteger(0, rect, OBJPROP_FILL, true);
   ObjectSetInteger(0, rect, OBJPROP_BACK, true);
}
```

This approach gives us session bars that behave like candlesticks—a new way to analyze sessions as their own micro timeframes.

6\. Retrieving Session OHLC Data

The GetOHLCInTimeRange() function extracts accurate open, high, low, and close data from the selected timeframe within each session’s duration. Using iBarShift() ensures precise alignment with existing chart bars.

```
bool GetOHLCInTimeRange(datetime start, datetime end, double &open, double &high, double &low, double &close) {
   int shift_start = iBarShift(_Symbol, _Period, start, false);
   int shift_end   = iBarShift(_Symbol, _Period, end, false);
   if (shift_start < 0 || shift_end < 0) return false;
   int count = shift_start - shift_end + 1;
   if (count <= 0) return false;

   double o[],h[],l[],c[];
   ArraySetAsSeries(o,true); ArraySetAsSeries(h,true); ArraySetAsSeries(l,true); ArraySetAsSeries(c,true);
   CopyOpen(_Symbol,_Period,shift_end,count,o);
   CopyHigh(_Symbol,_Period,shift_end,count,h);
   CopyLow(_Symbol,_Period,shift_end,count,l);
   CopyClose(_Symbol,_Period,shift_end,count,c);

   open = o[count-1]; high = h[ArrayMaximum(h,0,count)];
   low = l[ArrayMinimum(l,0,count)]; close = c[0];
   return true;
}
```

This provides the “session candlestick” OHLC data from standard price series—without re-querying external sources.

7\. Cleanup and Object Management

Good code always cleans up after itself. The DeleteAllSessionObjects() function ensures that old objects are removed when refreshing or deinitializing. Without this, you’d end up with thousands of leftover rectangles after long testing sessions.

```
void DeleteAllSessionObjects() {
   int total = ObjectsTotal(0);
   for (int i = total - 1; i >= 0; i--) {
      string name = ObjectName(0, i);
      if (StringFind(name, m_prefix) == 0)
         ObjectDelete(0, name);
   }
}
```

Developing the test EA:

We begin with a compact header and the include. This EA is intentionally tiny: it only boots the visualizer, starts a timer for periodic refreshes (so you can see sessions update while the market runs), and cleans up on exit. Keeping test harnesses small helps isolate problems and validate class behavior before integration into larger systems.

```
//+------------------------------------------------------------------+
//| SessionTest.mq5 - Standalone test for CSessionVisualizer         |
//| Purpose: lightweight EA to exercise the SessionVisualizer class  |
//+------------------------------------------------------------------+
#property strict

#include <SessionVisualizerTest.mqh>
```

Next, we declare the runtime instance. Using a pointer and new gives explicit control over lifetime: we create the visualizer in OnInit() and delete it in OnDeinit() so the destructor runs immediately and removes objects the class created. This pattern is convenient for tests and mirrors how you would manage objects in larger EAs that create/destroy subsystems dynamically.

```
// runtime pointer to the session visualizer
CSessionVisualizer *g_session_vis = NULL;
```

OnInit() is where we construct the visualizer, set a refresh timer, and perform the initial draw. The example uses a 30-second demo timer, short enough to see updates often but long enough to avoid excessive CPU usage. We pass a unique prefix to the visualizer so its objects are clearly namespaced (handy when testing multiple tools on the same chart). Immediately calling RefreshSessions(5) will render five days of sessions so you can verify historical plotting right away.

```
int OnInit()
{
   // allocate the session visualizer with a test-specific prefix to avoid name clashes
   g_session_vis = new CSessionVisualizer("TEST_SESS_");

   // Refresh every 30 seconds during testing (demo). Adjust for production usage.
   EventSetTimer(30);

   // initial rendering of the last 5 days (quick sanity check)
   if(g_session_vis != NULL)
      g_session_vis.RefreshSessions(5);

   return(INIT_SUCCEEDED);
}
```

Note: we do a null-check before calling the method. If new fails for any reason (very rare on normal desktops), this prevents a crash and makes the failure easier to detect in logs.

OnTimer() drives periodic updates. For session visuals, this is useful because session OHLCs can change slightly on live bars (depending on the timeframe you use to compute OHLC), and frequent redraws let you watch session wicks and bodies evolve. In a production EA, you may prefer to refresh only on new bars or on a longer interval—but for testing 30s is fine.

```
void OnTimer()
{
   // periodic refresh of the last 5 days
   if(g_session_vis != NULL)
      g_session_vis.RefreshSessions(5);
}
```

OnDeinit() cleans up: we kill the timer and delete the visualizer object. Deleting the object invokes its destructor, which (as implemented) removes all session objects that the class created on the chart—ensuring the chart stays clean after the test. Explicit cleanup is essential in repeated-load environments (loading/unloading many EAs) to avoid chart bloat and lingering objects.

```
void OnDeinit(const int reason)
{
   EventKillTimer();

   if(g_session_vis != NULL)
   {
      // destructor will call DeleteAllSessionObjects()
      delete g_session_vis;
      g_session_vis = NULL;
   }
}
```

Initial Test Result:

Our initial testing phase was straightforward—after compiling the Session Test EA successfully, we simply attached it to a live MetaTrader 5 chart. The result was an immediate and clean visualization of the Forex sessions. In the image below, you can clearly see how the sessions overlap, each distinguished by its own color for easy identification. The filled session rectangles also convey sentiment: lime green indicates a bullish session close, while pink marks a bearish one. This color-coded visualization makes it effortless to interpret the dominant direction of each trading session at a glance.

![Initial testing](https://c.mql5.com/2/179/terminal64_VYfT4amZ5H.gif)

Fig. 2. Session Test

**Step 2—Integration Phase**

Every great trading tool goes through a phase of transformation—from an experimental script to a fully integrated production module. Our Session Visualizer is no exception. Originally, the test header (SessionVisualizerTest.mqh) focused on establishing the foundation: defining trading sessions, retrieving OHLC data, and drawing colored rectangles that represent each Forex market period. While it served its purpose well during testing, the production integration in our Market Periods Synchronizer EA required refinement for better visual clarity, extensibility, and compatibility.

Why We Moved Away from Filled Session Bodies

In the test version, sessions were represented with solid color-filled rectangles, using session colors to highlight each active trading block. This approach looked vibrant but quickly became visually congested—especially when overlapping sessions or historical layers were displayed together. The fill style also obscured underlying chart details, which defeated one of our goals: to provide context without clutter.

Hence, in the new version, we introduce unfilled body rectangles with dashed borders. These act like hollow candlesticks, marking each session’s open–close range without covering the price chart. The design choice makes the chart breathable and ensures that the trader can still see price action clearly beneath session markers.

As developers, this is a critical lesson: sometimes, less color equals more information. By removing the opaque fills, we gained visibility and professionalism in the presentation.

Integrating CCanvas and ARGB: One Visual Engine

A subtle yet important technical change in the new header is the use of #include <Canvas/Canvas.mqh> for ARGB-based visuals. This ensures compatibility with our EA’s existing CCanvas feature, which also manages background rendering and transparency for other modules. Rather than redefining ARGB separately, the new version delegates this to the Canvas library, creating consistency across all graphics used by the EA.

This decision is both architectural and practical. When building modular EAs that reuse components, duplicate ARGB macros can lead to conflicts or unpredictable behavior, especially if one component defines transparency differently. Using a single shared engine—CCanvas—helps maintain unified rendering logic and reduces maintenance overhead.

In simple terms, we declare the Canvas inclusion globally so that every visual component—whether session boxes, wicks, or overlays—speaks the same graphical “language.”

Redefining Sessions: From Color Blocks to Market Frames

The structure of the session representation remains similar: we still use a SessionInfo struct containing name, type, open/close times, and color. However, we’ve reimagined how these attributes are visualized.

Each session now renders:

- A dashed rectangle marking the open–close price range
- Optional grey-filled “wick” rectangles showing the session’s total high–low excursion
- Text labels and vertical lines marking open and close times
- Foreground/background layering so that rectangles stay behind price candles while lines and labels remain visible to the user

This creates an analytical but elegant look—like a transparent blueprint of market rhythm. Traders can see which sessions are expanding or contracting without losing chart detail. The new “unfilled” concept acts as a frame for the market rather than a cover.

Runtime Configuration: Making the Visualizer Dynamic

To give users and EAs more flexibility, we’ve turned previously static settings into runtime variables. For instance, m\_show\_wicks, m\_wick\_alpha, and m\_gmt\_offset are now globally configurable. A trader or developer can toggle wick visibility, change transparency, or adjust time offset without recompiling.

You’ll notice functions like

```
void SetShowWicks(bool show) { m_show_wicks = show; }
void SetWickAlpha(int alpha) { m_wick_alpha = MathMax(0, MathMin(255, alpha)); }
void SetGMTOffset(int offset){ m_gmt_offset = offset; }
```

These small but meaningful additions allow real-time interaction between the EA interface and visualization layer—a fundamental aspect of modern modular design in MQL5.

Contextual Insights: Displaying Current and Previous Sessions

Unlike the test version, which only displayed static historical sessions, the new header introduces current and recent session tracking. During live trading, the system identifies which session is active and overlays a “Live” label on it. It also keeps the recently closed session visible for one hour after closure—offering a quick visual reference of how the previous session performed.

Now that we understand the adjustments made toward effective integration, I will proceed to break down and explain the new SessionVisualizer header in detail. After that, I’ll discuss its integration into the Expert Advisor (EA) and highlight the new EA features designed to help traders make more efficient use of market session periods for improved trading decisions.

1) Module description and compilation strictness

We start with a clear description because published code is documentation too. Versioning tells readers what’s new (e.g. “unfilled rectangles, dashed borders, and grey wicks”), and #property strict enforces modern MQL5 type safety—catching silent bugs that often slip through in experimental headers.

```
//+------------------------------------------------------------------+
//| SessionVisualizer.mqh                                            |
//| Modular class for Forex session OHLC visualization               |
//| Copyright 2025: Clemence Benjamin                                |
//| Version: 1.02                                                    |
//+------------------------------------------------------------------+
#property strict
```

2) Shared dependencies (Object API + unified ARGB via Canvas)

In the test phase, it’s common to roll our own ARGB macro; in production, we align with the EA’s single visual engine to avoid conflicts. Including Canvas.mqh gives us the canonical ARGB() and seamless compatibility with other drawing features in the Market Periods Synchronizer EA.

```
#include <Object.mqh>
#include <Canvas/Canvas.mqh>   // for ARGB()
```

3) Session domain model (enum + struct)

We formalize the problem space: four major sessions, human-friendly names, and a simple SessionInfo container. Even if a property isn’t used everywhere (e.g., sess\_color no longer colors the body fill), we keep it—colors still drive labels/lines and maintain backward compatibility with prior tools and user habits.

```
//----------------------------------------------
// Session enum and info
//----------------------------------------------
enum SESSION_TYPE {
   SESSION_SYDNEY = 0,   // 22:00-07:00 GMT
   SESSION_TOKYO  = 1,   // 00:00-09:00 GMT
   SESSION_LONDON = 2,   // 08:00-17:00 GMT
   SESSION_NEWYORK= 3    // 13:00-22:00 GMT
};

struct SessionInfo {
   SESSION_TYPE type;
   string       name;       // "SYD","TOK","LON","NY"
   int          open_hour;  // GMT open hour
   int          close_hour; // GMT close hour
   color        sess_color; // base tint (not used for border now)
   bool         enabled;
};
```

4) Runtime parameters (EA-controlled “knobs”)

We declare global, EA-tunable variables so features are toggled without recompiling. The lookback governs history density, m\_show\_wicks/m\_wick\_alpha control visibility and transparency of wick ranges, and m\_gmt\_offset lets the EA harmonize server time vs GMT—the classic real-world pain point.

```
//----------------------------------------------
// Runtime params (configured by EA)
//----------------------------------------------
int    m_session_lookback = 10;
bool   m_show_wicks       = false;
int    m_wick_alpha       = 120;
int    m_gmt_offset       = 0;
```

5) Class shell and constructor defaults

We keep the public API compact and the private state explicit. In the constructor, we initialize canonical GMT schedules and readable three-letter tags. You’ll notice we store an m\_prefix: this is the secret to safe object naming and clean deletion later—no accidental collisions with other chart objects.

```
class CSessionVisualizer : public CObject
{
private:
   SessionInfo m_sessions[4];
   string      m_prefix;

public:
   CSessionVisualizer(string prefix="SESS_") : m_prefix(prefix)
   {
      // Defaults (GMT schedule)
      m_sessions[SESSION_SYDNEY] .type=SESSION_SYDNEY;  m_sessions[SESSION_SYDNEY] .name="SYD"; m_sessions[SESSION_SYDNEY] .open_hour=22; m_sessions[SESSION_SYDNEY] .close_hour=7;  m_sessions[SESSION_SYDNEY] .sess_color=clrAqua;  m_sessions[SESSION_SYDNEY] .enabled=true;
      m_sessions[SESSION_TOKYO]  .type=SESSION_TOKYO;   m_sessions[SESSION_TOKYO]  .name="TOK"; m_sessions[SESSION_TOKYO]  .open_hour=0;  m_sessions[SESSION_TOKYO]  .close_hour=9;  m_sessions[SESSION_TOKYO]  .sess_color=clrYellow; m_sessions[SESSION_TOKYO]  .enabled=true;
      m_sessions[SESSION_LONDON] .type=SESSION_LONDON;  m_sessions[SESSION_LONDON] .name="LON"; m_sessions[SESSION_LONDON] .open_hour=8;  m_sessions[SESSION_LONDON] .close_hour=17; m_sessions[SESSION_LONDON] .sess_color=clrRed;   m_sessions[SESSION_LONDON] .enabled=true;
      m_sessions[SESSION_NEWYORK].type=SESSION_NEWYORK; m_sessions[SESSION_NEWYORK].name="NY";  m_sessions[SESSION_NEWYORK].open_hour=13; m_sessions[SESSION_NEWYORK].close_hour=22; m_sessions[SESSION_NEWYORK].sess_color=clrBlue;  m_sessions[SESSION_NEWYORK].enabled=true;
   }

   ~CSessionVisualizer() { DeleteAllSessionObjects(); }
```

6) Public API: refresh + switches (designed for live charts)

RefreshSessions() is the one-button rebuild—clear, redraw history, then draw the live window. We expose tiny, expressive setters so an EA panel can flip sessions, recolor themes, or sync GMT on the fly. One notable change from the test header: no body fills; SetFillColors() is now a no-op kept purely for backward compatibility.

```
   //---------------------------------------------------------------
   // PUBLIC API
   //---------------------------------------------------------------
   void RefreshSessions(int lookback_days=10)
   {
      m_session_lookback = lookback_days;
      DeleteAllSessionObjects();

      const datetime now   = TimeCurrent();
      const datetime start = now - (m_session_lookback*86400);

      // Historical completed sessions
      for(int d=0; d<m_session_lookback; ++d)
      {
         datetime day_start = start + d*86400;
         for(int s=0; s<4; ++s)
         {
            if(!m_sessions[s].enabled) continue;
            DrawHistoricalSessionForDay(m_sessions[s], day_start);
         }
      }

      // Active session windows
      DrawCurrentSessions();

      ChartRedraw();
   }

   void SetSessionEnabled(SESSION_TYPE type, bool enabled)
   {
      for(int i=0;i<4;i++) if(m_sessions[i].type==type){ m_sessions[i].enabled=enabled; break; }
   }

   void SetSessionColor(SESSION_TYPE type, color col)
   {
      for(int i=0;i<4;i++) if(m_sessions[i].type==type){ m_sessions[i].sess_color=col; break; }
   }

   // Kept for compatibility with EA; ignored (we no longer fill bodies).
   void SetFillColors(color /*bull*/, color /*bear*/) { /* no-op */ }

   void SetShowWicks(bool show) { m_show_wicks = show; }
   void SetWickAlpha(int alpha) { m_wick_alpha = MathMax(0, MathMin(255, alpha)); }
   void SetGMTOffset(int offset){ m_gmt_offset = offset; }

   // public cleanup
   void ClearAll() { DeleteAllSessionObjects(); }
```

7) Historical drawing (only completed sessions)

A common footgun in session tools is drawing partial data into “history.” Here we compute open/close with m\_gmt\_offset, handle the “wrap past midnight” case, and then skip anything not fully in the past. Only after we confirm a complete window do we compute OHLC and render.

```
private:
   //---------------------------------------------------------------
   // CORE DRAWING
   //---------------------------------------------------------------
   void DrawHistoricalSessionForDay(const SessionInfo &sess, datetime day_start)
   {
      // build open/close datetimes adjusted by GMT offset
      MqlDateTime dto, dtc; TimeToStruct(day_start, dto); dtc = dto;
      dto.hour = sess.open_hour  + m_gmt_offset; dto.min=0; dto.sec=0;
      dtc.hour = sess.close_hour + m_gmt_offset; dtc.min=0; dtc.sec=0;

      datetime t_open  = StructToTime(dto);
      datetime t_close = StructToTime(dtc);
      if(sess.close_hour < sess.open_hour) t_close += 86400; // wraps midnight

      // Only draw if fully in the past
      if(t_close >= TimeCurrent()) return;
      if(t_close < TimeCurrent() - (m_session_lookback*86400)) return;

      double o,h,l,c;
      if(!GetOHLCInTimeRange(t_open, t_close, o,h,l,c)) return;

      DrawSessionVisuals(sess, t_open, t_close, o,h,l,c, /*is_current=*/false);
   }
```

8) Live session handling (and 1-hour grace for the last one)

We detect the active session in real time by building today’s window and adjusting for overnight sessions. If we’re inside, we stream OHLC up to now, nudge h/l with the current bid so the box reflects the live tape, and label it as “(Live)”. We also keep the just-closed session on screen for an extra hour—super handy during rollovers when traders want context.

```
   void DrawCurrentSessions()
   {
      const datetime now = TimeCurrent();
      MqlDateTime cur; TimeToStruct(now, cur);

      for(int i=0;i<4;i++)
      {
         if(!m_sessions[i].enabled) continue;
         const SessionInfo sess = m_sessions[i];

         // compute session window for "today" (adjusted for wrap)
         MqlDateTime ds, de; TimeToStruct(now, ds); de = ds;
         ds.hour = sess.open_hour  + m_gmt_offset; ds.min=0; ds.sec=0;
         de.hour = sess.close_hour + m_gmt_offset; de.min=0; de.sec=0;

         datetime session_start = StructToTime(ds);
         datetime session_end   = StructToTime(de);

         if(sess.close_hour < sess.open_hour)
         {
            if(cur.hour >= sess.open_hour + m_gmt_offset) session_end += 86400; // ends tomorrow
            else session_start -= 86400; // started yesterday
         }

         if(now >= session_start && now <= session_end)
         {
            // pull OHLC until now (close = current)
            double o,h,l,c;
            if(!GetOHLCInTimeRange(session_start, now, o,h,l,c))
               o = h = l = c = SymbolInfoDouble(_Symbol, SYMBOL_BID);

            const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            h = MathMax(h, bid);
            l = MathMin(l, bid);
            c = bid;

            DrawSessionVisuals(sess, session_start, session_end, o,h,l,c, /*is_current=*/true);
         }
         else if(now > session_end && now <= session_end + 3600)
         {
            // Keep just-closed session visible for 1 hour
            double o,h,l,c;
            if(!GetOHLCInTimeRange(session_start, session_end, o,h,l,c)) continue;
            DrawSessionVisuals(sess, session_start, session_end, o,h,l,c, /*is_current=*/false);
         }
      }
   }
```

9) Visual grammar: unfilled bodies, dashed borders, optional grey wick

This is the signature change from the test header. Bodies are unfilled and set behind the price (OBJPROP\_BACK=true) so nothing obscures candles. Borders encode sentiment (dark green for bullish, red for bearish; otherwise session color). Wicks, when enabled, are soft grey ARGB rectangles drawn in the background to reveal the full high–low excursion without clutter. Labels and timelines stay in the foreground for clarity.

```
   // Renders:
   //  - unfilled body rectangle (Open..Close) with dashed border
   //    * border = clrDarkGreen if bullish, clrRed if bearish (Neutral uses session color)
   //  - optional GREY filled wick rectangles in background for visibility
   //  - open/close vlines + labels in foreground (thin)
   void DrawSessionVisuals(const SessionInfo &sess,
                           datetime t_open, datetime t_close,
                           double o,double h,double l,double c,
                           bool is_current)
   {
      const bool bullish = (c > o);
      const bool bearish = (c < o);

      // ---------- vertical lines ----------
      string vopen = m_prefix + sess.name + "_O" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_open);
      if(ObjectFind(0, vopen) == -1) ObjectCreate(0, vopen, OBJ_VLINE, 0, t_open, 0);
      ObjectSetInteger(0, vopen, OBJPROP_COLOR, sess.sess_color);
      ObjectSetInteger(0, vopen, OBJPROP_WIDTH, is_current?2:1);
      ObjectSetInteger(0, vopen, OBJPROP_STYLE, is_current?STYLE_SOLID:STYLE_DASH);
      ObjectSetInteger(0, vopen, OBJPROP_BACK,  false);
      ObjectSetInteger(0, vopen, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, vopen, OBJPROP_HIDDEN, false);

      string vclose = m_prefix + sess.name + "_C" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_close);
      if(ObjectFind(0, vclose) == -1) ObjectCreate(0, vclose, OBJ_VLINE, 0, t_close, 0);
      ObjectSetInteger(0, vclose, OBJPROP_COLOR, sess.sess_color);
      ObjectSetInteger(0, vclose, OBJPROP_WIDTH, is_current?2:1);
      ObjectSetInteger(0, vclose, OBJPROP_STYLE, is_current?STYLE_SOLID:STYLE_DASH);
      ObjectSetInteger(0, vclose, OBJPROP_BACK,  false);
      ObjectSetInteger(0, vclose, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, vclose, OBJPROP_HIDDEN, false);

      // ---------- body rectangle (UNFILLED, dashed) ----------
      const double body_top = MathMax(o,c);
      const double body_bot = MathMin(o,c);

      color border_col = sess.sess_color;  // neutral fallback
      if(bullish) border_col = clrDarkGreen;
      else if(bearish) border_col = clrRed;

      string rect = m_prefix + sess.name + "_R" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_open);
      if(ObjectFind(0, rect) == -1)
      {
         if(!ObjectCreate(0, rect, OBJ_RECTANGLE, 0, t_open, body_bot, t_close, body_top))
            PrintFormat("Failed to create session rectangle %s err=%d", rect, GetLastError());
      }
      // unfilled rectangle with dashed edge, kept in background
      ObjectSetInteger(0, rect, OBJPROP_FILL,   false);
      ObjectSetInteger(0, rect, OBJPROP_COLOR,  border_col);
      ObjectSetInteger(0, rect, OBJPROP_STYLE,  STYLE_DASH);
      ObjectSetInteger(0, rect, OBJPROP_WIDTH,  1);
      ObjectSetInteger(0, rect, OBJPROP_BACK,   true);
      ObjectSetInteger(0, rect, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, rect, OBJPROP_HIDDEN, false);
      // keep coordinates fresh
      ObjectMove(0, rect, 0, t_open,  body_bot);
      ObjectMove(0, rect, 1, t_close, body_top);

      // ---------- WICKS (optional) as GREY filled rectangles in background ----------
      if(m_show_wicks)
      {
         uint wick_col = ARGB(m_wick_alpha, 128,128,128); // semi-transparent grey

         // Upper wick: [body_top .. high]
         string wu = m_prefix + sess.name + "_WU" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_open);
         if(h > body_top)
         {
            if(ObjectFind(0, wu) == -1)
            {
               if(!ObjectCreate(0, wu, OBJ_RECTANGLE, 0, t_open, body_top, t_close, h))
                  PrintFormat("Failed to create upper wick %s err=%d", wu, GetLastError());
            }
            ObjectSetInteger(0, wu, OBJPROP_BGCOLOR, wick_col);
            ObjectSetInteger(0, wu, OBJPROP_COLOR,  wick_col);
            ObjectSetInteger(0, wu, OBJPROP_FILL,   true);
            ObjectSetInteger(0, wu, OBJPROP_BACK,   true);
            ObjectSetInteger(0, wu, OBJPROP_SELECTABLE, false);
            ObjectSetInteger(0, wu, OBJPROP_HIDDEN, false);
            ObjectMove(0, wu, 0, t_open,  body_top);
            ObjectMove(0, wu, 1, t_close, h);
         }
         else ObjectDelete(0, wu);

         // Lower wick: [low .. body_bot]
         string wl = m_prefix + sess.name + "_WL" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_open);
         if(l < body_bot)
         {
            if(ObjectFind(0, wl) == -1)
            {
               if(!ObjectCreate(0, wl, OBJ_RECTANGLE, 0, t_open, l, t_close, body_bot))
                  PrintFormat("Failed to create lower wick %s err=%d", wl, GetLastError());
            }
            ObjectSetInteger(0, wl, OBJPROP_BGCOLOR, wick_col);
            ObjectSetInteger(0, wl, OBJPROP_COLOR,  wick_col);
            ObjectSetInteger(0, wl, OBJPROP_FILL,   true);
            ObjectSetInteger(0, wl, OBJPROP_BACK,   true);
            ObjectSetInteger(0, wl, OBJPROP_SELECTABLE, false);
            ObjectSetInteger(0, wl, OBJPROP_HIDDEN, false);
            ObjectMove(0, wl, 0, t_open,  l);
            ObjectMove(0, wl, 1, t_close, body_bot);
         }
         else ObjectDelete(0, wl);
      }
      else
      {
         // Ensure wicks are removed if disabled
         ObjectDelete(0, m_prefix + sess.name + "_WU" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_open));
         ObjectDelete(0, m_prefix + sess.name + "_WL" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_open));
      }

      // ---------- Labels in foreground ----------
      string session_tag = sess.name + (is_current ? " (Live)" : "");
      string lbl_o = m_prefix + sess.name + "_OL" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_open);
      if(ObjectFind(0, lbl_o) == -1) ObjectCreate(0, lbl_o, OBJ_TEXT, 0, t_open, o);
      ObjectSetString (0, lbl_o, OBJPROP_TEXT, session_tag + " Open");
      ObjectSetInteger(0, lbl_o, OBJPROP_COLOR, sess.sess_color);
      ObjectSetInteger(0, lbl_o, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, lbl_o, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, lbl_o, OBJPROP_HIDDEN, false);
      ObjectSetInteger(0, lbl_o, OBJPROP_BACK, false);

      string lbl_c = m_prefix + sess.name + "_CL" + (is_current?"_CUR_":"_HIS_") + IntegerToString((int)t_close);
      if(ObjectFind(0, lbl_c) == -1) ObjectCreate(0, lbl_c, OBJ_TEXT, 0, t_close, c);
      ObjectSetString (0, lbl_c, OBJPROP_TEXT, session_tag + " Close");
      ObjectSetInteger(0, lbl_c, OBJPROP_COLOR, sess.sess_color);
      ObjectSetInteger(0, lbl_c, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, lbl_c, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, lbl_c, OBJPROP_HIDDEN, false);
      ObjectSetInteger(0, lbl_c, OBJPROP_BACK, false);
   }
```

10) Robust OHLC aggregation (bar-shift based)

For correctness across timeframes, we compute session OHLC using iBarShift boundaries and fetch a compact slice of arrays—then aggregate with ArrayMaximum/Minimum. This avoids off-by-one errors and works even if the timeframe granularity doesn’t align perfectly with session edges.

```
   //---------------------------------------------------------------
   // Data helpers
   //---------------------------------------------------------------
   bool GetOHLCInTimeRange(datetime start, datetime end,
                           double &open, double &high, double &low, double &close)
   {
      int shift_start = iBarShift(_Symbol, _Period, start, false);
      int shift_end   = iBarShift(_Symbol, _Period, end,   false);
      if(shift_start < 0 || shift_end < 0) return false;

      int bars = shift_start - shift_end + 1;
      if(bars <= 0) return false;

      double opens[], highs[], lows[], closes[];
      ArraySetAsSeries(opens,  true);
      ArraySetAsSeries(highs,  true);
      ArraySetAsSeries(lows,   true);
      ArraySetAsSeries(closes, true);

      if(CopyOpen (_Symbol,_Period, shift_end, bars, opens)  != bars) return false;
      if(CopyHigh (_Symbol,_Period, shift_end, bars, highs)  != bars) return false;
      if(CopyLow  (_Symbol,_Period, shift_end, bars, lows)   != bars) return false;
      if(CopyClose(_Symbol,_Period, shift_end, bars, closes) != bars) return false;

      open  = opens[bars-1];
      high  = highs[ArrayMaximum(highs,0,bars)];
      low   = lows [ArrayMinimum(lows ,0,bars)];
      close = closes[0];
      return true;
   }
```

11) Deterministic cleanup (prefix-scoped GC)

Professional chart tools must leave the stage clean. On refresh or EA removal, we iterate objects in reverse and delete only those bearing our m\_prefix. This is polite coexistence with other indicators/EAs and prevents “ghost drawings” that frustrate users.

```
   //---------------------------------------------------------------
   // Cleanup
   //---------------------------------------------------------------
   void DeleteAllSessionObjects()
   {
      int total = ObjectsTotal(0);
      for(int i = total-1; i >= 0; --i)
      {
         string name = ObjectName(0, i);
         if(StringFind(name, m_prefix) == 0) ObjectDelete(0, name);
      }
   }
};
```

**Integrating CSessionVisualizer and the Session Info into MarketPeriodsSynchronizerEA**

Before this upgrade, our previous EA version focused on major/minor period structure, body fills, wicks, and UI controls—no notion of market sessions or session summaries. In this version, we graft in the new CSessionVisualizer class and add a compact “Session Information” panel, plus tidy control toggles so sessions and their wick highlights can be driven independently of the major-period visuals. Below I’ll walk you through the integration as if we were evolving the earlier EA step-by-step, calling out the why behind each change and pasting the exact snippet that implements it. (For reference, the former EA baseline is v1.01 without sessions and without the info panel.

1) Bring the Session Visualizer into the EA

We include the header for our new session renderer alongside the Canvas helper. The test header already uses CSessionVisualizer with unfilled, dashed body rectangles and optional grey wick fills in the background, so visually it won’t fight with our HTF overlays. We keep includes minimal and explicit to avoid ODR/name collisions.

```
#include <Canvas/Canvas.mqh>   // Canvas helper library (expects Canvas.mqh to be present)
#include <SessionVisualizer.mqh>  // Integrated Session Visualizer class
```

2) New inputs for sessions and the info panel

Inputs are the contract with users. We add dedicated toggles for showing sessions, their lookback window, the broker GMT offset (so sessions line up), wick visibility/alpha just for sessions, and the flag for the “Session Information” panel. Notice we deliberately separate wick settings for majors vs. sessions—traders often want wicks for one but not the other.

```
// Session inputs
input bool           InpShowSessions      = true;     // Show Forex sessions
input int            InpSessionLookback   = 10;       // Session lookback (days)
input int            InpGMTOffset         = 0;        // GMT offset (broker hours)

// Wick visualization (separated for majors and sessions)
input bool           InpShowSessionWicks  = false;    // Show wick regions for sessions
input int            InpSessionWickAlpha  = 120;      // Alpha transparency for session wicks 0..255

// Session information panel
input bool           InpShowSessionInfo   = true;     // Show session info panel
```

Earlier versions had no session-specific inputs; everything revolved around majors/minors and one wick toggle. Now we decouple those concerns so the layout stays clean even when both features are active.

3) Runtime globals mirror the new inputs

The EA’s dashboard modifies state at runtime, so we keep mutable copies of inputs. Here we add g\_ShowSessions, session-wick runtime (g\_ShowSessionWicks, g\_SessionWickAlpha), and a flag for the info panel. Keeping these near the other globals preserves the mental model: inputs → runtime → UI.

```
// Wick runtime (separated for majors and sessions)
bool            g_ShowWicks;
int             g_WickAlpha;
bool            g_ShowSessionWicks;
int             g_SessionWickAlpha;

// Session runtime
bool            g_ShowSessions;

// Session information panel
bool            g_ShowSessionInfo;
```

4) A single, shared CSessionVisualizer instance

All session drawings should share a stable naming prefix and lifecycle. We instantiate one visualizer with a short prefix so object names remain unique and easy to sweep/clean up without touching other chart objects.

```
// Session visualizer instance
CSessionVisualizer g_sessions("SESS_");
```

5) Session info—compact state we can refresh on tick

Traders love at-a-glance context: “Which session am I in? How did the previous one close?” We store the minimal data we need to draw the panel: names, O/C, nature (bull/bear/neutral), and the last update time to avoid redundant work.

```
// Session data storage
string   g_CurrentSessionName = "";
double   g_CurrentSessionOpen = 0;
double   g_CurrentSessionCurrent = 0;
string   g_CurrentSessionNature = "";
datetime g_CurrentSessionStart = 0;

string   g_PrevSessionName = "";
double   g_PrevSessionOpen = 0;
double   g_PrevSessionClose = 0;
string   g_PrevSessionNature = "";
datetime g_LastSessionUpdate = 0;
```

6) Wider canvas background to fit the new panel

The session info block adds three concise lines for current and previous session summaries. We nudge panel width/height so the layout breathes and text doesn’t wrap awkwardly.

```
// Canvas background - INCREASED WIDTH
CCanvas g_bgCanvas;
string  g_bg_name = "";
int     g_bg_x = 6;
int     g_bg_y = 44;  // uses Y_OFFSET in OnInit
int     g_bg_w = 450; // increased from 430
int     g_bg_h = 340; // increased from 300 for session info
```

7) Initialize the new knobs in OnInit()

OnInit() is where inputs become live state. We also prime the visualizer so it knows which side-channel settings apply to sessions (not majors): GMT offset, wick toggle, and wick alpha. Even though the session class exposes SetFillColors(), our current session visuals use unfilled rectangles, but we keep the call for forward compatibility.

```
// Initialize session visualizer with separate session wick params
g_sessions.SetGMTOffset(InpGMTOffset);
g_sessions.SetFillColors(g_FillBull, g_FillBear);   // kept for compatibility (no body fill in visualizer)
g_sessions.SetShowWicks(g_ShowSessionWicks);        // session-specific wick setting
g_sessions.SetWickAlpha(g_SessionWickAlpha);        // session-specific wick alpha

// copy panel flag
g_ShowSessionInfo = InpShowSessionInfo;
```

8) Add the “Session Information” UI

We draw tiny labels inside the control panel to show the current session (name, open, nature) and the previous session (name, O/C, nature). Colors gently reinforce bull/bear. The panel is created once; content is updated every tick.

```
void UpdateSessionInfo()
{
   if(!g_ShowSessionInfo) return;

   datetime now = TimeCurrent();

   // Always refresh current session nature
   UpdateCurrentSessionInfo(now);

   // Update previous session only at boundaries
   if(g_LastSessionUpdate == 0 || IsNewSessionStart(now))
   {
      UpdatePreviousSessionInfo(now);
      g_LastSessionUpdate = now;
   }

   UpdateSessionDisplay();
}
```

9) Smart session bookkeeping (current + previous)

We separate concerns into three helpers:

1. UpdateCurrentSessionInfo: find which session we’re in (GMT-based), cache the session open price, and classify nature using live bid vs. session open.
2. UpdatePreviousSessionInfo: when a new session starts, snapshot the previous session’s O/C and nature.
3. UpdateSessionDisplay: push latest strings/colors into label objects.

This is intentionally lightweight—no heavy loops—so it plays nicely with the EA’s regular HTF refresh cycle. By contrast, the earlier EA had no such concept, which is why the panel and boundary logic are entirely new in this version.

```
void UpdateSessionInfo()
{
   if(!g_ShowSessionInfo) return;

   datetime now = TimeCurrent();

   // Always refresh current session nature
   UpdateCurrentSessionInfo(now);

   // Update previous session only at boundaries
   if(g_LastSessionUpdate == 0 || IsNewSessionStart(now))
   {
      UpdatePreviousSessionInfo(now);
      g_LastSessionUpdate = now;
   }

   UpdateSessionDisplay();
}
```

10) Session controls in the dashboard

Power should be at the trader’s fingertips. We add two new buttons: one to toggle Sessions on/off, another to toggle Session Wicks. We keep majors’ controls independent. Buttons live alongside existing toggles to preserve muscle memory.

```
// NEW ROW: Session wicks, Sessions, Major VLines (swapped positions for clarity)
CreateButton(btn_toggle_session_wicks, 12,  86 + Y_OFFSET, 130, 22, g_ShowSessionWicks ? "Sess Wicks: ON" : "Sess Wicks: OFF");
CreateButton(btn_toggle_sessions,      152, 86 + Y_OFFSET, 130, 22, g_ShowSessions ? "Sessions: ON" : "Sessions: OFF");
CreateButton(btn_toggle_maj_vlines,    292, 86 + Y_OFFSET, 130, 22, g_ShowMajorVLines ? "Maj VLines: ON" : "Maj VLines: OFF");
```

And the click handlers simply flip the bit and refresh:

```
if(obj == btn_toggle_session_wicks)
{
   g_ShowSessionWicks = !g_ShowSessionWicks;
   ObjectSetString(main_chart_id, btn_toggle_session_wicks, OBJPROP_TEXT,
                   g_ShowSessionWicks ? "Sess Wicks: ON" : "Sess Wicks: OFF");
   RefreshLines();
   return;
}

if(obj == btn_toggle_sessions)
{
   g_ShowSessions = !g_ShowSessions;
   ObjectSetString(main_chart_id, btn_toggle_sessions, OBJPROP_TEXT,
                   g_ShowSessions ? "Sessions: ON" : "Sessions: OFF");
   RefreshLines();
   return;
}
```

11) Draw sessions inside RefreshLines() (one place to rule them all)

The EA’s central redraw is RefreshLines(). After we manage HTF objects, we either ask the visualizer to render sessions (with the latest per-session settings) or clear them. Keeping this here ensures a single refresh cadence and one cleanup pass.

```
// Draw sessions if toggled - NOW WITH SEPARATE WICK SETTINGS
if(g_ShowSessions)
{
   g_sessions.SetFillColors(g_FillBull, g_FillBear);     // no-op for bodies today; future-proof
   g_sessions.SetShowWicks(g_ShowSessionWicks);
   g_sessions.SetWickAlpha(g_SessionWickAlpha);
   g_sessions.RefreshSessions(InpSessionLookback);
}
else
{
   g_sessions.ClearAll();
}
```

In the older EA, RefreshLines() only handled HTF/minor visuals. With sessions centralized here, a single timer/tick drives all visuals, keeping flicker and race conditions at bay.

12) Update and cleanup on tick & deinit

On every tick we recompute high-level state: new bars, and now the session info block. During deinit we also clear session drawings so a re-attach starts fresh.

```
void OnTick()
{
   bool need_refresh = false;
   // ... (bar-change checks for majors/minors)

   // Always update session info on tick for real-time current session nature
   UpdateSessionInfo();

   if(need_refresh) RefreshLines();
}

void DeleteAllHTFLines()
{
   // remove HTF objects
   // ...
   // Also clear sessions
   g_sessions.ClearAll();
}
```

13) Taller panel when restored, to fit the new section

When we minimize/restore, the background height should match the content. We bump the restored height to keep the session info fully visible without overlapping other chart elements.

```
void RestoreUI()
{
   UpdateBackgroundHeight(340);  // was 250; space for the Session Info
   CreateAllOtherUIObjects();
   ObjectSetString(main_chart_id, btn_minimize, OBJPROP_TEXT, "Minimize");
}
```

### Testing

For this test, we attach the EA directly to a live chart (not the Strategy Tester) since our focus is purely on the on-chart visuals. This lets us verify that the rendering and controls align with our goals. Below is an image that showcases the summarized feature in action.

![MPS featuring SessionVisualizer](https://c.mql5.com/2/179/terminal64_6vp9ChnZgz.gif)

Fig. 3. Final result featuring Forex Sessions and information display

### Conclusion

Forex and stock market sessions can provide in-depth insights for traders and financial analysts when properly understood and visualized. The idea implemented here helps us represent trading sessions as candlestick periods, revealing valuable details about each session’s high, low, open, and close. This approach allows us to apply our traditional candlestick knowledge to session-level analysis—helping predict potential movements in the next session based on the nature of the previous ones.

Through this development, we introduced visual markers and a dynamic control utility (Market Periods Synchronizer) that let us toggle, adjust, and synchronize session displays directly on the chart. Additionally, the Session Information Panel summarizes the open, close, and directional nature (bullish or bearish) of both the current and previous sessions—offering traders a quick, intuitive view of the underlying market forces.

Because a session’s bullish or bearish character is measured across extended time periods, it often reflects broader market bias and trend tendencies. Understanding these behaviors not only sharpens technical perspective but also strengthens trading psychology through direct, visual observation of market dynamics.

You’re welcome to experiment, modify, and share your insights in the comments below. The supporting source files are attached for your reference and continued exploration.

### Key Lessons

| Key Lesson: | Description: |
| --- | --- |
| Class Integration and Modular Design | Building reusable modules like CSessionVisualizer promotes cleaner architecture and easier maintenance. By separating visualization logic into a class, the EA gains flexibility—developers can modify session visuals or reuse the component in other projects without rewriting core code. |
| Runtime Variables and UI Synchronization | Mirroring input parameters into runtime globals ensures that real-time UI controls (like sliders, buttons, and toggles) can dynamically update visual behavior without reinitializing the EA. This approach teaches how to design responsive chart utilities using object-based UIs. |
| Session Management and Data Mapping | The use of structured session data (e.g., GMT-based start/end hours, open/close calculations) demonstrates how to bridge live chart data with time-based trading logic. It reinforces how developers can blend analytics and visuals to make market context more intuitive. |

### Attachments

| Source File Name | Version | Description |
| --- | --- | --- |
| SessionVisualizerTest.mqh | 1.0 | A standalone testing script designed to verify the rendering and behavior of the CSessionVisualizer class. It allows developers to isolate the visual logic, ensuring session rectangles, colors, and time mappings display correctly before integration into larger systems. |
| SessionTest.mq5 | 1.0 | A simplified Expert Advisor designed specifically to test and demonstrate the functionality of the SessionVisualizerTest header. |
| SessionVisualizer.mqh | 1.0 | The main class implementation that manages Forex session visualization. It handles time-zone adjustments, color mapping, and graphical rendering of session boundaries on the chart—serving as a reusable visual module for any EA or indicator that needs session-based context. |
| MarketPeriodsSychronizer\_EA.mq5 | 1.02 | The upgraded Expert Advisor integrating the SessionVisualizer and a new Session Information panel. It synchronizes multiple timeframe markers, session fills, and real-time analytics into a unified control dashboard—providing traders with an interactive and educational market-period visualization tool. |

[Back to contents](https://www.mql5.com/en/articles/20005#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20005.zip "Download all attachments in the single ZIP archive")

[SessionVisualizerTest.mqh](https://www.mql5.com/en/articles/download/20005/SessionVisualizerTest.mqh "Download SessionVisualizerTest.mqh")(10.61 KB)

[SessionTest.mq5](https://www.mql5.com/en/articles/download/20005/SessionTest.mq5 "Download SessionTest.mq5")(0.65 KB)

[SessionVisualizer.mqh](https://www.mql5.com/en/articles/download/20005/SessionVisualizer.mqh "Download SessionVisualizer.mqh")(31.1 KB)

[MarketPeriodsSynchronizer\_EA.mq5](https://www.mql5.com/en/articles/download/20005/MarketPeriodsSynchronizer_EA.mq5 "Download MarketPeriodsSynchronizer_EA.mq5")(126.83 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/499682)**
(1)


![stu48](https://c.mql5.com/avatar/avatar_na2.png)

**[stu48](https://www.mql5.com/en/users/stu48)**
\|
16 Nov 2025 at 07:39

Hi Clemence, Great content and great reading, great stuff thanks


![Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://c.mql5.com/2/180/19989-formulating-dynamic-multi-pair-logo.png)[Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)

This part explores how to design a Dynamic Multi-Pair Expert Advisor capable of adapting between Scalping and Swing Trading modes. It covers the structural and algorithmic differences in signal generation, trade execution, and risk management, allowing the EA to intelligently switch strategies based on market behavior and user input.

![Bivariate Copulae in MQL5 (Part 2): Implementing Archimedean copulae in MQL5](https://c.mql5.com/2/179/19931-bivariate-copulae-in-mql5-part-logo__1.png)[Bivariate Copulae in MQL5 (Part 2): Implementing Archimedean copulae in MQL5](https://www.mql5.com/en/articles/19931)

In the second installment of the series, we discuss the properties of bivariate Archimedean copulae and their implementation in MQL5. We also explore applying copulae to the development of a simple pairs trading strategy.

![Reimagining Classic Strategies (Part 18): Searching For Candlestick Patterns](https://c.mql5.com/2/180/20223-reimagining-classic-strategies-logo.png)[Reimagining Classic Strategies (Part 18): Searching For Candlestick Patterns](https://www.mql5.com/en/articles/20223)

This article helps new community members search for and discover their own candlestick patterns. Describing these patterns can be daunting, as it requires manually searching and creatively identifying improvements. Here, we introduce the engulfing candlestick pattern and show how it can be enhanced for more profitable trading applications.

![Risk-Based Trade Placement EA with On-Chart UI (Part 1): Designing the User Interface](https://c.mql5.com/2/179/19932-risk-based-trade-placement-logo.png)[Risk-Based Trade Placement EA with On-Chart UI (Part 1): Designing the User Interface](https://www.mql5.com/en/articles/19932)

Learn how to build a clean and professional on-chart control panel in MQL5 for a Risk-Based Trade Placement Expert Advisor. This step-by-step guide explains how to design a functional GUI that allows traders to input trade parameters, calculate lot size, and prepare for automated order placement.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/20005&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083082293242762403)

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