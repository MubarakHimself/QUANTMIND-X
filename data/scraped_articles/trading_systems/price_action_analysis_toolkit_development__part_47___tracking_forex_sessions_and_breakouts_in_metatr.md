---
title: Price Action Analysis Toolkit Development (Part 47): Tracking Forex Sessions and Breakouts in MetaTrader 5
url: https://www.mql5.com/en/articles/19944
categories: Trading Systems, Indicators
relevance_score: 7
scraped_at: 2026-01-22T17:48:26.352190
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=gezwcrbuhtfrkhpdqkxemlgqoljdunrv&ssn=1769093304817647386&ssn_dr=0&ssn_sr=0&fv_date=1769093304&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19944&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Price%20Action%20Analysis%20Toolkit%20Development%20(Part%2047)%3A%20Tracking%20Forex%20Sessions%20and%20Breakouts%20in%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690933048176968&fz_uniq=5049393256967547575&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

The foreign‑exchange market operates 24 hours a day, constantly cycling through major financial centers around the globe. Each region brings its own characteristics: the Asia session tends to start the day quietly, Tokyo often provides the first true direction, London injects strong volume and volatility, and New York carries momentum into the late hours with frequent reversals or continuations. Recognizing which of these sessions is currently active allows traders to adapt to changing market speed, volatility, and liquidity.

For new traders, keeping track of these sessions can be confusing. Broker server times often differ from local time zones, and manually calculating session boundaries can lead to mistakes. The All Sessions EA solves this by automatically synchronizing to the broker’s server time, displaying accurate session boxes for Asia, Tokyo, London, and New York directly on the chart. This gives beginners a clear visual understanding of how global markets hand over control through the day and how price behavior changes as one session transitions to the next.

Beyond simple visualization, the EA includes interactive features such as on‑chart toggle buttons to show or hide sessions, an information panel, and a scrolling headline ticker that reports real‑time events. It also integrates a breakout‑alert system that notifies the user whenever current prices exceed the high or low of a previous session—helping traders anticipate shifts in volatility and trade with greater timing accuracy. With full synchronization, improved readability, and an intuitive layout, the enhanced version also now highlights all four global sessions and uniformly spaced H‑L‑O‑C labels for clear, consistent reference.

Read on in this article to learn how to implement these features step by step in MQL5, from designing the interface to building the logic for real‑time monitoring and alerts.

### Contents

- [Introduction](https://www.mql5.com/en/articles/19944#para1)
- [Designing the Tool](https://www.mql5.com/en/articles/19944#para3)
- [Building the Interface](https://www.mql5.com/en/articles/19944#para4)
- [Implementing the Logic](https://www.mql5.com/en/articles/19944#para5)
- [Testing and Validation](https://www.mql5.com/en/articles/19944#para6)
- [Conclusion](https://www.mql5.com/en/articles/19944#para7)

### Designing the Tool

Before constructing the Expert Advisor, it is essential to understand what trading sessions represent and why they are central to market analysis. The 24‑hour Forex market rotates through four main hubs—Asia, Tokyo, London, and New York—each with distinct patterns of liquidity and volatility. Historically, the Asian market tends to be calmer, Tokyo marks the first wave of direction, London drives strong price movements, and New York often finishes the day with momentum or reversals. Because broker servers operate in different time zones, manually identifying these periods on a chart can lead to errors, especially for new traders who may mistake local time for market time. The All Sessions EA removes that complexity by using the broker’s server clock as the reference, ensuring all four sessions appear correctly aligned regardless of time zone differences. It equips traders with a practical, real‑time map of global session activity.

Purpose of the EA

This EA is designed to make session analysis visual, interactive, and actionable.

1. Visualize Market Cycles: Automatically draw shaded rectangles for Asia, Tokyo, London, and New York sessions using accurate start and end times.
2. Simplify Interaction: Allow users to toggle each session on or off through on‑chart buttons without navigating indicator settings.
3. Monitor Key Data Live: Display each session’s open, high, low, and close directly on the chart using clean, evenly spaced text labels.
4. Stay Synchronized Automatically: Operate entirely on broker‑server time, never on the user’s computer clock.
5. Highlight Trading Signals: Generate instant alerts when the current price breaks above or below the previous session’s range.
6. Organize Information: Present quick visual feedback through a top‑right info panel and a scrolling headline ticker that reports events in real time.

Interface and Functional Design

Just like a dashboard, every on‑chart element serves a clear purpose:

| Element | Type | Purpose |
| --- | --- | --- |
| Asia Button | OBJ\_BUTTON | Toggle the visibility of the Asia session rectangle |
| Tokyo Button | OBJ\_BUTTON | Toggle the visibility of the Tokyo session rectangle |
| London Button | OBJ\_BUTTON | Toggle the visibility of the London session rectangle |
| New York Button | OBJ\_BUTTON | Toggle the visibility of the New York session rectangle |
| Information Panel | OBJ\_RECTANGLE\_LABEL + OBJ\_LABEL | Show which sessions are currently active and their states |
| Ticker Headline | OBJ\_LABEL | Scroll trading updates and alerts across the bottom of the chart |
| Session Rectangles | OBJ\_RECTANGLE | Color‑coded areas representing each session’s trading hours |
| Session Labels | OBJ\_TEXT | Display the H / L / O / C values once per session (current day only) |

Each control is created programmatically with _ObjectCreate_() and updated dynamically as traders interact or as time passes. The interface remains minimal and responsive even during high‑volatility periods.

Visual Design

Clarity and spacing are fundamental. This version increases the vertical offset between session labels to maintain equal gaps—so the text marking London is separated from Tokyo by the same visual distance as Tokyo is from Asia. This uniform spacing prevents overlap and ensures readability regardless of chart scale. A consistent color scheme further differentiates activity zones:

- Sky Blue – Asia
- Light Green – Tokyo
- Light Pink – London
- Gold – New York

The scrolling ticker appears at the bottom of the chart and uses the _TickerColor_ input to blend with the trader’s preferred chart palette. A compact black information panel on the upper‑right corner displays current session states using white Arial text for sharp contrast.

Functional Requirements

The logic is modular and event‑driven:

| Logic | Description |
| --- | --- |
| Button Interaction | User clicks trigger _OnChartEvent_(), immediately redrawing sessions and updating the panel. |
| Session Computation | The EA calculates session start and end times from broker time, scans historical bars for H/L/O/C values, and plots rectangles accordingly. |
| Timer Events | Every second, the EA refreshes visual elements, scrolls the ticker, and checks for session openings, closings, and breakout conditions. |
| Breakout Detection | When the current price eclipses the previous session’s extreme, the EA issues an on‑screen and sound alert and posts the event in the ticker headline. |
| Resource Efficiency | All graphics rely on lightweight OBJ\_RECTANGLE and OBJ\_TEXT objects—no indicator buffers—keeping CPU usage low. |

Core Logic Concept

The system follows a simple cyclical logic anchored to broker‑time:

| Stages | Description |
| --- | --- |
| Initialization stage | Creates buttons, panel, and ticker; draws both the previous and current trading days. |
| Monitoring stage | A one‑second timer continuously checks for session open/close times relative to _TimeCurrent_() and manages alerts. |
| Visualization update | Every minute, sessions are redrawn to stay fully synchronized with server time. |
| Interaction response | Any button press immediately updates visibility and status text. |

This design keeps the code modular, readable, and extendable—so that later we can add features such as the Sydney session, push notifications, or statistical averages.

### Building the Interface

A functional interface is the backbone of any interactive EA. Rather than relying on input parameters buried in the settings window, the All Sessions EA places all critical controls directly on the chart. Traders can turn sessions on or off, view information instantly, and follow streaming updates without interrupting live analysis. In MetaTrader 5, graphical components are created with chart objects such as OBJ\_BUTTON, OBJ\_LABEL, and OBJ\_RECTANGLE\_LABEL. Each element is defined by position, size, color, and other properties that keep the interface consistent across chart styles.

Layout Concept

The chart layout follows a practical visual hierarchy:

| Area | Elements | Description |
| --- | --- | --- |
| Top Left | Asia · Tokyo · London · New York toggle buttons | Primary user controls. Each acts as an independent switch for its session box. |
| Top Right | Information Panel | Black rectangle showing current on/off status for each session. |
| Main Chart Area | Colored rectangles + text labels | Session time ranges and H/L/O/C data, evenly spaced to prevent overlap. |
| Bottom‑left | Scrolling ticker headline | Live feed displaying alerts and updates. |

This positioning keeps the price candles central while the interface elements occupy unused margins, ensuring clarity even on smaller screens.

![](https://c.mql5.com/2/176/sessions.png)

Creating the Session Buttons

Each button is created through a helper function:

```
void CreateButton(string name,string text,int x,int y,color c)
{
   ObjectCreate(0,name,OBJ_BUTTON,0,0,0);
   ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,name,OBJPROP_XSIZE,110);
   ObjectSetInteger(0,name,OBJPROP_YSIZE,20);
   ObjectSetInteger(0,name,OBJPROP_BGCOLOR,clrDimGray);
   ObjectSetInteger(0,name,OBJPROP_COLOR,c);
   ObjectSetInteger(0,name,OBJPROP_FONTSIZE,9);
   ObjectSetString (0,name,OBJPROP_TEXT,text);
}
```

Through this routine, four buttons are placed neatly in two rows:

```
CreateButton("BTN_ASIA","Asia ON/OFF",10,20,clrSkyBlue);
CreateButton("BTN_TOKYO","Tokyo ON/OFF",125,20,clrLightGreen);
CreateButton("BTN_LONDON","London ON/OFF",240,20,clrLightPink);
CreateButton("BTN_NEWYORK","New York ON/OFF",10,45,clrGold);
```

Each button triggers an event captured by _OnChartEvent_(). When clicked, it flips the corresponding Boolean ( _showAsia_, _showTokyo_, etc.), redraws visible sessions, updates the panel, and posts a status message in the ticker headline. This gives traders immediate control without stopping live updates.

Building the Information Panel

The right‑hand information box acts as a mini‑dashboard. It’s composed of a background rectangle ( _OBJ\_RECTANGLE\_LABEL_) and a text label ( _OBJ\_LABEL_) that displays the visibility state of each session:

```
ASIA    : ON
TOKYO   : OFF
LONDON  : ON
NEWYORK : OFF
```

Programmatically it’s built as follows:

```
ObjectCreate(0,"PANEL_BG",OBJ_RECTANGLE_LABEL,0,0,0);
ObjectSetInteger(0,"PANEL_BG",OBJPROP_CORNER,CORNER_RIGHT_UPPER);
ObjectSetInteger(0,"PANEL_BG",OBJPROP_XDISTANCE,360);
ObjectSetInteger(0,"PANEL_BG",OBJPROP_YDISTANCE,20);
ObjectSetInteger(0,"PANEL_BG",OBJPROP_XSIZE,360);
ObjectSetInteger(0,"PANEL_BG",OBJPROP_YSIZE,110);
ObjectSetInteger(0,"PANEL_BG",OBJPROP_BGCOLOR,clrBlack);
```

The update routine _UpdatePanel_() refreshes its content each time a button changes state or a redraw occurs. Keeping this information on screen helps beginners understand which sessions they are currently viewing.

Creating the Ticker Headline

The ticker runs along the bottom left corner of the chart, displaying alerts such as “London session opened” or “New York breaks above prior London high.” It’s implemented with an _OBJ\_LABEL_ object and scrolled by changing its text every second inside _OnTimer_():

```
ObjectCreate(0,"TICKER_OBJ",OBJ_LABEL,0,0,0);
ObjectSetInteger(0,"TICKER_OBJ",OBJPROP_CORNER,CORNER_LEFT_LOWER);
ObjectSetInteger(0,"TICKER_OBJ",OBJPROP_XDISTANCE,10);
ObjectSetInteger(0,"TICKER_OBJ",OBJPROP_YDISTANCE,18);
ObjectSetInteger(0,"TICKER_OBJ",OBJPROP_COLOR,TickerColor);
```

A one‑second timer moves text characters across the label, creating a smooth scrolling effect that mimics professional news feeds.

Managing Readability and Spacing

When multiple sessions are displayed together, label overlap can occur near the top of the rectangles. To eliminate clutter, this EA introduces a dynamic vertical offset for each label with the formula:

```
double offsetY = (slot + 1) * _Point * 120;
```

This equal‑spacing multiplier keeps gaps between Asia ↔ Tokyo and Tokyo ↔ London consistent, ensuring identical visual separation across all sessions. It automatically scales with symbol precision (\_Point) so labels maintain relative spacing on any instrument.

Color Palette and Font Choices

Each session is assigned a distinct tone that allows traders to identify market activity instantly without consulting legends or tooltips. The Asia session appears in Sky Blue, representing the calm, early part of the trading day when price movements are generally measured and stable. The Tokyo session uses Light Green, capturing the sense of renewal as volatility begins to build when Japanese markets open. For London, the color Light Pink has been chosen to contrast strongly with the preceding shades and to mark the period of highest trading intensity as Europe steps in. Finally, the New York session is drawn in Gold, symbolizing late‑day momentum and the transition toward daily closes across global markets.

Supporting text elements follow a consistent visual language: all session labels display their own session color, while informational text on the panel and ticker uses a clean white sans‑serif font for readability against dark backgrounds. Font weight and size are deliberately moderate so the interface remains visible yet unobtrusive, even during rapid market movement or when the chart background theme is changed. This combination of clear color coding and careful typography ensures that the EA maintains both clarity and aesthetic balance on any chart.

Text uses a sans‑serif font (Arial or system default) in white for panels and in session color for labels. Fonts are bold enough to remain legible against both light and dark chart backgrounds.

Putting the Interface Together

At initialization ( _OnInit_()), the EA calls:

```
CreateButton(... four times …);
CreatePanel();
CreateTicker();
DrawAll();
UpdatePanel();
UpdateTicker(Headline);
```

This sequence constructs the complete interface as soon as the EA loads. From there, all updates are handled dynamically by the timer and event handlers, so the trader never needs to refresh manually.

### Implementing the Logic

Once the interface is in place, the EA needs the logic that will make it think, react, and remain synchronized with the real market. The goal is to let traders watch, in real time, how the global trading day unfolds—from Asia to Tokyo, through London, and finally to New York — while the EA automatically draws each session, reports activity, and alerts on breakout opportunities. This section walks through every part of that logic in detail and shows the corresponding MQL5 implementations.

Understanding Time in MetaTrader 5

All trading sessions pivot around time, and in MetaTrader time can mean several things: local computer time, UTC, or server time. For absolute accuracy, our EA always uses broker‑server time, retrieved with _TimeCurrent_(). Every calculation , from drawing windows to triggering alerts , refers to that value, so sessions remain correctly aligned on any broker or time zone.

```
// Truncate to broker's midnight (00:00)
datetime Day0(datetime t)
{
   MqlDateTime mt;
   TimeToStruct(t, mt);
   mt.hour = mt.min = mt.sec = 0;
   return StructToTime(mt);
}
```

With the base day known, session start/finish times are built by converting human‑readable clock strings ( "07:00" , "16:00" ) into minutes:

```
int ParseHM(string s)
{
   int split = StringFind(s, ":");
   if(split < 0) return 0;
   return 60 * (int)StringToInteger(StringSubstr(s, 0, split))
        +     (int)StringToInteger(StringSubstr(s, split + 1));
}

datetime MakeTime(datetime base, string tstr)
{
   return base + ParseHM(tstr) * 60;    // add minutes to 00:00
}
```

Conceptually: Think of Day0() as the “anchor” for that trading day, and MakeTime() as a ruler measuring minutes from midnight. Whether your local computer shows GMT+2, EST, or CET doesn’t matter—everything remains in the broker’s timeline.

Finding the Highs and Lows of Each Session

Once time windows are defined, the EA must compute the open, high, low, and close for that range so it knows where to draw each rectangle.

```
void MakeSession(datetime base, string s1, string s2,
                 string pref, string name, color col, int order,
                 double &outHi, double &outLo, double &outOp, double &outCl,
                 bool labelIt)
{
   int m1 = ParseHM(s1), m2 = ParseHM(s2);
   datetime t1 = base + m1 * 60, t2 = base + m2 * 60;
   if(t2 <= t1) t2 += 86400;   // wrap around midnight if needed

   double hi = -DBL_MAX, lo = DBL_MAX, opn = 0, cls = 0;
   bool haveOpen = false;

   for(int i = 0; i < iBars(_Symbol, _Period); i++)
   {
      datetime bt = iTime(_Symbol, _Period, i);
      if(bt < t1) break;
      if(bt >= t1 && bt <= t2)
      {
         double bh = iHigh(_Symbol,_Period,i);
         double bl = iLow (_Symbol,_Period,i);
         if(bh > hi) hi = bh;
         if(bl < lo) lo = bl;
         if(!haveOpen){ opn = iOpen(_Symbol,_Period,i); haveOpen = true; }
         cls = iClose(_Symbol,_Period,i);
      }
   }

   if(hi > 0 && lo != DBL_MAX)
   {
      DrawSession(pref, name, col, t1, t2, hi, lo, opn, cls, order, labelIt);
      outHi = hi; outLo = lo; outOp = opn; outCl = cls;
   }
}
```

How it works:

- The loop scans historical candles only within that session.
- The moment it moves earlier than the opening time, it stops—saving CPU cycles.
- The resulting values act as both drawing coordinates and reference points for future breakout alerts.

Painting the Sessions

Visual clarity is achieved through the function _DrawSession_(). It creates a semi‑transparent rectangle spanning from the session’s start to end times, vertically bounded by the computed high and low. Each rectangle can optionally carry a small text label showing H/L/O/C for that session.

```
void DrawSession(string pref, string name, color col,
                 datetime t1, datetime t2, double hi, double lo,
                 double opn, double cls, int slot, bool labelIt)
{
   string box = pref + TimeToString(t1, TIME_DATE | TIME_MINUTES);
   SafeDelete(box);
   ObjectCreate(0, box, OBJ_RECTANGLE, 0, t1, hi, t2, lo);
   ObjectSetInteger(0, box, OBJPROP_COLOR, col);
   ObjectSetInteger(0, box, OBJPROP_BACK, true);
   ObjectSetInteger(0, box, OBJPROP_WIDTH, 1);

   if(labelIt)
   {
      string lbl  = box + "_LBL";
      string text = StringFormat("%s  H %.5f  L %.5f  O %.5f  C %.5f",
                                 name, hi, lo, opn, cls);
      DrawLabelNoOverlap(lbl, hi, slot, text, col);
   }
}
```

The helper _DrawLabelNoOverlap_() positions the text so that session labels are separated by a consistent distance:

```
void DrawLabelNoOverlap(string id,double baseY,int slot,string text,color col)
{
   datetime anchor = iTime(_Symbol,_Period,0);
   datetime offsetT = anchor - 4*PeriodSeconds(_Period);
   double offsetY = (slot+1) * _Point * 120;   // uniform vertical spacing

   ObjectCreate(0, id, OBJ_TEXT, 0, offsetT, baseY + offsetY);
   ObjectSetInteger(0, id, OBJPROP_COLOR, col);
   ObjectSetInteger(0, id, OBJPROP_FONTSIZE, 8);
   ObjectSetString (0, id, OBJPROP_TEXT, text);
}
```

The multiplier 120 ensures that the label for London lies the same distance below Tokyo as Tokyo does below Asia, preserving symmetry on any instrument’s price scale.

Controlling the Lifecycle of Sessions

Instead of redrawing each second (which could waste resources), the EA uses a smart update rhythm. A timer event fires once per second:

```
void OnTimer()
{
   ScrollTicker();            // move the text ticker
   CheckSessionAlerts();      // detect openings & closings
   CheckBreakouts();          // look for price range breaks

   static int counter = 0;
   if(++counter >= 60)        // refresh once per minute
   {
      DrawAll();
      UpdatePanel();
      counter = 0;
   }
}
```

Every minute it deletes existing rectangles (via _DeletePrefix_()) and recreates them to stay in sync with current server time. Because everything is event‑driven, the EA uses CPU efficiently even with multiple charts open.

Detecting Session Openings and Closings

The function _CheckSessionAlerts_() compares the current broker time against each session’s open/close schedule. As soon as a boundary is crossed, it pushes a short message into the ticker and, optionally, triggers a sound alert.

```
void CheckSessionAlerts()
{
   datetime now  = TimeCurrent(), base = Day0(now);
   datetime asO  = MakeTime(base, AsiaStart),   asC = MakeTime(base, AsiaEnd);
   datetime lnO  = MakeTime(base, LondonStart), lnC = MakeTime(base, LondonEnd);
   datetime nyO  = MakeTime(base, NewYorkStart),nyC = MakeTime(base, NewYorkEnd);

   if(!openedAsia  && now >= asO){ openedAsia  = true;  UpdateTicker("Asia session opened");  }
   if(!closedAsia  && now >= asC){ closedAsia  = true;  UpdateTicker("Asia session closed");  }
   if(!openedLondon&& now >= lnO){ openedLondon= true;  UpdateTicker("London session opened");}
   if(!closedLondon&& now >= lnC){ closedLondon= true;  UpdateTicker("London session closed");}
   if(!openedNewYork&& now >= nyO){openedNewYork=true;  UpdateTicker("New York session opened");}
   if(!closedNewYork&& now >= nyC){closedNewYork=true;  UpdateTicker("New York session closed");}
}
```

At runtime, these notifications appear smoothly in the scrolling headline, giving traders real‑time awareness of global market transitions.

Breakout Detection

Beyond timing, traders often want to know when price escapes the previous session’s bounds. The EA’s breakout engine performs exactly that job.

```
void CheckBreakouts()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // Example: New York breaking the prior London range
   if(prevLondonHigh > 0)
   {
      if(!newYorkBreakHighDone && bid > prevLondonHigh)
      {
         newYorkBreakHighDone = true;
         UpdateTicker("New York breaks above prior London high");
      }
      if(!newYorkBreakLowDone && bid < prevLondonLow)
      {
         newYorkBreakLowDone = true;
         UpdateTicker("New York breaks below prior London low");
      }
   }
}
```

Each flag ( _BreakHighDone_, _BreakLowDone_) ensures that an alert is sent only once per direction, so messages remain clean and relevant even during sustained trends.

Seeing a sudden alert such as “London breaks above prior Tokyo high” lets traders immediately infer that market volatility is expanding—an excellent tool for timing breakout or reversal strategies.

Handling User Interaction

The interface is interactive. Whenever a trader clicks a session button, MetaTrader sends a chart event captured by _OnChartEvent_(). The EA reacts by flipping a Boolean, redrawing affected rectangles, and updating the info panel.

```
void OnChartEvent(const int id,const long &l,const double &d,const string &s)
{
   if(id == CHARTEVENT_OBJECT_CLICK)
   {
      if(s == BTN_ASIA)    { showAsia    = !showAsia;    DrawAll(); UpdatePanel(); UpdateTicker("Asia toggle changed"); }
      if(s == BTN_TOKYO)   { showTokyo   = !showTokyo;   DrawAll(); UpdatePanel(); UpdateTicker("Tokyo toggle changed"); }
      if(s == BTN_LONDON)  { showLondon  = !showLondon;  DrawAll(); UpdatePanel(); UpdateTicker("London toggle changed"); }
      if(s == BTN_NEWYORK) { showNewYork = !showNewYork; DrawAll(); UpdatePanel(); UpdateTicker("New York toggle changed"); }
   }
}
```

This instantaneous response reinforces the concept of modularity—user interface actions are completely decoupled from analytical logic.

Initialization and Cleanup

Two simple routines frame the EA’s life cycle:

```
int OnInit()
{
   CreateButton(BTN_ASIA,   "Asia ON/OFF",   10, 20, AsiaColor);
   CreateButton(BTN_TOKYO,  "Tokyo ON/OFF",  125,20, TokyoColor);
   CreateButton(BTN_LONDON, "London ON/OFF", 240,20, LondonColor);
   CreateButton(BTN_NEWYORK,"New York ON/OFF",10,45, NewYorkColor);

   CreatePanel();
   CreateTicker();
   DrawAll();
   UpdatePanel();
   UpdateTicker(Headline);

   EventSetTimer(1);     // start 1‑second timer
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();              // stop the timer
   DeletePrefix(PREF_ASIA);
   DeletePrefix(PREF_TOKYO);
   DeletePrefix(PREF_LONDON);
   DeletePrefix(PREF_NEWYORK);
}
```

On start‑up the EA builds its full interface; when removed, it cleans up completely so no orphaned rectangles remain.

Logic Flow in Action

To visualize the continuous process:

![](https://c.mql5.com/2/177/flowchart_o28.png)

Each stage operates independently yet communicates through shared global data—creating an always‑aware, self‑correcting tool.

Compiling and Running the EA

Once the entire script is copied into MetaEditor, save it inside

```
MQL5\Experts\All Sessions Toggle EA.mq5
```

Then press Compile (F7) and attach the EA to any chart in MetaTrader 5. Once loaded, the chart should instantly transforms into an interactive dashboard: four toggle buttons appear in the upper‑left corner representing the Asia, Tokyo, London, and New York sessions; a sleek black information panel occupies the upper‑right corner displaying which sessions are currently ON or OFF; and a scrolling ticker headline runs along the bottom, reporting live status messages. Across the main chart area, color‑coded rectangles cover the previous and current trading days, each clearly labelled once with its own H, L, O, and C values for quick reference. If everything compiled correctly, the Experts tab at the bottom of the terminal will confirm successful initialization with the message “Sessions viewer running.”

### Testing and Validation

Before releasing or relying on any trading tool, thorough testing is essential. For this EA, testing confirms that all components—graphical, logical, and time‑based—behave consistently across different brokers, instruments, and time zones. This section outlines a systematic method to validate the EA’s performance and reliability.

![](https://c.mql5.com/2/176/session_ea.gif)

The diagram above illustrates the All Sessions EA running on the Deriv Demo Account (EURUSD H1) chart. During this test, all four sessions—Asia, Tokyo, London, and New York—were enabled simultaneously. Each rectangle is displayed in its assigned color: Sky Blue for Asia, Light Green for Tokyo, Light Pink for London, and Gold for New York. In the upper‑left corner, the toggle buttons (“Asia ON/OFF,” “Tokyo ON/OFF,” etc.) confirm that the interactive interface loaded correctly within Deriv’s MetaTrader 5 environment. When clicked, each button instantly hides or redraws its corresponding session box, showing that the event‑handling and redraw functions are fully responsive under this broker’s infrastructure. The black information panel on the upper‑right accurately reports the live state of all sessions:

- ASIA : ON TOKYO : ON LONDON : ON NEW YORK : ON

This confirms that the UpdatePanel() routine synchronized the graphical panel with the Boolean flags controlling session visibility. Below it, the active ticker headline appears with the text “Sessions viewer running – toggle button”, verifying that the ticker object initialized properly and continues scrolling without overlap. Within the main price area, colored rectangles align perfectly with candle time boundaries on Deriv’s H1 chart, showing that the EA reads Deriv’s server time correctly through TimeCurrent(). Only one H‑L‑O‑C label appears per session, matching the design rule of single‑label display for clarity. The values beside each session title, such as “NEW YORK H 1.16224” and “TOKYO H 1.16158 L 1.158xx”—confirm that the EA collected OHLC data precisely from Deriv’s price feed. No flickering or performance lag was observed while switching between symbols (EURUSD, USDCHF, GBPUSD, and USDJPY), indicating that the timer refresh mechanism (1‑second interval) performs stably on Deriv’s platform. CPU and memory usage remained minimal throughout the session.

Overall, this test validates that:

- The EA executes and renders perfectly in Deriv’s MetaTrader 5 environment.
- Time alignment and session lengths match global standards.
- Interactive elements—buttons, panel updates, and ticker—operate smoothly.
- The breakout‑alert system triggers accurately when price crosses the prior session’s extremes.

This Deriv‑based assessment confirms that the trading‑session logic, drawing routines, and alert mechanisms are stable and broker‑accurate under real‑time chart conditions.

The Gif below is the alert popup

![](https://c.mql5.com/2/176/alert.gif)

### Conclusion

The creation of the All Sessions EA showcases how a simple concept—visually dividing the trading day into global market sessions—can mature into a refined analytical instrument. Originally intended as a helper for identifying session times, the project has grown into a complete, interactive system that synchronizes seamlessly with broker‑server time, reports live market transitions, and alerts traders to meaningful price breakouts as they happen.

Through its blend of color‑coded rectangles, concise session labels, toggle buttons, and a scrolling ticker, the EA transforms the continuous rhythm of the foreign‑exchange market into an easy‑to‑read story. It allows traders to see at a glance which global region currently drives liquidity and how volatility shifts from Asia to Tokyo, through London, and on to New York. The modular, event‑driven structure ensures these visuals remain accurate and lightweight on any chart.

For new traders, the EA acts as an educational guide that reveals how market energy flows throughout the trading day. For experienced users, it becomes a context‑building overlay—streamlining intraday planning, confirming breakout behavior, and simplifying session analysis.

In this development cycle, the EA was thoroughly tested on Deriv’s trading environment, where it performed smoothly across different chart types and timeframes. The results confirmed full synchronization with Deriv’s server time, stable object rendering, accurate alert timing, and consistent H‑L‑O‑C label updates. Even under continuous operation, the EA maintained responsive performance and clean graphic behaviour, validating its efficiency in Deriv’s live and synthetic markets. Because the underlying architecture is modular, future customization remains simple. We can easily extend it to include features such as the Sydney session, push‑notification support, or historical session‑range statistics without rewriting the core logic.

Ultimately, the EA is more than a colored overlay; it is a teaching and analytical companion that helps traders understand not just what prices are doing but when and why they move. By following the ebb and flow of each global session, traders using the Deriv platform—and any future supported brokers—gain a clear, time‑based perspective on volatility and liquidity. With precision, clarity, and simplicity at its heart, this EA turns the continuous 24‑hour market into a structured, visually intuitive experience—one session at a time.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19944.zip "Download all attachments in the single ZIP archive")

[All\_Sessions\_EA.mq5](https://www.mql5.com/en/articles/download/19944/All_Sessions_EA.mq5 "Download All_Sessions_EA.mq5")(32.86 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Price Action Analysis Toolkit (Part 55): Designing a CPI Mini-Candle Overlay for Intra-bar Pressure](https://www.mql5.com/en/articles/20949)
- [Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://www.mql5.com/en/articles/20851)
- [Price Action Analysis Toolkit Development (Part 53): Pattern Density Heatmap for Support and Resistance Zone Discovery](https://www.mql5.com/en/articles/20390)
- [Price Action Analysis Toolkit Development (Part 52): Master Market Structure with Multi-Timeframe Visual Analysis](https://www.mql5.com/en/articles/20387)
- [Price Action Analysis Toolkit Development (Part 51): Revolutionary Chart Search Technology for Candlestick Pattern Discovery](https://www.mql5.com/en/articles/20313)
- [Price Action Analysis Toolkit Development (Part 50): Developing the RVGI, CCI and SMA Confluence Engine in MQL5](https://www.mql5.com/en/articles/20262)
- [Price Action Analysis Toolkit Development (Part 49): Integrating Trend, Momentum, and Volatility Indicators into One MQL5 System](https://www.mql5.com/en/articles/20168)

**[Go to discussion](https://www.mql5.com/en/forum/498771)**

![Building a Smart Trade Manager in MQL5: Automate Break-Even, Trailing Stop, and Partial Close](https://c.mql5.com/2/177/19911-building-a-smart-trade-manager-logo.png)[Building a Smart Trade Manager in MQL5: Automate Break-Even, Trailing Stop, and Partial Close](https://www.mql5.com/en/articles/19911)

Learn how to build a Smart Trade Manager Expert Advisor in MQL5 that automates trade management with break-even, trailing stop, and partial close features. A practical, step-by-step guide for traders who want to save time and improve consistency through automation.

![Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://c.mql5.com/2/177/20026-statistical-arbitrage-through-logo__1.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)

In this article, we propose a scoring system for mean-reversion strategies based on statistical arbitrage of cointegrated stocks. The article suggests criteria that go from liquidity and transaction costs to the number of cointegration ranks and time to mean-reversion, while taking into account the strategic criteria of data frequency (timeframe) and the lookback period for cointegration tests, which are evaluated before the score ranking properly. The files required for the reproduction of the backtest are provided, and their results are commented on as well.

![Automating Trading Strategies in MQL5 (Part 37): Regular RSI Divergence Convergence with Visual Indicators](https://c.mql5.com/2/176/20031-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 37): Regular RSI Divergence Convergence with Visual Indicators](https://www.mql5.com/en/articles/20031)

In this article, we build an MQL5 EA that detects regular RSI divergences using swing points with strength, bar limits, and tolerance checks. It executes trades on bullish or bearish signals with fixed lots, SL/TP in pips, and optional trailing stops. Visuals include colored lines on charts and labeled swings for better strategy insights.

![Introduction to MQL5 (Part 26): Building an EA Using Support and Resistance Zones](https://c.mql5.com/2/177/20021-introduction-to-mql5-part-26-logo.png)[Introduction to MQL5 (Part 26): Building an EA Using Support and Resistance Zones](https://www.mql5.com/en/articles/20021)

This article teaches you how to build an MQL5 Expert Advisor that automatically detects support and resistance zones and executes trades based on them. You’ll learn how to program your EA to identify these key market levels, monitor price reactions, and make trading decisions without manual intervention.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/19944&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049393256967547575)

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