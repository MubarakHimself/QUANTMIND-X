---
title: From Novice to Expert: Animated News Headline Using MQL5 (I)
url: https://www.mql5.com/en/articles/18299
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:06:57.262311
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/18299&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071585116762876607)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/18299para1)
- [Concept](https://www.mql5.com/en/articles/18299#para2)
- [Implementation](https://www.mql5.com/en/articles/18299#para3)
- [Testing](https://www.mql5.com/en/articles/18299#para4)
- [Conclusion](https://www.mql5.com/en/articles/18299#para5)

### Introduction

Today, we aim to address a common limitation in how economic news and calendar events are accessed within the MetaTrader 5 terminal—specifically, during active chart analysis.

In MetaTrader 5, both the News and Economic Calendar tabs are available under the Toolbox window. These tabs provide important information sourced from reputable news providers. However, it's essential to understand the difference between the two:

- News Tab: Displays headlines and updates that have already been released.
- Economic Calendar: Provides a schedule of upcoming economic events, categorized by date, time, and importance—useful for planning ahead.

Both tools are vital for market analysis. Experienced traders know that some economic releases—such as interest rate decisions or non-farm payrolls—can significantly move the market. With the right preparation, such events offer profitable trading opportunities. However, they also carry substantial risk if trades go against expectations.

![Accessing News and Calendar on MetaTrader 5](https://c.mql5.com/2/148/terminal64_UgLq3U2Lkw.gif)

Accessing the News and Calendar on MetaTrader 5.

In the screenshot shared earlier, you’ll notice that MetaTrader 5 provides access to both the News and the [Economic Calendar](https://www.mql5.com/en/book/advanced/calendar) within a single integrated environment. However, a key limitation becomes apparent: to view either of these features, the user must manually navigate to the Toolbox window.

Once inside the Toolbox, the information is presented in tabular form—rows and columns display various news items and scheduled events, along with relevant details. Users can scroll through the news feed to read headlines or browse the calendar to view upcoming economic releases. While the Toolbox panel can be expanded to show more content, doing so comes at the cost of reducing the chart window size—potentially obstructing price action, indicators, or graphical objects on the chart.

Our proposed solution addresses this limitation by projecting news headlines and calendar events directly onto the chart, maintaining full visibility without intruding on the chart space or interfering with trading objects. The goal is to improve accessibility and situational awareness while preserving a clean and functional chart environment.

In this introductory section of our miniseries and I will start by mainly implementing the MQL5 economic calendar to display a headline of upcoming news events. The solution here borrows from how news headlines are normally displayed at the bottom of a television screen, or how ads are normally aired that way on social media videos.

In the following section, we will present the approach to be used to build the News Headline EA using MQL5. We'll outline the design strategy, walk through key implementation decisions, and then dive into the codebase in detail.

Finally, we’ll conclude with testing outcomes and observations, providing a complete end-to-end view of the development lifecycle for this real-time news display system in MetaTrader 5.

### Concept

To bring this idea to life, we’ll leverage the [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas), which provides robust tools for graphical interface development. Specifically, we'll utilize the CCanvas class, located in the _MQL5\\Include\\Canvas\\Canvas.mqh_ file. This class allows us to create rectangular, transparent drawing surfaces—ideal for overlaying dynamic content like news headlines and economic event updates directly onto the chart.

Our implementation centers around a scrolling ticker system that continuously displays relevant information without interrupting chart functionality. This concept is inspired by real-world interfaces seen in television news broadcasts, financial websites, and even social media video content, where attention-grabbing headlines move across the screen to provide real-time updates.

![](https://c.mql5.com/2/148/chrome_aHVvuZVePi.png)

Headline definition snippet from [Google search](https://www.mql5.com/go?link=https://www.google.com/search?sca_esv=5cd7a0208e12e544%26sxsrf=AE3TifMTJ5iiNY8Yj7wIvjgQwLoLHcu0Yw:1749484245744%26q=headline%26source=lnms%26fbs=AIIjpHxU7SXXniUZfeShr2fp4giZ1Y6MJ25_tmWITc7uy4KIeqDdErwP5rACeJAty2zADJgeXKnD4z7v_UXM32TmNnj1yxfQDJKRFuKsiopx1kXI2L3_Ix7VdVKHJorz0W9iZI5L-yMQtsNVCCQflLZwbIMu_HARWNsZKmCew6HVlOn98Y4eg2baxo8ae09qoNX97cxoWFCUP4lJ6A3w-qJa6UrDnsETQg%26sa=X%26ved=2ahUKEwig3YfY2OSNAxXEdUEAHVB6IPYQ0pQJegQIEhAB%26biw=1522%26bih=696%26dpr=1.25 "https://www.google.com/search?sca_esv=5cd7a0208e12e544&sxsrf=AE3TifMTJ5iiNY8Yj7wIvjgQwLoLHcu0Yw:1749484245744&q=headline&source=lnms&fbs=AIIjpHxU7SXXniUZfeShr2fp4giZ1Y6MJ25_tmWITc7uy4KIeqDdErwP5rACeJAty2zADJgeXKnD4z7v_UXM32TmNnj1yxfQDJKRFuKsiopx1kXI2L3_Ix7VdVKHJorz0W9iZI5L-yMQtsNVCCQflLZwbIMu_HARWNsZKmCew6HVlOn98Y4eg2baxo8ae09qoNX97cxoWFCUP4lJ6A3w-qJa6UrDnsETQg&sa=X&ved=2ahUKEwig3YfY2OSNAxXEdUEAHVB6IPYQ0pQJegQIEhAB&biw=1522&bih=696&dpr=1.25").

At this point, you might wonder why we’re specifically calling this a “News headline” Expert Advisor. The term headline refers to a brief, prominent summary of key information—designed to be easily seen and quickly understood. In our case, it serves as a compact way to convey time-sensitive market data (e.g., news events or economic releases) in an accessible and visually appealing format.

![The news headline concept](https://c.mql5.com/2/148/Economic_Calendar.png)

The concept: Retrieve economic calendar data using the MQL5 API and display upcoming events directly on the chart through our custom News Headline EA.

To simplify this stage of development, we’ll begin by focusing on two core components:

- Understanding and utilizing the CCanvas class in MQL5, which enables custom graphical drawing on the chart.
- Implementing the MQL5 Economic Calendar, using built-in functions to retrieve upcoming events and display them using a horizontal scrolling effect.

This horizontal scrolling design is intentional—it preserves valuable vertical space on the chart and offers an inline, non-intrusive display of time-sensitive information.

As a forward-looking enhancement, we also plan to integrate an external news API in future iterations. This will allow us to display real-time market news in a separate scrolling lane below the calendar events. For now, we’ll lay the groundwork by inserting a placeholder text to represent the news feed.

At a glance, a trader using this system will be able to:

- Instantly identify important upcoming events.
- See how soon they will occur (e.g., in hours or minutes).
- Recognize their expected impact level—allowing the trader to adopt a more cautious or informed strategy.

Having covered the core concepts, we now transition to a deeper exploration of the implementation details.

### Implementation

Understanding the Canvas header

The [CCanvas](https://www.mql5.com/en/docs/standardlibrary/canvasgraphics/ccanvas) class in MQL5 is a powerful and versatile utility designed for creating and managing custom graphical interfaces and visual elements directly within MetaTrader 5 charts. At its core, CCanvas allows developers to create in-memory bitmap surfaces that are rendered through chart objects like OBJ\_BITMAP and OBJ\_BITMAP\_LABEL. This enables dynamic drawing of graphical content—including lines, shapes, polygons, and text—on top of charts without interfering with native chart elements. The class provides low-level control over rendering through a pixel buffer (m\_pixels\[\]), along with functions like CreateBitmapLabel() for initialization and Update() to refresh the output.

CCanvas offers strong resource management: developers can load or save bitmaps, attach/detach canvases from charts, and manipulate pixels directly using PixelSet() or PixelGet()—making it ideal for performance-sensitive applications. Its flexibility extends to color formats (ARGB, XRGB, with transparency), polygon rendering (including non-convex fills), and layered chart interfaces.

In practice, CCanvas opens the door to sophisticated UI development in MetTrader 5. It's commonly used for custom indicators (like smoothed or shaded overlays), interactive tools (such as trendlines with custom caps), trade visualization dashboards, and full-featured on-chart panels with buttons, sliders, or even scrolling news tickers. Though CPU-bound rendering limits it in ultra-high-resolution contexts, its pixel-level precision and complete customizability make it indispensable for high-end chart interfaces.

Undestanding the [Economic calendar](https://www.mql5.com/en/book/advanced/calendar)

When you work with the MetaTrader 5 economic calendar, the very first thing to understand is that every event is tied to a particular country (or economic union) via a unique country identifier. In MQL5 this is represented by the MqlCalendarCountry structure, which includes fields like id (the ISO 3166-1 code), the country name, two-letter code, currency code and symbol, and even the URL-friendly country name. By querying the calendar’s list of MqlCalendarCountry entries once, you gain access to all the attributes you need to filter or group events by region. Each calendar event then refers back to its country through the country\_id field in the MqlCalendarEvent structure.

That structure itself describes the general characteristics of a recurring event type—its name, importance, sector of the economy (GDP, jobs, prices, etc.), its periodicity (daily, monthly, quarterly), and the units in which its values are reported. Crucially, it does not represent a single occurrence but rather the “template” or definition of an event (for example, “US CPI release”), which the calendar service can schedule many times over its published history.

Actual scheduled occurrences of those event types—complete with timestamps, actual and forecast values, and any revisions—are held in a separate table of MqlCalendarValue structures. Each MqlCalendarValue record carries an event\_id (linking back to the template in MqlCalendarEvent), a precise time and period, plus four numeric fields (actual\_value, forecast\_value, prev\_value, revised\_prev\_value) that may or may not be populated yet. Helper methods like HasActualValue() and GetActualValue() make it easy to check for and retrieve real-world values (automatically scaled back from the calendar’s internal “ppm” representation).

This relational design—countries, event types, then event occurrences—ensures data is never duplicated needlessly: quarterly CPI entries, for instance, all point back to a single CPI definition that carries its importance level, units, and frequency. By understanding these structures and how they reference one another, we can precisely filter, format, and display only the upcoming events we care about, while keeping our code both efficient and maintainable.

The News Headline EA

Setting Up User Controls

First, we decide which parameters traders should be able to tweak. We expose scroll speeds for each importance lane (InpSpeedHigh, InpSpeedMed, InpSpeedLow), the news‐ticker speed (InpNewsSpeed), and the frame interval (InpTimerMs). We also let the user choose whether the ticker lives at the top or bottom (InpPositionTop), how far it should sit from the chart edge (InpTopOffset), and which lanes to show (ShowHigh, ShowMed, ShowLow). By isolating these in a neat “User Inputs” block, anyone can quickly adjust the behavior without digging into implementation details.

```
//+------------------------------------------------------------------+
//| 1) USER INPUTS                                                   |
//+------------------------------------------------------------------+
input int   InpSpeedHigh   = 4;    // px/frame for High-impact lane
input int   InpSpeedMed    = 2;    // px/frame for Medium-impact lane
input int   InpSpeedLow    = 1;    // px/frame for Low-impact lane
input int   InpNewsSpeed   = 5;    // px/frame for news ticker row
input int   InpTimerMs     = 50;   // ms between frames (~20 fps)
input bool  InpPositionTop = true; // true=top, false=bottom
input int   InpTopOffset   = 50;   // px offset from chart edge
input bool  ShowHigh       = true; // toggle High lane
input bool  ShowMed        = true; // toggle Medium lane
input bool  ShowLow        = true; // toggle Low lane
```

Defining Layout Constants

Next, we establish fixed spacing rules that govern the visual layout: how many pixels separate the time-left label from the currency symbol (GapTimeToSym), how much padding we give around our inline importance box (GapSymToRect, GapRectToName), and the size of that box (RectSize). By centralizing these values, we can fine-tune the look and feel in one place, rather than hunting through drawing code.

```
//+------------------------------------------------------------------+
//| 2) DEVELOPER CONSTANTS                                           |
//+------------------------------------------------------------------+
static const int GapTimeToSym = 10;  // px gap after “[1h]”
static const int GapSymToRect = 5;   // px gap before inline box
static const int RectSize     = 8;   // width & height of inline box
static const int GapRectToName= 10;  // px gap after inline box
```

Storing State and Drawing Buffers

We then declare global variables to hold chart dimensions (canvW), row height (lineH), placeholder text for the news lane, and a timestamp to avoid repeated calendar queries (lastReloadDay). We also instantiate two CCanvas objects—one for the three event lanes, and one for the news ticker. Finally, we define our CEvent class and three dynamic arrays (highArr, medArr, lowArr) to store incoming calendar events by importance. Each lane’s current scroll offset (offHigh, etc.) completes the state we maintain as the EA runs.

```
//+------------------------------------------------------------------+
//| 3) GLOBALS                                                       |
//+------------------------------------------------------------------+
int        lineH      = 16;           // row height in px
int        canvW;                     // chart width
string     placeholder =              // news ticker text
           "News feed coming soon – stay tuned with the calendar";
datetime   lastReloadDay = 0;         // daily reload guard

CCanvas    eventsCanvas, newsCanvas;  // two layers

// Event struct and arrays
class CEvent : public CObject
{
public:
  datetime time;
  string   sym, name;
  int      imp;
  CEvent(datetime t,const string &S,const string &N,int I)
    { time=t; sym=S; name=N; imp=I; }
};
CEvent *highArr[], *medArr[], *lowArr[];
int     offHigh, offMed, offLow, offNews;
```

Positioning and Sorting Helpers

To keep our main logic clean, we factor out two small helper functions. SetCanvas() places a canvas object at either the top or bottom of the chart, based on user settings. SortArr() is a simple bubble-sort that orders each importance array by event time, ensuring our lanes always display upcoming events in the right sequence.

```
//+------------------------------------------------------------------+
//| Helper: position a canvas label                                  |
//+------------------------------------------------------------------+
void SetCanvas(string name,bool top,int yDist)
{
  ObjectSetInteger(0,name,OBJPROP_CORNER,    top?CORNER_LEFT_UPPER:CORNER_LEFT_LOWER);
  ObjectSetInteger(0,name,OBJPROP_XDISTANCE, 0);
  ObjectSetInteger(0,name,OBJPROP_YDISTANCE, yDist);
}

//+------------------------------------------------------------------+
//| Helper: sort events by time                                      |
//+------------------------------------------------------------------+
void SortArr(CEvent* &arr[])
{
  int n=ArraySize(arr);
  for(int i=0;i<n-1;i++)
    for(int j=i+1;j<n;j++)
      if(arr[i].time > arr[j].time)
      {
        CEvent *tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp;
      }
}
```

Fetching Today’s Events

The ReloadEvents() function is central to how we pull and filter data. It queries MetaTrader’s economic calendar for events scheduled between midnight today and 24 hours later. We skip any event whose timestamp is already past. Each valid event is wrapped in a CEvent object, then placed into the high/medium/low array according to its importance. At the end, we sort each lane so that the earliest event appears first in that lane’s scroll.

```
//+------------------------------------------------------------------+
//| ReloadEvents: load only *future* events for *today*              |
//+------------------------------------------------------------------+
void ReloadEvents()
{
  datetime srv = TimeTradeServer();
  // midnight today
  MqlDateTime dt; TimeToStruct(srv, dt);
  MqlDateTime m0 = {dt.year, dt.mon, dt.day,0,0,0};
  datetime today = StructToTime(m0);
  if(today == lastReloadDay) return;
  lastReloadDay = today;

  // clear previous
  for(int i=0;i<ArraySize(highArr);i++) delete highArr[i];
  for(int i=0;i<ArraySize(medArr); i++) delete medArr[i];
  for(int i=0;i<ArraySize(lowArr); i++) delete lowArr[i];
  ArrayResize(highArr,0); ArrayResize(medArr,0); ArrayResize(lowArr,0);

  // fetch events [today, today+24h)\
  MqlCalendarValue vals[];\
  int cnt = CalendarValueHistory(vals, today, today+86400);\
  for(int i=0;i<cnt;i++)\
  {\
    if(vals[i].time <= srv) continue; // skip past\
    MqlCalendarEvent e;\
    if(!CalendarEventById(vals[i].event_id, e)) continue;\
    MqlCalendarCountry c;\
    if(!CalendarCountryById(e.country_id, c)) continue;\
    string sym = "[" + c.currency + "]";\
    CEvent *ev = new CEvent(vals[i].time, sym, e.name, e.importance);\
    // classify\
    if(e.importance==CALENDAR_IMPORTANCE_HIGH)\
      { int s=ArraySize(highArr)+1; ArrayResize(highArr,s); highArr[s-1]=ev; }\
    else if(e.importance==CALENDAR_IMPORTANCE_MODERATE)\
      { int s=ArraySize(medArr)+1; ArrayResize(medArr,s); medArr[s-1]=ev; }\
    else\
      { int s=ArraySize(lowArr)+1;  ArrayResize(lowArr,s);  lowArr[s-1]=ev; }\
  }\
  SortArr(highArr); SortArr(medArr); SortArr(lowArr);\
}\
```\
\
In detail, when ReloadEvents() runs, it pulls a full list of today’s calendar entries via CalendarValueHistory(), but each raw entry only tells us an event\_id and a country\_id. By joining those against the MqlCalendarEvent table—where each event type is defined with its name, frequency, sector, and most crucially its importance—we can present only the truly market-moving items. The MqlCalendarCountry structure ensures we label each headline with the correct currency (e.g. \[USD\] for the United States), drawn from its ISO code and symbol. This two-step lookup (value → event type → country) makes our EA both learn—pulling only what’s needed—and accurate, since we never hard-code country or event details, but rely on MetaTrader’s own constantly synchronized database.\
\
The importance constants (CALENDAR\_IMPORTANCE\_HIGH, ...\_MODERATE, ...\_LOW) lie at the heart of our lane logic. By selecting which importance levels to include (ShowHigh/ShowMed/ShowLow) and coloring each inline box red, orange, or white, we give the trader an immediate visual cue: red flags the highest-impact releases (think Fed rate decisions, non-farm payrolls), orange the medium-impact (CPI, retail sales), and white the lesser ones (minor speeches, lower-tier data).\
\
In practice, this helps a trader gauge at a glance whether they need to tighten stops or even pause automated strategies as a high-importance event approaches. Without that filter and color-coded labeling—powered by the importance field in MqlCalendarEvent—a scrolling list of dozens of entries would quickly become noise instead of signal.\
\
Rendering a Scrolling Lane\
\
DrawLane() encapsulates the logic for one horizontal strip. We choose a monospaced font (“Courier New”) so that every character—including brackets and digits—takes the same width, ensuring neat alignment. We then draw:\
\
1. Time-left label (hours or minutes).\
2. Currency symbol.\
3. Inline importance box (colored red/orange/white).\
4. Event name, followed by a separator if more events follow.\
\
Finally, we reduce the lane’s offset by the lane speed, and if the entire row has scrolled off the left edge, we wrap it back to the right.\
\
```\
//+------------------------------------------------------------------+\
//| DrawLane: scroll one lane with inline importance box             |\
//+------------------------------------------------------------------+\
void DrawLane(CEvent* &arr[], int &offset, int y, int speed)\
{\
  int n=ArraySize(arr);\
  if(n==0) return;\
\
  // monospaced for alignment\
  eventsCanvas.FontNameSet("Courier New");\
  eventsCanvas.FontSizeSet(-100);\
\
  int x = offset;\
  datetime srv = TimeTradeServer();\
\
  for(int i=0;i<n;i++)\
  {\
    CEvent *e = arr[i];\
    // time-left “[1h]” or “[45m]”\
    long diff = (long)e.time - (long)srv;\
    string tl = (diff>=3600 ? IntegerToString(diff/3600)+"h"\
                            : IntegerToString(diff/60)+"m");\
    string part = "[" + tl + "]";\
    eventsCanvas.TextOut(x,y,part,XRGB(255,255,255),ALIGN_LEFT);\
    x += eventsCanvas.TextWidth(part) -20;\
\
    // symbol “[USD]”\
    eventsCanvas.TextOut(x,y,e.sym,XRGB(255,255,255),ALIGN_RIGHT);\
    x += eventsCanvas.TextWidth(e.sym) + GapSymToRect;\
\
    // inline importance box\
    uint col = (e.imp==CALENDAR_IMPORTANCE_HIGH    ? XRGB(255,0,0) :\
                e.imp==CALENDAR_IMPORTANCE_MODERATE? XRGB(255,165,0):\
                                                     XRGB(255,255,255));\
    eventsCanvas.FillRectangle(x, y + (lineH-RectSize)/2,\
                               x+RectSize, y + (lineH-RectSize)/2 + RectSize,\
                               col);\
    x += RectSize + GapRectToName;\
\
    // event name + separator\
    eventsCanvas.TextOut(x,y,e.name,XRGB(255,255,255),ALIGN_RIGHT);\
    x += eventsCanvas.TextWidth(e.name)+60;\
    if(i+1<n)\
    {\
      eventsCanvas.TextOut(x,y,"|",XRGB(180,180,180),ALIGN_RIGHT);\
      x += eventsCanvas.TextWidth("|") + 20;\
    }\
  }\
\
  // scroll + wrap\
  int totalW = x - offset;\
  offset -= speed;\
  if(offset + totalW < 0) offset = canvW;\
}\
```\
\
Orchestrating All Lanes and the News Row\
\
In DrawAll(), we layer the three event lanes vertically, then the news placeholder underneath (or above, depending on position). After drawing events to eventsCanvas, we call Update(false) to push them to the chart object. The news lane uses its own newsCanvas, with a simpler text-only draw followed by Update(true) to refresh synchronously.\
\
```\
//+------------------------------------------------------------------+\
//| DrawAll: render lanes + news row                                |\
//+------------------------------------------------------------------+\
void DrawAll()\
{\
  // clear events\
  eventsCanvas.Erase(ARGB(180,0,0,0));\
  int y=0;\
\
  if(ShowHigh)\
  {\
    DrawLane(highArr, offHigh, y, InpSpeedHigh);\
    y += lineH;\
  }\
  if(ShowMed)\
  {\
    DrawLane(medArr, offMed, y, InpSpeedMed);\
    y += lineH;\
  }\
  if(ShowLow)\
  {\
    DrawLane(lowArr, offLow, y, InpSpeedLow);\
    y += lineH;\
  }\
  eventsCanvas.Update(false);\
\
  // news placeholder\
  newsCanvas.Erase(ARGB(170,0,0,0));\
  newsCanvas.FontNameSet("Tahoma");\
  newsCanvas.FontSizeSet(-120);\
  int yOff = (lineH - newsCanvas.TextHeight(placeholder)) / 2;\
  newsCanvas.TextOut(offNews, yOff, placeholder, XRGB(255,255,255), ALIGN_LEFT);\
  offNews -= InpNewsSpeed;\
  if(offNews + newsCanvas.TextWidth(placeholder) < -20) offNews = canvW;\
  newsCanvas.Update(true);\
}\
```\
\
Initialization, Timer, and Cleanup\
\
Finally, in OnInit() we create and configure our canvases, call ReloadEvents() for the first time, set all offsets to canvW, and position the two canvases based on InpPositionTop and InpTopOffset. We then draw the first frame and start the millisecond timer.\
\
OnTimer() simply repositions the canvases (so users can flip InpPositionTop live), reloads events once per day, adjusts for chart resizing, and calls DrawAll() again. OnDeinit() cleans up the canvases and deletes any allocated CEvent objects.\
\
```\
//+------------------------------------------------------------------+\
//| OnInit: setup canvases, initial load & position                 |\
//+------------------------------------------------------------------+\
int OnInit()\
{\
  // force reload Today\
  lastReloadDay = 0;\
\
  // clear arrays\
  ArrayResize(highArr,0);\
  ArrayResize(medArr,0);\
  ArrayResize(lowArr,0);\
\
  // chart width\
  canvW = (int)ChartGetInteger(0,CHART_WIDTH_IN_PIXELS);\
\
  // create events canvas (4 rows tall)\
  eventsCanvas.CreateBitmapLabel("EvCanvas",0,0,canvW,4*lineH,COLOR_FORMAT_ARGB_RAW);\
  eventsCanvas.TransparentLevelSet(150);\
\
  // create news canvas (1 row tall)\
  newsCanvas.CreateBitmapLabel("NwCanvas",0,0,canvW,lineH,COLOR_FORMAT_ARGB_RAW);\
  newsCanvas.TransparentLevelSet(0);\
\
  // load data + init offsets\
  ReloadEvents();\
  offHigh = offMed = offLow = offNews = canvW;\
\
  // initial positioning\
  {\
    int rows = (ShowHigh?1:0)+(ShowMed?1:0)+(ShowLow?1:0);\
    int yOff = InpTopOffset + (InpPositionTop ? 0 : rows*lineH);\
    SetCanvas("EvCanvas", InpPositionTop, InpTopOffset);\
    SetCanvas("NwCanvas", InpPositionTop, yOff + (InpPositionTop ? rows*lineH : 0));\
  }\
\
  // first draw & timer\
  DrawAll();\
  EventSetMillisecondTimer(InpTimerMs);\
  return INIT_SUCCEEDED;\
}\
\
//+------------------------------------------------------------------+\
//| OnTimer: reposition, daily reload, redraw                       |\
//+------------------------------------------------------------------+\
void OnTimer()\
{\
  // reposition every tick\
  int rows = (ShowHigh?1:0)+(ShowMed?1:0)+(ShowLow?1:0);\
  if(InpPositionTop)\
  {\
    SetCanvas("EvCanvas", true,  InpTopOffset);\
    SetCanvas("NwCanvas", true,  InpTopOffset + rows*lineH);\
  }\
  else\
  {\
    SetCanvas("EvCanvas", false, InpTopOffset);\
    SetCanvas("NwCanvas", false, InpTopOffset + lineH);\
  }\
\
  // reload once per day\
  ReloadEvents();\
\
  // adapt width\
  int wNew = (int)ChartGetInteger(0,CHART_WIDTH_IN_PIXELS);\
  if(wNew != canvW)\
  {\
    canvW = wNew;\
    ObjectSetInteger(0,"EvCanvas",OBJPROP_WIDTH,canvW);\
    ObjectSetInteger(0,"NwCanvas",OBJPROP_WIDTH,canvW);\
  }\
\
  // redraw\
  DrawAll();\
}\
\
//+------------------------------------------------------------------+\
//| OnDeinit: cleanup                                               |\
//+------------------------------------------------------------------+\
void OnDeinit(const int reason)\
{\
  EventKillTimer();\
  eventsCanvas.Destroy(); ObjectDelete(0,"EvCanvas");\
  newsCanvas.Destroy();   ObjectDelete(0,"NwCanvas");\
  for(int i=0;i<ArraySize(highArr);i++) delete highArr[i];\
  for(int i=0;i<ArraySize(medArr); i++) delete medArr[i];\
  for(int i=0;i<ArraySize(lowArr); i++) delete lowArr[i];\
}\
```\
\
With this stage complete, we can now proceed to test our code on the chart. During development, I resolved many of the compilation errors, resulting in a clean and well-structured final version. By compiling the components above, we now have a fully functional News Headline EA ready for deployment on the chart. Explore my testing experience in the next section.\
\
### Testing\
\
In the MetaTrader 5 terminal, navigate to the Expert Advisors section and drag the News Headline EA onto the chart. Once successfully added, the EA appears by default at the top of the chart with a vertical offset of 50 pixels. This offset prevents it from overlapping the Depth of Market and Trade Panel buttons, as well as the EA name in the top-right corner.\
\
You can adjust this offset to position the headline canvas at any desired vertical location on the chart. The EA features four lanes, each capable of optimized headline speed, running at a frame rate of 20 FPS. Upcoming news events are displayed in color-coded rectangles that indicate their level of importance.\
\
![Testing the News Headline EA.mq5](https://c.mql5.com/2/148/terminal64_U1mxhDbYeG.gif)\
\
Testing the News Headline EA\
\
The image above shows a successful deployment of the News Headline EA. It appears as intended, displaying all upcoming news events with smooth and seamless animation.\
\
### Conclusion\
\
This marks the end of yet another exciting development discussion, culminating in a practical and insightful tool for both traders and developers. We successfully leveraged the power of the Canvas class to achieve efficient rendering and visual clarity—an approach that also serves as a valuable shortcut in interface development.\
\
Throughout this project, we learned how to retrieve economic calendar data and display it in a meaningful, trader-friendly format. The result is a clean, minimalistic view of upcoming news events, right on the chart—solving a long-standing challenge in news-based trading tools.\
\
Looking ahead, the second version of this EA will incorporate API access for real-time news feeds, enabling even more dynamic updates. Additionally, the Canvas can be repurposed to display other trading-relevant data, making this approach highly versatile.\
\
With high-impact news clearly visible on the chart, traders can make better-informed decisions—choosing whether to participate in the market or stay on the sidelines. As a best practice, it's generally advised to avoid trading a few hours before and after major news releases to reduce risk and avoid volatility spikes\
\
You are welcome to share your thoughts or ask questions in the comments section. You can also find the attached files just below this article.\
\
| Filename | Description |\
| --- | --- |\
| NewsTicker.mq5 | Main Expert Advisor source implementing the three-lane scrolling economic calendar and news placeholder ticker directly on the chart using the CCanvas class, with per-lane speeds, inline importance boxes, and real-time countdowns. |\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/18299.zip "Download all attachments in the single ZIP archive")\
\
[News\_Headline\_EA.mq5](https://www.mql5.com/en/articles/download/18299/news_headline_ea.mq5 "Download News_Headline_EA.mq5")(21.47 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)\
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)\
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)\
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)\
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)\
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)\
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)\
\
**[Go to discussion](https://www.mql5.com/en/forum/489272)**\
\
![Mastering Log Records (Part 8): Error Records That Translate Themselves](https://c.mql5.com/2/148/18467-mastering-log-records-part-logo.png)[Mastering Log Records (Part 8): Error Records That Translate Themselves](https://www.mql5.com/en/articles/18467)\
\
In this eighth installment of the Mastering Log Records series, we explore the implementation of multilingual error messages in Logify, a powerful logging library for MQL5. You’ll learn how to structure errors with context, translate messages into multiple languages, and dynamically format logs by severity level. All of this with a clean, extensible, and production-ready design.\
\
![MQL5 Wizard Techniques you should know (Part 70):  Using Patterns of SAR and the RVI with a Exponential Kernel Network](https://c.mql5.com/2/150/18433-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 70): Using Patterns of SAR and the RVI with a Exponential Kernel Network](https://www.mql5.com/en/articles/18433)\
\
We follow up our last article, where we introduced the indicator pair of the SAR and the RVI, by considering how this indicator pairing could be extended with Machine Learning. SAR and RVI are a trend and momentum complimentary pairing. Our machine learning approach uses a convolution neural network that engages the Exponential kernel in sizing its kernels and channels, when fine-tuning the forecasts of this indicator pairing. As always, this is done in a custom signal class file that works with the MQL5 wizard to assemble an Expert Advisor.\
\
![Reimagining Classic Strategies (Part 13): Taking Our Crossover Strategy to New Dimensions (Part 2)](https://c.mql5.com/2/150/18525-reimagining-classic-strategies-logo.png)[Reimagining Classic Strategies (Part 13): Taking Our Crossover Strategy to New Dimensions (Part 2)](https://www.mql5.com/en/articles/18525)\
\
Join us in our discussion as we look for additional improvements to make to our moving-average cross over strategy to reduce the lag in our trading strategy to more reliable levels by leveraging our skills in data science. It is a well-studied fact that projecting your data to higher dimensions can at times improve the performance of your machine learning models. We will demonstrate what this practically means for you as a trader, and illustrate how you can weaponize this powerful principle using your MetaTrader 5 Terminal.\
\
![From Basic to Intermediate: Array (IV)](https://c.mql5.com/2/99/Do_bdsico_ao_intermedi9rio__Array_IV__LOGO.png)[From Basic to Intermediate: Array (IV)](https://www.mql5.com/en/articles/15501)\
\
In this article, we'll look at how you can do something very similar to what's implemented in languages like C, C++, and Java. I am talking about passing a virtually infinite number of parameters inside a function or procedure. While this may seem like a fairly advanced topic, in my opinion, what will be shown here can be easily implemented by anyone who has understood the previous concepts. Provided that they were really properly understood.\
\
[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=wodipqcbldqlucvekjjdmbuymxriauhr&ssn=1769191615318839407&ssn_dr=0&ssn_sr=0&fv_date=1769191615&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18299&back_ref=https%3A%2F%2Fwww.google.com%2F&title=From%20Novice%20to%20Expert%3A%20Animated%20News%20Headline%20Using%20MQL5%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919161567362276&fz_uniq=5071585116762876607&sv=2552)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
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
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)