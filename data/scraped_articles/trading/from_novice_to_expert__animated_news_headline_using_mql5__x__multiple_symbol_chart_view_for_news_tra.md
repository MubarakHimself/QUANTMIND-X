---
title: From Novice to Expert: Animated News Headline Using MQL5 (X)—Multiple Symbol Chart View for News Trading
url: https://www.mql5.com/en/articles/19299
categories: Trading, Integration
relevance_score: -2
scraped_at: 2026-01-24T14:14:41.195736
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=fpfhnefaaijytzzzdhtigieyxpwcvteg&ssn=1769253279077281180&ssn_dr=0&ssn_sr=0&fv_date=1769253279&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19299&back_ref=https%3A%2F%2Fwww.google.com%2F&title=From%20Novice%20to%20Expert%3A%20Animated%20News%20Headline%20Using%20MQL5%20(X)%E2%80%94Multiple%20Symbol%20Chart%20View%20for%20News%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925327951017778&fz_uniq=5083432852768431007&sv=2552)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/19299#para1)
- [Implementation Strategy](https://www.mql5.com/en/articles/19299#para2)
- [Testing and Results](https://www.mql5.com/en/articles/19299#para3)
- [Conclusion](https://www.mql5.com/en/articles/19299#para4)
- [Key Lessons](https://www.mql5.com/en/articles/19299#para5)
- [Attachments](https://www.mql5.com/en/articles/19299#para6)

### Introduction

In the [previous](https://www.mql5.com/en/articles/19008) article, we introduced a multiple-symbol management feature that allows traders to quickly select their preferred trading pairs. This reduced the time spent switching between charts just to configure a pair for trading.

However, a challenge remained: while traders could manage selected pairs, they still could not directly access chart APIs for those pairs, and thus lacked the ability to conveniently view their price for analysis. To address this, today we refine our multiple-symbol trading by developing a dedicated class for multi-chart visualization, seamlessly controlled within the chart and integrated as part of the broader feature set of the News Headline EA.

If at this point you feel lost, here’s a quick recap of our progress. I also encourage revisiting our [earlier publications](https://www.mql5.com/en/users/billionaire2024/publications) on this topic for a more in-depth understanding.

The earliest version of the News Headline EA focused on reproducing scheduled calendar events in clearly defined lanes on the chart, categorized by importance. The concept of “lanes” quickly expanded: we integrated a news feed API from [Alpha Vantage](https://www.mql5.com/en/articles/18465#para1), added locally hosted AI model insights, and created lanes for inbuilt indicator-based insights. Building on this, we introduced automated news trading features, including specialized functionality for NFP ( [Non-Farm Payrolls](https://www.mql5.com/en/articles/18817)) trading, combined with manual trading button integration.

Most recently, we advanced into multiple-symbol trading, designed for high-speed decision-making during volatile events. This feature made it significantly easier to manage and trade multiple pairs under fast market conditions.

Today, our focus is on refining this multiple-symbol trading approach by empowering traders with real-time multi-chart views directly inside the EA.

### Implementation Strategy

One of our key priorities is to deliver knowledge in a way that even a novice reader can easily grasp the concepts with clarity. In this section, let’s break down the approach and implementation we will apply to achieve that goal.

As our program grows, it is important to adopt a structured and maintainable approach to integrating new features. One of my favorite techniques is modularization, which ensures smooth development and allows us to create reusable utilities in the form of headers and classes.

For a sophisticated program like the News Headline EA, I follow a consistent workflow:

1. Start by developing a standalone mini program dedicated to testing the new feature.
2. Once the feature proves feasible and stable, proceed with integration into the main EA.

This process provides a focused workflow, reduces errors, and ensures that new features are seamlessly incorporated without breaking existing functionality.

Today, our design task is to create a CChartMiniTiles class, which will handle the display of multiple symbol charts within a single chart, each at customizable dimensions. We will then implement the class in a dummy EA (MiniChartsEA) to validate the concept. Once confirmed, the class will be integrated into the News Headline EA and adapted for smooth functionality.

Finally, note that names like CChartMiniTiles and MiniChartsEA are simply placeholders I chose for this walkthrough—you are free to use different names, as long as you understand how the program works.

In the next 4 subsections, we will focus on implementation:

1. ChartMiniTiles header
2. Example EA (MiniChartsEA) for testing the header
3. Initial testing
4. Integration of the ChartsMiniTiles into the News Headline EA

**1.0 ChartMiniTiles Header**

This header file defines the CChartMiniTiles class, a reusable utility designed to display and manage multiple mini-charts within a single MetaTrader 5 chart. The purpose of this class is to simplify integration of multi-symbol chart views into larger projects, such as the News Headline EA, while keeping the code modular and maintainable.

By isolating this functionality in a standalone header, we ensure that the feature can be tested independently (for example, with a dummy EA) and later integrated seamlessly into more complex systems. This approach improves workflow and reduces the likelihood of errors during development.

In the following sections, we will break down the code development process into numbered steps, with each step focusing on a specific aspect of the program’s structure and functionality.

1.1. Class Overview and Purpose

This opening block documents the header file’s purpose and defines the compile-time defaults used across the class. Using macros centralizes layout, timing, and toggle defaults so maintainers can quickly tune behavior without digging into implementation details. The comment header states the feature set—bottom-anchored mini-chart tiles created with [OBJ\_CHART](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_chart), automatic symbol resolution for different brokers, a toggle button for quick visibility control, responsive layout calculation, and a top-reserved area option to avoid conflicting with other UI elements. This section prepares the reader for how the class is organized and what it is intended to do.

```
//+------------------------------------------------------------------+
//| ChartMiniTilesClass.mqh                                          |
//| Class wrapper for ChartMiniTiles functionality                   |
//| - CChartMiniTiles class                                           |
//| - bottom-anchored mini-chart tiles using OBJ_CHART                |
//| - broker-adaptive symbol lookup, toggle button, responsive layout |
//| - supports top-reserved area to avoid overlapping top UI     |
//+------------------------------------------------------------------+
#ifndef __CHART_MINI_TILES_CLASS_MQH__
#define __CHART_MINI_TILES_CLASS_MQH__

//--- compile-time defaults (macros are safe in MQL5)
#define CMT_DEFAULT_WIDTH       120
#define CMT_DEFAULT_HEIGHT      112    // quadrupled default
#define CMT_DEFAULT_X_OFFSET    10
#define CMT_DEFAULT_Y_OFFSET    40     // bottom offset from bottom
#define CMT_DEFAULT_SPACING     6
#define CMT_DEFAULT_PERIOD      PERIOD_M1
#define CMT_DEFAULT_CHART_SCALE 2

// toggle button defaults
#define CMT_TOG_X               8
#define CMT_TOG_Y               6
#define CMT_TOG_W               72
#define CMT_TOG_H               20
#define CMT_TOG_NAME            "CMT_ToggleButton"
```

1.2. Constructor and Destructor

The constructor initializes the class to a known, safe state using the compile-time defaults. It prepares internal arrays, default tile sizes and offsets, and toggle button settings, and sets m\_top\_reserved to zero (no reserved area by default). The destructor calls Delete() to ensure any objects the class created are removed when the instance goes out of scope. This deterministic setup and teardown prevents leftover objects on the chart and reduces debugging headaches.

```
//+------------------------------------------------------------------+
//| CChartMiniTiles class declaration                                |
//+------------------------------------------------------------------+
class CChartMiniTiles
  {
public:
                     CChartMiniTiles(void);
                    ~CChartMiniTiles(void);

   bool              Init(const string majorSymbols,
                          const int width = -1,
                          const int height = -1,
                          const int xOffset = -1,
                          const int yOffset = -1,        // bottom offset (pixels from bottom)
                          const int spacing = -1,
                          const int period = -1,
                          const bool dateScale = true,
                          const bool priceScale = false,
                          const int chartScale = -1);

   void              UpdateLayout(void);
   void              Delete(void);

   bool              HandleEvent(const int id,const string sparam); // forward OnChartEvent
   void              SetToggleButtonPos(const int x,const int y);
   void              SetTilesVisible(const bool visible);
   void              Toggle(void);

   // NEW: reserve top area height (pixels from top) that tiles must NOT cover
   void              SetTopReservedHeight(const int pixels);

private:
   // state
   string            m_object_names[];   // object names created
   string            m_symbols[];        // resolved broker symbols
   int               m_count;

   int               m_width;
   int               m_height;
   int               m_xoffset;
   int               m_yoffset;         // bottom offset
   int               m_spacing;
   int               m_period;
   bool              m_date_scale;
   bool              m_price_scale;
   int               m_chart_scale;
   bool              m_visible;

   int               m_tog_x;
   int               m_tog_y;
   int               m_tog_w;
   int               m_tog_h;
   string            m_tog_name;

   // NEW
   int               m_top_reserved;    // pixels from top that must be left free for other UI

private:
   // helpers
   string            MakeObjectName(const string base);
   string            TrimString(const string s);
   string            FindBrokerSymbol(const string baseSymbol);
   int               ComputeBaseYFromTop(void);

   void              CreateToggleButton(void);
   void              DeleteToggleButton(void);
   void              CollapseAll(void);
  };

//+------------------------------------------------------------------+
//| Constructor / Destructor                                         |
//+------------------------------------------------------------------+
CChartMiniTiles::CChartMiniTiles(void)
  {
   m_count       = 0;
   ArrayResize(m_object_names,0);
   ArrayResize(m_symbols,0);

   m_width       = CMT_DEFAULT_WIDTH;
   m_height      = CMT_DEFAULT_HEIGHT;
   m_xoffset     = CMT_DEFAULT_X_OFFSET;
   m_yoffset     = CMT_DEFAULT_Y_OFFSET;
   m_spacing     = CMT_DEFAULT_SPACING;
   m_period      = CMT_DEFAULT_PERIOD;
   m_date_scale  = false;
   m_price_scale = false;
   m_chart_scale = CMT_DEFAULT_CHART_SCALE;
   m_visible     = true;

   m_tog_x = CMT_TOG_X;
   m_tog_y = CMT_TOG_Y;
   m_tog_w = CMT_TOG_W;
   m_tog_h = CMT_TOG_H;
   m_tog_name = CMT_TOG_NAME;

   m_top_reserved = 0; // default: no reserved area
  }

CChartMiniTiles::~CChartMiniTiles(void)
  {
   Delete();
  }
```

1.3. Helper Methods

These focused helper functions normalize inputs and hide broker quirks so the rest of the class can be simpler. MakeObjectName sanitizes strings into safe object names (replacing spaces and special characters). TrimString removes leading/trailing whitespace, including tabs and newlines. FindBrokerSymbol attempts an exact SymbolSelect and, failing that, performs a case-insensitive search across the broker’s symbol list to find a match—this is crucial for portability across brokers that append suffixes or use different naming conventions. ComputeBaseYFromTop determines the vertical baseline for bottom-anchored tiles while guarding against invalid chart height values.

```
//+------------------------------------------------------------------+
//| Helpers implementation                                           |
//+------------------------------------------------------------------+
string CChartMiniTiles::MakeObjectName(const string base)
  {
   string name = base;
   StringReplace(name, " ", "_");
   StringReplace(name, ".", "_");
   StringReplace(name, ":", "_");
   StringReplace(name, "/", "_");
   StringReplace(name, "\\", "_");
   return(name);
  }

string CChartMiniTiles::TrimString(const string s)
  {
   if(s == NULL) return("");
   int len = StringLen(s);
   if(len == 0) return("");
   int left = 0;
   int right = len - 1;
   while(left <= right)
     {
      int ch = StringGetCharacter(s, left);
      if(ch == 32 || ch == 9 || ch == 10 || ch == 13) left++;
      else break;
     }
   while(right >= left)
     {
      int ch = StringGetCharacter(s, right);
      if(ch == 32 || ch == 9 || ch == 10 || ch == 13) right--;
      else break;
     }
   if(left > right) return("");
   return StringSubstr(s, left, right - left + 1);
  }

string CChartMiniTiles::FindBrokerSymbol(const string baseSymbol)
  {
   if(StringLen(baseSymbol) == 0) return("");
   if(SymbolSelect(baseSymbol, true))
      return(baseSymbol);

   string baseUpper = baseSymbol; StringToUpper(baseUpper);

   int total = SymbolsTotal(true);
   for(int i = 0; i < total; i++)
     {
      string s = SymbolName(i, true);
      if(StringLen(s) == 0) continue;
      string sUpper = s; StringToUpper(sUpper);
      if(StringFind(sUpper, baseUpper) >= 0)
         return(s);
     }
   return("");
  }

int CChartMiniTiles::ComputeBaseYFromTop(void)
  {
   int chartTotalHeight = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
   if(chartTotalHeight <= 0) chartTotalHeight = 600;
   int base_y_from_top = chartTotalHeight - m_yoffset - m_height;
   if(base_y_from_top < 0) base_y_from_top = 0;
   if(base_y_from_top + m_height > chartTotalHeight) base_y_from_top = MathMax(0, chartTotalHeight - m_height);
   return base_y_from_top;
  }
```

1.4. Toggle Button Control

Interactive control is provided through a toggle button. CreateToggleButton ensures a button object exists (creates it if missing) and sets the visual properties: position, size, font, background color, text color, selectability, and the initial state text determined by m\_visible. DeleteToggleButton removes it cleanly. CollapseAll is the lightweight hide method that effectively hides tiles by moving them off-screen and setting their sizes to zero rather than deleting them—this preserves state for fast re-showing and avoids the overhead of recreating objects.

```
//+------------------------------------------------------------------+
//| Toggle button helpers                                            |
//+------------------------------------------------------------------+
void CChartMiniTiles::CreateToggleButton(void)
  {
   if(ObjectFind(ChartID(), m_tog_name) == -1)
      ObjectCreate(ChartID(), m_tog_name, OBJ_BUTTON, 0, 0, 0, 0, 0);

   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_XDISTANCE, m_tog_x);
   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_YDISTANCE, m_tog_y);
   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_XSIZE, m_tog_w);
   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_YSIZE, m_tog_h);

   ObjectSetString(ChartID(), m_tog_name, OBJPROP_FONT, "Arial");
   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_FONTSIZE, 10);
   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_BGCOLOR, clrDarkSlateGray);
   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_SELECTABLE, 1);
   ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_STATE, m_visible ? 1 : 0);
   ObjectSetString(ChartID(), m_tog_name, OBJPROP_TEXT, m_visible ? "Tiles: ON" : "Tiles: OFF");
  }

void CChartMiniTiles::DeleteToggleButton(void)
  {
   if(ObjectFind(ChartID(), m_tog_name) >= 0)
      ObjectDelete(ChartID(), m_tog_name);
  }

void CChartMiniTiles::CollapseAll(void)
  {
   for(int i = 0; i < ArraySize(m_object_names); i++)
     {
      string name = m_object_names[i];
      if(StringLen(name) == 0) continue;
      if(ObjectFind(ChartID(), name) == -1) continue;
      ObjectSetInteger(ChartID(), name, OBJPROP_XDISTANCE, -1);
      ObjectSetInteger(ChartID(), name, OBJPROP_YDISTANCE, -1);
      ObjectSetInteger(ChartID(), name, OBJPROP_XSIZE, 0);
      ObjectSetInteger(ChartID(), name, OBJPROP_YSIZE, 0);
     }
  }
```

1.5. Initialization of Mini-Charts

Init is the central setup routine: it first clears any prior state, applies passed parameters falling back to defaults, parses the comma-separated majorSymbols list, trims and resolves each to a broker-specific symbol via FindBrokerSymbol, and builds the m\_symbols array. It computes layout constraints from chart width and tile dimensions, then enforces a row-limit to avoid intruding into m\_top\_reserved (the reserved top area). For each resolved symbol, it creates an OBJ\_CHART, sets properties (symbol, period, distances, sizes, date/price scales, and chart scale), and stores the created object name for later updates or deletion. After creation, it makes the toggle button, respects the initial visibility flag, redraws the chart, and returns success. This method includes logging for symbol resolution and created objects so runtime behavior is transparent.

```
//+------------------------------------------------------------------+
//| NEW: set top reserved height (pixels from top)                   |
//+------------------------------------------------------------------+
void CChartMiniTiles::SetTopReservedHeight(const int pixels)
  {
   m_top_reserved = MathMax(0, pixels);
  }

//+------------------------------------------------------------------+
//| Init: create mini tiles                                           |
//+------------------------------------------------------------------+
bool CChartMiniTiles::Init(const string majorSymbols,
                           const int width,
                           const int height,
                           const int xOffset,
                           const int yOffset,
                           const int spacing,
                           const int period,
                           const bool dateScale,
                           const bool priceScale,
                           const int chartScale)
  {
   Delete();

   m_width       = (width  <= 0) ? CMT_DEFAULT_WIDTH  : width;
   m_height      = (height <= 0) ? CMT_DEFAULT_HEIGHT : height;
   m_xoffset     = (xOffset <= 0) ? CMT_DEFAULT_X_OFFSET : xOffset;
   m_yoffset     = (yOffset <= 0) ? CMT_DEFAULT_Y_OFFSET : yOffset;
   m_spacing     = (spacing <= 0) ? CMT_DEFAULT_SPACING : spacing;
   m_period      = (period <= 0) ? CMT_DEFAULT_PERIOD : period;
   m_date_scale  = dateScale;
   m_price_scale = priceScale;
   m_chart_scale = (chartScale <= 0) ? CMT_DEFAULT_CHART_SCALE : chartScale;

   ArrayFree(m_object_names);
   ArrayFree(m_symbols);
   m_count = 0;

   string raw[]; StringSplit(majorSymbols, ',', raw);
   int rawCount = ArraySize(raw);
   if(rawCount == 0) return(false);

   for(int i = 0; i < rawCount; i++)
     {
      string base = TrimString(raw[i]);
      if(StringLen(base) == 0) continue;
      string resolved = FindBrokerSymbol(base);
      if(resolved == "")
        {
         PrintFormat("CMT: symbol not found on this broker: %s", base);
         continue;
        }
      int n = ArraySize(m_symbols);
      ArrayResize(m_symbols, n + 1);
      m_symbols[n] = resolved;
     }

   m_count = ArraySize(m_symbols);
   PrintFormat("CMT: %d symbols resolved for mini-tiles.", m_count);
   for(int i=0;i<m_count;i++) PrintFormat("CMT: symbol[%d] = %s", i, m_symbols[i]);

   if(m_count == 0) return(false);

   ArrayResize(m_object_names, m_count);
   for(int i = 0; i < m_count; i++) m_object_names[i] = "";

   int base_y_from_top = ComputeBaseYFromTop();

   int chartW = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   if(chartW <= 0) chartW = 800;
   int columns = MathMax(1, chartW / (m_width + m_spacing));

   // --- NEW: limit rows to avoid top reserved region
   int rows = (m_count + columns - 1) / columns;
   int availableAbove = MathMax(0, base_y_from_top - m_top_reserved);
   int maxRowsAllowed = 1 + (availableAbove / (m_height + m_spacing)); // bottom row + how many rows can fit above
   if(maxRowsAllowed < 1) maxRowsAllowed = 1;
   if(rows > maxRowsAllowed)
     {
      // increase columns to fit within allowed rows
      columns = (m_count + maxRowsAllowed - 1) / maxRowsAllowed;
      if(columns < 1) columns = 1;
      rows = (m_count + columns - 1) / columns;
     }
   // ---

   int createdCount = 0;
   for(int i = 0; i < m_count; i++)
     {
      string sym = m_symbols[i];
      string objName = MakeObjectName("CMT_" + sym + "_" + IntegerToString(i));
      m_object_names[i] = objName;

      int col = i % columns;
      int row = i / columns;

      int xdist = m_xoffset + col * (m_width + m_spacing);
      int ydist = base_y_from_top - row * (m_height + m_spacing);
      if(ydist < 0) ydist = 0;

      bool created = ObjectCreate(ChartID(), objName, OBJ_CHART, 0, 0, 0, 0, 0);
      if(!created)
        {
         PrintFormat("CMT: failed to create OBJ_CHART for %s (obj=%s)", sym, objName);
         m_object_names[i] = "";
         continue;
        }

      ObjectSetString(ChartID(), objName, OBJPROP_SYMBOL, sym);
      ObjectSetInteger(ChartID(), objName, OBJPROP_PERIOD, m_period);
      ObjectSetInteger(ChartID(), objName, OBJPROP_XDISTANCE, xdist);
      ObjectSetInteger(ChartID(), objName, OBJPROP_YDISTANCE, ydist);
      ObjectSetInteger(ChartID(), objName, OBJPROP_XSIZE, m_width);
      ObjectSetInteger(ChartID(), objName, OBJPROP_YSIZE, m_height);
      ObjectSetInteger(ChartID(), objName, OBJPROP_DATE_SCALE, (int)m_date_scale);
      ObjectSetInteger(ChartID(), objName, OBJPROP_PRICE_SCALE, (int)m_price_scale);
      ObjectSetInteger(ChartID(), objName, OBJPROP_SELECTABLE, 1);
      ObjectSetInteger(ChartID(), objName, OBJPROP_CHART_SCALE, m_chart_scale);

      createdCount++;
     }

   PrintFormat("CMT: created %d / %d mini-chart objects.", createdCount, m_count);

   CreateToggleButton();

   if(!m_visible)
     SetTilesVisible(false);

   ChartRedraw();
   return(true);
  }
```

1.6. Layout Updates

UpdateLayout recalculates positions and sizes when chart geometry or visibility changes. It first handles the hidden state by collapsing tiles and updating the toggle button to “OFF.” When visible, it recomputes columns from chart width, enforces the top-reserved constraint (so rows never cross into reserved space), and updates each OBJ\_CHART with the appropriate OBJPROP\_XDISTANCE, OBJPROP\_YDISTANCE, sizes, and chart scale. Finally, it updates the toggle button state to “ON” and calls ChartRedraw() so the UI refreshes. This method is intended to be called whenever the chart resizes or when the EA wants to refresh the layout.

```
//+------------------------------------------------------------------+
//| UpdateLayout - reposition tiles (respects m_visible and top reserved)|
//+------------------------------------------------------------------+
void CChartMiniTiles::UpdateLayout(void)
  {
   if(!m_visible)
     {
      CollapseAll();
      if(ObjectFind(ChartID(), m_tog_name) >= 0)
        {
         ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_XDISTANCE, m_tog_x);
         ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_YDISTANCE, m_tog_y);
         ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_STATE, 0);
         ObjectSetString(ChartID(), m_tog_name, OBJPROP_TEXT, "Tiles: OFF");
        }
      ChartRedraw();
      return;
     }

   if(m_count == 0) return;

   int chartW = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   if(chartW <= 0) chartW = 800;
   int chartTotalHeight = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
   if(chartTotalHeight <= 0) chartTotalHeight = 600;

   int columns = MathMax(1, chartW / (m_width + m_spacing));

   int base_y_from_top = ComputeBaseYFromTop();

   // --- NEW: ensure rows don't surpass top-reserved area
   int rows = (m_count + columns - 1) / columns;
   int availableAbove = MathMax(0, base_y_from_top - m_top_reserved);
   int maxRowsAllowed = 1 + (availableAbove / (m_height + m_spacing));
   if(maxRowsAllowed < 1) maxRowsAllowed = 1;
   if(rows > maxRowsAllowed)
     {
      columns = (m_count + maxRowsAllowed - 1) / maxRowsAllowed;
      if(columns < 1) columns = 1;
      rows = (m_count + columns - 1) / columns;
     }
   // ---

   for(int i = 0; i < m_count; i++)
     {
      string name = m_object_names[i];
      if(StringLen(name) == 0) continue;
      if(ObjectFind(ChartID(), name) == -1) continue;

      int col = i % columns;
      int row = i / columns;
      int xdist = m_xoffset + col * (m_width + m_spacing);
      int ydist = base_y_from_top - row * (m_height + m_spacing);
      if(ydist < 0) ydist = 0;

      ObjectSetInteger(ChartID(), name, OBJPROP_XDISTANCE, xdist);
      ObjectSetInteger(ChartID(), name, OBJPROP_YDISTANCE, ydist);
      ObjectSetInteger(ChartID(), name, OBJPROP_XSIZE, m_width);
      ObjectSetInteger(ChartID(), name, OBJPROP_YSIZE, m_height);
      ObjectSetInteger(ChartID(), name, OBJPROP_CHART_SCALE, m_chart_scale);
     }

   if(ObjectFind(ChartID(), m_tog_name) >= 0)
     {
      ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_XDISTANCE, m_tog_x);
      ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_YDISTANCE, m_tog_y);
      ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_STATE, 1);
      ObjectSetString(ChartID(), m_tog_name, OBJPROP_TEXT, "Tiles: ON");
     }

   ChartRedraw();
  }
```

1.7. Visibility and Toggling

This small set of methods controls the visible state of tiles. SetTilesVisible updates the m\_visible flag and either repositions tiles or collapses them depending on the new state; it also updates the toggle button’s state and text. Toggle is a convenience wrapper that flips visibility and calls SetTilesVisible. SetToggleButtonPos allows dynamic repositioning of the toggle button; if the button already exists, it updates its OBJPROP\_XDISTANCE and OBJPROP\_YDISTANCE. These methods are the programmatic and UI entry points for showing, hiding, and repositioning the tile controls.

```
//+------------------------------------------------------------------+
//| Set visibility programmatically                                  |
//+------------------------------------------------------------------+
void CChartMiniTiles::SetTilesVisible(const bool visible)
  {
   m_visible = visible;
   if(m_count == 0)
     {
      if(ObjectFind(ChartID(), m_tog_name) >= 0)
        {
         ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_STATE, m_visible ? 1 : 0);
         ObjectSetString(ChartID(), m_tog_name, OBJPROP_TEXT, m_visible ? "Tiles: ON" : "Tiles: OFF");
         ChartRedraw();
        }
      return;
     }

   if(m_visible)
     UpdateLayout();
   else
     {
      CollapseAll();
      ChartRedraw();
     }

   if(ObjectFind(ChartID(), m_tog_name) >= 0)
     {
      ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_STATE, m_visible ? 1 : 0);
      ObjectSetString(ChartID(), m_tog_name, OBJPROP_TEXT, m_visible ? "Tiles: ON" : "Tiles: OFF");
     }
  }

//+------------------------------------------------------------------+
//| Toggle                                                           |
//+------------------------------------------------------------------+
void CChartMiniTiles::Toggle(void)
  {
   SetTilesVisible(!m_visible);
  }

//+------------------------------------------------------------------+
//| Set toggle position                                               |
//+------------------------------------------------------------------+
void CChartMiniTiles::SetToggleButtonPos(const int x,const int y)
  {
   m_tog_x = x;
   m_tog_y = y;
   if(ObjectFind(ChartID(), m_tog_name) >= 0)
     {
      ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_XDISTANCE, m_tog_x);
      ObjectSetInteger(ChartID(), m_tog_name, OBJPROP_YDISTANCE, m_tog_y);
     }
  }
```

1.8. Cleanup and Event Handling

Finally, Delete performs a thorough cleanup: it iterates the object name array and deletes each created OBJ\_CHART, removes the toggle button, frees the arrays with ArrayFree, resets m\_count, and redraws the chart. HandleEvent is the event-forwarding method intended to be called from an EA’s OnChartEvent; it filters for CHARTEVENT\_OBJECT\_CLICK and for the toggle button name—if the toggle was clicked, it calls Toggle() and returns true to indicate the event was handled. This keeps EA integration simple: forward events and call Init/Delete in OnInit/OnDeinit.

```
//+------------------------------------------------------------------+
//| Delete all objects                                                |
//+------------------------------------------------------------------+
void CChartMiniTiles::Delete(void)
  {
   for(int i = 0; i < ArraySize(m_object_names); i++)
     {
      string name = m_object_names[i];
      if(StringLen(name) == 0) continue;
      if(ObjectFind(ChartID(), name) >= 0) ObjectDelete(ChartID(), name);
     }
   DeleteToggleButton();
   ArrayFree(m_object_names);
   ArrayFree(m_symbols);
   m_count = 0;
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| HandleEvent - forward OnChartEvent to class (returns true if handled)|
//+------------------------------------------------------------------+
bool CChartMiniTiles::HandleEvent(const int id,const string sparam)
  {
   if(id != CHARTEVENT_OBJECT_CLICK) return(false);
   if(StringLen(sparam) == 0) return(false);
   if(sparam == m_tog_name)
     {
      Toggle();
      return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
#endif // __CHART_MINI_TILES_CLASS_MQH__
```

You will find the complete source code for this header at the end of this discussion, along with the other files referenced throughout.

**2.0 Example EA (MiniChartsEA) for testing the header**

2.1. EA Overview and Purpose

This EA is a testing ground for the CChartMiniTiles class. It instantiates the class, initializes mini-charts for multiple symbols, and validates toggling, resizing, and updating functionality before integration into larger projects.

```
//+------------------------------------------------------------------+
//|                                                  MiniChartsEA.mq5|
//|          Dummy EA to test CChartMiniTiles (ChartMiniTilesClass)  |
//+------------------------------------------------------------------+
#property copyright "2025"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.00"
#property description "Mini-charts EA using CChartMiniTiles class"

// --- Include the class header
#include <ChartMiniTiles.mqh>
```

2.2. Inputs and Global Variables

This section defines parameters and state variables that control the mini-charts (symbol list, dimensions, update behavior).

```
//--- Inputs
input string MajorSymbols      = "EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD";
input int    BarsWidth         = 20;   // bars used to estimate tile pixel width
input int    TileHeightPx      = 112;  // tile height in pixels
input int    HorizontalSpacing = 6;    // spacing between tiles
input int    UpdateInterval    = 1000; // ms timer update interval
input int    XOffset           = 10;   // left margin in pixels
input int    BottomOffset      = 40;   // distance from bottom in pixels
input int    ToggleButtonX     = 8;    // toggle button X
input int    ToggleButtonY     = 6;    // toggle button Y
input bool   DateScale         = false;// show date scale
input bool   PriceScale        = false;// show price scale
input int    ChartScale        = 2;    // chart zoom level

//--- object instance of our class
CChartMiniTiles tiles;

//--- internal state
int pixelWidth = 120;
```

2.3. Helper Methods

This method dynamically estimates tile width in pixels based on how many bars are currently visible.

```
//+------------------------------------------------------------------+
//| Helper: estimate pixel width from BarsWidth                      |
//+------------------------------------------------------------------+
int CalculateChartWidthFromBars(int barsWidth)
{
   int mainChartWidth = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int visibleBars    = (int)ChartGetInteger(0, CHART_VISIBLE_BARS);

   if(visibleBars <= 0 || mainChartWidth <= 0)
      return MathMax(80, BarsWidth * 6); // fallback

   return MathMax(80, barsWidth * mainChartWidth / visibleBars);
}
```

2.4. Toggle Button Control

Handled by the tiles object (CChartMiniTiles). The EA itself only sets the initial toggle button position.

```
// Inside OnInit we call:
tiles.SetToggleButtonPos(ToggleButtonX, ToggleButtonY);
```

2.5. Initialization of Mini-Charts

This stage creates all mini-charts with user-defined settings.

```
//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // compute pixel width from BarsWidth heuristic
   pixelWidth = CalculateChartWidthFromBars(BarsWidth);

   // set toggle button position
   tiles.SetToggleButtonPos(ToggleButtonX, ToggleButtonY);

   // initialize tiles
   bool ok = tiles.Init(MajorSymbols,
                        pixelWidth,
                        TileHeightPx,
                        XOffset,
                        BottomOffset,
                        HorizontalSpacing,
                        PERIOD_M1,
                        DateScale,
                        PriceScale,
                        ChartScale);

   if(!ok)
   {
      Print("MiniChartsEA: tiles.Init() failed. Check Experts log for symbol issues.");
      return(INIT_FAILED);
   }

   // start timer for adaptive updates
   EventSetMillisecondTimer(UpdateInterval);

   Print("MiniChartsEA initialized.");
   return(INIT_SUCCEEDED);
}
```

2.6. Layout Updates

This ensures tiles resize and reposition correctly when the chart changes or time progresses.

```
//+------------------------------------------------------------------+
//| Timer: update layout                                             |
//+------------------------------------------------------------------+
void OnTimer()
{
   tiles.UpdateLayout();
}
```

2.7. Visibility and Toggling

Here, the EA delegates toggle button clicks to the class, and listens for chart changes to keep everything aligned.

```
//+------------------------------------------------------------------+
//| Chart events - forward to tiles                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   // Let tiles handle toggle button clicks
   if(tiles.HandleEvent(id, sparam))
      return;

   // If chart resized or layout changed, reflow tiles
   if(id == CHARTEVENT_CHART_CHANGE)
      tiles.UpdateLayout();
}
```

2.8. Cleanup and Event Handling

This ensures a clean removal of objects when the EA stops.

```
//+------------------------------------------------------------------+
//| Deinitialization                                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   tiles.Delete();
   Print("MiniChartsEA deinitialized.");
}
```

**3.0 Initial testing**

At this stage, we add our example EA to the chart and observe the results. In my test run, the chart displayed tiles for the major pairs I selected, each adjusted to my broker’s symbol format (with a .0 suffix). The toggle button worked flawlessly, allowing me to switch the tiles on and off with ease. This feature allows traders the flexibility to view the main chart fully when needed—a simple yet powerful control.

Such functionality will become even more valuable once the class is integrated into the News Headline EA, where multiple components will create a more complex chart environment. Having the ability to quickly toggle the mini-tiles will ensure a cleaner, more manageable workspace.

See the image below for the deployment and output of the MiniChartsEA.

![Testing the MiniChatTilesEA](https://c.mql5.com/2/167/terminal64_f27zEwfh86.gif)

Figure 1: Testing results of the MiniChartsEA on the EURUSD chart.

Now we can proceed with integrating the CChartMiniTiles class into the News Headline EA. In the next section, we will explore the updated code in detail and then review the resulting behavior on the chart.

**4.0 Integration of the ChartsMiniTiles into the NewsHeadlineEA**

4.1. Include the ChartMiniTiles header—make the class available to the EA

To use the mini-tiles class you must include its header near the top of the EA. This brings the class declaration and implementation into the EA’s compilation unit so you can create an instance and call its methods. Placing the include with the other headers (TradingButtons, Canvas, Trade) keeps imports organized and signals the dependency to future maintainers. If the file is missing from MQL5/Include/ the compiler will complain here, so this is the very first integration step.

```
#include <TradingButtons.mqh>
#include <Trade\Trade.mqh>
#include <Canvas\Canvas.mqh>
#include <ChartMiniTiles.mqh>   // <-- CTM class include (make sure this file is in MQL5/Include/)
```

4.2. Declare the CChartMiniTiles instance—create the EA’s tile manager

Declaring a global CChartMiniTiles tiles gives the entire EA access to a single tiles manager. This instance holds state (created object names, symbol list, visibility flag, reserved top height) and exposes methods for initialization, layout updates, event handling, and cleanup. Declaring it among other globals makes its lifecycle easy to manage in OnInit / OnTimer / OnChartEvent / OnDeinit.

```
//+------------------------------------------------------------------+
//| ChartMiniTiles instance (CTM)                                     |
//+------------------------------------------------------------------+
CChartMiniTiles tiles;   // class instance for mini tiles
```

4.3. Build the symbol list and estimate tile pixel width—prepare CMT inputs

Before calling tiles.Init you need a comma-separated symbol string and a reasonable pixel width for tiles. This section constructs ctmSymbols from your majorPairs\[\] array using JoinSymbolArray() and computes pixelWidth using a bars-to-pixels heuristic based on the main chart’s width and visible bars. Doing this at initialization ensures tile sizes are adaptive to the current chart layout and the broker’s symbol naming will be resolved by FindBrokerSymbol inside the class.

```
   // Build a comma-separated symbol list from majorPairs[] and initialize CTM.
   string ctmSymbols = JoinSymbolArray(majorPairs);
   // Estimate pixel width from Bars heuristic (simple fallback)
   int mainChartWidth = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int visibleBars    = (int)ChartGetInteger(0, CHART_VISIBLE_BARS);
   int pixelWidth = 120;
   if(visibleBars > 0 && mainChartWidth > 0)
      pixelWidth = MathMax(80, CTM_TileWidthBars * mainChartWidth / visibleBars);
```

4.4. Place the toggle button and set its position—keep UI accessible.

The toggle button is the user-facing control to show/hide tiles. Positioning it intentionally just below the trading panel ensures it doesn’t overlap important UI controls. After computing tradingPanelBottomY, the EA calls tiles.SetToggleButtonPos(toggleX, toggleY) to instruct the class where to create the button. This placement is done before initializing tiles so the button exists and matches the reserved area logic that follows.

```
   // Place the CTM toggle button JUST BELOW the trading panel bottom
   int toggleX = 8;
   int toggleY = tradingPanelBottomY + 6; // +6 px margin so it doesn't touch trading controls
   tiles.SetToggleButtonPos(toggleX, toggleY);
```

4.5. Reserve top UI space for the trading controls—prevent overlap.

To avoid mini-tiles overlapping the trading panel and toggle button, the EA computes a topReserve value and calls tiles.SetTopReservedHeight (topReserve). The tiles class uses this value to limit how many rows can stack upward from the bottom; this preserves the top area for buttons, canvases, or other UI elements. Reserving the top space before calling Init guarantees layout calculations respect it immediately.

```
   // Reserve the area above tiles so the trading UI remains free. We reserve up to the toggle bottom.
   int topReserve = toggleY + CMT_TOG_H + 4; // leave a few px extra
   tiles.SetTopReservedHeight(topReserve);
```

4.6. Initialize the tiles—create OBJ\_CHART objects for each symbol

This is the core integration call. The EA passes the resolved symbol list, computed pixel width, tile height, offsets, spacing, chart period, and scale into tiles.Init(...). The class will resolve broker-specific names, create OBJ\_CHART objects, set their properties (symbol, period, scales, sizes), and create the toggle button. The EA checks the boolean return (ctm\_ok) to handle initialization failure gracefully (e.g., if no symbols were resolved on this broker).

```
   // Initialize tiles: (symbols, widthPx, heightPx, xOffset, bottomOffset, spacing, period, dateScale, priceScale, chartScale)
   bool ctm_ok = tiles.Init(ctmSymbols, pixelWidth, CTM_TileHeightPx, CTM_XOffset, CTM_BottomOffset, CTM_Spacing, PERIOD_M1, false, false, CTM_ChartScale);
   if(!ctm_ok)
   {
      Print("CTM: initialization failed (no matching symbols?); tiles disabled.");
   }
```

4.7. Keep layout responsive—call UpdateLayout when chart changes or on timer

Once initialized, the EA must keep tiles in the right place as chart size or layout changes. The integration covers three places:

- Forwarding chart-change events to reposition tiles (so user-resize or workspace changes are handled).
- Calling tiles.UpdateLayout() periodically in the OnTimer loop to handle dynamic UI shifts or to recover from external changes.
- Calling tiles.UpdateLayout() at the end of your main OnTimer processing loop so the CTM always refreshes after other UI updates (canvases, news, AI insights).

Below are the exact snippets used in your EA to handle these cases.

OnChartEvent forwarding & chart-change reaction:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   // Let CTM handle object clicks first (toggle button)
   if(tiles.HandleEvent(id, sparam))
      return;

   // Forward to the TradingButtons header afterward
   buttonsEA.HandleChartEvent(id, sparam, majorPairs, pairSelected);

   // Also respond to chart change events for CTM layout
   if(id == CHARTEVENT_CHART_CHANGE)
      tiles.UpdateLayout();
}
```

End-of-OnTimer update (keeps CTM in sync every timer tick):

```
   // Keep CTM updated every tick (timer-driven)
   tiles.UpdateLayout();
}
```

4.8. Cleanup at deinitialization—delete tiles and toggle button

When the EA stops or is removed, it must remove any objects it created. Calling tiles.Delete() removes every OBJ\_CHART and the toggle button, frees arrays and redraws the chart. This is the final integration step and prevents orphaned objects lingering on the chart.

```
   // delete CTM tiles and toggle button
   tiles.Delete();
```

Quick integration checklist (summary)

1. #include <ChartMiniTiles.mqh> at top.
2. Declare CChartMiniTiles tiles; globally.
3. Build ctmSymbols and compute pixelWidth before Init.
4. Set toggle position via tiles.SetToggleButtonPos(...).
5. Reserve top UI area via tiles.SetTopReservedHeight(...).
6. Initialize with tiles.Init(...) and check return.
7. Forward OnChartEvent to tiles.HandleEvent(...) and call tiles.UpdateLayout() on chart-change and in periodic updates.
8. Call tiles.Delete() in OnDeinit.

### Testing and Results

This marks the final testing phase of the integrated News Headline EA. After compiling the error-free program in MetaEditor 5, we deploy it onto a MetaTrader 5 terminal chart to observe its behavior. The screenshots below present the results of my test run.

![](https://c.mql5.com/2/167/terminal64_tK5aORM8iN.gif)

Figure 2: Deploying the News Headline EA on the AUDUSD chart featuring Mini Chart Tiles for multiple symbols trading

![Multiple Feature News Headline EA](https://c.mql5.com/2/167/terminal64_uCpkFA3obW.gif)

Figure 3: ChartMiniTiles working in News Headline EA

Reviewing the results above, we successfully integrated our solution to the identified problem. Traders can now view multiple pairs within a single chart, and the control buttons function as expected. Another notable achievement is that we implemented this new feature without overlapping or interfering with existing functionality in the News Headline EA, thanks to proper layout management and the capabilities of the MQL5 language. In the next section, we will conclude by highlighting the key accomplishments of this work.

### Conclusion

We have successfully integrated a class that manages multiple mini chart views on a single main chart using the New Headline EA. This approach provides traders with a powerful tool to monitor and trade multiple symbols simultaneously, improving efficiency—particularly during periods of high volatility, such as major news releases. When multiple positions are opened across different pairs, this tool allows traders to manage and track them from a single interface, enabling rapid control and execution of trades.

From an educational perspective, this discussion offers valuable insights, including custom class development, multi-symbol trading strategies, fast order execution, and pair selection—all without leaving the current chart.

While this demonstrates the feasibility of multi-pair trading, there remains considerable room for improvement. In particular, refining techniques that allow more flexible analysis of selected pairs is essential. With further development, the algorithm could incorporate automatic analysis methods to enhance the current system. Future publications will address these enhancements.

We invite your ideas and contributions to this discussion. Attached below are the source code and an image showing the main chart alongside the ChartMiniTiles setup.

### Key Lessons

| Key Lesson | Description: |
| --- | --- |
| Multiple-Symbol Management | Traders can efficiently select and monitor multiple trading pairs without switching charts repeatedly. |
| Dedicated Multi-Chart Class | Using a standalone class to manage multiple mini charts simplifies integration and maintains modular code. |
| Standalone Testing Before Integration | Developing and validating a feature in a separate test EA ensures stability before merging into the main EA. |
| Customizable Mini Chart Layouts | Mini charts can be resized and arranged according to user preferences, improving visibility and workflow. |
| Real-Time Multi-Symbol Visualization | Displaying multiple symbols on a single chart allows traders to react faster during volatile market conditions. |
| Modular Programming | Separating functionality into headers and classes promotes maintainability and reusability across projects. |
| Incremental Feature Development | Adding new features step by step reduces errors and ensures smooth integration with existing systems. |
| Event Handling in Mini Charts | Each mini chart can respond to user interactions, enabling interactive and dynamic analysis. |
| Integration with Main EA | After testing, the mini chart class is merged into the main EA to provide seamless multi-symbol trading functionality. |
| Reusable Utility Classes | Classes like CChartMiniTiles serve as modular tools that can be adapted for other projects and EAs. |
| Efficient Workflow Practices | Using a structured workflow—prototype, test, integrate—improves development speed and reduces bugs. |
| Rapid Decision Support | Mini charts give traders a consolidated view of multiple markets, supporting quicker trading decisions. |
| Dynamic Object Management | The code demonstrates how to create, update, and destroy chart objects dynamically for each mini chart, essential for scalable EA design. |
| Efficient Memory Usage | By managing mini charts as separate objects and only updating visible elements, the EA optimizes memory and CPU usage. |
| Seamless Event Propagation | The mini chart class shows how to handle chart events in a hierarchical manner, ensuring events reach the correct chart instance without conflict. |

### Attachments

| Filename | Version | Description |
| --- | --- | --- |
| ChartMiniTiles.mqh | 1.0 | Defines the CChartMiniTiles class, responsible for managing and displaying multiple mini-chart views within a single main chart. Provides modular multi-symbol visualization and interaction capabilities. |
| MiniChartsEA.mq5 | 1.0 | Dummy EA created for testing the CChartMiniTiles class independently before integration into the main News Headline EA. Validates layout, resizing, and event handling of mini charts. |
| NewsHeadlineEA.mq5 | 1.14 | Main Expert Advisor that integrates multiple features: news lane visualization, automated trading based on calendar events, and multi-symbol trading with mini charts. |
| TradingButtons.mqh | 1.0 | Provides button controls for executing trades, selecting pairs, and managing orders directly from the chart. Supports fast multi-symbol trading interactions in the EA. |
| terminal64\_Dp0JGQhX5.png | N/A | A wide terminal chart screenshot illustrating the ChartMiniTiles feature of the News Headline EA, showcasing multi-symbol trading, real-time monitoring, and quick visual analysis. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19299.zip "Download all attachments in the single ZIP archive")

[ChartMiniTiles.mqh](https://www.mql5.com/en/articles/download/19299/ChartMiniTiles.mqh "Download ChartMiniTiles.mqh")(38.64 KB)

[MiniChartsEA.mq5](https://www.mql5.com/en/articles/download/19299/MiniChartsEA.mq5 "Download MiniChartsEA.mq5")(9.85 KB)

[NewsHeadlineEA.mq5](https://www.mql5.com/en/articles/download/19299/NewsHeadlineEA.mq5 "Download NewsHeadlineEA.mq5")(68.81 KB)

[TradingButtons.mqh](https://www.mql5.com/en/articles/download/19299/TradingButtons.mqh "Download TradingButtons.mqh")(38.59 KB)

[terminal64\_Dp0JGQhX5a.png](https://www.mql5.com/en/articles/download/19299/terminal64_Dp0JGQhX5a.png "Download terminal64_Dp0JGQhX5a.png")(90.62 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/495045)**

![Automating Trading Strategies in MQL5 (Part 30): Creating a Price Action AB-CD Harmonic Pattern with Visual Feedback](https://c.mql5.com/2/168/19442-automating-trading-strategies-logo__2.png)[Automating Trading Strategies in MQL5 (Part 30): Creating a Price Action AB-CD Harmonic Pattern with Visual Feedback](https://www.mql5.com/en/articles/19442)

In this article, we develop an AB=CD Pattern EA in MQL5 that identifies bullish and bearish AB=CD harmonic patterns using pivot points and Fibonacci ratios, executing trades with precise entry, stop loss, and take-profit levels. We enhance trader insight with visual feedback through chart objects.

![Price Action Analysis Toolkit Development (Part 39): Automating BOS and ChoCH Detection in MQL5](https://c.mql5.com/2/168/19365-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 39): Automating BOS and ChoCH Detection in MQL5](https://www.mql5.com/en/articles/19365)

This article presents Fractal Reaction System, a compact MQL5 system that converts fractal pivots into actionable market-structure signals. Using closed-bar logic to avoid repainting, the EA detects Change-of-Character (ChoCH) warnings and confirms Breaks-of-Structure (BOS), draws persistent chart objects, and logs/alerts every confirmed event (desktop, mobile and sound). Read on for the algorithm design, implementation notes, testing results and the full EA code so you can compile, test and deploy the detector yourself.

![Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://c.mql5.com/2/130/Moving_to_MQL5_Algo_Forge_Part_LOGO__3.png)[Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)

When working on projects in MetaEditor, developers often face the need to manage code versions. MetaQuotes recently announced migration to GIT and the launch of MQL5 Algo Forge with code versioning and collaboration capabilities. In this article, we will discuss how to use the new and previously existing tools more efficiently.

![Polynomial models in trading](https://c.mql5.com/2/109/Polynomial_models_in_trading___LOGO.png)[Polynomial models in trading](https://www.mql5.com/en/articles/16779)

This article is about orthogonal polynomials. Their use can become the basis for a more accurate and effective analysis of market information allowing traders to make more informed decisions.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/19299&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083432852768431007)

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