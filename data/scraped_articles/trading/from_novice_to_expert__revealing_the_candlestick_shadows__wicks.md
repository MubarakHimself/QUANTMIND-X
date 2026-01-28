---
title: From Novice to Expert: Revealing the Candlestick Shadows (Wicks)
url: https://www.mql5.com/en/articles/19919
categories: Trading, Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:32:28.005098
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/19919&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071910237197250745)

MetaTrader 5 / Examples


### Contents

1. [Introduction](https://www.mql5.com/en/articles/19919#para1)
2. [Implementation—Enhancing Market Periods Synchronizer for Wick Visualization](https://www.mql5.com/en/articles/19919#para2)
3. [Testing](https://www.mql5.com/en/articles/19919#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19919#para4)
5. [Key Lessons](https://www.mql5.com/en/articles/19919#para5)
6. [Attachments](https://www.mql5.com/en/articles/19919#para6)

### Introduction

Every candlestick on a chart represents a small yet intricate story of market behavior—a dynamic interaction between buyers and sellers compressed into a single bar. In our earlier work, we built the [Market Periods Synchronizer,](https://www.mql5.com/en/articles/19918) a system designed to visualize higher-timeframe (HTF) structures within lower-timeframe charts. That project successfully isolated and filled the bodies of major candles, enabling traders to observe how lower-period movements collectively form those larger bars.

However, during that exploration, we noticed something remarkable. Once the bodies were properly visualized, the remaining unfilled regions naturally revealed the shadows or wicks—those thin vertical stretches that mark where price ventured before being rejected. We realized that while our initial focus was on the bodies, these untouched spaces—the very gaps between filled and unfilled regions—were quietly holding an entire layer of market intelligence.

The insight was subtle yet profound: by isolating the body, we had indirectly mapped out the shadows. Now, the challenge and opportunity before us are to study them deliberately. If we could highlight these wicks dynamically, making them visually distinct, we could begin analyzing their structure, behavior, and frequency—perhaps even uncover new intra-wick patterns or price behaviors that influence reversals, continuations, and liquidity events.

This new project represents that next step—a more in-depth investigation into the candlestick shadows. Instead of merely seeing them as leftover visuals outside the body, we now treat them as valuable data zones worthy of focused study. By highlighting and filling these regions programmatically, we aim to expose the hidden layers of volatility that define the upper and lower extremes of each candle.

Imagine being able to visualize, in real time, where the market previously rejected price—to see each wick not as an afterthought, but as a territory of market exploration. In doing so, we transform what was once visual noise into meaningful zones of study. The goal is to empower traders and developers alike to treat every candlestick shadow as a micro-environment for discovery, pattern recognition, and strategy development.

This approach builds upon the body-focused fills we developed earlier. Just as we discovered that candle bodies express directional conviction and momentum, the shadows tell another side of the story—one of hesitation, exhaustion, and liquidity imbalance. Together, the filled bodies and the highlighted shadows create a more holistic framework for understanding how price breathes within and across timeframes. By visualizing both, we begin to see not just where price has been accepted, but where it was rejected—a subtle but powerful difference in market structure analysis.

In this continuation, we’ll upgrade our existing Market Periods Synchronizer Control Utility to add Wick Visualization Utility, enhancing it with shadow-focused analytical capabilities. Instead of only filling higher-timeframe bodies, the system will now identify, isolate, and visually fill the wick regions—the upper and lower extremes of each candle—directly on lower-timeframe charts. These elements will update in real time, allowing us to witness how higher-timeframe wicks evolve dynamically as new lower-timeframe bars form. Ultimately, this isn’t just another technical enhancement—it’s an open invitation to explore what lies hidden within the candlestick shadows, to experiment, and to inspire others to push the boundaries of MQL5-based market visualization.

![Visualization of a price action in an H1 weak at M1](https://c.mql5.com/2/177/sync_scale.png)

Fig. 1. Visualization of a price action in an H1 wick at M1

In the illustration, **A** presents an H1 candlestick with a pronounced lower wick—a concise visual summary of a deeper market event. That single vertical shadow represents a swift rejection of lower prices, where sellers initially dominated before buyers regained control. While the higher timeframe compresses this sequence into one visual element, it actually captures a dynamic story of liquidity sweeps, absorption, and micro-accumulation. Each wick, though seemingly small, encodes valuable information about order flow, market psychology, and areas where institutional activity may have occurred. Ignoring it as mere noise overlooks an essential component of price behavior and structure.

**B** magnifies that same H1 wick through an M1 lens, revealing the hidden anatomy behind the rejection. What appeared as a single rejection tail in A unfolds into a detailed pattern of micro-support formations. In this instance, we observe a typical sequence: an initial support zone where price halted its decline, followed by a double-bottom structure that fueled the rejection momentum upward. While such patterns may vary across different rejections, this one serves as a textbook example of how lower timeframe formations contribute to higher timeframe wick behavior. Studying these transitions allows traders to connect intra-bar microstructure with macro candlestick interpretation—bridging the gap between what the chart summarizes and what actually transpired within it.

In the following paragraphs, we will revisit the structure of a candlestick to ensure every reader can clearly grasp the underlying concepts and appreciate the significance of each component within price action analysis.

**Understanding the Candlestick**

The very foundation of candlestick charting was laid not by modern financiers, but by a pioneering 18th-century Japanese rice trader, [Homma Munehisa](https://en.wikipedia.org/wiki/Honma_Munehisa "https://en.wikipedia.org/wiki/Honma_Munehisa"). His genius was in recognizing that the market was not just driven by supply and demand, but by the collective psychology of its participants—fear and greed. Homma's revolutionary methods for recording price action evolved into the candlestick system we use today. At its core, a candlestick is a visual compression of this battle, composed of two primary elements: the real body and the wicks.

The real body represents the core conflict between buyers and sellers by charting the distance between the period's opening and closing prices. In general, a green body signifies a bullish victory (close higher than open), while a red body shows a bearish victory (close lower than open). The wicks, the thin lines above and below, represent the range of price exploration, documenting the highest high and lowest low. The upper wick shows where price was rejected by sellers, and the lower wick indicates where it was rejected by buyers. The complete narrative of a single candlestick thus tells us not just who won the period (the body), but also the intensity of the conflict fought at the extremes (the wicks).

![Characteristics of a candlestick](https://c.mql5.com/2/177/rect18.png)

Fig. 2. Characteristics of a candlestick

Of course. Here is the completed table with all the essential characteristics of a candlestick.

| Characteristic | Description & Significance: |
| --- | --- |
| Real body | The wide part of the candlestick. It represents the core battle between buyers and sellers during the period, showing the range between the open and close prices. |
| Open Price | The price at the start of the time period. It forms the bottom of a bullish (green/white) body and the top of a bearish (red/black) body. |
| Close Price | The price at the end of the time period. It forms the top of a bullish body and the bottom of a bearish body. Its position relative to the open determines the candle's color and the period's "winner." |
| Body Color (Green/White) | Bullish Candle **.** Signifies that the closing price was _higher_ than the opening price, indicating buyer control and net buying pressure for the period. |
| Body Color (Red/Black) | Bearish Candle **.** Signifies that the closing price was _lower_ than the opening price, indicating seller control and net selling pressure for the period. |
| Upper Wick / Shadow | The thin line extending from the top of the body to the period's highest price. It represents a _rejection of higher prices_, showing where sellers stepped in and pushed the price down. |
| Lower Wick / Shadow | The thin line extending from the bottom of the body to the period's lowest price. It represents a rejection of lower prices, showing where buyers stepped in and pushed the price up. |
| High (of the period) | The absolute highest price traded during the candlestick's timeframe. It is the tip of the upper wick and represents the peak of buying interest. |
| Low (of the period) | The absolute lowest price traded during the candlestick's timeframe. It is the tip of the lower wick and represents the peak of selling interest. |

Understanding this structure allows traders to identify recurring patterns that signal potential market psychology shifts, a direct legacy of Homma's initial insights. Common structures include long-bodied candles for strong trends, "Doji" candles with minimal bodies for indecision, and candles with long lower wicks (like "Hammers") that suggest a rejection of lower prices.

However, a structure's meaning is never absolute; it is entirely contextual. A long green candle at the bottom of a downtrend is a potent bullish reversal signal, but the same candle after a long uptrend could signal a final climax of buying. Therefore, the true art lies not in memorizing shapes, but in interpreting what the structure's balance of body and wicks implies about the ongoing battle between fear and greed, within the context of its position on the chart.

Today, our main focus is on the wicks—those subtle yet powerful indicators of market rejection and hidden liquidity. We will leverage the capabilities of MQL5 to upgrade the concept behind the Market Periods Synchronizer, extending it to isolate and visualize wick regions. This enhancement will empower enthusiasts and analysts alike to integrate the tool into their studies, making it easier to decode and interpret higher-timeframe market structures with greater clarity.

**Wick in detail**

We are all familiar with the idea that candlestick wicks—or shadows—represent price rejections, the zones where market pressure temporarily resists or reverses the prevailing trend. Yet, few traders or analysts take the time to scale down these events to their lower-timeframe structures, where the real story unfolds. By examining the smaller timeframes, we can begin to understand the internal architecture of these wicks, revealing the micro-patterns that collectively form the visible rejection seen on higher-timeframe candles.

In this study, our goal is to visualize and isolate these rejection zones programmatically using MQL5, bridging the gap between higher-timeframe price formations and their underlying lower-timeframe mechanics. When we magnify a single higher-timeframe candle—say, an H1 bar—on the M1 chart, we uncover how familiar structures such as double bottoms, double tops, channels, and triangles often serve as the foundation for these wicks. These formations are far from random; they represent localized battles of liquidity, moments when buying and selling pressures collide and the market decides its next directional shift. By understanding these internal formations, we gain a deeper appreciation for how rejections, reversals, and momentum pivots emerge from within the shadows of price itself.

It’s worth noting that not all rejections are the same. Two H4 pin bars, for instance, might share similar external shapes yet differ entirely in their internal structures when viewed at lower timeframes. One might reveal a clean double bottom pattern, while another could exhibit a choppy consolidation or a volatility spike. Each wick tells a slightly different story about market sentiment, order flow, and trader psychology.

Our goal, therefore, is straightforward but insightful: to highlight the wick zones of higher-timeframe candles directly on lower-timeframe charts, making them visually distinct for detailed analysis. Through this, we hope to uncover new insights about market rejections, understand the hidden rhythm of liquidity shifts, and inspire new forms of technical study—perhaps even new trading strategies—built upon the more profound understanding of candlestick shadows.

With all the understanding we’ve built so far, it’s time to move into the implementation phase. In today’s session, we’ll be making a few key improvements to our existing codebase—introducing new features that extend the tool’s analytical depth. Continue to the next section below as we explore how these upgrades bring wick-focused visualization to life in a more dynamic and intuitive way.

### Implementation—Enhancing Market Periods Synchronizer for Wick Visualization

Step 1—Initialization and setup (UI + runtime state)

We begin by copying compile-time inputs into mutable runtime globals so the dashboard can change values without reloading the EA. The UI is built on a semi-transparent CCanvas bitmap label that sits behind interactive objects. To avoid name collisions and to make cleanup precise, every UI object name uses a per-chart prefix, MPS\_UI\_<chart\_id>\_. The canvas width is set to accommodate new Minimize and Quit buttons: g\_bg\_w = 430. The Minimize button is placed at x=340 (45px wide); Quit at x=385 (40px). Mouse move events are enabled so vertical sliders work, and EventSetTimer(g\_RefreshSec) starts periodic refreshes driven by the refresh slider.

Why this design:

- Mutable runtime copies permit immediate UI-driven changes,
- unique prefixes guarantee per-instance isolation,
- semi-transparent canvas provides a compact, readable container while allowing the chart to remain visible, and
- timers + OnTick coordination minimize unnecessary redraws.

```
int OnInit()
  {
   main_chart_id = ChartID();

   // copy inputs -> runtime
   g_HigherTF   = InpHigherTF;
   g_Lookback   = MathMax(10, InpLookback);
   g_ColorMajor = InpColorMajor;
   g_WidthMajor = MathMax(1, InpWidthMajor);
   g_RefreshSec = MathMax(1, InpRefreshSec);
   // + g_ShowWicks = InpShowWicks; etc.

   // unique UI prefix per chart
   UI_PREFIX = StringFormat("MPS_UI_%d_", main_chart_id);
   lbl_title   = UI_PREFIX + "LBL_TITLE";
   btn_minimize= UI_PREFIX + "BTN_MIN";
   btn_quit    = UI_PREFIX + "BTN_QUIT";
   // ... other object names ...

   // background area coordinates and size
   g_bg_x = 6; g_bg_y = Y_OFFSET - 6;
   g_bg_w = 430;        // fits title + Minimize + Quit
   g_bg_h = 250;

   // create UI background and widgets
   CreateUIBackground();                         // draws semi-transparent canvas
   CreateLabel(lbl_title, 12, 4 + Y_OFFSET, "Market Period Synchronizer Control Utility", 12);
   ObjectSetInteger(main_chart_id, lbl_title, OBJPROP_COLOR, XRGB(230,230,230));

   // create minimize and quit buttons at requested positions
   CreateButton(btn_minimize, 340, 4 + Y_OFFSET, 45, 20, "Minimize");
   CreateButton(btn_quit,     385, 4 + Y_OFFSET, 40, 20, "Quit");

   // create other UI elements (TF selectors, sliders, color buttons...)
   CreateAllOtherUIObjects();

   // interactive support
   ChartSetInteger(main_chart_id, CHART_EVENT_MOUSE_MOVE, true);
   EventSetTimer(g_RefreshSec);

   RefreshLines();
   return(INIT_SUCCEEDED);
  }
```

Step 2—Wick fill visualization (core drawing logic)

When the g\_ShowWicks toggle is ON, we calculate upper and lower wick regions of each higher-timeframe candle and draw them as semi-transparent rectangles. For every major bar, we compute body\_top, = MathMax(open, close) and body\_bot = MathMin(open, close). If high > body\_top, we create or update an upper wick rectangle named HTF\_MAJ\_W\_U\_<TF>\_<time>. Similarly, the lower wick rectangle is HTF\_MAJ\_W\_L\_<TF>\_<time>. All wick objects are set to OBJPROP\_BACK = true and OBJPROP\_SELECTABLE = false so they do not interfere with the UI. Each created wick object is appended to the keepNames\[\] list so garbage collection removes orphaned shapes.

Why this design:

- Filling the wick extents isolates rejection/liquidity zones visually,
- alpha blending makes wicks visually subtle but readable, and
- naming and keep-lists ensure consistent updates and clean removal.

```
// for each major bar (sorted_times/opens/closes prepared earlier)
double bar_high = sorted_highs[i];
double bar_low  = sorted_lows[i];
double body_top = MathMax(p_open, p_close);
double body_bot = MathMin(p_open, p_close);
uint wick_col   = ARGB(g_WickAlpha, 220,220,220);  // e.g. g_WickAlpha = 140

// upper wick
if(bar_high > body_top)
  {
   string wname_u = PREFIX_MAJ + "W_U_" + TFToString(g_HigherTF) + "_" + IntegerToString((int)t);
   if(ObjectFind(0, wname_u) == -1)
     ObjectCreate(0, wname_u, OBJ_RECTANGLE, 0, t, body_top, t + PeriodSeconds(g_HigherTF), bar_high);

   ObjectSetInteger(0, wname_u, OBJPROP_BGCOLOR, wick_col);
   ObjectSetInteger(0, wname_u, OBJPROP_COLOR,   wick_col);
   ObjectSetInteger(0, wname_u, OBJPROP_FILL,    true);
   ObjectSetInteger(0, wname_u, OBJPROP_BACK,    true);
   ObjectSetInteger(0, wname_u, OBJPROP_SELECTABLE, false);
   ObjectMove(0, wname_u, 0, t, body_top);
   ObjectMove(0, wname_u, 1, t + PeriodSeconds(g_HigherTF), bar_high);
   // add to keepNames[] so GC won't delete it
  }
else
  {
   // remove upper wick if it no longer exists
   ObjectDelete(0, wname_u);
  }

// lower wick similar logic...
```

Step 3—Interactive controls: toggles, sliders, minimize, and quit

All interactions flow through OnChartEvent. Clicking the wick toggle flips g\_ShowWicks and immediately calls RefreshLines() so users see results in real time. Sliders are implemented as vertical knob buttons placed over a track; dragging updates runtime values (e.g., g\_WidthMajor, g\_RefreshSec). Minimize and Quit are handled deterministically:

Minimize toggles g\_minimized. When minimized, we shrink the canvas height to a minimal strip (≈30px) and hide every UI object except the title and the Restore (previously Minimize) button so the chart is unobstructed while state is preserved.

Quit first calls DeleteAllHTFLines(), and then calls ExpertRemove() to exit cleanly.

Important safety rules implemented:

- Dropdowns and slider-dragging operations are disabled while g\_minimized is true.
- Clicking outside the TF dropdown hides it.
- Slider dragging uses CHARTEVENT\_MOUSE\_MOVE and clamps the knob between the track top/bottom.

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   // Minimize / Restore
   if(id == CHARTEVENT_OBJECT_CLICK && sparam == btn_minimize)
     {
      g_minimized = !g_minimized;
      if(g_minimized) MinimizeUI(); else RestoreUI();
      return;
     }

   // Quit
   if(id == CHARTEVENT_OBJECT_CLICK && sparam == btn_quit)
     {
      DeleteAllHTFLines();      // remove all HTF_ objects
      ExpertRemove();           // exit EA
      return;
     }

   // Wicks toggle
   if(id == CHARTEVENT_OBJECT_CLICK && sparam == btn_toggle_wicks)
     {
      g_ShowWicks = !g_ShowWicks;
      ObjectSetString(main_chart_id, btn_toggle_wicks, OBJPROP_TEXT, g_ShowWicks ? "Wicks: ON" : "Wicks: OFF");
      RefreshLines();
      return;
     }

   // Slider knob click => set g_slider_drag/g_current_slider; mouse-move used to update value
   // ... slider drag handling omitted for brevity ...
  }
```

Helper actions for minimize/restore:

```
void MinimizeUI()
  {
   // Hide most UI objects and shrink canvas
   HideAllUIExceptTitleAndRestore();
   UpdateBackgroundHeight(30);
  }

void RestoreUI()
  {
   // Re-create/re-show UI objects and expand canvas
   UpdateBackgroundHeight(250);
   ShowAllUIObjects();
  }
```

Step 4—Cleanup and deinitialization

On deinit (or when Quit is clicked), the EA performs a deterministic, complete cleanup:

- EventKillTimer() stops periodic timer callbacks.
- DeleteAllHTFLines() iterates ObjectsTotal() and deletes any object whose name contains the HTF\_ prefix (vertical lines, fills, open/close trend lines, wick rectangles, and their labels).
- All UI objects created with the UI\_PREFIX are explicitly deleted by iterating a names\[\] array.
- All slider objects are deleted.
- Any open dropdown is hidden, and g\_bgCanvas.Destroy() is called to remove the bitmap label.

Why this is necessary:

- Ensures the chart remains pristine (no ghost objects) after removal,
- Prevents timers from re-creating objects after the EA has been removed,
- Makes the EA safe to load/unload repeatedly.

```
void OnDeinit(const int reason)
  {
   EventKillTimer();

   // remove all HTF_* objects
   DeleteAllHTFLines();

   // explicit removal of UI names we created
   string names[] = { lbl_title, btn_major_tf, lbl_major_tf, btn_lookback_minus, lbl_lookback, btn_lookback_plus,
                      btn_toggle_openclose, btn_toggle_fill, btn_toggle_wicks, btn_major_col1, btn_major_col2,
                      btn_major_col3, btn_major_col4, btn_minor1_toggle, btn_minor1_tf, btn_minor2_toggle,
                      btn_minor2_tf, btn_clear_all, lbl_major_width, lbl_refresh_label, btn_minimize, btn_quit };
   for(int i=0;i<ArraySize(names);i++)
     if(StringLen(names[i])>0) ObjectDelete(main_chart_id, names[i]);

   // remove slider objects
   for(int s=0; s<SLIDER_COUNT; s++)
     {
      if(StringLen(g_slider_track_names[s])>0) ObjectDelete(main_chart_id, g_slider_track_names[s]);
      if(StringLen(g_slider_knob_names[s])>0)  ObjectDelete(main_chart_id, g_slider_knob_names[s]);
     }

   if(g_tf_dropdown_visible) HideTFDropdown();

   // destroy canvas bitmap
   g_bgCanvas.Destroy();
  }
```

Practical value and developer notes after the enhancements:

This enhancement converts the synchronizer from a passive marker generator into an interactive analysis utility:

- Wick fills help reveal rejection zones, traps, and potential liquidity clusters by highlighting the exact price ranges outside the candle body; alpha-blending provides immediate context without obscuring price bars.
- Minimize preserves UI state while reducing screen clutter—perfect for switching between focused study and unobstructed charting.
- Quit provides a one-click, zero-residue exit for clean multi-EA workflows.
- Sliders and color buttons make visual tuning fast and intuitive—no repeated re-opening of the inputs dialog.

Developer tips:

1. Keep all object names under a single predictable prefix: this dramatically simplifies deinit and debugging.
2. When creating many objects every tick, always maintain a keepNames\[\] and perform a single final sweep to delete stale objects—avoids build-up and keeps charts responsive.
3. For performance, avoid redrawing unless needed: coordinate OnTick new-bar detection with EventSetTimer, and only call RefreshLines() when needed (new major/minor bars or UI changes).
4. Make the canvas non-selectable and UI objects frontmost (OBJPROP\_BACK = false) so interaction remains reliable

### Testing

Live Chart Deployment

Deploying the Market Periods Synchronizer EA on a live MetaTrader 5 chart is both straightforward and highly optimized for real-time visual analysis. Unlike traditional strategy-testing environments, which limit dynamic object rendering, this tool is designed for direct chart interaction and market immersion. To begin, simply compile the EA source (at the bottom of this article) in MetaEditor using F7, producing the .ex5 file. Next, drag it from the Navigator panel onto your chosen chart—preferably a lower timeframe such as M5 or M15. These granular views best reveal the inner wick structures within higher-timeframe candles like H1 or D1, allowing detailed observation of rejections and liquidity reactions.

Once attached, the Inputs dialog provides flexibility to tailor visualization to your study preferences. Set InpHigherTF to define the primary analysis timeframe (e.g., H1), and adjust InpLookback between 100 and 300 bars for historical depth without overloading your chart. To activate wick analysis, enable InpShowWicks = true, and fine-tune InpWickAlpha (around 120) for a smooth, semi-transparent appearance that blends harmoniously with chart data. Although the EA uses no trading logic, enabling AutoTrading ensures full script responsiveness, which you can confirm by toggling the “AutoTrading” button on the toolbar.

Once initialized, the system auto-configures itself—instantly drawing all relevant major and minor timeframe markers, body fills, and now wick zones, each refreshing dynamically as the market progresses. The interactive dashboard, positioned neatly in the upper-left corner, allows immediate access to all controls, including the new Minimize option for compact display and the Wick ON/OFF switch for visual clarity during analysis. The OnTick() event continuously synchronizes all elements with the latest market data, ensuring smooth, live updates without interruptions.

This live deployment truly shines for intraday traders and analysts: one can observe wick formation as it unfolds, correlate it with micro-structure on lower timeframes, or toggle minor periods for multi-level confluence. When focus is required during trade execution, the minimize function collapses the interface, leaving the chart fully visible and unobstructed.

The animated image below demonstrates the EA in live action within the MetaTrader 5 terminal—showcasing real-time visualization, interactive controls, and wick filling effects. Every component operates exactly as designed, confirming the success of the new enhancements. With this solid foundation, we are already inspired to expand the tool further, integrating deeper analytics and even more intuitive control mechanisms.

![](https://c.mql5.com/2/177/terminal64_s4L4WEcsd9.gif)

Fig. 3. Testing the new features

### Conclusion

As live testing confirms, the Market Periods Synchronizer has evolved beyond the boundaries of a static indicator—it now serves as a dynamic lens into the hidden mechanics of price, empowering traders with deeper, more intuitive market insights. In a landscape where fleeting rejections can define opportunity or loss, this EA’s live responsiveness and seamless operation transform observation into precision.

This exploration is especially significant in its focus on wick price action—a vital but often overlooked component of market structure. By visualizing and isolating wick regions, analysts can now study how price behaves within those rejection zones, uncovering liquidity traps, absorption points, or reversal footprints that form the backbone of larger moves. Observing these reactions at lower timeframes, such as M5 within H1 or H4 candles, offers unprecedented clarity for understanding the interplay between short-term volatility and long-term structure.

Equally impressive is the enhanced control interface, featuring real-time interactivity through the new Minimize, Quit, and Wick ON/OFF controls. These additions streamline user experience—allowing quick toggling, workspace decluttering, and one-click termination that leaves the chart perfectly clean. The tool now runs smoothly, maintains stability under live conditions, and ensures every object is automatically cleared upon removal, reflecting thoughtful engineering and professional-grade execution.

### Key Lessons

| Key Lesson | Description: |
| --- | --- |
| Use Unique Prefixes for Object Identification. | Assign consistent prefixes (e.g., "HTF\_MAJ\_") to all chart objects created by the EA, enabling efficient targeted deletion via StringFind() in loops over ObjectsTotal(). This prevents naming conflicts with other indicators and ensures thorough cleanup during deinitialization or quit operations, as demonstrated in DeleteAllHTFLines(). |
| Implement Modular UI Creation and Destruction | Encapsulate UI element setup in dedicated functions like CreateAllOtherUIObjects() and counterpart deletion routines, facilitating dynamic states such as UI minimization. This modularity supports seamless toggling between expanded and collapsed views without redundant code, enhancing code readability and maintenance in interactive dashboards. |
| Leverage ARGB for Non-Intrusive Visual Overlays | Utilize alpha transparency in ARGB colors for rendering wick fills and backgrounds, allowing semi-transparent overlays that highlight rejection zones without obscuring price action. Adjustable alpha (e.g., g\_WickAlpha=120) balances visibility and clarity, as integrated in RefreshLines() for professional, layered chart enhancements. |
| Optimize Event Handling with State Guards | Incorporate boolean flags (e.g., g\_minimized) as guards in OnChartEvent() to conditionally process interactions like dropdowns or sliders only in active UI states. This prevents errors during collapsed modes and streamlines user flows, exemplified by gating TF dropdown visibility for responsive, error-free controls. |
| Ensure Comprehensive Resource Cleanup in OnDeinit | Systematically release all resources—timers via EventKillTimer(), objects via explicit deletions and loops, and canvases via Destroy()—in OnDeinit(), mirroring quit-button logic with DeleteAllHTFLines(). This practice avoids memory leaks and chart clutter, upholding MetaTrader 5 best practices for detachable EAs. |

### Attachments

| File Name | Version | Description |
| --- | --- | --- |
| MarketPeriodsSynchronizerEA.mq5 | 1.01 | Core Expert Advisor implementing multi-timeframe period synchronization, wick visualization with alpha-blended fills, interactive dashboard for toggles/sliders, and UI minimization/quit controls for enhanced trader workflow. |

[Back to Contents](https://www.mql5.com/en/articles/19919#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19919.zip "Download all attachments in the single ZIP archive")

[MarketPeriodsSynchronizer\_EA.mq5](https://www.mql5.com/en/articles/download/19919/MarketPeriodsSynchronizer_EA.mq5 "Download MarketPeriodsSynchronizer_EA.mq5")(94.4 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/499115)**

![Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (FinCon)](https://c.mql5.com/2/110/Neural_Networks_in_Trading____FinCon____LOGO2.png)[Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (FinCon)](https://www.mql5.com/en/articles/16916)

We invite you to explore the FinCon framework, which is a a Large Language Model (LLM)-based multi-agent system. The framework uses conceptual verbal reinforcement to improve decision making and risk management, enabling effective performance on a variety of financial tasks.

![Introduction to MQL5 (Part 27): Mastering API and WebRequest Function in MQL5](https://c.mql5.com/2/178/17774-introduction-to-mql5-part-27-logo.png)[Introduction to MQL5 (Part 27): Mastering API and WebRequest Function in MQL5](https://www.mql5.com/en/articles/17774)

This article introduces how to use the WebRequest() function and APIs in MQL5 to communicate with external platforms. You’ll learn how to create a Telegram bot, obtain chat and group IDs, and send, edit, and delete messages directly from MT5, building a strong foundation for mastering API integration in your future MQL5 projects.

![Self Optimizing Expert Advisors in MQL5 (Part 16): Supervised Linear System Identification](https://c.mql5.com/2/178/20023-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 16): Supervised Linear System Identification](https://www.mql5.com/en/articles/20023)

Linear system identifcation may be coupled to learn to correct the error in a supervised learning algorithm. This allows us to build applications that depend on statistical modelling techniques without necessarily inheriting the fragility of the model's restrictive assumptions. Classical supervised learning algorithms have many needs that may be supplemented by pairing these models with a feedback controller that can correct the model to keep up with current market conditions.

![Black-Scholes Greeks: Gamma and Delta](https://c.mql5.com/2/178/20054-black-scholes-greeks-gamma-logo.png)[Black-Scholes Greeks: Gamma and Delta](https://www.mql5.com/en/articles/20054)

Gamma and Delta measure how an option’s value reacts to changes in the underlying asset’s price. Delta represents the rate of change of the option’s price relative to the underlying, while Gamma measures how Delta itself changes as price moves. Together, they describe an option’s directional sensitivity and convexity—critical for dynamic hedging and volatility-based trading strategies.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/19919&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071910237197250745)

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