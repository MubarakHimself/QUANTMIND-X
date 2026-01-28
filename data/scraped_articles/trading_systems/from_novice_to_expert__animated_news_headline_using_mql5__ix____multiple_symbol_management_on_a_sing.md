---
title: From Novice to Expert: Animated News Headline Using MQL5 (IX) — Multiple Symbol Management on a single chart for News Trading
url: https://www.mql5.com/en/articles/19008
categories: Trading Systems, Integration
relevance_score: -2
scraped_at: 2026-01-24T14:16:50.017275
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=vbjcvcgomegpfxrlkknuaalwianqxdtx&ssn=1769253407934601135&ssn_dr=1&ssn_sr=0&fv_date=1769253407&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19008&back_ref=https%3A%2F%2Fwww.google.com%2F&title=From%20Novice%20to%20Expert%3A%20Animated%20News%20Headline%20Using%20MQL5%20(IX)%20%E2%80%94%20Multiple%20Symbol%20Management%20on%20a%20single%20chart%20for%20News%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925340800517140&fz_uniq=5083459524515339295&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/19008#para1)
- [Understanding the concept](https://www.mql5.com/en/articles/19008#para2)
- [Implementation>](https://www.mql5.com/en/articles/19008#para3)

  - [Modifying the CtradingButtons class for multiple-symbol trading](https://www.mql5.com/en/articles/19008#para3a)
  - [Integrating the Multi-Symbols trading functionality with the New Headline EA](https://www.mql5.com/en/articles/19008#para3b)

- [Testing](https://www.mql5.com/en/articles/19008#para4)
- [Conclusion](https://www.mql5.com/en/articles/19008#para5)
- [Key Lessons](https://www.mql5.com/en/articles/19008#para6)
- [Attachments](https://www.mql5.com/en/articles/19008#para7)

### Introduction

During periods of high volatility—such as economic news releases—traders often gamble on breakouts because the market’s immediate reaction is unpredictable. When there is a major news release, price typically spikes sharply, followed by corrections and possible trend continuations. In these conditions, traders may want to trade several symbols simultaneously, but this is difficult to achieve with the default MetaTrader 5 setup. By design, a single chart only supports one Expert Advisor, meaning traders must open multiple charts and attach a separate EA to each symbol.

In today’s discussion, we present a solution to this limitation, a multi-symbol trading feature integrated into the News Headline  EA. With this enhancement, traders can manage multiple pairs from a single chart using intuitive trading buttons. We will explore how the power of MQL5—leveraging both the Standard Library and custom trading classes—enables the creation of a sophisticated EA capable of handling multiple symbols seamlessly on one chart.

![One Expert Advisor support per pair.](https://c.mql5.com/2/163/ShareX_avTJERdZQ7.gif)

Fig. 1: Only one EA is allowed per chart on the MetaTrader 5 terminal

The image above illustrates the limitation of MetaTrader 5’s setup, where only one EA can run on a single chart. To trade multiple symbols effectively, we need a sophisticated EA capable of managing both the current chart pair and other pairs simultaneously—even while operating from just one chart.

By the end of this discussion, we aim to achieve the following:

- Develop a more sophisticated EA.
- Expand an existing MQL5 header class with new features.
- Leverage the MQL5 Standard Library to build new classes.
- Integrate new functionality into an existing EA.
- Apply modularization and structured grouping of inputs.

### Understanding the Concept

This stage begins with a quick review of our previous work. We started with a simple animated news headline EA, fetching data from the Economic Calendar and external news APIs such as Alpha Vantage. Over time, we integrated locally hosted AI models, automated news-trading strategies, and manual trading buttons to make the EA more reliable.

While these innovations improved the system, they were not a complete solution. Algorithmic trading continues to evolve, and with every technological advancement, new challenges emerge that drive us to upgrade our systems. Today, we are addressing one of those challenges: enabling multi-pair trading within the same EA.

Why Is It Necessary?

A valid question one might ask is, why do we need this feature?

During high-volatility events, such as economic news releases, traders must react quickly, often managing multiple positions and symbols within seconds. This development provides a critical advantage by merging algorithmic and manual trading in one place, enhancing efficiency and control. With one click, a trader can place trades across several symbols and manage multiple positions simultaneously—gaining both speed and performance efficiency.

Feature Integration Process

With this context in mind, let’s briefly outline how the new feature will be added. To expand our EA, we rely on header inclusion and custom trading button classes, which keep the main codebase clean and modular.

For multi-symbol trading, we need the ability to select desired pairs that will be executed alongside the current chart’s pair when manual trading buttons are pressed. To achieve this, we will use the CCheckBox and CLabel classes from the MQL5 Standard Library. These components will allow us to display selectable pairs, manage user input, and link selections directly to the button event handlers.

Finally, our CTradingButtons class will be extended to incorporate these new features seamlessly.

### Implementation

We will approach this in two main stages. First, we will modify the CTradingButtons class in the TradingButtons header to implement the multi-symbol trading features outlined in our design. The second stage will focus on adapting the News Headline EA to support these new capabilities.

Follow along carefully as we break down the code and explain how each part contributes to bringing the idea to life. For clarity, each code section and its explanation will be numbered sequentially from top to bottom, with emphasis placed on the new functionality.

If you would like to explore the foundational parts of the code, I encourage you to revisit earlier [publications](https://www.mql5.com/en/users/billionaire2024/publications) in this series, where we covered the [initial versions](https://www.mql5.com/en/articles/18299) in detail.

### Modifying the CtradingButtons class for multiple-symbol trading

We first introduced this header in the [previous](https://www.mql5.com/en/articles/18975) article, which you can refer to for clarity. In this section, we extend it with a new feature.

High-level overview

This class (CTradingButtons) bundles three responsibilities so it acts as a compact multipair trading module you can drop into an EA: (1) a UI (canvas + buttons + dynamically created checkboxes for symbols), (2) a small trade wrapper (a CTrade instance that sends orders), and (3) a symbol resolution & multipair engine (maps requested base names like EURUSD to broker symbols and applies actions across all selected symbols). The high-level design keeps index alignment across arrays: the requested list (what the EA passes in), the resolved broker symbols (what the terminal actually trades), and the checkboxes (what the user toggles)—index i represents the same pair in all arrays. This makes wiring UI → EA selection arrays straightforward and predictable.

```
// top-of-file: class skeleton + key members (from TradingButtons.mqh)

class CTradingButtons

{

private:

   // UI & buttons
   CButton btnMultiToggle;
   CButton btnBuy, btnSell, btnCloseAll, btnDeleteOrders, btnCloseProfit, btnCloseLoss, btnBuyStop, btnSellStop;
   CCanvas buttonPanel;

   // trading

   CTrade  trade;

   // multipair UI & resolution

   CCheckBox *pairChecks[];      // dynamic checkbox pointers, index-aligned with requested list
   string     availablePairs[];  // resolved broker symbols (index-aligned)
   string     resolvedBases[];   // original requested bases (for logging)
   bool       multiEnabled;
public:

   double LotSize;
   int    StopLoss;
   int    TakeProfit;
   int    StopOrderDistancePips;
   double RiskRewardRatio;
   CTradingButtons() { /* default init inlined in full file */ }
   void Init();
   void Deinit();
   // ... other methods follow

};
```

Fields & constructor — what the class stores

The class stores layout parameters (button width/height/spacing), checkbox sizing and starting coordinates, the dynamic checkbox array, the resolved symbol arrays, and trading configuration (lot size, stops, risk/reward). Its constructor sets sensible defaults (e.g., LotSize=0.01, StopLoss=50, TakeProfit=100, multiEnabled=true) so the EA can start with a working configuration and override what it needs later. Keeping these fields as public (for trading params) and private (for UI internals) keeps the interface simple and safe.

```
// constructor + key field defaults (actual defaults in your file)

CTradingButtons() :
   buttonWidth(100), buttonHeight(30), buttonSpacing(10),
   checkWidth(120), checkHeight(20), checkSpacing(6), checkStartX(10),
   LotSize(0.01), StopLoss(50), TakeProfit(100),
   StopOrderDistancePips(8), RiskRewardRatio(2.0),
   multiEnabled(true)

{

   // constructor body intentionally minimal — Init() performs heavier setup

}
```

Initialization & cleanup

Init() configures the CTrade wrapper (magic number, deviation) and builds the UI (panel, buttons, multi-toggle). Deinit() carefully destroys any dynamic objects (checkboxes, buttons, canvas) and frees arrays to avoid orphan chart objects or memory leaks. The cleanup loops over the pairChecks\[\] array and calls Destroy() and delete on dynamic pointers, then frees the array — critical when running/unloading the EA repeatedly during development.

```
// Init & Deinit excerpt

void Init()

{
   trade.SetExpertMagicNumber(123456);
   trade.SetDeviationInPoints(10);
   CreateButtonPanel();
   CreateButtons();
   CreateMultiToggle();
   UpdateMultiToggleVisual();

}

void Deinit()

{
   // destroy checkboxes
   for(int i = 0; i < ArraySize(pairChecks); i++)

   {
      if(CheckPointer(pairChecks[i]) == POINTER_DYNAMIC)
         pairChecks[i].Destroy();
         delete pairChecks[i];
      }
   }
   ArrayFree(pairChecks);

   // destroy buttons and panel
   btnMultiToggle.Destroy();
   btnBuy.Destroy();
   btnSell.Destroy();
   btnCloseAll.Destroy();
   btnDeleteOrders.Destroy();
   btnCloseProfit.Destroy();
   btnCloseLoss.Destroy();
   btnBuyStop.Destroy();
   btnSellStop.Destroy();
   buttonPanel.Destroy();
   ObjectDelete(0, "ButtonPanel");

}
```

Symbol resolution (important for multipair)

To trade a "friendly" name like EURUSD the EA must map it to the broker's exact symbol string (it might be EURUSD, EURUSD.ecn, FX.EURUSD, etc.). ResolveSymbol(base) first tries an exact match (fast path). If that fails, it loops all terminal symbols, searches for starts-with and then contains matches (prefers starts-with), and excludes disabled symbols. This resolution step produces availablePairs\[i\] entries used by trading routines and UI checkboxes—it’s the glue between the EA’s requested names and the broker’s actual tradable symbols.

```
// ResolveSymbol implementation (exact + starts-with + contains search)

string ResolveSymbol(const string base)

{
   if(StringLen(base) == 0) return("");
   // 1) Try exact symbol name first
   string baseName = base;
   if(SymbolInfoInteger(baseName, SYMBOL_SELECT) != 0 || SymbolSelect(baseName, false))
   {
      if(SymbolInfoInteger(baseName, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED)

      {
         PrintFormat("ResolveSymbol: exact match found %s", baseName);

         return(baseName);
      }
   }

   // 2) search all terminal symbols

   int total = SymbolsTotal(false);
   string base_u = base; StringToUpper(base_u);
   string firstStarts = ""; string firstContains = "";

   for(int i = 0; i < total; i++)

   {
      string sym = SymbolName(i, false);
      string sym_u = sym; StringToUpper(sym_u);
      if(sym_u == base_u) continue;
      if(SymbolInfoInteger(sym, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED) continue;
      if(StringFind(sym_u, base_u) == 0)
      {
         if(firstStarts == "") firstStarts = sym;
      }
      else if(StringFind(sym_u, base_u) >= 0)

      {
         if(firstContains == "") firstContains = sym;
      }
   }

   if(firstStarts != "") { PrintFormat("ResolveSymbol: resolved %s -> %s (starts-with)", base, firstStarts); return(firstStarts); }
   if(firstContains != "") { PrintFormat("ResolveSymbol: resolved %s -> %s (contains)", base, firstContains); return(firstContains); }
   PrintFormat("ResolveSymbol: no match for %s", base);
   return("");
}
```

Creating the multipair UI — CreatePairCheckboxes(...)

CreatePairCheckboxes(inMajorPairs\[\], inPairSelected\[\], yPos) is the routine that: (a) resolves each requested base to a broker symbol, (b) ensures the symbol is present in Market Watch (SymbolSelect) and is tradable, and (c) dynamically creates a CCheckBox for each resolved symbol while preserving index alignment with the EA’s arrays. Unresolved or disabled entries are kept as placeholders so availablePairs\[i\] maps correctly to the original inMajorPairs\[i\]. Initial checked state for each created checkbox is taken from inPairSelected\[i\], so the UI and EA selection arrays are synchronized from the start.

```
// CreatePairCheckboxes: resolve requested bases -> create checkboxes aligned by index

void CreatePairCheckboxes(string &inMajorPairs[], bool &inPairSelected[], int yPos)

{
   // cleanup previous

   for(int i = 0; i < ArraySize(pairChecks); i++)

   {
      if(CheckPointer(pairChecks[i]) == POINTER_DYNAMIC)
      {
         pairChecks[i].Destroy();
         delete pairChecks[i];
      }
   }

   ArrayFree(pairChecks);
   ArrayResize(availablePairs, ArraySize(inMajorPairs));
   ArrayResize(resolvedBases, ArraySize(inMajorPairs));
   for(int k = 0; k < ArraySize(availablePairs); k++) { availablePairs[k] = ""; resolvedBases[k] = ""; }
   int count = ArraySize(inMajorPairs);
   if(count == 0) return;

   // Resolve each requested base

   for(int i = 0; i < count; i++)
   {
      string requested = inMajorPairs[i];
      string resolved = ResolveSymbol(requested);
      if(resolved == "")
      {
         PrintFormat("CreatePairCheckboxes: could not resolve %s -> skipping checkbox", requested);
         availablePairs[i] = "";
         resolvedBases[i] = requested;
         continue;
      }
      if(!SymbolSelect(resolved, true))
         PrintFormat("CreatePairCheckboxes: SymbolSelect failed for %s (from %s) Err=%d", resolved, requested, GetLastError());
      if(SymbolInfoInteger(resolved, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED)
      {
         PrintFormat("CreatePairCheckboxes: resolved symbol %s is disabled (from %s) - skipping", resolved, requested);
         availablePairs[i] = "";
         resolvedBases[i] = requested;
         continue;
      }
      availablePairs[i] = resolved;
      resolvedBases[i] = requested;
   }

   // Create checkbox controls (preserve index alignment)

   ArrayResize(pairChecks, count);
   int xPos = checkStartX;
   int chartW = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int wrapX = chartW - (buttonWidth * 3) - 30;
   for(int i = 0; i < count; i++)
   {
      if(StringLen(availablePairs[i]) == 0) { pairChecks[i] = NULL; continue; }
      pairChecks[i] = new CCheckBox();
      string objName = "Chk_" + availablePairs[i];
      if(!pairChecks[i].Create(ChartID(), objName, 0, xPos, yPos, xPos + checkWidth, yPos + checkHeight))
      {

         PrintFormat("CreatePairCheckboxes: failed to create checkbox %s Err=%d", objName, GetLastError());
         delete pairChecks[i];
         pairChecks[i] = NULL;
         availablePairs[i] = "";
         continue;
      }
      pairChecks[i].Text(" " + availablePairs[i]);
      pairChecks[i].Color(clrBlack);
      bool checked = false;
      if(i < ArraySize(inPairSelected)) checked = inPairSelected[i];
      pairChecks[i].Checked(checked);
      xPos += checkWidth + checkSpacing;
      if(xPos + checkWidth > wrapX) { xPos = checkStartX; yPos += checkHeight + checkSpacing; }
   }

   ChartRedraw();
   PrintFormat("CreatePairCheckboxes: created checkboxes (resolved count=%d)", CountResolvedPairs());
}
```

Counting/helper — CountResolvedPairs()

Small helpers keep code readable. CountResolvedPairs() simply counts non-empty availablePairs\[\] entries and is used for logging or updating UI text. It’s a one-liner but useful during initialization and troubleshooting.

```
// Count resolved availablePairs entries

int CountResolvedPairs()
{
   int c = 0;
   for(int i = 0; i < ArraySize(availablePairs); i++)
      if(StringLen(availablePairs[i]) > 0) c++;
   return c;
}
```

Event handling — HandleChartEvent(...)

All clicks from the chart objects are funneled to HandleChartEvent. It recognizes three categories: (A) checkbox clicks (object names prefixed with Chk\_ — it finds which resolved symbol was clicked and synchronizes the inPairSelected\[i\] array), (B) the multi-toggle button (toggles multiEnabled and updates visuals), and (C) action buttons (Buy/Sell/Close all/Delete pending/Place stops) — each button click delegates to the appropriate operation, passing the EA’s requested pairs and selection flags. The function is the UI → engine router and keeps the UI and EA arrays in sync.

```
// HandleChartEvent: routes object clicks to checkboxes / toggle / actions

void HandleChartEvent(const int id, const string &sparam, string &inMajorPairs[], bool &inPairSelected[])

{
   if(id == CHARTEVENT_OBJECT_CLICK)
   {
      // Checkbox click handling
      if(StringFind(sparam, "Chk_") == 0)
      {
         for(int i = 0; i < ArraySize(availablePairs); i++)
         {
            if(StringLen(availablePairs[i]) == 0) continue;
            string expected = "Chk_" + availablePairs[i];
            if(expected == sparam)

            {
               if(CheckPointer(pairChecks[i]) == POINTER_DYNAMIC)
               {
                 bool current = pairChecks[i].Checked();
                  if(i < ArraySize(inPairSelected)) inPairSelected[i] = current;
                  else { ArrayResize(inPairSelected, i+1); inPairSelected[i] = current; }
                  PrintFormat("HandleChartEvent: checkbox for %s toggled -> %s", availablePairs[i], current ? "true":"false");
               }
               break;
            }
         }
         return;
      }
      // Multi toggle

      if(sparam == btnMultiToggle.Name())

      {
         multiEnabled = !multiEnabled;
         UpdateMultiToggleVisual();
         PrintFormat("HandleChartEvent: Multi toggle clicked. New multiEnabled=%s", (multiEnabled ? "true":"false"));
         return;
      }

      // Buttons - delegate to command handlers

      if(sparam == btnBuy.Name())        OpenBuyOrder(inMajorPairs, inPairSelected);
      else if(sparam == btnSell.Name()) OpenSellOrder(inMajorPairs, inPairSelected);
      else if(sparam == btnCloseAll.Name()) CloseAllPositions(inMajorPairs, inPairSelected);
      else if(sparam == btnDeleteOrders.Name()) DeleteAllPendingOrders(inMajorPairs, inPairSelected);
      else if(sparam == btnCloseProfit.Name()) CloseProfitablePositions(inMajorPairs, inPairSelected);
      else if(sparam == btnCloseLoss.Name())   CloseLosingPositions(inMajorPairs, inPairSelected);
      else if(sparam == btnBuyStop.Name())     PlaceBuyStop(inMajorPairs, inPairSelected);
      else if(sparam == btnSellStop.Name())    PlaceSellStop(inMajorPairs, inPairSelected);
   }
}
```

UI creation helpers

The UI creation is split into small helpers: CreateButtonPanel() makes a canvas bitmap (panel background and decorative rectangle), CreateButtons() instantiates and styles each action button with consistent font/size/positions, and CreateMultiToggle() creates the toggle button placed above the main action buttons. UpdateMultiToggleVisual() updates the toggle’s text and color to show whether multipair mode is active. These helpers keep visual code out of business logic and make style changes easy.

```
// Create panel + buttons + multi-toggle visual helpers

void CreateButtonPanel()

{
   int panelWidthLocal = buttonWidth + 20;
   int panelHeightLocal = (buttonHeight + buttonSpacing) * 9 + buttonSpacing + 40;
   int x = 0, y = 40;

   if(!buttonPanel.CreateBitmap(0, 0, "ButtonPanel", x, y, panelWidthLocal, panelHeightLocal, COLOR_FORMAT_ARGB_NORMALIZE))
   {
      Print("Failed to create button panel: Error=", GetLastError());
      return;
   }
   ObjectSetInteger(0, "ButtonPanel", OBJPROP_ZORDER, 10);
   buttonPanel.FillRectangle(0, 0, panelWidthLocal, panelHeightLocal, ColorToARGB(clrDarkGray, 200));
   buttonPanel.Rectangle(0, 0, panelWidthLocal - 1, panelHeightLocal - 1, ColorToARGB(clrRed, 255));
   buttonPanel.Update(true);
   ChartRedraw(0);
}
void CreateMultiToggle()
{
   int x = 10, y = 120;
   string font = "Calibri"; int fontSize = 8;
   color buttonBgColor = clrBlack;

   if(btnMultiToggle.Create(0, "btnMultiToggle", 0, x, y, x + buttonWidth, y + buttonHeight))
   {

      ObjectSetString(0, "btnMultiToggle", OBJPROP_FONT, font);
      ObjectSetInteger(0, "btnMultiToggle", OBJPROP_FONTSIZE, fontSize);
      ObjectSetInteger(0, "btnMultiToggle", OBJPROP_BGCOLOR, buttonBgColor);
      ObjectSetInteger(0, "btnMultiToggle", OBJPROP_ZORDER, 11);
   }
}

void UpdateMultiToggleVisual()
{
   if(multiEnabled)
   {
      btnMultiToggle.Text("MULTI:ON");
      btnMultiToggle.ColorBackground(clrGreen);
      btnMultiToggle.Color(clrWhite);
   }
   else
   {
      btnMultiToggle.Text("MULTI:OFF");
      btnMultiToggle.ColorBackground(clrRed);
      btnMultiToggle.Color(clrWhite);
   }
}
```

Trading helper functions

These helpers encapsulate the low-level trade mechanics. PipSize(symbol) returns a pip magnitude (uses SYMBOL\_POINT \* 10.0 in your code), IsSymbolValid(symbol) checks if bid/ask exist, and TradeBuySingle()/TradeSellSingle() validate tradeability, lot boundaries, compute price/SL/TP using pip size, set filling type and submit the order through trade.Buy()/trade.Sell(). These helper functions centralize the order submission logic so the multipair loops just call them per symbol.

```
// Pip size, validation, and single-symbol trade helpers

double PipSize(string symbol)

{
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0) { Print("Invalid point size for ", symbol, ": Error=", GetLastError()); return 0; }
   return point * 10.0;
}
bool IsSymbolValid(string symbol)
{
   bool inMarketWatch = SymbolInfoDouble(symbol, SYMBOL_BID) > 0 && SymbolInfoDouble(symbol, SYMBOL_ASK) > 0;
   if(!inMarketWatch) Print("Symbol ", symbol, " invalid: Not in Market Watch or no valid bid/ask price.");
   return inMarketWatch;
}
bool TradeBuySingle(const string symbol)
{
   if(!IsSymbolValid(symbol)) return false;
   long tradeMode = SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
   if(tradeMode != SYMBOL_TRADE_MODE_FULL) { Print("TradeBuySingle: Skipping ", symbol, ": Trading disabled"); return false; }
   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   if(LotSize < minLot || LotSize > maxLot) { Print("TradeBuySingle: invalid lot"); return false; }

   double price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double sl = StopLoss > 0 ? price - StopLoss * PipSize(symbol) : 0;
   double tp = TakeProfit > 0 ? price + TakeProfit * PipSize(symbol) : 0;
   trade.SetTypeFillingBySymbol(symbol);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   price = NormalizeDouble(price, digits);
   sl    = NormalizeDouble(sl, digits);
   tp    = NormalizeDouble(tp, digits);

   if(trade.Buy(LotSize, symbol, price, sl, tp))
   { Print("Buy order placed on ", symbol, ": Ticket #", trade.ResultOrder()); return true; }
   else { Print("Buy order failed on ", symbol, ": Retcode=", trade.ResultRetcode()); return false; }
}
bool TradeSellSingle(const string symbol)
{
   if(!IsSymbolValid(symbol)) return false;
   long tradeMode = SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
   if(tradeMode != SYMBOL_TRADE_MODE_FULL) { Print("TradeSellSingle: Skipping ", symbol, ": Trading disabled"); return false; }

   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   if(LotSize < minLot || LotSize > maxLot) { Print("TradeSellSingle: invalid lot"); return false; }

   double price = SymbolInfoDouble(symbol, SYMBOL_BID);
   double sl = StopLoss > 0 ? price + StopLoss * PipSize(symbol) : 0;
   double tp = TakeProfit > 0 ? price - TakeProfit * PipSize(symbol) : 0;
   trade.SetTypeFillingBySymbol(symbol);

   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);

   price = NormalizeDouble(price, digits);
   sl    = NormalizeDouble(sl, digits);
   tp    = NormalizeDouble(tp, digits);

   if(trade.Sell(LotSize, symbol, price, sl, tp))
   { Print("Sell order placed on ", symbol, ": Ticket #", trade.ResultOrder()); return true; }
   else { Print("Sell order failed on ", symbol, ": Retcode=", trade.ResultRetcode()); return false; }
}
```

Main operations — how multipair commands are applied

Main operations (e.g., OpenBuyOrder, OpenSellOrder, CloseAllPositions, PlaceBuyStop, PlaceSellStop) accept the EA-provided inMajorPairs\[\] and inPairSelected\[\] arrays. When multiEnabled is true the routines iterate all indices and call the trade helpers using availablePairs\[i\] for every selected index. If multiEnabled is false the routine only trades the chart symbol. OpenBuyOrder/OpenSellOrder also track whether the chart symbol was already traded by checkboxes and fall back to trading the chart symbol if not — this guarantees user expectations when switching between chart-focused and multipair modes.

```
// OpenBuyOrder / OpenSellOrder excerpt (multipair iteration + fallback to chart symbol)

void OpenBuyOrder(string &inMajorPairs[], bool &inPairSelected[])

{
   Print("Starting OpenBuyOrder");
   string chartSym = Symbol();
   if(!multiEnabled)

   {
      PrintFormat("OpenBuyOrder: multipair disabled => trading only chart symbol %s", chartSym);
      if(SymbolInfoInteger(chartSym, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED) { Print("chart symbol not tradeable"); return; }
      TradeBuySingle(chartSym);
      return;
   }
   bool chartTraded = false;
   for(int i = 0; i < ArraySize(inMajorPairs); i++)
   {
      if(i < ArraySize(inPairSelected) && inPairSelected[i] && StringLen(availablePairs[i]) > 0)
      {
         string symbol = availablePairs[i];
         Print("Attempting Buy order on ", symbol, " (requested ", resolvedBases[i], ")");
         if(TradeBuySingle(symbol) && symbol == chartSym) chartTraded = true;
      }
      else
      {
         if(i < ArraySize(inMajorPairs)) Print("Skipping ", inMajorPairs[i], ": Not selected or unresolved");
      }
   }
   if(!chartTraded && SymbolInfoInteger(chartSym, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED)
   {
      PrintFormat("OpenBuyOrder: attempting BUY on chart symbol %s", chartSym);
      TradeBuySingle(chartSym);
   }
}
void OpenSellOrder(string &inMajorPairs[], bool &inPairSelected[])

{
   // (same structure as OpenBuyOrder but calling TradeSellSingle)
   // Implementation mirrors the buy flow but uses SELL specifics.
}
```

### Integrating the Multi-Symbols trading functionality with the New Headline EA

Where the multipair system is introduced (includes & inputs)

The EA pulls in the multipair-capable UI/trading header at the top and exposes an input to enable/disable multipair at start. This is the single place where the EA declares: (a) it will use the external multipair UI/trading object, and (b) the user can set initial multipair mode. This makes the feature opt-in and visible to the EA user.

```
#include <TradingButtons.mqh>             // header that implements the multipair UI & trading logic.
// ... other includes ...
input bool   EnableMultipair = true;      // initial multipair enabled state
```

We  need  to include the header and expose an input so users pick the expected behavior at initialization.

Where multipair data lives (majorPairs and selection flags)

The EA defines a majorPairs\[\] string array with the requested pair names and a parallel pairSelected\[\] boolean array that tracks which pairs are checked. These two arrays are the contract between the EA and the header: index i in both arrays refers to the same currency pair. The header builds checkboxes and uses the boolean array to know which pairs are selected.

```
// MULTIPAIR arrays (provided to the header)
// default major pairs (you can edit or later replace with resolved broker symbols)
string majorPairs[] = {"EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"};
bool   pairSelected[];
```

To keep a simple, index-aligned pair list + selection flags. It’s readable, easy to pass by reference, and makes synchronization straightforward.

Initialize selection defaults & pass inputs to header (OnInit setup)

During OnInit() the EA resizes pairSelected to match majorPairs and defaults every element to true. Then the EA configures the buttonsEA public parameters (lot size, stops, risk settings) and initializes the header by calling Init() and SetMultiEnabled(EnableMultipair). This ensures the header starts in the EA’s chosen mode and uses the same trade parameters.

```
// In OnInit()
ArrayResize(pairSelected, ArraySize(majorPairs));
for(int i = 0; i < ArraySize(pairSelected); i++) pairSelected[i] = true;

// Initialize TradingButtons
buttonsEA.LotSize = ButtonLotSize;
buttonsEA.StopLoss = ButtonStopLoss;
buttonsEA.TakeProfit = ButtonTakeProfit;
buttonsEA.StopOrderDistancePips = StopOrderDistancePips;
buttonsEA.RiskRewardRatio = RiskRewardRatio;
buttonsEA.Init();
buttonsEA.SetMultiEnabled(EnableMultipair); // pass initial multipair state
```

Synchronize configuration before initializing the UI/trading object — set public fields and the mode first, then call Init() so the header has correct runtime parameters.

Creating checkboxes (UI alignment) — CreatePairCheckboxes call

Here, we let EA calculate a checkboxY vertical offset (so checkboxes appear below the news lanes) and call CreatePairCheckboxes(majorPairs, pairSelected, checkboxY). This creates the checkboxes in the header while preserving index alignment with majorPairs. The header will also set each checkbox initial state from pairSelected\[\] so UI and EA are in sync. So this lets the UI component render the controls but pass EA arrays by reference. This keeps the EA as the authoritative store of which pairs exist and which are selected (the header manipulates the same arrays).

```
// create pair checkboxes aligned below the canvas lanes

int checkboxY = InpTopOffset + (InpSeparateLanes ? 8 : 28) * lineH + 6; // adjust +6 px margin if needed

buttonsEA.CreatePairCheckboxes(majorPairs, pairSelected, checkboxY);
```

Event routing — forwarding chart events to the header

The EA does not implement button click logic itself, instead it forwards all chart object clicks to the header by calling buttonsEA.HandleChartEvent(...) from OnChartEvent. This single-call contract simplifies the EA because the header takes responsibility for multipair toggle, checkbox clicks, and manual trade button actions.

```
// OnChartEvent: forward to the header with majorPairs and pairSelected arrays
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   // Header will handle multipair toggle and trade behaviour
   buttonsEA.HandleChartEvent(id, sparam, majorPairs, pairSelected);
}
```

By applying separate event routing , the EA becomes an event conduit while the header cleanly handles UI events and trading decisions. This keeps responsibilities well bounded.

How manual multipair operations are triggered (header’s role)

The header is the execution engine for multiplexed manual trades, supply simple arrays and let the header iterate and resolve symbols. That keeps the EA uncluttered. When a user clicks a Buy/Sell button or toggles multipair, the header’s HandleChartEvent uses the passed majorPairs & pairSelected arrays to decide where to act — e.g., it will iterate indices and trade only those where pairSelected\[i\] == true. The EA only supplies the arrays and configuration; the header performs resolution and trading across multiple symbols. (See the header for the per-symbol trade helpers and multipair iteration.)

```
// (conceptual) header receives arrays and performs per-index iteration:
// Pseudocode excerpt of header behavior (actual code in TradingButtons.mqh)
for(i = 0; i < ArraySize(majorPairs); i++)
{
   if(i < ArraySize(pairSelected) && pairSelected[i])
   {
      // resolve broker symbol for majorPairs[i]
      // call TradeBuySingle(resolvedSymbol) or TradeSellSingle(...)
   }
}
```

Automated order logic remains chart-specific (how multi-symbol coexists with automation)

Automated pre-event stop placement and post-impact orders in this EA operate on the chart symbol (\_Symbol) rather than on majorPairs. The multipair manual system is separate: manual multipair trades (buttons) and automated event-driven trades are distinct flows. This separation avoids accidental automated multi-symbol orders unless you explicitly extend the automation to use majorPairs.

```
// Example: automated BuyStop/SellStop placement uses _Symbol (chart symbol)
if(trade.BuyStop(InpOrderVolume, buyPrice, _Symbol, buySL, buyTP))
   ticketBuyStop = trade.ResultOrder();
if(trade.SellStop(InpOrderVolume, sellPrice, _Symbol, sellSL, sellTP))
   ticketSellStop = trade.ResultOrder();
```

From the above code, our key lesson is  to keep manual multipair controls separate from automated chart-specific strategies unless you intentionally want automation to act across many symbols. Clear separation prevents surprises.

Synchronization pattern: EA owns data, header owns UI/logic

The EA defines and stores majorPairs\[\] and pairSelected\[\] (the authoritative state). The header reads these (creates controls, acts on checked items) and writes back changes (checkbox clicks set pairSelected\[i\]), because arrays are passed by reference. This two-way synchronization pattern is simple and robust: the EA can inspect pairSelected\[\] at any time (e.g., in OnTimer) and the header updates it when the user interacts.

```
// EA owns arrays; header is given references and updates them when checkboxes are toggled
buttonsEA.CreatePairCheckboxes(majorPairs, pairSelected, checkboxY);
...
// header's HandleChartEvent updates pairSelected[] in-place when checkboxes are clicked
```

We use pass-by-reference for shared runtime state. It’s low-overhead and keeps UI and EA in sync without extra message passing.

Placement and visual layout considerations (boxes under news lanes)

The EA computes checkboxY based on the news lanes and configuration options (InpTopOffset, InpSeparateLanes, lineH) so the multipair checkboxes appear visually under the news / indicator lanes. Integrating UI elements from other modules is as much a layout task as a logic task — computing offsets dynamically means the UI adapts if lane heights or positions change.

```
// compute vertical position for checkboxes so they sit below the lanes
int checkboxY = InpTopOffset + (InpSeparateLanes ? 8 : 28) * lineH + 6;
buttonsEA.CreatePairCheckboxes(majorPairs, pairSelected, checkboxY);
```

When combining canvases and external UI panels, centralize layout math in the EA so both systems share consistent spacing rules.

### Testing

Deploying the EA on the MetaTrader 5 terminal produced excellent results. I was able to select the pairs I wanted to trade, and they responded instantly to the trading buttons. Orders executed at algorithmic speed, and with a single click I could close all positions across multiple pairs—an invaluable feature for news trading and other high-volatility scalping strategies.

The image below shows the outcome of the live chart testing process. It’s important to note that manual features require real-time interaction with the chart, while the automated components of the EA can be thoroughly evaluated in the Strategy Tester to ensure efficiency.

![Multiple Symbols Trading with the News Headline EA](https://c.mql5.com/2/164/NewsHeadlineEA.gif)

Multiple Symbols Trading with the News Headline EA

### Conclusion

Part IX marked another significant milestone in the evolution of the News Headline EA: the integration of multi-symbol trading. This advancement addresses a long-standing limitation by enabling traders to manage multiple pairs from a single chart. While not a fully automated trading system, this feature acts as a manual trading interface powered by algorithmic execution, delivering speed and precision while leaving decision-making to the trader during high-volatility conditions.

The development process itself was insightful, as we applied principles of modularization and separation of concerns to create a compact yet powerful system. What began as a simple calendar and news feed display has now grown into a versatile framework, merging manual and automated features to solve practical challenges faced by news traders. Although designed with major currency pairs in mind, the system can be adapted for custom symbols with minor compatibility adjustments.

One key challenge we encountered was broker-specific symbol naming. For instance, trades initially failed because the account used EURUSD.0 rather than the standard EURUSD. To overcome this, we enhanced the EA to adapt dynamically to broker nomenclature of major pairs.

We also leveraged the CCheckBox class from the MQL5 Standard Library to allow smooth pair selection, demonstrating the flexibility and expandability of the MQL5 language.

I hope this discussion provided useful insights and practical lessons. The full source code is attached below, use it along this article which documents the implementation. For convenience, I have also summarized the key takeaways in a tabular format below. Feedback and comments are always welcome.

### Key Lessons

| Key Lesson | Description |
| --- | --- |
| Index-aligned arrays | Both projects use majorPairs\[\] and pairSelected\[\] arrays aligned by index. This ensures a checkbox, a requested base name, and a resolved broker symbol all reference the same currency pair consistently. |
| Pass-by-reference synchronization | Arrays are passed by reference from the EA to TradingButtons.mqh, allowing the header to update selection states directly when checkboxes are toggled, keeping EA state instantly in sync. |
| Explicit event forwarding | The EA does not handle button clicks directly. Instead, OnChartEvent forwards events to buttonsEA.HandleChartEvent(), where the header interprets multipair toggles, checkbox clicks, and trading actions. |
| Symbol resolution abstraction | ResolveSymbol() in the header maps user-friendly bases (e.g., EURUSD) to broker-specific symbols (e.g., EURUSD.ecn). The EA can remain agnostic of broker naming quirks. |
| Separate manual vs. automated flows | In the EA, automated pre-event/post-event orders always act on the chart symbol, while multipair functionality is reserved for manual button actions. This separation avoids unexpected mass trades. |
| Dynamic UI creation & cleanup | The header dynamically creates/destroys checkboxes and buttons during Init()/Deinit(). The EA computes layout offsets (below news lanes) so components fit seamlessly into its UI. |
| Initialize before Init() | In OnInit(), the EA sets LotSize, StopLoss, TakeProfit, and EnableMultipair before calling buttonsEA.Init(). This ensures the header builds itself with the correct configuration. |
| Centralized trading helpers | Trading logic is encapsulated in reusable helpers like TradeBuySingle() and TradeSellSingle(). This avoids code duplication across multipair loops and button handlers. |
| Multipair toggle behavior | The btnMultiToggle button switches between single-symbol and multipair modes. In multipair mode, actions iterate all selected pairs; in single mode, actions apply only to the chart symbol. |
| Fallback to chart symbol | If multipair mode is enabled but the chart symbol is not selected, the header still ensures a trade is placed on the chart symbol. This provides predictable results for users focused on their chart. |
| Layout coordination | The EA calculates checkboxY to place multipair checkboxes neatly below the scrolling news canvas. This shows how to integrate third-party UI panels with custom indicator overlays without overlap. |
| Error logging & clarity | Both modules print detailed logs (e.g., symbol resolution failures, order placement retcodes). This traceability helps programmers and users quickly diagnose configuration issues. |
| Two-way synchronization pattern | Checkbox states are initialized from pairSelected\[\] (EA to UI), and when clicked, they update pairSelected\[\] (UI to EA). This continuous loop guarantees both modules share the same selection state. |
| Safe dynamic memory use | The header uses new CCheckBox() for each pair and carefully deletes them in Deinit(). This teaches MQL5 programmers how to manage GUI objects safely in longer-running EAs. |
| Extensibility through modularity | By encapsulating multipair trading into a standalone header, the same class can be reused in multiple EAs (like News Headline EA) without rewriting multipair logic — a scalable pattern for code reuse. |

### Attachments

| Filename | Version | Description |
| --- | --- | --- |
| News\_Headline\_EA.mq5 | 1.13 | An Expert Advisor that integrates economic calendar events and news headlines directly on the chart. It manages pre-event stop orders, post-impact trades, and displays scrolling news. Version 1.13 expands its functionality with multipair trading support through the TradingButtons module, allowing manual multipair order execution alongside automated chart-based event trading. |
| TradingButtons.mqh | 1 | A modular header that provides a multipair trading interface. It creates buttons for Buy, Sell, Close, Delete, and Stop orders, plus checkboxes for selecting multiple currency pairs. It includes symbol resolution logic, order placement helpers, and a toggle between single-symbol and multipair trading. Designed for reuse across different EAs, including News Headline EA. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19008.zip "Download all attachments in the single ZIP archive")

[News\_Headline\_EA.mq5](https://www.mql5.com/en/articles/download/19008/news_headline_ea.mq5 "Download News_Headline_EA.mq5")(62.58 KB)

[TradingButtons.mqh](https://www.mql5.com/en/articles/download/19008/tradingbuttons.mqh "Download TradingButtons.mqh")(38.59 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/493949)**
(2)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
17 Sep 2025 at 18:34

What's the real sense of this stuff? The provided reasoning is irrational, because EAs are capable of trading many symbols from any chart by design (out of the box), and you can easily switch chart between symbols without affecting EAs - they are not reloaded when the chart's symbol/timeframe is changed.

PS. The original comment is posted in English - please read it for proper understanding - autotranslation can produce ridiculous texts.

![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
6 Oct 2025 at 06:02

**Stanislav Korotky [#](https://www.mql5.com/en/forum/493949#comment_58058938):**

What's the real sense of this stuff? The provided reasoning is irrational, because EAs are capable of trading many symbols from any chart by design (out of the box), and you can easily switch chart between symbols without affecting EAs - they are not reloaded when the chart's symbol/timeframe is changed.

PS. The original comment is posted in English - please read it for proper understanding - autotranslation can produce ridiculous texts.

Hi Stanislav Korotky,

Thank you for sharing your perspective. I completely understand your point — indeed, an EA can trade multiple symbols from any single chart, and switching symbols manually does not reload or disrupt the running EA.

However, my idea specifically targets situations where simultaneous execution across multiple pairs is required — for example, during high-impact news events when you may want to place synchronized orders on both GBPUSD and EURUSD at the exact same moment. In such cases, manual symbol switching isn’t practical.

That’s why I emphasize programmatic management of multiple symbols — ensuring the EA can handle and execute trades across selected pairs automatically, even if it’s attached to a chart with a different base symbol.

![Statistical Arbitrage Through Cointegrated Stocks (Part 3): Database Setup](https://c.mql5.com/2/165/19242-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 3): Database Setup](https://www.mql5.com/en/articles/19242)

This article presents a sample MQL5 Service implementation for updating a newly created database used as source for data analysis and for trading a basket of cointegrated stocks. The rationale behind the database design is explained in detail and the data dictionary is documented for reference. MQL5 and Python scripts are provided for the database creation, schema initialization, and market data insertion.

![Reimagining Classic Strategies (Part 15): Daily Breakout Trading Strategy](https://c.mql5.com/2/165/19130-reimagining-classic-strategies-logo__1.png)[Reimagining Classic Strategies (Part 15): Daily Breakout Trading Strategy](https://www.mql5.com/en/articles/19130)

Human traders had long participated in financial markets before the rise of computers, developing rules of thumb that guided their decisions. In this article, we revisit a well-known breakout strategy to test whether such market logic, learned through experience, can hold its own against systematic methods. Our findings show that while the original strategy produced high accuracy, it suffered from instability and poor risk control. By refining the approach, we demonstrate how discretionary insights can be adapted into more robust, algorithmic trading strategies.

![Introduction to MQL5 (Part 20): Introduction to Harmonic Patterns](https://c.mql5.com/2/165/19179-introduction-to-mql5-part-20-logo.png)[Introduction to MQL5 (Part 20): Introduction to Harmonic Patterns](https://www.mql5.com/en/articles/19179)

In this article, we explore the fundamentals of harmonic patterns, their structures, and how they are applied in trading. You’ll learn about Fibonacci retracements, extensions, and how to implement harmonic pattern detection in MQL5, setting the foundation for building advanced trading tools and Expert Advisors.

![From Basic to Intermediate: Template and Typename (IV)](https://c.mql5.com/2/114/Do_bgsico_ao_intermedikrio__Template_e_Typename_I___LOGO.png)[From Basic to Intermediate: Template and Typename (IV)](https://www.mql5.com/en/articles/15670)

In this article, we will take a very close look at how to solve the problem posed at the end of the previous article. There was an attempt to create a template of such type so that to be able to create a template for data union.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/19008&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083459524515339295)

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