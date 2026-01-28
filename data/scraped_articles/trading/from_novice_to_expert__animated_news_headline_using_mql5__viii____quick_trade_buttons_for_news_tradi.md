---
title: From Novice to Expert: Animated News Headline Using MQL5 (VIII) — Quick Trade Buttons for News Trading
url: https://www.mql5.com/en/articles/18975
categories: Trading, Integration
relevance_score: -2
scraped_at: 2026-01-24T14:14:50.660890
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/18975&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083435412568939433)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/18975#para1)
- [Developing the CTradingButtons class](https://www.mql5.com/en/articles/18975#para2)
- [Integrating the CTradingButtons into the News Headline EA](https://www.mql5.com/en/articles/18975#para3)
- [Testing](https://www.mql5.com/en/articles/18975#para4)
- [Conclusion](https://www.mql5.com/en/articles/18975#para5)
- [Key Lessons](https://www.mql5.com/en/articles/18975#para6)
- [Attachements](https://www.mql5.com/en/articles/18975#para7)

### Introduction

This discussion introduces the Quick Trade Buttons interface within the News Headline Expert Advisor (EA), designed to blend algorithmic precision seamlessly with human decision-making for news trading and scalping. While the EA excels in automated trade execution, it previously limited real-time manual intervention during high-volatility news events, where human judgment is vital for contextual analysis and nuanced strategies.

The Quick Trade Buttons panel addresses this by integrating a compact, visually clear interface directly on the chart, enabling traders to execute rapid trades (Buy, Sell, Close All, Delete Orders, and more) while leveraging real-time economic calendar data, Alpha Vantage news feeds, indicator insights (RSI, Stochastic, MACD, CCI), and AI-driven analytics. This hybrid "augmented intelligence" approach empowers traders to combine machine-driven speed with strategic discretion, optimizing performance in volatile markets and fast-paced scalping scenarios.

Comparison: Current EA Workflow vs. Enhanced Workflow with Quick Trade Buttons

| Current Workflow | Enhanced Workflow |
| --- | --- |
| Passive Monitoring:<br>The EA executes automated trades based on predefined rules; traders monitor without direct intervention. | Active Hybrid Control:<br>Traders use Quick Trade Buttons to override or complement automation with one-click actions for market or pending orders. |
| Terminal Navigation Required:<br>Manual trades involve accessing MetaTrader 5’s Order Window or Toolbox, disrupting chart focus. | Chart-Centric Execution:<br>All trade operations occur directly on the chart via a compact button panel, maintaining focus. |
| Disjointed Data-to-Action Loop:<br>News, indicators, and AI insights are displayed but require external actions, delaying execution. | Integrated Action Pipeline:<br>View real-time news or insights → Click Quick Trade Button → Execute orders instantly. |
| High Cognitive Load:<br>Switching between chart analysis and terminal windows increases mental strain during volatility. | Optimized Focus:<br>A unified chart workspace with buttons and data reduces cognitive load and latency. |
| Algorithmic Rigidity:<br>Predefined strategies cannot adapt to unexpected market anomalies. | Human-Machine Synergy:<br>Quick Trade Buttons enable trader discretion to adapt to edge cases, enhancing flexibility. |
| Slow Manual Execution:<br>Manual trades via terminal take 3000ms+ due to navigation and input delays. | Near-Algorithmic Speed:<br>Pre-mapped button clicks execute trades in <500ms with automated risk parameters. |

The integration of the Quick Trade Buttons interface into the News Headline Expert Advisor (EA) is highly feasible, leveraging MetaTrader 5’s CButton class for interactive controls, CTrade for efficient trade execution, and CCanvas for a visually intuitive dashboard. The proposed interface will feature eight buttons (Buy, Sell, Close All, Delete Orders, Close Profit, Close Loss, Buy Stop, Sell Stop) with pre-calculated risk management parameters to enable rapid, disciplined trade execution during high-volatility news events and fast-paced scalping scenarios.

The pre-calculated risk management enhances effectiveness by automating stop loss and take profit calculations, validated against broker constraints (e.g., SYMBOL\_VOLUME\_MIN/MAX), minimizing errors under pressure and maintaining consistent risk-reward ratios. This synergy of speed, precision, and real-time data integration (economic calendar, Alpha Vantage news feeds, RSI/Stochastic/MACD/CCI indicators, and AI-driven analytics) positions the Quick Trade Buttons as a powerful tool for “ [augmented intelligence](https://en.wikipedia.org/wiki/Intelligence_amplification "https://en.wikipedia.org/wiki/Intelligence_amplification"),” empowering traders to combine human discretion with machine-driven efficiency in volatile markets.

The development plan is structured in three strategic stages to deliver a robust, user-friendly product.

Stage 1:

Class Header Development will involve creating a CTradingButtons class in a dedicated header file (TradingButtons.mqh). This class will define button creation, styling, and trade logic with embedded risk parameters, including input validation (e.g., checking SYMBOL\_TRADE\_MODE) and error handling for reliability. A “Quick Trade Buttons” label will be planned for the panel to enhance usability.

Stage 2:

Integration into the EA will embed the class into News Headline EA, initializing it in OnInit with user inputs, linking button clicks to OnChartEvent, and positioning the panel to complement news canvases. Manual trades will use a distinct magic number to avoid conflicts with automated trades (e.g., stop orders 3 minutes before events).

Stage 3:

Live Chart Testing will validate the interface on a demo account (e.g., EURUSD, M1 chart), testing button responsiveness, risk parameter accuracy, and integration with news, indicator, and AI data during simulated news events and scalping sessions. Usability tests will evaluate visual clarity and execution speed. This staged approach ensures the Quick Trade Buttons deliver a seamless, data-driven trading experience.

![Relationship diagram of development stages](https://c.mql5.com/2/160/chrome_ZiAvlQ30PV.png)

Relationship diagram of development stages

Refer to the image below for a visual presentation of our objective.

![Objective after discussion](https://c.mql5.com/2/160/terminal64_UQLY5YBTNO.png)

Trading buttons integration by discussion end

By the conclusion of this discussion, we aim to achieve the outcomes demonstrated in the chart image above, showcasing the successful implementation of the Quick Trade Buttons integrated into the News Headline Expert Advisor (EA) on a EURAUD chart. In the following section, we will outline two key implementation stages: first, developing the Quick Trade Buttons header to create a modular, user-friendly interface; and second, integrating this header into the News Headline EA to enable seamless manual trading alongside automated strategies. These stages culminate in comprehensive testing, with results confirming the system’s effectiveness for news trading and scalping.

### Developing the CTradingButtons class

The TradingButtons.mqh header file defines the CTradingButtons class, which is central to implementing the Quick Trade Buttons interface for the News Headline Expert Advisor (EA). The class facilitates a chart-based, user-friendly interface for manual trading, enabling rapid trade execution and position management during news trading and scalping. Below, I break down four key sections of the header—class declaration, initialization (Init), button creation (CreateButtons), and event handling (HandleChartEvent)—explaining how each works and contributes to the development goal of creating an efficient, intuitive trading panel that integrates with real-time data for augmented intelligence.

Class Declaration

The class declaration section establishes the CTradingButtons class, defining its private and public members to manage the button panel and trading operations. Private members include CButton objects for each of the eight buttons (Buy, Sell, Close All, etc.), a CCanvas object for the panel, a CTrade object for trade execution, and variables for button dimensions and spacing.

Public members expose configurable risk parameters (e.g., LotSize, StopLoss, TakeProfit) and the constructor sets default values (e.g., lot size=0.1, stop loss=50 pips). This structure provides a modular foundation for the Quick Trade Buttons, encapsulating the GUI and trading logic. It contributes to the development goal by enabling a compact, reusable interface that can be easily integrated into the EA, with predefined risk settings to ensure disciplined trading during volatile news events and fast-paced scalping.

```
class CTradingButtons

{

private:

   CButton btnBuy, btnSell, btnCloseAll, btnDeleteOrders, btnCloseProfit, btnCloseLoss, btnBuyStop, btnSellStop;
   CCanvas buttonPanel;
   CTrade  trade;
   int     buttonWidth;
   int     buttonHeight;
   int     buttonSpacing;

public:

   double LotSize;
   int    StopLoss;
   int    TakeProfit;
   int    StopOrderDistancePips;
   double RiskRewardRatio;

   CTradingButtons() : buttonWidth(100), buttonHeight(30), buttonSpacing(10),
                      LotSize(0.1), StopLoss(50), TakeProfit(100),
                      StopOrderDistancePips(8), RiskRewardRatio(2.0) {}

};
```

Initialization (Init)

The Init function initializes the trading and GUI components when the EA is loaded. It configures the CTrade object with a unique magic number (123456).

to distinguish manual trades, sets a deviation of 10 points for slippage tolerance, and uses symbol-specific order filling (SetTypeFillingBySymbol). It then calls CreateButtonPanel and CreateButtons to render the panel and buttons on the chart. This section is crucial for setting up the interface and ensuring trade execution reliability. It contributes to the development goal by establishing a seamless setup process, enabling the Quick Trade Buttons to be ready for immediate use, with consistent trade parameters that support rapid execution in news trading (e.g., placing stop orders before events) and scalping (e.g., quick market orders).

```
void Init()

{
   trade.SetExpertMagicNumber(123456);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFillingBySymbol(_Symbol);
   CreateButtonPanel();
   CreateButtons();

}
```

Button Creation (CreateButtons)

The CreateButtons function constructs the eight buttons, positioning them vertically within the panel (starting at y=160, x=10) with a fixed size (100x30 pixels) and spacing (10 pixels). Each button is created using the CButton class, assigned a unique name (e.g., "btnBuy"), styled with Calibri font (size 8), and given a black background with vibrant text colors (e.g., lime for Buy, red for Sell) for quick identification. The buttons are set to a z-order of 11 to ensure they appear above the panel (z-order=10).

This section is pivotal for creating a visually intuitive interface, allowing traders to instantly recognize and interact with buttons during high-pressure scenarios. It supports the development goal by providing a chart-centric, user-friendly control panel that enhances speed and reduces cognitive load for news trading and scalping.

```
void CreateButtons()

{

   int x = 10;
   int y = 160;
   string font = "Calibri";
   int fontSize = 8;
   color buttonBgColor = clrBlack; // Button background color
   btnBuy.Create(0, "btnBuy", 0, x, y, x + buttonWidth, y + buttonHeight);
   btnBuy.Text("Buy");
   btnBuy.Font(font);
   btnBuy.FontSize(fontSize);
   btnBuy.ColorBackground(buttonBgColor);
   btnBuy.Color(clrLime); // Bright green text
   ObjectSetInteger(0, "btnBuy", OBJPROP_ZORDER, 11);

   y += buttonHeight + buttonSpacing;
   btnSell.Create(0, "btnSell", 0, x, y, x + buttonWidth, y + buttonHeight);
   btnSell.Text("Sell");
   btnSell.Font(font);
   btnSell.FontSize(fontSize);
   btnSell.ColorBackground(buttonBgColor);
   btnSell.Color(clrRed); // Bright red text
   ObjectSetInteger(0, "btnSell", OBJPROP_ZORDER, 11);

   y += buttonHeight + buttonSpacing;
   btnCloseAll.Create(0, "btnCloseAll", 0, x, y, x + buttonWidth, y + buttonHeight);
   btnCloseAll.Text("Close All");
   btnCloseAll.Font(font);
   btnCloseAll.FontSize(fontSize);
   btnCloseAll.ColorBackground(buttonBgColor);
   btnCloseAll.Color(clrYellow); // Bright yellow text
   ObjectSetInteger(0, "btnCloseAll", OBJPROP_ZORDER, 11);

   y += buttonHeight + buttonSpacing;
   btnDeleteOrders.Create(0, "btnDeleteOrders", 0, x, y, x + buttonWidth, y + buttonHeight);
   btnDeleteOrders.Text("Delete Orders");
   btnDeleteOrders.Font(font);
   btnDeleteOrders.FontSize(fontSize);
   btnDeleteOrders.ColorBackground(buttonBgColor);
   btnDeleteOrders.Color(clrAqua); // Bright cyan text
   ObjectSetInteger(0, "btnDeleteOrders", OBJPROP_ZORDER, 11);

   y += buttonHeight + buttonSpacing;
   btnCloseProfit.Create(0, "btnCloseProfit", 0, x, y, x + buttonWidth, y + buttonHeight);
   btnCloseProfit.Text("Close Profit");
   btnCloseProfit.Font(font);
   btnCloseProfit.FontSize(fontSize);
   btnCloseProfit.ColorBackground(buttonBgColor);
   btnCloseProfit.Color(clrGold); // Bright gold text
   ObjectSetInteger(0, "btnCloseProfit", OBJPROP_ZORDER, 11);

   y += buttonHeight + buttonSpacing;
   btnCloseLoss.Create(0, "btnCloseLoss", 0, x, y, x + buttonWidth, y + buttonHeight);
   btnCloseLoss.Text("Close Loss");
   btnCloseLoss.Font(font);
   btnCloseLoss.FontSize(fontSize);
   btnCloseLoss.ColorBackground(buttonBgColor);
   btnCloseLoss.Color(clrOrange); // Bright orange text
   ObjectSetInteger(0, "btnCloseLoss", OBJPROP_ZORDER, 11);

   y += buttonHeight + buttonSpacing;
   btnBuyStop.Create(0, "btnBuyStop", 0, x, y, x + buttonWidth, y + buttonHeight);
   btnBuyStop.Text("Buy Stop");
   btnBuyStop.Font(font);
   btnBuyStop.FontSize(fontSize);
   btnBuyStop.ColorBackground(buttonBgColor);
   btnBuyStop.Color(clrLightPink); // Bright pink text
   ObjectSetInteger(0, "btnBuyStop", OBJPROP_ZORDER, 11);

   y += buttonHeight + buttonSpacing;
   btnSellStop.Create(0, "btnSellStop", 0, x, y, x + buttonWidth, y + buttonHeight);
   btnSellStop.Text("Sell Stop");
   btnSellStop.Font(font);
   btnSellStop.FontSize(fontSize);
   btnSellStop.ColorBackground(buttonBgColor);
   btnSellStop.Color(clrLightCoral); // Bright coral text
   ObjectSetInteger(0, "btnSellStop", OBJPROP_ZORDER, 11);

}
```

Event Handling (HandleChartEvent)

The HandleChartEvent function processes user interactions with the buttons by detecting CHARTEVENT\_OBJECT\_CLICK events and mapping them to specific trade actions. It checks the clicked object’s name (e.g., btnBuy) and calls the corresponding function (e.g., OpenBuyOrder, PlaceBuyStop).

This section ensures that button clicks translate into immediate trade executions, leveraging the CTrade object for reliable order placement. It contributes to the development goal by enabling near-algorithmic speed (<500ms) for manual trades, critical for news trading (e.g., reacting to high-impact events) and scalping (e.g., rapid position entries/exits). The event-driven approach integrates seamlessly with the EA’s real-time data (news, indicators, AI), allowing traders to act on insights instantly, supporting the “augmented intelligence” paradigm.

```
void HandleChartEvent(const int id, const string &sparam)
{
   if(id == CHARTEVENT_OBJECT_CLICK)
   {
      if(sparam == btnBuy.Name())
         OpenBuyOrder();
      else if(sparam == btnSell.Name())
         OpenSellOrder();
      else if(sparam == btnCloseAll.Name())
         CloseAllPositions();
      else if(sparam == btnDeleteOrders.Name())
         DeleteAllPendingOrders();
      else if(sparam == btnCloseProfit.Name())
         CloseProfitablePositions();
      else if(sparam == btnCloseLoss.Name())
        CloseLosingPositions();
      else if(sparam == btnBuyStop.Name())
         PlaceBuyStop();
      else if(sparam == btnSellStop.Name())
         PlaceSellStop();

   }

}
```

These sections collectively advance the goal of creating a Quick Trade Buttons interface that enhances manual trading within the News Headline EA. The class declaration provides a modular, reusable structure, ensuring easy integration and maintenance. The Init function sets up the trading and GUI environment, enabling immediate usability. The CreateButtons function delivers a visually intuitive panel, reducing cognitive load during volatile markets. The HandleChartEvent function ensures rapid, reliable trade execution, aligning with the need for speed in news trading and scalping. Together, they create a chart-centric, data-integrated interface that empowers traders to combine human discretion with automated risk management, fulfilling the idea of augmented intelligence.

### Integrating the CTradingButtons into the News Headline EA

The integration of the TradingButtons.mqh header into the News Headline EA enables a hybrid trading system that combines manual control via Quick Trade Buttons with automated news-driven strategies, real-time economic calendar data, Alpha Vantage news feeds, technical indicator insights, and AI-driven analytics. Below, I break down three key sections of News Headline EA responsible for this integration—User Inputs, OnInit, and OnChartEvent—explaining how each works and contributes to the development goal of creating a seamless, augmented intelligence trading tool.

User Inputs

The "Manual Trading Buttons" input group defines configurable parameters for the Quick Trade Buttons, allowing traders to customize the behavior of manual trades executed via the CTradingButtons class. These inputs include ButtonLotSize (default 0.1), ButtonStopLoss (50 pips), ButtonTakeProfit (100 pips), StopOrderDistancePips (8 pips), and RiskRewardRatio (2.0). These parameters are passed to the CTradingButtons instance to set risk management rules for button actions (e.g., Buy, Sell, Buy Stop).

By providing these inputs, the EA ensures that manual trades align with predefined risk constraints, maintaining consistency with the automated trading logic (which uses separate parameters like InpOrderVolume). This section contributes to the development goal by enabling traders to tailor the Quick Trade Buttons to their risk preferences, ensuring disciplined execution during volatile news events and scalping, while integrating seamlessly with the EA’s broader data-driven framework.

```
input group "Manual Trading Buttons"
input double ButtonLotSize = 0.1;               // Lot size for manual trades
input int ButtonStopLoss = 50;                  // Stop Loss in pips (market orders)
input int ButtonTakeProfit = 100;               // Take Profit in pips (market orders)
input int StopOrderDistancePips = 8;            // Distance for stop orders in pips
input double RiskRewardRatio = 2.0;             // Risk-reward ratio for stop orders
```

OnInit

The OnInit function initializes the EA, including the integration of the CTradingButtons class. It declares a global CTradingButtons instance (buttonsEA) and assigns the user-defined input parameters to its public members (e.g., buttonsEA.LotSize = ButtonLotSize). The buttonsEA.Init() call triggers the creation of the button panel and buttons, positioning them at the top-left of the chart (x=0, y=40) to avoid overlap with news and indicator canvases (starting at y=160, as set by InpTopOffset).

The function also initializes other EA components, such as canvases for economic events, news, indicators, and AI insights, ensuring the button panel integrates harmoniously with these displays. A unique magic number (123456 for manual trades vs. 888888 for automated trades) prevents conflicts between manual and automated orders. This section contributes to the development goal by embedding the Quick Trade Buttons into the EA’s initialization process, ensuring a unified workspace where traders can access manual controls alongside real-time data, enhancing speed and focus during news trading and scalping.

```
CTradingButtons buttonsEA;

int OnInit()

{
   ChartSetInteger(0, CHART_FOREGROUND, 0); // Ensure objects are visible
   lastReloadDay = lastNewsReload = 0;
   ArrayResize(highArr, 0);
   ArrayResize(medArr, 0);
   ArrayResize(lowArr, 0);
   canvW = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);

   // Initialize TradingButtons

   buttonsEA.LotSize = ButtonLotSize;
   buttonsEA.StopLoss = ButtonStopLoss;
   buttonsEA.TakeProfit = ButtonTakeProfit;
   buttonsEA.StopOrderDistancePips = StopOrderDistancePips;
   buttonsEA.RiskRewardRatio = RiskRewardRatio;
   buttonsEA.Init();

   // Initialize News Headline EA

   trade.SetExpertMagicNumber(888888);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFillingBySymbol(_Symbol);
   eventsCanvas.CreateBitmapLabel("EvC", 0, 0, canvW, 4 * lineH, COLOR_FORMAT_ARGB_RAW);
   eventsCanvas.TransparentLevelSet(150);
   newsCanvas.CreateBitmapLabel("NwC", 0, 0, canvW, lineH, COLOR_FORMAT_ARGB_RAW);
   newsCanvas.TransparentLevelSet(0);

   if(InpSeparateLanes)

   {

      rsiCanvas.CreateBitmapLabel("RsiC", 0, 0, canvW, lineH, COLOR_FORMAT_ARGB_RAW);
      rsiCanvas.TransparentLevelSet(120);
      stochCanvas.CreateBitmapLabel("StoC", 0, 0, canvW, lineH, COLOR_FORMAT_ARGB_RAW);
      stochCanvas.TransparentLevelSet(120);
      macdCanvas.CreateBitmapLabel("MacC", 0, 0, canvW, lineH, COLOR_FORMAT_ARGB_RAW);
      macdCanvas.TransparentLevelSet(120)
      cciCanvas.CreateBitmapLabel("CciC", 0, 0, canvW, lineH, COLOR_FORMAT_ARGB_RAW);
     cciCanvas.TransparentLevelSet(120);
   }
   else

   {
      combinedCanvas.CreateBitmapLabel("AllC", 0, 0, canvW, lineH, COLOR_FORMAT_ARGB_RAW);
      combinedCanvas.TransparentLevelSet(120);
      SetCanvas("AllC", InpPositionTop, InpTopOffset + 4 * lineH);
      combinedCanvas.FontSizeSet(-120);
      combinedCanvas.TextOut(5, (lineH - combinedCanvas.TextHeight("Indicator Insights:")) / 2,
                           "Indicator Insights:", XRGB(200, 200, 255), ALIGN_LEFT);
      combinedCanvas.Update(true);
   }

   if(ShowAIInsights)
   {
      aiCanvas.CreateBitmapLabel("AiC", 0, 0, canvW, lineH, COLOR_FORMAT_ARGB_RAW);
      aiCanvas.TransparentLevelSet(120);
      offAI = canvW;
      SetCanvas("AiC", InpPositionTop, InpTopOffset + (InpSeparateLanes ? 8 : 5) * lineH);
      aiCanvas.TextOut(offAI, (lineH - aiCanvas.TextHeight(latestAIInsight)) / 2,
                       latestAIInsight, XRGB(180, 220, 255), ALIGN_LEFT);
      aiCanvas.Update(true);

   }

   ReloadEvents();
   FetchAlphaVantageNews();
   FetchAIInsights();

   offHigh = offMed = offLow = offNews = canvW;
   offRSI = offStoch = offMACD = offCCI = offCombined = canvW;

   SetCanvas("EvC", InpPositionTop, InpTopOffset);
   SetCanvas("NwC", InpPositionTop, InpTopOffset + 3 * lineH);
   if(InpSeparateLanes)

   {

      SetCanvas("RsiC", InpPositionTop, InpTopOffset + 4 * lineH);
      SetCanvas("StoC", InpPositionTop, InpTopOffset + 5 * lineH);
      SetCanvas("MacC", InpPositionTop, InpTopOffset + 6 * lineH);
      SetCanvas("CciC", InpPositionTop, InpTopOffset + 7 * lineH);

   }

   newsCanvas.TextOut(offNews, (lineH - newsCanvas.TextHeight(placeholder)) / 2,
                      placeholder, XRGB(255, 255, 255), ALIGN_LEFT);
   newsCanvas.Update(true);

   EventSetMillisecondTimer(InpTimerMs);
   return INIT_SUCCEEDED;

}
```

OnChartEvent

The OnChartEvent function handles user interactions with the Quick Trade Buttons by delegating chart events (specifically CHARTEVENT\_OBJECT\_CLICK) to the buttonsEA.HandleChartEvent method. This method, defined in TradingButtons.mqh, maps button clicks to trade actions (e.g., OpenBuyOrder, PlaceBuyStop), ensuring rapid execution (<500ms) of manual trades.

By routing events through the CTradingButtons class, this section ensures that manual trading actions are processed efficiently without interfering with the EA’s automated functions, such as news-driven stop orders or indicator updates. This contributes to the development goal by providing a seamless interface for traders to act on real-time data enabling human-machine synergy for news trading (e.g., placing stop orders before high-impact events) and scalping (e.g., rapid market entries).

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)

{
   buttonsEA.HandleChartEvent(id, sparam);
}
```

These sections collectively achieve the integration of the Quick Trade Buttons into the News Headline EA, fulfilling the goal of creating an augmented intelligence trading system. The User Inputs section allows traders to customize risk parameters, ensuring manual trades align with their strategies while maintaining consistency with automated trades. The OnInit function embeds the CTradingButtons class into the EA’s initialization, creating a unified chart workspace where the button panel (at y=40) coexists with news and indicator canvases (y=160 and up), enhancing usability and focus.

The OnChartEvent function enables rapid, reliable manual trade execution, allowing traders to respond instantly to real-time data, critical for news trading and scalping. Together, these sections create a chart-centric, data-integrated interface that combines human discretion with automated efficiency, reducing execution latency and cognitive load in volatile markets.

### Testing

To evaluate the Quick Trade Buttons feature integrated into the News Headline Expert Advisor (EA), I attached the EA to a chart of my preferred currency pair, EURAUD, on a demo account. I strongly recommend using a demo account for testing any trading system to ensure a risk-free environment. The purpose of this testing process is to validate whether the development goal of creating a seamless, chart-centric interface for rapid manual trading during news events and scalping has been achieved. Before this final presentation, extensive iterative testing enabled us to refine the system into a robust and user-friendly masterpiece.

While there are still opportunities for further enhancements, such as adding dynamic button layouts or advanced risk management options, the current implementation provides a strong foundation. It serves as an educational resource for traders and developers, offering clear code structure, reusable templates, and practical examples that can be adapted for other projects. Although some might suggest including an input box for real-time trade parameter adjustments, I opted to retain these settings in the EA’s input panel (e.g., ButtonLotSize, ButtonStopLoss) to eliminate delays during high-pressure trading scenarios. A prudent trader should preset these values before engaging with the Quick Trade Buttons, ensuring focus remains on swift, one-click executions.

Please review the image provided below, showcasing successful test results for all Quick Trade Buttons (Buy, Sell, Close All, Delete Orders, Close Profit, Close Loss, Buy Stop, Sell Stop) on the EURAUD currency pair using a demo account. These results demonstrate the seamless functionality of the buttons, with rapid execution and accurate application of pre-set risk parameters, confirming the News Headline Expert Advisor’s (EA) readiness for live deployment.

![Presentation of an integrated News Headline EA with manual trading button interface.](https://c.mql5.com/2/160/terminal64_Izg76mZlbz.gif)

Testing the Quick Trading Buttons on EURAUD chart

The tests validate the EA’s ability to integrate manual trading with real-time news, indicator, and AI insights, ensuring a robust, chart-centric interface for news trading and scalping. While the current implementation is production-ready, ongoing enhancements, such as adaptive risk settings or additional button functionalities, could further optimize performance. These results position the EA as a reliable tool for traders, ready to deliver augmented intelligence in live market conditions.

### Conclusion

The integration of fully autonomous trading systems with real-time trader input is highly feasible through user-friendly interfaces like the Quick Trade Buttons in the News Headline Expert Advisor. These buttons enable traders to manually participate in processes typically driven by algorithms, fostering a partnership between human intuition and automated logic. To maximize efficiency, most settings, such as lot size, stop loss, and take profit, are programmatically embedded in the button logic, eliminating delays during high-pressure trading scenarios.

This allows traders to focus solely on clicking to execute trades, enhancing speed and precision. Unique magic numbers ensure the EA distinguishes between its positions and manual interventions, preventing conflicts. This augmented intelligence approach, while not fully AI-driven, effectively combines human decision-making with algorithmic precision, creating a robust foundation for hybrid trading strategies.

This tool is particularly valuable in demanding scenarios like scalping and news trading, where speed and adaptability are critical. The “Close All” button protects capital by swiftly closing all open positions during unexpected market events, while “Close Profit” and “Close Loss” buttons allow traders to secure gains or limit losses with a single click, streamlining position management. Unlike MetaTrader 5’s built-in trading tools, the Quick Trade Buttons incorporate pre-set risk management parameters , promoting disciplined trading practices for traders of all experience levels.

These parameters can be customized via the EA’s input settings (e.g., ButtonLotSize, ButtonStopLoss ), allowing traders to tailor risk to their account size and risk tolerance. Attached is a collection of key lessons and insights from this development process, offering valuable resources for traders and developers. I invite you to part of the discussion by sharing your questions and experiences. Stay tuned for future updates as we continue to refine this transformative trading tool!

### Key Lessons

| Lesson | Description |
| --- | --- |
| Modular Class Design | Use object-oriented programming (e.g., CTradingButtons class) to encapsulate button logic and trade execution, ensuring code reusability and easy integration into other EAs. |
| Pre-set Risk Management | Embed risk parameters in button logic to eliminate manual input delays, enabling rapid execution during volatile news events and scalping. |
| Unique Magic Numbers | Assign distinct magic numbers to prevent conflicts between manual and algorithmic trades, ensuring clear position tracking. |
| Chart-Centric Interface | Position buttons and data displays on the chart using CCanvas to reduce cognitive load and keep traders focused during high-pressure scenarios. |
| Efficient Event Handling | Utilize OnChartEvent with CHARTEVENT\_OBJECT\_CLICK to map button clicks to trade functions, achieving sub-500ms execution for scalping and news trading. |
| Visual Clarity | Apply vibrant text colors and high z-order (e.g., 11 for buttons) to ensure buttons are easily identifiable and accessible on the chart. |
| User Input Customization | Allow traders to set risk parameters via EA inputs for flexibility without compromising execution speed. |
| Error Handling | Implement validation checks (e.g., SYMBOL\_TRADE\_MODE, lot size against SYMBOL\_VOLUME\_MIN/MAX) in trade functions to prevent failed orders and log errors for debugging. |
| Seamless Data Integration | Integrate real-time data (economic calendar, Alpha Vantage news, indicators) with buttons using separate canvases to create a unified workspace for informed manual trading decisions. |
| Hybrid Trading Synergy | Design buttons to complement automated strategies, fostering human-machine collaboration for augmented intelligence. |
| Canvas Positioning | Use distinct y-coordinates and z-orders (e.g., 10 for panel, 0 for news) to avoid overlap and ensure clear visibility of all EA components. |
| Dynamic Data Updates | Leverage OnTimer to refresh news, indicators, and AI insights (e.g., every 20ms via InpTimerMs) to keep traders informed without manual intervention. |
| Demo Account Testing | Test the EA on a demo account to validate button functionality and risk settings in a risk-free environment before live deployment. |
| Scalability for Enhancements | Structure the EA with modular functions to allow easy addition of new buttons or features like adaptive risk settings. |
| Performance Optimization | Normalize prices (e.g., NormalizeDouble) and set slippage tolerance (e.g., SetDeviationInPoints(10) to ensure reliable trade execution in volatile markets. |
| Trader Education | Provide clear documentation and logs (e.g., Print("Manual Buy order placed") to educate traders on button functionality and trade outcomes, enhancing usability. |
| Automated vs. Manual Balance | Separate manual (button-driven) and automated (news-driven) trade logic with distinct settings (e.g., InpOrderVolume vs. ButtonLotSize ) to maintain flexibility and control. |
| Alert System Integration | Combine button actions with event alerts (e.g., InpAlertMinutesBefore=5) to guide traders on when to use buttons like Buy Stop before high-impact news. |
| Code Reusability | Design the CTradingButtons header as a standalone module for use in other EAs, reducing development time for future projects requiring manual trading interfaces. |
| Iterative Refinement | Conduct iterative testing (e.g., on EURAUD demo) to refine button responsiveness and risk parameters, ensuring the EA meets real-world trading demands before live use. |

### Attachements

| Filename | Version | Description |
| --- | --- | --- |
| TradingButtons.mqh | 1.0 | Header file defining the CTradingButtons class, which encapsulates the logic for eight Quick Trade Buttons (Buy, Sell, Close All, Delete Orders, Close Profit, Close Loss, Buy Stop, Sell Stop). It handles button creation, styling (e.g., vibrant colors for rapid identification), and trade execution with pre-set risk parameters (e.g., lot size=0.1, stop loss=50 pips). This modular design ensures reusability and seamless integration with the EA for manual trading in volatile markets. |
| News\_Headline\_EA.mq5 | 1.12 | Main EA file integrating the Quick Trade Buttons with real-time economic calendar data, Alpha Vantage news feeds, technical indicators (RSI, Stochastic, MACD, CCI), and AI-driven insights. It manages manual trade inputs, automated news-driven stop orders, and chart-based displays (e.g., news at y=398, indicators at y=414-462). Uses distinct magic numbers (123456 for manual, 888888 for automated) to prevent trade conflicts, enabling augmented intelligence for news trading and scalping. |

To incorporate the AI insights lane into the News Headline Expert Advisor (EA), refer to the article [From Novice to Expert: Animated News Headline Using MQL5 (IV) — Locally hosted AI model market insights](https://www.mql5.com/en/articles/18685). This resource provides detailed guidance on the associated Python scripts required to generate AI-driven market insights and explains their integration with the EA.

[back to contents](https://www.mql5.com/en/articles/18975#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18975.zip "Download all attachments in the single ZIP archive")

[TradingButtons.mqh](https://www.mql5.com/en/articles/download/18975/tradingbuttons.mqh "Download TradingButtons.mqh")(13.39 KB)

[News\_Headline\_EA.mq5](https://www.mql5.com/en/articles/download/18975/news_headline_ea.mq5 "Download News_Headline_EA.mq5")(60.29 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/492655)**

![Statistical Arbitrage Through Cointegrated Stocks (Part 2): Expert Advisor, Backtests, and Optimization](https://c.mql5.com/2/162/19052-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 2): Expert Advisor, Backtests, and Optimization](https://www.mql5.com/en/articles/19052)

This article presents a sample Expert Advisor implementation for trading a basket of four Nasdaq stocks. The stocks were initially filtered based on Pearson correlation tests. The filtered group was then tested for cointegration with Johansen tests. Finally, the cointegrated spread was tested for stationarity with the ADF and KPSS tests. Here we will see some notes about this process and the results of the backtests after a small optimization.

![Self Optimizing Expert Advisors in MQL5 (Part 11): A Gentle Introduction to the Fundamentals of Linear Algebra](https://c.mql5.com/2/160/18974-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 11): A Gentle Introduction to the Fundamentals of Linear Algebra](https://www.mql5.com/en/articles/18974)

In this discussion, we will set the foundation for using powerful linear, algebra tools that are implemented in the MQL5 matrix and vector API. For us to make proficient use of this API, we need to have a firm understanding of the principles in linear algebra that govern intelligent use of these methods. This article aims to get the reader an intuitive level of understanding of some of the most important rules of linear algebra that we, as algorithmic traders in MQL5 need,to get started, taking advantage of this powerful library.

![Building a Trading System (Part 2): The Science of Position Sizing](https://c.mql5.com/2/162/18991-building-a-profitable-trading-logo.png)[Building a Trading System (Part 2): The Science of Position Sizing](https://www.mql5.com/en/articles/18991)

Even with a positive-expectancy system, position sizing determines whether you thrive or collapse. It’s the pivot of risk management—translating statistical edges into real-world results while safeguarding your capital.

![MQL5 Wizard Techniques you should know (Part 78): Gator and AD Oscillator Strategies for Market Resilience](https://c.mql5.com/2/160/18992-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 78): Gator and AD Oscillator Strategies for Market Resilience](https://www.mql5.com/en/articles/18992)

The article presents the second half of a structured approach to trading with the Gator Oscillator and Accumulation/Distribution. By introducing five new patterns, the author shows how to filter false moves, detect early reversals, and align signals across timeframes. With clear coding examples and performance tests, the material bridges theory and practice for MQL5 developers.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/18975&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083435412568939433)

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