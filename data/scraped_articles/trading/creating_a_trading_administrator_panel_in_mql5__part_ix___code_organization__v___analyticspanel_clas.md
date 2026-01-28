---
title: Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (V): AnalyticsPanel Class
url: https://www.mql5.com/en/articles/17397
categories: Trading, Integration
relevance_score: 6
scraped_at: 2026-01-22T17:57:18.125060
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/17397&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049510956251327692)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/17397#para1)
- [Overview of AnalyticsPanel class](https://www.mql5.com/en/articles/17397#para2)
- [Implementation of MQL5](https://www.mql5.com/en/articles/17397#para3)
- [Testing](https://www.mql5.com/en/articles/17397#para4)
- [Conclusion](https://www.mql5.com/en/articles/17397#para5)

### Introduction

In [earlier](https://www.mql5.com/en/articles/16356) sections, I introduced the concept of an Analytics Panel, which—at that stage—consisted mainly of static interface elements, such as a Pie Chart displaying the Win/Loss ratio. However, these components lacked real-time data updates, limiting their usefulness in dynamic trading environments. In this discussion, we take a significant step forward by enhancing both the design and functionality of the interface, focusing on how to retrieve and display real-time trading data.

Our development approach leverages the [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary), particularly the classes found in the _\\Include\\Controls\_ directory. We will extend and customize existing classes such as CLabel, CEdit, and CDialog to create responsive, data-driven UI components. These form the backbone of our evolving Trading Administrator Panel, which is designed as a modular, multi-interface tool offering comprehensive account and strategy management as well as a communication interface via Telegram for remote control and real-time notifications.

While the core objective remains the development of this advanced panel, the techniques shared here—especially those involving real-time data acquisition and dynamic UI updates—can be applied to various MQL5 projects beyond administrative interfaces, including Expert Advisors, custom indicators, and interactive learning tools.

### Overview of AnalyticsPanel class

As part of a modular development approach suited for large-scale MQL5 programs—and to promote code reusability and maintainability—we are creating a dedicated AnalyticsPanel class header file. This class is designed to encapsulate both the visual layout of the analytics panel and the real-time retrieval and display of market data.

In addition to providing standard account metrics, the panel will display various technical indicator values that feed into a custom strategy I’ve termed the Confluence Strategy. This strategy is based on the principle of confluence, where signals from multiple indicators are compared to generate a unified trading signal. If no agreement is found among the indicators, the panel simply displays a "No Consensus" message, thereby avoiding false or weak signals.

The AnalyticsPanel class will include methods for initializing and refreshing the panel layout, updating label values in real time, and managing visual signal feedback based on the strategy's logic. Below, I’ve included a visual design layout of the panel, and in the following discussion, we will walk through the implementation details that brought it to life.

![AnalyticsPanelDesign](https://c.mql5.com/2/131/AnalyticsPanelDesign.png)

AnalyticsPanel Features

As part of our ongoing development journey, after we have developed our class we will integrate it into the main Expert Advisor program, the New\_Admin\_Panel. This integration reinforces the modular architecture of our system and highlights the advantages of building reusable and independent components that can seamlessly interact within a larger application framework. Below is an image highlighting the final performance of our product, where trading signals are generated only when all indicator values align to form a strong consensus for either a buy or sell decision

![AnalyticsPanel Live](https://c.mql5.com/2/131/terminal64_bjZJTUuKI8.gif)

AnalyticsPanel Live

By integrating the AnalyticsPanel into the New\_Admin\_Panel, we unlock several practical benefits

- Centralized Monitoring: Real-time analytics such as win/loss ratios, equity changes, and trade summaries can now be accessed from the primary interface without requiring separate tools.
- Enhanced User Experience: Combining analytics with core functionalities like strategy execution and Telegram messaging provides a more unified and intuitive workflow for the user.
- Code Scalability and Maintenance: Separating interface components into classes like AnalyticsPanel improves code readability, facilitates testing, and supports future upgrades with minimal disruptions.
- Cross-Project Reusability: The modular design enables the AnalyticsPanel (and other panels) to be reused or adapted in other Expert Advisors or trading utilities.
- Trading Signal: The version that we have here gives useful confluence signals.

### Implementation of MQL5

In designing our Analytics Panel, we follow object-oriented principles that separate the essential components into different access levels to improve clarity, security, and maintainability. The class is structured so that certain features are available to other parts of the program—these are the "public" elements, which serve as the interface for interacting with the panel. Other internal details, which deal with the inner workings and data management, are kept hidden from outside components; these are the "private" elements. This method of encapsulation allows us to present a clear and organized blueprint for our panel, making it easier for future developers to understand the role and purpose of each section.

Before defining the CAnalyticsPanel class, two critical header files are included:

Dialog.mqh – This header provides access to dialog-related classes, including CAppDialog, which serves as the base class for creating the custom panel.

Label.mqh – This includes the definition of the CLabel class, which is used to create and manage all the text labels displayed on the panel.

These inclusions are necessary to give the custom panel access to MQL5’s standard UI control structures. Without them, the panel's class would not be able to inherit from CAppDialog or create label controls.

We also need to define macros for spacing and layout (AP\_GAP, AP\_DEFAULT\_LABEL\_HEIGHT) to keep its visual structure both responsive and clean.

```
#include <Controls\Dialog.mqh>
#include <Controls\Label.mqh>

// Use unique macro names for layout in this class:
#define AP_DEFAULT_LABEL_HEIGHT 20
#define AP_GAP                10
```

The following outline breaks down the main components of our class into five key areas:

**1\. Class Structure and Purpose**

The CAnalyticsPanel class serves as a highly visual and informative dashboard designed for real-time updates on trading and market conditions directly within a MetaTrader 5 chart. Derived from the CAppDialog class in the MQL5 Standard Library, it leverages object-oriented programming to display structured data in neatly arranged sections. The panel includes labels for account details, open trades, market quotes, indicators, and a synthesized signal summary—allowing traders to monitor everything in one compact window.

```
//+------------------------------------------------------------------+
//| Analytics Panel Class                                            |
//+------------------------------------------------------------------+
class CAnalyticsPanel : public CAppDialog
{
protected:
   // Helper: Create a text label.
   // The label's text color is set as provided and its background is forced to black.
   bool CreateLabelEx(CLabel &label, int x, int y, int width, int height, string label_name, string text, color clr)
   {
      // Construct a unique control name by combining the dialog name and label_name.
      string unique_name = m_name + "_" + label_name;
      if(!label.Create(m_chart_id, unique_name, m_subwin, x, y, x+width, y+height))
      {
         Print("Failed to create label: ", unique_name, " Err=", GetLastError());
         return false;
      }
      if(!Add(label))
      {
         Print("Failed to add label: ", unique_name, " Err=", GetLastError());
         return false;
      }
      if(!label.Text(text))
      {
         Print("Failed to set text for label: ", unique_name);
         return false;
      }
      label.Color(clr);          // Set text color
      label.Color(clrBlack);   // Force background to black
      return true;
   }

   // Account and Trade Data Labels:
   CLabel m_lblBalance;
   CLabel m_lblEquity;
   CLabel m_lblMargin;
   CLabel m_lblOpenTrades;
   // PnL is split into two labels:
   CLabel m_lblPnLName;
   CLabel m_lblPnLValue;

   // Market Data Labels (split Bid/Ask):
   CLabel m_lblBidName;
   CLabel m_lblBidValue;
   CLabel m_lblAskName;
   CLabel m_lblAskValue;
   CLabel m_lblSpread;

   // Indicator Section Separator:
   CLabel m_lblIndicatorSeparator;
   // New header for indicator section:
   CLabel m_lblIndicatorHeader;

   // Indicator Labels (static name and dynamic value labels):
   CLabel m_lblRSIName;
   CLabel m_lblRSIValue;
   CLabel m_lblStochName;
   CLabel m_lblStochK;  // %K value
   CLabel m_lblStochD;  // %D value
   CLabel m_lblCCIName;
   CLabel m_lblCCIValue;
   CLabel m_lblWilliamsName;
   CLabel m_lblWilliamsValue;

   // New Summary Labels (for consensus across indicators):
   CLabel m_lblSignalHeader;    // Header label for summary column ("SIGNAL")
   CLabel m_lblExpectation;     // Text summary (e.g., "Buy" or "Sell")
   CLabel m_lblConfirmation;    // Confirmation symbol (e.g., up or down arrow)

   // Previous values for dynamic color changes (market data):
   double m_prevBid;
   double m_prevAsk;

public:
   CAnalyticsPanel();
   virtual ~CAnalyticsPanel();

   // Create the panel and initialize UI controls.
   virtual bool CreatePanel(const long chart, const string name, const int subwin,
                            int x1, int y1, int x2, int y2);

   // Update the panel with the latest account, trade, market and indicator data.
   void UpdatePanel();
   void Toggle();

private:
   void CreateControls();
   void UpdateAccountInfo();
   void UpdateTradeStats();
   void UpdateMarketData();
   void UpdateIndicators();
};
```

I have provided a detailed outline and explanation for the variable below.

**Member Variables**

UI Labels for Account and Trade Data:

Account Labels:

- m\_lblBalance, m\_lblEquity, and m\_lblMargin display the account balance, equity, and margin, respectively.

Trade Statistics:

- m\_lblOpenTrades shows the number of currently open trades.

The PnL (Profit and Loss) information is split into two labels:

1. m\_lblPnLName: A static label that simply displays the text "PnL:" (this remains black).
2. m\_lblPnLValue: A dynamic label that displays the numeric PnL value and changes color (green for profit, red for loss).

Market Data Labels:

These labels show bid/ask values and spread information:

- m\_lblBidName and m\_lblBidValue for the bid price.
- m\_lblAskName and m\_lblAskValue for the ask price.
- m\_lblSpread displays the current spread in pips.
- The colors of the bid and ask values are updated dynamically to reflect market changes (for instance, blue when the value increases and red when it decreases).

Indicator Section:

- A separator (m\_lblIndicatorSeparator) visually distinguishes the indicator section from the rest of the panel.
- A header label (m\_lblIndicatorHeader) displays "INDICATORS", clearly identifying the section.

The section includes several indicators:

- RSI: Two labels, m\_lblRSIName and m\_lblRSIValue, show the name ("RSI:") and its current value.
- Stochastic Oscillator: This indicator is split into two parts with:

1. m\_lblStochName: The static part ("Stoch:").
2. m\_lblStochK and m\_lblStochD: Display the dynamic values for %K and %D.

- CCI: m\_lblCCIName and m\_lblCCIValue show the Commodity Channel Index.
- Williams %R: m\_lblWilliamsName and m\_lblWilliamsValue display the Williams %R values.

Summary (Consensus) Section:

- A header label (m\_lblSignalHeader) displays the word "SIGNAL" above the summary column.
- Two labels, m\_lblExpectation and m\_lblConfirmation, are used to show the consensus signal from all the indicators:

1. m\_lblExpectation provides a text summary such as "Buy", "Sell", or "No Consensus".
2. m\_lblConfirmation shows a symbolic confirmation (an up arrow "↑" for buy, a down arrow "↓" for SELL, or a dash "-" if no consensus is reached).

Internal State for Market Data:

Two variables, m\_prevBid and m\_prevAsk, store the previous bid and ask values to help determine whether these values are increasing or decreasing—thus informing the color-coding used for these labels.

**2\. Helper Method for Label Creation**

A core utility function within the class is CreateLabelEx, a reusable method for creating and customizing text labels on the panel. This method encapsulates all steps required to instantiate a label, assign it a unique name, configure its size and location, apply the desired text and colors, and add it to the panel’s control list.  If any step in this process fails, appropriate error messages are logged for debugging.

```
bool CAnalyticsPanel::CreateLabelEx(CLabel &label, int x, int y, int width, int height,
                                    string label_name, string text, color clr)
{
   string unique_name = m_name + "_" + label_name;
   if(!label.Create(m_chart_id, unique_name, m_subwin, x, y, x+width, y+height))
   {
      Print("Failed to create label: ", unique_name, " Err=", GetLastError());
      return false;
   }
   if(!Add(label))
   {
      Print("Failed to add label: ", unique_name, " Err=", GetLastError());
      return false;
   }
   if(!label.Text(text))
   {
      Print("Failed to set text for label: ", unique_name);
      return false;
   }
   label.Color(clr);          // Set text color
   label.BackColor(clrBlack); // Force background to black
   return true;
}

```

**3\. Organizing the User Interface Layout**

The CreateControls method organizes all user interface elements on the panel. Here we arrange different sections: account information (balance, equity, margin), trade statistics (open trades and profit and loss), market data (bid, ask, and spread), and the indicator section. In the indicator section, we create headers like "INDICATORS" and "SIGNAL" to clearly label the groups of information. The code uses predefined constants for label height and gaps to ensure all elements are evenly spaced and visually organized.

```
//+------------------------------------------------------------------+
//| CreateControls: Instantiate and add UI controls                  |
//+------------------------------------------------------------------+
void CAnalyticsPanel::CreateControls()
{
   int curX = AP_GAP;
   int curY = AP_GAP;
   int labelWidth = 150;

   // Account Information Labels:
   CreateLabelEx(m_lblBalance, curX, curY, labelWidth, AP_DEFAULT_LABEL_HEIGHT, "Balance", "Balance: $0.00", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;
   CreateLabelEx(m_lblEquity, curX, curY, labelWidth, AP_DEFAULT_LABEL_HEIGHT, "Equity", "Equity: $0.00", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;
   CreateLabelEx(m_lblMargin, curX, curY, labelWidth, AP_DEFAULT_LABEL_HEIGHT, "Margin", "Margin: 0", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;

   // Trade Statistics Labels:
   CreateLabelEx(m_lblOpenTrades, curX, curY, labelWidth, AP_DEFAULT_LABEL_HEIGHT, "OpenTrades", "Open Trades: 0", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;
   // PnL labels:
   CreateLabelEx(m_lblPnLName, curX, curY, 50, AP_DEFAULT_LABEL_HEIGHT, "PnLName", "PnL:", clrBlack);
   CreateLabelEx(m_lblPnLValue, curX+50, curY, labelWidth-50, AP_DEFAULT_LABEL_HEIGHT, "PnLValue", "$0.00", clrLime);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;

   // Market Data Labels:
   CreateLabelEx(m_lblBidName, curX, curY, 40, AP_DEFAULT_LABEL_HEIGHT, "BidName", "Bid:", clrDodgerBlue);
   CreateLabelEx(m_lblBidValue, curX+40, curY, 100, AP_DEFAULT_LABEL_HEIGHT, "BidValue", "0.00000", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;
   CreateLabelEx(m_lblAskName, curX, curY, 40, AP_DEFAULT_LABEL_HEIGHT, "AskName", "Ask:", clrDodgerBlue);
   CreateLabelEx(m_lblAskValue, curX+40, curY, 100, AP_DEFAULT_LABEL_HEIGHT, "AskValue", "0.00000", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;
   CreateLabelEx(m_lblSpread, curX, curY, labelWidth, AP_DEFAULT_LABEL_HEIGHT, "Spread", "Spread: 0.0 pips", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;

   // Indicator Section Separator:
   CreateLabelEx(m_lblIndicatorSeparator, curX, curY, labelWidth, AP_DEFAULT_LABEL_HEIGHT, "Separator", "------------------------------------------", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;

   // Indicator Section Header:
   CreateLabelEx(m_lblIndicatorHeader, curX, curY, labelWidth, AP_DEFAULT_LABEL_HEIGHT, "IndicatorHeader", "INDICATORS", clrDodgerBlue);
   // Summary Column Header for Signal:
   CreateLabelEx(m_lblSignalHeader, curX+250, curY , 100, AP_DEFAULT_LABEL_HEIGHT, "SignalHeader", "SIGNAL", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;

   // Indicator Labels:
   // RSI:
   CreateLabelEx(m_lblRSIName, curX, curY, 50, AP_DEFAULT_LABEL_HEIGHT, "RSIName", "RSI:", clrDodgerBlue);
   CreateLabelEx(m_lblRSIValue, curX+50, curY, 90, AP_DEFAULT_LABEL_HEIGHT, "RSIValue", "N/A", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;
   // Stochastic:
   CreateLabelEx(m_lblStochName, curX, curY, 60, AP_DEFAULT_LABEL_HEIGHT, "StochName", "Stoch:", clrDodgerBlue);
   CreateLabelEx(m_lblStochK, curX+60, curY, 70, AP_DEFAULT_LABEL_HEIGHT, "StochK", "K: N/A", clrDodgerBlue);
   CreateLabelEx(m_lblStochD, curX+150, curY, 70, AP_DEFAULT_LABEL_HEIGHT, "StochD", "D: N/A", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;
   // CCI:
   CreateLabelEx(m_lblCCIName, curX, curY, 50, AP_DEFAULT_LABEL_HEIGHT, "CCIName", "CCI:", clrDodgerBlue);
   CreateLabelEx(m_lblCCIValue, curX+50, curY, 90, AP_DEFAULT_LABEL_HEIGHT, "CCIValue", "N/A", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;
   // Williams %R:
   CreateLabelEx(m_lblWilliamsName, curX, curY, 70, AP_DEFAULT_LABEL_HEIGHT, "WilliamsName", "Williams:", clrDodgerBlue);
   CreateLabelEx(m_lblWilliamsValue, curX+70, curY, 70, AP_DEFAULT_LABEL_HEIGHT, "WilliamsValue", "N/A", clrDodgerBlue);
   curY += AP_DEFAULT_LABEL_HEIGHT + AP_GAP;

   // Summary Column: Expectation text and Confirmation symbol.
   CreateLabelEx(m_lblExpectation, curX+250, curY - (AP_DEFAULT_LABEL_HEIGHT+AP_GAP)*4, 100, AP_DEFAULT_LABEL_HEIGHT, "Expectation", "Expect: N/A", clrDodgerBlue);
   CreateLabelEx(m_lblConfirmation, curX+300, curY - (AP_DEFAULT_LABEL_HEIGHT+AP_GAP)*4, 50, AP_DEFAULT_LABEL_HEIGHT, "Confirmation", "N/A", clrDodgerBlue);

   ChartRedraw(m_chart_id);
}
```

**4\. Real-Time Data Updates for Account and Market Information**

Using MQL5’s built-in functions, the panel retrieves account information—such as balance, equity, and raw margin value—as well as market data like bid and ask prices and the spread. The UpdateAccountInfo and UpdateTradeStats methods directly query the account with functions such as AccountInfoDouble(). Meanwhile, UpdateMarketData gets real-time tick data using SymbolInfoTick(), compares current values with previous ones, and updates color coding for bid and ask labels based on changes. This enables traders to see the latest account and market status as they occur.

```
//+------------------------------------------------------------------+
//| UpdateAccountInfo: Refresh account-related data                  |
//+------------------------------------------------------------------+
void CAnalyticsPanel::UpdateAccountInfo()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double margin  = AccountInfoDouble(ACCOUNT_MARGIN);

   m_lblBalance.Text("Balance: $" + DoubleToString(balance, 2));
   m_lblEquity.Text("Equity: $" + DoubleToString(equity, 2));
   // Directly display the raw margin value.
   m_lblMargin.Text("Margin: " + DoubleToString(margin, 2));
}

//+------------------------------------------------------------------+
//| UpdateTradeStats: Refresh trade statistics                       |
//+------------------------------------------------------------------+
void CAnalyticsPanel::UpdateTradeStats()
{
   int total = PositionsTotal();
   double pnl = 0;
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket != 0)
         pnl += PositionGetDouble(POSITION_PROFIT);
   }
   // Update PnL labels:
   m_lblPnLName.Text("PnL:");  // static remains black
   m_lblPnLValue.Text("$" + DoubleToString(pnl, 2));
   if(pnl < 0)
      m_lblPnLValue.Color(clrRed);
   else
      m_lblPnLValue.Color(clrLime);

   m_lblOpenTrades.Text("Open Trades: " + IntegerToString(total));
}

//+------------------------------------------------------------------+
//| UpdateMarketData: Refresh market data display                    |
//+------------------------------------------------------------------+
void CAnalyticsPanel::UpdateMarketData()
{
   MqlTick last_tick;
   if(SymbolInfoTick(Symbol(), last_tick))
   {
      // Update Bid value and color based on change:
      color bidColor = clrBlue;
      if(m_prevBid != 0)
         bidColor = (last_tick.bid >= m_prevBid) ? clrBlue : clrRed;
      m_lblBidValue.Color(bidColor);
      m_lblBidValue.Text(DoubleToString(last_tick.bid, 5));

      // Update Ask value and color based on change:
      color askColor = clrBlue;
      if(m_prevAsk != 0)
         askColor = (last_tick.ask >= m_prevAsk) ? clrBlue : clrRed;
      m_lblAskValue.Color(askColor);
      m_lblAskValue.Text(DoubleToString(last_tick.ask, 5));

      m_prevBid = last_tick.bid;
      m_prevAsk = last_tick.ask;

      // Update Spread:
      double spread_pips = (last_tick.ask - last_tick.bid) * 1e4;
      m_lblSpread.Text("Spread: " + DoubleToString(spread_pips, 1) + " pips");
   }
}
```

**5\. Calculating Technical Indicators and Generating a Consensus Signal**

The UpdateIndicators method retrieves values for several technical indicators—RSI, Stochastic, CCI, and Williams %R—by calling indicator functions and reading their most recent data. Each indicator’s value is compared against predefined thresholds to determine whether it suggests a buy or sell condition. For instance, an RSI below 30 is interpreted as a buying signal, while above 70 suggests selling. The panel then consolidates these individual signals into an overall consensus; if all indicators agree, a clear "Buy" or "Sell" signal appears in the summary section along with an arrow symbol. This provides traders with a quick visual cue about market conditions.

```
//+------------------------------------------------------------------+
//| UpdateIndicators: Calculate and display various indicators       |
//+------------------------------------------------------------------+
void CAnalyticsPanel::UpdateIndicators()
{
   // Local suggestion variables (1 = Buy, -1 = Sell, 0 = Neutral)
   int suggestionRSI = 0, suggestionStoch = 0, suggestionCCI = 0, suggestionWilliams = 0;

   // --- RSI ---
   int handleRSI = iRSI(Symbol(), PERIOD_CURRENT, 14, PRICE_CLOSE);
   double rsi[1];
   color rsiValueColor = clrGreen; // default
   if(CopyBuffer(handleRSI, 0, 0, 1, rsi) > 0)
   {
      if(rsi[0] < 30)
      {
         rsiValueColor = clrBlue; // buy condition
         suggestionRSI = 1;
      }
      else if(rsi[0] > 70)
      {
         rsiValueColor = clrRed;  // sell condition
         suggestionRSI = -1;
      }
      m_lblRSIValue.Color(rsiValueColor);
      m_lblRSIValue.Text(DoubleToString(rsi[0], 2));
   }
   IndicatorRelease(handleRSI);

   // --- Stochastic ---
   int handleStoch = iStochastic(Symbol(), PERIOD_CURRENT, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   double stochK[1], stochD[1];
   if(CopyBuffer(handleStoch, 0, 0, 1, stochK) > 0 && CopyBuffer(handleStoch, 1, 0, 1, stochD) > 0)
   {
      // Initialize color variables.
      color colorK = clrGreen;
      color colorD = clrGreen;
      if(stochK[0] > stochD[0])
      {
         colorK = clrBlue;
         colorD = clrRed;
         suggestionStoch = 1;
      }
      else if(stochK[0] < stochD[0])
      {
         colorK = clrRed;
         colorD = clrBlue;
         suggestionStoch = -1;
      }
      else
         suggestionStoch = 0;

      m_lblStochK.Color(colorK);
      m_lblStochD.Color(colorD);
      m_lblStochK.Text("K: " + DoubleToString(stochK[0], 4));
      m_lblStochD.Text("D: " + DoubleToString(stochD[0], 4));
   }
   IndicatorRelease(handleStoch);

   // --- CCI ---
   int handleCCI = iCCI(Symbol(), PERIOD_CURRENT, 14, PRICE_TYPICAL);
   double cci[1];
   color cciValueColor = clrGreen;
   if(CopyBuffer(handleCCI, 0, 0, 1, cci) > 0)
   {
      if(cci[0] < -100)
      {
         cciValueColor = clrBlue; // buy condition
         suggestionCCI = 1;
      }
      else if(cci[0] > 100)
      {
         cciValueColor = clrRed;  // sell condition
         suggestionCCI = -1;
      }
      m_lblCCIValue.Color(cciValueColor);
      m_lblCCIValue.Text(DoubleToString(cci[0], 2));
   }
   IndicatorRelease(handleCCI);

   // --- Williams %R ---
   int handleWPR = iWPR(Symbol(), PERIOD_CURRENT, 14);
   double williams[1];
   color williamsValueColor = clrGreen;
   if(CopyBuffer(handleWPR, 0, 0, 1, williams) > 0)
   {
      if(williams[0] > -80)
      {
         williamsValueColor = clrBlue; // buy condition
         suggestionWilliams = 1;
      }
      else if(williams[0] < -20)
      {
         williamsValueColor = clrRed;  // sell condition
         suggestionWilliams = -1;
      }
      m_lblWilliamsValue.Color(williamsValueColor);
      m_lblWilliamsValue.Text(DoubleToString(williams[0], 2));
   }
   IndicatorRelease(handleWPR);

   // --- Consensus Summary ---
   int consensus = 0;
   if(suggestionRSI != 0 && suggestionRSI == suggestionStoch &&
      suggestionRSI == suggestionCCI && suggestionRSI == suggestionWilliams)
      consensus = suggestionRSI;

   if(consensus == 1)
   {
      m_lblExpectation.Text("Buy");
      m_lblExpectation.Color(clrBlue);
      m_lblConfirmation.Text("↑"); // Up arrow for Buy
      m_lblConfirmation.Color(clrBlue);
   }
   else if(consensus == -1)
   {
      m_lblExpectation.Text("Sell");
      m_lblExpectation.Color(clrRed);
      m_lblConfirmation.Text("↓"); // Down arrow for Sell
      m_lblConfirmation.Color(clrRed);
   }
   else
   {
      m_lblExpectation.Text("No Consensus");
      m_lblExpectation.Color(clrOrangeRed);
      m_lblConfirmation.Text("-");
      m_lblConfirmation.Color(clrOrangeRed);
   }
}
```

**Integrating the CAnalyticsPanel into the Main Admin Panel**

1\. At the top of the EA file, the AnalyticsPanel.mqh header file is included to PROVIDE access to the CAnalyticsPanel class:

```
#include <AnalyticsPanel.mqh>
```

This ensures the compiler recognizes the class and its members during compilation.

2\. Declare a Global Pointer

A global pointer of type CAnalyticsPanel is declared, which will be used to manage the instance of the panel dynamically:

```
CAnalyticsPanel *g_analyticsPanel = NULL;
```

This pointer helps track whether the panel exists and enables its dynamic creation, destruction, and visibility toggling.

3\. Create a Button to Launch the Panel

In the CreateAdminPanel() function, a button labeled “Analytics Panel” is created and added to the main admin panel. This button serves as the user’s trigger for launching the panel.

```
bool CreateAdminPanel()
{

   // Analytics Panel Button
   if(!CreateButton(g_analyticsButton, ANALYTICS_BTN_NAME, INDENT_LEFT, y, INDENT_LEFT+btnWidth, y+BUTTON_HEIGHT, "Analytics Panel"))
      return false;

return true;

}
```

4\. Handle Button Click Events

The HandleAnalytics() function is responsible for managing the panel’s lifecycle. It checks if the panel has already been created. If not, it instantiates the CAnalyticsPanel, calls its CreatePanel() method with positioning coordinates, and toggles its visibility. Otherwise, it simply toggles the panel on or off.

```
//------------------------------------------------------------------
// Handle Analytics Panel button click
void HandleAnalytics()
{
   if(g_analyticsPanel == NULL)
   {
      g_analyticsPanel = new CAnalyticsPanel();
      // Coordinates for Analytics panel (adjust as needed)
      if(!g_analyticsPanel.CreatePanel(g_chart_id, "AnalyticsPanel", g_subwin, 900, 20, 900+500, 20+460))
      {
         delete g_analyticsPanel;
         g_analyticsPanel = NULL;
         Print("Failed to create Analytics Panel");
         return;
      }
   }
   g_analyticsPanel.Toggle();
   ChartRedraw();
}
```

5\. Event Forwarding

In the AdminPanelOnEvent() function, events such as clicks or user interactions are passed to the g\_analyticsPanel if it is visible. This ensures that the panel can independently respond to user input without conflicting with the main panel or other sub-panels.

```
if(g_analyticsPanel != NULL && g_analyticsPanel.IsVisible())
      return g_analyticsPanel.OnEvent(id, lparam, dparam, sparam);
```

6\. Updating the Panel Dynamically

Inside the OnTick() function, if the analytics panel exists, its UpdatePanel() method is called regularly. This allows the panel to dynamically update its displayed analytics or UI contents during the EA's execution.

```
void OnTick()
  {

    if(g_analyticsPanel != NULL)
    {
      g_analyticsPanel.UpdatePanel();

    }

  }
```

7\. Cleaning Up

In the OnDeinit() function, the panel is properly destroyed and memory is freed by calling Destroy() and deleting the pointer. This is crucial to avoid memory leaks when the EA is removed or recompiled.

### Testing

I had an incredible testing experience. It felt almost live, watching the values update on the panel with every tick. I managed to take a trade using the confluence strategy, but unfortunately, by the time I started recording, the signal had already shifted to 'No Consensus'. Still, I'm thrilled to share the results — see the image below!

![Testing the Anlytics Panel and the trading New_Admin_Panel](https://c.mql5.com/2/131/terminal64_ZJE2U7Ytxq.gif)

Demo Testing the working of Analytics Panel

In the image above, the value of Open Trades was updated with every position closure. We began with 21 positions and concluded with 18.

![AnalyticsPanel](https://c.mql5.com/2/131/terminal64_l3iwAWkb6Y.gif)

All Panels are accessible from the home Panel

### Conclusion

We successfully expanded our program by integrating the Analytics Panel. This addition demonstrates that many more sub-panels can be incorporated using the same modular approach applied to the other panels. This discussion marks the conclusion of Part (IX) of the series. However, this does not mean the work is complete—there are still several areas for refinement. That said, the core concept is now well established and clearly presented.

Both beginners and experienced developers can benefit from this series at different levels. There are plenty of insightful notes to take away. These ideas lay a strong foundation for even more advanced developments in the future.

In the results presented, I had drawn channel lines on the chart, and coincidentally, a buy signal was generated by the Analytics Panel. This signal resulted from the confluence of all listed indicators, aligning perfectly with the support zone of the channel. I confidently took the trade based on this confluence strategy. The test was conducted on a demo account—as always, I strongly recommend all traders thoroughly test on a demo account before risking real money.

The strength of this setup is that traders can instantly gain access to crucial market and account information via the Analytic panel, while also benefiting from the analysis power of the confluence-based signal. One area for improvement is the lack of alerts for the strategy; adding notifications would make the system even more user-friendly and ensure no opportunities are missed.

Both the header file and the main program are attached below. Be sure to share your views and thoughts in the comment section. Until the next publication—happy developing and successful trading!

| File Name | Specification |
| --- | --- |
| [AnalyticsPanel.mqh](https://www.mql5.com/en/articles/download/17397/analyticspanel.mqh) | A header file that defines the structure and behavior of the Analytics Panel, which aggregates multiple indicators to generate confluence-based trading signals. |
| [New\_Admin\_Panel.mqh](https://www.mql5.com/en/articles/download/17397/new_admin_panel.mq5) | The main MQL5 EA program that initializes and manages the entire admin panel interface, including navigation, authentication, and integration of various subpanels like Communications, Analytics, and Trade Management. |

[Back to contents](https://www.mql5.com/en/articles/17397#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17397.zip "Download all attachments in the single ZIP archive")

[AnalyticsPanel.mqh](https://www.mql5.com/en/articles/download/17397/analyticspanel.mqh "Download AnalyticsPanel.mqh")(32.77 KB)

[New\_Admin\_Panel.mq5](https://www.mql5.com/en/articles/download/17397/new_admin_panel.mq5 "Download New_Admin_Panel.mq5")(19.61 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/484625)**
(1)


![amrhamed83](https://c.mql5.com/avatar/avatar_na2.png)

**[amrhamed83](https://www.mql5.com/en/users/amrhamed83)**
\|
16 Apr 2025 at 11:17

can you please post all files together in the attachement?


![Neural Networks in Trading: Hierarchical Feature Learning for Point Clouds](https://c.mql5.com/2/92/Neural_Networks_in_Trading_Hierarchical_Learning_of_Point_Cloud_Features___LOGO.png)[Neural Networks in Trading: Hierarchical Feature Learning for Point Clouds](https://www.mql5.com/en/articles/15789)

We continue to study algorithms for extracting features from a point cloud. In this article, we will get acquainted with the mechanisms for increasing the efficiency of the PointNet method.

![Introduction to MQL5 (Part 15): A Beginner's Guide to Building Custom Indicators (IV)](https://c.mql5.com/2/133/Introduction_to_MQL5_Part_15___LOGO.png)[Introduction to MQL5 (Part 15): A Beginner's Guide to Building Custom Indicators (IV)](https://www.mql5.com/en/articles/17689)

In this article, you'll learn how to build a price action indicator in MQL5, focusing on key points like low (L), high (H), higher low (HL), higher high (HH), lower low (LL), and lower high (LH) for analyzing trends. You'll also explore how to identify the premium and discount zones, mark the 50% retracement level, and use the risk-reward ratio to calculate profit targets. The article also covers determining entry points, stop loss (SL), and take profit (TP) levels based on the trend structure.

![Automating Trading Strategies in MQL5 (Part 14): Trade Layering Strategy with MACD-RSI Statistical Methods](https://c.mql5.com/2/133/Automating_Trading_Strategies_in_MQL5_Part_14__LOGO.png)[Automating Trading Strategies in MQL5 (Part 14): Trade Layering Strategy with MACD-RSI Statistical Methods](https://www.mql5.com/en/articles/17741)

In this article, we introduce a trade layering strategy that combines MACD and RSI indicators with statistical methods to automate dynamic trading in MQL5. We explore the architecture of this cascading approach, detail its implementation through key code segments, and guide readers on backtesting to optimize performance. Finally, we conclude by highlighting the strategy’s potential and setting the stage for further enhancements in automated trading.

![Developing a Trading System Based on the Order Book (Part I): Indicator](https://c.mql5.com/2/92/Desenvolvendo_um_Trading_System_com_base_no_Livro_de_Ofertas_Parte_I.png)[Developing a Trading System Based on the Order Book (Part I): Indicator](https://www.mql5.com/en/articles/15748)

Depth of Market is undoubtedly a very important element for executing fast trades, especially in High Frequency Trading (HFT) algorithms. In this series of articles, we will look at this type of trading events that can be obtained through a broker on many tradable symbols. We will start with an indicator, where you can customize the color palette, position and size of the histogram displayed directly on the chart. We will also look at how to generate BookEvent events to test the indicator under certain conditions. Other possible topics for future articles include how to store price distribution data and how to use it in a strategy tester.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/17397&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049510956251327692)

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