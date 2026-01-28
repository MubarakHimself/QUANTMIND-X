---
title: Developing a Trading System Based on the Order Book (Part I): Indicator
url: https://www.mql5.com/en/articles/15748
categories: Trading Systems
relevance_score: 9
scraped_at: 2026-01-22T17:34:59.635797
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/15748&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049231353880356754)

MetaTrader 5 / Examples


### Introduction

Let's recap what Depth of Market is. It is a series of pending limit orders. These orders represent the trading intentions of market participants and often do not result in an actual transaction. This is because traders have the ability to cancel their previously placed orders for various reasons. These may include changes in market conditions and the resulting loss of interest in executing the order at the previously specified price and quantity.

The value returned by the function SymbolInfoInteger(\_Symbol, SYMBOL\_TICKS\_BOOKDEPTH) corresponds precisely to the depth of the order book and represents half of the array that will be populated with price levels to be analyzed. Half of this array is allocated for the number of limit sell orders, while the other half is for limit buy orders that have been placed. According to the documentation, for assets that do not have an order queue, the value of this property is zero. An example of this can be seen in the figure below, which shows the order book with the depth of 10, showing all available price levels.

![Example: Order Book with depth 10](https://c.mql5.com/2/116/Depth-10__1.png)

It should be noted that depth can be obtained from the symbol and not necessarily from the market depth. Using the _SymbolInfoInteger_ function is sufficient to retrieve the property value, without resorting to the _OnBookEvent_ handler or related functions such as _MarketBookAdd_. Of course, we could arrive at the same result by counting the number of elements in the _MqlBookInfo_ array that the _OnBookEvent_ handler populates, as we will explore in more detail later.

You might be wondering why we should use this indicator instead of simply relying on MetaTrader 5's standard order book. Here are some key reasons:

- Optimized chart space utilization, allowing customization of histogram size and its position on the screen.

- Cleaner presentation of order book events, enhancing clarity.
- Usability in the strategy tester, with a future implementation of a disk-based storage mechanism for BookEvent events, considering that native testing is currently not supported.


### Generating a Custom Symbol

This process will enable us to test the indicator even when the market is closed or when the _broker_ does not transmit events for the given symbol. In such cases, there will be no live order queue, nor will these events be cached on the local computer. At this stage, we will not be working with past events from a real symbol but will instead focus on generating simulated BookEvent data for fictitious assets. This is necessary because creating such an asset and simulating events is essential for working with the CustomBookAdd function. This function is specifically designed for custom symbols.

Below is the CloneSymbolTicksAndRates script, which will generate the custom symbol. It has been adapted from the [documentation](https://www.mql5.com/en/docs/customsymbols/customsymbolcreate) to suit our needs and begins by defining some constants and including the standard DateTime.mqh library for working with dates. Note that the name of the custom symbol will be derived from the real symbol's nomenclature, which is passed to the script via the Symbol() function. Therefore, this script must be run on the real asset to be cloned. Although it is also possible to clone custom symbols, doing so does not seem particularly useful.

```
#define   CUSTOM_SYMBOL_NAME     Symbol()+".C"
#define   CUSTOM_SYMBOL_PATH     "Forex"
#define   CUSTOM_SYMBOL_ORIGIN   Symbol()

#define   DATATICKS_TO_COPY      UINT_MAX
#define   DAYS_TO_COPY           5
#include <Tools\DateTime.mqh>
```

The following fragment, inserted into the OnStart() function of the same script, creates the "timemaster" date object. It is used to calculate the time period in which ticks and bars will be collected for cloning. According to the DAYS\_TO\_COPY constant we defined, the Bars function will copy the last five days of the source symbol. This same initial time of the range is then converted to milliseconds and used by the CopyTicks function, thus completing the "cloning" of the symbol.

```
   CDateTime timemaster;
   datetime now = TimeTradeServer();
   timemaster.Date(now);
   timemaster.DayDec(DAYS_TO_COPY);
   long DaysAgoMsc = 1000 * timemaster.DateTime();
   int bars_origin = Bars(CUSTOM_SYMBOL_ORIGIN, PERIOD_M1, timemaster.DateTime(), now);
   int create = CreateCustomSymbol(CUSTOM_SYMBOL_NAME, CUSTOM_SYMBOL_PATH, CUSTOM_SYMBOL_ORIGIN);
   if(create != 0 && create != 5304)
      return;
   MqlTick array[] = {};
   MqlRates rates[] = {};
   int attempts = 0;
   while(attempts < 3)
     {
      int received = CopyTicks(CUSTOM_SYMBOL_ORIGIN, array, COPY_TICKS_ALL, DaysAgoMsc, DATATICKS_TO_COPY);
      if(received != -1)
        {
         if(GetLastError() == 0)
            break;
        }
      attempts++;
      Sleep(1000);
     }
```

Once the process is complete, the new symbol should appear in the market watch list with the name <AtivodeOrigem>.C. At this point, we need to open a new chart with this synthetic symbol and proceed to the next step.

If another synthetic symbol already exists, it can be reused, making it unnecessary to create a new one as explained in this section. In the end, we simply need to open a new chart with this custom symbol and run two other MQL5 applications that we will develop here: the indicator and the event generator script. We will provide all the details in the following sections.

### BookEvent-Type Event Generator Script for Testing

Having a custom symbol alone does not compensate for the absence of an online order book tick sequence when performing a backtest on the indicator that relies on order book events. Therefore, we need to generate simulated data. For this purpose, the following script has been developed.

```
//+------------------------------------------------------------------+
//|                                            GenerateBookEvent.mq5 |
//|                                               Daniel Santos      |
//+------------------------------------------------------------------+
#property copyright "Daniel Santos"
#property version   "1.00"
#define SYNTH_SYMBOL_MARKET_DEPTH      32
#define SYNTH_SYMBOL_BOOK_ITERATIONS   20
#include <Random.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double BidValue, tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
MqlBookInfo    books[];
int marketDepth = SYNTH_SYMBOL_MARKET_DEPTH;
CRandom rdn;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   if(!SymbolInfoInteger(_Symbol, SYMBOL_CUSTOM)) // if the symbol exists
     {
      Print("Custom symbol ", _Symbol, " does not exist");
      return;
     }
   else
      BookGenerationLoop();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void BookGenerationLoop()
  {
   MqlRates BarRates_D1[];
   CopyRates(_Symbol, PERIOD_D1, 0, 1, BarRates_D1);
   if(ArraySize(BarRates_D1) == 0)
      return;
   BidValue = BarRates_D1[0].close;
   ArrayResize(books, 2 * marketDepth);
   for(int j = 0; j < SYNTH_SYMBOL_BOOK_ITERATIONS; j++)
     {
      for(int i = 0, j = 0; i < marketDepth; i++)
        {
         books[i].type = BOOK_TYPE_SELL;
         books[i].price = BidValue + ((marketDepth - i) * tickSize);
         books[i].volume_real = rdn.RandomInteger(10, 500);
         books[i].volume_real = round((books[i].volume_real + books[j].volume_real) / 2);
         books[i].volume = (int)books[i].volume_real;
         //----
         books[marketDepth + i].type = BOOK_TYPE_BUY;
         books[marketDepth + i].price = BidValue - (i * tickSize);
         books[marketDepth + i].volume_real = rdn.RandomInteger(10, 500);
         books[marketDepth + i].volume_real = round((books[marketDepth + i].volume_real
                                              + books[marketDepth + j].volume_real) / 2);
         books[marketDepth + i].volume = (int)books[marketDepth + i].volume_real;
         if(j != i)
            j++;
        }
      CustomBookAdd(_Symbol, books);
      Sleep(rdn.RandomInteger(400, 1000));
     }
  }
//+------------------------------------------------------------------+
```

Instead of the standard MathRand() function, we used an [alternative implementation for generating 32-bit random numbers](https://www.mql5.com/en/code/25843). This choice was made for several reasons, including the ease of generating integer values within a specified range - an advantage we leveraged in this script by using the RandomInteger(min, max) function.

For the order book depth, we selected a relatively large value of 32, meaning that each iteration will generate 64 price levels. If needed, this value can be adjusted to a smaller one.

The algorithm first checks whether the symbol is a custom one. If it is, it proceeds to generate each element of the order book and repeats this process in another loop based on the specified number of iterations. In this implementation, 20 iterations are performed with randomly chosen pauses between 400 milliseconds and 1000 milliseconds (equivalent to 1 second). This dynamic approach makes the visualization of ticks more realistic and visually appealing.

Prices are vertically anchored to the last closing price of the daily timeframe, as indicated by the source symbol. Above this reference point, there are 32 levels of sell orders, while below it, there are 32 levels of buy orders. According to the indicator's standard color scheme, histogram bars corresponding to sell orders have a reddish hue, while buy orders are represented in light blue.

The price difference between consecutive levels is determined based on the tick size of the symbol, which is obtained through the SYMBOL\_TRADE\_TICK\_SIZE property.

### Indicator for Displaying Market Depth Changes

### Library Source Code

The indicator was developed using object-oriented programming. The BookEventHistogram class was created to manage the order book histogram, handling its creation, updates, and the removal of bars when the class object is destroyed.

Below are the variable and function declarations for the BookEventHistogram class:

```
class BookEventHistogram
  {
protected:
   color                histogramColors[]; //Extreme / Mid-high / Mid-low
   int                  bookSize;
   int                  currElements;
   int                  elementMaxPixelsWidth;
   bool                 showMessages;
   ENUM_ALIGN_MODE      corner;
   string               bookEventElementPrefix;
public:
   MqlBookInfo          lastBook[];
   datetime             lastDate;
   void                 SetAlignLeft(void);
   void                 SetCustomHistogramColors(color &colors[]);
   void                 SetBookSize(int value) {bookSize = value;}
   void                 SetElementMaxPixelsWidth(int m);
   int                  GetBookSize(void) {return bookSize;}
   void                 DrawBookElements(MqlBookInfo& book[], datetime now);
   void                 CleanBookElements(void);
   void                 CreateBookElements(MqlBookInfo& book[], datetime now);
   void                 CreateOrRefreshElement(int buttonHeigh, int buttonWidth, int i, color clr, int ydistance);
   //--- Default constructor
                     BookEventHistogram(void);
                    ~BookEventHistogram(void);
  };
```

Not all functions are defined in this segment; however, they are completed in the remaining lines of the BookEventHistogram.mqh file.

Among the most important functions, CreateBookElements and CreateOrRefreshElement work together to ensure that existing elements are updated while creating new ones when necessary. The remaining functions serve to keep properties up to date or to return the values of certain object variables.

### Source code of the indicator:

The beginning of the code defines the number of plots and buffers as 3. A deeper analysis will reveal that, in reality, the root structure buffers of an MQL5 indicator are not used. However, this declaration facilitates the generation of code that ensures user interaction with certain properties during the indicator's initialization. In this case, our focus is on color properties, where the input scheme is designed to provide an intuitive and user-friendly experience.

Each plot is assigned two colors - one for buy orders and one for sell orders. This set of six colors is used to determine the color of each segment based on predefined criteria. Broadly speaking, the largest segments in the histogram are classified as "extremes", those above the average size as "mid-high", and the rest as "mid-low".

Colors are retrieved using the PlotIndexGetInteger function, which specifies the plot and the position within the plot from which the information should be extracted.

```
#define NUMBER_OF_PLOTS 3
#property indicator_chart_window
#property indicator_buffers NUMBER_OF_PLOTS
#property indicator_plots   NUMBER_OF_PLOTS
//--- Invisible plots
#property indicator_label1  "Extreme volume elements colors"
#property indicator_type1   DRAW_NONE
#property indicator_color1  C'212,135,114', C'155,208,226'
//---
#property indicator_label2  "Mid-high volume elements colors"
#property indicator_type2   DRAW_NONE
#property indicator_color2  C'217,111,86', C'124,195,216'
//---
#property indicator_label3  "Mid-low volume elements color"
#property indicator_type3   DRAW_NONE
#property indicator_color3  C'208,101,74', C'114,190,214'
#include "BookEventHistogram.mqh"
enum HistogramPosition
  {
   LEFT,      //<<<< Histogram on the left
   RIGHT,     //Histogram on the right >>>>
  };
enum HistogramProportion
  {
   A_QUARTER,   // A quarter of the chart
   A_THIRD,     // A third of the chart
   HALF,        // Half of the chart
  };
input  HistogramPosition position = RIGHT; // Indicator position
input  HistogramProportion proportion = A_QUARTER; // Histogram ratio (compared to chart width)
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
double volumes[];
color histogramColors[];
BookEventHistogram bookEvent;
```

Next, we introduce two enumerators designed to provide users with precise options when loading the indicator. We want to determine where the histogram should be drawn: on the right or left side of the chart. Additionally, the user must specify the proportion of the chart width that the histogram will occupy: one-fourth, one-third, or half of the chart. For instance, if the chart width is 500 pixels and the user selects the half-width option, histogram bars can range in size from 0 to 250 pixels.

Finally, in the source code BookEvents.mq5, the OnBookEvent and OnChartEvent functions will trigger most of the histogram update requests. The OnCalculate function does not play a role in the algorithm and is only retained for MQL syntax compliance.

### Using the Scripts and Indicator

The correct sequence for running the scripts and the indicator, ensuring consistency with the resources developed so far, is as follows:

- Run the script CloneSymbolTicksAndRates on the chart of the real symbol to be cloned.
- -\> BookEvents indicator (on the chart of the generated custom symbol)
- -\> GenerateBookEvent script (on the chart of the generated custom symbol)

The BookEvent is broadcast to all graphical instances of the targeted custom asset. Therefore, the indicator and event generator script can be executed on separate charts or within the same chart, as long as they reference the same custom symbol.

The animation below illustrates this sequence as well as the functionality of the indicator. I hope you enjoy it!

![Demo - BookEvents indicator](https://c.mql5.com/2/116/output5__1.gif)

### Conclusion

Depth of Market is undoubtedly a very important element for executing fast trades, especially in High Frequency Trading (HFT) algorithms. It is a type of market event that brokers provide for many trading symbols. Over time, brokers may expand the coverage and availability of such data for additional assets.

However, I believe it is not advisable to build a _trading_ _system_ solely based on the order book. Instead, the DOM can help identify liquidity zones and may exhibit some correlation with price movements. Therefore, combining order book analysis with other tools and indicators is a prudent approach to achieving consistent trading results.

There is room for future enhancements to the indicator, such as implementing mechanisms to store BookEvent data and later use them in backtesting, both for manual trading and for automated strategies.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/15748](https://www.mql5.com/pt/articles/15748)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15748.zip "Download all attachments in the single ZIP archive")

[BookEventsIndicator.zip](https://www.mql5.com/en/articles/download/15748/bookeventsindicator.zip "Download BookEventsIndicator.zip")(10.83 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Custom Indicator: Plotting Partial Entry, Exit and Reversal Deals for Netting Accounts](https://www.mql5.com/en/articles/12576)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/484618)**
(4)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
11 Apr 2025 at 13:33

I don't see any reason why you need a custom symbol. It's perfectly doable to save and replay book events on the standard symbol itself - both on history of a regular chart (for indicator display) and in the tester (for EA testing).


![Samuel Manoel De Souza](https://c.mql5.com/avatar/2024/12/674fb11c-8142.jpg)

**[Samuel Manoel De Souza](https://www.mql5.com/en/users/samuelmnl)**
\|
11 Apr 2025 at 14:57

This is very short article with very little code. Let see in next part if it makes sense have part 1, 2 and so on.

SYMBOL\_TICKS\_BOOKDEPTH gives the maximal number of requests shown in Depth of Market. Is incorrect that this property gives the same result as counting the number of levels in the DOM. It gives the maximal number not precise number.

You can very that using this script:

```
//+------------------------------------------------------------------+
//|                                                TestOnderBook.mq5 |
//|                           Copyright 2025, Samuel Manoel De Souza |
//|                          https://www.mql5.com/en/users/samuelmnl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Samuel Manoel De Souza"
#property link      "https://www.mql5.com/en/users/samuelmnl"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   MarketBookAdd(_Symbol);

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {

   MarketBookRelease(_Symbol);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick(void)
  {

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnBookEvent(const string& symbol)
  {
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   MqlBookInfo book[];
   MarketBookGet(_Symbol, book);
   int total = ArraySize(book);
   if(total == 0)
     {
      Print("there is no order available on the book");
      ExpertRemove();
      return;
     }

   int buy_levels = 0, sell_levels = 0;
   int buy_gaps = 0, sell_gaps = 0, gaps = 0;
   for(int i = 0; i < total; i++)
     {
      Print("price: ", book[i].price, ", volume: ", book[i].volume, ", type: ", EnumToString(book[i].type));
      buy_levels += book[i].type == BOOK_TYPE_BUY ? 1 : 0;
      sell_levels += book[i].type == BOOK_TYPE_SELL ? 1 : 0;
      if(i > 0)
        {
         bool is_gap = fabs(book[i].price - book[i - 1].price) >= 2 * tick_size;
         gaps += is_gap ? 1 : 0;
         buy_gaps += is_gap && book[i].type == book[i - 1].type && book[i].type == BOOK_TYPE_BUY ? 1 : 0;
         sell_gaps += is_gap && book[i].type == book[i - 1].type && book[i].type == BOOK_TYPE_SELL ? 1 : 0;
        }
     }

   Print("max levels: ", SymbolInfoInteger(_Symbol, SYMBOL_TICKS_BOOKDEPTH));
   Print("levels: ", total);
   Print("buy levels: ", buy_levels);
   Print("sell levels: ", sell_levels);
   Print("gaps: ", gaps);
   Print("buy gaps: ", buy_gaps);
   Print("sell gap: ", sell_gaps);
   ExpertRemove();
  }
//+------------------------------------------------------------------+
```

![Thomas Gardling](https://c.mql5.com/avatar/2023/12/658f4388-3ccb.jpg)

**[Thomas Gardling](https://www.mql5.com/en/users/oneandonly666)**
\|
18 May 2025 at 13:26

What happened to the trading system that was going to be developed with this indicator?


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
19 May 2025 at 08:11

**Samuel Manoel De Souza [#](https://www.mql5.com/ru/forum/480918#comment_56432361):**

This is a very short article with very little code. We'll see in the next part if it makes sense to have part 1, 2 and so on.

SYMBOL\_TICKS\_BOOKDEPTH gives the maximum number of requests displayed in the Depth of Market. It is incorrect that this property gives the same result as counting the number of levels in the DOM. It gives a maximum number, not an exact number.

You can do this with this script:

The article is super! I also wanted to write a TS on the stack - more precisely with the use of the quotes stack!

![Introduction to MQL5 (Part 15): A Beginner's Guide to Building Custom Indicators (IV)](https://c.mql5.com/2/133/Introduction_to_MQL5_Part_15___LOGO.png)[Introduction to MQL5 (Part 15): A Beginner's Guide to Building Custom Indicators (IV)](https://www.mql5.com/en/articles/17689)

In this article, you'll learn how to build a price action indicator in MQL5, focusing on key points like low (L), high (H), higher low (HL), higher high (HH), lower low (LL), and lower high (LH) for analyzing trends. You'll also explore how to identify the premium and discount zones, mark the 50% retracement level, and use the risk-reward ratio to calculate profit targets. The article also covers determining entry points, stop loss (SL), and take profit (TP) levels based on the trend structure.

![Statistical Arbitrage Through Mean Reversion in Pairs Trading: Beating the Market by Math](https://c.mql5.com/2/132/Statistical_Arbitrage_Through_Mean_Reversion_in_Pairs_Trading__LOGO.png)[Statistical Arbitrage Through Mean Reversion in Pairs Trading: Beating the Market by Math](https://www.mql5.com/en/articles/17735)

This article describes the fundamentals of portfolio-level statistical arbitrage. Its goal is to facilitate the understanding of the principles of statistical arbitrage to readers without deep math knowledge and propose a starting point conceptual framework. The article includes a working Expert Advisor, some notes about its one-year backtest, and the respective backtest configuration settings (.ini file) for the reproduction of the experiment.

![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (V): AnalyticsPanel Class](https://c.mql5.com/2/133/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_X_CODEIV___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (V): AnalyticsPanel Class](https://www.mql5.com/en/articles/17397)

In this discussion, we explore how to retrieve real-time market data and trading account information, perform various calculations, and display the results on a custom panel. To achieve this, we will dive deeper into developing an AnalyticsPanel class that encapsulates all these features, including panel creation. This effort is part of our ongoing expansion of the New Admin Panel EA, introducing advanced functionalities using modular design principles and best practices for code organization.

![From Basic to Intermediate: The Include Directive](https://c.mql5.com/2/92/Do_bvsico_ao_intermediyrio_Diretiva_Include___LOGO.png)[From Basic to Intermediate: The Include Directive](https://www.mql5.com/en/articles/15383)

In today's article, we will discuss a compilation directive that is widely used in various codes that can be found in MQL5. Although this directive will be explained rather superficially here, it is important that you begin to understand how to use it, as it will soon become indispensable as you move to higher levels of programming. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=roilphernkubqipmnfqffjbjnlrmffzq&ssn=1769092498705765312&ssn_dr=0&ssn_sr=0&fv_date=1769092498&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15748&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Trading%20System%20Based%20on%20the%20Order%20Book%20(Part%20I)%3A%20Indicator%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690924983786698&fz_uniq=5049231353880356754&sv=2552)

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