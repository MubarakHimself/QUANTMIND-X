---
title: Elder-Ray (Bulls Power and Bears Power)
url: https://www.mql5.com/en/articles/5014
categories: Trading, Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:35:47.267400
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/5014&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082978775940993665)

MetaTrader 5 / Examples


### Introduction

Elder-Ray trading system was described by Alexander Elder in his book "Trading for a Living". It is based on [Bulls Power](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bulls "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bulls") and [Bears Power](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears") oscillators, as well as [Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") trend-following indicator (EMA — exponential averaging).

The system is both simple and complex:

- it is simple if we perceive it literally: buy if a trend is upwards (EMA) and [Bears Power](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears") is below zero, but is increasing;
- it is complex, if we read more carefully and also look at the chart EMA and Bears Power are launched at: it turns out, it is not as simple as it seems since such conditions are few.

In this article, we will go through all the stages from simple to complex and check two types of trading system:

1. all indicators on the same chart (and therefore, on a single timeframe);
2. in combination with the "Triple choice" system.

The EAs in the article are meant for working with netting accounts only.

### Key points

To grasp the idea behind the trading system, it is necessary to understand each Elder-Ray element: price, EMA, highs and lows of Bulls Power and Bears Power indicators at each bar, as well as the strength of bulls and bears.

- **Price** — current agreement on the value of an asset. All buys are made while expecting the price rise, while all sells are made in anticipation of the price fall. A dealer can be performed only when there are eager buyers and sellers.
- **EMA** — exponential moving average. It reflects the average agreement on an asset value for a certain period. For example, EMA(13) on D1 is an average asset value agreement for the last 13 days. Why is it better to use an exponential rather than a simple moving average? A. Elder answered this question in section 4.2 ("Moving Averages") of his book. In short, EMA is more sensitive to trend changes as compared to a simple average.
- **Bulls Power maximum** shows the maximum bulls strength on a given bar. When the price grows, bulls earn profit, therefore bulls buy till the price reaches the maximum level. Bulls Power maximum is a moment when bulls want to move the price higher, but they do not have money anymore.
- **Bears Power minimum** shows the maximum bears strength on a given bar. Bears make profit when the price goes down and, therefore, sell till the price reaches its minimum. Bears Power minimum is a moment when bears want to move the price further down but are no more capable of doing that.
- **Bulls Power** shows the ability of bulls to raise the price above the average agreement on the asset value. As a rule, the strength of bulls is above zero. If it is below zero, then this means the bulls are in panic and are about to lose their power.
- **Bears Power** reflects the ability of bears to lower the price below the average agreement on the asset value. Usually, the strength of bears is below zero. If it is above zero, then bulls are too strong, and bears are about to lose their power.

### Option 1: All indicators on a single chart

We will explore futures and stocks on the D1 timeframe. All three indicators (Bulls Power, Bears Power and EMA) are placed on a single chart. The averaging period of all indicators is 13.

**Buy rules**

- trend goes up (according to EMA);
- Bears Power is below zero but goes up;
- pending Buy stop order is located above the maximum of the last two days, while protective stop loss is placed below the last minimum.

![CATDaily Buy signals](https://c.mql5.com/2/33/CATDaily_Buy_signals_v2.png)

CAT, Daily Buy signals

**Sell rules**

- trend goes down (according to EMA);
- Bulls Power is above zero but goes down;
- pending Sell stop order is located below the minimum of the last two days, while protective stop loss is placed above the last maximum.

![CATDaily Sell signal](https://c.mql5.com/2/33/CATDaily_Sell_signal.png)

CAT, Daily Sell signals

**Trading rules**

Looking at Figures 1 and 2, we can see that in the option " **All indicators on a single chart**", the buy and sell rules are triggered at roll-backs on a stable trend. There are quite a few such favorable moments, especially since the analyzed timeframe is D1. Therefore, in the option " **All indicators on a single chart**", we need to analyze a very large number of instruments to increase the frequency of transactions on a trading account.

D1 charts also have a considerable advantage: analysis of EMA slope, as well as Bulls Power and Bears Power indicators, can be performed only once a day — when a new bar appears. This is exactly how the EA is to work: wait for a new bar on D1 at each specified symbol and look for possible entries after that.

Since futures and stocks are traded only in netting mode, oppositely directed positions (hedging) cannot be applied here, but we can increase position volume. The EA can trade on the current symbol only or on several symbols stored in the text file. While all is clear with the current symbol, selecting several symbols may present the following issues:

- we need to specify about a hundred symbols of one market (for example, securities only);
- we need to specify a lot of symbols from different markets (for example, futures and securities).

How to select all symbols from one market? Suppose, we have " **CAT**" symbol located at " **Stock Markets**\ **USA**\ **NYSE/NASDAQ(SnP100)**\\CAT"

![Symbols Specification](https://c.mql5.com/2/33/Symbols_Specification.png)

Suppose that this symbol suits us, and we want to select all other instruments from the "\\NYSE/NASDAQ(SnP100)\\" branch. In this case, we can do the following:

1. open the chart of this symbol;
2. launch the script (let's name it **Symbols on the specified path.mq5**) that will receive the symbol path (in the above example, for " **CAT**" symbol, it is " **Stock Markets**\ **USA**\ **NYSE/NASDAQ(SnP100)**") and save all symbols from the obtained path to the text file. The text file is to be saved to the Common Data Folder;
3. it only remains to set the text file name in the EA settings.

If we need symbols from several markets, then run the script on each of the markets (list of symbols is saved to its unique file) and combine both text files manually.

Implementation of the **Symbols on the specified path.mq5** script is to be described below.

**Assembling the EA. Option 1: All indicators on a single chart**

**Symbols on the specified path.mq5** — script to be used for obtaining the text file with the symbols.

NOTE: only the text **"Everything is fine. There are no errors"** in the Experts tab guarantees that the script's work has been successful, and the obtained file with symbols can be used for the EA operation!

To shorten the code of file operations, the [CFileTxt](https://www.mql5.com/en/docs/standardlibrary/fileoperations/cfiletxt) class is connected, and the work with the text file is performed by **m\_file\_txt** — [CFileTxt](https://www.mql5.com/en/docs/standardlibrary/fileoperations/cfiletxt) class object. The script performs its work in seven steps:

```
//+------------------------------------------------------------------+
//|                                Symbols on the specified path.mq5 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.002"
#property script_show_inputs
//---
#include <Files\FileTxt.mqh>
CFileTxt       m_file_txt;                   // file txt object
//--- input parameters
input string   InpFileName="Enter a unique name.txt";  // File name
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- STEP 1
   string current_path="";
   if(!SymbolInfoString(Symbol(),SYMBOL_PATH,current_path))
     {
      Print("ERROR: SYMBOL_PATH");
      return;
     }
//--- STEP 2
   string sep_="\\";                 // A separator as a character
   ushort u_sep_;                    // The code of the separator character
   string result_[];                 // An array to get strings
//--- Get the separator code
   u_sep_=StringGetCharacter(sep_,0);
//--- Split the string to substrings
   int k_=StringSplit(current_path,u_sep_,result_);
//--- STEP 3
//--- Now output all obtained strings
   if(k_>0)
     {
      current_path="";
      for(int i=0;i<k_-1;i++)
         current_path=current_path+result_[i]+sep_;
     }
//--- STEP 4
   string symbols_array[];
   int symbols_total=SymbolsTotal(false);
   for(int i=0;i<symbols_total;i++)
     {
      string symbol_name=SymbolName(i,false);
      string symbol_path="";
      if(!SymbolInfoString(symbol_name,SYMBOL_PATH,symbol_path))
         continue;
      if(StringFind(symbol_path,current_path,0)==-1)
         continue;

      int size=ArraySize(symbols_array);
      ArrayResize(symbols_array,size+1,10);
      symbols_array[size]=symbol_name];
     }
//--- STEP 5
   int size=ArraySize(symbols_array);
   if(size==0)
     {
      PrintFormat("ERROR: On path \"%s\" %d symbols",current_path,size);
      return;
     }
   PrintFormat("On path \"%s\" %d symbols",current_path,size);
//--- STEP 6
   if(m_file_txt.Open(InpFileName,FILE_WRITE|FILE_COMMON)==INVALID_HANDLE)
     {
      PrintFormat("ERROR: \"%s\" file in the Data Folder Common folder is not created",InpFileName);
      return;
     }
//--- STEP 7
   for(int i=0;i<size;i++)
      m_file_txt.WriteString(symbols_array[i]+"\r\n");
   m_file_txt.Close();
   Print("Everything is fine. There are no errors");
//---
  }
//+------------------------------------------------------------------+
```

Script operation algorithm:

- STEP 1: SYMBOL\_PATH (path in the symbols tree) is defined for the current symbol;
- STEP 2: the obtained path is divided into substrings with the "\\" separator;
- STEP 3: re-assemble the current path without the last substring, since it contains the symbol name;
- STEP 4: loop through all available symbols; if the symbol's path in the symbols tree matches the current one, select the symbol name and add it to the detected symbols array;
- STEP 5: check the size of the detected symbols array;
- STEP 6: create the file;
- STEP 7: write our array of detected symbols to the file and close it.

**Elder-Ray 1** — EA (or several EAs) with **1**.xxx version numbers to be traded according to the option 1: all indicators on a single chart.

**How to set the position volume — the minimum lot may be different**

Let's conduct a simple experiment: check the minimum lot size of futures and securities: go through all symbols located at the same path as the current one (similar to the **Symbols on the specified path.mq5** script), but inside saving symbols to the file, display statistics on the minimum lot size.

**Gets minimal volume.mq5** — script used to display statistics on the minimum volume of the symbol group. The script bypasses the symbol group and accumulates statistics ( **_minimal volume to close a deal_** and **_counter_**) in the two-dimensional array:

```
//--- STEP 4
/*
   symbols_array[][2]:
   [*][minimal volume to close a deal]
   [*][counter]
*/
```

Full script code:

```
//+------------------------------------------------------------------+
//|                                          Gets minimal volume.mq5 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.000"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- STEP 1
   string current_path="";
   if(!SymbolInfoString(Symbol(),SYMBOL_PATH,current_path))
     {
      Print("ERROR: SYMBOL_PATH");
      return;
     }
//--- STEP 2
   string sep_="\\";                 // A separator as a character
   ushort u_sep_;                    // The code of the separator character
   string result_[];                 // An array to get strings
//--- Get the separator code
   u_sep_=StringGetCharacter(sep_,0);
//--- Split the string to substrings
   int k_=StringSplit(current_path,u_sep_,result_);
//--- STEP 3
//--- Now output all obtained strings
   if(k_>0)
     {
      current_path="";
      for(int i=0;i<k_-1;i++)
         current_path=current_path+result_[i]+sep_;
     }
//--- STEP 4
/*
   symbols_array[][2]:
   [*][minimal volume to close a deal]
   [*][counter]
*/
   double symbols_array[][2];
   int symbols_total=SymbolsTotal(false);
   for(int i=0;i<symbols_total;i++)
     {
      string symbol_name=SymbolName(i,false);
      string symbol_path="";
      if(!SymbolInfoString(symbol_name,SYMBOL_PATH,symbol_path))
         continue;
      if(StringFind(symbol_path,current_path,0)==-1)
         continue;

      double min_volume=0.0;
      if(!SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_MIN,min_volume))
         continue;
      int size=ArrayRange(symbols_array,0);
      bool found=false;
      for(int j=0;j<size;j++)
        {
         if(symbols_array[j][0]==min_volume)
           {
            symbols_array[j][1]=symbols_array[j][1]+1;
            found=true;
            continue;
           }
        }
      if(!found)
        {
         ArrayResize(symbols_array,size+1,10);
         symbols_array[size][0]=min_volume;
         symbols_array[size][1]=1.0;
        }
     }
//--- STEP 5
   int size=ArrayRange(symbols_array,0);
   if(size==0)
     {
      PrintFormat("ERROR: On path \"%s\" %d symbols",current_path,size);
      return;
     }
//--- STEP 6
   for(int i=0;i<size;i++)
      PrintFormat("Minimal volume %.2f occurs %.1f times",symbols_array[i][0],symbols_array[i][1]);
   Print("Everything is fine. There are no errors");
//---
  }
//+------------------------------------------------------------------+
```

Script operation algorithm:

- STEP 1: SYMBOL\_PATH (path in the symbols tree) is defined for the current symbol;
- STEP 2: the obtained path is divided into substrings with the "\\" separator;
- STEP 3: re-assemble the current path without the last substring, since it contains the symbol name;
- STEP 4: loop through all available symbols; if the symbol's path in the symbols tree matches the current one, get the minimum symbol volume and perform a search in the symbols array. If such a value is already present, increase the counter. If there is no such value, add to the array and set the counter to "1.0";
- STEP 5: check the size of the detected symbols array;
- STEP 6: display statistics.

The result of launching on securities:

```
Gets minimal volume (CAT,D1)    Minimal volume 1.00 occurs 100.0 times
```

and on futures:

```
Gets minimal volume (RTSRIU8,D1)        Minimal volume 1.00 occurs 77.0 times
```

\- on the two markets, the lot size is the same — 1.0.

Thus, let's not over-complicate the system and set "1.0" as the minimum lot.

**Visualizing applied indicators**

When you launch a visualized test in the tester, you can see the indicators applied by the EA. But when the EA is launched on the terminal chart, the indicators are not displayed. In this trading system, I want to see these indicators on the chart for visual control of the EA's work. It should look like this:

![CATDaily visual trading](https://c.mql5.com/2/33/CATDaily_visual_trading.png)

As you can see, here I used custom color and line width settings for all indicators (of course, this was done manually). For auto visualization of the indicators applied on the terminal chart, we need to slightly re-write Moving Average, Bulls Power and Bears Power indicators. I already implemented a similar thing in the [Custom Moving Average Input Color](https://www.mql5.com/en/code/19864) code — the indicator color was included into the inputs: this input parameter is available when creating an indicator from an EA. Now, we just need to develop three more similar indicators.

You can download these indicators ( [Custom Moving Average Inputs](https://www.mql5.com/en/code/21779), [Custom Bulls Power Inputs](https://www.mql5.com/en/code/21780) and [Custom Bears Power Inputs](https://www.mql5.com/en/code/21781)) in CodeBase. Place the downloaded indicators to the root of \[data folder\]\\MQL5\ **Indicators**\\.

**Elder-Ray 1.001.mq5** — **visualizing the applied indicators** you can set the color and width for. It works both in the strategy tester and when launching the EA on the chart:

![Elder-Ray 1.001](https://c.mql5.com/2/33/Elder-Ray_1.001.gif)

How is this implemented?

The main condition is the presence of the [Custom Moving Average Inputs](https://www.mql5.com/en/code/21779), [Custom Bulls Power Inputs](https://www.mql5.com/en/code/21780) and [Custom Bears Power Inputs](https://www.mql5.com/en/code/21781) indicators in \[data folder\]\\MQL5\ **Indicators**\\

![Three indicators](https://c.mql5.com/2/33/Three_indicators.png)

The indicators' look is managed and the period is set in the inputs, while for working with the indicators, three variables the indicator handles are to be stored in are declared ( _**handle\_iCustom\_MA**_, **_handle\_iCustom\_Bulls_** and **_handle\_iCustom\_Bears_**).

```
//+------------------------------------------------------------------+
//|                                                  Elder-Ray 1.mq5 |
//|                              Copyright © 2018, Vladimir Karputov |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Vladimir Karputov"
#property link      "http://wmua.ru/slesar/"
#property version   "1.000"
//---

//---
enum ENUM_INPUT_SYMBOLS
  {
   INPUT_SYMBOLS_CURRENT=0,   // current symbol
   INPUT_SYMBOLS_FILE=1,      // text file
  };
//--- input parameters
input ENUM_INPUT_SYMBOLS   InpInputSymbol       = INPUT_SYMBOLS_FILE;   // works on ...
input uint                 InpNumberMinLots     = 1;                    // Number of minimum lots
//--- Custom Moving Average Inputs
input int                  Inp_MA_ma_period     = 13;                   // MA: averaging period
input int                  Inp_MA_ma_shift      = 0;                    // MA: horizontal shift
input ENUM_MA_METHOD       Inp_MA_ma_method     = MODE_EMA;             // MA: smoothing type
input ENUM_APPLIED_PRICE   Inp_MA_applied_price = PRICE_CLOSE;          // MA: type of price
input color                Inp_MA_Color         = clrChartreuse;        // MA: Color
input int                  Inp_MA_Width         = 2;                    // MA: Width
//--- Custom Bulls Power Inputs
input int                  Inp_Bulls_ma_period  = 13;                   // Bulls Power: averaging period
input color                Inp_Bulls_Color      = clrBlue;              // Bulls Power: Color
input int                  Inp_Bulls_Width      = 2;                    // Bulls Power: Width
//--- Custom Bears Power Inputs
input int                  Inp_Bears_ma_period  = 13;                   // Bears Power: averaging period
input color                Inp_Bears_Color      = clrRed;               // Bears Power: Color
input int                  Inp_Bears_Width      = 2;                    // Bears Power: Width

int    handle_iCustom_MA;                    // variable for storing the handle of the iCustom indicator
int    handle_iCustom_Bulls;                 // variable for storing the handle of the iCustom indicator
int    handle_iCustom_Bears;                 // variable for storing the handle of the iCustom indicator
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
```

In [OnInit()](https://www.mql5.com/en/docs/event_handlers/oninit), we create handles of custom indicators ( [iCustom](https://www.mql5.com/en/docs/indicators/icustom) is applied), and created indicators are added to the chart ( [ChartIndicatorAdd](https://www.mql5.com/en/docs/chart_operations/chartindicatoradd) is applied).

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create handle of the indicator iCustom
   handle_iCustom_MA=iCustom(Symbol(),Period(),"Custom Moving Average Inputs",
                             Inp_MA_ma_period,
                             Inp_MA_ma_shift,
                             Inp_MA_ma_method,
                             Inp_MA_Color,
                             Inp_MA_Width,
                             Inp_MA_applied_price);
//--- if the handle is not created
   if(handle_iCustom_MA==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the iCustom indicator (\"Custom Moving Average Inputs\") for the symbol %s/%s, error code %d",
                  Symbol(),
                  EnumToString(Period()),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
     }

//--- create handle of the indicator iCustom
   handle_iCustom_Bulls=iCustom(Symbol(),Period(),"Custom Bulls Power Inputs",
                                Inp_Bulls_ma_period,
                                Inp_Bulls_Color,
                                Inp_Bulls_Width);
//--- if the handle is not created
   if(handle_iCustom_Bulls==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the iCustom indicator (\"Custom Bulls Power Inputs\") for the symbol %s/%s, error code %d",
                  Symbol(),
                  EnumToString(Period()),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
     }

//--- create handle of the indicator iCustom
   handle_iCustom_Bears=iCustom(Symbol(),Period(),"Custom Bears Power Inputs",
                                Inp_Bears_ma_period,
                                Inp_Bears_Color,
                                Inp_Bears_Width);
//--- if the handle is not created
   if(handle_iCustom_Bears==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the iCustom indicator (\"Custom Bears Power Inputs\") for the symbol %s/%s, error code %d",
                  Symbol(),
                  EnumToString(Period()),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
     }

   ChartIndicatorAdd(0,0,handle_iCustom_MA);
   int windows_total=(int)ChartGetInteger(0,CHART_WINDOWS_TOTAL);
   ChartIndicatorAdd(0,windows_total,handle_iCustom_Bulls);
   ChartIndicatorAdd(0,windows_total+1,handle_iCustom_Bears);
//---
   return(INIT_SUCCEEDED);
  }
```

**Saving resources.** **Elder-Ray 1.010.mq5**

There may be about a hundred analyzed symbols of a single group. Thus, the issue of saving RAM becomes relevant, since each chart is to contain three indicators. Like with a minimum lot, the best thing to do is to check the resource consumption by the EA. At the same time, we will also make a little progress in assembling our EA by adding the code for reading symbol names of one group from the text file and working with them.

The [CFileTxt](https://www.mql5.com/en/docs/standardlibrary/fileoperations/cfiletxt) class (we already applied it in the script **Symbols on the specified path.mq5**) is included to the EA. Its **m\_file\_txt** object is responsible for accessing the text file and reading data from the file. We also include the [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class. Its object **m\_symbol** is responsible for checking the existence of a symbol and adding it to the Market Watch window. Why did I choose [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo), rather than implementing the function via SymbolInfoInteger and SymbolSelect? All is simple: in the CSymbolInfo class, all the code for checking, adding or informing of errors is hidden inside the class, and we need only to add the following three strings in the EA:

```
         if(!m_symbol.Name(name)) // sets symbol name
           {
            m_file_txt.Close();
            return(INIT_FAILED);
           }
```

Here we should recall that all symbols of a multi-currency EA should be added to the Market Watch window: [Enable required symbols in Market Watch for multi-currency Expert Advisors](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization#mw "https://www.metatrader5.com/en/terminal/help/testing#mw").

Thus, the EA works according to the following algorithm: a text file is opened in [OnInit()](https://www.mql5.com/en/docs/event_handlers/oninit), a symbol is read and an attempt is immediately made to create three custom indicators (Custom Moving Average Inputs, Custom Bulls Power Inputs and Custom Bears Power Inputs) for the read symbol on the current timeframe. If that attempt fails (for example, due to insufficient number of bars for creating Custom Moving Average Inputs), move on further along the loop. If the indicators are created, write the symbol name to m\_symbols\_array, while the handles of three indicators are sent to the three-dimensional m\_handles\_array. Thus, by the first dimension, both arrays synchronously contain data on the symbol name and symbol's handles:

```
//---
   if(m_file_txt.Open(InpFileName,FILE_READ|FILE_COMMON)==INVALID_HANDLE)
     {
      PrintFormat("ERROR: \"%s\" file in the Data Folder Common folder is not open: %d",InpFileName,GetLastError());
      return(INIT_FAILED);
     }
//--- symbol info object OR file txt object
   if(InpInputSymbol==INPUT_SYMBOLS_FILE)
     {
      //--- read data from the file
      int counter=0;
      while(!m_file_txt.IsEnding())
        {
         counter++;
         Print("Iteration ",counter);
         string name=m_file_txt.ReadString();
         if(!m_symbol.Name(name)) // sets symbol name
           {
            m_file_txt.Close();
            return(INIT_FAILED);
           }
         int MA,Bulls,Bears;
         if(!CreateHandles(name,Period(),MA,Bulls,Bears))
            continue; //return(INIT_FAILED);
         int size=ArraySize(m_symbols_array);
         ArrayResize(m_symbols_array,size+1,10);
         ArrayResize(m_handles_array,size+1,10);
         m_symbols_array[size]=name;
         m_handles_array[size][0]=MA;
         m_handles_array[size][1]=Bulls;
         m_handles_array[size][2]=Bears;
        }
      m_file_txt.Close();
     }
   else
     {
      if(!m_symbol.Name(Symbol())) // sets symbol name
         return(INIT_FAILED);
     }
//---
   ChartIndicatorAdd(0,0,handle_iCustom_MA);
```

Indicator handles are created in CreateHandles().

Thus, memory consumption was measured via TERMINAL\_MEMORY\_USED and visually in the Windows 10 task manager. To define memory consumption by steps, some strings were deliberately disabled (commented out) in the version 1.010. In the final version 1.010, all the strings for adding symbols and creating indicators were uncommented.

- The EA is launched in the usual way — by placing it to the chart:

  - launching the terminal (symbols from the text file have not yet been added to the Market Watch window) — TERMINAL\_MEMORY\_USED 345 MB, task manager from 26 to 90 MB;
  - adding about a hundred symbols to the Market Watch window — TERMINAL\_MEMORY\_USED 433 MB, task manager + 10 MB;
  - creating three indicators per each symbol — TERMINAL\_MEMORY\_USED 5523 MB, task manager 300 MB.

- Launching the tester (no visualization) — TERMINAL\_MEMORY\_USED 420 MB, while in the task manager — 5 GB.

Conclusion: TERMINAL\_MEMORY\_USED shows the total consumption of RAM and disk space. Since the RAM consumption in the normal mode does not exceed 300 MB, there is no need to save resources.

**Trend (EMA) is...**

The main EA objective is to define the trend ( **EMA**) direction. A trend cannot be determined by a single bar. We need data from several bars. Let's mark this parameter as " **bars**". Below are three securities charts — CAT, MCD and V. The trend is to be defined as follows: "+1" for uptrend, "0" for no trend and "—1" for downtrend

![Trend (EMA) is ...](https://c.mql5.com/2/33/Trend_xEMAq_is_1.png)

" **CAT**" chart shows trend "0" (4 bullish and 4 bearish bars, while the change is insignificant on the remaining bars), " **MCD**" shows trend "—1" (8 bearish bars, while others are in the uncertain state). Finally, " **V**" displays trend "0" (6 bullish bars, while 2 or 3 are bearish ones). We may need the ' **different**' parameter — minimum difference between the indicator readings on neighboring bars.

**Defining trend.** **Elder-Ray 1.020.mq5**

Trend existence conditions: **EMA** along **bars** should be in a certain direction. The two additional parameters should probably be checked afterwards:

- **different** — minimum difference between the indicator readings on neighboring bars;
- **trend percentage** — minimum percentage of the indicator readings in a single direction (on the screenshot: **CAT** symbol — **EMA** indicator is oppositely directed on the **bars** segment, while on **MCD**, all (or almost all) **EMA** indicator readings have a certain direction).

Additions and removals in the version 1.020:

- the **different** parameter is not implemented — minimum difference between the indicator readings on neighboring bars;
- "—" enum ENUM\_INPUT\_SYMBOLS enumeration — the EA is to work only with symbols from the text file;
- "+" parameter **number of bars for identifying the trend —** number of bars for identifying a trend by EMA;
- "+" parameter **minimum percentage of the trend** — minimum trend quality (certain direction);
- "+" **m\_prev\_bars** array — for storing previous bar open time;
- "+" 60 sec timer — check for a new bar.

**The block for catching a new bar and defining a trend direction**

In [OnTimer()](https://www.mql5.com/en/docs/event_handlers/ontimer), go through the symbols array ( **m\_symbols\_array**) downloaded from the text file once per 60 seconds and catch a new bar on a symbol from the array. Get EMA indicator data sufficient to determine a trend to  the **ema\_array** array. Perform calculation: number of bars the indicator moved upwards and downwards. Print detected patterns.

```
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---
   int size=ArraySize(m_symbols_array);
   for(int i=0;i<size;i++)
     {
      //--- we work only at the time of the birth of new bar
      datetime PrevBars=m_prev_bars[i];
      datetime time_0=iTime(m_symbols_array[i],Period(),0);
      if(time_0==PrevBars)
         continue;
      m_prev_bars[i]=time_0;
      double ema_array[];
      ArraySetAsSeries(ema_array,true);
/*
      m_handles_array[*][MA][Bulls][Bears]
*/
      if(!iMAGetArray(m_handles_array[i][0],0,InpBarsEMA+2,ema_array))
         continue;
      int upwards=0;
      int downward=0;
      for(int j=1;j<InpBarsEMA+1;j++)
        {
         if(ema_array[j]>ema_array[j+1])
            upwards++;
         else if(ema_array[j]<ema_array[j+1])
            downward++;
        }
      if((double)upwards>InpBarsEMA*InpMinTrendPercentage/100.0)
         Print("On ",m_symbols_array[i]," trend UP!");
      else if((double)downward>InpBarsEMA*InpMinTrendPercentage/100.0)
         Print("On ",m_symbols_array[i]," trend DOWN!");
      else
         Print("On ",m_symbols_array[i]," trend NONE!");
     }
  }
```

Trend search result. Settings: **number of bars for identifying the trend** — 6, **minimum percentage of the trend** — 75%. Keep in mind that a zero bar is not taken into account when working at the moment a new bar appears:

![Trend on symbols](https://c.mql5.com/2/33/Trend_on_symbols_1.png)

![Trend on symbols](https://c.mql5.com/2/33/Trend_on_symbols_2.png)

**Setting pending orders (Buy stop or Sell stop).** **Elder-Ray 1.030.mq5**

Is it possible to avoid the lack of funds error when opening a position? Since we work with pending orders, the answer is "No". There may be half-way measures, but nothing can be guaranteed. The main reason is that no one knows at what point a pending order is activated and whether it is activated at all.

Now that the EA knows how to define a trend, we need to use the **Buy** and **Sell rules** to find points suitable for setting a pending order. For BUY, there will be a simple check: Bears Power on bar #1 should be less than zero and greater than Bears Power on bar #2. For SELL, the condition is the opposite: Bulls Power on bar #1 should exceed zero and less than Bears Power on bar #2.

Describing the strategy, Alexander Elder pointed out for opening a buy position, protective stop loss should be placed below the last Low, while for opening a sell position, it should be placed above the last High. The very concept of "Last" is rather blurred, so I checked the two options:

1. setting a stop loss by bar #1 prices and
2. searching for the nearest extreme point.

The option 1 turned out to be inviable - stop loss activations were too frequent. Therefore, I implemented the option 2 (searching for the nearest extremum) in the **Elder-Ray 1.030.mq5** EA code.

**Searching for the nearest extremum**

The function look for the nearest extremum:

![Nearest extremum](https://c.mql5.com/2/33/Nearest_extremum.png)

If no extremum is found or an error is detected, 'false' is returned:

```
//+------------------------------------------------------------------+
//| Find the nearest extremum                                        |
//+------------------------------------------------------------------+
bool NearestExtremum(ENUM_SERIESMODE type,double &price)
  {
   if(type==MODE_LOW)
     {
      //--- search for the nearest minimum
      double low_array[];
      ArraySetAsSeries(low_array,true);
      int copy=CopyLow(m_symbol.Name(),Period(),0,100,low_array);
      if(copy==-1)
         return(false);
      double low=DBL_MAX;
      for(int k=0;k<copy;k++)
        {
         if(low_array[k]<low)
            low=low_array[k];
         else if(low_array[k]>low)
            break;
        }
      if(low!=DBL_MAX)
        {
         price=low;
         return(true);
        }
     }
   else if(type==MODE_HIGH)
     {
      //--- search for the nearest maximum
      double high_array[];
      ArraySetAsSeries(high_array,true);
      int copy=CopyHigh(m_symbol.Name(),Period(),0,100,high_array);
      if(copy==-1)
         return(false);
      double high=DBL_MIN;
      for(int k=0;k<copy;k++)
        {
         if(high_array[k]>high)
            high=high_array[k];
         else if(high_array[k]<high)
            break;
        }
      if(high!=DBL_MIN)
        {
         price=high;
         return(true);
        }
     }
//---
   return(false);
  }
```

Additions and removals in the version 1.030:

- "+" [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) trading class (and **m\_position** is its object);
- "+" [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) trading class (and **m\_trade** is its object);
- "+" [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) trading class (and **m\_order** is its object);
- "+" trailing (Trailing Stop and Trailing Step parameters);
- "+" **magic number** — unique EA identifier;
- "+" [OnInit()](https://www.mql5.com/en/docs/event_handlers/oninit) — check the account type: if this is a hedging account, disable trading and unload with the error;
- OnInit() — visualization order changed: if the strategy tester is launched and the current symbol (the one the EA is launched on) is present in the text file, no indicators are added to the current symbol ( [ChartIndicatorAdd](https://www.mql5.com/en/docs/chart_operations/chartindicatoradd) is not applied);
- [OnTimer()](https://www.mql5.com/en/docs/event_handlers/ontimer) — added signal confirmation code and trading operations for setting pending Buy Stop and Sell Stop orders;
- [OnTradeTransaction()](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) — added the compensation mechanism if there was a position reversal or partial closing;
- "+" when the Compensation Algorithm is activated, a stop loss is NOT SET in OnTradeTransaction(). Instead, the trailing function is improved: if a position without a stop loss is detected when checking positions, a stop loss is set according to the rule of searching for the nearest extreme point;
- "+" added the m\_magic\_compensation variable — compensation trades identifier.

To deal with situations when a pending order opposite to the current position is triggered, we need to consider three typical situations after a Buy Stop pending order is triggered:

| # | Existing position, volume | Pending order activated, volume | Resulting position, volume | Note to a pending order trigger moment | Position's previous magic | Compensation algorithm (NB: magic is set to m\_magic\_compensation before the compensation) | Position's new magic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Sell 1.0 | Buy Stop 3.0 | Buy 2.0 | Position reversal ( [DEAL\_ENTRY\_INOUT](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_entry) deal direction) | **m\_magic** | Open additional Buy with the volume of 3.0 — 2.0 = 1.0 | m\_magic\_compensation |
| 2 | Sell 1.0 | Buy Stop 1.0 | --- | Closing position in full ( [DEAL\_ENTRY\_OUT](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_entry) deal direction) | **m\_magic** | Searching for a position. If there is no position, open Buy with the volume of 1.0 | m\_magic\_compensation |
| 3 | Sell 2.0 | Buy Stop 1.0 | Sell 1.0 | Closing position partially ( [DEAL\_ENTRY\_OUT](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_entry) deal direction) | **m\_magic** | Searching for a position. If a position is present and it is opposite to Buy, open Buy with the volume of 1.0 + 1.0 = 2.0 | m\_magic\_compensation |

For each of the three cases, I prepared a printout of deals and orders (test on a real netting account, as well as in the tester). To generate a report on deals and orders, I used the code from the [History Deals and Orders](https://www.mql5.com/en/code/19019) script.

#1: Sell 1.0 -> Buy Stop 3.0

```
Sell 1.0, Buy Stop 3.0

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|1                   |0                   |2016.12.08 00:00:00 |1481155200000       |DEAL_TYPE_BALANCE   |DEAL_ENTRY_IN       |0                   |DEAL_REASON_CLIENT  |0
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|0.00                |0.00000             |0.00                |0.00                |50000.00            |                    |                                         |
Order 0 is not found in the trade history between the dates 2010.08.07 11:06:20 and 2018.08.10 00:00:00

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|2                   |2                   |2016.12.08 00:00:00 |1481155200100       |DEAL_TYPE_SELL      |DEAL_ENTRY_IN       |15489               |DEAL_REASON_EXPERT  |2
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |79.31               |0.00                |0.00                |0.00                |V                   |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|2                   |2016.12.08 00:00:00 |ORDER_TYPE_SELL     |ORDER_STATE_FILLED  |2016.12.08 00:00:00 |2016.12.08 00:00:00 |1481155200100       |1481155200100       |ORDER_FILLING_FOK
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |15489               |ORDER_REASON_EXPERT |2                   |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|1.00                |0.00                |79.31               |0.00                |0.00                |79.39               |0.00
|Symbol              |Comment                                  |Extarnal id
|V                   |                                         |

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|3                   |3                   |2016.12.08 17:27:37 |1481218057877       |DEAL_TYPE_BUY       |DEAL_ENTRY_INOUT    |15489               |DEAL_REASON_EXPERT  |2
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|3.00                |79.75               |0.00                |0.00                |-0.44               |V                   |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|3                   |2016.12.08 14:30:00 |ORDER_TYPE_BUY_STOP |ORDER_STATE_FILLED  |2016.12.08 14:30:00 |2016.12.08 17:27:37 |1481207400100       |1481218057877       |ORDER_FILLING_RETURN
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |15489               |ORDER_REASON_EXPERT |2                   |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|3.00                |0.00                |79.74               |75.17               |0.00                |79.74               |0.00
|Symbol              |Comment                                  |Extarnal id
|V                   |                                         |

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|4                   |4                   |2016.12.09 23:59:00 |1481327940000       |DEAL_TYPE_SELL      |DEAL_ENTRY_OUT      |0                   |DEAL_REASON_CLIENT  |2
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|2.00                |79.13               |0.00                |0.00                |-1.24               |V                   |end of test                              |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|4                   |2016.12.09 23:59:00 |ORDER_TYPE_SELL     |ORDER_STATE_FILLED  |2016.12.09 23:59:00 |2016.12.09 23:59:00 |1481327940000       |1481327940000       |ORDER_FILLING_FOK
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |0                   |ORDER_REASON_CLIENT |2                   |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|2.00                |0.00                |79.13               |0.00                |0.00                |79.13               |0.00
|Symbol              |Comment                                  |Extarnal id
|V                   |end of test                              |
```

#2: Sell 1.0 -> Buy Stop 1.0

```
Sell 1.0, Buy Stop 1.0

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|1                   |0                   |2016.12.08 00:00:00 |1481155200000       |DEAL_TYPE_BALANCE   |DEAL_ENTRY_IN       |0                   |DEAL_REASON_CLIENT  |0
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|0.00                |0.00000             |0.00                |0.00                |50000.00            |                    |                                         |
Order 0 is not found in the trade history between the dates 2010.08.07 11:06:20 and 2018.08.10 00:00:00

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|2                   |2                   |2016.12.08 00:00:00 |1481155200100       |DEAL_TYPE_SELL      |DEAL_ENTRY_IN       |15489               |DEAL_REASON_EXPERT  |2
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |79.31               |0.00                |0.00                |0.00                |V                   |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|2                   |2016.12.08 00:00:00 |ORDER_TYPE_SELL     |ORDER_STATE_FILLED  |2016.12.08 00:00:00 |2016.12.08 00:00:00 |1481155200100       |1481155200100       |ORDER_FILLING_FOK
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |15489               |ORDER_REASON_EXPERT |2                   |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|1.00                |0.00                |79.31               |0.00                |0.00                |79.39               |0.00
|Symbol              |Comment                                  |Extarnal id
|V                   |                                         |

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|3                   |3                   |2016.12.08 17:27:37 |1481218057877       |DEAL_TYPE_BUY       |DEAL_ENTRY_OUT      |15489               |DEAL_REASON_EXPERT  |2
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |79.75               |0.00                |0.00                |-0.44               |V                   |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|3                   |2016.12.08 14:30:00 |ORDER_TYPE_BUY_STOP |ORDER_STATE_FILLED  |2016.12.08 14:30:00 |2016.12.08 17:27:37 |1481207400100       |1481218057877       |ORDER_FILLING_RETURN
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |15489               |ORDER_REASON_EXPERT |2                   |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|1.00                |0.00                |79.74               |75.17               |0.00                |79.74               |0.00
|Symbol              |Comment                                  |Extarnal id
|V                   |                                         |
```

#3: Sell 2.0 -> Buy Stop 1.0

```
Sell 2.0, Buy Stop 1.0

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|1                   |0                   |2016.12.08 00:00:00 |1481155200000       |DEAL_TYPE_BALANCE   |DEAL_ENTRY_IN       |0                   |DEAL_REASON_CLIENT  |0
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|0.00                |0.00000             |0.00                |0.00                |50000.00            |                    |                                         |
Order 0 is not found in the trade history between the dates 2010.08.07 11:06:20 and 2018.08.10 00:00:00

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|2                   |2                   |2016.12.08 00:00:00 |1481155200100       |DEAL_TYPE_SELL      |DEAL_ENTRY_IN       |15489               |DEAL_REASON_EXPERT  |2
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|2.00                |79.31               |0.00                |0.00                |0.00                |V                   |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|2                   |2016.12.08 00:00:00 |ORDER_TYPE_SELL     |ORDER_STATE_FILLED  |2016.12.08 00:00:00 |2016.12.08 00:00:00 |1481155200100       |1481155200100       |ORDER_FILLING_FOK
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |15489               |ORDER_REASON_EXPERT |2                   |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|2.00                |0.00                |79.31               |0.00                |0.00                |79.39               |0.00
|Symbol              |Comment                                  |Extarnal id
|V                   |                                         |

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|3                   |3                   |2016.12.08 17:27:37 |1481218057877       |DEAL_TYPE_BUY       |DEAL_ENTRY_OUT      |15489               |DEAL_REASON_EXPERT  |2
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |79.75               |0.00                |0.00                |-0.44               |V                   |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|3                   |2016.12.08 14:30:00 |ORDER_TYPE_BUY_STOP |ORDER_STATE_FILLED  |2016.12.08 14:30:00 |2016.12.08 17:27:37 |1481207400100       |1481218057877       |ORDER_FILLING_RETURN
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |15489               |ORDER_REASON_EXPERT |2                   |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|1.00                |0.00                |79.74               |75.17               |0.00                |79.74               |0.00
|Symbol              |Comment                                  |Extarnal id
|V                   |                                         |

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|4                   |4                   |2016.12.09 23:59:00 |1481327940000       |DEAL_TYPE_BUY       |DEAL_ENTRY_OUT      |0                   |DEAL_REASON_CLIENT  |2
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |79.13               |0.00                |0.00                |0.18                |V                   |end of test                              |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|4                   |2016.12.09 23:59:00 |ORDER_TYPE_BUY      |ORDER_STATE_FILLED  |2016.12.09 23:59:00 |2016.12.09 23:59:00 |1481327940000       |1481327940000       |ORDER_FILLING_FOK
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |0                   |ORDER_REASON_CLIENT |2                   |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|1.00                |0.00                |79.13               |0.00                |0.00                |79.13               |0.00
|Symbol              |Comment                                  |Extarnal id
|V                   |end of test                              |
```

Below is yet another case on a real account and real time (not in the tester): the market Buy order with the volume of 2.0 (Buy trading order generated two deals with the volumes of 1.0 — 20087494 and 20087495), then Sell limit with the volume of 2.0 was placed to take the profit and close the position. A bit later, that Sell limit was completed in two parts (deals 20088091 and 20088145). The printout:

```
Buy 2.0, Sell limit 2.0

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|20087494            |29080489            |2018.08.10 07:23:34 |1533885814000       |DEAL_TYPE_BUY       |DEAL_ENTRY_IN       |0                   |DEAL_REASON_CLIENT  |29080489
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |14595               |-0.10               |0.00                |0.00                |RTSGZU8             |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|29080489            |2018.08.10 07:23:34 |ORDER_TYPE_BUY      |ORDER_STATE_FILLED  |2018.08.10 07:23:34 |2018.08.10 07:23:34 |1533885814000       |1533885814000       |ORDER_FILLING_RETURN
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |0                   |ORDER_REASON_CLIENT |29080489            |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|2.00                |0.00                |0                   |0                   |0                   |14588               |0
|Symbol              |Comment                                  |External id
|RTSGZU8             |                                         |13_31861873584

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|20087495            |29080489            |2018.08.10 07:23:34 |1533885814000       |DEAL_TYPE_BUY       |DEAL_ENTRY_IN       |0                   |DEAL_REASON_CLIENT  |29080489
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |14595               |-0.10               |0.00                |0.00                |RTSGZU8             |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|29080489            |2018.08.10 07:23:34 |ORDER_TYPE_BUY      |ORDER_STATE_FILLED  |2018.08.10 07:23:34 |2018.08.10 07:23:34 |1533885814000       |1533885814000       |ORDER_FILLING_RETURN
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |0                   |ORDER_REASON_CLIENT |29080489            |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|2.00                |0.00                |0                   |0                   |0                   |14588               |0
|Symbol              |Comment                                  |External id
|RTSGZU8             |                                         |13_31861873584

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|20088091            |29080662            |2018.08.10 08:03:08 |1533888188000       |DEAL_TYPE_SELL      |DEAL_ENTRY_OUT      |0                   |DEAL_REASON_CLIENT  |29080489
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |14626               |-0.10               |0.00                |0.46                |RTSGZU8             |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|29080662            |2018.08.10 07:27:19 |ORDER_TYPE_SELL_LIMIT |ORDER_STATE_FILLED  |2018.08.10 07:27:19 |2018.08.10 08:05:42 |1533886039000       |1533888342000       |ORDER_FILLING_RETURN
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |0                   |ORDER_REASON_CLIENT |29080489            |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|2.00                |0.00                |14626               |0                   |0                   |14624               |0
|Symbol              |Comment                                  |External id
|RTSGZU8             |                                         |13_31862155871

Deal:
|Ticket              |Order               |Time                |Time msc            |Type                |Entry               |Magic               |Reason              |Position ID
|20088145            |29080662            |2018.08.10 08:05:42 |1533888342000       |DEAL_TYPE_SELL      |DEAL_ENTRY_OUT      |0                   |DEAL_REASON_CLIENT  |29080489
|Volume              |Price               |Commission          |Swap                |Profit              |Symbol              |Comment                                  |External ID
|1.00                |14626               |-0.10               |0.00                |0.46                |RTSGZU8             |                                         |
Order:
|Ticket              |Time setup          |Type                |State               |Time expiration     |Time done           |Time setup msc      |Time done msc       |Type filling
|29080662            |2018.08.10 07:27:19 |ORDER_TYPE_SELL_LIMIT |ORDER_STATE_FILLED  |2018.08.10 07:27:19 |2018.08.10 08:05:42 |1533886039000       |1533888342000       |ORDER_FILLING_RETURN
|Type time           |Magic               |Reason              |Position id         |Position by id
|1970.01.01 00:00:00 |0                   |ORDER_REASON_CLIENT |29080489            |0
|Volume initial      |Volume current      |Open price          |sl                  |tp                  |Price current       |Price stoplimit
|2.00                |0.00                |14626               |0                   |0                   |14624               |0
|Symbol              |Comment                                  |External id
|RTSGZU8             |                                         |13_31862155871
```

**Tips on testing 1.xxx**

- Try to keep securities of approximately the same value in the text file.
- Leaving a small number of symbols is preferable when testing in the text file. The most perfect case is to leave a single symbol and perform a test on it.

### Option 2: in combination with the "Triple choice" system

In the Option 1 (all indicators on a single chart), a trend indicator was on the same timeframe. In the Option 2, the trend indicator will be located on a higher timeframe. Thus, only one new parameter is added — trend timeframe ( **Trend timeframe**).

The Option 2 is implemented in the Elder-Ray 2.000.mq5 EA.

Files attached to the article:

| Name | File type | Description |
| --- | --- | --- |
| Symbols on the specified path.mq5 | Script | Forms a text file with the symbols of the group and is saved in the Common Data Folder |
| Gets minimal volume.mq5 | Script | Displays statistics on the minimum volume of the group. |
| Elder-Ray 1.001.mq5 | Expert Advisor | Shows visualization of applied indicators |
| Elder-Ray 1.010.mq5 | Expert Advisor | Start working with the text file and create indicators for symbols from the file. The EA is used to monitor memory consumption |
| Elder-Ray 1.020.mq5 | Expert Advisor | Define a trend. Check the trend definition validity |
| Elder-Ray 1.030.mq5 | Expert Advisor | Run-time version for Option 1: all indicators on a single chart |
| Elder-Ray 2.000.mq5 | Expert Advisor | Option 2: in combination with the "Triple choice" system |

### Conclusion

Elder-Ray (Bulls Power and Bears Power) trading system is viable enough, especially in combination with the "Triple choice" system when EMA trend indicator is calculated on a higher timeframe than Bulls Power and Bears Power.

When testing the EA, we should keep in mind that the prepared text file may contain up to 100 symbols. Considering this number, you may need up to 10 minutes to launch the test, as well as up to 5 GB of memory during the test.

Regardless of your trust to the EA, you will surely want to intervene the process, like I did while writing the article and testing the EA versions:

![CAT and V manual](https://c.mql5.com/2/33/CAT_and_V_manual.png)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5014](https://www.mql5.com/ru/articles/5014)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5014.zip "Download all attachments in the single ZIP archive")

[Elder-Ray.zip](https://www.mql5.com/en/articles/download/5014/elder-ray.zip "Download Elder-Ray.zip")(45.08 KB)

[Gets\_minimal\_volume.mq5](https://www.mql5.com/en/articles/download/5014/gets_minimal_volume.mq5 "Download Gets_minimal_volume.mq5")(5.87 KB)

[Symbols\_on\_the\_specified\_path.mq5](https://www.mql5.com/en/articles/download/5014/symbols_on_the_specified_path.mq5 "Download Symbols_on_the_specified_path.mq5")(5.73 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717)
- [Gap - a profitable strategy or 50/50?](https://www.mql5.com/en/articles/5220)
- [Improving Panels: Adding transparency, changing background color and inheriting from CAppDialog/CWndClient](https://www.mql5.com/en/articles/4575)
- [How to create a graphical panel of any complexity level](https://www.mql5.com/en/articles/4503)
- [Comparing speeds of self-caching indicators](https://www.mql5.com/en/articles/4388)
- [LifeHack for traders: Blending ForEach with defines (#define)](https://www.mql5.com/en/articles/4332)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/282198)**
(8)


![Evgeniy Scherbina](https://c.mql5.com/avatar/2014/4/53426E3A-A025.jpg)

**[Evgeniy Scherbina](https://www.mql5.com/en/users/nume)**
\|
29 Dec 2018 at 11:52

One place talks about futures, the other about a multi-currency strategy. Figures are mentioned, but there are none. I see one figure here in the discussion, but the publication itself is blank. And the most sensible suggestion is a quote from someone else's book:

_"As a rule, the [strength of bulls](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/fi "MetaTrader 5 Help: Force Index Indicator") is above zero, if it is below zero, it means there is panic in the herd of bulls and they are sinking."_

There are a lot of publications about metatrader's capabilities. The programme is excellent, there is no doubt, probably the best. I would like to meet a publication of such an author who tried to get to the bottom of the processes and was so meticulous that he did not allow not only semantic but also stylistic errors in his publication.

You took the writing of the publication seriously, or do you not believe that anyone reads it?

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
3 Jan 2019 at 13:08

The pictures have been restored.


![kasinath](https://c.mql5.com/avatar/avatar_na2.png)

**[kasinath](https://www.mql5.com/en/users/kasinath)**
\|
14 May 2020 at 07:37

Wow. This is a fantastic write up, [@Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn). Thank you for taking the time to write so much detail.

I just learned about Elder-Ray and now I am going to study this article and code very deeply. I am considering combining it with other indicators that might boost probability.

Do you have any suggestions on what indicators might be good to combine with the Elder Ray? I was thinking about volume indicators and maybe IKH. What do you think?

![Aleksey Masterov](https://c.mql5.com/avatar/2015/11/5649AE14-724E.jpg)

**[Aleksey Masterov](https://www.mql5.com/en/users/reinhard)**
\|
4 Nov 2020 at 05:35

**Vasiliy Sokolov:**

Does it eventually pour at the speed of the spread or not?

Beautiful question, but no answer.... I think the article is raw... where are the reports? Author!!!

![Yordan Lechev](https://c.mql5.com/avatar/2020/5/5EC6F682-3925.jpg)

**[Yordan Lechev](https://www.mql5.com/en/users/jordan.l)**
\|
31 Dec 2021 at 12:18

Well done ... I like the idea very much, I will test and think what can be optimised .


![Automated Optimization of an EA for MetaTrader 5](https://c.mql5.com/2/33/process-accept-icon.png)[Automated Optimization of an EA for MetaTrader 5](https://www.mql5.com/en/articles/4917)

This article describes the implementation of a self-optimization mechanism under MetaTrader 5.

![50,000 completed orders in the MQL5.com Freelance service](https://c.mql5.com/2/34/freelance-icon.png)[50,000 completed orders in the MQL5.com Freelance service](https://www.mql5.com/en/articles/5226)

Members of the official MetaTrader Freelance service have completed more than 50,000 orders as at October 2018. This is the world's largest Freelance site for MQL programmers: more than a thousand developers, dozens of new orders daily and 7 languages localization.

![MQL5 Cookbook: Getting properties of an open hedge position](https://c.mql5.com/2/34/position.png)[MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)

MetaTrader 5 is a multi-asset platform. Moreover, it supports different position management systems. Such opportunities provide significantly expanded options for the implementation and formalization of trading ideas. In this article, we discuss methods of handling and accounting of position properties in the hedging mode. The article features a derived class, as well as examples showing how to get and process the properties of a hedge position.

![Combining trend and flat strategies](https://c.mql5.com/2/33/Trend_Flat__1.png)[Combining trend and flat strategies](https://www.mql5.com/en/articles/5022)

There are numerous trading strategies out there. Some of them look for a trend, while others define ranges of price fluctuations to trade within them. Is it possible to combine these two approaches to increase profitability?

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/5014&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082978775940993665)

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