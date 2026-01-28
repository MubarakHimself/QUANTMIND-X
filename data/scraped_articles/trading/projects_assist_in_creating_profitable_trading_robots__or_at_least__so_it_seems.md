---
title: Projects assist in creating profitable trading robots! Or at least, so it seems
url: https://www.mql5.com/en/articles/7863
categories: Trading, Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:27:53.228503
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/7863&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062487469791814408)

MetaTrader 5 / Trading


Creation of a trading robot always starts with a small file, which then grows in size as you implement more additional functions and custom objects. Most of MQL5 programmers utilize [include files](https://www.mql5.com/en/docs/basis/preprosessor/include) (MQH) to handle this problem. However, there is a better solution: start developing any trading application in a project. There are so many reasons to do so.

### Project benefits

[A project](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects "https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects") is a separate file with the MQPROJ extension, which stores program settings, compilation parameters and information about all files used in the project. A separate tab in the Navigator is provided for a convenient work with the project. All files, such as include, resource, header and other files are arranged into categories in this tab.

![Project example](https://c.mql5.com/2/39/project_sample__2.png)

You see, the project is not just a set of files and folders arranged under a separate directory. It allows breaking down a complex program into elements arranged into a well-balanced structure. All the required information is at hand:

- set files with input parameters for testing and optimization
- source codes of [OpenCL programs](https://www.metatrader5.com/en/metaeditor/help/development/opencl "https://www.metatrader5.com/en/metaeditor/help/development/opencl")
- image and sound media files

- [resources](https://www.mql5.com/en/docs/runtime/resources) and other data

All connections between program parts are clearly visible in the project, so you can easily navigate between all used files. Furthermore, multiple programmers can collaborate on a project via the built-in [MQL5 Storage](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#shared "https://www.metatrader5.com/en/metaeditor/help/mql5storage/mql5storage_working").

### Creating a Project

A new project is created using the MQL5 Wizard as an ordinary MQL5 program. Click "New project" and go through the required steps: set the program name, add input parameters and specify utilized [event handlers](https://www.mql5.com/en/docs/event_handlers). Upon the completion of the MQL5 Wizard, an MQPROJ file will be opened. This file allows managing project properties.

![Project Properties](https://c.mql5.com/2/39/project_window__4.png)\>

Here you can specify the version, set a program description, add an icon and manage additional options:

1. Maximum optimization — optimization of the EX5 executable file for maximum performance. If the option is disabled, compilation of the source code can be completed faster, but the resulting EX5 file can run much slower.

2. Check floating point dividers — check whether real numbers of double and float types are not equal to zero in division operations. The operation speed can be higher if the option is disabled. However, you should be totally confident in your code.
3. Use tester optimization cache — the tester is enabled by default, and thus the tester saves all results of completed passes to the [optimization cache](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization#cache "https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization#cache"). The data can further be used in re-calculations. The cache can be disabled using the[tester\_no\_cache](https://www.mql5.com/en/docs/basis/preprosessor/compilation) property. Optionally, you can uncheck the relevant option in the project.

If the project file is closed, it can be reopened using the appropriate command of the Properties context menu. For a deeper understanding of the MQPROJ file contents, you can open it in the text format using the Open command. Thus, you can view the internal structure of projects.

```
{
  "platform"    :"mt5",
  "program_type":"expert",
  "copyright"   :"Copyright 2019, MetaQuotes Software Corp.",
  "link"        :"https:\/\/www.mql5.com",
  "version"     :"1.00",
  "description" :"The mean reversion strategy: the price breaks the channel border outwards and reverts back towards the average. The channel is represented by Bollinger Bands. The Expert Advisor enters the market using limit orders, which can only be opened in the trend direction.",
  "icon"        :"Mean Reversion.ico",
  "optimize"    :"1",
  "fpzerocheck" :"1",
  "tester_no_cache":"0",
  "tester_everytick_calculate":"0",

  "files":
  [\
    {\
      "path":".\\Mean Reversion.mq5",\
      "compile":"true",\
      "relative_to_project":"true"\
    },\
    {\
      "path":"MQL5\\Include\\Trade\\Trade.mqh",\
      "compile":"false",\
      "relative_to_project":"false"\
    },\
....\
```\
\
### Trading rules\
\
Let us apply classical rule: enter the market when the price touches a [Bollinger Band](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/bb "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/bb"). This is one of the trading strategies, expecting the price to return to its mean value.\
\
![Market entry based on Bollinger Bands](https://c.mql5.com/2/39/Trade_Signal.png)\
\
Only limit orders will be used for market entries. The additional rule will be as follows: trade only in the trend direction. Thus, a Buy Limit will be placed at the lower channel border in an uptrend. During a downtrend, a Sell Limit will be placed at the upper border.\
\
There are many ways to determine trend direction. Let's use the simplest one: the relative position of two [moving averages](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma"). If the Fast EMA is above the Slow EMA, an uptrend is defined. The downtrend is recognized when the lines are arranged the opposite way.\
\
![Determining trend using two moving averages](https://c.mql5.com/2/39/Trend_Filter.png)\
\
This simple rule has one disadvantage: there will always be either an uptrend or a downtrend. Therefore, such a system can produce a lot of false entries during flat. To avoid this, add another rule which allows placing pending orders when the distance between the Bollinger bands is large enough. The optimal way is to measure the channel width in relative values rather than points. The width can be determined based on the [ATR indicator](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr") that measures volatility in points.\
\
- Channel width below k\*ATR indicates a flat, and thus placing orders is not allowed.\
- If the channel width is greater than k\*ATR, place a pending limit order at the channel border, in trend direction.\
\
\
Here k is a certain coefficient which needs to be found.\
\
![Calculating the Bollinger channel width with the ATR](https://c.mql5.com/2/39/2020-06-09_17h48_47.png)\
\
Thus, during project creation we need to specify 8 input parameters for determining trading signals. The Expert Advisor will trade a fixed lot which should be specified in the InpLot parameter. Another non-optimizable parameter is InpMagicNumber. By using it, we can instruct the EA to handle only its own orders and positions.\
\
```\
//--- Channel parameters\
input int             InpBBPeriod   =20;           // Bollinger indicator period\
input double          InpBBDeviation=2.0;          // Deviation of Bollinger bands from the MA\
//-- EMA periods for trend calculation\
input int             InpFastEMA    =12;           // Fast EMA period\
input int             InpSlowEMA    =26;           // Slow EMA period\
//-- ATR parameters\
input int             InpATRPeriod  =14;           // ATR period\
input double          InpATRCoeff   =1.0;          // ATR coefficient for determining the flat\
//--- Capital managements\
input double          InpLot        =0.1;          // Trading volume in lots\
//--- timeframe parameters\
input ENUM_TIMEFRAMES InpBBTF       =PERIOD_M15;   // the timeframe for Bollinger values calculation\
input ENUM_TIMEFRAMES InpMATF       =PERIOD_M15;   // the timeframe for trend determining\
//--- Expert Advisor identifier for trading transactions\
input long            InpMagicNumber=245600;       // Magic Number\
```\
\
The InpBBTF and InpMATF parameters have been added to avoid the manual selection of timeframes for determining the trend and the channel width. In this case, optimal timeframe values can be found during optimization. The EA can run on the M1 timeframe, while using Bollinger Bands data from M15 and Moving Averages from M30. No input parameter is used for ATR, otherwise there would be to many parameters for this example.\
\
### Writing functions\
\
After creating a project, we can proceed to developing the Expert Advisor. The below code shows the main three functions describing the rules.\
\
The calculation of the Bollinger channel width is simple: copy the values from the indicator buffers.\
\
```\
//+------------------------------------------------------------------+\
//| Gets the values of the channel borders                           |\
//+------------------------------------------------------------------+\
bool ChannelBoundsCalculate(double &up, double &low)\
  {\
//--- get the Bollinger Bands indicator values\
   double bbup_buffer[];\
   double bblow_buffer[];\
   if(CopyBuffer(ExtBBHandle, 1, 1, 1, bbup_buffer)==-1)\
     {\
      PrintFormat("%s: Failed CopyBuffer(ExtBBHandle,0,1,2,bbup_buffer), code=%d", __FILE__, GetLastError());\
      return(false);\
     }\
\
   if((CopyBuffer(ExtBBHandle, 2, 1, 1, bblow_buffer)==-1))\
     {\
      PrintFormat("%s: Failed CopyBuffer(ExtBBHandle,0,1,2,bblow_buffer), code=%d", __FILE__, GetLastError());\
      return(false);\
     }\
   low=bblow_buffer[0];\
   up =bbup_buffer[0];\
//--- successful\
   return(true);\
  }\
```\
\
Flat determining method is also simple enough. First, get the values of the channel borders, then calculate the width and compare to the ATR value multiplied by the InpATRCoeff coefficient.\
\
```\
//+------------------------------------------------------------------+\
//|  Returns true if the channel is too narrow (indication of flat)  |\
//+------------------------------------------------------------------+\
int IsRange()\
  {\
//--- get the ATR value on the last completed bar\
   double atr_buffer[];\
   if(CopyBuffer(ExtATRHandle, 0, 1, 1, atr_buffer)==-1)\
     {\
      PrintFormat("%s: Failed CopyBuffer(ExtATRHandle,0,1,2,atr_buffer), code=%d", __FILE__, GetLastError());\
      return(NO_VALUE);\
     }\
   double atr=atr_buffer[0];\
//--- get the channel borders\
   if(!ChannelBoundsCalculate(ExtUpChannel, ExtLowChannel))\
      return(NO_VALUE);\
   ExtChannelRange=ExtUpChannel-ExtLowChannel;\
//--- if the channel width is less than ATR*coefficients, this is a flat\
   if(ExtChannelRange<InpATRCoeff*atr)\
      return(true);\
//--- flat not detected\
   return(false);\
  }\
```\
\
As can be seen from the code, the NO\_VALUE macro code is returned sometimes, which means that calculation of a certain parameter failed.\
\
```\
#define NO_VALUE      INT_MAX                      // invalid value when calculating Signal or Trend\
```\
\
Trend determining function has the longest code.\
\
```\
//+------------------------------------------------------------------+\
//| Returns 1 for UpTrend or -1 for DownTrend (0 = no trend)         |\
//+------------------------------------------------------------------+\
int TrendCalculate()\
  {\
//--- first, check the flat\
   int is_range=IsRange();\
//--- check the result\
   if(is_range==NO_VALUE)\
     {\
      //--- if the check failed, early termination with "no value" response\
      return(NO_VALUE);\
     }\
//--- do not determine direction during flat\
   if(is_range==true) // narrow range, return "flat"\
      return(0);\
//--- get the ATR value on the last completed bar\
   double atr_buffer[];\
   if(CopyBuffer(ExtBBHandle, 0, 1, 1, atr_buffer)==-1)\
     {\
      PrintFormat("%s: Failed CopyBuffer(ExtATRHandle,0,1,2,atr_buffer), code=%d", __FILE__, GetLastError());\
      return(NO_VALUE);\
     }\
//--- get the Fast EMA value on the last completed bar\
   double fastma_buffer[];\
   if(CopyBuffer(ExtFastMAHandle, 0, 1, 1, fastma_buffer)==-1)\
     {\
      PrintFormat("%s: Failed CopyBuffer(ExtFastMAHandle,0,1,2,fastma_buffer), code=%d", __FILE__, GetLastError());\
      return(NO_VALUE);\
     }\
//--- get the Slow EMA value on the last completed bar\
   double slowma_buffer[];\
   if(CopyBuffer(ExtSlowMAHandle, 0, 1, 1, slowma_buffer)==-1)\
     {\
      PrintFormat("%s: Failed CopyBuffer(ExtSlowMAHandle,0,1,2,slowma_buffer), code=%d", __FILE__, GetLastError());\
      return(NO_VALUE);\
     }\
//--- trend is not defined by default\
   int trend=0;\
//--- if fast EMA is above the slow one\
   if(fastma_buffer[0]>slowma_buffer[0])\
      trend=1;   // uptrend\
//--- if fast EMA is below the slow one\
   if(fastma_buffer[0]<slowma_buffer[0])\
      trend=-1;  // downtrend\
//--- return trend direction\
   return(trend);\
  }\
```\
\
The last function of the trading algorithm is the determining of a new bar. According to the logic, the trend is determined when a new bar appears.\
\
```\
//+------------------------------------------------------------------+\
//| Checks the emergence of a new bar on the current timeframe,      |\
//| also calculates the trend and the signal                         |\
//+------------------------------------------------------------------+\
bool IsNewBar(int &trend)\
  {\
//--- permanently stores the current bar opening time between function calls\
   static datetime timeopen=0;\
//--- get the current bar open time\
   datetime time=iTime(NULL, InpMATF, 0);\
//--- if the time has not changed, the bar is not new, so exit with the 'false' value\
   if(time==timeopen)\
      return(false);\
//--- the bar is new, and this trend direction should be calculated\
   trend=TrendCalculate();\
//--- if trend direction could not be obtained, exit and try again during the next call\
   if(trend==NO_VALUE)\
      return(false);\
//--- all checks performed successfully: the bar is new and trend direction has been obtained\
   timeopen=time; //remember current time open time for further calls.\
//---\
   return(true);\
  }\
```\
\
The above logic allows organizing the EA operation so that all trading operations are performed only once during the entire bar. Therefore, testing results do not depend on the tick generation mode.\
\
The entire trading algorithm is presented in the OnTick() handler:\
\
- The emergence of a new bar and the trend direction are determined first.\
- If there is no trend or a position is open, an attempt is performed to delete pending orders and to exit from the handler.\
\
- If there is a directional trend and no pending order, an attempt is performed to place a limit order at the channel border.\
\
- If a pending order already exists and it has not been modified on the new bar, an attempt is performed to move it to the current channel border.\
\
\
```\
//+------------------------------------------------------------------+\
//| Expert tick function                                             |\
//+------------------------------------------------------------------+\
void OnTick()\
  {\
   static bool order_sent    =false;    // failed to place a limit order on the current bar\
   static bool order_deleted =false;    // failed to delete a limit order on the current bar\
   static bool order_modified=false;    // failed to modify a limit order on the current bar\
//--- if input parameters are invalid, stop testing at the first tick\
   if(!ExtInputsValidated)\
      TesterStop();\
//--- check the emergence of a new bar and the trend direction\
   if(IsNewBar(ExtTrend))\
     {\
      //--- reset values of static variables to their original state\
      order_sent    =false;\
      order_deleted =false;\
      order_modified=false;\
     }\
//--- create auxiliary variables to make check calls only once, on the current bar\
   bool order_exist   =OrderExist();\
   bool trend_detected=TrendDetected(ExtTrend);\
//--- if there is no trend or there is an open position, delete pending orders\
   if(!trend_detected || PositionExist())\
      if(!order_deleted)\
        {\
         order_deleted=DeleteLimitOrders();\
         //--- if the orders have been successfully deleted, no other operations are needed at this bar\
         if(order_deleted)\
           {\
            //--- prohibit placing and modification of orders\
            order_sent    =true;\
            order_modified=true;\
            return;\
           }\
        }\
\
//--- there is trend\
   if(trend_detected)\
     {\
      //--- place an order at the channel border if no order is found\
      if(!order_exist && !order_sent)\
        {\
         order_sent=SendLimitOrder(ExtTrend);\
         if(order_sent)\
            order_modified=true;\
        }\
      //--- try to move the order to the channel border if it has not been moved on the current bar\
      if(order_exist && !order_modified)\
         order_modified=ModifyLimitOrder(ExtTrend);\
     }\
//---\
  }\
```\
\
Other trading functions of the Expert Advisor are standard. The source codes are available in the MetaTrader 5 terminal standard package. They are located under MQL5\\Experts\\Examples.\
\
![MeanReversion project location in the Navigator](https://c.mql5.com/2/39/MeanReversion_Project.png)\
\
### Optimizing parameters and adding set files\
\
Now that the EA is ready, let's find optimal parameters in the strategy tester. Did you know that the tester provides options for easy copying of values from the "Settings" and "Inputs" tabs into the clipboard using the **Ctr+C** combination? Thus, you can provide your settings to another person, for example to a Customer via the Freelance chat, without having to save them to a [set file](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization#inputs "https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization#inputs"). The customer can copy the data to the clipboard and paste into the Settings tab of the tester using **Ctr+V**.\
\
Saving to a set file is also a convenient solution. Many sellers in the Market provide such files, so that product buyers can instantly load the appropriate sets of parameters and test or optimize the EA on a required instrument. Separate set files need to be created for each of the traded instruments. The number of such files on the computer can be quite large, if many Expert Advisors exist in the platform. With projects, your customers can instantly access the required files without the need to search for them on the disk each time the symbol is changed.\
\
Here is an example of how projects can help to add appropriate parameter sets straight in the EA's EX5 file. Select the symbol for which the optimization will be performed. For example, EURUSD. Set Start, Step and Stop for the parameters you want to optimize and launch the optimization process. Once it is over, double click on the best pass in the Optimizations tab, and the values of input parameters from this pass will be inserted in the Parameters tab, as well as a single test will be run. The found parameters can be saved to a set file. However, there is no need to provide it separately. Save the set of parameters under a clear name, such as EURUSD.set. It means that the parameters should be applied for this pair and not GBPJPY.\
\
![Saving inputs to a set file](https://c.mql5.com/2/39/Save_set_file.png)\
\
Repeat this operation for each symbol which your EA can trade. Thus, you have a number of ready set files, say 9. Add these files to your project. Create the appropriate folder "Settings and files\\Set", to separate them from source files. With Projects, you can maintain order and the correct file structure.\
\
![Adding set files to a project](https://c.mql5.com/2/39/Add_set_files__2.png)\
\
Now, compile the project and open the strategy tester with the MeanReversion EA. A new item "Load from EA" will appear in the context menu, on the Inputs tab. All available set files can be accessed from this menu.\
\
![Loading input parameters from the EA](https://c.mql5.com/2/39/load_set_from_EA.png)\
\
Thus, the compiled EX5 file of the Expert Advisor is a fully completed product, with ready sets of parameters. The strategy can be instantly tested without having to set borders and steps for each of the desired symbols. Users and buyers of your trading robots will definitely appreciate this convenience.\
\
### Strategy running on real data\
\
In September 2019, the MeanReversion Expert Advisor was launched on a demo account. The purpose was to find out programming and trading errors in real time. The EA was launched in a portfolio mode on multiple symbols (this was the initial idea during the optimization). A built-in VPS was rented for the EA, based on which a private signal [Many MeanReversion Optimized](https://www.mql5.com/en/signals/633444) was created for monitoring purposes.\
\
![Trade Results for 9 months](https://c.mql5.com/2/39/Monitoring__EN.png)\
\
The first month after the launch the EA showed positive results. This was followed by consecutive 5 losing months. The virtual hosting was rented with the automated renewal feature, and thus the EA was running in a fully autonomous mode. It kept trading towards a complete deposit loss. Then, in March, something changed in the forex market and the EA suddenly generated a record profit. During the next 2 months, the results were contradictory. The same growth can probably never be repeated again.\
\
The analysis of deals and results by symbols shows that loss was made by three yen pairs and AUDUSD. The Expert Adviser did not show impressive results. Nevertheless, even with such simple trading logic, it has been running for 9 months in a portfolio mode, due to which losses on some symbols are covered by profit on other pairs.\
\
![Distribution by symbols](https://c.mql5.com/2/39/Distribution_by_Symbols_EN.png)\
\
The Expert Advisor parameters have never been modified since its launch, no additional migrations were performed during this time. The EA was compiled 9 months ago and was launched on eight charts, on a built-in VPS. It is still running without any human interference. We cannot even remember why only eight out of nine set files were launched. Moreover, we cannot remember the parameters used. Nevertheless, the MeanReversion Expert Advisor project created for educational purposes is still running and is showing profit as of June 10, 2020.\
\
### Switch to Projects and enjoy the benefits\
\
Projects allow developers to create programs of any complexity level, as well as to collaborate during development. When working together with like-minded people, you can develop applications faster, exchange useful ideas and skills, as well as improve the quality of the code.\
\
The trading rules utilized within this Expert Advisor are very simple, but it can be used as a template for creating many other trading robots. Replace functions determining the trend direction, the flat state, or the entry levels and methods (for example, you may use market orders instead of limit ones). Perhaps, better results might be obtained if trading only during flat periods. Also, the EA lacks trailing stop, Stop Loss and TakeProfit settings. There is much more you can do to improve the EA.\
\
Using the MeanReversion EA from the MetaTrader 5 standard package, you can study and evaluate the advantages of projects. Create your own project or copy this one into a new folder and start experimenting. Start using Projects and evaluate the convenience for yourself!\
\
Translated from Russian by MetaQuotes Ltd.\
\
Original article: [https://www.mql5.com/ru/articles/7863](https://www.mql5.com/ru/articles/7863)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
#### Other articles by this author\
\
- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)\
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)\
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)\
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)\
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)\
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)\
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/343858)**\
(33)\
\
\
![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)\
\
**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**\
\|\
19 May 2021 at 10:49\
\
I am creating a project (Expert Advisor). What is the correct way to add a [custom indicator](https://www.mql5.com/en/articles/5 "Article: Switching to new rails: Custom indicators in MQL5") to the project? (I have the indicator on my PC).\
\
\
![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)\
\
**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**\
\|\
19 May 2021 at 13:17\
\
One project - one type of programme. Only this main programme will be compiled. It is not by chance that the type of programme is specified when [creating a project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not for sure ").\
\
It is necessary to address some indicator **from the** project **code** by the full path. Even if you put the source code of the indicator in the project folder, you will not be able to work with it.\
\
Or I don't understand - you need to specify what the question is.\
\
![CHIEN CHEN WU](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[CHIEN CHEN WU](https://www.mql5.com/en/users/qpt4cjqmzx-privaterelay.appleid)**\
\|\
29 Oct 2021 at 12:18\
\
Use the data algorithm to buy and sell crossovers along the rise! Input conventions towards adjustments\
\
\
![Tobias Johannes Zimmer](https://c.mql5.com/avatar/2022/3/6233327A-D1E7.JPG)\
\
**[Tobias Johannes Zimmer](https://www.mql5.com/en/users/pennyhunter)**\
\|\
17 Jan 2022 at 01:55\
\
Will there be a detailed article about shared [projects](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") too? It has so much potential but most people don't know it exists.\
\
There is a section on Metatrader help but I couldn't find the screen to invite friends to my projects.\
\
Then there is a lot of misunderstanding about what it can and can't do and some features just don't seem to work.\
\
![Sergey Gridnev](https://c.mql5.com/avatar/2014/5/53726F63-E57D.jpg)\
\
**[Sergey Gridnev](https://www.mql5.com/en/users/contender)**\
\|\
9 Oct 2022 at 07:37\
\
Проекты позволяют создавать программы любого уровня сложности и вести совместные разработки.\
\
The rule "one [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not for sure ") \- one type of programme" (one ex5 file) puts an end to "programmes of any level of complexity".\
\
![Forecasting Time Series (Part 2): Least-Square Support-Vector Machine (LS-SVM)](https://c.mql5.com/2/38/mql5-avatar-lssvm.png)[Forecasting Time Series (Part 2): Least-Square Support-Vector Machine (LS-SVM)](https://www.mql5.com/en/articles/7603)\
\
This article deals with the theory and practical application of the algorithm for forecasting time series, based on support-vector method. It also proposes its implementation in MQL and provides test indicators and Expert Advisors. This technology has not been implemented in MQL yet. But first, we have to get to know math for it.\
\
![Continuous Walk-Forward Optimization (Part 4): Optimization Manager (Auto Optimizer)](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization.png)[Continuous Walk-Forward Optimization (Part 4): Optimization Manager (Auto Optimizer)](https://www.mql5.com/en/articles/7538)\
\
The main purpose of the article is to describe the mechanism of working with our application and its capabilities. Thus the article can be treated as an instruction on how to use the application. It covers all possible pitfalls and specifics of the application usage.\
\
![Continuous Walk-Forward Optimization (Part 5): Auto Optimizer project overview and creation of a GUI](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__1.png)[Continuous Walk-Forward Optimization (Part 5): Auto Optimizer project overview and creation of a GUI](https://www.mql5.com/en/articles/7583)\
\
This article provides further description of the walk-forward optimization in the MetaTrader 5 terminal. In previous articles, we considered methods for generating and filtering the optimization report and started analyzing the internal structure of the application responsible for the optimization process. The Auto Optimizer is implemented as a C# application and it has its own graphical interface. The fifth article is devoted to the creation of this graphical interface.\
\
![Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__1.png)[Timeseries in DoEasy library (part 36): Object of timeseries for all used symbol periods](https://www.mql5.com/en/articles/7627)\
\
In this article, we will consider combining the lists of bar objects for each used symbol period into a single symbol timeseries object. Thus, each symbol will have an object storing the lists of all used symbol timeseries periods.\
\
[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/7863&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062487469791814408)\
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