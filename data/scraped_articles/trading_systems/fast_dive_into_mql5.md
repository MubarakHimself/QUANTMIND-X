---
title: Fast Dive into MQL5
url: https://www.mql5.com/en/articles/447
categories: Trading Systems, Indicators
relevance_score: 3
scraped_at: 2026-01-23T21:00:48.632635
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ajdejkisxlupqpyybmpdeeivmqdaixyy&ssn=1769191247823228694&ssn_dr=0&ssn_sr=0&fv_date=1769191247&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F447&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Fast%20Dive%20into%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919124765870541&fz_uniq=5071505092932217167&sv=2552)

MetaTrader 5 / Examples


### Why Do You Need MQL5?

There may be many reasons why you have decided to study the modern MQL5 trading strategies' programming language, and we welcome your decision! Experienced users can easily navigate in the language documentation, as well as in the variety of articles and services presented here. But, if you have just discovered the MetaTrader 5 client terminal, many things may seem to be unusual and confusing at first.

So, what benefits are there to knowing MQL5? Perhaps, you have decided to study a present-day OOP ( [object-oriented programming](https://www.mql5.com/en/docs/basis/oop)) language. After studying MQL5, you can easily master other high-level languages, such as C++, С#, Java and so on. Of course, that does not mean that they are very similar, but their basics have much in common.

Or maybe, you already know one of these languages and have an idea to make your own trading robot or information-analytical system for working on financial markets. In this case, you will easily master the specialized MQL5 language created specifically for that purpose.

Or perhaps, you already know [MQL4](https://book.mql4.com/ "https://book.mql4.com/") actively used for making various trading robots and indicators for the popular MetaTrader 4 trading terminal? Then, you just have to make a slight effort to see the whole power of the new MQL5 language and all the benefits of the new [MetaEditor 5 development environment](https://www.metatrader5.com/en/metaeditor/help "https://www.metatrader5.com/en/metaeditor/help").

There may be plenty of reasons to study MQL5, and we want to give you some tips on where to start and what to pay attention to. So, let us begin.

![Fast Dive into MQL5](https://c.mql5.com/2/4/fast_MQL5__1.jpg)

### The Language Possibilities and Features

[MetaQuotes Language 5 (MQL5)](https://www.mql5.com/en/docs) is developed by MetaQuotes Software Corp. based on their long experience in the creation of online trading platforms of several generations. The following are the language's main advantages.

- Syntax is as close to C++ as possible. That allows MQL5 to easily adopt applications written in other languages.

- The operating speed of MQL5 programs is almost as high as that of С++ programs.
- Rich, built-in features for creating technical indicators, graphic objects and user interfaces.
- Built-in [OpenCL support](https://www.mql5.com/en/docs/opencl).
- A big standard library and a lot of examples in the C [ode Base](https://www.mql5.com/en/code).
- Paralleling mathematical optimization tasks for dozens and thousands individual threads without having to write a special code.


**For Newcomers in Programming**

If you do not have experience programming in high-level languages, you can take any C++ manual and use it as an example when studying [MQL5 language basics](https://www.mql5.com/en/docs/basis) (syntax, data types, variables, operators, functions, OOP and so on). MQL5 developers sought to ensure the maximum compatibility of its features with the highly popular C++ language.

Experience shows that it is possible to learn MQL5 from scratch within a couple of months, while less than a year may be enough for some users to study and reveal all its features. See [Limitless Opportunities with MetaTrader 5 and MQL5](https://www.mql5.com/en/articles/392), and perhaps it will inspire you to create something really great.

**For MQL4 Users**

At first, you may find the new [approach to creating indicators](https://www.mql5.com/en/articles/10) uncomfortable. Lots of new [event handling functions](https://www.mql5.com/en/docs/basis/function/events) may astonish you, while C-like syntax and the new data types may initially seem to be unusual.

But, after a little while, you will appreciate all the advantages of MQL5 over MQL4. And, how do you like [the rich possibilities for working with charts and graphic objects](https://www.mql5.com/en/articles/62) and the ability to draw any image like on the canvas? You will be able to try all that after getting acquainted with MQL5.

**For Professional Programmers**

If you code in any of the present-day languages, it will be easy enough for you to quickly master MQL5. You already know ООP and the event model. Now, you only have to learn the following specific functions deliberately meant for algorithmic trading.

- [Chart Operations](https://www.mql5.com/en/docs/chart_operations).
- [Trade Functions](https://www.mql5.com/en/docs/trading).
- [Getting Market Information](https://www.mql5.com/en/docs/marketinformation).
- [Custom Indicators](https://www.mql5.com/en/docs/customind).
- [Object Functions](https://www.mql5.com/en/docs/objects).
- and others.


There are also the following slight differences in the language syntax implemented for reasons of safe code writing and optimal application operating time.

- No pointer arithmetic. [MQL5 pointers](https://www.mql5.com/en/docs/basis/types/object_pointers) are, in fact, descriptors.
- No exceptions.
- Arrays of any type are always [passed by reference](https://www.mql5.com/en/docs/basis/variables/formal).
- [Arrays](https://www.mql5.com/en/docs/basis/variables#array_define) cannot have more than four dimensions.
- Arrays and objects cannot be returned from the functions, but it is possible to return an object pointer.
- No goto operator.


If you have not performed trading operations before, you may have some questions concerning trading terms and the Strategy Tester when writing your trading robot. The [articles](https://www.mql5.com/en/articles) section contains useful publications to help you with that, such as the following publications.

- [The Fundamentals of Testing in MetaTrader 5](https://www.mql5.com/en/articles/239)
- [Orders, Positions and Deals in MetaTrader 5](https://www.mql5.com/en/articles/211)
- [Trade Events in MetaTrader 5](https://www.mql5.com/en/articles/232)
- [Speed Up Calculations with the MQL5 Cloud Network](https://www.mql5.com/en/articles/341)

Therefore, MQL5 language is not a problem for a professional programmer. The main issue is acquaintance with trading and related concepts.

### MetaTrader 5 Terminal Installation

MetaTrader 5 terminal web installer can be downloaded from the official website via the link [https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe](https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"). MetaTrader 5 terminal installation is easy enough. All is done in a few clicks. But, we recommend that you install the terminal on any drive but the one wherethe Windows operating system is installed. The reason is that Microsoft has implemented the new system for user actions control starting with Windows Vista, [UAC](https://en.wikipedia.org/wiki/User_Account_Control "https://en.wikipedia.org/wiki/User_Account_Control").

Thus, if you are not experienced in system administration or you do not like to navigate through many hidden folders, specify the terminal installation folder away from Program Files to allow the data terminal to be stored in the same directory as the MetaTrader 5 terminal. For example, install the terminal on drive **D:\**, if the operating system is installed on drive **C:\**.

More information about the differences in the modes of MetaTrader 5 operation that depend on the installation path can be found in [Getting started → For Advanced User → Platform Start](https://www.metatrader5.com/en/terminal/help/start_advanced/start "https://www.metatrader5.com/en/terminal/help/start_advanced/start") in the built-in user guide.

### Indicators, Scripts and Expert Advisors

MQL5 language realizes three basic program types. Each type is best suited for solving its specific tasks, such as the following.

- A script is a program designed for a single launch on a price chart. Once the execution reaches the end of the predefined [OnStart()](https://www.mql5.com/en/docs/basis/function/events#onstart) handler, the script is completed and unloaded from the chart. The OnStart() function is designed only for scripts. This is the only launch point in which the executable code must be located. A script may contain an infinite loop with short pauses between iterations and thus operate on a chart until it is forcibly stopped. Only one script can be executed at a time on each chart.

- An indicator is a program for calculating value arrays on the basis of price data. Special arrays for storing indicator values are called indicator buffers. The number of allowed buffers in one indicator is practically unlimited.

Each chart can simultaneously have multiple indicators including a few copies of the same indicator. [The functions for working with indicator properties](https://www.mql5.com/en/docs/customind) are available only from the indicators. They are not available from scripts or Expert Advisors. A program is considered to be an indicator, if the [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) handler has been detected in it. The indicator lifetime is unlimited. It is executed until it is removed from the chart. The OnCalculate() function can be called only in indicators.

- An Expert Advisor is another type of a program with unlimited lifetime. It also can be located on a chart for as much time as necessary. Expert Advisors usually have the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) event handler, which clearly indicates that we are dealing with a source code of an Expert Advisor. Only one Expert Advisor at a time can be located and, therefore, executed on one chart.

The primary goal of Expert Advisors is the automation of trading systems. But, they can also have exclusively service functions, such as the implementation of a graphical interface for manual trading, and/or the current market situation analysis and visual representation and so on.


![MetaTrader 5 indicator samples](https://c.mql5.com/2/4/Some_Indicators_MetaTrader5__1.png)

It would be better to start learning MQL5 basics from writing scripts, using examples from the MQL5 documentation or the [Code Base](https://www.mql5.com/en/code). Then, you can start working with [object functions](https://www.mql5.com/en/docs/objects) and experimenting with [trade operations](https://www.mql5.com/en/docs/trading) on a demo account.

The next stage is writing your own [custom indicators](https://www.mql5.com/en/docs/customind) and analyzing examples from the code case and [articles](https://www.mql5.com/en/articles/examples_indicators). By the time you master the indicators, you will be ready for learning [event-handling functions](https://www.mql5.com/en/docs/basis/function/events).

The final goal is the creation of simple Expert Advisors and proving them against historical data using the Strategy Tester in MetaTrader 5 terminal. There are many articles in [Experts](https://www.mql5.com/en/articles/examples_experts), [Tester](https://www.mql5.com/en/articles/strategy_tester) and [Trading Systems](https://www.mql5.com/en/articles/trading_systems) devoted to this subject.

And, of course, we should mention the most exciting feature concerning MQL5 programming, development of custom modules for Expert Advisors via the [MQL5 Wizard](https://www.mql5.com/en/articles/275). You can find many articles on this subject, and the Code Base contains a variety of ready-made [MQL5 Wizard modules](https://www.mql5.com/en/search#!keyword=MQL5%20Wizard).

### Event Model

An MQL5 program works only when some [events](https://www.mql5.com/en/docs/runtime/event_fire) occur. They may represent downloading and initializing an MQL5 program, new tick arrival (symbol price change), changing chart properties, changing a symbol or a timeframe on a chart, pending order actuation and so forth.

Therefore, the event model allows you to write interactive programs in the simplest way. There are great possibilities for writing custom graphical panels and creating convenient user interface that can suit your needs. Built-in functions for working with graphics allow creation of full-featured and well-designed applications.

![A simple example of event handling in MQL5 program](https://c.mql5.com/2/4/Event_model_EN.png)

The ability to generate custom events using the [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom) function for any active chart in MetaTrader 5 allows the creation of complex interactive systems. Event trapping and handling are performed by the [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) function. The mentioned features are shown in [The Player of Trading Based on Deal History](https://www.mql5.com/en/articles/242) and [EventChartCustom](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom)() function example.

### Debugging and User Guide

The MetaTrader 5 terminal and MetaEditor 5 contain well-documented built-in user guides that can be accessed by pressing **F1**. All documentation is updated automatically via LiveUpdate. Also, this user guide is available online in several languages on the official site of the MetaTrader 5 trading platform.

- [https://www.metatrader5.com/en/terminal/help](https://www.metatrader5.com/en/terminal/help "https://www.metatrader5.com/en/terminal/help") is the MetaTrader 5 terminal user guide.
- [https://www.metatrader5.com/en/metaeditor/help](https://www.metatrader5.com/en/metaeditor/help "https://www.metatrader5.com/en/metaeditor/help") is the MetaEditor 5 development environment user guide.

The most important learning source is MQL5 documentation, which is represented not only online at [https://www.mql5.com/en/docs](https://www.mql5.com/en/docs) but is also available for downloading in CHM and PDF formats in multiple languages.

The client terminal and MetaEditor 5 are closely integrated with each other. You can always switch to another application by pressing **F4**. This is a very convenient feature when editing code, especially if you are working with several terminals simultaneously.

Any MQL5 program can be [debugged](https://www.metatrader5.com/en/metaeditor/help/development/debug "https://www.metatrader5.com/en/metaeditor/help/development/debug") in the terminal right from the editor by pressing **F5**. The chart will be opened automatically, and your program (a script, an indicator or an Expert Advisor) will be launched on it. When debugging scripts, you should also consider that they are uploaded on their own after OnStart() operation completion. Therefore, the debugging process automatically completes at that stage, and the ["debug" chart](https://www.metatrader5.com/en/metaeditor/help/development/debug#settings "https://www.metatrader5.com/en/metaeditor/help/development/debug#settings") is closed without saving all your graphical objects made by the script. Thus, put a breakpoint or [Sleep()](https://www.mql5.com/en/docs/common/sleep) with a very large value before the return() operator at the very end of a script.

The debugging mode is necessary both for detecting errors in your program and for studying and learning MQL5. Besides using breakpoints in your code, there is a special [DebugBreak()](https://www.mql5.com/en/docs/common/debugbreak) function, which works only if a program is in the debugging mode.

And, of course, we should mention the powerful search engine integrated into MetaTrader 5, which allows searching any necessary data not only in a source file or a folder, but also on the MQL5.community website (Articles, Forum and Code Base).

![Setting search parameters in MetaEditor 5](https://c.mql5.com/2/4/search_metaEditor_Eng.png)

Obtained results can be filtered by the necessary categories. Therefore, the development environment provides not only the built-in user guide to the editor and MQL5 language but also the ability to find useful materials on the mql5.com website.

### Code Profiling

The MetaEditor 5 development environment offers programmers plenty of convenient features to simplify code writing and debugging. What else do programmers need apart from the debugging feature? [Code profiling](https://www.metatrader5.com/en/metaeditor/help/development/profiling "https://www.metatrader5.com/en/metaeditor/help/development/profiling"), of course. Profiling is gathering the application features, such as the execution time of its individual fragments (functions, lines), in a convenient form.

Profiling allows you to quickly detect the most time-consuming parts in your application. You can evaluate the implemented changes in terms of operation speed to choose the most efficient algorithms. Professional developers are well aware of what they can do using this feature, while newcomers can examine their programs in a new light.

![Code profiling in MetaTrader 5](https://c.mql5.com/2/4/Profiler_in_MetaTrader5_EN.png)

The screenshot above shows profiling of the code shown at the forum ( [https://www.mql5.com/en/forum/7134](https://www.mql5.com/en/forum/7144)). Try code profiling by downloading the code from the mentioned forum thread.

### MQL5 Storage: Store and Manage Your Work in a Uniform Way

Another interesting and convenient feature for MQL5 programming is your personal [MQL5 source codes storage](https://www.metatrader5.com/en/metaeditor/help/mql5storage "https://www.metatrader5.com/en/metaeditor/help/mql5storage"). Using it, you will always have direct access to your files via MetaEditor 5 from anywhere in the world. You can store not only MQL5 programs but also C++ sources (cpp, h), as well as BMP and WAV source files.

![Adding the file to MQL5 Storage](https://c.mql5.com/2/4/Add_File_To_MQL5_Storage_EN.png)

You can add and extract your codes, revert changes — in short, you can do anything that modern [SVN](https://en.wikipedia.org/wiki/Apache_Subversion "https://en.wikipedia.org/wiki/Apache_Subversion") systems have to offer. In addition to working with MQL5 Storage directly from MetaEditor 5, you can use any external client that supports [Subversion 1.7](https://www.mql5.com/go?link=http://subversion.tigris.org/ "http://subversion.tigris.org/"), like [Tortoise SVN](https://www.mql5.com/go?link=https://tortoisesvn.net/ "http://tortoisesvn.net/").

### The Life of Indicators, Charts and Graphic Objects

All the previous experience has been considered when developing MetaTrader 5. Therefore, some features may seem to be unusual at first. For example, the efficient model is used for [indicator calculation](https://www.mql5.com/en/docs/indicators) — an indicator represents a calculation part. Many Expert Advisors, scripts and other indicators can use results of one indicator. It also means that if one indicator is set on several charts with the same symbol and timeframe, the calculation will be performed in a single calculation entity. This approach saves both time and memory.

Also, the values of one indicator can be calculated using the values of another one or using the arrays' values in MQL5. That allows obtaining complex indicator calculations in a unified and simple way. As it has been mentioned before, the possibilities of [indicator values graphical representation](https://www.mql5.com/en/articles/45) in MQL5 language are really immense.

All operations concerning chart properties and graphic objects management are asynchronous ones. That prevents users from wasting time while waiting for the terminal video system to display changes in colors, sizes and so on. If you want to get immediate results of executing the functions from [Object functions](https://www.mql5.com/en/docs/objects) or [Chart operations](https://www.mql5.com/en/docs/chart_operations), call [ChartRedraw()](https://www.mql5.com/en/docs/chart_operations/chartredraw) for the chart forced redraw. Otherwise, the chart will be redrawn automatically by the terminal at the first opportunity.

### Trading Operations

Trading in MQL5 is performed by sending requests using [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function. A request is a special [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure filled with necessary values depending on a necessary trade action.

You can buy or sell, place pending orders to buy/sell under some definite terms or delete an existing pending order. If OrderSend() has been executed successfully, the trade request execution result is fixed in the [MqlTradeResult](https://www.mql5.com/en/docs/constants/structures/mqltraderesult) structure.

At first, you do not need to check the correctness of [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure filling when studying MQL5. The standard library has a special [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class for performing trade operations. This class has been designed to simplify the work of MQL5 programmers.

| Working with orders |
| --- |
| [OrderOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderopen) | Places the pending order with set parameters. |
| [OrderModify](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeordermodify) | Modifies the pending order parameters. |
| [OrderDelete](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderdelete) | Deletes the pending order. |
| Working with positions |
| [PositionOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionopen) | Opens the position with set parameters. |
| [PositionModify](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionmodify) | Modifies the position parameters. |
| [PositionClose](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionclose) | Closes the position by. |
| Additional methods |
| [Buy](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuy) | Opens a long position with specified parameters. |
| [Sell](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesell) | Opens a short position with specified parameters. |
| [BuyLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuylimit) | Places the pending order of Buy Limit type (buy at the price lower than current market price) with specified parameters. |
| [BuyStop](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuystop) | Places the pending order of Buy Stop type (buy at the price higher than current market price) with specified parameters. |
| [SellLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeselllimit) | Places the pending order of Sell Limit type (sell at the price higher than current market price) with specified parameters. |
| [SellStop](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesellstop) | Places the pending order of Buy Stop type (sell at the price lower than current market price) with specified parameters |

An example of a CTrade class application can be found in the MACD Sample training Expert Advisor from the terminal standard delivery. The Expert Advisor can be found at <terminal\_directory>\\MQL5\\Experts\\Examples\\MACD. Some other useful classes for working with orders, positions, deals and so on can be found in [Trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) section along with CTrade.

### MetaTrader 5 Strategy Tester

MetaTrader 5 not only allows trading on various financial markets using trading robots, but also provides the ability to check their profitability and stability over different parts of history. To achieve this, [Strategy Tester](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") has been implemented into the terminal.

It should be considered that the terminal acts like an execution manager, distributing tasks to individual services called [agents](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") when testing/optimizing an Expert Advisor. Thus, tests are performed as communication sessions between the terminal and the agents. The tester sends tasks to agents and gets execution results in return.

![Optimizing the trading system in MetaTrader 5 terminal tester](https://c.mql5.com/2/4/tester_MetaTrader5_EN.png)

Messages of the tester and the agents are placed to the [journal](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing"). When testing, the agents can send a very large number of messages generated in an Expert Advisor by the Print() and Alert() functions. Therefore, not all the messages from the agents are displayed in the journall some of them can be skipped. This is done to prevent testing slowdown because of the necessity to display all messages.

Therefore, the journal is stored separately at the <terminal\_folder>\\tester\\logs\\, while detailed logs with all messages are saved in the appropriate folders of tester agents. Keep this in mind when searching detailed logs for test analysis. Fortunately, the tester has a special [logs viewer](https://www.metatrader5.com/en/terminal/help/startworking/interface "https://www.metatrader5.com/en/terminal/help/startworking/interface") where you can find the logs for a definite interval.

Apart from testing, there is also the [optimization mode](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization "https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization") of an Expert Advisor's input parameters, in which the tester can use dozens, hundreds or even thousands of tester agents (for example, from [MQL5 Cloud Network](https://cloud.mql5.com/en "https://cloud.mql5.com/en")). In this case, messages sending and display by Print() and Alert() functions are completely suppressed not to increase the volume of the outgoing traffic in the direction of the tester and to save space on a PC hard disk where tester agents are located. The only exception is made for the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function. The function can use Print() to send the message that can clarify a reason for unsuccessful initialization or refusal to perform a test using the [ExpertRemove()](https://www.mql5.com/en/docs/common/expertremove) function for technical reasons.

You can find more interesting information in [Tester](https://www.mql5.com/en/articles/strategy_tester). We believe that you will appreciate the Strategy Tester possibilities in MetaTrader 5 client terminal.

### Pushing the Boundaries

Whoever you are, you will discover new opportunities after studying MQL5. That may include better understanding of programming languages, new insight into trading or acquaintance with new technologies. The new MetaTrader 5 terminal includes so many new features that we probably will not be able to find even a single developer who has managed to reach its whole potential up to now.

In this article, we still have not mentioned a lot of exciting things including convenient work with DLLs, downloading programs from the Code Base into the editor, their one-click launch in the terminal and much more. If you are not afraid of reading the long list of the terminal features, then you are welcome to do so in [MetaTrader 5: More Than You Can Imagine!](https://www.mql5.com/en/articles/384)

We wish you all well and hope to see you among the permanent members of MQL5.community!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/447](https://www.mql5.com/ru/articles/447)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/7312)**
(11)


![hongtao](https://c.mql5.com/avatar/avatar_na2.png)

**[hongtao](https://www.mql5.com/en/users/hongtao)**
\|
30 Nov 2014 at 07:07

[MetaTrader 5 - More than you can imagine!](https://www.mql5.com/en/articles/384) Endeavour!!!

![fengze.li](https://c.mql5.com/avatar/avatar_na2.png)

**[fengze.li](https://www.mql5.com/en/users/fengze.li)**
\|
27 Jul 2015 at 09:37

I studied computer science, I don't understand economics, and now my job is related to finance, so seeing this community is like seeing a lifesaver!


![Otto Pauser](https://c.mql5.com/avatar/2016/5/574C2261-ACAB.JPG)

**[Otto Pauser](https://www.mql5.com/en/users/kronenchakra)**
\|
25 Nov 2016 at 21:04

**MetaQuotes Software Corp.:**

New article [Quick introduction to MQL5](https://www.mql5.com/en/articles/447):

Author: [MetaQuotes Software Corp.](https://www.mql5.com/en/users/MetaQuotes "MetaQuotes")

"No conditional compilation of #ifdef, #else, #endif" etc."

This statement is wrong or mistranslated !!!!!!!!!

Conditional compilation is possible! Please correct!

![Alexey Petrov](https://c.mql5.com/avatar/2014/1/52E8F85F-2AD5.png)

**[Alexey Petrov](https://www.mql5.com/en/users/alexx)**
\|
28 Nov 2016 at 08:36

**Otto:**

"No conditional compilation of #ifdef, #else, #endif" etc."

This claim is incorrect or mistranslated !!!!!!!!!

Conditional compilation is possible! Please correct!

Hello,

Yes, you are right. This is a relatively old article. We have deleted the outdated information. Thank you

MfG


![Peng Peng Liu](https://c.mql5.com/avatar/avatar_na2.png)

**[Peng Peng Liu](https://www.mql5.com/en/users/yylnthz)**
\|
23 Dec 2023 at 07:49

Are there any more relevant courses


![Application of the Eigen-Coordinates Method to Structural Analysis of Nonextensive Statistical Distributions](https://c.mql5.com/2/0/Eigencoordinates_Nonextensive_Statistical_Distributions_MQL5.png)[Application of the Eigen-Coordinates Method to Structural Analysis of Nonextensive Statistical Distributions](https://www.mql5.com/en/articles/412)

The major problem of applied statistics is the problem of accepting statistical hypotheses. It was long considered impossible to be solved. The situation has changed with the emergence of the eigen-coordinates method. It is a fine and powerful tool for a structural study of a signal allowing to see more than what is possible using methods of modern applied statistics. The article focuses on practical use of this method and sets forth programs in MQL5. It also deals with the problem of function identification using as an example the distribution introduced by Hilhorst and Schehr.

![An Insight Into Accumulation/Distribution And Where It Can Get You](https://c.mql5.com/2/17/925_51.png)[An Insight Into Accumulation/Distribution And Where It Can Get You](https://www.mql5.com/en/articles/1357)

The Accumulation/Distribution (A/D) Indicator has one interesting feature - a breakout of the trend line plotted in this indicator chart suggests, with a certain degree of probability, a forthcoming breakout of the trend line in the price chart. This article will be useful and interesting for those who are new to programming in MQL4. Having this in view, I have tried to present the information in an easy to grasp manner and use the simplest code structures.

![Interview with Irina Korobeinikova (irishka.rf)](https://c.mql5.com/2/0/zh0ku.png)[Interview with Irina Korobeinikova (irishka.rf)](https://www.mql5.com/en/articles/465)

Having a female member on the MQL5.community is rare. This interview was inspired by a one of a kind case. Irina Korobeinikova (irishka.rf) is a fifteen-year-old programmer from Izhevsk. She is currently the only girl who actively participates in the "Jobs" service and is featured on the Top Developers list.

![How to Make a Trading Robot in No Time](https://c.mql5.com/2/0/development.png)[How to Make a Trading Robot in No Time](https://www.mql5.com/en/articles/443)

Trading on financial markets involves many risks including the most critical one - the risk of making a wrong trading decision. The dream of every trader is to find a trading robot, which is always in good shape and not subject to human weaknesses - fear, greed and impatience.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=bdwmxaxlebrxgdbkzylzbguxmjflzdow&ssn=1769191247823228694&ssn_dr=0&ssn_sr=0&fv_date=1769191247&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F447&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Fast%20Dive%20into%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919124765859876&fz_uniq=5071505092932217167&sv=2552)

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