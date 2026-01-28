---
title: How to Prepare a Trading Account for Migration to Virtual Hosting
url: https://www.mql5.com/en/articles/994
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:38:45.567657
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/994&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083010949541008179)

MetaTrader 5 / Examples


MetaTrader client terminal is perfect for automating trading strategies. It has all tools necessary for trading robot developers ‒ powerful C++ based MQL4/MQL5 programming language, convenient MetaEditor development environment and multi-threaded strategy tester that supports distributed computing in MQL5 Cloud Network. In this article, you will find out how to move your client terminal to the virtual environment with all custom elements.

### How to Provide the Reliable Round-the-Clock Operation of the Terminal?

A trader may need a terminal running 24 hours a day in the following three cases:

- a trader has a trading robot developed by his or her efforts or [ordered from programmers](https://www.mql5.com/en/articles/117);
- a trader has an Expert Advisor [purchased in the Market](https://www.mql5.com/en/articles/498);
- a trader has [subscribed to a Signal](https://www.mql5.com/en/articles/523).

All these cases require constant connection to a trade server and uninterrupted power supply. Using a home PC is not always possible and convenient. Until recently, the most popular solution has been renting allocated computing capacities as VDS or VPS from specialized companies.

MetaTrader platform offers much more convenient and quick solution ‒ you can rent a virtual server for your trading account right from the client terminal.

### What Is a Virtual Terminal

Virtual terminal has been developed specifically for working in Virtual Hosting Cloud network offering rental services. Any trader may rent a ready-made virtual server with already arranged trading environment in a few mouse clicks right from the client terminal.

### Allocating a Virtual Server

To receive a virtual terminal, select the appropriate trading account and execute "Register a Virtual Server" command in the context menu.

![](https://c.mql5.com/2/11/fig1_register__2.png)

Virtual Hosting Wizard window appears. It shows how the virtual hosting network works. The process of obtaining a virtual server consists of three steps. First, you will find out how to prepare for migration. After that, you will select the nearest virtual server with minimal network latency to your broker's trade server.

![](https://c.mql5.com/2/11/VirtualHostingMaster__2.gif)

You can choose 1440 free minutes provided to each registered MQL5.com user or select one of the offered service plans. Finally, you will select the data migration mode depending on your objectives:

- complete migration is necessary if you want to simultaneously launch Expert Advisors/indicators and trade copying;
- only Expert Advisors and indicators if subscription to signals is not required;
- only trade copying - only Signal copying settings (no charts or programs) are moved.


There are no limitations on the number of charts and Expert Advisors/indicators. The number of activations for products purchased in the Market is not decreased when launching them on a virtual terminal.

After selecting the migration mode, you can launch the virtual server immediately by clicking "Migrate now" or do that later at any time.

Congratulations! Now, you have your own virtual server with MetaTrader terminal ready for work!

### Preparing for Migration

Before launching the virtual terminal, you should prepare an active environment for it - charts, launched indicators and Expert Advisors, Signal copying parameters and the terminal settings.

**Charts and Market Watch**

In the [Market Watch](https://www.metatrader5.com/en/terminal/help/trading/market_watch "https://www.metatrader5.com/en/terminal/help/trading/market_watch"), set up the list of symbols critical for your Expert Advisors' operation. We recommend that you remove all unnecessary symbols to decrease the tick traffic received by the terminal. There is no point in keeping hundreds of symbols in the "Market Watch" if only a couple of them are used for trading.

Open only the charts that you really need. Although there are no limitations on the number of open charts, there is no point in opening unnecessary ones. Color settings do not matter.

Set "Max bars in chart" parameter in [Charts](https://www.metatrader5.com/en/terminal/help/startworking/settings "https://www.metatrader5.com/en/terminal/help/startworking/settings") tab of the terminal settings. Some custom indicators are developed in a wasteful way and perform calculations on all history available on the chart. In that case, the lesser the specified value, the better. However, make sure that the indicator works correctly with these settings by restarting the terminal after changing this parameter.

The virtual terminal has been designed so that it automatically downloads all available history from a trade server, but not more than 500 000 bars are available on a chart.

**Indicators and Expert Advisors**

Apply to the charts all indicators and Expert Advisors that are necessary for the terminal's autonomous operation. Most trading robots do not refer to indicators on the charts, so check out and decide what you really need.

Products purchased on the [Market](https://www.mql5.com/en/market) and launched on the chart are also moved during migration. They remain completely functional, and the number of available activations is not decreased. Automatic licensing of purchased products without spending available activations is provided only for the virtual terminal.

DLL calls are completely forbidden in the virtual terminal. During the first attempt to call a function from DLL, the launched program is stopped with the critical error.

All external parameters of indicators and Expert Advisors should be set correctly. Check them once again before launching synchronization.

Scripts cannot be moved during migration even if they have been launched in an endless loop on the chart at the time of synchronization.

Virtual terminal can be automatically restarted when it is updated through the [LiveUpdate](https://www.metatrader5.com/en/terminal/help/start_advanced/autoupdate "https://www.metatrader5.com/en/terminal/help/start_advanced/autoupdate"), as well as during the maintenance of virtual hosting. Thus, all the programs intended to operate on the virtual platform must correctly process terminal stop and restart in order to correctly continue their operation after these events.

[Terminal global variables](https://www.mql5.com/en/docs/globals) are not migrated to the virtual hosting. If you need to initialize lots of variables when starting a program, you can use reading from files that can be passed using the " [#property tester\_file](https://www.mql5.com/en/docs/basis/preprosessor/compilation)" directive.

**Sending Files**

If a certain file is required for an Expert Advisor or indicator, you can send it to the virtual terminal by specifying [a #property parameter](https://www.mql5.com/en/docs/basis/preprosessor/compilation):

- #property tester\_file "data\_file\_name"- for sending a file from <data\_folder>\\MQL5\\Files or <data\_folder>\\MQL4\\Files
- #property tester\_indicator "indicator\_name" - for sending a [custom indicator](https://www.mql5.com/en/docs/customind) from <data\_folder>\\MQL5\\Indicators or <data\_folder>\\MQL4\\Indicators
- #property tester\_library - "library\_name" - for sending a library from <data\_folder>\\MQL5\\Libraries or <data\_folder>\\MQL4\\Libraries

Please note that called libraries are identified and sent to the hosting automatically during the migration even if they are not specified. Therefore, you do not need to specify them. Also, you do not need to specify indicators that are explicitly called in the code by their names via the [iCustom()](https://www.mql5.com/en/docs/indicators/icustom) function.

When migrating, these directives are identified by the terminal and the necessary files are sent. The file size should not exceed 64 MB.

Sample code for sending files of the following three types to the virtual terminal:

```
    #property tester_file "trade_patterns.csv"    // A data file an Expert Advisor is to work with. It should be specified if needed on the hosting
    #property tester_indicator "smoothed_ma.ex5"  // Custom indicator file if the indicator name can be identified
    #property tester_library - "alglib.ex5"       // Library of the functions called in an Expert Advisor. You do not need to specify it
```

**Configuring Email, FTP and Signals**

If an Expert Advisor is to send emails, upload data via FTP or copy Signal trades, make sure to specify all necessary settings. Set correct login and password of your MQL5.community account in [Community](https://www.metatrader5.com/en/terminal/help/startworking/settings "https://www.metatrader5.com/en/terminal/help/startworking/settings") tab. This is necessary for Signal copying.

![](https://c.mql5.com/2/11/settings__2.gif)

It is highly recommended that you specify your MetaQuotes ID and allow sending messages of performed trades in Notifications tab. Thus, you will stay aware of what is going on at your trading account without even opening your terminal.

**Permission to Trade and Signal Copying**

The automated trading is always allowed in the virtual terminal. Therefore, any Expert Advisor with trading functions launched during synchronization can trade on the virtual terminal after the migration. Do not launch the Expert Advisors you are not sure about.

Regardless of whether autotrading is allowed or forbidden in your client terminal or in the properties of a launched Expert Advisor, any trading robot is allowed to trade after being moved to the virtual terminal.

Set necessary trade copying parameters in [Signals](https://www.metatrader5.com/en/terminal/help/startworking/settings "https://www.metatrader5.com/en/terminal/help/startworking/settings") tab. If a trading account has an active subscription and trade copying is allowed, permission to copy signals is disabled in the client terminal during migration. This is done in order to prevent the situation when two terminals connected to the same account copy the same trades simultaneously.

Trade copying is automatically enabled on the virtual terminal when migration is complete. Message about copy cancellation in the client terminal is also repeated in the journal.

**Setting WebRequest**

If a program that is to operate in the virtual terminal uses [WebRequest()](https://www.mql5.com/en/docs/network/webrequest) function for sending HTTP requests, you should set permission and list all trusted URLs in [Expert Advisors](https://www.metatrader5.com/en/terminal/help/startworking/settings "https://www.metatrader5.com/en/terminal/help/startworking/settings") tab.

### Migration

Migration is transferring the current active environment from the client terminal to the virtual one. This is a simple and straightforward way to change the set of launched programs, open charts and subscription parameters in the virtual terminal.

Migration is performed during each synchronization of the client terminal. Synchronization is always a one-direction process - the client terminal's environment is moved to the virtual terminal but never vice versa. The virtual terminal status can be monitored via requesting the terminal's and Expert Advisors' logs as well as virtual server's monitoring data.

To perform synchronization, execute "Synchronize Environment" command and select migration type.

![](https://c.mql5.com/2/11/fig7_synch_start__2.png)

Thus, you always can change the number of charts and the list of symbols in the Data Window, the set of launched programs and their input parameters, the terminal settings and Signal subscription.

When performing migration, all data is recorded in the client terminal's log.

![](https://c.mql5.com/2/11/fig8_migration__2.png)

After the synchronization, open the virtual terminal's main journal to examine the actions performed on it.

![](https://c.mql5.com/2/11/fig9_main_journal__2.png)

In the newly opened log window, you can set a piece of text the journal entries are to be filtered by and a desired interval. After that, click Request to download the found logs.

![](https://c.mql5.com/2/11/fig10_log_migration__3.png)

Virtual terminal logs themselves are updated at each request and saved in <terminal data folder>/logs/hosting.<hosting\_ID>.terminal and <terminal data folder>/logs/hosting.<hosting\_ID>.experts.

### Working with the Virtual Terminal

The rented virtual server status can also be easily monitored from the client terminal. Execute "Details" command in the context menu.

![](https://c.mql5.com/2/11/fig11_details__2.png)

The newly opened dialog window shows the virtual server's monitoring data:

- CPU usage graph, %;
- memory usage graph, Mb;
- hard disk usage graph, Mb.


![](https://c.mql5.com/2/11/UsageDetails.gif)

The main tab contains data on the virtual server itself and the terminal's active environment:

1. server name and rent number;
2. rent start date, MQL5.com account and trading account status;
3. used service plan and remaining rental time;

4. status  - started or stopped.

Besides, the following data is displayed for the virtual terminal:

- last migration date and mode;
- data on Signal subscription's migration and disabling trade copying on the client terminal (if an active subscription is present);
- number of opened charts, Expert Advisors/indicators launched on it, moved EX4/EX5 libraries and created files.

![](https://c.mql5.com/2/11/details1__1.png)

The context menu of the rented server's icon is also used to launch and stop the virtual terminal. The rent can also be canceled there. No refund is provided in case of an early cancellation.

![](https://c.mql5.com/2/11/fig12_stopserver__2.png)

### Virtual Hosting Is the Best Solution for the Automated Trading!

The benefits of the virtual hosting service are evident:

- fast and easy way to receive a virtual server directly from the client terminal;
- ability to examine and test the service within free 1440 minutes that can be used in parts;
- ready-made and configured virtual terminal;
- flexible service plans with discounts depending on a rental period duration;

- ability to select location with the minimum network latency to your broker's trade server;
- you can easily pay rental fees using your unified MQL5.community account. Ability to pay from a trading account is underway.

What do traders need to be able to perform the automated trading round-the-clock or copy trading signals? They need transparent and intuitive service that ensures reliable terminal operation with a guaranteed connection to a trade server and requires minimum efforts from a user.

Virtual hosting solves these issues - simply choose a virtual server and use your free 1440 minutes to check it out!

See also:

- [Videos on Virtual Hosting Released](https://www.mql5.com/en/forum/42396)
- [Why Virtual Hosting On The MetaTrader 4 And MetaTrader 5 Is Better Than Usual VPS](https://www.mql5.com/en/articles/1171)
- [Rules of Using the Virtual Hosting Service](https://www.mql5.com/en/hosting/rules)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/994](https://www.mql5.com/ru/articles/994)

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
**[Go to discussion](https://www.mql5.com/en/forum/36608)**
(507)


![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
14 May 2025 at 19:26

**Jungkook jungkook [#](https://www.mql5.com/en/forum/36608/page18#comment_56702467):**

How can I transfer money to my account

If you are talking about your trading account money, contact your broker.

MQL5.com is not a broker, nor has anything to do with your trading account money.

![vikor167](https://c.mql5.com/avatar/avatar_na2.png)

**[vikor167](https://www.mql5.com/en/users/vikor167)**
\|
18 Oct 2025 at 16:53

hint! If I pay for [shared hosting](https://www.mql5.com/en/vps "Forex VPS for MetaTrader 4/5"), is it only to one specific account?


![Sergey Golubev](https://c.mql5.com/avatar/2012/12/50D09349-3D90.jpg)

**[Sergey Golubev](https://www.mql5.com/en/users/newdigital)**
\|
18 Oct 2025 at 17:10

**vikor167 [#](https://www.mql5.com/en/forum/36608/page51#comment_58301085):**

hint! If I pay for [shared hosting](https://www.mql5.com/en/vps "Forex VPS for MetaTrader 4/5"), is it only to one specific account?

Yes, MQL5 VPS is per one trading account and per one forum login (Community login in Metatrader).

More in details:

- Launching MetaTrader VPS: [A step-by-step guide for first-time user](https://www.mql5.com/en/forum/455825)
- [Rules of Using the Virtual Hosting Service](https://www.mql5.com/en/vps/rules)  (там есть определенные ограничения, например, по dll)
- Step by step guide: [https://www.mql5.com/en/articles/13586](https://www.mql5.com/en/articles/13586)

![wazqaz](https://c.mql5.com/avatar/avatar_na2.png)

**[wazqaz](https://www.mql5.com/en/users/wazqaz)**
\|
26 Oct 2025 at 15:55

Is it possible to get copies of .csv and .json files [created by EA](https://www.metatrader5.com/en/terminal/help/algotrading/autotrading "MetaTrader 5 Help: Create an Expert Advisor in the MetaTrader 5 Client Terminal") during testing on VPS? If yes, how do I set this up to automatically save once every 24 hours to my mailbox or googledisk folder?

![Igor Bruno De Souza Cordeiro Freitas](https://c.mql5.com/avatar/avatar_na2.png)

**[Igor Bruno De Souza Cordeiro Freitas](https://www.mql5.com/en/users/iscfreitas)**
\|
7 Jan 2026 at 19:52

2026.01.07 20:36:37.186 Network xxxxxxxxx': authorisation on 170.82.68.216:443 failed (Invalid one-time password)

2026.01.07 20:37:06.609 Terminal 'xxxxxxxxx': 4 charts, 4 EAs, 0 custom indicators, signal disabled, not [connected to trade server](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus#enum_terminal_info_integer "MQL5 documentation: Client Terminal Properties")

can anyone help me with these errors?

I migrate my EAs but nothing happens because I can't authenticate.

![MQL5 Cookbook: Processing of the TradeTransaction Event](https://c.mql5.com/2/11/MQL5_Recipes_OnTradeTransaction_MetaTrader5.png)[MQL5 Cookbook: Processing of the TradeTransaction Event](https://www.mql5.com/en/articles/1111)

This article considers capabilities of the MQL5 language from the point of view of the event-driven programming. The greatest advantage of this approach is that the program can receive information about phased implementation of a trade operation. The article also contains an example of receiving and processing information about ongoing trade operation using the TradeTransaction event handler. In my opinion, such an approach can be used for copying deals from one terminal to another.

![Regression Analysis of the Influence of Macroeconomic Data on Currency Prices Fluctuation](https://c.mql5.com/2/11/fundamental_analysis_statistica_MQL5_MetaTrader5.png)[Regression Analysis of the Influence of Macroeconomic Data on Currency Prices Fluctuation](https://www.mql5.com/en/articles/1087)

This article considers the application of multiple regression analysis to macroeconomic statistics. It also gives an insight into the evaluation of the statistics impact on the currency exchange rate fluctuation based on the example of the currency pair EURUSD. Such evaluation allows automating the fundamental analysis which becomes available to even novice traders.

![MQL5 Cookbook: Handling Typical Chart Events](https://c.mql5.com/2/11/OnChartEvent_MetaTrader5.png)[MQL5 Cookbook: Handling Typical Chart Events](https://www.mql5.com/en/articles/689)

This article considers typical chart events and includes examples of their processing. We will focus on mouse events, keystrokes, creation/modification/removal of a graphical object, mouse click on a chart and on a graphical object, moving a graphical object with a mouse, finish editing of text in a text field, as well as on chart modification events. A sample of an MQL5 program is provided for each type of event considered.

![Indicator for Constructing a Three Line Break Chart](https://c.mql5.com/2/10/logo.png)[Indicator for Constructing a Three Line Break Chart](https://www.mql5.com/en/articles/902)

This article is dedicated to the Three Line Break chart, suggested by Steve Nison in his book "Beyond Candlesticks". The greatest advantage of this chart is that it allows filtering minor fluctuations of a price in relation to the previous movement. We are going to discuss the principle of the chart construction, the code of the indicator and some examples of trading strategies based on it.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qgrklmukqrlbxvtakvhghflkwpixjagp&ssn=1769251124087793907&ssn_dr=0&ssn_sr=0&fv_date=1769251124&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Prepare%20a%20Trading%20Account%20for%20Migration%20to%20Virtual%20Hosting%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925112443983732&fz_uniq=5083010949541008179&sv=2552)

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