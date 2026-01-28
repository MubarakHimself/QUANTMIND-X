---
title: MetaTrader 5 on macOS
url: https://www.mql5.com/en/articles/619
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T17:58:52.679148
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/619&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068871556425252523)

MetaTrader 5 / Trading


We provide a special installer for the MetaTrader 5 trading platform on macOS. It is a full-fledged wizard that allows you to install the application natively. The installer performs all the required steps: it identifies your system, downloads and installs the latest [Wine](https://www.mql5.com/go?link=https://www.winehq.org/ "https://www.winehq.org/") version, configures it, and then installs MetaTrader within it. All steps are completed in the automated mode, and you can start using the platform immediately after installation.

You can download the installer via this [link](https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/MetaTrader5.pkg.zip?utm_source=www.mql5.com&utm_campaign=download "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/MetaTrader5.pkg.zip?utm_source=www.mql5.com&utm_campaign=download") or via the Help menu in the trading platform:

![Download links in the platform menu](https://c.mql5.com/2/114/macos-linux-installers.png)

### System Requirements

The minimum macOS version required to install MetaTrader 5 is Catalina (10.15.7). The platform runs on all modern versions of macOS and supports all Apple processors, from M1 to the latest released versions.

### Preparation: Check the Wine version

If you are already using MetaTrader on macOS, please check the current Wine version, which is displayed in the platform log upon startup:

LP 0 15:56:29.402 Terminal MetaTrader 5 x64 build 4050 started for MetaQuotes Software Corp.

PF 0 15:56:29.403 Terminal Windows 10 build 18362 on Wine 8.0.1 Darwin 23.0.0, 12 x Intel Core i7-8750H  @ 2.20GHz, AVX2, 11 / 15 Gb memory, 65 / 233 Gb disk, admin, GMT+2

If your Wine version is below 8.0.1, we strongly recommend uninstalling the old platform along with the Wine prefix in which it is installed. Be sure to save all necessary files in advance, including templates, downloaded Expert Advisors, indicators, and others. You can uninstall the platform as usual by moving it from the "Applications" section to the Trash. The Wine prefix can be deleted using Finder. Select the "Go > Go to Folder" menu and enter the directory name: ~/Library/Application Support/.

![Go to the directory with the Wine prefix](https://c.mql5.com/2/114/finder.PNG)

Delete the following folders from this directory:

~/Library/Application Support/Metatrader 5

~/Library/Application Support/net.metaquotes.wine.metatrader5

### Installation

The MetaTrader 5 platform is installed like a standard macOS application. Run the downloaded file and follow the instructions. During the process, you will be prompted to install additional Wine packages (Mono, Gecko). Please agree to this as they are necessary for the platform functioning.

![Installing MetaTrader 5 in MacOS](https://c.mql5.com/2/114/macos_mt5.png)

Wait for the installation to complete, then begin working with MetaTrader 5:

![MetaTrader 5 on Mac OS](https://c.mql5.com/2/114/platform-macos.PNG)

### MetaTrader 5 Data Directory

A separate virtual logical drive with the necessary environment is created for MetaTrader 5 in Wine. The default path of the installed platform's data folder is as follows:

~/Library/Application Support/net.metaquotes.wine.metatrader5/drive\_c/Program Files/MetaTrader 5

### Interface Language Settings

When installing MetaTrader 5, Wine automatically adds support for the language (locale) currently set for macOS. In most cases, this is sufficient. If you wish to use a different language for the platform, switch the macOS language to the desired one before installation and restart your computer. Then, proceed with installing the platform. After the installation, you can set macOS to its original language.

[Download MetaTrader 5 for macOS](https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/MetaTrader5.pkg.zip "Download MetaTrader 5 for macOS")

Try the [MetaTrader 5 mobile app for iPhone/iPad](https://www.metatrader5.com/en/mobile-trading/iphone "MetaTrader 5 for iPhone"). It allows you to monitor the market, execute trades, and manage your trading account from anywhere in the world.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/619](https://www.mql5.com/ru/articles/619)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/10047)**
(69)


![cytrader](https://c.mql5.com/avatar/avatar_na2.png)

**[cytrader](https://www.mql5.com/en/users/cytrader)**
\|
20 Nov 2020 at 20:43

XM is already offering MT4 and MT5 terminals on Mac OS. It runs on Catalina and Big Sur.

I have recently updated to Big Sur and both terminals run fine.

![magicmarco](https://c.mql5.com/avatar/avatar_na2.png)

**[magicmarco](https://www.mql5.com/en/users/magicmarco)**
\|
21 Apr 2022 at 08:47

Hello, I installed the promoted MT5, that works with crossover. I can't find the folders on my mac to copy the [profiles](https://www.metatrader5.com/en/metaeditor/help/development/profiling "MetaEditor User Guide: Code profiling") and indicators. How does this crossover thing work?


![Thiago Duarte](https://c.mql5.com/avatar/2019/5/5CCB0EAE-0909.jpg)

**[Thiago Duarte](https://www.mql5.com/en/users/thiagoduarte)**
\|
25 Dec 2022 at 02:26

It worked fine, the dmg file works perfectly. Thank you!


![p4rnak](https://c.mql5.com/avatar/avatar_na2.png)

**[p4rnak](https://www.mql5.com/en/users/p4rnak)**
\|
11 Nov 2023 at 01:28

Guys please, just just just install MT5 via "Cross Over" application on MacOS (ventura is tested).

It's Amazing... High speed Like Windows.

Good luck

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
28 Dec 2023 at 06:03

p.p1 {margin: 0.0px 0.0px 0.0px 0.0px; font: 13.0px 'Helvetica Neue'}

Error in POL\_Wine

Starting 64-bit process mt5setup.exe is not supported in 32-bit virtual drives

Can u help me with this code

![MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://c.mql5.com/2/110/MQL5_Trading_Toolkit_Part_6___LOGO.png)[MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)

Learn how to create an EX5 module of exportable functions that seamlessly query and save data for the most recently filled pending order. In this comprehensive step-by-step guide, we will enhance the History Management EX5 library by developing dedicated and compartmentalized functions to retrieve essential properties of the last filled pending order. These properties include the order type, setup time, execution time, filling type, and other critical details necessary for effective pending orders trade history management and analysis.

![Neural Networks in Trading: Piecewise Linear Representation of Time Series](https://c.mql5.com/2/82/Neural_networks_are_simple_Piecewise_linear_representation_of_time_series__LOGO.png)[Neural Networks in Trading: Piecewise Linear Representation of Time Series](https://www.mql5.com/en/articles/15217)

This article is somewhat different from my earlier publications. In this article, we will talk about an alternative representation of time series. Piecewise linear representation of time series is a method of approximating a time series using linear functions over small intervals.

![Developing a Replay System (Part 56): Adapting the Modules](https://c.mql5.com/2/83/Desenvolvendo_um_sistema_de_Replay_Parte_56__LOGO_3_.png)[Developing a Replay System (Part 56): Adapting the Modules](https://www.mql5.com/en/articles/12000)

Although the modules already interact with each other properly, an error occurs when trying to use the mouse pointer in the replay service. We need to fix this before moving on to the next step. Additionally, we will fix an issue in the mouse indicator code. So this version will be finally stable and properly polished.

![Adaptive Social Behavior Optimization (ASBO): Schwefel, Box-Muller Method](https://c.mql5.com/2/84/Adaptive_Social_Behavior_Optimization___LOGO.png)[Adaptive Social Behavior Optimization (ASBO): Schwefel, Box-Muller Method](https://www.mql5.com/en/articles/15283)

This article provides a fascinating insight into the world of social behavior in living organisms and its influence on the creation of a new mathematical model - ASBO (Adaptive Social Behavior Optimization). We will examine how the principles of leadership, neighborhood, and cooperation observed in living societies inspire the development of innovative optimization algorithms.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/619&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068871556425252523)

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