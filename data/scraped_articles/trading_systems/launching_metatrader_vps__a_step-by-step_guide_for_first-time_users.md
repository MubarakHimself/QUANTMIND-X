---
title: Launching MetaTrader VPS: A step-by-step guide for first-time users
url: https://www.mql5.com/en/articles/13586
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:23:18.727160
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=qtssorbyzeqpeglnaybtgzgpueouubtd&ssn=1769185397232039428&ssn_dr=0&ssn_sr=0&fv_date=1769185397&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13586&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Launching%20MetaTrader%20VPS%3A%20A%20step-by-step%20guide%20for%20first-time%20users%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918539755952550&fz_uniq=5070244966707434100&sv=2552)

MetaTrader 5 / Trading systems


Everyone who uses trading robots or signal subscriptions sooner or later recognizes the need to rent a reliable 24/7 hosting server for their trading platform. We recommend using MetaTrader VPS [for several reasons](https://www.metatrader5.com/en/news/2293 "8 reasons to choose MetaTrader VPS over non-specialized hosting solutions"). You can conveniently pay and manage the subscription through your [MQL5.community](https://www.mql5.com/en "MQL5.community") account. If you haven't registered on MQL5.com yet, take a moment to sign up and specify your account in the platform settings.

![Launching MetaTrader VPS: A step-by-step guide for first-time users](https://c.mql5.com/2/59/aaa206-vps-cover-ill-big___3.png)

### 1\. Getting started

Connect to the trading account for which you want to run a VPS, and you are ready to migrate your platform to the cloud. The most straightforward way is to click on the "Open MQL5 Virtual Hosting" icon in the platform's top menu:

![Open MQL5 Virtual Hosting](https://c.mql5.com/2/59/01_vps_tips_icon_2___2.png)

You can also select the virtual server option in the Navigator or in the trading account's context menu. A window will open showing the closest server and an estimated reduction in the delay compared to your current connection. Lower network latency provides better trading execution conditions, such as minimized slippage and reduced probability of a requote.

### 2\. Service plans

Before choosing a payment plan, check if your broker offers MetaTrader 5 virtual hosting for free. Brokers participating in our recently launched Sponsored VPS program can provide virtual hosting to their clients as a bonus, subject to certain conditions.

![Choose a VPS subscription plan](https://c.mql5.com/2/59/02_vps-pay___2.png)

Choose a VPS subscription plan: the longer the rental period, the lower the monthly cost. The plan can be changed when the rental period expires, that is, after a month. An auto-renewal option is provided for those who prefer a hands-off approach to monitoring the service status. This option can be enabled or disabled at any time. Furthermore, before any payments, the system will check the hosting status. If it is inactive, the subscription will not be renewed, safeguarding you from unnecessary expenses on unused servers.

**Please note that if the subscription is interrupted, all hosting data will be lost. The server can be rented again, but you will have to re-configure the entire environment. The auto-renewal option can assist in avoiding such a situation.**

### 3\. Payment

All your payments are registered on MQL5.com, ensuring a unified and transparent hosting rental history. This eliminates the need to search for payments across different payment systems if such information is ever needed. Pay using your preferred method and access all transaction details in your profile. The hosting is ready for use immediately after payment, and thus you can proceed to migrate your local platform environment to the virtual server.

### 4\. Preparation

Migration means transferring the current active environment from the trading platform to the virtual one. Your pre-configured set of charts, Expert Advisors, indicators, and copied signals can be seamlessly transferred to the VPS with a single command. A copy of your platform with the relevant settings and programs will run on the virtual server. Therefore, you need to prepare your local platform before migrating.

In the [Market Watch](https://www.metatrader5.com/en/terminal/help/trading/market_watch "Market Watch") window, set up the list of symbols required for your Expert Advisors' operation. Remove unused symbols and charts to reduce resource consumption. Add the necessary indicators and Expert Advisors to the charts for autonomous operation. Most trading robots do not use on-chart indicators, so review all programs and leave only the required ones. If your Expert Advisor will send emails, submit data via FTP, or copy Signal trades, specify the relevant settings. To copy signals, please make sure to specify your MQL5.community account credentials in the Community tab.

[Read more about preparation](https://www.mql5.com/en/articles/994 "How to Prepare a Trading Account for Migration to Virtual Hosting")

### 5\. Migration

The trading environment is migrated with each synchronization of the platform. Synchronization is always performed in one direction: the local platform environment is migrated to the virtual platform, never the reverse. The virtual platform status can be monitored using platform and Expert Advisor logs, as well as via virtual server monitoring data.

To start synchronization, go to the VPS section and select the migration type:

- Full – if you want to simultaneously run Expert Advisor/indicators and copy signal subscriptions. In this mode, account connection data, all open charts, signal copying parameters, running Expert Advisors and indicators, FTP parameters, and email settings are copied to the virtual server.
- Expert – run Expert Advisors and indicators.
- Signal – copy a signal subscription.

The available history data for all open charts is automatically uploaded to the VPS during the first synchronization. Fetching history from the trading server may take some time, and all robots running on the charts should correctly process the updated data.

- Automated trading is always enabled in the virtual platform even if it is disabled in local platform settings or in the running Expert Advisor's parameters.
- Scripts are not transferred during migration even if they were running in an endless loop during synchronization.
- [Charts with non-standard or custom timeframes and symbols](https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments "Custom Financial Instruments") are not migrated.
- Accounts with one-time password authentication cannot be used on the VPS. Autonomous platform operation is impossible if manual one-time password specification is required for each connection.

[Read more about migration](https://www.metatrader5.com/en/terminal/help/virtual_hosting/virtual_hosting_migration "Migration")

### 6\. Operation

You can monitor the state of the rented server from the trading platform. The Tools \ VPS section provides the following options:

- View the virtual server data
- Synchronize the environment by performing the immediate migration
- Request platform and Expert Advisor logs
- Stop the server

![The Tools \ VPS section](https://c.mql5.com/2/59/vps_tips_eng_.jpg)

Watch [our new video](https://www.youtube.com/watch?v=SlKUjRISEfk "How to Control Resources and Manage Virtual Hosting Subscriptions") to learn how to analyze virtual hosting reports and how to control your subscriptions.

To monitor the operation of the virtual platform, use the VPS \ Log section.

![The VPS \ Log section](https://c.mql5.com/2/59/04_vps-journal___2.png)

For further details about how to monitor the VPS please read the [Documentation](https://www.metatrader5.com/en/terminal/help/virtual_hosting/virtual_hosting_terminal "Working with the Virtual Platform").

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/13586](https://www.mql5.com/ru/articles/13586)

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
**[Go to discussion](https://www.mql5.com/en/forum/455841)**
(24)


![Jay](https://c.mql5.com/avatar/avatar_na2.png)

**[Jay](https://www.mql5.com/en/users/bonanzagroove)**
\|
28 Aug 2025 at 16:52

Hi,

My VPS that's been running for about a month now is no longer starting up.

What I can see in the Journal locally is that around 30 second after the "start command" there's a "status is 'stopped'" being written.

I've tried starting the VPS in as many ways I could to no avail: right-click in Navigator, VPS tab in the chart window, [https://www.mql5.com/en/vps/subscriptions](https://www.mql5.com/en/vps/subscriptions "https://www.mql5.com/en/vps/subscriptions")

It has started acting up earlier today and has already missed a precious signal. How I can get assistance with this?

At a few points throughout the day, it looked like the VPS could come back up but then after a while it started no longer reacting to the start commands.

This is the last meaningful messages in the Journal:

2025.08.28 16:06:41.502    Terminal    'accountID': 1 chart, 1 EA, 1 custom indicator, signal disabled

2025.08.28 16:06:47.478    Network    'accountID': ping to current access point Access Server LIVE MT5 New is 0.52 ms

2025.08.28 16:06:47.478    Network    'accountID': scanning network finished

2025.08.28 16:06:59.798    Virtual Hosting    close command received from Hosting Server

2025.08.28 16:07:25.289    Terminal    cannot load config "C:\\Hosting\\instances\\instanceID\\start.ini" at start

2025.08.28 16:07:26.280    Startup    invalid configuration file (-1)

2025.08.28 16:07:26.280    Terminal    exit with code 0

2025.08.28 16:07:26.280    Terminal    initialization failed \[10015\]

2025.08.28 16:07:26.280    Terminal    stopped with 10015

2025.08.28 16:07:26.455    Terminal    shutdown with 10015

What to do next? Thanks

![domgo640](https://c.mql5.com/avatar/avatar_na2.png)

**[domgo640](https://www.mql5.com/en/users/domgo640)**
\|
28 Oct 2025 at 12:29

Hi, I rented a VPS, I have not been able to migrate my MT4, when I open the VPS button on the [platform](https://www.mql5.com/en/quotes/metals/XAGUSD "XAGUSD chart: technical analysis") I always get the payment window. Where should I click to migrate the platform to the VPS ? Thank you.


![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
28 Oct 2025 at 14:54

**domgo640 [#](https://www.mql5.com/en/forum/455841/page3#comment_58379264):**

Hi, I rented a VPS, I have not been able to migrate my MT4, when I open the VPS button on the [platform](https://www.mql5.com/en/quotes/metals/XAGUSD "XAGUSD chart: technical analysis") I always get the payment window. Where should I click to migrate the platform to the VPS ? Thank you.

You need to setup your MQL5 VPS from the trading account you used to subscribe in the first place.

Launching MetaTrader VPS: A step-by-step guide for first-time users

[https://www.mql5.com/en/articles/13586](https://www.mql5.com/en/articles/13586 "https://www.mql5.com/en/articles/13586")

[https://www.metatrader5.com/en/terminal/help/virtual\_hosting/virtual\_hosting\_terminal](https://www.metatrader5.com/en/terminal/help/virtual_hosting/virtual_hosting_terminal "https://www.metatrader5.com/en/terminal/help/virtual_hosting/virtual_hosting_terminal")

Read the last steps 15-25 of these guides below to understand how MQL5 VPS works:

[https://www.mql5.com/en/forum/366152](https://www.mql5.com/en/forum/366152 "https://www.mql5.com/en/forum/366152")  (MT4)

[https://www.mql5.com/en/forum/366161](https://www.mql5.com/en/forum/366161 "https://www.mql5.com/en/forum/366161")  (MT5)

![49Tomi-s](https://c.mql5.com/avatar/avatar_na2.png)

**[49Tomi-s](https://www.mql5.com/en/users/49tomi-s)**
\|
28 Nov 2025 at 11:43

I use Tickmill EU, I am not able to complete the purchase of VPS. I enter the data during the payment process, press Pay and the purchase is not completed?


![elvetia.mascol](https://c.mql5.com/avatar/avatar_na2.png)

**[elvetia.mascol](https://www.mql5.com/en/users/elvetia.mascol)**
\|
31 Dec 2025 at 22:22

can we switch brokers or we have to [get a vps](https://www.mql5.com/en/vps "Forex VPS for MetaTrader 4/5") for each broker? can i put the vps on another broker?


![Learn how to deal with date and time in MQL5](https://c.mql5.com/2/59/date_and_time_in_MQL5_logo__1.png)[Learn how to deal with date and time in MQL5](https://www.mql5.com/en/articles/13466)

A new article about a new important topic which is dealing with date and time. As traders or programmers of trading tools, it is very crucial to understand how to deal with these two aspects date and time very well and effectively. So, I will share some important information about how we can deal with date and time to create effective trading tools smoothly and simply without any complicity as much as I can.

![Mastering ONNX: The Game-Changer for MQL5 Traders](https://c.mql5.com/2/59/Mastering_ONNX_logo_up.png)[Mastering ONNX: The Game-Changer for MQL5 Traders](https://www.mql5.com/en/articles/13394)

Dive into the world of ONNX, the powerful open-standard format for exchanging machine learning models. Discover how leveraging ONNX can revolutionize algorithmic trading in MQL5, allowing traders to seamlessly integrate cutting-edge AI models and elevate their strategies to new heights. Uncover the secrets to cross-platform compatibility and learn how to unlock the full potential of ONNX in your MQL5 trading endeavors. Elevate your trading game with this comprehensive guide to Mastering ONNX

![Integrate Your Own LLM into EA (Part 1): Hardware and Environment Deployment](https://c.mql5.com/2/59/Hardware_icon_up__1.png)[Integrate Your Own LLM into EA (Part 1): Hardware and Environment Deployment](https://www.mql5.com/en/articles/13495)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

![StringFormat(). Review and ready-made examples](https://c.mql5.com/2/56/stringformatzj-avatar.png)[StringFormat(). Review and ready-made examples](https://www.mql5.com/en/articles/12953)

The article continues the review of the PrintFormat() function. We will briefly look at formatting strings using StringFormat() and their further use in the program. We will also write templates to display symbol data in the terminal journal. The article will be useful for both beginners and experienced developers.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hsxzpqryalnjbluijebbrorsjqxlfttg&ssn=1769185397232039428&ssn_dr=0&ssn_sr=0&fv_date=1769185397&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13586&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Launching%20MetaTrader%20VPS%3A%20A%20step-by-step%20guide%20for%20first-time%20users%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918539755867575&fz_uniq=5070244966707434100&sv=2552)

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