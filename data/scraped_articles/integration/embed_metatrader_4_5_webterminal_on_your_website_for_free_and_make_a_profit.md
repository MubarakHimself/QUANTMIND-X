---
title: Embed MetaTrader 4/5 WebTerminal on your website for free and make a profit
url: https://www.mql5.com/en/articles/3024
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:18:09.513688
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/3024&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071731712586624352)

MetaTrader 5 / Integration


Traders are well familiar with the [WebTerminal](https://www.mql5.com/en/trading), which allows trading on financial markets straight from the browser.

Add the WebTerminal widget to your website — you can do it absolutely free. This powerful functionality will enable your site visitors to trade using the most popular MetaTrader 5 and MetaTrader 4 platforms straight from your website!

![](https://c.mql5.com/2/26/screenshot__2.png)

### How to profit from it

In addition, the web terminal allows you to sell leads, i.e. potential clients to brokers. All accounts opened through your web terminal will be marked through account parameters, including 'Lead source' and 'Comment', which are only available to the broker.

- The Comment field will contain "WebTerminal \[short name of the domain from which the account was opened\]". Example: "WebTerminal mql5.com". The "www" part is removed from the address.
- The short domain name without 'www' is also added in the 'Lead source'‌ field. Example: "mql5.com". The value can be overridden by adding a [utm tag](https://www.mql5.com/en/articles/3024#utm) to the widget options.

So the broker will be able to track registrations performed through your site. You will be paid for each lead according to your agreement with the broker.

Owners of popular web resources can earn additional money by advertising for brokerage companies. For example, upon agreement, you can configure your web terminal so that it displays a particular broker's servers selected by default, and profit from it.

We do not charge any fees for the use of the WebTerminal on your sites.

### How does the WebTerminal work?

The WebTerminal is a modern HTML5 application that can be easily integrated into any website — you only need to add a simple iframe widget. It works in all operating systems and browsers, and does not require additional software.

The WebTerminal operation is provided through a geographically distributed network of servers that ensures the best connection conditions and robustness.

The site on which the widget is located does not have access to the web terminal data, including information entered during login and demo account opening. Users' data are securely protected and cannot be accessed by the owner of the site on which the web terminal is used.

### How to add the WebTerminal to your site

Insert below HTML code in order to add the WebTerminal to your site:

<iframe src="https://metatraderweb.app/trade?demo\_all\_servers=1&amp;startup\_mode=open\_demo&amp;lang=ru&amp;save\_password=off" allowfullscreen="allowfullscreen" style="width: 100%; height: 100%; border: none;"></iframe>

Servers of all brokers supporting the WebTerminal use are available to traders by default. It is defined by the demo\_all\_servers=1 parameter. If you want to limit the list of available servers, remove this parameter and add 'servers' instead. Specify in it a list of servers separated by commas. If you need to set a default server selected in the demo account opening and connection dialog, add the 'trade\_server' parameter.

<iframe src="https://metatraderweb.app/trade?servers=SomeBroker1-Demo,SomeBroker1-Live,SomeBroker2-Demo,SomeBroker2-Live&amp;trade\_server=SomeBroker-Demo&amp;startup\_mode=open\_demo&amp;lang=ru&amp;save\_password=off"allowfullscreen="allowfullscreen" style="width: 100%; height: 100%; border: none;"></iframe>

All demo accounts opened through your web terminal will be marked to let the broker know that the trader has come from your website. This information is added to the 'Comment' and 'Lead Source' parameters of an account. These parameters are only available to the broker. They contain a short domain name of the website (without "www") on which the WebTerminal is available. For example, if the widget is added to the www.mysite.com site, the following values will be specified in the account:

- Comment: WebTerminal mysite.com
- Lead Source: mysite.com

A value added to 'Lead Source' can be overridden by adding the utm\_campaign parameter to the terminal widget:

<iframe src="https://metatraderweb.app/trade?demo\_all\_servers=1&amp;startup\_mode=open\_demo&amp;lang=ru&amp;save\_password=off&amp;utm\_campaign=campaign\_name" allowfullscreen="allowfullscreen" style="width: 100%; height: 100%; border: none;"></iframe>

When we add the above widget to www.mysite.com, the account parameters will be filled as follows:

- Comment: WebTerminal mysite.com
- Lead Source: campaign\_name

The widget size is specified with standard CSS styles: style="width: 100%; height: 100%;". The recommended height and width values are 100% to allow the web terminal to automatically adjust to the maximum available web page space.

The WebTerminal supports operation in the full screen mode (menu View - Full screen), allowing users to comfortably use all of the available functions. The full screen attribute allowfullscreen="allowfullscreen" is already added to the example. You can delete it if you want to disable the full screen mode.

The WebTerminal interface is available in 41 languages, they can be switched using the View menu. If you need to set a default language, use the 'lang' parameter. If you have a multi-lingual site, you can set to use the currently selected language for the widget.

<iframe src="https://metatraderweb.app/trade?demo\_all\_servers=1&amp;startup\_mode=open\_demo&amp;lang=ru&amp;save\_password=off" allowfullscreen="allowfullscreen" style="width: 100%; height: 100%; border: none;"></iframe>

In this example Russian is selected as the default language. Here are all available values:

| ar — Arabic<br> bg — Bulgarian<br> zh — Chinese<br> hr — Croatian<br> cs — Czech<br> da — Danish<br> nl — Dutch<br> en — English | et — Estonian<br> fi — Finnish<br> fr — French<br> de — German<br> el — Greek<br> he — Hebrew<br> hi — Hindi<br> hu — Hungarian | id — Indonesian<br> it — Italian<br> ja — Japanese<br> ko — Korean<br> lv — Latvian<br> lt — Lithuanian<br> ms — Malay<br> mn — Mongolian | fa — Persian<br> pl — Polish<br> pt — Portuguese<br> ro — Romanian<br> ru — Russian<br> sr — Serbian<br> sk — Slovak<br> sl — Slovenian | es — Spanish<br> sv — Swedish<br> tg — Tajik<br> th — Thai<br> zt — Traditional Chinese<br> tr — Turkish<br> uk — Ukrainian<br> uz — Uzbek<br> vi — Vietnamese |
| --- | --- | --- | --- | --- |

The 'startup\_mode' parameter is responsible for terminal launch. It can have one of the following values:

- open\_demo — set this value to display a demo account opening window (instead of the login window) for the users who do not have accounts stored in the web terminal. If saved accounts are available in the local browser storage, connection to the last used account is established.
- no\_autologin — users can save their passwords in the browser storage in the account connection dialog. In this case, it will be possible for the user to connect to the account without entering the password further on. If a password of the last used account is saved, the web terminal automatically connects to it during the next launch. In order to disable auto connection, set startup\_mode=no\_autologin.


The web terminal also allows saving passwords of trading accounts in the user's **web browser storage** (the website does not save or store any information!), enabling users to automatically connect to accounts without entering a password. The appropriate option 'Save password' is available in the login dialog. The password saving option can be disabled by default by adding the save\_password=off parameter.

<iframe src="https://metatraderweb.app/trade?demo\_all\_servers=1&amp;startup\_mode=open\_demo&amp;lang=ru&amp;save\_password=off" allowfullscreen="allowfullscreen" style="width: 100%; height: 100%; border: none;"></iframe>

The web terminal supports additional customization parameters:

- startup\_version — the default web terminal version: 4 for MetaTrader 4 or 5 for MetaTrader 5. The parameter is used for the first launch of the web terminal. Further, the platform version will be defined based on the last used account.

- login — the trading account login. It can be used for creating personal profiles. If your website keeps information about account number, you can dynamically form a widget and add the desired account into it. It means that a user will only need to enter the password, while the account number will be inserted automatically.
- demo\_show\_phone — set this parameter to 1 (demo\_show\_phone=1) and add it to the widget in order to display the phone field in the demo account registration form. If the parameter is not specified or it is set to a value other than 1, the phone field is not displayed. By removing the phone field from the registration form, you can increase site conversion.


The WebTerminal supports the following web browser versions and above:

- Internet Explorer 11
- Microsoft Edge 12
- Mozilla Firefox 34
- Google Chrome 43
- Safari 8
- Opera 32

### One terminal for MetaTrader 5 and MetaTrader 4

One web terminal is used for the two versions of the platform. If a default server is specified in the 'trade\_server' parameter of the widget, the web terminal will automatically switch to the required platform version based on the server name.

If you are using two platform versions (for example you set demo\_all\_servers=1), a switch between the two versions appears in the web terminal interface. It is available in the account connection dialog, in the account opening dialog, and in the File menu.

![](https://c.mql5.com/2/26/screenshot_2__2.png)

You can set a default version that will be selected during WebTerminal launch by adding the 'startup\_version' parameter to the widget code:

<iframe src="https://metatraderweb.app/trade?demo\_all\_servers=1&amp;startup\_mode=open\_demo&amp;startup\_version=5&amp;lang=ru&amp;save\_password=off" allowfullscreen="allowfullscreen" style="width: 100%; height: 100%; border: none"></iframe>

In this example, the versions switch is set to MetaTrader 5 by default.

If a user switches to another platform, the selection will be remembered. During the next launch of the web terminal, it will be switched to the latest used version of the platform.

### A ready example of the HTML page with the WebTerminal

Try to launch the WebTerminal now. Save the following code in the HTML file and then open it in the browser.

<!DOCTYPE html>

<html>

<head>

<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

<title>WebTerminal for the MetaTrader 4 and MetaTrader 5 platforms</title>

<style type="text/css">

body {margin: 0; padding: 0; font-family: Arial, Tahoma; font-size: 16px; color: #000; background-color: #FFF; min-width: 1010px; }

.top {background-color: #0055A7; }

.top h1 {margin: 10px 20px 10px 10px; font-size: 25px; font-weight: normal; color: #FFF; display: inline-block; vertical-align: middle; }

.top .menu, .top .menu li {margin: 0; padding: 0; list-style: none; display: inline-block; vertical-align: middle; }

.top .menu li {margin: 0; padding: 0; list-style: none; display: inline-block; }

.top .menu li a {padding: 20px; font-size: 16px; color: #FFF; text-decoration: none; text-align: center; display: block; }

.top .menu li a:hover {background-color: #0B6ABF; }

.top .menu li a.selected {background-color: #2989DF; color: #FFF; }

.content { box-shadow: 0 0 20px rgba(0,0,0,0.5); position: fixed; width: 100%; top: 60px; bottom: 60px; }

.footer {text-align: center; padding: 20px; color: #0A0A0A; font-size: 14px; position: fixed; bottom: 0; width: 100%; }

</style>

</head>

<body>

<div class="top">

<h1>Company name</h1>

<ul class="menu">

<li><a href="#">Analytics</a></li>

<li><a href="#" class="selected">WebTerminal</a></li>

<li><a href="#">News</a></li>

<li><a href="#">Contacts</a></li>

</ul>

</div>

<div class="content">

<!\-\- Web Terminal Code Start -->

<iframe src="https://metatraderweb.app/trade?demo\_all\_servers=1&amp;startup\_mode=open\_demo&amp;lang=ru&amp;save\_password=off" allowfullscreen="allowfullscreen" style="width: 100%; height: 100%; border: none;"></iframe>

<!\-\- Web Terminal Code End -->

</div>

<div class="footer"> Copyright 2000-2017, Company name</div>

</body>

</html>

### We're almost there. You can embed the WebTerminal now

As you can see, integration of the WebTerminal with any site requires minimum effort, while our developers have prepared everything needed. If you have a website, you can start selling leads to brokers — we have a ready-to-use web-based solution for you. All you need to do is embed one iframe into your website.

Provide your site visitors with new possibilities and earn additional income.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/3024](https://www.mql5.com/ru/articles/3024)

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
**[Go to discussion](https://www.mql5.com/en/forum/170187)**
(95)


![Rene Taborete Repunte](https://c.mql5.com/avatar/2021/11/61A2FDB6-8563.gif)

**[Rene Taborete Repunte](https://www.mql5.com/en/users/kermanz)**
\|
23 Mar 2022 at 08:38

**PUBG Mobile [#](https://www.mql5.com/en/forum/170187/page6#comment_28391340):**

it does exist but they changed the code to "display: none" and added a height property to 0. So that basicly means you can put it in an iframe but you wont see anything pop up. Really unprofessional if you ask me

thanks sir hope it will fix and continue this beautiful idea.

![LONNV](https://c.mql5.com/avatar/avatar_na2.png)

**[LONNV](https://www.mql5.com/en/users/lonnv)**
\|
8 Apr 2022 at 20:47

how can create for [real account](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 documentation: Account Properties") server ?

only demo avaliable?

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
8 Apr 2022 at 21:56

**LONNV [#](https://www.mql5.com/en/forum/170187/page7#comment_28823789):** how can create for [real account](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 documentation: Account Properties") server ? only demo avaliable?

Use the _MetaTrader_ web terminal provided by your broker instead.

![LONNV](https://c.mql5.com/avatar/avatar_na2.png)

**[LONNV](https://www.mql5.com/en/users/lonnv)**
\|
9 Apr 2022 at 12:23

**Fernando Carreiro [#](https://www.mql5.com/en/forum/170187/page7#comment_28825361):**

Use the _MetaTrader_ web terminal provided by your broker instead.

what you mean ?

how edit here

```
 <iframe src="https://trade.mql5.com/trade?demo_all_servers=1&amp;amp;startup_mode=open_demo&amp;lang=en&amp;save_password=off" allowfullscreen="allowfullscreen" style="width: 100%; height: 100%; border: none;"></iframe>
```

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
24 Aug 2025 at 12:59

**[@LONNV](https://www.mql5.com/en/users/lonnv) [#](https://www.mql5.com/en/forum/170187/page10#comment_28831434):** what you mean ? how edit here

The _webterminal_ on this website, namely "trade.mql5.com", is only for _MetaQuotes_ demo accounts, not for real broker accounts.

Each broker has their own dedicated _webterminal_ with their own URL. Visit your broker's website to find out what their webterminal URL is, or contact their support for more details.

![3D Modeling in MQL5](https://c.mql5.com/2/25/3d-avatar.png)[3D Modeling in MQL5](https://www.mql5.com/en/articles/2828)

A time series is a dynamic system, in which values of a random variable are received continuously or at successive equally spaced points in time. Transition from 2D to 3D market analysis provides a new look at complex processes and research objects. The article describes visualization methods providing 3D representation of two-dimensional data.

![Auto detection of extreme points based on a specified price variation](https://c.mql5.com/2/25/math_compass.png)[Auto detection of extreme points based on a specified price variation](https://www.mql5.com/en/articles/2817)

Automation of trading strategies involving graphical patterns requires the ability to search for extreme points on the charts for further processing and interpretation. Existing tools do not always provide such an ability. The algorithms described in the article allow finding all extreme points on charts. The tools discussed here are equally efficient both during trends and flat movements. The obtained results are not strongly affected by a selected timeframe and are only defined by a specified scale.

![ZUP - universal ZigZag with Pesavento patterns. Graphical interface](https://c.mql5.com/2/26/MQL5-avatar-ZUP-001.png)[ZUP - universal ZigZag with Pesavento patterns. Graphical interface](https://www.mql5.com/en/articles/2966)

Over the ten years since the release of the first version of the ZUP platform, it has undergone through multiple changes and improvements. As a result, now we have a unique graphical add-on for MetaTrader 4 allowing you to quickly and conveniently analyze market data. The article describes how to work with the graphical interface of the ZUP indicator platform.

![Graphical interfaces X: Advanced management of lists and tables. Code optimization (build 7)](https://c.mql5.com/2/25/Graphic-interface_11-2.png)[Graphical interfaces X: Advanced management of lists and tables. Code optimization (build 7)](https://www.mql5.com/en/articles/2943)

The library code needs to be optimized: it should be more regularized, which is — more readable and comprehensible for studying. In addition, we will continue to develop the controls created previously: lists, tables and scrollbars.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/3024&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071731712586624352)

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