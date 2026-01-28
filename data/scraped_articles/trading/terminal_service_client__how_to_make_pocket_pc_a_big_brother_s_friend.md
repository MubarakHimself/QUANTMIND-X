---
title: Terminal Service Client. How to Make Pocket PC a Big Brother's Friend
url: https://www.mql5.com/en/articles/1458
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:41:52.583718
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/1458&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083047735935898607)

MetaTrader 4 / Trading


### Introduction

There are some special programs that allow one to control his or her PC using a
PDA. It can be controlled both via internet or via Wi-Fi, in a local network. If
you are a system administrator of a local network, you probably often have to come
to your workplace in order to make some settings. This running about can be avoided,
whereas no third-party programs should be used, because the **Terminal Service Client** is embedded in your PDA's Windows.

In this article, I will tell you how to use that. I take the Wi-Fi network as an
example since Bluetooth works on a small distance which is irrelevant. I also presume
that you have already set up your Wi-Fi network.

Everything described below works well under Windows 2000 and Windows XP Professional.

### Description

To be able to connect to the desktop PC, there are some settings needed. Let's start
with the desktop PC. Click with the right mouse button on icon "My Computer",
select "Properties" and, in the "Remote" tab, check "Allow
users to connect remotely to this computer".

![](https://c.mql5.com/2/15/img-01.jpg)

Go to "Control Panel" and select "User Accounts".

![](https://c.mql5.com/2/15/img-02.jpg)

Create an additional user account and name it, for example, Lan. It is also necessary
to create a password for this new account. Go to checking IP addresses. Click with
the right mouse button on the "My Network Places" and select Properties.

![](https://c.mql5.com/2/15/img-03.jpg)

Find "Wireless Network Connection" and click with the right mouse button
on it selecting "Properties".

![](https://c.mql5.com/2/15/img-04.jpg)

Check and remember the properties of TCP/IP.

![](https://c.mql5.com/2/15/img-05.jpg)

Here we see that the IP address is 192.168.0.1. This is what we need now. So we
won't work with PC anymore and go to our PDA. First of all, enable Wi-Fi for the
network to start functioning. Then find the Terminal Service Client program in
the "Programs" folder and launch it.

![](https://c.mql5.com/2/15/img-06.jpg)

In the "Server" field enter the IP address specified in the TCP/IP settings.
In our case, it is 192.168.0.1. If it is necessary to limit the screen size, check
"Limit size of server desktop to fit on this screen". Otherwise, the
PC screen will be large in the PDA and it will be necessary to scroll it for viewing.
In the PDA, there appear a window proposing you to enter your login and password:

![](https://c.mql5.com/2/15/img-07.jpg)

It is necessary to enter login Lan that we have created before, as well as its password. Press ОК and get the
warning about that some user has already entered the system.

![](https://c.mql5.com/2/15/img-08.jpg)

Press "Yes" undoubtedly and receive a warning in our PC that somebody
wants to enter the system.

![](https://c.mql5.com/2/15/img-09.jpg)

You can avoid such warnings if you have not logged in before. However, press "Yes".
The Big Brother opens the window below and waits for your choice (you should not
choose anything during the session),

![](https://c.mql5.com/2/15/img-10.jpg)

whereas the PDA starts displaying everything that happens in the Big Brother.

![](https://c.mql5.com/2/15/img-11.jpg)

Rules or licenses should not be violated, so those who use licensed Windows XP Professional
should not better use the same copy of the system on several machines simultaneously.
However, the system allows one to do it:

1. Add the key to the register:

_\[HKEY\_LOCAL\_MACHINE\\SYSTEM\\ControlSet001\\Control\\Terminal Server\\Licensing Core\]_
_"EnableConcurrentSessions"=dword:00000001_
2. Restart in a security mode (F8 key being pressed). This is necessary to enable all
the following actions.
3. Then we will need the file named termserv.dll from the beta version of Service Pack
2 for Windows XP (SP2 build 2055). This file should be copied into the folders:
C:\\WINDOWS\\system32 and C:\\WINDOWS\\ServicePackFiles\\i386 (don't forget to save
a copy of your version of termserv.dll, just in case).
4. Reload the machine, create one more user and connect from another machine. Only
users logged in under different names may work simultaneously. So, if you have
only one login, you will have to create another one.


### Conclusion

We have learned how to control the PC with the installed MetaTrader 4 Client Terminal
via PDA.

Translated from Russian by MetaQuotes Software Corp.

Original article: [/ru/articles/1458](https://www.mql5.com/ru/articles/1458)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1458](https://www.mql5.com/ru/articles/1458)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1458.zip "Download all attachments in the single ZIP archive")

[termserv.zip](https://www.mql5.com/en/articles/download/1458/termserv.zip "Download termserv.zip")(207.29 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Advanced Analysis of a Trading Account](https://www.mql5.com/en/articles/1383)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39386)**
(1)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
18 Feb 2008 at 20:32

We offer MetaTrader Hosting to our clients FREE of charge; they recieve a dedictated URL that allows them to login using RDP (Windows Remote Desktop Connection); and then they get to their own dedicated Windows XP Desktop.

We host MT and perform technical support when needed (plus custom development at no cost) > [www.GallantFX.com](https://www.mql5.com/go?link=http://gallantfx.com/ "Gallant FX") (and check out [www.MetaTraderHosting.com](https://www.mql5.com/go?link=http://www.metatraderhosting.com/ "Meta Trader Hosting"))


![MQL4 Language for Newbies. Custom Indicators (Part 1)](https://c.mql5.com/2/15/516_15.gif)[MQL4 Language for Newbies. Custom Indicators (Part 1)](https://www.mql5.com/en/articles/1500)

This is the fourth article from the series "MQL4 Languages for Newbies". Today we will learn to write custom indicators. We will get acquainted with the classification of indicator features, will see how these features influence the indicator, will learn about new functions and optimization, and, finally, we will write our own indicators. Moreover, at the end of the article you will find advice on the programming style. If this is the first article "for newbies" that you are reading, perhaps it would be better for you to read the previous ones. Besides, make sure that you have understood properly the previous material, because the given article does not explain the basics.

![Object Approach in MQL](https://c.mql5.com/2/15/499_6.gif)[Object Approach in MQL](https://www.mql5.com/en/articles/1499)

This article will be interesting first of all for programmers both beginners and professionals working in MQL environment. Also it would be useful if this article were read by MQL environment developers and ideologists, because questions that are analyzed here may become projects for future implementation of MetaTrader and MQL.

![Indicator Taichi - a Simple Idea of Formalizing the Values of Ichimoku Kinko Hyo](https://c.mql5.com/2/15/509_25.gif)[Indicator Taichi - a Simple Idea of Formalizing the Values of Ichimoku Kinko Hyo](https://www.mql5.com/en/articles/1501)

Hard to interpret Ichimoku signals? This article introduces some principles of formalizing values and signals of Ichimoku Kinko Hyo. For visualization of its usage the author chose the currency pair EURUSD based on his own preferences. However, the indicator can be used on any currency pair.

![Trading Strategy Based on Pivot Points Analysis](https://c.mql5.com/2/14/332_13.png)[Trading Strategy Based on Pivot Points Analysis](https://www.mql5.com/en/articles/1465)

Pivot Points (PP) analysis is one of the simplest and most effective strategies for high intraday volatility markets. It was used as early as in the precomputer times, when traders working at stocks could not use any ADP equipment, except for counting frames and arithmometers.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/1458&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083047735935898607)

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