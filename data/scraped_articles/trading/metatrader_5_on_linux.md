---
title: MetaTrader 5 on Linux
url: https://www.mql5.com/en/articles/625
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:21:27.857840
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/625&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069371052531844086)

MetaTrader 5 / Trading


In this article, we demonstrate how to install MetaTrader 5 on popular Linux versions, [Ubuntu](https://ru.wikipedia.org/wiki/Ubuntu "Узнать больше о Ubuntu"), [Debian](https://ru.wikipedia.org/wiki/Debian " Узнать больше о Debian"), [Linux Mint](https://www.mql5.com/go?link=https://linuxmint.com/ "https://linuxmint.com/") and [Fedora](https://www.mql5.com/go?link=https://fedoraproject.org/ "https://fedoraproject.org/"). These systems are widely used on companies’ server hardware as well as on traders’ personal computers.

### Installing the platform with one command

MetaTrader 5 runs on Linux using [Wine](https://www.mql5.com/go?link=https://www.winehq.org/ "Official Wine website"). Wine is a free compatibility layer that allows application software developed for Microsoft Windows to run on Unix-like operating systems.

We have prepared a special script to make the installation process as simple as possible. The script will automatically detect your system version, it supports Ubuntu, Debian, Linux Mint and Fedora distributions. Based on it, it will download and install the appropriate Wine package. After that, it will download and run the platform installer.

To start the installation, open the command line (Terminal) without the administrator privileges (no sudo) and specify the relevant command:

wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5linux.sh ; chmod +x mt5linux.sh ; ./mt5linux.sh

This command downloads the script, makes it executable and runs it. You only need to enter your account password to allow installation.

![Installing Wine and MetaTrader 5 with a single command](https://c.mql5.com/2/171/ubuntu-command-line.png)

If you are prompted to install additional Wine packages (Mono, Gecko), please agree, as these packages are required for platform operation. The MetaTrader 5 installer will launch after that, proceed with the standard steps. Once the installation is complete, restart your operating system, and the platform is ready to go.

![The MetaTrader 5 platform is ready to run on Linux](https://c.mql5.com/2/171/metatrader5__1.png)

### Install updates in a timely manner

It is highly recommended to always use the latest versions of the operating system and Wine. Timely updates increase platform operation stability and improve performance.

To update Wine, open a command prompt and type the following command:

sudo apt update ; sudo apt upgrade

For further information, please visit the [official Wine website](https://www.mql5.com/go?link=https://gitlab.winehq.org/wine/wine/-/wikis/Download "https://wiki.winehq.org/Ubuntu").

### MetaTrader 5 Data Directory

Wine creates a separate virtual logical drive with the necessary environment for every installed program. The default path of the installed terminal data folder is as follows:

Home directory\\.mt5\\drive\_c\\Program Files\\MetaTrader 5

Use MetaTrader 5 on Linux: install with a single command and enjoy all the platform features.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/625](https://www.mql5.com/ru/articles/625)

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
**[Go to discussion](https://www.mql5.com/en/forum/10114)**
(327)


![Federico Quintieri](https://c.mql5.com/avatar/2022/1/61EFFA9D-3302.png)

**[Federico Quintieri](https://www.mql5.com/en/users/fede98)**
\|
11 Dec 2025 at 17:28

**Federico Quintieri [#](https://www.mql5.com/it/forum/386006/page32#comment_58584529):**

In fact, I made it work with these steps (I am on CachyOS) and I am new to Linux

1\. Installed Bottles and created a bottle (application)

2\. Downloaded the mt5 setup for windows from the original mql5 site (other versions gave proxy errors during installation).

3\. In the bootle the "runner" is "ge-proton10-25", which I downloaded from the bootle's home (Preferences => Runner).

4\. In the metatrader5 bottle settings I switched to windows 11

5\. The bootle dependencies I downloaded are: dotnet48 - allfonts - vcredist2019 - vcredist2015

6\. Run mt5.exe in the metatrader5 bottle I just created.

7\. Then in the bottle settings you can open a terminal, type "winecfg", in the new window go to graphics and adjust the "dpi" according to how much you want your mt5 to zoom, mine is at 96 (I had the same zoom problem when I tried to install it with lucris, now it works fine).

This is what worked for me, I'm trying and coding different things and it seems to work fine.

If you have hyprland the solution is to use a virtual machine with windows11 installed, I have tried various things but metatrader5 on hyprland just doesn't work.

For the virtual machine I have used, KVM, Qemu and virt-manager

![wy1998](https://c.mql5.com/avatar/avatar_na2.png)

**[wy1998](https://www.mql5.com/en/users/wuyu1998)**
\|
12 Dec 2025 at 18:34

Fault phenomenon: the text on the coordinate axes (horizontal and vertical axes) is missing.

Reason for failure: reboot after upgrading mt5.

Tried and true:

1) Upgrade debian v12 to v13. upgrade failed, reinstall debian v13.

2) Downloaded mt5linux.sh from mql5.com and installed it, the failure is still the same. But at this point, it's a little different.

Just after installation, eur/usd, xau/usd, usd/rmb are able to display the text on the axes.

After importing the custom currency "CSI 300" .json file, loading the interface template, copying the historical data file, and entering mt5, the fault appears.

3) Move ~/.mt5 directory, reinitialise winecfg, re-execute mt5linux.sh. Failure reappears.

[![Missing text on axes](https://c.mql5.com/3/481/5g1g_20251213_021112.png)](https://c.mql5.com/3/481/0asx_20251213_021112.png "https://c.mql5.com/3/481/0asx_20251213_021112.png")

Current working environment:

Debian GNU/Linux 13.2.0 \_Trixie

kde plasma v6.36

kernel 6.12.57+debian13-amd64 (64-bit)

wine64 v10.0~repack-6

![wy1998](https://c.mql5.com/avatar/avatar_na2.png)

**[wy1998](https://www.mql5.com/en/users/wuyu1998)**
\|
15 Dec 2025 at 06:14

**Problem found: windows/Fonts problem.**

Clear files in Fonts, problem solved.

Restore files in Fonts, problem resumed.

![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
15 Dec 2025 at 10:22

**Tobias Johannes Zimmer [#](https://www.mql5.com/en/forum/10114/page32#comment_58692345):**

This Test EA should show four Rectangles in a line when thrown on a EURUSD chart. Only the last it rectangle visible. I also hat problems changing colors and filling of the rectangles. This came to my attention when I was trying to build a little EA with two rectangles and a few buttons. The buttons were not visible either.

Strangely the test Expert " [Controls](https://www.mql5.com/en/articles/310 "Article: Custom Graphic Controls Part 1. Creating a simple control")" shows all buttons beautifully.

In bottles I tried fixing the problem by installing different dependencies, namely directx, since I remember thinking that directx might have something to do with MT5 graphical objects but I am not sure if that is correct.

Your code is buggy. How can you hope to have 4 rectangles all with the same name ?

Also please report here issues which are related SPECIFICALLY to Wine/Linux.

![wy1998](https://c.mql5.com/avatar/avatar_na2.png)

**[wy1998](https://www.mql5.com/en/users/wuyu1998)**
\|
15 Dec 2025 at 12:02

Problem solved.

1) Empty the files in the windows/Fonts directory

2) Copy simsun.ttc into it!

Then, the problem is restored.

![Neural Networks: From Theory to Practice](https://c.mql5.com/2/0/ava_seti.png)[Neural Networks: From Theory to Practice](https://www.mql5.com/en/articles/497)

Nowadays, every trader must have heard of neural networks and knows how cool it is to use them. The majority believes that those who can deal with neural networks are some kind of superhuman. In this article, I will try to explain to you the neural network architecture, describe its applications and show examples of practical use.

![General information on Trading Signals for MetaTrader 4 and MetaTrader 5](https://c.mql5.com/2/0/signal_mt4_mt5__1.png)[General information on Trading Signals for MetaTrader 4 and MetaTrader 5](https://www.mql5.com/en/articles/618)

MetaTrader 4 / MetaTrader 5 Trading Signals is a service allowing traders to copy trading operations of a Signals Provider. Our goal was to develop the new massively used service protecting Subscribers and relieving them of unnecessary costs.

![Order Strategies. Multi-Purpose Expert Advisor](https://c.mql5.com/2/0/conveyor_ava.png)[Order Strategies. Multi-Purpose Expert Advisor](https://www.mql5.com/en/articles/495)

This article centers around strategies that actively use pending orders, a metalanguage that can be created to formally describe such strategies and the use of a multi-purpose Expert Advisor whose operation is based on those descriptions

![Interview with Alexey Masterov (ATC 2012)](https://c.mql5.com/2/0/avatar_reinhardf17.png)[Interview with Alexey Masterov (ATC 2012)](https://www.mql5.com/en/articles/624)

We do our best to introduce all the leading Championship Participants to our audience in reasonable time. To achieve that, we closely monitor the most promising contestants in our TOP-10 and arrange interviews with them. However, the sharp rise of Alexey Masterov (reinhard) up to the third place has become a real surprise!

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=njhvklossdvznokubhxgtffwmslwxbpa&ssn=1769181686942416220&ssn_dr=0&ssn_sr=0&fv_date=1769181686&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F625&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MetaTrader%205%20on%20Linux%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691816866701061&fz_uniq=5069371052531844086&sv=2552)

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