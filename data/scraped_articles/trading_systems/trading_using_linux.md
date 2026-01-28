---
title: Trading Using Linux
url: https://www.mql5.com/en/articles/1438
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:57:28.071928
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/1438&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083238797556062102)

MetaTrader 4 / Trading systems


### Introduction

Trading in financial markets is not principal earner for all users of the online
trading platform MetaTrader 4. It is not always convenient to keep the trading
terminal open all the time, and sound alerts require either continuous keeping
loudspeakers on (what can provide problems if you have, for example, little children)
or wearing headphones all the time which is not always comfortable or healthy.
The way out may be provided by operating system Linux.

The largest provider of reliable information about situation on financial markets
is search engine [Yahoo](https://www.mql5.com/go?link=https://www.yahoo.com/ "/go?link=https://www.yahoo.com/"). Programs described below in this article use information provided by this specific
service. The author analyzed the list of indicators working under linux. The present
article is the result of this analysis.

### List of Indicators

**1.** Any Internet browser capable of displaying graphical objects can be used as indicator
of exchange activity. Service [Yahoo! Finance](https://www.mql5.com/go?link=https://finance.yahoo.com/ "/go?link=https://finance.yahoo.com/") provides comprehensive information about situation in stock exchange, security
markets, banks. It is possible to view online indexes of Dow Jones, Nasdaq, and S&P 500. Other indexes of economic activity can also be found [here](https://ru.wikipedia.org/wiki/%D0%9F%D1%80%D0%BE%D0%BC%D1%8B%D1%88%D0%BB%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81_%D0%94%D0%BE%D1%83-%D0%94%D0%B6%D0%BE%D0%BD%D1%81 "https://ru.wikipedia.org/wiki/%D0%9F%D1%80%D0%BE%D0%BC%D1%8B%D1%88%D0%BB%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81_%D0%94%D0%BE%D1%83-%D0%94%D0%B6%D0%BE%D0%BD%D1%81"). To get to know of the symbol of a company included in the index lists, one should
use the corresponding services at [Yahoo! Finance](https://www.mql5.com/go?link=https://finance.yahoo.com/lookup "/go?link=https://finance.yahoo.com/lookup"). Using these indexes and symbols of companies, one can monitor practically every
area of world economy using a common browser.

Using [this link](https://www.mql5.com/go?link=http://screen.yahoo.com/stocks.html "https://www.mql5.com/go?link=http://screen.yahoo.com/stocks.html"), a trader can find necessary source information about a company or an industry.
For example, below is information about the company producing printer Lexmark.

[https://www.mql5.com/go?link=http://screen.yahoo.com/b?sc=815](https://www.mql5.com/go?link=http://screen.yahoo.com/b?sc=815 "https://www.mql5.com/go?link=http://screen.yahoo.com/b?sc=815")

![](https://c.mql5.com/2/15/s_6.png)

**Screenshot 1**

The author would like to draw readers' attention to the fact that search in services
of Yahoo! Finance is performed by symbols (suffixes) given to companies. The stock
suffixes of companies can be found [here](https://www.mql5.com/go?link=http://screen.yahoo.com/stocks.html).

![](https://c.mql5.com/2/15/s_7.png)

**Screenshot 2**

**2.** The program named GkrellStock is used to view quotes. The program is an external
plugin to be connected to another program [GkrellM](https://www.mql5.com/go?link=http://www.gkrellm.net/ "https://www.mql5.com/go?link=http://www.gkrellm.net/") widely used in the world of "open source software".

It is shown below how [GkrellM](https://www.mql5.com/go?link=http://www.gkrellm.net/ "https://www.mql5.com/go?link=http://www.gkrellm.net/") will look after installation and setup of plugin GkrellStock:

![](https://c.mql5.com/2/15/13s_1_3.png)

**Screenshot 3**

In the program settings, it is possible to select location of the server providing
information, stock indexes/suffixes alternating, updating time. The server providing
information and located in the USA is shown in the screenshot below, updating time
selected as 30 seconds and 5 minutes, respectively.

![](https://c.mql5.com/2/15/16s_1_2.png)

**Screenshot 4**

Stock index or suffix is shown in the toolbar of plugin GkrellStock. In the screenshot
below, it can be seen how things are going with General Electric Co. If the mouse
cursor is pointed at its toolbar, information about the company's quotes will be
displayed in a pop-up window.

![](https://c.mql5.com/2/15/14s_1_2.png)

**Screenshot 5**

The screenshot below shows the situation with DowJones index. Like in the previous
case, more details are given in a pop-up window.

![](https://c.mql5.com/2/15/15s_1_3.png)

**Screenshot 6**

To use this plugin, graphical server X-Window library [gtk2](https://www.mql5.com/go?link=https://www.gtk.org/ "/go?link=https://www.gtk.org/"), [GkrellM](https://www.mql5.com/go?link=http://www.gkrellm.net/ "https://www.mql5.com/go?link=http://www.gkrellm.net/"), and the plugin itself must be installed. The plugin can be found at [http://gkrellstock.sourceforge.net/](https://www.mql5.com/go?link=http://gkrellstock.sourceforge.net/ "https://www.mql5.com/go?link=http://gkrellstock.sourceforge.net/"). If the user does not want to compile, the ready rpm-package can be utilized that
can be found [here](https://www.mql5.com/go?ftp://fr.rpmfind.net/linux/mandrake/9.1/contrib/i586/gkrellm-plugins-2.1.7a-2mdk.i586.rpm "https://www.mql5.com/go?ftp://fr.rpmfind.net/linux/mandrake/9.1/contrib/i586/gkrellm-plugins-2.1.7a-2mdk.i586.rpm") or another source may be found using [Google](https://www.mql5.com/go?link=https://www.google.com.ua/ "/go?link=https://www.google.com.ua/") search engine.

**_For Linux-Users Utilizing Distributives Other Than RPM_**

_The author reckons, first of all, Slackware among non-rpm distributives. For those_
_who uses distributives built like Slackware or based on it, the author would recommend_
_to use utilities rpm2targz or rpm2tgz included into Slackware standard delivery._
_Applying these utilities, one can create a package in this distributive native_
_format, which is necessary to facilitate live updates of packages and to maintain_
_the file arrangement in the system._

**3.** Gtik is a program that is also a plugin, but this plugin is used for window manager,
[Gnome](https://www.mql5.com/go?link=https://www.gnome.org/ "/go?link=https://www.gnome.org/"), embedded in the toolbar. Plugin gnome-applets-gtik-2. 14.2 is in the package of
gnome-applets that can be downloaded [from here](https://www.mql5.com/go?link=http://www.sisyphus.ru/srpm/gnome-applets/get "https://www.mql5.com/go?link=http://www.sisyphus.ru/srpm/gnome-applets/get"). After installation

![](https://c.mql5.com/2/15/13s_1_4.png)

**Screenshot 7**

![](https://c.mql5.com/2/15/14s_1_3.png)

**Screenshot 8**

and setup

![](https://c.mql5.com/2/15/15s_1_4.png)

**Screenshot 9**

the user gets one more tool to watch the market situation.

![](https://c.mql5.com/2/15/16s_1_3.png)

**Screenshot 10**

The sequence of searching for stock indexes and suffixes described above is used
here, as well. To apply this plugin, it is necessary to have installed graphical
server X-Window, window manager [Gnome](https://www.mql5.com/go?link=https://www.gnome.org/ "/go?link=https://www.gnome.org/") and gtik.

**4.** Program Wmjstock is a plugin, too. It differs from those described above through
the fact that it can work independently. Experienced Linux-users will immediately
see by the name that this module belongs to plugins developed for window managers.
This is easily to detect from letters wm the name begins with. These are so-called
dockapps (dockapplets). Their list will impress even a WinXP-user. The above-mentioned
plugin can be taken [here](https://www.mql5.com/go?link=http://dockapps.org/ "https://www.mql5.com/go?link=http://dockapps.org/") or [by this, more exact link](https://www.mql5.com/go?link=http://dockapps.org/file.php/id/69 "https://www.mql5.com/go?link=http://dockapps.org/file.php/id/69"). One can get lost among the plentiful existing dockapplets.

After plugin installation and setup, the user will obtain the following result:

![](https://c.mql5.com/2/15/17s_1_4.png)

**Screenshot 11**

Like in all previous programs, the order of search for stock indexes or suffixes
does not change. The program allows one to change colors in this dockapplet. After
the program has been modified, it must be recompiled.

![](https://c.mql5.com/2/15/18s_1_1.png)

**Screenshot 12**

To use this plugin, it is necessary to install graphical server X-Window, any window
manager, and the program itself.

**5.** Tclticker is not a plugin or dockapplet, it is a fully independent application.
It can be downloaded at [Tom Poindexter's Tcl Page](https://www.mql5.com/go?link=http://www.nyx.net/~tpoindex/tcl.html "https://www.mql5.com/go?link=http://www.nyx.net/~tpoindex/tcl.html"). Its latest version is Tclticker-1.3. To the author's opinion, this program must
be one of the most useful ones in trader's work. It allows traders to get information
about the stock news and situations as a stock ticker.

![](https://c.mql5.com/2/15/screenshot.png)

**Screenshot 13**

In this program, there is a possibility to add necessary stock indexes and suffixes,
as well as to use a proxy server. To find stock indexes and suffixes, the user
may apply methods describe above in the article.

![](https://c.mql5.com/2/15/s_8.png)

**Screenshot 14**

To install this program, X-server is necessary. The packages of tcl and tk can
be taken from [here](https://www.mql5.com/go?link=http://www.tcl.tk/software/tcltk/ "https://www.mql5.com/go?link=http://www.tcl.tk/software/tcltk/").

**Conclusion**

The author himself uses in his trading practice the last program from the above
list - Tclticker. It can be placed in the upper part of the desktop as shown in
Screenshot 13. This allows the author to be always in touch with the situation
on the market, this program setup flexibility providing the point sampling of market
positions the trader is interested in. This allows the user to predict the Forex
development. And this, in its turn, allows to fit the Expert Advisors for automated
trading using MetaTrader 4 Client Terminal.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1438](https://www.mql5.com/ru/articles/1438)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Running MetaTrader 4 Client Terminal on Linux-Desktop](https://www.mql5.com/en/articles/1433)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39366)**
(1)


![808](https://c.mql5.com/avatar/2016/3/56E98414-0E84.png)

**[808](https://www.mql5.com/en/users/8bit.system)**
\|
8 Nov 2015 at 16:31

Or you could use Wine :) It's working very well with MT4.


![Break Through The Strategy Tester Limit On Testing Hedge EA](https://c.mql5.com/2/14/445_33.gif)[Break Through The Strategy Tester Limit On Testing Hedge EA](https://www.mql5.com/en/articles/1493)

An idea of testing the hedge Expert Advisors using the strategy tester.

![Filtering by History](https://c.mql5.com/2/14/244_1.png)[Filtering by History](https://www.mql5.com/en/articles/1441)

The article describes the usage of virtual trading as an integral part of trade opening filter.

![Practical Application of Cluster Indicators in FOREX](https://c.mql5.com/2/14/352_5.gif)[Practical Application of Cluster Indicators in FOREX](https://www.mql5.com/en/articles/1472)

Cluster indicators are sets of indicators that divide currency pairs into separate currencies. Indicators allow to trace the relative currency fluctuation, determine the potential of forming new currency trends, receive trade signals and follow medium-term and long-term positions.

![How Not to Fall into Optimization Traps?](https://c.mql5.com/2/14/218_2.png)[How Not to Fall into Optimization Traps?](https://www.mql5.com/en/articles/1434)

The article describes the methods of how to understand the tester optimization results better. It also gives some tips that help to avoid "harmful optimization".

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=knlmeyxxzauarxnurolhffblvimvejzn&ssn=1769252247125577176&ssn_dr=0&ssn_sr=0&fv_date=1769252247&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1438&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20Using%20Linux%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925224720539975&fz_uniq=5083238797556062102&sv=2552)

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