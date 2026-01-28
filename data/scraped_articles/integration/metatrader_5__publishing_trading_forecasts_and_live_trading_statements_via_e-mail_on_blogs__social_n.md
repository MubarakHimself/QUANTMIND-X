---
title: MetaTrader 5: Publishing trading forecasts and live trading statements via e-mail on blogs, social networks and dedicated websites
url: https://www.mql5.com/en/articles/80
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:22:26.029072
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/80&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071784514914561632)

MetaTrader 5 / Integration


### Introduction

Automatic web-publication of trading forecasts has become a widespread trend in the trading industry. Some traders or companies use Internet as a medium for selling subscribed signals, some traders use it for their own blogs to inform about their track record, some do it in order to offer programming or consultancy services. Others publish signals just for fame or fun.

This article aims to present ready-made solutions for publishing forecasts using MetaTrader 5. It covers a range of ideas: from using dedicated websites for publishing MetaTrader statements, through setting up one's own website with virtually no web programming experience needed and finally integration with a social network microblogging service that allows many readers to join and follow the forecasts.

All solutions presented here are 100% free and possible to setup by anyone with a basic knowledge of e-mail and ftp services. There are no obstacles to use the same techniques for professional hosting and commercial trading forecast services.

### 1\. Configuring MetaTrader 5 for E-Mail

**1.1. Step 1: Creating an external e-mail account**

In order to publish an automatic trading forecast from MetaTrader one has to setup an external e-mail account. For the purpose of this article one the biggest and widely known e-mail providers was chosen: Google's Gmail. The author assumes that a reader does not need to have a Gmail's account yet, therefore a quick step-by-step tutorial follows.

In order to create Gmail's account one has to point the web browser to [http://gmail.com](https://www.mql5.com/go?link=http://gmail.com/ "http://gmail.com/") and press ''Create and account'' button:

![](https://c.mql5.com/2/1/1__2.png)

Fig. 1 Create an account button at gmail.com

Google asks for data to fill in: First Name, Last Name, Desired Login Name, password and a security question in case the password was lost.

![](https://c.mql5.com/2/1/2b3.png)

Fig. 2 Create an account form

After CAPTCHA verification and agreeing to Terms of Service, the account is created. One should be immediately able to access the account.

The article will use **mql5signals@gmail.com** as the base e-mail address. The address has been setup for educational purposes only and does not aim to offer any real trading forecasts.

In order to check and verify the account, one can enter credentials and password using the web browser.

There should be a few welcome messages available.

![](https://c.mql5.com/2/1/3a__1.png)

Fig. 3 Welcome messages from gmail.com

After setting up the Gmail's account it is high time to integrate it with MetaTrader 5.

**1.2. Step 2: Configuring E-Mail in MetaTrader 5**

After launching MetaTrader 5 plase make sure that the terminal is properly connected and synchronized. Journal in Toolbox window should contain lines similar to the ones below:

2010.04.26 21:49:38 Network '80360': terminal synchronized with MetaQuotes Software Corp.

2010.04.26 21:49:38 Network '80360': authorized on MetaQuotes-Demo

After making sure the terminal is connected, please press CTRL+O or choose Tools->Options Menu. A popup dialog should appear:

![](https://c.mql5.com/2/1/4__1.png)

Fig. 4 Options dialog

The Email tab contains e-mail settings that need to be inserted in appropriate fields:

![](https://c.mql5.com/2/1/5__1.png)

Fig. 5 Email configuration tab

Enable checkbox tells Metatrader to enable Email settings. The following fields in the dialog box should contain:

1. SMTP server: [smtp.gmail.com](https://www.mql5.com/go?link=http://smtp.gmail.com/ "http://smtp.gmail.com/") – a default SMTP server provided by Google for sending e-mail messages

2. SMTP login: [mql5signals@gmail.com](https://www.mql5.com/go?link=http://gmail.com/ "http://mailto:mql5signals@gmail.com") – please enter your own account login in form of [myforexsignal@gmail.com](https://www.mql5.com/go?link=http://gmail.com/ "http://mailto:myforexsignal@gmail.com"), or any other if you use different E-mail provider

3. SMTP password: this is the password setup earlier for this Gmail's account

4. From: [mql5signals@gmail.com](https://www.mql5.com/go?link=http://gmail.com/ "http://mailto:mql5signals@gmail.com") – this field contains the address of the sender

5. To: [mql5signals@gmail.com](https://www.mql5.com/go?link=http://gmail.com/ "http://mailto:mql5signals@gmail.com") – this field contains the address of the recipient.


In order to try out the settings press ' **Test**' button in Options Dialog. If connection is successful, two things should happen:

In journal there should appear a line similar to:

2010.04.26 23:17:50 MailDispatcher send e-mail complete 'MetaTrader SendMail Test'

and you should receive an email from MetaTrader to Gmail account:

subject: MetaTrader SendMail Test

from: mql5signals@gmail.com

to: mql5signals@gmail.com

date: Mon, Apr 26, 2010 at 11:17 PM

MetaTrader SendMail test message body

**If this works, you will be able to send e-mail messages from MetaTrader 5 via Gmail's SMTP server to** **any e-mail address in the world.**

**This means e.g. trading forecasts or any other information.**

In order to send a message from MQL5 code to a chosen recipient, one could use [SendMail](https://www.mql5.com/en/docs/network/sendmail) function. The simple usage is in the script below. You could insert the [SendMail](https://www.mql5.com/en/docs/network/sendmail) function inside Expert Advisor that would trigger sending an e-mail each time a signal is generated.

This setup can also be used to send signals as new posts to a website, be it a commercial solution or based on blogging platforms. It will be presented later in the article.

```
//+------------------------------------------------------------------+
//|                                                     SendMail.mq5 |
//|                                      Copyright 2010, Investeo.pl |
//|                                                      Investeo.pl |
//+------------------------------------------------------------------+
#property copyright "2010, Investeo"
#property link      "http://Investeo.pl"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   if (SendMail("MQL5Script", "E-MAIL from MQL5")==false)
      Print(GetLastError());
   else Print("E-mail sent.");

  }
//+------------------------------------------------------------------+
```

### 2\. Dedicated websites for publishing live MetaTrader statements

In recent years due to high demand there has developed a new breed of services that offer storing, analyzing, publishing and storing MetaTrader's trading statements. The services are constantly improving and and very good add-on to our trading log, as they allow highly detailed analysis of profits and losses on our accounts.

An example of such a website I found that is already working with MetaTrader 5 is MT Intellingence: [http://mt4i.com](https://www.mql5.com/go?link=https://www.fxblue.com/ "http://mt4i.com/"). Publishing of a report is possible either via MetaTrader's FTP service or using Publisher EA, (during time of writing the article the latter option is available only for MetaTrader 4).

In order to publish own statements one has to create an account on MT Intelligence website and enter appropriate FTP settings into MetaTrader.

The integration process involves entering MT Intelligence FTP server, login, password and account numbers into Options->Publisher dialog:

![](https://c.mql5.com/2/1/fig6__1.png)

Fig. 6 Publisher configuration tab

If connection is setup correctly, MetaTrader will send updated statement via File Transfer Protocol every nth minute to MT Intelligence website.

The statement should be visible after login to [http://mt4i.com](https://www.mql5.com/go?link=https://www.fxblue.com/ "http://mt4i.com/"):

[![](https://c.mql5.com/2/1/mt4i_1c.png)](https://c.mql5.com/2/1/mt4i_1.png "https://c.mql5.com/2/1/mt4i_1.png")

Fig. 7 Trading statement

The complete example of a sample report can be found at [http://www.mt4i.com/users/mql5signals](https://www.mql5.com/go?link=https://www.fxblue.com/users/mql5signals "http://www.mt4i.com/users/mql5signals")

MT Intelligence also provides widgets that can be incorporated into our own website. The widgets contain important information about profits, winning, losing positions and many other highly customized reports.

![](https://c.mql5.com/2/1/ResultChart1.png)

Fig. 8 Daily cumulative net profit for each symbol/direction (generated by MT Intelligence website)

![](https://c.mql5.com/2/1/ResultChart5.png)

Fig. 9 Daily net banked profit/loss  (generated by MT Intelligence website)

![](https://c.mql5.com/2/1/ResultChart2.png)

Fig. 10 Number of winning and losing positions banked per symbol (generated by MT Intelligence website)

![](https://c.mql5.com/2/1/ResultChart3.png)

Fig. 11 Number of closed positions per symbol (generated by MT Intelligence website)

One of the nice thing offered by mt4i is RSS service that contains live account summary and lists all open orders:

![](https://c.mql5.com/2/1/rss.png)

Fig. 12 Open and recent orders for mql5signals (published by MT Intelligence website)

The MT Intelligence site is presented as an example.

Also two more services are worth looking at: [http://mt4live.com](https://www.mql5.com/go?link=http://mt4live.com/ "http://mt4live.com/") and [http://mt4stats.com](https://www.mql5.com/go?link=http://mt4stats.com/ "http://mt4stats.com/"). I strongly believe that both will provide support for MetaTrader 5 or develop their MT5 equivalents http://mt5live.com or http://mt5stats.com.

### 3\. Publishing forecasts on websites using two best blogging platforms: WordPress and Google's Blogger.

Blogging has become a point of interest to many people all over the world in recent years. Nowadays hobbyists do it, companies do it, professionals do it, politicitians including presidents do it, and guess who - yes, traders also do blog. There are numerous pros and cons of blogging, but this is not the point in this article.

The great thing is that we can incorporate automatic trading signals from MT5 straight into our websites. The article will present how to setup the preinstalled platforms: Google's Blogger and Wordpress and integrate it with MetaTrader 5 for trading forecasts.

**3.1. Google's Blogger**

First blogging platform described is Google's Blogger. It is already integrated with Gmail's account, therefore the same e-mail and password as for Gmail are used. In order to sign in one needs to point the web browser to [http://blogger.com](https://www.mql5.com/go?link=http://blogger.com/ "http://blogger.com/") and enter Gmail's  e-mail address and password.

![](https://c.mql5.com/2/1/blogger1a.png)

Fig. 13 Blogger welcome screen

After signing in in order to create a blog, one should click to  'Create your blog now' button:

![](https://c.mql5.com/2/1/blogger2a.png)

Fig. 14 Create Blogger account button

The blog address must be unique, therefore it is necessary to check if the name has not yet been taken:

![](https://c.mql5.com/2/1/blogger3a.png)

Fig. 15 Checking unique blog name at Blogger

If everything was done correctly, the new blog should be created in seconds:

![](https://c.mql5.com/2/1/blogger4a.png)

Fig. 16 Blogger account created message

In order to integrate our blog and MetaTrader 5 we need to setup automatic posts publishing via e-mail.

This will allow MetaTrader 5 to send the automatic trade forecasts and instantly publish them as a new post on the blog. The e-mail address **must be kept secret**, so that nobody else can post unwanted posts apart from the owner of the blog.

The E-mail publishing is done in Settings Tab under **"Email & Mobile"** section:

[![](https://c.mql5.com/2/1/blogger5a_3.png)](https://c.mql5.com/2/1/blogger5a.png "https://c.mql5.com/2/1/blogger5a.png")

Fig. 17 Mail2Blogger configuration

After the address is been set up, it is possible to send forecasts using MQL [SendMail](https://www.mql5.com/en/docs/network/sendmail) function, as presented earlier in the article.

The example blog can be found at [http://mql5signals.blogspot.com](https://www.mql5.com/go?link=http://mql5signals.blogspot.com/ "http://mql5signals.blogspot.com/"):

![](https://c.mql5.com/2/1/blogger6a.png)

Fig. 18 Blog at mql5signals.blogspot.com

**3.2. WordPress**

The alternative blogging platform to Blogger is WordPress. Wordpress also provides a possibility to add automatic posts insertion via e-mail. Apart from WordPress.com hosting it is possible to download and fully customize the platform on one's own web server.

The process of blog creation on [http://WordPress.com](https://www.mql5.com/go?link=https://wordpress.com/ "http://wordpress.com/") is also simple as in Blogger's case and takes only a few minutes.

In order to create a blog, point the web browser to [http://wordpress.com/signup/](https://www.mql5.com/go?link=http://wordpress.com/signup/ "http://wordpress.com/signup/") and enter your credentials.

![](https://c.mql5.com/2/1/wordpress1c2.png)

Fig. 19 Creating an account at Wordpress.com

Similarily to Blogger, blog's address must be unique, therefore the domain for the blog must be confirmed.

![](https://c.mql5.com/2/1/wp_2a.png)

Fig. 20 Checking unique blog name at Wordpress.com

After the account is set up, the activation e-mail is sent and confirmation is requested.

![](https://c.mql5.com/2/1/wp_4a_3.png)

Fig. 21 Confirmation e-mail notice

Upon confirming the activation e-mail account becomes active and we are ready to begin with configuration.

![](https://c.mql5.com/2/1/wp_5a.png)

Fig. 22 Account activation message

_Hint_: Login link is always in form of http://www.yourdomain.com/wp-login.php, or simply http://www.yourdomain.com/login. Since the blog is on WordPress.com domain, you can click on the login link above or point the web browser to address **http://yourdomain.wordpress.com/login**.

The article uses [http://mql5signals.wordpress.com/login](https://www.mql5.com/go?link=http://mql5signals.wordpress.com/login "http://mql5signals.wordpress.com/login") as an example.

![](https://c.mql5.com/2/1/wp_6a.png)

Fig. 23 Login to Wordpress blog

The first thing you should notice after login is a panel located on the left sidebar. As we are interested in MetaTrader 5 integration, please go to Settings->Writing menu.

![](https://c.mql5.com/2/1/wp_1c3.png)

Fig. 24 Wordpress setings panel

As you can see below, similarily to Blogger there is 'Post by Email' functionality incorporated into Wordpress. It can be set up by clicking on My Blogs hyperlink

![](https://c.mql5.com/2/1/wordpress1c3.png)

Fig. 25 Post by email setting

In **My Blogs** menu, after clicking on 'Enable' button and enabling 'Post by Email' functionality, a unique random **secret** e-mail address is generated. This is the address that MetaTrader will send e-mails to.

The Configuration is identical as in Blogger's case.

![](https://c.mql5.com/2/1/wordpress8a.png)![](https://c.mql5.com/2/1/rightarrow.png)

Fig. 26 Before activating Post by E-mail

![](https://c.mql5.com/2/1/wordpress8b.png)

Fig. 27 Post by E-mail activated

Now you should have fully integrated Wordpress blog to MetaTrader 5.

The example [http://mql5signals.wordpress.com](https://www.mql5.com/go?link=https://mql5signals.wordpress.com/ "http://mql5signals.wordpress.com/") is up and running:

![](https://c.mql5.com/2/1/wordpress11.png)

Fig. 28 Blog at mql5signals.wordpress.com

This is it - we setup two blogs that are integrated with MetaTrader 5.

The great thing this is that we can use some plugins or widget provided by other vendors e.g. [http://mt4i.com](https://www.mql5.com/go?link=https://www.fxblue.com/ "http://mt4i.com/") straight into our blog and enhance our website capabilities!

The html codes needed to insert appropriate widgets are available on vendors' websites. Feel free to add them by yourself and play around with the ones that suit you.

### 4\. Integration with Twitter, a microblogging service than many people can follow

![](https://c.mql5.com/2/1/twitter1.png)

Fig. 29 Twitter's logo

The shortest description for Twitter is a short message broadcast service. Any person can be a source of information as well as a destination for it. People following entries of other persons are called 'followers'. Millions of people use it on a daily basis to communicate with a given group of people. For some time it was used only by individuals, but it becomes a news source also for companies.

If you are interested in more details, please visit [http://business.twitter.com/twitter101/](https://www.mql5.com/go?link=http://business.twitter.com/twitter101/ "http://business.twitter.com/twitter101/") for more information on bussiness related usage of tweeter.

In order to signup for tweeter account, please point your browser to [http://twitter.com/signup](https://www.mql5.com/go?link=http://twitter.com/signup "http://twitter.com/signup"). The registration process is very straightforward and similar to those of Wordpress account. For the purpose of this article, a [http://twitter.com/mql5signals](https://www.mql5.com/go?link=http://twitter.com/mql5signals "http://twitter.com/mql5signals") account was created.

Suppose you would like to inform a group of followers on a trading signal. The trading signal would be posted on a Wordpress-based website. The solution is to integrate Wordpress with Twitter, thankfully it can be done in a few steps. Please login to your Wordpress website http://yourdomain.wordpress.com/login, in my case [http://mql5signals.wordpress.com/login.](https://www.mql5.com/go?link=https://mql5signals.wordpress.com/ "http://mql5signals.wordpress.com/")

The same settings page we used for E-mail integration has 'Publicize' options available and one of them is 'Twitter'. Please check this option and press 'Publicize' button.

![](https://c.mql5.com/2/1/twitter2.png)

Fig. 30 Wordpress publicize option

Wordpress will ask for authorization of Twitter account, please confirm it by clicking 'Authorize connection with Twitter'.

![](https://c.mql5.com/2/1/twitter3b__1.png)

Fig. 31 Twitter's account authorization at Wordpress

Then the Twitter message will pop up asking for your Twitter's username or Email and a password:

![](https://c.mql5.com/2/1/twitter4b.png)

Fig. 32 Connecting Wordpress to Twitter

If nothing went wrong Wordpress and Twitter are synchronized. Each time a new post is added (e.g. a forecast from MetaTrader 5 is sent), Twitter gets updated in seconds giving the direct hyperlink to the new post.

![](https://c.mql5.com/2/1/twitter6.png)

Fig. 33 Published Twitter message

This is it , we have fully integrated Twitter!

You may check [http://twitter.com/mql5signals](https://www.mql5.com/go?link=http://twitter.com/mql5signals "http://twitter.com/mql5signals") to see the example Twitter used for this article.

### Conclusion

After reading the article you should be equipped with a knowledge on how to publish automated trading forecasts using MetaTrader and ways to present the forecasts to the end user.

You should be able to setup Gmail's account, integrate it with MetaTrader, you should know where to publish MetaTrader statements, how to configure and integrate Google's Blogger or Wordpress blogging platform with MetaTrader and how to integrate Twitter with Wordpress.

The doors are open now and the rest is totally upon you to decide what kind of service you need and what to publish.

Have fun!

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)
- [MQL5-RPC. Remote Procedure Calls from MQL5: Web Service Access and XML-RPC ATC Analyzer for Fun and Profit](https://www.mql5.com/en/articles/342)
- [Applying The Fisher Transform and Inverse Fisher Transform to Markets Analysis in MetaTrader 5](https://www.mql5.com/en/articles/303)
- [Advanced Adaptive Indicators Theory and Implementation in MQL5](https://www.mql5.com/en/articles/288)
- [Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://www.mql5.com/en/articles/252)
- [Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)
- [Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/931)**
(14)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
18 Feb 2014 at 14:55

How to read from blog and display in terminal ?


![Sergey Golubev](https://c.mql5.com/avatar/2012/12/50D09349-3D90.jpg)

**[Sergey Golubev](https://www.mql5.com/en/users/newdigital)**
\|
6 Jun 2014 at 21:48

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/en/forum)

[Discussion of article "Time Series Forecasting Using Exponential Smoothing"](https://www.mql5.com/en/forum/6057#comment_939746)

[newdigital](https://www.mql5.com/en/users/newdigital "https://www.mql5.com/en/users/newdigital"), 2014.06.06 21:48

Becoming a Fearless Forex Trader

- Must You Know What Will Happen Next?
- Is There a Better Way?
- Strategies When You Know That You Don’t Know

_“Good investing is a peculiar balance between the conviction to follow your ideas and the flexibility to recognize when you have made a mistake.”_

_-Michael Steinhardt_

_"95% of the trading errors you are likely to make will stem from your attitudes about being wrong, losing money, missing out, and leaving money on the table – the four trading fears"_

_-Mark Douglas, Trading In the Zone_

Many traders become enamored with the idea of forecasting. The need for forecasting seems to be inherent to successful trading. After all, you reason, I must know what will happen next in order to make money, right? Thankfully, that’s not right and this article will break down how you can trade well without knowing what will happen next.

![](https://c.mql5.com/3/40/fundamentals1.png)

Must You Know What Will Happen Next?

While knowing what would happen next would be helpful, no one can know for sure. The reason that insider trading is a crime that is often tested in equity markets can help you see that some traders are so desperate to know the future that their willing to cheat and pay a stiff fine when caught. In short, it’s dangerous to think in terms of a certain future when your money is on the line and best to think of edges over certainties when taking a trade.

![](https://c.mql5.com/3/40/fundamentals2.png)

The problem with thinking that you must know what the future holds for your trade, is that when something adverse happens to your trade from your expectations, fear sets in. Fear in and of itself isn’t bad. However, most traders with their money on the line, will often freeze and fail to close out the trade.

If you don’t need to know what will happen next, what do you need? The list is surprisingly short and simple but what’s more important is that you don’t think you know what will happen because if you do, you’ll likely overleverage and downplay the risks which are ever-present in the world of trading.

- A Clean Edge That You’re Comfortable Entering A Trade On
- A Well Defined Invalidation Point Where Your Trade Set-Up No Longer
- A Potential Reversal Entry Point
- An Appropriate Trade Size / Money Management

Is There a Better Way?

Yesterday, the European Central Bank decided to cut their refi rate and deposit rate. Many traders went into this meeting short, yet EURUSD covered ~250% of its daily ATR range and closed near the highs, indicating EURUSD strength. Simply put, the outcome was outside of most trader’s realm of possibility and if you went short and were struck by fear, you likely did not close out that short and were another “victim of the market”, which is another way of saying a victim of your own fears of losing.

![](https://c.mql5.com/3/40/fundamentals3.png)

So what is the better way? Believe it or not, it’s to approach the market, understanding how emotional markets can be and that it is best not to get tied up in the direction the market “has to go”. Many traders will hold on to a losing trade, not to the benefit of their account, but rather to protect their ego. Of course, the better path to trading is to focus on protecting your account equity and leaving your ego at the door of your trading room so that it does not affect your trading negatively.

Strategies When You Know That You Don’t Know

There is one commonality with traders who can trade without fear. They build losing trades into their approach. It’s similar to a gambit in chess and it takes away the edge and strong-hold that fear has on many traders. For those non-chess players, a gambit is a play in which you sacrifice a low-value piece, like a pawn, for the sake of gaining an advantage. In trading, the gambit could be your first trade that allows you to get a better taste of the edge you’re sensing at the moment the trade is entered.

![](https://c.mql5.com/3/40/fundamentals4.png)

James Stanley’s USD Hedge is a great example of a strategy that works under the assumption that one trade will be a loser. What’s the significance of this? It pre-assumes the loss and will allow you to trade without the fear that plagues so many traders. Another tool that you can use to help you define if the trend is staying in your favor or going against you is a fractal.

If you look outside of the world of trading and chess, there are other businesses that presume a loss and therefore are able to act with a clear head when a loss comes. Those businesses are casinos and insurance companies. Both of these businesses presume a loss and work only in line with a calculated risk, they operate free of fear and you can as well if you presume small losses as part of your strategy.

Another great Mark Douglas quote:

_“The less I cared about whether or not I was wrong, the clearer things became, making it much easier to move in and out of positions, cutting my losses short to make myself mentally available to take the next opportunity.” -Mark Douglas_

Happy Trading!

The [source](https://www.mql5.com/go?link=https://www.dailyfx.com/forex/education/trading_tips/daily_trading_lesson/2014/06/06/Fearless-Forex-Trader.html "http://www.dailyfx.com/forex/education/trading_tips/daily_trading_lesson/2014/06/06/Fearless-Forex-Trader.html")

![Palich.3891 Shkrebets](https://c.mql5.com/avatar/2019/12/5DEC00BF-FCDB.jpg)

**[Palich.3891 Shkrebets](https://www.mql5.com/en/users/palich.3891)**
\|
3 May 2020 at 20:42

I threw 25 dollars to MQL5, and now how to find them in my phone, in the MT5 application?


![MyStur](https://c.mql5.com/avatar/avatar_na2.png)

**[MyStur](https://www.mql5.com/en/users/mystur)**
\|
17 Aug 2020 at 09:44

Hi!

need some help on that. There are outhere many informational sites like myfxbook or fxblue or similar, where you can connect your mt4 live account and see all the statistics. I want to build my own peraonal site for that, that i dont use thirdparties.

how this can be made. Has someone some instructions for that or can someone link me where can i find that?

I know this feature from [webterminal](https://www.mql5.com/en/trading "Web terminal for the MetaTrader trading platform") is perfect but i dont need a terminal, only to get the informations from trades , equity, open trades margin - only the statistic... Have someone fromyou an idea how to do that like fxblue or myfxbook has?

thanks for your help

![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
17 Aug 2020 at 10:51

**Don't triple post !!!**

Also don't you see how hard this is? Is like saying there are a few good cars out there, but I don't want to use any of them, I prefer making my own!

![Creating an Indicator with Graphical Control Options](https://c.mql5.com/2/0/macd__1.png)[Creating an Indicator with Graphical Control Options](https://www.mql5.com/en/articles/42)

Those who are familiar with market sentiments, know the MACD indicator (its full name is Moving Average Convergence/Divergence) - the powerful tool for analyzing the price movement, used by traders from the very first moments of appearance of the computer analysis methods. In this article we'll consider possible modifications of MACD and implement them in one indicator with the possibility to graphically switch between the modifications.

![Creating a "Snake" Game in MQL5](https://c.mql5.com/2/0/snake__2.png)[Creating a "Snake" Game in MQL5](https://www.mql5.com/en/articles/65)

This article describes an example of "Snake" game programming. In MQL5, the game programming became possible primarily due to event handling features. The object-oriented programming greatly simplifies this process. In this article, you will learn the event processing features, the examples of use of the Standard MQL5 Library classes and details of periodic function calls.

![A Virtual Order Manager to track orders within the position-centric MetaTrader 5 environment](https://c.mql5.com/2/0/virtual__1.png)[A Virtual Order Manager to track orders within the position-centric MetaTrader 5 environment](https://www.mql5.com/en/articles/88)

This class library can be added to an MetaTrader 5 Expert Advisor to enable it to be written with an order-centric approach broadly similar to MetaTrader 4, in comparison to the position-based approach of MetaTrader 5. It does this by keeping track of virtual orders at the MetaTrader 5 client terminal, while maintaining a protective broker stop for each position for disaster protection.

![Creating Tick Indicators in MQL5](https://c.mql5.com/2/0/Untitled7.png)[Creating Tick Indicators in MQL5](https://www.mql5.com/en/articles/60)

In this article, we will consider the creation of two indicators: the tick indicator, which plots the tick chart of the price and tick candle indicator, which plot candles with the specified number of ticks. Each of the indicators writes the incoming prices into a file, and uses the saved data after the restart of the indicator (these data also can be used by the other programs)

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/80&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071784514914561632)

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