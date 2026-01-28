---
title: Using MetaTrader 4 for a Time Based Pattern Analysis
url: https://www.mql5.com/en/articles/1508
categories: Trading Systems
relevance_score: 4
scraped_at: 2026-01-23T17:47:56.735335
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/1508&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068662769475058715)

MetaTrader 4 / Trading systems


### Introduction

Reading posts at the Automated Trading Championship forum and interviews leads always to a lot of interesting information buried in a lot of noise.

The arguments provided by William Boatright (Wackena) in his interview, [https://championship.mql5.com](https://championship.mql5.com/) aroused my attention for the time based approach used to chose one single hour of the day to take one single trade in a daily swing trading technique.

So I started gathering information about the time based entry approach and decided to realize a system that should be able to verify the real effectiveness of this technique.

A code attached to the article is realized without a real exit strategy just to give you an example of what kind of time patterns you could expect using MetaTrader for data mining and statistical investigation over a relatively long data sequence.

### Searching Literature

First of all I looked for the confirmation of this idea in literature and I found a very interesting article about this subject in "New Trading Systems and Methods" by Perry J. Kaufman, a really heavy bible in technical analysis adventure. Chapter 15 tells about pattern recognition, and one of the very first arguments is the time of day and trading habits.

In this chapter he mentions Frank Tubbs' Stock Market Correspondence Lessons where he explained the six dominant patterns in the stock market based on US trading hours, and where in rule 4 he asserts: "if the market has been bullish until 2:00 p.m. it will probably continue until the close and into the next day".

Well 2:00 p.m. GMT-5, is exactly 20:00 GMT+1, the time in which Wackena puts his trades in the contest. This is the first interesting confirmation about the effectiveness of this technique. Other interesting references about the subject are presented in this Kaufman's chapter.

### Implementing the Basic EA

The first consideration about implementing a system that should catch a trade direction at a specific time of the day, is that the only relevant signals you can look for are those that give you information about a trend direction, and that countertrend methods or breakout systems are not suitable for this purpose.

A basic Expert Advisor was presented in this article and a block diagram representing the operation flow is reported here.

![](https://c.mql5.com/2/15/flowchartccopia.gif)

Where:

- _Analyzer_ routine calls a sequence of signal detectors and gives response about the trend strength at that moment.

- _TrailingStopEngine_ dynamically evaluates next profit target and new trailing stop depending on average true range or something like that.

- _CurrentOpenOrders_ returns the number of opened orders.

- A _LoopThroughOrders_ loops through all orders and if necessary applies new trailing stops and new profit targets, or decides to close out the trade for some particular event.

- _BlockFilterTrading_ decides if there are specific conditions, in which we do not want to trade at all.

- _MoneyManagement_ returns the lot size as a function of risk.

- _PlaceOrder_, if possible, places orders in the direction defined by the Analyzer.

### Operation and Optimization Results

I use MetaTrader engine on Apple MacBookPro using a virtual pc running under Parallel Desktop where it runs fast and reliably and I can take fast snapshots of the MS-windows virtual machine very easily for documentation.

Backtesting has been done over the EURUSD currency pair over available data from January, 1 2007 up to December, 29 2007 in a 15 minutes time frame where results seem to be satisfactory.

Main operation parameters used in this article are taken from a first optimization process and you can try yourself different combination of such parameters.

The only consideration about choosing Take Profit and Stop Loss parameters used for this test, is that here we are not really interested in maximizing the balance nor any other possible optimization parameter presented by MetaTrader, but indeed we only need to maximize the number of profit trades to stress the entry strategy.

Any optimization of other results should be done in a later phase.

Here is the code of the Analyzer module where, for testing purpose, you can add any other signal detector.

Two different signals must agree to chose the right direction.

```
//+------------------------------------------------------------------+
//| Price Direction Analyzer
//+------------------------------------------------------------------+
int Analyzer()
{
 int  signalCount=0;
 signalCount += EntrySignal1();
 signalCount += EntrySignal2();
 return(signalCount);
}

//+------------------------------------------------------------------+
//| ENTRY SIGNALS BLOCK MODULES
//+------------------------------------------------------------------+
int EntrySignal1()
{ // Long term SMA trend detect
 int i,Signal;

 int LongTrend=0;
 for(i=0;i<3;i++)
 {
   if (iMA(Symbol(),PERIOD_H4,S1_MA_FAST,0,MODE_LWMA,PRICE_TYPICAL,i) > iMA(Symbol(),PERIOD_H4,S1_MA_FAST,0,
   MODE_LWMA,PRICE_TYPICAL,i+1))
     LongTrend++;
   else
     LongTrend--;
 }
 if( LongTrend < 0)
   Signal=-1;
 else
   Signal=1;
 return(Signal);
}

int EntrySignal2()
{ // Daily MACD
   int Signal;

   if (iMACD(NULL,PERIOD_D1,S2_OSMAFast,S2_OSMASlow,S2_OSMASignal,PRICE_WEIGHTED,MODE_MAIN,0) >
       iMACD(NULL,PERIOD_D1,S2_OSMAFast,S2_OSMASlow,S2_OSMASignal,PRICE_WEIGHTED,MODE_MAIN,1) )
     Signal=1;
   else
     Signal=-1;
   return (Signal);
}
```

Trading hour is matched in a block trading filter module that can be simply realized as follows.

Modular architecture described can easily leave space for new blocking filters in the operation flow.

```
//+------------------------------------------------------------------+
//| FILTER BLOCK MODULES
//+------------------------------------------------------------------+
bool BlockTradingFilter1()
{
 bool BlockTrade=false;  //trade by default
 if (UseHourTrade)
 {
   if( !(Hour() >= FromHourTrade && Hour() <= ToHourTrade && Minute()<= 3) )
     {
      //  Comment("Non-Trading Hours!");
      BlockTrade=true;
     }
  }
 return (BlockTrade);
}
```

As a matter of fact we should be able to have a lot of small profit trades without considering the whole balance at all.

Here is the main optimization setting:

![](https://c.mql5.com/2/15/parallelsipicturef0.gif)

To analyze time intervals that give best results you must set the FromHourTrades from 0 to 23 running at step of 1 hour and check Optimization flag in the backtesting setting form before starting the process.

![](https://c.mql5.com/2/15/parallelstpicturel1.gif)

Here are the optimization results:

![](https://c.mql5.com/2/15/tradinghourstcopia.gif)

As you can see there are hours in which trading using trend detection techniques should be very dangerous, while during other day hours choosing trend direction should be considerably more profitable, and this time interval, between 19:00 and 22:00 GMT+1, the afternoon in Eastern Time zone, is when every news has been digested by the market.

In this case, with available historic data, the peak hour obtained from optimization process corresponds to 21:00 GMT+1 (Central Europe Time).

Sure this results should be considered valid for the 2007 forex market, but Frank Tubbs old consideration about trading habits let us hope they should be valid in larger time periods.

And here are the detailed results from optimization process:

![](https://c.mql5.com/2/15/parallelsxpicturex5.gif)

You must consider that this results depend also on trending signals chosen to decide on the direction of orders, and I can tell you that in this time-based strategy, choosing from a large number of different signals leads to quite similar results, but surely you can test other signals and maybe let me know your results.

And here are backtesting reports at 21:00 GMT+1

![](https://c.mql5.com/2/15/parallelsnpicturen6.gif)

As you can see there are 160 total trades and 154 profit trades (96.86%).

This information confirms the one made by Wackena in his interview.

### Introducing Time-Based Stop Loss

The number of trades during one year backtesting process may suggest that we should be able to improve trading results by introducing a time based Stop Loss, that should leave room for new orders, if there are any losses after 23 hours being in the market.

To do this you can simply set the UseTimeBasedStopLoss flag to true and try different optimization parameters.

The following report table shows the consequence of this strategy change.

![](https://c.mql5.com/2/15/parallelscpicturet7.gif)

As you can see this exit strategy lets you guess that some long swing trade, in a weak market, could protect you from adding undesirable orders in bad moments, so it should be better to be patient and wait till the market exits dangerous sideways.

### Conclusion

The strategy of analyzing time based pattern recognition on trading habits suggests further more investigation and justifies adding a valid money management strategy to this Expert Advisor and most of all a valid trailing stop engine, but this should be the argument of a new article.

The code attached can also be used to make further investigation over other patterns and currency pairs to analyze other timely based trading habits and behavior. Just let me know your experience in this investigation.

### References

_New Trading Systems and Methods_, by Perry J. Kaufman

[https://championship.mql5.com](https://championship.mql5.com/)/p>

_The Encyclopedia of Trading Strategies_, by Jeffrey Owen Katz,Donna L. McCormick/p>

_Forex conquered_, by J.L.Person/p>

_Trading with the odds,_ by Cynthia A.Kase

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1508.zip "Download all attachments in the single ZIP archive")

[Sauron\_1\_3.mq4](https://www.mql5.com/en/articles/download/1508/Sauron_1_3.mq4 "Download Sauron_1_3.mq4")(7.15 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39403)**
(6)


![Giampiero Raschetti](https://c.mql5.com/avatar/2010/12/4D03B0F6-464A.jpg)

**[Giampiero Raschetti](https://www.mql5.com/en/users/giaras)**
\|
5 Feb 2008 at 17:33

**Erik.VH wrote:**

Hi Giampiero,

> Can't find the code ?
>
> Great article.

Regards

Erik

Just a misunderstanding in publication work flow.

It will be attached as soon as possible.

thanks and best regards

Giampiero


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
6 Feb 2008 at 07:20

**giaras:**


Just a misunderstanding in publication work flow.

It will be attached as soon as possible.

thanks and best regards

Giampiero

Done.


![Mariusz Woloszyn](https://c.mql5.com/avatar/2010/10/4CB37560-9ECA.jpg)

**[Mariusz Woloszyn](https://www.mql5.com/en/users/emsi)**
\|
8 Feb 2008 at 13:39

There is no file with EA attached to this article again.


![Giampiero Raschetti](https://c.mql5.com/avatar/2010/12/4D03B0F6-464A.jpg)

**[Giampiero Raschetti](https://www.mql5.com/en/users/giaras)**
\|
15 Feb 2008 at 12:13

The actual Time Based [Stop Loss](https://www.metatrader5.com/en/terminal/help/trading/general_concept "User Guide: Types of orders") is incorrect and it does not work for more than 23 hours.

This code should correct the problem considering closed trading hours too.

It should be placed in BUY and SELL position.

```
int k=0;
while(k < TradeHoldingPeriod)
{
  if(iTime(NULL,PERIOD_H1,k) > OrderOpenTime())
    k++;
  else break;
}
if(UseTimeBasedStopLoss && k >= TradeHoldingPeriod && OrderProfit() < 0 )
{
  if(SignalCount > 0)
    OrderClose(OrderTicket(),OrderLots(),Ask,3,Violet);
  return(0);
}
```

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
9 Feb 2009 at 02:08

[Giampiero Raschetti](https://www.mql5.com/ru/users/giaras) your timezone idea is actually works ! I also in my practice looked such good zones for opening of trade positions ! Please contact me for discussion ... very very -))


![An Expert Advisor Made to Order. Manual for a Trader](https://c.mql5.com/2/117/robot__2.png)[An Expert Advisor Made to Order. Manual for a Trader](https://www.mql5.com/en/articles/1460)

Not all traders are programmers. And not all of the programmers are really good ones. So, what should be done, if you need to automate your system by do not have time and desire to study MQL4?

![Equivolume Charting Revisited](https://c.mql5.com/2/15/537_33.gif)[Equivolume Charting Revisited](https://www.mql5.com/en/articles/1504)

The article dwells on the method of constructing charts, at which each bar consists of the equal number of ticks.

![Easy Way to Publish a Video at MQL4.Community](https://c.mql5.com/2/15/582_26.jpg)[Easy Way to Publish a Video at MQL4.Community](https://www.mql5.com/en/articles/1520)

It is usually easier to show, than to explain. We offer a simple and free way to create a video clip using CamStudio for publishing it in MQL.community forums.

![Betting Modeling as Means of Developing "Market Intuition"](https://c.mql5.com/2/15/538_7.gif)[Betting Modeling as Means of Developing "Market Intuition"](https://www.mql5.com/en/articles/1505)

The article dwells on the notion of "market intuition" and ways of developing it. The method described in the article is based on the modeling of financial betting in the form of a simple game.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hjszkhbsppmkbvkxfmwtkbjsqghbtmfv&ssn=1769179675759192205&ssn_dr=0&ssn_sr=0&fv_date=1769179675&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1508&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Using%20MetaTrader%204%20for%20a%20Time%20Based%20Pattern%20Analysis%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176917967583358182&fz_uniq=5068662769475058715&sv=2552)

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